"""Build one always-hallucinated detector record."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molhallulens.core import InjectedHallucination
from molhallulens.modules.annotation import AnnotatedHallucination
from molhallulens.modules.reference import ReferenceDAGArtifact


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ReleasedHallucinationRecord:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data


class UnifiedRecordBuilder:
    """Assemble one detector record with direct mutation and span provenance."""

    def build(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
        annotated: AnnotatedHallucination,
    ) -> ReleasedHallucinationRecord:
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("artifact must be ReferenceDAGArtifact")
        if type(injected) is not InjectedHallucination:
            raise TypeError("injected must be InjectedHallucination")
        if type(annotated) is not AnnotatedHallucination:
            raise TypeError("annotated must be AnnotatedHallucination")

        source = artifact.state_dag.values["source"].normalized_value
        instruction = artifact.state_dag.values["instruction"].normalized_value
        rendered = annotated.rendered
        serialized_text = (
            f"<MOLECULE>\n{source}\n\n"
            f"<INSTRUCTION>\n{instruction}\n\n"
            f"<REASONING>\n{rendered.reasoning_chain}\n\n"
            f"<FINAL_ANSWER>\n{rendered.final_answer}"
        )
        reasoning_start = serialized_text.index(rendered.reasoning_chain)
        final_answer_start = serialized_text.rindex(rendered.final_answer)
        serialized_spans = []
        for span in annotated.spans:
            component_start = (
                reasoning_start
                if span.component == "reasoning_chain"
                else final_answer_start
            )
            global_start = component_start + span.start
            global_end = component_start + span.end
            if serialized_text[global_start:global_end] != span.text:
                raise ValueError("serialized hallucination span does not round-trip")
            item = span.to_dict()
            item["serialized_span"] = [global_start, global_end]
            serialized_spans.append(item)

        plan = injected.plan
        data = {
            "record_id": plan.plan_id,
            "origin_id": plan.origin_id,
            "subtask": artifact.normalized_subtask.value,
            "variant_index": plan.variant_index,
            "derived_seed": plan.derived_seed,
            "edit_count": len(plan.mutations),
            "mutation_events": [item.to_dict() for item in plan.mutations],
            "propagation_events": [
                item.to_dict() for item in injected.propagation_events
            ],
            "edge_audit": [item.to_dict() for item in injected.edge_audit],
            "violated_edge_ids": list(injected.violated_edge_ids),
            "detector_input": {
                "indexed_smiles": source,
                "instruction": instruction,
                "reasoning_chain": rendered.reasoning_chain,
                "final_answer": rendered.final_answer,
            },
            "step_texts": list(rendered.step_texts),
            "text_realization": _plain(rendered.realization),
            "hallucination_spans": serialized_spans,
            "serialized": {
                "text": serialized_text,
                "sha256": hashlib.sha256(serialized_text.encode("utf-8")).hexdigest(),
            },
            "labels": {
                "hallucination_present": True,
                "hallucinated_semantic_points": len(plan.mutations),
                "root_changed_nodes": len(plan.edited_node_ids),
                "propagated_changed_nodes": len(injected.propagation_events),
                "violated_edges": len(injected.violated_edge_ids),
                "hallucinated_text_spans": len(serialized_spans),
            },
        }
        return ReleasedHallucinationRecord(data=data)


def write_jsonl(
    records: Iterable[ReleasedHallucinationRecord],
    output_path: Path,
) -> None:
    """Write new-schema records only; callers explicitly choose the destination."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if type(record) is not ReleasedHallucinationRecord:
                raise TypeError("records must contain ReleasedHallucinationRecord values")
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


__all__ = ["ReleasedHallucinationRecord", "UnifiedRecordBuilder", "write_jsonl"]
