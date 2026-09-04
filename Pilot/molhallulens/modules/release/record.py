"""Build and validate released H/N detector records."""

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
from molhallulens.modules.text_realization import (
    MatchedRenderedPair,
    PairAlignment,
    StepPairAlignment,
)


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
    """Assemble paired detector records with exact serialized offsets."""

    def _build(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
        annotated: AnnotatedHallucination,
        *,
        variant_label: str,
        pair_id: str,
        matched_record_id: str | None,
        step_pair_alignment: tuple[StepPairAlignment, ...],
        paired_lengths: Mapping[str, int],
    ) -> ReleasedHallucinationRecord:
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
            global_context_start = component_start + span.context_start
            global_context_end = component_start + span.context_end
            if serialized_text[global_start:global_end] != span.text:
                raise ValueError("serialized hallucination span does not round-trip")
            item = span.to_dict()
            item["serialized_span"] = [global_start, global_end]
            item["serialized_context_span"] = [
                global_context_start,
                global_context_end,
            ]
            item["pair_occurrence_id"] = span.mention_id
            item["same_char_length"] = (
                len(span.text) == paired_lengths[span.mention_id]
            )
            serialized_spans.append(item)

        serialized_controls = []
        for span in annotated.control_spans:
            component_start = (
                reasoning_start
                if span.component == "reasoning_chain"
                else final_answer_start
            )
            global_start = component_start + span.start
            global_end = component_start + span.end
            global_context_start = component_start + span.context_start
            global_context_end = component_start + span.context_end
            if serialized_text[global_start:global_end] != span.text:
                raise ValueError("serialized control span does not round-trip")
            item = span.to_dict()
            item["serialized_span"] = [global_start, global_end]
            item["serialized_context_span"] = [
                global_context_start,
                global_context_end,
            ]
            serialized_controls.append(item)

        present = annotated.hallucination_present
        if variant_label not in {"H", "N"}:
            raise ValueError("variant_label must be H or N")
        if present != (variant_label == "H"):
            raise ValueError("variant_label disagrees with annotation polarity")
        plan = injected.plan
        record_id = f"{pair_id}__{variant_label}"
        data = {
            "record_id": record_id,
            "origin_id": plan.origin_id,
            "pair_id": pair_id,
            "matched_record_id": matched_record_id,
            "variant_label": variant_label,
            "subtask": artifact.normalized_subtask.value,
            "variant_index": plan.variant_index,
            "derived_seed": plan.derived_seed,
            "edit_count": len(plan.mutations),
            "mutation_events": [item.to_dict() for item in plan.mutations],
            "propagation_events": [
                item.to_dict() for item in injected.propagation_events
            ],
            "edge_audit": [item.to_dict() for item in injected.edge_audit],
            "violated_edge_ids": list(injected.violated_edge_ids) if present else [],
            "detector_input": {
                "indexed_smiles": source,
                "instruction": instruction,
                "reasoning_chain": rendered.reasoning_chain,
                "final_answer": rendered.final_answer,
            },
            "step_texts": list(rendered.step_texts),
            "pair_alignment": [item.to_dict() for item in step_pair_alignment],
            "text_realization": _plain(rendered.realization),
            "hallucination_spans": serialized_spans,
            "control_spans": serialized_controls,
            "serialized": {
                "text": serialized_text,
                "sha256": hashlib.sha256(serialized_text.encode("utf-8")).hexdigest(),
            },
            "labels": {
                "hallucination_present": present,
                "hallucinated_semantic_points": len(plan.mutations) if present else 0,
                "root_changed_nodes": len(plan.edited_node_ids) if present else 0,
                "propagated_changed_nodes": (
                    len(injected.propagation_events) if present else 0
                ),
                "violated_edges": len(injected.violated_edge_ids) if present else 0,
                "hallucinated_text_spans": len(serialized_spans),
                "control_text_spans": len(serialized_controls),
            },
        }
        return ReleasedHallucinationRecord(data=data)

    def build(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
        annotated: AnnotatedHallucination,
    ) -> ReleasedHallucinationRecord:
        """Build a standalone H record; dataset generation should use build_pair."""

        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("artifact must be ReferenceDAGArtifact")
        if type(injected) is not InjectedHallucination:
            raise TypeError("injected must be InjectedHallucination")
        if type(annotated) is not AnnotatedHallucination:
            raise TypeError("annotated must be AnnotatedHallucination")
        if not annotated.hallucination_present:
            raise ValueError("standalone build only accepts a positive annotation")
        return self._build(
            artifact,
            injected,
            annotated,
            variant_label="H",
            pair_id=injected.plan.plan_id,
            matched_record_id=None,
            step_pair_alignment=(),
            paired_lengths={span.mention_id: len(span.text) for span in annotated.spans},
        )

    def build_pair(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
        rendered_pair: MatchedRenderedPair,
        positive: AnnotatedHallucination,
        negative: AnnotatedHallucination,
    ) -> tuple[ReleasedHallucinationRecord, ReleasedHallucinationRecord]:
        """Build H then N and enforce their span/control one-to-one mapping."""

        if type(rendered_pair) is not MatchedRenderedPair:
            raise TypeError("rendered_pair must be MatchedRenderedPair")
        if positive.rendered is not rendered_pair.hallucinated:
            raise ValueError("positive annotation does not belong to rendered pair")
        if negative.rendered is not rendered_pair.negative:
            raise ValueError("negative annotation does not belong to rendered pair")
        if not positive.hallucination_present or negative.hallucination_present:
            raise ValueError("build_pair requires one positive and one negative annotation")
        controls = {
            item.pair_occurrence_id: item for item in negative.control_spans
        }
        if len(controls) != len(negative.control_spans):
            raise ValueError("negative control pair IDs must be unique")
        positive_ids = {item.mention_id for item in positive.spans}
        if set(controls) != positive_ids:
            raise ValueError("H spans and N controls are not one-to-one")

        pair_id = injected.plan.plan_id
        h_record_id = f"{pair_id}__H"
        n_record_id = f"{pair_id}__N"
        paired_lengths = {
            occurrence_id: len(control.text)
            for occurrence_id, control in controls.items()
        }
        alignments = rendered_pair.step_pair_alignment
        h_record = self._build(
            artifact,
            injected,
            positive,
            variant_label="H",
            pair_id=pair_id,
            matched_record_id=n_record_id,
            step_pair_alignment=alignments,
            paired_lengths=paired_lengths,
        )
        n_record = self._build(
            artifact,
            injected,
            negative,
            variant_label="N",
            pair_id=pair_id,
            matched_record_id=h_record_id,
            step_pair_alignment=alignments,
            paired_lengths={
                item.pair_occurrence_id: len(item.text)
                for item in negative.control_spans
            },
        )

        if all(
            item.pair_alignment is PairAlignment.BYTE_IDENTICAL
            for item in alignments
        ):
            h_text = h_record.data["serialized"]["text"]
            pieces: list[str] = []
            cursor = 0
            for span in sorted(
                h_record.data["hallucination_spans"],
                key=lambda item: item["serialized_span"],
            ):
                start, end = span["serialized_span"]
                control = controls[span["pair_occurrence_id"]]
                pieces.extend((h_text[cursor:start], control.text))
                cursor = end
            pieces.append(h_text[cursor:])
            if "".join(pieces) != n_record.data["serialized"]["text"]:
                raise ValueError("full serialized matched-pair invariant failed")
        return h_record, n_record


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
