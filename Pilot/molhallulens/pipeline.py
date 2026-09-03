"""Explicit A-to-H dataset pipeline with visible inputs and outputs."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class IngestionModule(Protocol):
    """Input: dataset source. Output: normalized ChemCoTBench records."""

    def run(self, dataset_source: Any) -> Any: ...


class ReferenceModule(Protocol):
    """Input: normalized records. Output: validated reference DAGs and truth."""

    def run(self, ingested_records: Any) -> Any: ...


class ErrorPlanningModule(Protocol):
    """Input: reference states. Output: selected error plans."""

    def run(self, reference_states: Any) -> Any: ...


class ErrorInjectionModule(Protocol):
    """Input: selected error plans. Output: root-mutated states."""

    def run(self, error_plans: Any) -> Any: ...


class TrajectoryModule(Protocol):
    """Input: root mutations. Output: propagated DAGs and graph deltas."""

    def run(self, root_mutations: Any) -> Any: ...


class TextRealizationModule(Protocol):
    """Input: propagated DAGs. Output: rendered step text and mentions."""

    def run(self, trajectories: Any) -> Any: ...


class AnnotationModule(Protocol):
    """Input: rendered text and mentions. Output: character/token labels."""

    def run(self, rendered_text: Any) -> Any: ...


class ReleaseModule(Protocol):
    """Input: annotated records. Output: released dataset artifacts."""

    def run(self, annotated_records: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SequentialPipeline:
    """Run the eight named modules in one visible, fixed sequence."""

    ingestion: IngestionModule
    reference: ReferenceModule
    error_planning: ErrorPlanningModule
    error_injection: ErrorInjectionModule
    trajectory: TrajectoryModule
    text_realization: TextRealizationModule
    annotation: AnnotationModule
    release: ReleaseModule

    def run(self, dataset_source: Any) -> Any:
        """Transform one dataset source into released dataset artifacts."""

        ingested_records = self.ingestion.run(dataset_source)
        reference_states = self.reference.run(ingested_records)
        error_plans = self.error_planning.run(reference_states)
        root_mutations = self.error_injection.run(error_plans)
        trajectories = self.trajectory.run(root_mutations)
        rendered_text = self.text_realization.run(trajectories)
        annotated_records = self.annotation.run(rendered_text)
        released_dataset = self.release.run(annotated_records)
        return released_dataset


def _plain_data(value: Any) -> Any:
    """Convert immutable domain objects into complete JSON-printable data."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_data(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return value


def _print_full_stage(stage: str, input_value: Any, output_value: Any) -> None:
    """Print an untruncated module input and output as indented JSON."""

    print("\n" + "=" * 100)
    print(stage)
    print("-" * 100)
    print("FULL INPUT")
    print(json.dumps(_plain_data(input_value), ensure_ascii=False, indent=2))
    print("\nFULL OUTPUT")
    print(json.dumps(_plain_data(output_value), ensure_ascii=False, indent=2))


def _read_jsonl_record(path: Path, record_id: str) -> dict[str, Any]:
    """Read one complete stored artifact by record_id."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_id") == record_id:
                return record
    raise RuntimeError(f"record {record_id!r} was not found in {path}")


__all__ = [
    "AnnotationModule",
    "ErrorInjectionModule",
    "ErrorPlanningModule",
    "IngestionModule",
    "ReferenceModule",
    "ReleaseModule",
    "SequentialPipeline",
    "TextRealizationModule",
    "TrajectoryModule",
]


if __name__ == "__main__":
    # Make this file runnable both ways:
    #   1. From Pilot:       python -m molhallulens.pipeline
    #   2. From molhallulens: python pipeline.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from molhallulens.modules.ingestion import (
        ChemCoTMolEditAdapter,
        JoinedInputRecord,
    )
    from molhallulens.modules.reference import build_reference_dag

    origin_id = "mol_edit.add_v2.0003"
    record_id = "mol_edit.add_v2.0003__LOCAL__H"
    dataset_root = project_root / "Dataset"
    generated_root = project_root / "HallucinationDataset"

    # C-H replay the project's complete, already-generated artifacts for one
    # real LOCAL-H record.  A and B are executed live below.
    stored_provenance = _read_jsonl_record(
        generated_root / "provenance" / "train.jsonl",
        record_id,
    )
    stored_state_graph = _read_jsonl_record(
        generated_root / "state_graphs" / "train.jsonl",
        record_id,
    )
    stored_release_record = _read_jsonl_record(
        generated_root / "records" / "train.jsonl",
        record_id,
    )

    class DemoIngestion:
        def run(self, dataset_source: Any) -> Any:
            ingestion_input = {
                "dataset_root": str(dataset_source),
                "target_origin_id": origin_id,
            }
            all_records = ChemCoTMolEditAdapter().load(Path(dataset_source))
            selected_record = next(
                record
                for record in all_records
                if record.anonymous_sample_id == origin_id
            )
            ingested_records = _plain_data(selected_record)
            _print_full_stage(
                "A INGESTION — load, validate, and join raw/process/template data",
                ingestion_input,
                ingested_records,
            )
            return ingested_records

    class DemoReference:
        def run(self, ingested_records: Any) -> Any:
            joined_record = JoinedInputRecord(
                anonymous_sample_id=ingested_records["anonymous_sample_id"],
                raw_record=ingested_records["raw_record"],
                process_record=ingested_records["process_record"],
                formal_template=ingested_records["formal_template"],
            )
            reference_states = _plain_data(build_reference_dag(joined_record))
            _print_full_stage(
                "B REFERENCE — build the complete typed reference DAG",
                ingested_records,
                reference_states,
            )
            return reference_states

    class DemoErrorPlanning:
        def run(self, reference_states: Any) -> Any:
            error_plans = {
                "reference_artifact": reference_states,
                "record_id": stored_provenance["record_id"],
                "origin_id": stored_provenance["origin_id"],
                "pair_id": stored_provenance["pair_id"],
                "bundle_id": stored_provenance["bundle_id"],
                "recipe": stored_provenance["recipe"],
                "candidate_selection": stored_provenance["candidate_selection"],
                "donor": stored_provenance["donor"],
                "fallback": stored_provenance["fallback"],
                "execution_mode": stored_provenance["execution_mode"],
            }
            _print_full_stage(
                "C ERROR PLANNING — choose target, operator, candidate, and seed",
                reference_states,
                error_plans,
            )
            return error_plans

    class DemoErrorInjection:
        def run(self, error_plans: Any) -> Any:
            root_mutations = {
                "record_id": record_id,
                "error_plan": error_plans,
                "mutation_descriptor": stored_release_record["mutation"],
                "mutation_events": stored_state_graph["mutation_events"],
                "reference_graph": stored_state_graph["reference"],
                # For LOCAL, selected_nodes contains only the root, so the
                # stored locked graph is also the root-mutated graph.
                "root_mutated_graph": stored_state_graph["locked"],
            }
            _print_full_stage(
                "D ERROR INJECTION — apply product_heavy: 43 -> 37",
                error_plans,
                root_mutations,
            )
            return root_mutations

    class DemoTrajectory:
        def run(self, root_mutations: Any) -> Any:
            trajectories = {
                "record_id": record_id,
                "policy": stored_provenance["recipe"]["policy"],
                "root_mutation": stored_state_graph["mutation_events"],
                "propagation": stored_provenance["propagation"],
                "semantic_difference_targets": stored_state_graph[
                    "semantic_difference_targets"
                ],
                "candidate_graph_after_trajectory": stored_state_graph["locked"],
                "formal_trace_after_trajectory": stored_state_graph["formal_trace"],
            }
            _print_full_stage(
                "E TRAJECTORY — apply LOCAL propagation and record graph differences",
                root_mutations,
                trajectories,
            )
            return trajectories

    class DemoTextRealization:
        def run(self, trajectories: Any) -> Any:
            rendered_text = {
                "record_id": record_id,
                "trajectory": trajectories,
                "renderer": stored_provenance["renderer"],
                "detector_input": stored_release_record["detector_input"],
                "serialized": stored_release_record["serialized"],
            }
            _print_full_stage(
                "F TEXT REALIZATION — render every step and the complete detector prompt",
                trajectories,
                rendered_text,
            )
            return rendered_text

    class DemoAnnotation:
        def run(self, rendered_text: Any) -> Any:
            annotated_records = {
                "record_id": record_id,
                "rendered_text": rendered_text,
                "spans": stored_release_record["spans"],
                "trace_labels": stored_release_record["trace_labels"],
                "verification": stored_release_record["verification"],
                "tokenizer": stored_provenance["tokenizer"],
            }
            _print_full_stage(
                "G ANNOTATION — attach hallucination spans, labels, and QA results",
                rendered_text,
                annotated_records,
            )
            return annotated_records

    class DemoRelease:
        def run(self, annotated_records: Any) -> Any:
            released_dataset = stored_release_record
            _print_full_stage(
                "H RELEASE — emit the exact complete JSONL record",
                annotated_records,
                released_dataset,
            )
            return released_dataset

    print("MolHalluLens detailed A-to-H pipeline demo")
    print(f"Origin: {origin_id}")
    print(f"Hallucinated record: {record_id}")
    print("A-B run live; C-H replay the complete stored real artifact.")
    print("Nothing below is abbreviated or truncated.")

    # A. Dataset path -> one complete joined ChemCoTBench-V2 record.
    ingestion_module = DemoIngestion()
    ingested_records = ingestion_module.run(dataset_root)
    input("\nPress Enter to continue to B REFERENCE...")

    # B. Joined record -> complete reference DAG.
    reference_module = DemoReference()
    reference_states = reference_module.run(ingested_records)
    input("\nPress Enter to continue to C ERROR PLANNING...")

    # C. Reference DAG -> one complete, reproducible error plan.
    error_planning_module = DemoErrorPlanning()
    error_plans = error_planning_module.run(reference_states)
    input("\nPress Enter to continue to D ERROR INJECTION...")

    # D. Error plan -> graph containing the root mutation.
    error_injection_module = DemoErrorInjection()
    root_mutations = error_injection_module.run(error_plans)
    input("\nPress Enter to continue to E TRAJECTORY...")

    # E. Root mutation -> graph after applying the propagation policy.
    trajectory_module = DemoTrajectory()
    trajectories = trajectory_module.run(root_mutations)
    input("\nPress Enter to continue to F TEXT REALIZATION...")

    # F. Candidate graph -> complete textual reasoning and detector prompt.
    text_realization_module = DemoTextRealization()
    rendered_text = text_realization_module.run(trajectories)
    input("\nPress Enter to continue to G ANNOTATION...")

    # G. Rendered text -> hallucination spans and validation labels.
    annotation_module = DemoAnnotation()
    annotated_records = annotation_module.run(rendered_text)
    input("\nPress Enter to continue to H RELEASE...")

    # H. Annotated sample -> exact released JSONL record.
    release_module = DemoRelease()
    final_output = release_module.run(annotated_records)

    print("\n" + "=" * 100)
    print("PIPELINE FINISHED")
    print(f"record_id: {final_output['record_id']}")
    print(
        "root mutation: "
        f"{final_output['mutation']['root_state_id']} "
        "43 -> "
        f"{stored_state_graph['mutation_events'][0]['after']}"
    )
    print(f"hallucination spans: {len(final_output['spans'])}")
