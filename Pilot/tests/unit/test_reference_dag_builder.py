"""Unit tests for typed, fail-closed molecule-editing reference DAGs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import (
    AdditionReferenceDAGBuilder,
    DeletionReferenceDAGBuilder,
    ReferenceDAGBuildError,
    SubstitutionReferenceDAGBuilder,
    audit_reference_dag_corpus,
    build_reference_dag,
    reference_dag_builder_for,
)
from molhallulens.core import (
    ClaimValue,
    EditingSubtask,
    StateDAG,
    ValueProvenance,
    ValueType,
    Visibility,
    editing_schema_for,
)
from molhallulens.modules.text_realization import DetectorPromptSerializer


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


@pytest.fixture(scope="module")
def records() -> tuple[JoinedInputRecord, ...]:
    return ChemCoTMolEditAdapter().load(DATASET_ROOT)


def _one(
    records: tuple[JoinedInputRecord, ...], subtask: EditingSubtask
) -> JoinedInputRecord:
    expected = {
        EditingSubtask.ADD: "add_pilot_origin",
        EditingSubtask.DELETE: "delete_pilot_origin",
        EditingSubtask.SUBSTITUTE: "substitute_pilot_origin",
    }[subtask]
    return next(record for record in records if record.pilot_subtask == expected)


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


@pytest.mark.parametrize(
    ("subtask", "builder_type"),
    (
        (EditingSubtask.ADD, AdditionReferenceDAGBuilder),
        (EditingSubtask.DELETE, DeletionReferenceDAGBuilder),
        (EditingSubtask.SUBSTITUTE, SubstitutionReferenceDAGBuilder),
    ),
)
def test_three_typed_builders_bind_complete_reference_state(
    records: tuple[JoinedInputRecord, ...],
    subtask: EditingSubtask,
    builder_type: type,
) -> None:
    record = _one(records, subtask)
    builder = reference_dag_builder_for(subtask)
    artifact = builder.build(record)
    definition = editing_schema_for(subtask)

    assert type(builder) is builder_type
    assert artifact.normalized_subtask is subtask
    assert artifact.anonymous_sample_id == record.anonymous_sample_id
    assert artifact.state_dag.schema is definition.schema
    assert set(artifact.state_dag.values) == set(definition.schema.nodes_by_id)
    assert not artifact.state_dag.edge_values
    for field_name, node_id in definition.record_field_bindings.items():
        assert artifact.state_dag.values[node_id].provenance is ValueProvenance.REFERENCE
        source = record.process_record if field_name == "answer_smiles" else record.raw_record
        assert artifact.state_dag.values[node_id].raw_value == source[field_name]
    for field_name, node_id in definition.legacy_step_field_bindings.items():
        claim = artifact.state_dag.values[node_id]
        assert claim.provenance is ValueProvenance.REFERENCE
        assert claim.raw_value == record.process_record["parsed_reference_state"][field_name]
    for field_name, node_id in definition.rdkit_reference_bindings.items():
        claim = artifact.state_dag.values[node_id]
        assert claim.provenance is ValueProvenance.RDKIT
        assert claim.raw_value == record.process_record["parsed_reference_state"][field_name]
        assert not claim.mention_ids
    assert all(
        claim.locally_valid is None
        and claim.oracle_match is None
        and claim.confidence is None
        for claim in artifact.state_dag.values.values()
    )


def test_natural_payload_and_formal_slots_round_trip_exactly(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    for subtask in EditingSubtask:
        record = _one(records, subtask)
        artifact = build_reference_dag(record)
        source_steps = record.process_record["formal_cot_trace"]

        assert tuple(step.step_index for step in artifact.trace_steps) == tuple(
            range(1, len(artifact.trace_steps) + 1)
        )
        for built, source in zip(artifact.trace_steps, source_steps, strict=True):
            assert built.natural_language == source["natural_language"]
            assert built.formal_ab == source["formal_ab"]
            assert built.render(include_answer=True) == source["step_text"]
            assert all(binding.source_field for binding in built.slot_bindings)
        assert "\n\nAnswer:" not in artifact.reasoning_chain
        assert artifact.detector_input.final_answer == record.process_record["answer_smiles"]


def test_logical_mentions_are_slot_occurrences_not_text_search_results(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    deletion = build_reference_dag(_one(records, EditingSubtask.DELETE))
    definition = editing_schema_for(EditingSubtask.DELETE)

    step1 = deletion.state_dag.values["remove_group_step1"]
    step2 = deletion.state_dag.values["remove_group_step2"]
    assert definition.semantic_state_for_node("remove_group_step1") == "remove_group"
    assert definition.semantic_state_for_node("remove_group_step2") == "remove_group"
    assert len(step1.mention_ids) == 1
    assert len(step2.mention_ids) == 2
    assert set(step1.mention_ids).isdisjoint(step2.mention_ids)
    assert all(mention.channel != "natural" for mention in deletion.mentions)
    assert len(deletion.state_dag.values["anchor_idx"].mention_ids) == 2


def test_build_only_claims_cannot_flow_into_detector_projection(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    artifact = build_reference_dag(_one(records, EditingSubtask.SUBSTITUTE))
    before = DetectorPromptSerializer().serialize_input(artifact.detector_input)
    nodes = artifact.state_dag.schema.nodes_by_id
    changed_values = dict(artifact.state_dag.values)
    for node_id, node in nodes.items():
        if node.visibility is not Visibility.BUILD_ONLY:
            continue
        claim = changed_values[node_id]
        replacement: str | int
        if claim.value_type in {ValueType.INTEGER, ValueType.ATOM_INDEX, ValueType.COUNT}:
            replacement = 987654
        else:
            replacement = "__BUILD_ONLY_CANARY__"
        changed_values[node_id] = replace(
            claim,
            raw_value=replacement,
            normalized_value=replacement,
        )
    changed = replace(
        artifact,
        state_dag=StateDAG(artifact.state_dag.schema, changed_values),
    )
    after = DetectorPromptSerializer().serialize_input(changed.detector_input)

    assert before == after
    assert "__BUILD_ONLY_CANARY__" not in after.text
    assert all(
        nodes[node_id].visibility is not Visibility.BUILD_ONLY
        for node_id in changed.detector_visible_values
    )


def test_formal_tamper_is_reported_without_fallback_artifact(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    record = _one(records, EditingSubtask.ADD)
    process = _thaw(record.process_record)
    process["formal_cot_trace"][0]["formal_ab"] = process["formal_cot_trace"][0][
        "formal_ab"
    ].replace("idx=", "idx=999")
    tampered = replace(record, process_record=process)

    with pytest.raises(ReferenceDAGBuildError) as captured:
        build_reference_dag(tampered)
    assert tuple(issue.code for issue in captured.value.report.issues) == (
        "FORMAL_STATE_ROUNDTRIP",
    )


def test_controlled_failure_is_present_in_corpus_report(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    record = _one(records, EditingSubtask.ADD)
    process = _thaw(record.process_record)
    process["formal_cot_trace"][0]["step_name"] = "WRONG_STEP"
    tampered = replace(record, process_record=process)

    result = audit_reference_dag_corpus((tampered,))
    assert not result.artifacts
    assert result.report.attempted == 1
    assert result.report.failed == 1
    assert result.report.origins[0].issue_codes == ("STEP_SEQUENCE",)
    assert result.report.issues[0].node_ids == (record.anonymous_sample_id,)


def test_artifact_and_claims_are_deeply_immutable(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    artifact = build_reference_dag(_one(records, EditingSubtask.ADD))
    with pytest.raises(FrozenInstanceError):
        artifact.anonymous_sample_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        artifact.state_dag.values["source"] = ClaimValue(  # type: ignore[index]
            raw_value="C",
            normalized_value="C",
            value_type=ValueType.INDEXED_SMILES,
            provenance=ValueProvenance.REFERENCE,
        )


def test_answer_gt_raw_string_drift_is_not_a_t011_failure(
    records: tuple[JoinedInputRecord, ...],
) -> None:
    drift = next(
        record
        for record in records
        if record.process_record["answer_smiles"] != record.raw_record["gt_smiles"]
    )
    artifact = build_reference_dag(drift)

    assert artifact.state_dag.values["final_answer"].raw_value != artifact.state_dag.values[
        "oracle_gt"
    ].raw_value
    assert artifact.detector_input.final_answer == drift.process_record["answer_smiles"]
