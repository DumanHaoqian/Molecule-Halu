"""Four-stage reference/graph-edit validation gates and corpus audit."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.builders import (
    build_reference_dag,
    classify_edit_truth,
    derive_edit_truth,
)
from molhallulens.chemistry import (
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
)
from molhallulens.domain import (
    AtomReferenceNamespace,
    EditingSubtask,
    OperationSubtype,
    Severity,
    ValidationStage,
)
from molhallulens.validation import (
    GraphEditValidator,
    InputRecordValidator,
    OriginValidationInput,
    RDKitStructureValidator,
    ReferenceDAGValidator,
    ReferenceValidationError,
    audit_reference_corpus,
    validate_reference_origin,
    validate_reference_origin_strict,
)


DATASET_ROOT = Path(__file__).resolve().parents[1] / "Dataset"
REPORT_PATH = DATASET_ROOT / "reports" / "reference_graph_validation_report.json"
EXPECTED_STAGES = (
    ValidationStage.INPUT_RECORD,
    ValidationStage.REFERENCE_DAG,
    ValidationStage.RDKIT_STRUCTURE,
    ValidationStage.GRAPH_EDIT,
)
SPECIAL_ORIGINS = (
    "mol_edit.delete_v2.0081",
    "mol_edit.add_v2.0071",
    "mol_edit.substitute_v2.0064",
    "mol_edit.substitute_v2.0123",
    "mol_edit.substitute_v2.0191",
    "mol_edit.substitute_v2.0271",
)


@lru_cache(maxsize=1)
def _corpus_inputs() -> tuple[OriginValidationInput, ...]:
    inputs = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        artifact = build_reference_dag(record)
        inputs.append(
            OriginValidationInput(
                record=record,
                artifact=artifact,
                edit_truth=derive_edit_truth(artifact),
            )
        )
    return tuple(inputs)


def _origin(anonymous_sample_id: str) -> OriginValidationInput:
    return next(
        item
        for item in _corpus_inputs()
        if item.record.anonymous_sample_id == anonymous_sample_id
    )


def _report_fixture() -> dict[str, Any]:
    with REPORT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _record_with(
    record: JoinedInputRecord,
    *,
    raw_updates: dict[str, Any] | None = None,
    process_updates: dict[str, Any] | None = None,
    template_updates: dict[str, Any] | None = None,
    state_updates: dict[str, Any] | None = None,
) -> JoinedInputRecord:
    raw_record = dict(record.raw_record)
    process_record = dict(record.process_record)
    formal_template = dict(record.formal_template)
    raw_record.update({} if raw_updates is None else raw_updates)
    process_record.update({} if process_updates is None else process_updates)
    formal_template.update({} if template_updates is None else template_updates)
    if state_updates is not None:
        parsed_state = dict(process_record["parsed_reference_state"])
        parsed_state.update(state_updates)
        process_record["parsed_reference_state"] = parsed_state
    return JoinedInputRecord(
        anonymous_sample_id=record.anonymous_sample_id,
        raw_record=raw_record,
        process_record=process_record,
        formal_template=formal_template,
    )


def _artifact_with_claim(item: OriginValidationInput, node_id: str, value: Any):
    artifact = item.artifact
    claim = artifact.state_dag.values[node_id]
    changed_claim = replace(claim, raw_value=value, normalized_value=value)
    changed_dag = replace(
        artifact.state_dag,
        values={**artifact.state_dag.values, node_id: changed_claim},
    )
    return replace(artifact, state_dag=changed_dag)


def _issue_codes(report: Any) -> tuple[str, ...]:
    return tuple(issue.code for issue in report.issues)


def _assert_issue(
    report: Any,
    *,
    code: str,
    stage: ValidationStage,
    anonymous_sample_id: str,
    severity: Severity | None = None,
) -> None:
    matches = tuple(issue for issue in report.issues if issue.code == code)
    assert matches, _issue_codes(report)
    assert all(issue.stage is stage for issue in matches)
    assert all(anonymous_sample_id in issue.node_ids for issue in matches)
    if severity is not None:
        assert all(issue.severity is severity for issue in matches)
    assert report.all_pass is False
    json.dumps(
        [dict(issue.evidence) for issue in report.issues],
        ensure_ascii=False,
        sort_keys=True,
    )


def _fragment_hints(item: OriginValidationInput) -> tuple[str | None, str | None]:
    state = item.record.process_record["parsed_reference_state"]
    subtask = item.artifact.normalized_subtask
    if subtask is EditingSubtask.ADD:
        return None, state["step2_frag_smiles"]
    if subtask is EditingSubtask.DELETE:
        return state["step2_remove_smiles"], None
    return (
        state["step1_remove_group_smiles"],
        state["step1_add_fragment_smiles"],
    )


def _mapping_is_compatible(item: OriginValidationInput, mapping: Any) -> bool:
    truth = item.edit_truth
    mapped_sources = {pair.source.atom_id for pair in mapping.pairs}
    mapped_products = {pair.product for pair in mapping.pairs}
    added_references = {atom.reference for atom in truth.added_atoms}
    return (
        set(truth.valid_anchor_indices).issubset(mapped_sources)
        and set(truth.removed_atom_maps).isdisjoint(mapped_sources)
        and added_references.isdisjoint(mapped_products)
    )


def test_real_150_origin_report_is_frozen_and_all_four_gates_pass() -> None:
    inputs = _corpus_inputs()
    report = audit_reference_corpus(reversed(inputs))
    payload = report.to_dict()

    assert len(inputs) == 150
    assert payload == _report_fixture()
    assert payload["summary"]["attempted"] == 150
    assert payload["summary"]["passed"] == 150
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["all_pass"] is True
    assert payload["summary"]["counts_by_subtask"] == {
        "add": 50,
        "delete": 50,
        "substitute": 50,
    }
    assert payload["summary"]["counts_by_operation_subtype"] == {
        "delete_with_replacement": 1,
        "deprotection": 49,
        "standard": 100,
    }
    assert tuple(item["anonymous_sample_id"] for item in payload["origins"]) == tuple(
        sorted(item.record.anonymous_sample_id for item in inputs)
    )
    for origin in report.origins:
        assert origin.all_pass is True
        assert len(origin.stage_reports) == 4
        origin_payload = origin.to_dict()
        assert tuple(
            ValidationStage(gate["stage"]) for gate in origin_payload["gates"]
        ) == EXPECTED_STAGES


def test_process_product_and_answer_are_graph_equivalent_to_gt_not_raw_equal() -> None:
    raw_product_mismatches = 0
    raw_answer_mismatches = 0
    family_counts: Counter[EditingSubtask] = Counter()

    for item in _corpus_inputs():
        record = item.record
        artifact = item.artifact
        truth = item.edit_truth
        gt_smiles = record.raw_record["gt_smiles"]
        product_smiles = artifact.state_dag.values["product"].normalized_value
        answer_smiles = record.process_record["answer_smiles"]

        assert isomeric_graph_equivalent(product_smiles, gt_smiles)
        assert isomeric_graph_equivalent(answer_smiles, gt_smiles)
        raw_product_mismatches += product_smiles != gt_smiles
        raw_answer_mismatches += answer_smiles != gt_smiles
        family_counts[truth.normalized_subtask] += 1

        assert truth.mapping_evidence.optimal_mappings
        assert any(
            _mapping_is_compatible(item, mapping)
            for mapping in truth.mapping_evidence.optimal_mappings
        )
        assert set(truth.mapping_evidence.trace_anchor_indices).issubset(
            truth.valid_anchor_indices
        )

        remove_hint, add_hint = _fragment_hints(item)
        if truth.remove_fragment is None:
            assert remove_hint is None
        else:
            assert remove_hint is not None
            if record.anonymous_sample_id == "mol_edit.substitute_v2.0191":
                # Attachment capping preserves the one-atom oxygen group while
                # the standalone graph truth carries its explicit charge.
                assert truth.remove_fragment.canonical_smiles == "[O-]"
                assert remove_hint == "O"
                assert truth.remove_fragment.descriptors.heavy_atom_count == 1
            else:
                assert fragment_graph_equivalent(
                    truth.remove_fragment.canonical_smiles,
                    remove_hint,
                )
            assert set(truth.remove_fragment.boundary_bonds).issubset(
                truth.broken_bonds
            )
        if truth.add_fragment is None:
            assert add_hint is None
        else:
            if add_hint is None:
                # The audited delete-with-replacement is graph-derived rather
                # than silently forced through the ordinary deletion trace.
                assert record.anonymous_sample_id == "mol_edit.delete_v2.0081"
            else:
                assert fragment_graph_equivalent(
                    truth.add_fragment.canonical_smiles,
                    add_hint,
                )
            assert set(truth.add_fragment.boundary_bonds).issubset(
                truth.formed_bonds
            )

    assert raw_product_mismatches == 7
    assert raw_answer_mismatches == 7
    assert family_counts == {
        EditingSubtask.ADD: 50,
        EditingSubtask.DELETE: 50,
        EditingSubtask.SUBSTITUTE: 50,
    }


@pytest.mark.parametrize("anonymous_sample_id", SPECIAL_ORIGINS)
def test_registered_boundary_origins_pass_graph_edit_validation(
    anonymous_sample_id: str,
) -> None:
    item = _origin(anonymous_sample_id)
    report = GraphEditValidator().validate(
        item.record,
        item.artifact,
        item.edit_truth,
    )

    assert report.all_pass is True, _issue_codes(report)
    assert validate_reference_origin(item).all_pass is True

    truth = item.edit_truth
    if anonymous_sample_id == "mol_edit.delete_v2.0081":
        assert classify_edit_truth(truth).operation_subtype is (
            OperationSubtype.DELETE_WITH_REPLACEMENT
        )
        assert len(truth.removed_atom_maps) == 24
        assert len(truth.added_atoms) == 2
        assert len(truth.broken_bonds) == len(truth.formed_bonds) == 1
    elif anonymous_sample_id == "mol_edit.add_v2.0071":
        assert truth.valid_anchor_indices == (30,)
        assert truth.mapping_evidence.optimal_mapping_count > 1
        assert all(
            _mapping_is_compatible(item, mapping)
            for mapping in truth.mapping_evidence.optimal_mappings
        )
    elif anonymous_sample_id == "mol_edit.substitute_v2.0064":
        assert truth.valid_anchor_indices == (25,)
        assert truth.remove_fragment is not None
        assert truth.remove_fragment.canonical_smiles == "Cl"
    elif anonymous_sample_id == "mol_edit.substitute_v2.0123":
        assert truth.valid_anchor_indices == (22,)
        assert truth.add_fragment is not None
        assert fragment_graph_equivalent(
            truth.add_fragment.canonical_smiles,
            "N1C=NC=N1",
        )
    elif anonymous_sample_id == "mol_edit.substitute_v2.0191":
        classification = classify_edit_truth(truth)
        assert classification.registered is True
        assert classification.provenance
        assert truth.remove_fragment is not None
        assert truth.remove_fragment.canonical_smiles == "[O-]"
        assert artifact_claim(item, "remove_group") == "O"
    else:
        assert truth.valid_anchor_indices == (20, 35)
        assert truth.broken_bonds[0].begin.atom_id == 20
        retained_formed_endpoints = tuple(
            endpoint.atom_id
            for endpoint in (
                truth.formed_bonds[0].begin,
                truth.formed_bonds[0].end,
            )
            if endpoint.namespace is AtomReferenceNamespace.SOURCE_MAP
        )
        assert retained_formed_endpoints == (35,)


def test_input_id_subtask_and_required_field_tampering_fail_closed() -> None:
    item = _origin("mol_edit.add_v2.0003")
    validator = InputRecordValidator()

    bad_id = copy.copy(item.record)
    object.__setattr__(bad_id, "anonymous_sample_id", "fixture.cross_wired_id")
    _assert_issue(
        validator.validate(bad_id),
        code="INPUT_ID_MISMATCH",
        stage=ValidationStage.INPUT_RECORD,
        anonymous_sample_id="fixture.cross_wired_id",
        severity=Severity.FATAL,
    )

    invalid_subtask = _record_with(
        item.record,
        raw_updates={"subtask": "unknown_pilot_origin"},
        process_updates={"subtask": "unknown_pilot_origin"},
        template_updates={"subtask": "unknown_pilot_origin"},
    )
    _assert_issue(
        validator.validate(invalid_subtask),
        code="INPUT_SUBTASK_INVALID",
        stage=ValidationStage.INPUT_RECORD,
        anonymous_sample_id=item.record.anonymous_sample_id,
    )

    raw_without_instruction = dict(item.record.raw_record)
    del raw_without_instruction["instruction"]
    missing_required = _record_with(item.record)
    object.__setattr__(missing_required, "raw_record", raw_without_instruction)
    _assert_issue(
        validator.validate(missing_required),
        code="INPUT_REQUIRED_FIELDS_MISSING",
        stage=ValidationStage.INPUT_RECORD,
        anonymous_sample_id=item.record.anonymous_sample_id,
    )


def test_reference_round_trip_product_and_answer_tampering_are_ledgered() -> None:
    item = _origin("mol_edit.add_v2.0003")
    validator = ReferenceDAGValidator()

    bad_step = copy.copy(item.artifact.trace_steps[0])
    object.__setattr__(bad_step, "step_text", f"{bad_step.step_text} corrupted")
    bad_round_trip = replace(
        item.artifact,
        trace_steps=(bad_step, *item.artifact.trace_steps[1:]),
    )
    _assert_issue(
        validator.validate(item.record, bad_round_trip),
        code="REFERENCE_TRACE_ROUND_TRIP_FAILED",
        stage=ValidationStage.REFERENCE_DAG,
        anonymous_sample_id=item.record.anonymous_sample_id,
        severity=Severity.FATAL,
    )

    bad_product = _artifact_with_claim(item, "product", "C")
    _assert_issue(
        validator.validate(item.record, bad_product),
        code="REFERENCE_PRODUCT_GT_MISMATCH",
        stage=ValidationStage.REFERENCE_DAG,
        anonymous_sample_id=item.record.anonymous_sample_id,
        severity=Severity.ERROR,
    )

    bad_answer_record = _record_with(
        item.record,
        process_updates={"answer_smiles": "C"},
    )
    bad_answer_artifact = _artifact_with_claim(item, "final_answer", "C")
    _assert_issue(
        validator.validate(bad_answer_record, bad_answer_artifact),
        code="REFERENCE_ANSWER_GT_MISMATCH",
        stage=ValidationStage.REFERENCE_DAG,
        anonymous_sample_id=item.record.anonymous_sample_id,
    )


def test_rdkit_sanitize_and_descriptor_tampering_fail_closed() -> None:
    item = _origin("mol_edit.add_v2.0003")
    validator = RDKitStructureValidator()

    invalid_valence = _record_with(
        item.record,
        raw_updates={"indexed_smiles": "[CH5]"},
    )
    _assert_issue(
        validator.validate(invalid_valence, item.artifact, item.edit_truth),
        code="RDKIT_STRICT_SANITIZE_FAILED",
        stage=ValidationStage.RDKIT_STRUCTURE,
        anonymous_sample_id=item.record.anonymous_sample_id,
        severity=Severity.FATAL,
    )

    descriptor_drift = replace(
        item.edit_truth,
        source_descriptors=replace(
            item.edit_truth.source_descriptors,
            molecular_weight=item.edit_truth.source_descriptors.molecular_weight + 1.0,
        ),
    )
    _assert_issue(
        validator.validate(item.record, item.artifact, descriptor_drift),
        code="RDKIT_DESCRIPTOR_MISMATCH",
        stage=ValidationStage.RDKIT_STRUCTURE,
        anonymous_sample_id=item.record.anonymous_sample_id,
        severity=Severity.ERROR,
    )


def artifact_claim(item: OriginValidationInput, node_id: str) -> Any:
    return item.artifact.state_dag.values[node_id].normalized_value


def test_unregistered_attachment_charge_capping_does_not_get_a_silent_fallback() -> None:
    item = _origin("mol_edit.substitute_v2.0191")
    anonymous_sample_id = "mol_edit.substitute_v2.9999"
    raw_record = dict(item.record.raw_record)
    process_record = dict(item.record.process_record)
    raw_record["anonymous_sample_id"] = anonymous_sample_id
    process_record["anonymous_sample_id"] = anonymous_sample_id
    unknown_record = JoinedInputRecord(
        anonymous_sample_id=anonymous_sample_id,
        raw_record=raw_record,
        process_record=process_record,
        formal_template=item.record.formal_template,
    )
    unknown_artifact = replace(
        item.artifact,
        anonymous_sample_id=anonymous_sample_id,
    )
    unknown_truth = replace(
        item.edit_truth,
        anonymous_sample_id=anonymous_sample_id,
    )

    report = GraphEditValidator().validate(
        unknown_record,
        unknown_artifact,
        unknown_truth,
    )
    _assert_issue(
        report,
        code="GRAPH_FRAGMENT_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=anonymous_sample_id,
    )


def test_registered_provenance_does_not_authorize_a_different_charge_change() -> None:
    item = _origin("mol_edit.substitute_v2.0064")
    assert artifact_claim(item, "remove_group") == "Cl"
    charged_chloride = _artifact_with_claim(item, "remove_group", "[Cl-]")

    report = GraphEditValidator().validate(
        item.record,
        charged_chloride,
        item.edit_truth,
    )
    _assert_issue(
        report,
        code="GRAPH_FRAGMENT_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=item.record.anonymous_sample_id,
        severity=Severity.ERROR,
    )


def test_unregistered_delete_with_addition_remains_a_fatal_graph_issue() -> None:
    item = _origin("mol_edit.delete_v2.0081")
    anonymous_sample_id = "mol_edit.delete_v2.9999"
    raw_record = dict(item.record.raw_record)
    process_record = dict(item.record.process_record)
    raw_record["anonymous_sample_id"] = anonymous_sample_id
    process_record["anonymous_sample_id"] = anonymous_sample_id
    unknown_record = JoinedInputRecord(
        anonymous_sample_id=anonymous_sample_id,
        raw_record=raw_record,
        process_record=process_record,
        formal_template=item.record.formal_template,
    )
    unknown_artifact = replace(
        item.artifact,
        anonymous_sample_id=anonymous_sample_id,
    )
    unknown_truth = replace(
        item.edit_truth,
        anonymous_sample_id=anonymous_sample_id,
    )

    report = GraphEditValidator().validate(
        unknown_record,
        unknown_artifact,
        unknown_truth,
    )
    _assert_issue(
        report,
        code="UNREGISTERED_DELETE_WITH_ADDITION",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=anonymous_sample_id,
        severity=Severity.FATAL,
    )


def test_graph_anchor_group_and_fragment_tampering_fail_closed() -> None:
    validator = GraphEditValidator()

    add_item = _origin("mol_edit.add_v2.0003")
    mapped_source_ids = tuple(
        pair.source.atom_id
        for pair in add_item.edit_truth.mapping_evidence.optimal_mappings[0].pairs
    )
    wrong_anchor = next(
        atom_id
        for atom_id in mapped_source_ids
        if atom_id not in add_item.edit_truth.valid_anchor_indices
    )
    anchor_drift = replace(
        add_item.edit_truth,
        valid_anchor_indices=(wrong_anchor,),
        symmetry_equivalent_anchors=(),
    )
    _assert_issue(
        validator.validate(add_item.record, add_item.artifact, anchor_drift),
        code="GRAPH_ANCHOR_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=add_item.record.anonymous_sample_id,
    )

    delete_item = _origin("mol_edit.delete_v2.0016")
    group_drift = _artifact_with_claim(delete_item, "remove_group_step2", "N")
    _assert_issue(
        validator.validate(delete_item.record, group_drift, delete_item.edit_truth),
        code="GRAPH_FRAGMENT_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=delete_item.record.anonymous_sample_id,
    )

    fragment_drift = _artifact_with_claim(add_item, "add_fragment", "N")
    _assert_issue(
        validator.validate(add_item.record, fragment_drift, add_item.edit_truth),
        code="GRAPH_FRAGMENT_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=add_item.record.anonymous_sample_id,
    )


def test_graph_bond_and_edit_family_tampering_fail_closed() -> None:
    validator = GraphEditValidator()

    delete_item = _origin("mol_edit.delete_v2.0016")
    missing_bond = replace(delete_item.edit_truth, broken_bonds=())
    boundary_report = validator.validate(
        delete_item.record,
        delete_item.artifact,
        missing_bond,
    )
    assert {
        "GRAPH_BOUNDARY_MISMATCH",
        "GRAPH_EDIT_REDERIVATION_MISMATCH",
    }.intersection(_issue_codes(boundary_report))
    assert all(
        issue.stage is ValidationStage.GRAPH_EDIT for issue in boundary_report.issues
    )
    assert boundary_report.all_pass is False

    add_item = _origin("mol_edit.add_v2.0003")
    wrong_family = replace(
        add_item.edit_truth,
        normalized_subtask=EditingSubtask.DELETE,
    )
    _assert_issue(
        validator.validate(add_item.record, add_item.artifact, wrong_family),
        code="GRAPH_TRUTH_SUBTASK_MISMATCH",
        stage=ValidationStage.GRAPH_EDIT,
        anonymous_sample_id=add_item.record.anonymous_sample_id,
    )


def test_combined_ledger_is_deterministic_and_strict_entrypoint_rejects() -> None:
    item = _origin("mol_edit.add_v2.0003")
    bad_artifact = _artifact_with_claim(item, "product", "C")
    tampered = OriginValidationInput(
        record=item.record,
        artifact=bad_artifact,
        edit_truth=item.edit_truth,
    )

    first = validate_reference_origin(tampered)
    second = validate_reference_origin(tampered)
    assert first == second
    assert first.all_pass is False
    assert tuple(
        ValidationStage(gate["stage"]) for gate in first.to_dict()["gates"]
    ) == EXPECTED_STAGES
    assert {issue.stage for issue in first.issues}.issubset(set(EXPECTED_STAGES))
    assert "REFERENCE_PRODUCT_GT_MISMATCH" in first.issue_codes
    json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)

    with pytest.raises(ReferenceValidationError) as captured:
        validate_reference_origin_strict(tampered)
    assert captured.value.report == first.report

    corpus = audit_reference_corpus((tampered,), require_all_pass=False)
    assert corpus.to_dict()["summary"]["failed"] == 1
    assert corpus.origins == (first,)
