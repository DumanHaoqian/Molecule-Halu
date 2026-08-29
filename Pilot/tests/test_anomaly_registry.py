"""Exact-ID anomaly registry, subtype audit, and capability-policy tests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import (
    DEFAULT_ANOMALY_REGISTRY,
    AnomalyRegistryError,
    audit_anomaly_registry,
    build_reference_dag,
    classify_edit_truth,
    derive_edit_truth,
    structural_edit_signature,
)
from molhallulens.domain import (
    AnomalyProvenance,
    EditingSubtask,
    EditTruth,
    OperationSubtype,
    OperatorCapability,
    StructuralEditSignature,
)

DATASET_ROOT = Path(__file__).resolve().parents[1] / "Dataset"
REPORT_PATH = DATASET_ROOT / "reports" / "anomaly_registry_report.json"
REPLACEMENT_ID = "mol_edit.delete_v2.0081"
BOUNDARY_PROVENANCE = {
    "mol_edit.add_v2.0071": AnomalyProvenance.MAPPING_TRACE_DISAMBIGUATION,
    "mol_edit.substitute_v2.0064": (
        AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION
    ),
    "mol_edit.substitute_v2.0123": AnomalyProvenance.AROMATIC_FRAGMENT_CAPPING,
    "mol_edit.substitute_v2.0191": (
        AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION
    ),
    "mol_edit.substitute_v2.0271": AnomalyProvenance.MULTI_ANCHOR_RELOCATION,
    "mol_edit.substitute_v2.0276": (
        AnomalyProvenance.SUBSTITUTION_ANCHOR_STEREO_ASSIGNMENT
    ),
}


@lru_cache(maxsize=1)
def _corpus_truths() -> tuple[EditTruth, ...]:
    return tuple(
        derive_edit_truth(build_reference_dag(record))
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
    )


def _truth(anonymous_sample_id: str) -> EditTruth:
    return next(
        truth
        for truth in _corpus_truths()
        if truth.anonymous_sample_id == anonymous_sample_id
    )


def _report_fixture() -> dict[str, Any]:
    with REPORT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_real_150_subtype_counts_and_report_are_frozen() -> None:
    report = audit_anomaly_registry(_corpus_truths())
    classifications = report.classifications

    assert len(classifications) == 150
    assert Counter(item.operation_subtype for item in classifications) == {
        OperationSubtype.STANDARD: 100,
        OperationSubtype.DEPROTECTION: 49,
        OperationSubtype.DELETE_WITH_REPLACEMENT: 1,
    }
    assert report.registered_count == 7
    assert report.complete_registry is True
    assert report.observed_registry_ids == report.expected_registry_ids
    assert report.to_dict() == _report_fixture()


def test_delete_v2_0081_has_exact_replacement_signature_and_restrictions() -> None:
    truth = _truth(REPLACEMENT_ID)
    classification = classify_edit_truth(truth)

    expected_signature = StructuralEditSignature(
        removed_atom_count=24,
        added_atomic_numbers=(6, 7),
        broken_boundary_bond_count=1,
        formed_boundary_bond_count=1,
        remove_fragment_heavy_atoms=24,
        add_fragment_heavy_atoms=2,
    )
    assert structural_edit_signature(truth) == expected_signature
    assert classification.structural_signature == expected_signature
    assert classification.operation_subtype is OperationSubtype.DELETE_WITH_REPLACEMENT
    assert classification.registered is True
    assert classification.provenance == (AnomalyProvenance.DELETE_WITH_REPLACEMENT,)
    assert classification.allows(OperatorCapability.REMOVE_ONLY_DELTA_RULE) is False
    assert classification.allows(OperatorCapability.STRUCTURAL_DELETION) is False
    assert classification.allows(OperatorCapability.REPLACEMENT_AWARE_DELETION) is True
    assert classification.allows(OperatorCapability.CLAIM_PERTURBATION) is True
    assert classification.allows(OperatorCapability.TERMINAL_PERTURBATION) is True


def test_unregistered_delete_with_addition_fails_closed() -> None:
    unregistered = replace(
        _truth(REPLACEMENT_ID),
        anonymous_sample_id="fixture.unregistered_delete_with_addition",
    )

    with pytest.raises(AnomalyRegistryError) as captured:
        classify_edit_truth(unregistered)

    assert tuple(issue.code for issue in captured.value.report.issues) == (
        "UNREGISTERED_DELETE_WITH_ADDITION",
    )
    assert captured.value.report.issues[0].node_ids == (
        "fixture.unregistered_delete_with_addition",
    )


def test_registry_is_deeply_immutable_and_exact_id_only() -> None:
    registry = DEFAULT_ANOMALY_REGISTRY
    replacement_entry = registry.entry_for(REPLACEMENT_ID)

    assert replacement_entry is not None
    assert type(registry.entries) is tuple
    assert type(replacement_entry.provenance) is tuple
    assert replacement_entry.capability_policy is not None
    assert type(replacement_entry.capability_policy.allowed) is frozenset
    assert type(replacement_entry.capability_policy.forbidden) is frozenset

    with pytest.raises(TypeError):
        registry.entries_by_id["fixture.injected"] = replacement_entry  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        replacement_entry.provenance_task_id = "silently_changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        replacement_entry.capability_policy.allowed.add(  # type: ignore[union-attr]
            OperatorCapability.REMOVE_ONLY_DELTA_RULE
        )

    assert registry.entry_for("MOL_EDIT.DELETE_V2.0081") is None
    assert registry.entry_for("mol_edit.delete_v2.0081.json") is None
    assert registry.entry_for("prefix/mol_edit.delete_v2.0081") is None
    assert registry.entry_for(" mol_edit.delete_v2.0081 ") is None
    with pytest.raises(ValueError):
        registry.entry_for("")
    with pytest.raises(TypeError):
        registry.entry_for(None)  # type: ignore[arg-type]


def test_unknown_standard_origin_is_not_silently_special_cased() -> None:
    unknown_addition = replace(
        _truth("mol_edit.add_v2.0071"),
        anonymous_sample_id="fixture.opaque_unknown_standard_addition",
    )

    classification = classify_edit_truth(unknown_addition)
    assert (
        DEFAULT_ANOMALY_REGISTRY.entry_for(unknown_addition.anonymous_sample_id) is None
    )
    assert classification.operation_subtype is OperationSubtype.STANDARD
    assert classification.registered is False
    assert classification.provenance == ()


def test_ordinary_deletion_cannot_use_replacement_specific_capability() -> None:
    classification = classify_edit_truth(_truth("mol_edit.delete_v2.0016"))

    assert classification.operation_subtype is OperationSubtype.DEPROTECTION
    assert classification.allows(OperatorCapability.STRUCTURAL_DELETION) is True
    assert classification.allows(OperatorCapability.REPLACEMENT_AWARE_DELETION) is False


def test_boundary_entries_preserve_t013_provenance_without_subtype_override() -> None:
    for anonymous_sample_id, expected_provenance in BOUNDARY_PROVENANCE.items():
        entry = DEFAULT_ANOMALY_REGISTRY.entry_for(anonymous_sample_id)
        classification = classify_edit_truth(_truth(anonymous_sample_id))

        assert entry is not None
        assert entry.expected_subtask in {
            EditingSubtask.ADD,
            EditingSubtask.SUBSTITUTE,
        }
        assert entry.provenance == (expected_provenance,)
        assert entry.provenance_task_id == "T013"
        assert entry.operation_subtype_override is None
        assert entry.expected_signature is None
        assert entry.capability_policy is None
        assert classification.operation_subtype is OperationSubtype.STANDARD
        assert classification.provenance == (expected_provenance,)


def test_complete_audit_rejects_a_missing_registered_boundary_origin() -> None:
    missing_id = "mol_edit.substitute_v2.0123"
    incomplete = tuple(
        truth for truth in _corpus_truths() if truth.anonymous_sample_id != missing_id
    )

    with pytest.raises(AnomalyRegistryError) as captured:
        audit_anomaly_registry(incomplete, require_complete_registry=True)

    assert tuple(issue.code for issue in captured.value.report.issues) == (
        "MISSING_REGISTERED_ANOMALY",
    )
    assert captured.value.report.issues[0].node_ids == (missing_id,)
