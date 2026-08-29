"""Unit contracts for the deterministic T026 origin feature audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError
from functools import lru_cache
from pathlib import Path

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import (
    KNOWN_DUPLICATE_SCAFFOLD_GROUPS,
    KNOWN_DUPLICATE_SOURCE_GROUPS,
    QuantileThresholds,
    audit_origin_split_features,
    build_reference_dag,
    derive_edit_truth,
)
from molhallulens.chemistry import FragmentPolicy, murcko_scaffold_smiles
from molhallulens.domain import EditingSubtask, OperatorCapability
from molhallulens.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


@lru_cache(maxsize=1)
def _inputs() -> tuple[OriginValidationInput, ...]:
    values = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        artifact = build_reference_dag(record)
        values.append(
            OriginValidationInput(
                record=record,
                artifact=artifact,
                edit_truth=derive_edit_truth(artifact),
            )
        )
    return tuple(values)


@lru_cache(maxsize=1)
def _result():
    return audit_origin_split_features(_inputs())


def test_nearest_rank_quantiles_are_tie_preserving() -> None:
    thresholds = QuantileThresholds.from_values(
        "source_ring_count",
        (1, 1, 1, 2, 2, 3, 4, 4),
    )

    assert thresholds.to_dict() == {
        "feature": "source_ring_count",
        "population_count": 8,
        "q25": 1.0,
        "q50": 2.0,
        "q75": 3.0,
    }
    assert [thresholds.bin_for(value) for value in (1, 2, 3, 4)] == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]


def test_all_origin_hashes_and_stratification_fields_bind_t013_truth() -> None:
    audit = _result().audit
    inputs_by_id = {item.edit_truth.anonymous_sample_id: item for item in _inputs()}

    assert len(audit.records) == 150
    assert audit.t015_attempted == audit.t015_passed == 150
    assert Counter(item.subtask for item in audit.records) == {
        EditingSubtask.ADD: 50,
        EditingSubtask.DELETE: 50,
        EditingSubtask.SUBSTITUTE: 50,
    }
    for record in audit.records:
        truth = inputs_by_id[record.anonymous_sample_id].edit_truth
        assert (
            record.canonical_source_sha256
            == hashlib.sha256(truth.canonical_source_smiles.encode()).hexdigest()
        )
        assert (
            record.canonical_gt_sha256
            == hashlib.sha256(truth.canonical_gt_smiles.encode()).hexdigest()
        )
        scaffold = murcko_scaffold_smiles(
            truth.canonical_source_smiles,
            fragment_policy=FragmentPolicy.LARGEST_HEAVY,
        )
        assert record.scaffold_identity == scaffold
        assert (
            record.origin_id
            == inputs_by_id[record.anonymous_sample_id].record.raw_record["orig_id"]
        )
        assert record.rxn_cls
        assert record.anchor_element == "+".join(record.anchor_elements)
        assert record.source_heavy_atom_quantile_bin in {"q1", "q2", "q3", "q4"}
        assert record.source_ring_quantile_bin in {"q1", "q2", "q3", "q4"}
        assert record.fragment_size == max(
            record.remove_fragment_heavy_atom_count,
            record.add_fragment_heavy_atom_count,
        )


def test_duplicate_source_gt_and_scaffold_inventories_are_exact() -> None:
    audit = _result().audit

    assert tuple(
        group.anonymous_sample_ids for group in audit.duplicate_source_groups
    ) == tuple(sorted(KNOWN_DUPLICATE_SOURCE_GROUPS))
    assert tuple(
        group.anonymous_sample_ids for group in audit.duplicate_scaffold_groups
    ) == tuple(sorted(KNOWN_DUPLICATE_SCAFFOLD_GROUPS))
    assert len({item.canonical_source_sha256 for item in audit.records}) == 146
    assert len({item.canonical_gt_sha256 for item in audit.records}) == 150
    assert len({item.scaffold_sha256 for item in audit.records}) == 143
    assert (
        sum(len(group.anonymous_sample_ids) for group in audit.duplicate_source_groups)
        == 7
    )
    assert (
        sum(
            len(group.anonymous_sample_ids) for group in audit.duplicate_scaffold_groups
        )
        == 13
    )


def test_operator_eligibility_is_static_registry_plus_capability_policy() -> None:
    records = _result().audit.records
    anomaly = next(
        item
        for item in records
        if item.anonymous_sample_id == "mol_edit.delete_v2.0081"
    )

    assert len(anomaly.operator_availability.registered_operator_ids) == 12
    assert len(anomaly.operator_availability.eligible_operator_ids) == 4
    assert len(anomaly.operator_availability.ineligible_operator_reasons) == 8
    assert all(
        reasons == ("capability_forbidden:structural_deletion",)
        for _, reasons in anomaly.operator_availability.ineligible_operator_reasons
    )
    anomaly_flags = dict(anomaly.operator_availability.capability_flags)
    assert anomaly_flags == {
        OperatorCapability.REMOVE_ONLY_DELTA_RULE.value: False,
        OperatorCapability.STRUCTURAL_DELETION.value: False,
        OperatorCapability.CLAIM_PERTURBATION.value: True,
        OperatorCapability.TERMINAL_PERTURBATION.value: True,
    }
    assert Counter(
        (item.subtask.value, len(item.operator_availability.eligible_operator_ids))
        for item in records
    ) == {
        ("add", 11): 50,
        ("delete", 12): 49,
        ("delete", 4): 1,
        ("substitute", 12): 50,
    }


def test_raw_complexity_and_tanimoto_missingness_is_explicit() -> None:
    records = _result().audit.records
    deletion = tuple(item for item in records if item.subtask is EditingSubtask.DELETE)
    present = tuple(
        item for item in records if item.subtask is not EditingSubtask.DELETE
    )

    assert len(deletion) == 50
    assert all(
        item.mol_complexity is None
        and item.mol_complexity_source == "missing_in_raw_benchmark_data"
        and item.mol_complexity_quantile_bin == "missing"
        and item.tanimoto is None
        and item.tanimoto_source == "missing_in_raw_benchmark_data"
        and item.tanimoto_quantile_bin == "missing"
        for item in deletion
    )
    assert all(
        item.mol_complexity is not None
        and item.mol_complexity_source == "raw_benchmark_data"
        and item.tanimoto is not None
        and item.tanimoto_source == "raw_benchmark_data"
        for item in present
    )
    report = _result().feature_distribution.to_dict()
    assert report["missing_features"] == {"mol_complexity": 50, "tanimoto": 50}


def test_audit_result_is_immutable_and_rejects_incomplete_corpus() -> None:
    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.audit.records[0].rxn_cls = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="150-origin"):
        audit_origin_split_features(_inputs()[:-1])
