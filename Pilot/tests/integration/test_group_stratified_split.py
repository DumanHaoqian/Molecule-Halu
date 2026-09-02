"""Frozen-corpus acceptance tests for the T028 group-stratified split."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference.truth import derive_edit_truth
from molhallulens.modules.release.leakage import assign_leakage_groups
from molhallulens.modules.release.origin_audit import audit_origin_split_features
from molhallulens.modules.reference.builder import build_reference_dag
from molhallulens.modules.release.splitter import (
    FROZEN_SPLIT_SEED,
    GroupStratifiedSplitResult,
    GroupStratifiedSplitter,
    SplitName,
    SplitOrigin,
    split_origins_from_audit,
)
from molhallulens.core import EditingSubtask
from molhallulens.infrastructure.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
REPORT_PATH = DATASET_ROOT / "reports" / "split_balance_report.json"


@cache
def _validated_inputs() -> tuple[OriginValidationInput, ...]:
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


@cache
def _origins() -> tuple[SplitOrigin, ...]:
    items = _validated_inputs()
    audit = audit_origin_split_features(items).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in items
        },
    )
    return split_origins_from_audit(audit, leakage)


@cache
def _result() -> GroupStratifiedSplitResult:
    return GroupStratifiedSplitter().solve(_origins())


def test_real_corpus_hits_frozen_hard_targets_without_splitting_groups() -> None:
    result = _result()
    assert result.report.split_seed == FROZEN_SPLIT_SEED
    assert result.report.feasibility_proof.exact_target_reachable
    assert result.report.feasibility_proof.selected_target_kind == "exact"
    assert Counter(item.split for item in result.assignments) == {
        SplitName.TRAIN: 100,
        SplitName.VALIDATION: 25,
        SplitName.TEST: 25,
    }
    assert Counter((item.subtask, item.split) for item in result.assignments) == {
        (EditingSubtask.ADD, SplitName.TRAIN): 34,
        (EditingSubtask.ADD, SplitName.VALIDATION): 8,
        (EditingSubtask.ADD, SplitName.TEST): 8,
        (EditingSubtask.DELETE, SplitName.TRAIN): 33,
        (EditingSubtask.DELETE, SplitName.VALIDATION): 9,
        (EditingSubtask.DELETE, SplitName.TEST): 8,
        (EditingSubtask.SUBSTITUTE, SplitName.TRAIN): 33,
        (EditingSubtask.SUBSTITUTE, SplitName.VALIDATION): 8,
        (EditingSubtask.SUBSTITUTE, SplitName.TEST): 9,
    }
    group_splits: dict[str, set[SplitName]] = defaultdict(set)
    for assignment in result.assignments:
        group_splits[assignment.leakage_group_id].add(assignment.split)
    assert len(group_splits) == 142
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_real_corpus_split_is_input_order_and_byte_stable() -> None:
    result = _result()
    origins = _origins()
    reverse = GroupStratifiedSplitter().solve(reversed(origins))
    interleaved = GroupStratifiedSplitter().solve(origins[::2] + origins[1::2])
    assert reverse == result == interleaved
    assert reverse.to_json_bytes() == result.to_json_bytes()
    assert interleaved.to_json_bytes() == result.to_json_bytes()


def test_real_report_freezes_complete_strata_and_exact_bytes() -> None:
    result = _result()
    assert result.report.strata_balances
    assert result.report.soft_objective_after <= result.report.soft_objective_before
    features = {item.feature for item in result.report.strata_balances}
    assert {
        "subtask",
        "operation_subtype",
        "rxn_cls",
        "anchor_element",
        "source_heavy_atom_quantile_bin",
        "source_ring_quantile_bin",
        "heavy_atom_delta_bin",
        "ring_delta_bin",
        "fragment_size_bin",
        "mol_complexity_quantile_bin",
        "tanimoto_quantile_bin",
    } <= features
    assert any(feature.startswith("capability:") for feature in features)
    assert any(feature.startswith("operator:") for feature in features)
    assert REPORT_PATH.read_bytes() == result.to_json_bytes()
