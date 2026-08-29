"""Unit contracts for the deterministic T028 split solver."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from molhallulens.builders.splitter import (
    FROZEN_SPLIT_SEED,
    GroupStratifiedSplitter,
    SplitName,
    SplitOrigin,
    SplitSolverError,
    derive_split_seed,
)
from molhallulens.domain import EditingSubtask


def _origins(
    *,
    pair_delete_groups: bool = False,
    whole_add_group: bool = False,
) -> tuple[SplitOrigin, ...]:
    values = []
    for subtask in EditingSubtask:
        for index in range(50):
            anonymous_id = f"mol_edit.{subtask.value}.synthetic_{index:04d}"
            if whole_add_group and subtask is EditingSubtask.ADD:
                group_id = "lg:all-add"
            elif pair_delete_groups and subtask is EditingSubtask.DELETE:
                group_id = f"lg:delete-pair-{index // 2:02d}"
            elif index in {0, 1}:
                group_id = f"lg:{subtask.value}:known-pair"
            else:
                group_id = f"lg:{anonymous_id}"
            values.append(
                SplitOrigin(
                    origin_id=f"legacy-{subtask.value}-{index:04d}",
                    anonymous_sample_id=anonymous_id,
                    leakage_group_id=group_id,
                    subtask=subtask,
                    strata=(
                        ("rxn_cls", str(index % 5)),
                        ("anchor_element", ("C", "N", "O")[index % 3]),
                        ("source_heavy_atom_quantile_bin", f"q{index % 4 + 1}"),
                        ("operator:synthetic", "available"),
                    ),
                )
            )
    return tuple(values)


def test_pilot_seed_is_sha256_first_eight_bytes_big_endian() -> None:
    assert derive_split_seed("pilot_v1") == FROZEN_SPLIT_SEED
    assert FROZEN_SPLIT_SEED == 8347206628578381721
    assert derive_split_seed("pilot_v2") != FROZEN_SPLIT_SEED


def test_exact_solver_hits_frozen_totals_and_subtask_matrix() -> None:
    result = GroupStratifiedSplitter().solve(_origins())
    assert result.report.feasibility_proof.exact_target_reachable
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
    assert all(item.deviation == 0 for item in result.report.cell_balances)


def test_leakage_groups_are_atomic() -> None:
    result = GroupStratifiedSplitter().solve(_origins())
    group_splits: dict[str, set[SplitName]] = defaultdict(set)
    for assignment in result.assignments:
        group_splits[assignment.leakage_group_id].add(assignment.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert (
        result.report.to_dict()["hard_constraints"]["groups_split_across_partitions"]
        == 0
    )


def test_input_order_does_not_change_assignments_or_report_bytes() -> None:
    origins = _origins()
    forward = GroupStratifiedSplitter().solve(origins)
    reversed_result = GroupStratifiedSplitter().solve(tuple(reversed(origins)))
    shuffled_without_random = GroupStratifiedSplitter().solve(
        origins[::2] + origins[1::2]
    )
    assert forward == reversed_result == shuffled_without_random
    assert forward.to_json_bytes() == reversed_result.to_json_bytes()
    assert forward.to_json_bytes() == shuffled_without_random.to_json_bytes()


def test_exact_infeasibility_is_proved_before_within_one_fallback() -> None:
    result = GroupStratifiedSplitter().solve(_origins(pair_delete_groups=True))
    proof = result.report.feasibility_proof
    assert not proof.exact_target_reachable
    assert proof.selected_target_kind == "within_one"
    assert proof.to_dict()["exact_infeasibility_proved"]
    assert proof.target_matrices_examined > 1
    assert all(abs(item.deviation) <= 1 for item in result.report.cell_balances)
    assert any(item.deviation for item in result.report.cell_balances)
    assert Counter(item.split for item in result.assignments) == {
        SplitName.TRAIN: 100,
        SplitName.VALIDATION: 25,
        SplitName.TEST: 25,
    }


def test_infeasible_even_within_one_fails_closed() -> None:
    with pytest.raises(SplitSolverError) as captured:
        GroupStratifiedSplitter().solve(_origins(whole_add_group=True))
    assert captured.value.code == "SPLIT_INFEASIBLE"
    assert captured.value.evidence["target_matrices_examined"] > 1


def test_soft_balance_is_complete_and_never_worsens() -> None:
    report = GroupStratifiedSplitter().solve(_origins()).report
    assert report.strata_balances
    assert report.soft_objective_after <= report.soft_objective_before
    observed_features = {item.feature for item in report.strata_balances}
    assert {
        "subtask",
        "rxn_cls",
        "anchor_element",
        "source_heavy_atom_quantile_bin",
        "operator:synthetic",
    } <= observed_features
    for item in report.strata_balances:
        assert sum(count for _, count in item.split_counts) == item.global_count


def test_public_results_are_frozen_and_report_is_canonical_json() -> None:
    result = GroupStratifiedSplitter().solve(_origins())
    with pytest.raises(FrozenInstanceError):
        result.assignments = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.assignments[0].split = SplitName.TEST  # type: ignore[misc]
    payload = result.to_json_bytes()
    assert payload.endswith(b"\n")
    assert b'"uses_python_hash":false' in payload
    assert b'"uses_random_module":false' in payload


def test_bad_inventory_fails_before_search() -> None:
    with pytest.raises(SplitSolverError) as captured:
        GroupStratifiedSplitter().solve(_origins()[:-1])
    assert captured.value.code == "SPLIT_INPUT_MISMATCH"


def test_source_uses_no_process_randomization_primitives() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "molhallulens"
        / "builders"
        / "splitter.py"
    ).read_text(encoding="utf-8")
    assert "import random" not in source
    assert "random." not in source
    assert "hash(" not in source.replace("sha256(", "")
