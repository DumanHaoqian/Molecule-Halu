"""Deterministic group-stratified splitting for the frozen 150-origin pilot.

The solver treats leakage groups as indivisible items.  It first proves or
disproves the frozen exact subtask matrix with exhaustive dynamic programming;
only an exact infeasibility proof permits the documented per-cell ``±1``
fallback.  Soft stratum balance is improved afterwards without changing any
hard count by swapping groups with identical subtask count vectors.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.core import EditingSubtask

from .leakage import (
    LeakageGroupAssignments,
    build_leakage_group_assignments_from_dataset,
)
from .origin_audit import (
    OriginSplitAudit,
    OriginSplitAuditRecord,
    build_origin_split_audit,
)

SPLIT_REPORT_FORMAT_VERSION = "group_stratified_split_balance_v1"
DEFAULT_SPLIT_REPORT_FILENAME = "split_balance_report.json"
FROZEN_DATASET_VERSION = "pilot_v1"
FROZEN_SPLIT_SEED = 8347206628578381721


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_SPLITS = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
_SUBTASKS = (
    EditingSubtask.ADD,
    EditingSubtask.DELETE,
    EditingSubtask.SUBSTITUTE,
)
_SPLIT_TOTALS = MappingProxyType(
    {
        SplitName.TRAIN: 100,
        SplitName.VALIDATION: 25,
        SplitName.TEST: 25,
    }
)
_EXACT_TARGETS = MappingProxyType(
    {
        EditingSubtask.ADD: MappingProxyType(
            {
                SplitName.TRAIN: 34,
                SplitName.VALIDATION: 8,
                SplitName.TEST: 8,
            }
        ),
        EditingSubtask.DELETE: MappingProxyType(
            {
                SplitName.TRAIN: 33,
                SplitName.VALIDATION: 9,
                SplitName.TEST: 8,
            }
        ),
        EditingSubtask.SUBSTITUTE: MappingProxyType(
            {
                SplitName.TRAIN: 33,
                SplitName.VALIDATION: 8,
                SplitName.TEST: 9,
            }
        ),
    }
)


class SplitSolverError(RuntimeError):
    """Structured fail-closed split input or feasibility error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("split error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("split error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("split error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": _json_value(self.evidence),
        }


def _json_value(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    raise TypeError(f"unsupported stable JSON value: {type(value).__qualname__}")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def derive_split_seed(dataset_version: str) -> int:
    """Derive the local split seed from the first SHA256 eight bytes."""

    if type(dataset_version) is not str or not dataset_version:
        raise ValueError("dataset_version must be non-empty text")
    return int.from_bytes(
        hashlib.sha256(dataset_version.encode("utf-8")).digest()[:8],
        "big",
    )


def _rank(seed: int, *parts: Any) -> str:
    payload = json.dumps(
        _json_value((seed, *parts)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitOrigin:
    """One canonical origin plus its T027 group and low-dimensional strata."""

    origin_id: str
    anonymous_sample_id: str
    leakage_group_id: str
    subtask: EditingSubtask
    strata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.leakage_group_id, "leakage_group_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        strata = tuple(sorted(self.strata))
        if any(
            type(feature) is not str
            or not feature
            or type(value) is not str
            or not value
            for feature, value in strata
        ):
            raise TypeError("strata must contain non-empty text pairs")
        if len(strata) != len(set(strata)):
            raise ValueError("strata pairs must be unique")
        object.__setattr__(self, "strata", strata)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    origin_id: str
    anonymous_sample_id: str
    leakage_group_id: str
    subtask: EditingSubtask
    split: SplitName

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.leakage_group_id, "leakage_group_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        if type(self.split) is not SplitName:
            raise TypeError("split must be SplitName")

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "anonymous_sample_id": self.anonymous_sample_id,
            "leakage_group_id": self.leakage_group_id,
            "subtask": self.subtask.value,
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class SplitCellBalance:
    split: SplitName
    subtask: EditingSubtask
    target: int
    actual: int
    deviation: int

    def __post_init__(self) -> None:
        if type(self.split) is not SplitName:
            raise TypeError("split must be SplitName")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        if any(
            type(value) is not int or value < 0 for value in (self.target, self.actual)
        ):
            raise ValueError("split balance counts must be non-negative integers")
        if self.deviation != self.actual - self.target:
            raise ValueError("split balance deviation must equal actual-target")
        if abs(self.deviation) > 1:
            raise ValueError("subtask quota deviation cannot exceed one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "subtask": self.subtask.value,
            "target": self.target,
            "actual": self.actual,
            "deviation": self.deviation,
        }


@dataclass(frozen=True, slots=True)
class StratumBalance:
    feature: str
    value: str
    global_count: int
    split_counts: tuple[tuple[SplitName, int], ...]
    scaled_l1_error: int

    def __post_init__(self) -> None:
        if type(self.feature) is not str or not self.feature:
            raise ValueError("stratum feature must be non-empty text")
        if type(self.value) is not str or not self.value:
            raise ValueError("stratum value must be non-empty text")
        counts = tuple(self.split_counts)
        if tuple(split for split, _ in counts) != _SPLITS:
            raise ValueError("stratum split counts must use frozen split order")
        if any(type(count) is not int or count < 0 for _, count in counts):
            raise ValueError("stratum counts must be non-negative integers")
        if sum(count for _, count in counts) != self.global_count:
            raise ValueError("stratum split counts must sum to global count")
        expected_error = sum(
            abs(count * 150 - self.global_count * _SPLIT_TOTALS[split])
            for split, count in counts
        )
        if self.scaled_l1_error != expected_error:
            raise ValueError("stratum scaled error is inconsistent")
        object.__setattr__(self, "split_counts", counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "global_count": self.global_count,
            "split_counts": {split.value: count for split, count in self.split_counts},
            "expected_count": {
                split.value: {
                    "numerator": self.global_count * _SPLIT_TOTALS[split],
                    "denominator": 150,
                }
                for split in _SPLITS
            },
            "scaled_l1_error": self.scaled_l1_error,
        }


@dataclass(frozen=True, slots=True)
class SplitFeasibilityProof:
    exact_target_reachable: bool
    selected_target_kind: str
    target_matrices_examined: int
    mixed_dp_states_explored: int
    pure_dp_states_explored: int
    proof_method: str = "exhaustive_mixed_vector_plus_per_subtask_partition_dp_v1"

    def __post_init__(self) -> None:
        if self.selected_target_kind not in {"exact", "within_one"}:
            raise ValueError("selected_target_kind must be exact or within_one")
        if self.exact_target_reachable != (self.selected_target_kind == "exact"):
            raise ValueError("exact reachability and selected target kind disagree")
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.target_matrices_examined,
                self.mixed_dp_states_explored,
                self.pure_dp_states_explored,
            )
        ):
            raise ValueError("feasibility proof counters must be positive integers")
        if (
            self.proof_method
            != "exhaustive_mixed_vector_plus_per_subtask_partition_dp_v1"
        ):
            raise ValueError("unknown feasibility proof method")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_target_reachable": self.exact_target_reachable,
            "selected_target_kind": self.selected_target_kind,
            "target_matrices_examined": self.target_matrices_examined,
            "mixed_dp_states_explored": self.mixed_dp_states_explored,
            "pure_dp_states_explored": self.pure_dp_states_explored,
            "proof_method": self.proof_method,
            "exact_infeasibility_proved": not self.exact_target_reachable,
        }


@dataclass(frozen=True, slots=True)
class SplitBalanceReport:
    dataset_version: str
    split_seed: int
    assignments: tuple[SplitAssignment, ...]
    cell_balances: tuple[SplitCellBalance, ...]
    strata_balances: tuple[StratumBalance, ...]
    feasibility_proof: SplitFeasibilityProof
    leakage_group_count: int
    soft_objective_before: int
    soft_objective_after: int
    local_search_improvements: int
    format_version: str = SPLIT_REPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        assignments = tuple(
            sorted(self.assignments, key=lambda item: item.anonymous_sample_id)
        )
        cells = tuple(
            sorted(
                self.cell_balances,
                key=lambda item: (
                    _SPLITS.index(item.split),
                    _SUBTASKS.index(item.subtask),
                ),
            )
        )
        strata = tuple(
            sorted(self.strata_balances, key=lambda item: (item.feature, item.value))
        )
        if self.format_version != SPLIT_REPORT_FORMAT_VERSION:
            raise ValueError("unknown split report format")
        if self.dataset_version != FROZEN_DATASET_VERSION:
            raise ValueError("split report requires pilot_v1")
        if self.split_seed != derive_split_seed(self.dataset_version):
            raise ValueError("split report seed differs from dataset-version SHA256")
        if (
            len(assignments) != 150
            or len({item.anonymous_sample_id for item in assignments}) != 150
        ):
            raise ValueError("split report requires 150 unique origins")
        expected_cell_keys = {
            (split, subtask) for split in _SPLITS for subtask in _SUBTASKS
        }
        if (
            len(cells) != 9
            or {(item.split, item.subtask) for item in cells} != expected_cell_keys
        ):
            raise ValueError("split report requires all nine subtask cells")
        if type(self.feasibility_proof) is not SplitFeasibilityProof:
            raise TypeError("feasibility_proof must be SplitFeasibilityProof")
        if (
            type(self.leakage_group_count) is not int
            or not 1 <= self.leakage_group_count <= 150
        ):
            raise ValueError("leakage_group_count must be in [1, 150]")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.soft_objective_before,
                self.soft_objective_after,
                self.local_search_improvements,
            )
        ):
            raise ValueError("soft objective fields must be non-negative integers")
        if self.soft_objective_after > self.soft_objective_before:
            raise ValueError("local balancing cannot worsen the soft objective")
        split_counts = Counter(item.split for item in assignments)
        if split_counts != Counter(_SPLIT_TOTALS):
            raise ValueError("split report must satisfy frozen 100/25/25 totals")
        actual_cells = Counter((item.split, item.subtask) for item in assignments)
        if any(
            cell.target != _EXACT_TARGETS[cell.subtask][cell.split]
            or cell.actual != actual_cells[cell.split, cell.subtask]
            for cell in cells
        ):
            raise ValueError("split report cell balances do not bind assignments")
        group_splits: dict[str, set[SplitName]] = defaultdict(set)
        for assignment in assignments:
            group_splits[assignment.leakage_group_id].add(assignment.split)
        if len(group_splits) != self.leakage_group_count or any(
            len(splits) != 1 for splits in group_splits.values()
        ):
            raise ValueError("split report must preserve atomic leakage groups")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "cell_balances", cells)
        object.__setattr__(self, "strata_balances", strata)

    def to_dict(self) -> dict[str, Any]:
        split_counts = Counter(item.split for item in self.assignments)
        group_splits: dict[str, set[SplitName]] = defaultdict(set)
        for item in self.assignments:
            group_splits[item.leakage_group_id].add(item.split)
        return {
            "format_version": self.format_version,
            "dataset_version": self.dataset_version,
            "seed": {
                "algorithm": "sha256_utf8_first_8_bytes_big_endian",
                "sha256": hashlib.sha256(
                    self.dataset_version.encode("utf-8")
                ).hexdigest(),
                "value": self.split_seed,
            },
            "algorithm": {
                "hard_solver": self.feasibility_proof.proof_method,
                "soft_optimizer": "deterministic_same_vector_pair_swap_v1",
                "uses_python_hash": False,
                "uses_random_module": False,
            },
            "hard_constraints": {
                "origin_counts": {
                    split.value: split_counts[split] for split in _SPLITS
                },
                "record_counts": {
                    split.value: split_counts[split] * 8 for split in _SPLITS
                },
                "leakage_group_count": self.leakage_group_count,
                "groups_split_across_partitions": sum(
                    len(splits) != 1 for splits in group_splits.values()
                ),
                "subtask_cells": [item.to_dict() for item in self.cell_balances],
            },
            "feasibility_proof": self.feasibility_proof.to_dict(),
            "soft_balance": {
                "objective_scale": "abs(actual*150-global*split_origins)",
                "objective_before": self.soft_objective_before,
                "objective_after": self.soft_objective_after,
                "local_search_improvements": self.local_search_improvements,
                "strata": [item.to_dict() for item in self.strata_balances],
            },
            "assignments": [item.to_dict() for item in self.assignments],
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class GroupStratifiedSplitResult:
    assignments: tuple[SplitAssignment, ...]
    report: SplitBalanceReport

    def __post_init__(self) -> None:
        assignments = tuple(
            sorted(self.assignments, key=lambda item: item.anonymous_sample_id)
        )
        if type(self.report) is not SplitBalanceReport:
            raise TypeError("report must be SplitBalanceReport")
        if assignments != self.report.assignments:
            raise ValueError("result assignments must equal report assignments")
        object.__setattr__(self, "assignments", assignments)

    def assignment_by_origin(self) -> Mapping[str, SplitAssignment]:
        return MappingProxyType(
            {item.anonymous_sample_id: item for item in self.assignments}
        )

    def to_dict(self) -> dict[str, Any]:
        return self.report.to_dict()

    def to_json_bytes(self) -> bytes:
        return self.report.to_json_bytes()


@dataclass(frozen=True, slots=True)
class _Group:
    group_id: str
    origins: tuple[SplitOrigin, ...]
    subtask_counts: tuple[int, int, int]
    strata: Counter[tuple[str, str]]


def _strata_for_audit_record(
    record: OriginSplitAuditRecord,
) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = [
        ("operation_subtype", record.operation_subtype),
        ("rxn_cls", record.rxn_cls),
        ("anchor_element", record.anchor_element),
        ("source_heavy_atom_quantile_bin", record.source_heavy_atom_quantile_bin),
        ("source_ring_quantile_bin", record.source_ring_quantile_bin),
        ("heavy_atom_delta_bin", record.heavy_atom_delta_bin),
        ("ring_delta_bin", record.ring_delta_bin),
        ("fragment_size_bin", record.fragment_size_bin),
        ("mol_complexity_quantile_bin", record.mol_complexity_quantile_bin),
        ("tanimoto_quantile_bin", record.tanimoto_quantile_bin),
    ]
    values.extend(
        (f"capability:{name}", "available" if available else "unavailable")
        for name, available in record.operator_availability.capability_flags
    )
    values.extend(
        (f"operator:{operator_id}", "available" if available else "unavailable")
        for operator_id, available in record.operator_availability.operator_flags
    )
    return tuple(values)


def split_origins_from_audit(
    audit: OriginSplitAudit,
    leakage_assignments: LeakageGroupAssignments,
) -> tuple[SplitOrigin, ...]:
    """Bind the authoritative T026 and T027 typed artifacts."""

    if type(audit) is not OriginSplitAudit:
        raise TypeError("audit must be OriginSplitAudit")
    if type(leakage_assignments) is not LeakageGroupAssignments:
        raise TypeError("leakage_assignments must be LeakageGroupAssignments")
    if leakage_assignments.dataset_version != audit.dataset_version:
        raise SplitSolverError(
            "SPLIT_INPUT_MISMATCH",
            "T026 and T027 dataset versions differ",
        )
    audit_sha256 = hashlib.sha256(audit.to_json_bytes()).hexdigest()
    if leakage_assignments.source_audit_sha256 != audit_sha256:
        raise SplitSolverError(
            "SPLIT_INPUT_MISMATCH",
            "T027 assignments are not bound to the supplied T026 audit",
            evidence={
                "expected_source_audit_sha256": audit_sha256,
                "observed_source_audit_sha256": (
                    leakage_assignments.source_audit_sha256
                ),
            },
        )
    leakage_group_ids = {
        item.identity.anonymous_sample_id: item.leakage_group_id
        for item in leakage_assignments.index.assignments
    }
    expected = {item.anonymous_sample_id for item in audit.records}
    if set(leakage_group_ids) != expected:
        raise SplitSolverError(
            "SPLIT_INPUT_MISMATCH",
            "T027 group mapping must cover the exact T026 origin inventory",
            evidence={
                "missing": tuple(sorted(expected - set(leakage_group_ids))),
                "unknown": tuple(sorted(set(leakage_group_ids) - expected)),
            },
        )
    return tuple(
        SplitOrigin(
            origin_id=record.origin_id,
            anonymous_sample_id=record.anonymous_sample_id,
            leakage_group_id=leakage_group_ids[record.anonymous_sample_id],
            subtask=record.subtask,
            strata=_strata_for_audit_record(record),
        )
        for record in audit.records
    )


def build_group_stratified_split(
    audit: OriginSplitAudit,
    leakage_assignments: LeakageGroupAssignments,
) -> GroupStratifiedSplitResult:
    """Build T028 from already validated T026 and T027 artifacts."""

    return GroupStratifiedSplitter(dataset_version=audit.dataset_version).solve(
        split_origins_from_audit(audit, leakage_assignments)
    )


def build_group_stratified_split_from_dataset(
    dataset_root: Path,
) -> GroupStratifiedSplitResult:
    """Build the frozen T028 split directly from the Dataset directory."""

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be pathlib.Path")
    audit = build_origin_split_audit(dataset_root).audit
    leakage_assignments = build_leakage_group_assignments_from_dataset(dataset_root)
    return build_group_stratified_split(audit, leakage_assignments)


def _target_key(
    target: Mapping[EditingSubtask, Mapping[SplitName, int]],
) -> tuple[int, ...]:
    return tuple(target[subtask][split] for subtask in _SUBTASKS for split in _SPLITS)


def _target_matrices() -> tuple[Mapping[EditingSubtask, Mapping[SplitName, int]], ...]:
    deviations = ((0, 0, 0),) + tuple(
        sorted(
            {
                values
                for values in itertools.product((-1, 0, 1), repeat=3)
                if sum(values) == 0 and sum(abs(value) for value in values) == 2
            }
        )
    )
    targets = []
    for per_subtask in itertools.product(deviations, repeat=3):
        candidate = {
            subtask: {
                split: _EXACT_TARGETS[subtask][split]
                + per_subtask[subtask_index][split_index]
                for split_index, split in enumerate(_SPLITS)
            }
            for subtask_index, subtask in enumerate(_SUBTASKS)
        }
        if all(
            sum(candidate[subtask][split] for subtask in _SUBTASKS)
            == _SPLIT_TOTALS[split]
            for split in _SPLITS
        ):
            targets.append(candidate)
    return tuple(
        sorted(
            targets,
            key=lambda item: (
                sum(
                    abs(item[subtask][split] - _EXACT_TARGETS[subtask][split])
                    for subtask in _SUBTASKS
                    for split in _SPLITS
                ),
                _target_key(item),
            ),
        )
    )


def _build_groups(origins: tuple[SplitOrigin, ...]) -> tuple[_Group, ...]:
    by_group: dict[str, list[SplitOrigin]] = defaultdict(list)
    for origin in origins:
        by_group[origin.leakage_group_id].append(origin)
    groups = []
    for group_id, members in sorted(by_group.items()):
        ordered = tuple(sorted(members, key=lambda item: item.anonymous_sample_id))
        counts = Counter(item.subtask for item in ordered)
        strata: Counter[tuple[str, str]] = Counter()
        for item in ordered:
            strata[("subtask", item.subtask.value)] += 1
            strata.update(item.strata)
        groups.append(
            _Group(
                group_id=group_id,
                origins=ordered,
                subtask_counts=tuple(counts[subtask] for subtask in _SUBTASKS),
                strata=strata,
            )
        )
    return tuple(groups)


def _split_preference(seed: int, group_id: str) -> tuple[SplitName, ...]:
    return tuple(
        sorted(
            _SPLITS,
            key=lambda split: (_rank(seed, "split", group_id, split), split.value),
        )
    )


def _pure_partition(
    groups: tuple[_Group, ...],
    *,
    train_target: int,
    validation_target: int,
    test_target: int,
    seed: int,
    subtask: EditingSubtask,
) -> tuple[dict[str, SplitName] | None, int]:
    ordered = tuple(
        sorted(
            groups,
            key=lambda item: (
                _rank(seed, "pure", subtask, item.group_id),
                item.group_id,
            ),
        )
    )
    states: dict[tuple[int, int], tuple[tuple[str, SplitName], ...]] = {(0, 0): ()}
    explored = 1
    processed = 0
    subtask_index = _SUBTASKS.index(subtask)
    for group in ordered:
        size = group.subtask_counts[subtask_index]
        processed += size
        next_states: dict[tuple[int, int], tuple[tuple[str, SplitName], ...]] = {}
        for state, path in sorted(states.items()):
            for split in _split_preference(seed, group.group_id):
                train = state[0] + (size if split is SplitName.TRAIN else 0)
                validation = state[1] + (size if split is SplitName.VALIDATION else 0)
                test = processed - train - validation
                if (
                    train > train_target
                    or validation > validation_target
                    or test > test_target
                ):
                    continue
                next_states.setdefault(
                    (train, validation), (*path, (group.group_id, split))
                )
        states = next_states
        explored += len(states)
        if not states:
            return None, explored
    path = states.get((train_target, validation_target))
    return (None if path is None else dict(path)), explored


def _solve_target(
    groups: tuple[_Group, ...],
    target: Mapping[EditingSubtask, Mapping[SplitName, int]],
    *,
    seed: int,
) -> tuple[dict[str, SplitName] | None, int, int]:
    mixed = tuple(
        group
        for group in groups
        if sum(count > 0 for count in group.subtask_counts) > 1
    )
    pure = tuple(group for group in groups if group not in mixed)
    ordered_mixed = tuple(
        sorted(
            mixed,
            key=lambda item: (_rank(seed, "mixed", item.group_id), item.group_id),
        )
    )
    zero_state = (0, 0, 0, 0, 0, 0)
    states: dict[tuple[int, ...], tuple[tuple[str, SplitName], ...]] = {zero_state: ()}
    mixed_explored = 1
    processed = [0, 0, 0]
    for group in ordered_mixed:
        processed = [
            processed[index] + group.subtask_counts[index] for index in range(3)
        ]
        next_states: dict[tuple[int, ...], tuple[tuple[str, SplitName], ...]] = {}
        for state, path in sorted(states.items()):
            for split in _split_preference(seed, group.group_id):
                values = list(state)
                if split is SplitName.TRAIN:
                    for index, count in enumerate(group.subtask_counts):
                        values[index] += count
                elif split is SplitName.VALIDATION:
                    for index, count in enumerate(group.subtask_counts):
                        values[3 + index] += count
                valid = True
                for index, subtask in enumerate(_SUBTASKS):
                    test_count = processed[index] - values[index] - values[3 + index]
                    if (
                        values[index] > target[subtask][SplitName.TRAIN]
                        or values[3 + index] > target[subtask][SplitName.VALIDATION]
                        or test_count > target[subtask][SplitName.TEST]
                    ):
                        valid = False
                        break
                if valid:
                    next_states.setdefault(
                        tuple(values), (*path, (group.group_id, split))
                    )
        states = next_states
        mixed_explored += len(states)
        if not states:
            return None, mixed_explored, 1

    pure_by_subtask = {
        subtask: tuple(
            group
            for group in pure
            if group.subtask_counts[_SUBTASKS.index(subtask)] > 0
        )
        for subtask in _SUBTASKS
    }
    pure_cache: dict[
        tuple[EditingSubtask, int, int, int],
        tuple[dict[str, SplitName] | None, int],
    ] = {}
    pure_explored = 0
    ordered_states = tuple(
        sorted(
            states.items(),
            key=lambda item: (_rank(seed, "mixed-state", item[0]), item[0]),
        )
    )
    for state, mixed_path in ordered_states:
        assignments = dict(mixed_path)
        feasible = True
        for index, subtask in enumerate(_SUBTASKS):
            residual = (
                target[subtask][SplitName.TRAIN] - state[index],
                target[subtask][SplitName.VALIDATION] - state[3 + index],
                target[subtask][SplitName.TEST]
                - (
                    sum(group.subtask_counts[index] for group in mixed)
                    - state[index]
                    - state[3 + index]
                ),
            )
            if min(residual) < 0:
                feasible = False
                break
            cache_key = (subtask, *residual)
            if cache_key not in pure_cache:
                pure_cache[cache_key] = _pure_partition(
                    pure_by_subtask[subtask],
                    train_target=residual[0],
                    validation_target=residual[1],
                    test_target=residual[2],
                    seed=seed,
                    subtask=subtask,
                )
                pure_explored += pure_cache[cache_key][1]
            partition, _ = pure_cache[cache_key]
            if partition is None:
                feasible = False
                break
            assignments.update(partition)
        if feasible and len(assignments) == len(groups):
            return assignments, mixed_explored, max(pure_explored, 1)
    return None, mixed_explored, max(pure_explored, 1)


def _objective(
    groups: tuple[_Group, ...], assignments: Mapping[str, SplitName]
) -> tuple[int, Counter[tuple[str, str]], dict[SplitName, Counter[tuple[str, str]]]]:
    global_counts: Counter[tuple[str, str]] = Counter()
    split_counts = {split: Counter() for split in _SPLITS}
    for group in groups:
        global_counts.update(group.strata)
        split_counts[assignments[group.group_id]].update(group.strata)
    objective = sum(
        abs(split_counts[split][key] * 150 - count * _SPLIT_TOTALS[split])
        for key, count in global_counts.items()
        for split in _SPLITS
    )
    return objective, global_counts, split_counts


def _improve_soft_balance(
    groups: tuple[_Group, ...],
    assignments: dict[str, SplitName],
    *,
    seed: int,
) -> tuple[int, int, int]:
    before, global_counts, split_counts = _objective(groups, assignments)
    ordered_pairs = tuple(
        sorted(
            (
                (left, right)
                for index, left in enumerate(groups)
                for right in groups[index + 1 :]
                if left.subtask_counts == right.subtask_counts
            ),
            key=lambda pair: (
                _rank(seed, "swap", pair[0].group_id, pair[1].group_id),
                pair[0].group_id,
                pair[1].group_id,
            ),
        )
    )
    improvements = 0
    for _ in range(8):
        changed = False
        for left, right in ordered_pairs:
            left_split = assignments[left.group_id]
            right_split = assignments[right.group_id]
            if left_split is right_split:
                continue
            keys = set(left.strata) | set(right.strata)
            old_error = sum(
                abs(
                    split_counts[split][key] * 150
                    - global_counts[key] * _SPLIT_TOTALS[split]
                )
                for key in keys
                for split in (left_split, right_split)
            )
            new_error = 0
            for key in keys:
                left_new = (
                    split_counts[left_split][key] - left.strata[key] + right.strata[key]
                )
                right_new = (
                    split_counts[right_split][key]
                    - right.strata[key]
                    + left.strata[key]
                )
                new_error += abs(
                    left_new * 150 - global_counts[key] * _SPLIT_TOTALS[left_split]
                )
                new_error += abs(
                    right_new * 150 - global_counts[key] * _SPLIT_TOTALS[right_split]
                )
            if new_error >= old_error:
                continue
            split_counts[left_split].subtract(left.strata)
            split_counts[left_split].update(right.strata)
            split_counts[right_split].subtract(right.strata)
            split_counts[right_split].update(left.strata)
            assignments[left.group_id], assignments[right.group_id] = (
                right_split,
                left_split,
            )
            improvements += 1
            changed = True
        if not changed:
            break
    after = sum(
        abs(split_counts[split][key] * 150 - count * _SPLIT_TOTALS[split])
        for key, count in global_counts.items()
        for split in _SPLITS
    )
    return before, after, improvements


class GroupStratifiedSplitter:
    """Solve the frozen group-level pilot split without external solvers."""

    __slots__ = ("dataset_version", "split_seed")

    def __init__(self, *, dataset_version: str = FROZEN_DATASET_VERSION) -> None:
        if dataset_version != FROZEN_DATASET_VERSION:
            raise ValueError("T028 supports only the frozen pilot_v1 dataset")
        self.dataset_version = dataset_version
        self.split_seed = derive_split_seed(dataset_version)
        if self.split_seed != FROZEN_SPLIT_SEED:
            raise ValueError("pilot_v1 split seed drifted")

    def solve(self, origins: Iterable[SplitOrigin]) -> GroupStratifiedSplitResult:
        values = tuple(origins)
        if any(type(item) is not SplitOrigin for item in values):
            raise TypeError("origins must contain SplitOrigin values")
        ordered = tuple(sorted(values, key=lambda item: item.anonymous_sample_id))
        if (
            len(ordered) != 150
            or len({item.anonymous_sample_id for item in ordered}) != 150
        ):
            raise SplitSolverError(
                "SPLIT_INPUT_MISMATCH",
                "split solver requires exactly 150 unique anonymous origins",
            )
        if Counter(item.subtask for item in ordered) != Counter(
            {subtask: 50 for subtask in _SUBTASKS}
        ):
            raise SplitSolverError(
                "SPLIT_INPUT_MISMATCH",
                "split solver requires exactly 50 origins per editing subtask",
            )
        groups = _build_groups(ordered)
        targets = _target_matrices()
        exact_target = targets[0]
        exact_assignment, exact_mixed, exact_pure = _solve_target(
            groups,
            exact_target,
            seed=self.split_seed,
        )
        examined = 1
        mixed_explored = exact_mixed
        pure_explored = exact_pure
        selected_target = exact_target
        group_assignments = exact_assignment
        if group_assignments is None:
            for target in targets[1:]:
                examined += 1
                candidate, mixed_states, pure_states = _solve_target(
                    groups,
                    target,
                    seed=self.split_seed,
                )
                mixed_explored += mixed_states
                pure_explored += pure_states
                if candidate is not None:
                    selected_target = target
                    group_assignments = candidate
                    break
        if group_assignments is None:
            raise SplitSolverError(
                "SPLIT_INFEASIBLE",
                "no group-preserving 100/25/25 split exists within per-cell ±1",
                evidence={
                    "target_matrices_examined": examined,
                    "mixed_dp_states_explored": mixed_explored,
                    "pure_dp_states_explored": pure_explored,
                },
            )
        exact_reachable = exact_assignment is not None
        before, after, improvements = _improve_soft_balance(
            groups,
            group_assignments,
            seed=self.split_seed,
        )
        assignments = tuple(
            SplitAssignment(
                origin_id=origin.origin_id,
                anonymous_sample_id=origin.anonymous_sample_id,
                leakage_group_id=origin.leakage_group_id,
                subtask=origin.subtask,
                split=group_assignments[origin.leakage_group_id],
            )
            for origin in ordered
        )
        actual_cells = Counter((item.subtask, item.split) for item in assignments)
        cell_balances = tuple(
            SplitCellBalance(
                split=split,
                subtask=subtask,
                target=_EXACT_TARGETS[subtask][split],
                actual=actual_cells[subtask, split],
                deviation=(
                    actual_cells[subtask, split] - _EXACT_TARGETS[subtask][split]
                ),
            )
            for split in _SPLITS
            for subtask in _SUBTASKS
        )
        _, global_counts, split_counts = _objective(groups, group_assignments)
        strata = tuple(
            StratumBalance(
                feature=feature,
                value=value,
                global_count=count,
                split_counts=tuple(
                    (split, split_counts[split][(feature, value)]) for split in _SPLITS
                ),
                scaled_l1_error=sum(
                    abs(
                        split_counts[split][(feature, value)] * 150
                        - count * _SPLIT_TOTALS[split]
                    )
                    for split in _SPLITS
                ),
            )
            for (feature, value), count in sorted(global_counts.items())
        )
        proof = SplitFeasibilityProof(
            exact_target_reachable=exact_reachable,
            selected_target_kind="exact" if exact_reachable else "within_one",
            target_matrices_examined=examined,
            mixed_dp_states_explored=max(mixed_explored, 1),
            pure_dp_states_explored=max(pure_explored, 1),
        )
        report = SplitBalanceReport(
            dataset_version=self.dataset_version,
            split_seed=self.split_seed,
            assignments=assignments,
            cell_balances=cell_balances,
            strata_balances=strata,
            feasibility_proof=proof,
            leakage_group_count=len(groups),
            soft_objective_before=before,
            soft_objective_after=after,
            local_search_improvements=improvements,
        )
        result = GroupStratifiedSplitResult(assignments=assignments, report=report)
        self._validate_result(result, selected_target)
        return result

    @staticmethod
    def _validate_result(
        result: GroupStratifiedSplitResult,
        selected_target: Mapping[EditingSubtask, Mapping[SplitName, int]],
    ) -> None:
        split_counts = Counter(item.split for item in result.assignments)
        if split_counts != Counter(_SPLIT_TOTALS):
            raise SplitSolverError(
                "SPLIT_INTERNAL_ERROR", "solver output violates frozen split totals"
            )
        actual = Counter((item.subtask, item.split) for item in result.assignments)
        if any(
            actual[subtask, split] != selected_target[subtask][split]
            for subtask in _SUBTASKS
            for split in _SPLITS
        ):
            raise SplitSolverError(
                "SPLIT_INTERNAL_ERROR", "solver output differs from selected matrix"
            )
        groups: dict[str, set[SplitName]] = defaultdict(set)
        for item in result.assignments:
            groups[item.leakage_group_id].add(item.split)
        if any(len(splits) != 1 for splits in groups.values()):
            raise SplitSolverError(
                "SPLIT_INTERNAL_ERROR", "solver split an atomic leakage group"
            )


def write_split_balance_report(
    result: GroupStratifiedSplitResult,
    *,
    output_path: Path,
) -> None:
    if type(result) is not GroupStratifiedSplitResult:
        raise TypeError("result must be GroupStratifiedSplitResult")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.to_json_bytes())


__all__ = [
    "DEFAULT_SPLIT_REPORT_FILENAME",
    "FROZEN_DATASET_VERSION",
    "FROZEN_SPLIT_SEED",
    "SPLIT_REPORT_FORMAT_VERSION",
    "GroupStratifiedSplitResult",
    "GroupStratifiedSplitter",
    "SplitAssignment",
    "SplitBalanceReport",
    "SplitCellBalance",
    "SplitFeasibilityProof",
    "SplitName",
    "SplitOrigin",
    "SplitSolverError",
    "StratumBalance",
    "build_group_stratified_split",
    "build_group_stratified_split_from_dataset",
    "derive_split_seed",
    "split_origins_from_audit",
    "write_split_balance_report",
]
