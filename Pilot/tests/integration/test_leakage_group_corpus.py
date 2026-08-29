"""Real-corpus acceptance tests for T027 leakage-group assignments."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.leakage_groups import (
    KNOWN_GENERIC_MURCKO_GROUPS,
    LeakageGroupAssignments,
    LeakageReason,
    assign_leakage_groups,
    stable_leakage_group_id,
)
from molhallulens.builders.origin_audit import (
    KNOWN_DUPLICATE_SOURCE_GROUPS,
    OriginSplitAuditResult,
    audit_origin_split_features,
)
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.validation import (
    OriginValidationInput,
    validate_reference_origin_strict,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
ASSIGNMENTS_PATH = DATASET_ROOT / "reports" / "leakage_group_assignments.json"


@cache
def _items() -> tuple[OriginValidationInput, ...]:
    values = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        assert type(record) is JoinedInputRecord
        artifact = build_reference_dag(record)
        item = OriginValidationInput(
            record=record,
            artifact=artifact,
            edit_truth=derive_edit_truth(artifact),
        )
        validate_reference_origin_strict(item)
        values.append(item)
    return tuple(values)


@cache
def _audit() -> OriginSplitAuditResult:
    return audit_origin_split_features(_items())


@cache
def _assignments() -> LeakageGroupAssignments:
    return assign_leakage_groups(
        _audit().audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in _items()
        },
    )


def test_real_corpus_has_frozen_exact_and_generic_leakage_inventories() -> None:
    result = _assignments()
    assert len(result.index.assignments) == 150
    assert len(result.index.groups) == 142
    non_singletons = tuple(
        group for group in result.index.groups if len(group.anonymous_sample_ids) > 1
    )
    assert tuple(group.anonymous_sample_ids for group in non_singletons) == tuple(
        sorted(KNOWN_GENERIC_MURCKO_GROUPS)
    )
    assert len(non_singletons) == 7
    assert sum(len(group.anonymous_sample_ids) for group in non_singletons) == 15

    source_buckets: dict[str, list[str]] = defaultdict(list)
    gt_buckets: dict[str, list[str]] = defaultdict(list)
    for identity in result.index.identities:
        source_buckets[identity.canonical_source_sha256].append(
            identity.anonymous_sample_id
        )
        gt_buckets[identity.canonical_gt_sha256].append(identity.anonymous_sample_id)
    duplicate_sources = tuple(
        sorted(
            tuple(sorted(members))
            for members in source_buckets.values()
            if len(members) > 1
        )
    )
    assert duplicate_sources == tuple(sorted(KNOWN_DUPLICATE_SOURCE_GROUPS))
    assert sum(map(len, duplicate_sources)) == 7
    assert all(len(members) == 1 for members in gt_buckets.values())


def test_trigger_ledger_explains_every_union_and_group_id_recomputes() -> None:
    result = _assignments()
    reason_counts = Counter(
        evidence.reason
        for edge in result.index.trigger_edges
        for evidence in edge.evidence
    )
    assert reason_counts == Counter(
        {
            LeakageReason.CANONICAL_SOURCE: 5,
            LeakageReason.MURCKO_SCAFFOLD: 8,
            LeakageReason.GENERIC_MURCKO_SCAFFOLD: 9,
        }
    )
    assert LeakageReason.CANONICAL_GT not in reason_counts
    assert len(result.index.trigger_edges) == 9

    group_by_id = {group.leakage_group_id: group for group in result.index.groups}
    for group in result.index.groups:
        assert group.leakage_group_id == stable_leakage_group_id(
            result.dataset_version, group.anonymous_sample_ids
        )
    for edge in result.index.trigger_edges:
        left = next(
            item
            for item in result.index.assignments
            if item.identity.anonymous_sample_id == edge.left_origin_id
        )
        right = next(
            item
            for item in result.index.assignments
            if item.identity.anonymous_sample_id == edge.right_origin_id
        )
        assert left.leakage_group_id == right.leakage_group_id
        assert edge in group_by_id[left.leakage_group_id].trigger_edges


def test_artifact_is_bound_to_t026_and_byte_identical_under_input_reordering() -> None:
    result = _assignments()
    assert (
        result.source_audit_sha256
        == hashlib.sha256(_audit().audit.to_json_bytes()).hexdigest()
    )
    assert ASSIGNMENTS_PATH.read_bytes() == result.to_json_bytes()

    reverse_audit = audit_origin_split_features(reversed(_items())).audit
    reverse = assign_leakage_groups(
        reverse_audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in reversed(_items())
        },
    )
    assert reverse.to_json_bytes() == result.to_json_bytes()
