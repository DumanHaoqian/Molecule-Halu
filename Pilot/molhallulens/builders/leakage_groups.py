"""Deterministic union-find leakage groups for the frozen Pilot origins."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.chemistry import (
    FragmentPolicy,
    generic_murcko_scaffold_smiles,
)
from molhallulens.config import ConfigBundle

from .edit_truth import derive_edit_truth
from .origin_audit import (
    OriginSplitAudit,
    audit_origin_split_features,
)
from .reference_dag import build_reference_dag

if TYPE_CHECKING:
    from molhallulens.validation.reference import OriginValidationInput

LEAKAGE_ASSIGNMENTS_FORMAT_VERSION = "leakage_group_assignments_v1"
DEFAULT_LEAKAGE_ASSIGNMENTS_FILENAME = "leakage_group_assignments.json"
KNOWN_GENERIC_MURCKO_GROUPS = (
    ("mol_edit.add_v2.0140", "mol_edit.delete_v2.0046"),
    ("mol_edit.add_v2.0193", "mol_edit.substitute_v2.0101"),
    ("mol_edit.add_v2.0235", "mol_edit.add_v2.0279"),
    ("mol_edit.delete_v2.0185", "mol_edit.delete_v2.0186"),
    (
        "mol_edit.substitute_v2.0134",
        "mol_edit.substitute_v2.0136",
        "mol_edit.substitute_v2.0283",
    ),
    ("mol_edit.substitute_v2.0165", "mol_edit.substitute_v2.0185"),
    ("mol_edit.substitute_v2.0248", "mol_edit.substitute_v2.0270"),
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def stable_leakage_group_id(
    dataset_version: str,
    anonymous_sample_ids: Iterable[str],
) -> str:
    """Hash ``dataset_version NUL sorted-member-ids`` without Python hash()."""

    if (
        type(dataset_version) is not str
        or not dataset_version
        or "\0" in dataset_version
    ):
        raise ValueError("dataset_version must be non-empty NUL-free text")
    try:
        members = tuple(sorted(anonymous_sample_ids))
    except TypeError as error:
        raise TypeError("anonymous_sample_ids must be iterable") from error
    if (
        not members
        or any(type(item) is not str or not item or "\0" in item for item in members)
        or len(set(members)) != len(members)
    ):
        raise ValueError("group members must be unique non-empty NUL-free strings")
    payload = "\0".join((dataset_version, *members)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LeakageReason(StrEnum):
    CANONICAL_SOURCE = "canonical_source"
    CANONICAL_GT = "canonical_gt"
    MURCKO_SCAFFOLD = "murcko_scaffold"
    GENERIC_MURCKO_SCAFFOLD = "generic_murcko_scaffold"


@dataclass(frozen=True, slots=True)
class LeakageIdentity:
    """Four exact identities that may induce one leakage edge."""

    anonymous_sample_id: str
    canonical_source_sha256: str
    canonical_gt_sha256: str
    murcko_scaffold_sha256: str | None
    generic_murcko_scaffold_sha256: str | None

    def __post_init__(self) -> None:
        if (
            type(self.anonymous_sample_id) is not str
            or not self.anonymous_sample_id
            or "\0" in self.anonymous_sample_id
        ):
            raise ValueError("anonymous_sample_id must be non-empty NUL-free text")
        if not _is_sha256(self.canonical_source_sha256) or not _is_sha256(
            self.canonical_gt_sha256
        ):
            raise ValueError("canonical source and GT identities must be SHA256")
        for value, name in (
            (self.murcko_scaffold_sha256, "murcko_scaffold_sha256"),
            (
                self.generic_murcko_scaffold_sha256,
                "generic_murcko_scaffold_sha256",
            ),
        ):
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be SHA256 or None")

    def identity_for(self, reason: LeakageReason) -> str | None:
        if type(reason) is not LeakageReason:
            raise TypeError("reason must be LeakageReason")
        return {
            LeakageReason.CANONICAL_SOURCE: self.canonical_source_sha256,
            LeakageReason.CANONICAL_GT: self.canonical_gt_sha256,
            LeakageReason.MURCKO_SCAFFOLD: self.murcko_scaffold_sha256,
            LeakageReason.GENERIC_MURCKO_SCAFFOLD: (
                self.generic_murcko_scaffold_sha256
            ),
        }[reason]


@dataclass(frozen=True, slots=True)
class LeakageEdgeEvidence:
    reason: LeakageReason
    identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.reason) is not LeakageReason:
            raise TypeError("edge reason must be LeakageReason")
        if not _is_sha256(self.identity_sha256):
            raise ValueError("edge identity must be SHA256")

    def to_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason.value,
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class LeakageTriggerEdge:
    left_origin_id: str
    right_origin_id: str
    evidence: tuple[LeakageEdgeEvidence, ...]

    def __post_init__(self) -> None:
        evidence = tuple(sorted(self.evidence, key=lambda item: item.reason.value))
        if not (
            type(self.left_origin_id) is str
            and type(self.right_origin_id) is str
            and self.left_origin_id < self.right_origin_id
        ):
            raise ValueError("edge endpoints must be distinct canonical ordered IDs")
        if not evidence or any(
            type(item) is not LeakageEdgeEvidence for item in evidence
        ):
            raise ValueError("trigger edge requires typed evidence")
        if len({item.reason for item in evidence}) != len(evidence):
            raise ValueError("an edge may carry each reason at most once")
        object.__setattr__(self, "evidence", evidence)

    @property
    def reasons(self) -> tuple[LeakageReason, ...]:
        return tuple(item.reason for item in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_origin_id": self.left_origin_id,
            "right_origin_id": self.right_origin_id,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class LeakageGroup:
    leakage_group_id: str
    anonymous_sample_ids: tuple[str, ...]
    trigger_edges: tuple[LeakageTriggerEdge, ...]

    def __post_init__(self) -> None:
        members = tuple(sorted(self.anonymous_sample_ids))
        edges = tuple(
            sorted(
                self.trigger_edges,
                key=lambda item: (item.left_origin_id, item.right_origin_id),
            )
        )
        if not _is_sha256(self.leakage_group_id):
            raise ValueError("leakage_group_id must be full SHA256")
        if not members or len(set(members)) != len(members):
            raise ValueError("leakage group requires unique members")
        member_set = set(members)
        if any(
            edge.left_origin_id not in member_set
            or edge.right_origin_id not in member_set
            for edge in edges
        ):
            raise ValueError("group trigger edge escapes its member set")
        if len(members) == 1 and edges:
            raise ValueError("singleton groups cannot contain trigger edges")
        if len(members) > 1 and not edges:
            raise ValueError("non-singleton groups require trigger edges")
        object.__setattr__(self, "anonymous_sample_ids", members)
        object.__setattr__(self, "trigger_edges", edges)

    @property
    def reasons(self) -> tuple[LeakageReason, ...]:
        return tuple(
            sorted(
                {
                    evidence.reason
                    for edge in self.trigger_edges
                    for evidence in edge.evidence
                },
                key=lambda item: item.value,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "leakage_group_id": self.leakage_group_id,
            "anonymous_sample_ids": list(self.anonymous_sample_ids),
            "origin_count": len(self.anonymous_sample_ids),
            "singleton": len(self.anonymous_sample_ids) == 1,
            "reasons": [item.value for item in self.reasons],
            "trigger_edges": [item.to_dict() for item in self.trigger_edges],
        }


@dataclass(frozen=True, slots=True)
class OriginLeakageAssignment:
    identity: LeakageIdentity
    leakage_group_id: str
    leakage_group_size: int
    leakage_reasons: tuple[LeakageReason, ...]

    def __post_init__(self) -> None:
        reasons = tuple(sorted(set(self.leakage_reasons), key=lambda item: item.value))
        if type(self.identity) is not LeakageIdentity:
            raise TypeError("assignment identity must be LeakageIdentity")
        if not _is_sha256(self.leakage_group_id):
            raise ValueError("assignment group ID must be SHA256")
        if type(self.leakage_group_size) is not int or self.leakage_group_size <= 0:
            raise ValueError("assignment group size must be positive")
        if any(type(reason) is not LeakageReason for reason in reasons):
            raise TypeError("assignment reasons must be LeakageReason values")
        if self.leakage_group_size == 1 and reasons:
            raise ValueError("singleton assignment cannot carry leakage reasons")
        object.__setattr__(self, "leakage_reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anonymous_sample_id": self.identity.anonymous_sample_id,
            "leakage_group_id": self.leakage_group_id,
            "leakage_group_size": self.leakage_group_size,
            "leakage_reasons": [item.value for item in self.leakage_reasons],
            "canonical_source_sha256": self.identity.canonical_source_sha256,
            "canonical_gt_sha256": self.identity.canonical_gt_sha256,
            "murcko_scaffold_sha256": self.identity.murcko_scaffold_sha256,
            "generic_murcko_scaffold_sha256": (
                self.identity.generic_murcko_scaffold_sha256
            ),
        }


class _UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self._parent = {member: member for member in sorted(members)}

    def find(self, member: str) -> str:
        parent = self._parent[member]
        if parent != member:
            self._parent[member] = self.find(parent)
        return self._parent[member]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first

    def components(self) -> tuple[tuple[str, ...], ...]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for member in sorted(self._parent):
            grouped[self.find(member)].append(member)
        return tuple(sorted(tuple(members) for members in grouped.values()))


def _derive_trigger_edges(
    identities: tuple[LeakageIdentity, ...],
) -> tuple[LeakageTriggerEdge, ...]:
    accumulated: dict[tuple[str, str], dict[LeakageReason, str]] = defaultdict(dict)
    for reason in LeakageReason:
        buckets: dict[str, list[str]] = defaultdict(list)
        for identity in identities:
            value = identity.identity_for(reason)
            # None is absence, not an identity shared by acyclic molecules.
            if value is not None:
                buckets[value].append(identity.anonymous_sample_id)
        for digest, members in sorted(buckets.items()):
            for left, right in combinations(sorted(members), 2):
                accumulated[(left, right)][reason] = digest
    return tuple(
        LeakageTriggerEdge(
            left_origin_id=left,
            right_origin_id=right,
            evidence=tuple(
                LeakageEdgeEvidence(reason=reason, identity_sha256=digest)
                for reason, digest in sorted(
                    evidence.items(), key=lambda item: item[0].value
                )
            ),
        )
        for (left, right), evidence in sorted(accumulated.items())
    )


@dataclass(frozen=True, slots=True, init=False)
class LeakageGroupIndex:
    """A fully derived, order-independent union-find result."""

    dataset_version: str
    identities: tuple[LeakageIdentity, ...]
    trigger_edges: tuple[LeakageTriggerEdge, ...]
    groups: tuple[LeakageGroup, ...]
    assignments: tuple[OriginLeakageAssignment, ...]

    def __init__(
        self,
        *,
        dataset_version: str,
        identities: Iterable[LeakageIdentity],
    ) -> None:
        if (
            type(dataset_version) is not str
            or not dataset_version
            or "\0" in dataset_version
        ):
            raise ValueError("dataset_version must be non-empty NUL-free text")
        try:
            ordered = tuple(
                sorted(identities, key=lambda item: item.anonymous_sample_id)
            )
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "identities must be iterable LeakageIdentity values"
            ) from error
        if not ordered or any(type(item) is not LeakageIdentity for item in ordered):
            raise ValueError("identities must contain LeakageIdentity values")
        ids = tuple(item.anonymous_sample_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("leakage identities must use unique origin IDs")

        edges = _derive_trigger_edges(ordered)
        union_find = _UnionFind(ids)
        # Sorting here makes the contract independent of caller/union order.
        for edge in sorted(
            edges, key=lambda item: (item.right_origin_id, item.left_origin_id)
        ):
            union_find.union(edge.left_origin_id, edge.right_origin_id)
        components = union_find.components()
        groups: list[LeakageGroup] = []
        for component in components:
            component_set = set(component)
            component_edges = tuple(
                edge
                for edge in edges
                if edge.left_origin_id in component_set
                and edge.right_origin_id in component_set
            )
            groups.append(
                LeakageGroup(
                    leakage_group_id=stable_leakage_group_id(
                        dataset_version, component
                    ),
                    anonymous_sample_ids=component,
                    trigger_edges=component_edges,
                )
            )
        groups_tuple = tuple(sorted(groups, key=lambda item: item.anonymous_sample_ids))
        group_by_member = {
            member: group
            for group in groups_tuple
            for member in group.anonymous_sample_ids
        }
        identity_by_id = {item.anonymous_sample_id: item for item in ordered}
        assignments = tuple(
            OriginLeakageAssignment(
                identity=identity_by_id[origin_id],
                leakage_group_id=group_by_member[origin_id].leakage_group_id,
                leakage_group_size=len(group_by_member[origin_id].anonymous_sample_ids),
                leakage_reasons=group_by_member[origin_id].reasons,
            )
            for origin_id in ids
        )
        if {
            member for group in groups_tuple for member in group.anonymous_sample_ids
        } != set(ids):
            raise ValueError("leakage groups must partition all origins")
        object.__setattr__(self, "dataset_version", dataset_version)
        object.__setattr__(self, "identities", ordered)
        object.__setattr__(self, "trigger_edges", edges)
        object.__setattr__(self, "groups", groups_tuple)
        object.__setattr__(self, "assignments", assignments)


@dataclass(frozen=True, slots=True, init=False)
class LeakageGroupAssignments:
    """The frozen T027 150-origin assignment artifact."""

    source_audit_sha256: str
    index: LeakageGroupIndex
    format_version: str

    def __init__(
        self,
        *,
        dataset_version: str,
        source_audit_sha256: str,
        identities: Iterable[LeakageIdentity],
    ) -> None:
        if not _is_sha256(source_audit_sha256):
            raise ValueError("source_audit_sha256 must be full SHA256")
        index = LeakageGroupIndex(
            dataset_version=dataset_version,
            identities=identities,
        )
        if len(index.identities) != 150:
            raise ValueError("T027 assignments require exactly 150 origins")
        if len({item.canonical_source_sha256 for item in index.identities}) != 146:
            raise ValueError("T027 canonical-source inventory changed")
        if len({item.canonical_gt_sha256 for item in index.identities}) != 150:
            raise ValueError("T027 canonical-GT inventory changed")
        standard = tuple(
            item.murcko_scaffold_sha256
            for item in index.identities
            if item.murcko_scaffold_sha256 is not None
        )
        generic = tuple(
            item.generic_murcko_scaffold_sha256
            for item in index.identities
            if item.generic_murcko_scaffold_sha256 is not None
        )
        if len(set(standard)) != 143 or len(set(generic)) != 142:
            raise ValueError("T027 Murcko identity inventories changed")
        non_singletons = tuple(
            group for group in index.groups if len(group.anonymous_sample_ids) > 1
        )
        expected_groups = tuple(sorted(KNOWN_GENERIC_MURCKO_GROUPS))
        observed_groups = tuple(group.anonymous_sample_ids for group in non_singletons)
        if observed_groups != expected_groups:
            raise ValueError("frozen generic-Murcko leakage groups changed")
        if (
            len(non_singletons) != 7
            or sum(len(group.anonymous_sample_ids) for group in non_singletons) != 15
        ):
            raise ValueError("T027 requires seven non-singleton groups / 15 origins")
        if len(index.groups) != 142:
            raise ValueError("T027 requires 142 total leakage groups")
        object.__setattr__(self, "source_audit_sha256", source_audit_sha256)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "format_version", LEAKAGE_ASSIGNMENTS_FORMAT_VERSION)

    @property
    def dataset_version(self) -> str:
        return self.index.dataset_version

    def to_dict(self) -> dict[str, Any]:
        non_singletons = tuple(
            group for group in self.index.groups if len(group.anonymous_sample_ids) > 1
        )
        reason_edge_counts = Counter(
            evidence.reason.value
            for edge in self.index.trigger_edges
            for evidence in edge.evidence
        )
        return {
            "format_version": self.format_version,
            "dataset_version": self.dataset_version,
            "source_audit_sha256": self.source_audit_sha256,
            "algorithms": {
                "canonical_source": "t026_canonical_source_sha256",
                "canonical_gt": "t026_canonical_gt_sha256",
                "murcko_scaffold": "t026_rdkit_bemis_murcko_largest_heavy_v1",
                "generic_murcko_scaffold": (
                    "rdkit_make_scaffold_generic_largest_heavy_v1"
                ),
                "none_scaffold": "absent_not_a_shared_identity",
                "union": "exact_identity_union_find_transitive_closure_v1",
                "group_id": (
                    "sha256_utf8(dataset_version+NUL+sorted_origin_ids_NUL_joined)"
                ),
            },
            "summary": {
                "origin_count": len(self.index.identities),
                "leakage_group_count": len(self.index.groups),
                "singleton_group_count": len(self.index.groups) - len(non_singletons),
                "non_singleton_group_count": len(non_singletons),
                "non_singleton_origin_count": sum(
                    len(group.anonymous_sample_ids) for group in non_singletons
                ),
                "trigger_edge_count": len(self.index.trigger_edges),
                "trigger_evidence_counts_by_reason": dict(
                    sorted(reason_edge_counts.items())
                ),
                "unique_canonical_sources": len(
                    {item.canonical_source_sha256 for item in self.index.identities}
                ),
                "unique_canonical_gt": len(
                    {item.canonical_gt_sha256 for item in self.index.identities}
                ),
                "unique_murcko_scaffolds": len(
                    {
                        item.murcko_scaffold_sha256
                        for item in self.index.identities
                        if item.murcko_scaffold_sha256 is not None
                    }
                ),
                "unique_generic_murcko_scaffolds": len(
                    {
                        item.generic_murcko_scaffold_sha256
                        for item in self.index.identities
                        if item.generic_murcko_scaffold_sha256 is not None
                    }
                ),
            },
            "trigger_edges": [edge.to_dict() for edge in self.index.trigger_edges],
            "groups": [group.to_dict() for group in self.index.groups],
            "origins": [assignment.to_dict() for assignment in self.index.assignments],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def assign_leakage_groups(
    audit: OriginSplitAudit,
    *,
    canonical_source_smiles_by_id: Mapping[str, str],
) -> LeakageGroupAssignments:
    """Bind T026 identities to independently derived generic Murcko identities."""

    if type(audit) is not OriginSplitAudit:
        raise TypeError("audit must be an OriginSplitAudit")
    if not isinstance(canonical_source_smiles_by_id, Mapping):
        raise TypeError("canonical_source_smiles_by_id must be a mapping")
    audit_ids = {record.anonymous_sample_id for record in audit.records}
    if set(canonical_source_smiles_by_id) != audit_ids or any(
        type(key) is not str or type(value) is not str or not value
        for key, value in canonical_source_smiles_by_id.items()
    ):
        raise ValueError("canonical source mapping must exactly cover the T026 audit")

    identities = []
    for record in audit.records:
        source = canonical_source_smiles_by_id[record.anonymous_sample_id]
        if _sha256_text(source) != record.canonical_source_sha256:
            raise ValueError("canonical source plaintext does not match T026 hash")
        generic = generic_murcko_scaffold_smiles(
            source,
            fragment_policy=FragmentPolicy.LARGEST_HEAVY,
        )
        identities.append(
            LeakageIdentity(
                anonymous_sample_id=record.anonymous_sample_id,
                canonical_source_sha256=record.canonical_source_sha256,
                canonical_gt_sha256=record.canonical_gt_sha256,
                murcko_scaffold_sha256=(
                    record.scaffold_sha256 if record.scaffold_present else None
                ),
                generic_murcko_scaffold_sha256=(
                    None if generic is None else _sha256_text(generic)
                ),
            )
        )
    return LeakageGroupAssignments(
        dataset_version=audit.dataset_version,
        source_audit_sha256=hashlib.sha256(audit.to_json_bytes()).hexdigest(),
        identities=identities,
    )


def build_leakage_group_assignments(
    items: Iterable[OriginValidationInput],
    *,
    config: ConfigBundle | None = None,
) -> LeakageGroupAssignments:
    """Build T026, then derive T027 from the same validated typed corpus."""

    from molhallulens.validation.reference import OriginValidationInput

    values = tuple(items)
    if any(type(item) is not OriginValidationInput for item in values):
        raise TypeError("items must contain OriginValidationInput values")
    audit_result = audit_origin_split_features(values, config=config)
    canonical_sources = {
        item.edit_truth.anonymous_sample_id: item.edit_truth.canonical_source_smiles
        for item in values
    }
    if len(canonical_sources) != len(values):
        raise ValueError("validated inputs contain duplicate origin IDs")
    return assign_leakage_groups(
        audit_result.audit,
        canonical_source_smiles_by_id=canonical_sources,
    )


def build_leakage_group_assignments_from_dataset(
    dataset_root: Path,
) -> LeakageGroupAssignments:
    """Load the frozen Dataset directory and build all T027 assignments."""

    from molhallulens.validation.reference import (
        OriginValidationInput,
        validate_reference_origin_strict,
    )

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be pathlib.Path")
    items = []
    for record in ChemCoTMolEditAdapter().load(dataset_root):
        artifact = build_reference_dag(record)
        item = OriginValidationInput(
            record=record,
            artifact=artifact,
            edit_truth=derive_edit_truth(artifact),
        )
        validate_reference_origin_strict(item)
        items.append(item)
    return build_leakage_group_assignments(items)


def write_leakage_group_assignments(
    assignments: LeakageGroupAssignments,
    output_path: Path,
) -> None:
    if type(assignments) is not LeakageGroupAssignments:
        raise TypeError("assignments must be LeakageGroupAssignments")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(assignments.to_json_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = (
        args.output
        or args.dataset_root / "reports" / DEFAULT_LEAKAGE_ASSIGNMENTS_FILENAME
    )
    write_leakage_group_assignments(
        build_leakage_group_assignments_from_dataset(args.dataset_root),
        output,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_LEAKAGE_ASSIGNMENTS_FILENAME",
    "KNOWN_GENERIC_MURCKO_GROUPS",
    "LEAKAGE_ASSIGNMENTS_FORMAT_VERSION",
    "LeakageEdgeEvidence",
    "LeakageGroup",
    "LeakageGroupAssignments",
    "LeakageGroupIndex",
    "LeakageIdentity",
    "LeakageReason",
    "LeakageTriggerEdge",
    "OriginLeakageAssignment",
    "assign_leakage_groups",
    "build_leakage_group_assignments",
    "build_leakage_group_assignments_from_dataset",
    "stable_leakage_group_id",
    "write_leakage_group_assignments",
]
