"""Immutable anomaly provenance, structural signatures, and capability policy."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .enums import EditingSubtask, OperationSubtype, _DomainStrEnum


class AnomalyProvenance(_DomainStrEnum):
    """Audited reason that an origin requires an explicit registry entry."""

    DELETE_WITH_REPLACEMENT = "delete_with_replacement"
    MAPPING_TRACE_DISAMBIGUATION = "mapping_trace_disambiguation"
    RETAINED_BOUNDARY_VALENCE_RELAXATION = "retained_boundary_valence_relaxation"
    AROMATIC_FRAGMENT_CAPPING = "aromatic_fragment_capping"
    MULTI_ANCHOR_RELOCATION = "multi_anchor_relocation"
    SUBSTITUTION_ANCHOR_STEREO_ASSIGNMENT = "substitution_anchor_stereo_assignment"


class OperatorCapability(_DomainStrEnum):
    """Coarse downstream operator capabilities controlled by anomaly policy."""

    REMOVE_ONLY_DELTA_RULE = "remove_only_delta_rule"
    STRUCTURAL_DELETION = "structural_deletion"
    REPLACEMENT_AWARE_DELETION = "replacement_aware_deletion"
    CLAIM_PERTURBATION = "claim_perturbation"
    TERMINAL_PERTURBATION = "terminal_perturbation"


@dataclass(frozen=True, slots=True)
class OperatorCapabilityPolicy:
    """A complete, immutable allow/forbid partition."""

    allowed: frozenset[OperatorCapability]
    forbidden: frozenset[OperatorCapability]

    def __post_init__(self) -> None:
        try:
            allowed = frozenset(self.allowed)
            forbidden = frozenset(self.forbidden)
        except TypeError as error:
            raise TypeError("capability policy fields must be iterable") from error
        if any(type(item) is not OperatorCapability for item in (*allowed, *forbidden)):
            raise TypeError(
                "capability policies must contain OperatorCapability values"
            )
        if allowed.intersection(forbidden):
            raise ValueError("allowed and forbidden capabilities must be disjoint")
        if allowed.union(forbidden) != frozenset(OperatorCapability):
            raise ValueError("capability policy must classify every capability")
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "forbidden", forbidden)

    def allows(self, capability: OperatorCapability) -> bool:
        if type(capability) is not OperatorCapability:
            raise TypeError("capability must be an OperatorCapability")
        return capability in self.allowed


@dataclass(frozen=True, slots=True)
class StructuralEditSignature:
    """Small auditable signature sufficient to guard registered shape anomalies."""

    removed_atom_count: int
    added_atomic_numbers: tuple[int, ...]
    broken_boundary_bond_count: int
    formed_boundary_bond_count: int
    remove_fragment_heavy_atoms: int | None
    add_fragment_heavy_atoms: int | None

    def __post_init__(self) -> None:
        if type(self.removed_atom_count) is not int or self.removed_atom_count < 0:
            raise ValueError("removed_atom_count must be non-negative")
        try:
            atomic_numbers = tuple(sorted(self.added_atomic_numbers))
        except TypeError as error:
            raise TypeError("added_atomic_numbers must be iterable") from error
        if any(type(item) is not int or item <= 0 for item in atomic_numbers):
            raise ValueError("added_atomic_numbers must be positive integers")
        for value, name in (
            (self.broken_boundary_bond_count, "broken_boundary_bond_count"),
            (self.formed_boundary_bond_count, "formed_boundary_bond_count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for value, name in (
            (self.remove_fragment_heavy_atoms, "remove_fragment_heavy_atoms"),
            (self.add_fragment_heavy_atoms, "add_fragment_heavy_atoms"),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be positive or None")
        if bool(self.removed_atom_count) != (
            self.remove_fragment_heavy_atoms is not None
        ):
            raise ValueError("remove fragment presence must match removed atoms")
        if bool(atomic_numbers) != (self.add_fragment_heavy_atoms is not None):
            raise ValueError("add fragment presence must match added atoms")
        if (
            self.remove_fragment_heavy_atoms is not None
            and self.remove_fragment_heavy_atoms != self.removed_atom_count
        ):
            raise ValueError("remove fragment size must equal removed_atom_count")
        if (
            self.add_fragment_heavy_atoms is not None
            and self.add_fragment_heavy_atoms != len(atomic_numbers)
        ):
            raise ValueError("add fragment size must equal added atom count")
        object.__setattr__(self, "added_atomic_numbers", atomic_numbers)

    @property
    def has_remove_fragment(self) -> bool:
        return self.remove_fragment_heavy_atoms is not None

    @property
    def has_add_fragment(self) -> bool:
        return self.add_fragment_heavy_atoms is not None

    @property
    def added_atom_count(self) -> int:
        return len(self.added_atomic_numbers)

    @property
    def heavy_atom_delta(self) -> int:
        return self.added_atom_count - self.removed_atom_count

    @property
    def has_bidirectional_boundary(self) -> bool:
        return (
            self.broken_boundary_bond_count > 0 and self.formed_boundary_bond_count > 0
        )


@dataclass(frozen=True, slots=True)
class AnomalyRegistryEntry:
    """One exact-ID registry rule; opaque origin IDs are never parsed or fuzzed."""

    anonymous_sample_id: str
    expected_subtask: EditingSubtask
    provenance: tuple[AnomalyProvenance, ...]
    operation_subtype_override: OperationSubtype | None = None
    expected_signature: StructuralEditSignature | None = None
    capability_policy: OperatorCapabilityPolicy | None = None
    provenance_task_id: str = "T013"

    def __post_init__(self) -> None:
        if type(self.anonymous_sample_id) is not str or not self.anonymous_sample_id:
            raise ValueError("anonymous_sample_id must be non-empty text")
        if type(self.expected_subtask) is not EditingSubtask:
            raise TypeError("expected_subtask must be EditingSubtask")
        try:
            provenance = tuple(
                sorted(set(self.provenance), key=lambda item: item.value)
            )
        except TypeError as error:
            raise TypeError("provenance must be iterable") from error
        if not provenance or any(
            type(item) is not AnomalyProvenance for item in provenance
        ):
            raise ValueError("provenance must contain AnomalyProvenance values")
        if (
            self.operation_subtype_override is not None
            and type(self.operation_subtype_override) is not OperationSubtype
        ):
            raise TypeError(
                "operation_subtype_override must be OperationSubtype or None"
            )
        if (
            self.expected_signature is not None
            and type(self.expected_signature) is not StructuralEditSignature
        ):
            raise TypeError(
                "expected_signature must be StructuralEditSignature or None"
            )
        if (
            self.capability_policy is not None
            and type(self.capability_policy) is not OperatorCapabilityPolicy
        ):
            raise TypeError(
                "capability_policy must be OperatorCapabilityPolicy or None"
            )
        if type(self.provenance_task_id) is not str or not self.provenance_task_id:
            raise ValueError("provenance_task_id must be non-empty text")
        if self.operation_subtype_override is not None:
            if self.expected_subtask is not EditingSubtask.DELETE:
                raise ValueError("only delete entries may override operation subtype")
            if (
                self.operation_subtype_override
                is not OperationSubtype.DELETE_WITH_REPLACEMENT
            ):
                raise ValueError(
                    "registered subtype override must be delete-with-replacement"
                )
            if self.expected_signature is None or self.capability_policy is None:
                raise ValueError(
                    "subtype overrides require signature and capability policy"
                )
        elif self.expected_signature is not None or self.capability_policy is not None:
            raise ValueError(
                "provenance-only entries cannot alter signature or capabilities"
            )
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class AnomalyClassification:
    """One classified graph truth plus its effective downstream restrictions."""

    anonymous_sample_id: str
    normalized_subtask: EditingSubtask
    operation_subtype: OperationSubtype
    registered: bool
    provenance: tuple[AnomalyProvenance, ...]
    structural_signature: StructuralEditSignature
    capability_policy: OperatorCapabilityPolicy

    def __post_init__(self) -> None:
        if type(self.anonymous_sample_id) is not str or not self.anonymous_sample_id:
            raise ValueError("anonymous_sample_id must be non-empty text")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if type(self.operation_subtype) is not OperationSubtype:
            raise TypeError("operation_subtype must be OperationSubtype")
        if type(self.registered) is not bool:
            raise TypeError("registered must be bool")
        try:
            provenance = tuple(
                sorted(set(self.provenance), key=lambda item: item.value)
            )
        except TypeError as error:
            raise TypeError("provenance must be iterable") from error
        if any(type(item) is not AnomalyProvenance for item in provenance):
            raise TypeError("provenance must contain AnomalyProvenance values")
        if self.registered != bool(provenance):
            raise ValueError("registered status must match provenance presence")
        if type(self.structural_signature) is not StructuralEditSignature:
            raise TypeError("structural_signature must be StructuralEditSignature")
        if type(self.capability_policy) is not OperatorCapabilityPolicy:
            raise TypeError("capability_policy must be OperatorCapabilityPolicy")
        object.__setattr__(self, "provenance", provenance)

    def allows(self, capability: OperatorCapability) -> bool:
        return self.capability_policy.allows(capability)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class AnomalyAuditReport:
    """Deterministic corpus classification report."""

    registry_id: str
    classifications: tuple[AnomalyClassification, ...]
    expected_registry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.registry_id) is not str or not self.registry_id:
            raise ValueError("registry_id must be non-empty text")
        try:
            classifications = tuple(
                sorted(self.classifications, key=lambda item: item.anonymous_sample_id)
            )
            expected_ids = tuple(sorted(self.expected_registry_ids))
        except TypeError as error:
            raise TypeError("audit report collections must be iterable") from error
        if any(type(item) is not AnomalyClassification for item in classifications):
            raise TypeError("classifications must contain AnomalyClassification values")
        ids = tuple(item.anonymous_sample_id for item in classifications)
        if len(set(ids)) != len(ids):
            raise ValueError("classification origin IDs must be unique")
        if any(type(item) is not str or not item for item in expected_ids):
            raise ValueError("expected_registry_ids must be non-empty strings")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected_registry_ids must be unique")
        object.__setattr__(self, "classifications", classifications)
        object.__setattr__(self, "expected_registry_ids", expected_ids)

    @property
    def total_count(self) -> int:
        return len(self.classifications)

    @property
    def registered_count(self) -> int:
        return sum(item.registered for item in self.classifications)

    @property
    def observed_registry_ids(self) -> tuple[str, ...]:
        return tuple(
            item.anonymous_sample_id for item in self.classifications if item.registered
        )

    @property
    def complete_registry(self) -> bool:
        return self.observed_registry_ids == self.expected_registry_ids

    @property
    def subtype_counts(self) -> tuple[tuple[OperationSubtype, int], ...]:
        counts = Counter(item.operation_subtype for item in self.classifications)
        return tuple((subtype, counts.get(subtype, 0)) for subtype in OperationSubtype)

    @property
    def provenance_counts(self) -> tuple[tuple[AnomalyProvenance, int], ...]:
        counts = Counter(
            provenance
            for item in self.classifications
            for provenance in item.provenance
        )
        return tuple(
            (provenance, counts.get(provenance, 0)) for provenance in AnomalyProvenance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "total_count": self.total_count,
            "registered_count": self.registered_count,
            "complete_registry": self.complete_registry,
            "expected_registry_ids": list(self.expected_registry_ids),
            "observed_registry_ids": list(self.observed_registry_ids),
            "subtype_counts": {
                subtype.value: count for subtype, count in self.subtype_counts
            },
            "provenance_counts": {
                provenance.value: count for provenance, count in self.provenance_counts
            },
            "classifications": [item.to_dict() for item in self.classifications],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_json_safe(item) for item in value]
        return sorted(items) if isinstance(value, (set, frozenset)) else items
    return value


__all__ = [
    "AnomalyAuditReport",
    "AnomalyClassification",
    "AnomalyProvenance",
    "AnomalyRegistryEntry",
    "OperatorCapability",
    "OperatorCapabilityPolicy",
    "StructuralEditSignature",
]
