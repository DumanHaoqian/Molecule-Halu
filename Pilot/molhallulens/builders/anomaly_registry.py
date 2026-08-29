"""Explicit, fail-closed anomaly classification for graph-derived edit truth."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from molhallulens.domain.anomalies import (
    AnomalyAuditReport,
    AnomalyClassification,
    AnomalyProvenance,
    AnomalyRegistryEntry,
    OperatorCapability,
    OperatorCapabilityPolicy,
    StructuralEditSignature,
)
from molhallulens.domain.edit_truth import EditTruth
from molhallulens.domain.enums import (
    EditingSubtask,
    OperationSubtype,
    Severity,
    ValidationStage,
)
from molhallulens.domain.errors import ValidationIssue, ValidationReport


_VALIDATOR_ID = "molhallulens.anomaly_registry.v1"
_ALL_CAPABILITIES = frozenset(OperatorCapability)
_DELETE_CAPABILITIES = frozenset(
    {
        OperatorCapability.REMOVE_ONLY_DELTA_RULE,
        OperatorCapability.STRUCTURAL_DELETION,
    }
)
_CLAIM_AND_TERMINAL_CAPABILITIES = frozenset(
    {
        OperatorCapability.CLAIM_PERTURBATION,
        OperatorCapability.TERMINAL_PERTURBATION,
    }
)

_STANDARD_POLICY = OperatorCapabilityPolicy(
    allowed=_ALL_CAPABILITIES,
    forbidden=frozenset(),
)
_NON_DELETE_POLICY = OperatorCapabilityPolicy(
    allowed=_CLAIM_AND_TERMINAL_CAPABILITIES,
    forbidden=_DELETE_CAPABILITIES,
)
_DELETE_WITH_REPLACEMENT_POLICY = OperatorCapabilityPolicy(
    allowed=_CLAIM_AND_TERMINAL_CAPABILITIES,
    forbidden=_DELETE_CAPABILITIES,
)


class AnomalyRegistryError(RuntimeError):
    """A structured anomaly classification or corpus-audit failure."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("AnomalyRegistryError report must be a ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"anomaly registry validation failed ({codes})")


def structural_edit_signature(truth: EditTruth) -> StructuralEditSignature:
    """Project an EditTruth to the stable structural shape guarded by T014."""

    if type(truth) is not EditTruth:
        raise TypeError("structural_edit_signature requires an EditTruth")
    return StructuralEditSignature(
        removed_atom_count=len(truth.removed_atom_maps),
        added_atomic_numbers=tuple(atom.atomic_number for atom in truth.added_atoms),
        broken_boundary_bond_count=len(truth.broken_bonds),
        formed_boundary_bond_count=len(truth.formed_bonds),
        remove_fragment_heavy_atoms=(
            None
            if truth.remove_fragment is None
            else truth.remove_fragment.descriptors.heavy_atom_count
        ),
        add_fragment_heavy_atoms=(
            None
            if truth.add_fragment is None
            else truth.add_fragment.descriptors.heavy_atom_count
        ),
    )


def _signature_evidence(signature: StructuralEditSignature) -> dict[str, Any]:
    return {
        "removed_atom_count": signature.removed_atom_count,
        "added_atomic_numbers": signature.added_atomic_numbers,
        "broken_boundary_bond_count": signature.broken_boundary_bond_count,
        "formed_boundary_bond_count": signature.formed_boundary_bond_count,
        "remove_fragment_heavy_atoms": signature.remove_fragment_heavy_atoms,
        "add_fragment_heavy_atoms": signature.add_fragment_heavy_atoms,
    }


def _registry_error(
    code: str,
    message: str,
    *,
    node_ids: tuple[str, ...],
    evidence: Mapping[str, Any] | None = None,
) -> AnomalyRegistryError:
    return AnomalyRegistryError(
        ValidationReport(
            _VALIDATOR_ID,
            (
                ValidationIssue(
                    code=code,
                    severity=Severity.FATAL,
                    stage=ValidationStage.GRAPH_EDIT,
                    node_ids=node_ids,
                    message=message,
                    evidence={} if evidence is None else evidence,
                ),
            ),
        )
    )


def _is_add_shape(signature: StructuralEditSignature) -> bool:
    return (
        signature.removed_atom_count == 0
        and signature.added_atom_count > 0
        and not signature.has_remove_fragment
        and signature.has_add_fragment
        and signature.broken_boundary_bond_count == 0
        and signature.formed_boundary_bond_count > 0
    )


def _is_remove_only_shape(signature: StructuralEditSignature) -> bool:
    return (
        signature.removed_atom_count > 0
        and signature.added_atom_count == 0
        and signature.has_remove_fragment
        and not signature.has_add_fragment
        and signature.broken_boundary_bond_count > 0
        and signature.formed_boundary_bond_count == 0
    )


def _is_substitution_shape(signature: StructuralEditSignature) -> bool:
    atom_replacement = (
        signature.removed_atom_count > 0
        and signature.added_atom_count > 0
        and signature.has_remove_fragment
        and signature.has_add_fragment
        and signature.broken_boundary_bond_count > 0
        and signature.formed_boundary_bond_count > 0
    )
    bond_order_change = (
        signature.removed_atom_count == 0
        and signature.added_atom_count == 0
        and not signature.has_remove_fragment
        and not signature.has_add_fragment
        and signature.broken_boundary_bond_count > 0
        and signature.formed_boundary_bond_count > 0
    )
    return atom_replacement or bond_order_change


@dataclass(frozen=True, slots=True)
class AnomalyRegistry:
    """Immutable exact-ID registry with graph-shape guarded classification."""

    entries: tuple[AnomalyRegistryEntry, ...]
    registry_id: str = _VALIDATOR_ID
    _entries_by_id: Mapping[str, AnomalyRegistryEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.registry_id) is not str or not self.registry_id:
            raise ValueError("registry_id must be non-empty text")
        try:
            entries = tuple(
                sorted(self.entries, key=lambda entry: entry.anonymous_sample_id)
            )
        except TypeError as error:
            raise TypeError("entries must be iterable") from error
        if any(type(entry) is not AnomalyRegistryEntry for entry in entries):
            raise TypeError("entries must contain AnomalyRegistryEntry values")
        entry_ids = tuple(entry.anonymous_sample_id for entry in entries)
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("registry entries must use unique anonymous_sample_id values")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "_entries_by_id",
            MappingProxyType(
                {entry.anonymous_sample_id: entry for entry in entries}
            ),
        )

    @property
    def entries_by_id(self) -> Mapping[str, AnomalyRegistryEntry]:
        return self._entries_by_id

    @property
    def registered_ids(self) -> tuple[str, ...]:
        return tuple(entry.anonymous_sample_id for entry in self.entries)

    def entry_for(self, anonymous_sample_id: str) -> AnomalyRegistryEntry | None:
        if type(anonymous_sample_id) is not str:
            raise TypeError("anonymous_sample_id must be a string")
        if not anonymous_sample_id:
            raise ValueError("anonymous_sample_id cannot be empty")
        return self._entries_by_id.get(anonymous_sample_id)

    def classify(self, truth: EditTruth) -> AnomalyClassification:
        """Classify one truth, rejecting undeclared family-changing behavior."""

        if type(truth) is not EditTruth:
            raise TypeError("AnomalyRegistry.classify requires an EditTruth")
        signature = structural_edit_signature(truth)
        entry = self.entry_for(truth.anonymous_sample_id)

        if entry is not None and truth.normalized_subtask is not entry.expected_subtask:
            raise _registry_error(
                "ANOMALY_SUBTASK_MISMATCH",
                "registered anomaly appeared under a different normalized subtask",
                node_ids=(truth.anonymous_sample_id,),
                evidence={
                    "expected_subtask": entry.expected_subtask.value,
                    "actual_subtask": truth.normalized_subtask.value,
                },
            )
        if (
            entry is not None
            and entry.expected_signature is not None
            and signature != entry.expected_signature
        ):
            raise _registry_error(
                "ANOMALY_SIGNATURE_MISMATCH",
                "registered anomaly no longer matches its audited graph signature",
                node_ids=(truth.anonymous_sample_id,),
                evidence={
                    "expected": _signature_evidence(entry.expected_signature),
                    "actual": _signature_evidence(signature),
                },
            )

        if truth.normalized_subtask is EditingSubtask.ADD:
            if not _is_add_shape(signature):
                self._raise_family_mismatch(truth, signature)
            subtype = OperationSubtype.STANDARD
            policy = _NON_DELETE_POLICY
        elif truth.normalized_subtask is EditingSubtask.SUBSTITUTE:
            if not _is_substitution_shape(signature):
                self._raise_family_mismatch(truth, signature)
            subtype = OperationSubtype.STANDARD
            policy = _NON_DELETE_POLICY
        else:
            if entry is not None and entry.operation_subtype_override is not None:
                subtype = entry.operation_subtype_override
                policy = entry.capability_policy
                if policy is None:  # guarded by AnomalyRegistryEntry invariants
                    raise AssertionError("registered subtype override lost its policy")
            else:
                if (
                    signature.added_atom_count > 0
                    or signature.has_add_fragment
                    or signature.formed_boundary_bond_count > 0
                ):
                    raise _registry_error(
                        "UNREGISTERED_DELETE_WITH_ADDITION",
                        "delete truth contains addition/replacement edits without an exact registry entry",
                        node_ids=(truth.anonymous_sample_id,),
                        evidence=_signature_evidence(signature),
                    )
                if not _is_remove_only_shape(signature):
                    self._raise_family_mismatch(truth, signature)
                subtype = OperationSubtype.DEPROTECTION
                policy = _STANDARD_POLICY

        provenance = () if entry is None else entry.provenance
        return AnomalyClassification(
            anonymous_sample_id=truth.anonymous_sample_id,
            normalized_subtask=truth.normalized_subtask,
            operation_subtype=subtype,
            registered=entry is not None,
            provenance=provenance,
            structural_signature=signature,
            capability_policy=policy,
        )

    def _raise_family_mismatch(
        self,
        truth: EditTruth,
        signature: StructuralEditSignature,
    ) -> None:
        raise _registry_error(
            "EDIT_FAMILY_MISMATCH",
            "graph edit shape is inconsistent with its normalized subtask",
            node_ids=(truth.anonymous_sample_id,),
            evidence={
                "normalized_subtask": truth.normalized_subtask.value,
                **_signature_evidence(signature),
            },
        )

    def audit(
        self,
        truths: Iterable[EditTruth],
        *,
        require_complete_registry: bool = True,
    ) -> AnomalyAuditReport:
        """Classify a corpus and optionally require every registry entry to appear."""

        if type(require_complete_registry) is not bool:
            raise TypeError("require_complete_registry must be bool")
        try:
            truths = tuple(truths)
        except TypeError as error:
            raise TypeError("truths must be iterable") from error
        if any(type(truth) is not EditTruth for truth in truths):
            raise TypeError("truths must contain EditTruth values")
        ids = tuple(truth.anonymous_sample_id for truth in truths)
        duplicate_ids = tuple(sorted({item for item in ids if ids.count(item) > 1}))
        if duplicate_ids:
            raise _registry_error(
                "DUPLICATE_ANOMALY_ORIGIN",
                "anomaly audit input contains duplicate origin IDs",
                node_ids=duplicate_ids,
                evidence={"duplicate_count": len(duplicate_ids)},
            )

        classifications = tuple(
            self.classify(truth)
            for truth in sorted(truths, key=lambda item: item.anonymous_sample_id)
        )
        report = AnomalyAuditReport(
            registry_id=self.registry_id,
            classifications=classifications,
            expected_registry_ids=self.registered_ids,
        )
        if require_complete_registry and not report.complete_registry:
            missing = tuple(
                sorted(set(report.expected_registry_ids) - set(report.observed_registry_ids))
            )
            raise _registry_error(
                "MISSING_REGISTERED_ANOMALY",
                "strict anomaly audit did not observe every registered origin",
                node_ids=missing,
                evidence={
                    "expected_registry_count": len(report.expected_registry_ids),
                    "observed_registry_count": len(report.observed_registry_ids),
                },
            )
        return report


DEFAULT_ANOMALY_REGISTRY = AnomalyRegistry(
    entries=(
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.add_v2.0071",
            expected_subtask=EditingSubtask.ADD,
            provenance=(AnomalyProvenance.MAPPING_TRACE_DISAMBIGUATION,),
        ),
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.delete_v2.0081",
            expected_subtask=EditingSubtask.DELETE,
            provenance=(AnomalyProvenance.DELETE_WITH_REPLACEMENT,),
            operation_subtype_override=OperationSubtype.DELETE_WITH_REPLACEMENT,
            expected_signature=StructuralEditSignature(
                removed_atom_count=24,
                added_atomic_numbers=(6, 7),
                broken_boundary_bond_count=1,
                formed_boundary_bond_count=1,
                remove_fragment_heavy_atoms=24,
                add_fragment_heavy_atoms=2,
            ),
            capability_policy=_DELETE_WITH_REPLACEMENT_POLICY,
        ),
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.substitute_v2.0064",
            expected_subtask=EditingSubtask.SUBSTITUTE,
            provenance=(
                AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION,
            ),
        ),
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.substitute_v2.0123",
            expected_subtask=EditingSubtask.SUBSTITUTE,
            provenance=(AnomalyProvenance.AROMATIC_FRAGMENT_CAPPING,),
        ),
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.substitute_v2.0191",
            expected_subtask=EditingSubtask.SUBSTITUTE,
            provenance=(AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION,),
        ),
        AnomalyRegistryEntry(
            anonymous_sample_id="mol_edit.substitute_v2.0271",
            expected_subtask=EditingSubtask.SUBSTITUTE,
            provenance=(AnomalyProvenance.MULTI_ANCHOR_RELOCATION,),
        ),
    )
)


def classify_edit_truth(
    truth: EditTruth,
    *,
    registry: AnomalyRegistry = DEFAULT_ANOMALY_REGISTRY,
) -> AnomalyClassification:
    if type(registry) is not AnomalyRegistry:
        raise TypeError("registry must be an AnomalyRegistry")
    return registry.classify(truth)


def audit_anomaly_registry(
    truths: Iterable[EditTruth],
    *,
    registry: AnomalyRegistry = DEFAULT_ANOMALY_REGISTRY,
    require_complete_registry: bool = True,
) -> AnomalyAuditReport:
    if type(registry) is not AnomalyRegistry:
        raise TypeError("registry must be an AnomalyRegistry")
    return registry.audit(
        truths,
        require_complete_registry=require_complete_registry,
    )


__all__ = [
    "AnomalyRegistry",
    "AnomalyRegistryError",
    "DEFAULT_ANOMALY_REGISTRY",
    "audit_anomaly_registry",
    "classify_edit_truth",
    "structural_edit_signature",
]
