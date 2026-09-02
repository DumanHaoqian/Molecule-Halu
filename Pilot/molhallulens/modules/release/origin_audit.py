"""Deterministic split-feature audit for the 150 molecule-editing origins."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

import rdkit
from rdkit import Chem

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.infrastructure.chemistry import FragmentPolicy, murcko_scaffold_smiles
from molhallulens.config import ConfigBundle, load_config_bundle
from molhallulens.core import EditingSubtask, OperatorCapability
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    DeletionPerturbator,
    PerturbatorRegistry,
    SubstitutionPerturbator,
)

from molhallulens.modules.reference.anomaly_registry import classify_edit_truth
from molhallulens.modules.reference.truth import derive_edit_truth
from molhallulens.modules.reference.builder import build_reference_dag

if TYPE_CHECKING:
    from molhallulens.infrastructure.validation.reference import OriginValidationInput

AUDIT_FORMAT_VERSION = "origin_split_audit_v1"
FEATURE_DISTRIBUTION_FORMAT_VERSION = "origin_split_feature_distribution_v1"
DEFAULT_AUDIT_FILENAME = "origin_split_audit.json"
_FROZEN_SPLIT_CAPABILITIES = tuple(
    capability
    for capability in OperatorCapability
    if capability is not OperatorCapability.REPLACEMENT_AWARE_DELETION
)
DEFAULT_DISTRIBUTION_FILENAME = "origin_split_feature_distribution.json"
NO_SCAFFOLD_IDENTITY = "molhallulens:no_murcko_scaffold:v1"
KNOWN_DUPLICATE_SOURCE_GROUPS = (
    ("mol_edit.add_v2.0235", "mol_edit.add_v2.0279"),
    (
        "mol_edit.substitute_v2.0134",
        "mol_edit.substitute_v2.0136",
        "mol_edit.substitute_v2.0283",
    ),
    ("mol_edit.substitute_v2.0248", "mol_edit.substitute_v2.0270"),
)
KNOWN_DUPLICATE_SCAFFOLD_GROUPS = (
    ("mol_edit.add_v2.0140", "mol_edit.delete_v2.0046"),
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
_PERTURBATOR_TYPES = (
    AdditionPerturbator,
    DeletionPerturbator,
    SubstitutionPerturbator,
)
_QUANTILE_FEATURES = (
    "source_heavy_atom_count",
    "source_ring_count",
    "mol_complexity",
    "tanimoto",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _count_pairs(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(values).items()))


@dataclass(frozen=True, slots=True)
class QuantileThresholds:
    """Tie-preserving nearest-rank quartile thresholds."""

    feature: str
    population_count: int
    q25: float
    q50: float
    q75: float

    def __post_init__(self) -> None:
        if self.feature not in _QUANTILE_FEATURES:
            raise ValueError("unknown quantile feature")
        if type(self.population_count) is not int or self.population_count <= 0:
            raise ValueError("population_count must be positive")
        if any(
            type(value) is not float or not isfinite(value)
            for value in (self.q25, self.q50, self.q75)
        ):
            raise ValueError("quantile thresholds must be finite floats")
        if not self.q25 <= self.q50 <= self.q75:
            raise ValueError("quantile thresholds must be ordered")

    @classmethod
    def from_values(
        cls, feature: str, values: Iterable[int | float]
    ) -> QuantileThresholds:
        ordered = tuple(sorted(float(value) for value in values))
        if not ordered or any(not isfinite(value) for value in ordered):
            raise ValueError("quantile populations must contain finite values")

        def nearest_rank(fraction: float) -> float:
            return ordered[ceil(fraction * len(ordered)) - 1]

        return cls(
            feature=feature,
            population_count=len(ordered),
            q25=nearest_rank(0.25),
            q50=nearest_rank(0.50),
            q75=nearest_rank(0.75),
        )

    def bin_for(self, value: float) -> str:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("quantile value must be finite")
        if numeric <= self.q25:
            return "q1"
        if numeric <= self.q50:
            return "q2"
        if numeric <= self.q75:
            return "q3"
        return "q4"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "population_count": self.population_count,
            "q25": self.q25,
            "q50": self.q50,
            "q75": self.q75,
        }


@dataclass(frozen=True, slots=True)
class OperatorAvailability:
    """T014 capability policy projected through T017 static registrations."""

    capability_flags: tuple[tuple[str, bool], ...]
    registered_operator_ids: tuple[str, ...]
    eligible_operator_ids: tuple[str, ...]
    ineligible_operator_reasons: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        flags = tuple(sorted(self.capability_flags))
        registered = tuple(sorted(self.registered_operator_ids))
        eligible = tuple(sorted(self.eligible_operator_ids))
        reasons = tuple(
            sorted(
                (operator_id, tuple(sorted(items)))
                for operator_id, items in self.ineligible_operator_reasons
            )
        )
        expected_capabilities = tuple(
            sorted(item.value for item in _FROZEN_SPLIT_CAPABILITIES)
        )
        if tuple(name for name, _ in flags) != expected_capabilities:
            raise ValueError("capability_flags must classify every OperatorCapability")
        if any(type(value) is not bool for _, value in flags):
            raise TypeError("capability flags must be bool")
        if len(set(registered)) != len(registered) or not set(eligible) <= set(
            registered
        ):
            raise ValueError("eligible operators must be unique registered operators")
        if {operator_id for operator_id, _ in reasons} != set(registered) - set(
            eligible
        ):
            raise ValueError("every ineligible operator must have reasons")
        if any(not items for _, items in reasons):
            raise ValueError("ineligible operator reasons cannot be empty")
        object.__setattr__(self, "capability_flags", flags)
        object.__setattr__(self, "registered_operator_ids", registered)
        object.__setattr__(self, "eligible_operator_ids", eligible)
        object.__setattr__(self, "ineligible_operator_reasons", reasons)

    @property
    def operator_flags(self) -> tuple[tuple[str, bool], ...]:
        eligible = set(self.eligible_operator_ids)
        return tuple(
            (operator_id, operator_id in eligible)
            for operator_id in self.registered_operator_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_flags": dict(self.capability_flags),
            "operator_flags": dict(self.operator_flags),
            "registered_operator_ids": list(self.registered_operator_ids),
            "eligible_operator_ids": list(self.eligible_operator_ids),
            "ineligible_operator_reasons": {
                operator_id: list(reasons)
                for operator_id, reasons in self.ineligible_operator_reasons
            },
        }


@dataclass(frozen=True, slots=True)
class OriginSplitAuditRecord:
    origin_id: str
    anonymous_sample_id: str
    subtask: EditingSubtask
    operation_subtype: str
    rxn_cls: str
    anchor_element: str
    anchor_elements: tuple[str, ...]
    canonical_source_sha256: str
    canonical_gt_sha256: str
    scaffold_identity: str | None
    scaffold_sha256: str
    scaffold_present: bool
    source_heavy_atom_count: int
    source_heavy_atom_quantile_bin: str
    source_ring_count: int
    source_ring_quantile_bin: str
    heavy_atom_delta: int
    heavy_atom_delta_bin: str
    ring_delta: int
    ring_delta_bin: str
    remove_fragment_heavy_atom_count: int
    add_fragment_heavy_atom_count: int
    fragment_size: int
    fragment_size_bin: str
    mol_complexity: float | None
    mol_complexity_source: str
    mol_complexity_quantile_bin: str
    tanimoto: float | None
    tanimoto_source: str
    tanimoto_quantile_bin: str
    operator_availability: OperatorAvailability

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.operation_subtype, "operation_subtype"),
            (self.rxn_cls, "rxn_cls"),
            (self.anchor_element, "anchor_element"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        elements = tuple(sorted(set(self.anchor_elements)))
        if not elements or self.anchor_element != "+".join(elements):
            raise ValueError(
                "anchor_element must be the sorted anchor element identity"
            )
        for digest in (
            self.canonical_source_sha256,
            self.canonical_gt_sha256,
            self.scaffold_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("molecule identities must be lowercase SHA256")
        if type(self.scaffold_present) is not bool:
            raise TypeError("scaffold_present must be bool")
        if self.scaffold_present != (self.scaffold_identity is not None):
            raise ValueError("scaffold presence must match scaffold identity")
        if self.scaffold_identity is not None and (
            type(self.scaffold_identity) is not str or not self.scaffold_identity
        ):
            raise ValueError("scaffold identity must be non-empty text or None")
        expected_scaffold_hash = _sha256_text(
            self.scaffold_identity
            if self.scaffold_identity is not None
            else NO_SCAFFOLD_IDENTITY
        )
        if self.scaffold_sha256 != expected_scaffold_hash:
            raise ValueError("scaffold hash must bind its canonical identity")
        for value, name in (
            (self.source_heavy_atom_count, "source_heavy_atom_count"),
            (self.source_ring_count, "source_ring_count"),
            (self.remove_fragment_heavy_atom_count, "remove_fragment_heavy_atom_count"),
            (self.add_fragment_heavy_atom_count, "add_fragment_heavy_atom_count"),
            (self.fragment_size, "fragment_size"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.fragment_size != max(
            self.remove_fragment_heavy_atom_count, self.add_fragment_heavy_atom_count
        ):
            raise ValueError("fragment_size must be max(remove_size, add_size)")
        for value, source, bin_name, name in (
            (
                self.mol_complexity,
                self.mol_complexity_source,
                self.mol_complexity_quantile_bin,
                "mol_complexity",
            ),
            (
                self.tanimoto,
                self.tanimoto_source,
                self.tanimoto_quantile_bin,
                "tanimoto",
            ),
        ):
            if value is None:
                if source != "missing_in_raw_benchmark_data" or bin_name != "missing":
                    raise ValueError(f"missing {name} must be explicit")
            elif (
                type(value) is not float
                or not isfinite(value)
                or source != "raw_benchmark_data"
                or bin_name not in {"q1", "q2", "q3", "q4"}
            ):
                raise ValueError(
                    f"present {name} must be finite raw data with a quantile bin"
                )
        if type(self.operator_availability) is not OperatorAvailability:
            raise TypeError("operator_availability must be OperatorAvailability")
        object.__setattr__(self, "anchor_elements", elements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "anonymous_sample_id": self.anonymous_sample_id,
            "subtask": self.subtask.value,
            "operation_subtype": self.operation_subtype,
            "rxn_cls": self.rxn_cls,
            "anchor_element": self.anchor_element,
            "anchor_elements": list(self.anchor_elements),
            "canonical_source_sha256": self.canonical_source_sha256,
            "canonical_gt_sha256": self.canonical_gt_sha256,
            "scaffold_identity": self.scaffold_identity,
            "scaffold_sha256": self.scaffold_sha256,
            "scaffold_present": self.scaffold_present,
            "source_heavy_atom_count": self.source_heavy_atom_count,
            "source_heavy_atom_quantile_bin": self.source_heavy_atom_quantile_bin,
            "source_ring_count": self.source_ring_count,
            "source_ring_quantile_bin": self.source_ring_quantile_bin,
            "heavy_atom_delta": self.heavy_atom_delta,
            "heavy_atom_delta_bin": self.heavy_atom_delta_bin,
            "ring_delta": self.ring_delta,
            "ring_delta_bin": self.ring_delta_bin,
            "remove_fragment_heavy_atom_count": self.remove_fragment_heavy_atom_count,
            "add_fragment_heavy_atom_count": self.add_fragment_heavy_atom_count,
            "fragment_size": self.fragment_size,
            "fragment_size_bin": self.fragment_size_bin,
            "mol_complexity": self.mol_complexity,
            "mol_complexity_source": self.mol_complexity_source,
            "mol_complexity_quantile_bin": self.mol_complexity_quantile_bin,
            "tanimoto": self.tanimoto,
            "tanimoto_source": self.tanimoto_source,
            "tanimoto_quantile_bin": self.tanimoto_quantile_bin,
            "operator_availability": self.operator_availability.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DuplicateSourceGroup:
    group_id: str
    canonical_source_sha256: str
    anonymous_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = tuple(sorted(self.anonymous_sample_ids))
        if self.group_id != f"source:{self.canonical_source_sha256[:16]}":
            raise ValueError("duplicate group ID must derive from its source hash")
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("duplicate source groups require unique members")
        object.__setattr__(self, "anonymous_sample_ids", ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "canonical_source_sha256": self.canonical_source_sha256,
            "anonymous_sample_ids": list(self.anonymous_sample_ids),
            "origin_count": len(self.anonymous_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class DuplicateScaffoldGroup:
    group_id: str
    scaffold_sha256: str
    anonymous_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = tuple(sorted(self.anonymous_sample_ids))
        if self.group_id != f"scaffold:{self.scaffold_sha256[:16]}":
            raise ValueError("duplicate scaffold group ID must derive from its hash")
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("duplicate scaffold groups require unique members")
        object.__setattr__(self, "anonymous_sample_ids", ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "scaffold_sha256": self.scaffold_sha256,
            "anonymous_sample_ids": list(self.anonymous_sample_ids),
            "origin_count": len(self.anonymous_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class OriginSplitAudit:
    dataset_version: str
    rdkit_version: str
    quantile_thresholds: tuple[QuantileThresholds, ...]
    duplicate_source_groups: tuple[DuplicateSourceGroup, ...]
    duplicate_scaffold_groups: tuple[DuplicateScaffoldGroup, ...]
    records: tuple[OriginSplitAuditRecord, ...]
    t015_attempted: int
    t015_passed: int
    format_version: str = AUDIT_FORMAT_VERSION

    def __post_init__(self) -> None:
        records = tuple(sorted(self.records, key=lambda item: item.anonymous_sample_id))
        groups = tuple(
            sorted(
                self.duplicate_source_groups, key=lambda item: item.anonymous_sample_ids
            )
        )
        scaffold_groups = tuple(
            sorted(
                self.duplicate_scaffold_groups,
                key=lambda item: item.anonymous_sample_ids,
            )
        )
        thresholds = tuple(
            sorted(self.quantile_thresholds, key=lambda item: item.feature)
        )
        if (
            self.format_version != AUDIT_FORMAT_VERSION
            or self.dataset_version != "pilot_v1"
        ):
            raise ValueError("unknown origin split audit identity")
        if self.rdkit_version != rdkit.__version__:
            raise ValueError("audit RDKit version must match the executing runtime")
        if (
            len(records) != 150
            or len({item.anonymous_sample_id for item in records}) != 150
        ):
            raise ValueError("origin split audit requires 150 unique origins")
        if Counter(item.subtask for item in records) != Counter(
            {subtask: 50 for subtask in EditingSubtask}
        ):
            raise ValueError("origin split audit requires 50 origins per subtask")
        if self.t015_attempted != 150 or self.t015_passed != 150:
            raise ValueError("all 150 origins must pass T015 before split audit")
        if tuple(item.feature for item in thresholds) != tuple(
            sorted(_QUANTILE_FEATURES)
        ):
            raise ValueError("all frozen quantile features require thresholds")
        observed_groups = tuple(group.anonymous_sample_ids for group in groups)
        if observed_groups != tuple(sorted(KNOWN_DUPLICATE_SOURCE_GROUPS)):
            raise ValueError("known duplicate-source groups changed")
        if sum(len(group.anonymous_sample_ids) for group in groups) != 7:
            raise ValueError("duplicate-source groups must contain seven origins")
        observed_scaffold_groups = tuple(
            group.anonymous_sample_ids for group in scaffold_groups
        )
        if observed_scaffold_groups != tuple(sorted(KNOWN_DUPLICATE_SCAFFOLD_GROUPS)):
            raise ValueError("known duplicate-scaffold groups changed")
        if sum(len(group.anonymous_sample_ids) for group in scaffold_groups) != 13:
            raise ValueError("duplicate-scaffold groups must contain thirteen origins")
        if len({item.canonical_source_sha256 for item in records}) != 146:
            raise ValueError("canonical source inventory must contain 146 identities")
        if len({item.canonical_gt_sha256 for item in records}) != 150:
            raise ValueError("canonical GT inventory must contain 150 identities")
        if len({item.scaffold_sha256 for item in records}) != 143:
            raise ValueError("Murcko scaffold inventory must contain 143 identities")
        eligible_counts = Counter(
            (item.subtask, len(item.operator_availability.eligible_operator_ids))
            for item in records
        )
        if eligible_counts != Counter(
            {
                (EditingSubtask.ADD, 11): 50,
                (EditingSubtask.DELETE, 12): 49,
                (EditingSubtask.DELETE, 4): 1,
                (EditingSubtask.SUBSTITUTE, 12): 50,
            }
        ):
            raise ValueError("static operator eligibility inventory changed")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "duplicate_source_groups", groups)
        object.__setattr__(self, "duplicate_scaffold_groups", scaffold_groups)
        object.__setattr__(self, "quantile_thresholds", thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset_version": self.dataset_version,
            "algorithm": {
                "canonical_identity": "t013_edit_truth_canonical_isomeric_smiles",
                "hash": "sha256_utf8",
                "scaffold": "rdkit_bemis_murcko_largest_heavy_v1",
                "quantiles": "global_nearest_rank_quartiles_tie_preserving_v1",
                "fragment_size": "max_remove_add_fragment_heavy_atoms_v1",
                "operator_availability": "t014_capability_policy_plus_t017_static_registry_v1",
            },
            "rdkit_version": self.rdkit_version,
            "validation": {
                "pipeline": "molhallulens.reference_validation.v1",
                "attempted": self.t015_attempted,
                "passed": self.t015_passed,
                "all_pass": True,
            },
            "quantile_thresholds": {
                item.feature: item.to_dict() for item in self.quantile_thresholds
            },
            "summary": {
                "origin_count": len(self.records),
                "counts_by_subtask": {
                    subtask.value: sum(
                        record.subtask is subtask for record in self.records
                    )
                    for subtask in EditingSubtask
                },
                "unique_canonical_sources": len(
                    {item.canonical_source_sha256 for item in self.records}
                ),
                "unique_canonical_gt": len(
                    {item.canonical_gt_sha256 for item in self.records}
                ),
                "unique_murcko_scaffolds": len(
                    {item.scaffold_sha256 for item in self.records}
                ),
                "duplicate_source_group_count": len(self.duplicate_source_groups),
                "duplicate_source_origin_count": sum(
                    len(group.anonymous_sample_ids)
                    for group in self.duplicate_source_groups
                ),
                "duplicate_scaffold_group_count": len(self.duplicate_scaffold_groups),
                "duplicate_scaffold_origin_count": sum(
                    len(group.anonymous_sample_ids)
                    for group in self.duplicate_scaffold_groups
                ),
            },
            "duplicate_source_groups": [
                group.to_dict() for group in self.duplicate_source_groups
            ],
            "duplicate_scaffold_groups": [
                group.to_dict() for group in self.duplicate_scaffold_groups
            ],
            "origins": [record.to_dict() for record in self.records],
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class FeatureDistribution:
    feature: str
    counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        counts = tuple(sorted(self.counts))
        if type(self.feature) is not str or not self.feature:
            raise ValueError("distribution feature must be non-empty")
        if not counts or any(
            type(key) is not str or not key or type(count) is not int or count <= 0
            for key, count in counts
        ):
            raise ValueError("distribution counts must be positive labeled counts")
        if len({key for key, _ in counts}) != len(counts):
            raise ValueError("distribution count labels must be unique")
        object.__setattr__(self, "counts", counts)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass(frozen=True, slots=True)
class OriginSplitFeatureDistribution:
    dataset_version: str
    source_audit_sha256: str
    distributions: tuple[FeatureDistribution, ...]
    duplicate_source_groups: tuple[DuplicateSourceGroup, ...]
    duplicate_scaffold_groups: tuple[DuplicateScaffoldGroup, ...]
    origin_count: int = 150
    format_version: str = FEATURE_DISTRIBUTION_FORMAT_VERSION

    def __post_init__(self) -> None:
        distributions = tuple(sorted(self.distributions, key=lambda item: item.feature))
        if (
            self.format_version != FEATURE_DISTRIBUTION_FORMAT_VERSION
            or self.dataset_version != "pilot_v1"
        ):
            raise ValueError("unknown feature distribution identity")
        if len(self.source_audit_sha256) != 64:
            raise ValueError("source_audit_sha256 must be SHA256")
        if self.origin_count != 150:
            raise ValueError("feature distribution requires 150 origins")
        if len({item.feature for item in distributions}) != len(distributions):
            raise ValueError("feature distributions must be unique")
        object.__setattr__(self, "distributions", distributions)
        object.__setattr__(
            self, "duplicate_source_groups", tuple(self.duplicate_source_groups)
        )
        object.__setattr__(
            self, "duplicate_scaffold_groups", tuple(self.duplicate_scaffold_groups)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset_version": self.dataset_version,
            "source_audit_sha256": self.source_audit_sha256,
            "summary": {
                "origin_count": self.origin_count,
                "duplicate_source_group_count": len(self.duplicate_source_groups),
                "duplicate_source_origin_count": sum(
                    len(group.anonymous_sample_ids)
                    for group in self.duplicate_source_groups
                ),
                "duplicate_scaffold_group_count": len(self.duplicate_scaffold_groups),
                "duplicate_scaffold_origin_count": sum(
                    len(group.anonymous_sample_ids)
                    for group in self.duplicate_scaffold_groups
                ),
            },
            "missing_features": {
                "mol_complexity": next(
                    item.to_dict().get("missing", 0)
                    for item in self.distributions
                    if item.feature == "mol_complexity_quantile_bin"
                ),
                "tanimoto": next(
                    item.to_dict().get("missing", 0)
                    for item in self.distributions
                    if item.feature == "tanimoto_quantile_bin"
                ),
            },
            "duplicate_source_groups": [
                group.to_dict() for group in self.duplicate_source_groups
            ],
            "duplicate_scaffold_groups": [
                group.to_dict() for group in self.duplicate_scaffold_groups
            ],
            "distributions": {
                item.feature: item.to_dict() for item in self.distributions
            },
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class OriginSplitAuditResult:
    audit: OriginSplitAudit
    feature_distribution: OriginSplitFeatureDistribution

    def __post_init__(self) -> None:
        if (
            type(self.audit) is not OriginSplitAudit
            or type(self.feature_distribution) is not OriginSplitFeatureDistribution
        ):
            raise TypeError("origin split audit result contains invalid artifacts")
        if (
            self.feature_distribution.source_audit_sha256
            != hashlib.sha256(self.audit.to_json_bytes()).hexdigest()
        ):
            raise ValueError("feature distribution is not bound to the audit bytes")


@dataclass(frozen=True, slots=True)
class _PreparedOrigin:
    item: OriginValidationInput
    origin_id: str
    rxn_cls: str
    anchor_elements: tuple[str, ...]
    scaffold: str | None
    source_heavy: int
    source_rings: int
    heavy_delta: int
    ring_delta: int
    remove_size: int
    add_size: int
    mol_complexity: float | None
    tanimoto: float | None
    availability: OperatorAvailability
    operation_subtype: str


def _raw_optional_number(raw: Mapping[str, Any], field: str) -> float | None:
    value = raw.get(field)
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"raw {field} must be numeric or missing")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"raw {field} must be finite")
    if field == "tanimoto" and not 0.0 <= numeric <= 1.0:
        raise ValueError("raw tanimoto must be in [0, 1]")
    return numeric


def _anchor_elements(item: OriginValidationInput) -> tuple[str, ...]:
    molecule = Chem.MolFromSmiles(item.edit_truth.source_smiles)
    if molecule is None:
        raise ValueError("T015-passing source could not be parsed")
    elements_by_map = {
        atom.GetAtomMapNum(): atom.GetSymbol()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    try:
        return tuple(
            sorted(
                {
                    elements_by_map[index]
                    for index in item.edit_truth.valid_anchor_indices
                }
            )
        )
    except KeyError as error:
        raise ValueError("EditTruth anchor is absent from its source map") from error


def _availability(
    item: OriginValidationInput, registry: PerturbatorRegistry
) -> tuple[OperatorAvailability, str]:
    classification = classify_edit_truth(item.edit_truth)
    # T026 is an immutable split-stratification snapshot.  Later append-only
    # runtime capabilities must not perturb its frozen 150-origin assignment.
    from molhallulens.modules.error_injection.operators.deletion import (
        REPLACEMENT_DELETION_OPERATOR_ID,
    )

    registrations = registry.registrations_for(
        subtask=item.edit_truth.normalized_subtask
    )
    registrations = tuple(
        registration
        for registration in registrations
        if registration.operator_id != REPLACEMENT_DELETION_OPERATOR_ID
    )
    eligible: list[str] = []
    reasons: list[tuple[str, tuple[str, ...]]] = []
    for registration in registrations:
        forbidden = tuple(
            sorted(
                f"capability_forbidden:{capability.value}"
                for capability in registration.required_capabilities
                if not classification.allows(capability)
            )
        )
        if forbidden:
            reasons.append((registration.operator_id, forbidden))
        else:
            eligible.append(registration.operator_id)
    availability = OperatorAvailability(
        capability_flags=tuple(
            (capability.value, classification.allows(capability))
            for capability in _FROZEN_SPLIT_CAPABILITIES
        ),
        registered_operator_ids=tuple(item.operator_id for item in registrations),
        eligible_operator_ids=tuple(eligible),
        ineligible_operator_reasons=tuple(reasons),
    )
    return availability, classification.operation_subtype.value


def _heavy_delta_bin(value: int) -> str:
    if value <= -5:
        return "negative_large"
    if value < 0:
        return "negative_small"
    if value == 0:
        return "zero"
    if value <= 4:
        return "positive_small"
    return "positive_large"


def _ring_delta_bin(value: int) -> str:
    return "negative" if value < 0 else "positive" if value > 0 else "zero"


def _fragment_size_bin(value: int) -> str:
    if value == 0:
        return "none"
    if value == 1:
        return "single_atom"
    if value <= 3:
        return "small_2_3"
    if value <= 7:
        return "medium_4_7"
    return "large_8_plus"


def _prepare_origin(
    item: OriginValidationInput, registry: PerturbatorRegistry
) -> _PreparedOrigin:
    truth = item.edit_truth
    raw = item.record.raw_record
    origin_id = raw.get("orig_id")
    rxn_cls = raw.get("rxn_cls")
    if (
        type(origin_id) is not str
        or not origin_id
        or type(rxn_cls) is not str
        or not rxn_cls
    ):
        raise ValueError("origin audit requires raw orig_id and rxn_cls")
    availability, operation_subtype = _availability(item, registry)
    scaffold = murcko_scaffold_smiles(
        truth.canonical_source_smiles,
        fragment_policy=FragmentPolicy.LARGEST_HEAVY,
    )
    remove_size = (
        0
        if truth.remove_fragment is None
        else truth.remove_fragment.descriptors.heavy_atom_count
    )
    add_size = (
        0
        if truth.add_fragment is None
        else truth.add_fragment.descriptors.heavy_atom_count
    )
    heavy_delta = (
        truth.product_descriptors.heavy_atom_count
        - truth.source_descriptors.heavy_atom_count
    )
    if heavy_delta != truth.heavy_atom_delta:
        raise ValueError("EditTruth heavy delta identities disagree")
    return _PreparedOrigin(
        item=item,
        origin_id=origin_id,
        rxn_cls=rxn_cls,
        anchor_elements=_anchor_elements(item),
        scaffold=scaffold,
        source_heavy=truth.source_descriptors.heavy_atom_count,
        source_rings=truth.source_descriptors.ring_count,
        heavy_delta=heavy_delta,
        ring_delta=truth.product_descriptors.ring_count
        - truth.source_descriptors.ring_count,
        remove_size=remove_size,
        add_size=add_size,
        mol_complexity=_raw_optional_number(raw, "mol_complexity"),
        tanimoto=_raw_optional_number(raw, "tanimoto"),
        availability=availability,
        operation_subtype=operation_subtype,
    )


def _distribution(audit: OriginSplitAudit) -> OriginSplitFeatureDistribution:
    records = audit.records
    features: dict[str, Iterable[str]] = {
        "subtask": (item.subtask.value for item in records),
        "rxn_cls": (item.rxn_cls for item in records),
        "anchor_element": (item.anchor_element for item in records),
        "source_heavy_atom_quantile_bin": (
            item.source_heavy_atom_quantile_bin for item in records
        ),
        "source_ring_quantile_bin": (item.source_ring_quantile_bin for item in records),
        "heavy_atom_delta_bin": (item.heavy_atom_delta_bin for item in records),
        "ring_delta_bin": (item.ring_delta_bin for item in records),
        "fragment_size_bin": (item.fragment_size_bin for item in records),
        "mol_complexity_quantile_bin": (
            item.mol_complexity_quantile_bin for item in records
        ),
        "tanimoto_quantile_bin": (item.tanimoto_quantile_bin for item in records),
        "eligible_operator_count": (
            str(len(item.operator_availability.eligible_operator_ids))
            for item in records
        ),
    }
    for capability in _FROZEN_SPLIT_CAPABILITIES:
        features[f"capability:{capability.value}"] = (
            "available"
            if dict(item.operator_availability.capability_flags)[capability.value]
            else "unavailable"
            for item in records
        )
    distributions = tuple(
        FeatureDistribution(feature=feature, counts=_count_pairs(values))
        for feature, values in features.items()
    )
    return OriginSplitFeatureDistribution(
        dataset_version=audit.dataset_version,
        source_audit_sha256=hashlib.sha256(audit.to_json_bytes()).hexdigest(),
        distributions=distributions,
        duplicate_source_groups=audit.duplicate_source_groups,
        duplicate_scaffold_groups=audit.duplicate_scaffold_groups,
    )


def audit_origin_split_features(
    items: Iterable[OriginValidationInput],
    *,
    config: ConfigBundle | None = None,
    registry: PerturbatorRegistry | None = None,
) -> OriginSplitAuditResult:
    """Validate and audit exactly 150 T013/T015 origin inputs."""

    from molhallulens.infrastructure.validation.reference import (
        OriginValidationInput,
        audit_reference_corpus,
        validate_reference_origin_strict,
    )

    values = tuple(items)
    if any(type(item) is not OriginValidationInput for item in values):
        raise TypeError("items must contain OriginValidationInput values")
    loaded_config = load_config_bundle() if config is None else config
    if type(loaded_config) is not ConfigBundle:
        raise TypeError("config must be ConfigBundle or None")
    if registry is None:
        registry = PerturbatorRegistry.from_perturbator_types(
            _PERTURBATOR_TYPES,
            operators_config=loaded_config.operators,
        )
    if type(registry) is not PerturbatorRegistry:
        raise TypeError("registry must be PerturbatorRegistry or None")
    from molhallulens.modules.error_injection.operators.deletion import (
        REPLACEMENT_DELETION_OPERATOR_ID,
    )

    frozen_registrations = tuple(
        registration
        for registration in registry.registrations_for()
        if registration.operator_id != REPLACEMENT_DELETION_OPERATOR_ID
    )
    if len(frozen_registrations) != 35:
        raise ValueError("T026 requires the frozen 35-operator T017 registry")

    ordered_values = tuple(
        sorted(values, key=lambda value: value.record.anonymous_sample_id)
    )
    for item in ordered_values:
        validate_reference_origin_strict(item)
    validation = audit_reference_corpus(ordered_values, require_all_pass=True)
    expected_total = loaded_config.dataset.dataset.origins_total
    if validation.attempted != expected_total or validation.passed != expected_total:
        raise ValueError("split audit requires the frozen 150-origin T015 corpus")

    prepared = tuple(_prepare_origin(item, registry) for item in ordered_values)
    missing_complexity = tuple(
        item.item.edit_truth.normalized_subtask
        for item in prepared
        if item.mol_complexity is None
    )
    missing_tanimoto = tuple(
        item.item.edit_truth.normalized_subtask
        for item in prepared
        if item.tanimoto is None
    )
    expected_missing = (EditingSubtask.DELETE,) * 50
    if missing_complexity != expected_missing or missing_tanimoto != expected_missing:
        raise ValueError(
            "only the 50 deletion origins may lack raw complexity/tanimoto"
        )

    thresholds = {
        "source_heavy_atom_count": QuantileThresholds.from_values(
            "source_heavy_atom_count", (item.source_heavy for item in prepared)
        ),
        "source_ring_count": QuantileThresholds.from_values(
            "source_ring_count", (item.source_rings for item in prepared)
        ),
        "mol_complexity": QuantileThresholds.from_values(
            "mol_complexity",
            (
                item.mol_complexity
                for item in prepared
                if item.mol_complexity is not None
            ),
        ),
        "tanimoto": QuantileThresholds.from_values(
            "tanimoto",
            (item.tanimoto for item in prepared if item.tanimoto is not None),
        ),
    }

    records = tuple(
        OriginSplitAuditRecord(
            origin_id=item.origin_id,
            anonymous_sample_id=item.item.edit_truth.anonymous_sample_id,
            subtask=item.item.edit_truth.normalized_subtask,
            operation_subtype=item.operation_subtype,
            rxn_cls=item.rxn_cls,
            anchor_element="+".join(item.anchor_elements),
            anchor_elements=item.anchor_elements,
            canonical_source_sha256=_sha256_text(
                item.item.edit_truth.canonical_source_smiles
            ),
            canonical_gt_sha256=_sha256_text(item.item.edit_truth.canonical_gt_smiles),
            scaffold_identity=item.scaffold,
            scaffold_sha256=_sha256_text(
                item.scaffold if item.scaffold is not None else NO_SCAFFOLD_IDENTITY
            ),
            scaffold_present=item.scaffold is not None,
            source_heavy_atom_count=item.source_heavy,
            source_heavy_atom_quantile_bin=thresholds[
                "source_heavy_atom_count"
            ].bin_for(item.source_heavy),
            source_ring_count=item.source_rings,
            source_ring_quantile_bin=thresholds["source_ring_count"].bin_for(
                item.source_rings
            ),
            heavy_atom_delta=item.heavy_delta,
            heavy_atom_delta_bin=_heavy_delta_bin(item.heavy_delta),
            ring_delta=item.ring_delta,
            ring_delta_bin=_ring_delta_bin(item.ring_delta),
            remove_fragment_heavy_atom_count=item.remove_size,
            add_fragment_heavy_atom_count=item.add_size,
            fragment_size=max(item.remove_size, item.add_size),
            fragment_size_bin=_fragment_size_bin(max(item.remove_size, item.add_size)),
            mol_complexity=item.mol_complexity,
            mol_complexity_source="missing_in_raw_benchmark_data"
            if item.mol_complexity is None
            else "raw_benchmark_data",
            mol_complexity_quantile_bin="missing"
            if item.mol_complexity is None
            else thresholds["mol_complexity"].bin_for(item.mol_complexity),
            tanimoto=item.tanimoto,
            tanimoto_source="missing_in_raw_benchmark_data"
            if item.tanimoto is None
            else "raw_benchmark_data",
            tanimoto_quantile_bin="missing"
            if item.tanimoto is None
            else thresholds["tanimoto"].bin_for(item.tanimoto),
            operator_availability=item.availability,
        )
        for item in prepared
    )
    delete_anomaly = next(
        record
        for record in records
        if record.anonymous_sample_id == "mol_edit.delete_v2.0081"
    )
    if not (
        len(delete_anomaly.operator_availability.registered_operator_ids) == 12
        and len(delete_anomaly.operator_availability.eligible_operator_ids) == 4
        and len(delete_anomaly.operator_availability.ineligible_operator_reasons) == 8
    ):
        raise ValueError("delete-with-replacement static eligibility changed")
    source_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        source_groups[record.canonical_source_sha256].append(record.anonymous_sample_id)
    duplicate_groups = tuple(
        DuplicateSourceGroup(
            group_id=f"source:{source_hash[:16]}",
            canonical_source_sha256=source_hash,
            anonymous_sample_ids=tuple(origin_ids),
        )
        for source_hash, origin_ids in sorted(source_groups.items())
        if len(origin_ids) > 1
    )
    scaffold_groups_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        scaffold_groups_by_hash[record.scaffold_sha256].append(
            record.anonymous_sample_id
        )
    duplicate_scaffold_groups = tuple(
        DuplicateScaffoldGroup(
            group_id=f"scaffold:{scaffold_hash[:16]}",
            scaffold_sha256=scaffold_hash,
            anonymous_sample_ids=tuple(origin_ids),
        )
        for scaffold_hash, origin_ids in sorted(scaffold_groups_by_hash.items())
        if len(origin_ids) > 1
    )
    audit = OriginSplitAudit(
        dataset_version=loaded_config.dataset.dataset.version_name,
        rdkit_version=rdkit.__version__,
        quantile_thresholds=tuple(thresholds.values()),
        duplicate_source_groups=duplicate_groups,
        duplicate_scaffold_groups=duplicate_scaffold_groups,
        records=records,
        t015_attempted=validation.attempted,
        t015_passed=validation.passed,
    )
    return OriginSplitAuditResult(
        audit=audit, feature_distribution=_distribution(audit)
    )


def build_origin_split_audit(dataset_root: Path) -> OriginSplitAuditResult:
    """Load the frozen corpus and build both T026 artifacts."""

    from molhallulens.infrastructure.validation.reference import OriginValidationInput

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path")
    items = []
    for record in ChemCoTMolEditAdapter().load(dataset_root):
        artifact = build_reference_dag(record)
        item = OriginValidationInput(
            record=record,
            artifact=artifact,
            edit_truth=derive_edit_truth(artifact),
        )
        items.append(item)
    return audit_origin_split_features(items)


def write_origin_split_audit(
    result: OriginSplitAuditResult,
    *,
    audit_path: Path,
    distribution_path: Path,
) -> None:
    """Write the byte-canonical audit artifacts."""

    if type(result) is not OriginSplitAuditResult:
        raise TypeError("result must be OriginSplitAuditResult")
    for path in (audit_path, distribution_path):
        if not isinstance(path, Path):
            raise TypeError("artifact paths must be pathlib.Path")
        path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(result.audit.to_json_bytes())
    distribution_path.write_bytes(result.feature_distribution.to_json_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--distribution-output", type=Path, default=None)
    args = parser.parse_args()
    audit_output = (
        args.audit_output or args.dataset_root / "reports" / DEFAULT_AUDIT_FILENAME
    )
    distribution_output = (
        args.distribution_output
        or args.dataset_root / "reports" / DEFAULT_DISTRIBUTION_FILENAME
    )
    result = build_origin_split_audit(args.dataset_root)
    write_origin_split_audit(
        result,
        audit_path=audit_output,
        distribution_path=distribution_output,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_FORMAT_VERSION",
    "DEFAULT_AUDIT_FILENAME",
    "DEFAULT_DISTRIBUTION_FILENAME",
    "FEATURE_DISTRIBUTION_FORMAT_VERSION",
    "KNOWN_DUPLICATE_SCAFFOLD_GROUPS",
    "KNOWN_DUPLICATE_SOURCE_GROUPS",
    "DuplicateScaffoldGroup",
    "DuplicateSourceGroup",
    "FeatureDistribution",
    "OperatorAvailability",
    "OriginSplitAudit",
    "OriginSplitAuditRecord",
    "OriginSplitAuditResult",
    "OriginSplitFeatureDistribution",
    "QuantileThresholds",
    "audit_origin_split_features",
    "build_origin_split_audit",
    "write_origin_split_audit",
]
