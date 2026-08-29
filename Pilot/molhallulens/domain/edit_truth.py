"""Immutable, RDKit-free graph-difference truth for molecule edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isclose, isfinite
from typing import Any

from .enums import BondTypeName, EditingSubtask
from .molecules import (
    AtomDescriptor,
    AtomReference,
    AtomReferenceNamespace,
    MoleculeDescriptors,
)


@dataclass(frozen=True, slots=True)
class AtomMappingPair:
    source: AtomReference
    product: AtomReference

    def __post_init__(self) -> None:
        if type(self.source) is not AtomReference or type(self.product) is not AtomReference:
            raise TypeError("mapping endpoints must be AtomReference")
        if self.source.namespace is not AtomReferenceNamespace.SOURCE_MAP:
            raise ValueError("mapping source must use SOURCE_MAP")
        if self.product.namespace is not AtomReferenceNamespace.PRODUCT_CANONICAL:
            raise ValueError("mapping product must use PRODUCT_CANONICAL")

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.source.atom_id, self.product.atom_id)


@dataclass(frozen=True, slots=True)
class AtomMapping:
    pairs: tuple[AtomMappingPair, ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.pairs)
        if not pairs or any(type(pair) is not AtomMappingPair for pair in pairs):
            raise ValueError("pairs must contain AtomMappingPair values")
        pairs = tuple(sorted(pairs, key=lambda pair: pair.sort_key))
        if len({pair.source for pair in pairs}) != len(pairs):
            raise ValueError("source mapping endpoints must be unique")
        if len({pair.product for pair in pairs}) != len(pairs):
            raise ValueError("product mapping endpoints must be unique")
        object.__setattr__(self, "pairs", pairs)

    @property
    def sort_key(self) -> tuple[tuple[int, int], ...]:
        return tuple(pair.sort_key for pair in self.pairs)

    @property
    def mapped_atom_count(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True, slots=True)
class BondEdit:
    begin: AtomReference
    end: AtomReference
    bond_type: BondTypeName
    stereo: str
    aromatic: bool

    def __post_init__(self) -> None:
        if type(self.begin) is not AtomReference or type(self.end) is not AtomReference:
            raise TypeError("bond endpoints must be AtomReference")
        if self.begin == self.end:
            raise ValueError("self-bonds are invalid")
        if self.end.sort_key < self.begin.sort_key:
            begin = self.begin
            object.__setattr__(self, "begin", self.end)
            object.__setattr__(self, "end", begin)
        if type(self.bond_type) is not BondTypeName:
            raise TypeError("bond_type must be BondTypeName")
        if type(self.stereo) is not str or not self.stereo:
            raise ValueError("stereo must be non-empty text")
        if type(self.aromatic) is not bool:
            raise TypeError("aromatic must be bool")

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.begin.sort_key,
            self.end.sort_key,
            self.bond_type.value,
            self.stereo,
            self.aromatic,
        )


@dataclass(frozen=True, slots=True)
class FragmentSpec:
    canonical_smiles: str
    component_smiles: tuple[str, ...]
    atom_references: tuple[AtomReference, ...]
    attachment_atoms: tuple[AtomReference, ...]
    boundary_bonds: tuple[BondEdit, ...]
    descriptors: MoleculeDescriptors

    def __post_init__(self) -> None:
        if type(self.canonical_smiles) is not str or not self.canonical_smiles:
            raise ValueError("canonical_smiles must be non-empty")
        components = tuple(sorted(self.component_smiles))
        atoms = tuple(sorted(self.atom_references, key=lambda item: item.sort_key))
        attachments = tuple(sorted(self.attachment_atoms, key=lambda item: item.sort_key))
        bonds = tuple(sorted(self.boundary_bonds, key=lambda item: item.sort_key))
        if not components or any(type(item) is not str or not item for item in components):
            raise ValueError("component_smiles must contain non-empty strings")
        if not atoms or any(type(item) is not AtomReference for item in atoms):
            raise ValueError("atom_references must contain AtomReference values")
        if any(type(item) is not AtomReference for item in attachments):
            raise TypeError("attachment_atoms must contain AtomReference values")
        if any(type(item) is not BondEdit for item in bonds):
            raise TypeError("boundary_bonds must contain BondEdit values")
        if len(set(atoms)) != len(atoms) or len(set(attachments)) != len(attachments):
            raise ValueError("fragment atom collections must be unique")
        if len(set(bonds)) != len(bonds):
            raise ValueError("boundary_bonds must be unique")
        atom_set = set(atoms)
        if not set(attachments).issubset(atom_set):
            raise ValueError("attachment atoms must belong to the fragment")
        fragment_boundary_atoms: set[AtomReference] = set()
        for bond in bonds:
            begin_inside = bond.begin in atom_set
            end_inside = bond.end in atom_set
            if begin_inside == end_inside:
                raise ValueError("a boundary bond must have exactly one fragment endpoint")
            fragment_boundary_atoms.add(bond.begin if begin_inside else bond.end)
        if fragment_boundary_atoms != set(attachments):
            raise ValueError("attachment atoms must equal fragment-side boundary endpoints")
        if type(self.descriptors) is not MoleculeDescriptors:
            raise TypeError("descriptors must be MoleculeDescriptors")
        if self.descriptors.canonical_smiles != self.canonical_smiles:
            raise ValueError("fragment descriptor identity mismatch")
        if self.descriptors.heavy_atom_count != len(atoms):
            raise ValueError("fragment descriptor heavy-atom count mismatch")
        object.__setattr__(self, "component_smiles", components)
        object.__setattr__(self, "atom_references", atoms)
        object.__setattr__(self, "attachment_atoms", attachments)
        object.__setattr__(self, "boundary_bonds", bonds)


@dataclass(frozen=True, slots=True)
class MappingEvidence:
    algorithm: str
    rdkit_version: str
    mcs_smarts: str | None
    source_heavy_atoms: int
    product_heavy_atoms: int
    mapped_heavy_atoms: int
    optimal_mappings: tuple[AtomMapping, ...]
    inequivalent_edit_signature_count: int
    coverage: float
    ambiguity_penalty: float
    confidence: float
    trace_anchor_indices: tuple[int, ...] = ()
    trace_anchor_agreement: bool | None = None
    trace_consistent: bool | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.algorithm, "algorithm"),
            (self.rdkit_version, "rdkit_version"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.mcs_smarts is not None and (
            type(self.mcs_smarts) is not str or not self.mcs_smarts
        ):
            raise ValueError("mcs_smarts must be non-empty text or None")
        for value, name in (
            (self.source_heavy_atoms, "source_heavy_atoms"),
            (self.product_heavy_atoms, "product_heavy_atoms"),
            (self.mapped_heavy_atoms, "mapped_heavy_atoms"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mapped_heavy_atoms > min(self.source_heavy_atoms, self.product_heavy_atoms):
            raise ValueError("mapped heavy atoms exceed a molecule")
        mappings = tuple(sorted(self.optimal_mappings, key=lambda item: item.sort_key))
        if not mappings or any(type(item) is not AtomMapping for item in mappings):
            raise ValueError("optimal_mappings must contain AtomMapping values")
        if len(set(mappings)) != len(mappings):
            raise ValueError("optimal_mappings must be unique")
        if any(item.mapped_atom_count != self.mapped_heavy_atoms for item in mappings):
            raise ValueError("optimal mapping sizes must match mapped_heavy_atoms")
        if (
            type(self.inequivalent_edit_signature_count) is not int
            or not 1 <= self.inequivalent_edit_signature_count <= len(mappings)
        ):
            raise ValueError("inequivalent edit signature count is invalid")
        trace = tuple(sorted(self.trace_anchor_indices))
        if any(type(item) is not int or item <= 0 for item in trace) or len(set(trace)) != len(trace):
            raise ValueError("trace_anchor_indices must be unique positive integers")
        if self.trace_anchor_agreement is not None and type(self.trace_anchor_agreement) is not bool:
            raise TypeError("trace_anchor_agreement must be bool or None")
        if self.trace_consistent is not None and type(self.trace_consistent) is not bool:
            raise TypeError("trace_consistent must be bool or None")
        if (
            self.trace_anchor_agreement is not None
            and self.trace_consistent is not None
            and self.trace_anchor_agreement != self.trace_consistent
        ):
            raise ValueError("trace consistency aliases disagree")
        trace_consistency = (
            self.trace_anchor_agreement
            if self.trace_anchor_agreement is not None
            else self.trace_consistent
        )
        expected_coverage = self.mapped_heavy_atoms / min(
            self.source_heavy_atoms, self.product_heavy_atoms
        )
        expected_penalty = 1.0 / self.inequivalent_edit_signature_count
        expected_confidence = expected_coverage * expected_penalty
        for actual, expected, name in (
            (self.coverage, expected_coverage, "coverage"),
            (self.ambiguity_penalty, expected_penalty, "ambiguity_penalty"),
            (self.confidence, expected_confidence, "confidence"),
        ):
            if type(actual) is not float or not isfinite(actual) or not isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{name} does not match its formula")
        object.__setattr__(self, "optimal_mappings", mappings)
        object.__setattr__(self, "trace_anchor_indices", trace)
        object.__setattr__(self, "trace_anchor_agreement", trace_consistency)
        object.__setattr__(self, "trace_consistent", trace_consistency)

    @property
    def optimal_mapping_count(self) -> int:
        return len(self.optimal_mappings)

    @property
    def confidence_formula(self) -> str:
        """Versioned deterministic confidence definition."""

        return "mapped_heavy_over_smaller_graph_times_inverse_signature_count_v1"


@dataclass(frozen=True, slots=True)
class EditTruth:
    anonymous_sample_id: str
    normalized_subtask: EditingSubtask
    source_smiles: str
    gt_smiles: str
    canonical_source_smiles: str
    canonical_gt_smiles: str
    valid_anchor_indices: tuple[int, ...]
    symmetry_equivalent_anchors: tuple[tuple[int, ...], ...]
    removed_atom_maps: frozenset[int]
    added_atoms: tuple[AtomDescriptor, ...]
    broken_bonds: tuple[BondEdit, ...]
    formed_bonds: tuple[BondEdit, ...]
    remove_fragment: FragmentSpec | None
    add_fragment: FragmentSpec | None
    source_descriptors: MoleculeDescriptors
    product_descriptors: MoleculeDescriptors
    mapping_evidence: MappingEvidence
    mapping_confidence: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.source_smiles, "source_smiles"),
            (self.gt_smiles, "gt_smiles"),
            (self.canonical_source_smiles, "canonical_source_smiles"),
            (self.canonical_gt_smiles, "canonical_gt_smiles"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        anchors = tuple(sorted(self.valid_anchor_indices))
        removed = frozenset(self.removed_atom_maps)
        groups = tuple(sorted(tuple(sorted(group)) for group in self.symmetry_equivalent_anchors))
        added = tuple(sorted(self.added_atoms, key=lambda item: item.reference.sort_key))
        broken = tuple(sorted(self.broken_bonds, key=lambda item: item.sort_key))
        formed = tuple(sorted(self.formed_bonds, key=lambda item: item.sort_key))
        if not anchors or any(type(item) is not int or item <= 0 for item in anchors):
            raise ValueError("valid_anchor_indices must be positive integers")
        if len(set(anchors)) != len(anchors):
            raise ValueError("valid_anchor_indices must be unique")
        grouped: set[int] = set()
        for group in groups:
            if len(group) < 2 or len(set(group)) != len(group) or not set(group).issubset(anchors):
                raise ValueError("symmetry groups must be non-singleton subsets of valid anchors")
            if grouped.intersection(group):
                raise ValueError("symmetry groups must be disjoint")
            grouped.update(group)
        if any(type(item) is not int or item <= 0 for item in removed):
            raise ValueError("removed_atom_maps must be positive integers")
        if any(type(item) is not AtomDescriptor for item in added):
            raise TypeError("added_atoms must contain AtomDescriptor values")
        added_refs = tuple(item.reference for item in added)
        if len(set(added_refs)) != len(added_refs):
            raise ValueError("added atom identities must be unique")
        if any(item.namespace is not AtomReferenceNamespace.PRODUCT_CANONICAL for item in added_refs):
            raise ValueError("added atoms must use PRODUCT_CANONICAL identities")
        if any(type(item) is not BondEdit for item in (*broken, *formed)):
            raise TypeError("bond collections must contain BondEdit values")
        if len(set(broken)) != len(broken) or len(set(formed)) != len(formed):
            raise ValueError("bond collections must be unique")
        if any(
            endpoint.namespace is not AtomReferenceNamespace.SOURCE_MAP
            for bond in broken for endpoint in (bond.begin, bond.end)
        ):
            raise ValueError("broken bonds must use SOURCE_MAP endpoints")
        if bool(removed) != (self.remove_fragment is not None):
            raise ValueError("remove_fragment presence must match removed atoms")
        if bool(added) != (self.add_fragment is not None):
            raise ValueError("add_fragment presence must match added atoms")
        if self.remove_fragment is not None:
            expected = {AtomReference(AtomReferenceNamespace.SOURCE_MAP, item) for item in removed}
            if set(self.remove_fragment.atom_references) != expected:
                raise ValueError("remove_fragment atoms do not match removed_atom_maps")
        if self.add_fragment is not None and set(self.add_fragment.atom_references) != set(added_refs):
            raise ValueError("add_fragment atoms do not match added_atoms")
        if type(self.source_descriptors) is not MoleculeDescriptors or type(self.product_descriptors) is not MoleculeDescriptors:
            raise TypeError("molecule descriptors have invalid types")
        if self.source_descriptors.canonical_smiles != self.canonical_source_smiles:
            raise ValueError("source descriptor identity mismatch")
        if self.product_descriptors.canonical_smiles != self.canonical_gt_smiles:
            raise ValueError("product descriptor identity mismatch")
        if type(self.mapping_evidence) is not MappingEvidence:
            raise TypeError("mapping_evidence must be MappingEvidence")
        if self.mapping_evidence.source_heavy_atoms != self.source_descriptors.heavy_atom_count:
            raise ValueError("source mapping count mismatch")
        if self.mapping_evidence.product_heavy_atoms != self.product_descriptors.heavy_atom_count:
            raise ValueError("product mapping count mismatch")
        if self.mapping_evidence.mapped_heavy_atoms + len(removed) != self.source_descriptors.heavy_atom_count:
            raise ValueError("source atom partition mismatch")
        if self.mapping_evidence.mapped_heavy_atoms + len(added) != self.product_descriptors.heavy_atom_count:
            raise ValueError("product atom partition mismatch")
        anchor_set = set(anchors)
        removed_set = set(removed)
        added_reference_set = set(added_refs)
        compatible_mapping_exists = any(
            anchor_set.issubset(
                {pair.source.atom_id for pair in mapping.pairs}
            )
            and removed_set.isdisjoint(
                {pair.source.atom_id for pair in mapping.pairs}
            )
            and added_reference_set.isdisjoint(
                {pair.product for pair in mapping.pairs}
            )
            for mapping in self.mapping_evidence.optimal_mappings
        )
        if not compatible_mapping_exists:
            raise ValueError(
                "no optimal mapping is compatible with anchors and the atom partition"
            )
        if type(self.mapping_confidence) is not float or not isclose(
            self.mapping_confidence, self.mapping_evidence.confidence, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("mapping_confidence must equal evidence confidence")
        object.__setattr__(self, "valid_anchor_indices", anchors)
        object.__setattr__(self, "symmetry_equivalent_anchors", groups)
        object.__setattr__(self, "removed_atom_maps", removed)
        object.__setattr__(self, "added_atoms", added)
        object.__setattr__(self, "broken_bonds", broken)
        object.__setattr__(self, "formed_bonds", formed)

    @property
    def heavy_atom_delta(self) -> int:
        return len(self.added_atoms) - len(self.removed_atom_maps)

    def to_json_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): encode(item) for key, item in value.items()}
            if isinstance(value, (tuple, list, set, frozenset)):
                items = [encode(item) for item in value]
                return sorted(items) if isinstance(value, (set, frozenset)) else items
            return value

        return encode(asdict(self))


__all__ = [
    "AtomDescriptor",
    "AtomMapping",
    "AtomMappingPair",
    "AtomReference",
    "AtomReferenceNamespace",
    "BondEdit",
    "EditTruth",
    "FragmentSpec",
    "MappingEvidence",
]
