"""RDKit-independent immutable molecule value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class FragmentPolicy(StrEnum):
    """Explicit disconnected-component policy for molecular utilities."""

    KEEP_ALL = "keep_all"
    LARGEST_HEAVY = "largest_heavy"


@dataclass(frozen=True, slots=True)
class MainFragment:
    """Immutable result of the frozen largest-heavy-fragment strategy."""

    canonical_smiles: str
    heavy_atom_count: int
    contains_carbon: bool
    input_fragment_count: int
    policy: FragmentPolicy = FragmentPolicy.LARGEST_HEAVY

    def __post_init__(self) -> None:
        if type(self.canonical_smiles) is not str or not self.canonical_smiles:
            raise ValueError("MainFragment canonical_smiles must be non-empty")
        if type(self.heavy_atom_count) is not int or self.heavy_atom_count < 0:
            raise ValueError("MainFragment heavy_atom_count must be non-negative")
        if type(self.contains_carbon) is not bool:
            raise TypeError("MainFragment contains_carbon must be bool")
        if type(self.input_fragment_count) is not int or self.input_fragment_count <= 0:
            raise ValueError("MainFragment input_fragment_count must be positive")
        if self.policy is not FragmentPolicy.LARGEST_HEAVY:
            raise ValueError("MainFragment must use the frozen largest-heavy policy")


@dataclass(frozen=True, slots=True)
class MoleculeDescriptors:
    """Version-stable descriptor names with explicit RDKit definitions."""

    canonical_smiles: str
    fragment_policy: FragmentPolicy
    fragment_count: int
    heavy_atom_count: int
    ring_count: int
    aromatic_ring_count: int
    formal_charge: int
    heteroatom_counts: tuple[tuple[int, int], ...]
    molecular_weight: float
    exact_molecular_weight: float
    rotatable_bond_count: int
    hydrogen_bond_donor_count: int
    hydrogen_bond_acceptor_count: int
    topological_polar_surface_area: float
    log_p: float

    def __post_init__(self) -> None:
        if type(self.canonical_smiles) is not str or not self.canonical_smiles:
            raise ValueError("descriptor canonical_smiles must be non-empty")
        if type(self.fragment_policy) is not FragmentPolicy:
            raise TypeError("descriptor fragment_policy must be a FragmentPolicy")
        for value, name in (
            (self.fragment_count, "fragment_count"),
            (self.heavy_atom_count, "heavy_atom_count"),
            (self.ring_count, "ring_count"),
            (self.aromatic_ring_count, "aromatic_ring_count"),
            (self.rotatable_bond_count, "rotatable_bond_count"),
            (self.hydrogen_bond_donor_count, "hydrogen_bond_donor_count"),
            (self.hydrogen_bond_acceptor_count, "hydrogen_bond_acceptor_count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"descriptor {name} must be non-negative")
        if self.fragment_count <= 0:
            raise ValueError("descriptor fragment_count must be positive")
        if type(self.formal_charge) is not int:
            raise TypeError("descriptor formal_charge must be an integer")
        object.__setattr__(self, "heteroatom_counts", tuple(self.heteroatom_counts))
        previous_atomic_number = 0
        for item in self.heteroatom_counts:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or type(item[1]) is not int
                or item[0] <= previous_atomic_number
                or item[0] in {1, 6}
                or item[1] <= 0
            ):
                raise ValueError(
                    "heteroatom_counts must be sorted positive (atomic_number, count) pairs"
                )
            previous_atomic_number = item[0]
        for value, name in (
            (self.molecular_weight, "molecular_weight"),
            (self.exact_molecular_weight, "exact_molecular_weight"),
            (self.topological_polar_surface_area, "topological_polar_surface_area"),
            (self.log_p, "log_p"),
        ):
            if type(value) is not float or not isfinite(value):
                raise ValueError(f"descriptor {name} must be a finite float")
        if min(
            self.molecular_weight,
            self.exact_molecular_weight,
            self.topological_polar_surface_area,
        ) < 0:
            raise ValueError("weight and polar-surface descriptors must be non-negative")


class AtomReferenceNamespace(StrEnum):
    """Stable identity space used by graph-difference artifacts."""

    SOURCE_MAP = "source_map"
    PRODUCT_CANONICAL = "product_canonical"


@dataclass(frozen=True, slots=True)
class AtomReference:
    """Stable atom identity independent of transient RDKit atom order."""

    namespace: AtomReferenceNamespace
    atom_id: int

    def __post_init__(self) -> None:
        if type(self.namespace) is not AtomReferenceNamespace:
            raise TypeError("namespace must be AtomReferenceNamespace")
        if type(self.atom_id) is not int or self.atom_id <= 0:
            raise ValueError("atom_id must be a positive integer")

    @property
    def sort_key(self) -> tuple[int, int]:
        rank = 0 if self.namespace is AtomReferenceNamespace.SOURCE_MAP else 1
        return (rank, self.atom_id)


@dataclass(frozen=True, slots=True)
class AtomDescriptor:
    """JSON-safe atom attributes used to audit changed fragments."""

    reference: AtomReference
    atomic_number: int
    element: str
    isotope: int
    formal_charge: int
    aromatic: bool
    chiral_tag: str

    def __post_init__(self) -> None:
        if type(self.reference) is not AtomReference:
            raise TypeError("reference must be AtomReference")
        if type(self.atomic_number) is not int or self.atomic_number <= 0:
            raise ValueError("atomic_number must be positive")
        if type(self.element) is not str or not self.element:
            raise ValueError("element must be non-empty text")
        if type(self.isotope) is not int or self.isotope < 0:
            raise ValueError("isotope must be non-negative")
        if type(self.formal_charge) is not int:
            raise TypeError("formal_charge must be int")
        if type(self.aromatic) is not bool:
            raise TypeError("aromatic must be bool")
        if type(self.chiral_tag) is not str or not self.chiral_tag:
            raise ValueError("chiral_tag must be non-empty text")


__all__ = [
    "AtomDescriptor",
    "AtomReference",
    "AtomReferenceNamespace",
    "FragmentPolicy",
    "MainFragment",
    "MoleculeDescriptors",
]
