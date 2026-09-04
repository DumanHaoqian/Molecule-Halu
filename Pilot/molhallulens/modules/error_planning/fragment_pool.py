"""Corpus-scale fragment and functional-group replacement pool."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molhallulens.config.hallucination_generation import HallucinationGenerationConfig
from molhallulens.infrastructure.chemistry import (
    FragmentPolicy,
    MoleculeParseError,
    canonicalize_smiles,
    compute_descriptors,
)
from molhallulens.core import ValueType

if TYPE_CHECKING:
    from molhallulens.modules.reference import ReferenceDAGArtifact


_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)
_NONE_VALUES = frozenset({"", "none", "null", "nil", "n/a"})


@dataclass(frozen=True, slots=True)
class FragmentEntry:
    """One deduplicated, sanitized fragment with selection metadata."""

    canonical_smiles: str
    heavy_atom_count: int
    ring_count: int
    formal_charge: int
    source_origin_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.canonical_smiles) is not str or not self.canonical_smiles:
            raise ValueError("canonical_smiles must be non-empty")
        for value, name in (
            (self.heavy_atom_count, "heavy_atom_count"),
            (self.ring_count, "ring_count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.formal_charge) is not int:
            raise TypeError("formal_charge must be an integer")
        origins = tuple(sorted(set(self.source_origin_ids)))
        if any(type(item) is not str or not item for item in origins):
            raise ValueError("source_origin_ids must contain non-empty IDs")
        object.__setattr__(self, "source_origin_ids", origins)

    def to_dict(self) -> dict[str, object]:
        return {
            "smiles": self.canonical_smiles,
            "heavy_atoms": self.heavy_atom_count,
            "rings": self.ring_count,
            "formal_charge": self.formal_charge,
            "source_origin_ids": list(self.source_origin_ids),
        }


@dataclass(frozen=True, slots=True)
class FragmentSelection:
    entry: FragmentEntry
    similarity: float | None
    accepted_pool_size: int


class FragmentPool:
    """A deterministic, deduplicated pool built from every reference origin."""

    def __init__(self, entries: Iterable[FragmentEntry]) -> None:
        collected = tuple(sorted(entries, key=lambda item: item.canonical_smiles))
        if not collected:
            raise ValueError("fragment pool cannot be empty")
        smiles = tuple(item.canonical_smiles for item in collected)
        if len(smiles) != len(set(smiles)):
            raise ValueError("fragment pool entries must be canonical-SMILES unique")
        self._entries = collected

    @property
    def entries(self) -> tuple[FragmentEntry, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_smiles(cls, smiles_values: Iterable[str]) -> FragmentPool:
        return cls._from_pairs((value, "manual") for value in smiles_values)

    @classmethod
    def from_reference_artifacts(
        cls,
        artifacts: Iterable[ReferenceDAGArtifact],
    ) -> FragmentPool:
        pairs: list[tuple[str, str]] = []
        for artifact in artifacts:
            graph = artifact.state_dag
            for node in graph.schema.nodes:
                if node.value_type is not ValueType.FRAGMENT:
                    continue
                value = graph.values[node.node_id].normalized_value
                if type(value) is str and value.strip().casefold() not in _NONE_VALUES:
                    pairs.append((value, artifact.anonymous_sample_id))
        return cls._from_pairs(pairs)

    @classmethod
    def _from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> FragmentPool:
        origins_by_smiles: dict[str, set[str]] = defaultdict(set)
        for smiles, origin_id in pairs:
            try:
                canonical = canonicalize_smiles(
                    smiles,
                    fragment_policy=FragmentPolicy.KEEP_ALL,
                )
            except MoleculeParseError:
                continue
            origins_by_smiles[canonical].add(origin_id)
        entries = []
        for canonical, origins in origins_by_smiles.items():
            descriptors = compute_descriptors(
                canonical,
                fragment_policy=FragmentPolicy.KEEP_ALL,
            )
            entries.append(
                FragmentEntry(
                    canonical_smiles=canonical,
                    heavy_atom_count=descriptors.heavy_atom_count,
                    ring_count=descriptors.ring_count,
                    formal_charge=descriptors.formal_charge,
                    source_origin_ids=tuple(origins),
                )
            )
        return cls(entries)

    def select_replacement(
        self,
        reference_smiles: str,
        *,
        config: HallucinationGenerationConfig,
        random_source: Random,
    ) -> FragmentSelection:
        """Select a different fragment within explicit similarity/size bounds."""

        if type(reference_smiles) is not str:
            raise TypeError("reference_smiles must be text")
        if not isinstance(random_source, Random):
            raise TypeError("random_source must be random.Random")

        normalized = reference_smiles.strip().casefold()
        if normalized in _NONE_VALUES:
            lightweight = tuple(
                entry
                for entry in self._entries
                if entry.heavy_atom_count <= config.fragment_max_heavy_atom_difference
                and (
                    not config.fragment_require_same_charge
                    or entry.formal_charge == 0
                )
            )
            if not lightweight:
                raise ValueError("fragment pool has no replacement for a none leaving group")
            selected = lightweight[random_source.randrange(len(lightweight))]
            return FragmentSelection(selected, None, len(lightweight))

        canonical_reference = canonicalize_smiles(
            reference_smiles,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        reference_descriptors = compute_descriptors(
            canonical_reference,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        reference_molecule = Chem.MolFromSmiles(canonical_reference)
        if reference_molecule is None:
            raise ValueError("reference fragment cannot be parsed")
        reference_fp = _FINGERPRINT_GENERATOR.GetFingerprint(reference_molecule)

        accepted: list[tuple[float, FragmentEntry]] = []
        for entry in self._entries:
            if entry.canonical_smiles == canonical_reference:
                continue
            if (
                config.fragment_require_same_charge
                and entry.formal_charge != reference_descriptors.formal_charge
            ):
                continue
            if (
                abs(entry.heavy_atom_count - reference_descriptors.heavy_atom_count)
                > config.fragment_max_heavy_atom_difference
            ):
                continue
            molecule = Chem.MolFromSmiles(entry.canonical_smiles)
            if molecule is None:
                continue
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    reference_fp,
                    _FINGERPRINT_GENERATOR.GetFingerprint(molecule),
                )
            )
            if config.fragment_similarity_min <= similarity <= config.fragment_similarity_max:
                accepted.append((similarity, entry))
        if not accepted:
            raise ValueError("fragment pool has no candidate inside configured bounds")

        accepted.sort(
            key=lambda item: (
                abs(item[0] - config.fragment_target_similarity),
                item[1].canonical_smiles,
            )
        )
        best_distance = abs(accepted[0][0] - config.fragment_target_similarity)
        nearest = tuple(
            item
            for item in accepted
            if abs(abs(item[0] - config.fragment_target_similarity) - best_distance) < 1e-12
        )
        similarity, selected = nearest[random_source.randrange(len(nearest))]
        return FragmentSelection(selected, similarity, len(accepted))


__all__ = ["FragmentEntry", "FragmentPool", "FragmentSelection"]
