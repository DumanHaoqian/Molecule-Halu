"""Generate chemically valid but reference-different SMILES candidates."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from molhallulens.config.hallucination_generation import HallucinationGenerationConfig
from molhallulens.infrastructure.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
)


_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


@dataclass(frozen=True, slots=True)
class SmilesMutationSelection:
    smiles: str
    operator: str
    similarity: float
    accepted_pool_size: int
    metadata: dict[str, Any]


def _sanitized_smiles(molecule: Chem.Mol) -> str | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
            Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
            rendered = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
    except (RuntimeError, ValueError):
        return None
    return rendered or None


def _atom_replacement_candidates(
    reference: Chem.Mol,
    allowed_atomic_numbers: tuple[int, ...],
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates = []
    for atom in reference.GetAtoms():
        before_atomic_number = atom.GetAtomicNum()
        for after_atomic_number in allowed_atomic_numbers:
            if after_atomic_number == before_atomic_number:
                continue
            edited = Chem.RWMol(reference)
            edited_atom = edited.GetAtomWithIdx(atom.GetIdx())
            edited_atom.SetAtomicNum(after_atomic_number)
            rendered = _sanitized_smiles(edited.GetMol())
            if rendered is not None:
                candidates.append(
                    (
                        rendered,
                        "smiles_atom_replacement",
                        {
                            "atom_index": atom.GetIdx(),
                            "before_atomic_number": before_atomic_number,
                            "after_atomic_number": after_atomic_number,
                        },
                    )
                )
    return candidates


def _bond_order_candidates(
    reference: Chem.Mol,
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates = []
    toggles = {
        Chem.BondType.SINGLE: Chem.BondType.DOUBLE,
        Chem.BondType.DOUBLE: Chem.BondType.SINGLE,
    }
    for bond in reference.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() not in toggles:
            continue
        before_type = bond.GetBondType()
        after_type = toggles[before_type]
        edited = Chem.RWMol(reference)
        edited_bond = edited.GetBondBetweenAtoms(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
        )
        if edited_bond is None:
            continue
        edited_bond.SetBondType(after_type)
        rendered = _sanitized_smiles(edited.GetMol())
        if rendered is not None:
            candidates.append(
                (
                    rendered,
                    "smiles_bond_order_change",
                    {
                        "begin_atom_index": bond.GetBeginAtomIdx(),
                        "end_atom_index": bond.GetEndAtomIdx(),
                        "before_bond_type": str(before_type),
                        "after_bond_type": str(after_type),
                    },
                )
            )
    return candidates


def _terminal_deletion_candidates(
    reference: Chem.Mol,
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates = []
    if reference.GetNumHeavyAtoms() <= 1:
        return candidates
    for atom in reference.GetAtoms():
        if atom.GetDegree() != 1 or atom.GetAtomicNum() == 1:
            continue
        edited = Chem.RWMol(reference)
        atom_index = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        edited.RemoveAtom(atom_index)
        rendered = _sanitized_smiles(edited.GetMol())
        if rendered is not None:
            candidates.append(
                (
                    rendered,
                    "smiles_terminal_atom_deletion",
                    {
                        "removed_atom_index": atom_index,
                        "removed_atomic_number": atomic_number,
                    },
                )
            )
    return candidates


def select_smiles_mutation(
    reference_smiles: str,
    *,
    config: HallucinationGenerationConfig,
    random_source: Random,
) -> SmilesMutationSelection:
    """Enumerate, sanitize, de-duplicate, similarity-filter, then select."""

    if type(reference_smiles) is not str or not reference_smiles.strip():
        raise ValueError("reference_smiles must be non-empty text")
    if not isinstance(random_source, Random):
        raise TypeError("random_source must be random.Random")
    canonical_reference = canonicalize_smiles(
        reference_smiles,
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    reference = Chem.MolFromSmiles(canonical_reference)
    if reference is None:
        raise ValueError("reference SMILES cannot be parsed")

    enabled = set(config.smiles_mutation_operators)
    raw_candidates: list[tuple[str, str, dict[str, Any]]] = []
    if "smiles_atom_replacement" in enabled:
        raw_candidates.extend(
            _atom_replacement_candidates(
                reference,
                config.smiles_replacement_atomic_numbers,
            )
        )
    if "smiles_bond_order_change" in enabled:
        raw_candidates.extend(_bond_order_candidates(reference))
    if "smiles_terminal_atom_deletion" in enabled:
        raw_candidates.extend(_terminal_deletion_candidates(reference))

    reference_fp = _FINGERPRINT_GENERATOR.GetFingerprint(reference)
    accepted_by_smiles: dict[str, tuple[str, float, dict[str, Any]]] = {}
    for raw_smiles, operator, metadata in raw_candidates:
        canonical = canonicalize_smiles(
            raw_smiles,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        if config.require_different_from_reference and canonical == canonical_reference:
            continue
        candidate = Chem.MolFromSmiles(canonical)
        if candidate is None:
            continue
        similarity = float(
            DataStructs.TanimotoSimilarity(
                reference_fp,
                _FINGERPRINT_GENERATOR.GetFingerprint(candidate),
            )
        )
        if not config.smiles_similarity_min <= similarity <= config.smiles_similarity_max:
            continue
        accepted_by_smiles.setdefault(canonical, (operator, similarity, metadata))

    accepted = sorted(
        (
            (smiles, operator, similarity, metadata)
            for smiles, (operator, similarity, metadata) in accepted_by_smiles.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    if not accepted:
        raise ValueError("no valid reference-different SMILES mutation satisfies config")
    smiles, operator, similarity, metadata = accepted[
        random_source.randrange(len(accepted))
    ]
    return SmilesMutationSelection(
        smiles=smiles,
        operator=operator,
        similarity=similarity,
        accepted_pool_size=len(accepted),
        metadata=metadata,
    )


__all__ = ["SmilesMutationSelection", "select_smiles_mutation"]
