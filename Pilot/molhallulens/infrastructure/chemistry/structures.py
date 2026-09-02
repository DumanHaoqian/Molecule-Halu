"""Strict RDKit canonicalization, equivalence, descriptors, and scaffolds."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from molhallulens.core.molecules import (
    FragmentPolicy,
    MainFragment,
    MoleculeDescriptors,
)


class MoleculeErrorCode(StrEnum):
    INVALID_INPUT_TYPE = "invalid_input_type"
    EMPTY_SMILES = "empty_smiles"
    SMILES_PARSE_FAILED = "smiles_parse_failed"
    SMILES_SANITIZE_FAILED = "smiles_sanitize_failed"
    CANONICALIZATION_FAILED = "canonicalization_failed"


class MoleculeParseError(ValueError):
    """Stable parse/sanitize failure that never embeds molecule plaintext."""

    def __init__(
        self,
        code: MoleculeErrorCode,
        message: str,
        *,
        input_length: int | None,
    ) -> None:
        if type(code) is not MoleculeErrorCode:
            raise TypeError("MoleculeParseError code must be a MoleculeErrorCode")
        if type(message) is not str or not message:
            raise ValueError("MoleculeParseError message must be non-empty text")
        if input_length is not None and (
            type(input_length) is not int or input_length < 0
        ):
            raise ValueError("input_length must be a non-negative integer or None")
        self.code = code
        self.input_length = input_length
        super().__init__(message)


def _error(
    code: MoleculeErrorCode,
    message: str,
    value: Any,
) -> MoleculeParseError:
    return MoleculeParseError(
        code,
        message,
        input_length=len(value) if type(value) is str else None,
    )


def _parse_smiles_strict(smiles: str) -> Chem.Mol:
    """Parse with strict sanitization and standard RDKit explicit-H handling."""

    if type(smiles) is not str:
        raise _error(
            MoleculeErrorCode.INVALID_INPUT_TYPE,
            "SMILES input must be a string",
            smiles,
        )
    if not smiles.strip():
        raise _error(
            MoleculeErrorCode.EMPTY_SMILES,
            "SMILES input cannot be empty",
            smiles,
        )

    try:
        with rdBase.BlockLogs():
            parsed_without_sanitize = Chem.MolFromSmiles(smiles, sanitize=False)
    except (RuntimeError, ValueError) as error:
        raise _error(
            MoleculeErrorCode.SMILES_PARSE_FAILED,
            "SMILES syntax parsing failed",
            smiles,
        ) from error
    if parsed_without_sanitize is None:
        raise _error(
            MoleculeErrorCode.SMILES_PARSE_FAILED,
            "SMILES syntax parsing failed",
            smiles,
        )

    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except (RuntimeError, ValueError) as error:
        raise _error(
            MoleculeErrorCode.SMILES_SANITIZE_FAILED,
            "SMILES strict sanitization failed",
            smiles,
        ) from error
    if molecule is None:
        raise _error(
            MoleculeErrorCode.SMILES_SANITIZE_FAILED,
            "SMILES strict sanitization failed",
            smiles,
        )
    return molecule


def _without_atom_maps(molecule: Chem.Mol) -> Chem.Mol:
    copied = Chem.Mol(molecule)
    for atom in copied.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(copied, cleanIt=True, force=True)
    return copied


def _canonical_smiles(molecule: Chem.Mol, *, source_value: str) -> str:
    try:
        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except (RuntimeError, ValueError) as error:
        raise _error(
            MoleculeErrorCode.CANONICALIZATION_FAILED,
            "isomeric canonicalization failed",
            source_value,
        ) from error
    if type(canonical) is not str or not canonical:
        raise _error(
            MoleculeErrorCode.CANONICALIZATION_FAILED,
            "isomeric canonicalization failed",
            source_value,
        )
    return canonical


def _fragment_sort_key(
    molecule: Chem.Mol, *, source_value: str
) -> tuple[int, int, str]:
    canonical = _canonical_smiles(molecule, source_value=source_value)
    contains_carbon = any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    return (-molecule.GetNumHeavyAtoms(), -int(contains_carbon), canonical)


def _prepared_molecule(
    smiles: str,
    fragment_policy: FragmentPolicy,
) -> tuple[Chem.Mol, int]:
    if type(fragment_policy) is not FragmentPolicy:
        raise TypeError("fragment_policy must be a FragmentPolicy")
    full_molecule = _without_atom_maps(_parse_smiles_strict(smiles))
    fragments = tuple(Chem.GetMolFrags(full_molecule, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise _error(
            MoleculeErrorCode.CANONICALIZATION_FAILED,
            "molecule contains no canonicalizable fragments",
            smiles,
        )
    if fragment_policy is FragmentPolicy.KEEP_ALL:
        return full_molecule, len(fragments)
    selected = sorted(
        fragments,
        key=lambda molecule: _fragment_sort_key(molecule, source_value=smiles),
    )[0]
    return selected, len(fragments)


def canonicalize_smiles(
    smiles: str,
    *,
    fragment_policy: FragmentPolicy = FragmentPolicy.KEEP_ALL,
) -> str:
    """Return strict canonical isomeric SMILES after removing only atom maps.

    KEEP_ALL is the safe default: disconnected components are retained and sorted
    canonically. LARGEST_HEAVY must be requested explicitly and ranks fragments by
    descending heavy-atom count, then carbon presence, then canonical lexical form.
    """

    molecule, _ = _prepared_molecule(smiles, fragment_policy)
    return _canonical_smiles(molecule, source_value=smiles)


def isomeric_graph_equivalent(left_smiles: str, right_smiles: str) -> bool:
    """Compare complete sanitized molecular graphs, retaining stereochemistry."""

    left = canonicalize_smiles(left_smiles, fragment_policy=FragmentPolicy.KEEP_ALL)
    right = canonicalize_smiles(right_smiles, fragment_policy=FragmentPolicy.KEEP_ALL)
    return left == right


def fragment_graph_equivalent(left_smiles: str, right_smiles: str) -> bool:
    """Compare standalone fragment graphs; attachment orientation is out of scope."""

    return isomeric_graph_equivalent(left_smiles, right_smiles)


def select_main_fragment(smiles: str) -> MainFragment:
    """Select the frozen largest-heavy fragment without modifying the input."""

    molecule, input_fragment_count = _prepared_molecule(
        smiles, FragmentPolicy.LARGEST_HEAVY
    )
    return MainFragment(
        canonical_smiles=_canonical_smiles(molecule, source_value=smiles),
        heavy_atom_count=molecule.GetNumHeavyAtoms(),
        contains_carbon=any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()),
        input_fragment_count=input_fragment_count,
    )


def compute_descriptors(
    smiles: str,
    *,
    fragment_policy: FragmentPolicy = FragmentPolicy.KEEP_ALL,
) -> MoleculeDescriptors:
    """Compute the frozen descriptor vocabulary for the explicit fragment scope."""

    molecule, _ = _prepared_molecule(smiles, fragment_policy)
    heteroatom_counts = Counter(
        atom.GetAtomicNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() not in {1, 6}
    )
    return MoleculeDescriptors(
        canonical_smiles=_canonical_smiles(molecule, source_value=smiles),
        fragment_policy=fragment_policy,
        fragment_count=len(Chem.GetMolFrags(molecule)),
        heavy_atom_count=molecule.GetNumHeavyAtoms(),
        ring_count=int(rdMolDescriptors.CalcNumRings(molecule)),
        aromatic_ring_count=int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        formal_charge=int(Chem.GetFormalCharge(molecule)),
        heteroatom_counts=tuple(sorted(heteroatom_counts.items())),
        molecular_weight=float(Descriptors.MolWt(molecule)),
        exact_molecular_weight=float(rdMolDescriptors.CalcExactMolWt(molecule)),
        rotatable_bond_count=int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        hydrogen_bond_donor_count=int(rdMolDescriptors.CalcNumHBD(molecule)),
        hydrogen_bond_acceptor_count=int(rdMolDescriptors.CalcNumHBA(molecule)),
        topological_polar_surface_area=float(rdMolDescriptors.CalcTPSA(molecule)),
        log_p=float(Descriptors.MolLogP(molecule)),
    )


def murcko_scaffold_smiles(
    smiles: str,
    *,
    fragment_policy: FragmentPolicy,
) -> str | None:
    """Return a canonical isomeric Bemis-Murcko scaffold, or None if acyclic.

    The caller must choose KEEP_ALL or LARGEST_HEAVY explicitly so counterion
    removal can never happen as an implicit side effect.
    """

    molecule, _ = _prepared_molecule(smiles, fragment_policy)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumHeavyAtoms() == 0:
        return None
    scaffold = _without_atom_maps(scaffold)
    return _canonical_smiles(scaffold, source_value=smiles)


def generic_murcko_scaffold_smiles(
    smiles: str,
    *,
    fragment_policy: FragmentPolicy,
) -> str | None:
    """Return the atom/bond-generic Bemis-Murcko identity for leakage audit.

    This is deliberately a separate identity from :func:`murcko_scaffold_smiles`:
    RDKit's ``MakeScaffoldGeneric`` maps scaffold atoms to carbon and bonds to
    single bonds, allowing T027 to conservatively group close scaffold analogues.
    Acyclic molecules remain ``None`` and callers must never group that sentinel.
    """

    molecule, _ = _prepared_molecule(smiles, fragment_policy)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumHeavyAtoms() == 0:
        return None
    try:
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        Chem.SanitizeMol(generic)
    except (RuntimeError, ValueError) as error:
        raise _error(
            MoleculeErrorCode.CANONICALIZATION_FAILED,
            "generic Bemis-Murcko canonicalization failed",
            smiles,
        ) from error
    generic = _without_atom_maps(generic)
    return _canonical_smiles(generic, source_value=smiles)


__all__ = [
    "FragmentPolicy",
    "MainFragment",
    "MoleculeDescriptors",
    "MoleculeErrorCode",
    "MoleculeParseError",
    "canonicalize_smiles",
    "compute_descriptors",
    "fragment_graph_equivalent",
    "generic_murcko_scaffold_smiles",
    "isomeric_graph_equivalent",
    "murcko_scaffold_smiles",
    "select_main_fragment",
]
