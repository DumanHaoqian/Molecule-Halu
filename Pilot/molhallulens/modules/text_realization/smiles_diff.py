"""Character-level paired intervals for reference/candidate molecular strings."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from rdkit import Chem


DiffOpcode = tuple[str, int, int, int, int]


@dataclass(frozen=True, slots=True)
class MolecularTextDiff:
    """One contiguous bounding interval on both sides of a molecular edit."""

    reference_start: int
    reference_end: int
    candidate_start: int
    candidate_end: int
    opcodes: tuple[DiffOpcode, ...]

    def __post_init__(self) -> None:
        if not (
            0 <= self.reference_start < self.reference_end
            and 0 <= self.candidate_start < self.candidate_end
        ):
            raise ValueError("molecular diff intervals must be non-empty")
        if not self.opcodes:
            raise ValueError("molecular diff requires at least one changed opcode")


def molecular_text_diff(reference: str, candidate: str) -> MolecularTextDiff:
    """Return minimal changed bounding intervals with non-empty paired sides.

    A pure insertion or deletion has an empty interval on one side.  In that
    case one adjacent equal character is included on both sides so H and N can
    still expose non-empty spans and exact substitution remains possible.
    """

    if type(reference) is not str or not reference:
        raise ValueError("reference molecular text must be non-empty")
    if type(candidate) is not str or not candidate:
        raise ValueError("candidate molecular text must be non-empty")
    if reference == candidate:
        raise ValueError("molecular diff requires distinct strings")
    all_opcodes = tuple(
        SequenceMatcher(None, reference, candidate, autojunk=False).get_opcodes()
    )
    changed = tuple(opcode for opcode in all_opcodes if opcode[0] != "equal")
    if not changed:
        raise ValueError("molecular diff found no changed opcode")
    reference_start = min(item[1] for item in changed)
    reference_end = max(item[2] for item in changed)
    candidate_start = min(item[3] for item in changed)
    candidate_end = max(item[4] for item in changed)

    if reference_start == reference_end or candidate_start == candidate_end:
        if (
            reference_end < len(reference)
            and candidate_end < len(candidate)
            and reference[reference_end] == candidate[candidate_end]
        ):
            reference_end += 1
            candidate_end += 1
        elif (
            reference_start > 0
            and candidate_start > 0
            and reference[reference_start - 1] == candidate[candidate_start - 1]
        ):
            reference_start -= 1
            candidate_start -= 1
        else:
            raise ValueError(
                "pure molecular insertion/deletion has no shared boundary character"
            )

    return MolecularTextDiff(
        reference_start=reference_start,
        reference_end=reference_end,
        candidate_start=candidate_start,
        candidate_end=candidate_end,
        opcodes=changed,
    )


def align_candidate_molecular_text(reference: str, candidate: str) -> str:
    """Choose an equivalent candidate SMILES traversal closest to reference.

    RDKit canonicalization can move the start atom after a one-atom edit and
    make an otherwise local graph change look like a whole-string rewrite.
    This renderer-only normalization enumerates deterministic rooted SMILES for
    the already selected candidate molecule and chooses the representation with
    the smallest single paired diff interval.  It does not change the molecule
    or any planning/injection state.
    """

    molecule = Chem.MolFromSmiles(candidate)
    if molecule is None:
        raise ValueError("candidate molecular text is not valid SMILES")
    representations = {candidate}
    for atom_index in range(molecule.GetNumAtoms()):
        for canonical in (False, True):
            representations.add(
                Chem.MolToSmiles(
                    molecule,
                    rootedAtAtom=atom_index,
                    canonical=canonical,
                    isomericSmiles=True,
                )
            )
    scored = []
    for representation in representations:
        if not representation or representation == reference:
            continue
        difference = molecular_text_diff(reference, representation)
        reference_length = difference.reference_end - difference.reference_start
        candidate_length = difference.candidate_end - difference.candidate_start
        scored.append(
            (
                max(reference_length, candidate_length),
                reference_length + candidate_length,
                candidate_length,
                len(difference.opcodes),
                representation,
            )
        )
    if not scored:
        raise ValueError("candidate molecule has no reference-different SMILES rendering")
    return min(scored)[-1]


__all__ = [
    "DiffOpcode",
    "MolecularTextDiff",
    "align_candidate_molecular_text",
    "molecular_text_diff",
]
