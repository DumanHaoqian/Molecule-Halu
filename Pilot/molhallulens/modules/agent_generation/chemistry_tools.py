"""Bounded RDKit tools for explicit, reproducible graph edits.

``inspect_source`` accepts fully mapped or entirely unmapped SMILES. It assigns
maps in input atom order only for entirely unmapped input. ``apply_edit_plan``
requires the fully mapped source returned by inspection. Partial/duplicate maps
are ambiguous and are rejected.

Plans contain any subset of these keys (all values are JSON lists)::

    remove_atom_maps: [map_id, ...]
    remove_bonds: [[map_id, map_id], ...]
    adjust_hydrogens: [{"atom_map": map_id, "delta": -1}, ...]
    add_fragments: [{"smiles": "C(=O)C", "attach_atom_index": 0,
                     "anchor_map": 3, "bond_type": "SINGLE"}, ...]
    change_bonds: [{"atom_maps": [map_id, map_id], "bond_type": "DOUBLE"}, ...]
    add_bonds: [{"atom_maps": [map_id, map_id], "bond_type": "SINGLE"}, ...]

Every referenced map identifies a surviving source atom, except the explicit
deletion list. Each fragment must be connected and unmapped. Its attachment
index refers to input SMILES atom order, not canonical order. Newly added maps
increase monotonically from the maximum original map, including deleted maps.

Execution order is hydrogen adjustments, bond removals, atom removals, fragment
additions, bond changes, and bond additions. Conflicting/repeated edits to the
same atom property or bond are rejected. Intermediates are graph snapshots, not
claims of stable chemical intermediates; only the combined product is sanitized.

Bracketed explicit H counts are never inferred or repaired: attaching to [OH:3]
requires an explicit delta of -1. Ordinary implicit H follows RDKit valence.
Atoms, charges, isotopes, and unchanged stereochemistry otherwise remain intact;
no component is discarded. Radical/dummy atoms, empty products and changes that
fail RDKit sanitization are outside this closed-shell executor's scope. These
checks establish graph validity, not the intended reaction's chemical semantics.
Edits to the neighbors, H count or bond orders of a retained atom with defined
tetrahedral or double-bond stereochemistry require an explicit stereochemical
policy and are rejected as ambiguous in this version. Untouched stereo survives.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors


MAX_ATOMS = 512
MAX_OPERATIONS = 128
MAX_SMILES_LENGTH = 20_000

_BOND_TYPES = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}
_PLAN_KEYS = {
    "remove_atom_maps", "remove_bonds", "adjust_hydrogens",
    "add_fragments", "change_bonds", "add_bonds",
}


class ChemistryToolError(ValueError):
    """A typed, serializable tool failure; never an accepted molecular result."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise ChemistryToolError(code, message)


def _supported(mol: Chem.Mol) -> None:
    if not 0 < mol.GetNumAtoms() <= MAX_ATOMS:
        _fail("invalid_structure", f"Molecule must contain 1 to {MAX_ATOMS} atoms")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetNumRadicalElectrons():
            _fail("invalid_structure", "Dummy atoms and radicals are unsupported")


def _parse(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > MAX_SMILES_LENGTH:
        _fail("invalid_structure", "Expected nonempty SMILES within the length limit")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    params.allowCXSMILES = False
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        _fail("invalid_structure", "SMILES parsing or sanitization failed")
    _supported(mol)
    return mol


def _source(smiles: str, *, assign_maps: bool) -> Chem.Mol:
    mol = _parse(smiles)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if not any(maps) and assign_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    elif any(m <= 0 for m in maps) or len(set(maps)) != len(maps):
        _fail("ambiguous_edit", "Source atom maps must be complete, positive, and unique")
    return mol


def _unmapped(mol: Chem.Mol) -> Chem.Mol:
    result = Chem.Mol(mol)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    return result


def _canonical(mol: Chem.Mol, *, stereo: bool = True) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(_unmapped(mol)), canonical=True, isomericSmiles=stereo)


def _components(mol: Chem.Mol) -> list[list[int]]:
    return sorted(
        sorted(mol.GetAtomWithIdx(i).GetAtomMapNum() for i in indices)
        for indices in Chem.GetMolFrags(mol)
    )


def _facts(mol: Chem.Mol) -> dict[str, Any]:
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "components": _components(mol),
    }


def _atoms(mol: Chem.Mol) -> list[dict[str, Any]]:
    return [{
        "atom_index": atom.GetIdx(),
        "atom_map": atom.GetAtomMapNum(),
        "element": atom.GetSymbol(),
        "isotope": atom.GetIsotope(),
        "formal_charge": atom.GetFormalCharge(),
        "explicit_hydrogens": atom.GetNumExplicitHs(),
        "total_hydrogens": atom.GetTotalNumHs(includeNeighbors=True),
        "aromatic": atom.GetIsAromatic(),
        "chiral_tag": str(atom.GetChiralTag()),
        "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
    } for atom in mol.GetAtoms()]


def _bonds(mol: Chem.Mol) -> list[dict[str, Any]]:
    return [{
        "atom_maps": [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()],
        "atom_indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
        "bond_type": str(bond.GetBondType()),
        "stereo": str(bond.GetStereo()),
        "stereo_atom_maps": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in bond.GetStereoAtoms()],
    } for bond in mol.GetBonds()]


def inspect_source(source_smiles: str) -> dict[str, Any]:
    """Inspect a source and expose deterministic maps, components and stereo."""
    mol = _source(source_smiles, assign_maps=True)
    mapped_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Mapped SMILES brackets serialize implicit H as explicit H. Report the
    # actual representation the caller will pass back for graph editing.
    mol = _parse(mapped_smiles)
    return {
        "mapped_smiles": mapped_smiles,
        "canonical_smiles": _canonical(mol),
        "atoms": _atoms(mol),
        "bonds": _bonds(mol),
        **_facts(mol),
    }


def describe_fragment(smiles: str) -> dict[str, Any]:
    """Inspect a connected unmapped fragment; indices preserve input ordering."""
    mol = _fragment(smiles)
    return {"valid": True, "canonical_smiles": _canonical(mol),
            "atoms": _atoms(mol), "bonds": _bonds(mol), **_facts(mol)}


def _fragment(smiles: str) -> Chem.Mol:
    mol = _parse(smiles)
    if len(Chem.GetMolFrags(mol)) != 1:
        _fail("ambiguous_edit", "Each fragment must be one connected component")
    if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        _fail("ambiguous_edit", "Fragment atom maps are assigned by the executor")
    return mol


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        _fail("invalid_plan", f"{label} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _fields(item: Any, fields: set[str], label: str) -> None:
    if not isinstance(item, dict) or set(item) != fields:
        _fail("invalid_plan", f"{label} requires exactly these fields: {sorted(fields)}")


def _pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        _fail("invalid_plan", "Bond endpoints must be a pair of atom maps")
    pair = tuple(_integer(v, "atom_map", minimum=1) for v in value)
    if pair[0] == pair[1]:
        _fail("invalid_plan", "Bond endpoints must be distinct")
    return tuple(sorted(pair))


def _bond_type(value: Any) -> Chem.BondType:
    if not isinstance(value, str) or value not in _BOND_TYPES:
        _fail("invalid_plan", f"bond_type must be one of {sorted(_BOND_TYPES)}")
    return _BOND_TYPES[value]


def _map_indices(mol: Chem.Mol) -> dict[int, int]:
    return {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}


def _validate_plan(mol: Chem.Mol, plan: Any) -> dict[str, list]:
    if not isinstance(plan, dict) or set(plan) - _PLAN_KEYS:
        _fail("invalid_plan", f"Plan must be an object with keys from {sorted(_PLAN_KEYS)}")
    normalized = {key: plan.get(key, []) for key in _PLAN_KEYS}
    if any(not isinstance(value, list) for value in normalized.values()):
        _fail("invalid_plan", "Every plan value must be a list")
    if sum(map(len, normalized.values())) > MAX_OPERATIONS:
        _fail("invalid_plan", f"Plan exceeds {MAX_OPERATIONS} operations")
    indices = _map_indices(mol)
    removed = [_integer(v, "remove_atom_maps", minimum=1) for v in normalized["remove_atom_maps"]]
    if len(set(removed)) != len(removed):
        _fail("ambiguous_edit", "Repeated atom removal")
    if set(removed) - indices.keys():
        _fail("invalid_plan", "Removed atom map does not exist")

    def surviving(atom_map: Any) -> int:
        atom_map = _integer(atom_map, "atom_map", minimum=1)
        if atom_map not in indices or atom_map in removed:
            _fail("invalid_plan", f"Atom map {atom_map} must identify a surviving source atom")
        return indices[atom_map]

    adjusted = set()
    for item in normalized["adjust_hydrogens"]:
        _fields(item, {"atom_map", "delta"}, "Hydrogen adjustment")
        index = surviving(item["atom_map"])
        delta = _integer(item["delta"], "delta")
        if item["atom_map"] in adjusted or not delta:
            _fail("ambiguous_edit", "Hydrogen adjustments must be unique and nonzero")
        adjusted.add(item["atom_map"])
        if not 0 <= mol.GetAtomWithIdx(index).GetNumExplicitHs() + delta <= 8:
            _fail("invalid_plan", "Hydrogen adjustment must leave an explicit H count between 0 and 8")

    touched_bonds = set()
    for key in ("remove_bonds", "change_bonds", "add_bonds"):
        for item in normalized[key]:
            if key == "remove_bonds":
                pair = _pair(item)
            else:
                _fields(item, {"atom_maps", "bond_type"}, key)
                pair = _pair(item["atom_maps"])
                _bond_type(item["bond_type"])
            if pair in touched_bonds:
                _fail("ambiguous_edit", "A bond may be edited only once per plan")
            touched_bonds.add(pair)
            first, second = (surviving(v) for v in pair)
            bond = mol.GetBondBetweenAtoms(first, second)
            if (key == "add_bonds" and bond is not None) or (key != "add_bonds" and bond is None):
                _fail("invalid_plan", f"{key} endpoints have the wrong existing bond state")
            if key == "change_bonds" and bond.GetBondType() == _bond_type(item["bond_type"]):
                _fail("invalid_plan", "Bond change must change the bond type")

    total_atoms = mol.GetNumAtoms() - len(removed)
    for item in normalized["add_fragments"]:
        _fields(item, {"smiles", "attach_atom_index", "anchor_map", "bond_type"}, "Fragment addition")
        surviving(item["anchor_map"])
        _bond_type(item["bond_type"])
        fragment = _fragment(item["smiles"])
        attach = _integer(item["attach_atom_index"], "attach_atom_index", minimum=0)
        if attach >= fragment.GetNumAtoms():
            _fail("invalid_plan", "Fragment attachment atom index does not exist")
        total_atoms += fragment.GetNumAtoms()
    if not 0 < total_atoms <= MAX_ATOMS:
        _fail("invalid_structure", f"Product must contain 1 to {MAX_ATOMS} atoms")
    # RDKit chiral tags depend on neighbor ordering. Deleting a branch and then
    # appending its replacement can invert the molecule while retaining the tag.
    # This version has no explicit retention/inversion policy, so fail closed.
    stereo_maps = {
        atom.GetAtomMapNum() for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }
    for bond in mol.GetBonds():
        if bond.GetStereo() not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}:
            stereo_maps.update((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
    affected = {item["atom_map"] for item in normalized["adjust_hydrogens"]}
    affected.update(item["anchor_map"] for item in normalized["add_fragments"])
    for atom_map in removed:
        affected.update(atom.GetAtomMapNum() for atom in mol.GetAtomWithIdx(indices[atom_map]).GetNeighbors())
    for pair in normalized["remove_bonds"]:
        affected.update(pair)
    for item in normalized["change_bonds"] + normalized["add_bonds"]:
        affected.update(item["atom_maps"])
    ambiguous = (affected & stereo_maps) - set(removed)
    if ambiguous:
        _fail("ambiguous_edit", f"Editing defined stereochemistry at maps {sorted(ambiguous)} requires an explicit stereo policy")
    return normalized


def _graph(mol: Chem.Mol) -> dict[str, Any]:
    """A raw graph snapshot, safe even before the combined edit is sanitized."""
    return {"atoms": [{"atom_map": a.GetAtomMapNum(), "element": a.GetSymbol(),
                       "explicit_hydrogens": a.GetNumExplicitHs(),
                       "formal_charge": a.GetFormalCharge(), "chiral_tag": str(a.GetChiralTag())}
                      for a in mol.GetAtoms()], "bonds": _bonds(mol)}


def apply_edit_plan(source_smiles: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Apply all approved edits together and derive molecular facts from RDKit."""
    source = _source(source_smiles, assign_maps=False)
    edits = _validate_plan(source, plan)
    mol = Chem.RWMol(source)
    next_map = max(_map_indices(source)) + 1
    operations: list[dict[str, Any]] = []

    def record(operation: str, **details: Any) -> None:
        operations.append({"operation_id": f"edit_{len(operations) + 1}",
                           "operation": operation, **details, "intermediate_graph": _graph(mol)})

    for item in edits["adjust_hydrogens"]:
        atom = mol.GetAtomWithIdx(_map_indices(mol)[item["atom_map"]])
        before = atom.GetNumExplicitHs()
        atom.SetNumExplicitHs(before + item["delta"])
        record("adjust_hydrogens", **item, before=before, after=atom.GetNumExplicitHs())
    for pair in edits["remove_bonds"]:
        indices = _map_indices(mol)
        first, second = (indices[v] for v in pair)
        before = str(mol.GetBondBetweenAtoms(first, second).GetBondType())
        mol.RemoveBond(first, second)
        record("remove_bond", atom_maps=list(pair), before=before)
    for atom_map in edits["remove_atom_maps"]:
        index = _map_indices(mol)[atom_map]
        atom = mol.GetAtomWithIdx(index)
        before = {"element": atom.GetSymbol(), "neighbor_maps": [a.GetAtomMapNum() for a in atom.GetNeighbors()]}
        mol.RemoveAtom(index)
        record("remove_atom", atom_map=atom_map, before=before)
    for item in edits["add_fragments"]:
        fragment = _fragment(item["smiles"])
        new_maps = list(range(next_map, next_map + fragment.GetNumAtoms()))
        for atom, atom_map in zip(fragment.GetAtoms(), new_maps):
            atom.SetAtomMapNum(atom_map)
        next_map += fragment.GetNumAtoms()
        offset = mol.GetNumAtoms()
        mol = Chem.RWMol(Chem.CombineMols(mol, fragment))
        anchor = _map_indices(mol)[item["anchor_map"]]
        mol.AddBond(anchor, offset + item["attach_atom_index"], _bond_type(item["bond_type"]))
        record("add_fragment", **item, new_atom_maps=new_maps,
               attachment_atom_map=new_maps[item["attach_atom_index"]])
    for item in edits["change_bonds"]:
        indices = _map_indices(mol)
        first, second = (indices[v] for v in item["atom_maps"])
        bond = mol.GetBondBetweenAtoms(first, second)
        before = str(bond.GetBondType())
        bond.SetBondType(_bond_type(item["bond_type"]))
        bond.SetIsAromatic(item["bond_type"] == "AROMATIC")
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
        record("change_bond", **item, before=before)
    for item in edits["add_bonds"]:
        indices = _map_indices(mol)
        first, second = (indices[v] for v in item["atom_maps"])
        mol.AddBond(first, second, _bond_type(item["bond_type"]))
        record("add_bond", **item)
    product = mol.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(product)
            Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    except Exception as error:
        raise ChemistryToolError("invalid_structure", "Combined edit failed RDKit sanitization") from error
    _supported(product)
    indices = _map_indices(product)
    # Aromaticity perception can rewrite a formally requested bond order back
    # to the original order. Such a plan did not execute as requested.
    for item in edits["change_bonds"] + edits["add_bonds"]:
        bond = product.GetBondBetweenAtoms(*(indices[v] for v in item["atom_maps"]))
        if bond.GetBondType() != _bond_type(item["bond_type"]):
            _fail("invalid_structure", "Sanitization changed a requested bond type (possibly aromatic)")
    for operation in operations:
        if operation["operation"] == "add_fragment":
            bond = product.GetBondBetweenAtoms(indices[operation["anchor_map"]], indices[operation["attachment_atom_map"]])
            if bond.GetBondType() != _bond_type(operation["bond_type"]):
                _fail("invalid_structure", "Sanitization changed the requested fragment attachment bond type")
    return {
        "source_mapped_smiles": Chem.MolToSmiles(source, canonical=True, isomericSmiles=True),
        "product_smiles": _canonical(product),
        "mapped_product_smiles": Chem.MolToSmiles(product, canonical=True, isomericSmiles=True),
        "operations": operations,
        **_facts(product),
    }


def compare_molecules(a: str, b: str) -> dict[str, Any]:
    """Compare full component multisets and stereochemistry, ignoring atom maps."""
    first, second = _parse(a), _parse(b)
    canonical_a, canonical_b = _canonical(first), _canonical(second)
    equivalent = canonical_a == canonical_b
    same_connectivity = _canonical(first, stereo=False) == _canonical(second, stereo=False)
    return {
        "equivalent": equivalent,
        "same_connectivity": same_connectivity,
        "stereochemistry_equal": equivalent,
        "same_components": sorted(canonical_a.split(".")) == sorted(canonical_b.split(".")),
        "canonical_a": canonical_a,
        "canonical_b": canonical_b,
    }
