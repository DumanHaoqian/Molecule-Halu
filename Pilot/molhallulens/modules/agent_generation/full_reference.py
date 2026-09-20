"""Full-corpus clean reference recovery, with explicit execution extensions.

GT is consulted only by full_reference, never by execute_full_plan. Incoming
aromatic [nH] caps must be removed explicitly. A previously unspecified source
center may receive an explicit absolute CIP assignment; edits to already defined
stereo retain the strict executor's fail-closed policy.
"""
from copy import deepcopy
from typing import Any
from rdkit import Chem, rdBase
from .chemistry_tools import (
    ChemistryToolError, apply_edit_plan, compare_molecules, _source, _validate_plan,
    _map_indices, _graph, _fragment, _bond_type, _supported, _fail, _canonical, _facts,
)
from .quality import deterministic_reference_plan, validate_reference


def _extensions(source, plan):
    if not isinstance(plan, dict):
        _fail("invalid_plan", "Plan must be an object")
    base = deepcopy(plan)
    caps = base.pop("fragment_hydrogen_caps", [])
    stereo = base.pop("assign_tetrahedral", [])
    if not isinstance(caps, list) or not isinstance(stereo, list):
        _fail("invalid_plan", "Extensions must be lists")
    _validate_plan(source, base)
    seen = set()
    for item in caps:
        if not isinstance(item, dict) or set(item) != {"fragment_index", "atom_index", "delta"}:
            _fail("invalid_plan", "Invalid fragment hydrogen cap fields")
        fi, ai = item["fragment_index"], item["atom_index"]
        if type(fi) is not int or type(ai) is not int or fi < 0 or ai < 0 or type(item["delta"]) is not int or item["delta"] != -1:
            _fail("invalid_plan", "A cap requires nonnegative indices and delta -1")
        additions = base.get("add_fragments", [])
        if fi >= len(additions) or (fi, ai) in seen:
            _fail("invalid_plan", "Missing or duplicate fragment cap")
        fragment = _fragment(additions[fi]["smiles"])
        if ai >= fragment.GetNumAtoms() or ai != additions[fi]["attach_atom_index"]:
            _fail("invalid_plan", "Cap must identify the fragment attachment atom")
        atom = fragment.GetAtomWithIdx(ai)
        if atom.GetAtomicNum() != 7 or not atom.GetIsAromatic() or atom.GetNumExplicitHs() != 1:
            _fail("invalid_plan", "Only an aromatic nitrogen explicit H cap is supported")
        seen.add((fi, ai))
    seen = set()
    indices = _map_indices(source)
    for item in stereo:
        if not isinstance(item, dict) or set(item) != {"atom_map", "cip"}:
            _fail("invalid_plan", "Invalid tetrahedral assignment fields")
        am = item["atom_map"]
        if type(am) is not int or am not in indices or am in seen or am in base.get("remove_atom_maps", []) or item["cip"] not in ("R", "S"):
            _fail("invalid_plan", "Invalid tetrahedral assignment")
        atom = source.GetAtomWithIdx(indices[am])
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            _fail("ambiguous_edit", "Cannot overwrite defined source stereochemistry")
        seen.add(am)
    return base, caps, stereo


def execute_full_plan(source_smiles, plan):
    """Execute explicit graph/cap/stereo instructions without consulting GT."""
    if isinstance(plan, dict) and not (set(plan) & {"fragment_hydrogen_caps", "assign_tetrahedral"}):
        return apply_edit_plan(source_smiles, plan)
    return _execute_extended(source_smiles, plan)


def _execute_extended(source_smiles: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Apply all approved edits together and derive molecular facts from RDKit."""
    source = _source(source_smiles, assign_maps=False)
    base, caps, stereo = _extensions(source, plan)
    edits = _validate_plan(source, base)
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
    for fragment_index, item in enumerate(edits["add_fragments"]):
        fragment = _fragment(item["smiles"])
        for cap in caps:
            if cap["fragment_index"] == fragment_index:
                fragment.GetAtomWithIdx(cap["atom_index"]).SetNumExplicitHs(0)
        new_maps = list(range(next_map, next_map + fragment.GetNumAtoms()))
        for atom, atom_map in zip(fragment.GetAtoms(), new_maps):
            atom.SetAtomMapNum(atom_map)
        next_map += fragment.GetNumAtoms()
        offset = mol.GetNumAtoms()
        mol = Chem.RWMol(Chem.CombineMols(mol, fragment))
        anchor = _map_indices(mol)[item["anchor_map"]]
        mol.AddBond(anchor, offset + item["attach_atom_index"], _bond_type(item["bond_type"]))
        for cap in caps:
            if cap["fragment_index"] == fragment_index:
                record("remove_fragment_hydrogen_cap", **cap,
                       atom_map=new_maps[cap["atom_index"]], before=1, after=0)
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
    for item in stereo:
        atom = product.GetAtomWithIdx(_map_indices(product)[item["atom_map"]])
        for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atom.SetChiralTag(tag)
            Chem.AssignStereochemistry(product, cleanIt=True, force=True)
            if atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") == item["cip"]:
                break
        else:
            _fail("invalid_plan", "Explicit stereo assignment did not produce the requested CIP")
        record("assign_tetrahedral", **item)
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


def full_reference(row):
    """Recover an executable clean reference, requiring exact stereo and components.

    Four explicit source-map recipes cover documented single-anchor schema gaps.
    They never modify the input row and remain guarded by all frozen factual checks
    and full product equivalence. The one GT-defined, source-unspecified stereo
    center is selected here only, with its provenance returned for auditing.
    """
    validation = validate_reference(row)
    if validation['status'] != 'pass':
        _fail('reference_invalid', 'Frozen reference failed local factual checks')
    special = {
        'mol_edit.delete_v2.0081': (list(range(12, 36)), 11, 'C#N'),
        'mol_edit.substitute_v2.0123': ([23], 22, 'N1C=NC=N1'),
        'mol_edit.substitute_v2.0271': (list(range(21, 29)), 35, 'CCN(C)C'),
        'mol_edit.substitute_v2.0276': ([1], 2, 'N1C(Cl)=NC2=C1C=C([N+](=O)[O-])C([N+](=O)[O-])=C2'),
    }
    oid = row['origin_id']
    if oid not in special:
        return deterministic_reference_plan(row)
    removed, anchor, fragment = special[oid]
    plan = {'remove_atom_maps': removed, 'add_fragments': [
        {'smiles': fragment, 'attach_atom_index': 0, 'anchor_map': anchor, 'bond_type': 'SINGLE'}]}
    provenance = []
    if oid.endswith('.0271'):
        plan['adjust_hydrogens'] = [{'atom_map': 20, 'delta': 1}, {'atom_map': 35, 'delta': -1}]
        provenance.append('Frozen instruction explicitly specifies SEM deprotection at purine N20 and alkylation at O35.')
    if oid.endswith(('.0123', '.0276')):
        plan['fragment_hydrogen_caps'] = [{'fragment_index': 0, 'atom_index': 0, 'delta': -1}]
        provenance.append('Remove the standalone aromatic nitrogen hydrogen cap at the explicitly named N1 attachment atom.')
    candidates = [plan]
    if oid.endswith('.0276'):
        candidates = []
        for cip in ('R', 'S'):
            candidate = deepcopy(plan)
            candidate['assign_tetrahedral'] = [{'atom_map': 2, 'cip': cip}]
            candidates.append(candidate)
        provenance.append('Source map 2 is stereochemically unspecified; its absolute configuration is drawn only from immutable clean-reference GT, not inferred from source or instruction. This explicit assignment is carried unchanged into subsequent plans.')
    accepted = []
    for candidate in candidates:
        execution = execute_full_plan(row['indexed_smiles'], candidate)
        if compare_molecules(execution['product_smiles'], row['gt_smiles'])['equivalent']:
            accepted.append((candidate, execution))
    if len(accepted) != 1:
        _fail('reference_invalid', 'Explicit reference recipe did not uniquely reproduce full stereo and component GT')
    plan, execution = accepted[0]
    return {'edit_plan': plan, 'execution': execution, 'method': 'explicit_validated_full_reference_recipe',
            'reference_checks': validation, 'candidate_count': len(candidates),
            'provenance': provenance, 'reference_equivalence': compare_molecules(execution['product_smiles'], row['gt_smiles'])}
