"""Exact product-side connections for the bounded named-acyl diagnostic.

Select ALL atoms from the incoming operation and its original source anchor.
Replace each cut to an original source neighbor with a map-labelled dummy port.
``[*:k]`` refers to omitted source atom k; it is not a real wildcard substitution
or a hydrogen cap. The original source supplies the omitted context, including
external rings and stereochemistry. No radius, atom budget or new candidate is
introduced. Incoming runtime maps are provenance within one execution only and
are absent from displayed SMILES and clean/wrong graph comparisons.

Only ``N/H.open_port_smiles`` is intended for rendering. Evidence contains full
products for verification, never for inclusion in the model-visible trace.
The projection is not a standalone molecule; its formula/ring count must not
be substituted for product facts. No additional independent error root exists.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from rdkit import Chem

from .named_binding_chemistry import build_named_binding


SELECTION_RULE = 'all incoming-operation atoms plus the original source anchor'
PORT_SEMANTICS = (
    '[*:k] marks a single-bond continuation to omitted original source atom k; '
    'it is an open boundary, not a real atom substitution or hydrogen cap. '
    'Omitted source connectivity and stereochemistry remain those of the original source.'
)


def _record(checks: list, name: str, condition: bool, evidence: Any = None) -> None:
    item = {'check': name, 'status': 'pass' if condition else 'fail'}
    if evidence is not None:
        item['evidence'] = evidence
    checks.append(item)
    if not condition:
        raise ValueError('anchored connections: ' + name)


def _mol(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('anchored connections: invalid molecular graph')
    return molecule


def _canonical(molecule: Chem.Mol, *, mapped: bool = True) -> str:
    molecule = Chem.Mol(molecule)
    if not mapped:
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _atoms(molecule: Chem.Mol) -> dict[int, Chem.Atom]:
    return {atom.GetAtomMapNum(): atom for atom in molecule.GetAtoms()}


def _atom_state(atom: Chem.Atom) -> dict:
    # Ring membership is deliberately omitted: a ring may close outside a port.
    return {
        'element': atom.GetSymbol(), 'atomic_number': atom.GetAtomicNum(),
        'formal_charge': atom.GetFormalCharge(), 'isotope': atom.GetIsotope(),
        'aromatic': atom.GetIsAromatic(), 'explicit_hydrogens': atom.GetNumExplicitHs(),
        'total_hydrogens': atom.GetTotalNumHs(), 'no_implicit': atom.GetNoImplicit(),
        'degree': atom.GetDegree(), 'total_valence': atom.GetTotalValence(),
        'chiral_tag': str(atom.GetChiralTag()), 'radical_electrons': atom.GetNumRadicalElectrons(),
    }


def _ledger(molecule: Chem.Mol, keep: set[int] | None = None) -> dict:
    atoms = _atoms(molecule)
    keep = set(atoms) if keep is None else keep
    return {
        'atoms': {str(key): _atom_state(atoms[key]) for key in sorted(keep)},
        'bonds': sorted([
            min(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()),
            max(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()),
            str(bond.GetBondType()), str(bond.GetStereo()),
        ] for bond in molecule.GetBonds()
            if bond.GetBeginAtom().GetAtomMapNum() in keep
            and bond.GetEndAtom().GetAtomMapNum() in keep),
    }


def _without_ports(local: Chem.Mol, *, cap: bool) -> Chem.Mol:
    """A leakage-check-only projection; neither variant is rendered."""
    result = Chem.RWMol(local)
    if cap:
        additions: dict[int, int] = {}
        for atom in result.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbor = atom.GetNeighbors()[0]
                additions[neighbor.GetIdx()] = additions.get(neighbor.GetIdx(), 0) + 1
        for index, count in additions.items():
            atom = result.GetAtomWithIdx(index)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + count)
            atom.SetNoImplicit(True)
    for index in sorted((atom.GetIdx() for atom in result.GetAtoms()
                         if atom.GetAtomicNum() == 0), reverse=True):
        result.RemoveAtom(index)
    molecule = result.GetMol()
    if cap:
        Chem.SanitizeMol(molecule)
    else:
        molecule.UpdatePropertyCache(strict=False)
    return molecule


def _project(source: Chem.Mol, execution: dict, complete_components: set[str],
             label: str, checks: list) -> dict:
    product = _mol(execution['mapped_product_smiles'])
    source_atoms, product_atoms = _atoms(source), _atoms(product)
    operations = [operation for operation in execution['operations']
                  if operation['operation'] == 'add_fragment']
    _record(checks, label + '_one_incoming_operation', len(operations) == 1)
    operation = operations[0]
    anchor = operation['anchor_map']
    incoming = set(operation['new_atom_maps'])
    retained = incoming | {anchor}
    _record(checks, label + '_unique_phase_qualified_atom_maps',
            len(source_atoms) == source.GetNumAtoms()
            and len(product_atoms) == product.GetNumAtoms()
            and all(key > 0 for key in product_atoms)
            and set(source_atoms) <= set(product_atoms)
            and incoming == set(product_atoms) - set(source_atoms)
            and len(incoming) == len(operation['new_atom_maps'])
            and anchor in source_atoms and operation['attachment_atom_map'] in incoming)
    _record(checks, label + '_incoming_all_heavy_and_no_dummy_atoms',
            all(atom.GetAtomicNum() > 1 for atom in product_atoms.values()))

    ports = []
    boundary_bonds = []
    for bond in product.GetBonds():
        first, second = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if (first in retained) == (second in retained):
            continue
        inside, outside = (first, second) if first in retained else (second, first)
        ports.append({'anchor_map': inside, 'outside_source_map': outside,
                      'outside_element': product_atoms[outside].GetSymbol(),
                      'bond_type': str(bond.GetBondType())})
        boundary_bonds.append(bond)
    ports.sort(key=lambda item: item['outside_source_map'])
    outside_maps = {port['outside_source_map'] for port in ports}
    expected = {atom.GetAtomMapNum() for atom in source_atoms[anchor].GetNeighbors()}
    _record(checks, label + '_unambiguous_fixed_source_ports',
            bool(ports) and len(ports) == len(outside_maps) and outside_maps == expected
            and outside_maps <= set(source_atoms) and not outside_maps & retained
            and all(port['anchor_map'] == anchor for port in ports), ports)
    _record(checks, label + '_only_single_boundary_bonds',
            all(bond.GetBondType() == Chem.BondType.SINGLE
                and bond.GetStereo() == Chem.BondStereo.STEREONONE
                and bond.GetBondDir() == Chem.BondDir.NONE for bond in boundary_bonds))
    boundary_maps = outside_maps | {anchor}
    stereo_safe = all(product_atoms[key].GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                      for key in boundary_maps)
    for bond in product.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            defining_maps = {product.GetAtomWithIdx(index).GetAtomMapNum()
                             for index in bond.GetStereoAtoms()}
            defining_maps |= {bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()}
            stereo_safe = stereo_safe and not defining_maps & boundary_maps
    _record(checks, label + '_boundary_has_no_specified_stereo', stereo_safe)

    # Keeping each port in the old neighbor's atom position preserves ordering.
    local = Chem.RWMol(product)
    for key in outside_maps:
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(key)
        dummy.SetNoImplicit(True)
        local.ReplaceAtom(product_atoms[key].GetIdx(), dummy)
    for bond in list(local.GetBonds()):
        first, second = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        touches_port = first in outside_maps or second in outside_maps
        joins_anchor = ((first == anchor and second in outside_maps)
                        or (second == anchor and first in outside_maps))
        if touches_port and not joins_anchor:
            local.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    for index in sorted((atom.GetIdx() for atom in local.GetAtoms()
                         if atom.GetAtomMapNum() not in retained | outside_maps), reverse=True):
        local.RemoveAtom(index)
    local = local.GetMol()
    Chem.SanitizeMol(local)
    local_atoms = _atoms(local)
    _record(checks, label + '_real_atom_H_valence_and_bond_ledger_exact',
            _ledger(local, retained) == _ledger(product, retained))
    _record(checks, label + '_anchor_product_state_unchanged',
            _atom_state(local_atoms[anchor]) == _atom_state(product_atoms[anchor]))
    _record(checks, label + '_ports_are_H_free_single_bound_dummies',
            all(local_atoms[key].GetAtomicNum() == 0
                and local_atoms[key].GetDegree() == 1
                and local_atoms[key].GetTotalNumHs() == 0
                and local_atoms[key].GetFormalCharge() == 0 for key in outside_maps))
    _record(checks, label + '_local_connected_and_no_radicals',
            len(Chem.GetMolFrags(local)) == 1
            and not any(atom.GetNumRadicalElectrons() for atom in local.GetAtoms()))

    displayed = Chem.Mol(local)
    for atom in displayed.GetAtoms():
        if atom.GetAtomMapNum() in incoming:
            atom.SetAtomMapNum(0)
    text = _canonical(displayed)
    reparsed = _mol(text)
    _record(checks, label + '_display_roundtrip_exact', _canonical(reparsed) == text)
    _record(checks, label + '_only_source_anchor_and_port_maps_displayed',
            sorted(atom.GetAtomMapNum() for atom in reparsed.GetAtoms() if atom.GetAtomMapNum())
            == sorted(outside_maps | {anchor}))

    real = _without_ports(local, cap=False)
    capped = _without_ports(local, cap=True)
    raw_match = _canonical(real, mapped=False) in complete_components
    cap_match = _canonical(capped, mapped=False) in complete_components
    _record(checks, label + '_not_any_complete_product_component',
            _canonical(displayed, mapped=False) not in complete_components)
    _record(checks, label + '_dummy_free_projection_not_complete_product_component', not raw_match)
    _record(checks, label + '_H_capped_projection_not_complete_product_component', not cap_match,
            {'reason': 'A dummy must not be the only obstacle to a complete product component.'})

    # Recover omitted atoms/bonds from S, not by copying P's outside region.
    context = Chem.RWMol(source)
    context.RemoveAtom(source_atoms[anchor].GetIdx())
    restored = Chem.RWMol(Chem.CombineMols(context.GetMol(), real))
    restored_atoms = _atoms(restored)
    for port in ports:
        restored.AddBond(restored_atoms[anchor].GetIdx(),
                         restored_atoms[port['outside_source_map']].GetIdx(), Chem.BondType.SINGLE)
    restored = restored.GetMol()
    Chem.SanitizeMol(restored)
    restored_smiles = _canonical(restored)
    _record(checks, label + '_restore_original_context_exact_mapped_product',
            restored_smiles == _canonical(product))
    _record(checks, label + '_restored_atom_H_bond_stereo_ledger_exact',
            _ledger(_mol(restored_smiles)) == _ledger(_mol(_canonical(product))))
    _record(checks, label + '_strict_physical_subset',
            len(retained) < product.GetNumHeavyAtoms() and bool(set(source_atoms) - {anchor}))
    return {
        'open_port_smiles': text, 'anchor_map': anchor,
        'anchor_element': product_atoms[anchor].GetSymbol(),
        'anchor_product_state': _atom_state(product_atoms[anchor]), 'ports': ports,
        'incoming_atom_maps': sorted(incoming),
        'incoming_attachment_atom_map': operation['attachment_atom_map'],
        'retained_real_atom_maps': sorted(retained),
        'real_heavy_atoms': len(retained), 'port_count': len(ports),
        'evidence': {
            'source_maps_omitted': sorted(set(source_atoms) - {anchor}),
            'real_atom_and_bond_ledger': _ledger(product, retained),
            'reconstructed_mapped_product_smiles': restored_smiles,
            'dummy_free_component_match': raw_match, 'H_capped_component_match': cap_match,
            'incoming_maps_are_execution_local': True,
        },
    }


def build_anchored_connections(row: dict[str, Any], chemistry_bundle: dict[str, Any]) -> dict[str, Any]:
    """Return N/H fixed-anchor projections and exact verification evidence.

    The supplied chemistry bundle must equal the full deterministic
    ``build_named_binding(row)`` result. Ambiguous maps, unmatched provenance,
    multi-anchor boundaries, non-single cuts, specified boundary stereo, leaked
    complete components or unequal source ports raise ValueError. Chemical
    symmetry elsewhere is harmless because original-source port labels fix the
    correspondence; no unlabeled substructure matching chooses an anchor.
    Neither inputs nor files are mutated. No model outputs are consulted.
    """
    checks: list[dict] = []
    expected = build_named_binding(row)
    _record(checks, 'exact_supplied_chemistry_bundle', chemistry_bundle == expected)
    source = _mol(row['indexed_smiles'])
    complete_components = set()
    for smiles in [row['gt_smiles'], expected['clean_execution']['product_smiles'],
                   expected['wrong_execution']['product_smiles']]:
        for component in Chem.GetMolFrags(_mol(smiles), asMols=True, sanitizeFrags=True):
            complete_components.add(_canonical(component, mapped=False))
    clean = _project(source, expected['clean_execution'], complete_components, 'N', checks)
    wrong = _project(source, expected['wrong_execution'], complete_components, 'H', checks)
    comparison = {
        'same_anchor': clean['anchor_map'] == wrong['anchor_map'],
        'same_ports': clean['ports'] == wrong['ports'],
        'same_anchor_state': clean['anchor_product_state'] == wrong['anchor_product_state'],
        'different_at_same_anchor': clean['open_port_smiles'] != wrong['open_port_smiles'],
        'new_atom_maps_compared': False,
    }
    for name in ['same_anchor', 'same_ports', 'same_anchor_state', 'different_at_same_anchor']:
        _record(checks, 'anchored_comparison_' + name, comparison[name])
    return {
        'N': clean, 'H': wrong, 'comparison': comparison,
        'selection_rule': SELECTION_RULE, 'port_semantics': PORT_SEMANTICS,
        'checks': {'status': 'pass', 'checks': checks},
        'chemistry_reference': {key: deepcopy(expected[key]) for key in
                                ['clean_plan', 'wrong_plan', 'clean_execution', 'wrong_execution', 'roots']},
    }
