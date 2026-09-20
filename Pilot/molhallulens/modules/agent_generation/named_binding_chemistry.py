"""Bounded chemistry evidence for the four named-acyl diagnostic patterns.

The original instruction and validated reference determine the source role and
intended group. Only its incoming graph is replaced with a fixed para-CF3 benzoyl
moiety. GT is used to certify the clean reference and reject a no-op, never to
rank alternative corruptions. No files, model outputs, or network are accessed.

``facts[side].group_name`` remains the intended task symbol on both sides.
``represented_group_name`` describes the actual conditional graph. A native
name-to-graph binding intervention must not silently replace the original symbol
with that actual wrong-graph name. The one existing structural root is r1/fragment.

Fragment ``formula`` and ``element_counts`` describe the standalone H-capped
SMILES. ``attached_formula``/``attached_element_counts`` describe the group after
its carbonyl H is consumed. Atom role tags overlap; elemental totals do not.
Unsupported names, source roles, references, valence or stereo edits fail closed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .chemistry_tools import apply_edit_plan, compare_molecules, describe_fragment, inspect_source
from .orchestrator import build_nodes, semantic_roots
from .quality import deterministic_reference_plan

WRONG_FRAGMENT_SMILES = 'C(=O)c1ccc(C(F)(F)F)cc1'
WRONG_GROUP_NAME = '4-(trifluoromethyl)benzoyl'

# This is a declared mechanism-prototype scope, not a general reaction parser.
_PATTERNS = {
    'please acylate the secondary amine of the piperidine ring with a cyanoacetyl group.':
        ('cyanoacetyl', 'C(=O)CC#N', 'secondary piperidine amine'),
    'please acylate the secondary hydroxyl group with a stearoyl group.':
        ('stearoyl', 'C(=O)' + 'C' * 17, 'secondary hydroxyl'),
    'please acylate the primary amine of the benzothiazole ring with a cyclopropanecarbonyl group.':
        ('cyclopropanecarbonyl', 'C(=O)C1CC1', 'benzothiazole exocyclic primary amine'),
    'please acetylate the primary amino group on the phenyl ring.':
        ('acetyl', 'C(=O)C', 'primary aniline amine'),
}
_ROLE_NAMES = (
    'carbonyl_carbon', 'carbonyl_oxygen', 'nitrile_carbon', 'nitrile_nitrogen',
    'chain_carbon', 'methylene_carbon', 'methyl_carbon', 'aromatic_carbon',
    'phenyl_carbon', 'saturated_ring_carbon', 'cyclopropane_carbon',
    'trifluoromethyl_carbon', 'fluorine',
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('named binding: ' + message)


def _record(checks: list, name: str, condition: bool, evidence: Any = None) -> None:
    item = {'check': name, 'status': 'pass' if condition else 'fail'}
    if evidence is not None:
        item['evidence'] = evidence
    checks.append(item)
    _require(condition, name)


def _mol(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    _require(molecule is not None, 'invalid molecular SMILES')
    return molecule


def _canonical(molecule: Chem.Mol) -> str:
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)


def _elements(molecule: Chem.Mol, indices=None) -> dict[str, int]:
    counts = Counter({'C': 0, 'N': 0, 'O': 0, 'F': 0, 'H': 0})
    for index in range(molecule.GetNumAtoms()) if indices is None else indices:
        atom = molecule.GetAtomWithIdx(index)
        counts[atom.GetSymbol()] += 1
        counts['H'] += atom.GetTotalNumHs()
    return dict(sorted(counts.items()))


def _formula(counts: dict[str, int]) -> str:
    names = ['C', 'H'] + sorted(name for name in counts if name not in {'C', 'H'})
    return ''.join(name + (str(counts[name]) if counts[name] != 1 else '')
                   for name in names if counts.get(name))


def _rings(molecule: Chem.Mol) -> list[dict[str, Any]]:
    return [{'atom_indices': list(ring), 'size': len(ring),
             'aromatic': all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring),
             'element_counts': dict(sorted(Counter(molecule.GetAtomWithIdx(i).GetSymbol()
                                                   for i in ring).items()))}
            for ring in molecule.GetRingInfo().AtomRings()]


def _descriptors(molecule: Chem.Mol) -> dict[str, Any]:
    rings = _rings(molecule)
    return {'heavy_atoms': molecule.GetNumHeavyAtoms(), 'rings': len(rings),
            'formula': rdMolDescriptors.CalcMolFormula(molecule),
            'element_counts': _elements(molecule), 'formal_charge': Chem.GetFormalCharge(molecule),
            'component_count': len(Chem.GetMolFrags(molecule)),
            'ring_sizes': sorted(ring['size'] for ring in rings),
            'aromatic_ring_count': sum(ring['aromatic'] for ring in rings),
            'aromatic_ring_sizes': sorted(ring['size'] for ring in rings if ring['aromatic'])}


def _carbonyl(molecule: Chem.Mol, index: int) -> bool:
    atom = molecule.GetAtomWithIdx(index)
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetAtomicNum() == 8
        for bond in atom.GetBonds())


def _fragment_facts(smiles: str, attachment: int) -> dict[str, Any]:
    # Public fragment validation rejects disconnected/mapped/radical ingredients.
    describe_fragment(smiles)
    molecule = _mol(smiles)
    _require(type(attachment) is int and 0 <= attachment < molecule.GetNumAtoms(), 'invalid carbonyl attachment index')
    _require(_carbonyl(molecule, attachment), 'named fragment must attach through its carbonyl carbon')
    attached = molecule.GetAtomWithIdx(attachment)
    _require(attached.GetTotalNumHs() == 1 and attached.GetDegree() == 2,
             'named fragment carbonyl must have exactly one removable cap hydrogen')
    rings = _rings(molecule)
    role_atoms = []
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        carbonyl = _carbonyl(molecule, index)
        nitrile = any(bond.GetBondType() == Chem.BondType.TRIPLE and
                      {atom.GetAtomicNum(), bond.GetOtherAtom(atom).GetAtomicNum()} == {6, 7}
                      for bond in atom.GetBonds())
        in_rings = [ring for ring in rings if index in ring['atom_indices']]
        tags = []
        if carbonyl:
            tags.append('carbonyl_carbon')
        if atom.GetAtomicNum() == 8 and any(bond.GetBondType() == Chem.BondType.DOUBLE and
                                           bond.GetOtherAtom(atom).GetAtomicNum() == 6 for bond in atom.GetBonds()):
            tags.append('carbonyl_oxygen')
        if nitrile:
            tags.append('nitrile_carbon' if atom.GetAtomicNum() == 6 else 'nitrile_nitrogen')
        if atom.GetAtomicNum() == 6:
            if not atom.IsInRing() and not carbonyl and not nitrile and not atom.GetIsAromatic():
                tags.append('chain_carbon')
            if atom.GetTotalNumHs() == 2:
                tags.append('methylene_carbon')
            if atom.GetTotalNumHs() == 3:
                tags.append('methyl_carbon')
            if atom.GetIsAromatic():
                tags.append('aromatic_carbon')
            if any(ring['aromatic'] and ring['size'] == 6 and ring['element_counts'] == {'C': 6} for ring in in_rings):
                tags.append('phenyl_carbon')
            if atom.IsInRing() and not atom.GetIsAromatic():
                tags.append('saturated_ring_carbon')
            if any(not ring['aromatic'] and ring['size'] == 3 and ring['element_counts'] == {'C': 3} for ring in in_rings):
                tags.append('cyclopropane_carbon')
            if sum(neighbor.GetAtomicNum() == 9 for neighbor in atom.GetNeighbors()) == 3:
                tags.append('trifluoromethyl_carbon')
        if atom.GetAtomicNum() == 9:
            tags.append('fluorine')
        role_atoms.append({'atom_index': index, 'element': atom.GetSymbol(),
                           'hydrogens': atom.GetTotalNumHs(), 'aromatic': atom.GetIsAromatic(),
                           'roles': tags, 'neighbor_indices': [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]})
    facts = _descriptors(molecule)
    counts = deepcopy(facts['element_counts'])
    counts['H'] -= 1
    facts.update(smiles=smiles, canonical_smiles=_canonical(molecule),
                 standalone_formula=facts['formula'], attached_formula=_formula(counts),
                 attached_element_counts=counts, carbonyl_attachment_index=attachment,
                 carbonyl_attachment_index_1based=attachment + 1,
                 carbonyl_cap_hydrogens=1, atom_roles=role_atoms,
                 role_counts={name: sum(name in atom['roles'] for atom in role_atoms) for name in _ROLE_NAMES},
                 ring_evidence=rings, is_acyclic=not rings)
    return facts


def _source_role(source: Chem.Mol, anchor_map: int, group: str) -> dict[str, Any]:
    atoms = {atom.GetAtomMapNum(): atom for atom in source.GetAtoms()}
    _require(type(anchor_map) is int and anchor_map in atoms, 'source role has no unique mapped anchor')
    anchor = atoms[anchor_map]
    neighbors = list(anchor.GetNeighbors())
    rings = [ring for ring in source.GetRingInfo().AtomRings() if anchor.GetIdx() in ring]
    valid = anchor.GetFormalCharge() == 0 and not anchor.GetIsAromatic()
    if group == 'cyanoacetyl':
        valid = valid and anchor.GetSymbol() == 'N' and anchor.GetDegree() == 2 and anchor.GetTotalNumHs() == 1
        valid = valid and not any(_carbonyl(source, atom.GetIdx()) for atom in neighbors)
        valid = valid and any(
            len(ring) == 6 and
            Counter(source.GetAtomWithIdx(i).GetSymbol() for i in ring) == Counter({'C': 5, 'N': 1}) and
            all(source.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType() == Chem.BondType.SINGLE
                for i in range(len(ring)))
            for ring in rings)
    elif group == 'stearoyl':
        valid = valid and anchor.GetSymbol() == 'O' and len(neighbors) == 1 and anchor.GetTotalNumHs() == 1
        if valid:
            carbon = neighbors[0]
            valid = carbon.GetAtomicNum() == 6 and carbon.GetHybridization() == Chem.HybridizationType.SP3 and carbon.GetTotalNumHs() == 1 and sum(atom.GetAtomicNum() == 6 for atom in carbon.GetNeighbors()) == 2
    else:
        valid = valid and anchor.GetSymbol() == 'N' and len(neighbors) == 1 and anchor.GetTotalNumHs() == 2
        if valid:
            ring_atom = neighbors[0]
            local_rings = [ring for ring in source.GetRingInfo().AtomRings() if ring_atom.GetIdx() in ring]
            valid = ring_atom.GetAtomicNum() == 6 and ring_atom.GetIsAromatic()
            if group == 'acetyl':
                valid = valid and any(len(ring) == 6 and all(source.GetAtomWithIdx(i).GetAtomicNum() == 6 and source.GetAtomWithIdx(i).GetIsAromatic() for i in ring) for ring in local_rings)
            else:
                # A benzothiazole five-ring has N/S and shares its C-C edge with a phenyl six-ring.
                five_rings = [set(ring) for ring in local_rings if len(ring) == 5 and Counter(source.GetAtomWithIdx(i).GetSymbol() for i in ring) == Counter({'C': 3, 'N': 1, 'S': 1})]
                benzene = [set(ring) for ring in source.GetRingInfo().AtomRings() if len(ring) == 6 and all(source.GetAtomWithIdx(i).GetAtomicNum() == 6 and source.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
                valid = valid and any(len(five & six) == 2 for five in five_rings for six in benzene)
    _require(valid, 'source role does not match the declared acylation instruction')
    return {'atom_map': anchor_map, 'element': anchor.GetSymbol(), 'hydrogens': anchor.GetTotalNumHs(),
            'formal_charge': anchor.GetFormalCharge(),
            'neighbors': [{'atom_map': atom.GetAtomMapNum(), 'element': atom.GetSymbol(),
                           'aromatic': atom.GetIsAromatic(), 'hydrogens': atom.GetTotalNumHs()}
                          for atom in neighbors],
            'ring_atom_maps': [[source.GetAtomWithIdx(i).GetAtomMapNum() for i in ring] for ring in rings]}


def _bond_map(molecule: Chem.Mol) -> dict:
    return {tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))): bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()}


def _atom_attributes(atom: Chem.Atom) -> tuple:
    return atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetIsAromatic()


def _tetra(atom: Chem.Atom) -> int:
    ids = [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]
    parity = (-1) ** sum(ids[i] > ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids)))
    return (1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1) * parity


def _double_stereo(molecule: Chem.Mol, bond: Chem.Bond):
    sign = {Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
            Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1}.get(bond.GetStereo())
    references = list(bond.GetStereoAtoms())
    if sign is None or len(references) != 2:
        return None
    ends = [bond.GetBeginAtom(), bond.GetEndAtom()]
    neighborhoods = {}
    for end, opposite, reference in zip(ends, reversed(ends), references):
        neighbors = sorted(atom.GetAtomMapNum() for atom in end.GetNeighbors() if atom.GetIdx() != opposite.GetIdx())
        ref_map = molecule.GetAtomWithIdx(reference).GetAtomMapNum()
        if ref_map not in neighbors:
            return None
        sign *= 1 if ref_map == neighbors[0] else -1
        neighborhoods[end.GetAtomMapNum()] = neighbors
    return sign, sorted(neighborhoods.items())


def _verify_execution(source: Chem.Mol, operation: dict, execution: dict,
                      fragment_facts: dict, label: str, checks: list) -> Chem.Mol:
    product = _mol(execution['mapped_product_smiles'])
    before = {atom.GetAtomMapNum(): atom for atom in source.GetAtoms()}
    after = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms()}
    original_ids = set(before)
    start = max(original_ids) + 1
    fragment = _mol(operation['smiles'])
    new_ids = set(range(start, start + fragment.GetNumAtoms()))
    anchor = operation['anchor_map']
    attach = operation['attach_atom_index']
    _record(checks, label + '_complete_source_and_fragment_map_set', set(after) == original_ids | new_ids and len(after) == product.GetNumAtoms())
    _record(checks, label + '_source_atom_attributes_unchanged', all(_atom_attributes(before[key]) == _atom_attributes(after[key]) for key in original_ids))
    _record(checks, label + '_source_H_ledger', all(after[key].GetTotalNumHs() == before[key].GetTotalNumHs() - (1 if key == anchor else 0) for key in original_ids))
    expected_bonds = _bond_map(source)
    expected_bonds.update({tuple(sorted((start + bond.GetBeginAtomIdx(), start + bond.GetEndAtomIdx()))): bond.GetBondTypeAsDouble() for bond in fragment.GetBonds()})
    expected_bonds[tuple(sorted((anchor, start + attach)))] = 1.0
    _record(checks, label + '_all_bonds_exact', _bond_map(product) == expected_bonds)
    _record(checks, label + '_fragment_atom_and_H_ledger', all(
        _atom_attributes(after[start + atom.GetIdx()]) == _atom_attributes(atom) and
        after[start + atom.GetIdx()].GetTotalNumHs() == atom.GetTotalNumHs() - (1 if atom.GetIdx() == attach else 0)
        for atom in fragment.GetAtoms()))
    _record(checks, label + '_attached_moiety_composition', _elements(product, [after[key].GetIdx() for key in new_ids]) == fragment_facts['attached_element_counts'])
    _record(checks, label + '_components_and_no_radicals', len(Chem.GetMolFrags(product)) == len(Chem.GetMolFrags(source)) and not any(atom.GetNumRadicalElectrons() for atom in product.GetAtoms()))
    _record(checks, label + '_mapped_and_unmapped_product_equal', _canonical(product) == _canonical(_mol(execution['product_smiles'])))
    actual = _descriptors(product)
    _record(checks, label + '_product_descriptors', all(actual[key] == execution[key] for key in ['heavy_atoms', 'rings', 'formula', 'formal_charge']))
    for key, atom in before.items():
        retained = after[key]
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            _record(checks, label + f'_no_invented_stereo_map_{key}', retained.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED)
        else:
            neighbors = {neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()}
            _record(checks, label + f'_preserved_stereo_map_{key}', neighbors == {neighbor.GetAtomMapNum() for neighbor in retained.GetNeighbors()} and _tetra(atom) == _tetra(retained))
    for bond in source.GetBonds():
        if bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
            continue
        ends = [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()]
        retained = product.GetBondBetweenAtoms(*(after[key].GetIdx() for key in ends))
        signature = _double_stereo(source, bond)
        _record(checks, label + '_preserved_double_stereo_' + '_'.join(map(str, ends)), signature is not None and signature == _double_stereo(product, retained))
    return product


def build_named_binding(row: dict[str, Any]) -> dict[str, Any]:
    """Build both factual states and one controlled name-to-graph binding error.

    Input is an ordinary parsed benchmark row (instruction, mapped source, GT,
    parsed ``state``, and original N_raw). Four literal instruction patterns are
    supported; the module does not look at origin IDs or outcome metadata.
    Invalid scope/reference/chemistry raises ValueError. No fallback repairs occur.
    """
    instruction = ' '.join(row.get('instruction', '').split()).casefold()
    _require(row.get('subtask') == 'add' and instruction in _PATTERNS, 'outside the four declared acylation patterns (scope)')
    group_name, expected_fragment, role_name = _PATTERNS[instruction]
    if row.get('raw_record'):
        _require(all(row[key] == row['raw_record'][key] for key in ['instruction', 'indexed_smiles', 'gt_smiles']), 'original question or GT differs from raw reference')
    source = _mol(row['indexed_smiles'])
    maps = [atom.GetAtomMapNum() for atom in source.GetAtoms()]
    _require(maps and all(key > 0 for key in maps) and len(set(maps)) == len(maps), 'source requires unique explicit atom maps')
    state = row.get('state', {})
    fragment_smiles = state.get('step2_frag_smiles', '')
    _require(compare_molecules(fragment_smiles, expected_fragment)['equivalent'], 'named fragment does not match the original requested group')
    source_role = _source_role(source, state.get('step1_anchor_idx'), group_name)
    reference = deterministic_reference_plan(row)
    clean_plan = deepcopy(reference['edit_plan'])
    _require(len(clean_plan.get('add_fragments', [])) == 1 and all(not clean_plan.get(key) for key in ['remove_atom_maps', 'remove_bonds', 'add_bonds', 'change_bonds']), 'clean reference must be a single N/O acylation without source skeleton changes')
    operation = clean_plan['add_fragments'][0]
    _require(operation['anchor_map'] == source_role['atom_map'] and operation['bond_type'] == 'SINGLE', 'clean reference source attachment differs')
    _require(clean_plan.get('adjust_hydrogens') == [{'atom_map': operation['anchor_map'], 'delta': -1}], 'clean reference H adjustment differs from N/O acylation')
    clean_fragment = _fragment_facts(operation['smiles'], operation['attach_atom_index'])
    wrong_fragment = _fragment_facts(WRONG_FRAGMENT_SMILES, 0)
    wrong_plan = deepcopy(clean_plan)
    wrong_plan['add_fragments'][0].update(smiles=WRONG_FRAGMENT_SMILES, attach_atom_index=0)
    clean_execution = apply_edit_plan(row['indexed_smiles'], clean_plan)
    wrong_execution = apply_edit_plan(row['indexed_smiles'], wrong_plan)
    checks = []
    _record(checks, 'clean_execution_matches_reference_and_GT', clean_execution == reference['execution'] and compare_molecules(clean_execution['product_smiles'], row['gt_smiles'])['equivalent'])
    _record(checks, 'wrong_product_is_distinct_from_GT', not compare_molecules(wrong_execution['product_smiles'], row['gt_smiles'])['equivalent'])
    _record(checks, 'reference_source_role', True, source_role)
    _record(checks, 'named_clean_fragment_matches_instruction', True, {'group_name': group_name, 'smiles': operation['smiles']})
    _record(checks, 'only_incoming_graph_and_its_carbonyl_index_change',
            {key: value for key, value in clean_plan.items() if key != 'add_fragments'} == {key: value for key, value in wrong_plan.items() if key != 'add_fragments'} and
            {key: value for key, value in operation.items() if key not in ['smiles', 'attach_atom_index']} == {key: value for key, value in wrong_plan['add_fragments'][0].items() if key not in ['smiles', 'attach_atom_index']})
    represented = _mol(WRONG_FRAGMENT_SMILES)
    carbonyl = represented.GetAtomWithIdx(0)
    ring_neighbor = next(atom for atom in carbonyl.GetNeighbors() if atom.GetIsAromatic())
    cf3_index = next(atom['atom_index'] for atom in wrong_fragment['atom_roles'] if 'trifluoromethyl_carbon' in atom['roles'])
    cf3_ring_neighbor = next(atom for atom in represented.GetAtomWithIdx(cf3_index).GetNeighbors() if atom.GetIsAromatic())
    _record(checks, 'wrong_group_is_para_CF3_benzoyl_at_carbonyl_index_0',
            wrong_fragment['element_counts'] == {'C': 8, 'F': 3, 'H': 5, 'N': 0, 'O': 1} and
            wrong_fragment['ring_sizes'] == wrong_fragment['aromatic_ring_sizes'] == [6] and
            len(Chem.GetShortestPath(represented, ring_neighbor.GetIdx(), cf3_ring_neighbor.GetIdx())) - 1 == 3 and
            wrong_fragment['standalone_formula'] == 'C8H5F3O' and wrong_fragment['attached_formula'] == 'C8H4F3O')
    clean_product = _verify_execution(source, operation, clean_execution, clean_fragment, 'N', checks)
    wrong_product = _verify_execution(source, wrong_plan['add_fragments'][0], wrong_execution, wrong_fragment, 'H', checks)
    ca = {atom.GetAtomMapNum(): atom for atom in clean_product.GetAtoms()}
    ha = {atom.GetAtomMapNum(): atom for atom in wrong_product.GetAtoms()}
    _record(checks, 'same_original_source_region_in_both_products', all(_atom_attributes(ca[key]) == _atom_attributes(ha[key]) and ca[key].GetTotalNumHs() == ha[key].GetTotalNumHs() for key in maps))
    roots = semantic_roots(clean_plan, wrong_plan)
    _record(checks, 'one_structural_fragment_binding_root', roots == [{'id': 'r1', 'node_id': 'fragment', 'before': operation['smiles'], 'after': WRONG_FRAGMENT_SMILES, 'type': 'structural'}])
    source_facts = _descriptors(source)
    fact_states = {}
    for side, fragment, product, op, represented_name in [
        ('N', clean_fragment, clean_product, operation, group_name),
        ('H', wrong_fragment, wrong_product, wrong_plan['add_fragments'][0], WRONG_GROUP_NAME),
    ]:
        product_facts = _descriptors(product)
        fact_states[side] = {'group_name': group_name, 'represented_group_name': represented_name,
            'source': deepcopy(source_facts), 'fragment': fragment, 'product': product_facts,
            'attachment': {'anchor_map': op['anchor_map'], 'source_element': source_role['element'],
                'source_role': role_name, 'source_role_evidence': deepcopy(source_role),
                'fragment_atom_index': op['attach_atom_index'], 'fragment_atom_role': 'carbonyl_carbon',
                'bond_type': 'SINGLE', 'source_hydrogen_delta': -1, 'fragment_cap_hydrogen_delta': -1,
                'product_linkage': 'ester' if source_role['element'] == 'O' else 'amide'},
            'heavy_delta': product_facts['heavy_atoms'] - source_facts['heavy_atoms'],
            'ring_delta': product_facts['rings'] - source_facts['rings']}
    ledger_n = build_nodes(row, clean_plan, clean_plan, [], clean_execution, clean_execution, include_local_environment=False)
    ledger_h = build_nodes(row, clean_plan, wrong_plan, roots, clean_execution, wrong_execution, include_local_environment=False)
    _record(checks, 'no_local_or_numeric_error_nodes', all(not key.startswith('local_environment') for key in ledger_h) and all(root['type'] == 'structural' for root in roots))
    return {'clean_plan': clean_plan, 'clean_execution': clean_execution,
            'wrong_plan': wrong_plan, 'wrong_execution': wrong_execution, 'roots': roots,
            'ledger_nodes_N': ledger_n, 'ledger_nodes_H': ledger_h, 'facts': fact_states,
            'checks': {'status': 'pass', 'checks': checks},
            'reference_provenance': {'method': reference['method'], 'candidate_count': reference['candidate_count'],
                                     'attachment_alternatives': reference['attachment_alternatives'],
                                     'reference_checks': reference['reference_checks']}}
