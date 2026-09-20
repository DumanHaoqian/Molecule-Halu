"""Four fixed native name→fragment bindings with graph-verified dependencies.

The unchanged requested name is a task symbol. Its wrong graph value is the
single intentional root; chemical name/graph contradictions are deliberate.
The native/ledger comparison changes the complete expression package. It does
not isolate wording: the existing ledger does not display the requested name.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from .labels import compile_patches
from .orchestrator import project_reference
from .quality import find_answer_leakage
from .renderer import render_reasoning

INVENTORY_PATH = Path(__file__).with_name('named_binding_inventory.json')


def load_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text())


def _get(data, path):
    for key in path:
        data = data[key]
    return deepcopy(data)


def _nodes(record, facts):
    result = {}
    for name, spec in record['node_specs'].items():
        def value(side):
            if 'path' in spec:
                return _get(facts[side], spec['path'])
            return {key: _get(facts[side], path) for key, path in spec['paths'].items()}
        before, after = value('N'), value('H')
        changed = before != after
        fragment_derived = name in {'fragment_composition', 'fragment_heavy', 'product_heavy', 'product_rings',
                                    'heavy_delta', 'ring_delta', 'fragment_aromaticity', 'fragment_rings',
                                    'fragment_ring_identity', 'fragment_carbonyl_counts'}
        result[name] = {
            'before': before, 'after': after, 'changed': changed,
            'kind': 'unchanged' if not changed else 'root_error' if name == 'fragment' else 'propagated_error',
            'root_ids': ['r1'] if name == 'fragment' or fragment_derived else [],
            'parents': ['fragment'] if fragment_derived else [],
            'evidence': {'source': 'replayed_named_binding_chemistry', 'fact_spec': deepcopy(spec),
                         'label_scope': 'whole semantic value or specified relation, not per-word/per-atom truth'},
        }
    return result


def _check_counts(record, facts):
    expected = record['expected_counts']
    actual = {
        'source_heavy': facts['N']['source']['heavy_atoms'], 'source_rings': facts['N']['source']['rings'],
        'clean_fragment_heavy': facts['N']['fragment']['heavy_atoms'], 'wrong_fragment_heavy': facts['H']['fragment']['heavy_atoms'],
        'clean_product_heavy': facts['N']['product']['heavy_atoms'], 'wrong_product_heavy': facts['H']['product']['heavy_atoms'],
        'clean_product_rings': facts['N']['product']['rings'], 'wrong_product_rings': facts['H']['product']['rings'],
        'clean_heavy_delta': facts['N']['heavy_delta'], 'wrong_heavy_delta': facts['H']['heavy_delta'],
        'clean_ring_delta': facts['N']['ring_delta'], 'wrong_ring_delta': facts['H']['ring_delta'],
    }
    if expected != actual:
        raise ValueError('Inventory numerical evidence differs from executed chemistry')


def _verify_phrase_evidence(text, fragment):
    """Check the bounded composition/topology vocabulary against atom witnesses."""
    roles = fragment['role_counts']
    vocabulary = {
        'carbonyl carbon': 'carbonyl_carbon', 'carbonyl oxygen': 'carbonyl_oxygen',
        'methylene carbon': 'methylene_carbon', 'cyano carbon': 'nitrile_carbon',
        'cyano nitrogen': 'nitrile_nitrogen', 'cyclopropane carbons': 'cyclopropane_carbon',
        'phenyl carbons': 'phenyl_carbon', 'trifluoromethyl carbon': 'trifluoromethyl_carbon',
        'fluorine atoms': 'fluorine', 'carbonyl C': 'carbonyl_carbon', 'carbonyl O': 'carbonyl_oxygen',
        'phenyl C': 'phenyl_carbon', 'trifluoromethyl C': 'trifluoromethyl_carbon', 'F': 'fluorine',
    }
    checks = 0
    for phrase, role in vocabulary.items():
        for match in re.finditer(r'\b(\d+) '+re.escape(phrase)+r'\b', text):
            if int(match.group(1)) != roles.get(role, 0):
                raise ValueError(f'Composition phrase lacks graph evidence: {match.group()}')
            checks += 1
    for phrase, role in [('carbonyl C', 'carbonyl_carbon'), ('carbonyl O', 'carbonyl_oxygen'), ('methyl C', 'methyl_carbon')]:
        for match in re.finditer(r'(?<!\d )\b'+phrase+r'\b', text):
            if roles.get(role, 0) != 1:
                raise ValueError(f'Implicit single-atom phrase lacks evidence: {match.group()}')
            checks += 1
    for phrase, element in [('carbon atoms', 'C'), ('oxygen atom', 'O'), ('fluorine atoms', 'F')]:
        for match in re.finditer(r'\b(\d+) '+phrase+r'\b', text):
            if int(match.group(1)) != fragment['element_counts'].get(element, 0):
                raise ValueError(f'Element enumeration differs from graph: {match.group()}')
            checks += 1
    for match in re.finditer(r'\b(\d+)-carbon\b', text):
        if int(match.group(1)) != fragment['element_counts'].get('C', 0):
            raise ValueError('Carbon-chain descriptor count differs from graph')
        checks += 1
    if 'long aliphatic chain' in text and not (fragment['is_acyclic'] and fragment['aromatic_ring_count'] == 0 and roles.get('chain_carbon') == 17):
        raise ValueError('Original long-aliphatic-chain descriptor lacks fixed stearoyl evidence')
    if 'is acyclic' in text and not fragment['is_acyclic']:
        raise ValueError('Acyclic descriptor differs from graph')
    if 'aromatic acyl group' in text and not (fragment['aromatic_ring_count'] == 1 and roles.get('carbonyl_carbon') == 1):
        raise ValueError('Aromatic acyl descriptor differs from graph')
    if 'newly added cyclopropane ring' in text and not (fragment['ring_sizes'] == [3] and roles.get('cyclopropane_carbon') == 3):
        raise ValueError('New cyclopropane identity differs from the incoming graph')
    if 'newly added phenyl ring' in text or 'contains one phenyl ring' in text:
        if not (fragment['ring_sizes'] == [6] and fragment['aromatic_ring_sizes'] == [6] and roles.get('phenyl_carbon') == 6):
            raise ValueError('New phenyl identity differs from the incoming graph')
    return checks


def _validate_bindings(text, bindings, nodes, spans, side):
    keys = []
    for item in bindings:
        start, end = item['start'], item['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError('Invalid inventory binding offsets')
        if text[start:end] != item['text'] or item['node_id'] not in nodes:
            raise ValueError('Inventory binding differs from compiled text or semantic nodes')
        if item['node_id'] in {'fragment', 'fragment_heavy', 'product_heavy', 'product_rings', 'heavy_delta', 'ring_delta', 'source_heavy', 'source_rings', 'anchor', 'anchor_element', 'group_name', 'source_map_max'}:
            actual = nodes[item['node_id']]['before' if side == 'N' else 'after']
            if item['text'] != str(actual):
                raise ValueError('Displayed semantic value differs from executed graph evidence')
        if type(item['error_annotated']) is not bool:
            raise ValueError('Binding error flag must be a boolean')
        if item['error_annotated']:
            keys.append((start, end, item['node_id']))
    expected = [(s['start'], s['end'], s['node_id']) for s in spans]
    if sorted(keys) != sorted(expected):
        raise ValueError('Incomplete or extraneous inventory error bindings')
    if set(nodes) - {b['node_id'] for b in bindings}:
        raise ValueError('Inventory leaves semantic nodes unbound')


def _native(row, record, bundle, binding_evidence):
    from .chemistry_tools import inspect_source
    facts = deepcopy(bundle['facts'])
    max_map = max(atom['atom_map'] for atom in inspect_source(row['indexed_smiles'])['atoms'])
    for side in ('N', 'H'):
        facts[side]['source']['max_atom_map'] = max_map
    nodes = _nodes(record, facts)
    _check_counts(record, facts)
    patches = deepcopy(record['patches'])
    for patch in patches:
        for span in patch['error_spans']:
            node = nodes[span['node_id']]
            if not node['changed'] or span['kind'] != node['kind'] or span['root_ids'] != ['r1']:
                raise ValueError('Inventory annotation disagrees with actual changed semantic value')
            span['evidence'] = {'source': 'replayed_named_binding_chemistry', 'before': deepcopy(node['before']),
                                'after': deepcopy(node['after']), 'label_scope': node['evidence']['label_scope']}
            if span['node_id'] == 'fragment':
                span['evidence']['binding'] = deepcopy(binding_evidence)
    rendered = compile_patches(row['N_visible'], patches)
    if rendered['text'] != record['expected_compiled_text']:
        raise ValueError('Incomplete dependency inventory or unexpected compiled text')
    if len(rendered['spans']) != record['coverage']['error_spans']:
        raise ValueError('Inventory omitted a semantic error label')
    changed = {name for name, node in nodes.items() if node['changed']}
    if changed != set(record['coverage']['changed_nodes']) or changed != {s['node_id'] for s in rendered['spans']}:
        raise ValueError('Incomplete changed-node dependency closure')
    clean = {'text': row['N_visible'], 'spans': [], 'edits': [], 'bindings': deepcopy(record['bindings_N'])}
    rendered['bindings'] = deepcopy(record['bindings_H'])
    for side, result in [('N', clean), ('H', rendered)]:
        _validate_bindings(result['text'], result['bindings'], nodes, result['spans'], side)
        _verify_phrase_evidence(result['text'], facts[side]['fragment'])
    name = facts['N']['group_name']
    name_count = lambda text: len(re.findall(r'\b'+re.escape(name)+r'\b', text))
    if (name != facts['H']['group_name'] or name_count(clean['text']) != record['coverage']['unchanged_name_occurrences']
            or name_count(clean['text']) != name_count(rendered['text'])):
        raise ValueError('Original requested-name aliases were changed')
    if rendered['text'].count(facts['H']['fragment']['smiles']) != record['coverage']['wrong_smiles_occurrences']:
        raise ValueError('Incorrect number of repeated binding graph values')
    return clean, rendered, nodes


def render_named_binding(row, chemistry_bundle, style='native') -> dict[str, Any]:
    """Return both traces, nodes, one existing structural root and exact checks.

    Native rendering is restricted to the four declared original texts. Ledger
    rendering reuses the existing no-local template byte-for-byte. Both validate
    the complete deterministic chemistry bundle before exposing any claims.
    """
    if style not in {'native', 'ledger'}:
        raise ValueError('style must be native or ledger')
    inventory = load_inventory()
    record = inventory.get('origins', {}).get(row.get('origin_id'))
    if record is None:
        raise ValueError('Origin is outside the four-example named-binding inventory')
    for row_key, record_key in [('N_visible', 'reference_text'), ('instruction', 'instruction'), ('indexed_smiles', 'indexed_smiles')]:
        if row.get(row_key) != record[record_key]:
            raise ValueError(f'Frozen named-binding input drift: {row_key}')
    if 'N_raw' in row and project_reference(row['N_raw']) != row['N_visible']:
        raise ValueError('N_visible is not the exact answer-redacted original N')
    from .named_binding_chemistry import build_named_binding
    if chemistry_bundle != build_named_binding(row):
        raise ValueError('Supplied named-binding chemistry bundle differs from exact replay')
    roots = chemistry_bundle['roots']
    if len(roots) != 1 or roots[0]['id'] != 'r1' or roots[0]['node_id'] != 'fragment' or roots[0]['type'] != 'structural':
        raise ValueError('Named binding requires exactly the single structural fragment root')
    binding_evidence = {
        'name_symbol': chemistry_bundle['facts']['N']['group_name'],
        'before': {'smiles': chemistry_bundle['clean_plan']['add_fragments'][0]['smiles'],
                   'attach_atom_index': chemistry_bundle['clean_plan']['add_fragments'][0]['attach_atom_index']},
        'after': {'smiles': chemistry_bundle['wrong_plan']['add_fragments'][0]['smiles'],
                  'attach_atom_index': chemistry_bundle['wrong_plan']['add_fragments'][0]['attach_atom_index']},
        'root_id': 'r1', 'annotation_convention': 'Wrong graph value locates the original-name→attachment-qualified graph relation; unchanged names are task-symbol references.',
        'consistency_scope': 'Executed consequences conditional on the false binding; conventional name/graph inconsistency is the intentional root.',
    }
    if style == 'native':
        clean, wrong, nodes = _native(row, record, chemistry_bundle, binding_evidence)
    else:
        nodes = deepcopy(chemistry_bundle['ledger_nodes_H'])
        if any(k.startswith('local_environment') for k in nodes):
            raise ValueError('The diagnostic ledger must have no local product window')
        clean = render_reasoning(row['indexed_smiles'], chemistry_bundle['clean_plan'], chemistry_bundle['ledger_nodes_N'], label_errors=False)
        wrong = render_reasoning(row['indexed_smiles'], chemistry_bundle['wrong_plan'], nodes, label_errors=True)
    products = [row['gt_smiles'], chemistry_bundle['clean_execution']['product_smiles'], chemistry_bundle['wrong_execution']['product_smiles']]
    for result in (clean, wrong):
        if find_answer_leakage(result['text'], products):
            raise ValueError('Complete product leaked into named-binding reasoning')
    return {'N_render': clean, 'H_render': wrong, 'nodes': nodes, 'roots': deepcopy(roots),
            'binding_evidence': binding_evidence,
            'checks': {'status': 'pass', 'style': style, 'native_names_retained': style == 'native',
                       'ledger_names_displayed': False, 'same_fixed_chemistry': True,
                       'complete_product_excluded': True, 'native_reference_is_original_projection': style == 'native',
                       'comparison_scope': 'Original-name/source-role/carbonyl-connection expression package versus the existing ledger; not isolated wording or name effects.',
                       'annotation_scope': 'Specified semantic values and relations relative to the clean task; no per-atom or universal token-truth claim.',
                       'diagnostic_only': True, 'general_agent_generator': False}}
