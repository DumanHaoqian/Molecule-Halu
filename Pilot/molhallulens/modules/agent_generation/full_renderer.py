"""Deterministic full-corpus task interpretation rendering.

N is the caller's frozen original visible reasoning. H uses a canonical template,
so this design has a declared style confound and is not v17 byte-style parity.
The caller supplies independently verified clean and altered execution facts.
"""
from copy import deepcopy
import json

from .orchestrator import semantic_roots, build_nodes
from .renderer import render_reasoning
from .labels import compile_patches, label_tokens
from .quality import find_answer_leakage
from .chemistry_tools import compare_molecules


def render_full_reasoning(row, reference, wrong_plan, wrong_execution, *, tokenizers=None):
    """Render false task requirements and their conditionally valid consequences.

    No model, tokenizer loading, or graph execution occurs here. Tokenizers, when
    supplied, are used only to expose exact annotated-token density to callers.
    """
    clean_plan, clean = reference['edit_plan'], reference['execution']
    if compare_molecules(clean['product_smiles'], wrong_execution['product_smiles'])['equivalent']:
        raise ValueError('The altered execution must produce a different molecular graph')
    if len(clean_plan.get('add_fragments', [])) != len(wrong_plan.get('add_fragments', [])):
        raise ValueError('Changing the number of incoming fragments is unsupported')
    original_roots = semantic_roots(clean_plan, wrong_plan)
    if not original_roots:
        raise ValueError('No changed task interpretation')
    nodes = build_nodes(row, clean_plan, wrong_plan, original_roots, clean, wrong_execution,
                        include_local_environment=False)
    # Legacy build_nodes omits unchanged plan slots although consequences may
    # reference them. Materialize those facts to close the saved dependency DAG.
    unchanged_slots = {'remove_set': sorted(wrong_plan.get('remove_atom_maps', []))}
    if wrong_plan.get('add_fragments'):
        addition = wrong_plan['add_fragments'][0]
        unchanged_slots.update(fragment=addition['smiles'], anchor=addition['anchor_map'])
    for key, value in unchanged_slots.items():
        if key not in nodes:
            nodes[key] = {'before': deepcopy(value), 'after': deepcopy(value),
                          'changed': False, 'parents': [], 'root_ids': [], 'kind': 'unchanged'}
    roots = []
    task_nodes = {}
    for root in original_roots:
        physical_id = root['node_id']
        task_id = physical_id
        task_nodes[task_id] = {
            'before': deepcopy(root['before']), 'after': deepcopy(root['after']),
            'changed': True, 'parents': [], 'root_ids': [root['id']], 'kind': 'root_error',
            'claim_type': 'task_interpretation',
        }
        nodes[physical_id]['claim_type'] = 'task_interpretation'
        roots.append({**deepcopy(root), 'node_id': task_id, 'type': 'task_interpretation',
                      'physical_node_id': physical_id,
                      'semantics': 'Required operation relative to the unchanged original input instruction'})
    for node in nodes.values():
        node['parents'] = list(dict.fromkeys(parent for p in node['parents']
            for parent in ([r['node_id'] for r in roots] if p == 'edit_plan' else [p])))
    # The legacy template renders physical claims; required-operation mentions are bound
    # separately below, avoiding unbound custom nodes in that strict renderer.
    physical = render_reasoning(row['indexed_smiles'], wrong_plan, nodes, label_errors=True)
    nodes.update(task_nodes)
    nodes['product_formula'] = {
        'before': clean['formula'], 'after': wrong_execution['formula'],
        'changed': clean['formula'] != wrong_execution['formula'],
        'parents': [r['node_id'] for r in roots], 'root_ids': [r['id'] for r in roots],
        'kind': 'propagated_error' if clean['formula'] != wrong_execution['formula'] else 'unchanged',
    }
    phrases = {
        'fragment': 'The required edit uses the incoming fragment ',
        'anchor': 'The required edit uses source attachment atom ',
        'fragment_attachment': 'The required edit uses the fragment attachment specification ',
        'bond_type': 'The required edit uses attachment bond type ',
        'remove_set': 'The required edit removes source atoms with map IDs ',
        'remove_bonds': 'The required edit removes the source bonds ',
        'add_bonds': 'The required edit adds the source bonds ',
        'change_bonds': 'The required edit changes source bond orders according to ',
    }
    prefix, spans, bindings = '', [], []
    for root in roots:
        node_id = root['node_id']
        prefix += phrases[root['physical_node_id']]
        value = root['after']
        displayed = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        start = len(prefix)
        prefix += displayed
        end = len(prefix)
        spans.append({'start': start, 'end': end, 'text': displayed, 'node_id': node_id,
                      'kind': 'root_error', 'root_ids': [root['id']]})
        bindings.append({'start': start, 'end': end, 'text': displayed, 'node_id': node_id,
                         'origin': 'approved_node', 'error_annotated': True})
        prefix += '.\n'
    prefix += 'The product molecular formula is '
    start = len(prefix)
    prefix += nodes['product_formula']['after']
    end = len(prefix)
    bindings.append({'start': start, 'end': end, 'text': prefix[start:end],
                     'node_id': 'product_formula', 'origin': 'approved_node',
                     'error_annotated': nodes['product_formula']['changed']})
    if nodes['product_formula']['changed']:
        spans.append({'start': start, 'end': end, 'text': prefix[start:end],
                      'node_id': 'product_formula', 'kind': 'propagated_error',
                      'root_ids': nodes['product_formula']['root_ids']})
    prefix += '.\n\n'
    offset = len(prefix)
    spans += [{**s, 'start': s['start'] + offset, 'end': s['end'] + offset} for s in physical['spans']]
    bindings += [{**b, 'start': b['start'] + offset, 'end': b['end'] + offset} for b in physical['bindings']]
    text = prefix + physical['text']
    for span in spans:
        node = nodes[span['node_id']]
        span['evidence'] = {
            'source': 'verified_execution_task_interpretation',
            'before': deepcopy(node['before']), 'after': deepcopy(node['after']),
            'parents': deepcopy(node['parents']), 'root_ids': deepcopy(node['root_ids']),
            'label_scope': ('Mistaken required operation relative to the original input instruction.'
                            if span['kind'] == 'root_error' else
                            'Physical consequence of the mistaken required operation; conditionally chemistry-consistent.'),
        }
    if any(parent not in nodes for node in nodes.values() for parent in node['parents']):
        raise ValueError('Dependency graph references absent semantic nodes')
    changed = {key for key, node in nodes.items() if node['changed']}
    if {s['node_id'] for s in spans} != changed:
        raise ValueError('Changed-node annotation closure failed')
    products = [row['gt_smiles'], clean['product_smiles'], wrong_execution['product_smiles']]
    for visible in [row['N_visible'], text]:
        if find_answer_leakage(visible, products):
            raise ValueError('Complete product leakage in visible reasoning')
    n = {'text': row['N_visible'], 'spans': [], 'edits': [], 'bindings': []}
    h = compile_patches(n['text'], [{'start': 0, 'end': len(n['text']), 'original': n['text'],
                                   'replacement': text, 'error_spans': spans}])
    h['bindings'] = bindings
    if h['text'] != text or h['edits'][0]['original'] != n['text']:
        raise ValueError('Full paired rewrite reconstruction failed')
    for binding in bindings:
        if text[binding['start']:binding['end']] != binding['text']:
            raise ValueError('Binding offsets drifted')
    counts = {name: sum(any(label != 'unchanged' for label in t['labels'])
                        for t in label_tokens(text, h['spans'], tokenizer))
              for name, tokenizer in (tokenizers or {}).items()}
    return {'N_render': n, 'H_render': h, 'nodes': nodes, 'roots': roots,
            'checks': {'status': 'pass', 'N_text_exact': True,
                       'full_N_to_H_forward_and_reverse_exact': True,
                       'changed_node_count': len(changed), 'root_count': len(roots),
                       'error_span_count': len(spans), 'annotated_tokens': counts,
                       'style_confound': 'Original N versus deterministic canonical H; not v17 byte-style parity.',
                       'external_instruction_conflict_remains': True,
                       'local_environment_included': False,
                       'execution_policy': 'Caller-supplied independently verified execution facts; no reexecution.'}}
