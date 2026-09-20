"""Append a fixed-scope product connection value to frozen native N/H traces.

Open ports denote bonds to omitted original source atoms, not replacement
wildcard atoms or H caps. Only the complete differing H connection value is a
new propagated claim. Existing root events, text edits and spans stay exact.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .labels import compile_patches
from .named_binding_chemistry import build_named_binding
from .named_binding_renderer import render_named_binding
from .quality import find_answer_leakage


def _require(condition, message):
    if not condition:
        raise ValueError('anchored connection renderer: ' + message)


def _validated_base(row, base, chemistry):
    _require(base.get('protocol') == 'agent_named_binding_diagnostic_v1'
             and base.get('style') == 'native' and base.get('status') == 'accepted',
             'base must be an accepted frozen v15 native record')
    _require(base.get('origin_id') == row.get('origin_id'), 'base origin differs from row')
    rendered = render_named_binding(row, chemistry, style='native')
    n, h = rendered['N_render'], rendered['H_render']
    _require(base.get('N_raw') == row['N_raw'] and base.get('N_source_visible') == row['N_visible']
             and base.get('N_visible') == n['text'], 'frozen original N differs')
    _require(base.get('N_semantic_bindings') == n['bindings'], 'frozen N bindings differ')
    _require(base.get('annotations') == h['spans'] and base.get('semantic_bindings') == h['bindings']
             and base.get('text_edits') == h['edits'], 'frozen H annotations, bindings or edits differ')
    plan = base.get('plan', {})
    _require(plan.get('render') == h and plan.get('nodes') == rendered['nodes']
             and plan.get('roots') == rendered['roots'], 'frozen H render, nodes or roots differ')
    _require(plan.get('edit_plan') == chemistry['wrong_plan']
             and plan.get('execution') == chemistry['wrong_execution'], 'frozen chemistry differs')
    pairs = base.get('pairs', [])
    _require(len(pairs) == 2 and [p.get('variant_label') for p in pairs] == ['N', 'H'],
             'base must contain exactly ordered N/H records')
    for pair, view in zip(pairs, (n, h)):
        expected = {'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
                    'reasoning_chain': view['text'], 'final_answer': row['gt_smiles']}
        _require(pair.get('origin_id') == row['origin_id'] and pair.get('detector_input') == expected,
                 'base pair question, answer or reasoning differs')
    return deepcopy(n), deepcopy(h), deepcopy(rendered['nodes']), deepcopy(rendered['roots'])


def _suffix(side, *, is_error):
    pieces, bindings = [], []
    cursor = 0

    def write(value):
        nonlocal cursor
        pieces.append(value)
        cursor += len(value)

    def slot(node_id, value, error=False):
        start = cursor
        text = str(value)
        write(text)
        item = {'start': start, 'end': cursor, 'text': text, 'node_id': node_id,
                'origin': 'approved_node', 'error_annotated': error}
        if node_id == 'anchored_connection':
            item['semantic_note'] = ('One open-boundary connection value at the fixed source anchor; '
                                     'not per-atom truth, a standalone molecule or a complete ring environment.')
        bindings.append(item)
        return start, cursor

    write('\n\nStep 6 [PRODUCT_CONNECTIONS]: Describe the product connections.\n'
          'The incoming group and source atom ')
    slot('anchor', side['anchor_map'])
    write(' (')
    slot('anchor_element', side['anchor_element'])
    write(') have the open-boundary connection representation "')
    start, end = slot('anchored_connection', side['open_port_smiles'], is_error)
    write('".\nIn this representation, [*:k] marks a boundary port to omitted original source atom k. '
          'The map k identifies the omitted atom; the wildcard itself is a port. '
          'These ports retain connections to the surrounding source structure. '
          'This is a partial connection representation, not a standalone molecule or the complete ring environment.')
    return {'text': ''.join(pieces), 'bindings': bindings, 'connection_start': start, 'connection_end': end}


def _shift_bindings(bindings, offset):
    return [{**deepcopy(item), 'start': item['start'] + offset, 'end': item['end'] + offset}
            for item in bindings]


def _base_patches(rendered):
    """Recover the original source-relative patch declarations without drift."""
    patches = []
    for edit in rendered['edits']:
        patch = {'start': edit['source_start'], 'end': edit['source_end'],
                 'original': edit['original'], 'replacement': edit['text'], 'error_spans': []}
        for key in ('kind', 'node_id', 'root_ids', 'evidence'):
            if key in edit:
                patch[key] = deepcopy(edit[key])
        for index in edit['error_span_indices']:
            span = rendered['spans'][index]
            annotation = {key: deepcopy(span[key]) for key in ('kind', 'node_id', 'root_ids', 'evidence') if key in span}
            annotation.update(start=span['start'] - edit['start'], end=span['end'] - edit['start'], text=span['text'])
            patch['error_spans'].append(annotation)
        patches.append(patch)
    return patches


def _verify_pair_edits(n, h):
    pieces, source_cursor, output_cursor = [], 0, 0
    for edit in h['edits']:
        start, end = edit['source_start'], edit['source_end']
        _require(source_cursor <= start <= end <= len(n['text']), 'source edit order or offsets invalid')
        _require(n['text'][start:end] == edit['original'], 'source edit quote changed')
        untouched = n['text'][source_cursor:start]
        _require(untouched.encode() == h['text'][output_cursor:edit['start']].encode(),
                 'unmodified paired text is not byte-equivalent')
        _require(h['text'][edit['start']:edit['end']] == edit['text'], 'output edit differs')
        pieces.extend((untouched, edit['text']))
        source_cursor, output_cursor = end, edit['end']
    pieces.append(n['text'][source_cursor:])
    _require(''.join(pieces) == h['text'], 'forward full N-to-H patch reconstruction failed')
    restored = h['text']
    for edit in reversed(h['edits']):
        restored = restored[:edit['start']] + edit['original'] + restored[edit['end']:]
    _require(restored.encode() == n['text'].encode(), 'inverse full H-to-N patch reconstruction failed')


def _verify_bindings(rendered):
    for item in rendered['bindings'] + rendered['spans']:
        _require(rendered['text'][item['start']:item['end']] == item['text'], 'binding or span offsets differ')
    _require({(b['start'], b['end'], b['node_id']) for b in rendered['bindings'] if b['error_annotated']}
             == {(s['start'], s['end'], s['node_id']) for s in rendered['spans']},
             'error bindings differ from spans')


def render_anchored_connection(row, base_record, connection_bundle, style='native') -> dict[str, Any]:
    """Render an unchanged native control or a symmetrically appended pair.

    Inputs are revalidated by the independent chemistry builder and the frozen
    native compiler. ``edits`` always map the complete returned N to H, including
    the one appended graph replacement in the added condition. No roots are added.
    """
    _require(style in {'native', 'native_with_connections'}, 'unsupported style')
    chemistry = build_named_binding(row)
    n, h, nodes, roots = _validated_base(row, base_record, chemistry)
    from .anchored_connections import build_anchored_connections
    expected = build_anchored_connections(row, chemistry)
    _require(connection_bundle == expected, 'connection evidence differs from exact reconstruction')
    _require(connection_bundle['checks']['status'] == 'pass', 'connection proof failed')
    comparison = connection_bundle['comparison']
    _require(all(comparison.get(k) is True for k in ('same_anchor', 'same_ports', 'same_anchor_state', 'different_at_same_anchor'))
             and comparison.get('new_atom_maps_compared') is False, 'connection scope or semantic comparison differs')
    _require(len(roots) == 1 and roots[0]['id'] == 'r1' and roots[0]['node_id'] == 'fragment',
             'expected exactly one existing fragment root')
    n_base, h_base = deepcopy(n), deepcopy(h)
    if style == 'native_with_connections':
        cn, ch = connection_bundle['N'], connection_bundle['H']
        _require(cn['open_port_smiles'] != ch['open_port_smiles'], 'appended connection value is unchanged')
        evidence = {
            'source': 'verified_open_boundary_product_connections',
            'before': cn['open_port_smiles'], 'after': ch['open_port_smiles'],
            'selection_rule': connection_bundle['selection_rule'],
            'anchor_map': cn['anchor_map'], 'ports': deepcopy(cn['ports']),
            'comparison': deepcopy(comparison),
            'label_scope': 'Whole graph value differs at a fixed original-source anchor and the same open ports; unchanged atom/port characters are not individually false.',
        }
        nodes['anchored_connection'] = {
            'before': cn['open_port_smiles'], 'after': ch['open_port_smiles'], 'changed': True,
            'kind': 'propagated_error', 'parents': ['fragment'], 'root_ids': ['r1'], 'evidence': deepcopy(evidence),
        }
        ns, hs = _suffix(cn, is_error=False), _suffix(ch, is_error=True)
        n['text'] += ns['text']
        n['bindings'].extend(_shift_bindings(ns['bindings'], len(n_base['text'])))
        patches = _base_patches(h_base)
        patches.append({
            'start': len(n_base['text']) + ns['connection_start'],
            'end': len(n_base['text']) + ns['connection_end'],
            'original': cn['open_port_smiles'], 'replacement': ch['open_port_smiles'],
            'kind': 'propagated_error', 'node_id': 'anchored_connection', 'root_ids': ['r1'],
            'evidence': deepcopy(evidence),
        })
        compiled = compile_patches(n['text'], patches)
        _require(compiled['text'] == h_base['text'] + hs['text'], 'symmetric appended template changed')
        _require(compiled['spans'][:-1] == h_base['spans'] and compiled['edits'][:-1] == h_base['edits'],
                 'appending changed original semantic annotations or paired edits')
        compiled['bindings'] = h_base['bindings'] + _shift_bindings(hs['bindings'], len(h_base['text']))
        h = compiled
    _verify_pair_edits(n, h)
    for rendered in (n, h):
        _verify_bindings(rendered)
        _require(not find_answer_leakage(rendered['text'], [row['gt_smiles'], chemistry['wrong_execution']['product_smiles']]),
                 'complete correct or wrong product in reasoning')
    _require(n['text'].startswith(n_base['text']) and h['text'].startswith(h_base['text']), 'base text prefix changed')
    _require(n['bindings'][:len(n_base['bindings'])] == n_base['bindings']
             and h['bindings'][:len(h_base['bindings'])] == h_base['bindings'], 'base bindings changed')
    _require({s['node_id'] for s in h['spans']} == {k for k, v in nodes.items() if v['changed']},
             'changed semantic nodes are not exactly covered')
    checks = {
        'status': 'pass', 'style': style, 'frozen_native_prefixes_exact': True,
        'base_spans_exact': True, 'base_bindings_exact': True, 'base_edits_exact': True,
        'complete_forward_inverse_edit_relation': True, 'unmodified_paired_bytes_exact': True,
        'connection_evidence_recomputed': True, 'same_selection_anchor_ports_and_anchor_state': True,
        'complete_product_and_component_projection_exclusion': 'verified_by_anchored_connections',
        'no_new_root': True, 'new_derived_nodes': int(style == 'native_with_connections'),
        'N_characters': len(n['text']), 'H_characters': len(h['text']),
        'N_bytes': len(n['text'].encode()), 'H_bytes': len(h['text'].encode()),
        'N_added_characters': len(n['text']) - len(n_base['text']),
        'H_added_characters': len(h['text']) - len(h_base['text']),
        'N_spans': len(n['spans']), 'H_spans': len(h['spans']),
        'N_bindings': len(n['bindings']), 'H_bindings': len(h['bindings']),
        'N_edits': len(n['edits']), 'H_edits': len(h['edits']),
        'base_H_spans': len(h_base['spans']), 'base_H_edits': len(h_base['edits']),
        'annotation_scope': 'One propagated open-boundary graph value; no new independent anchor, port, H or atom error.',
        'contrast_scope': 'Fixed chemistry with versus without an appended partial product-connection representation and its explanatory text; not a natural-error-frequency estimate.',
    }
    return {'N_render': n, 'H_render': h, 'nodes': nodes, 'roots': roots, 'checks': checks}
