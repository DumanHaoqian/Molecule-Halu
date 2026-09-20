"""Close a mistaken task-group interpretation over four frozen H traces.

The original external instruction stays unchanged. H's names agree with its
already-fixed wrong graph; they remain wrong requirements for the original task.
This is a declared four-record prototype, not a general text rewrite system.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from .labels import compile_patches
from .quality import find_answer_leakage

TASK_INTENT_INVENTORY_PATH = Path(__file__).with_name('task_intent_inventory.json')


def load_inventory():
    return json.loads(TASK_INTENT_INVENTORY_PATH.read_text())


def _require(ok, message):
    if not ok:
        raise ValueError('task intent renderer: ' + message)


def _evidence(node_id, nodes, intent_evidence):
    node = nodes[node_id]
    return {'source': 'verified_task_intent_chemistry', 'before': deepcopy(node['before']),
            'after': deepcopy(node['after']), 'parents': deepcopy(node['parents']),
            'root_ids': deepcopy(node['root_ids']),
            'task_requirement': deepcopy(intent_evidence),
            'label_scope': ('Mistaken required group relative to the unchanged original instruction.' if node_id == 'task_group'
                            else 'Fixed wrong physical value propagated from the mistaken task requirement; not a current name-to-graph contradiction.')}


def _parent_patches(parent_h, nodes, intent_evidence):
    patches = []
    for edit in parent_h['edits']:
        patch = {'start': edit['source_start'], 'end': edit['source_end'],
                 'original': edit['original'], 'replacement': edit['text'], 'error_spans': []}
        for key in ('node_id', 'root_ids'):
            if key in edit:
                patch[key] = deepcopy(edit[key])
        if edit.get('node_id') in nodes and nodes[edit['node_id']]['changed']:
            node_id = edit['node_id']
            patch['kind'] = nodes[node_id]['kind']
            patch['evidence'] = _evidence(node_id, nodes, intent_evidence)
        for index in edit['error_span_indices']:
            span = parent_h['spans'][index]; node_id = span['node_id']; node = nodes[node_id]
            patch['error_spans'].append({'start': span['start'] - edit['start'], 'end': span['end'] - edit['start'],
                                        'text': span['text'], 'node_id': node_id, 'kind': node['kind'],
                                        'root_ids': deepcopy(node['root_ids']), 'evidence': _evidence(node_id, nodes, intent_evidence)})
        patches.append(patch)
    return patches


def _unchanged_parent_interval(start, end, parent_edits):
    """Map a newly edited, previously untouched parent-H interval back into N."""
    delta = 0
    for edit in parent_edits:
        _require(not (start < edit['end'] and end > edit['start']),
                 'new task-name patch overlaps a pre-existing physical edit')
        if edit['end'] <= start:
            delta += edit['source_end'] - edit['source_start'] - (edit['end'] - edit['start'])
    return start + delta, end + delta


def _move_binding(binding, supplemental_edits):
    item = deepcopy(binding); start, end = item['start'], item['end']; delta = 0
    for edit in supplemental_edits:
        lo, hi = edit['source_start'], edit['source_end']
        if hi <= start:
            delta += len(edit['text']) - (hi - lo)
            continue
        if lo >= end:
            break
        _require(lo <= start and end <= hi, 'supplemental edit partially overlaps an existing semantic binding')
        # The one rewritten instruction clause retains its exact source-role phrase.
        _require(edit['text'].count(item['text']) == 1, 'retained binding is absent or ambiguous inside a rewritten clause')
        at = edit['text'].index(item['text'])
        item['start'], item['end'] = edit['start'] + at, edit['start'] + at + len(item['text'])
        break
    else:
        item['start'], item['end'] = start + delta, end + delta
        return item
    if item['start'] == start and item['end'] == end:
        item['start'], item['end'] = start + delta, end + delta
    return item


def _verify_render(rendered, nodes):
    for item in rendered['bindings'] + rendered['spans']:
        _require(rendered['text'][item['start']:item['end']] == item['text'], 'semantic text or offsets drifted')
        _require(item['node_id'] in nodes, 'semantic binding references an absent node')
    expected = [(s['start'], s['end'], s['node_id']) for s in rendered['spans']]
    actual = [(b['start'], b['end'], b['node_id']) for b in rendered['bindings'] if b['error_annotated']]
    _require(sorted(expected) == sorted(actual), 'error bindings do not exactly cover spans')
    for span in rendered['spans']:
        node = nodes[span['node_id']]
        _require(node['changed'] and span['kind'] == node['kind'] and span['root_ids'] == node['root_ids'],
                 'annotation disagrees with intent dependency graph')


def _verify_edits(n, h):
    chunks, source_cursor, output_cursor = [], 0, 0
    for edit in h['edits']:
        lo, hi = edit['source_start'], edit['source_end']
        _require(source_cursor <= lo <= hi <= len(n), 'full source edit order invalid')
        _require(n[lo:hi] == edit['original'], 'full source edit quote differs')
        untouched = n[source_cursor:lo]
        _require(untouched.encode() == h['text'][output_cursor:edit['start']].encode(), 'unmodified full paired bytes differ')
        chunks.extend((untouched, edit['text'])); source_cursor, output_cursor = hi, edit['end']
    chunks.append(n[source_cursor:])
    _require(''.join(chunks) == h['text'], 'complete forward edit reconstruction failed')
    restored = h['text']
    for edit in reversed(h['edits']):
        restored = restored[:edit['start']] + edit['original'] + restored[edit['end']:]
    _require(restored.encode() == n.encode(), 'complete reverse edit reconstruction failed')


def render_task_intent(row, base_record, intent_bundle, style='binding') -> dict[str, Any]:
    """Keep the parent binding control or change only task-group text in H.

    N's text is byte-identical in both styles. Intent N's group-name bindings
    reference task_group instead of group_name; this metadata change is explicit.
    Full paired edits always use returned N coordinates, not parent-H positions.
    """
    _require(style in {'binding', 'intent'}, 'unsupported style')
    record = load_inventory().get('origins', {}).get(row.get('origin_id'))
    _require(record is not None, 'origin is outside the fixed four-record inventory')
    _require(row['instruction'] == record['original_instruction']
             and base_record['N_visible'] == record['parent_N']
             and base_record['plan']['render']['text'] == record['parent_H'], 'frozen parent or instruction drift')
    from .task_intent_chemistry import build_task_intent
    _require(intent_bundle == build_task_intent(row, base_record), 'intent evidence differs from checked parent chemistry')
    _require(intent_bundle['checks']['status'] == 'pass', 'intent chemistry check failed')
    n = {'text': base_record['N_visible'], 'spans': [], 'bindings': deepcopy(base_record['N_semantic_bindings']), 'edits': []}
    parent_h = deepcopy(base_record['plan']['render'])
    h, nodes, roots = deepcopy(parent_h), deepcopy(base_record['plan']['nodes']), deepcopy(base_record['plan']['roots'])
    current_intent = None
    added = []
    if style == 'intent':
        nodes, roots = deepcopy(intent_bundle['nodes']), deepcopy(intent_bundle['roots'])
        current_intent = deepcopy(intent_bundle['intent_evidence'])
        _require(len(roots) == 1 and roots[0]['id'] == 'r1' and roots[0]['node_id'] == 'task_group'
                 and nodes['fragment']['kind'] == 'propagated_error' and nodes['fragment']['parents'] == ['task_group'],
                 'intent must replace the fragment root with exactly one task-group root')
        for binding in n['bindings']:
            if binding['node_id'] == 'group_name':
                binding['node_id'] = 'task_group'
                binding['semantic_note'] = 'Correct requested group in unchanged N; node identity follows the intent-condition dependency graph.'
        added = deepcopy(record['patches_from_parent_H'])
        for patch in added:
            for span in patch['error_spans']:
                _require(span['node_id'] == 'task_group' and span['kind'] == 'root_error'
                         and span['root_ids'] == ['r1'] and span['text'] == nodes['task_group']['after'],
                         'supplemental annotation is not the declared task-group root')
                span['evidence'] = _evidence('task_group', nodes, current_intent)
        supplemental = compile_patches(parent_h['text'], added)
        _require(supplemental['text'] == record['expected_intent_H'], 'incomplete task-name or grammar closure')
        _require(len(supplemental['spans']) == record['expected_task_group_spans'], 'incorrect task-group root coverage')
        patches = _parent_patches(parent_h, nodes, current_intent)
        for patch in added:
            converted = deepcopy(patch)
            converted['start'], converted['end'] = _unchanged_parent_interval(patch['start'], patch['end'], parent_h['edits'])
            _require(n['text'][converted['start']:converted['end']] == patch['original'],
                     'new task patch cannot be mapped exactly to frozen full N')
            patches.append(converted)
        h = compile_patches(n['text'], patches)
        _require(h['text'] == supplemental['text'], 'combined paired compilation changed physical text')
        bindings = []
        for binding in parent_h['bindings']:
            if binding['node_id'] == 'group_name':
                continue
            item = _move_binding(binding, supplemental['edits'])
            if item['error_annotated']:
                item['semantic_note'] = 'Error value relative to the unchanged original task, propagated from mistaken task_group.'
            bindings.append(item)
        for span in h['spans']:
            if span['node_id'] == 'task_group':
                bindings.append({'start': span['start'], 'end': span['end'], 'text': span['text'],
                                 'node_id': 'task_group', 'origin': 'approved_node', 'error_annotated': True,
                                 'semantic_note': 'Mistaken task requirement; generic acylation verb, grammar and source-role text are not included.'})
        h['bindings'] = sorted(bindings, key=lambda item: (item['start'], item['end'], item['node_id']))
        name = record['requested_group']
        _require(not re.search(r'\b'+re.escape(name)+r'\b', h['text']), 'an original group-name alias remains in H')
        _require(h['text'].count(record['perceived_group']) == record['expected_task_group_spans'],
                 'unbound or extra perceived group-name occurrence')
        if row['origin_id'].endswith('0194'):
            _require('acetylating' not in h['text'] and 'an '+record['perceived_group'] not in h['text'],
                     'acetylation instruction clause or article was not closed')
        for node_id in {s['node_id'] for s in parent_h['spans']}:
            _require([s['text'] for s in h['spans'] if s['node_id'] == node_id]
                     == [s['text'] for s in parent_h['spans'] if s['node_id'] == node_id],
                     'an existing physical error value was changed')
        _require({s['node_id'] for s in h['spans'] if s['kind'] == 'root_error'} == {'task_group'},
                 'old fragment root annotation remained')
    _verify_edits(n['text'], h)
    for rendered in (n, h):
        _verify_render(rendered, nodes)
        _require(not find_answer_leakage(rendered['text'], [row['gt_smiles'], base_record['plan']['execution']['product_smiles']]),
                 'complete correct or wrong product in CoT')
    _require({s['node_id'] for s in h['spans']} == {key for key, node in nodes.items() if node['changed']},
             'changed node closure differs from annotations')
    checks = {
        'status': 'pass', 'style': style, 'N_text_exact': True, 'N_binding_node_ids_changed': style == 'intent',
        'N_metadata_policy': 'Intent remaps original group_name bindings to task_group; only N text is identical across styles.',
        'H_changes_limited_to_declared_names_and_grammar': True,
        'physical_values_plans_execution_anchor_ports_unchanged': True,
        'full_N_to_H_forward_and_reverse_exact': True, 'unmodified_paired_bytes_exact': True,
        'root_count': len(roots), 'root_node': roots[0]['node_id'],
        'new_task_group_mentions': record['expected_task_group_spans'] if style == 'intent' else 0,
        'supplemental_H_text_patches': len(added),
        'N_characters': len(n['text']), 'H_characters': len(h['text']),
        'H_characters_added_vs_parent': len(h['text']) - len(parent_h['text']),
        'N_spans': len(n['spans']), 'H_spans': len(h['spans']), 'H_edits': len(h['edits']),
        'N_bindings': len(n['bindings']), 'H_bindings': len(h['bindings']),
        'external_instruction_conflict_remains': True,
        'interpretation': ('Internal task-name/graph consistency under one mistaken requirement; original instruction still disagrees.' if style == 'intent'
                           else 'Unchanged parent name-to-graph binding conflict.'),
        'partial_product_information_policy': 'Parent open-boundary product connection information is retained; exclusion of a complete product does not mean no answer information.',
    }
    return {'N_render': n, 'H_render': h, 'nodes': nodes, 'roots': roots, 'checks': checks,
            'intent_evidence': current_intent,
            'parent_provenance': deepcopy(intent_bundle['parent_provenance']) if style == 'intent' else None}
