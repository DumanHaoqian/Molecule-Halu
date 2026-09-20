"""Reassign one frozen named-fragment error to a task-requirement reading.

The instruction, source, attachment, wrong fragment and both executed products
stay fixed. Only the causal interpretation changes: ``task_group`` is the one
root and ``fragment`` is a correctly named graph conditional on that wrong task
reading. The original requirement is still contradicted. This module makes no
claim about naturalness or model behavior and reads no model-output fields.

``nodes``/``roots``/``intent_evidence`` describe the current intent intervention.
``physical_reference`` has no old causal root or binding interpretation.
``parent_provenance`` explicitly retains the old relation as historical only.
Full products in physical evidence are for verification, never trace rendering.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .anchored_connection_renderer import render_anchored_connection
from .anchored_connections import build_anchored_connections
from .chemistry_tools import compare_molecules
from .named_binding_chemistry import build_named_binding, WRONG_FRAGMENT_SMILES, WRONG_GROUP_NAME
from .named_binding_renderer import render_named_binding


def _record(checks: list, name: str, condition: bool, evidence: Any = None) -> None:
    check = {'check': name, 'status': 'pass' if condition else 'fail'}
    if evidence is not None:
        check['evidence'] = evidence
    checks.append(check)
    if not condition:
        raise ValueError('task intent: ' + name)


def _replay_parent_surface(row: dict, chemistry: dict, connections: dict) -> tuple[dict, dict]:
    """Use frozen compilers without reading ancestors from caller-supplied paths.

    The temporary adapter below only supplies the older renderer's input schema;
    it is not a new accepted artifact or a fresh review/approval decision.
    """
    native = render_named_binding(row, chemistry, style='native')
    n, h = native['N_render'], native['H_render']
    adapter = {
        'protocol': 'agent_named_binding_diagnostic_v1', 'style': 'native',
        'status': 'accepted', 'origin_id': row['origin_id'],
        'N_raw': row['N_raw'], 'N_source_visible': row['N_visible'], 'N_visible': n['text'],
        'N_semantic_bindings': n['bindings'], 'annotations': h['spans'],
        'semantic_bindings': h['bindings'], 'text_edits': h['edits'],
        'plan': {'render': h, 'nodes': native['nodes'], 'roots': native['roots'],
                 'edit_plan': chemistry['wrong_plan'], 'execution': chemistry['wrong_execution']},
        'pairs': [
            {'variant_label': label, 'origin_id': row['origin_id'],
             'detector_input': {'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
                                'reasoning_chain': rendered['text'], 'final_answer': row['gt_smiles']}}
            for label, rendered in [('N', n), ('H', h)]
        ],
    }
    return render_anchored_connection(row, adapter, connections, style='native_with_connections'), native


def _validate_parent(row: dict, parent: dict, chemistry: dict,
                     connections: dict, checks: list) -> dict:
    _record(checks, 'parent_is_frozen_v16_augmented_record',
            parent.get('protocol') == 'agent_anchored_connections_diagnostic_v1'
            and parent.get('style') == 'native_with_connections'
            and parent.get('status') == 'accepted' and parent.get('origin_id') == row.get('origin_id'))
    _record(checks, 'parent_connection_bundle_exact_replay', parent.get('connection_evidence') == connections)
    plan = parent.get('plan', {})
    _record(checks, 'parent_physical_plan_execution_and_old_root_exact',
            plan.get('edit_plan') == chemistry['wrong_plan']
            and plan.get('execution') == chemistry['wrong_execution']
            and plan.get('roots') == chemistry['roots'])
    replay, native = _replay_parent_surface(row, chemistry, connections)
    n, h = replay['N_render'], replay['H_render']
    _record(checks, 'parent_nodes_and_render_exact_frozen_compilers',
            plan.get('nodes') == replay['nodes'] and plan.get('render') == h)
    _record(checks, 'parent_original_and_visible_N_exact',
            parent.get('N_raw') == row['N_raw'] and parent.get('N_source_visible') == row['N_visible']
            and parent.get('N_visible') == n['text'])
    _record(checks, 'parent_spans_bindings_and_edits_exact',
            parent.get('N_semantic_bindings') == n['bindings']
            and parent.get('annotations') == h['spans']
            and parent.get('semantic_bindings') == h['bindings']
            and parent.get('text_edits') == h['edits'])
    _record(checks, 'parent_binding_explanation_is_validated_historical_record',
            parent.get('binding_evidence') == native['binding_evidence'])
    pairs = parent.get('pairs', [])
    _record(checks, 'parent_exactly_one_N_H_pair',
            len(pairs) == 2 and [pair.get('variant_label') for pair in pairs] == ['N', 'H'])
    for pair, rendered in zip(pairs, [n, h]):
        _record(checks, 'unchanged_original_input_' + pair['variant_label'],
                pair.get('origin_id') == row['origin_id']
                and pair.get('detector_input') == {
                    'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
                    'reasoning_chain': rendered['text'], 'final_answer': row['gt_smiles'],
                })
    return replay


def build_task_intent(row: dict[str, Any], base_record: dict[str, Any]) -> dict[str, Any]:
    """Return current intent nodes/roots separately from fixed physical evidence.

    ``base_record`` must be the exact augmented v16 surface for one of the four
    declared named-acyl cases. Both old compilers, chemistry and open connections
    are replayed before changing causal metadata. A root alias renames the old
    ``group_name`` slot to ``task_group``; the renderer may retarget its N bindings
    while preserving every N byte. All original H name aliases must use this
    same root variable. No plan, product, requirement or source is edited here.
    Invalid parent data or an unsupported original instruction raises ValueError.
    """
    checks: list[dict] = []
    chemistry = build_named_binding(row)
    connections = build_anchored_connections(row, chemistry)
    replay = _validate_parent(row, base_record, chemistry, connections, checks)
    requested = chemistry['facts']['N']['group_name']
    represented = chemistry['facts']['H']['represented_group_name']
    wrong_operation = chemistry['wrong_plan']['add_fragments'][0]
    _record(checks, 'perceived_group_is_actual_wrong_graph_name',
            represented == WRONG_GROUP_NAME
            and compare_molecules(wrong_operation['smiles'], WRONG_FRAGMENT_SMILES)['equivalent']
            and wrong_operation['attach_atom_index'] == 0
            and chemistry['facts']['H']['fragment']['carbonyl_attachment_index'] == 0)
    _record(checks, 'perceived_task_group_conflicts_with_original_required_group', requested != represented)
    _record(checks, 'same_source_anchor_H_adjustment_and_plan_except_existing_incoming_graph',
            chemistry['clean_plan']['adjust_hydrogens'] == chemistry['wrong_plan']['adjust_hydrogens']
            and chemistry['clean_plan']['add_fragments'][0]['anchor_map'] == wrong_operation['anchor_map']
            and connections['comparison']['same_anchor']
            and connections['comparison']['same_ports']
            and connections['comparison']['same_anchor_state'])
    intent_evidence = {
        'source': 'verified_task_requirement_interpretation',
        'scope': 'task_requirement_interpretation',
        'original_instruction': row['instruction'],
        'requested_group': requested, 'perceived_group': represented,
        'actual_wrong_group_name': represented,
        'actual_wrong_fragment_smiles': wrong_operation['smiles'],
        'name_scope': 'attached_acyl_moiety',
        'attachment_atom_index': wrong_operation['attach_atom_index'],
        'source_anchor_map': wrong_operation['anchor_map'],
        'standalone_fragment_formula': chemistry['facts']['H']['fragment']['standalone_formula'],
        'attached_fragment_formula': chemistry['facts']['H']['fragment']['attached_formula'],
        'fragment_cap_hydrogens_consumed': chemistry['facts']['H']['fragment']['carbonyl_cap_hydrogens'],
        'name_matches_actual_wrong_graph': True,
        'perceived_requirement_conflicts_with_original_instruction': True,
        'original_instruction_changed': False, 'original_source_changed': False,
        'physical_plan_changed': False, 'parent_binding_is_current': False,
        'label_scope': (
            'One incorrect interpretation of the required incoming group. Repeated name slots '
            'refer to the same root variable; a chemically correct name for the wrong graph '
            'does not make it the group required by the unchanged original instruction.'
        ),
    }
    nodes = deepcopy(replay['nodes'])
    previous_group = nodes.pop('group_name')
    _record(checks, 'old_group_slot_was_original_task_symbol',
            previous_group['before'] == previous_group['after'] == requested
            and previous_group['parents'] == [] and previous_group['root_ids'] == [])
    nodes['task_group'] = {
        'before': requested, 'after': represented, 'changed': True,
        'kind': 'root_error', 'parents': [], 'root_ids': ['r1'],
        'evidence': deepcopy(intent_evidence),
    }
    # Refresh causal evidence rather than presenting the old false-name binding
    # as the current mechanism. Physical fact specifications remain unchanged.
    for key, node in nodes.items():
        if key == 'task_group' or not node['root_ids']:
            continue
        node['evidence'] = {
            **node['evidence'],
            'source': 'verified_conditional_chemistry_after_task_interpretation',
            'root_semantics': 'task_requirement_interpretation',
        }
    fragment = nodes['fragment']
    fragment['kind'] = 'propagated_error'
    fragment['parents'] = ['task_group']
    fragment['evidence'].update({
        'source': 'verified_name_to_fragment_under_perceived_task',
        'before_group': requested, 'after_group': represented,
        'mapping_is_correct_conditional_on_perceived_group': True,
        'original_task_satisfied': False,
        'before_attachment_atom_index': chemistry['clean_plan']['add_fragments'][0]['attach_atom_index'],
        'after_attachment_atom_index': wrong_operation['attach_atom_index'],
    })
    roots = [{'id': 'r1', 'node_id': 'task_group', 'type': 'task_interpretation',
              'before': requested, 'after': represented}]
    _record(checks, 'one_root_replaces_fragment_root_without_duplication',
            [key for key, node in nodes.items() if node['kind'] == 'root_error'] == ['task_group']
            and fragment['root_ids'] == ['r1'])
    for key, before in replay['nodes'].items():
        if key == 'group_name':
            continue
        after = nodes[key]
        _record(checks, 'unchanged_physical_node_' + key,
                all(after[field] == before[field] for field in ['before', 'after', 'changed'])
                and (key == 'fragment' or (after['kind'] == before['kind'] and after['parents'] == before['parents'])))

    def ancestors(key: str, visited: frozenset = frozenset()) -> set[str]:
        if key == 'task_group':
            return {'r1'}
        if key not in nodes or key in visited:
            raise ValueError('task intent: invalid dependency graph')
        return set().union(*(ancestors(parent, visited | {key}) for parent in nodes[key]['parents']))

    _record(checks, 'all_node_root_ancestry_matches_current_interpretation',
            all(set(node['root_ids']) == ancestors(key) for key, node in nodes.items()))
    return {
        'nodes': nodes, 'roots': roots, 'intent_evidence': intent_evidence,
        'physical_reference': {
            'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
            'gt_smiles': row['gt_smiles'],
            **{key: deepcopy(chemistry[key]) for key in
               ['clean_plan', 'wrong_plan', 'clean_execution', 'wrong_execution']},
            'connections': {key: deepcopy(connections[key]) for key in
                            ['N', 'H', 'comparison', 'selection_rule', 'port_semantics']},
        },
        'parent_provenance': {
            'protocol': base_record['protocol'], 'style': base_record['style'],
            'pair_id': base_record['pair_id'], 'interpretation': 'historical_parent_only',
            'roots': deepcopy(base_record['plan']['roots']),
            'binding_evidence': deepcopy(base_record['binding_evidence']),
        },
        'checks': {'status': 'pass', 'checks': checks},
    }
