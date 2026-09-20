"""Bounded source-state interventions with an unchanged original question."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from . import agents
from .chemistry_tools import inspect_source
from .labels import label_tokens
from .orchestrator import check_leakage, choose_candidate, review_diagnostics, write_json
from .poe_transport import PoeClient
from .quality import deterministic_reference_plan, validate_formal_claims, validate_reference
from .severity import measure_severity
from .source_state_candidates import enumerate_source_state_candidates
from .source_state_renderer import render_source_state_pair, render_source_state_reference


PROTOCOL = 'agent_source_state_v1'
MODEL_NAMES = ('ChemDFM-R-14B', 'Chem-R-8B')
CLEAN_SOURCE = '''Review the clean reference chemistry before any corruption is constructed.
The original unchanged question, source graph facts, executable reference edit and its verified
product facts are provided. N starts by reproducing the original source SMILES and formula,
then explains that edit. The source representation is not a final product. Verify consistency
with the instruction; graph-verified counts are authoritative, not a mental-counting task.
No corrupted trace or candidate is present. Return JSON only
{"consistent":true|false,"concerns":[{"quote":"exact N substring","explanation":"specific mismatch"}]}.
If no mismatch is found return true and an empty concerns list. Report uncertainty explicitly.
'''
SELECT_SOURCE = '''Choose interpretable mistakes in reconstructing the starting molecular graph.
The original question and reference edit stay fixed. Each candidate omits remote source branches,
then applies exactly the correct reference edit to that mistaken perceived source. These are
intentionally incorrect source interpretations, not proposed correct solutions. The protected edit
region and executable chemistry have already been checked. Select at most 6 existing candidate IDs
that could be described coherently as overlooking source substituents in a complex molecule.
No final product or target-model output is available. Do not invent new operations or products.
Return JSON only {"candidate_ids":["candidate_001",...],"rationale":"brief"}.
An empty list is allowed when no candidate is interpretable. This is a controlled diagnostic,
not a statement about the frequency or distribution of natural model errors.
'''
CONTROL_SOURCE = agents.CONTROL_AUDIT + '''
Here source_graph is the intentionally false root: the trace reconstructs the starting molecule
incorrectly. Its formula and source/product counts must describe that approved perceived source
and the result of applying the unchanged reference edit to it. Do not replace those source counts
with the original question's counts. The starting-molecule representation is not a final product;
flag a product leak only with specific evidence, not merely because a complete source is shown.
'''


def generate_one(row, directory, *, seed=20260920, min_nodes=6, min_tokens=40,
                 tokenizers=None, model='gpt-5.4-mini'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frozen = {'protocol': PROTOCOL, 'row': row, 'seed': seed, 'min_roots': 1,
        'min_nodes': min_nodes, 'min_tokens': min_tokens, 'model': model,
        'candidate_mode': 'source_state', 'max_candidates': 12,
        'role_prompts': {'clean_reference': CLEAN_SOURCE, 'select': SELECT_SOURCE,
                        'blind': agents.BLIND_AUDIT, 'control_audit': CONTROL_SOURCE},
        'required_tokenizers': sorted((tokenizers or {}).keys())}
    path = directory/'input.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('Resume input/config changed')
    write_json(path, frozen)
    for status in ('accepted', 'rejected'):
        path = directory/f'{status}.json'
        if path.exists():
            return json.loads(path.read_text())
    state = {'origin_id': row['origin_id'], 'status': 'started', 'protocol': PROTOCOL}
    client = None
    try:
        if set(tokenizers or {}) != set(MODEL_NAMES):
            raise ValueError('Both model tokenizers are required')
        checks = validate_reference(row)
        write_json(directory/'reference_checks.json', checks)
        if checks['status'] != 'pass':
            raise ValueError('reference_invalid: '+str(checks.get('failures')))
        reference = deterministic_reference_plan(row)
        clean_plan, clean = reference['edit_plan'], reference['execution']
        write_json(directory/'reference.json', reference)
        frozen_n = render_source_state_reference(row, clean_plan, clean)
        n = frozen_n['render']['text']
        source = inspect_source(row['indexed_smiles'])
        check_leakage(n, [row['gt_smiles']])
        n_checks = validate_formal_claims(n, frozen_n['nodes'], source)
        if n_checks['status'] != 'pass':
            raise ValueError('reference_renderer_invalid: '+str(n_checks))
        write_json(directory/'reference_render.json', frozen_n)
        client = PoeClient(directory/'plan_calls', model=model, max_calls=6)
        refpayload = {'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
            'N': n, 'clean_plan': clean_plan, 'source_facts': source,
            'reference_product_facts': {k: clean[k] for k in ('heavy_atoms', 'rings', 'formula')},
            'source_representation_policy': 'The initial complete SMILES is the original source, copied from the input, not the final product.'}
        reference_review = client.ask('clean_reference', CLEAN_SOURCE, refpayload, temperature=0)
        candidates = enumerate_source_state_candidates(row, clean_plan, max_candidates=12)
        feasible, rejected = [], []
        for candidate in candidates:
            try:
                rendered = render_source_state_pair(row, clean_plan, clean, candidate)
                if rendered['N_render']['text'] != n:
                    raise ValueError('N changed in response to an H candidate')
                h_render, nodes = rendered['H_render'], rendered['nodes']
                h = h_render['text']
                check_leakage(n, [candidate['execution']['product_smiles']])
                check_leakage(h, [row['gt_smiles'], candidate['execution']['product_smiles']])
                observed = {s['node_id'] for s in h_render['spans']}
                labels = {name: label_tokens(h, h_render['spans'], tok) for name, tok in tokenizers.items()}
                density = {name: sum(any(x != 'unchanged' for x in token['labels']) for token in values)
                           for name, values in labels.items()}
                if len(observed) < min_nodes or min(density.values()) < min_tokens:
                    raise ValueError('insufficient_error_density: '+str({'nodes': len(observed), 'tokens': density}))
                conditional_checks = validate_formal_claims(h, nodes, rendered['working_source_facts'])
                if conditional_checks['status'] != 'pass':
                    raise ValueError('conditional_claims_invalid: '+str(conditional_checks))
                # Joint execution uses the real S for a comparable source-map
                # metric; the visible reasoning uses the separately saved S′ path.
                severity = measure_severity(row['indexed_smiles'], clean, candidate['joint_execution'],
                    clean_trace=n, h_trace=h, annotations=h_render['spans'], token_labels=labels)
                feasible.append({**deepcopy(candidate), 'rendered': rendered, 'severity': severity,
                    'token_labels': labels, 'token_counts': density, 'checks': conditional_checks})
            except (ValueError, KeyError, TypeError) as error:
                rejected.append({'candidate_id': candidate['candidate_id'], 'error': str(error)})
        write_json(directory/'candidate_pool.json', {'all_candidates': candidates, 'eligible': feasible, 'rejected': rejected})
        if not feasible:
            raise ValueError('no_feasible_source_state')
        selection = client.ask('select', SELECT_SOURCE, {**refpayload, 'candidates': [
            {'candidate_id': c['candidate_id'], 'description': c['description'],
             'perceived_source_smiles': c['perceived_source_execution']['product_smiles'],
             'omitted_atom_maps': c['source_plan'].get('remove_atom_maps', []),
             'protected_maps': c['protected_maps']} for c in feasible]})
        ids = selection.get('candidate_ids')
        eligible_ids = {c['candidate_id'] for c in feasible}
        if isinstance(ids, list):
            ids = [('candidate_'+i[1:]) if isinstance(i, str) and i not in eligible_ids
                   and re.fullmatch(r'c[0-9]{3}', i) else i for i in ids]
        if (not isinstance(ids, list) or not 0 < len(ids) <= 6
                or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)
                or any(i not in eligible_ids for i in ids)):
            raise ValueError('invalid_or_empty_source_selection')
        selected = choose_candidate([c for c in feasible if c['candidate_id'] in ids], seed=seed, mode='severe')
        write_json(directory/'candidate_selection.json', {'selection': selection, 'eligible_ids': sorted(eligible_ids)})
        rendered = selected.pop('rendered')
        token_labels = selected.pop('token_labels')
        token_counts = selected.pop('token_counts')
        compiled = rendered['H_render']
        h, roots, nodes = compiled['text'], rendered['roots'], rendered['nodes']
        root_observed = {rid for s in compiled['spans'] if s['kind'] == 'root_error' for rid in s['root_ids']}
        if root_observed != {root['id'] for root in roots}:
            raise ValueError('Missing root annotation')
        plan = {**selected, 'edit_plan': clean_plan, 'roots': roots, 'nodes': nodes,
            'render': compiled, 'candidate_mode': 'source_state', 'seed': seed,
            'sampling_policy': 'max_graph_distance_among_agent_approved_feasible_source_states',
            'selection_probability': 1.0, 'selection_probability_scope': 'conditional on the saved approved candidate set',
            'execution_scope': 'The visible edit acts on perceived_source_execution; joint_plan separately acts on the real source.'}
        write_json(directory/'fixed_plan.json', plan)
        blind = client.ask('blind', agents.BLIND_AUDIT, {'indexed_smiles': row['indexed_smiles'],
            'instruction': row['instruction'], 'reasoning': h}, temperature=0)
        audit_payload = {'H': h,
            'fixed_roots': [{'id': r['id'], 'node_id': r['node_id'], 'required_value': r['after']} for r in roots],
            'approved_claims': {k: {'after_value': v['after'], 'kind': v['kind'],
                                  'error_relative_to_clean': v['changed'], 'root_ids': v['root_ids'],
                                  'parents': v['parents']} for k, v in nodes.items()},
            'annotations': [{k: s[k] for k in ('start', 'end', 'text', 'node_id', 'kind', 'root_ids')}
                            for s in compiled['spans']],
            'semantic_bindings': compiled['bindings'], 'executed_operations': clean_plan,
            'root_evidence_options': [{'span_id': f's{i}', 'root_ids': s['root_ids'], 'text': s['text']}
                                     for i, s in enumerate(compiled['spans']) if s['kind'] == 'root_error']}
        audit = client.ask('control_audit', CONTROL_SOURCE, audit_payload, temperature=0)
        write_json(directory/'audits.json', {'blind': blind, 'compliance': audit})
        review_status = review_diagnostics(reference_review, audit, h, compiled['spans'], roots, blind)
        pair_id = row['origin_id']+'__'+PROTOCOL
        pairs = [{'record_id': pair_id+'__'+label, 'pair_id': pair_id, 'origin_id': row['origin_id'],
            'subtask': row['subtask'], 'variant_label': label, 'edit_count': len(roots) if label == 'H' else 0,
            'detector_input': {'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
                               'reasoning_chain': trace, 'final_answer': row['gt_smiles']}}
                 for label, trace in (('N', n), ('H', h))]
        if any(p['detector_input']['instruction'] != row['raw_record']['instruction']
               or p['detector_input']['indexed_smiles'] != row['raw_record']['indexed_smiles'] for p in pairs):
            raise ValueError('Original question changed')
        state.update({'status': 'accepted', 'pair_id': pair_id, 'pairs': pairs, 'plan': plan,
            'N_raw': row['N_raw'], 'N_source_visible': row['N_visible'], 'N_visible': n,
            'reference_projection': 'typed_reference_with_explicit_source_state_v1',
            'annotations': compiled['spans'], 'semantic_bindings': compiled['bindings'],
            'N_semantic_bindings': rendered['N_render']['bindings'], 'token_labels': token_labels,
            'text_edits': compiled.get('edits', []), 'source_contexts': rendered['contexts'],
            'error_counts': {'roots': len(roots), 'distinct_wrong_nodes': len({s['node_id'] for s in compiled['spans']}),
                             'error_spans': len(compiled['spans']), 'tokens': token_counts},
            'release_contract': {'version': 'program_verified_source_state_v1', 'status': 'pass',
                'reference_basis': 'Unchanged edit applied to the true source exactly reproduces benchmark GT.',
                'program_gates': 'Protected edit region; executable perceived source; unchanged reference edit; independent joint composition; conditional arithmetic; whole-claim/token labels; density; complete-product/cue exclusions.',
                'model_outputs_used_for_selection': False,
                'interpretation': 'Incorrect source-state uptake diagnostic; complete-source copying is a possible mechanism. Not a fitted natural error distribution.'},
            'review_status': review_status, 'reference_review': reference_review,
            'blind_audit': blind, 'audit': audit, 'actual_api_calls': client.calls})
        write_json(directory/'accepted.json', state)
        return state
    except Exception as error:
        state.update({'status': 'rejected', 'reason': str(error), 'error_type': type(error).__name__,
                      'actual_api_calls': client.calls if client else 0})
        write_json(directory/'rejected.json', state)
        return state
