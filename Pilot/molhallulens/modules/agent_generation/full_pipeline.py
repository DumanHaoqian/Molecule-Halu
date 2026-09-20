"""Full-corpus, outcome-independent task corruption and isolated agent reviews."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
from rdkit import rdBase
from .orchestrator import write_json
from .candidate_tools import _source_graph, _hydrogen_plan
from .full_reference import full_reference, execute_full_plan
from .full_renderer import render_full_reasoning
from .severe_candidates import _proposal_pool, _balanced_proposals
from .quality import find_answer_leakage
from .severity import measure_severity
from .labels import label_tokens
from .poe_transport import PoeClient

PROTOCOL = 'agent_full_task_interpretation_v1'
CLEAN_PROMPT = '''You are an independent chemistry reference reviewer. Return JSON with status (pass or concerns) and concerns (array). Review only this original question and its original reasoning against the executable reference. Check task interpretation and chemistry. Distinguish an incorrect original prose claim from a graph-execution error. Do not rewrite the instruction or reasoning. You see no corrupted candidates.'''
SELECT_PROMPT = '''You are a chemistry error-construction agent. Each candidate is an executable mistaken interpretation of a fixed molecule-edit instruction, with downstream chemistry recomputed. Select exactly one candidate whose mistaken operation is intelligible as a task interpretation mistake and whose consequences form a coherent chain. Candidate severity is supplied, but no target-model results are available or may be inferred. Return JSON with candidate_id (one exact supplied string) and rationale. Do not edit any text, add answers, or invent a candidate.'''
AUDIT_PROMPT = '''You are an independent conditional chemistry auditor. Only the H string is visible to the target model; all other fields are private audit evidence. Expected task-requirement mistakes are the intervention, not by themselves a conditional-consistency defect. Examine leakage ONLY inside H. Return JSON with status (pass or concerns), concerns (array), and explanation. The supplied H intentionally contains mistaken task requirements relative to the immutable question. Check whether the downstream statements follow chemically from those specified operations, and whether the supplied labelled values differ from the verified clean execution. Repeated mentions are not separate semantic errors. Distinguish conditional consistency from compliance with the original instruction. Report any unlabelled false claim, full-product answer leak, explicit intervention cue, or unsupported dependency. Do not rewrite anything.'''


def project_row(row):
    result = deepcopy(row)
    kept, removed = [], []
    for line in row['N_visible'].splitlines(keepends=True):
        if find_answer_leakage(line, [row['gt_smiles']]):
            removed.append(line)
        else:
            kept.append(line)
    result['N_visible'] = ''.join(kept).strip()
    result['projection_removed_lines'] = removed
    if not result['N_visible'] or find_answer_leakage(result['N_visible'], [row['gt_smiles']]):
        raise ValueError('Original N cannot be projected without unresolved answer leakage')
    return result


def normalize_candidate(source_smiles, clean_plan, proposal):
    plan = deepcopy(proposal)
    old, new = clean_plan.get('add_fragments', []), plan.get('add_fragments', [])
    caps = []
    for cap in plan.get('fragment_hydrogen_caps', []):
        i = cap['fragment_index']
        if i < len(old) and i < len(new) and all(old[i].get(k) == new[i].get(k) for k in ('smiles','attach_atom_index')):
            caps.append(cap)
    if caps: plan['fragment_hydrogen_caps'] = caps
    else: plan.pop('fragment_hydrogen_caps', None)
    assignments = [a for a in plan.get('assign_tetrahedral', []) if a['atom_map'] not in plan.get('remove_atom_maps', [])]
    if assignments: plan['assign_tetrahedral'] = assignments
    else: plan.pop('assign_tetrahedral', None)
    _, source = _source_graph(source_smiles)
    return _hydrogen_plan(source, plan)


def selected_id(response, candidates):
    ids = [c['candidate_id'] for c in candidates]
    value = response.get('candidate_id')
    if len(set(ids)) != len(ids) or not isinstance(value, str) or value not in ids:
        raise ValueError('Selector must return exactly one known candidate_id')
    return value


def prepare_row(row, directory, tokenizers, pool_size=12):
    directory = Path(directory)
    projected = project_row(row)
    existing = directory/'input.json'
    data = {'protocol':PROTOCOL, 'row':projected, 'pool_size':pool_size, 'min_nodes':6, 'min_tokens':40}
    if existing.exists() and json.loads(existing.read_text()) != data:
        raise ValueError('Frozen generation input changed')
    write_json(existing, data)
    if (directory/'candidate_pool.json').exists():
        return json.loads((directory/'prepared.json').read_text())
    with rdBase.BlockLogs():
        reference = full_reference(projected)
        write_json(directory/'reference.json', reference)
        facts, source = _source_graph(projected['indexed_smiles'])
        proposals = _balanced_proposals(_proposal_pool(source, facts, reference['edit_plan']))
        executed, seen, errors = [], set(), Counter()
        for proposal in proposals:
            try:
                plan = normalize_candidate(projected['indexed_smiles'], reference['edit_plan'], proposal['edit_plan'])
                key = json.dumps(plan, sort_keys=True)
                if key in seen: continue
                seen.add(key)
                execution = execute_full_plan(projected['indexed_smiles'], plan)
                severity = measure_severity(projected['indexed_smiles'], reference['execution'], execution)
                if severity['product_tanimoto'] == 1 and execution['product_smiles'] == reference['execution']['product_smiles']:
                    continue
                executed.append({'edit_plan':plan, 'execution':execution, 'severity':severity, 'family':proposal['family']})
            except Exception as exc:
                errors[type(exc).__name__+': '+str(exc)[:180]] += 1
        executed.sort(key=lambda c:(c['severity']['product_tanimoto'], -c['severity']['source_footprint_ratio'], json.dumps(c['edit_plan'],sort_keys=True)))
        candidates = []
        for candidate in executed:
            try:
                rendered = render_full_reasoning(projected, reference, candidate['edit_plan'], candidate['execution'], tokenizers=tokenizers)
                if rendered['checks']['changed_node_count'] < 6 or min(rendered['checks']['annotated_tokens'].values()) < 40:
                    raise ValueError('Below fixed semantic-node/token minimum')
                candidates.append({**candidate, 'candidate_id':f'candidate_{len(candidates)+1:03d}', 'rendered':rendered})
                if len(candidates) == pool_size: break
            except Exception as exc:
                errors[type(exc).__name__+': '+str(exc)[:180]] += 1
    state = {'origin_id':row['origin_id'], 'status':'prepared' if candidates else 'rejected', 'proposal_count':len(seen), 'executable_count':len(executed), 'eligible_pool_count':len(candidates), 'reasons':dict(errors)}
    write_json(directory/'prepared.json', state)
    if candidates: write_json(directory/'candidate_pool.json', candidates)
    return state


def review_row(directory, tokenizers):
    directory = Path(directory)
    if (directory/'accepted.json').exists(): return json.loads((directory/'accepted.json').read_text())
    data = json.loads((directory/'input.json').read_text()); row = data['row']
    client = PoeClient(directory/'api', max_calls=12)
    state = {'origin_id':row['origin_id'], 'protocol':PROTOCOL}
    try:
        candidates = json.loads((directory/'candidate_pool.json').read_text())
        reference = json.loads((directory/'reference.json').read_text())
        question = {k:row[k] for k in ('indexed_smiles','instruction')}
        def ask(role, prompt, payload, validator=None):
            failure = None
            for attempt in range(3):
                try:
                    answer = client.ask(role+f'_{attempt}', prompt, payload, temperature=0)
                    if validator: validator(answer)
                    return answer
                except Exception as exc: failure = exc
            raise failure
        clean = ask('clean_review', CLEAN_PROMPT, {**question,'N':row['N_visible'],'reference':reference})
        selector = ask('selector', SELECT_PROMPT, {**question,'clean_edit':reference['edit_plan'], 'candidates':[{k:c[k] for k in ('candidate_id','edit_plan','severity','family')} for c in candidates]}, lambda a:selected_id(a,candidates))
        chosen = next(c for c in candidates if c['candidate_id']==selector['candidate_id'])
        rendered = chosen['rendered']; h = rendered['H_render']; n = rendered['N_render']
        audit = ask('conditional_audit', AUDIT_PROMPT, {**question, 'H':h['text'],'edit_plan':chosen['edit_plan'],'execution_facts':{k:chosen['execution'][k] for k in ('heavy_atoms','rings','formula','formal_charge')},'nodes':rendered['nodes'],'roots':rendered['roots'],'annotations':h['spans']})
        write_json(directory/'audits.json', {'clean_reference':clean,'selector':selector,'conditional_fidelity':audit})
        labels = {name:label_tokens(h['text'], h['spans'], tok) for name,tok in tokenizers.items()}
        counts = {name:sum(any(v!='unchanged' for v in t['labels']) for t in ts) for name,ts in labels.items()}
        if min(counts.values()) < 40: raise ValueError('Frozen candidate token-label gate changed')
        plan = {'protocol':PROTOCOL, **{k:chosen[k] for k in ('edit_plan','execution','severity','candidate_id')}, 'render':h,'roots':rendered['roots'],'nodes':rendered['nodes'],'checks':rendered['checks']}
        pair_id = row['origin_id']+'__'+PROTOCOL
        pairs = [{'record_id':pair_id+'__'+v,'pair_id':pair_id,'origin_id':row['origin_id'],'subtask':row['subtask'],'variant_label':v,'edit_count':len(plan['roots']) if v=='H' else 0,'detector_input':{**question,'reasoning_chain':t,'final_answer':row['gt_smiles']}} for v,t in [('N',n['text']),('H',h['text'])]]
        state.update(status='accepted',pair_id=pair_id,pairs=pairs,plan=plan,N_raw=row['N_raw'],N_source_visible=row['N_visible'],N_visible=n['text'],annotations=h['spans'],semantic_bindings=h['bindings'],N_semantic_bindings=[],text_edits=h['edits'],token_labels=labels,error_counts={'roots':len(plan['roots']),'distinct_wrong_nodes':rendered['checks']['changed_node_count'],'error_spans':len(h['spans']),'tokens':counts},review_status={'clean_reference':clean,'conditional_fidelity':audit,'gating':'Program construction gates; agent disagreements retained, not unanimous approval'},release_contract={'protocol':PROTOCOL,'status':'pass','model_outputs_used_for_selection':False,'style_confound':rendered['checks']['style_confound']},actual_api_calls=client.calls)
        write_json(directory/'fixed_plan.json',plan)
        write_json(directory/'accepted.json',state)
    except Exception as exc:
        state.update(status='rejected',reason=str(exc),error_type=type(exc).__name__,actual_api_calls=client.calls)
        write_json(directory/'rejected.json',state)
    return state
