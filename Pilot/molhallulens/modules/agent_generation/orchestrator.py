"""Frozen-reference, bounded agent corruption with executable chemistry evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import re
import time
from . import agents
from .poe_transport import PoeClient

PROTOCOL = 'agent_corruption_v12'


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def project_reference(text):
    text=re.sub(r'\s*-->\s*PRODUCT_SMILES\("[^"\n]*"\)', '',text)
    text=re.sub(r'\n\s*(?:Final\s+)?Answer:\s*[^\n]*', '',text,flags=re.I)
    from rdkit import Chem, rdBase
    def molecule(token):
        token=token.strip('`"').rstrip('.')
        with rdBase.BlockLogs():
            return bool(token and Chem.MolFromSmiles(token))
    # Versioned, symmetric CoT projection: only explicit product-answer contexts.
    def remove_answer(match):
        return '' if molecule(match.group(1)) else match.group(0)
    text=re.sub(r'(?i)\bProduct(?:\s+SMILES)?[ \t]*:[ \t]*(\S+)[ \t]*',remove_answer,text)
    def remove_inline(match):
        return 'Product ' if molecule(match.group(1)) else match.group(0)
    text=re.sub(r'(?i)\bProduct[ \t]+(\S+)[ \t]+(?=has\b)',remove_inline,text)
    text=re.sub(r'(?m)^[ \t]+\n','',text)
    return text.strip()


def load_origins(root):
    root=Path(root); result=[]
    for subtask in ['add','delete','substitute']:
        rawpath=root/f'Dataset/raw_benchmark_data/mol_edit/{subtask}_pilot_origin.json'
        procpath=root/f'Dataset/process_evaluation_data/mol_edit/{subtask}_pilot_origin.json'
        raw={x['anonymous_sample_id']:x for x in json.loads(rawpath.read_text())}
        for p in json.loads(procpath.read_text()):
            r=raw[p['anonymous_sample_id']]
            original='\n\n'.join(s['step_text'] for s in p['formal_cot_trace'])
            result.append({'origin_id':r['anonymous_sample_id'],'subtask':subtask,
                'indexed_smiles':r['indexed_smiles'],'instruction':r['instruction'],
                'gt_smiles':r['gt_smiles'],'N_raw':original,'N_visible':project_reference(original),
                'source_path':str(procpath.resolve()),'state':p['parsed_reference_state'],
                'reference_outcome':p['outcome'],'raw_record':r})
    return sorted(result,key=lambda x:x['origin_id'])


def make_split(rows, per_subtask=6, seed=20260920):
    rng=random.Random(seed); development=[]; heldout=[]
    for sub in sorted({r['subtask'] for r in rows}):
        ids=sorted(r['origin_id'] for r in rows if r['subtask']==sub)
        rng.shuffle(ids);development.extend(ids[:per_subtask]);heldout.extend(ids[per_subtask:])
    return {'seed':seed,'development':development,'heldout':heldout}


def anchored_patches(reference, patches):
    result=[]
    for p in patches:
        old=p['old'];new=p['new']
        if not old or reference.count(old)!=1:raise ValueError('Patch old text is absent or ambiguous')
        start=reference.index(old)
        spans=[]
        for c in p['claims']:
            text=c['text']
            if not text:raise ValueError('Empty error claim')
            hits=[m.start() for m in re.finditer(re.escape(text),new)]
            occurrence=c.get('occurrence',0)
            if type(occurrence) is not int or not 0<=occurrence<len(hits):raise ValueError('Claim occurrence absent')
            if len(hits)>1 and 'occurrence' not in c:raise ValueError('Ambiguous error claim')
            pos=hits[occurrence]
            spans.append({'start':pos,'end':pos+len(text),'kind':c['kind'],
                          'node_id':c['node_id'],'root_ids':c['root_ids']})
        # Empty spans explicitly mean style-only changed text, never mislabeled as erroneous.
        first=spans[0] if spans else {}
        result.append({'start':start,'end':start+len(old),'original':old,'replacement':new,
                       **{k:first[k] for k in ['kind','node_id','root_ids'] if k in first},'error_spans':spans})
    return result


def check_leakage(text, products):
    from .quality import find_answer_leakage
    findings=find_answer_leakage(text,products)
    if findings:raise ValueError('answer_leakage: '+str(findings))


def semantic_roots(clean, altered):
    roots=[]
    def add(node,before,after):
        if before!=after:roots.append({'id':f'r{len(roots)+1}','node_id':node,'before':before,'after':after,'type':'structural'})
    a=clean.get('add_fragments',[]);b=altered.get('add_fragments',[])
    if len(a)!=len(b):add('fragment',a,b)
    elif len(a)==1:
        from .chemistry_tools import compare_molecules
        same_fragment=compare_molecules(a[0]['smiles'],b[0]['smiles'])['equivalent']
        if not same_fragment:add('fragment',a[0]['smiles'],b[0]['smiles'])
        add('anchor',a[0]['anchor_map'],b[0]['anchor_map'])
        if same_fragment:
            from rdkit import Chem
            def attached_signature(f):
                m=Chem.MolFromSmiles(f['smiles']);m.GetAtomWithIdx(f.get('attach_atom_index',0)).SetAtomMapNum(999)
                return Chem.MolToSmiles(m)
            if attached_signature(a[0])!=attached_signature(b[0]):
                add('fragment_attachment',{'smiles':a[0]['smiles'],'index':a[0].get('attach_atom_index',0)}, {'smiles':b[0]['smiles'],'index':b[0].get('attach_atom_index',0)})
        add('bond_type',a[0].get('bond_type','SINGLE'),b[0].get('bond_type','SINGLE'))
    elif a!=b:add('fragment',a,b)
    add('remove_set',sorted(clean.get('remove_atom_maps',[])),sorted(altered.get('remove_atom_maps',[])))
    for key in ['remove_bonds','add_bonds','change_bonds']:
        add(key,clean.get(key,[]),altered.get(key,[]))
    # Hydrogens are required valence consequences, not counted as extra independent roots.
    return roots


def validate_audit(audit):
    required=['roots_complete','propagation_valid','labels_complete','no_extra_errors','no_leakage','no_obvious_cues']
    if audit.get('status')!='pass' or audit.get('reference_status')!='pass' or any(audit.get(k) is not True for k in required):
        raise ValueError('audit_uncertain_or_failed: '+json.dumps(audit.get('findings',[]),ensure_ascii=False)[:2000])


def prepare_roots(clean, altered, *, source_heavy, min_roots, mode):
    roots=semantic_roots(clean,altered)
    if not 1<=len(roots)<=3:raise ValueError('Requires one to three independent structural roots')
    if mode=='standard' and len(roots)<min_roots:
        roots.append({'id':f'r{len(roots)+1}','node_id':'source_heavy',
                      'before':source_heavy,'after':source_heavy+1,'type':'numeric_claim'})
    if len(roots)<min_roots:raise ValueError('Not enough structural roots for the declared contract')
    return roots


def choose_candidate(candidates, *, seed, mode):
    if not candidates:raise ValueError('No eligible candidates')
    if mode=='severe':
        return copy.deepcopy(min(candidates,key=lambda c:(c['severity']['molecular']['tanimoto'],c['candidate_id'])))
    if mode!='standard':raise ValueError('Unknown candidate mode')
    return copy.deepcopy(random.Random(seed).choice(candidates))


def validate_control_audit(audit, text, spans, roots):
    if not isinstance(audit,dict) or not isinstance(audit.get('matched_roots'),list) or not isinstance(audit.get('unexpected_defects'),list) or not isinstance(audit.get('reference_concerns'),list):
        raise ValueError('audit_invalid_schema')
    if audit['unexpected_defects'] or audit['reference_concerns']:
        raise ValueError('audit_unresolved_defects: '+json.dumps(audit,ensure_ascii=False))
    expected={r['id'] for r in roots};found=set()
    for finding in audit['matched_roots']:
        key=finding.get('root_id');quote=finding.get('quote')
        if 'span_id' in finding:
            span_id=finding['span_id']
            if not isinstance(span_id,str) or not re.fullmatch(r's[0-9]+',span_id):raise ValueError('audit_invalid_span_id')
            index=int(span_id[1:])
            if index>=len(spans):raise ValueError('audit_invalid_span_id')
            span=spans[index]
            if span['kind']!='root_error' or key not in span['root_ids']:raise ValueError('audit_span_root_mismatch')
            quote=text[span['start']:span['end']]
        if key not in expected or key in found or not isinstance(quote,str) or not quote:
            raise ValueError('audit_invalid_root_evidence')
        occurrences=[m.start() for m in re.finditer(re.escape(quote),text)] if len(quote)<=500 else []
        if not any(s['kind']=='root_error' and key in s['root_ids'] and pos<=s['start'] and s['end']<=pos+len(quote)
                   for pos in occurrences for s in spans):
            raise ValueError('audit_root_quote_does_not_contain_authorized_span')
        found.add(key)
    if found!=expected:raise ValueError('audit_missing_roots')


def reference_payload(row):
    return {'indexed_smiles':row['indexed_smiles'],'instruction':row['instruction'],'N':row['N_visible']}


def review_diagnostics(reference, audit, text, spans, roots, blind):
    """Preserve fallible model judgments without certifying them as passed proofs.

    Deterministic chemistry, rendering, leakage and annotation checks are the
    release contract. These separate model opinions never override those checks.
    """
    reference_ok=isinstance(reference,dict) and reference.get('consistent') is True and reference.get('concerns')==[]
    audit_error=None
    try:
        validate_control_audit(audit,text,spans,roots)
    except (ValueError,TypeError,KeyError,AttributeError) as error:
        audit_error=str(error)
    return {'role':'diagnostic_not_release_gate',
            'reference_status':'agreed' if reference_ok else 'disputed',
            'conformance_status':'agreed' if audit_error is None else 'disputed',
            'conformance_issue':audit_error,
            'agent_consensus':bool(reference_ok and audit_error is None),
            'agent_consensus_scope':'reference and conformance reviews only; blind observations stored separately',
            'blind_leakage_flag':blind.get('answer_leakage') if isinstance(blind,dict) else None,
            'interpretation':'Disputes are unresolved model opinions; acceptance means program-verified construction, not unanimous semantic certification.'}


def build_nodes(row, clean_plan, plan, roots, original, corrupted, *, include_local_environment=False):
    from .chemistry_tools import inspect_source,describe_fragment
    source=inspect_source(row['indexed_smiles']);nodes={}
    all_ids=[r['id'] for r in roots]; structural=[r['id'] for r in roots if r['type']=='structural']
    def add(name,before,after,parents,ids):
        root=next((r for r in roots if r['node_id']==name),None)
        nodes[name]={'before':before,'after':after,'changed':before!=after,'parents':parents,
                     'root_ids':[root['id']] if root else ids,'kind':('unchanged' if before==after else 'root_error' if root else 'propagated_error')}
    for r in roots:add(r['node_id'],r['before'],r['after'],[],[r['id']])
    sh=source['heavy_atoms'];claimed_sh=nodes.get('source_heavy',{}).get('after',sh)
    add('source_heavy',sh,claimed_sh,[],nodes.get('source_heavy',{}).get('root_ids',[]))
    claimed_product=claimed_sh+(corrupted['heavy_atoms']-sh)
    add('product_heavy',original['heavy_atoms'],claimed_product,['source_heavy','edit_plan'],all_ids)
    add('product_rings',original['rings'],corrupted['rings'],['edit_plan'],structural)
    add('heavy_delta',original['heavy_atoms']-sh,claimed_product-claimed_sh,['source_heavy','product_heavy'],all_ids)
    add('ring_delta',original['rings']-source['rings'],corrupted['rings']-source['rings'],['product_rings'],structural)
    a=clean_plan.get('add_fragments',[]);b=plan.get('add_fragments',[])
    if len(a)==len(b)==1:
        fa=describe_fragment(a[0]['smiles']);fb=describe_fragment(b[0]['smiles'])
        atoms={x['atom_map']:x for x in source['atoms']}
        anchor_roots=[r['id'] for r in roots if r['node_id']=='anchor']
        add('anchor_element',atoms[a[0]['anchor_map']]['element'],atoms[b[0]['anchor_map']]['element'],['anchor'],anchor_roots)
        fr=[r['id'] for r in roots if r['node_id']=='fragment']
        add('fragment_heavy',fa['heavy_atoms'],fb['heavy_atoms'],['fragment'],fr)
        add('fragment_composition',fa.get('formula'),fb.get('formula'),['fragment'],fr)
    sr=source['rings']
    add('source_rings',sr,sr,[],[])
    source_atoms={x['atom_map']:x for x in source['atoms']}
    removed_a=sum(source_atoms[m]['element']!='H' for m in clean_plan.get('remove_atom_maps',[]))
    removed_b=sum(source_atoms[m]['element']!='H' for m in plan.get('remove_atom_maps',[]))
    removal_roots=[r['id'] for r in roots if r['node_id']=='remove_set']
    add('remove_heavy',removed_a,removed_b,['remove_set'],removal_roots)
    if clean_plan.get('remove_atom_maps') or plan.get('remove_atom_maps'):
        from .candidate_tools import describe_removed_fragment
        ra=describe_removed_fragment(row['indexed_smiles'],clean_plan.get('remove_atom_maps',[]))
        rb=describe_removed_fragment(row['indexed_smiles'],plan.get('remove_atom_maps',[]))
        add('remove_fragment',ra['smiles'],rb['smiles'],['remove_set'],removal_roots)
    if include_local_environment:
        from .local_environment import derive_local_environment
        clean_local=derive_local_environment(row['indexed_smiles'],clean_plan,
            {**original,'reference_product_smiles':original['product_smiles']})
        altered_local=derive_local_environment(row['indexed_smiles'],plan,
            {**corrupted,'reference_product_smiles':original['product_smiles']})
        add('local_environment',clean_local['smiles'],altered_local['smiles'],['edit_plan'],structural)
        add('local_environment_center',clean_local['center_map'],altered_local['center_map'],['edit_plan'],structural)
        nodes['local_environment']['execution_evidence']={'before':clean_local,'after':altered_local,
            'reference_product_smiles':original['product_smiles']}
    return nodes


def generate_one(row, directory, *, seed=20260920, min_roots=1, min_nodes=6, min_tokens=40, candidate_mode='severe', tokenizers=None, model='gpt-5.4-mini'):
    from .chemistry_tools import apply_edit_plan,compare_molecules,inspect_source
    from .quality import validate_reference,validate_formal_claims,deterministic_reference_plan
    from .renderer import render_reasoning
    from .candidate_tools import enumerate_corruption_candidates
    from .labels import label_tokens
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    frozen={'protocol':PROTOCOL,'row':row,'seed':seed,'min_roots':min_roots,'min_nodes':min_nodes,'min_tokens':min_tokens,'model':model,'candidate_mode':candidate_mode,
        'role_prompts':{k:getattr(agents,k) for k in ['CLEAN_REFERENCE','SELECT','SELECT_SEVERE','BLIND_AUDIT','CONTROL_AUDIT']},'required_tokenizers':sorted((tokenizers or {}).keys())}
    config=directory/'input.json'
    if config.exists() and json.loads(config.read_text())!=frozen:raise ValueError('Resume input/config changed')
    write_json(config,frozen)
    for terminal in ['accepted','rejected']:
        if (directory/f'{terminal}.json').exists():return json.loads((directory/f'{terminal}.json').read_text())
    state={'origin_id':row['origin_id'],'status':'started','protocol':PROTOCOL}
    try:
        if set(tokenizers or {})!={'ChemDFM-R-14B','Chem-R-8B'}:raise ValueError('Both model tokenizers are required')
        source=inspect_source(row['indexed_smiles'])
        reference_checks=validate_reference(row)
        write_json(directory/'reference_checks.json',reference_checks)
        if reference_checks['status']!='pass':raise ValueError('reference_invalid: '+str(reference_checks['failures']))
        reference=deterministic_reference_plan(row)
        clean_plan=reference['edit_plan'];clean=reference['execution']
        write_json(directory/'reference.json',reference)
        clean_nodes=build_nodes(row,clean_plan,clean_plan,[],clean,clean,include_local_environment=True)
        clean_render=render_reasoning(row['indexed_smiles'],clean_plan,clean_nodes,label_errors=False)
        clean_visible=clean_render['text']
        check_leakage(clean_visible,[row['gt_smiles']])
        clean_formal=validate_formal_claims(clean_visible,clean_nodes,source)
        if clean_formal['status']!='pass':raise ValueError('renderer_reference_invalid: '+str(clean_formal))
        refpayload={'indexed_smiles':row['indexed_smiles'],'instruction':row['instruction'],'N':clean_visible,
                    'source_facts':{k:source[k] for k in ['heavy_atoms','rings','atoms']},'clean_plan':clean_plan,
                    'local_environment_facts':clean_nodes['local_environment']['execution_evidence']['before']}
        client=PoeClient(directory/'plan_calls',model=model,max_calls=10)
        reference_review=client.ask('clean_reference',agents.CLEAN_REFERENCE,{**refpayload,
            'reference_product_facts':{k:clean[k] for k in ['heavy_atoms','rings','formula']},'local_reference_checks':reference_checks},temperature=0)
        # The executable reference reproduces the benchmark graph exactly and
        # every template value is tool checked. A language-model opinion is
        # retained, not treated as an infallible veto over these typed proofs.
        feasible=[];pool_history=[]
        if candidate_mode=='severe':
            from .severe_candidates import enumerate_severe_candidates
            candidates=enumerate_severe_candidates(row,clean_plan,max_candidates=36)
        elif candidate_mode=='standard':
            candidates=enumerate_corruption_candidates(row,clean_plan,max_candidates=24)
        else:raise ValueError('Unknown candidate mode')
        by_id={c['candidate_id']:c for c in candidates}
        rejected=[]
        for index,candidate in enumerate(candidates):
            candidate_id=candidate['candidate_id']
            try:
                alternative=candidate['edit_plan']
                roots=prepare_roots(clean_plan,alternative,source_heavy=source['heavy_atoms'],min_roots=min_roots,mode=candidate_mode)
                altered=candidate['execution']
                if compare_molecules(clean['product_smiles'],altered['product_smiles'])['equivalent']:
                    raise ValueError('Diagnostic subset requires a different product')
                # N is frozen before any H candidate exists. Reject a candidate
                # that would turn an existing N fragment into its full answer.
                check_leakage(clean_visible,[altered['product_smiles']])
                check_leakage(clean_nodes['local_environment']['after'],
                    [altered['product_smiles'],*altered['product_smiles'].split('.')])
                nodes=build_nodes(row,clean_plan,alternative,roots,clean,altered,include_local_environment=True)
                candidate_render=render_reasoning(row['indexed_smiles'],alternative,nodes,label_errors=True)
                observed={s['node_id'] for s in candidate_render['spans']}
                density={name:sum(any(label!='unchanged' for label in t['labels']) for t in label_tokens(candidate_render['text'],candidate_render['spans'],tok)) for name,tok in tokenizers.items()}
                if min(density.values())<min_tokens:raise ValueError('Not enough hallucinated tokens')
                if len(observed)<min_nodes:raise ValueError('Not enough distinct wrong rendered semantic nodes')
                checks=validate_formal_claims(candidate_render['text'],nodes,source)
                if checks['status']!='pass':raise ValueError('renderer_claims_invalid: '+str(checks))
                check_leakage(candidate_render['text'],[row['gt_smiles'],altered['product_smiles']])
                from .severity import measure_severity
                severity=measure_severity(row['indexed_smiles'],clean,altered,
                    clean_trace=clean_visible,h_trace=candidate_render['text'],annotations=candidate_render['spans'])
                feasible.append({'candidate_index':index,'candidate_id':candidate_id,'edit_plan':alternative,'severity':severity,
                                 'roots':roots,'nodes':nodes,'execution':altered,'render':candidate_render,'checks':checks})
            except (ValueError,KeyError,TypeError) as e:
                rejected.append({'candidate_id':candidate_id,'error':str(e)})
        write_json(directory/'candidate_pool.json',{'all_candidates':candidates,'eligible':feasible,'rejected':rejected})
        if not feasible:raise ValueError('no_feasible_plan')
        selection=client.ask('select',agents.SELECT_SEVERE if candidate_mode=='severe' else agents.SELECT,{**refpayload,'candidates':[
            {k:by_id[c['candidate_id']][k] for k in ['candidate_id','edit_plan','description']} for c in feasible]})
        ids=selection.get('candidate_ids');eligible={c['candidate_id'] for c in feasible}
        # Versioned alias grammar: cNNN is exactly the same candidate as candidate_NNN.
        # This normalizes identifiers only, never edits or replaces a proposed plan.
        if isinstance(ids,list):ids=[('candidate_'+i[1:]) if isinstance(i,str) and i not in eligible and re.fullmatch(r'c[0-9]{3}',i) else i for i in ids]
        if not isinstance(ids,list) or not 0<len(ids)<=6 or len(set(ids))!=len(ids) or any(i not in eligible for i in ids):
            raise ValueError('no_feasible_plan: invalid/empty candidate selection')
        feasible=[c for c in feasible if c['candidate_id'] in ids]
        write_json(directory/'candidate_selection.json',{'selection':selection,'feasible_ids':[c['candidate_id'] for c in feasible]})
        selected=choose_candidate(feasible,seed=seed,mode=candidate_mode)
        plan={**selected,'selection_probability':1.0 if candidate_mode=='severe' else 1/len(feasible),'seed':seed,
              'sampling_policy':('max_graph_distance_among_agent_approved_feasible_candidates' if candidate_mode=='severe'
                                 else 'program_executable_pool_agent_plausibility_uniform_structural_then_numeric_supplement'),
              'candidate_mode':candidate_mode,'min_roots':min_roots}
        compiled=plan['render'];h=compiled['text']
        token_labels={name:label_tokens(h,compiled['spans'],tok) for name,tok in tokenizers.items()}
        plan['severity']=measure_severity(row['indexed_smiles'],clean,plan['execution'],
            clean_trace=clean_visible,h_trace=h,annotations=compiled['spans'],token_labels=token_labels)
        write_json(directory/'fixed_plan.json',plan)
        token_counts={name:sum(any(x!='unchanged' for x in t['labels']) for t in labels) for name,labels in token_labels.items()}
        if min(token_counts.values())<min_tokens:raise ValueError('Too few hallucinated tokens')
        observed={s['node_id'] for s in compiled['spans']}
        root_observed={r for s in compiled['spans'] if s['kind']=='root_error' for r in s['root_ids']}
        if root_observed!={r['id'] for r in plan['roots']}:raise ValueError('Missing root annotation')
        # Isolated blind review sees question/trace only. Plan review gets independently recomputed facts.
        blind=client.ask('blind',agents.BLIND_AUDIT,{'indexed_smiles':row['indexed_smiles'],'instruction':row['instruction'],'reasoning':h},temperature=0)
        auditpayload={'H':h,'fixed_roots':[{'id':r['id'],'node_id':r['node_id'],'required_value':r['after']} for r in plan['roots']],
            'approved_claims':{key:{'after_value':node['after'],'kind':node['kind'],'error_relative_to_clean':node['changed'],'root_ids':node['root_ids'],'parents':node['parents']} for key,node in plan['nodes'].items()},
            'annotations':[{k:s[k] for k in ['start','end','text','node_id','kind','root_ids']} for s in compiled['spans']], 'executed_operations':plan['edit_plan'],'semantic_bindings':compiled['bindings'],'root_evidence_options':[{'span_id':f's{i}','root_ids':s['root_ids'],'text':s['text']} for i,s in enumerate(compiled['spans']) if s['kind']=='root_error'],
            'arithmetic_policy':'claimed_product_heavy = claimed_source_heavy + fragment_heavy - remove_heavy; heavy_delta = claimed_product_heavy - claimed_source_heavy; ring_delta = product_rings - source_rings'}
        audit=client.ask('control_audit',agents.CONTROL_AUDIT,auditpayload,temperature=0)
        write_json(directory/'audits.json',{'blind':blind,'compliance':audit})
        review_status=review_diagnostics(reference_review,audit,h,compiled['spans'],plan['roots'],blind)
        pair_id=row['origin_id']+'__'+PROTOCOL
        pairs=[]
        for label,trace in [('N',clean_visible),('H',h)]:
            pairs.append({'record_id':pair_id+'__'+label,'pair_id':pair_id,'origin_id':row['origin_id'],'subtask':row['subtask'],
                'variant_label':label,'edit_count':len(plan['roots']) if label=='H' else 0,
                'detector_input':{'indexed_smiles':row['indexed_smiles'],'instruction':row['instruction'],'reasoning_chain':trace,'final_answer':row['gt_smiles']}})
        assert all(r['detector_input']['instruction']==row['raw_record']['instruction'] for r in pairs)
        state.update({'status':'accepted','pair_id':pair_id,'pairs':pairs,'plan':plan,'N_raw':row['N_raw'],
            'N_source_visible':row['N_visible'],'N_visible':clean_visible,'reference_projection':'typed_reference_semantics_v4_local_environment',
            'annotations':compiled['spans'],'semantic_bindings':compiled['bindings'],'N_semantic_bindings':clean_render['bindings'],'text_edits':compiled['edits'],'token_labels':token_labels,
            'error_counts':{'roots':len(plan['roots']),'distinct_wrong_nodes':len(observed),'error_spans':len(compiled['spans']),'tokens':token_counts},
            'release_contract':{'version':'program_verified_semantic_construction_v1',
                'reference_basis':'Frozen benchmark reference edit exactly reproduces benchmark GT graph',
                'program_gates':'Executable chemistry, stereo policy, conditional arithmetic, deterministic semantic bindings, complete root/changed-node labels, density and full-answer/cue exclusions',
                'status':'pass','model_outputs_used_for_selection':False},
            'review_status':review_status,
            'reference_review':reference_review,'blind_audit':blind,'audit':audit,'actual_api_calls':client.calls})
        write_json(directory/'accepted.json',state);return state
    except Exception as e:
        state.update({'status':'rejected','reason':str(e),'error_type':type(e).__name__,'actual_api_calls':client.calls if 'client' in locals() else 0})
        write_json(directory/'rejected.json',state);return state
