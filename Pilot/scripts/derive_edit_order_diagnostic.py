#!/usr/bin/env python3
"""CPU-only fixed-v12 edit-order diagnostic; no resampling or model calls.

Publish account_last and edit_last together in a fresh directory. Both omit the
optional local-product window and retain exactly the same chemical plans/claims.
Every parent accepted origin remains accepted for diagnostic use; all original
rejections and density exceptions remain in the planning denominator. No fresh
Agent review or production-quality approval is implied by derived acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from molhallulens.modules.agent_generation.chemistry_tools import apply_edit_plan,compare_molecules,inspect_source
from molhallulens.modules.agent_generation.labels import label_tokens
from molhallulens.modules.agent_generation.orchestrator import build_nodes,semantic_roots
from molhallulens.modules.agent_generation.quality import find_answer_leakage,validate_formal_claims
from molhallulens.modules.agent_generation.renderer import render_reasoning
from molhallulens.modules.agent_generation.severity import measure_severity

PROTOCOL='agent_edit_order_diagnostic_v1'
ORDERS=('account_last','edit_last')
MODEL_NAMES=('ChemDFM-R-14B','Chem-R-8B')
TITLES=('Identify the source and the edit.','Apply the edit.','Account for the atoms and rings.')
STATUS_MEANING='accepted_for_paired_diagnostic_only'


class FrozenTokenizer:
    """Load the exact saved fast backend; never load model weights or download."""
    is_fast=True
    def __init__(self,backend,name):
        from tokenizers import Tokenizer
        self.backend=Tokenizer.from_str(backend);self.name_or_path=name
    def __call__(self,text,*,add_special_tokens=False,return_offsets_mapping=True):
        encoded=self.backend.encode(text,add_special_tokens=add_special_tokens)
        return {'input_ids':encoded.ids,'offset_mapping':encoded.offsets}
    def convert_ids_to_tokens(self,ids):
        return [self.backend.id_to_token(i) for i in ids]


def _require(condition,message):
    if not condition:raise ValueError(message)


def _blocks(text):
    headings=list(re.finditer(r'^Step ([1-3]): ([^\n]+)\n',text,re.M))
    _require(len(headings)==3 and headings[0].start()==0,'Expected exactly three canonical steps')
    _require([int(m[1]) for m in headings]==[1,2,3],'Step numbers must follow display order')
    _require({m[2] for m in headings}==set(TITLES),'Unexpected step titles')
    result={}
    for index,match in enumerate(headings):
        end=headings[index+1].start()-2 if index<2 else len(text)
        if index<2:_require(text[end:end+2]=='\n\n','Expected the canonical block separator')
        _require(end>match.end(),'Empty canonical step')
        result[match[2]]={'start':match.start(),'end':end,'text':text[match.start():end]}
    return result


def normalize_order(text):
    """Undo the section permutation and normalize only its step-number digits."""
    blocks=_blocks(text)
    return '\n\n'.join(re.sub(r'^Step [1-3]:',f'Step {i}:',blocks[title]['text'],count=1)
                       for i,title in enumerate(TITLES,1))


def reorder_render(rendered,order):
    """Move complete blocks; every annotation must remain inside its own block."""
    _require(order in ORDERS,'Unknown order')
    text=rendered['text'];blocks=_blocks(text)
    _require(normalize_order(text)==text,'Input render must use canonical account_last order')
    sequence=TITLES if order=='account_last' else (TITLES[0],TITLES[2],TITLES[1])
    target=[];transform=[];cursor=0
    for index,title in enumerate(sequence,1):
        old=blocks[title];block=re.sub(r'^Step [1-3]:',f'Step {index}:',old['text'],count=1)
        target.append(block)
        transform.append({'title':title,'source_start':old['start'],'source_end':old['end'],
                          'start':cursor,'end':cursor+len(block)})
        cursor+=len(block)+2
    result={'text':'\n\n'.join(target),'spans':[],'bindings':[],'edits':[],
            'order_transform':{'order':order,'blocks':transform}}
    for field in ('spans','bindings'):
        for item in rendered[field]:
            owners=[b for b in transform if b['source_start']<=item['start']<item['end']<=b['source_end']]
            _require(len(owners)==1,'A semantic slot crosses a step boundary')
            block=owners[0];shift=block['start']-block['source_start'];moved=deepcopy(item)
            moved['start']+=shift;moved['end']+=shift
            _require(result['text'][moved['start']:moved['end']]==item['text'],'Moved semantic value changed')
            result[field].append(moved)
        result[field].sort(key=lambda x:(x['start'],x['end']))
    _require(normalize_order(result['text'])==text,'Reverse permutation changed text content')
    return result


def _without_local(text):
    return text.split('\n\nStep 4: ',1)[0]


def derive_record(parent,parent_input,reference,*,tokenizers,parent_directory,order):
    """Derive one ordered pair while preserving the exact selected chemistry."""
    _require(order in ORDERS,'Unknown order')
    _require(set(tokenizers)==set(MODEL_NAMES),'Both frozen tokenizers are required')
    _require(parent.get('status')=='accepted' and parent.get('protocol')=='agent_corruption_v12','Expected an accepted v12 parent')
    row=parent_input['row'];origin=row['origin_id'];directory=Path(parent_directory).resolve()
    pairs={p['variant_label']:p for p in parent['pairs']}
    _require(len(parent['pairs'])==2 and set(pairs)=={'N','H'},'Parent must have exactly one N/H pair')
    for pair in pairs.values():
        _require(pair['origin_id']==parent['origin_id']==origin and pair['pair_id']==parent['pair_id'] and pair['subtask']==row['subtask'],'Parent pair metadata mismatch')
        for field,key in [('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')]:
            _require(pair['detector_input'][field]==row[key]==row['raw_record'][key],'Original benchmark question or GT changed')
    _require(parent['N_visible']==pairs['N']['detector_input']['reasoning_chain'],'Parent N disagrees with its pair')
    clean_plan,plan=reference['edit_plan'],parent['plan']['edit_plan']
    clean=apply_edit_plan(row['indexed_smiles'],clean_plan);altered=apply_edit_plan(row['indexed_smiles'],plan)
    _require(clean==reference['execution'],'Parent clean execution drift')
    _require(altered==parent['plan']['execution'],'Parent H execution drift')
    _require(compare_molecules(clean['product_smiles'],row['gt_smiles'])['equivalent'],'Reference product differs from GT')
    _require(not compare_molecules(clean['product_smiles'],altered['product_smiles'])['equivalent'],'H product must differ')
    roots=deepcopy(parent['plan']['roots'])
    _require(roots and all(r['type']=='structural' for r in roots),'Only the unchanged v12 structural roots are supported')
    without_ids=lambda values:[{k:v for k,v in item.items() if k!='id'} for item in values]
    _require(without_ids(roots)==without_ids(semantic_roots(clean_plan,plan)),'Root semantics differ from fixed edit plans')
    clean_nodes=build_nodes(row,clean_plan,clean_plan,[],clean,clean,include_local_environment=False)
    nodes=build_nodes(row,clean_plan,plan,roots,clean,altered,include_local_environment=False)
    prior_nodes={k:v for k,v in parent['plan']['nodes'].items() if k not in {'local_environment','local_environment_center'}}
    _require(nodes==prior_nodes,'Nonlocal semantic nodes differ from the parent')
    nr=render_reasoning(row['indexed_smiles'],clean_plan,clean_nodes,label_errors=False)
    hr=render_reasoning(row['indexed_smiles'],plan,nodes,label_errors=True)
    for label,rendered in [('N',nr),('H',hr)]:
        _require(rendered['text']==_without_local(pairs[label]['detector_input']['reasoning_chain']),label+' parent body drift outside removed local window')
    nr,hr=(reorder_render(rendered,order) for rendered in (nr,hr))
    source=inspect_source(row['indexed_smiles']);checks={}
    for label,rendered,approved in [('N',nr,clean_nodes),('H',hr,nodes)]:
        formal=validate_formal_claims(rendered['text'],approved,source)
        _require(formal['status']=='pass',label+' formal claim failure')
        leaks=find_answer_leakage(rendered['text'],[row['gt_smiles'],altered['product_smiles']])
        _require(not leaks,label+' full-product/cue leakage')
        checks[label]={'formal_claims':formal,'answer_leakage':leaks,'reverse_order_verified':True}
    _require({root for span in hr['spans'] if span['kind']=='root_error' for root in span['root_ids']}=={r['id'] for r in roots},'A retained structural root lost all annotations')
    labels={name:label_tokens(hr['text'],hr['spans'],tok) for name,tok in tokenizers.items()}
    counts={name:sum(any(label!='unchanged' for label in t['labels']) for t in records) for name,records in labels.items()}
    wrong_nodes={s['node_id'] for s in hr['spans']}
    thresholds={key:parent_input[key] for key in ('min_roots','min_nodes','min_tokens')}
    violations=[]
    if len(roots)<thresholds['min_roots']:violations.append('min_roots')
    if len(wrong_nodes)<thresholds['min_nodes']:violations.append('min_nodes')
    if min(counts.values())<thresholds['min_tokens']:violations.append('min_tokens')
    pair_id=origin+'__'+PROTOCOL;new_pairs=[]
    for label,rendered in [('N',nr),('H',hr)]:
        pair=deepcopy(pairs[label]);pair.update(pair_id=pair_id,record_id=pair_id+'__'+label,edit_count=len(roots) if label=='H' else 0)
        pair['detector_input']['reasoning_chain']=rendered['text'];new_pairs.append(pair)
    result=deepcopy(parent)
    result.update(protocol=PROTOCOL,order=order,pair_id=pair_id,pairs=new_pairs,status='accepted',
                  acceptance_scope='paired_diagnostic_only',N_visible=nr['text'],N_semantic_bindings=nr['bindings'],
                  reference_projection='typed_reference_without_local_window_edit_order_v1',
                  annotations=hr['spans'],semantic_bindings=hr['bindings'],text_edits=[],token_labels=labels,actual_api_calls=0,
                  error_counts={'roots':len(roots),'distinct_wrong_nodes':len(wrong_nodes),'error_spans':len(hr['spans']),'tokens':counts})
    result['plan'].update(roots=roots,nodes=nodes,render=hr,checks=checks,diagnostic_order=order,
        sampling_policy='fixed_parent_plan_edit_order_no_resampling',
        severity=measure_severity(row['indexed_smiles'],clean,altered,clean_trace=nr['text'],h_trace=hr['text'],annotations=hr['spans'],token_labels=labels))
    for key in ('selection_probability','selection_probability_scope','seed'):result['plan'].pop(key,None)
    result['density_report']={'parent_thresholds':thresholds,'violations':violations,'filtered':False}
    result['release_contract']={'version':PROTOCOL,'status':'diagnostic_only','diagnostic_only':True,
        'production_eligible':False,'status_meaning':STATUS_MEANING,
        'program_gates':'Fixed v12 plans/roots/products; no local window; same chemical content after inverse order normalization; exact question/GT; rebuilt formal/spans/tokens',
        'density_policy':'All parent accepted pairs retained; exceptions reported, never filtered',
        'model_outputs_used_for_selection':False}
    result['historical_parent_reviews']={k:deepcopy(parent.get(k)) for k in ('reference_review','blind_audit','audit','review_status')}
    result['reference_review']={'status':'unknown_not_repeated','role':'Historical parent review retained separately; derived N changed'}
    result['reference_review_provenance']='Historical only; local removal/order changed N; no fresh model review'
    result['audit']={'status':'unknown_not_repeated','role':'No fresh conformance review'}
    result['blind_audit']={'status':'unknown_not_repeated','answer_leakage':None}
    result['review_status']={'role':'historical_parent_reviews_only','reference_status':'unknown_not_repeated',
        'conformance_status':'unknown_not_repeated','blind_status':'unknown_not_repeated','agent_consensus':None,
        'interpretation':'No Agent review was repeated or claimed for either ordered diagnostic text'}
    result['derivation']={'parent_protocol':parent['protocol'],'parent_pair_id':parent['pair_id'],
        'parent_origin_directory':str(directory),'parent_accepted_path':str(directory/'accepted.json'),
        'parent_actual_api_calls':parent.get('actual_api_calls',0),'derivation_api_calls':0,
        'order':order,'removed_nodes':['local_environment','local_environment_center'],
        'unchanged_structural_plan':True,'unchanged_roots':True,'unchanged_executed_product':True,
        'unchanged_clean_trace':False,'same_content_across_orders':True,'inverse_order_normalization_verified':True,
        'N_order_transform':nr['order_transform'],'H_order_transform':hr['order_transform'],
        'parent_selection_metadata':{k:deepcopy(parent['plan'].get(k)) for k in ('candidate_id','seed','selection_probability','sampling_policy')}}
    return result


def _read(path):return json.loads(path.read_text())
def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def materialize(parent_batch,output):
    """Validate the whole terminal parent batch before publishing both orders."""
    parent_batch,output=Path(parent_batch).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh diagnostic output directory: '+str(output))
    _require(parent_batch not in output.parents and parent_batch!=output,'Output cannot overwrite or enter parent batch')
    manifest=_read(parent_batch/'manifest.json');summary=_read(parent_batch/'summary.json')
    _require(manifest['protocol']=='agent_corruption_v12','Expected a v12 parent batch')
    _require(summary['processed']==summary['planned'],'Parent generation is not terminal')
    selected=manifest['selected'];_require(len(selected)==len(set(selected))==summary['planned'],'Parent selected denominator differs')
    rows=[json.loads(line) for line in (parent_batch/'pairs.jsonl').read_text().splitlines() if line.strip()]
    origins=sorted({r['origin_id'] for r in rows});found=sorted(p.parent.name for p in (parent_batch/'origins').glob('*/accepted.json'))
    rejected=sorted(p.parent.name for p in (parent_batch/'origins').glob('*/rejected.json'))
    _require(found==origins and len(rows)==len(origins)*2 and len(origins)==summary['accepted'],'Parent accepted records/dataset differ')
    _require(set(origins).isdisjoint(rejected) and set(origins)|set(rejected)==set(selected) and len(rejected)==summary['rejected'],'Parent rejections or planning denominator missing')
    snapshots=_read(parent_batch/'tokenizer_snapshots.json');_require(set(snapshots)==set(MODEL_NAMES),'Incomplete frozen tokenizer set')
    tokenizers={name:FrozenTokenizer(snapshots[name]['backend'],name) for name in MODEL_NAMES}
    results={order:[] for order in ORDERS};sources={}
    for origin in origins:
        directory=parent_batch/'origins'/origin
        parent,inp,reference=(_read(directory/(name+'.json')) for name in ('accepted','input','reference'))
        _require(sorted([r for r in rows if r['origin_id']==origin],key=lambda r:r['variant_label'])==sorted(parent['pairs'],key=lambda r:r['variant_label']),'Parent pair file differs from accepted record')
        for name,tok in tokenizers.items():
            prior=parent.get('token_labels',{}).get(name,[]);tok.name_or_path=prior[0]['tokenizer'] if prior else name
            if prior:
                text=next(r['detector_input']['reasoning_chain'] for r in parent['pairs'] if r['variant_label']=='H')
                _require(label_tokens(text,parent['annotations'],tok)==prior,'Parent token labels do not match frozen backend')
        for order in ORDERS:results[order].append(derive_record(parent,inp,reference,tokenizers=tokenizers,parent_directory=directory,order=order))
        sources[origin]={'accepted':parent,'input':inp,'reference':reference}
    for a,b in zip(results['account_last'],results['edit_last']):
        _require(a['origin_id']==b['origin_id'] and a['plan']['nodes']==b['plan']['nodes'],'Ordered condition semantics differ')
        for pa,pb in zip(a['pairs'],b['pairs']):
            _require(normalize_order(pa['detector_input']['reasoning_chain'])==normalize_order(pb['detector_input']['reasoning_chain']),'Inverse text permutation differs between conditions')
    parent_order=[r['origin_id'] for r in sorted(rows,key=lambda r:r['pair_id']) if r['variant_label']=='N']
    _require(parent_order==origins,'New pair suffix would change origin/donor ordering')
    inherited={origin:_read(parent_batch/'origins'/origin/'rejected.json') for origin in rejected}
    implementation={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    implementation['scripts/derive_edit_order_diagnostic.py']=Path(__file__).read_text()
    combined={'protocol':PROTOCOL,'parent_batch':str(parent_batch),'diagnostic_only':True,'actual_api_calls':0,'orders':{}}
    output.parent.mkdir(parents=True,exist_ok=True)
    staged=Path(tempfile.mkdtemp(prefix='.edit-order-',dir=output.parent))
    try:
        for order in ORDERS:
            directory=staged/order;directory.mkdir()
            under=Counter(v for r in results[order] for v in r['density_report']['violations'])
            common={'protocol':PROTOCOL,'order':order,'diagnostic_only':True,'status_meaning':STATUS_MEANING,
                    'parent_batch':str(parent_batch),'source_planned':summary['planned'],'source_accepted':summary['accepted'],
                    'source_rejected':summary['rejected'],'actual_api_calls':0,'production_accepted':0,
                    'density_violation_origins':[r['origin_id'] for r in results[order] if r['density_report']['violations']],
                    'density_violation_counts':dict(under)}
            rejected_records=[]
            for origin,old in inherited.items():
                record=deepcopy(old);record.update(protocol=PROTOCOL,order=order,actual_api_calls=0,
                    inherited_parent_rejection=True,reason='inherited_parent_rejection: '+str(old.get('reason')),
                    historical_parent_reason=old.get('reason'),historical_parent_actual_api_calls=old.get('actual_api_calls',0))
                rejected_records.append(record);dest=directory/'origins'/origin
                _write(dest/'rejected.json',record);_write(dest/'parent_rejected_snapshot.json',old)
                _write(dest/'input.json',{'protocol':PROTOCOL,'order':order,'parent_input':_read(parent_batch/'origins'/origin/'input.json')})
            current_manifest={**common,'selected':list(selected),'split':deepcopy(manifest.get('split')),
                'parent_manifest':manifest,'parent_origin_order':parent_order,'e_donor_origin_preserved_by_order':True,
                'selection_note':'All v12 accepted parents and original rejections retained; no new selection or target outputs',
                'order_policy':'Only permutation of the same three no-local blocks and their step-number digits differ',
                'e_donor_policy':'Same full-dataset origin order and deterministic donor selection; donor trace uses this order'}
            brief=[{k:r[k] for k in ('origin_id','status','acceptance_scope','error_counts','density_report','review_status')} for r in results[order]]
            brief.extend({k:r[k] for k in ('origin_id','status','reason','inherited_parent_rejection')} for r in rejected_records)
            current_summary={**common,'planned':summary['planned'],'processed':summary['planned'],
                'accepted':len(results[order]),'rejected':len(rejected_records),'accepted_for_paired_diagnostic_only':len(results[order]),
                'inherited_parent_rejections':len(rejected_records),'results':sorted(brief,key=lambda r:r['origin_id'])}
            _write(directory/'manifest.json',current_manifest);_write(directory/'summary.json',current_summary)
            _write(directory/'implementation_snapshot.json',implementation)
            for name in ('tokenizer_snapshots.json','implementation_snapshot.json','manifest.json','summary.json'):
                dest=name if name=='tokenizer_snapshots.json' else 'parent_'+name
                shutil.copyfile(parent_batch/name,directory/dest)
            with (directory/'pairs.jsonl').open('w') as handle:
                for record in results[order]:
                    for pair in record['pairs']:handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
            for record in results[order]:
                origin=record['origin_id'];dest=directory/'origins'/origin
                _write(dest/'accepted.json',record);_write(dest/'parent_record_snapshot.json',sources[origin]['accepted'])
                _write(dest/'parent_input_snapshot.json',sources[origin]['input'])
                inp=deepcopy(sources[origin]['input']);inp.update(protocol=PROTOCOL,order=order,diagnostic_only=True)
                _write(dest/'input.json',inp);_write(dest/'reference.json',sources[origin]['reference'])
                _write(dest/'fixed_plan.json',record['plan'])
            combined['orders'][order]=current_summary
        _write(staged/'derivation_summary.json',combined)
        staged.rename(output)
    except BaseException:
        shutil.rmtree(staged,ignore_errors=True);raise
    return combined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh root containing account_last and edit_last')
    args=parser.parse_args();result=materialize(args.parent,args.output)
    print(json.dumps({order:{k:value[k] for k in ('planned','processed','accepted','rejected','density_violation_origins')} for order,value in result['orders'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
