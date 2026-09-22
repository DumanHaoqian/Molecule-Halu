"""Prepare independent, auditable chemistry SAE text corpora. CPU only."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil

from corpus import (LocalTokenizer,blocked_reasons,digest,group_records,iter_rows,
                    molecule_keys,norm,normalize,render_and_count)

HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=Path('/home/haoqian/Data/past/Molecule')
DEFAULT_MODEL=HERE.parent/'chemical_models/ChemDFM-R-14B'

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def line(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    return h.hexdigest()

def pilot_block(pilot):
    block={'questions':set(),'cids':set(),'molecules':set()};fingerprints=[];count=0
    for path in sorted((pilot/'data/tasks').glob('*/questions.jsonl')):
        fingerprints.append({'path':str(path),'sha256':file_sha(path)})
        for q in iter_rows(path):
            count+=1;block['questions'].add(norm(q['input']))
            task=path.parent.name
            if task in ('cap2mol','mol2cap'):block['cids'].add(str(q['question_id']))
            if task=='mol2cap' and q.get('reference'):block['questions'].add(norm(q['reference']))
            strings=[q.get(k) for k in ('input','reference','source_gt')]
            strings.extend(v for k,v in q.get('metadata',{}).items() if k in ('source_molecule','molecule','smiles','reactants','products'))
            for s in strings:
                if isinstance(s,str):block['molecules'].update(molecule_keys(s))
    prepared=pilot/'experiments/post_token_layer26/prepared/records.jsonl'
    if prepared.exists():
        fingerprints.append({'path':str(prepared),'sha256':file_sha(prepared)})
        for r in iter_rows(prepared):
            block['questions'].add(norm(r['input_text']))
            block['questions'].add(norm(r['response_text']))
            if r.get('molecule_key'):block['molecules'].update(molecule_keys(r['molecule_key']))
    if count!=600:raise ValueError(f'Expected 600 Pilot questions, found {count}')
    return block,{'question_count':count,'fingerprints':fingerprints,**{k:sorted(v) for k,v in block.items()}}

def jobs(root):
    result=[]
    for p in sorted((root/'Downloads/Train/Chemcot').glob('*/*.json')):
        result.append({'source':'chemcot','task':p.parent.name+'/'+p.stem,'path':str(p.relative_to(root))})
    base=Path('Molecule/Baselines/Mol-LLaMA/data')
    for name in ('detailed_structural_descriptions','structure2chemical_features_relationships','structure2biological_features_relationships','comprehensive_conversations'):
        result.append({'source':'mollama','task':name,'path':str(base/'Mol-LLaMA-Instruct'/f'{name}.json')})
    result.append({'source':'pubchem','task':'molecule_description','path':str(base/'Mol-LLaMA-Instruct/pubchem-molecules.json')})
    result.append({'source':'chebi','task':'molecule_description','path':str(base/'ChEBI-20/train.txt')})
    result.append({'source':'moleculeqa','task':'molecule_qa','path':str(base/'moleculeqa/train.json')})
    for partition in range(4):result.append({'source':'openmolins','task':'from_row','path':'Downloads/Train/OpenMolIns/xlarge/train.csv','partition':partition,'partitions':4})
    for i,j in enumerate(result):j['job_id']=i
    return result

def init_worker(model,block):
    global TOKENIZER,BLOCK
    os.environ['TOKENIZERS_PARALLELISM']='false'
    TOKENIZER=LocalTokenizer(model);BLOCK=block

def process_job(job,root,staging,limit,max_length):
    source=job['source'];pool=source in ('openmolins','moleculeqa');counts=Counter()
    accepted=Path(staging)/f'{job["job_id"]:03d}.accepted.jsonl'
    excluded=Path(staging)/f'{job["job_id"]:03d}.excluded.jsonl'
    with accepted.open('w') as out,excluded.open('w') as reject:
        for index,raw in enumerate(iter_rows(Path(root)/job['path'])):
            if index%job.get('partitions',1)!=job.get('partition',0):continue
            if limit and counts['input']>=limit:break
            counts['input']+=1
            row=normalize(source,job['task'],raw,job['path'],index)
            reasons=blocked_reasons(row,BLOCK)+row['quality_flags']
            if source=='pubchem' and raw.get('split') not in ('pretrain','train','finetune'):
                reasons.append('source_split_not_train')
            if not reasons and not pool:
                row['tokenization']=render_and_count(row['messages'],TOKENIZER)
                if row['tokenization']['sequence_tokens']>max_length:reasons.append('over_context_limit')
                if row['tokenization']['assistant_tokens']<16:reasons.append('fewer_than_16_assistant_tokens')
            if reasons:
                counts['excluded']+=1
                for reason in set(reasons):counts['reason:'+reason]+=1
                reject.write(line({'id':row['id'],'source':source,'task':row['task'],'source_ref':row['source_ref'],'reasons':sorted(set(reasons))}))
            else:
                counts['accepted']+=1;out.write(line(row))
            if counts['input']%25000==0:print(line({'job':job['job_id'],'source':source,'rows':counts['input']}),end='',flush=True)
    summary={**job,'counts':dict(counts),'accepted_path':str(accepted),'excluded_path':str(excluded)}
    dump(Path(staging)/f'{job["job_id"]:03d}.summary.json',summary)
    print(line({'job_done':job['job_id'],'source':source,'counts':dict(counts)}),end='',flush=True)
    return summary

class Shards:
    def __init__(self,out,max_rows=10000):self.out=Path(out);self.max_rows=max_rows;self.handles={};self.entries={}
    def write(self,category,source,row):
        key=(category,source);entry=self.entries.get(key)
        if entry is None or entry['rows']>=self.max_rows:
            if key in self.handles:self.handles.pop(key).close()
            seq=0 if entry is None else entry['part']+1
            relative=Path(category)/f'{source}-{seq:05d}.jsonl';path=self.out/relative;path.parent.mkdir(parents=True,exist_ok=True)
            entry={'path':str(relative),'part':seq,'rows':0};self.entries[key]=entry;self.handles[key]=path.open('w')
        location={'path':entry['path'],'line_index':entry['rows']};self.handles[key].write(line(row));entry['rows']+=1
        return location
    def close(self):
        for f in self.handles.values():f.close()

def selections(out,eligible,seed,train_budget=10000000,val_budget=500000):
    summaries={}
    for split,budget in [('train',train_budget),('validation',val_budget)]:
        chosen=[];per_source={}
        for source,weight in [('chemcot',.3),('mollama',.7)]:
            candidates=[r for r in eligible if r['split']==split and r['source']==source]
            candidates.sort(key=lambda r:digest([seed,'pilot_10m',r['id']]))
            target=round(budget*weight);actual=0;number=0
            for row in candidates:
                if actual>=target:break
                chosen.append(row);actual+=row['assistant_tokens'];number+=1
            per_source[source]={'target_tokens':target,'actual_tokens':actual,'records':number,'shortfall':max(0,target-actual)}
        chosen.sort(key=lambda r:digest([seed,'mixed_order',r['id']]))
        path=out/'selections/pilot_10m'/f'{split}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            for row in chosen:f.write(line(row))
        summaries[split]={'records':len(chosen),'assistant_tokens':sum(r['assistant_tokens'] for r in chosen),'sources':per_source,'path':str(path.relative_to(out))}
    dump(out/'selections/pilot_10m/summary.json',summaries)
    return summaries

def build(args):
    out=Path(args.out).resolve();root=Path(args.source_root).resolve();model=Path(args.model).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}; choose a new version directory')
    out.mkdir(parents=True);staging=out/'_build';staging.mkdir()
    block,receipt=pilot_block(Path(args.pilot).resolve());dump(out/'audit/pilot_exclusion.json',receipt)
    task_jobs=jobs(root)
    config={**vars(args),'source_root':str(root),'model':str(model),'out':str(out),'schema_version':'sae-corpus-v1','created_utc':datetime.now(timezone.utc).isoformat(),'layer_number':26,'block_index':25,'timing':'post','hidden_size':5120,'min_assistant_tokens':16,'split_policy':'98/2 by connected molecular/CID/prompt/long-answer groups; seeded hash','pool_policy':'candidate only; no activations; rerun grouping before admitting generated answers'}
    dump(out/'config.json',config)
    # Large independent source files are processed with bounded CPU workers; no CUDA imports.
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(model),block)) as executor:
        pending={executor.submit(process_job,j,str(root),str(staging),args.limit_per_file,args.max_length):j for j in task_jobs}
        for future in as_completed(pending):summaries.append(future.result())
    finalize(args,out,root,model,summaries,receipt,config,task_jobs)

def finalize(args,out,root,model,summaries,receipt,config,task_jobs):
    summaries.sort(key=lambda r:r['job_id'])
    print('Normalizing complete; deduplicating and building connected groups.',flush=True)
    corpus_meta=[];seen={};accepted_ids=set();pool_ids=set();dedup_counts=Counter()
    with (out/'audit/excluded.jsonl').open('w') as reject:
        for job in summaries:
            with open(job['excluded_path']) as f:shutil.copyfileobj(f,reject)
        for job in summaries:
            pool=job['source'] in ('openmolins','moleculeqa')
            for row in iter_rows(job['accepted_path']):
                key=('pool' if pool else 'corpus',row['content_sha256'])
                if key in seen:
                    dedup_counts[row['source']]+=1
                    reject.write(line({'id':row['id'],'source':row['source'],'source_ref':row['source_ref'],'reasons':['duplicate_prompt' if pool else 'duplicate_dialog'],'duplicate_of':seen[key]}));continue
                seen[key]=row['id']
                if pool:pool_ids.add(row['id'])
                else:
                    accepted_ids.add(row['id']);corpus_meta.append({'id':row['id'],'group_keys':row['group_keys']})
    assignment=group_records(corpus_meta,args.seed)
    group_sizes=Counter(g for g,s in assignment.values());del corpus_meta,seen
    writers=Shards(out);counts=defaultdict(Counter);eligible=[];groups_by_split=defaultdict(set)
    for job in summaries:
        pool=job['source'] in ('openmolins','moleculeqa')
        for row in iter_rows(job['accepted_path']):
            if row['id'] not in (pool_ids if pool else accepted_ids):continue
            if pool:
                row['split']='candidate';row['tier']='prompt_pool';category='prompt_pool'
            else:
                row['group_id'],row['split']=assignment[row['id']]
                row['tier']='main' if row['source'] in ('chemcot','mollama') else 'supplement'
                category=row['tier']+'/'+row['split'];groups_by_split[row['split']].add(row['group_id'])
            location=writers.write(category,row['source'],row)
            stat=counts[(row['tier'],row['split'],row['source'])];stat['records']+=1
            if not pool:
                stat['assistant_tokens']+=row['tokenization']['assistant_tokens'];stat['sequence_tokens']+=row['tokenization']['sequence_tokens']
                if row['tier']=='main':eligible.append({'id':row['id'],'group_id':row['group_id'],'source':row['source'],'task':row['task'],'split':row['split'],**location,'assistant_tokens':row['tokenization']['assistant_tokens'],'sequence_tokens':row['tokenization']['sequence_tokens']})
    writers.close()
    selection=selections(out,eligible,args.seed,10000 if args.limit_per_file else 10000000,1000 if args.limit_per_file else 500000)
    stats={'tables':[{'tier':k[0],'split':k[1],'source':k[2],**v} for k,v in sorted(counts.items())],'job_counts':[{k:v for k,v in s.items() if k not in ('accepted_path','excluded_path')} for s in summaries],
           'deduplicated':dict(dedup_counts),'groups':{'total':len(group_sizes),'largest':group_sizes.most_common(10),'train':len(groups_by_split['train']),'validation':len(groups_by_split['validation'])},'selection':selection}
    dump(out/'stats.json',stats)
    fingerprints=[]
    for path in sorted({j['path'] for j in task_jobs}):
        p=root/path;fingerprints.append({'path':path,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':file_sha(p)})
    outputs=[]
    for top in ('main','supplement','prompt_pool','selections'):
        for path in sorted((out/top).rglob('*.jsonl')):
            outputs.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    manifest={'config':config,'sources':fingerprints,'model_files':{name:file_sha(model/name) for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json')},'code_files':{p.name:file_sha(p) for p in HERE.glob('*.py')},'versions':{name:importlib.metadata.version(name) for name in ('rdkit','tokenizers','jinja2')},'outputs':outputs,'pilot_question_count':receipt['question_count'],'status':'built_pending_independent_verification'}
    dump(out/'manifest.json',manifest)
    print(line({'build_complete':str(out),'tables':stats['tables'],'selection':selection}),flush=True)

def refresh_chemcot(args):
    """Repair an unverified build after source-specific review; regroup all corpora."""
    out=Path(args.out).resolve()
    if (out/'COMPLETE.json').exists():raise ValueError('Cannot modify a verified release')
    manifest=json.loads((out/'manifest.json').read_text());config=manifest['config']
    if config['out']!=str(out):raise ValueError('Wrong output directory')
    root=Path(config['source_root']);model=Path(config['model']);staging=out/'_build'
    args.seed=config['seed'];args.max_length=config['max_length'];args.limit_per_file=config['limit_per_file']
    block,receipt=pilot_block(Path(config['pilot']).resolve())
    dump(out/'audit/pilot_exclusion.json',receipt)
    task_jobs=jobs(root);summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker,initargs=(str(model),block)) as executor:
        pending=[executor.submit(process_job,j,str(root),str(staging),args.limit_per_file,args.max_length) for j in task_jobs if j['source']=='chemcot']
        for f in as_completed(pending):summaries.append(f.result())
    from corpus import ANSWER_CUE
    for j in task_jobs:
        if j['source']=='chemcot':continue
        summary=json.loads((staging/f'{j["job_id"]:03d}.summary.json').read_text())
        accepted=Path(summary['accepted_path']);temporary=accepted.with_suffix('.refined.tmp')
        with temporary.open('w') as target,open(summary['excluded_path'],'a') as reject:
            for row in iter_rows(accepted):
                if ANSWER_CUE.search('\n'.join(m['content'] for m in row['messages'])):
                    summary['counts']['accepted']-=1;summary['counts']['excluded']=summary['counts'].get('excluded',0)+1
                    summary['counts']['reason:answer_cue']=summary['counts'].get('reason:answer_cue',0)+1
                    reject.write(line({'id':row['id'],'source':row['source'],'source_ref':row['source_ref'],'reasons':['answer_cue']}))
                else:target.write(line(row))
        temporary.replace(accepted);dump(staging/f'{j["job_id"]:03d}.summary.json',summary);summaries.append(summary)
    # These directories contain only this unverified builder's generated outputs.
    dump(staging/'pre_review_manifest.json',manifest)
    for directory in ('main','supplement','prompt_pool','selections'):
        if (out/directory).exists():shutil.rmtree(out/directory)
    for filename in ('verification.json','README.md','task_counts.json'):
        (out/filename).unlink(missing_ok=True)
    config['review_refinement']='ChemCoT re-normalized for trailing source punctuation and explicit secret-target cues; all other accepted messages re-screened; all corpus groups/splits/selections regenerated.'
    dump(out/'config.json',config)
    finalize(args,out,root,model,summaries,receipt,config,task_jobs)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',default=str(DEFAULT_ROOT));parser.add_argument('--model',default=str(DEFAULT_MODEL));parser.add_argument('--pilot',default=str(HERE.parent/'Pilot_v2'));parser.add_argument('--out',default=str(HERE/'data/v1'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--seed',type=int,default=20260921);parser.add_argument('--max-length',type=int,default=16384);parser.add_argument('--limit-per-file',type=int,default=0)
    parser.add_argument('--refresh-chemcot',action='store_true')
    args=parser.parse_args()
    if args.refresh_chemcot:refresh_chemcot(args)
    else:build(args)
