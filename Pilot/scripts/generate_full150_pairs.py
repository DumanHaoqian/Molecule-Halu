#!/usr/bin/env python3
"""Resumable full150 generation; immutable question, fixed construction rules."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from molhallulens.modules.agent_generation.orchestrator import load_origins, write_json
from molhallulens.modules.agent_generation.full_pipeline import PROTOCOL, prepare_row, review_row, CLEAN_PROMPT, SELECT_PROMPT, AUDIT_PROMPT
TOKENS = None


def tokenizers():
    global TOKENS
    if TOKENS is None:
        from transformers import AutoTokenizer
        TOKENS = {name:AutoTokenizer.from_pretrained(str(ROOT.parent/'chemical_models'/name),local_files_only=True,use_fast=True) for name in ('ChemDFM-R-14B','Chem-R-8B')}
    return TOKENS


def prepare_job(row, directory):
    try: return prepare_row(row,directory,tokenizers())
    except Exception as exc:
        state={'origin_id':row['origin_id'],'status':'rejected','reason':str(exc),'error_type':type(exc).__name__}
        write_json(Path(directory)/'prepared.json',state)
        return state


def review_job(directory):
    return review_row(directory,tokenizers())


def aggregate(base, selected):
    states, pairs = [], []
    for origin in selected:
        directory=base/'origins'/origin
        path=next((directory/p for p in ('accepted.json','rejected.json','prepared.json') if (directory/p).exists()),None)
        if path:
            state=json.loads(path.read_text());states.append(state)
            if state['status']=='accepted': pairs.extend(state['pairs'])
    temp=base/'pairs.jsonl.tmp';temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs));temp.replace(base/'pairs.jsonl')
    summary={'protocol':PROTOCOL,'planned':len(selected),'processed':len(states),'accepted':sum(s['status']=='accepted' for s in states),'rejected':sum(s['status']=='rejected' for s in states),'prepared':sum(s['status']=='prepared' for s in states),'updated_at':time.time()}
    write_json(base/'summary.json',summary)
    return summary


def freeze(base,rows):
    manifest={'protocol':PROTOCOL,'selected':[r['origin_id'] for r in rows],'original_rows':rows,'pool_size':12,'min_nodes':6,'min_tokens':40,'model':'gpt-5.4-mini','seed':20260920,'scope':'Full corpus exploratory; includes development cases. Original N versus canonical H is a style confound. No target outputs used for selection.'}
    if (base/'manifest.json').exists():
        if json.loads((base/'manifest.json').read_text())!=manifest: raise ValueError('Frozen run manifest differs')
    else:
        write_json(base/'manifest.json',manifest)
    sources=[Path(__file__),*sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))]
    for source in sources:
        dest=base/'implementation'/source.relative_to(ROOT)
        if dest.exists():
            if dest.read_bytes()!=source.read_bytes(): raise ValueError('Frozen implementation differs: '+str(source))
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    write_json(base/'prompts.json',{'clean':CLEAN_PROMPT,'selector':SELECT_PROMPT,'audit':AUDIT_PROMPT})
    for name,tok in tokenizers().items():
        dest=base/'tokenizers'/name
        if not dest.exists():
            tok.save_pretrained(str(dest))
        else:
            with tempfile.TemporaryDirectory() as temporary:
                tok.save_pretrained(temporary)
                current={p.name:p.read_bytes() for p in Path(temporary).iterdir() if p.is_file()}
                frozen={p.name:p.read_bytes() for p in dest.iterdir() if p.is_file()}
                if current != frozen: raise ValueError('Frozen tokenizer differs: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'Experiments/agent_corruption_v18/full150')
    p.add_argument('--phase',choices=['prepare','review','all'],default='all')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int,default=0,help='Process first N selected origins, manifest remains all150')
    args=p.parse_args();os.environ['TOKENIZERS_PARALLELISM']='false'
    rows=load_origins(ROOT);freeze(args.output,rows)
    selected=[r['origin_id'] for r in rows]
    jobs=rows[:args.limit] if args.limit else rows
    print(f'PID={os.getpid()} phase={args.phase} origins={len(jobs)}/150',flush=True)
    for phase in (['prepare','review'] if args.phase=='all' else [args.phase]):
        pool=ProcessPoolExecutor if phase=='prepare' else ThreadPoolExecutor
        with pool(max_workers=args.workers) as executor:
            futures={executor.submit(prepare_job,row,args.output/'origins'/row['origin_id']) if phase=='prepare' else executor.submit(review_job,args.output/'origins'/row['origin_id']):row['origin_id'] for row in jobs if phase=='prepare' or (args.output/'origins'/row['origin_id']/'candidate_pool.json').exists()}
            for future in as_completed(futures):
                try:
                    state=future.result()
                    print(phase,futures[future],state['status'],state.get('eligible_pool_count',''),state.get('reason',''),flush=True)
                except Exception as exc:
                    print(phase,futures[future],'ERROR',type(exc).__name__,str(exc),flush=True)
                aggregate(args.output,selected)
    summary=aggregate(args.output,selected);print(json.dumps(summary),flush=True)
    if args.phase in ('review','all') and not args.limit and summary['accepted']!=150: raise SystemExit(1)

if __name__=='__main__':main()
