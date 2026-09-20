#!/usr/bin/env python3
"""Generate isolated, resumable agent corruption datasets from original references."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from molhallulens.modules.agent_generation.orchestrator import load_origins,make_split,write_json,generate_one,PROTOCOL


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['development','heldout','all'],default='development')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20260920)
    parser.add_argument('--candidate-mode',choices=['standard','severe'],default='severe')
    parser.add_argument('--min-roots',type=int,default=1)
    parser.add_argument('--min-nodes',type=int,default=6)
    parser.add_argument('--min-tokens',type=int,default=40)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    rows=load_origins(ROOT);split=make_split(rows,seed=args.seed)
    selected={r['origin_id'] for r in rows} if args.split=='all' else set(split[args.split])
    # Round-robin tasks, so --limit 3 includes one of each task.
    ordered=[]
    groups=[[r for r in rows if r['origin_id'] in selected and r['subtask']==s] for s in ['add','delete','substitute']]
    for i in range(max(map(len,groups))):
        ordered.extend(g[i] for g in groups if i<len(g))
    if args.limit:ordered=ordered[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'protocol':PROTOCOL,'split':split,'selected':[r['origin_id'] for r in ordered],
              'options':{'split':args.split,'seed':args.seed,'min_roots':args.min_roots,'min_nodes':args.min_nodes,'min_tokens':args.min_tokens,'candidate_mode':args.candidate_mode},
              'quality_gate':'Program-verified semantic construction; model reviews are retained non-gating diagnostics',
              'selection_note':'All planned origins retained; product-changing structural diagnostic subset; no target-model outcome filtering'}
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Manifest changed; use fresh output')
    write_json(path,manifest)
    if args.offline:
        print(json.dumps(manifest,indent=2));return
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    from transformers import AutoTokenizer
    modelroot=ROOT.parent/'chemical_models'
    tokenizers={name:AutoTokenizer.from_pretrained(str(modelroot/name),local_files_only=True,use_fast=True) for name in ['ChemDFM-R-14B','Chem-R-8B']}
    snapshots={name:{'backend':tok.backend_tokenizer.to_str(),'chat_template':tok.chat_template,
                         'special_tokens_map':tok.special_tokens_map,'class':type(tok).__name__}
               for name,tok in tokenizers.items()}
    snap_path=args.output/'tokenizer_snapshots.json'
    if snap_path.exists() and json.loads(snap_path.read_text())!=snapshots:
        raise ValueError('Tokenizer vocabulary/template/config changed; use a new output directory')
    write_json(snap_path,snapshots)
    source_files={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    source_files['scripts/generate_agent_pairs.py']=Path(__file__).read_text()
    implementation=args.output/'implementation_snapshot.json'
    if implementation.exists() and json.loads(implementation.read_text())!=source_files:
        raise ValueError('Implementation changed; use a new output directory for reproducibility')
    write_json(implementation,source_files)
    results=[]
    def work(item):
        idx,row=item
        return generate_one(row,args.output/'origins'/row['origin_id'],seed=args.seed+idx,
            min_roots=args.min_roots,min_nodes=args.min_nodes,min_tokens=args.min_tokens,candidate_mode=args.candidate_mode,tokenizers=tokenizers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,item) for item in enumerate(ordered)]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            print(json.dumps({k:r[k] for k in ['origin_id','status','reason','error_counts'] if k in r},ensure_ascii=False),flush=True)
            successful=sorted((r for r in results if r['status']=='accepted'),key=lambda x:x['origin_id'])
            with (args.output/'pairs.jsonl.tmp').open('w') as f:
                for r in successful:
                    for pair in r['pairs']:f.write(json.dumps(pair,ensure_ascii=False)+'\n')
            (args.output/'pairs.jsonl.tmp').replace(args.output/'pairs.jsonl')
            write_json(args.output/'summary.json',{'planned':len(ordered),'processed':len(results),'accepted':len(successful),'rejected':len(results)-len(successful),
                'results':[{k:r[k] for k in ['origin_id','status','reason','error_counts','review_status'] if k in r} for r in results]})
    print('GENERATION COMPLETE',flush=True)


if __name__=='__main__':main()
