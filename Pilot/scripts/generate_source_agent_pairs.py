#!/usr/bin/env python3
"""Resumable source-state CoT diagnostic; only the supplied trace changes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from molhallulens.modules.agent_generation.orchestrator import load_origins, make_split, write_json
from molhallulens.modules.agent_generation.source_state_orchestrator import PROTOCOL, MODEL_NAMES, generate_one

_TOKENIZERS = None


def tokenizers():
    global _TOKENIZERS
    if _TOKENIZERS is None:
        from rdkit import RDLogger
        from transformers import AutoTokenizer
        RDLogger.DisableLog('rdApp.error')
        _TOKENIZERS = {name: AutoTokenizer.from_pretrained(str(ROOT.parent/'chemical_models'/name),
                          local_files_only=True, use_fast=True) for name in MODEL_NAMES}
    return _TOKENIZERS


def work(row, index, output, options):
    return generate_one(row, Path(output)/'origins'/row['origin_id'], tokenizers=tokenizers(),
        seed=options['seed']+index, min_nodes=options['min_nodes'], min_tokens=options['min_tokens'])


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Frozen {path.name} changed; use a new output directory')
    write_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'heldout', 'all'), default='development')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--min-nodes', type=int, default=6)
    parser.add_argument('--min-tokens', type=int, default=40)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.min_nodes < 1 or args.min_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers, density thresholds and any limit must be positive')
    rows = load_origins(ROOT)
    split = make_split(rows, seed=args.seed)
    selected = {r['origin_id'] for r in rows} if args.split == 'all' else set(split[args.split])
    groups = [[r for r in rows if r['origin_id'] in selected and r['subtask'] == task]
              for task in ('add', 'delete', 'substitute')]
    ordered = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    if args.limit:
        ordered = ordered[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    options = {'split': args.split, 'seed': args.seed, 'min_roots': 1, 'min_nodes': args.min_nodes,
               'min_tokens': args.min_tokens, 'candidate_mode': 'source_state'}
    manifest = {'protocol': PROTOCOL, 'split': split, 'selected': [r['origin_id'] for r in ordered],
        'options': options, 'quality_gate': 'Program-verified staged source reconstruction; model review disputes retained.',
        'selection_note': 'All planned origins and failures retained. No target-output selection.',
        'interpretation': 'CoT source-state intervention; unchanged original question. Complete-source copying is a possible mechanism.'}
    freeze(args.output/'manifest.json', manifest)
    if args.offline:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    snapshots = {name: {'backend': tok.backend_tokenizer.to_str(), 'chat_template': tok.chat_template,
                       'special_tokens_map': tok.special_tokens_map, 'class': type(tok).__name__}
                 for name, tok in tokenizers().items()}
    freeze(args.output/'tokenizer_snapshots.json', snapshots)
    source_files = {str(p.relative_to(ROOT)): p.read_text()
                    for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    source_files['scripts/generate_source_agent_pairs.py'] = Path(__file__).read_text()
    freeze(args.output/'implementation_snapshot.json', source_files)
    freeze(args.output/'worker_config.json', {'executor': 'ProcessPoolExecutor', 'start_method': 'spawn',
        'workers': args.workers, 'tokenizers_parallelism': False})
    results = []
    # Worker processes isolate RDKit/Python CPU work. Credentials are inherited
    # through the environment and never serialized into requests or snapshots.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = [pool.submit(work, row, index, str(args.output), options) for index, row in enumerate(ordered)]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({k: result[k] for k in ('origin_id', 'status', 'reason', 'error_counts') if k in result},
                             ensure_ascii=False), flush=True)
            successful = sorted((r for r in results if r['status'] == 'accepted'), key=lambda r: r['origin_id'])
            temporary = args.output/'pairs.jsonl.tmp'
            with temporary.open('w') as handle:
                for record in successful:
                    for pair in record['pairs']:
                        handle.write(json.dumps(pair, ensure_ascii=False)+'\n')
            temporary.replace(args.output/'pairs.jsonl')
            write_json(args.output/'summary.json', {'planned': len(ordered), 'processed': len(results),
                'accepted': len(successful), 'rejected': len(results)-len(successful),
                'results': [{k: r[k] for k in ('origin_id', 'status', 'reason', 'error_counts', 'review_status') if k in r}
                            for r in results]})
    print('GENERATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
