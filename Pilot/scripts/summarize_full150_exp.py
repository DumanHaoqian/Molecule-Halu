#!/usr/bin/env python3
"""Strict full150 readiness checks and fixed-denominator ABC readout."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from molhallulens.modules.agent_generation.metrics import paired_comparison

PROTOCOL = 'agent_full_task_interpretation_v1'
MODELS = {'ChemDFM-R-14B': 'chemdfm_r14b', 'Chem-R-8B': 'chem_r8b'}

def read(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_generation(base, raw_dir):
    raw_rows = [r for task in ('add', 'delete', 'substitute') for r in read(raw_dir / f'{task}_pilot_origin.json')]
    raw = {r['anonymous_sample_id']: r for r in raw_rows}
    require(len(raw_rows) == len(raw) == 150, 'Raw benchmark must contain exactly 150 unique origins')
    manifest, summary = read(base/'manifest.json'), read(base/'summary.json')
    require(manifest.get('protocol') == PROTOCOL, 'Unexpected full150 protocol')
    selected = manifest.get('selected', [])
    require(len(selected) == 150 and set(selected) == set(raw), 'Selected origins must equal all raw150')
    require(all(summary.get(k) == n for k,n in [('planned',150),('processed',150),('accepted',150),('rejected',0)]), 'Generation is not ready: require planned/processed/accepted=150 and rejected=0')
    pairs = rows(base/'pairs.jsonl')
    indexed = {(p['origin_id'],p['variant_label']):p for p in pairs}
    require(len(pairs) == len(indexed) == 300 and set(indexed) == {(o,v) for o in raw for v in 'NH'}, 'Require exactly 300 unique N/H rows across raw150')
    for origin, gt in raw.items():
        n,h = indexed[origin,'N'], indexed[origin,'H']
        pair_id = origin+'__'+PROTOCOL
        require(n['pair_id'] == h['pair_id'] == pair_id, 'Mixed or stale pair protocol')
        for p in (n,h):
            for field, original in [('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')]:
                require(p['detector_input'][field] == gt[original], f'Immutable benchmark field changed: {origin}/{field}')
        require(n['detector_input']['reasoning_chain'] != h['detector_input']['reasoning_chain'], 'N/H must differ')
        accepted = read(base/'origins'/origin/'accepted.json')
        require(accepted.get('protocol', accepted.get('plan',{}).get('protocol')) == PROTOCOL, 'Accepted sidecar has mixed protocol')
        require(accepted.get('origin_id') == origin and accepted.get('pair_id') == pair_id, 'Accepted sidecar identity differs')
        require({p['variant_label']:p for p in accepted['pairs']} == {'N':n,'H':h}, 'Accepted sidecar views differ')
        require(isinstance(accepted.get('annotations'),list) and accepted['annotations'], 'Missing audited H annotations')
    return raw

def validate_prepared(base, folder, raw):
    directory = base/'evaluation'/folder
    manifest = read(directory/'manifest.json')
    require(manifest['agent_evaluation']['groups'] == 'ABC' and manifest['agent_evaluation']['placement'] == 'prefix', 'Only ABC prefix is authorized')
    require(manifest['n_origins'] == manifest['n_pairs'] == 150 and manifest['n_requests'] == 450, 'Prepared coverage differs from full150')
    requests = rows(directory/'requests.jsonl')
    require(len(requests) == 450 and len({r['request_id'] for r in requests}) == 450 and {(r['origin_id'],r['group']) for r in requests} == {(o,g) for o in raw for g in 'ABC'}, 'Requests must cover ABC exactly once for every raw origin')
    return manifest

def render(base, output, raw):
    summary = read(base/'summary.json')
    lines = ['# Full150 exploratory task-interpretation experiment', '',
             'This full150 exploratory analysis includes the previous 18 development origins; it is not a held-out evaluation.',
             f'Generation protocol: `{PROTOCOL}`. Planned/processed/accepted/rejected: {summary["planned"]}/{summary["processed"]}/{summary["accepted"]}/{summary["rejected"]}.',
             'Style confound: C retains original N after explicit-answer projection; B uses a deterministic canonical rewrite. Differences cannot isolate corruption from writing style.',
             'A = empty reasoning; B = H; C = N. Original system prompts and assistant-prefix placement are unchanged. Only input reasoning traces differ.',
             'Accuracy is atom-map-normalized main-fragment equality; FTS is full-molecule Morgan fingerprint Tanimoto. Invalid, missing and unfinished outputs contribute zero to both fixed-denominator metrics. Partial results are provisional.', '']
    validation_path=base/'release_validation.json'
    if validation_path.exists():
        validation=read(validation_path)
        checked=validation.get('rows',[])
        lines += [f"Program reconstruction audit: {validation.get('status')}, {len(checked)}/150 origins. Agent review concerns: clean reference {sum(r.get('agent_clean_status')!='pass' for r in checked)}/{len(checked)}, conditional fidelity {sum(r.get('agent_conditional_status')!='pass' for r in checked)}/{len(checked)}. These diagnostic concerns are retained; this dataset is not represented as independently agent-approved.", '']
    for name, folder in MODELS.items():
        directory = base/'evaluation'/folder
        outcomes = rows(directory/'outcome_records.jsonl')
        keys = [(r['origin_id'],r['group']) for r in outcomes]
        require(len(set(keys)) == len(keys) and set(keys) <= {(o,g) for o in raw for g in 'ABC'}, 'Duplicate or out-of-scope outcomes')
        lines += [f'## {name}', '', f'Scored outcomes: {len(outcomes)}/450; '+('complete.' if len(outcomes)==450 else 'incomplete; do not interpret as final.'), '', '| Group | n | Accuracy | FTS | Invalid/missing answer | Not yet scored | Truncated |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for group in 'ABC':
            current = [r for r in outcomes if r['group']==group]
            valid = [r for r in current if r.get('valid_smiles') is True]
            require(all(r.get('primary_match') in (0,1) and type(r.get('fts')) in (int,float) and math.isfinite(r['fts']) and 0<=r['fts']<=1 for r in valid), 'Invalid outcome metric')
            lines.append(f'| {group} | 150 | {sum(r["primary_match"] for r in valid)/150:.2%} | {sum(r["fts"] for r in valid)/150:.4f} | {len(current)-len(valid)} | {150-len(current)} | {sum(r.get("finish_reason")=="length" for r in current)} |')
        if len(outcomes)==450:
            by_key={(r['origin_id'],r['group']):r for r in outcomes}
            lines += ['', '| Paired contrast | Accuracy difference | Right → wrong | Wrong → right | Exact McNemar p |', '| --- | ---: | ---: | ---: | ---: |']
            for baseline in 'AC':
                comparison=paired_comparison([bool(by_key[o,baseline].get('valid_smiles') and by_key[o,baseline]['primary_match']) for o in sorted(raw)], [bool(by_key[o,'B'].get('valid_smiles') and by_key[o,'B']['primary_match']) for o in sorted(raw)],baseline_label=baseline,treatment_label='B')
                c=comparison['contingency']
                lines.append(f"| B − {baseline} | {comparison['accuracy_difference']*100:+.2f} pp | {c['right_to_wrong']} | {c['wrong_to_right']} | {comparison['mcnemar_exact_p']:.6g} |")
        mechanism_path=base/'review/plan_following.json'
        if mechanism_path.exists():
            mechanism=read(mechanism_path)['models'][name]['groups']['B']['matches']['executed_h']
            lines += ['', f"B outputs matching the executed H plan's full product: {mechanism['full_count']}/150. This is a descriptive agreement check, not a causal mediation test.", 'Paired McNemar p values above are exploratory and unadjusted for multiple comparisons.']
        lines += ['', f'Preserved requests, rendered prefixes, predictions and scored outcomes: `{directory}`.', '']
    lines += [f'Generation audit: `{base}/origins/<origin>/accepted.json` (plans, annotations and exact N/H views). Accepted denotes program construction checks, not new independent model review.']
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(output)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-dir',type=Path,default=ROOT/'Experiments/agent_corruption_v18/full150')
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'Dataset/raw_benchmark_data/mol_edit')
    parser.add_argument('--output',type=Path,default=ROOT/'scripts/run_full150_exp_result.md')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--prepared-folder',choices=list(MODELS.values()))
    args=parser.parse_args()
    raw=validate_generation(args.generation_dir,args.raw_dir)
    if args.prepared_folder:
        validate_prepared(args.generation_dir,args.prepared_folder,raw)
    if not args.check_only:
        render(args.generation_dir,args.output,raw)
    print('Full150 checks passed',flush=True)

if __name__ == '__main__':
    main()
