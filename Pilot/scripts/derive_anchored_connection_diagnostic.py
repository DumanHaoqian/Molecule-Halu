#!/usr/bin/env python3
"""Publish fixed native versus appended open-boundary connection diagnostics.

Reads frozen generation artifacts only, never target outputs or heldout rows.
Both styles keep the same four origins and exactly the same wrong plans.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('named_diagnostic_common', ROOT/'scripts/derive_named_binding_diagnostic.py')
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
read, write, require = common.read, common.write, common.require
PROTOCOL = 'agent_anchored_connections_diagnostic_v1'
STYLES = ('native', 'native_with_connections')
THRESHOLDS = {'min_roots': 1, 'min_nodes': 6, 'min_tokens': 40}


def build_record(row, base, connections, rendered, style, tokenizers):
    from molhallulens.modules.agent_generation.labels import label_tokens
    from molhallulens.modules.agent_generation.quality import find_answer_leakage
    from molhallulens.modules.agent_generation.severity import measure_severity

    require(connections['checks']['status'] == rendered['checks']['status'] == 'pass', 'Connection or rendering checks failed')
    nr, hr = rendered['N_render'], rendered['H_render']
    require(rendered['roots'] == base['plan']['roots'] and len(rendered['roots']) == 1,
            'The frozen fragment root must remain the only root')
    require(rendered['roots'][0]['node_id'] == 'fragment', 'Unexpected frozen root')
    original = {p['variant_label']: p for p in base['pairs']}
    for label, view in [('N', nr), ('H', hr)]:
        if style == 'native':
            require(view['text'] == original[label]['detector_input']['reasoning_chain'], 'Native control text changed')
        else:
            require(view['text'].startswith(original[label]['detector_input']['reasoning_chain']),
                    'The augmented style may only append to its frozen native trace')
        require(not find_answer_leakage(view['text'], [row['gt_smiles'], base['plan']['execution']['product_smiles']]),
                'A full answer or forbidden cue appears in '+label)
        for field in ['spans', 'bindings']:
            for item in view[field]:
                require(0 <= item['start'] < item['end'] <= len(view['text'])
                        and view['text'][item['start']:item['end']] == item['text'],
                        'Semantic offset/text mismatch in '+label)
    require(not nr['spans'] and hr['spans'], 'N must have no injected errors; H must have annotations')
    require({r for s in hr['spans'] if s['kind'] == 'root_error' for r in s['root_ids']} == {'r1'},
            'Root annotation coverage changed')
    require(hr['spans'][:len(base['annotations'])] == base['annotations'], 'Frozen native annotations changed')
    labels = {name: label_tokens(hr['text'], hr['spans'], tokenizer) for name,tokenizer in tokenizers.items()}
    counts = {name: sum(any(label != 'unchanged' for label in item['labels']) for item in values)
              for name,values in labels.items()}
    nodes = {s['node_id'] for s in hr['spans']}
    violations = []
    if len(nodes) < THRESHOLDS['min_nodes']:
        violations.append('min_nodes')
    if min(counts.values()) < THRESHOLDS['min_tokens']:
        violations.append('min_tokens')
    result = deepcopy(base)
    pair_id = row['origin_id']+'__'+PROTOCOL
    pairs = []
    for label, view in [('N', nr), ('H', hr)]:
        pair = deepcopy(original[label])
        pair.update(pair_id=pair_id, record_id=pair_id+'__'+label)
        pair['detector_input']['reasoning_chain'] = view['text']
        for field,raw in [('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')]:
            require(pair['detector_input'][field] == row[raw] == row['raw_record'][raw], 'Original question or GT changed')
        pairs.append(pair)
    severity = measure_severity(row['indexed_smiles'], connections['chemistry_reference']['clean_execution'],
        base['plan']['execution'], clean_trace=nr['text'], h_trace=hr['text'], annotations=hr['spans'], token_labels=labels)
    result.update(protocol=PROTOCOL, style=style, pair_id=pair_id, pairs=pairs,
        N_visible=nr['text'], N_source_visible=row['N_visible'],
        reference_projection='original_answer_stripped_N' if style == 'native' else 'original_answer_stripped_N_with_open_boundary_connections',
        N_semantic_bindings=deepcopy(nr['bindings']), annotations=deepcopy(hr['spans']),
        semantic_bindings=deepcopy(hr['bindings']), text_edits=deepcopy(hr.get('edits', [])), token_labels=labels,
        connection_evidence=deepcopy(connections),
        error_counts={'roots':1,'distinct_wrong_nodes':len(nodes),'error_spans':len(hr['spans']),'tokens':counts},
        density_report={'parent_thresholds':THRESHOLDS,'violations':violations,'filtered':False},
        derivation={'parent_protocol':base['protocol'],'parent_style':base['style'],
                    'parent_pair_id':base['pair_id'],'plans_changed':False,'original_annotations_changed':False})
    result['plan'].update(nodes=deepcopy(rendered['nodes']), render=deepcopy(hr), severity=severity,
        candidate_mode='fixed_named_binding_with_connection_context_diagnostic',
        sampling_policy='fixed_four_origins_and_parent_plans_no_candidates_or_target_output_selection')
    result['plan']['checks']['connections'] = deepcopy(connections['checks'])
    result['plan']['checks']['renderer'] = deepcopy(rendered['checks'])
    result['release_contract'].update(version=PROTOCOL,
        program_gates='Frozen native traces and wrong plans; exact open-boundary projection and reconstruction; original question and token labels verified',
        expression_contrast='Same native traces, with or without a symmetric appended product-connection representation; chemical amplitude unchanged',
        interpretation='Open ports represent omitted original source connections, not atoms or hydrogen caps. No complete product is displayed; partial answer information is supplied. Propagation is conditional on the original intentionally false name-to-graph binding, not globally contradiction-free chemistry.')
    return result


def materialize(parent, output):
    if Path(output).is_symlink():
        raise FileExistsError('Use a fresh output directory: '+str(output))
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory: '+str(output))
    require(parent not in output.parents, 'Output cannot be inside its frozen parent')
    manifest, summary = read(parent/'manifest.json'), read(parent/'summary.json')
    original_manifest = read(parent/'source_manifest_snapshot.json')
    common.validate_development_manifest(original_manifest, summary)
    require(manifest['protocol'] == 'agent_named_binding_diagnostic_v1' and manifest['style'] == 'native'
            and manifest['selected'] == original_manifest['selected'] and manifest['split'] == original_manifest['split'],
            'Expected the frozen v15 native generation and development partition')
    require([summary[k] for k in ['planned','processed','accepted','rejected','outside_scope','production_accepted']]
            == [18,18,4,14,14,0], 'Parent diagnostic scope changed')
    cohort = list(common.EXPECTED_ORIGINS)
    require(sorted(manifest['eligible_origins']) == cohort, 'Parent eligible origins changed')
    selected = manifest['selected']; excluded = sorted(set(selected)-set(cohort))
    inputs = {o:read(parent/'origins'/o/'input.json') for o in selected}
    rows = {o:item['row'] for o,item in inputs.items()}
    require(all(o == row['origin_id'] for o,row in rows.items()), 'Origin input identity mismatch')
    require(sorted(o for o,row in rows.items() if common.eligible(row)) == cohort, 'Chemical scope drift')
    for origin in excluded:
        old = read(parent/'origins'/origin/'rejected.json')
        require(old['reason'] == 'outside_named_acyl_diagnostic_scope' and old['actual_api_calls'] == 0,
                'Parent exclusions must be the fixed outside-scope records')
    snapshots = read(parent/'tokenizer_snapshots.json')
    require(set(snapshots) == set(common.MODEL_NAMES), 'Both frozen tokenizer snapshots required')
    tokenizers = {name:common.FrozenTokenizer(snapshots[name]['backend'], name) for name in common.MODEL_NAMES}
    from molhallulens.modules.agent_generation.anchored_connections import build_anchored_connections
    from molhallulens.modules.agent_generation.anchored_connection_renderer import render_anchored_connection
    records, bundles, connection_bundles, bases = {s:[] for s in STYLES}, {}, {}, {}
    for origin in cohort:
        row = rows[origin]
        base = read(parent/'origins'/origin/'accepted.json')
        bundle = read(parent/'origins'/origin/'chemistry_bundle.json')
        require(base['protocol'] == manifest['protocol'] and base['style'] == 'native'
                and base['origin_id'] == origin and base['N_visible'] == row['N_visible'], 'Frozen native base mismatch')
        for key,value in [('edit_plan',bundle['wrong_plan']),('execution',bundle['wrong_execution']),('roots',bundle['roots'])]:
            require(base['plan'][key] == value, 'Frozen base chemistry mismatch')
        frozen = deepcopy((row,base,bundle))
        connections = build_anchored_connections(row,bundle)
        require((row,base,bundle) == frozen and connections['checks']['status'] == 'pass', 'Connection compiler failed or mutated inputs')
        frozen_connections = deepcopy(connections)
        for style in STYLES:
            rendered = render_anchored_connection(row,base,connections,style=style)
            require((row,base,bundle) == frozen and connections == frozen_connections, 'Renderer mutated frozen inputs')
            records[style].append(build_record(row,base,connections,rendered,style,tokenizers))
        bases[origin],bundles[origin],connection_bundles[origin] = deepcopy(base),deepcopy(bundle),deepcopy(connections)
    for a,b in zip(records[STYLES[0]],records[STYLES[1]],strict=True):
        for key in ['edit_plan','execution','roots']:
            require(a['plan'][key] == b['plan'][key], 'Graph intervention changed across styles')
        require(a['pair_id'] == b['pair_id'], 'Paired identities differ')
    implementation = {str(p.relative_to(ROOT)):p.read_text()
                      for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    for p in [ROOT/'molhallulens/modules/agent_generation/named_binding_inventory.json',
              ROOT/'scripts/derive_named_binding_diagnostic.py', Path(__file__)]:
        implementation[str(p.relative_to(ROOT))] = p.read_text()
    combined = {'protocol':PROTOCOL,'diagnostic_only':True,'actual_api_calls':0,'parent_generation':str(parent),'styles':{}}
    output.parent.mkdir(parents=True,exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output.name+'.data-',dir=output.parent))
    try:
        for style in STYLES:
            directory = staged/style
            violations = Counter(v for r in records[style] for v in r['density_report']['violations'])
            metadata = {'protocol':PROTOCOL,'style':style,'diagnostic_only':True,'production_accepted':0,
                'status_meaning':'accepted_for_paired_diagnostic_only','actual_api_calls':0,'outside_scope':14,
                'density_violation_counts':dict(violations),
                'density_violation_origins':[r['origin_id'] for r in records[style] if r['density_report']['violations']]}
            new_manifest = {**metadata,'selected':selected,'split':manifest['split'],'eligible_origins':cohort,
                'excluded_origins':excluded,'thresholds':THRESHOLDS,'parent_generation':str(parent),
                'development_source':manifest['development_source'],
                'selection_note':'All four fixed origins and wrong plans retained; no candidate selection or target-output access',
                'expression_contrast':'Frozen native traces with or without symmetric appended open-boundary product connections',
                'review_policy':'No new API review. Chemistry, text and token artifacts require independent checks before targets.'}
            brief = []
            for record in records[style]:
                origin = record['origin_id']; dest = directory/'origins'/origin
                write(dest/'accepted.json',record)
                write(dest/'input.json',{'protocol':PROTOCOL,'style':style,'row':rows[origin],'thresholds':THRESHOLDS})
                write(dest/'parent_input_snapshot.json',inputs[origin])
                write(dest/'parent_accepted_snapshot.json',bases[origin])
                write(dest/'reference.json',{'edit_plan':bundles[origin]['clean_plan'],'execution':bundles[origin]['clean_execution']})
                write(dest/'chemistry_bundle.json',bundles[origin])
                write(dest/'connection_bundle.json',connection_bundles[origin])
                write(dest/'fixed_plan.json',record['plan'])
                brief.append({k:record[k] for k in ['origin_id','status','acceptance_scope','error_counts','density_report','review_status']})
            for origin in excluded:
                record = {'origin_id':origin,'protocol':PROTOCOL,'style':style,'status':'rejected',
                    'reason':'outside_named_acyl_diagnostic_scope','exclusion_scope':'outside_diagnostic_scope','actual_api_calls':0}
                write(directory/'origins'/origin/'rejected.json',record)
                write(directory/'origins'/origin/'input.json',{'protocol':PROTOCOL,'style':style,'row':rows[origin]})
                brief.append(record)
            new_summary = {**metadata,'planned':18,'processed':18,'accepted':4,'rejected':14,
                'accepted_for_paired_diagnostic_only':4,'results':sorted(brief,key=lambda r:r['origin_id'])}
            for name,value in [('manifest',new_manifest),('summary',new_summary),('implementation_snapshot',implementation),
                               ('tokenizer_snapshots',snapshots),('source_manifest_snapshot',original_manifest),
                               ('parent_manifest_snapshot',manifest)]:
                write(directory/(name+'.json'),value)
            with (directory/'pairs.jsonl').open('w') as handle:
                for record in records[style]:
                    for pair in record['pairs']:
                        handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
            combined['styles'][style] = new_summary
        write(staged/'derivation_summary.json',combined)
        common.publish_noreplace(staged,output)
    except BaseException:
        shutil.rmtree(staged,ignore_errors=True)
        raise
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = materialize(args.parent_dir,args.output)
    print(json.dumps({style:{k:value[k] for k in ['planned','accepted','outside_scope','density_violation_counts']}
                      for style,value in result['styles'].items()},indent=2))


if __name__ == '__main__':
    main()
