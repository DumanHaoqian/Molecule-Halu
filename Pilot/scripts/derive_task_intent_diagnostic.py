#!/usr/bin/env python3
"""Publish a fixed-graph binding versus perceived-task-group diagnostic.

Only frozen development generation artifacts are read. No target predictions,
new candidates, API calls, or heldout evaluation influence this compiler.
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
_spec = importlib.util.spec_from_file_location('intent_diagnostic_common', ROOT/'scripts/derive_named_binding_diagnostic.py')
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
read, write, require = common.read, common.write, common.require
PROTOCOL = 'agent_task_intent_diagnostic_v1'
STYLES = ('binding', 'intent')
THRESHOLDS = {'min_roots': 1, 'min_nodes': 6, 'min_tokens': 40}


def build_record(row, base, bundle, rendered, style, tokenizers):
    from molhallulens.modules.agent_generation.labels import label_tokens
    from molhallulens.modules.agent_generation.quality import find_answer_leakage
    from molhallulens.modules.agent_generation.severity import measure_severity

    require(bundle['checks']['status'] == rendered['checks']['status'] == 'pass', 'Chemistry or rendering checks failed')
    nr,hr = rendered['N_render'],rendered['H_render']
    original = {p['variant_label']:p for p in base['pairs']}
    physical = bundle['physical_reference']
    require(nr['text'] == base['N_visible'] == original['N']['detector_input']['reasoning_chain'], 'Frozen N text changed')
    require(len(rendered['roots']) == 1 and rendered['roots'][0]['id'] == 'r1', 'Exactly one fixed root required')
    root = rendered['roots'][0]
    require((root['node_id'],root['type']) == (('fragment','structural') if style=='binding' else ('task_group','task_interpretation')),
            'Incorrect current semantic root')
    if style == 'binding':
        require(hr['text'] == original['H']['detector_input']['reasoning_chain'] and hr['spans'] == base['annotations'],
                'Frozen binding control changed')
    else:
        require(hr['text'] != original['H']['detector_input']['reasoning_chain'], 'Intent H must change its perceived group')
    for label,view in [('N',nr),('H',hr)]:
        require(not find_answer_leakage(view['text'],[row['gt_smiles'],physical['wrong_execution']['product_smiles']]),
                'A complete answer or forbidden cue appears in '+label)
        for field in ['spans','bindings']:
            for item in view[field]:
                require(0 <= item['start'] < item['end'] <= len(view['text'])
                        and view['text'][item['start']:item['end']] == item['text'], 'Semantic offset/text mismatch')
    require(not nr['spans'] and hr['spans'], 'Expected clean N and annotated H')
    require({s['node_id'] for s in hr['spans'] if s['kind']=='root_error'} == {root['node_id']},
            'Old and new semantic roots cannot coexist')
    labels = {name:label_tokens(hr['text'],hr['spans'],tokenizer) for name,tokenizer in tokenizers.items()}
    counts = {name:sum(any(x!='unchanged' for x in t['labels']) for t in values) for name,values in labels.items()}
    nodes = {s['node_id'] for s in hr['spans']}
    violations = []
    if len(nodes)<THRESHOLDS['min_nodes']: violations.append('min_nodes')
    if min(counts.values())<THRESHOLDS['min_tokens']: violations.append('min_tokens')
    pair_id = row['origin_id']+'__'+PROTOCOL
    pairs = []
    for label,view in [('N',nr),('H',hr)]:
        pair = deepcopy(original[label])
        pair.update(pair_id=pair_id,record_id=pair_id+'__'+label)
        pair['detector_input']['reasoning_chain'] = view['text']
        for field,raw in [('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')]:
            require(pair['detector_input'][field] == row[raw] == row['raw_record'][raw], 'Original question or GT changed')
        pairs.append(pair)
    severity = measure_severity(row['indexed_smiles'],physical['clean_execution'],physical['wrong_execution'],
        clean_trace=nr['text'],h_trace=hr['text'],annotations=hr['spans'],token_labels=labels)
    result = {
        'origin_id':row['origin_id'],'protocol':PROTOCOL,'style':style,'status':'accepted',
        'acceptance_scope':'paired_diagnostic_only','pair_id':pair_id,'pairs':pairs,
        'N_raw':base['N_raw'],'N_source_visible':row['N_visible'],'N_visible':nr['text'],
        'reference_projection':'frozen_parent_N_with_open_boundary_connections',
        'N_semantic_bindings':deepcopy(nr['bindings']),'annotations':deepcopy(hr['spans']),
        'semantic_bindings':deepcopy(hr['bindings']),'text_edits':deepcopy(hr.get('edits',[])),
        'token_labels':labels,'intent_evidence':deepcopy(rendered['intent_evidence']),
        'physical_reference':deepcopy(physical),'parent_provenance':deepcopy(bundle['parent_provenance']),
        'plan':{'edit_plan':deepcopy(physical['wrong_plan']),'execution':deepcopy(physical['wrong_execution']),
            'roots':deepcopy(rendered['roots']),'nodes':deepcopy(rendered['nodes']),'render':deepcopy(hr),
            'severity':severity,'candidate_mode':'fixed_graph_task_interpretation_diagnostic',
            'sampling_policy':'fixed_four_origins_and_parent_graphs_no_candidates_or_target_output_selection',
            'checks':{'physical_and_intent':deepcopy(bundle['checks']),'renderer':deepcopy(rendered['checks'])}},
        'error_counts':{'roots':1,'distinct_wrong_nodes':len(nodes),'error_spans':len(hr['spans']),'tokens':counts},
        'density_report':{'parent_thresholds':THRESHOLDS,'violations':violations,'filtered':False},
        'actual_api_calls':0,'review_status':deepcopy(base['review_status']),
        'release_contract':{'version':PROTOCOL,'status':'diagnostic_only','diagnostic_only':True,
            'production_eligible':False,'model_outputs_used_for_selection':False,
            'program_gates':'Original input and physical plans frozen; current semantic root and complete text/token provenance checked',
            'expression_contrast':'Binding versus perceived task group in H; N text and physical graphs identical',
            'interpretation':('Historical false group-name-to-graph binding retained as control.' if style=='binding' else
                'H names the actual wrong group consistently but conflicts with the unchanged original instruction. The fragment propagates a task-interpretation root.')+
                ' Open ports are omitted source connections. No complete answer is displayed; partial answer information remains reconstructible.'},
        'derivation':{'parent_protocol':base['protocol'],'parent_style':base['style'],'parent_pair_id':base['pair_id'],
            'physical_plans_changed':False,'N_text_changed':False,'semantic_root_changed':style=='intent'}
    }
    if style=='binding':
        result['binding_evidence'] = deepcopy(base['binding_evidence'])
    return result


def materialize(parent,output):
    if Path(output).is_symlink(): raise FileExistsError('Use a fresh output directory: '+str(output))
    parent,output = Path(parent).resolve(),Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh output directory: '+str(output))
    require(parent not in output.parents,'Output cannot be inside its frozen parent')
    manifest,summary = read(parent/'manifest.json'),read(parent/'summary.json')
    original_manifest = read(parent/'source_manifest_snapshot.json')
    common.validate_development_manifest(original_manifest,summary)
    require(manifest['protocol']=='agent_anchored_connections_diagnostic_v1' and manifest['style']=='native_with_connections'
            and manifest['selected']==original_manifest['selected'] and manifest['split']==original_manifest['split'],
            'Expected frozen v16 augmented parent and original development partition')
    require([summary[k] for k in ['planned','processed','accepted','rejected','outside_scope','production_accepted']]
            == [18,18,4,14,14,0],'Parent diagnostic scope changed')
    cohort = list(common.EXPECTED_ORIGINS)
    require(sorted(manifest['eligible_origins'])==cohort,'Parent eligible origins changed')
    selected = manifest['selected']; excluded = sorted(set(selected)-set(cohort))
    inputs = {o:read(parent/'origins'/o/'input.json') for o in selected}
    rows = {o:item['row'] for o,item in inputs.items()}
    require(all(o==row['origin_id'] for o,row in rows.items()),'Origin input identity mismatch')
    require(sorted(o for o,row in rows.items() if common.eligible(row))==cohort,'Chemical scope drift')
    for origin in excluded:
        old = read(parent/'origins'/origin/'rejected.json')
        require(old['reason']=='outside_named_acyl_diagnostic_scope' and old['actual_api_calls']==0,'Exclusions changed')
    snapshots = read(parent/'tokenizer_snapshots.json')
    require(set(snapshots)==set(common.MODEL_NAMES),'Both frozen tokenizer snapshots required')
    tokenizers = {name:common.FrozenTokenizer(snapshots[name]['backend'],name) for name in common.MODEL_NAMES}
    from molhallulens.modules.agent_generation.task_intent_chemistry import build_task_intent
    from molhallulens.modules.agent_generation.task_intent_renderer import render_task_intent,TASK_INTENT_INVENTORY_PATH
    records,bundles,bases = {s:[] for s in STYLES},{},{}
    for origin in cohort:
        row = rows[origin]; base = read(parent/'origins'/origin/'accepted.json')
        require(base['protocol']==manifest['protocol'] and base['style']==manifest['style'] and base['origin_id']==origin,
                'Frozen base identity mismatch')
        frozen = deepcopy((row,base))
        bundle = build_task_intent(row,base)
        require((row,base)==frozen and bundle['checks']['status']=='pass','Intent compiler failed or mutated inputs')
        for key,bkey in [('edit_plan','wrong_plan'),('execution','wrong_execution')]:
            require(base['plan'][key]==bundle['physical_reference'][bkey],'Frozen physical plan changed')
        frozen_bundle = deepcopy(bundle)
        for style in STYLES:
            rendered = render_task_intent(row,base,bundle,style=style)
            require((row,base)==frozen and bundle==frozen_bundle,'Renderer mutated frozen inputs')
            records[style].append(build_record(row,base,bundle,rendered,style,tokenizers))
        bases[origin],bundles[origin] = deepcopy(base),deepcopy(bundle)
    for a,b in zip(records['binding'],records['intent'],strict=True):
        require(a['pair_id']==b['pair_id'] and a['N_visible']==b['N_visible']
                and a['physical_reference']==b['physical_reference'],'Paired identity, N or physics differs')
    implementation = {str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    for p in [ROOT/'molhallulens/modules/agent_generation/named_binding_inventory.json',Path(TASK_INTENT_INVENTORY_PATH),
              ROOT/'scripts/derive_named_binding_diagnostic.py',Path(__file__)]:
        implementation[str(p.relative_to(ROOT))] = p.read_text()
    combined = {'protocol':PROTOCOL,'diagnostic_only':True,'actual_api_calls':0,'parent_generation':str(parent),'styles':{}}
    output.parent.mkdir(parents=True,exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output.name+'.data-',dir=output.parent))
    try:
        for style in STYLES:
            directory = staged/style
            metadata = {'protocol':PROTOCOL,'style':style,'diagnostic_only':True,'production_accepted':0,
                'status_meaning':'accepted_for_paired_diagnostic_only','actual_api_calls':0,'outside_scope':14,
                'density_violation_counts':dict(Counter(v for r in records[style] for v in r['density_report']['violations'])),
                'density_violation_origins':[r['origin_id'] for r in records[style] if r['density_report']['violations']]}
            new_manifest = {**metadata,'selected':selected,'split':manifest['split'],'eligible_origins':cohort,
                'excluded_origins':excluded,'thresholds':THRESHOLDS,'parent_generation':str(parent),
                'development_source':manifest['development_source'],
                'selection_note':'Same four frozen origins and physical plans; no candidate or target-output selection',
                'expression_contrast':'Perceived-task-group closure only in H; N and physical graphs unchanged',
                'review_policy':'No new API review. Independent chemistry/text/token and request checks required before targets.'}
            brief = []
            for record in records[style]:
                origin = record['origin_id']; dest = directory/'origins'/origin
                values = {'accepted':record,'input':{'protocol':PROTOCOL,'style':style,'row':rows[origin],'thresholds':THRESHOLDS},
                    'parent_input_snapshot':inputs[origin],'parent_accepted_snapshot':bases[origin],
                    'reference':{'edit_plan':bundles[origin]['physical_reference']['clean_plan'],
                                 'execution':bundles[origin]['physical_reference']['clean_execution']},
                    'intent_bundle':bundles[origin],'physical_reference':bundles[origin]['physical_reference'],
                    'fixed_plan':record['plan']}
                for name,value in values.items(): write(dest/(name+'.json'),value)
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
                               ('parent_manifest_snapshot',manifest)]: write(directory/(name+'.json'),value)
            with (directory/'pairs.jsonl').open('w') as handle:
                for record in records[style]:
                    for pair in record['pairs']: handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
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
    print(json.dumps({s:{k:v[k] for k in ['planned','accepted','outside_scope','density_violation_counts']}
                      for s,v in result['styles'].items()},indent=2))


if __name__=='__main__': main()
