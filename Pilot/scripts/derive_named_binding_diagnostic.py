#!/usr/bin/env python3
"""Publish two fixed four-origin named-fragment diagnostics on CPU.

No model calls, candidate search, outcome filtering or heldout access occurs.
The complete 18-origin development denominator is retained. Existing outputs
are never overwritten, and both styles are published together after validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = 'agent_named_binding_diagnostic_v1'
STYLES = ('ledger', 'native')
MODEL_NAMES = ('ChemDFM-R-14B', 'Chem-R-8B')
EXPECTED_ORIGINS = tuple('mol_edit.add_v2.' + n for n in ('0098','0172','0193','0194'))
EXPECTED_DEVELOPMENT = frozenset(
    ['mol_edit.add_v2.' + n for n in ('0098','0101','0172','0193','0194','0287')]
    + ['mol_edit.delete_v2.' + n for n in ('0024','0033','0071','0078','0126','0255')]
    + ['mol_edit.substitute_v2.' + n for n in ('0005','0009','0043','0150','0197','0272')])
THRESHOLDS = {'min_roots': 1, 'min_nodes': 6, 'min_tokens': 40}


def eligible(row):
    return row['subtask'] == 'add' and bool(re.search(r'\b(?:acylate|acetylate)\b', row['instruction'], re.I))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_development_manifest(manifest, summary):
    """Check partition membership before opening any origin input files."""
    selected, split = manifest['selected'], manifest['split']
    require(manifest['protocol'] == 'agent_source_state_v1' and split['seed'] == 20260920,
            'Expected the frozen source-state development protocol and seed')
    require(len(selected) == len(set(selected)) == summary['planned'] == summary['processed'] == 18,
            'Expected the complete fixed18 development inputs')
    require(set(selected) == set(split['development']) == EXPECTED_DEVELOPMENT
            and len(split['development']) == 18, 'Development partition membership drift')
    require(len(split['heldout']) == len(set(split['heldout'])) == 132
            and not set(selected).intersection(split['heldout']), 'Invalid or overlapping heldout partition')


def publish_noreplace(staged, output):
    """Publish without replacing any destination, including on the NAS.

    Some network filesystems reject renameat2 flags. There, an atomic symlink
    publishes the already complete sibling directory; that directory becomes
    permanent backing storage and must remain for the lifetime of the artifact.
    """
    require(staged.is_dir() and staged.parent.resolve() == output.parent.resolve(),
            'Publication requires a complete sibling directory')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is not None:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staged), -100, os.fsencode(output), 1) == 0:
            return
        error = ctypes.get_errno()
        if error not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise OSError(error, os.strerror(error), str(output))
    os.symlink(staged.name, output, target_is_directory=True)


class FrozenTokenizer:
    is_fast = True

    def __init__(self, backend, name):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(backend)
        self.name_or_path = name

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=True):
        encoded = self.backend.encode(text, add_special_tokens=add_special_tokens)
        return {'input_ids': encoded.ids, 'offset_mapping': encoded.offsets}

    def convert_ids_to_tokens(self, ids):
        return [self.backend.id_to_token(token_id) for token_id in ids]


def build_record(row, bundle, rendered, style, tokenizers):
    from molhallulens.modules.agent_generation.labels import label_tokens
    from molhallulens.modules.agent_generation.quality import find_answer_leakage
    from molhallulens.modules.agent_generation.severity import measure_severity

    require(rendered['checks']['status'] == 'pass', 'Renderer checks failed')
    nr, hr = rendered['N_render'], rendered['H_render']
    roots = rendered['roots']
    require(len(roots) == 1 and roots[0]['node_id'] == 'fragment', 'Expected one fragment binding root')
    require(roots == bundle['roots'], 'Rendered roots differ from checked chemistry')
    require(nr['text'] != hr['text'], 'N and H must differ')
    if style == 'native':
        require(nr['text'] == row['N_visible'], 'Native N changed outside the original answer projection')
    for label, view in (('N', nr), ('H', hr)):
        require(not find_answer_leakage(view['text'], [row['gt_smiles'], bundle['wrong_execution']['product_smiles']]),
                label + ' contains a complete answer or forbidden cue')
        for field in ('spans', 'bindings'):
            for item in view[field]:
                require(view['text'][item['start']:item['end']] == item['text'], label + ' semantic offset mismatch')
    require({r for s in hr['spans'] if s['kind'] == 'root_error' for r in s['root_ids']} == {r['id'] for r in roots},
            'Root annotation coverage differs')
    labels = {name: label_tokens(hr['text'], hr['spans'], tok) for name,tok in tokenizers.items()}
    counts = {name: sum(any(label != 'unchanged' for label in item['labels']) for item in values)
              for name,values in labels.items()}
    wrong_nodes = {span['node_id'] for span in hr['spans']}
    violations = []
    if len(roots) < THRESHOLDS['min_roots']:
        violations.append('min_roots')
    if len(wrong_nodes) < THRESHOLDS['min_nodes']:
        violations.append('min_nodes')
    if min(counts.values()) < THRESHOLDS['min_tokens']:
        violations.append('min_tokens')
    pair_id = row['origin_id'] + '__' + PROTOCOL
    pairs = [{'record_id': pair_id+'__'+label, 'pair_id': pair_id, 'origin_id': row['origin_id'],
              'subtask': row['subtask'], 'variant_label': label, 'edit_count': len(roots) if label == 'H' else 0,
              'detector_input': {'indexed_smiles': row['indexed_smiles'], 'instruction': row['instruction'],
                                 'reasoning_chain': view['text'], 'final_answer': row['gt_smiles']}}
             for label,view in (('N',nr),('H',hr))]
    for pair in pairs:
        for field,raw in (('instruction','instruction'),('indexed_smiles','indexed_smiles'),('final_answer','gt_smiles')):
            require(pair['detector_input'][field] == row['raw_record'][raw], 'Original question or GT changed')
    severity = measure_severity(row['indexed_smiles'], bundle['clean_execution'], bundle['wrong_execution'],
        clean_trace=nr['text'], h_trace=hr['text'], annotations=hr['spans'], token_labels=labels)
    return deepcopy({'origin_id': row['origin_id'], 'protocol': PROTOCOL, 'style': style, 'status': 'accepted',
        'acceptance_scope': 'paired_diagnostic_only', 'pair_id': pair_id, 'pairs': pairs,
        'N_raw': row['N_raw'], 'N_source_visible': row['N_visible'], 'N_visible': nr['text'],
        'reference_projection': 'original_answer_stripped_N' if style == 'native' else 'typed_reference_no_local_window',
        'N_semantic_bindings': nr['bindings'], 'annotations': hr['spans'], 'semantic_bindings': hr['bindings'],
        'text_edits': hr.get('edits', []), 'token_labels': labels,
        'binding_evidence': deepcopy(rendered.get('binding_evidence', {})),
        'plan': {'edit_plan': bundle['wrong_plan'], 'execution': bundle['wrong_execution'],
                 'roots': roots, 'nodes': rendered['nodes'], 'render': hr, 'severity': severity,
                 'candidate_mode': 'fixed_named_fragment_binding_diagnostic',
                 'sampling_policy': 'one_predeclared_graph_on_every_eligible_origin_no_selection',
                 'checks': {'chemistry': bundle['checks'], 'renderer': rendered['checks']}},
        'error_counts': {'roots': len(roots), 'distinct_wrong_nodes': len(wrong_nodes),
                         'error_spans': len(hr['spans']), 'tokens': counts},
        'density_report': {'parent_thresholds': THRESHOLDS, 'violations': violations, 'filtered': False},
        'actual_api_calls': 0,
        'review_status': {'reference_status': 'unknown_not_repeated', 'conformance_status': 'unknown_not_repeated',
                         'blind_status': 'unknown_not_repeated', 'agent_consensus': None,
                         'interpretation': 'No fresh API review or approval is claimed for this bounded diagnostic'},
        'release_contract': {'version': PROTOCOL, 'status': 'diagnostic_only', 'diagnostic_only': True,
            'production_eligible': False, 'model_outputs_used_for_selection': False,
            'program_gates': 'Fixed correct anchors/source and wrong acyl graph; exact original question; checked text spans and graph-conditioned consequences; no complete products',
            'interpretation': 'Native name-to-graph binding is intentionally false; conditional propagation is coherent, not globally contradiction-free chemistry',
            'expression_contrast': 'Original names/source-role/connection-language package versus a ledger without explicit names; not a name-only effect'}})


def materialize(development, output):
    if Path(output).is_symlink():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    development, output = Path(development).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    require(development not in output.parents, 'Output cannot be nested inside the frozen development source')
    manifest, summary = read(development/'manifest.json'), read(development/'summary.json')
    validate_development_manifest(manifest, summary)
    selected = manifest['selected']
    inputs = {origin: read(development/'origins'/origin/'input.json') for origin in selected}
    rows = {origin: item['row'] for origin,item in inputs.items()}
    require(all(origin == row['origin_id'] for origin,row in rows.items()), 'Development origin identity drift')
    cohort = sorted(origin for origin,row in rows.items() if eligible(row))
    require(tuple(cohort) == EXPECTED_ORIGINS, 'Acylation eligibility differs from the predeclared four-origin scope')
    excluded = sorted(set(selected) - set(cohort))
    snapshots = read(development/'tokenizer_snapshots.json')
    require(set(snapshots) == set(MODEL_NAMES), 'Both frozen tokenizer snapshots are required')
    tokenizers = {name: FrozenTokenizer(snapshots[name]['backend'],name) for name in MODEL_NAMES}
    from molhallulens.modules.agent_generation.named_binding_chemistry import build_named_binding
    from molhallulens.modules.agent_generation.named_binding_renderer import render_named_binding, INVENTORY_PATH
    bundles, records = {}, {style: [] for style in STYLES}
    for origin in cohort:
        row = rows[origin]
        frozen_row = deepcopy(row)
        bundle = build_named_binding(row)
        require(row == frozen_row, 'Chemistry compiler changed original input')
        require(bundle['checks']['status'] == 'pass', 'Named chemistry failed for '+origin)
        bundles[origin] = deepcopy(bundle)
        for style in STYLES:
            view = render_named_binding(row, bundle, style=style)
            require(row == frozen_row and bundle == bundles[origin],
                    'Renderer changed frozen input or chemistry bundle')
            records[style].append(build_record(row,bundle,view,style,tokenizers))
    for a,b in zip(records['ledger'],records['native'],strict=True):
        for key in ('edit_plan','execution','roots'):
            require(a['plan'][key] == b['plan'][key], 'Chemistry differs across expressions')
        require(a['pair_id'] == b['pair_id'], 'Paired IDs differ across expressions')
    implementation = {str(p.relative_to(ROOT)): p.read_text()
                      for p in sorted((ROOT/'molhallulens/modules/agent_generation').glob('*.py'))}
    implementation[str(Path(INVENTORY_PATH).relative_to(ROOT))] = Path(INVENTORY_PATH).read_text()
    implementation['scripts/derive_named_binding_diagnostic.py'] = Path(__file__).read_text()
    combined = {'protocol': PROTOCOL, 'diagnostic_only': True, 'actual_api_calls': 0,
                'development_source': str(development), 'styles': {}}
    output.parent.mkdir(parents=True,exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=output.name+'.data-',dir=output.parent))
    try:
        for style in STYLES:
            directory = staged/style
            violations = Counter(v for record in records[style] for v in record['density_report']['violations'])
            common = {'protocol': PROTOCOL, 'style': style, 'diagnostic_only': True,
                      'status_meaning': 'accepted_for_paired_diagnostic_only', 'production_accepted': 0,
                      'actual_api_calls': 0, 'outside_scope': len(excluded),
                      'density_violation_counts': dict(violations),
                      'density_violation_origins': [r['origin_id'] for r in records[style] if r['density_report']['violations']]}
            current_manifest = {**common, 'selected': selected, 'split': manifest['split'],
                'development_source': str(development), 'eligible_origins': cohort, 'excluded_origins': excluded,
                'eligibility_rule': 'Addition instructions specifying acylate or acetylate; fixed four development origins',
                'thresholds': THRESHOLDS, 'selection_note': 'All eligible examples retained; fixed wrong graph; no target outputs or resampling',
                'review_policy': 'No new API review; exact program and independent artifact review required before target runs',
                'expression_contrast': 'Native original-name/source-role/connection package versus nameless typed ledger'}
            brief = []
            for record in records[style]:
                origin = record['origin_id']; dest = directory/'origins'/origin
                write(dest/'accepted.json',record)
                write(dest/'input.json',{'protocol':PROTOCOL,'style':style,'row':rows[origin],'thresholds':THRESHOLDS})
                write(dest/'parent_input_snapshot.json',inputs[origin])
                write(dest/'reference.json',{'edit_plan':bundles[origin]['clean_plan'],'execution':bundles[origin]['clean_execution']})
                write(dest/'chemistry_bundle.json',bundles[origin])
                write(dest/'fixed_plan.json',record['plan'])
                brief.append({k:record[k] for k in ('origin_id','status','acceptance_scope','error_counts','density_report','review_status')})
            for origin in excluded:
                record = {'origin_id':origin,'protocol':PROTOCOL,'style':style,'status':'rejected',
                          'reason':'outside_named_acyl_diagnostic_scope','exclusion_scope':'outside_diagnostic_scope',
                          'actual_api_calls':0}
                write(directory/'origins'/origin/'rejected.json',record)
                write(directory/'origins'/origin/'input.json',{'protocol':PROTOCOL,'style':style,'row':rows[origin]})
                brief.append(record)
            current_summary = {**common,'planned':18,'processed':18,'accepted':4,'rejected':14,
                               'accepted_for_paired_diagnostic_only':4,'results':sorted(brief,key=lambda r:r['origin_id'])}
            write(directory/'manifest.json',current_manifest)
            write(directory/'summary.json',current_summary)
            write(directory/'implementation_snapshot.json',implementation)
            write(directory/'tokenizer_snapshots.json',snapshots)
            write(directory/'source_manifest_snapshot.json',manifest)
            with (directory/'pairs.jsonl').open('w') as handle:
                for record in records[style]:
                    for pair in record['pairs']:
                        handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
            combined['styles'][style] = current_summary
        write(staged/'derivation_summary.json',combined)
        publish_noreplace(staged, output)
    except BaseException:
        shutil.rmtree(staged,ignore_errors=True)
        raise
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = materialize(args.development_dir,args.output)
    print(json.dumps({style:{k:value[k] for k in ('planned','accepted','outside_scope','density_violation_counts')}
                      for style,value in result['styles'].items()},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
