"""Validate and merge explicitly supplied H/N batches; never overwrite inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.error_planning import FragmentPool, UnifiedHallucinationPlanner
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pair(h, n):
    require(h['variant_label'] == 'H' and n['variant_label'] == 'N', 'H/N required')
    for key in ('origin_id', 'pair_id', 'variant_index', 'pair_alignment'):
        require(h[key] == n[key], f'pair mismatch: {key}')
    require(h['matched_record_id'] == n['record_id'] and n['matched_record_id'] == h['record_id'], 'matched IDs')
    require(h['labels']['hallucination_present'] is True and n['labels']['hallucination_present'] is False, 'polarity')
    require(h['hallucination_spans'] and not h['control_spans'] and not n['hallucination_spans'], 'span polarity')
    for record, key in ((h, 'hallucination_spans'), (n, 'control_spans')):
        data = record['detector_input']
        text = record['serialized']['text']
        expected = (f"<MOLECULE>\n{data['indexed_smiles']}\n\n<INSTRUCTION>\n{data['instruction']}"
                    f"\n\n<REASONING>\n{data['reasoning_chain']}\n\n<FINAL_ANSWER>\n{data['final_answer']}")
        require(text == expected, 'serialized composition')
        require(hashlib.sha256(text.encode()).hexdigest() == record['serialized']['sha256'], 'text hash')
        require(data['reasoning_chain'] == '\n\n'.join(record['step_texts']), 'step composition')
        roots = {node for item in record['mutation_events'] for node in item['target_node_ids']}
        propagated = {item['target_node_id'] for item in record['propagation_events']}
        require(not roots & propagated, 'root/propagated overlap')
        cursor = 0
        for span in sorted(record[key], key=lambda item: item['serialized_span']):
            start, end = span['serialized_span']
            require(cursor <= start < end <= len(text), 'overlapping or invalid span')
            require(text[start:end] == span['text'], 'serialized span round-trip')
            lo, hi = span['span']
            require(data[span['component']][lo:hi] == span['text'], 'component span round-trip')
            a, b = span['serialized_context_span']
            require(0 <= a <= start < end <= b <= len(text), 'context bounds')
            cursor = end
    controls = {span['pair_occurrence_id']: span for span in n['control_spans']}
    require(len(controls) == len(n['control_spans']), 'duplicate control IDs')
    ids = [span['pair_occurrence_id'] for span in h['hallucination_spans']]
    require(len(set(ids)) == len(ids) and set(ids) == set(controls), 'one-to-one spans')
    for span in h['hallucination_spans']:
        control = controls[span['pair_occurrence_id']]
        same = len(span['text']) == len(control['text'])
        require(span['same_char_length'] == control['same_char_length'] == same, 'length flag')
        require(span['node_id'] == control['node_id'], 'node attribution')
    # Validate every byte-identical step, even if another step was regenerated.
    offset = 0
    for index, (hs, ns, alignment) in enumerate(zip(h['step_texts'], n['step_texts'], h['pair_alignment']), 1):
        require(alignment['step_index'] == index, 'step alignment order')
        if alignment['pair_alignment'] == 'byte_identical':
            replaced = hs
            spans = [s for s in h['hallucination_spans'] if s['component'] == 'reasoning_chain' and s['step_index'] == index]
            for span in sorted(spans, key=lambda s: s['span'], reverse=True):
                start, end = span['span']
                replaced = replaced[:start-offset] + controls[span['pair_occurrence_id']]['text'] + replaced[end-offset:]
            require(replaced == ns, f'step {index} substitution invariant')
        else:
            require(alignment['pair_alignment'] == 'regenerated', 'unknown alignment')
        offset += len(hs) + 2
    require(len(h['step_texts']) == len(n['step_texts']) == len(h['pair_alignment']), 'step counts')
    if all(a['pair_alignment'] == 'byte_identical' for a in h['pair_alignment']):
        replaced = h['serialized']['text']
        for span in sorted(h['hallucination_spans'], key=lambda s: s['serialized_span'], reverse=True):
            start, end = span['serialized_span']
            replaced = replaced[:start] + controls[span['pair_occurrence_id']]['text'] + replaced[end:]
        require(replaced == n['serialized']['text'], 'full pair substitution invariant')


def merge(inputs, output, dataset):
    require(not output.exists() and not output.with_suffix('.validation.json').exists(), 'refusing overwrite')
    references = tuple(build_reference_dag(o) for o in ChemCoTMolEditAdapter().load(dataset))
    planner = UnifiedHallucinationPlanner(FragmentPool.from_reference_artifacts(references),
                                         replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode='maximum'))
    plans = {p.origin_id: p for p in (planner.plan(r, variant_index=0) for r in references)}
    pairs, lines, sources = {}, [], []
    protocol_corrections = []
    for path in inputs:
        raw = path.read_bytes()
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()})
        for line in raw.splitlines(keepends=True):
            if not line.strip():
                continue
            record = json.loads(line)
            realization = record['text_realization']
            observed = {s['protocol'] for s in realization.get('step_execution', []) if 'protocol' in s}
            if len(observed) == 1 and realization.get('protocol_version') not in observed:
                actual = next(iter(observed))
                previous = realization.get('protocol_version')
                # v18 renderer had a stale hard-coded v15 metadata field. Use
                # unanimous per-step provenance, never infer from text/model.
                require(previous == 'poe_segments_v15' and actual == 'poe_segments_v18', 'unrecognized metadata mismatch')
                realization['protocol_version'] = actual
                realization['protocol_metadata_correction'] = {'previous': previous, 'source': 'unanimous_step_execution_protocol'}
                protocol_corrections.append(record['record_id'])
                line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode()
            origin, variant = record['origin_id'], record['variant_label']
            require(origin in plans and record['variant_index'] == 0, 'unexpected origin/variant')
            row = pairs.setdefault(origin, {})
            require(variant not in row, f'duplicate {origin}/{variant}')
            row[variant] = record
            lines.append(line if line.endswith(b'\n') else line + b'\n')
    require(set(pairs) == set(plans), f'incomplete origin coverage: {len(pairs)}/{len(plans)}')
    alignment, modes, models, protocols, execution = Counter(), Counter(), Counter(), Counter(), Counter()
    span_count = same_count = 0
    for origin, pair in pairs.items():
        require(set(pair) == {'H', 'N'}, f'incomplete pair: {origin}')
        h, n = pair['H'], pair['N']
        validate_pair(h, n)
        expected = plans[origin]
        require(h['edit_count'] == n['edit_count'] == len(expected.mutations), f'not maximum: {origin}')
        expected_events = json.loads(json.dumps([m.to_dict() for m in expected.mutations]))
        require(h['mutation_events'] == n['mutation_events'] == expected_events, f'plan drift: {origin}')
        alignment.update(a['pair_alignment'] for a in h['pair_alignment'])
        modes.update(a['rewrite_mode'] for a in h['pair_alignment'])
        models[h['text_realization'].get('bot_name', 'unknown')] += 1
        protocols[h['text_realization'].get('protocol_version', 'legacy_unrecorded')] += 1
        execution.update(s.get('response_mode', 'legacy_unrecorded') for s in h['text_realization'].get('step_execution', []))
        span_count += len(h['hallucination_spans'])
        same_count += sum(s['same_char_length'] for s in h['hallucination_spans'])
    merged = b''.join(lines)
    report = dict(origins=len(pairs), records=len(lines), H=len(pairs), N=len(pairs),
                  sources=sources, sha256=hashlib.sha256(merged).hexdigest(),
                  protocol_metadata_corrections=protocol_corrections,
                  pair_alignment=dict(alignment), rewrite_modes=dict(modes), models=dict(models),
                  protocols=dict(protocols), response_modes=dict(execution), paired_spans=span_count,
                  same_char_length_count=same_count, same_char_length_ratio=same_count/span_count,
                  checks='offsets, hashes, polarity, pairing, per-step/full substitution, exact maximum mutation plans, complete coverage')
    with output.open('xb') as handle:
        handle.write(merged)
    with output.with_suffix('.validation.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1] / 'Dataset')
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.output, args.dataset), ensure_ascii=False, indent=2))
