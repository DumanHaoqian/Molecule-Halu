"""Inspect top validation-token contexts for selected SAE features.

This is a read-only interpretability aid, not a semantic or hallucination score.
Only local tokenizer files and the SAE checkpoint are loaded; ChemDFM weights are
not needed. Output retains document, cache-row, token and character provenance.
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from activation_inputs import encode_document, iter_selected_documents, read_selection
from activation_store import ActivationStore, normalize, sha256_file
from sae_models import reconstruct

HERE = Path(__file__).resolve().parent


def _metadata(store, entry):
    if 'metadata_path' not in entry:
        raise ValueError('Feature inspection requires per-document cache metadata')
    path = (store.root / entry['metadata_path']).resolve()
    digest = entry.get('metadata_sha256', entry.get('meta_sha256'))
    if not path.is_relative_to(store.root) or not path.is_file() or sha256_file(path) != digest:
        raise ValueError('Cache metadata path or checksum mismatch')
    rows, end = [], 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count = row['row_end'] - row['row_start']
            if row['row_start'] != end or count <= 0 or count != len(row['positions']) or count != len(row['token_ids']):
                raise ValueError('Cache token metadata ranges are not contiguous or aligned')
            if type(row['selection_index']) is not int or row['selection_index'] < 0:
                raise ValueError('Invalid metadata selection_index')
            if any(type(p) is not int or p < 0 for p in row['positions']) or any(a >= b for a, b in zip(row['positions'], row['positions'][1:])):
                raise ValueError('Invalid metadata token positions')
            rows.append(row)
            end = row['row_end']
    if end != entry['n_rows']:
        raise ValueError('Metadata row count differs from activation shard')
    return rows


@torch.inference_mode()
def collect_feature_examples(sae, store: ActivationStore, normalizer: dict, feature_ids,
                             *, top_k: int = 10, batch_size: int = 256,
                             max_tokens: int | None = None, min_activation: float = 0.) -> dict:
    """Stream held-out rows, retaining only top_k hits per requested feature.

    Gates use the same batch-independent inference as eval_sae. Scores strictly
    above min_activation qualify; equal scores prefer the earlier cache row.
    ``max_tokens`` selects a fixed prefix, making a partial scan explicit.
    """
    features = list(feature_ids)
    if not features or len(set(features)) != len(features) or any(type(f) is not int or not 0 <= f < sae.cfg.d_sae for f in features):
        raise ValueError('feature_ids must be distinct integers in [0, d_sae)')
    if top_k <= 0 or batch_size <= 0 or (max_tokens is not None and max_tokens <= 0):
        raise ValueError('top_k, batch_size and max_tokens must be positive')
    if not math.isfinite(min_activation) or min_activation < 0:
        raise ValueError('min_activation must be finite and nonnegative')
    if store.d_in != sae.cfg.d_in:
        raise ValueError('Activation input dimension differs from SAE')
    sae.eval()
    heaps = {f: [] for f in features}
    counts = {f: 0 for f in features}
    seen = 0
    limit = min(store.n_tokens('validation'), max_tokens) if max_tokens else store.n_tokens('validation')
    columns = torch.tensor(features, device=sae.W_dec.device)
    for shard_index, entry in enumerate(store.manifest['splits']['validation']['shards']):
        if seen >= limit:
            break
        records = _metadata(store, entry)
        starts = [row['row_start'] for row in records]
        array = store.get_shard('validation', shard_index)
        for start in range(0, len(array), batch_size):
            count = min(batch_size, len(array) - start, limit - seen)
            if count <= 0:
                break
            x = torch.from_numpy(np.array(array[start:start + count], copy=True)).to(sae.W_dec.device)
            _, z = reconstruct(sae, normalize(x, normalizer))
            selected = z.index_select(-1, columns)
            if not torch.isfinite(selected).all():
                raise ValueError('Nonfinite feature activation')
            active_counts = (selected > min_activation).sum(0).cpu().tolist()
            # A stable sort keeps row order for tied scores, including at top-k boundaries.
            indices = selected.argsort(dim=0, descending=True, stable=True)[:min(top_k, count)]
            values = selected.gather(0, indices).cpu().tolist()
            offsets = indices.cpu().tolist()
            for j, feature in enumerate(features):
                counts[feature] += active_counts[j]
                for value_row, offset_row in zip(values, offsets):
                    activation, offset = value_row[j], offset_row[j]
                    if activation <= min_activation:
                        continue
                    shard_row = start + offset
                    global_row = seen + offset
                    record = records[bisect.bisect_right(starts, shard_row) - 1]
                    local = shard_row - record['row_start']
                    example = {
                        'activation': activation, 'global_row_index': global_row,
                        'shard_path': entry['path'], 'shard_row_index': shard_row,
                        'document_id': record['id'], 'group_id': record['group_id'],
                        'source': record['source'], 'task': record['task'],
                        'selection_index': record['selection_index'],
                        'token_position': record['positions'][local], 'token_id': record['token_ids'][local],
                    }
                    candidate = (activation, -global_row, example)
                    heap = heaps[feature]
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    elif candidate[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, candidate)
            seen += count
    return {
        'schema_version': 1, 'inspection': 'top_feature_token_contexts', 'split': 'validation',
        'cache_dir': str(store.root), 'cache_fingerprint': store.fingerprint,
        'tokens_evaluated': seen, 'available_validation_tokens': store.n_tokens('validation'),
        'max_tokens': max_tokens, 'top_k': top_k, 'min_activation_exclusive': min_activation,
        'ranking': 'descending activation, then ascending global cache row for ties',
        'features': {str(f): {'count_above_threshold': counts[f],
                              'frequency_above_threshold': counts[f] / seen if seen else 0.,
                              'examples': [x[2] for x in sorted(heaps[f], key=lambda x: (-x[0], -x[1]))]}
                     for f in features},
        'interpretation': 'Examples support manual feature inspection; they do not assign semantics or establish hallucination detection quality.',
    }


def attach_contexts(report: dict, corpus_root, selection_path, tokenizer, *, window_chars: int = 160) -> None:
    """Join only winning documents and verify frozen token identities in place."""
    if window_chars < 0:
        raise ValueError('window_chars must be nonnegative')
    requested = {}
    for feature in report['features'].values():
        for example in feature['examples']:
            requested.setdefault(example['selection_index'], []).append(example)
    if not requested:
        return
    references = read_selection(corpus_root, selection_path)
    ordered = sorted(requested)
    for index in ordered:
        if not 0 <= index < len(references):
            raise ValueError('Token metadata points outside the frozen selection')
        if any(hit['document_id'] != references[index]['id'] for hit in requested[index]):
            raise ValueError('Token metadata document differs from frozen selection')
    # A temporary subset lets the corpus reader scan each source file once and
    # load only winning documents, even when selected indices are far apart.
    with tempfile.TemporaryDirectory(prefix='sae-feature-contexts-') as tmp:
        subset = Path(tmp) / 'selection.jsonl'
        subset.write_text(''.join(json.dumps(references[i]) + '\n' for i in ordered))
        for index, document in zip(ordered, iter_selected_documents(corpus_root, subset)):
            ids, eligible = encode_document(document, tokenizer)
            eligible = set(eligible.tolist())
            text = tokenizer.apply_chat_template(document['messages'], tokenize=False, add_generation_prompt=False)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            for example in requested[index]:
                for key in ('group_id', 'source', 'task'):
                    if document[key] != example[key]:
                        raise ValueError(f'Cache token provenance differs for {key}')
                position = example['token_position']
                if position not in eligible or ids[position] != example['token_id']:
                    raise ValueError('Cached token identity/position differs from frozen document')
                a, b = encoded['offset_mapping'][position]
                left, right = max(0, a - window_chars), min(len(text), b + window_chars)
                example.update(token_text=text[a:b], token_char_span=[a, b],
                               context=text[left:right], context_char_span=[left, right],
                               token_span_in_context=[a - left, b - left],
                               corpus_path=references[index]['path'],
                               corpus_line_index=references[index]['line_index'],
                               rendered_sha256=document['tokenization']['rendered_sha256'])
    report['context_window_chars_each_side'] = window_chars


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=_project_path, required=True)
    parser.add_argument('--features', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=_project_path, required=True)
    parser.add_argument('--cache-dir', type=_project_path)
    parser.add_argument('--tokenizer', type=_project_path, help='Defaults to the extraction manifest model_path; loads tokenizer files only')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--min-activation', type=float, default=0.)
    parser.add_argument('--window-chars', type=int, default=160)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpus', default='4,5,6,7')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.device.startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpus):
            raise ValueError('CUDA_VISIBLE_DEVICES conflicts with --gpus')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    from eval_sae import load_checkpoint, write_json
    from corpus import LocalTokenizer
    sae, normalizer, payload = load_checkpoint(args.checkpoint, device=args.device, require_receipt=True)
    store = ActivationStore(args.cache_dir or payload['config']['cache_dir'])
    if store.fingerprint != payload['cache_fingerprint']:
        raise ValueError('Inspection cache fingerprint differs from checkpoint')
    corpus_root = Path(store.manifest['corpus_root'])
    manifest_path = corpus_root / 'manifest.json'
    if sha256_file(manifest_path) != store.manifest['corpus_manifest_sha256']:
        raise ValueError('Corpus manifest differs from extraction manifest')
    selection = Path(store.manifest['splits']['validation']['selection_path'])
    if not selection.is_absolute():
        selection = corpus_root / selection
    if sha256_file(selection) != store.manifest['splits']['validation']['selection_sha256']:
        raise ValueError('Frozen validation selection checksum differs')
    model_path = args.tokenizer or Path(store.manifest['model_path'])
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if sha256_file(model_path / name) != store.manifest['model_files'][name]:
            raise ValueError(f'Tokenizer fingerprint differs: {name}')
    report = collect_feature_examples(sae, store, normalizer, args.features, top_k=args.top_k,
                                      batch_size=args.batch_size, max_tokens=args.max_tokens,
                                      min_activation=args.min_activation)
    references = read_selection(corpus_root, selection)
    paths = {references[e['selection_index']]['path'] for f in report['features'].values() for e in f['examples']}
    expected_hashes = {entry['path']: entry['sha256'] for entry in json.loads(manifest_path.read_text())['outputs']}
    for name in paths:
        if sha256_file(corpus_root / name) != expected_hashes.get(name):
            raise ValueError(f'Winning context corpus file changed: {name}')
    attach_contexts(report, corpus_root, selection, LocalTokenizer(model_path), window_chars=args.window_chars)
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
                  checkpoint_step=payload.get('step'), architecture=payload['config']['sae']['architecture'],
                  corpus_root=str(corpus_root), selection_path=str(selection),
                  selection_sha256=sha256_file(selection), tokenizer_path=str(model_path))
    write_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'features': args.features, 'tokens_evaluated': report['tokens_evaluated']}))


if __name__ == '__main__':
    main()
