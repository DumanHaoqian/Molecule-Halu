"""Token provenance and streaming top-feature inspection on a tiny CPU cache."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CharTokenizer:
    all_special_ids = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ''.join(f"[{m['role']}]{m['content']}" for m in messages) + ('[assistant]' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class ToySAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('W_dec', torch.eye(2))
        self.cfg = SimpleNamespace(d_in=2, d_sae=2)

    def encode(self, x):
        return x.relu()

    def decode(self, z):
        return z


class FeatureExamplesTest(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('feature_examples'), 'Feature inspection module is missing')
        import feature_examples
        return feature_examples

    def fixture(self, root):
        from activation_store import ActivationStore, sha256_file
        from activation_inputs import encode_document
        from corpus import render_and_count
        root = Path(root)
        corpus = root / 'corpus'
        corpus.mkdir()
        tokenizer = CharTokenizer()
        docs, refs, metadata = [], [], []
        for i, text in enumerate(('AB', 'CD')):
            row = {'id': f'doc{i}', 'group_id': f'group{i}', 'source': 'toy', 'task': 'inspect',
                   'split': 'validation', 'messages': [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': text}]}
            row['tokenization'] = render_and_count(row['messages'], tokenizer)
            ids, positions = encode_document(row, tokenizer)
            docs.append(row)
            refs.append({k: row[k] for k in ('id', 'group_id', 'source', 'split')})
            refs[-1].update(path='docs.jsonl', line_index=i, **{k: row['tokenization'][k] for k in ('assistant_tokens', 'sequence_tokens')})
            metadata.append({k: row[k] for k in ('id', 'group_id', 'source', 'task')})
            metadata[-1].update(row_start=2 * i, row_end=2 * i + 2, selection_index=i,
                                positions=positions.tolist(), token_ids=[ids[p] for p in positions])
        (corpus / 'docs.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in docs))
        selection = corpus / 'selection.jsonl'
        selection.write_text(''.join(json.dumps(x) + '\n' for x in refs))
        cache = root / 'cache'
        cache.mkdir()
        manifest = {'format_version': 1, 'status': 'complete', 'layer': 26, 'block_index': 25,
                    'activation_site': 'post_block_residual', 'd_in': 2, 'dtype': 'float16',
                    'corpus_root': str(corpus), 'splits': {}}
        for split, values in [('train', [[0., 0.]]), ('validation', [[1., 5.], [4., 0.], [2., 6.], [-2., -1.]])]:
            path = cache / f'{split}.npy'
            np.save(path, np.array(values, dtype=np.float16))
            shard = {'path': path.name, 'n_rows': len(values), 'sha256': sha256_file(path)}
            if split == 'validation':
                meta = cache / 'validation.jsonl'
                meta.write_text(''.join(json.dumps(x) + '\n' for x in metadata))
                shard.update(metadata_path=meta.name, metadata_sha256=sha256_file(meta))
            manifest['splits'][split] = {'n_tokens': len(values), 'shards': [shard]}
        manifest['splits']['validation'].update(selection_path='selection.jsonl', selection_sha256=sha256_file(selection))
        (cache / 'manifest.json').write_text(json.dumps(manifest))
        return ActivationStore(cache), corpus, tokenizer

    def test_streaming_top_rows_match_across_batch_sizes_and_keep_token_provenance(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            store, corpus, tokenizer = self.fixture(tmp)
            normalizer = {'mean': torch.zeros(2), 'scale': 1.}
            first = api.collect_feature_examples(ToySAE(), store, normalizer, [0, 1], top_k=2, batch_size=1)
            second = api.collect_feature_examples(ToySAE(), store, normalizer, [0, 1], top_k=2, batch_size=3)
            self.assertEqual(first['features'], second['features'])
            self.assertEqual(first['tokens_evaluated'], 4)
            self.assertEqual([x['activation'] for x in first['features']['0']['examples']], [4., 2.])
            self.assertEqual([x['global_row_index'] for x in first['features']['1']['examples']], [2, 0])
            api.attach_contexts(first, corpus, 'selection.jsonl', tokenizer, window_chars=12)
            hit = first['features']['0']['examples'][0]
            self.assertEqual(hit['document_id'], 'doc0')
            self.assertEqual(hit['token_text'], 'B')
            self.assertEqual(hit['token_id'], ord('B'))
            start, end = hit['token_span_in_context']
            self.assertEqual(hit['context'][start:end], 'B')

    def test_min_activation_and_fixed_token_limit_are_explicit(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            store, _, _ = self.fixture(tmp)
            result = api.collect_feature_examples(ToySAE(), store, {'mean': torch.zeros(2), 'scale': 1.},
                                                  [1], top_k=3, batch_size=2, max_tokens=2, min_activation=5.)
            self.assertEqual(result['tokens_evaluated'], 2)
            self.assertEqual(result['features']['1']['examples'], [])
            with self.assertRaises(ValueError):
                api.collect_feature_examples(ToySAE(), store, {'mean': torch.zeros(2), 'scale': 1.}, [2])

    def test_context_join_rejects_changed_token_identity(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            store, corpus, tokenizer = self.fixture(tmp)
            result = api.collect_feature_examples(ToySAE(), store, {'mean': torch.zeros(2), 'scale': 1.}, [0], top_k=1)
            result['features']['0']['examples'][0]['token_id'] = 999
            with self.assertRaisesRegex(ValueError, 'token'):
                api.attach_contexts(result, corpus, 'selection.jsonl', tokenizer)

    def test_equal_scores_prefer_earlier_rows_even_across_batches(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            store, _, _ = self.fixture(tmp)
            # Keep the same provenance while making three equally strong hits.
            path = store.root / 'validation.npy'
            np.save(path, np.array([[2., 0.], [2., 0.], [2., 0.], [0., 0.]], dtype=np.float16))
            for batch_size in (1, 2, 4):
                result = api.collect_feature_examples(ToySAE(), store, {'mean': torch.zeros(2), 'scale': 1.},
                                                      [0], top_k=2, batch_size=batch_size)
                self.assertEqual([e['global_row_index'] for e in result['features']['0']['examples']], [0, 1])


if __name__ == '__main__':
    unittest.main()
