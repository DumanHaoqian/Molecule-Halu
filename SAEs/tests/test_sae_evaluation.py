"""CPU tests for global SAE metrics and causally aligned LM interventions."""
import math
import sys
import tempfile
import json
import hashlib

import numpy as np
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReconstructionMetricsTest(unittest.TestCase):
    def test_global_statistics_are_invariant_to_uneven_batches(self):
        from metrics import ReconstructionMetrics
        x = torch.tensor([[0., 1.], [1., 2.], [10., 8.], [20., 1.]])
        recon = x + torch.tensor([[1., 1.], [2., 0.], [3., 1.], [0., 2.]])
        z = torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 0., 0.]])
        full, batched = ReconstructionMetrics(), ReconstructionMetrics()
        full.update(x, recon, z)
        batched.update(x[:1], recon[:1], z[:1])
        batched.update(x[1:], recon[1:], z[1:])
        a, b = full.compute(), batched.compute()
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=10, msg=key)
        denominator = ((x - x.mean(0)) ** 2).sum()
        self.assertAlmostEqual(a['mse'], ((x - recon) ** 2).mean().item())
        self.assertAlmostEqual(a['nmse'], ((x - recon) ** 2).sum().item() / denominator.item())
        expected_ev = 1 - ((x - recon).var(0, unbiased=False).sum() / x.var(0, unbiased=False).sum()).item()
        self.assertAlmostEqual(a['explained_variance'], expected_ev, places=6)
        self.assertAlmostEqual(a['dead_feature_fraction'], 1 / 3)
        self.assertAlmostEqual(a['l0_mean'], 1.)
        self.assertEqual(a['l0_p50'], 1.)
        torch.testing.assert_close(batched.feature_frequencies, torch.tensor([.5, .5, 0.], dtype=torch.float64))

    def test_zero_vectors_and_single_token_are_finite(self):
        from metrics import ReconstructionMetrics
        metrics = ReconstructionMetrics()
        metrics.update(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 4))
        self.assertTrue(all(math.isfinite(value) for value in metrics.compute().values()))
        self.assertEqual(metrics.compute()['variance_is_zero'], 1.)
        self.assertEqual(metrics.compute()['mse'], 0.)
        with self.assertRaises(ValueError):
            ReconstructionMetrics().compute()


class HalfSAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('W_dec', torch.eye(2))
        self.cfg = SimpleNamespace(d_in=2, d_sae=2)
    def encode(self, x):
        return x
    def decode(self, z):
        return z * .5


class TupleBlock(torch.nn.Module):
    def forward(self, x):
        return x + .5, 'preserved-cache'


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [2., 1.], [1., 3.], [3., 2.]]))
        self.block = TupleBlock()
        self.head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]]))
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h, cache = self.block(self.embedding(input_ids))
        assert cache == 'preserved-cache'
        return SimpleNamespace(logits=self.head(h))


class DeltaLMTest(unittest.TestCase):
    def test_assistant_target_mask_is_causally_shifted(self):
        from delta_lm import masked_next_token_loss
        ids = torch.tensor([[0, 1, 2, 3]])
        logits = torch.tensor([[[2., 3., 1., 0.], [5., 1., 2., 3.], [1., 0., 3., 2.], [1., 2., 3., 4.]]])
        total, count = masked_next_token_loss(logits, ids, [1, 3])
        expected = F.cross_entropy(logits[0, [0, 2]], ids[0, [1, 3]], reduction='sum')
        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, expected.item())
        self.assertEqual(masked_next_token_loss(logits, ids, [0]), (0., 0))

    def test_tuple_hook_normalizes_inverts_and_patches_only_requested_positions(self):
        from delta_lm import reconstruction_hook
        original = torch.tensor([[[2., 4.], [4., 8.], [6., 12.]]])
        normalizer = {'mean': torch.tensor([2., 4.]), 'scale': 2.}
        hook = reconstruction_hook(HalfSAE(), normalizer, positions=[1])
        actual, marker = hook(None, (), (original, 'keep'))
        self.assertEqual(marker, 'keep')
        torch.testing.assert_close(actual[:, 0], original[:, 0])
        torch.testing.assert_close(actual[:, 2], original[:, 2])
        torch.testing.assert_close(actual[:, 1], torch.tensor([[3., 6.]]))
        torch.testing.assert_close(original[:, 1], torch.tensor([[4., 8.]]))

    def test_batchtopk_hook_uses_same_threshold_inference_as_cache_evaluation(self):
        from delta_lm import reconstruction_hook
        from sae_models import build_sae, reconstruct
        sae = build_sae({'architecture': 'batchtopk', 'd_in': 2, 'd_sae': 4, 'k': 1})
        sae.eval()
        with torch.no_grad():
            sae.W_enc.fill_(1.)
            sae.b_enc.zero_()
            sae.b_dec.zero_()
            sae.topk_threshold.fill_(.1)
        hidden = torch.tensor([[[1., 2.], [3., 4.], [2., 3.]]])
        normalizer = {'mean': torch.zeros(2), 'scale': 1.}
        expected, _ = reconstruct(sae, hidden[0])
        result = reconstruction_hook(sae, normalizer)(None, (), hidden)
        torch.testing.assert_close(result[0], expected)

    def test_lm_evaluation_is_token_weighted_and_removes_hooks(self):
        from delta_lm import evaluate_lm, masked_next_token_loss
        model = TinyLM()
        docs = [
            {'id': 'a', 'input_ids': [0, 1, 2, 3], 'assistant_positions': [1, 2, 3]},
            {'id': 'b', 'input_ids': [3, 2, 1], 'assistant_positions': [2]},
        ]
        normalizer = {'mean': torch.tensor([1., 2.]), 'scale': 2.}
        result = evaluate_lm(model, model.block, HalfSAE(), normalizer, docs, patch_scope='assistant')
        losses = [masked_next_token_loss(model(torch.tensor([d['input_ids']])).logits,
                    torch.tensor([d['input_ids']]), d['assistant_positions']) for d in docs]
        self.assertEqual(result['n_scored_tokens'], 4)
        self.assertAlmostEqual(result['clean_ce'], sum(v[0] for v in losses) / 4)
        self.assertAlmostEqual(result['delta_ce'], result['reconstruction_ce'] - result['clean_ce'])
        self.assertEqual(len(model.block._forward_hooks), 0)
        self.assertTrue(math.isfinite(result['mean_ablation_ce']))
        self.assertEqual(result['n_documents'], 2)


    def test_mean_ablation_uses_data_mean_when_centering_is_disabled(self):
        from delta_lm import reconstruction_hook
        normalizer = {'mean': torch.zeros(2), 'data_mean': torch.tensor([2., 5.]), 'scale': 2.}
        result = reconstruction_hook(HalfSAE(), normalizer, ablation='mean')(None, (), torch.ones(1, 3, 2))
        torch.testing.assert_close(result, torch.tensor([[[2., 5.], [2., 5.], [2., 5.]]]))


class HeldoutEvaluationTest(unittest.TestCase):
    def test_checkpoint_evaluation_uses_only_validation_and_saved_normalizer(self):
        from eval_sae import evaluate_checkpoint
        from sae_models import build_sae
        from activation_store import ActivationStore
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {'format_version': 1, 'status': 'complete', 'layer': 26,
                        'block_index': 25, 'activation_site': 'post_block_residual',
                        'd_in': 2, 'dtype': 'float16', 'splits': {}}
            for split, values in [('train', [[99., 99.]]), ('validation', [[1., 2.], [3., 4.], [7., 2.]])]:
                shard = root / (split + '.npy')
                np.save(shard, np.asarray(values, dtype=np.float16))
                manifest['splits'][split] = {'n_tokens': len(values), 'shards': [{
                    'path': shard.name, 'n_rows': len(values), 'sha256': hashlib.sha256(shard.read_bytes()).hexdigest()}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            store = ActivationStore(root)
            cfg = {'architecture': 'topk', 'd_in': 2, 'd_sae': 4, 'k': 2}
            sae = build_sae(cfg)
            checkpoint = root / 'test.pt'
            normalizer = {'mean': torch.tensor([1., 2.]), 'scale': 3., 'center': True, 'n_tokens': 1}
            torch.save({'format_version': 1, 'config': {'sae': cfg, 'cache_dir': str(root)},
                        'sae_state_dict': sae.state_dict(), 'normalizer': normalizer,
                        'cache_fingerprint': store.fingerprint, 'step': 2}, checkpoint)
            from eval_sae import load_checkpoint
            with self.assertRaisesRegex(ValueError, 'receipt'):
                load_checkpoint(checkpoint, require_receipt=True)
            report = evaluate_checkpoint(checkpoint, batch_size=2)
            self.assertEqual(report['metrics']['n_tokens'], 3)
            self.assertEqual(report['evaluation_split'], 'validation')
            self.assertEqual(report['normalization_fit_split'], 'train')
            self.assertEqual(report['normalizer']['mean'], [1., 2.])
            self.assertAlmostEqual(report['metrics']['mse'] / 9, report['normalized_metrics']['mse'], places=5)
            self.assertEqual(len(report['feature_frequencies']), 4)
            payload = torch.load(checkpoint, weights_only=False)
            payload['cache_fingerprint'] = 'changed'
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                evaluate_checkpoint(checkpoint, batch_size=2)


    def test_delta_provenance_rejects_changed_validation_text(self):
        from delta_lm import validate_lm_provenance
        from eval_sae import sha256_file
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text('{}')
            weights = root / 'model.safetensors'
            weights.write_bytes(b'toy')
            selection = root / 'selection.jsonl'
            selection.write_text(json.dumps({'id': 'a', 'path': 'validation.jsonl', 'line_index': 0}) + '\n')
            data = root / 'validation.jsonl'
            data.write_text('original')
            (root / 'manifest.json').write_text(json.dumps({'outputs': [{'path': data.name, 'sha256': sha256_file(data)}]}))
            manifest = {'layer': 26, 'block_index': 25, 'd_in': 5120, 'activation_site': 'post_block_residual',
                        'model_files': {'config.json': sha256_file(root / 'config.json')},
                        'model_weights_identity': [{'path': weights.name, 'bytes': weights.stat().st_size,
                                                    'mtime_ns': weights.stat().st_mtime_ns}],
                        'corpus_manifest_sha256': sha256_file(root / 'manifest.json'),
                        'splits': {'validation': {'selection_sha256': sha256_file(selection)}}}
            checkpoint = {'cache_manifest': manifest}
            validate_lm_provenance(checkpoint, root, root, selection)
            data.write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'shard'):
                validate_lm_provenance(checkpoint, root, root, selection)


if __name__ == '__main__':
    unittest.main()
