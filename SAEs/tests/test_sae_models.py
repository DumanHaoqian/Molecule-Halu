"""CPU-only behavioral tests for the shared SAE training/inference API."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SAEModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("sae_models"), "SAE adapter module is missing")
        import sae_models
        return sae_models

    def config(self, architecture):
        cfg = {"architecture": architecture, "d_in": 8, "d_sae": 24, "k": 3}
        if architecture == "jumprelu":
            cfg.update(l0_coefficient=0.001, l0_warm_up_steps=4)
        if architecture == "sparsemax_attention":
            cfg.update(key_dim=6, preselect_k=8, l0_coefficient=0.0)
        return cfg

    def test_each_architecture_has_finite_training_gradients(self):
        api = self.api()
        for arch in api.ARCHITECTURES:
            with self.subTest(architecture=arch):
                sae = api.build_sae(self.config(arch), "cpu")
                x = torch.randn(12, 8)
                dead_mask = torch.arange(24) % 3 == 0
                output = api.training_step(sae, x, step=2, dead_mask=dead_mask)
                self.assertEqual(output.sae_out.shape, x.shape)
                self.assertEqual(output.feature_acts.shape, (12, 24))
                self.assertTrue(torch.isfinite(output.loss))
                self.assertIn("mse_loss", output.losses)
                output.loss.backward()
                grads = [p.grad for p in sae.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_topk_training_sparsity_is_per_row_and_batchtopk_is_global(self):
        api = self.api()
        x = torch.randn(12, 8)
        for arch in ("topk", "batchtopk", "matryoshka"):
            with self.subTest(architecture=arch):
                sae = api.build_sae(self.config(arch), "cpu")
                acts = api.training_step(sae, x).feature_acts
                if arch == "topk":
                    self.assertTrue(((acts > 0).sum(-1) <= 3).all())
                else:
                    self.assertLessEqual((acts > 0).sum().item(), 12 * 3)

    def test_eval_batchtopk_matches_exported_jumprelu_and_is_batch_independent(self):
        api = self.api()
        from sae_lens import SAE
        for arch in ("batchtopk", "matryoshka"):
            with self.subTest(architecture=arch):
                sae = api.build_sae(self.config(arch), "cpu")
                api.training_step(sae, torch.randn(24, 8))
                sae.eval()
                x = torch.randn(7, 8)
                threshold_before = sae.topk_threshold.clone()
                recon, acts = api.reconstruct(sae, x)
                single = torch.cat([api.reconstruct(sae, row[None])[0] for row in x])
                torch.testing.assert_close(recon, single)
                torch.testing.assert_close(threshold_before, sae.topk_threshold)
                with tempfile.TemporaryDirectory() as tmp:
                    sae.save_inference_model(tmp)
                    exported = SAE.load_from_disk(tmp, device="cpu")
                    torch.testing.assert_close(recon, exported(x))
                    torch.testing.assert_close(acts, exported.encode(x))

    def test_untrained_batchtopk_inference_is_sparse_and_batch_independent(self):
        api = self.api()
        sae = api.build_sae(self.config("batchtopk"), "cpu").eval()
        x = torch.randn(7, 8)
        _, acts = api.reconstruct(sae, x)
        single = torch.cat([api.reconstruct(sae, row[None])[1] for row in x])
        torch.testing.assert_close(acts, single)
        self.assertTrue(((acts > 0).sum(-1) <= 3).all())

    def test_matryoshka_prefix_reconstruction_uses_decoder_bias_and_scaling(self):
        api = self.api()
        sae = api.build_sae(self.config("matryoshka"), "cpu").eval()
        with torch.no_grad():
            sae.b_dec.fill_(0.3)
        x = torch.randn(4, 8)
        _, full = api.reconstruct(sae, x)
        partial_recon, partial = api.reconstruct(sae, x, width=6)
        expected_acts = full.clone()
        expected_acts[:, 6:] = 0
        torch.testing.assert_close(partial, expected_acts)
        torch.testing.assert_close(partial_recon, sae.decode(expected_acts))

    def test_jumprelu_coefficient_warmup_and_post_step_threshold_constraint(self):
        api = self.api()
        cfg = self.config("jumprelu")
        cfg.update(l0_coefficient=2.0, l0_warm_up_steps=4)
        sae = api.build_sae(cfg, "cpu")
        x = torch.randn(10, 8)
        first = api.training_step(sae, x, step=0)
        last = api.training_step(sae, x, step=3)
        torch.testing.assert_close(last.losses["l0_loss"], 4 * first.losses["l0_loss"])
        with torch.no_grad():
            sae.threshold.fill_(-1)
        api.post_optimizer_step(sae)
        self.assertTrue((sae.threshold >= 0).all())

    def test_each_architecture_can_learn_small_synthetic_cpu_problem(self):
        api = self.api()
        x = torch.randn(24, 8) * 0.25 + torch.tensor([1., 0., -1., 0., 1., 0., -1., 0.])
        for arch in api.ARCHITECTURES:
            with self.subTest(architecture=arch):
                torch.manual_seed(7)
                sae = api.build_sae(self.config(arch), "cpu")
                optimizer = torch.optim.Adam(sae.parameters(), lr=0.02)
                initial = api.training_step(sae, x).loss.item()
                for step in range(45):
                    optimizer.zero_grad(set_to_none=True)
                    out = api.training_step(sae, x, step=step)
                    out.loss.backward()
                    optimizer.step()
                    api.post_optimizer_step(sae)
                self.assertLess(api.training_step(sae, x, step=45).loss.item(), initial * 0.65)

    def test_each_architecture_checkpoint_roundtrip(self):
        api = self.api()
        for arch in api.ARCHITECTURES:
            with self.subTest(architecture=arch):
                cfg = self.config(arch)
                sae = api.build_sae(cfg, "cpu")
                api.training_step(sae, torch.randn(12, 8))
                sae.eval()
                x = torch.randn(5, 8)
                before = api.reconstruct(sae, x)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.pt"
                    api.save_sae(sae, path, cfg)
                    loaded = api.load_sae(path, "cpu")
                    after = api.reconstruct(loaded, x)
                for left, right in zip(before, after):
                    torch.testing.assert_close(left, right)
                self.assertFalse(loaded.training)

    def test_invalid_configuration_is_rejected(self):
        api = self.api()
        invalid = [
            {"architecture": "unknown", "d_in": 8, "d_sae": 24, "k": 3},
            {**self.config("topk"), "k": 25},
            {**self.config("topk"), "k": 2.5},
            {**self.config("batchtopk"), "k": 0},
            {**self.config("topk"), "misspelled_option": True},
            {**self.config("topk"), "normalize_activations": "layer_norm"},
            {**self.config("matryoshka"), "matryoshka_widths": [12, 6, 24]},
        ]
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises((ValueError, TypeError)):
                api.build_sae(cfg, "cpu")

    def test_resolve_config_validates_large_model_without_allocating_parameters(self):
        api = self.api()
        self.assertTrue(hasattr(api, "resolve_model_config"))
        from unittest.mock import patch
        import sae_lens
        with patch.object(sae_lens, "BatchTopKTrainingSAE", side_effect=AssertionError("allocated model")):
            resolved = api.resolve_model_config({"architecture": "batchtopk", "d_in": 5120, "d_sae": 81920, "k": 32})
        self.assertEqual(resolved["architecture"], "batchtopk")
        self.assertEqual(resolved["d_sae"], 81920)
        self.assertTrue(resolved["rescale_acts_by_decoder_norm"])
        with self.assertRaises(ValueError):
            api.resolve_model_config({**self.config("matryoshka"), "matryoshka_widths": [12, 4]})

    def test_resolve_config_rejects_invalid_architecture_hyperparameters(self):
        api = self.api()
        invalid = [
            {**self.config("topk"), "decoder_init_norm": 0},
            {**self.config("topk"), "aux_loss_coefficient": -1},
            {**self.config("batchtopk"), "topk_threshold_lr": 1.1},
            {**self.config("jumprelu"), "jumprelu_bandwidth": 0},
            {**self.config("jumprelu"), "jumprelu_sparsity_loss_mode": "bad"},
            {**self.config("jumprelu"), "jumprelu_sparsity_loss_mode": "quadratic", "target_l0": 25},
            {**self.config("sparsemax_attention"), "key_dim": 0},
            {**self.config("sparsemax_attention"), "preselect_k": -2},
        ]
        for cfg in invalid:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                api.resolve_model_config(cfg)

    def test_sparsemax_is_sparse_simplex_and_safe_when_all_features_are_idf_masked(self):
        self.api()
        from sparsemax_attention_sae import SparsemaxAttentionSAE
        scores = torch.tensor([[3., 0., -2.], [0., 0., 0.]], requires_grad=True)
        probs = SparsemaxAttentionSAE.sparsemax(scores)
        torch.testing.assert_close(probs.sum(-1), torch.ones(2))
        self.assertTrue((probs >= 0).all())
        torch.testing.assert_close(probs[0], torch.tensor([1., 0., 0.]))
        probs.square().sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        cfg = {**self.config("sparsemax_attention"), "use_idf_mask": True, "use_input_norm": False}
        sae = self.api().build_sae(cfg, "cpu").eval()
        sae.idf_score.fill_(1)
        before_calls = sae.num_encode_calls.clone()
        recon, acts = self.api().reconstruct(sae, torch.randn(4, 8))
        self.assertTrue(torch.isfinite(recon).all())
        torch.testing.assert_close(acts.sum(-1), torch.ones(4))
        torch.testing.assert_close(sae.num_encode_calls, before_calls)


if __name__ == "__main__":
    unittest.main()
