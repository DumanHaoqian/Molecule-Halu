# Layer 26 TopK SAE implementation plan

User asked to proceed toward SAE training after the verified corpus preparation. GPUs 4–7 remain the authorized devices. Existing third-party GPU processes must not be stopped. Current launch is resource-blocked: each authorized A6000 has only about 9 GiB free. CPU implementation and input preparation can proceed independently.

## First experiment

- Checkpoint: local ChemDFM-R-14B, block index 25 output, post-token, before final layer norm. Frozen BF16 model; FP16 activation storage. Use the verified v1 `pilot_10m` selection (10,000,474 train / 443,731 validation assistant tokens).
- TopK SAE: input 5,120; dictionary 16,384; k=64; nonnegative activations; encoder initialized from normalized decoder transpose. Decoder columns constrained to unit norm. Learn decoder bias and latent bias.
- Train-only global mean and scalar RMS normalization, saved with the checkpoint. No per-token normalization, labels, reference answers, or Pilot evaluation data are introduced.
- Adam, lr=3e-4, batch=4096, one pass over the selected tokens for the initial experiment, seed=20260921. Warmup 100 steps, gradient clipping=1.0. Optional AuxK residual loss for features inactive over 262,144 train tokens (k_aux=128, coefficient=1/32). No claim of convergence after one pass.
- Chunk/within-chunk seeded token shuffling. Report validation NMSE/FVE, observed L0, never-fired features and train firing counts. Save normalization, model/optimizer state, data/checkpoint fingerprints, progress and final metrics. Validate reconstruction as an SAE property; do not equate reconstruction improvement with hallucination detection.

## Work

1. CPU input packing: verify dataset completion and selected source-shard checksums, reproduce exact chat template/assistant mask, pack token IDs and eligible post-token positions with per-document offsets. Avoid loading model weights. Add small tests for masking and chunk/sample accounting.
2. Activation extractor: enforce allowed GPUs and adequate free memory before loading any model. Resume only matching caches; use one full frozen model per sufficiently free GPU. Capture block 26, compare first capture with full forward and a causal prefix. Preserve document/token lookup per activation row. Atomic chunk writes and completed-worker manifests.
3. SAE module and trainer: implement tested TopK, normalization, decoder constraints, AuxK and streaming batches. Test learning on synthetic CPU activations, validation accounting and checkpoint round trips. Do not train on prior Pilot activation caches.
4. Run resource guard. Start a small GPU end-to-end pilot only after a permitted GPU has sufficient free memory; then extract the full selected corpus and train/evaluate. If hardware is still occupied, leave concrete scripts/config/packed inputs ready and report the blocker without claiming training has started.

Method references: [Gao et al.](https://arxiv.org/html/2406.04093v1), [official reference model](https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py). This is a compact local implementation with documented normalization/budget choices, not an exact reproduction of that paper.
