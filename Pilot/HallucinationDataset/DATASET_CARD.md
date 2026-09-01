# MolHalluLens Molecule Editing `pilot_v1`

## Summary

This release is a counterfactual hallucination-detection pilot for molecule editing. It contains 150 origins and 1,200 detector records: 800 train, 200 validation, and 200 test. Every origin contributes four reciprocal H/N pairs—LOCAL, PARTIAL, FULL_CF, and TERMINAL—for exactly four hallucinated and four faithful records.

Detector-visible text is always serialized as indexed SMILES → instruction → reasoning → final answer. Hidden oracle, typed state graphs, real ChemDFM-R token labels, private provenance, and layer-26 activation artifacts are stored separately and joined only by record identity.

## Intended use and Risk 6 decision

These 150 ChemCoTBench-V2 origins are **pipeline/schema smoke-test material only**. They must not be used for formal detector training, detector-layer selection, or final threshold tuning. Split names describe construction isolation and do not grant training authorization. Statistical units are origins/leakage groups, not the 1,200 correlated records.

## Composition

- Origins: train/validation/test = 100/25/25.
- Records: train/validation/test = 800/200/200.
- Labels: 600 H and 600 matched N.
- Per split and per policy, H/N counts are exactly balanced.
- Accepted candidate source: deterministic RULE paths.
- Strict validation: 4,800 artifact gates plus 150 complete-bundle gates.

## Tokens and activations

All 1,200 records use one real local ChemDFM-R fast-tokenizer fingerprint. Every direct and multi-axis label array has the exact input-token length. Activations are ChemDFM-R layer-26 `resid_post` features with `post_token_h_t` alignment, zero label shift, 5,120 hidden dimensions, and 1824606 total token positions. No pre-token target is claimed.

## Shortcut and symbolic baselines

Recommended learned screens use train+validation only; held-out test is a one-time diagnostic that cannot feed candidate, layer, renderer, or threshold selection. Not all recommended engineering screens passed; the exact findings are retained below and in `KNOWN_LIMITATIONS.md`.

- `metadata_auroc`: 0.5 (<= 0.55; passed)
- `reasoning_only_shallow_auroc`: 0.614298 (<= 0.6; did not pass)
- `span_only_tfidf_auroc`: 0.5953 (<= 0.55; did not pass)
- `style_pair_matching`: 0 (== 0; passed)
- `token_length_standardized_difference`: 0.015264205236 (< 0.1; passed)

The report also includes nearest-neighbor retrieval, RDKit visible validity, visible reasoning/answer graph comparison, hidden-oracle graph comparison, per-policy symbolic slices, and the strict graph-edit verifier.

## Poe provenance

The frozen provider model is `gpt-5.4-mini`. Deterministic capability mocks passed. Live capability status is `offline_not_executed` because Poe exposes no selectable upstream snapshot. The final release made zero live Poe calls and contains no cached Poe response entries. Its zero-call usage ledger is stored outside the project on a filesystem that enforces owner-only mode; the repository contains only a secret-free export descriptor.

## Release identity

At the user's explicit instruction, T052 performs no cryptographic identity computation or verification. The original criterion is recorded as `overridden_not_evaluated`, never as passed. The effective release freeze is `pilot_v1` plus exact artifact paths, exact row counts, and exact record/origin identity sets.

See `KNOWN_LIMITATIONS.md` and `reports/release_validation_report.json` before use.
