# Completion notes

## Task 1 — per-occurrence derivation markers

Measured on all 150 origins with `variant_index=0` and the repository's current default
generation configuration.

| Metric | Before | After |
|---|---:|---:|
| `copy` steps | 479 | 442 |
| `occurrence_patch` steps | 164 | 166 |
| `derivation_rewrite` steps | 157 | 192 |
| Mean natural-language span length, occurrence patch | 2.09 characters | 1.89 characters |
| Mean natural-language span length, derivation rewrite | 73.83 characters | 13.10 characters |
| Records with a detected stale before-value in prose | 26 | 0 |

The wider all-node audit intentionally upgrades more steps to `derivation_rewrite`. Those
steps now emit one or more node-attributed value spans rather than a whole-body span.
