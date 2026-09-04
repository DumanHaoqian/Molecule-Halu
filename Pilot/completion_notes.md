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

## Task 2 — matched H/N pairs

- `emit_matched_negative=True` is the default and is defined in the single generation
  configuration file.
- H and N share `pair_id`; N has `hallucination_present=false`, empty
  `hallucination_spans`, and one truth-valued `control_span` per H span.
- Every step reports `byte_identical` or `regenerated`. Direct truth substitution is
  accepted only after reference-FORMAL, residual-value, and arithmetic checks.
- On all 150 origins with the deterministic renderer (`variant_index=0`), alignment was
  800/800 `byte_identical`, 0/800 `regenerated`. Character lengths matched for 919/1300
  paired spans (70.69%).
- The bounded fake-transport smoke used the first three IDs:
  `mol_edit.add_v2.0003`, `mol_edit.add_v2.0012`, and `mol_edit.add_v2.0013`; it emitted
  3 H + 3 N records and 15/15 byte-identical steps.
- A real `--max-origins 3` Poe smoke was not run because `POE_API_KEY` was absent from the
  execution environment. No key or authorization material was read, printed, or stored.

## Validation

- `conda run -n molhallulens python -m pytest -q` — 52 passed.
- `conda run -n molhallulens python -m compileall -q molhallulens` — passed.
- `conda run -n molhallulens python -m molhallulens.generate_dataset --help` — passed;
  `--max-origins` and `--failure-manifest` are exposed.

## Task 3 — enumeration consistency

- `scripts/survey_enumerations.py` scanned all 150 origins and identified 126 explicit,
  mechanically checkable component-total sentences: 24 item-first `totaling`/`giving a
  total` forms, 85 total-first parenthetical forms, 12 equals forms, and 5 unit-first
  parenthetical forms. The corpus uses both digits and English number words, nested group
  subtotals, and explicit fused-system overrides such as `counts as 2 rings`.
- The validator sums only totals stated in `heavy atom(s)` or `ring(s)`. Ambiguous fused
  `core`/`system` descriptions without an explicit contribution are skipped rather than
  guessed. The raw-corpus audit produced 0 false positives across all 800 natural-language
  steps.
- Enumeration-bearing changed count claims are routed to `derivation_rewrite`; the
  variant-0 mode distribution changed from 442 copy / 166 occurrence patch / 192
  derivation rewrite to 442 / 158 / 200.
- All 150 variant-0 outputs from `DeterministicTextRenderer` passed the enumeration audit.

## Task 4 — character-level molecular spans

Measured on all 150 default variant-0 origins, except the bond-order row, which uses the
same 150 origins with `product` as the only root and the bond-order operator enabled.

| Operator / propagation rule | Whole-value mean before | Character span mean after |
|---|---:|---:|
| `smiles_atom_replacement` | 70.22 | 3.72 |
| `smiles_bond_order_change` | 71.85 | 2.96 |
| `smiles_terminal_atom_deletion` | 65.11 | 1.00 |
| `final_answer_to_product` | 69.87 | 4.38 |
| `product_to_final_answer` | 69.25 | 1.54 |

- Each molecular occurrence has one contiguous H interval and one contiguous N interval;
  pure insertions/deletions expand across one common neighbor so both are non-empty.
- `context_span` and `serialized_context_span` retain the complete SMILES location, while
  `diff_opcodes` records every non-equal SequenceMatcher opcode.
- The post-change 150-origin pair audit remained 800/800 `byte_identical`; 923/1300 paired
  spans (71.00%) now have equal character length.
- Canonical SMILES can change its traversal after a one-atom edit. The renderer therefore
  deterministically chooses an equivalent rooted candidate traversal that minimizes the
  paired interval. This is display-only: planner and injection state are unchanged.

## Task 5 — batch robustness and Poe smoke status

- Generation now flushes each complete H/N pair incrementally. An injected interruption
  test confirms the first pair remains readable when the second origin is interrupted.
- Per-origin/variant exceptions are written immediately to a separate failure JSONL with
  stage and error type. The batch continues, `GenerationSummary` reports success/failure
  counts and the full failure list, and the CLI exits with status 1 after completion when
  any failure occurred.
- Stale renderer/bot/prompt/hash metadata and cached responses rejected by the current
  validator are cache misses and are refreshed. Malformed/unreadable cache JSON remains a
  hard error.
- Poe telemetry now reports logical calls, cache hits, network attempts, retry count,
  requests requiring retry, and rejection counts split into `false_enumeration`,
  `false_arithmetic`, and other step-text contract errors. Fake-transport tests verify that
  both new logical checks reject attempt 1 and recover on attempt 2.
- **Real Poe smoke: blocked.** `POE_API_KEY` was absent from this execution environment, so
  no real `--max-origins 3` request was attempted and no real regenerated/rewrite/retry
  percentages are claimed. Required condition: export `POE_API_KEY` in the running shell,
  then run `python -m molhallulens.generate_dataset --max-origins 3 --output
  GeneratedDataset/smoke.jsonl`.
- `POE_MAX_ATTEMPTS` remains 2. Recommendation: keep 2 provisionally—the deterministic and
  injected-rejection tests recover with one retry—but do not treat this as production
  evidence. Re-evaluate only from the real smoke's `poe_retry_rate`, terminal failure
  manifest, and rejection distribution; increase the limit only if valid responses
  repeatedly require more than one correction.
