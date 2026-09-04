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

- `conda run -n molhallulens python -m pytest -q` — 32 passed.
- `conda run -n molhallulens python -m compileall -q molhallulens` — passed.
- `conda run -n molhallulens python -m molhallulens.generate_dataset --help` — passed;
  `--max-origins` is exposed.
