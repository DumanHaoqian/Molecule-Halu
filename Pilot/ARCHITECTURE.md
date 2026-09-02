# MolHalluLens module architecture

The package has one supported data-flow direction:

```text
A ingestion
  -> B reference
  -> C error_planning
  -> D error_injection
  -> E trajectory
  -> F text_realization
  -> G annotation
  -> H release
```

## Module contracts

| Stage | Package | Owns | Produces |
|---|---|---|---|
| A | `modules.ingestion` | ChemCoTBench loading, joining, subtask normalization | normalized origin records |
| B | `modules.reference` | FORMAL parsing, reference DAG, executable edit truth | validated reference state |
| C | `modules.error_planning` | candidate sources, ranking, donor pools, recipe coverage | selected error plan/root patch |
| D | `modules.error_injection` | add/delete/substitute operators and typed edit actions | root-mutated state |
| E | `modules.trajectory` | dependency closure and downstream derivation | candidate state plus graph delta |
| F | `modules.text_realization` | FORMAL, natural trace, detector prompt rendering | rendered detector text and mentions |
| G | `modules.annotation` | character spans and tokenizer projection | aligned labels and masks |
| H | `modules.release` | split assembly, artifacts, tokenization, activations, QA | publishable dataset release |

`core` contains only shared immutable contracts. `config` owns configuration.
`infrastructure` contains chemistry, validation, and external provider services;
these are dependencies of the stages, not additional pipeline stages.
`orchestration.py` contains the cross-stage runtime ports and sealed execution
template. It coordinates stages but owns no chemistry or dataset policy.

## Dependency rules

1. A stage may depend on `core`, `config`, and `infrastructure`.
2. A stage may consume public output from an earlier stage.
3. A stage must not import a later stage.
4. `release` is the only stage allowed to write final dataset artifacts.
5. The top-level `pipeline.py` performs orchestration only; it contains no chemistry,
   mutation, rendering, or labeling logic.

## Source tree

```text
molhallulens/
├── pipeline.py
├── orchestration.py
├── core/
├── config/
├── modules/
│   ├── ingestion/
│   ├── reference/
│   ├── error_planning/
│   ├── error_injection/
│   ├── trajectory/
│   ├── text_realization/
│   ├── annotation/
│   └── release/
├── infrastructure/
│   ├── chemistry/
│   ├── validation/
│   └── providers/
└── cli/
```

The current release behavior remains frozen. This refactor changes ownership and
imports; it does not yet replace the legacy H/N bundle semantics or implement the
new cumulative hallucination trajectory model.
