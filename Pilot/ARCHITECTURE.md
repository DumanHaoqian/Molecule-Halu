# Current architecture

## Modules and contracts

| Stage | Code | Input | Output | Responsibility |
|---|---|---|---|---|
| A | `modules/ingestion/` | `Dataset/` JSON + manifest | `JoinedInputRecord` | Join and validate raw, process, and template data |
| B | `modules/reference/` | joined record | `ReferenceDAGArtifact` | Parse every `formal_ab` claim into a typed reference DAG |
| B | `modules/error_planning/fragment_pool.py` | all reference DAGs | `FragmentPool` | Build a deduplicated corpus-level functional-group pool |
| C | `modules/error_planning/unified.py` | DAG + config + pool | `UnifiedHallucinationPlan` | Select K points, operators, replacements, and magnitudes |
| D | `modules/error_injection/` | DAG + plan | `InjectedHallucination` | Apply root edits, propagate deterministic claims, and audit every edge |
| E1 | `modules/text_realization/renderer.py` | original `step_text` + candidate DAG | original complete steps + modified `formal_ab` + required HALLU occurrences | Inventory every changed natural-language mention |
| E2 | `modules/text_realization/poe_agent.py` | E1 request | validated, minimally rewritten complete `step_text` with temporary markers | Update only prose affected by modified FORMAL |
| E3 | `modules/text_realization/renderer.py` | marked Poe steps + candidate DAG | `RenderedHallucination` | Strip markers, append exact local FORMAL, and calculate spans |
| F | `modules/annotation/spans.py` | rendered mentions + injection trace | `AnnotatedHallucination` | Label root and propagated spans with separate causal roles |
| G/H | `modules/release/record.py` | graph, text, spans | JSONL record | Assemble, verify span offsets, and write the dataset |

The two entry points are explicit:

- `pipeline.py` prints one sample's complete input/output at every stage and pauses between stages.
- `generate_dataset.py` applies those same concrete modules to the complete corpus.

## Source tree

```text
molhallulens/
├── config/
│   ├── hallucination_generation.py   # the only generation-parameter file
│   └── paths.py
├── core/                             # typed DAG and unified mutation contracts
├── infrastructure/chemistry/         # RDKit parsing and descriptors
├── modules/
│   ├── ingestion/
│   ├── reference/
│   ├── error_planning/
│   ├── error_injection/
│   ├── text_realization/
│   ├── annotation/
│   └── release/
├── pipeline.py                       # interactive single-record inspection
└── generate_dataset.py               # complete dataset generation
```

## Invariants

1. Each record contains at least one actual semantic change.
2. `edit_count` counts semantic points; a repeated claim may map to several DAG nodes/spans.
3. Integer and float mutations use separate configuration and operators.
4. Fragment replacements are valid, different, pool-backed, and similarity/size filtered.
5. Product and final-answer changes are sanitized structural SMILES edits.
6. Candidate/reference DAG differences exactly equal root nodes plus recorded propagation nodes.
7. Every root or propagated node appears in at least one verified hallucination span.
8. Poe receives the original complete `step_text`; its returned FORMAL must exactly equal the locally rendered modified `formal_ab`.
9. Every identified natural occurrence is wrapped exactly once using `[[HALLU:node_id.NN]]...[[/HALLU]]`; missing, duplicate, and unplanned occurrences fail closed.
10. If a step changes only in FORMAL, its natural-language head remains byte-identical.
11. Arithmetic, repeated-value, and product/final-answer edges must pass after propagation; all known edge statuses are released for audit.
12. The Poe token is read only from `POE_API_KEY` and is absent from cache and records.
