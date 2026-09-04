# Current architecture

## Modules and contracts

| Stage | Code | Input | Output | Responsibility |
|---|---|---|---|---|
| A | `modules/ingestion/` | `Dataset/` JSON + manifest | `JoinedInputRecord` | Join and validate raw, process, and template data |
| B | `modules/reference/` | joined record | `ReferenceDAGArtifact` | Parse every `formal_ab` claim into a typed reference DAG |
| B | `modules/error_planning/fragment_pool.py` | all reference DAGs | `FragmentPool` | Build a deduplicated corpus-level functional-group pool |
| C | `modules/error_planning/unified.py` | DAG + config + pool | `UnifiedHallucinationPlan` | Select K points, operators, replacements, and magnitudes |
| D | `modules/error_injection/unified.py` | DAG + plan | `InjectedHallucination` | Apply exactly the planned node changes |
| E1 | `modules/text_realization/renderer.py` | candidate DAG | locked modified `formal_ab` + placeholder draft | Keep modified claims and FORMAL under local control |
| E2 | `modules/text_realization/poe_agent.py` | original context + locked draft | validated natural-language templates | Use Poe to rewrite prose across the complete chain |
| E3 | `modules/text_realization/renderer.py` | Poe templates + candidate DAG | `RenderedHallucination` | Insert values, append exact FORMAL, and calculate mentions |
| F | `modules/annotation/spans.py` | rendered mentions + plan | `AnnotatedHallucination` | Label every textual occurrence of every edited node |
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
6. Candidate/reference DAG differences exactly equal the planned node list.
7. Every edited node appears in at least one verified hallucination span.
8. Poe cannot author or alter `FORMAL`; its placeholder multiset must round-trip exactly.
9. The Poe token is read only from `POE_API_KEY` and is absent from cache and records.
