# Current architecture

## Modules and contracts

| Stage | Code | Input | Output | Responsibility |
|---|---|---|---|---|
| A | `modules/ingestion/` | `Dataset/` JSON + manifest | `JoinedInputRecord` | Join and validate raw, process, and template data |
| B | `modules/reference/` | joined record | `ReferenceDAGArtifact` | Parse every `formal_ab` claim into a typed reference DAG |
| B | `modules/error_planning/fragment_pool.py` | all reference DAGs | `FragmentPool` | Build a deduplicated corpus-level functional-group pool |
| C | `modules/error_planning/unified.py` | DAG + config + pool | `UnifiedHallucinationPlan` | Select K points, operators, replacements, and magnitudes |
| D | `modules/error_injection/` | DAG + plan | `InjectedHallucination` | Apply root edits, propagate deterministic claims, and audit every edge |
| E1 | `modules/text_realization/occurrence_audit.py` + `renderer.py` | original `step_text` + candidate DAG | coverage audit + `copy` / `occurrence_patch` / `derivation_rewrite` contract | Scan every changed DAG node in every step; route incomplete or derived prose to derivation rewrite |
| E2 | `modules/text_realization/poe_agent.py` | E1 request | validated natural-language bodies with temporary per-occurrence markers | Patch simple claims or rewrite a complete derivation; every affected claim stays attributable to its DAG node |
| E3 | `modules/text_realization/renderer.py` | marked Poe natural bodies + candidate DAG | `RenderedHallucination` | Locally append the exact Step header and modified FORMAL, strip markers, and calculate spans |
| E4 | `modules/text_realization/pairing.py` | validated H rendering + reference DAG | `MatchedRenderedPair` | Swap marked claims back to truth; regenerate only a step that fails FORMAL, residual, or arithmetic checks |
| F | `modules/annotation/spans.py` | H/N rendered mentions + injection trace | positive/negative `AnnotatedHallucination` | Label H root/propagated spans and one-to-one N control spans without weakening positive validation |
| G/H | `modules/release/record.py` | graph, paired text, spans | paired JSONL records | Assemble H/N records, verify offsets and the byte-identical substitution invariant, then write the dataset |

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
8. Poe receives the original `step_text` and modified `formal_ab` as context but returns only natural language. If it redundantly returns outer whitespace, a Step header, FORMAL, or Answer, the parser extracts prose and discards those model-owned copies. Local code exclusively owns and appends the canonical Step header and modified FORMAL.
9. Strict occurrence matches are compared with a separate high-recall scan for every changed node in every step, including claims outside that step's FORMAL template. Extra loose matches never fail silently: the step is routed to `derivation_rewrite`.
10. Both occurrence patches and derivation rewrites mark changed claim values individually as `[[HALLU:node_id.NN]]after_text[[/HALLU]]`. A derivation rewrite may change surrounding prose for coherence, but only marked claim values become hallucination spans; whole-body spans are forbidden.
11. Natural language is byte-identical only in explicit `copy` mode; an empty strict match list alone is not evidence that a step is unaffected.
12. Old semantic claims and false displayed arithmetic are checked after rewriting. Any violation triggers a Poe retry and ultimately rejects the record.
13. Arithmetic, repeated-value, and product/final-answer DAG edges must pass after propagation; all known edge statuses are released for audit.
14. The Poe token is read only from `POE_API_KEY` and is absent from cache and records.
15. With `emit_matched_negative=True`, every plan releases exactly one H and one N record sharing `pair_id`; their `record_id` values end in `__H` and `__N`.
16. A `byte_identical` step is reconstructed exactly by replacing each H span value with its paired N control value. A failed truth swap is never accepted: that step is reverse-regenerated and disclosed as `regenerated`.
17. N records have `hallucination_present=false`, no hallucination spans, and one truth-valued `control_span` for every H span. `same_char_length` is recorded occurrence by occurrence.
