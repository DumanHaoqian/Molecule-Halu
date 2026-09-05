# Current architecture

## Modules and contracts

Current wire version: `poe_segments_v20`. Editable requests expose only
`original_natural_body`; FORMAL and context steps are read-only fields. PATCH
response examples contain the exact original prose and occurrence-reference
slots, while DERIVATION examples use actual claim/enum IDs. The wire contract
is repeated in the user message rather than relying solely on the system role.
Diagnostics distinguish a wrong reference type from an unknown node. Header,
FORMAL, prose, residual, enumeration, arithmetic and pairing validators are
unchanged. PATCH may explicitly select `patch_ref: original_occurrences` to
expand all exact slots locally. DERIVATION may explicitly select
`draft_ref: complete_derivation` after reviewing the provided complete natural
body, or return expanded authored segments. Expanded-reference rules apply only
to expanded responses, not to these exclusive shorthand operations. Invalid or
missing model responses never trigger draft selection automatically.
`step_execution.response_mode` distinguishes Poe-selected local drafts, exact
patches, and Poe-authored prose. This deliberately produces more templated text
and must not be described as unrestricted model-authored reasoning. All shorthand
expansions pass the same marker, residual, inventory, arithmetic and pair checks.
PATCH requests also show the full rendered candidate; repair instructions explicitly
reject occurrence-only lists that would drop prose. Renderer protocol metadata now
comes from the protocol constant rather than a stale hard-coded version.

| Stage | Code | Input | Output | Responsibility |
|---|---|---|---|---|
| A | `modules/ingestion/` | `Dataset/` JSON + manifest | `JoinedInputRecord` | Join and validate raw, process, and template data |
| B | `modules/reference/` | joined record | `ReferenceDAGArtifact` | Parse every `formal_ab` claim into a typed reference DAG |
| B | `modules/error_planning/fragment_pool.py` | all reference DAGs | `FragmentPool` | Build a deduplicated corpus-level functional-group pool |
| C | `modules/error_planning/unified.py` | DAG + config + pool | `UnifiedHallucinationPlan` | Select K points, operators, replacements, and magnitudes |
| D | `modules/error_injection/` | DAG + plan | `InjectedHallucination` | Apply root edits, propagate deterministic claims, and audit every edge |
| E1 | `modules/text_realization/occurrence_audit.py` + `renderer.py` | original `step_text` + candidate DAG | coverage audit + `copy` / `occurrence_patch` / `derivation_rewrite` contract | Scan every changed DAG node in every step; route incomplete or derived prose to derivation rewrite |
| E2 | `modules/text_realization/poe_agent.py` + `segments.py` + `diagnostics.py` | E1 request | validated locally compiled markers + per-step execution/diagnostics | COPY locally; Poe returns references and prose; compile exact values and inventories, retain passed steps, retry only failures |
| E3 | `modules/text_realization/renderer.py` + `smiles_diff.py` | marked Poe natural bodies + candidate DAG | `RenderedHallucination` | Locally append the exact Step header and modified FORMAL, strip markers, and calculate claim or character-level molecular spans |
| E4 | `modules/text_realization/pairing.py` | validated H rendering + reference DAG | `MatchedRenderedPair` | Swap marked claims back to truth; regenerate only a step that fails FORMAL, residual, arithmetic, or enumeration checks |
| F | `modules/annotation/spans.py` | H/N rendered mentions + injection trace | positive/negative `AnnotatedHallucination` | Label H root/propagated spans and one-to-one N control spans without weakening positive validation |
| G/H | `modules/release/record.py` + `generate_dataset.py` | graph, paired text, spans | paired JSONL records + failure manifest | Assemble H/N records, verify offsets and the byte-identical substitution invariant, then incrementally write each complete pair and explicitly record per-item failures |

The two entry points are explicit:

- `pipeline.py` prints one sample's complete input/output at every stage and pauses between stages.
- `generate_dataset.py` applies those same concrete modules to the complete corpus.

For maximum-density root editing, `generate_dataset.py --max-edits` selects the
central configuration's `maximum` count mode. The existing planner supplies each
DAG's exact non-conflicting root capacity; this mode returns that capacity without
sampling K or applying the fixed/range count cap. Planner candidate selection,
propagation, and edge auditing are unchanged, and failures never lower K silently.

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

### Claim binding and controlled prose equivalence (renderer v14)

`text_realization/claim_surfaces.py` binds anchor element symbols/names and
supplies exact reversible surface pairs to the Poe prompt and validator. The
renderer records the appropriate reference surface; pairing verifies it against
the reference element before restoring a word such as `carbon`. No new graph
node or causal role is introduced for an alias. Molecular strings remain opaque
to anchor/prose normalization.

PATCH accepts whitespace layout, scoped source/product count-verb equivalence,
and count-unit inflection, but not arbitrary paraphrases, new facts, negations,
numeric changes, or molecular-string changes. COPY remains byte-exact. N copies
H's accepted prose and restores only marked values, so release's paired-text
invariant is unchanged. Derivation markers may use only the explicit surface
pairs for their node; unknown synonyms still fail closed.

Coverage includes cross-step `REMOVE_GROUP`/`ADD_FRAGMENT` counts, sentence-final
numbers (without matching an integer inside a decimal), and symbolic atomic
inventories. Generic delta wording is scoped by heavy-atom versus ring steps;
explicitly named cross-step deltas remain auditable. Symbolic inventories become
explicit coefficient lists shared by H/N, and edited coefficients retain their
existing enumeration-child parent and propagated-error attribution. Repeated
totals must retain identical values and bindings, not merely equal digits.

### Preserved enumerations and text-level propagation

`text_realization/enumeration_plan.py` inventories mechanically countable source
breakdowns. E keeps every inventoried component; deleting or generalizing the
breakdown is not an allowed repair. An edited total produces explicit derived
component claims, named `<parent>__enumeration_<clause>_<item>_<surface>`.
The deterministic rule allocates the delta from the last component backwards,
retaining zero-count items when necessary. Repeated number-word/digit spellings
are separately marked. Unknown/ambiguous constructions fail closed.

These are text-layer claims, not additional edits to the reference/candidate DAG.
Each has source-trace `before_text`, `after_text`, and `parent_node_id`; the source
enumeration sum is checked, but this is not a new independent chemical oracle.
Poe receives structured `enumeration_blocks` containing bound count/component
items and totals. It selects a block by enumeration_ref; the compiler renders the
canonical local plan. The final validator still matches inventory semantics rather than whole sentences:
item order, total-first/last wording, punctuation, whitespace, and known noun plurals
may vary. Each list and its total stay in one claim clause (line breaks allowed).
Explicit source/product/delta transitions separate independent claims even when
they share a sentence. Boundaries inside a parenthesized breakdown are not split,
and arbitrary extra markers are never discarded to make an inventory pass.
Unrecognized connectors, missing/extra items, changed counts, or misplaced markers
fail closed with inventory diagnostics. Molecular annotation syntax remains significant.
Derived spans use `propagated_error`, `text.enumeration_count_propagation`,
and an explicit `text:` propagation event, traced through the parent to its root
mutation. Both H spans and N controls carry the parent ID. Swapping all marked
totals and components restores the reference breakdown without changing its shell.
This policy does not alter planner, graph propagation, or edge-audit logic.

1. Each record contains at least one actual semantic change.
2. `edit_count` counts semantic points; a repeated claim may map to several DAG nodes/spans.
3. Integer and float mutations use separate configuration and operators.
4. Fragment replacements are valid, different, pool-backed, and similarity/size filtered.
5. Product and final-answer changes are sanitized structural SMILES edits.
6. Candidate/reference DAG differences exactly equal root nodes plus recorded propagation nodes.
7. Every root or propagated node appears in at least one verified hallucination span.
8. The internal request retains original `step_text`; the Poe wire exposes its body as `original_natural_body`, modified `formal_ab`, and separate read-only full-chain context. Poe returns only structured segments for requested non-COPY steps. Literal markers, Step headers, FORMAL and Answer are rejected; local code exclusively fills claim values, assigns marker IDs, expands complete inventories and appends Step headers/FORMAL. Complete `PoeRewriteRequest` ordering remains consecutive; transport subsets are separate payloads retaining real step indices.
9. Strict occurrence matches are compared with a separate high-recall scan for every changed node in every step, including claims outside that step's FORMAL template. Extra loose matches never fail silently: the step is routed to `derivation_rewrite`.
10. Both occurrence patches and derivation rewrites compile references to local `[[HALLU:node_id.NN]]after_text[[/HALLU]]` markers. A derivation rewrite may change surrounding prose for coherence, but only marked claim values become hallucination spans; whole-body spans are forbidden. Each enumeration_ref is expanded exactly once using the already validated text-layer component plan. No dynamic unplanned error is inferred or accepted from model prose.
11. Natural language is byte-identical only in explicit `copy` mode; an empty strict match list alone is not evidence that a step is unaffected.
12. Old semantic claims, false displayed arithmetic, and explicit heavy-atom/ring enumerations whose components do not sum to their total are checked after rewriting. Any violation triggers a Poe retry and ultimately rejects the record. Ambiguous fused-system prose is never assigned an inferred count.
13. Arithmetic, repeated-value, and product/final-answer DAG edges must pass after propagation; all known edge statuses are released for audit.
14. The Poe token is read only from `POE_API_KEY` and is absent from cache and records.
15. With `emit_matched_negative=True`, every plan releases exactly one H and one N record sharing `pair_id`; their `record_id` values end in `__H` and `__N`.
16. A `byte_identical` step is reconstructed exactly by replacing each H span value with its paired N control value. A failed truth swap is never accepted: that step is reverse-regenerated and disclosed as `regenerated`.
17. N records have `hallucination_present=false`, no hallucination spans, and one truth-valued `control_span` for every H span. `same_char_length` is recorded occurrence by occurrence.
18. Changed SMILES/MOLECULE claims use one contiguous `SequenceMatcher` bounding interval on each side. Pure insertions/deletions include one shared boundary character so neither side is empty. `context_span` retains the complete molecular string and `diff_opcodes` retains the changed opcodes. A deterministic renderer-only rooted-SMILES traversal prevents RDKit canonical start-atom changes from turning local graph edits into whole-string labels; the displayed SMILES remains molecularly equivalent to the injected DAG value.
19. Batch generation flushes every complete H/N pair before starting the next item. A failed origin/variant is written immediately to a separate JSONL failure manifest, processing continues, the summary reports all failures and Poe retry/validation counters, and the CLI exits nonzero if any item failed. Stale cache metadata or a response rejected by the current validator is a cache miss; unreadable cache JSON remains a hard error.
20. Response rows are associated by strict integer step IDs. Accepted steps are not requested again within a rewrite. Every failed step produces a bounded, redacted diagnostic; retries include the preceding rejection. Complete assembled output is revalidated before storing a versioned cache. Cache identities include the full context, direction-specific origin ID, protocol prompt and model directory. Legacy string responses are not silently accepted.
21. Diagnostic storage is separate from success caches, controlled only through the central generation config. Rejected responses never enter success caches or release records. Successful retries may publish their sanitized rejection history outside detector_input. N reverse-regeneration retains the same checks and discloses its execution/diagnostic trace.
