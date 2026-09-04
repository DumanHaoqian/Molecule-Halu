# MolHalluLens Pilot

This version turns ChemCoTBench-V2 molecule-editing traces into **matched positive/negative
hallucination pairs**. Each H record contains one or more independently selected edits in
the reasoning steps and/or final-answer SMILES, plus exact text spans. Its N partner uses
the same text shell with those values restored to reference truth and exposes aligned
`control_spans`. Sampled root errors remain separate from deterministic consequences.

The single data flow is:

```text
A ingestion
  -> B reference DAG + fragment pool
  -> C edit planning
  -> D root mutation + deterministic propagation + edge audit
  -> E audit prose coverage, then patch occurrences or rewrite a derivation with Poe
  -> E2 restore marked values to construct the matched N text
  -> F hallucination/control span annotation
  -> G paired record assembly and invariant checks
  -> H paired JSONL release
```

All generation controls live in
[`molhallulens/config/hallucination_generation.py`](molhallulens/config/hallucination_generation.py):
editable semantic points, number of edits, integer/float magnitude, fragment similarity,
final-answer probability, SMILES operators, matched-negative emission, Poe bot, cache,
and random seed.

## Poe API token

Create a key at <https://poe.com/api/keys>, then export it in the same terminal
before running generation:

```bash
export POE_API_KEY='YOUR_POE_API_KEY'
```

The program reads `POE_API_KEY` at request time. It has no command-line token
argument and never writes the token to configuration, prompts, logs, cache, or output.
The bot name and environment-variable name are configured in
`hallucination_generation.py`.

Generation makes one Poe request per uncached record. Poe charges the key owner's
compute points, so first inspect one record with `pipeline.py`; validated responses
are cached under `GeneratedDataset/.poe_text_cache/` and can be replayed without a key.

## Run

Activate the existing environment and work from `Pilot/`:

```bash
conda activate molhallulens

# Launch the local A-to-E visual walkthrough (annotation is merged into E).
molhallulens-demo

# Inspect one sample module by module; stage E calls Poe.
python -m molhallulens.pipeline

# Run the same one-sample demo without pauses.
python -m molhallulens.pipeline --no-pause

# Generate the complete 150-origin dataset.
python -m molhallulens.generate_dataset \
  --variants-per-origin 1 \
  --output GeneratedDataset/example.jsonl

# Small real-Poe smoke test: 3 origins -> 3 H/N pairs -> 6 records.
python -m molhallulens.generate_dataset \
  --max-origins 3 \
  --output GeneratedDataset/smoke.jsonl

# Verify the implementation.
python -m pytest
```

By default, each origin/variant produces an H record with `hallucination_present: true`
and an N record with `hallucination_present: false`; both share `pair_id`.
Poe receives each original complete `step_text` together with its modified `formal_ab`.
Before the call, local code compares a precise occurrence finder with a separate high-recall
semantic scan across every changed node and every step. Complete simple matches use
occurrence-specific HALLU markers; incomplete, arithmetic, enumerative, or compositional prose is rewritten
as a derivation, while every changed claim value is still marked separately by node and
occurrence. Local validation rejects whole-body or unplanned markers, stale old claims, false
displayed arithmetic, inconsistent heavy-atom/ring component totals, and missing/duplicate markers. Poe
returns natural-language bodies only; local code appends the exact Step header and modified
FORMAL, making FORMAL drift impossible. Only explicit `copy` steps are locked byte-for-byte.
For matched controls, COPY shares its prose directly, while patched and rewritten steps
replace verified marker contents with truth. Local code checks the reference FORMAL,
candidate-value residuals, arithmetic, and explicit component sums after replacement. A step that cannot be restored
safely is reverse-regenerated through Poe and disclosed as `pair_alignment: regenerated`;
otherwise its alignment is `byte_identical`. Each N `control_span` maps to one H span by
`pair_occurrence_id`, and both sides record `same_char_length`.
For changed product/final-answer molecular strings, the paired span is the smallest single
contiguous character interval that reconstructs N from H. `context_span` points to the full
SMILES and `diff_opcodes` records the underlying changed SequenceMatcher operations. Pure
insertions/deletions include one shared boundary character so both H and N remain non-empty.
The text layer may choose an equivalent rooted SMILES traversal to keep a local graph edit
local in text; planning and injected DAG values are unchanged.
Generation writes and flushes each complete pair immediately. Failures do not erase earlier
pairs or stop later origins: each failed origin/variant is written to
`<output-stem>.failures.jsonl`, counted in the summary, and causes a nonzero CLI exit after
the batch finishes. Summary telemetry includes rewrite modes, pair alignments, Poe network
attempts/retries, and arithmetic/enumeration rejection counts. A cache entry with stale
metadata or a response rejected by the current validator is treated as a cache miss and
refreshed; malformed unreadable cache JSON still fails closed.
The test suite uses an injected fake Poe transport and therefore spends no points.

See [ARCHITECTURE.md](ARCHITECTURE.md) for file-level module ownership and contracts.
