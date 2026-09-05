# MolHalluLens Pilot

## Install

Use Python 3.11-3.13 and the single dependency file `requirements.txt`, containing
both runtime and test dependencies. From `Pilot/`:

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Maintain `requirements.txt` rather than separate `.in`, `.lock`, or development
requirements files. `pyproject.toml` remains the package/build metadata.

## Final saved dataset

`GeneratedDataset/` contains only `maximum_edits_complete.jsonl`: 150 origins,
150 H/N pairs (300 records). Historical intermediate outputs, logs, caches and
dataset-folder reports were moved to the macOS Trash during cleanup.
Recovery details and limitations remain summarized in `completion_notes.md`.

From `Pilot/`, replay all 150 saved pairs without calling Poe:

```sh
python scripts/gendemo.py
```

Reusable `gendemo.py`, `merge_recovered_pairs.py` and `retry_failed_cases.py` live
under `scripts/`; retrying requires an explicit `--failures` manifest.

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

Generation batches the non-COPY steps of an uncached record into its first Poe
request; subsequent attempts request only failed steps. All-COPY records need no
Poe request. Poe charges the key owner's
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

# Maximum root edits for EVERY origin (not a random count up to the maximum).
python -m molhallulens.generate_dataset \
  --max-edits \
  --output GeneratedDataset/maximum_edits.jsonl

# Verify the implementation.
python -m pytest
```

`--max-edits` selects `edit_count_mode="maximum"` from the central generation
configuration for this run. K is the exact maximum number of mutually
non-conflicting root edits for each DAG; propagated graph/text errors are counted
separately. This mode ignores fixed/range count settings, does not reduce K on
failure, and leaves the default random-range configuration unchanged. Existing
output/manifest paths are overwritten on rerun; use a new output path to retain
an earlier dataset.

The demo now has one **Run A–E** button: local stages run first, followed
automatically by Poe (which may consume API points). E displays hallucination
tokens / all tokens and their percentage over the complete `serialized.text`.
The reference tokenizer is `tiktoken`'s `cl100k_base`, configured by
`DEMO_TOKEN_ENCODING` in `config/hallucination_generation.py`; it is not claimed
to be the Poe bot's or downstream chemistry model's tokenizer. Token/character
alignment uses UTF-8 bytes; any overlap with an annotated hallucination span
counts the token once. Model-added chat templates and BOS/EOS are excluded.
The first tokenizer load may download its public vocabulary; record text stays local.

By default, each origin/variant produces an H record with `hallucination_present: true`
and an N record with `hallucination_present: false`; both share `pair_id`.
Poe receives original `step_text`, modified `formal_ab`, and read-only full-chain context.
Before the call, local code compares a precise occurrence finder with a separate high-recall
semantic scan across every changed node and every step. Complete simple matches use
occurrence-specific HALLU markers; incomplete, arithmetic, enumerative, or compositional prose is rewritten
as a derivation, while every changed claim value is still marked separately by node and
occurrence. Local validation rejects whole-body or unplanned markers, stale old claims, false
displayed arithmetic, inconsistent heavy-atom/ring component totals, and missing/duplicate markers. Poe
returns structured text/claim/enumeration segments; local code appends the exact Step header and modified
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

Enumerations must be retained. If a total changes from 9 to 10, a component may
change from `3 fluorines` to `4 fluorines`, but both changes are marked. The text
layer supplies a deterministic component-count propagation plan before Poe runs;
Poe cannot invent an unlisted change or remove the breakdown. Derived component
claims have their own IDs, a `parent_node_id`, and `propagated_error` spans. The N
partner restores both values. Mechanically unsupported enumerations fail closed;
source component truths are inherited from the reference trace, not inferred by Poe.
Enumeration preservation checks count/component bindings, not verbatim sentences.
Poe can reorder the list, move the total, change punctuation, and use supported noun
plurals; it must keep the complete list and total in one claim clause. Explicit
source/product/delta clauses may share a sentence and are checked separately. Missing/extra
components, wrong values/markers, and unsupported paraphrases still reject. H/N
byte-identical substitution checks are unchanged. Renderer v15 invalidates older caches.

Claim surface binding lives in `modules/text_realization/claim_surfaces.py`.
An anchor's `C` and `carbon` are reversible surfaces of the same node; changing
them to `S` and `sulfur` produces separate spans with unchanged causal attribution.
Quoted fragment SMILES and branches such as `N(C)` are not anchor mentions.
PATCH permits only explicitly checked prose equivalences: whitespace layout,
source/product `has` / `contains` (with optional `molecule`), and numeric
atom(s)/ring(s) unit inflection. Other wording changes still reject. This is not
an H-versus-original byte-identity requirement: accepted wording is shared by H
and its restored N, whose span-substitution invariant remains mandatory.
Symbolic inventories such as `Heavy atoms: N + C + O = 3` are audited too; a total
edit requires explicit, marked component coefficients, never deletion of items.
Enumeration connectors `comprising` / `consisting` and repeated identical totals
are accepted only with the correct count/component and marker bindings.

See [ARCHITECTURE.md](ARCHITECTURE.md) for file-level module ownership and contracts.

## Structured Poe protocol (v20)

Poe can explicitly select a fully rendered local candidate using
`{"draft_ref":"complete_derivation"}` (DERIVATION), or the exact original-body
patch using `{"patch_ref":"original_occurrences"}` (PATCH). Each is an exclusive
one-segment operation. Alternatively, expanded authored segments remain supported.
There is no automatic draft fallback after rejection. All paths use the same
strict validators. Per-step `response_mode` records which path was selected;
Poe-selected local drafts are more templated, not freely authored reasoning.

Poe does not copy claim values or write markers. For a DERIVATION step, for example:

```json
{"steps":[{"step_index":1,"segments":[
  {"text":"The ANCHOR is the "},
  {"claim_ref":"anchor_element","surface":"name"},
  {"text":" atom."}
]}]}
```

`segments.py` fills exact values, assigns marker suffixes, expands planned
`enumeration_ref` blocks and runs the unchanged text validators. PATCH uses the
provided `occurrence_ref` IDs at their original positions; DERIVATION uses
`claim_ref` and an allowed surface. Every enumeration must be referenced exactly
once, with all original components and planned component changes locally rendered.
Editable wire input is now `original_natural_body`, without a Step header or
FORMAL block. Full-chain context and modified FORMAL remain separately read-only.
Each PATCH request includes an exact occurrence-reference example constructed
from the original offsets; DERIVATION examples use actual claim and inventory IDs.
The body-only, mode-specific contract is repeated in the user message. Validators
remain strict; a reference of the wrong type now gets an actionable diagnostic.
Arbitrary literal markers, values supplied with a claim reference, unknown references,
missing claims and unsafe prose changes reject. The old string response format is
not a fallback. The deterministic renderer remains an offline fixture, not Poe.

Passed steps are retained within the current rewrite; retries send only failed
steps with redacted diagnostics and their preceding rejected response. The full
assembled chain is revalidated before a complete success cache is saved. Partial
successes from an ultimately failed record are not a resumable success cache.
N reverse-regeneration uses the same protocol and records its own execution trace.

Diagnostic controls are `POE_SAVE_DIAGNOSTICS`, `POE_DIAGNOSTIC_DIRECTORY` and
`POE_DIAGNOSTIC_MAX_CHARACTERS` in the central config. By default rejections are
written to `GeneratedDataset/.poe_text_diagnostics/`, separate from successful
caches. Logs include run/origin/step/attempt/code, expected/observed data, response
excerpts and hashes; prose-drift errors include an explicit text diff. Text fields
are redacted and length-limited. Transport exceptions expose only their type, not
HTTP headers. Failure manifests carry the structured diagnostics and their path;
successful retry records carry diagnostics and per-step attempts too. Summary
distinguishes network retry rounds from retried-step counts and local COPY counts.

Live validation on 2026-09-05: the v15 retry was paused at 64 completed failures
and zero recovered pairs; one case was interrupted. After the v16 interface
correction, three representative failed origins (add/delete/substitute) were
tested, with six network attempts. All three still failed complete release,
although header/FORMAL and mixed-mode claim-reference errors disappeared in
those cases. Remaining failures concern inventory placement, duplicate occurrence
IDs, and stale prose/arithmetic values. No further full batch was started. See
`completion_notes.md` and `GeneratedDataset/retry_v16_20260905_195335.summary.json`.
