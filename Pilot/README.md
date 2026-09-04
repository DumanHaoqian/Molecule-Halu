# MolHalluLens Pilot

This version turns ChemCoTBench-V2 molecule-editing traces into **always-positive,
configurable hallucination records**. A record contains one or more independently
selected edits in the reasoning steps and/or final-answer SMILES, plus exact text spans.
Sampled root errors are separated from deterministic downstream consequences.

The single data flow is:

```text
A ingestion
  -> B reference DAG + fragment pool
  -> C edit planning
  -> D root mutation + deterministic propagation + edge audit
  -> E audit prose coverage, then patch occurrences or rewrite a derivation with Poe
  -> F span annotation
  -> G record assembly
  -> H JSONL release
```

All generation controls live in
[`molhallulens/config/hallucination_generation.py`](molhallulens/config/hallucination_generation.py):
editable semantic points, number of edits, integer/float magnitude, fragment similarity,
final-answer probability, SMILES operators, Poe bot, cache, and random seed.

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

# Verify the implementation.
python -m pytest
```

Every output record has `hallucination_present: true` and one unified schema.
Poe receives each original complete `step_text` together with its modified `formal_ab`.
Before the call, local code compares a precise occurrence finder with a separate high-recall
semantic scan across every changed node and every step. Complete simple matches use
occurrence-specific HALLU markers; incomplete, arithmetic, or compositional prose is rewritten
as a derivation, while every changed claim value is still marked separately by node and
occurrence. Local validation rejects whole-body or unplanned markers, stale old claims, false
displayed arithmetic, and missing/duplicate markers. Poe
returns natural-language bodies only; local code appends the exact Step header and modified
FORMAL, making FORMAL drift impossible. Only explicit `copy` steps are locked byte-for-byte.
The test suite uses an injected fake Poe transport and therefore spends no points.

See [ARCHITECTURE.md](ARCHITECTURE.md) for file-level module ownership and contracts.
