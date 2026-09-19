# ChemDFM outcome experiment

`run_chemdfm_outcome.py` is the single entry point replacing outcome v1–v4.
It evaluates existing H/N pairs; it does not train a model or generate dataset samples.

| Group | Input | Output |
| --- | --- | --- |
| A | Question | Answer only |
| B | Question + saved H reasoning | Answer only |
| C | Question + saved N reasoning | Answer only |
| D | Question | Model-generated reasoning + answer |

The dataset must already have product-answer clauses removed. H/N reasoning is
passed through unchanged. Dataset labels, spans and final-answer fields are not
included in prompts. Both plain and atom-indexed source SMILES are provided.

Run from `Pilot/scripts` in an environment with RDKit, Transformers and vLLM:

```bash
python run_chemdfm_outcome.py self-check
python run_chemdfm_outcome.py prepare
CUDA_VISIBLE_DEVICES=0,1 python run_chemdfm_outcome.py run
python run_chemdfm_outcome.py summarize
```

The default model is `/mnt_nas1/shared/ChemDFM-R-14B`; choose a different local
path with `--model` during `prepare`. The default output directory is
`Pilot/Experiments/chemdfm_r14b_outcome_unified`. Use the same `--output` for all
three commands when selecting another directory. Historical versioned experiment
directories use a different format; prepare a new directory for this runner.

- `prepare --pairs-per-subtask 2` prepares a deterministic small sample.
- `prepare --batch-size 8` sets the inference batch size.
- `run --limit 4` processes at most four pending requests.
- Repeat `run` to resume. Completed rows are flushed to disk; engine-aborted
  responses are not marked complete. Request contents are checked directly,
  without hashes. Do not edit the manifest, requests or ground truth mid-run.
- `summarize --allow-partial` summarizes completed rows before all requests finish.

Inference uses two GPUs (tensor parallelism 2), selected through
`CUDA_VISIBLE_DEVICES`, greedy decoding and a 2048-token output budget.
Missing/invalid answers count as incorrect; length-limited outputs are retained
and counted as truncated. Scoring reports atom-map-normalized main-fragment and
exact matches, raw exact matches, fingerprint similarity, and paired comparisons.
Outputs include `predictions.jsonl`, `outcome_records.jsonl`, `summary.json`,
`summary.csv` and runtime metadata.

The wrong-anchor/wrong-fragment causal experiment runners and their analysis
script have been removed from this directory.
