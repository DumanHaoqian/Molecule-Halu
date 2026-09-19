# Chemical LLM outcome experiment

`run_chemLLM_outcome.py` is the single entry point replacing outcome v1–v4.
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
source /home/haoqian/.venvs/chemllm-outcome/bin/activate
python run_chemLLM_outcome.py self-check
python run_chemLLM_outcome.py prepare --model Chem-R-8B
CUDA_VISIBLE_DEVICES=0,1 python run_chemLLM_outcome.py run --model Chem-R-8B
python run_chemLLM_outcome.py summarize --model Chem-R-8B
```

The local `chemllm-outcome` environment inherits the existing skillopt packages
and pins `huggingface-hub==0.36.2` for Transformers 4.57.6 / vLLM 0.19.0.
The original skillopt environment retains Hub 1.32.0 for Gradio 6.28.0.
Use this inference environment for the runner and the original environment for
the Gradio demo.

Choose `--model Chem-R-8B` or `--model ChemDFM-R-14B` for each command.
The default remains ChemDFM-R-14B. Each uses its own tokenizer and chat template;
EOS IDs are loaded from the model configuration and tokenizer, including Llama
and Qwen end markers. The A/B/C/D instructions and decoding budget are the same
for both models.

| Model | Default local path | Output under `Pilot/Experiments/` |
| --- | --- | --- |
| Chem-R-8B | `chemical_models/Chem-R-8B` under the repository root | `chem_r8b_outcome_unified` |
| ChemDFM-R-14B | `/mnt_nas1/shared/ChemDFM-R-14B` | `chemdfm_r14b_outcome_unified` |

Use `--model-path /path/to/weights` to override a preset. Use the same model,
path override and `--output` (if supplied) for prepare/run/summarize. The runner
rejects a selection that differs from the saved experiment, preventing mixed-model
results. Existing unified ChemDFM manifests remain readable; older versioned
experiment directories require a fresh prepare.

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
