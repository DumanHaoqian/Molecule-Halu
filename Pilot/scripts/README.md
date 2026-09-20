# Chemical LLM outcome experiment

`run_chemLLM_outcome.py` is the single entry point replacing outcome v1–v4.
It evaluates existing H/N pairs; it does not train a model or generate dataset samples.

| Group | Input | Output |
| --- | --- | --- |
| A | Question | Answer only |
| B | Question + saved H reasoning | Answer only |
| C | Question + saved N reasoning | Answer only |
| D | Question | Model-generated reasoning + answer |
| E | Question + saved N reasoning from another question | Answer only |

The dataset must already have product-answer clauses removed. H/N reasoning is
passed through unchanged. Dataset labels, spans and final-answer fields are not
included in prompts. Both plain and atom-indexed source SMILES are provided.

E borrows the unmodified N reasoning from a different origin in the full input
JSONL, even when `--pairs-per-subtask` selects fewer target questions. Assignment
is deterministic: visit sorted pair IDs circularly, prefer another origin with
the same edit subtask, and fall back to a different subtask only if necessary.
At least two distinct origins are required. No same-origin variant can serve as
a donor. B/C/E share the same supplied-reasoning introduction and direct-answer
prefix; only the supplied reasoning changes. The borrowed question and final
answer are not added. `reasoning_source` in each E request records the donor pair,
origin, record and subtask outside the model messages. Predictions retain this
metadata in their saved `request`.

The `follow_reasoning_v2` protocol requires B/C/E to follow the supplied reasoning
as fixed premises without independently re-solving, verifying, correcting or
discarding it. This replaces the previous instruction allowing corrections.
A/D prompts are unchanged. Old runs remain separate; they cannot be resumed
under the new protocol. Prompt instructions do not guarantee model compliance.

A full 150-question dataset now produces 750 requests per model. Summaries include
E scores and paired comparisons E vs A, C vs E, and B vs E, alongside the original
comparisons. Donor text lengths are not matched to target reasoning lengths.

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
and Qwen end markers. The A/B/C/D/E instructions and decoding budget are the same
for both models.

| Model | Default local path | Output under `Pilot/Experiments/` |
| --- | --- | --- |
| Chem-R-8B | `chemical_models/Chem-R-8B` under the repository root | `chem_r8b_outcome_abcde_follow_reasoning` |
| ChemDFM-R-14B | `/mnt_nas1/shared/ChemDFM-R-14B` | `chemdfm_r14b_outcome_abcde_follow_reasoning` |

Use `--model-path /path/to/weights` to override a preset. Use the same model,
path override and `--output` (if supplied) for prepare/run/summarize. The runner
rejects a selection that differs from the saved experiment, preventing mixed-model
results. Four-group experiment manifests cannot be resumed as five-group experiments;
run prepare in a fresh directory. The new default directories end in `_abcde_follow_reasoning`.

- `prepare --pairs-per-subtask 2` prepares a deterministic small sample.
- `prepare --batch-size 8` sets the inference batch size.
- `run --limit 4` processes at most four pending requests.
- Repeat `run` to resume. Completed rows are flushed to disk; engine-aborted
  responses are not marked complete. Request contents are checked directly,
  without hashes. Do not edit the manifest, requests or ground truth mid-run.
- `summarize --allow-partial` summarizes completed rows before all requests finish.

Inference defaults to two GPUs (`prepare --tensor-parallel-size 2`), selected through
`CUDA_VISIBLE_DEVICES`, greedy decoding and a 2048-token output budget.
Missing/invalid answers count as incorrect; length-limited outputs are retained
and counted as truncated. Scoring reports atom-map-normalized main-fragment and
exact matches, raw exact matches, fingerprint similarity, and paired comparisons.
Outputs include `predictions.jsonl`, `outcome_records.jsonl`, `summary.json`,
`summary.csv` and runtime metadata.

`bash scripts/run_exp.sh` launches both models with nohup, sequentially running
ChemDFM-R-14B then Chem-R-8B on GPUs 4,5,6,7 with tensor parallelism 4. Settings
are hardcoded in the script. `scripts/run_exp.log` contains all output and starts
with the supervisor PID and process-group stop command. The launcher prevents
duplicate runs and resumes existing checkpoints. After each model finishes,
`scripts/run_exp_result.md` is updated with one A–E table per model, reporting
primary (main-fragment) accuracy and mean fingerprint similarity.

The wrong-anchor/wrong-fragment causal experiment runners and their analysis
script have been removed from this directory.

## Replay Agent H against original N

The existing `gendemo.py` entry point accepts the saved Agent pair format:

```bash
cd /home/haoqian/Data/Molecule/Pilot/scripts
/home/haoqian/.venvs/chemllm-outcome/bin/python gendemo.py \
  --records ../Experiments/agent_corruption_v17/intent/pairs.jsonl \
  --port 7838 --no-browser
```

Keep the adjacent `origins/<origin>/accepted.json` and `input.json` files with
the JSONL. The viewer checks exact pair contents and annotation offsets, then
shows original N versus H, actual experiment N versus H, root/propagated spans,
dependency values and saved tokenizer counts. Original N excludes the historical
complete answer clause; paired N may include the additional connection block.
Literal text differences are separate from semantic error labels. Counts refer
to standalone H, not full chat prompts. v17 contains four diagnostic pairs,
not the complete development or heldout datasets. Viewing calls no model/API
and does not rewrite data. The no-argument default remains the older release.

### Full150 ABC experiment (agent corruption v18)

The full150 run uses `Experiments/agent_corruption_v18/full150`, protocol
`agent_full_task_interpretation_v1`. It includes all 150 original examples
(add/delete/substitute: 50 each), including prior development examples. It is
an exploratory full-corpus evaluation, not a held-out test.

A uses no trace; B uses the new H; C retains the original N after explicit-answer
projection. H uses a deterministic canonical template, so style differs from N.
Original instructions, indexed source molecules and reference answers are unchanged.
Each H has at least six distinct changed semantic nodes and at least 40 annotated
CoT tokens under both model tokenizers. Agent concerns are retained as diagnostic
reviews; program acceptance does not mean unanimous agent approval.

Generation: `python scripts/generate_full150_pairs.py --phase all --workers 6`
(with `POE_API_KEY` loaded by the existing `poe` alias). Generation is resumable;
code, complete original rows and tokenizer bytes are checked against the frozen
run. The initial source-derived candidate pools were imported from the documented
construction preflight; its pilot API selections were not imported.

Before evaluation: `python scripts/validate_full150_pairs.py` reexecutes all selected
edits and reconstructs every H and token label. Then `bash scripts/run_full150_exp.sh`
starts a detached job, sequentially ChemDFM-R-14B and Chem-R-8B on GPUs 4–7.
The launcher rejects partial datasets and runs 450 ABC requests per model.

- Generation log: `scripts/run_full150_generation.log`; PID file: `scripts/run_full150_generation.pid`.
- Evaluation log: `scripts/run_full150_exp.log`; its first lines give the PID and stop command.
- Results: `scripts/run_full150_exp_result.md`, one ABC table per model, plus paired B−A/B−C comparisons.
- Audit: `Experiments/agent_corruption_v18/full150/release_validation.json` and per-origin sidecars.

`python scripts/summarize_full150_exp.py` refreshes the result tables; incomplete
runs are explicitly provisional and keep the denominator at 150 per group.
