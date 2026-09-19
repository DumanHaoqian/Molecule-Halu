# ChemDFM-R-14B: A/B/C/D outcome-only experiment

Input: `Pilot/GeneratedDataset/maximum_edits_complete.jsonl` (300 H/N records,
150 unique questions, 50 each of add/delete/substitute). Each question is run
once in each condition, for 600 generations. No training or model updates occur.

| Group | Model input | Generated response |
| --- | --- | --- |
| A | Question | Direct product SMILES |
| B | Question + full H reasoning | Direct product SMILES |
| C | Question + full N reasoning | Direct product SMILES |
| D | Question | Model-generated CoT, followed by product SMILES |

Question contains the original indexed source SMILES and edit instruction.
B/C receive identical wrapper prompts, with no H/N labels or correctness hints.
Their entire `reasoning_chain` is retained. The `final_answer` dataset field is
never appended to the input. **The reasoning itself already contains a correct
product SMILES in all 150 H and all 150 N records.** Thus, B/C measure answering
with a supplied product-bearing trace; copying a product from that trace is
possible. H/N here differ in intermediate assertions, not in their products.

The ground truth is the original raw benchmark `gt_smiles`, joined by
`origin_id == anonymous_sample_id`. All generated-dataset `final_answer` values
were independently checked for molecular equivalence to that original target.
Ground truth is stored separately from inference requests.

## Inference

Local model: `/mnt_nas1/shared/ChemDFM-R-14B` (BF16, no quantization).
Greedy decoding: temperature 0, top_p 1, top_k -1, repetition penalty 1, seed 42,
one output per condition and question, up to 8,192 newly generated tokens in
all groups; 16,384-token total context. No truncation of inputs is permitted.
Two replicas each use tensor parallelism across two 4090 GPUs. Questions are
assigned to replicas by sorted pair index modulo 2, so all four conditions for
each question use the same replica. Completed requests are checkpointed
individually, with continuous batching (up to 16 active sequences per replica).
The initial execution used batches of 16; it was resumed with continuous
queuing so a long continuation would not block other waiting questions.
Previously saved outputs are retained. Prompt and sampling settings are unchanged.

The model's bundled chat template is used. A/B/C use the assistant prefill
`<think>\n</think>\n<answer>\n`, skipping explicit generated CoT. D uses
`<think>\n` and generates its own reasoning and answer. This controls the
observable token sequence, not unobservable computation inside the network.
Generation stops at `</answer>` or an EOS token. Every rendered prompt's hash,
raw generated continuation, assistant prefill, finish reason, and token counts
are saved. Actual environment versions are in `runtime.shard*.json`.

## Outcome evaluation

Only the explicitly generated final answer is scored. Molecules mentioned in
the generated CoT are never used as fallback answers. The last complete
`<answer>...</answer>` is used; an explicit `Answer:` line after a closed CoT is
also accepted. Missing, incomplete, or non-single-SMILES answers fail.

Metrics reuse the ChemCoTBench-V2 MolEdit utility functions directly:

- Primary: `smiles_match_exact`, canonical RDKit molecular equality, preserving
  stereochemistry and all fragments (not literal SMILES string equality).
- Secondary: `smiles_match_main_frag`, equality of largest heavy-atom fragments.
- Secondary: `fts`, Tanimoto of union Morgan fingerprints (radius 2, 2,048 bits).
- Auxiliary outcome diagnostic: `map_normalized_exact_accuracy` removes atom-map
  IDs from valid predictions before exact matching, preserving connectivity,
  formal charge, isotopes and stereochemistry. The original benchmark helper
  retains atom-map IDs, so otherwise equivalent mapped output can fail its
  exact comparison. Both scores are retained; the benchmark score is primary.

All questions remain in the denominator. No Layer 2, Layer 3, hallucination,
span, or reasoning-correctness scores are computed. Valid-SMILES, missing-answer,
and output-length-limit counts are recorded solely to explain outcome failures.
There is no best-of-N, resampling, or oracle-assisted answer selection.

## Reproduce / resume

From `/home/haoqian/Data/Molecule`, first prepare a **new** directory if starting
a new run (the existing manifest is protected against accidental replacement):

```bash
python3 Pilot/scripts/run_chemdfm_outcome.py self-check
python3 Pilot/scripts/run_chemdfm_outcome.py prepare --output /path/to/new_run
```

The following commands run/resume this directory's immutable manifest, on
separate GPU pairs (execute the two commands in separate terminals):

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  /home/haoqian/Data/miniconda3/envs/skillopt/bin/python -u \
  Pilot/scripts/run_chemdfm_outcome.py run --shard 0

CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  /home/haoqian/Data/miniconda3/envs/skillopt/bin/python -u \
  Pilot/scripts/run_chemdfm_outcome.py run --shard 1

python3 Pilot/scripts/run_chemdfm_outcome.py summarize
```

For a different output directory, add the same `--output` to every command.
`summarize` refuses incomplete runs unless `--allow-partial` is explicit.
Resume skips completed request IDs and validates their request hashes.

## Files

- `manifest.json`: protocol, exact settings, input checksums, dataset audit.
- `requests.jsonl`: all 600 model inputs, with no separate GT answer field.
- `ground_truth.jsonl`: target molecules and pair/source IDs.
- `runtime.shard*.json`, `run.shard*.log`: runtime metadata and execution logs.
- `predictions.shard*.jsonl`: checkpointed generations before scoring.
- `outcome_records.jsonl`: per-output molecule extraction and outcome scores.
- `summary.json`, `summary.csv`: overall and subtask outcome metrics, plus
  paired comparisons of exact correctness.
