---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- chemistry
- chemical-reasoning
- chain-of-thought
- process-supervision
- benchmark
pretty_name: ChemCoTBench-V2
size_categories:
- 1K<n<10K
configs:
- config_name: default
  default: true
  data_files:
  - split: test
    path: viewer/chemcotbench_v2_viewer.jsonl
---

# ChemCoTBench-V2

This dataset contains the public 5,620-sample active benchmark for
ChemCoTBench-V2, **From Answers to States: Verifiable Process-Level Evaluation
of Chemical Reasoning in Large Language Models**.

ChemCoTBench-V2 evaluates both final-answer correctness and process-level
chemical reasoning. Each benchmark item pairs model-facing inputs with a
verified formal reasoning trace.

## Contents

- `viewer/`: flattened, viewer-friendly JSONL table used by the Hugging Face
  Dataset Viewer.
- `raw_benchmark_data/`: model-facing benchmark inputs and final-answer labels.
- `process_evaluation_data/`: verified process references, including
  `formal_cot_trace`, `parsed_reference_state`, and `verifier_checks`.
- `task_schema/`: task counts and field inventory after public-release cleanup.
- `formal_templates/`: structured step-field schemas for each subtask.
- `prompt_templates/`: evaluation-facing prompt format guidance.
- `verifier_rule_descriptions/`: verifier names and task-level rule notes.
- `sample_examples/`: small examples linking raw inputs with process references.
- `evaluation_split_metadata/`: manifest and split/count metadata.

## Size

The benchmark contains 5,620 samples across four task families:

| Family | Samples | Scope |
| --- | ---: | --- |
| MolEdit | 900 | Addition, deletion, and substitution edits |
| MolUnd | 1,500 | Functional groups, rings, scaffolds, and SMILES equivalence |
| RxnPred | 2,200 | Product prediction, retrosynthesis, mechanism/template reasoning, components, conditions, and yield |
| MolOpt | 1,020 | Single- and dual-objective molecular optimization |

These 31 implementation subtasks are aggregated into 18 reporting tasks in the
paper. The aggregation is presentation-level only and does not change the
benchmark files.

## Record Format

Every raw record has a corresponding process-level record with the same
`anonymous_sample_id`, in the same order. This field is retained as the stable
sample identifier used by the companion evaluation code.

The Hugging Face Dataset Viewer displays `viewer/chemcotbench_v2_viewer.jsonl`.
This file is a flattened preview table with uniform column types. The canonical
evaluation data remain in `raw_benchmark_data/` and `process_evaluation_data/`;
the viewer file is provided only to make the benchmark browsable on the Hub.

The canonical file list and sample counts are stored in `manifest.json`.

Difficulty labels and internal patch/debug metadata are intentionally not
included, because they are not part of the paper-facing benchmark.

## Companion Code

The official code repository contains the evaluation framework, data generation
framework, formal-CoT parsers/verifiers, and environment files. Use that
repository together with this dataset to reproduce benchmark evaluation.

## Citation

Please cite the ChemCoTBench-V2 paper if you use this dataset.
