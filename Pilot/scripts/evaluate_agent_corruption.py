#!/usr/bin/env python3
"""Evaluate paired agent corruptions with supplied-text or assistant-prefix placement.

Preparation and reporting run on CPU. ``run`` and ``probe`` each load one local
vLLM engine; invoke them sequentially when sharing the same GPU allocation.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Pilot"))
_legacy_spec = importlib.util.spec_from_file_location(
    "_agent_evaluation_legacy", Path(__file__).with_name("run_chemLLM_outcome.py"))
legacy = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(legacy)
EVALUATION_VERSION = "agent_corruption_causal_v1"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_annotations(args, pairs: dict) -> dict:
    """Read audited character annotations, binding them to the exact saved H."""
    source = args.annotations
    if source is not None and not source.exists():
        raise ValueError(f"Annotations path does not exist: {source}")
    objects = []
    if source is not None and source.is_file():
        data = legacy.read_jsonl(source) if source.suffix == ".jsonl" else json.loads(source.read_text())
        objects = [(item, source) for item in (data if isinstance(data, list) else [data])]
    else:
        base = source or args.dataset.parent
        for origin_id in sorted({pair["H"]["origin_id"] for pair in pairs.values()}):
            candidates = [base / "origins" / origin_id / "accepted.json", base / origin_id / "accepted.json"]
            for path in candidates:
                if path.exists():
                    objects.append((json.loads(path.read_text()), path))
                    break
    result = {}
    for item, path in objects:
        pair_id = item["pair_id"]
        if pair_id not in pairs:
            continue
        if pair_id in result:
            raise ValueError(f"Duplicate annotations for pair: {pair_id}")
        trace = pairs[pair_id]["H"]["detector_input"]["reasoning_chain"]
        if "pairs" in item:
            saved_h = [row for row in item["pairs"] if row["variant_label"] == "H"]
            if len(saved_h) != 1 or saved_h[0]["detector_input"]["reasoning_chain"] != trace:
                raise ValueError(f"Audited H trace differs from dataset: {pair_id}")
        elif "reasoning_chain" in item and item["reasoning_chain"] != trace:
            raise ValueError(f"Annotated H trace differs from dataset: {pair_id}")
        if not isinstance(item.get("annotations"), list):
            raise ValueError(f"Annotations must be a list: {pair_id}")
        result[pair_id] = {"trace": trace, "annotations": item["annotations"], "source": str(path.resolve())}
    if source is not None and not result:
        raise ValueError("No matching pair annotations found")
    return result


def prepare_prompt_labels(args, requests: list[dict], pairs: dict) -> list[dict]:
    annotations = load_annotations(args, pairs)
    tokenizer = None
    if annotations:
        from transformers import AutoTokenizer
        from molhallulens.modules.agent_generation.labels import label_prompt_tokens
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    result = []
    for request in requests:
        row = {key: request[key] for key in ("request_id", "group", "pair_id", "origin_id", "subtask")}
        group = request["group"]
        if group not in "BCE":
            result.append({**row, "status": "no_supplied_trace"})
            continue
        if tokenizer is None or (group == "B" and request["pair_id"] not in annotations):
            result.append({**row, "status": "unavailable", "reason": "No audited character annotations for this H trace"})
            continue
        source_pair = request["reasoning_source"]["pair_id"] if group == "E" else request["pair_id"]
        trace = pairs[source_pair]["H" if group == "B" else "N"]["detector_input"]["reasoning_chain"]
        annotation = annotations.get(source_pair) if group == "B" else None
        spans = annotation["annotations"] if annotation else []
        chat = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
        prompt = chat + request["assistant_prefix"]
        if args.placement == "prefix":
            trace_start = len(chat) + len("<think>\n")
        else:
            user = request["messages"][1]["content"]
            user_start = chat.find(user)
            if user_start < 0 or chat.find(user, user_start + 1) >= 0 or not user.endswith(trace):
                raise ValueError("Cannot uniquely locate supplied reasoning in rendered chat template")
            trace_start = user_start + len(user) - len(trace)
        labels = label_prompt_tokens(prompt, trace, spans, tokenizer, trace_start=trace_start)
        trace_end = trace_start + len(trace)
        trace_tokens = [token for token in labels["tokens"] if token["start"] < trace_end and token["end"] > trace_start]
        result.append({**row, "status": "available", "label_scope": labels["label_scope"],
                       "tokenizer": str(args.model.resolve()), "trace_char_range": [trace_start, trace_end],
                       "prompt_token_ids": labels["prompt_token_ids"], "trace_tokens": trace_tokens,
                       "annotations": spans,
                       "annotation_source": annotation["source"] if annotation else "Original N; no injected error spans",
                       "offset_note": "Offsets and token indices refer to exact rendered full prompt; boundary tokens may include adjacent control text",
                       "error_token_count": sum(any(label != "unchanged" for label in token["labels"]) for token in trace_tokens)})
    return result


def prepare(args) -> None:
    """Freeze full requests and ground truth; never include audit metadata in prompts."""
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Manifest already exists; choose a fresh output directory.")
    chem = legacy.metrics_module()
    pairs = collections.defaultdict(dict)
    for row in legacy.read_jsonl(args.dataset):
        label, pair_id = row["variant_label"], row["pair_id"]
        if label not in {"H", "N"} or label in pairs[pair_id]:
            raise ValueError(f"Unknown or duplicate variant for pair {pair_id}")
        pairs[pair_id][label] = row
    if not pairs or any(set(pair) != {"H", "N"} for pair in pairs.values()):
        raise ValueError("Dataset must contain complete H/N pairs")
    raw = {}
    for subtask in ("add", "delete", "substitute"):
        path = ROOT / f"Pilot/Dataset/raw_benchmark_data/mol_edit/{subtask}_pilot_origin.json"
        for row in json.loads(path.read_text()):
            origin_id = row["anonymous_sample_id"]
            if origin_id in raw:
                raise ValueError(f"Duplicate benchmark origin: {origin_id}")
            raw[origin_id] = row
    selected = legacy.select_pairs(pairs, args.pairs_per_subtask)
    if args.pair_ids:
        keep = set(args.pair_ids.read_text().splitlines())
        unknown = keep - set(pairs)
        if unknown:
            raise ValueError(f"Unknown selected pair IDs: {sorted(unknown)}")
        selected = [(key, pair) for key, pair in selected if key in keep]
    if not selected:
        raise ValueError("No pairs selected")
    donors = legacy.select_reasoning_donors(pairs) if "E" in args.groups else {}
    pair_indices = {pair_id: index for index, pair_id in enumerate(sorted(pairs))}
    requests, truths = [], []
    checks = collections.Counter()
    for pair_id, pair in selected:
        pair_index = pair_indices[pair_id]
        h, n = pair["H"], pair["N"]
        if h["origin_id"] != n["origin_id"] or h["subtask"] != n["subtask"]:
            raise ValueError(f"Pair question metadata differs: {pair_id}")
        gt = raw[h["origin_id"]]
        indexed, instruction = n["detector_input"]["indexed_smiles"], n["detector_input"]["instruction"]
        for row in (h, n):
            visible = row["detector_input"]
            if visible["indexed_smiles"] != indexed or indexed != gt["indexed_smiles"]:
                raise ValueError(f"Source molecule differs from benchmark: {pair_id}")
            if visible["instruction"] != instruction or instruction != gt["instruction"]:
                raise ValueError(f"Original question instruction differs: {pair_id}")
            if not chem.smiles_match_exact(gt["gt_smiles"], visible["final_answer"]):
                raise ValueError(f"Ground truth differs from benchmark: {pair_id}")
        plain = legacy.unmap_smiles(indexed)
        if not plain:
            raise ValueError(f"Invalid source molecule: {pair_id}")
        reasoning = {label: pair[label]["detector_input"]["reasoning_chain"] for label in "HN"}
        if "E" in args.groups:
            donor_id = donors[pair_id]
            donor = pairs[donor_id]["N"]
            reasoning["E"] = donor["detector_input"]["reasoning_chain"]
            checks["E_different_origin"] += donor["origin_id"] != h["origin_id"]
            checks["E_same_subtask"] += donor["subtask"] == h["subtask"]
        checks["H_N_reasoning_differ"] += reasoning["H"] != reasoning["N"]
        checks[f'H_root_errors_{h["edit_count"]}'] += 1
        checks["H_root_errors_total"] += h["edit_count"]
        question = ("[Question]\n" f"Plain source SMILES: {plain}\n"
                    f"Indexed source SMILES: {indexed}\n" f"Instruction: {instruction}")
        truths.append({"pair_id": pair_id, "origin_id": h["origin_id"], "subtask": h["subtask"],
                       "source_smiles": plain, "gt_smiles": gt["gt_smiles"],
                       "h_record_id": h.get("record_id", f"{pair_id}:H"),
                       "n_record_id": n.get("record_id", f"{pair_id}:N")})
        for group in args.groups:
            user, system, prefix = question, legacy.DIRECT, legacy.PREFIXES[group]
            if group == "D":
                system = legacy.COT
            elif group in "BCE":
                trace = reasoning[{"B": "H", "C": "N", "E": "E"}[group]]
                if args.placement == "prefix":
                    prefix = f"<think>\n{trace}\n</think>\n<answer>\n"
                else:
                    system = legacy.FOLLOW_REASONING
                    user += ("\n\n[Supplied reasoning]\n"
                             "Follow the reasoning below to produce the final answer without revising it.\n" + trace)
            request = {"request_id": f"{group}:{pair_id}", "group": group,
                       "pair_index": pair_index, "pair_id": pair_id,
                       "origin_id": h["origin_id"], "subtask": h["subtask"],
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "assistant_prefix": prefix}
            if group == "E":
                request["reasoning_source"] = {"pair_id": donor_id, "origin_id": donor["origin_id"],
                                                "record_id": donor.get("record_id", f"{donor_id}:N"),
                                                "subtask": donor["subtask"]}
            requests.append(request)
    prompt_labels = prepare_prompt_labels(args, requests, pairs)
    manifest = {
        # Keep the existing runtime's protocol untouched. The extension has its own version.
        "protocol_version": legacy.PROTOCOL_VERSION,
        "agent_evaluation": {"version": EVALUATION_VERSION, "placement": args.placement,
                             "groups": args.groups, "question_policy": "Original benchmark instruction preserved verbatim",
                             "resume_policy": "Compare complete saved manifest, request and ground-truth objects"},
        "dataset": str(args.dataset.resolve()),
        "model": {"name": args.model_name, "path": str(args.model.resolve())},
        "n_pairs": len(truths), "n_origins": len({row["origin_id"] for row in truths}),
        "n_requests": len(requests),
        "groups": {group: {"A": "Empty reasoning", "B": "Paired corrupted reasoning",
                           "C": "Paired original correct reasoning", "D": "Model-generated reasoning",
                           "E": "Correct reasoning borrowed from a different origin"}[group] for group in args.groups},
        "e_reasoning_policy": ("Full-dataset sorted circular assignment; different origin required; same subtask preferred"
                               if "E" in args.groups else "Not evaluated; no donor assignment"),
        "supplied_reasoning_policy": ("Reasoning is an assistant continuation prefix; same DIRECT system and user question for ABCE"
                                      if args.placement == "prefix" else "Legacy supplied user text with FOLLOW_REASONING system"),
        "subtasks": dict(collections.Counter(row["subtask"] for row in truths)),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.05,
                     "max_tokens": 2048, "seed": 42, "n": 1},
        "engine": {"dtype": "bfloat16", "tensor_parallel_size": args.tensor_parallel_size,
                   "max_model_len": 16384, "gpu_memory_utilization": args.gpu_memory_utilization, "max_num_seqs": 16,
                   "max_num_batched_tokens": 8192, "enforce_eager": True,
                   "disable_custom_all_reduce": True, "enable_prefix_caching": False,
                   "generation_config": "vllm"},
        "batch_size": args.batch_size, "dataset_checks": dict(checks),
        "gt_source": "Original benchmark gt_smiles joined by origin_id == anonymous_sample_id",
        "primary_metric": "Atom-map-normalized main-fragment molecular equality",
        "secondary_metrics": ["atom-map-normalized exact equality", "raw strict equality", "FTS"],
        "prompt_label_status": dict(collections.Counter(row["status"] for row in prompt_labels)),
    }
    write_jsonl(out / "requests.jsonl", requests)
    write_jsonl(out / "ground_truth.jsonl", truths)
    write_jsonl(out / "prompt_token_labels.jsonl", prompt_labels)
    legacy.save_json(out / "prepared_inputs.json", {"manifest": manifest, "requests": requests, "ground_truth": truths,
                                                    "prompt_token_labels": prompt_labels})
    legacy.save_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "pairs": len(truths), "requests": len(requests),
                      "placement": args.placement, "groups": args.groups}), flush=True)


def load_experiment(args) -> tuple[dict, list[dict], list[dict]]:
    manifest, requests = legacy.load_experiment(args.output)
    legacy.check_model_selection(args, manifest)
    if manifest.get("agent_evaluation", {}).get("version") != EVALUATION_VERSION:
        raise ValueError("Not an agent corruption evaluation experiment")
    truths = legacy.read_jsonl(args.output / "ground_truth.jsonl")
    frozen = json.loads((args.output / "prepared_inputs.json").read_text())
    prompt_labels = legacy.read_jsonl(args.output / "prompt_token_labels.jsonl")
    if frozen != {"manifest": manifest, "requests": requests, "ground_truth": truths, "prompt_token_labels": prompt_labels}:
        raise ValueError("Prepared manifest, request or ground-truth content differs; use a fresh output directory")
    return manifest, requests, truths


def run(args) -> None:
    load_experiment(args)
    legacy.run(args)


def comparison(rows: list[dict], baseline: str, treatment: str, metric: str) -> dict:
    from molhallulens.modules.agent_generation.metrics import paired_comparison
    by_pair = collections.defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["group"]] = row
    pairs = [value for _, value in sorted(by_pair.items()) if baseline in value and treatment in value]
    result = paired_comparison([bool(pair[baseline][metric]) for pair in pairs],
                               [bool(pair[treatment][metric]) for pair in pairs],
                               baseline_label=baseline, treatment_label=treatment)
    # Retain explicit directional names for downstream causal readouts.
    flip = result["flip_to_wrong"]
    return {**result, "harmful_flip_rate": flip["rate"], "harmful_flip_wilson_95": flip["ci95"]}


def report(args) -> None:
    manifest, _, truths = load_experiment(args)
    legacy.summarize(args)
    rows = legacy.read_jsonl(args.output / "outcome_records.jsonl")
    summary = json.loads((args.output / "summary.json").read_text())
    comparisons = {"H_vs_N": ("C", "B"), "H_vs_empty": ("A", "B"),
                   "N_vs_empty": ("A", "C"), "irrelevant_vs_empty": ("A", "E"),
                   "H_vs_irrelevant": ("E", "B")}
    summary["causal_comparisons"] = {
        name: {metric: comparison(rows, baseline, treatment, metric)
               for metric in ("primary_match", "normalized_exact_match")}
        for name, (baseline, treatment) in comparisons.items()
        if baseline in manifest["agent_evaluation"]["groups"] and treatment in manifest["agent_evaluation"]["groups"]
    }
    summary["agent_evaluation"] = manifest["agent_evaluation"]
    summary["inference_unit"] = "pair"
    summary["repeated_origins"] = len(truths) != len({row["origin_id"] for row in truths})
    summary["inference_note"] = ("Variants sharing an origin are correlated; pair-level exact tests and Wilson intervals are descriptive when repeated_origins=true")
    summary["metric_scope"] = {"primary": "Main-fragment equality can ignore auxiliary components",
                               "normalized_exact": "Full molecular equality after atom-map normalization"}
    legacy.save_json(args.output / "summary.json", summary)
    print(json.dumps({"causal_comparisons": summary["causal_comparisons"]}, indent=2), flush=True)


def teacher_forcing_payload(tokenizer, request: dict, answer: str) -> dict:
    """Score one fixed answer token continuation after an exact rendered prefix.

    Encode the prefix and answer separately, then concatenate token IDs. This
    defines an identical answer event in every condition and prevents a token
    crossing the prefix/answer character boundary from contaminating the score.
    """
    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                           add_generation_prompt=True) + request["assistant_prefix"]
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not prefix_ids or not answer_ids:
        raise ValueError("Teacher forcing requires nonempty prompt and answer token sequences")
    return {"prompt_token_ids": prefix_ids + answer_ids, "answer_start": len(prefix_ids),
            "answer_token_ids": answer_ids, "answer": answer}


def answer_logprob(prompt_logprobs, payload: dict) -> dict:
    if prompt_logprobs is None or len(prompt_logprobs) != len(payload["prompt_token_ids"]):
        raise ValueError("Missing or misaligned vLLM prompt log probabilities")
    values = []
    for offset, token_id in enumerate(payload["answer_token_ids"], payload["answer_start"]):
        scores = prompt_logprobs[offset]
        if scores is None or token_id not in scores:
            raise ValueError(f"Ground-truth token log probability missing at offset {offset}")
        value = scores[token_id]
        score = float(value.logprob if hasattr(value, "logprob") else value["logprob"])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite ground-truth token log probability at offset {offset}")
        values.append(score)
    return {"sum_logprob": sum(values), "mean_logprob": statistics.mean(values), "answer_tokens": len(values)}


def probe_report(args) -> dict:
    manifest, requests, _ = load_experiment(args)
    expected = {row["request_id"]: row for row in requests if row["group"] in "ABCE"}
    records = legacy.read_jsonl(args.output / "probe_records.jsonl")
    by_pair = collections.defaultdict(dict)
    seen = set()
    for row in records:
        key = row["request_id"]
        if key in seen or key not in expected or row["request"] != expected[key]:
            raise ValueError(f"Unknown, duplicate or changed probe request: {key}")
        seen.add(key)
        by_pair[row["pair_id"]][row["group"]] = row
    contrasts = {}
    for label, baseline, treatment in (("H_minus_N", "C", "B"), ("H_minus_empty", "A", "B"),
                                       ("N_minus_empty", "A", "C"), ("irrelevant_minus_empty", "A", "E")):
        pairs = []
        for pair_id, values in sorted(by_pair.items()):
            if baseline in values and treatment in values:
                if values[baseline]["answer_token_ids"] != values[treatment]["answer_token_ids"]:
                    raise ValueError(f"Teacher-forced continuation differs across conditions: {pair_id}")
                pairs.append({"pair_id": pair_id, "origin_id": values[treatment]["origin_id"],
                              "subtask": values[treatment]["subtask"],
                              "delta_mean_logprob": values[treatment]["mean_logprob"] - values[baseline]["mean_logprob"],
                              "delta_sum_logprob": values[treatment]["sum_logprob"] - values[baseline]["sum_logprob"]})
        contrasts[label] = {"n": len(pairs), "mean_delta_mean_logprob": statistics.mean(p["delta_mean_logprob"] for p in pairs) if pairs else None,
                            "mean_delta_sum_logprob": statistics.mean(p["delta_sum_logprob"] for p in pairs) if pairs else None,
                            "pairs": pairs}
    summary = {"model": manifest["model"], "complete": len(records) == len(expected),
               "n_expected": len(expected), "n_completed": len(records), "units": "natural log probability (nats)",
               "answer_event": "Original benchmark GT SMILES tokenized once without special tokens; identical token continuation across conditions",
               "scope": "GT SMILES tokens only; excludes closing answer tag and EOS; D excluded",
               "contrasts": contrasts}
    legacy.save_json(args.output / "probe_summary.json", summary)
    return summary


def probe(args) -> None:
    """Teacher-force GT tokens with vLLM prompt_logprobs, independently resumable."""
    manifest, requests, truths_list = load_experiment(args)
    probe_config = {"version": EVALUATION_VERSION, "model": manifest["model"], "engine": manifest["engine"],
                    "sampling": {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1},
                    "token_boundary_policy": "Separate prefix and fixed answer tokenization, followed by token-ID concatenation"}
    config_path = args.output / "probe_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != probe_config:
        raise ValueError("Saved probe configuration differs")
    truths = {row["pair_id"]: row for row in truths_list}
    expected = {row["request_id"]: row for row in requests if row["group"] in "ABCE"}
    results_path = args.output / "probe_records.jsonl"
    existing = legacy.read_jsonl(results_path)
    done = set()
    for row in existing:
        key = row["request_id"]
        if key in done or key not in expected or row["request"] != expected[key]:
            raise ValueError(f"Unknown, duplicate or changed probe request: {key}")
        if row["answer"] != truths[row["pair_id"]]["gt_smiles"]:
            raise ValueError(f"Saved probe ground truth differs: {key}")
        done.add(key)
    pending = [row for key, row in expected.items() if key not in done]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print(json.dumps(probe_report(args)), flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = manifest["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    payloads = {key: teacher_forcing_payload(tokenizer, row, truths[row["pair_id"]]["gt_smiles"])
                for key, row in expected.items()}
    for row in existing:
        if row["probe_input"] != payloads[row["request_id"]]:
            raise ValueError(f"Saved probe tokenized request differs: {row['request_id']}")
    if max(len(payloads[row["request_id"]]["prompt_token_ids"]) for row in pending) + 1 > manifest["engine"]["max_model_len"]:
        raise ValueError("Teacher-forcing input exceeds model context length")
    legacy.save_json(config_path, probe_config)
    print(json.dumps({"probe_pending": len(pending), "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "all")}), flush=True)
    llm = LLM(model=model, seed=manifest["sampling"]["seed"], **manifest["engine"])
    params = SamplingParams(**probe_config["sampling"])
    started = time.monotonic()
    with results_path.open("a") as handle:
        for offset in range(0, len(pending), manifest["batch_size"]):
            batch = pending[offset:offset + manifest["batch_size"]]
            outputs = llm.generate([{"prompt_token_ids": payloads[row["request_id"]]["prompt_token_ids"]} for row in batch],
                                   params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("Incomplete teacher-forcing batch")
            for request, output in zip(batch, outputs, strict=True):
                payload = payloads[request["request_id"]]
                if output.prompt_token_ids != payload["prompt_token_ids"]:
                    raise ValueError("vLLM returned a different teacher-forcing token sequence")
                scores = answer_logprob(output.prompt_logprobs, payload)
                record = {**{key: request[key] for key in ("request_id", "group", "pair_id", "origin_id", "subtask")},
                          "request": request, "probe_input": payload, "answer": payload["answer"],
                          "answer_token_ids": payload["answer_token_ids"], **scores}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(request["request_id"])
            print(f"PROBE completed={len(done)}/{len(expected)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    print(json.dumps(probe_report(args)), flush=True)


def entropy_config(args, manifest: dict) -> dict:
    sampling = {**manifest["sampling"], "n": args.entropy_samples,
                "temperature": 0.8, "top_p": 0.95}
    config = {"version": EVALUATION_VERSION, "model": manifest["model"], "engine": manifest["engine"],
              "sampling": sampling, "cluster_policy": "Atom-map-free canonical full SMILES; invalid and missing are separate clusters"}
    path = args.output / "entropy_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Saved entropy configuration differs")
    return config


def entropy_rows(args, expected: dict, count: int) -> list[dict]:
    rows = legacy.read_jsonl(args.output / "entropy_records.jsonl")
    seen = set()
    for row in rows:
        key = row["request_id"]
        if key in seen or key not in expected or row["request"] != expected[key]:
            raise ValueError(f"Unknown, duplicate or changed entropy request: {key}")
        if len(row["samples"]) != count or any(sample["finish_reason"] not in {"stop", "length"} for sample in row["samples"]):
            raise ValueError(f"Incomplete entropy samples: {key}")
        seen.add(key)
    return rows


def entropy_report(args) -> dict:
    manifest, requests, _ = load_experiment(args)
    config = entropy_config(args, manifest)
    expected = {row["request_id"]: row for row in requests if row["group"] in "ABCE"}
    rows = entropy_rows(args, expected, config["sampling"]["n"])
    groups = {}
    for group in "ABCE":
        members = [row for row in rows if row["group"] == group]
        if members:
            groups[group] = {"n": len(members), "mean_entropy_bits": statistics.mean(row["entropy"]["entropy_bits"] for row in members),
                             "mean_valid_coverage": statistics.mean(row["entropy"]["coverage"] for row in members)}
    summary = {"model": manifest["model"], "sampling": config["sampling"], "groups": groups,
               "n_expected": len(expected), "n_completed": len(rows), "complete": len(rows) == len(expected),
               "interpretation": "Finite-sample empirical canonical-SMILES entropy; lower entropy does not imply correctness; D excluded",
               "invalid_policy": "All samples remain in denominator; invalid and missing outputs have separate categories"}
    legacy.save_json(args.output / "entropy_summary.json", summary)
    return summary


def entropy(args) -> None:
    """Estimate answer dispersion from eight sampled continuations per prompt."""
    from molhallulens.modules.agent_generation.metrics import canonical_smiles_entropy
    manifest, requests, _ = load_experiment(args)
    config = entropy_config(args, manifest)
    expected = {row["request_id"]: row for row in requests if row["group"] in "ABCE"}
    done = {row["request_id"] for row in entropy_rows(args, expected, config["sampling"]["n"])}
    pending = [row for key, row in expected.items() if key not in done]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print(json.dumps(entropy_report(args)), flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = manifest["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompt_ids = [tokenizer.encode(tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                                   + row["assistant_prefix"], add_special_tokens=False) for row in pending]
    if max(map(len, prompt_ids)) + config["sampling"]["max_tokens"] > manifest["engine"]["max_model_len"]:
        raise ValueError("Entropy prompt exceeds model context length")
    legacy.save_json(args.output / "entropy_config.json", config)
    print(json.dumps({"entropy_pending": len(pending), "samples_per_request": args.entropy_samples,
                      "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "all")}), flush=True)
    llm = LLM(model=model, seed=config["sampling"]["seed"], **manifest["engine"])
    params = SamplingParams(**config["sampling"], stop=["</answer>"],
                            stop_token_ids=legacy.stop_token_ids(tokenizer, Path(model)), include_stop_str_in_output=True)
    started = time.monotonic()
    with (args.output / "entropy_records.jsonl").open("a") as handle:
        for offset in range(0, len(pending), manifest["batch_size"]):
            batch = pending[offset:offset + manifest["batch_size"]]
            outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids[offset:offset + len(batch)]], params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("Incomplete entropy batch")
            interrupted = []
            for request, output in zip(batch, outputs, strict=True):
                if len(output.outputs) != args.entropy_samples or any(pred.finish_reason not in {"stop", "length"} for pred in output.outputs):
                    interrupted.append(request["request_id"])
                    continue
                samples = []
                for index, pred in enumerate(output.outputs):
                    response = request["assistant_prefix"] + pred.text
                    answer, method = legacy.extract_answer(response)
                    samples.append({"index": index, "generated_text": pred.text, "predicted_smiles": answer,
                                    "answer_extraction": method, "output_tokens": len(pred.token_ids),
                                    "finish_reason": pred.finish_reason, "stop_reason": pred.stop_reason})
                record = {**{key: request[key] for key in ("request_id", "group", "pair_id", "origin_id", "subtask")},
                          "request": request, "samples": samples,
                          "entropy": canonical_smiles_entropy([sample["predicted_smiles"] for sample in samples],
                                                               canonicalizer=legacy.unmap_smiles)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(request["request_id"])
            print(f"ENTROPY completed={len(done)}/{len(expected)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
            if interrupted:
                raise RuntimeError("Retryable entropy interruption; incomplete requests not checkpointed: " + ", ".join(interrupted))
    print(json.dumps(entropy_report(args)), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "summarize", "probe", "probe-report", "entropy", "entropy-report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--annotations", type=Path, help="Audited annotation JSON/JSONL or generation directory; defaults to dataset sibling origins/*/accepted.json")
    parser.add_argument("--model", dest="model_name", choices=legacy.MODELS, default="ChemDFM-R-14B")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--groups", choices=("ABC", "ABCE", "ABCDE"), default="ABCE")
    parser.add_argument("--placement", choices=("supplied", "prefix"), default="prefix")
    parser.add_argument("--pairs-per-subtask", type=int)
    parser.add_argument("--pair-ids", type=Path, help="Optional newline-separated pair IDs; donor assignment still uses the full dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--entropy-samples", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    args.model = args.model_path or legacy.MODELS[args.model_name]
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.command == "prepare" and args.dataset is None:
        parser.error("prepare requires --dataset")
    for name in ("batch_size", "tensor_parallel_size", "limit", "pairs_per_subtask", "entropy_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command in {"report", "summarize"}:
        report(args)
    elif args.command == "probe-report":
        print(json.dumps(probe_report(args), indent=2), flush=True)
    elif args.command == "entropy-report":
        print(json.dumps(entropy_report(args), indent=2), flush=True)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
