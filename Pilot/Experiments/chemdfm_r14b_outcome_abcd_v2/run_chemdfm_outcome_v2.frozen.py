#!/usr/bin/env python3
"""Leakage-controlled, reproducible ChemDFM-R MolEdit outcome experiment."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "Pilot/Experiments/chemdfm_r14b_outcome_abcd_v2"
MODEL = Path("/mnt_nas1/shared/ChemDFM-R-14B")
REDACTION = "<REDACTED_PRODUCT_SMILES>"
BASE = (
    "You are an expert chemist. Apply the requested edit to the source molecule. "
    "The plain and atom-indexed SMILES describe the same source molecule. "
    "Return the final product as one valid, canonicalizable SMILES without atom-map numbers. "
    "Preserve stereochemistry and every molecular component not affected by the edit."
)
DIRECT = BASE + " Respond only with <answer>PRODUCT_SMILES</answer>. Do not explain."
COT = BASE + (
    " Reason step by step inside <think>...</think>, then return "
    "<answer>PRODUCT_SMILES</answer>."
)
PREFIXES = {g: "<think>\n</think>\n<answer>\n" for g in "ABC"}
PREFIXES["D"] = "<think>\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def script_sha() -> str:
    return sha(Path(__file__).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path: Path, obj) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def metrics_module():
    sys.path.insert(0, str(ROOT / "ChemCoTBench-V2"))
    from baselines.cot_eval.mol_edit.mol_edit_structured import utils
    return utils


def rdkit_modules():
    from rdkit import Chem
    return Chem


def unmap_smiles(smiles: str) -> str:
    """Canonicalize a valid SMILES after removing atom-map metadata."""
    Chem = rdkit_modules()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def has_atom_maps(smiles: str) -> bool:
    Chem = rdkit_modules()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return bool(mol is not None and any(atom.GetAtomMapNum() for atom in mol.GetAtoms()))


def redact_product(reasoning: str, product_strings: set[str]) -> str:
    """Remove literal outcome molecules while retaining intermediate reasoning."""
    result = reasoning
    for value in sorted({x for x in product_strings if x}, key=len, reverse=True):
        result = result.replace(value, REDACTION)
    result = re.sub(
        r'PRODUCT_SMILES\("[^"\n]*"\)',
        f'PRODUCT_SMILES("{REDACTION}")',
        result,
    )
    return result


def extract_answer(response: str) -> tuple[str, str]:
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.S | re.I)
    if blocks:
        value, method = blocks[-1], "answer_tag"
    else:
        if "<think>" in response and "</think>" not in response:
            return "", "unclosed_think"
        tail = response.rsplit("</think>", 1)[-1]
        labels = re.findall(r"^\s*(?:Final\s+)?Answer:\s*(\S+)\s*$", tail, re.M | re.I)
        if not labels:
            return "", "no_final_answer"
        value, method = labels[-1], "answer_line"
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:smiles)?\s*|\s*```$", "", value, flags=re.I).strip()
    value = value.strip("`\"'")
    if not value or re.search(r"\s", value) or "<" in value or ">" in value:
        return "", "non_single_smiles_answer"
    return value, method


def select_pairs(pairs: dict[str, dict], per_subtask: int | None) -> list[tuple[str, dict]]:
    selected = []
    counts = collections.Counter()
    for pair_id, pair in sorted(pairs.items()):
        subtask = pair["N"]["subtask"]
        if per_subtask is not None and counts[subtask] >= per_subtask:
            continue
        selected.append((pair_id, pair))
        counts[subtask] += 1
    return selected


def model_metadata(model: Path) -> dict:
    result = {"path": str(model)}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = model / name
        if path.exists():
            result[f"{name}_sha256"] = sha(path.read_bytes())
    return result


def prepare(args) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Manifest already exists; choose a fresh output directory.")
    chem = metrics_module()
    rows = read_jsonl(args.dataset)
    pairs = collections.defaultdict(dict)
    for row in rows:
        assert row["variant_label"] not in pairs[row["pair_id"]]
        pairs[row["pair_id"]][row["variant_label"]] = row

    raw, raw_hashes = {}, {}
    for subtask in ("add", "delete", "substitute"):
        path = ROOT / f"Pilot/Dataset/raw_benchmark_data/mol_edit/{subtask}_pilot_origin.json"
        raw_hashes[str(path)] = sha(path.read_bytes())
        for row in json.loads(path.read_text()):
            assert row["anonymous_sample_id"] not in raw
            raw[row["anonymous_sample_id"]] = row

    requests, truths = [], []
    checks = collections.Counter()
    selected = select_pairs(pairs, args.pairs_per_subtask)
    for pair_index, (pair_id, pair) in enumerate(selected):
        assert set(pair) == {"H", "N"}
        h, n = pair["H"], pair["N"]
        assert h["origin_id"] == n["origin_id"]
        gt = raw[h["origin_id"]]
        indexed = n["detector_input"]["indexed_smiles"]
        instruction = n["detector_input"]["instruction"]
        assert h["detector_input"]["indexed_smiles"] == indexed == gt["indexed_smiles"]
        assert h["detector_input"]["instruction"] == instruction == gt["instruction"]
        plain = unmap_smiles(indexed)
        assert plain and chem.smiles_match_exact(plain, unmap_smiles(indexed))
        assert chem.smiles_match_exact(gt["gt_smiles"], h["detector_input"]["final_answer"])
        assert chem.smiles_match_exact(gt["gt_smiles"], n["detector_input"]["final_answer"])

        product_strings = {gt["gt_smiles"], h["detector_input"]["final_answer"], n["detector_input"]["final_answer"]}
        for record in (h, n):
            product_strings.update(re.findall(r'PRODUCT_SMILES\("([^"\n]+)"\)', record["detector_input"]["reasoning_chain"]))
        redacted = {
            "H": redact_product(h["detector_input"]["reasoning_chain"], product_strings),
            "N": redact_product(n["detector_input"]["reasoning_chain"], product_strings),
        }
        for label in ("H", "N"):
            assert REDACTION in redacted[label]
            assert all(value not in redacted[label] for value in product_strings if value)
            assert not re.search(r'PRODUCT_SMILES\("(?!<REDACTED_PRODUCT_SMILES>)[^"\n]+"\)', redacted[label])
            for candidate in re.findall(r'"([^"\n]+)"', redacted[label]):
                if chem.smiles_valid(candidate):
                    assert not chem.smiles_match_main_frag(
                        unmap_smiles(candidate), unmap_smiles(gt["gt_smiles"])
                    ), f"Equivalent product leaked in {label} reasoning for {pair_id}"
            checks[f"{label}_product_mentions_removed"] += 1
        checks["redacted_H_N_reasoning_differ"] += redacted["H"] != redacted["N"]

        question = (
            "[Question]\n"
            f"Plain source SMILES: {plain}\n"
            f"Indexed source SMILES: {indexed}\n"
            f"Instruction: {instruction}"
        )
        truths.append({
            "pair_id": pair_id,
            "origin_id": h["origin_id"],
            "subtask": h["subtask"],
            "source_smiles": plain,
            "gt_smiles": gt["gt_smiles"],
            "h_record_id": h["record_id"],
            "n_record_id": n["record_id"],
        })
        for group in "ABCD":
            user = question
            if group in "BC":
                label = "H" if group == "B" else "N"
                user += (
                    "\n\n[Candidate reasoning]\n"
                    "The final product string has been redacted. Intermediate claims may be wrong; "
                    "independently determine the answer.\n"
                    + redacted[label]
                )
            request = {
                "request_id": f"{group}:{pair_id}",
                "group": group,
                "pair_index": pair_index,
                "pair_id": pair_id,
                "origin_id": h["origin_id"],
                "subtask": h["subtask"],
                "messages": [
                    {"role": "system", "content": COT if group == "D" else DIRECT},
                    {"role": "user", "content": user},
                ],
                "assistant_prefix": PREFIXES[group],
            }
            request["request_sha256"] = sha(json.dumps(request, sort_keys=True).encode())
            requests.append(request)

    assert len(truths) == len({x["origin_id"] for x in truths})
    for name, data in (("requests.jsonl", requests), ("ground_truth.jsonl", truths)):
        with (out / name).open("x") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "protocol_version": "outcome_abcd_v2_no_product_leakage",
        "script_sha256": script_sha(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha(args.dataset.read_bytes()),
        "raw_reference_sha256": raw_hashes,
        "model": model_metadata(args.model),
        "n_pairs": len(truths),
        "n_requests": len(requests),
        "groups": {
            "A": "Plain+indexed question; direct answer",
            "B": "Plain+indexed question + product-redacted H reasoning; direct answer",
            "C": "Plain+indexed question + product-redacted N reasoning; direct answer",
            "D": "Plain+indexed question; model-generated reasoning and answer",
        },
        "subtasks": dict(collections.Counter(r["subtask"] for r in truths)),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.05, "max_tokens": 2048, "seed": 42, "n": 1},
        "engine": {"dtype": "bfloat16", "tensor_parallel_size": 2, "max_model_len": 16384, "gpu_memory_utilization": 0.88, "max_num_seqs": 16, "max_num_batched_tokens": 8192, "enforce_eager": True, "disable_custom_all_reduce": True, "enable_prefix_caching": False, "generation_config": "vllm"},
        "allowed_visible_gpus": ["6", "7"],
        "batch_size": args.batch_size,
        "assistant_prefixes": PREFIXES,
        "dataset_checks": dict(checks),
        "gt_source": "Original benchmark gt_smiles joined by origin_id == anonymous_sample_id",
        "primary_metric": "Atom-map-normalized main-fragment molecular equality",
        "secondary_metrics": ["atom-map-normalized exact equality", "raw strict equality", "FTS"],
        "requests_sha256": sha((out / "requests.jsonl").read_bytes()),
        "ground_truth_sha256": sha((out / "ground_truth.jsonl").read_bytes()),
    }
    save_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def validate_frozen(out: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((out / "manifest.json").read_text())
    assert script_sha() == manifest["script_sha256"], "Experiment script changed after prepare"
    assert sha((out / "requests.jsonl").read_bytes()) == manifest["requests_sha256"]
    assert sha((out / "ground_truth.jsonl").read_bytes()) == manifest["ground_truth_sha256"]
    return manifest, read_jsonl(out / "requests.jsonl")


def require_gpu_6_7() -> None:
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    assert visible == ["6", "7"], f"Refusing to run: CUDA_VISIBLE_DEVICES must be exactly 6,7, got {visible}"


def run(args) -> None:
    require_gpu_6_7()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    manifest, requests = validate_frozen(args.output)
    results_path = args.output / "predictions.jsonl"
    existing = read_jsonl(results_path)
    done = {r["request_id"]: r for r in existing}
    assert len(done) == len(existing)
    for request in requests:
        if request["request_id"] in done:
            assert done[request["request_id"]]["request_sha256"] == request["request_sha256"]
    pending = [r for r in requests if r["request_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("All requested rows already complete", flush=True)
        return

    model = manifest["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) + r["assistant_prefix"] for r in pending]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    assert max(lengths) + manifest["sampling"]["max_tokens"] <= manifest["engine"]["max_model_len"]
    runtime = {
        "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": sys.executable,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "vllm", "transformers", "huggingface-hub")},
        "pending": len(pending),
        "prompt_tokens_min_max": [min(lengths), max(lengths)],
        "script_sha256": script_sha(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (args.output / "runtime_history.jsonl").open("a") as handle:
        handle.write(json.dumps(runtime) + "\n")
    save_json(args.output / "runtime.json", runtime)
    print("INITIALIZING " + json.dumps(runtime), flush=True)

    llm = LLM(model=model, seed=manifest["sampling"]["seed"], **manifest["engine"])
    stop_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    stop_ids.discard(None)
    params = SamplingParams(**manifest["sampling"], stop=["</answer>"], stop_token_ids=sorted(stop_ids), include_stop_str_in_output=True)
    started = time.monotonic()
    with results_path.open("a") as handle:
        for offset in range(0, len(pending), manifest["batch_size"]):
            request_batch = pending[offset : offset + manifest["batch_size"]]
            prompt_batch = prompts[offset : offset + manifest["batch_size"]]
            outputs = llm.generate(prompt_batch, params, use_tqdm=False)
            assert len(outputs) == len(request_batch)
            for request, prompt, output in zip(request_batch, prompt_batch, outputs, strict=True):
                pred = output.outputs[0]
                result = {k: request[k] for k in ("request_id", "group", "pair_id", "origin_id", "subtask", "request_sha256")}
                result.update({
                    "rendered_prompt_sha256": sha(prompt.encode()),
                    "script_sha256": script_sha(),
                    "assistant_prefix": request["assistant_prefix"],
                    "generated_text": pred.text,
                    "assistant_response": request["assistant_prefix"] + pred.text,
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(pred.token_ids),
                    "finish_reason": pred.finish_reason,
                    "stop_reason": pred.stop_reason,
                })
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"PROGRESS completed={len(done) + min(offset + len(request_batch), len(pending))}/{len(requests)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    print(f"COMPLETE records={len(read_jsonl(results_path))}", flush=True)


def summarize(args) -> None:
    manifest, requests = validate_frozen(args.output)
    chem = metrics_module()
    truths = {r["pair_id"]: r for r in read_jsonl(args.output / "ground_truth.jsonl")}
    expected = {r["request_id"]: r for r in requests}
    predictions = read_jsonl(args.output / "predictions.jsonl")
    assert len({r["request_id"] for r in predictions}) == len(predictions)
    assert {r["request_id"] for r in predictions} <= set(expected)
    if len(predictions) != len(expected) and not args.allow_partial:
        raise RuntimeError(f"Incomplete: {len(predictions)}/{len(expected)}")

    records = []
    for pred in predictions:
        request = expected[pred["request_id"]]
        assert pred["request_sha256"] == request["request_sha256"]
        assert pred["script_sha256"] == manifest["script_sha256"]
        answer, extraction = extract_answer(pred["assistant_response"])
        gt = truths[pred["pair_id"]]
        normalized_pred = unmap_smiles(answer)
        normalized_gt = unmap_smiles(gt["gt_smiles"])
        valid = bool(normalized_pred)
        scores = {
            "primary_match": int(chem.smiles_match_main_frag(normalized_pred, normalized_gt)),
            "normalized_exact_match": int(chem.smiles_match_exact(normalized_pred, normalized_gt)),
            "raw_exact_match": int(chem.smiles_match_exact(answer, gt["gt_smiles"])),
            "fts": float(chem.fts(normalized_pred, normalized_gt)),
        }
        records.append({
            **pred,
            "predicted_smiles": answer,
            "normalized_predicted_smiles": normalized_pred,
            "gt_smiles": gt["gt_smiles"],
            "answer_extraction": extraction,
            "valid_smiles": valid,
            "has_atom_maps": has_atom_maps(answer),
            **scores,
        })
    records.sort(key=lambda r: r["request_id"])
    with (args.output / "outcome_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def aggregate(items: list[dict]) -> dict:
        n = len(items)
        return {
            "n": n,
            "primary_correct": sum(r["primary_match"] for r in items),
            "primary_accuracy": sum(r["primary_match"] for r in items) / n,
            "normalized_exact_correct": sum(r["normalized_exact_match"] for r in items),
            "normalized_exact_accuracy": sum(r["normalized_exact_match"] for r in items) / n,
            "raw_exact_correct": sum(r["raw_exact_match"] for r in items),
            "raw_exact_accuracy": sum(r["raw_exact_match"] for r in items) / n,
            "mean_fts": statistics.mean(r["fts"] for r in items),
            "valid_smiles": sum(r["valid_smiles"] for r in items),
            "atom_mapped_outputs": sum(r["has_atom_maps"] for r in items),
            "missing_answers": sum(not r["predicted_smiles"] for r in items),
            "truncated": sum(r["finish_reason"] == "length" for r in items),
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in items),
        }

    summary = {"complete": len(records) == len(expected), "n_expected": len(expected), "n_completed": len(records), "groups": {}, "paired_primary_comparisons": {}, "dataset_checks": manifest["dataset_checks"], "script_sha256": script_sha()}
    table = []
    for group in "ABCD":
        items = [r for r in records if r["group"] == group]
        if not items:
            continue
        stats = aggregate(items)
        stats["by_subtask"] = {}
        table.append({"group": group, "subtask": "all", **{k: v for k, v in stats.items() if k != "by_subtask"}})
        for subtask in ("add", "delete", "substitute"):
            subitems = [r for r in items if r["subtask"] == subtask]
            if subitems:
                substats = aggregate(subitems)
                stats["by_subtask"][subtask] = substats
                table.append({"group": group, "subtask": subtask, **substats})
        summary["groups"][group] = stats

    by_pair = collections.defaultdict(dict)
    for row in records:
        by_pair[row["pair_id"]][row["group"]] = row
    for lhs, rhs in (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B")):
        pairs = [p for p in by_pair.values() if lhs in p and rhs in p]
        if pairs:
            summary["paired_primary_comparisons"][f"{lhs}_vs_{rhs}"] = {
                "n": len(pairs),
                "both_correct": sum(p[lhs]["primary_match"] and p[rhs]["primary_match"] for p in pairs),
                "lhs_only_correct": sum(p[lhs]["primary_match"] and not p[rhs]["primary_match"] for p in pairs),
                "rhs_only_correct": sum(p[rhs]["primary_match"] and not p[lhs]["primary_match"] for p in pairs),
                "both_wrong": sum(not p[lhs]["primary_match"] and not p[rhs]["primary_match"] for p in pairs),
            }
    save_json(args.output / "summary.json", summary)
    with (args.output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(summary, indent=2), flush=True)


def self_check() -> None:
    assert extract_answer("<think>\n</think>\n<answer>\nCCO</answer>") == ("CCO", "answer_tag")
    assert extract_answer("<think>CCN</think><answer>CCO</answer>")[0] == "CCO"
    assert extract_answer("<think>Answer: CCN")[0] == ""
    text = 'FORMAL: X --> PRODUCT_SMILES("CCO") and repeated CCO'
    redacted = redact_product(text, {"CCO"})
    assert "CCO" not in redacted and redacted.count(REDACTION) == 2
    assert unmap_smiles("[CH3:1][CH2:2][OH:3]") == "CCO"
    assert has_atom_maps("[CH3:1][CH2:2][OH:3]") and not has_atom_maps("OCC")
    print("PASS: extraction, leakage redaction, and atom-map normalization")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize", "self-check"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Pilot/GeneratedDataset/maximum_edits_complete.jsonl")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--pairs-per-subtask", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0
    if args.pairs_per_subtask is not None:
        assert args.pairs_per_subtask > 0
    if args.command == "self-check":
        self_check()
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
