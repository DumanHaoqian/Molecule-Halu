#!/usr/bin/env python3
"""Derive an unfiltered, diagnostic-only numeric-root ablation on CPU.

Every parent accepted pair is retained. Molecular edit plans and products stay
fixed, N stays byte-identical, and numeric claim roots are removed before typed
claims/rendering/labels are rebuilt. Density exceptions are reported, never
filtered. No model, network, API, or GPU execution occurs in this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from molhallulens.modules.agent_generation.chemistry_tools import apply_edit_plan, compare_molecules, inspect_source
from molhallulens.modules.agent_generation.labels import label_tokens
from molhallulens.modules.agent_generation.orchestrator import build_nodes, semantic_roots
from molhallulens.modules.agent_generation.quality import find_answer_leakage, validate_formal_claims
from molhallulens.modules.agent_generation.renderer import render_reasoning


PROTOCOL = "agent_structural_ablation_v1"
MODEL_NAMES = ("ChemDFM-R-14B", "Chem-R-8B")
STATUS_MEANING = "accepted_for_paired_diagnostic_only"


class FrozenTokenizer:
    """Minimal fast-tokenizer interface backed by the parent's exact snapshot."""
    is_fast = True

    def __init__(self, backend: str, name: str):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(backend)
        self.name_or_path = name

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=True):
        encoded = self.backend.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

    def convert_ids_to_tokens(self, ids):
        return [self.backend.id_to_token(token_id) for token_id in ids]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def derive_record(parent, parent_input, reference, *, tokenizers, parent_directory):
    """Return a fresh accepted-for-diagnostic record without mutating inputs."""
    _require(set(tokenizers) == set(MODEL_NAMES), "Both frozen tokenizers are required")
    _require(parent.get("status") == "accepted", "Only accepted parent pairs can be derived")
    row = parent_input["row"]
    parent_directory = Path(parent_directory).resolve()
    pairs = {p["variant_label"]: p for p in parent["pairs"]}
    _require(len(parent["pairs"]) == 2 and set(pairs) == {"N", "H"}, "Parent must have exactly one N/H pair")
    for pair in pairs.values():
        visible = pair["detector_input"]
        _require(pair["origin_id"] == row["origin_id"] == parent["origin_id"], "Parent origin mismatch")
        _require(pair["pair_id"] == parent["pair_id"] and pair["subtask"] == row["subtask"], "Parent pair metadata mismatch")
        for key, target in (("indexed_smiles", "indexed_smiles"), ("instruction", "instruction"), ("final_answer", "gt_smiles")):
            _require(visible[key] == row[target], "Parent question or GT mismatch: " + key)
        for key in ("indexed_smiles", "instruction", "gt_smiles"):
            _require(row[key] == row["raw_record"][key], "Parent raw benchmark mismatch: " + key)

    clean_plan, plan = reference["edit_plan"], parent["plan"]["edit_plan"]
    clean = apply_edit_plan(row["indexed_smiles"], clean_plan)
    altered = apply_edit_plan(row["indexed_smiles"], plan)
    _require(clean == reference["execution"], "Parent clean execution no longer matches")
    _require(altered == parent["plan"]["execution"], "Parent H execution no longer matches")
    _require(compare_molecules(clean["product_smiles"], row["gt_smiles"])["equivalent"], "Clean graph differs from benchmark")
    _require(not compare_molecules(clean["product_smiles"], altered["product_smiles"])["equivalent"], "H must retain a different structural product")
    roots = deepcopy([r for r in parent["plan"]["roots"] if r["type"] == "structural"])
    removed = deepcopy([r for r in parent["plan"]["roots"] if r["type"] == "numeric_claim"])
    _require(len(roots) + len(removed) == len(parent["plan"]["roots"]) and roots, "Unsupported or absent structural roots")
    strip_id = lambda items: [{k: v for k, v in item.items() if k != "id"} for item in items]
    _require(strip_id(roots) == strip_id(semantic_roots(clean_plan, plan)), "Parent structural roots differ from edit plans")

    clean_nodes = build_nodes(row, clean_plan, clean_plan, [], clean, clean, include_local_environment=True)
    nodes = build_nodes(row, clean_plan, plan, roots, clean, altered, include_local_environment=True)
    nr = render_reasoning(row["indexed_smiles"], clean_plan, clean_nodes, label_errors=False)
    hr = render_reasoning(row["indexed_smiles"], plan, nodes, label_errors=True)
    _require(nr["text"] == parent["N_visible"] == pairs["N"]["detector_input"]["reasoning_chain"], "Recomputed N is not byte-identical to parent")
    if not removed:
        _require(hr["text"] == pairs["H"]["detector_input"]["reasoning_chain"], "Nonnumeric control H unexpectedly changed")
    for name in ("local_environment", "local_environment_center"):
        _require(nodes[name]["before"] == parent["plan"]["nodes"][name]["before"] and
                 nodes[name]["after"] == parent["plan"]["nodes"][name]["after"], "Structural local evidence changed")
    source = inspect_source(row["indexed_smiles"])
    _require(nodes["source_heavy"]["after"] == source["heavy_atoms"] and
             nodes["product_heavy"]["after"] == altered["heavy_atoms"], "Numeric claims did not return to molecular facts")
    checks = {}
    for label, rendered, approved in (("N", nr, clean_nodes), ("H", hr, nodes)):
        facts = validate_formal_claims(rendered["text"], approved, source)
        _require(facts["status"] == "pass", label + " formal claims failed")
        leaks = find_answer_leakage(rendered["text"], [row["gt_smiles"], altered["product_smiles"]])
        _require(not leaks, label + " answer leakage: " + str(leaks))
        checks[label] = {"formal_claims": facts, "answer_leakage": leaks}
    observed_roots = {r for s in hr["spans"] if s["kind"] == "root_error" for r in s["root_ids"]}
    _require(observed_roots == {r["id"] for r in roots}, "Structural root annotation incomplete")
    removed_ids = {r["id"] for r in removed}
    _require(not any(removed_ids.intersection(n["root_ids"]) for n in nodes.values()), "Stale numeric ancestry remains")
    labels = {name: label_tokens(hr["text"], hr["spans"], tok) for name, tok in tokenizers.items()}
    counts = {name: sum(any(x != "unchanged" for x in t["labels"]) for t in ts) for name, ts in labels.items()}
    wrong_nodes = {s["node_id"] for s in hr["spans"]}
    thresholds = {key: parent_input[key] for key in ("min_roots", "min_nodes", "min_tokens")}
    violations = []
    if len(wrong_nodes) < thresholds["min_nodes"]:
        violations.append("min_nodes")
    if min(counts.values()) < thresholds["min_tokens"]:
        violations.append("min_tokens")
    all_violations = (["min_roots"] if len(roots) < thresholds["min_roots"] else []) + violations
    new_id = row["origin_id"] + "__" + PROTOCOL
    result = deepcopy(parent)
    new_pairs = []
    for label, rendered in (("N", nr), ("H", hr)):
        pair = deepcopy(pairs[label])
        pair.update(pair_id=new_id, record_id=new_id + "__" + label,
                    edit_count=len(roots) if label == "H" else 0)
        pair["detector_input"]["reasoning_chain"] = rendered["text"]
        new_pairs.append(pair)
    result.update(protocol=PROTOCOL, pair_id=new_id, pairs=new_pairs,
                  status="accepted", acceptance_scope="paired_diagnostic_only",
                  N_visible=nr["text"], N_semantic_bindings=nr["bindings"],
                  annotations=hr["spans"], semantic_bindings=hr["bindings"], text_edits=hr["edits"],
                  token_labels=labels, actual_api_calls=0,
                  error_counts={"roots": len(roots), "distinct_wrong_nodes": len(wrong_nodes),
                                "error_spans": len(hr["spans"]), "tokens": counts})
    result["plan"].update(roots=roots, nodes=nodes, render=hr, checks=checks,
                          sampling_policy="fixed_parent_plan_numeric_root_removal_no_resampling")
    for key in ("selection_probability", "seed", "min_roots"):
        result["plan"].pop(key, None)
    result["density_report"] = {"parent_thresholds": thresholds, "violations": violations,
                                "parent_threshold_violations": all_violations, "filtered": False}
    result["release_contract"] = {"version": PROTOCOL, "status": "diagnostic_only",
        "diagnostic_only": True, "production_eligible": False, "status_meaning": STATUS_MEANING,
        "program_gates": "Same executed structural plans/products; byte-identical N; original question/GT; true molecular counts; typed rendering/labels; no full answer",
        "density_policy": "All parent accepted pairs retained; original thresholds reported, never used to filter",
        "model_outputs_used_for_selection": False}
    result["historical_parent_reviews"] = {key: deepcopy(parent.get(key)) for key in
        ("reference_review", "blind_audit", "audit", "review_status")}
    result["reference_review_provenance"] = "Historical parent clean review; N is byte-identical; no fresh call"
    result["audit"] = {"status": "unknown_not_repeated", "role": "No fresh conformance review"}
    result["blind_audit"] = {"status": "unknown_not_repeated", "answer_leakage": None}
    result["review_status"] = {"role": "historical_parent_reviews_only", "reference_status": "inherited_same_N",
        "conformance_status": "unknown_not_repeated", "blind_status": "unknown_not_repeated",
        "agent_consensus": None, "interpretation": "No LLM review was repeated for this diagnostic derivation"}
    result["derivation"] = {"parent_protocol": parent["protocol"], "parent_pair_id": parent["pair_id"],
        "parent_origin_directory": str(parent_directory), "parent_accepted_path": str(parent_directory / "accepted.json"),
        "parent_input_path": str(parent_directory / "input.json"), "parent_reference_path": str(parent_directory / "reference.json"),
        "parent_request_directory": str(parent_directory / "plan_calls"),
        "parent_actual_api_calls": parent.get("actual_api_calls", 0), "derivation_api_calls": 0,
        "removed_numeric_roots": removed, "unchanged_structural_plan": True,
        "unchanged_executed_product": True, "unchanged_clean_trace": True,
        "parent_selection_metadata": {k: deepcopy(parent["plan"].get(k)) for k in
                                      ("candidate_id", "seed", "selection_probability", "sampling_policy")}}
    return result


def _read_json(path):
    return json.loads(path.read_text())


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def materialize(parent_batch, output):
    """Validate every parent pair, then atomically publish a fresh CPU artifact."""
    parent_batch, output = Path(parent_batch).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh ablation output directory: " + str(output))
    _require(output != parent_batch and parent_batch not in output.parents, "Output must be outside the parent batch")
    parent_manifest = _read_json(parent_batch / "manifest.json")
    parent_summary = _read_json(parent_batch / "summary.json")
    snapshots = _read_json(parent_batch / "tokenizer_snapshots.json")
    _require(set(snapshots) == set(MODEL_NAMES), "Parent frozen tokenizer set is incomplete")
    original_pairs = [json.loads(x) for x in (parent_batch / "pairs.jsonl").read_text().splitlines() if x.strip()]
    origins = sorted({p["origin_id"] for p in original_pairs})
    _require(origins and len(original_pairs) == len(origins) * 2, "Parent dataset must contain exactly one pair per origin")
    _require(parent_summary["accepted"] == len(origins), "Parent accepted count differs from dataset")
    found = sorted(p.parent.name for p in (parent_batch / "origins").glob("*/accepted.json"))
    _require(found == origins, "Parent accepted records and dataset origins differ")
    tokenizers = {name: FrozenTokenizer(snapshots[name]["backend"], name) for name in MODEL_NAMES}
    results, sources = [], {}
    for origin in origins:
        directory = parent_batch / "origins" / origin
        parent, inp, reference = (_read_json(directory / (name + ".json")) for name in ("accepted", "input", "reference"))
        dataset_rows = [p for p in original_pairs if p["origin_id"] == origin]
        _require(sorted(dataset_rows, key=lambda x: x["variant_label"]) == sorted(parent["pairs"], key=lambda x: x["variant_label"]), "Parent dataset differs from accepted record")
        for name, tokenizer in tokenizers.items():
            prior = parent.get("token_labels", {}).get(name, [])
            tokenizer.name_or_path = prior[0]["tokenizer"] if prior else name
        results.append(derive_record(parent, inp, reference, tokenizers=tokenizers, parent_directory=directory))
        sources[origin] = {"accepted": parent, "input": inp, "reference": reference}
    old_order = [p["origin_id"] for p in sorted(original_pairs, key=lambda x: x["pair_id"]) if p["variant_label"] == "N"]
    _require(old_order == origins, "New pair suffix would change parent donor ordering")
    under = Counter(v for r in results for v in r["density_report"]["parent_threshold_violations"])
    density_origins = [r["origin_id"] for r in results if r["density_report"]["violations"]]
    common = {"protocol": PROTOCOL, "diagnostic_only": True, "status_meaning": STATUS_MEANING,
              "parent_batch": str(parent_batch), "source_planned": parent_summary["planned"],
              "source_accepted": parent_summary["accepted"], "source_rejected": parent_summary["rejected"],
              "actual_api_calls": 0, "production_accepted": 0,
              "density_violation_origins": density_origins, "parent_threshold_violation_counts": dict(under)}
    manifest = {**common, "selected": origins, "parent_manifest": parent_manifest,
                "selection_note": "All parent accepted pairs retained; no filtering, resampling, or target-model outputs",
                "e_donor_policy": "Full-dataset sorted circular assignment; same origin order and N preserve parent donor origins",
                "parent_origin_order": old_order, "e_donor_origin_preserved_by_order": True}
    summary = {**common, "planned": len(origins), "processed": len(results), "accepted": len(results), "rejected": 0,
               "accepted_for_paired_diagnostic_only": len(results),
               "results": [{key: r[key] for key in ("origin_id", "status", "acceptance_scope", "error_counts", "density_report", "review_status")} for r in results]}
    implementation = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "molhallulens/modules/agent_generation").glob("*.py"))}
    implementation["scripts/derive_agent_ablation.py"] = Path(__file__).read_text()
    implementation["tests/test_agent_ablation.py"] = (ROOT / "tests/test_agent_ablation.py").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "summary.json", summary)
        _write_json(staging / "implementation_snapshot.json", implementation)
        for filename, destination in (("tokenizer_snapshots.json", "tokenizer_snapshots.json"),
                                      ("implementation_snapshot.json", "parent_implementation_snapshot.json"),
                                      ("manifest.json", "parent_manifest.json"), ("summary.json", "parent_summary.json")):
            shutil.copyfile(parent_batch / filename, staging / destination)
        with (staging / "pairs.jsonl").open("w") as handle:
            for result in results:
                origin = result["origin_id"]
                directory = staging / "origins" / origin
                source = sources[origin]
                _write_json(directory / "accepted.json", result)
                _write_json(directory / "reference.json", source["reference"])
                _write_json(directory / "input.json", {"protocol": PROTOCOL, "row": source["input"]["row"],
                    "parent_input_path": result["derivation"]["parent_input_path"], "parent_thresholds": result["density_report"]["parent_thresholds"],
                    "derivation": "Remove numeric roots only; all structural plans fixed"})
                _write_json(directory / "parent_record_snapshot.json", source)
                _write_json(directory / "fixed_plan.json", result["plan"])
                request_dir = parent_batch / "origins" / origin / "plan_calls"
                if request_dir.exists():
                    _write_json(directory / "historical_parent_requests.json", {p.name: _read_json(p) for p in sorted(request_dir.glob("*.json"))})
                for pair in result["pairs"]:
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = materialize(args.parent_batch, args.output)
    print(json.dumps({key: summary[key] for key in ("protocol", "accepted", "production_accepted", "actual_api_calls", "density_violation_origins")}, indent=2))


if __name__ == "__main__":
    main()
