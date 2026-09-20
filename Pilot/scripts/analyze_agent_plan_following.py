#!/usr/bin/env python3
"""CPU-only post-hoc comparison of model outputs with clean and corrupted plans.

Reads completed outcome_records.jsonl files (or explicit prediction snapshots),
never calls models, selects examples, or modifies evaluation inputs. Full graph
and largest-heavy-component comparisons retain stereochemistry and ignore atom
maps. A product match is evidence of output agreement with a plan, not proof of
the model's internal reasoning or of causal mediation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from rdkit import Chem, rdBase


def _canonical(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    params = Chem.SmilesParserParams()
    params.parseName = False
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromSmiles(smiles, params)
        except (ValueError, RuntimeError):
            return None
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    full = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    # The main evaluator selects the first largest fragment after full
    # canonicalization, so equal-size ties use canonical component order.
    components = [(Chem.MolFromSmiles(piece).GetNumHeavyAtoms(), piece) for piece in full.split(".")]
    primary = sorted(components, key=lambda pair: (-pair[0], pair[1]))[0][1]
    return {"full": full, "primary": primary}


def compare_prediction(predicted_smiles, ground_truth, clean_product, corrupted_product):
    references = {"ground_truth": _canonical(ground_truth), "clean_plan": _canonical(clean_product),
                  "executed_h": _canonical(corrupted_product)}
    if any(value is None for value in references.values()):
        raise ValueError("Invalid saved reference or executed-plan molecule")
    prediction = _canonical(predicted_smiles)
    matches = {name: {scope: bool(prediction and prediction[scope] == molecule[scope])
                      for scope in ("full", "primary")} for name, molecule in references.items()}

    def classify(scope):
        if prediction is None:
            return "invalid_output"
        truth, wrong = matches["ground_truth"][scope], matches["executed_h"][scope]
        if truth and wrong:
            return "ground_truth_and_executed_h"
        if truth:
            return "ground_truth_only"
        return "executed_h_only" if wrong else "neither"

    return {"predicted_smiles": predicted_smiles, "valid_smiles": prediction is not None,
            "canonical_prediction": prediction, "canonical_targets": references, "matches": matches,
            "full_classification": classify("full"), "primary_classification": classify("primary")}


def summarize(rows):
    result = {}
    for group in sorted({row["group"] for row in rows}):
        items = [row for row in rows if row["group"] == group]
        result[group] = {"n": len(items), "valid_outputs": sum(row["valid_smiles"] for row in items),
                         "full_classifications": dict(Counter(row["full_classification"] for row in items)),
                         "primary_classifications": dict(Counter(row["primary_classification"] for row in items)),
                         "matches": {}}
        for target in ("ground_truth", "clean_plan", "executed_h"):
            values = {}
            for scope in ("full", "primary"):
                count = sum(row["matches"][target][scope] for row in items)
                values.update({scope + "_count": count, scope + "_rate": count / len(items)})
            result[group]["matches"][target] = values
    return result


def _answer(record):
    if "predicted_smiles" in record:
        return record["predicted_smiles"]
    response = record.get("assistant_response", "")
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.S | re.I)
    if blocks:
        value = blocks[-1].strip()
    else:
        if "<think>" in response and "</think>" not in response:
            return ""
        tail = response.rsplit("</think>", 1)[-1]
        labels = re.findall(r"^\s*(?:Final\s+)?Answer:\s*(\S+)\s*$", tail, re.M | re.I)
        if not labels:
            return ""
        value = labels[-1]
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:smiles)?\s*|\s*```$", "", value, flags=re.I).strip()
    value = value.strip("`\"'")
    return value if value and not re.search(r"\s|<|>", value) else ""


def analyze(batch_directory: Path, evaluation_directories: list[Path], groups="BC"):
    references = {}
    for path in sorted((batch_directory / "origins").glob("*/accepted.json")):
        accepted = json.loads(path.read_text())
        row = json.loads((path.parent / "input.json").read_text())["row"]
        clean = json.loads((path.parent / "reference.json").read_text())["execution"]["product_smiles"]
        references[accepted["origin_id"]] = {"pair_id": accepted["pair_id"], "gt": row["gt_smiles"],
            "clean": clean, "h": accepted["plan"]["execution"]["product_smiles"], "accepted_path": str(path.resolve())}
    if not references:
        raise ValueError("No accepted pairs in batch directory")
    records, models = [], {}
    for directory in evaluation_directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        model = manifest["model"]["name"]
        if model in models:
            raise ValueError(f"Duplicate model name: {model}")
        path = directory / "outcome_records.jsonl"
        if not path.exists():
            path = directory / "predictions.jsonl"
        if not path.exists():
            models[model] = {"status": "pending", "groups": {}}
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        seen = set()
        selected = []
        for record in rows:
            if record["request_id"] in seen:
                raise ValueError(f"Duplicate request ID in {path}")
            seen.add(record["request_id"])
            if record["group"] not in groups:
                continue
            reference = references[record["origin_id"]]
            if reference["pair_id"] != record["pair_id"]:
                raise ValueError("Outcome pair ID differs from accepted artifact")
            prediction = compare_prediction(_answer(record), reference["gt"], reference["clean"], reference["h"])
            prediction.update({key: record[key] for key in ("request_id", "group", "pair_id", "origin_id", "subtask")})
            prediction.update(model=model, source_path=str(path.resolve()), accepted_path=reference["accepted_path"])
            selected.append(prediction)
        records.extend(selected)
        expected = manifest["n_pairs"] * len(set(groups) & set(manifest["groups"]))
        models[model] = {"status": "complete" if len(selected) == expected else "partial",
                         "source": str(path.resolve()), "n_expected": expected, "n_observed": len(selected),
                         "groups": summarize(selected)}
    return {"analysis": "agent_plan_following_v1", "batch": str(batch_directory.resolve()),
            "scope": "Post-hoc output agreement; no model calls or example selection",
            "interpretation": "Matching the executed H product supports output agreement with the wrong edit; it does not identify internal reasoning or prove mediation.",
            "metrics": "Full isomeric graph equality and largest-heavy-component equality after atom-map removal; no tautomer or charge normalization.",
            "models": models, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--evaluation-dir", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--groups", default="BC")
    args = parser.parse_args()
    directories = args.evaluation_dir or sorted(path.parent for path in (args.batch_dir / "evaluation").glob("*/manifest.json"))
    report = analyze(args.batch_dir, directories, args.groups)
    output = args.output or args.batch_dir / "review" / "plan_following.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Model agreement with executed plans", "", report["interpretation"], "",
             "| Model | Status | Group | N | GT full/primary | Clean full/primary | H plan full/primary |", "|---|---|---|---:|---:|---:|---:|"]
    for model, details in report["models"].items():
        for group, stats in details["groups"].items():
            counts = [f"{stats['matches'][target]['full_count']}/{stats['matches'][target]['primary_count']}" for target in ("ground_truth", "clean_plan", "executed_h")]
            lines.append(f"| {model} | {details['status']} | {group} | {stats['n']} | " + " | ".join(counts) + " |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "models": report["models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
