#!/usr/bin/env python3
"""Detailed paired and mechanism analysis for the causal hallucination dataset."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics

from rdkit import Chem


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_mcnemar(lhs_only: int, rhs_only: int) -> float:
    n = lhs_only + rhs_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(lhs_only, rhs_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def has_fragment(smiles: str, fragment: str) -> bool:
    molecule, query = mol(smiles), mol(fragment)
    return bool(molecule is not None and query is not None and molecule.HasSubstructMatch(query))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output
    requests = read_jsonl(directory / "requests.jsonl")
    predictions = read_jsonl(directory / "predictions.jsonl")
    outcomes = read_jsonl(directory / "outcome_records.jsonl")
    interventions = {row["pair_id"]: row for row in read_jsonl(directory / "interventions.jsonl")}

    request_by_id = {row["request_id"]: row for row in requests}
    prediction_by_id = {row["request_id"]: row for row in predictions}
    outcome_by_id = {row["request_id"]: row for row in outcomes}
    if not (
        len(requests) == len(request_by_id)
        and len(predictions) == len(prediction_by_id)
        and len(outcomes) == len(outcome_by_id)
        and set(request_by_id) == set(prediction_by_id) == set(outcome_by_id)
    ):
        raise RuntimeError("Request/prediction/outcome integrity failure")
    if not all(prediction_by_id[key]["request_sha256"] == request_by_id[key]["request_sha256"] for key in request_by_id):
        raise RuntimeError("Request hash mismatch")

    by_pair = collections.defaultdict(dict)
    for row in outcomes:
        by_pair[row["pair_id"]][row["group"]] = row
    if not all(set(rows) == {"A", "N", "H", "L"} for rows in by_pair.values()):
        raise RuntimeError("Incomplete paired groups")

    comparisons = {}
    for scope in ("all", "add", "delete", "substitute"):
        pairs = [
            rows for rows in by_pair.values()
            if scope == "all" or rows["A"]["subtask"] == scope
        ]
        comparisons[scope] = {}
        for lhs, rhs in (("N", "A"), ("H", "N"), ("L", "H"), ("L", "N"), ("H", "A")):
            lhs_only = sum(rows[lhs]["primary_match"] and not rows[rhs]["primary_match"] for rows in pairs)
            rhs_only = sum(rows[rhs]["primary_match"] and not rows[lhs]["primary_match"] for rows in pairs)
            comparisons[scope][f"{lhs}_vs_{rhs}"] = {
                "n": len(pairs),
                "accuracy_delta_pp": 100 * (lhs_only - rhs_only) / len(pairs),
                "lhs_only_correct": lhs_only,
                "rhs_only_correct": rhs_only,
                "exact_mcnemar_p": exact_mcnemar(lhs_only, rhs_only),
            }

    prompt_tokens = {}
    for group in ("A", "N", "H", "L"):
        values = [row["input_tokens"] for row in predictions if row["group"] == group]
        prompt_tokens[group] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
    nh_deltas = [rows["H"]["input_tokens"] - rows["N"]["input_tokens"] for rows in by_pair.values()]
    prompt_tokens["H_minus_N_paired"] = {
        "mean": statistics.mean(nh_deltas),
        "median": statistics.median(nh_deltas),
        "min": min(nh_deltas),
        "max": max(nh_deltas),
        "equal": sum(value == 0 for value in nh_deltas),
    }

    adoption = {}
    for subtask in ("add", "substitute"):
        eligible = []
        counts = collections.Counter()
        for pair_id, rows in by_pair.items():
            if rows["A"]["subtask"] != subtask:
                continue
            wrong = interventions[pair_id]["wrong_fragment"]
            if has_fragment(rows["A"]["lure_smiles"], wrong) or has_fragment(rows["A"]["gt_smiles"], wrong):
                continue
            eligible.append(pair_id)
            for group in ("A", "N", "H", "L"):
                counts[group] += has_fragment(rows[group]["normalized_predicted_smiles"], wrong)
        adoption[subtask] = {"eligible": len(eligible), "wrong_fragment_substructure_matches": dict(counts)}

    delete_pairs = [
        (pair_id, rows) for pair_id, rows in by_pair.items()
        if rows["A"]["subtask"] == "delete"
    ]
    adoption["delete"] = {
        "n": len(delete_pairs),
        "wrong_remove_group_present_in_source": sum(
            has_fragment(rows["A"]["lure_smiles"], interventions[pair_id]["wrong_fragment"])
            for pair_id, rows in delete_pairs
        ),
    }

    release_rows = []
    quality_counts = collections.Counter()
    for pair_id, intervention in sorted(interventions.items()):
        rows = by_pair[pair_id]
        if intervention["subtask"] in {"add", "substitute"}:
            quality_tier = (
                "primary_matched_fragment"
                if intervention["same_heavy_atom_count"] and intervention["same_first_element"]
                else "secondary_fragment"
            )
        else:
            present = has_fragment(rows["A"]["lure_smiles"], intervention["wrong_fragment"])
            quality_tier = "diagnostic_delete_present" if present else "diagnostic_delete_conflict"
        quality_counts[quality_tier] += 1
        release_rows.append({
            "pair_id": pair_id,
            "origin_id": intervention["origin_id"],
            "subtask": intervention["subtask"],
            "quality_tier": quality_tier,
            "question": request_by_id[f"A:{pair_id}"]["messages"][1]["content"],
            "normal_rationale": intervention["normal_rationale"],
            "hallucinated_rationale": intervention["hallucinated_rationale"],
            "intervention_field": intervention["intervention_field"],
            "correct_fragment": intervention[intervention["intervention_field"]],
            "wrong_fragment": intervention["wrong_fragment"],
            "same_heavy_atom_count": intervention["same_heavy_atom_count"],
            "same_first_element": intervention["same_first_element"],
            "source_smiles": rows["A"]["lure_smiles"],
            "gt_smiles": rows["A"]["gt_smiles"],
        })
    with (directory / "hallucination_dataset.jsonl").open("w") as handle:
        for row in release_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    primary_ids = {
        row["pair_id"] for row in release_rows
        if row["quality_tier"] == "primary_matched_fragment"
    }
    primary_n_only = sum(
        by_pair[pair_id]["N"]["primary_match"] and not by_pair[pair_id]["H"]["primary_match"]
        for pair_id in primary_ids
    )
    primary_h_only = sum(
        by_pair[pair_id]["H"]["primary_match"] and not by_pair[pair_id]["N"]["primary_match"]
        for pair_id in primary_ids
    )

    result = {
        "integrity": {
            "requests": len(requests),
            "predictions": len(predictions),
            "outcomes": len(outcomes),
            "all_request_hashes_match": True,
            "finish_reasons": dict(collections.Counter(row["finish_reason"] for row in predictions)),
            "valid_smiles": sum(row["valid_smiles"] for row in outcomes),
            "truncated": sum(row["finish_reason"] == "length" for row in outcomes),
            "max_output_tokens": max(row["output_tokens"] for row in predictions),
        },
        "groups": {
            group: {
                "n": sum(row["group"] == group for row in outcomes),
                "correct": sum(row["primary_match"] for row in outcomes if row["group"] == group),
                "accuracy": statistics.mean(row["primary_match"] for row in outcomes if row["group"] == group),
                "mean_fts": statistics.mean(row["fts"] for row in outcomes if row["group"] == group),
                "lure_matches": sum(row["lure_match"] for row in outcomes if row["group"] == group),
            }
            for group in ("A", "N", "H", "L")
        },
        "paired_comparisons": comparisons,
        "output_identity": {
            "H_equals_N": sum(rows["H"]["generated_text"] == rows["N"]["generated_text"] for rows in by_pair.values()),
            "L_equals_H": sum(rows["L"]["generated_text"] == rows["H"]["generated_text"] for rows in by_pair.values()),
        },
        "prompt_tokens": prompt_tokens,
        "fragment_adoption": adoption,
        "intervention_quality": {
            "same_heavy_atom_count": sum(row["same_heavy_atom_count"] for row in interventions.values()),
            "same_first_element": sum(row["same_first_element"] for row in interventions.values()),
            "n": len(interventions),
            "quality_tiers": dict(quality_counts),
            "recommended_primary_H_vs_N": {
                "n": len(primary_ids),
                "N_correct": sum(by_pair[pair_id]["N"]["primary_match"] for pair_id in primary_ids),
                "H_correct": sum(by_pair[pair_id]["H"]["primary_match"] for pair_id in primary_ids),
                "H_minus_N_accuracy_pp": 100 * (primary_h_only - primary_n_only) / len(primary_ids),
                "H_only_correct": primary_h_only,
                "N_only_correct": primary_n_only,
                "exact_mcnemar_p": exact_mcnemar(primary_h_only, primary_n_only),
            },
        },
    }
    (directory / "causal_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
