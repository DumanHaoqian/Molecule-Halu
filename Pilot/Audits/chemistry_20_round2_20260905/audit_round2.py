"""Second predeclared, disjoint sample; reuse the independent RDKit checker.

Run with molhallulens Python. No production imports or network calls.
The first audit's source and evidence are dependencies and remain unchanged.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random

from rdkit import rdBase

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "chemistry_20_20260905"
DATA = HERE.parents[1] / "Dataset"
SEED = 20260906  # Random seed, not a claimed audit date.
ALLOCATION = {"add": 7, "delete": 7, "substitute": 6}
spec = importlib.util.spec_from_file_location("independent_round1", PREVIOUS / "audit.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def run():
    previous = json.loads((PREVIOUS / "evidence.json").read_text())
    excluded = {r["origin_id"] for r in previous["records"]}
    assert len(excluded) == 20
    rng = random.Random(SEED)
    records, population, hashes = [], [], {}
    selection = []
    # Select all IDs before reading selected process traces or chemical results.
    for task, number in ALLOCATION.items():
        raw_path = DATA / "raw_benchmark_data/mol_edit" / f"{task}_pilot_origin.json"
        process_path = DATA / "process_evaluation_data/mol_edit" / f"{task}_pilot_origin.json"
        raw_rows = json.loads(raw_path.read_text())
        proc_rows = json.loads(process_path.read_text())
        raw = {r["anonymous_sample_id"]: r for r in raw_rows}
        proc = {r["anonymous_sample_id"]: r for r in proc_rows}
        assert len(raw) == len(raw_rows) == len(proc) == len(proc_rows) == 50
        assert set(raw) == set(proc)
        eligible = sorted(set(raw) - excluded)
        ids = sorted(rng.sample(eligible, number))
        selection.extend((raw[x], proc[x]) for x in ids)
        population.append({"task": task, "original_available": 50, "remaining": len(eligible), "selected": ids})
        for path in (raw_path, process_path):
            relative = str(path.relative_to(DATA.parent))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashes[relative] == previous["source_sha256"][relative], "Source changed since round 1"
    assert sum(x["remaining"] for x in population) == 130
    with rdBase.BlockLogs():
        for raw, proc in selection:
            records.append(checker.inspect(raw, proc))
    assert not excluded.intersection(r["origin_id"] for r in records)
    evidence = {"round": 2, "seed": SEED, "rdkit_version": rdBase.rdkitVersion,
                "sampling": "7/7/6 from sorted remaining IDs, same seeded RNG, task order add/delete/substitute; no swaps",
                "excluded_ids": sorted(excluded), "population": population,
                "source_sha256": hashes, "previous_checker_sha256": hashlib.sha256((PREVIOUS / "audit.py").read_bytes()).hexdigest(),
                "previous_evidence_sha256": hashlib.sha256((PREVIOUS / "evidence.json").read_bytes()).hexdigest(),
                "records": records}
    (HERE / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    paragraphs = []
    for r in records:
        paragraphs.extend(["="*85, r["origin_id"], "INSTRUCTION: " + r["raw"]["instruction"],
                           "SOURCE: " + r["raw"]["indexed_smiles"], "ANSWER: " + r["raw"]["gt_smiles"],
                           "ANCHOR: " + json.dumps(r["anchor"]),
                           "COUNTS: " + json.dumps([r["source_computed"], r["answer_computed"]]),
                           "RINGS: " + json.dumps(r["ring_details"]),
                           "GRAPH: " + json.dumps(r["graph_reconstruction"])])
        paragraphs.extend(s["step_text"] for s in r["process"]["formal_cot_trace"])
    (HERE / "sample_review.txt").write_text("\n\n".join(paragraphs)+"\n")
    summary = {"origins": len(records), "overlap_with_round1": 0,
               "steps": sum(len(r["process"]["formal_cot_trace"]) for r in records),
               "checks": sum(len(r["checks"])+len(r["fragment_checks"]) for r in records),
               "failed_checks": [{"origin_id":r["origin_id"], "check":c} for r in records for c in r["checks"]+r["fragment_checks"] if not c["pass"]],
               "reconstructed": sum(r["graph_reconstruction"]["exact_isomeric_answer_recovered"] for r in records),
               "selected": [r["origin_id"] for r in records]}
    print(json.dumps(summary, indent=2))
    return evidence


if __name__ == "__main__":
    run()
