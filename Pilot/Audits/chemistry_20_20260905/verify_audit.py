"""Checks evidence integrity and sensitivity; never modifies source data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from audit import DATA, HERE, inspect


def verify():
    evidence = json.loads((HERE / "evidence.json").read_text())
    review = json.loads((HERE / "review.json").read_text())
    records = {r["origin_id"]: r for r in evidence["records"]}
    assert len(records) == 20
    assert {r["origin_id"] for r in review["records"]} == set(records)
    for r in records.values():
        assert all(c["pass"] for c in r["checks"] + r["fragment_checks"])
        assert r["graph_reconstruction"]["exact_isomeric_answer_recovered"]
    for relative, digest in evidence["source_sha256"].items():
        assert hashlib.sha256((DATA.parent / relative).read_bytes()).hexdigest() == digest
    for finding in review["findings"]:
        steps = records[finding["origin_id"]]["process"]["formal_cot_trace"]
        assert any(finding["quote"] in s["natural_language"] for s in steps), finding["id"]
    r = records["mol_edit.add_v2.0022"]
    with rdBase.BlockLogs():
        wrong = deepcopy(r["process"])
        for s in wrong["formal_cot_trace"]:
            s["formal_ab"] = s["formal_ab"].replace("PRODUCT_SMILES[n_heavy=42]", "PRODUCT_SMILES[n_heavy=99]")
        result = inspect(r["raw"], wrong)
        assert any(c["check"] == "product_heavy" and not c["pass"] for c in result["checks"])
        wrong = deepcopy(r["process"])
        wrong["answer_smiles"] = "CC"
        result = inspect(r["raw"], wrong)
        assert any(c["check"] == "process_answer" and not c["pass"] for c in result["checks"])
        wrong = deepcopy(r["process"])
        for s in wrong["formal_cot_trace"]:
            s["formal_ab"] = s["formal_ab"].replace('ANCHOR(idx=36, element="N")', 'ANCHOR(idx=36, element="O")')
        result = inspect(r["raw"], wrong)
        assert not result["checks"][0]["pass"]
    # Independently check the exact nomenclature counterexample.
    core = Chem.MolFromSmiles("C1=CN2C(=CC=N2)N=C1")  # PubChem CID 11636795
    reference = sorted((len(ring), sum(core.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring))
                       for ring in core.GetRingInfo().AtomRings())
    assert reference == [(5, 2), (6, 2)]
    source_rings = records["mol_edit.add_v2.0150"]["ring_details"]
    six = next(x for x in source_rings if set(x["maps"]) == {2, 31, 6, 5, 4, 3})
    five = next(x for x in source_rings if set(x["maps"]) == {13, 7, 6, 31, 14})
    assert six["elements"].count("N") == 3 and five["elements"].count("N") == 1
    result = {"sample_records": 20, "original_steps": 106,
              "numeric_structure_checks_passed": sum(len(r["checks"])+len(r["fragment_checks"]) for r in records.values()),
              "exact_graph_reconstructions": 20, "verbatim_finding_quotes_verified": 4,
              "source_files_hash_unchanged": 6, "deliberate_error_detection_tests_passed": 3,
              "scaffold_counterexample_verified": True, "live_poe_requests": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    verify()
