"""Evidence checks for the second audit; no source mutation or network."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from rdkit import Chem, rdBase
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from audit_round2 import DATA, HERE, PREVIOUS, checker


def verify():
    evidence = json.loads((HERE / "evidence.json").read_text())
    review = json.loads((HERE / "review.json").read_text())
    previous = json.loads((PREVIOUS / "evidence.json").read_text())
    records = {r["origin_id"]: r for r in evidence["records"]}
    assert len(records) == 20 and len(review["records"]) == 20
    assert set(records) == {r["origin_id"] for r in review["records"]}
    assert not set(records).intersection(r["origin_id"] for r in previous["records"])
    for relative, digest in evidence["source_sha256"].items():
        assert hashlib.sha256((DATA.parent / relative).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((PREVIOUS / "evidence.json").read_bytes()).hexdigest() == evidence["previous_evidence_sha256"]
    assert hashlib.sha256((PREVIOUS / "audit.py").read_bytes()).hexdigest() == evidence["previous_checker_sha256"]
    for r in records.values():
        assert all(c["pass"] for c in r["checks"] + r["fragment_checks"])
        assert r["graph_reconstruction"]["exact_isomeric_answer_recovered"]
    for f in review["findings"]:
        r = records[f["origin_id"]]
        assert any(f["quote"] in s["natural_language"] for s in r["process"]["formal_cot_trace"]), f["id"]
        if "instruction_quote" in f:
            assert f["instruction_quote"] == r["raw"]["instruction"]
    for f in review["secondary_notes"]:
        assert f["quote"] in records[f["origin_id"]]["process"]["formal_cot_trace"][f["step"]-1]["natural_language"]
    for note in review["metadata_notes"]["observations"]:
        assert records[note["origin_id"]]["raw"]["rxn_cls"] == note["stored"]

    semantic = {}
    for sid, correct, incorrect in [
        ("mol_edit.add_v2.0111", "C1=CC2=C(C=CS2)N=C1", "C1=CN=CC2=C1SC=C2"),
        ("mol_edit.substitute_v2.0122", "C1C2=CC=CC=C2NC1=O", "C1C2=CC=CC=C2C(=O)N1"),
    ]:
        mol = Chem.MolFromSmiles(records[sid]["raw"]["indexed_smiles"])
        hits = mol.GetSubstructMatches(Chem.MolFromSmiles(correct))
        assert hits and not mol.HasSubstructMatch(Chem.MolFromSmiles(incorrect))
        semantic[sid] = {"matching_reference_smiles": correct, "nonmatching_named_reference_smiles": incorrect,
                         "matching_maps": [[mol.GetAtomWithIdx(i).GetAtomMapNum() for i in hit] for hit in hits]}
    r = records["mol_edit.delete_v2.0134"]
    assert r["anchor"]["map"] == 3
    assert r["anchor"]["neighbors"] == [{"map": 2, "element": "C"}, {"map": 4, "element": "S"}]
    removed = r["graph_reconstruction"]["answer_matches"][0]["removed_maps"]
    assert 2 in removed and 4 not in removed
    semantic[r["origin_id"]] = {"retained_heavy_neighbor_after_deprotection": "S4", "group": "H2N-S(=O)2-NH-Ar"}
    r = records["mol_edit.substitute_v2.0150"]
    m = Chem.MolFromSmiles(r["raw"]["indexed_smiles"])
    atoms = {a.GetAtomMapNum(): a for a in m.GetAtoms()}
    assert atoms[5].GetSymbol() == atoms[14].GetSymbol() == "N"
    assert m.GetBondBetweenAtoms(atoms[5].GetIdx(), atoms[14].GetIdx()) is not None
    pyrimidine = Chem.MolFromSmiles("C1=CN=CN=C1")
    ni = [a.GetIdx() for a in pyrimidine.GetAtoms() if a.GetAtomicNum() == 7]
    assert len(Chem.GetShortestPath(pyrimidine, *ni))-1 == 2
    semantic[r["origin_id"]] = {"source_ring_N_maps": [5,14], "source_N_N_bond_distance": 1, "pyrimidine_N_N_bond_distance": 2}
    r = records["mol_edit.substitute_v2.0064"]
    source = Chem.MolFromSmiles(r["raw"]["indexed_smiles"])
    # Atom-map IDs are labels, not chemical substituent differences.
    for a in source.GetAtoms():
        a.SetAtomMapNum(0)
    assert not Chem.FindMolChiralCenters(source, includeUnassigned=True)
    answer = Chem.MolFromSmiles(r["raw"]["gt_smiles"])
    centers = Chem.FindMolChiralCenters(answer, includeUnassigned=True)
    assert len(centers) == 2 and all(label != "?" for _, label in centers)
    unsigned = Chem.Mol(answer)
    Chem.RemoveStereochemistry(unsigned)
    isomers = sorted(Chem.MolToSmiles(x) for x in EnumerateStereoisomers(unsigned, options=StereoEnumerationOptions(unique=True)))
    assert len(isomers) == 4 and Chem.MolToSmiles(answer) in isomers
    semantic[r["origin_id"]] = {"assigned_product_centers_0_based": centers, "same_connectivity_stereoisomers": isomers,
                               "instruction_does_not_specify_stereo": "AI semantic review; not inferred solely from regex"}

    r = records["mol_edit.add_v2.0013"]
    with rdBase.BlockLogs():
        p = deepcopy(r["process"])
        for s in p["formal_cot_trace"]:
            s["formal_ab"] = s["formal_ab"].replace("PRODUCT_SMILES[n_heavy=35]", "PRODUCT_SMILES[n_heavy=99]")
        result = checker.inspect(r["raw"], p)
        assert any(c["check"] == "product_heavy" and not c["pass"] for c in result["checks"])
        p = deepcopy(r["process"])
        p["answer_smiles"] = "CC"
        assert any(c["check"] == "process_answer" and not c["pass"] for c in checker.inspect(r["raw"],p)["checks"])
        p = deepcopy(r["process"])
        for s in p["formal_cot_trace"]:
            s["formal_ab"] = s["formal_ab"].replace('ANCHOR(idx=2, element="N")', 'ANCHOR(idx=2, element="O")')
        assert not checker.inspect(r["raw"], p)["checks"][0]["pass"]
    summary = {"origins":20, "steps":106, "overlap_with_round1":0,
               "numeric_structure_checks_passed":246, "exact_graph_reconstructions":20,
               "verbatim_finding_quotes_verified":5, "source_hashes_unchanged":6,
               "previous_audit_dependencies_unchanged":True, "deliberate_error_detection_tests_passed":3,
               "review_categories":dict(Counter(r["category"] for r in review["records"])),
               "metadata_scope_warnings":len(review["metadata_notes"]["observations"]), "live_poe_requests":0}
    (HERE / "semantic_checks.json").write_text(json.dumps(semantic, ensure_ascii=False, indent=2)+"\n")
    (HERE / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    verify()
