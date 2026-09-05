"""Independent, read-only audit of a predeclared stratified 20-origin sample.

Does not import molhallulens builders, planners, validators, or generated data.
Raw molecules are independently parsed with RDKit; graph edits enumerate the
attachment atom rather than assuming canonical fragment atom 0 is the root.
This is not a reaction-yield or experimental-feasibility oracle.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import re

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "Dataset"
SEED = 20260905
ALLOCATION = {"add": 7, "delete": 7, "substitute": 6}


def canonical(mol, stereo=True):
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, isomericSmiles=stereo)


def descriptors(mol):
    return {"heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": int(rdMolDescriptors.CalcNumRings(mol)),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "elements": dict(Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)),
            "formal_charge": Chem.GetFormalCharge(mol)}


def ring_details(mol):
    return [{"maps": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in ring],
             "elements": [mol.GetAtomWithIdx(i).GetSymbol() for i in ring],
             "size": len(ring),
             "all_atoms_aromatic": all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring),
             "bonds": [str(mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)]).GetBondType()) for i in range(len(ring))]}
            for ring in mol.GetRingInfo().AtomRings()]


def boundary_removals(source, anchor, group):
    if group is None:
        return [set()]
    found = []
    for neighbor in source.GetAtomWithIdx(anchor).GetNeighbors():
        bond_order = source.GetBondBetweenAtoms(anchor, neighbor.GetIdx()).GetBondTypeAsDouble()
        if bond_order not in (1, 2, 3):
            continue
        cut = Chem.RWMol(source)
        cut.RemoveBond(anchor, neighbor.GetIdx())
        components = Chem.GetMolFrags(cut)
        for component in components:
            if neighbor.GetIdx() not in component or anchor in component:
                continue
            # Cap the detached endpoint with H to compare ordinary group SMILES.
            detached = Chem.RWMol(source)
            endpoint = detached.GetAtomWithIdx(neighbor.GetIdx())
            endpoint.SetNumExplicitHs(endpoint.GetTotalNumHs() + int(bond_order))
            for index in sorted(set(range(source.GetNumAtoms())) - set(component), reverse=True):
                detached.RemoveAtom(index)
            try:
                Chem.SanitizeMol(detached)
                if canonical(detached, False) == canonical(group, False):
                    found.append(set(component))
            except (ValueError, RuntimeError):
                pass
    return found


def reconstruct(source, anchor, removed_group, incoming, answer):
    removals = boundary_removals(source, anchor, removed_group)
    hits, candidates = [], set()
    for removed in removals:
        base = Chem.RWMol(source)
        anchor_atom = base.GetAtomWithIdx(anchor)
        removed_valence = sum(source.GetBondBetweenAtoms(anchor, i).GetBondTypeAsDouble()
                              for i in removed if source.GetBondBetweenAtoms(anchor, i) is not None)
        hydrogens = anchor_atom.GetTotalNumHs() + removed_valence - (incoming is not None)
        if hydrogens < 0:
            continue
        anchor_atom.SetNumExplicitHs(int(hydrogens))
        for index in sorted(removed, reverse=True):
            base.RemoveAtom(index)
        retained_anchor = anchor - sum(i < anchor for i in removed)
        attachment_indices = range(incoming.GetNumAtoms()) if incoming is not None else [None]
        for attachment in attachment_indices:
            try:
                if incoming is None:
                    product = Chem.RWMol(base)
                else:
                    fragment = Chem.Mol(incoming)
                    fa = fragment.GetAtomWithIdx(attachment)
                    # Bracket-H atoms require explicit hydrogen consumption.
                    if fa.GetNumExplicitHs():
                        fa.SetNumExplicitHs(fa.GetNumExplicitHs() - 1)
                    product = Chem.RWMol(Chem.CombineMols(base, fragment))
                    product.AddBond(retained_anchor, base.GetNumAtoms() + attachment, Chem.BondType.SINGLE)
                Chem.SanitizeMol(product)
                product_smiles = canonical(product)
                candidates.add(product_smiles)
                if product_smiles == canonical(answer):
                    hits.append({"removed_maps": sorted(source.GetAtomWithIdx(i).GetAtomMapNum() for i in removed),
                                 "fragment_attachment_index_0_based": attachment,
                                 "fragment_attachment_element": incoming.GetAtomWithIdx(attachment).GetSymbol() if incoming is not None else None})
            except (ValueError, RuntimeError):
                continue
    return {"matching_removal_sets": len(removals), "valid_candidate_count": len(candidates),
            "exact_isomeric_answer_recovered": bool(hits), "answer_matches": hits,
            "candidates": sorted(candidates),
            "scope": "single boundary-bond removal (order 1/2/3) / single-bond addition; valence-aware H capping, atom identity and stereochemical graph retained; not a reaction mechanism"}


def inspect(raw, process):
    source = Chem.MolFromSmiles(raw["indexed_smiles"])
    answer = Chem.MolFromSmiles(raw["gt_smiles"])
    assert source is not None and answer is not None
    steps = process["formal_cot_trace"]
    formal = "\n".join(s["formal_ab"] for s in steps)
    anchor_match = re.search(r'ANCHOR\(idx=(\d+), element="([A-Za-z]+)"\)', formal)
    idx, element = int(anchor_match[1]), anchor_match[2]
    anchors = [a for a in source.GetAtoms() if a.GetAtomMapNum() == idx]
    checks = [{"check": "unique_anchor_map_and_element", "pass": len(anchors)==1 and anchors[0].GetSymbol()==element}]
    maps = [a.GetAtomMapNum() for a in source.GetAtoms()]
    checks.append({"check": "contiguous_1_based_heavy_atom_maps", "pass": maps==list(range(1, source.GetNumHeavyAtoms()+1))})
    atom = anchors[0]
    source_desc, answer_desc = descriptors(source), descriptors(answer)
    field_patterns = {
        "source_heavy": (r'(?<!PRODUCT_)SMILES\[n_heavy=(\d+)\]', source_desc["heavy_atoms"]),
        "product_heavy": (r'PRODUCT_SMILES\[n_heavy=(\d+)\]', answer_desc["heavy_atoms"]),
        "heavy_delta": (r'HEAVY_ATOM_DELTA\(([+-]?\d+)\)', answer_desc["heavy_atoms"]-source_desc["heavy_atoms"]),
        "source_rings": (r'(?<!PRODUCT_)SMILES\[n_rings=(\d+)\]', source_desc["rings"]),
        "product_rings": (r'PRODUCT_SMILES\[n_rings=(\d+)\]', answer_desc["rings"]),
        "ring_delta": (r'RING_DELTA\(([+-]?\d+)\)', answer_desc["rings"]-source_desc["rings"]),
    }
    for label, (pattern, actual) in field_patterns.items():
        observed = [int(x) for x in re.findall(pattern, formal)]
        checks.append({"check": label, "claimed": observed, "computed": actual,
                       "pass": bool(observed) and all(x==actual for x in observed)})
    products = re.findall(r'PRODUCT_SMILES\("([^"]+)"\)', formal)
    for label, smiles in [("process_answer", process["answer_smiles"]), ("process_gt", process["gt_smiles"])] + [("formal_product", x) for x in products]:
        mol = Chem.MolFromSmiles(smiles)
        checks.append({"check": label, "pass": mol is not None and canonical(mol)==canonical(answer)})
    group_texts = {}
    for token in ("ADD_FRAGMENT", "REMOVE_GROUP", "LEAVING"):
        values = re.findall(token + r'\(smiles="([^"]+)"', formal)
        group_texts[token] = values[0] if values else None
    fragments = {k: (Chem.MolFromSmiles(v) if v and v!="none" else None) for k,v in group_texts.items()}
    fragment_checks = []
    for step in steps:
        text = step["formal_ab"]
        count_match = re.search(r'(?:heavy_atoms=|HEAVY_ATOMS\(|REMOVE_HEAVY\(|ADD_HEAVY\()(\d+)', text)
        if count_match:
            token = "REMOVE_GROUP" if "REMOVE_GROUP" in text else "ADD_FRAGMENT"
            actual = fragments[token].GetNumHeavyAtoms()
            fragment_checks.append({"step": step["step_index"], "claimed": int(count_match[1]), "computed": actual,
                                    "pass": actual==int(count_match[1])})
    removed = fragments["REMOVE_GROUP"] or fragments["LEAVING"]
    graph = reconstruct(source, atom.GetIdx(), removed, fragments["ADD_FRAGMENT"], answer)
    return {"origin_id": raw["anonymous_sample_id"], "task": raw["subtask"].replace("_pilot_origin", ""),
            "raw": raw, "process": process, "source_computed": source_desc, "answer_computed": answer_desc,
            "anchor": {"map": idx, "element": atom.GetSymbol(), "total_H": atom.GetTotalNumHs(),
                       "neighbors": [{"map": a.GetAtomMapNum(), "element": a.GetSymbol()} for a in atom.GetNeighbors()]},
            "fragments": {k: {"smiles": group_texts[k], "computed": descriptors(v) if v is not None else None} for k,v in fragments.items()},
            "ring_details": ring_details(source), "checks": checks, "fragment_checks": fragment_checks,
            "graph_reconstruction": graph}


def run():
    rng, evidence, population, files = random.Random(SEED), [], [], {}
    for task, count in ALLOCATION.items():
        datasets = {}
        for kind in ("raw_benchmark_data", "process_evaluation_data"):
            path = DATA / kind / "mol_edit" / f"{task}_pilot_origin.json"
            rows = json.loads(path.read_text())
            ids = [r["anonymous_sample_id"] for r in rows]
            assert len(ids)==len(set(ids))==50
            datasets[kind] = {r["anonymous_sample_id"]: r for r in rows}
            files[str(path.relative_to(HERE.parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw, process = datasets.values()
        assert set(raw)==set(process)
        selected = sorted(rng.sample(sorted(raw), count))
        population.append({"task": task, "available": len(raw), "sampled": count, "ids": selected,
                           "stored_outcome_counts": dict(Counter(str(r['outcome']) for r in process.values())),
                           "stored_all_pass_counts": dict(Counter(str(r['verifier_checks'].get('all_pass')) for r in process.values()))})
        with rdBase.BlockLogs():
            evidence.extend(inspect(raw[sid], process[sid]) for sid in selected)
    result = {"seed": SEED, "sampling": "sequential random.Random(seed).sample(sorted(ids), n), task order add/delete/substitute; fixed before semantic inspection",
              "rdkit_version": rdBase.rdkitVersion, "population": population, "source_sha256": files,
              "records": evidence}
    (HERE / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    text = []
    for record in evidence:
        text.extend(["="*90, record["origin_id"], "INSTRUCTION: "+record["raw"]["instruction"],
                     "SOURCE: "+record["raw"]["indexed_smiles"], "ANSWER: "+record["raw"]["gt_smiles"],
                     "ANCHOR: "+json.dumps(record["anchor"]), "COMPUTED: "+json.dumps([record["source_computed"],record["answer_computed"]]),
                     "RECONSTRUCTED: "+json.dumps(record["graph_reconstruction"]),
                     "RINGS: "+json.dumps(record["ring_details"])])
        text.extend(s["step_text"] for s in record["process"]["formal_cot_trace"])
    (HERE / "sample_review.txt").write_text("\n\n".join(text)+"\n")
    print(json.dumps({"sample_size":len(evidence), "failed_numeric_or_answer_checks":sum(not c["pass"] for r in evidence for c in r["checks"]+r["fragment_checks"]),
                      "reconstructed_answers":sum(r["graph_reconstruction"]["exact_isomeric_answer_recovered"] for r in evidence),
                      "ids":[r["origin_id"] for r in evidence]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
