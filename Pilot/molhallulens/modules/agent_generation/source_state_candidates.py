"""Bounded incorrect source reconstructions followed by the unchanged clean edit.

Each proposal omits one connected remote branch across a SINGLE bridge bond.
The branch must contain a ring, leave at least half the active heavy atoms, and
avoid every source atom required by the clean edit and its direct neighbors.
Those protected atoms may not receive boundary hydrogen adjustments either.
Other source components are spectators and remain untouched.

At most 128 proposals are strictly executed and at most 12 are returned, ranked
by full clean/altered product Morgan similarity. This is a staged diagnostic,
not a model of natural mistake frequency. Only the source and clean plan enter
construction; the caller establishes that the clean plan reproduces benchmark
GT. Target outputs and row outcome fields are never read.

``execution`` uses the perceived mapped source and the exact supplied clean
plan. ``joint_execution`` uses the original source and a net combined plan, for
severity correspondence. New fragment maps can differ between these paths when
the omitted branch held the original highest map. Isomeric graph equality and
retained original-source correspondence are proved separately. Full perceived
source strings are source-state evidence, never final-product answers.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from rdkit import Chem, rdBase

from .candidate_tools import _hydrogen_plan, _source_graph, describe_removed_fragment
from .chemistry_tools import ChemistryToolError, apply_edit_plan
from .severity import measure_severity


MAX_CANDIDATES = 12
MAX_PROPOSALS = 128


def _required_maps(plan: dict) -> set[int]:
    required = set(plan.get("remove_atom_maps", []))
    required.update(x["anchor_map"] for x in plan.get("add_fragments", []))
    required.update(x["atom_map"] for x in plan.get("adjust_hydrogens", []))
    required.update(k for edge in plan.get("remove_bonds", []) for k in edge)
    required.update(k for field in ("add_bonds", "change_bonds")
                    for edge in plan.get(field, []) for k in edge["atom_maps"])
    return required


def _remote_branches(source, active, protected):
    neighbors = {a.GetAtomMapNum(): {n.GetAtomMapNum() for n in a.GetNeighbors()}
                 for a in source.GetAtoms()}
    heavy = {a.GetAtomMapNum(): a.GetAtomicNum() > 1 for a in source.GetAtoms()}
    maximum = sum(heavy[k] for k in active) // 2
    rings = [{source.GetAtomWithIdx(i).GetAtomMapNum() for i in ring}
             for ring in source.GetRingInfo().AtomRings()]
    found = set()
    for bond in source.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        first, second = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if first not in active or {first, second} & protected:
            continue
        for start, stop in ((first, second), (second, first)):
            seen, pending = {start}, [start]
            while pending:
                current = pending.pop()
                for nxt in sorted(neighbors[current] - seen):
                    if {current, nxt} == {first, second}:
                        continue
                    seen.add(nxt)
                    pending.append(nxt)
            if (stop not in seen and seen <= active and not seen & protected
                    and sum(heavy[k] for k in seen) <= maximum
                    and any(ring <= seen for ring in rings)):
                found.add(tuple(sorted(seen)))
    return sorted(found, key=lambda maps: (-sum(heavy[k] for k in maps), maps))[:MAX_PROPOSALS]


def _retained_signature(mapped_smiles, retained):
    molecule = Chem.MolFromSmiles(mapped_smiles)
    for atom in molecule.GetAtoms():
        if atom.GetAtomMapNum() not in retained:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def enumerate_source_state_candidates(
    row: dict, clean_plan: dict, max_candidates: int = 12,
) -> list[dict[str, Any]]:
    """Return validated S→S′→P′ plans, with an equivalent original-S joint plan.

    Returning an empty list is a real coverage limitation: the implementation
    does not relax ring loss, protected neighborhoods, stereo or valence rules.
    All candidate products and perceived-source strings are controller evidence;
    the caller owns final rendering and answer-leak checks in their text context.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ChemistryToolError("invalid_plan", "max_candidates must be an integer from 1 to 12")
    facts, source = _source_graph(row["indexed_smiles"])
    clean = apply_edit_plan(facts["mapped_smiles"], clean_plan)
    required = _required_maps(clean_plan)
    if not required:
        raise ChemistryToolError("invalid_plan", "A nonempty clean source edit is required")
    active_components = [set(c) for c in facts["components"] if required & set(c)]
    if len(active_components) != 1 or not required <= active_components[0]:
        raise ChemistryToolError("ambiguous_edit", "Clean edit must act in one source component")
    active = active_components[0]
    protected = required | {n.GetAtomMapNum() for a in source.GetAtoms()
                            if a.GetAtomMapNum() in required for n in a.GetNeighbors()}
    source_maps = {a.GetAtomMapNum() for a in source.GetAtoms()}
    proposals = _remote_branches(source, active, protected)
    candidates, product_seen = [], set()
    for removed in proposals:
        try:
            source_plan = _hydrogen_plan(source, {"remove_atom_maps": list(removed)})
            if {x["atom_map"] for x in source_plan.get("adjust_hydrogens", [])} & protected:
                continue
            with rdBase.BlockLogs():
                perceived = apply_edit_plan(facts["mapped_smiles"], source_plan)
                removed_fragment = describe_removed_fragment(facts["mapped_smiles"], list(removed))
                staged = apply_edit_plan(perceived["mapped_product_smiles"], clean_plan)
            joint = deepcopy(clean_plan)
            joint["remove_atom_maps"] = sorted(set(clean_plan.get("remove_atom_maps", [])) | set(removed))
            joint = _hydrogen_plan(source, joint)
            with rdBase.BlockLogs():
                executed_joint = apply_edit_plan(facts["mapped_smiles"], joint)
            retained = source_maps - set(joint["remove_atom_maps"])
            same_graph = staged["product_smiles"] == executed_joint["product_smiles"]
            same_correspondence = (_retained_signature(staged["mapped_product_smiles"], retained)
                                   == _retained_signature(executed_joint["mapped_product_smiles"], retained))
            if not same_graph or not same_correspondence:
                continue
            if (perceived["rings"] >= facts["rings"] or staged["rings"] >= clean["rings"]
                    or staged["product_smiles"] == clean["product_smiles"]
                    or perceived["product_smiles"] in {clean["product_smiles"], staged["product_smiles"]}
                    or len(perceived["components"]) != len(facts["components"])):
                continue
            # Chemically identical source-branch omissions do not create extra
            # sampling weight through source symmetry. Keep the first exact plan.
            if staged["product_smiles"] in product_seen:
                continue
            severity = measure_severity(facts["mapped_smiles"], clean, executed_joint)
            candidates.append({
                "source_plan": source_plan, "perceived_source_execution": perceived,
                "wrong_source_mapped_smiles": perceived["mapped_product_smiles"],
                "execution": staged, "joint_plan": joint, "joint_execution": executed_joint,
                "severity": severity, "protected_maps": sorted(protected),
                "description": (f"Perceive the source without remote branch maps {list(removed)} "
                                f"({facts['rings'] - perceived['rings']} fewer rings), then apply the unchanged clean edit."),
                "construction": {
                    "kind": "remote_source_branch_omission", "required_edit_maps": sorted(required),
                    "active_source_maps": sorted(active), "removed_fragment": removed_fragment,
                    "source_ring_loss": facts["rings"] - perceived["rings"],
                    "source_heavy_loss": facts["heavy_atoms"] - perceived["heavy_atoms"],
                    "staged_joint_isomeric_equivalent": same_graph,
                    "retained_source_correspondence_preserved": same_correspondence,
                    "clean_edit_applied_unmodified": True,
                    "proposals_examined": len(proposals), "proposal_limit": MAX_PROPOSALS,
                    "selection_uses_target_outputs": False,
                },
            })
            product_seen.add(staged["product_smiles"])
        except ChemistryToolError:
            continue
    candidates.sort(key=lambda c: (c["severity"]["product_tanimoto"],
                                   -c["severity"]["source_footprint_ratio"],
                                   c["source_plan"]["remove_atom_maps"]))
    for index, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"candidate_{index:03d}"
    return candidates[:max_candidates]
