"""Bounded compositions of larger, purely structural edit mistakes.

Only indexed source SMILES and a clean edit plan are inputs to construction.
Executed clean/H molecular similarity measures construction severity; no target
outputs or success labels enter this module. One changed removal set is one
semantic dimension even when it contains several disconnected branches.

At most 768 distinct proposals are examined and at most 36 candidates returned.
Deletion can open a saturated ring, provided the retained active source graph
stays connected. Non-single bond endpoints are removed together during boundary
expansion. Source spectators remain untouched. The strict edit executor and
hydrogen-capped removed-fragment validation reject ambiguous stereo or valence.
The caller owns instruction plausibility, rendering, and full-answer exclusion.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import combinations
import json
from typing import Any

from rdkit import Chem

from .candidate_tools import (
    FRAGMENT_POOL, _hydrogen_plan, _source_graph, _stereo_atoms,
    describe_removed_fragment,
)
from .chemistry_tools import ChemistryToolError, apply_edit_plan, describe_fragment
from .severity import measure_severity


LARGE_FRAGMENTS = (
    "C(=O)c1ccc(-c2ccccc2)cc1", "C(=O)c1ccc(C(F)(F)F)cc1",
    "S(=O)(=O)c1ccc(C)cc1", "C(=O)C1CCCCC1", "C(=O)CCCCCCC",
    "NC(=O)OC(C)(C)C", "N1CCN(C)CC1", "C(=O)OC(C)(C)C",
)
MAX_CANDIDATES = 36
MAX_PROPOSALS = 768


@lru_cache(maxsize=128)
def _fragment_identity(smiles):
    return describe_fragment(smiles)["canonical_smiles"]


def _neighbors(source):
    return {a.GetAtomMapNum(): {n.GetAtomMapNum() for n in a.GetNeighbors()} for a in source.GetAtoms()}


def _connected(selected, neighbors):
    if not selected:
        return False
    seen, pending = {min(selected)}, [min(selected)]
    while pending:
        new = (neighbors[pending.pop()] & selected) - seen
        seen.update(new)
        pending.extend(sorted(new))
    return seen == selected


def _close_multiple_bonds(source, removed):
    """Do not leave an aromatic atom or half a carbonyl/nitrile at a cut."""
    removed = set(removed)
    while True:
        previous = set(removed)
        for bond in source.GetBonds():
            ends = {bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()}
            if bond.GetBondType() != Chem.BondType.SINGLE and removed & ends:
                removed.update(ends)
        if previous == removed:
            return removed


def _legal_removal(removed, active, neighbors, heavy, maximum, anchor=None):
    return (bool(removed) and removed <= active and anchor not in removed
            and sum(heavy[x] for x in removed) <= maximum
            and sum(heavy[x] for x in active - removed) >= 4
            and _connected(active - removed, neighbors))


def _branches(source, active, neighbors, heavy, maximum):
    found = set()
    for bond in source.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if a not in active:
            continue
        for start, stop in ((a, b), (b, a)):
            seen, pending = {start}, [start]
            while pending:
                current = pending.pop()
                for nxt in sorted(neighbors[current] - seen):
                    if {current, nxt} == {a, b}:
                        continue
                    seen.add(nxt)
                    pending.append(nxt)
            if stop not in seen and _legal_removal(seen, active, neighbors, heavy, maximum):
                found.add(tuple(sorted(seen)))
    # Favor broad changes but retain distinct sizes/locations in finite sources.
    return [set(x) for x in sorted(found, key=lambda x: (-sum(heavy[k] for k in x), x))[:64]]


def _dimensions(clean_plan, plan):
    dimensions = []
    a, b = clean_plan.get("add_fragments", []), plan.get("add_fragments", [])
    if a and b:
        if _fragment_identity(a[0]["smiles"]) != _fragment_identity(b[0]["smiles"]):
            dimensions.append("fragment")
        if a[0]["anchor_map"] != b[0]["anchor_map"]:
            dimensions.append("anchor")
    if sorted(clean_plan.get("remove_atom_maps", [])) != sorted(plan.get("remove_atom_maps", [])):
        dimensions.append("remove_set")
    return dimensions


def _proposal_pool(source, facts, clean_plan):
    neighbors = _neighbors(source)
    atoms = {a.GetAtomMapNum(): a for a in source.GetAtoms()}
    heavy = {k: a.GetAtomicNum() > 1 for k, a in atoms.items()}
    removed = set(clean_plan.get("remove_atom_maps", []))
    additions = clean_plan.get("add_fragments", [])
    if len(additions) > 1:
        raise ChemistryToolError("ambiguous_edit", "Severe pool supports at most one incoming fragment")
    centers = {a["anchor_map"] for a in additions} or removed
    affected = [set(c) for c in facts["components"] if centers & set(c)]
    if len(affected) != 1:
        raise ChemistryToolError("ambiguous_edit", "A single source component must contain the reference edit")
    active = affected[0]
    maximum = min(24, sum(heavy[k] for k in active) // 2)
    branches = _branches(source, active, neighbors, heavy, maximum)
    proposals = []

    def emit(family, plan, **evidence):
        proposals.append({"family": family, "edit_plan": plan, "construction": evidence})

    if additions:
        original = additions[0]
        anchor = original["anchor_map"]
        stereo = _stereo_atoms(source)
        choices = []
        for atom_map in active - removed - {anchor} - stereo:
            atom = atoms[atom_map]
            if atom.GetTotalNumHs() < 1 or atom.GetAtomicNum() not in {6, 7, 8, 16}:
                continue
            priority = (0 if atom.GetAtomicNum() == atoms[anchor].GetAtomicNum()
                        else 1 if atom.GetAtomicNum() in {7, 8, 16} else 2)
            distance = len(Chem.GetShortestPath(source, atoms[anchor].GetIdx(), atom.GetIdx())) - 1
            choices.append((priority, -distance, atom_map))
        anchors = [x[2] for x in sorted(choices)[:12]]
        original_identity = _fragment_identity(original["smiles"])
        fragments = [s for s in (*LARGE_FRAGMENTS, *FRAGMENT_POOL)
                     if _fragment_identity(s) != original_identity]

        def make(fragment, site, removal):
            plan = deepcopy(clean_plan)
            plan["add_fragments"][0] = {"smiles": fragment, "anchor_map": site,
                                        "attach_atom_index": 0, "bond_type": "SINGLE"}
            if removal:
                plan["remove_atom_maps"] = sorted(removal)
            else:
                plan.pop("remove_atom_maps", None)
            return plan

        for fragment in fragments:
            emit("fragment", make(fragment, anchor, removed))
        for site in anchors:
            plan = deepcopy(clean_plan)
            plan["add_fragments"][0]["anchor_map"] = site
            emit("anchor", plan)
            for fragment in fragments:
                emit("fragment_anchor", make(fragment, site, removed))
        for site in [anchor, *anchors]:
            possible = []
            for branch in branches:
                selection = removed | branch
                if selection != removed and _legal_removal(selection, active, neighbors, heavy, maximum, site):
                    if selection not in possible:
                        possible.append(selection)
            possible = possible[:12]
            for index, selection in enumerate(possible):
                if site != anchor:
                    plan = deepcopy(clean_plan)
                    plan["add_fragments"][0]["anchor_map"] = site
                    plan["remove_atom_maps"] = sorted(selection)
                    emit("anchor_removal", plan, additional_removed_maps=sorted(selection - removed))
                # Rotate fragments across branch/anchor combinations rather than
                # filling the finite proposal budget with one fragment identity.
                count = len(fragments) if site == anchor else min(6, len(fragments))
                for offset in range(count):
                    fragment = fragments[(index + offset + site) % len(fragments)]
                    family = "fragment_removal" if site == anchor else "fragment_anchor_removal"
                    emit(family, make(fragment, site, selection), additional_removed_maps=sorted(selection - removed))
    elif removed:
        frontier = {tuple(sorted(removed))}
        for depth in (1, 2):
            next_frontier = set()
            for old in sorted(frontier):
                current = set(old)
                adjacent = set().union(*(neighbors[k] for k in current)) - current
                additions_to_set = [{k} for k in sorted(adjacent)] + ([adjacent] if adjacent else [])
                for extra in additions_to_set:
                    selection = _close_multiple_bonds(source, current | extra)
                    if _legal_removal(selection, active, neighbors, heavy, maximum):
                        key = tuple(sorted(selection))
                        next_frontier.add(key)
                        emit("delete_boundary", {"remove_atom_maps": list(key)}, expansion_steps=depth,
                             additional_removed_maps=sorted(selection - removed))
            frontier = next_frontier
        for branch in branches:
            emit("delete_branch", {"remove_atom_maps": sorted(branch)})
            selection = removed | branch
            if selection != removed and _legal_removal(selection, active, neighbors, heavy, maximum):
                emit("delete_reference_union", {"remove_atom_maps": sorted(selection)},
                     additional_removed_maps=sorted(selection - removed))
        for first, second in combinations(branches[:32], 2):
            if first & second:
                continue
            selection = first | second
            if _legal_removal(selection, active, neighbors, heavy, maximum):
                emit("delete_disjoint_union", {"remove_atom_maps": sorted(selection)},
                     removed_branches=[sorted(first), sorted(second)])
    else:
        raise ChemistryToolError("invalid_plan", "Reference plan must add a fragment or remove atoms")
    return proposals


def _balanced_proposals(proposals):
    buckets = defaultdict(list)
    for item in proposals:
        buckets[item["family"]].append(item)
    priorities = ("fragment_anchor_removal", "delete_boundary", "delete_disjoint_union",
                  "fragment_anchor", "delete_reference_union", "fragment_removal", "anchor_removal",
                  "fragment", "anchor", "delete_branch")
    for items in buckets.values():
        items.sort(key=lambda x: (-len(x["edit_plan"].get("remove_atom_maps", [])),
                                  json.dumps(x["edit_plan"], sort_keys=True)))
    count = 0
    while any(buckets.values()) and count < MAX_PROPOSALS:
        for family in priorities:
            if buckets[family]:
                yield buckets[family].pop(0)
                count += 1
                if count == MAX_PROPOSALS:
                    break


def enumerate_severe_candidates(row: dict, clean_plan: dict, max_candidates: int = 36) -> list[dict[str, Any]]:
    """Return source-derived executable plans and independent severity evidence.

    Balanced returned quotas favor 3/2/1 changed dimensions in a 3:2:1 schedule.
    Deletion families are balanced separately and always carry one remove_set
    root. Within each bucket larger actual clean/H differences rank first.
    Full-product strings are execution evidence for the controller, never a
    prescription to include a complete product in visible reasoning.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ChemistryToolError("invalid_plan", "max_candidates must be an integer from 1 to 36")
    facts, source = _source_graph(row["indexed_smiles"])
    clean = apply_edit_plan(facts["mapped_smiles"], clean_plan)
    proposals = _proposal_pool(source, facts, clean_plan)
    candidates, seen, removed_cache = [], set(), {}
    for proposal in _balanced_proposals(proposals):
        try:
            plan = _hydrogen_plan(source, proposal["edit_plan"])
            encoded = json.dumps(plan, sort_keys=True)
            if encoded in seen:
                continue
            seen.add(encoded)
            execution = apply_edit_plan(facts["mapped_smiles"], plan)
            if execution["product_smiles"] == clean["product_smiles"]:
                continue
            removal_key = tuple(plan.get("remove_atom_maps", []))
            if removal_key not in removed_cache:
                removed_cache[removal_key] = describe_removed_fragment(facts["mapped_smiles"], list(removal_key))
            removed = removed_cache[removal_key]
            dimensions = _dimensions(clean_plan, plan)
            if not dimensions:
                continue
            severity = measure_severity(facts["mapped_smiles"], clean, execution)
        except (ChemistryToolError, KeyError):
            continue
        candidates.append({"family": proposal["family"], "edit_plan": plan, "execution": execution,
            "removed_fragment": removed, "edit_dimensions": dimensions, "structural_root_count": len(dimensions),
            "description": "Change " + ", ".join(dimensions) + "; " +
                           f"remove source maps {plan.get('remove_atom_maps', [])}" +
                           (f"; attach {plan['add_fragments'][0]['smiles']} at source map {plan['add_fragments'][0]['anchor_map']}."
                            if plan.get("add_fragments") else "."),
            "construction": {**proposal["construction"], "proposal_budget": MAX_PROPOSALS,
                             "incoming_heavy_atoms": (Chem.MolFromSmiles(plan["add_fragments"][0]["smiles"]).GetNumHeavyAtoms()
                                                       if plan.get("add_fragments") else 0),
                             "numeric_claim_roots": 0, "selection_uses_target_outputs": False},
            "severity": severity})
    groups = defaultdict(list)
    addition_mode = bool(clean_plan.get("add_fragments"))
    for item in candidates:
        groups[item["structural_root_count"] if addition_mode else item["family"]].append(item)
    for items in groups.values():
        items.sort(key=lambda x: (x["severity"]["product_tanimoto"], -x["severity"]["source_footprint_ratio"],
                                  json.dumps(x["edit_plan"], sort_keys=True)))
    schedule = ((3, 3, 2, 3, 2, 1) if addition_mode else
                ("delete_boundary", "delete_disjoint_union", "delete_reference_union", "delete_boundary", "delete_disjoint_union", "delete_branch"))
    fallback = ((3, 2, 1) if addition_mode else
                ("delete_boundary", "delete_disjoint_union", "delete_reference_union", "delete_branch"))
    result, selected_per_group = [], defaultdict(int)
    for i in range(max_candidates):
        key = next((key for key in (schedule[i % len(schedule)], *fallback) if groups[key]), None)
        if key is None:
            break
        # Similarity alone can favor halogens over every larger fragment.
        # Reserve every third slot within a dimension family for an available
        # >=11-heavy-atom incoming group; remaining slots follow severity rank.
        index = 0
        if addition_mode and selected_per_group[key] % 3 == 0:
            index = next((j for j, c in enumerate(groups[key])
                          if c["construction"]["incoming_heavy_atoms"] >= 11), 0)
        item = groups[key].pop(index)
        selected_per_group[key] += 1
        item["candidate_id"] = f"candidate_{len(result) + 1:03d}"
        result.append(item)
    return result
