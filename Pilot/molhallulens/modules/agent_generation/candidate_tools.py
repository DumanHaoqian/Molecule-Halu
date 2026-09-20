"""Deterministic finite, executable edit candidates derived from source graphs.

This module never reads GT, outcome labels, or victim-model measurements. The
clean plan is executed solely to exclude its unchanged product. Candidate choice
is separate: the pool is an explicitly engineered diagnostic distribution, not
an estimate of naturally occurring reasoning mistakes.

The pool includes a fixed incoming-fragment vocabulary, same-element attachment
sites, combined fragment/site mistakes, and connected branch removals across
single bridge bonds. Hydrogen changes are recomputed from net source bond
valence. Source spectator components remain intact. RDKit validity is necessary
but does not establish that a candidate is a plausible instruction confusion.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from typing import Any

from rdkit import Chem, rdBase

from .chemistry_tools import (
    ChemistryToolError, apply_edit_plan, describe_fragment, inspect_source,
)


FRAGMENT_POOL = (
    "C(=O)C", "C(=O)CC", "C(=O)c1ccccc1", "S(=O)(=O)C", "CC",
    "C1CCCCC1", "Br", "Cl", "OC", "C#N",
)
MAX_CANDIDATES = 64
MAX_PROPOSALS = 128


def _source_graph(source_smiles: str) -> tuple[dict, Chem.Mol]:
    facts = inspect_source(source_smiles)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(facts["mapped_smiles"], params)
    return facts, mol


def _stereo_atoms(mol: Chem.Mol) -> set[int]:
    atoms = {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED}
    for bond in mol.GetBonds():
        if bond.GetStereo() not in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}:
            atoms.update((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
    return atoms


def describe_removed_fragment(source_smiles: str, removed_maps: list[int]) -> dict[str, Any]:
    """Describe the induced removed graph with explicit H caps at cut bonds.

    Caps replace lost integer bond valence on the removed side. The formula is
    that of this standalone capped molecule, not a detached radical or reagent.
    All removed components are retained. Aromatic cuts and caps at defined
    stereocenters are rejected. Empty removal has the explicit sentinel ``none``.
    """
    facts, source = _source_graph(source_smiles)
    known = {a.GetAtomMapNum() for a in source.GetAtoms()}
    if not isinstance(removed_maps, list) or any(type(v) is not int or v <= 0 for v in removed_maps):
        raise ChemistryToolError("invalid_plan", "Removed maps must be a list of positive integers")
    if len(set(removed_maps)) != len(removed_maps) or not set(removed_maps) <= known:
        raise ChemistryToolError("invalid_plan", "Removed maps must identify distinct existing atoms")
    removed = set(removed_maps)
    if not removed:
        return {"smiles": "none", "heavy_atoms": 0, "formula": None, "rings": 0,
                "removed_atom_maps": [], "cap_hydrogens": [], "component_count": 0}
    caps: dict[int, int] = defaultdict(int)
    for bond in source.GetBonds():
        first, second = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if (first in removed) == (second in removed):
            continue
        order = bond.GetBondTypeAsDouble()
        if order not in (1, 2, 3):
            raise ChemistryToolError("ambiguous_edit", "Aromatic boundary bonds cannot be hydrogen-capped")
        caps[first if first in removed else second] += int(order)
    if set(caps) & _stereo_atoms(source):
        raise ChemistryToolError("ambiguous_edit", "Hydrogen caps at defined stereocenters require an explicit policy")
    fragment = Chem.RWMol(source)
    for atom in fragment.GetAtoms():
        if atom.GetAtomMapNum() in caps:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + caps[atom.GetAtomMapNum()])
    for atom in reversed(list(source.GetAtoms())):
        if atom.GetAtomMapNum() not in removed:
            fragment.RemoveAtom(atom.GetIdx())
    result = fragment.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(result)
            Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    except Exception as error:
        raise ChemistryToolError("invalid_structure", "Removed fragment cannot form the specified hydrogen-capped graph") from error
    if any(atom.GetNumRadicalElectrons() for atom in result.GetAtoms()):
        raise ChemistryToolError("invalid_structure", "Hydrogen-capped removed fragment contains radicals")
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(result), canonical=True, isomericSmiles=True)
    described = inspect_source(smiles)
    return {"smiles": described["canonical_smiles"], "heavy_atoms": described["heavy_atoms"],
            "formula": described["formula"], "rings": described["rings"],
            "removed_atom_maps": sorted(removed),
            "cap_hydrogens": [{"atom_map": atom_map, "delta": delta} for atom_map, delta in sorted(caps.items())],
            "component_count": len(described["components"])}


def _hydrogen_plan(source: Chem.Mol, plan: dict) -> dict:
    """Replace old H adjustments with net valence consequences of this graph edit."""
    result = deepcopy(plan)
    result.pop("adjust_hydrogens", None)
    removed = set(result.get("remove_atom_maps", []))
    bonds = {frozenset((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())): b
             for b in source.GetBonds()}
    deltas: dict[int, int] = defaultdict(int)

    def order(bond: Chem.Bond) -> int:
        value = bond.GetBondTypeAsDouble()
        if value not in (1, 2, 3):
            raise ChemistryToolError("ambiguous_edit", "Candidate generation supports integer boundary bond orders")
        return int(value)

    for pair, bond in bonds.items():
        remaining = pair - removed
        if len(remaining) == 1 and pair & removed:
            deltas[next(iter(remaining))] += order(bond)
    for pair in result.get("remove_bonds", []):
        for atom_map in pair:
            deltas[atom_map] += order(bonds[frozenset(pair)])
    for item in result.get("add_fragments", []):
        deltas[item["anchor_map"]] -= {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}[item["bond_type"]]
    for item in result.get("add_bonds", []):
        for atom_map in item["atom_maps"]:
            deltas[atom_map] -= {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}[item["bond_type"]]
    for item in result.get("change_bonds", []):
        change = order(bonds[frozenset(item["atom_maps"])]) - {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}[item["bond_type"]]
        for atom_map in item["atom_maps"]:
            deltas[atom_map] += change
    adjustments = [{"atom_map": atom_map, "delta": value} for atom_map, value in sorted(deltas.items()) if value]
    if adjustments:
        result["adjust_hydrogens"] = adjustments
    return result


def _balanced(proposals: list[dict], clean_removed_heavy: int) -> list[dict]:
    """Round-robin semantic categories/sizes before applying the fixed bound."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for proposal in proposals:
        buckets[proposal["bucket"]].append(proposal)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: json.dumps(item["edit_plan"], sort_keys=True))
    keys = sorted(buckets, key=lambda k: (k[0], abs(k[1] - clean_removed_heavy) if k[0] == "delete" else k[1], k[1:]))
    result = []
    while any(buckets.values()) and len(result) < MAX_PROPOSALS:
        for key in keys:
            if buckets[key]:
                result.append(buckets[key].pop(0))
                if len(result) == MAX_PROPOSALS:
                    break
    return result


def _allocate_families(candidates: list[dict], maximum: int) -> list[dict]:
    """Prefer 12 fragment / 6 anchor / 6 combined slots at a cap of 24.

    Unused family slots are filled deterministically, preferring combined edits
    before extra anchor edits. At least the first two available fragment changes
    survive when the requested cap is two or more. Deletion-only pools use all
    available slots. This allocation uses no molecular outcome scores.
    """
    families: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        families[candidate["family"]].append(candidate)
    schedule = ("fragment", "fragment", "anchor", "combined")
    fallback = ("combined", "fragment", "anchor", "delete")
    selected = []
    for index in range(maximum):
        preferred = schedule[index % len(schedule)]
        family = next((name for name in (preferred, *fallback) if families[name]), None)
        if family is None:
            break
        selected.append(families[family].pop(0))
    for index, candidate in enumerate(selected, 1):
        candidate["candidate_id"] = f"candidate_{index:03d}"
    return selected


def enumerate_corruption_candidates(row: dict, clean_plan: dict, max_candidates: int = 24) -> list[dict[str, Any]]:
    """Return a reproducible pool of executable, source-derived complete plans.

    At most 128 diverse proposals are examined and 1..64 results may be requested.
    Pools preserve source-map distinctions but deduplicate identical plans and
    exclude products identical to the clean edit. Family allocation prefers
    fragment:anchor:combined proportions 2:1:1, redistributing unavailable slots.
    Candidate IDs are sequential, not digests. Only ``row['indexed_smiles']`` is read.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ChemistryToolError("invalid_plan", f"max_candidates must be an integer from 1 to {MAX_CANDIDATES}")
    source_facts, source = _source_graph(row["indexed_smiles"])
    source_smiles = source_facts["mapped_smiles"]
    clean_execution = apply_edit_plan(source_smiles, clean_plan)
    additions = clean_plan.get("add_fragments", [])
    if len(additions) > 1:
        raise ChemistryToolError("ambiguous_edit", "Candidate pool supports at most one incoming fragment")
    atoms = {a["atom_map"]: a for a in source_facts["atoms"]}
    removed = set(clean_plan.get("remove_atom_maps", []))
    clean_removed_heavy = sum(atoms[v]["element"] != "H" for v in removed)
    proposals: list[dict] = []
    if additions:
        original = additions[0]
        anchor = original["anchor_map"]
        active_component = next(set(c) for c in source_facts["components"] if anchor in c)
        original_fragment = describe_fragment(original["smiles"])["canonical_smiles"]
        fragment_facts = {smiles: describe_fragment(smiles) for smiles in FRAGMENT_POOL}
        for smiles in FRAGMENT_POOL:
            plan = deepcopy(clean_plan)
            plan["add_fragments"][0] = {"smiles": smiles, "attach_atom_index": 0,
                                         "anchor_map": anchor, "bond_type": "SINGLE"}
            proposals.append({"edit_plan": plan, "description": f"Use incoming fragment {smiles} at source map {anchor}.",
                              "bucket": ("fragment", fragment_facts[smiles]["heavy_atoms"], "")})
        stereo = _stereo_atoms(source)
        for atom_map in sorted(active_component - removed - {anchor}):
            atom = atoms[atom_map]
            if atom["element"] != atoms[anchor]["element"] or atom["explicit_hydrogens"] < 1 or atom_map in stereo:
                continue
            plan = deepcopy(clean_plan)
            plan["add_fragments"][0]["anchor_map"] = atom_map
            proposals.append({"edit_plan": plan, "description": f"Attach the reference fragment at alternative {atom['element']} source map {atom_map}.",
                              "bucket": ("anchor", 0, atom["element"])})
            for smiles in FRAGMENT_POOL:
                if fragment_facts[smiles]["canonical_smiles"] == original_fragment:
                    continue
                combined = deepcopy(clean_plan)
                combined["add_fragments"][0] = {"smiles": smiles, "attach_atom_index": 0,
                                                "anchor_map": atom_map, "bond_type": "SINGLE"}
                proposals.append({"edit_plan": combined,
                    "description": f"Use incoming fragment {smiles} at alternative {atom['element']} source map {atom_map}.",
                    "bucket": ("combined", fragment_facts[smiles]["heavy_atoms"], atom["element"])})
    elif removed:
        # Only the reference-affected component(s) are eligible; solvent and
        # counterion components are spectators even when they contain bridges.
        active_maps = set().union(*(set(c) for c in source_facts["components"] if removed & set(c)))
        maximum_removed = min(clean_removed_heavy + 5, source_facts["heavy_atoms"] // 2)
        neighbors: dict[int, set[int]] = {a.GetAtomMapNum(): set() for a in source.GetAtoms()}
        for bond in source.GetBonds():
            a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            neighbors[a].add(b)
            neighbors[b].add(a)
        for bond in source.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            if a not in active_maps:
                continue
            for start, retained in ((a, b), (b, a)):
                visited = {start}
                pending = [start]
                while pending:
                    current = pending.pop()
                    for neighbor in neighbors[current]:
                        if {current, neighbor} == {a, b} or neighbor in visited:
                            continue
                        visited.add(neighbor)
                        pending.append(neighbor)
                if retained in visited:  # Ring bond, not a bridge.
                    continue
                size = sum(atoms[v]["element"] != "H" for v in visited)
                if not 1 <= size <= maximum_removed:
                    continue
                plan = {"remove_atom_maps": sorted(visited)}
                proposals.append({"edit_plan": plan,
                    "description": f"Remove the {size}-heavy-atom branch across source bond {retained}-{start}; retain map {retained}.",
                    "bucket": ("delete", size, atoms[retained]["element"])})
    pool = []
    seen = set()
    for proposal in _balanced(proposals, clean_removed_heavy):
        try:
            plan = _hydrogen_plan(source, proposal["edit_plan"])
            encoded = json.dumps(plan, sort_keys=True)
            if encoded in seen:
                continue
            seen.add(encoded)
            execution = apply_edit_plan(source_smiles, plan)
            if execution["product_smiles"] == clean_execution["product_smiles"]:
                continue
            fragment = describe_removed_fragment(source_smiles, plan.get("remove_atom_maps", []))
        except (ChemistryToolError, KeyError):
            continue
        pool.append({"family": proposal["bucket"][0], "edit_plan": plan,
                     "description": proposal["description"], "execution": execution,
                     "removed_fragment": fragment})
    return _allocate_families(pool, max_candidates)
