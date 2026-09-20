"""Deterministic, bounded hydrogen-capped environments of executed edit sites.

The clean and corrupted calls are independent. Neighborhood radii are 1..3;
intersected rings and unsaturated bonds are completed transitively. Only single
bonds may be hydrogen-capped; carbonyls and nitriles cannot become alkyl groups.
A candidate must be connected,
closed shell, at most 16 heavy atoms and at most half the complete product's
heavy atoms. Largest heavy-atom count wins, then smallest radius and map list.
Hydrogen caps describe a standalone local structure, not an isolated reaction
intermediate. Ambiguous stereo boundary caps are rejected, never stripped.

Callers must supply ``execution['reference_product_smiles']`` to exclude an
original GT distinct from this execution's product. Without that optional field,
only the actual product (and its complete components) can be checked; the return
evidence makes this distinction explicit. The field is used only as an exclusion,
never to construct a neighborhood. No target-model outputs enter this function.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from .candidate_tools import describe_removed_fragment
from .chemistry_tools import ChemistryToolError, apply_edit_plan, inspect_source


def _complete_features(mol: Chem.Mol, selected: set[int]) -> set[int]:
    selected = set(selected)
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    while True:
        expanded = set(selected)
        for ring in rings:
            if selected & ring:
                expanded.update(ring)
        for bond in mol.GetBonds():
            endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if bond.GetBondType() != Chem.BondType.SINGLE and endpoints & selected:
                expanded.update(endpoints)
        if expanded == selected:
            return selected
        selected = expanded


def _answer_forms(smiles: str) -> set[str]:
    canonical = inspect_source(smiles)["canonical_smiles"]
    # A spectator salt must not allow a complete answer component to masquerade
    # as a partial structure. Exclude all complete components as well as the full
    # component multiset, with stereochemistry retained.
    return {canonical, *canonical.split(".")}


def derive_local_environment(
    source_smiles: str, edit_plan: dict[str, Any], execution: dict[str, Any],
) -> dict[str, Any]:
    """Return a local SMILES, original atom maps and inspectable selection evidence.

    Raises ``ChemistryToolError`` with code ``local_environment_unavailable`` if
    no eligible neighborhood exists, or ``local_environment_invalid_execution``
    when the supplied execution does not match the source and plan. Inputs are
    never mutated. Multiple possible centers use the smallest surviving source
    map: addition anchors take precedence over neighbors of deleted source atoms.
    """
    source_facts = inspect_source(source_smiles)
    actual = apply_edit_plan(source_facts["mapped_smiles"], edit_plan)
    if not isinstance(execution, dict) or any(execution.get(key) != value for key, value in actual.items()):
        raise ChemistryToolError(
            "local_environment_invalid_execution", "Execution does not match the supplied source and edit plan",
        )
    params = Chem.SmilesParserParams()
    params.removeHs = False
    source = Chem.MolFromSmiles(source_facts["mapped_smiles"], params)
    product = Chem.MolFromSmiles(actual["mapped_product_smiles"], params)
    map_indices = {atom.GetAtomMapNum(): atom.GetIdx() for atom in product.GetAtoms()}
    source_maps = {atom.GetAtomMapNum() for atom in source.GetAtoms()}
    anchors = sorted({addition["anchor_map"] for addition in edit_plan.get("add_fragments", [])}
                     & map_indices.keys() & source_maps)
    center_origin = "addition_anchor"
    if not anchors:
        removed = set(edit_plan.get("remove_atom_maps", []))
        anchors = sorted({neighbor.GetAtomMapNum()
                          for atom in source.GetAtoms() if atom.GetAtomMapNum() in removed
                          for neighbor in atom.GetNeighbors()
                          if neighbor.GetAtomMapNum() not in removed
                          and neighbor.GetAtomMapNum() in map_indices})
        center_origin = "surviving_removal_neighbor"
    if not anchors:
        raise ChemistryToolError("local_environment_unavailable", "No surviving source edit center exists")
    center_map = anchors[0]
    forbidden = _answer_forms(actual["product_smiles"])
    reference = execution.get("reference_product_smiles")
    if reference is not None:
        forbidden.update(_answer_forms(reference))
    max_heavy = min(16, actual["heavy_atoms"] // 2)
    selected = {map_indices[center_map]}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for radius in (1, 2, 3):
        # Expand graph distance from the previous uncompleted neighborhood.
        # Ring completion must not increase the radius of the next iteration.
        selected |= {neighbor.GetIdx() for index in tuple(selected)
                     for neighbor in product.GetAtomWithIdx(index).GetNeighbors()}
        complete = _complete_features(product, selected)
        atom_maps = sorted(product.GetAtomWithIdx(index).GetAtomMapNum() for index in complete)
        heavy = sum(product.GetAtomWithIdx(index).GetAtomicNum() > 1 for index in complete)
        record = {"radius": radius, "atom_maps": atom_maps, "heavy_atoms": heavy}
        if not 0 < heavy <= max_heavy or len(complete) == product.GetNumAtoms():
            rejected.append({**record, "reason": "size_limit"})
            continue
        try:
            fragment = describe_removed_fragment(actual["mapped_product_smiles"], atom_maps)
        except ChemistryToolError as error:
            rejected.append({**record, "reason": error.code, "message": error.message})
            continue
        if fragment["component_count"] != 1:
            rejected.append({**record, "reason": "disconnected"})
        elif len(fragment["smiles"]) < 6:
            rejected.append({**record, "reason": "smiles_too_short"})
        elif fragment["smiles"] in forbidden:
            rejected.append({**record, "reason": "complete_answer"})
        else:
            candidates.append({**record, **fragment})
    if not candidates:
        raise ChemistryToolError(
            "local_environment_unavailable",
            f"No valid radius 1..3 environment at map {center_map}; rejected={rejected}",
        )
    best = min(candidates, key=lambda item: (-item["heavy_atoms"], item["radius"], item["atom_maps"]))
    return {
        "center_map": center_map,
        "smiles": best["smiles"],
        "atom_maps": best["atom_maps"],
        "heavy_atoms": best["heavy_atoms"],
        "evidence": {
            "policy": "radius_1_to_3_ring_unsaturated_complete_single_bond_caps_v2",
            "center_origin": center_origin,
            "eligible_center_maps": anchors,
            "radius": best["radius"],
            "product_heavy_atoms": actual["heavy_atoms"],
            "maximum_heavy_atoms": max_heavy,
            "cap_hydrogens": best["cap_hydrogens"],
            "formula": best["formula"],
            "rings": best["rings"],
            "reference_product_checked": reference is not None,
            "complete_product_and_components_excluded": True,
            "valid_candidate_count": len(candidates),
            "rejected_candidates": rejected,
        },
    }
