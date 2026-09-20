"""Render N/H with the same semantic template and compiler-owned error spans.

This is a canonical public view, not a verbatim rewrite of the original prose.
The controller must retain original N separately and render both conditions
through this function. Values come from approved nodes; executable operations
come from the plan. Complete products are never rendered as molecular strings.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any

from .chemistry_tools import describe_fragment, inspect_source


_MISSING = object()


def render_reasoning(
    source_smiles: str,
    edit_plan: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    *,
    label_errors: bool,
) -> dict[str, Any]:
    """Render approved ``after`` states; N supplies after=before and no labels.

    The first renderer supports zero or one incoming fragment, atom removals,
    and explicitly listed bond edits. Every changed supplied node must have a
    rendered slot. Labels cover the exact semantic value, preserve shared root
    provenance, and never turn repeated mentions into extra semantic roots.
    Hydrogen adjustments remain in execution evidence rather than in prose.
    """
    if not isinstance(edit_plan, Mapping) or not isinstance(nodes, Mapping):
        raise TypeError("edit_plan and nodes must be objects")
    if type(label_errors) is not bool:
        raise TypeError("label_errors must be a boolean")
    source = inspect_source(source_smiles)
    atoms = {atom["atom_map"]: atom for atom in source["atoms"]}
    additions = edit_plan.get("add_fragments", [])
    if not isinstance(additions, list) or len(additions) > 1:
        raise ValueError("renderer supports at most one incoming fragment")
    removed = sorted(edit_plan.get("remove_atom_maps", []))
    if len(set(removed)) != len(removed) or any(atom_map not in atoms for atom_map in removed):
        raise ValueError("remove_set must contain distinct existing source atom maps")

    pieces: list[str] = []
    spans: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    bound: set[str] = set()
    cursor = 0

    def write(text: str) -> None:
        nonlocal cursor
        pieces.append(text)
        cursor += len(text)

    def value(name: str, fallback: Any = _MISSING) -> Any:
        if name in nodes:
            data = nodes[name]
            if not isinstance(data, Mapping) or "after" not in data or type(data.get("changed")) is not bool:
                raise ValueError(f"Invalid semantic node: {name}")
            return data["after"]
        if fallback is _MISSING:
            raise ValueError(f"Missing approved semantic node: {name}")
        return fallback

    def plan_value(name: str, actual: Any) -> Any:
        approved = value(name, actual)
        if approved != actual:
            raise ValueError(f"Approved {name} differs from the executed edit plan")
        return approved

    def number(name: str, fallback: Any = _MISSING, *, nonnegative: bool = True) -> int:
        result = value(name, fallback)
        if type(result) is not int or (nonnegative and result < 0):
            raise ValueError(f"{name} must be an integer" + (" >= 0" if nonnegative else ""))
        return result

    def slot(name: str, displayed: Any) -> None:
        bound.add(name)
        text = str(displayed)
        start = cursor
        write(text)
        data = nodes.get(name)
        annotate = bool(label_errors and data and data["changed"])
        binding = {"start": start, "end": cursor, "text": text, "node_id": name,
                   "origin": "approved_node" if data else "executed_plan", "error_annotated": annotate}
        if name == "fragment_attachment" and not data:
            binding["semantic_note"] = "representation-relative fragment atom index, locally valid; not an independently incorrect claim"
        bindings.append(binding)
        if not annotate:
            return
        if data.get("kind") not in {"root_error", "propagated_error"}:
            raise ValueError(f"Invalid error kind for {name}")
        roots = data.get("root_ids")
        if not isinstance(roots, (list, tuple)) or not roots or any(not isinstance(root, str) or not root for root in roots):
            raise ValueError(f"Changed node {name} requires root_ids")
        if len(set(roots)) != len(roots):
            raise ValueError(f"Changed node {name} has duplicate root_ids")
        spans.append({"start": start, "end": cursor, "text": text, "node_id": name,
                      "kind": data["kind"], "root_ids": list(roots),
                      "evidence": {"source": "approved_semantic_node", "before": deepcopy(data.get("before")),
                                   "after": deepcopy(data["after"])}})

    source_heavy = number("source_heavy")
    source_rings = number("source_rings", source["rings"])
    product_heavy = number("product_heavy")
    product_rings = number("product_rings")
    heavy_delta = number("heavy_delta", nonnegative=False)
    ring_delta = number("ring_delta", nonnegative=False)
    remove_set = plan_value("remove_set", removed)
    actual_removed_heavy = sum(atoms[atom_map]["element"] != "H" for atom_map in removed)
    removed_heavy = number("remove_heavy", actual_removed_heavy)
    if removed_heavy != actual_removed_heavy:
        raise ValueError("Approved remove_heavy differs from the executed edit plan")
    removed_fragment = value("remove_fragment") if "remove_fragment" in nodes else None
    if "remove_fragment" in nodes and (
        not isinstance(removed_fragment, str) or not removed_fragment.strip() or not removed
    ):
        raise ValueError("remove_fragment requires nonempty approved SMILES and removed source atoms")

    fragment_heavy = 0
    fragment = anchor = anchor_element = attachment = bond_type = composition = None
    if additions:
        addition = additions[0]
        fragment = plan_value("fragment", addition["smiles"])
        facts = describe_fragment(fragment)
        anchor = plan_value("anchor", addition["anchor_map"])
        if anchor not in atoms or anchor in removed:
            raise ValueError("anchor must identify a surviving source atom")
        anchor_element = plan_value("anchor_element", atoms[anchor]["element"])
        attachment = value("fragment_attachment", addition.get("attach_atom_index", 0))
        if isinstance(attachment, Mapping):
            if attachment.get("smiles") != fragment:
                raise ValueError("Approved fragment_attachment SMILES differs from the executed edit plan")
            attachment = attachment.get("index")
        if attachment != addition.get("attach_atom_index", 0):
            raise ValueError("Approved fragment_attachment differs from the executed edit plan")
        if type(attachment) is not int or not 0 <= attachment < len(facts["atoms"]):
            raise ValueError("fragment_attachment must identify an incoming fragment atom")
        bond_type = plan_value("bond_type", addition.get("bond_type", "SINGLE"))
        fragment_heavy = number("fragment_heavy", facts["heavy_atoms"])
        if fragment_heavy != facts["heavy_atoms"]:
            raise ValueError("Approved fragment_heavy differs from the executed edit plan")
        composition = plan_value("fragment_composition", facts["formula"])
    elif "fragment_heavy" in nodes:
        fragment_heavy = number("fragment_heavy")
        if fragment_heavy != 0:
            raise ValueError("fragment_heavy is nonzero without an incoming fragment")

    if product_heavy != source_heavy + fragment_heavy - removed_heavy or heavy_delta != product_heavy - source_heavy:
        raise ValueError("Heavy atom accounting is inconsistent with approved claims")
    if ring_delta != product_rings - source_rings:
        raise ValueError("Ring accounting is inconsistent with approved claims")

    write("Step 1: Identify the source and the edit.\nThe source contains ")
    slot("source_heavy", source_heavy)
    write(" heavy atoms and ")
    slot("source_rings", source_rings)
    write(" rings.\n")
    if additions:
        write("The attachment site is source atom ")
        slot("anchor", anchor)
        write(" (")
        slot("anchor_element", anchor_element)
        write(").\nThe incoming fragment is ")
        slot("fragment", fragment)
        write(", with attachment atom number ")
        slot("fragment_attachment", attachment + 1)
        write(" (1-based) and ")
        slot("fragment_heavy", fragment_heavy)
        write(" heavy atoms. Its standalone molecular formula is ")
        slot("fragment_composition", composition)
        write(".\n")
    else:
        write("There is no incoming fragment.\n")
    if removed_fragment is not None:
        write("The hydrogen-capped standalone representation of the removed fragment is ")
        slot("remove_fragment", removed_fragment)
        write(". Its source atoms have map IDs ")
    else:
        write("The atoms selected for removal have map IDs ")
    slot("remove_set", remove_set)
    write(" and contribute ")
    slot("remove_heavy", removed_heavy)
    write(" heavy atoms.\n")
    if "fragment_heavy" in nodes:
        write("FORMAL: ADD_HEAVY(")
        slot("fragment_heavy", fragment_heavy)
        write(")\n")
    if "remove_heavy" in nodes:
        write("FORMAL: REMOVE_HEAVY(")
        slot("remove_heavy", removed_heavy)
        write(")\n")
    write("\nStep 2: Apply the edit.\nRemove ")
    if removed_fragment is not None:
        write("group ")
        slot("remove_fragment", removed_fragment)
        write(" (hydrogen-capped representation) at source atoms ")
    else:
        write("source atoms ")
    slot("remove_set", remove_set)
    write(" from the source molecule.\n")
    if additions:
        write("Join fragment ")
        slot("fragment", fragment)
        write(" at its atom number ")
        slot("fragment_attachment", attachment + 1)
        write(" to source atom ")
        slot("anchor", anchor)
        write(" (")
        slot("anchor_element", anchor_element)
        write(") using a ")
        slot("bond_type", bond_type)
        write(" bond.\n")
    for key, verb in (("remove_bonds", "Remove bonds"), ("add_bonds", "Add bonds"), ("change_bonds", "Change bond orders")):
        operations = edit_plan.get(key, [])
        if operations or key in nodes:
            approved = plan_value(key, operations)
            write(verb + " at these source atom maps: ")
            slot(key, json.dumps(approved, sort_keys=True))
            write(".\n")
    write("\nStep 3: Account for the atoms and rings.\nThe product contains ")
    slot("product_heavy", product_heavy)
    write(" heavy atoms: ")
    slot("source_heavy", source_heavy)
    write(" + ")
    slot("fragment_heavy", fragment_heavy)
    write(" - ")
    slot("remove_heavy", removed_heavy)
    write(" = ")
    slot("product_heavy", product_heavy)
    write(".\nThe heavy atom change is ")
    slot("heavy_delta", heavy_delta)
    write(": ")
    slot("product_heavy", product_heavy)
    write(" - ")
    slot("source_heavy", source_heavy)
    write(" = ")
    slot("heavy_delta", heavy_delta)
    write(".\nThe product contains ")
    slot("product_rings", product_rings)
    write(" rings. The ring change is ")
    slot("ring_delta", ring_delta)
    write(": ")
    slot("product_rings", product_rings)
    write(" - ")
    slot("source_rings", source_rings)
    write(" = ")
    slot("ring_delta", ring_delta)
    write(".")

    write("\nFORMAL: SMILES[n_heavy=")
    slot("source_heavy", source_heavy)
    write("]; PRODUCT_SMILES[n_heavy=")
    slot("product_heavy", product_heavy)
    write("]; HEAVY_ATOM_DELTA(")
    slot("heavy_delta", heavy_delta)
    write(")\nFORMAL: SMILES[n_rings=")
    slot("source_rings", source_rings)
    write("]; PRODUCT_SMILES[n_rings=")
    slot("product_rings", product_rings)
    write("]; RING_DELTA(")
    slot("ring_delta", ring_delta)
    write(")")

    if "local_environment" in nodes:
        from .chemistry_tools import apply_edit_plan
        from .local_environment import derive_local_environment
        execution=apply_edit_plan(source_smiles,dict(edit_plan))
        evidence=nodes["local_environment"].get("execution_evidence",{})
        execution["reference_product_smiles"]=evidence.get("reference_product_smiles",execution["product_smiles"])
        local=derive_local_environment(source_smiles,dict(edit_plan),execution)
        if (value("local_environment") != local["smiles"]
                or value("local_environment_center") != local["center_map"]):
            raise ValueError("Approved local environment differs from executed partial structure")
        write("\n\nStep 4: Describe the edited local connectivity.\nThe local environment centered on source atom ")
        slot("local_environment_center",local["center_map"])
        write(" has the hydrogen-capped standalone representation ")
        slot("local_environment",local["smiles"])
        write(".")

    unbound = [name for name, data in nodes.items() if data.get("changed") and name not in bound]
    if unbound:
        raise ValueError("Changed semantic nodes are unbound in the renderer: " + ", ".join(sorted(unbound)))
    return {"text": "".join(pieces), "spans": spans, "edits": [], "bindings": bindings}
