"""Local evidence gates for frozen references and approved numeric claims.

Pass means that the implemented checks passed, not that reaction semantics or
arbitrary natural-language assertions were proved. Missing required numeric
checks return ``unknown``. Agent review remains necessary for chemical names,
intent, and unsupported prose. Product-equivalent fragments are conservatively
flagged as leakage: this gate cannot prove that an identical molecule is merely
an ingredient, and never silently exempts such a span.
"""
from __future__ import annotations

import re
from typing import Any

from rdkit import Chem, rdBase

from .chemistry_tools import (
    ChemistryToolError, apply_edit_plan, compare_molecules, describe_fragment, inspect_source,
)


def _result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [c for c in checks if c["status"] == "fail"]
    unknowns = [c for c in checks if c["status"] == "unknown"]
    return {"status": "fail" if failures else "unknown" if unknowns else "pass",
            "checks": checks, "failures": failures, "unknowns": unknowns}


def _check(checks, name, node, observed, expected, evidence="", *, missing=False):
    status = "unknown" if missing or expected is None else "pass" if observed == expected and type(observed) is type(expected) else "fail"
    checks.append({"check": name, "node_id": node, "status": status,
                   "observed": observed, "expected": expected, "evidence": evidence})


_NUMBER = r"([+-]?\d+)"
_QUOTED = r'''(?:"[^"\n]*"|'[^'\n]*')'''
_FORMAL_NUMBERS = [
    ("source_heavy", rf"(?<![\w])SMILES\s*\[\s*n_heavy\s*=\s*{_NUMBER}\s*\]"),
    ("product_heavy", rf"\bPRODUCT_SMILES\s*\[\s*n_heavy\s*=\s*{_NUMBER}\s*\]"),
    ("source_rings", rf"(?<![\w])SMILES\s*\[\s*n_rings\s*=\s*{_NUMBER}\s*\]"),
    ("product_rings", rf"\bPRODUCT_SMILES\s*\[\s*n_rings\s*=\s*{_NUMBER}\s*\]"),
    ("heavy_delta", rf"\bHEAVY_ATOM_DELTA\s*\(\s*{_NUMBER}\s*\)"),
    ("ring_delta", rf"\bRING_DELTA\s*\(\s*{_NUMBER}\s*\)"),
    ("fragment_heavy", rf"\bADD_FRAGMENT\s*\(\s*smiles\s*=\s*{_QUOTED}\s*,\s*heavy_atoms\s*=\s*{_NUMBER}\s*\)"),
    ("fragment_heavy", rf"\bADD_HEAVY\s*\(\s*{_NUMBER}\s*\)"),
    ("remove_heavy", rf"\b(?:REMOVE_HEAVY|HEAVY_ATOMS)\s*\(\s*{_NUMBER}\s*\)"),
]


def validate_formal_claims(text: str, nodes: dict, source_facts: dict) -> dict[str, Any]:
    """Check every recognized occurrence against approved ``nodes[*].after``.

    A false source-heavy root is used as the arithmetic parent without changing
    source RDKit facts. ``fragment_heavy`` (or ``add_heavy``) and ``remove_heavy``
    are distinct variables. Six source/product/delta FORMAL checks are required;
    incoming/removed counts are also required when those nodes are supplied.
    """
    expected = {name: node.get("after") for name, node in nodes.items() if isinstance(node, dict)}
    if "fragment_heavy" not in expected and "add_heavy" in expected:
        expected["fragment_heavy"] = expected["add_heavy"]
    expected.setdefault("source_heavy", source_facts.get("heavy_atoms"))
    expected.setdefault("source_rings", source_facts.get("rings"))
    checks: list[dict[str, Any]] = []
    found = set()
    for node, pattern in _FORMAL_NUMBERS:
        for match in re.finditer(pattern, text, re.I):
            found.add(node)
            _check(checks, "formal_numeric_claim", node, int(match[1]), expected.get(node), match[0])
    required = {"source_heavy", "product_heavy", "heavy_delta", "source_rings", "product_rings", "ring_delta"}
    required.update(name for name in ("fragment_heavy", "remove_heavy") if name in expected)
    for node in sorted(required - found):
        _check(checks, "missing_formal_claim", node, None, expected.get(node), missing=True)

    # Common prose repetitions are checked independently of the FORMAL blocks.
    for subject, prefix in (("source", "source"), ("product", "product")):
        for noun, suffix in (("heavy atoms?", "heavy"), ("rings?", "rings")):
            pattern = rf"\b{subject}\b(?:\s+molecule)?(?:\s+also)?\s+(?:has|contains|comprises)\s+{_NUMBER}\s+{noun}\b"
            for match in re.finditer(pattern, text, re.I):
                node = f"{prefix}_{suffix}"
                _check(checks, "prose_numeric_claim", node, int(match[1]), expected.get(node), match[0])
    for variable, product, source, delta in (
        ("HEAVY_ATOM_DELTA", "product_heavy", "source_heavy", "heavy_delta"),
        ("RING_DELTA", "product_rings", "source_rings", "ring_delta"),
    ):
        pattern = rf"\b{variable}\s*=\s*{_NUMBER}\s*-\s*{_NUMBER}\s*=\s*{_NUMBER}"
        for match in re.finditer(pattern, text, re.I):
            a, b, c = map(int, match.groups())
            for name, value in ((product, a), (source, b), (delta, c)):
                _check(checks, "arithmetic_operand", name, value, expected.get(name), match[0])
            _check(checks, "arithmetic_subtraction", delta, c, a - b, match[0])
    maps = [a.get("atom_map") for a in source_facts.get("atoms", [])]
    maximum = max(maps) if maps and all(type(v) is int for v in maps) else None
    for match in re.finditer(r"\b(?:max(?:imum)?\s+(?:atom[- ]?)?map(?:\s+number)?)\s*:?\s*(\d+)", text, re.I):
        _check(checks, "source_maximum_map", "source_map_max", int(match[1]), maximum, match[0])
    return _result(checks)


def _reference_fields(row: dict) -> dict[str, Any]:
    """Map the three existing parsed-state schemas to shared semantic names."""
    subtask = row.get("subtask")
    if subtask not in {"add", "delete", "substitute"}:
        raise ChemistryToolError("reference_invalid", "Unknown reference subtask")
    heavy_step, ring_step, product_step = (5, 6, 4) if subtask == "substitute" else (4, 5, 3)
    fields = {
        "source_heavy": f"step{heavy_step}_n_heavy_src",
        "product_heavy": f"step{heavy_step}_n_heavy_prod",
        "heavy_delta": f"step{heavy_step}_heavy_delta",
        "source_rings": f"step{ring_step}_n_rings_src",
        "product_rings": f"step{ring_step}_n_rings_prod",
        "ring_delta": f"step{ring_step}_ring_delta",
        "reference_product": f"step{product_step}_product_smiles",
        "anchor": "step1_anchor_idx", "anchor_element": "step1_anchor_element",
    }
    if subtask == "add":
        fields.update(fragment="step2_frag_smiles", fragment_heavy="step2_heavy_atoms")
    elif subtask == "delete":
        fields.update(remove_fragment="step2_remove_smiles", remove_heavy="step2_heavy_atoms")
    else:
        fields.update(fragment="step1_add_fragment_smiles", fragment_heavy="step3_add_heavy",
                      remove_fragment="step1_remove_group_smiles", remove_heavy="step2_remove_heavy")
    return fields


def validate_reference(row: dict) -> dict[str, Any]:
    """Recompute source, GT, parsed-state and frozen-text numeric evidence.

    Stored outcome flags and cached RDKit fields are never accepted as truth.
    The original product must equal GT, including components and stereochemistry.
    This does not certify that a named reagent or reaction matches the instruction.
    """
    checks: list[dict[str, Any]] = []
    try:
        fields = _reference_fields(row)
        state = row.get("state", {})
        source = inspect_source(row["indexed_smiles"])
        product = inspect_source(row["gt_smiles"])
        expected = {
            "source_heavy": source["heavy_atoms"], "product_heavy": product["heavy_atoms"],
            "heavy_delta": product["heavy_atoms"] - source["heavy_atoms"],
            "source_rings": source["rings"], "product_rings": product["rings"],
            "ring_delta": product["rings"] - source["rings"],
        }
        anchor = state.get(fields["anchor"])
        atoms = {a["atom_map"]: a for a in source["atoms"]}
        _check(checks, "source_anchor_exists", "anchor", type(anchor) is int and anchor in atoms, True,
               fields["anchor"], missing=anchor is None)
        if type(anchor) is int and anchor in atoms:
            expected["anchor_element"] = atoms[anchor]["element"]
        for fragment_node, count_node in (("fragment", "fragment_heavy"), ("remove_fragment", "remove_heavy")):
            if fragment_node not in fields:
                continue
            fragment = state.get(fields[fragment_node])
            if fragment is None:
                _check(checks, "missing_fragment", fragment_node, None, None, missing=True)
            else:
                expected[count_node] = describe_fragment(fragment)["heavy_atoms"]
        for node, value in expected.items():
            key = fields[node]
            _check(checks, "parsed_reference_fact", node, state.get(key), value, key, missing=key not in state)
        original_product = state.get(fields["reference_product"])
        if original_product is None:
            _check(checks, "reference_product_equals_gt", "reference_product", None, True, missing=True)
        else:
            _check(checks, "reference_product_equals_gt", "reference_product",
                   compare_molecules(original_product, row["gt_smiles"])["equivalent"], True)
        cache_fields = {
            "rdkit_src_heavy": "source_heavy", "rdkit_prod_heavy": "product_heavy",
            "rdkit_src_rings": "source_rings", "rdkit_prod_rings": "product_rings",
            "rdkit_frag_heavy": "fragment_heavy", "rdkit_add_heavy": "fragment_heavy",
            "rdkit_group_heavy": "remove_heavy", "rdkit_remove_heavy": "remove_heavy",
            "rdkit_elem_at_anchor": "anchor_element",
        }
        for key, node in cache_fields.items():
            if key in state:
                _check(checks, "cached_fact_recomputed", node, state[key], expected.get(node), key)
        if "step1_remove_group" in state and "step2_remove_smiles" in state:
            _check(checks, "removed_fragment_repetition", "remove_fragment",
                   compare_molecules(state["step1_remove_group"], state["step2_remove_smiles"])["equivalent"], True)
        text = row.get("N_raw")
        if not isinstance(text, str) or not text:
            _check(checks, "frozen_reference_text", "N_raw", text, None, missing=True)
        else:
            numeric = validate_formal_claims(text, {k: {"after": v} for k, v in expected.items()}, source)
            checks.extend(numeric["checks"])
            for match in re.finditer(r"\bANCHOR\s*\(\s*idx\s*=\s*(\d+)(?:\s*,\s*element\s*=\s*['\"]([A-Za-z]+)['\"])?", text):
                _check(checks, "formal_anchor_map", "anchor", int(match[1]), anchor, match[0])
                if match[2]:
                    _check(checks, "formal_anchor_element", "anchor_element", match[2], expected.get("anchor_element"), match[0])
            for macro, node in (("ADD_FRAGMENT", "fragment"), ("REMOVE_GROUP", "remove_fragment")):
                if node not in fields:
                    continue
                pattern = rf"\b{macro}\s*\(\s*(?:smiles\s*=\s*)?({_QUOTED})"
                for match in re.finditer(pattern, text):
                    expected_smiles = state.get(fields[node])
                    if expected_smiles:
                        equal = compare_molecules(match[1][1:-1], expected_smiles)["equivalent"]
                        _check(checks, "formal_fragment_identity", node, equal, True, match[0])
    except (ChemistryToolError, KeyError, TypeError, ValueError) as error:
        _check(checks, "reference_structure", "reference", str(error), "valid parseable reference")
    return _result(checks)


def _molecular_tokens(text: str) -> set[str]:
    candidates: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9@+\[\]().=#$%:/\\-]+", text):
        token = raw.strip(".,:;")
        if token:
            candidates.add(token)
        # Human parentheses/brackets may wrap a complete molecule. Keep the
        # original too, because atom brackets and branch parentheses are SMILES.
        for _ in range(3):
            if len(token) > 2 and ((token[0], token[-1]) in {("(", ")"), ("[", "]")}):
                token = token[1:-1].strip(".,:;")
                candidates.add(token)
            else:
                break
    return candidates


def find_answer_leakage(text: str, products: list[str]) -> list[str]:
    """Find product-equivalent complete tokens and overt answer/intervention cues.

    Equivalence is component- and stereo-sensitive and uses complete molecular
    tokens, never substring matches against fragments of a larger molecule.
    """
    reasons = []
    cues = r"ground[ _-]truth|\boracle\b|counterfactual|hallucinat|corrupt(?:ion|ed)|(?:deliberate(?:ly)?|intentional(?:ly)?)\s+(?:wrong|false)|known\s+(?:wrong|false)|ignore\s+(?:the\s+)?(?:original\s+)?(?:instruction|question)|</?answer>|\b(?:final\s+)?answer\s*:"
    if re.search(cues, text, re.I):
        reasons.append("answer_leakage: explicit answer, oracle, or intervention cue")
    if re.search(r"\bPRODUCT_SMILES\s*\(|\b(?:final\s+)?product(?:\s+SMILES)?\s*:|\b(?:final\s+product|product\s+name)\s*(?:is\b|=)", text, re.I):
        reasons.append("answer_leakage: explicit product-answer clause")
    canonical_products = set()
    for product in products:
        if product:
            canonical_products.add(compare_molecules(product, product)["canonical_a"])
    with rdBase.BlockLogs():
        for token in sorted(_molecular_tokens(text)):
            if not re.search(r"[A-Za-z]", token):
                continue
            try:
                # No names or trailing descriptive text may be accepted by RDKit.
                params = Chem.SmilesParserParams()
                params.parseName = False
                mol = Chem.MolFromSmiles(token, params)
                if mol is None:
                    continue
                canonical = compare_molecules(token, token)["canonical_a"]
            except ChemistryToolError:
                continue
            if canonical in canonical_products:
                reasons.append(f"answer_leakage: full product molecule token {token!r}")
    return reasons


def deterministic_reference_plan(row: dict) -> dict[str, Any]:
    """Recover a bounded, unambiguous edit for the already validated reference.

    Only single-anchor additions, removals and substitutions are supported.
    A removed group must match the stated fragment and have exactly one boundary
    bond to the stated anchor. Incoming attachment atoms are enumerated by
    symmetry class; chemically indistinguishable new atom indices use the lowest
    index with their alternatives recorded. Different original removed-map sets
    remain ambiguous even if they give the same product.

    GT verifies the immutable reference only. This helper must never be used to
    select corruptions or to repair a conflicting reference. Up to 64 executable
    candidate plans are considered; semantic review of original N is still needed.
    """
    validation = validate_reference(row)
    if validation["status"] != "pass":
        raise ChemistryToolError("reference_invalid", "Frozen reference failed local factual checks")
    fields = _reference_fields(row)
    state = row["state"]
    source_info = inspect_source(row["indexed_smiles"])
    source = Chem.MolFromSmiles(source_info["mapped_smiles"])
    anchor_map = state[fields["anchor"]]
    anchor_index = next(a.GetIdx() for a in source.GetAtoms() if a.GetAtomMapNum() == anchor_map)
    removal_choices: list[tuple[list[int], int]] = [([], 0)]
    if "remove_fragment" in fields:
        query = Chem.MolFromSmiles(state[fields["remove_fragment"]])
        matches = source.GetSubstructMatches(query, uniquify=True, useChirality=True, maxMatches=129)
        if len(matches) > 128:
            raise ChemistryToolError("reference_ambiguous", "Removed-group match limit exceeded")
        removal_choices = []
        seen = set()
        for match in matches:
            removed = set(match)
            if anchor_index in removed:
                continue
            boundary = [b for b in source.GetBonds()
                        if (b.GetBeginAtomIdx() in removed) != (b.GetEndAtomIdx() in removed)]
            if len(boundary) != 1:
                continue
            bond = boundary[0]
            retained = bond.GetEndAtomIdx() if bond.GetBeginAtomIdx() in removed else bond.GetBeginAtomIdx()
            order = bond.GetBondTypeAsDouble()
            if retained != anchor_index or order not in (1, 2, 3):
                continue
            atom_maps = tuple(sorted(source.GetAtomWithIdx(i).GetAtomMapNum() for i in removed))
            if atom_maps not in seen:
                seen.add(atom_maps)
                removal_choices.append((list(atom_maps), int(order)))
        if not removal_choices:
            raise ChemistryToolError("reference_invalid", "Stated removed group has no supported single-anchor match")
    attachment_choices: list[tuple[int | None, list[int]]] = [(None, [])]
    if "fragment" in fields:
        fragment = Chem.MolFromSmiles(state[fields["fragment"]])
        ranks = list(Chem.CanonicalRankAtoms(fragment, breakTies=False, includeChirality=True, includeIsotopes=True))
        classes: dict[int, list[int]] = {}
        for index, rank in enumerate(ranks):
            classes.setdefault(rank, []).append(index)
        attachment_choices = [(indices[0], indices) for indices in classes.values()]
    if len(removal_choices) * len(attachment_choices) > 64:
        raise ChemistryToolError("reference_ambiguous", "Reference edit candidate limit exceeded")
    accepted = []
    candidate_count = 0
    rejected = []
    for removed_maps, removed_order in removal_choices:
        for attach_index, alternatives in attachment_choices:
            candidate_count += 1
            plan: dict[str, Any] = {}
            if removed_maps:
                plan["remove_atom_maps"] = removed_maps
            if attach_index is not None:
                plan["add_fragments"] = [{"smiles": state[fields["fragment"]],
                    "attach_atom_index": attach_index, "anchor_map": anchor_map, "bond_type": "SINGLE"}]
            hydrogen_delta = removed_order - (1 if attach_index is not None else 0)
            if hydrogen_delta:
                plan["adjust_hydrogens"] = [{"atom_map": anchor_map, "delta": hydrogen_delta}]
            try:
                execution = apply_edit_plan(source_info["mapped_smiles"], plan)
                if compare_molecules(execution["product_smiles"], row["gt_smiles"])["equivalent"]:
                    accepted.append({"edit_plan": plan, "execution": execution,
                                     "attachment_alternatives": alternatives})
                else:
                    rejected.append({"edit_plan": plan, "reason": "reference_product_mismatch"})
            except ChemistryToolError as error:
                rejected.append({"edit_plan": plan, "reason": error.code})
    if not accepted:
        raise ChemistryToolError("reference_invalid", "No supported reference edit reproduces the immutable reference product")
    if len(accepted) != 1:
        raise ChemistryToolError("reference_ambiguous", "Multiple distinct reference edits reproduce the reference product")
    return {**accepted[0], "method": "bounded_reference_graph_match", "candidate_count": candidate_count,
            "rejected_candidates": rejected, "reference_checks": validation}
