"""Paired source-state traces with an unchanged executable reference edit.

N is constructed independently of candidate selection. H describes a different
source graph and applies the same plan to that graph. Counts in H are physically
correct conditional on its reconstructed source, but labels compare their
values with N. A whole source SMILES is one semantic claim, not per-atom truth.
Root IDs record computational ancestry, including computations whose errors
cancel; only unequal displayed values receive error spans.

Both traces omit the optional product-local window. Changing its radius after
source truncation can change a representation without making its local graph
false; that representation change must not create an additional error label.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .chemistry_tools import ChemistryToolError, apply_edit_plan, compare_molecules, inspect_source
from .quality import find_answer_leakage
from .renderer import render_reasoning


_PRELUDE_NODES = {"source_graph", "source_formula"}
_PLAN_ONLY_NODES = {"fragment_heavy", "fragment_composition"}


def _invalid(message: str) -> None:
    raise ChemistryToolError("source_state_invalid_execution", message)


def _replay(source: str, plan: Mapping[str, Any], supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Check all executor fields; harmless controller metadata may be extra."""
    if not isinstance(plan, Mapping) or not isinstance(supplied, Mapping):
        _invalid("The plan and execution must be objects")
    try:
        actual = apply_edit_plan(source, deepcopy(dict(plan)))
    except ChemistryToolError as error:
        _invalid("The supplied plan does not replay: " + str(error))
    mismatches = [key for key, value in actual.items() if supplied.get(key) != value]
    if mismatches:
        _invalid("Execution differs from replay in: " + ", ".join(sorted(mismatches)))
    return actual


def _product_targets(*products: str) -> list[str]:
    return sorted({value for product in products for value in [product, *product.split(".")] if value})


def _no_answer(text: str, products: list[str]) -> None:
    problems = find_answer_leakage(text, products)
    if problems:
        raise ChemistryToolError("source_state_answer_leakage", "; ".join(problems))


def _node(before: Any, after: Any, *, parents: list[str], roots: list[str], root: bool = False) -> dict[str, Any]:
    changed = before != after
    return {"before": deepcopy(before), "after": deepcopy(after), "changed": changed,
            "parents": parents, "root_ids": roots,
            "kind": "unchanged" if not changed else "root_error" if root else "propagated_error"}


def _core_nodes(row: Mapping[str, Any], plan: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    # The controller imports this module; resolve the shared builder only at
    # call time to keep module loading independent of that integration.
    from .orchestrator import build_nodes
    return build_nodes(dict(row), plan, plan, [], execution, execution, include_local_environment=False)


def _with_prelude(core: dict[str, Any], nodes: Mapping[str, Any], *, label_errors: bool) -> dict[str, Any]:
    pieces: list[str] = []
    bindings: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    cursor = 0

    def write(text: str) -> None:
        nonlocal cursor
        pieces.append(text)
        cursor += len(text)

    def slot(name: str) -> None:
        node = nodes[name]
        value = str(node["after"])
        start = cursor
        write(value)
        error = bool(label_errors and node["changed"])
        bindings.append({"start": start, "end": cursor, "text": value, "node_id": name,
                         "origin": "approved_node", "error_annotated": error})
        if error:
            spans.append({"start": start, "end": cursor, "text": value, "node_id": name,
                          "kind": node["kind"], "root_ids": list(node["root_ids"]),
                          "evidence": {"source": "approved_semantic_node",
                                       "before": deepcopy(node["before"]), "after": deepcopy(node["after"])}})

    write("The starting molecule is represented by ")
    slot("source_graph")
    write(". Its molecular formula is ")
    slot("source_formula")
    write(".\n\n")
    for target, original in ((bindings, core["bindings"]), (spans, core["spans"])):
        for item in original:
            shifted = deepcopy(item)
            shifted["start"] += cursor
            shifted["end"] += cursor
            target.append(shifted)
    return {"text": "".join(pieces) + core["text"], "spans": spans, "bindings": bindings, "edits": []}


def render_source_state_reference(
    row: Mapping[str, Any], clean_plan: Mapping[str, Any], clean_execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the true source prelude and canonical core without any H input.

    Returns ``render``, ``nodes``, ``source_facts`` and replayed ``execution``.
    The original row, raw instruction and source reasoning are never mutated.
    """
    source = inspect_source(row["indexed_smiles"])
    clean = _replay(row["indexed_smiles"], clean_plan, clean_execution)
    gt = row.get("gt_smiles")
    if gt and not compare_molecules(gt, clean["product_smiles"])["equivalent"]:
        _invalid("The reference execution does not reproduce the supplied ground truth")
    products = [clean["product_smiles"]]
    # Before rendering, reject no-op plans that would expose the answer.
    _no_answer(source["canonical_smiles"], products)
    nodes = _core_nodes(row, clean_plan, clean)
    core = render_reasoning(row["indexed_smiles"], clean_plan, nodes, label_errors=False)
    _no_answer(core["text"], _product_targets(*products))
    nodes["source_graph"] = _node(source["canonical_smiles"], source["canonical_smiles"], parents=[], roots=[])
    nodes["source_formula"] = _node(source["formula"], source["formula"], parents=["source_graph"], roots=[])
    rendered = _with_prelude(core, nodes, label_errors=False)
    _no_answer(rendered["text"], products)
    return {"render": rendered, "nodes": nodes, "source_facts": source, "execution": clean}


def render_source_state_pair(
    row: Mapping[str, Any], clean_plan: Mapping[str, Any], clean_execution: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one incorrect source reconstruction with the unchanged clean edit.

    Candidate fields are ``source_plan``, ``perceived_source_execution``,
    ``wrong_source_mapped_smiles`` and staged ``execution``. These executions
    replay S→S′ and S′→P′ respectively; no candidate edit can replace clean_plan.
    ``nodes``/``H_nodes`` compare clean before-values with H after-values.
    Contexts preserve the distinct source graphs needed by program checks.
    """
    reference = render_source_state_reference(row, clean_plan, clean_execution)
    source = reference["source_facts"]
    perceived = _replay(row["indexed_smiles"], candidate["source_plan"], candidate["perceived_source_execution"])
    wrong_source = candidate["wrong_source_mapped_smiles"]
    working = inspect_source(wrong_source)
    if working["mapped_smiles"] != inspect_source(perceived["mapped_product_smiles"])["mapped_smiles"]:
        _invalid("The working-source alias disagrees with the reconstructed mapped source")
    if working["canonical_smiles"] == source["canonical_smiles"]:
        _invalid("The reconstructed source must differ from the reference source")
    if len(working["components"]) != len(source["components"]):
        _invalid("Source reconstruction must preserve connected component count")
    altered = _replay(wrong_source, clean_plan, candidate["execution"])
    clean = reference["execution"]
    if compare_molecules(clean["product_smiles"], altered["product_smiles"])["equivalent"]:
        _invalid("The diagnostic requires a different resulting product")
    products = [clean["product_smiles"], altered["product_smiles"]]
    _no_answer(reference["render"]["text"], products)
    _no_answer(reference["render"]["text"].split("\n\n", 1)[1], _product_targets(*products))
    _no_answer(working["canonical_smiles"], products)

    working_row = deepcopy(dict(row))
    working_row["indexed_smiles"] = wrong_source
    conditional_nodes = _core_nodes(working_row, clean_plan, altered)
    clean_nodes = reference["nodes"]
    if set(conditional_nodes) != set(clean_nodes) - _PRELUDE_NODES:
        _invalid("The unchanged edit produced incompatible semantic node schemas")
    nodes: dict[str, Any] = {}
    for name, conditional in conditional_nodes.items():
        before, after = clean_nodes[name]["after"], conditional["after"]
        roots = [] if name in _PLAN_ONLY_NODES else ["r1"]
        parents = deepcopy(conditional["parents"])
        if roots and "source_graph" not in parents:
            parents.append("source_graph")
        nodes[name] = _node(before, after, parents=parents, roots=roots)
    core = render_reasoning(wrong_source, clean_plan, nodes, label_errors=True)
    _no_answer(core["text"], _product_targets(*products))
    nodes["source_graph"] = _node(source["canonical_smiles"], working["canonical_smiles"],
                                  parents=[], roots=["r1"], root=True)
    nodes["source_formula"] = _node(source["formula"], working["formula"],
                                    parents=["source_graph"], roots=["r1"])
    rendered = _with_prelude(core, nodes, label_errors=True)
    _no_answer(rendered["text"], products)
    roots = [{"id": "r1", "node_id": "source_graph", "type": "source_state",
              "before": source["canonical_smiles"], "after": working["canonical_smiles"]}]
    return {"N_render": deepcopy(reference["render"]), "H_render": rendered,
            "nodes": nodes, "roots": roots, "clean_nodes": deepcopy(clean_nodes), "H_nodes": deepcopy(nodes),
            "source_facts": deepcopy(source), "working_source_facts": deepcopy(working),
            "contexts": {
                "N": {"source_smiles": row["indexed_smiles"], "source_facts": deepcopy(source),
                      "edit_plan": deepcopy(dict(clean_plan)), "execution": deepcopy(clean)},
                "H": {"source_smiles": wrong_source, "source_facts": deepcopy(working),
                      "edit_plan": deepcopy(dict(clean_plan)), "execution": deepcopy(altered)}}}
