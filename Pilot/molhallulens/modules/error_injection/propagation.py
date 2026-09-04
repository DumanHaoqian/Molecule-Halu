"""Deterministic consistency closure and dependency audit for edited DAGs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rdkit import Chem, rdBase

from molhallulens.core import (
    ClaimValue,
    DependencyType,
    EdgeAuditResult,
    EditingSubtask,
    PlannedMutation,
    PropagationEvent,
    StateDAG,
    StateEdge,
    UnifiedHallucinationPlan,
    ValueProvenance,
)
from molhallulens.infrastructure.chemistry import (
    compute_descriptors,
    isomeric_graph_equivalent,
)


class PropagationError(RuntimeError):
    """Raised when deterministic closure is impossible or remains inconsistent."""


def _subtask(graph: StateDAG) -> EditingSubtask:
    prefix = "mol_edit."
    if not graph.schema.schema_id.startswith(prefix):
        raise PropagationError("unsupported state schema for molecule-edit propagation")
    return EditingSubtask(graph.schema.schema_id[len(prefix) :])


def _fragment_delta(subtask: EditingSubtask, values: dict[str, ClaimValue]) -> int:
    if subtask is EditingSubtask.ADD:
        return int(values["fragment_heavy"].normalized_value)
    if subtask is EditingSubtask.DELETE:
        return -int(values["remove_heavy"].normalized_value)
    return int(values["add_heavy"].normalized_value) - int(
        values["remove_heavy"].normalized_value
    )


def _descriptors(smiles: Any):
    if type(smiles) is not str:
        raise PropagationError("molecular propagation requires a SMILES string")
    try:
        return compute_descriptors(smiles)
    except ValueError as error:
        raise PropagationError("molecular propagation could not parse a SMILES value") from error


def _fragment_heavy(smiles: Any) -> int:
    if smiles == "none":
        return 0
    return _descriptors(smiles).heavy_atom_count


def _mapped_element(indexed_smiles: Any, atom_map: Any) -> str:
    if type(indexed_smiles) is not str or type(atom_map) is not int:
        raise PropagationError("anchor propagation requires indexed SMILES and integer map")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(indexed_smiles)
    if molecule is None:
        raise PropagationError("anchor propagation could not parse indexed source SMILES")
    atom = next(
        (item for item in molecule.GetAtoms() if item.GetAtomMapNum() == atom_map),
        None,
    )
    if atom is None:
        raise PropagationError(f"propagated anchor map {atom_map} does not exist")
    return atom.GetSymbol()


def evaluate_editing_edge(edge: StateEdge, graph: StateDAG) -> bool | None:
    """Evaluate the dependency relations that have deterministic local semantics."""

    values = graph.values
    source = values[edge.source].normalized_value
    target = values[edge.target].normalized_value
    try:
        if edge.relation is DependencyType.MUST_EQUAL:
            return source == target
        if edge.relation is DependencyType.MOLECULARLY_EQUIVALENT_TO:
            return isomeric_graph_equivalent(str(source), str(target))
        if edge.relation is DependencyType.COUNT_OF:
            if edge.target in {"source_heavy", "product_heavy"}:
                return _descriptors(source).heavy_atom_count == target
            if edge.target in {"source_rings", "product_rings"}:
                return _descriptors(source).ring_count == target
            if edge.target in {"fragment_heavy", "remove_heavy", "add_heavy"}:
                return _fragment_heavy(source) == target
        if edge.relation is DependencyType.DELTA_OF:
            if edge.target == "heavy_delta":
                if edge.source in {"source_heavy", "product_heavy"}:
                    expected = (
                        values["product_heavy"].normalized_value
                        - values["source_heavy"].normalized_value
                    )
                else:
                    expected = _fragment_delta(_subtask(graph), dict(values))
                return target == expected
            if edge.target == "ring_delta":
                expected = (
                    values["product_rings"].normalized_value
                    - values["source_rings"].normalized_value
                )
                return target == expected
        if edge.edge_id == "anchor_to_element":
            expected = _mapped_element(
                values["source"].normalized_value,
                values["anchor_idx"].normalized_value,
            )
            return values["anchor_element"].normalized_value == expected
    except (ValueError, PropagationError):
        return False
    return None


def audit_edges(graph: StateDAG) -> tuple[EdgeAuditResult, ...]:
    """Materialize known/unknown status for every edge in stable ID order."""

    return tuple(
        EdgeAuditResult(
            edge_id=edge.edge_id,
            relation=edge.relation,
            status=graph.edge_satisfaction(edge.edge_id, evaluate_editing_edge),
        )
        for edge in sorted(graph.schema.edges, key=lambda item: item.edge_id)
    )


def propagate_deterministic_claims(
    reference_graph: StateDAG,
    root_graph: StateDAG,
    plan: UnifiedHallucinationPlan,
) -> tuple[StateDAG, tuple[PropagationEvent, ...]]:
    """Propagate each non-overlapping root through deterministic claim relations."""

    subtask = _subtask(root_graph)
    values = dict(root_graph.values)
    root_nodes = frozenset(plan.edited_node_ids)
    events: list[PropagationEvent] = []
    propagated_nodes: set[str] = set()

    def set_value(
        target_node_id: str,
        after: Any,
        *,
        sources: tuple[str, ...],
        rule_id: str,
        root: PlannedMutation,
        provenance: ValueProvenance = ValueProvenance.RULE,
    ) -> None:
        old_claim = values[target_node_id]
        if old_claim.normalized_value == after:
            return
        if target_node_id in root_nodes:
            raise PropagationError(
                f"root edit {target_node_id!r} conflicts with deterministic rule {rule_id!r}"
            )
        if target_node_id in propagated_nodes:
            raise PropagationError(
                f"multiple propagation rules attempted to rewrite {target_node_id!r}"
            )
        if type(old_claim.normalized_value) is not type(after):
            raise PropagationError(
                f"propagation rule {rule_id!r} changed the type of {target_node_id!r}"
            )
        values[target_node_id] = replace(
            old_claim,
            raw_value=after,
            normalized_value=after,
            provenance=provenance,
            locally_valid=True,
            oracle_match=False,
            confidence=1.0,
        )
        propagated_nodes.add(target_node_id)
        events.append(
            PropagationEvent(
                event_id=f"{plan.plan_id}.propagation.{len(events) + 1:02d}",
                root_mutation_id=root.mutation_id,
                root_semantic_target_id=root.semantic_target_id,
                source_node_ids=sources,
                target_node_id=target_node_id,
                rule_id=rule_id,
                before=old_claim.normalized_value,
                after=after,
            )
        )

    mutation_by_semantic = {
        mutation.semantic_target_id: mutation for mutation in plan.mutations
    }
    product_root = mutation_by_semantic.get("product") or mutation_by_semantic.get(
        "final_answer"
    )
    if product_root is not None:
        if product_root.semantic_target_id == "product":
            set_value(
                "final_answer",
                values["product"].normalized_value,
                sources=("product",),
                rule_id="product_to_final_answer",
                root=product_root,
            )
        else:
            set_value(
                "product",
                values["final_answer"].normalized_value,
                sources=("final_answer",),
                rule_id="final_answer_to_product",
                root=product_root,
            )
        product_descriptors = _descriptors(values["product"].normalized_value)
        set_value(
            "product_heavy",
            product_descriptors.heavy_atom_count,
            sources=("product",),
            rule_id="product_to_product_heavy",
            root=product_root,
            provenance=ValueProvenance.RDKIT,
        )
        set_value(
            "product_rings",
            product_descriptors.ring_count,
            sources=("product",),
            rule_id="product_to_product_rings",
            root=product_root,
            provenance=ValueProvenance.RDKIT,
        )

    anchor_root = mutation_by_semantic.get("anchor_idx")
    if anchor_root is not None:
        set_value(
            "anchor_element",
            _mapped_element(
                values["source"].normalized_value,
                values["anchor_idx"].normalized_value,
            ),
            sources=("source", "anchor_idx"),
            rule_id="anchor_to_element",
            root=anchor_root,
            provenance=ValueProvenance.RDKIT,
        )

    heavy_nodes = {
        "add_fragment",
        "fragment_heavy",
        "remove_group",
        "remove_group_step1",
        "remove_group_step2",
        "remove_heavy",
        "add_heavy",
        "source_heavy",
        "product_heavy",
    }
    heavy_root = product_root or next(
        (
            mutation
            for mutation in plan.mutations
            if set(mutation.target_node_ids) & heavy_nodes
        ),
        None,
    )
    if heavy_root is not None:
        semantic_id = heavy_root.semantic_target_id
        if semantic_id == "add_fragment":
            set_value(
                "add_heavy" if subtask is EditingSubtask.SUBSTITUTE else "fragment_heavy",
                _fragment_heavy(values["add_fragment"].normalized_value),
                sources=("add_fragment",),
                rule_id="add_fragment_to_heavy",
                root=heavy_root,
                provenance=ValueProvenance.RDKIT,
            )
        if semantic_id == "remove_group":
            remove_node = (
                "remove_group_step2"
                if subtask is EditingSubtask.DELETE
                else "remove_group"
            )
            set_value(
                "remove_heavy",
                _fragment_heavy(values[remove_node].normalized_value),
                sources=(remove_node,),
                rule_id="remove_group_to_heavy",
                root=heavy_root,
                provenance=ValueProvenance.RDKIT,
            )

        fragment_authoritative = semantic_id in {
            "add_fragment",
            "remove_group",
            "fragment_heavy",
            "remove_heavy",
            "add_heavy",
            "source_heavy",
        }
        if fragment_authoritative:
            heavy_delta = _fragment_delta(subtask, values)
            set_value(
                "product_heavy",
                int(values["source_heavy"].normalized_value) + heavy_delta,
                sources=("source_heavy",),
                rule_id="heavy_delta_to_product_heavy",
                root=heavy_root,
            )
        else:
            heavy_delta = int(values["product_heavy"].normalized_value) - int(
                values["source_heavy"].normalized_value
            )
            if subtask is EditingSubtask.ADD:
                if heavy_delta >= 0:
                    set_value(
                        "fragment_heavy",
                        heavy_delta,
                        sources=("source_heavy", "product_heavy"),
                        rule_id="heavy_delta_to_fragment_heavy",
                        root=heavy_root,
                    )
            elif subtask is EditingSubtask.DELETE:
                if heavy_delta <= 0:
                    set_value(
                        "remove_heavy",
                        -heavy_delta,
                        sources=("source_heavy", "product_heavy"),
                        rule_id="heavy_delta_to_remove_heavy",
                        root=heavy_root,
                    )
            else:
                remove_heavy = int(values["remove_heavy"].normalized_value)
                new_add_heavy = remove_heavy + heavy_delta
                if new_add_heavy < 0:
                    set_value(
                        "remove_heavy",
                        int(values["add_heavy"].normalized_value) - heavy_delta,
                        sources=("add_heavy", "source_heavy", "product_heavy"),
                        rule_id="heavy_delta_to_remove_heavy",
                        root=heavy_root,
                    )
                else:
                    set_value(
                        "add_heavy",
                        new_add_heavy,
                        sources=("remove_heavy", "source_heavy", "product_heavy"),
                        rule_id="heavy_delta_to_add_heavy",
                        root=heavy_root,
                    )
        set_value(
            "heavy_delta",
            heavy_delta,
            sources=("source_heavy", "product_heavy"),
            rule_id="heavy_atom_delta",
            root=heavy_root,
        )

    ring_nodes = {"source_rings", "product_rings"}
    ring_root = product_root or next(
        (
            mutation
            for mutation in plan.mutations
            if set(mutation.target_node_ids) & ring_nodes
        ),
        None,
    )
    if ring_root is not None:
        set_value(
            "ring_delta",
            int(values["product_rings"].normalized_value)
            - int(values["source_rings"].normalized_value),
            sources=("source_rings", "product_rings"),
            rule_id="ring_count_delta",
            root=ring_root,
        )

    return (
        StateDAG(
            schema=reference_graph.schema,
            values=values,
            edge_values=reference_graph.edge_values,
        ),
        tuple(events),
    )


__all__ = [
    "PropagationError",
    "audit_edges",
    "evaluate_editing_edge",
    "propagate_deterministic_claims",
]
