"""Apply every mutation in a unified multi-point plan directly."""

from __future__ import annotations

from dataclasses import replace

from molhallulens.core import (
    ClaimValue,
    InjectedHallucination,
    StateDAG,
    UnifiedHallucinationPlan,
    ValueProvenance,
)


class HallucinationInjectionError(RuntimeError):
    """Raised when a plan does not match the supplied reference graph."""


class UnifiedHallucinationInjector:
    """Apply every planned semantic edit exactly once and record no implicit edits."""

    def apply(
        self,
        reference_graph: StateDAG,
        plan: UnifiedHallucinationPlan,
    ) -> InjectedHallucination:
        if type(reference_graph) is not StateDAG:
            raise TypeError("reference_graph must be StateDAG")
        if type(plan) is not UnifiedHallucinationPlan:
            raise TypeError("plan must be UnifiedHallucinationPlan")

        values = dict(reference_graph.values)
        for mutation in plan.mutations:
            for node_id in mutation.target_node_ids:
                if node_id not in reference_graph.schema.nodes_by_id:
                    raise HallucinationInjectionError(
                        f"planned node {node_id!r} is not in the reference schema"
                    )
                spec = reference_graph.schema.nodes_by_id[node_id]
                if not spec.mutable:
                    raise HallucinationInjectionError(
                        f"planned node {node_id!r} is immutable"
                    )
                old_claim = reference_graph.values[node_id]
                if (
                    old_claim.value_type is not mutation.value_type
                    or old_claim.normalized_value != mutation.before
                ):
                    raise HallucinationInjectionError(
                        f"planned before-value does not match node {node_id!r}"
                    )
                values[node_id] = replace(
                    old_claim,
                    raw_value=mutation.after,
                    normalized_value=mutation.after,
                    provenance=ValueProvenance.RULE,
                    locally_valid=True,
                    oracle_match=False,
                    confidence=1.0,
                )

        candidate_graph = StateDAG(
            schema=reference_graph.schema,
            values=values,
            edge_values=reference_graph.edge_values,
        )
        return InjectedHallucination(
            reference_graph=reference_graph,
            candidate_graph=candidate_graph,
            plan=plan,
        )


__all__ = ["HallucinationInjectionError", "UnifiedHallucinationInjector"]
