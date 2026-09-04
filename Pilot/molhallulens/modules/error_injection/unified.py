"""Apply every mutation in a unified multi-point plan directly."""

from __future__ import annotations

from dataclasses import replace

from molhallulens.core import (
    ClaimValue,
    DependencyType,
    InjectedHallucination,
    StateDAG,
    UnifiedHallucinationPlan,
    ValueProvenance,
)
from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)

from .propagation import audit_edges, propagate_deterministic_claims


class HallucinationInjectionError(RuntimeError):
    """Raised when a plan does not match the supplied reference graph."""


class UnifiedHallucinationInjector:
    """Apply sampled roots, close deterministic dependencies, and audit every edge."""

    def __init__(
        self,
        config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
    ) -> None:
        if type(config) is not HallucinationGenerationConfig:
            raise TypeError("config must be HallucinationGenerationConfig")
        self.config = config

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

        root_graph = StateDAG(
            schema=reference_graph.schema,
            values=values,
            edge_values=reference_graph.edge_values,
        )
        if self.config.enable_deterministic_propagation:
            candidate_graph, propagation_events = propagate_deterministic_claims(
                reference_graph,
                root_graph,
                plan,
            )
        else:
            candidate_graph = root_graph
            propagation_events = ()
        edge_audit = audit_edges(candidate_graph)
        if self.config.fail_on_trivial_edge_violation:
            hard_relations = {
                DependencyType.DELTA_OF,
                DependencyType.MUST_EQUAL,
                DependencyType.MOLECULARLY_EQUIVALENT_TO,
            }
            violations = tuple(
                item.edge_id
                for item in edge_audit
                if item.status is False and item.relation in hard_relations
            )
            if violations:
                raise HallucinationInjectionError(
                    f"deterministic propagation left trivial edge violations: {violations}"
                )
        return InjectedHallucination(
            reference_graph=reference_graph,
            candidate_graph=candidate_graph,
            plan=plan,
            propagation_events=propagation_events,
            edge_audit=edge_audit,
        )


__all__ = ["HallucinationInjectionError", "UnifiedHallucinationInjector"]
