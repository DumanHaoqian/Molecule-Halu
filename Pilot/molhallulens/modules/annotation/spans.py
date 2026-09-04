"""Project planned mutations onto every rendered occurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molhallulens.core import CausalRole, InjectedHallucination, RenderedHallucination


@dataclass(frozen=True, slots=True)
class HallucinationSpan:
    mutation_id: str
    semantic_target_id: str
    node_id: str
    operator: str
    component: str
    step_index: int | None
    start: int
    end: int
    text: str
    causal_role: CausalRole
    propagation_event_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.causal_role) is not CausalRole:
            raise TypeError("causal_role must be CausalRole")
        if self.causal_role is CausalRole.ROOT_HALLUCINATION:
            if self.propagation_event_id is not None:
                raise ValueError("root spans cannot reference a propagation event")
        elif type(self.propagation_event_id) is not str or not self.propagation_event_id:
            raise ValueError("propagated spans require a propagation_event_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "semantic_target_id": self.semantic_target_id,
            "node_id": self.node_id,
            "operator": self.operator,
            "component": self.component,
            "step_index": self.step_index,
            "span": [self.start, self.end],
            "text": self.text,
            "causal_role": self.causal_role.value,
            "propagation_event_id": self.propagation_event_id,
        }


@dataclass(frozen=True, slots=True)
class AnnotatedHallucination:
    rendered: RenderedHallucination
    spans: tuple[HallucinationSpan, ...]

    def __post_init__(self) -> None:
        if type(self.rendered) is not RenderedHallucination:
            raise TypeError("rendered must be RenderedHallucination")
        spans = tuple(self.spans)
        if not spans or any(type(item) is not HallucinationSpan for item in spans):
            raise ValueError("every generated record must have hallucination spans")
        object.__setattr__(self, "spans", spans)


class UnifiedHallucinationAnnotator:
    """Label every rendered root and deterministically propagated claim."""

    def annotate(
        self,
        rendered: RenderedHallucination,
        injected: InjectedHallucination,
    ) -> AnnotatedHallucination:
        if type(rendered) is not RenderedHallucination:
            raise TypeError("rendered must be RenderedHallucination")
        if type(injected) is not InjectedHallucination:
            raise TypeError("injected must be InjectedHallucination")
        plan = injected.plan
        mutation_by_node = {
            node_id: mutation
            for mutation in plan.mutations
            for node_id in mutation.target_node_ids
        }
        event_by_node = {
            event.target_node_id: event for event in injected.propagation_events
        }
        mutation_by_id = {
            mutation.mutation_id: mutation for mutation in plan.mutations
        }
        spans = []
        for mention in rendered.mentions:
            if not mention.hallucinated:
                continue
            event = event_by_node.get(mention.node_id)
            if event is None:
                mutation = mutation_by_node[mention.node_id]
                operator = mutation.operator
                propagation_event_id = None
                causal_role = CausalRole.ROOT_HALLUCINATION
            else:
                mutation = mutation_by_id[event.root_mutation_id]
                operator = event.rule_id
                propagation_event_id = event.event_id
                causal_role = CausalRole.PROPAGATED_ERROR
            if mention.causal_role is not causal_role:
                raise ValueError("rendered mention causal role disagrees with injection")
            spans.append(
                HallucinationSpan(
                    mutation_id=mutation.mutation_id,
                    semantic_target_id=mutation.semantic_target_id,
                    node_id=mention.node_id,
                    operator=operator,
                    component=mention.component,
                    step_index=mention.step_index,
                    start=mention.start,
                    end=mention.end,
                    text=mention.value,
                    causal_role=causal_role,
                    propagation_event_id=propagation_event_id,
                )
            )
        spans = tuple(spans)
        covered_nodes = {span.node_id for span in spans}
        missing = set(injected.changed_node_ids) - covered_nodes
        if missing:
            raise ValueError(
                "changed nodes have no rendered hallucination span: "
                f"{sorted(missing)}"
            )
        return AnnotatedHallucination(rendered=rendered, spans=spans)


__all__ = [
    "AnnotatedHallucination",
    "HallucinationSpan",
    "UnifiedHallucinationAnnotator",
]
