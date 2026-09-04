"""Project planned mutations onto every rendered occurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molhallulens.core import RenderedHallucination, UnifiedHallucinationPlan


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
    """Label all text mentions of every directly edited semantic point."""

    def annotate(
        self,
        rendered: RenderedHallucination,
        plan: UnifiedHallucinationPlan,
    ) -> AnnotatedHallucination:
        if type(rendered) is not RenderedHallucination:
            raise TypeError("rendered must be RenderedHallucination")
        if type(plan) is not UnifiedHallucinationPlan:
            raise TypeError("plan must be UnifiedHallucinationPlan")
        mutation_by_node = {
            node_id: mutation
            for mutation in plan.mutations
            for node_id in mutation.target_node_ids
        }
        spans = tuple(
            HallucinationSpan(
                mutation_id=mutation_by_node[mention.node_id].mutation_id,
                semantic_target_id=mutation_by_node[
                    mention.node_id
                ].semantic_target_id,
                node_id=mention.node_id,
                operator=mutation_by_node[mention.node_id].operator,
                component=mention.component,
                step_index=mention.step_index,
                start=mention.start,
                end=mention.end,
                text=mention.value,
            )
            for mention in rendered.mentions
            if mention.hallucinated
        )
        covered_nodes = {span.node_id for span in spans}
        missing = set(plan.edited_node_ids) - covered_nodes
        if missing:
            raise ValueError(f"edited nodes have no rendered hallucination span: {sorted(missing)}")
        return AnnotatedHallucination(rendered=rendered, spans=spans)


__all__ = [
    "AnnotatedHallucination",
    "HallucinationSpan",
    "UnifiedHallucinationAnnotator",
]
