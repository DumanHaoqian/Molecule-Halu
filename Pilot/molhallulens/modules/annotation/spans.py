"""Project planned mutations onto every rendered occurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from molhallulens.core import CausalRole, InjectedHallucination, RenderedHallucination


@dataclass(frozen=True, slots=True)
class HallucinationSpan:
    mention_id: str
    mutation_id: str
    semantic_target_id: str
    node_id: str
    operator: str
    component: str
    step_index: int | None
    start: int
    end: int
    text: str
    context_start: int
    context_end: int
    causal_role: CausalRole
    propagation_event_id: str | None = None
    parent_node_id: str | None = None
    diff_opcodes: tuple[tuple[str, int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.mention_id) is not str or not self.mention_id:
            raise ValueError("mention_id must be non-empty text")
        if type(self.causal_role) is not CausalRole:
            raise TypeError("causal_role must be CausalRole")
        if self.parent_node_id and (
            self.causal_role is not CausalRole.PROPAGATED_ERROR
            or not self.node_id.startswith(self.parent_node_id + "__enumeration_")
            or not (self.propagation_event_id or "").startswith("text:")
        ):
            raise ValueError("text-derived spans require a namespaced parent and text propagation event")
        if not (
            0 <= self.context_start <= self.start
            and self.end <= self.context_end
        ):
            raise ValueError("context span must contain the hallucination span")
        if self.causal_role is CausalRole.ROOT_HALLUCINATION:
            if self.propagation_event_id is not None:
                raise ValueError("root spans cannot reference a propagation event")
        elif type(self.propagation_event_id) is not str or not self.propagation_event_id:
            raise ValueError("propagated spans require a propagation_event_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "mutation_id": self.mutation_id,
            "semantic_target_id": self.semantic_target_id,
            "node_id": self.node_id,
            "operator": self.operator,
            "component": self.component,
            "step_index": self.step_index,
            "span": [self.start, self.end],
            "context_span": [self.context_start, self.context_end],
            "text": self.text,
            "diff_opcodes": [
                {
                    "tag": item[0],
                    "reference_span": [item[1], item[2]],
                    "candidate_span": [item[3], item[4]],
                }
                for item in self.diff_opcodes
            ],
            "causal_role": self.causal_role.value,
            "propagation_event_id": self.propagation_event_id,
            "parent_node_id": self.parent_node_id,
            "claim_scope": "text_derived" if self.parent_node_id else "dag",
        }


@dataclass(frozen=True, slots=True)
class ControlSpan:
    """One truth-valued N location paired to exactly one H span."""

    pair_occurrence_id: str
    node_id: str
    component: str
    step_index: int | None
    start: int
    end: int
    text: str
    context_start: int
    context_end: int
    same_char_length: bool
    parent_node_id: str | None = None
    diff_opcodes: tuple[tuple[str, int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.pair_occurrence_id, "pair_occurrence_id"),
            (self.node_id, "node_id"),
            (self.component, "component"),
            (self.text, "text"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.component not in {"reasoning_chain", "final_answer"}:
            raise ValueError("component must be reasoning_chain or final_answer")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("control offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("control offsets must describe a non-empty span")
        if self.end - self.start != len(self.text):
            raise ValueError("control offsets must exactly cover text")
        if not (
            0 <= self.context_start <= self.start
            and self.end <= self.context_end
        ):
            raise ValueError("context span must contain the control span")
        if type(self.same_char_length) is not bool:
            raise TypeError("same_char_length must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_occurrence_id": self.pair_occurrence_id,
            "node_id": self.node_id,
            "component": self.component,
            "step_index": self.step_index,
            "span": [self.start, self.end],
            "context_span": [self.context_start, self.context_end],
            "text": self.text,
            "diff_opcodes": [
                {
                    "tag": item[0],
                    "reference_span": [item[1], item[2]],
                    "candidate_span": [item[3], item[4]],
                }
                for item in self.diff_opcodes
            ],
            "same_char_length": self.same_char_length,
            "parent_node_id": self.parent_node_id,
            "claim_scope": "text_derived" if self.parent_node_id else "dag",
        }


@dataclass(frozen=True, slots=True)
class AnnotatedHallucination:
    rendered: RenderedHallucination
    spans: tuple[HallucinationSpan, ...]
    hallucination_present: bool = True
    control_spans: tuple[ControlSpan, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rendered) is not RenderedHallucination:
            raise TypeError("rendered must be RenderedHallucination")
        spans = tuple(self.spans)
        controls = tuple(self.control_spans)
        if type(self.hallucination_present) is not bool:
            raise TypeError("hallucination_present must be bool")
        if any(type(item) is not HallucinationSpan for item in spans):
            raise TypeError("spans must contain HallucinationSpan values")
        if any(type(item) is not ControlSpan for item in controls):
            raise TypeError("control_spans must contain ControlSpan values")
        if self.hallucination_present:
            if not spans:
                raise ValueError("positive records must have hallucination spans")
            if controls:
                raise ValueError("positive records cannot have control spans")
        else:
            if spans:
                raise ValueError("negative records cannot have hallucination spans")
            if not controls:
                raise ValueError("negative records must have paired control spans")
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "control_spans", controls)


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
            causal_node = mention.parent_node_id or mention.node_id
            event = event_by_node.get(causal_node)
            if event is None:
                mutation = mutation_by_node[causal_node]
                operator = mutation.operator
                propagation_event_id = None
                causal_role = CausalRole.ROOT_HALLUCINATION
            else:
                mutation = mutation_by_id[event.root_mutation_id]
                operator = event.rule_id
                propagation_event_id = event.event_id
                causal_role = CausalRole.PROPAGATED_ERROR
            if mention.parent_node_id:
                operator = "text.enumeration_count_propagation"
                propagation_event_id = f"text:{mention.mention_id}"
                causal_role = CausalRole.PROPAGATED_ERROR
            if mention.causal_role is not causal_role:
                raise ValueError("rendered mention causal role disagrees with injection")
            spans.append(
                HallucinationSpan(
                    mention_id=mention.mention_id,
                    mutation_id=mutation.mutation_id,
                    semantic_target_id=mutation.semantic_target_id,
                    node_id=mention.node_id,
                    operator=operator,
                    component=mention.component,
                    step_index=mention.step_index,
                    start=mention.start,
                    end=mention.end,
                    text=mention.value,
                    context_start=mention.context_start,
                    context_end=mention.context_end,
                    causal_role=causal_role,
                    propagation_event_id=propagation_event_id,
                    parent_node_id=mention.parent_node_id,
                    diff_opcodes=mention.diff_opcodes,
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
        return AnnotatedHallucination(
            rendered=rendered,
            spans=spans,
            hallucination_present=True,
        )

    def annotate_negative(
        self,
        rendered: RenderedHallucination,
        positive: AnnotatedHallucination,
    ) -> AnnotatedHallucination:
        """Create an explicitly negative annotation with paired truth controls."""

        if type(rendered) is not RenderedHallucination:
            raise TypeError("rendered must be RenderedHallucination")
        if type(positive) is not AnnotatedHallucination:
            raise TypeError("positive must be AnnotatedHallucination")
        if not positive.hallucination_present:
            raise ValueError("positive annotation must be hallucinated")
        controls_by_id = {item.mention_id: item for item in rendered.mentions}
        if len(controls_by_id) != len(rendered.mentions):
            raise ValueError("negative rendered control mention IDs must be unique")
        expected_ids = {span.mention_id for span in positive.spans}
        if set(controls_by_id) != expected_ids:
            raise ValueError("negative controls must map one-to-one to positive spans")
        controls = []
        for span in positive.spans:
            mention = controls_by_id[span.mention_id]
            if mention.hallucinated or mention.causal_role is not None:
                raise ValueError("negative controls cannot be hallucinated mentions")
            if (
                mention.node_id != span.node_id
                or mention.component != span.component
                or mention.step_index != span.step_index
                or mention.diff_opcodes != span.diff_opcodes
            ):
                raise ValueError("negative control metadata disagrees with positive span")
            controls.append(
                ControlSpan(
                    pair_occurrence_id=span.mention_id,
                    node_id=span.node_id,
                    component=span.component,
                    step_index=span.step_index,
                    start=mention.start,
                    end=mention.end,
                    text=mention.value,
                    context_start=mention.context_start,
                    context_end=mention.context_end,
                    same_char_length=len(span.text) == len(mention.value),
                    parent_node_id=span.parent_node_id,
                    diff_opcodes=mention.diff_opcodes,
                )
            )
        return AnnotatedHallucination(
            rendered=rendered,
            spans=(),
            hallucination_present=False,
            control_spans=tuple(controls),
        )


__all__ = [
    "AnnotatedHallucination",
    "ControlSpan",
    "HallucinationSpan",
    "UnifiedHallucinationAnnotator",
]
