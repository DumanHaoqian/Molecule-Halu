"""Domain objects for the single configurable multi-point hallucination path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .enums import MutationCategory, ValueType
from .state_dag import FrozenMap, StateDAG, deep_freeze


@dataclass(frozen=True, slots=True)
class PlannedMutation:
    """One semantic edit; it may update several repeated DAG nodes/mentions."""

    mutation_id: str
    semantic_target_id: str
    target_node_ids: tuple[str, ...]
    value_type: ValueType
    mutation_category: MutationCategory
    operator: str
    before: Any
    after: Any
    magnitude: int | float | None = None
    similarity: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.mutation_id, "mutation_id"),
            (self.semantic_target_id, "semantic_target_id"),
            (self.operator, "operator"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        targets = tuple(self.target_node_ids)
        if not targets or any(type(item) is not str or not item for item in targets):
            raise ValueError("target_node_ids must contain non-empty node IDs")
        if len(targets) != len(set(targets)):
            raise ValueError("target_node_ids cannot contain duplicates")
        if type(self.value_type) is not ValueType:
            raise TypeError("value_type must be ValueType")
        if type(self.mutation_category) is not MutationCategory:
            raise TypeError("mutation_category must be MutationCategory")
        before = deep_freeze(self.before)
        after = deep_freeze(self.after)
        if type(before) is not type(after) or before == after:
            raise ValueError("a planned mutation must change a value without changing its type")
        if self.magnitude is not None and (
            type(self.magnitude) not in {int, float}
            or not isfinite(float(self.magnitude))
        ):
            raise ValueError("magnitude must be finite numeric or None")
        if self.similarity is not None and (
            type(self.similarity) not in {int, float}
            or not isfinite(float(self.similarity))
            or not 0.0 <= float(self.similarity) <= 1.0
        ):
            raise ValueError("similarity must be in [0, 1] or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "target_node_ids", targets)
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "metadata", FrozenMap(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "semantic_target_id": self.semantic_target_id,
            "target_node_ids": list(self.target_node_ids),
            "value_type": self.value_type.value,
            "mutation_category": self.mutation_category.value,
            "operator": self.operator,
            "before": self.before,
            "after": self.after,
            "magnitude": self.magnitude,
            "similarity": self.similarity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class UnifiedHallucinationPlan:
    """All independently selected edits for one always-hallucinated record."""

    plan_id: str
    origin_id: str
    variant_index: int
    derived_seed: int
    requested_edit_count: int
    mutations: tuple[PlannedMutation, ...]

    def __post_init__(self) -> None:
        for value, name in ((self.plan_id, "plan_id"), (self.origin_id, "origin_id")):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.variant_index) is not int or self.variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer")
        if type(self.derived_seed) is not int or self.derived_seed < 0:
            raise ValueError("derived_seed must be a non-negative integer")
        if type(self.requested_edit_count) is not int or self.requested_edit_count < 1:
            raise ValueError("requested_edit_count must be positive")
        mutations = tuple(self.mutations)
        if len(mutations) != self.requested_edit_count:
            raise ValueError("mutations must exactly match requested_edit_count")
        if any(type(item) is not PlannedMutation for item in mutations):
            raise TypeError("mutations must contain PlannedMutation values")
        semantic_targets = tuple(item.semantic_target_id for item in mutations)
        node_targets = tuple(
            node_id for item in mutations for node_id in item.target_node_ids
        )
        if len(semantic_targets) != len(set(semantic_targets)):
            raise ValueError("one plan cannot edit a semantic target twice")
        if len(node_targets) != len(set(node_targets)):
            raise ValueError("one plan cannot edit a DAG node twice")
        object.__setattr__(self, "mutations", mutations)

    @property
    def edited_node_ids(self) -> tuple[str, ...]:
        return tuple(node_id for item in self.mutations for node_id in item.target_node_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "origin_id": self.origin_id,
            "variant_index": self.variant_index,
            "derived_seed": self.derived_seed,
            "edit_count": len(self.mutations),
            "mutations": [item.to_dict() for item in self.mutations],
        }


@dataclass(frozen=True, slots=True)
class InjectedHallucination:
    """Reference/candidate graph pair after all planned edits are applied directly."""

    reference_graph: StateDAG
    candidate_graph: StateDAG
    plan: UnifiedHallucinationPlan

    def __post_init__(self) -> None:
        if type(self.reference_graph) is not StateDAG or type(self.candidate_graph) is not StateDAG:
            raise TypeError("reference_graph and candidate_graph must be StateDAG values")
        if type(self.plan) is not UnifiedHallucinationPlan:
            raise TypeError("plan must be UnifiedHallucinationPlan")
        differences = self.reference_graph.semantic_differences(self.candidate_graph)
        actual_nodes = {target_id for target_kind, target_id in differences if target_kind.value == "node"}
        if actual_nodes != set(self.plan.edited_node_ids) or len(differences) != len(actual_nodes):
            raise ValueError("candidate graph differences must exactly match planned node edits")


@dataclass(frozen=True, slots=True)
class RenderedMention:
    """One exact occurrence of a DAG value in reasoning text or final answer."""

    mention_id: str
    component: str
    node_id: str
    step_index: int | None
    start: int
    end: int
    value: str
    hallucinated: bool

    def __post_init__(self) -> None:
        if self.component not in {"reasoning_chain", "final_answer"}:
            raise ValueError("component must be reasoning_chain or final_answer")
        for value, name in (
            (self.mention_id, "mention_id"),
            (self.node_id, "node_id"),
            (self.value, "value"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("mention offsets must be integers")
        if self.start < 0 or self.end <= self.start or self.end - self.start != len(self.value):
            raise ValueError("mention offsets must exactly cover value")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index <= 0
        ):
            raise ValueError("step_index must be positive or None")
        if type(self.hallucinated) is not bool:
            raise TypeError("hallucinated must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "component": self.component,
            "node_id": self.node_id,
            "step_index": self.step_index,
            "span": [self.start, self.end],
            "value": self.value,
            "hallucinated": self.hallucinated,
        }


@dataclass(frozen=True, slots=True)
class RenderedHallucination:
    """Complete detector-visible reasoning and final answer with exact mentions."""

    reasoning_chain: str
    final_answer: str
    step_texts: tuple[str, ...]
    mentions: tuple[RenderedMention, ...]
    realization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reasoning_chain or not self.final_answer:
            raise ValueError("rendered reasoning and final answer cannot be empty")
        steps = tuple(self.step_texts)
        mentions = tuple(self.mentions)
        if not steps or any(type(item) is not str or not item for item in steps):
            raise ValueError("step_texts must contain non-empty text")
        if any(type(item) is not RenderedMention for item in mentions):
            raise TypeError("mentions must contain RenderedMention values")
        if not isinstance(self.realization, Mapping):
            raise TypeError("realization must be a mapping")
        for mention in mentions:
            source = self.reasoning_chain if mention.component == "reasoning_chain" else self.final_answer
            if source[mention.start : mention.end] != mention.value:
                raise ValueError("mention offsets do not round-trip to rendered text")
        object.__setattr__(self, "step_texts", steps)
        object.__setattr__(self, "mentions", mentions)
        object.__setattr__(self, "realization", FrozenMap(self.realization))

    @property
    def hallucination_spans(self) -> tuple[RenderedMention, ...]:
        return tuple(item for item in self.mentions if item.hallucinated)


__all__ = [
    "InjectedHallucination",
    "PlannedMutation",
    "RenderedHallucination",
    "RenderedMention",
    "UnifiedHallucinationPlan",
]
