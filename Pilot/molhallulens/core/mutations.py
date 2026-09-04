"""Domain objects for the single configurable multi-point hallucination path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .enums import CausalRole, DependencyType, MutationCategory, ValueType
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
class PropagationEvent:
    """One deterministic claim change caused by a sampled root mutation."""

    event_id: str
    root_mutation_id: str
    root_semantic_target_id: str
    source_node_ids: tuple[str, ...]
    target_node_id: str
    rule_id: str
    before: Any
    after: Any

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.root_mutation_id, "root_mutation_id"),
            (self.root_semantic_target_id, "root_semantic_target_id"),
            (self.target_node_id, "target_node_id"),
            (self.rule_id, "rule_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        sources = tuple(self.source_node_ids)
        if not sources or any(type(item) is not str or not item for item in sources):
            raise ValueError("source_node_ids must contain non-empty node IDs")
        if len(sources) != len(set(sources)):
            raise ValueError("source_node_ids cannot contain duplicates")
        before = deep_freeze(self.before)
        after = deep_freeze(self.after)
        if type(before) is not type(after) or before == after:
            raise ValueError("a propagation event must change a value without changing type")
        object.__setattr__(self, "source_node_ids", sources)
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "root_mutation_id": self.root_mutation_id,
            "root_semantic_target_id": self.root_semantic_target_id,
            "source_node_ids": list(self.source_node_ids),
            "target_node_id": self.target_node_id,
            "rule_id": self.rule_id,
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True, slots=True)
class EdgeAuditResult:
    """Materialized satisfaction status for one candidate-DAG dependency."""

    edge_id: str
    relation: DependencyType
    status: bool | None

    def __post_init__(self) -> None:
        if type(self.edge_id) is not str or not self.edge_id:
            raise ValueError("edge_id must be non-empty text")
        if type(self.relation) is not DependencyType:
            raise TypeError("relation must be DependencyType")
        if self.status is not None and type(self.status) is not bool:
            raise TypeError("status must be bool or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "relation": self.relation.value,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class InjectedHallucination:
    """Reference/candidate pair with sampled roots and deterministic consequences."""

    reference_graph: StateDAG
    candidate_graph: StateDAG
    plan: UnifiedHallucinationPlan
    propagation_events: tuple[PropagationEvent, ...] = ()
    edge_audit: tuple[EdgeAuditResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.reference_graph) is not StateDAG or type(self.candidate_graph) is not StateDAG:
            raise TypeError("reference_graph and candidate_graph must be StateDAG values")
        if type(self.plan) is not UnifiedHallucinationPlan:
            raise TypeError("plan must be UnifiedHallucinationPlan")
        events = tuple(self.propagation_events)
        audit = tuple(self.edge_audit)
        if any(type(item) is not PropagationEvent for item in events):
            raise TypeError("propagation_events must contain PropagationEvent values")
        if any(type(item) is not EdgeAuditResult for item in audit):
            raise TypeError("edge_audit must contain EdgeAuditResult values")
        propagated_nodes = tuple(item.target_node_id for item in events)
        if len(propagated_nodes) != len(set(propagated_nodes)):
            raise ValueError("a propagated node may be changed only once")
        if set(propagated_nodes) & set(self.plan.edited_node_ids):
            raise ValueError("root and propagated node sets must be disjoint")
        differences = self.reference_graph.semantic_differences(self.candidate_graph)
        actual_nodes = {target_id for target_kind, target_id in differences if target_kind.value == "node"}
        expected_nodes = set(self.plan.edited_node_ids) | set(propagated_nodes)
        if actual_nodes != expected_nodes or len(differences) != len(actual_nodes):
            raise ValueError(
                "candidate graph differences must exactly match root and propagated edits"
            )
        if audit:
            audit_ids = tuple(item.edge_id for item in audit)
            schema_ids = tuple(sorted(self.candidate_graph.schema.edges_by_id))
            if tuple(sorted(audit_ids)) != schema_ids or len(audit_ids) != len(set(audit_ids)):
                raise ValueError("edge_audit must contain every schema edge exactly once")
        object.__setattr__(self, "propagation_events", events)
        object.__setattr__(self, "edge_audit", audit)

    @property
    def changed_node_ids(self) -> tuple[str, ...]:
        return self.plan.edited_node_ids + tuple(
            item.target_node_id for item in self.propagation_events
        )

    @property
    def causal_roles_by_node(self) -> FrozenMap[str, CausalRole]:
        roles = {
            node_id: CausalRole.ROOT_HALLUCINATION
            for node_id in self.plan.edited_node_ids
        }
        roles.update(
            {
                item.target_node_id: CausalRole.PROPAGATED_ERROR
                for item in self.propagation_events
            }
        )
        return FrozenMap(roles)

    @property
    def violated_edge_ids(self) -> tuple[str, ...]:
        return tuple(item.edge_id for item in self.edge_audit if item.status is False)


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
    causal_role: CausalRole | None = None
    context_start: int | None = None
    context_end: int | None = None
    paired_start: int | None = None
    paired_end: int | None = None
    diff_opcodes: tuple[tuple[str, int, int, int, int], ...] = ()

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
        if self.hallucinated:
            if type(self.causal_role) is not CausalRole:
                raise TypeError("hallucinated mentions require a CausalRole")
        elif self.causal_role is not None:
            raise ValueError("non-hallucinated mentions cannot have a causal_role")
        if (self.context_start is None) != (self.context_end is None):
            raise ValueError("context offsets must both be present or both be absent")
        if self.context_start is None:
            object.__setattr__(self, "context_start", self.start)
            object.__setattr__(self, "context_end", self.end)
        elif not (
            0 <= self.context_start <= self.start
            and self.end <= self.context_end
        ):
            raise ValueError("context span must contain the mention span")
        if (self.paired_start is None) != (self.paired_end is None):
            raise ValueError("paired offsets must both be present or both be absent")
        if self.paired_start is not None and not (
            0 <= self.paired_start < self.paired_end
        ):
            raise ValueError("paired offsets must describe a non-empty interval")
        opcodes = tuple(self.diff_opcodes)
        for opcode in opcodes:
            if (
                type(opcode) is not tuple
                or len(opcode) != 5
                or opcode[0] not in {"replace", "delete", "insert"}
                or any(type(value) is not int or value < 0 for value in opcode[1:])
            ):
                raise ValueError("diff_opcodes contains an invalid SequenceMatcher opcode")
        if bool(opcodes) != (self.paired_start is not None):
            raise ValueError("molecular diff opcodes and paired offsets must appear together")
        object.__setattr__(self, "diff_opcodes", opcodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "component": self.component,
            "node_id": self.node_id,
            "step_index": self.step_index,
            "span": [self.start, self.end],
            "context_span": [self.context_start, self.context_end],
            "value": self.value,
            "diff_opcodes": [
                {
                    "tag": item[0],
                    "reference_span": [item[1], item[2]],
                    "candidate_span": [item[3], item[4]],
                }
                for item in self.diff_opcodes
            ],
            "hallucinated": self.hallucinated,
            "causal_role": (
                self.causal_role.value if self.causal_role is not None else None
            ),
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
    "EdgeAuditResult",
    "PlannedMutation",
    "PropagationEvent",
    "RenderedHallucination",
    "RenderedMention",
    "UnifiedHallucinationPlan",
]
