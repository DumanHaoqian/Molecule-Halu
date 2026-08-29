"""Typed contracts and fail-closed planning for deterministic propagation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from molhallulens.domain import (
    CandidatePatch,
    CausalRole,
    ClaimValue,
    EditTruth,
    PropagationPolicy,
    StateDAG,
    ValueType,
)
from molhallulens.perturbators.base import PerturbationContext


class PropagationError(RuntimeError):
    """Structured failure raised whenever propagation would require guessing."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        node_id: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("PropagationError code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("PropagationError detail must be non-empty text")
        if node_id is not None and (type(node_id) is not str or not node_id):
            raise ValueError("PropagationError node_id must be non-empty text or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("PropagationError evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.node_id = node_id
        self.evidence = MappingProxyType(dict(evidence or {}))
        location = "" if node_id is None else f" at {node_id!r}"
        super().__init__(f"{code}{location}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "node_id": self.node_id,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class DerivationContext:
    """Closed inputs available to a downstream derivation rule."""

    context: PerturbationContext[EditTruth]
    root_patch: CandidatePatch
    candidate_product_smiles: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.context, PerturbationContext):
            raise TypeError("DerivationContext context must be PerturbationContext")
        if type(self.context.truth) is not EditTruth:
            raise TypeError("DerivationContext requires EditTruth")
        if type(self.root_patch) is not CandidatePatch:
            raise TypeError("DerivationContext root_patch must be CandidatePatch")
        if self.candidate_product_smiles is not None and (
            type(self.candidate_product_smiles) is not str
            or not self.candidate_product_smiles
        ):
            raise ValueError("candidate_product_smiles must be non-empty text or None")


@runtime_checkable
class DerivationRule(Protocol):
    """One typed, deterministic output derivation."""

    rule_id: str
    output_node: str
    input_nodes: tuple[str, ...]
    input_types: tuple[ValueType, ...]
    output_type: ValueType
    causal_role: CausalRole
    schema_ids: frozenset[str]

    def derive(
        self,
        state: StateDAG,
        context: DerivationContext,
    ) -> ClaimValue: ...


DeriveFunction = Callable[[StateDAG, DerivationContext], ClaimValue]


@dataclass(frozen=True, slots=True)
class TypedDerivationRule:
    """Immutable concrete rule with an explicit input/output type signature."""

    rule_id: str
    output_node: str
    input_nodes: tuple[str, ...]
    input_types: tuple[ValueType, ...]
    output_type: ValueType
    derive_fn: DeriveFunction
    causal_role: CausalRole = CausalRole.PROPAGATED_CONDITIONAL
    schema_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for value, name in (
            (self.rule_id, "rule_id"),
            (self.output_node, "output_node"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"TypedDerivationRule {name} must be non-empty text")
        if not isinstance(self.input_nodes, tuple):
            raise TypeError("input_nodes must be a tuple")
        if not isinstance(self.input_types, tuple):
            raise TypeError("input_types must be a tuple")
        if not self.input_nodes:
            raise ValueError("a derivation rule must declare at least one input")
        if len(self.input_nodes) != len(self.input_types):
            raise ValueError("input_nodes and input_types must have equal length")
        if len(set(self.input_nodes)) != len(self.input_nodes):
            raise ValueError("input_nodes must be unique")
        if any(type(node_id) is not str or not node_id for node_id in self.input_nodes):
            raise TypeError("input_nodes must contain non-empty strings")
        if any(type(value_type) is not ValueType for value_type in self.input_types):
            raise TypeError("input_types must contain ValueType values")
        if type(self.output_type) is not ValueType:
            raise TypeError("output_type must be ValueType")
        if not callable(self.derive_fn):
            raise TypeError("derive_fn must be callable")
        if self.causal_role not in {
            CausalRole.PROPAGATED_FALSE,
            CausalRole.PROPAGATED_CONDITIONAL,
        }:
            raise ValueError("derivation causal_role must be a propagated role")
        if isinstance(self.schema_ids, (str, bytes)):
            raise TypeError("schema_ids must be a collection")
        schema_ids = frozenset(self.schema_ids)
        if any(type(schema_id) is not str or not schema_id for schema_id in schema_ids):
            raise TypeError("schema_ids must contain non-empty strings")
        object.__setattr__(self, "schema_ids", schema_ids)

    def derive(
        self,
        state: StateDAG,
        context: DerivationContext,
    ) -> ClaimValue:
        return self.derive_fn(state, context)


def _validate_rule_contract(rule: DerivationRule) -> None:
    if not isinstance(rule, DerivationRule):
        raise TypeError("rules must implement DerivationRule")
    for value, name in (
        (rule.rule_id, "rule_id"),
        (rule.output_node, "output_node"),
    ):
        if type(value) is not str or not value:
            raise ValueError(f"DerivationRule {name} must be non-empty text")
    if type(rule.input_nodes) is not tuple or not rule.input_nodes:
        raise TypeError("DerivationRule input_nodes must be a non-empty tuple")
    if type(rule.input_types) is not tuple:
        raise TypeError("DerivationRule input_types must be a tuple")
    if len(rule.input_nodes) != len(rule.input_types):
        raise ValueError("DerivationRule input signature lengths differ")
    if len(set(rule.input_nodes)) != len(rule.input_nodes):
        raise ValueError("DerivationRule input_nodes must be unique")
    if any(type(item) is not str or not item for item in rule.input_nodes):
        raise TypeError("DerivationRule input_nodes must contain non-empty strings")
    if any(type(item) is not ValueType for item in rule.input_types):
        raise TypeError("DerivationRule input_types must contain ValueType values")
    if type(rule.output_type) is not ValueType:
        raise TypeError("DerivationRule output_type must be ValueType")
    if rule.causal_role not in {
        CausalRole.PROPAGATED_FALSE,
        CausalRole.PROPAGATED_CONDITIONAL,
    }:
        raise ValueError("DerivationRule causal_role must be propagated")
    if type(rule.schema_ids) is not frozenset or any(
        type(schema_id) is not str or not schema_id for schema_id in rule.schema_ids
    ):
        raise TypeError("DerivationRule schema_ids must be a frozenset of strings")
    if not callable(rule.derive):
        raise TypeError("DerivationRule derive must be callable")


@dataclass(frozen=True, slots=True)
class DerivationRuleRegistry:
    """Immutable output-node index for deterministic rules."""

    rules: tuple[DerivationRule, ...]

    def __post_init__(self) -> None:
        if isinstance(self.rules, (str, bytes)) or not isinstance(self.rules, Iterable):
            raise TypeError("rules must be a non-string iterable")
        rules = tuple(self.rules)
        for rule in rules:
            _validate_rule_contract(rule)
        rule_ids = tuple(rule.rule_id for rule in rules)
        output_scopes = tuple(
            (schema_id, rule.output_node)
            for rule in rules
            for schema_id in (rule.schema_ids or frozenset({"*"}))
        )
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("DerivationRule rule IDs must be unique")
        if len(output_scopes) != len(set(output_scopes)):
            raise ValueError("DerivationRule outputs must be unique per schema")
        wildcard_outputs = {rule.output_node for rule in rules if not rule.schema_ids}
        scoped_outputs = {rule.output_node for rule in rules if rule.schema_ids}
        if wildcard_outputs & scoped_outputs:
            raise ValueError("wildcard and schema-scoped rules cannot share an output")
        object.__setattr__(
            self,
            "rules",
            tuple(sorted(rules, key=lambda rule: (rule.output_node, rule.rule_id))),
        )

    def rule_for(
        self,
        output_node: str,
        *,
        schema_id: str,
    ) -> DerivationRule:
        if type(output_node) is not str or not output_node:
            raise TypeError("output_node must be non-empty text")
        if type(schema_id) is not str or not schema_id:
            raise TypeError("schema_id must be non-empty text")
        matches = tuple(
            rule
            for rule in self.rules
            if rule.output_node == output_node
            and (not rule.schema_ids or schema_id in rule.schema_ids)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:  # pragma: no cover - constructor prevents ambiguity
            raise RuntimeError("ambiguous derivation rule registry")
        raise KeyError(output_node)

    def has_rule(self, output_node: str, *, schema_id: str) -> bool:
        try:
            self.rule_for(output_node, schema_id=schema_id)
        except KeyError:
            return False
        return True


@dataclass(frozen=True, slots=True)
class PropagationPlan:
    """Auditable topo-ordered recomputation boundary for one policy."""

    policy: PropagationPolicy
    root_node_id: str
    full_closure: tuple[str, ...]
    selected_nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("PropagationPlan policy must be PropagationPolicy")
        if type(self.root_node_id) is not str or not self.root_node_id:
            raise ValueError("PropagationPlan root_node_id must be non-empty text")
        for values, name in (
            (self.full_closure, "full_closure"),
            (self.selected_nodes, "selected_nodes"),
        ):
            if type(values) is not tuple:
                raise TypeError(f"PropagationPlan {name} must be a tuple")
            if any(type(value) is not str or not value for value in values):
                raise TypeError(f"PropagationPlan {name} must contain node IDs")
            if len(values) != len(set(values)):
                raise ValueError(f"PropagationPlan {name} must be unique")
        if not self.full_closure or self.full_closure[0] != self.root_node_id:
            raise ValueError("full_closure must begin with its root")
        if not self.selected_nodes or self.selected_nodes[0] != self.root_node_id:
            raise ValueError("selected_nodes must begin with its root")
        if not set(self.selected_nodes) <= set(self.full_closure):
            raise ValueError("selected_nodes must be a subset of full_closure")


__all__ = [
    "DerivationContext",
    "DerivationRule",
    "DerivationRuleRegistry",
    "PropagationError",
    "PropagationPlan",
    "TypedDerivationRule",
]
