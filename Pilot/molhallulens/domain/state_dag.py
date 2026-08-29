"""Typed, immutable state-schema and state-instance objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from heapq import heapify, heappop, heappush
from math import isfinite
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from .enums import (
    CausalRole,
    ComparatorKind,
    DependencyType,
    EditErrorSubtype,
    HallucinationType,
    MutationTargetKind,
    NodeRole,
    ValueProvenance,
    ValueType,
    Visibility,
    is_domain_enum_member,
)


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class FrozenMap(Mapping[KeyT, ValueT], Generic[KeyT, ValueT]):
    """A defensive-copy, recursively immutable mapping for domain objects."""

    __slots__ = ("_data",)

    def __init__(
        self,
        values: Mapping[KeyT, ValueT] | None = None,
        *,
        _active: set[int] | None = None,
    ) -> None:
        if values is None:
            copied: dict[KeyT, ValueT] = {}
        else:
            copied = _freeze_mapping(values, active=set() if _active is None else _active)
        object.__setattr__(self, "_data", MappingProxyType(copied))

    def __getitem__(self, key: KeyT) -> ValueT:
        return self._data[key]

    def __iter__(self) -> Iterator[KeyT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("FrozenMap is immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("FrozenMap is immutable")

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._data)!r})"

    def to_dict(self) -> dict[KeyT, ValueT]:
        return dict(self._data)


def _freeze_mapping(values: Mapping[Any, Any], *, active: set[int]) -> dict[Any, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("frozen mapping values must implement Mapping")
    identity = id(values)
    if identity in active:
        raise ValueError("cyclic containers cannot be frozen")
    active.add(identity)
    try:
        copied: dict[Any, Any] = {}
        for raw_key, raw_value in values.items():
            key = _deep_freeze(raw_key, active=active)
            try:
                hash(key)
            except TypeError as error:
                raise TypeError("frozen mapping keys must be hashable") from error
            if key in copied:
                raise ValueError("mapping keys collide after freezing")
            copied[key] = _deep_freeze(raw_value, active=active)
        return copied
    finally:
        active.remove(identity)


def _deep_freeze(value: Any, *, active: set[int]) -> Any:
    if value is None or type(value) in {str, bytes, bool, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("domain float values must be finite")
        return value
    if isinstance(value, Enum):
        if not is_domain_enum_member(value) or type(value.value) not in {str, int}:
            raise TypeError("only scalar MolHalluLens domain enums can be frozen")
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        return FrozenMap(value, _active=active)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic containers cannot be frozen")
        active.add(identity)
        try:
            return tuple(_deep_freeze(item, active=active) for item in value)
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic containers cannot be frozen")
        active.add(identity)
        try:
            return frozenset(_deep_freeze(item, active=active) for item in value)
        finally:
            active.remove(identity)
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_parameters = getattr(type(value), "__dataclass_params__", None)
        if dataclass_parameters is None or not dataclass_parameters.frozen:
            raise TypeError("only frozen dataclass instances can be embedded in domain values")
        if hasattr(value, "__dict__"):
            raise TypeError("embedded frozen dataclasses must use slots and have no __dict__")
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic dataclass values cannot be frozen")
        active.add(identity)
        try:
            frozen_fields = {
                item.name: _deep_freeze(getattr(value, item.name), active=active)
                for item in fields(value)
            }
            rebuilt = replace(
                value,
                **{
                    item.name: frozen_fields[item.name]
                    for item in fields(value)
                    if item.init
                },
            )
            for item in fields(value):
                if not item.init:
                    object.__setattr__(rebuilt, item.name, frozen_fields[item.name])
            return rebuilt
        finally:
            active.remove(identity)
    raise TypeError(
        f"unsupported mutable or opaque domain value type: {type(value).__qualname__}"
    )


def deep_freeze(value: Any) -> Any:
    """Copy a value into the closed, recursively immutable domain value algebra."""

    return _deep_freeze(value, active=set())


def freeze_string_mapping(values: Mapping[str, Any], *, name: str) -> FrozenMap[str, Any]:
    """Freeze a JSON-like mapping and require exact, non-empty string keys recursively."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = FrozenMap(values)

    def validate_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if type(key) is not str or not key:
                    raise TypeError(f"{name} must use non-empty string keys recursively")
                validate_keys(item)
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                validate_keys(item)
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                validate_keys(getattr(value, item.name))

    validate_keys(frozen)
    return frozen


@dataclass(frozen=True, slots=True)
class ClaimValue:
    raw_value: Any
    normalized_value: Any
    value_type: ValueType
    provenance: ValueProvenance
    locally_valid: bool | None = None
    oracle_match: bool | None = None
    confidence: float | None = None
    mention_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.value_type) is not ValueType:
            raise TypeError("ClaimValue value_type must be a ValueType")
        if type(self.provenance) is not ValueProvenance:
            raise TypeError("ClaimValue provenance must be a ValueProvenance")
        if not isinstance(self.mention_ids, (list, tuple)):
            raise TypeError("ClaimValue mention_ids must be a list or tuple")
        object.__setattr__(self, "raw_value", deep_freeze(self.raw_value))
        object.__setattr__(self, "normalized_value", deep_freeze(self.normalized_value))
        object.__setattr__(self, "mention_ids", tuple(self.mention_ids))
        for boolean, name in (
            (self.locally_valid, "locally_valid"),
            (self.oracle_match, "oracle_match"),
        ):
            if boolean is not None and type(boolean) is not bool:
                raise TypeError(f"ClaimValue {name} must be bool or None")
        if self.confidence is not None:
            if type(self.confidence) not in {int, float}:
                raise TypeError("ClaimValue confidence must be numeric or None")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("ClaimValue confidence must be in [0, 1]")
        if len(self.mention_ids) != len(set(self.mention_ids)):
            raise ValueError("ClaimValue mention_ids must be unique")
        if any(type(mention_id) is not str or not mention_id for mention_id in self.mention_ids):
            raise ValueError("ClaimValue mention_ids must be non-empty strings")
        self._validate_normalized_payload_type()

    def _validate_normalized_payload_type(self) -> None:
        value = self.normalized_value
        integer_types = {ValueType.INTEGER, ValueType.ATOM_INDEX, ValueType.COUNT}
        string_types = {
            ValueType.STRING,
            ValueType.SMILES,
            ValueType.INDEXED_SMILES,
            ValueType.ELEMENT,
            ValueType.FRAGMENT,
            ValueType.MOLECULE,
        }
        if self.value_type in integer_types:
            if type(value) is not int:
                raise TypeError(f"{self.value_type.value} claims require an integer normalized value")
            if self.value_type in {ValueType.ATOM_INDEX, ValueType.COUNT} and value < 0:
                raise ValueError(f"{self.value_type.value} claims must be non-negative")
        elif self.value_type is ValueType.FLOAT:
            if type(value) not in {int, float}:
                raise TypeError("float claims require a numeric normalized value")
            if not isfinite(value):
                raise ValueError("float claims require a finite normalized value")
        elif self.value_type is ValueType.BOOLEAN:
            if type(value) is not bool:
                raise TypeError("boolean claims require a bool normalized value")
        elif self.value_type in string_types:
            if type(value) is not str:
                raise TypeError(f"{self.value_type.value} claims require a string normalized value")
        elif self.value_type is ValueType.ATOM_SET:
            if not isinstance(value, (tuple, frozenset)) or any(
                type(atom) is not int for atom in value
            ):
                raise TypeError("atom_set claims require an integer collection normalized value")
            if any(atom < 0 for atom in value):
                raise ValueError("atom_set claims must contain non-negative atom indices")
            object.__setattr__(self, "normalized_value", frozenset(value))
        elif self.value_type is ValueType.BOND_EDIT and not isinstance(
            value, (str, tuple, frozenset, FrozenMap)
        ):
            raise TypeError("bond_edit claims require a structured immutable normalized value")

    def semantically_equals(self, other: ClaimValue) -> bool:
        return (
            type(other) is ClaimValue
            and self.value_type is other.value_type
            and self.normalized_value == other.normalized_value
        )


@dataclass(frozen=True, slots=True)
class StateNodeSpec:
    node_id: str
    value_type: ValueType
    step_index: int | None
    role: NodeRole
    visibility: Visibility
    mutable: bool
    comparator: ComparatorKind
    renderer_slot: str | None = None

    def __post_init__(self) -> None:
        if type(self.node_id) is not str:
            raise TypeError("StateNodeSpec node_id must be a string")
        if type(self.value_type) is not ValueType:
            raise TypeError("StateNodeSpec value_type must be a ValueType")
        if type(self.role) is not NodeRole:
            raise TypeError("StateNodeSpec role must be a NodeRole")
        if type(self.visibility) is not Visibility:
            raise TypeError("StateNodeSpec visibility must be a Visibility")
        if type(self.comparator) is not ComparatorKind:
            raise TypeError("StateNodeSpec comparator must be a ComparatorKind")
        if type(self.mutable) is not bool:
            raise TypeError("StateNodeSpec mutable must be bool")
        if self.step_index is not None and (
            type(self.step_index) is not int
        ):
            raise TypeError("StateNodeSpec step_index must be an integer or None")
        if self.renderer_slot is not None and type(self.renderer_slot) is not str:
            raise TypeError("StateNodeSpec renderer_slot must be a string or None")
        if not self.node_id:
            raise ValueError("StateNodeSpec node_id cannot be empty")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("StateNodeSpec step_index cannot be negative")
        if self.renderer_slot == "":
            raise ValueError("renderer_slot must be non-empty or None")
        if self.visibility is Visibility.BUILD_ONLY and self.renderer_slot is not None:
            raise ValueError("BUILD_ONLY nodes cannot declare a renderer slot")
        if self.role is NodeRole.INTERNAL_TRUTH and self.visibility is not Visibility.BUILD_ONLY:
            raise ValueError("INTERNAL_TRUTH nodes must be BUILD_ONLY")
        if (
            self.role in {NodeRole.EVIDENCE, NodeRole.INTERNAL_TRUTH}
            or self.visibility is Visibility.BUILD_ONLY
        ) and self.mutable:
            raise ValueError("EVIDENCE, INTERNAL_TRUTH, and BUILD_ONLY nodes must be immutable")


@dataclass(frozen=True, slots=True)
class StateEdge:
    edge_id: str
    source: str
    target: str
    relation: DependencyType
    mutable: bool = False
    renderer_slot: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.edge_id, "edge_id"),
            (self.source, "source"),
            (self.target, "target"),
        ):
            if type(value) is not str:
                raise TypeError(f"StateEdge {name} must be a string")
        if type(self.relation) is not DependencyType:
            raise TypeError("StateEdge relation must be a DependencyType")
        if type(self.mutable) is not bool:
            raise TypeError("StateEdge mutable must be bool")
        if self.renderer_slot is not None and type(self.renderer_slot) is not str:
            raise TypeError("StateEdge renderer_slot must be a string or None")
        if not self.edge_id or not self.source or not self.target:
            raise ValueError("StateEdge IDs and endpoints cannot be empty")
        if self.source == self.target:
            raise ValueError("StateEdge cannot be a self-loop")
        if self.renderer_slot == "":
            raise ValueError("renderer_slot must be non-empty or None")


@dataclass(frozen=True, slots=True)
class StateSchema:
    schema_id: str
    version: str
    nodes: tuple[StateNodeSpec, ...]
    edges: tuple[StateEdge, ...]

    def __post_init__(self) -> None:
        if type(self.schema_id) is not str or type(self.version) is not str:
            raise TypeError("StateSchema schema_id and version must be strings")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not self.schema_id or not self.version:
            raise ValueError("StateSchema ID and version cannot be empty")
        if not self.nodes:
            raise ValueError("StateSchema must contain at least one node")
        if any(type(node) is not StateNodeSpec for node in self.nodes):
            raise TypeError("StateSchema nodes must contain StateNodeSpec values")
        if any(type(edge) is not StateEdge for edge in self.edges):
            raise TypeError("StateSchema edges must contain StateEdge values")
        node_ids = tuple(node.node_id for node in self.nodes)
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("StateSchema node IDs must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("StateSchema edge IDs must be unique")
        if set(node_ids) & set(edge_ids):
            raise ValueError("StateSchema node and edge IDs must be globally disjoint")
        node_id_set = set(node_ids)
        dangling = tuple(
            edge.edge_id
            for edge in self.edges
            if edge.source not in node_id_set or edge.target not in node_id_set
        )
        if dangling:
            raise ValueError(f"StateSchema contains dangling edges: {dangling!r}")
        nodes_by_id = {node.node_id: node for node in self.nodes}
        private_edges = tuple(
            edge
            for edge in self.edges
            if Visibility.BUILD_ONLY
            in {
                nodes_by_id[edge.source].visibility,
                nodes_by_id[edge.target].visibility,
            }
        )
        if any(edge.renderer_slot is not None for edge in private_edges):
            raise ValueError("edges touching BUILD_ONLY nodes cannot declare renderer slots")
        if any(edge.mutable for edge in private_edges):
            raise ValueError("edges touching BUILD_ONLY nodes must be immutable")
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
        ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
        visited = 0
        while ready:
            node_id = ready.pop()
            visited += 1
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(node_ids):
            raise ValueError("StateSchema must be acyclic")

    @property
    def nodes_by_id(self) -> FrozenMap[str, StateNodeSpec]:
        return FrozenMap({node.node_id: node for node in self.nodes})

    @property
    def edges_by_id(self) -> FrozenMap[str, StateEdge]:
        return FrozenMap({edge.edge_id: edge for edge in self.edges})

    def _validated_node_ids(
        self,
        node_ids: Iterable[str],
        *,
        name: str,
    ) -> frozenset[str]:
        if isinstance(node_ids, (str, bytes)) or not isinstance(node_ids, Iterable):
            raise TypeError(f"{name} must be a non-string iterable of node IDs")
        collected: set[str] = set()
        for node_id in node_ids:
            if type(node_id) is not str:
                raise TypeError(f"{name} must contain only string node IDs")
            collected.add(node_id)
        unknown = tuple(sorted(collected - set(self.nodes_by_id)))
        if unknown:
            raise KeyError(f"{name} contains unknown node IDs: {unknown!r}")
        return frozenset(collected)

    def _validated_node_id(self, node_id: str, *, name: str) -> str:
        if type(node_id) is not str:
            raise TypeError(f"{name} must be a string node ID")
        if node_id not in self.nodes_by_id:
            raise KeyError(node_id)
        return node_id

    def _adjacency(self) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        outgoing_sets = {node.node_id: set() for node in self.nodes}
        incoming_sets = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            outgoing_sets[edge.source].add(edge.target)
            incoming_sets[edge.target].add(edge.source)
        outgoing = {
            node_id: tuple(sorted(targets))
            for node_id, targets in outgoing_sets.items()
        }
        incoming = {
            node_id: tuple(sorted(sources))
            for node_id, sources in incoming_sets.items()
        }
        return outgoing, incoming

    def topological_order(self) -> tuple[str, ...]:
        """Return the unique lexical-tie-broken topological node order."""

        indegree = {node.node_id: 0 for node in self.nodes}
        outgoing = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            indegree[edge.target] += 1
            outgoing[edge.source].append(edge.target)
        for targets in outgoing.values():
            targets.sort()
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        heapify(ready)
        ordered: list[str] = []
        while ready:
            node_id = heappop(ready)
            ordered.append(node_id)
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heappush(ready, target)
        if len(ordered) != len(self.nodes):
            raise ValueError("StateSchema must be acyclic")
        return tuple(ordered)

    def descendants(self, root_node_id: str) -> tuple[str, ...]:
        """Return strict reachable downstream nodes in deterministic topo order."""

        root = self._validated_node_id(root_node_id, name="root_node_id")
        outgoing, _ = self._adjacency()
        reachable: set[str] = set()
        frontier = list(outgoing[root])
        while frontier:
            node_id = frontier.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            frontier.extend(outgoing[node_id])
        return tuple(
            node_id for node_id in self.topological_order() if node_id in reachable
        )

    def dependency_closure(self, seed_node_ids: Iterable[str]) -> tuple[str, ...]:
        """Return seeds and all reachable descendants in deterministic topo order."""

        seeds = self._validated_node_ids(seed_node_ids, name="seed_node_ids")
        outgoing, _ = self._adjacency()
        closure = set(seeds)
        frontier = list(seeds)
        while frontier:
            node_id = frontier.pop()
            for target in outgoing[node_id]:
                if target not in closure:
                    closure.add(target)
                    frontier.append(target)
        return tuple(
            node_id for node_id in self.topological_order() if node_id in closure
        )

    def is_connected_downstream_subgraph(
        self,
        root_node_ids: Iterable[str],
        selected_node_ids: Iterable[str],
    ) -> bool:
        """Whether selected nodes form a directed root-reachable downstream subgraph."""

        roots = self._validated_node_ids(root_node_ids, name="root_node_ids")
        if not roots:
            raise ValueError("root_node_ids cannot be empty")
        selected = self._validated_node_ids(
            selected_node_ids,
            name="selected_node_ids",
        )
        full_downstream = frozenset(self.dependency_closure(roots))
        if not roots <= selected or not selected <= full_downstream:
            return False
        outgoing, _ = self._adjacency()
        reached = set(roots)
        frontier = list(roots)
        while frontier:
            node_id = frontier.pop()
            for target in outgoing[node_id]:
                if target in selected and target not in reached:
                    reached.add(target)
                    frontier.append(target)
        if reached != set(selected):
            return False
        return True

    def stale_downstream_edges(
        self,
        changed_node_ids: Iterable[str],
    ) -> tuple[StateEdge, ...]:
        """Return the deterministic stale-downstream frontier of a changed set.

        This is a structural propagation query: an edge is reported when its source
        changed but its target did not. Chemistry and comparator-based truth checks
        remain validator responsibilities.
        """

        changed = self._validated_node_ids(
            changed_node_ids,
            name="changed_node_ids",
        )
        topo_index = {
            node_id: index for index, node_id in enumerate(self.topological_order())
        }
        return tuple(
            sorted(
                (
                    edge
                    for edge in self.edges
                    if edge.source in changed and edge.target not in changed
                ),
                key=lambda edge: (
                    topo_index[edge.source],
                    topo_index[edge.target],
                    edge.edge_id,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class StateDAG:
    schema: StateSchema
    values: Mapping[str, ClaimValue]
    edge_values: Mapping[str, ClaimValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema) is not StateSchema:
            raise TypeError("StateDAG schema must be a StateSchema")
        if not isinstance(self.values, Mapping) or not isinstance(self.edge_values, Mapping):
            raise TypeError("StateDAG values and edge_values must be mappings")
        for value_map, name in ((self.values, "values"), (self.edge_values, "edge_values")):
            if any(type(key) is not str for key in value_map):
                raise TypeError(f"StateDAG {name} keys must be strings")
            if any(type(value) is not ClaimValue for value in value_map.values()):
                raise TypeError(f"StateDAG {name} must contain ClaimValue values")
        values = FrozenMap(self.values)
        edge_values = FrozenMap(self.edge_values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "edge_values", edge_values)
        expected_node_ids = set(self.schema.nodes_by_id)
        actual_node_ids = set(values)
        if actual_node_ids != expected_node_ids:
            missing = sorted(expected_node_ids - actual_node_ids)
            unknown = sorted(actual_node_ids - expected_node_ids)
            raise ValueError(f"StateDAG value keys differ from schema; missing={missing}, unknown={unknown}")
        unknown_edges = sorted(set(edge_values) - set(self.schema.edges_by_id))
        if unknown_edges:
            raise ValueError(f"StateDAG has values for unknown edges: {unknown_edges}")
        for node in self.schema.nodes:
            if values[node.node_id].value_type is not node.value_type:
                raise ValueError(
                    f"StateDAG value type mismatch for {node.node_id!r}: "
                    f"{values[node.node_id].value_type} != {node.value_type}"
                )

    def value_for(self, node_id: str) -> ClaimValue:
        return self.values[node_id]

    def edge_satisfaction(
        self,
        edge_id: str,
        evaluator: Callable[[StateEdge, StateDAG], bool | None] | None = None,
    ) -> bool | None:
        """Return a known edge-satisfaction status, or ``None`` when unverified.

        A relation can be evaluated by an explicit edge claim's ``locally_valid``
        flag or an injected chemistry/derivation evaluator. Independent known
        statuses must agree. No relation is inferred from raw endpoint equality.
        """

        if type(edge_id) is not str:
            raise TypeError("edge_id must be a string")
        edge = next(
            (item for item in self.schema.edges if item.edge_id == edge_id),
            None,
        )
        if edge is None:
            raise KeyError(edge_id)
        if evaluator is not None and not callable(evaluator):
            raise TypeError("evaluator must be callable or None")
        return self._edge_satisfaction(edge, evaluator)

    def _edge_satisfaction(
        self,
        edge: StateEdge,
        evaluator: Callable[[StateEdge, StateDAG], bool | None] | None,
    ) -> bool | None:
        statuses: list[bool] = []
        if edge.edge_id in self.edge_values:
            explicit_status = self.edge_values[edge.edge_id].locally_valid
            if explicit_status is not None:
                statuses.append(explicit_status)
        if evaluator is not None:
            evaluated_status = evaluator(edge, self)
            if evaluated_status is not None and type(evaluated_status) is not bool:
                raise TypeError("edge evaluator must return bool or None")
            if evaluated_status is not None:
                statuses.append(evaluated_status)
        if len(set(statuses)) > 1:
            raise ValueError(
                f"conflicting satisfaction statuses for edge {edge.edge_id!r}"
            )
        return statuses[0] if statuses else None

    def violated_edges(
        self,
        evaluator: Callable[[StateEdge, StateDAG], bool | None] | None = None,
    ) -> tuple[StateEdge, ...]:
        """Return known-unsatisfied edges in deterministic edge-ID order.

        Edges with no built-in, explicit, or injected status remain unknown and are
        not silently treated as either satisfied or violated.
        """

        if evaluator is not None and not callable(evaluator):
            raise TypeError("evaluator must be callable or None")
        return tuple(
            edge
            for edge in sorted(self.schema.edges, key=lambda item: item.edge_id)
            if self._edge_satisfaction(edge, evaluator) is False
        )

    def semantic_differences(
        self,
        other: StateDAG,
    ) -> frozenset[tuple[MutationTargetKind, str]]:
        """Return node/edge targets whose normalized semantic values differ."""

        if type(other) is not StateDAG:
            raise TypeError("StateDAG semantic comparison requires another StateDAG")
        if self.schema != other.schema:
            raise ValueError("StateDAG semantic comparison requires identical schemas")
        differences: set[tuple[MutationTargetKind, str]] = set()
        for node_id in self.values:
            if not self.values[node_id].semantically_equals(other.values[node_id]):
                differences.add((MutationTargetKind.NODE, node_id))
        for edge_id in set(self.edge_values) | set(other.edge_values):
            if edge_id not in self.edge_values or edge_id not in other.edge_values:
                differences.add((MutationTargetKind.EDGE, edge_id))
            elif not self.edge_values[edge_id].semantically_equals(other.edge_values[edge_id]):
                differences.add((MutationTargetKind.EDGE, edge_id))
        return frozenset(differences)

    def semantically_equals(self, other: StateDAG) -> bool:
        try:
            return not self.semantic_differences(other)
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class MutationEvent:
    event_id: str
    target_kind: MutationTargetKind
    node_or_edge_id: str
    before: ClaimValue
    after: ClaimValue
    causal_role: CausalRole
    hallucination_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]
    operator_id: str
    root_event_id: str

    def __post_init__(self) -> None:
        if type(self.target_kind) is not MutationTargetKind:
            raise TypeError("MutationEvent target_kind must be a MutationTargetKind")
        if type(self.causal_role) is not CausalRole:
            raise TypeError("MutationEvent causal_role must be a CausalRole")
        if type(self.before) is not ClaimValue or type(self.after) is not ClaimValue:
            raise TypeError("MutationEvent before and after must be ClaimValue values")
        object.__setattr__(self, "hallucination_types", frozenset(self.hallucination_types))
        object.__setattr__(self, "edit_subtypes", frozenset(self.edit_subtypes))
        if any(type(value) is not HallucinationType for value in self.hallucination_types):
            raise TypeError("MutationEvent hallucination_types must contain HallucinationType values")
        if any(type(value) is not EditErrorSubtype for value in self.edit_subtypes):
            raise TypeError("MutationEvent edit_subtypes must contain EditErrorSubtype values")
        for value, name in (
            (self.event_id, "event_id"),
            (self.node_or_edge_id, "node_or_edge_id"),
            (self.operator_id, "operator_id"),
            (self.root_event_id, "root_event_id"),
        ):
            if type(value) is not str:
                raise TypeError(f"MutationEvent {name} must be a string")
            if not value:
                raise ValueError(f"MutationEvent {name} cannot be empty")
        if self.before.value_type is not self.after.value_type:
            raise ValueError("MutationEvent cannot change a claim's ValueType")
        if self.before.semantically_equals(self.after):
            raise ValueError("MutationEvent before and after normalized values must differ")
        if not self.hallucination_types:
            raise ValueError("MutationEvent must carry at least one semantic type")
        if HallucinationType.UNVERIFIABLE in self.hallucination_types:
            raise ValueError("adjudicated MutationEvent values cannot be UNVERIFIABLE")
        if not self.edit_subtypes:
            raise ValueError("MutationEvent must carry at least one editing subtype")
        if self.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}:
            if self.root_event_id != self.event_id:
                raise ValueError("ROOT and TERMINAL events must identify themselves as root")
        elif self.root_event_id == self.event_id:
            raise ValueError("propagated events must refer to a distinct root event")
        if self.causal_role is CausalRole.TERMINAL:
            if (
                self.target_kind is not MutationTargetKind.NODE
                or self.node_or_edge_id != "final_answer"
            ):
                raise ValueError("TERMINAL events must target the final_answer node")
            if EditErrorSubtype.FINAL_ANSWER_IDENTITY not in self.edit_subtypes:
                raise ValueError("TERMINAL events must carry FINAL_ANSWER_IDENTITY")
        if (
            self.causal_role is CausalRole.ROOT
            and self.target_kind is MutationTargetKind.NODE
            and self.node_or_edge_id == "final_answer"
        ):
            raise ValueError("an independent final_answer mutation must use TERMINAL")


@dataclass(frozen=True, slots=True)
class GraphDelta:
    events: tuple[MutationEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if any(type(event) is not MutationEvent for event in self.events):
            raise TypeError("GraphDelta events must contain MutationEvent values")
        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("GraphDelta event IDs must be unique")
        event_targets = tuple(
            (event.target_kind, event.node_or_edge_id) for event in self.events
        )
        if len(event_targets) != len(set(event_targets)):
            raise ValueError("GraphDelta mutation targets must be unique")
        if len({event.operator_id for event in self.events}) > 1:
            raise ValueError("GraphDelta events must share one operator_id")
        root_ids = {
            event.event_id
            for event in self.events
            if event.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}
        }
        missing_roots = sorted({event.root_event_id for event in self.events} - root_ids)
        if missing_roots:
            raise ValueError(f"GraphDelta references non-root or unknown root events: {missing_roots}")
        if self.events and len(root_ids) != 1:
            raise ValueError("a non-empty GraphDelta must contain exactly one independent root event")
        if any(event.causal_role is CausalRole.TERMINAL for event in self.events) and len(
            self.events
        ) != 1:
            raise ValueError("a TERMINAL GraphDelta cannot contain propagated events")

    @property
    def root_events(self) -> tuple[MutationEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}
        )
