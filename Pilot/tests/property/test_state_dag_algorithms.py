"""Property tests for deterministic, directional state-schema graph queries."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

import pytest
from hypothesis import given, settings, strategies as st

from molhallulens.domain.enums import (
    ComparatorKind,
    DependencyType,
    NodeRole,
    ValueProvenance,
    ValueType,
    Visibility,
)
from molhallulens.domain.state_dag import (
    ClaimValue,
    StateDAG,
    StateEdge,
    StateNodeSpec,
    StateSchema,
)


@dataclass(frozen=True, slots=True)
class _DAGCase:
    node_ids: tuple[str, ...]
    edge_pairs: tuple[tuple[str, str], ...]
    node_order: tuple[str, ...]
    edge_order: tuple[tuple[str, str], ...]
    root: str
    roots: frozenset[str]
    selected: frozenset[str]
    seeds_a: frozenset[str]
    seeds_b: frozenset[str]
    changed: frozenset[str]


@st.composite
def _dag_cases(draw: st.DrawFn) -> _DAGCase:
    """Generate DAGs by selecting only edges that advance through a random rank."""

    node_count = draw(st.integers(min_value=1, max_value=8))
    node_ids = tuple(f"n{index:02d}" for index in range(node_count))
    rank_order = draw(st.permutations(node_ids))
    allowed_edges = tuple(
        (rank_order[source_index], rank_order[target_index])
        for source_index in range(node_count)
        for target_index in range(source_index + 1, node_count)
    )
    if allowed_edges:
        edge_pairs = tuple(
            sorted(draw(st.sets(st.sampled_from(allowed_edges), max_size=len(allowed_edges))))
        )
    else:
        edge_pairs = ()

    node_order = draw(st.permutations(node_ids))
    edge_order = draw(st.permutations(edge_pairs))
    root = draw(st.sampled_from(node_ids))
    node_subset = st.sets(st.sampled_from(node_ids), max_size=node_count)
    return _DAGCase(
        node_ids=node_ids,
        edge_pairs=edge_pairs,
        node_order=node_order,
        edge_order=edge_order,
        root=root,
        roots=frozenset(
            draw(st.sets(st.sampled_from(node_ids), min_size=1, max_size=node_count))
        ),
        selected=frozenset(draw(node_subset)),
        seeds_a=frozenset(draw(node_subset)),
        seeds_b=frozenset(draw(node_subset)),
        changed=frozenset(draw(node_subset)),
    )


def _node(node_id: str) -> StateNodeSpec:
    return StateNodeSpec(
        node_id=node_id,
        value_type=ValueType.STRING,
        step_index=None,
        role=NodeRole.DERIVED_CLAIM,
        visibility=Visibility.CANDIDATE_OUTPUT,
        mutable=True,
        comparator=ComparatorKind.EXACT,
    )


def _edge(
    source: str,
    target: str,
    *,
    suffix: str = "",
    relation: DependencyType = DependencyType.DERIVED_FROM,
) -> StateEdge:
    return StateEdge(
        edge_id=f"edge.{source}.{target}{suffix}",
        source=source,
        target=target,
        relation=relation,
    )


def _schema(
    node_order: Iterable[str],
    edge_order: Iterable[tuple[str, str]],
) -> StateSchema:
    return StateSchema(
        schema_id="property.state_dag",
        version="1.0",
        nodes=tuple(_node(node_id) for node_id in node_order),
        edges=tuple(_edge(source, target) for source, target in edge_order),
    )


def _claim(
    normalized_value: str,
    *,
    raw_value: str | None = None,
    locally_valid: bool | None = None,
) -> ClaimValue:
    return ClaimValue(
        raw_value=normalized_value if raw_value is None else raw_value,
        normalized_value=normalized_value,
        value_type=ValueType.STRING,
        provenance=ValueProvenance.RULE,
        locally_valid=locally_valid,
    )


def _state_dag(
    schema: StateSchema,
    normalized_values: dict[str, str],
    *,
    edge_statuses: dict[str, bool | None] | None = None,
) -> StateDAG:
    return StateDAG(
        schema=schema,
        values={
            node.node_id: _claim(normalized_values[node.node_id])
            for node in schema.nodes
        },
        edge_values={
            edge_id: _claim("edge-status", locally_valid=status)
            for edge_id, status in (edge_statuses or {}).items()
        },
    )


def _lexical_topological_order(
    node_ids: Iterable[str],
    edge_pairs: Iterable[tuple[str, str]],
) -> tuple[str, ...]:
    nodes = tuple(node_ids)
    indegree = dict.fromkeys(nodes, 0)
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for source, target in edge_pairs:
        outgoing[source].append(target)
        indegree[target] += 1
    ready = [node_id for node_id in nodes if indegree[node_id] == 0]
    heapify(ready)
    result: list[str] = []
    while ready:
        node_id = heappop(ready)
        result.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heappush(ready, target)
    assert len(result) == len(nodes)
    return tuple(result)


def _strict_descendants(
    root: str,
    edge_pairs: Iterable[tuple[str, str]],
) -> frozenset[str]:
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edge_pairs:
        outgoing[source].append(target)
    reached: set[str] = set()
    frontier = deque(outgoing[root])
    while frontier:
        node_id = frontier.popleft()
        if node_id in reached:
            continue
        reached.add(node_id)
        frontier.extend(outgoing[node_id])
    return frozenset(reached)


def _inclusive_downstream_closure(
    roots: Iterable[str],
    edge_pairs: Iterable[tuple[str, str]],
) -> frozenset[str]:
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edge_pairs:
        outgoing[source].append(target)
    closure = set(roots)
    frontier = deque(closure)
    while frontier:
        node_id = frontier.popleft()
        for target in outgoing[node_id]:
            if target not in closure:
                closure.add(target)
                frontier.append(target)
    return frozenset(closure)


def _is_connected_downstream(
    roots: frozenset[str],
    selected: frozenset[str],
    edge_pairs: Iterable[tuple[str, str]],
) -> bool:
    allowed = _inclusive_downstream_closure(roots, edge_pairs)
    if not roots <= selected or not selected <= allowed:
        return False
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edge_pairs:
        outgoing[source].append(target)
    reached = set(roots)
    frontier = deque(roots)
    while frontier:
        node_id = frontier.popleft()
        for target in outgoing[node_id]:
            if target in selected and target not in reached:
                reached.add(target)
                frontier.append(target)
    return reached == set(selected)


@settings(max_examples=100, deadline=None)
@given(_dag_cases())
def test_topological_order_is_lexical_complete_and_declaration_invariant(
    case: _DAGCase,
) -> None:
    schema = _schema(case.node_order, case.edge_order)
    expected = _lexical_topological_order(case.node_ids, case.edge_pairs)

    actual = schema.topological_order()

    assert isinstance(actual, tuple)
    assert actual == expected
    assert actual == schema.topological_order()
    assert set(actual) == set(case.node_ids)
    positions = {node_id: index for index, node_id in enumerate(actual)}
    assert all(positions[source] < positions[target] for source, target in case.edge_pairs)
    reordered_schema = _schema(reversed(case.node_order), reversed(case.edge_order))
    assert reordered_schema.topological_order() == expected


@settings(max_examples=100, deadline=None)
@given(_dag_cases())
def test_descendants_match_independent_reachability_oracle(case: _DAGCase) -> None:
    schema = _schema(case.node_order, case.edge_order)
    expected_set = _strict_descendants(case.root, case.edge_pairs)
    expected = tuple(
        node_id for node_id in schema.topological_order() if node_id in expected_set
    )

    actual = schema.descendants(case.root)

    assert isinstance(actual, tuple)
    assert actual == expected
    assert case.root not in actual
    assert len(actual) == len(set(actual))
    assert _schema(reversed(case.node_order), reversed(case.edge_order)).descendants(
        case.root
    ) == expected


@settings(max_examples=100, deadline=None)
@given(_dag_cases())
def test_dependency_closure_matches_oracle_and_obeys_set_laws(case: _DAGCase) -> None:
    schema = _schema(case.node_order, case.edge_order)
    topology = schema.topological_order()
    closure_a = schema.dependency_closure(case.seeds_a)
    closure_b = schema.dependency_closure(case.seeds_b)
    closure_union = schema.dependency_closure(case.seeds_a | case.seeds_b)
    expected_a = _inclusive_downstream_closure(case.seeds_a, case.edge_pairs)

    assert isinstance(closure_a, tuple)
    assert closure_a == tuple(node_id for node_id in topology if node_id in expected_a)
    assert case.seeds_a <= set(closure_a)
    assert schema.dependency_closure(closure_a) == closure_a
    assert set(closure_union) == set(closure_a) | set(closure_b)
    assert set(closure_a) <= set(closure_union)
    assert set(closure_b) <= set(closure_union)
    assert _schema(reversed(case.node_order), reversed(case.edge_order)).dependency_closure(
        case.seeds_a
    ) == closure_a


@settings(max_examples=100, deadline=None)
@given(_dag_cases())
def test_connected_downstream_query_matches_induced_reachability_oracle(
    case: _DAGCase,
) -> None:
    schema = _schema(case.node_order, case.edge_order)
    structural = _is_connected_downstream(case.roots, case.selected, case.edge_pairs)

    assert (
        schema.is_connected_downstream_subgraph(case.roots, case.selected)
        is structural
    )
    reversed_schema = _schema(reversed(case.node_order), reversed(case.edge_order))
    assert (
        reversed_schema.is_connected_downstream_subgraph(case.roots, case.selected)
        is structural
    )


@settings(max_examples=100, deadline=None)
@given(_dag_cases())
def test_stale_downstream_edges_are_exact_frontier_and_declaration_invariant(
    case: _DAGCase,
) -> None:
    schema = _schema(case.node_order, case.edge_order)
    topology = _lexical_topological_order(case.node_ids, case.edge_pairs)
    topo_index = {node_id: index for index, node_id in enumerate(topology)}
    expected_ids = tuple(
        f"edge.{source}.{target}"
        for source, target in sorted(
            (
                (source, target)
                for source, target in case.edge_pairs
                if source in case.changed and target not in case.changed
            ),
            key=lambda pair: (
                topo_index[pair[0]],
                topo_index[pair[1]],
                f"edge.{pair[0]}.{pair[1]}",
            ),
        )
    )

    actual = schema.stale_downstream_edges(case.changed)

    assert isinstance(actual, tuple)
    assert tuple(edge.edge_id for edge in actual) == expected_ids
    assert all(
        edge.source in case.changed and edge.target not in case.changed
        for edge in actual
    )
    reversed_schema = _schema(reversed(case.node_order), reversed(case.edge_order))
    reordered_ids = tuple(
        edge.edge_id for edge in reversed_schema.stale_downstream_edges(case.changed)
    )
    assert reordered_ids == expected_ids


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=2, max_value=8), st.data())
def test_schema_rejects_generated_cycles_including_reordered_declarations(
    node_count: int,
    data: st.DataObject,
) -> None:
    node_ids = tuple(f"n{index:02d}" for index in range(node_count))
    cycle_edges = tuple(
        (node_ids[index], node_ids[(index + 1) % node_count])
        for index in range(node_count)
    )
    node_order = data.draw(st.permutations(node_ids))
    edge_order = data.draw(st.permutations(cycle_edges))

    with pytest.raises(ValueError, match="acyclic"):
        _schema(node_order, edge_order)


def test_disconnected_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        _schema(
            ("root", "leaf", "cycle_a", "cycle_b"),
            (
                ("root", "leaf"),
                ("cycle_a", "cycle_b"),
                ("cycle_b", "cycle_a"),
            ),
        )


def test_diamond_requires_an_induced_path_from_each_declared_root() -> None:
    schema = _schema(
        ("join", "right", "root", "left"),
        (
            ("right", "join"),
            ("root", "right"),
            ("left", "join"),
            ("root", "left"),
        ),
    )
    full = {"root", "left", "right", "join"}

    assert schema.descendants("root") == ("left", "right", "join")
    assert not schema.is_connected_downstream_subgraph(
        {"root"}, {"root", "join"}
    )
    assert schema.is_connected_downstream_subgraph(
        {"root"}, {"root", "left", "join"}
    )
    assert schema.is_connected_downstream_subgraph({"root"}, {"root"})
    assert schema.is_connected_downstream_subgraph({"root"}, full)
    assert schema.is_connected_downstream_subgraph(
        {"left", "right"}, {"left", "right", "join"}
    )
    assert not schema.is_connected_downstream_subgraph(
        {"left", "right"}, {"left", "join"}
    )


def test_parallel_edges_do_not_shift_topology_and_each_is_in_stale_frontier() -> None:
    schema = StateSchema(
        schema_id="property.parallel_edges",
        version="1.0",
        nodes=(_node("target"), _node("source")),
        edges=(
            _edge("source", "target", suffix=".z"),
            _edge("source", "target", suffix=".a"),
        ),
    )

    assert schema.topological_order() == ("source", "target")
    assert schema.descendants("source") == ("target",)
    assert schema.dependency_closure(("source",)) == ("source", "target")
    assert tuple(
        edge.edge_id for edge in schema.stale_downstream_edges(("source",))
    ) == (
        "edge.source.target.a",
        "edge.source.target.z",
    )


def test_full_downstream_change_has_no_stale_frontier_and_direction_is_not_reversed() -> None:
    schema = _schema(
        ("root", "left", "right", "leaf"),
        (("root", "left"), ("root", "right"), ("left", "leaf")),
    )

    assert schema.stale_downstream_edges(()) == ()
    assert schema.stale_downstream_edges(("left",)) == (
        schema.edges_by_id["edge.left.leaf"],
    )
    assert schema.stale_downstream_edges(("leaf",)) == ()
    assert schema.stale_downstream_edges(
        {"root", *schema.descendants("root")}
    ) == ()


def test_must_equal_remains_unknown_until_an_evaluator_supplies_semantics() -> None:
    edge = _edge(
        "source",
        "target",
        relation=DependencyType.MUST_EQUAL,
    )
    schema = StateSchema(
        schema_id="property.must_equal",
        version="1.0",
        nodes=(_node("target"), _node("source")),
        edges=(edge,),
    )
    matching = StateDAG(
        schema=schema,
        values={
            "source": _claim("canonical", raw_value="raw source spelling"),
            "target": _claim("canonical", raw_value="raw target spelling"),
        },
    )
    mismatching = _state_dag(
        schema,
        {"source": "canonical", "target": "different"},
    )

    def evaluator(_edge: StateEdge, dag: StateDAG) -> bool:
        return dag.values["source"].semantically_equals(dag.values["target"])

    assert matching.edge_satisfaction(edge.edge_id) is None
    assert matching.violated_edges() == ()
    assert mismatching.edge_satisfaction(edge.edge_id) is None
    assert mismatching.violated_edges() == ()
    assert matching.edge_satisfaction(edge.edge_id, evaluator=evaluator) is True
    assert matching.violated_edges(evaluator=evaluator) == ()
    assert mismatching.edge_satisfaction(edge.edge_id, evaluator=evaluator) is False
    assert mismatching.violated_edges(evaluator=evaluator) == (edge,)


@pytest.mark.parametrize(
    ("status", "expected_violations"),
    ((True, ()), (False, ("edge.source.target",)), (None, ())),
)
def test_explicit_edge_local_valid_status_controls_violation_query(
    status: bool | None,
    expected_violations: tuple[str, ...],
) -> None:
    edge = _edge("source", "target")
    schema = StateSchema(
        schema_id="property.explicit_status",
        version="1.0",
        nodes=(_node("source"), _node("target")),
        edges=(edge,),
    )
    dag = _state_dag(
        schema,
        {"source": "same", "target": "same"},
        edge_statuses={edge.edge_id: status},
    )

    assert dag.edge_satisfaction(edge.edge_id) is status
    assert tuple(item.edge_id for item in dag.violated_edges()) == expected_violations


def test_non_must_equal_edge_without_status_or_evaluator_remains_unknown() -> None:
    edge = _edge("source", "target")
    schema = StateSchema(
        schema_id="property.unknown_satisfaction",
        version="1.0",
        nodes=(_node("source"), _node("target")),
        edges=(edge,),
    )
    dag = _state_dag(schema, {"source": "same", "target": "same"})

    assert dag.edge_satisfaction(edge.edge_id) is None
    assert dag.violated_edges() == ()


def test_injected_evaluator_drives_unknown_relations_and_violation_order_is_stable() -> None:
    edges = (
        StateEdge("edge.z", "root", "z", DependencyType.DERIVED_FROM),
        StateEdge("edge.m", "root", "m", DependencyType.DERIVED_FROM),
        StateEdge("edge.a", "root", "a", DependencyType.DERIVED_FROM),
    )
    schema = StateSchema(
        schema_id="property.injected_evaluator",
        version="1.0",
        nodes=tuple(_node(node_id) for node_id in ("z", "root", "m", "a")),
        edges=edges,
    )
    dag = _state_dag(schema, {node.node_id: "value" for node in schema.nodes})
    statuses = {"edge.a": False, "edge.m": None, "edge.z": False}

    def evaluator(edge: StateEdge, observed_dag: StateDAG) -> bool | None:
        assert observed_dag is dag
        return statuses[edge.edge_id]

    assert dag.edge_satisfaction("edge.m", evaluator=evaluator) is None
    assert tuple(edge.edge_id for edge in dag.violated_edges(evaluator=evaluator)) == (
        "edge.a",
        "edge.z",
    )

    reordered_schema = StateSchema(
        schema_id="property.injected_evaluator",
        version="1.0",
        nodes=tuple(reversed(schema.nodes)),
        edges=tuple(reversed(edges)),
    )
    reordered_dag = _state_dag(
        reordered_schema,
        {node.node_id: "value" for node in reordered_schema.nodes},
    )

    def reordered_evaluator(edge: StateEdge, observed_dag: StateDAG) -> bool | None:
        assert observed_dag is reordered_dag
        return statuses[edge.edge_id]

    assert tuple(
        edge.edge_id
        for edge in reordered_dag.violated_edges(evaluator=reordered_evaluator)
    ) == ("edge.a", "edge.z")


def test_agreeing_explicit_and_injected_statuses_pass_but_conflicts_fail_closed() -> None:
    must_equal = _edge(
        "source",
        "target",
        relation=DependencyType.MUST_EQUAL,
    )
    must_equal_schema = StateSchema(
        schema_id="property.satisfaction_conflict.must_equal",
        version="1.0",
        nodes=(_node("source"), _node("target")),
        edges=(must_equal,),
    )
    agreeing = _state_dag(
        must_equal_schema,
        {"source": "same", "target": "same"},
        edge_statuses={must_equal.edge_id: True},
    )
    assert agreeing.edge_satisfaction(
        must_equal.edge_id,
        evaluator=lambda _edge, _dag: True,
    ) is True
    with pytest.raises(ValueError, match="conflict"):
        agreeing.edge_satisfaction(
            must_equal.edge_id,
            evaluator=lambda _edge, _dag: False,
        )
    with pytest.raises(ValueError, match="conflict"):
        agreeing.violated_edges(evaluator=lambda _edge, _dag: False)


@pytest.mark.parametrize("invalid_result", (0, 1, "false", object()))
def test_evaluator_rejects_non_boolean_non_none_results(invalid_result: object) -> None:
    edge = _edge("source", "target")
    schema = StateSchema(
        schema_id="property.invalid_evaluator_result",
        version="1.0",
        nodes=(_node("source"), _node("target")),
        edges=(edge,),
    )
    dag = _state_dag(schema, {"source": "same", "target": "same"})

    with pytest.raises(TypeError, match="evaluator"):
        dag.edge_satisfaction(
            edge.edge_id,
            evaluator=lambda _edge, _dag: invalid_result,  # type: ignore[return-value]
        )
    with pytest.raises(TypeError, match="evaluator"):
        dag.violated_edges(
            evaluator=lambda _edge, _dag: invalid_result,  # type: ignore[return-value]
        )


def test_semantic_edge_query_inputs_fail_closed() -> None:
    edge = _edge("source", "target")
    schema = StateSchema(
        schema_id="property.semantic_inputs",
        version="1.0",
        nodes=(_node("source"), _node("target")),
        edges=(edge,),
    )
    dag = _state_dag(schema, {"source": "same", "target": "same"})

    with pytest.raises(TypeError):
        dag.edge_satisfaction(1)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        dag.edge_satisfaction("missing")
    with pytest.raises(TypeError, match="evaluator"):
        dag.edge_satisfaction(edge.edge_id, evaluator=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evaluator"):
        dag.violated_edges(evaluator=True)  # type: ignore[arg-type]

    def evaluator_failure(_edge: StateEdge, _dag: StateDAG) -> bool | None:
        raise RuntimeError("evaluator failure")

    with pytest.raises(RuntimeError, match="evaluator failure"):
        dag.edge_satisfaction(edge.edge_id, evaluator=evaluator_failure)


@pytest.mark.parametrize(
    ("call", "error_type"),
    (
        (lambda schema: schema.descendants(1), TypeError),
        (lambda schema: schema.descendants("missing"), KeyError),
        (lambda schema: schema.dependency_closure("root"), TypeError),
        (lambda schema: schema.dependency_closure((None,)), TypeError),
        (lambda schema: schema.dependency_closure(("missing",)), KeyError),
        (
            lambda schema: schema.is_connected_downstream_subgraph("root", ("root",)),
            TypeError,
        ),
        (
            lambda schema: schema.is_connected_downstream_subgraph((), ("root",)),
            ValueError,
        ),
        (
            lambda schema: schema.is_connected_downstream_subgraph(
                ("missing",), ("root",)
            ),
            KeyError,
        ),
        (lambda schema: schema.stale_downstream_edges("root"), TypeError),
        (lambda schema: schema.stale_downstream_edges((1,)), TypeError),
        (lambda schema: schema.stale_downstream_edges(("missing",)), KeyError),
    ),
)
def test_query_inputs_fail_closed(call: object, error_type: type[Exception]) -> None:
    schema = _schema(("root", "leaf"), (("root", "leaf"),))

    with pytest.raises(error_type):
        call(schema)  # type: ignore[operator]
