"""Tests for immutable typed state-DAG objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from enum import Enum
from unittest.mock import Mock

import pytest

from molhallulens.core.enums import (
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
)
from molhallulens.core.state_dag import (
    ClaimValue,
    FrozenMap,
    GraphDelta,
    MutationEvent,
    StateDAG,
    StateEdge,
    StateNodeSpec,
    StateSchema,
    deep_freeze,
)


def _node(
    node_id: str,
    value_type: ValueType,
    role: NodeRole,
    visibility: Visibility,
    *,
    mutable: bool = True,
    renderer_slot: str | None = None,
) -> StateNodeSpec:
    return StateNodeSpec(
        node_id=node_id,
        value_type=value_type,
        step_index=None,
        role=role,
        visibility=visibility,
        mutable=mutable,
        comparator=ComparatorKind.EXACT,
        renderer_slot=renderer_slot,
    )


def _claim(value: object, value_type: ValueType, provenance: ValueProvenance) -> ClaimValue:
    return ClaimValue(
        raw_value=value,
        normalized_value=value,
        value_type=value_type,
        provenance=provenance,
    )


def _schema() -> StateSchema:
    return StateSchema(
        schema_id="mol_edit.test",
        version="1.0",
        nodes=(
            _node(
                "source",
                ValueType.INDEXED_SMILES,
                NodeRole.EVIDENCE,
                Visibility.PROMPT_PREFIX,
                mutable=False,
                renderer_slot="indexed_smiles",
            ),
            _node(
                "oracle_gt",
                ValueType.SMILES,
                NodeRole.INTERNAL_TRUTH,
                Visibility.BUILD_ONLY,
                mutable=False,
            ),
            _node(
                "anchor.idx",
                ValueType.ATOM_INDEX,
                NodeRole.PRIMARY_CLAIM,
                Visibility.CANDIDATE_OUTPUT,
                renderer_slot="anchor_idx",
            ),
            _node(
                "final_answer",
                ValueType.SMILES,
                NodeRole.FINAL_ANSWER,
                Visibility.CANDIDATE_OUTPUT,
                renderer_slot="final_answer",
            ),
        ),
        edges=(
            StateEdge("edge.source.anchor", "source", "anchor.idx", DependencyType.DERIVED_FROM),
            StateEdge(
                "edge.anchor.answer",
                "anchor.idx",
                "final_answer",
                DependencyType.EDIT_PRODUCES,
                mutable=True,
                renderer_slot="edit_relation",
            ),
        ),
    )


def _dag() -> StateDAG:
    return StateDAG(
        schema=_schema(),
        values={
            "source": _claim("[C:1]", ValueType.INDEXED_SMILES, ValueProvenance.REFERENCE),
            "oracle_gt": _claim("CN", ValueType.SMILES, ValueProvenance.REFERENCE),
            "anchor.idx": _claim(1, ValueType.ATOM_INDEX, ValueProvenance.REFERENCE),
            "final_answer": _claim("CN", ValueType.SMILES, ValueProvenance.REFERENCE),
        },
    )


def _algorithm_schema(*, reverse_declarations: bool = False) -> StateSchema:
    nodes = tuple(
        _node(
            node_id,
            ValueType.STRING,
            NodeRole.PRIMARY_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
        )
        for node_id in ("root", "left", "right", "join", "leaf", "isolated")
    )
    edges = (
        StateEdge("edge.root.left", "root", "left", DependencyType.DERIVED_FROM),
        StateEdge("edge.root.right", "root", "right", DependencyType.DERIVED_FROM),
        StateEdge("edge.left.join", "left", "join", DependencyType.DERIVED_FROM),
        StateEdge("edge.right.join", "right", "join", DependencyType.DERIVED_FROM),
        StateEdge("edge.join.leaf", "join", "leaf", DependencyType.DERIVED_FROM),
    )
    return StateSchema(
        schema_id="algorithm.test",
        version="1.0",
        nodes=tuple(reversed(nodes)) if reverse_declarations else nodes,
        edges=tuple(reversed(edges)) if reverse_declarations else edges,
    )


def test_claim_value_recursively_freezes_container_values() -> None:
    original = {"atoms": [1, 2]}
    claim = ClaimValue(
        raw_value=original,
        normalized_value={1, 2},
        value_type=ValueType.ATOM_SET,
        provenance=ValueProvenance.RDKIT,
    )
    original["atoms"].append(3)

    assert isinstance(claim.raw_value, FrozenMap)
    assert claim.raw_value["atoms"] == (1, 2)
    assert claim.normalized_value == frozenset({1, 2})
    with pytest.raises(TypeError):
        claim.raw_value["atoms"] = (9,)  # type: ignore[index]

    reordered = ClaimValue(
        raw_value=[2, 1, 1],
        normalized_value=[2, 1, 1],
        value_type=ValueType.ATOM_SET,
        provenance=ValueProvenance.RULE,
    )
    assert reordered.normalized_value == frozenset({1, 2})


def test_claim_value_rejects_forged_types_and_opaque_mutable_values() -> None:
    with pytest.raises(TypeError, match="integer normalized"):
        ClaimValue(
            raw_value="not-an-int",
            normalized_value="not-an-int",
            value_type=ValueType.INTEGER,
            provenance=ValueProvenance.RULE,
        )
    with pytest.raises(TypeError, match="locally_valid"):
        ClaimValue(
            raw_value=1,
            normalized_value=1,
            value_type=ValueType.INTEGER,
            provenance=ValueProvenance.RULE,
            locally_valid="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite"):
        ClaimValue(
            raw_value=float("nan"),
            normalized_value=float("nan"),
            value_type=ValueType.FLOAT,
            provenance=ValueProvenance.RULE,
        )
    with pytest.raises(ValueError, match="non-negative"):
        ClaimValue(
            raw_value=-1,
            normalized_value=-1,
            value_type=ValueType.ATOM_INDEX,
            provenance=ValueProvenance.RULE,
        )

    class MutablePayload:
        pass

    with pytest.raises(TypeError, match="unsupported mutable or opaque"):
        ClaimValue(
            raw_value=MutablePayload(),
            normalized_value="opaque",
            value_type=ValueType.UNKNOWN,
            provenance=ValueProvenance.RULE,
        )

    binary = bytearray(b"abc")
    claim = ClaimValue(
        raw_value=binary,
        normalized_value=binary,
        value_type=ValueType.UNKNOWN,
        provenance=ValueProvenance.RULE,
    )
    binary[0] = 0
    assert claim.raw_value == b"abc"
    assert claim.normalized_value == b"abc"


def test_deep_freeze_covers_init_false_fields_and_rejects_scalar_subclasses() -> None:
    @dataclass(frozen=True, slots=True)
    class HiddenMutable:
        values: list[int] = field(default_factory=list, init=False)

    original = HiddenMutable()
    original.values.append(1)
    frozen = deep_freeze(original)
    original.values.append(2)
    assert frozen.values == (1,)

    @dataclass(frozen=True)
    class Unslotted:
        value: int

    with pytest.raises(TypeError, match="slots"):
        deep_freeze(Unslotted(1))

    class MutableInt(int):
        pass

    number = MutableInt(7)
    number.sidecar = []
    with pytest.raises(TypeError, match="unsupported mutable or opaque"):
        deep_freeze(number)

    class ForgedEnum(Enum):
        VALUE = "value"

    ForgedEnum.__module__ = CausalRole.__module__
    with pytest.raises(TypeError, match="MolHalluLens domain enums"):
        deep_freeze(ForgedEnum.VALUE)

    backing = {"safe": [1]}
    frozen_map = FrozenMap(backing)
    backing["safe"].append(2)
    assert frozen_map["safe"] == (1,)
    assert not hasattr(FrozenMap, "_from_frozen_data")
    with pytest.raises(TypeError, match="immutable"):
        del frozen_map._data


def test_schema_and_dag_are_frozen_and_typed() -> None:
    dag = _dag()

    assert dag.value_for("anchor.idx").raw_value == 1
    assert not isinstance(dag.values, dict)
    with pytest.raises(TypeError):
        dag.values["anchor.idx"] = _claim(2, ValueType.ATOM_INDEX, ValueProvenance.RULE)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        dag.schema.schema_id = "changed"  # type: ignore[misc]

    with pytest.raises(TypeError, match="value_type"):
        StateNodeSpec(
            node_id="forged",
            value_type=Mock(spec=ValueType),
            step_index=None,
            role=Mock(spec=NodeRole),
            visibility=Mock(spec=Visibility),
            mutable=False,
            comparator=Mock(spec=ComparatorKind),
        )
    with pytest.raises(TypeError, match="node_id"):
        _node(
            NodeRole.EVIDENCE,  # type: ignore[arg-type]
            ValueType.STRING,
            NodeRole.EVIDENCE,
            Visibility.PROMPT_PREFIX,
            mutable=False,
        )


def test_schema_rejects_private_renderer_slots_and_dangling_edges() -> None:
    with pytest.raises(ValueError, match="BUILD_ONLY"):
        _node(
            "oracle",
            ValueType.SMILES,
            NodeRole.INTERNAL_TRUTH,
            Visibility.BUILD_ONLY,
            renderer_slot="leak_gt",
        )

    with pytest.raises(TypeError, match="role"):
        StateNodeSpec(
            node_id="oracle",
            value_type=ValueType.SMILES,
            step_index=None,
            role="internal_truth",  # type: ignore[arg-type]
            visibility="candidate_output",  # type: ignore[arg-type]
            mutable=False,
            comparator=ComparatorKind.EXACT,
            renderer_slot="gt_slot",
        )

    with pytest.raises(ValueError, match="dangling"):
        StateSchema(
            schema_id="broken",
            version="1.0",
            nodes=(
                _node(
                    "source",
                    ValueType.STRING,
                    NodeRole.EVIDENCE,
                    Visibility.PROMPT_PREFIX,
                    mutable=False,
                ),
            ),
            edges=(StateEdge("bad", "source", "missing", DependencyType.DERIVED_FROM),),
        )

    oracle = _node(
        "oracle",
        ValueType.SMILES,
        NodeRole.INTERNAL_TRUTH,
        Visibility.BUILD_ONLY,
        mutable=False,
    )
    output = _node(
        "output",
        ValueType.SMILES,
        NodeRole.FINAL_ANSWER,
        Visibility.CANDIDATE_OUTPUT,
    )
    with pytest.raises(ValueError, match="BUILD_ONLY"):
        StateSchema(
            "leaky-edge",
            "1.0",
            (oracle, output),
            (
                StateEdge(
                    "oracle-to-output",
                    "oracle",
                    "output",
                    DependencyType.MUST_EQUAL,
                    renderer_slot="oracle_relation",
                ),
            ),
        )


def test_state_schema_rejects_cycles() -> None:
    nodes = (
        _node("a", ValueType.STRING, NodeRole.PRIMARY_CLAIM, Visibility.CANDIDATE_OUTPUT),
        _node("b", ValueType.STRING, NodeRole.DERIVED_CLAIM, Visibility.CANDIDATE_OUTPUT),
    )
    edges = (
        StateEdge("a-to-b", "a", "b", DependencyType.DERIVED_FROM),
        StateEdge("b-to-a", "b", "a", DependencyType.DERIVED_FROM),
    )

    with pytest.raises(ValueError, match="acyclic"):
        StateSchema("cyclic", "1.0", nodes, edges)


def test_graph_algorithms_are_deterministic_and_directionally_distinct() -> None:
    schema = _algorithm_schema()
    reversed_schema = _algorithm_schema(reverse_declarations=True)
    expected_topology = ("isolated", "root", "left", "right", "join", "leaf")

    assert schema.topological_order() == expected_topology
    assert reversed_schema.topological_order() == expected_topology
    assert schema.descendants("root") == ("left", "right", "join", "leaf")
    assert schema.descendants("isolated") == ()
    assert schema.dependency_closure(("root",)) == (
        "root",
        "left",
        "right",
        "join",
        "leaf",
    )
    assert schema.dependency_closure(("right", "left", "right")) == (
        "left",
        "right",
        "join",
        "leaf",
    )
    assert schema.dependency_closure(()) == ()


def test_connected_downstream_subgraph_requires_induced_directed_paths() -> None:
    schema = _algorithm_schema()
    full = {"root", *schema.descendants("root")}

    assert schema.is_connected_downstream_subgraph(("root",), {"root"})
    assert schema.is_connected_downstream_subgraph(
        ("root",), {"root", "left", "join"}
    )
    assert schema.is_connected_downstream_subgraph(
        ("root",), {"root", "left"}
    )
    assert schema.is_connected_downstream_subgraph(("root",), full)
    assert not schema.is_connected_downstream_subgraph(
        ("root",), {"root", "join"}
    )
    assert not schema.is_connected_downstream_subgraph(
        ("root",), {"root", "isolated"}
    )
    assert schema.is_connected_downstream_subgraph(
        ("left", "right"), {"left", "right", "join", "leaf"}
    )


def test_violated_edges_are_the_stable_stale_downstream_frontier() -> None:
    schema = _algorithm_schema(reverse_declarations=True)

    assert tuple(edge.edge_id for edge in schema.stale_downstream_edges(("root",))) == (
        "edge.root.left",
        "edge.root.right",
    )
    assert tuple(
        edge.edge_id for edge in schema.stale_downstream_edges(("root", "left"))
    ) == ("edge.root.right", "edge.left.join")
    assert tuple(edge.edge_id for edge in schema.stale_downstream_edges(("left",))) == (
        "edge.left.join",
    )
    assert schema.stale_downstream_edges(()) == ()
    assert schema.stale_downstream_edges(schema.topological_order()) == ()


def test_edge_violation_query_preserves_unknown_and_accepts_evaluators() -> None:
    schema = StateSchema(
        schema_id="edge-status.test",
        version="1.0",
        nodes=(
            _node("a", ValueType.STRING, NodeRole.PRIMARY_CLAIM, Visibility.CANDIDATE_OUTPUT),
            _node("b", ValueType.STRING, NodeRole.DERIVED_CLAIM, Visibility.CANDIDATE_OUTPUT),
            _node("c", ValueType.STRING, NodeRole.DERIVED_CLAIM, Visibility.CANDIDATE_OUTPUT),
        ),
        edges=(
            StateEdge("edge.must", "a", "b", DependencyType.MUST_EQUAL),
            StateEdge("edge.explicit", "a", "c", DependencyType.DERIVED_FROM),
            StateEdge("edge.unknown", "b", "c", DependencyType.DERIVED_FROM),
        ),
    )
    dag = StateDAG(
        schema=schema,
        values={
            "a": _claim("C", ValueType.STRING, ValueProvenance.REFERENCE),
            "b": _claim("[CH4]", ValueType.STRING, ValueProvenance.REFERENCE),
            "c": _claim("C", ValueType.STRING, ValueProvenance.REFERENCE),
        },
        edge_values={
            "edge.explicit": ClaimValue(
                raw_value=True,
                normalized_value=True,
                value_type=ValueType.BOOLEAN,
                provenance=ValueProvenance.REFERENCE,
                locally_valid=False,
            )
        },
    )

    assert dag.edge_satisfaction("edge.explicit") is False
    assert dag.edge_satisfaction("edge.must") is None
    assert dag.edge_satisfaction("edge.unknown") is None
    assert tuple(edge.edge_id for edge in dag.violated_edges()) == ("edge.explicit",)

    def evaluator(edge: StateEdge, _dag: StateDAG) -> bool | None:
        return {
            "edge.explicit": False,
            "edge.must": True,
            "edge.unknown": False,
        }[edge.edge_id]

    assert dag.edge_satisfaction("edge.must", evaluator) is True
    assert tuple(edge.edge_id for edge in dag.violated_edges(evaluator)) == (
        "edge.explicit",
        "edge.unknown",
    )
    with pytest.raises(ValueError, match="conflicting satisfaction"):
        dag.edge_satisfaction("edge.explicit", lambda _edge, _dag: True)
    with pytest.raises(TypeError, match="bool or None"):
        dag.edge_satisfaction("edge.must", lambda _edge, _dag: "yes")  # type: ignore[return-value]
    with pytest.raises(TypeError, match="callable"):
        dag.violated_edges(False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="edge_id"):
        dag.edge_satisfaction(1)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        dag.edge_satisfaction("missing")


def test_graph_algorithm_inputs_fail_closed() -> None:
    schema = _algorithm_schema()

    with pytest.raises(TypeError, match="root_node_id"):
        schema.descendants(1)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        schema.descendants("missing")
    with pytest.raises(TypeError, match="non-string iterable"):
        schema.dependency_closure("root")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="missing"):
        schema.dependency_closure(("missing",))
    with pytest.raises(TypeError, match="root_node_ids"):
        schema.is_connected_downstream_subgraph(
            "root", ("root",)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        schema.is_connected_downstream_subgraph((), ())
    with pytest.raises(TypeError, match="non-string iterable"):
        schema.stale_downstream_edges("root")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="missing"):
        schema.stale_downstream_edges(("missing",))


def test_dag_requires_one_typed_value_for_every_node() -> None:
    schema = _schema()
    with pytest.raises(ValueError, match="missing"):
        StateDAG(schema=schema, values={"source": _claim("C", ValueType.STRING, ValueProvenance.RULE)})

    values = _dag().values.to_dict()
    values["anchor.idx"] = _claim("one", ValueType.STRING, ValueProvenance.RULE)
    with pytest.raises(ValueError, match="type mismatch"):
        StateDAG(schema=schema, values=values)


def test_mutation_events_encode_independent_and_propagated_roles() -> None:
    before = _claim(1, ValueType.ATOM_INDEX, ValueProvenance.REFERENCE)
    root_after = _claim(2, ValueType.ATOM_INDEX, ValueProvenance.RDKIT)
    root = MutationEvent(
        event_id="event.root",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="anchor.idx",
        before=before,
        after=root_after,
        causal_role=CausalRole.ROOT,
        hallucination_types=frozenset(
            {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
        ),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        operator_id="mol_edit.add.alternate_anchor",
        root_event_id="event.root",
    )
    propagated = MutationEvent(
        event_id="event.propagated",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="product",
        before=_claim("CN", ValueType.SMILES, ValueProvenance.REFERENCE),
        after=_claim("NC", ValueType.SMILES, ValueProvenance.PROPAGATED),
        causal_role=CausalRole.PROPAGATED_CONDITIONAL,
        hallucination_types=frozenset({HallucinationType.CONSTRAINT_VIOLATION}),
        edit_subtypes=frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        operator_id="mol_edit.add.alternate_anchor",
        root_event_id="event.root",
    )
    delta = GraphDelta((root, propagated))

    assert delta.root_events == (root,)
    with pytest.raises(ValueError, match="unknown root"):
        GraphDelta((propagated,))


def test_mutation_and_delta_require_a_real_semantic_root_change() -> None:
    before = _claim(1, ValueType.ATOM_INDEX, ValueProvenance.REFERENCE)
    provenance_only = _claim(1, ValueType.ATOM_INDEX, ValueProvenance.RDKIT)
    common = {
        "target_kind": MutationTargetKind.NODE,
        "node_or_edge_id": "anchor.idx",
        "before": before,
        "after": provenance_only,
        "hallucination_types": frozenset({HallucinationType.CONTRADICTION}),
        "edit_subtypes": frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        "operator_id": "mol_edit.add.alternate_anchor",
    }
    with pytest.raises(ValueError, match="normalized values"):
        MutationEvent(
            event_id="event.root",
            causal_role=CausalRole.ROOT,
            root_event_id="event.root",
            **common,
        )

    with pytest.raises(ValueError, match="UNVERIFIABLE"):
        MutationEvent(
            event_id="event.unverifiable",
            target_kind=MutationTargetKind.NODE,
            node_or_edge_id="anchor.idx",
            before=before,
            after=_claim(2, ValueType.ATOM_INDEX, ValueProvenance.RDKIT),
            causal_role=CausalRole.ROOT,
            hallucination_types=frozenset({HallucinationType.UNVERIFIABLE}),
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            operator_id="mol_edit.add.alternate_anchor",
            root_event_id="event.unverifiable",
        )

    def propagated(event_id: str, root_event_id: str) -> MutationEvent:
        return MutationEvent(
            event_id=event_id,
            target_kind=MutationTargetKind.NODE,
            node_or_edge_id=f"node.{event_id}",
            before=before,
            after=_claim(2, ValueType.ATOM_INDEX, ValueProvenance.PROPAGATED),
            causal_role=CausalRole.PROPAGATED_FALSE,
            hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            operator_id="mol_edit.add.alternate_anchor",
            root_event_id=root_event_id,
        )

    with pytest.raises(ValueError, match="non-root or unknown root"):
        GraphDelta((propagated("event.a", "event.b"), propagated("event.b", "event.a")))


def test_terminal_delta_only_changes_final_answer_without_propagation() -> None:
    before = _claim("CN", ValueType.SMILES, ValueProvenance.REFERENCE)
    after = _claim("NC", ValueType.SMILES, ValueProvenance.RULE)
    common = {
        "event_id": "event.terminal",
        "target_kind": MutationTargetKind.NODE,
        "before": before,
        "after": after,
        "causal_role": CausalRole.TERMINAL,
        "hallucination_types": frozenset({HallucinationType.CONTRADICTION}),
        "operator_id": "mol_edit.terminal.answer",
        "root_event_id": "event.terminal",
    }
    with pytest.raises(ValueError, match="final_answer"):
        MutationEvent(
            node_or_edge_id="anchor.idx",
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            **common,
        )

    terminal = MutationEvent(
        node_or_edge_id="final_answer",
        edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        **common,
    )
    child = MutationEvent(
        event_id="event.child",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="product",
        before=before,
        after=after,
        causal_role=CausalRole.PROPAGATED_FALSE,
        hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        operator_id="mol_edit.terminal.answer",
        root_event_id="event.terminal",
    )
    with pytest.raises(ValueError, match="cannot contain propagated"):
        GraphDelta((terminal, child))
