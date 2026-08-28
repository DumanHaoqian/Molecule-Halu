"""Tests for immutable typed state-DAG objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from enum import Enum
from unittest.mock import Mock

import pytest

from molhallulens.domain.enums import (
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
from molhallulens.domain.state_dag import (
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
