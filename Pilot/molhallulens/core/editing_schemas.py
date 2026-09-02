"""Concrete typed state schemas for the three molecule-editing subtasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

from molhallulens.core.enums import (
    ComparatorKind,
    DependencyType,
    EditingSubtask,
    NodeRole,
    ValueType,
    Visibility,
)
from molhallulens.core.state_dag import (
    FrozenMap,
    StateEdge,
    StateNodeSpec,
    StateSchema,
    freeze_string_mapping,
)


_RECORD_FIELD_BINDINGS = {
    "indexed_smiles": "source",
    "instruction": "instruction",
    "gt_smiles": "oracle_gt",
    "answer_smiles": "final_answer",
}


RegistryValueT = TypeVar("RegistryValueT")


class _EditingSubtaskRegistry(Mapping[EditingSubtask, RegistryValueT], Generic[RegistryValueT]):
    """Read-only registry that does not inherit StrEnum/string key equality."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[EditingSubtask, RegistryValueT]) -> None:
        if any(type(key) is not EditingSubtask for key in values):
            raise TypeError("editing schema registry keys must be EditingSubtask members")
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("editing schema registry is immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("editing schema registry is immutable")

    def __getitem__(self, key: EditingSubtask) -> RegistryValueT:
        if type(key) is not EditingSubtask:
            raise TypeError("editing schema registry keys must be EditingSubtask members")
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _node(
    node_id: str,
    value_type: ValueType,
    step_index: int | None,
    role: NodeRole,
    visibility: Visibility,
    mutable: bool,
    comparator: ComparatorKind,
    renderer_slot: str | None,
) -> StateNodeSpec:
    return StateNodeSpec(
        node_id=node_id,
        value_type=value_type,
        step_index=step_index,
        role=role,
        visibility=visibility,
        mutable=mutable,
        comparator=comparator,
        renderer_slot=renderer_slot,
    )


def _evidence_nodes() -> tuple[StateNodeSpec, ...]:
    return (
        _node(
            "source",
            ValueType.INDEXED_SMILES,
            None,
            NodeRole.EVIDENCE,
            Visibility.PROMPT_PREFIX,
            False,
            ComparatorKind.EXACT,
            None,
        ),
        _node(
            "instruction",
            ValueType.STRING,
            None,
            NodeRole.EVIDENCE,
            Visibility.PROMPT_PREFIX,
            False,
            ComparatorKind.EXACT,
            None,
        ),
        _node(
            "oracle_gt",
            ValueType.SMILES,
            None,
            NodeRole.INTERNAL_TRUTH,
            Visibility.BUILD_ONLY,
            False,
            ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
            None,
        ),
    )


def _anchor_nodes() -> tuple[StateNodeSpec, ...]:
    return (
        _node(
            "anchor_idx",
            ValueType.ATOM_INDEX,
            1,
            NodeRole.PRIMARY_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "anchor.idx",
        ),
        _node(
            "anchor_element",
            ValueType.ELEMENT,
            1,
            NodeRole.PRIMARY_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.EXACT,
            "anchor.element",
        ),
    )


def _count_nodes(
    *,
    heavy_step: int,
    ring_step: int,
) -> tuple[StateNodeSpec, ...]:
    return (
        _node(
            "source_heavy",
            ValueType.COUNT,
            heavy_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "source_heavy",
        ),
        _node(
            "product_heavy",
            ValueType.COUNT,
            heavy_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "product_heavy",
        ),
        _node(
            "heavy_delta",
            ValueType.INTEGER,
            heavy_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "heavy_delta",
        ),
        _node(
            "source_rings",
            ValueType.COUNT,
            ring_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "source_rings",
        ),
        _node(
            "product_rings",
            ValueType.COUNT,
            ring_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "product_rings",
        ),
        _node(
            "ring_delta",
            ValueType.INTEGER,
            ring_step,
            NodeRole.DERIVED_CLAIM,
            Visibility.CANDIDATE_OUTPUT,
            True,
            ComparatorKind.INTEGER_EQUAL,
            "ring_delta",
        ),
    )


def _terminal_node() -> StateNodeSpec:
    return _node(
        "final_answer",
        ValueType.SMILES,
        None,
        NodeRole.FINAL_ANSWER,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
        "final_answer.smiles",
    )


def _common_oracle_nodes() -> tuple[StateNodeSpec, ...]:
    specs = (
        ("oracle_anchor_element", ValueType.ELEMENT, ComparatorKind.EXACT),
        ("oracle_source_heavy", ValueType.COUNT, ComparatorKind.INTEGER_EQUAL),
        ("oracle_product_heavy", ValueType.COUNT, ComparatorKind.INTEGER_EQUAL),
        ("oracle_source_rings", ValueType.COUNT, ComparatorKind.INTEGER_EQUAL),
        ("oracle_product_rings", ValueType.COUNT, ComparatorKind.INTEGER_EQUAL),
    )
    return tuple(
        _node(
            node_id,
            value_type,
            None,
            NodeRole.INTERNAL_TRUTH,
            Visibility.BUILD_ONLY,
            False,
            comparator,
            None,
        )
        for node_id, value_type, comparator in specs
    )


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation: DependencyType,
    *,
    mutable: bool = False,
    renderer_slot: str | None = None,
) -> StateEdge:
    return StateEdge(
        edge_id=edge_id,
        source=source,
        target=target,
        relation=relation,
        mutable=mutable,
        renderer_slot=renderer_slot,
    )


def _anchor_edges() -> tuple[StateEdge, ...]:
    return (
        _edge("source_to_anchor", "source", "anchor_idx", DependencyType.DERIVED_FROM),
        _edge(
            "instruction_to_anchor",
            "instruction",
            "anchor_idx",
            DependencyType.CONSTRAINED_BY_INSTRUCTION,
        ),
        _edge(
            "anchor_to_element",
            "anchor_idx",
            "anchor_element",
            DependencyType.DERIVED_FROM,
        ),
    )


def _count_edges() -> tuple[StateEdge, ...]:
    return (
        _edge("source_to_source_heavy", "source", "source_heavy", DependencyType.COUNT_OF),
        _edge(
            "product_to_product_heavy",
            "product",
            "product_heavy",
            DependencyType.COUNT_OF,
        ),
        _edge(
            "source_heavy_to_delta",
            "source_heavy",
            "heavy_delta",
            DependencyType.DELTA_OF,
        ),
        _edge(
            "product_heavy_to_delta",
            "product_heavy",
            "heavy_delta",
            DependencyType.DELTA_OF,
            mutable=True,
            renderer_slot="relation.heavy_delta",
        ),
        _edge("source_to_source_rings", "source", "source_rings", DependencyType.COUNT_OF),
        _edge(
            "product_to_product_rings",
            "product",
            "product_rings",
            DependencyType.COUNT_OF,
        ),
        _edge(
            "source_rings_to_delta",
            "source_rings",
            "ring_delta",
            DependencyType.DELTA_OF,
        ),
        _edge(
            "product_rings_to_delta",
            "product_rings",
            "ring_delta",
            DependencyType.DELTA_OF,
            mutable=True,
            renderer_slot="relation.ring_delta",
        ),
        _edge(
            "product_to_final_answer",
            "product",
            "final_answer",
            DependencyType.MOLECULARLY_EQUIVALENT_TO,
            mutable=True,
            renderer_slot="relation.product_final",
        ),
    )


@dataclass(frozen=True, slots=True)
class EditingStateSchema:
    """A concrete StateSchema plus explicit build-input field bindings."""

    normalized_subtask: EditingSubtask
    schema: StateSchema
    record_field_bindings: Mapping[str, str]
    legacy_step_field_bindings: Mapping[str, str]
    rdkit_reference_bindings: Mapping[str, str]
    semantic_state_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be an EditingSubtask")
        if type(self.schema) is not StateSchema:
            raise TypeError("schema must be a StateSchema")
        if self.schema.schema_id != f"mol_edit.{self.normalized_subtask.value}":
            raise ValueError("schema_id must match normalized_subtask")
        for name in (
            "record_field_bindings",
            "legacy_step_field_bindings",
            "rdkit_reference_bindings",
            "semantic_state_groups",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, freeze_string_mapping(value, name=name))

        if dict(self.record_field_bindings) != _RECORD_FIELD_BINDINGS:
            raise ValueError("record field bindings must equal the frozen input mapping")
        binding_maps = (
            self.record_field_bindings,
            self.legacy_step_field_bindings,
            self.rdkit_reference_bindings,
        )
        all_fields = [field for bindings in binding_maps for field in bindings]
        if len(all_fields) != len(set(all_fields)):
            raise ValueError("schema field bindings must use globally unique source fields")
        node_specs = self.schema.nodes_by_id
        for bindings in binding_maps:
            if any(type(target) is not str or target not in node_specs for target in bindings.values()):
                raise ValueError("schema field bindings must target known node IDs")
        if any(
            node_specs[target].visibility is not Visibility.CANDIDATE_OUTPUT
            for target in self.legacy_step_field_bindings.values()
        ):
            raise ValueError("legacy step fields must bind to candidate-output nodes")
        if any(
            node_specs[target].visibility is not Visibility.BUILD_ONLY
            for target in self.rdkit_reference_bindings.values()
        ):
            raise ValueError("RDKit reference fields must bind to BUILD_ONLY nodes")

        grouped_nodes: set[str] = set()
        for semantic_id, members in self.semantic_state_groups.items():
            if semantic_id in node_specs:
                raise ValueError("semantic state group IDs must be distinct from node IDs")
            if type(members) is not tuple or len(members) < 2:
                raise ValueError("semantic state groups must contain at least two node IDs")
            if any(type(member) is not str or not member for member in members):
                raise TypeError("semantic state group members must be non-empty strings")
            if len(members) != len(set(members)):
                raise ValueError("semantic state group members must be unique")
            if any(member not in node_specs for member in members):
                raise ValueError("semantic state groups must reference known node IDs")
            if grouped_nodes.intersection(members):
                raise ValueError("schema nodes cannot belong to multiple semantic state groups")
            grouped_nodes.update(members)
            signatures = {
                (
                    node_specs[member].value_type,
                    node_specs[member].role,
                    node_specs[member].visibility,
                    node_specs[member].mutable,
                    node_specs[member].comparator,
                )
                for member in members
            }
            if len(signatures) != 1:
                raise ValueError("semantic state group nodes must have compatible specs")
            legacy_targets = set(self.legacy_step_field_bindings.values())
            if not set(members) <= legacy_targets:
                raise ValueError("semantic state group nodes must have legacy field bindings")

        record_visibility = {
            field: node_specs[target].visibility
            for field, target in self.record_field_bindings.items()
        }
        expected_visibility = {
            "indexed_smiles": Visibility.PROMPT_PREFIX,
            "instruction": Visibility.PROMPT_PREFIX,
            "gt_smiles": Visibility.BUILD_ONLY,
            "answer_smiles": Visibility.CANDIDATE_OUTPUT,
        }
        if record_visibility != expected_visibility:
            raise ValueError("record field bindings have invalid visibility")

        bound_nodes = {
            target for bindings in binding_maps for target in bindings.values()
        }
        if bound_nodes != set(node_specs):
            missing = tuple(sorted(set(node_specs) - bound_nodes))
            raise ValueError(f"every schema node must have a source binding; missing={missing}")
        candidate_slots = tuple(
            node.renderer_slot
            for node in self.schema.nodes
            if node.visibility is Visibility.CANDIDATE_OUTPUT
        )
        if any(slot is None for slot in candidate_slots):
            raise ValueError("candidate-output nodes must declare renderer slots")
        rendered_slots = tuple(
            node.renderer_slot
            for node in self.schema.nodes
            if node.renderer_slot is not None
        ) + tuple(
            edge.renderer_slot
            for edge in self.schema.edges
            if edge.renderer_slot is not None
        )
        if len(rendered_slots) != len(set(rendered_slots)):
            raise ValueError("renderer slots must be unique within an editing schema")
        if any(edge.mutable != (edge.renderer_slot is not None) for edge in self.schema.edges):
            raise ValueError("editing schema edges must be mutable exactly when rendered")

    def node_id_for_field(self, field_name: str) -> str:
        if type(field_name) is not str:
            raise TypeError("field_name must be a string")
        for bindings in (
            self.record_field_bindings,
            self.legacy_step_field_bindings,
            self.rdkit_reference_bindings,
        ):
            if field_name in bindings:
                return bindings[field_name]
        raise KeyError(field_name)

    def legacy_fields_for_node(self, node_id: str) -> tuple[str, ...]:
        if type(node_id) is not str:
            raise TypeError("node_id must be a string")
        if node_id not in self.schema.nodes_by_id:
            raise KeyError(node_id)
        return tuple(
            sorted(
                field
                for field, target in self.legacy_step_field_bindings.items()
                if target == node_id
            )
        )

    def semantic_state_for_node(self, node_id: str) -> str:
        if type(node_id) is not str:
            raise TypeError("node_id must be a string")
        if node_id not in self.schema.nodes_by_id:
            raise KeyError(node_id)
        for semantic_id, members in self.semantic_state_groups.items():
            if node_id in members:
                return semantic_id
        return node_id

    def legacy_fields_for_semantic_state(self, semantic_id: str) -> tuple[str, ...]:
        if type(semantic_id) is not str:
            raise TypeError("semantic_id must be a string")
        if semantic_id in self.semantic_state_groups:
            members = set(self.semantic_state_groups[semantic_id])
        elif semantic_id in self.schema.nodes_by_id:
            members = {semantic_id}
        else:
            raise KeyError(semantic_id)
        return tuple(
            sorted(
                field
                for field, target in self.legacy_step_field_bindings.items()
                if target in members
            )
        )

    @property
    def multi_mention_legacy_fields(self) -> FrozenMap[str, tuple[str, ...]]:
        semantic_ids = tuple(self.semantic_state_groups) + tuple(
            node.node_id
            for node in self.schema.nodes
            if node.node_id not in {
                member
                for members in self.semantic_state_groups.values()
                for member in members
            }
        )
        groups = {
            semantic_id: fields
            for semantic_id in semantic_ids
            if len(
                fields := self.legacy_fields_for_semantic_state(semantic_id)
            )
            > 1
        }
        return FrozenMap(groups)


_ADDITION_NODES = (
    *_evidence_nodes(),
    *_anchor_nodes(),
    _node(
        "leaving",
        ValueType.STRING,
        1,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.CUSTOM,
        "remove_group.smiles",
    ),
    _node(
        "add_fragment",
        ValueType.FRAGMENT,
        2,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
        "add_fragment.smiles",
    ),
    _node(
        "fragment_heavy",
        ValueType.COUNT,
        2,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.INTEGER_EQUAL,
        "add_heavy",
    ),
    _node(
        "product",
        ValueType.SMILES,
        3,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
        "product.smiles",
    ),
    *_count_nodes(heavy_step=4, ring_step=5),
    _terminal_node(),
    *_common_oracle_nodes(),
    _node(
        "oracle_fragment_heavy",
        ValueType.COUNT,
        None,
        NodeRole.INTERNAL_TRUTH,
        Visibility.BUILD_ONLY,
        False,
        ComparatorKind.INTEGER_EQUAL,
        None,
    ),
)

_ADDITION_EDGES = (
    *_anchor_edges(),
    _edge(
        "instruction_to_leaving",
        "instruction",
        "leaving",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "source_to_leaving",
        "source",
        "leaving",
        DependencyType.DERIVED_FROM,
    ),
    _edge(
        "instruction_to_add_fragment",
        "instruction",
        "add_fragment",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "add_fragment_to_heavy",
        "add_fragment",
        "fragment_heavy",
        DependencyType.COUNT_OF,
    ),
    _edge(
        "source_to_product",
        "source",
        "product",
        DependencyType.EDIT_PRODUCES,
        mutable=True,
        renderer_slot="relation.edit_produces",
    ),
    _edge(
        "anchor_to_product",
        "anchor_idx",
        "product",
        DependencyType.ATTACHED_TO,
        mutable=True,
        renderer_slot="relation.anchor_product",
    ),
    _edge(
        "leaving_to_product",
        "leaving",
        "product",
        DependencyType.REMOVED_FROM,
        mutable=True,
        renderer_slot="relation.leaving_product",
    ),
    _edge(
        "fragment_to_product",
        "add_fragment",
        "product",
        DependencyType.ATTACHED_TO,
        mutable=True,
        renderer_slot="relation.add_fragment_product",
    ),
    *_count_edges(),
)

ADDITION_STATE_SCHEMA = StateSchema(
    schema_id="mol_edit.add",
    version="1.0",
    nodes=_ADDITION_NODES,
    edges=_ADDITION_EDGES,
)

ADDITION_EDITING_SCHEMA = EditingStateSchema(
    normalized_subtask=EditingSubtask.ADD,
    schema=ADDITION_STATE_SCHEMA,
    record_field_bindings=_RECORD_FIELD_BINDINGS,
    legacy_step_field_bindings={
        "step1_anchor_idx": "anchor_idx",
        "step1_anchor_element": "anchor_element",
        "step1_leaving_smiles": "leaving",
        "step2_frag_smiles": "add_fragment",
        "step2_heavy_atoms": "fragment_heavy",
        "step3_product_smiles": "product",
        "step4_n_heavy_src": "source_heavy",
        "step4_n_heavy_prod": "product_heavy",
        "step4_heavy_delta": "heavy_delta",
        "step5_n_rings_src": "source_rings",
        "step5_n_rings_prod": "product_rings",
        "step5_ring_delta": "ring_delta",
    },
    rdkit_reference_bindings={
        "rdkit_elem_at_anchor": "oracle_anchor_element",
        "rdkit_frag_heavy": "oracle_fragment_heavy",
        "rdkit_src_heavy": "oracle_source_heavy",
        "rdkit_prod_heavy": "oracle_product_heavy",
        "rdkit_src_rings": "oracle_source_rings",
        "rdkit_prod_rings": "oracle_product_rings",
    },
)


_DELETION_NODES = (
    *_evidence_nodes(),
    *_anchor_nodes(),
    _node(
        "remove_group_step1",
        ValueType.FRAGMENT,
        1,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
        "step1.remove_group.smiles",
    ),
    _node(
        "remove_group_step2",
        ValueType.FRAGMENT,
        2,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
        "step2.remove_group.smiles",
    ),
    _node(
        "remove_heavy",
        ValueType.COUNT,
        2,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.INTEGER_EQUAL,
        "remove_heavy",
    ),
    _node(
        "product",
        ValueType.SMILES,
        3,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
        "product.smiles",
    ),
    *_count_nodes(heavy_step=4, ring_step=5),
    _terminal_node(),
    *_common_oracle_nodes(),
    _node(
        "oracle_remove_heavy",
        ValueType.COUNT,
        None,
        NodeRole.INTERNAL_TRUTH,
        Visibility.BUILD_ONLY,
        False,
        ComparatorKind.INTEGER_EQUAL,
        None,
    ),
)

_DELETION_EDGES = (
    *_anchor_edges(),
    _edge(
        "source_to_remove_group_step1",
        "source",
        "remove_group_step1",
        DependencyType.DERIVED_FROM,
    ),
    _edge(
        "instruction_to_remove_group_step1",
        "instruction",
        "remove_group_step1",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "source_to_remove_group_step2",
        "source",
        "remove_group_step2",
        DependencyType.DERIVED_FROM,
    ),
    _edge(
        "instruction_to_remove_group_step2",
        "instruction",
        "remove_group_step2",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "remove_group_step1_to_step2",
        "remove_group_step1",
        "remove_group_step2",
        DependencyType.MUST_EQUAL,
        mutable=True,
        renderer_slot="relation.remove_group_consistency",
    ),
    _edge(
        "remove_group_step2_to_heavy",
        "remove_group_step2",
        "remove_heavy",
        DependencyType.COUNT_OF,
    ),
    _edge(
        "source_to_product",
        "source",
        "product",
        DependencyType.EDIT_PRODUCES,
        mutable=True,
        renderer_slot="relation.edit_produces",
    ),
    _edge("anchor_to_product", "anchor_idx", "product", DependencyType.DERIVED_FROM),
    _edge(
        "remove_group_step2_to_product",
        "remove_group_step2",
        "product",
        DependencyType.REMOVED_FROM,
        mutable=True,
        renderer_slot="relation.remove_group_product",
    ),
    *_count_edges(),
)

DELETION_STATE_SCHEMA = StateSchema(
    schema_id="mol_edit.delete",
    version="1.0",
    nodes=_DELETION_NODES,
    edges=_DELETION_EDGES,
)

DELETION_EDITING_SCHEMA = EditingStateSchema(
    normalized_subtask=EditingSubtask.DELETE,
    schema=DELETION_STATE_SCHEMA,
    record_field_bindings=_RECORD_FIELD_BINDINGS,
    legacy_step_field_bindings={
        "step1_anchor_idx": "anchor_idx",
        "step1_anchor_element": "anchor_element",
        "step1_remove_group": "remove_group_step1",
        "step2_remove_smiles": "remove_group_step2",
        "step2_heavy_atoms": "remove_heavy",
        "step3_product_smiles": "product",
        "step4_n_heavy_src": "source_heavy",
        "step4_n_heavy_prod": "product_heavy",
        "step4_heavy_delta": "heavy_delta",
        "step5_n_rings_src": "source_rings",
        "step5_n_rings_prod": "product_rings",
        "step5_ring_delta": "ring_delta",
    },
    rdkit_reference_bindings={
        "rdkit_elem_at_anchor": "oracle_anchor_element",
        "rdkit_group_heavy": "oracle_remove_heavy",
        "rdkit_src_heavy": "oracle_source_heavy",
        "rdkit_prod_heavy": "oracle_product_heavy",
        "rdkit_src_rings": "oracle_source_rings",
        "rdkit_prod_rings": "oracle_product_rings",
    },
    semantic_state_groups={
        "remove_group": ("remove_group_step1", "remove_group_step2"),
    },
)


_SUBSTITUTION_NODES = (
    *_evidence_nodes(),
    *_anchor_nodes(),
    _node(
        "remove_group",
        ValueType.FRAGMENT,
        1,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
        "remove_group.smiles",
    ),
    _node(
        "add_fragment",
        ValueType.FRAGMENT,
        1,
        NodeRole.PRIMARY_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
        "add_fragment.smiles",
    ),
    _node(
        "remove_heavy",
        ValueType.COUNT,
        2,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.INTEGER_EQUAL,
        "remove_heavy",
    ),
    _node(
        "add_heavy",
        ValueType.COUNT,
        3,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.INTEGER_EQUAL,
        "add_heavy",
    ),
    _node(
        "product",
        ValueType.SMILES,
        4,
        NodeRole.DERIVED_CLAIM,
        Visibility.CANDIDATE_OUTPUT,
        True,
        ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
        "product.smiles",
    ),
    *_count_nodes(heavy_step=5, ring_step=6),
    _terminal_node(),
    *_common_oracle_nodes(),
    _node(
        "oracle_remove_heavy",
        ValueType.COUNT,
        None,
        NodeRole.INTERNAL_TRUTH,
        Visibility.BUILD_ONLY,
        False,
        ComparatorKind.INTEGER_EQUAL,
        None,
    ),
    _node(
        "oracle_add_heavy",
        ValueType.COUNT,
        None,
        NodeRole.INTERNAL_TRUTH,
        Visibility.BUILD_ONLY,
        False,
        ComparatorKind.INTEGER_EQUAL,
        None,
    ),
)

_SUBSTITUTION_EDGES = (
    *_anchor_edges(),
    _edge(
        "source_to_remove_group",
        "source",
        "remove_group",
        DependencyType.DERIVED_FROM,
    ),
    _edge(
        "instruction_to_remove_group",
        "instruction",
        "remove_group",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "instruction_to_add_fragment",
        "instruction",
        "add_fragment",
        DependencyType.CONSTRAINED_BY_INSTRUCTION,
    ),
    _edge(
        "source_to_add_fragment",
        "source",
        "add_fragment",
        DependencyType.DERIVED_FROM,
    ),
    _edge(
        "remove_group_to_heavy",
        "remove_group",
        "remove_heavy",
        DependencyType.COUNT_OF,
    ),
    _edge(
        "add_fragment_to_heavy",
        "add_fragment",
        "add_heavy",
        DependencyType.COUNT_OF,
    ),
    _edge(
        "source_to_product",
        "source",
        "product",
        DependencyType.EDIT_PRODUCES,
        mutable=True,
        renderer_slot="relation.edit_produces",
    ),
    _edge(
        "anchor_to_product",
        "anchor_idx",
        "product",
        DependencyType.ATTACHED_TO,
        mutable=True,
        renderer_slot="relation.anchor_product",
    ),
    _edge(
        "remove_group_to_product",
        "remove_group",
        "product",
        DependencyType.REMOVED_FROM,
        mutable=True,
        renderer_slot="relation.remove_group_product",
    ),
    _edge(
        "add_fragment_to_product",
        "add_fragment",
        "product",
        DependencyType.ATTACHED_TO,
        mutable=True,
        renderer_slot="relation.add_fragment_product",
    ),
    _edge(
        "remove_heavy_to_delta",
        "remove_heavy",
        "heavy_delta",
        DependencyType.DELTA_OF,
    ),
    _edge(
        "add_heavy_to_delta",
        "add_heavy",
        "heavy_delta",
        DependencyType.DELTA_OF,
    ),
    *_count_edges(),
)

SUBSTITUTION_STATE_SCHEMA = StateSchema(
    schema_id="mol_edit.substitute",
    version="1.0",
    nodes=_SUBSTITUTION_NODES,
    edges=_SUBSTITUTION_EDGES,
)

SUBSTITUTION_EDITING_SCHEMA = EditingStateSchema(
    normalized_subtask=EditingSubtask.SUBSTITUTE,
    schema=SUBSTITUTION_STATE_SCHEMA,
    record_field_bindings=_RECORD_FIELD_BINDINGS,
    legacy_step_field_bindings={
        "step1_anchor_idx": "anchor_idx",
        "step1_anchor_element": "anchor_element",
        "step1_remove_group_smiles": "remove_group",
        "step1_add_fragment_smiles": "add_fragment",
        "step2_remove_heavy": "remove_heavy",
        "step3_add_heavy": "add_heavy",
        "step4_product_smiles": "product",
        "step5_n_heavy_src": "source_heavy",
        "step5_n_heavy_prod": "product_heavy",
        "step5_heavy_delta": "heavy_delta",
        "step6_n_rings_src": "source_rings",
        "step6_n_rings_prod": "product_rings",
        "step6_ring_delta": "ring_delta",
    },
    rdkit_reference_bindings={
        "rdkit_elem_at_anchor": "oracle_anchor_element",
        "rdkit_remove_heavy": "oracle_remove_heavy",
        "rdkit_add_heavy": "oracle_add_heavy",
        "rdkit_src_heavy": "oracle_source_heavy",
        "rdkit_prod_heavy": "oracle_product_heavy",
        "rdkit_src_rings": "oracle_source_rings",
        "rdkit_prod_rings": "oracle_product_rings",
    },
)


EDITING_SCHEMA_DEFINITIONS: Mapping[EditingSubtask, EditingStateSchema] = _EditingSubtaskRegistry(
    {
        EditingSubtask.ADD: ADDITION_EDITING_SCHEMA,
        EditingSubtask.DELETE: DELETION_EDITING_SCHEMA,
        EditingSubtask.SUBSTITUTE: SUBSTITUTION_EDITING_SCHEMA,
    }
)

EDITING_STATE_SCHEMAS: Mapping[EditingSubtask, StateSchema] = _EditingSubtaskRegistry(
    {
        subtask: definition.schema
        for subtask, definition in EDITING_SCHEMA_DEFINITIONS.items()
    }
)


def editing_schema_for(subtask: EditingSubtask) -> EditingStateSchema:
    if type(subtask) is not EditingSubtask:
        raise TypeError("subtask must be an EditingSubtask")
    return EDITING_SCHEMA_DEFINITIONS[subtask]


def state_schema_for(subtask: EditingSubtask) -> StateSchema:
    return editing_schema_for(subtask).schema
