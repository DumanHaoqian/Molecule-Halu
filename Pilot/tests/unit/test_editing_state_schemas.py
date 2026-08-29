"""Tests for the three concrete molecule-editing StateSchemas."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from molhallulens.adapters import DEFAULT_SUBTASK_NORMALIZER
from molhallulens.domain import (
    ADDITION_EDITING_SCHEMA,
    DELETION_EDITING_SCHEMA,
    EDITING_SCHEMA_DEFINITIONS,
    EDITING_STATE_SCHEMAS,
    SUBSTITUTION_EDITING_SCHEMA,
    ClaimValue,
    ComparatorKind,
    DependencyType,
    EditingStateSchema,
    EditingSubtask,
    NodeRole,
    StateDAG,
    StateSchema,
    ValueProvenance,
    ValueType,
    Visibility,
    editing_schema_for,
    state_schema_for,
)


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "state_schemas"


def _serialize(definition: EditingStateSchema) -> dict[str, object]:
    schema = definition.schema
    return {
        "normalized_subtask": definition.normalized_subtask.value,
        "schema_id": schema.schema_id,
        "version": schema.version,
        "nodes": [
            {
                "node_id": node.node_id,
                "value_type": node.value_type.value,
                "step_index": node.step_index,
                "role": node.role.value,
                "visibility": node.visibility.value,
                "mutable": node.mutable,
                "comparator": node.comparator.value,
                "renderer_slot": node.renderer_slot,
            }
            for node in schema.nodes
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation.value,
                "mutable": edge.mutable,
                "renderer_slot": edge.renderer_slot,
            }
            for edge in schema.edges
        ],
        "record_field_bindings": dict(definition.record_field_bindings),
        "legacy_step_field_bindings": dict(definition.legacy_step_field_bindings),
        "rdkit_reference_bindings": dict(definition.rdkit_reference_bindings),
        "semantic_state_groups": {
            semantic_id: list(node_ids)
            for semantic_id, node_ids in definition.semantic_state_groups.items()
        },
        "multi_mention_legacy_fields": {
            node_id: list(fields)
            for node_id, fields in definition.multi_mention_legacy_fields.items()
        },
    }


def _ancestors(schema: StateSchema, target: str) -> set[str]:
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in schema.edges:
        incoming[edge.target].add(edge.source)
    result: set[str] = set()
    frontier = list(incoming[target])
    while frontier:
        node_id = frontier.pop()
        if node_id in result:
            continue
        result.add(node_id)
        frontier.extend(incoming[node_id])
    return result


def _dummy_value(value_type: ValueType) -> object:
    if value_type is ValueType.ATOM_INDEX:
        return 1
    if value_type is ValueType.COUNT:
        return 1
    if value_type is ValueType.INTEGER:
        return 0
    if value_type is ValueType.BOOLEAN:
        return True
    if value_type is ValueType.INDEXED_SMILES:
        return "[CH4:1]"
    return "C"


def test_registry_is_complete_typed_immutable_and_table_driven() -> None:
    assert tuple(EDITING_SCHEMA_DEFINITIONS) == tuple(EditingSubtask)
    assert tuple(EDITING_STATE_SCHEMAS) == tuple(EditingSubtask)
    for subtask in EditingSubtask:
        definition = editing_schema_for(subtask)
        assert type(definition) is EditingStateSchema
        assert definition.normalized_subtask is subtask
        assert state_schema_for(subtask) is definition.schema
        assert EDITING_STATE_SCHEMAS[subtask] is definition.schema

    with pytest.raises(TypeError):
        editing_schema_for("add")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EDITING_SCHEMA_DEFINITIONS["add"]  # type: ignore[index]
    with pytest.raises(TypeError):
        EDITING_STATE_SCHEMAS["add"]  # type: ignore[index]
    with pytest.raises(TypeError):
        EDITING_SCHEMA_DEFINITIONS[EditingSubtask.ADD] = DELETION_EDITING_SCHEMA  # type: ignore[index]
    with pytest.raises(TypeError):
        EDITING_SCHEMA_DEFINITIONS._data = {}  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        ADDITION_EDITING_SCHEMA.schema = DELETION_EDITING_SCHEMA.schema  # type: ignore[misc]


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
def test_schema_matches_frozen_golden_fixture(subtask: EditingSubtask) -> None:
    expected = json.loads((FIXTURE_ROOT / f"{subtask.value}.json").read_text())

    assert _serialize(editing_schema_for(subtask)) == expected


def test_real_formal_templates_are_mapped_exactly_without_verifier_leakage() -> None:
    total_step_fields = 0
    for path in sorted((DATASET_ROOT / "formal_templates" / "mol_edit").glob("*.json")):
        template = json.loads(path.read_text())
        normalized = DEFAULT_SUBTASK_NORMALIZER.normalize(template["subtask"])
        definition = editing_schema_for(normalized.normalized_subtask)

        assert set(definition.legacy_step_field_bindings) == set(template["step_fields"])
        assert set(definition.rdkit_reference_bindings) == set(
            template["rdkit_reference_fields"]
        )
        assert set(template["verifier_fields"]).isdisjoint(
            definition.legacy_step_field_bindings
        )
        assert set(template["verifier_fields"]).isdisjoint(
            definition.rdkit_reference_bindings
        )
        total_step_fields += len(template["step_fields"])

    assert total_step_fields == 37


def test_all_150_legacy_states_fit_the_bound_node_types() -> None:
    counts: Counter[EditingSubtask] = Counter()
    for path in sorted(
        (DATASET_ROOT / "process_evaluation_data" / "mol_edit").glob("*.json")
    ):
        records = json.loads(path.read_text())
        normalized = DEFAULT_SUBTASK_NORMALIZER.normalize(records[0]["subtask"])
        definition = editing_schema_for(normalized.normalized_subtask)
        nodes = definition.schema.nodes_by_id
        for record in records:
            state = record["parsed_reference_state"]
            assert set(state) == (
                set(definition.legacy_step_field_bindings)
                | set(definition.rdkit_reference_bindings)
            )
            for field, value in state.items():
                spec = nodes[definition.node_id_for_field(field)]
                claim = ClaimValue(
                    raw_value=value,
                    normalized_value=value,
                    value_type=spec.value_type,
                    provenance=ValueProvenance.REFERENCE,
                )
                assert claim.value_type is spec.value_type
            counts[normalized.normalized_subtask] += 1

    assert counts == Counter(
        {
            EditingSubtask.ADD: 50,
            EditingSubtask.DELETE: 50,
            EditingSubtask.SUBSTITUTE: 50,
        }
    )


@pytest.mark.parametrize("definition", tuple(EDITING_SCHEMA_DEFINITIONS.values()))
def test_visibility_comparator_and_renderer_contracts(
    definition: EditingStateSchema,
) -> None:
    nodes = definition.schema.nodes_by_id
    assert nodes["source"].visibility is Visibility.PROMPT_PREFIX
    assert nodes["instruction"].visibility is Visibility.PROMPT_PREFIX
    assert nodes["oracle_gt"].visibility is Visibility.BUILD_ONLY
    assert nodes["oracle_gt"].comparator is ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE
    assert nodes["product"].comparator is ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE
    assert nodes["final_answer"].comparator is ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE
    assert nodes["product"] is not nodes["final_answer"]
    assert nodes["heavy_delta"].value_type is ValueType.INTEGER
    assert nodes["ring_delta"].value_type is ValueType.INTEGER
    for node_id in ("source_heavy", "product_heavy", "source_rings", "product_rings"):
        assert nodes[node_id].value_type is ValueType.COUNT
    for node in definition.schema.nodes:
        if node.visibility is Visibility.BUILD_ONLY:
            assert node.role is NodeRole.INTERNAL_TRUTH
            assert node.mutable is False
            assert node.renderer_slot is None
        elif node.visibility is Visibility.CANDIDATE_OUTPUT:
            assert node.mutable is True
            assert node.renderer_slot is not None
    rendered_edges = tuple(edge for edge in definition.schema.edges if edge.renderer_slot)
    assert rendered_edges
    assert all(edge.mutable for edge in rendered_edges)
    assert DependencyType.EDIT_PRODUCES in {edge.relation for edge in rendered_edges}


def test_fragment_and_none_sentinel_comparators_are_semantically_explicit() -> None:
    addition = ADDITION_EDITING_SCHEMA.schema.nodes_by_id
    deletion = DELETION_EDITING_SCHEMA.schema.nodes_by_id
    substitution = SUBSTITUTION_EDITING_SCHEMA.schema.nodes_by_id

    assert addition["leaving"].value_type is ValueType.STRING
    assert addition["leaving"].comparator is ComparatorKind.CUSTOM
    for node in (
        addition["add_fragment"],
        deletion["remove_group_step1"],
        deletion["remove_group_step2"],
        substitution["remove_group"],
        substitution["add_fragment"],
    ):
        assert node.value_type is ValueType.FRAGMENT
        assert node.comparator is ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE

    assert addition["fragment_heavy"].renderer_slot == "add_heavy"
    assert substitution["add_heavy"].renderer_slot == "add_heavy"


@pytest.mark.parametrize(
    ("definition", "required_product_ancestors"),
    (
        (
            ADDITION_EDITING_SCHEMA,
            {"source", "instruction", "anchor_idx", "leaving", "add_fragment"},
        ),
        (
            DELETION_EDITING_SCHEMA,
            {
                "source",
                "instruction",
                "anchor_idx",
                "remove_group_step1",
                "remove_group_step2",
            },
        ),
        (
            SUBSTITUTION_EDITING_SCHEMA,
            {"source", "instruction", "anchor_idx", "remove_group", "add_fragment"},
        ),
    ),
)
def test_key_dependencies_are_present_not_merely_acyclic(
    definition: EditingStateSchema,
    required_product_ancestors: set[str],
) -> None:
    schema = definition.schema
    incoming = Counter(edge.target for edge in schema.edges)
    assert required_product_ancestors <= _ancestors(schema, "product")
    expected_heavy_inputs = {
        "source_heavy",
        "product_heavy",
    }
    if definition.normalized_subtask is EditingSubtask.SUBSTITUTE:
        expected_heavy_inputs |= {"remove_heavy", "add_heavy"}
    assert {
        edge.source for edge in schema.edges if edge.target == "heavy_delta"
    } == expected_heavy_inputs
    assert {edge.source for edge in schema.edges if edge.target == "ring_delta"} == {
        "source_rings",
        "product_rings",
    }
    assert any(
        edge.source == "product"
        and edge.target == "final_answer"
        and edge.relation is DependencyType.MOLECULARLY_EQUIVALENT_TO
        for edge in schema.edges
    )
    for node in schema.nodes:
        if node.role in {NodeRole.DERIVED_CLAIM, NodeRole.FINAL_ANSWER}:
            assert incoming[node.node_id] > 0
    assert all(
        not edge.source.startswith("oracle_") and not edge.target.startswith("oracle_")
        for edge in schema.edges
    )


@pytest.mark.parametrize(
    ("definition", "target"),
    (
        (ADDITION_EDITING_SCHEMA, "leaving"),
        (DELETION_EDITING_SCHEMA, "remove_group_step1"),
        (DELETION_EDITING_SCHEMA, "remove_group_step2"),
        (SUBSTITUTION_EDITING_SCHEMA, "remove_group"),
        (SUBSTITUTION_EDITING_SCHEMA, "add_fragment"),
    ),
)
def test_source_and_instruction_jointly_ground_edit_claims(
    definition: EditingStateSchema,
    target: str,
) -> None:
    sources = {edge.source for edge in definition.schema.edges if edge.target == target}
    assert {"source", "instruction"} <= sources


def test_deletion_remove_group_instances_share_one_semantic_state() -> None:
    definition = DELETION_EDITING_SCHEMA
    assert dict(definition.semantic_state_groups) == {
        "remove_group": ("remove_group_step1", "remove_group_step2")
    }
    assert definition.semantic_state_for_node("remove_group_step1") == "remove_group"
    assert definition.semantic_state_for_node("remove_group_step2") == "remove_group"
    assert definition.legacy_fields_for_semantic_state("remove_group") == (
        "step1_remove_group",
        "step2_remove_smiles",
    )
    assert dict(definition.multi_mention_legacy_fields) == {
        "remove_group": ("step1_remove_group", "step2_remove_smiles")
    }

    records = json.loads(
        (
            DATASET_ROOT
            / "process_evaluation_data"
            / "mol_edit"
            / "delete_pilot_origin.json"
        ).read_text()
    )
    assert all(
        record["parsed_reference_state"]["step1_remove_group"]
        == record["parsed_reference_state"]["step2_remove_smiles"]
        for record in records
    )

    values = {
        node.node_id: ClaimValue(
            raw_value=_dummy_value(node.value_type),
            normalized_value=_dummy_value(node.value_type),
            value_type=node.value_type,
            provenance=ValueProvenance.REFERENCE,
            mention_ids={
                "remove_group_step1": ("step1.natural",),
                "remove_group_step2": ("step2.formal",),
            }.get(node.node_id, ()),
        )
        for node in definition.schema.nodes
    }
    dag = StateDAG(schema=definition.schema, values=values)
    grouped_mentions = tuple(
        mention_id
        for node_id in definition.semantic_state_groups["remove_group"]
        for mention_id in dag.value_for(node_id).mention_ids
    )
    assert grouped_mentions == (
        "step1.natural",
        "step2.formal",
    )
    assert any(
        edge.source == "remove_group_step1"
        and edge.target == "remove_group_step2"
        and edge.relation is DependencyType.MUST_EQUAL
        and edge.mutable
        and edge.renderer_slot is not None
        for edge in definition.schema.edges
    )


def test_binding_contract_rejects_unknown_private_or_incomplete_targets() -> None:
    with pytest.raises(ValueError, match="schema_id"):
        replace(ADDITION_EDITING_SCHEMA, normalized_subtask=EditingSubtask.DELETE)
    with pytest.raises(ValueError, match="frozen input mapping"):
        replace(
            ADDITION_EDITING_SCHEMA,
            record_field_bindings={
                "indexed_smiles": "instruction",
                "instruction": "source",
                "gt_smiles": "oracle_gt",
                "answer_smiles": "final_answer",
            },
        )
    with pytest.raises(ValueError, match="known node"):
        replace(
            ADDITION_EDITING_SCHEMA,
            legacy_step_field_bindings={"step1_anchor_idx": "missing"},
        )
    with pytest.raises(ValueError, match="candidate-output"):
        replace(
            ADDITION_EDITING_SCHEMA,
            legacy_step_field_bindings={"step1_anchor_idx": "oracle_gt"},
        )
    with pytest.raises(ValueError, match="BUILD_ONLY"):
        replace(
            ADDITION_EDITING_SCHEMA,
            rdkit_reference_bindings={"rdkit_elem_at_anchor": "anchor_element"},
        )
    with pytest.raises(TypeError):
        ADDITION_EDITING_SCHEMA.legacy_step_field_bindings["new"] = "anchor_idx"  # type: ignore[index]
