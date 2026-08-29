"""Property tests for T022 policy planning over deterministic random DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import build_reference_dag, derive_edit_truth
from molhallulens.domain import (
    CandidatePatch,
    CandidateSourceType,
    ClaimValue,
    ComparatorKind,
    DependencyType,
    MutationTargetKind,
    NodeRole,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    StateEdge,
    StateNodeSpec,
    StateSchema,
    ValueProvenance,
    ValueType,
    Visibility,
)
from molhallulens.perturbators import PerturbationContext, task_record_from_joined_input
from molhallulens.perturbators.editing.addition import ADDITION_OPERATOR_IDS
from molhallulens.propagation import (
    DerivationRuleRegistry,
    EditingPropagationEngine,
    TypedDerivationRule,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


@dataclass(frozen=True)
class _LayeredCase:
    widths: tuple[int, ...]
    cut_layer: int


@st.composite
def _layered_cases(draw) -> _LayeredCase:
    layer_count = draw(st.integers(min_value=2, max_value=5))
    widths = tuple(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=3),
                min_size=layer_count,
                max_size=layer_count,
            )
        )
    )
    cut_layer = draw(st.integers(min_value=0, max_value=layer_count - 2))
    return _LayeredCase(widths=widths, cut_layer=cut_layer)


@cache
def _real_add_inputs():
    joined = next(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".add_v2." in record.anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    return (
        task_record_from_joined_input(joined),
        derive_edit_truth(artifact),
    )


def _claim(value: int, *, provenance: ValueProvenance) -> ClaimValue:
    return ClaimValue(
        raw_value=value,
        normalized_value=value,
        value_type=ValueType.INTEGER,
        provenance=provenance,
    )


def _node(node_id: str, *, value_type: ValueType = ValueType.INTEGER) -> StateNodeSpec:
    return StateNodeSpec(
        node_id=node_id,
        value_type=value_type,
        step_index=0,
        role=NodeRole.DERIVED_CLAIM,
        visibility=Visibility.CANDIDATE_OUTPUT,
        mutable=True,
        comparator=ComparatorKind.INTEGER_EQUAL,
        renderer_slot=f"slot.{node_id}",
    )


def _layer_ids(case: _LayeredCase) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(f"n{layer:02d}_{index:02d}" for index in range(width))
        for layer, width in enumerate(case.widths)
    )


def _layered_state(
    case: _LayeredCase,
    *,
    root_node_id: str = "heavy_delta",
    root_type: ValueType = ValueType.INTEGER,
) -> tuple[StateDAG, DerivationRuleRegistry, tuple[tuple[str, ...], ...]]:
    layers = _layer_ids(case)
    nodes = (_node(root_node_id, value_type=root_type),) + tuple(
        _node(node_id) for layer in layers for node_id in layer
    )
    edge_specs: list[StateEdge] = []
    parents_by_node: dict[str, tuple[str, ...]] = {}
    previous = (root_node_id,)
    for layer_index, layer in enumerate(layers):
        for target in layer:
            parents_by_node[target] = previous
            for source in previous:
                edge_specs.append(
                    StateEdge(
                        edge_id=f"e:{source}:{target}",
                        source=source,
                        target=target,
                        relation=DependencyType.DERIVED_FROM,
                    )
                )
        previous = layer
    schema = StateSchema(
        schema_id="mol_edit.add",
        version="t022-property-v1",
        nodes=nodes,
        edges=tuple(reversed(edge_specs)),
    )

    values: dict[str, ClaimValue] = {
        root_node_id: ClaimValue(
            raw_value=1,
            normalized_value=1,
            value_type=root_type,
            provenance=ValueProvenance.REFERENCE,
        )
    }
    rules: list[TypedDerivationRule] = []
    for layer in layers:
        for node_id in layer:
            parents = parents_by_node[node_id]
            value = sum(int(values[parent].normalized_value) for parent in parents) + 1
            values[node_id] = _claim(value, provenance=ValueProvenance.REFERENCE)

            def derive(state, _context, *, inputs=parents):
                return _claim(
                    sum(int(state.values[parent].normalized_value) for parent in inputs)
                    + 1,
                    provenance=ValueProvenance.PROPAGATED,
                )

            rules.append(
                TypedDerivationRule(
                    rule_id=f"t022.derive.{node_id}",
                    output_node=node_id,
                    input_nodes=parents,
                    input_types=tuple(
                        schema.nodes_by_id[parent].value_type for parent in parents
                    ),
                    output_type=ValueType.INTEGER,
                    derive_fn=derive,
                    schema_ids=frozenset({schema.schema_id}),
                )
            )
    edge_values = {
        edge.edge_id: ClaimValue(
            raw_value=True,
            normalized_value=True,
            value_type=ValueType.BOOLEAN,
            provenance=ValueProvenance.REFERENCE,
            locally_valid=True,
        )
        for edge in edge_specs
    }
    return (
        StateDAG(schema=schema, values=values, edge_values=edge_values),
        DerivationRuleRegistry(tuple(reversed(rules))),
        layers,
    )


def _context(
    reference: StateDAG,
    *,
    operator_id: str,
    policy: PropagationPolicy,
    target_node_id: str,
    partial_cut_nodes: frozenset[str] = frozenset(),
) -> PerturbationContext:
    record, truth = _real_add_inputs()
    recipe = PerturbationRecipe(
        recipe_id=f"t022:property:{policy.dataset_name}:{target_node_id}",
        origin_id=record.origin_id,
        operator_id=operator_id,
        policy=policy,
        target_node_id=target_node_id,
        candidate_source_mode=CandidateSourceType.RULE,
        variant_index=0,
        derived_seed=7919,
        rewrite_budget=RewriteBudget(
            max_changed_claims=64,
            max_added_characters=64,
            length_bucket="property",
        ),
        candidate_difficulty_bucket="property",
        renderer_style_id="property",
        partial_cut_nodes=partial_cut_nodes,
    )
    return PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=reference.schema,
        reference_graph=reference,
        truth=truth,
    )


def _patch(context: PerturbationContext) -> CandidatePatch:
    old = context.reference_graph.values[context.recipe.target_node_id]
    assert type(old.normalized_value) is int
    return CandidatePatch(
        candidate_id="t022:property:root",
        root_node_id=context.recipe.target_node_id,
        old_value=old,
        new_value=ClaimValue(
            raw_value=old.normalized_value + 7,
            normalized_value=old.normalized_value + 7,
            value_type=old.value_type,
            provenance=ValueProvenance.RULE,
        ),
        edit_action=None,
        source=CandidateSourceType.RULE,
    )


@given(_layered_cases())
@settings(max_examples=40, deadline=None)
def test_partial_random_layered_dags_stop_at_inclusive_cut_and_are_deterministic(
    case: _LayeredCase,
) -> None:
    reference, registry, layers = _layered_state(case)
    cuts = frozenset(layers[case.cut_layer])
    context = _context(
        reference,
        operator_id=ADDITION_OPERATOR_IDS[7],
        policy=PropagationPolicy.PARTIAL,
        target_node_id="heavy_delta",
        partial_cut_nodes=cuts,
    )
    patch = _patch(context)
    engine = EditingPropagationEngine(rule_registry=registry)
    first_plan = engine.plan(context, patch)
    second_plan = engine.plan(context, patch)
    first = engine.propagate(context, patch)
    second = engine.propagate(context, patch)

    topo = reference.schema.topological_order()
    expected_selected_set = {
        "heavy_delta",
        *(node_id for layer in layers[: case.cut_layer + 1] for node_id in layer),
    }
    expected_selected = tuple(
        node_id for node_id in topo if node_id in expected_selected_set
    )
    assert first_plan == second_plan
    assert first_plan.selected_nodes == expected_selected
    assert set(first_plan.selected_nodes) < set(first_plan.full_closure)
    assert reference.schema.is_connected_downstream_subgraph(
        {"heavy_delta"}, first_plan.selected_nodes
    )
    assert first == second
    assert (
        tuple(event.node_or_edge_id for event in first.graph_delta.events)
        == expected_selected
    )
    assert first.candidate_graph.semantic_differences(reference) == frozenset(
        (MutationTargetKind.NODE, node_id) for node_id in expected_selected
    )
    assert all(
        first.candidate_graph.values[node_id] == reference.values[node_id]
        for layer in layers[case.cut_layer + 1 :]
        for node_id in layer
    )


@given(_layered_cases())
@settings(max_examples=40, deadline=None)
def test_full_plan_is_exact_stable_topological_derivable_closure(
    case: _LayeredCase,
) -> None:
    reference, registry, layers = _layered_state(
        case,
        root_node_id="anchor_idx",
        root_type=ValueType.ATOM_INDEX,
    )
    context = _context(
        reference,
        operator_id=ADDITION_OPERATOR_IDS[0],
        policy=PropagationPolicy.FULL_CF,
        target_node_id="anchor_idx",
    )
    patch = _patch(context)
    engine = EditingPropagationEngine(rule_registry=registry)
    plan = engine.plan(context, patch)
    all_nodes = {"anchor_idx", *(node for layer in layers for node in layer)}
    expected = tuple(
        node_id
        for node_id in reference.schema.topological_order()
        if node_id in all_nodes
    )
    assert plan.full_closure == expected
    assert plan.selected_nodes == expected
    assert engine.plan(context, patch) == plan


@given(_layered_cases())
@settings(max_examples=30, deadline=None)
def test_stop_random_dag_never_recomputes_descendants(case: _LayeredCase) -> None:
    reference, registry, _ = _layered_state(case)
    context = _context(
        reference,
        operator_id=ADDITION_OPERATOR_IDS[7],
        policy=PropagationPolicy.STOP,
        target_node_id="heavy_delta",
    )
    patch = _patch(context)
    engine = EditingPropagationEngine(rule_registry=registry)
    outcome = engine.propagate(context, patch)
    assert engine.plan(context, patch).selected_nodes == ("heavy_delta",)
    assert outcome.candidate_graph.semantic_differences(reference) == frozenset(
        {(MutationTargetKind.NODE, "heavy_delta")}
    )
    assert tuple(event.node_or_edge_id for event in outcome.graph_delta.events) == (
        "heavy_delta",
    )
