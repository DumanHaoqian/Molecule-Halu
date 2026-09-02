"""T022 editing propagation, derivation, and GraphDelta contracts."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import build_reference_dag, derive_edit_truth
from molhallulens.infrastructure.chemistry import compute_descriptors, isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.core import (
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    CausalRole,
    ClaimValue,
    EditingSubtask,
    MutationTargetKind,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    ValueProvenance,
    ValueType,
    Visibility,
)
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    DeletionPerturbator,
    LabelProjector,
    PerturbationContext,
    PerturbatorRegistry,
    PropagationEngine,
    SubstitutionPerturbator,
    TraceRenderer,
    ValidatorChain,
    task_record_from_joined_input,
)
from molhallulens.modules.error_injection.operators.addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateEngine,
)
from molhallulens.modules.error_injection.operators.deletion import (
    DELETION_OPERATOR_IDS,
    DeletionCandidateEngine,
)
from molhallulens.modules.error_injection.operators.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
    SubstitutionCandidateEngine,
)
from molhallulens.modules.trajectory import (
    DEFAULT_EDITING_DERIVATION_RULE_REGISTRY,
    DerivationRuleRegistry,
    EditingPropagationEngine,
    PropagationError,
    TypedDerivationRule,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators


class _UnusedPropagation(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("candidate enumeration must not invoke propagation")


class _UnusedRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T022 tests do not render")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T022 tests do not execute the full template")

    def validate_artifact(self, draft):
        raise AssertionError("T022 tests do not validate rendered artifacts")


class _UnusedProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T022 tests do not project labels")


SUBTASK_CASES = {
    EditingSubtask.ADD: (
        AdditionPerturbator,
        AdditionCandidateEngine,
        ADDITION_OPERATOR_IDS,
    ),
    EditingSubtask.DELETE: (
        DeletionPerturbator,
        DeletionCandidateEngine,
        DELETION_OPERATOR_IDS,
    ),
    EditingSubtask.SUBSTITUTE: (
        SubstitutionPerturbator,
        SubstitutionCandidateEngine,
        SUBSTITUTION_OPERATOR_IDS,
    ),
}


@cache
def _records(subtask: EditingSubtask) -> tuple[JoinedInputRecord, ...]:
    marker = f".{subtask.value}_v2."
    return tuple(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if marker in record.anonymous_sample_id
        and record.anonymous_sample_id != "delete_v2.0081"
    )


@cache
def _reference(anonymous_sample_id: str):
    joined = next(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    return (
        joined,
        artifact,
        derive_edit_truth(artifact),
        task_record_from_joined_input(joined),
    )


@cache
def _registry(subtask: EditingSubtask) -> PerturbatorRegistry:
    perturbator_type = SUBTASK_CASES[subtask][0]
    return PerturbatorRegistry.from_perturbator_types(
        (perturbator_type,), operators_config=OPERATORS_CONFIG
    )


def _recipe(
    record,
    *,
    operator_id: str,
    policy: PropagationPolicy,
    target_node_id: str,
    partial_cut_nodes: frozenset[str] = frozenset(),
) -> PerturbationRecipe:
    return PerturbationRecipe(
        recipe_id=f"t022:{record.origin_id}:{operator_id}:{policy.dataset_name}",
        origin_id=record.origin_id,
        operator_id=operator_id,
        policy=policy,
        target_node_id=target_node_id,
        candidate_source_mode=CandidateSourceType.RULE,
        variant_index=0,
        derived_seed=20260829,
        rewrite_budget=RewriteBudget(
            max_changed_claims=64,
            max_added_characters=256,
            length_bucket="t022",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
        partial_cut_nodes=partial_cut_nodes,
    )


def _context(
    subtask: EditingSubtask,
    joined: JoinedInputRecord,
    *,
    operator_id: str,
    policy: PropagationPolicy,
    target_node_id: str,
    partial_cut_nodes: frozenset[str] = frozenset(),
) -> PerturbationContext:
    _, artifact, truth, record = _reference(joined.anonymous_sample_id)
    return PerturbationContext(
        record=record,
        recipe=_recipe(
            record,
            operator_id=operator_id,
            policy=policy,
            target_node_id=target_node_id,
            partial_cut_nodes=partial_cut_nodes,
        ),
        state_schema=artifact.state_dag.schema,
        reference_graph=artifact.state_dag,
        truth=truth,
    )


def _production_perturbator(subtask: EditingSubtask):
    perturbator_type, engine_type, _ = SUBTASK_CASES[subtask]
    candidate_engine = engine_type(operators_config=OPERATORS_CONFIG)
    return perturbator_type(
        candidate_engine=candidate_engine,
        propagator=_UnusedPropagation(),
        renderer=_UnusedRenderer(),
        validators=_UnusedValidators(),
        label_projector=_UnusedProjector(),
    )


def _first_candidate(
    subtask: EditingSubtask,
    *,
    operator_id: str,
    target_node_id: str,
    policy: PropagationPolicy,
    partial_cut_nodes: frozenset[str] = frozenset(),
) -> tuple[PerturbationContext, CandidatePatch]:
    for joined in _records(subtask):
        context = _context(
            subtask,
            joined,
            operator_id=operator_id,
            target_node_id=target_node_id,
            policy=policy,
            partial_cut_nodes=partial_cut_nodes,
        )
        pool: CandidatePool = _production_perturbator(
            subtask
        ).candidate_engine.enumerate_root_patches(context)
        matching = tuple(
            patch for patch in pool.candidates if patch.root_node_id == target_node_id
        )
        if matching:
            return context, matching[0]
    pytest.fail(f"no {subtask.value} candidate for {operator_id}/{target_node_id}")


def _changed_claim(old: ClaimValue) -> ClaimValue:
    value = old.normalized_value
    assert type(value) is int
    return ClaimValue(
        raw_value=value + 1,
        normalized_value=value + 1,
        value_type=old.value_type,
        provenance=ValueProvenance.RULE,
    )


def _numeric_stop_case(
    subtask: EditingSubtask,
) -> tuple[PerturbationContext, CandidatePatch]:
    operator_id = {
        EditingSubtask.ADD: ADDITION_OPERATOR_IDS[7],
        EditingSubtask.DELETE: DELETION_OPERATOR_IDS[9],
        EditingSubtask.SUBSTITUTE: SUBSTITUTION_OPERATOR_IDS[9],
    }[subtask]
    context = _context(
        subtask,
        _records(subtask)[0],
        operator_id=operator_id,
        policy=PropagationPolicy.STOP,
        target_node_id="product_heavy",
    )
    old = context.reference_graph.values["product_heavy"]
    return context, CandidatePatch(
        candidate_id=f"t022:{subtask.value}:stop",
        root_node_id="product_heavy",
        old_value=old,
        new_value=_changed_claim(old),
        edit_action=None,
        source=CandidateSourceType.RULE,
        metadata={"must_not_supply_semantic_axes": True},
    )


def _assert_event_contract(context: PerturbationContext, outcome) -> None:
    registration = _registry(context.record.normalized_subtask).registration(
        context.recipe.operator_id
    )
    root = outcome.graph_delta.root_events[0]
    for event in outcome.graph_delta.events:
        assert event.operator_id == registration.operator_id
        assert event.hallucination_types == registration.spec.hallucination_types
        assert event.edit_subtypes == registration.edit_subtypes
        assert event.before == context.reference_graph.values[event.node_or_edge_id]
        assert event.after == outcome.candidate_graph.values[event.node_or_edge_id]
        assert event.target_kind is MutationTargetKind.NODE
        assert event.root_event_id == root.event_id


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
def test_stop_changes_exactly_root_and_preserves_stale_edges(
    subtask: EditingSubtask,
) -> None:
    context, patch = _numeric_stop_case(subtask)
    engine = EditingPropagationEngine()
    plan = engine.plan(context, patch)
    outcome = engine.propagate(context, patch)

    assert plan.selected_nodes == (patch.root_node_id,)
    assert {event.node_or_edge_id for event in outcome.graph_delta.events} == {
        patch.root_node_id
    }
    assert outcome.graph_delta.root_events[0].causal_role is CausalRole.ROOT
    assert outcome.candidate_graph.edge_values == context.reference_graph.edge_values
    assert outcome.candidate_graph.semantic_differences(context.reference_graph) == {
        (MutationTargetKind.NODE, patch.root_node_id)
    }
    assert context.state_schema.stale_downstream_edges({patch.root_node_id})
    _assert_event_contract(context, outcome)


PARTIAL_CASES = (
    (
        EditingSubtask.ADD,
        ADDITION_OPERATOR_IDS[2],
        "add_fragment",
        frozenset({"product"}),
    ),
    (
        EditingSubtask.DELETE,
        DELETION_OPERATOR_IDS[6],
        "remove_group_step1",
        frozenset({"remove_group_step2"}),
    ),
    (
        EditingSubtask.SUBSTITUTE,
        SUBSTITUTION_OPERATOR_IDS[2],
        "add_fragment",
        frozenset({"product"}),
    ),
)


@pytest.mark.parametrize(("subtask", "operator_id", "root", "cuts"), PARTIAL_CASES)
def test_partial_cut_is_inclusive_connected_nontrivial_and_strict(
    subtask: EditingSubtask,
    operator_id: str,
    root: str,
    cuts: frozenset[str],
) -> None:
    context, patch = _first_candidate(
        subtask,
        operator_id=operator_id,
        target_node_id=root,
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=cuts,
    )
    engine = EditingPropagationEngine()
    plan = engine.plan(context, patch)
    outcome = engine.propagate(context, patch)

    assert plan.selected_nodes[0] == root
    assert cuts <= set(plan.selected_nodes)
    assert len(plan.selected_nodes) > 1
    assert set(plan.selected_nodes) < set(plan.full_closure)
    assert context.state_schema.is_connected_downstream_subgraph(
        {root}, plan.selected_nodes
    )
    allowed = set(plan.full_closure)
    expected = {root}
    frontier = [root]
    while frontier:
        node_id = frontier.pop()
        if node_id in cuts:
            continue
        for edge in context.state_schema.edges:
            if edge.source == node_id and edge.target in allowed - expected:
                expected.add(edge.target)
                frontier.append(edge.target)
    assert set(plan.selected_nodes) == expected
    assert all(
        context.state_schema.nodes_by_id[node_id].visibility
        is not Visibility.BUILD_ONLY
        for node_id in plan.selected_nodes
    )
    actual_differences = {
        node_id
        for kind, node_id in outcome.candidate_graph.semantic_differences(
            context.reference_graph
        )
        if kind is MutationTargetKind.NODE
    }
    assert actual_differences <= set(plan.selected_nodes)
    assert root in actual_differences
    assert all(
        event.causal_role
        in {
            CausalRole.ROOT,
            CausalRole.PROPAGATED_CONDITIONAL,
            CausalRole.PROPAGATED_FALSE,
        }
        for event in outcome.graph_delta.events
    )
    _assert_event_contract(context, outcome)


FULL_CASES = (
    (EditingSubtask.ADD, ADDITION_OPERATOR_IDS[3]),
    (EditingSubtask.DELETE, DELETION_OPERATOR_IDS[1]),
    (EditingSubtask.SUBSTITUTE, SUBSTITUTION_OPERATOR_IDS[3]),
)


@pytest.mark.parametrize(("subtask", "operator_id"), FULL_CASES)
def test_full_cf_recomputes_real_product_descendants_in_topological_order(
    subtask: EditingSubtask,
    operator_id: str,
) -> None:
    context, patch = _first_candidate(
        subtask,
        operator_id=operator_id,
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
    )
    engine = EditingPropagationEngine()
    plan = engine.plan(context, patch)
    first = engine.propagate(context, patch)
    second = engine.propagate(context, patch)

    expected_closure = ("product", *context.state_schema.descendants("product"))
    topo = context.state_schema.topological_order()
    expected_closure = tuple(node for node in topo if node in set(expected_closure))
    assert plan.full_closure == expected_closure
    assert plan.selected_nodes == expected_closure
    assert first == second

    candidate_smiles = first.candidate_graph.values["product"].normalized_value
    descriptors = compute_descriptors(candidate_smiles)
    values = first.candidate_graph.values
    assert values["product_heavy"].normalized_value == descriptors.heavy_atom_count
    assert values["product_rings"].normalized_value == descriptors.ring_count
    assert values["heavy_delta"].normalized_value == (
        descriptors.heavy_atom_count - values["source_heavy"].normalized_value
    )
    assert values["ring_delta"].normalized_value == (
        descriptors.ring_count - values["source_rings"].normalized_value
    )
    assert values["source_heavy"] == context.reference_graph.values["source_heavy"]
    assert values["source_rings"] == context.reference_graph.values["source_rings"]
    assert isomeric_graph_equivalent(
        values["final_answer"].normalized_value,
        candidate_smiles,
    )
    event_positions = [
        topo.index(event.node_or_edge_id) for event in first.graph_delta.events
    ]
    assert event_positions == sorted(event_positions)
    assert {event.node_or_edge_id for event in first.graph_delta.events} <= set(
        plan.selected_nodes
    )
    assert all(
        event.causal_role is CausalRole.PROPAGATED_CONDITIONAL
        for event in first.graph_delta.events[1:]
    )
    assert all(
        first.candidate_graph.values[node_id] == context.reference_graph.values[node_id]
        for node_id, spec in context.state_schema.nodes_by_id.items()
        if spec.visibility is Visibility.BUILD_ONLY
    )
    _assert_event_contract(context, first)


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
def test_terminal_changes_only_final_answer_and_preserves_reasoning_and_edges(
    subtask: EditingSubtask,
) -> None:
    operator_id = SUBTASK_CASES[subtask][2][-1]
    context = _context(
        subtask,
        _records(subtask)[0],
        operator_id=operator_id,
        policy=PropagationPolicy.TERMINAL,
        target_node_id="final_answer",
    )
    old = context.reference_graph.values["final_answer"]
    wrong_smiles = "C" if old.normalized_value != "C" else "CC"
    patch = CandidatePatch(
        candidate_id=f"t022:{subtask.value}:terminal",
        root_node_id="final_answer",
        old_value=old,
        new_value=ClaimValue(
            raw_value=wrong_smiles,
            normalized_value=wrong_smiles,
            value_type=ValueType.SMILES,
            provenance=ValueProvenance.RULE,
        ),
        edit_action=None,
        source=CandidateSourceType.RULE,
    )
    engine = EditingPropagationEngine()
    plan = engine.plan(context, patch)
    outcome = engine.propagate(context, patch)

    assert plan.selected_nodes == ("final_answer",)
    assert len(outcome.graph_delta.events) == 1
    assert outcome.graph_delta.events[0].causal_role is CausalRole.TERMINAL
    assert outcome.candidate_graph.edge_values == context.reference_graph.edge_values
    assert all(
        outcome.candidate_graph.values[node_id] == value
        for node_id, value in context.reference_graph.values.items()
        if node_id != "final_answer"
    )
    _assert_event_contract(context, outcome)


@pytest.mark.parametrize(
    ("cuts", "code"),
    (
        (frozenset({"unknown"}), "PARTIAL_CUT_UNKNOWN"),
        (frozenset({"add_fragment"}), "PARTIAL_CUT_NOT_DESCENDANT"),
        (frozenset({"instruction"}), "PARTIAL_CUT_NOT_DESCENDANT"),
    ),
)
def test_partial_rejects_unknown_root_and_upstream_cuts(
    cuts: frozenset[str], code: str
) -> None:
    valid_cuts = frozenset({"product"})
    context, patch = _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=valid_cuts,
    )
    bad_context = replace(
        context, recipe=replace(context.recipe, partial_cut_nodes=cuts)
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().plan(bad_context, patch)
    assert captured.value.code == code


def test_partial_rejects_empty_and_full_equivalent_selection() -> None:
    valid_cuts = frozenset({"product"})
    context, patch = _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=valid_cuts,
    )
    engine = EditingPropagationEngine()
    full_context = replace(
        context,
        recipe=replace(
            context.recipe,
            policy=PropagationPolicy.FULL_CF,
            partial_cut_nodes=frozenset(),
        ),
    )
    full = engine.plan(full_context, patch).full_closure

    empty_recipe = replace(context.recipe, partial_cut_nodes=valid_cuts)
    object.__setattr__(empty_recipe, "partial_cut_nodes", frozenset())
    with pytest.raises(PropagationError) as empty_error:
        engine.plan(replace(context, recipe=empty_recipe), patch)
    assert empty_error.value.code == "PARTIAL_NOT_STRICT"

    full_cut_context = replace(
        context,
        recipe=replace(context.recipe, partial_cut_nodes=frozenset(full[1:])),
    )
    with pytest.raises(PropagationError) as full_error:
        engine.plan(full_cut_context, patch)
    assert full_error.value.code == "PARTIAL_NOT_STRICT"


def test_partial_rejects_nested_cuts_when_a_downstream_cut_is_never_reached() -> None:
    context, patch = _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"product"}),
    )
    nested = replace(
        context,
        recipe=replace(
            context.recipe,
            partial_cut_nodes=frozenset({"product", "product_heavy"}),
        ),
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().plan(nested, patch)
    assert captured.value.code == "PARTIAL_NOT_STRICT"


def test_partial_defensively_rejects_a_disconnected_selected_subgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, patch = _first_candidate(
        EditingSubtask.DELETE,
        operator_id=DELETION_OPERATOR_IDS[6],
        target_node_id="remove_group_step1",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"remove_group_step2"}),
    )
    monkeypatch.setattr(
        type(context.state_schema),
        "is_connected_downstream_subgraph",
        lambda _schema, _roots, _selected: False,
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().plan(context, patch)
    assert captured.value.code == "PARTIAL_NOT_CONNECTED"


def _full_add_case() -> tuple[PerturbationContext, CandidatePatch]:
    return _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[3],
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
    )


def test_missing_rule_and_rule_signature_fail_closed() -> None:
    context, patch = _full_add_case()
    default_rules = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
    product_heavy_rule = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "product_heavy", schema_id=context.state_schema.schema_id
    )

    missing = DerivationRuleRegistry(
        tuple(rule for rule in default_rules if rule is not product_heavy_rule)
    )
    with pytest.raises(PropagationError) as missing_error:
        EditingPropagationEngine(rule_registry=missing).propagate(context, patch)
    assert missing_error.value.code == "DERIVATION_RULE_MISSING"

    wrong_input = replace(
        product_heavy_rule,
        rule_id=f"{product_heavy_rule.rule_id}.wrong-input",
        input_types=(ValueType.INTEGER,),
    )
    wrong_input_registry = DerivationRuleRegistry(
        tuple(
            wrong_input if rule is product_heavy_rule else rule
            for rule in default_rules
        )
    )
    with pytest.raises(PropagationError) as input_error:
        EditingPropagationEngine(rule_registry=wrong_input_registry).propagate(
            context, patch
        )
    assert input_error.value.code == "DERIVATION_INPUT_TYPE_MISMATCH"

    def wrong_type(_state, _derivation_context):
        return ClaimValue(
            raw_value="wrong",
            normalized_value="wrong",
            value_type=ValueType.STRING,
            provenance=ValueProvenance.PROPAGATED,
        )

    wrong_output = replace(
        product_heavy_rule,
        rule_id=f"{product_heavy_rule.rule_id}.wrong-output",
        derive_fn=wrong_type,
    )
    wrong_output_registry = DerivationRuleRegistry(
        tuple(
            wrong_output if rule is product_heavy_rule else rule
            for rule in default_rules
        )
    )
    with pytest.raises(PropagationError) as output_error:
        EditingPropagationEngine(rule_registry=wrong_output_registry).propagate(
            context, patch
        )
    assert output_error.value.code == "DERIVATION_OUTPUT_TYPE_MISMATCH"


def test_partial_requires_rules_beyond_the_cut_to_define_its_full_closure() -> None:
    context, patch = _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"product"}),
    )
    final_rule = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "final_answer", schema_id="mol_edit.add"
    )
    incomplete = DerivationRuleRegistry(
        tuple(
            rule
            for rule in DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
            if rule is not final_rule
        )
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine(rule_registry=incomplete).plan(context, patch)
    assert captured.value.code == "DERIVATION_RULE_MISSING"
    assert captured.value.node_id == "final_answer"


def test_substitution_heavy_delta_rule_declares_both_authoritative_identities() -> None:
    rule = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "heavy_delta", schema_id="mol_edit.substitute"
    )
    assert set(rule.input_nodes) == {
        "source_heavy",
        "product_heavy",
        "remove_heavy",
        "add_heavy",
    }


def test_derivation_exception_is_preserved_as_structured_failure() -> None:
    context, patch = _full_add_case()
    default_rules = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
    rule = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "product_heavy", schema_id=context.state_schema.schema_id
    )

    def explode(_state, _derivation_context):
        raise RuntimeError("sentinel derivation failure")

    broken = replace(rule, rule_id=f"{rule.rule_id}.explodes", derive_fn=explode)
    registry = DerivationRuleRegistry(
        tuple(broken if item is rule else item for item in default_rules)
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine(rule_registry=registry).propagate(context, patch)
    assert captured.value.code == "DERIVATION_FAILED"
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_partial_false_derivation_gets_false_role_and_full_rejects_conflict() -> None:
    partial_context, partial_patch = _first_candidate(
        EditingSubtask.DELETE,
        operator_id=DELETION_OPERATOR_IDS[6],
        target_node_id="remove_group_step1",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"remove_group_step2"}),
    )
    default_rules = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
    step2 = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "remove_group_step2", schema_id="mol_edit.delete"
    )

    def false_fragment(state, _derivation_context):
        before = state.values["remove_group_step2"]
        value = "N" if before.normalized_value != "N" else "O"
        return replace(
            before,
            raw_value=value,
            normalized_value=value,
            provenance=ValueProvenance.PROPAGATED,
        )

    false_step2 = replace(
        step2,
        rule_id=f"{step2.rule_id}.false",
        derive_fn=false_fragment,
    )
    false_registry = DerivationRuleRegistry(
        tuple(false_step2 if rule is step2 else rule for rule in default_rules)
    )
    partial = EditingPropagationEngine(rule_registry=false_registry).propagate(
        partial_context, partial_patch
    )
    propagated = next(
        event
        for event in partial.graph_delta.events
        if event.node_or_edge_id == "remove_group_step2"
    )
    assert propagated.causal_role is CausalRole.PROPAGATED_FALSE

    full_context = replace(
        partial_context,
        recipe=replace(
            partial_context.recipe,
            policy=PropagationPolicy.FULL_CF,
            partial_cut_nodes=frozenset(),
        ),
    )
    with pytest.raises(PropagationError) as conflict:
        EditingPropagationEngine(rule_registry=false_registry).propagate(
            full_context, partial_patch
        )
    assert conflict.value.code == "CROSS_FIELD_MISMATCH"
    assert conflict.value.evidence["reason"] == "relation_conflict"


def test_partial_rejects_semantic_delta_that_degenerates_to_stop() -> None:
    context, patch = _first_candidate(
        EditingSubtask.DELETE,
        operator_id=DELETION_OPERATOR_IDS[6],
        target_node_id="remove_group_step1",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"remove_group_step2"}),
    )
    default_rules = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
    step2 = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
        "remove_group_step2", schema_id="mol_edit.delete"
    )

    def unchanged(state, _derivation_context):
        return state.values["remove_group_step2"]

    unchanged_rule = replace(
        step2,
        rule_id=f"{step2.rule_id}.unchanged",
        derive_fn=unchanged,
    )
    registry = DerivationRuleRegistry(
        tuple(unchanged_rule if rule is step2 else rule for rule in default_rules)
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine(rule_registry=registry).propagate(context, patch)
    assert captured.value.code == "PARTIAL_NOT_NONTRIVIAL"


def test_derived_claims_preserve_reference_mention_ids() -> None:
    context, patch = _full_add_case()
    outcome = EditingPropagationEngine().propagate(context, patch)
    for node_id in EditingPropagationEngine().plan(context, patch).selected_nodes[1:]:
        assert outcome.candidate_graph.values[node_id].mention_ids == (
            context.reference_graph.values[node_id].mention_ids
        )


def test_substitution_full_checks_both_heavy_delta_identities() -> None:
    context, patch = _first_candidate(
        EditingSubtask.SUBSTITUTE,
        operator_id=SUBSTITUTION_OPERATOR_IDS[3],
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
    )
    values = dict(context.reference_graph.values)
    add_heavy = values["add_heavy"]
    values["add_heavy"] = replace(
        add_heavy,
        raw_value=add_heavy.normalized_value + 1,
        normalized_value=add_heavy.normalized_value + 1,
    )
    bad_reference = replace(context.reference_graph, values=values)
    bad_context = replace(context, reference_graph=bad_reference)
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().propagate(bad_context, patch)
    assert captured.value.code == "CROSS_FIELD_MISMATCH"
    assert captured.value.evidence["reason"] == "relation_conflict"


def test_invalid_root_patch_action_and_cross_field_values_fail_closed() -> None:
    context, patch = _full_add_case()
    engine = EditingPropagationEngine()

    mismatched_old = replace(
        patch,
        old_value=ClaimValue(
            raw_value="C",
            normalized_value="C",
            value_type=ValueType.SMILES,
            provenance=ValueProvenance.REFERENCE,
        ),
    )
    with pytest.raises(PropagationError) as old_error:
        engine.propagate(context, mismatched_old)
    assert old_error.value.code == "ROOT_PATCH_MISMATCH"

    with pytest.raises(PropagationError) as action_error:
        engine.propagate(context, replace(patch, edit_action=None))
    assert action_error.value.code == "STRUCTURAL_ACTION_REQUIRED"

    different_product = "C" if patch.new_value.normalized_value != "C" else "CC"
    wrong_product = replace(
        patch,
        new_value=replace(
            patch.new_value,
            raw_value=different_product,
            normalized_value=different_product,
        ),
    )
    with pytest.raises(PropagationError) as product_error:
        engine.propagate(context, wrong_product)
    assert product_error.value.code == "ACTION_PRODUCT_MISMATCH"

    wrong_root_context = replace(
        context,
        recipe=replace(context.recipe, target_node_id="anchor_idx"),
    )
    with pytest.raises(PropagationError) as root_error:
        engine.propagate(wrong_root_context, patch)
    assert root_error.value.code == "ROOT_PATCH_MISMATCH"


@pytest.mark.parametrize("invalid_smiles", ("C1(", "[C"))
def test_invalid_product_comparator_input_is_structured(invalid_smiles: str) -> None:
    context, patch = _full_add_case()
    invalid = replace(
        patch,
        new_value=replace(
            patch.new_value,
            raw_value=invalid_smiles,
            normalized_value=invalid_smiles,
        ),
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().propagate(context, invalid)
    assert captured.value.code in {"ACTION_PRODUCT_MISMATCH", "CROSS_FIELD_MISMATCH"}


def test_invalid_fragment_comparator_input_is_structured() -> None:
    context, patch = _first_candidate(
        EditingSubtask.ADD,
        operator_id=ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"product"}),
    )
    invalid = replace(
        patch,
        new_value=replace(
            patch.new_value,
            raw_value="C1(",
            normalized_value="C1(",
        ),
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine().propagate(context, invalid)
    assert captured.value.code in {"ACTION_PRODUCT_MISMATCH", "CROSS_FIELD_MISMATCH"}


def test_default_rules_never_target_build_only_state() -> None:
    for subtask in EditingSubtask:
        context, _ = _numeric_stop_case(subtask)
        schema_id = context.state_schema.schema_id
        outputs = {
            rule.output_node
            for rule in DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rules
            if not rule.schema_ids or schema_id in rule.schema_ids
        }
        assert all(
            context.state_schema.nodes_by_id[node_id].visibility
            is not Visibility.BUILD_ONLY
            for node_id in outputs
        )


def test_registry_rejects_build_only_derivation_output() -> None:
    rule = TypedDerivationRule(
        rule_id="t022.invalid.oracle",
        output_node="oracle_gt",
        input_nodes=("source",),
        input_types=(ValueType.INDEXED_SMILES,),
        output_type=ValueType.SMILES,
        derive_fn=lambda state, _context: state.values["oracle_gt"],
        schema_ids=frozenset({"mol_edit.add"}),
    )
    with pytest.raises(PropagationError) as captured:
        EditingPropagationEngine(rule_registry=DerivationRuleRegistry((rule,)))
    assert captured.value.code == "DERIVATION_RULE_MISSING"


def test_same_seed_and_inputs_produce_identical_plan_graph_and_delta() -> None:
    context, patch = _full_add_case()
    first_engine = EditingPropagationEngine()
    second_engine = EditingPropagationEngine()
    assert first_engine.plan(context, patch) == second_engine.plan(context, patch)
    assert first_engine.propagate(context, patch) == second_engine.propagate(
        context, patch
    )
