"""T023 deterministic FORMAL and Answer rendering contracts."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from functools import cache
from pathlib import Path

import pytest
from rdkit import Chem

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.builders import build_reference_dag, derive_edit_truth
from molhallulens.chemistry import isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.domain import (
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    ClaimValue,
    EditingSubtask,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    ValueProvenance,
    ValueType,
    Visibility,
    editing_schema_for,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    DeletionPerturbator,
    LabelProjector,
    PerturbationContext,
    PropagationEngine,
    SubstitutionPerturbator,
    TraceRenderer,
    ValidatorChain,
    task_record_from_joined_input,
)
from molhallulens.perturbators.editing.addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateEngine,
)
from molhallulens.perturbators.editing.deletion import (
    DELETION_OPERATOR_IDS,
    DeletionCandidateEngine,
)
from molhallulens.perturbators.editing.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
    SubstitutionCandidateEngine,
)
from molhallulens.propagation import EditingPropagationEngine
from molhallulens.rendering import (
    DeterministicAnswerRenderer,
    DeterministicFormalRenderer,
    FormalRenderError,
    FormalSlotValue,
    RenderedFormalTrace,
    parse_formal,
    render_answer,
    render_formal,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators


class _UnusedPropagation(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("candidate enumeration must not invoke propagation")


class _UnusedTraceRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T023 tests invoke deterministic renderers directly")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T023 tests do not execute the full template")

    def validate_artifact(self, draft):
        raise AssertionError("T023 tests do not validate a rendered draft")


class _UnusedProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T023 tests do not project labels")


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

EXPECTED_STEP_NAMES = {
    EditingSubtask.ADD: (
        "ANCHOR_IDENTIFICATION",
        "FRAGMENT_IDENTIFICATION",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ),
    EditingSubtask.DELETE: (
        "ANCHOR_IDENTIFICATION",
        "GROUP_SIZE_VERIFICATION",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ),
    EditingSubtask.SUBSTITUTE: (
        "ANCHOR_IDENTIFICATION",
        "REMOVE_GROUP_SIZE",
        "ADD_FRAGMENT_SIZE",
        "PRODUCT_CONSTRUCTION",
        "HEAVY_ATOM_VERIFICATION",
        "RING_VERIFICATION",
    ),
}


@cache
def _records() -> tuple[JoinedInputRecord, ...]:
    return ChemCoTMolEditAdapter().load(DATASET_ROOT)


def _subtask_records(subtask: EditingSubtask) -> tuple[JoinedInputRecord, ...]:
    marker = f".{subtask.value}_v2."
    return tuple(
        record
        for record in _records()
        if marker in record.anonymous_sample_id
        and record.anonymous_sample_id != "delete_v2.0081"
    )


@cache
def _reference(anonymous_sample_id: str):
    joined = next(
        record
        for record in _records()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    return (
        joined,
        artifact,
        derive_edit_truth(artifact),
        task_record_from_joined_input(joined),
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
        recipe_id=f"t023:{record.origin_id}:{operator_id}:{policy.dataset_name}",
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
            length_bucket="t023",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="formal-v1",
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
    perturbator_type, candidate_engine_type, _ = SUBTASK_CASES[subtask]
    return perturbator_type(
        candidate_engine=candidate_engine_type(operators_config=OPERATORS_CONFIG),
        propagator=_UnusedPropagation(),
        renderer=_UnusedTraceRenderer(),
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
    for joined in _subtask_records(subtask):
        context = _context(
            subtask,
            joined,
            operator_id=operator_id,
            policy=policy,
            target_node_id=target_node_id,
            partial_cut_nodes=partial_cut_nodes,
        )
        pool: CandidatePool = _production_perturbator(
            subtask
        ).candidate_engine.enumerate_root_patches(context)
        for patch in pool.candidates:
            if patch.root_node_id == target_node_id:
                return context, patch
    pytest.fail(f"no {subtask.value} candidate for {operator_id}/{target_node_id}")


def _changed_integer(old: ClaimValue) -> ClaimValue:
    assert type(old.normalized_value) is int
    return replace(
        old,
        raw_value=old.normalized_value + 1,
        normalized_value=old.normalized_value + 1,
        provenance=ValueProvenance.RULE,
    )


def _stop_state(subtask: EditingSubtask) -> StateDAG:
    operator_id = {
        EditingSubtask.ADD: ADDITION_OPERATOR_IDS[7],
        EditingSubtask.DELETE: DELETION_OPERATOR_IDS[9],
        EditingSubtask.SUBSTITUTE: SUBSTITUTION_OPERATOR_IDS[9],
    }[subtask]
    context = _context(
        subtask,
        _subtask_records(subtask)[0],
        operator_id=operator_id,
        policy=PropagationPolicy.STOP,
        target_node_id="product_heavy",
    )
    old = context.reference_graph.values["product_heavy"]
    patch = CandidatePatch(
        candidate_id=f"t023:{subtask.value}:stop",
        root_node_id="product_heavy",
        old_value=old,
        new_value=_changed_integer(old),
        edit_action=None,
        source=CandidateSourceType.RULE,
    )
    return EditingPropagationEngine().propagate(context, patch).candidate_graph


PARTIAL_CASES = {
    EditingSubtask.ADD: (
        ADDITION_OPERATOR_IDS[2],
        "add_fragment",
        frozenset({"product"}),
    ),
    EditingSubtask.DELETE: (
        DELETION_OPERATOR_IDS[6],
        "remove_group_step1",
        frozenset({"remove_group_step2"}),
    ),
    EditingSubtask.SUBSTITUTE: (
        SUBSTITUTION_OPERATOR_IDS[2],
        "add_fragment",
        frozenset({"product"}),
    ),
}


def _partial_state(subtask: EditingSubtask) -> StateDAG:
    operator_id, target, cuts = PARTIAL_CASES[subtask]
    context, patch = _first_candidate(
        subtask,
        operator_id=operator_id,
        target_node_id=target,
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=cuts,
    )
    return EditingPropagationEngine().propagate(context, patch).candidate_graph


FULL_CASES = {
    EditingSubtask.ADD: ADDITION_OPERATOR_IDS[3],
    EditingSubtask.DELETE: DELETION_OPERATOR_IDS[1],
    EditingSubtask.SUBSTITUTE: SUBSTITUTION_OPERATOR_IDS[3],
}


def _full_state(subtask: EditingSubtask) -> StateDAG:
    context, patch = _first_candidate(
        subtask,
        operator_id=FULL_CASES[subtask],
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
    )
    return EditingPropagationEngine().propagate(context, patch).candidate_graph


def _terminal_state(subtask: EditingSubtask, *, answer: str = "C") -> StateDAG:
    _, artifact, _, _ = _reference(_subtask_records(subtask)[0].anonymous_sample_id)
    values = dict(artifact.state_dag.values)
    old = values["final_answer"]
    selected = answer if answer != old.normalized_value else "CC"
    values["final_answer"] = replace(
        old,
        raw_value=selected,
        normalized_value=selected,
        provenance=ValueProvenance.RULE,
    )
    return StateDAG(artifact.state_dag.schema, values, artifact.state_dag.edge_values)


@cache
def _policy_state(subtask: EditingSubtask, policy: PropagationPolicy) -> StateDAG:
    return {
        PropagationPolicy.STOP: _stop_state,
        PropagationPolicy.PARTIAL: _partial_state,
        PropagationPolicy.FULL_CF: _full_state,
        PropagationPolicy.TERMINAL: _terminal_state,
    }[policy](subtask)


def _assert_typed_formal_projection(
    state: StateDAG, trace: RenderedFormalTrace
) -> None:
    parsed = parse_formal(trace)
    definition = editing_schema_for(parsed.normalized_subtask)
    expected_nodes = set(definition.legacy_step_field_bindings.values())
    assert set(parsed.values) == expected_nodes
    for node_id in expected_nodes:
        claim = state.values[node_id]
        assert type(parsed.values[node_id]) is type(claim.normalized_value)
        assert parsed.values[node_id] == claim.normalized_value
    DeterministicFormalRenderer().assert_round_trip(state, trace)


def test_all_150_reference_formal_traces_are_byte_exact_and_typed() -> None:
    renderer = DeterministicFormalRenderer()
    for record in _records():
        artifact = build_reference_dag(record)
        trace = renderer.render(artifact.state_dag)

        assert trace.normalized_subtask is artifact.normalized_subtask
        assert tuple(step.step_index for step in trace.steps) == tuple(
            range(1, len(trace.steps) + 1)
        )
        assert (
            tuple(step.step_name for step in trace.steps)
            == EXPECTED_STEP_NAMES[artifact.normalized_subtask]
        )
        assert tuple(step.formal_ab for step in trace.steps) == tuple(
            step.formal_ab for step in artifact.trace_steps
        )
        assert trace.formal_lines == tuple(step.formal_line for step in trace.steps)
        _assert_typed_formal_projection(artifact.state_dag, trace)


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
@pytest.mark.parametrize("policy", tuple(PropagationPolicy))
def test_real_t022_candidate_states_round_trip_for_every_subtask_and_policy(
    subtask: EditingSubtask,
    policy: PropagationPolicy,
) -> None:
    state = _policy_state(subtask, policy)
    first = render_formal(state)
    second = render_formal(
        StateDAG(
            state.schema, dict(reversed(tuple(state.values.items()))), state.edge_values
        )
    )
    assert first == second
    _assert_typed_formal_projection(state, first)

    recovered = dict(parse_formal(first).values)
    recovered["final_answer"] = render_answer(state, policy=policy).smiles
    candidate_output_nodes = {
        node.node_id
        for node in state.schema.nodes
        if node.visibility is Visibility.CANDIDATE_OUTPUT
    }
    assert set(recovered) == candidate_output_nodes
    assert recovered == {
        node_id: state.values[node_id].normalized_value
        for node_id in candidate_output_nodes
    }


def test_deletion_multi_mentions_keep_exact_distinct_node_bindings() -> None:
    state = _policy_state(EditingSubtask.DELETE, PropagationPolicy.STOP)
    trace = render_formal(state)
    step1 = trace.steps[0]
    step2 = trace.steps[1]
    step3 = trace.steps[2]

    assert any(slot.node_id == "remove_group_step1" for slot in step1.slots)
    assert any(slot.node_id == "remove_group_step2" for slot in step2.slots)
    assert any(slot.node_id == "remove_group_step2" for slot in step3.slots)
    assert (
        sum(
            slot.node_id == "remove_group_step2"
            for step in trace.steps
            for slot in step.slots
        )
        == 2
    )
    _assert_typed_formal_projection(state, trace)


def test_formal_renderer_is_oracle_blind_and_does_not_repair_locked_values() -> None:
    state = _policy_state(EditingSubtask.ADD, PropagationPolicy.STOP)
    baseline = render_formal(state)
    values = dict(state.values)
    values["oracle_gt"] = replace(
        values["oracle_gt"],
        raw_value="__UNIQUE_GT_CANARY__",
        normalized_value="__UNIQUE_GT_CANARY__",
    )
    values["oracle_product_heavy"] = replace(
        values["oracle_product_heavy"],
        raw_value=987654,
        normalized_value=987654,
    )
    values["product_heavy"] = replace(
        values["product_heavy"],
        raw_value=123456,
        normalized_value=123456,
    )
    changed = StateDAG(state.schema, values, state.edge_values)
    rendered = render_formal(changed)

    assert "__UNIQUE_GT_CANARY__" not in repr(rendered)
    assert rendered != baseline
    parsed = parse_formal(rendered)
    assert parsed.values["product_heavy"] == 123456
    assert all(
        state.schema.nodes_by_id[slot.node_id].visibility is Visibility.CANDIDATE_OUTPUT
        for step in rendered.steps
        for slot in step.slots
    )


@pytest.mark.parametrize(
    "unusual_fragment",
    (
        r"F/C=C\F",
        "[13CH3][C@@H](F)Cl",
        "[NH3+]CC(=O)[O-]",
        "C%12CCCCC%12",
        "CC.CN",
        "{not-a-template-placeholder}",
        "分子片段",
    ),
)
def test_formal_round_trip_preserves_unusual_safe_string_payloads(
    unusual_fragment: str,
) -> None:
    _, artifact, _, _ = _reference(
        _subtask_records(EditingSubtask.ADD)[0].anonymous_sample_id
    )
    values = dict(artifact.state_dag.values)
    values["add_fragment"] = replace(
        values["add_fragment"],
        raw_value=unusual_fragment,
        normalized_value=unusual_fragment,
    )
    state = StateDAG(artifact.state_dag.schema, values)
    trace = render_formal(state)
    assert unusual_fragment in "\n".join(trace.formal_lines)
    _assert_typed_formal_projection(state, trace)


@pytest.mark.parametrize("unsafe", ('C"N', "C\nN", "C\rN", "C\x00N"))
def test_unsafe_quoted_slot_payloads_fail_closed(unsafe: str) -> None:
    _, artifact, _, _ = _reference(
        _subtask_records(EditingSubtask.ADD)[0].anonymous_sample_id
    )
    values = dict(artifact.state_dag.values)
    values["add_fragment"] = replace(
        values["add_fragment"], raw_value=unsafe, normalized_value=unsafe
    )
    with pytest.raises(FormalRenderError) as captured:
        render_formal(StateDAG(artifact.state_dag.schema, values))
    assert captured.value.code == "FORMAL_SLOT_MISMATCH"


def test_schema_valid_but_formal_invalid_atom_index_is_structured() -> None:
    state = _policy_state(EditingSubtask.ADD, PropagationPolicy.STOP)
    values = dict(state.values)
    values["anchor_idx"] = replace(
        values["anchor_idx"],
        raw_value=0,
        normalized_value=0,
    )
    locked = StateDAG(state.schema, values, state.edge_values)

    with pytest.raises(FormalRenderError) as captured:
        render_formal(locked)
    assert captured.value.code == "FORMAL_SLOT_MISMATCH"


def _add_trace() -> tuple[StateDAG, RenderedFormalTrace]:
    state = _policy_state(EditingSubtask.ADD, PropagationPolicy.STOP)
    return state, render_formal(state)


def _adversarial_trace(trace: RenderedFormalTrace, steps: tuple) -> RenderedFormalTrace:
    """Bypass frozen construction only to exercise parser fail-closed boundaries."""

    tampered = replace(trace)
    object.__setattr__(tampered, "steps", steps)
    return tampered


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reorder"))
def test_missing_duplicate_and_reordered_steps_fail_closed(mutation: str) -> None:
    _state, trace = _add_trace()
    if mutation == "missing":
        steps = trace.steps[:-1]
    elif mutation == "duplicate":
        steps = (*trace.steps, trace.steps[-1])
    else:
        steps = (trace.steps[1], trace.steps[0], *trace.steps[2:])
    tampered = _adversarial_trace(trace, tuple(steps))
    with pytest.raises(FormalRenderError) as captured:
        parse_formal(tampered)
    assert captured.value.code == "FORMAL_STEP_MISMATCH"


@pytest.mark.parametrize(
    "change",
    (
        lambda step: replace(step, step_index=99),
        lambda step: replace(step, step_name="UNKNOWN_STEP"),
        lambda step: replace(
            step, formal_ab=step.formal_ab.replace(" --> ", " -> ", 1)
        ),
        lambda step: replace(
            step, formal_ab=step.formal_ab.replace('element="', "element=", 1)
        ),
    ),
)
def test_step_identity_literal_punctuation_and_quotes_are_byte_exact(change) -> None:
    _state, trace = _add_trace()
    tampered = _adversarial_trace(trace, (change(trace.steps[0]), *trace.steps[1:]))
    with pytest.raises(FormalRenderError) as captured:
        parse_formal(tampered)
    assert captured.value.code in {"FORMAL_STEP_MISMATCH", "FORMAL_PARSE_ERROR"}


def test_parser_reads_formal_text_and_uses_slots_only_as_typed_audit() -> None:
    state, trace = _add_trace()
    first = trace.steps[0]
    anchor = next(slot for slot in first.slots if slot.node_id == "anchor_idx")
    changed = anchor.value + 1
    tampered_text = first.formal_ab.replace(f"idx={anchor.value}", f"idx={changed}", 1)
    text_only = replace(first, formal_ab=tampered_text)
    with pytest.raises(FormalRenderError) as slot_error:
        parse_formal(replace(trace, steps=(text_only, *trace.steps[1:])))
    assert slot_error.value.code == "FORMAL_SLOT_MISMATCH"

    matching_slots = tuple(
        replace(slot, value=changed) if slot.node_id == "anchor_idx" else slot
        for slot in first.slots
    )
    one_occurrence_tamper = replace(
        first,
        formal_ab=tampered_text,
        slots=matching_slots,
    )
    with pytest.raises(FormalRenderError) as cross_step_error:
        parse_formal(replace(trace, steps=(one_occurrence_tamper, *trace.steps[1:])))
    assert cross_step_error.value.code == "FORMAL_SLOT_MISMATCH"

    updated_steps = tuple(
        replace(
            step,
            formal_ab=step.formal_ab.replace(f"idx={anchor.value}", f"idx={changed}"),
            slots=tuple(
                replace(slot, value=changed) if slot.node_id == "anchor_idx" else slot
                for slot in step.slots
            ),
        )
        if any(slot.node_id == "anchor_idx" for slot in step.slots)
        else step
        for step in trace.steps
    )
    tampered_trace = replace(trace, steps=updated_steps)
    parsed = parse_formal(tampered_trace)
    assert parsed.values["anchor_idx"] == changed
    with pytest.raises(FormalRenderError) as roundtrip_error:
        DeterministicFormalRenderer().assert_round_trip(state, tampered_trace)
    assert roundtrip_error.value.code == "FORMAL_ROUND_TRIP_MISMATCH"


def test_unknown_missing_and_wrong_typed_slots_fail_closed() -> None:
    _state, trace = _add_trace()
    first = trace.steps[0]
    unknown = FormalSlotValue(
        field_name="unknown_field",
        node_id="unknown_node",
        value_type=ValueType.STRING,
        value="unknown",
    )
    wrong_typed = replace(first.slots[0])
    object.__setattr__(wrong_typed, "value_type", ValueType.STRING)
    cases = (
        replace(first, slots=first.slots[:-1]),
        replace(first, slots=(*first.slots, unknown)),
        replace(
            first,
            slots=(
                replace(first.slots[0], node_id="unknown_node"),
                *first.slots[1:],
            ),
        ),
        replace(
            first,
            slots=(
                wrong_typed,
                *first.slots[1:],
            ),
        ),
    )
    for malformed in cases:
        with pytest.raises(FormalRenderError) as captured:
            parse_formal(replace(trace, steps=(malformed, *trace.steps[1:])))
        assert captured.value.code == "FORMAL_SLOT_MISMATCH"


def test_integer_grammar_requires_canonical_signs() -> None:
    _, artifact, _, _ = _reference(
        _subtask_records(EditingSubtask.ADD)[0].anonymous_sample_id
    )
    values = dict(artifact.state_dag.values)
    values["heavy_delta"] = replace(
        values["heavy_delta"], raw_value=3, normalized_value=3
    )
    state = StateDAG(artifact.state_dag.schema, values)
    trace = render_formal(state)
    heavy_step = next(
        step for step in trace.steps if "HEAVY_ATOM_DELTA" in step.formal_ab
    )
    assert "HEAVY_ATOM_DELTA(+3)" in heavy_step.formal_ab

    unsigned = trace.steps[0]
    anchor = next(slot for slot in unsigned.slots if slot.node_id == "anchor_idx")
    plus_unsigned = replace(
        unsigned,
        formal_ab=unsigned.formal_ab.replace(
            f"idx={anchor.value}", f"idx=+{anchor.value}", 1
        ),
    )
    missing_delta_sign = replace(
        heavy_step,
        formal_ab=heavy_step.formal_ab.replace("DELTA(+3)", "DELTA(3)"),
    )
    for malformed in (plus_unsigned, missing_delta_sign):
        steps = tuple(
            malformed if step.step_index == malformed.step_index else step
            for step in trace.steps
        )
        with pytest.raises(FormalRenderError) as captured:
            parse_formal(replace(trace, steps=steps))
        assert captured.value.code == "FORMAL_PARSE_ERROR"


def test_unknown_or_drifted_schema_fails_closed() -> None:
    state = _policy_state(EditingSubtask.ADD, PropagationPolicy.STOP)
    wrong_version = replace(state.schema, version="unknown-version")
    with pytest.raises(FormalRenderError) as captured:
        render_formal(StateDAG(wrong_version, state.values, state.edge_values))
    assert captured.value.code == "FORMAL_SCHEMA_MISMATCH"


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
@pytest.mark.parametrize(
    "policy",
    (PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.TERMINAL),
)
def test_non_full_answer_policies_are_faithful_and_only_audit_product_relation(
    subtask: EditingSubtask,
    policy: PropagationPolicy,
) -> None:
    state = _policy_state(subtask, policy)
    answer = render_answer(state, policy=policy)
    locked = state.values["final_answer"].normalized_value
    product = state.values["product"].normalized_value

    assert answer.policy is policy
    assert answer.source_node_id == "final_answer"
    assert answer.smiles == locked
    assert answer.product_equivalent is isomeric_graph_equivalent(locked, product)


@pytest.mark.parametrize("subtask", tuple(EditingSubtask))
def test_full_answer_requires_candidate_product_equivalence_and_exact_surface(
    subtask: EditingSubtask,
) -> None:
    state = _full_state(subtask)
    answer = DeterministicAnswerRenderer().render(
        state, policy=PropagationPolicy.FULL_CF
    )
    assert answer.smiles == state.values["final_answer"].normalized_value
    assert answer.product_equivalent
    assert isomeric_graph_equivalent(
        answer.smiles, state.values["product"].normalized_value
    )


def _alternate_graph_equivalent_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    assert molecule is not None
    for index, atom in enumerate(molecule.GetAtoms(), start=1):
        atom.SetAtomMapNum(index)
    alternate = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)
    assert alternate != smiles
    assert isomeric_graph_equivalent(alternate, smiles)
    return alternate


def test_string_different_graph_equivalent_full_answer_is_not_rewritten() -> None:
    state = _full_state(EditingSubtask.ADD)
    product = state.values["product"].normalized_value
    alternate = _alternate_graph_equivalent_smiles(product)
    values = dict(state.values)
    values["final_answer"] = replace(
        values["final_answer"], raw_value=alternate, normalized_value=alternate
    )
    changed = StateDAG(state.schema, values, state.edge_values)
    answer = render_answer(changed, policy=PropagationPolicy.FULL_CF)
    assert answer.smiles == alternate
    assert answer.product_equivalent


def test_terminal_allows_both_h_and_matched_n_relations_without_repair() -> None:
    reference = _policy_state(EditingSubtask.ADD, PropagationPolicy.STOP)
    product = reference.values["product"].normalized_value
    equivalent = _alternate_graph_equivalent_smiles(product)
    n_state = _terminal_state(EditingSubtask.ADD, answer=equivalent)
    h_state = _terminal_state(EditingSubtask.ADD, answer="C")
    if isomeric_graph_equivalent("C", h_state.values["product"].normalized_value):
        h_state = _terminal_state(EditingSubtask.ADD, answer="N")

    n_answer = render_answer(n_state, policy=PropagationPolicy.TERMINAL)
    h_answer = render_answer(h_state, policy=PropagationPolicy.TERMINAL)
    assert n_answer.smiles == equivalent
    assert n_answer.product_equivalent
    assert h_answer.smiles == h_state.values["final_answer"].normalized_value
    assert not h_answer.product_equivalent


def test_full_mismatch_and_invalid_answer_or_product_fail_closed() -> None:
    state = _full_state(EditingSubtask.ADD)
    values = dict(state.values)
    mismatch = (
        "C"
        if not isomeric_graph_equivalent("C", values["product"].normalized_value)
        else "N"
    )
    values["final_answer"] = replace(
        values["final_answer"], raw_value=mismatch, normalized_value=mismatch
    )
    with pytest.raises(FormalRenderError) as mismatch_error:
        render_answer(
            StateDAG(state.schema, values, state.edge_values),
            policy=PropagationPolicy.FULL_CF,
        )
    assert mismatch_error.value.code == "ANSWER_PRODUCT_MISMATCH"

    for node_id in ("final_answer", "product"):
        invalid_values = dict(state.values)
        invalid_values[node_id] = replace(
            invalid_values[node_id], raw_value="C1(", normalized_value="C1("
        )
        for policy in PropagationPolicy:
            with pytest.raises(FormalRenderError) as invalid_error:
                render_answer(
                    StateDAG(state.schema, invalid_values, state.edge_values),
                    policy=policy,
                )
            assert invalid_error.value.code == "ANSWER_INVALID_SMILES"


def test_answer_and_formal_renderers_are_oracle_blind_and_have_closed_signatures() -> (
    None
):
    state = _full_state(EditingSubtask.SUBSTITUTE)
    baseline_formal = render_formal(state)
    baseline_answer = render_answer(state, policy=PropagationPolicy.FULL_CF)
    values = dict(state.values)
    values["oracle_gt"] = replace(
        values["oracle_gt"],
        raw_value="__GT_MUST_NEVER_BE_READ__",
        normalized_value="__GT_MUST_NEVER_BE_READ__",
    )
    canary = StateDAG(state.schema, values, state.edge_values)
    assert render_formal(canary) == baseline_formal
    assert render_answer(canary, policy=PropagationPolicy.FULL_CF) == baseline_answer
    assert "__GT_MUST_NEVER_BE_READ__" not in repr(baseline_formal)
    assert "__GT_MUST_NEVER_BE_READ__" not in repr(baseline_answer)

    signatures = (
        inspect.signature(DeterministicFormalRenderer.render),
        inspect.signature(DeterministicFormalRenderer.parse),
        inspect.signature(DeterministicAnswerRenderer.render),
        inspect.signature(render_formal),
        inspect.signature(parse_formal),
        inspect.signature(render_answer),
    )
    forbidden = {"gt", "truth", "edit_truth", "record", "task_record", "labels"}
    assert all(not forbidden & set(signature.parameters) for signature in signatures)


def test_rendered_contracts_are_frozen_and_deeply_immutable() -> None:
    state, trace = _add_trace()
    parsed = parse_formal(trace)
    answer = render_answer(state, policy=PropagationPolicy.STOP)
    with pytest.raises(FrozenInstanceError):
        trace.schema_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        trace.steps[0].formal_ab = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        parsed.values["anchor_idx"] = 999  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        answer.smiles = "changed"  # type: ignore[misc]
