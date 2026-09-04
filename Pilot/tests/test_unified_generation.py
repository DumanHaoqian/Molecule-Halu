from __future__ import annotations

import json
from dataclasses import replace
from random import Random

import pytest
from rdkit import Chem

import molhallulens.core as core
from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.release import UnifiedRecordBuilder
from molhallulens.modules.text_realization import (
    AffectedNodeClaim,
    DeterministicTextRenderer,
    FORMAL_MARKER,
    PoeStepRewriteInput,
    PoeStepTextAgent,
    PoeTextRealizationError,
    PoeTextRenderer,
    RequiredHallucinationOccurrence,
    StepRewriteMode,
    build_poe_rewrite_request,
    validate_rewritten_step_text,
)
from molhallulens.modules.text_realization.occurrence_audit import arithmetic_violations
from molhallulens.modules.text_realization.poe_agent import _extract_natural_body
from molhallulens.modules.text_realization.renderer import (
    _original_occurrence_spans,
    _render_marked_natural_body,
)


def _only_target(subtask: str, semantic_id: str):
    return {
        name: ((semantic_id,) if name == subtask else ())
        for name in ("add", "delete", "substitute")
    }


def _mark_required_occurrences(step: dict) -> str:
    prefix = f"Step {step['step_index']} [{step['step_name']}]: "
    if step["rewrite_mode"] == "derivation_rewrite":
        claims = "; ".join(
            f"{item['node_id']}={item['after_text']}"
            for item in step["affected_node_claims"]
        )
        return f"[[HALLU:rewrite.01]]Updated claims: {claims}.[[/HALLU]]"
    head = step["original_step_text"].split(FORMAL_MARKER, 1)[0]
    body = head[len(prefix) :]
    for occurrence in sorted(
        step["required_hallucination_occurrences"],
        key=lambda item: item["original_span"][0],
        reverse=True,
    ):
        start, end = occurrence["original_span"]
        assert body[start:end] == occurrence["before_text"]
        marker = (
            f"[[HALLU:{occurrence['occurrence_id']}]]"
            f"{occurrence['after_text']}[[/HALLU]]"
        )
        body = body[:start] + marker + body[end:]
    return body


def _record(reference, plan):
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    rendered = DeterministicTextRenderer().render(reference, injected)
    annotated = UnifiedHallucinationAnnotator().annotate(rendered, injected)
    return injected, rendered, annotated, UnifiedRecordBuilder().build(
        reference, injected, annotated
    )


def test_old_strategy_and_control_types_are_absent():
    for old_name in (
        "PropagationPolicy",
        "VariantLabel",
        "OriginBundle",
        "PerturbationResult",
        "GraphDelta",
    ):
        assert not hasattr(core, old_name)


def test_reference_corpus_and_dag_shape(all_references):
    assert len(all_references) == 150
    assert {item.normalized_subtask.value for item in all_references} == {
        "add",
        "delete",
        "substitute",
    }
    addition = next(
        item for item in all_references
        if item.anonymous_sample_id == "mol_edit.add_v2.0003"
    )
    assert len(addition.state_dag.values) == 22
    assert len(addition.state_dag.schema.edges) == 20
    assert len(addition.trace_steps) == 5


def test_fragment_pool_is_corpus_scale_and_deduplicated(fragment_pool):
    assert len(fragment_pool) >= 70
    smiles = [item.canonical_smiles for item in fragment_pool.entries]
    assert len(smiles) == len(set(smiles))
    selection = fragment_pool.select_replacement(
        "none",
        config=DEFAULT_HALLUCINATION_CONFIG,
        random_source=Random(1),
    )
    assert Chem.MolFromSmiles(selection.entry.canonical_smiles) is not None


def test_planner_is_deterministic_and_honors_fixed_edit_count(
    references,
    fragment_pool,
):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=3,
        final_answer_probability=1.0,
    )
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    first = planner.plan(references["add"], variant_index=9)
    second = planner.plan(references["add"], variant_index=9)
    assert first == second
    assert len(first.mutations) == 3
    assert len({item.semantic_target_id for item in first.mutations}) == 3
    assert len(set(first.edited_node_ids)) == len(first.edited_node_ids)


def test_integer_root_propagates_to_the_derived_delta(references, fragment_pool):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        integer_deltas=(5,),
        editable_nodes_by_subtask=_only_target("add", "product_rings"),
    )
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(references["add"])
    mutation = plan.mutations[0]
    assert mutation.operator == "integer_offset"
    assert mutation.mutation_category.value == "numeric_integer"
    assert type(mutation.before) is int and type(mutation.after) is int
    assert mutation.magnitude == 5
    assert mutation.after - mutation.before == 5
    injected = UnifiedHallucinationInjector(config).apply(
        references["add"].state_dag,
        plan,
    )
    expected_delta = (
        injected.candidate_graph.values["product_rings"].normalized_value
        - injected.candidate_graph.values["source_rings"].normalized_value
    )
    assert injected.candidate_graph.values["ring_delta"].normalized_value == expected_delta
    assert {event.target_node_id for event in injected.propagation_events} == {
        "ring_delta"
    }


def test_delete_remove_group_is_one_semantic_edit_but_two_dag_nodes(
    references,
    fragment_pool,
):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        editable_nodes_by_subtask=_only_target("delete", "remove_group"),
        fragment_similarity_min=0.0,
        fragment_similarity_max=0.999,
        fragment_target_similarity=0.5,
        fragment_require_same_charge=False,
        fragment_max_heavy_atom_difference=100,
    )
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(references["delete"])
    mutation = plan.mutations[0]
    assert mutation.semantic_target_id == "remove_group"
    assert mutation.target_node_ids == ("remove_group_step1", "remove_group_step2")
    injected, _, annotated, record = _record(references["delete"], plan)
    differences = injected.reference_graph.semantic_differences(injected.candidate_graph)
    assert {target_id for _, target_id in differences} == set(injected.changed_node_ids)
    assert record.data["edit_count"] == 1
    assert {span.node_id for span in annotated.spans} == set(injected.changed_node_ids)


def test_final_answer_is_a_valid_structural_smiles_edit(references, fragment_pool):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_reasoning_steps=False,
        include_final_answer=True,
        final_answer_probability=1.0,
        editable_nodes_by_subtask=_only_target("substitute", "final_answer"),
    )
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(
        references["substitute"]
    )
    mutation = plan.mutations[0]
    assert mutation.semantic_target_id == "final_answer"
    assert mutation.mutation_category.value == "smiles_structure"
    assert mutation.before != mutation.after
    assert Chem.MolFromSmiles(mutation.after) is not None
    injected, rendered, annotated, record = _record(references["substitute"], plan)
    assert rendered.final_answer == mutation.after
    assert "final_answer" in {span.component for span in annotated.spans}
    assert injected.candidate_graph.edge_satisfaction(
        "product_to_final_answer",
        None,
    ) is None
    assert "product_to_final_answer" not in injected.violated_edge_ids
    assert injected.candidate_graph.values["product"].normalized_value == rendered.final_answer
    assert record.data["labels"]["hallucination_present"] is True


def test_default_end_to_end_for_all_three_subtasks(references, fragment_pool):
    planner = UnifiedHallucinationPlanner(fragment_pool)
    forbidden = {"policy", "variant_label", "pair_id", "bundle_id", "matched_record_id"}
    for variant_index, reference in enumerate(references.values()):
        plan = planner.plan(reference, variant_index=variant_index)
        injected, rendered, annotated, record = _record(reference, plan)
        data = record.to_dict()
        assert 1 <= len(plan.mutations) <= planner.maximum_edit_count(reference)
        assert len(injected.reference_graph.semantic_differences(injected.candidate_graph)) \
            == len(injected.changed_node_ids)
        assert len(rendered.step_texts) == (6 if reference.normalized_subtask.value == "substitute" else 5)
        assert annotated.spans
        assert forbidden.isdisjoint(data)
        serialized = data["serialized"]["text"]
        for span in data["hallucination_spans"]:
            start, end = span["serialized_span"]
            assert serialized[start:end] == span["text"]
        json.dumps(data, ensure_ascii=False)


def test_full_corpus_has_deterministic_closure_and_complete_occurrence_contract(
    all_references,
    fragment_pool,
):
    planner = UnifiedHallucinationPlanner(fragment_pool)
    injector = UnifiedHallucinationInjector()
    hard_relations = {
        core.DependencyType.DELTA_OF,
        core.DependencyType.MUST_EQUAL,
        core.DependencyType.MOLECULARLY_EQUIVALENT_TO,
    }
    for variant_index, reference in enumerate(all_references):
        plan = planner.plan(reference, variant_index=variant_index)
        injected = injector.apply(reference.state_dag, plan)
        assert not {
            item.edge_id
            for item in injected.edge_audit
            if item.status is False and item.relation in hard_relations
        }
        request = build_poe_rewrite_request(reference, injected)
        for step in request.steps:
            occurrence_ids = [
                item.occurrence_id
                for item in step.required_hallucination_occurrences
            ]
            assert len(occurrence_ids) == len(set(occurrence_ids))
        rendered = DeterministicTextRenderer().render(reference, injected)
        annotated = UnifiedHallucinationAnnotator().annotate(rendered, injected)
        assert {span.node_id for span in annotated.spans} == set(
            injected.changed_node_ids
        )


def test_variant_zero_full_corpus_routes_incomplete_mentions_fail_closed(
    all_references,
    fragment_pool,
):
    planner = UnifiedHallucinationPlanner(fragment_pool)
    injector = UnifiedHallucinationInjector()
    rewrite_count = 0
    for reference in all_references:
        plan = planner.plan(reference, variant_index=0)
        injected = injector.apply(reference.state_dag, plan)
        request = build_poe_rewrite_request(reference, injected)
        rewrite_count += sum(
            step.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE
            for step in request.steps
        )
        rendered = DeterministicTextRenderer().render(reference, injected)
        assert not arithmetic_violations(rendered.reasoning_chain)
        annotated = UnifiedHallucinationAnnotator().annotate(rendered, injected)
        assert {span.node_id for span in annotated.spans} == set(
            injected.changed_node_ids
        )
    assert rewrite_count > 0


@pytest.mark.parametrize(
    ("origin_id", "step_index", "node_id", "after_text"),
    (
        ("mol_edit.add_v2.0016", 2, "fragment_heavy", "2"),
        ("mol_edit.add_v2.0022", 4, "product_heavy", "47"),
        ("mol_edit.add_v2.0057", 2, "fragment_heavy", "8"),
    ),
)
def test_claude_recall_examples_use_derivation_rewrite(
    all_references,
    fragment_pool,
    origin_id,
    step_index,
    node_id,
    after_text,
):
    reference = next(
        item for item in all_references if item.anonymous_sample_id == origin_id
    )
    # Preserve the original three-edit sampling envelope used to discover these
    # exact regression fixtures; the production default now auto-detects capacity.
    fixture_config = replace(DEFAULT_HALLUCINATION_CONFIG, max_edit_count=3)
    plan = UnifiedHallucinationPlanner(fragment_pool, fixture_config).plan(
        reference,
        variant_index=0,
    )
    injected = UnifiedHallucinationInjector(fixture_config).apply(
        reference.state_dag,
        plan,
    )
    request = build_poe_rewrite_request(reference, injected)
    step = request.steps[step_index - 1]
    assert step.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE
    assert any(
        claim.node_id == node_id and claim.after_text == after_text
        for claim in step.affected_node_claims
    )
    rendered = DeterministicTextRenderer().render(reference, injected)
    assert after_text in rendered.step_texts[step_index - 1]
    assert not arithmetic_violations(rendered.step_texts[step_index - 1])


def test_poe_agent_rewrites_natural_text_but_local_code_locks_formal_and_spans(
    references,
    fragment_pool,
    tmp_path,
):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        integer_deltas=(1,),
        editable_nodes_by_subtask=_only_target("add", "product_rings"),
    )
    reference = references["add"]
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(reference)
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    request = build_poe_rewrite_request(reference, injected)
    mutation = plan.mutations[0]
    assert f"PRODUCT_SMILES[n_rings={mutation.after}]" in request.steps[4].modified_formal_ab
    assert f"PRODUCT_SMILES[n_rings={mutation.before}]" in request.steps[4].original_formal_ab
    assert request.steps[4].original_step_text == reference.trace_steps[4].render(
        include_answer=False
    )

    calls = []

    def fake_poe(system_prompt, user_prompt, bot_name, temperature):
        calls.append((system_prompt, bot_name, temperature))
        payload = json.loads(user_prompt.split("\nINPUT:\n", 1)[1])
        return json.dumps(
            {
                "steps": [
                    {
                        "step_index": step["step_index"],
                        "rewritten_natural_language": (
                            "\n  "
                            f"Step {step['step_index']} [{step['step_name']}]: "
                            + (
                                step["original_step_text"].split(FORMAL_MARKER, 1)[0][
                                    len(
                                        f"Step {step['step_index']} "
                                        f"[{step['step_name']}]: "
                                    ) :
                                ]
                                if step["rewrite_mode"] == "copy"
                                else _mark_required_occurrences(step)
                            )
                            + FORMAL_MARKER
                            + "POE_MUST_NOT_CONTROL_THIS\n"
                        ),
                    }
                    for step in payload["steps"]
                ]
            }
        )

    renderer = PoeTextRenderer(
        PoeStepTextAgent(
            config,
            transport=fake_poe,
            environment={"POE_API_KEY": "do-not-store-this-test-secret"},
            cache_directory=tmp_path,
        )
    )
    rendered = renderer.render(reference, injected)
    assert len(calls) == 1
    assert rendered.realization["backend"] == "poe_agent"
    assert rendered.realization["network_request_count"] == 1
    assert f"PRODUCT_SMILES[n_rings={mutation.after}]" in rendered.step_texts[4]
    assert "POE_MUST_NOT_CONTROL_THIS" not in rendered.reasoning_chain
    assert "[[HALLU:" not in rendered.reasoning_chain
    assert rendered.step_texts[:4] == tuple(
        step.render(include_answer=False) for step in reference.trace_steps[:4]
    )
    for step_text, step_request in zip(rendered.step_texts, request.steps, strict=True):
        assert step_text.split("\n  FORMAL: ", 1)[1] == step_request.modified_formal_ab
    annotated = UnifiedHallucinationAnnotator().annotate(rendered, injected)
    assert str(mutation.after) in {span.text for span in annotated.spans}
    assert {span.causal_role.value for span in annotated.spans} == {
        "root_hallucination",
        "propagated_error",
    }

    # A validated cache hit works without a transport and without reading a token.
    replay = PoeTextRenderer(
        PoeStepTextAgent(config, environment={}, cache_directory=tmp_path)
    ).render(reference, injected)
    assert replay.step_texts == rendered.step_texts
    assert replay.realization["cache_hit"] is True
    assert replay.realization["network_request_count"] == 0
    cache_contents = "".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    assert "do-not-store-this-test-secret" not in cache_contents


def test_poe_key_is_read_only_from_the_named_environment_variable(
    references,
    fragment_pool,
    tmp_path,
):
    reference = references["add"]
    plan = UnifiedHallucinationPlanner(fragment_pool).plan(reference, variant_index=31)
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    request = build_poe_rewrite_request(reference, injected)
    agent = PoeStepTextAgent(environment={}, cache_directory=tmp_path)
    with pytest.raises(PoeTextRealizationError, match="export POE_API_KEY"):
        agent.rewrite(request)


def test_poe_cannot_drop_invent_or_change_hallucination_markers():
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="RING_CHECK",
        original_step_text=(
            "Step 1 [RING_CHECK]: The product has 4 rings."
            "\n  FORMAL: PRODUCT_SMILES[n_rings=4]"
        ),
        modified_formal_ab="PRODUCT_SMILES[n_rings=5]",
        required_hallucination_occurrences=(
            RequiredHallucinationOccurrence(
                occurrence_id="product_rings.01",
                node_id="product_rings",
                before_text="4",
                after_text="5",
                original_start=16,
                original_end=17,
            ),
        ),
    )
    valid = (
        "Step 1 [RING_CHECK]: The product has "
        "[[HALLU:product_rings.01]]5[[/HALLU]] rings."
        "\n  FORMAL: PRODUCT_SMILES[n_rings=5]"
    )
    assert validate_rewritten_step_text(valid, expected) == valid
    with pytest.raises(PoeTextRealizationError, match="omitted required"):
        validate_rewritten_step_text(
            "Step 1 [RING_CHECK]: The product has 5 rings."
            "\n  FORMAL: PRODUCT_SMILES[n_rings=5]",
            expected,
        )
    with pytest.raises(PoeTextRealizationError, match="unplanned"):
        validate_rewritten_step_text(
            valid.replace(
                " rings.",
                " rings and [[HALLU:ring_delta.01]]0[[/HALLU]] delta.",
            ),
            expected,
        )
    with pytest.raises(PoeTextRealizationError, match="exact modified value"):
        validate_rewritten_step_text(valid.replace("]]5[[", "]]6[["), expected)
    with pytest.raises(PoeTextRealizationError, match="modified_formal_ab exactly"):
        validate_rewritten_step_text(
            valid.replace("PRODUCT_SMILES[n_rings=5]", "PRODUCT_SMILES[n_rings=6]"),
            expected,
        )


def test_poe_must_copy_an_unaffected_step_byte_for_byte():
    original = "Step 1 [CHECK]: Keep this exact.\n  FORMAL: VALUE(4)"
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="CHECK",
        original_step_text=original,
        modified_formal_ab="VALUE(4)",
        required_hallucination_occurrences=(),
    )
    assert validate_rewritten_step_text(original, expected) == original
    with pytest.raises(PoeTextRealizationError, match="without any required"):
        validate_rewritten_step_text(
            "Step 1 [CHECK]: Paraphrased.\n  FORMAL: VALUE(4)",
            expected,
        )


def test_poe_must_lock_natural_text_when_only_formal_changes():
    original = "Step 1 [PRODUCT]: Keep this exact.\n  FORMAL: PRODUCT(OLD)"
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="PRODUCT",
        original_step_text=original,
        modified_formal_ab="PRODUCT(NEW)",
        required_hallucination_occurrences=(),
    )
    valid = "Step 1 [PRODUCT]: Keep this exact.\n  FORMAL: PRODUCT(NEW)"
    assert validate_rewritten_step_text(valid, expected) == valid
    with pytest.raises(PoeTextRealizationError, match="without any required"):
        validate_rewritten_step_text(
            "Step 1 [PRODUCT]: Poe paraphrased this.\n  FORMAL: PRODUCT(NEW)",
            expected,
        )


def test_poe_requires_every_occurrence_exactly_once():
    original = "Step 1 [CHECK]: Value 4, repeated 4.\n  FORMAL: VALUE(4)"
    occurrences = (
        RequiredHallucinationOccurrence(
            occurrence_id="count.01",
            node_id="count",
            before_text="4",
            after_text="5",
            original_start=6,
            original_end=7,
        ),
        RequiredHallucinationOccurrence(
            occurrence_id="count.02",
            node_id="count",
            before_text="4",
            after_text="5",
            original_start=18,
            original_end=19,
        ),
    )
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="CHECK",
        original_step_text=original,
        modified_formal_ab="VALUE(5)",
        required_hallucination_occurrences=occurrences,
    )
    valid = (
        "Step 1 [CHECK]: Value [[HALLU:count.01]]5[[/HALLU]], repeated "
        "[[HALLU:count.02]]5[[/HALLU]].\n  FORMAL: VALUE(5)"
    )
    assert validate_rewritten_step_text(valid, expected) == valid
    with pytest.raises(PoeTextRealizationError, match="omitted required"):
        validate_rewritten_step_text(
            valid.replace("[[HALLU:count.02]]5[[/HALLU]]", "4"),
            expected,
        )
    with pytest.raises(PoeTextRealizationError, match="duplicated"):
        validate_rewritten_step_text(
            valid.replace(
                "[[HALLU:count.02]]5[[/HALLU]]",
                "[[HALLU:count.01]]5[[/HALLU]]",
            ),
            expected,
        )


def test_derivation_rewrite_rejects_stale_claims_and_false_arithmetic():
    expected = PoeStepRewriteInput(
        step_index=2,
        step_name="FRAGMENT_IDENTIFICATION",
        original_step_text=(
            "Step 2 [FRAGMENT_IDENTIFICATION]: Count precisely: 3 heavy atoms "
            "(two carbons and one oxygen).\n  FORMAL: HEAVY_ATOMS(3)"
        ),
        modified_formal_ab="HEAVY_ATOMS(2)",
        required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(
            AffectedNodeClaim(
                node_id="fragment_heavy",
                before_text="3",
                after_text="2",
            ),
        ),
        rewrite_node_id="fragment_heavy",
    )
    valid = (
        "Step 2 [FRAGMENT_IDENTIFICATION]: "
        "[[HALLU:rewrite.01]]The fragment contains 2 heavy atoms.[[/HALLU]]"
        "\n  FORMAL: HEAVY_ATOMS(2)"
    )
    assert validate_rewritten_step_text(valid, expected) == valid
    with pytest.raises(PoeTextRealizationError, match="stale claim"):
        validate_rewritten_step_text(
            valid.replace("2 heavy atoms", "3 heavy atoms"),
            expected,
        )
    with pytest.raises(PoeTextRealizationError, match="false displayed arithmetic"):
        validate_rewritten_step_text(
            valid.replace(
                "The fragment contains 2 heavy atoms.",
                "The fragment contains 2 heavy atoms because 1 + 1 = 3.",
            ),
            expected,
        )


def test_poe_redundant_step_header_inside_derivation_marker_is_removed():
    expected = PoeStepRewriteInput(
        step_index=5,
        step_name="RING_VERIFICATION",
        original_step_text=(
            "Step 5 [RING_VERIFICATION]: Verify ring count."
            "\n  FORMAL: RING_DELTA(0)"
        ),
        modified_formal_ab="RING_DELTA(+1)",
        required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(
            AffectedNodeClaim(
                node_id="ring_delta",
                before_text="0",
                after_text="+1",
            ),
        ),
        rewrite_node_id="ring_delta",
    )
    raw = (
        "[[HALLU:rewrite.01]]Step 5 [RING_VERIFICATION]: "
        "The product contains 5 rings, so 5 - 4 = +1."
        "[[/HALLU]]"
    )

    normalized = _extract_natural_body(raw, expected)
    assert normalized == (
        "[[HALLU:rewrite.01]]The product contains 5 rings, so 5 - 4 = +1."
        "[[/HALLU]]"
    )
    complete = (
        "Step 5 [RING_VERIFICATION]: "
        + normalized
        + FORMAL_MARKER
        + expected.modified_formal_ab
    )
    assert complete.count("Step 5 [RING_VERIFICATION]:") == 1
    assert validate_rewritten_step_text(complete, expected) == complete

    with pytest.raises(PoeTextRealizationError, match="mismatched Step header"):
        _extract_natural_body(
            raw.replace("Step 5 [RING_VERIFICATION]", "Step 4 [RING_VERIFICATION]"),
            expected,
        )


def test_anchor_occurrence_inventory_handles_common_surface_forms():
    natural = "Use atom 21; this is N21 and the indexed form is [NH:21]."
    spans = _original_occurrence_spans("anchor_idx", "21", natural)

    assert [natural[start:end] for start, end in spans] == ["21", "21", "21"]


def test_renderer_rejects_an_unmarked_stale_occurrence_after_valid_markers():
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="RING_CHECK",
        original_step_text=(
            "Step 1 [RING_CHECK]: The product has 4 rings and product has 4 rings."
            "\n  FORMAL: PRODUCT_SMILES[n_rings=4]"
        ),
        modified_formal_ab="PRODUCT_SMILES[n_rings=5]",
        required_hallucination_occurrences=(
            RequiredHallucinationOccurrence(
                occurrence_id="product_rings.01",
                node_id="product_rings",
                before_text="4",
                after_text="5",
                original_start=16,
                original_end=17,
            ),
        ),
    )

    with pytest.raises(PoeTextRealizationError, match="retained a stale value"):
        _render_marked_natural_body(
            marked_natural_body=(
                "The product has [[HALLU:product_rings.01]]5[[/HALLU]] rings "
                "and product has 4 rings."
            ),
            origin_id="example",
            step_index=1,
            reasoning_offset=0,
            occurrence_counts={},
            expected=expected,
            causal_roles_by_node={
                "product_rings": core.CausalRole.ROOT_HALLUCINATION,
            },
        )
