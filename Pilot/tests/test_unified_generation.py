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
    DeterministicTextRenderer,
    PoeStepTextAgent,
    PoeTextRealizationError,
    PoeTextRenderer,
    build_poe_rewrite_request,
    validate_natural_template,
)


def _only_target(subtask: str, semantic_id: str):
    return {
        name: ((semantic_id,) if name == subtask else ())
        for name in ("add", "delete", "substitute")
    }


def _record(reference, plan):
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    rendered = DeterministicTextRenderer().render(reference, injected)
    annotated = UnifiedHallucinationAnnotator().annotate(rendered, plan)
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
    assert first.mutations[0].semantic_target_id == "final_answer"
    assert len(set(first.edited_node_ids)) == len(first.edited_node_ids)


def test_integer_magnitude_is_exactly_configured(references, fragment_pool):
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        integer_deltas=(5,),
        editable_nodes_by_subtask=_only_target("add", "ring_delta"),
    )
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(references["add"])
    mutation = plan.mutations[0]
    assert mutation.operator == "integer_offset"
    assert mutation.mutation_category.value == "numeric_integer"
    assert type(mutation.before) is int and type(mutation.after) is int
    assert mutation.magnitude == 5
    assert mutation.after - mutation.before == 5


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
    assert {target_id for _, target_id in differences} == set(mutation.target_node_ids)
    assert record.data["edit_count"] == 1
    assert {span.node_id for span in annotated.spans} == set(mutation.target_node_ids)


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
    _, rendered, annotated, record = _record(references["substitute"], plan)
    assert rendered.final_answer == mutation.after
    assert {span.component for span in annotated.spans} == {"final_answer"}
    assert record.data["labels"]["hallucination_present"] is True


def test_default_end_to_end_for_all_three_subtasks(references, fragment_pool):
    planner = UnifiedHallucinationPlanner(fragment_pool)
    forbidden = {"policy", "variant_label", "pair_id", "bundle_id", "matched_record_id"}
    for variant_index, reference in enumerate(references.values()):
        plan = planner.plan(reference, variant_index=variant_index)
        injected, rendered, annotated, record = _record(reference, plan)
        data = record.to_dict()
        assert len(plan.mutations) in {1, 2, 3}
        assert len(injected.reference_graph.semantic_differences(injected.candidate_graph)) \
            == len(plan.edited_node_ids)
        assert len(rendered.step_texts) == (6 if reference.normalized_subtask.value == "substitute" else 5)
        assert annotated.spans
        assert forbidden.isdisjoint(data)
        serialized = data["serialized"]["text"]
        for span in data["hallucination_spans"]:
            start, end = span["serialized_span"]
            assert serialized[start:end] == span["text"]
        json.dumps(data, ensure_ascii=False)


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

    calls = []

    def fake_poe(system_prompt, user_prompt, bot_name, temperature):
        calls.append((system_prompt, bot_name, temperature))
        payload = json.loads(user_prompt.split("\nINPUT:\n", 1)[1])
        return json.dumps(
            {
                "steps": [
                    {
                        "step_index": step["step_index"],
                        "natural_language_template": (
                            "According to the rewritten context, "
                            + step["natural_template_draft"]
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
    for step_text, step_request in zip(rendered.step_texts, request.steps, strict=True):
        assert step_text.split("\n  FORMAL: ", 1)[1] == step_request.modified_formal_ab
    annotated = UnifiedHallucinationAnnotator().annotate(rendered, plan)
    assert {span.text for span in annotated.spans} == {str(mutation.after)}

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


def test_poe_cannot_drop_or_invent_locked_claim_placeholders():
    required = {"anchor_idx": 1, "anchor_element": 1}
    assert validate_natural_template(
        "Use atom {{anchor_idx}} with element {{anchor_element}}.",
        required,
    )
    with pytest.raises(PoeTextRealizationError, match="placeholder counts changed"):
        validate_natural_template("Use atom {{anchor_idx}}.", required)
    with pytest.raises(PoeTextRealizationError, match="placeholder counts changed"):
        validate_natural_template(
            "Use {{anchor_idx}}, {{anchor_element}}, and {{product}}.",
            required,
        )
