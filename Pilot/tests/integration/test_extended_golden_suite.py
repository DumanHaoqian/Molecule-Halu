"""T044 extended end-to-end golden-suite integration tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from molhallulens.annotation.token_projection import rebase_char_annotations
from molhallulens.builders.golden_validation import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_REPORT_PATH,
    DELETE_WITH_REPLACEMENT_ORIGIN_ID,
    T044_FIXTURE_FORMAT_VERSION,
    T044_GOLDEN_ORIGIN_CASES,
    T044_REPORT_FORMAT_VERSION,
    ExtendedGoldenSuiteBuild,
    build_t044_extended_golden_suite,
)
from molhallulens.domain import EditingSubtask, PropagationPolicy, VariantLabel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def suite() -> ExtendedGoldenSuiteBuild:
    return build_t044_extended_golden_suite()


def test_inventory_has_three_complete_real_origins_per_subtask(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    assert len(T044_GOLDEN_ORIGIN_CASES) == 9
    assert len(suite.origins) == 9
    assert len(suite.artifacts) == 72
    assert Counter(
        item.case.spec.normalized_subtask for item in suite.origins
    ) == Counter({subtask: 3 for subtask in EditingSubtask})
    assert (
        sum(
            item.draft.variant_label is VariantLabel.HALLUCINATED
            for item in suite.artifacts
        )
        == 36
    )
    assert (
        sum(
            item.draft.variant_label is VariantLabel.FAITHFUL
            for item in suite.artifacts
        )
        == 36
    )


def test_every_origin_freezes_full_eight_record_matrix_and_passes_t043(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    expected = Counter(
        (policy, label)
        for policy in PropagationPolicy
        for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
    )
    for origin in suite.origins:
        assert len(origin.artifacts) == 8
        assert (
            Counter(
                (item.draft.policy, item.draft.variant_label)
                for item in origin.artifacts
            )
            == expected
        )
        assert origin.validation.all_pass
        assert not origin.validation.issues
        by_id = {item.record_id: item for item in origin.artifacts}
        assert all(
            by_id[item.draft.matched_record_id].draft.matched_record_id
            == item.record_id
            for item in origin.artifacts
        )


def test_rendered_mentions_and_character_annotations_round_trip_exactly(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    for artifact in suite.artifacts:
        text = artifact.rendered.detector_text
        for mention in artifact.rendered.mentions:
            assert text[mention.literal_span.start : mention.literal_span.end] == (
                mention.literal_text
            )
            assert mention.claim_span.start <= mention.literal_span.start
            assert mention.literal_span.end <= mention.claim_span.end
        linked = {
            span_id
            for link in artifact.char_annotations.event_links
            for span_id in link.span_ids
        }
        assert linked == {
            item.span_id for item in artifact.char_annotations.annotations
        }


def test_token_outputs_are_complete_and_faithful_controls_are_zero_masked(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    for artifact in suite.artifacts:
        labels = artifact.token_labels
        assert labels is not None
        length = len(labels.input_ids)
        assert length > 2
        direct = (
            labels.attention_mask,
            labels.offset_mapping,
            labels.segment_ids,
            labels.evaluation_mask,
            labels.hallucination_core_mask,
            labels.error_any_mask,
            labels.local_falsehood_mask,
            labels.off_task_branch_mask,
            labels.reasoning_mask,
            labels.answer_mask,
            labels.boundary_ambiguous_mask,
            labels.error_char_fraction,
        )
        assert all(len(values) == length for values in direct)
        assert all(
            len(values) == length
            for mapping in (
                labels.semantic_type_masks,
                labels.edit_subtype_masks,
                labels.causal_role_masks,
            )
            for values in mapping.values()
        )
        if artifact.draft.variant_label is VariantLabel.FAITHFUL:
            assert labels.matched_target_span is not None
            assert not labels.has_positive_labels
            assert not any(labels.error_any_mask)
            assert not any(
                value
                for mapping in (
                    labels.semantic_type_masks,
                    labels.edit_subtype_masks,
                    labels.causal_role_masks,
                )
                for values in mapping.values()
                for value in values
            )
            continue

        assert labels.has_positive_labels
        rebased = rebase_char_annotations(
            artifact.rendered,
            artifact.serialized,
            artifact.char_annotations,
        )
        for annotation in rebased.annotations:
            assert any(
                labels.evaluation_mask[index]
                and start < annotation.literal_span.end
                and annotation.literal_span.start < end
                for index, (start, end) in enumerate(labels.offset_mapping)
            )


def test_real_multi_mapping_dual_anchor_and_terminal_near_miss_are_frozen(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    by_id = {item.case.spec.origin_id: item for item in suite.origins}
    add_multi = by_id["mol_edit.add_v2.0071"].golden.executions[0].context.truth
    substitute_multi = (
        by_id["mol_edit.substitute_v2.0271"].golden.executions[0].context.truth
    )
    assert add_multi.mapping_evidence.optimal_mapping_count > 1
    assert len(substitute_multi.valid_anchor_indices) > 1

    terminal_count = 0
    for origin in suite.origins:
        terminal = next(
            item
            for item in origin.golden.executions
            if item.context.recipe.policy is PropagationPolicy.TERMINAL
        )
        trace = terminal.to_trace_dict()["candidate_pool"]
        assert trace["selected_answer_similarity"] > 0.0
        assert trace["selected_answer_similarity"] == trace["max_answer_similarity"]
        assert terminal.outcome.candidate_graph.value_for(
            "product"
        ).semantically_equals(terminal.context.reference_graph.value_for("product"))
        assert not terminal.outcome.candidate_graph.value_for(
            "final_answer"
        ).semantically_equals(
            terminal.context.reference_graph.value_for("final_answer")
        )
        terminal_count += 1
    assert terminal_count == 9


def test_delete_with_replacement_is_an_exact_fail_closed_negative_golden(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    value = suite.delete_with_replacement
    assert value["origin_id"] == DELETE_WITH_REPLACEMENT_ORIGIN_ID
    assert value["classification"]["operation_subtype"] == ("delete_with_replacement")
    assert value["classification"]["structural_signature"] == {
        "removed_atom_count": 24,
        "added_atomic_numbers": [6, 7],
        "broken_boundary_bond_count": 1,
        "formed_boundary_bond_count": 1,
        "remove_fragment_heavy_atoms": 24,
        "add_fragment_heavy_atoms": 2,
    }
    assert value["expected_rejection"]["code"] == ("OPERATOR_CAPABILITY_FORBIDDEN")
    assert value["expected_rejection"]["evidence"] == {
        "forbidden_capabilities": ("structural_deletion",)
    }
    assert value["full_eight_record_status"] == "not_constructed_by_design"
    assert value["unsupported_policy"] == PropagationPolicy.FULL_CF.dataset_name


def test_frozen_fixture_and_report_are_exact_offline_replays(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    fixture_text = DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8")
    report_text = DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
    assert fixture_text == suite.render_fixture_json()
    assert report_text == suite.render_report_json()
    assert suite.render_fixture_json() == suite.render_fixture_json()
    assert suite.render_report_json() == suite.render_report_json()

    fixture = json.loads(fixture_text)
    report = json.loads(report_text)
    assert fixture["format_version"] == T044_FIXTURE_FORMAT_VERSION
    assert report["format_version"] == T044_REPORT_FORMAT_VERSION
    assert fixture["execution"] == {
        "deterministic_replay": True,
        "live_poe_attempted": False,
        "network_mode": "offline",
        "renderer": "natural_rule_v1",
        "tokenizer_backend": "deterministic_fast_offset_fixture",
    }
    assert report["all_pass"] is True
    assert report["summary"]["record_count"] == 72
    assert report["summary"]["live_poe_attempt_count"] == 0


def test_fixture_contains_every_required_full_record_layer(
    suite: ExtendedGoldenSuiteBuild,
) -> None:
    fixture = suite.fixture_artifact()
    required = {
        "draft",
        "state",
        "rendered",
        "char_annotations",
        "serialized_detector_input",
        "token_labels",
        "trace_labels",
        "split",
        "leakage_group_id",
        "validation",
    }
    records = [
        record for origin in fixture["origin_bundles"] for record in origin["records"]
    ]
    assert len(records) == 72
    assert all(set(record) == required for record in records)
    assert all(
        record["validation"]["chain"]["all_pass"]
        and len(record["validation"]["gates"]) == 4
        and all(gate["all_pass"] for gate in record["validation"]["gates"])
        for record in records
    )
