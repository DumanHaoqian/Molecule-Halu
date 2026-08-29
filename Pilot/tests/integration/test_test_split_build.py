from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pytest

from molhallulens.builders.test_split import (
    T050_H_COUNT,
    T050_N_COUNT,
    T050_ORIGIN_COUNT,
    T050_RECORD_COUNT,
    T050_TEST_ID,
    build_t050_test_split,
    write_t050_test_artifacts,
)
from molhallulens.builders.test_split import TestBuildAttempt as BuildAttempt
from molhallulens.builders.test_split import TestSplitBuild as SplitBuild
from molhallulens.builders.test_split import TestSplitBuildError as SplitBuildError
from molhallulens.domain import EditingSubtask, PropagationPolicy, VariantLabel


@pytest.fixture(scope="module")
def test_build() -> SplitBuild:
    return build_t050_test_split()


def _jsonl(payload: str) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in payload.splitlines())


def test_prerequisite_gate_runs_before_heldout_construction(tmp_path: Path) -> None:
    missing = tmp_path / "t049-not-published.json"
    with pytest.raises(SplitBuildError) as blocked:
        build_t050_test_split(validation_report_path=missing)
    assert blocked.value.code == "TEST_BUILD_ORDER"
    assert blocked.value.evidence["task_id"] == "T049"


def test_build_covers_exact_frozen_test_manifest(test_build: SplitBuild) -> None:
    assert len(test_build.origins) == T050_ORIGIN_COUNT == 25
    assert len(test_build.artifacts) == T050_RECORD_COUNT == 200
    assert Counter(
        origin.case.spec.normalized_subtask for origin in test_build.origins
    ) == Counter(
        {
            EditingSubtask.ADD: 8,
            EditingSubtask.DELETE: 8,
            EditingSubtask.SUBSTITUTE: 9,
        }
    )
    manifest_test = {
        row.anonymous_sample_id
        for row in test_build.split_manifest.rows
        if row.split.value == "test"
    }
    assert {
        origin.case.spec.origin_id for origin in test_build.origins
    } == manifest_test
    assert all(artifact.split.value == "test" for artifact in test_build.artifacts)


def test_every_test_origin_has_four_reciprocal_strict_pairs(
    test_build: SplitBuild,
) -> None:
    record_ids: set[str] = set()
    for origin in test_build.origins:
        assert origin.validation.all_pass
        assert len(origin.artifacts) == 8
        assert Counter(
            (artifact.draft.policy, artifact.draft.variant_label)
            for artifact in origin.artifacts
        ) == Counter(
            {
                (policy, label): 1
                for policy in PropagationPolicy
                for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
            }
        )
        pairs: dict[str, list[object]] = defaultdict(list)
        for artifact in origin.artifacts:
            assert artifact.record_id not in record_ids
            record_ids.add(artifact.record_id)
            pairs[artifact.draft.pair_id].append(artifact)
        assert len(pairs) == 4
        for pair in pairs.values():
            assert len(pair) == 2
            left, right = pair
            assert left.draft.matched_record_id == right.record_id
            assert right.draft.matched_record_id == left.record_id


def test_global_test_variant_and_policy_balance(test_build: SplitBuild) -> None:
    assert Counter(
        artifact.draft.variant_label for artifact in test_build.artifacts
    ) == Counter(
        {
            VariantLabel.HALLUCINATED: T050_H_COUNT,
            VariantLabel.FAITHFUL: T050_N_COUNT,
        }
    )
    assert Counter(
        (artifact.draft.policy, artifact.draft.variant_label)
        for artifact in test_build.artifacts
    ) == Counter(
        {
            (policy, label): 25
            for policy in PropagationPolicy
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        }
    )


def test_detector_oracle_state_token_and_provenance_are_isolated(
    test_build: SplitBuild,
) -> None:
    dataset = test_build.dataset_records()
    oracle = test_build.oracle_records()
    states = test_build.state_records()
    tokens = test_build.token_records()
    provenance = test_build.provenance_records()
    assert (
        len(dataset)
        == len(oracle)
        == len(states)
        == len(tokens)
        == len(provenance)
        == 200
    )
    expected_ids = {record["record_id"] for record in dataset}
    assert all(
        {record["record_id"] for record in family} == expected_ids
        for family in (oracle, states, tokens, provenance)
    )
    assert all(
        record["test_build_id"] == T050_TEST_ID and "dry_run_id" not in record
        for family in (dataset, oracle, states, tokens, provenance)
        for record in family
    )
    assert all(
        set(record["detector_input"])
        == {"indexed_smiles", "instruction", "reasoning_chain", "final_answer"}
        for record in dataset
    )
    assert all("gt_smiles" not in record for record in dataset)
    assert all(record["visible_to_detector"] is False for record in oracle)
    assert all(
        record["tokenizer_fingerprint"]["normalization_config"][
            "production_weights_loaded"
        ]
        is False
        and record["activation_alignment"] == "post_token_h_t"
        for record in tokens
    )
    assert all(
        record["execution_mode"]["network_mode"] == "frozen_offline"
        and record["execution_mode"]["live_poe_attempted"] is False
        and record["donor"]["recipient_split"] == "test"
        and record["donor"]["verified_split_local"] is True
        and record["test_isolation"]["test_used_to_modify_candidate_rules"] is False
        and record["test_isolation"]["test_used_for_layer_selection"] is False
        and record["test_isolation"]["test_used_for_threshold_selection"] is False
        for record in provenance
    )


def test_failed_test_recipe_is_zero_record_and_order_is_immutable(
    test_build: SplitBuild,
) -> None:
    first = test_build.attempts[0]
    rejected = BuildAttempt(
        attempt_index=0,
        origin_id=first.origin_id,
        subtask=first.subtask,
        case_id=f"{first.case_id}.injected-rejection",
        frozen_recipe_index=0,
        status="rejected",
        emitted_record_count=0,
        error_code="INJECTED_STRICT_REJECTION",
        exception_type="RuntimeError",
    )
    shifted_attempts = []
    for attempt in test_build.attempts:
        shifted = replace(attempt, attempt_index=attempt.attempt_index + 1)
        if attempt.origin_id == first.origin_id:
            shifted = replace(
                shifted,
                frozen_recipe_index=attempt.frozen_recipe_index + 1,
            )
        shifted_attempts.append(shifted)
    rebuilt = replace(test_build, attempts=(rejected, *shifted_attempts))
    assert rebuilt.artifacts == test_build.artifacts
    assert rebuilt.attempts[0].emitted_record_count == 0
    assert rebuilt.attempts[1].frozen_recipe_index == 1

    tampered = replace(
        rebuilt.attempts[1],
        frozen_recipe_index=2,
    )
    with pytest.raises(SplitBuildError) as order_error:
        replace(
            rebuilt, attempts=(rebuilt.attempts[0], tampered, *rebuilt.attempts[2:])
        )
    assert order_error.value.code == "TEST_FROZEN_RECIPE_ORDER"


def test_production_builder_has_no_test_selection_injection_hook() -> None:
    parameters = signature(build_t050_test_split).parameters
    assert "recipe_resolver" not in parameters
    assert "origin_builder" not in parameters
    assert "candidate_selector" not in parameters
    assert "renderer" not in parameters
    assert "layer" not in parameters
    assert "threshold" not in parameters


def test_machine_readable_isolation_declaration_and_strict_report(
    test_build: SplitBuild,
) -> None:
    declaration = test_build.isolation_declaration()
    report = test_build.build_report()
    validation = test_build.validation_report()
    selection = test_build.selection_manifest()

    assert declaration["all_pass"] is True
    assert declaration["build_order"] == {
        "test_built_last": True,
        "required_predecessors_checked_before_test_construction": [
            "T047",
            "T048",
            "T049",
        ],
        "train_complete_before_test": True,
        "validation_complete_before_test": True,
    }
    assert all(
        declaration["test_usage"][key] is False
        for key in (
            "used_for_candidate_rule_selection",
            "used_for_candidate_generation_tuning",
            "used_for_propagation_layer_selection",
            "used_for_detector_layer_selection",
            "used_for_detector_threshold_selection",
            "used_for_renderer_selection_or_tuning",
            "used_for_shortcut_threshold_selection",
            "diagnostic_results_feed_back_into_build",
        )
    )
    assert declaration["test_usage"]["used_for_strict_record_acceptance"] is True
    assert declaration["failure_semantics"]["cross_split_backfill_allowed"] is False
    assert declaration["detector_scope"]["formal_detector_training_authorized"] is False

    assert report["all_pass"] is True
    assert report["summary"]["origin_count"] == 25
    assert report["summary"]["record_count"] == 200
    assert report["strict_validation"] == {
        "artifact_gate_count": 800,
        "bundle_gate_count": 25,
        "all_pass": True,
    }
    assert report["isolation"]["test_used_to_modify_candidate_rules"] is False
    assert report["isolation"]["test_used_for_detector_layer_selection"] is False
    assert report["isolation"]["test_used_for_detector_threshold_selection"] is False
    assert report["isolation"]["diagnostic_feedback_into_build_count"] == 0
    assert report["execution"]["production_chemdfm_r_weights_loaded"] is False
    assert report["execution"]["explicit_digest_or_sha_verification_performed"] is False
    assert validation["all_pass"] is True
    assert validation["artifact_gate_count"] == 800
    assert validation["bundle_gate_count"] == 25
    assert selection["selection_split"] == "test"
    assert selection["test_used_to_modify_selection_rules"] is False


def test_release_payload_inventory_and_test_only_atomic_writer(
    tmp_path: Path,
    test_build: SplitBuild,
) -> None:
    payloads = test_build.artifact_payloads()
    assert set(payloads) == {
        "records/test.jsonl",
        "oracle/test.jsonl",
        "state_graphs/test.jsonl",
        "tokenized/chemdfm_r/test.jsonl",
        "provenance/test.jsonl",
        "reports/test_selection_manifest.json",
        "reports/test_validation_report.json",
        "reports/test_isolation_declaration.json",
        "reports/test_backfill_ledger.jsonl",
        "reports/test_build_report.json",
    }
    for family in (
        "records",
        "oracle",
        "state_graphs",
        "tokenized/chemdfm_r",
        "provenance",
    ):
        rows = _jsonl(payloads[f"{family}/test.jsonl"])
        assert len(rows) == 200
        assert all(row["split"] == "test" for row in rows)
        assert all(
            row["test_build_id"] == T050_TEST_ID and "dry_run_id" not in row
            for row in rows
        )

    release_root = tmp_path / "HallucinationDataset"
    report_path = tmp_path / "Dataset/reports/t050_test_build.json"
    train_sentinel = release_root / "records/train.jsonl"
    validation_sentinel = release_root / "records/validation.jsonl"
    train_sentinel.parent.mkdir(parents=True)
    train_sentinel.write_text("train-sentinel\n", encoding="utf-8")
    validation_sentinel.write_text("validation-sentinel\n", encoding="utf-8")

    assert (
        write_t050_test_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=test_build,
        )
        is test_build
    )
    assert (
        write_t050_test_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=test_build,
        )
        is test_build
    )
    assert train_sentinel.read_text(encoding="utf-8") == "train-sentinel\n"
    assert validation_sentinel.read_text(encoding="utf-8") == "validation-sentinel\n"
    assert json.loads(report_path.read_text(encoding="utf-8"))["all_pass"] is True

    (release_root / "records/test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SplitBuildError) as conflict:
        write_t050_test_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=test_build,
        )
    assert conflict.value.code == "TEST_ARTIFACT_CONFLICT"
