from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from molhallulens.builders.validation_split import (
    T049_H_COUNT,
    T049_N_COUNT,
    T049_ORIGIN_COUNT,
    T049_RECORD_COUNT,
    T049_VALIDATION_ID,
    ValidationBuildAttempt,
    ValidationSplitBuild,
    ValidationSplitBuildError,
    build_t049_validation_split,
    write_t049_validation_artifacts,
)
from molhallulens.domain import EditingSubtask, PropagationPolicy, VariantLabel


@pytest.fixture(scope="module")
def validation_build() -> ValidationSplitBuild:
    return build_t049_validation_split()


def _jsonl(payload: str) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in payload.splitlines())


def test_build_covers_exact_frozen_validation_manifest(
    validation_build: ValidationSplitBuild,
) -> None:
    assert len(validation_build.origins) == T049_ORIGIN_COUNT == 25
    assert len(validation_build.artifacts) == T049_RECORD_COUNT == 200
    assert Counter(
        origin.case.spec.normalized_subtask for origin in validation_build.origins
    ) == Counter(
        {
            EditingSubtask.ADD: 8,
            EditingSubtask.DELETE: 9,
            EditingSubtask.SUBSTITUTE: 8,
        }
    )
    manifest_validation = {
        row.anonymous_sample_id
        for row in validation_build.split_manifest.rows
        if row.split.value == "validation"
    }
    assert {
        origin.case.spec.origin_id for origin in validation_build.origins
    } == manifest_validation
    assert all(
        artifact.split.value == "validation" for artifact in validation_build.artifacts
    )


def test_every_origin_has_four_reciprocal_strict_pairs(
    validation_build: ValidationSplitBuild,
) -> None:
    record_ids: set[str] = set()
    for origin in validation_build.origins:
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


def test_global_variant_and_policy_balance(
    validation_build: ValidationSplitBuild,
) -> None:
    assert Counter(
        artifact.draft.variant_label for artifact in validation_build.artifacts
    ) == Counter(
        {
            VariantLabel.HALLUCINATED: T049_H_COUNT,
            VariantLabel.FAITHFUL: T049_N_COUNT,
        }
    )
    assert Counter(
        (artifact.draft.policy, artifact.draft.variant_label)
        for artifact in validation_build.artifacts
    ) == Counter(
        {
            (policy, label): 25
            for policy in PropagationPolicy
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        }
    )


def test_detector_oracle_state_token_and_provenance_are_isolated(
    validation_build: ValidationSplitBuild,
) -> None:
    dataset = validation_build.dataset_records()
    oracle = validation_build.oracle_records()
    states = validation_build.state_records()
    tokens = validation_build.token_records()
    provenance = validation_build.provenance_records()
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
        record["validation_build_id"] == T049_VALIDATION_ID
        and "dry_run_id" not in record
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
        and record["donor"]["recipient_split"] == "validation"
        and record["donor"]["verified_split_local"] is True
        for record in provenance
    )


def test_failed_recipe_attempt_is_a_zero_record_atomic_unit(
    validation_build: ValidationSplitBuild,
) -> None:
    first = validation_build.attempts[0]
    rejected = ValidationBuildAttempt(
        attempt_index=0,
        origin_id=first.origin_id,
        subtask=first.subtask,
        case_id=f"{first.case_id}.injected-rejection",
        status="rejected",
        emitted_record_count=0,
        error_code="INJECTED_STRICT_REJECTION",
        exception_type="RuntimeError",
    )
    shifted = tuple(
        replace(attempt, attempt_index=attempt.attempt_index + 1)
        for attempt in validation_build.attempts
    )
    rebuilt = replace(validation_build, attempts=(rejected, *shifted))
    assert rebuilt.artifacts == validation_build.artifacts
    assert rebuilt.attempts[0].emitted_record_count == 0
    same_origin = tuple(
        attempt
        for attempt in rebuilt.attempts[1:]
        if attempt.origin_id == rejected.origin_id
    )
    assert same_origin
    assert same_origin[-1].status == "accepted"
    assert all(
        attempt.emitted_record_count == 0
        for attempt in same_origin
        if attempt.status == "rejected"
    )


def test_reports_prove_strict_gates_and_validation_only_selection(
    validation_build: ValidationSplitBuild,
) -> None:
    report = validation_build.build_report()
    validation = validation_build.validation_report()
    selection = validation_build.selection_manifest()
    assert report["all_pass"] is True
    assert report["summary"]["origin_count"] == 25
    assert report["summary"]["record_count"] == 200
    assert report["strict_validation"] == {
        "artifact_gate_count": 800,
        "bundle_gate_count": 25,
        "all_pass": True,
    }
    assert report["isolation"]["candidate_selection_split"] == "validation"
    assert report["isolation"]["train_used_for_candidate_selection"] is False
    assert report["isolation"]["test_used_for_candidate_selection"] is False
    assert report["execution"]["production_chemdfm_r_weights_loaded"] is False
    assert report["execution"]["renderer"] == "deterministic-formal-v1"
    assert report["execution"]["explicit_digest_or_sha_verification_performed"] is False
    assert validation["all_pass"] is True
    assert validation["artifact_gate_count"] == 800
    assert validation["bundle_gate_count"] == 25
    assert selection["selection_split"] == "validation"
    assert selection["test_used_for_candidate_selection"] is False
    assert all(
        attempt.emitted_record_count == 0
        for attempt in validation_build.attempts
        if attempt.status == "rejected"
    )


def test_release_payload_inventory_and_atomic_writer(
    tmp_path: Path,
    validation_build: ValidationSplitBuild,
) -> None:
    payloads = validation_build.artifact_payloads()
    assert set(payloads) == {
        "records/validation.jsonl",
        "oracle/validation.jsonl",
        "state_graphs/validation.jsonl",
        "tokenized/chemdfm_r/validation.jsonl",
        "provenance/validation.jsonl",
        "reports/validation_selection_manifest.json",
        "reports/validation_validation_report.json",
        "reports/validation_backfill_ledger.jsonl",
        "reports/validation_build_report.json",
    }
    for family in (
        "records",
        "oracle",
        "state_graphs",
        "tokenized/chemdfm_r",
        "provenance",
    ):
        rows = _jsonl(payloads[f"{family}/validation.jsonl"])
        assert len(rows) == 200
        assert all(row["split"] == "validation" for row in rows)
        assert all(
            row["validation_build_id"] == T049_VALIDATION_ID and "dry_run_id" not in row
            for row in rows
        )

    release_root = tmp_path / "HallucinationDataset"
    report_path = tmp_path / "Dataset/reports/t049_validation_build.json"
    assert (
        write_t049_validation_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=validation_build,
        )
        is validation_build
    )
    assert (
        write_t049_validation_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=validation_build,
        )
        is validation_build
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["all_pass"] is True

    (release_root / "records/validation.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationSplitBuildError) as conflict:
        write_t049_validation_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=validation_build,
        )
    assert conflict.value.code == "VALIDATION_ARTIFACT_CONFLICT"
