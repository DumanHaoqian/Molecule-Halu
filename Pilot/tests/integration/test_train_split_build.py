from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from molhallulens.modules.release.train import (
    T048_H_COUNT,
    T048_N_COUNT,
    T048_ORIGIN_COUNT,
    T048_RECORD_COUNT,
    T048_TRAIN_ID,
    TrainBuildAttempt,
    TrainSplitBuild,
    TrainSplitBuildError,
    build_t048_train_split,
    write_t048_train_artifacts,
)
from molhallulens.core import EditingSubtask, PropagationPolicy, VariantLabel


@pytest.fixture(scope="module")
def train_build() -> TrainSplitBuild:
    return build_t048_train_split()


def _jsonl(payload: str) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in payload.splitlines())


def test_build_covers_exact_frozen_train_manifest(train_build: TrainSplitBuild) -> None:
    assert len(train_build.origins) == T048_ORIGIN_COUNT == 100
    assert len(train_build.artifacts) == T048_RECORD_COUNT == 800
    assert Counter(
        origin.case.spec.normalized_subtask for origin in train_build.origins
    ) == Counter(
        {
            EditingSubtask.ADD: 34,
            EditingSubtask.DELETE: 33,
            EditingSubtask.SUBSTITUTE: 33,
        }
    )
    manifest_train = {
        row.anonymous_sample_id
        for row in train_build.split_manifest.rows
        if row.split.value == "train"
    }
    assert {
        origin.case.spec.origin_id for origin in train_build.origins
    } == manifest_train
    assert all(artifact.split.value == "train" for artifact in train_build.artifacts)


def test_every_origin_has_four_reciprocal_strict_pairs(
    train_build: TrainSplitBuild,
) -> None:
    record_ids: set[str] = set()
    for origin in train_build.origins:
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


def test_global_variant_and_policy_balance(train_build: TrainSplitBuild) -> None:
    assert Counter(
        artifact.draft.variant_label for artifact in train_build.artifacts
    ) == Counter(
        {
            VariantLabel.HALLUCINATED: T048_H_COUNT,
            VariantLabel.FAITHFUL: T048_N_COUNT,
        }
    )
    assert Counter(
        (artifact.draft.policy, artifact.draft.variant_label)
        for artifact in train_build.artifacts
    ) == Counter(
        {
            (policy, label): 100
            for policy in PropagationPolicy
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        }
    )


def test_detector_oracle_state_token_and_provenance_are_isolated(
    train_build: TrainSplitBuild,
) -> None:
    dataset = train_build.dataset_records()
    oracle = train_build.oracle_records()
    states = train_build.state_records()
    tokens = train_build.token_records()
    provenance = train_build.provenance_records()
    assert (
        len(dataset)
        == len(oracle)
        == len(states)
        == len(tokens)
        == len(provenance)
        == 800
    )
    expected_ids = {record["record_id"] for record in dataset}
    assert all(
        {record["record_id"] for record in family} == expected_ids
        for family in (oracle, states, tokens, provenance)
    )
    assert all(
        record["train_build_id"] == T048_TRAIN_ID and "dry_run_id" not in record
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
        and record["donor"]["recipient_split"] == "train"
        and record["donor"]["verified_split_local"] is True
        for record in provenance
    )


def test_failed_recipe_attempt_is_a_zero_record_atomic_unit(
    train_build: TrainSplitBuild,
) -> None:
    first = train_build.attempts[0]
    rejected = TrainBuildAttempt(
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
        for attempt in train_build.attempts
    )
    rebuilt = replace(train_build, attempts=(rejected, *shifted))
    assert rebuilt.artifacts == train_build.artifacts
    assert rebuilt.attempts[0].emitted_record_count == 0
    assert rebuilt.attempts[1].origin_id == rejected.origin_id
    assert rebuilt.attempts[1].status == "accepted"


def test_reports_prove_all_strict_gates_and_train_only_selection(
    train_build: TrainSplitBuild,
) -> None:
    report = train_build.build_report()
    validation = train_build.validation_report()
    selection = train_build.selection_manifest()
    assert report["all_pass"] is True
    assert report["summary"]["origin_count"] == 100
    assert report["summary"]["record_count"] == 800
    assert report["strict_validation"] == {
        "artifact_gate_count": 3200,
        "bundle_gate_count": 100,
        "all_pass": True,
    }
    assert report["isolation"]["validation_used_for_candidate_selection"] is False
    assert report["isolation"]["test_used_for_candidate_selection"] is False
    assert report["execution"]["production_chemdfm_r_weights_loaded"] is False
    assert report["execution"]["renderer"] == "deterministic-formal-v1"
    assert report["execution"]["explicit_digest_or_sha_verification_performed"] is False
    assert validation["all_pass"] is True
    assert validation["artifact_gate_count"] == 3200
    assert validation["bundle_gate_count"] == 100
    assert selection["selection_split"] == "train"
    assert selection["validation_or_test_used_for_selection"] is False


def test_release_payload_inventory_and_atomic_writer(
    tmp_path: Path,
    train_build: TrainSplitBuild,
) -> None:
    payloads = train_build.artifact_payloads()
    assert set(payloads) == {
        "records/train.jsonl",
        "oracle/train.jsonl",
        "state_graphs/train.jsonl",
        "tokenized/chemdfm_r/train.jsonl",
        "provenance/train.jsonl",
        "reports/train_selection_manifest.json",
        "reports/train_validation_report.json",
        "reports/train_backfill_ledger.jsonl",
        "reports/train_build_report.json",
    }
    for family in (
        "records",
        "oracle",
        "state_graphs",
        "tokenized/chemdfm_r",
        "provenance",
    ):
        rows = _jsonl(payloads[f"{family}/train.jsonl"])
        assert len(rows) == 800
        assert all(row["split"] == "train" for row in rows)
        assert all(
            row["train_build_id"] == T048_TRAIN_ID and "dry_run_id" not in row
            for row in rows
        )

    release_root = tmp_path / "HallucinationDataset"
    report_path = tmp_path / "Dataset/reports/t048_train_build.json"
    assert (
        write_t048_train_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=train_build,
        )
        is train_build
    )
    assert (
        write_t048_train_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=train_build,
        )
        is train_build
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["all_pass"] is True

    (release_root / "records/train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TrainSplitBuildError) as conflict:
        write_t048_train_artifacts(
            release_root=release_root,
            report_path=report_path,
            build=train_build,
        )
    assert conflict.value.code == "TRAIN_ARTIFACT_CONFLICT"
