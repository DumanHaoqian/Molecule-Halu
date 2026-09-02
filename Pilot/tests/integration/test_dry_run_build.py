from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from molhallulens.modules.release.dry_run import (
    _FORBIDDEN_DATASET_KEYS,
    T045_DRY_RUN_ID,
    T045_ORIGIN_COUNT,
    T045_ORIGINS_PER_SUBTASK,
    T045_RECORD_COUNT,
    DryRunBuild,
    DryRunBuildError,
    _assert_no_forbidden_key,
    _assert_no_secret,
    _validate_donor_edge,
    build_t045_dry_run,
    write_t045_dry_run_artifacts,
)
from molhallulens.infrastructure.chemistry import isomeric_graph_equivalent
from molhallulens.core import EditingSubtask, PropagationPolicy, VariantLabel


@pytest.fixture(scope="module")
def dry_run() -> DryRunBuild:
    return build_t045_dry_run()


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_real_dry_run_builds_fifteen_complete_strict_origins(
    dry_run: DryRunBuild,
) -> None:
    assert len(dry_run.origins) == T045_ORIGIN_COUNT == 15
    assert len(dry_run.artifacts) == T045_RECORD_COUNT == 120
    assert Counter(
        origin.case.spec.normalized_subtask for origin in dry_run.origins
    ) == Counter(
        {
            EditingSubtask.ADD: T045_ORIGINS_PER_SUBTASK,
            EditingSubtask.DELETE: T045_ORIGINS_PER_SUBTASK,
            EditingSubtask.SUBSTITUTE: T045_ORIGINS_PER_SUBTASK,
        }
    )
    assert Counter(
        artifact.draft.variant_label for artifact in dry_run.artifacts
    ) == Counter({VariantLabel.HALLUCINATED: 60, VariantLabel.FAITHFUL: 60})

    record_ids = set()
    for origin in dry_run.origins:
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
        by_pair: dict[str, list[object]] = defaultdict(list)
        for artifact in origin.artifacts:
            assert artifact.record_id not in record_ids
            record_ids.add(artifact.record_id)
            by_pair[artifact.draft.pair_id].append(artifact)
        assert len(by_pair) == 4
        for pair in by_pair.values():
            assert len(pair) == 2
            left, right = pair
            assert left.draft.matched_record_id == right.record_id
            assert right.draft.matched_record_id == left.record_id


def test_backfill_ledger_discards_failed_origin_atomically(
    dry_run: DryRunBuild,
) -> None:
    rejected = tuple(
        attempt for attempt in dry_run.attempts if attempt.status == "rejected"
    )
    accepted = tuple(
        attempt for attempt in dry_run.attempts if attempt.status == "accepted"
    )
    assert len(rejected) == 2
    assert len(accepted) == 15
    assert {attempt.error_code for attempt in rejected} == {
        "GOLDEN_CANDIDATE_UNAVAILABLE"
    }
    assert all(attempt.emitted_record_count == 0 for attempt in rejected)
    assert all(attempt.replacement_origin_id for attempt in rejected)
    assert all(attempt.emitted_record_count == 8 for attempt in accepted)
    assert {
        replaced for attempt in accepted for replaced in attempt.replaced_origin_ids
    } == {attempt.origin_id for attempt in rejected}
    selected_ids = {origin.case.spec.origin_id for origin in dry_run.origins}
    assert not selected_ids & {attempt.origin_id for attempt in rejected}


def test_verified_manifest_and_split_local_donor_boundary(
    dry_run: DryRunBuild,
) -> None:
    for origin in dry_run.origins:
        row = dry_run.split_manifest.row_for_origin(origin.case.spec.origin_id)
        assert {
            (artifact.split, artifact.leakage_group_id) for artifact in origin.artifacts
        } == {(row.split, row.leakage_group_id)}

    selected = {
        origin.case.spec.origin_id: origin.artifacts[0].split.value
        for origin in dry_run.origins
    }
    test_origin = next(
        origin_id for origin_id, split in selected.items() if split == "test"
    )
    train_origin = next(
        origin_id for origin_id, split in selected.items() if split == "train"
    )
    with pytest.raises(DryRunBuildError) as caught:
        _validate_donor_edge(
            recipient_origin_id=test_origin,
            donor_origin_id=train_origin,
            split="test",
            manifest=dry_run.split_manifest,
            donor_pools=dry_run.donor_pools,
        )
    assert caught.value.code == "DRY_RUN_CROSS_SPLIT_DONOR"
    report = dry_run.build_report()
    assert report["acceptance"]["cross_split_donor_edge_count"] == 0


def test_cross_balance_selection_keeps_held_out_test_origins_frozen(
    dry_run: DryRunBuild,
) -> None:
    selected_by_split = defaultdict(set)
    for origin in dry_run.origins:
        selected_by_split[origin.artifacts[0].split.value].add(
            origin.case.spec.origin_id
        )

    assert selected_by_split["test"] == {
        "mol_edit.add_v2.0023",
        "mol_edit.delete_v2.0046",
        "mol_edit.substitute_v2.0057",
    }
    assert {
        "mol_edit.add_v2.0229",
        "mol_edit.add_v2.0040",
        "mol_edit.substitute_v2.0029",
        "mol_edit.substitute_v2.0032",
        "mol_edit.substitute_v2.0064",
    }.isdisjoint(origin.case.spec.origin_id for origin in dry_run.origins)


def test_public_records_are_oracle_separated_and_tokens_are_label_safe(
    dry_run: DryRunBuild,
) -> None:
    dataset = dry_run.dataset_records()
    oracle = dry_run.oracle_records()
    tokens = dry_run.token_records()
    provenance = dry_run.provenance_records()
    assert len(dataset) == len(oracle) == len(tokens) == len(provenance) == 120
    assert (
        {record["record_id"] for record in dataset}
        == {record["record_id"] for record in oracle}
        == {record["record_id"] for record in tokens}
        == {record["record_id"] for record in provenance}
    )
    assert all(
        set(record["detector_input"])
        == {"indexed_smiles", "instruction", "reasoning_chain", "final_answer"}
        for record in dataset
    )
    assert all(
        record["dataset_version"] == dry_run.dataset_version for record in dataset
    )
    assert all(record["visible_to_detector"] is False for record in oracle)
    assert all(record["gt_smiles"] for record in oracle)
    artifacts_by_id = {artifact.record_id: artifact for artifact in dry_run.artifacts}
    for record in dataset:
        artifact = artifacts_by_id[record["record_id"]]
        assert record["detector_input"]["instruction"] == (
            artifact.draft.locked_state.value_for("instruction").normalized_value
        )
        assert record["trace_labels"]["answer_correct"] is (
            isomeric_graph_equivalent(
                artifact.draft.locked_state.value_for("final_answer").normalized_value,
                artifact.draft.reference_graph.value_for("oracle_gt").normalized_value,
            )
        )

    labels_by_id = {
        artifact.record_id: artifact.draft.variant_label
        for artifact in dry_run.artifacts
    }
    for record in tokens:
        masks = (
            record["hallucination_core_mask"],
            record["error_any_mask"],
            record["local_falsehood_mask"],
            record["off_task_branch_mask"],
            *record["semantic_type_masks"].values(),
            *record["edit_subtype_masks"].values(),
            *record["causal_role_masks"].values(),
        )
        if labels_by_id[record["record_id"]] is VariantLabel.FAITHFUL:
            assert all(not any(mask) for mask in masks)
        else:
            assert any(record["error_any_mask"])
        assert len(record["input_ids"]) == len(record["offset_mapping"])
        assert record["activation_alignment"] == "post_token_h_t"
        assert (
            record["tokenizer_fingerprint"]["normalization_config"][
                "production_weights_loaded"
            ]
            is False
        )

    assert all(
        record["execution_mode"]["network_mode"] == "frozen_offline"
        and record["execution_mode"]["live_poe_attempted"] is False
        and record["execution_mode"]["live_availability_probe_performed"] is False
        and record["execution_mode"]["network_request_count"] == 0
        and record["execution_mode"]["provider"] is None
        and record["tokenizer"]["production_chemdfm_r_weights_loaded"] is False
        for record in provenance
    )


def test_nested_gt_and_secret_payloads_fail_closed(dry_run: DryRunBuild) -> None:
    leaked_dataset = copy.deepcopy(dry_run.dataset_records()[0])
    leaked_dataset["mutation"]["nested"] = {"oracle_gt": "hidden"}
    with pytest.raises(DryRunBuildError) as gt_error:
        _assert_no_forbidden_key(
            leaked_dataset,
            _FORBIDDEN_DATASET_KEYS,
            "DRY_RUN_GT_LEAKAGE",
        )
    assert gt_error.value.code == "DRY_RUN_GT_LEAKAGE"

    leaked_provenance = copy.deepcopy(dry_run.provenance_records()[0])
    leaked_provenance["private"] = {"Authorization": "Bearer definitely-not-allowed"}
    with pytest.raises(DryRunBuildError) as secret_error:
        _assert_no_secret(leaked_provenance)
    assert secret_error.value.code == "DRY_RUN_SECRET_KEY"


def test_release_payloads_have_complete_split_scoped_inventory(
    dry_run: DryRunBuild,
) -> None:
    payloads = dry_run.artifact_payloads()
    assert len(payloads) == 21
    for family in (
        "records",
        "oracle",
        "state_graphs",
        "tokenized/chemdfm_r",
        "provenance",
    ):
        for split, expected in (("train", 72), ("validation", 24), ("test", 24)):
            rows = tuple(
                json.loads(line)
                for line in payloads[f"{family}/{split}.jsonl"].splitlines()
            )
            assert len(rows) == expected
            assert all(row["split"] == split for row in rows)
    manifest = json.loads(payloads["dataset_manifest.json"])
    report = json.loads(payloads["reports/build_report.json"])
    assert manifest["dry_run_id"] == report["dry_run_id"] == T045_DRY_RUN_ID
    assert manifest["record_count"] == report["summary"]["record_count"] == 120
    assert report["all_pass"] is True
    assert report["execution"]["live_llm_material_participation_count"] == 0
    assert report["execution"]["content_hashes_added_by_t045"] is False


def test_writer_publishes_complete_release_and_is_replay_idempotent(
    tmp_path: Path,
    dry_run: DryRunBuild,
) -> None:
    root = tmp_path / "HallucinationDataset/dry_run"
    report = tmp_path / "Dataset/reports/t045_dry_run_build.json"
    assert (
        write_t045_dry_run_artifacts(
            dry_run_root=root,
            report_path=report,
            build=dry_run,
        )
        is dry_run
    )
    assert (
        write_t045_dry_run_artifacts(
            dry_run_root=root,
            report_path=report,
            build=dry_run,
        )
        is dry_run
    )
    assert json.loads(report.read_text(encoding="utf-8"))["all_pass"] is True
    assert len(_jsonl(root / "records/train.jsonl")) == 72
    assert len(_jsonl(root / "records/validation.jsonl")) == 24
    assert len(_jsonl(root / "records/test.jsonl")) == 24
    assert len(tuple(path for path in root.rglob("*") if path.is_file())) == 21

    (root / "records/train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DryRunBuildError) as conflict:
        write_t045_dry_run_artifacts(
            dry_run_root=root,
            report_path=report,
            build=dry_run,
        )
    assert conflict.value.code == "DRY_RUN_ARTIFACT_CONFLICT"
