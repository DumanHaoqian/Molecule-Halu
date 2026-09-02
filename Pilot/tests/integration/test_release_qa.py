from __future__ import annotations

import ast
import csv
import json
import shutil
import stat
from pathlib import Path

import pytest

from molhallulens.modules.release import qa as release_qa
from molhallulens.modules.release.qa import (
    ReleaseQAError,
    run_t052_release_qa,
    write_t052_release_artifacts,
)

MOCK_ACTIVATION_SHARD_COUNT = 3
MOCK_ACTIVATION_TOKEN_COUNT = 4800


@pytest.fixture(scope="module", autouse=True)
def _use_compact_activation_inventory_for_mock_release() -> object:
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        release_qa, "EXPECTED_ACTIVATION_SHARD_COUNT", MOCK_ACTIVATION_SHARD_COUNT
    )
    patcher.setattr(
        release_qa, "EXPECTED_ACTIVATION_TOKEN_COUNT", MOCK_ACTIVATION_TOKEN_COUNT
    )
    yield
    patcher.undo()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _serialized(detector: dict[str, str]) -> tuple[str, list[dict[str, object]]]:
    values = (
        ("indexed_smiles", "source", "<MOLECULE>"),
        ("instruction", "instruction", "<INSTRUCTION>"),
        ("reasoning_chain", "reasoning", "<REASONING>"),
        ("final_answer", "final_answer", "<FINAL_ANSWER>"),
    )
    parts: list[str] = []
    segments: list[dict[str, object]] = []
    cursor = 0
    for index, (field, kind, delimiter) in enumerate(values):
        if index:
            parts.append("\n\n")
            cursor += 2
        prefix = f"{delimiter}\n"
        parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        parts.append(detector[field])
        cursor += len(detector[field])
        segments.append(
            {
                "field_name": field,
                "segment_kind": kind,
                "start": start,
                "end": cursor,
            }
        )
    return "".join(parts), segments


def _fingerprint() -> dict[str, object]:
    return {
        "tokenizer_name": "ChemDFM-R-14B",
        "tokenizer_revision": "approved-local-checkpoint-no-digest",
        "tokenizer_vocab_hash": "not-computed-per-user-instruction",
        "special_token_config": {
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "unk_token_id": None,
        },
        "normalization_config": {
            "offset_unit": "python_char",
            "production_weights_loaded": True,
            "fast_tokenizer": True,
            "tokenizer_class": "MockFastTokenizer",
            "checkpoint_path": "/approved/ChemDFM-R-14B",
            "digest_computation_performed": False,
            "truncation": False,
            "padding": False,
        },
    }


def _record_family(
    *,
    split: str,
    origin_id: str,
    leakage_group_id: str,
    policy: str,
    label: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    pair_id = f"{origin_id}__{policy}"
    record_id = f"{pair_id}__{label}"
    matched = f"{pair_id}__{'N' if label == 'H' else 'H'}"
    bundle_id = f"{origin_id}__bundle"
    detector = {
        "indexed_smiles": "[CH3:1][CH3:2]",
        "instruction": "Keep the molecule unchanged for this QA fixture.",
        "reasoning_chain": 'Step 1 [PRODUCT_CONSTRUCTION]: CC\n  FORMAL: SOURCE --> PRODUCT_SMILES("CC")',
        "final_answer": "CC",
    }
    text, segments = _serialized(detector)
    reason_start = segments[2]["start"]
    role = (
        "TERMINAL"
        if policy == "TERMINAL"
        else "PROPAGATED_CONDITIONAL"
        if policy in {"PARTIAL", "FULL_CF"}
        else "ROOT"
    )
    spans = (
        [
            {
                "span_id": f"span:{record_id}",
                "component": "reasoning",
                "step_index": 1,
                "state_or_edge_id": "product",
                "literal_span": [reason_start, reason_start + 4],
                "claim_span": [reason_start, reason_start + 8],
                "semantic_types": [0, 2],
                "edit_subtypes": ["E06"],
                "evidence_relations": ["CONTRADICTS_REFERENCE_STATE"],
                "causal_role": role,
                "root_span_id": f"span:{record_id}",
            }
        ]
        if label == "H"
        else []
    )
    identity = {
        "schema_version": "molhallulens.edit.v1",
        "dataset_version": "pilot_v1",
        "record_id": record_id,
        "origin_id": origin_id,
        "pair_id": pair_id,
        "bundle_id": bundle_id,
        "leakage_group_id": leakage_group_id,
        "split": split,
    }
    record = {
        **identity,
        "family": "mol_edit",
        "subtask": "add",
        "detector_input": detector,
        "serialized": {
            "text": text,
            "sha256": "carried-source-identity",
            "segments": segments,
            "template_version": "detector_prompt_v1",
        },
        "variant": {
            "label": label,
            "propagation": policy,
            "matched_record_id": matched,
        },
        "mutation": {
            "candidate_id": f"candidate:{origin_id}:{policy}",
            "candidate_source": "RULE",
            "operator_family": "numeric_count_claim",
            "operator_id": "mol_edit.add.heavy_count_claim",
            "renderer_id": "formal-v1",
            "root_state_id": "product",
            "donor_origin_id": None,
            "fallback": None,
            "seed": 1,
        },
        "spans": spans,
        "trace_labels": {"hallucination_present": label == "H"},
        "verification": {
            "bundle_integrity_verified": True,
            "graph_edit_verified": True,
            "propagation_verified": True,
            "rdkit_sanitize": True,
            "renderer_verified": True,
            "span_verified": True,
            "token_alignment_verified": True,
        },
    }
    oracle = {
        **identity,
        "visible_to_detector": False,
        "gt_smiles": "CC",
        "reference_state_graph": {"nodes": {}},
        "candidate_state_graph": {"nodes": {}},
        "graph_delta": {},
    }
    state = {
        **identity,
        "artifact_scope": "build_only_non_detector",
        "reference": {},
        "formal_trace": {},
        "mutation_events": [],
        "semantic_difference_targets": [],
        "locked": {},
    }
    bit = 1 if label == "H" else 0
    token = {
        **identity,
        "activation_alignment": "post_token_h_t",
        "tokenizer_fingerprint": _fingerprint(),
        "serialized_text_sha256": "carried-source-identity",
        "input_ids": [10, 11, 12, 13],
        "attention_mask": [1, 1, 1, 1],
        "offset_mapping": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "segment_ids": ["reasoning"] * 4,
        "evaluation_mask": [1, 1, 1, 1],
        "hallucination_core_mask": [bit, 0, 0, 0],
        "error_any_mask": [bit, 0, 0, 0],
        "local_falsehood_mask": [bit, 0, 0, 0],
        "off_task_branch_mask": [0, 0, 0, 0],
        "reasoning_mask": [1, 1, 1, 1],
        "answer_mask": [0, 0, 0, 0],
        "boundary_ambiguous_mask": [0, 0, 0, 0],
        "error_char_fraction": [float(bit), 0.0, 0.0, 0.0],
        "semantic_type_masks": {
            "0": [bit, 0, 0, 0],
            "2": [bit, 0, 0, 0],
        },
        "edit_subtype_masks": {"E06": [bit, 0, 0, 0]},
        "causal_role_masks": {role: [bit, 0, 0, 0]},
        "matched_target_span": None,
    }
    provenance = {
        **identity,
        "artifact_scope": "private_build_provenance",
        "candidate_selection": {
            "selected_candidate_source": "RULE",
            "selected_from_complete_pool": True,
        },
        "donor": {
            "donor_origin_id": None,
            "donor_split": None,
            "pool_split": split,
            "recipient_split": split,
            "verified_split_local": True,
        },
        "execution_mode": {
            "network_mode": "frozen_offline",
            "live_poe_attempted": False,
            "live_availability_probe_performed": False,
            "network_request_count": 0,
            "provider": None,
            "requested_model_id": None,
            "response_model": None,
            "cache_keys": [],
        },
        "renderer": {"backend": "deterministic-formal-v1", "live_llm_used": False},
        "recipe": {"policy": policy},
        "propagation": {},
        "tokenizer": {"production_chemdfm_r_weights_loaded": True},
        "fallback": None,
    }
    return record, oracle, state, token, provenance


def _shortcut_report() -> dict[str, object]:
    baselines = {
        key: {"auroc": 0.5, "record_count": 1000}
        for key in (
            "metadata_only_logistic",
            "span_only_char_tfidf_logistic",
            "reasoning_only_word_tfidf_logistic",
            "nearest_neighbor_retrieval_k5",
            "smiles_validity",
            "visible_reasoning_answer_graph_comparator",
            "hidden_oracle_answer_graph_comparator",
        )
    }
    baselines["slices"] = {
        policy: {"auroc": 0.5, "record_count": 250}
        for policy in ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL")
    }
    return {
        "all_pass": True,
        "inventory": {
            "origin_count": 150,
            "record_count": 1200,
            "split_record_counts": {"train": 800, "validation": 200, "test": 200},
        },
        "development_inventory": {"origin_count": 125, "record_count": 1000},
        "audit_protocol": {
            "test_used_for_model_or_threshold_selection": False,
            "mandatory_gate_splits": ["train", "validation"],
        },
        "mandatory_gates": {
            "metadata_auroc": {
                "actual": 0.5,
                "comparator": "<=",
                "threshold": 0.55,
                "passed": True,
            },
            "span_only_tfidf_auroc": {
                "actual": 0.5,
                "comparator": "<=",
                "threshold": 0.55,
                "passed": True,
            },
            "reasoning_only_shallow_auroc": {
                "actual": 0.5,
                "comparator": "<=",
                "threshold": 0.6,
                "passed": True,
            },
            "token_length_standardized_difference": {
                "actual": 0.0,
                "comparator": "<",
                "threshold": 0.1,
                "passed": True,
            },
        },
        "baselines": baselines,
        "matching": {"length": {}, "style": {"mismatch_count": 0}},
        "heldout_test_diagnostics": {
            "record_count": 200,
            "origin_count": 25,
            "used_for_candidate_layer_or_threshold_selection": False,
        },
    }


def _make_release(project: Path) -> tuple[Path, dict[str, object]]:
    root = project / "HallucinationDataset"
    fingerprint = _fingerprint()
    split_specs = (("train", 100), ("validation", 25), ("test", 25))
    split_manifest_rows: list[dict[str, str]] = []
    record_ids_by_split: dict[str, list[str]] = {}
    token_shard_splits: dict[str, dict[str, object]] = {}
    build_contracts = {
        "train": (
            "train_build_id",
            "t048_frozen_train_100_origin_v1",
            "t048_train_build_report_v1",
            "t048_train_strict_validation_v1",
        ),
        "validation": (
            "validation_build_id",
            "t049_frozen_validation_25_origin_v1",
            "t049_validation_build_report_v1",
            "t049_validation_strict_validation_v1",
        ),
        "test": (
            "test_build_id",
            "t050_frozen_test_25_origin_v1",
            "t050_test_build_report_v1",
            "t050_test_strict_validation_v1",
        ),
    }
    artifact_validator_ids = (
        "molhallulens.validation.hallucination_semantics.v1",
        "molhallulens.validation.propagation.v1",
        "molhallulens.validation.renderer.v1",
        "molhallulens.validation.token_alignment.v1",
    )
    for split, origin_count in split_specs:
        records: list[dict[str, object]] = []
        oracle: list[dict[str, object]] = []
        states: list[dict[str, object]] = []
        tokens: list[dict[str, object]] = []
        provenance: list[dict[str, object]] = []
        for origin_index in range(origin_count):
            origin_id = f"mol_edit.add_v2.{split}.{origin_index:04d}"
            leakage = f"group-{split}-{origin_index:04d}"
            split_manifest_rows.append(
                {
                    "origin_id": f"anonymous-{split}-{origin_index:04d}",
                    "anonymous_sample_id": origin_id,
                    "leakage_group_id": leakage,
                    "subtask": "add",
                    "split": split,
                    "canonical_source_hash": "not-evaluated",
                    "canonical_gt_hash": "not-evaluated",
                    "scaffold_hash": "not-evaluated",
                    "split_seed": "1",
                    "dataset_version": "pilot_v1",
                }
            )
            for policy in ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL"):
                for label in ("H", "N"):
                    values = _record_family(
                        split=split,
                        origin_id=origin_id,
                        leakage_group_id=leakage,
                        policy=policy,
                        label=label,
                    )
                    for target, value in zip(
                        (records, oracle, states, tokens, provenance),
                        values,
                        strict=True,
                    ):
                        target.append(value)
        record_ids_by_split[split] = [str(row["record_id"]) for row in records]
        for directory, values in (
            ("records", records),
            ("oracle", oracle),
            ("state_graphs", states),
            ("tokenized/chemdfm_r", tokens),
            ("provenance", provenance),
        ):
            _jsonl(root / directory / f"{split}.jsonl", values)
        shard_path = root / f"tokenized/chemdfm_r/git_shards/{split}-00000.jsonl"
        _jsonl(shard_path, tokens)
        token_shard_splits[split] = {
            "canonical_relative_path": f"{split}.jsonl",
            "canonical_bytes": (root / f"tokenized/chemdfm_r/{split}.jsonl")
            .stat()
            .st_size,
            "record_count": len(tokens),
            "token_count": len(tokens) * 4,
            "shard_count": 1,
            "shards": [
                {
                    "order": 0,
                    "path": shard_path.name,
                    "bytes": shard_path.stat().st_size,
                    "row_count": len(tokens),
                    "first_record_id": tokens[0]["record_id"],
                    "last_record_id": tokens[-1]["record_id"],
                }
            ],
        }
        build_id_key, build_id, build_format, validation_format = build_contracts[split]
        _json(
            root / f"reports/{split}_build_report.json",
            {
                "all_pass": True,
                "format_version": build_format,
                build_id_key: build_id,
                "summary": {"origin_count": origin_count, "record_count": len(records)},
            },
        )
        validation_records = []
        for record in records:
            validation_records.append(
                {
                    **{
                        key: record[key]
                        for key in (
                            "record_id",
                            "origin_id",
                            "pair_id",
                            "bundle_id",
                            "leakage_group_id",
                            "split",
                            "dataset_version",
                        )
                    },
                    build_id_key: build_id,
                    "artifact_gates": [
                        {
                            "validator_id": validator_id,
                            "all_pass": True,
                            "issues": [],
                        }
                        for validator_id in artifact_validator_ids
                    ],
                    "artifact_chain": {
                        "validator_id": "molhallulens.validation.artifact_chain.v1",
                        "all_pass": True,
                        "issues": [],
                    },
                    "bundle_validator_id": (
                        "molhallulens.validation.bundle_integrity.v1"
                    ),
                    "bundle_all_pass": True,
                }
            )
        validation_origins = [
            {
                "origin_id": origin_id,
                "record_count": 8,
                "all_pass": True,
                "issue_codes": [],
                "subtask": "add",
            }
            for origin_id in sorted({str(record["origin_id"]) for record in records})
        ]
        _json(
            root / f"reports/{split}_validation_report.json",
            {
                "all_pass": True,
                "format_version": validation_format,
                build_id_key: build_id,
                "required_validator_ids": [
                    *artifact_validator_ids,
                    "molhallulens.validation.bundle_integrity.v1",
                ],
                "artifact_gate_count": len(records) * 4,
                "bundle_gate_count": origin_count,
                "records": validation_records,
                "origins": validation_origins,
            },
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(split_manifest_rows[0]))
        writer.writeheader()
        writer.writerows(split_manifest_rows)
    _json(
        root / "reports/test_isolation_declaration.json",
        {
            "format_version": "t050_test_isolation_declaration_v1",
            "all_pass": True,
            "build_order": {
                "required_predecessors_checked_before_test_construction": [
                    "T047",
                    "T048",
                    "T049",
                ],
                "test_built_last": True,
                "train_complete_before_test": True,
                "validation_complete_before_test": True,
            },
            "detector_scope": {
                "detector_layer_selected": False,
                "detector_threshold_selected": False,
                "formal_detector_training_authorized": False,
            },
            "failure_semantics": {
                "cross_split_backfill_allowed": False,
                "failed_attempt_emitted_record_count": 0,
            },
            "frozen_design": {
                "candidate_generation_rules_frozen_before_test_build": True,
                "operator_rules_frozen_before_test_build": True,
                "propagation_rules_frozen_before_test_build": True,
                "recipe_order_frozen_before_test_build": True,
                "renderer_rules_frozen_before_test_build": True,
                "thresholds_frozen_before_test_build": True,
                "test_failure_may_add_remove_or_reorder_recipe": False,
                "test_failure_may_change_operator_or_candidate_rule": False,
                "test_failure_may_change_propagation_or_renderer": False,
            },
            "test_usage": {
                "diagnostic_results_feed_back_into_build": False,
                "strict_acceptance_can_mutate_frozen_design": False,
                "used_for_candidate_generation_tuning": False,
                "used_for_candidate_rule_selection": False,
                "used_for_detector_layer_selection": False,
                "used_for_detector_threshold_selection": False,
                "used_for_propagation_layer_selection": False,
                "used_for_renderer_selection_or_tuning": False,
                "used_for_shortcut_threshold_selection": False,
                "used_for_strict_record_acceptance": True,
            },
        },
    )
    _json(
        root / "tokenized/chemdfm_r/manifest.json",
        {
            "format_version": "t051_real_tokenization_v1",
            "status": "complete",
            "mode": "release",
            "activation_alignment": "post_token_h_t",
            "label_shift": 0,
            "tokenizer_fingerprint": fingerprint,
            "splits": [
                {"split": split, "record_count": count * 8}
                for split, count in split_specs
            ],
            "record_count": 1200,
            "token_count": MOCK_ACTIVATION_TOKEN_COUNT,
            "all_token_arrays_equal_length": True,
            "git_shards": {
                "index_path": "git_shards/index.json",
                "max_shard_bytes_exclusive": 49_000_000,
                "canonical_storage": "server_only",
                "digest_computation_performed": False,
            },
        },
    )
    _json(
        root / "tokenized/chemdfm_r/git_shards/index.json",
        {
            "format_version": "t051_tokenized_git_shards_v1",
            "status": "complete",
            "canonical_storage": "server_only",
            "canonical_files_retained": True,
            "reconstruction": "byte_concatenation_in_index_order",
            "max_shard_bytes_exclusive": 49_000_000,
            "digest_computation_performed": False,
            "split_order": ["train", "validation", "test"],
            "splits": token_shard_splits,
            "record_count": 1200,
            "token_count": 4800,
        },
    )
    activation_root = root / "activations/chemdfm_r/layer_26"
    shards: list[dict[str, object]] = []
    split_summaries: dict[str, dict[str, int]] = {}
    for split, _origin_count in split_specs:
        record_ids = record_ids_by_split[split]
        token_counts = [4] * len(record_ids)
        offsets = [4 * index for index in range(len(record_ids) + 1)]
        tensor_path = activation_root / split / f"{split}-00000.pt"
        sidecar_path = activation_root / split / f"{split}-00000.json"
        tensor_relative_path = f"{split}/{tensor_path.name}"
        metadata_relative_path = f"{split}/{sidecar_path.name}"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path.write_bytes(b"x")
        _json(
            sidecar_path,
            {
                "format_version": "t051_chemdfm_r_post_token_v1",
                "status": "complete",
                "split": split,
                "shard_index": 0,
                "record_ids": record_ids,
                "token_counts": token_counts,
                "row_offsets": offsets,
                "activation_alignment": "post_token_h_t",
                "label_shift": 0,
                "layer_index": 26,
                "feature_name": "resid_post",
                "activation_dtype": "bfloat16",
                "tensor_relative_path": tensor_relative_path,
                "activation_shape": [len(record_ids) * 4, 5120],
                "file_bytes": 1,
                "digest_computation_performed": False,
            },
        )
        shards.append(
            {
                "split": split,
                "shard_index": 0,
                "tensor_path": tensor_relative_path,
                "metadata_path": metadata_relative_path,
                "record_count": len(record_ids),
                "token_count": len(record_ids) * 4,
                "hidden_size": 5120,
                "layer_index": 26,
                "resumed": False,
                "file_bytes": 1,
            }
        )
        split_summaries[split] = {
            "record_count": len(record_ids),
            "token_count": len(record_ids) * 4,
            "shard_count": 1,
        }
    _json(
        activation_root / "manifest.json",
        {
            "format_version": "t051_activation_manifest_v1",
            "status": "complete",
            "mode": "release",
            "model": {
                "hidden_size": 5120,
                "digest_computation_performed": False,
            },
            "feature": {
                "name": "resid_post",
                "layer_index": 26,
                "selected_using_pilot_records": False,
            },
            "alignment": {
                "activation_alignment": "post_token_h_t",
                "label_shift": 0,
                "hidden_token_axis_equals_label_length": True,
                "pre_token_claimed": False,
            },
            "tokenizer_fingerprint": fingerprint,
            "splits": split_summaries,
            "record_count": 1200,
            "token_count": MOCK_ACTIVATION_TOKEN_COUNT,
            "shard_count": MOCK_ACTIVATION_SHARD_COUNT,
            "tensor_payload_validation": {
                "performed": True,
                "all_pass": True,
                "shard_count": MOCK_ACTIVATION_SHARD_COUNT,
                "record_count": 1200,
                "token_count": MOCK_ACTIVATION_TOKEN_COUNT,
                "activation_dtype": "bfloat16",
                "ordered_record_ids_exact": True,
                "token_counts_exact": True,
                "row_offsets_exact": True,
                "activation_shape_exact": True,
                "activation_alignment_exact": True,
                "layer_index_exact": True,
                "digest_verification_performed": False,
            },
            "shards": shards,
        },
    )
    _json(
        project / "Dataset/reports/poe_capability_probe.json",
        {
            "required_model_id": "gpt-5.4-mini",
            "deterministic_mock_validation": {
                "execution_status": "passed",
                "requested_model_id": "gpt-5.4-mini",
            },
            "live_probe": {
                "execution_status": "offline_not_executed",
                "reason_code": "POE_TRANSPORT_FAILED",
            },
        },
    )
    _json(
        project / "Dataset/reports/t044_golden_validation.json",
        {
            "format_version": "t044_golden_validation_v1",
            "all_pass": True,
            "summary": {
                "complete_real_origin_count": 9,
                "record_count": 72,
                "live_poe_attempt_count": 0,
            },
            "origins": [
                {
                    "origin_id": f"golden-origin-{index}",
                    "record_count": 8,
                    "all_pass": True,
                    "issue_codes": [],
                }
                for index in range(9)
            ],
        },
    )
    _json(
        project / "tests/golden/t044_extended_golden_suite.json",
        {
            "format_version": "t044_extended_golden_suite_v1",
            "dataset_version": "pilot_v1",
            "coverage": {
                "complete_real_origin_count": 9,
                "complete_record_count": 72,
            },
            "execution": {
                "deterministic_replay": True,
                "live_poe_attempted": False,
            },
            "origin_bundles": [
                {
                    "origin_id": f"golden-origin-{index}",
                    "records": [{} for _ in range(8)],
                }
                for index in range(9)
            ],
        },
    )
    return root, _shortcut_report()


@pytest.fixture(scope="module")
def complete_release(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[str, object]]:
    project = tmp_path_factory.mktemp("t052-project")
    root, shortcut = _make_release(project)
    return project, root, shortcut


def test_release_qa_source_never_imports_or_calls_digest_implementations() -> None:
    source = Path(release_qa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "hashlib" not in imported
    assert "torch" not in imported
    assert not {"md5", "sha1", "sha256", "digest", "hexdigest"} & called
    integer_constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"EXPECTED_ACTIVATION_SHARD_COUNT", "EXPECTED_ACTIVATION_TOKEN_COUNT"}
    }
    assert integer_constants == {
        "EXPECTED_ACTIVATION_SHARD_COUNT": 401,
        "EXPECTED_ACTIVATION_TOKEN_COUNT": 1_824_606,
    }


def test_complete_release_passes_and_publishes_all_t052_outputs(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, root, shortcut = complete_release
    private = tmp_path / "local-private/poe_usage_ledger.json"
    result = run_t052_release_qa(
        release_root=root,
        project_root=project,
        private_ledger_path=private,
        shortcut_runner=lambda _root: shortcut,
    )
    assert result.validation_report["all_effective_gates_pass"] is True
    assert result.validation_report["strict_validation"] == {
        "artifact_gate_count": 4800,
        "bundle_gate_count": 150,
        "all_pass": True,
    }
    override = result.dataset_manifest["identity_override"]
    assert override["status"] == "overridden_not_evaluated"
    assert override["passed"] is None
    assert override["checkpoint_tokenizer_identity_scope"] == {
        "evidence": "approved_local_path_plus_qwen2_configuration",
        "provenance_claim_only": True,
        "byte_exact_identity_proven": False,
        "byte_exact_reproducibility_claimed": False,
    }
    assert result.dataset_manifest["summary"]["split_record_counts"] == {
        "train": 800,
        "validation": 200,
        "test": 200,
    }
    external = project / "Dataset/reports/t052_release_qa.json"
    assert (
        write_t052_release_artifacts(
            release_root=root,
            project_root=project,
            external_report_path=external,
            private_ledger_path=private,
            result=result,
        )
        is result
    )
    assert (
        write_t052_release_artifacts(
            release_root=root,
            project_root=project,
            external_report_path=external,
            private_ledger_path=private,
            result=result,
        )
        is result
    )
    assert (root / "dataset_manifest.json").is_file()
    assert (root / "reports/release_validation_report.json").is_file()
    assert (root / "reports/shortcut_baseline_report.json").is_file()
    assert (root / "DATASET_CARD.md").is_file()
    assert (root / "KNOWN_LIMITATIONS.md").is_file()
    limitations = (root / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8")
    assert "do not establish byte-exact checkpoint or tokenizer identity" in limitations
    assert (
        "does not claim byte-exact checkpoint/tokenizer reproducibility" in limitations
    )
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    ledger = json.loads(private.read_text(encoding="utf-8"))
    assert ledger["network_request_count"] == 0
    assert ledger["records"] == []
    assert not (root / "private/poe_usage_ledger.json").exists()
    descriptor_path = root / "reports/poe_usage_ledger_export_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor == result.poe_ledger_descriptor
    assert descriptor["external_ledger_path"] == str(private.resolve())
    assert descriptor["network_request_count"] == 0
    assert descriptor["recorded_cost_points"] == 0.0
    assert descriptor["secret_free"] is True
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert all(
        not Path(str(item["path"])).is_absolute()
        and ".." not in Path(str(item["path"])).parts
        for item in manifest["artifact_inventory"]
    )
    assert manifest["reports"]["poe_usage_ledger"] == {
        "storage": "external_owner_only",
        "external_path": str(private.resolve()),
        "required_mode": "0600",
        "public_descriptor": "reports/poe_usage_ledger_export_descriptor.json",
        "project_contains_private_ledger": False,
    }
    assert external.is_file()


def test_ignored_mode_on_9p_style_storage_fails_then_env_fallback_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    root, shortcut = _make_release(project)
    bad_private = tmp_path / "nine-p-mount/poe_usage_ledger.json"
    bad_result = run_t052_release_qa(
        release_root=root,
        project_root=project,
        private_ledger_path=bad_private,
        shortcut_runner=lambda _root: shortcut,
    )
    bad_private.parent.mkdir(parents=True)
    bad_private.write_text(bad_result.private_payload(), encoding="utf-8")
    bad_private.chmod(0o644)
    real_chmod = release_qa.os.chmod

    def ignored_on_nine_p(path: object, mode: int) -> None:
        if Path(path).resolve() == bad_private.resolve():
            return
        real_chmod(path, mode)

    monkeypatch.setattr(release_qa.os, "chmod", ignored_on_nine_p)
    with pytest.raises(ReleaseQAError) as ignored_mode:
        write_t052_release_artifacts(
            release_root=root,
            project_root=project,
            private_ledger_path=bad_private,
            shortcut_runner=lambda _root: shortcut,
            result=bad_result,
        )
    assert ignored_mode.value.code == "RELEASE_QA_PRIVATE_MODE"
    assert not (root / "dataset_manifest.json").exists()
    assert not (root / "reports/poe_usage_ledger_export_descriptor.json").exists()

    good_private = tmp_path / "local-fs/poe_usage_ledger.json"
    monkeypatch.setenv("MOLHALLULENS_PRIVATE_LEDGER_PATH", str(good_private))
    result = write_t052_release_artifacts(
        release_root=root,
        project_root=project,
        shortcut_runner=lambda _root: shortcut,
    )
    assert good_private.is_file()
    assert stat.S_IMODE(good_private.stat().st_mode) == 0o600
    descriptor = json.loads(
        (root / "reports/poe_usage_ledger_export_descriptor.json").read_text(
            encoding="utf-8"
        )
    )
    assert descriptor["external_ledger_path"] == str(good_private.resolve())
    assert result.dataset_manifest["reports"]["poe_usage_ledger"][
        "external_path"
    ] == str(good_private.resolve())


def test_ordered_git_shards_replace_missing_canonical_token_files(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release
    shard_project = tmp_path / "shard-fallback"
    shutil.copytree(project, shard_project)
    shard_root = shard_project / "HallucinationDataset"
    for split in ("train", "validation", "test"):
        (shard_root / f"tokenized/chemdfm_r/{split}.jsonl").unlink()
    result = run_t052_release_qa(
        release_root=shard_root,
        project_root=shard_project,
        shortcut_runner=lambda _release_root: shortcut,
    )
    assert result.validation_report["all_effective_gates_pass"] is True
    shard_paths = {
        item["path"]
        for item in result.dataset_manifest["artifact_inventory"]
        if item["artifact_family"] == "tokenized_git_shard"
    }
    assert len(shard_paths) == 3
    assert all(not Path(path).is_absolute() for path in shard_paths)


def test_token_git_shard_paths_reject_escape_symlink_and_duplicate(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release

    absolute_project = tmp_path / "token-absolute"
    shutil.copytree(project, absolute_project)
    absolute_root = absolute_project / "HallucinationDataset"
    index_path = absolute_root / "tokenized/chemdfm_r/git_shards/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = absolute_root / "tokenized/chemdfm_r/git_shards/train-00000.jsonl"
    index["splits"]["train"]["shards"][0]["path"] = str(shard.resolve())
    _json(index_path, index)
    with pytest.raises(ReleaseQAError) as absolute:
        run_t052_release_qa(
            release_root=absolute_root,
            project_root=absolute_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert absolute.value.code == "RELEASE_TOKEN_SHARD_PATH"

    traversal_project = tmp_path / "token-traversal"
    shutil.copytree(project, traversal_project)
    traversal_root = traversal_project / "HallucinationDataset"
    index_path = traversal_root / "tokenized/chemdfm_r/git_shards/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["splits"]["train"]["shards"][0]["path"] = "../train.jsonl"
    _json(index_path, index)
    with pytest.raises(ReleaseQAError) as traversal:
        run_t052_release_qa(
            release_root=traversal_root,
            project_root=traversal_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert traversal.value.code == "RELEASE_TOKEN_SHARD_PATH"

    symlink_project = tmp_path / "token-symlink"
    shutil.copytree(project, symlink_project)
    symlink_root = symlink_project / "HallucinationDataset"
    shard_root = symlink_root / "tokenized/chemdfm_r/git_shards"
    outside = tmp_path / "outside-token-shard.jsonl"
    shutil.copyfile(shard_root / "train-00000.jsonl", outside)
    (shard_root / "escape.jsonl").symlink_to(outside)
    index_path = shard_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["splits"]["train"]["shards"][0]["path"] = "escape.jsonl"
    _json(index_path, index)
    with pytest.raises(ReleaseQAError) as symlink:
        run_t052_release_qa(
            release_root=symlink_root,
            project_root=symlink_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert symlink.value.code == "RELEASE_TOKEN_SHARD_PATH"

    duplicate_project = tmp_path / "token-duplicate"
    shutil.copytree(project, duplicate_project)
    duplicate_root = duplicate_project / "HallucinationDataset"
    index_path = duplicate_root / "tokenized/chemdfm_r/git_shards/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["splits"]["validation"]["shards"][0]["path"] = "train-00000.jsonl"
    _json(index_path, index)
    with pytest.raises(ReleaseQAError) as duplicate:
        run_t052_release_qa(
            release_root=duplicate_root,
            project_root=duplicate_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert duplicate.value.code == "RELEASE_TOKEN_SHARD_PATH_DUPLICATE"


def test_tensor_payload_audit_and_relative_activation_paths_fail_closed(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release
    audit_project = tmp_path / "missing-tensor-audit"
    shutil.copytree(project, audit_project)
    audit_root = audit_project / "HallucinationDataset"
    manifest_path = audit_root / "activations/chemdfm_r/layer_26/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_payload_validation"]["performed"] = False
    _json(manifest_path, manifest)
    with pytest.raises(ReleaseQAError) as missing_audit:
        run_t052_release_qa(
            release_root=audit_root,
            project_root=audit_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert missing_audit.value.code == "RELEASE_ACTIVATION_MANIFEST"

    absolute_project = tmp_path / "absolute-activation-path"
    shutil.copytree(project, absolute_project)
    absolute_root = absolute_project / "HallucinationDataset"
    manifest_path = absolute_root / "activations/chemdfm_r/layer_26/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = Path(manifest["shards"][0]["tensor_path"])
    manifest["shards"][0]["tensor_path"] = str(
        (manifest_path.parent / relative).resolve()
    )
    _json(manifest_path, manifest)
    with pytest.raises(ReleaseQAError) as absolute_path:
        run_t052_release_qa(
            release_root=absolute_root,
            project_root=absolute_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert absolute_path.value.code == "RELEASE_ACTIVATION_PATH"


def test_activation_sidecars_bind_split_and_exact_ordered_token_axis(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release

    cross_split_project = tmp_path / "activation-cross-split"
    shutil.copytree(project, cross_split_project)
    cross_root = cross_split_project / "HallucinationDataset"
    activation_root = cross_root / "activations/chemdfm_r/layer_26"
    train_path = activation_root / "train/train-00000.json"
    test_path = activation_root / "test/test-00000.json"
    train_sidecar = json.loads(train_path.read_text(encoding="utf-8"))
    test_sidecar = json.loads(test_path.read_text(encoding="utf-8"))
    train_sidecar["record_ids"][0] = test_sidecar["record_ids"][0]
    _json(train_path, train_sidecar)
    with pytest.raises(ReleaseQAError) as cross_split:
        run_t052_release_qa(
            release_root=cross_root,
            project_root=cross_split_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert cross_split.value.code == "RELEASE_ACTIVATION_SPLIT_AXIS"

    reordered_project = tmp_path / "activation-reordered"
    shutil.copytree(project, reordered_project)
    reordered_root = reordered_project / "HallucinationDataset"
    sidecar_path = (
        reordered_root / "activations/chemdfm_r/layer_26/train/train-00000.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["record_ids"][0], sidecar["record_ids"][1] = (
        sidecar["record_ids"][1],
        sidecar["record_ids"][0],
    )
    _json(sidecar_path, sidecar)
    with pytest.raises(ReleaseQAError) as reordered:
        run_t052_release_qa(
            release_root=reordered_root,
            project_root=reordered_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert reordered.value.code == "RELEASE_ACTIVATION_ORDER"

    wrong_split_project = tmp_path / "activation-sidecar-split"
    shutil.copytree(project, wrong_split_project)
    wrong_split_root = wrong_split_project / "HallucinationDataset"
    sidecar_path = (
        wrong_split_root / "activations/chemdfm_r/layer_26/train/train-00000.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["split"] = "test"
    _json(sidecar_path, sidecar)
    with pytest.raises(ReleaseQAError) as wrong_split:
        run_t052_release_qa(
            release_root=wrong_split_root,
            project_root=wrong_split_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert wrong_split.value.code == "RELEASE_ACTIVATION_SIDECAR"


def test_public_oracle_leak_and_token_axis_drift_fail_closed(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release
    leaked_project = tmp_path / "leaked"
    shutil.copytree(project, leaked_project)
    leaked_root = leaked_project / "HallucinationDataset"
    record_path = leaked_root / "records/test.jsonl"
    rows = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["gt_smiles"] = "CC"
    _jsonl(record_path, rows)
    with pytest.raises(ReleaseQAError) as leaked:
        run_t052_release_qa(
            release_root=leaked_root,
            project_root=leaked_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert leaked.value.code == "RELEASE_GT_BOUNDARY"

    drift_project = tmp_path / "drift"
    shutil.copytree(project, drift_project)
    drift_root = drift_project / "HallucinationDataset"
    token_path = drift_root / "tokenized/chemdfm_r/test.jsonl"
    token_rows = [
        json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()
    ]
    token_rows[0]["evaluation_mask"] = [1, 1, 1]
    _jsonl(token_path, token_rows)
    with pytest.raises(ReleaseQAError) as drifted:
        run_t052_release_qa(
            release_root=drift_root,
            project_root=drift_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert drifted.value.code in {
        "RELEASE_TOKEN_LENGTH",
        "RELEASE_TOKEN_CANONICAL_SHARD_MISMATCH",
    }


def test_t043_evidence_and_record_verification_are_exact(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release

    fake_verification_project = tmp_path / "fake-record-verification"
    shutil.copytree(project, fake_verification_project)
    fake_root = fake_verification_project / "HallucinationDataset"
    record_path = fake_root / "records/train.jsonl"
    rows = [json.loads(line) for line in record_path.read_text().splitlines()]
    rows[0]["verification"] = {"made_up_gate": True}
    _jsonl(record_path, rows)
    with pytest.raises(ReleaseQAError) as fake_verification:
        run_t052_release_qa(
            release_root=fake_root,
            project_root=fake_verification_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert fake_verification.value.code == "RELEASE_STRICT_RECORD"

    missing_gate_project = tmp_path / "missing-t043-gate"
    shutil.copytree(project, missing_gate_project)
    missing_gate_root = missing_gate_project / "HallucinationDataset"
    report_path = missing_gate_root / "reports/train_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["records"][0]["artifact_gates"].pop()
    _json(report_path, report)
    with pytest.raises(ReleaseQAError) as missing_gate:
        run_t052_release_qa(
            release_root=missing_gate_root,
            project_root=missing_gate_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert missing_gate.value.code == "RELEASE_STRICT_REPORT"

    stale_report_project = tmp_path / "stale-t043-report"
    shutil.copytree(project, stale_report_project)
    stale_root = stale_report_project / "HallucinationDataset"
    report_path = stale_root / "reports/test_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["format_version"] = "stale-report"
    _json(report_path, report)
    with pytest.raises(ReleaseQAError) as stale_report:
        run_t052_release_qa(
            release_root=stale_root,
            project_root=stale_report_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert stale_report.value.code == "RELEASE_STRICT_REPORT"


def test_test_feedback_and_shortcut_findings_are_fail_closed_and_disclosed(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, root, shortcut = complete_release
    isolated_project = tmp_path / "isolation"
    shutil.copytree(project, isolated_project)
    isolated_root = isolated_project / "HallucinationDataset"
    declaration_path = isolated_root / "reports/test_isolation_declaration.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    declaration["test_usage"]["used_for_detector_threshold_selection"] = True
    _json(declaration_path, declaration)
    with pytest.raises(ReleaseQAError) as isolation:
        run_t052_release_qa(
            release_root=isolated_root,
            project_root=isolated_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert isolation.value.code == "RELEASE_TEST_ISOLATION"

    missing_key_project = tmp_path / "isolation-missing-key"
    shutil.copytree(project, missing_key_project)
    missing_key_root = missing_key_project / "HallucinationDataset"
    declaration_path = missing_key_root / "reports/test_isolation_declaration.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    declaration["test_usage"].pop("used_for_detector_layer_selection")
    _json(declaration_path, declaration)
    with pytest.raises(ReleaseQAError) as missing_key:
        run_t052_release_qa(
            release_root=missing_key_root,
            project_root=missing_key_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert missing_key.value.code == "RELEASE_TEST_ISOLATION"

    unfrozen_project = tmp_path / "isolation-unfrozen"
    shutil.copytree(project, unfrozen_project)
    unfrozen_root = unfrozen_project / "HallucinationDataset"
    declaration_path = unfrozen_root / "reports/test_isolation_declaration.json"
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    declaration["frozen_design"][
        "candidate_generation_rules_frozen_before_test_build"
    ] = False
    _json(declaration_path, declaration)
    with pytest.raises(ReleaseQAError) as unfrozen:
        run_t052_release_qa(
            release_root=unfrozen_root,
            project_root=unfrozen_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert unfrozen.value.code == "RELEASE_TEST_ISOLATION"

    inconsistent = json.loads(json.dumps(shortcut))
    inconsistent["mandatory_gates"]["metadata_auroc"]["passed"] = False
    with pytest.raises(ReleaseQAError) as shortcut_failure:
        run_t052_release_qa(
            release_root=root,
            project_root=project,
            shortcut_runner=lambda _root: inconsistent,
        )
    assert shortcut_failure.value.code == "RELEASE_SHORTCUT_REPORT"

    disclosed = json.loads(json.dumps(inconsistent))
    disclosed["mandatory_gates"]["metadata_auroc"]["actual"] = 0.6
    disclosed["all_pass"] = False
    result = run_t052_release_qa(
        release_root=root,
        project_root=project,
        shortcut_runner=lambda _root: disclosed,
    )
    assert result.validation_report["all_effective_gates_pass"] is True
    assert result.shortcut_report["all_pass"] is False
    assert result.shortcut_report["recommended_go_no_go_all_pass"] is False
    assert result.shortcut_report["failed_recommended_gates"] == ("metadata_auroc",)
    assert result.dataset_manifest["shortcut_diagnostics"] == {
        "audit_completed": True,
        "recommended_go_no_go_all_pass": False,
        "failed_recommended_gates": ("metadata_auroc",),
        "release_acceptance_role": (
            "diagnostic_disclosure_not_a_threshold_gate_in_section_19"
        ),
    }
    assert "metadata_auroc=0.6 (<= 0.55)" in result.known_limitations
    assert "did not pass" in result.dataset_card


def test_t044_replay_fixture_must_bind_reported_72_records(
    complete_release: tuple[Path, Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    project, _root, shortcut = complete_release
    replay_project = tmp_path / "incomplete-t044"
    shutil.copytree(project, replay_project)
    replay_root = replay_project / "HallucinationDataset"
    fixture_path = replay_project / "tests/golden/t044_extended_golden_suite.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["origin_bundles"][0]["records"].pop()
    _json(fixture_path, fixture)
    with pytest.raises(ReleaseQAError) as incomplete:
        run_t052_release_qa(
            release_root=replay_root,
            project_root=replay_project,
            shortcut_runner=lambda _root: shortcut,
        )
    assert incomplete.value.code == "RELEASE_T044_REPLAY_EVIDENCE"


def test_atomic_publish_rolls_back_every_new_output_on_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "atomic-project"
    root, shortcut = _make_release(project)
    private = tmp_path / "atomic-private/poe_usage_ledger.json"
    external = project / "Dataset/reports/t052_release_qa.json"
    result = run_t052_release_qa(
        release_root=root,
        project_root=project,
        private_ledger_path=private,
        shortcut_runner=lambda _root: shortcut,
    )
    real_replace = Path.replace

    def fail_known_limitations(staged: Path, target: Path) -> Path:
        if Path(target).name == "KNOWN_LIMITATIONS.md":
            raise OSError("injected install failure")
        return real_replace(staged, target)

    monkeypatch.setattr(Path, "replace", fail_known_limitations)
    with pytest.raises(ReleaseQAError) as failed:
        write_t052_release_artifacts(
            release_root=root,
            project_root=project,
            external_report_path=external,
            private_ledger_path=private,
            result=result,
        )
    assert failed.value.code == "RELEASE_QA_ATOMIC_PUBLISH"
    assert not private.exists()
    assert not external.exists()
    assert all(not (root / relative).exists() for relative in result.public_payloads())
    assert not tuple(root.glob(".t052-staging-*"))

    monkeypatch.setattr(Path, "replace", real_replace)
    assert (
        write_t052_release_artifacts(
            release_root=root,
            project_root=project,
            external_report_path=external,
            private_ledger_path=private,
            result=result,
        )
        is result
    )
