"""T051 real-tokenizer projection and post-token activation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molhallulens.modules.release.chemdfm import (
    ACTIVATION_ALIGNMENT,
    EXPECTED_HIDDEN_SIZE,
    FROZEN_LAYER_INDEX,
    T051_ACTIVATION_MANIFEST_VERSION,
    T051ArtifactError,
    _manifest_member_path,
    _resume_metadata,
    assert_post_token_axis,
    finalize_activation_inventory,
    iter_git_tokenized_rows,
    plan_activation_shards,
    reassemble_git_tokenized_split,
    tokenize_records,
    tokenize_release,
    validate_git_shard_inventory,
    validate_tokenized_row,
    write_git_tokenized_shards,
)


class _CharacterFastTokenizer:
    is_fast = True
    bos_token_id = None
    eos_token_id = 151645
    pad_token_id = 151643
    unk_token_id = None

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "truncation": False,
            "padding": False,
        }
        count = len(text)
        return {
            "input_ids": [1000 + index for index in range(count)],
            "attention_mask": [1] * count,
            "offset_mapping": [(index, index + 1) for index in range(count)],
            "special_tokens_mask": [0] * count,
        }


def _serialized(
    *,
    reasoning: str,
    answer: str,
) -> tuple[str, list[dict[str, object]]]:
    values = (
        ("indexed_smiles", "source", "CC"),
        ("instruction", "instruction", "Edit the molecule."),
        ("reasoning_chain", "reasoning", reasoning),
        ("final_answer", "final_answer", answer),
    )
    delimiters = {
        "indexed_smiles": "<MOLECULE>",
        "instruction": "<INSTRUCTION>",
        "reasoning_chain": "<REASONING>",
        "final_answer": "<FINAL_ANSWER>",
    }
    parts: list[str] = []
    segments: list[dict[str, object]] = []
    cursor = 0
    for index, (field_name, kind, value) in enumerate(values):
        if index:
            parts.append("\n\n")
            cursor += 2
        prefix = f"{delimiters[field_name]}\n"
        parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        parts.append(value)
        cursor += len(value)
        segments.append(
            {
                "field_name": field_name,
                "segment_kind": kind,
                "start": start,
                "end": cursor,
            }
        )
    return "".join(parts), segments


def _record(
    record_id: str,
    label: str,
    *,
    split: str = "train",
) -> dict[str, object]:
    reasoning = "The product has 9 heavy atoms."
    answer = "CN"
    text, segments = _serialized(reasoning=reasoning, answer=answer)
    literal_start = text.index("9 heavy")
    spans: list[dict[str, object]] = []
    if label == "H":
        spans.append(
            {
                "span_id": f"span:{record_id}",
                "component": "reasoning",
                "step_index": 0,
                "state_or_edge_id": "product_heavy",
                "literal_span": [literal_start, literal_start + 1],
                "claim_span": [literal_start, literal_start + len("9 heavy atoms")],
                "semantic_types": [0, 2],
                "edit_subtypes": ["E06", "E07"],
                "evidence_relations": ["CONTRADICTS_REFERENCE_STATE"],
                "causal_role": "ROOT",
                "root_span_id": f"span:{record_id}",
            }
        )
    return {
        "schema_version": "molhallulens.edit.v1",
        "dataset_version": "pilot_v1",
        "record_id": record_id,
        "origin_id": "mol_edit.add_v2.0001",
        "pair_id": "mol_edit.add_v2.0001__LOCAL",
        "bundle_id": "mol_edit.add_v2.0001__bundle",
        "leakage_group_id": "group-1",
        "split": split,
        "variant": {
            "label": label,
            "propagation": "LOCAL",
            "matched_record_id": record_id[:-1] + ("N" if label == "H" else "H"),
        },
        "serialized": {
            "text": text,
            "sha256": f"carried-text-identity-{label}",
            "segments": segments,
            "template_version": "detector_prompt_v1",
        },
        "spans": spans,
    }


def _prior(record: dict[str, object]) -> dict[str, object]:
    row = {
        key: record[key]
        for key in (
            "schema_version",
            "dataset_version",
            "record_id",
            "origin_id",
            "pair_id",
            "bundle_id",
            "leakage_group_id",
            "split",
        )
    }
    row["train_build_id"] = "t048-test"
    row["matched_target_span"] = (
        None
        if record["variant"]["label"] == "H"
        else list(record["serialized"]["segments"][2].values())[-2:]
    )
    return row


def _rows() -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    records = (
        _record("mol_edit.add_v2.0001__LOCAL__H", "H"),
        _record("mol_edit.add_v2.0001__LOCAL__N", "N"),
    )
    return records, tuple(_prior(record) for record in records)


def _rows_for_split(
    split: str,
    prefix: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    records = (
        _record(f"{prefix}__H", "H", split=split),
        _record(f"{prefix}__N", "N", split=split),
    )
    return records, tuple(_prior(record) for record in records)


def _mock_tokenized_release(
    tmp_path: Path,
) -> tuple[Path, Path, tuple[dict[str, object], ...]]:
    records, prior = _rows()
    root = tmp_path / "HallucinationDataset"
    (root / "records").mkdir(parents=True)
    (root / "tokenized/chemdfm_r").mkdir(parents=True)
    (root / "records/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    (root / "tokenized/chemdfm_r/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prior),
        encoding="utf-8",
    )
    canonical_root = root / "real-tokenized"
    tokenize_release(
        root,
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        tokenizer=_CharacterFastTokenizer(),
        splits=("train",),
        expected_counts=None,
        output_root=canonical_root,
    )
    write_git_tokenized_shards(
        canonical_root,
        splits=("train",),
        max_shard_bytes=1_000_000,
    )
    index_path = canonical_root / "git_shards/index.json"
    rows = iter_git_tokenized_rows(index_path, "train")
    return canonical_root, index_path, rows


def _mock_activation_inventory(
    tmp_path: Path,
) -> tuple[Path, Path, Path, tuple[dict[str, object], ...]]:
    canonical_root, index_path, rows = _mock_tokenized_release(tmp_path)
    root = tmp_path / "layer_26"
    split_root = root / "train"
    split_root.mkdir(parents=True)
    tensor_path = split_root / "train-00000.pt"
    tensor_path.write_bytes(b"server-only-tensor-payload")
    sidecar_path = split_root / "train-00000.json"
    token_counts = [len(row["input_ids"]) for row in rows]
    row_offsets = [0]
    for token_count in token_counts:
        row_offsets.append(row_offsets[-1] + token_count)
    sidecar = {
        "format_version": "t051_chemdfm_r_post_token_v1",
        "status": "complete",
        "split": "train",
        "shard_index": 0,
        "record_ids": [row["record_id"] for row in rows],
        "token_counts": token_counts,
        "row_offsets": row_offsets,
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "layer_index": FROZEN_LAYER_INDEX,
        "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        "feature_name": "resid_post",
        "activation_shape": [sum(token_counts), EXPECTED_HIDDEN_SIZE],
        "activation_dtype": "bfloat16",
        "file_bytes": tensor_path.stat().st_size,
        "digest_computation_performed": False,
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = {
        "format_version": T051_ACTIVATION_MANIFEST_VERSION,
        "status": "complete",
        "mode": "test",
        "alignment": {
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "label_shift": 0,
            "hidden_token_axis_equals_label_length": True,
        },
        "splits": {
            "train": {
                "record_count": len(rows),
                "token_count": sum(token_counts),
                "shard_count": 1,
            }
        },
        "record_count": len(rows),
        "token_count": sum(token_counts),
        "shard_count": 1,
        "shards": [
            {
                "split": "train",
                "shard_index": 0,
                "tensor_path": str(tensor_path),
                "metadata_path": str(sidecar_path),
                "record_count": len(rows),
                "token_count": sum(token_counts),
                "hidden_size": EXPECTED_HIDDEN_SIZE,
                "layer_index": FROZEN_LAYER_INDEX,
                "file_bytes": tensor_path.stat().st_size,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, canonical_root, index_path, rows


def test_real_projection_replaces_fixture_tokens_and_preserves_char_identity() -> None:
    records, prior = _rows()
    rows, summary = tokenize_records(
        records,
        prior,
        _CharacterFastTokenizer(),
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        split="train",
    )

    assert summary.record_count == 2
    assert summary.carried_text_identity_count == 2
    assert all(row["activation_alignment"] == ACTIVATION_ALIGNMENT for row in rows)
    assert rows[0]["serialized_text_sha256"] == "carried-text-identity-H"
    fingerprint = rows[0]["tokenizer_fingerprint"]
    assert fingerprint["tokenizer_revision"] == "approved-local-checkpoint-no-digest"
    assert fingerprint["tokenizer_vocab_hash"] == "not-computed-per-user-instruction"
    assert fingerprint["normalization_config"]["digest_computation_performed"] is False

    h_row, n_row = rows
    h_token_count = validate_tokenized_row(h_row)
    n_token_count = validate_tokenized_row(n_row)
    assert h_token_count == len(records[0]["serialized"]["text"])
    assert n_token_count == len(records[1]["serialized"]["text"])
    literal_start = records[0]["spans"][0]["literal_span"][0]
    assert h_row["semantic_type_masks"]["0"][literal_start] == 1
    assert h_row["semantic_type_masks"]["2"][literal_start] == 1
    assert h_row["causal_role_masks"]["ROOT"][literal_start] == 1
    assert h_row["error_any_mask"][literal_start] == 1
    assert all(
        sum(mask) == 0
        for axis in (
            n_row["semantic_type_masks"],
            n_row["edit_subtype_masks"],
            n_row["causal_role_masks"],
        )
        for mask in axis.values()
    )
    assert n_row["matched_target_span"] == prior[1]["matched_target_span"]


def test_hidden_axis_is_same_index_post_token_and_rejects_both_shifts() -> None:
    records, prior = _rows()
    rows, _ = tokenize_records(
        records,
        prior,
        _CharacterFastTokenizer(),
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        split="train",
    )
    token_count = len(rows[0]["input_ids"])
    assert (
        assert_post_token_axis(
            rows[0], (1, token_count, EXPECTED_HIDDEN_SIZE)
        )
        == token_count
    )
    for drift in (-1, 1):
        with pytest.raises(
            T051ArtifactError, match="HIDDEN_LABEL_TOKEN_AXIS_MISMATCH"
        ):
            assert_post_token_axis(
                rows[0], (1, token_count + drift, EXPECTED_HIDDEN_SIZE)
            )
    shifted = dict(rows[0])
    shifted["activation_alignment"] = "pre_token"
    with pytest.raises(T051ArtifactError, match="ACTIVATION_ALIGNMENT_INVALID"):
        assert_post_token_axis(
            shifted, (1, token_count, EXPECTED_HIDDEN_SIZE)
        )


def test_activation_shards_are_deterministic_and_keep_exact_record_order() -> None:
    records, prior = _rows()
    rows, _ = tokenize_records(
        records,
        prior,
        _CharacterFastTokenizer(),
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        split="train",
    )
    plans = plan_activation_shards({"train": rows}, shard_size=1)
    assert [plan.stem for plan in plans] == ["train-00000", "train-00001"]
    assert [plan.record_ids for plan in plans] == [
        ("mol_edit.add_v2.0001__LOCAL__H",),
        ("mol_edit.add_v2.0001__LOCAL__N",),
    ]
    assert all(plan.token_counts == (len(rows[index]["input_ids"]),) for index, plan in enumerate(plans))


def test_tokenize_release_writes_atomic_smoke_artifacts_without_digest_work(
    tmp_path: Path,
) -> None:
    records, prior = _rows()
    root = tmp_path / "HallucinationDataset"
    (root / "records").mkdir(parents=True)
    (root / "tokenized/chemdfm_r").mkdir(parents=True)
    (root / "records/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    (root / "tokenized/chemdfm_r/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prior),
        encoding="utf-8",
    )
    output_root = tmp_path / "real-tokenized"

    manifest = tokenize_release(
        root,
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        tokenizer=_CharacterFastTokenizer(),
        splits=("train",),
        expected_counts=None,
        output_root=output_root,
    )

    assert manifest["record_count"] == 2
    assert manifest["activation_alignment"] == ACTIVATION_ALIGNMENT
    assert manifest["label_shift"] == 0
    assert manifest["identity_handling"] == {
        "source": "records[*].serialized.sha256",
        "carried_forward": True,
        "recomputed": False,
        "verified_by_digest": False,
    }
    assert (output_root / "train.jsonl").is_file()
    assert (output_root / "manifest.json").is_file()
    assert not list(output_root.glob("*.partial"))


def test_slow_tokenizer_and_array_drift_fail_closed() -> None:
    records, prior = _rows()

    class _SlowTokenizer(_CharacterFastTokenizer):
        is_fast = False

    with pytest.raises(T051ArtifactError, match="FAST_TOKENIZER_REQUIRED"):
        tokenize_records(
            records,
            prior,
            _SlowTokenizer(),
            checkpoint_path=Path("/approved/ChemDFM-R-14B"),
            split="train",
        )

    rows, _ = tokenize_records(
        records,
        prior,
        _CharacterFastTokenizer(),
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        split="train",
    )
    drifted = dict(rows[0])
    drifted["evaluation_mask"] = drifted["evaluation_mask"][:-1]
    with pytest.raises(T051ArtifactError, match="TOKEN_ARRAY_LENGTH_MISMATCH"):
        validate_tokenized_row(drifted)


def test_git_shards_are_byte_bounded_indexed_and_reconstructable(
    tmp_path: Path,
) -> None:
    records, prior = _rows()
    root = tmp_path / "HallucinationDataset"
    (root / "records").mkdir(parents=True)
    (root / "tokenized/chemdfm_r").mkdir(parents=True)
    (root / "records/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    (root / "tokenized/chemdfm_r/train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prior),
        encoding="utf-8",
    )
    canonical_root = root / "real-tokenized"
    tokenize_release(
        root,
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        tokenizer=_CharacterFastTokenizer(),
        splits=("train",),
        expected_counts=None,
        output_root=canonical_root,
    )
    canonical = canonical_root / "train.jsonl"
    line_sizes = [len(line) for line in canonical.read_bytes().splitlines(keepends=True)]
    max_bytes = max(line_sizes) + 1

    index = write_git_tokenized_shards(
        canonical_root,
        splits=("train",),
        max_shard_bytes=max_bytes,
    )
    index_path = canonical_root / "git_shards/index.json"
    assert index["splits"]["train"]["shard_count"] == 2
    assert all(
        shard["bytes"] < max_bytes
        for shard in index["splits"]["train"]["shards"]
    )
    restored_rows = iter_git_tokenized_rows(index_path, "train")
    assert [row["record_id"] for row in restored_rows] == [
        record["record_id"] for record in records
    ]
    validation = validate_git_shard_inventory(index_path)
    assert validation["all_pass"] is True
    assert validation["record_count"] == 2
    assert validation["digest_verification_performed"] is False

    reassembled = tmp_path / "reassembled/train.jsonl"
    result = reassemble_git_tokenized_split(index_path, "train", reassembled)
    assert result["record_count"] == 2
    assert reassembled.read_bytes() == canonical.read_bytes()


def test_git_shards_reject_cross_split_row_permutation(tmp_path: Path) -> None:
    canonical_root, index_path, _ = _mock_tokenized_release(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_relative = index["splits"]["train"]["shards"][0]["path"]
    shard_path = canonical_root / "git_shards" / shard_relative
    rows = [json.loads(line) for line in shard_path.read_text().splitlines()]
    rows[0]["split"] = "test"
    shard_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    index["splits"]["train"]["shards"][0]["bytes"] = shard_path.stat().st_size
    index["splits"]["train"]["canonical_bytes"] = shard_path.stat().st_size
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(T051ArtifactError, match="TOKENIZED_ROW_SPLIT_MISMATCH"):
        iter_git_tokenized_rows(index_path, "train")


def test_activation_inventory_finalizes_server_only_sidecars(tmp_path: Path) -> None:
    manifest_path, canonical_root, index_path, _ = _mock_activation_inventory(
        tmp_path
    )

    report = finalize_activation_inventory(
        manifest_path,
        tokenized_root=canonical_root,
        tokenized_index_path=index_path,
        strict_payload_validation=False,
    )

    assert report["all_pass"] is True
    assert report["record_count"] == 2
    assert report["token_count"] > 5
    assert report["unique_record_id_count"] == 2
    assert report["digest_verification_performed"] is False
    assert report["tensor_payload_validation"]["performed"] is False
    assert report["tensor_payload_validation"]["all_pass"] is False
    sidecar_path = manifest_path.parent / "train/train-00000.json"
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["tensor_relative_path"] == "train/train-00000.pt"
    assert updated_sidecar["external_storage"] == "server_only"
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["inventory_validation"] == report
    assert updated_manifest["shards"][0]["tensor_path"] == (
        "train/train-00000.pt"
    )
    assert updated_manifest["shards"][0]["metadata_path"] == (
        "train/train-00000.json"
    )
    assert updated_manifest["external_storage"]["tensor_root"] == "."


def test_release_inventory_cannot_disable_strict_payload_validation(
    tmp_path: Path,
) -> None:
    manifest_path, canonical_root, index_path, _ = _mock_activation_inventory(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mode"] = "release"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        T051ArtifactError, match="STRICT_RELEASE_PAYLOAD_VALIDATION_REQUIRED"
    ):
        finalize_activation_inventory(
            manifest_path,
            tokenized_root=canonical_root,
            tokenized_index_path=index_path,
            strict_payload_validation=False,
        )


def test_activation_inventory_rejects_per_split_summary_drift(tmp_path: Path) -> None:
    manifest_path, canonical_root, index_path, _ = _mock_activation_inventory(
        tmp_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["token_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        T051ArtifactError, match="ACTIVATION_MANIFEST_SPLIT_TOTAL_MISMATCH"
    ):
        finalize_activation_inventory(
            manifest_path,
            tokenized_root=canonical_root,
            tokenized_index_path=index_path,
            strict_payload_validation=False,
        )


def test_activation_inventory_rejects_intra_split_order_drift(
    tmp_path: Path,
) -> None:
    manifest_path, canonical_root, index_path, rows = _mock_activation_inventory(
        tmp_path
    )
    sidecar_path = manifest_path.parent / "train/train-00000.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["record_ids"] = list(reversed([row["record_id"] for row in rows]))
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(T051ArtifactError, match="ACTIVATION_TOKENIZED_AXIS_MISMATCH"):
        finalize_activation_inventory(
            manifest_path,
            tokenized_root=canonical_root,
            tokenized_index_path=index_path,
            strict_payload_validation=False,
        )


def test_activation_inventory_rejects_equal_total_cross_split_swap(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "HallucinationDataset"
    (release_root / "records").mkdir(parents=True)
    (release_root / "tokenized/chemdfm_r").mkdir(parents=True)
    for split, prefix in (("train", "train-pair"), ("test", "test-pair")):
        records, prior = _rows_for_split(split, prefix)
        (release_root / f"records/{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )
        (release_root / f"tokenized/chemdfm_r/{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in prior),
            encoding="utf-8",
        )
    canonical_root = release_root / "real-tokenized"
    tokenize_release(
        release_root,
        checkpoint_path=Path("/approved/ChemDFM-R-14B"),
        tokenizer=_CharacterFastTokenizer(),
        splits=("train", "test"),
        expected_counts=None,
        output_root=canonical_root,
    )
    write_git_tokenized_shards(
        canonical_root,
        splits=("train", "test"),
        max_shard_bytes=1_000_000,
    )
    index_path = canonical_root / "git_shards/index.json"
    rows_by_split = {
        split: iter_git_tokenized_rows(index_path, split)
        for split in ("train", "test")
    }
    activation_root = tmp_path / "layer_26"
    manifest_shards: list[dict[str, object]] = []
    manifest_splits: dict[str, dict[str, int]] = {}
    sidecar_paths: dict[str, Path] = {}
    for split, rows in rows_by_split.items():
        split_root = activation_root / split
        split_root.mkdir(parents=True)
        tensor_path = split_root / f"{split}-00000.pt"
        tensor_path.write_bytes(f"{split}-tensor-placeholder".encode())
        token_counts = [len(row["input_ids"]) for row in rows]
        row_offsets = [0]
        for token_count in token_counts:
            row_offsets.append(row_offsets[-1] + token_count)
        sidecar_path = split_root / f"{split}-00000.json"
        sidecar_paths[split] = sidecar_path
        sidecar_path.write_text(
            json.dumps(
                {
                    "format_version": "t051_chemdfm_r_post_token_v1",
                    "status": "complete",
                    "split": split,
                    "shard_index": 0,
                    "record_ids": [row["record_id"] for row in rows],
                    "token_counts": token_counts,
                    "row_offsets": row_offsets,
                    "activation_alignment": ACTIVATION_ALIGNMENT,
                    "label_shift": 0,
                    "layer_index": FROZEN_LAYER_INDEX,
                    "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
                    "activation_shape": [sum(token_counts), EXPECTED_HIDDEN_SIZE],
                    "activation_dtype": "bfloat16",
                    "file_bytes": tensor_path.stat().st_size,
                }
            ),
            encoding="utf-8",
        )
        manifest_shards.append(
            {
                "split": split,
                "shard_index": 0,
                "tensor_path": str(tensor_path),
                "metadata_path": str(sidecar_path),
                "record_count": len(rows),
                "token_count": sum(token_counts),
                "hidden_size": EXPECTED_HIDDEN_SIZE,
                "layer_index": FROZEN_LAYER_INDEX,
                "file_bytes": tensor_path.stat().st_size,
            }
        )
        manifest_splits[split] = {
            "record_count": len(rows),
            "token_count": sum(token_counts),
            "shard_count": 1,
        }
    train_sidecar = json.loads(sidecar_paths["train"].read_text())
    test_sidecar = json.loads(sidecar_paths["test"].read_text())
    train_sidecar["record_ids"], test_sidecar["record_ids"] = (
        test_sidecar["record_ids"],
        train_sidecar["record_ids"],
    )
    sidecar_paths["train"].write_text(json.dumps(train_sidecar), encoding="utf-8")
    sidecar_paths["test"].write_text(json.dumps(test_sidecar), encoding="utf-8")
    manifest_path = activation_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": T051_ACTIVATION_MANIFEST_VERSION,
                "status": "complete",
                "mode": "test",
                "alignment": {
                    "activation_alignment": ACTIVATION_ALIGNMENT,
                    "label_shift": 0,
                    "hidden_token_axis_equals_label_length": True,
                },
                "splits": manifest_splits,
                "record_count": 4,
                "token_count": sum(
                    split["token_count"] for split in manifest_splits.values()
                ),
                "shard_count": 2,
                "shards": manifest_shards,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(T051ArtifactError, match="ACTIVATION_TOKENIZED_AXIS_MISMATCH"):
        finalize_activation_inventory(
            manifest_path,
            tokenized_root=canonical_root,
            tokenized_index_path=index_path,
            strict_payload_validation=False,
        )


def test_manifest_member_paths_reject_traversal_and_root_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "layer_26"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.pt").write_bytes(b"external")

    with pytest.raises(T051ArtifactError, match="PATH_TRAVERSAL"):
        _manifest_member_path(root, "../outside/payload.pt")
    with pytest.raises(T051ArtifactError, match="PATH_OUTSIDE_ROOT"):
        _manifest_member_path(
            root,
            str(outside / "payload.pt"),
            allow_legacy_absolute=True,
        )
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(T051ArtifactError, match="PATH_OUTSIDE_ROOT"):
        _manifest_member_path(root, "escape/payload.pt")


class _FakeTensor:
    def __init__(self, shape: tuple[int, int], dtype: str = "bfloat16") -> None:
        self.shape = shape
        self.dtype = dtype


class _FakeTorch:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def load(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return self.payload

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)


def test_strict_resume_rejects_same_size_wrong_tensor_payload(
    tmp_path: Path,
) -> None:
    _, _, rows = _mock_tokenized_release(tmp_path)
    plan = plan_activation_shards({"train": rows}, shard_size=2)[0]
    split_root = tmp_path / "layer_26/train"
    split_root.mkdir(parents=True)
    tensor_path = split_root / "train-00000.pt"
    tensor_path.write_bytes(b"x" * 128)
    token_counts = list(plan.token_counts)
    row_offsets = [0]
    for token_count in token_counts:
        row_offsets.append(row_offsets[-1] + token_count)
    metadata_path = split_root / "train-00000.json"
    metadata_path.write_text(
        json.dumps(
            {
                "format_version": "t051_chemdfm_r_post_token_v1",
                "status": "complete",
                "split": "train",
                "shard_index": 0,
                "record_ids": list(plan.record_ids),
                "token_counts": token_counts,
                "row_offsets": row_offsets,
                "activation_alignment": ACTIVATION_ALIGNMENT,
                "label_shift": 0,
                "layer_index": FROZEN_LAYER_INDEX,
                "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
                "activation_shape": [sum(token_counts), EXPECTED_HIDDEN_SIZE],
                "activation_dtype": "bfloat16",
                "file_bytes": 128,
            }
        ),
        encoding="utf-8",
    )
    wrong_payload = {
        "format_version": "t051_chemdfm_r_post_token_v1",
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "layer_index": FROZEN_LAYER_INDEX,
        "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        "record_ids": list(reversed(plan.record_ids)),
        "token_counts": token_counts,
        "row_offsets": row_offsets,
        "activations": _FakeTensor((sum(token_counts), EXPECTED_HIDDEN_SIZE)),
    }

    with pytest.raises(
        T051ArtifactError, match="ACTIVATION_TENSOR_PAYLOAD_MISMATCH"
    ):
        _resume_metadata(
            plan,
            tensor_path,
            metadata_path,
            torch_runtime=_FakeTorch(wrong_payload),
            strict_payload_validation=True,
        )
