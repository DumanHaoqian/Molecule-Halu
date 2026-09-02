"""Unit and adversarial contracts for the frozen T029 split manifest."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import pytest

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference.truth import derive_edit_truth
from molhallulens.modules.release.leakage import assign_leakage_groups
from molhallulens.modules.release.origin_audit import audit_origin_split_features
from molhallulens.modules.reference.builder import build_reference_dag
from molhallulens.modules.release.manifest import (
    MANIFEST_FIELDS,
    FrozenSplitManifest,
    SplitManifestError,
    VerifiedSplitManifest,
    build_frozen_split_manifest,
    load_verified_split_manifest,
    write_frozen_split_manifest,
)
from molhallulens.modules.release.splitter import (
    GroupStratifiedSplitResult,
    SplitName,
    build_group_stratified_split,
)
from molhallulens.infrastructure.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


@cache
def _sources():
    items = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        artifact = build_reference_dag(record)
        items.append(
            OriginValidationInput(
                record=record,
                artifact=artifact,
                edit_truth=derive_edit_truth(artifact),
            )
        )
    audit = audit_origin_split_features(items).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in items
        },
    )
    split = build_group_stratified_split(audit, leakage)
    return audit, split


@cache
def _manifest() -> FrozenSplitManifest:
    audit, split = _sources()
    return build_frozen_split_manifest(split, audit)


def _write_expected(directory: Path) -> tuple[Path, Path]:
    manifest_path = directory / "split_manifest.csv"
    metadata_path = directory / "split_manifest.metadata.json"
    write_frozen_split_manifest(
        _manifest(),
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    return manifest_path, metadata_path


def _csv_table(payload: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload.decode(), newline=""), strict=True))


def _csv_bytes(table: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(table)
    return buffer.getvalue().encode()


def _load(manifest_path: Path, metadata_path: Path):
    audit, split = _sources()
    return load_verified_split_manifest(
        manifest_path,
        metadata_path,
        split_result=split,
        audit=audit,
    )


def test_manifest_has_exact_fields_counts_order_and_identity() -> None:
    manifest = _manifest()
    payload = manifest.to_csv_bytes()
    table = _csv_table(payload)

    assert tuple(table[0]) == MANIFEST_FIELDS
    assert len(table) == 151
    assert b"\r" not in payload and payload.endswith(b"\n")
    assert [row.origin_id for row in manifest.rows] == sorted(
        row.origin_id for row in manifest.rows
    )
    assert len({row.anonymous_sample_id for row in manifest.rows}) == 150
    assert Counter(row.split for row in manifest.rows) == {
        SplitName.TRAIN: 100,
        SplitName.VALIDATION: 25,
        SplitName.TEST: 25,
    }
    assert manifest.manifest_sha256 == manifest.metadata.manifest_sha256
    assert manifest.metadata.to_dict()["variant_policy"] == (
        "inherit_origin_split_only"
    )
    assert "generated_at" not in manifest.metadata.to_dict()
    assert "timestamp" not in manifest.metadata.to_dict()


def test_writer_is_idempotent_and_loader_is_the_only_verified_constructor(
    tmp_path: Path,
) -> None:
    manifest_path, metadata_path = _write_expected(tmp_path)
    before = (manifest_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
    write_frozen_split_manifest(
        _manifest(),
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    after = (manifest_path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)
    assert before == after

    audit, split = _sources()
    verified = load_verified_split_manifest(
        manifest_path,
        metadata_path,
        split_result=split,
        audit=audit,
    )
    assert type(verified) is VerifiedSplitManifest
    assert verified.rows == _manifest().rows
    assert verified.manifest_sha256 == _manifest().manifest_sha256
    row = verified.rows[0]
    assert verified.row_for_origin(row.anonymous_sample_id) is row
    assert verified.split_for_origin(row.anonymous_sample_id) is row.split
    assert (
        verified.require_same_split(
            row.anonymous_sample_id,
            row.anonymous_sample_id,
        )
        is row.split
    )
    with pytest.raises(TypeError, match="only be created"):
        VerifiedSplitManifest(_manifest(), _manifest().metadata)


def test_conflicting_existing_artifact_is_never_overwritten(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split_manifest.csv"
    metadata_path = tmp_path / "split_manifest.metadata.json"
    manifest_path.write_bytes(b"do-not-overwrite\n")

    with pytest.raises(SplitManifestError) as captured:
        write_frozen_split_manifest(
            _manifest(),
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )
    assert captured.value.code == "MANIFEST_CONFLICT"
    assert manifest_path.read_bytes() == b"do-not-overwrite\n"
    assert not metadata_path.exists()


@pytest.mark.parametrize(
    "mutation", ["unknown_column", "missing_row", "duplicate_row", "hash_tamper"]
)
def test_loader_rejects_malformed_or_tampered_csv(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest_path, metadata_path = _write_expected(tmp_path)
    table = _csv_table(manifest_path.read_bytes())
    if mutation == "unknown_column":
        table[0].append("unexpected")
        for row in table[1:]:
            row.append("value")
    elif mutation == "missing_row":
        table.pop()
    elif mutation == "duplicate_row":
        table[2] = list(table[1])
    else:
        table[1][5] = "0" * 64
    manifest_path.write_bytes(_csv_bytes(table))

    with pytest.raises(SplitManifestError):
        _load(manifest_path, metadata_path)


def test_loader_rejects_a_group_split_even_when_cell_counts_are_preserved(
    tmp_path: Path,
) -> None:
    manifest_path, metadata_path = _write_expected(tmp_path)
    table = _csv_table(manifest_path.read_bytes())
    header = {name: index for index, name in enumerate(table[0])}
    rows = table[1:]
    by_group: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        by_group[row[header["leakage_group_id"]]].append(row)
    group_member = next(members[0] for members in by_group.values() if len(members) > 1)
    source_split = group_member[header["split"]]
    source_subtask = group_member[header["subtask"]]
    singleton = next(
        members[0]
        for members in by_group.values()
        if len(members) == 1
        and members[0][header["subtask"]] == source_subtask
        and members[0][header["split"]] != source_split
    )
    group_member[header["split"]], singleton[header["split"]] = (
        singleton[header["split"]],
        group_member[header["split"]],
    )
    manifest_path.write_bytes(_csv_bytes(table))

    with pytest.raises(SplitManifestError) as captured:
        _load(manifest_path, metadata_path)
    assert captured.value.code == "MANIFEST_INVARIANT_FAILED"
    assert "atomic leakage group" in captured.value.evidence["detail"]


def test_loader_rejects_metadata_unknown_fields_and_artifact_identity_tamper(
    tmp_path: Path,
) -> None:
    manifest_path, metadata_path = _write_expected(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["unknown"] = True
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(SplitManifestError) as unknown:
        _load(manifest_path, metadata_path)
    assert unknown.value.code == "MANIFEST_METADATA_MISMATCH"

    metadata = _manifest().metadata.to_dict()
    metadata["source_artifacts"]["split_balance_report_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(SplitManifestError) as identity:
        _load(manifest_path, metadata_path)
    assert identity.value.code == "MANIFEST_METADATA_MISMATCH"


def test_verified_lookup_rejects_unknown_ambiguous_and_cross_split_origins(
    tmp_path: Path,
) -> None:
    verified = _load(*_write_expected(tmp_path))
    with pytest.raises(SplitManifestError) as unknown:
        verified.row_for_origin("mol_edit.unknown")
    assert unknown.value.code == "MANIFEST_ORIGIN_UNKNOWN"

    legacy_counts = Counter(row.origin_id for row in verified.rows)
    ambiguous_id = next(key for key, count in legacy_counts.items() if count > 1)
    assert len(verified.rows_for_legacy_origin(ambiguous_id)) > 1
    with pytest.raises(SplitManifestError) as ambiguous:
        verified.row_for_legacy_origin(ambiguous_id)
    assert ambiguous.value.code == "MANIFEST_ORIGIN_AMBIGUOUS"

    left = next(row for row in verified.rows if row.split is SplitName.TRAIN)
    right = next(row for row in verified.rows if row.split is SplitName.TEST)
    with pytest.raises(SplitManifestError) as cross_split:
        verified.require_same_split(
            left.anonymous_sample_id,
            right.anonymous_sample_id,
        )
    assert cross_split.value.code == "CROSS_SPLIT_EDGE"


def test_builder_rejects_wrong_typed_sources() -> None:
    audit, split = _sources()
    with pytest.raises(TypeError):
        build_frozen_split_manifest(object(), audit)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_frozen_split_manifest(split, object())  # type: ignore[arg-type]
    assert type(split) is GroupStratifiedSplitResult
