"""T030 split-local donor pool and import boundary tests."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.leakage_groups import assign_leakage_groups
from molhallulens.builders.origin_audit import audit_origin_split_features
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.builders.split_manifest import (
    FrozenSplitManifest,
    VerifiedSplitManifest,
    build_frozen_split_manifest,
    load_verified_split_manifest,
    write_frozen_split_manifest,
)
from molhallulens.builders.splitter import build_group_stratified_split
from molhallulens.candidates.donors import (
    DONOR_POOL_FORMAT_VERSION,
    DONOR_POOL_SCHEMA_VERSION,
    DonorKind,
    DonorPoolError,
    SplitBoundDonorQuery,
    SplitDonorPool,
    build_split_local_donor_pools,
    load_split_donor_pool,
    write_split_donor_pools,
)
from molhallulens.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
DONOR_POOL_ROOT = (
    Path(__file__).resolve().parents[2] / "HallucinationDataset" / "donor_pools"
)


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
    items = tuple(items)
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
    frozen = build_frozen_split_manifest(split, audit)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "split_manifest.csv"
        metadata_path = root / "split_manifest.metadata.json"
        write_frozen_split_manifest(
            frozen,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )
        verified = load_verified_split_manifest(
            manifest_path,
            metadata_path,
            split_result=split,
            audit=audit,
        )
    return items, audit, split, frozen, verified


@cache
def _pools() -> tuple[SplitDonorPool, ...]:
    items, audit, _, _, verified = _sources()
    return build_split_local_donor_pools(
        verified,
        items=items,
        audit=audit,
    )


def _pool(split: str) -> SplitDonorPool:
    return next(item for item in _pools() if item.split == split)


def test_three_pools_are_manifest_bound_split_local_and_deterministic() -> None:
    _, _, _, _, manifest = _sources()
    pools = _pools()
    assert type(manifest) is VerifiedSplitManifest
    assert tuple(item.split for item in pools) == ("train", "validation", "test")
    assert {item.manifest_sha256 for item in pools} == {manifest.manifest_sha256}
    assert {item.dataset_version for item in pools} == {manifest.dataset_version}
    assert {item.schema_version for item in pools} == {DONOR_POOL_SCHEMA_VERSION}
    assert {item.format_version for item in pools} == {DONOR_POOL_FORMAT_VERSION}

    assert {pool.split: (len(pool.donors), len(pool.edges)) for pool in pools} == {
        "train": (234, 1104),
        "validation": (58, 76),
        "test": (59, 68),
    }
    for pool in pools:
        donor_by_id = {item.donor_id: item for item in pool.donors}
        assert len(donor_by_id) == len(pool.donors)
        assert all(
            manifest.split_for_origin(item.donor_origin_id).value == pool.split
            for item in pool.donors
        )
        assert all(
            edge.recipient_origin_id != edge.donor_origin_id
            and manifest.require_same_split(
                edge.recipient_origin_id, edge.donor_origin_id
            ).value
            == pool.split
            and donor_by_id[edge.donor_id].bucket_id == edge.bucket_id
            for edge in pool.edges
        )

    reversed_pools = build_split_local_donor_pools(
        manifest,
        items=reversed(_sources()[0]),
        audit=_sources()[1],
    )
    assert tuple(item.to_json_bytes() for item in reversed_pools) == tuple(
        item.to_json_bytes() for item in pools
    )


def test_frozen_donor_pool_artifacts_are_byte_identical_to_rebuild() -> None:
    for pool in _pools():
        path = DONOR_POOL_ROOT / f"{pool.split}.json"
        assert path.read_bytes() == pool.to_json_bytes()


def test_split_bound_query_never_returns_self_or_another_bucket() -> None:
    _, _, _, _, manifest = _sources()
    pool = _pool("train")
    edge = pool.edges[0]
    expected = next(item for item in pool.donors if item.donor_id == edge.donor_id)
    query = SplitBoundDonorQuery(
        manifest_sha256=manifest.manifest_sha256,
        split=pool.split,
        recipient_origin_id=edge.recipient_origin_id,
        kind=edge.kind,
        attachment_bucket=expected.attachment_bucket,
        descriptor_bucket=expected.descriptor_bucket,
        difficulty_bucket=expected.difficulty_bucket,
    )
    donors = pool.query(query)
    assert expected in donors
    assert all(
        item.donor_origin_id != query.recipient_origin_id
        and item.kind is query.kind
        and item.attachment_bucket == query.attachment_bucket
        and item.descriptor_bucket == query.descriptor_bucket
        and item.difficulty_bucket == query.difficulty_bucket
        for item in donors
    )
    assert (
        pool.query(
            SplitBoundDonorQuery(
                manifest_sha256=manifest.manifest_sha256,
                split=pool.split,
                recipient_origin_id=edge.recipient_origin_id,
                kind=edge.kind,
                limit=1,
            )
        )
        == donors[:1]
    )

    with pytest.raises(DonorPoolError) as wrong_manifest:
        pool.query(
            SplitBoundDonorQuery(
                manifest_sha256="0" * 64,
                split=pool.split,
                recipient_origin_id=edge.recipient_origin_id,
                kind=edge.kind,
            )
        )
    assert wrong_manifest.value.code == "QUERY_MANIFEST_SPLIT_MISMATCH"
    with pytest.raises(DonorPoolError) as wrong_split:
        pool.query(
            SplitBoundDonorQuery(
                manifest_sha256=manifest.manifest_sha256,
                split="validation",
                recipient_origin_id=edge.recipient_origin_id,
                kind=edge.kind,
            )
        )
    assert wrong_split.value.code == "QUERY_MANIFEST_SPLIT_MISMATCH"


def test_writer_loader_round_trip_and_frozen_manifest_type_boundary(
    tmp_path: Path,
) -> None:
    _, _, _, frozen, manifest = _sources()
    write_split_donor_pools(_pools(), output_directory=tmp_path)
    loaded = tuple(
        load_split_donor_pool(
            tmp_path / f"{split}.json",
            manifest=manifest,
            expected_split=split,
        )
        for split in ("train", "validation", "test")
    )
    assert tuple(item.to_json_bytes() for item in loaded) == tuple(
        item.to_json_bytes() for item in _pools()
    )

    assert type(frozen) is FrozenSplitManifest
    with pytest.raises(TypeError, match="VerifiedSplitManifest"):
        build_split_local_donor_pools(
            frozen,  # type: ignore[arg-type]
            items=_sources()[0],
            audit=_sources()[1],
        )


@pytest.mark.parametrize(
    "mutation",
    ("manifest", "unknown_origin", "cross_split_edge", "cross_split_donor"),
)
def test_import_rejects_manifest_unknown_and_cross_split_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, _, _, _, manifest = _sources()
    write_split_donor_pools(_pools(), output_directory=tmp_path)
    train_path = tmp_path / "train.json"
    payload = json.loads(train_path.read_text())

    if mutation == "manifest":
        payload["split_manifest_sha256"] = "0" * 64
    elif mutation == "unknown_origin":
        referenced = {edge["recipient_origin_id"] for edge in payload["edges"]} | {
            edge["donor_origin_id"] for edge in payload["edges"]
        }
        donor = next(
            item
            for item in payload["donors"]
            if item["donor_origin_id"] not in referenced
        )
        donor["donor_origin_id"] = "unknown.origin"
        donor["donor_id"] = f"{donor['kind']}:unknown.origin"
    elif mutation == "cross_split_edge":
        validation_id = next(
            row.anonymous_sample_id
            for row in manifest.rows
            if row.split.value == "validation"
        )
        payload["edges"][0]["recipient_origin_id"] = validation_id
        payload["edges"][0]["edge_id"] = (
            f"{validation_id}->{payload['edges'][0]['donor_id']}"
        )
    else:
        validation_donor = _pool("validation").donors[0].to_dict()
        payload["donors"].append(validation_donor)
    train_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )

    before = manifest.rows
    with pytest.raises(DonorPoolError) as captured:
        load_split_donor_pool(
            train_path,
            manifest=manifest,
            expected_split="train",
        )
    assert manifest.rows == before
    expected_codes = {
        "manifest": "DONOR_MANIFEST_BINDING_MISMATCH",
        "unknown_origin": "UNKNOWN_MANIFEST_ORIGIN",
        "cross_split_edge": "CROSS_SPLIT_DONOR_EDGE",
        "cross_split_donor": "CROSS_SPLIT_DONOR",
    }
    assert captured.value.code == expected_codes[mutation]


def test_builder_does_not_call_split_solver_or_rewrite_manifest(monkeypatch) -> None:
    from molhallulens.builders.splitter import GroupStratifiedSplitter

    _, _, _, _, manifest = _sources()
    rows_before = manifest.rows

    def forbidden(*args, **kwargs):
        raise AssertionError("T030 must never call the split solver")

    monkeypatch.setattr(GroupStratifiedSplitter, "solve", forbidden)
    rebuilt = build_split_local_donor_pools(
        manifest,
        items=_sources()[0],
        audit=_sources()[1],
    )
    assert tuple(item.to_json_bytes() for item in rebuilt) == tuple(
        item.to_json_bytes() for item in _pools()
    )
    assert manifest.rows == rows_before
    assert all(
        donor.kind in {DonorKind.FRAGMENT, DonorKind.GROUP, DonorKind.PRODUCT}
        for pool in rebuilt
        for donor in pool.donors
    )
