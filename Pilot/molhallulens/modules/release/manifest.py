"""Freeze and verify the immutable origin-level split manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.core import EditingSubtask

from .origin_audit import OriginSplitAudit
from .splitter import (
    FROZEN_DATASET_VERSION,
    FROZEN_SPLIT_SEED,
    GroupStratifiedSplitResult,
    SplitName,
    derive_split_seed,
)

MANIFEST_FIELDS = (
    "origin_id",
    "anonymous_sample_id",
    "leakage_group_id",
    "subtask",
    "split",
    "canonical_source_hash",
    "canonical_gt_hash",
    "scaffold_hash",
    "split_seed",
    "dataset_version",
)
MANIFEST_FORMAT_VERSION = "split_manifest_csv_v1"
METADATA_FORMAT_VERSION = "split_manifest_metadata_v1"
DEFAULT_MANIFEST_FILENAME = "split_manifest.csv"
DEFAULT_METADATA_FILENAME = "split_manifest.metadata.json"
FROZEN_LEAKAGE_GROUP_COUNT = 142
_SPLIT_COUNTS = MappingProxyType(
    {
        SplitName.TRAIN: 100,
        SplitName.VALIDATION: 25,
        SplitName.TEST: 25,
    }
)
_SUBTASK_SPLIT_COUNTS = MappingProxyType(
    {
        (EditingSubtask.ADD, SplitName.TRAIN): 34,
        (EditingSubtask.ADD, SplitName.VALIDATION): 8,
        (EditingSubtask.ADD, SplitName.TEST): 8,
        (EditingSubtask.DELETE, SplitName.TRAIN): 33,
        (EditingSubtask.DELETE, SplitName.VALIDATION): 9,
        (EditingSubtask.DELETE, SplitName.TEST): 8,
        (EditingSubtask.SUBSTITUTE, SplitName.TRAIN): 33,
        (EditingSubtask.SUBSTITUTE, SplitName.VALIDATION): 8,
        (EditingSubtask.SUBSTITUTE, SplitName.TEST): 9,
    }
)


class SplitManifestError(RuntimeError):
    """Structured fail-closed manifest build, write, or load error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("manifest error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("manifest error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("manifest error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


def _is_sha256(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain_text(value: str, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{name} must be non-empty single-line text")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SplitManifestRow:
    """One origin assignment; variants may only inherit this row."""

    origin_id: str
    anonymous_sample_id: str
    leakage_group_id: str
    subtask: EditingSubtask
    split: SplitName
    canonical_source_hash: str
    canonical_gt_hash: str
    scaffold_hash: str
    split_seed: int
    dataset_version: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.leakage_group_id, "leakage_group_id"),
            (self.dataset_version, "dataset_version"),
        ):
            _plain_text(value, name)
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        if type(self.split) is not SplitName:
            raise TypeError("split must be SplitName")
        for value, name in (
            (self.leakage_group_id, "leakage_group_id"),
            (self.canonical_source_hash, "canonical_source_hash"),
            (self.canonical_gt_hash, "canonical_gt_hash"),
            (self.scaffold_hash, "scaffold_hash"),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be lowercase SHA256")
        if self.dataset_version != FROZEN_DATASET_VERSION:
            raise ValueError("manifest rows require pilot_v1")
        if type(self.split_seed) is not int or self.split_seed != derive_split_seed(
            self.dataset_version
        ):
            raise ValueError("manifest row seed must derive from dataset version")

    def to_csv_fields(self) -> tuple[str, ...]:
        return (
            self.origin_id,
            self.anonymous_sample_id,
            self.leakage_group_id,
            self.subtask.value,
            self.split.value,
            self.canonical_source_hash,
            self.canonical_gt_hash,
            self.scaffold_hash,
            str(self.split_seed),
            self.dataset_version,
        )

    @classmethod
    def from_csv_fields(cls, values: tuple[str, ...]) -> SplitManifestRow:
        if len(values) != len(MANIFEST_FIELDS):
            raise SplitManifestError(
                "MANIFEST_ROW_SHAPE",
                "CSV row does not have the frozen field count",
                evidence={"observed_fields": len(values)},
            )
        if any(type(value) is not str or not value for value in values):
            raise SplitManifestError(
                "MANIFEST_ROW_MISSING_FIELD",
                "CSV row contains a missing field",
            )
        try:
            split_seed = int(values[8])
            if str(split_seed) != values[8]:
                raise ValueError
            return cls(
                origin_id=values[0],
                anonymous_sample_id=values[1],
                leakage_group_id=values[2],
                subtask=EditingSubtask(values[3]),
                split=SplitName(values[4]),
                canonical_source_hash=values[5],
                canonical_gt_hash=values[6],
                scaffold_hash=values[7],
                split_seed=split_seed,
                dataset_version=values[9],
            )
        except (TypeError, ValueError) as error:
            raise SplitManifestError(
                "MANIFEST_ROW_INVALID",
                "CSV row violates the typed manifest contract",
                evidence={"error_type": type(error).__name__},
            ) from error


@dataclass(frozen=True, slots=True)
class FrozenSplitManifest:
    """Canonical bytes prepared from T026 identities and T028 assignments."""

    rows: tuple[SplitManifestRow, ...]
    source_origin_audit_sha256: str
    source_split_report_sha256: str
    format_version: str = MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        rows = tuple(
            sorted(
                self.rows,
                key=lambda item: (item.origin_id, item.anonymous_sample_id),
            )
        )
        if self.format_version != MANIFEST_FORMAT_VERSION:
            raise ValueError("unknown split manifest format")
        if any(type(item) is not SplitManifestRow for item in rows):
            raise TypeError("rows must contain SplitManifestRow values")
        if not _is_sha256(self.source_origin_audit_sha256) or not _is_sha256(
            self.source_split_report_sha256
        ):
            raise ValueError("source artifact identities must be SHA256")
        if len(rows) != 150:
            raise ValueError("split manifest requires exactly 150 rows")
        if len({item.anonymous_sample_id for item in rows}) != 150:
            raise ValueError("split manifest anonymous_sample_id values must be unique")
        if Counter(item.split for item in rows) != Counter(_SPLIT_COUNTS):
            raise ValueError("split manifest requires frozen 100/25/25 counts")
        if Counter(item.subtask for item in rows) != Counter(
            {subtask: 50 for subtask in EditingSubtask}
        ):
            raise ValueError("split manifest requires 50 origins per subtask")
        if Counter((item.subtask, item.split) for item in rows) != Counter(
            _SUBTASK_SPLIT_COUNTS
        ):
            raise ValueError("split manifest requires the frozen subtask matrix")
        if {item.dataset_version for item in rows} != {FROZEN_DATASET_VERSION}:
            raise ValueError("split manifest dataset version drifted")
        if {item.split_seed for item in rows} != {FROZEN_SPLIT_SEED}:
            raise ValueError("split manifest seed drifted")
        group_splits: dict[str, set[SplitName]] = defaultdict(set)
        for row in rows:
            group_splits[row.leakage_group_id].add(row.split)
        if len(group_splits) != FROZEN_LEAKAGE_GROUP_COUNT:
            raise ValueError("split manifest leakage group count drifted")
        if any(len(splits) != 1 for splits in group_splits.values()):
            raise ValueError("split manifest splits an atomic leakage group")
        object.__setattr__(self, "rows", rows)

    @property
    def dataset_version(self) -> str:
        return FROZEN_DATASET_VERSION

    @property
    def split_seed(self) -> int:
        return FROZEN_SPLIT_SEED

    @property
    def leakage_group_count(self) -> int:
        return len({item.leakage_group_id for item in self.rows})

    def to_csv_bytes(self) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.writer(
            buffer,
            dialect="excel",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(MANIFEST_FIELDS)
        writer.writerows(row.to_csv_fields() for row in self.rows)
        return buffer.getvalue().encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.to_csv_bytes())

    @property
    def metadata(self) -> SplitManifestMetadata:
        return SplitManifestMetadata(
            manifest_sha256=self.manifest_sha256,
            source_origin_audit_sha256=self.source_origin_audit_sha256,
            source_split_report_sha256=self.source_split_report_sha256,
            split_origin_counts=tuple(
                (split.value, sum(row.split is split for row in self.rows))
                for split in SplitName
            ),
            leakage_group_count=self.leakage_group_count,
        )


@dataclass(frozen=True, slots=True)
class SplitManifestMetadata:
    """Timestamp-free sidecar binding manifest bytes to T026 and T028."""

    manifest_sha256: str
    source_origin_audit_sha256: str
    source_split_report_sha256: str
    split_origin_counts: tuple[tuple[str, int], ...]
    leakage_group_count: int
    format_version: str = METADATA_FORMAT_VERSION
    dataset_version: str = FROZEN_DATASET_VERSION
    split_seed: int = FROZEN_SPLIT_SEED
    origin_count: int = 150
    record_count: int = 1200

    def __post_init__(self) -> None:
        counts = tuple(sorted(self.split_origin_counts))
        if self.format_version != METADATA_FORMAT_VERSION:
            raise ValueError("unknown split manifest metadata format")
        if self.dataset_version != FROZEN_DATASET_VERSION:
            raise ValueError("metadata dataset version drifted")
        if self.split_seed != derive_split_seed(self.dataset_version):
            raise ValueError("metadata split seed drifted")
        if self.origin_count != 150 or self.record_count != 1200:
            raise ValueError("metadata frozen corpus counts drifted")
        if self.leakage_group_count != FROZEN_LEAKAGE_GROUP_COUNT:
            raise ValueError("metadata leakage group count drifted")
        for value in (
            self.manifest_sha256,
            self.source_origin_audit_sha256,
            self.source_split_report_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("metadata artifact identities must be SHA256")
        expected_counts = tuple(
            sorted((split.value, count) for split, count in _SPLIT_COUNTS.items())
        )
        if counts != expected_counts:
            raise ValueError("metadata split counts drifted")
        object.__setattr__(self, "split_origin_counts", counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "manifest": {
                "filename": DEFAULT_MANIFEST_FILENAME,
                "sha256": self.manifest_sha256,
                "encoding": "utf-8",
                "line_ending": "LF",
                "field_order": list(MANIFEST_FIELDS),
                "sort_key": ["origin_id", "anonymous_sample_id"],
            },
            "dataset_version": self.dataset_version,
            "split_seed": self.split_seed,
            "origin_count": self.origin_count,
            "record_count": self.record_count,
            "leakage_group_count": self.leakage_group_count,
            "split_origin_counts": dict(self.split_origin_counts),
            "source_artifacts": {
                "origin_split_audit_sha256": self.source_origin_audit_sha256,
                "split_balance_report_sha256": self.source_split_report_sha256,
            },
            "variant_policy": "inherit_origin_split_only",
        }

    def to_json_bytes(self) -> bytes:
        return _json_bytes(self.to_dict())


_VERIFICATION_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSplitManifest:
    """Only loader-verified manifests may provide downstream split lookups."""

    _manifest: FrozenSplitManifest
    _metadata: SplitManifestMetadata
    _by_anonymous_id: Mapping[str, SplitManifestRow] = field(
        repr=False,
        compare=False,
    )
    _by_origin_id: Mapping[str, tuple[SplitManifestRow, ...]] = field(
        repr=False,
        compare=False,
    )
    _verification_marker: object = field(repr=False, compare=False)

    def __init__(
        self,
        manifest: FrozenSplitManifest,
        metadata: SplitManifestMetadata,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _VERIFICATION_TOKEN:
            raise TypeError(
                "VerifiedSplitManifest can only be created by load_verified_split_manifest"
            )
        if (
            type(manifest) is not FrozenSplitManifest
            or type(metadata) is not SplitManifestMetadata
        ):
            raise TypeError("verified manifest inputs use invalid concrete types")
        if metadata.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("verified metadata does not bind manifest bytes")
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(
            self,
            "_by_anonymous_id",
            MappingProxyType({row.anonymous_sample_id: row for row in manifest.rows}),
        )
        object.__setattr__(
            self,
            "_by_origin_id",
            MappingProxyType(
                {
                    origin_id: tuple(
                        row for row in manifest.rows if row.origin_id == origin_id
                    )
                    for origin_id in sorted({row.origin_id for row in manifest.rows})
                }
            ),
        )
        object.__setattr__(self, "_verification_marker", _VERIFICATION_TOKEN)

    @property
    def rows(self) -> tuple[SplitManifestRow, ...]:
        return self._manifest.rows

    @property
    def dataset_version(self) -> str:
        return self._manifest.dataset_version

    @property
    def split_seed(self) -> int:
        return self._manifest.split_seed

    @property
    def manifest_sha256(self) -> str:
        return self._metadata.manifest_sha256

    @property
    def source_origin_audit_sha256(self) -> str:
        return self._metadata.source_origin_audit_sha256

    @property
    def source_split_report_sha256(self) -> str:
        return self._metadata.source_split_report_sha256

    def row_for_origin(self, anonymous_sample_id: str) -> SplitManifestRow:
        _plain_text(anonymous_sample_id, "anonymous_sample_id")
        try:
            return self._by_anonymous_id[anonymous_sample_id]
        except KeyError as error:
            raise SplitManifestError(
                "MANIFEST_ORIGIN_UNKNOWN",
                "anonymous origin is absent from verified manifest",
                evidence={"anonymous_sample_id": anonymous_sample_id},
            ) from error

    def row_for_legacy_origin(self, origin_id: str) -> SplitManifestRow:
        _plain_text(origin_id, "origin_id")
        try:
            rows = self._by_origin_id[origin_id]
        except KeyError as error:
            raise SplitManifestError(
                "MANIFEST_ORIGIN_UNKNOWN",
                "legacy origin is absent from verified manifest",
                evidence={"origin_id": origin_id},
            ) from error
        if len(rows) != 1:
            raise SplitManifestError(
                "MANIFEST_ORIGIN_AMBIGUOUS",
                "legacy origin maps to multiple anonymous editing tasks",
                evidence={"origin_id": origin_id, "match_count": len(rows)},
            )
        return rows[0]

    def rows_for_legacy_origin(
        self,
        origin_id: str,
    ) -> tuple[SplitManifestRow, ...]:
        _plain_text(origin_id, "origin_id")
        try:
            return self._by_origin_id[origin_id]
        except KeyError as error:
            raise SplitManifestError(
                "MANIFEST_ORIGIN_UNKNOWN",
                "legacy origin is absent from verified manifest",
                evidence={"origin_id": origin_id},
            ) from error

    def split_for_origin(self, anonymous_sample_id: str) -> SplitName:
        return self.row_for_origin(anonymous_sample_id).split

    def require_same_split(
        self,
        left_anonymous_sample_id: str,
        right_anonymous_sample_id: str,
    ) -> SplitName:
        left = self.row_for_origin(left_anonymous_sample_id)
        right = self.row_for_origin(right_anonymous_sample_id)
        if left.split is not right.split:
            raise SplitManifestError(
                "CROSS_SPLIT_EDGE",
                "origins belong to different verified splits",
                evidence={
                    "left_split": left.split.value,
                    "right_split": right.split.value,
                },
            )
        return left.split


def build_frozen_split_manifest(
    split_result: GroupStratifiedSplitResult,
    audit: OriginSplitAudit,
) -> FrozenSplitManifest:
    """Bind the exact T028 result to T026 molecular identity hashes."""

    if type(split_result) is not GroupStratifiedSplitResult:
        raise TypeError("split_result must be GroupStratifiedSplitResult")
    if type(audit) is not OriginSplitAudit:
        raise TypeError("audit must be OriginSplitAudit")
    if split_result.report.dataset_version != audit.dataset_version:
        raise SplitManifestError(
            "MANIFEST_SOURCE_MISMATCH",
            "T026 and T028 dataset versions differ",
        )
    audit_by_id = {row.anonymous_sample_id: row for row in audit.records}
    assignments_by_id = {
        row.anonymous_sample_id: row for row in split_result.assignments
    }
    if set(audit_by_id) != set(assignments_by_id):
        raise SplitManifestError(
            "MANIFEST_SOURCE_MISMATCH",
            "T026 and T028 origin inventories differ",
            evidence={
                "missing_from_split": tuple(
                    sorted(set(audit_by_id) - set(assignments_by_id))
                ),
                "unknown_in_split": tuple(
                    sorted(set(assignments_by_id) - set(audit_by_id))
                ),
            },
        )
    rows = []
    for anonymous_sample_id in sorted(audit_by_id):
        identity = audit_by_id[anonymous_sample_id]
        assignment = assignments_by_id[anonymous_sample_id]
        if not (
            identity.origin_id == assignment.origin_id
            and identity.subtask is assignment.subtask
        ):
            raise SplitManifestError(
                "MANIFEST_SOURCE_MISMATCH",
                "T026 and T028 origin identities differ",
                evidence={"anonymous_sample_id": anonymous_sample_id},
            )
        rows.append(
            SplitManifestRow(
                origin_id=assignment.origin_id,
                anonymous_sample_id=anonymous_sample_id,
                leakage_group_id=assignment.leakage_group_id,
                subtask=assignment.subtask,
                split=assignment.split,
                canonical_source_hash=identity.canonical_source_sha256,
                canonical_gt_hash=identity.canonical_gt_sha256,
                scaffold_hash=identity.scaffold_sha256,
                split_seed=split_result.report.split_seed,
                dataset_version=audit.dataset_version,
            )
        )
    return FrozenSplitManifest(
        rows=tuple(rows),
        source_origin_audit_sha256=_sha256(audit.to_json_bytes()),
        source_split_report_sha256=_sha256(split_result.to_json_bytes()),
    )


def _parse_csv_bytes(
    payload: bytes,
    *,
    source_origin_audit_sha256: str,
    source_split_report_sha256: str,
) -> FrozenSplitManifest:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SplitManifestError(
            "MANIFEST_ENCODING",
            "manifest must be strict UTF-8",
        ) from error
    if text.startswith("\ufeff") or not text.endswith("\n") or "\r" in text:
        raise SplitManifestError(
            "MANIFEST_ENCODING",
            "manifest must use UTF-8 without BOM and LF line endings",
        )
    try:
        table = tuple(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise SplitManifestError(
            "MANIFEST_CSV_INVALID",
            "manifest is not valid strict CSV",
        ) from error
    if not table or tuple(table[0]) != MANIFEST_FIELDS:
        raise SplitManifestError(
            "MANIFEST_FIELDS_INVALID",
            "manifest header must exactly match the frozen field order",
            evidence={"observed": () if not table else tuple(table[0])},
        )
    rows = tuple(
        SplitManifestRow.from_csv_fields(tuple(values)) for values in table[1:]
    )
    try:
        manifest = FrozenSplitManifest(
            rows=rows,
            source_origin_audit_sha256=source_origin_audit_sha256,
            source_split_report_sha256=source_split_report_sha256,
        )
    except (TypeError, ValueError) as error:
        raise SplitManifestError(
            "MANIFEST_INVARIANT_FAILED",
            "manifest rows violate frozen inventory constraints",
            evidence={"error_type": type(error).__name__, "detail": str(error)},
        ) from error
    if payload != manifest.to_csv_bytes():
        raise SplitManifestError(
            "MANIFEST_NOT_CANONICAL",
            "manifest bytes are not in canonical origin_id order/CSV form",
        )
    return manifest


class _DuplicateJsonKey(ValueError):
    pass


def _strict_json_mapping(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise _DuplicateJsonKey(key)
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        ValueError,
    ) as error:
        raise SplitManifestError(
            "MANIFEST_METADATA_INVALID",
            "metadata must be strict duplicate-free finite UTF-8 JSON",
        ) from error
    if not isinstance(value, Mapping):
        raise SplitManifestError(
            "MANIFEST_METADATA_INVALID",
            "metadata root must be an object",
        )
    return value


def load_verified_split_manifest(
    manifest_path: Path,
    metadata_path: Path,
    *,
    split_result: GroupStratifiedSplitResult,
    audit: OriginSplitAudit,
) -> VerifiedSplitManifest:
    """Load only bytes exactly bound to the supplied T026/T028 artifacts."""

    for path in (manifest_path, metadata_path):
        if not isinstance(path, Path):
            raise TypeError("manifest paths must be pathlib.Path")
        if not path.is_file():
            raise SplitManifestError(
                "MANIFEST_FILE_MISSING",
                "required frozen manifest artifact is missing",
                evidence={"filename": path.name},
            )
    expected = build_frozen_split_manifest(split_result, audit)
    manifest_payload = manifest_path.read_bytes()
    parsed = _parse_csv_bytes(
        manifest_payload,
        source_origin_audit_sha256=expected.source_origin_audit_sha256,
        source_split_report_sha256=expected.source_split_report_sha256,
    )
    if parsed != expected:
        raise SplitManifestError(
            "MANIFEST_ARTIFACT_MISMATCH",
            "manifest rows do not equal the supplied T026/T028 artifacts",
        )
    metadata_payload = metadata_path.read_bytes()
    observed_metadata = _strict_json_mapping(metadata_payload)
    expected_metadata = expected.metadata
    if observed_metadata != expected_metadata.to_dict():
        raise SplitManifestError(
            "MANIFEST_METADATA_MISMATCH",
            "metadata does not exactly bind manifest and source artifacts",
        )
    if metadata_payload != expected_metadata.to_json_bytes():
        raise SplitManifestError(
            "MANIFEST_METADATA_NOT_CANONICAL",
            "metadata bytes are not canonical timestamp-free JSON",
        )
    return VerifiedSplitManifest(
        parsed,
        expected_metadata,
        _token=_VERIFICATION_TOKEN,
    )


def write_frozen_split_manifest(
    manifest: FrozenSplitManifest,
    *,
    manifest_path: Path,
    metadata_path: Path,
) -> None:
    """Create missing artifacts, reject conflicting bytes, and remain idempotent."""

    if type(manifest) is not FrozenSplitManifest:
        raise TypeError("manifest must be FrozenSplitManifest")
    for path in (manifest_path, metadata_path):
        if not isinstance(path, Path):
            raise TypeError("manifest paths must be pathlib.Path")
    if manifest_path.resolve() == metadata_path.resolve():
        raise ValueError("manifest and metadata paths must differ")
    expected = (
        (manifest_path, manifest.to_csv_bytes()),
        (metadata_path, manifest.metadata.to_json_bytes()),
    )
    for path, payload in expected:
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            raise SplitManifestError(
                "MANIFEST_CONFLICT",
                "existing artifact differs; immutable overwrite refused",
                evidence={"filename": path.name},
            )
    for path, payload in expected:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_METADATA_FILENAME",
    "FROZEN_LEAKAGE_GROUP_COUNT",
    "MANIFEST_FIELDS",
    "MANIFEST_FORMAT_VERSION",
    "METADATA_FORMAT_VERSION",
    "FrozenSplitManifest",
    "SplitManifestError",
    "SplitManifestMetadata",
    "SplitManifestRow",
    "VerifiedSplitManifest",
    "build_frozen_split_manifest",
    "load_verified_split_manifest",
    "write_frozen_split_manifest",
]
