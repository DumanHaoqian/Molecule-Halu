"""ChemCoTBench-V2 molecule-editing input loader and anonymous-ID join."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molhallulens.domain import DomainValidationError

from .base import (
    InputAdapter,
    InputAdapterError,
    JoinedInputRecord,
    input_issue,
    raise_for_input_issues,
)


_ANONYMOUS_ID = "anonymous_sample_id"
_IDENTITY_FIELDS = ("task_family", "subtask", "reporting_task")
_SHARED_AUTHORITATIVE_FIELDS = (
    "task_family",
    "subtask",
    "reporting_task",
    "orig_id",
    "gt_smiles",
)
_RAW_REQUIRED_FIELDS = (
    _ANONYMOUS_ID,
    *_SHARED_AUTHORITATIVE_FIELDS,
    "indexed_smiles",
    "instruction",
)
_PROCESS_REQUIRED_FIELDS = (
    _ANONYMOUS_ID,
    *_SHARED_AUTHORITATIVE_FIELDS,
    "sample_id",
    "formal_cot_trace",
    "answer_smiles",
    "outcome",
    "parsed_reference_state",
    "verifier_checks",
)
_TEMPLATE_REQUIRED_FIELDS = (
    *_IDENTITY_FIELDS,
    "n_samples",
    "step_fields",
    "rdkit_reference_fields",
    "verifier_fields",
)
_MANIFEST_FIELDS = (
    "family",
    "subtask",
    "reporting_task",
    "n_samples",
    "raw_file",
    "process_file",
)


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON key: {key}")


class _NonFiniteJsonNumberError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    task_family: str
    subtask: str
    reporting_task: str
    n_samples: int
    raw_file: str
    process_file: str

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.task_family, self.subtask, self.reporting_task)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _parse_finite_json_float(payload: str) -> float:
    value = float(payload)
    if not math.isfinite(value):
        raise _NonFiniteJsonNumberError
    return value


def _reject_nonfinite_json_constant(_: str) -> float:
    raise _NonFiniteJsonNumberError


def _json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_length(value: Any) -> int | None:
    return len(value) if isinstance(value, (str, bytes, list, tuple, Mapping)) else None


def _comparison_evidence(
    field: str,
    left: Any,
    right: Any,
    *,
    left_source: str,
    right_source: str,
) -> dict[str, Any]:
    return {
        "field": field,
        "left_source": left_source,
        "right_source": right_source,
        "left_type": type(left).__name__,
        "right_type": type(right).__name__,
        "left_length": _value_length(left),
        "right_length": _value_length(right),
        "left_sha256": _json_sha256(left),
        "right_sha256": _json_sha256(right),
    }


def _identity_from(record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    values = tuple(record.get(field) for field in _IDENTITY_FIELDS)
    if any(type(value) is not str or not value.strip() for value in values):
        return None
    return values  # type: ignore[return-value]


def _record_type_issues(
    record: Mapping[str, Any],
    *,
    source: str,
    anonymous_sample_id: str,
) -> tuple[Any, ...]:
    issues = []
    string_fields = (
        _RAW_REQUIRED_FIELDS
        if source == "raw"
        else tuple(
            field
            for field in _PROCESS_REQUIRED_FIELDS
            if field not in {"sample_id", "formal_cot_trace", "outcome", "parsed_reference_state", "verifier_checks"}
        )
    )
    invalid_strings = tuple(
        sorted(
            field
            for field in string_fields
            if field in record
            and (type(record[field]) is not str or not record[field].strip())
        )
    )
    if invalid_strings:
        issues.append(
            input_issue(
                "INVALID_FIELD_TYPE",
                f"{source} record contains invalid required string fields",
                anonymous_sample_id=anonymous_sample_id,
                evidence={"source": source, "fields": invalid_strings},
            )
        )
    if source == "process":
        expected_types = {
            "sample_id": int,
            "formal_cot_trace": list,
            "outcome": bool,
            "parsed_reference_state": Mapping,
            "verifier_checks": Mapping,
        }
        invalid_typed = tuple(
            sorted(
                field
                for field, expected in expected_types.items()
                if field in record
                and (
                    type(record[field]) is not expected
                    if expected in {int, list, bool}
                    else not isinstance(record[field], expected)
                )
            )
        )
        if invalid_typed:
            issues.append(
                input_issue(
                    "INVALID_FIELD_TYPE",
                    "process record contains invalid structured fields",
                    anonymous_sample_id=anonymous_sample_id,
                    evidence={"source": source, "fields": invalid_typed},
                )
            )
    return tuple(issues)


def _index_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    required_fields: tuple[str, ...],
) -> tuple[dict[str, Mapping[str, Any]], set[str], tuple[Any, ...]]:
    issues = []
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    invalid_ids: set[str] = set()
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        return {}, set(), (
            input_issue(
                "INVALID_SOURCE_SHAPE",
                f"{source} records must be an array-like sequence",
                evidence={"source": source, "actual_type": type(records).__name__},
            ),
        )
    for record in records:
        if not isinstance(record, Mapping):
            issues.append(
                input_issue(
                    "INVALID_SOURCE_SHAPE",
                    f"{source} array entries must be objects",
                    evidence={
                        "source": source,
                        "actual_type": type(record).__name__,
                        "record_sha256": _json_sha256(record),
                    },
                )
            )
            continue
        raw_id = record.get(_ANONYMOUS_ID)
        anonymous_id = raw_id if type(raw_id) is str and raw_id.strip() else None
        missing_fields = tuple(sorted(field for field in required_fields if field not in record))
        if missing_fields:
            issues.append(
                input_issue(
                    "MISSING_REQUIRED_FIELD",
                    f"{source} record is missing required fields",
                    anonymous_sample_id=anonymous_id,
                    evidence={
                        "source": source,
                        "fields": missing_fields,
                        "record_sha256": _json_sha256(record),
                    },
                )
            )
        if anonymous_id is None:
            issues.append(
                input_issue(
                    "INVALID_ID",
                    f"{source} anonymous_sample_id must be a non-empty string",
                    evidence={
                        "source": source,
                        "id_type": type(raw_id).__name__,
                        "record_sha256": _json_sha256(record),
                    },
                )
            )
            continue
        buckets[anonymous_id].append(record)
        record_issues = _record_type_issues(
            record,
            source=source,
            anonymous_sample_id=anonymous_id,
        )
        if missing_fields or record_issues:
            invalid_ids.add(anonymous_id)
            issues.extend(record_issues)

    duplicate_code = "DUPLICATE_RAW_ID" if source == "raw" else "DUPLICATE_PROCESS_ID"
    index: dict[str, Mapping[str, Any]] = {}
    for anonymous_id, matches in buckets.items():
        if len(matches) != 1:
            invalid_ids.add(anonymous_id)
            issues.append(
                input_issue(
                    duplicate_code,
                    f"{source} anonymous_sample_id is duplicated",
                    anonymous_sample_id=anonymous_id,
                    evidence={
                        "source": source,
                        "count": len(matches),
                        "record_sha256s": tuple(sorted(_json_sha256(record) for record in matches)),
                    },
                )
            )
            continue
        index[anonymous_id] = matches[0]
    return index, invalid_ids, tuple(issues)


def _template_inventory(
    template: Mapping[str, Any],
    field: str,
    *,
    identity: tuple[str, str, str],
) -> tuple[tuple[str, ...] | None, tuple[Any, ...]]:
    value = template.get(field)
    if not isinstance(value, (list, tuple)) or any(type(item) is not str or not item for item in value):
        return None, (
            input_issue(
                "TEMPLATE_INVENTORY_MISMATCH",
                "formal template inventory must be a string array",
                evidence={"identity": identity, "field": field},
            ),
        )
    values = tuple(value)
    if len(values) != len(set(values)):
        return None, (
            input_issue(
                "TEMPLATE_INVENTORY_MISMATCH",
                "formal template inventory cannot contain duplicates",
                evidence={"identity": identity, "field": field},
            ),
        )
    return values, ()


def _index_templates(
    templates: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, Any]],
    set[tuple[str, str, str]],
    tuple[Any, ...],
]:
    issues = []
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    invalid_identities: set[tuple[str, str, str]] = set()
    if isinstance(templates, (str, bytes)) or not isinstance(templates, Sequence):
        return {}, set(), (
            input_issue(
                "INVALID_SOURCE_SHAPE",
                "formal templates must be an array-like sequence",
                evidence={"source": "formal_template", "actual_type": type(templates).__name__},
            ),
        )
    for template in templates:
        if not isinstance(template, Mapping):
            issues.append(
                input_issue(
                    "INVALID_SOURCE_SHAPE",
                    "formal template entries must be objects",
                    evidence={
                        "source": "formal_template",
                        "actual_type": type(template).__name__,
                        "record_sha256": _json_sha256(template),
                    },
                )
            )
            continue
        missing_fields = tuple(sorted(field for field in _TEMPLATE_REQUIRED_FIELDS if field not in template))
        identity = _identity_from(template)
        if missing_fields:
            issues.append(
                input_issue(
                    "MISSING_REQUIRED_FIELD",
                    "formal template is missing required fields",
                    evidence={
                        "source": "formal_template",
                        "fields": missing_fields,
                        "identity": () if identity is None else identity,
                        "record_sha256": _json_sha256(template),
                    },
                )
            )
        if identity is None:
            issues.append(
                input_issue(
                    "TEMPLATE_FIELD_CONFLICT",
                    "formal template discriminators must be non-empty strings",
                    evidence={
                        "source": "formal_template",
                        "fields": _IDENTITY_FIELDS,
                        "record_sha256": _json_sha256(template),
                    },
                )
            )
            continue
        buckets[identity].append(template)
        if missing_fields:
            invalid_identities.add(identity)
        if type(template.get("n_samples")) is not int or template["n_samples"] <= 0:
            invalid_identities.add(identity)
            issues.append(
                input_issue(
                    "TEMPLATE_FIELD_CONFLICT",
                    "formal template n_samples must be a positive integer",
                    evidence={"identity": identity, "field": "n_samples"},
                )
            )
        for field in ("step_fields", "rdkit_reference_fields", "verifier_fields"):
            _, inventory_issues = _template_inventory(template, field, identity=identity)
            if inventory_issues:
                invalid_identities.add(identity)
                issues.extend(inventory_issues)

    registry: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    duplicate_identities: set[tuple[str, str, str]] = set()
    for identity, matches in buckets.items():
        if len(matches) != 1:
            duplicate_identities.add(identity)
            invalid_identities.add(identity)
            issues.append(
                input_issue(
                    "DUPLICATE_TEMPLATE",
                    "formal template discriminator is duplicated",
                    evidence={
                        "identity": identity,
                        "count": len(matches),
                        "template_sha256s": tuple(sorted(_json_sha256(item) for item in matches)),
                    },
                )
            )
            continue
        registry[identity] = matches[0]
    return registry, invalid_identities | duplicate_identities, tuple(issues)


class ChemCoTMolEditAdapter(InputAdapter):
    """Load the active molecule-editing pilot and join it without positional assumptions."""

    EXPECTED_ORIGIN_COUNT = 150
    MANIFEST_NAME = "active_benchmark_manifest.csv"
    VALIDATOR_ID = "chemcot_mol_edit_input_join"

    def load(self, dataset_root: Path) -> tuple[JoinedInputRecord, ...]:
        root = Path(dataset_root).resolve()
        entries = self._load_manifest(root)
        load_issues = []
        raw_records: list[Mapping[str, Any]] = []
        process_records: list[Mapping[str, Any]] = []
        for entry in entries:
            raw_path = self._resolve_manifest_path(root, entry.raw_file, source="raw")
            process_path = self._resolve_manifest_path(root, entry.process_file, source="process")
            raw_batch = self._load_json_array(raw_path, root=root, source="raw")
            process_batch = self._load_json_array(process_path, root=root, source="process")
            for source, records in (("raw", raw_batch), ("process", process_batch)):
                if len(records) != entry.n_samples:
                    load_issues.append(
                        input_issue(
                            "ORIGIN_COUNT_MISMATCH",
                            f"{source} shard count does not match manifest",
                            evidence={
                                "source": source,
                                "identity": entry.identity,
                                "manifest_count": entry.n_samples,
                                "actual_count": len(records),
                            },
                        )
                    )
                load_issues.extend(
                    self._manifest_identity_issues(records, entry=entry, source=source)
                )
            raw_records.extend(raw_batch)
            process_records.extend(process_batch)

        required_identities = {entry.identity for entry in entries}
        template_paths = []
        for family in sorted({entry.task_family for entry in entries}):
            family_dir = self._resolve_manifest_path(
                root,
                str(Path("formal_templates") / family),
                source="formal_template",
            )
            for discovered_path in sorted(family_dir.rglob("*.json")):
                template_paths.append(
                    self._resolve_manifest_path(
                        root,
                        str(discovered_path.relative_to(root)),
                        source="formal_template",
                    )
                )
        loaded_templates = [
            self._load_json_object(path, root=root, source="formal_template")
            for path in template_paths
        ]
        templates = [
            template
            for template in loaded_templates
            if (identity := _identity_from(template)) is None
            or identity in required_identities
        ]
        try:
            joined = self.join_records(
                raw_records=raw_records,
                process_records=process_records,
                formal_templates=templates,
                expected_origin_count=self.EXPECTED_ORIGIN_COUNT,
            )
        except InputAdapterError as error:
            load_issues.extend(error.report.issues)
            joined = ()
        raise_for_input_issues("chemcot_mol_edit_dataset_load", load_issues)
        return joined

    @classmethod
    def join_records(
        cls,
        *,
        raw_records: Sequence[Mapping[str, Any]],
        process_records: Sequence[Mapping[str, Any]],
        formal_templates: Sequence[Mapping[str, Any]],
        expected_origin_count: int | None = None,
    ) -> tuple[JoinedInputRecord, ...]:
        """Join already loaded sources by exact anonymous ID and template discriminator."""

        if expected_origin_count is not None and (
            type(expected_origin_count) is not int or expected_origin_count <= 0
        ):
            raise TypeError("expected_origin_count must be a positive integer or None")
        raw_index, raw_invalid, raw_issues = _index_records(
            raw_records,
            source="raw",
            required_fields=_RAW_REQUIRED_FIELDS,
        )
        process_index, process_invalid, process_issues = _index_records(
            process_records,
            source="process",
            required_fields=_PROCESS_REQUIRED_FIELDS,
        )
        template_index, template_invalid, template_issues = _index_templates(formal_templates)
        issues = [*raw_issues, *process_issues, *template_issues]

        raw_ids = set(raw_index)
        process_ids = set(process_index)
        for anonymous_id in sorted(raw_ids - process_ids):
            if anonymous_id not in process_invalid:
                issues.append(
                    input_issue(
                        "MISSING_PROCESS_RECORD",
                        "raw record has no process partner",
                        anonymous_sample_id=anonymous_id,
                        evidence={"source": "process"},
                    )
                )
        for anonymous_id in sorted(process_ids - raw_ids):
            if anonymous_id not in raw_invalid:
                issues.append(
                    input_issue(
                        "MISSING_RAW_RECORD",
                        "process record has no raw partner",
                        anonymous_sample_id=anonymous_id,
                        evidence={"source": "raw"},
                    )
                )

        candidate_records: list[tuple[str, Mapping[str, Any], Mapping[str, Any], tuple[str, str, str]]] = []
        missing_templates: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for anonymous_id in sorted(raw_ids & process_ids):
            if anonymous_id in raw_invalid or anonymous_id in process_invalid:
                continue
            raw = raw_index[anonymous_id]
            process = process_index[anonymous_id]
            conflict = False
            for field in _SHARED_AUTHORITATIVE_FIELDS:
                if field not in raw or field not in process:
                    continue
                if type(raw[field]) is not type(process[field]) or raw[field] != process[field]:
                    conflict = True
                    issues.append(
                        input_issue(
                            "FIELD_CONFLICT",
                            f"raw and process disagree on authoritative field {field}",
                            anonymous_sample_id=anonymous_id,
                            evidence=_comparison_evidence(
                                field,
                                raw[field],
                                process[field],
                                left_source="raw",
                                right_source="process",
                            ),
                        )
                    )
            identity = _identity_from(raw)
            if conflict or identity is None:
                continue
            if identity not in template_index or identity in template_invalid:
                if identity not in template_invalid:
                    missing_templates[identity].append(anonymous_id)
                continue
            candidate_records.append((anonymous_id, raw, process, identity))

        for identity, anonymous_ids in sorted(missing_templates.items()):
            issues.append(
                input_issue(
                    "MISSING_TEMPLATE",
                    "joined records have no unique formal template",
                    evidence={
                        "identity": identity,
                        "origin_count": len(anonymous_ids),
                        "anonymous_sample_ids": tuple(sorted(anonymous_ids)),
                    },
                )
            )

        raw_identity_counts = Counter(
            identity
            for record in raw_index.values()
            if (identity := _identity_from(record)) is not None
        )
        process_identity_counts = Counter(
            identity
            for record in process_index.values()
            if (identity := _identity_from(record)) is not None
        )
        for identity, template in sorted(template_index.items()):
            if identity in template_invalid:
                continue
            expected = template["n_samples"]
            raw_count = raw_identity_counts[identity]
            process_count = process_identity_counts[identity]
            if expected != raw_count or expected != process_count:
                issues.append(
                    input_issue(
                        "ORIGIN_COUNT_MISMATCH",
                        "formal template n_samples does not match its source records",
                        evidence={
                            "identity": identity,
                            "template_count": expected,
                            "raw_count": raw_count,
                            "process_count": process_count,
                        },
                    )
                )

        joined = []
        for anonymous_id, raw, process, identity in candidate_records:
            template = template_index[identity]
            step_fields, _ = _template_inventory(template, "step_fields", identity=identity)
            rdkit_fields, _ = _template_inventory(
                template,
                "rdkit_reference_fields",
                identity=identity,
            )
            verifier_fields, _ = _template_inventory(
                template,
                "verifier_fields",
                identity=identity,
            )
            if step_fields is None or rdkit_fields is None or verifier_fields is None:
                continue
            parsed_state = process["parsed_reference_state"]
            verifier_checks = process["verifier_checks"]
            expected_state = set(step_fields) | set(rdkit_fields)
            expected_verifier = set(verifier_fields) | {"all_pass"}
            inventory_problem = False
            for field, actual, expected in (
                ("parsed_reference_state", set(parsed_state), expected_state),
                ("verifier_checks", set(verifier_checks), expected_verifier),
            ):
                missing = tuple(sorted(expected - actual))
                unexpected = tuple(sorted(actual - expected))
                if missing or unexpected:
                    inventory_problem = True
                    issues.append(
                        input_issue(
                            "TEMPLATE_INVENTORY_MISMATCH",
                            f"process {field} does not match the formal template inventory",
                            anonymous_sample_id=anonymous_id,
                            evidence={
                                "identity": identity,
                                "field": field,
                                "missing_fields": missing,
                                "unexpected_fields": unexpected,
                            },
                        )
                    )
            if not inventory_problem:
                try:
                    joined_record = JoinedInputRecord(
                        anonymous_sample_id=anonymous_id,
                        raw_record=raw,
                        process_record=process,
                        formal_template=template,
                    )
                except (DomainValidationError, TypeError, ValueError) as error:
                    issues.append(
                        input_issue(
                            "INVALID_RECORD_VALUE",
                            "joined source values cannot be represented safely",
                            anonymous_sample_id=anonymous_id,
                            evidence={
                                "error_type": type(error).__name__,
                                "raw_sha256": _json_sha256(raw),
                                "process_sha256": _json_sha256(process),
                                "template_sha256": _json_sha256(template),
                            },
                        )
                    )
                else:
                    joined.append(joined_record)

        if expected_origin_count is not None and len(joined) != expected_origin_count:
            issues.append(
                input_issue(
                    "ORIGIN_COUNT_MISMATCH",
                    "joined origin count does not match the frozen Pilot count",
                    evidence={
                        "expected_count": expected_origin_count,
                        "joined_count": len(joined),
                    },
                )
            )
        raise_for_input_issues(cls.VALIDATOR_ID, issues)
        return tuple(sorted(joined, key=lambda record: record.anonymous_sample_id))

    @classmethod
    def _load_manifest(cls, root: Path) -> tuple[_ManifestEntry, ...]:
        path = root / cls.MANIFEST_NAME
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = tuple(reader.fieldnames or ())
                rows = list(reader)
        except OSError as error:
            raise InputAdapterError(
                cls._single_issue_report(
                    "SOURCE_READ_ERROR",
                    "active benchmark manifest could not be read",
                    evidence={"source": cls.MANIFEST_NAME, "error_type": type(error).__name__},
                )
            ) from error
        issues = []
        missing_columns = tuple(sorted(set(_MANIFEST_FIELDS) - set(fieldnames)))
        if missing_columns:
            issues.append(
                input_issue(
                    "MISSING_REQUIRED_FIELD",
                    "active benchmark manifest is missing required columns",
                    evidence={"source": cls.MANIFEST_NAME, "fields": missing_columns},
                )
            )
        if len(fieldnames) != len(set(fieldnames)):
            issues.append(
                input_issue(
                    "DUPLICATE_MANIFEST_FIELD",
                    "active benchmark manifest contains duplicate columns",
                    evidence={"source": cls.MANIFEST_NAME},
                )
            )
        entries = []
        for row in rows:
            values = tuple(row.get(field) for field in _MANIFEST_FIELDS)
            if any(type(value) is not str or not value.strip() for value in values):
                issues.append(
                    input_issue(
                        "MISSING_REQUIRED_FIELD",
                        "active benchmark manifest row has empty required fields",
                        evidence={"source": cls.MANIFEST_NAME, "row_sha256": _json_sha256(row)},
                    )
                )
                continue
            try:
                n_samples = int(row["n_samples"])
            except ValueError:
                n_samples = -1
            if n_samples <= 0 or str(n_samples) != row["n_samples"]:
                issues.append(
                    input_issue(
                        "ORIGIN_COUNT_MISMATCH",
                        "manifest n_samples must be a canonical positive integer",
                        evidence={"source": cls.MANIFEST_NAME, "row_sha256": _json_sha256(row)},
                    )
                )
                continue
            entries.append(
                _ManifestEntry(
                    task_family=row["family"],
                    subtask=row["subtask"],
                    reporting_task=row["reporting_task"],
                    n_samples=n_samples,
                    raw_file=row["raw_file"],
                    process_file=row["process_file"],
                )
            )
        identity_counts = Counter(entry.identity for entry in entries)
        raw_counts = Counter(entry.raw_file for entry in entries)
        process_counts = Counter(entry.process_file for entry in entries)
        for field, counts in (
            ("identity", identity_counts),
            ("raw_file", raw_counts),
            ("process_file", process_counts),
        ):
            for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
                if count > 1:
                    issues.append(
                        input_issue(
                            "DUPLICATE_MANIFEST_SHARD",
                            "active benchmark manifest contains a duplicate shard",
                            evidence={
                                "source": cls.MANIFEST_NAME,
                                "field": field,
                                "count": count,
                                "value_sha256": _json_sha256(value),
                            },
                        )
                    )
        if sum(entry.n_samples for entry in entries) != cls.EXPECTED_ORIGIN_COUNT:
            issues.append(
                input_issue(
                    "ORIGIN_COUNT_MISMATCH",
                    "manifest origin total does not match the frozen Pilot count",
                    evidence={
                        "expected_count": cls.EXPECTED_ORIGIN_COUNT,
                        "manifest_count": sum(entry.n_samples for entry in entries),
                    },
                )
            )
        raise_for_input_issues("chemcot_mol_edit_manifest", issues)
        return tuple(entries)

    @staticmethod
    def _resolve_manifest_path(root: Path, relative: str, *, source: str) -> Path:
        raw_path = Path(relative)
        if raw_path.is_absolute() or ".." in raw_path.parts:
            raise InputAdapterError(
                ChemCoTMolEditAdapter._single_issue_report(
                    "INVALID_SOURCE_PATH",
                    "manifest paths must be relative and cannot traverse parents",
                    evidence={"source": source, "path_sha256": _json_sha256(relative)},
                )
            )
        resolved = (root / raw_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise InputAdapterError(
                ChemCoTMolEditAdapter._single_issue_report(
                    "INVALID_SOURCE_PATH",
                    "manifest path resolves outside the dataset root",
                    evidence={"source": source, "path_sha256": _json_sha256(relative)},
                )
            ) from error
        return resolved

    @classmethod
    def _load_json_array(
        cls,
        path: Path,
        *,
        root: Path,
        source: str,
    ) -> list[Mapping[str, Any]]:
        value = cls._load_json(path, root=root, source=source)
        if type(value) is not list:
            raise InputAdapterError(
                cls._single_issue_report(
                    "INVALID_SOURCE_SHAPE",
                    f"{source} JSON root must be an array",
                    evidence={"source": source, "actual_type": type(value).__name__},
                )
            )
        return value

    @classmethod
    def _load_json_object(
        cls,
        path: Path,
        *,
        root: Path,
        source: str,
    ) -> Mapping[str, Any]:
        value = cls._load_json(path, root=root, source=source)
        if type(value) is not dict:
            raise InputAdapterError(
                cls._single_issue_report(
                    "INVALID_SOURCE_SHAPE",
                    f"{source} JSON root must be an object",
                    evidence={"source": source, "actual_type": type(value).__name__},
                )
            )
        return value

    @classmethod
    def _load_json(cls, path: Path, *, root: Path, source: str) -> Any:
        relative = str(path.relative_to(root))
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as error:
            raise InputAdapterError(
                cls._single_issue_report(
                    "SOURCE_READ_ERROR",
                    f"{source} JSON source could not be read",
                    evidence={"source": source, "path": relative, "error_type": type(error).__name__},
                )
            ) from error
        try:
            return json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_float=_parse_finite_json_float,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except _DuplicateJsonKeyError as error:
            raise InputAdapterError(
                cls._single_issue_report(
                    "DUPLICATE_JSON_KEY",
                    f"{source} JSON source contains a duplicate object key",
                    evidence={"source": source, "path": relative, "field": error.key},
                )
            ) from error
        except json.JSONDecodeError as error:
            raise InputAdapterError(
                cls._single_issue_report(
                    "INVALID_JSON",
                    f"{source} JSON source is malformed",
                    evidence={"source": source, "path": relative, "line": error.lineno, "column": error.colno},
                )
            ) from error
        except _NonFiniteJsonNumberError as error:
            raise InputAdapterError(
                cls._single_issue_report(
                    "INVALID_JSON_NUMBER",
                    f"{source} JSON source contains a non-finite number",
                    evidence={"source": source, "path": relative},
                )
            ) from error

    @staticmethod
    def _manifest_identity_issues(
        records: Sequence[Mapping[str, Any]],
        *,
        entry: _ManifestEntry,
        source: str,
    ) -> tuple[Any, ...]:
        issues = []
        expected = {
            "task_family": entry.task_family,
            "subtask": entry.subtask,
            "reporting_task": entry.reporting_task,
        }
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw_id = record.get(_ANONYMOUS_ID)
            anonymous_id = raw_id if type(raw_id) is str and raw_id.strip() else None
            for field, expected_value in expected.items():
                if field not in record:
                    continue
                if type(record[field]) is not str or record[field] != expected_value:
                    issues.append(
                        input_issue(
                            "FIELD_CONFLICT",
                            f"{source} record conflicts with manifest field {field}",
                            anonymous_sample_id=anonymous_id,
                            evidence=_comparison_evidence(
                                field,
                                record[field],
                                expected_value,
                                left_source=source,
                                right_source="manifest",
                            ),
                        )
                    )
        return tuple(issues)

    @staticmethod
    def _single_issue_report(
        code: str,
        message: str,
        *,
        evidence: Mapping[str, Any],
    ):
        from molhallulens.domain import ValidationReport

        return ValidationReport(
            "chemcot_mol_edit_source_loader",
            (input_issue(code, message, evidence=evidence),),
        )
