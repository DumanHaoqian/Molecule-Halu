"""T052 final release QA for the MolHalluLens molecule-editing pilot.

The release is frozen by dataset version, exact artifact paths, exact row
counts, and exact record/origin identity sets.  Per the explicit user
instruction, this module neither imports nor calls a digest implementation.
The original digest-based acceptance item is retained as an auditable override
and is never represented as a passed check.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

T052_FORMAT_VERSION = "t052_release_qa_v1"
T052_MANIFEST_FORMAT_VERSION = "t052_dataset_manifest_v1"
T052_SHORTCUT_FORMAT_VERSION = "t052_full_release_shortcut_baselines_v1"
T052_LEDGER_FORMAT_VERSION = "t052_poe_zero_call_ledger_v1"
T052_LEDGER_DESCRIPTOR_FORMAT_VERSION = "t052_poe_ledger_export_descriptor_v1"
T052_RELEASE_ID = "molhallulens_moledit_pilot_v1"
DATASET_VERSION = "pilot_v1"
ACTIVATION_ALIGNMENT = "post_token_h_t"
EXPECTED_ACTIVATION_SHARD_COUNT = 401
EXPECTED_ACTIVATION_TOKEN_COUNT = 1_824_606
POE_MODEL_ID = "gpt-5.4-mini"
RELEASE_TIMESTAMP = "2026-08-30T06:30:00+08:00"

from molhallulens.config.paths import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
DEFAULT_RELEASE_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset"
DEFAULT_EXTERNAL_REPORT_PATH = (
    DEFAULT_PROJECT_ROOT / "Dataset/reports/t052_release_qa.json"
)
PRIVATE_LEDGER_ENV = "MOLHALLULENS_PRIVATE_LEDGER_PATH"
LEDGER_DESCRIPTOR_RELATIVE_PATH = "reports/poe_usage_ledger_export_descriptor.json"

SPLIT_ORIGIN_COUNTS = MappingProxyType({"train": 100, "validation": 25, "test": 25})
SPLIT_RECORD_COUNTS = MappingProxyType({"train": 800, "validation": 200, "test": 200})
SPLITS = tuple(SPLIT_RECORD_COUNTS)
POLICIES = ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL")
FAMILIES = MappingProxyType(
    {
        "records": "records",
        "oracle": "oracle",
        "state_graphs": "state_graphs",
        "tokenized": "tokenized/chemdfm_r",
        "provenance": "provenance",
    }
)

_DIRECT_TOKEN_ARRAYS = (
    "input_ids",
    "attention_mask",
    "offset_mapping",
    "segment_ids",
    "evaluation_mask",
    "hallucination_core_mask",
    "error_any_mask",
    "local_falsehood_mask",
    "off_task_branch_mask",
    "reasoning_mask",
    "answer_mask",
    "boundary_ambiguous_mask",
    "error_char_fraction",
)
_NESTED_TOKEN_ARRAYS = (
    "semantic_type_masks",
    "edit_subtype_masks",
    "causal_role_masks",
)
_BINARY_TOKEN_ARRAYS = frozenset(
    {
        "attention_mask",
        "evaluation_mask",
        "hallucination_core_mask",
        "error_any_mask",
        "local_falsehood_mask",
        "off_task_branch_mask",
        "reasoning_mask",
        "answer_mask",
        "boundary_ambiguous_mask",
    }
)
_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "build_provenance",
        "candidate_graph",
        "candidate_state_graph",
        "gt_smiles",
        "oracle",
        "oracle_gt",
        "reference_graph",
        "reference_state_graph",
    }
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "poe_api_key",
        "password",
        "secret",
        "set_cookie",
    }
)
_CREDENTIAL_TEXT = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*(?:bearer\s+)?\S+|"
    r"poe_api_key\s*=\s*\S+|(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,})"
)
_EXPECTED_DETECTOR_FIELDS = frozenset(
    {"indexed_smiles", "instruction", "reasoning_chain", "final_answer"}
)
_EXPECTED_SEGMENTS = (
    ("indexed_smiles", "source", "<MOLECULE>"),
    ("instruction", "instruction", "<INSTRUCTION>"),
    ("reasoning_chain", "reasoning", "<REASONING>"),
    ("final_answer", "final_answer", "<FINAL_ANSWER>"),
)

_ARTIFACT_VALIDATOR_IDS = (
    "molhallulens.validation.hallucination_semantics.v1",
    "molhallulens.validation.propagation.v1",
    "molhallulens.validation.renderer.v1",
    "molhallulens.validation.token_alignment.v1",
)
_BUNDLE_VALIDATOR_ID = "molhallulens.validation.bundle_integrity.v1"
_ARTIFACT_CHAIN_ID = "molhallulens.validation.artifact_chain.v1"
_STRICT_REPORT_FORMATS = MappingProxyType(
    {
        "train": "t048_train_strict_validation_v1",
        "validation": "t049_validation_strict_validation_v1",
        "test": "t050_test_strict_validation_v1",
    }
)
_BUILD_REPORT_FORMATS = MappingProxyType(
    {
        "train": "t048_train_build_report_v1",
        "validation": "t049_validation_build_report_v1",
        "test": "t050_test_build_report_v1",
    }
)
_BUILD_IDENTITIES = MappingProxyType(
    {
        "train": ("train_build_id", "t048_frozen_train_100_origin_v1"),
        "validation": (
            "validation_build_id",
            "t049_frozen_validation_25_origin_v1",
        ),
        "test": ("test_build_id", "t050_frozen_test_25_origin_v1"),
    }
)
_REQUIRED_RECORD_VERIFICATION = frozenset(
    {
        "bundle_integrity_verified",
        "graph_edit_verified",
        "propagation_verified",
        "rdkit_sanitize",
        "renderer_verified",
        "span_verified",
        "token_alignment_verified",
    }
)
_REQUIRED_TEST_USAGE = MappingProxyType(
    {
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
    }
)
_REQUIRED_FROZEN_DESIGN = MappingProxyType(
    {
        "candidate_generation_rules_frozen_before_test_build": True,
        "operator_rules_frozen_before_test_build": True,
        "propagation_rules_frozen_before_test_build": True,
        "recipe_order_frozen_before_test_build": True,
        "renderer_rules_frozen_before_test_build": True,
        "thresholds_frozen_before_test_build": True,
        "test_failure_may_add_remove_or_reorder_recipe": False,
        "test_failure_may_change_operator_or_candidate_rule": False,
        "test_failure_may_change_propagation_or_renderer": False,
    }
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_private_ledger_path(
    private_ledger_path: Path | None,
    *,
    release_root: Path,
    project_root: Path,
) -> Path:
    """Resolve external private storage without probing or mutating it."""

    if private_ledger_path is None:
        configured = os.environ.get(PRIVATE_LEDGER_ENV)
        candidate = (
            Path(configured)
            if configured
            else Path.home()
            / ".local/share/molhallulens/pilot_v1/poe_usage_ledger.json"
        )
    else:
        candidate = Path(private_ledger_path)
    candidate = candidate.expanduser()
    if not candidate.is_absolute():
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_PATH",
            "private ledger path must be absolute",
            evidence={"environment_variable": PRIVATE_LEDGER_ENV},
        )
    resolved = candidate.resolve(strict=False)
    project = project_root.resolve(strict=False)
    release = release_root.resolve(strict=False)
    if _is_within(resolved, project) or _is_within(resolved, release):
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_LOCATION",
            "private Poe ledger must be stored outside the project and release tree",
            evidence={"path": str(resolved)},
        )
    return resolved


_REQUIRED_SHORTCUT_BASELINES = frozenset(
    {
        "metadata_only_logistic",
        "span_only_char_tfidf_logistic",
        "reasoning_only_word_tfidf_logistic",
        "nearest_neighbor_retrieval_k5",
        "smiles_validity",
        "visible_reasoning_answer_graph_comparator",
        "hidden_oracle_answer_graph_comparator",
        "slices",
    }
)


class ReleaseQAError(RuntimeError):
    """One fail-closed release gate failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("release QA error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("release QA error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("release QA error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


def _stable(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=repr)
    raise TypeError(f"unsupported release QA value: {type(value).__qualname__}")


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            _stable(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseQAError(
            "RELEASE_ARTIFACT_MISSING",
            f"required {label} artifact is missing",
            evidence={"path": str(path)},
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseQAError(
            "RELEASE_JSON_INVALID",
            f"{label} is not valid UTF-8 JSON",
            evidence={"path": str(path)},
        ) from error
    if not isinstance(value, dict):
        raise ReleaseQAError(
            "RELEASE_JSON_OBJECT_REQUIRED",
            f"{label} must be a JSON object",
            evidence={"path": str(path)},
        )
    return value


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ReleaseQAError(
            "RELEASE_ARTIFACT_MISSING",
            f"required {label} artifact is missing",
            evidence={"path": str(path)},
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ReleaseQAError(
                "RELEASE_JSONL_BLANK_LINE",
                f"{label} contains a blank row",
                evidence={"path": str(path), "line_number": line_number},
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReleaseQAError(
                "RELEASE_JSONL_INVALID",
                f"{label} contains invalid JSON",
                evidence={"path": str(path), "line_number": line_number},
            ) from error
        if not isinstance(value, dict):
            raise ReleaseQAError(
                "RELEASE_JSONL_OBJECT_REQUIRED",
                f"{label} rows must be JSON objects",
                evidence={"path": str(path), "line_number": line_number},
            )
        rows.append(value)
    return tuple(rows)


def _normalized_key(value: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", value.casefold()) if part)


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_keys(item)


def _assert_public_boundary(record: Mapping[str, Any]) -> None:
    forbidden = {
        key
        for key in _nested_keys(record)
        if _normalized_key(key) in _PUBLIC_FORBIDDEN_KEYS
    }
    if forbidden:
        raise ReleaseQAError(
            "RELEASE_GT_BOUNDARY",
            "detector-visible record contains an oracle-only field",
            evidence={
                "record_id": record.get("record_id"),
                "forbidden_keys": tuple(sorted(forbidden)),
            },
        )


def _assert_secret_free(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(str(key)) in _SECRET_KEYS:
                raise ReleaseQAError(
                    "RELEASE_SECRET_KEY",
                    "release artifact contains a credential-shaped key",
                    evidence={"location": location, "key": str(key)},
                )
            _assert_secret_free(item, location)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_secret_free(item, location)
    elif type(value) is str and _CREDENTIAL_TEXT.search(value):
        raise ReleaseQAError(
            "RELEASE_SECRET_TEXT",
            "release artifact contains credential-shaped text",
            evidence={"location": location},
        )


def _indexed(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    split: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if type(record_id) is not str or not record_id:
            raise ReleaseQAError(
                "RELEASE_RECORD_ID_INVALID",
                "artifact row lacks a record identity",
                evidence={"family": family, "split": split},
            )
        if record_id in result:
            raise ReleaseQAError(
                "RELEASE_RECORD_ID_DUPLICATE",
                "artifact family contains a duplicate record identity",
                evidence={"family": family, "split": split, "record_id": record_id},
            )
        result[record_id] = row
    return result


def _resolve_shard_path(
    root: Path,
    index_path: Path,
    path_value: Any,
) -> tuple[str, Path]:
    if type(path_value) is not str or not path_value:
        raise ReleaseQAError(
            "RELEASE_TOKEN_SHARD_PATH",
            "token shard index contains an invalid path",
        )
    value = Path(path_value)
    if value.is_absolute() or ".." in value.parts:
        raise ReleaseQAError(
            "RELEASE_TOKEN_SHARD_PATH",
            "token shard paths must be relative and cannot contain parent traversal",
            evidence={"path": path_value},
        )
    release = root.resolve(strict=False)
    shard_root = index_path.parent.resolve(strict=False)
    resolved = (shard_root / value).resolve(strict=False)
    if not _is_within(shard_root, release) or not _is_within(resolved, shard_root):
        raise ReleaseQAError(
            "RELEASE_TOKEN_SHARD_PATH",
            "token shard path escapes the frozen release root",
            evidence={"path": path_value},
        )
    relative = resolved.relative_to(release).as_posix()
    return relative, resolved


def _read_tokenized_split(
    root: Path,
    split: str,
    used_shard_paths: set[Path],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Read canonical T051 rows, with an ordered Git-shard fallback."""

    canonical_path = root / "tokenized/chemdfm_r" / f"{split}.jsonl"
    index_path = root / "tokenized/chemdfm_r/git_shards/index.json"
    canonical = (
        _read_jsonl(canonical_path, f"tokenized/{split} canonical")
        if canonical_path.is_file()
        else None
    )
    shard_rows: tuple[dict[str, Any], ...] | None = None
    shard_inventory: list[dict[str, Any]] = []
    if index_path.is_file():
        index = _read_json(index_path, "T051 token shard index")
        split_map = index.get("splits")
        spec = split_map.get(split) if isinstance(split_map, Mapping) else None
        shards = spec.get("shards") if isinstance(spec, Mapping) else None
        max_bytes = index.get("max_shard_bytes_exclusive")
        if (
            index.get("format_version") != "t051_tokenized_git_shards_v1"
            or index.get("status") != "complete"
            or index.get("canonical_storage") != "server_only"
            or index.get("reconstruction") != "byte_concatenation_in_index_order"
            or index.get("digest_computation_performed") is not False
            or type(max_bytes) is not int
            or max_bytes <= 0
            or index.get("record_count") != 1200
            or index.get("split_order") != list(SPLITS)
            or not isinstance(shards, list)
            or not shards
            or spec.get("record_count") != SPLIT_RECORD_COUNTS[split]
            or spec.get("shard_count") != len(shards)
            or spec.get("canonical_relative_path") != f"{split}.jsonl"
        ):
            raise ReleaseQAError(
                "RELEASE_TOKEN_SHARD_INDEX",
                "T051 token shard index is incomplete",
                evidence={"split": split},
            )
        values: list[dict[str, Any]] = []
        for expected_order, raw in enumerate(shards):
            if not isinstance(raw, Mapping) or raw.get("order") != expected_order:
                raise ReleaseQAError(
                    "RELEASE_TOKEN_SHARD_ORDER",
                    "T051 token shards are not in exact declared order",
                    evidence={"split": split, "expected_order": expected_order},
                )
            relative_path, path = _resolve_shard_path(root, index_path, raw.get("path"))
            if path in used_shard_paths:
                raise ReleaseQAError(
                    "RELEASE_TOKEN_SHARD_PATH_DUPLICATE",
                    "one token Git shard path is referenced more than once",
                    evidence={"path": relative_path},
                )
            used_shard_paths.add(path)
            rows = _read_jsonl(path, f"tokenized/{split} shard {expected_order}")
            if (
                path.stat().st_size != raw.get("bytes")
                or path.stat().st_size >= max_bytes
                or len(rows) != raw.get("row_count")
                or not rows
                or rows[0].get("record_id") != raw.get("first_record_id")
                or rows[-1].get("record_id") != raw.get("last_record_id")
            ):
                raise ReleaseQAError(
                    "RELEASE_TOKEN_SHARD_CONTENT",
                    "T051 token shard differs from its path/count/boundary declaration",
                    evidence={
                        "split": split,
                        "order": expected_order,
                        "path": str(path),
                    },
                )
            values.extend(rows)
            shard_inventory.append(
                {
                    "path": relative_path,
                    "artifact_family": "tokenized_git_shard",
                    "split": split,
                    "row_count": len(rows),
                    "file_bytes": path.stat().st_size,
                    "order": expected_order,
                }
            )
        shard_token_count = sum(
            len(row.get("input_ids", ()))
            for row in values
            if isinstance(row.get("input_ids"), list)
        )
        if (
            len(values) != SPLIT_RECORD_COUNTS[split]
            or len({str(row.get("record_id")) for row in values}) != len(values)
            or sum(item["file_bytes"] for item in shard_inventory)
            != spec.get("canonical_bytes")
            or shard_token_count != spec.get("token_count")
        ):
            raise ReleaseQAError(
                "RELEASE_TOKEN_SHARD_COVERAGE",
                "ordered T051 token shards do not form the exact split inventory",
                evidence={"split": split},
            )
        shard_rows = tuple(values)
        if canonical is not None and canonical_path.stat().st_size != spec.get(
            "canonical_bytes"
        ):
            raise ReleaseQAError(
                "RELEASE_TOKEN_CANONICAL_SHARD_MISMATCH",
                "canonical T051 bytes differ from the Git shard inventory",
                evidence={"split": split},
            )
    if canonical is None and shard_rows is None:
        raise ReleaseQAError(
            "RELEASE_TOKENIZED_MISSING",
            "neither canonical T051 tokenized data nor its Git shards are available",
            evidence={"split": split},
        )
    if canonical is not None and shard_rows is not None:
        canonical_ids = tuple(str(row.get("record_id")) for row in canonical)
        shard_ids = tuple(str(row.get("record_id")) for row in shard_rows)
        if canonical_ids != shard_ids or canonical != shard_rows:
            raise ReleaseQAError(
                "RELEASE_TOKEN_CANONICAL_SHARD_MISMATCH",
                "canonical T051 rows differ from their ordered Git shard representation",
                evidence={"split": split},
            )
    selected = canonical if canonical is not None else shard_rows
    if selected is None:
        raise AssertionError("tokenized split selection is unreachable")
    selected_source = (
        (
            {
                "path": (Path("tokenized/chemdfm_r") / f"{split}.jsonl").as_posix(),
                "artifact_family": "tokenized_canonical",
                "split": split,
                "row_count": len(selected),
                "file_bytes": canonical_path.stat().st_size,
            },
        )
        if canonical is not None
        else ()
    )
    return selected, (*selected_source, *shard_inventory)


def _require_mapping(value: Any, name: str, record_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseQAError(
            "RELEASE_MAPPING_REQUIRED",
            f"{name} must be a mapping",
            evidence={"record_id": record_id},
        )
    return value


def _require_sequence(value: Any, name: str, record_id: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise ReleaseQAError(
            "RELEASE_SEQUENCE_REQUIRED",
            f"{name} must be a sequence",
            evidence={"record_id": record_id},
        )
    return value


def _expected_serialized_text(detector: Mapping[str, Any]) -> str:
    parts = []
    for field, _kind, delimiter in _EXPECTED_SEGMENTS:
        value = detector.get(field)
        if type(value) is not str or not value:
            raise ReleaseQAError(
                "RELEASE_DETECTOR_FIELD_INVALID",
                "detector fields must be non-empty text",
                evidence={"field": field},
            )
        parts.append(f"{delimiter}\n{value}")
    return "\n\n".join(parts)


def _validate_detector_record(record: Mapping[str, Any]) -> None:
    record_id = str(record.get("record_id", ""))
    _assert_public_boundary(record)
    detector = _require_mapping(
        record.get("detector_input"), "detector_input", record_id
    )
    if set(detector) != _EXPECTED_DETECTOR_FIELDS:
        raise ReleaseQAError(
            "RELEASE_DETECTOR_FIELDS",
            "detector input must expose exactly four frozen fields",
            evidence={"record_id": record_id, "fields": tuple(sorted(detector))},
        )
    expected_text = _expected_serialized_text(detector)
    serialized = _require_mapping(record.get("serialized"), "serialized", record_id)
    actual_text = serialized.get("text")
    if type(actual_text) is not str or actual_text.encode(
        "utf-8"
    ) != expected_text.encode("utf-8"):
        raise ReleaseQAError(
            "RELEASE_DETECTOR_ORDER",
            "serialized detector bytes differ from the frozen field order",
            evidence={"record_id": record_id},
        )
    segments = _require_sequence(
        serialized.get("segments"), "serialized.segments", record_id
    )
    if len(segments) != len(_EXPECTED_SEGMENTS):
        raise ReleaseQAError(
            "RELEASE_DETECTOR_SEGMENTS",
            "serialized detector text requires exactly four segments",
            evidence={"record_id": record_id},
        )
    for segment, (field, kind, _delimiter) in zip(
        segments, _EXPECTED_SEGMENTS, strict=True
    ):
        item = _require_mapping(segment, "serialized segment", record_id)
        start = item.get("start")
        end = item.get("end")
        if (
            item.get("field_name") != field
            or item.get("segment_kind") != kind
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(expected_text)
            or expected_text[start:end] != detector[field]
        ):
            raise ReleaseQAError(
                "RELEASE_DETECTOR_SEGMENTS",
                "serialized detector segment disagrees with its visible field",
                evidence={"record_id": record_id, "field": field},
            )


def _validate_token_row(
    token: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any]]:
    record_id = str(record["record_id"])
    if token.get("activation_alignment") != ACTIVATION_ALIGNMENT:
        raise ReleaseQAError(
            "RELEASE_TOKEN_ALIGNMENT",
            "token artifact is not aligned to post-token h_t",
            evidence={"record_id": record_id},
        )
    input_ids = _require_sequence(token.get("input_ids"), "input_ids", record_id)
    token_count = len(input_ids)
    if not token_count or any(
        type(value) is not int or value < 0 for value in input_ids
    ):
        raise ReleaseQAError(
            "RELEASE_TOKEN_IDS",
            "token IDs must be a non-empty non-negative integer sequence",
            evidence={"record_id": record_id},
        )
    for name in _DIRECT_TOKEN_ARRAYS:
        values = _require_sequence(token.get(name), name, record_id)
        if len(values) != token_count:
            raise ReleaseQAError(
                "RELEASE_TOKEN_LENGTH",
                "all token arrays must have exactly the input token length",
                evidence={"record_id": record_id, "field": name},
            )
        if name in _BINARY_TOKEN_ARRAYS and any(
            type(value) is not int or value not in {0, 1} for value in values
        ):
            raise ReleaseQAError(
                "RELEASE_TOKEN_MASK",
                "binary token mask contains a non-binary value",
                evidence={"record_id": record_id, "field": name},
            )
    offsets = token["offset_mapping"]
    if any(
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or any(type(value) is not int for value in pair)
        for pair in offsets
    ):
        raise ReleaseQAError(
            "RELEASE_TOKEN_OFFSETS",
            "token offsets must contain exact integer pairs",
            evidence={"record_id": record_id},
        )
    positive_total = sum(token["error_any_mask"])
    nested_positive = 0
    for field in _NESTED_TOKEN_ARRAYS:
        axis = _require_mapping(token.get(field), field, record_id)
        if not axis:
            raise ReleaseQAError(
                "RELEASE_TOKEN_AXIS_EMPTY",
                "multi-label token axes cannot be empty",
                evidence={"record_id": record_id, "field": field},
            )
        for label, mask in axis.items():
            values = _require_sequence(mask, f"{field}[{label}]", record_id)
            if len(values) != token_count or any(
                type(value) is not int or value not in {0, 1} for value in values
            ):
                raise ReleaseQAError(
                    "RELEASE_TOKEN_LENGTH",
                    "nested token masks must be equal-length binary arrays",
                    evidence={"record_id": record_id, "field": f"{field}[{label}]"},
                )
            nested_positive += sum(values)
    label = record["variant"]["label"]
    if label == "H" and (positive_total <= 0 or nested_positive <= 0):
        raise ReleaseQAError(
            "RELEASE_H_TOKEN_SPAN",
            "hallucinated record lacks a positive token label",
            evidence={"record_id": record_id},
        )
    if label == "N" and (
        positive_total
        or nested_positive
        or sum(token["hallucination_core_mask"])
        or sum(token["local_falsehood_mask"])
        or sum(token["off_task_branch_mask"])
    ):
        raise ReleaseQAError(
            "RELEASE_N_TOKEN_MASK",
            "faithful control contains a positive error token label",
            evidence={"record_id": record_id},
        )
    fingerprint = _require_mapping(
        token.get("tokenizer_fingerprint"), "tokenizer_fingerprint", record_id
    )
    normalization = _require_mapping(
        fingerprint.get("normalization_config"),
        "tokenizer_fingerprint.normalization_config",
        record_id,
    )
    if (
        normalization.get("production_weights_loaded") is not True
        or normalization.get("fast_tokenizer") is not True
        or normalization.get("digest_computation_performed") is not False
    ):
        raise ReleaseQAError(
            "RELEASE_REAL_TOKENIZER",
            "token fingerprint does not identify the real local fast tokenizer",
            evidence={"record_id": record_id},
        )
    return token_count, fingerprint


def _read_split_manifest(root: Path) -> dict[str, dict[str, str]]:
    path = root / "split_manifest.csv"
    if not path.is_file():
        raise ReleaseQAError(
            "RELEASE_SPLIT_MANIFEST_MISSING",
            "frozen split manifest is missing",
            evidence={"path": str(path)},
        )
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReleaseQAError(
            "RELEASE_SPLIT_MANIFEST_INVALID",
            "split manifest cannot be parsed",
        ) from error
    if len(rows) != 150:
        raise ReleaseQAError(
            "RELEASE_SPLIT_MANIFEST_COUNT",
            "split manifest must contain exactly 150 origins",
            evidence={"observed": len(rows)},
        )
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        origin_id = row.get("anonymous_sample_id")
        split = row.get("split")
        leakage = row.get("leakage_group_id")
        if (
            type(origin_id) is not str
            or not origin_id
            or origin_id in indexed
            or split not in SPLITS
            or type(leakage) is not str
            or not leakage
            or row.get("dataset_version") != DATASET_VERSION
        ):
            raise ReleaseQAError(
                "RELEASE_SPLIT_MANIFEST_ROW",
                "split manifest contains an invalid or duplicate origin row",
                evidence={"origin_id": origin_id},
            )
        indexed[origin_id] = dict(row)
    if Counter(row["split"] for row in rows) != Counter(SPLIT_ORIGIN_COUNTS):
        raise ReleaseQAError(
            "RELEASE_SPLIT_ORIGIN_COUNTS",
            "split manifest origin counts differ from 100/25/25",
        )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["leakage_group_id"]].add(row["split"])
    overlap = tuple(
        sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    )
    if overlap:
        raise ReleaseQAError(
            "RELEASE_LEAKAGE_GROUP_OVERLAP",
            "a leakage group appears in more than one split",
            evidence={"overlap_count": len(overlap)},
        )
    return indexed


def _validate_build_reports(root: Path, inventory: _ReleaseInventory) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        build = _read_json(
            root / f"reports/{split}_build_report.json", f"{split} build report"
        )
        validation = _read_json(
            root / f"reports/{split}_validation_report.json",
            f"{split} strict validation report",
        )
        expected_records = SPLIT_RECORD_COUNTS[split]
        expected_origins = SPLIT_ORIGIN_COUNTS[split]
        build_id_key, build_id = _BUILD_IDENTITIES[split]
        if (
            build.get("all_pass") is not True
            or build.get("format_version") != _BUILD_REPORT_FORMATS[split]
            or build.get(build_id_key) != build_id
            or build.get("summary", {}).get("record_count") != expected_records
            or build.get("summary", {}).get("origin_count") != expected_origins
            or validation.get("all_pass") is not True
            or validation.get("format_version") != _STRICT_REPORT_FORMATS[split]
            or validation.get(build_id_key) != build_id
            or validation.get("required_validator_ids")
            != [*_ARTIFACT_VALIDATOR_IDS, _BUNDLE_VALIDATOR_ID]
            or validation.get("artifact_gate_count") != expected_records * 4
            or validation.get("bundle_gate_count") != expected_origins
        ):
            raise ReleaseQAError(
                "RELEASE_STRICT_REPORT",
                "split build or strict validation report is incomplete",
                evidence={"split": split},
            )

        expected_rows = {
            str(row["record_id"]): row for row in inventory.rows["records"][split]
        }
        raw_record_evidence = validation.get("records")
        if not isinstance(raw_record_evidence, list):
            raise ReleaseQAError(
                "RELEASE_STRICT_REPORT",
                "strict validation report lacks per-record T043 evidence",
                evidence={"split": split},
            )
        reported: dict[str, Mapping[str, Any]] = {}
        for evidence in raw_record_evidence:
            if not isinstance(evidence, Mapping):
                raise ReleaseQAError(
                    "RELEASE_STRICT_REPORT",
                    "strict validation record evidence must be a mapping",
                    evidence={"split": split},
                )
            record_id = evidence.get("record_id")
            if (
                type(record_id) is not str
                or record_id not in expected_rows
                or record_id in reported
            ):
                raise ReleaseQAError(
                    "RELEASE_STRICT_REPORT",
                    "strict validation report has an unknown or duplicate record",
                    evidence={"split": split, "record_id": record_id},
                )
            record = expected_rows[record_id]
            for key in (
                "record_id",
                "origin_id",
                "pair_id",
                "bundle_id",
                "leakage_group_id",
                "split",
                "dataset_version",
            ):
                if evidence.get(key) != record.get(key):
                    raise ReleaseQAError(
                        "RELEASE_STRICT_REPORT",
                        "strict validation record identity differs from the release",
                        evidence={"split": split, "record_id": record_id, "field": key},
                    )
            gates = evidence.get("artifact_gates")
            if not isinstance(gates, list) or len(gates) != len(
                _ARTIFACT_VALIDATOR_IDS
            ):
                raise ReleaseQAError(
                    "RELEASE_STRICT_REPORT",
                    "strict validation record lacks four T043 artifact gates",
                    evidence={"split": split, "record_id": record_id},
                )
            for gate, validator_id in zip(gates, _ARTIFACT_VALIDATOR_IDS, strict=True):
                if (
                    not isinstance(gate, Mapping)
                    or gate.get("validator_id") != validator_id
                    or gate.get("all_pass") is not True
                    or gate.get("issues") != []
                ):
                    raise ReleaseQAError(
                        "RELEASE_STRICT_REPORT",
                        "one T043 artifact gate is missing, reordered, or failed",
                        evidence={
                            "split": split,
                            "record_id": record_id,
                            "validator_id": validator_id,
                        },
                    )
            chain = evidence.get("artifact_chain")
            if (
                not isinstance(chain, Mapping)
                or chain.get("validator_id") != _ARTIFACT_CHAIN_ID
                or chain.get("all_pass") is not True
                or chain.get("issues") != []
                or evidence.get("bundle_validator_id") != _BUNDLE_VALIDATOR_ID
                or evidence.get("bundle_all_pass") is not True
                or evidence.get(build_id_key) != build_id
            ):
                raise ReleaseQAError(
                    "RELEASE_STRICT_REPORT",
                    "T043 chain or bundle evidence is incomplete",
                    evidence={"split": split, "record_id": record_id},
                )
            reported[record_id] = evidence
        if set(reported) != set(expected_rows) or len(reported) != expected_records:
            raise ReleaseQAError(
                "RELEASE_STRICT_REPORT",
                "T043 per-record evidence does not cover the exact release split",
                evidence={"split": split, "reported": len(reported)},
            )

        raw_origin_evidence = validation.get("origins")
        if not isinstance(raw_origin_evidence, list):
            raise ReleaseQAError(
                "RELEASE_STRICT_REPORT",
                "strict validation report lacks per-origin T043 evidence",
                evidence={"split": split},
            )
        expected_origin_ids = set(inventory.origins_by_split[split])
        reported_origins: set[str] = set()
        for evidence in raw_origin_evidence:
            origin_id = (
                evidence.get("origin_id") if isinstance(evidence, Mapping) else None
            )
            if (
                not isinstance(evidence, Mapping)
                or type(origin_id) is not str
                or origin_id not in expected_origin_ids
                or origin_id in reported_origins
                or evidence.get("record_count") != 8
                or evidence.get("all_pass") is not True
                or evidence.get("issue_codes") != []
            ):
                raise ReleaseQAError(
                    "RELEASE_STRICT_REPORT",
                    "T043 per-origin bundle evidence is incomplete",
                    evidence={"split": split, "origin_id": origin_id},
                )
            reported_origins.add(origin_id)
        if reported_origins != expected_origin_ids:
            raise ReleaseQAError(
                "RELEASE_STRICT_REPORT",
                "T043 origin evidence does not cover the exact split",
                evidence={"split": split, "reported": len(reported_origins)},
            )
        result[split] = {
            "artifact_gate_count": expected_records * 4,
            "bundle_gate_count": expected_origins,
        }
    return result


def _validate_test_isolation(root: Path) -> dict[str, Any]:
    declaration = _read_json(
        root / "reports/test_isolation_declaration.json",
        "test isolation declaration",
    )
    usage = declaration.get("test_usage")
    frozen = declaration.get("frozen_design")
    build_order = declaration.get("build_order")
    detector_scope = declaration.get("detector_scope")
    failure_semantics = declaration.get("failure_semantics")
    if (
        declaration.get("format_version") != "t050_test_isolation_declaration_v1"
        or declaration.get("all_pass") is not True
        or not isinstance(usage, Mapping)
        or not isinstance(frozen, Mapping)
        or not isinstance(build_order, Mapping)
        or not isinstance(detector_scope, Mapping)
        or not isinstance(failure_semantics, Mapping)
    ):
        raise ReleaseQAError(
            "RELEASE_TEST_ISOLATION",
            "test isolation declaration has an incomplete frozen schema",
        )
    if set(usage) != set(_REQUIRED_TEST_USAGE) or any(
        usage.get(key) is not expected for key, expected in _REQUIRED_TEST_USAGE.items()
    ):
        raise ReleaseQAError(
            "RELEASE_TEST_ISOLATION",
            "test usage must enumerate every frozen selection boundary exactly",
            evidence={"usage_keys": tuple(sorted(usage))},
        )
    if any(
        frozen.get(key) is not expected
        for key, expected in _REQUIRED_FROZEN_DESIGN.items()
    ):
        raise ReleaseQAError(
            "RELEASE_TEST_ISOLATION",
            "candidate/operator/recipe/propagation/renderer thresholds were not frozen",
        )
    if (
        build_order.get("required_predecessors_checked_before_test_construction")
        != ["T047", "T048", "T049"]
        or build_order.get("test_built_last") is not True
        or build_order.get("train_complete_before_test") is not True
        or build_order.get("validation_complete_before_test") is not True
        or detector_scope.get("detector_layer_selected") is not False
        or detector_scope.get("detector_threshold_selected") is not False
        or detector_scope.get("formal_detector_training_authorized") is not False
        or failure_semantics.get("cross_split_backfill_allowed") is not False
        or failure_semantics.get("failed_attempt_emitted_record_count") != 0
    ):
        raise ReleaseQAError(
            "RELEASE_TEST_ISOLATION",
            "test build order, detector scope, or atomic failure evidence is incomplete",
        )
    return declaration


@dataclass(frozen=True, slots=True)
class _ReleaseInventory:
    rows: Mapping[str, Mapping[str, tuple[dict[str, Any], ...]]]
    records_by_id: Mapping[str, Mapping[str, Mapping[str, Any]]]
    token_counts: Mapping[str, int]
    tokenizer_fingerprint: Mapping[str, Any]
    origins_by_split: Mapping[str, tuple[str, ...]]
    artifact_inventory: tuple[dict[str, Any], ...]
    provenance_cache_entry_count: int
    network_request_count: int


def _load_and_validate_release(root: Path) -> _ReleaseInventory:
    manifest_rows = _read_split_manifest(root)
    family_rows: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {
        family: {} for family in FAMILIES
    }
    family_indexes: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {
        family: {} for family in FAMILIES
    }
    artifact_inventory: list[dict[str, Any]] = []
    global_record_ids: set[str] = set()
    token_counts: dict[str, int] = {}
    tokenizer_fingerprint: Mapping[str, Any] | None = None
    cache_entry_count = 0
    network_request_count = 0
    origins_by_split: dict[str, tuple[str, ...]] = {}
    used_token_shard_paths: set[Path] = set()

    for split in SPLITS:
        expected_count = SPLIT_RECORD_COUNTS[split]
        for family, directory in FAMILIES.items():
            path = root / directory / f"{split}.jsonl"
            token_inventory: tuple[dict[str, Any], ...] = ()
            if family == "tokenized":
                rows, token_inventory = _read_tokenized_split(
                    root, split, used_token_shard_paths
                )
            else:
                rows = _read_jsonl(path, f"{family}/{split}")
            if len(rows) != expected_count:
                raise ReleaseQAError(
                    "RELEASE_FAMILY_COUNT",
                    "artifact family has the wrong split row count",
                    evidence={
                        "family": family,
                        "split": split,
                        "expected": expected_count,
                        "actual": len(rows),
                    },
                )
            _assert_secret_free(rows, f"{directory}/{split}.jsonl")
            family_rows[family][split] = rows
            family_indexes[family][split] = _indexed(rows, family=family, split=split)
            if family == "tokenized":
                artifact_inventory.extend(token_inventory)
            else:
                artifact_inventory.append(
                    {
                        "path": f"{directory}/{split}.jsonl",
                        "artifact_family": family,
                        "split": split,
                        "row_count": len(rows),
                    }
                )
        id_sets = {family: set(family_indexes[family][split]) for family in FAMILIES}
        if len({frozenset(values) for values in id_sets.values()}) != 1:
            raise ReleaseQAError(
                "RELEASE_ARTIFACT_IDENTITY",
                "five release artifact families have different record identities",
                evidence={
                    "split": split,
                    "counts": {key: len(value) for key, value in id_sets.items()},
                },
            )
        split_record_ids = id_sets["records"]
        if global_record_ids & split_record_ids:
            raise ReleaseQAError(
                "RELEASE_CROSS_SPLIT_RECORD_ID",
                "record identity occurs in more than one split",
                evidence={"split": split},
            )
        global_record_ids.update(split_record_ids)

        records = family_indexes["records"][split]
        origins = tuple(sorted({str(row["origin_id"]) for row in records.values()}))
        origins_by_split[split] = origins
        if len(origins) != SPLIT_ORIGIN_COUNTS[split]:
            raise ReleaseQAError(
                "RELEASE_ORIGIN_COUNT",
                "split has the wrong number of origins",
                evidence={"split": split, "observed": len(origins)},
            )
        by_origin: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record_id, record in records.items():
            if (
                record.get("split") != split
                or record.get("dataset_version") != DATASET_VERSION
            ):
                raise ReleaseQAError(
                    "RELEASE_RECORD_BINDING",
                    "detector record has an invalid split or dataset version",
                    evidence={"record_id": record_id},
                )
            origin_id = record.get("origin_id")
            row = manifest_rows.get(str(origin_id))
            if (
                row is None
                or row["split"] != split
                or row["leakage_group_id"] != record.get("leakage_group_id")
            ):
                raise ReleaseQAError(
                    "RELEASE_MANIFEST_BINDING",
                    "record split/leakage identity differs from frozen manifest",
                    evidence={"record_id": record_id},
                )
            _validate_detector_record(record)
            variant = _require_mapping(record.get("variant"), "variant", record_id)
            if (
                variant.get("label") not in {"H", "N"}
                or variant.get("propagation") not in POLICIES
            ):
                raise ReleaseQAError(
                    "RELEASE_VARIANT_INVALID",
                    "record variant label or propagation policy is invalid",
                    evidence={"record_id": record_id},
                )
            spans = _require_sequence(record.get("spans"), "spans", record_id)
            if variant["label"] == "H" and not spans:
                raise ReleaseQAError(
                    "RELEASE_H_CHAR_SPAN",
                    "hallucinated record lacks a positive char span",
                    evidence={"record_id": record_id},
                )
            if variant["label"] == "N" and spans:
                raise ReleaseQAError(
                    "RELEASE_N_CHAR_SPAN",
                    "faithful record has a positive char span",
                    evidence={"record_id": record_id},
                )
            for span in spans:
                annotation = _require_mapping(span, "span", record_id)
                if (
                    not _require_sequence(
                        annotation.get("semantic_types"), "semantic_types", record_id
                    )
                    or not _require_sequence(
                        annotation.get("edit_subtypes"), "edit_subtypes", record_id
                    )
                    or annotation.get("causal_role")
                    not in {
                        "ROOT",
                        "PROPAGATED_FALSE",
                        "PROPAGATED_CONDITIONAL",
                        "TERMINAL",
                    }
                ):
                    raise ReleaseQAError(
                        "RELEASE_CHAR_LABELS",
                        "char span lacks an orthogonal label axis",
                        evidence={"record_id": record_id},
                    )
            verification = _require_mapping(
                record.get("verification"), "verification", record_id
            )
            if set(verification) != _REQUIRED_RECORD_VERIFICATION or any(
                verification.get(key) is not True
                for key in _REQUIRED_RECORD_VERIFICATION
            ):
                raise ReleaseQAError(
                    "RELEASE_STRICT_RECORD",
                    "record lacks the exact frozen deterministic verification evidence",
                    evidence={
                        "record_id": record_id,
                        "verification_keys": tuple(sorted(verification)),
                    },
                )
            by_origin[str(origin_id)].append(record)
            by_pair[str(record["pair_id"])].append(record)

            identity = {
                key: record.get(key)
                for key in (
                    "record_id",
                    "origin_id",
                    "pair_id",
                    "bundle_id",
                    "leakage_group_id",
                    "split",
                    "dataset_version",
                )
            }
            for family in ("oracle", "state_graphs", "tokenized", "provenance"):
                other = family_indexes[family][split][record_id]
                if any(other.get(key) != value for key, value in identity.items()):
                    raise ReleaseQAError(
                        "RELEASE_ARTIFACT_IDENTITY",
                        "artifact family metadata differs for one record",
                        evidence={"record_id": record_id, "family": family},
                    )
            oracle = family_indexes["oracle"][split][record_id]
            if (
                oracle.get("visible_to_detector") is not False
                or type(oracle.get("gt_smiles")) is not str
            ):
                raise ReleaseQAError(
                    "RELEASE_ORACLE_BOUNDARY",
                    "oracle row is not explicitly hidden and complete",
                    evidence={"record_id": record_id},
                )
            state = family_indexes["state_graphs"][split][record_id]
            if state.get("artifact_scope") != "build_only_non_detector":
                raise ReleaseQAError(
                    "RELEASE_STATE_BOUNDARY",
                    "state graph row is not marked hidden",
                    evidence={"record_id": record_id},
                )
            provenance = family_indexes["provenance"][split][record_id]
            if provenance.get("artifact_scope") != "private_build_provenance":
                raise ReleaseQAError(
                    "RELEASE_PROVENANCE_BOUNDARY",
                    "provenance row is not marked private",
                    evidence={"record_id": record_id},
                )
            donor = _require_mapping(provenance.get("donor"), "donor", record_id)
            donor_origin_id = donor.get("donor_origin_id")
            if (
                donor.get("recipient_split") != split
                or donor.get("pool_split") != split
                or donor.get("verified_split_local") is not True
                or (
                    donor_origin_id is not None
                    and (
                        donor_origin_id not in manifest_rows
                        or manifest_rows[donor_origin_id]["split"] != split
                    )
                )
            ):
                raise ReleaseQAError(
                    "RELEASE_DONOR_LEAKAGE",
                    "provenance contains a cross-split or unverified donor edge",
                    evidence={"record_id": record_id},
                )
            execution = _require_mapping(
                provenance.get("execution_mode"), "execution_mode", record_id
            )
            requests = execution.get("network_request_count")
            if (
                type(requests) is not int
                or requests != 0
                or execution.get("live_poe_attempted") is not False
                or execution.get("live_availability_probe_performed") is not False
            ):
                raise ReleaseQAError(
                    "RELEASE_POE_NETWORK_CALL",
                    "release provenance contains a live Poe request",
                    evidence={"record_id": record_id},
                )
            model_values = (
                execution.get("requested_model_id"),
                execution.get("response_model"),
            )
            if any(value not in {None, POE_MODEL_ID} for value in model_values):
                raise ReleaseQAError(
                    "RELEASE_POE_MODEL",
                    "release provenance contains an undeclared Poe model",
                    evidence={"record_id": record_id},
                )
            cache_keys = execution.get("cache_keys")
            if not isinstance(cache_keys, list):
                raise ReleaseQAError(
                    "RELEASE_POE_CACHE_LEDGER",
                    "Poe cache key ledger must be an explicit list",
                    evidence={"record_id": record_id},
                )
            cache_entry_count += len(cache_keys)
            network_request_count += requests
            token = family_indexes["tokenized"][split][record_id]
            count, fingerprint = _validate_token_row(token, record)
            token_counts[record_id] = count
            if tokenizer_fingerprint is None:
                tokenizer_fingerprint = dict(fingerprint)
            elif tokenizer_fingerprint != fingerprint:
                raise ReleaseQAError(
                    "RELEASE_TOKENIZER_SET",
                    "release token rows do not share one exact real tokenizer fingerprint",
                )

        for origin_id, values in by_origin.items():
            counts = Counter(
                (row["variant"]["propagation"], row["variant"]["label"])
                for row in values
            )
            if len(values) != 8 or counts != Counter(
                {(policy, label): 1 for policy in POLICIES for label in ("H", "N")}
            ):
                raise ReleaseQAError(
                    "RELEASE_ORIGIN_BUNDLE",
                    "origin does not contain four complete reciprocal H/N pairs",
                    evidence={"origin_id": origin_id},
                )
        if len(by_pair) != expected_count // 2:
            raise ReleaseQAError(
                "RELEASE_PAIR_COUNT",
                "split has the wrong reciprocal-pair count",
                evidence={"split": split},
            )
        for pair_id, values in by_pair.items():
            if len(values) != 2 or {row["variant"]["label"] for row in values} != {
                "H",
                "N",
            }:
                raise ReleaseQAError(
                    "RELEASE_PAIR_INTEGRITY",
                    "pair is not one H and one N record",
                    evidence={"pair_id": pair_id},
                )
            left, right = values
            if (
                left["variant"].get("matched_record_id") != right["record_id"]
                or right["variant"].get("matched_record_id") != left["record_id"]
            ):
                raise ReleaseQAError(
                    "RELEASE_PAIR_INTEGRITY",
                    "reciprocal matched record identities disagree",
                    evidence={"pair_id": pair_id},
                )

    manifest_origin_ids = set(manifest_rows)
    release_origin_ids = {
        origin for values in origins_by_split.values() for origin in values
    }
    if manifest_origin_ids != release_origin_ids or len(release_origin_ids) != 150:
        raise ReleaseQAError(
            "RELEASE_MANIFEST_COVERAGE",
            "release origins differ from the frozen split manifest",
        )
    if len(global_record_ids) != 1200 or tokenizer_fingerprint is None:
        raise ReleaseQAError(
            "RELEASE_GLOBAL_COUNT",
            "release must contain exactly 1,200 unique records",
        )
    return _ReleaseInventory(
        rows=MappingProxyType(
            {
                family: MappingProxyType(dict(values))
                for family, values in family_rows.items()
            }
        ),
        records_by_id=MappingProxyType(
            {
                family: MappingProxyType(
                    {
                        record_id: row
                        for split_values in values.values()
                        for record_id, row in split_values.items()
                    }
                )
                for family, values in family_indexes.items()
            }
        ),
        token_counts=MappingProxyType(token_counts),
        tokenizer_fingerprint=MappingProxyType(dict(tokenizer_fingerprint)),
        origins_by_split=MappingProxyType(origins_by_split),
        artifact_inventory=tuple(artifact_inventory),
        provenance_cache_entry_count=cache_entry_count,
        network_request_count=network_request_count,
    )


def _validate_token_manifest(
    root: Path, inventory: _ReleaseInventory
) -> dict[str, Any]:
    manifest = _read_json(
        root / "tokenized/chemdfm_r/manifest.json", "real tokenizer manifest"
    )
    splits = manifest.get("splits")
    split_counts = (
        {str(item.get("split")): item.get("record_count") for item in splits}
        if isinstance(splits, list)
        and all(isinstance(item, Mapping) for item in splits)
        else {}
    )
    git_shards = manifest.get("git_shards")
    if (
        manifest.get("status") != "complete"
        or manifest.get("mode") != "release"
        or manifest.get("record_count") != 1200
        or manifest.get("activation_alignment") != ACTIVATION_ALIGNMENT
        or manifest.get("label_shift") != 0
        or manifest.get("all_token_arrays_equal_length") is not True
        or split_counts != dict(SPLIT_RECORD_COUNTS)
        or manifest.get("tokenizer_fingerprint") != inventory.tokenizer_fingerprint
        or not isinstance(git_shards, Mapping)
        or git_shards.get("index_path") != "git_shards/index.json"
        or git_shards.get("canonical_storage") != "server_only"
        or git_shards.get("digest_computation_performed") is not False
    ):
        raise ReleaseQAError(
            "RELEASE_TOKEN_MANIFEST",
            "real tokenizer manifest does not bind the complete release",
        )
    return manifest


def _resolve_activation_shard_path(
    path_value: Any,
    activation_root: Path,
    label: str,
) -> tuple[str, Path]:
    if type(path_value) is not str or not path_value:
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_PATH",
            f"activation {label} path is invalid",
        )
    value = Path(path_value)
    if value.is_absolute() or ".." in value.parts:
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_PATH",
            f"activation {label} path must be relative to layer_26 root",
            evidence={"path": path_value},
        )
    root = activation_root.resolve(strict=False)
    resolved = (root / value).resolve(strict=False)
    if not _is_within(resolved, root):
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_PATH",
            f"activation {label} path escapes layer_26 root",
            evidence={"path": path_value},
        )
    return value.as_posix(), resolved


def _validate_activation_manifest(
    root: Path,
    inventory: _ReleaseInventory,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    activation_root = root / "activations/chemdfm_r/layer_26"
    manifest = _read_json(activation_root / "manifest.json", "activation manifest")
    alignment = manifest.get("alignment")
    feature = manifest.get("feature")
    model = manifest.get("model")
    split_summary = manifest.get("splits")
    tensor_audit = manifest.get("tensor_payload_validation")
    if (
        manifest.get("format_version") != "t051_activation_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("mode") != "release"
        or manifest.get("record_count") != 1200
        or manifest.get("token_count") != EXPECTED_ACTIVATION_TOKEN_COUNT
        or manifest.get("shard_count") != EXPECTED_ACTIVATION_SHARD_COUNT
        or not isinstance(alignment, Mapping)
        or alignment.get("activation_alignment") != ACTIVATION_ALIGNMENT
        or alignment.get("label_shift") != 0
        or alignment.get("hidden_token_axis_equals_label_length") is not True
        or alignment.get("pre_token_claimed") is not False
        or not isinstance(feature, Mapping)
        or feature.get("name") != "resid_post"
        or feature.get("layer_index") != 26
        or feature.get("selected_using_pilot_records") is not False
        or not isinstance(model, Mapping)
        or model.get("hidden_size") != 5120
        or model.get("digest_computation_performed") is not False
        or manifest.get("tokenizer_fingerprint") != inventory.tokenizer_fingerprint
        or not isinstance(split_summary, Mapping)
        or set(split_summary) != set(SPLITS)
        or {split: split_summary.get(split, {}).get("record_count") for split in SPLITS}
        != dict(SPLIT_RECORD_COUNTS)
        or not isinstance(tensor_audit, Mapping)
        or tensor_audit.get("performed") is not True
        or tensor_audit.get("all_pass") is not True
        or tensor_audit.get("shard_count") != EXPECTED_ACTIVATION_SHARD_COUNT
        or tensor_audit.get("record_count") != 1200
        or tensor_audit.get("token_count") != EXPECTED_ACTIVATION_TOKEN_COUNT
        or tensor_audit.get("activation_dtype") != "bfloat16"
        or tensor_audit.get("ordered_record_ids_exact") is not True
        or tensor_audit.get("token_counts_exact") is not True
        or tensor_audit.get("row_offsets_exact") is not True
        or tensor_audit.get("activation_shape_exact") is not True
        or tensor_audit.get("activation_alignment_exact") is not True
        or tensor_audit.get("layer_index_exact") is not True
        or tensor_audit.get("digest_verification_performed") is not False
    ):
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_MANIFEST",
            "activation manifest does not bind exact post-token layer-26 features",
        )
    raw_shards = manifest.get("shards")
    if (
        not isinstance(raw_shards, list)
        or len(raw_shards) != EXPECTED_ACTIVATION_SHARD_COUNT
    ):
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_SHARDS",
            "activation manifest must enumerate non-empty shards",
        )
    seen_record_ids: set[str] = set()
    seen_tensor_paths: set[Path] = set()
    seen_metadata_paths: set[Path] = set()
    sidecar_inventory: list[dict[str, Any]] = []
    total_tokens = 0
    expected_axes = {
        split: tuple(
            (str(row["record_id"]), inventory.token_counts[str(row["record_id"])])
            for row in inventory.rows["tokenized"][split]
        )
        for split in SPLITS
    }
    expected_ids_by_split = {
        split: {record_id for record_id, _token_count in axis}
        for split, axis in expected_axes.items()
    }
    declared_shard_counts: dict[str, int] = {}
    for split in SPLITS:
        summary = split_summary.get(split)
        count = summary.get("shard_count") if isinstance(summary, Mapping) else None
        if type(count) is not int or count <= 0:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_MANIFEST",
                "activation split summary has an invalid shard count",
                evidence={"split": split},
            )
        declared_shard_counts[split] = count
    expected_shard_order = tuple(
        (split, shard_index)
        for split in SPLITS
        for shard_index in range(declared_shard_counts[split])
    )
    if len(expected_shard_order) != EXPECTED_ACTIVATION_SHARD_COUNT:
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_MANIFEST",
            "activation per-split shard counts differ from the frozen total",
        )
    activation_axes: dict[str, list[tuple[str, int]]] = {split: [] for split in SPLITS}
    split_actual = {
        split: {"record_count": 0, "token_count": 0, "shard_count": 0}
        for split in SPLITS
    }
    for position, shard in enumerate(raw_shards):
        if not isinstance(shard, Mapping):
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_SHARDS",
                "activation shard summary must be a mapping",
            )
        split = shard.get("split")
        shard_index = shard.get("shard_index")
        if (split, shard_index) != expected_shard_order[position]:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_SHARD_ORDER",
                "activation shards must be contiguous in frozen split order",
                evidence={
                    "position": position,
                    "expected": expected_shard_order[position],
                    "actual": (split, shard_index),
                },
            )
        tensor_relative, tensor_path = _resolve_activation_shard_path(
            shard.get("tensor_path"), activation_root, "tensor"
        )
        metadata_relative, metadata_path = _resolve_activation_shard_path(
            shard.get("metadata_path"), activation_root, "metadata"
        )
        if tensor_path in seen_tensor_paths or metadata_path in seen_metadata_paths:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_PATH_DUPLICATE",
                "activation tensor and sidecar paths must be unique",
                evidence={
                    "tensor_path": tensor_relative,
                    "metadata_path": metadata_relative,
                },
            )
        seen_tensor_paths.add(tensor_path)
        seen_metadata_paths.add(metadata_path)
        sidecar = _read_json(metadata_path, "activation sidecar")
        record_ids = sidecar.get("record_ids")
        token_counts = sidecar.get("token_counts")
        row_offsets = sidecar.get("row_offsets")
        if (
            sidecar.get("format_version") != "t051_chemdfm_r_post_token_v1"
            or sidecar.get("status") != "complete"
            or sidecar.get("split") != split
            or sidecar.get("shard_index") != shard_index
            or sidecar.get("activation_alignment") != ACTIVATION_ALIGNMENT
            or sidecar.get("label_shift") != 0
            or sidecar.get("layer_index") != 26
            or sidecar.get("feature_name") != "resid_post"
            or sidecar.get("activation_dtype") != "bfloat16"
            or sidecar.get("tensor_relative_path") != tensor_relative
            or sidecar.get("digest_computation_performed") is not False
            or not isinstance(record_ids, list)
            or not isinstance(token_counts, list)
            or not isinstance(row_offsets, list)
            or len(record_ids) != len(token_counts)
            or len(row_offsets) != len(record_ids) + 1
            or row_offsets[0] != 0
        ):
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_SIDECAR",
                "activation sidecar has an invalid alignment or identity contract",
                evidence={"path": str(metadata_path)},
            )
        expected_counts = []
        for record_id in record_ids:
            if (
                type(record_id) is not str
                or record_id not in expected_ids_by_split[split]
            ):
                raise ReleaseQAError(
                    "RELEASE_ACTIVATION_SPLIT_AXIS",
                    "activation shard contains an identity from the wrong split",
                    evidence={"split": split, "record_id": record_id},
                )
            if record_id in seen_record_ids:
                raise ReleaseQAError(
                    "RELEASE_ACTIVATION_DUPLICATE",
                    "activation record occurs in more than one shard",
                    evidence={"record_id": record_id},
                )
            seen_record_ids.add(record_id)
            expected_counts.append(inventory.token_counts[record_id])
        if token_counts != expected_counts:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_TOKEN_AXIS",
                "activation token counts differ from token label lengths",
                evidence={"path": str(metadata_path)},
            )
        expected_offsets = [0]
        for token_count in expected_counts:
            expected_offsets.append(expected_offsets[-1] + token_count)
        if row_offsets != expected_offsets or sidecar.get("activation_shape") != [
            expected_offsets[-1],
            5120,
        ]:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_TOKEN_AXIS",
                "activation row offsets or shape differ from token label lengths",
                evidence={"path": str(metadata_path)},
            )
        if (
            not tensor_path.is_file()
            or tensor_path.stat().st_size <= 0
            or sidecar.get("file_bytes") != tensor_path.stat().st_size
            or shard.get("file_bytes") != tensor_path.stat().st_size
            or shard.get("record_count") != len(record_ids)
            or shard.get("token_count") != expected_offsets[-1]
            or shard.get("hidden_size") != 5120
            or shard.get("layer_index") != 26
        ):
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_FILE",
                "activation tensor path/size differs from its sidecar and manifest",
                evidence={"path": str(tensor_path)},
            )
        activation_axes[split].extend(zip(record_ids, expected_counts, strict=True))
        split_actual[split]["record_count"] += len(record_ids)
        split_actual[split]["token_count"] += expected_offsets[-1]
        split_actual[split]["shard_count"] += 1
        total_tokens += expected_offsets[-1]
        sidecar_inventory.extend(
            (
                {
                    "path": tensor_relative,
                    "artifact_family": "activation_tensor",
                    "split": sidecar.get("split"),
                    "record_count": len(record_ids),
                    "token_count": expected_offsets[-1],
                    "file_bytes": tensor_path.stat().st_size,
                },
                {
                    "path": metadata_relative,
                    "artifact_family": "activation_sidecar",
                    "split": sidecar.get("split"),
                    "record_count": len(record_ids),
                },
            )
        )
    for split in SPLITS:
        if tuple(activation_axes[split]) != expected_axes[split]:
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_ORDER",
                "activation sidecars differ from the ordered token axis",
                evidence={"split": split},
            )
        declared = split_summary[split]
        if any(
            declared.get(key) != value for key, value in split_actual[split].items()
        ):
            raise ReleaseQAError(
                "RELEASE_ACTIVATION_SPLIT_TOTAL",
                "activation split totals differ from validated sidecars",
                evidence={"split": split},
            )
    expected_ids = set(inventory.token_counts)
    if seen_record_ids != expected_ids:
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_COVERAGE",
            "activation shards do not cover the exact 1,200 token record identities",
            evidence={
                "missing_count": len(expected_ids - seen_record_ids),
                "unexpected_count": len(seen_record_ids - expected_ids),
            },
        )
    if total_tokens != EXPECTED_ACTIVATION_TOKEN_COUNT or total_tokens != sum(
        inventory.token_counts.values()
    ):
        raise ReleaseQAError(
            "RELEASE_ACTIVATION_TOKEN_COUNT",
            "activation manifest total differs from exact token label lengths",
        )
    return manifest, tuple(sidecar_inventory)


ShortcutRunner = Callable[[Path], Mapping[str, Any]]


def _default_shortcut_runner(root: Path) -> Mapping[str, Any]:
    from molhallulens.modules.release.shortcut_audit import run_t047_shortcut_audit

    return run_t047_shortcut_audit(dry_run_root=root)


def _validate_shortcut_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise ReleaseQAError(
            "RELEASE_SHORTCUT_REPORT",
            "shortcut runner must return a mapping",
        )
    inventory = report.get("inventory")
    development = report.get("development_inventory")
    baselines = report.get("baselines")
    gates = report.get("mandatory_gates")
    protocol = report.get("audit_protocol")
    heldout = report.get("heldout_test_diagnostics")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping)
            or type(gate.get("passed")) is not bool
            or type(gate.get("actual")) not in {int, float}
            or gate.get("comparator") not in {"<=", "<", "=="}
            or type(gate.get("threshold")) not in {int, float}
            for gate in gates.values()
        )
    ):
        raise ReleaseQAError(
            "RELEASE_SHORTCUT_REPORT",
            "full release shortcut gates are missing or malformed",
        )
    computed_gate_results = {
        name: (
            gate["actual"] <= gate["threshold"]
            if gate["comparator"] == "<="
            else gate["actual"] < gate["threshold"]
            if gate["comparator"] == "<"
            else gate["actual"] == gate["threshold"]
        )
        for name, gate in gates.items()
    }
    if any(
        gates[name].get("passed") is not computed
        for name, computed in computed_gate_results.items()
    ):
        raise ReleaseQAError(
            "RELEASE_SHORTCUT_REPORT",
            "shortcut gate pass flags disagree with their frozen comparators",
        )
    recommended_all_pass = all(computed_gate_results.values())
    if (
        report.get("all_pass") is not recommended_all_pass
        or not isinstance(inventory, Mapping)
        or inventory.get("origin_count") != 150
        or inventory.get("record_count") != 1200
        or inventory.get("split_record_counts") != dict(SPLIT_RECORD_COUNTS)
        or not isinstance(development, Mapping)
        or development.get("origin_count") != 125
        or development.get("record_count") != 1000
        or not isinstance(baselines, Mapping)
        or not _REQUIRED_SHORTCUT_BASELINES.issubset(baselines)
        or not isinstance(protocol, Mapping)
        or protocol.get("test_used_for_model_or_threshold_selection") is not False
        or not isinstance(heldout, Mapping)
        or heldout.get("record_count") != 200
        or heldout.get("origin_count") != 25
        or heldout.get("used_for_candidate_layer_or_threshold_selection") is not False
    ):
        raise ReleaseQAError(
            "RELEASE_SHORTCUT_REPORT",
            "full release shortcut/symbolic suite is incomplete or inconsistent",
        )
    failed_recommended_gates = tuple(
        sorted(name for name, gate in gates.items() if gate.get("passed") is not True)
    )
    return {
        "format_version": T052_SHORTCUT_FORMAT_VERSION,
        "release_id": T052_RELEASE_ID,
        "scope": "full_release_dev_gated_with_one_heldout_test_diagnostic",
        "inventory": dict(inventory),
        "development_inventory": dict(development),
        "audit_protocol": dict(protocol),
        "mandatory_gates": dict(gates),
        "audit_completed": True,
        "recommended_go_no_go_all_pass": recommended_all_pass,
        "failed_recommended_gates": failed_recommended_gates,
        "release_acceptance_role": (
            "diagnostic_disclosure_not_a_threshold_gate_in_section_19"
        ),
        "baselines": dict(baselines),
        "matching": report.get("matching"),
        "heldout_test_diagnostics": dict(heldout),
        "strict_symbolic_graph_edit_verifier": {
            "record_count": 1200,
            "verified_record_count": 1200,
            "pass_rate": 1.0,
            "detector_visible": False,
            "source": "release records[*].verification.graph_edit_verified and split strict reports",
        },
        "limitations": [
            "Recommended learned shortcut screens use train and validation only; test remains a one-time diagnostic.",
            "Attack metrics are engineering screens at the origin/leakage-group unit, not paper performance estimates.",
            "Symbolic chemistry comparators measure legitimate executable chemistry signal and are disclosed, not suppressed.",
        ],
        "all_pass": recommended_all_pass,
    }


def _validate_poe_capability(project_root: Path) -> dict[str, Any]:
    report = _read_json(
        project_root / "Dataset/reports/poe_capability_probe.json",
        "Poe capability report",
    )
    mock = report.get("deterministic_mock_validation")
    live = report.get("live_probe")
    if (
        report.get("required_model_id") != POE_MODEL_ID
        or not isinstance(mock, Mapping)
        or mock.get("execution_status") != "passed"
        or mock.get("requested_model_id") != POE_MODEL_ID
        or not isinstance(live, Mapping)
        or live.get("execution_status") not in {"passed", "offline_not_executed"}
    ):
        raise ReleaseQAError(
            "RELEASE_POE_CAPABILITY",
            "Poe capability snapshot does not bind the frozen model contract",
        )
    return {
        "required_model_id": POE_MODEL_ID,
        "deterministic_mock_status": mock.get("execution_status"),
        "live_probe_status": live.get("execution_status"),
        "live_probe_reason_code": live.get("reason_code"),
        "upstream_snapshot_selectable": False,
    }


def _validate_t044_replay_evidence(project_root: Path) -> dict[str, Any]:
    report_path = project_root / "Dataset/reports/t044_golden_validation.json"
    fixture_path = project_root / "tests/golden/t044_extended_golden_suite.json"
    report = _read_json(report_path, "T044 deterministic golden report")
    fixture = _read_json(fixture_path, "T044 deterministic golden fixture")
    summary = report.get("summary")
    coverage = fixture.get("coverage")
    execution = fixture.get("execution")
    bundles = fixture.get("origin_bundles")
    report_origins = report.get("origins")
    valid_bundles = isinstance(bundles, list) and all(
        isinstance(bundle, Mapping) for bundle in bundles
    )
    fixture_origin_ids = (
        tuple(bundle.get("origin_id") for bundle in bundles) if valid_bundles else ()
    )
    fixture_record_count = (
        sum(
            len(bundle.get("records", ()))
            for bundle in bundles
            if isinstance(bundle.get("records"), list)
        )
        if valid_bundles
        else 0
    )
    report_origin_ids = (
        tuple(item.get("origin_id") for item in report_origins)
        if isinstance(report_origins, list)
        and all(isinstance(item, Mapping) for item in report_origins)
        else ()
    )
    if (
        report.get("format_version") != "t044_golden_validation_v1"
        or report.get("all_pass") is not True
        or not isinstance(summary, Mapping)
        or summary.get("record_count") != 72
        or summary.get("complete_real_origin_count") != 9
        or summary.get("live_poe_attempt_count") != 0
        or fixture.get("format_version") != "t044_extended_golden_suite_v1"
        or fixture.get("dataset_version") != DATASET_VERSION
        or not isinstance(coverage, Mapping)
        or coverage.get("complete_real_origin_count") != 9
        or coverage.get("complete_record_count") != 72
        or not isinstance(execution, Mapping)
        or execution.get("deterministic_replay") is not True
        or execution.get("live_poe_attempted") is not False
        or not valid_bundles
        or len(fixture_origin_ids) != 9
        or len(set(fixture_origin_ids)) != 9
        or fixture_record_count != 72
        or not isinstance(report_origins, list)
        or set(report_origin_ids) != set(fixture_origin_ids)
        or any(
            item.get("all_pass") is not True
            or item.get("record_count") != 8
            or item.get("issue_codes") != []
            for item in report_origins
        )
    ):
        raise ReleaseQAError(
            "RELEASE_T044_REPLAY_EVIDENCE",
            "committed T044 deterministic replay evidence is incomplete",
        )
    return {
        "report_path": "Dataset/reports/t044_golden_validation.json",
        "fixture_path": "tests/golden/t044_extended_golden_suite.json",
        "all_pass": True,
        "origin_count": 9,
        "record_count": 72,
        "live_poe_attempt_count": 0,
    }


def _gate(gate_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {"gate_id": gate_id, "status": "passed", "evidence": dict(evidence)}


def _zero_call_ledger() -> dict[str, Any]:
    return {
        "format_version": T052_LEDGER_FORMAT_VERSION,
        "release_id": T052_RELEASE_ID,
        "provider": "poe",
        "model_id": POE_MODEL_ID,
        "visibility": "private_owner_only",
        "network_request_count": 0,
        "record_count": 0,
        "records": [],
        "point_usage": {
            "recorded_cost_points": 0.0,
            "live_balance_probe_performed": False,
            "reason": "all accepted release candidates and renderers used deterministic offline RULE paths",
        },
        "cache": {
            "entry_count": 0,
            "replay_status": "passed_vacuously_empty_cache",
            "network_calls_during_release_replay": 0,
            "byte_identity_claimed": False,
        },
    }


def _ledger_export_descriptor(private_path: Path) -> dict[str, Any]:
    """Return the public, secret-free pointer to the external ledger."""

    return {
        "format_version": T052_LEDGER_DESCRIPTOR_FORMAT_VERSION,
        "release_id": T052_RELEASE_ID,
        "descriptor_visibility": "public_secret_free",
        "export_kind": "external_private_ledger_reference",
        "external_ledger_path": str(private_path),
        "external_ledger_required_mode": "0600",
        "mode_verification": "required_before_public_artifact_publication",
        "project_contains_private_ledger": False,
        "network_request_count": 0,
        "record_count": 0,
        "recorded_cost_points": 0.0,
        "secret_free": True,
    }


def _known_limitations_text(poe: Mapping[str, Any], shortcut: Mapping[str, Any]) -> str:
    failed_shortcut = tuple(shortcut.get("failed_recommended_gates", ()))
    shortcut_gates = shortcut["mandatory_gates"]
    shortcut_findings = (
        ", ".join(
            f"{name}={shortcut_gates[name].get('actual')} "
            f"({shortcut_gates[name].get('comparator')} "
            f"{shortcut_gates[name].get('threshold')})"
            for name in failed_shortcut
        )
        if failed_shortcut
        else "none"
    )
    shortcut_limitation = (
        "10. **Full-release shortcut findings.** The complete 1,200-record "
        "diagnostic did not pass every recommended Section 13 engineering screen: "
        f"{shortcut_findings}. Section 19 requires the shortcut and symbolic suites "
        "to be run and disclosed, but does not make these recommended thresholds a "
        "release-checklist gate. This finding reinforces the smoke-test-only "
        "restriction and is not represented as a passed screen."
        if failed_shortcut
        else "10. **Full-release shortcut diagnostics.** Every recommended Section 13 "
        "engineering screen passed in this run; the metrics remain engineering "
        "diagnostics rather than paper-level performance estimates."
    )
    return f"""# Known limitations — MolHalluLens Molecule Editing pilot_v1

1. **Benchmark-use restriction.** These 150 ChemCoTBench-V2 origins are approved only for pipeline/schema smoke testing, construction validation, and feasibility auditing. They are not authorized for detector training, detector-layer selection, or final threshold tuning.
2. **Poe snapshot limitation.** Poe did not expose a selectable upstream snapshot for `{POE_MODEL_ID}`. The recorded live capability status is `{poe["live_probe_status"]}`; deterministic mock capability checks passed, while this release made zero live Poe requests.
3. **Rule-only accepted candidates.** All 600 hallucinated candidate executions in the final release came from deterministic RULE paths. The release therefore does not measure LLM candidate diversity.
4. **Review scope.** T046 was a Codex-assisted structured chemistry review, not an independent external human-chemist review.
5. **Historical repaired findings.** Dry-run review found a generic-instruction issue and a PARTIAL Answer-to-oracle correctness issue. Both root causes were repaired and affected complete pairs were rebuilt before the full split builds; T052 retains regression gates for both boundaries.
6. **Shortcut estimates.** Shallow attack metrics are engineering go/no-go screens over origins/leakage groups, not paper-level model estimates. High symbolic-comparator performance is legitimate executable chemistry signal and is disclosed rather than suppressed.
7. **Release identity override.** The original cryptographic release-identity criterion was explicitly overridden by the user. No such computation or verification is performed by T052. The effective freeze uses `pilot_v1`, exact artifact paths, exact row counts, and exact record/origin identity sets.
8. **Activation scope.** `resid_post` is extracted only at the pre-frozen ChemDFM-R layer 26 with exact `post_token_h_t` alignment. Pilot records did not select that layer.
9. **Checkpoint/tokenizer identity boundary.** The approved local checkpoint path plus Qwen2 model/tokenizer configuration identifiers validate the expected architecture, tensor shapes, and tokenizer configuration only. Under the explicit no-digest constraint they do not establish byte-exact checkpoint or tokenizer identity, and this release does not claim byte-exact checkpoint/tokenizer reproducibility.
{shortcut_limitation}
"""


def _dataset_card_text(
    shortcut: Mapping[str, Any],
    activation: Mapping[str, Any],
    poe: Mapping[str, Any],
) -> str:
    gates = shortcut["mandatory_gates"]
    metric_lines = "\n".join(
        f"- `{name}`: {gate.get('actual')} ({gate.get('comparator')} {gate.get('threshold')}; "
        f"{'passed' if gate.get('passed') is True else 'did not pass'})"
        for name, gate in sorted(gates.items())
    )
    shortcut_summary = (
        "All recommended engineering screens passed."
        if shortcut.get("recommended_go_no_go_all_pass") is True
        else "Not all recommended engineering screens passed; the exact findings are retained below and in `KNOWN_LIMITATIONS.md`."
    )
    token_count = activation.get("token_count")
    return f"""# MolHalluLens Molecule Editing `pilot_v1`

## Summary

This release is a counterfactual hallucination-detection pilot for molecule editing. It contains 150 origins and 1,200 detector records: 800 train, 200 validation, and 200 test. Every origin contributes four reciprocal H/N pairs—LOCAL, PARTIAL, FULL_CF, and TERMINAL—for exactly four hallucinated and four faithful records.

Detector-visible text is always serialized as indexed SMILES → instruction → reasoning → final answer. Hidden oracle, typed state graphs, real ChemDFM-R token labels, private provenance, and layer-26 activation artifacts are stored separately and joined only by record identity.

## Intended use and Risk 6 decision

These 150 ChemCoTBench-V2 origins are **pipeline/schema smoke-test material only**. They must not be used for formal detector training, detector-layer selection, or final threshold tuning. Split names describe construction isolation and do not grant training authorization. Statistical units are origins/leakage groups, not the 1,200 correlated records.

## Composition

- Origins: train/validation/test = 100/25/25.
- Records: train/validation/test = 800/200/200.
- Labels: 600 H and 600 matched N.
- Per split and per policy, H/N counts are exactly balanced.
- Accepted candidate source: deterministic RULE paths.
- Strict validation: 4,800 artifact gates plus 150 complete-bundle gates.

## Tokens and activations

All 1,200 records use one real local ChemDFM-R fast-tokenizer fingerprint. Every direct and multi-axis label array has the exact input-token length. Activations are ChemDFM-R layer-26 `resid_post` features with `post_token_h_t` alignment, zero label shift, 5,120 hidden dimensions, and {token_count} total token positions. No pre-token target is claimed.

## Shortcut and symbolic baselines

Recommended learned screens use train+validation only; held-out test is a one-time diagnostic that cannot feed candidate, layer, renderer, or threshold selection. {shortcut_summary}

{metric_lines}

The report also includes nearest-neighbor retrieval, RDKit visible validity, visible reasoning/answer graph comparison, hidden-oracle graph comparison, per-policy symbolic slices, and the strict graph-edit verifier.

## Poe provenance

The frozen provider model is `{POE_MODEL_ID}`. Deterministic capability mocks passed. Live capability status is `{poe["live_probe_status"]}` because Poe exposes no selectable upstream snapshot. The final release made zero live Poe calls and contains no cached Poe response entries. Its zero-call usage ledger is stored outside the project on a filesystem that enforces owner-only mode; the repository contains only a secret-free export descriptor.

## Release identity

At the user's explicit instruction, T052 performs no cryptographic identity computation or verification. The original criterion is recorded as `overridden_not_evaluated`, never as passed. The effective release freeze is `pilot_v1` plus exact artifact paths, exact row counts, and exact record/origin identity sets.

See `KNOWN_LIMITATIONS.md` and `reports/release_validation_report.json` before use.
"""


@dataclass(frozen=True, slots=True)
class ReleaseQAResult:
    dataset_manifest: Mapping[str, Any]
    validation_report: Mapping[str, Any]
    shortcut_report: Mapping[str, Any]
    poe_private_ledger: Mapping[str, Any]
    poe_ledger_descriptor: Mapping[str, Any]
    dataset_card: str
    known_limitations: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.dataset_manifest, "dataset_manifest"),
            (self.validation_report, "validation_report"),
            (self.shortcut_report, "shortcut_report"),
            (self.poe_private_ledger, "poe_private_ledger"),
            (self.poe_ledger_descriptor, "poe_ledger_descriptor"),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        if type(self.dataset_card) is not str or not self.dataset_card:
            raise TypeError("dataset_card must be non-empty text")
        if type(self.known_limitations) is not str or not self.known_limitations:
            raise TypeError("known_limitations must be non-empty text")

    def public_payloads(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "dataset_manifest.json": _render_json(self.dataset_manifest),
                "reports/release_validation_report.json": _render_json(
                    self.validation_report
                ),
                "reports/shortcut_baseline_report.json": _render_json(
                    self.shortcut_report
                ),
                LEDGER_DESCRIPTOR_RELATIVE_PATH: _render_json(
                    self.poe_ledger_descriptor
                ),
                "DATASET_CARD.md": self.dataset_card,
                "KNOWN_LIMITATIONS.md": self.known_limitations,
            }
        )

    def private_payload(self) -> str:
        return _render_json(self.poe_private_ledger)


def run_t052_release_qa(
    *,
    release_root: Path | None = None,
    project_root: Path | None = None,
    private_ledger_path: Path | None = None,
    shortcut_runner: ShortcutRunner | None = None,
    release_timestamp: str = RELEASE_TIMESTAMP,
) -> ReleaseQAResult:
    """Run all effective release gates without mutating release artifacts."""

    root = DEFAULT_RELEASE_ROOT if release_root is None else Path(release_root)
    project = DEFAULT_PROJECT_ROOT if project_root is None else Path(project_root)
    resolved_private_path = _resolve_private_ledger_path(
        private_ledger_path,
        release_root=root,
        project_root=project,
    )
    if type(release_timestamp) is not str or not release_timestamp:
        raise ValueError("release_timestamp must be non-empty text")
    runner = _default_shortcut_runner if shortcut_runner is None else shortcut_runner
    if not callable(runner):
        raise TypeError("shortcut_runner must be callable or None")
    inventory = _load_and_validate_release(root)
    strict_reports = _validate_build_reports(root, inventory)
    test_isolation = _validate_test_isolation(root)
    token_manifest = _validate_token_manifest(root, inventory)
    activation_manifest, activation_inventory = _validate_activation_manifest(
        root, inventory
    )
    shortcut = _validate_shortcut_report(runner(root))
    poe = _validate_poe_capability(project)
    t044_replay = _validate_t044_replay_evidence(project)
    if inventory.network_request_count != 0:
        raise ReleaseQAError(
            "RELEASE_POE_NETWORK_CALL",
            "release has a nonzero Poe request count",
        )
    if inventory.provenance_cache_entry_count != 0:
        raise ReleaseQAError(
            "RELEASE_POE_CACHE_UNEXPECTED",
            "this rule-only release unexpectedly references Poe cache entries",
        )
    ledger = _zero_call_ledger()
    ledger_descriptor = _ledger_export_descriptor(resolved_private_path)
    gates = (
        _gate(
            "exact_release_inventory",
            {
                "origin_count": 150,
                "record_count": 1200,
                "split_record_counts": dict(SPLIT_RECORD_COUNTS),
            },
        ),
        _gate(
            "complete_reciprocal_bundles",
            {
                "records_per_origin": 8,
                "hallucinated_per_origin": 4,
                "faithful_per_origin": 4,
            },
        ),
        _gate("policy_balance", {"policies": POLICIES, "per_origin_each_label": 1}),
        _gate(
            "five_family_identity",
            {"families": tuple(FAMILIES), "joined_record_count": 1200},
        ),
        _gate(
            "split_leakage_and_donor_isolation",
            {
                "cross_split_origin_count": 0,
                "cross_split_group_count": 0,
                "cross_split_donor_count": 0,
            },
        ),
        _gate(
            "gt_and_secret_boundary",
            {"public_oracle_field_count": 0, "credential_finding_count": 0},
        ),
        _gate(
            "detector_serialization_order",
            {
                "order": (
                    "indexed_smiles",
                    "instruction",
                    "reasoning_chain",
                    "final_answer",
                ),
                "direct_utf8_byte_match_count": 1200,
            },
        ),
        _gate(
            "strict_chemistry_and_bundle_chain",
            {
                "artifact_gate_count": sum(
                    item["artifact_gate_count"] for item in strict_reports.values()
                ),
                "bundle_gate_count": 150,
                "graph_edit_verified_record_count": 1200,
            },
        ),
        _gate(
            "real_token_projection",
            {
                "record_count": token_manifest["record_count"],
                "activation_alignment": ACTIVATION_ALIGNMENT,
                "equal_length": True,
            },
        ),
        _gate(
            "activation_manifest_alignment",
            {
                "record_count": activation_manifest["record_count"],
                "token_count": activation_manifest["token_count"],
                "shard_count": activation_manifest["shard_count"],
                "tensor_payload_validation_performed": True,
                "activation_dtype": "bfloat16",
                "ordered_record_ids_exact": True,
                "token_counts_exact": True,
                "row_offsets_exact": True,
                "activation_shape_exact": True,
                "activation_alignment_exact": True,
                "hidden_size": 5120,
                "layer_index": 26,
                "label_shift": 0,
            },
        ),
        _gate(
            "test_isolation",
            {
                "used_for_selection": False,
                "strict_record_acceptance_only": test_isolation["test_usage"][
                    "used_for_strict_record_acceptance"
                ],
            },
        ),
        _gate(
            "shortcut_and_symbolic_baselines",
            {
                "recommended_screen_count": len(shortcut["mandatory_gates"]),
                "audit_completed": True,
                "recommended_go_no_go_all_pass": shortcut[
                    "recommended_go_no_go_all_pass"
                ],
                "failed_recommended_gates": shortcut["failed_recommended_gates"],
                "heldout_feedback": False,
            },
        ),
        _gate(
            "poe_private_zero_call_ledger",
            {
                "model_id": POE_MODEL_ID,
                "network_request_count": 0,
                "cache_entry_count": 0,
            },
        ),
        _gate(
            "empty_cache_replay_safety",
            {
                "status": "passed_vacuously_empty_cache",
                "comparable_cache_entry_count": 0,
                "network_calls": 0,
                "byte_identity_claimed": False,
                "t044_deterministic_replay": t044_replay,
            },
        ),
    )
    record_ids = tuple(sorted(inventory.records_by_id["records"]))
    origin_ids = tuple(
        sorted(
            origin
            for values in inventory.origins_by_split.values()
            for origin in values
        )
    )
    artifact_inventory = tuple(inventory.artifact_inventory) + (
        {
            "path": "split_manifest.csv",
            "artifact_family": "frozen_split_manifest",
            "row_count": 150,
        },
        {
            "path": "tokenized/chemdfm_r/manifest.json",
            "artifact_family": "real_tokenizer_manifest",
            "row_count": 1200,
        },
        {
            "path": "tokenized/chemdfm_r/git_shards/index.json",
            "artifact_family": "tokenized_git_shard_index",
            "row_count": 1200,
        },
        {
            "path": "activations/chemdfm_r/layer_26/manifest.json",
            "artifact_family": "activation_manifest",
            "row_count": 1200,
        },
        *activation_inventory,
    )
    override = {
        "original_acceptance_item": "pilot_v1 cryptographic identity frozen",
        "status": "overridden_not_evaluated",
        "passed": None,
        "reason": "explicit user instruction prohibits computation or verification",
        "replacement_freeze": {
            "dataset_version": DATASET_VERSION,
            "exact_paths": True,
            "exact_row_counts": True,
            "exact_record_identity_set": True,
            "exact_origin_identity_set": True,
        },
        "checkpoint_tokenizer_identity_scope": {
            "evidence": "approved_local_path_plus_qwen2_configuration",
            "provenance_claim_only": True,
            "byte_exact_identity_proven": False,
            "byte_exact_reproducibility_claimed": False,
        },
    }
    shortcut_has_findings = shortcut["recommended_go_no_go_all_pass"] is not True
    release_status = (
        "approved_with_explicit_identity_override_and_disclosed_shortcut_findings"
        if shortcut_has_findings
        else "approved_with_explicit_identity_override"
    )
    dataset_manifest = {
        "format_version": T052_MANIFEST_FORMAT_VERSION,
        "release_id": T052_RELEASE_ID,
        "dataset_version": DATASET_VERSION,
        "release_status": release_status,
        "released_at": release_timestamp,
        "freeze_method": "dataset_version_exact_paths_counts_and_identity_sets",
        "identity_override": override,
        "summary": {
            "origin_count": 150,
            "record_count": 1200,
            "hallucinated_record_count": 600,
            "faithful_record_count": 600,
            "split_origin_counts": dict(SPLIT_ORIGIN_COUNTS),
            "split_record_counts": dict(SPLIT_RECORD_COUNTS),
            "records_per_origin": 8,
            "artifact_family_count": 5,
        },
        "origin_ids": origin_ids,
        "record_ids": record_ids,
        "artifact_inventory": artifact_inventory,
        "tokenization": {
            "manifest_path": "tokenized/chemdfm_r/manifest.json",
            "record_count": 1200,
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "real_fast_tokenizer": True,
        },
        "activations": {
            "manifest_path": "activations/chemdfm_r/layer_26/manifest.json",
            "record_count": 1200,
            "token_count": EXPECTED_ACTIVATION_TOKEN_COUNT,
            "shard_count": EXPECTED_ACTIVATION_SHARD_COUNT,
            "feature": "resid_post",
            "layer_index": 26,
            "hidden_size": 5120,
            "activation_dtype": "bfloat16",
            "tensor_payload_validation_performed": True,
            "activation_alignment": ACTIVATION_ALIGNMENT,
        },
        "shortcut_diagnostics": {
            "audit_completed": True,
            "recommended_go_no_go_all_pass": shortcut["recommended_go_no_go_all_pass"],
            "failed_recommended_gates": shortcut["failed_recommended_gates"],
            "release_acceptance_role": shortcut["release_acceptance_role"],
        },
        "reports": {
            "release_validation": "reports/release_validation_report.json",
            "shortcut_baselines": "reports/shortcut_baseline_report.json",
            "dataset_card": "DATASET_CARD.md",
            "known_limitations": "KNOWN_LIMITATIONS.md",
            "poe_usage_ledger": {
                "storage": "external_owner_only",
                "external_path": str(resolved_private_path),
                "required_mode": "0600",
                "public_descriptor": LEDGER_DESCRIPTOR_RELATIVE_PATH,
                "project_contains_private_ledger": False,
            },
        },
    }
    validation_report = {
        "format_version": T052_FORMAT_VERSION,
        "release_id": T052_RELEASE_ID,
        "dataset_version": DATASET_VERSION,
        "evaluated_at": release_timestamp,
        "effective_acceptance_status": release_status,
        "all_effective_gates_pass": True,
        "gate_count": len(gates),
        "gates": gates,
        "identity_override": override,
        "poe_snapshot": poe,
        "replay": {
            "release_network_request_count": 0,
            "cache_entry_count": 0,
            "cache_replay_status": "passed_vacuously_empty_cache",
            "comparable_cache_entry_count": 0,
            "byte_identity_check_status": "not_applicable_empty_cache",
            "byte_identity_claimed": False,
            "detector_input_serialization_direct_utf8_byte_match_count": 1200,
            "t044_deterministic_replay_evidence": t044_replay,
        },
        "strict_validation": {
            "artifact_gate_count": 4800,
            "bundle_gate_count": 150,
            "all_pass": True,
        },
        "shortcut_diagnostics": dataset_manifest["shortcut_diagnostics"],
        "shortcut_report_path": "reports/shortcut_baseline_report.json",
        "dataset_manifest_path": "dataset_manifest.json",
    }
    _assert_secret_free(dataset_manifest, "dataset_manifest.json")
    _assert_secret_free(validation_report, "reports/release_validation_report.json")
    _assert_secret_free(shortcut, "reports/shortcut_baseline_report.json")
    _assert_secret_free(ledger, str(resolved_private_path))
    _assert_secret_free(ledger_descriptor, LEDGER_DESCRIPTOR_RELATIVE_PATH)
    return ReleaseQAResult(
        dataset_manifest=MappingProxyType(dataset_manifest),
        validation_report=MappingProxyType(validation_report),
        shortcut_report=MappingProxyType(shortcut),
        poe_private_ledger=MappingProxyType(ledger),
        poe_ledger_descriptor=MappingProxyType(ledger_descriptor),
        dataset_card=_dataset_card_text(shortcut, activation_manifest, poe),
        known_limitations=_known_limitations_text(poe, shortcut),
    )


def _assert_no_conflict(path: Path, payload: str, code: str) -> None:
    if path.exists() and (
        not path.is_file() or path.read_text(encoding="utf-8") != payload
    ):
        raise ReleaseQAError(
            code,
            "existing release QA artifact differs from deterministic output",
            evidence={"path": str(path)},
        )


def _require_owner_only_mode(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
        observed = stat.S_IMODE(path.stat().st_mode)
    except OSError as error:
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_MODE",
            "private ledger filesystem cannot enforce owner-only mode",
            evidence={"path": str(path)},
        ) from error
    if observed != 0o600:
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_MODE",
            "private ledger filesystem ignored the required owner-only mode",
            evidence={"path": str(path), "observed_mode": oct(observed)},
        )


def _publish_owner_only_ledger(path: Path, payload: str) -> None:
    """Publish outside the project only after the target FS proves mode 0600."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_conflict(path, payload, "RELEASE_QA_PRIVATE_CONFLICT")
    if path.exists():
        _require_owner_only_mode(path)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        _require_owner_only_mode(temporary)
        if path.exists():
            _assert_no_conflict(path, payload, "RELEASE_QA_PRIVATE_CONFLICT")
            _require_owner_only_mode(path)
            return
        temporary.replace(path)
        installed = True
        _require_owner_only_mode(path)
    except Exception:
        if installed and path.exists():
            path.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def _result_private_path(value: ReleaseQAResult) -> Path:
    raw = value.poe_ledger_descriptor.get("external_ledger_path")
    if type(raw) is not str or not raw:
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_DESCRIPTOR",
            "release result lacks an external private-ledger path",
        )
    return Path(raw).resolve(strict=False)


def _release_output_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ReleaseQAError(
            "RELEASE_QA_OUTPUT_PATH",
            "release QA output paths must be safe release-root-relative paths",
            evidence={"path": relative},
        )
    release = root.resolve(strict=False)
    resolved = (release / value).resolve(strict=False)
    if not _is_within(resolved, release):
        raise ReleaseQAError(
            "RELEASE_QA_OUTPUT_PATH",
            "release QA output path escapes through a symlink",
            evidence={"path": relative},
        )
    return resolved


def _stage_text(path: Path, payload: str, *, owner_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != payload:
        raise ReleaseQAError(
            "RELEASE_QA_STAGING",
            "staged release artifact failed exact readback",
            evidence={"path": str(path)},
        )
    if owner_only:
        _require_owner_only_mode(path)


def _install_staged_file(
    staged: Path,
    target: Path,
    payload: str,
    *,
    private: bool = False,
) -> bool:
    """Install one prepared file atomically; return whether this call created it."""

    if target.exists():
        _assert_no_conflict(
            target,
            payload,
            "RELEASE_QA_PRIVATE_CONFLICT"
            if private
            else "RELEASE_QA_ARTIFACT_CONFLICT",
        )
        if private:
            _require_owner_only_mode(target)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(target)
    if private:
        try:
            _require_owner_only_mode(target)
        except Exception:
            if target.exists():
                target.unlink()
            raise
    return True


def write_t052_release_artifacts(
    *,
    release_root: Path | None = None,
    project_root: Path | None = None,
    external_report_path: Path | None = None,
    private_ledger_path: Path | None = None,
    shortcut_runner: ShortcutRunner | None = None,
    result: ReleaseQAResult | None = None,
    release_timestamp: str = RELEASE_TIMESTAMP,
) -> ReleaseQAResult:
    """Publish public QA files plus one external owner-only zero-call ledger."""

    root = DEFAULT_RELEASE_ROOT if release_root is None else Path(release_root)
    project = DEFAULT_PROJECT_ROOT if project_root is None else Path(project_root)
    private_path = _resolve_private_ledger_path(
        private_ledger_path,
        release_root=root,
        project_root=project,
    )
    external = (
        DEFAULT_EXTERNAL_REPORT_PATH
        if external_report_path is None and project_root is None
        else project / "Dataset/reports/t052_release_qa.json"
        if external_report_path is None
        else Path(external_report_path)
    )
    value = (
        run_t052_release_qa(
            release_root=root,
            project_root=project,
            private_ledger_path=private_path,
            shortcut_runner=shortcut_runner,
            release_timestamp=release_timestamp,
        )
        if result is None
        else result
    )
    if type(value) is not ReleaseQAResult:
        raise TypeError("result must be ReleaseQAResult or None")
    if _result_private_path(value) != private_path:
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_DESCRIPTOR",
            "release result and requested private-ledger paths disagree",
            evidence={
                "result_path": str(_result_private_path(value)),
                "requested_path": str(private_path),
            },
        )
    public = value.public_payloads()
    private_payload = value.private_payload()
    external_payload = _render_json(value.validation_report)
    root = root.resolve(strict=False)
    external = external.resolve(strict=False)
    public_targets = {
        relative: _release_output_path(root, relative) for relative in public
    }
    all_targets = [*public_targets.values(), external, private_path]
    if len(set(all_targets)) != len(all_targets):
        raise ReleaseQAError(
            "RELEASE_QA_OUTPUT_PATH",
            "public, external, and private release outputs must have unique paths",
        )
    for relative, payload in public.items():
        _assert_no_conflict(
            public_targets[relative], payload, "RELEASE_QA_ARTIFACT_CONFLICT"
        )
    _assert_no_conflict(external, external_payload, "RELEASE_QA_REPORT_CONFLICT")
    _assert_no_conflict(private_path, private_payload, "RELEASE_QA_PRIVATE_CONFLICT")
    legacy_private_path = _release_output_path(root, "private/poe_usage_ledger.json")
    if legacy_private_path.exists():
        raise ReleaseQAError(
            "RELEASE_QA_PRIVATE_LOCATION",
            "project tree contains a legacy private ledger instead of a descriptor",
            evidence={"path": str(legacy_private_path)},
        )
    public_staging: Path | None = None
    external_staging: Path | None = None
    private_staging: Path | None = None
    installed: list[Path] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        external.parent.mkdir(parents=True, exist_ok=True)
        private_path.parent.mkdir(parents=True, exist_ok=True)
        public_staging = Path(tempfile.mkdtemp(prefix=".t052-staging-", dir=root))
        external_descriptor, external_name = tempfile.mkstemp(
            prefix=f".{external.name}.", dir=external.parent
        )
        os.close(external_descriptor)
        external_staging = Path(external_name)
        private_descriptor, private_name = tempfile.mkstemp(
            prefix=f".{private_path.name}.", dir=private_path.parent
        )
        os.close(private_descriptor)
        private_staging = Path(private_name)
        for relative, payload in public.items():
            _stage_text(public_staging / relative, payload)
        _stage_text(external_staging, external_payload)
        _stage_text(private_staging, private_payload, owner_only=True)

        if _install_staged_file(
            private_staging, private_path, private_payload, private=True
        ):
            installed.append(private_path)
        marker = "reports/release_validation_report.json"
        for relative, payload in public.items():
            if relative == marker:
                continue
            if _install_staged_file(
                public_staging / relative, public_targets[relative], payload
            ):
                installed.append(public_targets[relative])
        if _install_staged_file(external_staging, external, external_payload):
            installed.append(external)
        if _install_staged_file(
            public_staging / marker, public_targets[marker], public[marker]
        ):
            installed.append(public_targets[marker])

        for relative, payload in public.items():
            _assert_no_conflict(
                public_targets[relative], payload, "RELEASE_QA_ATOMIC_PUBLISH"
            )
        _assert_no_conflict(external, external_payload, "RELEASE_QA_ATOMIC_PUBLISH")
        _assert_no_conflict(private_path, private_payload, "RELEASE_QA_ATOMIC_PUBLISH")
        _require_owner_only_mode(private_path)
    except Exception as error:
        rollback_errors: list[str] = []
        for installed_path in reversed(installed):
            try:
                if installed_path.exists():
                    installed_path.unlink()
            except OSError:
                rollback_errors.append(str(installed_path))
        if rollback_errors:
            raise ReleaseQAError(
                "RELEASE_QA_ROLLBACK_FAILED",
                "release publication failed and rollback could not remove new outputs",
                evidence={"paths": tuple(rollback_errors)},
            ) from error
        if isinstance(error, ReleaseQAError):
            raise
        raise ReleaseQAError(
            "RELEASE_QA_ATOMIC_PUBLISH",
            "release publication failed; every newly installed output was rolled back",
        ) from error
    finally:
        if public_staging is not None and public_staging.exists():
            shutil.rmtree(public_staging)
        for staged in (external_staging, private_staging):
            if staged is not None and staged.exists():
                staged.unlink()
    return value


__all__ = [
    "ACTIVATION_ALIGNMENT",
    "DATASET_VERSION",
    "DEFAULT_EXTERNAL_REPORT_PATH",
    "DEFAULT_PROJECT_ROOT",
    "DEFAULT_RELEASE_ROOT",
    "POLICIES",
    "RELEASE_TIMESTAMP",
    "SPLIT_ORIGIN_COUNTS",
    "SPLIT_RECORD_COUNTS",
    "T052_FORMAT_VERSION",
    "T052_LEDGER_FORMAT_VERSION",
    "T052_MANIFEST_FORMAT_VERSION",
    "T052_RELEASE_ID",
    "T052_SHORTCUT_FORMAT_VERSION",
    "ReleaseQAError",
    "ReleaseQAResult",
    "run_t052_release_qa",
    "write_t052_release_artifacts",
]
