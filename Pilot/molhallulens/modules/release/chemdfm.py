"""Real ChemDFM-R tokenization and post-token residual extraction for T051.

The release records already contain canonical detector text, character spans,
and a frozen text-identity value.  This module deliberately carries that value
forward without recomputing or verifying a digest.  Token labels are projected
again from the canonical character spans using the approved local ChemDFM-R
fast tokenizer.

Activation alignment has exactly one meaning here: decoder token ``x_t`` is
consumed, ``model.model.layers[26]`` returns its block output, and that same
index is paired with label ``y_t``.  No label shift is supported.

PyTorch and Transformers are imported only by the runtime entry points so the
projection, validation, shard planning, and resume contract remain testable in
the lightweight Pilot environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

from molhallulens.config.paths import DEFAULT_RELEASE_ROOT, PROJECT_ROOT
from molhallulens.core import (
    CausalRole,
    EditErrorSubtype,
    HallucinationType,
    SegmentKind,
)

T051_FORMAT_VERSION: Final = "t051_chemdfm_r_post_token_v1"
T051_TOKENIZATION_FORMAT_VERSION: Final = "t051_real_tokenization_v1"
T051_ACTIVATION_MANIFEST_VERSION: Final = "t051_activation_manifest_v1"
T051_GIT_SHARD_INDEX_VERSION: Final = "t051_tokenized_git_shards_v1"
ACTIVATION_ALIGNMENT: Final = "post_token_h_t"
FROZEN_LAYER_INDEX: Final = 26
EXPECTED_MODEL_TYPE: Final = "qwen2"
EXPECTED_ARCHITECTURE: Final = "Qwen2ForCausalLM"
EXPECTED_HIDDEN_SIZE: Final = 5120
EXPECTED_LAYER_COUNT: Final = 48
DEFAULT_CHECKPOINT_PATH: Final = Path(
    os.environ.get(
        "MOLHALLULENS_CHEMDFM_CHECKPOINT",
        PROJECT_ROOT / "models" / "ChemDFM-R-14B",
    )
)
# This is frozen release provenance, not a path used to load code or weights.
FROZEN_LAYER_SELECTION_SOURCE: Final = (
    "/home/haoqian/Data/SAERAG/v3_Chem_SAE/"
    "Stage1_layer_selection_v3_multimodel"
)
GIT_SHARD_MAX_BYTES: Final = 48_000_000
DEFAULT_SPLIT_COUNTS: Final = {
    "train": 800,
    "validation": 200,
    "test": 200,
}

_EVALUATED_SEGMENTS: Final = frozenset(
    {SegmentKind.REASONING.value, SegmentKind.FINAL_ANSWER.value}
)
_LOCAL_FALSEHOOD_ROLES: Final = frozenset(
    {
        CausalRole.ROOT.value,
        CausalRole.PROPAGATED_FALSE.value,
        CausalRole.TERMINAL.value,
    }
)
_DIRECT_TOKEN_ARRAY_FIELDS: Final = (
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
_NESTED_TOKEN_ARRAY_FIELDS: Final = (
    "semantic_type_masks",
    "edit_subtype_masks",
    "causal_role_masks",
)
_REPLACED_TOKEN_FIELDS: Final = frozenset(
    {
        "activation_alignment",
        "tokenizer_fingerprint",
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "segment_ids",
        "evaluation_mask",
        "hallucination_core_mask",
        "error_any_mask",
        "semantic_type_masks",
        "edit_subtype_masks",
        "causal_role_masks",
        "local_falsehood_mask",
        "off_task_branch_mask",
        "reasoning_mask",
        "answer_mask",
        "boundary_ambiguous_mask",
        "error_char_fraction",
    }
)


class T051ArtifactError(RuntimeError):
    """Fail-closed T051 tokenization or extraction error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("T051 error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("T051 error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("T051 error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = dict(evidence or {})
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class TokenizationSummary:
    split: str
    record_count: int
    token_count: int
    min_tokens: int
    max_tokens: int
    positive_span_count: int
    carried_text_identity_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "record_count": self.record_count,
            "token_count": self.token_count,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "positive_span_count": self.positive_span_count,
            "carried_text_identity_count": self.carried_text_identity_count,
        }


@dataclass(frozen=True, slots=True)
class ActivationShardPlan:
    split: str
    shard_index: int
    rows: tuple[Mapping[str, object], ...]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(str(row["record_id"]) for row in self.rows)

    @property
    def token_counts(self) -> tuple[int, ...]:
        return tuple(len(_require_sequence(row.get("input_ids"), "input_ids")) for row in self.rows)

    @property
    def stem(self) -> str:
        return f"{self.split}-{self.shard_index:05d}"


@dataclass(frozen=True, slots=True)
class ActivationShardSummary:
    split: str
    shard_index: int
    tensor_path: str
    metadata_path: str
    record_count: int
    token_count: int
    hidden_size: int
    layer_index: int
    resumed: bool
    file_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "shard_index": self.shard_index,
            "tensor_path": self.tensor_path,
            "metadata_path": self.metadata_path,
            "record_count": self.record_count,
            "token_count": self.token_count,
            "hidden_size": self.hidden_size,
            "layer_index": self.layer_index,
            "resumed": self.resumed,
            "file_bytes": self.file_bytes,
        }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise T051ArtifactError(
            "MAPPING_REQUIRED",
            f"{name} must be a mapping",
        )
    if any(type(key) is not str for key in value):
        raise T051ArtifactError(
            "STRING_KEYS_REQUIRED",
            f"{name} must use string keys",
        )
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise T051ArtifactError(
            "SEQUENCE_REQUIRED",
            f"{name} must be a non-string sequence",
        )
    return value


def _exact_int_sequence(value: object, name: str) -> tuple[int, ...]:
    values = tuple(_require_sequence(value, name))
    if values and isinstance(values[0], Sequence) and not isinstance(
        values[0], (str, bytes)
    ):
        raise T051ArtifactError(
            "TOKENIZER_BATCHED_OUTPUT",
            f"{name} must be one unbatched sequence",
        )
    if any(type(item) is not int for item in values):
        raise T051ArtifactError(
            "TOKENIZER_INTEGER_ARRAY_REQUIRED",
            f"{name} must contain exact integers",
        )
    return values


def _exact_string_sequence(value: object, name: str) -> tuple[str, ...]:
    values = tuple(_require_sequence(value, name))
    if any(type(item) is not str or not item for item in values):
        raise T051ArtifactError(
            "EXACT_STRING_ARRAY_REQUIRED",
            f"{name} must contain exact non-empty strings",
        )
    return values


def _offset_sequence(value: object) -> tuple[tuple[int, int], ...]:
    raw = _require_sequence(value, "offset_mapping")
    offsets: list[tuple[int, int]] = []
    for index, pair in enumerate(raw):
        values = _require_sequence(pair, f"offset_mapping[{index}]")
        if len(values) != 2 or any(type(item) is not int for item in values):
            raise T051ArtifactError(
                "TOKENIZER_OFFSET_INVALID",
                "each tokenizer offset must be an exact two-integer pair",
                evidence={"token_index": index},
            )
        start, end = values
        offsets.append((start, end))
    return tuple(offsets)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise T051ArtifactError(
            "JSONL_MISSING",
            "required release artifact is missing",
            evidence={"path": str(path)},
        )
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise T051ArtifactError(
                    "JSONL_BLANK_LINE",
                    "release JSONL cannot contain blank lines",
                    evidence={"path": str(path), "line": line_number},
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise T051ArtifactError(
                    "JSONL_INVALID",
                    "release JSONL contains invalid JSON",
                    evidence={"path": str(path), "line": line_number},
                ) from error
            if not isinstance(row, dict):
                raise T051ArtifactError(
                    "JSONL_OBJECT_REQUIRED",
                    "each release JSONL line must be an object",
                    evidence={"path": str(path), "line": line_number},
                )
            rows.append(row)
    return tuple(rows)


def _render_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rendered = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    _write_text_atomic(path, rendered)


def load_real_tokenizer(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> object:
    """Load only the approved local fast tokenizer; no network fallback."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise T051ArtifactError(
            "TRANSFORMERS_REQUIRED",
            "real ChemDFM-R tokenization requires Transformers",
        ) from error
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise T051ArtifactError(
            "CHECKPOINT_MISSING",
            "approved ChemDFM-R checkpoint directory is missing",
            evidence={"path": str(checkpoint)},
        )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    if getattr(tokenizer, "is_fast", None) is not True:
        raise T051ArtifactError(
            "FAST_TOKENIZER_REQUIRED",
            "approved ChemDFM-R tokenizer must expose reliable offsets",
        )
    return tokenizer


def build_tokenizer_fingerprint(
    tokenizer: object,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> dict[str, object]:
    """Describe the runtime tokenizer without computing a digest."""

    if getattr(tokenizer, "is_fast", None) is not True:
        raise T051ArtifactError(
            "FAST_TOKENIZER_REQUIRED",
            "tokenizer fingerprint requires a fast tokenizer",
        )
    return {
        "tokenizer_name": Path(checkpoint_path).name,
        "tokenizer_revision": "approved-local-checkpoint-no-digest",
        # Kept for backwards-compatible TokenLabelSet JSON shape.  The value
        # explicitly records that no vocabulary digest was computed.
        "tokenizer_vocab_hash": "not-computed-per-user-instruction",
        "special_token_config": {
            name: getattr(tokenizer, name, None)
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
        "normalization_config": {
            "offset_unit": "python_char",
            "production_weights_loaded": True,
            "fast_tokenizer": True,
            "tokenizer_class": type(tokenizer).__name__,
            "checkpoint_path": str(Path(checkpoint_path)),
            "digest_computation_performed": False,
            "truncation": False,
            "padding": False,
        },
    }


def _call_tokenizer(tokenizer: object, text: str) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
]:
    if getattr(tokenizer, "is_fast", None) is not True or not callable(tokenizer):
        raise T051ArtifactError(
            "FAST_TOKENIZER_REQUIRED",
            "real projection requires a callable fast tokenizer",
        )
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
            padding=False,
        )
    except Exception as error:
        raise T051ArtifactError(
            "TOKENIZATION_FAILED",
            "approved ChemDFM-R tokenizer failed",
            evidence={"exception_type": type(error).__name__},
        ) from error
    mapping = _require_mapping(encoded, "tokenizer output")
    required = {
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "special_tokens_mask",
    }
    missing = tuple(sorted(required - set(mapping)))
    if missing:
        raise T051ArtifactError(
            "TOKENIZER_OUTPUT_MISSING",
            "tokenizer omitted required fast-offset fields",
            evidence={"fields": missing},
        )
    input_ids = _exact_int_sequence(mapping["input_ids"], "input_ids")
    attention = _exact_int_sequence(mapping["attention_mask"], "attention_mask")
    offsets = _offset_sequence(mapping["offset_mapping"])
    special = _exact_int_sequence(
        mapping["special_tokens_mask"], "special_tokens_mask"
    )
    length = len(input_ids)
    if not length or any(
        len(values) != length for values in (attention, offsets, special)
    ):
        raise T051ArtifactError(
            "TOKENIZER_ARRAY_LENGTH_MISMATCH",
            "all real tokenizer arrays must be non-empty and exactly equal length",
        )
    if any(value < 0 for value in input_ids):
        raise T051ArtifactError(
            "TOKENIZER_ID_INVALID",
            "token IDs must be non-negative",
        )
    if any(value not in {0, 1} for value in attention + special):
        raise T051ArtifactError(
            "TOKENIZER_BINARY_ARRAY_INVALID",
            "attention and special-token masks must contain exact 0/1",
        )
    previous_end = 0
    for index, ((start, end), attended, is_special) in enumerate(
        zip(offsets, attention, special, strict=True)
    ):
        if start < 0 or end < start or end > len(text):
            raise T051ArtifactError(
                "TOKENIZER_OFFSET_OUT_OF_RANGE",
                "tokenizer offset is outside the Python string",
                evidence={"token_index": index},
            )
        if start == end and (start, end) != (0, 0):
            raise T051ArtifactError(
                "TOKENIZER_EMPTY_OFFSET_NONCANONICAL",
                "empty offsets must use the canonical zero pair",
                evidence={"token_index": index},
            )
        if not attended and (start, end) != (0, 0):
            raise T051ArtifactError(
                "TOKENIZER_PADDING_OFFSET_NONEMPTY",
                "unattended tokenizer entries must use empty offsets",
                evidence={"token_index": index},
            )
        if is_special and (start, end) != (0, 0):
            raise T051ArtifactError(
                "TOKENIZER_SPECIAL_OFFSET_NONEMPTY",
                "tokenizer-added special tokens must use empty offsets",
                evidence={"token_index": index},
            )
        if (start, end) == (0, 0):
            if attended and not is_special:
                raise T051ArtifactError(
                    "TOKENIZER_ZERO_OFFSET_NON_SPECIAL",
                    "attended zero-offset tokens must be marked special",
                    evidence={"token_index": index},
                )
            continue
        if start < previous_end:
            raise T051ArtifactError(
                "TOKENIZER_OFFSETS_NON_MONOTONIC",
                "non-empty tokenizer offsets must be disjoint and monotonic",
                evidence={"token_index": index},
            )
        previous_end = end
    return input_ids, attention, offsets, special


def _overlap(start: int, end: int, span: tuple[int, int]) -> int:
    return max(0, min(end, span[1]) - max(start, span[0]))


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _serialized_segments(
    serialized: Mapping[str, object],
    text: str,
) -> tuple[tuple[str, int, int], ...]:
    raw_segments = _require_sequence(serialized.get("segments"), "serialized.segments")
    segments: list[tuple[str, int, int]] = []
    allowed = {
        SegmentKind.SOURCE.value,
        SegmentKind.INSTRUCTION.value,
        SegmentKind.REASONING.value,
        SegmentKind.FINAL_ANSWER.value,
    }
    for index, raw in enumerate(raw_segments):
        segment = _require_mapping(raw, f"serialized.segments[{index}]")
        kind = segment.get("segment_kind")
        start = segment.get("start")
        end = segment.get("end")
        if kind not in allowed or type(start) is not int or type(end) is not int:
            raise T051ArtifactError(
                "SERIALIZED_SEGMENT_INVALID",
                "serialized segment has invalid kind or boundaries",
                evidence={"segment_index": index},
            )
        if start < 0 or end <= start or end > len(text):
            raise T051ArtifactError(
                "SERIALIZED_SEGMENT_OUT_OF_RANGE",
                "serialized content segment is outside detector text",
                evidence={"segment_index": index},
            )
        segments.append((str(kind), start, end))
    if len(segments) != 4 or len({kind for kind, _, _ in segments}) != 4:
        raise T051ArtifactError(
            "SERIALIZED_SEGMENT_SET_INVALID",
            "detector text must expose exactly four unique content segments",
        )
    return tuple(segments)


def _segment_ids(
    offsets: Sequence[tuple[int, int]],
    input_ids: Sequence[int],
    attention: Sequence[int],
    special: Sequence[int],
    segments: Sequence[tuple[str, int, int]],
    *,
    pad_token_id: object,
) -> tuple[str, ...]:
    output: list[str] = []
    for index, ((start, end), token_id, attended, is_special) in enumerate(
        zip(offsets, input_ids, attention, special, strict=True)
    ):
        if not attended or (
            (start, end) == (0, 0)
            and type(pad_token_id) is int
            and token_id == pad_token_id
        ):
            output.append(SegmentKind.PADDING.value)
            continue
        if is_special or (start, end) == (0, 0):
            output.append(SegmentKind.SPECIAL.value)
            continue
        overlapping = tuple(
            kind
            for kind, segment_start, segment_end in segments
            if max(0, min(end, segment_end) - max(start, segment_start)) > 0
        )
        if len(overlapping) > 1:
            raise T051ArtifactError(
                "TOKEN_CROSSES_CONTENT_SEGMENTS",
                "one real tokenizer token overlaps multiple detector fields",
                evidence={"token_index": index},
            )
        output.append(overlapping[0] if overlapping else SegmentKind.SPECIAL.value)
    return tuple(output)


def _span_pair(value: object, name: str, text_length: int) -> tuple[int, int]:
    pair = _require_sequence(value, name)
    if len(pair) != 2 or any(type(item) is not int for item in pair):
        raise T051ArtifactError(
            "CHAR_SPAN_INVALID",
            f"{name} must be an exact two-integer pair",
        )
    start, end = pair
    if start < 0 or end <= start or end > text_length:
        raise T051ArtifactError(
            "CHAR_SPAN_OUT_OF_RANGE",
            f"{name} is outside canonical detector text",
        )
    return start, end


def _validated_annotations(
    record: Mapping[str, object],
    text: str,
    segments: Sequence[tuple[str, int, int]],
) -> tuple[dict[str, object], ...]:
    raw_annotations = _require_sequence(record.get("spans"), "record.spans")
    variant = _require_mapping(record.get("variant"), "record.variant")
    label = variant.get("label")
    if label not in {"H", "N"}:
        raise T051ArtifactError(
            "VARIANT_LABEL_INVALID",
            "release variant label must be H or N",
        )
    if label == "H" and not raw_annotations:
        raise T051ArtifactError(
            "HALLUCINATED_SPAN_MISSING",
            "every hallucinated release record needs a localized char span",
            evidence={"record_id": record.get("record_id")},
        )
    if label == "N" and raw_annotations:
        raise T051ArtifactError(
            "FAITHFUL_CONTROL_HAS_SPANS",
            "faithful release controls must not carry positive char spans",
            evidence={"record_id": record.get("record_id")},
        )
    annotations: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_annotations):
        annotation = _require_mapping(raw, f"record.spans[{index}]")
        span_id = annotation.get("span_id")
        component = annotation.get("component")
        literal = _span_pair(
            annotation.get("literal_span"),
            f"record.spans[{index}].literal_span",
            len(text),
        )
        claim = _span_pair(
            annotation.get("claim_span"),
            f"record.spans[{index}].claim_span",
            len(text),
        )
        if type(span_id) is not str or not span_id or span_id in seen_ids:
            raise T051ArtifactError(
                "CHAR_SPAN_ID_INVALID",
                "char span IDs must be unique non-empty strings",
            )
        seen_ids.add(span_id)
        if component not in _EVALUATED_SEGMENTS:
            raise T051ArtifactError(
                "CHAR_SPAN_COMPONENT_INVALID",
                "positive char spans must belong to reasoning or final answer",
                evidence={"span_id": span_id},
            )
        if not (claim[0] <= literal[0] and literal[1] <= claim[1]):
            raise T051ArtifactError(
                "CHAR_LITERAL_OUTSIDE_CLAIM",
                "literal span must be contained by claim span",
                evidence={"span_id": span_id},
            )
        containing = tuple(
            kind
            for kind, start, end in segments
            if start <= literal[0] and literal[1] <= end
        )
        claim_containing = tuple(
            kind
            for kind, start, end in segments
            if start <= claim[0] and claim[1] <= end
        )
        if containing != (component,) or claim_containing != (component,):
            raise T051ArtifactError(
                "CHAR_SPAN_SEGMENT_MISMATCH",
                "char span coordinates disagree with serialized detector segments",
                evidence={"span_id": span_id},
            )
        semantic = tuple(
            _exact_int_sequence(annotation.get("semantic_types"), "semantic_types")
        )
        edit = tuple(_require_sequence(annotation.get("edit_subtypes"), "edit_subtypes"))
        role = annotation.get("causal_role")
        valid_semantic = {int(item.value) for item in HallucinationType}
        valid_edit = {item.value for item in EditErrorSubtype}
        valid_roles = {item.value for item in CausalRole}
        if not semantic or any(item not in valid_semantic for item in semantic):
            raise T051ArtifactError(
                "SEMANTIC_LABEL_INVALID",
                "char span has an invalid semantic label",
                evidence={"span_id": span_id},
            )
        if not edit or any(type(item) is not str or item not in valid_edit for item in edit):
            raise T051ArtifactError(
                "EDIT_LABEL_INVALID",
                "char span has an invalid edit subtype",
                evidence={"span_id": span_id},
            )
        if role not in valid_roles:
            raise T051ArtifactError(
                "CAUSAL_ROLE_INVALID",
                "char span has an invalid causal role",
                evidence={"span_id": span_id},
            )
        annotations.append(
            {
                "span_id": span_id,
                "component": component,
                "literal": literal,
                "claim": claim,
                "semantic": semantic,
                "edit": edit,
                "role": role,
            }
        )
    annotations.sort(
        key=lambda item: (
            item["literal"][0],
            item["literal"][1],
            item["span_id"],
        )
    )
    for previous, current in pairwise(annotations):
        previous_literal = previous["literal"]
        current_literal = current["literal"]
        if previous_literal[1] > current_literal[0]:
            raise T051ArtifactError(
                "CHAR_SPANS_OVERLAP",
                "canonical positive literal spans cannot overlap",
            )
    return tuple(annotations)


def project_record_with_real_tokenizer(
    record: Mapping[str, object],
    prior_token_row: Mapping[str, object],
    tokenizer: object,
    tokenizer_fingerprint: Mapping[str, object],
) -> dict[str, object]:
    """Reproject one canonical release record with the real fast tokenizer."""

    record_id = record.get("record_id")
    if type(record_id) is not str or not record_id:
        raise T051ArtifactError(
            "RECORD_ID_INVALID",
            "release record_id must be non-empty text",
        )
    if prior_token_row.get("record_id") != record_id:
        raise T051ArtifactError(
            "PRIOR_TOKEN_ID_MISMATCH",
            "prior token row must identify the exact same record",
            evidence={"record_id": record_id},
        )
    serialized = _require_mapping(record.get("serialized"), "record.serialized")
    text = serialized.get("text")
    carried_identity = serialized.get("sha256")
    if type(text) is not str or not text:
        raise T051ArtifactError(
            "SERIALIZED_TEXT_INVALID",
            "canonical detector text must be non-empty text",
            evidence={"record_id": record_id},
        )
    if type(carried_identity) is not str or not carried_identity:
        raise T051ArtifactError(
            "CARRIED_TEXT_IDENTITY_MISSING",
            "record must contain the already-frozen text identity",
            evidence={"record_id": record_id},
        )
    segments = _serialized_segments(serialized, text)
    annotations = _validated_annotations(record, text, segments)
    input_ids, attention, offsets, special = _call_tokenizer(tokenizer, text)
    segment_ids = _segment_ids(
        offsets,
        input_ids,
        attention,
        special,
        segments,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    token_count = len(input_ids)
    semantic_sets = [set[int]() for _ in range(token_count)]
    edit_sets = [set[str]() for _ in range(token_count)]
    role_sets = [set[str]() for _ in range(token_count)]
    intersections: list[list[tuple[int, int]]] = [
        [] for _ in range(token_count)
    ]
    boundary = [0] * token_count
    covered: set[str] = set()
    for annotation in annotations:
        literal = annotation["literal"]
        for token_index, (start, end) in enumerate(offsets):
            overlap = _overlap(start, end, literal)
            if overlap <= 0 or segment_ids[token_index] not in _EVALUATED_SEGMENTS:
                continue
            covered.add(str(annotation["span_id"]))
            semantic_sets[token_index].update(annotation["semantic"])
            edit_sets[token_index].update(annotation["edit"])
            role_sets[token_index].add(str(annotation["role"]))
            intersections[token_index].append(
                (max(start, literal[0]), min(end, literal[1]))
            )
            if start < literal[0] < end or start < literal[1] < end:
                boundary[token_index] = 1
    missing = tuple(
        annotation["span_id"]
        for annotation in annotations
        if annotation["span_id"] not in covered
    )
    if missing:
        raise T051ArtifactError(
            "POSITIVE_SPAN_UNCOVERED",
            "every positive char span must overlap a real evaluated token",
            evidence={"record_id": record_id, "span_ids": missing},
        )
    for token_index, roles in enumerate(role_sets):
        if len(roles) > 1:
            raise T051ArtifactError(
                "CAUSAL_ROLE_TOKEN_COLLISION",
                "one real tokenizer token cannot represent multiple causal roles",
                evidence={"record_id": record_id, "token_index": token_index},
            )
        if int(HallucinationType.UNVERIFIABLE.value) in semantic_sets[token_index] and len(
            semantic_sets[token_index]
        ) > 1:
            raise T051ArtifactError(
                "UNVERIFIABLE_TOKEN_COLLISION",
                "UNVERIFIABLE cannot share a real token with adjudicated semantics",
                evidence={"record_id": record_id, "token_index": token_index},
            )
    semantic_masks = {
        str(int(label.value)): [
            int(int(label.value) in values) for values in semantic_sets
        ]
        for label in HallucinationType
    }
    edit_masks = {
        label.value: [int(label.value in values) for values in edit_sets]
        for label in EditErrorSubtype
    }
    role_masks = {
        label.value: [int(label.value in values) for values in role_sets]
        for label in CausalRole
    }
    hallucination_core = [
        int(
            int(HallucinationType.CONTRADICTION.value) in values
            or int(HallucinationType.UNSUPPORTED.value) in values
        )
        for values in semantic_sets
    ]
    error_any = [
        int(
            any(
                value != int(HallucinationType.UNVERIFIABLE.value)
                for value in values
            )
        )
        for values in semantic_sets
    ]
    local_falsehood = [
        int(bool(values & _LOCAL_FALSEHOOD_ROLES)) for values in role_sets
    ]
    off_task = [
        int(CausalRole.PROPAGATED_CONDITIONAL.value in values)
        for values in role_sets
    ]
    fractions = [
        0.0
        if end == start
        else _union_length(intersections[index]) / (end - start)
        for index, (start, end) in enumerate(offsets)
    ]
    reasoning = [int(value == SegmentKind.REASONING.value) for value in segment_ids]
    answer = [int(value == SegmentKind.FINAL_ANSWER.value) for value in segment_ids]
    evaluation = [
        int(segment in _EVALUATED_SEGMENTS and attention[index] == 1)
        for index, segment in enumerate(segment_ids)
    ]
    output = {
        key: value
        for key, value in prior_token_row.items()
        if key not in _REPLACED_TOKEN_FIELDS
    }
    output.update(
        {
            "record_id": record_id,
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "tokenizer_fingerprint": dict(tokenizer_fingerprint),
            # This value is inherited from the canonical record.  It is not
            # recomputed or compared by T051.
            "serialized_text_sha256": carried_identity,
            "input_ids": list(input_ids),
            "attention_mask": list(attention),
            "offset_mapping": [list(pair) for pair in offsets],
            "segment_ids": list(segment_ids),
            "evaluation_mask": evaluation,
            "hallucination_core_mask": hallucination_core,
            "error_any_mask": error_any,
            "semantic_type_masks": semantic_masks,
            "edit_subtype_masks": edit_masks,
            "causal_role_masks": role_masks,
            "local_falsehood_mask": local_falsehood,
            "off_task_branch_mask": off_task,
            "reasoning_mask": reasoning,
            "answer_mask": answer,
            "boundary_ambiguous_mask": boundary,
            "error_char_fraction": fractions,
        }
    )
    validate_tokenized_row(output)
    return output


def validate_tokenized_row(
    row: Mapping[str, object],
    *,
    expected_split: str | None = None,
) -> int:
    """Validate exact post-token arrays and return their shared token count."""

    row_split = row.get("split")
    if type(row_split) is not str or not row_split:
        raise T051ArtifactError(
            "TOKENIZED_ROW_SPLIT_INVALID",
            "tokenized row split must be non-empty text",
            evidence={"record_id": row.get("record_id")},
        )
    if expected_split is not None and row_split != expected_split:
        raise T051ArtifactError(
            "TOKENIZED_ROW_SPLIT_MISMATCH",
            "tokenized row split differs from its containing artifact",
            evidence={
                "record_id": row.get("record_id"),
                "expected_split": expected_split,
                "actual_split": row_split,
            },
        )
    if row.get("activation_alignment") != ACTIVATION_ALIGNMENT:
        raise T051ArtifactError(
            "ACTIVATION_ALIGNMENT_INVALID",
            "T051 supports only exact post_token_h_t alignment",
            evidence={"record_id": row.get("record_id")},
        )
    input_ids = _exact_int_sequence(row.get("input_ids"), "input_ids")
    token_count = len(input_ids)
    if not token_count:
        raise T051ArtifactError(
            "EMPTY_TOKEN_SEQUENCE",
            "tokenized release rows cannot be empty",
        )
    for name in _DIRECT_TOKEN_ARRAY_FIELDS:
        values = _require_sequence(row.get(name), name)
        if len(values) != token_count:
            raise T051ArtifactError(
                "TOKEN_ARRAY_LENGTH_MISMATCH",
                "all direct token arrays must match input_ids exactly",
                evidence={
                    "record_id": row.get("record_id"),
                    "field": name,
                    "expected": token_count,
                    "actual": len(values),
                },
            )
    for field in _NESTED_TOKEN_ARRAY_FIELDS:
        masks = _require_mapping(row.get(field), field)
        if not masks:
            raise T051ArtifactError(
                "TOKEN_MASK_AXIS_EMPTY",
                "multi-axis token masks cannot be empty",
                evidence={"field": field},
            )
        for label, mask in masks.items():
            values = _require_sequence(mask, f"{field}[{label}]")
            if len(values) != token_count:
                raise T051ArtifactError(
                    "TOKEN_ARRAY_LENGTH_MISMATCH",
                    "all nested token arrays must match input_ids exactly",
                    evidence={
                        "record_id": row.get("record_id"),
                        "field": f"{field}[{label}]",
                        "expected": token_count,
                        "actual": len(values),
                    },
                )
            if any(type(value) is not int or value not in {0, 1} for value in values):
                raise T051ArtifactError(
                    "TOKEN_MASK_NONBINARY",
                    "token masks must contain exact integer 0/1",
                    evidence={"field": f"{field}[{label}]"},
                )
    for name in (
        "attention_mask",
        "evaluation_mask",
        "hallucination_core_mask",
        "error_any_mask",
        "local_falsehood_mask",
        "off_task_branch_mask",
        "reasoning_mask",
        "answer_mask",
        "boundary_ambiguous_mask",
    ):
        if any(
            type(value) is not int or value not in {0, 1}
            for value in _require_sequence(row[name], name)
        ):
            raise T051ArtifactError(
                "TOKEN_MASK_NONBINARY",
                "token masks must contain exact integer 0/1",
                evidence={"field": name},
            )
    fingerprint = _require_mapping(
        row.get("tokenizer_fingerprint"), "tokenizer_fingerprint"
    )
    normalization = _require_mapping(
        fingerprint.get("normalization_config"),
        "tokenizer_fingerprint.normalization_config",
    )
    if normalization.get("fast_tokenizer") is not True:
        raise T051ArtifactError(
            "REAL_FAST_TOKENIZER_NOT_RECORDED",
            "T051 fingerprint must record the real fast tokenizer",
        )
    if normalization.get("digest_computation_performed") is not False:
        raise T051ArtifactError(
            "DIGEST_POLICY_INVALID",
            "T051 fingerprint must record that digest computation was skipped",
        )
    variant = row.get("record_id")
    if type(variant) is not str or not variant:
        raise T051ArtifactError(
            "RECORD_ID_INVALID",
            "tokenized row record_id must be non-empty text",
        )
    return token_count


def tokenize_records(
    records: Sequence[Mapping[str, object]],
    prior_token_rows: Sequence[Mapping[str, object]],
    tokenizer: object,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    split: str,
) -> tuple[tuple[dict[str, object], ...], TokenizationSummary]:
    """Tokenize an exact record set after joining prior metadata by record_id."""

    if type(split) is not str or not split:
        raise TypeError("split must be non-empty text")
    prior_by_id: dict[str, Mapping[str, object]] = {}
    for row in prior_token_rows:
        record_id = row.get("record_id")
        if type(record_id) is not str or not record_id or record_id in prior_by_id:
            raise T051ArtifactError(
                "PRIOR_TOKEN_ID_SET_INVALID",
                "prior token rows need unique non-empty record IDs",
            )
        prior_by_id[record_id] = row
    record_ids = tuple(record.get("record_id") for record in records)
    if any(type(record_id) is not str or not record_id for record_id in record_ids):
        raise T051ArtifactError(
            "RECORD_ID_SET_INVALID",
            "release records need non-empty record IDs",
        )
    if len(record_ids) != len(set(record_ids)):
        raise T051ArtifactError(
            "RECORD_ID_SET_INVALID",
            "release record IDs must be unique",
        )
    if set(record_ids) != set(prior_by_id):
        raise T051ArtifactError(
            "PRIOR_TOKEN_EXACT_SET_MISMATCH",
            "prior token artifacts must have the exact release record ID set",
            evidence={
                "record_count": len(record_ids),
                "prior_count": len(prior_by_id),
            },
        )
    fingerprint = build_tokenizer_fingerprint(tokenizer, checkpoint_path)
    output = tuple(
        project_record_with_real_tokenizer(
            record,
            prior_by_id[str(record["record_id"])],
            tokenizer,
            fingerprint,
        )
        for record in records
    )
    token_counts = tuple(
        validate_tokenized_row(row, expected_split=split) for row in output
    )
    span_count = sum(
        len(_require_sequence(record.get("spans"), "record.spans"))
        for record in records
    )
    return output, TokenizationSummary(
        split=split,
        record_count=len(output),
        token_count=sum(token_counts),
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        positive_span_count=span_count,
        carried_text_identity_count=len(output),
    )


def tokenize_release(
    release_root: Path = DEFAULT_RELEASE_ROOT,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    tokenizer: object | None = None,
    splits: Sequence[str] = ("train", "validation", "test"),
    expected_counts: Mapping[str, int] | None = DEFAULT_SPLIT_COUNTS,
    output_root: Path | None = None,
    limit_per_split: int | None = None,
    progress: bool = False,
) -> dict[str, object]:
    """Rebuild requested release token files and a real-tokenizer manifest."""

    root = Path(release_root)
    target_root = (
        root / "tokenized/chemdfm_r"
        if output_root is None
        else Path(output_root)
    )
    if limit_per_split is not None and output_root is None:
        raise T051ArtifactError(
            "SMOKE_OUTPUT_ROOT_REQUIRED",
            "limited tokenization must use a separate output root",
        )
    selected_splits = tuple(splits)
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise T051ArtifactError(
            "SPLIT_SELECTION_INVALID",
            "tokenization splits must be non-empty and unique",
        )
    if limit_per_split is not None and (
        type(limit_per_split) is not int or limit_per_split <= 0
    ):
        raise ValueError("limit_per_split must be a positive integer or None")
    runtime_tokenizer = (
        load_real_tokenizer(checkpoint_path) if tokenizer is None else tokenizer
    )
    results: dict[str, tuple[dict[str, object], ...]] = {}
    summaries: list[TokenizationSummary] = []
    for split in selected_splits:
        records = _read_jsonl(root / "records" / f"{split}.jsonl")
        prior = _read_jsonl(root / "tokenized/chemdfm_r" / f"{split}.jsonl")
        if expected_counts is not None and limit_per_split is None:
            expected = expected_counts.get(split)
            if expected is None or len(records) != expected or len(prior) != expected:
                raise T051ArtifactError(
                    "SPLIT_RECORD_COUNT_MISMATCH",
                    "release and prior-token counts must match the frozen split",
                    evidence={
                        "split": split,
                        "expected": expected,
                        "records": len(records),
                        "prior": len(prior),
                    },
                )
        if limit_per_split is not None:
            selected_ids = {
                str(row["record_id"]) for row in records[:limit_per_split]
            }
            records = records[:limit_per_split]
            prior = tuple(
                row for row in prior if row.get("record_id") in selected_ids
            )
        tokenized, summary = tokenize_records(
            records,
            prior,
            runtime_tokenizer,
            checkpoint_path=checkpoint_path,
            split=split,
        )
        results[split] = tokenized
        summaries.append(summary)
        if progress:
            print(
                json.dumps(
                    {
                        "event": "tokenization_split_complete",
                        **summary.to_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    for split, rows in results.items():
        _write_jsonl_atomic(target_root / f"{split}.jsonl", rows)
    manifest = {
        "format_version": T051_TOKENIZATION_FORMAT_VERSION,
        "status": "complete",
        "mode": "smoke" if limit_per_split is not None else "release",
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "checkpoint_path": str(Path(checkpoint_path)),
        "tokenizer_fingerprint": build_tokenizer_fingerprint(
            runtime_tokenizer, checkpoint_path
        ),
        "identity_handling": {
            "source": "records[*].serialized.sha256",
            "carried_forward": True,
            "recomputed": False,
            "verified_by_digest": False,
        },
        "splits": [summary.to_dict() for summary in summaries],
        "record_count": sum(summary.record_count for summary in summaries),
        "token_count": sum(summary.token_count for summary in summaries),
        "all_token_arrays_equal_length": True,
    }
    _write_text_atomic(target_root / "manifest.json", _render_json(manifest))
    return manifest


def _canonical_jsonl_lines(path: Path) -> tuple[bytes, ...]:
    if not path.is_file():
        raise T051ArtifactError(
            "CANONICAL_TOKENIZED_MISSING",
            "canonical real-tokenized JSONL is missing",
            evidence={"path": str(path)},
        )
    with path.open("rb") as handle:
        lines = tuple(handle)
    if not lines:
        raise T051ArtifactError(
            "CANONICAL_TOKENIZED_EMPTY",
            "canonical real-tokenized JSONL cannot be empty",
            evidence={"path": str(path)},
        )
    for line_number, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or not line.strip():
            raise T051ArtifactError(
                "CANONICAL_TOKENIZED_LINE_INVALID",
                "canonical JSONL lines must be non-empty and newline terminated",
                evidence={"path": str(path), "line": line_number},
            )
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise T051ArtifactError(
                "CANONICAL_TOKENIZED_LINE_INVALID",
                "canonical JSONL line is not valid UTF-8 JSON",
                evidence={"path": str(path), "line": line_number},
            ) from error
        if not isinstance(row, dict):
            raise T051ArtifactError(
                "CANONICAL_TOKENIZED_LINE_INVALID",
                "canonical JSONL lines must contain objects",
                evidence={"path": str(path), "line": line_number},
            )
        validate_tokenized_row(row)
    return lines


def _partition_jsonl_lines(
    lines: Sequence[bytes],
    *,
    max_shard_bytes: int,
) -> tuple[tuple[bytes, ...], ...]:
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    partitions: list[tuple[bytes, ...]] = []
    current: list[bytes] = []
    current_bytes = 0
    for line_number, line in enumerate(lines, start=1):
        if len(line) >= max_shard_bytes:
            raise T051ArtifactError(
                "TOKENIZED_ROW_EXCEEDS_SHARD_LIMIT",
                "one canonical JSONL row cannot fit below the shard byte limit",
                evidence={
                    "line": line_number,
                    "line_bytes": len(line),
                    "max_shard_bytes": max_shard_bytes,
                },
            )
        if current and current_bytes + len(line) >= max_shard_bytes:
            partitions.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += len(line)
    if current:
        partitions.append(tuple(current))
    return tuple(partitions)


def write_git_tokenized_shards(
    canonical_root: Path,
    *,
    shard_root: Path | None = None,
    splits: Sequence[str] = ("train", "validation", "test"),
    max_shard_bytes: int = GIT_SHARD_MAX_BYTES,
) -> dict[str, object]:
    """Write byte-bounded JSONL shards that reconstruct canonical files exactly.

    Partitioning uses only canonical line order and encoded byte length.  The
    index intentionally contains no digest and records that policy explicitly.
    """

    root = Path(canonical_root)
    target_root = root / "git_shards" if shard_root is None else Path(shard_root)
    selected_splits = tuple(splits)
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise T051ArtifactError(
            "SPLIT_SELECTION_INVALID",
            "git shard splits must be non-empty and unique",
        )
    split_index: dict[str, object] = {}
    expected_paths: set[Path] = set()
    for split in selected_splits:
        canonical_path = root / f"{split}.jsonl"
        lines = _canonical_jsonl_lines(canonical_path)
        partitions = _partition_jsonl_lines(
            lines,
            max_shard_bytes=max_shard_bytes,
        )
        shard_entries: list[dict[str, object]] = []
        split_record_ids: list[str] = []
        token_count = 0
        for order, partition in enumerate(partitions):
            relative_path = Path(split) / f"part-{order:05d}.jsonl"
            output_path = target_root / relative_path
            expected_paths.add(output_path)
            payload = b"".join(partition)
            if len(payload) >= max_shard_bytes:
                raise T051ArtifactError(
                    "GIT_SHARD_BYTE_LIMIT_EXCEEDED",
                    "partitioned tokenized shard reached the exclusive byte limit",
                    evidence={"path": str(output_path), "bytes": len(payload)},
                )
            parsed = tuple(json.loads(line) for line in partition)
            record_ids = tuple(str(row["record_id"]) for row in parsed)
            if len(record_ids) != len(set(record_ids)):
                raise T051ArtifactError(
                    "GIT_SHARD_RECORD_ID_DUPLICATE",
                    "one tokenized git shard contains duplicate record IDs",
                    evidence={"path": str(output_path)},
                )
            split_record_ids.extend(record_ids)
            token_count += sum(
                validate_tokenized_row(row, expected_split=split) for row in parsed
            )
            _write_bytes_atomic(output_path, payload)
            shard_entries.append(
                {
                    "order": order,
                    "path": relative_path.as_posix(),
                    "bytes": len(payload),
                    "row_count": len(parsed),
                    "first_record_id": record_ids[0],
                    "last_record_id": record_ids[-1],
                }
            )
        if len(split_record_ids) != len(set(split_record_ids)):
            raise T051ArtifactError(
                "GIT_SHARD_SPLIT_ID_DUPLICATE",
                "tokenized git shards contain duplicate IDs across one split",
                evidence={"split": split},
            )
        split_index[split] = {
            "canonical_relative_path": f"{split}.jsonl",
            "canonical_bytes": canonical_path.stat().st_size,
            "record_count": len(lines),
            "token_count": token_count,
            "shard_count": len(shard_entries),
            "shards": shard_entries,
        }
    if target_root.is_dir():
        unexpected = tuple(
            sorted(
                str(path)
                for path in target_root.glob("*/part-*.jsonl")
                if path not in expected_paths
            )
        )
        if unexpected:
            raise T051ArtifactError(
                "STALE_GIT_SHARDS_PRESENT",
                "unreferenced tokenized shards must be resolved before republishing",
                evidence={"paths": unexpected},
            )
    index = {
        "format_version": T051_GIT_SHARD_INDEX_VERSION,
        "status": "complete",
        "canonical_storage": "server_only",
        "canonical_files_retained": True,
        "reconstruction": "byte_concatenation_in_index_order",
        "max_shard_bytes_exclusive": max_shard_bytes,
        "digest_computation_performed": False,
        "split_order": list(selected_splits),
        "splits": split_index,
        "record_count": sum(
            int(value["record_count"]) for value in split_index.values()
        ),
        "token_count": sum(
            int(value["token_count"]) for value in split_index.values()
        ),
    }
    index_path = target_root / "index.json"
    _write_text_atomic(index_path, _render_json(index))
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise T051ArtifactError(
                "TOKENIZATION_MANIFEST_INVALID",
                "real-tokenization manifest is invalid JSON",
            ) from error
        if not isinstance(manifest, dict):
            raise T051ArtifactError(
                "TOKENIZATION_MANIFEST_INVALID",
                "real-tokenization manifest must be an object",
            )
        manifest["git_shards"] = {
            "index_path": "git_shards/index.json",
            "max_shard_bytes_exclusive": max_shard_bytes,
            "canonical_storage": "server_only",
            "digest_computation_performed": False,
        }
        _write_text_atomic(manifest_path, _render_json(manifest))
    return index


def _load_git_shard_index(index_path: Path) -> tuple[Path, dict[str, object]]:
    path = Path(index_path)
    if not path.is_file():
        raise T051ArtifactError(
            "GIT_SHARD_INDEX_MISSING",
            "tokenized git shard index is missing",
            evidence={"path": str(path)},
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise T051ArtifactError(
            "GIT_SHARD_INDEX_INVALID",
            "tokenized git shard index is invalid JSON",
        ) from error
    if not isinstance(value, dict) or value.get("format_version") != (
        T051_GIT_SHARD_INDEX_VERSION
    ):
        raise T051ArtifactError(
            "GIT_SHARD_INDEX_INVALID",
            "tokenized git shard index has an unsupported format",
        )
    if value.get("digest_computation_performed") is not False:
        raise T051ArtifactError(
            "GIT_SHARD_INDEX_POLICY_INVALID",
            "tokenized git shard index must record the no-digest policy",
        )
    return path.parent, value


def iter_git_tokenized_rows(
    index_path: Path,
    split: str,
) -> tuple[dict[str, object], ...]:
    """Read one split strictly in indexed shard order and validate inventory."""

    shard_root, index = _load_git_shard_index(index_path)
    splits = _require_mapping(index.get("splits"), "git shard index splits")
    if split not in splits:
        raise T051ArtifactError(
            "GIT_SHARD_SPLIT_MISSING",
            "requested split is absent from tokenized git shard index",
            evidence={"split": split},
        )
    split_info = _require_mapping(splits[split], f"git shard split {split}")
    raw_shards = _require_sequence(split_info.get("shards"), "git shard entries")
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    total_bytes = 0
    max_bytes = index.get("max_shard_bytes_exclusive")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise T051ArtifactError(
            "GIT_SHARD_INDEX_INVALID",
            "git shard index byte limit must be a positive integer",
        )
    for expected_order, raw_entry in enumerate(raw_shards):
        entry = _require_mapping(raw_entry, "git shard entry")
        if entry.get("order") != expected_order:
            raise T051ArtifactError(
                "GIT_SHARD_ORDER_INVALID",
                "git shard orders must be contiguous from zero",
                evidence={"split": split, "expected_order": expected_order},
            )
        relative = entry.get("path")
        if type(relative) is not str or not relative:
            raise T051ArtifactError(
                "GIT_SHARD_PATH_INVALID",
                "git shard path must be non-empty text",
            )
        shard_path = shard_root / relative
        if not shard_path.is_file():
            raise T051ArtifactError(
                "GIT_SHARD_MISSING",
                "indexed tokenized git shard is missing",
                evidence={"path": str(shard_path)},
            )
        file_bytes = shard_path.stat().st_size
        if (
            entry.get("bytes") != file_bytes
            or file_bytes <= 0
            or file_bytes >= max_bytes
        ):
            raise T051ArtifactError(
                "GIT_SHARD_SIZE_INVALID",
                "tokenized git shard size differs from its bounded inventory",
                evidence={"path": str(shard_path), "bytes": file_bytes},
            )
        total_bytes += file_bytes
        shard_rows = _read_jsonl(shard_path)
        if entry.get("row_count") != len(shard_rows):
            raise T051ArtifactError(
                "GIT_SHARD_ROW_COUNT_MISMATCH",
                "tokenized git shard row count differs from its index",
                evidence={"path": str(shard_path)},
            )
        record_ids = tuple(str(row.get("record_id")) for row in shard_rows)
        if (
            not record_ids
            or entry.get("first_record_id") != record_ids[0]
            or entry.get("last_record_id") != record_ids[-1]
        ):
            raise T051ArtifactError(
                "GIT_SHARD_BOUNDARY_ID_MISMATCH",
                "tokenized git shard boundary IDs differ from its index",
                evidence={"path": str(shard_path)},
            )
        for row, record_id in zip(shard_rows, record_ids, strict=True):
            if record_id in seen_ids:
                raise T051ArtifactError(
                    "GIT_SHARD_RECORD_ID_DUPLICATE",
                    "tokenized git shard inventory contains duplicate record IDs",
                    evidence={"record_id": record_id},
                )
            seen_ids.add(record_id)
            validate_tokenized_row(row, expected_split=split)
            rows.append(row)
    if split_info.get("record_count") != len(rows):
        raise T051ArtifactError(
            "GIT_SHARD_SPLIT_COUNT_MISMATCH",
            "tokenized git shard split count differs from its index",
            evidence={"split": split},
        )
    if split_info.get("canonical_bytes") != total_bytes:
        raise T051ArtifactError(
            "GIT_SHARD_CANONICAL_SIZE_MISMATCH",
            "indexed shards do not sum to the canonical JSONL byte count",
            evidence={"split": split},
        )
    return tuple(rows)


def validate_git_shard_inventory(index_path: Path) -> dict[str, object]:
    """Validate every indexed shard and return exact non-digest counts."""

    _, index = _load_git_shard_index(index_path)
    split_order = tuple(
        _require_sequence(index.get("split_order"), "git shard split_order")
    )
    summaries: dict[str, object] = {}
    record_ids: set[str] = set()
    total_tokens = 0
    for split_value in split_order:
        if type(split_value) is not str or not split_value:
            raise T051ArtifactError(
                "GIT_SHARD_SPLIT_ORDER_INVALID",
                "git shard split_order must contain non-empty strings",
            )
        rows = iter_git_tokenized_rows(index_path, split_value)
        ids = tuple(str(row["record_id"]) for row in rows)
        overlap = record_ids.intersection(ids)
        if overlap:
            raise T051ArtifactError(
                "GIT_SHARD_CROSS_SPLIT_ID_DUPLICATE",
                "tokenized git shard IDs must be unique across splits",
                evidence={"record_ids": tuple(sorted(overlap))},
            )
        record_ids.update(ids)
        tokens = sum(
            validate_tokenized_row(row, expected_split=split_value) for row in rows
        )
        split_info = _require_mapping(
            _require_mapping(index.get("splits"), "git shard index splits").get(
                split_value
            ),
            f"git shard split {split_value}",
        )
        if split_info.get("token_count") != tokens:
            raise T051ArtifactError(
                "GIT_SHARD_SPLIT_TOKEN_COUNT_MISMATCH",
                "tokenized git shard split token count differs from its index",
                evidence={"split": split_value},
            )
        total_tokens += tokens
        summaries[split_value] = {
            "record_count": len(rows),
            "token_count": tokens,
        }
    if index.get("record_count") != len(record_ids) or index.get(
        "token_count"
    ) != total_tokens:
        raise T051ArtifactError(
            "GIT_SHARD_GLOBAL_COUNT_MISMATCH",
            "tokenized git shard global counts differ from the index",
        )
    return {
        "format_version": T051_GIT_SHARD_INDEX_VERSION,
        "all_pass": True,
        "digest_verification_performed": False,
        "record_count": len(record_ids),
        "token_count": total_tokens,
        "splits": summaries,
    }


def reassemble_git_tokenized_split(
    index_path: Path,
    split: str,
    output_path: Path,
) -> dict[str, object]:
    """Byte-concatenate indexed shards into an explicit output path."""

    shard_root, index = _load_git_shard_index(index_path)
    splits = _require_mapping(index.get("splits"), "git shard index splits")
    if split not in splits:
        raise T051ArtifactError(
            "GIT_SHARD_SPLIT_MISSING",
            "requested split is absent from tokenized git shard index",
        )
    split_info = _require_mapping(splits[split], f"git shard split {split}")
    entries = _require_sequence(split_info.get("shards"), "git shard entries")
    parts: list[bytes] = []
    for expected_order, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, "git shard entry")
        if entry.get("order") != expected_order:
            raise T051ArtifactError(
                "GIT_SHARD_ORDER_INVALID",
                "git shard orders must be contiguous from zero",
            )
        relative = entry.get("path")
        if type(relative) is not str:
            raise T051ArtifactError(
                "GIT_SHARD_PATH_INVALID",
                "git shard path must be text",
            )
        parts.append((shard_root / relative).read_bytes())
    payload = b"".join(parts)
    if len(payload) != split_info.get("canonical_bytes"):
        raise T051ArtifactError(
            "GIT_SHARD_REASSEMBLY_SIZE_MISMATCH",
            "reassembled JSONL byte count differs from the index",
        )
    target = Path(output_path)
    _write_bytes_atomic(target, payload)
    rows = _read_jsonl(target)
    if len(rows) != split_info.get("record_count"):
        raise T051ArtifactError(
            "GIT_SHARD_REASSEMBLY_ROW_MISMATCH",
            "reassembled JSONL row count differs from the index",
        )
    for row in rows:
        validate_tokenized_row(row, expected_split=split)
    return {
        "split": split,
        "output_path": str(target),
        "bytes": len(payload),
        "record_count": len(rows),
        "digest_verification_performed": False,
    }


def assert_post_token_axis(
    token_row: Mapping[str, object],
    hidden_shape: Sequence[int],
    *,
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE,
) -> int:
    """Assert ``[batch=1, token=T, hidden]`` against labels at the same T."""

    token_count = validate_tokenized_row(token_row)
    shape = tuple(hidden_shape)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != expected_hidden_size:
        raise T051ArtifactError(
            "ACTIVATION_SHAPE_INVALID",
            "resid_post must have shape [1, tokens, hidden_size]",
            evidence={"shape": shape, "expected_hidden_size": expected_hidden_size},
        )
    if shape[1] != token_count:
        raise T051ArtifactError(
            "HIDDEN_LABEL_TOKEN_AXIS_MISMATCH",
            "resid_post token axis must equal labels exactly with no shift",
            evidence={
                "record_id": token_row.get("record_id"),
                "hidden_tokens": shape[1],
                "label_tokens": token_count,
                "label_shift": 0,
            },
        )
    return token_count


def plan_activation_shards(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    shard_size: int = 8,
) -> tuple[ActivationShardPlan, ...]:
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    plans: list[ActivationShardPlan] = []
    for split, rows in rows_by_split.items():
        if type(split) is not str or not split:
            raise TypeError("activation split names must be non-empty text")
        for row in rows:
            validate_tokenized_row(row, expected_split=split)
        for start in range(0, len(rows), shard_size):
            plans.append(
                ActivationShardPlan(
                    split=split,
                    shard_index=start // shard_size,
                    rows=tuple(rows[start : start + shard_size]),
                )
            )
    return tuple(plans)


def _load_checkpoint_config(checkpoint_path: Path) -> dict[str, object]:
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise T051ArtifactError(
            "MODEL_CONFIG_MISSING",
            "ChemDFM-R config.json is missing",
            evidence={"path": str(config_path)},
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise T051ArtifactError(
            "MODEL_CONFIG_INVALID",
            "ChemDFM-R config.json is invalid",
        ) from error
    if not isinstance(config, dict):
        raise T051ArtifactError(
            "MODEL_CONFIG_INVALID",
            "ChemDFM-R config.json must be an object",
        )
    expected = {
        "model_type": EXPECTED_MODEL_TYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_LAYER_COUNT,
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or EXPECTED_ARCHITECTURE not in architectures
    ):
        drift["architectures"] = {
            "expected": EXPECTED_ARCHITECTURE,
            "actual": architectures,
        }
    if drift:
        raise T051ArtifactError(
            "MODEL_CONFIG_DRIFT",
            "approved ChemDFM-R checkpoint config does not match the frozen Qwen2 shape",
            evidence=drift,
        )
    return config


def _load_runtime_model(checkpoint_path: Path, device: str) -> tuple[object, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise T051ArtifactError(
            "ACTIVATION_RUNTIME_REQUIRED",
            "activation extraction requires PyTorch and Transformers",
        ) from error
    if device != "cuda":
        raise T051ArtifactError(
            "CUDA_REQUIRED",
            "the frozen T051 release extraction runs on the approved CUDA host",
        )
    if not torch.cuda.is_available():
        raise T051ArtifactError(
            "CUDA_UNAVAILABLE",
            "CUDA is unavailable for ChemDFM-R activation extraction",
        )
    _load_checkpoint_config(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise T051ArtifactError(
            "QWEN2_LAYER_PATH_MISSING",
            "expected decoder blocks at model.model.layers",
        )
    if len(model.model.layers) != EXPECTED_LAYER_COUNT:
        raise T051ArtifactError(
            "QWEN2_LAYER_COUNT_MISMATCH",
            "runtime decoder layer count drifted from approved config",
        )
    return torch, model


def _manifest_member_path(
    manifest_root: Path,
    value: object,
    *,
    allow_legacy_absolute: bool = False,
) -> tuple[str, Path]:
    """Resolve a manifest member below its root without permitting traversal."""

    if type(value) is not str or not value:
        raise T051ArtifactError(
            "ACTIVATION_SHARD_PATH_INVALID",
            "activation shard paths must be non-empty text",
        )
    root = Path(manifest_root)
    raw = Path(value)
    if raw.is_absolute():
        if not allow_legacy_absolute:
            raise T051ArtifactError(
                "ACTIVATION_SHARD_PATH_ABSOLUTE",
                "activation shard paths must be relative to the manifest directory",
                evidence={"path": value},
            )
        try:
            relative = raw.relative_to(root)
        except ValueError:
            try:
                relative = raw.resolve().relative_to(root.resolve())
            except ValueError as error:
                raise T051ArtifactError(
                    "ACTIVATION_SHARD_PATH_OUTSIDE_ROOT",
                    "legacy activation shard path is outside the manifest directory",
                    evidence={"path": value, "manifest_root": str(root)},
                ) from error
    else:
        relative = raw
    if relative == Path(".") or ".." in relative.parts:
        raise T051ArtifactError(
            "ACTIVATION_SHARD_PATH_TRAVERSAL",
            "activation shard paths cannot traverse outside the manifest directory",
            evidence={"path": value},
        )
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise T051ArtifactError(
            "ACTIVATION_SHARD_PATH_OUTSIDE_ROOT",
            "activation shard path resolves outside the manifest directory",
            evidence={"path": value, "manifest_root": str(root)},
        ) from error
    return relative.as_posix(), candidate


def _expected_row_offsets(token_counts: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for token_count in token_counts:
        offsets.append(offsets[-1] + token_count)
    return tuple(offsets)


def _validate_activation_tensor_payload(
    tensor_path: Path,
    *,
    torch_runtime: object,
    split: str,
    shard_index: int,
    record_ids: Sequence[str],
    token_counts: Sequence[int],
) -> None:
    """Load one completed tensor on CPU and validate its full release contract."""

    try:
        payload = torch_runtime.load(
            tensor_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise T051ArtifactError(
            "ACTIVATION_TENSOR_PAYLOAD_LOAD_FAILED",
            "existing activation tensor payload could not be loaded safely on CPU",
            evidence={"path": str(tensor_path)},
        ) from error
    activations: object | None = None
    try:
        if not isinstance(payload, Mapping):
            raise T051ArtifactError(
                "ACTIVATION_TENSOR_PAYLOAD_INVALID",
                "activation tensor payload must be a mapping",
                evidence={"path": str(tensor_path)},
            )
        expected_ids = tuple(record_ids)
        expected_counts = tuple(token_counts)
        expected_offsets = _expected_row_offsets(expected_counts)
        expected_tokens = expected_offsets[-1]
        mismatches: dict[str, object] = {}
        expected_scalars = {
            "format_version": T051_FORMAT_VERSION,
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "label_shift": 0,
            "layer_index": FROZEN_LAYER_INDEX,
            "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        }
        for key, expected in expected_scalars.items():
            if payload.get(key) != expected:
                mismatches[key] = {
                    "expected": expected,
                    "actual": payload.get(key),
                }
        try:
            payload_ids = _exact_string_sequence(
                payload.get("record_ids"), "tensor payload record_ids"
            )
            payload_counts = _exact_int_sequence(
                payload.get("token_counts"), "tensor payload token_counts"
            )
            payload_offsets = _exact_int_sequence(
                payload.get("row_offsets"), "tensor payload row_offsets"
            )
        except T051ArtifactError as error:
            mismatches["row_partition"] = error.to_dict()
        else:
            if payload_ids != expected_ids:
                mismatches["record_ids"] = {
                    "expected": expected_ids,
                    "actual": payload_ids,
                }
            if payload_counts != expected_counts:
                mismatches["token_counts"] = {
                    "expected": expected_counts,
                    "actual": payload_counts,
                }
            if payload_offsets != expected_offsets:
                mismatches["row_offsets"] = {
                    "expected": expected_offsets,
                    "actual": payload_offsets,
                }
        activations = payload.get("activations")
        if not torch_runtime.is_tensor(activations):
            mismatches["activations"] = "not-a-tensor"
        else:
            actual_shape = tuple(activations.shape)
            expected_shape = (expected_tokens, EXPECTED_HIDDEN_SIZE)
            if actual_shape != expected_shape:
                mismatches["activation_shape"] = {
                    "expected": expected_shape,
                    "actual": actual_shape,
                }
            actual_dtype = str(activations.dtype).removeprefix("torch.")
            if actual_dtype != "bfloat16":
                mismatches["activation_dtype"] = {
                    "expected": "bfloat16",
                    "actual": actual_dtype,
                }
        if mismatches:
            raise T051ArtifactError(
                "ACTIVATION_TENSOR_PAYLOAD_MISMATCH",
                "existing activation tensor payload differs from its exact shard plan",
                evidence={
                    "path": str(tensor_path),
                    "split": split,
                    "shard_index": shard_index,
                    "mismatches": mismatches,
                },
            )
    finally:
        del activations
        del payload
        gc.collect()


def _resume_metadata(
    plan: ActivationShardPlan,
    tensor_path: Path,
    metadata_path: Path,
    *,
    torch_runtime: object | None,
    strict_payload_validation: bool,
) -> ActivationShardSummary | None:
    if not tensor_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise T051ArtifactError(
            "ACTIVATION_RESUME_METADATA_INVALID",
            "existing activation sidecar is invalid JSON",
            evidence={"path": str(metadata_path)},
        ) from error
    if not isinstance(metadata, dict):
        raise T051ArtifactError(
            "ACTIVATION_RESUME_METADATA_INVALID",
            "existing activation sidecar must be an object",
        )
    expected_tokens = sum(plan.token_counts)
    expected = {
        "format_version": T051_FORMAT_VERSION,
        "status": "complete",
        "split": plan.split,
        "shard_index": plan.shard_index,
        "record_ids": list(plan.record_ids),
        "token_counts": list(plan.token_counts),
        "row_offsets": list(_expected_row_offsets(plan.token_counts)),
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "layer_index": FROZEN_LAYER_INDEX,
        "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        "activation_shape": [expected_tokens, EXPECTED_HIDDEN_SIZE],
        "activation_dtype": "bfloat16",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    file_bytes = tensor_path.stat().st_size
    if metadata.get("file_bytes") != file_bytes or file_bytes <= 0:
        mismatches["file_bytes"] = {
            "expected": metadata.get("file_bytes"),
            "actual": file_bytes,
        }
    if mismatches:
        raise T051ArtifactError(
            "ACTIVATION_RESUME_MISMATCH",
            "existing activation shard does not match the deterministic plan",
            evidence={"path": str(tensor_path), "mismatches": mismatches},
        )
    if strict_payload_validation:
        if torch_runtime is None:
            raise T051ArtifactError(
                "STRICT_RESUME_RUNTIME_REQUIRED",
                "strict resume validation requires the PyTorch runtime",
            )
        _validate_activation_tensor_payload(
            tensor_path,
            torch_runtime=torch_runtime,
            split=plan.split,
            shard_index=plan.shard_index,
            record_ids=plan.record_ids,
            token_counts=plan.token_counts,
        )
    return ActivationShardSummary(
        split=plan.split,
        shard_index=plan.shard_index,
        tensor_path=f"{plan.split}/{tensor_path.name}",
        metadata_path=f"{plan.split}/{metadata_path.name}",
        record_count=len(plan.rows),
        token_count=expected_tokens,
        hidden_size=EXPECTED_HIDDEN_SIZE,
        layer_index=FROZEN_LAYER_INDEX,
        resumed=True,
        file_bytes=file_bytes,
    )


def _extract_plan(
    plan: ActivationShardPlan,
    *,
    torch: object,
    model: object,
    output_root: Path,
    device: str,
    strict_resume_validation: bool,
) -> ActivationShardSummary:
    split_root = output_root / plan.split
    tensor_path = split_root / f"{plan.stem}.pt"
    metadata_path = split_root / f"{plan.stem}.json"
    resumed = _resume_metadata(
        plan,
        tensor_path,
        metadata_path,
        torch_runtime=torch,
        strict_payload_validation=strict_resume_validation,
    )
    if resumed is not None:
        return resumed
    split_root.mkdir(parents=True, exist_ok=True)
    layer = model.model.layers[FROZEN_LAYER_INDEX]
    captured: list[object] = []
    token_counts: list[int] = []

    def hook(_module: object, _inputs: object, output: object) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            raise T051ArtifactError(
                "LAYER_HOOK_OUTPUT_INVALID",
                "Qwen2 decoder block did not return a tensor resid_post",
            )
        captured.append(hidden.detach())

    handle = layer.register_forward_hook(hook)
    activation_rows: list[object] = []
    try:
        with torch.inference_mode():
            for row in plan.rows:
                captured.clear()
                input_ids = torch.tensor(
                    [list(_exact_int_sequence(row["input_ids"], "input_ids"))],
                    dtype=torch.long,
                    device=device,
                )
                attention = torch.tensor(
                    [list(_exact_int_sequence(row["attention_mask"], "attention_mask"))],
                    dtype=torch.long,
                    device=device,
                )
                model.model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
                if len(captured) != 1:
                    raise T051ArtifactError(
                        "LAYER_HOOK_CAPTURE_COUNT",
                        "layer-26 hook must fire exactly once per record",
                        evidence={
                            "record_id": row.get("record_id"),
                            "captures": len(captured),
                        },
                    )
                hidden = captured[0]
                token_count = assert_post_token_axis(
                    row,
                    tuple(hidden.shape),
                    expected_hidden_size=EXPECTED_HIDDEN_SIZE,
                )
                activation_rows.append(
                    hidden[0].to(device="cpu", dtype=torch.bfloat16).contiguous()
                )
                token_counts.append(token_count)
    finally:
        handle.remove()
    activations = torch.cat(activation_rows, dim=0)
    expected_shape = (sum(token_counts), EXPECTED_HIDDEN_SIZE)
    if tuple(activations.shape) != expected_shape:
        raise T051ArtifactError(
            "ACTIVATION_SHARD_SHAPE_INVALID",
            "concatenated resid_post shard has an unexpected shape",
            evidence={"expected": expected_shape, "actual": tuple(activations.shape)},
        )
    row_offsets = [0]
    for token_count in token_counts:
        row_offsets.append(row_offsets[-1] + token_count)
    payload = {
        "format_version": T051_FORMAT_VERSION,
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "layer_index": FROZEN_LAYER_INDEX,
        "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        "record_ids": list(plan.record_ids),
        "token_counts": token_counts,
        "row_offsets": row_offsets,
        "activations": activations,
    }
    temporary = tensor_path.with_name(tensor_path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, tensor_path)
    file_bytes = tensor_path.stat().st_size
    sidecar = {
        "format_version": T051_FORMAT_VERSION,
        "status": "complete",
        "split": plan.split,
        "shard_index": plan.shard_index,
        "record_ids": list(plan.record_ids),
        "token_counts": token_counts,
        "row_offsets": row_offsets,
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "layer_index": FROZEN_LAYER_INDEX,
        "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
        "feature_name": "resid_post",
        "activation_shape": list(expected_shape),
        "activation_dtype": "bfloat16",
        "tensor_relative_path": f"{plan.split}/{tensor_path.name}",
        "external_storage": "server_only",
        "file_bytes": file_bytes,
        "digest_computation_performed": False,
    }
    _write_text_atomic(metadata_path, _render_json(sidecar))
    return ActivationShardSummary(
        split=plan.split,
        shard_index=plan.shard_index,
        tensor_path=f"{plan.split}/{tensor_path.name}",
        metadata_path=f"{plan.split}/{metadata_path.name}",
        record_count=len(plan.rows),
        token_count=sum(token_counts),
        hidden_size=EXPECTED_HIDDEN_SIZE,
        layer_index=FROZEN_LAYER_INDEX,
        resumed=False,
        file_bytes=file_bytes,
    )


def extract_release_activations(
    release_root: Path = DEFAULT_RELEASE_ROOT,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    tokenized_root: Path | None = None,
    output_root: Path | None = None,
    splits: Sequence[str] = ("train", "validation", "test"),
    expected_counts: Mapping[str, int] | None = DEFAULT_SPLIT_COUNTS,
    shard_size: int = 8,
    limit_per_split: int | None = None,
    device: str = "cuda",
    progress: bool = False,
    strict_resume_validation: bool = True,
) -> dict[str, object]:
    """Extract restartable layer-26 resid_post shards from exact token IDs."""

    root = Path(release_root)
    source_root = (
        root / "tokenized/chemdfm_r"
        if tokenized_root is None
        else Path(tokenized_root)
    )
    target_root = (
        root / f"activations/chemdfm_r/layer_{FROZEN_LAYER_INDEX}"
        if output_root is None
        else Path(output_root)
    )
    if limit_per_split is not None and output_root is None:
        raise T051ArtifactError(
            "SMOKE_OUTPUT_ROOT_REQUIRED",
            "limited activation extraction must use a separate output root",
        )
    if limit_per_split is not None and (
        type(limit_per_split) is not int or limit_per_split <= 0
    ):
        raise ValueError("limit_per_split must be a positive integer or None")
    if limit_per_split is None and strict_resume_validation is not True:
        raise T051ArtifactError(
            "STRICT_RELEASE_RESUME_VALIDATION_REQUIRED",
            "release extraction requires strict payload validation for every resumed shard",
        )
    rows_by_split: dict[str, tuple[dict[str, object], ...]] = {}
    for split in splits:
        rows = _read_jsonl(source_root / f"{split}.jsonl")
        if expected_counts is not None and limit_per_split is None:
            expected = expected_counts.get(split)
            if expected is None or len(rows) != expected:
                raise T051ArtifactError(
                    "TOKENIZED_SPLIT_COUNT_MISMATCH",
                    "real tokenized split count differs from frozen release count",
                    evidence={
                        "split": split,
                        "expected": expected,
                        "actual": len(rows),
                    },
                )
        if limit_per_split is not None:
            rows = rows[:limit_per_split]
        for row in rows:
            validate_tokenized_row(row, expected_split=split)
        rows_by_split[split] = rows
    plans = plan_activation_shards(rows_by_split, shard_size=shard_size)
    checkpoint = Path(checkpoint_path)
    config = _load_checkpoint_config(checkpoint)
    if progress:
        print(
            json.dumps(
                {
                    "event": "model_load_started",
                    "checkpoint_path": str(checkpoint),
                    "layer_index": FROZEN_LAYER_INDEX,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch, model = _load_runtime_model(checkpoint, device)
    if progress:
        print(
            json.dumps(
                {
                    "event": "model_load_complete",
                    "checkpoint_path": str(checkpoint),
                    "layer_index": FROZEN_LAYER_INDEX,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summaries: list[ActivationShardSummary] = []
    extraction_started = time.monotonic()
    total_records = sum(len(plan.rows) for plan in plans)
    completed_records = 0
    try:
        for plan_index, plan in enumerate(plans, start=1):
            summary = _extract_plan(
                plan,
                torch=torch,
                model=model,
                output_root=target_root,
                device=device,
                strict_resume_validation=strict_resume_validation,
            )
            summaries.append(summary)
            completed_records += summary.record_count
            if progress:
                elapsed = max(time.monotonic() - extraction_started, 1e-9)
                records_per_second = completed_records / elapsed
                remaining_records = total_records - completed_records
                print(
                    json.dumps(
                        {
                            "event": "activation_shard_complete",
                            "split": summary.split,
                            "shard_index": summary.shard_index,
                            "shards_complete": plan_index,
                            "shards_total": len(plans),
                            "records_complete": completed_records,
                            "records_total": total_records,
                            "token_count": summary.token_count,
                            "resumed": summary.resumed,
                            "elapsed_seconds": round(elapsed, 3),
                            "eta_seconds": round(
                                remaining_records / records_per_second, 3
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    split_rows = {
        split: [summary for summary in summaries if summary.split == split]
        for split in rows_by_split
    }
    tokenizer_fingerprints = {
        json.dumps(row["tokenizer_fingerprint"], sort_keys=True)
        for rows in rows_by_split.values()
        for row in rows
    }
    if len(tokenizer_fingerprints) != 1:
        raise T051ArtifactError(
            "TOKENIZER_FINGERPRINT_SET_MISMATCH",
            "all release token rows must use one exact real tokenizer fingerprint",
        )
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    manifest = {
        "format_version": T051_ACTIVATION_MANIFEST_VERSION,
        "status": "complete",
        "mode": "smoke" if limit_per_split is not None else "release",
        "model": {
            "name": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "model_type": config["model_type"],
            "architecture": EXPECTED_ARCHITECTURE,
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "num_hidden_layers": EXPECTED_LAYER_COUNT,
            "weight_dtype": str(config.get("torch_dtype", "bfloat16")),
            "checkpoint_identity_method": "approved-local-path-and-config-no-digest",
            "digest_computation_performed": False,
        },
        "feature": {
            "name": "resid_post",
            "layer_index": FROZEN_LAYER_INDEX,
            "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
            "hook_output": "decoder block output[0]",
            "activation_dtype": "bfloat16",
            "selection_status": "pre_frozen_before_pilot_activation_extraction",
            "selection_source": FROZEN_LAYER_SELECTION_SOURCE,
            "selected_using_pilot_records": False,
        },
        "alignment": {
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "token_consumed_before_capture": True,
            "label_shift": 0,
            "hidden_token_axis_equals_label_length": True,
            "pre_token_claimed": False,
        },
        "tokenizer_fingerprint": json.loads(next(iter(tokenizer_fingerprints))),
        "environment": versions,
        "resume_contract": {
            "sharded": True,
            "shard_size_records": shard_size,
            "atomic_tensor_publish": True,
            "atomic_sidecar_publish": True,
            "completed_shards_are_resumable": True,
            "strict_payload_validation": strict_resume_validation,
            "tensor_payload_loaded_on_cpu": strict_resume_validation,
            "digest_verification_performed": False,
        },
        "external_storage": {
            "tensor_policy": "server_only",
            "tensor_root": ".",
            "approved_server_path": str(target_root.resolve()),
            "path_basis": "relative_to_activation_manifest_directory",
            "git_payload": "manifest_and_json_sidecars_only",
            "digest_computation_performed": False,
        },
        "splits": {
            split: {
                "record_count": len(rows_by_split[split]),
                "token_count": sum(summary.token_count for summary in split_rows[split]),
                "shard_count": len(split_rows[split]),
            }
            for split in rows_by_split
        },
        "record_count": sum(len(rows) for rows in rows_by_split.values()),
        "token_count": sum(summary.token_count for summary in summaries),
        "shard_count": len(summaries),
        "resumed_shard_count": sum(summary.resumed for summary in summaries),
        "shards": [summary.to_dict() for summary in summaries],
    }
    _write_text_atomic(target_root / "manifest.json", _render_json(manifest))
    return manifest


def _load_release_token_axes(
    tokenized_root: Path,
    tokenized_index_path: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[tuple[str, int], ...]]]:
    """Load exact ordered ID/token-count axes from canonical and Git artifacts."""

    _, index = _load_git_shard_index(tokenized_index_path)
    split_order_raw = _require_sequence(
        index.get("split_order"), "git shard split_order"
    )
    split_order: list[str] = []
    seen_splits: set[str] = set()
    axes: dict[str, tuple[tuple[str, int], ...]] = {}
    root = Path(tokenized_root)
    for split_value in split_order_raw:
        if type(split_value) is not str or not split_value or split_value in seen_splits:
            raise T051ArtifactError(
                "GIT_SHARD_SPLIT_ORDER_INVALID",
                "git shard split order must contain unique non-empty names",
            )
        split = split_value
        seen_splits.add(split)
        split_order.append(split)
        git_rows = iter_git_tokenized_rows(tokenized_index_path, split)
        git_axis = tuple(
            (
                str(row["record_id"]),
                validate_tokenized_row(row, expected_split=split),
            )
            for row in git_rows
        )
        canonical_path = root / f"{split}.jsonl"
        if canonical_path.is_file():
            canonical_rows = _read_jsonl(canonical_path)
            canonical_axis = tuple(
                (
                    str(row["record_id"]),
                    validate_tokenized_row(row, expected_split=split),
                )
                for row in canonical_rows
            )
            if canonical_axis != git_axis:
                raise T051ArtifactError(
                    "TOKENIZED_CANONICAL_SHARD_AXIS_MISMATCH",
                    "canonical and Git tokenized artifacts differ in ordered IDs or token counts",
                    evidence={"split": split},
                )
            axes[split] = canonical_axis
        else:
            axes[split] = git_axis
    return tuple(split_order), axes


def finalize_activation_inventory(
    manifest_path: Path,
    *,
    tokenized_root: Path,
    tokenized_index_path: Path,
    strict_payload_validation: bool = True,
    torch_runtime: object | None = None,
) -> dict[str, object]:
    """Validate server-side tensors, token axes, and the full payload inventory."""

    path = Path(manifest_path)
    if not path.is_file():
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_MISSING",
            "activation extraction manifest is missing",
            evidence={"path": str(path)},
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_INVALID",
            "activation extraction manifest is invalid JSON",
        ) from error
    if not isinstance(manifest, dict) or manifest.get("format_version") != (
        T051_ACTIVATION_MANIFEST_VERSION
    ):
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_INVALID",
            "activation extraction manifest has an unsupported format",
        )
    if manifest.get("status") != "complete":
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_INCOMPLETE",
            "activation inventory can be finalized only after complete extraction",
        )
    if strict_payload_validation is not True and manifest.get("mode") not in {
        "smoke",
        "test",
    }:
        raise T051ArtifactError(
            "STRICT_RELEASE_PAYLOAD_VALIDATION_REQUIRED",
            "release inventory finalization requires loading every tensor payload",
        )
    if strict_payload_validation and torch_runtime is None:
        try:
            import torch as imported_torch
        except ImportError as error:
            raise T051ArtifactError(
                "STRICT_RESUME_RUNTIME_REQUIRED",
                "strict tensor payload validation requires PyTorch",
            ) from error
        torch_runtime = imported_torch
    alignment = _require_mapping(manifest.get("alignment"), "activation alignment")
    if (
        alignment.get("activation_alignment") != ACTIVATION_ALIGNMENT
        or alignment.get("label_shift") != 0
        or alignment.get("hidden_token_axis_equals_label_length") is not True
    ):
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_ALIGNMENT_INVALID",
            "activation manifest must record exact unshifted post-token alignment",
        )
    split_order, token_axes = _load_release_token_axes(
        Path(tokenized_root), Path(tokenized_index_path)
    )
    raw_shards = _require_sequence(manifest.get("shards"), "activation shards")
    seen_ids: set[str] = set()
    total_records = 0
    total_tokens = 0
    total_bytes = 0
    max_file_bytes = 0
    split_counts: dict[str, dict[str, int]] = {}
    activation_axes: dict[str, list[tuple[str, int]]] = {
        split: [] for split in split_order
    }
    next_shard_index = {split: 0 for split in split_order}
    normalized_shards: list[dict[str, object]] = []
    sidecar_updates: list[tuple[Path, dict[str, object]]] = []
    for raw_summary in raw_shards:
        summary = _require_mapping(raw_summary, "activation shard summary")
        split = summary.get("split")
        shard_index = summary.get("shard_index")
        if type(split) is not str or type(shard_index) is not int:
            raise T051ArtifactError(
                "ACTIVATION_SHARD_SUMMARY_INVALID",
                "activation shard summary has invalid identity fields",
            )
        if split not in token_axes or shard_index != next_shard_index[split]:
            raise T051ArtifactError(
                "ACTIVATION_SHARD_ORDER_INVALID",
                "activation shard indices must be contiguous within each tokenized split",
                evidence={
                    "split": split,
                    "expected_shard_index": next_shard_index.get(split),
                    "actual_shard_index": shard_index,
                },
            )
        next_shard_index[split] += 1
        tensor_relative, tensor_path = _manifest_member_path(
            path.parent,
            summary.get("tensor_path"),
            allow_legacy_absolute=True,
        )
        metadata_relative, metadata_path = _manifest_member_path(
            path.parent,
            summary.get("metadata_path"),
            allow_legacy_absolute=True,
        )
        if not tensor_path.is_file() or not metadata_path.is_file():
            raise T051ArtifactError(
                "ACTIVATION_SHARD_FILE_MISSING",
                "activation tensor and sidecar must both exist on the server",
                evidence={
                    "tensor_path": tensor_relative,
                    "metadata_path": metadata_relative,
                },
            )
        try:
            sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise T051ArtifactError(
                "ACTIVATION_SIDECAR_INVALID",
                "activation sidecar is invalid JSON",
                evidence={"path": metadata_relative},
            ) from error
        if not isinstance(sidecar, dict):
            raise T051ArtifactError(
                "ACTIVATION_SIDECAR_INVALID",
                "activation sidecar must be an object",
                evidence={"path": metadata_relative},
            )
        record_ids = _exact_string_sequence(
            sidecar.get("record_ids"), "activation sidecar record_ids"
        )
        token_counts = _exact_int_sequence(
            sidecar.get("token_counts"), "activation sidecar token_counts"
        )
        row_offsets = _exact_int_sequence(
            sidecar.get("row_offsets"), "activation sidecar row_offsets"
        )
        shard_tokens = sum(token_counts)
        file_bytes = tensor_path.stat().st_size
        expected = {
            "format_version": T051_FORMAT_VERSION,
            "status": "complete",
            "split": split,
            "shard_index": shard_index,
            "activation_alignment": ACTIVATION_ALIGNMENT,
            "label_shift": 0,
            "layer_index": FROZEN_LAYER_INDEX,
            "hook_path": f"model.model.layers[{FROZEN_LAYER_INDEX}]",
            "activation_shape": [shard_tokens, EXPECTED_HIDDEN_SIZE],
            "activation_dtype": "bfloat16",
            "file_bytes": file_bytes,
        }
        mismatch = {
            key: {"expected": value, "actual": sidecar.get(key)}
            for key, value in expected.items()
            if sidecar.get(key) != value
        }
        expected_offsets = _expected_row_offsets(token_counts)
        if len(record_ids) != len(token_counts) or row_offsets != expected_offsets:
            mismatch["row_partition"] = {
                "record_count": len(record_ids),
                "token_counts": token_counts,
                "expected_row_offsets": expected_offsets,
                "actual_row_offsets": row_offsets,
            }
        if (
            summary.get("record_count") != len(record_ids)
            or summary.get("token_count") != shard_tokens
            or summary.get("hidden_size") != EXPECTED_HIDDEN_SIZE
            or summary.get("layer_index") != FROZEN_LAYER_INDEX
            or summary.get("file_bytes") != file_bytes
        ):
            mismatch["manifest_summary"] = {
                "record_count": summary.get("record_count"),
                "token_count": summary.get("token_count"),
                "hidden_size": summary.get("hidden_size"),
                "layer_index": summary.get("layer_index"),
                "file_bytes": summary.get("file_bytes"),
            }
        overlap = seen_ids.intersection(record_ids)
        if overlap:
            mismatch["duplicate_record_ids"] = tuple(sorted(overlap))
        if mismatch:
            raise T051ArtifactError(
                "ACTIVATION_INVENTORY_MISMATCH",
                "activation tensor/sidecar inventory violates the frozen plan",
                evidence={"path": metadata_relative, "mismatch": mismatch},
            )
        if strict_payload_validation:
            if torch_runtime is None:
                raise AssertionError("strict payload runtime was not initialized")
            _validate_activation_tensor_payload(
                tensor_path,
                torch_runtime=torch_runtime,
                split=split,
                shard_index=shard_index,
                record_ids=record_ids,
                token_counts=token_counts,
            )
        seen_ids.update(record_ids)
        activation_axes[split].extend(zip(record_ids, token_counts, strict=True))
        updated_sidecar = dict(sidecar)
        updated_sidecar["tensor_relative_path"] = tensor_relative
        updated_sidecar["external_storage"] = "server_only"
        updated_sidecar["digest_computation_performed"] = False
        sidecar_updates.append((metadata_path, updated_sidecar))
        normalized_summary = dict(summary)
        normalized_summary["tensor_path"] = tensor_relative
        normalized_summary["metadata_path"] = metadata_relative
        normalized_shards.append(normalized_summary)
        total_records += len(record_ids)
        total_tokens += shard_tokens
        total_bytes += file_bytes
        max_file_bytes = max(max_file_bytes, file_bytes)
        counts = split_counts.setdefault(
            split, {"records": 0, "tokens": 0, "shards": 0}
        )
        counts["records"] += len(record_ids)
        counts["tokens"] += shard_tokens
        counts["shards"] += 1
    for split in split_order:
        actual_axis = tuple(activation_axes[split])
        if actual_axis != token_axes[split]:
            raise T051ArtifactError(
                "ACTIVATION_TOKENIZED_AXIS_MISMATCH",
                "activation sidecars differ from tokenized ordered IDs or token counts",
                evidence={"split": split},
            )
    manifest_splits = _require_mapping(manifest.get("splits"), "activation splits")
    if set(manifest_splits) != set(split_order) or set(split_counts) != set(split_order):
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_SPLIT_SET_MISMATCH",
            "activation manifest split set differs from the tokenized release",
        )
    split_mismatches: dict[str, object] = {}
    for split in split_order:
        declared = _require_mapping(
            manifest_splits.get(split), f"activation manifest split {split}"
        )
        actual = split_counts[split]
        expected_declared = {
            "record_count": actual["records"],
            "token_count": actual["tokens"],
            "shard_count": actual["shards"],
        }
        mismatch = {
            key: {"expected": value, "actual": declared.get(key)}
            for key, value in expected_declared.items()
            if declared.get(key) != value
        }
        if mismatch:
            split_mismatches[split] = mismatch
    if split_mismatches:
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_SPLIT_TOTAL_MISMATCH",
            "activation manifest per-split totals differ from validated sidecars",
            evidence=split_mismatches,
        )
    if (
        manifest.get("record_count") != total_records
        or manifest.get("token_count") != total_tokens
        or manifest.get("shard_count") != len(raw_shards)
    ):
        raise T051ArtifactError(
            "ACTIVATION_MANIFEST_TOTAL_MISMATCH",
            "activation manifest totals differ from validated sidecars",
            evidence={
                "records": total_records,
                "tokens": total_tokens,
                "shards": len(raw_shards),
            },
        )
    payload_validation = {
        "performed": strict_payload_validation,
        "all_pass": strict_payload_validation,
        "shard_count": len(raw_shards),
        "record_count": total_records,
        "token_count": total_tokens,
        "activation_dtype": "bfloat16" if strict_payload_validation else None,
        "ordered_record_ids_exact": strict_payload_validation,
        "token_counts_exact": strict_payload_validation,
        "row_offsets_exact": strict_payload_validation,
        "activation_shape_exact": strict_payload_validation,
        "activation_alignment_exact": strict_payload_validation,
        "layer_index_exact": strict_payload_validation,
        "digest_verification_performed": False,
    }
    report = {
        "all_pass": True,
        "activation_alignment": ACTIVATION_ALIGNMENT,
        "label_shift": 0,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "layer_index": FROZEN_LAYER_INDEX,
        "record_count": total_records,
        "token_count": total_tokens,
        "shard_count": len(raw_shards),
        "tensor_bytes": total_bytes,
        "max_tensor_file_bytes": max_file_bytes,
        "unique_record_id_count": len(seen_ids),
        "splits": split_counts,
        "tokenized_axis_validation": {
            "canonical_root": str(Path(tokenized_root)),
            "git_shard_index": str(Path(tokenized_index_path)),
            "ordered_record_ids_exact": True,
            "token_counts_exact": True,
        },
        "tensor_payload_validation": payload_validation,
        "digest_verification_performed": False,
    }
    for metadata_path, sidecar in sidecar_updates:
        _write_text_atomic(metadata_path, _render_json(sidecar))
    manifest["shards"] = normalized_shards
    manifest["external_storage"] = {
        "tensor_policy": "server_only",
        "tensor_root": ".",
        "approved_server_path": str(path.parent.resolve()),
        "path_basis": "relative_to_activation_manifest_directory",
        "git_payload": "manifest_and_json_sidecars_only",
        "checkpoint_identity_limitation": (
            "approved path and Qwen2 config are provenance claims, not byte-exact identity"
        ),
        "digest_computation_performed": False,
    }
    manifest["checkpoint_tokenizer_identity_scope"] = {
        "method": "approved-local-path-and-validated-qwen2-config-no-digest",
        "provenance_claimed": True,
        "byte_exact_identity_proven": False,
        "byte_exact_reproducibility_claimed": False,
        "digest_computation_performed": False,
    }
    manifest["tensor_payload_validation"] = payload_validation
    manifest["inventory_validation"] = report
    _write_text_atomic(path, _render_json(manifest))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("tokenize", "extract", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
        child.add_argument(
            "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH
        )
        child.add_argument(
            "--splits",
            nargs="+",
            default=["train", "validation", "test"],
        )
        child.add_argument("--limit-per-split", type=int, default=None)
    tokenize = subparsers.choices["tokenize"]
    tokenize.add_argument("--output-root", type=Path, default=None)
    extract = subparsers.choices["extract"]
    extract.add_argument("--tokenized-root", type=Path, default=None)
    extract.add_argument("--output-root", type=Path, default=None)
    extract.add_argument("--shard-size", type=int, default=8)
    run = subparsers.choices["run"]
    run.add_argument("--activation-root", type=Path, default=None)
    run.add_argument("--shard-size", type=int, default=8)
    shard = subparsers.add_parser("shard")
    shard.add_argument(
        "--canonical-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT / "tokenized/chemdfm_r",
    )
    shard.add_argument("--shard-root", type=Path, default=None)
    shard.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )
    shard.add_argument(
        "--max-shard-bytes",
        type=int,
        default=GIT_SHARD_MAX_BYTES,
    )
    validate_activations = subparsers.add_parser("validate-activations")
    validate_activations.add_argument("--manifest-path", type=Path, required=True)
    validate_activations.add_argument("--tokenized-root", type=Path, required=True)
    validate_activations.add_argument(
        "--tokenized-index-path", type=Path, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "shard":
        result = write_git_tokenized_shards(
            args.canonical_root,
            shard_root=args.shard_root,
            splits=args.splits,
            max_shard_bytes=args.max_shard_bytes,
        )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-activations":
        result = finalize_activation_inventory(
            args.manifest_path,
            tokenized_root=args.tokenized_root,
            tokenized_index_path=args.tokenized_index_path,
            strict_payload_validation=True,
        )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
        return 0
    expected = None if args.limit_per_split is not None else DEFAULT_SPLIT_COUNTS
    if args.command == "run" and args.limit_per_split is not None:
        raise T051ArtifactError(
            "SMOKE_RUN_UNSUPPORTED",
            "limited smoke runs must use separate tokenize and extract commands",
        )
    if args.command == "tokenize":
        result = tokenize_release(
            args.release_root,
            checkpoint_path=args.checkpoint_path,
            splits=args.splits,
            expected_counts=expected,
            output_root=args.output_root,
            limit_per_split=args.limit_per_split,
            progress=True,
        )
    elif args.command == "extract":
        result = extract_release_activations(
            args.release_root,
            checkpoint_path=args.checkpoint_path,
            tokenized_root=args.tokenized_root,
            output_root=args.output_root,
            splits=args.splits,
            expected_counts=expected,
            shard_size=args.shard_size,
            limit_per_split=args.limit_per_split,
            progress=True,
        )
    else:
        tokenized = tokenize_release(
            args.release_root,
            checkpoint_path=args.checkpoint_path,
            splits=args.splits,
            expected_counts=expected,
            limit_per_split=args.limit_per_split,
            progress=True,
        )
        activations = extract_release_activations(
            args.release_root,
            checkpoint_path=args.checkpoint_path,
            output_root=args.activation_root,
            splits=args.splits,
            expected_counts=expected,
            shard_size=args.shard_size,
            limit_per_split=args.limit_per_split,
            progress=True,
        )
        result = {"tokenization": tokenized, "activations": activations}
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
