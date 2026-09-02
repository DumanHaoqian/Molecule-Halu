"""ChemDFM-R character-to-token projection with reliable fast offsets.

Character annotations remain canonical.  This module creates a tokenizer-
specific derived :class:`TokenLabelSet` by applying half-open any-overlap to
literal spans.  It never decodes tokens or searches rendered text for values.

The authoritative input is :class:`SerializedDetectorInput`: its constructor
already binds the exact detector text, content segments, and serialized-text
identity.  The identical ``text`` object is passed once to an injected fast
tokenizer, and the frozen identity is copied into the resulting label set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from molhallulens.modules.annotation.char_annotations import CharAnnotationBuildResult
from molhallulens.modules.text_realization.spans import validate_char_span
from molhallulens.core import (
    CausalRole,
    CharAnnotation,
    CharSpan,
    EditErrorSubtype,
    HallucinationType,
    SegmentKind,
    TokenizerFingerprint,
    TokenLabelSet,
    VariantLabel,
)
from molhallulens.modules.text_realization.detector_prompt import SerializedDetectorInput
from molhallulens.modules.text_realization.trace_ast import RenderedExample

TOKEN_PROJECTION_VERSION = "chemdfm_r_token_projection_v1"
ACTIVATION_ALIGNMENT = "post_token_h_t"

_VISIBLE_SEGMENTS = frozenset(
    {
        SegmentKind.SOURCE,
        SegmentKind.INSTRUCTION,
        SegmentKind.REASONING,
        SegmentKind.FINAL_ANSWER,
    }
)
_EVALUATED_SEGMENTS = frozenset({SegmentKind.REASONING, SegmentKind.FINAL_ANSWER})
_LOCAL_FALSEHOOD_ROLES = frozenset(
    {CausalRole.ROOT, CausalRole.PROPAGATED_FALSE, CausalRole.TERMINAL}
)


class TokenProjectionError(RuntimeError):
    """Structured fail-closed tokenizer/projection failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("projection error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("projection error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("projection error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def _serialized_content_span(
    serialized: SerializedDetectorInput,
    component: SegmentKind,
) -> CharSpan:
    matches = tuple(
        CharSpan(segment.start, segment.end)
        for segment in serialized.segments
        if segment.segment_kind is component
    )
    if len(matches) != 1:
        raise TokenProjectionError(
            "SERIALIZED_COMPONENT_AMBIGUOUS",
            "serialized detector input must have exactly one requested content segment",
            evidence={"component": component.value},
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class DetectorCoordinateMap:
    """Exact trace-local to serialized-detector coordinate translation.

    T039/T041 offsets are bound to :attr:`RenderedExample.detector_text`, while
    token offsets are bound to the complete T005 detector prompt. The mapping
    is constructed only after comparing entire reasoning and answer surfaces;
    it never searches for a repeated literal.
    """

    trace_text: str
    serialized_text: str
    reasoning_trace_span: CharSpan
    reasoning_serialized_span: CharSpan
    answer_value_trace_span: CharSpan
    answer_serialized_span: CharSpan

    def __post_init__(self) -> None:
        if type(self.trace_text) is not str or type(self.serialized_text) is not str:
            raise TypeError("coordinate-map texts must be strings")
        for span, text, name in (
            (self.reasoning_trace_span, self.trace_text, "reasoning trace"),
            (
                self.reasoning_serialized_span,
                self.serialized_text,
                "reasoning serialized",
            ),
            (self.answer_value_trace_span, self.trace_text, "answer trace"),
            (
                self.answer_serialized_span,
                self.serialized_text,
                "answer serialized",
            ),
        ):
            try:
                validate_char_span(text, span)
            except (TypeError, ValueError) as error:
                raise TokenProjectionError(
                    "COORDINATE_MAP_SPAN_INVALID",
                    f"{name} coordinate-map span is invalid",
                ) from error
        if (
            self.trace_text[
                self.reasoning_trace_span.start : self.reasoning_trace_span.end
            ]
            != self.serialized_text[
                self.reasoning_serialized_span.start : self.reasoning_serialized_span.end
            ]
            or self.trace_text[
                self.answer_value_trace_span.start : self.answer_value_trace_span.end
            ]
            != self.serialized_text[
                self.answer_serialized_span.start : self.answer_serialized_span.end
            ]
        ):
            raise TokenProjectionError(
                "TRACE_SERIALIZED_TEXT_MISMATCH",
                "trace and serialized component surfaces must be exactly identical",
            )

    @classmethod
    def from_rendered(
        cls,
        rendered_example: RenderedExample,
        serialized: SerializedDetectorInput,
    ) -> DetectorCoordinateMap:
        """Build an exact mapping without substring lookup or literal recovery."""

        if type(rendered_example) is not RenderedExample:
            raise TypeError("rendered_example must be a RenderedExample")
        if type(serialized) is not SerializedDetectorInput:
            raise TypeError("serialized must be a SerializedDetectorInput")
        reasoning_serialized = _serialized_content_span(
            serialized, SegmentKind.REASONING
        )
        answer_serialized = _serialized_content_span(
            serialized, SegmentKind.FINAL_ANSWER
        )

        # An exact full-text rendering is already in canonical coordinates.
        if rendered_example.detector_text == serialized.text:
            return cls(
                trace_text=rendered_example.detector_text,
                serialized_text=serialized.text,
                reasoning_trace_span=reasoning_serialized,
                reasoning_serialized_span=reasoning_serialized,
                answer_value_trace_span=answer_serialized,
                answer_serialized_span=answer_serialized,
            )

        unknown_segments = tuple(
            sorted(
                segment_id
                for segment_id in rendered_example.segment_spans
                if not segment_id.startswith("reasoning.step.")
                and segment_id != "final_answer"
            )
        )
        reasoning_segments = tuple(
            sorted(
                (
                    span
                    for segment_id, span in rendered_example.segment_spans.items()
                    if segment_id.startswith("reasoning.step.")
                ),
                key=lambda span: (span.start, span.end),
            )
        )
        answer_trace = rendered_example.segment_spans.get("final_answer")
        if unknown_segments or not reasoning_segments or answer_trace is None:
            raise TokenProjectionError(
                "TRACE_SEGMENT_LAYOUT_INVALID",
                "rendered trace must expose reasoning.step.* and final_answer segments only",
                evidence={"unknown_segments": unknown_segments},
            )
        if (
            reasoning_segments[0].start != 0
            or reasoning_segments[-1].end >= answer_trace.start
            or answer_trace.end != len(rendered_example.detector_text)
        ):
            raise TokenProjectionError(
                "TRACE_SEGMENT_LAYOUT_INVALID",
                "rendered reasoning and final-answer segments have an invalid layout",
            )
        reasoning_trace = CharSpan(0, reasoning_segments[-1].end)
        reasoning_surface = rendered_example.detector_text[
            reasoning_trace.start : reasoning_trace.end
        ]
        if reasoning_surface != serialized.detector_input.reasoning_chain:
            raise TokenProjectionError(
                "TRACE_SERIALIZED_TEXT_MISMATCH",
                "rendered reasoning is not the exact serialized reasoning field",
                evidence={"component": SegmentKind.REASONING.value},
            )

        answer_surface = rendered_example.detector_text[
            answer_trace.start : answer_trace.end
        ]
        answer_value = serialized.detector_input.final_answer
        answer_prefix = "Answer: "
        if answer_surface == answer_value:
            answer_value_trace = answer_trace
        elif answer_surface == answer_prefix + answer_value:
            answer_value_trace = CharSpan(
                answer_trace.start + len(answer_prefix), answer_trace.end
            )
        else:
            raise TokenProjectionError(
                "TRACE_SERIALIZED_TEXT_MISMATCH",
                "rendered Answer segment is not the exact serialized final-answer value",
                evidence={"component": SegmentKind.FINAL_ANSWER.value},
            )
        return cls(
            trace_text=rendered_example.detector_text,
            serialized_text=serialized.text,
            reasoning_trace_span=reasoning_trace,
            reasoning_serialized_span=reasoning_serialized,
            answer_value_trace_span=answer_value_trace,
            answer_serialized_span=answer_serialized,
        )

    @staticmethod
    def _translate(
        span: CharSpan,
        *,
        source: CharSpan,
        target: CharSpan,
        allow_context_clamp: bool,
    ) -> CharSpan:
        if type(span) is not CharSpan:
            raise TypeError("translated span must be a CharSpan")
        if allow_context_clamp:
            local_start = max(span.start, source.start)
            local_end = min(span.end, source.end)
            if local_end <= local_start:
                raise TokenProjectionError(
                    "TRACE_SPAN_OUTSIDE_COMPONENT",
                    "claim context does not overlap its serialized component value",
                )
        else:
            if not (source.start <= span.start and span.end <= source.end):
                raise TokenProjectionError(
                    "TRACE_SPAN_OUTSIDE_COMPONENT",
                    "literal span is outside its rendered component value",
                )
            local_start, local_end = span.start, span.end
        translated = CharSpan(
            target.start + local_start - source.start,
            target.start + local_end - source.start,
        )
        if translated.end > target.end:
            raise TokenProjectionError(
                "TRACE_SPAN_TRANSLATION_OVERFLOW",
                "translated span exceeds its serialized component",
            )
        return translated

    def rebase_span(
        self,
        span: CharSpan,
        component: SegmentKind,
        *,
        claim: bool = False,
    ) -> CharSpan:
        """Translate one component-qualified span to serialized coordinates."""

        if type(component) is not SegmentKind:
            raise TypeError("component must be a SegmentKind")
        if type(claim) is not bool:
            raise TypeError("claim must be a bool")
        if component is SegmentKind.REASONING:
            return self._translate(
                span,
                source=self.reasoning_trace_span,
                target=self.reasoning_serialized_span,
                allow_context_clamp=False,
            )
        if component is SegmentKind.FINAL_ANSWER:
            return self._translate(
                span,
                source=self.answer_value_trace_span,
                target=self.answer_serialized_span,
                allow_context_clamp=claim,
            )
        raise TokenProjectionError(
            "TRACE_SPAN_COMPONENT_INVALID",
            "only reasoning and final-answer spans can be rebased",
            evidence={"component": component.value},
        )

    def rebase_any_span(self, span: CharSpan) -> CharSpan:
        """Translate an unlabeled matched-control span by exact containment."""

        if type(span) is not CharSpan:
            raise TypeError("matched span must be a CharSpan")
        candidates: list[tuple[CharSpan, CharSpan]] = []
        for source, target in (
            (self.reasoning_trace_span, self.reasoning_serialized_span),
            (self.answer_value_trace_span, self.answer_serialized_span),
        ):
            if source.start <= span.start and span.end <= source.end:
                candidates.append((source, target))
        if len(candidates) != 1:
            raise TokenProjectionError(
                "MATCHED_TARGET_COMPONENT_AMBIGUOUS",
                "matched trace span must belong to exactly one evaluated component",
            )
        source, target = candidates[0]
        return self._translate(
            span,
            source=source,
            target=target,
            allow_context_clamp=False,
        )


def rebase_char_annotations(
    rendered_example: RenderedExample,
    serialized: SerializedDetectorInput,
    annotations: CharAnnotationBuildResult | Sequence[CharAnnotation],
) -> CharAnnotationBuildResult | tuple[CharAnnotation, ...]:
    """Rebase T041 trace-local annotations into T005 detector coordinates."""

    coordinate_map = DetectorCoordinateMap.from_rendered(rendered_example, serialized)
    if type(annotations) is CharAnnotationBuildResult:
        values = annotations.annotations
    else:
        if isinstance(annotations, (str, bytes)) or not isinstance(
            annotations, Sequence
        ):
            raise TypeError(
                "annotations must be CharAnnotationBuildResult or a sequence"
            )
        values = tuple(annotations)
    if any(type(annotation) is not CharAnnotation for annotation in values):
        raise TypeError("annotations must contain CharAnnotation values")
    rebased = tuple(
        replace(
            annotation,
            literal_span=coordinate_map.rebase_span(
                annotation.literal_span, annotation.component
            ),
            claim_span=coordinate_map.rebase_span(
                annotation.claim_span, annotation.component, claim=True
            ),
        )
        for annotation in values
    )
    if type(annotations) is CharAnnotationBuildResult:
        return CharAnnotationBuildResult(
            annotations=rebased,
            event_links=annotations.event_links,
            unlocalized_omissions=annotations.unlocalized_omissions,
            builder_version=annotations.builder_version,
        )
    return rebased


@runtime_checkable
class FastOffsetTokenizer(Protocol):
    """Hugging Face fast-tokenizer surface required by the projector."""

    is_fast: bool

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class FastTokenization:
    """One validated, unbatched fast-tokenizer result."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        length = len(self.input_ids)
        if not length:
            raise ValueError("FastTokenization cannot be empty")
        if any(type(value) is not int or value < 0 for value in self.input_ids):
            raise TypeError("FastTokenization input_ids must be non-negative integers")
        for values, name in (
            (self.attention_mask, "attention_mask"),
            (self.special_tokens_mask, "special_tokens_mask"),
        ):
            if len(values) != length:
                raise ValueError(f"{name} must match input_ids length")
            if any(type(value) is not int or value not in {0, 1} for value in values):
                raise TypeError(f"{name} must contain exact integer 0/1 values")
        if len(self.offset_mapping) != length:
            raise ValueError("offset_mapping must match input_ids length")
        if any(
            type(start) is not int or type(end) is not int or start < 0 or end < start
            for start, end in self.offset_mapping
        ):
            raise ValueError(
                "offset_mapping must contain valid half-open integer pairs"
            )


def _exact_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_SHAPE",
            f"tokenizer {name} must be an unbatched sequence",
        )
    values = tuple(value)
    if (
        values
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], (str, bytes))
    ):
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_BATCHED",
            "projector accepts exactly one unbatched tokenizer sequence",
        )
    if any(type(item) is not int for item in values):
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_TYPE",
            f"tokenizer {name} must contain exact integers",
        )
    return values


def _offset_tuple(value: object) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenProjectionError(
            "TOKENIZER_OFFSETS_MISSING",
            "fast tokenizer must return an offset_mapping sequence",
        )
    offsets: list[tuple[int, int]] = []
    for pair in value:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TokenProjectionError(
                "TOKENIZER_OFFSET_SHAPE",
                "each fast-tokenizer offset must be a two-integer pair",
            )
        start, end = pair
        if type(start) is not int or type(end) is not int:
            raise TokenProjectionError(
                "TOKENIZER_OFFSET_TYPE",
                "fast-tokenizer offsets must use exact integers",
            )
        offsets.append((start, end))
    return tuple(offsets)


def _tokenize_once(
    tokenizer: FastOffsetTokenizer | Callable[..., Mapping[str, object]],
    text: str,
) -> FastTokenization:
    if getattr(tokenizer, "is_fast", None) is not True:
        raise TokenProjectionError(
            "FAST_TOKENIZER_REQUIRED",
            "ChemDFM-R projection requires a fast tokenizer with reliable offsets",
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
        raise TokenProjectionError(
            "TOKENIZATION_FAILED",
            "injected ChemDFM-R tokenizer failed",
            evidence={"exception_type": type(error).__name__},
        ) from error
    if not isinstance(encoded, Mapping):
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_SHAPE",
            "tokenizer output must be a mapping",
        )
    required = {
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "special_tokens_mask",
    }
    missing = tuple(sorted(required - set(encoded)))
    if missing:
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_MISSING",
            "tokenizer omitted required fast-offset fields",
            evidence={"fields": missing},
        )
    if "input_text" in encoded and encoded["input_text"] != text:
        raise TokenProjectionError(
            "TOKENIZER_TEXT_IDENTITY_MISMATCH",
            "tokenizer reports a different input text identity",
        )
    for truncation_field in ("overflowing_tokens", "overflow_to_sample_mapping"):
        if encoded.get(truncation_field):
            raise TokenProjectionError(
                "TOKENIZER_TRUNCATION_DETECTED",
                "tokenizer returned overflow/truncation metadata",
            )
    input_ids = _exact_int_tuple(encoded["input_ids"], name="input_ids")
    attention = _exact_int_tuple(encoded["attention_mask"], name="attention_mask")
    special = _exact_int_tuple(
        encoded["special_tokens_mask"], name="special_tokens_mask"
    )
    offsets = _offset_tuple(encoded["offset_mapping"])
    try:
        return FastTokenization(input_ids, attention, offsets, special)
    except (TypeError, ValueError) as error:
        raise TokenProjectionError(
            "TOKENIZER_OUTPUT_INVALID",
            "fast-tokenizer arrays violate the frozen unbatched contract",
            evidence={"exception_type": type(error).__name__},
        ) from error


def _validate_runtime_fingerprint(
    tokenizer: object,
    fingerprint: TokenizerFingerprint,
) -> None:
    if type(fingerprint) is not TokenizerFingerprint:
        raise TypeError("fingerprint must be a TokenizerFingerprint")
    if fingerprint.normalization_config.get("offset_unit") != "python_char":
        raise TokenProjectionError(
            "TOKENIZER_OFFSET_UNIT_UNRELIABLE",
            "tokenizer fingerprint must declare Python-string character offsets",
        )
    for field_name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        if field_name not in fingerprint.special_token_config:
            continue
        expected = fingerprint.special_token_config[field_name]
        missing = object()
        actual = getattr(tokenizer, field_name, missing)
        if actual is missing or actual != expected:
            raise TokenProjectionError(
                "TOKENIZER_FINGERPRINT_MISMATCH",
                "runtime special-token configuration differs from its fingerprint",
                evidence={"field": field_name},
            )


def _validate_offsets(
    text: str,
    tokenization: FastTokenization,
) -> None:
    previous_end = 0
    for index, ((start, end), attention, special) in enumerate(
        zip(
            tokenization.offset_mapping,
            tokenization.attention_mask,
            tokenization.special_tokens_mask,
            strict=True,
        )
    ):
        if end > len(text):
            raise TokenProjectionError(
                "TOKENIZER_OFFSET_OUT_OF_RANGE",
                "tokenizer offset exceeds Python-string character length",
                evidence={"token_index": index},
            )
        if start == end and (start, end) != (0, 0):
            raise TokenProjectionError(
                "NONCANONICAL_EMPTY_OFFSET",
                "empty tokenizer offsets must use the canonical (0, 0) pair",
                evidence={"token_index": index},
            )
        if attention == 0 and (start, end) != (0, 0):
            raise TokenProjectionError(
                "PADDING_OFFSET_NONEMPTY",
                "unattended tokenizer entries must use an empty offset",
                evidence={"token_index": index},
            )
        if special and (start, end) != (0, 0):
            raise TokenProjectionError(
                "SPECIAL_OFFSET_NONEMPTY",
                "tokenizer-added special tokens must use an empty offset",
                evidence={"token_index": index},
            )
        if (start, end) == (0, 0):
            if attention and not special:
                raise TokenProjectionError(
                    "ZERO_OFFSET_NON_SPECIAL",
                    "attended zero-offset tokens must be marked special",
                    evidence={"token_index": index},
                )
            continue
        if start < previous_end:
            raise TokenProjectionError(
                "TOKENIZER_OFFSETS_NON_MONOTONIC",
                "non-empty tokenizer offsets must be disjoint and monotonic",
                evidence={"token_index": index},
            )
        previous_end = end


def _overlap(start: int, end: int, span: CharSpan) -> int:
    return max(0, min(end, span.end) - max(start, span.start))


def _segment_ids(
    serialized: SerializedDetectorInput,
    tokenization: FastTokenization,
    *,
    pad_token_id: object,
) -> tuple[SegmentKind, ...]:
    segment_spans = tuple(
        (segment.segment_kind, CharSpan(segment.start, segment.end))
        for segment in serialized.segments
    )
    output: list[SegmentKind] = []
    for index, ((start, end), token_id, attention, special) in enumerate(
        zip(
            tokenization.offset_mapping,
            tokenization.input_ids,
            tokenization.attention_mask,
            tokenization.special_tokens_mask,
            strict=True,
        )
    ):
        if attention == 0 or (
            (start, end) == (0, 0)
            and type(pad_token_id) is int
            and token_id == pad_token_id
        ):
            output.append(SegmentKind.PADDING)
            continue
        if special or (start, end) == (0, 0):
            output.append(SegmentKind.SPECIAL)
            continue
        overlapping = tuple(
            kind for kind, span in segment_spans if _overlap(start, end, span) > 0
        )
        if len(overlapping) > 1:
            raise TokenProjectionError(
                "TOKEN_CROSSES_CONTENT_SEGMENTS",
                "one token offset overlaps multiple detector content segments",
                evidence={"token_index": index},
            )
        output.append(overlapping[0] if overlapping else SegmentKind.SPECIAL)
    return tuple(output)


def _normalize_annotations(
    value: CharAnnotationBuildResult | Sequence[CharAnnotation],
) -> tuple[CharAnnotation, ...]:
    if type(value) is CharAnnotationBuildResult:
        if value.unlocalized_omissions:
            raise TokenProjectionError(
                "UNLOCALIZED_OMISSION_NOT_PROJECTABLE",
                "pure omissions cannot be assigned to neighboring tokens",
                evidence={"count": len(value.unlocalized_omissions)},
            )
        annotations = value.annotations
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                "annotations must be CharAnnotationBuildResult or a sequence"
            )
        annotations = tuple(value)
    if any(type(annotation) is not CharAnnotation for annotation in annotations):
        raise TypeError("annotations must contain CharAnnotation values")
    ordered = tuple(
        sorted(
            annotations,
            key=lambda annotation: (
                annotation.literal_span.start,
                annotation.literal_span.end,
                annotation.span_id,
            ),
        )
    )
    span_ids = tuple(annotation.span_id for annotation in ordered)
    if len(span_ids) != len(set(span_ids)):
        raise TokenProjectionError(
            "DUPLICATE_ANNOTATION_ID",
            "character annotation IDs must be unique",
        )
    for previous, current in pairwise(ordered):
        if previous.literal_span.overlaps(current.literal_span):
            raise TokenProjectionError(
                "OVERLAPPING_LITERAL_ANNOTATIONS",
                "canonical literal annotations must not overlap",
            )
    return ordered


def _containing_segment(
    serialized: SerializedDetectorInput,
    span: CharSpan,
) -> SegmentKind:
    containing = tuple(
        segment.segment_kind
        for segment in serialized.segments
        if segment.start <= span.start and span.end <= segment.end
    )
    if len(containing) != 1:
        raise TokenProjectionError(
            "ANNOTATION_SEGMENT_MISMATCH",
            "annotation span must be contained by exactly one detector content segment",
        )
    return containing[0]


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


@dataclass(frozen=True, slots=True)
class TokenLabelSetWriter:
    """Tokenize one canonical detector input and write its derived labels."""

    tokenizer: FastOffsetTokenizer | Callable[..., Mapping[str, object]]
    tokenizer_fingerprint: TokenizerFingerprint
    projection_version: str = TOKEN_PROJECTION_VERSION

    def __post_init__(self) -> None:
        if not callable(self.tokenizer):
            raise TypeError("tokenizer must be callable")
        if type(self.tokenizer_fingerprint) is not TokenizerFingerprint:
            raise TypeError("tokenizer_fingerprint must be a TokenizerFingerprint")
        if self.projection_version != TOKEN_PROJECTION_VERSION:
            raise ValueError("unsupported token projection version")
        _validate_runtime_fingerprint(self.tokenizer, self.tokenizer_fingerprint)

    def write(
        self,
        serialized: SerializedDetectorInput,
        annotations: CharAnnotationBuildResult | Sequence[CharAnnotation],
        *,
        variant_label: VariantLabel,
        matched_target_span: CharSpan | None = None,
        rendered_example: RenderedExample | None = None,
    ) -> TokenLabelSet:
        if type(serialized) is not SerializedDetectorInput:
            raise TypeError("serialized must be a SerializedDetectorInput")
        if type(variant_label) is not VariantLabel:
            raise TypeError("variant_label must be a VariantLabel")
        if rendered_example is not None:
            if type(rendered_example) is not RenderedExample:
                raise TypeError("rendered_example must be a RenderedExample or None")
            coordinate_map = DetectorCoordinateMap.from_rendered(
                rendered_example, serialized
            )
            annotations = rebase_char_annotations(
                rendered_example, serialized, annotations
            )
            if matched_target_span is not None:
                matched_target_span = coordinate_map.rebase_any_span(
                    matched_target_span
                )
        normalized_annotations = _normalize_annotations(annotations)
        if variant_label is VariantLabel.FAITHFUL and normalized_annotations:
            raise TokenProjectionError(
                "FAITHFUL_CONTROL_HAS_ANNOTATIONS",
                "faithful controls must project all-positive masks to zero",
            )
        if variant_label is VariantLabel.HALLUCINATED and not normalized_annotations:
            raise TokenProjectionError(
                "HALLUCINATED_SPAN_MISSING",
                "hallucinated records require at least one localized char annotation",
            )
        if matched_target_span is not None:
            if type(matched_target_span) is not CharSpan:
                raise TypeError("matched_target_span must be a CharSpan or None")
            if variant_label is not VariantLabel.FAITHFUL:
                raise TokenProjectionError(
                    "MATCHED_TARGET_ON_HALLUCINATED",
                    "matched_target_span is reserved for faithful controls",
                )
            validate_char_span(serialized.text, matched_target_span)
            if _containing_segment(serialized, matched_target_span) not in (
                _EVALUATED_SEGMENTS
            ):
                raise TokenProjectionError(
                    "MATCHED_TARGET_NOT_EVALUATED",
                    "matched target must belong to reasoning or final answer",
                )

        for annotation in normalized_annotations:
            validate_char_span(serialized.text, annotation.literal_span)
            validate_char_span(serialized.text, annotation.claim_span)
            segment = _containing_segment(serialized, annotation.literal_span)
            claim_segment = _containing_segment(serialized, annotation.claim_span)
            if segment is not annotation.component:
                raise TokenProjectionError(
                    "ANNOTATION_COMPONENT_MISMATCH",
                    "annotation component differs from serialized detector segment",
                    evidence={"span_id": annotation.span_id},
                )
            if claim_segment is not segment:
                raise TokenProjectionError(
                    "ANNOTATION_CLAIM_COMPONENT_MISMATCH",
                    "annotation claim and literal spans must share one detector segment",
                    evidence={"span_id": annotation.span_id},
                )

        tokenization = _tokenize_once(self.tokenizer, serialized.text)
        _validate_offsets(serialized.text, tokenization)
        segment_ids = _segment_ids(
            serialized,
            tokenization,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        count = len(tokenization.input_ids)
        semantic_sets = [set[HallucinationType]() for _ in range(count)]
        edit_sets = [set[EditErrorSubtype]() for _ in range(count)]
        role_sets = [set[CausalRole]() for _ in range(count)]
        intersections: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        boundary = [0] * count
        covered_span_ids: set[str] = set()

        for annotation in normalized_annotations:
            for index, (start, end) in enumerate(tokenization.offset_mapping):
                overlap = _overlap(start, end, annotation.literal_span)
                if overlap <= 0:
                    continue
                if segment_ids[index] not in _EVALUATED_SEGMENTS:
                    continue
                covered_span_ids.add(annotation.span_id)
                semantic_sets[index].update(annotation.semantic_types)
                edit_sets[index].update(annotation.edit_subtypes)
                if annotation.causal_role is not None:
                    role_sets[index].add(annotation.causal_role)
                intersections[index].append(
                    (
                        max(start, annotation.literal_span.start),
                        min(end, annotation.literal_span.end),
                    )
                )
                if (
                    start < annotation.literal_span.start < end
                    or start < annotation.literal_span.end < end
                ):
                    boundary[index] = 1

        missing_coverage = tuple(
            annotation.span_id
            for annotation in normalized_annotations
            if annotation.span_id not in covered_span_ids
        )
        if missing_coverage:
            raise TokenProjectionError(
                "POSITIVE_SPAN_UNCOVERED",
                "every positive character span must overlap an evaluated token",
                evidence={"span_ids": missing_coverage},
            )
        for index in range(count):
            if len(role_sets[index]) > 1:
                raise TokenProjectionError(
                    "CAUSAL_ROLE_TOKEN_COLLISION",
                    "one tokenizer token cannot represent multiple causal roles",
                    evidence={"token_index": index},
                )
            if (
                HallucinationType.UNVERIFIABLE in semantic_sets[index]
                and len(semantic_sets[index]) > 1
            ):
                raise TokenProjectionError(
                    "UNVERIFIABLE_TOKEN_COLLISION",
                    "UNVERIFIABLE cannot share a token with adjudicated semantics",
                    evidence={"token_index": index},
                )

        semantic_masks = {
            label: tuple(int(label in values) for values in semantic_sets)
            for label in HallucinationType
        }
        edit_masks = {
            label: tuple(int(label in values) for values in edit_sets)
            for label in EditErrorSubtype
        }
        role_masks = {
            label: tuple(int(label in values) for values in role_sets)
            for label in CausalRole
        }
        hallucination_core = tuple(
            int(
                HallucinationType.CONTRADICTION in semantic_sets[index]
                or HallucinationType.UNSUPPORTED in semantic_sets[index]
            )
            for index in range(count)
        )
        error_any = tuple(
            int(
                any(
                    label is not HallucinationType.UNVERIFIABLE
                    for label in semantic_sets[index]
                )
            )
            for index in range(count)
        )
        local_falsehood = tuple(
            int(bool(role_sets[index] & _LOCAL_FALSEHOOD_ROLES))
            for index in range(count)
        )
        off_task_branch = tuple(
            int(CausalRole.PROPAGATED_CONDITIONAL in role_sets[index])
            for index in range(count)
        )
        fractions = tuple(
            0.0 if end == start else _union_length(intersections[index]) / (end - start)
            for index, (start, end) in enumerate(tokenization.offset_mapping)
        )
        reasoning_mask = tuple(
            int(segment is SegmentKind.REASONING) for segment in segment_ids
        )
        answer_mask = tuple(
            int(segment is SegmentKind.FINAL_ANSWER) for segment in segment_ids
        )
        evaluation_mask = tuple(
            int(
                segment in _EVALUATED_SEGMENTS
                and tokenization.attention_mask[index] == 1
            )
            for index, segment in enumerate(segment_ids)
        )

        try:
            return TokenLabelSet(
                activation_alignment=ACTIVATION_ALIGNMENT,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                serialized_text_sha256=serialized.sha256,
                input_ids=tokenization.input_ids,
                attention_mask=tokenization.attention_mask,
                offset_mapping=tokenization.offset_mapping,
                segment_ids=segment_ids,
                evaluation_mask=evaluation_mask,
                hallucination_core_mask=hallucination_core,
                error_any_mask=error_any,
                semantic_type_masks=semantic_masks,
                edit_subtype_masks=edit_masks,
                causal_role_masks=role_masks,
                local_falsehood_mask=local_falsehood,
                off_task_branch_mask=off_task_branch,
                reasoning_mask=reasoning_mask,
                answer_mask=answer_mask,
                boundary_ambiguous_mask=tuple(boundary),
                error_char_fraction=fractions,
                matched_target_span=matched_target_span,
            )
        except (TypeError, ValueError) as error:
            raise TokenProjectionError(
                "TOKEN_LABEL_SET_INVALID",
                "projected arrays violate the frozen TokenLabelSet contract",
                evidence={"exception_type": type(error).__name__},
            ) from error

    project = write


ChemDFMRTokenProjector = TokenLabelSetWriter


def project_char_annotations(
    serialized: SerializedDetectorInput,
    annotations: CharAnnotationBuildResult | Sequence[CharAnnotation],
    *,
    tokenizer: FastOffsetTokenizer | Callable[..., Mapping[str, object]],
    tokenizer_fingerprint: TokenizerFingerprint,
    variant_label: VariantLabel,
    matched_target_span: CharSpan | None = None,
    rendered_example: RenderedExample | None = None,
) -> TokenLabelSet:
    """Convenience entry point using one injected ChemDFM-R tokenizer."""

    return TokenLabelSetWriter(tokenizer, tokenizer_fingerprint).write(
        serialized,
        annotations,
        variant_label=variant_label,
        matched_target_span=matched_target_span,
        rendered_example=rendered_example,
    )


__all__ = [
    "ACTIVATION_ALIGNMENT",
    "TOKEN_PROJECTION_VERSION",
    "ChemDFMRTokenProjector",
    "DetectorCoordinateMap",
    "FastOffsetTokenizer",
    "FastTokenization",
    "TokenLabelSetWriter",
    "TokenProjectionError",
    "project_char_annotations",
    "rebase_char_annotations",
]
