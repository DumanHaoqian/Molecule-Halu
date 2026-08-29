"""Canonical serialization for the detector-visible prompt."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final

from molhallulens.domain import DetectorInput, SegmentKind


DETECTOR_PROMPT_VERSION: Final = "detector_prompt_v1"
DETECTOR_FIELD_ORDER: Final = (
    "indexed_smiles",
    "instruction",
    "reasoning_chain",
    "final_answer",
)
DETECTOR_DELIMITERS: Final = MappingProxyType(
    {
        "indexed_smiles": "<MOLECULE>",
        "instruction": "<INSTRUCTION>",
        "reasoning_chain": "<REASONING>",
        "final_answer": "<FINAL_ANSWER>",
    }
)
_SEGMENT_KINDS: Final = MappingProxyType(
    {
        "indexed_smiles": SegmentKind.SOURCE,
        "instruction": SegmentKind.INSTRUCTION,
        "reasoning_chain": SegmentKind.REASONING,
        "final_answer": SegmentKind.FINAL_ANSWER,
    }
)
_RESERVED_DELIMITERS: Final = frozenset(DETECTOR_DELIMITERS.values())


def _normalize_visible_field(value: str, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} cannot contain a NUL character")
    injected = sorted(marker for marker in _RESERVED_DELIMITERS if marker in normalized)
    if injected:
        raise ValueError(
            f"{field_name} contains reserved detector delimiter(s): {injected!r}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class DetectorPromptSegment:
    """Half-open normalized Python ``str`` code-point span for one field."""

    field_name: str
    segment_kind: SegmentKind
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.field_name) is not str:
            raise TypeError("detector segment field_name must be a string")
        if self.field_name not in DETECTOR_FIELD_ORDER:
            raise ValueError(f"unknown detector field: {self.field_name!r}")
        if type(self.segment_kind) is not SegmentKind:
            raise TypeError("detector segment segment_kind must be a SegmentKind")
        if self.segment_kind is not _SEGMENT_KINDS[self.field_name]:
            raise ValueError("segment kind does not match detector field")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("detector segment offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("detector segment must be a non-empty half-open span")


def _normalize_detector_input(detector_input: DetectorInput) -> DetectorInput:
    if type(detector_input) is not DetectorInput:
        raise TypeError("detector_input must be a DetectorInput")
    actual_fields = tuple(DetectorInput.__dataclass_fields__)
    if actual_fields != DETECTOR_FIELD_ORDER:
        raise RuntimeError(
            "DetectorInput fields drifted from the frozen detector serializer contract"
        )
    return DetectorInput(
        **{
            field_name: _normalize_visible_field(
                getattr(detector_input, field_name), field_name=field_name
            )
            for field_name in DETECTOR_FIELD_ORDER
        }
    )


def _render_canonical(
    detector_input: DetectorInput,
) -> tuple[str, tuple[DetectorPromptSegment, ...]]:
    parts: list[str] = []
    segments: list[DetectorPromptSegment] = []
    cursor = 0
    for index, field_name in enumerate(DETECTOR_FIELD_ORDER):
        value = getattr(detector_input, field_name)
        if index:
            separator = "\n\n"
            parts.append(separator)
            cursor += len(separator)
        prefix = f"{DETECTOR_DELIMITERS[field_name]}\n"
        parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        parts.append(value)
        cursor += len(value)
        segments.append(
            DetectorPromptSegment(
                field_name=field_name,
                segment_kind=_SEGMENT_KINDS[field_name],
                start=start,
                end=cursor,
            )
        )
    return "".join(parts), tuple(segments)


@dataclass(frozen=True, slots=True)
class SerializedDetectorInput:
    """Canonical detector text, digest, and exact visible-field spans."""

    detector_input: DetectorInput
    text: str
    sha256: str
    segments: tuple[DetectorPromptSegment, ...]
    template_version: str = DETECTOR_PROMPT_VERSION

    def __post_init__(self) -> None:
        normalized_input = _normalize_detector_input(self.detector_input)
        if normalized_input != self.detector_input:
            raise ValueError("detector_input must use canonical LF line endings")
        if type(self.text) is not str:
            raise TypeError("serialized detector text must be a string")
        if not self.text:
            raise ValueError("serialized detector text must be non-empty")
        if type(self.sha256) is not str:
            raise TypeError("serialized detector sha256 must be a string")
        if type(self.segments) is not tuple:
            raise TypeError("serialized detector segments must be a tuple")
        if any(type(segment) is not DetectorPromptSegment for segment in self.segments):
            raise TypeError("segments must contain DetectorPromptSegment values")
        if type(self.template_version) is not str:
            raise TypeError("detector prompt template_version must be a string")
        expected_text, expected_segments = _render_canonical(normalized_input)
        if self.text != expected_text:
            raise ValueError("serialized detector text does not match the canonical template")
        if self.segments != expected_segments:
            raise ValueError("serialized detector segments do not match the canonical template")
        expected_hash = sha256(expected_text.encode("utf-8")).hexdigest()
        if self.sha256 != expected_hash:
            raise ValueError("serialized detector sha256 does not match text")
        if self.template_version != DETECTOR_PROMPT_VERSION:
            raise ValueError("unknown detector prompt template version")

    @property
    def field_order(self) -> tuple[str, ...]:
        return DETECTOR_FIELD_ORDER


class DetectorPromptSerializer:
    """Serialize exactly the four detector-visible fields in frozen order."""

    field_order = DETECTOR_FIELD_ORDER
    delimiters = DETECTOR_DELIMITERS
    template_version = DETECTOR_PROMPT_VERSION

    def serialize(
        self,
        *,
        indexed_smiles: str,
        instruction: str,
        reasoning_chain: str,
        final_answer: str,
    ) -> SerializedDetectorInput:
        """Return canonical UTF-8 text without accepting any oracle field."""

        values = {
            "indexed_smiles": _normalize_visible_field(
                indexed_smiles, field_name="indexed_smiles"
            ),
            "instruction": _normalize_visible_field(
                instruction, field_name="instruction"
            ),
            "reasoning_chain": _normalize_visible_field(
                reasoning_chain, field_name="reasoning_chain"
            ),
            "final_answer": _normalize_visible_field(
                final_answer, field_name="final_answer"
            ),
        }
        detector_input = DetectorInput(**values)
        return self.serialize_input(detector_input)

    def serialize_input(self, detector_input: DetectorInput) -> SerializedDetectorInput:
        """Serialize the already-isolated domain view."""

        normalized_input = _normalize_detector_input(detector_input)
        text, segments = _render_canonical(normalized_input)
        return SerializedDetectorInput(
            detector_input=normalized_input,
            text=text,
            sha256=sha256(text.encode("utf-8")).hexdigest(),
            segments=segments,
        )


__all__ = [
    "DETECTOR_DELIMITERS",
    "DETECTOR_FIELD_ORDER",
    "DETECTOR_PROMPT_VERSION",
    "DetectorPromptSegment",
    "DetectorPromptSerializer",
    "SerializedDetectorInput",
]
