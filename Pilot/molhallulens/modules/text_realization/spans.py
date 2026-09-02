"""Canonical Python-string character spans for rendered trace mentions.

Offsets in this module index Unicode code points in a Python :class:`str`, not
UTF-8 bytes.  Rendering code creates them while appending text; these helpers
only validate already-bound occurrences and never recover locations by
searching rendered text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from molhallulens.core import MutationTargetKind, SegmentKind
from molhallulens.core.labels import CharSpan


def span_text(text: str, span: CharSpan) -> str:
    """Return the exact code-point slice after strict bounds validation."""

    validate_char_span(text, span)
    return text[span.start : span.end]


def validate_char_span(
    text: str,
    span: CharSpan,
    *,
    expected_text: str | None = None,
) -> None:
    """Validate one non-empty half-open span against a Python string."""

    if type(text) is not str:
        raise TypeError("span text must be a string")
    if type(span) is not CharSpan:
        raise TypeError("span must be a CharSpan")
    if span.end > len(text):
        raise ValueError("character span falls outside rendered text")
    if expected_text is not None:
        if type(expected_text) is not str:
            raise TypeError("expected_text must be a string or None")
        if not expected_text:
            raise ValueError("expected_text cannot be empty")
        if text[span.start : span.end] != expected_text:
            raise ValueError("character span does not cover its expected text")


def validate_non_overlapping_spans(
    spans: Iterable[CharSpan],
    *,
    name: str = "spans",
) -> tuple[CharSpan, ...]:
    """Return spans in canonical order and reject any overlap."""

    if type(name) is not str or not name:
        raise ValueError("span collection name must be non-empty text")
    materialized = tuple(spans)
    if any(type(span) is not CharSpan for span in materialized):
        raise TypeError(f"{name} must contain CharSpan values")
    ordered = tuple(sorted(materialized, key=lambda span: (span.start, span.end)))
    for previous, current in pairwise(ordered):
        if previous.overlaps(current):
            raise ValueError(f"{name} must not overlap")
    return ordered


@dataclass(frozen=True, slots=True)
class MentionSpan:
    """One rendered occurrence linked to one atomic claim and DAG target.

    ``claim_id`` plus ``(target_kind, state_or_edge_id)`` is the stable join
    key needed by the later annotation stage to recover the matching mutation
    event, its root event, and causal role.  No correctness or label axis is
    stored or inferred here.
    """

    mention_id: str
    claim_id: str
    state_or_edge_id: str
    target_kind: MutationTargetKind
    component: SegmentKind
    step_index: int | None
    literal_text: str
    literal_span: CharSpan
    claim_span: CharSpan

    def __post_init__(self) -> None:
        for value, name in (
            (self.mention_id, "mention_id"),
            (self.claim_id, "claim_id"),
            (self.state_or_edge_id, "state_or_edge_id"),
            (self.literal_text, "literal_text"),
        ):
            if type(value) is not str:
                raise TypeError(f"MentionSpan {name} must be a string")
            if not value:
                raise ValueError(f"MentionSpan {name} cannot be empty")
        if "\x00" in self.literal_text or "\r" in self.literal_text:
            raise ValueError("MentionSpan literal_text must be canonical NUL-free text")
        if type(self.target_kind) is not MutationTargetKind:
            raise TypeError("MentionSpan target_kind must be a MutationTargetKind")
        if type(self.component) is not SegmentKind:
            raise TypeError("MentionSpan component must be a SegmentKind")
        if self.component not in {SegmentKind.REASONING, SegmentKind.FINAL_ANSWER}:
            raise ValueError("trace mentions must belong to reasoning or final answer")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index <= 0
        ):
            raise ValueError("MentionSpan step_index must be positive or None")
        if self.component is SegmentKind.REASONING and self.step_index is None:
            raise ValueError("reasoning mentions require a step_index")
        if self.component is SegmentKind.FINAL_ANSWER and self.step_index is not None:
            raise ValueError("final-answer mentions cannot have a step_index")
        if (
            type(self.literal_span) is not CharSpan
            or type(self.claim_span) is not CharSpan
        ):
            raise TypeError("MentionSpan offsets must be CharSpan values")
        if not (
            self.claim_span.start <= self.literal_span.start
            and self.literal_span.end <= self.claim_span.end
        ):
            raise ValueError("literal_span must be contained by claim_span")

    def validate_against(self, text: str) -> None:
        """Validate bounds, literal content, and claim containment in ``text``."""

        validate_char_span(text, self.claim_span)
        validate_char_span(text, self.literal_span, expected_text=self.literal_text)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "mention_id": self.mention_id,
            "claim_id": self.claim_id,
            "state_or_edge_id": self.state_or_edge_id,
            "target_kind": self.target_kind.value,
            "component": self.component.value,
            "step_index": self.step_index,
            "literal_text": self.literal_text,
            "literal_span": {
                "start": self.literal_span.start,
                "end": self.literal_span.end,
            },
            "claim_span": {
                "start": self.claim_span.start,
                "end": self.claim_span.end,
            },
        }


def validate_mention_spans(
    text: str,
    mentions: Iterable[MentionSpan],
) -> tuple[MentionSpan, ...]:
    """Validate and canonically order all occurrence spans for one rendering.

    Literal occurrences must be disjoint.  Multiple literals may share one
    claim interval, but different atomic claims cannot overlap or nest.
    """

    if type(text) is not str:
        raise TypeError("rendered text must be a string")
    materialized = tuple(mentions)
    if any(type(mention) is not MentionSpan for mention in materialized):
        raise TypeError("mentions must contain MentionSpan values")
    mention_ids = tuple(mention.mention_id for mention in materialized)
    if len(mention_ids) != len(set(mention_ids)):
        raise ValueError("mention IDs must be unique per rendered example")
    for mention in materialized:
        mention.validate_against(text)

    ordered = tuple(
        sorted(
            materialized,
            key=lambda mention: (
                mention.literal_span.start,
                mention.literal_span.end,
                mention.mention_id,
            ),
        )
    )
    validate_non_overlapping_spans(
        (mention.literal_span for mention in ordered),
        name="mention literal spans",
    )

    claims: dict[str, CharSpan] = {}
    for mention in ordered:
        existing = claims.setdefault(mention.claim_id, mention.claim_span)
        if existing != mention.claim_span:
            raise ValueError("one claim_id must resolve to exactly one claim span")
    claim_items = tuple(sorted(claims.items(), key=lambda item: (item[1], item[0])))
    for (_, previous), (_, current) in pairwise(claim_items):
        if previous.overlaps(current):
            raise ValueError("distinct claim spans must not overlap")
    return ordered


__all__ = [
    "CharSpan",
    "MentionSpan",
    "span_text",
    "validate_char_span",
    "validate_mention_spans",
    "validate_non_overlapping_spans",
]
