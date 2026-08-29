"""Canonical character annotations and token projections."""

from .char_spans import (
    CharSpan,
    MentionSpan,
    span_text,
    validate_char_span,
    validate_mention_spans,
    validate_non_overlapping_spans,
)

__all__ = [
    "CharSpan",
    "MentionSpan",
    "span_text",
    "validate_char_span",
    "validate_mention_spans",
    "validate_non_overlapping_spans",
]
