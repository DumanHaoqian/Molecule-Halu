"""Canonical character annotations and token projections."""

from .char_annotations import (
    CHAR_ANNOTATION_BUILDER_VERSION,
    CharAnnotationBuilder,
    CharAnnotationBuildError,
    CharAnnotationBuildResult,
    EventAnnotationLink,
    UnlocalizedOmission,
    build_char_annotations,
    derive_evidence_relations,
    is_pure_omission,
)
from .char_spans import (
    CharSpan,
    MentionSpan,
    span_text,
    validate_char_span,
    validate_mention_spans,
    validate_non_overlapping_spans,
)

__all__ = [
    "CHAR_ANNOTATION_BUILDER_VERSION",
    "CharAnnotationBuildError",
    "CharAnnotationBuildResult",
    "CharAnnotationBuilder",
    "CharSpan",
    "EventAnnotationLink",
    "MentionSpan",
    "UnlocalizedOmission",
    "build_char_annotations",
    "derive_evidence_relations",
    "is_pure_omission",
    "span_text",
    "validate_char_span",
    "validate_mention_spans",
    "validate_non_overlapping_spans",
]
