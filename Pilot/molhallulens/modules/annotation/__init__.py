"""Canonical character annotations and token projections."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from molhallulens.modules.text_realization.spans import (
    CharSpan,
    MentionSpan,
    span_text,
    validate_char_span,
    validate_mention_spans,
    validate_non_overlapping_spans,
)

if TYPE_CHECKING:
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
    from .token_projection import (
        ACTIVATION_ALIGNMENT,
        TOKEN_PROJECTION_VERSION,
        ChemDFMRTokenProjector,
        DetectorCoordinateMap,
        FastOffsetTokenizer,
        FastTokenization,
        TokenLabelSetWriter,
        TokenProjectionError,
        project_char_annotations,
        rebase_char_annotations,
    )

_LAZY_EXPORTS = {
    **{
        name: ".char_annotations"
        for name in (
            "CHAR_ANNOTATION_BUILDER_VERSION",
            "CharAnnotationBuildError",
            "CharAnnotationBuildResult",
            "CharAnnotationBuilder",
            "EventAnnotationLink",
            "UnlocalizedOmission",
            "build_char_annotations",
            "derive_evidence_relations",
            "is_pure_omission",
        )
    },
    **{
        name: ".token_projection"
        for name in (
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
        )
    },
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = [
    "ACTIVATION_ALIGNMENT",
    "CHAR_ANNOTATION_BUILDER_VERSION",
    "TOKEN_PROJECTION_VERSION",
    "CharAnnotationBuildError",
    "CharAnnotationBuildResult",
    "CharAnnotationBuilder",
    "CharSpan",
    "ChemDFMRTokenProjector",
    "DetectorCoordinateMap",
    "EventAnnotationLink",
    "FastOffsetTokenizer",
    "FastTokenization",
    "MentionSpan",
    "TokenLabelSetWriter",
    "TokenProjectionError",
    "UnlocalizedOmission",
    "build_char_annotations",
    "derive_evidence_relations",
    "is_pure_omission",
    "project_char_annotations",
    "rebase_char_annotations",
    "span_text",
    "validate_char_span",
    "validate_mention_spans",
    "validate_non_overlapping_spans",
]
