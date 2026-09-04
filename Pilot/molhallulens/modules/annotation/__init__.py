"""Character-span annotation for unified hallucination records."""

from .spans import (
    AnnotatedHallucination,
    ControlSpan,
    HallucinationSpan,
    UnifiedHallucinationAnnotator,
)

__all__ = [
    "AnnotatedHallucination",
    "ControlSpan",
    "HallucinationSpan",
    "UnifiedHallucinationAnnotator",
]
