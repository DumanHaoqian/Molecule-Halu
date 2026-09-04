"""Locked-FORMAL text realization with a context-aware Poe agent."""

from .poe_agent import (
    FORMAL_MARKER,
    HALLU_MARKER_PATTERN,
    POE_RENDERER_VERSION,
    PoeRewriteRequest,
    PoeRewriteResult,
    RequiredHallucinationOccurrence,
    PoeStepRewriteInput,
    PoeStepTextAgent,
    PoeTextRealizationError,
    parse_hallucination_markers,
    strip_hallucination_markers,
    validate_rewritten_step_text,
)
from .renderer import (
    DeterministicTextRenderer,
    PoeTextRenderer,
    build_poe_rewrite_request,
)

__all__ = [
    "DeterministicTextRenderer",
    "FORMAL_MARKER",
    "HALLU_MARKER_PATTERN",
    "POE_RENDERER_VERSION",
    "PoeRewriteRequest",
    "PoeRewriteResult",
    "RequiredHallucinationOccurrence",
    "PoeStepRewriteInput",
    "PoeStepTextAgent",
    "PoeTextRealizationError",
    "PoeTextRenderer",
    "build_poe_rewrite_request",
    "parse_hallucination_markers",
    "strip_hallucination_markers",
    "validate_rewritten_step_text",
]
