"""Locked-FORMAL text realization with a context-aware Poe agent."""

from .poe_agent import (
    AffectedNodeClaim,
    FORMAL_MARKER,
    HALLU_MARKER_PATTERN,
    POE_RENDERER_VERSION,
    PoeRewriteRequest,
    PoeRewriteResult,
    RequiredHallucinationOccurrence,
    StepRewriteMode,
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
from .pairing import (
    MatchedNegativeTextBuilder,
    MatchedRenderedPair,
    PairAlignment,
    StepPairAlignment,
)

__all__ = [
    "AffectedNodeClaim",
    "DeterministicTextRenderer",
    "FORMAL_MARKER",
    "HALLU_MARKER_PATTERN",
    "MatchedNegativeTextBuilder",
    "MatchedRenderedPair",
    "POE_RENDERER_VERSION",
    "PoeRewriteRequest",
    "PoeRewriteResult",
    "RequiredHallucinationOccurrence",
    "StepRewriteMode",
    "PoeStepRewriteInput",
    "PoeStepTextAgent",
    "PoeTextRealizationError",
    "PoeTextRenderer",
    "PairAlignment",
    "StepPairAlignment",
    "build_poe_rewrite_request",
    "parse_hallucination_markers",
    "strip_hallucination_markers",
    "validate_rewritten_step_text",
]
