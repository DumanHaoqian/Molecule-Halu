"""Direct application of unified multi-point hallucination plans."""

from .unified import HallucinationInjectionError, UnifiedHallucinationInjector
from .propagation import PropagationError, audit_edges, evaluate_editing_edge

__all__ = [
    "HallucinationInjectionError",
    "PropagationError",
    "UnifiedHallucinationInjector",
    "audit_edges",
    "evaluate_editing_edge",
]
