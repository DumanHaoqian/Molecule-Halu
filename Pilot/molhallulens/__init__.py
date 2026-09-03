"""MolHalluLens molecular hallucination dataset builder."""

from __future__ import annotations

from .pipeline import (
    AnnotationModule,
    ErrorInjectionModule,
    ErrorPlanningModule,
    IngestionModule,
    ReferenceModule,
    ReleaseModule,
    SequentialPipeline,
    TextRealizationModule,
    TrajectoryModule,
)

__all__ = [
    "AnnotationModule",
    "ErrorInjectionModule",
    "ErrorPlanningModule",
    "IngestionModule",
    "ReferenceModule",
    "ReleaseModule",
    "SequentialPipeline",
    "TextRealizationModule",
    "TrajectoryModule",
    "__version__",
]

__version__ = "0.1.0"
