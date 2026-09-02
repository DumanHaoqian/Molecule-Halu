"""Small, explicit A-to-H orchestration boundary for dataset construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class PipelineStage(StrEnum):
    INGESTION = "A_INGESTION"
    REFERENCE = "B_REFERENCE"
    ERROR_PLANNING = "C_ERROR_PLANNING"
    ERROR_INJECTION = "D_ERROR_INJECTION"
    TRAJECTORY = "E_TRAJECTORY"
    TEXT_REALIZATION = "F_TEXT_REALIZATION"
    ANNOTATION = "G_ANNOTATION"
    RELEASE = "H_RELEASE"


PIPELINE_ORDER = tuple(PipelineStage)


@runtime_checkable
class PipelineModule(Protocol):
    """One independently testable transformation in the linear pipeline."""

    stage: PipelineStage

    def run(self, value: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SequentialPipeline:
    """Execute exactly one implementation of every A-to-H stage in order."""

    modules: tuple[PipelineModule, ...]

    def __post_init__(self) -> None:
        modules = tuple(self.modules)
        if tuple(module.stage for module in modules) != PIPELINE_ORDER:
            raise ValueError(
                "pipeline modules must follow the exact A-to-H stage order"
            )
        object.__setattr__(self, "modules", modules)

    @classmethod
    def from_modules(cls, modules: Sequence[PipelineModule]) -> SequentialPipeline:
        return cls(tuple(modules))

    def run(self, source: Any) -> Any:
        value = source
        for module in self.modules:
            value = module.run(value)
        return value


__all__ = [
    "PIPELINE_ORDER",
    "PipelineModule",
    "PipelineStage",
    "SequentialPipeline",
]
