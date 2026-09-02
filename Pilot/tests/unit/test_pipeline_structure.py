"""The package keeps one explicit and stable A-to-H module chain."""

from dataclasses import dataclass
from typing import Any

import pytest

from molhallulens.pipeline import PIPELINE_ORDER, PipelineStage, SequentialPipeline


@dataclass(frozen=True)
class _Module:
    stage: PipelineStage

    def run(self, value: Any) -> Any:
        return (*value, self.stage)


def test_pipeline_executes_every_module_in_declared_order() -> None:
    pipeline = SequentialPipeline.from_modules(tuple(_Module(stage) for stage in PIPELINE_ORDER))
    assert pipeline.run(()) == PIPELINE_ORDER


def test_pipeline_rejects_missing_or_reordered_modules() -> None:
    with pytest.raises(ValueError, match="exact A-to-H"):
        SequentialPipeline.from_modules(tuple(_Module(stage) for stage in reversed(PIPELINE_ORDER)))
