"""Future reaction-prediction family interface."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from molhallulens.domain import StateSchema

from ..base import Perturbator


class ReactionPredictionPerturbator(Perturbator[Any], ABC):
    """Future abstract ``rxn_pred`` family boundary."""

    family: ClassVar[str] = "rxn_pred"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


__all__ = ["ReactionPredictionPerturbator"]
