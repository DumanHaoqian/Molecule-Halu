"""Abstract boundaries for task families not implemented by this pilot."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from molhallulens.core import StateSchema

from molhallulens.orchestration import Perturbator


class MolecularOptimizationPerturbator(Perturbator[Any], ABC):
    family: ClassVar[str] = "mol_opt"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


class MoleculeUnderstandingPerturbator(Perturbator[Any], ABC):
    family: ClassVar[str] = "mol_und"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


class ReactionPredictionPerturbator(Perturbator[Any], ABC):
    family: ClassVar[str] = "rxn_pred"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


__all__ = [
    "MolecularOptimizationPerturbator",
    "MoleculeUnderstandingPerturbator",
    "ReactionPredictionPerturbator",
]
