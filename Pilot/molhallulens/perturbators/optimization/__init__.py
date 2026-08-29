"""Future molecular-optimization family interface."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from molhallulens.domain import StateSchema

from ..base import Perturbator


class MolecularOptimizationPerturbator(Perturbator[Any], ABC):
    """Future abstract ``mol_opt`` family boundary."""

    family: ClassVar[str] = "mol_opt"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


__all__ = ["MolecularOptimizationPerturbator"]
