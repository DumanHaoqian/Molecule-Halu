"""Future molecule-understanding family interface."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from molhallulens.domain import StateSchema

from ..base import Perturbator


class MoleculeUnderstandingPerturbator(Perturbator[Any], ABC):
    """Future abstract ``mol_und`` family boundary."""

    family: ClassVar[str] = "mol_und"

    @abstractmethod
    def state_schema(self) -> StateSchema:
        raise NotImplementedError


__all__ = ["MoleculeUnderstandingPerturbator"]
