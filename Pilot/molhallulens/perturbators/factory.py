"""Normalized-record factory for stable perturbator family/subtask types."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

from molhallulens.domain import EditingSubtask, EditTruth, TaskFamily, TaskRecord

from .base import (
    CandidateEngine,
    LabelProjector,
    Perturbator,
    PropagationEngine,
    TraceRenderer,
    ValidatorChain,
)
from .editing import (
    AdditionPerturbator,
    DeletionPerturbator,
    SubstitutionPerturbator,
)


class PerturbatorFactoryError(ValueError):
    """Raised when no perturbator type is registered for a normalized record."""

    def __init__(self, *, family: object, subtask: object) -> None:
        self.family = family
        self.subtask = subtask
        super().__init__(
            "unsupported normalized perturbator key: "
            f"family={family!r}, subtask={subtask!r}"
        )


_PERTURBATOR_TYPES = MappingProxyType(
    {
        (TaskFamily.MOLECULE_EDITING, EditingSubtask.ADD): AdditionPerturbator,
        (TaskFamily.MOLECULE_EDITING, EditingSubtask.DELETE): DeletionPerturbator,
        (TaskFamily.MOLECULE_EDITING, EditingSubtask.SUBSTITUTE): SubstitutionPerturbator,
    }
)


class PerturbatorFactory:
    """Construct from authoritative enums; never re-parse source task names."""

    @classmethod
    def from_record(
        cls,
        record: TaskRecord,
        *,
        candidate_engine: CandidateEngine[EditTruth],
        propagator: PropagationEngine[EditTruth],
        renderer: TraceRenderer[EditTruth],
        validators: ValidatorChain[EditTruth],
        label_projector: LabelProjector[EditTruth],
    ) -> Perturbator[EditTruth]:
        if type(record) is not TaskRecord:
            raise TypeError("PerturbatorFactory record must be TaskRecord")
        # Re-run the authoritative TaskRecord invariants so a caller cannot
        # route an object that was corrupted after frozen construction.  The
        # factory still never owns or duplicates source-task normalization.
        replace(record)
        key = (record.family, record.normalized_subtask)
        perturbator_type: Any = _PERTURBATOR_TYPES.get(key)
        if perturbator_type is None:
            raise PerturbatorFactoryError(family=key[0], subtask=key[1])
        return perturbator_type(
            candidate_engine=candidate_engine,
            propagator=propagator,
            renderer=renderer,
            validators=validators,
            label_projector=label_projector,
        )


__all__ = ["PerturbatorFactory", "PerturbatorFactoryError"]
