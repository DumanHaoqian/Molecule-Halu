"""Required-composition deterministic RDKit candidate source."""

from __future__ import annotations

from dataclasses import dataclass

from molhallulens.core import CandidateSourceType

from .core import (
    CandidateProposal,
    CandidateRejectCode,
    CandidateRequest,
    CandidateSourceError,
    ProposalFunction,
)


@dataclass(frozen=True, slots=True)
class RDKitCandidateSource:
    """Adapt an operator-owned RDKit enumerator to the T018 source contract."""

    proposer: ProposalFunction
    source_type = CandidateSourceType.RDKIT

    def __post_init__(self) -> None:
        if not callable(self.proposer):
            raise TypeError("RDKitCandidateSource proposer must be callable")

    def propose(self, request: CandidateRequest) -> tuple[CandidateProposal, ...]:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        try:
            proposals = tuple(self.proposer(request))
        except CandidateSourceError:
            raise
        except Exception as error:
            raise CandidateSourceError(
                code=CandidateRejectCode.SOURCE_FAILED,
                source=self.source_type,
                detail=f"RDKit proposer raised {type(error).__name__}",
            ) from error
        if any(type(item) is not CandidateProposal for item in proposals):
            raise CandidateSourceError(
                code=CandidateRejectCode.INVALID_PROPOSAL,
                source=self.source_type,
                detail="RDKit proposer returned a non-CandidateProposal value",
            )
        return proposals


__all__ = ["RDKitCandidateSource"]
