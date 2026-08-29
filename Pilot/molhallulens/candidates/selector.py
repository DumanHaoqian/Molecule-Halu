"""Source-neutral deterministic selection from a complete T037 candidate pool."""

from __future__ import annotations

from dataclasses import dataclass

from molhallulens.domain import CandidatePatch, CandidateSourceType

from .core import CandidateBuildResult, RankedCandidate
from .hybrid_engine import HybridCandidateBuildResult

SELECTION_RULE_VERSION = "molhallulens.candidate_selector.v1"


class CandidateSelectionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("detail must be non-empty text")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CandidateRankAudit:
    rank: int
    candidate_id: str
    proposal_id: str
    source: CandidateSourceType
    structural_similarity: float | None
    source_score: float

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.proposal_id, "proposal_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.source) is not CandidateSourceType:
            raise TypeError("source must be CandidateSourceType")


@dataclass(frozen=True, slots=True)
class CandidateSelectionDecision:
    selected: CandidatePatch
    selected_proposal_id: str
    ranking: tuple[CandidateRankAudit, ...]
    rule_version: str = SELECTION_RULE_VERSION

    def __post_init__(self) -> None:
        if type(self.selected) is not CandidatePatch:
            raise TypeError("selected must be CandidatePatch")
        if type(self.selected_proposal_id) is not str or not self.selected_proposal_id:
            raise ValueError("selected_proposal_id must be non-empty text")
        ranking = tuple(self.ranking)
        if not ranking:
            raise ValueError("selection ranking cannot be empty")
        if any(type(item) is not CandidateRankAudit for item in ranking):
            raise TypeError("ranking must contain CandidateRankAudit values")
        if tuple(item.rank for item in ranking) != tuple(range(1, len(ranking) + 1)):
            raise ValueError("ranking must be contiguous and one-indexed")
        if (
            ranking[0].candidate_id != self.selected.candidate_id
            or ranking[0].proposal_id != self.selected_proposal_id
        ):
            raise ValueError("selection must be rank one from the complete pool")
        if self.rule_version != SELECTION_RULE_VERSION:
            raise ValueError("unsupported selection rule version")
        object.__setattr__(self, "ranking", ranking)

    @property
    def considered_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.ranking)


def _build_result(
    result: HybridCandidateBuildResult | CandidateBuildResult,
) -> CandidateBuildResult:
    if type(result) is HybridCandidateBuildResult:
        return result.build_result
    if type(result) is CandidateBuildResult:
        return result
    raise TypeError("result must be a complete candidate build result")


def _rank_audit(rank: int, item: RankedCandidate) -> CandidateRankAudit:
    features = item.difficulty_features
    return CandidateRankAudit(
        rank=rank,
        candidate_id=item.proposal.patch.candidate_id,
        proposal_id=item.proposal.proposal_id,
        source=item.proposal.patch.source,
        structural_similarity=features.structural_similarity,
        source_score=float(features.source_score),
    )


@dataclass(frozen=True, slots=True)
class FrozenCandidateSelector:
    """Choose rank one only after every validated source has been merged."""

    rule_version: str = SELECTION_RULE_VERSION

    def __post_init__(self) -> None:
        if self.rule_version != SELECTION_RULE_VERSION:
            raise ValueError("selection rule is frozen for this dataset version")

    def select(
        self,
        result: HybridCandidateBuildResult | CandidateBuildResult,
    ) -> CandidateSelectionDecision:
        build = _build_result(result)
        ranked = tuple(build.ranked_candidates)
        if not ranked:
            raise CandidateSelectionError(
                "EMPTY_VALIDATED_POOL",
                "selection requires a non-empty fully validated candidate pool",
            )
        if tuple(item.proposal.patch for item in ranked) != build.pool.candidates:
            raise CandidateSelectionError(
                "INCOMPLETE_POOL_RANKING",
                "ranked candidates do not exactly cover the complete pool",
            )
        ordered = tuple(sorted(ranked, key=lambda item: item.rank_key))
        if ordered != ranked:
            raise CandidateSelectionError(
                "UNFROZEN_POOL_ORDER",
                "candidate pool is not ordered by the frozen T018 rank key",
            )
        ranking = tuple(
            _rank_audit(index, item) for index, item in enumerate(ordered, start=1)
        )
        selected = ordered[0]
        return CandidateSelectionDecision(
            selected=selected.proposal.patch,
            selected_proposal_id=selected.proposal.proposal_id,
            ranking=ranking,
            rule_version=self.rule_version,
        )


__all__ = [
    "SELECTION_RULE_VERSION",
    "CandidateRankAudit",
    "CandidateSelectionDecision",
    "CandidateSelectionError",
    "FrozenCandidateSelector",
]
