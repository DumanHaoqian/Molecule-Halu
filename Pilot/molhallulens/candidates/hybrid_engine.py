"""Hybrid deterministic/LLM candidate pooling for T037.

The engine gathers a complete pool and writes an audit ledger.  It never
selects a first response item and never lets the proposal model decide local
validity, acceptance, labels, policy, or split membership.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from molhallulens.chemistry import MoleculeParseError
from molhallulens.domain import (
    CandidatePool,
    CandidateSourceType,
    ValueProvenance,
)
from molhallulens.providers.poe.schemas import ProposalRequest

from . import core as candidate_core
from .core import (
    CandidateBuildResult,
    CandidateProposal,
    CandidateRejectCode,
    CandidateRejection,
    CandidateRequest,
    CandidateSource,
    CandidateSourceError,
    RankedCandidate,
    rank_candidates,
)
from .donors import DonorEntry, SplitBoundDonorQuery
from .llm_agent_source import (
    MAX_LLM_CANDIDATE_ATTEMPTS,
    LLMAgentCandidateSource,
    LLMProposalRound,
    LLMProposalSourceError,
    LLMRetryFeedback,
)

HYBRID_ENGINE_VERSION = "molhallulens.hybrid_candidate_engine.v1"


@runtime_checkable
class SplitLocalDonorPool(Protocol):
    manifest_sha256: str
    split: str

    def query(self, query: SplitBoundDonorQuery) -> Sequence[DonorEntry]: ...


@dataclass(frozen=True, slots=True)
class HybridAuditRejection:
    code: str
    proposal_id: str
    attempt_index: int | None
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code:
            raise ValueError("audit rejection code must be non-empty text")
        if type(self.proposal_id) is not str or not self.proposal_id:
            raise ValueError("audit proposal_id must be non-empty text")
        if self.attempt_index is not None and (
            type(self.attempt_index) is not int
            or not 1 <= self.attempt_index <= MAX_LLM_CANDIDATE_ATTEMPTS
        ):
            raise ValueError("attempt_index must be in [1, 3] or None")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class LLMAttemptLedger:
    attempt_index: int
    feedback_reject_codes: tuple[str, ...]
    response_candidate_ids: tuple[str, ...]
    validated_candidate_ids: tuple[str, ...]
    rejection_codes: tuple[str, ...]
    client_error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or not 1 <= self.attempt_index <= 3:
            raise ValueError("attempt_index must be in [1, 3]")
        for values, name in (
            (self.feedback_reject_codes, "feedback_reject_codes"),
            (self.response_candidate_ids, "response_candidate_ids"),
            (self.validated_candidate_ids, "validated_candidate_ids"),
            (self.rejection_codes, "rejection_codes"),
        ):
            normalized = tuple(values)
            if any(type(value) is not str or not value for value in normalized):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, normalized)
        if self.client_error_code is not None and (
            type(self.client_error_code) is not str or not self.client_error_code
        ):
            raise ValueError("client_error_code must be non-empty text or None")


@dataclass(frozen=True, slots=True)
class DeterministicFallbackLedger:
    triggered: bool
    operator_id: str
    policy: str
    target_root: str
    source: CandidateSourceType | None
    validated_candidate_ids: tuple[str, ...]
    rejection_codes: tuple[str, ...]
    scheduler_backfill_required: bool

    def __post_init__(self) -> None:
        if (
            type(self.triggered) is not bool
            or type(self.scheduler_backfill_required) is not bool
        ):
            raise TypeError("fallback booleans must be exact bool values")
        for value, name in (
            (self.operator_id, "operator_id"),
            (self.policy, "policy"),
            (self.target_root, "target_root"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.source is not None and self.source not in {
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
        }:
            raise ValueError("fallback source must be RULE, RDKIT, or None")
        for values, name in (
            (self.validated_candidate_ids, "validated_candidate_ids"),
            (self.rejection_codes, "rejection_codes"),
        ):
            normalized = tuple(values)
            if any(type(value) is not str or not value for value in normalized):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, normalized)
        if not self.triggered and (
            self.source is not None
            or self.validated_candidate_ids
            or self.rejection_codes
            or self.scheduler_backfill_required
        ):
            raise ValueError("an untriggered fallback cannot carry fallback outcomes")


@dataclass(frozen=True, slots=True)
class HybridCandidateBuildResult:
    """A complete validated pool and audit trail, intentionally without selection."""

    build_result: CandidateBuildResult
    llm_attempts: tuple[LLMAttemptLedger, ...]
    fallback: DeterministicFallbackLedger
    audit_rejections: tuple[HybridAuditRejection, ...]
    source_counts: Mapping[str, int]
    engine_version: str = HYBRID_ENGINE_VERSION

    def __post_init__(self) -> None:
        if type(self.build_result) is not CandidateBuildResult:
            raise TypeError("build_result must be CandidateBuildResult")
        attempts = tuple(self.llm_attempts)
        audit = tuple(self.audit_rejections)
        if any(type(item) is not LLMAttemptLedger for item in attempts):
            raise TypeError("llm_attempts must contain LLMAttemptLedger values")
        if tuple(item.attempt_index for item in attempts) != tuple(
            range(1, len(attempts) + 1)
        ):
            raise ValueError("LLM attempts must be contiguous and one-indexed")
        if type(self.fallback) is not DeterministicFallbackLedger:
            raise TypeError("fallback must be DeterministicFallbackLedger")
        if any(type(item) is not HybridAuditRejection for item in audit):
            raise TypeError("audit_rejections must contain HybridAuditRejection values")
        if not isinstance(self.source_counts, Mapping):
            raise TypeError("source_counts must be a mapping")
        counts = dict(self.source_counts)
        if any(
            type(key) is not str or not key or type(value) is not int or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("source_counts must map text to non-negative integers")
        if sum(counts.values()) != len(self.build_result.pool.candidates):
            raise ValueError("source_counts must exactly cover the complete pool")
        if self.engine_version != HYBRID_ENGINE_VERSION:
            raise ValueError("unsupported hybrid engine version")
        object.__setattr__(self, "llm_attempts", attempts)
        object.__setattr__(self, "audit_rejections", audit)
        object.__setattr__(self, "source_counts", MappingProxyType(counts))

    @property
    def pool(self) -> CandidatePool:
        return self.build_result.pool

    @property
    def llm_materially_participated(self) -> bool:
        return self.source_counts.get(CandidateSourceType.LLM.value, 0) > 0


def _candidate_rejection(
    request: CandidateRequest,
    proposal_id: str,
    code: CandidateRejectCode,
    **evidence: object,
) -> CandidateRejection:
    return CandidateRejection(
        code=code,
        proposal_id=proposal_id,
        operator_id=request.operator_id,
        evidence=evidence or {"phase": "t037_candidate_gate"},
    )


@dataclass(frozen=True, slots=True)
class T018CandidateGate:
    """Apply the same chemistry/replay/signature boundary to any proposal source."""

    def validate(
        self,
        request: CandidateRequest,
        proposals: Sequence[CandidateProposal],
        *,
        allowed_sources: frozenset[CandidateSourceType],
    ) -> CandidateBuildResult:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        values = tuple(proposals)
        if any(type(item) is not CandidateProposal for item in values):
            raise TypeError("proposals must contain CandidateProposal values")
        if not allowed_sources or any(
            source
            not in {
                CandidateSourceType.RULE,
                CandidateSourceType.RDKIT,
                CandidateSourceType.LLM,
            }
            for source in allowed_sources
        ):
            raise ValueError("allowed_sources contains an unsupported proposal source")

        rejections: list[CandidateRejection] = []
        validated: list[CandidateProposal] = []
        proposal_ids: set[str] = set()
        for proposal in sorted(values, key=lambda item: item.proposal_id):
            if proposal.proposal_id in proposal_ids:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.DUPLICATE,
                        duplicate_scope="proposal_id",
                    )
                )
                continue
            proposal_ids.add(proposal.proposal_id)
            patch = proposal.patch
            expected_provenance = {
                CandidateSourceType.RULE: ValueProvenance.RULE,
                CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
                CandidateSourceType.LLM: ValueProvenance.LLM,
            }.get(patch.source)
            if patch.source not in allowed_sources or expected_provenance is None:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.SOURCE_MISMATCH,
                        actual_source=patch.source.value,
                    )
                )
                continue
            if patch.new_value.provenance is not expected_provenance:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.SOURCE_MISMATCH,
                        actual_provenance=patch.new_value.provenance.value,
                    )
                )
                continue
            if patch.root_node_id != request.resolution.target_node_id:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.ROOT_MISMATCH,
                    )
                )
                continue
            reference = request.context.reference_graph.value_for(patch.root_node_id)
            if patch.old_value != reference:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.REFERENCE_VALUE_MISMATCH,
                    )
                )
                continue
            try:
                candidate_core._validate_attachment_semantics(request, proposal)
                if candidate_core._reference_equivalent(request, proposal):
                    rejections.append(
                        _candidate_rejection(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.REFERENCE_EQUIVALENT,
                        )
                    )
                    continue
                canonical = candidate_core._canonicalize_proposal(proposal)
                candidate_core._validate_action_product(request, canonical)
                if candidate_core._symmetry_equivalent(request, canonical):
                    rejections.append(
                        _candidate_rejection(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.SYMMETRY_EQUIVALENT,
                        )
                    )
                    continue
                validated.append(canonical)
            except MoleculeParseError as error:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.SMILES_INVALID,
                        molecule_error_code=error.code.value,
                        input_length=error.input_length,
                    )
                )
            except ValueError as error:
                raw_code = error.args[0] if error.args else None
                code = (
                    CandidateRejectCode(raw_code)
                    if raw_code in CandidateRejectCode._value2member_map_
                    else CandidateRejectCode.INVALID_PROPOSAL
                )
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        code,
                        phase="t018_chemistry_gate",
                        exception_type=type(error).__name__,
                    )
                )
            except (KeyError, RuntimeError, TypeError) as error:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.INVALID_PROPOSAL,
                        phase="t018_chemistry_gate",
                        exception_type=type(error).__name__,
                    )
                )

        ranked: list[RankedCandidate] = []
        for proposal in validated:
            try:
                ranked.extend(rank_candidates(request, (proposal,)))
            except (MoleculeParseError, RuntimeError, TypeError, ValueError) as error:
                rejections.append(
                    _candidate_rejection(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.INVALID_PROPOSAL,
                        phase="difficulty_ranking",
                        exception_type=type(error).__name__,
                    )
                )

        accepted: list[RankedCandidate] = []
        keys: dict[str, str] = {}
        candidate_ids: dict[str, str] = {}
        for item in sorted(ranked, key=lambda value: value.rank_key):
            duplicate_of = keys.get(item.canonical_key) or candidate_ids.get(
                item.proposal.patch.candidate_id
            )
            if duplicate_of is not None:
                rejections.append(
                    _candidate_rejection(
                        request,
                        item.proposal.proposal_id,
                        CandidateRejectCode.DUPLICATE,
                        duplicate_of=duplicate_of,
                        duplicate_scope="semantic_or_candidate_id",
                    )
                )
                continue
            keys[item.canonical_key] = item.proposal.proposal_id
            candidate_ids[item.proposal.patch.candidate_id] = item.proposal.proposal_id
            accepted.append(item)
        ordered_rejections = tuple(
            sorted(rejections, key=lambda item: (item.proposal_id, item.code.value))
        )
        rejection_codes = tuple(
            sorted({item.code.value for item in ordered_rejections})
        )
        if not accepted and not rejection_codes:
            rejection_codes = (CandidateRejectCode.INVALID_PROPOSAL.value,)
        return CandidateBuildResult(
            pool=CandidatePool(
                request_id=request.request_id,
                candidates=tuple(item.proposal.patch for item in accepted),
                rejection_codes=rejection_codes,
            ),
            rejections=ordered_rejections,
            ranked_candidates=tuple(accepted),
        )


def _request_binding_is_exact(
    candidate_request: CandidateRequest,
    proposal_request: ProposalRequest,
) -> bool:
    recipe = candidate_request.context.recipe
    return (
        proposal_request.request_id == candidate_request.request_id
        and proposal_request.origin_id == recipe.origin_id
        and proposal_request.operator_id == candidate_request.operator_id
        and proposal_request.propagation == recipe.policy.dataset_name
        and proposal_request.candidate_source_mode == recipe.candidate_source_mode.value
        and proposal_request.target_root == candidate_request.resolution.target_node_id
        and proposal_request.dataset_version
        == proposal_request.manifest_identity.dataset_version
        and proposal_request.variant_index == recipe.variant_index
        and proposal_request.derived_seed == recipe.derived_seed
    )


def _query_current_split_donors(
    proposal_request: ProposalRequest,
    donor_pool: SplitLocalDonorPool,
    donor_query: SplitBoundDonorQuery,
) -> tuple[DonorEntry, ...]:
    if not isinstance(donor_pool, SplitLocalDonorPool):
        raise TypeError("donor_pool must implement the split-local donor protocol")
    if type(donor_query) is not SplitBoundDonorQuery:
        raise TypeError("donor_query must be SplitBoundDonorQuery")
    if not (
        donor_query.manifest_sha256
        == proposal_request.manifest_identity.manifest_sha256
        == donor_pool.manifest_sha256
        and donor_query.split == proposal_request.split == donor_pool.split
        and donor_query.recipient_origin_id == proposal_request.origin_id
    ):
        raise ValueError(
            "donor query is not bound to the current manifest split/origin"
        )
    donors = tuple(donor_pool.query(donor_query))
    if any(
        type(donor) is not DonorEntry or donor.split != proposal_request.split
        for donor in donors
    ):
        raise ValueError("donor pool returned a cross-split or malformed donor")
    return donors


def _constraint_filter(
    request: CandidateRequest,
    proposal_request: ProposalRequest,
    result: CandidateBuildResult,
) -> CandidateBuildResult:
    constraints = proposal_request.constraints
    retained: list[RankedCandidate] = []
    rejections = list(result.rejections)
    for item in result.ranked_candidates:
        features = item.difficulty_features
        mismatches = []
        if constraints.same_attachment_element is True and (
            features.anchor_element_match is not True
        ):
            mismatches.append("same_attachment_element")
        if constraints.match_heavy_count is True and features.heavy_atom_delta != 0:
            mismatches.append("match_heavy_count")
        if constraints.match_ring_count is True and features.ring_delta != 0:
            mismatches.append("match_ring_count")
        if mismatches:
            rejections.append(
                _candidate_rejection(
                    request,
                    item.proposal.proposal_id,
                    CandidateRejectCode.INVALID_PROPOSAL,
                    phase="proposal_constraints",
                    constraints=tuple(mismatches),
                )
            )
        else:
            retained.append(item)
    codes = tuple(sorted({item.code.value for item in rejections}))
    if not retained and not codes:
        codes = (CandidateRejectCode.INVALID_PROPOSAL.value,)
    return CandidateBuildResult(
        pool=CandidatePool(
            request_id=request.request_id,
            candidates=tuple(item.proposal.patch for item in retained),
            rejection_codes=codes,
        ),
        rejections=tuple(
            sorted(rejections, key=lambda item: (item.proposal_id, item.code.value))
        ),
        ranked_candidates=tuple(retained),
    )


def _merge_validated(
    request: CandidateRequest,
    results: Sequence[CandidateBuildResult],
) -> CandidateBuildResult:
    proposals = tuple(
        item.proposal for result in results for item in result.ranked_candidates
    )
    rejections = [item for result in results for item in result.rejections]
    ranked = rank_candidates(request, proposals) if proposals else ()
    retained: list[RankedCandidate] = []
    semantic: dict[str, str] = {}
    candidate_ids: dict[str, str] = {}
    for item in ranked:
        duplicate_of = semantic.get(item.canonical_key) or candidate_ids.get(
            item.proposal.patch.candidate_id
        )
        if duplicate_of is not None:
            rejections.append(
                _candidate_rejection(
                    request,
                    item.proposal.proposal_id,
                    CandidateRejectCode.DUPLICATE,
                    duplicate_of=duplicate_of,
                    duplicate_scope="complete_pool",
                )
            )
            continue
        semantic[item.canonical_key] = item.proposal.proposal_id
        candidate_ids[item.proposal.patch.candidate_id] = item.proposal.proposal_id
        retained.append(item)
    codes = tuple(sorted({item.code.value for item in rejections}))
    if not retained and not codes:
        codes = (CandidateRejectCode.INSUFFICIENT_CANDIDATES.value,)
    return CandidateBuildResult(
        pool=CandidatePool(
            request_id=request.request_id,
            candidates=tuple(item.proposal.patch for item in retained),
            rejection_codes=codes,
        ),
        rejections=tuple(
            sorted(rejections, key=lambda item: (item.proposal_id, item.code.value))
        ),
        ranked_candidates=tuple(retained),
    )


@dataclass(frozen=True, slots=True)
class HybridCandidateEngine:
    """Build a full verified pool, retry LLM proposals, then fail over exactly."""

    llm_source: LLMAgentCandidateSource
    deterministic_sources: tuple[CandidateSource, ...] = ()
    fallback_by_operator: Mapping[str, CandidateSource] = field(default_factory=dict)
    minimum_valid_llm_candidates: int = 1
    max_llm_attempts: int = MAX_LLM_CANDIDATE_ATTEMPTS
    gate: T018CandidateGate = T018CandidateGate()

    def __post_init__(self) -> None:
        if type(self.llm_source) is not LLMAgentCandidateSource:
            raise TypeError("llm_source must be LLMAgentCandidateSource")
        deterministic = tuple(self.deterministic_sources)
        if any(not isinstance(source, CandidateSource) for source in deterministic):
            raise TypeError("deterministic_sources must implement CandidateSource")
        if any(
            source.source_type
            not in {CandidateSourceType.RULE, CandidateSourceType.RDKIT}
            for source in deterministic
        ):
            raise ValueError("deterministic_sources must be RULE or RDKIT")
        if len({source.source_type for source in deterministic}) != len(deterministic):
            raise ValueError("deterministic source types must be unique")
        if not isinstance(self.fallback_by_operator, Mapping):
            raise TypeError("fallback_by_operator must be a mapping")
        fallback = dict(self.fallback_by_operator)
        if any(type(key) is not str or not key for key in fallback):
            raise TypeError("fallback operator IDs must be non-empty strings")
        if any(
            not isinstance(source, CandidateSource)
            or source.source_type
            not in {CandidateSourceType.RULE, CandidateSourceType.RDKIT}
            for source in fallback.values()
        ):
            raise ValueError("predeclared fallbacks must be deterministic sources")
        if type(self.minimum_valid_llm_candidates) is not int or not (
            1 <= self.minimum_valid_llm_candidates <= 5
        ):
            raise ValueError("minimum_valid_llm_candidates must be in [1, 5]")
        if type(self.max_llm_attempts) is not int or not (
            1 <= self.max_llm_attempts <= MAX_LLM_CANDIDATE_ATTEMPTS
        ):
            raise ValueError("max_llm_attempts must be in [1, 3]")
        if type(self.gate) is not T018CandidateGate:
            raise TypeError("gate must be T018CandidateGate")
        object.__setattr__(self, "deterministic_sources", deterministic)
        object.__setattr__(self, "fallback_by_operator", MappingProxyType(fallback))

    def _validate_source(
        self,
        request: CandidateRequest,
        source: CandidateSource,
    ) -> CandidateBuildResult:
        try:
            proposals = tuple(source.propose(request))
        except (CandidateSourceError, RuntimeError, TypeError, ValueError):
            rejection = _candidate_rejection(
                request,
                f"source:{source.source_type.value}",
                CandidateRejectCode.SOURCE_FAILED,
                source=source.source_type.value,
            )
            return CandidateBuildResult(
                pool=CandidatePool(
                    request_id=request.request_id,
                    candidates=(),
                    rejection_codes=(rejection.code.value,),
                ),
                rejections=(rejection,),
            )
        return self.gate.validate(
            request,
            proposals,
            allowed_sources=frozenset({source.source_type}),
        )

    def build_pool(
        self,
        request: CandidateRequest,
        *,
        proposal_request: ProposalRequest,
        donor_pool: SplitLocalDonorPool,
        donor_query: SplitBoundDonorQuery,
    ) -> HybridCandidateBuildResult:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        if type(proposal_request) is not ProposalRequest:
            raise TypeError("proposal_request must be ProposalRequest")
        if request.context.recipe.candidate_source_mode not in {
            CandidateSourceType.LLM,
            CandidateSourceType.HYBRID,
        }:
            raise ValueError("T037 engine requires an LLM or HYBRID recipe")
        if not _request_binding_is_exact(request, proposal_request):
            raise ValueError("proposal request changed operator, policy, root, or seed")
        donors = _query_current_split_donors(
            proposal_request,
            donor_pool,
            donor_query,
        )

        validated_results: list[CandidateBuildResult] = []
        audit: list[HybridAuditRejection] = []
        if request.context.recipe.candidate_source_mode is CandidateSourceType.HYBRID:
            for source in self.deterministic_sources:
                validated_results.append(self._validate_source(request, source))

        feedback = LLMRetryFeedback()
        llm_attempts: list[LLMAttemptLedger] = []
        validated_llm: list[CandidateBuildResult] = []
        llm_candidate_ids: set[str] = set()
        llm_semantic_keys: set[str] = set()
        for attempt_index in range(1, self.max_llm_attempts + 1):
            round_request = LLMProposalRound(
                request=proposal_request,
                attempt_index=attempt_index,
                feedback=feedback,
                donors=donors,
            )
            try:
                batch = self.llm_source.propose_round(request, round_request)
            except LLMProposalSourceError as error:
                audit.append(
                    HybridAuditRejection(
                        code=error.code.value,
                        proposal_id=f"source:{request.request_id}",
                        attempt_index=attempt_index,
                        evidence={"phase": "llm_client"},
                    )
                )
                llm_attempts.append(
                    LLMAttemptLedger(
                        attempt_index=attempt_index,
                        feedback_reject_codes=feedback.reject_codes,
                        response_candidate_ids=(),
                        validated_candidate_ids=(),
                        rejection_codes=(error.code.value,),
                        client_error_code=error.code.value,
                    )
                )
                feedback = LLMRetryFeedback((error.code.value,))
                if error.code.value == "LLM_RETRY_UNSUPPORTED":
                    break
                continue

            for rejection in batch.rejections:
                audit.append(
                    HybridAuditRejection(
                        code=rejection.code,
                        proposal_id=rejection.proposal_id,
                        attempt_index=attempt_index,
                        evidence=rejection.evidence,
                    )
                )
            gate_result = self.gate.validate(
                request,
                batch.proposals,
                allowed_sources=frozenset({CandidateSourceType.LLM}),
            )
            gate_result = _constraint_filter(request, proposal_request, gate_result)
            retained: list[RankedCandidate] = []
            cross_attempt_rejections = list(gate_result.rejections)
            for item in gate_result.ranked_candidates:
                candidate_id = item.proposal.patch.candidate_id
                if (
                    candidate_id in llm_candidate_ids
                    or item.canonical_key in llm_semantic_keys
                ):
                    rejection = _candidate_rejection(
                        request,
                        item.proposal.proposal_id,
                        CandidateRejectCode.DUPLICATE,
                        duplicate_scope="llm_retry_history",
                    )
                    cross_attempt_rejections.append(rejection)
                    audit.append(
                        HybridAuditRejection(
                            code=rejection.code.value,
                            proposal_id=rejection.proposal_id,
                            attempt_index=attempt_index,
                            evidence=rejection.evidence,
                        )
                    )
                    continue
                llm_candidate_ids.add(candidate_id)
                llm_semantic_keys.add(item.canonical_key)
                retained.append(item)
            attempt_codes = tuple(
                sorted(
                    {
                        *(item.code for item in batch.rejections),
                        *(item.code.value for item in cross_attempt_rejections),
                    }
                )
            )
            attempt_result = CandidateBuildResult(
                pool=CandidatePool(
                    request_id=request.request_id,
                    candidates=tuple(item.proposal.patch for item in retained),
                    rejection_codes=(
                        attempt_codes
                        if attempt_codes
                        else (
                            ()
                            if retained
                            else (CandidateRejectCode.INVALID_PROPOSAL.value,)
                        )
                    ),
                ),
                rejections=tuple(
                    sorted(
                        cross_attempt_rejections,
                        key=lambda item: (item.proposal_id, item.code.value),
                    )
                ),
                ranked_candidates=tuple(retained),
            )
            validated_llm.append(attempt_result)
            llm_attempts.append(
                LLMAttemptLedger(
                    attempt_index=attempt_index,
                    feedback_reject_codes=feedback.reject_codes,
                    response_candidate_ids=batch.response_candidate_ids,
                    validated_candidate_ids=tuple(
                        item.proposal.patch.candidate_id for item in retained
                    ),
                    rejection_codes=attempt_codes,
                )
            )
            valid_count = sum(len(item.ranked_candidates) for item in validated_llm)
            if valid_count >= self.minimum_valid_llm_candidates:
                break
            feedback = LLMRetryFeedback(
                attempt_codes or (CandidateRejectCode.INSUFFICIENT_CANDIDATES.value,)
            )

        validated_results.extend(validated_llm)
        llm_valid_count = sum(len(item.ranked_candidates) for item in validated_llm)
        fallback_source: CandidateSource | None = None
        fallback_result: CandidateBuildResult | None = None
        if llm_valid_count == 0:
            fallback_source = self.fallback_by_operator.get(request.operator_id)
            if fallback_source is not None:
                fallback_result = self._validate_source(request, fallback_source)
                fallback_result = _constraint_filter(
                    request,
                    proposal_request,
                    fallback_result,
                )
                validated_results.append(fallback_result)

        merged = _merge_validated(request, validated_results)
        fallback_triggered = llm_valid_count == 0
        fallback_ids = (
            ()
            if fallback_result is None
            else tuple(
                item.proposal.patch.candidate_id
                for item in fallback_result.ranked_candidates
            )
        )
        fallback_codes = (
            () if fallback_result is None else fallback_result.pool.rejection_codes
        )
        fallback = DeterministicFallbackLedger(
            triggered=fallback_triggered,
            operator_id=request.operator_id,
            policy=request.context.recipe.policy.dataset_name,
            target_root=request.context.recipe.target_node_id,
            source=(None if fallback_source is None else fallback_source.source_type),
            validated_candidate_ids=fallback_ids,
            rejection_codes=(
                fallback_codes
                if fallback_source is not None
                else (
                    ("DETERMINISTIC_FALLBACK_UNAVAILABLE",)
                    if fallback_triggered
                    else ()
                )
            ),
            scheduler_backfill_required=(
                fallback_triggered and not fallback_ids and not merged.pool.candidates
            ),
        )
        if fallback_triggered and fallback_source is None:
            audit.append(
                HybridAuditRejection(
                    code="DETERMINISTIC_FALLBACK_UNAVAILABLE",
                    proposal_id=f"fallback:{request.operator_id}",
                    attempt_index=None,
                    evidence={
                        "policy": request.context.recipe.policy.dataset_name,
                        "target_root": request.context.recipe.target_node_id,
                    },
                )
            )
        source_counts = Counter(patch.source.value for patch in merged.pool.candidates)
        return HybridCandidateBuildResult(
            build_result=merged,
            llm_attempts=tuple(llm_attempts),
            fallback=fallback,
            audit_rejections=tuple(
                sorted(
                    audit,
                    key=lambda item: (
                        item.attempt_index is None,
                        item.attempt_index or 0,
                        item.proposal_id,
                        item.code,
                    ),
                )
            ),
            source_counts=dict(source_counts),
        )


__all__ = [
    "HYBRID_ENGINE_VERSION",
    "DeterministicFallbackLedger",
    "HybridAuditRejection",
    "HybridCandidateBuildResult",
    "HybridCandidateEngine",
    "LLMAttemptLedger",
    "SplitLocalDonorPool",
    "T018CandidateGate",
]
