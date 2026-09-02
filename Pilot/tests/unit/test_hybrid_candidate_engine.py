"""T037 proposal-only retry, deterministic gate, fallback, and selection tests."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from molhallulens.modules.error_planning.core import (
    CandidateDifficultyFeatures,
    CandidateProposal,
    CandidateRequest,
)
from molhallulens.modules.error_planning.donors import (
    DonorEntry,
    DonorKind,
    SplitBoundDonorQuery,
)
from molhallulens.modules.error_planning.hybrid import (
    HybridCandidateEngine,
    T018CandidateGate,
)
from molhallulens.modules.error_planning.llm import (
    LLMAgentCandidateSource,
    LLMProposalRound,
    verified_candidate_adapter,
)
from molhallulens.modules.error_planning.rules import RuleCandidateSource
from molhallulens.modules.error_planning.selector import (
    CandidateSelectionError,
    FrozenCandidateSelector,
)
from molhallulens.core import (
    CandidatePatch,
    CandidateSourceType,
    ClaimValue,
    ComparatorKind,
    EditingSubtask,
    PropagationPolicy,
    ValueProvenance,
    ValueType,
)
from molhallulens.infrastructure.providers.poe.schemas import (
    FROZEN_GLOBAL_SEED,
    ProposalCandidatePatch,
    ProposalConstraints,
    ProposalManifestIdentity,
    ProposalReplacement,
    ProposalRequest,
    ProposalResponse,
    derive_proposal_seed,
)

MANIFEST_ID = "a" * 64
ORIGIN_ID = "mol_edit.add_v2.0022"
OPERATOR_ID = "mol_edit.add.heavy_count_claim"
ROOT = "heavy_delta"


class _ReferenceGraph:
    def __init__(self, value: ClaimValue) -> None:
        self.value = value

    def value_for(self, node_id: str) -> ClaimValue:
        assert node_id == ROOT
        return self.value


def _candidate_request() -> CandidateRequest:
    """Exact CandidateRequest instance with the scalar fields exercised here."""

    derived_seed = derive_proposal_seed(
        global_seed=FROZEN_GLOBAL_SEED,
        dataset_version="pilot_v1",
        origin_id=ORIGIN_ID,
        operator_id=OPERATOR_ID,
        policy="LOCAL",
        variant_index=0,
    )
    old = ClaimValue(
        raw_value=5,
        normalized_value=5,
        value_type=ValueType.INTEGER,
        provenance=ValueProvenance.REFERENCE,
    )
    recipe = SimpleNamespace(
        recipe_id="recipe:t037:scalar",
        origin_id=ORIGIN_ID,
        operator_id=OPERATOR_ID,
        policy=PropagationPolicy.STOP,
        candidate_source_mode=CandidateSourceType.HYBRID,
        target_node_id=ROOT,
        variant_index=0,
        derived_seed=derived_seed,
    )
    context = SimpleNamespace(
        recipe=recipe,
        reference_graph=_ReferenceGraph(old),
        state_schema=SimpleNamespace(
            nodes_by_id={ROOT: SimpleNamespace(comparator=ComparatorKind.EXACT)}
        ),
    )
    resolution = SimpleNamespace(
        target_node_id=ROOT,
        registration=SimpleNamespace(
            operator_id=OPERATOR_ID,
            operator_family="numeric_count_claim",
        ),
    )
    request = object.__new__(CandidateRequest)
    object.__setattr__(request, "context", context)
    object.__setattr__(request, "resolution", resolution)
    return request


def _proposal_request(request: CandidateRequest) -> ProposalRequest:
    return ProposalRequest(
        request_id=request.request_id,
        origin_id=ORIGIN_ID,
        operator_id=OPERATOR_ID,
        propagation="LOCAL",
        candidate_source_mode="HYBRID",
        target_root=ROOT,
        constraints=ProposalConstraints(),
        global_seed=FROZEN_GLOBAL_SEED,
        dataset_version="pilot_v1",
        variant_index=0,
        derived_seed=request.derived_seed,
        split="train",
        manifest_identity=ProposalManifestIdentity(
            dataset_version="pilot_v1",
            split_seed=8347206628578381721,
            manifest_sha256=MANIFEST_ID,
            source_origin_audit_sha256="b" * 64,
            source_split_report_sha256="c" * 64,
        ),
    )


def _response(
    request: ProposalRequest,
    attempt: int,
    replacements: tuple[ProposalReplacement, ProposalReplacement, ProposalReplacement],
) -> ProposalResponse:
    return ProposalResponse(
        request_id=request.request_id,
        candidates=tuple(
            ProposalCandidatePatch(
                candidate_id=f"attempt{attempt}.candidate{index}",
                root_field=ROOT,
                replacement=replacement,
                minimal_surface_realization=f"alternate scalar {index}",
                plausibility_reason="Near-reference deterministic test proposal.",
            )
            for index, replacement in enumerate(replacements, start=1)
        ),
    )


class _RoundClient:
    def __init__(
        self,
        request: ProposalRequest,
        responses: tuple[ProposalResponse, ...],
    ) -> None:
        self.request = request
        self.responses = responses
        self.rounds: list[LLMProposalRound] = []

    def propose_round(self, round_request: LLMProposalRound) -> object:
        self.rounds.append(round_request)
        return SimpleNamespace(
            request=self.request,
            response=self.responses[round_request.attempt_index - 1],
        )


class _DonorPool:
    manifest_sha256 = MANIFEST_ID
    split = "train"

    def __init__(self) -> None:
        self.queries: list[SplitBoundDonorQuery] = []
        self.donor = DonorEntry(
            donor_id="product:donor.origin",
            donor_origin_id="donor.origin",
            kind=DonorKind.PRODUCT,
            split="train",
            canonical_smiles="CC",
            heavy_atom_count=2,
            ring_count=0,
            formal_charge=0,
            heteroatom_counts=(),
            attachment_atomic_numbers=(),
            boundary_bond_types=(),
            difficulty_bucket="mid",
        )

    def query(self, query: SplitBoundDonorQuery) -> tuple[DonorEntry, ...]:
        self.queries.append(query)
        return (self.donor,)


def _donor_query(*, split: str = "train") -> SplitBoundDonorQuery:
    return SplitBoundDonorQuery(
        manifest_sha256=MANIFEST_ID,
        split=split,
        recipient_origin_id=ORIGIN_ID,
        kind=DonorKind.PRODUCT,
    )


def _deterministic_proposal(
    request: CandidateRequest,
    value: int,
) -> CandidateProposal:
    old = request.context.reference_graph.value_for(ROOT)
    new = replace(
        old,
        raw_value=value,
        normalized_value=value,
        provenance=ValueProvenance.RULE,
        locally_valid=True,
        oracle_match=False,
        confidence=1.0,
    )
    return CandidateProposal(
        proposal_id=f"rule:value:{value}",
        patch=CandidatePatch(
            candidate_id=f"rule.value.{value}",
            root_node_id=ROOT,
            old_value=old,
            new_value=new,
            edit_action=None,
            source=CandidateSourceType.RULE,
        ),
        difficulty_features=CandidateDifficultyFeatures(source_score=2.0),
    )


def _structural_request() -> CandidateRequest:
    old = ClaimValue(
        raw_value="CNCCO",
        normalized_value="CNCCO",
        value_type=ValueType.SMILES,
        provenance=ValueProvenance.REFERENCE,
    )
    request = object.__new__(CandidateRequest)
    object.__setattr__(
        request,
        "context",
        SimpleNamespace(
            recipe=SimpleNamespace(recipe_id="structural", derived_seed=19),
            record=SimpleNamespace(indexed_smiles="[NH2:2][CH2:1][CH2:3][OH:4]"),
            reference_graph=SimpleNamespace(value_for=lambda _root: old),
            state_schema=SimpleNamespace(
                nodes_by_id={
                    "product": SimpleNamespace(
                        comparator=ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE
                    )
                }
            ),
            truth=SimpleNamespace(
                canonical_gt_smiles="CNCCO",
                valid_anchor_indices=(2,),
            ),
        ),
    )
    object.__setattr__(
        request,
        "resolution",
        SimpleNamespace(
            target_node_id="product",
            registration=SimpleNamespace(
                operator_id="mol_edit.add.valid_wrong_site_product",
                operator_family="attachment_bond_edit",
                subtask=EditingSubtask.ADD,
            ),
        ),
    )
    return request


def test_structural_adapter_reuses_signature_and_local_simulation_before_gate() -> None:
    request = _structural_request()
    arguments = {
        "family": "add",
        "source_smiles": request.context.record.indexed_smiles,
        "candidate_product_smiles": "CC(N)CO",
        "anchor_idx": 1,
        "remove_anchor_idx": None,
        "remove_group_smiles": None,
        "add_fragment_smiles": "C",
        "fragment_attachment_atom": 0,
        "bond_type": "SINGLE",
    }
    execution = SimpleNamespace(
        tool="check_candidate_signature",
        tool_call_id="signature-call-1",
        arguments_json=json.dumps(arguments),
        result_json=json.dumps({"result": {"valid": True}}),
    )
    result = SimpleNamespace(
        provenance=SimpleNamespace(
            attempts=(SimpleNamespace(tool_executions=(execution,)),)
        )
    )
    candidate = ProposalCandidatePatch(
        candidate_id="structural.candidate.1",
        root_field="product",
        replacement=ProposalReplacement(smiles="CC(N)CO"),
        minimal_surface_realization="addition at a neighboring carbon",
        plausibility_reason="The alternate site is locally similar.",
    )

    proposal = verified_candidate_adapter(request, candidate, result)
    assert proposal.patch.edit_action is not None
    assert proposal.patch.edit_action.source_anchor_index == 1
    assert proposal.candidate_product_smiles == "CC(N)CO"
    assert proposal.patch.metadata["typed_action_source"] == "local_simulate_edit"

    gated = T018CandidateGate().validate(
        request,
        (proposal,),
        allowed_sources=frozenset({CandidateSourceType.LLM}),
    )
    assert tuple(item.candidate_id for item in gated.pool.candidates) == (
        "structural.candidate.1",
    )


def test_validate_retry_uses_code_only_feedback_and_complete_pool_selection() -> None:
    request = _candidate_request()
    proposal_request = _proposal_request(request)
    responses = (
        _response(
            proposal_request,
            1,
            (
                ProposalReplacement(integer=5),
                ProposalReplacement(text="wrong payload type"),
                ProposalReplacement(integer=5),
            ),
        ),
        _response(
            proposal_request,
            2,
            (
                ProposalReplacement(integer=6),
                ProposalReplacement(integer=6),
                ProposalReplacement(integer=7),
            ),
        ),
        _response(
            proposal_request,
            3,
            (
                ProposalReplacement(integer=6),
                ProposalReplacement(integer=8),
                ProposalReplacement(integer=9),
            ),
        ),
    )
    client = _RoundClient(proposal_request, responses)
    donor_pool = _DonorPool()
    engine = HybridCandidateEngine(
        llm_source=LLMAgentCandidateSource(client),
        minimum_valid_llm_candidates=3,
    )

    result = engine.build_pool(
        request,
        proposal_request=proposal_request,
        donor_pool=donor_pool,
        donor_query=_donor_query(),
    )

    assert len(client.rounds) == 3
    assert all(
        round_request.request is proposal_request for round_request in client.rounds
    )
    assert client.rounds[0].feedback.reject_codes == ()
    assert client.rounds[1].feedback.reject_codes == (
        "LLM_PROPOSAL_CONVERSION_FAILED",
        "REFERENCE_EQUIVALENT",
    )
    assert "DUPLICATE" in client.rounds[2].feedback.reject_codes
    assert all(
        donor.split == "train" for item in client.rounds for donor in item.donors
    )
    assert len(result.pool.candidates) == 4
    assert result.llm_materially_participated
    assert {item.new_value.normalized_value for item in result.pool.candidates} == {
        6,
        7,
        8,
        9,
    }
    assert all(item.new_value.locally_valid is None for item in result.pool.candidates)
    assert all(item.new_value.oracle_match is None for item in result.pool.candidates)
    assert any(item.code == "DUPLICATE" for item in result.audit_rejections)
    assert not result.fallback.triggered

    decision = FrozenCandidateSelector().select(result)
    assert len(decision.considered_candidate_ids) == len(result.pool.candidates)
    assert decision.selected in result.pool.candidates
    assert set(decision.considered_candidate_ids) == {
        item.candidate_id for item in result.pool.candidates
    }


def test_three_failed_rounds_use_predeclared_same_phenotype_fallback() -> None:
    request = _candidate_request()
    proposal_request = _proposal_request(request)
    invalid = tuple(
        _response(
            proposal_request,
            attempt,
            (
                ProposalReplacement(integer=5),
                ProposalReplacement(text="invalid for integer root"),
                ProposalReplacement(integer=5),
            ),
        )
        for attempt in (1, 2, 3)
    )
    client = _RoundClient(proposal_request, invalid)
    observed_requests: list[CandidateRequest] = []

    def fallback_proposer(candidate_request: CandidateRequest):
        observed_requests.append(candidate_request)
        return (_deterministic_proposal(candidate_request, 10),)

    fallback = RuleCandidateSource(fallback_proposer)
    engine = HybridCandidateEngine(
        llm_source=LLMAgentCandidateSource(client),
        fallback_by_operator={OPERATOR_ID: fallback},
    )
    result = engine.build_pool(
        request,
        proposal_request=proposal_request,
        donor_pool=_DonorPool(),
        donor_query=_donor_query(),
    )

    assert len(client.rounds) == 3
    assert observed_requests == [request]
    assert result.fallback.triggered
    assert result.fallback.operator_id == OPERATOR_ID
    assert result.fallback.policy == "LOCAL"
    assert result.fallback.target_root == ROOT
    assert result.fallback.source is CandidateSourceType.RULE
    assert not result.fallback.scheduler_backfill_required
    assert tuple(item.source for item in result.pool.candidates) == (
        CandidateSourceType.RULE,
    )
    assert result.pool.candidates[0].new_value.normalized_value == 10


def test_missing_fallback_requests_scheduler_backfill_without_relaxing_gate() -> None:
    request = _candidate_request()
    proposal_request = _proposal_request(request)
    abstentions = tuple(
        ProposalResponse(
            request_id=proposal_request.request_id,
            candidates=(),
            abstain_reason="No proposal satisfies the fixed constraints.",
        )
        for _ in range(3)
    )
    engine = HybridCandidateEngine(
        llm_source=LLMAgentCandidateSource(_RoundClient(proposal_request, abstentions))
    )
    result = engine.build_pool(
        request,
        proposal_request=proposal_request,
        donor_pool=_DonorPool(),
        donor_query=_donor_query(),
    )

    assert result.fallback.triggered
    assert result.fallback.source is None
    assert result.fallback.scheduler_backfill_required
    assert result.pool.candidates == ()
    with pytest.raises(CandidateSelectionError, match="EMPTY_VALIDATED_POOL"):
        FrozenCandidateSelector().select(result)


def test_cross_split_donor_query_fails_before_any_model_call() -> None:
    request = _candidate_request()
    proposal_request = _proposal_request(request)
    client = _RoundClient(
        proposal_request,
        (
            _response(
                proposal_request,
                1,
                (
                    ProposalReplacement(integer=6),
                    ProposalReplacement(integer=7),
                    ProposalReplacement(integer=8),
                ),
            ),
        ),
    )
    engine = HybridCandidateEngine(llm_source=LLMAgentCandidateSource(client))

    with pytest.raises(ValueError, match="current manifest split/origin"):
        engine.build_pool(
            request,
            proposal_request=proposal_request,
            donor_pool=_DonorPool(),
            donor_query=_donor_query(split="validation"),
        )
    assert client.rounds == []


def test_llm_only_source_without_retry_adapter_fails_closed_to_fallback() -> None:
    request = _candidate_request()
    proposal_request = _proposal_request(request)
    first_response = _response(
        proposal_request,
        1,
        (
            ProposalReplacement(integer=5),
            ProposalReplacement(integer=5),
            ProposalReplacement(text="invalid for integer root"),
        ),
    )

    class _SingleCallClient:
        def __init__(self) -> None:
            self.calls = 0

        def propose(self, current: ProposalRequest) -> object:
            self.calls += 1
            return SimpleNamespace(request=current, response=first_response)

    client = _SingleCallClient()
    fallback = RuleCandidateSource(
        lambda current: (_deterministic_proposal(current, 11),)
    )
    result = HybridCandidateEngine(
        llm_source=LLMAgentCandidateSource(client),
        fallback_by_operator={OPERATOR_ID: fallback},
    ).build_pool(
        request,
        proposal_request=proposal_request,
        donor_pool=_DonorPool(),
        donor_query=_donor_query(),
    )

    assert client.calls == 1
    assert len(result.llm_attempts) == 2
    assert result.llm_attempts[-1].client_error_code == "LLM_RETRY_UNSUPPORTED"
    assert result.fallback.triggered
    assert result.pool.candidates[0].new_value.normalized_value == 11
