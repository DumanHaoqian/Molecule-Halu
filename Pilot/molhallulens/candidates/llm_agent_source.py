"""Proposal-only LLM candidate source with an explicit retry boundary.

The proposal agent is deliberately unable to accept a candidate or assign a
label.  It returns untrusted :class:`CandidateProposal` values which still
have to cross the deterministic T018 gate in ``hybrid_engine``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from molhallulens.domain import (
    BondTypeName,
    CandidatePatch,
    CandidateSourceType,
    EditAction,
    EditKind,
    ValueProvenance,
    ValueType,
)
from molhallulens.providers.poe.client import POE_MODEL_ID
from molhallulens.providers.poe.schemas import (
    CheckCandidateSignatureArgs,
    ProposalCandidatePatch,
    ProposalRequest,
    ProposalResponse,
    SimulateEditArgs,
)

from .core import (
    CandidateDifficultyFeatures,
    CandidateProposal,
    CandidateRejectCode,
    CandidateRequest,
)
from .donors import DonorEntry

MAX_LLM_CANDIDATE_ATTEMPTS = 3


class LLMProposalRejectCode(StrEnum):
    """Stable local failures; none of these imply candidate acceptance."""

    CLIENT_FAILED = "LLM_CLIENT_FAILED"
    RESPONSE_INVALID = "LLM_RESPONSE_INVALID"
    RETRY_UNSUPPORTED = "LLM_RETRY_UNSUPPORTED"
    PROPOSAL_CONVERSION_FAILED = "LLM_PROPOSAL_CONVERSION_FAILED"
    PROPOSAL_ABSTAINED = "LLM_PROPOSAL_ABSTAINED"


class LLMProposalSourceError(RuntimeError):
    """Fail-closed proposal-source error safe to place in an audit ledger."""

    def __init__(self, code: LLMProposalRejectCode, detail: str) -> None:
        if type(code) is not LLMProposalRejectCode:
            raise TypeError("code must be LLMProposalRejectCode")
        if type(detail) is not str or not detail or "\n" in detail or "\r" in detail:
            raise ValueError("detail must be non-empty single-line text")
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class LLMCandidateConversionError(ValueError):
    """Deterministic pre-gate rejection discovered while typing a proposal."""

    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("detail must be non-empty text")
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class LLMRetryFeedback:
    """The only validator information returned to a retrying proposal agent."""

    reject_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        codes = tuple(sorted(set(self.reject_codes)))
        if any(type(code) is not str or not code for code in codes):
            raise TypeError("reject_codes must contain non-empty strings")
        object.__setattr__(self, "reject_codes", codes)

    def to_dict(self) -> dict[str, object]:
        return {"reject_codes": list(self.reject_codes)}


@dataclass(frozen=True, slots=True)
class LLMProposalRound:
    """One fixed request plus split-local donors and code-only retry feedback."""

    request: ProposalRequest
    attempt_index: int
    feedback: LLMRetryFeedback = LLMRetryFeedback()
    donors: tuple[DonorEntry, ...] = ()

    def __post_init__(self) -> None:
        if type(self.request) is not ProposalRequest:
            raise TypeError("request must be ProposalRequest")
        if type(self.attempt_index) is not int or not (
            1 <= self.attempt_index <= MAX_LLM_CANDIDATE_ATTEMPTS
        ):
            raise ValueError("attempt_index must be in [1, 3]")
        if type(self.feedback) is not LLMRetryFeedback:
            raise TypeError("feedback must be LLMRetryFeedback")
        donors = tuple(self.donors)
        if any(type(donor) is not DonorEntry for donor in donors):
            raise TypeError("donors must contain DonorEntry values")
        if any(donor.split != self.request.split for donor in donors):
            raise ValueError("proposal rounds may contain only current-split donors")
        object.__setattr__(self, "donors", donors)


@runtime_checkable
class LLMProposalRoundClient(Protocol):
    """Retry-aware adapter normally wrapped around a T034/T036 client."""

    def propose_round(self, round_request: LLMProposalRound) -> object: ...


@runtime_checkable
class LLMProposalClient(Protocol):
    """The single-request public surface implemented by T034 clients."""

    def propose(self, request: ProposalRequest) -> object: ...


CandidateProposalAdapter = Callable[
    [CandidateRequest, ProposalCandidatePatch, object], CandidateProposal
]


@dataclass(frozen=True, slots=True)
class LLMSourceRejection:
    code: str
    proposal_id: str
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.code) is not str or not self.code:
            raise ValueError("rejection code must be non-empty text")
        if type(self.proposal_id) is not str or not self.proposal_id:
            raise ValueError("proposal_id must be non-empty text")
        if not isinstance(self.evidence, dict):
            raise TypeError("evidence must be a dict")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class LLMProposalBatch:
    """Converted but still unaccepted proposals from one model response."""

    proposals: tuple[CandidateProposal, ...]
    rejections: tuple[LLMSourceRejection, ...]
    response_candidate_ids: tuple[str, ...]
    abstain_reason: str | None
    client_result: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        rejections = tuple(self.rejections)
        candidate_ids = tuple(self.response_candidate_ids)
        if any(type(item) is not CandidateProposal for item in proposals):
            raise TypeError("proposals must contain CandidateProposal values")
        if any(type(item) is not LLMSourceRejection for item in rejections):
            raise TypeError("rejections must contain LLMSourceRejection values")
        if any(type(item) is not str or not item for item in candidate_ids):
            raise TypeError("response_candidate_ids must contain non-empty strings")
        if self.abstain_reason is not None and (
            type(self.abstain_reason) is not str or not self.abstain_reason
        ):
            raise ValueError("abstain_reason must be non-empty text or None")
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "response_candidate_ids", candidate_ids)

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(sorted({item.code for item in self.rejections}))


def _replacement_value(candidate: ProposalCandidatePatch, old_value: object) -> object:
    replacement = candidate.replacement
    value_type = getattr(old_value, "value_type", None)
    if value_type in {
        ValueType.SMILES,
        ValueType.INDEXED_SMILES,
        ValueType.MOLECULE,
        ValueType.FRAGMENT,
    }:
        value = replacement.smiles
    elif value_type is ValueType.ATOM_INDEX:
        value = replacement.atom_index
    elif value_type in {ValueType.INTEGER, ValueType.COUNT}:
        value = replacement.integer
    elif value_type in {ValueType.STRING, ValueType.ELEMENT}:
        value = replacement.text
    else:
        raise ValueError("proposal replacement does not support this root value type")
    if value is None:
        raise ValueError("proposal replacement type does not match the target root")
    return value


def scalar_candidate_adapter(
    request: CandidateRequest,
    candidate: ProposalCandidatePatch,
    _client_result: object,
) -> CandidateProposal:
    """Convert scalar/root proposals; structural actions use an injected adapter.

    This adapter never marks local validity, oracle correctness, acceptance, or
    labels.  Product/action proposals without a typed edit action consequently
    fail the T018 gate instead of being trusted.
    """

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if type(candidate) is not ProposalCandidatePatch:
        raise TypeError("candidate must be ProposalCandidatePatch")
    old = request.context.reference_graph.value_for(request.resolution.target_node_id)
    value = _replacement_value(candidate, old)
    new = replace(
        old,
        raw_value=value,
        normalized_value=value,
        provenance=ValueProvenance.LLM,
        locally_valid=None,
        oracle_match=None,
        confidence=None,
        mention_ids=(),
    )
    if old.semantically_equals(new):
        raise LLMCandidateConversionError(
            CandidateRejectCode.REFERENCE_EQUIVALENT.value,
            "proposal replacement is equivalent to the reference root",
        )
    proposal_id = f"llm:{candidate.candidate_id}"
    return CandidateProposal(
        proposal_id=proposal_id,
        patch=CandidatePatch(
            candidate_id=candidate.candidate_id,
            root_node_id=candidate.root_field,
            old_value=old,
            new_value=new,
            edit_action=None,
            source=CandidateSourceType.LLM,
            metadata={
                "generator": "poe_proposal_agent",
                "minimal_surface_realization": candidate.minimal_surface_realization,
                "plausibility_reason": candidate.plausibility_reason,
                "bond_edits": tuple(
                    (
                        edit.operation,
                        edit.begin_atom,
                        edit.end_atom,
                        edit.bond_type,
                    )
                    for edit in candidate.bond_edits
                ),
            },
        ),
        candidate_product_smiles=(
            value
            if old.value_type in {ValueType.SMILES, ValueType.MOLECULE}
            and type(value) is str
            else None
        ),
        difficulty_features=CandidateDifficultyFeatures(source_score=0.5),
    )


def _tool_executions(client_result: object) -> tuple[object, ...]:
    provenance = getattr(client_result, "provenance", None)
    attempts = tuple(getattr(provenance, "attempts", ()))
    return tuple(
        execution
        for attempt in attempts
        for execution in tuple(getattr(attempt, "tool_executions", ()))
    )


def _result_payload(result_json: str) -> dict[str, object]:
    decoded = json.loads(result_json)
    if not isinstance(decoded, dict):
        raise TypeError("chemistry tool result must be a JSON object")
    nested = decoded.get("result")
    if isinstance(nested, dict):
        return nested
    return decoded


def _candidate_matches_signature(
    request: CandidateRequest,
    candidate: ProposalCandidatePatch,
    arguments: CheckCandidateSignatureArgs,
) -> bool:
    from molhallulens.chemistry import (
        fragment_graph_equivalent,
        isomeric_graph_equivalent,
    )

    root = request.resolution.target_node_id
    replacement = candidate.replacement
    if root in {"product", "final_answer"}:
        return replacement.smiles is not None and isomeric_graph_equivalent(
            replacement.smiles,
            arguments.candidate_product_smiles,
        )
    if root in {
        "add_fragment",
        "remove_group",
        "remove_group_step1",
        "remove_group_step2",
    }:
        expected = (
            arguments.add_fragment_smiles
            if root == "add_fragment"
            else arguments.remove_group_smiles
        )
        return (
            replacement.smiles is not None
            and expected is not None
            and fragment_graph_equivalent(replacement.smiles, expected)
        )
    if root == "anchor_idx":
        return replacement.atom_index == arguments.anchor_idx
    return False


def _typed_action_from_tool_evidence(
    request: CandidateRequest,
    candidate: ProposalCandidatePatch,
    client_result: object,
) -> tuple[EditAction, str, str]:
    from molhallulens.chemistry import isomeric_graph_equivalent
    from molhallulens.providers.poe.chemistry_tools import dispatch_chemistry_tool

    source_smiles = request.context.record.indexed_smiles
    matches: list[tuple[EditAction, str, str]] = []
    for execution in _tool_executions(client_result):
        if getattr(execution, "tool", None) != "check_candidate_signature":
            continue
        arguments_json = getattr(execution, "arguments_json", None)
        result_json = getattr(execution, "result_json", None)
        tool_call_id = getattr(execution, "tool_call_id", None)
        if not all(
            type(value) is str and value
            for value in (
                arguments_json,
                result_json,
                tool_call_id,
            )
        ):
            continue
        try:
            arguments = CheckCandidateSignatureArgs.model_validate_json(
                arguments_json,
                strict=True,
            )
            signature = _result_payload(result_json)
        except (TypeError, ValueError):
            continue
        if (
            arguments.source_smiles != source_smiles
            or signature.get("valid") is not True
            or not _candidate_matches_signature(request, candidate, arguments)
        ):
            continue
        simulation_arguments = SimulateEditArgs.model_validate(
            {
                key: value
                for key, value in arguments.model_dump(mode="python").items()
                if key != "candidate_product_smiles"
            },
            strict=True,
        )
        simulation_result = dispatch_chemistry_tool(
            "simulate_edit",
            simulation_arguments.model_dump(mode="python"),
        ).result
        for product in simulation_result.get("products", []):
            if not isinstance(product, dict):
                continue
            product_smiles = product.get("product_smiles")
            action_value = product.get("action")
            if (
                type(product_smiles) is not str
                or not isinstance(action_value, dict)
                or not isomeric_graph_equivalent(
                    product_smiles,
                    arguments.candidate_product_smiles,
                )
            ):
                continue
            occurrence = action_value.get("occurrence_atom_maps", [])
            if not isinstance(occurrence, list):
                continue
            action = EditAction(
                edit_kind=EditKind(action_value["edit_kind"]),
                source_anchor_index=action_value.get("source_anchor_index"),
                remove_anchor_index=action_value.get("remove_anchor_index"),
                remove_fragment_smiles=action_value.get("remove_fragment_smiles"),
                add_fragment_smiles=action_value.get("add_fragment_smiles"),
                fragment_attachment_atom=action_value.get("fragment_attachment_atom"),
                bond_type=(
                    None
                    if action_value.get("bond_type") is None
                    else BondTypeName(action_value["bond_type"])
                ),
                metadata=(
                    {}
                    if not occurrence
                    else {"occurrence_atom_maps": tuple(occurrence)}
                ),
            )
            matches.append((action, product_smiles, tool_call_id))
    unique: dict[tuple[object, ...], tuple[EditAction, str, str]] = {}
    for action, product, tool_call_id in matches:
        identity = (
            action.edit_kind,
            action.source_anchor_index,
            action.remove_anchor_index,
            action.remove_fragment_smiles,
            action.add_fragment_smiles,
            action.fragment_attachment_atom,
            action.bond_type,
            tuple(action.metadata.get("occurrence_atom_maps", ())),
            product,
        )
        unique[identity] = (action, product, tool_call_id)
    if len(unique) != 1:
        raise ValueError(
            "structural proposal requires one unambiguous validated signature tool run"
        )
    return next(iter(unique.values()))


def verified_candidate_adapter(
    request: CandidateRequest,
    candidate: ProposalCandidatePatch,
    client_result: object,
) -> CandidateProposal:
    """Convert scalars directly and structural roots from verified tool evidence."""

    structural_root = request.resolution.target_node_id in {
        "product",
        "final_answer",
        "add_fragment",
        "remove_group",
        "remove_group_step1",
        "remove_group_step2",
        "anchor_idx",
    }
    if not structural_root:
        return scalar_candidate_adapter(request, candidate, client_result)

    action, product_smiles, tool_call_id = _typed_action_from_tool_evidence(
        request,
        candidate,
        client_result,
    )
    base = scalar_candidate_adapter(request, candidate, client_result)
    return replace(
        base,
        patch=replace(
            base.patch,
            edit_action=action,
            metadata={
                **dict(base.patch.metadata),
                "signature_tool_call_id": tool_call_id,
                "typed_action_source": "local_simulate_edit",
            },
        ),
        candidate_product_smiles=product_smiles,
    )


@dataclass(frozen=True, slots=True)
class LLMAgentCandidateSource:
    """Invoke one proposal round without making any accept/reject decision."""

    client: object = field(repr=False, compare=False)
    candidate_adapter: CandidateProposalAdapter = verified_candidate_adapter
    source_type: CandidateSourceType = field(
        default=CandidateSourceType.LLM, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.client, (LLMProposalRoundClient, LLMProposalClient)):
            raise TypeError("client must implement propose_round or propose")
        if not callable(self.candidate_adapter):
            raise TypeError("candidate_adapter must be callable")

    def _invoke(self, round_request: LLMProposalRound) -> object:
        if isinstance(self.client, LLMProposalRoundClient):
            return self.client.propose_round(round_request)
        if round_request.attempt_index != 1:
            raise LLMProposalSourceError(
                LLMProposalRejectCode.RETRY_UNSUPPORTED,
                "the injected client has no code-feedback retry adapter",
            )
        assert isinstance(self.client, LLMProposalClient)
        return self.client.propose(round_request.request)

    def propose_round(
        self,
        request: CandidateRequest,
        round_request: LLMProposalRound,
    ) -> LLMProposalBatch:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        if type(round_request) is not LLMProposalRound:
            raise TypeError("round_request must be LLMProposalRound")
        try:
            raw_result = self._invoke(round_request)
        except LLMProposalSourceError:
            raise
        except Exception as error:
            raise LLMProposalSourceError(
                LLMProposalRejectCode.CLIENT_FAILED,
                f"proposal client raised {type(error).__name__}",
            ) from error

        result = getattr(raw_result, "result", raw_result)
        result_request = getattr(result, "request", None)
        response = getattr(result, "response", None)
        if (
            result_request != round_request.request
            or type(response) is not ProposalResponse
        ):
            raise LLMProposalSourceError(
                LLMProposalRejectCode.RESPONSE_INVALID,
                "proposal result is not bound to the unchanged request",
            )
        provenance = getattr(result, "provenance", None)
        if provenance is not None and (
            getattr(provenance, "requested_model_id", None) != POE_MODEL_ID
        ):
            raise LLMProposalSourceError(
                LLMProposalRejectCode.RESPONSE_INVALID,
                "proposal retry changed the frozen model identity",
            )
        try:
            response.validate_for_request(round_request.request)
        except (TypeError, ValueError) as error:
            raise LLMProposalSourceError(
                LLMProposalRejectCode.RESPONSE_INVALID,
                "proposal response failed local request binding",
            ) from error

        proposals: list[CandidateProposal] = []
        rejections: list[LLMSourceRejection] = []
        for candidate in response.candidates:
            try:
                proposal = self.candidate_adapter(request, candidate, result)
                if type(proposal) is not CandidateProposal:
                    raise TypeError("candidate adapter returned a non-proposal")
                if (
                    proposal.patch.source is not CandidateSourceType.LLM
                    or proposal.patch.new_value.provenance is not ValueProvenance.LLM
                ):
                    raise ValueError("candidate adapter changed LLM provenance")
            except LLMCandidateConversionError as error:
                rejections.append(
                    LLMSourceRejection(
                        code=error.code,
                        proposal_id=f"llm:{candidate.candidate_id}",
                        evidence={"phase": "proposal_conversion"},
                    )
                )
                continue
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                rejections.append(
                    LLMSourceRejection(
                        code=LLMProposalRejectCode.PROPOSAL_CONVERSION_FAILED.value,
                        proposal_id=f"llm:{candidate.candidate_id}",
                        evidence={"exception_type": type(error).__name__},
                    )
                )
                continue
            proposals.append(proposal)
        if not response.candidates:
            rejections.append(
                LLMSourceRejection(
                    code=LLMProposalRejectCode.PROPOSAL_ABSTAINED.value,
                    proposal_id=f"source:{request.request_id}",
                    evidence={"attempt_index": round_request.attempt_index},
                )
            )
        return LLMProposalBatch(
            proposals=tuple(proposals),
            rejections=tuple(rejections),
            response_candidate_ids=tuple(
                candidate.candidate_id for candidate in response.candidates
            ),
            abstain_reason=response.abstain_reason,
            client_result=result,
        )


__all__ = [
    "MAX_LLM_CANDIDATE_ATTEMPTS",
    "CandidateProposalAdapter",
    "LLMAgentCandidateSource",
    "LLMCandidateConversionError",
    "LLMProposalBatch",
    "LLMProposalClient",
    "LLMProposalRejectCode",
    "LLMProposalRound",
    "LLMProposalRoundClient",
    "LLMProposalSourceError",
    "LLMRetryFeedback",
    "LLMSourceRejection",
    "scalar_candidate_adapter",
    "verified_candidate_adapter",
]
