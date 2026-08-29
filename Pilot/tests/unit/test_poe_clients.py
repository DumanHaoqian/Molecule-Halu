"""Mocked transport tests for the T034 Poe client boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from molhallulens.providers.poe.client import (
    POE_BOT_NAME,
    POE_CHAT_ENDPOINT,
    POE_MODEL_ID,
    POE_RESPONSES_ENDPOINT,
    FastApiPoeTextClient,
    PoeAttemptStatus,
    PoeChatCompletionsClient,
    PoeClientError,
    PoeResponsesClient,
    PoeTransport,
    PoeTransportFallbackClient,
)
from molhallulens.providers.poe.schemas import (
    FROZEN_GLOBAL_SEED,
    ProposalConstraints,
    ProposalManifestIdentity,
    ProposalRequest,
    derive_proposal_seed,
)


@dataclass(frozen=True)
class _CatalogEntry:
    id: str = POE_MODEL_ID
    display_name: str = POE_BOT_NAME
    supported_endpoints: tuple[str, ...] = (
        POE_RESPONSES_ENDPOINT,
        POE_CHAT_ENDPOINT,
    )
    supported_features: tuple[str, ...] = ("tools", "web_search")


class _Catalog:
    def __init__(self, entry: _CatalogEntry | None = None) -> None:
        self.entry = _CatalogEntry() if entry is None else entry
        self.required: list[str] = []

    def require(self, model_id: str) -> _CatalogEntry:
        self.required.append(model_id)
        return self.entry


class _QueueTransport:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("mock transport response queue exhausted")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class _Dispatcher:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def dispatch(self, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((tool, arguments))
        return {"canonical_smiles": "CCO", "atom_count": 3}


class _FastApiTransport:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _request() -> ProposalRequest:
    values: dict[str, Any] = {
        "request_id": "request:substitute:0216",
        "origin_id": "mol_edit.substitute_v2.0216",
        "operator_id": "mol_edit.substitute.incoming_fragment_bucket_swap",
        "propagation": "FULL_CF",
        "candidate_source_mode": "HYBRID",
        "target_root": "add_fragment",
        "constraints": ProposalConstraints(
            same_attachment_element=True,
            match_heavy_count=True,
            match_ring_count=True,
        ),
        "global_seed": FROZEN_GLOBAL_SEED,
        "dataset_version": "pilot_v1",
        "variant_index": 0,
        "split": "train",
        "manifest_identity": ProposalManifestIdentity(
            dataset_version="pilot_v1",
            split_seed=8347206628578381721,
            manifest_sha256="a" * 64,
            source_origin_audit_sha256="b" * 64,
            source_split_report_sha256="c" * 64,
        ),
    }
    values["derived_seed"] = derive_proposal_seed(
        global_seed=values["global_seed"],
        dataset_version=values["dataset_version"],
        origin_id=values["origin_id"],
        operator_id=values["operator_id"],
        policy=values["propagation"],
        variant_index=values["variant_index"],
    )
    return ProposalRequest.model_validate(values, strict=True)


def _proposal_json(request: ProposalRequest) -> str:
    return json.dumps(
        {
            "proposal_version": "1.0",
            "request_id": request.request_id,
            "candidates": [
                {
                    "candidate_id": f"c{index}",
                    "root_field": request.target_root,
                    "replacement": {
                        "smiles": smiles,
                        "attachment_atom": 0,
                    },
                    "bond_edits": [],
                    "minimal_surface_realization": f"candidate fragment {smiles}",
                    "plausibility_reason": "A matched local chemical near-neighbor.",
                }
                for index, smiles in enumerate(("N", "O", "S"), start=1)
            ],
            "abstain_reason": None,
        }
    )


def _responses_final(
    request: ProposalRequest, response_id: str = "resp-final"
) -> dict[str, object]:
    return {
        "id": response_id,
        "model": POE_MODEL_ID,
        "query_id": f"query-{response_id}",
        "_request_id": f"x-{response_id}",
        "output_text": _proposal_json(request),
        "output": [],
    }


def _chat_final(
    request: ProposalRequest, response_id: str = "chat-final"
) -> dict[str, object]:
    return {
        "id": response_id,
        "model": POE_MODEL_ID,
        "query_id": f"query-{response_id}",
        "_request_id": f"x-{response_id}",
        "choices": [{"message": {"content": _proposal_json(request)}}],
    }


def test_responses_uses_text_format_and_serial_validated_tool_continuation() -> None:
    request = _request()
    first = {
        "id": "resp-tool",
        "model": POE_MODEL_ID,
        "query_id": "query-tool",
        "_request_id": "x-tool",
        "output": [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "analyze_smiles",
                "arguments": json.dumps({"smiles": "CCO"}),
            }
        ],
    }
    transport = _QueueTransport(first, _responses_final(request))
    dispatcher = _Dispatcher()
    result = PoeResponsesClient(
        transport=transport,
        dispatcher=dispatcher,
        model_catalog=_Catalog(),
    ).propose(request)

    assert result.response.request_id == request.request_id
    assert len(dispatcher.calls) == 1
    assert len(transport.calls) == 2
    initial, continuation = transport.calls
    assert initial["model"] == POE_MODEL_ID
    assert initial["parallel_tool_calls"] is False
    assert "format" in initial["text"]  # type: ignore[operator]
    assert "response_format" not in initial
    assert "seed" not in initial
    assert continuation["previous_response_id"] == "resp-tool"
    assert continuation["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"atom_count":3,"canonical_smiles":"CCO"}',
        }
    ]
    attempt = result.provenance.attempts[0]
    assert attempt.response_ids == ("resp-tool", "resp-final")
    assert attempt.query_ids == ("query-tool", "query-resp-final")
    assert attempt.x_request_ids == ("x-tool", "x-resp-final")
    assert attempt.turns == 2
    assert len(attempt.tool_executions) == 1


def test_invalid_tool_arguments_are_rejected_before_dispatch() -> None:
    request = _request()
    invalid = {
        "id": "resp-invalid-tool",
        "model": POE_MODEL_ID,
        "output": [
            {
                "type": "function_call",
                "call_id": "call-valid-but-not-dispatched",
                "name": "analyze_smiles",
                "arguments": '{"smiles":"CCO"}',
            },
            {
                "type": "function_call",
                "call_id": "call-shell",
                "name": "shell",
                "arguments": '{"command":"rm"}',
            },
        ],
    }
    dispatcher = _Dispatcher()
    client = PoeResponsesClient(
        transport=_QueueTransport(invalid),
        dispatcher=dispatcher,
        model_catalog=_Catalog(),
    )
    with pytest.raises(PoeClientError) as captured:
        client.propose(request)
    assert captured.value.code == "POE_TOOL_ARGUMENT_INVALID"
    assert dispatcher.calls == []
    assert captured.value.attempt is not None
    assert captured.value.attempt.tool_executions == ()


def test_tool_call_and_agent_turn_limits_fail_closed() -> None:
    request = _request()
    seven_calls = {
        "id": "resp-too-many-tools",
        "model": POE_MODEL_ID,
        "output": [
            {
                "type": "function_call",
                "call_id": f"call-{index}",
                "name": "analyze_smiles",
                "arguments": '{"smiles":"CCO"}',
            }
            for index in range(7)
        ],
    }
    dispatcher = _Dispatcher()
    with pytest.raises(PoeClientError) as too_many:
        PoeResponsesClient(
            transport=_QueueTransport(seven_calls),
            dispatcher=dispatcher,
            model_catalog=_Catalog(),
        ).propose(request)
    assert too_many.value.code == "POE_TOOL_CALL_LIMIT"
    assert dispatcher.calls == []

    tool_turns = tuple(
        {
            "id": f"resp-turn-{turn}",
            "model": POE_MODEL_ID,
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"turn-call-{turn}",
                    "name": "analyze_smiles",
                    "arguments": '{"smiles":"CCO"}',
                }
            ],
        }
        for turn in range(1, 4)
    )
    turn_dispatcher = _Dispatcher()
    with pytest.raises(PoeClientError) as turn_limit:
        PoeResponsesClient(
            transport=_QueueTransport(*tool_turns),
            dispatcher=turn_dispatcher,
            model_catalog=_Catalog(),
        ).propose(request)
    assert turn_limit.value.code == "POE_AGENT_TURN_LIMIT"
    assert turn_limit.value.attempt is not None
    assert turn_limit.value.attempt.turns == 3
    assert len(turn_limit.value.attempt.tool_executions) == 3


def test_chat_tool_loop_omits_response_format_strict_and_seed() -> None:
    request = _request()
    first = {
        "id": "chat-tool",
        "model": POE_MODEL_ID,
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "chat-call-1",
                            "type": "function",
                            "function": {
                                "name": "analyze_smiles",
                                "arguments": '{"smiles":"CCO"}',
                            },
                        }
                    ],
                }
            }
        ],
    }
    transport = _QueueTransport(first, _chat_final(request))
    dispatcher = _Dispatcher()
    result = PoeChatCompletionsClient(
        transport=transport,
        dispatcher=dispatcher,
        model_catalog=_Catalog(),
    ).propose(request)

    assert result.provenance.selected_transport is PoeTransport.CHAT_COMPLETIONS
    assert len(dispatcher.calls) == 1
    assert len(transport.calls) == 2
    for call in transport.calls:
        assert call["model"] == POE_MODEL_ID
        assert call["parallel_tool_calls"] is False
        assert "response_format" not in call
        assert "seed" not in call
        assert "strict" not in json.dumps(call["tools"])
    continuation_messages = transport.calls[1]["messages"]
    assert isinstance(continuation_messages, list)
    assert continuation_messages[-1]["role"] == "tool"
    assert continuation_messages[-1]["tool_call_id"] == "chat-call-1"


def test_fastapi_poe_is_tool_free_simple_text_and_tracks_query_ids() -> None:
    request = _request()
    transport = _FastApiTransport(
        [
            {
                "text": _proposal_json(request),
                "id": "poe-response-1",
                "query_id": "poe-query-1",
                "headers": {"x-request-id": "poe-x-1"},
            }
        ]
    )
    result = FastApiPoeTextClient(
        transport=transport,
        model_catalog=_Catalog(),
    ).propose(request)

    assert result.provenance.selected_transport is PoeTransport.FASTAPI_POE
    assert transport.calls[0]["bot_name"] == POE_BOT_NAME
    assert set(transport.calls[0]) == {"messages", "bot_name"}
    attempt = result.provenance.attempts[0]
    assert attempt.response_ids == ("poe-response-1",)
    assert attempt.query_ids == ("poe-query-1",)
    assert attempt.x_request_ids == ("poe-x-1",)
    assert attempt.tool_executions == ()


def test_fallback_records_failures_without_changing_model_or_request() -> None:
    request = _request()
    catalog = _Catalog()
    dispatcher = _Dispatcher()
    responses_transport = _QueueTransport(RuntimeError("responses unavailable"))
    chat_transport = _QueueTransport(
        {
            "id": "chat-invalid",
            "model": POE_MODEL_ID,
            "choices": [{"message": {"content": "not json"}}],
        }
    )
    fast_transport = _FastApiTransport([{"text": _proposal_json(request)}])
    client = PoeTransportFallbackClient(
        responses=PoeResponsesClient(
            transport=responses_transport,
            dispatcher=dispatcher,
            model_catalog=catalog,
        ),
        chat_completions=PoeChatCompletionsClient(
            transport=chat_transport,
            dispatcher=dispatcher,
            model_catalog=catalog,
        ),
        fastapi_poe=FastApiPoeTextClient(
            transport=fast_transport,
            model_catalog=catalog,
        ),
    )
    result = client.propose(request)

    assert result.request is request
    assert result.provenance.selected_transport is PoeTransport.FASTAPI_POE
    assert tuple(item.status for item in result.provenance.attempts) == (
        PoeAttemptStatus.FAILED,
        PoeAttemptStatus.FAILED,
        PoeAttemptStatus.SUCCEEDED,
    )
    assert tuple(item.error_code for item in result.provenance.attempts) == (
        "POE_RESPONSES_TRANSPORT_ERROR",
        "POE_PROPOSAL_INVALID",
        None,
    )
    assert all(
        item.request_id == request.request_id
        and item.requested_model_id == POE_MODEL_ID
        for item in result.provenance.attempts
    )
    assert result.provenance.operator_id == request.operator_id
    assert result.provenance.propagation == request.propagation
    assert result.provenance.target_root == request.target_root
    assert responses_transport.calls[0]["model"] == POE_MODEL_ID
    assert chat_transport.calls[0]["model"] == POE_MODEL_ID
    assert fast_transport.calls[0]["bot_name"] == POE_BOT_NAME


def test_catalog_capability_and_response_model_changes_fail_before_acceptance() -> None:
    request = _request()
    missing_tools = _Catalog(_CatalogEntry(supported_features=("web_search",)))
    untouched = _QueueTransport(_responses_final(request))
    with pytest.raises(PoeClientError) as captured:
        PoeResponsesClient(
            transport=untouched,
            dispatcher=_Dispatcher(),
            model_catalog=missing_tools,
        ).propose(request)
    assert captured.value.code == "POE_MODEL_CAPABILITY_UNAVAILABLE"
    assert untouched.calls == []

    changed_model = _responses_final(request)
    changed_model["model"] = "another-model"
    with pytest.raises(PoeClientError) as mismatch:
        PoeResponsesClient(
            transport=_QueueTransport(changed_model),
            dispatcher=_Dispatcher(),
            model_catalog=_Catalog(),
        ).propose(request)
    assert mismatch.value.code == "POE_RESPONSE_MODEL_MISMATCH"
