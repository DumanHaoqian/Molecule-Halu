"""Poe proposal transports with local tool and schema trust boundaries.

The adapters in this module do not create SDK clients, read API keys, or make
network requests by themselves.  A caller injects already-configured transport
objects, the verified model catalog, and the read-only chemistry dispatcher.
This keeps secrets outside artifacts and makes every transport path testable
without spend.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from .schemas import (
    CHEMISTRY_TOOL_ARGUMENT_MODELS,
    ChemistryToolCall,
    ProposalRequest,
    ProposalResponse,
    parse_chemistry_tool_call,
    parse_proposal_response,
    proposal_response_json_schema,
)

POE_MODEL_ID = "gpt-5.4-mini"
POE_BOT_NAME = "GPT-5.4-Mini"
POE_RESPONSES_ENDPOINT = "/v1/responses"
POE_CHAT_ENDPOINT = "/v1/chat/completions"
MAX_AGENT_TURNS = 3
MAX_CHEMISTRY_TOOL_CALLS = 6

PROPOSAL_SYSTEM_PROMPT = (
    "Propose only candidates for the single requested molecular-edit root. "
    "Use only the supplied read-only chemistry tools. Return proposal_v1 JSON; "
    "never decide acceptance, labels, split, operator, or propagation policy."
)


class PoeTransport(StrEnum):
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"
    FASTAPI_POE = "fastapi_poe"


class PoeAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@runtime_checkable
class PoeModelCatalog(Protocol):
    """Minimum T033 verified-catalog surface consumed by T034."""

    def require(self, model_id: str) -> object: ...


@runtime_checkable
class ChemistryToolDispatcher(Protocol):
    """T032 boundary: dispatch an already validated allow-listed call."""

    def dispatch(self, tool: str, arguments: Mapping[str, object]) -> object: ...


@runtime_checkable
class CreateTransport(Protocol):
    """OpenAI SDK resource-like object (``responses`` or ``completions``)."""

    def create(self, **kwargs: object) -> object: ...


FastApiPoeTransport = Callable[..., object]


def _single_line(value: str, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{name} must be non-empty trimmed single-line text")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: object) -> object:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("JSON object keys must be strings")
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return _json_value(to_json_dict())
    raise TypeError(f"tool result is not closed JSON: {type(value).__name__}")


def _member(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_identifier(value: object) -> str | None:
    return value if type(value) is str and value else None


def _header_identifier(response: object, header: str) -> str | None:
    candidates = (
        _member(response, "headers"),
        _member(response, "response_headers"),
        _member(response, "_headers"),
    )
    lowered = header.lower()
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            for key, value in candidate.items():
                if type(key) is str and key.lower() == lowered:
                    return _optional_identifier(value)
    return None


def _response_identifiers(
    response: object,
) -> tuple[str | None, str | None, str | None]:
    response_id = _optional_identifier(_member(response, "id"))
    query_id = _optional_identifier(_member(response, "query_id"))
    if query_id is None:
        query_id = _header_identifier(response, "query-id")
    request_id = _optional_identifier(_member(response, "_request_id"))
    if request_id is None:
        request_id = _optional_identifier(_member(response, "request_id"))
    if request_id is None:
        request_id = _header_identifier(response, "x-request-id")
    return response_id, query_id, request_id


@dataclass(frozen=True, slots=True)
class PoeClientConfig:
    """Frozen parameters shared by every proposal transport."""

    model_id: str = POE_MODEL_ID
    bot_name: str = POE_BOT_NAME
    reasoning_effort: str = "medium"
    max_agent_turns: int = MAX_AGENT_TURNS
    max_tool_calls: int = MAX_CHEMISTRY_TOOL_CALLS

    def __post_init__(self) -> None:
        if self.model_id != POE_MODEL_ID:
            raise ValueError("T034 is frozen to gpt-5.4-mini")
        if self.bot_name != POE_BOT_NAME:
            raise ValueError("T034 is frozen to GPT-5.4-Mini")
        if self.reasoning_effort != "medium":
            raise ValueError("proposal reasoning effort must be medium")
        if self.max_agent_turns != MAX_AGENT_TURNS:
            raise ValueError("proposal agent turn limit must be three")
        if self.max_tool_calls != MAX_CHEMISTRY_TOOL_CALLS:
            raise ValueError("proposal chemistry tool-call limit must be six")


@dataclass(frozen=True, slots=True)
class PoeToolExecution:
    turn: int
    sequence: int
    tool_call_id: str
    tool: str
    arguments_json: str
    result_json: str

    def __post_init__(self) -> None:
        if type(self.turn) is not int or not 1 <= self.turn <= MAX_AGENT_TURNS:
            raise ValueError("tool execution turn is outside the frozen limit")
        if (
            type(self.sequence) is not int
            or not 1 <= self.sequence <= MAX_CHEMISTRY_TOOL_CALLS
        ):
            raise ValueError("tool execution sequence is outside the frozen limit")
        for value, name in (
            (self.tool_call_id, "tool_call_id"),
            (self.tool, "tool"),
            (self.arguments_json, "arguments_json"),
            (self.result_json, "result_json"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "sequence": self.sequence,
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "arguments_json": self.arguments_json,
            "result_json": self.result_json,
        }


@dataclass(frozen=True, slots=True)
class PoeTransportAttempt:
    transport: PoeTransport
    status: PoeAttemptStatus
    request_id: str
    requested_model_id: str
    response_model: str | None
    response_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    x_request_ids: tuple[str, ...]
    turns: int
    tool_executions: tuple[PoeToolExecution, ...]
    error_code: str | None = None
    error_detail: str | None = None

    def __post_init__(self) -> None:
        if type(self.transport) is not PoeTransport:
            raise TypeError("transport must be PoeTransport")
        if type(self.status) is not PoeAttemptStatus:
            raise TypeError("status must be PoeAttemptStatus")
        _single_line(self.request_id, "request_id")
        if self.requested_model_id != POE_MODEL_ID:
            raise ValueError("attempt requested_model_id changed")
        if self.response_model is not None:
            _single_line(self.response_model, "response_model")
        if type(self.turns) is not int or not 0 <= self.turns <= MAX_AGENT_TURNS:
            raise ValueError("attempt turns are outside the frozen limit")
        if len(self.tool_executions) > MAX_CHEMISTRY_TOOL_CALLS:
            raise ValueError("attempt exceeds the chemistry tool-call limit")
        for values, name in (
            (self.response_ids, "response_ids"),
            (self.query_ids, "query_ids"),
            (self.x_request_ids, "x_request_ids"),
        ):
            for value in values:
                _single_line(value, name)
        failed = self.status is PoeAttemptStatus.FAILED
        if failed != (self.error_code is not None and self.error_detail is not None):
            raise ValueError("failed attempt must carry exactly one structured error")
        if self.error_code is not None:
            _single_line(self.error_code, "error_code")
            _single_line(self.error_detail or "", "error_detail")

    def to_dict(self) -> dict[str, object]:
        return {
            "transport": self.transport.value,
            "status": self.status.value,
            "request_id": self.request_id,
            "requested_model_id": self.requested_model_id,
            "response_model": self.response_model,
            "response_ids": list(self.response_ids),
            "query_ids": list(self.query_ids),
            "x_request_ids": list(self.x_request_ids),
            "turns": self.turns,
            "tool_executions": [item.to_dict() for item in self.tool_executions],
            "error_code": self.error_code,
            "error_detail": self.error_detail,
        }


@dataclass(frozen=True, slots=True)
class PoeClientProvenance:
    provider: str
    request_id: str
    origin_id: str
    operator_id: str
    propagation: str
    candidate_source_mode: str
    target_root: str
    constraints_json: str
    requested_model_id: str
    selected_transport: PoeTransport
    attempts: tuple[PoeTransportAttempt, ...]

    def __post_init__(self) -> None:
        if self.provider != "poe":
            raise ValueError("provider must be poe")
        for value, name in (
            (self.request_id, "request_id"),
            (self.origin_id, "origin_id"),
            (self.operator_id, "operator_id"),
            (self.propagation, "propagation"),
            (self.candidate_source_mode, "candidate_source_mode"),
            (self.target_root, "target_root"),
            (self.constraints_json, "constraints_json"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.requested_model_id != POE_MODEL_ID:
            raise ValueError("provenance requested_model_id changed")
        if (
            not self.attempts
            or self.attempts[-1].status is not PoeAttemptStatus.SUCCEEDED
        ):
            raise ValueError("provenance must end in a successful attempt")
        if self.attempts[-1].transport is not self.selected_transport:
            raise ValueError("selected transport does not match the final attempt")
        if any(
            item.request_id != self.request_id
            or item.requested_model_id != self.requested_model_id
            for item in self.attempts
        ):
            raise ValueError("fallback attempts changed request or model identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "request_id": self.request_id,
            "origin_id": self.origin_id,
            "operator_id": self.operator_id,
            "propagation": self.propagation,
            "candidate_source_mode": self.candidate_source_mode,
            "target_root": self.target_root,
            "constraints_json": self.constraints_json,
            "requested_model_id": self.requested_model_id,
            "selected_transport": self.selected_transport.value,
            "attempts": [item.to_dict() for item in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class PoeClientResult:
    request: ProposalRequest
    response: ProposalResponse
    provenance: PoeClientProvenance

    def __post_init__(self) -> None:
        if type(self.request) is not ProposalRequest:
            raise TypeError("request must be ProposalRequest")
        if type(self.response) is not ProposalResponse:
            raise TypeError("response must be ProposalResponse")
        if type(self.provenance) is not PoeClientProvenance:
            raise TypeError("provenance must be PoeClientProvenance")
        self.response.validate_for_request(self.request)
        if self.provenance.request_id != self.request.request_id:
            raise ValueError("result provenance is bound to another request")


class PoeClientError(RuntimeError):
    """Structured client failure safe for fallback and audit ledgers."""

    def __init__(
        self,
        *,
        code: str,
        transport: PoeTransport,
        detail: str,
        request_id: str,
        attempt: PoeTransportAttempt | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        _single_line(code, "code")
        if type(transport) is not PoeTransport:
            raise TypeError("transport must be PoeTransport")
        _single_line(detail, "detail")
        _single_line(request_id, "request_id")
        if attempt is not None and (
            type(attempt) is not PoeTransportAttempt
            or attempt.status is not PoeAttemptStatus.FAILED
        ):
            raise TypeError("attempt must be a failed PoeTransportAttempt or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("evidence must be a mapping or None")
        self.code = code
        self.transport = transport
        self.detail = detail
        self.request_id = request_id
        self.attempt = attempt
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code} via {transport.value}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "transport": self.transport.value,
            "detail": self.detail,
            "request_id": self.request_id,
            "attempt": None if self.attempt is None else self.attempt.to_dict(),
            "evidence": dict(self.evidence),
        }


class PoeFallbackExhaustedError(PoeClientError):
    def __init__(
        self, *, request_id: str, attempts: tuple[PoeTransportAttempt, ...]
    ) -> None:
        self.attempts = attempts
        super().__init__(
            code="POE_FALLBACK_EXHAUSTED",
            transport=attempts[-1].transport,
            detail="all configured Poe transports failed closed",
            request_id=request_id,
            attempt=attempts[-1],
            evidence={"attempt_codes": [item.error_code for item in attempts]},
        )


@dataclass(slots=True)
class _AttemptState:
    request: ProposalRequest
    transport: PoeTransport
    response_ids: list[str]
    query_ids: list[str]
    x_request_ids: list[str]
    tool_executions: list[PoeToolExecution]
    turns: int = 0
    response_model: str | None = None

    @classmethod
    def new(cls, request: ProposalRequest, transport: PoeTransport) -> _AttemptState:
        return cls(request, transport, [], [], [], [])

    def observe(self, response: object, *, require_model: bool = False) -> None:
        response_id, query_id, request_id = _response_identifiers(response)
        if response_id is not None:
            self.response_ids.append(response_id)
        if query_id is not None:
            self.query_ids.append(query_id)
        if request_id is not None:
            self.x_request_ids.append(request_id)
        response_model = _optional_identifier(_member(response, "model"))
        if require_model and response_model is None:
            raise ValueError("provider response omitted the requested model identity")
        if response_model is not None:
            if response_model != POE_MODEL_ID:
                raise ValueError("provider response changed the frozen model")
            if self.response_model not in {None, response_model}:
                raise ValueError("provider response model changed during one loop")
            self.response_model = response_model

    def attempt(
        self,
        status: PoeAttemptStatus,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> PoeTransportAttempt:
        return PoeTransportAttempt(
            transport=self.transport,
            status=status,
            request_id=self.request.request_id,
            requested_model_id=POE_MODEL_ID,
            response_model=self.response_model,
            response_ids=tuple(self.response_ids),
            query_ids=tuple(self.query_ids),
            x_request_ids=tuple(self.x_request_ids),
            turns=self.turns,
            tool_executions=tuple(self.tool_executions),
            error_code=error_code,
            error_detail=error_detail,
        )


def _model_entry(catalog: PoeModelCatalog, endpoint: str | None) -> object:
    if not isinstance(catalog, PoeModelCatalog):
        raise TypeError("model_catalog must implement require(model_id)")
    entry = catalog.require(POE_MODEL_ID)
    entry_id = _member(entry, "id", _member(entry, "model_id"))
    if entry_id != POE_MODEL_ID:
        raise ValueError("verified model catalog returned another model")
    if endpoint is not None:
        endpoints = _member(entry, "supported_endpoints")
        if (
            not isinstance(endpoints, (tuple, list, frozenset, set))
            or endpoint not in endpoints
        ):
            raise ValueError(f"verified model lacks required endpoint {endpoint}")
        features = _member(entry, "supported_features")
        if (
            not isinstance(features, (tuple, list, frozenset, set))
            or "tools" not in features
        ):
            raise ValueError("verified model lacks chemistry tool capability")
    bot_name = _member(entry, "bot_name", _member(entry, "display_name"))
    if bot_name is not None and bot_name != POE_BOT_NAME:
        raise ValueError("verified model catalog bot mapping changed")
    return entry


def _proposal_input(request: ProposalRequest) -> str:
    return _canonical_json(
        {
            "instruction": "Return three to five proposals or a structured abstention.",
            "request": request.model_dump(mode="json"),
        }
    )


def _tool_schema(name: str) -> dict[str, object]:
    model = CHEMISTRY_TOOL_ARGUMENT_MODELS[name]
    return model.model_json_schema(mode="validation")


def _responses_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "name": name,
            "description": f"Read-only deterministic chemistry tool: {name}.",
            "parameters": _tool_schema(name),
        }
        for name in CHEMISTRY_TOOL_ARGUMENT_MODELS
    ]


def _chat_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Read-only deterministic chemistry tool: {name}.",
                "parameters": _tool_schema(name),
            },
        }
        for name in CHEMISTRY_TOOL_ARGUMENT_MODELS
    ]


def _response_text_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": "proposal_v1",
        "schema": proposal_response_json_schema(),
    }


def _validated_tool_call(name: object, arguments: object) -> ChemistryToolCall:
    if type(name) is not str:
        raise TypeError("tool name must be text")
    if isinstance(arguments, str):
        decoded = json.loads(arguments)
    elif isinstance(arguments, Mapping):
        decoded = dict(arguments)
    else:
        raise TypeError("tool arguments must be JSON text or a mapping")
    if not isinstance(decoded, Mapping):
        raise TypeError("tool arguments JSON must be an object")
    return parse_chemistry_tool_call({"tool": name, "arguments": decoded})


def _execute_tool(
    *,
    dispatcher: ChemistryToolDispatcher,
    call: ChemistryToolCall,
    call_id: str,
    turn: int,
    sequence: int,
) -> PoeToolExecution:
    result = dispatcher.dispatch(
        call.tool,
        call.arguments.model_dump(mode="python"),
    )
    return PoeToolExecution(
        turn=turn,
        sequence=sequence,
        tool_call_id=call_id,
        tool=call.tool,
        arguments_json=_canonical_json(call.arguments),
        result_json=_canonical_json(result),
    )


def _response_tool_calls(response: object) -> tuple[tuple[str, str, object], ...]:
    calls: list[tuple[str, str, object]] = []
    output = _member(response, "output", ())
    if output is None:
        output = ()
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise TypeError("Responses output must be a sequence")
    for item in output:
        if _member(item, "type") != "function_call":
            continue
        call_id = _member(item, "call_id", _member(item, "id"))
        name = _member(item, "name")
        arguments = _member(item, "arguments")
        if type(call_id) is not str or not call_id:
            raise TypeError("Responses tool call lacks call_id")
        if type(name) is not str or not name:
            raise TypeError("Responses tool call lacks name")
        calls.append((call_id, name, arguments))
    return tuple(calls)


def _responses_output_text(response: object) -> str | None:
    direct = _member(response, "output_text")
    if type(direct) is str and direct:
        return direct
    parts: list[str] = []
    output = _member(response, "output", ())
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if _member(item, "type") != "message":
                continue
            content = _member(item, "content", ())
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for part in content:
                if _member(part, "type") in {"output_text", "text"}:
                    text = _member(part, "text")
                    if type(text) is str:
                        parts.append(text)
    return "".join(parts) or None


def _chat_message(response: object) -> object:
    choices = _member(response, "choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or len(choices) != 1
    ):
        raise TypeError("Chat response must contain exactly one choice")
    message = _member(choices[0], "message")
    if message is None:
        raise TypeError("Chat choice lacks a message")
    return message


def _chat_tool_calls(message: object) -> tuple[tuple[str, str, object], ...]:
    raw = _member(message, "tool_calls", ())
    if raw is None:
        raw = ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("Chat tool_calls must be a sequence")
    calls: list[tuple[str, str, object]] = []
    for item in raw:
        call_id = _member(item, "id")
        function = _member(item, "function")
        name = _member(function, "name")
        arguments = _member(function, "arguments")
        if type(call_id) is not str or not call_id:
            raise TypeError("Chat tool call lacks id")
        if type(name) is not str or not name:
            raise TypeError("Chat tool call lacks function name")
        calls.append((call_id, name, arguments))
    return tuple(calls)


def _request_provenance(
    request: ProposalRequest,
    attempts: tuple[PoeTransportAttempt, ...],
) -> PoeClientProvenance:
    return PoeClientProvenance(
        provider="poe",
        request_id=request.request_id,
        origin_id=request.origin_id,
        operator_id=request.operator_id,
        propagation=request.propagation,
        candidate_source_mode=request.candidate_source_mode,
        target_root=request.target_root,
        constraints_json=_canonical_json(request.constraints),
        requested_model_id=POE_MODEL_ID,
        selected_transport=attempts[-1].transport,
        attempts=attempts,
    )


class _ProposalClientBase:
    transport_name: PoeTransport
    endpoint: str | None

    def __init__(
        self,
        *,
        model_catalog: PoeModelCatalog,
        config: PoeClientConfig | None = None,
    ) -> None:
        self.model_catalog = model_catalog
        self.config = PoeClientConfig() if config is None else config
        if type(self.config) is not PoeClientConfig:
            raise TypeError("config must be PoeClientConfig")

    def _start(self, request: ProposalRequest) -> _AttemptState:
        if type(request) is not ProposalRequest:
            raise TypeError("request must be ProposalRequest")
        state = _AttemptState.new(request, self.transport_name)
        try:
            _model_entry(self.model_catalog, self.endpoint)
        except (TypeError, ValueError, LookupError, RuntimeError) as error:
            self._fail(
                state,
                "POE_MODEL_CAPABILITY_UNAVAILABLE",
                "verified model catalog rejected the frozen transport binding",
                cause=error,
            )
        return state

    @staticmethod
    def _succeed(state: _AttemptState, response: ProposalResponse) -> PoeClientResult:
        attempt = state.attempt(PoeAttemptStatus.SUCCEEDED)
        return PoeClientResult(
            request=state.request,
            response=response,
            provenance=_request_provenance(state.request, (attempt,)),
        )

    @staticmethod
    def _fail(
        state: _AttemptState,
        code: str,
        detail: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        attempt = state.attempt(
            PoeAttemptStatus.FAILED,
            error_code=code,
            error_detail=detail,
        )
        raise PoeClientError(
            code=code,
            transport=state.transport,
            detail=detail,
            request_id=state.request.request_id,
            attempt=attempt,
            evidence={} if cause is None else {"cause_type": type(cause).__name__},
        ) from cause

    def _validate_calls(
        self,
        state: _AttemptState,
        calls: tuple[tuple[str, str, object], ...],
    ) -> tuple[tuple[str, str, ChemistryToolCall], ...]:
        if len(state.tool_executions) + len(calls) > self.config.max_tool_calls:
            self._fail(
                state,
                "POE_TOOL_CALL_LIMIT",
                "provider response exceeds six serial chemistry tool calls",
            )
        validated: list[tuple[str, str, ChemistryToolCall]] = []
        for call_id, name, arguments in calls:
            try:
                call = _validated_tool_call(name, arguments)
            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ValidationError,
            ) as error:
                self._fail(
                    state,
                    "POE_TOOL_ARGUMENT_INVALID",
                    "provider emitted invalid or unknown chemistry tool arguments",
                    cause=error,
                )
            validated.append((call_id, name, call))
        return tuple(validated)


class PoeResponsesClient(_ProposalClientBase):
    """Primary Responses transport with serial local function execution."""

    transport_name = PoeTransport.RESPONSES
    endpoint = POE_RESPONSES_ENDPOINT

    def __init__(
        self,
        *,
        transport: CreateTransport,
        dispatcher: ChemistryToolDispatcher,
        model_catalog: PoeModelCatalog,
        config: PoeClientConfig | None = None,
    ) -> None:
        super().__init__(model_catalog=model_catalog, config=config)
        if not isinstance(transport, CreateTransport):
            raise TypeError("transport must implement create")
        if not isinstance(dispatcher, ChemistryToolDispatcher):
            raise TypeError("dispatcher must implement dispatch")
        self.transport = transport
        self.dispatcher = dispatcher

    def propose(self, request: ProposalRequest) -> PoeClientResult:
        state = self._start(request)
        pending_input: object = _proposal_input(request)
        previous_response_id: str | None = None
        for turn in range(1, self.config.max_agent_turns + 1):
            state.turns = turn
            kwargs: dict[str, object] = {
                "model": self.config.model_id,
                "instructions": PROPOSAL_SYSTEM_PROMPT,
                "input": pending_input,
                "tools": _responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "reasoning": {"effort": self.config.reasoning_effort},
                "text": {"format": _response_text_format()},
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id
            try:
                raw = self.transport.create(**kwargs)
            except Exception as error:  # noqa: BLE001 - injected transport boundary
                self._fail(
                    state,
                    "POE_RESPONSES_TRANSPORT_ERROR",
                    "Responses transport failed before a response was available",
                    cause=error,
                )
            try:
                state.observe(raw, require_model=True)
            except ValueError as error:
                self._fail(
                    state,
                    "POE_RESPONSE_MODEL_MISMATCH",
                    "Responses omitted or changed the frozen model identity",
                    cause=error,
                )
            try:
                calls = _response_tool_calls(raw)
            except (TypeError, ValueError) as error:
                self._fail(
                    state,
                    "POE_RESPONSE_ENVELOPE_INVALID",
                    "Responses returned an invalid response envelope",
                    cause=error,
                )
            if calls:
                outputs: list[dict[str, str]] = []
                validated_calls = self._validate_calls(state, calls)
                for call_id, _name, call in validated_calls:
                    try:
                        execution = _execute_tool(
                            dispatcher=self.dispatcher,
                            call=call,
                            call_id=call_id,
                            turn=turn,
                            sequence=len(state.tool_executions) + 1,
                        )
                    except Exception as error:  # noqa: BLE001 - injected dispatcher
                        self._fail(
                            state,
                            "POE_TOOL_DISPATCH_FAILED",
                            "validated chemistry tool execution failed closed",
                            cause=error,
                        )
                    state.tool_executions.append(execution)
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": execution.result_json,
                        }
                    )
                if turn == self.config.max_agent_turns:
                    self._fail(
                        state,
                        "POE_AGENT_TURN_LIMIT",
                        "Responses requested tools on the final allowed agent turn",
                    )
                previous_response_id = _response_identifiers(raw)[0]
                if previous_response_id is None:
                    self._fail(
                        state,
                        "POE_RESPONSE_ID_MISSING",
                        "Responses tool continuation requires a response id",
                    )
                pending_input = outputs
                continue
            output_text = _responses_output_text(raw)
            if output_text is None:
                self._fail(
                    state,
                    "POE_RESPONSE_TEXT_MISSING",
                    "Responses returned neither tool calls nor proposal JSON",
                )
            try:
                response = parse_proposal_response(output_text, request=request)
            except (TypeError, ValueError, ValidationError) as error:
                self._fail(
                    state,
                    "POE_PROPOSAL_INVALID",
                    "Responses proposal failed local proposal_v1 validation",
                    cause=error,
                )
            return self._succeed(state, response)
        self._fail(state, "POE_AGENT_TURN_LIMIT", "Responses exhausted agent turns")


class PoeChatCompletionsClient(_ProposalClientBase):
    """Compatibility fallback with an explicit serial Chat tool loop."""

    transport_name = PoeTransport.CHAT_COMPLETIONS
    endpoint = POE_CHAT_ENDPOINT

    def __init__(
        self,
        *,
        transport: CreateTransport,
        dispatcher: ChemistryToolDispatcher,
        model_catalog: PoeModelCatalog,
        config: PoeClientConfig | None = None,
    ) -> None:
        super().__init__(model_catalog=model_catalog, config=config)
        if not isinstance(transport, CreateTransport):
            raise TypeError("transport must implement create")
        if not isinstance(dispatcher, ChemistryToolDispatcher):
            raise TypeError("dispatcher must implement dispatch")
        self.transport = transport
        self.dispatcher = dispatcher

    def propose(self, request: ProposalRequest) -> PoeClientResult:
        state = self._start(request)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
            {"role": "user", "content": _proposal_input(request)},
        ]
        for turn in range(1, self.config.max_agent_turns + 1):
            state.turns = turn
            try:
                raw = self.transport.create(
                    model=self.config.model_id,
                    messages=messages,
                    tools=_chat_tools(),
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
            except Exception as error:  # noqa: BLE001 - injected transport boundary
                self._fail(
                    state,
                    "POE_CHAT_TRANSPORT_ERROR",
                    "Chat transport failed before a response was available",
                    cause=error,
                )
            try:
                state.observe(raw, require_model=True)
            except ValueError as error:
                self._fail(
                    state,
                    "POE_RESPONSE_MODEL_MISMATCH",
                    "Chat omitted or changed the frozen model identity",
                    cause=error,
                )
            try:
                message = _chat_message(raw)
                calls = _chat_tool_calls(message)
            except (TypeError, ValueError) as error:
                self._fail(
                    state,
                    "POE_RESPONSE_ENVELOPE_INVALID",
                    "Chat returned an invalid response envelope",
                    cause=error,
                )
            if calls:
                assistant_calls: list[dict[str, object]] = []
                tool_messages: list[dict[str, object]] = []
                validated_calls = self._validate_calls(state, calls)
                for call_id, name, call in validated_calls:
                    try:
                        execution = _execute_tool(
                            dispatcher=self.dispatcher,
                            call=call,
                            call_id=call_id,
                            turn=turn,
                            sequence=len(state.tool_executions) + 1,
                        )
                    except Exception as error:  # noqa: BLE001 - injected dispatcher
                        self._fail(
                            state,
                            "POE_TOOL_DISPATCH_FAILED",
                            "validated chemistry tool execution failed closed",
                            cause=error,
                        )
                    state.tool_executions.append(execution)
                    assistant_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": execution.arguments_json,
                            },
                        }
                    )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": execution.result_json,
                        }
                    )
                if turn == self.config.max_agent_turns:
                    self._fail(
                        state,
                        "POE_AGENT_TURN_LIMIT",
                        "Chat requested tools on the final allowed agent turn",
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": assistant_calls,
                    }
                )
                messages.extend(tool_messages)
                continue
            content = _member(message, "content")
            if type(content) is not str or not content:
                self._fail(
                    state,
                    "POE_RESPONSE_TEXT_MISSING",
                    "Chat returned neither tool calls nor proposal JSON",
                )
            try:
                response = parse_proposal_response(content, request=request)
            except (TypeError, ValueError, ValidationError) as error:
                self._fail(
                    state,
                    "POE_PROPOSAL_INVALID",
                    "Chat proposal failed local proposal_v1 validation",
                    cause=error,
                )
            return self._succeed(state, response)
        self._fail(state, "POE_AGENT_TURN_LIMIT", "Chat exhausted agent turns")


class FastApiPoeTextClient(_ProposalClientBase):
    """Tool-free simple text fallback using a pre-bound fastapi_poe callable."""

    transport_name = PoeTransport.FASTAPI_POE
    endpoint = None

    def __init__(
        self,
        *,
        transport: FastApiPoeTransport,
        model_catalog: PoeModelCatalog,
        config: PoeClientConfig | None = None,
    ) -> None:
        super().__init__(model_catalog=model_catalog, config=config)
        if not callable(transport):
            raise TypeError("fastapi_poe transport must be callable")
        self.transport = transport

    def propose(self, request: ProposalRequest) -> PoeClientResult:
        state = self._start(request)
        state.turns = 1
        messages = (
            {
                "role": "user",
                "content": (
                    f"{PROPOSAL_SYSTEM_PROMPT}\n"
                    f"JSON schema: {_canonical_json(proposal_response_json_schema())}\n"
                    f"Input: {_proposal_input(request)}"
                ),
            },
        )
        try:
            raw = self.transport(messages=messages, bot_name=self.config.bot_name)
            chunks: Iterable[object]
            if type(raw) is str or isinstance(raw, Mapping):
                chunks = (raw,)
            elif isinstance(raw, Iterable):
                chunks = raw
            else:
                chunks = (raw,)
            text_parts: list[str] = []
            for chunk in chunks:
                if type(chunk) is str:
                    text_parts.append(chunk)
                else:
                    part = _member(chunk, "text", _member(chunk, "content"))
                    if type(part) is not str:
                        raise TypeError("fastapi_poe chunk lacks text")
                    text_parts.append(part)
                    state.observe(chunk)
            output_text = "".join(text_parts)
        except Exception as error:  # noqa: BLE001 - injected streaming transport
            self._fail(
                state,
                "POE_FASTAPI_TRANSPORT_ERROR",
                "fastapi_poe simple text transport failed",
                cause=error,
            )
        if not output_text:
            self._fail(
                state,
                "POE_RESPONSE_TEXT_MISSING",
                "fastapi_poe returned empty simple text",
            )
        try:
            response = parse_proposal_response(output_text, request=request)
        except (TypeError, ValueError, ValidationError) as error:
            self._fail(
                state,
                "POE_PROPOSAL_INVALID",
                "fastapi_poe proposal failed local proposal_v1 validation",
                cause=error,
            )
        return self._succeed(state, response)


class PoeTransportFallbackClient:
    """Try the frozen transport chain without mutating the proposal request."""

    def __init__(
        self,
        *,
        responses: PoeResponsesClient,
        chat_completions: PoeChatCompletionsClient,
        fastapi_poe: FastApiPoeTextClient,
    ) -> None:
        if type(responses) is not PoeResponsesClient:
            raise TypeError("responses must be PoeResponsesClient")
        if type(chat_completions) is not PoeChatCompletionsClient:
            raise TypeError("chat_completions must be PoeChatCompletionsClient")
        if type(fastapi_poe) is not FastApiPoeTextClient:
            raise TypeError("fastapi_poe must be FastApiPoeTextClient")
        configs = (responses.config, chat_completions.config, fastapi_poe.config)
        if len(set(configs)) != 1:
            raise ValueError("fallback clients must share the exact frozen config")
        self.clients = (responses, chat_completions, fastapi_poe)

    def propose(self, request: ProposalRequest) -> PoeClientResult:
        if type(request) is not ProposalRequest:
            raise TypeError("request must be ProposalRequest")
        failures: list[PoeTransportAttempt] = []
        for client in self.clients:
            try:
                result = client.propose(request)
            except PoeClientError as error:
                if error.attempt is None:
                    raise
                failures.append(error.attempt)
                continue
            success = result.provenance.attempts[-1]
            attempts = (*failures, success)
            provenance = _request_provenance(request, attempts)
            return replace(result, provenance=provenance)
        raise PoeFallbackExhaustedError(
            request_id=request.request_id,
            attempts=tuple(failures),
        )


__all__ = [
    "MAX_AGENT_TURNS",
    "MAX_CHEMISTRY_TOOL_CALLS",
    "POE_BOT_NAME",
    "POE_CHAT_ENDPOINT",
    "POE_MODEL_ID",
    "POE_RESPONSES_ENDPOINT",
    "PROPOSAL_SYSTEM_PROMPT",
    "ChemistryToolDispatcher",
    "CreateTransport",
    "FastApiPoeTextClient",
    "FastApiPoeTransport",
    "PoeAttemptStatus",
    "PoeChatCompletionsClient",
    "PoeClientConfig",
    "PoeClientError",
    "PoeClientProvenance",
    "PoeClientResult",
    "PoeFallbackExhaustedError",
    "PoeModelCatalog",
    "PoeResponsesClient",
    "PoeToolExecution",
    "PoeTransport",
    "PoeTransportAttempt",
    "PoeTransportFallbackClient",
]
