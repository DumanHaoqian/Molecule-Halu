"""Private, secret-free Poe token and point usage accounting.

The ledger accepts only closed audit fields.  It deliberately has no API-key
or request-header parameter.  Balance/history access is injected so tests and
release replay never need a network connection.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from .rate_limiter import PoeRetryEvent

POE_MODEL_ID = "gpt-5.4-mini"
POE_BALANCE_ENDPOINT = "https://api.poe.com/usage/current_balance"
POE_POINTS_HISTORY_ENDPOINT = "https://api.poe.com/usage/points_history"
PRIVATE_FILE_MODE = 0o600
REDACTED = "[REDACTED]"
_TRANSPORTS = frozenset({"responses", "chat_completions", "fastapi_poe"})
_SENSITIVE_KEY_MARKERS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "apikey",
        "poeapikey",
        "xapikey",
        "secret",
        "password",
        "cookie",
        "setcookie",
    }
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)(\bauthorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\]}]+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;\]}]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:poe_api_key|api[_-]?key|x-api-key|secret|password)\s*[:=]\s*)"
    r"[^\s,;\]}]+"
)


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_sensitive_key(value: str) -> bool:
    return _normalized_key(value) in _SENSITIVE_KEY_MARKERS


def redact_text(value: str) -> str:
    """Redact credential-like assignments while preserving useful error codes."""

    if type(value) is not str:
        raise TypeError("value must be text")
    value = _AUTHORIZATION_RE.sub(rf"\1{REDACTED}", value)
    value = _BEARER_RE.sub(f"Bearer {REDACTED}", value)
    return _SECRET_ASSIGNMENT_RE.sub(rf"\1{REDACTED}", value)


def redact_sensitive(value: object) -> object:
    """Recursively redact known credential keys from a closed JSON value."""

    if value is None or type(value) in {int, float, bool}:
        return value
    if type(value) is str:
        return redact_text(value)
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("audit mapping keys must be strings")
            redacted[key] = (
                REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            )
        return redacted
    if isinstance(value, (tuple, list)):
        return [redact_sensitive(item) for item in value]
    raise TypeError(f"audit value is not closed JSON: {type(value).__name__}")


def _single_line(value: str, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{name} must be non-empty trimmed single-line text")
    return value


def _optional_single_line(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be text or None")
    return _single_line(value, name)


def _unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    result: list[str] = []
    for value in values:
        identifier = _single_line(value, name)
        if identifier not in result:
            result.append(identifier)
    return tuple(result)


def _non_negative_number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _timestamp(value: object) -> str:
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if type(value) is str:
        return _single_line(value, "timestamp")
    raise TypeError("clock must return datetime or timestamp text")


def _member(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_member(value: object, names: Sequence[str]) -> object:
    for name in names:
        candidate = _member(value, name)
        if candidate is not None:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class PoeTokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        for value, name in (
            (self.prompt_tokens, "prompt_tokens"),
            (self.output_tokens, "output_tokens"),
            (self.total_tokens, "total_tokens"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_tokens < self.prompt_tokens + self.output_tokens:
            raise ValueError("total_tokens cannot be below prompt plus output tokens")

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    @classmethod
    def from_provider(cls, value: object) -> PoeTokenUsage:
        if value is None:
            return cls()
        prompt = _first_member(value, ("prompt_tokens", "input_tokens"))
        output = _first_member(value, ("output_tokens", "completion_tokens"))
        total = _member(value, "total_tokens")
        prompt = 0 if prompt is None else prompt
        output = 0 if output is None else output
        if type(prompt) is not int or type(output) is not int:
            raise TypeError("provider token usage values must be integers")
        total = prompt + output if total is None else total
        if type(total) is not int:
            raise TypeError("provider total_tokens must be an integer")
        return cls(prompt_tokens=prompt, output_tokens=output, total_tokens=total)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class PoeUsageErrorEvent:
    attempt: int
    code: str
    retryable: bool
    detail: str
    status_code: int | None = None

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("error attempt must be a positive integer")
        _single_line(self.code, "error code")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be bool")
        safe_detail = redact_text(_single_line(self.detail, "error detail"))
        object.__setattr__(self, "detail", safe_detail)
        if self.status_code is not None and (
            type(self.status_code) is not int or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be a valid HTTP status or None")

    @classmethod
    def from_retry_event(cls, event: PoeRetryEvent) -> PoeUsageErrorEvent:
        if type(event) is not PoeRetryEvent:
            raise TypeError("event must be PoeRetryEvent")
        return cls(
            attempt=event.failed_attempt,
            code=event.classification.code.value,
            retryable=True,
            detail=event.classification.detail,
            status_code=event.classification.status_code,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "code": self.code,
            "retryable": self.retryable,
            "detail": self.detail,
            "status_code": self.status_code,
        }


@dataclass(frozen=True, slots=True)
class PoeUsageRecord:
    recorded_at: str
    request_id: str
    transport: str
    model_id: str
    token_usage: PoeTokenUsage
    attempt_count: int
    retry_count: int
    response_ids: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()
    x_request_ids: tuple[str, ...] = ()
    cost_points: float | None = None
    errors: tuple[PoeUsageErrorEvent, ...] = ()

    def __post_init__(self) -> None:
        _single_line(self.recorded_at, "recorded_at")
        _single_line(self.request_id, "request_id")
        if self.transport not in _TRANSPORTS:
            raise ValueError("transport is outside the frozen Poe transport set")
        if self.model_id != POE_MODEL_ID:
            raise ValueError("usage record model must remain gpt-5.4-mini")
        if type(self.token_usage) is not PoeTokenUsage:
            raise TypeError("token_usage must be PoeTokenUsage")
        if type(self.attempt_count) is not int or self.attempt_count < 1:
            raise ValueError("attempt_count must be a positive integer")
        if (
            type(self.retry_count) is not int
            or self.retry_count < 0
            or self.retry_count >= self.attempt_count
        ):
            raise ValueError("retry_count must be below attempt_count")
        object.__setattr__(
            self,
            "response_ids",
            _unique_ids(self.response_ids, "response_ids"),
        )
        object.__setattr__(self, "query_ids", _unique_ids(self.query_ids, "query_ids"))
        object.__setattr__(
            self,
            "x_request_ids",
            _unique_ids(self.x_request_ids, "x_request_ids"),
        )
        if self.cost_points is not None:
            object.__setattr__(
                self,
                "cost_points",
                _non_negative_number(self.cost_points, "cost_points"),
            )
        if not isinstance(self.errors, tuple) or any(
            type(item) is not PoeUsageErrorEvent for item in self.errors
        ):
            raise TypeError("errors must be a tuple of PoeUsageErrorEvent")
        if any(item.attempt > self.attempt_count for item in self.errors):
            raise ValueError("error attempt exceeds the recorded attempt count")

    @property
    def response_id(self) -> str | None:
        return self.response_ids[-1] if self.response_ids else None

    @property
    def query_id(self) -> str | None:
        return self.query_ids[-1] if self.query_ids else None

    @property
    def x_request_id(self) -> str | None:
        return self.x_request_ids[-1] if self.x_request_ids else None

    def with_cost_points(self, value: float) -> PoeUsageRecord:
        return replace(self, cost_points=_non_negative_number(value, "cost_points"))

    def to_dict(self) -> dict[str, object]:
        return {
            "recorded_at": self.recorded_at,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "query_id": self.query_id,
            "x_request_id": self.x_request_id,
            "response_ids": list(self.response_ids),
            "query_ids": list(self.query_ids),
            "x_request_ids": list(self.x_request_ids),
            "transport": self.transport,
            "model_id": self.model_id,
            "token_usage": self.token_usage.to_dict(),
            "cost_points": self.cost_points,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "errors": [item.to_dict() for item in self.errors],
        }


@dataclass(frozen=True, slots=True)
class PoePointsHistoryEntry:
    cost_points: float
    event_id: str | None = None
    request_id: str | None = None
    response_id: str | None = None
    query_id: str | None = None
    x_request_id: str | None = None
    model_id: str | None = None
    occurred_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cost_points",
            _non_negative_number(self.cost_points, "history cost_points"),
        )
        for name in (
            "event_id",
            "request_id",
            "response_id",
            "query_id",
            "x_request_id",
            "model_id",
            "occurred_at",
        ):
            object.__setattr__(
                self,
                name,
                _optional_single_line(getattr(self, name), name),
            )
        if not any(
            (
                self.event_id,
                self.request_id,
                self.response_id,
                self.query_id,
                self.x_request_id,
            )
        ):
            raise ValueError("points history entry lacks an auditable identifier")

    @classmethod
    def from_provider(cls, value: object) -> PoePointsHistoryEntry:
        if not isinstance(value, Mapping):
            raise TypeError("points history item must be a mapping")
        metadata = value.get("metadata")
        sources = (value, metadata) if isinstance(metadata, Mapping) else (value,)

        def lookup(names: Sequence[str]) -> object:
            for source in sources:
                candidate = _first_member(source, names)
                if candidate is not None:
                    return candidate
            return None

        points = lookup(("cost_points", "points", "cost", "amount"))
        if points is None:
            balance_change = lookup(("balance_change", "points_change"))
            if type(balance_change) in {int, float}:
                points = abs(float(balance_change))
        if points is None:
            raise ValueError("points history item lacks a point cost")
        return cls(
            cost_points=_non_negative_number(points, "history point cost"),
            event_id=lookup(("event_id", "transaction_id", "id")),
            request_id=lookup(("request_id", "requestId")),
            response_id=lookup(("response_id", "responseId")),
            query_id=lookup(("query_id", "queryId")),
            x_request_id=lookup(("x_request_id", "x-request-id", "xRequestId")),
            model_id=lookup(("model_id", "model")),
            occurred_at=lookup(("occurred_at", "created_at", "timestamp")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "query_id": self.query_id,
            "x_request_id": self.x_request_id,
            "model_id": self.model_id,
            "occurred_at": self.occurred_at,
            "cost_points": self.cost_points,
        }


@dataclass(frozen=True, slots=True)
class PoeUsageReconciliation:
    reconciled_at: str
    pre_build_point_balance: float | None
    post_build_point_balance: float | None
    balance_spent_points: float | None
    history_cost_points: float
    recorded_cost_points: float
    unmatched_request_ids: tuple[str, ...]
    unmatched_history_event_ids: tuple[str, ...]
    ambiguous_history_event_ids: tuple[str, ...]
    cost_mismatch_request_ids: tuple[str, ...]
    balanced: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "reconciled_at": self.reconciled_at,
            "pre_build_point_balance": self.pre_build_point_balance,
            "post_build_point_balance": self.post_build_point_balance,
            "balance_spent_points": self.balance_spent_points,
            "history_cost_points": self.history_cost_points,
            "recorded_cost_points": self.recorded_cost_points,
            "unmatched_request_ids": list(self.unmatched_request_ids),
            "unmatched_history_event_ids": list(self.unmatched_history_event_ids),
            "ambiguous_history_event_ids": list(self.ambiguous_history_event_ids),
            "cost_mismatch_request_ids": list(self.cost_mismatch_request_ids),
            "balanced": self.balanced,
        }


@runtime_checkable
class PoeUsageTransport(Protocol):
    """Injected usage API surface; authentication stays inside the transport."""

    def get_current_balance(self) -> object: ...

    def get_points_history(self) -> object: ...


def _response_payload(value: object) -> object:
    if isinstance(value, Mapping) or type(value) in {int, float}:
        return value
    json_method = getattr(value, "json", None)
    if callable(json_method):
        return json_method()
    return value


def _extract_balance(value: object) -> float:
    payload = _response_payload(value)
    if type(payload) in {int, float}:
        return _non_negative_number(payload, "point balance")
    if not isinstance(payload, Mapping):
        raise TypeError("balance response must be numeric or a mapping")
    nested = payload.get("data")
    sources = (payload, nested) if isinstance(nested, Mapping) else (payload,)
    for source in sources:
        candidate = _first_member(
            source,
            ("current_balance", "balance", "points_balance", "points"),
        )
        if candidate is not None:
            return _non_negative_number(candidate, "point balance")
    raise ValueError("balance response lacks a recognized point balance")


def _extract_history(value: object) -> tuple[PoePointsHistoryEntry, ...]:
    payload = _response_payload(value)
    if isinstance(payload, Mapping):
        sequence = _first_member(
            payload, ("data", "items", "history", "points_history")
        )
    else:
        sequence = payload
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
        raise TypeError("points history response must contain a sequence")
    return tuple(PoePointsHistoryEntry.from_provider(item) for item in sequence)


class PoeUsageLedger:
    """Thread-safe per-build ledger with balance/history reconciliation."""

    def __init__(
        self,
        *,
        transport: PoeUsageTransport | object | None = None,
        clock: Callable[[], datetime | str] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._created_at = self._now()
        self._updated_at = self._created_at
        self._records: list[PoeUsageRecord] = []
        self._pre_build_point_balance: float | None = None
        self._post_build_point_balance: float | None = None
        self._history: tuple[PoePointsHistoryEntry, ...] = ()
        self._reconciliation: PoeUsageReconciliation | None = None

    def _now(self) -> str:
        return _timestamp(self._clock())

    def _touch(self) -> None:
        self._updated_at = self._now()

    @property
    def records(self) -> tuple[PoeUsageRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def pre_build_point_balance(self) -> float | None:
        return self._pre_build_point_balance

    @property
    def post_build_point_balance(self) -> float | None:
        return self._post_build_point_balance

    @property
    def reconciliation(self) -> PoeUsageReconciliation | None:
        return self._reconciliation

    def _usage_call(self, method_name: str, endpoint: str) -> object:
        if self._transport is None:
            raise RuntimeError("Poe usage transport was not configured")
        method = getattr(self._transport, method_name, None)
        if callable(method):
            return method()
        get_method = getattr(self._transport, "get", None)
        if callable(get_method):
            return get_method(endpoint)
        raise TypeError(
            f"usage transport must implement {method_name}() or get(endpoint)"
        )

    def capture_pre_build_balance(self) -> float:
        balance = _extract_balance(
            self._usage_call("get_current_balance", POE_BALANCE_ENDPOINT)
        )
        with self._lock:
            self._pre_build_point_balance = balance
            self._reconciliation = None
            self._touch()
        return balance

    def capture_post_build_balance(self) -> float:
        balance = _extract_balance(
            self._usage_call("get_current_balance", POE_BALANCE_ENDPOINT)
        )
        with self._lock:
            self._post_build_point_balance = balance
            self._reconciliation = None
            self._touch()
        return balance

    def fetch_points_history(self) -> tuple[PoePointsHistoryEntry, ...]:
        history = _extract_history(
            self._usage_call("get_points_history", POE_POINTS_HISTORY_ENDPOINT)
        )
        with self._lock:
            self._history = history
            self._reconciliation = None
            self._touch()
        return history

    # Build lifecycle aliases used by orchestration code.
    begin_build = capture_pre_build_balance

    def finish_build(self) -> PoeUsageReconciliation:
        if self._pre_build_point_balance is None:
            raise RuntimeError("capture pre-build point balance before finishing build")
        self.capture_post_build_balance()
        history = self.fetch_points_history()
        return self.reconcile(history)

    def append(self, record: PoeUsageRecord) -> PoeUsageRecord:
        if type(record) is not PoeUsageRecord:
            raise TypeError("record must be PoeUsageRecord")
        with self._lock:
            if any(item.request_id == record.request_id for item in self._records):
                raise ValueError(f"duplicate Poe usage request_id: {record.request_id}")
            self._records.append(record)
            self._reconciliation = None
            self._touch()
        return record

    def record_request(
        self,
        *,
        request_id: str,
        transport: str,
        model_id: str = POE_MODEL_ID,
        token_usage: PoeTokenUsage | Mapping[str, object] | object | None = None,
        attempt_count: int = 1,
        retry_count: int = 0,
        response_id: str | None = None,
        query_id: str | None = None,
        x_request_id: str | None = None,
        response_ids: Sequence[str] = (),
        query_ids: Sequence[str] = (),
        x_request_ids: Sequence[str] = (),
        cost_points: float | None = None,
        errors: Sequence[PoeUsageErrorEvent] = (),
    ) -> PoeUsageRecord:
        usage = (
            token_usage
            if type(token_usage) is PoeTokenUsage
            else PoeTokenUsage.from_provider(token_usage)
        )
        all_response_ids = tuple(response_ids) + (
            () if response_id is None else (response_id,)
        )
        all_query_ids = tuple(query_ids) + (() if query_id is None else (query_id,))
        all_x_request_ids = tuple(x_request_ids) + (
            () if x_request_id is None else (x_request_id,)
        )
        record = PoeUsageRecord(
            recorded_at=self._now(),
            request_id=request_id,
            transport=transport,
            model_id=model_id,
            token_usage=usage,
            attempt_count=attempt_count,
            retry_count=retry_count,
            response_ids=all_response_ids,
            query_ids=all_query_ids,
            x_request_ids=all_x_request_ids,
            cost_points=cost_points,
            errors=tuple(errors),
        )
        return self.append(record)

    def record_client_result(
        self,
        result: object,
        *,
        token_usage: PoeTokenUsage | Mapping[str, object] | object | None = None,
        retry_events: Sequence[PoeRetryEvent] = (),
        cost_points: float | None = None,
    ) -> PoeUsageRecord:
        """Consume the public T034 result contract without importing its clients."""

        provenance = _member(result, "provenance")
        if provenance is None:
            raise TypeError("client result lacks provenance")
        attempts = _member(provenance, "attempts", ())
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise TypeError("client provenance attempts must be a sequence")
        if not attempts:
            raise ValueError("client provenance must include at least one attempt")
        selected_transport = _member(provenance, "selected_transport")
        transport = _member(selected_transport, "value", selected_transport)
        request_id = _member(provenance, "request_id")
        model_id = _member(provenance, "requested_model_id")
        response_ids: list[str] = []
        query_ids: list[str] = []
        x_request_ids: list[str] = []
        error_events: list[PoeUsageErrorEvent] = []
        for attempt_number, attempt in enumerate(attempts, start=1):
            response_ids.extend(_member(attempt, "response_ids", ()))
            query_ids.extend(_member(attempt, "query_ids", ()))
            x_request_ids.extend(_member(attempt, "x_request_ids", ()))
            error_code = _member(attempt, "error_code")
            if type(error_code) is str:
                error_events.append(
                    PoeUsageErrorEvent(
                        attempt=attempt_number,
                        code=error_code,
                        retryable=attempt_number < len(attempts),
                        detail=_member(
                            attempt, "error_detail", "transport attempt failed"
                        ),
                    )
                )
        for event in retry_events:
            error_events.append(PoeUsageErrorEvent.from_retry_event(event))
        if token_usage is None:
            token_usage = _member(_member(result, "response"), "usage")
        retry_count = len(retry_events)
        return self.record_request(
            request_id=request_id,
            transport=transport,
            model_id=model_id,
            token_usage=token_usage,
            attempt_count=len(attempts) + retry_count,
            retry_count=retry_count,
            response_ids=response_ids,
            query_ids=query_ids,
            x_request_ids=x_request_ids,
            cost_points=cost_points,
            errors=error_events,
        )

    @staticmethod
    def _history_identifier(entry: PoePointsHistoryEntry, index: int) -> str:
        return (
            entry.event_id
            or entry.query_id
            or entry.response_id
            or entry.x_request_id
            or entry.request_id
            or f"history-index-{index}"
        )

    @staticmethod
    def _match_score(record: PoeUsageRecord, entry: PoePointsHistoryEntry) -> int:
        scores = (
            5
            if entry.query_id is not None and entry.query_id in record.query_ids
            else 0,
            4
            if entry.response_id is not None
            and entry.response_id in record.response_ids
            else 0,
            3
            if entry.x_request_id is not None
            and entry.x_request_id in record.x_request_ids
            else 0,
            2 if entry.request_id == record.request_id else 0,
        )
        return max(scores)

    def reconcile(
        self,
        history: Sequence[PoePointsHistoryEntry | Mapping[str, object]] | None = None,
    ) -> PoeUsageReconciliation:
        if history is None:
            parsed = self._history
        else:
            parsed = tuple(
                item
                if type(item) is PoePointsHistoryEntry
                else PoePointsHistoryEntry.from_provider(item)
                for item in history
            )
        with self._lock:
            records = list(self._records)
            allocations: dict[int, float] = {}
            matched_records: set[int] = set()
            unmatched_history: list[str] = []
            ambiguous_history: list[str] = []
            for history_index, entry in enumerate(parsed):
                event_id = self._history_identifier(entry, history_index)
                scored = [
                    (self._match_score(record, entry), index)
                    for index, record in enumerate(records)
                ]
                highest = max((score for score, _ in scored), default=0)
                candidates = [
                    index for score, index in scored if score == highest and score > 0
                ]
                if not candidates:
                    unmatched_history.append(event_id)
                    continue
                if len(candidates) != 1:
                    ambiguous_history.append(event_id)
                    continue
                record_index = candidates[0]
                allocations[record_index] = (
                    allocations.get(record_index, 0.0) + entry.cost_points
                )
                matched_records.add(record_index)
            mismatches: list[str] = []
            for index, points in allocations.items():
                existing = records[index].cost_points
                if existing is not None and not math.isclose(
                    existing,
                    points,
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    mismatches.append(records[index].request_id)
                records[index] = records[index].with_cost_points(points)
            unmatched_requests = tuple(
                record.request_id
                for index, record in enumerate(records)
                if index not in matched_records
            )
            history_cost = sum(item.cost_points for item in parsed)
            recorded_cost = sum(item.cost_points or 0.0 for item in records)
            balance_spent = None
            balance_matches = True
            if (
                self._pre_build_point_balance is not None
                and self._post_build_point_balance is not None
            ):
                balance_spent = (
                    self._pre_build_point_balance - self._post_build_point_balance
                )
                balance_matches = math.isclose(
                    balance_spent,
                    history_cost,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            balanced = (
                not any(
                    (
                        unmatched_requests,
                        unmatched_history,
                        ambiguous_history,
                        mismatches,
                    )
                )
                and math.isclose(
                    history_cost,
                    recorded_cost,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                and balance_matches
            )
            reconciliation = PoeUsageReconciliation(
                reconciled_at=self._now(),
                pre_build_point_balance=self._pre_build_point_balance,
                post_build_point_balance=self._post_build_point_balance,
                balance_spent_points=balance_spent,
                history_cost_points=history_cost,
                recorded_cost_points=recorded_cost,
                unmatched_request_ids=unmatched_requests,
                unmatched_history_event_ids=tuple(unmatched_history),
                ambiguous_history_event_ids=tuple(ambiguous_history),
                cost_mismatch_request_ids=tuple(mismatches),
                balanced=balanced,
            )
            self._records = records
            self._history = parsed
            self._reconciliation = reconciliation
            self._touch()
            return reconciliation

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            payload: dict[str, object] = {
                "format_version": "poe_usage_ledger_v1",
                "provider": "poe",
                "model_id": POE_MODEL_ID,
                "created_at": self._created_at,
                "updated_at": self._updated_at,
                "pre_build_point_balance": self._pre_build_point_balance,
                "post_build_point_balance": self._post_build_point_balance,
                "records": [record.to_dict() for record in self._records],
                "points_history": [item.to_dict() for item in self._history],
                "reconciliation": (
                    None
                    if self._reconciliation is None
                    else self._reconciliation.to_dict()
                ),
            }
        redacted = redact_sensitive(payload)
        if not isinstance(redacted, dict):
            raise TypeError("ledger payload must remain a mapping")
        return redacted

    def export_private_json(self, path: str | os.PathLike[str]) -> Path:
        """Write a mode-0600 private artifact containing no raw credentials."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, PRIVATE_FILE_MODE)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.chmod(destination, PRIVATE_FILE_MODE)
        if stat.S_IMODE(destination.stat().st_mode) != PRIVATE_FILE_MODE:
            try:
                destination.unlink()
            except OSError:
                pass
            raise PermissionError(
                "destination filesystem cannot enforce owner-only ledger mode"
            )
        return destination

    # Compact aliases for orchestration code.
    export = export_private_json
    record_usage = record_request
    reconcile_points_history = reconcile


PoeUsageEntry = PoeUsageRecord


__all__ = [
    "POE_BALANCE_ENDPOINT",
    "POE_MODEL_ID",
    "POE_POINTS_HISTORY_ENDPOINT",
    "PRIVATE_FILE_MODE",
    "REDACTED",
    "PoePointsHistoryEntry",
    "PoeTokenUsage",
    "PoeUsageEntry",
    "PoeUsageErrorEvent",
    "PoeUsageLedger",
    "PoeUsageReconciliation",
    "PoeUsageRecord",
    "PoeUsageTransport",
    "redact_sensitive",
    "redact_text",
]
