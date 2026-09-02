"""Bounded Poe request scheduling, retry, and fail-stop error policy.

The module is deliberately transport agnostic.  Network clients inject an
operation while tests inject clocks, sleepers, and jitter sources.  No error
body or request header is retained in the structured exceptions exposed here.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from random import random
from typing import TypeVar

REQUESTS_PER_MINUTE_CEILING = 450
INITIAL_MAX_CONCURRENCY = 8
RETRY_BASE_SECONDS = 0.250
REQUESTS_PER_MINUTE = REQUESTS_PER_MINUTE_CEILING
MAX_CONCURRENCY = INITIAL_MAX_CONCURRENCY
RETRY_BASE_MS = 250
DEFAULT_MAX_ATTEMPTS = 3
RATE_LIMIT_WINDOW_SECONDS = 60.0


class PoeErrorAction(StrEnum):
    """Whether a classified failure may be retried automatically."""

    FAIL_STOP = "fail_stop"
    RETRY = "retry"


class PoeErrorCode(StrEnum):
    """Stable, secret-free failure codes used by ledgers and callers."""

    BAD_REQUEST = "POE_BAD_REQUEST"
    CONTEXT_LENGTH_EXCEEDED = "POE_CONTEXT_LENGTH_EXCEEDED"
    UNSUPPORTED_MODEL = "POE_UNSUPPORTED_MODEL"
    UNSUPPORTED_ENDPOINT = "POE_UNSUPPORTED_ENDPOINT"
    AUTHENTICATION_FAILED = "POE_AUTHENTICATION_FAILED"
    INSUFFICIENT_POINTS = "POE_INSUFFICIENT_POINTS"
    PERMISSION_OR_MODERATION = "POE_PERMISSION_OR_MODERATION"
    MODEL_OR_ENDPOINT_NOT_FOUND = "POE_MODEL_OR_ENDPOINT_NOT_FOUND"
    REQUEST_TIMEOUT = "POE_REQUEST_TIMEOUT"
    RATE_LIMITED = "POE_RATE_LIMITED"
    PROVIDER_TRANSIENT = "POE_PROVIDER_TRANSIENT"
    HTTP_FATAL = "POE_HTTP_FATAL"
    NETWORK_TIMEOUT = "POE_NETWORK_TIMEOUT"
    NETWORK_TRANSIENT = "POE_NETWORK_TRANSIENT"
    CLIENT_FAILURE = "POE_CLIENT_FAILURE"


@dataclass(frozen=True, slots=True)
class PoeErrorClassification:
    """A safe classification that never embeds provider response text."""

    code: PoeErrorCode
    action: PoeErrorAction
    detail: str
    status_code: int | None = None
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.code) is not PoeErrorCode:
            raise TypeError("code must be a PoeErrorCode")
        if type(self.action) is not PoeErrorAction:
            raise TypeError("action must be a PoeErrorAction")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("detail must be non-empty text")
        if self.status_code is not None and (
            type(self.status_code) is not int or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be a valid HTTP status or None")
        if self.retry_after_seconds is not None and (
            type(self.retry_after_seconds) not in {int, float}
            or not math.isfinite(self.retry_after_seconds)
            or self.retry_after_seconds < 0
        ):
            raise ValueError("retry_after_seconds must be finite and non-negative")
        if (
            self.action is PoeErrorAction.FAIL_STOP
            and self.retry_after_seconds is not None
        ):
            raise ValueError("fail-stop errors cannot carry a retry delay")

    @property
    def retryable(self) -> bool:
        return self.action is PoeErrorAction.RETRY

    @property
    def fail_stop(self) -> bool:
        return self.action is PoeErrorAction.FAIL_STOP

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "action": self.action.value,
            "detail": self.detail,
            "status_code": self.status_code,
            "retry_after_seconds": self.retry_after_seconds,
        }


def _header_value(headers: Mapping[str, object] | None, name: str) -> object | None:
    if headers is None:
        return None
    target = name.casefold()
    for key, value in headers.items():
        if type(key) is str and key.casefold() == target:
            return value
    return None


def parse_retry_after(
    headers: Mapping[str, object] | None,
    *,
    wall_clock: Callable[[], datetime] | None = None,
) -> float | None:
    """Parse a ``Retry-After`` delta or HTTP date without retaining headers."""

    value = _header_value(headers, "retry-after")
    if value is None:
        return None
    if type(value) in {int, float}:
        delay = float(value)
        return delay if math.isfinite(delay) and delay >= 0 else None
    if type(value) is not str or not value.strip():
        return None
    text = value.strip()
    try:
        delay = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        now = datetime.now(UTC) if wall_clock is None else wall_clock()
        if not isinstance(now, datetime):
            raise TypeError("wall_clock must return datetime")
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        delay = max(0.0, (target - now).total_seconds())
    if not math.isfinite(delay) or delay < 0:
        return None
    return delay


def _marker_text(value: object) -> str:
    """Extract only classification markers; the returned text is never persisted."""

    if value is None:
        return ""
    if type(value) is str:
        return value.casefold()
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            if type(key) is str and key.casefold() in {
                "code",
                "error",
                "message",
                "type",
            }:
                parts.append(_marker_text(item))
        return " ".join(parts)
    if isinstance(value, (tuple, list)):
        return " ".join(_marker_text(item) for item in value)
    return ""


def classify_http_error(
    status_code: int,
    *,
    headers: Mapping[str, object] | None = None,
    response_body: object = None,
    wall_clock: Callable[[], datetime] | None = None,
) -> PoeErrorClassification:
    """Apply the frozen Poe HTTP policy with closed, auditable categories."""

    if type(status_code) is not int or not 100 <= status_code <= 599:
        raise ValueError("status_code must be a valid HTTP status")
    markers = _marker_text(response_body)
    if status_code == 400:
        if any(
            marker in markers
            for marker in (
                "context_length",
                "context length",
                "maximum context",
                "too many tokens",
            )
        ):
            code = PoeErrorCode.CONTEXT_LENGTH_EXCEEDED
            detail = "request exceeded the verified model context contract"
        elif any(
            marker in markers
            for marker in (
                "unsupported model",
                "unsupported_model",
                "model_not_supported",
                "model_not_found",
                "model does not exist",
                "invalid model",
            )
        ):
            code = PoeErrorCode.UNSUPPORTED_MODEL
            detail = "provider rejected the frozen Poe model"
        elif any(
            marker in markers
            for marker in (
                "unsupported endpoint",
                "unsupported_endpoint",
                "endpoint_not_supported",
                "unknown endpoint",
            )
        ):
            code = PoeErrorCode.UNSUPPORTED_ENDPOINT
            detail = "provider rejected the verified Poe endpoint"
        else:
            code = PoeErrorCode.BAD_REQUEST
            detail = "request or schema is invalid; configuration must be repaired"
        return PoeErrorClassification(
            code, PoeErrorAction.FAIL_STOP, detail, status_code
        )
    fail_stop = {
        401: (
            PoeErrorCode.AUTHENTICATION_FAILED,
            "Poe authentication failed; stop before issuing another request",
        ),
        402: (
            PoeErrorCode.INSUFFICIENT_POINTS,
            "Poe point balance is insufficient; preserve cache and checkpoint",
        ),
        403: (
            PoeErrorCode.PERMISSION_OR_MODERATION,
            "Poe permission or moderation decision requires review",
        ),
        404: (
            PoeErrorCode.MODEL_OR_ENDPOINT_NOT_FOUND,
            "verified Poe model or endpoint is unavailable; refresh catalog and fail closed",
        ),
    }
    if status_code in fail_stop:
        code, detail = fail_stop[status_code]
        return PoeErrorClassification(
            code, PoeErrorAction.FAIL_STOP, detail, status_code
        )
    retry_after = parse_retry_after(headers, wall_clock=wall_clock)
    if status_code == 408:
        return PoeErrorClassification(
            PoeErrorCode.REQUEST_TIMEOUT,
            PoeErrorAction.RETRY,
            "Poe request timed out",
            status_code,
            retry_after,
        )
    if status_code == 429:
        return PoeErrorClassification(
            PoeErrorCode.RATE_LIMITED,
            PoeErrorAction.RETRY,
            "Poe request rate limit was reached",
            status_code,
            retry_after,
        )
    if 500 <= status_code <= 599:
        return PoeErrorClassification(
            PoeErrorCode.PROVIDER_TRANSIENT,
            PoeErrorAction.RETRY,
            "Poe provider returned a transient server failure",
            status_code,
            retry_after,
        )
    return PoeErrorClassification(
        PoeErrorCode.HTTP_FATAL,
        PoeErrorAction.FAIL_STOP,
        "Poe returned a non-retryable HTTP failure",
        status_code,
    )


class PoeHTTPError(RuntimeError):
    """Minimal HTTP error whose public state excludes raw body and headers."""

    def __init__(
        self,
        status_code: int,
        *,
        headers: Mapping[str, object] | None = None,
        response_body: object = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.classification = classify_http_error(
            status_code,
            headers=headers,
            response_body=response_body,
            wall_clock=wall_clock,
        )
        self.status_code = status_code
        super().__init__(
            f"{self.classification.code.value}: {self.classification.detail}"
        )


def _exception_http_parts(
    error: BaseException,
) -> tuple[int | None, Mapping[str, object] | None, object]:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if type(status) is not int and response is not None:
        status = getattr(response, "status_code", None)
    headers = getattr(error, "headers", None)
    if not isinstance(headers, Mapping) and response is not None:
        headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        headers = None
    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except (AttributeError, TypeError, ValueError):
            body = None
    return status if type(status) is int else None, headers, body


def classify_exception(
    error: BaseException,
    *,
    wall_clock: Callable[[], datetime] | None = None,
) -> PoeErrorClassification:
    """Classify common SDK/HTTP/network exceptions without serializing them."""

    if isinstance(error, PoeHTTPError):
        return error.classification
    status, headers, body = _exception_http_parts(error)
    if status is not None:
        return classify_http_error(
            status,
            headers=headers,
            response_body=body,
            wall_clock=wall_clock,
        )
    class_name = type(error).__name__.casefold()
    if isinstance(error, TimeoutError) or "timeout" in class_name:
        return PoeErrorClassification(
            PoeErrorCode.NETWORK_TIMEOUT,
            PoeErrorAction.RETRY,
            "Poe network operation timed out",
        )
    if isinstance(error, (ConnectionError, OSError)) or any(
        marker in class_name
        for marker in ("connection", "connect", "network", "transport")
    ):
        return PoeErrorClassification(
            PoeErrorCode.NETWORK_TRANSIENT,
            PoeErrorAction.RETRY,
            "Poe network transport failed transiently",
        )
    return PoeErrorClassification(
        PoeErrorCode.CLIENT_FAILURE,
        PoeErrorAction.FAIL_STOP,
        "unclassified Poe client failure requires review",
    )


@dataclass(frozen=True, slots=True)
class PoeRetryPolicy:
    """Finite exponential retry policy; ``max_attempts`` includes the first call."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_seconds: float = RETRY_BASE_SECONDS
    max_delay_seconds: float = 8.0
    jitter_fraction: float = 0.20
    respect_retry_after: bool = True
    jitter_source: Callable[[], float] = field(
        default=random,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between one and ten")
        for value, name in (
            (self.base_delay_seconds, "base_delay_seconds"),
            (self.max_delay_seconds, "max_delay_seconds"),
            (self.jitter_fraction, "jitter_fraction"),
        ):
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.base_delay_seconds != RETRY_BASE_SECONDS:
            raise ValueError("Poe retry base delay is frozen at 250 ms")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be below the base delay")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be between zero and one")
        if type(self.respect_retry_after) is not bool:
            raise TypeError("respect_retry_after must be bool")
        if not callable(self.jitter_source):
            raise TypeError("jitter_source must be callable")

    def delay_for_retry(
        self,
        failed_attempt: int,
        *,
        retry_after_seconds: float | None = None,
    ) -> float:
        if type(failed_attempt) is not int or failed_attempt < 1:
            raise ValueError("failed_attempt must be a positive integer")
        if self.respect_retry_after and retry_after_seconds is not None:
            if (
                type(retry_after_seconds) not in {int, float}
                or not math.isfinite(retry_after_seconds)
                or retry_after_seconds < 0
            ):
                raise ValueError("retry_after_seconds must be finite and non-negative")
            return float(retry_after_seconds)
        source_value = self.jitter_source()
        if (
            type(source_value) not in {int, float}
            or not math.isfinite(source_value)
            or not 0 <= source_value <= 1
        ):
            raise ValueError("jitter_source must return a finite value in [0, 1]")
        exponential = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (failed_attempt - 1)),
        )
        return exponential * (1 + self.jitter_fraction * float(source_value))


@dataclass(frozen=True, slots=True)
class PoeRateLimitSnapshot:
    requests_per_minute: int
    max_concurrency: int
    requests_in_window: int
    active_requests: int


class PoeRateLimiter:
    """Thread-safe rolling-window limiter with a bounded request semaphore."""

    def __init__(
        self,
        *,
        requests_per_minute: int = REQUESTS_PER_MINUTE_CEILING,
        max_concurrency: int = INITIAL_MAX_CONCURRENCY,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        if (
            type(requests_per_minute) is not int
            or not 1 <= requests_per_minute <= REQUESTS_PER_MINUTE_CEILING
        ):
            raise ValueError("requests_per_minute must be between one and 450")
        if (
            type(max_concurrency) is not int
            or not 1 <= max_concurrency <= INITIAL_MAX_CONCURRENCY
        ):
            raise ValueError("max_concurrency must be between one and eight")
        if not callable(clock) or not callable(sleeper):
            raise TypeError("clock and sleeper must be callable")
        if (
            type(window_seconds) not in {int, float}
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError("window_seconds must be finite and positive")
        self.requests_per_minute = requests_per_minute
        self.max_concurrency = max_concurrency
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._sleeper = sleeper
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._active_requests = 0

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("clock must return a finite number")
        return float(value)

    def _discard_expired(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def acquire(self) -> None:
        """Reserve one request, sleeping only until the rolling window permits it."""

        while True:
            with self._lock:
                now = self._now()
                self._discard_expired(now)
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                wait_seconds = max(
                    0.0,
                    self._timestamps[0] + self.window_seconds - now,
                )
            self.sleep(wait_seconds)

    def sleep(self, seconds: float) -> None:
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("sleep duration must be finite and non-negative")
        self._sleeper(float(seconds))

    @contextmanager
    def request_slot(self) -> Iterator[None]:
        """Hold one of at most eight network slots for exactly one attempt."""

        self._semaphore.acquire()
        with self._lock:
            self._active_requests += 1
        try:
            self.acquire()
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
            self._semaphore.release()

    def snapshot(self) -> PoeRateLimitSnapshot:
        with self._lock:
            now = self._now()
            self._discard_expired(now)
            return PoeRateLimitSnapshot(
                requests_per_minute=self.requests_per_minute,
                max_concurrency=self.max_concurrency,
                requests_in_window=len(self._timestamps),
                active_requests=self._active_requests,
            )


@dataclass(frozen=True, slots=True)
class PoeRetryEvent:
    failed_attempt: int
    next_attempt: int
    delay_seconds: float
    classification: PoeErrorClassification

    def to_dict(self) -> dict[str, object]:
        return {
            "failed_attempt": self.failed_attempt,
            "next_attempt": self.next_attempt,
            "delay_seconds": self.delay_seconds,
            "classification": self.classification.to_dict(),
        }


class PoeCallError(RuntimeError):
    """Secret-free terminal error for fail-stop and exhausted retry paths."""

    def __init__(
        self,
        classification: PoeErrorClassification,
        *,
        attempt_count: int,
        retry_events: tuple[PoeRetryEvent, ...],
    ) -> None:
        if type(classification) is not PoeErrorClassification:
            raise TypeError("classification must be PoeErrorClassification")
        if type(attempt_count) is not int or attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        self.classification = classification
        self.code = classification.code.value
        self.attempt_count = attempt_count
        self.retry_events = retry_events
        super().__init__(f"{self.code}: {classification.detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.to_dict(),
            "attempt_count": self.attempt_count,
            "retry_events": [event.to_dict() for event in self.retry_events],
        }


class PoeFailStopError(PoeCallError):
    """A non-retryable policy decision."""


class PoeRetryExhaustedError(PoeCallError):
    """A retryable failure that reached the finite attempt bound."""


ResultT = TypeVar("ResultT")


def call_with_retry(
    operation: Callable[[], ResultT],
    *,
    limiter: PoeRateLimiter | None = None,
    policy: PoeRetryPolicy | None = None,
    on_retry: Callable[[PoeRetryEvent], None] | None = None,
    wall_clock: Callable[[], datetime] | None = None,
) -> ResultT:
    """Run one Poe operation under a finite, auditable retry loop."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    active_limiter = PoeRateLimiter() if limiter is None else limiter
    active_policy = PoeRetryPolicy() if policy is None else policy
    if type(active_limiter) is not PoeRateLimiter:
        raise TypeError("limiter must be PoeRateLimiter")
    if type(active_policy) is not PoeRetryPolicy:
        raise TypeError("policy must be PoeRetryPolicy")
    if on_retry is not None and not callable(on_retry):
        raise TypeError("on_retry must be callable or None")
    events: list[PoeRetryEvent] = []
    for attempt in range(1, active_policy.max_attempts + 1):
        try:
            with active_limiter.request_slot():
                return operation()
        except PoeCallError:
            raise
        except Exception as error:
            classification = classify_exception(error, wall_clock=wall_clock)
            if classification.fail_stop:
                raise PoeFailStopError(
                    classification,
                    attempt_count=attempt,
                    retry_events=tuple(events),
                ) from error
            if attempt == active_policy.max_attempts:
                raise PoeRetryExhaustedError(
                    classification,
                    attempt_count=attempt,
                    retry_events=tuple(events),
                ) from error
            delay = active_policy.delay_for_retry(
                attempt,
                retry_after_seconds=classification.retry_after_seconds,
            )
            event = PoeRetryEvent(
                failed_attempt=attempt,
                next_attempt=attempt + 1,
                delay_seconds=delay,
                classification=classification,
            )
            events.append(event)
            if on_retry is not None:
                on_retry(event)
            active_limiter.sleep(delay)
    raise AssertionError("finite retry loop terminated without a result")


# Descriptive aliases for callers that prefer shorter names.
RateLimiter = PoeRateLimiter
RetryPolicy = PoeRetryPolicy
classify_poe_error = classify_exception
retry_with_backoff = call_with_retry


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "INITIAL_MAX_CONCURRENCY",
    "MAX_CONCURRENCY",
    "RATE_LIMIT_WINDOW_SECONDS",
    "REQUESTS_PER_MINUTE",
    "REQUESTS_PER_MINUTE_CEILING",
    "RETRY_BASE_MS",
    "RETRY_BASE_SECONDS",
    "PoeCallError",
    "PoeErrorAction",
    "PoeErrorClassification",
    "PoeErrorCode",
    "PoeFailStopError",
    "PoeHTTPError",
    "PoeRateLimitSnapshot",
    "PoeRateLimiter",
    "PoeRetryEvent",
    "PoeRetryExhaustedError",
    "PoeRetryPolicy",
    "RateLimiter",
    "RetryPolicy",
    "call_with_retry",
    "classify_exception",
    "classify_http_error",
    "classify_poe_error",
    "parse_retry_after",
    "retry_with_backoff",
]
