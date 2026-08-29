"""Deterministic tests for the T035 Poe rate and error policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from molhallulens.providers.poe.rate_limiter import (
    INITIAL_MAX_CONCURRENCY,
    REQUESTS_PER_MINUTE_CEILING,
    RETRY_BASE_SECONDS,
    PoeErrorCode,
    PoeFailStopError,
    PoeHTTPError,
    PoeRateLimiter,
    PoeRetryExhaustedError,
    PoeRetryPolicy,
    call_with_retry,
    classify_http_error,
    parse_retry_after,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def test_frozen_defaults_match_llm_configuration() -> None:
    limiter = PoeRateLimiter()
    policy = PoeRetryPolicy(jitter_source=lambda: 0.0)

    assert limiter.requests_per_minute == REQUESTS_PER_MINUTE_CEILING == 450
    assert limiter.max_concurrency == INITIAL_MAX_CONCURRENCY == 8
    assert policy.base_delay_seconds == RETRY_BASE_SECONDS == 0.250
    assert policy.max_attempts == 3
    with pytest.raises(ValueError, match="450"):
        PoeRateLimiter(requests_per_minute=451)
    with pytest.raises(ValueError, match="eight"):
        PoeRateLimiter(max_concurrency=9)


def test_rolling_window_reserves_at_most_configured_rpm() -> None:
    clock = FakeClock()
    limiter = PoeRateLimiter(
        requests_per_minute=2,
        max_concurrency=1,
        clock=clock,
        sleeper=clock.sleep,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert clock.sleeps == [60.0]
    assert limiter.snapshot().requests_in_window == 1


def test_retry_after_supports_delta_and_http_date() -> None:
    now = datetime(2026, 8, 30, 2, 0, tzinfo=UTC)
    later = now + timedelta(seconds=7)

    assert parse_retry_after({"Retry-After": "1.5"}) == 1.5
    assert (
        parse_retry_after(
            {"retry-after": later.strftime("%a, %d %b %Y %H:%M:%S GMT")},
            wall_clock=lambda: now,
        )
        == 7.0
    )
    assert (
        parse_retry_after(
            {
                "Retry-After": (now - timedelta(seconds=1)).strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                )
            },
            wall_clock=lambda: now,
        )
        == 0.0
    )
    assert parse_retry_after({"Retry-After": "not-a-delay"}) is None


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, PoeErrorCode.AUTHENTICATION_FAILED),
        (402, PoeErrorCode.INSUFFICIENT_POINTS),
        (403, PoeErrorCode.PERMISSION_OR_MODERATION),
        (404, PoeErrorCode.MODEL_OR_ENDPOINT_NOT_FOUND),
    ],
)
def test_auth_points_permission_and_catalog_errors_fail_stop(
    status: int,
    expected_code: PoeErrorCode,
) -> None:
    classified = classify_http_error(status)

    assert classified.code is expected_code
    assert classified.fail_stop
    assert not classified.retryable


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (
            {"error": {"code": "context_length_exceeded"}},
            PoeErrorCode.CONTEXT_LENGTH_EXCEEDED,
        ),
        ({"error": {"message": "unsupported model"}}, PoeErrorCode.UNSUPPORTED_MODEL),
        ({"message": "unsupported endpoint"}, PoeErrorCode.UNSUPPORTED_ENDPOINT),
    ],
)
def test_context_model_and_endpoint_contract_errors_fail_closed(
    body: object,
    expected_code: PoeErrorCode,
) -> None:
    classified = classify_http_error(400, response_body=body)

    assert classified.code is expected_code
    assert classified.fail_stop


def test_retry_after_precedes_backoff_then_exponential_retry_succeeds() -> None:
    clock = FakeClock()
    limiter = PoeRateLimiter(clock=clock, sleeper=clock.sleep)
    policy = PoeRetryPolicy(jitter_source=lambda: 0.0)
    outcomes: list[object] = [
        PoeHTTPError(429, headers={"Retry-After": "2"}),
        PoeHTTPError(503),
        "ok",
    ]
    events = []

    def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)

    result = call_with_retry(
        operation,
        limiter=limiter,
        policy=policy,
        on_retry=events.append,
    )

    assert result == "ok"
    assert clock.sleeps == [2.0, 0.5]
    assert [item.failed_attempt for item in events] == [1, 2]
    assert events[0].classification.code is PoeErrorCode.RATE_LIMITED
    assert events[1].classification.code is PoeErrorCode.PROVIDER_TRANSIENT


def test_fail_stop_does_not_retry_or_leak_provider_body() -> None:
    clock = FakeClock()
    calls = 0
    secret = "poe-super-secret"

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise PoeHTTPError(
            401,
            headers={"Authorization": f"Bearer {secret}"},
            response_body={"error": f"api_key={secret}"},
        )

    with pytest.raises(PoeFailStopError) as captured:
        call_with_retry(
            operation,
            limiter=PoeRateLimiter(clock=clock, sleeper=clock.sleep),
            policy=PoeRetryPolicy(jitter_source=lambda: 0.0),
        )

    assert calls == 1
    assert clock.sleeps == []
    assert captured.value.code == PoeErrorCode.AUTHENTICATION_FAILED.value
    assert secret not in str(captured.value)
    assert secret not in str(captured.value.to_dict())


def test_transient_failure_has_a_hard_attempt_bound() -> None:
    clock = FakeClock()
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("network stalled with potentially unsafe raw text")

    with pytest.raises(PoeRetryExhaustedError) as captured:
        call_with_retry(
            operation,
            limiter=PoeRateLimiter(clock=clock, sleeper=clock.sleep),
            policy=PoeRetryPolicy(jitter_source=lambda: 0.0),
        )

    assert calls == 3
    assert clock.sleeps == [0.25, 0.5]
    assert captured.value.attempt_count == 3
    assert len(captured.value.retry_events) == 2
    assert "network stalled" not in str(captured.value.to_dict())
