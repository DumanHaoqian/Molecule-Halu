"""Secret-free balance, history, and T034 usage accounting tests."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from molhallulens.infrastructure.providers.poe.rate_limiter import (
    PoeErrorAction,
    PoeErrorClassification,
    PoeErrorCode,
    PoeRetryEvent,
)
from molhallulens.infrastructure.providers.poe.usage_ledger import (
    POE_BALANCE_ENDPOINT,
    POE_MODEL_ID,
    POE_POINTS_HISTORY_ENDPOINT,
    PoeTokenUsage,
    PoeUsageErrorEvent,
    PoeUsageLedger,
    redact_sensitive,
)

FIXED_TIME = "2026-08-30T03:00:00Z"


class ScriptedUsageTransport:
    def __init__(self) -> None:
        self.balances: list[object] = [
            {"current_balance": 100},
            {"data": {"balance": 94}},
        ]
        self.history = {
            "data": [
                {
                    "id": "points-1",
                    "query_id": "query-1",
                    "request_id": "request-1",
                    "model": POE_MODEL_ID,
                    "cost_points": 6,
                }
            ]
        }
        self.calls: list[str] = []

    def get_current_balance(self) -> object:
        self.calls.append(POE_BALANCE_ENDPOINT)
        return self.balances.pop(0)

    def get_points_history(self) -> object:
        self.calls.append(POE_POINTS_HISTORY_ENDPOINT)
        return self.history


def test_token_usage_accepts_responses_and_chat_field_names() -> None:
    responses = PoeTokenUsage.from_provider(
        {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    )
    chat = PoeTokenUsage.from_provider(
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    )

    assert responses.to_dict() == {
        "prompt_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert chat.input_tokens == 5
    assert chat.completion_tokens == 2


def test_balance_history_and_request_usage_reconcile_exactly() -> None:
    transport = ScriptedUsageTransport()
    ledger = PoeUsageLedger(transport=transport, clock=lambda: FIXED_TIME)

    assert ledger.begin_build() == 100
    record = ledger.record_request(
        request_id="request-1",
        response_id="response-1",
        query_id="query-1",
        x_request_id="x-request-1",
        transport="responses",
        token_usage={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
        attempt_count=2,
        retry_count=1,
        errors=(
            PoeUsageErrorEvent(
                attempt=1,
                code="POE_RATE_LIMITED",
                retryable=True,
                detail="rate limited",
                status_code=429,
            ),
        ),
    )
    reconciliation = ledger.finish_build()

    assert record.cost_points is None
    assert ledger.records[0].cost_points == 6
    assert reconciliation.balance_spent_points == 6
    assert reconciliation.history_cost_points == 6
    assert reconciliation.recorded_cost_points == 6
    assert reconciliation.balanced
    assert transport.calls == [
        POE_BALANCE_ENDPOINT,
        POE_BALANCE_ENDPOINT,
        POE_POINTS_HISTORY_ENDPOINT,
    ]
    payload = ledger.to_dict()
    assert payload["records"][0]["request_id"] == "request-1"
    assert payload["records"][0]["response_id"] == "response-1"
    assert payload["records"][0]["query_id"] == "query-1"
    assert payload["records"][0]["x_request_id"] == "x-request-1"
    assert payload["records"][0]["token_usage"]["total_tokens"] == 15


def test_t034_public_result_contract_is_consumed_without_client_import() -> None:
    failed_attempt = SimpleNamespace(
        response_ids=(),
        query_ids=(),
        x_request_ids=("x-failed",),
        error_code="POE_RESPONSES_FAILED",
        error_detail="Responses adapter failed",
    )
    success_attempt = SimpleNamespace(
        response_ids=("chat-final",),
        query_ids=("query-final",),
        x_request_ids=("x-final",),
        error_code=None,
        error_detail=None,
    )
    provenance = SimpleNamespace(
        request_id="request-fallback",
        requested_model_id=POE_MODEL_ID,
        selected_transport=SimpleNamespace(value="chat_completions"),
        attempts=(failed_attempt, success_attempt),
    )
    result = SimpleNamespace(
        provenance=provenance,
        response=SimpleNamespace(
            usage={"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}
        ),
    )
    retry = PoeRetryEvent(
        failed_attempt=1,
        next_attempt=2,
        delay_seconds=0.25,
        classification=PoeErrorClassification(
            PoeErrorCode.PROVIDER_TRANSIENT,
            PoeErrorAction.RETRY,
            "provider transient",
            503,
        ),
    )
    ledger = PoeUsageLedger(clock=lambda: FIXED_TIME)

    record = ledger.record_client_result(result, retry_events=(retry,))

    assert record.request_id == "request-fallback"
    assert record.transport == "chat_completions"
    assert record.response_id == "chat-final"
    assert record.query_id == "query-final"
    assert record.x_request_ids == ("x-failed", "x-final")
    assert record.attempt_count == 3
    assert record.retry_count == 1
    assert record.token_usage.total_tokens == 18
    assert [error.code for error in record.errors] == [
        "POE_RESPONSES_FAILED",
        "POE_PROVIDER_TRANSIENT",
    ]


def test_private_export_redacts_credentials_and_has_owner_only_mode(
    tmp_path: Path,
) -> None:
    secret = "poe-secret-never-persist"
    ledger = PoeUsageLedger(clock=lambda: FIXED_TIME)
    ledger.record_request(
        request_id="request-secret-test",
        transport="fastapi_poe",
        errors=(
            PoeUsageErrorEvent(
                attempt=1,
                code="POE_AUTHENTICATION_FAILED",
                retryable=False,
                detail=f"Authorization: Bearer {secret}",
                status_code=401,
            ),
        ),
    )

    output_path = tmp_path / "private" / "ledger.json"
    serialized_ledger = json.dumps(ledger.to_dict())
    assert secret not in serialized_ledger
    assert "[REDACTED]" in serialized_ledger

    try:
        destination = ledger.export_private_json(output_path)
    except PermissionError:
        # Some mounted filesystems report every file as executable and ignore
        # chmod. The exporter must fail closed and remove the unsafe artifact.
        assert not output_path.exists()
        return
    text = destination.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert secret not in text
    assert "Bearer poe-" not in text
    assert "[REDACTED]" in text
    assert payload["records"][0]["errors"][0]["status_code"] == 401
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_recursive_redaction_covers_header_and_secret_key_variants() -> None:
    secret = "top-secret"
    value = redact_sensitive(
        {
            "headers": {"Authorization": f"Bearer {secret}", "X-API-Key": secret},
            "detail": f"POE_API_KEY={secret}",
        }
    )
    serialized = json.dumps(value, sort_keys=True)

    assert secret not in serialized
    assert serialized.count("[REDACTED]") == 3


def test_unmatched_history_and_balance_mismatch_are_explicit() -> None:
    ledger = PoeUsageLedger(clock=lambda: FIXED_TIME)
    ledger.record_request(request_id="request-a", transport="responses")

    report = ledger.reconcile(
        [
            {
                "id": "unmatched-points",
                "query_id": "another-query",
                "cost_points": 2,
            }
        ]
    )

    assert not report.balanced
    assert report.unmatched_request_ids == ("request-a",)
    assert report.unmatched_history_event_ids == ("unmatched-points",)


def test_duplicate_request_ids_are_rejected() -> None:
    ledger = PoeUsageLedger(clock=lambda: FIXED_TIME)
    ledger.record_request(request_id="request-duplicate", transport="responses")

    with pytest.raises(ValueError, match="duplicate"):
        ledger.record_request(request_id="request-duplicate", transport="responses")


def test_finish_build_requires_the_pre_build_balance_checkpoint() -> None:
    ledger = PoeUsageLedger(
        transport=ScriptedUsageTransport(), clock=lambda: FIXED_TIME
    )

    with pytest.raises(RuntimeError, match="pre-build"):
        ledger.finish_build()
