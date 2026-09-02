"""T033 Poe catalog and capability-probe trust-boundary tests."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from molhallulens.infrastructure.providers.poe.model_registry import (
    POE_CHAT_COMPLETIONS_URL,
    POE_MODEL_CATALOG_URL,
    POE_RESPONSES_URL,
    REQUIRED_MODEL_ID,
    PoeCapabilityProbeArtifact,
    PoeCapabilityReport,
    PoeHTTPResponse,
    PoeModelEntry,
    PoeModelRegistry,
    PoeModelRegistryError,
    write_capability_probe_artifact,
)

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Dataset"
    / "reports"
    / "poe_capability_probe.json"
)
FIXED_TIME = "2026-08-30T02:30:00Z"
TEST_SECRET = "poe-test-secret-that-must-never-be-persisted"


def _entry() -> dict[str, Any]:
    return {
        "id": "gpt-5.4-mini",
        "object": "model",
        "created": 1788015600000,
        "display_name": "GPT-5.4-Mini",
        "owned_by": "OpenAI",
        "permission": [],
        "root": "gpt-5.4-mini",
        "parent": None,
        "supported_features": ["tools", "web_search"],
        "supported_endpoints": [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/messages",
        ],
        "context_length": 400000,
        "max_output_tokens": 128000,
    }


def _catalog(entry: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"object": "list", "data": [entry or _entry()]}


def _responses_payload(identifier: str, text: str, *, model: str = REQUIRED_MODEL_ID):
    return {
        "id": identifier,
        "object": "response",
        "created_at": 1788015600,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def _tool_call_payload() -> dict[str, Any]:
    return {
        "id": "chat-tool-call",
        "object": "chat.completion",
        "created": 1788015601,
        "model": REQUIRED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-echo",
                            "type": "function",
                            "function": {
                                "name": "poe_capability_echo",
                                "arguments": '{"value":"probe"}',
                            },
                        }
                    ],
                },
            }
        ],
    }


def _tool_final_payload(text: str = "POE_TOOL_OK") -> dict[str, Any]:
    return {
        "id": "chat-tool-final",
        "object": "chat.completion",
        "created": 1788015602,
        "model": REQUIRED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def _successful_responses() -> list[PoeHTTPResponse]:
    return [
        PoeHTTPResponse(200, _catalog()),
        PoeHTTPResponse(200, _responses_payload("resp-plain", "POE_PLAIN_OK")),
        PoeHTTPResponse(200, _responses_payload("resp-json", '{"probe":"ok"}')),
        PoeHTTPResponse(200, _tool_call_payload()),
        PoeHTTPResponse(200, _tool_final_payload()),
    ]


class ScriptedTransport:
    def __init__(self, responses: list[PoeHTTPResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers,
        json_body=None,
    ) -> PoeHTTPResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "json_body": None if json_body is None else deepcopy(json_body),
            }
        )
        if not self.responses:
            raise AssertionError("unexpected transport call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _registry(transport: ScriptedTransport, **kwargs) -> PoeModelRegistry:
    return PoeModelRegistry(
        transport=transport,
        api_key_provider=lambda: TEST_SECRET,
        clock=lambda: FIXED_TIME,
        **kwargs,
    )


@cache
def _mock_report() -> PoeCapabilityReport:
    return _registry(ScriptedTransport(_successful_responses())).refresh_and_probe(
        execution_mode="deterministic_mock"
    )


@cache
def _offline_artifact() -> PoeCapabilityProbeArtifact:
    return PoeCapabilityProbeArtifact(
        deterministic_validation=_mock_report(),
        offline_reason_code="POE_TRANSPORT_FAILED",
    )


def test_key_read_is_lazy_and_successful_probe_covers_both_endpoints() -> None:
    calls = 0

    def key_provider() -> str:
        nonlocal calls
        calls += 1
        return TEST_SECRET

    transport = ScriptedTransport(_successful_responses())
    registry = PoeModelRegistry(
        transport=transport,
        api_key_provider=key_provider,
        clock=lambda: FIXED_TIME,
    )
    assert calls == 0

    report = registry.refresh_and_probe(execution_mode="deterministic_mock")

    assert calls == 1
    assert report.catalog_fetched_at == FIXED_TIME
    assert tuple(item.name for item in report.probes) == (
        "plain_response",
        "json_schema_response",
        "single_local_tool_loop",
    )
    assert [item["url"] for item in transport.calls] == [
        POE_MODEL_CATALOG_URL,
        POE_RESPONSES_URL,
        POE_RESPONSES_URL,
        POE_CHAT_COMPLETIONS_URL,
        POE_CHAT_COMPLETIONS_URL,
    ]
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["json_body"] is None
    assert all(
        item["headers"]["Authorization"] == f"Bearer {TEST_SECRET}"
        for item in transport.calls
    )
    json_probe = transport.calls[2]["json_body"]
    assert json_probe["text"]["format"]["type"] == "json_schema"
    first_tool = transport.calls[3]["json_body"]
    final_tool = transport.calls[4]["json_body"]
    assert first_tool["parallel_tool_calls"] is False
    assert final_tool["parallel_tool_calls"] is False
    assert final_tool["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-echo",
        "content": '{"value":"probe"}',
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("missing_model", "REQUIRED_MODEL_MISSING"),
        ("missing_endpoint", "REQUIRED_MODEL_CAPABILITY_MISSING"),
        ("missing_tools", "REQUIRED_MODEL_CAPABILITY_MISSING"),
        ("bot_mapping", "MODEL_BOT_MAPPING_CHANGED"),
        ("owner", "MODEL_OWNER_CHANGED"),
        ("entry_identity", "MODEL_CATALOG_ENTRY_CHANGED"),
    ),
)
def test_catalog_model_or_capability_change_fails_closed(
    mutation: str,
    expected_code: str,
) -> None:
    entry = _entry()
    expected_identity = PoeModelEntry.from_catalog_payload(entry).entry_sha256
    if mutation == "missing_model":
        entry["id"] = "another-model"
    elif mutation == "missing_endpoint":
        entry["supported_endpoints"].remove("/v1/responses")
    elif mutation == "missing_tools":
        entry["supported_features"].remove("tools")
    elif mutation == "bot_mapping":
        entry["display_name"] = "Another-Bot"
    elif mutation == "owner":
        entry["owned_by"] = "Another-Owner"
    else:
        entry["description"] = "catalog drift"
    transport = ScriptedTransport([PoeHTTPResponse(200, _catalog(entry))])
    registry = _registry(
        transport,
        expected_entry_sha256=(
            expected_identity if mutation == "entry_identity" else None
        ),
    )

    with pytest.raises(PoeModelRegistryError) as captured:
        registry.refresh()

    assert captured.value.code == expected_code
    assert len(transport.calls) == 1


@pytest.mark.parametrize("mutation", ("root_field", "duplicate_id", "entry_shape"))
def test_catalog_schema_drift_and_duplicate_ids_are_rejected(mutation: str) -> None:
    payload = _catalog()
    if mutation == "root_field":
        payload["next"] = None
    elif mutation == "duplicate_id":
        payload["data"].append(deepcopy(payload["data"][0]))
    else:
        del payload["data"][0]["supported_features"]
    registry = _registry(ScriptedTransport([PoeHTTPResponse(200, payload)]))

    with pytest.raises(PoeModelRegistryError) as captured:
        registry.refresh()

    assert captured.value.code in {
        "MODEL_CATALOG_SCHEMA_CHANGED",
        "MODEL_CATALOG_DUPLICATE_ID",
    }


@pytest.mark.parametrize(
    ("responses", "expected_code"),
    (
        (
            [
                PoeHTTPResponse(200, _catalog()),
                PoeHTTPResponse(200, _responses_payload("plain", "wrong")),
            ],
            "PLAIN_RESPONSE_PROBE_FAILED",
        ),
        (
            [
                PoeHTTPResponse(200, _catalog()),
                PoeHTTPResponse(200, _responses_payload("plain", "POE_PLAIN_OK")),
                PoeHTTPResponse(200, _responses_payload("json", '{"probe":true}')),
            ],
            "JSON_SCHEMA_PROBE_FAILED",
        ),
        (
            [
                PoeHTTPResponse(200, _catalog()),
                PoeHTTPResponse(200, _responses_payload("plain", "POE_PLAIN_OK")),
                PoeHTTPResponse(200, _responses_payload("json", '{"probe":"ok"}')),
                PoeHTTPResponse(200, _tool_final_payload("skipped tool")),
            ],
            "TOOL_LOOP_PROBE_FAILED",
        ),
    ),
)
def test_any_failed_smoke_probe_fails_closed(
    responses: list[PoeHTTPResponse],
    expected_code: str,
) -> None:
    registry = _registry(ScriptedTransport(responses))

    with pytest.raises(PoeModelRegistryError) as captured:
        registry.refresh_and_probe(execution_mode="deterministic_mock")

    assert captured.value.code == expected_code


def test_missing_key_and_transport_exception_never_disclose_secret() -> None:
    unused_transport = ScriptedTransport([])
    missing = PoeModelRegistry(
        transport=unused_transport,
        api_key_provider=lambda: None,
    )
    with pytest.raises(PoeModelRegistryError) as no_key:
        missing.refresh()
    assert no_key.value.code == "POE_API_KEY_UNAVAILABLE"
    assert unused_transport.calls == []

    failing = _registry(
        ScriptedTransport([RuntimeError(f"upstream echoed {TEST_SECRET}")])
    )
    with pytest.raises(PoeModelRegistryError) as transport_error:
        failing.refresh()
    serialized = json.dumps(
        {
            "error": str(transport_error.value),
            "evidence": dict(transport_error.value.evidence),
        }
    )
    assert transport_error.value.code == "POE_TRANSPORT_FAILED"
    assert transport_error.value.__cause__ is None
    assert TEST_SECRET not in serialized


def test_offline_artifact_is_truthful_stable_and_contains_no_secret(
    tmp_path: Path,
) -> None:
    artifact = _offline_artifact()
    payload = artifact.to_dict()
    assert payload["live_probe"]["execution_status"] == "offline_not_executed"
    assert payload["live_probe"]["catalog_entry"] is None
    assert all(
        item["status"] == "not_executed" for item in payload["live_probe"]["probes"]
    )
    assert payload["deterministic_mock_validation"]["execution_status"] == "passed"
    assert TEST_SECRET.encode() not in artifact.to_json_bytes()

    path = tmp_path / "probe.json"
    write_capability_probe_artifact(artifact, path=path)
    first = path.read_bytes()
    write_capability_probe_artifact(artifact, path=path)
    assert path.read_bytes() == first == artifact.to_json_bytes()


def test_committed_offline_report_is_byte_identical_to_mocked_rebuild() -> None:
    assert REPORT_PATH.read_bytes() == _offline_artifact().to_json_bytes()
