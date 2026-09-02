"""Fail-closed Poe model discovery and capability smoke probes.

The module deliberately keeps credentials outside every persisted type.  An API
key supplier is called only at the live HTTP boundary, and response bodies or
transport exception text are never copied into structured errors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

POE_BASE_URL = "https://api.poe.com/v1"
POE_MODEL_CATALOG_URL = f"{POE_BASE_URL}/models"
POE_RESPONSES_URL = f"{POE_BASE_URL}/responses"
POE_CHAT_COMPLETIONS_URL = f"{POE_BASE_URL}/chat/completions"
POE_API_KEY_ENV = "POE_API_KEY"
REQUIRED_MODEL_ID = "gpt-5.4-mini"
REQUIRED_BOT_NAME = "GPT-5.4-Mini"
REQUIRED_MODEL_OWNER = "OpenAI"
REQUIRED_ENDPOINTS = ("/v1/responses", "/v1/chat/completions")
REQUIRED_FEATURES = ("tools",)
CAPABILITY_REPORT_FORMAT_VERSION = "poe_capability_probe_v1"
DEFAULT_CAPABILITY_REPORT_PATH = Path("Dataset/reports/poe_capability_probe.json")

_PROBE_NAMES = (
    "plain_response",
    "json_schema_response",
    "single_local_tool_loop",
)
_ECHO_TOOL_NAME = "poe_capability_echo"


class PoeModelRegistryError(RuntimeError):
    """A structured error whose fields are safe to persist or display."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("registry error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("registry error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("registry error evidence must be a mapping or None")
        safe_evidence = _json_object(evidence or {}, field_name="error evidence")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(safe_evidence)
        super().__init__(f"{code}: {detail}")


def _json_value(value: object, *, field_name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or key in result:
                raise ValueError(f"{field_name} requires unique string object keys")
            result[key] = _json_value(item, field_name=field_name)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, field_name=field_name) for item in value]
    raise TypeError(f"{field_name} must contain only JSON values")


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    normalized = _json_value(value, field_name=field_name)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return normalized


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _single_line(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{field_name} must be non-empty single-line text")
    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    items = tuple(_single_line(item, field_name=field_name) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return items


def _timestamp(value: object, *, field_name: str) -> str:
    text = _single_line(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field_name} must use UTC")
    return text


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class PoeHTTPResponse:
    """Minimal transport-neutral JSON response."""

    status_code: int
    json_body: object
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("HTTP status code must be an exact integer")
        headers = dict(self.headers)
        if any(
            type(key) is not str or type(value) is not str
            for key, value in headers.items()
        ):
            raise TypeError("HTTP headers must be text pairs")
        object.__setattr__(self, "headers", MappingProxyType(headers))


@runtime_checkable
class PoeHTTPTransport(Protocol):
    """Injectable synchronous JSON transport used by discovery and probes."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None = None,
    ) -> PoeHTTPResponse: ...


@dataclass(frozen=True, slots=True)
class HttpxPoeTransport:
    """Small production transport; it never logs request or response content."""

    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any] | None = None,
    ) -> PoeHTTPResponse:
        try:
            import httpx

            response = httpx.request(
                method,
                url,
                headers=dict(headers),
                json=None if json_body is None else dict(json_body),
                timeout=float(self.timeout_seconds),
            )
            payload = response.json()
        except Exception:  # noqa: BLE001 - redact arbitrary client/body failures
            raise PoeModelRegistryError(
                "POE_TRANSPORT_FAILED",
                "Poe HTTP request failed before a trusted JSON response was available",
                evidence={"method": method, "url": url},
            ) from None
        return PoeHTTPResponse(
            status_code=response.status_code,
            json_body=payload,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"x-request-id", "request-id"}
            },
        )


@dataclass(frozen=True, slots=True)
class PoeModelEntry:
    """One exact catalog entry plus the capabilities required by this build."""

    model_id: str
    bot_name: str
    owned_by: str
    created: int
    supported_endpoints: tuple[str, ...]
    supported_features: tuple[str, ...]
    _canonical_entry: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _single_line(self.model_id, field_name="model id")
        _single_line(self.bot_name, field_name="bot name")
        _single_line(self.owned_by, field_name="model owner")
        if type(self.created) is not int or self.created < 0:
            raise ValueError("model created must be a non-negative exact integer")
        if (
            type(self.supported_endpoints) is not tuple
            or type(self.supported_features) is not tuple
        ):
            raise TypeError("model capability arrays must be exact tuples")
        if not self.supported_endpoints or not self.supported_features:
            raise ValueError("model capability arrays cannot be empty")
        for values, name in (
            (self.supported_endpoints, "supported_endpoints"),
            (self.supported_features, "supported_features"),
        ):
            for item in values:
                _single_line(item, field_name=name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicates")
        if type(self._canonical_entry) is not bytes:
            raise TypeError("canonical model entry must be bytes")

    @classmethod
    def from_catalog_payload(cls, payload: object) -> PoeModelEntry:
        try:
            raw = _json_object(payload, field_name="model catalog entry")
            if raw.get("object") != "model":
                raise ValueError("catalog entry object must be model")
            model_id = _single_line(raw.get("id"), field_name="model id")
            owned_by = _single_line(raw.get("owned_by"), field_name="model owner")
            created = raw.get("created")
            if type(created) is not int or created < 0:
                raise ValueError("model created must be a non-negative integer")
            display_name = raw.get("display_name")
            bot_name = raw.get("bot_name")
            if (
                display_name is not None
                and bot_name is not None
                and display_name != bot_name
            ):
                raise ValueError("catalog display_name and bot_name disagree")
            resolved_bot_name = _single_line(
                display_name if display_name is not None else bot_name,
                field_name="bot name",
            )
            endpoints = _string_tuple(
                raw.get("supported_endpoints"),
                field_name="supported_endpoints",
            )
            features = _string_tuple(
                raw.get("supported_features"),
                field_name="supported_features",
            )
        except (TypeError, ValueError) as error:
            raise PoeModelRegistryError(
                "MODEL_CATALOG_SCHEMA_CHANGED",
                "Poe model catalog entry no longer satisfies the frozen schema",
            ) from error
        return cls(
            model_id=model_id,
            bot_name=resolved_bot_name,
            owned_by=owned_by,
            created=created,
            supported_endpoints=endpoints,
            supported_features=features,
            _canonical_entry=_canonical_json_bytes(raw),
        )

    @property
    def entry_sha256(self) -> str:
        return _sha256(self._canonical_entry)

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self._canonical_entry)
        assert isinstance(value, dict)
        return value


@dataclass(frozen=True, slots=True)
class PoeModelCatalog:
    entries: tuple[PoeModelEntry, ...]
    fetched_at: str

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("catalog entries must be an exact tuple")
        ordered = tuple(sorted(self.entries, key=lambda item: item.model_id))
        if any(type(item) is not PoeModelEntry for item in ordered):
            raise TypeError("catalog entries must contain PoeModelEntry values")
        identifiers = tuple(item.model_id for item in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise PoeModelRegistryError(
                "MODEL_CATALOG_DUPLICATE_ID",
                "Poe model catalog contains duplicate model IDs",
            )
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(
            self,
            "fetched_at",
            _timestamp(self.fetched_at, field_name="catalog_fetched_at"),
        )

    @classmethod
    def from_response_payload(
        cls,
        payload: object,
        *,
        fetched_at: str,
    ) -> PoeModelCatalog:
        try:
            root = _json_object(payload, field_name="model catalog")
            if set(root) != {"object", "data"} or root.get("object") != "list":
                raise ValueError("catalog root fields changed")
            rows = root.get("data")
            if not isinstance(rows, list) or not rows:
                raise ValueError("catalog data must be a non-empty array")
            entries = tuple(PoeModelEntry.from_catalog_payload(row) for row in rows)
        except PoeModelRegistryError:
            raise
        except (TypeError, ValueError) as error:
            raise PoeModelRegistryError(
                "MODEL_CATALOG_SCHEMA_CHANGED",
                "Poe model catalog response no longer satisfies the frozen schema",
            ) from error
        return cls(entries=entries, fetched_at=fetched_at)

    def require(
        self,
        model_id: str = REQUIRED_MODEL_ID,
        *,
        expected_entry_sha256: str | None = None,
    ) -> PoeModelEntry:
        if model_id != REQUIRED_MODEL_ID:
            raise PoeModelRegistryError(
                "UNAPPROVED_MODEL_ID",
                "only the frozen Poe model ID may be selected",
            )
        matches = tuple(item for item in self.entries if item.model_id == model_id)
        if not matches:
            raise PoeModelRegistryError(
                "REQUIRED_MODEL_MISSING",
                "the frozen Poe model is absent from the refreshed catalog",
                evidence={"model_id": model_id},
            )
        entry = matches[0]
        if entry.bot_name != REQUIRED_BOT_NAME:
            raise PoeModelRegistryError(
                "MODEL_BOT_MAPPING_CHANGED",
                "the API model ID no longer maps to the frozen Poe bot name",
            )
        if entry.owned_by != REQUIRED_MODEL_OWNER:
            raise PoeModelRegistryError(
                "MODEL_OWNER_CHANGED",
                "the frozen Poe model no longer declares the expected owner",
            )
        missing_endpoints = tuple(
            item for item in REQUIRED_ENDPOINTS if item not in entry.supported_endpoints
        )
        missing_features = tuple(
            item for item in REQUIRED_FEATURES if item not in entry.supported_features
        )
        if missing_endpoints or missing_features:
            raise PoeModelRegistryError(
                "REQUIRED_MODEL_CAPABILITY_MISSING",
                "the frozen Poe model lacks a required endpoint or feature",
                evidence={
                    "missing_endpoints": list(missing_endpoints),
                    "missing_features": list(missing_features),
                },
            )
        if expected_entry_sha256 is not None:
            if not _is_sha256(expected_entry_sha256):
                raise ValueError("expected_entry_sha256 must be lowercase SHA256")
            if entry.entry_sha256 != expected_entry_sha256:
                raise PoeModelRegistryError(
                    "MODEL_CATALOG_ENTRY_CHANGED",
                    "the required model catalog entry differs from the frozen identity",
                )
        return entry


class CapabilityProbeStatus(StrEnum):
    PASSED = "passed"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True, slots=True)
class CapabilityProbeResult:
    name: str
    endpoint: str
    status: CapabilityProbeStatus
    request_ids: tuple[str, ...]
    response_models: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.name not in _PROBE_NAMES:
            raise ValueError("unknown capability probe name")
        if self.endpoint not in REQUIRED_ENDPOINTS:
            raise ValueError("capability probe endpoint is not approved")
        if type(self.status) is not CapabilityProbeStatus:
            raise TypeError("probe status must be CapabilityProbeStatus")
        for values, name in (
            (self.request_ids, "request_ids"),
            (self.response_models, "response_models"),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            if any(_single_line(item, field_name=name) != item for item in values):
                raise ValueError(f"invalid {name}")
        if self.status is CapabilityProbeStatus.PASSED:
            if not self.request_ids or not self.response_models:
                raise ValueError("passed probes require response provenance")
            if set(self.response_models) != {REQUIRED_MODEL_ID}:
                raise ValueError("probe response model differs from requested model")
        elif self.request_ids or self.response_models:
            raise ValueError("unexecuted probes cannot claim response provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "request_ids": list(self.request_ids),
            "response_models": list(self.response_models),
        }


@dataclass(frozen=True, slots=True)
class PoeCapabilityReport:
    """A successful catalog-bound result from either live or mock transport."""

    execution_mode: str
    model_entry: PoeModelEntry
    catalog_fetched_at: str
    probes: tuple[CapabilityProbeResult, ...]

    def __post_init__(self) -> None:
        if self.execution_mode not in {"live", "deterministic_mock"}:
            raise ValueError("execution_mode must be live or deterministic_mock")
        if type(self.model_entry) is not PoeModelEntry:
            raise TypeError("model_entry must be PoeModelEntry")
        _timestamp(self.catalog_fetched_at, field_name="catalog_fetched_at")
        if type(self.probes) is not tuple:
            raise TypeError("probes must be an exact tuple")
        if any(type(item) is not CapabilityProbeResult for item in self.probes):
            raise TypeError("probes must contain CapabilityProbeResult values")
        if tuple(item.name for item in self.probes) != _PROBE_NAMES:
            raise ValueError("capability report requires all probes in frozen order")
        if any(item.status is not CapabilityProbeStatus.PASSED for item in self.probes):
            raise ValueError("successful report cannot contain an unexecuted probe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "execution_status": "passed",
            "requested_model_id": REQUIRED_MODEL_ID,
            "bot_name": self.model_entry.bot_name,
            "catalog_entry": self.model_entry.to_dict(),
            "catalog_entry_sha256": self.model_entry.entry_sha256,
            "catalog_fetched_at": self.catalog_fetched_at,
            "probes": [item.to_dict() for item in self.probes],
        }


def _not_executed_probes() -> list[dict[str, Any]]:
    endpoints = (
        "/v1/responses",
        "/v1/responses",
        "/v1/chat/completions",
    )
    return [
        CapabilityProbeResult(
            name=name,
            endpoint=endpoint,
            status=CapabilityProbeStatus.NOT_EXECUTED,
            request_ids=(),
            response_models=(),
        ).to_dict()
        for name, endpoint in zip(_PROBE_NAMES, endpoints, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class PoeCapabilityProbeArtifact:
    """Persisted live status plus deterministic mocked contract validation."""

    deterministic_validation: PoeCapabilityReport
    live_report: PoeCapabilityReport | None = None
    offline_reason_code: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.deterministic_validation) is not PoeCapabilityReport
            or self.deterministic_validation.execution_mode != "deterministic_mock"
        ):
            raise TypeError(
                "deterministic_validation must be a deterministic_mock report"
            )
        if self.live_report is None:
            _single_line(
                self.offline_reason_code,
                field_name="offline_reason_code",
            )
        else:
            if (
                type(self.live_report) is not PoeCapabilityReport
                or self.live_report.execution_mode != "live"
            ):
                raise TypeError("live_report must be a live PoeCapabilityReport")
            if self.offline_reason_code is not None:
                raise ValueError("a live report cannot also have an offline reason")

    def to_dict(self) -> dict[str, Any]:
        if self.live_report is None:
            live: dict[str, Any] = {
                "execution_mode": "live",
                "execution_status": "offline_not_executed",
                "reason_code": self.offline_reason_code,
                "requested_model_id": REQUIRED_MODEL_ID,
                "catalog_entry": None,
                "catalog_entry_sha256": None,
                "catalog_fetched_at": None,
                "probes": _not_executed_probes(),
            }
        else:
            live = self.live_report.to_dict()
        return {
            "format_version": CAPABILITY_REPORT_FORMAT_VERSION,
            "required_model_id": REQUIRED_MODEL_ID,
            "required_bot_name": REQUIRED_BOT_NAME,
            "required_model_owner": REQUIRED_MODEL_OWNER,
            "required_endpoints": list(REQUIRED_ENDPOINTS),
            "required_features": list(REQUIRED_FEATURES),
            "live_probe": live,
            "deterministic_mock_validation": self.deterministic_validation.to_dict(),
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def write_capability_probe_artifact(
    artifact: PoeCapabilityProbeArtifact,
    *,
    path: Path = DEFAULT_CAPABILITY_REPORT_PATH,
) -> None:
    if type(artifact) is not PoeCapabilityProbeArtifact:
        raise TypeError("artifact must be PoeCapabilityProbeArtifact")
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_json_bytes()
    if path.exists() and path.read_bytes() == payload:
        return
    path.write_bytes(payload)


def _default_api_key_provider() -> str | None:
    # Environment access is intentionally delayed until a live registry method.
    return os.environ.get(POE_API_KEY_ENV)


def _require_api_key(provider: Callable[[], str | None]) -> str:
    try:
        value = provider()
    except Exception:  # noqa: BLE001 - provider is injected and must be redacted
        raise PoeModelRegistryError(
            "POE_API_KEY_UNAVAILABLE",
            "the Poe API key provider failed",
        ) from None
    if (
        type(value) is not str
        or not value
        or len(value) < 16
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise PoeModelRegistryError(
            "POE_API_KEY_UNAVAILABLE",
            "POE_API_KEY is unavailable for a live capability probe",
        )
    return value


def _authorization_headers(api_key: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )


def _response_request_id(payload: Mapping[str, Any]) -> str:
    try:
        return _single_line(payload.get("id"), field_name="Poe response id")
    except ValueError:
        raise PoeModelRegistryError(
            "CAPABILITY_PROBE_RESPONSE_INVALID",
            "Poe capability probe response omitted a valid request ID",
        ) from None


def _response_model(payload: Mapping[str, Any]) -> str:
    try:
        model = _single_line(payload.get("model"), field_name="Poe response model")
    except ValueError:
        raise PoeModelRegistryError(
            "CAPABILITY_PROBE_RESPONSE_INVALID",
            "Poe capability probe response omitted a valid model ID",
        ) from None
    if model != REQUIRED_MODEL_ID:
        raise PoeModelRegistryError(
            "PROBE_MODEL_SUBSTITUTED",
            "Poe returned a model other than the frozen requested model",
        )
    return model


def _response_output_text(payload: Mapping[str, Any]) -> str:
    try:
        if payload.get("object") != "response" or payload.get("status") != "completed":
            raise ValueError
        output = payload["output"]
        if not isinstance(output, list) or len(output) != 1:
            raise ValueError
        message = output[0]
        if not isinstance(message, Mapping) or message.get("type") != "message":
            raise ValueError
        content = message["content"]
        if not isinstance(content, list) or len(content) != 1:
            raise ValueError
        block = content[0]
        if not isinstance(block, Mapping) or block.get("type") != "output_text":
            raise ValueError
        return _single_line(block.get("text"), field_name="probe output text")
    except (KeyError, TypeError, ValueError) as error:
        raise PoeModelRegistryError(
            "CAPABILITY_PROBE_RESPONSE_INVALID",
            "Poe Responses output did not satisfy the smoke-test contract",
        ) from error


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = item
    return value


def _strict_json_object_text(
    text: str,
    *,
    error_code: str,
    error_detail: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (json.JSONDecodeError, ValueError, _DuplicateJsonKeyError):
        raise PoeModelRegistryError(
            error_code,
            error_detail,
        ) from None
    if type(value) is not dict:
        raise PoeModelRegistryError(error_code, error_detail)
    return value


def _parse_probe_json(text: str) -> Mapping[str, Any]:
    value = _strict_json_object_text(
        text,
        error_code="JSON_SCHEMA_PROBE_FAILED",
        error_detail="structured-output probe did not return strict JSON",
    )
    if value != {"probe": "ok"}:
        raise PoeModelRegistryError(
            "JSON_SCHEMA_PROBE_FAILED",
            "structured-output probe did not satisfy the local schema",
        )
    return value


def _chat_message(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        if payload.get("object") != "chat.completion":
            raise ValueError
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        if not isinstance(choice, Mapping) or choice.get("index") != 0:
            raise ValueError
        message = choice["message"]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise ValueError
        return message
    except (KeyError, TypeError, ValueError) as error:
        raise PoeModelRegistryError(
            "TOOL_LOOP_PROBE_FAILED",
            "chat tool probe response did not satisfy the local envelope",
        ) from error


@dataclass(frozen=True, slots=True)
class PoeModelRegistry:
    """Refresh the runtime catalog and prove all required Poe capabilities."""

    transport: PoeHTTPTransport
    api_key_provider: Callable[[], str | None] = field(
        default=_default_api_key_provider,
        repr=False,
        compare=False,
    )
    clock: Callable[[], str] = field(default=_utc_now, repr=False, compare=False)
    expected_entry_sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transport, PoeHTTPTransport):
            raise TypeError("transport must satisfy PoeHTTPTransport")
        if not callable(self.api_key_provider) or not callable(self.clock):
            raise TypeError("api_key_provider and clock must be callable")
        if self.expected_entry_sha256 is not None and not _is_sha256(
            self.expected_entry_sha256
        ):
            raise ValueError("expected_entry_sha256 must be lowercase SHA256")

    def _request(
        self,
        *,
        api_key: str,
        method: str,
        url: str,
        json_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        try:
            response = self.transport.request(
                method,
                url,
                headers=_authorization_headers(api_key),
                json_body=json_body,
            )
        except PoeModelRegistryError:
            raise
        except Exception:  # noqa: BLE001 - transport is an injected trust boundary
            raise PoeModelRegistryError(
                "POE_TRANSPORT_FAILED",
                "injected Poe transport failed",
                evidence={"method": method, "url": url},
            ) from None
        if type(response) is not PoeHTTPResponse:
            raise PoeModelRegistryError(
                "POE_TRANSPORT_CONTRACT",
                "injected transport returned an untyped response",
            )
        if response.status_code != 200:
            raise PoeModelRegistryError(
                "POE_HTTP_STATUS",
                "Poe returned a non-success status",
                evidence={"status_code": response.status_code, "url": url},
            )
        try:
            return _json_object(response.json_body, field_name="Poe response body")
        except (TypeError, ValueError) as error:
            raise PoeModelRegistryError(
                "POE_RESPONSE_NOT_JSON_OBJECT",
                "Poe returned a response outside the JSON object contract",
            ) from error

    def _refresh_with_key(self, api_key: str) -> PoeModelCatalog:
        payload = self._request(
            api_key=api_key,
            method="GET",
            url=POE_MODEL_CATALOG_URL,
        )
        try:
            fetched_at = self.clock()
        except Exception as error:
            raise PoeModelRegistryError(
                "REGISTRY_CLOCK_FAILED",
                "catalog clock failed",
            ) from error
        catalog = PoeModelCatalog.from_response_payload(
            payload,
            fetched_at=fetched_at,
        )
        catalog.require(
            REQUIRED_MODEL_ID,
            expected_entry_sha256=self.expected_entry_sha256,
        )
        return catalog

    def refresh(self) -> PoeModelCatalog:
        """Refresh catalog; this is the first point that reads POE_API_KEY."""

        return self._refresh_with_key(_require_api_key(self.api_key_provider))

    def _probe_with_key(
        self,
        catalog: PoeModelCatalog,
        api_key: str,
        *,
        execution_mode: str,
    ) -> PoeCapabilityReport:
        entry = catalog.require(
            REQUIRED_MODEL_ID,
            expected_entry_sha256=self.expected_entry_sha256,
        )
        probes = (
            self._probe_plain(api_key),
            self._probe_json_schema(api_key),
            self._probe_tool_loop(api_key),
        )
        report = PoeCapabilityReport(
            execution_mode=execution_mode,
            model_entry=entry,
            catalog_fetched_at=catalog.fetched_at,
            probes=probes,
        )
        if api_key.encode("utf-8") in _canonical_json_bytes(report.to_dict()):
            raise PoeModelRegistryError(
                "SECRET_REDACTION_BOUNDARY",
                "a provider response attempted to place the API key in provenance",
            )
        return report

    def probe(
        self,
        catalog: PoeModelCatalog,
        *,
        execution_mode: str = "live",
    ) -> PoeCapabilityReport:
        if type(catalog) is not PoeModelCatalog:
            raise TypeError("catalog must be PoeModelCatalog")
        return self._probe_with_key(
            catalog,
            _require_api_key(self.api_key_provider),
            execution_mode=execution_mode,
        )

    def refresh_and_probe(
        self,
        *,
        execution_mode: str = "live",
    ) -> PoeCapabilityReport:
        """Read one ephemeral credential, then run discovery and all probes."""

        api_key = _require_api_key(self.api_key_provider)
        catalog = self._refresh_with_key(api_key)
        return self._probe_with_key(
            catalog,
            api_key,
            execution_mode=execution_mode,
        )

    def _probe_plain(self, api_key: str) -> CapabilityProbeResult:
        payload = self._request(
            api_key=api_key,
            method="POST",
            url=POE_RESPONSES_URL,
            json_body={
                "model": REQUIRED_MODEL_ID,
                "input": "Reply with exactly POE_PLAIN_OK.",
                "max_output_tokens": 16,
                "temperature": 0,
            },
        )
        request_id = _response_request_id(payload)
        model = _response_model(payload)
        if _response_output_text(payload) != "POE_PLAIN_OK":
            raise PoeModelRegistryError(
                "PLAIN_RESPONSE_PROBE_FAILED",
                "plain response probe did not return the exact sentinel",
            )
        return CapabilityProbeResult(
            name="plain_response",
            endpoint="/v1/responses",
            status=CapabilityProbeStatus.PASSED,
            request_ids=(request_id,),
            response_models=(model,),
        )

    def _probe_json_schema(self, api_key: str) -> CapabilityProbeResult:
        schema = {
            "type": "object",
            "properties": {"probe": {"type": "string", "const": "ok"}},
            "required": ["probe"],
            "additionalProperties": False,
        }
        payload = self._request(
            api_key=api_key,
            method="POST",
            url=POE_RESPONSES_URL,
            json_body={
                "model": REQUIRED_MODEL_ID,
                "input": "Return the requested capability probe object.",
                "temperature": 0,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "poe_capability_probe",
                        "schema": schema,
                        "strict": True,
                    }
                },
            },
        )
        request_id = _response_request_id(payload)
        model = _response_model(payload)
        _parse_probe_json(_response_output_text(payload))
        return CapabilityProbeResult(
            name="json_schema_response",
            endpoint="/v1/responses",
            status=CapabilityProbeStatus.PASSED,
            request_ids=(request_id,),
            response_models=(model,),
        )

    def _probe_tool_loop(self, api_key: str) -> CapabilityProbeResult:
        tool = {
            "type": "function",
            "function": {
                "name": _ECHO_TOOL_NAME,
                "description": "Return the supplied probe value unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string", "const": "probe"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Call poe_capability_echo exactly once with value probe. "
                    "After its result, reply exactly POE_TOOL_OK."
                ),
            }
        ]
        first = self._request(
            api_key=api_key,
            method="POST",
            url=POE_CHAT_COMPLETIONS_URL,
            json_body={
                "model": REQUIRED_MODEL_ID,
                "messages": messages,
                "tools": [tool],
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "temperature": 0,
            },
        )
        first_id = _response_request_id(first)
        first_model = _response_model(first)
        first_message = _chat_message(first)
        try:
            calls = first_message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError
            call = calls[0]
            if not isinstance(call, Mapping) or call.get("type") != "function":
                raise ValueError
            call_id = _single_line(call.get("id"), field_name="tool call id")
            function = call["function"]
            if (
                not isinstance(function, Mapping)
                or function.get("name") != _ECHO_TOOL_NAME
            ):
                raise ValueError
            arguments_text = _single_line(
                function.get("arguments"),
                field_name="tool arguments",
            )
            arguments = _strict_json_object_text(
                arguments_text,
                error_code="TOOL_LOOP_PROBE_FAILED",
                error_detail="tool arguments were not strict JSON",
            )
            if arguments != {"value": "probe"}:
                raise ValueError
        except (KeyError, TypeError, ValueError, PoeModelRegistryError):
            raise PoeModelRegistryError(
                "TOOL_LOOP_PROBE_FAILED",
                "model did not issue the one allowed local tool call",
            ) from None

        assistant_message = {
            "role": "assistant",
            "content": first_message.get("content"),
            "tool_calls": _json_value(calls, field_name="tool calls"),
        }
        messages.extend(
            (
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"value": arguments["value"]},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            )
        )
        final = self._request(
            api_key=api_key,
            method="POST",
            url=POE_CHAT_COMPLETIONS_URL,
            json_body={
                "model": REQUIRED_MODEL_ID,
                "messages": messages,
                "tools": [tool],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "temperature": 0,
            },
        )
        final_id = _response_request_id(final)
        final_model = _response_model(final)
        final_message = _chat_message(final)
        if final_message.get("content") != "POE_TOOL_OK" or final_message.get(
            "tool_calls"
        ) not in (None, []):
            raise PoeModelRegistryError(
                "TOOL_LOOP_PROBE_FAILED",
                "tool-result continuation did not return the exact sentinel",
            )
        return CapabilityProbeResult(
            name="single_local_tool_loop",
            endpoint="/v1/chat/completions",
            status=CapabilityProbeStatus.PASSED,
            request_ids=(first_id, final_id),
            response_models=(first_model, final_model),
        )


__all__ = [
    "CAPABILITY_REPORT_FORMAT_VERSION",
    "DEFAULT_CAPABILITY_REPORT_PATH",
    "POE_API_KEY_ENV",
    "POE_BASE_URL",
    "POE_CHAT_COMPLETIONS_URL",
    "POE_MODEL_CATALOG_URL",
    "POE_RESPONSES_URL",
    "REQUIRED_BOT_NAME",
    "REQUIRED_ENDPOINTS",
    "REQUIRED_FEATURES",
    "REQUIRED_MODEL_ID",
    "REQUIRED_MODEL_OWNER",
    "CapabilityProbeResult",
    "CapabilityProbeStatus",
    "HttpxPoeTransport",
    "PoeCapabilityProbeArtifact",
    "PoeCapabilityReport",
    "PoeHTTPResponse",
    "PoeHTTPTransport",
    "PoeModelCatalog",
    "PoeModelEntry",
    "PoeModelRegistry",
    "PoeModelRegistryError",
    "write_capability_probe_artifact",
]
