"""Immutable content-addressed Poe response cache and frozen replay.

The cache is a local reproducibility boundary, not a network client.  Callers
inject a T034-compatible producer for a cache miss; release/frozen mode never
invokes it.  Entries are canonical JSON, created atomically without replacing
an existing object, and bind the request to the verified model catalog and the
local prompt/schema/tool identities.

Only provider-safe structured data may cross this boundary.  Raw headers,
authorization material, API keys, cookies, and bearer credentials are rejected
before any filesystem write.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

CACHE_SCHEMA = "molhallulens.poe.response_cache"
CACHE_SCHEMA_VERSION = "1.0"
CACHE_KEY_VERSION = "1.0"
CLIENT_RESULT_SCHEMA_VERSION = "1.0"
ACCEPTED_ARTIFACT_SCHEMA_VERSION = "1.0"
CONTENT_IDENTITY_ALGORITHM = "sha256"
FROZEN_POE_MODEL_ID = "gpt-5.4-mini"

_HEX_CHARACTERS = frozenset("0123456789abcdef")
_RECORD_FIELDS = frozenset(
    {
        "cache_schema",
        "schema_version",
        "cache_key_version",
        "kind",
        "cache_key",
        "content_identity_algorithm",
        "content_sha256",
        "artifact_sha256",
        "created_at",
        "key_material",
        "payload",
        "record_sha256",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "proxy_authorization",
        "headers",
        "request_headers",
        "response_headers",
        "_headers",
        "api_key",
        "api-key",
        "x-api-key",
        "poe_api_key",
        "cookie",
        "set-cookie",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "bearer_token",
    }
)


class PoeCacheMode(StrEnum):
    """Whether a missing proposal may call the injected producer."""

    READ_WRITE = "read_write"
    FROZEN_REPLAY = "frozen_replay"


class PoeCacheKind(StrEnum):
    """Frozen on-disk namespaces from the implementation plan."""

    PROPOSAL = "proposals"
    TOOL_RUN = "tool_runs"
    RENDER = "renders"
    ACCEPTED = "accepted"


class PoeResponseCacheError(RuntimeError):
    """Secret-free structured failure from the cache trust boundary."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        kind: PoeCacheKind | None = None,
        cache_key: str | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("cache error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("cache error detail must be non-empty text")
        if kind is not None and type(kind) is not PoeCacheKind:
            raise TypeError("kind must be PoeCacheKind or None")
        if cache_key is not None:
            _require_digest(cache_key, field_name="cache_key")
        self.code = code
        self.detail = detail
        self.kind = kind
        self.cache_key = cache_key
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "kind": None if self.kind is None else self.kind.value,
            "cache_key": self.cache_key,
        }


@runtime_checkable
class PoeResultProducer(Protocol):
    """T034-compatible producer; typically one of the Poe client adapters."""

    def propose(self, request: object) -> object: ...


@dataclass(frozen=True, slots=True)
class PoeCacheContext:
    """All non-request identities that determine a proposal cache key.

    ``prompt_identity``, ``proposal_schema_identity``, and
    ``tool_schema_identity`` default to identities derived from the installed
    T031/T034 contracts.  ``tool_result_identities`` binds deterministic local
    T032 results that were prepared before the provider request.
    """

    source_record_sha256: str
    model_catalog_identity: object
    operator_version: str
    attempt_index: int
    tool_result_identities: tuple[object, ...] = ()
    prompt_identity: str | None = None
    proposal_schema_identity: str | None = None
    tool_schema_identity: str | None = None
    requested_model_id: str = FROZEN_POE_MODEL_ID

    def __post_init__(self) -> None:
        _require_digest(self.source_record_sha256, field_name="source_record_sha256")
        _single_line(self.operator_version, field_name="operator_version")
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative exact integer")
        if type(self.tool_result_identities) is not tuple:
            raise TypeError("tool_result_identities must be an exact tuple")
        for identity in self.tool_result_identities:
            _tool_result_identity(identity)
        for identity, field_name in (
            (self.prompt_identity, "prompt_identity"),
            (self.proposal_schema_identity, "proposal_schema_identity"),
            (self.tool_schema_identity, "tool_schema_identity"),
        ):
            if identity is not None:
                _require_digest(identity, field_name=field_name)
        if self.requested_model_id != FROZEN_POE_MODEL_ID:
            raise ValueError("Poe response cache is frozen to gpt-5.4-mini")
        _model_catalog_identity(self.model_catalog_identity)

    def key_material(
        self, request: object, *, config: object | None = None
    ) -> dict[str, Any]:
        request_value = _request_value(request)
        model_id = self.requested_model_id
        if config is not None:
            configured_model = getattr(config, "model_id", None)
            if configured_model != model_id:
                raise PoeResponseCacheError(
                    "CACHE_CONFIG_MISMATCH",
                    "PoeClientConfig model does not match the frozen cache context",
                )
        defaults = _local_contract_identities()
        return {
            "source_record_sha256": self.source_record_sha256,
            "canonical_request": request_value,
            "requested_model_id": model_id,
            "model_catalog_identity": _model_catalog_identity(
                self.model_catalog_identity
            ),
            "prompt_identity": self.prompt_identity or defaults[0],
            "proposal_schema_identity": self.proposal_schema_identity or defaults[1],
            "tool_schema_identity": self.tool_schema_identity or defaults[2],
            "tool_result_identities": [
                _tool_result_identity(item) for item in self.tool_result_identities
            ],
            "operator_version": self.operator_version,
            "attempt_index": self.attempt_index,
        }


@dataclass(frozen=True, slots=True)
class PoeCacheArtifact:
    """One validated immutable object from a cache namespace."""

    kind: PoeCacheKind
    cache_key: str
    content_sha256: str
    artifact_sha256: str | None
    path: Path
    cache_hit: bool
    _payload_json: bytes

    @property
    def content_identity(self) -> str:
        return self.artifact_sha256 or self.content_sha256

    @property
    def payload(self) -> object:
        return json.loads(self._payload_json)


@dataclass(frozen=True, slots=True)
class CachedPoeResult:
    """A decoded T034 result plus immutable cache provenance."""

    result: object
    cache_key: str
    content_sha256: str
    path: Path
    cache_hit: bool


def _single_line(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{field_name} must be non-empty trimmed single-line text")
    return value


def _require_digest(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA256 hex")
    return value


def _json_value(value: object) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        # json.dumps(..., allow_nan=False) performs the finite-value check.
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("cache JSON object keys must be strings")
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return _json_value(to_json_dict())
    raise TypeError(f"cache value is not closed JSON: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole canonical JSON representation used by this module."""

    normalized = _json_value(value)
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("cache content must contain finite JSON values") from error


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request_value(request: object) -> dict[str, Any]:
    value = _json_value(request)
    if not isinstance(value, dict):
        raise TypeError("Poe proposal request must serialize to a JSON object")
    return value


def _model_catalog_identity(value: object) -> str:
    direct = getattr(value, "entry_sha256", None)
    if direct is None:
        model_entry = getattr(value, "model_entry", None)
        direct = getattr(model_entry, "entry_sha256", None)
    if direct is None:
        entries = getattr(value, "entries", None)
        if isinstance(entries, (tuple, list)):
            matches = [
                entry
                for entry in entries
                if getattr(entry, "model_id", getattr(entry, "id", None))
                == FROZEN_POE_MODEL_ID
            ]
            if len(matches) == 1:
                direct = getattr(matches[0], "entry_sha256", None)
    if direct is None and isinstance(value, Mapping):
        direct = value.get("catalog_entry_sha256", value.get("entry_sha256"))
    if direct is not None:
        return _require_digest(direct, field_name="model_catalog_identity")
    if type(value) is str:
        return _require_digest(value, field_name="model_catalog_identity")
    require = getattr(value, "require", None)
    if callable(require):
        entry = require(FROZEN_POE_MODEL_ID)
        direct = getattr(entry, "entry_sha256", None)
        if direct is not None:
            return _require_digest(direct, field_name="model_catalog_identity")
    return _digest(value)


def _tool_result_identity(value: object) -> str:
    direct = getattr(value, "cache_key", None)
    if direct is None and isinstance(value, Mapping):
        direct = value.get("cache_key", value.get("content_sha256"))
    if direct is not None:
        return _require_digest(direct, field_name="tool_result_identity")
    if type(value) is str:
        return _require_digest(value, field_name="tool_result_identity")
    return _digest(value)


@lru_cache(maxsize=1)
def _local_contract_identities() -> tuple[str, str, str]:
    # Delayed imports keep this module independent of client construction and
    # avoid a response-cache/client import cycle.
    from .client import PROPOSAL_SYSTEM_PROMPT
    from .schemas import (
        CHEMISTRY_TOOL_ARGUMENT_MODELS,
        ProposalRequest,
        ProposalResponse,
    )

    prompt_identity = _digest(
        {"prompt_version": "proposal_v1", "system_prompt": PROPOSAL_SYSTEM_PROMPT}
    )
    schema_identity = _digest(
        {
            "proposal_request": ProposalRequest.model_json_schema(mode="validation"),
            "proposal_response": ProposalResponse.model_json_schema(mode="validation"),
        }
    )
    tool_schema_identity = _digest(
        {
            name: model.model_json_schema(mode="validation")
            for name, model in CHEMISTRY_TOOL_ARGUMENT_MODELS.items()
        }
    )
    return prompt_identity, schema_identity, tool_schema_identity


def _assert_secret_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise PoeResponseCacheError(
                    "CACHE_NON_JSON_CONTENT",
                    "cache mappings must use text keys",
                )
            if key.casefold() in _FORBIDDEN_KEYS:
                raise PoeResponseCacheError(
                    "CACHE_SECRET_MATERIAL_REJECTED",
                    "raw headers or credential-bearing fields cannot be cached",
                )
            _assert_secret_free(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_secret_free(item)
        return
    if type(value) is str:
        folded = value.lstrip().casefold()
        if folded.startswith(("bearer ", "basic ")):
            raise PoeResponseCacheError(
                "CACHE_SECRET_MATERIAL_REJECTED",
                "credential-bearing text cannot be cached",
            )


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cache clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    text = _single_line(value, field_name="created_at")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _cache_key(kind: PoeCacheKind, key_material: Mapping[str, object]) -> str:
    return _digest(
        {
            "cache_key_version": CACHE_KEY_VERSION,
            "kind": kind.value,
            "key_material": key_material,
        }
    )


def _result_payload(
    result: object,
    *,
    usage: object | None,
    raw_request: object | None,
    raw_response: object | None,
) -> dict[str, Any]:
    request = getattr(result, "request", None)
    response = getattr(result, "response", None)
    provenance = getattr(result, "provenance", None)
    if request is None or response is None or provenance is None:
        raise PoeResponseCacheError(
            "CACHE_PRODUCER_RESULT_INVALID",
            "producer result lacks the T034 request/response/provenance API",
        )
    request_value = _request_value(request)
    response_value = _json_value(response)
    provenance_value = _json_value(provenance)
    if not isinstance(response_value, dict) or not isinstance(provenance_value, dict):
        raise PoeResponseCacheError(
            "CACHE_PRODUCER_RESULT_INVALID",
            "producer response and provenance must serialize to JSON objects",
        )
    attempts = provenance_value.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise PoeResponseCacheError(
            "CACHE_PRODUCER_RESULT_INVALID",
            "producer provenance must contain transport attempts",
        )
    tool_transcript: list[object] = []
    response_ids: list[str] = []
    query_ids: list[str] = []
    x_request_ids: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise PoeResponseCacheError(
                "CACHE_PRODUCER_RESULT_INVALID",
                "producer attempt provenance is not a JSON object",
            )
        for source_name, destination in (
            ("response_ids", response_ids),
            ("query_ids", query_ids),
            ("x_request_ids", x_request_ids),
        ):
            values = attempt.get(source_name, [])
            if not isinstance(values, list) or not all(
                type(item) is str and item for item in values
            ):
                raise PoeResponseCacheError(
                    "CACHE_PRODUCER_RESULT_INVALID",
                    "producer provenance contains invalid response identifiers",
                )
            destination.extend(values)
        executions = attempt.get("tool_executions", [])
        if not isinstance(executions, list):
            raise PoeResponseCacheError(
                "CACHE_PRODUCER_RESULT_INVALID",
                "producer tool transcript is not a JSON array",
            )
        tool_transcript.extend(executions)
    request_id = request_value.get("request_id")
    provenance_request_id = provenance_value.get("request_id")
    if (
        type(request_id) is not str
        or not request_id
        or provenance_request_id != request_id
    ):
        raise PoeResponseCacheError(
            "CACHE_PRODUCER_RESULT_INVALID",
            "producer request and provenance identities differ",
        )
    selected_transport = provenance_value.get("selected_transport")
    if type(selected_transport) is not str or not selected_transport:
        raise PoeResponseCacheError(
            "CACHE_PRODUCER_RESULT_INVALID",
            "producer provenance lacks the selected transport",
        )
    usage_value: object = (
        {"status": "not_reported"} if usage is None else _json_value(usage)
    )
    if not isinstance(usage_value, dict):
        raise TypeError("usage must serialize to a JSON object")
    payload = {
        "result_schema_version": CLIENT_RESULT_SCHEMA_VERSION,
        "provider": "poe",
        "request": request_value,
        "response": response_value,
        "provenance": provenance_value,
        "request_id": request_id,
        "response_ids": response_ids,
        "query_ids": query_ids,
        "x_request_ids": x_request_ids,
        "transport": selected_transport,
        "tool_transcript": tool_transcript,
        "usage": usage_value,
        "raw_request": None if raw_request is None else _json_value(raw_request),
        "raw_response": None if raw_response is None else _json_value(raw_response),
    }
    _assert_secret_free(payload)
    return payload


def _exact_mapping(value: object, fields: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            f"cached {name} does not satisfy its frozen schema",
        )
    return value


def _decode_client_result(payload: object) -> object:
    fields = {
        "result_schema_version",
        "provider",
        "request",
        "response",
        "provenance",
        "request_id",
        "response_ids",
        "query_ids",
        "x_request_ids",
        "transport",
        "tool_transcript",
        "usage",
        "raw_request",
        "raw_response",
    }
    value = _exact_mapping(payload, fields, name="Poe client result")
    if (
        value["result_schema_version"] != CLIENT_RESULT_SCHEMA_VERSION
        or value["provider"] != "poe"
    ):
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            "cached Poe client result version or provider changed",
        )
    # Import only when a hit is decoded.  response_cache remains independent of
    # T034 module initialization and works with its frozen public dataclasses.
    from .client import (
        PoeAttemptStatus,
        PoeClientProvenance,
        PoeClientResult,
        PoeToolExecution,
        PoeTransport,
        PoeTransportAttempt,
    )
    from .schemas import ProposalRequest, ProposalResponse

    try:
        request = ProposalRequest.model_validate_json(
            canonical_json_bytes(value["request"]), strict=True
        )
        response = ProposalResponse.model_validate_json(
            canonical_json_bytes(value["response"]), strict=True
        )
        provenance_value = value["provenance"]
        if not isinstance(provenance_value, dict):
            raise TypeError("provenance must be an object")
        attempts_value = provenance_value.get("attempts")
        if not isinstance(attempts_value, list):
            raise TypeError("attempts must be an array")
        attempts = []
        for attempt_value in attempts_value:
            if not isinstance(attempt_value, dict):
                raise TypeError("attempt must be an object")
            executions_value = attempt_value.get("tool_executions")
            if not isinstance(executions_value, list):
                raise TypeError("tool_executions must be an array")
            executions = tuple(
                PoeToolExecution(
                    turn=item["turn"],
                    sequence=item["sequence"],
                    tool_call_id=item["tool_call_id"],
                    tool=item["tool"],
                    arguments_json=item["arguments_json"],
                    result_json=item["result_json"],
                )
                for item in executions_value
            )
            attempts.append(
                PoeTransportAttempt(
                    transport=PoeTransport(attempt_value["transport"]),
                    status=PoeAttemptStatus(attempt_value["status"]),
                    request_id=attempt_value["request_id"],
                    requested_model_id=attempt_value["requested_model_id"],
                    response_model=attempt_value["response_model"],
                    response_ids=tuple(attempt_value["response_ids"]),
                    query_ids=tuple(attempt_value["query_ids"]),
                    x_request_ids=tuple(attempt_value["x_request_ids"]),
                    turns=attempt_value["turns"],
                    tool_executions=executions,
                    error_code=attempt_value["error_code"],
                    error_detail=attempt_value["error_detail"],
                )
            )
        provenance = PoeClientProvenance(
            provider=provenance_value["provider"],
            request_id=provenance_value["request_id"],
            origin_id=provenance_value["origin_id"],
            operator_id=provenance_value["operator_id"],
            propagation=provenance_value["propagation"],
            candidate_source_mode=provenance_value["candidate_source_mode"],
            target_root=provenance_value["target_root"],
            constraints_json=provenance_value["constraints_json"],
            requested_model_id=provenance_value["requested_model_id"],
            selected_transport=PoeTransport(provenance_value["selected_transport"]),
            attempts=tuple(attempts),
        )
        result = PoeClientResult(
            request=request,
            response=response,
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            "cached Poe client result cannot be reconstructed",
        ) from error
    if (
        value["request_id"] != result.request.request_id
        or value["transport"] != result.provenance.selected_transport.value
    ):
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            "cached result identifiers disagree with reconstructed provenance",
        )
    try:
        normalized = _result_payload(
            result,
            usage=value["usage"],
            raw_request=value["raw_request"],
            raw_response=value["raw_response"],
        )
    except (PoeResponseCacheError, TypeError, ValueError) as error:
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            "cached Poe result metadata cannot be normalized",
        ) from error
    if normalized != value:
        raise PoeResponseCacheError(
            "CACHE_ENTRY_CORRUPT",
            "cached Poe identifiers or transcript disagree with provenance",
        )
    return result


class PoeResponseCache:
    """Filesystem-backed immutable cache for Poe proposals and build artifacts."""

    def __init__(
        self,
        root: Path | str,
        *,
        mode: PoeCacheMode | str = PoeCacheMode.READ_WRITE,
        release_mode: bool = False,
        clock: Callable[[], datetime | str] = _default_clock,
    ) -> None:
        self.root = Path(root)
        try:
            parsed_mode = mode if type(mode) is PoeCacheMode else PoeCacheMode(mode)
        except ValueError as error:
            raise ValueError("unsupported Poe cache mode") from error
        if type(release_mode) is not bool:
            raise TypeError("release_mode must be bool")
        self.mode = PoeCacheMode.FROZEN_REPLAY if release_mode else parsed_mode
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock

    @property
    def frozen(self) -> bool:
        return self.mode is PoeCacheMode.FROZEN_REPLAY

    def proposal_key(
        self,
        request: object,
        *,
        context: PoeCacheContext,
        config: object | None = None,
    ) -> str:
        if type(context) is not PoeCacheContext:
            raise TypeError("context must be PoeCacheContext")
        material = context.key_material(request, config=config)
        _assert_secret_free(material)
        return _cache_key(PoeCacheKind.PROPOSAL, material)

    def load_or_produce(
        self,
        request: object,
        *,
        context: PoeCacheContext,
        producer: PoeResultProducer | Callable[[object], object] | None = None,
        config: object | None = None,
        usage: object | None = None,
        raw_request: object | None = None,
        raw_response: object | None = None,
    ) -> CachedPoeResult:
        """Load a proposal or create it once through the injected producer."""

        if type(context) is not PoeCacheContext:
            raise TypeError("context must be PoeCacheContext")
        key_material = context.key_material(request, config=config)
        _assert_secret_free(key_material)
        if raw_request is not None:
            _assert_secret_free(_json_value(raw_request))
        if raw_response is not None:
            _assert_secret_free(_json_value(raw_response))
        if usage is not None:
            _assert_secret_free(_json_value(usage))
        key = _cache_key(PoeCacheKind.PROPOSAL, key_material)
        path = self._path(PoeCacheKind.PROPOSAL, key)
        if path.exists():
            artifact = self._load_artifact(
                PoeCacheKind.PROPOSAL,
                key,
                expected_key_material=key_material,
            )
            result = _decode_client_result(artifact.payload)
            if _request_value(result.request) != _request_value(request):
                raise PoeResponseCacheError(
                    "CACHE_ENTRY_MISMATCH",
                    "cached proposal is bound to another canonical request",
                    kind=PoeCacheKind.PROPOSAL,
                    cache_key=key,
                )
            return CachedPoeResult(
                result=result,
                cache_key=key,
                content_sha256=artifact.content_sha256,
                path=path,
                cache_hit=True,
            )
        if self.frozen:
            raise PoeResponseCacheError(
                "CACHE_MISS_FROZEN",
                "frozen replay requires an existing validated proposal entry",
                kind=PoeCacheKind.PROPOSAL,
                cache_key=key,
            )
        if producer is None:
            raise PoeResponseCacheError(
                "CACHE_PRODUCER_MISSING",
                "a read-write cache miss requires an injected producer",
                kind=PoeCacheKind.PROPOSAL,
                cache_key=key,
            )
        if isinstance(producer, PoeResultProducer):
            result = producer.propose(request)
        elif callable(producer):
            result = producer(request)
        else:
            raise TypeError("producer must be callable or implement propose(request)")
        payload = _result_payload(
            result,
            usage=usage,
            raw_request=raw_request,
            raw_response=raw_response,
        )
        if payload["request"] != _request_value(request):
            raise PoeResponseCacheError(
                "CACHE_PRODUCER_RESULT_INVALID",
                "producer returned a result for another canonical request",
                kind=PoeCacheKind.PROPOSAL,
                cache_key=key,
            )
        artifact = self._store(
            PoeCacheKind.PROPOSAL,
            key_material=key_material,
            payload=payload,
        )
        # A concurrent writer may have won.  Always decode the immutable entry
        # that is actually on disk before returning it.
        persisted_result = _decode_client_result(artifact.payload)
        return CachedPoeResult(
            result=persisted_result,
            cache_key=key,
            content_sha256=artifact.content_sha256,
            path=path,
            cache_hit=artifact.cache_hit,
        )

    def get_or_produce(self, request: object, **kwargs: object) -> object:
        """Compatibility convenience returning the T034 ``PoeClientResult``."""

        return self.load_or_produce(request, **kwargs).result

    def replay(
        self,
        request: object,
        *,
        context: PoeCacheContext,
        config: object | None = None,
    ) -> object:
        """Replay one validated entry without accepting a producer."""

        key_material = context.key_material(request, config=config)
        key = _cache_key(PoeCacheKind.PROPOSAL, key_material)
        artifact = self._load_artifact(
            PoeCacheKind.PROPOSAL,
            key,
            expected_key_material=key_material,
        )
        result = _decode_client_result(artifact.payload)
        if _request_value(result.request) != _request_value(request):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_MISMATCH",
                "cached proposal is bound to another canonical request",
                kind=PoeCacheKind.PROPOSAL,
                cache_key=key,
            )
        return result

    def store_tool_run(
        self,
        *,
        key_material: Mapping[str, object],
        result: object,
        provenance: Mapping[str, object] | None = None,
    ) -> PoeCacheArtifact:
        return self._store_named_payload(
            PoeCacheKind.TOOL_RUN,
            key_material=key_material,
            payload_name="result",
            payload=result,
            provenance=provenance,
        )

    def store_render(
        self,
        *,
        key_material: Mapping[str, object],
        render: object,
        provenance: Mapping[str, object] | None = None,
    ) -> PoeCacheArtifact:
        return self._store_named_payload(
            PoeCacheKind.RENDER,
            key_material=key_material,
            payload_name="render",
            payload=render,
            provenance=provenance,
        )

    def load_tool_run(self, *, key_material: Mapping[str, object]) -> object:
        """Load and validate one deterministic local tool result."""

        return self._load_named_payload(
            PoeCacheKind.TOOL_RUN,
            key_material=key_material,
            payload_name="result",
        )

    def load_render(self, *, key_material: Mapping[str, object]) -> object:
        """Load and validate one deterministic render."""

        return self._load_named_payload(
            PoeCacheKind.RENDER,
            key_material=key_material,
            payload_name="render",
        )

    def store_accepted_artifact(
        self,
        artifact: object,
        *,
        provenance: Mapping[str, object] | None = None,
        request_id: str | None = None,
        response_ids: Sequence[str] = (),
        transport: str | None = None,
        usage: object | None = None,
    ) -> PoeCacheArtifact:
        artifact_value = _json_value(artifact)
        artifact_sha256 = _digest(artifact_value)
        if request_id is not None:
            _single_line(request_id, field_name="request_id")
        response_id_values = list(response_ids)
        for response_id in response_id_values:
            _single_line(response_id, field_name="response_id")
        if transport is not None:
            _single_line(transport, field_name="transport")
        payload = {
            "artifact_schema_version": ACCEPTED_ARTIFACT_SCHEMA_VERSION,
            "artifact": artifact_value,
            "artifact_sha256": artifact_sha256,
            "provenance": _json_value(provenance or {}),
            "request_id": request_id,
            "response_ids": response_id_values,
            "transport": transport,
            "usage": _json_value(usage or {"status": "not_reported"}),
        }
        return self._store(
            PoeCacheKind.ACCEPTED,
            key_material={"artifact_sha256": artifact_sha256},
            payload=payload,
            cache_key_override=artifact_sha256,
            artifact_sha256=artifact_sha256,
        )

    def load_accepted_artifact(self, artifact_sha256: str) -> object:
        _require_digest(artifact_sha256, field_name="artifact_sha256")
        entry = self._load_artifact(PoeCacheKind.ACCEPTED, artifact_sha256)
        payload = entry.payload
        if not isinstance(payload, dict):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "accepted artifact payload is not a JSON object",
            )
        expected_fields = {
            "artifact_schema_version",
            "artifact",
            "artifact_sha256",
            "provenance",
            "request_id",
            "response_ids",
            "transport",
            "usage",
        }
        _exact_mapping(payload, expected_fields, name="accepted artifact")
        if (
            payload["artifact_schema_version"] != ACCEPTED_ARTIFACT_SCHEMA_VERSION
            or payload["artifact_sha256"] != artifact_sha256
            or _digest(payload["artifact"]) != artifact_sha256
        ):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_MISMATCH",
                "accepted artifact content identity does not match its path",
                kind=PoeCacheKind.ACCEPTED,
                cache_key=artifact_sha256,
            )
        return payload["artifact"]

    def _store_named_payload(
        self,
        kind: PoeCacheKind,
        *,
        key_material: Mapping[str, object],
        payload_name: str,
        payload: object,
        provenance: Mapping[str, object] | None,
    ) -> PoeCacheArtifact:
        return self._store(
            kind,
            key_material=key_material,
            payload={
                "payload_schema_version": "1.0",
                payload_name: _json_value(payload),
                "provenance": _json_value(provenance or {}),
            },
        )

    def _load_named_payload(
        self,
        kind: PoeCacheKind,
        *,
        key_material: Mapping[str, object],
        payload_name: str,
    ) -> object:
        material_value = _json_value(key_material)
        if not isinstance(material_value, dict):
            raise TypeError("key_material must serialize to a JSON object")
        key = _cache_key(kind, material_value)
        entry = self._load_artifact(
            kind,
            key,
            expected_key_material=material_value,
        )
        payload = entry.payload
        value = _exact_mapping(
            payload,
            {"payload_schema_version", payload_name, "provenance"},
            name=kind.value,
        )
        if value["payload_schema_version"] != "1.0":
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                f"cached {kind.value} payload version changed",
                kind=kind,
                cache_key=key,
            )
        return value[payload_name]

    def _path(self, kind: PoeCacheKind, key: str) -> Path:
        _require_digest(key, field_name="cache key")
        return self.root / kind.value / f"{key}.json"

    def _store(
        self,
        kind: PoeCacheKind,
        *,
        key_material: Mapping[str, object],
        payload: object,
        cache_key_override: str | None = None,
        artifact_sha256: str | None = None,
    ) -> PoeCacheArtifact:
        if self.frozen:
            raise PoeResponseCacheError(
                "CACHE_FROZEN_WRITE_REJECTED",
                "frozen replay cache cannot create or replace entries",
                kind=kind,
            )
        material_value = _json_value(key_material)
        if not isinstance(material_value, dict):
            raise TypeError("key_material must serialize to a JSON object")
        payload_value = _json_value(payload)
        _assert_secret_free(material_value)
        _assert_secret_free(payload_value)
        key = (
            _cache_key(kind, material_value)
            if cache_key_override is None
            else _require_digest(cache_key_override, field_name="cache_key_override")
        )
        if artifact_sha256 is not None:
            _require_digest(artifact_sha256, field_name="artifact_sha256")
        path = self._path(kind, key)
        if path.exists():
            existing = self._load_artifact(
                kind,
                key,
                expected_key_material=material_value,
            )
            if (
                existing.payload != payload_value
                or existing.artifact_sha256 != artifact_sha256
            ):
                raise PoeResponseCacheError(
                    "CACHE_IMMUTABILITY_VIOLATION",
                    "an immutable cache key already contains different content",
                    kind=kind,
                    cache_key=key,
                )
            return PoeCacheArtifact(
                kind=existing.kind,
                cache_key=existing.cache_key,
                content_sha256=existing.content_sha256,
                artifact_sha256=existing.artifact_sha256,
                path=existing.path,
                cache_hit=True,
                _payload_json=existing._payload_json,
            )
        created_at = _timestamp(self._clock())
        content_sha256 = _digest(payload_value)
        record: dict[str, object] = {
            "cache_schema": CACHE_SCHEMA,
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key_version": CACHE_KEY_VERSION,
            "kind": kind.value,
            "cache_key": key,
            "content_identity_algorithm": CONTENT_IDENTITY_ALGORITHM,
            "content_sha256": content_sha256,
            "artifact_sha256": artifact_sha256,
            "created_at": created_at,
            "key_material": material_value,
            "payload": payload_value,
        }
        record["record_sha256"] = _digest(record)
        _assert_secret_free(record)
        serialized = canonical_json_bytes(record) + b"\n"
        created = self._atomic_create(path, serialized)
        if not created:
            existing = self._load_artifact(
                kind,
                key,
                expected_key_material=material_value,
            )
            if (
                existing.payload != payload_value
                or existing.artifact_sha256 != artifact_sha256
            ):
                raise PoeResponseCacheError(
                    "CACHE_IMMUTABILITY_VIOLATION",
                    "a concurrent writer stored different immutable content",
                    kind=kind,
                    cache_key=key,
                )
            return PoeCacheArtifact(
                kind=existing.kind,
                cache_key=existing.cache_key,
                content_sha256=existing.content_sha256,
                artifact_sha256=existing.artifact_sha256,
                path=existing.path,
                cache_hit=True,
                _payload_json=existing._payload_json,
            )
        return PoeCacheArtifact(
            kind=kind,
            cache_key=key,
            content_sha256=content_sha256,
            artifact_sha256=artifact_sha256,
            path=path,
            cache_hit=False,
            _payload_json=canonical_json_bytes(payload_value),
        )

    def _load_artifact(
        self,
        kind: PoeCacheKind,
        key: str,
        *,
        expected_key_material: Mapping[str, object] | None = None,
    ) -> PoeCacheArtifact:
        path = self._path(kind, key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise PoeResponseCacheError(
                "CACHE_MISS_FROZEN",
                "required immutable cache entry is missing",
                kind=kind,
                cache_key=key,
            ) from error
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry is not valid UTF-8 JSON",
                kind=kind,
                cache_key=key,
            ) from error
        if not isinstance(decoded, dict) or set(decoded) != _RECORD_FIELDS:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry does not satisfy the frozen record schema",
                kind=kind,
                cache_key=key,
            )
        try:
            canonical = canonical_json_bytes(decoded) + b"\n"
        except (TypeError, ValueError) as error:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry contains non-canonical JSON values",
                kind=kind,
                cache_key=key,
            ) from error
        if raw != canonical:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry is not canonical JSON",
                kind=kind,
                cache_key=key,
            )
        try:
            stored_kind = PoeCacheKind(decoded["kind"])
            stored_key = _require_digest(decoded["cache_key"], field_name="cache_key")
            content_sha256 = _require_digest(
                decoded["content_sha256"], field_name="content_sha256"
            )
            record_sha256 = _require_digest(
                decoded["record_sha256"], field_name="record_sha256"
            )
            stored_artifact_sha256 = decoded["artifact_sha256"]
            if stored_artifact_sha256 is not None:
                _require_digest(
                    stored_artifact_sha256,
                    field_name="artifact_sha256",
                )
        except (TypeError, ValueError) as error:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry contains invalid identity fields",
                kind=kind,
                cache_key=key,
            ) from error
        if (
            decoded["cache_schema"] != CACHE_SCHEMA
            or decoded["schema_version"] != CACHE_SCHEMA_VERSION
            or decoded["cache_key_version"] != CACHE_KEY_VERSION
            or decoded["content_identity_algorithm"] != CONTENT_IDENTITY_ALGORITHM
            or stored_kind is not kind
            or stored_key != key
        ):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_MISMATCH",
                "cache entry schema, namespace, or path identity changed",
                kind=kind,
                cache_key=key,
            )
        try:
            _timestamp(decoded["created_at"])
        except (TypeError, ValueError) as error:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache entry contains an invalid creation timestamp",
                kind=kind,
                cache_key=key,
            ) from error
        key_material = decoded["key_material"]
        if not isinstance(key_material, dict):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache key material is not a JSON object",
                kind=kind,
                cache_key=key,
            )
        expected_key = (
            stored_artifact_sha256
            if kind is PoeCacheKind.ACCEPTED
            else _cache_key(kind, key_material)
        )
        if expected_key != key:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_MISMATCH",
                "cache key does not match canonical key material",
                kind=kind,
                cache_key=key,
            )
        if expected_key_material is not None and key_material != _json_value(
            expected_key_material
        ):
            raise PoeResponseCacheError(
                "CACHE_ENTRY_MISMATCH",
                "cache entry is bound to different key material",
                kind=kind,
                cache_key=key,
            )
        payload = decoded["payload"]
        if _digest(payload) != content_sha256:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache payload content identity does not match",
                kind=kind,
                cache_key=key,
            )
        record_without_identity = dict(decoded)
        del record_without_identity["record_sha256"]
        if _digest(record_without_identity) != record_sha256:
            raise PoeResponseCacheError(
                "CACHE_ENTRY_CORRUPT",
                "cache record identity does not match",
                kind=kind,
                cache_key=key,
            )
        if kind is PoeCacheKind.ACCEPTED:
            if not isinstance(payload, dict) or stored_artifact_sha256 is None:
                raise PoeResponseCacheError(
                    "CACHE_ENTRY_CORRUPT",
                    "accepted cache entry lacks its artifact identity",
                    kind=kind,
                    cache_key=key,
                )
            artifact = payload.get("artifact")
            if (
                payload.get("artifact_sha256") != stored_artifact_sha256
                or _digest(artifact) != stored_artifact_sha256
            ):
                raise PoeResponseCacheError(
                    "CACHE_ENTRY_MISMATCH",
                    "accepted artifact identity does not match cached content",
                    kind=kind,
                    cache_key=key,
                )
        _assert_secret_free(decoded)
        return PoeCacheArtifact(
            kind=kind,
            cache_key=key,
            content_sha256=content_sha256,
            artifact_sha256=stored_artifact_sha256,
            path=path,
            cache_hit=True,
            _payload_json=canonical_json_bytes(payload),
        )

    @staticmethod
    def _atomic_create(path: Path, payload: bytes) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                os.chmod(stream.name, 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                return False
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


# Concise public aliases for callers that use provider-neutral naming.
ResponseCache = PoeResponseCache
ResponseCacheError = PoeResponseCacheError
CacheMode = PoeCacheMode
CacheKind = PoeCacheKind
CacheContext = PoeCacheContext

__all__ = [
    "ACCEPTED_ARTIFACT_SCHEMA_VERSION",
    "CACHE_KEY_VERSION",
    "CACHE_SCHEMA",
    "CACHE_SCHEMA_VERSION",
    "CacheContext",
    "CacheKind",
    "CacheMode",
    "CachedPoeResult",
    "PoeCacheArtifact",
    "PoeCacheContext",
    "PoeCacheKind",
    "PoeCacheMode",
    "PoeResponseCache",
    "PoeResponseCacheError",
    "ResponseCache",
    "ResponseCacheError",
    "canonical_json_bytes",
]
