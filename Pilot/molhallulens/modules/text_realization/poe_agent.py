"""Poe agent boundary for context-aware natural-language step rewriting.

The agent never owns FORMAL text. It returns only natural-language templates whose
claim placeholders are validated locally before candidate DAG values are inserted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)
from molhallulens.config.paths import PROJECT_ROOT


POE_RENDERER_VERSION = "poe_step_text_v1"
_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}", flags=re.ASCII)
_RESERVED_STRUCTURE = re.compile(
    r"(?:^|\n)\s*(?:Step\s+\d+\s*\[|FORMAL:|Answer:)",
    flags=re.IGNORECASE,
)


class PoeTextRealizationError(RuntimeError):
    """Fail-closed Poe configuration, transport, or response error."""


@dataclass(frozen=True, slots=True)
class PoeStepRewriteInput:
    step_index: int
    step_name: str
    original_natural_language: str
    original_formal_ab: str
    modified_formal_ab: str
    natural_template_draft: str
    placeholder_values: Mapping[str, str]
    required_placeholder_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must be a positive integer")
        for value, name in (
            (self.step_name, "step_name"),
            (self.original_natural_language, "original_natural_language"),
            (self.original_formal_ab, "original_formal_ab"),
            (self.modified_formal_ab, "modified_formal_ab"),
            (self.natural_template_draft, "natural_template_draft"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        values = dict(self.placeholder_values)
        counts = dict(self.required_placeholder_counts)
        if set(values) != set(counts):
            raise ValueError("placeholder values and counts must use identical names")
        if any(
            type(name) is not str
            or _PLACEHOLDER.fullmatch("{{" + name + "}}") is None
            or type(value) is not str
            or not value
            for name, value in values.items()
        ):
            raise ValueError("placeholder_values contains an invalid name or value")
        if any(type(count) is not int or count < 1 for count in counts.values()):
            raise ValueError("required placeholder counts must be positive integers")
        validate_natural_template(self.natural_template_draft, counts)
        object.__setattr__(
            self,
            "placeholder_values",
            MappingProxyType(dict(sorted(values.items()))),
        )
        object.__setattr__(
            self,
            "required_placeholder_counts",
            MappingProxyType(dict(sorted(counts.items()))),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "original_natural_language": self.original_natural_language,
            "original_formal_ab": self.original_formal_ab,
            "modified_formal_ab": self.modified_formal_ab,
            "natural_template_draft": self.natural_template_draft,
            "placeholder_values": dict(self.placeholder_values),
            "required_placeholder_counts": dict(self.required_placeholder_counts),
        }


@dataclass(frozen=True, slots=True)
class PoeRewriteRequest:
    origin_id: str
    subtask: str
    indexed_smiles: str
    instruction: str
    steps: tuple[PoeStepRewriteInput, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.subtask, "subtask"),
            (self.indexed_smiles, "indexed_smiles"),
            (self.instruction, "instruction"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        steps = tuple(self.steps)
        if not steps or any(type(step) is not PoeStepRewriteInput for step in steps):
            raise ValueError("steps must contain PoeStepRewriteInput values")
        if tuple(step.step_index for step in steps) != tuple(range(1, len(steps) + 1)):
            raise ValueError("Poe rewrite steps must be consecutive and ordered")
        object.__setattr__(self, "steps", steps)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "subtask": self.subtask,
            "indexed_smiles": self.indexed_smiles,
            "instruction": self.instruction,
            "steps": [step.to_prompt_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class PoeRewriteResult:
    natural_templates: tuple[str, ...]
    bot_name: str
    api_key_environment_variable: str
    prompt_sha256: str
    response_sha256: str
    network_request_count: int
    cache_hit: bool


PoeTransport = Callable[[str, str, str, float], str]


def validate_natural_template(
    template: str,
    required_counts: Mapping[str, int],
) -> str:
    """Require the exact placeholder multiset and forbid structural injection."""

    if type(template) is not str or not template.strip():
        raise PoeTextRealizationError("natural template must be non-empty text")
    if template != template.strip() or "\r" in template or "\x00" in template:
        raise PoeTextRealizationError(
            "natural template must be trimmed and contain no CR/NUL characters"
        )
    if _RESERVED_STRUCTURE.search(template):
        raise PoeTextRealizationError(
            "agent output must contain natural-language body only, without Step/FORMAL/Answer"
        )
    actual = Counter(_PLACEHOLDER.findall(template))
    expected = Counter(dict(required_counts))
    if actual != expected:
        raise PoeTextRealizationError(
            "placeholder counts changed; "
            f"expected={dict(expected)}, actual={dict(actual)}"
        )
    without_placeholders = _PLACEHOLDER.sub("", template)
    if "{" in without_placeholders or "}" in without_placeholders:
        raise PoeTextRealizationError("agent output contains malformed braces")
    return template


def _system_prompt() -> str:
    return (
        "You rewrite molecule-editing chain-of-thought steps for dataset construction. "
        "The MODIFIED_FORMAL fields are authoritative claims, even when they are "
        "chemically false or mutually inconsistent. Rewrite only the natural-language "
        "body of every step so it reads fluently and agrees with those claims. Preserve "
        "the original level of detail and the reasoning flow across steps. Never correct "
        "a modified claim, never disclose that it was modified, and never output a Step "
        "header, FORMAL line, final answer, Markdown fence, or commentary. Variable claim "
        "values are locked placeholders such as {{anchor_idx}}. Preserve the exact "
        "placeholder multiset specified for each step. Return JSON only."
    )


def _user_prompt(request: PoeRewriteRequest) -> str:
    response_shape = {
        "steps": [
            {
                "step_index": step.step_index,
                "natural_language_template": "string",
            }
            for step in request.steps
        ]
    }
    return (
        "Rewrite this complete reasoning chain.\n"
        "RESPONSE_SHAPE:\n"
        + json.dumps(response_shape, ensure_ascii=False, separators=(",", ":"))
        + "\nINPUT:\n"
        + json.dumps(
            request.to_prompt_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _extract_json_object(text: str) -> Mapping[str, Any]:
    if type(text) is not str or not text.strip():
        raise PoeTextRealizationError("Poe returned empty text")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise PoeTextRealizationError("Poe response contains no JSON object")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as error:
        raise PoeTextRealizationError("Poe response is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise PoeTextRealizationError("Poe response root must be a JSON object")
    return value


def _parse_and_validate_response(
    response_text: str,
    request: PoeRewriteRequest,
) -> tuple[str, ...]:
    value = _extract_json_object(response_text)
    if set(value) != {"steps"} or not isinstance(value["steps"], list):
        raise PoeTextRealizationError("Poe response must contain exactly one steps array")
    rows = value["steps"]
    if len(rows) != len(request.steps):
        raise PoeTextRealizationError("Poe response step count does not match the request")
    templates = []
    for expected, row in zip(request.steps, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "step_index",
            "natural_language_template",
        }:
            raise PoeTextRealizationError("each Poe step has an invalid JSON shape")
        if row["step_index"] != expected.step_index:
            raise PoeTextRealizationError("Poe response changed the step order")
        template = row["natural_language_template"]
        templates.append(
            validate_natural_template(template, expected.required_placeholder_counts)
        )
    return tuple(templates)


def _default_poe_transport(
    system_prompt: str,
    user_prompt: str,
    bot_name: str,
    temperature: float,
    *,
    api_key: str,
) -> str:
    try:
        import fastapi_poe as fp
    except ImportError as error:
        raise PoeTextRealizationError(
            "fastapi-poe is not installed; run: python -m pip install -r requirements.lock"
        ) from error
    messages = [
        fp.ProtocolMessage(role="system", content=system_prompt),
        fp.ProtocolMessage(role="user", content=user_prompt),
    ]
    parts = []
    try:
        for chunk in fp.get_bot_response_sync(
            messages=messages,
            bot_name=bot_name,
            api_key=api_key,
            temperature=temperature,
        ):
            text = getattr(chunk, "text", None)
            if type(text) is str:
                parts.append(text)
    except Exception as error:  # The SDK exposes several transport-specific errors.
        raise PoeTextRealizationError(
            f"Poe request failed ({type(error).__name__}); no API token was logged"
        ) from None
    response = "".join(parts)
    if not response:
        raise PoeTextRealizationError("Poe returned no text")
    return response


class PoeStepTextAgent:
    """Call Poe, validate its placeholder JSON, and cache only secret-free output."""

    __slots__ = ("config", "transport", "_environment", "cache_directory")

    def __init__(
        self,
        config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
        *,
        transport: PoeTransport | None = None,
        environment: Mapping[str, str] | None = None,
        cache_directory: Path | None = None,
    ) -> None:
        if type(config) is not HallucinationGenerationConfig:
            raise TypeError("config must be HallucinationGenerationConfig")
        if transport is not None and not callable(transport):
            raise TypeError("transport must be callable or None")
        if environment is not None and not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping or None")
        self.config = config
        self.transport = transport
        # Keep the environment behind a private boundary.  It is consulted only at
        # request time and is never serialized into provenance, cache, or output.
        self._environment = os.environ if environment is None else environment
        configured_cache = Path(config.poe_cache_directory)
        self.cache_directory = (
            Path(cache_directory)
            if cache_directory is not None
            else (
                configured_cache
                if configured_cache.is_absolute()
                else PROJECT_ROOT / configured_cache
            )
        )

    def _api_key(self) -> str:
        value = self._environment.get(self.config.poe_api_key_env)
        if type(value) is not str or not value.strip():
            raise PoeTextRealizationError(
                f"missing {self.config.poe_api_key_env}; in your terminal run: "
                f"export {self.config.poe_api_key_env}='YOUR_POE_API_KEY'"
            )
        return value.strip()

    def _cache_path(self, prompt_sha256: str) -> Path:
        return self.cache_directory / self.config.poe_bot_name / f"{prompt_sha256}.json"

    def _load_cache(
        self,
        prompt_sha256: str,
        request: PoeRewriteRequest,
    ) -> tuple[str, tuple[str, ...]] | None:
        path = self._cache_path(prompt_sha256)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, Mapping)
                or payload.get("renderer_version") != POE_RENDERER_VERSION
                or payload.get("bot_name") != self.config.poe_bot_name
                or payload.get("prompt_sha256") != prompt_sha256
                or type(payload.get("response_text")) is not str
            ):
                raise PoeTextRealizationError("cached Poe response metadata is invalid")
            response_text = payload["response_text"]
            templates = _parse_and_validate_response(response_text, request)
        except (OSError, json.JSONDecodeError) as error:
            raise PoeTextRealizationError("cached Poe response cannot be read") from error
        return response_text, templates

    def _store_cache(self, prompt_sha256: str, response_text: str) -> None:
        path = self._cache_path(prompt_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "renderer_version": POE_RENDERER_VERSION,
            "bot_name": self.config.poe_bot_name,
            "prompt_sha256": prompt_sha256,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "response_text": response_text,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def rewrite(self, request: PoeRewriteRequest) -> PoeRewriteResult:
        if type(request) is not PoeRewriteRequest:
            raise TypeError("request must be PoeRewriteRequest")
        system_prompt = _system_prompt()
        base_user_prompt = _user_prompt(request)
        prompt_identity = system_prompt + "\n\n" + base_user_prompt
        prompt_sha256 = hashlib.sha256(prompt_identity.encode("utf-8")).hexdigest()

        cached = self._load_cache(prompt_sha256, request)
        if cached is not None:
            response_text, templates = cached
            return PoeRewriteResult(
                natural_templates=templates,
                bot_name=self.config.poe_bot_name,
                api_key_environment_variable=self.config.poe_api_key_env,
                prompt_sha256=prompt_sha256,
                response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                network_request_count=0,
                cache_hit=True,
            )

        validation_error = ""
        response_text = ""
        for attempt in range(1, self.config.poe_max_attempts + 1):
            user_prompt = base_user_prompt
            if validation_error:
                user_prompt += (
                    "\nPREVIOUS_RESPONSE_REJECTED:\n"
                    + validation_error
                    + "\nReturn a corrected JSON object."
                )
            if self.transport is None:
                response_text = _default_poe_transport(
                    system_prompt,
                    user_prompt,
                    self.config.poe_bot_name,
                    self.config.poe_temperature,
                    api_key=self._api_key(),
                )
            else:
                response_text = self.transport(
                    system_prompt,
                    user_prompt,
                    self.config.poe_bot_name,
                    self.config.poe_temperature,
                )
            try:
                templates = _parse_and_validate_response(response_text, request)
            except PoeTextRealizationError as error:
                validation_error = str(error)
                continue
            self._store_cache(prompt_sha256, response_text)
            return PoeRewriteResult(
                natural_templates=templates,
                bot_name=self.config.poe_bot_name,
                api_key_environment_variable=self.config.poe_api_key_env,
                prompt_sha256=prompt_sha256,
                response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                network_request_count=attempt,
                cache_hit=False,
            )
        raise PoeTextRealizationError(
            "Poe response failed the local step_text contract after "
            f"{self.config.poe_max_attempts} attempts: {validation_error}"
        )


__all__ = [
    "POE_RENDERER_VERSION",
    "PoeRewriteRequest",
    "PoeRewriteResult",
    "PoeStepRewriteInput",
    "PoeStepTextAgent",
    "PoeTextRealizationError",
    "PoeTransport",
    "validate_natural_template",
]
