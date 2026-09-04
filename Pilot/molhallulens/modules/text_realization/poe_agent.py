"""Poe boundary for minimal, marker-tracked edits of original ``step_text``.

Poe receives each original complete step and its locally rendered modified FORMAL.
It returns a minimally edited complete step. Temporary HALLU markers identify only
the changed natural-language claim values; local code validates and removes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)
from molhallulens.config.paths import PROJECT_ROOT


POE_RENDERER_VERSION = "poe_step_text_v3"
FORMAL_MARKER = "\n  FORMAL: "
HALLU_MARKER_PATTERN = re.compile(
    r"\[\[HALLU:([a-z][a-z0-9_]*\.[0-9]{2})\]\](.*?)\[\[/HALLU\]\]",
    flags=re.ASCII | re.DOTALL,
)
_NODE_ID = re.compile(r"[a-z][a-z0-9_]*", flags=re.ASCII)
_MARKER_TOKEN = re.compile(r"\[\[(?:HALLU:[^\]]*|/HALLU)\]\]", flags=re.ASCII)


class PoeTextRealizationError(RuntimeError):
    """Fail-closed Poe configuration, transport, or response error."""


def _split_complete_step(step_text: str) -> tuple[str, str]:
    if step_text.count(FORMAL_MARKER) != 1:
        raise PoeTextRealizationError(
            "rewritten step_text must contain exactly one canonical FORMAL boundary"
        )
    head, formal = step_text.split(FORMAL_MARKER, 1)
    return head, formal


@dataclass(frozen=True, slots=True)
class RequiredHallucinationOccurrence:
    """One exact original mention that Poe must replace and mark once."""

    occurrence_id: str
    node_id: str
    before_text: str
    after_text: str
    original_start: int
    original_end: int

    def __post_init__(self) -> None:
        if (
            type(self.occurrence_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]*\.[0-9]{2}", self.occurrence_id) is None
        ):
            raise ValueError("occurrence_id must be node_id plus a two-digit suffix")
        if type(self.node_id) is not str or _NODE_ID.fullmatch(self.node_id) is None:
            raise ValueError("node_id is invalid")
        if not self.occurrence_id.startswith(self.node_id + "."):
            raise ValueError("occurrence_id must be namespaced by node_id")
        for value, name in (
            (self.before_text, "before_text"),
            (self.after_text, "after_text"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
            if _MARKER_TOKEN.search(value) is not None:
                raise ValueError(f"{name} cannot contain marker tokens")
        if type(self.original_start) is not int or type(self.original_end) is not int:
            raise TypeError("original occurrence offsets must be integers")
        if (
            self.original_start < 0
            or self.original_end <= self.original_start
            or self.original_end - self.original_start != len(self.before_text)
        ):
            raise ValueError("original occurrence offsets must exactly cover before_text")

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "occurrence_id": self.occurrence_id,
            "node_id": self.node_id,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "original_span": [self.original_start, self.original_end],
        }


@dataclass(frozen=True, slots=True)
class PoeStepRewriteInput:
    """One original step plus the exact FORMAL the rewritten step must express."""

    step_index: int
    step_name: str
    original_step_text: str
    modified_formal_ab: str
    required_hallucination_occurrences: tuple[RequiredHallucinationOccurrence, ...]

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 1:
            raise ValueError("step_index must be a positive integer")
        for value, name in (
            (self.step_name, "step_name"),
            (self.original_step_text, "original_step_text"),
            (self.modified_formal_ab, "modified_formal_ab"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        prefix = f"Step {self.step_index} [{self.step_name}]: "
        if not self.original_step_text.startswith(prefix):
            raise ValueError("original_step_text has an unexpected step header")
        if self.original_step_text != self.original_step_text.strip():
            raise ValueError("original_step_text must be trimmed")
        if _MARKER_TOKEN.search(self.original_step_text) is not None:
            raise ValueError("original_step_text must not contain HALLU markers")
        if any(character in self.modified_formal_ab for character in ("\r", "\n", "\x00")):
            raise ValueError("modified_formal_ab must be a single safe line")
        _split_complete_step(self.original_step_text)

        required = tuple(self.required_hallucination_occurrences)
        if any(type(item) is not RequiredHallucinationOccurrence for item in required):
            raise TypeError(
                "required_hallucination_occurrences must contain occurrence values"
            )
        occurrence_ids = tuple(item.occurrence_id for item in required)
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("required hallucination occurrence IDs must be unique")
        original_head, _ = _split_complete_step(self.original_step_text)
        original_natural = original_head[len(prefix) :]
        ordered_by_position = sorted(required, key=lambda item: item.original_start)
        previous_end = 0
        for item in ordered_by_position:
            if item.original_start < previous_end:
                raise ValueError("required hallucination occurrences cannot overlap")
            if original_natural[item.original_start : item.original_end] != item.before_text:
                raise ValueError(
                    "required hallucination occurrence does not match original_step_text"
                )
            previous_end = item.original_end
        object.__setattr__(
            self,
            "required_hallucination_occurrences",
            tuple(sorted(required, key=lambda item: item.occurrence_id)),
        )

    @property
    def original_formal_ab(self) -> str:
        return _split_complete_step(self.original_step_text)[1]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "original_step_text": self.original_step_text,
            "modified_formal_ab": self.modified_formal_ab,
            "required_hallucination_occurrences": [
                item.to_prompt_dict()
                for item in self.required_hallucination_occurrences
            ],
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
    rewritten_step_texts: tuple[str, ...]
    bot_name: str
    api_key_environment_variable: str
    prompt_sha256: str
    response_sha256: str
    network_request_count: int
    cache_hit: bool


PoeTransport = Callable[[str, str, str, float], str]


def parse_hallucination_markers(
    marked_natural_language: str,
) -> tuple[tuple[str, str, int, int], ...]:
    """Return ``(occurrence_id, value, start, end)`` after marker removal."""

    if type(marked_natural_language) is not str:
        raise TypeError("marked_natural_language must be text")
    results = []
    clean_length = 0
    cursor = 0
    for match in HALLU_MARKER_PATTERN.finditer(marked_natural_language):
        prefix = marked_natural_language[cursor : match.start()]
        if _MARKER_TOKEN.search(prefix) is not None:
            raise PoeTextRealizationError("natural language contains malformed HALLU markers")
        value = match.group(2)
        if not value or _MARKER_TOKEN.search(value) is not None:
            raise PoeTextRealizationError("HALLU marker value is empty or nested")
        clean_length += len(prefix)
        start = clean_length
        end = start + len(value)
        results.append((match.group(1), value, start, end))
        clean_length = end
        cursor = match.end()
    if _MARKER_TOKEN.search(marked_natural_language[cursor:]) is not None:
        raise PoeTextRealizationError("natural language contains malformed HALLU markers")
    return tuple(results)


def strip_hallucination_markers(marked_natural_language: str) -> str:
    """Remove validated temporary markers while preserving their inner values."""

    parse_hallucination_markers(marked_natural_language)
    return HALLU_MARKER_PATTERN.sub(lambda match: match.group(2), marked_natural_language)


def validate_rewritten_step_text(
    rewritten_step_text: str,
    expected: PoeStepRewriteInput,
) -> str:
    """Validate structure, exact FORMAL, and the marker-to-mutation contract."""

    if type(expected) is not PoeStepRewriteInput:
        raise TypeError("expected must be PoeStepRewriteInput")
    if type(rewritten_step_text) is not str or not rewritten_step_text.strip():
        raise PoeTextRealizationError("rewritten step_text must be non-empty text")
    if (
        rewritten_step_text != rewritten_step_text.strip()
        or "\r" in rewritten_step_text
        or "\x00" in rewritten_step_text
    ):
        raise PoeTextRealizationError(
            "rewritten step_text must be trimmed and contain no CR/NUL characters"
        )
    prefix = f"Step {expected.step_index} [{expected.step_name}]: "
    if not rewritten_step_text.startswith(prefix):
        raise PoeTextRealizationError("Poe response changed the exact Step header")
    marked_head, formal_ab = _split_complete_step(rewritten_step_text)
    if formal_ab != expected.modified_formal_ab:
        raise PoeTextRealizationError("Poe response did not preserve modified_formal_ab exactly")
    if "\n\nAnswer:" in marked_head or marked_head.startswith("Answer:"):
        raise PoeTextRealizationError("Poe response must not contain a final answer")

    marked_natural = marked_head[len(prefix) :]
    markers = parse_hallucination_markers(marked_natural)
    required = {
        item.occurrence_id: item
        for item in expected.required_hallucination_occurrences
    }
    observed: set[str] = set()
    for occurrence_id, value, _, _ in markers:
        if occurrence_id not in required:
            raise PoeTextRealizationError(
                "Poe marked an unplanned natural-language occurrence: "
                f"{occurrence_id}"
            )
        if occurrence_id in observed:
            raise PoeTextRealizationError(
                f"Poe duplicated HALLU marker occurrence: {occurrence_id}"
            )
        if value != required[occurrence_id].after_text:
            raise PoeTextRealizationError(
                f"Poe marker for {occurrence_id} does not contain its exact modified value"
            )
        observed.add(occurrence_id)
    missing = sorted(set(required) - observed)
    if missing:
        raise PoeTextRealizationError(
            f"Poe omitted required natural-language HALLU occurrences: {missing}"
        )

    # If FORMAL is the only changed channel, the natural-language head remains
    # byte-identical. This closes the product-only hole where Poe could previously
    # paraphrase an unlabelled step merely because modified_formal_ab had changed.
    if not required:
        original_head, _ = _split_complete_step(expected.original_step_text)
        if marked_head != original_head:
            raise PoeTextRealizationError(
                "Poe changed natural language without any required occurrences"
            )
    return rewritten_step_text


def _system_prompt() -> str:
    return (
        "You minimally edit existing molecule-editing step_text records. For each step, "
        "ORIGINAL_STEP_TEXT is the text to preserve and MODIFIED_FORMAL_AB is the new "
        "authoritative claim, even when chemically false or internally inconsistent. "
        "Change only the natural-language phrases necessary to agree with the modified "
        "FORMAL; retain the original wording, detail, order, Step header, and formatting "
        "as much as possible. Copy an unaffected step byte-for-byte. Return the complete "
        "rewritten step_text and include exactly one FORMAL line whose content is exactly "
        "MODIFIED_FORMAL_AB. Never add an Answer or commentary. In natural language, wrap "
        "each REQUIRED_HALLUCINATION_OCCURRENCE exactly once with its occurrence-specific "
        "temporary marker: [[HALLU:occurrence_id]]after_text[[/HALLU]]. Never mark an "
        "unplanned occurrence, never omit or duplicate an occurrence, never alter a "
        "marker value, never expose that the claim was modified, and never put markers in "
        "FORMAL. Return JSON only."
    )


def _user_prompt(request: PoeRewriteRequest) -> str:
    response_shape = {
        "steps": [
            {
                "step_index": step.step_index,
                "rewritten_step_text": "string",
            }
            for step in request.steps
        ]
    }
    return (
        "Minimally update these original step_text records from their modified FORMAL.\n"
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
    rewritten = []
    for expected, row in zip(request.steps, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "step_index",
            "rewritten_step_text",
        }:
            raise PoeTextRealizationError("each Poe step has an invalid JSON shape")
        if row["step_index"] != expected.step_index:
            raise PoeTextRealizationError("Poe response changed the step order")
        rewritten.append(
            validate_rewritten_step_text(row["rewritten_step_text"], expected)
        )
    return tuple(rewritten)


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
    """Call Poe, validate its marked step_text JSON, and cache secret-free output."""

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
            rewritten = _parse_and_validate_response(response_text, request)
        except (OSError, json.JSONDecodeError) as error:
            raise PoeTextRealizationError("cached Poe response cannot be read") from error
        return response_text, rewritten

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
            response_text, rewritten = cached
            return PoeRewriteResult(
                rewritten_step_texts=rewritten,
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
                rewritten = _parse_and_validate_response(response_text, request)
            except PoeTextRealizationError as error:
                validation_error = str(error)
                continue
            self._store_cache(prompt_sha256, response_text)
            return PoeRewriteResult(
                rewritten_step_texts=rewritten,
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
    "FORMAL_MARKER",
    "HALLU_MARKER_PATTERN",
    "POE_RENDERER_VERSION",
    "PoeRewriteRequest",
    "PoeRewriteResult",
    "RequiredHallucinationOccurrence",
    "PoeStepRewriteInput",
    "PoeStepTextAgent",
    "PoeTextRealizationError",
    "PoeTransport",
    "parse_hallucination_markers",
    "strip_hallucination_markers",
    "validate_rewritten_step_text",
]
