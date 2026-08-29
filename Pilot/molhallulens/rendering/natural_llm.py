"""Poe placeholder prose rewriting with a local locked-fact trust boundary.

The provider sees only one step name, locked slot *values*, a style identifier,
and a style excerpt.  It returns prose containing double-brace placeholders.
Local code rejects every output that changes the frozen placeholder allowlist,
introduces an unlocked number or structural header, or leaks annotation terms;
only then are the original immutable :class:`LiteralNode` objects inserted.

All provider and cache dependencies are injected.  Frozen replay performs a
cache read first and never invokes the producer when the entry is missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from molhallulens.domain import FrozenMap
from molhallulens.providers.poe.response_cache import (
    PoeResponseCache,
    PoeResponseCacheError,
)
from molhallulens.rendering.natural_rule import (
    NaturalRenderError,
    contains_unlocked_number,
    scan_label_leakage,
)
from molhallulens.rendering.trace_ast import ClaimNode, LiteralNode

LLM_NATURAL_RENDERER_VERSION = "natural_llm_v1"
POE_RENDER_MODEL_ID = "gpt-5.4-mini"
PLACEHOLDER_TEMPLATE_MODE = "double_brace_placeholders"

_PLACEHOLDER_NAME = re.compile(r"[a-z][a-z0-9_]*", flags=re.ASCII)
_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}", flags=re.ASCII)
_STYLE_ID = re.compile(r"[a-z][a-z0-9_.-]*", flags=re.ASCII)
_STEP_NAME = re.compile(r"[A-Z][A-Z_]*", flags=re.ASCII)
_SEMANTIC_DRIFT = re.compile(
    r"\b(?:approximately|maybe|not|perhaps|possibly|rather|unlikely|wrong)\b|"
    r"\binstead\s+of\b",
    flags=re.ASCII | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LockedSlotSnapshot:
    """Local-only immutable identity for one allowed placeholder."""

    placeholder: str
    mention_id: str
    state_or_edge_id: str
    target_kind: str
    value: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.placeholder, "placeholder"),
            (self.mention_id, "mention_id"),
            (self.state_or_edge_id, "state_or_edge_id"),
            (self.target_kind, "target_kind"),
            (self.value, "value"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"LockedSlotSnapshot {name} must be non-empty text")

    def to_dict(self) -> dict[str, str]:
        return {
            "placeholder": self.placeholder,
            "mention_id": self.mention_id,
            "state_or_edge_id": self.state_or_edge_id,
            "target_kind": self.target_kind,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class NarrativeRewriteRequest:
    """Exact label-blind provider request for one editable narrative claim."""

    claim_id: str
    step_name: str
    locked_slots: Mapping[str, LiteralNode]
    style_id: str
    original_style_excerpt: str
    renderer_version: str = LLM_NATURAL_RENDERER_VERSION

    def __post_init__(self) -> None:
        if type(self.claim_id) is not str or not self.claim_id:
            raise ValueError("claim_id must be non-empty text")
        if any(character.isspace() for character in self.claim_id):
            raise ValueError("claim_id cannot contain whitespace")
        if (
            type(self.step_name) is not str
            or _STEP_NAME.fullmatch(self.step_name) is None
        ):
            raise ValueError("step_name must be uppercase snake case")
        if type(self.style_id) is not str or _STYLE_ID.fullmatch(self.style_id) is None:
            raise ValueError("style_id must be a stable lowercase identifier")
        if (
            type(self.original_style_excerpt) is not str
            or not self.original_style_excerpt.strip()
        ):
            raise ValueError("original_style_excerpt must be non-empty text")
        if "\r" in self.original_style_excerpt or "\x00" in self.original_style_excerpt:
            raise ValueError("original_style_excerpt must be canonical NUL-free text")
        excerpt_leakage = scan_label_leakage(self.original_style_excerpt)
        if excerpt_leakage:
            raise NaturalRenderError(
                "STYLE_EXCERPT_LEAKAGE",
                f"style excerpt contains forbidden phrases: {excerpt_leakage!r}",
            )
        if not isinstance(self.locked_slots, Mapping):
            raise TypeError("locked_slots must be a mapping")
        normalized: dict[str, LiteralNode] = {}
        for placeholder, literal in self.locked_slots.items():
            if (
                type(placeholder) is not str
                or _PLACEHOLDER_NAME.fullmatch(placeholder) is None
            ):
                raise ValueError("locked slot names must be lowercase identifiers")
            if type(literal) is not LiteralNode:
                raise TypeError("locked_slots must contain LiteralNode values")
            normalized[placeholder] = literal
        if not normalized:
            raise ValueError("locked_slots cannot be empty")
        mention_ids = tuple(literal.mention_id for literal in normalized.values())
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("locked slots must have unique occurrence identities")
        if self.renderer_version != LLM_NATURAL_RENDERER_VERSION:
            raise ValueError("unsupported LLM natural renderer version")
        object.__setattr__(
            self,
            "locked_slots",
            FrozenMap(dict(sorted(normalized.items()))),
        )

    @property
    def allowed_placeholders(self) -> tuple[str, ...]:
        return tuple(self.locked_slots)

    @property
    def locked_snapshot(self) -> tuple[LockedSlotSnapshot, ...]:
        return tuple(
            LockedSlotSnapshot(
                placeholder=placeholder,
                mention_id=literal.mention_id,
                state_or_edge_id=literal.state_or_edge_id,
                target_kind=literal.target_kind.value,
                value=literal.value,
            )
            for placeholder, literal in self.locked_slots.items()
        )

    def provider_payload(self) -> dict[str, Any]:
        """Return the only fields visible to an injected Poe producer."""

        return {
            "step_name": self.step_name,
            "locked_slots": {
                placeholder: literal.value
                for placeholder, literal in self.locked_slots.items()
            },
            "style_id": self.style_id,
            "original_style_excerpt": self.original_style_excerpt,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the local cache identity, including occurrence identities."""

        return {
            "renderer_version": self.renderer_version,
            "claim_id": self.claim_id,
            "step_name": self.step_name,
            "locked_slots": [snapshot.to_dict() for snapshot in self.locked_snapshot],
            "style_id": self.style_id,
            "original_style_excerpt": self.original_style_excerpt,
        }


@runtime_checkable
class PoeNarrativeProducer(Protocol):
    """T034-shaped injected producer; tests and adapters implement ``propose``."""

    def propose(self, request: Mapping[str, Any]) -> object: ...


@dataclass(frozen=True, slots=True)
class NarrativeRewriteResult:
    """Locally validated template and reconstructed immutable AST claim."""

    request: NarrativeRewriteRequest
    template: str
    claim: ClaimNode
    cache_hit: bool

    def __post_init__(self) -> None:
        if type(self.request) is not NarrativeRewriteRequest:
            raise TypeError("request must be a NarrativeRewriteRequest")
        if type(self.template) is not str or not self.template:
            raise ValueError("template must be non-empty text")
        if type(self.claim) is not ClaimNode:
            raise TypeError("claim must be a ClaimNode")
        if type(self.cache_hit) is not bool:
            raise TypeError("cache_hit must be bool")
        local_template = validate_narrative_template(self.request, self.template)
        expected_claim = ClaimNode.from_template(
            self.request.claim_id,
            local_template,
            self.request.locked_slots,
        )
        if self.claim != expected_claim:
            raise NaturalRenderError(
                "LOCKED_IDENTITY_CHANGED",
                "rewrite result differs from locally reconstructed locked claim",
            )


def _template_parts(template: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(template) is not str or not template.strip():
        raise NaturalRenderError("EMPTY_TEMPLATE", "renderer template cannot be empty")
    if template != template.strip():
        raise NaturalRenderError(
            "TEMPLATE_WHITESPACE",
            "renderer template cannot have leading or trailing whitespace",
        )
    if "\r" in template or "\n" in template or "\x00" in template:
        raise NaturalRenderError(
            "TEMPLATE_MULTILINE",
            "renderer narrative template must be one canonical line",
        )
    placeholders = tuple(match.group(1) for match in _PLACEHOLDER.finditer(template))
    literal_parts = tuple(_PLACEHOLDER.split(template)[::2])
    without_placeholders = _PLACEHOLDER.sub("", template)
    if "{" in without_placeholders or "}" in without_placeholders:
        raise NaturalRenderError(
            "PLACEHOLDER_SYNTAX",
            "renderer template contains malformed or non-allowlisted braces",
        )
    return placeholders, literal_parts


def validate_narrative_template(
    request: NarrativeRewriteRequest,
    template: str,
) -> str:
    """Validate a provider template and return its local single-brace form."""

    if type(request) is not NarrativeRewriteRequest:
        raise TypeError("request must be a NarrativeRewriteRequest")
    placeholders, literal_parts = _template_parts(template)
    allowed = request.allowed_placeholders
    if len(placeholders) != len(set(placeholders)):
        raise NaturalRenderError(
            "PLACEHOLDER_REPEATED",
            "each locked placeholder must occur exactly once",
        )
    missing = tuple(sorted(set(allowed) - set(placeholders)))
    unknown = tuple(sorted(set(placeholders) - set(allowed)))
    if missing:
        raise NaturalRenderError(
            "LOCKED_FACT_OMITTED",
            f"renderer template omitted locked placeholders: {missing!r}",
        )
    if unknown:
        raise NaturalRenderError(
            "UNKNOWN_PLACEHOLDER",
            f"renderer template added unknown placeholders: {unknown!r}",
        )
    if any(contains_unlocked_number(part) for part in literal_parts):
        raise NaturalRenderError(
            "UNLOCKED_NUMBER",
            "renderer template contains a number outside a locked placeholder",
        )
    literal_text = "".join(literal_parts)
    folded = literal_text.casefold()
    if any(marker in folded for marker in ("formal:", "answer:", "<final_answer>")):
        raise NaturalRenderError(
            "RESERVED_STRUCTURE",
            "renderer template cannot inject FORMAL or Answer structure",
        )
    leaked = scan_label_leakage(literal_text)
    if leaked:
        raise NaturalRenderError(
            "LABEL_LEAKAGE",
            f"renderer template contains forbidden phrases: {leaked!r}",
        )
    if _SEMANTIC_DRIFT.search(literal_text) is not None:
        raise NaturalRenderError(
            "LOCKED_FACT_SEMANTIC_DRIFT",
            "renderer template negates or qualifies a locked factual claim",
        )

    return _PLACEHOLDER.sub(lambda match: "{" + match.group(1) + "}", template)


def _response_template(response: object) -> str:
    """Extract only a strict template response from an injected producer."""

    if type(response) is str:
        return response
    if isinstance(response, Mapping):
        if set(response) != {"template"} or type(response["template"]) is not str:
            raise NaturalRenderError(
                "RENDER_RESPONSE_SCHEMA",
                "renderer response must contain exactly one text template field",
            )
        return response["template"]
    raise NaturalRenderError(
        "RENDER_RESPONSE_SCHEMA",
        "renderer producer must return text or an exact one-field template mapping",
    )


def _cache_material(request: NarrativeRewriteRequest) -> dict[str, Any]:
    return {
        "kind": "natural_narrative_rewrite",
        "renderer_version": request.renderer_version,
        "requested_model_id": POE_RENDER_MODEL_ID,
        "output_mode": PLACEHOLDER_TEMPLATE_MODE,
        "request": request.to_dict(),
    }


class LLMNaturalRenderer:
    """Validate/replay one Poe placeholder rewrite before reconstructing AST."""

    __slots__ = ("cache", "producer")

    def __init__(
        self,
        *,
        producer: PoeNarrativeProducer | Callable[[Mapping[str, Any]], object] | None,
        cache: PoeResponseCache | None = None,
    ) -> None:
        if producer is not None and not (
            isinstance(producer, PoeNarrativeProducer) or callable(producer)
        ):
            raise TypeError(
                "producer must implement propose(request), be callable, or None"
            )
        if cache is not None and type(cache) is not PoeResponseCache:
            raise TypeError("cache must be a PoeResponseCache or None")
        if producer is None and cache is None:
            raise ValueError("LLMNaturalRenderer requires a cache or injected producer")
        self.producer = producer
        self.cache = cache

    def _produce(self, request: NarrativeRewriteRequest) -> object:
        if self.producer is None:
            raise NaturalRenderError(
                "RENDER_PRODUCER_MISSING",
                "a cache miss requires an injected renderer producer",
            )
        payload = request.provider_payload()
        if isinstance(self.producer, PoeNarrativeProducer):
            return self.producer.propose(payload)
        return self.producer(payload)

    def render(self, request: NarrativeRewriteRequest) -> NarrativeRewriteResult:
        if type(request) is not NarrativeRewriteRequest:
            raise TypeError("LLMNaturalRenderer requires a NarrativeRewriteRequest")
        key_material = _cache_material(request)
        response: object
        cache_hit = False
        if self.cache is not None:
            try:
                response = self.cache.load_render(key_material=key_material)
                cache_hit = True
            except PoeResponseCacheError as error:
                if error.code != "CACHE_MISS_FROZEN" or self.cache.frozen:
                    raise
                produced = self._produce(request)
                template = _response_template(produced)
                validate_narrative_template(request, template)
                artifact = self.cache.store_render(
                    key_material=key_material,
                    render={"template": template},
                    provenance={
                        "provider": "poe",
                        "requested_model_id": POE_RENDER_MODEL_ID,
                        "output_mode": PLACEHOLDER_TEMPLATE_MODE,
                    },
                )
                response = artifact.payload["render"]
                cache_hit = artifact.cache_hit
        else:
            response = self._produce(request)

        template = _response_template(response)
        local_template = validate_narrative_template(request, template)
        claim = ClaimNode.from_template(
            request.claim_id,
            local_template,
            request.locked_slots,
        )
        actual_literals = tuple(
            child for child in claim.content.children if type(child) is LiteralNode
        )
        expected_identity = tuple(
            sorted(
                (
                    snapshot.mention_id,
                    snapshot.state_or_edge_id,
                    snapshot.target_kind,
                    snapshot.value,
                )
                for snapshot in request.locked_snapshot
            )
        )
        actual_identity = tuple(
            sorted(
                (
                    literal.mention_id,
                    literal.state_or_edge_id,
                    literal.target_kind.value,
                    literal.value,
                )
                for literal in actual_literals
            )
        )
        if actual_identity != expected_identity:
            raise NaturalRenderError(
                "LOCKED_IDENTITY_CHANGED",
                "renderer reconstruction changed locked claim/literal identity",
            )
        return NarrativeRewriteResult(
            request=request,
            template=template,
            claim=claim,
            cache_hit=cache_hit,
        )


__all__ = [
    "LLM_NATURAL_RENDERER_VERSION",
    "PLACEHOLDER_TEMPLATE_MODE",
    "POE_RENDER_MODEL_ID",
    "LLMNaturalRenderer",
    "LockedSlotSnapshot",
    "NarrativeRewriteRequest",
    "NarrativeRewriteResult",
    "PoeNarrativeProducer",
    "validate_narrative_template",
]
