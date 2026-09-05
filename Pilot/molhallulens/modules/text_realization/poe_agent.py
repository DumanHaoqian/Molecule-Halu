"""Poe boundary for audited, locally compiled edits of original ``step_text``.

Poe receives each original complete step and its locally rendered modified FORMAL.
The model returns prose and typed references; local code fills values and marker
IDs. COPY is local, passed steps survive retries, and diagnostics are redacted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from uuid import uuid4
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)
from molhallulens.config.paths import PROJECT_ROOT

from .occurrence_audit import (
    arithmetic_violations,
    enumeration_violations,
    loose_occurrence_spans,
)
from .enumeration_plan import enumeration_inventory, validate_enumeration_inventory
from .claim_surfaces import claim_surface_pairs, patch_prose_signature
from .segments import SegmentContractError, compile_segments, local_copy, step_payload, response_segments_example
from .diagnostics import append_diagnostic, make_diagnostic, redact, rejection_code

POE_RENDERER_VERSION = "poe_segments_v20"
FORMAL_MARKER = "\n  FORMAL: "
HALLU_MARKER_PATTERN = re.compile(
    r"\[\[HALLU:([a-z][a-z0-9_]*\.[0-9]{2})\]\](.*?)\[\[/HALLU\]\]",
    flags=re.ASCII | re.DOTALL,
)
_NODE_ID = re.compile(r"[a-z][a-z0-9_]*", flags=re.ASCII)
_MARKER_TOKEN = re.compile(r"\[\[(?:HALLU:[^\]]*|/HALLU)\]\]", flags=re.ASCII)
_LEADING_STEP_HEADER = re.compile(
    r"^Step\s+([0-9]+)\s+\[([^\]\r\n]+)\]:[ \t]*",
    flags=re.ASCII,
)
_LINE_INITIAL_STEP_HEADER = re.compile(
    r"(?:^|\n)[ \t]*Step\s+[0-9]+\s+\[[^\]\r\n]+\]:",
    flags=re.ASCII,
)


class PoeTextRealizationError(RuntimeError):
    """Fail-closed Poe configuration, transport, or response error."""

    def __init__(self, message, *, diagnostics=(), diagnostic_path=None, code=None, expected=None, observed=None):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)
        self.diagnostic_path = diagnostic_path
        self.code = code
        self.expected = expected
        self.observed = observed


class StepRewriteMode(StrEnum):
    """How one step's natural-language channel may be changed."""

    COPY = "copy"
    OCCURRENCE_PATCH = "occurrence_patch"
    DERIVATION_REWRITE = "derivation_rewrite"


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
class AffectedNodeClaim:
    """One changed DAG claim audited against a step's natural language."""

    node_id: str
    before_text: str
    after_text: str
    required_occurrence_count: int | None = None
    parent_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.parent_node_id is not None and (
            type(self.parent_node_id) is not str
            or _NODE_ID.fullmatch(self.parent_node_id) is None
            or not self.node_id.startswith(self.parent_node_id + "__enumeration_")
        ):
            raise ValueError("derived enumeration claim requires its parent namespace")
        if type(self.node_id) is not str or _NODE_ID.fullmatch(self.node_id) is None:
            raise ValueError("node_id is invalid")
        for value, name in (
            (self.before_text, "before_text"),
            (self.after_text, "after_text"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
            if _MARKER_TOKEN.search(value) is not None:
                raise ValueError(f"{name} cannot contain marker tokens")
        if self.before_text == self.after_text:
            raise ValueError("an affected claim must change its rendered value")
        if self.required_occurrence_count is not None and (
            type(self.required_occurrence_count) is not int
            or self.required_occurrence_count < 1
        ):
            raise ValueError(
                "required_occurrence_count must be a positive integer or None"
            )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "required_occurrence_count": self.required_occurrence_count,
            "parent_node_id": self.parent_node_id,
            "allowed_surface_pairs": [
                {"before_text": before, "after_text": after}
                for before, after in claim_surface_pairs(self.node_id, self.before_text, self.after_text)
            ],
        }


@dataclass(frozen=True, slots=True)
class PoeStepRewriteInput:
    """One original step plus the exact FORMAL the rewritten step must express."""

    step_index: int
    step_name: str
    original_step_text: str
    modified_formal_ab: str
    required_hallucination_occurrences: tuple[RequiredHallucinationOccurrence, ...]
    rewrite_mode: StepRewriteMode | None = None
    affected_node_claims: tuple[AffectedNodeClaim, ...] = ()
    preserved_enumerations: tuple[str, ...] = ()

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
        claims = tuple(self.affected_node_claims)
        if any(type(item) is not AffectedNodeClaim for item in claims):
            raise TypeError("affected_node_claims must contain AffectedNodeClaim values")
        claim_ids = tuple(item.node_id for item in claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("affected_node_claims cannot repeat a node")
        for claim in claims:
            if claim.parent_node_id and claim.parent_node_id not in claim_ids:
                raise ValueError("derived enumeration parent must be an affected claim")
            if claim.parent_node_id and not any(
                f"[[HALLU:{claim.node_id}." in clause for clause in self.preserved_enumerations
            ):
                raise ValueError("derived enumeration claim requires a preserved breakdown")
        mode = self.rewrite_mode
        if mode is None:
            mode = (
                StepRewriteMode.OCCURRENCE_PATCH
                if required
                else StepRewriteMode.COPY
            )
        if type(mode) is not StepRewriteMode:
            raise TypeError("rewrite_mode must be a StepRewriteMode or None")
        if self.preserved_enumerations and mode is not StepRewriteMode.DERIVATION_REWRITE:
            raise ValueError("preserved enumeration plans require derivation rewrite mode")
        if mode is StepRewriteMode.COPY:
            if required or claims:
                raise ValueError("COPY steps cannot have occurrences or affected claims")
        elif mode is StepRewriteMode.OCCURRENCE_PATCH:
            if not required:
                raise ValueError("OCCURRENCE_PATCH requires occurrences")
        else:
            if required or not claims:
                raise ValueError(
                    "DERIVATION_REWRITE requires affected claims and no patch occurrences"
                )
        object.__setattr__(self, "rewrite_mode", mode)
        object.__setattr__(
            self,
            "affected_node_claims",
            tuple(sorted(claims, key=lambda item: item.node_id)),
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
            "rewrite_mode": self.rewrite_mode.value,
            "preserved_enumerations": list(self.preserved_enumerations),
            "enumeration_inventory": enumeration_inventory(self.preserved_enumerations),
            "affected_node_claims": [
                item.to_prompt_dict() for item in self.affected_node_claims
            ],
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
    validation_rejection_codes: tuple[str, ...]
    step_execution: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()
    diagnostic_path: str | None = None


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
    if expected.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
        claims_by_node = {
            claim.node_id: claim for claim in expected.affected_node_claims
        }
        observed_ids: set[str] = set()
        observed_by_node: dict[str, list[int]] = {
            node_id: [] for node_id in claims_by_node
        }
        for occurrence_id, value, _, _ in markers:
            node_id, suffix = occurrence_id.rsplit(".", 1)
            if node_id not in claims_by_node:
                raise PoeTextRealizationError(
                    "Poe marked an unplanned derivation claim: " f"{occurrence_id}"
                )
            if occurrence_id in observed_ids:
                raise PoeTextRealizationError(
                    f"Poe duplicated HALLU marker occurrence: {occurrence_id}"
                )
            claim = claims_by_node[node_id]
            if value not in {after for _, after in claim_surface_pairs(node_id, claim.before_text, claim.after_text)}:
                raise PoeTextRealizationError(
                    f"Poe marker for {occurrence_id} does not contain the exact "
                    f"after_text for {node_id!r}"
                )
            observed_ids.add(occurrence_id)
            observed_by_node[node_id].append(int(suffix))
        missing_nodes = sorted(
            node_id for node_id, suffixes in observed_by_node.items() if not suffixes
        )
        if missing_nodes:
            raise PoeTextRealizationError(
                "Poe omitted affected derivation claims: " f"{missing_nodes}"
            )
        for node_id, suffixes in observed_by_node.items():
            if sorted(suffixes) != list(range(1, len(suffixes) + 1)):
                raise PoeTextRealizationError(
                    "DERIVATION_REWRITE occurrence suffixes must be consecutive for "
                    f"{node_id!r}"
                )
            required_count = claims_by_node[node_id].required_occurrence_count
            if required_count is not None and len(suffixes) != required_count:
                raise PoeTextRealizationError(
                    "DERIVATION_REWRITE marker count does not match the paired "
                    f"occurrence count for {node_id!r}"
                )
    else:
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

    # COPY means the coverage audit proved that no changed claim is present in
    # prose. It is deliberately independent of whether FORMAL changed.
    if expected.rewrite_mode is StepRewriteMode.COPY:
        original_head, _ = _split_complete_step(expected.original_step_text)
        if marked_head != original_head:
            raise PoeTextRealizationError(
                "Poe changed natural language without any required occurrences "
                "(COPY mode)"
            )

    clean_natural = strip_hallucination_markers(marked_natural)
    try:
        validate_enumeration_inventory(marked_natural, expected.preserved_enumerations)
    except ValueError as error:
        raise PoeTextRealizationError(str(error)) from error
    if expected.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH:
        original_head, _ = _split_complete_step(expected.original_step_text)
        expected_natural = original_head[len(prefix) :]
        for occurrence in sorted(
            expected.required_hallucination_occurrences,
            key=lambda item: item.original_start,
            reverse=True,
        ):
            expected_natural = (
                expected_natural[: occurrence.original_start]
                + occurrence.after_text
                + expected_natural[occurrence.original_end :]
            )
        if patch_prose_signature(clean_natural) != patch_prose_signature(expected_natural):
            raise PoeTextRealizationError(
                "OCCURRENCE_PATCH changed text outside required claim values",
                code="unapproved_prose_change", expected=expected_natural, observed=clean_natural,
            )
    if _LINE_INITIAL_STEP_HEADER.search(clean_natural) is not None:
        raise PoeTextRealizationError(
            "Poe natural-language body still contains a redundant Step header"
        )
    stale_search = list(clean_natural)
    for _, _, start, end in markers:
        stale_search[start:end] = " " * (end - start)
    stale_search_text = "".join(stale_search)
    for claim in expected.affected_node_claims:
        if not claim.parent_node_id:
            continue
        # Scope component counts by their preserved noun phrase, not the digit
        # alone (3 carbons and 3 fluorines are different claims).
        for clause in expected.preserved_enumerations:
            component = re.search(
                rf"\[\[HALLU:{re.escape(claim.node_id)}\.\d{{2}}\]\].*?\[\[/HALLU\]\]"
                r"\s+([A-Za-z][A-Za-z -]*?)(?=\s+(?:and|which|in)\b|[,=+().]|$)",
                clause,
            )
            if component:
                label = component[1].strip()
                if label.lower() in {"ring", "rings", "atom", "atoms", "heavy atom", "heavy atoms"}:
                    # A unit alone is not a component identity; total claims
                    # have their own node-aware audit below. The actual
                    # breakdown inventory remains validated above.
                    continue
                for value in (claim.before_text, claim.after_text):
                    pattern = rf"(?<!\w){re.escape(value)}\s+{re.escape(label)}\b"
                    # Another preserved enumeration may legitimately contain
                    # the same count/noun (source vs product, for example).
                    preserved_unmarked = re.sub(
                        r"\[\[HALLU:[^\]]+\]\].*?\[\[/HALLU\]\]", " ",
                        "\n".join(expected.preserved_enumerations),
                    )
                    if len(re.findall(pattern, stale_search_text, re.I)) > len(re.findall(pattern, preserved_unmarked, re.I)):
                        raise PoeTextRealizationError(
                            f"enumeration_unmarked_component: {claim.node_id} ({value!r} {label!r})"
                        )
    if expected.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
        for claim in expected.affected_node_claims:
            if claim.parent_node_id:
                continue  # Component occurrences are locked by preserved_enumerations.
            unmarked_after = loose_occurrence_spans(
                claim.node_id,
                claim.after_text,
                stale_search_text,
                step_name=expected.step_name,
            )
            if unmarked_after:
                raise PoeTextRealizationError(
                    "rewritten derivation retained an unmarked after_text claim for "
                    f"{claim.node_id!r}"
                )
    for claim in expected.affected_node_claims:
        if claim.parent_node_id:
            continue
        stale = loose_occurrence_spans(
            claim.node_id,
            claim.before_text,
            stale_search_text,
            step_name=expected.step_name,
        )
        if stale:
            raise PoeTextRealizationError(
                "rewritten natural language retained a stale claim for "
                f"{claim.node_id!r}"
            )
    if expected.affected_node_claims:
        invalid_equations = arithmetic_violations(clean_natural)
        if invalid_equations:
            raise PoeTextRealizationError(
                "rewritten natural language contains false displayed arithmetic: "
                f"{list(invalid_equations)}"
            )
        invalid_enumerations = enumeration_violations(clean_natural)
        if invalid_enumerations:
            raise PoeTextRealizationError(
                "rewritten natural language contains an enumeration whose component "
                f"sum disagrees with its total: {list(invalid_enumerations)}"
            )
    return rewritten_step_text




def _system_prompt() -> str:
    return (
        "Rewrite molecule-editing reasoning using STRUCTURED SEGMENTS, never literal markers. "
        "Return exactly {steps:[{step_index:integer,segments:[...]}]} for the requested steps only. "
        "Allowed segments: {text:string}; {claim_ref:node_id,surface:canonical|symbol|name|title_name}; "
        "{occurrence_ref:provided_occurrence_id}; {enumeration_ref:provided_enumeration_id}; "
        "or exclusively [{patch_ref:original_occurrences}] for occurrence_patch; "
        "or exclusively [{draft_ref:complete_derivation}] for derivation_rewrite. "
        "Each segment contains only the indicated keys. Local code inserts all claim values, "
        "numbers, marker IDs, Step headers and FORMAL. Never return a value field or HALLU markup. "
        "The editable input is original_natural_body, NOT a complete step_text. "
        "The RESPONSE_SHAPE example is mode-specific and uses actual valid IDs. "
        "In occurrence_patch, return the supplied patch_ref operation. It locally preserves "
        "the original body and inserts EVERY exact replacement at its original location. "
        "Do not copy occurrence IDs manually unless a listed grammatical adjustment is needed. "
        "claim_ref is FORBIDDEN in this mode. "
        "ONLY IF choosing expanded segments instead of patch_ref, reference EACH required occurrence exactly once at its original "
        "position using occurrence_ref. Preserve other prose except whitespace, scoped "
        "source/product [molecule] has/contains, and numeric atom(s)/ring(s) inflection. "
        "In derivation_rewrite, first review and explicitly select the complete_draft via draft_ref. "
        "ONLY IF rejecting that draft and submitting expanded segments instead, write coherent prose using claim_ref for EVERY occurrence of "
        "each affected non-component claim. Available surfaces are listed per claim. Do not "
        "write old or new claim values literally in text. Source facts not edited remain fixed. "
        "Every affected claim must appear; obey required_occurrence_count when supplied. "
        "In expanded segments ONLY, use EVERY enumeration_ref exactly once, as a separate complete sentence, with suitable "
        "sentence/line separators. Local code renders its entire component list and total. "
        "Never delete, paraphrase, duplicate, or manually render that inventory or change its "
        "components. Derived component claims are emitted only through enumeration_ref. "
        "Explanations and arithmetic must agree internally with MODIFIED_FORMAL_AB and the "
        "planned claims, even when those claims are chemically false relative to the source. "
        "The DERIVATION example is a complete candidate rewrite, not merely a schema. "
        "INPUT.complete_draft shows its fully rendered natural body. Prefer selecting that "
        "draft with draft_ref when it expresses the requested claims; optional prose "
        "improvements must obey all constraints. Never retain original chemical descriptors "
        "such as carbonyl carbon when the anchor element/position has changed. "
        "Do not invent an unlisted fact or error. Do not mention editing or hallucinations. "
        "Context steps are read-only; do not return them. A repair request contains only failed "
        "steps and their prior rejected response/diagnostics; passed steps are already retained. "
        "No Step header, FORMAL, Answer, CR/NUL or markdown in text segments. JSON only."
    )


def _user_prompt(request: PoeRewriteRequest, steps=None, repair=()) -> str:
    steps = tuple(s for s in request.steps if s.rewrite_mode is not StepRewriteMode.COPY) if steps is None else tuple(steps)
    response_shape = {
        "steps": [
            {
                "step_index": step.step_index,
                "segments": ([{"patch_ref": "original_occurrences"}]
                             if step.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH
                             else [{"draft_ref": "complete_derivation"}]),
            }
            for step in steps
        ]
    }
    # Repeat the wire contract in the user message: named Poe bots may apply
    # their own system instructions. Never rely on the system channel alone.
    prompt_repair = []
    for diagnostic in repair:
        item = dict(diagnostic)
        item.pop("original_step_text", None)
        if item.get("rewrite_mode") == "occurrence_patch":
            item["repair_instruction"] = (
                'Return exactly segments:[{"patch_ref":"original_occurrences"}] for this step. '
                'It includes ALL original prose and ALL required occurrences. A list of only '
                'occurrence_ref objects deletes the prose and is invalid. Do not repeat that list.'
            )
        prompt_repair.append(item)
    return (
        _system_prompt() + "\n\n"
        "TASK: Return body-only segments for INPUT.steps. PATCH examples are exact; "
        "DERIVATION examples are complete prose candidates; prefer them unchanged if suitable. "
        "The two shorthand operations ALREADY include every required occurrence and inventory. "
        "Do NOT expand them just to satisfy the expanded-segment rules. "
        "Only for expanded derivation segments, use enumeration_ref exactly once per inventory; never recreate its "
        "component list in text. INPUT.context_steps and modified_formal_ab are read-only. "
        "Repair entries describe REJECTED responses: fix them, do not copy their forbidden syntax.\n"
        "RESPONSE_SHAPE:\n"
        + json.dumps(response_shape, ensure_ascii=False, separators=(",", ":"))
        + "\nINPUT:\n"
        + json.dumps(
            {
                "origin_id": request.origin_id, "subtask": request.subtask,
                "indexed_smiles": request.indexed_smiles, "instruction": request.instruction,
                "steps": [step_payload(s) for s in steps],
                "context_steps": [{"step_index": s.step_index, "step_name": s.step_name,
                                   "original_natural_body": step_payload(s)["original_natural_body"],
                                   "modified_formal_ab": s.modified_formal_ab} for s in request.steps],
                "repair": prompt_repair,
            },
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


def _strip_model_step_header(body: str, expected: PoeStepRewriteInput) -> str:
    """Remove one redundant exact header and reject a mismatched one."""

    match = _LEADING_STEP_HEADER.match(body)
    if match is None:
        return body
    observed_index = int(match.group(1))
    observed_name = match.group(2)
    if observed_index != expected.step_index or observed_name != expected.step_name:
        raise PoeTextRealizationError(
            "Poe returned a mismatched Step header inside rewritten natural language"
        )
    return body[match.end() :].lstrip()


def _discard_model_owned_suffix(body: str) -> str:
    """Discard FORMAL/Answer fields because local typed state owns both."""

    boundary = re.search(r"\n[ \t]*FORMAL\s*:\s*", body, flags=re.IGNORECASE)
    if boundary is not None:
        body = body[: boundary.start()].rstrip()
    answer = re.search(r"\n[ \t]*Answer\s*:\s*", body, flags=re.IGNORECASE)
    if answer is not None:
        body = body[: answer.start()].rstrip()
    return body


def _extract_natural_body(value: Any, expected: PoeStepRewriteInput) -> str:
    """Recover only prose even when Poe redundantly returns a complete step.

    Model-supplied Step headers, FORMAL blocks, and Answers are discarded rather
    than trusted. The caller always reconstructs them from local typed state.
    """

    if type(value) is not str or not value.strip():
        raise PoeTextRealizationError(
            "Poe rewritten_natural_language must be non-empty text"
        )
    if "\x00" in value:
        raise PoeTextRealizationError(
            "Poe rewritten_natural_language contains a forbidden NUL token"
        )
    body = value.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Normalize a redundant model-owned header before local code adds the one
    # authoritative header.
    body = _strip_model_step_header(body, expected)
    body = _discard_model_owned_suffix(body)
    if not body:
        raise PoeTextRealizationError(
            "Poe response contains no natural language after local field extraction"
        )
    return body


def _response_rows(response_text: str, steps) -> dict[int, dict]:
    value = _extract_json_object(response_text)
    if set(value) != {"steps"} or not isinstance(value["steps"], list):
        raise PoeTextRealizationError("Poe response must contain exactly one steps array")
    allowed = {s.step_index for s in steps}
    rows = {}
    for row in value["steps"]:
        if not isinstance(row, Mapping) or set(row) != {
            "step_index",
            "segments",
        }:
            raise SegmentContractError("response_shape", "each Poe step requires exactly step_index and segments")
        index = row["step_index"]
        if type(index) is not int or index not in allowed or index in rows:
            raise SegmentContractError("step_identity", "response contains an unknown, duplicate or non-integer step index", expected=sorted(allowed), observed=index)
        rows[index] = dict(row)
    return rows


def _parse_and_validate_response(response_text: str, request: PoeRewriteRequest) -> tuple[str, ...]:
    pending = tuple(s for s in request.steps if s.rewrite_mode is not StepRewriteMode.COPY)
    rows = _response_rows(response_text, pending)
    if set(rows) != {s.step_index for s in pending}:
        raise SegmentContractError("missing_step", "cached response is missing a requested step")
    return tuple(local_copy(s) if s.rewrite_mode is StepRewriteMode.COPY else compile_segments(rows[s.step_index]["segments"], s) for s in request.steps)


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
            "fastapi-poe is not installed; run: python -m pip install -r requirements.txt"
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
    """Compile Poe segments, retry failed steps and cache fully validated output."""

    __slots__ = (
        "config",
        "transport",
        "_environment",
        "cache_directory",
        "_rewrite_call_count",
        "_uncached_request_count",
        "_cache_hit_count",
        "_network_request_count",
        "_retry_count",
        "_requests_with_retry",
        "_validation_rejection_counts",
        "diagnostic_directory",
        "_step_retry_count",
        "_local_copy_step_count",
    )

    def __init__(
        self,
        config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
        *,
        transport: PoeTransport | None = None,
        environment: Mapping[str, str] | None = None,
        cache_directory: Path | None = None,
        diagnostic_directory: Path | None = None,
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
        self._rewrite_call_count = 0
        configured_diagnostics = Path(config.poe_diagnostic_directory)
        self.diagnostic_directory = (
            Path(diagnostic_directory) if diagnostic_directory is not None else
            Path(cache_directory) / "diagnostics" if cache_directory is not None else
            configured_diagnostics if configured_diagnostics.is_absolute() else PROJECT_ROOT / configured_diagnostics
        )
        self._uncached_request_count = 0
        self._cache_hit_count = 0
        self._network_request_count = 0
        self._retry_count = 0
        self._requests_with_retry = 0
        self._validation_rejection_counts: dict[str, int] = {}
        self._step_retry_count = 0
        self._local_copy_step_count = 0

    def telemetry(self) -> dict[str, Any]:
        """Return secret-free cumulative counters for batch summaries."""

        return {
            "rewrite_call_count": self._rewrite_call_count,
            "uncached_request_count": self._uncached_request_count,
            "cache_hit_count": self._cache_hit_count,
            "network_request_count": self._network_request_count,
            "retry_count": self._retry_count,
            "requests_with_retry": self._requests_with_retry,
            "step_retry_count": self._step_retry_count,
            "local_copy_step_count": self._local_copy_step_count,
            "validation_rejection_counts": dict(
                sorted(self._validation_rejection_counts.items())
            ),
        }

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
        except (OSError, json.JSONDecodeError) as error:
            raise PoeTextRealizationError("cached Poe response cannot be read") from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("renderer_version") != POE_RENDERER_VERSION
            or payload.get("bot_name") != self.config.poe_bot_name
            or payload.get("prompt_sha256") != prompt_sha256
            or type(payload.get("response_text")) is not str
            or payload.get("response_sha256")
            != hashlib.sha256(
                str(payload.get("response_text", "")).encode("utf-8")
            ).hexdigest()
        ):
            return None
        response_text = payload["response_text"]
        secret = self._environment.get(self.config.poe_api_key_env)
        if redact(response_text, secret.strip() if type(secret) is str else None) != response_text:
            return None
        try:
            rewritten = _parse_and_validate_response(response_text, request)
        except (PoeTextRealizationError, SegmentContractError):
            # A stricter validator may invalidate an otherwise well-formed old
            # cache entry even when prompt text did not change. Treat it as a
            # stale artifact and obtain a fresh response.
            return None
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
        self._rewrite_call_count += 1
        self._local_copy_step_count += sum(s.rewrite_mode is StepRewriteMode.COPY for s in request.steps)
        system_prompt = _system_prompt()
        base_user_prompt = _user_prompt(request)
        prompt_identity = system_prompt + "\n\n" + base_user_prompt
        prompt_sha256 = hashlib.sha256(prompt_identity.encode("utf-8")).hexdigest()

        run_id = uuid4().hex
        diagnostic_path = self.diagnostic_directory / f"{prompt_sha256}.{run_id}.jsonl"
        diagnostics = []
        secret = self._environment.get(self.config.poe_api_key_env)
        secret = secret.strip() if type(secret) is str else None
        attempts = {s.step_index: 0 for s in request.steps}

        def record_error(error, step, attempt, response):
            diagnostic = make_diagnostic(
                run_id=run_id, origin_id=request.origin_id, model=self.config.poe_bot_name,
                version=POE_RENDERER_VERSION, prompt_hash=prompt_sha256,
                step=step, attempt=attempt, error=error, response=response,
                secret=secret, limit=self.config.poe_diagnostic_max_characters,
            )
            diagnostics.append(diagnostic)
            code = diagnostic["error_code"]
            self._validation_rejection_counts[code] = self._validation_rejection_counts.get(code, 0) + 1
            if self.config.poe_save_diagnostics:
                append_diagnostic(diagnostic_path, diagnostic)
            return diagnostic

        def result(rewritten, response_text, network_count, cache_hit):
            rows = {row["step_index"]: row["segments"] for row in json.loads(response_text)["steps"]}
            def response_mode(step):
                segments = rows.get(step.step_index, [])
                if step.rewrite_mode is StepRewriteMode.COPY:
                    return "local_copy"
                if segments == [{"draft_ref": "complete_derivation"}]:
                    return "poe_selected_local_draft"
                if segments == [{"patch_ref": "original_occurrences"}]:
                    return "poe_selected_exact_patch"
                return "poe_authored_segments"
            return PoeRewriteResult(
                rewritten_step_texts=rewritten, bot_name=self.config.poe_bot_name,
                api_key_environment_variable=self.config.poe_api_key_env,
                prompt_sha256=prompt_sha256,
                response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                network_request_count=network_count, cache_hit=cache_hit,
                validation_rejection_codes=tuple(d["error_code"] for d in diagnostics),
                step_execution=tuple({
                    "step_index": s.step_index, "attempts": attempts[s.step_index],
                    "backend": "local_copy" if s.rewrite_mode is StepRewriteMode.COPY else "validated_cache" if cache_hit else "poe_segments",
                    "protocol": POE_RENDERER_VERSION,
                    "response_mode": response_mode(s),
                } for s in request.steps),
                diagnostics=tuple(diagnostics),
                diagnostic_path=str(diagnostic_path) if diagnostics and self.config.poe_save_diagnostics else None,
            )

        cached = self._load_cache(prompt_sha256, request)
        if cached is not None:
            self._cache_hit_count += 1
            response_text, rewritten = cached
            return result(rewritten, response_text, 0, True)

        accepted = {s.step_index: local_copy(s) for s in request.steps if s.rewrite_mode is StepRewriteMode.COPY}
        pending = tuple(s for s in request.steps if s.step_index not in accepted)
        if not pending:
            return result(tuple(accepted[s.step_index] for s in request.steps), '{"steps":[]}', 0, False)
        self._uncached_request_count += 1
        accepted_rows, repair = {}, []
        network_count = 0
        for attempt in range(1, self.config.poe_max_attempts + 1):
            if attempt > 1:
                self._retry_count += 1
                self._step_retry_count += len(pending)
                if attempt == 2:
                    self._requests_with_retry += 1
            user_prompt = _user_prompt(request, pending, repair)
            api_key = self._api_key() if self.transport is None else None
            self._network_request_count += 1
            network_count += 1
            for step in pending:
                attempts[step.step_index] += 1
            try:
                response_text = (
                    _default_poe_transport(system_prompt, user_prompt, self.config.poe_bot_name, self.config.poe_temperature, api_key=api_key)
                    if self.transport is None else
                    self.transport(system_prompt, user_prompt, self.config.poe_bot_name, self.config.poe_temperature)
                )
            except Exception as error:
                # Never log/rethrow SDK exception text: it may contain headers.
                safe_error = SegmentContractError("transport_error", f"Poe transport failed ({type(error).__name__})")
                record_error(safe_error, None, attempt, "[transport response unavailable]")
                raise PoeTextRealizationError(str(safe_error), diagnostics=diagnostics,
                    diagnostic_path=str(diagnostic_path) if self.config.poe_save_diagnostics else None) from None
            try:
                if type(response_text) is str and redact(response_text, secret) != response_text:
                    raise SegmentContractError("sensitive_output", "response contained credential-like content; discarded")
                rows = _response_rows(response_text, pending)
            except (PoeTextRealizationError, SegmentContractError) as error:
                repair = [record_error(error, None, attempt, response_text)]
                continue
            repair, failed = [], []
            for step in pending:
                row = rows.get(step.step_index)
                try:
                    if row is None:
                        raise SegmentContractError("missing_step", "response omitted this requested step", expected=step.step_index)
                    accepted[step.step_index] = compile_segments(row["segments"], step)
                    accepted_rows[step.step_index] = row
                except (PoeTextRealizationError, SegmentContractError) as error:
                    failed.append(step)
                    repair.append(record_error(error, step, attempt, row))
            pending = tuple(failed)
            if not pending:
                response_text = json.dumps({"steps": [accepted_rows[i] for i in sorted(accepted_rows)]}, ensure_ascii=False)
                # Revalidate the complete assembled response before cache/release.
                rewritten = _parse_and_validate_response(response_text, request)
                if rewritten != tuple(accepted[s.step_index] for s in request.steps):
                    raise PoeTextRealizationError("assembled step validation changed accepted text")
                self._store_cache(prompt_sha256, response_text)
                return result(rewritten, response_text, network_count, False)
        raise PoeTextRealizationError(
            "Poe response failed the local step_text contract after "
            f"{self.config.poe_max_attempts} attempts; failed steps={[s.step_index for s in pending]}: "
            + (diagnostics[-1]["message"] if diagnostics else "unknown contract failure"),
            diagnostics=diagnostics,
            diagnostic_path=str(diagnostic_path) if self.config.poe_save_diagnostics else None,
        )


__all__ = [
    "AffectedNodeClaim",
    "FORMAL_MARKER",
    "HALLU_MARKER_PATTERN",
    "POE_RENDERER_VERSION",
    "PoeRewriteRequest",
    "PoeRewriteResult",
    "RequiredHallucinationOccurrence",
    "StepRewriteMode",
    "PoeStepRewriteInput",
    "PoeStepTextAgent",
    "PoeTextRealizationError",
    "PoeTransport",
    "parse_hallucination_markers",
    "strip_hallucination_markers",
    "validate_rewritten_step_text",
]
