"""Deterministic label-blind assembly of locked natural trace facts.

This module receives only detector-visible AST claims, an exact programmatic
FORMAL snapshot, and a locked final-answer literal.  It never accepts a state
DAG (which may contain build-only oracle nodes), mutation labels, operators, or
correctness flags.  Narrative transitions are fixed local text; all factual
literals and every number are represented by :class:`LiteralNode` values.
"""

from __future__ import annotations

import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any

from molhallulens.core import MutationTargetKind
from molhallulens.modules.text_realization.trace_ast import (
    AnswerDocument,
    ClaimNode,
    LiteralNode,
    RenderedExample,
    SequenceNode,
    StepDocument,
    TextNode,
    TraceDocument,
    TraceNode,
    render_trace,
)

RULE_NATURAL_RENDERER_VERSION = "natural_rule_v1"

LABEL_LEAKAGE_PHRASES = (
    "causal role",
    "corrupted",
    "ground truth",
    "gt_smiles",
    "hallucinated",
    "hallucination",
    "incorrect",
    "label leakage",
    "operator correctness",
    "oracle",
    "reference answer",
    "reference-only",
)

_STEP_NAME = re.compile(r"[A-Z][A-Z_]*", flags=re.ASCII)
_NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|dozen)\b",
    flags=re.ASCII | re.IGNORECASE,
)


class NaturalRenderError(ValueError):
    """Structured fail-closed error from a natural renderer boundary."""

    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("NaturalRenderError code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("NaturalRenderError detail must be non-empty text")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def scan_label_leakage(text: str) -> tuple[str, ...]:
    """Return frozen label/reviewer phrases present in detector-visible text."""

    if type(text) is not str:
        raise TypeError("leakage scanner input must be a string")
    normalized = " ".join(text.casefold().split())
    return tuple(phrase for phrase in LABEL_LEAKAGE_PHRASES if phrase in normalized)


def contains_unlocked_number(text: str) -> bool:
    """Return whether prose spells or writes a number outside a literal slot."""

    if type(text) is not str:
        raise TypeError("number scanner input must be a string")
    return (
        any(character.isdigit() for character in text)
        or _NUMBER_WORD.search(text) is not None
    )


def trace_node_text(node: TraceNode) -> str:
    """Return an AST node's deterministic surface without assigning offsets."""

    if type(node) is TextNode:
        return node.text
    if type(node) is LiteralNode:
        return node.value
    if type(node) is ClaimNode:
        return trace_node_text(node.content)
    if type(node) is SequenceNode:
        return "".join(trace_node_text(child) for child in node.children)
    raise TypeError("trace_node_text requires a trace AST node")


def _walk(node: TraceNode) -> tuple[TraceNode, ...]:
    if type(node) in {TextNode, LiteralNode}:
        return (node,)
    if type(node) is ClaimNode:
        return (node, *_walk(node.content))
    return (node, *(item for child in node.children for item in _walk(child)))


def _validate_locked_ast_text(node: TraceNode, *, channel: str) -> None:
    """Require numbers to be literals and reject structural/leakage injection."""

    for item in _walk(node):
        if type(item) is not TextNode:
            continue
        if contains_unlocked_number(item.text):
            raise NaturalRenderError(
                "UNLOCKED_NUMBER",
                f"{channel} contains a number outside a locked LiteralNode",
            )
        folded = item.text.casefold()
        if "formal:" in folded or "answer:" in folded:
            raise NaturalRenderError(
                "RESERVED_STRUCTURE",
                f"{channel} cannot inject FORMAL or Answer headers",
            )
    leaked = scan_label_leakage(trace_node_text(node))
    if leaked:
        raise NaturalRenderError(
            "LABEL_LEAKAGE",
            f"{channel} contains forbidden renderer leakage phrases: {leaked!r}",
        )


@dataclass(frozen=True, slots=True)
class LockedNaturalStep:
    """One visible step with editable prose claims and exact locked FORMAL AST."""

    step_index: int
    step_name: str
    narrative_claims: tuple[ClaimNode, ...]
    formal_content: SequenceNode
    formal_text: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("LockedNaturalStep step_index must be positive")
        if (
            type(self.step_name) is not str
            or _STEP_NAME.fullmatch(self.step_name) is None
        ):
            raise ValueError("LockedNaturalStep step_name must be uppercase snake case")
        if isinstance(self.narrative_claims, (str, bytes)) or not isinstance(
            self.narrative_claims, SequenceABC
        ):
            raise TypeError("narrative_claims must be a sequence")
        claims = tuple(self.narrative_claims)
        if not claims or any(type(claim) is not ClaimNode for claim in claims):
            raise TypeError("narrative_claims must contain ClaimNode values")
        if type(self.formal_content) is not SequenceNode:
            raise TypeError("formal_content must be a SequenceNode")
        if type(self.formal_text) is not str or not self.formal_text:
            raise ValueError("formal_text must be non-empty text")
        if "\r" in self.formal_text or "\x00" in self.formal_text:
            raise ValueError("formal_text must use canonical LF-only, NUL-free text")
        actual_formal = trace_node_text(self.formal_content)
        if actual_formal != self.formal_text:
            raise NaturalRenderError(
                "LOCKED_FORMAL_CHANGED",
                "FORMAL AST differs from its programmatic text snapshot",
            )
        if "\n" in self.formal_text:
            raise NaturalRenderError(
                "FORMAL_MULTILINE",
                "one step FORMAL block must remain on one line",
            )
        for claim in claims:
            _validate_locked_ast_text(claim, channel="natural narrative")
        _validate_locked_ast_text(self.formal_content, channel="FORMAL block")
        object.__setattr__(self, "narrative_claims", claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "narrative_claims": [claim.to_dict() for claim in self.narrative_claims],
            "formal_content": self.formal_content.to_dict(),
            "formal_text": self.formal_text,
        }


@dataclass(frozen=True, slots=True)
class LockedFinalAnswer:
    """One exact final-answer literal owned entirely by the program."""

    literal: LiteralNode
    claim_id: str = "claim.final_answer"

    def __post_init__(self) -> None:
        if type(self.literal) is not LiteralNode:
            raise TypeError("LockedFinalAnswer literal must be a LiteralNode")
        if (
            self.literal.target_kind is not MutationTargetKind.NODE
            or self.literal.state_or_edge_id != "final_answer"
        ):
            raise ValueError("LockedFinalAnswer literal must target final_answer")
        if type(self.claim_id) is not str or not self.claim_id:
            raise ValueError("LockedFinalAnswer claim_id must be non-empty text")
        if any(character.isspace() for character in self.claim_id):
            raise ValueError("LockedFinalAnswer claim_id cannot contain whitespace")

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "literal": self.literal.to_dict()}


@dataclass(frozen=True, slots=True)
class NaturalRenderRequest:
    """The complete public rule-renderer input; deliberately label/oracle-free."""

    steps: tuple[LockedNaturalStep, ...]
    final_answer: LockedFinalAnswer
    segment_separator: str = "\n\n"
    renderer_version: str = RULE_NATURAL_RENDERER_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.steps, (str, bytes)) or not isinstance(
            self.steps, SequenceABC
        ):
            raise TypeError("NaturalRenderRequest steps must be a sequence")
        steps = tuple(self.steps)
        if not steps or any(type(step) is not LockedNaturalStep for step in steps):
            raise TypeError("steps must contain LockedNaturalStep values")
        if tuple(step.step_index for step in steps) != tuple(range(1, len(steps) + 1)):
            raise ValueError("natural render steps must be contiguous and one-based")
        if type(self.final_answer) is not LockedFinalAnswer:
            raise TypeError("final_answer must be a LockedFinalAnswer")
        if type(self.segment_separator) is not str or not self.segment_separator:
            raise ValueError("segment_separator must be non-empty text")
        if "\r" in self.segment_separator or "\x00" in self.segment_separator:
            raise ValueError("segment_separator must use canonical NUL-free text")
        if self.renderer_version != RULE_NATURAL_RENDERER_VERSION:
            raise ValueError("unsupported rule natural renderer version")
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "renderer_version": self.renderer_version,
            "segment_separator": self.segment_separator,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer.to_dict(),
        }


class RuleNaturalRenderer:
    """Add fixed transitions around immutable narrative/FORMAL/Answer ASTs."""

    __slots__ = ()

    narrative_separator = TextNode("; ")

    def build_document(self, request: NaturalRenderRequest) -> TraceDocument:
        if type(request) is not NaturalRenderRequest:
            raise TypeError("RuleNaturalRenderer requires a NaturalRenderRequest")
        steps: list[StepDocument] = []
        for step in request.steps:
            narrative: list[TraceNode] = []
            for index, claim in enumerate(step.narrative_claims):
                if index:
                    narrative.append(self.narrative_separator)
                narrative.append(claim)
            content = SequenceNode(
                (
                    TextNode(f"Step {step.step_index} [{step.step_name}]: "),
                    SequenceNode(tuple(narrative)),
                    TextNode("\n  FORMAL: "),
                    step.formal_content,
                )
            )
            steps.append(StepDocument(step.step_index, content))
        answer_claim = ClaimNode.from_template(
            request.final_answer.claim_id,
            "Answer: {answer}",
            {"answer": request.final_answer.literal},
        )
        return TraceDocument(
            steps=tuple(steps),
            answer=AnswerDocument(SequenceNode((answer_claim,))),
            segment_separator=request.segment_separator,
        )

    def render(self, request: NaturalRenderRequest) -> RenderedExample:
        document = self.build_document(request)
        rendered = render_trace(document)
        leaked = scan_label_leakage(rendered.detector_text)
        if leaked:
            raise NaturalRenderError(
                "LABEL_LEAKAGE",
                f"rendered trace contains forbidden phrases: {leaked!r}",
            )
        for step in request.steps:
            expected = f"\n  FORMAL: {step.formal_text}"
            segment = rendered.segment_spans[f"reasoning.step.{step.step_index:02d}"]
            segment_text = rendered.detector_text[segment.start : segment.end]
            if not segment_text.endswith(expected):
                raise NaturalRenderError(
                    "LOCKED_FORMAL_OMITTED",
                    "rendered trace omitted an exact programmatic FORMAL block",
                )
        expected_answer = f"Answer: {request.final_answer.literal.value}"
        answer_span = rendered.segment_spans["final_answer"]
        if (
            rendered.detector_text[answer_span.start : answer_span.end]
            != expected_answer
        ):
            raise NaturalRenderError(
                "LOCKED_ANSWER_CHANGED",
                "rendered final Answer differs from its locked literal",
            )
        return rendered


def render_natural_rule(request: NaturalRenderRequest) -> RenderedExample:
    return RuleNaturalRenderer().render(request)


__all__ = [
    "LABEL_LEAKAGE_PHRASES",
    "RULE_NATURAL_RENDERER_VERSION",
    "LockedFinalAnswer",
    "LockedNaturalStep",
    "NaturalRenderError",
    "NaturalRenderRequest",
    "RuleNaturalRenderer",
    "contains_unlocked_number",
    "render_natural_rule",
    "scan_label_leakage",
    "trace_node_text",
]
