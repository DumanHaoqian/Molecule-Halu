"""Immutable trace AST with single-pass deterministic mention offsets.

The AST contains only detector-visible locked text and semantic target
identities.  It has no ground-truth, correctness, operator, or label fields.
Every offset is captured at the instant its literal is appended to the output
buffer, so repeated surface values remain distinct occurrences.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from string import Formatter
from typing import Any, TypeAlias

from molhallulens.modules.text_realization.spans import (
    CharSpan,
    MentionSpan,
    validate_char_span,
    validate_mention_spans,
    validate_non_overlapping_spans,
)
from molhallulens.core import FrozenMap, MutationTargetKind, SegmentKind

TRACE_AST_VERSION = "trace_ast_v1"


def _canonical_text(value: str, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if "\x00" in value or "\r" in value:
        raise ValueError(f"{name} must use canonical LF-only, NUL-free text")
    return value


def _identifier(value: str, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} cannot contain whitespace")
    return value


@dataclass(frozen=True, slots=True)
class TextNode:
    """Unannotated, detector-visible text."""

    text: str

    def __post_init__(self) -> None:
        _canonical_text(self.text, name="TextNode text")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class LiteralNode:
    """One exact surface occurrence of a typed state-node or edge value."""

    mention_id: str
    state_or_edge_id: str
    value: str
    target_kind: MutationTargetKind = MutationTargetKind.NODE

    def __post_init__(self) -> None:
        _identifier(self.mention_id, name="LiteralNode mention_id")
        _identifier(self.state_or_edge_id, name="LiteralNode state_or_edge_id")
        _canonical_text(self.value, name="LiteralNode value")
        if type(self.target_kind) is not MutationTargetKind:
            raise TypeError("LiteralNode target_kind must be a MutationTargetKind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "literal",
            "mention_id": self.mention_id,
            "state_or_edge_id": self.state_or_edge_id,
            "target_kind": self.target_kind.value,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class SequenceNode:
    """An ordered, separator-free sequence of trace nodes."""

    children: tuple[TraceNode, ...]

    def __post_init__(self) -> None:
        if isinstance(self.children, (str, bytes)) or not isinstance(
            self.children, SequenceABC
        ):
            raise TypeError("SequenceNode children must be a sequence")
        children = tuple(self.children)
        if not children:
            raise ValueError("SequenceNode children cannot be empty")
        if any(
            type(child) not in {TextNode, LiteralNode, SequenceNode, ClaimNode}
            for child in children
        ):
            raise TypeError("SequenceNode children must be trace AST nodes")
        object.__setattr__(self, "children", children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sequence",
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class ClaimNode:
    """One atomic proposition whose literals receive a shared claim span."""

    claim_id: str
    content: SequenceNode

    def __post_init__(self) -> None:
        _identifier(self.claim_id, name="ClaimNode claim_id")
        if type(self.content) is not SequenceNode:
            raise TypeError("ClaimNode content must be a SequenceNode")
        if not _contains_literal(self.content):
            raise ValueError("ClaimNode must contain at least one LiteralNode")

    @classmethod
    def from_template(
        cls,
        claim_id: str,
        template: str,
        slots: Mapping[str, LiteralNode],
    ) -> ClaimNode:
        """Fill named slots into an atomic claim without post-render lookup."""

        _canonical_text(template, name="claim template")
        if not isinstance(slots, Mapping):
            raise TypeError("claim template slots must be a mapping")
        if any(type(name) is not str or not name for name in slots):
            raise TypeError("claim template slot names must be non-empty strings")
        if any(type(slot) is not LiteralNode for slot in slots.values()):
            raise TypeError("claim template slots must contain LiteralNode values")
        children: list[TraceNode] = []
        used: list[str] = []
        for literal, field_name, format_spec, conversion in Formatter().parse(template):
            if literal:
                children.append(TextNode(literal))
            if field_name is None:
                continue
            if not field_name or format_spec or conversion:
                raise ValueError(
                    "claim templates require simple named placeholders without formatting"
                )
            if field_name not in slots:
                raise ValueError(f"claim template has no bound slot {field_name!r}")
            if field_name in used:
                raise ValueError("each claim template placeholder may occur only once")
            used.append(field_name)
            children.append(slots[field_name])
        unused = sorted(set(slots) - set(used))
        if unused:
            raise ValueError(f"claim template has unused slots: {unused}")
        if not used:
            raise ValueError("claim template must contain at least one placeholder")
        return cls(claim_id=claim_id, content=SequenceNode(tuple(children)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "claim",
            "claim_id": self.claim_id,
            "content": self.content.to_dict(),
        }


TraceNode: TypeAlias = TextNode | LiteralNode | SequenceNode | ClaimNode


def _contains_literal(node: TraceNode) -> bool:
    if type(node) is LiteralNode:
        return True
    if type(node) is TextNode:
        return False
    if type(node) is ClaimNode:
        return _contains_literal(node.content)
    return any(_contains_literal(child) for child in node.children)


@dataclass(frozen=True, slots=True)
class StepDocument:
    """One reasoning segment in the trace AST."""

    step_index: int
    content: SequenceNode
    segment_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("StepDocument step_index must be positive")
        if type(self.content) is not SequenceNode:
            raise TypeError("StepDocument content must be a SequenceNode")
        segment_id = self.segment_id or f"reasoning.step.{self.step_index:02d}"
        _identifier(segment_id, name="StepDocument segment_id")
        object.__setattr__(self, "segment_id", segment_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "segment_id": self.segment_id,
            "content": self.content.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AnswerDocument:
    """The final-answer segment, kept distinct from reasoning product claims."""

    content: SequenceNode
    segment_id: str = "final_answer"

    def __post_init__(self) -> None:
        if type(self.content) is not SequenceNode:
            raise TypeError("AnswerDocument content must be a SequenceNode")
        _identifier(self.segment_id, name="AnswerDocument segment_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "content": self.content.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TraceDocument:
    """A complete reasoning trace and final answer in deterministic order."""

    steps: tuple[StepDocument, ...]
    answer: AnswerDocument
    segment_separator: str = "\n\n"
    ast_version: str = TRACE_AST_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.steps, (str, bytes)) or not isinstance(
            self.steps, SequenceABC
        ):
            raise TypeError("TraceDocument steps must be a sequence")
        steps = tuple(self.steps)
        if not steps or any(type(step) is not StepDocument for step in steps):
            raise TypeError("TraceDocument steps must contain StepDocument values")
        if tuple(step.step_index for step in steps) != tuple(range(1, len(steps) + 1)):
            raise ValueError("TraceDocument steps must be contiguous and one-based")
        if type(self.answer) is not AnswerDocument:
            raise TypeError("TraceDocument answer must be an AnswerDocument")
        _canonical_text(self.segment_separator, name="TraceDocument segment_separator")
        if type(self.ast_version) is not str:
            raise TypeError("TraceDocument ast_version must be a string")
        if self.ast_version != TRACE_AST_VERSION:
            raise ValueError("unsupported trace AST version")
        segment_ids = tuple(step.segment_id for step in steps) + (
            self.answer.segment_id,
        )
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("TraceDocument segment IDs must be unique")
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ast_version": self.ast_version,
            "segment_separator": self.segment_separator,
            "steps": [step.to_dict() for step in self.steps],
            "answer": self.answer.to_dict(),
        }


def _freeze_span_mapping(
    values: Mapping[str, CharSpan], *, name: str
) -> FrozenMap[str, CharSpan]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(key) is not str or not key for key in values):
        raise TypeError(f"{name} keys must be non-empty strings")
    if any(type(span) is not CharSpan for span in values.values()):
        raise TypeError(f"{name} values must be CharSpan values")
    return FrozenMap(dict(sorted(values.items())))


def _freeze_mentions_mapping(
    values: Mapping[str, tuple[CharSpan, ...]], *, name: str
) -> FrozenMap[str, tuple[CharSpan, ...]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, tuple[CharSpan, ...]] = {}
    for key, raw_spans in values.items():
        if type(key) is not str or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        if isinstance(raw_spans, (str, bytes)) or not isinstance(
            raw_spans, SequenceABC
        ):
            raise TypeError(f"{name} values must be span sequences")
        spans = tuple(raw_spans)
        if not spans or any(type(span) is not CharSpan for span in spans):
            raise TypeError(f"{name} values must contain CharSpan values")
        normalized[key] = tuple(sorted(spans))
    return FrozenMap(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class RenderedExample:
    """Rendered trace text plus exact segment and occurrence-level offsets."""

    detector_text: str
    segment_spans: Mapping[str, CharSpan]
    node_mentions: Mapping[str, tuple[CharSpan, ...]]
    edge_mentions: Mapping[str, tuple[CharSpan, ...]]
    mentions: tuple[MentionSpan, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _canonical_text(self.detector_text, name="RenderedExample detector_text")
        segments = _freeze_span_mapping(self.segment_spans, name="segment_spans")
        nodes = _freeze_mentions_mapping(self.node_mentions, name="node_mentions")
        edges = _freeze_mentions_mapping(self.edge_mentions, name="edge_mentions")
        for span in segments.values():
            validate_char_span(self.detector_text, span)
        validate_non_overlapping_spans(segments.values(), name="segment spans")
        for mapping in (nodes, edges):
            for spans in mapping.values():
                for span in spans:
                    validate_char_span(self.detector_text, span)

        mentions = validate_mention_spans(self.detector_text, self.mentions)
        if mentions:
            expected_nodes = _group_literal_spans(
                mentions, target_kind=MutationTargetKind.NODE
            )
            expected_edges = _group_literal_spans(
                mentions, target_kind=MutationTargetKind.EDGE
            )
            if dict(nodes) != expected_nodes or dict(edges) != expected_edges:
                raise ValueError("mention maps must exactly match occurrence metadata")
            for mention in mentions:
                containing_segments = tuple(
                    segment_id
                    for segment_id, segment_span in segments.items()
                    if segment_span.start <= mention.claim_span.start
                    and mention.claim_span.end <= segment_span.end
                )
                if len(containing_segments) != 1:
                    raise ValueError(
                        "every claim span must belong to exactly one segment"
                    )
                expected_segment = (
                    f"reasoning.step.{mention.step_index:02d}"
                    if mention.component is SegmentKind.REASONING
                    else "final_answer"
                )
                if expected_segment in segments and containing_segments != (
                    expected_segment,
                ):
                    raise ValueError("mention component does not match its segment")

        object.__setattr__(self, "segment_spans", segments)
        object.__setattr__(self, "node_mentions", nodes)
        object.__setattr__(self, "edge_mentions", edges)
        object.__setattr__(self, "mentions", mentions)

    def mention(self, mention_id: str) -> MentionSpan:
        _identifier(mention_id, name="mention_id")
        for mention in self.mentions:
            if mention.mention_id == mention_id:
                return mention
        raise KeyError(mention_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_text": self.detector_text,
            "segment_spans": {
                key: {"start": span.start, "end": span.end}
                for key, span in self.segment_spans.items()
            },
            "node_mentions": {
                key: [{"start": span.start, "end": span.end} for span in spans]
                for key, spans in self.node_mentions.items()
            },
            "edge_mentions": {
                key: [{"start": span.start, "end": span.end} for span in spans]
                for key, spans in self.edge_mentions.items()
            },
            "mentions": [mention.to_dict() for mention in self.mentions],
        }


@dataclass(frozen=True, slots=True)
class _PendingMention:
    literal: LiteralNode
    literal_span: CharSpan


class _RenderBuffer:
    __slots__ = ("cursor", "parts")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.cursor = 0

    def append(self, text: str) -> CharSpan:
        _canonical_text(text, name="rendered fragment")
        start = self.cursor
        self.parts.append(text)
        self.cursor += len(text)
        return CharSpan(start, self.cursor)

    def text(self) -> str:
        return "".join(self.parts)


def _render_node(
    node: TraceNode,
    *,
    buffer: _RenderBuffer,
    component: SegmentKind,
    step_index: int | None,
    active_claim: str | None,
    mentions: list[MentionSpan],
) -> tuple[_PendingMention, ...]:
    if type(node) is TextNode:
        buffer.append(node.text)
        return ()
    if type(node) is LiteralNode:
        if active_claim is None:
            raise ValueError("LiteralNode must be contained by a ClaimNode")
        return (_PendingMention(node, buffer.append(node.value)),)
    if type(node) is SequenceNode:
        pending: list[_PendingMention] = []
        for child in node.children:
            pending.extend(
                _render_node(
                    child,
                    buffer=buffer,
                    component=component,
                    step_index=step_index,
                    active_claim=active_claim,
                    mentions=mentions,
                )
            )
        return tuple(pending)
    if active_claim is not None:
        raise ValueError("ClaimNode values cannot be nested")
    claim_start = buffer.cursor
    pending = _render_node(
        node.content,
        buffer=buffer,
        component=component,
        step_index=step_index,
        active_claim=node.claim_id,
        mentions=mentions,
    )
    claim_span = CharSpan(claim_start, buffer.cursor)
    for item in pending:
        mentions.append(
            MentionSpan(
                mention_id=item.literal.mention_id,
                claim_id=node.claim_id,
                state_or_edge_id=item.literal.state_or_edge_id,
                target_kind=item.literal.target_kind,
                component=component,
                step_index=step_index,
                literal_text=item.literal.value,
                literal_span=item.literal_span,
                claim_span=claim_span,
            )
        )
    return ()


def _group_literal_spans(
    mentions: SequenceABC[MentionSpan],
    *,
    target_kind: MutationTargetKind,
) -> dict[str, tuple[CharSpan, ...]]:
    grouped: defaultdict[str, list[CharSpan]] = defaultdict(list)
    for mention in mentions:
        if mention.target_kind is target_kind:
            grouped[mention.state_or_edge_id].append(mention.literal_span)
    return {
        target_id: tuple(sorted(grouped[target_id])) for target_id in sorted(grouped)
    }


class TraceASTRenderer:
    """Render one immutable trace AST and bind all offsets in one pass."""

    __slots__ = ()

    def render(self, document: TraceDocument) -> RenderedExample:
        if type(document) is not TraceDocument:
            raise TypeError("TraceASTRenderer requires a TraceDocument")
        buffer = _RenderBuffer()
        segments: dict[str, CharSpan] = {}
        mentions: list[MentionSpan] = []

        for index, step in enumerate(document.steps):
            if index:
                buffer.append(document.segment_separator)
            start = buffer.cursor
            pending = _render_node(
                step.content,
                buffer=buffer,
                component=SegmentKind.REASONING,
                step_index=step.step_index,
                active_claim=None,
                mentions=mentions,
            )
            if pending:
                raise AssertionError("unbound reasoning literal")
            segments[step.segment_id] = CharSpan(start, buffer.cursor)

        buffer.append(document.segment_separator)
        answer_start = buffer.cursor
        pending = _render_node(
            document.answer.content,
            buffer=buffer,
            component=SegmentKind.FINAL_ANSWER,
            step_index=None,
            active_claim=None,
            mentions=mentions,
        )
        if pending:
            raise AssertionError("unbound final-answer literal")
        segments[document.answer.segment_id] = CharSpan(answer_start, buffer.cursor)

        text = buffer.text()
        ordered_mentions = validate_mention_spans(text, mentions)
        return RenderedExample(
            detector_text=text,
            segment_spans=segments,
            node_mentions=_group_literal_spans(
                ordered_mentions, target_kind=MutationTargetKind.NODE
            ),
            edge_mentions=_group_literal_spans(
                ordered_mentions, target_kind=MutationTargetKind.EDGE
            ),
            mentions=ordered_mentions,
        )


def render_trace(document: TraceDocument) -> RenderedExample:
    """Render ``document`` with the stateless deterministic renderer."""

    return TraceASTRenderer().render(document)


# Concise aliases for callers that build template ASTs directly.
Text = TextNode
Literal = LiteralNode
Claim = ClaimNode
Sequence = SequenceNode


__all__ = [
    "TRACE_AST_VERSION",
    "AnswerDocument",
    "Claim",
    "ClaimNode",
    "Literal",
    "LiteralNode",
    "RenderedExample",
    "Sequence",
    "SequenceNode",
    "StepDocument",
    "Text",
    "TextNode",
    "TraceASTRenderer",
    "TraceDocument",
    "TraceNode",
    "render_trace",
]
