"""T039 immutable trace AST and deterministic occurrence offsets."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from molhallulens.modules.text_realization.spans import CharSpan
from molhallulens.core import MutationTargetKind
from molhallulens.modules.text_realization.trace_ast import (
    AnswerDocument,
    ClaimNode,
    LiteralNode,
    RenderedExample,
    SequenceNode,
    StepDocument,
    TextNode,
    TraceASTRenderer,
    TraceDocument,
    render_trace,
)


def _claim(
    claim_id: str,
    mention_id: str,
    *,
    target_id: str,
    value: str,
    prefix: str,
    target_kind: MutationTargetKind = MutationTargetKind.NODE,
) -> ClaimNode:
    return ClaimNode.from_template(
        claim_id,
        prefix + "{value}",
        {
            "value": LiteralNode(
                mention_id=mention_id,
                state_or_edge_id=target_id,
                value=value,
                target_kind=target_kind,
            )
        },
    )


def _document() -> TraceDocument:
    repeated_one = _claim(
        "claim.remove.step1",
        "mention.remove.1",
        target_id="remove_group_step1",
        value="Br",
        prefix="Remove group: ",
    )
    repeated_two = _claim(
        "claim.remove.step2",
        "mention.remove.2",
        target_id="remove_group_step2",
        value="Br",
        prefix="Confirm group: ",
    )
    edge = _claim(
        "claim.relation",
        "mention.edge.1",
        target_id="source->product:edit_produces",
        value="valid",
        prefix="Edit relation: ",
        target_kind=MutationTargetKind.EDGE,
    )
    answer = _claim(
        "claim.answer",
        "mention.answer.1",
        target_id="final_answer",
        value="Br",
        prefix="Answer: ",
    )
    return TraceDocument(
        steps=(
            StepDocument(1, SequenceNode((repeated_one,))),
            StepDocument(
                2,
                SequenceNode(
                    (
                        TextNode("步骤二 — "),
                        repeated_two,
                        TextNode("; "),
                        edge,
                    )
                ),
            ),
        ),
        answer=AnswerDocument(SequenceNode((answer,))),
    )


def test_repeated_surface_values_receive_independent_occurrence_spans() -> None:
    rendered = render_trace(_document())

    assert rendered.detector_text == (
        "Remove group: Br\n\n"
        "步骤二 — Confirm group: Br; Edit relation: valid\n\n"
        "Answer: Br"
    )
    occurrences = tuple(
        mention for mention in rendered.mentions if mention.literal_text == "Br"
    )
    assert len(occurrences) == 3
    assert len({mention.literal_span for mention in occurrences}) == 3
    assert all(
        rendered.detector_text[mention.literal_span.start : mention.literal_span.end]
        == "Br"
        for mention in occurrences
    )
    assert (
        rendered.node_mentions["remove_group_step1"]
        != rendered.node_mentions["remove_group_step2"]
    )
    assert rendered.edge_mentions == {
        "source->product:edit_produces": (
            rendered.mention("mention.edge.1").literal_span,
        )
    }


def test_literal_and_context_claim_spans_are_both_emitted() -> None:
    rendered = render_trace(_document())
    mention = rendered.mention("mention.remove.2")

    assert (
        rendered.detector_text[mention.literal_span.start : mention.literal_span.end]
        == "Br"
    )
    assert (
        rendered.detector_text[mention.claim_span.start : mention.claim_span.end]
        == "Confirm group: Br"
    )
    assert mention.claim_span.start < mention.literal_span.start
    assert mention.literal_span.end == mention.claim_span.end


def test_two_literals_in_one_claim_share_context_but_not_literal_offsets() -> None:
    claim = ClaimNode.from_template(
        "claim.anchor",
        "the attachment atom is {element}{index}",
        {
            "element": LiteralNode("mention.element", "anchor_element", "N"),
            "index": LiteralNode("mention.index", "anchor_idx", "21"),
        },
    )
    document = TraceDocument(
        steps=(StepDocument(1, SequenceNode((claim,))),),
        answer=AnswerDocument(
            SequenceNode(
                (
                    _claim(
                        "claim.answer",
                        "mention.answer",
                        target_id="final_answer",
                        value="C",
                        prefix="Answer: ",
                    ),
                )
            )
        ),
    )

    rendered = TraceASTRenderer().render(document)
    element = rendered.mention("mention.element")
    index = rendered.mention("mention.index")
    assert element.claim_span == index.claim_span
    assert not element.literal_span.overlaps(index.literal_span)
    assert rendered.detector_text[
        element.claim_span.start : element.claim_span.end
    ] == ("the attachment atom is N21")


def test_unicode_offsets_count_python_characters_not_utf8_bytes() -> None:
    rendered = render_trace(_document())
    mention = rendered.mention("mention.remove.2")
    prefix = rendered.detector_text[: mention.literal_span.start]

    assert "步骤二" in prefix
    assert mention.literal_span.start == len(prefix)
    assert len(prefix.encode("utf-8")) > mention.literal_span.start


def test_ast_and_rendered_maps_are_immutable_and_serialization_is_stable() -> None:
    document = _document()
    rendered = render_trace(document)

    with pytest.raises(FrozenInstanceError):
        document.segment_separator = "\n"  # type: ignore[misc]
    with pytest.raises(TypeError):
        rendered.segment_spans["extra"] = CharSpan(0, 1)  # type: ignore[index]

    payload = rendered.to_dict()
    assert tuple(payload) == (
        "detector_text",
        "segment_spans",
        "node_mentions",
        "edge_mentions",
        "mentions",
    )
    assert tuple(payload["segment_spans"]) == (
        "final_answer",
        "reasoning.step.01",
        "reasoning.step.02",
    )
    assert document.to_dict() == _document().to_dict()
    assert rendered.to_dict() == render_trace(_document()).to_dict()


def test_ast_rejects_empty_invalid_nested_or_ambiguous_templates() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        TextNode("")
    with pytest.raises(ValueError, match="cannot be empty"):
        LiteralNode("m", "node", "")
    with pytest.raises(ValueError, match="cannot be empty"):
        SequenceNode(())
    with pytest.raises(ValueError, match="at least one LiteralNode"):
        ClaimNode("claim", SequenceNode((TextNode("context only"),)))
    with pytest.raises(ValueError, match="unused slots"):
        ClaimNode.from_template(
            "claim",
            "value={first}",
            {
                "first": LiteralNode("m.1", "node.1", "A"),
                "second": LiteralNode("m.2", "node.2", "B"),
            },
        )
    with pytest.raises(ValueError, match="only once"):
        ClaimNode.from_template(
            "claim",
            "{value} then {value}",
            {"value": LiteralNode("m.1", "node.1", "A")},
        )

    inner = _claim(
        "inner",
        "mention.inner",
        target_id="node.inner",
        value="A",
        prefix="inner=",
    )
    outer = ClaimNode(
        "outer",
        SequenceNode((TextNode("outer="), inner)),
    )
    document = TraceDocument(
        steps=(StepDocument(1, SequenceNode((outer,))),),
        answer=AnswerDocument(
            SequenceNode(
                (
                    _claim(
                        "answer",
                        "mention.answer",
                        target_id="final_answer",
                        value="C",
                        prefix="Answer: ",
                    ),
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="cannot be nested"):
        render_trace(document)


def test_duplicate_occurrence_ids_fail_even_when_values_are_repeated() -> None:
    first = _claim(
        "claim.1",
        "same.mention",
        target_id="node.1",
        value="7",
        prefix="first=",
    )
    second = _claim(
        "claim.2",
        "same.mention",
        target_id="node.2",
        value="7",
        prefix="second=",
    )
    document = TraceDocument(
        steps=(StepDocument(1, SequenceNode((first, TextNode("; "), second))),),
        answer=AnswerDocument(
            SequenceNode(
                (
                    _claim(
                        "answer",
                        "answer.mention",
                        target_id="final_answer",
                        value="C",
                        prefix="Answer: ",
                    ),
                )
            )
        ),
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        render_trace(document)


def test_rendered_example_rejects_overlap_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="segment spans must not overlap"):
        RenderedExample(
            detector_text="abcdef",
            segment_spans={"one": CharSpan(0, 4), "two": CharSpan(3, 6)},
            node_mentions={},
            edge_mentions={},
        )
    with pytest.raises(ValueError, match="outside"):
        RenderedExample(
            detector_text="abc",
            segment_spans={"one": CharSpan(0, 4)},
            node_mentions={},
            edge_mentions={},
        )


def test_renderer_contract_contains_no_truth_or_label_inference_inputs() -> None:
    document_fields = set(TraceDocument.__dataclass_fields__)
    renderer_parameters = set(inspect.signature(TraceASTRenderer.render).parameters)
    forbidden = {"gt_smiles", "oracle", "label", "operator_id", "correctness"}

    assert document_fields.isdisjoint(forbidden)
    assert renderer_parameters == {"self", "document"}
