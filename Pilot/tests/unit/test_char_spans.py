"""T039 strict character-span contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from molhallulens.modules.text_realization.spans import (
    CharSpan,
    MentionSpan,
    span_text,
    validate_char_span,
    validate_mention_spans,
    validate_non_overlapping_spans,
)
from molhallulens.core import MutationTargetKind, SegmentKind


def _mention(
    *,
    mention_id: str = "m.1",
    claim_id: str = "claim.1",
    literal_span: CharSpan | None = None,
    claim_span: CharSpan | None = None,
    literal_text: str = "氯",
) -> MentionSpan:
    return MentionSpan(
        mention_id=mention_id,
        claim_id=claim_id,
        state_or_edge_id="add_fragment",
        target_kind=MutationTargetKind.NODE,
        component=SegmentKind.REASONING,
        step_index=1,
        literal_text=literal_text,
        literal_span=literal_span or CharSpan(2, 3),
        claim_span=claim_span or CharSpan(0, 4),
    )


def test_offsets_are_half_open_python_character_offsets_for_unicode() -> None:
    text = "加 氯 原子"
    mention = _mention()

    mention.validate_against(text)

    assert len(text.encode("utf-8")) > len(text)
    assert mention.literal_span == CharSpan(2, 3)
    assert span_text(text, mention.literal_span) == "氯"
    assert mention.literal_span.length == 1
    with pytest.raises(FrozenInstanceError):
        mention.literal_span.end = 9  # type: ignore[misc]


def test_empty_out_of_bounds_and_wrong_literal_spans_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty half-open"):
        CharSpan(1, 1)
    with pytest.raises(ValueError, match="outside"):
        validate_char_span("abc", CharSpan(2, 4))
    with pytest.raises(ValueError, match="expected text"):
        validate_char_span("abc", CharSpan(1, 2), expected_text="c")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_char_span("abc", CharSpan(1, 2), expected_text="")
    with pytest.raises(ValueError, match="contained"):
        replace(_mention(), claim_span=CharSpan(0, 2))


def test_literal_overlap_is_rejected_but_one_claim_can_have_two_literals() -> None:
    text = "A=12"
    first = _mention(
        mention_id="m.a",
        literal_text="1",
        literal_span=CharSpan(2, 3),
        claim_span=CharSpan(0, 4),
    )
    second = _mention(
        mention_id="m.b",
        literal_text="2",
        literal_span=CharSpan(3, 4),
        claim_span=CharSpan(0, 4),
    )
    assert validate_mention_spans(text, (second, first)) == (first, second)

    overlapping = replace(
        second,
        literal_text="12",
        literal_span=CharSpan(2, 4),
    )
    with pytest.raises(ValueError, match="literal spans must not overlap"):
        validate_mention_spans(text, (first, overlapping))


def test_distinct_claims_cannot_overlap_or_reuse_one_claim_id_at_new_bounds() -> None:
    text = "abcdef"
    first = _mention(
        mention_id="m.a",
        claim_id="claim.a",
        literal_text="b",
        literal_span=CharSpan(1, 2),
        claim_span=CharSpan(0, 3),
    )
    crossed = _mention(
        mention_id="m.b",
        claim_id="claim.b",
        literal_text="d",
        literal_span=CharSpan(3, 4),
        claim_span=CharSpan(2, 5),
    )
    with pytest.raises(ValueError, match="claim spans must not overlap"):
        validate_mention_spans(text, (first, crossed))

    reused = replace(crossed, claim_id="claim.a")
    with pytest.raises(ValueError, match="exactly one claim span"):
        validate_mention_spans(text, (first, reused))


def test_duplicate_mentions_and_overlapping_collections_are_rejected() -> None:
    text = "A B"
    first = _mention(
        mention_id="same",
        literal_text="A",
        literal_span=CharSpan(0, 1),
        claim_span=CharSpan(0, 1),
    )
    second = replace(
        first,
        literal_text="B",
        literal_span=CharSpan(2, 3),
        claim_span=CharSpan(2, 3),
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_mention_spans(text, (first, second))
    with pytest.raises(ValueError, match="must not overlap"):
        validate_non_overlapping_spans(
            (CharSpan(0, 2), CharSpan(1, 3)), name="adversarial spans"
        )


def test_component_step_contract_and_stable_serialization() -> None:
    with pytest.raises(ValueError, match="require a step_index"):
        replace(_mention(), step_index=None)
    with pytest.raises(ValueError, match="cannot have a step_index"):
        replace(_mention(), component=SegmentKind.FINAL_ANSWER)

    payload = _mention().to_dict()
    assert tuple(payload) == (
        "mention_id",
        "claim_id",
        "state_or_edge_id",
        "target_kind",
        "component",
        "step_index",
        "literal_text",
        "literal_span",
        "claim_span",
    )
    assert payload["literal_span"] == {"start": 2, "end": 3}
