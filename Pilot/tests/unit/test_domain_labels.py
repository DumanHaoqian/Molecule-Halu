"""Tests for orthogonal character and token label objects."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from molhallulens.domain.enums import (
    CausalRole,
    EditErrorSubtype,
    EvidenceRelation,
    HallucinationType,
    SegmentKind,
)
from molhallulens.domain.labels import (
    CharAnnotation,
    CharSpan,
    ClaimLabel,
    TokenLabelSet,
    TokenizerFingerprint,
)


def test_char_annotation_expresses_all_orthogonal_axes() -> None:
    annotation = CharAnnotation(
        span_id="span.root",
        component=SegmentKind.REASONING,
        step_index=1,
        state_or_edge_id="anchor.idx",
        literal_span=CharSpan(12, 14),
        claim_span=CharSpan(4, 20),
        semantic_types=frozenset(
            {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
        ),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        evidence_relations=frozenset(
            {EvidenceRelation.CONTRADICTS_INSTRUCTION, EvidenceRelation.CONTRADICTS_REFERENCE_STATE}
        ),
        causal_role=CausalRole.ROOT,
        root_span_id="span.root",
    )

    assert len(annotation.semantic_types) == 2
    assert len(annotation.evidence_relations) == 2
    assert annotation.literal_span.length == 2


def test_terminal_annotation_must_be_in_final_answer() -> None:
    with pytest.raises(ValueError, match="final answer"):
        CharAnnotation(
            span_id="terminal",
            component=SegmentKind.REASONING,
            step_index=None,
            state_or_edge_id="final_answer",
            literal_span=CharSpan(1, 2),
            claim_span=CharSpan(1, 2),
            semantic_types=frozenset({HallucinationType.CONTRADICTION}),
            edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
            evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
            causal_role=CausalRole.TERMINAL,
            root_span_id="terminal",
        )

    with pytest.raises(ValueError, match="target final_answer"):
        CharAnnotation(
            span_id="terminal",
            component=SegmentKind.FINAL_ANSWER,
            step_index=None,
            state_or_edge_id="anchor.idx",
            literal_span=CharSpan(1, 2),
            claim_span=CharSpan(1, 2),
            semantic_types=frozenset({HallucinationType.CONTRADICTION}),
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
            causal_role=CausalRole.TERMINAL,
            root_span_id="terminal",
        )


def _zero_masks(length: int) -> tuple[int, ...]:
    return (0,) * length


def _token_labels() -> TokenLabelSet:
    length = 4
    semantic = {label: _zero_masks(length) for label in HallucinationType}
    semantic[HallucinationType.CONTRADICTION] = (0, 0, 1, 0)
    semantic[HallucinationType.REASONING_ERROR] = (0, 0, 1, 0)
    edit = {label: _zero_masks(length) for label in EditErrorSubtype}
    edit[EditErrorSubtype.ANCHOR_GROUNDING] = (0, 0, 1, 0)
    roles = {label: _zero_masks(length) for label in CausalRole}
    roles[CausalRole.ROOT] = (0, 0, 1, 0)
    return TokenLabelSet(
        activation_alignment="post_token_h_t",
        tokenizer_fingerprint=TokenizerFingerprint(
            tokenizer_name="ChemDFM-R-14B",
            tokenizer_revision="frozen-revision",
            tokenizer_vocab_hash="abc123",
            special_token_config={"pad": 0},
            normalization_config={"normalizer": "none"},
        ),
        serialized_text_sha256="text-hash",
        input_ids=(10, 11, 12, 13),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 1), (1, 2), (2, 3), (3, 4)),
        segment_ids=(
            SegmentKind.SOURCE,
            SegmentKind.INSTRUCTION,
            SegmentKind.REASONING,
            SegmentKind.FINAL_ANSWER,
        ),
        evaluation_mask=(0, 0, 1, 1),
        hallucination_core_mask=(0, 0, 1, 0),
        error_any_mask=(0, 0, 1, 0),
        semantic_type_masks=semantic,
        edit_subtype_masks=edit,
        causal_role_masks=roles,
        local_falsehood_mask=(0, 0, 1, 0),
        off_task_branch_mask=(0, 0, 0, 0),
        reasoning_mask=(0, 0, 1, 0),
        answer_mask=(0, 0, 0, 1),
        boundary_ambiguous_mask=(0, 0, 0, 0),
        error_char_fraction=(0.0, 0.0, 1.0, 0.0),
    )


def _token_label_values(labels: TokenLabelSet) -> dict[str, object]:
    return {
        field: getattr(labels, field)
        for field in labels.__dataclass_fields__  # type: ignore[attr-defined]
    }


def test_token_label_set_is_post_token_aligned_and_deeply_frozen() -> None:
    labels = _token_labels()

    assert labels.hallucination_core_mask[2] == 1
    assert not isinstance(labels.semantic_type_masks, dict)
    with pytest.raises(TypeError):
        labels.semantic_type_masks[HallucinationType.OMISSION] = (1, 1, 1, 1)  # type: ignore[index]

    values = _token_label_values(labels)
    mutable_offsets = [[0, 1], [1, 2], [2, 3], [3, 4]]
    values["offset_mapping"] = mutable_offsets
    copied = TokenLabelSet(**values)  # type: ignore[arg-type]
    mutable_offsets[2][0] = 99
    assert copied.offset_mapping[2] == (2, 3)

    with pytest.raises(TypeError, match="string keys"):
        TokenizerFingerprint(
            "tokenizer",
            "revision",
            "hash",
            {1: "pad"},  # type: ignore[dict-item]
            {},
        )

    @dataclass(frozen=True, slots=True)
    class NestedConfig:
        payload: dict[object, object]

    with pytest.raises(TypeError, match="string keys"):
        TokenizerFingerprint(
            "tokenizer",
            "revision",
            "hash",
            {"nested": NestedConfig({1: "not-json-safe"})},
            {},
        )


def test_token_label_set_rejects_pre_token_or_shifted_masks() -> None:
    labels = _token_labels()
    values = _token_label_values(labels)
    values["activation_alignment"] = "pre_token_h_t_minus_1"
    with pytest.raises(ValueError, match="post_token"):
        TokenLabelSet(**values)

    values = _token_label_values(labels)
    values["input_ids"] = (-1, 11, 12, 13)
    with pytest.raises(ValueError, match="non-negative"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    values["activation_alignment"] = "post_token_h_t"
    values["evaluation_mask"] = (0, 1, 1, 1)
    with pytest.raises(ValueError, match="evaluation_mask"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    values["attention_mask"] = (1, 1, 0, 1)
    with pytest.raises(ValueError, match="attention_mask"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    values["offset_mapping"] = ((0, 1), (1, 2), (2, 2), (3, 4))
    with pytest.raises(ValueError, match="non-empty character offsets"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_token_label_set_rejects_labels_outside_visible_error_tokens() -> None:
    labels = _token_labels()
    values = _token_label_values(labels)
    semantic = {label: _zero_masks(4) for label in HallucinationType}
    semantic[HallucinationType.CONTRADICTION] = (1, 0, 1, 0)
    values.update(
        semantic_type_masks=semantic,
        hallucination_core_mask=(1, 0, 1, 0),
        error_any_mask=(1, 0, 1, 0),
        error_char_fraction=(1.0, 0.0, 1.0, 0.0),
    )
    with pytest.raises(ValueError, match="evaluated, attended"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    edit = dict(labels.edit_subtype_masks)
    edit[EditErrorSubtype.PRODUCT_CONSTRUCTION] = (0, 0, 0, 1)
    values["edit_subtype_masks"] = edit
    with pytest.raises(ValueError, match="editing subtype"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    roles = dict(labels.causal_role_masks)
    roles[CausalRole.TERMINAL] = (0, 0, 0, 1)
    values["causal_role_masks"] = roles
    with pytest.raises(ValueError, match="exactly one causal role"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_token_label_set_requires_real_enum_keys_and_segments() -> None:
    labels = _token_labels()
    values = _token_label_values(labels)
    segments = list(labels.segment_ids)
    segments[2] = "reasoning"  # type: ignore[list-item]
    values["segment_ids"] = segments
    with pytest.raises(TypeError, match="SegmentKind"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    semantic = dict(labels.semantic_type_masks)
    contradiction = semantic.pop(HallucinationType.CONTRADICTION)
    semantic[0] = contradiction  # type: ignore[index]
    values["semantic_type_masks"] = semantic
    with pytest.raises(TypeError, match="HallucinationType"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_unverifiable_is_non_adjudicated_but_still_token_localizable() -> None:
    annotation = CharAnnotation(
        span_id="span.uncertain",
        component=SegmentKind.REASONING,
        step_index=1,
        state_or_edge_id="claim.uncertain",
        literal_span=CharSpan(1, 2),
        claim_span=CharSpan(1, 2),
        semantic_types=frozenset({HallucinationType.UNVERIFIABLE}),
        edit_subtypes=frozenset(),
        evidence_relations=frozenset(),
        causal_role=None,
        root_span_id=None,
    )
    label = ClaimLabel(
        semantic_types=frozenset({HallucinationType.UNVERIFIABLE}),
        edit_subtypes=frozenset(),
        evidence_relations=frozenset(),
        causal_role=None,
        root_event_id=None,
    )
    assert annotation.causal_role is None
    assert label.root_event_id is None

    labels = _token_labels()
    values = _token_label_values(labels)
    semantic = dict(labels.semantic_type_masks)
    semantic[HallucinationType.UNVERIFIABLE] = (0, 0, 0, 1)
    values["semantic_type_masks"] = semantic
    values["error_char_fraction"] = (0.0, 0.0, 1.0, 1.0)
    projected = TokenLabelSet(**values)  # type: ignore[arg-type]
    assert projected.semantic_type_masks[HallucinationType.UNVERIFIABLE][3] == 1
    assert projected.error_any_mask[3] == 0

    values = _token_label_values(labels)
    semantic = dict(labels.semantic_type_masks)
    semantic[HallucinationType.UNVERIFIABLE] = (0, 0, 1, 0)
    values["semantic_type_masks"] = semantic
    with pytest.raises(ValueError, match="mutually exclusive"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_terminal_causal_mask_is_answer_only() -> None:
    labels = _token_labels()
    values = _token_label_values(labels)
    roles = dict(labels.causal_role_masks)
    roles[CausalRole.ROOT] = _zero_masks(4)
    roles[CausalRole.TERMINAL] = (0, 0, 1, 0)
    values["causal_role_masks"] = roles
    with pytest.raises(ValueError, match="final answer"):
        TokenLabelSet(**values)  # type: ignore[arg-type]

    values = _token_label_values(labels)
    roles = dict(labels.causal_role_masks)
    roles[CausalRole.ROOT] = _zero_masks(4)
    roles[CausalRole.TERMINAL] = (0, 0, 0, 1)
    semantic = {label: _zero_masks(4) for label in HallucinationType}
    semantic[HallucinationType.CONTRADICTION] = (0, 0, 0, 1)
    edit = dict(labels.edit_subtype_masks)
    edit[EditErrorSubtype.ANCHOR_GROUNDING] = (0, 0, 0, 1)
    values.update(
        causal_role_masks=roles,
        semantic_type_masks=semantic,
        edit_subtype_masks=edit,
        hallucination_core_mask=(0, 0, 0, 1),
        error_any_mask=(0, 0, 0, 1),
        local_falsehood_mask=(0, 0, 0, 1),
        error_char_fraction=(0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="FINAL_ANSWER_IDENTITY"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_matched_target_span_must_overlap_evaluated_text() -> None:
    labels = _token_labels()
    values = _token_label_values(labels)
    values["matched_target_span"] = CharSpan(1000, 1001)
    with pytest.raises(ValueError, match="overlap"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


def test_claim_label_requires_each_axis() -> None:
    with pytest.raises(ValueError, match="axes"):
        ClaimLabel(
            semantic_types=frozenset(),
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_SOURCE}),
            causal_role=CausalRole.ROOT,
            root_event_id="event.root",
        )


def test_annotations_reject_raw_string_enums() -> None:
    with pytest.raises(TypeError, match="component"):
        CharAnnotation(
            span_id="span.root",
            component="reasoning",  # type: ignore[arg-type]
            step_index=1,
            state_or_edge_id="anchor.idx",
            literal_span=CharSpan(1, 2),
            claim_span=CharSpan(1, 2),
            semantic_types=frozenset({HallucinationType.CONTRADICTION}),
            edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
            evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_SOURCE}),
            causal_role=CausalRole.ROOT,
            root_span_id="span.root",
        )

    with pytest.raises(TypeError, match="boundaries"):
        CharSpan(HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR)
