"""Contract tests for post-token resid-post activation/label alignment."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from molhallulens.config.loader import load_config_bundle
from molhallulens.core import (
    CausalRole,
    EditErrorSubtype,
    HallucinationType,
    SegmentKind,
    TokenLabelSet,
    TokenizerFingerprint,
)


def _zeros(length: int) -> tuple[int, ...]:
    return (0,) * length


def _post_token_labels(
    *,
    leading_special_attention: int = 1,
    trailing_special_attention: int = 0,
    padding_attention: int = 0,
) -> TokenLabelSet:
    length = 8
    return TokenLabelSet(
        activation_alignment="post_token_h_t",
        tokenizer_fingerprint=TokenizerFingerprint(
            tokenizer_name="ChemDFM-R-14B",
            tokenizer_revision="frozen-revision",
            tokenizer_vocab_hash="vocab-sha256",
            special_token_config={"bos_token_id": 1, "pad_token_id": 0},
            normalization_config={"normalizer": "none"},
        ),
        serialized_text_sha256="serialized-sha256",
        input_ids=(1, 101, 201, 102, 103, 104, 2, 0),
        attention_mask=(
            leading_special_attention,
            1,
            1,
            1,
            1,
            1,
            trailing_special_attention,
            padding_attention,
        ),
        offset_mapping=(
            (0, 0),
            (0, 4),
            (4, 5),
            (5, 10),
            (11, 20),
            (21, 25),
            (0, 0),
            (0, 0),
        ),
        segment_ids=(
            SegmentKind.SPECIAL,
            SegmentKind.SOURCE,
            SegmentKind.SPECIAL,
            SegmentKind.INSTRUCTION,
            SegmentKind.REASONING,
            SegmentKind.FINAL_ANSWER,
            SegmentKind.SPECIAL,
            SegmentKind.PADDING,
        ),
        evaluation_mask=(0, 0, 0, 0, 1, 1, 0, 0),
        hallucination_core_mask=_zeros(length),
        error_any_mask=_zeros(length),
        semantic_type_masks={label: _zeros(length) for label in HallucinationType},
        edit_subtype_masks={label: _zeros(length) for label in EditErrorSubtype},
        causal_role_masks={label: _zeros(length) for label in CausalRole},
        local_falsehood_mask=_zeros(length),
        off_task_branch_mask=_zeros(length),
        reasoning_mask=(0, 0, 0, 0, 1, 0, 0, 0),
        answer_mask=(0, 0, 0, 0, 0, 1, 0, 0),
        boundary_ambiguous_mask=_zeros(length),
        error_char_fraction=(0.0,) * length,
    )


def test_post_token_contract_uses_exact_token_indices_without_label_shift() -> None:
    labels = _post_token_labels()
    indexed_labels = tuple(
        (
            index,
            labels.input_ids[index],
            labels.segment_ids[index],
            labels.evaluation_mask[index],
        )
        for index in range(len(labels.input_ids))
    )

    assert labels.activation_alignment == "post_token_h_t"
    assert indexed_labels[4] == (4, 103, SegmentKind.REASONING, 1)
    assert indexed_labels[5] == (5, 104, SegmentKind.FINAL_ANSWER, 1)


def test_evaluation_mask_ignores_prefix_special_and_padding_segments() -> None:
    labels = _post_token_labels()

    assert labels.segment_ids == (
        SegmentKind.SPECIAL,
        SegmentKind.SOURCE,
        SegmentKind.SPECIAL,
        SegmentKind.INSTRUCTION,
        SegmentKind.REASONING,
        SegmentKind.FINAL_ANSWER,
        SegmentKind.SPECIAL,
        SegmentKind.PADDING,
    )
    assert labels.attention_mask == (1, 1, 1, 1, 1, 1, 0, 0)
    assert labels.evaluation_mask == (0, 0, 0, 0, 1, 1, 0, 0)
    assert labels.reasoning_mask == (0, 0, 0, 0, 1, 0, 0, 0)
    assert labels.answer_mask == (0, 0, 0, 0, 0, 1, 0, 0)
    assert labels.offset_mapping[2] == (4, 5)


@pytest.mark.parametrize(
    ("leading_special_attention", "trailing_special_attention", "padding_attention"),
    ((0, 0, 0), (1, 1, 1)),
)
def test_special_and_padding_attention_follow_model_config_but_stay_ignored(
    leading_special_attention: int,
    trailing_special_attention: int,
    padding_attention: int,
) -> None:
    labels = _post_token_labels(
        leading_special_attention=leading_special_attention,
        trailing_special_attention=trailing_special_attention,
        padding_attention=padding_attention,
    )

    assert tuple(labels.attention_mask[index] for index in (0, 2, 6, 7)) == (
        leading_special_attention,
        1,
        trailing_special_attention,
        padding_attention,
    )
    assert tuple(
        labels.evaluation_mask[index]
        for index, segment in enumerate(labels.segment_ids)
        if segment in {SegmentKind.SPECIAL, SegmentKind.PADDING}
    ) == (0, 0, 0, 0)


def test_every_token_array_and_axis_mask_has_the_same_length() -> None:
    labels = _post_token_labels()
    token_count = len(labels.input_ids)
    direct_arrays = (
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "segment_ids",
        "evaluation_mask",
        "hallucination_core_mask",
        "error_any_mask",
        "local_falsehood_mask",
        "off_task_branch_mask",
        "reasoning_mask",
        "answer_mask",
        "boundary_ambiguous_mask",
        "error_char_fraction",
    )

    assert all(len(getattr(labels, name)) == token_count for name in direct_arrays)
    assert all(
        len(mask) == token_count
        for field_name in (
            "semantic_type_masks",
            "edit_subtype_masks",
            "causal_role_masks",
        )
        for mask in getattr(labels, field_name).values()
    )
    assert {field.name for field in fields(labels)} >= set(direct_arrays)


def _label_values(labels: TokenLabelSet) -> dict[str, object]:
    return {field.name: getattr(labels, field.name) for field in fields(labels)}


@pytest.mark.parametrize(
    "field_name",
    (
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "segment_ids",
        "evaluation_mask",
        "hallucination_core_mask",
        "error_any_mask",
        "local_falsehood_mask",
        "off_task_branch_mask",
        "reasoning_mask",
        "answer_mask",
        "boundary_ambiguous_mask",
        "error_char_fraction",
    ),
)
@pytest.mark.parametrize("delta", (-1, 1))
def test_each_direct_token_array_rejects_length_drift(
    field_name: str,
    delta: int,
) -> None:
    labels = _post_token_labels()
    values = _label_values(labels)
    original = tuple(values[field_name])  # type: ignore[arg-type]
    values[field_name] = original[:-1] if delta < 0 else original + (original[-1],)

    with pytest.raises(ValueError, match="length|token length|match token|per token"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "key"),
    (
        ("semantic_type_masks", HallucinationType.CONTRADICTION),
        ("edit_subtype_masks", EditErrorSubtype.ANCHOR_GROUNDING),
        ("causal_role_masks", CausalRole.ROOT),
    ),
)
@pytest.mark.parametrize("delta", (-1, 1))
def test_each_taxonomy_axis_rejects_mask_length_drift(
    field_name: str,
    key: object,
    delta: int,
) -> None:
    labels = _post_token_labels()
    values = _label_values(labels)
    masks = dict(values[field_name])  # type: ignore[arg-type]
    original = tuple(masks[key])
    masks[key] = original[:-1] if delta < 0 else original + (original[-1],)
    values[field_name] = masks

    with pytest.raises(ValueError, match="length"):
        TokenLabelSet(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("activation_alignment", "error_type"),
    (
        ("", ValueError),
        ("pre_token_h_t_minus_1", ValueError),
        ("post_token_h_t_plus_1", ValueError),
        (None, TypeError),
        (7, TypeError),
    ),
)
def test_noncanonical_alignment_names_fail_closed(
    activation_alignment: object,
    error_type: type[Exception],
) -> None:
    labels = _post_token_labels()

    with pytest.raises(error_type, match="post_token_h_t|must be a string"):
        replace(labels, activation_alignment=activation_alignment)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("index", "replacement"),
    ((0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 0), (6, 1), (7, 1)),
)
def test_shifted_evaluation_mask_fails_closed(index: int, replacement: int) -> None:
    labels = _post_token_labels()
    shifted = list(labels.evaluation_mask)
    shifted[index] = replacement

    with pytest.raises(ValueError, match="evaluation_mask"):
        replace(labels, evaluation_mask=tuple(shifted))


@pytest.mark.parametrize("index", (1, 3, 4, 5))
def test_visible_text_segments_require_attention(index: int) -> None:
    labels = _post_token_labels()
    attention = list(labels.attention_mask)
    attention[index] = 0

    with pytest.raises(ValueError, match="attention_mask"):
        replace(labels, attention_mask=tuple(attention))


def _with_adjudicated_positive(
    labels: TokenLabelSet,
    *,
    index: int,
) -> TokenLabelSet:
    length = len(labels.input_ids)
    one_hot = tuple(int(token_index == index) for token_index in range(length))
    semantic = {label: _zeros(length) for label in HallucinationType}
    semantic[HallucinationType.CONTRADICTION] = one_hot
    edit = {label: _zeros(length) for label in EditErrorSubtype}
    roles = {label: _zeros(length) for label in CausalRole}
    if labels.segment_ids[index] is SegmentKind.FINAL_ANSWER:
        edit[EditErrorSubtype.FINAL_ANSWER_IDENTITY] = one_hot
        roles[CausalRole.TERMINAL] = one_hot
    else:
        edit[EditErrorSubtype.ANCHOR_GROUNDING] = one_hot
        roles[CausalRole.ROOT] = one_hot
    return replace(
        labels,
        hallucination_core_mask=one_hot,
        error_any_mask=one_hot,
        semantic_type_masks=semantic,
        edit_subtype_masks=edit,
        causal_role_masks=roles,
        local_falsehood_mask=one_hot,
        error_char_fraction=tuple(float(value) for value in one_hot),
    )


@pytest.mark.parametrize("index", (4, 5))
def test_reasoning_and_answer_tokens_accept_positive_labels(index: int) -> None:
    labels = _with_adjudicated_positive(_post_token_labels(), index=index)

    assert labels.positive_label_indices == (index,)
    assert labels.evaluation_mask[index] == 1


@pytest.mark.parametrize("index", (0, 1, 2, 3, 6, 7))
def test_ignored_segments_reject_positive_labels(index: int) -> None:
    with pytest.raises(ValueError, match="evaluated, attended"):
        _with_adjudicated_positive(_post_token_labels(), index=index)


def test_post_token_contract_matches_both_frozen_configs() -> None:
    bundle = load_config_bundle()

    assert bundle.dataset.detector.activation_alignment == "post_token_h_t"
    assert bundle.labels.canonical_annotation.activation_alignment == "post_token_h_t"
    assert bundle.dataset.detector.ignored_evaluation_segments == (
        "source",
        "instruction",
        "special",
        "padding",
    )
    assert bundle.dataset.detector.evaluated_segments == (
        "reasoning_chain",
        "final_answer",
    )
    assert tuple(SegmentKind) == (
        SegmentKind.SOURCE,
        SegmentKind.INSTRUCTION,
        SegmentKind.REASONING,
        SegmentKind.FINAL_ANSWER,
        SegmentKind.SPECIAL,
        SegmentKind.PADDING,
    )
