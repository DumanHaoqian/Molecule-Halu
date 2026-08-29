"""T042 ChemDFM-R fast-offset character-to-token projection tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from molhallulens.annotation.char_annotations import (
    CharAnnotationBuildResult,
    EventAnnotationLink,
    UnlocalizedOmission,
)
from molhallulens.annotation.token_projection import (
    ACTIVATION_ALIGNMENT,
    DetectorCoordinateMap,
    TokenLabelSetWriter,
    TokenProjectionError,
    project_char_annotations,
    rebase_char_annotations,
)
from molhallulens.domain import (
    CausalRole,
    CharAnnotation,
    CharSpan,
    EditErrorSubtype,
    EvidenceRelation,
    HallucinationType,
    MutationTargetKind,
    SegmentKind,
    TokenizerFingerprint,
    VariantLabel,
)
from molhallulens.rendering.detector_prompt import (
    DetectorPromptSerializer,
    SerializedDetectorInput,
)
from molhallulens.rendering.trace_ast import (
    AnswerDocument,
    ClaimNode,
    LiteralNode,
    RenderedExample,
    SequenceNode,
    StepDocument,
    TraceDocument,
    render_trace,
)

_UNSET = object()


def _serialized(
    *,
    reasoning: str = "原子 N21 is selected.",
    final_answer: str = "CO",
) -> SerializedDetectorInput:
    return DetectorPromptSerializer().serialize(
        indexed_smiles="[CH3:1][NH2:2]",
        instruction="Replace the selected atom.",
        reasoning_chain=reasoning,
        final_answer=final_answer,
    )


def _fingerprint(**special_overrides: object) -> TokenizerFingerprint:
    special = {"bos_token_id": 101, "eos_token_id": 102, "pad_token_id": 0}
    special.update(special_overrides)
    return TokenizerFingerprint(
        tokenizer_name="ChemDFM-R-14B",
        tokenizer_revision="frozen-revision",
        tokenizer_vocab_hash="frozen-vocabulary-identity",
        special_token_config=special,
        normalization_config={"normalizer": "none", "offset_unit": "python_char"},
    )


def _literal_span(
    serialized: SerializedDetectorInput,
    literal: str = "N21",
    *,
    prefix: str = "原子 ",
) -> CharSpan:
    reasoning = serialized.segments[2]
    start = reasoning.start + len(prefix)
    return CharSpan(start, start + len(literal))


def _root_annotation(
    serialized: SerializedDetectorInput,
    *,
    span_id: str = "char:root",
    literal_span: CharSpan | None = None,
    claim_span: CharSpan | None = None,
    component: SegmentKind = SegmentKind.REASONING,
    semantic_types: frozenset[HallucinationType] = frozenset(
        {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
    ),
    edit_subtypes: frozenset[EditErrorSubtype] = frozenset(
        {EditErrorSubtype.ANCHOR_GROUNDING}
    ),
    causal_role: CausalRole | None = CausalRole.ROOT,
    root_span_id: str | None | object = _UNSET,
    state_or_edge_id: str = "anchor_idx",
) -> CharAnnotation:
    span = literal_span or _literal_span(serialized)
    if component is SegmentKind.REASONING:
        segment = serialized.segments[2]
        step_index = 1
    else:
        segment = serialized.segments[3]
        step_index = None
    return CharAnnotation(
        span_id=span_id,
        component=component,
        step_index=step_index,
        state_or_edge_id=state_or_edge_id,
        literal_span=span,
        claim_span=claim_span or CharSpan(segment.start, segment.end),
        semantic_types=semantic_types,
        edit_subtypes=edit_subtypes,
        evidence_relations=(
            frozenset()
            if semantic_types == frozenset({HallucinationType.UNVERIFIABLE})
            else frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE})
        ),
        causal_role=causal_role,
        root_span_id=(
            root_span_id
            if root_span_id is not _UNSET
            else span_id
            if causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}
            else "char:root"
        ),  # type: ignore[arg-type]
    )


def _default_offsets(
    serialized: SerializedDetectorInput,
) -> tuple[tuple[int, int], ...]:
    source, instruction, _, answer = serialized.segments
    literal = _literal_span(serialized)
    return (
        (0, 0),
        (source.start, source.end),
        (instruction.start, instruction.end),
        (literal.start - 1, literal.end + 1),
        (answer.start, answer.end),
        (0, 0),
    )


@dataclass
class _FakeFastTokenizer:
    offsets: tuple[tuple[int, int], ...]
    output_overrides: dict[str, object] | None = None
    is_fast: bool = True
    bos_token_id: int = 101
    eos_token_id: int = 102
    pad_token_id: int = 0
    calls: int = 0
    received_text: str | None = None
    received_kwargs: dict[str, object] | None = None

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        self.received_text = text
        self.received_kwargs = dict(kwargs)
        length = len(self.offsets)
        output: dict[str, object] = {
            "input_ids": tuple(range(201, 201 + length)),
            "attention_mask": (1,) * length,
            "offset_mapping": self.offsets,
            "special_tokens_mask": tuple(
                int(offset == (0, 0)) for offset in self.offsets
            ),
            "input_text": text,
        }
        output.update(self.output_overrides or {})
        return output


def _writer(
    serialized: SerializedDetectorInput,
    *,
    tokenizer: _FakeFastTokenizer | None = None,
    fingerprint: TokenizerFingerprint | None = None,
) -> tuple[TokenLabelSetWriter, _FakeFastTokenizer]:
    selected = tokenizer or _FakeFastTokenizer(_default_offsets(serialized))
    return TokenLabelSetWriter(selected, fingerprint or _fingerprint()), selected


def _trace_local_fixture() -> tuple[
    RenderedExample,
    SerializedDetectorInput,
    CharAnnotation,
    CharAnnotation,
]:
    reasoning_claim = ClaimNode.from_template(
        "claim.reasoning",
        "原子 {atom} is selected.",
        {
            "atom": LiteralNode(
                mention_id="mention.reasoning.atom",
                state_or_edge_id="anchor_idx",
                value="N21",
            )
        },
    )
    answer_claim = ClaimNode.from_template(
        "claim.answer",
        "Answer: {answer}",
        {
            "answer": LiteralNode(
                mention_id="mention.answer",
                state_or_edge_id="final_answer",
                value="CO",
            )
        },
    )
    rendered = render_trace(
        TraceDocument(
            steps=(StepDocument(1, SequenceNode((reasoning_claim,))),),
            answer=AnswerDocument(SequenceNode((answer_claim,))),
        )
    )
    reasoning_trace = rendered.segment_spans["reasoning.step.01"]
    serialized = _serialized(
        reasoning=rendered.detector_text[reasoning_trace.start : reasoning_trace.end],
        final_answer="CO",
    )
    reasoning_mention = rendered.mention("mention.reasoning.atom")
    answer_mention = rendered.mention("mention.answer")
    root = CharAnnotation(
        span_id="char:mention.reasoning.atom",
        component=SegmentKind.REASONING,
        step_index=1,
        state_or_edge_id="anchor_idx",
        literal_span=reasoning_mention.literal_span,
        claim_span=reasoning_mention.claim_span,
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.ROOT,
        root_span_id="char:mention.reasoning.atom",
    )
    terminal = CharAnnotation(
        span_id="char:mention.answer",
        component=SegmentKind.FINAL_ANSWER,
        step_index=None,
        state_or_edge_id="final_answer",
        literal_span=answer_mention.literal_span,
        claim_span=answer_mention.claim_span,
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.TERMINAL,
        root_span_id="char:mention.answer",
    )
    return rendered, serialized, root, terminal


def test_any_overlap_projects_all_axes_fraction_and_boundary_ambiguity() -> None:
    serialized = _serialized()
    annotation = _root_annotation(serialized)
    writer, tokenizer = _writer(serialized)

    labels = writer.write(
        serialized,
        (annotation,),
        variant_label=VariantLabel.HALLUCINATED,
    )

    assert tokenizer.calls == 1
    assert tokenizer.received_text is serialized.text
    assert tokenizer.received_kwargs == {
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "truncation": False,
        "padding": False,
    }
    assert labels.activation_alignment == ACTIVATION_ALIGNMENT == "post_token_h_t"
    assert labels.serialized_text_sha256 == serialized.sha256
    assert labels.tokenizer_fingerprint == _fingerprint()
    assert labels.segment_ids == (
        SegmentKind.SPECIAL,
        SegmentKind.SOURCE,
        SegmentKind.INSTRUCTION,
        SegmentKind.REASONING,
        SegmentKind.FINAL_ANSWER,
        SegmentKind.SPECIAL,
    )
    positive = 3
    assert labels.evaluation_mask == (0, 0, 0, 1, 1, 0)
    assert labels.error_any_mask[positive] == 1
    assert labels.hallucination_core_mask[positive] == 1
    assert labels.semantic_type_masks[HallucinationType.CONTRADICTION][positive] == 1
    assert labels.semantic_type_masks[HallucinationType.REASONING_ERROR][positive] == 1
    assert labels.edit_subtype_masks[EditErrorSubtype.ANCHOR_GROUNDING][positive] == 1
    assert labels.causal_role_masks[CausalRole.ROOT][positive] == 1
    assert labels.local_falsehood_mask[positive] == 1
    assert labels.off_task_branch_mask[positive] == 0
    assert labels.boundary_ambiguous_mask[positive] == 1
    assert labels.error_char_fraction[positive] == pytest.approx(3 / 5)
    assert labels.reasoning_mask[positive] == 1
    assert labels.answer_mask[positive] == 0


def test_one_char_span_projects_to_every_token_with_positive_overlap() -> None:
    serialized = _serialized()
    literal = _literal_span(serialized)
    source, instruction, _, answer = serialized.segments
    offsets = (
        (0, 0),
        (source.start, source.end),
        (instruction.start, instruction.end),
        (literal.start - 1, literal.start + 1),
        (literal.start + 1, literal.end + 1),
        (answer.start, answer.end),
        (0, 0),
    )
    writer, _ = _writer(serialized, tokenizer=_FakeFastTokenizer(offsets))

    labels = writer.project(
        serialized,
        (_root_annotation(serialized),),
        variant_label=VariantLabel.HALLUCINATED,
    )

    assert labels.positive_label_indices == (3, 4)
    assert labels.error_char_fraction[3] == pytest.approx(1 / 2)
    assert labels.error_char_fraction[4] == pytest.approx(2 / 3)
    assert labels.boundary_ambiguous_mask[3:5] == (1, 1)


def test_terminal_annotation_projects_only_to_final_answer_token() -> None:
    serialized = _serialized()
    answer = serialized.segments[3]
    terminal = _root_annotation(
        serialized,
        span_id="char:terminal",
        literal_span=CharSpan(answer.start, answer.end),
        component=SegmentKind.FINAL_ANSWER,
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        causal_role=CausalRole.TERMINAL,
        state_or_edge_id="final_answer",
    )
    writer, _ = _writer(serialized)

    labels = writer.write(
        serialized,
        (terminal,),
        variant_label=VariantLabel.HALLUCINATED,
    )

    assert labels.positive_label_indices == (4,)
    assert labels.answer_mask[4] == 1
    assert labels.causal_role_masks[CausalRole.TERMINAL][4] == 1
    assert labels.edit_subtype_masks[EditErrorSubtype.FINAL_ANSWER_IDENTITY][4] == 1
    assert labels.local_falsehood_mask[4] == 1


def test_unverifiable_projects_semantic_fraction_without_error_or_role() -> None:
    serialized = _serialized()
    uncertain = _root_annotation(
        serialized,
        span_id="char:uncertain",
        semantic_types=frozenset({HallucinationType.UNVERIFIABLE}),
        edit_subtypes=frozenset(),
        causal_role=None,
        root_span_id=None,
    )
    writer, _ = _writer(serialized)

    labels = writer.write(
        serialized,
        (uncertain,),
        variant_label=VariantLabel.HALLUCINATED,
    )

    assert labels.semantic_type_masks[HallucinationType.UNVERIFIABLE][3] == 1
    assert labels.error_any_mask[3] == 0
    assert not any(mask[3] for mask in labels.edit_subtype_masks.values())
    assert not any(mask[3] for mask in labels.causal_role_masks.values())
    assert labels.error_char_fraction[3] > 0


def test_faithful_control_has_all_zero_masks_and_optional_matched_target() -> None:
    serialized = _serialized()
    target = _literal_span(serialized)
    writer, _ = _writer(serialized)

    labels = writer.write(
        serialized,
        CharAnnotationBuildResult(annotations=(), event_links=()),
        variant_label=VariantLabel.FAITHFUL,
        matched_target_span=target,
    )

    assert labels.matched_target_span == target
    assert labels.has_positive_labels is False
    assert not any(labels.error_any_mask)
    assert not any(labels.hallucination_core_mask)
    assert not any(labels.boundary_ambiguous_mask)
    assert not any(labels.error_char_fraction)
    assert all(not any(mask) for mask in labels.semantic_type_masks.values())
    assert all(not any(mask) for mask in labels.edit_subtype_masks.values())
    assert all(not any(mask) for mask in labels.causal_role_masks.values())


def test_trace_local_annotations_rebase_to_full_serialized_coordinates() -> None:
    rendered, serialized, root, terminal = _trace_local_fixture()
    annotations = CharAnnotationBuildResult(
        annotations=(root, terminal),
        event_links=(
            EventAnnotationLink("event.root", (root.span_id,)),
            EventAnnotationLink("event.terminal", (terminal.span_id,)),
        ),
    )

    rebased = rebase_char_annotations(rendered, serialized, annotations)

    assert type(rebased) is CharAnnotationBuildResult
    reasoning = serialized.segments[2]
    answer = serialized.segments[3]
    rebased_root, rebased_terminal = rebased.annotations
    assert rebased_root.literal_span == CharSpan(
        reasoning.start + root.literal_span.start,
        reasoning.start + root.literal_span.end,
    )
    assert rebased_root.claim_span == CharSpan(reasoning.start, reasoning.end)
    assert rebased_terminal.literal_span == CharSpan(answer.start, answer.end)
    # The trace claim includes the programmatic ``Answer: `` context. The
    # serialized final-answer field contains only the value, so context is
    # clamped while literal containment and identity remain intact.
    assert rebased_terminal.claim_span == CharSpan(answer.start, answer.end)
    assert rebased.event_links == annotations.event_links

    writer, _ = _writer(serialized)
    labels = writer.write(
        serialized,
        annotations,
        variant_label=VariantLabel.HALLUCINATED,
        rendered_example=rendered,
    )
    assert labels.positive_label_indices == (3, 4)
    assert labels.causal_role_masks[CausalRole.ROOT][3] == 1
    assert labels.causal_role_masks[CausalRole.TERMINAL][4] == 1


def test_trace_local_matched_control_span_is_rebased_before_projection() -> None:
    rendered, serialized, root, _ = _trace_local_fixture()
    coordinate_map = DetectorCoordinateMap.from_rendered(rendered, serialized)
    expected = coordinate_map.rebase_any_span(root.literal_span)
    writer, _ = _writer(serialized)

    labels = writer.write(
        serialized,
        (),
        variant_label=VariantLabel.FAITHFUL,
        matched_target_span=root.literal_span,
        rendered_example=rendered,
    )

    assert labels.matched_target_span == expected
    assert serialized.text[expected.start : expected.end] == "N21"
    assert not any(labels.error_any_mask)


def test_trace_to_serialized_rebase_requires_exact_component_surfaces() -> None:
    rendered, _, root, _ = _trace_local_fixture()
    mismatched = _serialized(reasoning="原子 N22 is selected.")
    writer, tokenizer = _writer(mismatched)

    with pytest.raises(TokenProjectionError) as captured:
        writer.write(
            mismatched,
            (root,),
            variant_label=VariantLabel.HALLUCINATED,
            rendered_example=rendered,
        )

    assert captured.value.code == "TRACE_SERIALIZED_TEXT_MISMATCH"
    assert tokenizer.calls == 0


def test_variant_span_contracts_and_pure_omission_fail_closed() -> None:
    serialized = _serialized()
    annotation = _root_annotation(serialized)
    writer, _ = _writer(serialized)

    with pytest.raises(TokenProjectionError) as faithful_positive:
        writer.write(
            serialized,
            (annotation,),
            variant_label=VariantLabel.FAITHFUL,
        )
    assert faithful_positive.value.code == "FAITHFUL_CONTROL_HAS_ANNOTATIONS"

    with pytest.raises(TokenProjectionError) as missing_h:
        writer.write(
            serialized,
            (),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert missing_h.value.code == "HALLUCINATED_SPAN_MISSING"

    omission = CharAnnotationBuildResult(
        annotations=(),
        event_links=(),
        unlocalized_omissions=(
            UnlocalizedOmission(
                event_id="event.omission",
                target_kind=MutationTargetKind.NODE,
                state_or_edge_id="product",
            ),
        ),
    )
    with pytest.raises(TokenProjectionError) as omission_error:
        writer.write(
            serialized,
            omission,
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert omission_error.value.code == "UNLOCALIZED_OMISSION_NOT_PROJECTABLE"


def test_all_token_arrays_and_axis_masks_have_identical_length() -> None:
    serialized = _serialized()
    labels = project_char_annotations(
        serialized,
        (_root_annotation(serialized),),
        tokenizer=_FakeFastTokenizer(_default_offsets(serialized)),
        tokenizer_fingerprint=_fingerprint(),
        variant_label=VariantLabel.HALLUCINATED,
    )
    count = len(labels.input_ids)
    direct = (
        labels.attention_mask,
        labels.offset_mapping,
        labels.segment_ids,
        labels.evaluation_mask,
        labels.hallucination_core_mask,
        labels.error_any_mask,
        labels.local_falsehood_mask,
        labels.off_task_branch_mask,
        labels.reasoning_mask,
        labels.answer_mask,
        labels.boundary_ambiguous_mask,
        labels.error_char_fraction,
    )
    assert all(len(values) == count for values in direct)
    assert all(
        len(mask) == count
        for masks in (
            labels.semantic_type_masks,
            labels.edit_subtype_masks,
            labels.causal_role_masks,
        )
        for mask in masks.values()
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ({"offset_mapping": None}, "TOKENIZER_OFFSETS_MISSING"),
        ({"attention_mask": (1,)}, "TOKENIZER_OUTPUT_INVALID"),
        ({"input_text": "different text"}, "TOKENIZER_TEXT_IDENTITY_MISMATCH"),
        ({"overflowing_tokens": (999,)}, "TOKENIZER_TRUNCATION_DETECTED"),
    ),
)
def test_unreliable_or_mismatched_tokenizer_outputs_fail_closed(
    mutation: dict[str, object],
    code: str,
) -> None:
    serialized = _serialized()
    tokenizer = _FakeFastTokenizer(
        _default_offsets(serialized), output_overrides=mutation
    )
    writer, _ = _writer(serialized, tokenizer=tokenizer)

    with pytest.raises(TokenProjectionError) as captured:
        writer.write(
            serialized,
            (_root_annotation(serialized),),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert captured.value.code == code


def test_slow_tokenizer_and_runtime_fingerprint_drift_are_rejected() -> None:
    serialized = _serialized()
    slow = _FakeFastTokenizer(_default_offsets(serialized), is_fast=False)
    writer, _ = _writer(serialized, tokenizer=slow)
    with pytest.raises(TokenProjectionError) as captured:
        writer.write(
            serialized,
            (_root_annotation(serialized),),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert captured.value.code == "FAST_TOKENIZER_REQUIRED"

    drifted = _FakeFastTokenizer(_default_offsets(serialized), pad_token_id=99)
    with pytest.raises(TokenProjectionError) as fingerprint_error:
        TokenLabelSetWriter(drifted, _fingerprint())
    assert fingerprint_error.value.code == "TOKENIZER_FINGERPRINT_MISMATCH"

    with pytest.raises(TokenProjectionError) as missing_runtime_field:
        TokenLabelSetWriter(
            _FakeFastTokenizer(_default_offsets(serialized)),
            _fingerprint(unk_token_id=999),
        )
    assert missing_runtime_field.value.code == "TOKENIZER_FINGERPRINT_MISMATCH"

    byte_offset_fingerprint = TokenizerFingerprint(
        tokenizer_name="ChemDFM-R-14B",
        tokenizer_revision="frozen-revision",
        tokenizer_vocab_hash="frozen-vocabulary-identity",
        special_token_config={
            "bos_token_id": 101,
            "eos_token_id": 102,
            "pad_token_id": 0,
        },
        normalization_config={"normalizer": "none", "offset_unit": "utf8_byte"},
    )
    with pytest.raises(TokenProjectionError) as unreliable_unit:
        TokenLabelSetWriter(
            _FakeFastTokenizer(_default_offsets(serialized)),
            byte_offset_fingerprint,
        )
    assert unreliable_unit.value.code == "TOKENIZER_OFFSET_UNIT_UNRELIABLE"


@pytest.mark.parametrize(
    ("offsets", "special_override", "code"),
    (
        (
            ((0, 0), (2, 4), (3, 5)),
            None,
            "TOKENIZER_OFFSETS_NON_MONOTONIC",
        ),
        (((0, 0), (0, 9999)), None, "TOKENIZER_OFFSET_OUT_OF_RANGE"),
        (((0, 0), (1, 2)), (1, 1), "SPECIAL_OFFSET_NONEMPTY"),
        (((0, 0), (5, 5)), None, "NONCANONICAL_EMPTY_OFFSET"),
    ),
)
def test_invalid_offset_boundaries_fail_closed(
    offsets: tuple[tuple[int, int], ...],
    special_override: tuple[int, ...] | None,
    code: str,
) -> None:
    serialized = _serialized()
    tokenizer = _FakeFastTokenizer(
        offsets,
        output_overrides=(
            None
            if special_override is None
            else {"special_tokens_mask": special_override}
        ),
    )
    writer, _ = _writer(serialized, tokenizer=tokenizer)
    with pytest.raises(TokenProjectionError) as captured:
        writer.write(
            serialized,
            (_root_annotation(serialized),),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert captured.value.code == code


def test_unicode_byte_offsets_are_rejected_as_out_of_character_range() -> None:
    serialized = _serialized()
    offsets = list(_default_offsets(serialized))
    offsets[3] = (offsets[3][0], len(serialized.text.encode("utf-8")))
    tokenizer = _FakeFastTokenizer(tuple(offsets))
    writer, _ = _writer(serialized, tokenizer=tokenizer)

    with pytest.raises(TokenProjectionError) as captured:
        writer.write(
            serialized,
            (_root_annotation(serialized),),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert captured.value.code == "TOKENIZER_OFFSET_OUT_OF_RANGE"


def test_positive_annotation_must_match_component_and_hit_evaluated_token() -> None:
    serialized = _serialized()
    final = serialized.segments[3]
    misplaced = _root_annotation(
        serialized,
        literal_span=CharSpan(final.start, final.end),
        claim_span=CharSpan(final.start, final.end),
        component=SegmentKind.REASONING,
    )
    writer, _ = _writer(serialized)
    with pytest.raises(TokenProjectionError) as component_error:
        writer.write(
            serialized,
            (misplaced,),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert component_error.value.code == "ANNOTATION_COMPONENT_MISMATCH"

    source, instruction, reasoning, answer = serialized.segments
    offsets = (
        (0, 0),
        (source.start, source.end),
        (instruction.start, instruction.end),
        (reasoning.start, reasoning.start + 2),
        (answer.start, answer.end),
        (0, 0),
    )
    uncovered_writer, _ = _writer(serialized, tokenizer=_FakeFastTokenizer(offsets))
    with pytest.raises(TokenProjectionError) as uncovered:
        uncovered_writer.write(
            serialized,
            (_root_annotation(serialized),),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert uncovered.value.code == "POSITIVE_SPAN_UNCOVERED"


def test_coarse_token_cannot_merge_different_roles_or_unverifiable_semantics() -> None:
    serialized = _serialized(reasoning="A B")
    reasoning = serialized.segments[2]
    root_span = CharSpan(reasoning.start, reasoning.start + 1)
    propagated_span = CharSpan(reasoning.start + 2, reasoning.start + 3)
    root = _root_annotation(
        serialized,
        span_id="char:root",
        literal_span=root_span,
    )
    propagated = _root_annotation(
        serialized,
        span_id="char:propagated",
        literal_span=propagated_span,
        causal_role=CausalRole.PROPAGATED_FALSE,
        root_span_id="char:root",
        state_or_edge_id="product",
    )
    source, instruction, _, answer = serialized.segments
    offsets = (
        (0, 0),
        (source.start, source.end),
        (instruction.start, instruction.end),
        (reasoning.start, reasoning.end),
        (answer.start, answer.end),
        (0, 0),
    )
    writer, _ = _writer(serialized, tokenizer=_FakeFastTokenizer(offsets))
    with pytest.raises(TokenProjectionError) as role_collision:
        writer.write(
            serialized,
            (root, propagated),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert role_collision.value.code == "CAUSAL_ROLE_TOKEN_COLLISION"

    uncertain = _root_annotation(
        serialized,
        span_id="char:uncertain",
        literal_span=propagated_span,
        semantic_types=frozenset({HallucinationType.UNVERIFIABLE}),
        edit_subtypes=frozenset(),
        causal_role=None,
        root_span_id=None,
        state_or_edge_id="uncertain_claim",
    )
    with pytest.raises(TokenProjectionError) as uncertain_collision:
        writer.write(
            serialized,
            (root, uncertain),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert uncertain_collision.value.code == "UNVERIFIABLE_TOKEN_COLLISION"


def test_matched_target_must_be_evaluated_and_annotations_must_be_unique() -> None:
    serialized = _serialized()
    writer, _ = _writer(serialized)
    source = serialized.segments[0]
    with pytest.raises(TokenProjectionError) as target_error:
        writer.write(
            serialized,
            (),
            variant_label=VariantLabel.FAITHFUL,
            matched_target_span=CharSpan(source.start, source.end),
        )
    assert target_error.value.code == "MATCHED_TARGET_NOT_EVALUATED"

    annotation = _root_annotation(serialized)
    with pytest.raises(TokenProjectionError) as duplicate:
        writer.write(
            serialized,
            (annotation, annotation),
            variant_label=VariantLabel.HALLUCINATED,
        )
    assert duplicate.value.code == "DUPLICATE_ANNOTATION_ID"


def test_projector_does_not_download_or_write_model_artifacts(tmp_path: Path) -> None:
    serialized = _serialized()
    tokenizer = _FakeFastTokenizer(_default_offsets(serialized))
    labels = project_char_annotations(
        serialized,
        (_root_annotation(serialized),),
        tokenizer=tokenizer,
        tokenizer_fingerprint=_fingerprint(),
        variant_label=VariantLabel.HALLUCINATED,
    )

    assert labels.input_ids
    assert tokenizer.calls == 1
    assert tuple(tmp_path.iterdir()) == ()
