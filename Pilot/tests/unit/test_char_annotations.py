"""T041 multi-axis character annotation taxonomy and omission tests."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from molhallulens.annotation.char_annotations import (
    CharAnnotationBuildError,
    build_char_annotations,
    derive_evidence_relations,
    is_pure_omission,
)
from molhallulens.domain import (
    CausalRole,
    ClaimValue,
    EditErrorSubtype,
    EvidenceRelation,
    GraphDelta,
    HallucinationType,
    MutationEvent,
    MutationTargetKind,
    ValueProvenance,
    ValueType,
)
from molhallulens.rendering.trace_ast import (
    AnswerDocument,
    ClaimNode,
    LiteralNode,
    SequenceNode,
    StepDocument,
    TextNode,
    TraceDocument,
    render_trace,
)


def _claim(value: int | str) -> ClaimValue:
    value_type = ValueType.INTEGER if type(value) is int else ValueType.STRING
    return ClaimValue(
        raw_value=value,
        normalized_value=value,
        value_type=value_type,
        provenance=ValueProvenance.REFERENCE,
    )


def _event(
    event_id: str,
    target: str,
    role: CausalRole,
    *,
    root_event_id: str,
    semantic_types: Iterable[HallucinationType] = (HallucinationType.CONTRADICTION,),
    edit_subtypes: Iterable[EditErrorSubtype] = (EditErrorSubtype.HEAVY_ATOM_COUNT,),
    before: int | str = 1,
    after: int | str = 2,
    target_kind: MutationTargetKind = MutationTargetKind.NODE,
) -> MutationEvent:
    return MutationEvent(
        event_id=event_id,
        target_kind=target_kind,
        node_or_edge_id=target,
        before=_claim(before),
        after=_claim(after),
        causal_role=role,
        hallucination_types=frozenset(semantic_types),
        edit_subtypes=frozenset(edit_subtypes),
        operator_id="mol_edit.add.test_char_taxonomy",
        root_event_id=root_event_id,
    )


def _literal_claim(
    mention_id: str,
    target: str,
    value: str,
    *,
    target_kind: MutationTargetKind = MutationTargetKind.NODE,
) -> ClaimNode:
    return ClaimNode.from_template(
        f"claim.{mention_id}",
        "value={value}",
        {
            "value": LiteralNode(
                mention_id=mention_id,
                state_or_edge_id=target,
                value=value,
                target_kind=target_kind,
            )
        },
    )


def _render_reasoning(
    mentions: tuple[tuple[str, str, str, MutationTargetKind], ...],
):
    children = []
    for index, (mention_id, target, value, target_kind) in enumerate(mentions):
        if index:
            children.append(TextNode("; "))
        children.append(
            _literal_claim(
                mention_id,
                target,
                value,
                target_kind=target_kind,
            )
        )
    answer = _literal_claim("mention.control.answer", "control_answer", "OK")
    return render_trace(
        TraceDocument(
            steps=(StepDocument(1, SequenceNode(tuple(children))),),
            answer=AnswerDocument(SequenceNode((answer,))),
        )
    )


def _render_terminal(*, terminal_in_reasoning: bool = False):
    reasoning_target = "final_answer" if terminal_in_reasoning else "control"
    reasoning = _literal_claim("mention.reasoning", reasoning_target, "CO")
    answer_target = "control_answer" if terminal_in_reasoning else "final_answer"
    answer = _literal_claim("mention.final", answer_target, "CO")
    return render_trace(
        TraceDocument(
            steps=(StepDocument(1, SequenceNode((reasoning,))),),
            answer=AnswerDocument(SequenceNode((answer,))),
        )
    )


def test_multi_axis_labels_are_non_exclusive_and_root_linkage_is_exact() -> None:
    root = _event(
        "event.root",
        "heavy_delta",
        CausalRole.ROOT,
        root_event_id="event.root",
        semantic_types=(
            HallucinationType.CONTRADICTION,
            HallucinationType.REASONING_ERROR,
        ),
        edit_subtypes=(
            EditErrorSubtype.HEAVY_ATOM_COUNT,
            EditErrorSubtype.HEAVY_ATOM_ARITHMETIC,
        ),
    )
    propagated_false = _event(
        "event.false",
        "product_heavy",
        CausalRole.PROPAGATED_FALSE,
        root_event_id=root.event_id,
        edit_subtypes=(EditErrorSubtype.INTERNAL_INCONSISTENCY,),
    )
    propagated_conditional = _event(
        "event.conditional",
        "product_rings",
        CausalRole.PROPAGATED_CONDITIONAL,
        root_event_id=root.event_id,
        semantic_types=(HallucinationType.REASONING_ERROR,),
        edit_subtypes=(EditErrorSubtype.RING_COUNT,),
    )
    rendered = _render_reasoning(
        (
            ("mention.root.1", "heavy_delta", "2", MutationTargetKind.NODE),
            ("mention.root.2", "heavy_delta", "2", MutationTargetKind.NODE),
            (
                "mention.false",
                "product_heavy",
                "2",
                MutationTargetKind.NODE,
            ),
            (
                "mention.conditional",
                "product_rings",
                "2",
                MutationTargetKind.NODE,
            ),
        )
    )

    result = build_char_annotations(
        GraphDelta((root, propagated_false, propagated_conditional)),
        rendered,
        additional_evidence_relations={
            root.event_id: (EvidenceRelation.CONTRADICTS_SOURCE,)
        },
    )

    assert len(result.annotations) == 4
    root_annotations = tuple(
        item for item in result.annotations if item.state_or_edge_id == "heavy_delta"
    )
    assert len(root_annotations) == 2
    assert all(
        item.semantic_types
        == frozenset(
            {
                HallucinationType.CONTRADICTION,
                HallucinationType.REASONING_ERROR,
            }
        )
        for item in root_annotations
    )
    assert all(
        item.edit_subtypes
        == frozenset(
            {
                EditErrorSubtype.HEAVY_ATOM_COUNT,
                EditErrorSubtype.HEAVY_ATOM_ARITHMETIC,
            }
        )
        for item in root_annotations
    )
    assert all(item.root_span_id == item.span_id for item in root_annotations)
    assert all(
        item.evidence_relations
        == frozenset(
            {
                EvidenceRelation.CONTRADICTS_SOURCE,
                EvidenceRelation.CONTRADICTS_REFERENCE_STATE,
            }
        )
        for item in root_annotations
    )

    by_target = {item.state_or_edge_id: item for item in result.annotations}
    canonical_root = root_annotations[0].span_id
    assert by_target["product_heavy"].causal_role is CausalRole.PROPAGATED_FALSE
    assert by_target["product_rings"].causal_role is CausalRole.PROPAGATED_CONDITIONAL
    assert by_target["product_heavy"].root_span_id == canonical_root
    assert by_target["product_rings"].root_span_id == canonical_root
    assert EvidenceRelation.INTERNAL_INCONSISTENCY in (
        by_target["product_heavy"].evidence_relations
    )
    assert tuple(link.event_id for link in result.event_links) == (
        "event.root",
        "event.false",
        "event.conditional",
    )
    assert not result.has_unlocalized_omissions


def test_terminal_role_is_final_answer_only_and_self_rooted() -> None:
    terminal = _event(
        "event.terminal",
        "final_answer",
        CausalRole.TERMINAL,
        root_event_id="event.terminal",
        semantic_types=(HallucinationType.CONTRADICTION,),
        edit_subtypes=(EditErrorSubtype.FINAL_ANSWER_IDENTITY,),
        before="CC",
        after="CO",
    )

    result = build_char_annotations(GraphDelta((terminal,)), _render_terminal())

    assert len(result.annotations) == 1
    annotation = result.annotations[0]
    assert annotation.causal_role is CausalRole.TERMINAL
    assert annotation.component.value == "final_answer"
    assert annotation.step_index is None
    assert annotation.root_span_id == annotation.span_id

    with pytest.raises(CharAnnotationBuildError) as captured:
        build_char_annotations(
            GraphDelta((terminal,)),
            _render_terminal(terminal_in_reasoning=True),
        )
    assert captured.value.code == "CAUSAL_COMPONENT_MISMATCH"


def test_full_cf_propagated_final_answer_links_back_to_reasoning_root() -> None:
    root = _event(
        "event.root",
        "product_smiles",
        CausalRole.ROOT,
        root_event_id="event.root",
        before="CC",
        after="CO",
    )
    final_answer = _event(
        "event.answer",
        "final_answer",
        CausalRole.PROPAGATED_CONDITIONAL,
        root_event_id=root.event_id,
        edit_subtypes=(EditErrorSubtype.FINAL_ANSWER_IDENTITY,),
        before="CC",
        after="CO",
    )
    reasoning = _literal_claim(
        "mention.product",
        "product_smiles",
        "CO",
    )
    answer = _literal_claim("mention.answer", "final_answer", "CO")
    rendered = render_trace(
        TraceDocument(
            steps=(StepDocument(1, SequenceNode((reasoning,))),),
            answer=AnswerDocument(SequenceNode((answer,))),
        )
    )

    result = build_char_annotations(GraphDelta((root, final_answer)), rendered)
    by_target = {item.state_or_edge_id: item for item in result.annotations}

    assert by_target["product_smiles"].component.value == "reasoning"
    assert by_target["final_answer"].component.value == "final_answer"
    assert by_target["final_answer"].causal_role is CausalRole.PROPAGATED_CONDITIONAL
    assert (
        by_target["final_answer"].root_span_id
        == by_target["product_smiles"].span_id
    )


def test_pure_omission_is_unlocalized_even_if_a_neighbor_literal_is_misbound() -> None:
    omission = _event(
        "event.omission",
        "missing_claim",
        CausalRole.ROOT,
        root_event_id="event.omission",
        semantic_types=(HallucinationType.OMISSION,),
        edit_subtypes=(EditErrorSubtype.UNSUPPORTED_NATURAL_CLAIM,),
    )
    rendered = _render_reasoning(
        (
            (
                "mention.neighbor",
                "missing_claim",
                "adjacent",
                MutationTargetKind.NODE,
            ),
        )
    )

    result = build_char_annotations(GraphDelta((omission,)), rendered)

    assert is_pure_omission(omission)
    assert result.annotations == ()
    assert result.event_links == ()
    assert result.has_unlocalized_omissions
    assert result.unlocalized_omissions[0].suppressed_mention_ids == (
        "mention.neighbor",
    )


def test_mixed_omission_with_visible_falsehood_keeps_all_semantic_axes() -> None:
    event = _event(
        "event.mixed",
        "visible_claim",
        CausalRole.ROOT,
        root_event_id="event.mixed",
        semantic_types=(
            HallucinationType.OMISSION,
            HallucinationType.CONTRADICTION,
        ),
        edit_subtypes=(EditErrorSubtype.INTERNAL_INCONSISTENCY,),
    )
    rendered = _render_reasoning(
        (("mention.visible", "visible_claim", "2", MutationTargetKind.NODE),)
    )

    result = build_char_annotations(GraphDelta((event,)), rendered)

    assert not is_pure_omission(event)
    assert result.annotations[0].semantic_types == event.hallucination_types
    assert EvidenceRelation.INTERNAL_INCONSISTENCY in (
        result.annotations[0].evidence_relations
    )


def test_non_omission_missing_exact_target_mention_fails_closed() -> None:
    root = _event(
        "event.root",
        "heavy_delta",
        CausalRole.ROOT,
        root_event_id="event.root",
    )
    rendered = _render_reasoning(
        (("mention.other", "other_node", "2", MutationTargetKind.NODE),)
    )

    with pytest.raises(CharAnnotationBuildError) as captured:
        build_char_annotations(GraphDelta((root,)), rendered)
    assert captured.value.code == "ROOT_MENTION_MISSING"

    wrong_kind = _render_reasoning(
        (("mention.edge", "heavy_delta", "2", MutationTargetKind.EDGE),)
    )
    with pytest.raises(CharAnnotationBuildError) as captured:
        build_char_annotations(GraphDelta((root,)), wrong_kind)
    assert captured.value.code == "ROOT_MENTION_MISSING"


def test_evidence_taxonomy_is_typed_multi_axis_and_unknown_event_is_rejected() -> None:
    event = _event(
        "event.taxonomy",
        "constraint_claim",
        CausalRole.ROOT,
        root_event_id="event.taxonomy",
        semantic_types=(
            HallucinationType.UNSUPPORTED,
            HallucinationType.CONSTRAINT_VIOLATION,
        ),
        edit_subtypes=(EditErrorSubtype.INTERNAL_INCONSISTENCY,),
    )
    relations = derive_evidence_relations(
        event,
        additional=(EvidenceRelation.CONTRADICTS_SOURCE,),
    )
    assert relations == frozenset(
        {
            EvidenceRelation.CONTRADICTS_REFERENCE_STATE,
            EvidenceRelation.UNSUPPORTED_BY_EVIDENCE,
            EvidenceRelation.CONTRADICTS_INSTRUCTION,
            EvidenceRelation.INTERNAL_INCONSISTENCY,
            EvidenceRelation.CONTRADICTS_SOURCE,
        }
    )

    rendered = _render_reasoning(
        (
            (
                "mention.taxonomy",
                "constraint_claim",
                "2",
                MutationTargetKind.NODE,
            ),
        )
    )
    with pytest.raises(CharAnnotationBuildError) as captured:
        build_char_annotations(
            GraphDelta((event,)),
            rendered,
            additional_evidence_relations={
                "event.unknown": (EvidenceRelation.CONTRADICTS_SOURCE,)
            },
        )
    assert captured.value.code == "UNKNOWN_EVIDENCE_EVENT"
