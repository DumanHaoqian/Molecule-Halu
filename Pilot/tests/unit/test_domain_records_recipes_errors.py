"""Tests for records, recipes, candidates, and validation reports."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

from molhallulens.domain.enums import (
    CandidateSourceType,
    CausalRole,
    ComparatorKind,
    EditingSubtask,
    EditErrorSubtype,
    EditKind,
    EvidenceRelation,
    HallucinationType,
    MutationTargetKind,
    NodeRole,
    OperationSubtype,
    PropagationPolicy,
    SegmentKind,
    Severity,
    TaskFamily,
    ValidationStage,
    ValueProvenance,
    ValueType,
    VariantLabel,
    Visibility,
)
from molhallulens.domain.errors import ValidationIssue, ValidationReport
from molhallulens.domain.labels import (
    CharAnnotation,
    CharSpan,
    TokenLabelSet,
    TokenizerFingerprint,
)
from molhallulens.domain.recipes import (
    CandidatePatch,
    EditAction,
    OperatorSpec,
    PerturbationRecipe,
    RewriteBudget,
)
from molhallulens.domain.records import (
    BuildProvenance,
    DetectorInput,
    PerturbationResult,
    TaskRecord,
    TraceLabels,
)
from molhallulens.domain.state_dag import (
    ClaimValue,
    GraphDelta,
    MutationEvent,
    StateDAG,
    StateNodeSpec,
    StateSchema,
)


def _claim(value: int) -> ClaimValue:
    return ClaimValue(value, value, ValueType.ATOM_INDEX, ValueProvenance.REFERENCE)


def _single_answer_dag(value: str = "CN") -> StateDAG:
    schema = StateSchema(
        schema_id="test.answer-only",
        version="1.0",
        nodes=(
            StateNodeSpec(
                node_id="final_answer",
                value_type=ValueType.SMILES,
                step_index=None,
                role=NodeRole.FINAL_ANSWER,
                visibility=Visibility.CANDIDATE_OUTPUT,
                mutable=True,
                comparator=ComparatorKind.EXACT,
                renderer_slot="final_answer",
            ),
        ),
        edges=(),
    )
    return StateDAG(
        schema,
        {
            "final_answer": ClaimValue(
                value,
                value,
                ValueType.SMILES,
                ValueProvenance.REFERENCE,
            )
        },
    )


def _answer_token_labels(*, positive: bool) -> TokenLabelSet:
    semantic = {label: (0,) for label in HallucinationType}
    edit = {label: (0,) for label in EditErrorSubtype}
    roles = {label: (0,) for label in CausalRole}
    if positive:
        semantic[HallucinationType.CONTRADICTION] = (1,)
        edit[EditErrorSubtype.FINAL_ANSWER_IDENTITY] = (1,)
        roles[CausalRole.TERMINAL] = (1,)
    return TokenLabelSet(
        activation_alignment="post_token_h_t",
        tokenizer_fingerprint=TokenizerFingerprint(
            "test-tokenizer",
            "revision",
            "vocab-hash",
            {},
            {},
        ),
        serialized_text_sha256="text-hash",
        input_ids=(1,),
        attention_mask=(1,),
        offset_mapping=((0, 1),),
        segment_ids=(SegmentKind.FINAL_ANSWER,),
        evaluation_mask=(1,),
        hallucination_core_mask=(int(positive),),
        error_any_mask=(int(positive),),
        semantic_type_masks=semantic,
        edit_subtype_masks=edit,
        causal_role_masks=roles,
        local_falsehood_mask=(int(positive),),
        off_task_branch_mask=(0,),
        reasoning_mask=(0,),
        answer_mask=(1,),
        boundary_ambiguous_mask=(0,),
        error_char_fraction=(float(positive),),
        matched_target_span=CharSpan(0, 1),
    )


def _faithful_result(token_labels: TokenLabelSet) -> PerturbationResult:
    graph = _single_answer_dag()
    return PerturbationResult(
        record_id="record.n",
        origin_id="origin.1",
        leakage_group_id="leakage.1",
        bundle_id="bundle.1",
        pair_id="pair.1",
        matched_record_id="record.h",
        variant_label=VariantLabel.FAITHFUL,
        policy=PropagationPolicy.STOP,
        detector_input=DetectorInput("[C:1]", "instruction", "reasoning", "CN"),
        serialized_text="serialized candidate",
        serialized_text_sha256="text-hash",
        reference_graph=graph,
        candidate_graph=graph,
        graph_delta=GraphDelta(()),
        char_annotations=(),
        token_labels=token_labels,
        trace_labels=TraceLabels(
            hallucination_present=False,
            reasoning_valid=True,
            answer_correct=True,
            chemically_valid=True,
            constraint_satisfied=True,
            format_valid=True,
            answer_complete=True,
        ),
        validation_report=ValidationReport("all-pass"),
        provenance=BuildProvenance(
            provider="rules",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
        ),
    )


def _terminal_result(
    annotation: CharAnnotation,
    *,
    token_labels: TokenLabelSet | None = None,
) -> PerturbationResult:
    reference = _single_answer_dag("CN")
    candidate = _single_answer_dag("NC")
    event = MutationEvent(
        event_id="event.terminal",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="final_answer",
        before=reference.value_for("final_answer"),
        after=candidate.value_for("final_answer"),
        causal_role=CausalRole.TERMINAL,
        hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        operator_id="mol_edit.terminal.answer",
        root_event_id="event.terminal",
    )
    return PerturbationResult(
        record_id="record.h",
        origin_id="origin.1",
        leakage_group_id="leakage.1",
        bundle_id="bundle.1",
        pair_id="pair.terminal",
        matched_record_id="record.n",
        variant_label=VariantLabel.HALLUCINATED,
        policy=PropagationPolicy.TERMINAL,
        detector_input=DetectorInput("[C:1]", "instruction", "reasoning", "NC"),
        serialized_text="serialized candidate",
        serialized_text_sha256="text-hash",
        reference_graph=reference,
        candidate_graph=candidate,
        graph_delta=GraphDelta((event,)),
        char_annotations=(annotation,),
        token_labels=token_labels or _answer_token_labels(positive=True),
        trace_labels=TraceLabels(
            hallucination_present=True,
            reasoning_valid=True,
            answer_correct=False,
            chemically_valid=True,
            constraint_satisfied=True,
            format_valid=True,
            answer_complete=True,
        ),
        validation_report=ValidationReport("all-pass"),
        provenance=BuildProvenance(
            provider="rules",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
        ),
    )


def _full_cf_result() -> PerturbationResult:
    schema = StateSchema(
        schema_id="test.full-cf",
        version="1.0",
        nodes=(
            StateNodeSpec(
                node_id="anchor.idx",
                value_type=ValueType.ATOM_INDEX,
                step_index=0,
                role=NodeRole.PRIMARY_CLAIM,
                visibility=Visibility.CANDIDATE_OUTPUT,
                mutable=True,
                comparator=ComparatorKind.EXACT,
                renderer_slot="anchor",
            ),
            StateNodeSpec(
                node_id="product",
                value_type=ValueType.SMILES,
                step_index=1,
                role=NodeRole.DERIVED_CLAIM,
                visibility=Visibility.CANDIDATE_OUTPUT,
                mutable=True,
                comparator=ComparatorKind.EXACT,
                renderer_slot="product",
            ),
        ),
        edges=(),
    )
    reference = StateDAG(
        schema,
        {
            "anchor.idx": ClaimValue(
                1,
                1,
                ValueType.ATOM_INDEX,
                ValueProvenance.REFERENCE,
            ),
            "product": ClaimValue(
                "CN",
                "CN",
                ValueType.SMILES,
                ValueProvenance.REFERENCE,
            ),
        },
    )
    candidate = StateDAG(
        schema,
        {
            "anchor.idx": ClaimValue(
                2,
                2,
                ValueType.ATOM_INDEX,
                ValueProvenance.RULE,
            ),
            "product": ClaimValue(
                "NC",
                "NC",
                ValueType.SMILES,
                ValueProvenance.PROPAGATED,
            ),
        },
    )
    root_event = MutationEvent(
        event_id="event.root",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="anchor.idx",
        before=reference.value_for("anchor.idx"),
        after=candidate.value_for("anchor.idx"),
        causal_role=CausalRole.ROOT,
        hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        operator_id="mol_edit.add.alternate_anchor",
        root_event_id="event.root",
    )
    child_event = MutationEvent(
        event_id="event.child",
        target_kind=MutationTargetKind.NODE,
        node_or_edge_id="product",
        before=reference.value_for("product"),
        after=candidate.value_for("product"),
        causal_role=CausalRole.PROPAGATED_CONDITIONAL,
        hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        operator_id="mol_edit.add.alternate_anchor",
        root_event_id="event.root",
    )
    root_annotation = CharAnnotation(
        span_id="span.root",
        component=SegmentKind.REASONING,
        step_index=0,
        state_or_edge_id="anchor.idx",
        literal_span=CharSpan(0, 6),
        claim_span=CharSpan(0, 6),
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.ROOT,
        root_span_id="span.root",
    )
    child_annotation = CharAnnotation(
        span_id="span.child",
        component=SegmentKind.REASONING,
        step_index=1,
        state_or_edge_id="product",
        literal_span=CharSpan(7, 14),
        claim_span=CharSpan(7, 14),
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.PROPAGATED_CONDITIONAL,
        root_span_id="span.root",
    )
    return PerturbationResult(
        record_id="record.h.full-cf",
        origin_id="origin.1",
        leakage_group_id="leakage.1",
        bundle_id="bundle.1",
        pair_id="pair.full-cf",
        matched_record_id="record.n.full-cf",
        variant_label=VariantLabel.HALLUCINATED,
        policy=PropagationPolicy.FULL_CF,
        detector_input=DetectorInput("[C:1]", "instruction", "anchor product", "NC"),
        serialized_text="anchor product",
        serialized_text_sha256="full-cf-text-hash",
        reference_graph=reference,
        candidate_graph=candidate,
        graph_delta=GraphDelta((root_event, child_event)),
        char_annotations=(root_annotation, child_annotation),
        token_labels=None,
        trace_labels=TraceLabels(
            hallucination_present=True,
            reasoning_valid=False,
            answer_correct=False,
            chemically_valid=True,
            constraint_satisfied=False,
            format_valid=True,
            answer_complete=True,
        ),
        validation_report=ValidationReport("all-pass"),
        provenance=BuildProvenance(
            provider="rules",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
        ),
    )


def test_task_record_is_frozen_and_defensively_copies_metadata() -> None:
    metadata = {"tags": ["pilot"]}
    record = TaskRecord(
        origin_id="mol_edit.add_v2.0001",
        anonymous_sample_id="sample-1",
        family=TaskFamily.MOLECULE_EDITING,
        source_subtask="add_v2",
        normalized_subtask=EditingSubtask.ADD,
        operation_subtype=OperationSubtype.STANDARD,
        indexed_smiles="[C:1]",
        instruction="Add an amine.",
        gt_smiles="CN",
        reference_reasoning_chain="Reasoning",
        reference_final_answer="CN",
        parsed_reference_state={"anchor": 1},
        raw_metadata=metadata,
    )
    metadata["tags"].append("changed")

    assert record.raw_metadata["tags"] == ("pilot",)
    with pytest.raises(FrozenInstanceError):
        record.gt_smiles = "leak"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.parsed_reference_state["anchor"] = 2  # type: ignore[index]
    with pytest.raises(TypeError, match="normalized_subtask"):
        replace(record, normalized_subtask="add")  # type: ignore[arg-type]


def test_detector_input_has_no_gt_or_oracle_field() -> None:
    detector_input = DetectorInput("[C:1]", "instruction", "reasoning", "CN")

    assert detector_input.field_order == (
        "indexed_smiles",
        "instruction",
        "reasoning_chain",
        "final_answer",
    )
    assert "gt_smiles" not in detector_input.__dataclass_fields__  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="indexed_smiles"):
        DetectorInput(1, "instruction", "reasoning", "CN")  # type: ignore[arg-type]


def test_operator_and_candidate_contracts_are_root_only_and_immutable() -> None:
    spec = OperatorSpec(
        operator_id="mol_edit.add.alternate_anchor",
        root_fields=frozenset({"anchor.idx"}),
        supported_policies=frozenset(
            {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
        ),
        supported_sources=frozenset({CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}),
        hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
    )
    patch = CandidatePatch(
        candidate_id="candidate-1",
        root_node_id="anchor.idx",
        old_value=_claim(1),
        new_value=ClaimValue(2, 2, ValueType.ATOM_INDEX, ValueProvenance.RDKIT),
        edit_action=EditAction(
            edit_kind=EditKind.ADDITION,
            source_anchor_index=2,
            add_fragment_smiles="N",
            fragment_attachment_atom=0,
        ),
        source=CandidateSourceType.RDKIT,
        metadata={"rank": 1},
    )

    assert spec.root_fields == frozenset({"anchor.idx"})
    assert patch.root_node_id == "anchor.idx"
    with pytest.raises(TypeError):
        patch.metadata["rank"] = 2  # type: ignore[index]

    with pytest.raises(TypeError, match="edit_kind"):
        EditAction(
            edit_kind="addition",  # type: ignore[arg-type]
            add_fragment_smiles="N",
        )
    with pytest.raises(TypeError, match="string keys"):
        EditAction(
            edit_kind=EditKind.ADDITION,
            add_fragment_smiles="N",
            metadata={"nested": {1: "not-json-safe"}},
        )
    with pytest.raises(TypeError, match="supported_policies"):
        OperatorSpec(
            operator_id="raw-string-policy",
            root_fields=frozenset({"anchor.idx"}),
            supported_policies=frozenset({"local"}),  # type: ignore[arg-type]
            supported_sources=frozenset({CandidateSourceType.RULE}),
            hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        )
    with pytest.raises(ValueError, match="normalized root value"):
        CandidatePatch(
            candidate_id="provenance-only",
            root_node_id="anchor.idx",
            old_value=_claim(1),
            new_value=ClaimValue(1, 1, ValueType.ATOM_INDEX, ValueProvenance.RDKIT),
            edit_action=None,
            source=CandidateSourceType.RDKIT,
        )
    with pytest.raises(ValueError, match="normalized root value"):
        CandidatePatch(
            candidate_id="atom-set-reorder",
            root_node_id="atoms",
            old_value=ClaimValue(
                [1, 2],
                [1, 2],
                ValueType.ATOM_SET,
                ValueProvenance.REFERENCE,
            ),
            new_value=ClaimValue(
                [2, 1],
                [2, 1],
                ValueType.ATOM_SET,
                ValueProvenance.RULE,
            ),
            edit_action=None,
            source=CandidateSourceType.RULE,
        )


def test_recipe_enforces_policy_specific_cut_and_terminal_rules() -> None:
    common = {
        "recipe_id": "recipe-1",
        "origin_id": "mol_edit.add_v2.0001",
        "operator_id": "mol_edit.add.alternate_anchor",
        "target_node_id": "anchor.idx",
        "candidate_source_mode": CandidateSourceType.HYBRID,
        "variant_index": 0,
        "derived_seed": 42,
        "rewrite_budget": RewriteBudget(3, 50, "short"),
        "candidate_difficulty_bucket": "hard",
        "renderer_style_id": "style_01",
    }
    recipe = PerturbationRecipe(
        **common,
        policy=PropagationPolicy.PARTIAL,
        partial_cut_nodes=frozenset({"product"}),
    )
    assert recipe.partial_cut_nodes == frozenset({"product"})

    with pytest.raises(ValueError, match="cut node"):
        PerturbationRecipe(**common, policy=PropagationPolicy.PARTIAL)
    with pytest.raises(ValueError, match="final_answer"):
        PerturbationRecipe(**common, policy=PropagationPolicy.TERMINAL)
    with pytest.raises(TypeError, match="policy"):
        PerturbationRecipe(**common, policy="partial")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_changed_claims"):
        RewriteBudget(True, 10, "short")  # type: ignore[arg-type]


def test_validation_report_preserves_structured_evidence() -> None:
    issue = ValidationIssue(
        code="GRAPH_EDIT_MISMATCH",
        severity=Severity.ERROR,
        stage=ValidationStage.GRAPH_EDIT,
        node_ids=("anchor.idx", "product"),
        message="Claimed edit does not produce candidate product.",
        evidence={"expected_bonds": [(1, 2)]},
    )
    report = ValidationReport("graph-edit-validator", (issue,))

    assert report.all_pass is False
    assert report.by_severity(Severity.ERROR) == (issue,)
    assert issue.evidence["expected_bonds"] == ((1, 2),)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"severity": "critical"}, "severity"),
        ({"stage": "unknown-stage"}, "stage"),
    ],
)
def test_validation_issue_rejects_unknown_enum_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "code": "BROKEN",
        "severity": Severity.ERROR,
        "stage": ValidationStage.GRAPH_EDIT,
        "node_ids": (),
        "message": "broken",
    }
    values.update(overrides)
    with pytest.raises(TypeError, match=message):
        ValidationIssue(**values)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ValidationIssue"):
        ValidationReport("validator", ("not-an-issue",))  # type: ignore[arg-type]
    forged = Mock(spec=ValidationIssue)
    forged.severity = "catastrophic"
    forged.code = "FORGED"
    with pytest.raises(TypeError, match="ValidationIssue"):
        ValidationReport("validator", (forged,))


def test_provenance_and_trace_labels_enforce_runtime_types() -> None:
    with pytest.raises(TypeError, match="attempt_count"):
        BuildProvenance(
            provider="poe",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
            attempt_count=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="hallucination_present"):
        TraceLabels(
            hallucination_present="yes",  # type: ignore[arg-type]
            reasoning_valid=False,
            answer_correct=False,
            chemically_valid=True,
            constraint_satisfied=False,
            format_valid=True,
            answer_complete=True,
        )
    with pytest.raises(ValueError, match="secrets"):
        BuildProvenance(
            provider="poe",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
            extra={"request": {"api_key": "must-not-be-stored"}},
        )


def test_faithful_result_rejects_positive_token_labels() -> None:
    clean = _faithful_result(_answer_token_labels(positive=False))
    assert clean.token_labels is not None
    assert not clean.token_labels.has_positive_labels
    with pytest.raises(ValueError, match="fully faithful"):
        replace(
            clean,
            trace_labels=replace(clean.trace_labels, answer_correct=False),
        )

    with pytest.raises(ValueError, match="positive token labels"):
        _faithful_result(_answer_token_labels(positive=True))

    with pytest.raises(ValueError, match="GraphDelta targets"):
        replace(clean, candidate_graph=_single_answer_dag("CO"))


def test_terminal_result_binds_graph_char_token_and_trace_contracts() -> None:
    terminal_annotation = CharAnnotation(
        span_id="span.terminal",
        component=SegmentKind.FINAL_ANSWER,
        step_index=None,
        state_or_edge_id="final_answer",
        literal_span=CharSpan(0, 1),
        claim_span=CharSpan(0, 1),
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.TERMINAL,
        root_span_id="span.terminal",
    )
    result = _terminal_result(terminal_annotation)
    assert result.graph_delta.root_events[0].node_or_edge_id == "final_answer"

    with pytest.raises(ValueError, match="valid reasoning"):
        replace(
            result,
            trace_labels=replace(
                result.trace_labels,
                reasoning_valid=False,
                answer_correct=True,
            ),
        )

    wrong_root = CharAnnotation(
        span_id="span.root",
        component=SegmentKind.REASONING,
        step_index=1,
        state_or_edge_id="anchor.idx",
        literal_span=CharSpan(0, 1),
        claim_span=CharSpan(0, 1),
        semantic_types=frozenset({HallucinationType.CONTRADICTION}),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        evidence_relations=frozenset({EvidenceRelation.CONTRADICTS_REFERENCE_STATE}),
        causal_role=CausalRole.ROOT,
        root_span_id="span.root",
    )
    with pytest.raises(ValueError, match="unknown state or edge"):
        _terminal_result(wrong_root)

    uncertain_only = CharAnnotation(
        span_id="span.uncertain",
        component=SegmentKind.FINAL_ANSWER,
        step_index=None,
        state_or_edge_id="final_answer",
        literal_span=CharSpan(0, 1),
        claim_span=CharSpan(0, 1),
        semantic_types=frozenset({HallucinationType.UNVERIFIABLE}),
        edit_subtypes=frozenset(),
        evidence_relations=frozenset(),
        causal_role=None,
        root_span_id=None,
    )
    with pytest.raises(ValueError, match="adjudicated positive char"):
        _terminal_result(uncertain_only)

    clean_tokens = _answer_token_labels(positive=False)
    semantic = dict(clean_tokens.semantic_type_masks)
    semantic[HallucinationType.UNVERIFIABLE] = (1,)
    uncertain_tokens = replace(
        clean_tokens,
        semantic_type_masks=semantic,
        error_char_fraction=(1.0,),
    )
    with pytest.raises(ValueError, match="adjudicated positive token"):
        _terminal_result(terminal_annotation, token_labels=uncertain_tokens)


def test_full_cf_result_cross_binds_every_delta_target_to_annotations() -> None:
    result = _full_cf_result()
    root_annotation, child_annotation = result.char_annotations

    with pytest.raises(ValueError, match="every GraphDelta target"):
        replace(result, char_annotations=(root_annotation,))

    wrong_role = replace(
        child_annotation,
        causal_role=CausalRole.PROPAGATED_FALSE,
    )
    with pytest.raises(ValueError, match="causal roles"):
        replace(result, char_annotations=(root_annotation, wrong_role))

    extra_semantic_type = replace(
        child_annotation,
        semantic_types=frozenset(
            {HallucinationType.CONTRADICTION, HallucinationType.UNSUPPORTED}
        ),
    )
    with pytest.raises(ValueError, match="semantic types"):
        replace(result, char_annotations=(root_annotation, extra_semantic_type))

    extra_edit_subtype = replace(
        child_annotation,
        edit_subtypes=frozenset(
            {
                EditErrorSubtype.PRODUCT_CONSTRUCTION,
                EditErrorSubtype.INTERNAL_INCONSISTENCY,
            }
        ),
    )
    with pytest.raises(ValueError, match="edit subtypes"):
        replace(result, char_annotations=(root_annotation, extra_edit_subtype))
