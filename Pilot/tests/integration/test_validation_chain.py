"""End-to-end T024/T040/T041/T042 validation-chain integration tests."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.annotation.char_annotations import (
    CharAnnotationBuildResult,
    build_char_annotations,
)
from molhallulens.annotation.token_projection import TokenLabelSetWriter

# Import the builder package before validation.  This preserves the repository's
# current package initialization order while T043 itself remains __init__-free.
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.golden_bundles import build_t025_golden_corpus
from molhallulens.builders.leakage_groups import assign_leakage_groups
from molhallulens.builders.origin_audit import audit_origin_split_features
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.builders.split_manifest import load_verified_split_manifest
from molhallulens.builders.splitter import SplitName, build_group_stratified_split
from molhallulens.domain import (
    CharSpan,
    MutationTargetKind,
    PropagationPolicy,
    SegmentKind,
    TokenizerFingerprint,
    TraceLabels,
    VariantLabel,
)
from molhallulens.rendering.detector_prompt import DetectorPromptSerializer
from molhallulens.rendering.natural_rule import (
    LockedFinalAnswer,
    LockedNaturalStep,
    NaturalRenderRequest,
    render_natural_rule,
)
from molhallulens.rendering.trace_ast import ClaimNode, LiteralNode, SequenceNode
from molhallulens.validation.chain import (
    ArtifactValidationInput,
    BundleIntegrityValidator,
    BundleValidationInput,
    HallucinationSemanticValidator,
    RendererValidator,
    TokenAlignmentValidator,
    ValidatorChain,
)
from molhallulens.validation.reference import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
MANIFEST_ROOT = Path(__file__).resolve().parents[2] / "HallucinationDataset"
_UNSET = object()


class _CharacterTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "truncation": False,
            "padding": False,
        }
        count = len(text)
        return {
            "input_ids": (
                self.bos_token_id,
                *(100 + index for index in range(count)),
                self.eos_token_id,
            ),
            "attention_mask": (1,) * (count + 2),
            "offset_mapping": (
                (0, 0),
                *((index, index + 1) for index in range(count)),
                (0, 0),
            ),
            "special_tokens_mask": (1, *((0,) * count), 1),
            "input_text": text,
        }


TOKENIZER_FINGERPRINT = TokenizerFingerprint(
    tokenizer_name="ChemDFM-R-14B",
    tokenizer_revision="frozen-test-revision",
    tokenizer_vocab_hash="frozen-test-vocabulary",
    special_token_config={
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
    },
    normalization_config={"normalizer": "none", "offset_unit": "python_char"},
)


@cache
def _verified_manifest():
    items = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        reference = build_reference_dag(record)
        items.append(
            OriginValidationInput(
                record=record,
                artifact=reference,
                edit_truth=derive_edit_truth(reference),
            )
        )
    audit = audit_origin_split_features(items).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in items
        },
    )
    split = build_group_stratified_split(audit, leakage)
    return load_verified_split_manifest(
        MANIFEST_ROOT / "split_manifest.csv",
        MANIFEST_ROOT / "split_manifest.metadata.json",
        split_result=split,
        audit=audit,
    )


def _literal(mention_id: str, target: str, value: object) -> LiteralNode:
    return LiteralNode(
        mention_id=mention_id,
        state_or_edge_id=target,
        value=str(value),
        target_kind=MutationTargetKind.NODE,
    )


def _claim(mention_id: str, target: str, value: object) -> ClaimNode:
    return ClaimNode.from_template(
        f"claim.{mention_id}",
        "The locked value is {value}.",
        {"value": _literal(mention_id, target, value)},
    )


def _formal_content(record_id: str, step: Any) -> SequenceNode:
    literal = _literal(
        f"{record_id}.formal.{step.step_index}",
        "source",
        step.formal_ab,
    )
    return SequenceNode(
        (
            ClaimNode.from_template(
                f"claim.{record_id}.formal.{step.step_index}",
                "{formal}",
                {"formal": literal},
            ),
        )
    )


def _trace_labels(record: Any) -> TraceLabels:
    if record.variant_label is VariantLabel.FAITHFUL:
        return TraceLabels(False, True, True, True, True, True, True)
    if record.policy is PropagationPolicy.STOP:
        return TraceLabels(True, False, True, True, True, True, True)
    if record.policy is PropagationPolicy.PARTIAL:
        return TraceLabels(
            True, False, record.answer.product_equivalent, True, False, True, True
        )
    if record.policy is PropagationPolicy.FULL_CF:
        return TraceLabels(True, False, False, True, False, True, True)
    return TraceLabels(True, True, False, True, True, True, True)


def _render(record: Any, *, target_override: object = _UNSET):
    events_by_step: dict[int, list[Any]] = {}
    for event in record.graph_delta.events:
        if event.node_or_edge_id == "final_answer":
            continue
        step_index = record.locked_state.schema.nodes_by_id[
            event.node_or_edge_id
        ].step_index
        assert step_index is not None
        events_by_step.setdefault(step_index, []).append(event)

    steps = []
    for formal_step in record.formal_trace.steps:
        claims = tuple(
            _claim(
                f"{record.record_id}.event.{event.event_id}",
                event.node_or_edge_id,
                event.after.normalized_value,
            )
            for event in events_by_step.get(formal_step.step_index, ())
        )
        if not claims and (
            record.variant_label is VariantLabel.FAITHFUL
            and record.target_step_index == formal_step.step_index
            and record.target_node_id != "final_answer"
        ):
            target_value = (
                record.locked_state.value_for(record.target_node_id).normalized_value
                if target_override is _UNSET
                else target_override
            )
            claims = (
                _claim(
                    f"{record.record_id}.matched.target",
                    record.target_node_id,
                    target_value,
                ),
            )
        if not claims:
            claims = (
                _claim(
                    f"{record.record_id}.control.{formal_step.step_index}",
                    "source",
                    record.locked_state.value_for("source").normalized_value,
                ),
            )
        steps.append(
            LockedNaturalStep(
                step_index=formal_step.step_index,
                step_name=formal_step.step_name,
                narrative_claims=claims,
                formal_content=_formal_content(record.record_id, formal_step),
                formal_text=formal_step.formal_ab,
            )
        )
    return render_natural_rule(
        NaturalRenderRequest(
            steps=tuple(steps),
            final_answer=LockedFinalAnswer(
                _literal(
                    f"{record.record_id}.answer",
                    "final_answer",
                    record.answer.smiles,
                )
            ),
        )
    )


def _artifact(
    record: Any,
    *,
    target_override: object = _UNSET,
) -> ArtifactValidationInput:
    rendered = _render(record, target_override=target_override)
    if record.variant_label is VariantLabel.FAITHFUL:
        annotations = CharAnnotationBuildResult(annotations=(), event_links=())
    else:
        annotations = build_char_annotations(record.graph_delta, rendered)

    reasoning_spans = tuple(
        span
        for segment_id, span in rendered.segment_spans.items()
        if segment_id.startswith("reasoning.step.")
    )
    reasoning = rendered.detector_text[: max(span.end for span in reasoning_spans)]
    serialized = DetectorPromptSerializer().serialize(
        indexed_smiles=record.locked_state.value_for("source").normalized_value,
        instruction="Apply the requested molecular edit.",
        reasoning_chain=reasoning,
        final_answer=record.answer.smiles,
    )
    matched_target_span = None
    if record.variant_label is VariantLabel.FAITHFUL:
        expected_component = (
            SegmentKind.FINAL_ANSWER
            if record.policy is PropagationPolicy.TERMINAL
            else SegmentKind.REASONING
        )
        matched_target_span = next(
            mention.literal_span
            for mention in rendered.mentions
            if mention.state_or_edge_id == record.target_node_id
            and mention.component is expected_component
        )
    labels = TokenLabelSetWriter(_CharacterTokenizer(), TOKENIZER_FINGERPRINT).write(
        serialized,
        annotations,
        variant_label=record.variant_label,
        matched_target_span=matched_target_span,
        rendered_example=rendered,
    )
    manifest_row = _verified_manifest().row_for_origin(record.origin_id)
    return ArtifactValidationInput(
        draft=record,
        rendered=rendered,
        char_annotations=annotations,
        serialized=serialized,
        token_labels=labels,
        trace_labels=_trace_labels(record),
        split=manifest_row.split,
        leakage_group_id=manifest_row.leakage_group_id,
    )


@cache
def _bundle_and_artifacts():
    bundle = build_t025_golden_corpus().origins[0].bundle
    return bundle, tuple(_artifact(record) for record in bundle.records)


def _codes(report: Any) -> set[str]:
    return {issue.code for issue in report.issues}


def _slice_token_labels(labels: Any, end: int):
    direct_fields = (
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
    mapped_fields = (
        "semantic_type_masks",
        "edit_subtype_masks",
        "causal_role_masks",
    )
    return replace(
        labels,
        **{name: getattr(labels, name)[:end] for name in direct_fields},
        **{
            name: {key: values[:end] for key, values in getattr(labels, name).items()}
            for name in mapped_fields
        },
    )


def test_complete_real_t024_bundle_passes_all_five_validation_gates() -> None:
    bundle, artifacts = _bundle_and_artifacts()
    report = ValidatorChain().validate_bundle(
        BundleValidationInput(
            bundle=bundle,
            artifacts=artifacts,
            split_manifest=_verified_manifest(),
        )
    )

    assert report.all_pass, [
        (issue.code, issue.node_ids, dict(issue.evidence)) for issue in report.issues
    ]
    assert not report.issues


def test_four_h_phenotypes_and_all_n_masks_are_independently_accepted() -> None:
    _, artifacts = _bundle_and_artifacts()
    h_records = [
        item
        for item in artifacts
        if item.draft.variant_label is VariantLabel.HALLUCINATED
    ]
    n_records = [
        item for item in artifacts if item.draft.variant_label is VariantLabel.FAITHFUL
    ]

    assert {item.draft.policy for item in h_records} == set(PropagationPolicy)
    assert all(
        HallucinationSemanticValidator().validate(item).all_pass for item in h_records
    )
    assert all(TokenAlignmentValidator().validate(item).all_pass for item in artifacts)
    assert all(
        not item.token_labels.has_positive_labels
        and item.token_labels.matched_target_span is not None
        for item in n_records
    )


@pytest.mark.parametrize(
    ("policy", "trace_overrides", "code"),
    (
        (
            PropagationPolicy.STOP,
            {"answer_correct": False},
            "SEMANTIC_LOCAL_TRACE_LABELS",
        ),
        (
            PropagationPolicy.PARTIAL,
            {"reasoning_valid": True},
            "SEMANTIC_PARTIAL_TRACE_LABELS",
        ),
        (
            PropagationPolicy.FULL_CF,
            {"constraint_satisfied": True},
            "SEMANTIC_FULL_CF_TRACE_LABELS",
        ),
        (
            PropagationPolicy.TERMINAL,
            {"reasoning_valid": False},
            "SEMANTIC_TERMINAL_TRACE_LABELS",
        ),
    ),
)
def test_each_h_phenotype_rejects_its_defining_invariant_tamper(
    policy: PropagationPolicy,
    trace_overrides: dict[str, bool],
    code: str,
) -> None:
    _, artifacts = _bundle_and_artifacts()
    source = next(
        item
        for item in artifacts
        if item.draft.variant_label is VariantLabel.HALLUCINATED
        and item.draft.policy is policy
    )
    tampered = replace(
        source,
        trace_labels=replace(source.trace_labels, **trace_overrides),
    )

    report = HallucinationSemanticValidator().validate(tampered)

    assert code in _codes(report)


@pytest.mark.parametrize(
    "leaked_text",
    (
        "The hallucinated branch is shown here.",
        "GT: hidden product",
        "GT SMILES: hidden product",
        "GT_PRODUCT: hidden product",
        "Reference: hidden state",
        "Reference_Only_Answer: hidden state",
        "<ORACLE> hidden value",
    ),
)
def test_renderer_rejects_leakage_phrases_and_reference_headers(
    leaked_text: str,
) -> None:
    _, artifacts = _bundle_and_artifacts()
    source = artifacts[0]
    detector = replace(
        source.serialized.detector_input,
        instruction=leaked_text,
    )
    serialized = DetectorPromptSerializer().serialize_input(detector)
    tampered = replace(source, serialized=serialized)

    report = RendererValidator().validate(tampered)

    assert not report.all_pass
    assert _codes(report) & {
        "RENDERER_LABEL_LEAKAGE",
        "RENDERER_REFERENCE_HEADER",
    }


def test_renderer_binds_faithful_natural_mentions_to_locked_state() -> None:
    bundle, _ = _bundle_and_artifacts()
    record = next(
        item
        for item in bundle.records
        if item.variant_label is VariantLabel.FAITHFUL
        and item.policy is PropagationPolicy.STOP
    )
    locked = record.locked_state.value_for(record.target_node_id).normalized_value
    assert type(locked) is int
    independently_consistent_tamper = _artifact(
        record,
        target_override=locked + 1,
    )

    report = RendererValidator().validate(independently_consistent_tamper)

    assert "RENDERER_LOCKED_VALUE_MISMATCH" in _codes(report)


def test_token_alignment_rejects_silent_suffix_truncation() -> None:
    _, artifacts = _bundle_and_artifacts()
    source = next(
        item
        for item in artifacts
        if item.draft.variant_label is VariantLabel.FAITHFUL
        and item.draft.policy is PropagationPolicy.STOP
    )
    answer_start = next(
        index
        for index, segment in enumerate(source.token_labels.segment_ids)
        if segment is SegmentKind.FINAL_ANSWER
    )
    truncated = _slice_token_labels(source.token_labels, answer_start)

    report = TokenAlignmentValidator().validate(replace(source, token_labels=truncated))

    assert "TOKEN_TEXT_COVERAGE_INCOMPLETE" in _codes(report)


def test_bundle_requires_exact_matched_target_mention_span() -> None:
    bundle, artifacts = _bundle_and_artifacts()
    source = next(
        item
        for item in artifacts
        if item.draft.variant_label is VariantLabel.FAITHFUL
        and item.draft.policy is PropagationPolicy.STOP
    )
    original = source.token_labels.matched_target_span
    assert original is not None
    shifted = CharSpan(original.start + 1, original.end + 1)
    forged = replace(
        source,
        token_labels=replace(source.token_labels, matched_target_span=shifted),
    )
    completed = tuple(
        forged if item.record_id == source.record_id else item for item in artifacts
    )

    report = BundleIntegrityValidator().validate(
        BundleValidationInput(
            bundle=bundle,
            artifacts=completed,
            split_manifest=_verified_manifest(),
        )
    )

    assert "BUNDLE_PAIR_TARGET_SPAN_MISMATCH" in _codes(report)


def test_bundle_split_and_group_are_derived_from_verified_manifest() -> None:
    bundle, artifacts = _bundle_and_artifacts()
    manifest_row = _verified_manifest().row_for_origin(bundle.origin_id)
    forged_split = (
        SplitName.TEST if manifest_row.split is not SplitName.TEST else SplitName.TRAIN
    )
    forged = tuple(
        replace(
            item,
            split=forged_split,
            leakage_group_id="caller.chosen.group",
        )
        for item in artifacts
    )

    report = BundleIntegrityValidator().validate(
        BundleValidationInput(
            bundle=bundle,
            artifacts=forged,
            split_manifest=_verified_manifest(),
        )
    )

    assert {
        "BUNDLE_SPLIT_MISMATCH",
        "BUNDLE_LEAKAGE_GROUP_MISMATCH",
    } <= _codes(report)


def test_token_projection_tamper_and_cross_split_bundle_fail_closed() -> None:
    bundle, artifacts = _bundle_and_artifacts()
    hallucinated = next(
        item
        for item in artifacts
        if item.draft.variant_label is VariantLabel.HALLUCINATED
    )
    forged_labels = object.__new__(type(hallucinated.token_labels))
    for name in hallucinated.token_labels.__dataclass_fields__:
        object.__setattr__(
            forged_labels,
            name,
            getattr(hallucinated.token_labels, name),
        )
    object.__setattr__(
        forged_labels,
        "error_any_mask",
        (0,) * len(hallucinated.token_labels.input_ids),
    )
    token_report = TokenAlignmentValidator().validate(
        replace(hallucinated, token_labels=forged_labels)
    )
    assert "TOKEN_PROJECTION_MISMATCH" in _codes(token_report)

    cross_split = replace(artifacts[-1], split=SplitName.TEST)
    bundle_report = ValidatorChain().validate_bundle(
        BundleValidationInput(
            bundle=bundle,
            artifacts=(*artifacts[:-1], cross_split),
            split_manifest=_verified_manifest(),
        )
    )
    assert "BUNDLE_SPLIT_MISMATCH" in _codes(bundle_report)


def test_missing_post_token_artifact_is_a_structured_fatal_rejection() -> None:
    _, artifacts = _bundle_and_artifacts()

    report = TokenAlignmentValidator().validate(
        replace(artifacts[0], token_labels=None)
    )

    assert not report.all_pass
    assert "TOKEN_LABELS_MISSING" in _codes(report)


def test_chain_converts_unexpected_validator_exception_to_fatal_issue() -> None:
    _, artifacts = _bundle_and_artifacts()
    artifact = artifacts[0]
    object.__setattr__(artifact.draft, "locked_state", object())

    report = ValidatorChain().validate_artifact(artifact)

    assert not report.all_pass
    assert "VALIDATOR_INTERNAL_FAILURE" in _codes(report)
