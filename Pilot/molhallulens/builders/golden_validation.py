"""T044 end-to-end golden validation fixtures for molecule editing.

Nine real origins traverse the production T019--T043 path.  The suite uses a
small deterministic fast-offset tokenizer so the token artifacts are replayable
without downloading ChemDFM-R weights; the writer interface and all projection
invariants are the same as for the production tokenizer.  No live Poe request is
made.  The registered delete-with-replacement origin is retained separately as
an expected fail-closed capability diagnostic because its contract cannot form
a legal four-policy T024 bundle.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.annotation.char_annotations import (
    CharAnnotationBuildResult,
    build_char_annotations,
)
from molhallulens.annotation.token_projection import TokenLabelSetWriter
from molhallulens.builders.anomaly_registry import classify_edit_truth
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.golden_bundles import (
    T025_GOLDEN_ORIGINS,
    GoldenOriginBundle,
    GoldenOriginSpec,
    GoldenPolicySpec,
    _build_origin,
    _records_by_id,
    _UnusedLabelProjector,
    _UnusedTraceRenderer,
    _UnusedValidatorChain,
)
from molhallulens.builders.leakage_groups import assign_leakage_groups
from molhallulens.builders.origin_audit import audit_origin_split_features
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.builders.split_manifest import (
    VerifiedSplitManifest,
    load_verified_split_manifest,
)
from molhallulens.builders.splitter import build_group_stratified_split
from molhallulens.chemistry import isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.config.loader import ConfigBundle
from molhallulens.domain import (
    CandidateSourceType,
    CharSpan,
    EditingSubtask,
    MutationTargetKind,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    SegmentKind,
    TokenizerFingerprint,
    TokenLabelSet,
    TraceLabels,
    ValidationReport,
    VariantLabel,
)
from molhallulens.perturbators import (
    DeletionPerturbator,
    PerturbationContext,
    task_record_from_joined_input,
)
from molhallulens.perturbators.editing.addition import ADDITION_OPERATOR_IDS
from molhallulens.perturbators.editing.deletion import (
    DELETION_OPERATOR_IDS,
    DeletionCandidateEngine,
)
from molhallulens.perturbators.editing.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
)
from molhallulens.perturbators.registry import (
    OperatorRegistryError,
    PerturbatorRegistry,
)
from molhallulens.propagation import EditingPropagationEngine
from molhallulens.rendering.detector_prompt import (
    DetectorPromptSerializer,
    SerializedDetectorInput,
)
from molhallulens.rendering.natural_rule import (
    LockedFinalAnswer,
    LockedNaturalStep,
    NaturalRenderRequest,
    render_natural_rule,
)
from molhallulens.rendering.trace_ast import (
    ClaimNode,
    LiteralNode,
    RenderedExample,
    SequenceNode,
)
from molhallulens.validation.chain import (
    ARTIFACT_VALIDATOR_IDS,
    BUNDLE_INTEGRITY_VALIDATOR_ID,
    ArtifactValidationInput,
    BundleValidationInput,
    ValidatorChain,
)
from molhallulens.validation.reference import OriginValidationInput

T044_FIXTURE_FORMAT_VERSION = "t044_extended_golden_suite_v1"
T044_REPORT_FORMAT_VERSION = "t044_golden_validation_v1"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "Dataset"
DEFAULT_MANIFEST_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset"
DEFAULT_FIXTURE_PATH = (
    DEFAULT_PROJECT_ROOT / "tests/golden/t044_extended_golden_suite.json"
)
DEFAULT_REPORT_PATH = (
    DEFAULT_PROJECT_ROOT / "Dataset/reports/t044_golden_validation.json"
)
DELETE_WITH_REPLACEMENT_ORIGIN_ID = "mol_edit.delete_v2.0081"

_POLICIES = tuple(PropagationPolicy)


def _base_spec(subtask: EditingSubtask) -> GoldenOriginSpec:
    return next(
        item for item in T025_GOLDEN_ORIGINS if item.normalized_subtask is subtask
    )


def _with_full_cf(
    spec: GoldenOriginSpec,
    *,
    operator_id: str,
    target_node_id: str,
    quota_bucket: str,
) -> GoldenOriginSpec:
    full = GoldenPolicySpec(
        PropagationPolicy.FULL_CF,
        operator_id,
        target_node_id,
        quota_bucket,
    )
    return replace(spec, policies=(*spec.policies[:2], full, spec.policies[3]))


@dataclass(frozen=True, slots=True)
class ExtendedGoldenOriginCase:
    """One real origin plus its stable coverage role and four-policy spec."""

    case_id: str
    case_kind: str
    spec: GoldenOriginSpec
    coverage_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case_id"),
            (self.case_kind, "case_kind"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.spec) is not GoldenOriginSpec:
            raise TypeError("spec must be GoldenOriginSpec")
        tags = tuple(self.coverage_tags)
        if not tags or any(type(item) is not str or not item for item in tags):
            raise ValueError("coverage_tags must contain non-empty text")
        if len(tags) != len(set(tags)):
            raise ValueError("coverage_tags must be unique")
        object.__setattr__(self, "coverage_tags", tags)


_ADD_MULTI = _with_full_cf(
    replace(
        _base_spec(EditingSubtask.ADD),
        origin_id="mol_edit.add_v2.0071",
    ),
    operator_id=ADDITION_OPERATOR_IDS[2],
    target_node_id="add_fragment",
    quota_bucket="valid_wrong_group_fragment",
)
_SUBSTITUTE_MULTI = _with_full_cf(
    replace(
        _base_spec(EditingSubtask.SUBSTITUTE),
        origin_id="mol_edit.substitute_v2.0271",
    ),
    operator_id=SUBSTITUTION_OPERATOR_IDS[2],
    target_node_id="add_fragment",
    quota_bucket="valid_wrong_group_fragment",
)

T044_GOLDEN_ORIGIN_CASES = (
    ExtendedGoldenOriginCase(
        "add.standard",
        "standard",
        _base_spec(EditingSubtask.ADD),
        ("baseline", "terminal_near_miss"),
    ),
    ExtendedGoldenOriginCase(
        "add.multi_mapping",
        "multi_candidate",
        _ADD_MULTI,
        ("multiple_optimal_mappings", "mapping_trace_disambiguation"),
    ),
    ExtendedGoldenOriginCase(
        "add.hard_large",
        "hard",
        replace(
            _base_spec(EditingSubtask.ADD),
            origin_id="mol_edit.add_v2.0229",
        ),
        ("large_fragment", "high_heavy_atom_count", "high_ring_count"),
    ),
    ExtendedGoldenOriginCase(
        "delete.standard",
        "standard",
        _base_spec(EditingSubtask.DELETE),
        ("baseline", "terminal_near_miss"),
    ),
    ExtendedGoldenOriginCase(
        "delete.hard_large",
        "hard",
        replace(
            _base_spec(EditingSubtask.DELETE),
            origin_id="mol_edit.delete_v2.0202",
        ),
        ("large_fragment", "high_heavy_atom_count", "high_ring_count"),
    ),
    ExtendedGoldenOriginCase(
        "delete.duplicate_scaffold",
        "boundary",
        replace(
            _base_spec(EditingSubtask.DELETE),
            origin_id="mol_edit.delete_v2.0185",
        ),
        ("duplicate_scaffold_group", "split_manifest_lock"),
    ),
    ExtendedGoldenOriginCase(
        "substitute.standard",
        "standard",
        _base_spec(EditingSubtask.SUBSTITUTE),
        ("baseline", "terminal_near_miss"),
    ),
    ExtendedGoldenOriginCase(
        "substitute.multi_anchor",
        "multi_candidate",
        _SUBSTITUTE_MULTI,
        ("multi_anchor_relocation", "typed_dual_anchor", "terminal_near_miss"),
    ),
    ExtendedGoldenOriginCase(
        "substitute.valence_boundary",
        "boundary",
        replace(
            _base_spec(EditingSubtask.SUBSTITUTE),
            origin_id="mol_edit.substitute_v2.0064",
        ),
        ("retained_boundary_valence_relaxation", "terminal_near_miss"),
    ),
)


class _FrozenWhitespaceTokenizer:
    """Deterministic fast-offset test backend; no vocabulary/model download."""

    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        expected = {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "truncation": False,
            "padding": False,
        }
        if kwargs != expected:
            raise ValueError("T044 tokenizer requires the exact fast-offset call")
        offsets = tuple(
            (match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )
        count = len(offsets)
        return {
            "input_ids": (
                self.bos_token_id,
                *(100 + index for index in range(count)),
                self.eos_token_id,
            ),
            "attention_mask": (1,) * (count + 2),
            "offset_mapping": ((0, 0), *offsets, (0, 0)),
            "special_tokens_mask": (1, *((0,) * count), 1),
            "input_text": text,
        }


T044_TOKENIZER_FINGERPRINT = TokenizerFingerprint(
    tokenizer_name="ChemDFM-R-14B-compatible-fast-offset-fixture",
    tokenizer_revision="t044-frozen-whitespace-v1",
    tokenizer_vocab_hash="t044-static-test-vocabulary-identity",
    special_token_config={
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
    },
    normalization_config={"normalizer": "none", "offset_unit": "python_char"},
)


def _stable(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=repr)
    raise TypeError(f"unsupported T044 JSON value: {type(value).__qualname__}")


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            _stable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
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
        "{value}",
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


def _answer_correct_against_oracle(record: Any) -> bool:
    answer = record.locked_state.value_for("final_answer").normalized_value
    oracle = record.reference_graph.value_for("oracle_gt").normalized_value
    if type(answer) is not str or not answer or type(oracle) is not str or not oracle:
        raise ValueError("Answer and authoritative oracle must be non-empty SMILES")
    try:
        return isomeric_graph_equivalent(answer, oracle)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Answer correctness cannot be resolved against authoritative oracle"
        ) from error


def _trace_labels(record: Any) -> TraceLabels:
    answer_correct = _answer_correct_against_oracle(record)
    if record.variant_label is VariantLabel.FAITHFUL:
        return TraceLabels(False, True, answer_correct, True, True, True, True)
    if record.policy is PropagationPolicy.STOP:
        return TraceLabels(True, False, answer_correct, True, True, True, True)
    if record.policy is PropagationPolicy.PARTIAL:
        return TraceLabels(
            True,
            False,
            answer_correct,
            True,
            False,
            True,
            True,
        )
    if record.policy is PropagationPolicy.FULL_CF:
        return TraceLabels(True, False, answer_correct, True, False, True, True)
    return TraceLabels(True, True, answer_correct, True, True, True, True)


def render_extended_golden_record(
    record: Any,
    *,
    matched_hallucinated_record: Any | None = None,
) -> RenderedExample:
    """Render one T024 draft using only locked detector-visible values."""

    events_by_step: dict[int, list[Any]] = {}
    for event in record.graph_delta.events:
        if event.node_or_edge_id == "final_answer":
            continue
        node = record.locked_state.schema.nodes_by_id[event.node_or_edge_id]
        if node.step_index is None:
            raise ValueError("rendered mutation target must have a trace step")
        events_by_step.setdefault(node.step_index, []).append(event)

    faithful_targets_by_step: dict[int, list[str]] = {}
    if matched_hallucinated_record is not None:
        if record.variant_label is not VariantLabel.FAITHFUL:
            raise ValueError("only faithful records accept a matched H render shape")
        if matched_hallucinated_record.record_id != record.matched_record_id:
            raise ValueError("matched H render shape has the wrong record identity")
        for event in matched_hallucinated_record.graph_delta.events:
            if event.target_kind is not MutationTargetKind.NODE:
                raise ValueError("T044 natural renderer only supports node mutations")
            if event.node_or_edge_id == "final_answer":
                continue
            node = record.locked_state.schema.nodes_by_id[event.node_or_edge_id]
            if node.step_index is None:
                raise ValueError("matched target must have a trace step")
            faithful_targets_by_step.setdefault(node.step_index, []).append(
                event.node_or_edge_id
            )

    steps: list[LockedNaturalStep] = []
    for formal_step in record.formal_trace.steps:
        claims = tuple(
            _claim(
                f"{record.record_id}.event.{event.event_id}",
                event.node_or_edge_id,
                event.after.normalized_value,
            )
            for event in events_by_step.get(formal_step.step_index, ())
        )
        if not claims and faithful_targets_by_step.get(formal_step.step_index):
            claims = tuple(
                _claim(
                    f"{record.record_id}.matched.{target_id}",
                    target_id,
                    record.locked_state.value_for(target_id).normalized_value,
                )
                for target_id in faithful_targets_by_step[formal_step.step_index]
            )
        if not claims and (
            record.variant_label is VariantLabel.FAITHFUL
            and record.target_step_index == formal_step.step_index
            and record.target_node_id != "final_answer"
        ):
            claims = (
                _claim(
                    f"{record.record_id}.matched.target",
                    record.target_node_id,
                    record.locked_state.value_for(
                        record.target_node_id
                    ).normalized_value,
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


def _matched_target_span(record: Any, rendered: RenderedExample) -> CharSpan | None:
    if record.variant_label is not VariantLabel.FAITHFUL:
        return None
    expected_component = (
        SegmentKind.FINAL_ANSWER
        if record.policy is PropagationPolicy.TERMINAL
        else SegmentKind.REASONING
    )
    matches = tuple(
        mention.literal_span
        for mention in rendered.mentions
        if mention.state_or_edge_id == record.target_node_id
        and mention.component is expected_component
        and (
            expected_component is SegmentKind.FINAL_ANSWER
            or mention.step_index == record.target_step_index
        )
    )
    if len(matches) != 1:
        raise ValueError("faithful control must expose one exact matched target span")
    return matches[0]


def build_extended_record_artifact(
    record: Any,
    *,
    manifest: VerifiedSplitManifest,
    token_writer: TokenLabelSetWriter,
    matched_hallucinated_record: Any | None = None,
) -> ArtifactValidationInput:
    """Run T040--T042 for one already-built T024 draft record."""

    rendered = render_extended_golden_record(
        record,
        matched_hallucinated_record=matched_hallucinated_record,
    )
    annotations = (
        CharAnnotationBuildResult(annotations=(), event_links=())
        if record.variant_label is VariantLabel.FAITHFUL
        else build_char_annotations(record.graph_delta, rendered)
    )
    reasoning_spans = tuple(
        span
        for segment_id, span in rendered.segment_spans.items()
        if segment_id.startswith("reasoning.step.")
    )
    reasoning = rendered.detector_text[: max(span.end for span in reasoning_spans)]
    serialized = DetectorPromptSerializer().serialize(
        indexed_smiles=record.locked_state.value_for("source").normalized_value,
        instruction=record.locked_state.value_for("instruction").normalized_value,
        reasoning_chain=reasoning,
        final_answer=record.answer.smiles,
    )
    labels = token_writer.write(
        serialized,
        annotations,
        variant_label=record.variant_label,
        matched_target_span=_matched_target_span(record, rendered),
        rendered_example=rendered,
    )
    row = manifest.row_for_origin(record.origin_id)
    return ArtifactValidationInput(
        draft=record,
        rendered=rendered,
        char_annotations=annotations,
        serialized=serialized,
        token_labels=labels,
        trace_labels=_trace_labels(record),
        split=row.split,
        leakage_group_id=row.leakage_group_id,
    )


@dataclass(frozen=True, slots=True)
class ExtendedGoldenOriginBuild:
    case: ExtendedGoldenOriginCase
    golden: GoldenOriginBundle
    artifacts: tuple[ArtifactValidationInput, ...]
    validation: ValidationReport

    def __post_init__(self) -> None:
        if type(self.case) is not ExtendedGoldenOriginCase:
            raise TypeError("case must be ExtendedGoldenOriginCase")
        if type(self.golden) is not GoldenOriginBundle:
            raise TypeError("golden must be GoldenOriginBundle")
        artifacts = tuple(self.artifacts)
        if len(artifacts) != 8 or any(
            type(item) is not ArtifactValidationInput for item in artifacts
        ):
            raise ValueError("extended golden origin requires eight artifacts")
        if (
            type(self.validation) is not ValidationReport
            or not self.validation.all_pass
        ):
            raise ValueError("extended golden origin must pass T043")
        object.__setattr__(self, "artifacts", artifacts)


def build_extended_origin(
    case: ExtendedGoldenOriginCase,
    *,
    config: ConfigBundle,
    joined: JoinedInputRecord,
    manifest: VerifiedSplitManifest,
    token_writer: TokenLabelSetWriter,
    global_seed: int,
) -> ExtendedGoldenOriginBuild:
    """Build one reusable real-origin T019--T043 golden bundle."""

    if type(case) is not ExtendedGoldenOriginCase:
        raise TypeError("case must be ExtendedGoldenOriginCase")
    golden = _build_origin(
        config,
        joined,
        case.spec,
        global_seed=global_seed,
    )
    records_by_id = {record.record_id: record for record in golden.bundle.records}
    artifacts = tuple(
        build_extended_record_artifact(
            record,
            manifest=manifest,
            token_writer=token_writer,
            matched_hallucinated_record=(
                records_by_id[record.matched_record_id]
                if record.variant_label is VariantLabel.FAITHFUL
                else None
            ),
        )
        for record in golden.bundle.records
    )
    value = BundleValidationInput(
        bundle=golden.bundle,
        artifacts=artifacts,
        split_manifest=manifest,
    )
    report = ValidatorChain().validate_bundle_strict(value)
    return ExtendedGoldenOriginBuild(case, golden, artifacts, report)


def _verified_manifest(
    records: Sequence[JoinedInputRecord],
    manifest_root: Path,
) -> VerifiedSplitManifest:
    validated: list[OriginValidationInput] = []
    for record in records:
        reference = build_reference_dag(record)
        validated.append(
            OriginValidationInput(
                record=record,
                artifact=reference,
                edit_truth=derive_edit_truth(reference),
            )
        )
    audit = audit_origin_split_features(validated).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in validated
        },
    )
    split = build_group_stratified_split(audit, leakage)
    return load_verified_split_manifest(
        manifest_root / "split_manifest.csv",
        manifest_root / "split_manifest.metadata.json",
        split_result=split,
        audit=audit,
    )


def _delete_with_replacement_diagnostic(
    joined: JoinedInputRecord,
    config: ConfigBundle,
) -> dict[str, Any]:
    reference = build_reference_dag(joined)
    truth = derive_edit_truth(reference)
    classification = classify_edit_truth(truth)
    record = task_record_from_joined_input(joined)
    engine = DeletionCandidateEngine(operators_config=config.operators)
    perturbator = DeletionPerturbator(
        candidate_engine=engine,
        propagator=EditingPropagationEngine(),
        renderer=_UnusedTraceRenderer(),
        validators=_UnusedValidatorChain(),
        label_projector=_UnusedLabelProjector(),
    )
    registry = PerturbatorRegistry.from_perturbator_types(
        (DeletionPerturbator,),
        operators_config=config.operators,
    )
    operator_id = DELETION_OPERATOR_IDS[0]
    recipe = PerturbationRecipe(
        recipe_id="t044:delete-with-replacement:blocked-structural",
        origin_id=record.origin_id,
        operator_id=operator_id,
        policy=PropagationPolicy.STOP,
        target_node_id="product",
        candidate_source_mode=CandidateSourceType.RULE,
        variant_index=0,
        derived_seed=20260830,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=0,
            length_bucket="t044-anomaly-diagnostic",
        ),
        candidate_difficulty_bucket="anomaly",
        renderer_style_id="not-rendered",
    )
    context = PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=reference.state_dag.schema,
        reference_graph=reference.state_dag,
        truth=truth,
    )
    try:
        registry.resolve(perturbator, context)
    except OperatorRegistryError as error:
        if error.code != "OPERATOR_CAPABILITY_FORBIDDEN":
            raise
        rejection = error.to_dict()
    else:
        raise AssertionError("delete-with-replacement structural operator was accepted")
    registration = registry.registration(operator_id)
    return {
        "origin_id": DELETE_WITH_REPLACEMENT_ORIGIN_ID,
        "classification": classification.to_dict(),
        "attempted_registration": {
            "operator_id": registration.operator_id,
            "operator_family": registration.operator_family,
            "required_capabilities": registration.required_capabilities,
        },
        "attempt": {
            "policy": recipe.policy.dataset_name,
            "target_node_id": recipe.target_node_id,
            "candidate_source": recipe.candidate_source_mode.value,
        },
        "expected_rejection": rejection,
        "full_eight_record_status": "not_constructed_by_design",
        "unsupported_policy": PropagationPolicy.FULL_CF.dataset_name,
        "structural_operator_ids": DELETION_OPERATOR_IDS[:8],
        "permitted_claim_terminal_operator_ids": DELETION_OPERATOR_IDS[8:],
        "reason": (
            "replacement capability forbids structural deletion; remaining claim and "
            "terminal operators cannot supply a legal FULL_CF product"
        ),
    }


def _span(span: CharSpan | None) -> dict[str, int] | None:
    return None if span is None else {"start": span.start, "end": span.end}


def _annotation_snapshot(value: CharAnnotationBuildResult) -> dict[str, Any]:
    return {
        "builder_version": value.builder_version,
        "annotations": [
            {
                "span_id": item.span_id,
                "component": item.component.value,
                "step_index": item.step_index,
                "state_or_edge_id": item.state_or_edge_id,
                "literal_span": _span(item.literal_span),
                "claim_span": _span(item.claim_span),
                "semantic_types": item.semantic_types,
                "edit_subtypes": item.edit_subtypes,
                "evidence_relations": item.evidence_relations,
                "causal_role": item.causal_role,
                "root_span_id": item.root_span_id,
            }
            for item in value.annotations
        ],
        "event_links": [
            {"event_id": item.event_id, "span_ids": item.span_ids}
            for item in value.event_links
        ],
        "unlocalized_omissions": [
            {
                "event_id": item.event_id,
                "target_kind": item.target_kind.value,
                "state_or_edge_id": item.state_or_edge_id,
                "suppressed_mention_ids": item.suppressed_mention_ids,
                "reason": item.reason,
            }
            for item in value.unlocalized_omissions
        ],
    }


def _serialized_snapshot(value: SerializedDetectorInput) -> dict[str, Any]:
    return {
        "template_version": value.template_version,
        "text": value.text,
        "sha256": value.sha256,
        "segments": [
            {
                "field_name": item.field_name,
                "segment_kind": item.segment_kind.value,
                "start": item.start,
                "end": item.end,
            }
            for item in value.segments
        ],
    }


def _token_snapshot(value: TokenLabelSet) -> dict[str, Any]:
    fingerprint = value.tokenizer_fingerprint
    return {
        "activation_alignment": value.activation_alignment,
        "tokenizer_fingerprint": {
            "tokenizer_name": fingerprint.tokenizer_name,
            "tokenizer_revision": fingerprint.tokenizer_revision,
            "tokenizer_vocab_hash": fingerprint.tokenizer_vocab_hash,
            "special_token_config": fingerprint.special_token_config,
            "normalization_config": fingerprint.normalization_config,
        },
        "serialized_text_sha256": value.serialized_text_sha256,
        "input_ids": value.input_ids,
        "attention_mask": value.attention_mask,
        "offset_mapping": value.offset_mapping,
        "segment_ids": value.segment_ids,
        "evaluation_mask": value.evaluation_mask,
        "hallucination_core_mask": value.hallucination_core_mask,
        "error_any_mask": value.error_any_mask,
        "semantic_type_masks": value.semantic_type_masks,
        "edit_subtype_masks": value.edit_subtype_masks,
        "causal_role_masks": value.causal_role_masks,
        "local_falsehood_mask": value.local_falsehood_mask,
        "off_task_branch_mask": value.off_task_branch_mask,
        "reasoning_mask": value.reasoning_mask,
        "answer_mask": value.answer_mask,
        "boundary_ambiguous_mask": value.boundary_ambiguous_mask,
        "error_char_fraction": value.error_char_fraction,
        "matched_target_span": _span(value.matched_target_span),
    }


def _report_snapshot(value: ValidationReport) -> dict[str, Any]:
    return {
        "validator_id": value.validator_id,
        "all_pass": value.all_pass,
        "issues": [
            {
                "code": item.code,
                "severity": item.severity.value,
                "stage": item.stage.value,
                "node_ids": item.node_ids,
                "message": item.message,
                "evidence": item.evidence,
            }
            for item in value.issues
        ],
    }


def _validation_snapshot(
    artifact: ArtifactValidationInput,
    chain: ValidatorChain,
) -> dict[str, Any]:
    reports = (
        chain.semantic.validate(artifact),
        chain.propagation.validate(artifact),
        chain.renderer.validate(artifact),
        chain.token_alignment.validate(artifact),
    )
    return {
        "chain": _report_snapshot(chain.validate_artifact(artifact)),
        "gates": [_report_snapshot(item) for item in reports],
    }


def _truth_snapshot(origin: GoldenOriginBundle) -> dict[str, Any]:
    truth = origin.executions[0].context.truth
    return {
        "optimal_mapping_count": truth.mapping_evidence.optimal_mapping_count,
        "inequivalent_edit_signature_count": (
            truth.mapping_evidence.inequivalent_edit_signature_count
        ),
        "valid_anchor_indices": truth.valid_anchor_indices,
        "symmetry_equivalent_anchors": truth.symmetry_equivalent_anchors,
        "removed_atom_count": len(truth.removed_atom_maps),
        "added_atom_count": len(truth.added_atoms),
        "broken_boundary_bond_count": len(truth.broken_bonds),
        "formed_boundary_bond_count": len(truth.formed_bonds),
    }


def _record_snapshot(
    artifact: ArtifactValidationInput,
    chain: ValidatorChain,
) -> dict[str, Any]:
    labels = artifact.token_labels
    if labels is None:
        raise AssertionError("T044 record lost token labels after strict validation")
    return {
        "draft": artifact.draft.to_dict(),
        "state": {
            "schema_id": artifact.draft.locked_state.schema.schema_id,
            "locked_values": {
                key: value.normalized_value
                for key, value in artifact.draft.locked_state.values.items()
            },
            "semantic_difference_targets": artifact.draft.locked_state.semantic_differences(
                artifact.draft.reference_graph
            ),
        },
        "rendered": artifact.rendered.to_dict(),
        "char_annotations": _annotation_snapshot(artifact.char_annotations),
        "serialized_detector_input": _serialized_snapshot(artifact.serialized),
        "token_labels": _token_snapshot(labels),
        "trace_labels": {
            name: getattr(artifact.trace_labels, name)
            for name in artifact.trace_labels.__dataclass_fields__
        },
        "split": artifact.split.value,
        "leakage_group_id": artifact.leakage_group_id,
        "validation": _validation_snapshot(artifact, chain),
    }


@dataclass(frozen=True, slots=True)
class ExtendedGoldenSuiteBuild:
    dataset_version: str
    global_seed: int
    origins: tuple[ExtendedGoldenOriginBuild, ...]
    delete_with_replacement: Mapping[str, Any]

    def __post_init__(self) -> None:
        origins = tuple(self.origins)
        if len(origins) != 9:
            raise ValueError("T044 requires exactly nine complete real origins")
        counts = Counter(item.case.spec.normalized_subtask for item in origins)
        if counts != Counter({subtask: 3 for subtask in EditingSubtask}):
            raise ValueError("T044 requires three origins per editing subtask")
        if any(not item.validation.all_pass for item in origins):
            raise ValueError("every complete T044 origin must pass T043")
        if not isinstance(self.delete_with_replacement, Mapping):
            raise TypeError("delete_with_replacement must be a mapping")
        object.__setattr__(self, "origins", origins)

    @property
    def artifacts(self) -> tuple[ArtifactValidationInput, ...]:
        return tuple(item for origin in self.origins for item in origin.artifacts)

    def fixture_artifact(self) -> dict[str, Any]:
        chain = ValidatorChain()
        return {
            "format_version": T044_FIXTURE_FORMAT_VERSION,
            "dataset_version": self.dataset_version,
            "global_seed": self.global_seed,
            "execution": {
                "network_mode": "offline",
                "live_poe_attempted": False,
                "renderer": "natural_rule_v1",
                "tokenizer_backend": "deterministic_fast_offset_fixture",
                "deterministic_replay": True,
            },
            "coverage": {
                "complete_real_origin_count": len(self.origins),
                "complete_record_count": len(self.artifacts),
                "origins_per_subtask": {
                    subtask.value: sum(
                        item.case.spec.normalized_subtask is subtask
                        for item in self.origins
                    )
                    for subtask in EditingSubtask
                },
                "real_symmetry_equivalent_origin_count": sum(
                    bool(
                        item.golden.executions[
                            0
                        ].context.truth.symmetry_equivalent_anchors
                    )
                    for item in self.origins
                ),
                "symmetry_coverage_note": (
                    "the frozen real corpus has no non-empty symmetry-equivalent "
                    "anchor case; T044 covers real multi-mapping and typed dual-anchor "
                    "cases, while synthetic symmetry stays in the T013 truth tests"
                ),
            },
            "origin_bundles": [
                {
                    "case_id": item.case.case_id,
                    "case_kind": item.case.case_kind,
                    "coverage_tags": item.case.coverage_tags,
                    "origin_id": item.case.spec.origin_id,
                    "normalized_subtask": item.case.spec.normalized_subtask.value,
                    "truth_evidence": _truth_snapshot(item.golden),
                    "candidate_and_propagation_trace": [
                        execution.to_trace_dict()
                        for execution in item.golden.executions
                    ],
                    "records": [
                        _record_snapshot(artifact, chain) for artifact in item.artifacts
                    ],
                    "bundle_validation": _report_snapshot(item.validation),
                }
                for item in self.origins
            ],
            "delete_with_replacement_expected_rejection": (
                self.delete_with_replacement
            ),
        }

    def validation_report(self) -> dict[str, Any]:
        terminal = tuple(
            execution
            for origin in self.origins
            for execution in origin.golden.executions
            if execution.context.recipe.policy is PropagationPolicy.TERMINAL
        )
        candidate_counts = tuple(
            len(execution.pool.candidates)
            for origin in self.origins
            for execution in origin.golden.executions
        )
        return {
            "format_version": T044_REPORT_FORMAT_VERSION,
            "dataset_version": self.dataset_version,
            "all_pass": True,
            "summary": {
                "complete_real_origin_count": len(self.origins),
                "record_count": len(self.artifacts),
                "hallucinated_record_count": sum(
                    item.draft.variant_label is VariantLabel.HALLUCINATED
                    for item in self.artifacts
                ),
                "faithful_record_count": sum(
                    item.draft.variant_label is VariantLabel.FAITHFUL
                    for item in self.artifacts
                ),
                "validated_artifact_gate_count": (
                    len(self.artifacts) * len(ARTIFACT_VALIDATOR_IDS)
                ),
                "validated_bundle_gate_count": len(self.origins),
                "minimum_candidate_pool_size": min(candidate_counts),
                "maximum_candidate_pool_size": max(candidate_counts),
                "multi_candidate_execution_count": sum(
                    count > 1 for count in candidate_counts
                ),
                "terminal_near_miss_count": len(terminal),
                "terminal_selected_is_pool_similarity_max": all(
                    execution.to_trace_dict()["candidate_pool"][
                        "selected_answer_similarity"
                    ]
                    == execution.to_trace_dict()["candidate_pool"][
                        "max_answer_similarity"
                    ]
                    for execution in terminal
                ),
                "delete_with_replacement_expected_rejection_count": 1,
                "live_poe_attempt_count": 0,
            },
            "required_validator_ids": (
                *ARTIFACT_VALIDATOR_IDS,
                BUNDLE_INTEGRITY_VALIDATOR_ID,
            ),
            "origins": [
                {
                    "origin_id": item.case.spec.origin_id,
                    "case_id": item.case.case_id,
                    "subtask": item.case.spec.normalized_subtask.value,
                    "record_count": len(item.artifacts),
                    "all_pass": item.validation.all_pass,
                    "issue_codes": tuple(
                        issue.code for issue in item.validation.issues
                    ),
                    "candidate_pool_sizes": tuple(
                        len(execution.pool.candidates)
                        for execution in item.golden.executions
                    ),
                }
                for item in self.origins
            ],
            "delete_with_replacement": self.delete_with_replacement,
        }

    def render_fixture_json(self) -> str:
        return _render_json(self.fixture_artifact())

    def render_report_json(self) -> str:
        return _render_json(self.validation_report())


def build_t044_extended_golden_suite(
    dataset_root: Path | None = None,
    manifest_root: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    cases: tuple[ExtendedGoldenOriginCase, ...] = T044_GOLDEN_ORIGIN_CASES,
) -> ExtendedGoldenSuiteBuild:
    """Replay nine real origins and the registered expected rejection."""

    root = DEFAULT_DATASET_ROOT if dataset_root is None else Path(dataset_root)
    manifest_dir = (
        DEFAULT_MANIFEST_ROOT if manifest_root is None else Path(manifest_root)
    )
    loaded_config = load_config_bundle() if config is None else config
    if type(loaded_config) is not ConfigBundle:
        raise TypeError("config must be ConfigBundle or None")
    selected_cases = tuple(cases)
    if any(type(item) is not ExtendedGoldenOriginCase for item in selected_cases):
        raise TypeError("cases must contain ExtendedGoldenOriginCase values")
    records = ChemCoTMolEditAdapter().load(root)
    indexed = _records_by_id(records)
    manifest = _verified_manifest(records, manifest_dir)
    writer = TokenLabelSetWriter(
        _FrozenWhitespaceTokenizer(),
        T044_TOKENIZER_FINGERPRINT,
    )
    origins = []
    for case in selected_cases:
        joined = indexed.get(case.spec.origin_id)
        if joined is None:
            raise ValueError(f"missing T044 origin {case.spec.origin_id!r}")
        origins.append(
            build_extended_origin(
                case,
                config=loaded_config,
                joined=joined,
                manifest=manifest,
                token_writer=writer,
                global_seed=loaded_config.dataset.dataset.global_seed,
            )
        )
    replacement = indexed.get(DELETE_WITH_REPLACEMENT_ORIGIN_ID)
    if replacement is None:
        raise ValueError("missing registered delete-with-replacement origin")
    return ExtendedGoldenSuiteBuild(
        dataset_version=loaded_config.dataset.dataset.version_name,
        global_seed=loaded_config.dataset.dataset.global_seed,
        origins=tuple(origins),
        delete_with_replacement=_delete_with_replacement_diagnostic(
            replacement,
            loaded_config,
        ),
    )


def write_t044_golden_artifacts(
    dataset_root: Path | None = None,
    manifest_root: Path | None = None,
    fixture_path: Path | None = None,
    report_path: Path | None = None,
) -> ExtendedGoldenSuiteBuild:
    """Build and write deterministic UTF-8 T044 fixture and report files."""

    suite = build_t044_extended_golden_suite(dataset_root, manifest_root)
    fixture = DEFAULT_FIXTURE_PATH if fixture_path is None else Path(fixture_path)
    report = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    for path, payload in (
        (fixture, suite.render_fixture_json()),
        (report, suite.render_report_json()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
    return suite


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_FIXTURE_PATH",
    "DEFAULT_MANIFEST_ROOT",
    "DEFAULT_REPORT_PATH",
    "DELETE_WITH_REPLACEMENT_ORIGIN_ID",
    "T044_FIXTURE_FORMAT_VERSION",
    "T044_GOLDEN_ORIGIN_CASES",
    "T044_REPORT_FORMAT_VERSION",
    "T044_TOKENIZER_FINGERPRINT",
    "ExtendedGoldenOriginBuild",
    "ExtendedGoldenOriginCase",
    "ExtendedGoldenSuiteBuild",
    "build_extended_origin",
    "build_extended_record_artifact",
    "build_t044_extended_golden_suite",
    "render_extended_golden_record",
    "write_t044_golden_artifacts",
]
