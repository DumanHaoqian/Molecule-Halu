"""Deterministic validation chain for completed molecule-editing artifacts.

The first four validators operate on one post-render artifact assembled from
the frozen T024 draft, the T040 rendered trace, T041 character annotations,
and the T042 tokenizer-specific labels.  The final validator audits all eight
records together.  Every rejection is represented by a structured domain
``ValidationIssue``; no validator repairs or relaxes an invalid artifact.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from molhallulens.annotation.char_annotations import (
    CharAnnotationBuildResult,
    is_pure_omission,
)
from molhallulens.annotation.token_projection import (
    DetectorCoordinateMap,
    rebase_char_annotations,
)
from molhallulens.builders.bundles import MatchedBundleDraft, MatchedDraftRecord
from molhallulens.builders.split_manifest import VerifiedSplitManifest
from molhallulens.builders.splitter import SplitName
from molhallulens.chemistry import compute_descriptors, isomeric_graph_equivalent
from molhallulens.domain import (
    CausalRole,
    CharAnnotation,
    EditErrorSubtype,
    HallucinationType,
    MutationTargetKind,
    PropagationPolicy,
    SegmentKind,
    Severity,
    TokenLabelSet,
    TraceLabels,
    ValidationIssue,
    ValidationReport,
    ValidationStage,
    VariantLabel,
    Visibility,
)
from molhallulens.domain.errors import ArtifactValidationError
from molhallulens.rendering.detector_prompt import SerializedDetectorInput
from molhallulens.rendering.natural_rule import scan_label_leakage
from molhallulens.rendering.trace_ast import RenderedExample

SEMANTIC_VALIDATOR_ID = "molhallulens.validation.hallucination_semantics.v1"
PROPAGATION_VALIDATOR_ID = "molhallulens.validation.propagation.v1"
RENDERER_VALIDATOR_ID = "molhallulens.validation.renderer.v1"
TOKEN_ALIGNMENT_VALIDATOR_ID = "molhallulens.validation.token_alignment.v1"
BUNDLE_INTEGRITY_VALIDATOR_ID = "molhallulens.validation.bundle_integrity.v1"
VALIDATOR_CHAIN_ID = "molhallulens.validation.artifact_chain.v1"

ARTIFACT_VALIDATOR_IDS = (
    SEMANTIC_VALIDATOR_ID,
    PROPAGATION_VALIDATOR_ID,
    RENDERER_VALIDATOR_ID,
    TOKEN_ALIGNMENT_VALIDATOR_ID,
)

_POLICIES = (
    PropagationPolicy.STOP,
    PropagationPolicy.PARTIAL,
    PropagationPolicy.FULL_CF,
    PropagationPolicy.TERMINAL,
)
_POSITIVE_SEGMENTS = frozenset({SegmentKind.REASONING, SegmentKind.FINAL_ANSWER})
_HEADER_SEPARATOR = r"[ _-]*"
_HEADER_VALUE_KIND = r"(?:answer|product|smiles|state|value)"
_REFERENCE_HEADER = re.compile(
    rf"(?im)(?:^|\n)[ \t]*(?:"
    rf"(?:<|\[)[ \t]*(?:"
    rf"gt(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|ground{_HEADER_SEPARATOR}truth"
    rf"(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|reference(?:{_HEADER_SEPARATOR}only)?"
    rf"(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|oracle(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf")[ \t]*(?:>|\])"
    rf"|(?:"
    rf"gt(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|ground{_HEADER_SEPARATOR}truth"
    rf"(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|reference(?:{_HEADER_SEPARATOR}only)?"
    rf"(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf"|oracle(?:{_HEADER_SEPARATOR}{_HEADER_VALUE_KIND})?"
    rf")[ \t]*(?::|=|->|\|)"
    rf")"
)


@dataclass(frozen=True, slots=True)
class ArtifactValidationInput:
    """One complete pre-``PerturbationResult`` artifact to audit."""

    draft: MatchedDraftRecord
    rendered: RenderedExample
    char_annotations: CharAnnotationBuildResult
    serialized: SerializedDetectorInput
    token_labels: TokenLabelSet | None
    trace_labels: TraceLabels
    split: SplitName
    leakage_group_id: str

    def __post_init__(self) -> None:
        for value, expected, name in (
            (self.draft, MatchedDraftRecord, "draft"),
            (self.rendered, RenderedExample, "rendered"),
            (
                self.char_annotations,
                CharAnnotationBuildResult,
                "char_annotations",
            ),
            (self.serialized, SerializedDetectorInput, "serialized"),
            (self.trace_labels, TraceLabels, "trace_labels"),
        ):
            if type(value) is not expected:
                raise TypeError(
                    f"ArtifactValidationInput {name} must be {expected.__name__}"
                )
        if (
            self.token_labels is not None
            and type(self.token_labels) is not TokenLabelSet
        ):
            raise TypeError(
                "ArtifactValidationInput token_labels must be TokenLabelSet or None"
            )
        if type(self.split) is not SplitName:
            raise TypeError("ArtifactValidationInput split must be SplitName")
        if type(self.leakage_group_id) is not str or not self.leakage_group_id:
            raise ValueError("leakage_group_id must be non-empty text")

    @property
    def record_id(self) -> str:
        return self.draft.record_id


@dataclass(frozen=True, slots=True)
class BundleValidationInput:
    """One T024 bundle plus its eight rendered/tokenized split descendants."""

    bundle: MatchedBundleDraft
    artifacts: tuple[ArtifactValidationInput, ...]
    split_manifest: VerifiedSplitManifest

    def __post_init__(self) -> None:
        if type(self.bundle) is not MatchedBundleDraft:
            raise TypeError("bundle must be MatchedBundleDraft")
        artifacts = tuple(self.artifacts)
        if any(type(item) is not ArtifactValidationInput for item in artifacts):
            raise TypeError("artifacts must contain ArtifactValidationInput values")
        if type(self.split_manifest) is not VerifiedSplitManifest:
            raise TypeError(
                "split_manifest must be a loader-verified VerifiedSplitManifest"
            )
        object.__setattr__(self, "artifacts", artifacts)


def _stable_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Enum):
        return ("enum", value.value)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _stable_key(item)) for key, item in value.items())),
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return ("sequence", tuple(sorted(_stable_key(item) for item in value)))
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _report(validator_id: str, issues: Iterable[ValidationIssue]) -> ValidationReport:
    return ValidationReport(
        validator_id,
        tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.node_ids,
                    item.stage.value,
                    item.code,
                    item.message,
                    _stable_key(item.evidence),
                ),
            )
        ),
    )


def _add(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    *,
    stage: ValidationStage,
    node_ids: Sequence[str],
    severity: Severity = Severity.ERROR,
    **evidence: Any,
) -> None:
    issues.append(
        ValidationIssue(
            code=code,
            severity=severity,
            stage=stage,
            node_ids=tuple(dict.fromkeys(node_ids)),
            message=message,
            evidence={
                key: value for key, value in evidence.items() if value is not None
            },
        )
    )


def _targets(record: MatchedDraftRecord) -> frozenset[tuple[MutationTargetKind, str]]:
    return frozenset(
        (event.target_kind, event.node_or_edge_id)
        for event in record.graph_delta.events
    )


def _molecule_equivalent(left: object, right: object) -> bool | None:
    if type(left) is not str or not left or type(right) is not str or not right:
        return None
    try:
        return isomeric_graph_equivalent(left, right)
    except (RuntimeError, TypeError, ValueError):
        return None


def _all_label_arrays(labels: TokenLabelSet) -> tuple[tuple[object, ...], ...]:
    direct = (
        labels.hallucination_core_mask,
        labels.error_any_mask,
        labels.local_falsehood_mask,
        labels.off_task_branch_mask,
        labels.boundary_ambiguous_mask,
        labels.error_char_fraction,
    )
    mapped = (
        tuple(labels.semantic_type_masks.values())
        + tuple(labels.edit_subtype_masks.values())
        + tuple(labels.causal_role_masks.values())
    )
    return tuple(tuple(values) for values in (*direct, *mapped))


def _overlap(start: int, end: int, span: Any) -> int:
    return max(0, min(end, span.end) - max(start, span.start))


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


@dataclass(frozen=True, slots=True)
class HallucinationSemanticValidator:
    """Validate H roots/operators and fully faithful matched controls."""

    validator_id: ClassVar[str] = SEMANTIC_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.HALLUCINATION_SEMANTICS

    def validate(self, artifact: ArtifactValidationInput) -> ValidationReport:
        if type(artifact) is not ArtifactValidationInput:
            raise TypeError("semantic validator requires ArtifactValidationInput")
        record = artifact.draft
        node_ids = (record.record_id,)
        issues: list[ValidationIssue] = []
        try:
            differences = record.locked_state.semantic_differences(
                record.reference_graph
            )
        except (RuntimeError, TypeError, ValueError) as error:
            _add(
                issues,
                "SEMANTIC_STATE_COMPARISON_FAILED",
                "reference and locked state cannot be compared",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
                exception_type=type(error).__name__,
            )
            return _report(self.validator_id, issues)

        delta_targets = _targets(record)
        if differences != delta_targets:
            _add(
                issues,
                "SEMANTIC_DELTA_MISMATCH",
                "GraphDelta targets differ from locked/reference semantics",
                stage=self.stage,
                node_ids=node_ids,
                differences=tuple(
                    sorted((kind.value, target) for kind, target in differences)
                ),
                delta_targets=tuple(
                    sorted((kind.value, target) for kind, target in delta_targets)
                ),
            )

        events = tuple(record.graph_delta.events)
        linked_event_ids = {
            link.event_id for link in artifact.char_annotations.event_links
        }
        omission_event_ids = {
            omission.event_id
            for omission in artifact.char_annotations.unlocalized_omissions
        }
        expected_event_ids = {event.event_id for event in events}
        if linked_event_ids | omission_event_ids != expected_event_ids or (
            linked_event_ids & omission_event_ids
        ):
            _add(
                issues,
                "SEMANTIC_ANNOTATION_LEDGER_MISMATCH",
                "character annotation ledger does not partition mutation events",
                stage=self.stage,
                node_ids=node_ids,
                expected_event_ids=tuple(sorted(expected_event_ids)),
                linked_event_ids=tuple(sorted(linked_event_ids)),
                omission_event_ids=tuple(sorted(omission_event_ids)),
            )
        annotations_by_id = {
            annotation.span_id: annotation
            for annotation in artifact.char_annotations.annotations
        }
        links_by_event = {
            link.event_id: link for link in artifact.char_annotations.event_links
        }
        omissions_by_event = {
            omission.event_id: omission
            for omission in artifact.char_annotations.unlocalized_omissions
        }
        for event in events:
            omission = omissions_by_event.get(event.event_id)
            if omission is not None:
                if (
                    not is_pure_omission(event)
                    or omission.target_kind is not event.target_kind
                    or omission.state_or_edge_id != event.node_or_edge_id
                ):
                    _add(
                        issues,
                        "SEMANTIC_OMISSION_LEDGER_MISMATCH",
                        "unlocalized omission does not describe its mutation event",
                        stage=self.stage,
                        node_ids=node_ids,
                        event_id=event.event_id,
                    )
                continue
            link = links_by_event.get(event.event_id)
            linked = tuple(
                annotations_by_id[span_id]
                for span_id in (() if link is None else link.span_ids)
                if span_id in annotations_by_id
            )
            semantic_types = frozenset(
                label for annotation in linked for label in annotation.semantic_types
            )
            edit_subtypes = frozenset(
                label for annotation in linked for label in annotation.edit_subtypes
            )
            if (
                not linked
                or any(
                    annotation.state_or_edge_id != event.node_or_edge_id
                    or annotation.causal_role is not event.causal_role
                    or annotation.semantic_types != event.hallucination_types
                    or annotation.edit_subtypes != event.edit_subtypes
                    for annotation in linked
                )
                or semantic_types != event.hallucination_types
                or edit_subtypes != event.edit_subtypes
            ):
                _add(
                    issues,
                    "SEMANTIC_ANNOTATION_AXIS_MISMATCH",
                    "character annotations do not exactly carry their event target and axes",
                    stage=self.stage,
                    node_ids=node_ids,
                    event_id=event.event_id,
                )

        if record.variant_label is VariantLabel.FAITHFUL:
            faithful_flags = (
                artifact.trace_labels.reasoning_valid,
                artifact.trace_labels.answer_correct,
                artifact.trace_labels.chemically_valid,
                artifact.trace_labels.constraint_satisfied,
                artifact.trace_labels.format_valid,
                artifact.trace_labels.answer_complete,
            )
            if (
                differences
                or events
                or artifact.char_annotations.annotations
                or artifact.char_annotations.unlocalized_omissions
                or artifact.trace_labels.hallucination_present
                or not all(faithful_flags)
            ):
                _add(
                    issues,
                    "SEMANTIC_N_NOT_FAITHFUL",
                    "N control must preserve exact state and all faithful trace flags",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            return _report(self.validator_id, issues)

        if record.variant_label is not VariantLabel.HALLUCINATED:
            _add(
                issues,
                "SEMANTIC_VARIANT_UNKNOWN",
                "record variant is outside the frozen H/N vocabulary",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
            )
            return _report(self.validator_id, issues)

        roots = tuple(record.graph_delta.root_events)
        if len(roots) != 1:
            _add(
                issues,
                "SEMANTIC_ROOT_COUNT",
                "H artifact must contain exactly one independent root",
                stage=self.stage,
                node_ids=node_ids,
                root_count=len(roots),
            )
        else:
            root = roots[0]
            if root.node_or_edge_id != record.target_node_id:
                _add(
                    issues,
                    "SEMANTIC_ROOT_TARGET_MISMATCH",
                    "mutation root differs from the frozen T024 target",
                    stage=self.stage,
                    node_ids=node_ids,
                    expected_target=record.target_node_id,
                    actual_target=root.node_or_edge_id,
                )
            if root.target_kind is MutationTargetKind.NODE:
                reference_value = record.reference_graph.values.get(
                    root.node_or_edge_id
                )
                locked_value = record.locked_state.values.get(root.node_or_edge_id)
            else:
                reference_value = record.reference_graph.edge_values.get(
                    root.node_or_edge_id
                )
                locked_value = record.locked_state.edge_values.get(root.node_or_edge_id)
            if (
                reference_value is None
                or locked_value is None
                or not root.before.semantically_equals(reference_value)
                or not root.after.semantically_equals(locked_value)
                or root.before.semantically_equals(root.after)
            ):
                _add(
                    issues,
                    "SEMANTIC_ROOT_NOT_WRONG",
                    "root event is not an actual locked deviation from reference",
                    stage=self.stage,
                    node_ids=node_ids,
                    target=root.node_or_edge_id,
                )

        if not differences or not artifact.trace_labels.hallucination_present:
            _add(
                issues,
                "SEMANTIC_H_MISSING_ERROR",
                "H artifact requires a semantic deviation and positive trace label",
                stage=self.stage,
                node_ids=node_ids,
            )
        wrong_operators = tuple(
            sorted(
                {
                    event.operator_id
                    for event in events
                    if event.operator_id != record.operator_id
                }
            )
        )
        if wrong_operators:
            _add(
                issues,
                "SEMANTIC_OPERATOR_MISMATCH",
                "mutation events differ from the frozen operator identity",
                stage=self.stage,
                node_ids=node_ids,
                expected_operator=record.operator_id,
                actual_operators=wrong_operators,
            )

        if record.policy is PropagationPolicy.STOP and (
            artifact.trace_labels.reasoning_valid
            or not artifact.trace_labels.answer_correct
        ):
            _add(
                issues,
                "SEMANTIC_LOCAL_TRACE_LABELS",
                "H_LOCAL requires wrong reasoning and a correct answer",
                stage=self.stage,
                node_ids=node_ids,
            )
        elif record.policy is PropagationPolicy.PARTIAL and (
            artifact.trace_labels.reasoning_valid
        ):
            _add(
                issues,
                "SEMANTIC_PARTIAL_TRACE_LABELS",
                "H_PARTIAL must expose a reasoning error",
                stage=self.stage,
                node_ids=node_ids,
            )
        elif record.policy is PropagationPolicy.FULL_CF and (
            artifact.trace_labels.answer_correct
            or artifact.trace_labels.constraint_satisfied
            or not artifact.trace_labels.chemically_valid
        ):
            _add(
                issues,
                "SEMANTIC_FULL_CF_TRACE_LABELS",
                "H_FULL_CF must be chemically valid but off-task with a wrong answer",
                stage=self.stage,
                node_ids=node_ids,
            )
        elif record.policy is PropagationPolicy.TERMINAL and (
            not artifact.trace_labels.reasoning_valid
            or artifact.trace_labels.answer_correct
        ):
            _add(
                issues,
                "SEMANTIC_TERMINAL_TRACE_LABELS",
                "H_TERMINAL requires valid reasoning and an incorrect answer",
                stage=self.stage,
                node_ids=node_ids,
            )

        return _report(self.validator_id, issues)


@dataclass(frozen=True, slots=True)
class PropagationValidator:
    """Validate exact LOCAL/PARTIAL/FULL_CF/TERMINAL state phenotypes."""

    validator_id: ClassVar[str] = PROPAGATION_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.PROPAGATION

    def validate(self, artifact: ArtifactValidationInput) -> ValidationReport:
        if type(artifact) is not ArtifactValidationInput:
            raise TypeError("propagation validator requires ArtifactValidationInput")
        record = artifact.draft
        node_ids = (record.record_id,)
        issues: list[ValidationIssue] = []
        differences = record.locked_state.semantic_differences(record.reference_graph)
        if record.variant_label is VariantLabel.FAITHFUL:
            if differences or record.graph_delta.events:
                _add(
                    issues,
                    "PROPAGATION_N_STATE_DRIFT",
                    "faithful control may not contain propagated state changes",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            return _report(self.validator_id, issues)

        roots = tuple(record.graph_delta.root_events)
        if len(roots) != 1 or roots[0].target_kind is not MutationTargetKind.NODE:
            _add(
                issues,
                "PROPAGATION_ROOT_INVALID",
                "editing propagation requires one typed state-node root",
                stage=self.stage,
                node_ids=node_ids,
            )
            return _report(self.validator_id, issues)
        root = roots[0]
        changed_nodes = {
            target for kind, target in differences if kind is MutationTargetKind.NODE
        }
        if len(changed_nodes) != len(differences):
            _add(
                issues,
                "PROPAGATION_EDGE_MUTATION",
                "editing phenotype cannot mutate edge claims",
                stage=self.stage,
                node_ids=node_ids,
            )
        schema = record.locked_state.schema
        full_closure = tuple(
            target
            for target in schema.topological_order()
            if target in set(schema.dependency_closure((root.node_or_edge_id,)))
            and (
                target == root.node_or_edge_id
                or (
                    schema.nodes_by_id[target].mutable
                    and schema.nodes_by_id[target].visibility
                    is Visibility.CANDIDATE_OUTPUT
                )
            )
        )
        full_set = set(full_closure)

        roles = {event.causal_role for event in record.graph_delta.events[1:]}
        if record.policy is PropagationPolicy.STOP:
            if changed_nodes != {root.node_or_edge_id} or len(differences) != 1:
                _add(
                    issues,
                    "PROPAGATION_LOCAL_CHANGED_SET",
                    "H_LOCAL changed set must equal its single root",
                    stage=self.stage,
                    node_ids=node_ids,
                    changed_nodes=tuple(sorted(changed_nodes)),
                )
            if root.causal_role is not CausalRole.ROOT:
                _add(
                    issues,
                    "PROPAGATION_LOCAL_ROLE",
                    "H_LOCAL root must carry the ROOT causal role",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            reference_product = record.reference_graph.value_for(
                "product"
            ).normalized_value
            answer = record.locked_state.value_for("final_answer").normalized_value
            equivalent = _molecule_equivalent(answer, reference_product)
            if equivalent is not True:
                _add(
                    issues,
                    "PROPAGATION_LOCAL_ANSWER_DRIFT",
                    "H_LOCAL answer must remain graph-equivalent to reference product",
                    stage=self.stage,
                    node_ids=node_ids,
                    comparison_known=equivalent is not None,
                )

        elif record.policy is PropagationPolicy.PARTIAL:
            connected = False
            try:
                connected = schema.is_connected_downstream_subgraph(
                    {root.node_or_edge_id}, changed_nodes
                )
            except (KeyError, TypeError, ValueError):
                connected = False
            if not (len(changed_nodes) > 1 and changed_nodes < full_set and connected):
                _add(
                    issues,
                    "PROPAGATION_PARTIAL_CHANGED_SET",
                    "H_PARTIAL changes must be a nontrivial strict connected root subgraph",
                    stage=self.stage,
                    node_ids=node_ids,
                    changed_nodes=tuple(sorted(changed_nodes)),
                    full_closure=full_closure,
                    connected=connected,
                )
            if not roles <= {
                CausalRole.PROPAGATED_FALSE,
                CausalRole.PROPAGATED_CONDITIONAL,
            }:
                _add(
                    issues,
                    "PROPAGATION_PARTIAL_ROLE",
                    "H_PARTIAL descendants use an invalid causal role",
                    stage=self.stage,
                    node_ids=node_ids,
                    roles=tuple(sorted(role.value for role in roles)),
                )

        elif record.policy is PropagationPolicy.FULL_CF:
            if not (
                root.node_or_edge_id in changed_nodes
                and changed_nodes <= full_set
                and "product" in changed_nodes
            ):
                _add(
                    issues,
                    "PROPAGATION_FULL_CHANGED_SET",
                    "H_FULL_CF semantic changes must stay in the complete root closure and alter product",
                    stage=self.stage,
                    node_ids=node_ids,
                    changed_nodes=tuple(sorted(changed_nodes)),
                    full_closure=full_closure,
                )
            if roles - {CausalRole.PROPAGATED_CONDITIONAL}:
                _add(
                    issues,
                    "PROPAGATION_FULL_ROLE",
                    "H_FULL_CF descendants must be conditionally valid in the candidate world",
                    stage=self.stage,
                    node_ids=node_ids,
                    roles=tuple(sorted(role.value for role in roles)),
                )
            if any(
                event.causal_role is CausalRole.PROPAGATED_CONDITIONAL
                and event.after.locally_valid is not True
                for event in record.graph_delta.events[1:]
            ):
                _add(
                    issues,
                    "PROPAGATION_FULL_INCOHERENT",
                    "H_FULL_CF descendant claims are not locally valid",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            self._validate_full_chemistry(record, issues)

        elif record.policy is PropagationPolicy.TERMINAL:
            expected = {(MutationTargetKind.NODE, "final_answer")}
            if differences != expected or root.causal_role is not CausalRole.TERMINAL:
                _add(
                    issues,
                    "PROPAGATION_TERMINAL_CHANGED_SET",
                    "H_TERMINAL may change only final_answer with TERMINAL role",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            reasoning_drift = tuple(
                sorted(
                    node_id
                    for node_id, reference_value in record.reference_graph.values.items()
                    if node_id != "final_answer"
                    and record.locked_state.values.get(node_id) != reference_value
                )
            )
            if reasoning_drift or (
                record.locked_state.edge_values != record.reference_graph.edge_values
            ):
                _add(
                    issues,
                    "PROPAGATION_TERMINAL_REASONING_DRIFT",
                    "H_TERMINAL must preserve the complete reasoning graph",
                    stage=self.stage,
                    node_ids=node_ids,
                    reasoning_drift=reasoning_drift,
                )
            reference_product = record.reference_graph.value_for(
                "product"
            ).normalized_value
            answer = record.locked_state.value_for("final_answer").normalized_value
            equivalent = _molecule_equivalent(answer, reference_product)
            if equivalent is not False:
                _add(
                    issues,
                    "PROPAGATION_TERMINAL_ANSWER_EQUIVALENT",
                    "H_TERMINAL answer must be known non-equivalent to reference product",
                    stage=self.stage,
                    node_ids=node_ids,
                    comparison_known=equivalent is not None,
                )
        else:
            _add(
                issues,
                "PROPAGATION_POLICY_UNKNOWN",
                "artifact uses an unsupported propagation policy",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
            )
        return _report(self.validator_id, issues)

    def _validate_full_chemistry(
        self,
        record: MatchedDraftRecord,
        issues: list[ValidationIssue],
    ) -> None:
        node_ids = (record.record_id,)
        candidate = record.locked_state
        reference = record.reference_graph
        product = candidate.value_for("product").normalized_value
        reference_product = reference.value_for("product").normalized_value
        answer = candidate.value_for("final_answer").normalized_value
        product_differs = _molecule_equivalent(product, reference_product)
        answer_matches = _molecule_equivalent(answer, product)
        if product_differs is not False or answer_matches is not True:
            _add(
                issues,
                "PROPAGATION_FULL_PRODUCT_ANSWER",
                "H_FULL_CF requires a wrong product and an equivalent candidate answer",
                stage=self.stage,
                node_ids=node_ids,
                product_comparison_known=product_differs is not None,
                answer_comparison_known=answer_matches is not None,
            )
            return
        try:
            descriptors = compute_descriptors(product)
            expected = {
                "product_heavy": descriptors.heavy_atom_count,
                "product_rings": descriptors.ring_count,
                "heavy_delta": descriptors.heavy_atom_count
                - candidate.value_for("source_heavy").normalized_value,
                "ring_delta": descriptors.ring_count
                - candidate.value_for("source_rings").normalized_value,
            }
            mismatches = tuple(
                sorted(
                    node_id
                    for node_id, expected_value in expected.items()
                    if candidate.value_for(node_id).normalized_value != expected_value
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            _add(
                issues,
                "PROPAGATION_FULL_CHEMISTRY_UNKNOWN",
                "H_FULL_CF product descriptors cannot be independently recomputed",
                stage=self.stage,
                node_ids=node_ids,
                exception_type=type(error).__name__,
            )
            return
        if mismatches:
            _add(
                issues,
                "PROPAGATION_FULL_DESCRIPTOR_MISMATCH",
                "H_FULL_CF downstream counts/deltas disagree with candidate product",
                stage=self.stage,
                node_ids=node_ids,
                mismatched_nodes=mismatches,
            )


@dataclass(frozen=True, slots=True)
class RendererValidator:
    """Validate T040/state surface identity and scan detector-only text."""

    validator_id: ClassVar[str] = RENDERER_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.RENDERER

    def validate(self, artifact: ArtifactValidationInput) -> ValidationReport:
        if type(artifact) is not ArtifactValidationInput:
            raise TypeError("renderer validator requires ArtifactValidationInput")
        record = artifact.draft
        node_ids = (record.record_id,)
        issues: list[ValidationIssue] = []
        rendered = artifact.rendered
        serialized = artifact.serialized

        leaked = tuple(
            sorted(
                set(scan_label_leakage(rendered.detector_text))
                | set(scan_label_leakage(serialized.text))
            )
        )
        if leaked:
            _add(
                issues,
                "RENDERER_LABEL_LEAKAGE",
                "detector-visible text contains frozen label/reviewer leakage phrases",
                stage=self.stage,
                node_ids=node_ids,
                phrases=leaked,
            )
        headers = tuple(
            match.group(0).strip()
            for match in _REFERENCE_HEADER.finditer(serialized.text)
        )
        if headers:
            _add(
                issues,
                "RENDERER_REFERENCE_HEADER",
                "detector-visible text contains a GT/reference-only header",
                stage=self.stage,
                node_ids=node_ids,
                headers=headers,
            )

        reasoning_segments = tuple(
            sorted(
                (
                    (segment_id, span)
                    for segment_id, span in rendered.segment_spans.items()
                    if segment_id.startswith("reasoning.step.")
                ),
                key=lambda item: item[1].start,
            )
        )
        answer_span = rendered.segment_spans.get("final_answer")
        if not reasoning_segments or answer_span is None:
            _add(
                issues,
                "RENDERER_SEGMENTS_MISSING",
                "rendered trace must expose reasoning steps and final_answer",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
            )
        else:
            first_span = reasoning_segments[0][1]
            last_span = reasoning_segments[-1][1]
            reasoning_text = rendered.detector_text[first_span.start : last_span.end]
            detector = serialized.detector_input
            if first_span.start != 0 or detector.reasoning_chain != reasoning_text:
                _add(
                    issues,
                    "RENDERER_REASONING_MISMATCH",
                    "serialized reasoning is not the exact T040 reasoning surface",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            expected_answer = record.answer.answer_line
            actual_answer = rendered.detector_text[answer_span.start : answer_span.end]
            if (
                actual_answer != expected_answer
                or detector.final_answer != record.answer.smiles
            ):
                _add(
                    issues,
                    "RENDERER_ANSWER_MISMATCH",
                    "rendered/serialized Answer differs from the locked T024 answer",
                    stage=self.stage,
                    node_ids=node_ids,
                )

        for formal_step in record.formal_trace.steps:
            segment = rendered.segment_spans.get(
                f"reasoning.step.{formal_step.step_index:02d}"
            )
            expected_suffix = f"\n  FORMAL: {formal_step.formal_ab}"
            if segment is None or not rendered.detector_text[
                segment.start : segment.end
            ].endswith(expected_suffix):
                _add(
                    issues,
                    "RENDERER_FORMAL_MISMATCH",
                    "natural trace omitted or changed a locked FORMAL step",
                    stage=self.stage,
                    node_ids=node_ids,
                    step_index=formal_step.step_index,
                )

        formal_ranges = []
        for formal_step in record.formal_trace.steps:
            segment = rendered.segment_spans.get(
                f"reasoning.step.{formal_step.step_index:02d}"
            )
            if segment is not None and len(formal_step.formal_ab) <= segment.length:
                formal_ranges.append(
                    (
                        segment.end - len(formal_step.formal_ab),
                        segment.end,
                    )
                )
        for mention in rendered.mentions:
            # The exact FORMAL suffix was independently regenerated from the
            # locked state above.  Its literals may use grammar-specific
            # surfaces, so only natural/final mentions need scalar binding.
            if any(
                start <= mention.claim_span.start and mention.claim_span.end <= end
                for start, end in formal_ranges
            ):
                continue
            values = (
                record.locked_state.values
                if mention.target_kind is MutationTargetKind.NODE
                else record.locked_state.edge_values
            )
            locked = values.get(mention.state_or_edge_id)
            if locked is None:
                _add(
                    issues,
                    "RENDERER_UNKNOWN_MENTION_TARGET",
                    "natural/Answer mention does not resolve to the locked state",
                    stage=self.stage,
                    node_ids=node_ids,
                    mention_id=mention.mention_id,
                    target_kind=mention.target_kind.value,
                    target_id=mention.state_or_edge_id,
                )
                continue
            normalized = locked.normalized_value
            allowed_surfaces = {str(normalized)}
            if type(normalized) is int and normalized > 0:
                allowed_surfaces.add(f"+{normalized}")
            if mention.literal_text not in allowed_surfaces:
                _add(
                    issues,
                    "RENDERER_LOCKED_VALUE_MISMATCH",
                    "natural/Answer mention differs from its locked state value",
                    stage=self.stage,
                    node_ids=node_ids,
                    mention_id=mention.mention_id,
                    target_kind=mention.target_kind.value,
                    target_id=mention.state_or_edge_id,
                    observed_surface=mention.literal_text,
                )

        mentions_by_id = {mention.mention_id: mention for mention in rendered.mentions}
        for annotation in artifact.char_annotations.annotations:
            mention_id = annotation.span_id.removeprefix("char:")
            mention = mentions_by_id.get(mention_id)
            if (
                mention is None
                or mention.state_or_edge_id != annotation.state_or_edge_id
                or mention.component is not annotation.component
                or mention.step_index != annotation.step_index
                or mention.literal_span != annotation.literal_span
                or mention.claim_span != annotation.claim_span
            ):
                _add(
                    issues,
                    "RENDERER_ANNOTATION_MISMATCH",
                    "T041 annotation no longer identifies its exact T040 mention",
                    stage=self.stage,
                    node_ids=node_ids,
                    span_id=annotation.span_id,
                )
                continue
            if (
                rendered.detector_text[
                    annotation.literal_span.start : annotation.literal_span.end
                ]
                != mention.literal_text
            ):
                _add(
                    issues,
                    "RENDERER_SPAN_TEXT_MISMATCH",
                    "annotation literal span does not recover the rendered literal",
                    stage=self.stage,
                    node_ids=node_ids,
                    span_id=annotation.span_id,
                )

        if record.variant_label is VariantLabel.FAITHFUL and (
            artifact.char_annotations.annotations
            or artifact.char_annotations.unlocalized_omissions
        ):
            _add(
                issues,
                "RENDERER_N_POSITIVE_ANNOTATION",
                "faithful controls cannot carry positive character annotations",
                stage=self.stage,
                node_ids=node_ids,
            )
        return _report(self.validator_id, issues)


@dataclass(frozen=True, slots=True)
class TokenAlignmentValidator:
    """Recompute exact any-overlap masks from rebased canonical char spans."""

    validator_id: ClassVar[str] = TOKEN_ALIGNMENT_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.TOKEN_ALIGNMENT

    def validate(self, artifact: ArtifactValidationInput) -> ValidationReport:
        if type(artifact) is not ArtifactValidationInput:
            raise TypeError("token validator requires ArtifactValidationInput")
        record = artifact.draft
        node_ids = (record.record_id,)
        issues: list[ValidationIssue] = []
        labels = artifact.token_labels
        if labels is None:
            _add(
                issues,
                "TOKEN_LABELS_MISSING",
                "completed artifacts require a T042 TokenLabelSet",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
            )
            return _report(self.validator_id, issues)
        token_count = len(labels.input_ids)
        direct_arrays = (
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
        bad_lengths = tuple(
            name for name in direct_arrays if len(getattr(labels, name)) != token_count
        )
        mapped_lengths = tuple(
            f"{name}[{key.value}]"
            for name in (
                "semantic_type_masks",
                "edit_subtype_masks",
                "causal_role_masks",
            )
            for key, values in getattr(labels, name).items()
            if len(values) != token_count
        )
        if bad_lengths or mapped_lengths:
            _add(
                issues,
                "TOKEN_ARRAY_LENGTH_MISMATCH",
                "all direct arrays and taxonomy masks must match token length",
                stage=self.stage,
                node_ids=node_ids,
                fields=tuple(sorted((*bad_lengths, *mapped_lengths))),
            )
            return _report(self.validator_id, issues)

        if labels.serialized_text_sha256 != artifact.serialized.sha256:
            _add(
                issues,
                "TOKEN_TEXT_IDENTITY_MISMATCH",
                "token labels do not carry the serialized detector identity",
                stage=self.stage,
                node_ids=node_ids,
            )
        if labels.activation_alignment != "post_token_h_t":
            _add(
                issues,
                "TOKEN_ALIGNMENT_MODE_INVALID",
                "labels must align to the same post-token residual position",
                stage=self.stage,
                node_ids=node_ids,
            )

        try:
            rebased = rebase_char_annotations(
                artifact.rendered,
                artifact.serialized,
                artifact.char_annotations,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            _add(
                issues,
                "TOKEN_CHAR_REBASE_FAILED",
                "T040/T041 spans cannot be mapped into detector coordinates",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
                exception_type=type(error).__name__,
                error_code=getattr(error, "code", None),
            )
            return _report(self.validator_id, issues)

        annotations = tuple(rebased.annotations)
        if record.variant_label is VariantLabel.FAITHFUL:
            if annotations or any(any(values) for values in _all_label_arrays(labels)):
                _add(
                    issues,
                    "TOKEN_N_POSITIVE_LABEL",
                    "every faithful N taxonomy/error/boundary/fraction mask must be zero",
                    stage=self.stage,
                    node_ids=node_ids,
                )
            if labels.matched_target_span is None:
                _add(
                    issues,
                    "TOKEN_N_MATCHED_TARGET_MISSING",
                    "tokenized N control must preserve its pair-specific target span",
                    stage=self.stage,
                    node_ids=node_ids,
                )
        elif not annotations:
            _add(
                issues,
                "TOKEN_H_POSITIVE_SPAN_MISSING",
                "tokenized H artifact requires a localized positive character span",
                stage=self.stage,
                node_ids=node_ids,
            )

        expected_segments = self._expected_segments(artifact, issues)
        expected = self._expected_masks(labels, annotations, issues, node_ids)
        if expected_segments is not None and labels.segment_ids != expected_segments:
            _add(
                issues,
                "TOKEN_SEGMENT_MISMATCH",
                "token segment IDs disagree with serialized detector spans",
                stage=self.stage,
                node_ids=node_ids,
            )
        if expected_segments is not None:
            expected_evaluation = tuple(
                int(segment in _POSITIVE_SEGMENTS and labels.attention_mask[index] == 1)
                for index, segment in enumerate(expected_segments)
            )
            expected_reasoning = tuple(
                int(segment is SegmentKind.REASONING) for segment in expected_segments
            )
            expected_answer = tuple(
                int(segment is SegmentKind.FINAL_ANSWER)
                for segment in expected_segments
            )
            control_mismatches = tuple(
                name
                for name, expected_values in (
                    ("evaluation_mask", expected_evaluation),
                    ("reasoning_mask", expected_reasoning),
                    ("answer_mask", expected_answer),
                )
                if getattr(labels, name) != expected_values
            )
            if control_mismatches:
                _add(
                    issues,
                    "TOKEN_CONTROL_MASK_MISMATCH",
                    "evaluation/reasoning/answer masks disagree with exact token segments",
                    stage=self.stage,
                    node_ids=node_ids,
                    fields=control_mismatches,
                )
        if expected is not None:
            self._compare_expected(labels, expected, issues, node_ids)

        positive_outside = tuple(
            index
            for index in range(token_count)
            if not labels.evaluation_mask[index]
            and any(values[index] for values in _all_label_arrays(labels))
        )
        if positive_outside:
            _add(
                issues,
                "TOKEN_POSITIVE_OUTSIDE_EVALUATION",
                "source/instruction/special/padding tokens must remain label-free",
                stage=self.stage,
                node_ids=node_ids,
                token_indices=positive_outside,
            )
        return _report(self.validator_id, issues)

    def _expected_segments(
        self,
        artifact: ArtifactValidationInput,
        issues: list[ValidationIssue],
    ) -> tuple[SegmentKind, ...] | None:
        output: list[SegmentKind] = []
        previous_end = 0
        for index, ((start, end), attention) in enumerate(
            zip(
                artifact.token_labels.offset_mapping,
                artifact.token_labels.attention_mask,
                strict=True,
            )
        ):
            if end > len(artifact.serialized.text) or start < 0 or end < start:
                _add(
                    issues,
                    "TOKEN_OFFSET_OUT_OF_RANGE",
                    "token offset is outside serialized detector text",
                    stage=self.stage,
                    node_ids=(artifact.record_id,),
                    token_index=index,
                )
                return None
            if attention not in {0, 1} or (attention == 0 and (start, end) != (0, 0)):
                _add(
                    issues,
                    "TOKEN_ATTENTION_OFFSET_INVALID",
                    "unattended tokens must use an empty offset",
                    stage=self.stage,
                    node_ids=(artifact.record_id,),
                    token_index=index,
                )
                return None
            if (start, end) != (0, 0):
                if start < previous_end:
                    _add(
                        issues,
                        "TOKEN_OFFSETS_NON_MONOTONIC",
                        "non-empty token offsets must be disjoint and monotonic",
                        stage=self.stage,
                        node_ids=(artifact.record_id,),
                        token_index=index,
                    )
                    return None
                previous_end = end
            if attention == 0:
                output.append(SegmentKind.PADDING)
                continue
            if start == end:
                output.append(SegmentKind.SPECIAL)
                continue
            overlapping = tuple(
                segment.segment_kind
                for segment in artifact.serialized.segments
                if max(0, min(end, segment.end) - max(start, segment.start)) > 0
            )
            if len(overlapping) > 1:
                _add(
                    issues,
                    "TOKEN_CROSSES_CONTENT_SEGMENTS",
                    "one token cannot overlap multiple detector content segments",
                    stage=self.stage,
                    node_ids=(artifact.record_id,),
                    token_index=index,
                )
                return None
            output.append(overlapping[0] if overlapping else SegmentKind.SPECIAL)

        # Fast tokenizers may omit whitespace from offsets, but every
        # non-whitespace code point in the exact serialized input must be
        # covered.  This rejects silent truncation as well as removed middle
        # tokens without assuming a particular BOS/EOS convention.
        uncovered: list[tuple[int, int]] = []
        cursor = 0
        for start, end in (
            offset
            for offset in artifact.token_labels.offset_mapping
            if offset != (0, 0)
        ):
            if start > cursor and artifact.serialized.text[cursor:start].strip():
                uncovered.append((cursor, start))
            cursor = max(cursor, end)
        if (
            cursor < len(artifact.serialized.text)
            and artifact.serialized.text[cursor:].strip()
        ):
            uncovered.append((cursor, len(artifact.serialized.text)))
        if uncovered:
            _add(
                issues,
                "TOKEN_TEXT_COVERAGE_INCOMPLETE",
                "token offsets do not cover the complete serialized detector text",
                stage=self.stage,
                node_ids=(artifact.record_id,),
                uncovered_ranges=tuple(uncovered),
            )
        return tuple(output)

    def _expected_masks(
        self,
        labels: TokenLabelSet,
        annotations: tuple[CharAnnotation, ...],
        issues: list[ValidationIssue],
        node_ids: tuple[str, ...],
    ) -> Mapping[str, Any] | None:
        count = len(labels.input_ids)
        semantic_sets = [set[HallucinationType]() for _ in range(count)]
        edit_sets = [set[EditErrorSubtype]() for _ in range(count)]
        role_sets = [set[CausalRole]() for _ in range(count)]
        intersections: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        boundary = [0] * count
        covered: set[str] = set()
        for annotation in annotations:
            for index, (start, end) in enumerate(labels.offset_mapping):
                overlap = _overlap(start, end, annotation.literal_span)
                if overlap <= 0 or not labels.evaluation_mask[index]:
                    continue
                covered.add(annotation.span_id)
                semantic_sets[index].update(annotation.semantic_types)
                edit_sets[index].update(annotation.edit_subtypes)
                if annotation.causal_role is not None:
                    role_sets[index].add(annotation.causal_role)
                intersections[index].append(
                    (
                        max(start, annotation.literal_span.start),
                        min(end, annotation.literal_span.end),
                    )
                )
                if (
                    start < annotation.literal_span.start < end
                    or start < annotation.literal_span.end < end
                ):
                    boundary[index] = 1
        missing = tuple(
            annotation.span_id
            for annotation in annotations
            if annotation.span_id not in covered
        )
        if missing:
            _add(
                issues,
                "TOKEN_POSITIVE_SPAN_UNCOVERED",
                "every positive character span must overlap an evaluated token",
                stage=self.stage,
                node_ids=node_ids,
                span_ids=missing,
            )
        collisions = tuple(
            index for index, roles in enumerate(role_sets) if len(roles) > 1
        )
        if collisions:
            _add(
                issues,
                "TOKEN_CAUSAL_ROLE_COLLISION",
                "one token cannot carry multiple causal roles",
                stage=self.stage,
                node_ids=node_ids,
                token_indices=collisions,
            )
            return None
        semantic_masks = {
            label: tuple(int(label in values) for values in semantic_sets)
            for label in HallucinationType
        }
        edit_masks = {
            label: tuple(int(label in values) for values in edit_sets)
            for label in EditErrorSubtype
        }
        role_masks = {
            label: tuple(int(label in values) for values in role_sets)
            for label in CausalRole
        }
        return {
            "semantic_type_masks": semantic_masks,
            "edit_subtype_masks": edit_masks,
            "causal_role_masks": role_masks,
            "hallucination_core_mask": tuple(
                int(
                    HallucinationType.CONTRADICTION in values
                    or HallucinationType.UNSUPPORTED in values
                )
                for values in semantic_sets
            ),
            "error_any_mask": tuple(
                int(
                    any(label is not HallucinationType.UNVERIFIABLE for label in values)
                )
                for values in semantic_sets
            ),
            "local_falsehood_mask": tuple(
                int(
                    bool(
                        values
                        & {
                            CausalRole.ROOT,
                            CausalRole.PROPAGATED_FALSE,
                            CausalRole.TERMINAL,
                        }
                    )
                )
                for values in role_sets
            ),
            "off_task_branch_mask": tuple(
                int(CausalRole.PROPAGATED_CONDITIONAL in values) for values in role_sets
            ),
            "boundary_ambiguous_mask": tuple(boundary),
            "error_char_fraction": tuple(
                0.0
                if end == start
                else _union_length(intersections[index]) / (end - start)
                for index, (start, end) in enumerate(labels.offset_mapping)
            ),
        }

    def _compare_expected(
        self,
        labels: TokenLabelSet,
        expected: Mapping[str, Any],
        issues: list[ValidationIssue],
        node_ids: tuple[str, ...],
    ) -> None:
        mismatches: list[str] = []
        for name, expected_value in expected.items():
            actual = getattr(labels, name)
            if name.endswith("_masks"):
                if dict(actual) != expected_value:
                    mismatches.append(name)
            elif actual != expected_value:
                mismatches.append(name)
        if mismatches:
            _add(
                issues,
                "TOKEN_PROJECTION_MISMATCH",
                "token masks are not the exact any-overlap projection of char labels",
                stage=self.stage,
                node_ids=node_ids,
                fields=tuple(sorted(mismatches)),
            )


@dataclass(frozen=True, slots=True)
class BundleIntegrityValidator:
    """Validate exact four-pair coverage, reciprocal matching, and one split."""

    validator_id: ClassVar[str] = BUNDLE_INTEGRITY_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.BUNDLE_INTEGRITY

    def validate(self, value: BundleValidationInput) -> ValidationReport:
        if type(value) is not BundleValidationInput:
            raise TypeError("bundle validator requires BundleValidationInput")
        bundle = value.bundle
        node_ids = (bundle.origin_id,)
        issues: list[ValidationIssue] = []
        artifacts = tuple(value.artifacts)
        draft_records = tuple(bundle.records)
        if len(draft_records) != 8 or len(artifacts) != 8:
            _add(
                issues,
                "BUNDLE_RECORD_COUNT",
                "each origin requires exactly eight completed artifacts",
                stage=self.stage,
                node_ids=node_ids,
                draft_count=len(draft_records),
                artifact_count=len(artifacts),
            )

        artifact_counts = Counter(item.record_id for item in artifacts)
        duplicate_ids = tuple(
            sorted(
                record_id for record_id, count in artifact_counts.items() if count > 1
            )
        )
        if duplicate_ids:
            _add(
                issues,
                "BUNDLE_RECORD_ID_DUPLICATE",
                "completed bundle artifact IDs must be unique",
                stage=self.stage,
                node_ids=node_ids,
                record_ids=duplicate_ids,
            )
        artifacts_by_id = {item.record_id: item for item in artifacts}
        expected_ids = {record.record_id for record in draft_records}
        if set(artifacts_by_id) != expected_ids:
            _add(
                issues,
                "BUNDLE_RECORD_SET_MISMATCH",
                "completed artifacts do not exactly cover the T024 draft records",
                stage=self.stage,
                node_ids=node_ids,
                missing=tuple(sorted(expected_ids - set(artifacts_by_id))),
                unknown=tuple(sorted(set(artifacts_by_id) - expected_ids)),
            )

        observed_matrix = Counter(
            (record.policy, record.variant_label) for record in draft_records
        )
        expected_matrix = Counter(
            (policy, label)
            for policy in _POLICIES
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        )
        if observed_matrix != expected_matrix:
            _add(
                issues,
                "BUNDLE_POLICY_MATRIX_INCOMPLETE",
                "origin must contain one H/N pair for every frozen phenotype",
                stage=self.stage,
                node_ids=node_ids,
            )

        try:
            manifest_row = value.split_manifest.row_for_origin(bundle.origin_id)
        except (RuntimeError, TypeError, ValueError) as error:
            _add(
                issues,
                "BUNDLE_SPLIT_MANIFEST_LOOKUP_FAILED",
                "bundle origin cannot be resolved in the verified split manifest",
                stage=self.stage,
                node_ids=node_ids,
                severity=Severity.FATAL,
                exception_type=type(error).__name__,
            )
            manifest_row = None
        if manifest_row is not None:
            split_counts = Counter(item.split for item in artifacts)
            if split_counts != Counter({manifest_row.split: len(artifacts)}):
                _add(
                    issues,
                    "BUNDLE_SPLIT_MISMATCH",
                    "all eight variants must inherit the verified origin split",
                    stage=self.stage,
                    node_ids=node_ids,
                    expected_split=manifest_row.split.value,
                    observed_splits=tuple(
                        sorted(
                            (split.value, count)
                            for split, count in split_counts.items()
                        )
                    ),
                )
            if any(
                item.leakage_group_id != manifest_row.leakage_group_id
                for item in artifacts
            ):
                _add(
                    issues,
                    "BUNDLE_LEAKAGE_GROUP_MISMATCH",
                    "all variants must preserve the verified origin leakage group",
                    stage=self.stage,
                    node_ids=node_ids,
                    expected_leakage_group_id=manifest_row.leakage_group_id,
                )

        records_by_id = {record.record_id: record for record in draft_records}
        if len(records_by_id) != len(draft_records):
            _add(
                issues,
                "BUNDLE_DRAFT_RECORD_ID_DUPLICATE",
                "T024 draft record IDs must remain unique",
                stage=self.stage,
                node_ids=node_ids,
            )
        if len({record.control_identity for record in draft_records}) != 4:
            _add(
                issues,
                "BUNDLE_CONTROL_IDENTITY_REUSE",
                "each H/N pair requires one distinct faithful-control identity",
                stage=self.stage,
                node_ids=node_ids,
            )
        if len({record.render_identity for record in draft_records}) != 8:
            _add(
                issues,
                "BUNDLE_RENDER_IDENTITY_REUSE",
                "each record requires a distinct pair-specific render identity",
                stage=self.stage,
                node_ids=node_ids,
            )
        pair_groups: defaultdict[str, list[MatchedDraftRecord]] = defaultdict(list)
        for record in draft_records:
            pair_groups[record.pair_id].append(record)
            artifact = artifacts_by_id.get(record.record_id)
            if artifact is not None and artifact.draft != record:
                _add(
                    issues,
                    "BUNDLE_DRAFT_BINDING_MISMATCH",
                    "completed artifact is not bound to its exact T024 draft",
                    stage=self.stage,
                    node_ids=(record.record_id,),
                )
            if (
                record.origin_id != bundle.origin_id
                or record.bundle_id != bundle.bundle_id
            ):
                _add(
                    issues,
                    "BUNDLE_ORIGIN_ID_MISMATCH",
                    "draft record differs from its origin/bundle identity",
                    stage=self.stage,
                    node_ids=(record.record_id,),
                )
            matched = records_by_id.get(record.matched_record_id)
            if (
                matched is None
                or matched.matched_record_id != record.record_id
                or matched.pair_id != record.pair_id
                or matched.policy is not record.policy
                or matched.variant_label is record.variant_label
            ):
                _add(
                    issues,
                    "BUNDLE_PAIR_LINK_INVALID",
                    "matched record links must resolve reciprocally",
                    stage=self.stage,
                    node_ids=(record.record_id,),
                )

        for pair_id, pair in sorted(pair_groups.items()):
            if len(pair) != 2:
                _add(
                    issues,
                    "BUNDLE_PAIR_CARDINALITY",
                    "every pair_id must identify exactly one H and one N",
                    stage=self.stage,
                    node_ids=node_ids,
                    pair_id=pair_id,
                    record_count=len(pair),
                )
                continue
            h_records = [
                record
                for record in pair
                if record.variant_label is VariantLabel.HALLUCINATED
            ]
            n_records = [
                record
                for record in pair
                if record.variant_label is VariantLabel.FAITHFUL
            ]
            if len(h_records) != 1 or len(n_records) != 1:
                _add(
                    issues,
                    "BUNDLE_PAIR_VARIANTS",
                    "pair must contain one hallucinated and one faithful record",
                    stage=self.stage,
                    node_ids=node_ids,
                    pair_id=pair_id,
                )
                continue
            h_record, n_record = h_records[0], n_records[0]
            shared_axes = (
                "policy",
                "input_view_id",
                "target_node_id",
                "target_step_index",
                "operator_id",
                "operator_family",
                "quota_bucket",
                "candidate_source",
                "renderer_backend",
                "renderer_style_id",
                "rewrite_budget",
                "candidate_difficulty_bucket",
                "fallback_decision",
                "control_identity",
            )
            drift = tuple(
                name
                for name in shared_axes
                if getattr(h_record, name) != getattr(n_record, name)
            )
            if drift:
                _add(
                    issues,
                    "BUNDLE_PAIR_AXIS_MISMATCH",
                    "matched H/N records differ on frozen construction axes",
                    stage=self.stage,
                    node_ids=(h_record.record_id, n_record.record_id),
                    fields=drift,
                )
            h_artifact = artifacts_by_id.get(h_record.record_id)
            n_artifact = artifacts_by_id.get(n_record.record_id)
            if h_artifact is None or n_artifact is None:
                continue
            length_delta = abs(
                len(h_artifact.serialized.text) - len(n_artifact.serialized.text)
            )
            if length_delta > h_record.rewrite_budget.max_added_characters:
                _add(
                    issues,
                    "BUNDLE_PAIR_LENGTH_BUDGET",
                    "matched surface length difference exceeds frozen rewrite budget",
                    stage=self.stage,
                    node_ids=(h_record.record_id, n_record.record_id),
                    length_delta=length_delta,
                    maximum=h_record.rewrite_budget.max_added_characters,
                )
            target_span = (
                None
                if n_artifact.token_labels is None
                else n_artifact.token_labels.matched_target_span
            )
            if target_span is None:
                _add(
                    issues,
                    "BUNDLE_PAIR_TARGET_SPAN_MISSING",
                    "faithful control must preserve its pair-specific matched target span",
                    stage=self.stage,
                    node_ids=(n_record.record_id,),
                )
            elif not self._target_span_matches(
                n_artifact,
                target_span,
                h_record.policy,
                n_record.target_node_id,
                n_record.target_step_index,
            ):
                _add(
                    issues,
                    "BUNDLE_PAIR_TARGET_SPAN_MISMATCH",
                    "faithful matched target is not an exact occurrence of its locked target",
                    stage=self.stage,
                    node_ids=(n_record.record_id,),
                )
        return _report(self.validator_id, issues)

    @staticmethod
    def _target_span_matches(
        artifact: ArtifactValidationInput,
        target_span: Any,
        policy: PropagationPolicy,
        target_node_id: str,
        target_step_index: int | None,
    ) -> bool:
        expected_component = (
            SegmentKind.FINAL_ANSWER
            if policy is PropagationPolicy.TERMINAL
            else SegmentKind.REASONING
        )
        try:
            coordinate_map = DetectorCoordinateMap.from_rendered(
                artifact.rendered,
                artifact.serialized,
            )
            exact_spans = {
                coordinate_map.rebase_span(
                    mention.literal_span,
                    mention.component,
                )
                for mention in artifact.rendered.mentions
                if mention.target_kind is MutationTargetKind.NODE
                and mention.state_or_edge_id == target_node_id
                and mention.component is expected_component
                and mention.step_index == target_step_index
            }
        except (RuntimeError, TypeError, ValueError):
            return False
        return target_span in exact_spans


@dataclass(frozen=True, slots=True)
class ValidatorChain:
    """Compose all remaining gates without dropping validator failures."""

    semantic: HallucinationSemanticValidator = HallucinationSemanticValidator()
    propagation: PropagationValidator = PropagationValidator()
    renderer: RendererValidator = RendererValidator()
    token_alignment: TokenAlignmentValidator = TokenAlignmentValidator()
    bundle_integrity: BundleIntegrityValidator = BundleIntegrityValidator()
    validator_id: str = VALIDATOR_CHAIN_ID

    def __post_init__(self) -> None:
        expected = (
            (self.semantic, HallucinationSemanticValidator),
            (self.propagation, PropagationValidator),
            (self.renderer, RendererValidator),
            (self.token_alignment, TokenAlignmentValidator),
            (self.bundle_integrity, BundleIntegrityValidator),
        )
        if any(type(value) is not expected_type for value, expected_type in expected):
            raise TypeError("ValidatorChain requires the five frozen validator types")
        if type(self.validator_id) is not str or not self.validator_id:
            raise ValueError("validator_id must be non-empty text")

    def validate_artifact(
        self,
        artifact: ArtifactValidationInput,
    ) -> ValidationReport:
        if type(artifact) is not ArtifactValidationInput:
            raise TypeError("validate_artifact requires ArtifactValidationInput")
        reports: list[ValidationReport] = []
        for validator in (
            self.semantic,
            self.propagation,
            self.renderer,
            self.token_alignment,
        ):
            try:
                reports.append(validator.validate(artifact))
            except Exception as error:  # noqa: BLE001 - validation must fail closed
                reports.append(
                    _report(
                        validator.validator_id,
                        (
                            ValidationIssue(
                                code="VALIDATOR_INTERNAL_FAILURE",
                                severity=Severity.FATAL,
                                stage=validator.stage,
                                node_ids=(artifact.record_id,),
                                message="validator could not safely adjudicate artifact",
                                evidence={
                                    "validator_id": validator.validator_id,
                                    "exception_type": type(error).__name__,
                                    "error_code": getattr(error, "code", None),
                                },
                            ),
                        ),
                    )
                )
        return ValidationReport.combine(self.validator_id, reports)

    def validate_artifact_strict(
        self,
        artifact: ArtifactValidationInput,
    ) -> ValidationReport:
        report = self.validate_artifact(artifact)
        if not report.all_pass:
            raise ArtifactValidationError(report)
        return report

    def validate_bundle(self, value: BundleValidationInput) -> ValidationReport:
        if type(value) is not BundleValidationInput:
            raise TypeError("validate_bundle requires BundleValidationInput")
        reports = [self.validate_artifact(item) for item in value.artifacts]
        try:
            reports.append(self.bundle_integrity.validate(value))
        except Exception as error:  # noqa: BLE001 - validation must fail closed
            reports.append(
                _report(
                    self.bundle_integrity.validator_id,
                    (
                        ValidationIssue(
                            code="VALIDATOR_INTERNAL_FAILURE",
                            severity=Severity.FATAL,
                            stage=self.bundle_integrity.stage,
                            node_ids=(value.bundle.origin_id,),
                            message="bundle validator could not safely adjudicate artifact",
                            evidence={
                                "validator_id": self.bundle_integrity.validator_id,
                                "exception_type": type(error).__name__,
                                "error_code": getattr(error, "code", None),
                            },
                        ),
                    ),
                )
            )
        return ValidationReport.combine(self.validator_id, reports)

    def validate_bundle_strict(
        self,
        value: BundleValidationInput,
    ) -> ValidationReport:
        report = self.validate_bundle(value)
        if not report.all_pass:
            raise ArtifactValidationError(report)
        return report


def validate_artifact(artifact: ArtifactValidationInput) -> ValidationReport:
    return ValidatorChain().validate_artifact(artifact)


def validate_artifact_strict(artifact: ArtifactValidationInput) -> ValidationReport:
    return ValidatorChain().validate_artifact_strict(artifact)


def validate_bundle(value: BundleValidationInput) -> ValidationReport:
    return ValidatorChain().validate_bundle(value)


def validate_bundle_strict(value: BundleValidationInput) -> ValidationReport:
    return ValidatorChain().validate_bundle_strict(value)


__all__ = [
    "ARTIFACT_VALIDATOR_IDS",
    "BUNDLE_INTEGRITY_VALIDATOR_ID",
    "PROPAGATION_VALIDATOR_ID",
    "RENDERER_VALIDATOR_ID",
    "SEMANTIC_VALIDATOR_ID",
    "TOKEN_ALIGNMENT_VALIDATOR_ID",
    "VALIDATOR_CHAIN_ID",
    "ArtifactValidationInput",
    "BundleIntegrityValidator",
    "BundleValidationInput",
    "HallucinationSemanticValidator",
    "PropagationValidator",
    "RendererValidator",
    "TokenAlignmentValidator",
    "ValidatorChain",
    "validate_artifact",
    "validate_artifact_strict",
    "validate_bundle",
    "validate_bundle_strict",
]
