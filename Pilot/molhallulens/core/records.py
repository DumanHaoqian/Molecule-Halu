"""Immutable source, detector-view, and built-artifact records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from typing import Any

from .enums import (
    CausalRole,
    EditingSubtask,
    MutationTargetKind,
    OperationSubtype,
    PropagationPolicy,
    SegmentKind,
    TaskFamily,
    VariantLabel,
    Visibility,
)
from .errors import ValidationReport
from .labels import CharAnnotation, TokenLabelSet
from .state_dag import FrozenMap, GraphDelta, StateDAG, freeze_string_mapping


@dataclass(frozen=True, slots=True)
class TaskRecord:
    origin_id: str
    anonymous_sample_id: str
    family: TaskFamily
    source_subtask: str
    normalized_subtask: EditingSubtask
    operation_subtype: OperationSubtype
    indexed_smiles: str
    instruction: str
    gt_smiles: str
    reference_reasoning_chain: str
    reference_final_answer: str
    parsed_reference_state: Mapping[str, Any]
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)
    process_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.family) is not TaskFamily:
            raise TypeError("TaskRecord family must be a TaskFamily")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("TaskRecord normalized_subtask must be an EditingSubtask")
        if type(self.operation_subtype) is not OperationSubtype:
            raise TypeError("TaskRecord operation_subtype must be an OperationSubtype")
        for value, name in (
            (self.parsed_reference_state, "parsed_reference_state"),
            (self.raw_metadata, "raw_metadata"),
            (self.process_metadata, "process_metadata"),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"TaskRecord {name} must be a mapping")
        object.__setattr__(
            self,
            "parsed_reference_state",
            freeze_string_mapping(
                self.parsed_reference_state,
                name="TaskRecord parsed_reference_state",
            ),
        )
        object.__setattr__(
            self,
            "raw_metadata",
            freeze_string_mapping(self.raw_metadata, name="TaskRecord raw_metadata"),
        )
        object.__setattr__(
            self,
            "process_metadata",
            freeze_string_mapping(self.process_metadata, name="TaskRecord process_metadata"),
        )
        required_strings = {
            "origin_id": self.origin_id,
            "anonymous_sample_id": self.anonymous_sample_id,
            "source_subtask": self.source_subtask,
            "indexed_smiles": self.indexed_smiles,
            "instruction": self.instruction,
            "gt_smiles": self.gt_smiles,
            "reference_reasoning_chain": self.reference_reasoning_chain,
            "reference_final_answer": self.reference_final_answer,
        }
        if any(type(value) is not str for value in required_strings.values()):
            raise TypeError("TaskRecord required text fields must be strings")
        empty = sorted(name for name, value in required_strings.items() if not value.strip())
        if empty:
            raise ValueError(f"TaskRecord required fields cannot be empty: {empty}")
        if self.family is not TaskFamily.MOLECULE_EDITING:
            raise ValueError("this Pilot TaskRecord currently supports only mol_edit")
        expected_source_prefix = {
            EditingSubtask.ADD: "add",
            EditingSubtask.DELETE: "delete",
            EditingSubtask.SUBSTITUTE: "substitute",
        }[self.normalized_subtask]
        if not self.source_subtask.startswith(expected_source_prefix):
            raise ValueError("source_subtask is inconsistent with normalized_subtask")


@dataclass(frozen=True, slots=True)
class DetectorInput:
    """The complete detector-visible view; deliberately has no GT/oracle field."""

    indexed_smiles: str
    instruction: str
    reasoning_chain: str
    final_answer: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.indexed_smiles, "indexed_smiles"),
            (self.instruction, "instruction"),
            (self.reasoning_chain, "reasoning_chain"),
            (self.final_answer, "final_answer"),
        ):
            if type(value) is not str:
                raise TypeError(f"DetectorInput {name} must be a string")
            if not value.strip():
                raise ValueError(f"DetectorInput {name} cannot be empty")

    @property
    def field_order(self) -> tuple[str, ...]:
        return ("indexed_smiles", "instruction", "reasoning_chain", "final_answer")


@dataclass(frozen=True, slots=True)
class BuildProvenance:
    provider: str
    transport: str | None
    requested_model_id: str | None
    response_model: str | None
    model_catalog_entry_sha256: str | None
    request_ids: tuple[str, ...] = ()
    response_ids: tuple[str, ...] = ()
    prompt_hashes: tuple[str, ...] = ()
    tool_schema_sha256: str | None = None
    cache_keys: tuple[str, ...] = ()
    attempt_count: int = 0
    token_usage: Mapping[str, int] = field(default_factory=dict)
    cost_points: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.provider) is not str:
            raise TypeError("BuildProvenance provider must be a string")
        for value, name in (
            (self.transport, "transport"),
            (self.requested_model_id, "requested_model_id"),
            (self.response_model, "response_model"),
            (self.model_catalog_entry_sha256, "model_catalog_entry_sha256"),
            (self.tool_schema_sha256, "tool_schema_sha256"),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"BuildProvenance {name} must be a string or None")
        for field_name in ("request_ids", "response_ids", "prompt_hashes", "cache_keys"):
            if not isinstance(getattr(self, field_name), (list, tuple)):
                raise TypeError(f"BuildProvenance {field_name} must be a list or tuple")
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
            if any(
                type(value) is not str or not value for value in getattr(self, field_name)
            ):
                raise TypeError(
                    f"BuildProvenance {field_name} must contain non-empty strings"
                )
        if not isinstance(self.token_usage, Mapping):
            raise TypeError("BuildProvenance token_usage must be a mapping")
        if any(type(key) is not str or not key for key in self.token_usage):
            raise TypeError("BuildProvenance token_usage keys must be non-empty strings")
        if any(
            type(value) is not int
            for value in self.token_usage.values()
        ):
            raise TypeError("BuildProvenance token_usage values must be integers")
        if not isinstance(self.extra, Mapping):
            raise TypeError("BuildProvenance extra must be a mapping")
        if any(type(key) is not str or not key for key in self.extra):
            raise TypeError("BuildProvenance extra keys must be non-empty strings")
        object.__setattr__(self, "token_usage", FrozenMap(self.token_usage))
        object.__setattr__(
            self,
            "extra",
            freeze_string_mapping(self.extra, name="BuildProvenance extra"),
        )
        if not self.provider:
            raise ValueError("BuildProvenance provider cannot be empty")
        if type(self.attempt_count) is not int:
            raise TypeError("BuildProvenance attempt_count must be an integer")
        if self.attempt_count < 0:
            raise ValueError("BuildProvenance attempt_count cannot be negative")
        if any(value < 0 for value in self.token_usage.values()):
            raise ValueError("BuildProvenance token usage cannot be negative")
        if self.cost_points is not None:
            if type(self.cost_points) not in {int, float}:
                raise TypeError("BuildProvenance cost_points must be numeric or None")
            if not isfinite(self.cost_points) or self.cost_points < 0:
                raise ValueError("BuildProvenance cost_points must be finite and non-negative")
        forbidden_keys = {"poe_api_key", "authorization", "api_key", "secret", "password"}

        def contains_forbidden_key(value: Any) -> bool:
            if isinstance(value, Mapping):
                return any(
                    key.casefold() in forbidden_keys or contains_forbidden_key(item)
                    for key, item in value.items()
                )
            if isinstance(value, (tuple, frozenset)):
                return any(contains_forbidden_key(item) for item in value)
            if is_dataclass(value) and not isinstance(value, type):
                return any(
                    contains_forbidden_key(getattr(value, item.name)) for item in fields(value)
                )
            return False

        if contains_forbidden_key(self.extra):
            raise ValueError("BuildProvenance extra cannot contain secrets")


@dataclass(frozen=True, slots=True)
class TraceLabels:
    hallucination_present: bool
    reasoning_valid: bool
    answer_correct: bool
    chemically_valid: bool
    constraint_satisfied: bool
    format_valid: bool
    answer_complete: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"TraceLabels {field_name} must be bool")


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    record_id: str
    origin_id: str
    leakage_group_id: str
    bundle_id: str
    pair_id: str
    matched_record_id: str
    variant_label: VariantLabel
    policy: PropagationPolicy
    detector_input: DetectorInput
    serialized_text: str
    serialized_text_sha256: str
    reference_graph: StateDAG
    candidate_graph: StateDAG
    graph_delta: GraphDelta
    char_annotations: tuple[CharAnnotation, ...]
    token_labels: TokenLabelSet | None
    trace_labels: TraceLabels
    validation_report: ValidationReport
    provenance: BuildProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_annotations", tuple(self.char_annotations))
        if type(self.variant_label) is not VariantLabel:
            raise TypeError("PerturbationResult variant_label must be a VariantLabel")
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("PerturbationResult policy must be a PropagationPolicy")
        for value, expected_type, name in (
            (self.detector_input, DetectorInput, "detector_input"),
            (self.reference_graph, StateDAG, "reference_graph"),
            (self.candidate_graph, StateDAG, "candidate_graph"),
            (self.graph_delta, GraphDelta, "graph_delta"),
            (self.trace_labels, TraceLabels, "trace_labels"),
            (self.validation_report, ValidationReport, "validation_report"),
            (self.provenance, BuildProvenance, "provenance"),
        ):
            if type(value) is not expected_type:
                raise TypeError(
                    f"PerturbationResult {name} must be a {expected_type.__name__}"
                )
        if self.token_labels is not None and type(self.token_labels) is not TokenLabelSet:
            raise TypeError("PerturbationResult token_labels must be a TokenLabelSet or None")
        if any(type(item) is not CharAnnotation for item in self.char_annotations):
            raise TypeError(
                "PerturbationResult char_annotations must contain CharAnnotation values"
            )
        if self.reference_graph.schema != self.candidate_graph.schema:
            raise ValueError("reference_graph and candidate_graph must use the same StateSchema")
        schema = self.reference_graph.schema
        node_specs = schema.nodes_by_id
        edge_specs = schema.edges_by_id
        delta_targets = frozenset(
            (event.target_kind, event.node_or_edge_id) for event in self.graph_delta.events
        )
        events_by_target = {
            (event.target_kind, event.node_or_edge_id): event
            for event in self.graph_delta.events
        }
        semantic_differences = self.reference_graph.semantic_differences(self.candidate_graph)
        if semantic_differences != delta_targets:
            raise ValueError(
                "GraphDelta targets must exactly match reference/candidate semantic differences"
            )
        for event in self.graph_delta.events:
            if event.target_kind is MutationTargetKind.NODE:
                spec = node_specs.get(event.node_or_edge_id)
                if spec is None:
                    raise ValueError("GraphDelta references an unknown schema node")
                if (
                    not spec.mutable
                    or spec.visibility is not Visibility.CANDIDATE_OUTPUT
                    or spec.renderer_slot is None
                ):
                    raise ValueError(
                        "GraphDelta node targets must be mutable, rendered candidate-output nodes"
                    )
                reference_value = self.reference_graph.values[event.node_or_edge_id]
                candidate_value = self.candidate_graph.values[event.node_or_edge_id]
            else:
                spec = edge_specs.get(event.node_or_edge_id)
                if spec is None:
                    raise ValueError("GraphDelta references an unknown schema edge")
                if not spec.mutable or spec.renderer_slot is None:
                    raise ValueError("GraphDelta edge targets must be mutable rendered edges")
                if (
                    event.node_or_edge_id not in self.reference_graph.edge_values
                    or event.node_or_edge_id not in self.candidate_graph.edge_values
                ):
                    raise ValueError("mutated edges require reference and candidate ClaimValue values")
                reference_value = self.reference_graph.edge_values[event.node_or_edge_id]
                candidate_value = self.candidate_graph.edge_values[event.node_or_edge_id]
            if not event.before.semantically_equals(reference_value) or not event.after.semantically_equals(
                candidate_value
            ):
                raise ValueError("MutationEvent before/after must match reference/candidate graph values")

        span_ids = tuple(annotation.span_id for annotation in self.char_annotations)
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("PerturbationResult char annotation span IDs must be unique")
        known_span_ids = set(span_ids)
        annotations_by_id = {
            annotation.span_id: annotation for annotation in self.char_annotations
        }
        annotation_targets: list[tuple[MutationTargetKind, str]] = []
        for annotation in self.char_annotations:
            if annotation.root_span_id is not None and annotation.root_span_id not in known_span_ids:
                raise ValueError("CharAnnotation root_span_id must resolve within the record")
            if annotation.causal_role in {
                CausalRole.PROPAGATED_FALSE,
                CausalRole.PROPAGATED_CONDITIONAL,
            }:
                root_annotation = annotations_by_id[annotation.root_span_id]
                if root_annotation.causal_role not in {CausalRole.ROOT, CausalRole.TERMINAL}:
                    raise ValueError(
                        "propagated CharAnnotation root_span_id must identify an independent root"
                    )
                graph_roots = self.graph_delta.root_events
                if (
                    not graph_roots
                    or root_annotation.state_or_edge_id != graph_roots[0].node_or_edge_id
                ):
                    raise ValueError(
                        "propagated CharAnnotation root span must match the GraphDelta root"
                    )
            if annotation.state_or_edge_id in node_specs:
                node = node_specs[annotation.state_or_edge_id]
                if (
                    node.visibility is not Visibility.CANDIDATE_OUTPUT
                    or node.renderer_slot is None
                ):
                    raise ValueError(
                        "CharAnnotation nodes must be rendered candidate-output nodes"
                    )
                annotation_targets.append(
                    (MutationTargetKind.NODE, annotation.state_or_edge_id)
                )
            elif annotation.state_or_edge_id in edge_specs:
                edge = edge_specs[annotation.state_or_edge_id]
                if edge.renderer_slot is None:
                    raise ValueError("CharAnnotation edges must have renderer slots")
                annotation_targets.append(
                    (MutationTargetKind.EDGE, annotation.state_or_edge_id)
                )
            else:
                raise ValueError("CharAnnotation references an unknown state or edge ID")
        if any(target not in delta_targets for target in annotation_targets):
            raise ValueError("positive CharAnnotation targets must be present in GraphDelta")
        adjudicated_annotation_targets = {
            target
            for annotation, target in zip(self.char_annotations, annotation_targets, strict=True)
            if annotation.is_adjudicated
        }
        if adjudicated_annotation_targets != delta_targets:
            raise ValueError(
                "every GraphDelta target must have an adjudicated positive char annotation"
            )
        for target in delta_targets:
            event = events_by_target[target]
            target_annotations = tuple(
                annotation
                for annotation, annotation_target in zip(
                    self.char_annotations,
                    annotation_targets,
                    strict=True,
                )
                if annotation.is_adjudicated and annotation_target == target
            )
            if any(
                annotation.causal_role is not event.causal_role
                for annotation in target_annotations
            ):
                raise ValueError(
                    "CharAnnotation causal roles must match their MutationEvent"
                )
            annotated_semantic_types = frozenset(
                label
                for annotation in target_annotations
                for label in annotation.semantic_types
            )
            annotated_edit_subtypes = frozenset(
                label
                for annotation in target_annotations
                for label in annotation.edit_subtypes
            )
            if annotated_semantic_types != event.hallucination_types:
                raise ValueError(
                    "CharAnnotation semantic types must exactly cover their MutationEvent"
                )
            if annotated_edit_subtypes != event.edit_subtypes:
                raise ValueError(
                    "CharAnnotation edit subtypes must exactly cover their MutationEvent"
                )
        for value, name in (
            (self.record_id, "record_id"),
            (self.origin_id, "origin_id"),
            (self.leakage_group_id, "leakage_group_id"),
            (self.bundle_id, "bundle_id"),
            (self.pair_id, "pair_id"),
            (self.matched_record_id, "matched_record_id"),
            (self.serialized_text, "serialized_text"),
            (self.serialized_text_sha256, "serialized_text_sha256"),
        ):
            if type(value) is not str:
                raise TypeError(f"PerturbationResult {name} must be a string")
            if not value:
                raise ValueError(f"PerturbationResult {name} cannot be empty")
        if any(
            annotation.claim_span.end > len(self.serialized_text)
            for annotation in self.char_annotations
        ):
            raise ValueError("CharAnnotation spans must fall within serialized_text")
        if self.token_labels is not None:
            if self.token_labels.serialized_text_sha256 != self.serialized_text_sha256:
                raise ValueError("token label and record serialized_text_sha256 values must match")
            if any(end > len(self.serialized_text) for _, end in self.token_labels.offset_mapping):
                raise ValueError("token offsets must fall within serialized_text")
            if (
                self.token_labels.matched_target_span is not None
                and self.token_labels.matched_target_span.end > len(self.serialized_text)
            ):
                raise ValueError("matched_target_span must fall within serialized_text")
        if self.matched_record_id == self.record_id:
            raise ValueError("PerturbationResult cannot be matched to itself")
        if not self.validation_report.all_pass:
            raise ValueError("PerturbationResult must pass deterministic validation")
        if self.variant_label is VariantLabel.HALLUCINATED:
            if not self.graph_delta.events or not self.char_annotations:
                raise ValueError("H records require graph mutations and positive char annotations")
            if not self.trace_labels.hallucination_present:
                raise ValueError("H record trace_labels must indicate hallucination")
            if not any(annotation.is_adjudicated for annotation in self.char_annotations):
                raise ValueError("H records require an adjudicated positive char annotation")
            if self.token_labels is not None and not any(self.token_labels.error_any_mask):
                raise ValueError("tokenized H records must carry adjudicated positive token labels")
            root_event = self.graph_delta.root_events[0]
            root_role = root_event.causal_role
            independent_annotations = tuple(
                annotation
                for annotation in self.char_annotations
                if annotation.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}
            )
            if not independent_annotations:
                raise ValueError("H records require an independent root char annotation")
            if any(
                annotation.state_or_edge_id != root_event.node_or_edge_id
                for annotation in independent_annotations
            ):
                raise ValueError("independent char annotations must identify the GraphDelta root")
            if self.policy is PropagationPolicy.TERMINAL and root_role is not CausalRole.TERMINAL:
                raise ValueError("H_TERMINAL records require a TERMINAL graph root")
            if self.policy is not PropagationPolicy.TERMINAL and root_role is CausalRole.TERMINAL:
                raise ValueError("only H_TERMINAL records may carry a TERMINAL graph root")
            if self.policy is PropagationPolicy.STOP and len(self.graph_delta.events) != 1:
                raise ValueError("H_LOCAL/STOP records may only mutate the independent root")
            if self.policy is PropagationPolicy.TERMINAL:
                if not self.trace_labels.reasoning_valid or self.trace_labels.answer_correct:
                    raise ValueError(
                        "H_TERMINAL requires valid reasoning and an incorrect final answer"
                    )
                if any(
                    annotation.component is not SegmentKind.FINAL_ANSWER
                    or annotation.causal_role is not CausalRole.TERMINAL
                    for annotation in self.char_annotations
                ):
                    raise ValueError(
                        "H_TERMINAL char annotations must be TERMINAL labels in final_answer"
                    )
                if self.token_labels is not None and any(
                    self.token_labels.segment_ids[index] is not SegmentKind.FINAL_ANSWER
                    or not self.token_labels.causal_role_masks[CausalRole.TERMINAL][index]
                    for index in self.token_labels.positive_label_indices
                ):
                    raise ValueError(
                        "H_TERMINAL token labels must be TERMINAL labels in final_answer"
                    )
            else:
                if any(
                    annotation.causal_role is CausalRole.TERMINAL
                    for annotation in self.char_annotations
                ):
                    raise ValueError("non-TERMINAL records cannot carry TERMINAL annotations")
                if self.token_labels is not None and any(
                    self.token_labels.causal_role_masks[CausalRole.TERMINAL]
                ):
                    raise ValueError("non-TERMINAL records cannot carry TERMINAL token labels")
        else:
            if self.graph_delta.events:
                raise ValueError("N records cannot carry graph mutations")
            if self.char_annotations:
                raise ValueError("N records cannot carry positive char annotations")
            if self.trace_labels.hallucination_present:
                raise ValueError("N record trace_labels cannot indicate hallucination")
            faithful_flags = (
                self.trace_labels.reasoning_valid,
                self.trace_labels.answer_correct,
                self.trace_labels.chemically_valid,
                self.trace_labels.constraint_satisfied,
                self.trace_labels.format_valid,
                self.trace_labels.answer_complete,
            )
            if not all(faithful_flags):
                raise ValueError("N record trace_labels must be fully faithful and valid")
            if self.token_labels is not None:
                if self.token_labels.has_positive_labels:
                    raise ValueError("N records cannot carry positive token labels")
                if self.token_labels.matched_target_span is None:
                    raise ValueError("tokenized N records must preserve matched_target_span")


@dataclass(frozen=True, slots=True)
class OriginBundle:
    origin_id: str
    records: tuple[PerturbationResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if type(self.origin_id) is not str:
            raise TypeError("OriginBundle origin_id must be a string")
        if any(type(record) is not PerturbationResult for record in self.records):
            raise TypeError("OriginBundle records must contain PerturbationResult values")
        if not self.origin_id:
            raise ValueError("OriginBundle origin_id cannot be empty")
        if len(self.records) != 8:
            raise ValueError("OriginBundle must contain exactly 8 records")
        if any(record.origin_id != self.origin_id for record in self.records):
            raise ValueError("every OriginBundle record must share origin_id")
        record_ids = tuple(record.record_id for record in self.records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("OriginBundle record IDs must be unique")
        records_by_id = {record.record_id: record for record in self.records}
        if len({record.bundle_id for record in self.records}) != 1:
            raise ValueError("every OriginBundle record must share bundle_id")
        if len({record.leakage_group_id for record in self.records}) != 1:
            raise ValueError("every OriginBundle record must share leakage_group_id")
        pair_counts = Counter(record.pair_id for record in self.records)
        if len(pair_counts) != 4 or set(pair_counts.values()) != {2}:
            raise ValueError("OriginBundle must contain four distinct two-record pair IDs")
        for record in self.records:
            matched = records_by_id.get(record.matched_record_id)
            if matched is None:
                raise ValueError("every matched_record_id must resolve within its OriginBundle")
            if matched.matched_record_id != record.record_id:
                raise ValueError("OriginBundle matched record links must be reciprocal")
            if matched.pair_id != record.pair_id or matched.policy is not record.policy:
                raise ValueError("matched records must share pair_id and propagation policy")
            if matched.variant_label is record.variant_label:
                raise ValueError("matched records must have opposite H/N variant labels")
        counts = Counter((record.policy, record.variant_label) for record in self.records)
        expected = {
            (policy, label): 1
            for policy in (
                PropagationPolicy.STOP,
                PropagationPolicy.PARTIAL,
                PropagationPolicy.FULL_CF,
                PropagationPolicy.TERMINAL,
            )
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        }
        if counts != expected:
            raise ValueError("OriginBundle must contain one H/N record for each propagation policy")
