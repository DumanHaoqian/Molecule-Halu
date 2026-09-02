"""Phase-2 matched-bundle drafts and deterministic quota scheduling.

The objects in this module intentionally stop before ``PerturbationResult``:
natural-language rendering, character annotations, token labels, split IDs,
and final artifact validation are owned by later tasks.  T024 instead freezes
the semantic H/N pairing axes, faithful control identity, and exact operator
quota allocation needed by T025 and the later artifact builders.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from molhallulens.infrastructure.chemistry import isomeric_graph_equivalent
from molhallulens.config.models import OperatorsConfig
from molhallulens.core import (
    CandidateSourceType,
    EditingSubtask,
    GraphDelta,
    MutationTargetKind,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    VariantLabel,
    Visibility,
    state_schema_for,
)
from molhallulens.modules.error_injection.registry import FallbackDecision, PerturbatorRegistry
from molhallulens.modules.text_realization import (
    DeterministicAnswerRenderer,
    DeterministicFormalRenderer,
    RenderedAnswer,
    RenderedFormalTrace,
)

_POLICY_ORDER = (
    PropagationPolicy.STOP,
    PropagationPolicy.PARTIAL,
    PropagationPolicy.FULL_CF,
    PropagationPolicy.TERMINAL,
)
_POLICY_INDEX = {policy: index for index, policy in enumerate(_POLICY_ORDER)}


class BundleDraftError(RuntimeError):
    """Structured fail-closed bundle or prepared-variant contract failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        origin_id: str | None = None,
        policy: PropagationPolicy | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("BundleDraftError code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("BundleDraftError detail must be non-empty text")
        if origin_id is not None and (type(origin_id) is not str or not origin_id):
            raise ValueError("origin_id must be non-empty text or None")
        if policy is not None and type(policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.origin_id = origin_id
        self.policy = policy
        self.evidence = MappingProxyType(dict(evidence or {}))
        location = ""
        if origin_id is not None:
            location += f" origin={origin_id!r}"
        if policy is not None:
            location += f" policy={policy.dataset_name}"
        super().__init__(f"{code}{location}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "origin_id": self.origin_id,
            "policy": None if self.policy is None else self.policy.dataset_name,
            "evidence": _json_value(self.evidence),
        }


def _nonempty(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _json_value(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if hasattr(value, "value") and type(value.value) in {str, int}:
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    raise TypeError(f"unsupported stable JSON value: {type(value).__qualname__}")


def _delta_targets(delta: GraphDelta) -> frozenset[tuple[MutationTargetKind, str]]:
    return frozenset(
        (event.target_kind, event.node_or_edge_id) for event in delta.events
    )


def _validate_delta_binding(
    reference: StateDAG,
    candidate: StateDAG,
    delta: GraphDelta,
    *,
    origin_id: str,
    policy: PropagationPolicy,
) -> None:
    for event in delta.events:
        if event.target_kind is MutationTargetKind.NODE:
            before = reference.values.get(event.node_or_edge_id)
            after = candidate.values.get(event.node_or_edge_id)
        else:
            before = reference.edge_values.get(event.node_or_edge_id)
            after = candidate.edge_values.get(event.node_or_edge_id)
        if event.before != before or event.after != after:
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "GraphDelta event before/after is not bound to locked state",
                origin_id=origin_id,
                policy=policy,
                evidence={
                    "event_id": event.event_id,
                    "target_kind": event.target_kind.value,
                    "target_id": event.node_or_edge_id,
                },
            )


def _stable_digest(*parts: Any) -> str:
    payload = json.dumps(
        _json_value(parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _graph_equivalent(
    left: Any,
    right: Any,
    *,
    origin_id: str,
    policy: PropagationPolicy,
    detail: str,
) -> bool:
    if type(left) is not str or not left or type(right) is not str or not right:
        raise BundleDraftError(
            "BUNDLE_H_STATE_INVALID",
            detail,
            origin_id=origin_id,
            policy=policy,
            evidence={"reason": "non_string_or_empty_smiles"},
        )
    try:
        return isomeric_graph_equivalent(left, right)
    except (RuntimeError, TypeError, ValueError) as error:
        raise BundleDraftError(
            "BUNDLE_H_STATE_INVALID",
            detail,
            origin_id=origin_id,
            policy=policy,
            evidence={"exception_type": type(error).__name__},
        ) from error


@dataclass(frozen=True, slots=True)
class PreparedHallucinatedVariant:
    """One validated H state, ready for quota assignment and H/N matching."""

    origin_id: str
    normalized_subtask: EditingSubtask
    input_view_id: str
    recipe: PerturbationRecipe
    operator_family: str
    quota_bucket: str
    renderer_backend: str
    reference_graph: StateDAG
    candidate_graph: StateDAG
    graph_delta: GraphDelta
    fallback_decision: FallbackDecision | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.input_view_id, "input_view_id"),
            (self.operator_family, "operator_family"),
            (self.quota_bucket, "quota_bucket"),
            (self.renderer_backend, "renderer_backend"),
        ):
            _nonempty(value, name=name)
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if type(self.recipe) is not PerturbationRecipe:
            raise TypeError("recipe must be PerturbationRecipe")
        if (
            type(self.reference_graph) is not StateDAG
            or type(self.candidate_graph) is not StateDAG
        ):
            raise TypeError("reference_graph and candidate_graph must be StateDAG")
        if type(self.graph_delta) is not GraphDelta:
            raise TypeError("graph_delta must be GraphDelta")
        if (
            self.fallback_decision is not None
            and type(self.fallback_decision) is not FallbackDecision
        ):
            raise TypeError("fallback_decision must be FallbackDecision or None")
        if self.recipe.origin_id != self.origin_id:
            raise BundleDraftError(
                "BUNDLE_INPUT_MISMATCH",
                "prepared variant recipe must share origin_id",
                origin_id=self.origin_id,
                policy=self.recipe.policy,
            )
        if self.recipe.policy not in _POLICY_INDEX:
            raise ValueError("prepared variant recipe uses unsupported policy")
        authoritative = state_schema_for(self.normalized_subtask)
        if (
            self.reference_graph.schema != authoritative
            or self.candidate_graph.schema != authoritative
        ):
            raise ValueError(
                "prepared variant graphs must use authoritative subtask schema"
            )
        differences = self.candidate_graph.semantic_differences(self.reference_graph)
        if not differences or differences != _delta_targets(self.graph_delta):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "prepared H graph_delta must exactly cover non-empty semantic differences",
                origin_id=self.origin_id,
                policy=self.recipe.policy,
            )
        _validate_delta_binding(
            self.reference_graph,
            self.candidate_graph,
            self.graph_delta,
            origin_id=self.origin_id,
            policy=self.recipe.policy,
        )
        roots = self.graph_delta.root_events
        if (
            len(roots) != 1
            or roots[0].target_kind is not MutationTargetKind.NODE
            or roots[0].node_or_edge_id != self.recipe.target_node_id
        ):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "prepared H delta root must equal recipe target_node_id",
                origin_id=self.origin_id,
                policy=self.recipe.policy,
            )
        if any(
            event.operator_id != self.recipe.operator_id
            for event in self.graph_delta.events
        ):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "prepared H delta operator_id must equal recipe operator_id",
                origin_id=self.origin_id,
                policy=self.recipe.policy,
            )
        if self.fallback_decision is not None:
            decision = self.fallback_decision
            if not (
                decision.selected_operator_id == self.recipe.operator_id
                and decision.selected_operator_family == self.operator_family
                and decision.policy is self.recipe.policy
                and decision.candidate_source is self.recipe.candidate_source_mode
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback decision does not describe prepared variant",
                    origin_id=self.origin_id,
                    policy=self.recipe.policy,
                )
        self._validate_policy_shape(differences)

    def _validate_policy_shape(
        self,
        differences: frozenset[tuple[MutationTargetKind, str]],
    ) -> None:
        policy = self.recipe.policy
        node_differences = frozenset(
            target_id
            for target_kind, target_id in differences
            if target_kind is MutationTargetKind.NODE
        )
        if len(node_differences) != len(differences):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "Phase-2 editing variants may mutate typed state nodes only",
                origin_id=self.origin_id,
                policy=policy,
            )
        try:
            rendered_answer = DeterministicAnswerRenderer().render(
                self.candidate_graph,
                policy=policy,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "prepared H state fails the deterministic Answer contract",
                origin_id=self.origin_id,
                policy=policy,
                evidence={"exception_type": type(error).__name__},
            ) from error
        if policy is PropagationPolicy.STOP:
            if len(differences) != 1 or self.recipe.target_node_id == "final_answer":
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "STOP H must contain exactly its non-terminal root mutation",
                    origin_id=self.origin_id,
                    policy=policy,
                )
            return
        if policy is PropagationPolicy.PARTIAL:
            if len(differences) <= 1 or self.recipe.target_node_id == "final_answer":
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "PARTIAL H must contain a non-terminal root and downstream mutation",
                    origin_id=self.origin_id,
                    policy=policy,
                )
            return
        if policy is PropagationPolicy.FULL_CF:
            reference_product = self.reference_graph.value_for(
                "product"
            ).normalized_value
            candidate_product = self.candidate_graph.value_for(
                "product"
            ).normalized_value
            if rendered_answer.product_equivalent is not True or _graph_equivalent(
                reference_product,
                candidate_product,
                origin_id=self.origin_id,
                policy=policy,
                detail="FULL_CF reference/candidate products cannot be compared",
            ):
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "FULL_CF H requires a wrong product and an equivalent locked Answer",
                    origin_id=self.origin_id,
                    policy=policy,
                )
            return
        if not (
            policy is PropagationPolicy.TERMINAL
            and differences == frozenset({(MutationTargetKind.NODE, "final_answer")})
            and self.recipe.target_node_id == "final_answer"
            and rendered_answer.product_equivalent is False
        ):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "TERMINAL H must change only Answer to a non-equivalent molecule",
                origin_id=self.origin_id,
                policy=policy,
            )

    @property
    def requested_operator_id(self) -> str:
        if self.fallback_decision is None:
            return self.recipe.operator_id
        return self.fallback_decision.requested_operator_id

    @property
    def quota_deviation(self) -> bool:
        return (
            False
            if self.fallback_decision is None
            else self.fallback_decision.quota_deviation
        )

    @property
    def fallback_rank(self) -> int:
        if self.fallback_decision is None:
            return 0
        return 2 if self.fallback_decision.quota_deviation else 1

    @property
    def stable_key(self) -> tuple[Any, ...]:
        return (
            self.origin_id,
            _POLICY_INDEX[self.recipe.policy],
            self.fallback_rank,
            self.recipe.operator_id,
            self.quota_bucket,
            self.recipe.target_node_id,
            self.recipe.candidate_source_mode.value,
            self.recipe.variant_index,
            self.recipe.derived_seed,
            self.recipe.recipe_id,
            self.selection_fingerprint,
        )

    @property
    def selection_fingerprint(self) -> str:
        def claim_payload(claim: Any) -> tuple[Any, ...]:
            return (
                claim.raw_value,
                claim.normalized_value,
                claim.value_type,
                claim.provenance,
                claim.locally_valid,
                claim.oracle_match,
                claim.confidence,
                claim.mention_ids,
            )

        reference_values = {
            node_id: claim_payload(claim)
            for node_id, claim in self.reference_graph.values.items()
        }
        candidate_values = {
            node_id: claim_payload(claim)
            for node_id, claim in self.candidate_graph.values.items()
        }
        reference_edges = {
            edge_id: claim_payload(claim)
            for edge_id, claim in self.reference_graph.edge_values.items()
        }
        candidate_edges = {
            edge_id: claim_payload(claim)
            for edge_id, claim in self.candidate_graph.edge_values.items()
        }
        delta = tuple(
            (
                event.event_id,
                event.target_kind,
                event.node_or_edge_id,
                event.before.normalized_value,
                event.after.normalized_value,
                event.causal_role,
            )
            for event in self.graph_delta.events
        )
        return _stable_digest(
            "prepared_h_v1",
            self.origin_id,
            self.normalized_subtask,
            self.recipe.recipe_id,
            self.recipe.operator_id,
            self.recipe.policy,
            self.recipe.target_node_id,
            self.recipe.candidate_source_mode,
            self.input_view_id,
            self.renderer_backend,
            self.operator_family,
            self.quota_bucket,
            self.recipe.renderer_style_id,
            (
                self.recipe.rewrite_budget.max_changed_claims,
                self.recipe.rewrite_budget.max_added_characters,
                self.recipe.rewrite_budget.length_bucket,
            ),
            self.recipe.candidate_difficulty_bucket,
            tuple(sorted(self.recipe.partial_cut_nodes)),
            self.recipe.constraints,
            (
                None
                if self.fallback_decision is None
                else (
                    self.fallback_decision.requested_operator_id,
                    self.fallback_decision.selected_operator_id,
                    self.fallback_decision.requested_operator_family,
                    self.fallback_decision.selected_operator_family,
                    self.fallback_decision.policy,
                    self.fallback_decision.candidate_source,
                    self.fallback_decision.quota_bucket,
                    self.fallback_decision.attempted_operator_ids,
                    self.fallback_decision.quota_deviation,
                    self.fallback_decision.target_change_required,
                )
            ),
            reference_values,
            candidate_values,
            reference_edges,
            candidate_edges,
            delta,
        )


@dataclass(frozen=True, slots=True)
class QuotaScheduleRequest:
    normalized_subtask: EditingSubtask
    origin_ids: tuple[str, ...]
    variants: tuple[PreparedHallucinatedVariant, ...]
    global_seed: int
    seed_namespace: str
    allow_quota_deviation: bool = False

    def __post_init__(self) -> None:
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if type(self.allow_quota_deviation) is not bool:
            raise TypeError("allow_quota_deviation must be bool")
        if type(self.global_seed) is not int or self.global_seed < 0:
            raise ValueError("global_seed must be a non-negative integer")
        _nonempty(self.seed_namespace, name="seed_namespace")
        origins = tuple(self.origin_ids)
        if any(type(origin_id) is not str or not origin_id for origin_id in origins):
            raise TypeError("origin_ids must contain non-empty strings")
        if len(origins) != 50 or len(set(origins)) != 50:
            raise ValueError("quota scheduling requires exactly 50 unique origins")
        variants = tuple(self.variants)
        if any(type(item) is not PreparedHallucinatedVariant for item in variants):
            raise TypeError("variants must contain PreparedHallucinatedVariant values")
        if any(
            item.origin_id not in origins
            or item.normalized_subtask is not self.normalized_subtask
            for item in variants
        ):
            raise ValueError("quota variants differ from request origins/subtask")
        object.__setattr__(self, "origin_ids", tuple(sorted(origins)))
        object.__setattr__(
            self,
            "variants",
            tuple(sorted(variants, key=lambda item: item.stable_key)),
        )


@dataclass(frozen=True, slots=True)
class QuotaAssignment:
    origin_id: str
    policy: PropagationPolicy
    quota_bucket: str
    variant: PreparedHallucinatedVariant

    def __post_init__(self) -> None:
        _nonempty(self.origin_id, name="origin_id")
        _nonempty(self.quota_bucket, name="quota_bucket")
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        if type(self.variant) is not PreparedHallucinatedVariant:
            raise TypeError("variant must be PreparedHallucinatedVariant")
        if not (
            self.variant.origin_id == self.origin_id
            and self.variant.recipe.policy is self.policy
            and self.variant.quota_bucket == self.quota_bucket
        ):
            raise ValueError(
                "quota assignment identity/phenotype differs from prepared variant"
            )

    def to_dict(self) -> dict[str, Any]:
        decision = self.variant.fallback_decision
        return {
            "origin_id": self.origin_id,
            "policy": self.policy.dataset_name,
            "quota_bucket": self.quota_bucket,
            "requested_quota_bucket": (
                self.quota_bucket if decision is None else decision.quota_bucket
            ),
            "requested_operator_id": self.variant.requested_operator_id,
            "selected_operator_id": self.variant.recipe.operator_id,
            "selected_operator_family": self.variant.operator_family,
            "candidate_source": self.variant.recipe.candidate_source_mode.value,
            "quota_deviation": self.variant.quota_deviation,
        }


@dataclass(frozen=True, slots=True)
class QuotaDeviationEntry:
    origin_id: str
    policy: PropagationPolicy
    quota_bucket: str
    selected_quota_bucket: str
    requested_operator_id: str
    selected_operator_id: str
    requested_operator_family: str
    selected_operator_family: str
    attempted_operator_ids: tuple[str, ...]
    candidate_source: CandidateSourceType
    quota_deviation: bool
    target_change_required: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.quota_bucket, "quota_bucket"),
            (self.selected_quota_bucket, "selected_quota_bucket"),
            (self.requested_operator_id, "requested_operator_id"),
            (self.selected_operator_id, "selected_operator_id"),
            (self.requested_operator_family, "requested_operator_family"),
            (self.selected_operator_family, "selected_operator_family"),
        ):
            _nonempty(value, name=name)
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        if type(self.candidate_source) is not CandidateSourceType:
            raise TypeError("candidate_source must be CandidateSourceType")
        attempts = tuple(self.attempted_operator_ids)
        if any(type(item) is not str or not item for item in attempts):
            raise TypeError("attempted_operator_ids must contain non-empty strings")
        if len(attempts) != len(set(attempts)):
            raise ValueError("attempted_operator_ids must be unique")
        if type(self.quota_deviation) is not bool:
            raise TypeError("quota_deviation must be bool")
        if type(self.target_change_required) is not bool:
            raise TypeError("target_change_required must be bool")
        object.__setattr__(self, "attempted_operator_ids", attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "policy": self.policy.dataset_name,
            "quota_bucket": self.quota_bucket,
            "selected_quota_bucket": self.selected_quota_bucket,
            "requested_operator_id": self.requested_operator_id,
            "selected_operator_id": self.selected_operator_id,
            "requested_operator_family": self.requested_operator_family,
            "selected_operator_family": self.selected_operator_family,
            "attempted_operator_ids": list(self.attempted_operator_ids),
            "candidate_source": self.candidate_source.value,
            "quota_deviation": self.quota_deviation,
            "target_change_required": self.target_change_required,
        }


@dataclass(frozen=True, slots=True)
class QuotaBucketCount:
    policy: PropagationPolicy
    quota_bucket: str
    target_count: int
    actual_count: int

    def __post_init__(self) -> None:
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        _nonempty(self.quota_bucket, name="quota_bucket")
        if any(
            type(value) is not int or value < 0
            for value in (self.target_count, self.actual_count)
        ):
            raise ValueError("quota counts must be non-negative integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.dataset_name,
            "quota_bucket": self.quota_bucket,
            "target_count": self.target_count,
            "actual_count": self.actual_count,
        }


@dataclass(frozen=True, slots=True)
class QuotaBackfillRequirement:
    policy: PropagationPolicy
    quota_bucket: str
    missing_count: int
    unassigned_origin_ids: tuple[str, ...]
    reason_code: str = "BACKFILL_REQUIRED"

    def __post_init__(self) -> None:
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        _nonempty(self.quota_bucket, name="quota_bucket")
        _nonempty(self.reason_code, name="reason_code")
        if type(self.missing_count) is not int or self.missing_count <= 0:
            raise ValueError("missing_count must be positive")
        origins = tuple(self.unassigned_origin_ids)
        if any(type(item) is not str or not item for item in origins):
            raise TypeError("unassigned_origin_ids must contain non-empty strings")
        object.__setattr__(self, "unassigned_origin_ids", tuple(sorted(set(origins))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.dataset_name,
            "quota_bucket": self.quota_bucket,
            "missing_count": self.missing_count,
            "unassigned_origin_ids": list(self.unassigned_origin_ids),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class QuotaScheduleReport:
    normalized_subtask: EditingSubtask
    origin_count: int
    counts: tuple[QuotaBucketCount, ...]
    deviations: tuple[QuotaDeviationEntry, ...] = ()
    backfills: tuple[QuotaBackfillRequirement, ...] = ()
    format_version: str = "quota_schedule_report_v1"

    def __post_init__(self) -> None:
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if self.origin_count != 50:
            raise ValueError("quota report origin_count must equal 50")
        counts = tuple(self.counts)
        deviations = tuple(self.deviations)
        backfills = tuple(self.backfills)
        if any(type(item) is not QuotaBucketCount for item in counts):
            raise TypeError("counts must contain QuotaBucketCount")
        if any(type(item) is not QuotaDeviationEntry for item in deviations):
            raise TypeError("deviations must contain QuotaDeviationEntry")
        if any(type(item) is not QuotaBackfillRequirement for item in backfills):
            raise TypeError("backfills must contain QuotaBackfillRequirement")
        if self.format_version != "quota_schedule_report_v1":
            raise ValueError("unsupported quota report format version")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "deviations", deviations)
        object.__setattr__(self, "backfills", backfills)

    @property
    def all_pass(self) -> bool:
        return not self.backfills and all(
            item.actual_count == item.target_count for item in self.counts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "normalized_subtask": self.normalized_subtask.value,
            "origin_count": self.origin_count,
            "all_pass": self.all_pass,
            "counts": [item.to_dict() for item in self.counts],
            "deviations": [item.to_dict() for item in self.deviations],
            "backfills": [item.to_dict() for item in self.backfills],
        }


class QuotaScheduleError(BundleDraftError):
    """Raised when an exact per-50 schedule requires explicit backfill."""

    def __init__(self, report: QuotaScheduleReport) -> None:
        if type(report) is not QuotaScheduleReport:
            raise TypeError("QuotaScheduleError requires QuotaScheduleReport")
        self.report = report
        super().__init__(
            "BACKFILL_REQUIRED",
            "exact frozen operator quotas cannot be filled by prepared variants",
            evidence=report.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class QuotaSchedule:
    assignments: tuple[QuotaAssignment, ...]
    deviations: tuple[QuotaDeviationEntry, ...]
    report: QuotaScheduleReport

    def __post_init__(self) -> None:
        assignments = tuple(self.assignments)
        deviations = tuple(self.deviations)
        if any(type(item) is not QuotaAssignment for item in assignments):
            raise TypeError("assignments must contain QuotaAssignment")
        if any(type(item) is not QuotaDeviationEntry for item in deviations):
            raise TypeError("deviations must contain QuotaDeviationEntry")
        if type(self.report) is not QuotaScheduleReport or not self.report.all_pass:
            raise ValueError("successful QuotaSchedule requires passing report")
        if deviations != self.report.deviations:
            raise ValueError("schedule deviations must equal report deviations")
        keys = tuple((item.origin_id, item.policy) for item in assignments)
        if len(assignments) != 200 or len(keys) != len(set(keys)):
            raise ValueError("schedule must assign four policies to exactly 50 origins")
        expected_order = tuple(
            sorted(keys, key=lambda item: (item[0], _POLICY_INDEX[item[1]]))
        )
        if keys != expected_order:
            raise ValueError("quota assignments must use stable origin/policy order")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "deviations", deviations)

    def assignments_for_origin(self, origin_id: str) -> tuple[QuotaAssignment, ...]:
        _nonempty(origin_id, name="origin_id")
        return tuple(item for item in self.assignments if item.origin_id == origin_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [item.to_dict() for item in self.assignments],
            "deviations": [item.to_dict() for item in self.deviations],
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MatchedBundleBuildRequest:
    origin_id: str
    normalized_subtask: EditingSubtask
    input_view_id: str
    assignments: tuple[QuotaAssignment, ...]

    def __post_init__(self) -> None:
        _nonempty(self.origin_id, name="origin_id")
        _nonempty(self.input_view_id, name="input_view_id")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        assignments = tuple(self.assignments)
        if any(type(item) is not QuotaAssignment for item in assignments):
            raise BundleDraftError(
                "BUNDLE_POLICY_SET_MISMATCH",
                "assignments must contain QuotaAssignment values",
                origin_id=self.origin_id,
            )
        if len(assignments) != 4 or {item.policy for item in assignments} != set(
            _POLICY_ORDER
        ):
            raise BundleDraftError(
                "BUNDLE_POLICY_SET_MISMATCH",
                "bundle request requires exactly one assignment per policy",
                origin_id=self.origin_id,
            )
        if any(
            item.origin_id != self.origin_id
            or item.variant.normalized_subtask is not self.normalized_subtask
            or item.variant.input_view_id != self.input_view_id
            for item in assignments
        ):
            raise BundleDraftError(
                "BUNDLE_INPUT_MISMATCH",
                "bundle assignments differ from request identity",
                origin_id=self.origin_id,
            )
        object.__setattr__(
            self,
            "assignments",
            tuple(sorted(assignments, key=lambda item: _POLICY_INDEX[item.policy])),
        )


@dataclass(frozen=True, slots=True)
class MatchedDraftRecord:
    record_id: str
    origin_id: str
    bundle_id: str
    pair_id: str
    matched_record_id: str
    variant_label: VariantLabel
    policy: PropagationPolicy
    input_view_id: str
    target_node_id: str
    target_step_index: int | None
    operator_id: str
    operator_family: str
    quota_bucket: str
    candidate_source: CandidateSourceType
    renderer_backend: str
    renderer_style_id: str
    rewrite_budget: RewriteBudget
    candidate_difficulty_bucket: str
    fallback_decision: FallbackDecision | None
    control_identity: str
    render_identity: str
    reference_graph: StateDAG
    locked_state: StateDAG
    graph_delta: GraphDelta
    formal_trace: RenderedFormalTrace
    answer: RenderedAnswer

    def __post_init__(self) -> None:
        for value, name in (
            (self.record_id, "record_id"),
            (self.origin_id, "origin_id"),
            (self.bundle_id, "bundle_id"),
            (self.pair_id, "pair_id"),
            (self.matched_record_id, "matched_record_id"),
            (self.input_view_id, "input_view_id"),
            (self.target_node_id, "target_node_id"),
            (self.operator_id, "operator_id"),
            (self.operator_family, "operator_family"),
            (self.quota_bucket, "quota_bucket"),
            (self.renderer_backend, "renderer_backend"),
            (self.renderer_style_id, "renderer_style_id"),
            (self.candidate_difficulty_bucket, "candidate_difficulty_bucket"),
            (self.control_identity, "control_identity"),
            (self.render_identity, "render_identity"),
        ):
            _nonempty(value, name=name)
        if type(self.variant_label) is not VariantLabel:
            raise TypeError("variant_label must be VariantLabel")
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        if type(self.candidate_source) is not CandidateSourceType:
            raise TypeError("candidate_source must be CandidateSourceType")
        if type(self.rewrite_budget) is not RewriteBudget:
            raise TypeError("rewrite_budget must be RewriteBudget")
        if (
            self.fallback_decision is not None
            and type(self.fallback_decision) is not FallbackDecision
        ):
            raise TypeError("fallback_decision must be FallbackDecision or None")
        if self.fallback_decision is not None:
            decision = self.fallback_decision
            if not (
                decision.selected_operator_id == self.operator_id
                and decision.selected_operator_family == self.operator_family
                and decision.policy is self.policy
                and decision.candidate_source is self.candidate_source
                and not decision.target_change_required
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "draft fallback ledger differs from its frozen matching axes",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
        if self.target_step_index is not None and (
            type(self.target_step_index) is not int or self.target_step_index <= 0
        ):
            raise ValueError("target_step_index must be positive or None")
        for value, expected, name in (
            (self.reference_graph, StateDAG, "reference_graph"),
            (self.locked_state, StateDAG, "locked_state"),
            (self.graph_delta, GraphDelta, "graph_delta"),
            (self.formal_trace, RenderedFormalTrace, "formal_trace"),
            (self.answer, RenderedAnswer, "answer"),
        ):
            if type(value) is not expected:
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.reference_graph.schema != self.locked_state.schema:
            raise ValueError("draft reference/locked graphs must use identical schema")
        target = self.locked_state.schema.nodes_by_id.get(self.target_node_id)
        if target is None or target.visibility is not Visibility.CANDIDATE_OUTPUT:
            raise ValueError("draft target must be a candidate-output node")
        if target.step_index != self.target_step_index:
            raise ValueError("target_step_index differs from schema target")
        if self.formal_trace.schema_id != self.locked_state.schema.schema_id:
            raise ValueError("FORMAL trace schema differs from locked state")
        if self.answer.policy is not self.policy:
            raise ValueError("Answer policy differs from draft policy")
        try:
            expected_formal = DeterministicFormalRenderer().render(self.locked_state)
            expected_answer = DeterministicAnswerRenderer().render(
                self.locked_state,
                policy=self.policy,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "locked draft state fails deterministic FORMAL/Answer rendering",
                origin_id=self.origin_id,
                policy=self.policy,
                evidence={"exception_type": type(error).__name__},
            ) from error
        if self.formal_trace != expected_formal or self.answer != expected_answer:
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "draft rendering is not the exact T023 projection of locked state",
                origin_id=self.origin_id,
                policy=self.policy,
            )
        differences = self.locked_state.semantic_differences(self.reference_graph)
        if differences != _delta_targets(self.graph_delta):
            raise BundleDraftError(
                "BUNDLE_H_STATE_INVALID",
                "draft GraphDelta differs from locked/reference semantics",
                origin_id=self.origin_id,
                policy=self.policy,
            )
        _validate_delta_binding(
            self.reference_graph,
            self.locked_state,
            self.graph_delta,
            origin_id=self.origin_id,
            policy=self.policy,
        )
        if self.variant_label is VariantLabel.HALLUCINATED:
            if not differences:
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "H draft records require semantic mutation",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
            roots = self.graph_delta.root_events
            if (
                len(roots) != 1
                or roots[0].target_kind is not MutationTargetKind.NODE
                or roots[0].node_or_edge_id != self.target_node_id
            ):
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "H draft delta root differs from its frozen target",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
            if self.policy is PropagationPolicy.STOP and len(differences) != 1:
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "STOP H must change exactly one root",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
            if self.policy is PropagationPolicy.PARTIAL and len(differences) <= 1:
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "PARTIAL H must include a downstream semantic mutation",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
            if self.policy is PropagationPolicy.FULL_CF:
                reference_product = self.reference_graph.value_for(
                    "product"
                ).normalized_value
                locked_product = self.locked_state.value_for("product").normalized_value
                if not self.answer.product_equivalent or _graph_equivalent(
                    reference_product,
                    locked_product,
                    origin_id=self.origin_id,
                    policy=self.policy,
                    detail="FULL_CF reference/candidate products cannot be compared",
                ):
                    raise BundleDraftError(
                        "BUNDLE_H_STATE_INVALID",
                        "FULL_CF H requires a wrong product and equivalent locked Answer",
                        origin_id=self.origin_id,
                        policy=self.policy,
                    )
            if self.policy is PropagationPolicy.TERMINAL and not (
                differences == frozenset({(MutationTargetKind.NODE, "final_answer")})
                and self.target_node_id == "final_answer"
                and not self.answer.product_equivalent
            ):
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "TERMINAL H must be an Answer-only non-equivalent mutation",
                    origin_id=self.origin_id,
                    policy=self.policy,
                )
        elif differences or self.graph_delta.events:
            raise BundleDraftError(
                "BUNDLE_PAIR_MISMATCH",
                "N draft records require faithful state and empty GraphDelta",
                origin_id=self.origin_id,
                policy=self.policy,
            )
        elif not self.answer.product_equivalent:
            raise BundleDraftError(
                "BUNDLE_PAIR_MISMATCH",
                "faithful N Answer must be graph-equivalent to its locked product",
                origin_id=self.origin_id,
                policy=self.policy,
            )
        if self.record_id == self.matched_record_id:
            raise ValueError("draft record cannot match itself")

    def to_dict(self) -> dict[str, Any]:
        visible_values = {
            node_id: claim.normalized_value
            for node_id, claim in self.locked_state.values.items()
            if self.locked_state.schema.nodes_by_id[node_id].visibility
            is Visibility.CANDIDATE_OUTPUT
        }
        return {
            "record_id": self.record_id,
            "origin_id": self.origin_id,
            "bundle_id": self.bundle_id,
            "pair_id": self.pair_id,
            "matched_record_id": self.matched_record_id,
            "variant_label": self.variant_label.value,
            "policy": self.policy.dataset_name,
            "input_view_id": self.input_view_id,
            "target_node_id": self.target_node_id,
            "target_step_index": self.target_step_index,
            "operator_id": self.operator_id,
            "operator_family": self.operator_family,
            "quota_bucket": self.quota_bucket,
            "candidate_source": self.candidate_source.value,
            "renderer_backend": self.renderer_backend,
            "renderer_style_id": self.renderer_style_id,
            "rewrite_budget": {
                "max_changed_claims": self.rewrite_budget.max_changed_claims,
                "max_added_characters": self.rewrite_budget.max_added_characters,
                "length_bucket": self.rewrite_budget.length_bucket,
            },
            "candidate_difficulty_bucket": self.candidate_difficulty_bucket,
            "fallback": (
                None
                if self.fallback_decision is None
                else {
                    "requested_operator_id": (
                        self.fallback_decision.requested_operator_id
                    ),
                    "selected_operator_id": self.fallback_decision.selected_operator_id,
                    "requested_operator_family": (
                        self.fallback_decision.requested_operator_family
                    ),
                    "selected_operator_family": (
                        self.fallback_decision.selected_operator_family
                    ),
                    "policy": self.fallback_decision.policy.dataset_name,
                    "candidate_source": self.fallback_decision.candidate_source.value,
                    "quota_bucket": self.fallback_decision.quota_bucket,
                    "attempted_operator_ids": list(
                        self.fallback_decision.attempted_operator_ids
                    ),
                    "quota_deviation": self.fallback_decision.quota_deviation,
                    "target_change_required": (
                        self.fallback_decision.target_change_required
                    ),
                }
            ),
            "control_identity": self.control_identity,
            "render_identity": self.render_identity,
            "state_values": _json_value(visible_values),
            "graph_delta": [
                {
                    "event_id": event.event_id,
                    "target_kind": event.target_kind.value,
                    "target_id": event.node_or_edge_id,
                    "causal_role": event.causal_role.value,
                    "operator_id": event.operator_id,
                }
                for event in self.graph_delta.events
            ],
            "formal": [
                {
                    "step_index": step.step_index,
                    "step_name": step.step_name,
                    "formal_ab": step.formal_ab,
                }
                for step in self.formal_trace.steps
            ],
            "answer": {
                "smiles": self.answer.smiles,
                "product_equivalent": self.answer.product_equivalent,
            },
        }


@dataclass(frozen=True, slots=True)
class MatchedBundleDraft:
    origin_id: str
    bundle_id: str
    records: tuple[MatchedDraftRecord, ...]

    def __post_init__(self) -> None:
        _nonempty(self.origin_id, name="origin_id")
        _nonempty(self.bundle_id, name="bundle_id")
        records = tuple(self.records)
        if len(records) != 8 or any(
            type(item) is not MatchedDraftRecord for item in records
        ):
            raise BundleDraftError(
                "BUNDLE_POLICY_SET_MISMATCH",
                "MatchedBundleDraft requires exactly eight draft records",
                origin_id=self.origin_id,
            )
        expected_order = tuple(
            (policy, label)
            for policy in _POLICY_ORDER
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        )
        if (
            tuple((item.policy, item.variant_label) for item in records)
            != expected_order
        ):
            raise BundleDraftError(
                "BUNDLE_POLICY_SET_MISMATCH",
                "draft records must use stable policy H/N order",
                origin_id=self.origin_id,
            )
        if any(
            item.origin_id != self.origin_id or item.bundle_id != self.bundle_id
            for item in records
        ):
            raise BundleDraftError(
                "BUNDLE_PAIR_MISMATCH",
                "draft records must share origin_id and bundle_id",
                origin_id=self.origin_id,
            )
        by_id = {item.record_id: item for item in records}
        if len(by_id) != 8:
            raise BundleDraftError(
                "BUNDLE_PAIR_MISMATCH",
                "draft record IDs must be unique",
                origin_id=self.origin_id,
            )
        controls: set[str] = set()
        render_ids: set[str] = set()
        common_reference = records[0].reference_graph
        for h_record, n_record in zip(records[::2], records[1::2], strict=True):
            shared_axes = (
                "pair_id",
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
            if any(
                getattr(h_record, name) != getattr(n_record, name)
                for name in shared_axes
            ):
                raise BundleDraftError(
                    "BUNDLE_PAIR_MISMATCH",
                    "matched H/N records differ on a frozen matching axis",
                    origin_id=self.origin_id,
                    policy=h_record.policy,
                )
            if not (
                h_record.matched_record_id == n_record.record_id
                and n_record.matched_record_id == h_record.record_id
            ):
                raise BundleDraftError(
                    "BUNDLE_PAIR_MISMATCH",
                    "matched H/N record links must be reciprocal",
                    origin_id=self.origin_id,
                    policy=h_record.policy,
                )
            if not (
                h_record.reference_graph == common_reference
                and n_record.reference_graph == common_reference
                and n_record.locked_state == common_reference
            ):
                raise BundleDraftError(
                    "BUNDLE_PAIR_MISMATCH",
                    "all pairs must share one exact faithful reference state",
                    origin_id=self.origin_id,
                    policy=h_record.policy,
                )
            if h_record.control_identity in controls:
                raise BundleDraftError(
                    "BUNDLE_CONTROL_REUSE",
                    "faithful control identity cannot be reused across pairs",
                    origin_id=self.origin_id,
                    policy=h_record.policy,
                )
            controls.add(h_record.control_identity)
            for record in (h_record, n_record):
                if record.render_identity in render_ids:
                    raise BundleDraftError(
                        "BUNDLE_CONTROL_REUSE",
                        "pair-specific render identity cannot be reused",
                        origin_id=self.origin_id,
                        policy=h_record.policy,
                    )
                render_ids.add(record.render_identity)
        object.__setattr__(self, "records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": "matched_bundle_draft_v1",
            "origin_id": self.origin_id,
            "bundle_id": self.bundle_id,
            "records": [item.to_dict() for item in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class MatchedBundleBuilder:
    """Build four reciprocal H/N semantic drafts from scheduled H variants."""

    __slots__ = ("_answer_renderer", "_formal_renderer")

    def __init__(self) -> None:
        self._formal_renderer = DeterministicFormalRenderer()
        self._answer_renderer = DeterministicAnswerRenderer()

    def build(self, request: MatchedBundleBuildRequest) -> MatchedBundleDraft:
        if type(request) is not MatchedBundleBuildRequest:
            raise TypeError("build requires MatchedBundleBuildRequest")
        assignments = tuple(request.assignments)
        if (
            len(assignments) != 4
            or any(type(item) is not QuotaAssignment for item in assignments)
            or {item.policy for item in assignments} != set(_POLICY_ORDER)
        ):
            raise BundleDraftError(
                "BUNDLE_POLICY_SET_MISMATCH",
                "bundle build requires one exact assignment for every policy",
                origin_id=request.origin_id,
            )
        if any(
            item.origin_id != request.origin_id
            or item.variant.normalized_subtask is not request.normalized_subtask
            or item.variant.input_view_id != request.input_view_id
            or item.variant.recipe.policy is not item.policy
            or item.variant.quota_bucket != item.quota_bucket
            for item in assignments
        ):
            raise BundleDraftError(
                "BUNDLE_INPUT_MISMATCH",
                "bundle assignments differ from the frozen build identity",
                origin_id=request.origin_id,
            )
        bundle_id = f"{request.origin_id}__bundle_phase2"
        records: list[MatchedDraftRecord] = []
        reference_graph: StateDAG | None = None
        for assignment in request.assignments:
            variant = assignment.variant
            if reference_graph is None:
                reference_graph = variant.reference_graph
            elif reference_graph != variant.reference_graph:
                raise BundleDraftError(
                    "BUNDLE_INPUT_MISMATCH",
                    "all four H variants must share one reference state",
                    origin_id=request.origin_id,
                )
            try:
                h_formal = self._formal_renderer.render(variant.candidate_graph)
                h_answer = self._answer_renderer.render(
                    variant.candidate_graph,
                    policy=assignment.policy,
                )
                n_formal = self._formal_renderer.render(variant.reference_graph)
                n_answer = self._answer_renderer.render(
                    variant.reference_graph,
                    policy=assignment.policy,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise BundleDraftError(
                    "BUNDLE_H_STATE_INVALID",
                    "T023 FORMAL/Answer audit rejected a locked bundle state",
                    origin_id=request.origin_id,
                    policy=assignment.policy,
                    evidence={"exception_type": type(error).__name__},
                ) from error
            policy_name = assignment.policy.dataset_name
            pair_id = f"{request.origin_id}__{policy_name}"
            h_id = f"{pair_id}__H"
            n_id = f"{pair_id}__N"
            control_identity = f"{pair_id}__faithful_control"
            target_spec = variant.candidate_graph.schema.nodes_by_id[
                variant.recipe.target_node_id
            ]
            common = {
                "origin_id": request.origin_id,
                "bundle_id": bundle_id,
                "pair_id": pair_id,
                "policy": assignment.policy,
                "input_view_id": request.input_view_id,
                "target_node_id": variant.recipe.target_node_id,
                "target_step_index": target_spec.step_index,
                "operator_id": variant.recipe.operator_id,
                "operator_family": variant.operator_family,
                "quota_bucket": assignment.quota_bucket,
                "candidate_source": variant.recipe.candidate_source_mode,
                "renderer_backend": variant.renderer_backend,
                "renderer_style_id": variant.recipe.renderer_style_id,
                "rewrite_budget": variant.recipe.rewrite_budget,
                "candidate_difficulty_bucket": (
                    variant.recipe.candidate_difficulty_bucket
                ),
                "fallback_decision": variant.fallback_decision,
                "control_identity": control_identity,
                "reference_graph": variant.reference_graph,
            }
            records.extend(
                (
                    MatchedDraftRecord(
                        record_id=h_id,
                        matched_record_id=n_id,
                        variant_label=VariantLabel.HALLUCINATED,
                        render_identity=f"{h_id}__render",
                        locked_state=variant.candidate_graph,
                        graph_delta=variant.graph_delta,
                        formal_trace=h_formal,
                        answer=h_answer,
                        **common,
                    ),
                    MatchedDraftRecord(
                        record_id=n_id,
                        matched_record_id=h_id,
                        variant_label=VariantLabel.FAITHFUL,
                        render_identity=f"{n_id}__faithful_rerender",
                        locked_state=variant.reference_graph,
                        graph_delta=GraphDelta(()),
                        formal_trace=n_formal,
                        answer=n_answer,
                        **common,
                    ),
                )
            )
        return MatchedBundleDraft(
            origin_id=request.origin_id,
            bundle_id=bundle_id,
            records=tuple(records),
        )


class DeterministicQuotaScheduler:
    """Assign exact frozen per-50 buckets with explicit fallback provenance."""

    __slots__ = ("_operators_config", "_registry")

    def __init__(
        self,
        *,
        operators_config: OperatorsConfig,
        registry: PerturbatorRegistry,
    ) -> None:
        if type(operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")
        if type(registry) is not PerturbatorRegistry:
            raise TypeError("registry must be PerturbatorRegistry")
        self._operators_config = operators_config
        self._registry = registry

    def _validate_variant(self, variant: PreparedHallucinatedVariant) -> None:
        try:
            registration = self._registry.registration(variant.recipe.operator_id)
        except RuntimeError as error:
            raise BundleDraftError(
                "QUOTA_INPUT_MISMATCH",
                "prepared variant operator is absent from registry",
                origin_id=variant.origin_id,
                policy=variant.recipe.policy,
            ) from error
        if not (
            registration.subtask is variant.normalized_subtask
            and registration.operator_family == variant.operator_family
            and variant.recipe.policy in registration.spec.supported_policies
            and variant.recipe.candidate_source_mode
            in registration.spec.supported_sources
            and variant.recipe.target_node_id in registration.spec.root_fields
        ):
            raise BundleDraftError(
                "QUOTA_INPUT_MISMATCH",
                "prepared variant differs from frozen operator registration",
                origin_id=variant.origin_id,
                policy=variant.recipe.policy,
            )
        family = self._operators_config.families.get(variant.operator_family)
        if family is None:
            raise BundleDraftError(
                "QUOTA_CONFIG_MISMATCH",
                "prepared variant operator family is absent from frozen config",
                origin_id=variant.origin_id,
                policy=variant.recipe.policy,
            )
        mapped_selected_families = self._operators_config.quota_bucket_mappings.get(
            variant.quota_bucket
        )
        if (
            mapped_selected_families is None
            or variant.operator_family not in mapped_selected_families
        ):
            raise BundleDraftError(
                "QUOTA_CONFIG_MISMATCH",
                "prepared variant quota phenotype is incompatible with its operator family",
                origin_id=variant.origin_id,
                policy=variant.recipe.policy,
                evidence={
                    "quota_bucket": variant.quota_bucket,
                    "operator_family": variant.operator_family,
                },
            )
        decision = variant.fallback_decision
        if decision is not None:
            try:
                requested = self._registry.registration(decision.requested_operator_id)
            except RuntimeError as error:
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback requested operator is absent from registry",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                ) from error
            if not (
                requested.operator_family == decision.requested_operator_family
                and requested.task_family is registration.task_family
                and requested.subtask is variant.normalized_subtask
                and variant.recipe.policy in requested.spec.supported_policies
                and variant.recipe.candidate_source_mode
                in requested.spec.supported_sources
                and variant.recipe.target_node_id in requested.spec.root_fields
                and decision.selected_operator_id == variant.recipe.operator_id
                and decision.selected_operator_family == variant.operator_family
                and decision.policy is variant.recipe.policy
                and decision.candidate_source is variant.recipe.candidate_source_mode
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback ledger differs from requested/selected registrations",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )
            attempted = decision.attempted_operator_ids
            if (
                not attempted
                or attempted[0] != decision.requested_operator_id
                or decision.selected_operator_id in attempted
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback ledger must prove the requested failure before selection",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )
            attempted_registrations = []
            for attempted_operator_id in attempted:
                try:
                    attempted_registration = self._registry.registration(
                        attempted_operator_id
                    )
                except RuntimeError as error:
                    raise BundleDraftError(
                        "QUOTA_FALLBACK_UNDECLARED",
                        "fallback attempt history contains an unknown operator",
                        origin_id=variant.origin_id,
                        policy=variant.recipe.policy,
                        evidence={"operator_id": attempted_operator_id},
                    ) from error
                if not (
                    attempted_registration.task_family is registration.task_family
                    and attempted_registration.subtask is variant.normalized_subtask
                    and variant.recipe.policy
                    in attempted_registration.spec.supported_policies
                    and variant.recipe.candidate_source_mode
                    in attempted_registration.spec.supported_sources
                    and variant.recipe.target_node_id
                    in attempted_registration.spec.root_fields
                ):
                    raise BundleDraftError(
                        "QUOTA_FALLBACK_UNDECLARED",
                        "fallback attempt changes task, policy, source, or frozen target",
                        origin_id=variant.origin_id,
                        policy=variant.recipe.policy,
                        evidence={"operator_id": attempted_operator_id},
                    )
                attempted_registrations.append(attempted_registration)
            if decision.target_change_required:
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback changes the target frozen by H/N matching",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )
            requested_families = self._operators_config.quota_bucket_mappings.get(
                decision.quota_bucket
            )
            if (
                requested_families is None
                or decision.requested_operator_family not in requested_families
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback quota bucket does not map to the requested family",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )
            if any(
                item.operator_family != decision.requested_operator_family
                and item.operator_family not in requested_families
                for item in attempted_registrations
            ):
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "fallback attempt is outside the requested quota phenotype",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )
            cross_family = (
                decision.requested_operator_family != decision.selected_operator_family
            )
            bucket_change = decision.quota_bucket != variant.quota_bucket
            if (cross_family or bucket_change) != decision.quota_deviation:
                raise BundleDraftError(
                    "QUOTA_FALLBACK_UNDECLARED",
                    "family/bucket fallback must exactly declare quota deviation",
                    origin_id=variant.origin_id,
                    policy=variant.recipe.policy,
                )

    def _eligible_variants(
        self,
        variants: tuple[PreparedHallucinatedVariant, ...],
        *,
        quota_bucket: str,
        include_cross_family: bool,
        global_seed: int,
        seed_namespace: str,
        normalized_subtask: EditingSubtask,
        policy: PropagationPolicy,
    ) -> tuple[PreparedHallucinatedVariant, ...]:
        eligible = tuple(
            item
            for item in variants
            if item.quota_bucket == quota_bucket
            and (
                item.fallback_decision is None
                or include_cross_family
                or not item.fallback_decision.quota_deviation
            )
        )
        return tuple(
            sorted(
                eligible,
                key=lambda item: (
                    item.fallback_rank,
                    _stable_digest(
                        seed_namespace,
                        global_seed,
                        normalized_subtask,
                        policy,
                        item.origin_id,
                        quota_bucket,
                        item.selection_fingerprint,
                    ),
                    item.stable_key,
                ),
            )
        )

    def _match_policy(
        self,
        *,
        origins: tuple[str, ...],
        policy: PropagationPolicy,
        variants_by_origin: Mapping[str, tuple[PreparedHallucinatedVariant, ...]],
        quota_items: tuple[Any, ...],
        include_cross_family: bool,
        global_seed: int,
        seed_namespace: str,
        normalized_subtask: EditingSubtask,
    ) -> tuple[
        dict[str, tuple[str, PreparedHallucinatedVariant]],
        tuple[QuotaBucketCount, ...],
        tuple[QuotaBackfillRequirement, ...],
    ]:
        slots = tuple(
            sorted(
                (
                    (item.family, slot_index)
                    for item in quota_items
                    for slot_index in range(item.target_per_50)
                ),
                key=lambda slot: (
                    _stable_digest(
                        seed_namespace,
                        global_seed,
                        normalized_subtask,
                        policy,
                        "quota_slot",
                        slot,
                    ),
                    slot,
                ),
            )
        )
        owner_by_slot: dict[tuple[str, int], str] = {}
        slot_by_origin: dict[str, tuple[str, int]] = {}

        def compatible_slots(origin_id: str) -> tuple[tuple[str, int], ...]:
            candidates = variants_by_origin.get(origin_id, ())
            eligible_buckets = {
                item.family
                for item in quota_items
                if self._eligible_variants(
                    candidates,
                    quota_bucket=item.family,
                    include_cross_family=include_cross_family,
                    global_seed=global_seed,
                    seed_namespace=seed_namespace,
                    normalized_subtask=normalized_subtask,
                    policy=policy,
                )
            }
            return tuple(slot for slot in slots if slot[0] in eligible_buckets)

        edges = {origin_id: compatible_slots(origin_id) for origin_id in origins}

        def augment(origin_id: str, visited: set[tuple[str, int]]) -> bool:
            for slot in edges[origin_id]:
                if slot in visited:
                    continue
                visited.add(slot)
                prior = owner_by_slot.get(slot)
                if prior is None or augment(prior, visited):
                    owner_by_slot[slot] = origin_id
                    slot_by_origin[origin_id] = slot
                    return True
            return False

        ordered_origins = tuple(
            sorted(
                origins,
                key=lambda origin_id: (
                    _stable_digest(
                        seed_namespace,
                        global_seed,
                        normalized_subtask,
                        policy,
                        origin_id,
                    ),
                    origin_id,
                ),
            )
        )
        for origin_id in ordered_origins:
            augment(origin_id, set())

        selected: dict[str, tuple[str, PreparedHallucinatedVariant]] = {}
        for origin_id, slot in sorted(slot_by_origin.items()):
            bucket = slot[0]
            candidates = self._eligible_variants(
                variants_by_origin[origin_id],
                quota_bucket=bucket,
                include_cross_family=include_cross_family,
                global_seed=global_seed,
                seed_namespace=seed_namespace,
                normalized_subtask=normalized_subtask,
                policy=policy,
            )
            if candidates:
                selected[origin_id] = (bucket, candidates[0])

        actual = Counter(bucket for bucket, _ in slot_by_origin.values())
        counts = tuple(
            QuotaBucketCount(
                policy=policy,
                quota_bucket=item.family,
                target_count=item.target_per_50,
                actual_count=actual[item.family],
            )
            for item in quota_items
        )
        unassigned = tuple(sorted(set(origins) - set(selected)))
        backfills = tuple(
            QuotaBackfillRequirement(
                policy=policy,
                quota_bucket=item.family,
                missing_count=item.target_per_50 - actual[item.family],
                unassigned_origin_ids=unassigned,
            )
            for item in quota_items
            if actual[item.family] < item.target_per_50
        )
        return selected, counts, backfills

    def schedule(self, request: QuotaScheduleRequest) -> QuotaSchedule:
        if type(request) is not QuotaScheduleRequest:
            raise TypeError("schedule requires QuotaScheduleRequest")
        for variant in request.variants:
            self._validate_variant(variant)
        variants_by_policy_origin: dict[
            tuple[PropagationPolicy, str], list[PreparedHallucinatedVariant]
        ] = defaultdict(list)
        for variant in request.variants:
            variants_by_policy_origin[
                (variant.recipe.policy, variant.origin_id)
            ].append(variant)

        assignments: list[QuotaAssignment] = []
        deviations: list[QuotaDeviationEntry] = []
        counts: list[QuotaBucketCount] = []
        backfills: list[QuotaBackfillRequirement] = []
        for policy in _POLICY_ORDER:
            quota_items = tuple(
                self._operators_config.quotas_per_subtask_policy[policy.dataset_name]
            )
            if sum(item.target_per_50 for item in quota_items) != 50:
                raise BundleDraftError(
                    "QUOTA_CONFIG_MISMATCH",
                    "frozen policy quota targets do not sum to 50",
                    policy=policy,
                )
            by_origin = {
                origin_id: tuple(
                    sorted(
                        variants_by_policy_origin.get((policy, origin_id), ()),
                        key=lambda item: item.stable_key,
                    )
                )
                for origin_id in request.origin_ids
            }
            selected, policy_counts, policy_backfills = self._match_policy(
                origins=request.origin_ids,
                policy=policy,
                variants_by_origin=by_origin,
                quota_items=quota_items,
                include_cross_family=False,
                global_seed=request.global_seed,
                seed_namespace=request.seed_namespace,
                normalized_subtask=request.normalized_subtask,
            )
            if policy_backfills and request.allow_quota_deviation:
                selected, policy_counts, policy_backfills = self._match_policy(
                    origins=request.origin_ids,
                    policy=policy,
                    variants_by_origin=by_origin,
                    quota_items=quota_items,
                    include_cross_family=True,
                    global_seed=request.global_seed,
                    seed_namespace=request.seed_namespace,
                    normalized_subtask=request.normalized_subtask,
                )
            counts.extend(policy_counts)
            backfills.extend(policy_backfills)
            for origin_id, (bucket, variant) in sorted(selected.items()):
                assignment = QuotaAssignment(origin_id, policy, bucket, variant)
                assignments.append(assignment)
                decision = variant.fallback_decision
                if decision is not None:
                    deviations.append(
                        QuotaDeviationEntry(
                            origin_id=origin_id,
                            policy=policy,
                            quota_bucket=decision.quota_bucket,
                            selected_quota_bucket=bucket,
                            requested_operator_id=decision.requested_operator_id,
                            selected_operator_id=decision.selected_operator_id,
                            requested_operator_family=(
                                decision.requested_operator_family
                            ),
                            selected_operator_family=(
                                decision.selected_operator_family
                            ),
                            attempted_operator_ids=(decision.attempted_operator_ids),
                            candidate_source=decision.candidate_source,
                            quota_deviation=decision.quota_deviation,
                            target_change_required=decision.target_change_required,
                        )
                    )

        assignments.sort(key=lambda item: (item.origin_id, _POLICY_INDEX[item.policy]))
        deviations.sort(
            key=lambda item: (
                item.origin_id,
                _POLICY_INDEX[item.policy],
                item.quota_bucket,
                item.selected_operator_id,
            )
        )
        report = QuotaScheduleReport(
            normalized_subtask=request.normalized_subtask,
            origin_count=len(request.origin_ids),
            counts=tuple(counts),
            deviations=tuple(deviations),
            backfills=tuple(backfills),
        )
        if backfills:
            raise QuotaScheduleError(report)
        return QuotaSchedule(tuple(assignments), tuple(deviations), report)


__all__ = [
    "BundleDraftError",
    "DeterministicQuotaScheduler",
    "MatchedBundleBuildRequest",
    "MatchedBundleBuilder",
    "MatchedBundleDraft",
    "MatchedDraftRecord",
    "PreparedHallucinatedVariant",
    "QuotaAssignment",
    "QuotaBackfillRequirement",
    "QuotaBucketCount",
    "QuotaDeviationEntry",
    "QuotaSchedule",
    "QuotaScheduleError",
    "QuotaScheduleReport",
    "QuotaScheduleRequest",
]
