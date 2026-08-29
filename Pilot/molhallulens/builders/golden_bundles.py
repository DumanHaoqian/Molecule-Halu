"""Deterministic T025 golden bundles built through the production Phase-2 path.

This module deliberately freezes semantic bundle drafts, not the natural-language,
character-span, or token-label artifacts owned by T039--T044.  Every hallucinated
state below comes from a T019/T020/T021 candidate engine and T022 propagation;
the T024 builder then creates its pair-specific faithful control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.candidates import replay_edit_action_from_source
from molhallulens.chemistry import isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.config.loader import ConfigBundle
from molhallulens.domain import (
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    EditingSubtask,
    EditTruth,
    FrozenMap,
    MutationTargetKind,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    TaskRecord,
    VariantLabel,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    DeletionPerturbator,
    PerturbationContext,
    SubstitutionPerturbator,
    task_record_from_joined_input,
)
from molhallulens.perturbators.base import CandidateEngine, PropagationOutcome
from molhallulens.perturbators.editing.addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateEngine,
)
from molhallulens.perturbators.editing.deletion import (
    DELETION_OPERATOR_IDS,
    DeletionCandidateEngine,
)
from molhallulens.perturbators.editing.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
    SubstitutionCandidateEngine,
)
from molhallulens.perturbators.registry import PerturbatorRegistry
from molhallulens.propagation import EditingPropagationEngine, PropagationPlan

from .bundles import (
    MatchedBundleBuilder,
    MatchedBundleBuildRequest,
    MatchedBundleDraft,
    PreparedHallucinatedVariant,
    QuotaAssignment,
)
from .edit_truth import derive_edit_truth
from .reference_dag import build_reference_dag

GOLDEN_BUNDLE_FORMAT_VERSION = "t025_deterministic_golden_bundles_v1"
GOLDEN_VALIDATION_FORMAT_VERSION = "t025_golden_bundle_validation_v1"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "Dataset"
DEFAULT_GOLDEN_BUNDLE_PATH = Path("tests/golden/t025_deterministic_bundles.json")
DEFAULT_GOLDEN_VALIDATION_PATH = Path(
    "Dataset/reports/t025_golden_bundle_validation.json"
)

_POLICIES = (
    PropagationPolicy.STOP,
    PropagationPolicy.PARTIAL,
    PropagationPolicy.FULL_CF,
    PropagationPolicy.TERMINAL,
)


class GoldenBundleBuildError(RuntimeError):
    """Structured failure for a frozen T025 case that no longer replays."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        origin_id: str | None = None,
        policy: PropagationPolicy | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("detail must be non-empty text")
        if origin_id is not None and (type(origin_id) is not str or not origin_id):
            raise ValueError("origin_id must be non-empty text or None")
        if policy is not None and type(policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy or None")
        self.code = code
        self.detail = detail
        self.origin_id = origin_id
        self.policy = policy
        location = ""
        if origin_id is not None:
            location += f" origin={origin_id!r}"
        if policy is not None:
            location += f" policy={policy.dataset_name}"
        super().__init__(f"{code}{location}: {detail}")


@dataclass(frozen=True, slots=True)
class GoldenPolicySpec:
    """One frozen production recipe within a golden origin bundle."""

    policy: PropagationPolicy
    operator_id: str
    target_node_id: str
    quota_bucket: str
    partial_cut_nodes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        for value, name in (
            (self.operator_id, "operator_id"),
            (self.target_node_id, "target_node_id"),
            (self.quota_bucket, "quota_bucket"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        cuts = frozenset(self.partial_cut_nodes)
        if any(type(node_id) is not str or not node_id for node_id in cuts):
            raise TypeError("partial_cut_nodes must contain non-empty strings")
        if self.policy is PropagationPolicy.PARTIAL and not cuts:
            raise ValueError("PARTIAL golden recipes require cut nodes")
        if self.policy is not PropagationPolicy.PARTIAL and cuts:
            raise ValueError("only PARTIAL golden recipes may carry cut nodes")
        object.__setattr__(self, "partial_cut_nodes", cuts)


@dataclass(frozen=True, slots=True)
class GoldenOriginSpec:
    """One exact real Pilot origin and its four deterministic recipes."""

    normalized_subtask: EditingSubtask
    origin_id: str
    policies: tuple[GoldenPolicySpec, ...]

    def __post_init__(self) -> None:
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if type(self.origin_id) is not str or not self.origin_id:
            raise ValueError("origin_id must be non-empty text")
        policies = tuple(self.policies)
        if len(policies) != 4 or tuple(item.policy for item in policies) != _POLICIES:
            raise ValueError(
                "golden origin requires stable STOP/PARTIAL/FULL/TERMINAL specs"
            )
        object.__setattr__(self, "policies", policies)


T025_GOLDEN_ORIGINS = (
    GoldenOriginSpec(
        normalized_subtask=EditingSubtask.ADD,
        origin_id="mol_edit.add_v2.0022",
        policies=(
            GoldenPolicySpec(
                PropagationPolicy.STOP,
                ADDITION_OPERATOR_IDS[7],
                "product_heavy",
                "heavy_ring_count_claim",
            ),
            GoldenPolicySpec(
                PropagationPolicy.PARTIAL,
                ADDITION_OPERATOR_IDS[2],
                "add_fragment",
                "entity_partial_propagation",
                frozenset({"product"}),
            ),
            GoldenPolicySpec(
                PropagationPolicy.FULL_CF,
                ADDITION_OPERATOR_IDS[3],
                "product",
                "wrong_attachment_atom_bond",
            ),
            GoldenPolicySpec(
                PropagationPolicy.TERMINAL,
                ADDITION_OPERATOR_IDS[10],
                "final_answer",
                "terminal_valid_high_similarity",
            ),
        ),
    ),
    GoldenOriginSpec(
        normalized_subtask=EditingSubtask.DELETE,
        origin_id="mol_edit.delete_v2.0016",
        policies=(
            GoldenPolicySpec(
                PropagationPolicy.STOP,
                DELETION_OPERATOR_IDS[9],
                "product_heavy",
                "heavy_ring_count_claim",
            ),
            GoldenPolicySpec(
                PropagationPolicy.PARTIAL,
                DELETION_OPERATOR_IDS[1],
                "product",
                "product_dependency_cross_step",
                frozenset({"product_heavy", "product_rings"}),
            ),
            GoldenPolicySpec(
                PropagationPolicy.FULL_CF,
                DELETION_OPERATOR_IDS[7],
                "product",
                "valid_wrong_group_fragment",
            ),
            GoldenPolicySpec(
                PropagationPolicy.TERMINAL,
                DELETION_OPERATOR_IDS[11],
                "final_answer",
                "terminal_valid_high_similarity",
            ),
        ),
    ),
    GoldenOriginSpec(
        normalized_subtask=EditingSubtask.SUBSTITUTE,
        origin_id="mol_edit.substitute_v2.0029",
        policies=(
            GoldenPolicySpec(
                PropagationPolicy.STOP,
                SUBSTITUTION_OPERATOR_IDS[9],
                "product_heavy",
                "heavy_ring_count_claim",
            ),
            GoldenPolicySpec(
                PropagationPolicy.PARTIAL,
                SUBSTITUTION_OPERATOR_IDS[2],
                "add_fragment",
                "entity_partial_propagation",
                frozenset({"product"}),
            ),
            GoldenPolicySpec(
                PropagationPolicy.FULL_CF,
                SUBSTITUTION_OPERATOR_IDS[3],
                "product",
                "wrong_attachment_atom_bond",
            ),
            GoldenPolicySpec(
                PropagationPolicy.TERMINAL,
                SUBSTITUTION_OPERATOR_IDS[11],
                "final_answer",
                "terminal_valid_high_similarity",
            ),
        ),
    ),
)


def _stable_value(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(item) for item in value), key=repr)
    raise TypeError(f"unsupported golden JSON value: {type(value).__qualname__}")


def _derived_seed(
    global_seed: int,
    dataset_version: str,
    spec: GoldenOriginSpec,
    policy: GoldenPolicySpec,
    *,
    variant_index: int,
) -> int:
    payload = "\0".join(
        (
            str(global_seed),
            dataset_version,
            spec.origin_id,
            policy.operator_id,
            policy.policy.dataset_name,
            str(variant_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _strict_molecule(smiles: Any, *, field: str) -> Chem.Mol:
    if type(smiles) is not str or not smiles:
        raise GoldenBundleBuildError(
            "GOLDEN_CHEMISTRY_INVALID", f"{field} must be non-empty SMILES"
        )
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise GoldenBundleBuildError(
            "GOLDEN_CHEMISTRY_INVALID", f"{field} failed strict RDKit parsing"
        )
    return molecule


def _equivalent(left: Any, right: Any, *, detail: str) -> bool:
    _strict_molecule(left, field=f"{detail}.left")
    _strict_molecule(right, field=f"{detail}.right")
    try:
        return isomeric_graph_equivalent(left, right)
    except (RuntimeError, TypeError, ValueError) as error:
        raise GoldenBundleBuildError(
            "GOLDEN_CHEMISTRY_INVALID", f"{detail} graph comparison failed"
        ) from error


def _morgan_similarity(left: Any, right: Any, *, detail: str) -> float:
    left_molecule = _strict_molecule(left, field=f"{detail}.left")
    right_molecule = _strict_molecule(right, field=f"{detail}.right")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(left_molecule),
            generator.GetFingerprint(right_molecule),
        )
    )


def _terminal_similarity_evidence(
    selected_patch: CandidatePatch,
    pool: CandidatePool,
) -> tuple[float, float]:
    reference = selected_patch.old_value.normalized_value
    similarities = tuple(
        _morgan_similarity(
            reference,
            candidate.new_value.normalized_value,
            detail="TERMINAL.candidate_similarity",
        )
        for candidate in pool.candidates
    )
    if not similarities:
        raise GoldenBundleBuildError(
            "GOLDEN_QUOTA_MISMATCH",
            "TERMINAL candidate pool must be non-empty",
            policy=PropagationPolicy.TERMINAL,
        )
    selected_index = pool.candidates.index(selected_patch)
    return similarities[selected_index], max(similarities)


def _exact_quota_bucket(
    *,
    policy: PropagationPolicy,
    operator_family: str,
    root_node_id: str,
    selected_patch: CandidatePatch,
    pool: CandidatePool,
) -> str:
    """Derive the exact frozen bucket from the selected phenotype, not family alone."""

    if policy is PropagationPolicy.STOP:
        by_family = {
            "wrong_anchor_site": "anchor_site_grounding",
            "wrong_fragment_group": "group_fragment_identity",
            "attachment_bond_edit": "attachment_internal_relation",
            "numeric_count_claim": "heavy_ring_count_claim",
            "nl_formal_internal_relation": "attachment_internal_relation",
        }
    elif policy is PropagationPolicy.PARTIAL:
        if operator_family == "numeric_count_claim":
            return "count_ring_propagation"
        if operator_family == "nl_formal_internal_relation":
            return "nl_formal_internal_relation"
        if operator_family in {
            "wrong_anchor_site",
            "wrong_fragment_group",
            "attachment_bond_edit",
        }:
            return (
                "product_dependency_cross_step"
                if root_node_id == "product"
                else "entity_partial_propagation"
            )
        by_family = {}
    elif policy is PropagationPolicy.FULL_CF:
        by_family = {
            "wrong_anchor_site": "valid_wrong_site_occurrence_regioisomer",
            "wrong_fragment_group": "valid_wrong_group_fragment",
            "attachment_bond_edit": "wrong_attachment_atom_bond",
        }
    else:
        if operator_family != "final_answer_identity":
            raise GoldenBundleBuildError(
                "GOLDEN_QUOTA_MISMATCH",
                "TERMINAL golden candidate must use final_answer_identity",
                policy=policy,
            )
        if not pool.candidates or selected_patch != pool.candidates[0]:
            raise GoldenBundleBuildError(
                "GOLDEN_QUOTA_MISMATCH",
                "high-similarity TERMINAL candidate must be the production top rank",
                policy=policy,
            )
        before = selected_patch.old_value.normalized_value
        after = selected_patch.new_value.normalized_value
        if _equivalent(before, after, detail="TERMINAL.top_ranked_identity"):
            raise GoldenBundleBuildError(
                "GOLDEN_QUOTA_MISMATCH",
                "high-similarity TERMINAL candidate must be valid and non-equivalent",
                policy=policy,
            )
        selected_similarity, pool_max_similarity = _terminal_similarity_evidence(
            selected_patch,
            pool,
        )
        if selected_similarity != pool_max_similarity:
            raise GoldenBundleBuildError(
                "GOLDEN_QUOTA_MISMATCH",
                "TERMINAL high-similarity candidate is not the pool similarity maximum",
                policy=policy,
            )
        return "terminal_valid_high_similarity"
    try:
        return by_family[operator_family]
    except KeyError as error:
        raise GoldenBundleBuildError(
            "GOLDEN_QUOTA_MISMATCH",
            "selected operator phenotype has no exact frozen quota bucket",
            policy=policy,
        ) from error


class _UnusedTraceRenderer:
    def render(self, *_args: Any) -> Any:
        raise AssertionError("T025 renders only after the locked T022 state exists")


class _UnusedValidatorChain:
    def validate_reference(self, *_args: Any) -> Any:
        raise AssertionError("T025 uses the already-strict T015 reference boundary")

    def validate_artifact(self, *_args: Any) -> Any:
        raise AssertionError("final artifact validation belongs to T043")


class _UnusedLabelProjector:
    def project(self, *_args: Any) -> Any:
        raise AssertionError("token projection belongs to T042")


_PERTURBATOR_TYPES = {
    EditingSubtask.ADD: AdditionPerturbator,
    EditingSubtask.DELETE: DeletionPerturbator,
    EditingSubtask.SUBSTITUTE: SubstitutionPerturbator,
}
_ENGINE_TYPES = {
    EditingSubtask.ADD: AdditionCandidateEngine,
    EditingSubtask.DELETE: DeletionCandidateEngine,
    EditingSubtask.SUBSTITUTE: SubstitutionCandidateEngine,
}


@dataclass(frozen=True, slots=True)
class GoldenPolicyExecution:
    """Audit carrier proving one H state traversed candidate and propagation APIs."""

    context: PerturbationContext[EditTruth]
    pool: CandidatePool
    selected_patch: CandidatePatch
    plan: PropagationPlan
    outcome: PropagationOutcome
    candidate_engine_name: str
    propagation_engine_name: str
    operator_family: str
    quota_bucket: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.candidate_engine_name, "candidate_engine_name"),
            (self.propagation_engine_name, "propagation_engine_name"),
            (self.operator_family, "operator_family"),
            (self.quota_bucket, "quota_bucket"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")

    def to_trace_dict(self) -> dict[str, Any]:
        action = self.selected_patch.edit_action
        terminal_similarity: tuple[float, float] | None = None
        if self.context.recipe.policy is PropagationPolicy.TERMINAL:
            terminal_similarity = _terminal_similarity_evidence(
                self.selected_patch,
                self.pool,
            )
        return {
            "policy": self.context.recipe.policy.dataset_name,
            "recipe_id": self.context.recipe.recipe_id,
            "derived_seed": self.context.recipe.derived_seed,
            "operator_id": self.context.recipe.operator_id,
            "operator_family": self.operator_family,
            "quota_bucket": self.quota_bucket,
            "candidate_engine": self.candidate_engine_name,
            "propagation_engine": self.propagation_engine_name,
            "selected_from_pool": True,
            "candidate_pool": {
                "request_id": self.pool.request_id,
                "accepted_count": len(self.pool.candidates),
                "rejection_codes": list(self.pool.rejection_codes),
                "selected_rank": self.pool.candidates.index(self.selected_patch),
                "selected_answer_similarity": (
                    None if terminal_similarity is None else terminal_similarity[0]
                ),
                "max_answer_similarity": (
                    None if terminal_similarity is None else terminal_similarity[1]
                ),
            },
            "selected_root_patch": {
                "candidate_id": self.selected_patch.candidate_id,
                "candidate_source": self.selected_patch.source.value,
                "root_node_id": self.selected_patch.root_node_id,
                "before": _stable_value(self.selected_patch.old_value.normalized_value),
                "after": _stable_value(self.selected_patch.new_value.normalized_value),
                "edit_action": (
                    None
                    if action is None
                    else {
                        "edit_kind": action.edit_kind.value,
                        "source_anchor_index": action.source_anchor_index,
                        "remove_anchor_index": action.remove_anchor_index,
                        "remove_fragment_smiles": action.remove_fragment_smiles,
                        "add_fragment_smiles": action.add_fragment_smiles,
                        "fragment_attachment_atom": action.fragment_attachment_atom,
                        "bond_type": (
                            None if action.bond_type is None else action.bond_type.value
                        ),
                        "metadata": _stable_value(action.metadata),
                    }
                ),
            },
            "propagation_plan": {
                "full_closure": list(self.plan.full_closure),
                "selected_nodes": list(self.plan.selected_nodes),
            },
            "mutation_events": [
                {
                    "event_id": event.event_id,
                    "target_kind": event.target_kind.value,
                    "target_id": event.node_or_edge_id,
                    "before": _stable_value(event.before.normalized_value),
                    "after": _stable_value(event.after.normalized_value),
                    "causal_role": event.causal_role.value,
                    "hallucination_types": sorted(
                        label.value for label in event.hallucination_types
                    ),
                    "edit_subtypes": sorted(
                        label.value for label in event.edit_subtypes
                    ),
                    "operator_id": event.operator_id,
                    "root_event_id": event.root_event_id,
                }
                for event in self.outcome.graph_delta.events
            ],
        }


def _validation_sections_pass(
    validation: Mapping[str, Any],
    spec: GoldenOriginSpec,
) -> bool:
    if (
        validation.get("all_pass") is not True
        or validation.get("origin_id") != spec.origin_id
        or validation.get("normalized_subtask") != spec.normalized_subtask.value
    ):
        return False
    chemistry = validation.get("chemistry")
    propagation = validation.get("propagation")
    bundle = validation.get("bundle")
    if not all(isinstance(item, Mapping) for item in (chemistry, propagation, bundle)):
        return False
    assert isinstance(chemistry, Mapping)
    assert isinstance(propagation, Mapping)
    assert isinstance(bundle, Mapping)
    if any(
        section.get("all_pass") is not True
        for section in (chemistry, propagation, bundle)
    ):
        return False
    expected_policies = tuple(policy.dataset_name for policy in _POLICIES)
    checks_by_section: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for section_name, section in (
        ("chemistry", chemistry),
        ("propagation", propagation),
    ):
        checks = section.get("checks")
        if (
            isinstance(checks, (str, bytes))
            or not isinstance(checks, Sequence)
            or len(checks) != 4
            or any(not isinstance(item, Mapping) for item in checks)
        ):
            return False
        frozen_checks = tuple(checks)
        if tuple(item.get("policy") for item in frozen_checks) != expected_policies:
            return False
        checks_by_section[section_name] = frozen_checks
    chemistry_checks = checks_by_section["chemistry"]
    for check in chemistry_checks:
        if (
            check.get("strict_product_parse") is not True
            or check.get("strict_answer_parse") is not True
            or check.get("selected_from_validated_candidate_pool") is not True
            or check.get("selected_candidate_rank") != 0
            or type(check.get("selected_candidate_id")) is not str
            or not check.get("selected_candidate_id")
            or check.get("candidate_source") not in {"RULE", "RDKIT", "HYBRID"}
        ):
            return False
        if check.get("policy") == PropagationPolicy.FULL_CF.dataset_name and (
            check.get("product_matches_reference") is not False
            or check.get("answer_matches_product") is not True
            or check.get("structural_edit_action_present") is not True
        ):
            return False
        if check.get("policy") == PropagationPolicy.TERMINAL.dataset_name:
            selected_similarity = check.get("selected_answer_similarity")
            if (
                check.get("product_matches_reference") is not True
                or check.get("answer_matches_product") is not False
                or type(selected_similarity) is not float
                or selected_similarity <= 0.0
                or selected_similarity != check.get("pool_max_answer_similarity")
            ):
                return False
    for check in checks_by_section["propagation"]:
        selected_nodes = check.get("selected_nodes")
        difference_nodes = check.get("semantic_difference_nodes")
        if (
            check.get("graph_delta_exact") is not True
            or check.get("exact_quota_bucket_verified") is not True
            or check.get("policy_shape_valid") is not True
            or check.get("selected_root_patch_bound_to_state") is not True
            or type(check.get("root_node_id")) is not str
            or not check.get("root_node_id")
            or isinstance(selected_nodes, (str, bytes))
            or not isinstance(selected_nodes, Sequence)
            or not selected_nodes
            or selected_nodes[0] != check.get("root_node_id")
            or isinstance(difference_nodes, (str, bytes))
            or not isinstance(difference_nodes, Sequence)
            or not difference_nodes
        ):
            return False
        if (
            check.get("policy") == PropagationPolicy.FULL_CF.dataset_name
            and check.get("selected_action_replay_bound_to_product") is not True
        ):
            return False
    return (
        bundle.get("record_count") == 8
        and bundle.get("hallucinated_count") == 4
        and bundle.get("faithful_count") == 4
        and bundle.get("unique_record_ids") == 8
        and all(
            bundle.get(key) is True
            for key in (
                "reciprocal_pairs",
                "faithful_controls_exact",
                "unique_control_identities",
                "unique_render_identities",
                "hallucinated_states_bound_to_propagation",
            )
        )
    )


@dataclass(frozen=True, slots=True)
class GoldenOriginBundle:
    spec: GoldenOriginSpec
    bundle: MatchedBundleDraft
    executions: tuple[GoldenPolicyExecution, ...]
    validation: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            len(self.executions) != 4
            or tuple(item.context.recipe.policy for item in self.executions)
            != _POLICIES
        ):
            raise ValueError("golden origin must retain four stable executions")
        if self.bundle.origin_id != self.spec.origin_id:
            raise ValueError("golden bundle origin must match its frozen spec")
        operators_config = load_config_bundle().operators
        perturbator_type = _PERTURBATOR_TYPES[self.spec.normalized_subtask]
        registry = PerturbatorRegistry.from_perturbator_types(
            (perturbator_type,),
            operators_config=operators_config,
        )
        expected_candidate_engine = _ENGINE_TYPES[self.spec.normalized_subtask].__name__
        for execution, policy_spec in zip(
            self.executions,
            self.spec.policies,
            strict=True,
        ):
            recipe = execution.context.recipe
            registration = registry.registration(policy_spec.operator_id)
            if (
                execution.context.record.origin_id != self.spec.origin_id
                or execution.context.record.normalized_subtask
                is not self.spec.normalized_subtask
                or recipe.origin_id != self.spec.origin_id
                or recipe.policy is not policy_spec.policy
                or recipe.operator_id != policy_spec.operator_id
                or recipe.target_node_id != policy_spec.target_node_id
                or recipe.partial_cut_nodes != policy_spec.partial_cut_nodes
                or recipe.variant_index != 0
                or execution.selected_patch.root_node_id != policy_spec.target_node_id
                or execution.quota_bucket != policy_spec.quota_bucket
                or execution.operator_family != registration.operator_family
                or execution.candidate_engine_name != expected_candidate_engine
                or execution.propagation_engine_name
                != EditingPropagationEngine.__name__
                or execution.selected_patch.source is not recipe.candidate_source_mode
                or execution.plan.root_node_id != policy_spec.target_node_id
                or execution.outcome.candidate_graph.schema
                != execution.context.state_schema
            ):
                raise ValueError(
                    "golden execution recipe must match its exact frozen policy spec"
                )
        derived_validation = _validate_origin(
            self.spec,
            self.bundle,
            self.executions,
        )
        if not _validation_sections_pass(derived_validation, self.spec):
            raise ValueError("validation sections must all pass exact T025 gates")
        object.__setattr__(self, "validation", FrozenMap(derived_validation))

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "normalized_subtask": self.spec.normalized_subtask.value,
            "origin_id": self.spec.origin_id,
            "candidate_and_propagation_trace": [
                execution.to_trace_dict() for execution in self.executions
            ],
            "bundle": self.bundle.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GoldenCorpusBuild:
    dataset_version: str
    global_seed: int
    origins: tuple[GoldenOriginBundle, ...]

    def __post_init__(self) -> None:
        if type(self.dataset_version) is not str or not self.dataset_version:
            raise ValueError("dataset_version must be non-empty text")
        if type(self.global_seed) is not int or self.global_seed < 0:
            raise ValueError("global_seed must be non-negative")
        origins = tuple(self.origins)
        if len(origins) != 3:
            raise ValueError("T025 corpus requires exactly three origin bundles")
        if tuple(item.spec.normalized_subtask for item in origins) != tuple(
            EditingSubtask
        ):
            raise ValueError("T025 corpus must use ADD/DELETE/SUBSTITUTE order")
        canonical_dataset = load_config_bundle().dataset.dataset
        if (
            self.dataset_version != canonical_dataset.version_name
            or self.global_seed != canonical_dataset.global_seed
        ):
            raise ValueError("T025 corpus metadata must equal the frozen config")
        for origin in origins:
            for execution, policy_spec in zip(
                origin.executions,
                origin.spec.policies,
                strict=True,
            ):
                recipe = execution.context.recipe
                expected_seed = _derived_seed(
                    self.global_seed,
                    self.dataset_version,
                    origin.spec,
                    policy_spec,
                    variant_index=recipe.variant_index,
                )
                if recipe.derived_seed != expected_seed:
                    raise ValueError(
                        "corpus metadata must match every execution derived seed"
                    )
        object.__setattr__(self, "origins", origins)

    def bundle_artifact(self) -> dict[str, Any]:
        return {
            "format_version": GOLDEN_BUNDLE_FORMAT_VERSION,
            "dataset_version": self.dataset_version,
            "global_seed": self.global_seed,
            "phase_boundary": "semantic_matched_bundle_draft_before_t039_t040",
            "origin_bundles": [item.to_artifact_dict() for item in self.origins],
        }

    def validation_report(self) -> dict[str, Any]:
        records = tuple(
            record for origin in self.origins for record in origin.bundle.records
        )
        return {
            "format_version": GOLDEN_VALIDATION_FORMAT_VERSION,
            "dataset_version": self.dataset_version,
            "global_seed": self.global_seed,
            "all_pass": all(
                _validation_sections_pass(item.validation, item.spec)
                for item in self.origins
            ),
            "summary": {
                "origin_bundle_count": len(self.origins),
                "record_count": len(records),
                "hallucinated_record_count": sum(
                    record.variant_label is VariantLabel.HALLUCINATED
                    for record in records
                ),
                "faithful_record_count": sum(
                    record.variant_label is VariantLabel.FAITHFUL for record in records
                ),
                "candidate_engine_invocation_count": sum(
                    len(item.executions) for item in self.origins
                ),
                "propagation_engine_invocation_count": sum(
                    len(item.executions) for item in self.origins
                ),
            },
            "origins": [_stable_value(item.validation) for item in self.origins],
        }

    def render_bundle_json(self) -> str:
        return _render_json(self.bundle_artifact())

    def render_validation_json(self) -> str:
        return _render_json(self.validation_report())


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _recipe(
    origin: GoldenOriginSpec,
    policy: GoldenPolicySpec,
    *,
    dataset_version: str,
    global_seed: int,
) -> PerturbationRecipe:
    variant_index = 0
    return PerturbationRecipe(
        recipe_id=(
            f"t025:{origin.origin_id}:{policy.policy.dataset_name}:{policy.operator_id}"
        ),
        origin_id=origin.origin_id,
        operator_id=policy.operator_id,
        policy=policy.policy,
        target_node_id=policy.target_node_id,
        candidate_source_mode=CandidateSourceType.RULE,
        variant_index=variant_index,
        derived_seed=_derived_seed(
            global_seed,
            dataset_version,
            origin,
            policy,
            variant_index=variant_index,
        ),
        rewrite_budget=RewriteBudget(
            max_changed_claims=64,
            max_added_characters=256,
            length_bucket="t025-golden",
        ),
        candidate_difficulty_bucket="golden-hard",
        renderer_style_id="formal-v1",
        partial_cut_nodes=policy.partial_cut_nodes,
    )


def _records_by_id(
    records: Sequence[JoinedInputRecord],
) -> dict[str, JoinedInputRecord]:
    indexed: dict[str, JoinedInputRecord] = {}
    for record in records:
        if record.anonymous_sample_id in indexed:
            raise GoldenBundleBuildError(
                "GOLDEN_ORIGIN_DUPLICATE",
                f"duplicate joined origin {record.anonymous_sample_id!r}",
            )
        indexed[record.anonymous_sample_id] = record
    return indexed


def _build_execution(
    *,
    config: ConfigBundle,
    origin_spec: GoldenOriginSpec,
    policy_spec: GoldenPolicySpec,
    record: TaskRecord,
    reference_graph: StateDAG,
    truth: EditTruth,
    candidate_engine: CandidateEngine[EditTruth],
    propagator: EditingPropagationEngine,
    registry: PerturbatorRegistry,
    global_seed: int,
) -> tuple[GoldenPolicyExecution, QuotaAssignment]:
    recipe = _recipe(
        origin_spec,
        policy_spec,
        dataset_version=config.dataset.dataset.version_name,
        global_seed=global_seed,
    )
    context = PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=reference_graph.schema,
        reference_graph=reference_graph,
        truth=truth,
    )
    try:
        pool = candidate_engine.enumerate_root_patches(context)
        patch = candidate_engine.select_root_patch(context, pool)
    except (RuntimeError, TypeError, ValueError) as error:
        raise GoldenBundleBuildError(
            "GOLDEN_CANDIDATE_UNAVAILABLE",
            f"production candidate engine failed with {type(error).__name__}",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        ) from error
    if patch not in pool.candidates:
        raise GoldenBundleBuildError(
            "GOLDEN_CANDIDATE_UNAVAILABLE",
            "selected root patch is absent from its accepted candidate pool",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        )
    if patch.source is not recipe.candidate_source_mode:
        recipe = replace(recipe, candidate_source_mode=patch.source)
        context = replace(context, recipe=recipe)
    try:
        plan = propagator.plan(context, patch)
        outcome = propagator.propagate(context, patch)
    except (RuntimeError, TypeError, ValueError) as error:
        raise GoldenBundleBuildError(
            "GOLDEN_PROPAGATION_FAILED",
            f"production propagation failed with {type(error).__name__}",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        ) from error
    if not outcome.candidate_graph.value_for(patch.root_node_id).semantically_equals(
        patch.new_value
    ):
        raise GoldenBundleBuildError(
            "GOLDEN_PROPAGATION_FAILED",
            "propagated root state is not bound to the selected candidate patch",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        )
    if patch.edit_action is not None and "product" in plan.selected_nodes:
        try:
            replayed_products = replay_edit_action_from_source(
                record.indexed_smiles,
                patch.edit_action,
            )
            candidate_product = outcome.candidate_graph.value_for(
                "product"
            ).normalized_value
            replay_matches = len(replayed_products) == 1 and _equivalent(
                replayed_products[0],
                candidate_product,
                detail="selected_action.propagated_product",
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise GoldenBundleBuildError(
                "GOLDEN_PROPAGATION_FAILED",
                "selected edit action could not be independently replayed",
                origin_id=origin_spec.origin_id,
                policy=policy_spec.policy,
            ) from error
        if not replay_matches:
            raise GoldenBundleBuildError(
                "GOLDEN_PROPAGATION_FAILED",
                "selected edit action does not reproduce the propagated product",
                origin_id=origin_spec.origin_id,
                policy=policy_spec.policy,
            )
    registration = registry.registration(policy_spec.operator_id)
    compatible_families = config.operators.quota_bucket_mappings.get(
        policy_spec.quota_bucket,
        (),
    )
    if registration.operator_family not in compatible_families:
        raise GoldenBundleBuildError(
            "GOLDEN_QUOTA_MISMATCH",
            "frozen quota bucket is incompatible with the selected operator family",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        )
    exact_quota_bucket = _exact_quota_bucket(
        policy=policy_spec.policy,
        operator_family=registration.operator_family,
        root_node_id=plan.root_node_id,
        selected_patch=patch,
        pool=pool,
    )
    if policy_spec.quota_bucket != exact_quota_bucket:
        raise GoldenBundleBuildError(
            "GOLDEN_QUOTA_MISMATCH",
            "frozen quota bucket does not match the selected exact phenotype",
            origin_id=origin_spec.origin_id,
            policy=policy_spec.policy,
        )
    variant = PreparedHallucinatedVariant(
        origin_id=origin_spec.origin_id,
        normalized_subtask=origin_spec.normalized_subtask,
        input_view_id=f"t025:{origin_spec.origin_id}:source_instruction_v1",
        recipe=recipe,
        operator_family=registration.operator_family,
        quota_bucket=policy_spec.quota_bucket,
        renderer_backend="deterministic-formal-v1",
        reference_graph=reference_graph,
        candidate_graph=outcome.candidate_graph,
        graph_delta=outcome.graph_delta,
    )
    assignment = QuotaAssignment(
        origin_id=origin_spec.origin_id,
        policy=policy_spec.policy,
        quota_bucket=policy_spec.quota_bucket,
        variant=variant,
    )
    return (
        GoldenPolicyExecution(
            context=context,
            pool=pool,
            selected_patch=patch,
            plan=plan,
            outcome=outcome,
            candidate_engine_name=type(candidate_engine).__name__,
            propagation_engine_name=type(propagator).__name__,
            operator_family=registration.operator_family,
            quota_bucket=policy_spec.quota_bucket,
        ),
        assignment,
    )


def _validate_origin(
    origin: GoldenOriginSpec,
    bundle: MatchedBundleDraft,
    executions: tuple[GoldenPolicyExecution, ...],
) -> dict[str, Any]:
    chemistry_checks: list[dict[str, Any]] = []
    propagation_checks: list[dict[str, Any]] = []
    reference = executions[0].context.reference_graph
    reference_product = reference.value_for("product").normalized_value
    _strict_molecule(reference_product, field="reference.product")
    for execution in executions:
        policy = execution.context.recipe.policy
        state = execution.outcome.candidate_graph
        product = state.value_for("product").normalized_value
        answer = state.value_for("final_answer").normalized_value
        _strict_molecule(product, field=f"{policy.dataset_name}.product")
        _strict_molecule(answer, field=f"{policy.dataset_name}.final_answer")
        product_matches_reference = _equivalent(
            product,
            reference_product,
            detail=f"{policy.dataset_name}.product_reference",
        )
        answer_matches_product = _equivalent(
            answer,
            product,
            detail=f"{policy.dataset_name}.answer_product",
        )
        if policy is PropagationPolicy.FULL_CF and (
            product_matches_reference
            or not answer_matches_product
            or execution.selected_patch.edit_action is None
        ):
            raise GoldenBundleBuildError(
                "GOLDEN_CHEMISTRY_INVALID",
                "FULL_CF must carry an action-backed wrong product and matching Answer",
                origin_id=origin.origin_id,
                policy=policy,
            )
        if policy is PropagationPolicy.TERMINAL and (
            not product_matches_reference or answer_matches_product
        ):
            raise GoldenBundleBuildError(
                "GOLDEN_CHEMISTRY_INVALID",
                "TERMINAL must preserve product and mutate only Answer identity",
                origin_id=origin.origin_id,
                policy=policy,
            )
        terminal_similarity: tuple[float, float] | None = None
        if policy is PropagationPolicy.TERMINAL:
            terminal_similarity = _terminal_similarity_evidence(
                execution.selected_patch,
                execution.pool,
            )
        chemistry_checks.append(
            {
                "policy": policy.dataset_name,
                "strict_product_parse": True,
                "strict_answer_parse": True,
                "selected_from_validated_candidate_pool": True,
                "selected_candidate_rank": execution.pool.candidates.index(
                    execution.selected_patch
                ),
                "selected_answer_similarity": (
                    None if terminal_similarity is None else terminal_similarity[0]
                ),
                "pool_max_answer_similarity": (
                    None if terminal_similarity is None else terminal_similarity[1]
                ),
                "structural_edit_action_present": (
                    execution.selected_patch.edit_action is not None
                ),
                "selected_candidate_id": execution.selected_patch.candidate_id,
                "candidate_source": execution.selected_patch.source.value,
                "product_matches_reference": product_matches_reference,
                "answer_matches_product": answer_matches_product,
            }
        )

        differences = state.semantic_differences(reference)
        difference_nodes = {
            target_id
            for target_kind, target_id in differences
            if target_kind is MutationTargetKind.NODE
        }
        delta_nodes = {
            event.node_or_edge_id for event in execution.outcome.graph_delta.events
        }
        plan = execution.plan
        root_patch_bound = state.value_for(
            execution.selected_patch.root_node_id
        ).semantically_equals(execution.selected_patch.new_value)
        action_replay_bound: bool | None = None
        if (
            execution.selected_patch.edit_action is not None
            and "product" in plan.selected_nodes
        ):
            replayed_products = replay_edit_action_from_source(
                execution.context.record.indexed_smiles,
                execution.selected_patch.edit_action,
            )
            action_replay_bound = len(replayed_products) == 1 and _equivalent(
                replayed_products[0],
                product,
                detail=f"{policy.dataset_name}.action_product",
            )
        valid_shape = (
            difference_nodes == delta_nodes
            and plan.root_node_id in difference_nodes
            and root_patch_bound
            and action_replay_bound is not False
        )
        if policy is PropagationPolicy.STOP:
            valid_shape = valid_shape and difference_nodes == {plan.root_node_id}
        elif policy is PropagationPolicy.PARTIAL:
            valid_shape = (
                valid_shape
                and len(difference_nodes) > 1
                and set(plan.selected_nodes) < set(plan.full_closure)
                and difference_nodes <= set(plan.selected_nodes)
                and reference.schema.is_connected_downstream_subgraph(
                    {plan.root_node_id}, difference_nodes
                )
            )
        elif policy is PropagationPolicy.FULL_CF:
            valid_shape = (
                valid_shape
                and plan.selected_nodes == plan.full_closure
                and difference_nodes <= set(plan.selected_nodes)
                and action_replay_bound is True
            )
        else:
            valid_shape = valid_shape and difference_nodes == {"final_answer"}
        if not valid_shape:
            raise GoldenBundleBuildError(
                "GOLDEN_PROPAGATION_INVALID",
                "T022 result violates the frozen policy shape",
                origin_id=origin.origin_id,
                policy=policy,
            )
        propagation_checks.append(
            {
                "policy": policy.dataset_name,
                "root_node_id": plan.root_node_id,
                "full_closure": list(plan.full_closure),
                "selected_nodes": list(plan.selected_nodes),
                "semantic_difference_nodes": sorted(difference_nodes),
                "selected_root_patch_bound_to_state": root_patch_bound,
                "selected_action_replay_bound_to_product": action_replay_bound,
                "graph_delta_exact": True,
                "exact_quota_bucket_verified": True,
                "policy_shape_valid": True,
            }
        )

    records = bundle.records
    h_records = tuple(
        record
        for record in records
        if record.variant_label is VariantLabel.HALLUCINATED
    )
    n_records = tuple(
        record for record in records if record.variant_label is VariantLabel.FAITHFUL
    )
    by_id = {record.record_id: record for record in records}
    reciprocal = all(
        record.matched_record_id in by_id
        and by_id[record.matched_record_id].matched_record_id == record.record_id
        for record in records
    )
    faithful_controls = all(
        record.locked_state == record.reference_graph and not record.graph_delta.events
        for record in n_records
    )
    unique_controls = len({record.control_identity for record in n_records}) == 4
    unique_renders = len({record.render_identity for record in records}) == 8
    execution_by_policy = {
        execution.context.recipe.policy: execution for execution in executions
    }
    h_states_bound = all(
        record.locked_state
        == execution_by_policy[record.policy].outcome.candidate_graph
        for record in h_records
    )
    bundle_pass = (
        len(records) == 8
        and len(h_records) == 4
        and len(n_records) == 4
        and len(by_id) == 8
        and reciprocal
        and faithful_controls
        and unique_controls
        and unique_renders
        and h_states_bound
    )
    if not bundle_pass:
        raise GoldenBundleBuildError(
            "GOLDEN_BUNDLE_INVALID",
            "T024 matched bundle invariants failed",
            origin_id=origin.origin_id,
        )
    return {
        "origin_id": origin.origin_id,
        "normalized_subtask": origin.normalized_subtask.value,
        "all_pass": True,
        "chemistry": {"all_pass": True, "checks": chemistry_checks},
        "propagation": {"all_pass": True, "checks": propagation_checks},
        "bundle": {
            "all_pass": True,
            "record_count": len(records),
            "hallucinated_count": len(h_records),
            "faithful_count": len(n_records),
            "unique_record_ids": len(by_id),
            "reciprocal_pairs": reciprocal,
            "faithful_controls_exact": faithful_controls,
            "unique_control_identities": unique_controls,
            "unique_render_identities": unique_renders,
            "hallucinated_states_bound_to_propagation": h_states_bound,
        },
    }


def _build_origin(
    config: ConfigBundle,
    joined: JoinedInputRecord,
    spec: GoldenOriginSpec,
    *,
    global_seed: int,
) -> GoldenOriginBundle:
    artifact = build_reference_dag(joined)
    if artifact.normalized_subtask is not spec.normalized_subtask:
        raise GoldenBundleBuildError(
            "GOLDEN_ORIGIN_MISMATCH",
            "frozen origin normalized to a different editing subtask",
            origin_id=spec.origin_id,
        )
    truth = derive_edit_truth(artifact)
    record = task_record_from_joined_input(joined)
    propagator = EditingPropagationEngine()
    candidate_engine_type = _ENGINE_TYPES[spec.normalized_subtask]
    perturbator_type = _PERTURBATOR_TYPES[spec.normalized_subtask]
    candidate_engine = candidate_engine_type(operators_config=config.operators)
    perturbator_type(
        candidate_engine=candidate_engine,
        propagator=propagator,
        renderer=_UnusedTraceRenderer(),
        validators=_UnusedValidatorChain(),
        label_projector=_UnusedLabelProjector(),
    )
    registry = PerturbatorRegistry.from_perturbator_types(
        (perturbator_type,),
        operators_config=config.operators,
    )
    executions: list[GoldenPolicyExecution] = []
    assignments: list[QuotaAssignment] = []
    for policy_spec in spec.policies:
        execution, assignment = _build_execution(
            config=config,
            origin_spec=spec,
            policy_spec=policy_spec,
            record=record,
            reference_graph=artifact.state_dag,
            truth=truth,
            candidate_engine=candidate_engine,
            propagator=propagator,
            registry=registry,
            global_seed=global_seed,
        )
        executions.append(execution)
        assignments.append(assignment)
    request = MatchedBundleBuildRequest(
        origin_id=spec.origin_id,
        normalized_subtask=spec.normalized_subtask,
        input_view_id=f"t025:{spec.origin_id}:source_instruction_v1",
        assignments=tuple(assignments),
    )
    bundle = MatchedBundleBuilder().build(request)
    frozen_executions = tuple(executions)
    return GoldenOriginBundle(
        spec=spec,
        bundle=bundle,
        executions=frozen_executions,
    )


def build_t025_golden_corpus(
    dataset_root: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    global_seed: int | None = None,
    origin_specs: tuple[GoldenOriginSpec, ...] = T025_GOLDEN_ORIGINS,
) -> GoldenCorpusBuild:
    """Replay all three real origins through T019--T024 and validate 24 records."""

    root = DEFAULT_DATASET_ROOT if dataset_root is None else Path(dataset_root)
    if not root.is_dir():
        raise GoldenBundleBuildError(
            "GOLDEN_DATASET_MISSING", f"dataset root does not exist: {root}"
        )
    loaded_config = load_config_bundle() if config is None else config
    if type(loaded_config) is not ConfigBundle:
        raise TypeError("config must be ConfigBundle or None")
    configured_seed = loaded_config.dataset.dataset.global_seed
    selected_seed = configured_seed if global_seed is None else global_seed
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("global_seed must be a non-negative integer or None")
    if selected_seed != configured_seed:
        raise GoldenBundleBuildError(
            "GOLDEN_SEED_MISMATCH",
            "T025 replay requires the frozen config global seed",
        )
    specs = tuple(origin_specs)
    if len(specs) != 3 or tuple(item.normalized_subtask for item in specs) != tuple(
        EditingSubtask
    ):
        raise ValueError("origin_specs must contain ADD/DELETE/SUBSTITUTE exactly once")
    records = _records_by_id(ChemCoTMolEditAdapter().load(root))
    built: list[GoldenOriginBundle] = []
    for spec in specs:
        joined = records.get(spec.origin_id)
        if joined is None:
            raise GoldenBundleBuildError(
                "GOLDEN_ORIGIN_MISSING",
                "frozen real origin is absent from the joined Pilot corpus",
                origin_id=spec.origin_id,
            )
        built.append(
            _build_origin(
                loaded_config,
                joined,
                spec,
                global_seed=selected_seed,
            )
        )
    corpus = GoldenCorpusBuild(
        dataset_version=loaded_config.dataset.dataset.version_name,
        global_seed=selected_seed,
        origins=tuple(built),
    )
    report = corpus.validation_report()
    if not report["all_pass"] or report["summary"]["record_count"] != 24:
        raise GoldenBundleBuildError(
            "GOLDEN_CORPUS_INVALID", "T025 corpus failed its final 24-record gate"
        )
    return corpus


def write_t025_golden_artifacts(
    dataset_root: Path | None = None,
    bundle_path: Path | None = None,
    validation_path: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    global_seed: int | None = None,
) -> GoldenCorpusBuild:
    """Build and write canonical UTF-8 JSON snapshots, returning the typed corpus."""

    corpus = build_t025_golden_corpus(
        dataset_root,
        config=config,
        global_seed=global_seed,
    )
    bundle_target = (
        DEFAULT_PROJECT_ROOT / DEFAULT_GOLDEN_BUNDLE_PATH
        if bundle_path is None
        else Path(bundle_path)
    )
    validation_target = (
        DEFAULT_PROJECT_ROOT / DEFAULT_GOLDEN_VALIDATION_PATH
        if validation_path is None
        else Path(validation_path)
    )
    bundle_target.parent.mkdir(parents=True, exist_ok=True)
    validation_target.parent.mkdir(parents=True, exist_ok=True)
    bundle_target.write_text(corpus.render_bundle_json(), encoding="utf-8")
    validation_target.write_text(corpus.render_validation_json(), encoding="utf-8")
    return corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / DEFAULT_GOLDEN_BUNDLE_PATH,
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / DEFAULT_GOLDEN_VALIDATION_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = write_t025_golden_artifacts(
        args.dataset_root,
        args.bundle_output,
        args.validation_output,
    )
    summary = corpus.validation_report()["summary"]
    print(
        "T025 golden bundles generated: "
        f"{summary['origin_bundle_count']} origins, {summary['record_count']} records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_GOLDEN_BUNDLE_PATH",
    "DEFAULT_GOLDEN_VALIDATION_PATH",
    "DEFAULT_PROJECT_ROOT",
    "GOLDEN_BUNDLE_FORMAT_VERSION",
    "GOLDEN_VALIDATION_FORMAT_VERSION",
    "T025_GOLDEN_ORIGINS",
    "GoldenBundleBuildError",
    "GoldenCorpusBuild",
    "GoldenOriginBundle",
    "GoldenOriginSpec",
    "GoldenPolicyExecution",
    "GoldenPolicySpec",
    "build_t025_golden_corpus",
    "main",
    "write_t025_golden_artifacts",
]
