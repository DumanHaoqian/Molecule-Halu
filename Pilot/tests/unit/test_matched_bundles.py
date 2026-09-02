"""T024 matched H/N bundle and deterministic quota scheduling contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from functools import cache
from pathlib import Path

import pytest

from molhallulens.modules.release import assembly as builders
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import build_reference_dag
from molhallulens.modules.release.assembly import (
    BundleDraftError,
    DeterministicQuotaScheduler,
    MatchedBundleBuilder,
    MatchedBundleBuildRequest,
    MatchedBundleDraft,
    PreparedHallucinatedVariant,
    QuotaAssignment,
    QuotaScheduleError,
    QuotaScheduleRequest,
)
from molhallulens.infrastructure.chemistry import isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.core import (
    CandidateSourceType,
    CausalRole,
    EditErrorSubtype,
    EditingSubtask,
    GraphDelta,
    HallucinationType,
    MutationEvent,
    MutationTargetKind,
    OperatorCapability,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    ValueProvenance,
    VariantLabel,
)
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    DeletionPerturbator,
    SubstitutionPerturbator,
)
from molhallulens.modules.error_injection.operators.addition import ADDITION_OPERATOR_IDS
from molhallulens.modules.error_injection.operators.deletion import DELETION_OPERATOR_IDS
from molhallulens.modules.error_injection.operators.substitution import SUBSTITUTION_OPERATOR_IDS
from molhallulens.modules.error_injection.registry import (
    FallbackDecision,
    PerturbatorRegistry,
    operator,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
CONFIG = load_config_bundle()
POLICIES = (
    PropagationPolicy.STOP,
    PropagationPolicy.PARTIAL,
    PropagationPolicy.FULL_CF,
    PropagationPolicy.TERMINAL,
)
POLICY_BUCKET = {
    PropagationPolicy.STOP: "heavy_ring_count_claim",
    PropagationPolicy.PARTIAL: "count_ring_propagation",
    PropagationPolicy.FULL_CF: "wrong_attachment_atom_bond",
    PropagationPolicy.TERMINAL: "terminal_valid_high_similarity",
}
SYNTHETIC_PARTIAL_RELATION_ID = "mol_edit.add.synthetic_partial_relation"


class _SyntheticQuotaAddition(AdditionPerturbator):
    @operator(
        operator_id=SYNTHETIC_PARTIAL_RELATION_ID,
        operator_family="nl_formal_internal_relation",
        root_fields={"anchor_idx"},
        supported_policies={PropagationPolicy.PARTIAL},
        supported_sources={CandidateSourceType.RULE},
        hallucination_types={HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
        required_capabilities={OperatorCapability.CLAIM_PERTURBATION},
    )
    def synthetic_partial_relation(self, context):
        raise AssertionError("quota scheduling must never invoke an operator")


_BUNDLE_OPERATOR = {
    EditingSubtask.ADD: {
        PropagationPolicy.STOP: (ADDITION_OPERATOR_IDS[7], "numeric_count_claim"),
        PropagationPolicy.PARTIAL: (
            ADDITION_OPERATOR_IDS[7],
            "numeric_count_claim",
        ),
        PropagationPolicy.FULL_CF: (
            ADDITION_OPERATOR_IDS[3],
            "attachment_bond_edit",
        ),
        PropagationPolicy.TERMINAL: (
            ADDITION_OPERATOR_IDS[10],
            "final_answer_identity",
        ),
    },
    EditingSubtask.DELETE: {
        PropagationPolicy.STOP: (DELETION_OPERATOR_IDS[9], "numeric_count_claim"),
        PropagationPolicy.PARTIAL: (
            DELETION_OPERATOR_IDS[9],
            "numeric_count_claim",
        ),
        PropagationPolicy.FULL_CF: (
            DELETION_OPERATOR_IDS[2],
            "attachment_bond_edit",
        ),
        PropagationPolicy.TERMINAL: (
            DELETION_OPERATOR_IDS[11],
            "final_answer_identity",
        ),
    },
    EditingSubtask.SUBSTITUTE: {
        PropagationPolicy.STOP: (
            SUBSTITUTION_OPERATOR_IDS[9],
            "numeric_count_claim",
        ),
        PropagationPolicy.PARTIAL: (
            SUBSTITUTION_OPERATOR_IDS[9],
            "numeric_count_claim",
        ),
        PropagationPolicy.FULL_CF: (
            SUBSTITUTION_OPERATOR_IDS[3],
            "attachment_bond_edit",
        ),
        PropagationPolicy.TERMINAL: (
            SUBSTITUTION_OPERATOR_IDS[11],
            "final_answer_identity",
        ),
    },
}

_ADD_FAMILY_OPERATOR = {
    "wrong_anchor_site": (ADDITION_OPERATOR_IDS[0], "anchor_idx"),
    "wrong_fragment_group": (ADDITION_OPERATOR_IDS[2], "add_fragment"),
    "attachment_bond_edit": (ADDITION_OPERATOR_IDS[3], "product"),
    "numeric_count_claim": (ADDITION_OPERATOR_IDS[7], "product_heavy"),
    "nl_formal_internal_relation": (SYNTHETIC_PARTIAL_RELATION_ID, "anchor_idx"),
    "final_answer_identity": (ADDITION_OPERATOR_IDS[10], "final_answer"),
}


@cache
def _records() -> tuple[JoinedInputRecord, ...]:
    return ChemCoTMolEditAdapter().load(DATASET_ROOT)


def _different_smiles(reference: str) -> str:
    for candidate in ("C", "N", "CC", "CO", "C1CC1"):
        if not isomeric_graph_equivalent(candidate, reference):
            return candidate
    raise AssertionError("test fixture could not choose a non-equivalent molecule")


def _changed_claim(reference_graph: StateDAG, node_id: str):
    old = reference_graph.values[node_id]
    value = old.normalized_value
    if type(value) is int:
        changed = value + 1
    elif node_id == "anchor_element":
        changed = "N" if value != "N" else "C"
    elif node_id in {"add_fragment", "remove_group", "remove_group_step1"}:
        changed = "F" if value != "F" else "Cl"
    else:
        changed = _different_smiles(value)
    return replace(
        old,
        raw_value=changed,
        normalized_value=changed,
        provenance=ValueProvenance.RULE,
    )


def _candidate_and_delta(
    reference_graph: StateDAG,
    *,
    operator_id: str,
    target_node_id: str,
    policy: PropagationPolicy,
) -> tuple[StateDAG, GraphDelta]:
    values = dict(reference_graph.values)
    before = values[target_node_id]
    after = _changed_claim(reference_graph, target_node_id)
    values[target_node_id] = after
    root_id = f"t024:{operator_id}:{policy.dataset_name}:root"
    terminal = policy is PropagationPolicy.TERMINAL
    events = [
        MutationEvent(
            event_id=root_id,
            target_kind=MutationTargetKind.NODE,
            node_or_edge_id=target_node_id,
            before=before,
            after=after,
            causal_role=CausalRole.TERMINAL if terminal else CausalRole.ROOT,
            hallucination_types=frozenset({HallucinationType.REASONING_ERROR}),
            edit_subtypes=frozenset(
                {
                    EditErrorSubtype.FINAL_ANSWER_IDENTITY
                    if terminal
                    else EditErrorSubtype.INTERNAL_INCONSISTENCY
                }
            ),
            operator_id=operator_id,
            root_event_id=root_id,
        )
    ]
    if policy is PropagationPolicy.PARTIAL:
        downstream_by_root = {
            "anchor_idx": "anchor_element",
            "add_fragment": "fragment_heavy",
            "product": "product_heavy",
            "product_heavy": "heavy_delta",
        }
        downstream_id = downstream_by_root.get(target_node_id)
        if downstream_id is None:
            raise AssertionError(
                f"fixture root {target_node_id!r} has no bounded PARTIAL descendant"
            )
        downstream_before = values[downstream_id]
        downstream_after = _changed_claim(reference_graph, downstream_id)
        values[downstream_id] = downstream_after
        events.append(
            MutationEvent(
                event_id=f"{root_id}:{downstream_id}",
                target_kind=MutationTargetKind.NODE,
                node_or_edge_id=downstream_id,
                before=downstream_before,
                after=downstream_after,
                causal_role=CausalRole.PROPAGATED_CONDITIONAL,
                hallucination_types=frozenset({HallucinationType.REASONING_ERROR}),
                edit_subtypes=frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
                operator_id=operator_id,
                root_event_id=root_id,
            )
        )
    if policy is PropagationPolicy.FULL_CF:
        old_product = values["product"]
        product = _different_smiles(old_product.normalized_value)
        new_product = replace(
            old_product,
            raw_value=product,
            normalized_value=product,
            provenance=ValueProvenance.RULE,
        )
        if target_node_id == "product":
            values["product"] = new_product
            events[0] = replace(events[0], after=new_product)
        else:
            values["product"] = new_product
            events.append(
                MutationEvent(
                    event_id=f"{root_id}:product",
                    target_kind=MutationTargetKind.NODE,
                    node_or_edge_id="product",
                    before=old_product,
                    after=new_product,
                    causal_role=CausalRole.PROPAGATED_CONDITIONAL,
                    hallucination_types=frozenset({HallucinationType.REASONING_ERROR}),
                    edit_subtypes=frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
                    operator_id=operator_id,
                    root_event_id=root_id,
                )
            )
        old_answer = values["final_answer"]
        new_answer = replace(
            old_answer,
            raw_value=product,
            normalized_value=product,
            provenance=ValueProvenance.RULE,
        )
        if not old_answer.semantically_equals(new_answer):
            values["final_answer"] = new_answer
            events.append(
                MutationEvent(
                    event_id=f"{root_id}:final_answer",
                    target_kind=MutationTargetKind.NODE,
                    node_or_edge_id="final_answer",
                    before=old_answer,
                    after=new_answer,
                    causal_role=CausalRole.PROPAGATED_CONDITIONAL,
                    hallucination_types=frozenset({HallucinationType.REASONING_ERROR}),
                    edit_subtypes=frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
                    operator_id=operator_id,
                    root_event_id=root_id,
                )
            )
    candidate = StateDAG(reference_graph.schema, values, reference_graph.edge_values)
    return candidate, GraphDelta(tuple(events))


def _recipe(
    origin_id: str,
    *,
    operator_id: str,
    target_node_id: str,
    policy: PropagationPolicy,
    variant_index: int = 0,
    candidate_source: CandidateSourceType = CandidateSourceType.RULE,
) -> PerturbationRecipe:
    return PerturbationRecipe(
        recipe_id=f"t024:{origin_id}:{policy.dataset_name}:{operator_id}:{variant_index}",
        origin_id=origin_id,
        operator_id=operator_id,
        policy=policy,
        target_node_id=target_node_id,
        candidate_source_mode=candidate_source,
        variant_index=variant_index,
        derived_seed=20260830,
        rewrite_budget=RewriteBudget(16, 128, "matched-medium"),
        candidate_difficulty_bucket="hard",
        renderer_style_id="formal-v1",
        partial_cut_nodes=(
            frozenset({"product"})
            if policy is PropagationPolicy.PARTIAL
            else frozenset()
        ),
    )


def _prepared(
    *,
    origin_id: str,
    subtask: EditingSubtask,
    reference_graph: StateDAG,
    policy: PropagationPolicy,
    operator_id: str,
    operator_family: str,
    quota_bucket: str,
    target_node_id: str,
    variant_index: int = 0,
    fallback_decision: FallbackDecision | None = None,
    candidate_source: CandidateSourceType = CandidateSourceType.RULE,
) -> PreparedHallucinatedVariant:
    candidate, delta = _candidate_and_delta(
        reference_graph,
        operator_id=operator_id,
        target_node_id=target_node_id,
        policy=policy,
    )
    return PreparedHallucinatedVariant(
        origin_id=origin_id,
        normalized_subtask=subtask,
        input_view_id=f"input:{origin_id}",
        recipe=_recipe(
            origin_id,
            operator_id=operator_id,
            target_node_id=target_node_id,
            policy=policy,
            variant_index=variant_index,
            candidate_source=candidate_source,
        ),
        operator_family=operator_family,
        quota_bucket=quota_bucket,
        renderer_backend="deterministic-formal",
        reference_graph=reference_graph,
        candidate_graph=candidate,
        graph_delta=delta,
        fallback_decision=fallback_decision,
    )


def _bundle_request(record: JoinedInputRecord) -> MatchedBundleBuildRequest:
    artifact = build_reference_dag(record)
    assignments = []
    for policy in POLICIES:
        operator_id, family = _BUNDLE_OPERATOR[artifact.normalized_subtask][policy]
        target = (
            "product"
            if policy is PropagationPolicy.FULL_CF
            else (
                "final_answer"
                if policy is PropagationPolicy.TERMINAL
                else "product_heavy"
            )
        )
        variant = _prepared(
            origin_id=record.anonymous_sample_id,
            subtask=artifact.normalized_subtask,
            reference_graph=artifact.state_dag,
            policy=policy,
            operator_id=operator_id,
            operator_family=family,
            quota_bucket=POLICY_BUCKET[policy],
            target_node_id=target,
        )
        assignments.append(
            QuotaAssignment(
                record.anonymous_sample_id,
                policy,
                POLICY_BUCKET[policy],
                variant,
            )
        )
    return MatchedBundleBuildRequest(
        origin_id=record.anonymous_sample_id,
        normalized_subtask=artifact.normalized_subtask,
        input_view_id=f"input:{record.anonymous_sample_id}",
        assignments=tuple(assignments),
    )


@cache
def _all_add_quota_variants() -> tuple[PreparedHallucinatedVariant, ...]:
    records = tuple(
        record for record in _records() if ".add_v2." in record.anonymous_sample_id
    )
    reference = build_reference_dag(records[0]).state_dag
    variants = []
    for origin_index, record in enumerate(records):
        for policy in POLICIES:
            quota_items = CONFIG.operators.quotas_per_subtask_policy[
                policy.dataset_name
            ]
            for family_index, quota in enumerate(quota_items):
                family = CONFIG.operators.quota_bucket_mappings[quota.family][0]
                operator_id, target = _ADD_FAMILY_OPERATOR[family]
                variants.append(
                    _prepared(
                        origin_id=record.anonymous_sample_id,
                        subtask=EditingSubtask.ADD,
                        reference_graph=reference,
                        policy=policy,
                        operator_id=operator_id,
                        operator_family=family,
                        quota_bucket=quota.family,
                        target_node_id=target,
                        variant_index=origin_index * 10 + family_index,
                    )
                )
    return tuple(variants)


@cache
def _addition_registry() -> PerturbatorRegistry:
    return PerturbatorRegistry.from_perturbator_types(
        (AdditionPerturbator, _SyntheticQuotaAddition, DeletionPerturbator),
        operators_config=CONFIG.operators,
    )


def _scheduler() -> DeterministicQuotaScheduler:
    return DeterministicQuotaScheduler(
        operators_config=CONFIG.operators,
        registry=_addition_registry(),
    )


def _quota_request(
    origins: tuple[str, ...],
    variants: tuple[PreparedHallucinatedVariant, ...],
    *,
    global_seed: int = 20260830,
    allow_quota_deviation: bool = False,
) -> QuotaScheduleRequest:
    return QuotaScheduleRequest(
        normalized_subtask=EditingSubtask.ADD,
        origin_ids=origins,
        variants=variants,
        global_seed=global_seed,
        seed_namespace="t024.matched-bundles.unit",
        allow_quota_deviation=allow_quota_deviation,
    )


def _fallback_decision(
    *,
    selected_operator_id: str,
    selected_family: str,
    quota_bucket: str,
    quota_deviation: bool,
) -> FallbackDecision:
    return FallbackDecision(
        requested_operator_id=ADDITION_OPERATOR_IDS[3],
        selected_operator_id=selected_operator_id,
        requested_operator_family="attachment_bond_edit",
        selected_operator_family=selected_family,
        policy=PropagationPolicy.FULL_CF,
        candidate_source=CandidateSourceType.RULE,
        quota_bucket=quota_bucket,
        attempted_operator_ids=(ADDITION_OPERATOR_IDS[3],),
        quota_deviation=quota_deviation,
        target_change_required=False,
    )


def _fallback_partition_variants(
    *, cross_family: bool, include_same_family: bool = False
) -> tuple[PreparedHallucinatedVariant, ...]:
    records = tuple(
        record for record in _records() if ".add_v2." in record.anonymous_sample_id
    )
    reference = build_reference_dag(records[0]).state_dag
    baseline = tuple(
        variant
        for variant in _all_add_quota_variants()
        if variant.recipe.policy is not PropagationPolicy.FULL_CF
    )
    full = []
    for index, record in enumerate(records):
        if index < 18:
            operator_id, target = _ADD_FAMILY_OPERATOR["wrong_anchor_site"]
            full.append(
                _prepared(
                    origin_id=record.anonymous_sample_id,
                    subtask=EditingSubtask.ADD,
                    reference_graph=reference,
                    policy=PropagationPolicy.FULL_CF,
                    operator_id=operator_id,
                    operator_family="wrong_anchor_site",
                    quota_bucket="valid_wrong_site_occurrence_regioisomer",
                    target_node_id=target,
                    variant_index=index,
                )
            )
        elif index < 33:
            operator_id, target = _ADD_FAMILY_OPERATOR["wrong_fragment_group"]
            full.append(
                _prepared(
                    origin_id=record.anonymous_sample_id,
                    subtask=EditingSubtask.ADD,
                    reference_graph=reference,
                    policy=PropagationPolicy.FULL_CF,
                    operator_id=operator_id,
                    operator_family="wrong_fragment_group",
                    quota_bucket="valid_wrong_group_fragment",
                    target_node_id=target,
                    variant_index=index,
                )
            )
        elif index < 43:
            operator_id, target = _ADD_FAMILY_OPERATOR["attachment_bond_edit"]
            full.append(
                _prepared(
                    origin_id=record.anonymous_sample_id,
                    subtask=EditingSubtask.ADD,
                    reference_graph=reference,
                    policy=PropagationPolicy.FULL_CF,
                    operator_id=operator_id,
                    operator_family="attachment_bond_edit",
                    quota_bucket="wrong_attachment_atom_bond",
                    target_node_id=target,
                    variant_index=index,
                )
            )
        else:
            quota_bucket = "alternate_valid_edit_boundary"
            if include_same_family:
                full.append(
                    _prepared(
                        origin_id=record.anonymous_sample_id,
                        subtask=EditingSubtask.ADD,
                        reference_graph=reference,
                        policy=PropagationPolicy.FULL_CF,
                        operator_id=ADDITION_OPERATOR_IDS[4],
                        operator_family="attachment_bond_edit",
                        quota_bucket=quota_bucket,
                        target_node_id="product",
                        variant_index=index,
                        fallback_decision=_fallback_decision(
                            selected_operator_id=ADDITION_OPERATOR_IDS[4],
                            selected_family="attachment_bond_edit",
                            quota_bucket=quota_bucket,
                            quota_deviation=False,
                        ),
                    )
                )
            if cross_family:
                full.append(
                    _prepared(
                        origin_id=record.anonymous_sample_id,
                        subtask=EditingSubtask.ADD,
                        reference_graph=reference,
                        policy=PropagationPolicy.FULL_CF,
                        operator_id=ADDITION_OPERATOR_IDS[5],
                        operator_family="wrong_anchor_site",
                        quota_bucket=quota_bucket,
                        target_node_id="product",
                        variant_index=index + 100,
                        fallback_decision=_fallback_decision(
                            selected_operator_id=ADDITION_OPERATOR_IDS[5],
                            selected_family="wrong_anchor_site",
                            quota_bucket=quota_bucket,
                            quota_deviation=True,
                        ),
                    )
                )
    return (*baseline, *full)


def test_all_150_origins_build_exact_deterministic_matched_drafts() -> None:
    builder = MatchedBundleBuilder()
    for record in _records():
        request = _bundle_request(record)
        bundle = builder.build(request)
        repeated = builder.build(request)

        assert bundle == repeated
        assert bundle.to_json() == repeated.to_json()
        assert len(bundle.records) == 8
        assert Counter(item.variant_label for item in bundle.records) == {
            VariantLabel.HALLUCINATED: 4,
            VariantLabel.FAITHFUL: 4,
        }
        assert Counter(item.policy for item in bundle.records) == {
            policy: 2 for policy in POLICIES
        }
        assert tuple(
            (item.policy, item.variant_label) for item in bundle.records
        ) == tuple(
            (policy, label)
            for policy in POLICIES
            for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
        )


def test_each_pair_is_reciprocal_matched_and_faithful_control_is_not_reused() -> None:
    bundle = MatchedBundleBuilder().build(_bundle_request(_records()[0]))
    controls = set()
    faithful_render_ids = set()
    for hallucinated, faithful in zip(
        bundle.records[::2], bundle.records[1::2], strict=True
    ):
        assert hallucinated.matched_record_id == faithful.record_id
        assert faithful.matched_record_id == hallucinated.record_id
        assert hallucinated.record_id != faithful.record_id
        for attribute in (
            "origin_id",
            "bundle_id",
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
            "control_identity",
        ):
            assert getattr(hallucinated, attribute) == getattr(faithful, attribute)
        assert hallucinated.graph_delta.events
        assert faithful.locked_state == faithful.reference_graph
        assert not faithful.graph_delta.events
        assert faithful.control_identity not in controls
        assert faithful.render_identity not in faithful_render_ids
        controls.add(faithful.control_identity)
        faithful_render_ids.add(faithful.render_identity)


def test_full_and_terminal_pairs_lock_exact_answer_phenotypes() -> None:
    bundle = MatchedBundleBuilder().build(_bundle_request(_records()[0]))
    by_key = {
        (record.policy, record.variant_label): record for record in bundle.records
    }
    full_h = by_key[(PropagationPolicy.FULL_CF, VariantLabel.HALLUCINATED)]
    terminal_h = by_key[(PropagationPolicy.TERMINAL, VariantLabel.HALLUCINATED)]
    terminal_n = by_key[(PropagationPolicy.TERMINAL, VariantLabel.FAITHFUL)]

    assert full_h.answer.product_equivalent
    assert isomeric_graph_equivalent(
        full_h.answer.smiles,
        full_h.locked_state.values["product"].normalized_value,
    )
    assert not terminal_h.answer.product_equivalent
    assert terminal_n.answer.product_equivalent
    assert terminal_n.graph_delta == GraphDelta(())


def test_bundle_request_missing_or_duplicate_policy_fails_closed() -> None:
    request = _bundle_request(_records()[0])
    for assignments in (
        request.assignments[:-1],
        (*request.assignments[:-1], request.assignments[0]),
    ):
        adversarial = replace(request)
        object.__setattr__(adversarial, "assignments", assignments)
        with pytest.raises(BundleDraftError) as captured:
            MatchedBundleBuilder().build(adversarial)
        assert captured.value.code == "BUNDLE_POLICY_SET_MISMATCH"


def test_bundle_record_pair_and_control_tampering_fail_closed() -> None:
    bundle = MatchedBundleBuilder().build(_bundle_request(_records()[0]))
    first_h, first_n, second_h, second_n, *rest = bundle.records
    tamper_cases = (
        (
            (
                first_h,
                replace(first_n, matched_record_id=second_h.record_id),
                second_h,
                second_n,
                *rest,
            ),
            "BUNDLE_PAIR_MISMATCH",
        ),
        (
            (
                first_h,
                first_n,
                replace(second_h, control_identity=first_h.control_identity),
                replace(second_n, control_identity=first_h.control_identity),
                *rest,
            ),
            "BUNDLE_CONTROL_REUSE",
        ),
    )
    for records, code in tamper_cases:
        with pytest.raises(BundleDraftError) as captured:
            MatchedBundleDraft(bundle.origin_id, bundle.bundle_id, tuple(records))
        assert captured.value.code == code


def test_rendered_formal_answer_and_delta_must_match_locked_state() -> None:
    bundle = MatchedBundleBuilder().build(_bundle_request(_records()[0]))
    full_h = next(
        item
        for item in bundle.records
        if item.policy is PropagationPolicy.FULL_CF
        and item.variant_label is VariantLabel.HALLUCINATED
    )
    full_n = next(
        item
        for item in bundle.records
        if item.policy is PropagationPolicy.FULL_CF
        and item.variant_label is VariantLabel.FAITHFUL
    )
    for change in (
        {"formal_trace": full_n.formal_trace},
        {"answer": full_n.answer},
        {"graph_delta": GraphDelta(())},
    ):
        with pytest.raises(BundleDraftError) as captured:
            replace(full_h, **change)
        assert captured.value.code == "BUNDLE_H_STATE_INVALID"


def test_prepared_variant_delta_tamper_is_structured_not_raw_value_error() -> None:
    variant = _bundle_request(_records()[0]).assignments[0].variant
    with pytest.raises(BundleDraftError) as captured:
        replace(variant, graph_delta=GraphDelta(()))
    assert captured.value.code == "BUNDLE_H_STATE_INVALID"


@pytest.mark.parametrize("field_name", ("before", "after"))
def test_delta_event_claims_must_exactly_bind_reference_and_candidate(
    field_name: str,
) -> None:
    variant = _bundle_request(_records()[0]).assignments[0].variant
    event = variant.graph_delta.events[0]
    original = getattr(event, field_name)
    assert type(original.normalized_value) is int
    forged = replace(
        original,
        raw_value=original.normalized_value + 17,
        normalized_value=original.normalized_value + 17,
    )
    forged_event = replace(event, **{field_name: forged})
    forged_delta = GraphDelta((forged_event, *variant.graph_delta.events[1:]))

    with pytest.raises(BundleDraftError) as captured:
        replace(variant, graph_delta=forged_delta)
    assert captured.value.code == "BUNDLE_H_STATE_INVALID"


def test_synthetic_full_availability_hits_every_frozen_quota_exactly() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = _all_add_quota_variants()
    forward = _scheduler().schedule(_quota_request(origins, variants))
    reversed_input = _scheduler().schedule(
        _quota_request(tuple(reversed(origins)), tuple(reversed(variants)))
    )
    different_seed = _scheduler().schedule(
        _quota_request(origins, variants, global_seed=20260831)
    )

    assert forward == reversed_input
    assert forward.to_dict() == reversed_input.to_dict()
    assert len(forward.assignments) == 200
    assert not forward.deviations
    actual = Counter(
        (assignment.policy.dataset_name, assignment.quota_bucket)
        for assignment in forward.assignments
    )
    expected = {
        (policy_name, quota.family): quota.target_per_50
        for policy_name, quotas in CONFIG.operators.quotas_per_subtask_policy.items()
        for quota in quotas
    }
    assert actual == expected
    assert all(len(forward.assignments_for_origin(origin)) == 4 for origin in origins)
    assert different_seed.report.counts == forward.report.counts
    assert different_seed.assignments != forward.assignments


def test_stable_keys_include_matching_axes_and_input_order_is_irrelevant() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = _all_add_quota_variants()
    baseline = _scheduler().schedule(_quota_request(origins, variants))
    selected = baseline.assignments[0].variant
    tied = replace(selected, renderer_backend="deterministic-formal-alternate")
    assert tied.stable_key != selected.stable_key
    with_tie = (*variants, tied)

    forward = _scheduler().schedule(_quota_request(origins, with_tie))
    reversed_input = _scheduler().schedule(
        _quota_request(tuple(reversed(origins)), tuple(reversed(with_tie)))
    )
    assert forward == reversed_input
    assert forward.to_dict() == reversed_input.to_dict()


def test_same_family_fallback_is_used_before_cross_family_deviation() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = _fallback_partition_variants(
        cross_family=True,
        include_same_family=True,
    )
    schedule = _scheduler().schedule(_quota_request(origins, variants))

    alternate_full = tuple(
        assignment
        for assignment in schedule.assignments
        if assignment.policy is PropagationPolicy.FULL_CF
        and assignment.quota_bucket == "alternate_valid_edit_boundary"
    )
    assert len(alternate_full) == 7
    assert all(
        item.variant.recipe.operator_id == ADDITION_OPERATOR_IDS[4]
        and item.variant.operator_family == "attachment_bond_edit"
        and not item.variant.quota_deviation
        for item in alternate_full
    )
    assert all(
        deviation.policy is PropagationPolicy.FULL_CF and not deviation.quota_deviation
        for deviation in schedule.deviations
    )


def test_cross_family_fallback_requires_opt_in_and_preserves_phenotype() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = _fallback_partition_variants(cross_family=True)
    with pytest.raises(QuotaScheduleError) as captured:
        _scheduler().schedule(_quota_request(origins, variants))
    assert captured.value.code == "BACKFILL_REQUIRED"

    schedule = _scheduler().schedule(
        _quota_request(origins, variants, allow_quota_deviation=True)
    )
    assert len(schedule.deviations) == 7
    assert all(item.quota_deviation for item in schedule.deviations)
    for assignment in schedule.assignments:
        decision = assignment.variant.fallback_decision
        if decision is None:
            continue
        assert assignment.policy is decision.policy is PropagationPolicy.FULL_CF
        assert (
            assignment.variant.recipe.candidate_source_mode
            is decision.candidate_source
            is CandidateSourceType.RULE
        )
        assert assignment.variant.recipe.target_node_id == "product"
        assert not decision.target_change_required
        assert assignment.quota_bucket == decision.quota_bucket


@pytest.mark.parametrize(
    "decision_change",
    (
        {"attempted_operator_ids": ()},
        {"target_change_required": True},
    ),
)
def test_fallback_ledger_tampering_is_rejected_before_scheduling(
    decision_change,
) -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = list(_fallback_partition_variants(cross_family=True))
    index = next(
        index
        for index, variant in enumerate(variants)
        if variant.fallback_decision is not None
    )
    original = variants[index]
    variants[index] = replace(
        original,
        fallback_decision=replace(original.fallback_decision, **decision_change),
    )

    with pytest.raises(BundleDraftError) as captured:
        _scheduler().schedule(
            _quota_request(
                origins,
                tuple(variants),
                allow_quota_deviation=True,
            )
        )
    assert captured.value.code == "QUOTA_FALLBACK_UNDECLARED"


def test_forged_requested_or_attempted_registration_contract_is_rejected() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    reference = build_reference_dag(
        next(
            record for record in _records() if ".add_v2." in record.anonymous_sample_id
        )
    ).state_dag
    origin_id = origins[0]
    cases = (
        (
            ADDITION_OPERATOR_IDS[5],
            "wrong_anchor_site",
            "product",
            PropagationPolicy.FULL_CF,
            CandidateSourceType.RULE,
            "alternate_valid_edit_boundary",
            FallbackDecision(
                requested_operator_id=ADDITION_OPERATOR_IDS[7],
                selected_operator_id=ADDITION_OPERATOR_IDS[5],
                requested_operator_family="numeric_count_claim",
                selected_operator_family="wrong_anchor_site",
                policy=PropagationPolicy.FULL_CF,
                candidate_source=CandidateSourceType.RULE,
                quota_bucket="heavy_ring_count_claim",
                attempted_operator_ids=(ADDITION_OPERATOR_IDS[7],),
                quota_deviation=True,
                target_change_required=False,
            ),
        ),
        (
            ADDITION_OPERATOR_IDS[0],
            "wrong_anchor_site",
            "anchor_idx",
            PropagationPolicy.PARTIAL,
            CandidateSourceType.RDKIT,
            "entity_partial_propagation",
            FallbackDecision(
                requested_operator_id=ADDITION_OPERATOR_IDS[9],
                selected_operator_id=ADDITION_OPERATOR_IDS[0],
                requested_operator_family="nl_formal_internal_relation",
                selected_operator_family="wrong_anchor_site",
                policy=PropagationPolicy.PARTIAL,
                candidate_source=CandidateSourceType.RDKIT,
                quota_bucket="nl_formal_internal_relation",
                attempted_operator_ids=(ADDITION_OPERATOR_IDS[9],),
                quota_deviation=True,
                target_change_required=False,
            ),
        ),
        (
            ADDITION_OPERATOR_IDS[5],
            "wrong_anchor_site",
            "product",
            PropagationPolicy.FULL_CF,
            CandidateSourceType.RULE,
            "alternate_valid_edit_boundary",
            FallbackDecision(
                requested_operator_id=DELETION_OPERATOR_IDS[2],
                selected_operator_id=ADDITION_OPERATOR_IDS[5],
                requested_operator_family="attachment_bond_edit",
                selected_operator_family="wrong_anchor_site",
                policy=PropagationPolicy.FULL_CF,
                candidate_source=CandidateSourceType.RULE,
                quota_bucket="alternate_valid_edit_boundary",
                attempted_operator_ids=(DELETION_OPERATOR_IDS[2],),
                quota_deviation=True,
                target_change_required=False,
            ),
        ),
        (
            ADDITION_OPERATOR_IDS[4],
            "attachment_bond_edit",
            "product",
            PropagationPolicy.FULL_CF,
            CandidateSourceType.RULE,
            "alternate_valid_edit_boundary",
            FallbackDecision(
                requested_operator_id=ADDITION_OPERATOR_IDS[3],
                selected_operator_id=ADDITION_OPERATOR_IDS[4],
                requested_operator_family="attachment_bond_edit",
                selected_operator_family="attachment_bond_edit",
                policy=PropagationPolicy.FULL_CF,
                candidate_source=CandidateSourceType.RULE,
                quota_bucket="alternate_valid_edit_boundary",
                attempted_operator_ids=(
                    ADDITION_OPERATOR_IDS[3],
                    ADDITION_OPERATOR_IDS[10],
                ),
                quota_deviation=False,
                target_change_required=False,
            ),
        ),
    )
    for (
        selected_operator,
        selected_family,
        target,
        policy,
        source,
        selected_bucket,
        decision,
    ) in cases:
        forged = _prepared(
            origin_id=origin_id,
            subtask=EditingSubtask.ADD,
            reference_graph=reference,
            policy=policy,
            operator_id=selected_operator,
            operator_family=selected_family,
            quota_bucket=selected_bucket,
            target_node_id=target,
            fallback_decision=decision,
            candidate_source=source,
        )
        with pytest.raises(BundleDraftError) as captured:
            _scheduler().schedule(
                _quota_request(
                    origins,
                    (*_all_add_quota_variants(), forged),
                    allow_quota_deviation=True,
                )
            )
        assert captured.value.code == "QUOTA_FALLBACK_UNDECLARED"


def test_missing_family_availability_requires_structured_backfill() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = tuple(
        variant
        for variant in _all_add_quota_variants()
        if not (
            variant.recipe.policy is PropagationPolicy.PARTIAL
            and variant.operator_family == "nl_formal_internal_relation"
        )
    )
    with pytest.raises(QuotaScheduleError) as captured:
        _scheduler().schedule(_quota_request(origins, variants))

    assert captured.value.code == "BACKFILL_REQUIRED"
    assert not captured.value.report.all_pass
    missing = next(
        item
        for item in captured.value.report.backfills
        if item.policy is PropagationPolicy.PARTIAL
        and item.quota_bucket == "nl_formal_internal_relation"
    )
    assert missing.missing_count == 5
    assert missing.reason_code == "BACKFILL_REQUIRED"
    assert len(missing.unassigned_origin_ids) >= 5
    actual_relation = _addition_registry().registration(ADDITION_OPERATOR_IDS[9])
    assert actual_relation.spec.root_fields == frozenset({"anchor_element"})
    assert not build_reference_dag(
        next(
            record for record in _records() if ".add_v2." in record.anonymous_sample_id
        )
    ).state_dag.schema.descendants("anchor_element")


def test_declared_quota_bucket_must_match_selected_operator_family() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = list(_all_add_quota_variants())
    original = variants[0]
    variants[0] = replace(original, quota_bucket="group_fragment_identity")

    with pytest.raises(BundleDraftError) as captured:
        _scheduler().schedule(_quota_request(origins, tuple(variants)))
    assert captured.value.code == "QUOTA_CONFIG_MISMATCH"


def test_one_terminal_candidate_cannot_silently_fill_three_quota_buckets() -> None:
    origins = tuple(
        record.anonymous_sample_id
        for record in _records()
        if ".add_v2." in record.anonymous_sample_id
    )
    variants = tuple(
        variant
        for variant in _all_add_quota_variants()
        if variant.recipe.policy is not PropagationPolicy.TERMINAL
        or variant.quota_bucket == "terminal_valid_high_similarity"
    )
    with pytest.raises(QuotaScheduleError) as captured:
        _scheduler().schedule(_quota_request(origins, variants))

    terminal_backfills = {
        item.quota_bucket: item.missing_count
        for item in captured.value.report.backfills
        if item.policy is PropagationPolicy.TERMINAL
    }
    assert terminal_backfills == {
        "terminal_stereo_connectivity_regio": 10,
        "terminal_invalid_format_diagnostic": 5,
    }


def test_phase2_carriers_are_frozen_and_do_not_expose_t039_t040_outputs() -> None:
    bundle = MatchedBundleBuilder().build(_bundle_request(_records()[0]))
    record = bundle.records[0]
    with pytest.raises(FrozenInstanceError):
        record.record_id = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bundle.records = ()  # type: ignore[misc]

    serialized_keys = set(record.to_dict())
    assert (
        not {
            "natural_language",
            "serialized_text",
            "char_annotations",
            "token_labels",
        }
        & serialized_keys
    )
    assert "state_values" in serialized_keys
    assert (
        bundle.to_json()
        == MatchedBundleBuilder().build(_bundle_request(_records()[0])).to_json()
    )


def test_public_builder_exports_include_phase2_bundle_contracts() -> None:
    for name in (
        "BundleDraftError",
        "DeterministicQuotaScheduler",
        "MatchedBundleBuildRequest",
        "MatchedBundleBuilder",
        "MatchedBundleDraft",
        "MatchedDraftRecord",
        "PreparedHallucinatedVariant",
        "QuotaAssignment",
        "QuotaScheduleError",
        "QuotaScheduleRequest",
    ):
        assert getattr(builders, name) is not None


@pytest.mark.parametrize(
    "perturbator_type",
    (AdditionPerturbator, DeletionPerturbator, SubstitutionPerturbator),
)
def test_registry_sources_remain_owned_by_existing_perturbator_families(
    perturbator_type,
) -> None:
    registry = PerturbatorRegistry.from_perturbator_types(
        (perturbator_type,), operators_config=CONFIG.operators
    )
    assert registry.registrations_for()
