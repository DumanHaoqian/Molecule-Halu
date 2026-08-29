"""Acceptance tests for the three frozen T025 deterministic golden bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem, rdBase

from molhallulens.builders import golden_bundles as golden_bundle_module
from molhallulens.builders.bundles import MatchedBundleBuilder
from molhallulens.builders.golden_bundles import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_GOLDEN_BUNDLE_PATH,
    DEFAULT_GOLDEN_VALIDATION_PATH,
    T025_GOLDEN_ORIGINS,
    GoldenBundleBuildError,
    GoldenCorpusBuild,
    GoldenOriginBundle,
    build_t025_golden_corpus,
)
from molhallulens.candidates import replay_edit_action_from_source
from molhallulens.chemistry import isomeric_graph_equivalent
from molhallulens.config import load_config_bundle
from molhallulens.domain import (
    EditingSubtask,
    GraphDelta,
    MutationTargetKind,
    PropagationPolicy,
    StateDAG,
    StateSchema,
)
from molhallulens.perturbators import (
    AdditionCandidateEngine,
    AdditionPerturbator,
    DeletionCandidateEngine,
    DeletionPerturbator,
    SubstitutionCandidateEngine,
    SubstitutionPerturbator,
)
from molhallulens.perturbators.base import PropagationOutcome
from molhallulens.perturbators.registry import PerturbatorRegistry
from molhallulens.propagation import EditingPropagationEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL")
LABEL_ORDER = tuple((policy, label) for policy in POLICIES for label in ("H", "N"))
ENGINE_BY_SUBTASK = {
    "add": "AdditionCandidateEngine",
    "delete": "DeletionCandidateEngine",
    "substitute": "SubstitutionCandidateEngine",
}


def _frozen_json(relative_path: Path) -> dict[str, Any]:
    value = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _strict_molecule(smiles: Any) -> Chem.Mol:
    assert isinstance(smiles, str) and smiles
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    assert molecule is not None
    return molecule


def _equivalent(left: Any, right: Any) -> bool:
    _strict_molecule(left)
    _strict_molecule(right)
    return isomeric_graph_equivalent(left, right)


def _records_by_policy(origin: dict[str, Any]) -> dict[str, tuple[dict, dict]]:
    records = origin["bundle"]["records"]
    assert [(item["policy"], item["variant_label"]) for item in records] == list(
        LABEL_ORDER
    )
    return {
        policy: (records[index], records[index + 1])
        for index, policy in zip(range(0, 8, 2), POLICIES, strict=True)
    }


def test_frozen_artifacts_are_exact_public_api_replays_with_config_seed() -> None:
    """The same frozen config seed must rebuild both snapshots byte-for-byte."""

    config = load_config_bundle()
    global_seed = config.dataset.dataset.global_seed
    first = build_t025_golden_corpus(
        DEFAULT_DATASET_ROOT,
        global_seed=global_seed,
    )
    second = build_t025_golden_corpus(
        DEFAULT_DATASET_ROOT,
        global_seed=global_seed,
    )

    frozen_bundles = (PROJECT_ROOT / DEFAULT_GOLDEN_BUNDLE_PATH).read_text(
        encoding="utf-8"
    )
    frozen_validation = (PROJECT_ROOT / DEFAULT_GOLDEN_VALIDATION_PATH).read_text(
        encoding="utf-8"
    )
    assert first.global_seed == global_seed
    assert first.render_bundle_json() == second.render_bundle_json() == frozen_bundles
    assert (
        first.render_validation_json()
        == second.render_validation_json()
        == frozen_validation
    )


def test_recipe_seeds_follow_the_frozen_plan_formula() -> None:
    """Seeds bind dataset version and variant index as required by Plan section 8.5."""

    config = load_config_bundle()
    corpus = build_t025_golden_corpus(DEFAULT_DATASET_ROOT, config=config)
    for origin in corpus.origins:
        for execution in origin.executions:
            recipe = execution.context.recipe
            payload = "\0".join(
                (
                    str(config.dataset.dataset.global_seed),
                    config.dataset.dataset.version_name,
                    recipe.origin_id,
                    recipe.operator_id,
                    recipe.policy.dataset_name,
                    str(recipe.variant_index),
                )
            ).encode("utf-8")
            expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            assert recipe.derived_seed == expected


def test_direct_constructors_bind_origin_and_corpus_metadata() -> None:
    corpus = build_t025_golden_corpus(DEFAULT_DATASET_ROOT)
    real_origin = corpus.origins[0]
    with pytest.raises(ValueError, match="origin"):
        GoldenOriginBundle(
            spec=replace(real_origin.spec, origin_id="forged.add.origin"),
            bundle=real_origin.bundle,
            executions=real_origin.executions,
        )
    forged_execution = replace(
        real_origin.executions[0],
        candidate_engine_name="FakeEngine",
        propagation_engine_name="FakePropagation",
    )
    with pytest.raises(ValueError, match="policy spec"):
        GoldenOriginBundle(
            spec=real_origin.spec,
            bundle=real_origin.bundle,
            executions=(forged_execution, *real_origin.executions[1:]),
        )
    with pytest.raises(ValueError, match="frozen config"):
        GoldenCorpusBuild(
            dataset_version="forged-version",
            global_seed=corpus.global_seed,
            origins=corpus.origins,
        )
    with pytest.raises(ValueError, match="frozen config"):
        GoldenCorpusBuild(
            dataset_version=corpus.dataset_version,
            global_seed=corpus.global_seed + 1,
            origins=corpus.origins,
        )


def test_replay_calls_production_candidate_propagation_and_bundle_apis(
    monkeypatch,
) -> None:
    """Measured calls prevent trace strings from standing in for real execution."""

    calls: Counter[str] = Counter()

    def counted(original, key: str):
        def wrapper(self, *args, **kwargs):
            calls[key] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for engine_type in (
        AdditionCandidateEngine,
        DeletionCandidateEngine,
        SubstitutionCandidateEngine,
    ):
        engine_name = engine_type.__name__
        for method_name in ("enumerate_root_patches", "select_root_patch"):
            original = getattr(engine_type, method_name)
            monkeypatch.setattr(
                engine_type,
                method_name,
                counted(original, f"{engine_name}.{method_name}"),
            )
    original_propagate = EditingPropagationEngine.propagate
    monkeypatch.setattr(
        EditingPropagationEngine,
        "propagate",
        counted(original_propagate, "EditingPropagationEngine.propagate"),
    )
    original_build = MatchedBundleBuilder.build
    monkeypatch.setattr(
        MatchedBundleBuilder,
        "build",
        counted(original_build, "MatchedBundleBuilder.build"),
    )
    config = load_config_bundle()
    owner_by_subtask = {
        "add": AdditionPerturbator,
        "delete": DeletionPerturbator,
        "substitute": SubstitutionPerturbator,
    }
    expected_member_calls: set[str] = set()
    for origin_spec in T025_GOLDEN_ORIGINS:
        owner_type = owner_by_subtask[origin_spec.normalized_subtask.value]
        registry = PerturbatorRegistry.from_perturbator_types(
            (owner_type,),
            operators_config=config.operators,
        )
        for policy_spec in origin_spec.policies:
            registration = registry.registration(policy_spec.operator_id)
            method_name = registration.method_name
            key = f"{owner_type.__name__}.{method_name}"
            expected_member_calls.add(key)
            original = getattr(owner_type, method_name)

            @wraps(original)
            def counted_member(self, *args, _original=original, _key=key, **kwargs):
                calls[_key] += 1
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(owner_type, method_name, counted_member)

    corpus = build_t025_golden_corpus(DEFAULT_DATASET_ROOT)

    assert len(corpus.origins) == 3
    for engine_type in (
        AdditionCandidateEngine,
        DeletionCandidateEngine,
        SubstitutionCandidateEngine,
    ):
        assert calls[f"{engine_type.__name__}.enumerate_root_patches"] == 4
        assert calls[f"{engine_type.__name__}.select_root_patch"] == 4
    assert calls["EditingPropagationEngine.propagate"] == 12
    assert calls["MatchedBundleBuilder.build"] == 3
    assert len(expected_member_calls) == 12
    assert all(calls[key] == 1 for key in expected_member_calls)


def test_nested_validation_failure_cannot_be_hidden_by_top_level_pass(
    monkeypatch,
) -> None:
    def forged_validation(origin, _bundle, _executions):
        return {
            "origin_id": origin.origin_id,
            "normalized_subtask": origin.normalized_subtask.value,
            "all_pass": True,
            "chemistry": {
                "all_pass": True,
                "checks": [
                    {
                        "policy": policy,
                        "strict_product_parse": False,
                        "strict_answer_parse": True,
                        "selected_from_validated_candidate_pool": True,
                    }
                    for policy in POLICIES
                ],
            },
            "propagation": {
                "all_pass": True,
                "checks": [
                    {
                        "policy": policy,
                        "graph_delta_exact": False,
                        "exact_quota_bucket_verified": True,
                        "policy_shape_valid": True,
                    }
                    for policy in POLICIES
                ],
            },
            "bundle": {
                "all_pass": True,
                "record_count": 8,
                "hallucinated_count": 4,
                "faithful_count": 4,
                "unique_record_ids": 8,
                "reciprocal_pairs": True,
                "faithful_controls_exact": True,
                "unique_control_identities": True,
                "unique_render_identities": True,
                "hallucinated_states_bound_to_propagation": True,
            },
        }

    monkeypatch.setattr(
        golden_bundle_module,
        "_validate_origin",
        forged_validation,
    )
    with pytest.raises(ValueError, match="validation sections"):
        build_t025_golden_corpus(DEFAULT_DATASET_ROOT)


def test_partial_validation_checks_actual_changed_nodes_for_connectivity(
    monkeypatch,
) -> None:
    corpus = build_t025_golden_corpus(DEFAULT_DATASET_ROOT)
    expected_by_schema: dict[str, frozenset[str]] = {}
    for origin in corpus.origins:
        execution = next(
            item
            for item in origin.executions
            if item.context.recipe.policy is PropagationPolicy.PARTIAL
        )
        expected_by_schema[execution.context.state_schema.schema_id] = frozenset(
            target_id
            for target_kind, target_id in execution.outcome.candidate_graph.semantic_differences(
                execution.context.reference_graph
            )
            if target_kind is MutationTargetKind.NODE
        )

    observed_by_schema: dict[str, frozenset[str]] = {}
    original = StateSchema.is_connected_downstream_subgraph

    def audited_connectivity(self, roots, nodes):
        if self.schema_id in expected_by_schema:
            observed_by_schema[self.schema_id] = frozenset(nodes)
        return original(self, roots, nodes)

    monkeypatch.setattr(
        StateSchema,
        "is_connected_downstream_subgraph",
        audited_connectivity,
    )
    for origin in corpus.origins:
        golden_bundle_module._validate_origin(
            origin.spec,
            origin.bundle,
            origin.executions,
        )

    assert observed_by_schema == expected_by_schema


def test_selected_patches_and_full_actions_bind_to_locked_candidate_state() -> None:
    corpus = build_t025_golden_corpus(DEFAULT_DATASET_ROOT)
    for origin in corpus.origins:
        for execution in origin.executions:
            patch = execution.selected_patch
            state = execution.outcome.candidate_graph
            assert state.value_for(patch.root_node_id).semantically_equals(
                patch.new_value
            )
            if execution.context.recipe.policy is not PropagationPolicy.FULL_CF:
                continue
            assert patch.edit_action is not None
            replayed = replay_edit_action_from_source(
                execution.context.record.indexed_smiles,
                patch.edit_action,
            )
            assert len(replayed) == 1
            assert _equivalent(
                replayed[0],
                state.value_for("product").normalized_value,
            )


def test_forged_full_outcome_cannot_detach_selected_patch_and_action(
    monkeypatch,
) -> None:
    original = EditingPropagationEngine.propagate

    def forged_full(self, context, patch):
        outcome = original(self, context, patch)
        if not (
            context.record.normalized_subtask is EditingSubtask.ADD
            and context.recipe.policy is PropagationPolicy.FULL_CF
        ):
            return outcome
        fake_product = "CC"
        assert not _equivalent(
            patch.new_value.normalized_value,
            fake_product,
        )
        values = dict(context.reference_graph.values)
        product_claim = replace(
            outcome.candidate_graph.value_for("product"),
            raw_value=fake_product,
            normalized_value=fake_product,
        )
        answer_claim = replace(
            outcome.candidate_graph.value_for("final_answer"),
            raw_value=fake_product,
            normalized_value=fake_product,
        )
        values["product"] = product_claim
        values["final_answer"] = answer_claim
        events_by_target = {
            event.node_or_edge_id: event for event in outcome.graph_delta.events
        }
        return PropagationOutcome(
            candidate_graph=StateDAG(
                context.reference_graph.schema,
                values,
                context.reference_graph.edge_values,
            ),
            graph_delta=GraphDelta(
                (
                    replace(events_by_target["product"], after=product_claim),
                    replace(events_by_target["final_answer"], after=answer_claim),
                )
            ),
        )

    monkeypatch.setattr(EditingPropagationEngine, "propagate", forged_full)
    with pytest.raises(GoldenBundleBuildError) as captured:
        build_t025_golden_corpus(DEFAULT_DATASET_ROOT)
    assert captured.value.code == "GOLDEN_PROPAGATION_FAILED"


def test_three_real_subtasks_freeze_24_stably_matched_records() -> None:
    artifact = _frozen_json(DEFAULT_GOLDEN_BUNDLE_PATH)
    origins = artifact["origin_bundles"]

    assert artifact["global_seed"] == load_config_bundle().dataset.dataset.global_seed
    assert artifact["phase_boundary"] == (
        "semantic_matched_bundle_draft_before_t039_t040"
    )
    assert [item["normalized_subtask"] for item in origins] == [
        "add",
        "delete",
        "substitute",
    ]
    assert len({item["origin_id"] for item in origins}) == 3
    assert sum(len(item["bundle"]["records"]) for item in origins) == 24

    for origin in origins:
        records = origin["bundle"]["records"]
        pairs = _records_by_policy(origin)
        assert len(records) == 8
        assert len({item["record_id"] for item in records}) == 8
        assert len({item["render_identity"] for item in records}) == 8
        assert all(item["origin_id"] == origin["origin_id"] for item in records)
        assert all(
            item["bundle_id"] == origin["bundle"]["bundle_id"] for item in records
        )

        by_id = {item["record_id"]: item for item in records}
        faithful = []
        for policy, (hallucinated, control) in pairs.items():
            assert hallucinated["variant_label"] == "H"
            assert control["variant_label"] == "N"
            assert hallucinated["policy"] == control["policy"] == policy
            assert hallucinated["pair_id"] == control["pair_id"]
            assert hallucinated["control_identity"] == control["control_identity"]
            assert hallucinated["render_identity"] != control["render_identity"]
            assert hallucinated["matched_record_id"] == control["record_id"]
            assert control["matched_record_id"] == hallucinated["record_id"]
            assert by_id[hallucinated["matched_record_id"]] is control
            assert by_id[control["matched_record_id"]] is hallucinated
            assert control["graph_delta"] == []
            assert control["answer"]["product_equivalent"] is True
            faithful.append(control)

        assert len({item["pair_id"] for item in faithful}) == 4
        assert len({item["control_identity"] for item in faithful}) == 4
        assert all(
            item["state_values"] == faithful[0]["state_values"] for item in faithful
        )
        assert all(item["formal"] == faithful[0]["formal"] for item in faithful)
        assert all(item["answer"] == faithful[0]["answer"] for item in faithful)


def test_golden_traces_prove_real_candidates_and_t022_policy_shapes() -> None:
    artifact = _frozen_json(DEFAULT_GOLDEN_BUNDLE_PATH)
    report = _frozen_json(DEFAULT_GOLDEN_VALIDATION_PATH)
    report_by_origin = {item["origin_id"]: item for item in report["origins"]}

    for origin in artifact["origin_bundles"]:
        traces = origin["candidate_and_propagation_trace"]
        propagation_checks = {
            item["policy"]: item
            for item in report_by_origin[origin["origin_id"]]["propagation"]["checks"]
        }
        assert [item["policy"] for item in traces] == list(POLICIES)

        for trace in traces:
            policy = trace["policy"]
            pool = trace["candidate_pool"]
            selected = trace["selected_root_patch"]
            plan = trace["propagation_plan"]
            events = trace["mutation_events"]
            check = propagation_checks[policy]
            changed = set(check["semantic_difference_nodes"])
            event_targets = {item["target_id"] for item in events}
            root = selected["root_node_id"]

            assert (
                trace["candidate_engine"]
                == ENGINE_BY_SUBTASK[origin["normalized_subtask"]]
            )
            assert trace["propagation_engine"] == "EditingPropagationEngine"
            assert trace["selected_from_pool"] is True
            assert pool["accepted_count"] > 0
            assert pool["request_id"]
            assert pool["selected_rank"] == 0
            assert selected["candidate_id"]
            assert selected["candidate_source"] in {"RULE", "RDKIT", "HYBRID"}
            assert events
            assert event_targets == changed
            assert root == plan["selected_nodes"][0]
            assert root in changed
            assert all(item["operator_id"] == trace["operator_id"] for item in events)
            assert check["graph_delta_exact"] is True
            assert check["policy_shape_valid"] is True

            if policy == "LOCAL":
                assert plan["selected_nodes"] == [root]
                assert changed == {root}
                assert len(events) == 1
                assert events[0]["causal_role"] == "ROOT"
            elif policy == "PARTIAL":
                assert 1 < len(plan["selected_nodes"]) < len(plan["full_closure"])
                assert set(plan["selected_nodes"]) < set(plan["full_closure"])
                assert len(changed) > 1
                assert changed <= set(plan["selected_nodes"])
                expected_bucket = (
                    "product_dependency_cross_step"
                    if root == "product"
                    else "entity_partial_propagation"
                )
                assert trace["quota_bucket"] == expected_bucket
            elif policy == "FULL_CF":
                assert plan["selected_nodes"] == plan["full_closure"]
                assert selected["edit_action"] is not None
            else:
                assert plan["selected_nodes"] == ["final_answer"]
                assert changed == {"final_answer"}
                assert len(events) == 1
                assert events[0]["causal_role"] == "TERMINAL"
                assert pool["selected_answer_similarity"] > 0.0
                assert (
                    pool["selected_answer_similarity"] == pool["max_answer_similarity"]
                )


def test_full_and_terminal_chemistry_and_all_faithful_controls_are_sound() -> None:
    artifact = _frozen_json(DEFAULT_GOLDEN_BUNDLE_PATH)

    for origin in artifact["origin_bundles"]:
        pairs = _records_by_policy(origin)
        for hallucinated, control in pairs.values():
            reference_product = control["state_values"]["product"]
            reference_answer = control["state_values"]["final_answer"]
            _strict_molecule(reference_product)
            _strict_molecule(reference_answer)
            assert _equivalent(reference_answer, reference_product)
            assert control["answer"]["smiles"] == reference_answer

        full_h, full_n = pairs["FULL_CF"]
        candidate_product = full_h["state_values"]["product"]
        candidate_answer = full_h["state_values"]["final_answer"]
        _strict_molecule(candidate_product)
        _strict_molecule(candidate_answer)
        assert not _equivalent(candidate_product, full_n["state_values"]["product"])
        assert _equivalent(candidate_answer, candidate_product)
        assert full_h["answer"]["product_equivalent"] is True

        terminal_h, terminal_n = pairs["TERMINAL"]
        terminal_reasoning = {
            key: value
            for key, value in terminal_h["state_values"].items()
            if key != "final_answer"
        }
        reference_reasoning = {
            key: value
            for key, value in terminal_n["state_values"].items()
            if key != "final_answer"
        }
        assert terminal_reasoning == reference_reasoning
        assert terminal_h["formal"] == terminal_n["formal"]
        assert _equivalent(
            terminal_h["state_values"]["product"],
            terminal_n["state_values"]["product"],
        )
        assert not _equivalent(
            terminal_h["state_values"]["final_answer"],
            terminal_n["state_values"]["final_answer"],
        )
        assert not _equivalent(
            terminal_h["state_values"]["final_answer"],
            terminal_h["state_values"]["product"],
        )
        assert terminal_h["answer"]["product_equivalent"] is False


def test_frozen_validation_report_passes_every_t025_gate() -> None:
    report = _frozen_json(DEFAULT_GOLDEN_VALIDATION_PATH)

    assert report["all_pass"] is True
    assert report["global_seed"] == load_config_bundle().dataset.dataset.global_seed
    assert report["summary"] == {
        "candidate_engine_invocation_count": 12,
        "faithful_record_count": 12,
        "hallucinated_record_count": 12,
        "origin_bundle_count": 3,
        "propagation_engine_invocation_count": 12,
        "record_count": 24,
    }
    assert [item["normalized_subtask"] for item in report["origins"]] == [
        "add",
        "delete",
        "substitute",
    ]

    for origin in report["origins"]:
        assert origin["all_pass"] is True
        assert origin["chemistry"]["all_pass"] is True
        assert origin["propagation"]["all_pass"] is True
        bundle = origin["bundle"]
        assert bundle == {
            "all_pass": True,
            "faithful_controls_exact": True,
            "faithful_count": 4,
            "hallucinated_count": 4,
            "hallucinated_states_bound_to_propagation": True,
            "record_count": 8,
            "reciprocal_pairs": True,
            "unique_control_identities": True,
            "unique_record_ids": 8,
            "unique_render_identities": True,
        }
        assert len(origin["chemistry"]["checks"]) == 4
        assert len(origin["propagation"]["checks"]) == 4
        assert all(
            item["strict_product_parse"]
            and item["strict_answer_parse"]
            and item["selected_from_validated_candidate_pool"]
            and item["selected_candidate_rank"] == 0
            for item in origin["chemistry"]["checks"]
        )
        terminal_chemistry = next(
            item
            for item in origin["chemistry"]["checks"]
            if item["policy"] == "TERMINAL"
        )
        assert terminal_chemistry["selected_answer_similarity"] > 0.0
        assert (
            terminal_chemistry["selected_answer_similarity"]
            == terminal_chemistry["pool_max_answer_similarity"]
        )
        assert all(
            item["graph_delta_exact"]
            and item["exact_quota_bucket_verified"]
            and item["policy_shape_valid"]
            and item["selected_root_patch_bound_to_state"]
            for item in origin["propagation"]["checks"]
        )
        full_propagation = next(
            item
            for item in origin["propagation"]["checks"]
            if item["policy"] == "FULL_CF"
        )
        assert full_propagation["selected_action_replay_bound_to_product"] is True
