"""Trust-boundary tests for T031 Poe proposal and tool schemas."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.leakage_groups import assign_leakage_groups
from molhallulens.builders.origin_audit import audit_origin_split_features
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.builders.split_manifest import (
    VerifiedSplitManifest,
    load_verified_split_manifest,
)
from molhallulens.builders.splitter import (
    GroupStratifiedSplitter,
    split_origins_from_audit,
)
from molhallulens.candidates import CandidateRequest
from molhallulens.config import load_config_bundle
from molhallulens.domain import (
    CandidateSourceType,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    PerturbationContext,
    task_record_from_joined_input,
)
from molhallulens.perturbators.base import (
    CandidateEngine,
    LabelProjector,
    PropagationEngine,
    TraceRenderer,
    ValidatorChain,
)
from molhallulens.perturbators.editing.addition import ADDITION_OPERATOR_IDS
from molhallulens.perturbators.registry import PerturbatorRegistry
from molhallulens.providers.poe.schemas import (
    CHEMISTRY_TOOL_ARGUMENT_MODELS,
    CHEMISTRY_TOOL_NAMES,
    FROZEN_GLOBAL_SEED,
    AnalyzeSmilesArgs,
    ProposalCandidatePatch,
    ProposalConstraints,
    ProposalManifestIdentity,
    ProposalReplacement,
    ProposalRequest,
    ProposalResponse,
    chemistry_tool_call_json_schema,
    derive_proposal_seed,
    parse_chemistry_tool_call,
    parse_proposal_response,
    proposal_request_from_candidate_request,
    proposal_request_json_schema,
    proposal_response_json_schema,
    validate_chemistry_tool_arguments,
)
from molhallulens.validation import OriginValidationInput

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MANIFEST_PATH = PROJECT_ROOT / "HallucinationDataset" / "split_manifest.csv"
MANIFEST_METADATA_PATH = (
    PROJECT_ROOT / "HallucinationDataset" / "split_manifest.metadata.json"
)


class _CandidateEngine(CandidateEngine):
    def enumerate_root_patches(self, context):
        raise AssertionError("schema factory does not enumerate candidates")

    def select_root_patch(self, context, pool):
        raise AssertionError("schema factory does not select candidates")


class _PropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("schema factory does not propagate")


class _TraceRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("schema factory does not render")


class _ValidatorChain(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("schema factory does not validate artifacts")

    def validate_artifact(self, draft):
        raise AssertionError("schema factory does not validate artifacts")


class _LabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("schema factory does not project labels")


def _ports() -> dict[str, object]:
    return {
        "candidate_engine": _CandidateEngine(),
        "propagator": _PropagationEngine(),
        "renderer": _TraceRenderer(),
        "validators": _ValidatorChain(),
        "label_projector": _LabelProjector(),
    }


@cache
def _validated_inputs() -> tuple[OriginValidationInput, ...]:
    values = []
    for joined in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        artifact = build_reference_dag(joined)
        values.append(
            OriginValidationInput(
                record=joined,
                artifact=artifact,
                edit_truth=derive_edit_truth(artifact),
            )
        )
    return tuple(values)


@cache
def _factory_inputs() -> tuple[CandidateRequest, VerifiedSplitManifest]:
    items = _validated_inputs()
    audit = audit_origin_split_features(items).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: item.edit_truth.canonical_source_smiles
            for item in items
        },
    )
    split_result = GroupStratifiedSplitter().solve(
        split_origins_from_audit(audit, leakage)
    )
    verified_manifest = load_verified_split_manifest(
        MANIFEST_PATH,
        MANIFEST_METADATA_PATH,
        split_result=split_result,
        audit=audit,
    )
    item = next(
        value
        for value in items
        if value.edit_truth.anonymous_sample_id == "mol_edit.add_v2.0022"
    )
    operator_id = ADDITION_OPERATOR_IDS[2]
    policy = PropagationPolicy.STOP
    variant_index = 0
    recipe = PerturbationRecipe(
        recipe_id="poe:test:add:0022",
        origin_id=item.edit_truth.anonymous_sample_id,
        operator_id=operator_id,
        policy=policy,
        target_node_id="add_fragment",
        candidate_source_mode=CandidateSourceType.HYBRID,
        variant_index=variant_index,
        derived_seed=derive_proposal_seed(
            global_seed=FROZEN_GLOBAL_SEED,
            dataset_version="pilot_v1",
            origin_id=item.edit_truth.anonymous_sample_id,
            operator_id=operator_id,
            policy=policy.dataset_name,
            variant_index=variant_index,
        ),
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=32,
            length_bucket="proposal",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="formal-v1",
    )
    context = PerturbationContext(
        record=task_record_from_joined_input(item.record),
        recipe=recipe,
        state_schema=item.artifact.state_dag.schema,
        reference_graph=item.artifact.state_dag,
        truth=item.edit_truth,
    )
    perturbator = AdditionPerturbator(**_ports())
    registry = PerturbatorRegistry.from_perturbator_types(
        (AdditionPerturbator,), operators_config=load_config_bundle().operators
    )
    resolution = registry.resolve(perturbator, context)
    return CandidateRequest(context=context, resolution=resolution), verified_manifest


def _request(**updates: object) -> ProposalRequest:
    values: dict[str, object] = {
        "request_id": "request:substitute:0216",
        "origin_id": "mol_edit.substitute_v2.0216",
        "operator_id": "mol_edit.substitute.incoming_fragment_bucket_swap",
        "propagation": "FULL_CF",
        "candidate_source_mode": "HYBRID",
        "target_root": "add_fragment",
        "constraints": ProposalConstraints(
            same_attachment_element=True,
            match_heavy_count=True,
            match_ring_count=True,
        ),
        "global_seed": FROZEN_GLOBAL_SEED,
        "dataset_version": "pilot_v1",
        "variant_index": 0,
        "split": "train",
        "manifest_identity": ProposalManifestIdentity(
            dataset_version="pilot_v1",
            split_seed=8347206628578381721,
            manifest_sha256="a" * 64,
            source_origin_audit_sha256="b" * 64,
            source_split_report_sha256="c" * 64,
        ),
    }
    values.update(updates)
    values.setdefault(
        "derived_seed",
        derive_proposal_seed(
            global_seed=values["global_seed"],  # type: ignore[arg-type]
            dataset_version=values["dataset_version"],  # type: ignore[arg-type]
            origin_id=values["origin_id"],  # type: ignore[arg-type]
            operator_id=values["operator_id"],  # type: ignore[arg-type]
            policy=values["propagation"],  # type: ignore[arg-type]
            variant_index=values["variant_index"],  # type: ignore[arg-type]
        ),
    )
    return ProposalRequest.model_validate(values, strict=True)


def _candidate(candidate_id: str, *, root: str = "add_fragment") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "root_field": root,
        "replacement": {"smiles": "N1CCCC1", "attachment_atom": 0},
        "bond_edits": [],
        "minimal_surface_realization": "a pyrrolidin-1-yl group",
        "plausibility_reason": "Same fragment class and similar local size.",
    }


def _response_payload(
    *, count: int = 3, root: str = "add_fragment"
) -> dict[str, object]:
    return {
        "proposal_version": "1.0",
        "request_id": "request:substitute:0216",
        "candidates": [_candidate(f"c{index}", root=root) for index in range(count)],
        "abstain_reason": None,
    }


def _assert_all_objects_closed(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
        for value in schema.values():
            _assert_all_objects_closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_all_objects_closed(value)


def test_request_is_frozen_strict_closed_and_seed_bound() -> None:
    request = _request()
    assert request.target_root == "add_fragment"
    assert request.derived_seed == derive_proposal_seed(
        global_seed=FROZEN_GLOBAL_SEED,
        dataset_version="pilot_v1",
        origin_id=request.origin_id,
        operator_id=request.operator_id,
        policy="FULL_CF",
        variant_index=0,
    )
    with pytest.raises(ValidationError):
        _request(derived_seed=request.derived_seed + 1)
    with pytest.raises(ValidationError):
        ProposalRequest.model_validate(
            {**request.model_dump(), "unexpected": True}, strict=True
        )
    with pytest.raises(ValidationError):
        ProposalRequest.model_validate(
            {**request.model_dump(), "variant_index": True}, strict=True
        )
    with pytest.raises(ValidationError):
        ProposalRequest.model_validate(
            {**request.model_dump(), "target_root": ["add_fragment"]}, strict=True
        )
    with pytest.raises(ValidationError):
        ProposalRequest.model_validate(
            {**request.model_dump(), "candidate_source_mode": "RULE"}, strict=True
        )
    with pytest.raises(ValidationError):
        request.target_root = "product"  # type: ignore[misc]


def test_factory_binds_t017_request_to_verified_t029_manifest() -> None:
    candidate_request, verified_manifest = _factory_inputs()
    request = proposal_request_from_candidate_request(
        candidate_request,
        verified_manifest=verified_manifest,
    )
    row = verified_manifest.row_for_origin(request.origin_id)
    assert request.request_id == candidate_request.request_id
    assert request.operator_id == candidate_request.operator_id
    assert request.target_root == candidate_request.context.recipe.target_node_id
    assert request.candidate_source_mode == "HYBRID"
    assert request.split == row.split.value
    assert request.manifest_identity.manifest_sha256 == (
        verified_manifest.manifest_sha256
    )
    assert request.manifest_identity.source_origin_audit_sha256 == (
        verified_manifest.source_origin_audit_sha256
    )
    assert request.manifest_identity.source_split_report_sha256 == (
        verified_manifest.source_split_report_sha256
    )


def test_response_requires_three_to_five_unique_candidates_or_exact_abstention() -> (
    None
):
    request = _request()
    response = parse_proposal_response(json.dumps(_response_payload()), request=request)
    assert len(response.candidates) == 3
    assert all(item.root_field == request.target_root for item in response.candidates)

    for count in (1, 2, 6):
        with pytest.raises(ValidationError):
            ProposalResponse.model_validate(_response_payload(count=count))

    abstention = ProposalResponse(
        request_id=request.request_id,
        candidates=(),
        abstain_reason="No schema-valid candidate can satisfy the fixed constraints.",
    )
    assert not abstention.candidates
    with pytest.raises(ValidationError):
        ProposalResponse.model_validate(
            {
                "request_id": request.request_id,
                "candidates": [],
                "abstain_reason": None,
            }
        )
    with pytest.raises(ValidationError):
        ProposalResponse.model_validate(
            {**_response_payload(), "abstain_reason": "also abstaining"}
        )


def test_response_binds_request_id_single_root_and_rejects_labels() -> None:
    request = _request()
    wrong_request = _response_payload()
    wrong_request["request_id"] = "request:other"
    with pytest.raises(ValueError, match="request_id"):
        parse_proposal_response(wrong_request, request=request)
    with pytest.raises(ValueError, match="escape target_root"):
        parse_proposal_response(_response_payload(root="product"), request=request)

    accepted = _response_payload()
    accepted_candidates = accepted["candidates"]
    assert isinstance(accepted_candidates, list)
    accepted_candidates[0]["accepted"] = True  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_proposal_response(accepted, request=request)

    labelled = _response_payload()
    labelled_candidates = labelled["candidates"]
    assert isinstance(labelled_candidates, list)
    labelled_candidates[0]["hallucination_label"] = "CONTRADICTION"  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_proposal_response(labelled, request=request)


def test_replacement_has_one_primary_value_and_strict_attachment_index() -> None:
    assert ProposalReplacement(smiles="CC", attachment_atom=0).attachment_atom == 0
    with pytest.raises(ValidationError):
        ProposalReplacement()
    with pytest.raises(ValidationError):
        ProposalReplacement(smiles="CC", integer=2)
    with pytest.raises(ValidationError):
        ProposalReplacement(text="fragment", attachment_atom=0)
    with pytest.raises(ValidationError):
        ProposalReplacement(smiles="CC", attachment_atom=True)
    with pytest.raises(ValidationError):
        ProposalReplacement(smiles="CC", attachment_atom=-1)


def test_candidate_and_nested_models_are_frozen() -> None:
    candidate = ProposalCandidatePatch.model_validate_json(
        json.dumps(_candidate("c1")), strict=True
    )
    assert isinstance(candidate.bond_edits, tuple)
    with pytest.raises(ValidationError):
        candidate.replacement.smiles = "CO"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        candidate.candidate_id = "c2"  # type: ignore[misc]


def test_all_emitted_json_schema_objects_forbid_additional_properties() -> None:
    for schema in (
        proposal_request_json_schema(),
        proposal_response_json_schema(),
        chemistry_tool_call_json_schema(),
    ):
        _assert_all_objects_closed(schema)


def test_fixed_tool_allowlist_has_nine_independent_argument_models() -> None:
    assert CHEMISTRY_TOOL_NAMES == (
        "inspect_atoms",
        "enumerate_alternate_anchors",
        "analyze_smiles",
        "find_group_at_anchor",
        "enumerate_removable_groups",
        "simulate_edit",
        "compute_descriptors",
        "compare_molecules",
        "check_candidate_signature",
    )
    assert len(set(CHEMISTRY_TOOL_ARGUMENT_MODELS.values())) == 9

    calls = (
        {"tool": "inspect_atoms", "arguments": {"smiles": "CCO", "atom_indices": [0]}},
        {
            "tool": "enumerate_alternate_anchors",
            "arguments": {"source_smiles": "CCO", "reference_anchor_idx": 1},
        },
        {"tool": "analyze_smiles", "arguments": {"smiles": "CCO"}},
        {
            "tool": "find_group_at_anchor",
            "arguments": {"source_smiles": "CCBr", "anchor_idx": 2},
        },
        {
            "tool": "enumerate_removable_groups",
            "arguments": {"source_smiles": "CCBr", "anchor_idx": 2},
        },
        {
            "tool": "simulate_edit",
            "arguments": {
                "family": "substitute",
                "source_smiles": "CCBr",
                "anchor_idx": 2,
                "remove_group_smiles": "Br",
                "add_fragment_smiles": "N1CCCC1",
                "fragment_attachment_atom": 0,
                "bond_type": "SINGLE",
            },
        },
        {"tool": "compute_descriptors", "arguments": {"smiles": "CCO"}},
        {
            "tool": "compare_molecules",
            "arguments": {"left_smiles": "CCO", "right_smiles": "OCC"},
        },
        {
            "tool": "check_candidate_signature",
            "arguments": {
                "family": "add",
                "source_smiles": "CC",
                "candidate_product_smiles": "CCO",
                "anchor_idx": 2,
                "add_fragment_smiles": "O",
                "fragment_attachment_atom": 0,
                "bond_type": "SINGLE",
            },
        },
    )
    assert (
        tuple(parse_chemistry_tool_call(json.dumps(call)).tool for call in calls)
        == CHEMISTRY_TOOL_NAMES
    )


def test_tool_dispatch_rejects_unknown_extra_and_coerced_arguments() -> None:
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call({"tool": "unknown", "arguments": {}})
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "analyze_smiles",
                "arguments": {"smiles": "CC", "shell": "echo unsafe"},
            }
        )
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "find_group_at_anchor",
                "arguments": {"source_smiles": "CCBr", "anchor_idx": True},
            }
        )
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "find_group_at_anchor",
                "arguments": {"source_smiles": "CCBr", "anchor_idx": 0},
            }
        )
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "simulate_edit",
                "arguments": {
                    "family": "substitute",
                    "source_smiles": "CCBr",
                    "anchor_idx": 2,
                    "remove_group_smiles": "Br",
                    "add_fragment_smiles": "N",
                    "fragment_attachment_atom": 0,
                    "bond_type": "QUADRUPLE",
                },
            }
        )
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "simulate_edit",
                "arguments": {
                    "family": "optimize",
                    "source_smiles": "CC",
                    "anchor_idx": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="unknown chemistry tool"):
        validate_chemistry_tool_arguments("shell", {})


def test_tool_edit_shapes_are_family_closed_before_execution() -> None:
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "simulate_edit",
                "arguments": {
                    "family": "add",
                    "source_smiles": "CC",
                    "anchor_idx": 1,
                    "remove_group_smiles": "Br",
                    "add_fragment_smiles": "N",
                    "fragment_attachment_atom": 0,
                    "bond_type": "SINGLE",
                },
            }
        )
    with pytest.raises(ValidationError):
        parse_chemistry_tool_call(
            {
                "tool": "simulate_edit",
                "arguments": {
                    "family": "delete",
                    "source_smiles": "CCBr",
                    "anchor_idx": 2,
                    "remove_group_smiles": "Br",
                    "bond_type": "SINGLE",
                },
            }
        )


def test_argument_model_direct_validation_is_strict_and_frozen() -> None:
    model = validate_chemistry_tool_arguments("analyze_smiles", {"smiles": "CC"})
    assert type(model) is AnalyzeSmilesArgs
    with pytest.raises(ValidationError):
        model.smiles = "CO"  # type: ignore[misc]
