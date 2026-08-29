"""T018 deterministic candidate-source and pool-normalization contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import (
    EditTruthBuilder,
    build_reference_dag,
    derive_edit_truth,
)
from molhallulens.candidates import (
    CandidateBuildResult,
    CandidateDifficultyFeatures,
    CandidateProposal,
    CandidateRejectCode,
    CandidateRequest,
    DeterministicCandidateEngine,
    RankedCandidate,
    RDKitCandidateSource,
    RuleCandidateSource,
    canonical_candidate_key,
    rank_candidates,
)
from molhallulens.config import load_config_bundle
from molhallulens.domain import (
    BondTypeName,
    CandidatePatch,
    CandidateSourceType,
    ClaimValue,
    EditAction,
    EditErrorSubtype,
    EditKind,
    HallucinationType,
    OperatorCapability,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    ValueProvenance,
    ValueType,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    CandidateEngine,
    DeletionPerturbator,
    LabelProjector,
    PerturbationContext,
    PerturbatorRegistry,
    PropagationEngine,
    TraceRenderer,
    ValidatorChain,
    operator,
    task_record_from_validated_reference,
)
from molhallulens.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATOR_ID = "mol_edit.add.t018_candidate_contract"
DELETION_OPERATOR_ID = "mol_edit.delete.t018_candidate_contract"


class _UnusedCandidateEngine(CandidateEngine):
    def enumerate_root_patches(self, context):
        raise AssertionError(
            "T018 tests exercise DeterministicCandidateEngine directly"
        )

    def select_root_patch(self, context, pool):
        raise AssertionError("T018 tests do not exercise the T016 template")


class _UnusedPropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T018 tests do not execute T022 propagation")


class _UnusedRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T018 tests do not execute rendering")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T018 tests do not execute the full perturbator")

    def validate_artifact(self, draft):
        raise AssertionError("T018 tests do not execute the full perturbator")


class _UnusedLabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T018 tests do not project token labels")


def _ports() -> dict[str, object]:
    return {
        "candidate_engine": _UnusedCandidateEngine(),
        "propagator": _UnusedPropagationEngine(),
        "renderer": _UnusedRenderer(),
        "validators": _UnusedValidators(),
        "label_projector": _UnusedLabelProjector(),
    }


class _CandidateContractAddition(AdditionPerturbator):
    @operator(
        operator_id=OPERATOR_ID,
        operator_family="wrong_anchor_site",
        root_fields={"anchor_idx", "add_fragment", "product"},
        supported_policies={PropagationPolicy.STOP},
        supported_sources={
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
            CandidateSourceType.HYBRID,
        },
        hallucination_types={HallucinationType.CONTRADICTION},
        edit_subtypes={EditErrorSubtype.ANCHOR_GROUNDING},
        required_capabilities={OperatorCapability.CLAIM_PERTURBATION},
    )
    def t018_contract(self, context):
        raise AssertionError("T019 concrete operators are outside T018 tests")


class _CandidateContractDeletion(DeletionPerturbator):
    @operator(
        operator_id=DELETION_OPERATOR_ID,
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies={PropagationPolicy.STOP},
        supported_sources={
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
            CandidateSourceType.HYBRID,
        },
        hallucination_types={HallucinationType.CONTRADICTION},
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities={OperatorCapability.STRUCTURAL_DELETION},
    )
    def t018_contract(self, context):
        raise AssertionError("T020 concrete operators are outside T018 tests")


@cache
def _validated_addition() -> OriginValidationInput:
    joined = next(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".add_v2." in record.anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    truth = derive_edit_truth(artifact)
    return OriginValidationInput(record=joined, artifact=artifact, edit_truth=truth)


@cache
def _validated_deletion() -> OriginValidationInput:
    joined = next(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".delete_v2." in record.anonymous_sample_id
        and not record.anonymous_sample_id.endswith(".0081")
    )
    artifact = build_reference_dag(joined)
    truth = derive_edit_truth(artifact)
    return OriginValidationInput(record=joined, artifact=artifact, edit_truth=truth)


def _claim(value: Any, value_type: ValueType) -> ClaimValue:
    return ClaimValue(
        raw_value=value,
        normalized_value=value,
        value_type=value_type,
        provenance=ValueProvenance.REFERENCE,
    )


def _request(
    target_node_id: str,
    *,
    source: CandidateSourceType = CandidateSourceType.HYBRID,
    reference_value: Any | None = None,
) -> CandidateRequest:
    item = _validated_addition()
    record = task_record_from_validated_reference(item)
    mapped = tuple(
        sorted(
            pair.source.atom_id
            for pair in item.edit_truth.mapping_evidence.optimal_mappings[0].pairs
        )
    )
    anchors = mapped[:3]
    assert len(anchors) == 3
    truth = replace(
        item.edit_truth,
        valid_anchor_indices=anchors,
        symmetry_equivalent_anchors=((anchors[0], anchors[1]),),
    )
    values = dict(item.artifact.state_dag.values)
    if target_node_id == "anchor_idx":
        default_value, value_type = truth.valid_anchor_indices[0], ValueType.ATOM_INDEX
    elif target_node_id == "add_fragment":
        default_value, value_type = "CN", ValueType.FRAGMENT
    elif target_node_id == "product":
        default_value, value_type = "CCO", ValueType.SMILES
    else:  # pragma: no cover - helper is intentionally closed to three real roots
        raise AssertionError(target_node_id)
    values[target_node_id] = _claim(
        default_value if reference_value is None else reference_value,
        value_type,
    )
    reference_graph = StateDAG(
        schema=item.artifact.state_dag.schema,
        values=values,
        edge_values=item.artifact.state_dag.edge_values,
    )
    recipe = PerturbationRecipe(
        recipe_id=f"recipe:{target_node_id}:{source.value}",
        origin_id=record.origin_id,
        operator_id=OPERATOR_ID,
        policy=PropagationPolicy.STOP,
        target_node_id=target_node_id,
        candidate_source_mode=source,
        variant_index=0,
        derived_seed=23,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=32,
            length_bucket="fixture",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
    )
    context = PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=reference_graph.schema,
        reference_graph=reference_graph,
        truth=truth,
    )
    registry = PerturbatorRegistry.from_perturbator_types(
        (_CandidateContractAddition,),
        operators_config=load_config_bundle().operators,
    )
    resolution = registry.resolve(_CandidateContractAddition(**_ports()), context)
    return CandidateRequest(context=context, resolution=resolution)


@cache
def _action_replay_request() -> CandidateRequest:
    """Return a compact, internally coherent addition context for graph replay."""

    request = _request("product")
    source = "[NH2:2][CH2:1][CH2:3][OH:4]"
    reference_product = "CNCCO"
    truth = EditTruthBuilder().derive(
        source,
        reference_product,
        anonymous_sample_id=request.context.record.origin_id,
        normalized_subtask=request.context.record.normalized_subtask,
        trace_anchor_indices=(2,),
        add_fragment_hint="C",
    )
    assert truth.valid_anchor_indices == (2,)
    assert truth.add_fragment is not None

    values = dict(request.context.reference_graph.values)
    replacements = {
        "source": _claim(source, ValueType.INDEXED_SMILES),
        "oracle_gt": _claim(reference_product, ValueType.SMILES),
        "anchor_idx": _claim(2, ValueType.ATOM_INDEX),
        "anchor_element": _claim("N", ValueType.ELEMENT),
        "add_fragment": _claim("C", ValueType.FRAGMENT),
        "fragment_heavy": _claim(1, ValueType.COUNT),
        "product": _claim(reference_product, ValueType.SMILES),
        "source_heavy": _claim(
            truth.source_descriptors.heavy_atom_count,
            ValueType.COUNT,
        ),
        "product_heavy": _claim(
            truth.product_descriptors.heavy_atom_count,
            ValueType.COUNT,
        ),
        "heavy_delta": _claim(truth.heavy_atom_delta, ValueType.INTEGER),
        "source_rings": _claim(
            truth.source_descriptors.ring_count,
            ValueType.COUNT,
        ),
        "product_rings": _claim(
            truth.product_descriptors.ring_count,
            ValueType.COUNT,
        ),
        "ring_delta": _claim(
            truth.product_descriptors.ring_count
            - truth.source_descriptors.ring_count,
            ValueType.INTEGER,
        ),
        "final_answer": _claim(reference_product, ValueType.SMILES),
        "oracle_anchor_element": _claim("N", ValueType.ELEMENT),
        "oracle_fragment_heavy": _claim(1, ValueType.COUNT),
        "oracle_source_heavy": _claim(
            truth.source_descriptors.heavy_atom_count,
            ValueType.COUNT,
        ),
        "oracle_product_heavy": _claim(
            truth.product_descriptors.heavy_atom_count,
            ValueType.COUNT,
        ),
        "oracle_source_rings": _claim(
            truth.source_descriptors.ring_count,
            ValueType.COUNT,
        ),
        "oracle_product_rings": _claim(
            truth.product_descriptors.ring_count,
            ValueType.COUNT,
        ),
    }
    values.update({key: value for key, value in replacements.items() if key in values})
    reference_graph = StateDAG(
        schema=request.context.reference_graph.schema,
        values=values,
        edge_values=request.context.reference_graph.edge_values,
    )
    context = replace(
        request.context,
        record=replace(
            request.context.record,
            indexed_smiles=source,
            gt_smiles=reference_product,
            reference_final_answer=reference_product,
        ),
        reference_graph=reference_graph,
        truth=truth,
    )
    return CandidateRequest(context=context, resolution=request.resolution)


@cache
def _deletion_request() -> CandidateRequest:
    item = _validated_deletion()
    record = task_record_from_validated_reference(item)
    recipe = PerturbationRecipe(
        recipe_id="recipe:delete:product:hybrid",
        origin_id=record.origin_id,
        operator_id=DELETION_OPERATOR_ID,
        policy=PropagationPolicy.STOP,
        target_node_id="product",
        candidate_source_mode=CandidateSourceType.HYBRID,
        variant_index=0,
        derived_seed=29,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=32,
            length_bucket="fixture",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
    )
    context = PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=item.artifact.state_dag.schema,
        reference_graph=item.artifact.state_dag,
        truth=item.edit_truth,
    )
    registry = PerturbatorRegistry.from_perturbator_types(
        (_CandidateContractDeletion,),
        operators_config=load_config_bundle().operators,
    )
    resolution = registry.resolve(_CandidateContractDeletion(**_ports()), context)
    return CandidateRequest(context=context, resolution=resolution)


@cache
def _deletion_action_replay_request() -> CandidateRequest:
    request = _deletion_request()
    source = "[CH3:1][O:2][CH2:3][CH3:4]"
    reference_product = "COC"
    truth = EditTruthBuilder().derive(
        source,
        reference_product,
        anonymous_sample_id=request.context.record.origin_id,
        normalized_subtask=request.context.record.normalized_subtask,
        trace_anchor_indices=(3,),
        remove_fragment_hint="C",
    )
    values = dict(request.context.reference_graph.values)
    replacements = {
        "source": _claim(source, ValueType.INDEXED_SMILES),
        "oracle_gt": _claim(reference_product, ValueType.SMILES),
        "anchor_idx": _claim(3, ValueType.ATOM_INDEX),
        "anchor_element": _claim("C", ValueType.ELEMENT),
        "remove_group_step1": _claim("C", ValueType.FRAGMENT),
        "remove_group_step2": _claim("C", ValueType.FRAGMENT),
        "remove_heavy": _claim(1, ValueType.COUNT),
        "product": _claim(reference_product, ValueType.SMILES),
        "source_heavy": _claim(4, ValueType.COUNT),
        "product_heavy": _claim(3, ValueType.COUNT),
        "heavy_delta": _claim(-1, ValueType.INTEGER),
        "source_rings": _claim(0, ValueType.COUNT),
        "product_rings": _claim(0, ValueType.COUNT),
        "ring_delta": _claim(0, ValueType.INTEGER),
        "final_answer": _claim(reference_product, ValueType.SMILES),
        "oracle_anchor_element": _claim("C", ValueType.ELEMENT),
        "oracle_remove_heavy": _claim(1, ValueType.COUNT),
        "oracle_source_heavy": _claim(4, ValueType.COUNT),
        "oracle_product_heavy": _claim(3, ValueType.COUNT),
        "oracle_source_rings": _claim(0, ValueType.COUNT),
        "oracle_product_rings": _claim(0, ValueType.COUNT),
    }
    values.update({key: value for key, value in replacements.items() if key in values})
    reference_graph = StateDAG(
        schema=request.context.reference_graph.schema,
        values=values,
        edge_values=request.context.reference_graph.edge_values,
    )
    context = replace(
        request.context,
        record=replace(
            request.context.record,
            indexed_smiles=source,
            gt_smiles=reference_product,
            reference_final_answer=reference_product,
        ),
        reference_graph=reference_graph,
        truth=truth,
    )
    return CandidateRequest(context=context, resolution=request.resolution)


def _edit_action(
    *,
    anchor: int,
    add_fragment_smiles: str = "CN",
    attachment_atom: int | None = 0,
    bond_type: BondTypeName | None = BondTypeName.SINGLE,
    metadata: dict[str, object] | None = None,
) -> EditAction:
    return EditAction(
        edit_kind=EditKind.ADDITION,
        source_anchor_index=anchor,
        add_fragment_smiles=add_fragment_smiles,
        fragment_attachment_atom=attachment_atom,
        bond_type=bond_type,
        metadata=metadata or {},
    )


def _proposal(
    request: CandidateRequest,
    proposal_id: str,
    new_value: Any,
    *,
    source: CandidateSourceType = CandidateSourceType.RULE,
    edit_action: EditAction | None = None,
    candidate_product_smiles: str | None = None,
    difficulty_features: CandidateDifficultyFeatures | None = None,
) -> CandidateProposal:
    root = request.context.recipe.target_node_id
    old = request.context.reference_graph.values[root]
    new = ClaimValue(
        raw_value=new_value,
        normalized_value=new_value,
        value_type=old.value_type,
        provenance=(
            ValueProvenance.RDKIT
            if source is CandidateSourceType.RDKIT
            else ValueProvenance.RULE
        ),
    )
    return CandidateProposal(
        proposal_id=proposal_id,
        patch=CandidatePatch(
            candidate_id=f"candidate:{proposal_id}",
            root_node_id=root,
            old_value=old,
            new_value=new,
            edit_action=edit_action,
            source=source,
        ),
        candidate_product_smiles=candidate_product_smiles,
        difficulty_features=difficulty_features or CandidateDifficultyFeatures(),
    )


def _build(
    request: CandidateRequest,
    proposals: tuple[CandidateProposal, ...],
    *,
    source_type: CandidateSourceType = CandidateSourceType.RULE,
) -> CandidateBuildResult:
    source = (
        RuleCandidateSource(lambda _: proposals)
        if source_type is CandidateSourceType.RULE
        else RDKitCandidateSource(lambda _: proposals)
    )
    return DeterministicCandidateEngine((source,)).build_pool(request)


def test_canonical_key_uses_semantic_root_and_the_complete_edit_action() -> None:
    request = _request("product")
    anchor = request.context.truth.valid_anchor_indices[-1]
    base = _proposal(
        request,
        "base",
        "CCN",
        edit_action=_edit_action(
            anchor=anchor,
            add_fragment_smiles="CN",
            attachment_atom=0,
            bond_type=BondTypeName.SINGLE,
            metadata={"orientation": "forward"},
        ),
        candidate_product_smiles="CCN",
    )
    same_semantics = _proposal(
        request,
        "different-id-and-source",
        "NCC",
        source=CandidateSourceType.RDKIT,
        edit_action=_edit_action(
            anchor=anchor,
            add_fragment_smiles="NC",
            attachment_atom=1,
            bond_type=BondTypeName.SINGLE,
            metadata={"orientation": "forward"},
        ),
        candidate_product_smiles="NCC",
    )

    assert canonical_candidate_key(base) == canonical_candidate_key(same_semantics)

    changed_semantics = (
        replace(
            base,
            patch=replace(
                base.patch,
                edit_action=replace(
                    base.patch.edit_action, source_anchor_index=anchor + 1
                ),
            ),
        ),
        replace(
            base,
            patch=replace(
                base.patch,
                edit_action=replace(base.patch.edit_action, fragment_attachment_atom=1),
            ),
        ),
        replace(
            base,
            patch=replace(
                base.patch,
                edit_action=replace(
                    base.patch.edit_action, bond_type=BondTypeName.DOUBLE
                ),
            ),
        ),
        replace(
            base,
            patch=replace(
                base.patch,
                edit_action=replace(base.patch.edit_action, add_fragment_smiles="CO"),
            ),
        ),
    )
    base_key = canonical_candidate_key(base)
    assert all(canonical_candidate_key(item) != base_key for item in changed_semantics)


def test_attachment_atom_and_bond_semantics_survive_pool_normalization() -> None:
    request = _action_replay_request()
    proposals = (
        _proposal(
            request,
            "attachment-0-single",
            "NCCOCO",
            edit_action=_edit_action(
                anchor=4,
                add_fragment_smiles="CO",
                attachment_atom=0,
            ),
            candidate_product_smiles="NCCOCO",
        ),
        _proposal(
            request,
            "attachment-1-single",
            "NCCOOC",
            edit_action=_edit_action(
                anchor=4,
                add_fragment_smiles="CO",
                attachment_atom=1,
            ),
            candidate_product_smiles="NCCOOC",
        ),
        _proposal(
            request,
            "attachment-0-double",
            "NCC(=O)O",
            edit_action=_edit_action(
                anchor=3,
                add_fragment_smiles="O",
                attachment_atom=0,
                bond_type=BondTypeName.DOUBLE,
            ),
            candidate_product_smiles="NCC(=O)O",
        ),
        _proposal(
            request,
            "attachment-0-single-reverse",
            "NCC(O)O",
            edit_action=_edit_action(
                anchor=3,
                add_fragment_smiles="O",
                attachment_atom=0,
                metadata={"orientation": "reverse"},
            ),
            candidate_product_smiles="NCC(O)O",
        ),
    )

    result = _build(request, proposals)

    assert len(result.pool.candidates) == 4
    assert not result.rejections
    assert {
        (
            candidate.edit_action.fragment_attachment_atom,
            candidate.edit_action.bond_type,
        )
        for candidate in result.pool.candidates
        if candidate.edit_action is not None
    } == {
        (0, BondTypeName.SINGLE),
        (1, BondTypeName.SINGLE),
        (0, BondTypeName.DOUBLE),
    }
    assert {
        candidate.edit_action.metadata.get("orientation")
        for candidate in result.pool.candidates
    } == {
        None,
        "reverse",
    }


def test_symmetry_equivalent_anchor_is_rejected_but_distinct_site_survives() -> None:
    request = _request("anchor_idx")
    reference, symmetric, *_ = request.context.truth.valid_anchor_indices
    distinct = max(request.context.truth.valid_anchor_indices) + 100
    assert symmetric in request.context.truth.valid_anchor_indices
    assert distinct not in request.context.truth.valid_anchor_indices
    assert (
        request.context.reference_graph.values["anchor_idx"].normalized_value
        == reference
    )
    # Deliberately remove the mapping-derived symmetry ledger.  The source graph
    # still makes its two terminal carbons automorphic while the centre carbon is
    # distinct, so acceptance cannot depend only on EditTruth's cached groups.
    request = replace(
        request,
        context=replace(
            request.context,
            record=replace(
                request.context.record,
                indexed_smiles=(f"[CH3:{reference}][CH2:{distinct}][CH3:{symmetric}]"),
            ),
            truth=replace(
                request.context.truth,
                symmetry_equivalent_anchors=(),
            ),
        ),
    )
    proposals = (
        _proposal(
            request,
            "symmetric",
            symmetric,
            edit_action=None,
            candidate_product_smiles="CCN",
        ),
        _proposal(
            request,
            "distinct",
            distinct,
            edit_action=None,
            candidate_product_smiles="CCN",
        ),
    )

    result = _build(request, proposals)

    assert tuple(item.candidate_id for item in result.pool.candidates) == (
        "candidate:distinct",
    )
    assert {(item.proposal_id, item.code) for item in result.rejections} == {
        ("symmetric", CandidateRejectCode.SYMMETRY_EQUIVALENT),
    }


def test_strict_sanitize_and_root_comparator_reject_invalid_or_equivalent_smiles() -> (
    None
):
    request = _request("add_fragment", reference_value="CCO")
    proposals = (
        _proposal(
            request,
            "invalid",
            "C1(",
            edit_action=None,
            candidate_product_smiles=None,
        ),
        _proposal(
            request,
            "equivalent",
            "OCC",
            edit_action=None,
            candidate_product_smiles=None,
        ),
        _proposal(
            request,
            "valid",
            "CCN",
            edit_action=None,
            candidate_product_smiles=None,
        ),
    )

    result = _build(request, proposals)

    assert tuple(item.candidate_id for item in result.pool.candidates) == (
        "candidate:valid",
    )
    assert {item.proposal_id: item.code for item in result.rejections} == {
        "equivalent": CandidateRejectCode.REFERENCE_EQUIVALENT,
        "invalid": CandidateRejectCode.SMILES_INVALID,
    }
    assert result.pool.rejection_codes == tuple(
        sorted(code.value for code in {item.code for item in result.rejections})
    )


def test_isomeric_comparator_does_not_collapse_opposite_stereochemistry() -> None:
    request = _request("add_fragment", reference_value="F[C@H](Cl)Br")
    opposite = _proposal(
        request,
        "opposite-stereo",
        "F[C@@H](Cl)Br",
        edit_action=None,
        candidate_product_smiles=None,
    )

    result = _build(request, (opposite,))

    assert tuple(item.candidate_id for item in result.pool.candidates) == (
        "candidate:opposite-stereo",
    )
    assert not result.rejections


def test_canonical_duplicates_are_collapsed_with_per_proposal_rejection() -> None:
    request = _request("add_fragment")
    proposals = (
        _proposal(
            request,
            "z-noncanonical",
            "NCC",
            edit_action=None,
            candidate_product_smiles=None,
        ),
        _proposal(
            request,
            "a-canonical",
            "CCN",
            edit_action=None,
            candidate_product_smiles=None,
        ),
    )

    forward = _build(request, proposals)
    reverse = _build(request, tuple(reversed(proposals)))

    assert forward == reverse
    assert len(forward.pool.candidates) == 1
    assert len(forward.ranked_candidates) == 1
    assert forward.ranked_candidates[0].canonical_product_smiles is None
    assert tuple(item.code for item in forward.rejections) == (
        CandidateRejectCode.DUPLICATE,
    )
    accepted_id = forward.ranked_candidates[0].proposal.proposal_id
    assert {accepted_id, forward.rejections[0].proposal_id} == {
        "a-canonical",
        "z-noncanonical",
    }
    assert forward.rejections[0].evidence == {
        "duplicate_of": accepted_id,
        "duplicate_scope": "semantic",
    }


def test_structural_contract_failures_have_stable_per_proposal_codes() -> None:
    request = _request("product")
    anchor = request.context.truth.valid_anchor_indices[-1]
    proposals = (
        _proposal(
            request,
            "missing-product",
            "CCN",
            edit_action=_edit_action(anchor=anchor),
            candidate_product_smiles=None,
        ),
        _proposal(
            request,
            "missing-attachment",
            "CCN",
            edit_action=_edit_action(
                anchor=anchor,
                attachment_atom=None,
                bond_type=None,
            ),
            candidate_product_smiles="CCN",
        ),
    )

    result = _build(request, proposals)

    assert not result.pool.candidates
    assert {item.proposal_id: item.code for item in result.rejections} == {
        "missing-attachment": CandidateRejectCode.ATTACHMENT_SEMANTICS_MISSING,
        "missing-product": CandidateRejectCode.STRUCTURAL_PRODUCT_MISSING,
    }
    assert all(item.operator_id == OPERATOR_ID for item in result.rejections)
    assert all(item.evidence for item in result.rejections)
    with pytest.raises(TypeError):
        result.rejections[0].evidence["mutate"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.rejections[0].proposal_id = "changed"  # type: ignore[misc]


def test_rule_and_rdkit_sources_fail_closed_on_patch_source_mismatch() -> None:
    request = _request("product")
    anchor = request.context.truth.valid_anchor_indices[-1]
    rdkit_patch_from_rule_source = _proposal(
        request,
        "wrong-source",
        "CCN",
        source=CandidateSourceType.RDKIT,
        edit_action=_edit_action(anchor=anchor),
        candidate_product_smiles="CCN",
    )

    result = _build(request, (rdkit_patch_from_rule_source,))

    assert not result.pool.candidates
    assert tuple(item.code for item in result.rejections) == (
        CandidateRejectCode.SOURCE_MISMATCH,
    )
    assert result.rejections[0].proposal_id == "wrong-source"


def test_difficulty_ranking_and_final_tie_break_are_input_order_independent() -> None:
    request = _request("product")
    anchor = request.context.truth.valid_anchor_indices[-1]
    tied = CandidateDifficultyFeatures(
        structural_similarity=0.8,
        heavy_atom_delta=0,
        ring_delta=0,
        formal_charge_delta=0,
        heteroatom_l1_distance=1,
        anchor_element_match=True,
        anchor_aromaticity_match=True,
        anchor_degree_match=True,
        anchor_hybridization_match=True,
        source_score=0.5,
    )
    proposals = (
        _proposal(
            request,
            "candidate-z",
            "CCCl",
            edit_action=_edit_action(anchor=anchor, add_fragment_smiles="Cl"),
            candidate_product_smiles="CCCl",
            difficulty_features=tied,
        ),
        _proposal(
            request,
            "candidate-a",
            "CCBr",
            edit_action=_edit_action(anchor=anchor, add_fragment_smiles="Br"),
            candidate_product_smiles="CCBr",
            difficulty_features=tied,
        ),
    )

    forward = rank_candidates(request, proposals)
    reverse = rank_candidates(request, tuple(reversed(proposals)))

    assert forward == reverse
    assert all(type(item) is RankedCandidate for item in forward)
    assert tuple(item.rank_key for item in forward) == tuple(
        sorted(item.rank_key for item in forward)
    )
    assert tuple(item.proposal.proposal_id for item in forward) == tuple(
        item.proposal.proposal_id for item in reverse
    )


def test_derived_seed_stably_controls_tied_candidate_order() -> None:
    request = _request("add_fragment")
    tied = CandidateDifficultyFeatures(source_score=0.5)
    proposals = (
        _proposal(
            request,
            "seeded-ether",
            "CO",
            edit_action=None,
            candidate_product_smiles=None,
            difficulty_features=tied,
        ),
        _proposal(
            request,
            "seeded-thioether",
            "CS",
            edit_action=None,
            candidate_product_smiles=None,
            difficulty_features=tied,
        ),
    )

    orders: dict[int, tuple[str, ...]] = {}
    for seed in range(32):
        seeded_request = CandidateRequest(
            context=replace(
                request.context,
                recipe=replace(request.context.recipe, derived_seed=seed),
            ),
            resolution=request.resolution,
        )
        forward = rank_candidates(seeded_request, proposals)
        reverse = rank_candidates(seeded_request, tuple(reversed(proposals)))
        order = tuple(item.proposal.proposal_id for item in forward)
        assert order == tuple(item.proposal.proposal_id for item in reverse)
        assert order == tuple(
            item.proposal.proposal_id
            for item in rank_candidates(seeded_request, proposals)
        )
        orders[seed] = order

    assert len(set(orders.values())) > 1


def test_mixed_rule_rdkit_engine_is_deterministic_across_source_order() -> None:
    request = _request("add_fragment")
    rule = _proposal(
        request,
        "rule",
        "CCN",
        source=CandidateSourceType.RULE,
        edit_action=None,
        candidate_product_smiles=None,
    )
    rdkit = _proposal(
        request,
        "rdkit",
        "CCCl",
        source=CandidateSourceType.RDKIT,
        edit_action=None,
        candidate_product_smiles=None,
    )
    rule_source = RuleCandidateSource(lambda _: (rule,))
    rdkit_source = RDKitCandidateSource(lambda _: (rdkit,))

    forward = DeterministicCandidateEngine((rule_source, rdkit_source)).build_pool(
        request
    )
    reverse = DeterministicCandidateEngine((rdkit_source, rule_source)).build_pool(
        request
    )

    assert forward == reverse
    assert {candidate.source for candidate in forward.pool.candidates} == {
        CandidateSourceType.RULE,
        CandidateSourceType.RDKIT,
    }


def test_candidate_product_equivalent_to_ground_truth_is_rejected() -> None:
    request = _request("add_fragment")
    anchor = request.context.truth.valid_anchor_indices[-1]
    proposal = _proposal(
        request,
        "gt-equivalent",
        "CO",
        edit_action=_edit_action(anchor=anchor),
        candidate_product_smiles=request.context.truth.gt_smiles,
    )

    result = _build(request, (proposal,))

    assert not result.pool.candidates
    assert tuple((item.proposal_id, item.code) for item in result.rejections) == (
        ("gt-equivalent", CandidateRejectCode.REFERENCE_EQUIVALENT),
    )


def test_canonical_key_removes_atom_maps_but_preserves_stereo_and_components() -> None:
    request = _request("product")
    anchor = request.context.truth.valid_anchor_indices[-1]
    action = _edit_action(anchor=anchor)

    mapped = _proposal(
        request,
        "mapped",
        "[CH3:7][CH2:8][NH2:9]",
        edit_action=action,
        candidate_product_smiles="[CH3:7][CH2:8][NH2:9]",
    )
    unmapped = _proposal(
        request,
        "unmapped",
        "CCN",
        edit_action=action,
        candidate_product_smiles="CCN",
    )
    disconnected_forward = _proposal(
        request,
        "disconnected-forward",
        "CCN.[Na+]",
        edit_action=action,
        candidate_product_smiles="CCN.[Na+]",
    )
    disconnected_reverse = _proposal(
        request,
        "disconnected-reverse",
        "[Na+].NCC",
        edit_action=action,
        candidate_product_smiles="[Na+].NCC",
    )
    clockwise = _proposal(
        request,
        "clockwise",
        "F[C@H](Cl)Br",
        edit_action=action,
        candidate_product_smiles="F[C@H](Cl)Br",
    )
    anticlockwise = _proposal(
        request,
        "anticlockwise",
        "F[C@@H](Cl)Br",
        edit_action=action,
        candidate_product_smiles="F[C@@H](Cl)Br",
    )

    assert canonical_candidate_key(mapped) == canonical_candidate_key(unmapped)
    assert canonical_candidate_key(disconnected_forward) == canonical_candidate_key(
        disconnected_reverse
    )
    assert canonical_candidate_key(disconnected_forward) != canonical_candidate_key(
        unmapped
    )
    assert canonical_candidate_key(clockwise) != canonical_candidate_key(anticlockwise)


def test_request_requires_exact_context_resolution_binding() -> None:
    request = _request("product")
    mismatched_context = replace(
        request.context,
        recipe=replace(
            request.context.recipe,
            candidate_source_mode=CandidateSourceType.RULE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="context and operator resolution do not describe one request",
    ):
        CandidateRequest(context=mismatched_context, resolution=request.resolution)


def test_source_and_new_value_provenance_are_exact_and_fail_closed() -> None:
    request = _request("add_fragment")
    rule = _proposal(
        request,
        "rule-provenance",
        "CCN",
        source=CandidateSourceType.RULE,
        edit_action=None,
        candidate_product_smiles=None,
    )
    rdkit = _proposal(
        request,
        "rdkit-provenance",
        "CCCl",
        source=CandidateSourceType.RDKIT,
        edit_action=None,
        candidate_product_smiles=None,
    )
    accepted = DeterministicCandidateEngine(
        (
            RuleCandidateSource(lambda _: (rule,)),
            RDKitCandidateSource(lambda _: (rdkit,)),
        )
    ).build_pool(request)

    assert {
        (patch.source, patch.new_value.provenance) for patch in accepted.pool.candidates
    } == {
        (CandidateSourceType.RULE, ValueProvenance.RULE),
        (CandidateSourceType.RDKIT, ValueProvenance.RDKIT),
    }

    wrong_provenance = replace(
        rule,
        proposal_id="wrong-provenance",
        patch=replace(
            rule.patch,
            candidate_id="candidate:wrong-provenance",
            new_value=replace(
                rule.patch.new_value,
                provenance=ValueProvenance.REFERENCE,
            ),
        ),
    )
    rejected = _build(request, (wrong_provenance,))
    assert not rejected.pool.candidates
    assert tuple(item.code for item in rejected.rejections) == (
        CandidateRejectCode.SOURCE_MISMATCH,
    )
    assert rejected.rejections[0].evidence == {
        "actual_provenance": ValueProvenance.REFERENCE.value,
        "expected_provenance": ValueProvenance.RULE.value,
    }


def test_non_hybrid_request_never_invokes_or_admits_the_other_source() -> None:
    request = _request("add_fragment", source=CandidateSourceType.RULE)
    rule = _proposal(
        request,
        "rule-only",
        "CCN",
        source=CandidateSourceType.RULE,
        edit_action=None,
        candidate_product_smiles=None,
    )
    rdkit_calls: list[str] = []

    def forbidden_rdkit_source(_: CandidateRequest) -> tuple[CandidateProposal, ...]:
        rdkit_calls.append("called")
        return (
            _proposal(
                request,
                "must-not-be-admitted",
                "CCCl",
                source=CandidateSourceType.RDKIT,
                edit_action=None,
                candidate_product_smiles=None,
            ),
        )

    result = DeterministicCandidateEngine(
        (
            RDKitCandidateSource(forbidden_rdkit_source),
            RuleCandidateSource(lambda _: (rule,)),
        )
    ).build_pool(request)

    assert rdkit_calls == []
    assert tuple(patch.candidate_id for patch in result.pool.candidates) == (
        "candidate:rule-only",
    )
    assert not result.rejections


def test_cross_source_duplicate_primary_and_ledger_are_permutation_invariant() -> None:
    request = _request("add_fragment")
    rule = _proposal(
        request,
        "rule-duplicate",
        "CCN",
        source=CandidateSourceType.RULE,
        edit_action=None,
        candidate_product_smiles=None,
    )
    rdkit = _proposal(
        request,
        "rdkit-duplicate",
        "NCC",
        source=CandidateSourceType.RDKIT,
        edit_action=None,
        candidate_product_smiles=None,
    )
    assert canonical_candidate_key(rule) == canonical_candidate_key(rdkit)
    rule_source = RuleCandidateSource(lambda _: (rule,))
    rdkit_source = RDKitCandidateSource(lambda _: (rdkit,))

    forward = DeterministicCandidateEngine((rule_source, rdkit_source)).build_pool(
        request
    )
    reverse = DeterministicCandidateEngine((rdkit_source, rule_source)).build_pool(
        request
    )

    assert forward == reverse
    assert len(forward.ranked_candidates) == 1
    assert len(forward.rejections) == 1
    accepted = forward.ranked_candidates[0].proposal
    duplicate = forward.rejections[0]
    assert duplicate.code is CandidateRejectCode.DUPLICATE
    assert {accepted.proposal_id, duplicate.proposal_id} == {
        "rule-duplicate",
        "rdkit-duplicate",
    }
    assert duplicate.evidence == {
        "duplicate_of": accepted.proposal_id,
        "duplicate_scope": "semantic",
    }
    source_by_id = {
        rule.proposal_id: rule.patch.source,
        rdkit.proposal_id: rdkit.patch.source,
    }
    assert accepted.patch.source is source_by_id[accepted.proposal_id]


def test_source_exception_is_contained_in_a_structured_rejection() -> None:
    request = _request("product")

    def raising_source(_: CandidateRequest) -> tuple[CandidateProposal, ...]:
        raise RuntimeError("private chemistry implementation detail")

    result = DeterministicCandidateEngine(
        (RuleCandidateSource(raising_source),)
    ).build_pool(request)

    assert not result.pool.candidates
    assert tuple((item.proposal_id, item.code) for item in result.rejections) == (
        (
            f"source:{CandidateSourceType.RULE.value}",
            CandidateRejectCode.SOURCE_FAILED,
        ),
    )
    assert result.rejections[0].evidence == {
        "exception_type": "CandidateSourceError",
        "source": CandidateSourceType.RULE.value,
    }


def test_action_product_replay_accepts_exact_addition_and_rejects_tampering() -> None:
    request = _action_replay_request()
    exact_action = _edit_action(
        anchor=4,
        add_fragment_smiles="CO",
        attachment_atom=0,
    )
    proposals = (
        _proposal(
            request,
            "exact-replay",
            "NCCOCO",
            edit_action=exact_action,
            candidate_product_smiles="NCCOCO",
        ),
        _proposal(
            request,
            "unrelated-product",
            "CCN",
            edit_action=exact_action,
            candidate_product_smiles="CCN",
        ),
        _proposal(
            request,
            "wrong-fragment",
            "NCCOCO",
            edit_action=replace(exact_action, add_fragment_smiles="CN"),
            candidate_product_smiles="NCCOCO",
        ),
        _proposal(
            request,
            "wrong-attachment",
            "NCCOCO",
            edit_action=replace(exact_action, fragment_attachment_atom=1),
            candidate_product_smiles="NCCOCO",
        ),
        _proposal(
            request,
            "wrong-anchor",
            "NCCOCO",
            edit_action=replace(exact_action, source_anchor_index=2),
            candidate_product_smiles="NCCOCO",
        ),
        _proposal(
            request,
            "wrong-bond",
            "NCCOCO",
            edit_action=replace(exact_action, bond_type=BondTypeName.DOUBLE),
            candidate_product_smiles="NCCOCO",
        ),
    )

    result = _build(request, proposals)

    assert tuple(patch.candidate_id for patch in result.pool.candidates) == (
        "candidate:exact-replay",
    )
    assert {item.proposal_id: item.code for item in result.rejections} == {
        "unrelated-product": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
        "wrong-anchor": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
        "wrong-attachment": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
        "wrong-bond": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
        "wrong-fragment": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
    }
    assert all(
        item.evidence == {
            "edit_kind": EditKind.ADDITION.value,
            "phase": "action_product_replay",
        }
        for item in result.rejections
    )


def test_product_root_requires_action_and_edit_kind_shapes_are_exact() -> None:
    addition_request = _action_replay_request()
    no_action = _proposal(
        addition_request,
        "product-without-action",
        "NCCOCO",
        edit_action=None,
        candidate_product_smiles="NCCOCO",
    )
    addition_with_remove = _proposal(
        addition_request,
        "addition-with-remove",
        "NCCOCO",
        edit_action=EditAction(
            edit_kind=EditKind.ADDITION,
            source_anchor_index=4,
            remove_fragment_smiles="Cl",
            add_fragment_smiles="CO",
            fragment_attachment_atom=0,
            bond_type=BondTypeName.SINGLE,
        ),
        candidate_product_smiles="NCCOCO",
    )
    addition_result = _build(
        addition_request,
        (no_action, addition_with_remove),
    )

    deletion_request = _deletion_request()
    truth = deletion_request.context.truth
    assert truth.remove_fragment is not None
    deletion_with_add = _proposal(
        deletion_request,
        "deletion-with-add",
        "CCN",
        edit_action=EditAction(
            edit_kind=EditKind.DELETION,
            source_anchor_index=truth.valid_anchor_indices[0],
            remove_fragment_smiles=truth.remove_fragment.canonical_smiles,
            add_fragment_smiles="C",
            fragment_attachment_atom=None,
            bond_type=truth.broken_bonds[0].bond_type,
            metadata={"remove_atom_maps": tuple(sorted(truth.removed_atom_maps))},
        ),
        candidate_product_smiles="CCN",
    )
    deletion_result = _build(deletion_request, (deletion_with_add,))

    assert {item.proposal_id: item.code for item in addition_result.rejections} == {
        "addition-with-remove": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
        "product-without-action": CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
    }
    assert tuple(item.code for item in deletion_result.rejections) == (
        CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
    )


def test_valid_anchor_is_excluded_for_anchor_and_product_roots() -> None:
    anchor_request = _request("anchor_idx")
    reference = anchor_request.context.reference_graph.values[
        "anchor_idx"
    ].normalized_value
    alternate_valid = next(
        anchor
        for anchor in anchor_request.context.truth.valid_anchor_indices
        if anchor != reference
    )
    anchor_result = _build(
        anchor_request,
        (
            _proposal(
                anchor_request,
                "valid-anchor-root",
                alternate_valid,
                edit_action=None,
                candidate_product_smiles="CCN",
            ),
        ),
    )

    product_request = _action_replay_request()
    product_result = _build(
        product_request,
        (
            _proposal(
                product_request,
                "valid-product-anchor",
                "CCNCCO",
                edit_action=_edit_action(
                    anchor=2,
                    add_fragment_smiles="CC",
                    attachment_atom=0,
                ),
                candidate_product_smiles="CCNCCO",
            ),
        ),
    )

    assert tuple(item.code for item in anchor_result.rejections) == (
        CandidateRejectCode.SYMMETRY_EQUIVALENT,
    )
    assert tuple(item.code for item in product_result.rejections) == (
        CandidateRejectCode.SYMMETRY_EQUIVALENT,
    )


def test_symmetric_fragment_attachment_indices_share_one_semantic_candidate() -> None:
    request = _action_replay_request()
    proposals = tuple(
        _proposal(
            request,
            f"ethane-attachment-{attachment}",
            "CCOCCN",
            edit_action=_edit_action(
                anchor=4,
                add_fragment_smiles="CC",
                attachment_atom=attachment,
            ),
            candidate_product_smiles="CCOCCN",
        )
        for attachment in (0, 1)
    )

    assert canonical_candidate_key(proposals[0]) == canonical_candidate_key(
        proposals[1]
    )
    result = _build(request, proposals)
    assert len(result.pool.candidates) == 1
    assert tuple(item.code for item in result.rejections) == (
        CandidateRejectCode.DUPLICATE,
    )


def test_nonchemical_action_metadata_does_not_defeat_semantic_deduplication() -> None:
    request = _action_replay_request()
    proposals = tuple(
        _proposal(
            request,
            f"metadata-{nonce}",
            "NCCOCO",
            edit_action=_edit_action(
                anchor=4,
                add_fragment_smiles="CO",
                attachment_atom=0,
                metadata={"audit": f"run-{nonce}", "nonce": nonce},
            ),
            candidate_product_smiles="NCCOCO",
        )
        for nonce in (1, 2)
    )

    assert canonical_candidate_key(proposals[0]) == canonical_candidate_key(
        proposals[1]
    )
    result = _build(request, proposals)
    assert len(result.pool.candidates) == 1
    assert tuple(item.code for item in result.rejections) == (
        CandidateRejectCode.DUPLICATE,
    )


def test_unavailable_and_unsupported_source_modes_emit_structured_ledgers() -> None:
    unavailable_request = _request(
        "product",
        source=CandidateSourceType.RDKIT,
    )
    unavailable = DeterministicCandidateEngine(
        (RuleCandidateSource(lambda _: ()),)
    ).build_pool(unavailable_request)

    base = _request("product")
    unsupported_request = CandidateRequest(
        context=replace(
            base.context,
            recipe=replace(
                base.context.recipe,
                candidate_source_mode=CandidateSourceType.LLM,
            ),
        ),
        resolution=replace(
            base.resolution,
            candidate_source=CandidateSourceType.LLM,
        ),
    )
    unsupported = DeterministicCandidateEngine(
        (RuleCandidateSource(lambda _: ()),)
    ).build_pool(unsupported_request)

    assert tuple((item.proposal_id, item.code) for item in unavailable.rejections) == (
        (
            f"source:{CandidateSourceType.RDKIT.value}",
            CandidateRejectCode.SOURCE_UNAVAILABLE,
        ),
    )
    assert unavailable.rejections[0].evidence == {
        "configured_sources": (CandidateSourceType.RULE.value,),
        "requested_source": CandidateSourceType.RDKIT.value,
    }
    assert tuple((item.proposal_id, item.code) for item in unsupported.rejections) == (
        (
            f"source:{CandidateSourceType.LLM.value}",
            CandidateRejectCode.UNSUPPORTED_SOURCE_MODE,
        ),
    )
    assert unsupported.rejections[0].evidence == {
        "source_mode": CandidateSourceType.LLM.value,
    }


def test_metadata_carriers_and_occurrence_aliases_cannot_bypass_deduplication() -> None:
    addition_request = _action_replay_request()
    addition_metadata = (
        {"boundary_atom_maps": (3, 4)},
        {"orientation": "forward"},
        {"attachment_orientation": "head", "stereo": "STEREONONE"},
    )
    addition_proposals = tuple(
        _proposal(
            addition_request,
            f"addition-metadata-{index}",
            "NCCOCO",
            edit_action=_edit_action(
                anchor=4,
                add_fragment_smiles="CO",
                attachment_atom=0,
                metadata=metadata,
            ),
            candidate_product_smiles="NCCOCO",
        )
        for index, metadata in enumerate(addition_metadata)
    )
    assert len({canonical_candidate_key(item) for item in addition_proposals}) == 1
    addition_result = _build(addition_request, addition_proposals)
    assert len(addition_result.pool.candidates) == 1
    assert tuple(item.code for item in addition_result.rejections) == (
        CandidateRejectCode.DUPLICATE,
        CandidateRejectCode.DUPLICATE,
    )

    deletion_request = _deletion_action_replay_request()
    occurrence_aliases = (
        {"remove_atom_maps": (3, 4)},
        {"occurrence_atom_maps": (4, 3)},
    )
    deletion_proposals = tuple(
        _proposal(
            deletion_request,
            f"deletion-alias-{index}",
            "CO",
            edit_action=EditAction(
                edit_kind=EditKind.DELETION,
                source_anchor_index=2,
                remove_fragment_smiles="CC",
                fragment_attachment_atom=None,
                bond_type=BondTypeName.SINGLE,
                metadata=metadata,
            ),
            candidate_product_smiles="CO",
        )
        for index, metadata in enumerate(occurrence_aliases)
    )
    assert canonical_candidate_key(deletion_proposals[0]) == canonical_candidate_key(
        deletion_proposals[1]
    )
    deletion_result = _build(deletion_request, deletion_proposals)
    assert len(deletion_result.pool.candidates) == 1
    assert tuple(item.code for item in deletion_result.rejections) == (
        CandidateRejectCode.DUPLICATE,
    )
