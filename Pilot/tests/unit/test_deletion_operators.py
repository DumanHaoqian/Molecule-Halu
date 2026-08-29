"""T020 concrete Deletion operator contracts and graph-cut regressions."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
from rdkit import Chem

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.builders import build_reference_dag, derive_edit_truth
from molhallulens.candidates import (
    CandidateRejectCode,
    CandidateRequest,
    replay_edit_action,
)
from molhallulens.chemistry import (
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
)
from molhallulens.config import load_config_bundle
from molhallulens.domain import (
    BondTypeName,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    EditAction,
    EditErrorSubtype,
    EditKind,
    HallucinationType,
    OperationSubtype,
    OperatorCapability,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    ValueProvenance,
)
from molhallulens.perturbators import (
    CandidateEngine,
    DeletionPerturbator,
    LabelProjector,
    OperatorRegistration,
    OperatorRegistryError,
    PerturbationContext,
    PerturbatorRegistry,
    PropagationEngine,
    TraceRenderer,
    ValidatorChain,
    task_record_from_joined_input,
)
from molhallulens.perturbators.editing.deletion import (
    DELETION_OPERATOR_IDS,
    REPLACEMENT_DELETION_OPERATOR_ID,
    DeletionCandidateDispatcher,
    DeletionCandidateEngine,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators
REPLACEMENT_ID = "mol_edit.delete_v2.0081"

EXPECTED_METHODS = (
    "perturb_wrong_group_occurrence",
    "perturb_wrong_adjacent_group",
    "perturb_group_boundary_contract",
    "perturb_group_boundary_expand",
    "perturb_partial_deletion",
    "perturb_over_deletion",
    "perturb_matched_remove_group",
    "perturb_alternative_deprotection_product",
    "perturb_cross_step_group_identity",
    "perturb_heavy_count_claim",
    "perturb_ring_count_claim",
    "perturb_terminal_answer",
)

STRUCTURAL_POLICIES = frozenset(
    {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
)
CLAIM_POLICIES = frozenset({PropagationPolicy.STOP, PropagationPolicy.PARTIAL})
DETERMINISTIC_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
RELATION_SOURCES = frozenset({CandidateSourceType.RULE, CandidateSourceType.HYBRID})
COMMON_HALLUCINATIONS = frozenset(
    {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
)

EXPECTED_METADATA = {
    DELETION_OPERATOR_IDS[0]: (
        EXPECTED_METHODS[0],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[1]: (
        EXPECTED_METHODS[1],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[2]: (
        EXPECTED_METHODS[2],
        "attachment_bond_edit",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[3]: (
        EXPECTED_METHODS[3],
        "attachment_bond_edit",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[4]: (
        EXPECTED_METHODS[4],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[5]: (
        EXPECTED_METHODS[5],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[6]: (
        EXPECTED_METHODS[6],
        "wrong_fragment_group",
        frozenset({"remove_group_step1", "remove_group_step2"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[7]: (
        EXPECTED_METHODS[7],
        "wrong_fragment_group",
        frozenset({"product"}),
        frozenset({PropagationPolicy.FULL_CF}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.STRUCTURAL_DELETION}),
    ),
    DELETION_OPERATOR_IDS[8]: (
        EXPECTED_METHODS[8],
        "nl_formal_internal_relation",
        frozenset({"remove_group_step1", "remove_group_step2"}),
        CLAIM_POLICIES,
        RELATION_SOURCES,
        frozenset({HallucinationType.REASONING_ERROR}),
        frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    DELETION_OPERATOR_IDS[9]: (
        EXPECTED_METHODS[9],
        "numeric_count_claim",
        frozenset({"remove_heavy", "source_heavy", "product_heavy", "heavy_delta"}),
        CLAIM_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset(
            {EditErrorSubtype.HEAVY_ATOM_COUNT, EditErrorSubtype.HEAVY_ATOM_ARITHMETIC}
        ),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    DELETION_OPERATOR_IDS[10]: (
        EXPECTED_METHODS[10],
        "numeric_count_claim",
        frozenset({"source_rings", "product_rings", "ring_delta"}),
        CLAIM_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    DELETION_OPERATOR_IDS[11]: (
        EXPECTED_METHODS[11],
        "final_answer_identity",
        frozenset({"final_answer"}),
        frozenset({PropagationPolicy.TERMINAL}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        frozenset({OperatorCapability.TERMINAL_PERTURBATION}),
    ),
    REPLACEMENT_DELETION_OPERATOR_ID: (
        "perturb_replacement_product",
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.REPLACEMENT_AWARE_DELETION}),
    ),
}

OPERATOR_RUNTIME_CASES = (
    (DELETION_OPERATOR_IDS[0], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[1], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[2], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[3], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[4], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[5], "product", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[6], "remove_group_step1", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[7], "product", PropagationPolicy.FULL_CF),
    (DELETION_OPERATOR_IDS[8], "remove_group_step2", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[9], "heavy_delta", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[10], "ring_delta", PropagationPolicy.STOP),
    (DELETION_OPERATOR_IDS[11], "final_answer", PropagationPolicy.TERMINAL),
)


class _UnusedPropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T020 operator tests do not execute T022 propagation")


class _UnusedRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T020 operator tests do not execute rendering")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T020 tests do not execute the full template")

    def validate_artifact(self, draft):
        raise AssertionError("T020 tests do not validate rendered artifacts")


class _UnusedLabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T020 tests do not project token labels")


def _ports(candidate_engine: CandidateEngine) -> dict[str, object]:
    return {
        "candidate_engine": candidate_engine,
        "propagator": _UnusedPropagationEngine(),
        "renderer": _UnusedRenderer(),
        "validators": _UnusedValidators(),
        "label_projector": _UnusedLabelProjector(),
    }


@cache
def _delete_records() -> tuple[JoinedInputRecord, ...]:
    return tuple(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".delete_v2." in record.anonymous_sample_id
    )


@cache
def _origin_artifacts(anonymous_sample_id: str):
    joined = next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    truth = derive_edit_truth(artifact)
    record = task_record_from_joined_input(joined)
    return joined, artifact, truth, record


def _registry() -> PerturbatorRegistry:
    return PerturbatorRegistry.from_perturbator_types(
        (DeletionPerturbator,),
        operators_config=OPERATORS_CONFIG,
    )


def _production_perturbator() -> DeletionPerturbator:
    return DeletionPerturbator(
        **_ports(DeletionCandidateEngine(operators_config=OPERATORS_CONFIG))
    )


def _context(
    registration: OperatorRegistration,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
    joined: JoinedInputRecord | None = None,
) -> PerturbationContext:
    selected = joined or next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id != REPLACEMENT_ID
    )
    _, artifact, truth, record = _origin_artifacts(selected.anonymous_sample_id)
    root = target_node_id or min(registration.spec.root_fields)
    selected_policy = policy or (
        PropagationPolicy.TERMINAL
        if PropagationPolicy.TERMINAL in registration.spec.supported_policies
        else PropagationPolicy.STOP
    )
    recipe = PerturbationRecipe(
        recipe_id=f"t020:{record.origin_id}:{registration.operator_id}:{root}",
        origin_id=record.origin_id,
        operator_id=registration.operator_id,
        policy=selected_policy,
        target_node_id=root,
        candidate_source_mode=source,
        variant_index=0,
        derived_seed=20260829,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=128,
            length_bucket="t020",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
        partial_cut_nodes=(
            frozenset({"product_heavy", "product_rings"})
            if selected_policy is PropagationPolicy.PARTIAL
            else frozenset()
        ),
    )
    return PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=artifact.state_dag.schema,
        reference_graph=artifact.state_dag,
        truth=truth,
    )


def _invoke(
    operator_id: str,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
    joined: JoinedInputRecord | None = None,
) -> tuple[PerturbationContext, CandidatePool]:
    context = _context(
        _registry().registration(operator_id),
        target_node_id=target_node_id,
        policy=policy,
        source=source,
        joined=joined,
    )
    perturbator = _production_perturbator()
    return context, perturbator.candidate_engine.enumerate_root_patches(context)


def _first_nonempty(
    operator_id: str,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
) -> tuple[JoinedInputRecord, PerturbationContext, CandidatePool]:
    rejections: set[str] = set()
    for joined in _delete_records():
        if joined.anonymous_sample_id == REPLACEMENT_ID:
            continue
        context, pool = _invoke(
            operator_id,
            target_node_id=target_node_id,
            policy=policy,
            source=source,
            joined=joined,
        )
        if pool.candidates:
            return joined, context, pool
        rejections.update(pool.rejection_codes)
    pytest.fail(
        f"{operator_id} produced no candidate for 49 deprotections; "
        f"rejections={sorted(rejections)!r}"
    )


def _candidate_request(context: PerturbationContext) -> CandidateRequest:
    perturbator = _production_perturbator()
    return CandidateRequest(
        context=context,
        resolution=_registry().resolve(perturbator, context),
    )


def _occurrence_maps(patch: CandidatePatch) -> tuple[int, ...]:
    action = patch.edit_action
    assert action is not None
    primary = action.metadata.get("remove_atom_maps")
    alternate = action.metadata.get("occurrence_atom_maps")
    assert primary is None or alternate is None or primary == alternate
    occurrence = primary if primary is not None else alternate
    assert type(occurrence) is tuple
    assert occurrence
    assert all(type(atom_map) is int and atom_map > 0 for atom_map in occurrence)
    assert tuple(sorted(set(occurrence))) == occurrence
    return occurrence


def _source_cut(
    indexed_smiles: str,
    occurrence: tuple[int, ...],
) -> tuple[Chem.Mol, dict[int, int], tuple[tuple[int, int, BondTypeName], ...]]:
    molecule = Chem.MolFromSmiles(indexed_smiles, sanitize=True)
    assert molecule is not None
    mapped = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    assert len(mapped) == molecule.GetNumHeavyAtoms()
    removed = set(occurrence)
    assert removed <= set(mapped)
    start = next(iter(removed))
    visited = {start}
    frontier = [start]
    while frontier:
        atom_map = frontier.pop()
        atom = molecule.GetAtomWithIdx(mapped[atom_map])
        for neighbor in atom.GetNeighbors():
            neighbor_map = neighbor.GetAtomMapNum()
            if neighbor_map in removed and neighbor_map not in visited:
                visited.add(neighbor_map)
                frontier.append(neighbor_map)
    assert visited == removed

    boundary: list[tuple[int, int, BondTypeName]] = []
    for removed_map in sorted(removed):
        atom = molecule.GetAtomWithIdx(mapped[removed_map])
        for neighbor in atom.GetNeighbors():
            retained_map = neighbor.GetAtomMapNum()
            if retained_map in removed:
                continue
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            boundary.append(
                (
                    removed_map,
                    retained_map,
                    BondTypeName(str(bond.GetBondType()).upper()),
                )
            )
    return molecule, mapped, tuple(sorted(boundary))


def _assert_connected_single_boundary_replay(
    context: PerturbationContext,
    patch: CandidatePatch,
) -> tuple[int, ...]:
    action = patch.edit_action
    assert action is not None
    assert action.edit_kind is EditKind.DELETION
    assert action.remove_fragment_smiles is not None
    assert action.add_fragment_smiles is None
    assert action.fragment_attachment_atom is None
    assert type(action.bond_type) is BondTypeName
    occurrence = _occurrence_maps(patch)
    source, _, boundary = _source_cut(context.record.indexed_smiles, occurrence)
    assert len(boundary) == 1
    _, retained_map, boundary_type = boundary[0]
    assert action.source_anchor_index == retained_map
    assert action.bond_type is boundary_type

    products = replay_edit_action(_candidate_request(context), action)
    assert products
    for product in products:
        parsed = Chem.MolFromSmiles(product, sanitize=True)
        assert parsed is not None
        Chem.SanitizeMol(parsed)
        assert all(atom.GetNumRadicalElectrons() == 0 for atom in parsed.GetAtoms())
        assert parsed.GetNumHeavyAtoms() == source.GetNumHeavyAtoms() - len(occurrence)
        assert canonicalize_smiles(product) == product
        assert not isomeric_graph_equivalent(product, context.truth.gt_smiles)
        if patch.root_node_id in {"product", "final_answer"}:
            assert isomeric_graph_equivalent(product, patch.new_value.normalized_value)
    return occurrence


def test_deletion_preserves_twelve_blueprint_operators_and_adds_typed_replacement() -> (
    None
):
    registrations = _registry().registrations_for(
        task_family="mol_edit", subtask="delete"
    )
    assert DELETION_OPERATOR_IDS == tuple(
        f"mol_edit.delete.{method.removeprefix('perturb_')}"
        for method in EXPECTED_METHODS
    )
    assert {registration.operator_id for registration in registrations} == {
        *DELETION_OPERATOR_IDS,
        REPLACEMENT_DELETION_OPERATOR_ID,
    }

    for registration in registrations:
        (
            method,
            family,
            roots,
            policies,
            sources,
            hallucinations,
            subtypes,
            capabilities,
        ) = EXPECTED_METADATA[registration.operator_id]
        assert registration.method_name == method
        assert registration.operator_family == family
        assert registration.spec.root_fields == roots
        assert registration.spec.supported_policies == policies
        assert registration.spec.supported_sources == sources
        assert registration.spec.hallucination_types == hallucinations
        assert registration.edit_subtypes == subtypes
        assert registration.required_capabilities == capabilities
        assert registration.spec.diagnostic_only is False


def test_deletion_engine_implements_t016_port_and_registry_dispatch() -> None:
    engine = DeletionCandidateEngine(operators_config=OPERATORS_CONFIG)
    assert isinstance(engine, CandidateEngine)
    perturbator = DeletionPerturbator(**_ports(engine))
    registration = _registry().registration(DELETION_OPERATOR_IDS[0])
    context = _context(registration)

    direct = engine.enumerate_root_patches(context)
    dispatched = DeletionCandidateDispatcher(OPERATORS_CONFIG).invoke(
        perturbator, context
    )

    assert direct == dispatched
    assert direct.request_id == context.recipe.recipe_id
    for patch in direct.candidates:
        assert patch.root_node_id == context.recipe.target_node_id
        assert patch.old_value == context.reference_graph.value_for(
            context.recipe.target_node_id
        )


def test_llm_is_rejected_before_any_deletion_member_invocation() -> None:
    registry = _registry()
    perturbator = _production_perturbator()
    for registration in registry.registrations_for(
        task_family="mol_edit", subtask="delete"
    ):
        context = _context(registration, source=CandidateSourceType.LLM)
        with pytest.raises(OperatorRegistryError) as caught:
            registry.invoke(perturbator, context)
        assert caught.value.code == "INCOMPATIBLE_SOURCE"
        assert caught.value.operator_id == registration.operator_id


@pytest.mark.parametrize(
    ("operator_id", "target", "policy"),
    OPERATOR_RUNTIME_CASES,
)
def test_each_operator_is_root_only_bound_deterministic_and_source_audited(
    operator_id: str,
    target: str,
    policy: PropagationPolicy,
) -> None:
    joined, context, pool = _first_nonempty(
        operator_id,
        target_node_id=target,
        policy=policy,
    )
    assert (
        1
        <= len(pool.candidates)
        <= (OPERATORS_CONFIG.candidate_generation.candidates_per_recipe_max)
    )
    for patch in pool.candidates:
        assert patch.root_node_id == target
        assert patch.old_value == context.reference_graph.value_for(target)
        assert not patch.old_value.semantically_equals(patch.new_value)
        assert patch.source is CandidateSourceType.RULE
        assert patch.new_value.provenance is ValueProvenance.RULE
        assert patch.metadata["generator"] == "deletion_t020"
        assert patch.metadata["operator_id"] == operator_id

    repeated_context, repeated_pool = _invoke(
        operator_id,
        target_node_id=target,
        policy=policy,
        joined=joined,
    )
    assert repeated_context == context
    assert repeated_pool == pool

    registration = _registry().registration(operator_id)
    for source in sorted(
        registration.spec.supported_sources - {CandidateSourceType.RULE},
        key=lambda item: item.value,
    ):
        source_context, source_pool = _invoke(
            operator_id,
            target_node_id=target,
            policy=policy,
            source=source,
            joined=joined,
        )
        assert source_pool.candidates
        for patch in source_pool.candidates:
            assert patch.root_node_id == target
            assert patch.old_value == source_context.reference_graph.value_for(target)
            if source is CandidateSourceType.RDKIT:
                assert patch.source is CandidateSourceType.RDKIT
                assert patch.new_value.provenance is ValueProvenance.RDKIT
            else:
                assert patch.source in {
                    CandidateSourceType.RULE,
                    CandidateSourceType.RDKIT,
                }
                assert (
                    patch.new_value.provenance
                    is {
                        CandidateSourceType.RULE: ValueProvenance.RULE,
                        CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
                    }[patch.source]
                )


def test_delete_with_replacement_blocks_structural_operators_at_resolution() -> None:
    replacement = next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id == REPLACEMENT_ID
    )
    _, _, truth, record = _origin_artifacts(REPLACEMENT_ID)
    assert record.operation_subtype is OperationSubtype.DELETE_WITH_REPLACEMENT
    assert truth.add_fragment is not None
    registry = _registry()
    perturbator = _production_perturbator()

    for operator_id, target, policy in OPERATOR_RUNTIME_CASES[:8]:
        context = _context(
            registry.registration(operator_id),
            target_node_id=target,
            policy=policy,
            joined=replacement,
        )
        with pytest.raises(OperatorRegistryError) as caught:
            registry.resolve(perturbator, context)
        assert caught.value.code == "OPERATOR_CAPABILITY_FORBIDDEN"
        assert caught.value.evidence == {
            "forbidden_capabilities": (OperatorCapability.STRUCTURAL_DELETION.value,)
        }

    for operator_id, target, policy in OPERATOR_RUNTIME_CASES[8:]:
        context = _context(
            registry.registration(operator_id),
            target_node_id=target,
            policy=policy,
            joined=replacement,
        )
        resolution = registry.resolve(perturbator, context)
        assert resolution.classification.operation_subtype is (
            OperationSubtype.DELETE_WITH_REPLACEMENT
        )
        assert resolution.registration.required_capabilities <= {
            OperatorCapability.CLAIM_PERTURBATION,
            OperatorCapability.TERMINAL_PERTURBATION,
        }
        runtime_context, pool = _invoke(
            operator_id,
            target_node_id=target,
            policy=policy,
            joined=replacement,
        )
        assert runtime_context == context
        assert pool.candidates
        assert all(patch.root_node_id == target for patch in pool.candidates)
        assert all(patch.edit_action is None for patch in pool.candidates)


@pytest.mark.parametrize(
    "policy",
    (
        PropagationPolicy.STOP,
        PropagationPolicy.PARTIAL,
        PropagationPolicy.FULL_CF,
    ),
)
def test_registered_replacement_operator_is_strict_remove_add_replay(
    policy: PropagationPolicy,
) -> None:
    replacement = next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id == REPLACEMENT_ID
    )
    context, pool = _invoke(
        REPLACEMENT_DELETION_OPERATOR_ID,
        target_node_id="product",
        policy=policy,
        source=CandidateSourceType.RULE,
        joined=replacement,
    )
    resolution = _registry().resolve(_production_perturbator(), context)
    assert resolution.classification.operation_subtype is (
        OperationSubtype.DELETE_WITH_REPLACEMENT
    )
    assert resolution.registration.required_capabilities == frozenset(
        {OperatorCapability.REPLACEMENT_AWARE_DELETION}
    )
    assert (
        len(pool.candidates)
        >= OPERATORS_CONFIG.candidate_generation.candidates_per_recipe_min
    )
    assert CandidateRejectCode.INSUFFICIENT_CANDIDATES.value not in pool.rejection_codes

    request = _candidate_request(context)
    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        assert action.edit_kind is EditKind.DELETION
        assert action.is_replacement_deletion
        assert action.remove_fragment_smiles is not None
        assert action.add_fragment_smiles is not None
        assert action.fragment_attachment_atom is not None
        assert _occurrence_maps(patch) == tuple(sorted(context.truth.removed_atom_maps))
        products = replay_edit_action(request, action)
        assert len(products) == 1
        assert isomeric_graph_equivalent(products[0], patch.new_value.normalized_value)
        assert not isomeric_graph_equivalent(products[0], context.truth.gt_smiles)
        parsed = Chem.MolFromSmiles(products[0], sanitize=True)
        assert parsed is not None
        Chem.SanitizeMol(parsed)
        assert all(atom.GetNumRadicalElectrons() == 0 for atom in parsed.GetAtoms())


def test_replacement_operator_is_forbidden_for_ordinary_remove_only_delete() -> None:
    ordinary = next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id != REPLACEMENT_ID
    )
    registration = _registry().registration(REPLACEMENT_DELETION_OPERATOR_ID)
    context = _context(
        registration,
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
        joined=ordinary,
    )
    with pytest.raises(OperatorRegistryError) as caught:
        _registry().resolve(_production_perturbator(), context)
    assert caught.value.code == "OPERATOR_CAPABILITY_FORBIDDEN"
    assert caught.value.evidence == {
        "forbidden_capabilities": (OperatorCapability.REPLACEMENT_AWARE_DELETION.value,)
    }


def test_replacement_counts_never_apply_the_remove_only_delta_formula() -> None:
    replacement = next(
        record
        for record in _delete_records()
        if record.anonymous_sample_id == REPLACEMENT_ID
    )
    _, artifact, _, _ = _origin_artifacts(REPLACEMENT_ID)
    source_heavy = artifact.state_dag.value_for("source_heavy").normalized_value
    remove_heavy = artifact.state_dag.value_for("remove_heavy").normalized_value
    forbidden = {
        "product_heavy": source_heavy - remove_heavy,
        "heavy_delta": -remove_heavy,
    }

    for target, forbidden_value in forbidden.items():
        context, pool = _invoke(
            DELETION_OPERATOR_IDS[9],
            target_node_id=target,
            joined=replacement,
        )
        assert context.record.operation_subtype is (
            OperationSubtype.DELETE_WITH_REPLACEMENT
        )
        assert pool.candidates
        assert all(patch.edit_action is None for patch in pool.candidates)
        assert all(
            patch.new_value.normalized_value != forbidden_value
            for patch in pool.candidates
        )


def test_all_forty_nine_deprotections_enable_structural_ops_and_truth_replays_gt() -> (
    None
):
    ordinary = tuple(
        record
        for record in _delete_records()
        if record.anonymous_sample_id != REPLACEMENT_ID
    )
    assert len(ordinary) == 49
    registry = _registry()
    perturbator = _production_perturbator()

    for joined in ordinary:
        _, _, truth, record = _origin_artifacts(joined.anonymous_sample_id)
        assert record.operation_subtype is OperationSubtype.DEPROTECTION
        assert truth.add_fragment is None
        assert truth.remove_fragment is not None
        assert len(truth.broken_bonds) == 1
        removed = tuple(sorted(truth.removed_atom_maps))
        broken = truth.broken_bonds[0]
        removed_endpoints = tuple(
            endpoint.atom_id
            for endpoint in (broken.begin, broken.end)
            if endpoint.atom_id in truth.removed_atom_maps
        )
        retained_endpoints = tuple(
            endpoint.atom_id
            for endpoint in (broken.begin, broken.end)
            if endpoint.atom_id not in truth.removed_atom_maps
        )
        assert len(removed_endpoints) == len(retained_endpoints) == 1

        replay_context = _context(
            registry.registration(DELETION_OPERATOR_IDS[0]),
            target_node_id="product",
            policy=PropagationPolicy.FULL_CF,
            joined=joined,
        )
        action = EditAction(
            edit_kind=EditKind.DELETION,
            source_anchor_index=retained_endpoints[0],
            remove_fragment_smiles=truth.remove_fragment.canonical_smiles,
            bond_type=broken.bond_type,
            metadata={"remove_atom_maps": removed},
        )
        products = replay_edit_action(_candidate_request(replay_context), action)
        assert any(
            isomeric_graph_equivalent(product, truth.gt_smiles) for product in products
        )

        for operator_id, target, policy in OPERATOR_RUNTIME_CASES[:8]:
            context = _context(
                registry.registration(operator_id),
                target_node_id=target,
                policy=policy,
                joined=joined,
            )
            resolution = registry.resolve(perturbator, context)
            assert resolution.classification.operation_subtype is (
                OperationSubtype.DEPROTECTION
            )
            assert resolution.registration.required_capabilities == frozenset(
                {OperatorCapability.STRUCTURAL_DELETION}
            )


@pytest.mark.parametrize(
    ("operator_id", "target", "policy"),
    OPERATOR_RUNTIME_CASES[:8],
)
def test_structural_deletions_are_connected_single_boundary_replays(
    operator_id: str,
    target: str,
    policy: PropagationPolicy,
) -> None:
    _, context, pool = _first_nonempty(
        operator_id,
        target_node_id=target,
        policy=policy,
        source=CandidateSourceType.RDKIT,
    )
    truth_removed = frozenset(context.truth.removed_atom_maps)
    truth_fragment = context.truth.remove_fragment
    assert truth_fragment is not None

    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        assert "." not in action.remove_fragment_smiles
        occurrence = frozenset(_assert_connected_single_boundary_replay(context, patch))
        assert occurrence != truth_removed

        if operator_id == DELETION_OPERATOR_IDS[0]:
            assert fragment_graph_equivalent(
                action.remove_fragment_smiles, truth_fragment.canonical_smiles
            )
        elif operator_id == DELETION_OPERATOR_IDS[1]:
            assert (
                occurrence.intersection(truth_removed)
                or action.source_anchor_index in truth_removed
                or set(context.truth.valid_anchor_indices).intersection(occurrence)
            )
        elif operator_id in {
            DELETION_OPERATOR_IDS[2],
            DELETION_OPERATOR_IDS[4],
        }:
            assert occurrence < truth_removed
        elif operator_id in {
            DELETION_OPERATOR_IDS[3],
            DELETION_OPERATOR_IDS[5],
        }:
            assert occurrence > truth_removed
        elif operator_id == DELETION_OPERATOR_IDS[6]:
            actual = compute_descriptors(action.remove_fragment_smiles)
            expected = truth_fragment.descriptors
            assert not fragment_graph_equivalent(
                action.remove_fragment_smiles, truth_fragment.canonical_smiles
            )
            assert (
                actual.heavy_atom_count,
                actual.ring_count,
                actual.formal_charge,
                actual.heteroatom_counts,
            ) == (
                expected.heavy_atom_count,
                expected.ring_count,
                expected.formal_charge,
                expected.heteroatom_counts,
            )


def test_replay_rejects_disconnected_and_noninduced_removal_queries() -> None:
    selected: tuple[PerturbationContext, CandidatePatch] | None = None
    for joined in _delete_records():
        if joined.anonymous_sample_id == REPLACEMENT_ID:
            continue
        context, pool = _invoke(
            DELETION_OPERATOR_IDS[1],
            target_node_id="product",
            joined=joined,
        )
        source = Chem.MolFromSmiles(context.record.indexed_smiles, sanitize=True)
        assert source is not None
        mapped = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in source.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        for patch in pool.candidates:
            occurrence = _occurrence_maps(patch)
            indices = {mapped[atom_map] for atom_map in occurrence}
            internal_bonds = tuple(
                bond
                for bond in source.GetBonds()
                if bond.GetBeginAtomIdx() in indices and bond.GetEndAtomIdx() in indices
            )
            if (
                len(indices) == 6
                and len(internal_bonds) == 6
                and all(
                    source.GetAtomWithIdx(index).GetAtomicNum() == 6
                    for index in indices
                )
            ):
                selected = (context, patch)
                break
        if selected is not None:
            break

    assert selected is not None
    context, patch = selected
    action = patch.edit_action
    assert action is not None
    _assert_connected_single_boundary_replay(context, patch)
    source = Chem.MolFromSmiles(context.record.indexed_smiles, sanitize=True)
    assert source is not None
    mapped = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in source.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    occurrence_indices = frozenset(
        mapped[atom_map] for atom_map in _occurrence_maps(patch)
    )
    request = _candidate_request(context)

    for tampered_smiles in ("C~C~C~C~C.C", "C~C~C~C~C~C"):
        query = Chem.MolFromSmiles(tampered_smiles, sanitize=True)
        assert query is not None
        matched_sets = {
            frozenset(match)
            for match in source.GetSubstructMatches(
                query,
                uniquify=False,
                maxMatches=10000,
            )
        }
        assert occurrence_indices in matched_sets
        with pytest.raises(ValueError):
            replay_edit_action(
                request,
                replace(action, remove_fragment_smiles=tampered_smiles),
            )


def test_all_applicable_origin_operator_pools_honor_cap_and_underfill_stops_selection() -> (
    None
):
    generation = OPERATORS_CONFIG.candidate_generation
    underfilled: tuple[PerturbationContext, CandidatePool] | None = None
    selectable: tuple[PerturbationContext, CandidatePool] | None = None

    for joined in _delete_records():
        cases = (
            OPERATOR_RUNTIME_CASES[8:]
            if joined.anonymous_sample_id == REPLACEMENT_ID
            else OPERATOR_RUNTIME_CASES
        )
        for operator_id, target, policy in cases:
            context, pool = _invoke(
                operator_id,
                target_node_id=target,
                policy=policy,
                joined=joined,
            )
            assert len(pool.candidates) <= generation.candidates_per_recipe_max
            if len(pool.candidates) >= generation.candidates_per_recipe_min:
                assert (
                    CandidateRejectCode.INSUFFICIENT_CANDIDATES.value
                    not in pool.rejection_codes
                )
                selectable = selectable or (context, pool)
            elif pool.candidates:
                assert (
                    CandidateRejectCode.INSUFFICIENT_CANDIDATES.value
                    in pool.rejection_codes
                )
                if operator_id in DELETION_OPERATOR_IDS[:8] and underfilled is None:
                    underfilled = (context, pool)
                    for patch in pool.candidates:
                        assert patch.edit_action is not None
                        assert patch.edit_action.edit_kind is EditKind.DELETION
                        assert patch.metadata["operator_id"] == operator_id
                        _assert_connected_single_boundary_replay(context, patch)

    assert underfilled is not None
    assert selectable is not None
    engine = DeletionCandidateEngine(operators_config=OPERATORS_CONFIG)
    _ = DeletionPerturbator(**_ports(engine))

    underfilled_context, underfilled_pool = underfilled
    with pytest.raises(OperatorRegistryError) as caught:
        engine.select_root_patch(underfilled_context, underfilled_pool)
    assert caught.value.code == CandidateRejectCode.INSUFFICIENT_CANDIDATES.value
    assert caught.value.evidence == {
        "actual": len(underfilled_pool.candidates),
        "minimum": generation.candidates_per_recipe_min,
    }

    selectable_context, selectable_pool = selectable
    assert (
        engine.select_root_patch(selectable_context, selectable_pool)
        == (selectable_pool.candidates[0])
    )


def test_count_candidates_are_descriptor_derived_not_fixed_plus_or_minus_one() -> None:
    heavy_differences: set[int] = set()
    ring_differences: set[int] = set()
    for joined in _delete_records():
        for operator_id, target, accumulator in (
            (DELETION_OPERATOR_IDS[9], "heavy_delta", heavy_differences),
            (DELETION_OPERATOR_IDS[10], "ring_delta", ring_differences),
        ):
            context, pool = _invoke(
                operator_id,
                target_node_id=target,
                joined=joined,
            )
            old = context.reference_graph.value_for(target).normalized_value
            accumulator.update(
                patch.new_value.normalized_value - old for patch in pool.candidates
            )

    assert len(heavy_differences) >= 4
    assert any(abs(delta) > 1 for delta in heavy_differences)
    assert len(ring_differences) >= 2
    assert any(abs(delta) > 1 for delta in ring_differences)


@pytest.mark.parametrize("root", ("remove_group_step1", "remove_group_step2"))
def test_cross_step_group_identity_changes_one_claim_without_structural_action(
    root: str,
) -> None:
    _, context, pool = _first_nonempty(
        DELETION_OPERATOR_IDS[8],
        target_node_id=root,
    )
    other = (
        "remove_group_step2" if root == "remove_group_step1" else "remove_group_step1"
    )
    other_reference = context.reference_graph.value_for(other)

    for patch in pool.candidates:
        assert patch.root_node_id == root
        assert patch.edit_action is None
        assert not patch.new_value.semantically_equals(other_reference)


@pytest.mark.parametrize(
    "anonymous_sample_id",
    (
        REPLACEMENT_ID,
        "ordinary",
    ),
)
def test_terminal_is_exact_actionless_valid_wrong_final_answer(
    anonymous_sample_id: str,
) -> None:
    joined = (
        next(
            record
            for record in _delete_records()
            if record.anonymous_sample_id != REPLACEMENT_ID
        )
        if anonymous_sample_id == "ordinary"
        else next(
            record
            for record in _delete_records()
            if record.anonymous_sample_id == anonymous_sample_id
        )
    )
    context, pool = _invoke(
        DELETION_OPERATOR_IDS[11],
        target_node_id="final_answer",
        policy=PropagationPolicy.TERMINAL,
        source=CandidateSourceType.RDKIT,
        joined=joined,
    )

    assert pool.candidates
    for patch in pool.candidates:
        assert patch.root_node_id == "final_answer"
        assert patch.old_value == context.reference_graph.value_for("final_answer")
        assert patch.edit_action is None
        candidate = patch.new_value.normalized_value
        assert canonicalize_smiles(candidate) == candidate
        assert not isomeric_graph_equivalent(candidate, context.truth.gt_smiles)
