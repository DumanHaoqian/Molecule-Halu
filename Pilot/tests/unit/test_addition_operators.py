"""T019 concrete Addition operator contracts and chemistry regressions."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import build_reference_dag, derive_edit_truth
from molhallulens.modules.error_planning import (
    CandidateRejectCode,
    CandidateRequest,
    replay_edit_action,
)
from molhallulens.infrastructure.chemistry import (
    canonicalize_smiles,
    compute_descriptors,
    isomeric_graph_equivalent,
)
from molhallulens.config import load_config_bundle
from molhallulens.core import (
    BondTypeName,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    EditErrorSubtype,
    EditKind,
    HallucinationType,
    OperatorCapability,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    ValueProvenance,
)
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    CandidateEngine,
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
from molhallulens.modules.error_injection.operators.addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateDispatcher,
    AdditionCandidateEngine,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators

EXPECTED_METHODS = (
    "perturb_alternate_anchor_same_element",
    "perturb_neighborhood_matched_anchor",
    "perturb_fragment_bucket_swap",
    "perturb_fragment_attachment_atom",
    "perturb_attachment_bond_order",
    "perturb_valid_wrong_site_product",
    "perturb_valid_regioisomer_product",
    "perturb_heavy_count_claim",
    "perturb_ring_count_claim",
    "perturb_internal_relation_claim",
    "perturb_terminal_answer",
)

ALL_POLICIES = frozenset(
    {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
)
CLAIM_POLICIES = frozenset({PropagationPolicy.STOP, PropagationPolicy.PARTIAL})
DETERMINISTIC_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
COUNT_SOURCES = DETERMINISTIC_SOURCES
RELATION_SOURCES = frozenset({CandidateSourceType.RULE, CandidateSourceType.HYBRID})
COMMON_HALLUCINATIONS = frozenset(
    {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
)

EXPECTED_METADATA = {
    ADDITION_OPERATOR_IDS[0]: (
        EXPECTED_METHODS[0],
        "wrong_anchor_site",
        frozenset({"anchor_idx"}),
        ALL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[1]: (
        EXPECTED_METHODS[1],
        "wrong_anchor_site",
        frozenset({"anchor_idx"}),
        ALL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[2]: (
        EXPECTED_METHODS[2],
        "wrong_fragment_group",
        frozenset({"add_fragment"}),
        ALL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ADD_FRAGMENT_IDENTIFICATION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[3]: (
        EXPECTED_METHODS[3],
        "attachment_bond_edit",
        frozenset({"product"}),
        ALL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[4]: (
        EXPECTED_METHODS[4],
        "attachment_bond_edit",
        frozenset({"product"}),
        ALL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[5]: (
        EXPECTED_METHODS[5],
        "wrong_anchor_site",
        frozenset({"product"}),
        frozenset({PropagationPolicy.FULL_CF}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[6]: (
        EXPECTED_METHODS[6],
        "wrong_anchor_site",
        frozenset({"product"}),
        frozenset({PropagationPolicy.FULL_CF}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[7]: (
        EXPECTED_METHODS[7],
        "numeric_count_claim",
        frozenset(
            {"fragment_heavy", "source_heavy", "product_heavy", "heavy_delta"}
        ),
        CLAIM_POLICIES,
        COUNT_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset(
            {EditErrorSubtype.HEAVY_ATOM_COUNT, EditErrorSubtype.HEAVY_ATOM_ARITHMETIC}
        ),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[8]: (
        EXPECTED_METHODS[8],
        "numeric_count_claim",
        frozenset({"source_rings", "product_rings", "ring_delta"}),
        CLAIM_POLICIES,
        COUNT_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[9]: (
        EXPECTED_METHODS[9],
        "nl_formal_internal_relation",
        frozenset({"anchor_element"}),
        CLAIM_POLICIES,
        RELATION_SOURCES,
        frozenset({HallucinationType.REASONING_ERROR}),
        frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    ADDITION_OPERATOR_IDS[10]: (
        EXPECTED_METHODS[10],
        "final_answer_identity",
        frozenset({"final_answer"}),
        frozenset({PropagationPolicy.TERMINAL}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY}),
        frozenset({OperatorCapability.TERMINAL_PERTURBATION}),
    ),
}

OPERATOR_RUNTIME_CASES = (
    (ADDITION_OPERATOR_IDS[0], "anchor_idx", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[1], "anchor_idx", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[2], "add_fragment", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[3], "product", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[4], "product", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[5], "product", PropagationPolicy.FULL_CF),
    (ADDITION_OPERATOR_IDS[6], "product", PropagationPolicy.FULL_CF),
    (ADDITION_OPERATOR_IDS[7], "heavy_delta", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[8], "ring_delta", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[9], "anchor_element", PropagationPolicy.STOP),
    (ADDITION_OPERATOR_IDS[10], "final_answer", PropagationPolicy.TERMINAL),
)


class _UnusedPropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T019 operator tests do not execute T022 propagation")


class _UnusedRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T019 operator tests do not execute rendering")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T019 operator tests do not execute the full template")

    def validate_artifact(self, draft):
        raise AssertionError("T019 operator tests do not validate rendered artifacts")


class _UnusedLabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T019 operator tests do not project token labels")


def _ports(candidate_engine: CandidateEngine) -> dict[str, object]:
    return {
        "candidate_engine": candidate_engine,
        "propagator": _UnusedPropagationEngine(),
        "renderer": _UnusedRenderer(),
        "validators": _UnusedValidators(),
        "label_projector": _UnusedLabelProjector(),
    }


@cache
def _add_records() -> tuple[JoinedInputRecord, ...]:
    return tuple(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".add_v2." in record.anonymous_sample_id
    )


@cache
def _origin_artifacts(anonymous_sample_id: str):
    joined = next(
        record
        for record in _add_records()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    truth = derive_edit_truth(artifact)
    record = task_record_from_joined_input(joined)
    return joined, artifact, truth, record


def _registry() -> PerturbatorRegistry:
    return PerturbatorRegistry.from_perturbator_types(
        (AdditionPerturbator,),
        operators_config=OPERATORS_CONFIG,
    )


def _production_perturbator() -> AdditionPerturbator:
    return AdditionPerturbator(
        **_ports(AdditionCandidateEngine(operators_config=OPERATORS_CONFIG))
    )


def _context(
    registration: OperatorRegistration,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
    joined: JoinedInputRecord | None = None,
) -> PerturbationContext:
    selected = joined or _add_records()[0]
    _, artifact, truth, record = _origin_artifacts(selected.anonymous_sample_id)
    root = target_node_id or min(registration.spec.root_fields)
    if policy is not None:
        selected_policy = policy
    elif PropagationPolicy.TERMINAL in registration.spec.supported_policies:
        selected_policy = PropagationPolicy.TERMINAL
    elif PropagationPolicy.STOP in registration.spec.supported_policies:
        selected_policy = PropagationPolicy.STOP
    else:
        selected_policy = min(
            registration.spec.supported_policies,
            key=lambda item: item.value,
        )
    recipe = PerturbationRecipe(
        recipe_id=f"t019:{record.origin_id}:{registration.operator_id}:{root}",
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
            length_bucket="t019",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
    )
    return PerturbationContext(
        record=record,
        recipe=recipe,
        state_schema=artifact.state_dag.schema,
        reference_graph=artifact.state_dag,
        truth=truth,
    )


def _mapped_source_environment(
    indexed_smiles: str,
) -> tuple[Chem.Mol, dict[int, int], tuple[int, ...]]:
    molecule = Chem.MolFromSmiles(indexed_smiles, sanitize=True)
    assert molecule is not None
    mapped = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    map_free = Chem.Mol(molecule)
    for atom in map_free.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(map_free, cleanIt=True, force=True)
    ranks = tuple(
        Chem.CanonicalRankAtoms(
            map_free,
            breakTies=False,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    return molecule, mapped, ranks


def _neighborhood_signature(molecule: Chem.Mol, atom_index: int) -> tuple[Any, ...]:
    atom = molecule.GetAtomWithIdx(atom_index)
    return (
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetDegree(),
        str(atom.GetHybridization()),
        tuple(
            sorted(
                (
                    neighbor.GetAtomicNum(),
                    str(
                        molecule.GetBondBetweenAtoms(
                            atom_index, neighbor.GetIdx()
                        ).GetBondType()
                    ),
                )
                for neighbor in atom.GetNeighbors()
            )
        ),
    )


def _invoke(
    operator_id: str,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
    joined: JoinedInputRecord | None = None,
) -> tuple[PerturbationContext, CandidatePool]:
    registration = _registry().registration(operator_id)
    context = _context(
        registration,
        target_node_id=target_node_id,
        policy=policy,
        source=source,
        joined=joined,
    )
    perturbator = _production_perturbator()
    pool = perturbator.candidate_engine.enumerate_root_patches(context)
    return context, pool


def _first_nonempty(
    operator_id: str,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
) -> tuple[JoinedInputRecord, PerturbationContext, CandidatePool]:
    observed_rejections: set[str] = set()
    for joined in _add_records():
        context, pool = _invoke(
            operator_id,
            target_node_id=target_node_id,
            policy=policy,
            source=source,
            joined=joined,
        )
        if pool.candidates:
            return joined, context, pool
        observed_rejections.update(pool.rejection_codes)
    pytest.fail(
        f"{operator_id} produced no candidate for 50 Addition origins; "
        f"rejections={sorted(observed_rejections)!r}"
    )


def _candidate_request(context: PerturbationContext) -> CandidateRequest:
    perturbator = _production_perturbator()
    resolution = _registry().resolve(perturbator, context)
    return CandidateRequest(context=context, resolution=resolution)


def test_addition_declares_exactly_the_eleven_blueprint_operator_members() -> None:
    registrations = _registry().registrations_for(task_family="mol_edit", subtask="add")

    assert ADDITION_OPERATOR_IDS == tuple(
        f"mol_edit.add.{method.removeprefix('perturb_')}"
        for method in EXPECTED_METHODS
    )
    assert {registration.operator_id for registration in registrations} == set(
        ADDITION_OPERATOR_IDS
    )
    for registration in registrations:
        (
            method_name,
            family,
            roots,
            policies,
            sources,
            hallucination_types,
            edit_subtypes,
            capabilities,
        ) = EXPECTED_METADATA[registration.operator_id]
        assert registration.method_name == method_name
        assert registration.operator_family == family
        assert registration.spec.root_fields == roots
        assert registration.spec.supported_policies == policies
        assert registration.spec.supported_sources == sources
        assert registration.spec.hallucination_types == hallucination_types
        assert registration.edit_subtypes == edit_subtypes
        assert registration.required_capabilities == capabilities
        assert registration.spec.diagnostic_only is False


def test_addition_engine_implements_t016_port_and_dispatches_through_registry() -> None:
    engine = AdditionCandidateEngine(operators_config=OPERATORS_CONFIG)
    assert isinstance(engine, CandidateEngine)
    perturbator = AdditionPerturbator(**_ports(engine))
    registration = _registry().registration(ADDITION_OPERATOR_IDS[0])
    context = _context(registration)

    direct = engine.enumerate_root_patches(context)
    dispatched = AdditionCandidateDispatcher(OPERATORS_CONFIG).invoke(
        perturbator, context
    )

    assert direct == dispatched
    assert direct.request_id == context.recipe.recipe_id
    for patch in direct.candidates:
        assert patch.root_node_id == context.recipe.target_node_id
        assert patch.old_value == context.reference_graph.value_for(
            context.recipe.target_node_id
        )
    if direct.candidates:
        assert engine.select_root_patch(context, direct) == direct.candidates[0]
    else:
        with pytest.raises(ValueError, match="empty"):
            engine.select_root_patch(context, direct)


def test_llm_source_is_rejected_before_any_addition_member_is_invoked() -> None:
    registry = _registry()
    engine = AdditionCandidateEngine(operators_config=OPERATORS_CONFIG)
    perturbator = AdditionPerturbator(**_ports(engine))

    for registration in registry.registrations_for(
        task_family="mol_edit", subtask="add"
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
def test_each_operator_emits_only_bound_root_patches_with_stable_rule_provenance(
    operator_id: str,
    target: str,
    policy: PropagationPolicy,
) -> None:
    joined, context, pool = _first_nonempty(
        operator_id,
        target_node_id=target,
        policy=policy,
    )

    assert 1 <= len(pool.candidates) <= OPERATORS_CONFIG.candidate_generation.candidates_per_recipe_max
    for patch in pool.candidates:
        assert patch.root_node_id == target
        assert patch.old_value == context.reference_graph.value_for(target)
        assert not patch.old_value.semantically_equals(patch.new_value)
        assert patch.source is CandidateSourceType.RULE
        assert patch.new_value.provenance is ValueProvenance.RULE
        assert patch.metadata["generator"] == "addition_t019"
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
                assert patch.new_value.provenance is {
                    CandidateSourceType.RULE: ValueProvenance.RULE,
                    CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
                }[patch.source]


@pytest.mark.parametrize(
    "source",
    (
        CandidateSourceType.RULE,
        CandidateSourceType.RDKIT,
        CandidateSourceType.HYBRID,
    ),
)
def test_rule_rdkit_and_hybrid_preserve_exact_candidate_source_provenance(
    source: CandidateSourceType,
) -> None:
    _, _, pool = _first_nonempty(
        ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        source=source,
    )

    admitted = {
        CandidateSourceType.RULE: {CandidateSourceType.RULE},
        CandidateSourceType.RDKIT: {CandidateSourceType.RDKIT},
        CandidateSourceType.HYBRID: {
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
        },
    }[source]
    provenance = {
        CandidateSourceType.RULE: ValueProvenance.RULE,
        CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
    }
    assert pool.candidates
    assert {patch.source for patch in pool.candidates} <= admitted
    for patch in pool.candidates:
        assert patch.new_value.provenance is provenance[patch.source]


@pytest.mark.parametrize(
    ("operator_id", "requires_neighborhood_match"),
    (
        (ADDITION_OPERATOR_IDS[0], False),
        (ADDITION_OPERATOR_IDS[1], True),
    ),
)
def test_wrong_anchor_is_same_element_but_not_valid_or_automorphic(
    operator_id: str,
    requires_neighborhood_match: bool,
) -> None:
    joined, context, pool = _first_nonempty(
        operator_id,
        target_node_id="anchor_idx",
    )
    molecule, mapped, ranks = _mapped_source_environment(joined.raw_record["indexed_smiles"])
    valid = context.truth.valid_anchor_indices
    valid_atomic_numbers = {
        molecule.GetAtomWithIdx(mapped[anchor]).GetAtomicNum() for anchor in valid
    }
    valid_ranks = {ranks[mapped[anchor]] for anchor in valid}
    valid_neighborhoods = {
        _neighborhood_signature(molecule, mapped[anchor]) for anchor in valid
    }

    assert pool.candidates
    for patch in pool.candidates:
        candidate_anchor = patch.new_value.normalized_value
        assert type(candidate_anchor) is int
        assert candidate_anchor in mapped
        assert candidate_anchor not in valid
        candidate_atom = molecule.GetAtomWithIdx(mapped[candidate_anchor])
        assert candidate_atom.GetAtomicNum() in valid_atomic_numbers
        assert ranks[mapped[candidate_anchor]] not in valid_ranks
        assert patch.edit_action is not None
        assert patch.edit_action.source_anchor_index == candidate_anchor
        if requires_neighborhood_match:
            assert (
                _neighborhood_signature(molecule, mapped[candidate_anchor])
                in valid_neighborhoods
            )


def test_fragment_bucket_swap_carries_a_replayable_wrong_fragment_identity() -> None:
    _, context, pool = _first_nonempty(
        ADDITION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        source=CandidateSourceType.RDKIT,
    )
    truth_fragment = context.truth.add_fragment
    assert truth_fragment is not None
    request = _candidate_request(context)

    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        assert action.edit_kind is EditKind.ADDITION
        assert action.add_fragment_smiles == patch.new_value.normalized_value
        assert canonicalize_smiles(action.add_fragment_smiles) == action.add_fragment_smiles
        assert not isomeric_graph_equivalent(
            action.add_fragment_smiles, truth_fragment.canonical_smiles
        )
        fragment = Chem.MolFromSmiles(action.add_fragment_smiles, sanitize=True)
        assert fragment is not None
        assert action.fragment_attachment_atom is not None
        assert 0 <= action.fragment_attachment_atom < fragment.GetNumAtoms()
        products = replay_edit_action(request, action)
        assert products
        assert all(
            not isomeric_graph_equivalent(product, context.truth.gt_smiles)
            for product in products
        )


def _assert_action_replays_patch(
    context: PerturbationContext,
    patch: CandidatePatch,
) -> None:
    assert patch.edit_action is not None
    products = replay_edit_action(_candidate_request(context), patch.edit_action)
    assert products
    assert any(
        isomeric_graph_equivalent(product, patch.new_value.normalized_value)
        for product in products
    )


@pytest.mark.parametrize(
    ("operator_id", "changes_bond_order"),
    (
        (ADDITION_OPERATOR_IDS[3], False),
        (ADDITION_OPERATOR_IDS[4], True),
    ),
)
def test_attachment_operators_retain_fragment_atom_bond_and_replay_semantics(
    operator_id: str,
    changes_bond_order: bool,
) -> None:
    _, context, pool = _first_nonempty(
        operator_id,
        target_node_id="product",
        source=CandidateSourceType.RDKIT,
    )
    truth_fragment = context.truth.add_fragment
    assert truth_fragment is not None
    reference_bonds = {bond.bond_type for bond in context.truth.formed_bonds}

    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        assert action.edit_kind is EditKind.ADDITION
        assert action.source_anchor_index in context.truth.valid_anchor_indices
        assert action.add_fragment_smiles is not None
        assert isomeric_graph_equivalent(
            action.add_fragment_smiles, truth_fragment.canonical_smiles
        )
        fragment = Chem.MolFromSmiles(action.add_fragment_smiles, sanitize=True)
        assert fragment is not None
        assert action.fragment_attachment_atom is not None
        assert 0 <= action.fragment_attachment_atom < fragment.GetNumAtoms()
        assert type(action.bond_type) is BondTypeName
        if changes_bond_order:
            assert action.bond_type not in reference_bonds
        else:
            assert action.bond_type in reference_bonds
        _assert_action_replays_patch(context, patch)
        assert not isomeric_graph_equivalent(
            patch.new_value.normalized_value, context.truth.gt_smiles
        )


@pytest.mark.parametrize(
    "operator_id",
    (ADDITION_OPERATOR_IDS[5], ADDITION_OPERATOR_IDS[6]),
)
def test_full_cf_products_are_real_sanitized_rdkit_replays_and_not_gt(
    operator_id: str,
) -> None:
    _, context, pool = _first_nonempty(
        operator_id,
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
        source=CandidateSourceType.RDKIT,
    )

    assert context.recipe.policy is PropagationPolicy.FULL_CF
    assert pool.candidates
    for patch in pool.candidates:
        assert patch.source is CandidateSourceType.RDKIT
        assert patch.new_value.provenance is ValueProvenance.RDKIT
        assert patch.root_node_id == "product"
        assert patch.edit_action is not None
        assert patch.edit_action.edit_kind is EditKind.ADDITION
        assert (
            patch.edit_action.source_anchor_index
            not in context.truth.valid_anchor_indices
        )
        candidate_smiles = patch.new_value.normalized_value
        assert canonicalize_smiles(candidate_smiles) == candidate_smiles
        assert not isomeric_graph_equivalent(candidate_smiles, context.truth.gt_smiles)
        _assert_action_replays_patch(context, patch)


def test_all_fifty_add_origins_keep_none_leaving_out_of_core_operators() -> None:
    assert len(_add_records()) == 50
    for joined in _add_records():
        _, artifact, truth, _ = _origin_artifacts(joined.anonymous_sample_id)
        leaving = artifact.state_dag.value_for("leaving")
        assert str(leaving.normalized_value).casefold() == "none"
        assert truth.remove_fragment is None
        assert not truth.removed_atom_maps

    for registration in _registry().registrations_for(
        task_family="mol_edit", subtask="add"
    ):
        assert "leaving" not in registration.spec.root_fields
        assert (
            EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION
            not in registration.edit_subtypes
        )


def test_count_candidates_do_not_collapse_to_a_fixed_plus_or_minus_one_rule() -> None:
    heavy_differences: set[int] = set()
    ring_differences: set[int] = set()

    for joined in _add_records():
        for operator_id, target, accumulator in (
            (ADDITION_OPERATOR_IDS[7], "heavy_delta", heavy_differences),
            (ADDITION_OPERATOR_IDS[8], "ring_delta", ring_differences),
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


def test_all_origin_operator_pools_honor_cap_and_underfill_fails_selection() -> None:
    generation = OPERATORS_CONFIG.candidate_generation
    structural_ids = frozenset(ADDITION_OPERATOR_IDS[:7])
    underfilled: tuple[PerturbationContext, CandidatePool] | None = None
    selectable: tuple[PerturbationContext, CandidatePool] | None = None

    for joined in _add_records():
        for operator_id, target, policy in OPERATOR_RUNTIME_CASES:
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
                if operator_id in structural_ids and underfilled is None:
                    underfilled = (context, pool)
                    request = _candidate_request(context)
                    for patch in pool.candidates:
                        assert patch.root_node_id == target
                        assert patch.edit_action is not None
                        assert patch.edit_action.edit_kind is EditKind.ADDITION
                        assert patch.metadata["operator_id"] == operator_id
                        products = replay_edit_action(request, patch.edit_action)
                        assert products
                        assert all(
                            not isomeric_graph_equivalent(
                                product, context.truth.gt_smiles
                            )
                            for product in products
                        )

    assert underfilled is not None
    assert selectable is not None
    selection_engine = AdditionCandidateEngine(operators_config=OPERATORS_CONFIG)
    _ = AdditionPerturbator(**_ports(selection_engine))

    underfilled_context, underfilled_pool = underfilled
    with pytest.raises(OperatorRegistryError) as caught:
        selection_engine.select_root_patch(underfilled_context, underfilled_pool)
    assert caught.value.code == CandidateRejectCode.INSUFFICIENT_CANDIDATES.value
    assert caught.value.evidence == {
        "actual": len(underfilled_pool.candidates),
        "minimum": generation.candidates_per_recipe_min,
    }

    selectable_context, selectable_pool = selectable
    assert (
        selection_engine.select_root_patch(selectable_context, selectable_pool)
        == selectable_pool.candidates[0]
    )


def test_ring_count_uses_a_real_three_ring_descriptor_and_meets_minimum() -> None:
    anthracene = "c1ccc2cc3ccccc3cc2c1"
    assert compute_descriptors(anthracene).ring_count == 3
    joined = next(
        record
        for record in _add_records()
        if _origin_artifacts(record.anonymous_sample_id)[1]
        .state_dag.value_for("ring_delta")
        .normalized_value
        != 3
    )
    context, pool = _invoke(
        ADDITION_OPERATOR_IDS[8],
        target_node_id="ring_delta",
        joined=joined,
    )
    values = {patch.new_value.normalized_value for patch in pool.candidates}

    assert 3 in values
    assert len(pool.candidates) >= (
        OPERATORS_CONFIG.candidate_generation.candidates_per_recipe_min
    )
    assert CandidateRejectCode.INSUFFICIENT_CANDIDATES.value not in pool.rejection_codes
    assert context.reference_graph.value_for("ring_delta").normalized_value != 3


def test_terminal_operator_changes_exactly_final_answer_with_a_valid_near_miss() -> None:
    _, context, pool = _first_nonempty(
        ADDITION_OPERATOR_IDS[10],
        target_node_id="final_answer",
        policy=PropagationPolicy.TERMINAL,
        source=CandidateSourceType.RDKIT,
    )

    assert context.recipe.policy is PropagationPolicy.TERMINAL
    assert context.recipe.target_node_id == "final_answer"
    for patch in pool.candidates:
        assert patch.root_node_id == "final_answer"
        assert patch.old_value == context.reference_graph.value_for("final_answer")
        assert patch.edit_action is not None
        assert patch.edit_action.edit_kind is EditKind.ADDITION
        candidate_smiles = patch.new_value.normalized_value
        assert canonicalize_smiles(candidate_smiles) == candidate_smiles
        assert not isomeric_graph_equivalent(candidate_smiles, context.truth.gt_smiles)
        _assert_action_replays_patch(context, patch)
