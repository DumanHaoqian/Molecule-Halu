"""T021 concrete Substitution operator and dual-anchor graph-edit contracts."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
from rdkit import Chem

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import build_reference_dag, derive_edit_truth
from molhallulens.modules.error_planning import (
    CandidateDifficultyFeatures,
    CandidateProposal,
    CandidateRejectCode,
    CandidateRequest,
    DeterministicCandidateEngine,
    RuleCandidateSource,
    canonical_candidate_key,
    replay_edit_action,
)
from molhallulens.infrastructure.chemistry import (
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
)
from molhallulens.config import load_config_bundle
from molhallulens.core import (
    AtomReferenceNamespace,
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
from molhallulens.modules.error_injection import (
    CandidateEngine,
    LabelProjector,
    OperatorRegistration,
    OperatorRegistryError,
    PerturbationContext,
    PerturbatorRegistry,
    PropagationEngine,
    SubstitutionPerturbator,
    TraceRenderer,
    ValidatorChain,
    task_record_from_joined_input,
)
from molhallulens.modules.error_injection.operators import substitution as substitution_module
from molhallulens.modules.error_injection.operators.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
    SubstitutionCandidateDispatcher,
    SubstitutionCandidateEngine,
)

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators

EXPECTED_METHODS = (
    "perturb_alternate_substitution_site",
    "perturb_wrong_leaving_occurrence",
    "perturb_incoming_fragment_bucket_swap",
    "perturb_fragment_attachment_atom",
    "perturb_attachment_bond_order",
    "perturb_leaving_group_swap",
    "perturb_partial_substitution",
    "perturb_valid_wrong_regioisomer",
    "perturb_add_remove_role_claim",
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
    SUBSTITUTION_OPERATOR_IDS[0]: (
        EXPECTED_METHODS[0],
        "wrong_anchor_site",
        frozenset({"anchor_idx"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[1]: (
        EXPECTED_METHODS[1],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[2]: (
        EXPECTED_METHODS[2],
        "wrong_fragment_group",
        frozenset({"add_fragment"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ADD_FRAGMENT_IDENTIFICATION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[3]: (
        EXPECTED_METHODS[3],
        "attachment_bond_edit",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[4]: (
        EXPECTED_METHODS[4],
        "attachment_bond_edit",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[5]: (
        EXPECTED_METHODS[5],
        "wrong_fragment_group",
        frozenset({"remove_group"}),
        CLAIM_POLICIES,
        RELATION_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[6]: (
        EXPECTED_METHODS[6],
        "wrong_fragment_group",
        frozenset({"product"}),
        STRUCTURAL_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[7]: (
        EXPECTED_METHODS[7],
        "wrong_anchor_site",
        frozenset({"product"}),
        frozenset({PropagationPolicy.FULL_CF}),
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.PRODUCT_CONSTRUCTION}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[8]: (
        EXPECTED_METHODS[8],
        "nl_formal_internal_relation",
        frozenset({"add_fragment", "remove_group"}),
        CLAIM_POLICIES,
        RELATION_SOURCES,
        frozenset({HallucinationType.REASONING_ERROR}),
        frozenset({EditErrorSubtype.INTERNAL_INCONSISTENCY}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[9]: (
        EXPECTED_METHODS[9],
        "numeric_count_claim",
        frozenset(
            {
                "add_heavy",
                "heavy_delta",
                "product_heavy",
                "remove_heavy",
                "source_heavy",
            }
        ),
        CLAIM_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset(
            {EditErrorSubtype.HEAVY_ATOM_COUNT, EditErrorSubtype.HEAVY_ATOM_ARITHMETIC}
        ),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[10]: (
        EXPECTED_METHODS[10],
        "numeric_count_claim",
        frozenset({"product_rings", "ring_delta", "source_rings"}),
        CLAIM_POLICIES,
        DETERMINISTIC_SOURCES,
        COMMON_HALLUCINATIONS,
        frozenset({EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC}),
        frozenset({OperatorCapability.CLAIM_PERTURBATION}),
    ),
    SUBSTITUTION_OPERATOR_IDS[11]: (
        EXPECTED_METHODS[11],
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
    (SUBSTITUTION_OPERATOR_IDS[0], "anchor_idx", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[1], "product", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[2], "add_fragment", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[3], "product", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[4], "product", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[5], "remove_group", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[6], "product", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[7], "product", PropagationPolicy.FULL_CF),
    (SUBSTITUTION_OPERATOR_IDS[8], "remove_group", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[9], "heavy_delta", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[10], "ring_delta", PropagationPolicy.STOP),
    (SUBSTITUTION_OPERATOR_IDS[11], "final_answer", PropagationPolicy.TERMINAL),
)


class _UnusedPropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T021 tests do not execute T022 propagation")


class _UnusedRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T021 tests do not render output")


class _UnusedValidators(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T021 tests do not execute the full template")

    def validate_artifact(self, draft):
        raise AssertionError("T021 tests do not validate rendered artifacts")


class _UnusedLabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T021 tests do not project labels")


def _ports(candidate_engine: CandidateEngine) -> dict[str, object]:
    return {
        "candidate_engine": candidate_engine,
        "propagator": _UnusedPropagationEngine(),
        "renderer": _UnusedRenderer(),
        "validators": _UnusedValidators(),
        "label_projector": _UnusedLabelProjector(),
    }


@cache
def _substitution_records() -> tuple[JoinedInputRecord, ...]:
    return tuple(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if ".substitute_v2." in record.anonymous_sample_id
    )


@cache
def _origin_artifacts(anonymous_sample_id: str):
    joined = next(
        record
        for record in _substitution_records()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    truth = derive_edit_truth(artifact)
    record = task_record_from_joined_input(joined)
    return joined, artifact, truth, record


def _registry() -> PerturbatorRegistry:
    return PerturbatorRegistry.from_perturbator_types(
        (SubstitutionPerturbator,),
        operators_config=OPERATORS_CONFIG,
    )


def _production_perturbator() -> SubstitutionPerturbator:
    return SubstitutionPerturbator(
        **_ports(SubstitutionCandidateEngine(operators_config=OPERATORS_CONFIG))
    )


def _context(
    registration: OperatorRegistration,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
    joined: JoinedInputRecord | None = None,
) -> PerturbationContext:
    selected = joined or _substitution_records()[0]
    _, artifact, truth, record = _origin_artifacts(selected.anonymous_sample_id)
    root = target_node_id or min(registration.spec.root_fields)
    selected_policy = policy or (
        PropagationPolicy.TERMINAL
        if PropagationPolicy.TERMINAL in registration.spec.supported_policies
        else PropagationPolicy.STOP
    )
    recipe = PerturbationRecipe(
        recipe_id=f"t021:{record.origin_id}:{registration.operator_id}:{root}",
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
            length_bucket="t021",
        ),
        candidate_difficulty_bucket="hard",
        renderer_style_id="fixture",
        partial_cut_nodes=(
            frozenset({"product"})
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


def _candidate_request(context: PerturbationContext) -> CandidateRequest:
    perturbator = _production_perturbator()
    return CandidateRequest(
        context=context,
        resolution=_registry().resolve(perturbator, context),
    )


def _first_nonempty(
    operator_id: str,
    *,
    target_node_id: str | None = None,
    policy: PropagationPolicy | None = None,
    source: CandidateSourceType = CandidateSourceType.RULE,
) -> tuple[JoinedInputRecord, PerturbationContext, CandidatePool]:
    rejection_codes: set[str] = set()
    for joined in _substitution_records():
        context, pool = _invoke(
            operator_id,
            target_node_id=target_node_id,
            policy=policy,
            source=source,
            joined=joined,
        )
        if pool.candidates:
            return joined, context, pool
        rejection_codes.update(pool.rejection_codes)
    pytest.fail(
        f"{operator_id} produced no candidate for 50 substitutions; "
        f"rejections={sorted(rejection_codes)!r}"
    )


def _occurrence_maps(action: EditAction) -> tuple[int, ...]:
    primary = action.metadata.get("remove_atom_maps")
    alternate = action.metadata.get("occurrence_atom_maps")
    assert primary is None or alternate is None or primary == alternate
    occurrence = primary if primary is not None else alternate
    assert type(occurrence) is tuple
    assert occurrence
    assert tuple(sorted(set(occurrence))) == occurrence
    assert all(type(atom_map) is int and atom_map > 0 for atom_map in occurrence)
    return occurrence


def _source_boundary(
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
    boundaries: list[tuple[int, int, BondTypeName]] = []
    rdkit_bonds = {
        Chem.BondType.SINGLE: BondTypeName.SINGLE,
        Chem.BondType.DOUBLE: BondTypeName.DOUBLE,
        Chem.BondType.TRIPLE: BondTypeName.TRIPLE,
        Chem.BondType.AROMATIC: BondTypeName.AROMATIC,
    }
    for removed_map in occurrence:
        atom = molecule.GetAtomWithIdx(mapped[removed_map])
        for bond in atom.GetBonds():
            neighbor_map = bond.GetOtherAtom(atom).GetAtomMapNum()
            if neighbor_map not in removed:
                boundaries.append(
                    (removed_map, neighbor_map, rdkit_bonds[bond.GetBondType()])
                )
    return molecule, mapped, tuple(sorted(set(boundaries)))


def _assert_replayable_patch(
    context: PerturbationContext,
    patch: CandidatePatch,
) -> str:
    action = patch.edit_action
    assert action is not None
    assert action.edit_kind is EditKind.SUBSTITUTION
    assert action.source_anchor_index is not None
    assert action.remove_anchor_index is None or action.remove_anchor_index > 0
    assert action.fragment_attachment_atom is not None
    assert action.bond_type is not None
    occurrence = _occurrence_maps(action)
    _, _, boundaries = _source_boundary(context.record.indexed_smiles, occurrence)
    assert len(boundaries) == 1
    expected_remove_anchor = action.remove_anchor_index or action.source_anchor_index
    assert boundaries[0][1] == expected_remove_anchor
    products = replay_edit_action(_candidate_request(context), action)
    assert products
    product = products[0]
    assert not isomeric_graph_equivalent(product, context.truth.gt_smiles)
    if patch.root_node_id in {"product", "final_answer"}:
        assert isomeric_graph_equivalent(product, patch.new_value.normalized_value)
    elif patch.root_node_id == "anchor_idx":
        assert patch.new_value.normalized_value == action.source_anchor_index
    elif patch.root_node_id == "add_fragment":
        assert fragment_graph_equivalent(
            patch.new_value.normalized_value,
            action.add_fragment_smiles,
        )
    elif patch.root_node_id == "remove_group":
        assert fragment_graph_equivalent(
            patch.new_value.normalized_value,
            action.remove_fragment_smiles,
        )
    return product


def _reference_actions(
    context: PerturbationContext, *, dual_anchor: bool
) -> tuple[EditAction, ...]:
    truth = context.truth
    assert truth.remove_fragment is not None and truth.add_fragment is not None
    assert len(truth.broken_bonds) == len(truth.formed_bonds) == 1
    broken = truth.broken_bonds[0]
    formed = truth.formed_bonds[0]
    removed = tuple(sorted(truth.removed_atom_maps))
    remove_anchor = next(
        endpoint.atom_id
        for endpoint in (broken.begin, broken.end)
        if endpoint.namespace is AtomReferenceNamespace.SOURCE_MAP
        and endpoint.atom_id not in truth.removed_atom_maps
    )
    formed_source_anchors = tuple(
        endpoint.atom_id
        for endpoint in (formed.begin, formed.end)
        if endpoint.namespace is AtomReferenceNamespace.SOURCE_MAP
    )
    add_molecule = Chem.MolFromSmiles(
        truth.add_fragment.canonical_smiles, sanitize=True
    )
    assert add_molecule is not None
    request = _candidate_request(context)
    occurrence = substitution_module._RemovalOccurrence(
        atom_maps=removed,
        anchor_map=remove_anchor,
        fragment_smiles=truth.remove_fragment.canonical_smiles,
    )
    candidates = []
    for add_anchor in tuple(
        dict.fromkeys((*formed_source_anchors, *truth.valid_anchor_indices))
    ):
        if not dual_anchor and add_anchor != remove_anchor:
            continue
        for attachment_atom in range(add_molecule.GetNumAtoms()):
            candidates.append(
                substitution_module._action(
                    request,
                    add_anchor=add_anchor,
                    occurrence=occurrence,
                    add_fragment=truth.add_fragment.canonical_smiles,
                    attachment_atom=attachment_atom,
                    bond_type=formed.bond_type,
                )
            )
    return tuple(candidates)


def _reference_replay_matches(
    context: PerturbationContext,
    *,
    dual_anchor: bool,
) -> tuple[EditAction, ...]:
    request = _candidate_request(context)
    matches = []
    for action in _reference_actions(context, dual_anchor=dual_anchor):
        try:
            products = replay_edit_action(request, action)
        except (RuntimeError, TypeError, ValueError):
            continue
        if any(
            isomeric_graph_equivalent(product, context.truth.gt_smiles)
            for product in products
        ):
            matches.append(action)
    return tuple(matches)


def test_exact_twelve_operator_metadata_and_registry_binding() -> None:
    assert SUBSTITUTION_OPERATOR_IDS == tuple(
        f"mol_edit.substitute.{method.removeprefix('perturb_')}"
        for method in EXPECTED_METHODS
    )
    registry = _registry()
    registrations = registry.registrations_for(
        task_family="mol_edit", subtask="substitute"
    )
    assert tuple(item.operator_id for item in registrations) == tuple(
        sorted(SUBSTITUTION_OPERATOR_IDS)
    )
    for operator_id, expected in EXPECTED_METADATA.items():
        registration = registry.registration(operator_id)
        (
            method_name,
            family,
            roots,
            policies,
            sources,
            hallucinations,
            edit_subtypes,
            capabilities,
        ) = expected
        assert registration.perturbator_type is SubstitutionPerturbator
        assert registration.task_family == "mol_edit"
        assert registration.subtask == "substitute"
        assert registration.method_name == method_name
        assert registration.operator_family == family
        assert registration.spec.root_fields == roots
        assert registration.spec.supported_policies == policies
        assert registration.spec.supported_sources == sources
        assert registration.spec.hallucination_types == hallucinations
        assert registration.edit_subtypes == edit_subtypes
        assert registration.required_capabilities == capabilities


def test_edit_action_dual_anchor_is_typed_backward_compatible_and_not_metadata() -> (
    None
):
    legacy = EditAction(
        EditKind.SUBSTITUTION,
        5,
        "Cl",
        "N",
        0,
        BondTypeName.SINGLE,
        {"occurrence_atom_maps": (6,)},
    )
    assert legacy.source_anchor_index == 5
    assert legacy.remove_anchor_index is None
    assert legacy.metadata == {"occurrence_atom_maps": (6,)}

    dual = replace(legacy, source_anchor_index=9, remove_anchor_index=5)
    assert dual.source_anchor_index == 9
    assert dual.remove_anchor_index == 5
    assert dual.metadata == legacy.metadata
    with pytest.raises((TypeError, ValueError)):
        replace(legacy, remove_anchor_index=True)
    with pytest.raises(ValueError):
        replace(legacy, remove_anchor_index=-1)
    smuggled = replace(
        legacy,
        metadata={**legacy.metadata, "remove_anchor_index": 5},
    )
    assert smuggled.remove_anchor_index is None
    assert smuggled.metadata["remove_anchor_index"] == 5
    for kind, remove, add in (
        (EditKind.ADDITION, None, "N"),
        (EditKind.DELETION, "Cl", None),
    ):
        with pytest.raises(ValueError):
            EditAction(
                edit_kind=kind,
                source_anchor_index=5,
                remove_fragment_smiles=remove,
                add_fragment_smiles=add,
                fragment_attachment_atom=0 if add is not None else None,
                bond_type=BondTypeName.SINGLE,
                metadata=({} if remove is None else {"occurrence_atom_maps": (6,)}),
                remove_anchor_index=7,
            )


def test_dual_anchor_participates_in_canonical_action_identity() -> None:
    registration = _registry().registration(SUBSTITUTION_OPERATOR_IDS[1])
    context = _context(
        registration,
        target_node_id="product",
        joined=next(
            item
            for item in _substitution_records()
            if item.anonymous_sample_id == "mol_edit.substitute_v2.0271"
        ),
    )
    reference = context.reference_graph.value_for("product")
    action = _reference_actions(context, dual_anchor=True)[0]
    patch = CandidatePatch(
        candidate_id="dual-anchor",
        root_node_id="product",
        old_value=reference,
        new_value=replace(
            reference,
            raw_value="C",
            normalized_value="C",
            provenance=ValueProvenance.RULE,
            oracle_match=False,
        ),
        edit_action=action,
        source=CandidateSourceType.RULE,
    )
    dual = CandidateProposal(
        proposal_id="dual",
        patch=patch,
        candidate_product_smiles="C",
    )
    collapsed = CandidateProposal(
        proposal_id="collapsed",
        patch=replace(
            patch,
            edit_action=replace(action, remove_anchor_index=action.source_anchor_index),
        ),
        candidate_product_smiles="C",
    )
    assert canonical_candidate_key(dual) != canonical_candidate_key(collapsed)
    implicit_same_anchor = replace(
        collapsed,
        proposal_id="implicit-same-anchor",
        patch=replace(
            collapsed.patch,
            edit_action=replace(
                collapsed.patch.edit_action,
                remove_anchor_index=None,
            ),
        ),
    )
    assert canonical_candidate_key(collapsed) == canonical_candidate_key(
        implicit_same_anchor
    )


def test_candidate_boundary_rejects_anchor_metadata_smuggling() -> None:
    joined = next(
        item
        for item in _substitution_records()
        if item.anonymous_sample_id == "mol_edit.substitute_v2.0271"
    )
    context = _context(
        _registry().registration(SUBSTITUTION_OPERATOR_IDS[1]),
        target_node_id="product",
        joined=joined,
    )
    action = _reference_replay_matches(context, dual_anchor=True)[0]
    smuggled = replace(
        action,
        remove_anchor_index=None,
        metadata={**action.metadata, "remove_anchor_index": action.remove_anchor_index},
    )
    reference = context.reference_graph.value_for("product")
    proposal = CandidateProposal(
        proposal_id="smuggled-remove-anchor",
        patch=CandidatePatch(
            candidate_id="smuggled-remove-anchor",
            root_node_id="product",
            old_value=reference,
            new_value=replace(
                reference,
                raw_value="C",
                normalized_value="C",
                provenance=ValueProvenance.RULE,
                oracle_match=False,
            ),
            edit_action=smuggled,
            source=CandidateSourceType.RULE,
        ),
        candidate_product_smiles=context.truth.gt_smiles,
        difficulty_features=CandidateDifficultyFeatures(),
    )
    request = _candidate_request(context)
    result = DeterministicCandidateEngine(
        (RuleCandidateSource(lambda _: (proposal,)),)
    ).build_pool(request)
    assert result.pool.candidates == ()
    assert tuple(item.code for item in result.rejections) == (
        CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
    )


def test_fifty_origin_reference_replay_baseline_and_typed_boundary_cases() -> None:
    assert len(_substitution_records()) == 50
    registration = _registry().registration(SUBSTITUTION_OPERATOR_IDS[1])
    legacy_successes: set[str] = set()
    dual_successes: set[str] = set()
    for joined in _substitution_records():
        _, _, truth, record = _origin_artifacts(joined.anonymous_sample_id)
        assert record.operation_subtype is OperationSubtype.STANDARD
        assert truth.remove_fragment is not None and truth.add_fragment is not None
        assert len(truth.broken_bonds) == len(truth.formed_bonds) == 1
        assert len(truth.remove_fragment.component_smiles) == 1
        assert len(truth.add_fragment.component_smiles) == 1
        assert len(truth.remove_fragment.attachment_atoms) == 1
        assert len(truth.add_fragment.attachment_atoms) == 1
        context = _context(registration, target_node_id="product", joined=joined)
        if _reference_replay_matches(context, dual_anchor=False):
            legacy_successes.add(joined.anonymous_sample_id)
        if _reference_replay_matches(context, dual_anchor=True):
            dual_successes.add(joined.anonymous_sample_id)

    assert len(legacy_successes) == 49
    assert "mol_edit.substitute_v2.0271" not in legacy_successes
    assert "mol_edit.substitute_v2.0271" in dual_successes
    assert dual_successes == legacy_successes | {"mol_edit.substitute_v2.0271"}
    assert len(dual_successes) == 50


def test_0271_replay_requires_distinct_typed_remove_and_add_anchors() -> None:
    joined = next(
        item
        for item in _substitution_records()
        if item.anonymous_sample_id == "mol_edit.substitute_v2.0271"
    )
    context = _context(
        _registry().registration(SUBSTITUTION_OPERATOR_IDS[1]),
        target_node_id="product",
        joined=joined,
    )
    matches = _reference_replay_matches(context, dual_anchor=True)
    assert matches
    action = matches[0]
    assert action.remove_anchor_index == 20
    assert action.source_anchor_index == 35
    assert _occurrence_maps(action) == tuple(range(21, 29))
    products = replay_edit_action(_candidate_request(context), action)
    assert any(
        isomeric_graph_equivalent(product, context.truth.gt_smiles)
        for product in products
    )
    with pytest.raises(
        ValueError, match=CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
    ):
        replay_edit_action(
            _candidate_request(context),
            replace(action, remove_anchor_index=None),
        )
    smuggled = replace(
        action,
        remove_anchor_index=None,
        metadata={**action.metadata, "remove_anchor_index": 20},
    )
    with pytest.raises(
        ValueError, match=CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
    ):
        replay_edit_action(_candidate_request(context), smuggled)


def test_registered_boundary_cases_do_not_authorize_permissive_replay() -> None:
    expected_reference_replay = {
        "mol_edit.substitute_v2.0064": True,
        "mol_edit.substitute_v2.0123": True,
        "mol_edit.substitute_v2.0191": True,
        "mol_edit.substitute_v2.0216": True,
        "mol_edit.substitute_v2.0271": True,
        "mol_edit.substitute_v2.0276": True,
    }
    registration = _registry().registration(SUBSTITUTION_OPERATOR_IDS[1])
    for anonymous_sample_id, expected in expected_reference_replay.items():
        joined = next(
            item
            for item in _substitution_records()
            if item.anonymous_sample_id == anonymous_sample_id
        )
        context = _context(registration, target_node_id="product", joined=joined)
        matches = _reference_replay_matches(context, dual_anchor=True)
        assert bool(matches) is expected


@pytest.mark.parametrize(
    "anonymous_sample_id",
    (
        "mol_edit.substitute_v2.0191",
        "mol_edit.substitute_v2.0276",
    ),
)
def test_registered_charge_and_stereo_contracts_are_exact_and_fail_closed(
    anonymous_sample_id: str,
) -> None:
    joined = next(
        item
        for item in _substitution_records()
        if item.anonymous_sample_id == anonymous_sample_id
    )
    context = _context(
        _registry().registration(SUBSTITUTION_OPERATOR_IDS[2]),
        target_node_id="add_fragment",
        policy=PropagationPolicy.FULL_CF,
        source=CandidateSourceType.RDKIT,
        joined=joined,
    )
    request = _candidate_request(context)
    parameters = substitution_module._reference_parameters(request)
    assert len(parameters) == 1
    add_anchor, attachment, bond_type = parameters[0]
    action = substitution_module._action(
        request,
        add_anchor=add_anchor,
        occurrence=substitution_module._truth_occurrence(request),
        add_fragment=context.truth.add_fragment.canonical_smiles,
        attachment_atom=attachment,
        bond_type=bond_type,
    )
    contract_key = substitution_module._REGISTERED_REPLAY_CONTRACT_KEY
    assert dict(action.metadata[contract_key]) == dict(
        substitution_module._REGISTERED_REPLAY_CONTRACTS[anonymous_sample_id]
    )
    reference_product = replay_edit_action(request, action)[0]
    assert isomeric_graph_equivalent(reference_product, context.truth.gt_smiles)

    stripped = replace(
        action,
        metadata={"occurrence_atom_maps": _occurrence_maps(action)},
    )
    try:
        stripped_product = replay_edit_action(request, stripped)[0]
    except ValueError as error:
        assert str(error) == CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
    else:
        assert not isomeric_graph_equivalent(
            stripped_product,
            context.truth.gt_smiles,
        )

    tampered_contract = dict(action.metadata[contract_key])
    tampered_contract["expected_degree"] += 1
    tampered = replace(
        action,
        metadata={
            "occurrence_atom_maps": _occurrence_maps(action),
            contract_key: tampered_contract,
        },
    )
    with pytest.raises(
        ValueError,
        match=CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value,
    ):
        replay_edit_action(request, tampered)

    unknown_id = "mol_edit.substitute_v2.unregistered_same_shape"
    unknown_context = replace(
        context,
        record=replace(context.record, origin_id=unknown_id),
        recipe=replace(context.recipe, origin_id=unknown_id),
        truth=replace(context.truth, anonymous_sample_id=unknown_id),
    )
    unknown_classification = replace(
        request.resolution.classification,
        anonymous_sample_id=unknown_id,
        registered=False,
        provenance=(),
    )
    unknown_request = CandidateRequest(
        context=unknown_context,
        resolution=replace(
            request.resolution,
            classification=unknown_classification,
        ),
    )
    with pytest.raises(
        ValueError,
        match=CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value,
    ):
        replay_edit_action(unknown_request, action)


@pytest.mark.parametrize(
    "anonymous_sample_id",
    (
        "mol_edit.substitute_v2.0191",
        "mol_edit.substitute_v2.0276",
    ),
)
@pytest.mark.parametrize(
    "policy",
    (PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF),
)
def test_t048_registered_boundary_structural_pools_are_selectable(
    anonymous_sample_id: str,
    policy: PropagationPolicy,
) -> None:
    joined = next(
        item
        for item in _substitution_records()
        if item.anonymous_sample_id == anonymous_sample_id
    )
    context, pool = _invoke(
        SUBSTITUTION_OPERATOR_IDS[2],
        target_node_id="add_fragment",
        policy=policy,
        source=CandidateSourceType.RDKIT,
        joined=joined,
    )
    assert len(pool.candidates) == 10
    assert "INSUFFICIENT_CANDIDATES" not in pool.rejection_codes
    for patch in pool.candidates:
        assert patch.edit_action is not None
        assert (
            substitution_module._REGISTERED_REPLAY_CONTRACT_KEY
            in patch.edit_action.metadata
        )
        _assert_replayable_patch(context, patch)


def test_substitution_engine_implements_t016_port_and_registry_dispatch() -> None:
    engine = SubstitutionCandidateEngine(operators_config=OPERATORS_CONFIG)
    assert isinstance(engine, CandidateEngine)
    perturbator = SubstitutionPerturbator(**_ports(engine))
    registration = _registry().registration(SUBSTITUTION_OPERATOR_IDS[0])
    context = _context(registration)

    direct = engine.enumerate_root_patches(context)
    dispatched = SubstitutionCandidateDispatcher(OPERATORS_CONFIG).invoke(
        perturbator, context
    )

    assert direct == dispatched
    assert direct.request_id == context.recipe.recipe_id
    for patch in direct.candidates:
        assert patch.root_node_id == context.recipe.target_node_id
        assert patch.old_value == context.reference_graph.value_for(
            context.recipe.target_node_id
        )


def test_llm_is_rejected_before_any_substitution_member_invocation() -> None:
    registry = _registry()
    perturbator = _production_perturbator()
    for registration in registry.registrations_for(
        task_family="mol_edit", subtask="substitute"
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
def test_each_operator_is_root_only_deterministic_and_source_audited(
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
        <= OPERATORS_CONFIG.candidate_generation.candidates_per_recipe_max
    )
    for patch in pool.candidates:
        assert patch.root_node_id == target
        assert patch.old_value == context.reference_graph.value_for(target)
        assert not patch.old_value.semantically_equals(patch.new_value)
        assert patch.source is CandidateSourceType.RULE
        assert patch.new_value.provenance is ValueProvenance.RULE
        assert patch.metadata["generator"] == "substitution_t021"
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


@pytest.mark.parametrize(
    ("operator_id", "target", "policy"),
    OPERATOR_RUNTIME_CASES[:5] + OPERATOR_RUNTIME_CASES[6:8],
)
def test_all_accepted_structural_candidates_are_exact_non_gt_replays(
    operator_id: str,
    target: str,
    policy: PropagationPolicy,
) -> None:
    saw_candidate = False
    for joined in _substitution_records():
        context, pool = _invoke(
            operator_id,
            target_node_id=target,
            policy=policy,
            source=CandidateSourceType.RDKIT,
            joined=joined,
        )
        for patch in pool.candidates:
            saw_candidate = True
            _assert_replayable_patch(context, patch)
    assert saw_candidate


def test_wrong_occurrence_is_distinct_from_true_remove_and_has_exact_boundary() -> None:
    _, context, pool = _first_nonempty(
        SUBSTITUTION_OPERATOR_IDS[1],
        target_node_id="product",
        source=CandidateSourceType.RDKIT,
    )
    truth_removed = frozenset(context.truth.removed_atom_maps)
    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        occurrence = frozenset(_occurrence_maps(action))
        assert occurrence != truth_removed
        _assert_replayable_patch(context, patch)


def test_wrong_regio_uses_true_removal_but_a_distinct_add_site() -> None:
    _, context, pool = _first_nonempty(
        SUBSTITUTION_OPERATOR_IDS[7],
        target_node_id="product",
        policy=PropagationPolicy.FULL_CF,
        source=CandidateSourceType.RDKIT,
    )
    truth_removed = tuple(sorted(context.truth.removed_atom_maps))
    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        assert _occurrence_maps(action) == truth_removed
        _, _, boundaries = _source_boundary(
            context.record.indexed_smiles,
            truth_removed,
        )
        assert len(boundaries) == 1
        assert action.remove_anchor_index == boundaries[0][1]
        assert action.source_anchor_index != action.remove_anchor_index
        assert action.source_anchor_index not in context.truth.valid_anchor_indices
        _assert_replayable_patch(context, patch)


def test_source_anchor_and_fragment_attachment_are_independent_coordinates() -> None:
    _, context, pool = _first_nonempty(
        SUBSTITUTION_OPERATOR_IDS[3],
        target_node_id="product",
        source=CandidateSourceType.RDKIT,
    )
    patch = pool.candidates[0]
    action = patch.edit_action
    assert action is not None
    product = _assert_replayable_patch(context, patch)
    request = _candidate_request(context)

    fragment = Chem.MolFromSmiles(action.add_fragment_smiles, sanitize=True)
    assert fragment is not None
    assert 0 <= action.fragment_attachment_atom < fragment.GetNumAtoms()
    assert action.source_anchor_index > 0

    with pytest.raises(
        ValueError, match=CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
    ):
        replay_edit_action(
            request,
            replace(action, source_anchor_index=action.fragment_attachment_atom),
        )
    alternate_attachment = next(
        (
            index
            for index in range(fragment.GetNumAtoms())
            if index != action.fragment_attachment_atom
        ),
        fragment.GetNumAtoms(),
    )
    try:
        tampered_products = replay_edit_action(
            request,
            replace(action, fragment_attachment_atom=alternate_attachment),
        )
    except ValueError:
        tampered_products = ()
    assert not any(
        isomeric_graph_equivalent(candidate, product) for candidate in tampered_products
    )


@pytest.mark.parametrize(
    ("operator_id", "target", "wrong_value"),
    (
        (SUBSTITUTION_OPERATOR_IDS[0], "anchor_idx", 999999),
        (SUBSTITUTION_OPERATOR_IDS[2], "add_fragment", "[Xe]"),
    ),
)
def test_candidate_gate_rejects_root_action_cross_field_mismatch(
    operator_id: str,
    target: str,
    wrong_value: object,
) -> None:
    _, context, pool = _first_nonempty(
        operator_id,
        target_node_id=target,
        source=CandidateSourceType.RULE,
    )
    patch = next(item for item in pool.candidates if item.edit_action is not None)
    action = patch.edit_action
    assert action is not None
    product = replay_edit_action(_candidate_request(context), action)[0]
    tampered = replace(
        patch,
        candidate_id=f"{patch.candidate_id}:cross-field-tamper",
        new_value=replace(
            patch.new_value,
            raw_value=wrong_value,
            normalized_value=wrong_value,
        ),
    )
    proposal = CandidateProposal(
        proposal_id="cross-field-tamper",
        patch=tampered,
        candidate_product_smiles=product,
    )
    result = DeterministicCandidateEngine(
        (RuleCandidateSource(lambda _: (proposal,)),)
    ).build_pool(_candidate_request(context))
    assert result.pool.candidates == ()
    assert tuple(item.code for item in result.rejections) == (
        CandidateRejectCode.ACTION_PRODUCT_MISMATCH,
    )


def test_partial_substitution_is_a_connected_induced_strict_incoming_subfragment() -> (
    None
):
    _, context, pool = _first_nonempty(
        SUBSTITUTION_OPERATOR_IDS[6],
        target_node_id="product",
        source=CandidateSourceType.RDKIT,
    )
    truth_fragment = context.truth.add_fragment
    assert truth_fragment is not None
    truth_molecule = Chem.MolFromSmiles(truth_fragment.canonical_smiles, sanitize=True)
    assert truth_molecule is not None

    for patch in pool.candidates:
        action = patch.edit_action
        assert action is not None
        fragment = Chem.MolFromSmiles(action.add_fragment_smiles, sanitize=True)
        assert fragment is not None
        assert len(Chem.GetMolFrags(fragment)) == 1
        assert 0 < fragment.GetNumHeavyAtoms() < truth_molecule.GetNumHeavyAtoms()
        matches = truth_molecule.GetSubstructMatches(
            fragment,
            uniquify=False,
            useChirality=True,
            maxMatches=10000,
        )
        assert matches
        assert any(
            sum(
                bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match
                for bond in truth_molecule.GetBonds()
            )
            == fragment.GetNumBonds()
            for match in matches
        )
        _assert_replayable_patch(context, patch)


def test_largest_incoming_fragment_partial_enumeration_is_structurally_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joined = next(
        record
        for record in _substitution_records()
        if record.anonymous_sample_id == "mol_edit.substitute_v2.0276"
    )
    context = _context(
        _registry().registration(SUBSTITUTION_OPERATOR_IDS[6]),
        target_node_id="product",
        joined=joined,
    )
    fragment = context.truth.add_fragment
    assert fragment is not None
    molecule = Chem.MolFromSmiles(fragment.canonical_smiles, sanitize=True)
    assert molecule is not None
    assert molecule.GetNumHeavyAtoms() == 16

    calls = 0
    original = substitution_module._induced_fragment_smiles

    def counted_induced_fragment_smiles(
        candidate_molecule: Chem.Mol,
        atom_indices: frozenset[int],
    ) -> str | None:
        nonlocal calls
        calls += 1
        return original(candidate_molecule, atom_indices)

    monkeypatch.setattr(
        substitution_module,
        "_induced_fragment_smiles",
        counted_induced_fragment_smiles,
    )
    fragments = substitution_module._partial_incoming_fragments(
        _candidate_request(context)
    )
    assert fragments
    assert len(fragments) <= 64
    assert calls <= 8 * molecule.GetNumHeavyAtoms()


@pytest.mark.parametrize("root", ("remove_group", "add_fragment"))
def test_add_remove_role_claim_changes_one_role_without_graph_action(root: str) -> None:
    _, context, pool = _first_nonempty(
        SUBSTITUTION_OPERATOR_IDS[8],
        target_node_id=root,
    )
    other = "add_fragment" if root == "remove_group" else "remove_group"
    other_reference = context.reference_graph.value_for(other)
    for patch in pool.candidates:
        assert patch.root_node_id == root
        assert patch.edit_action is None
        assert patch.new_value.semantically_equals(other_reference)


def test_halogen_leaving_swaps_follow_a_static_balanced_cycle_and_are_actionless() -> (
    None
):
    halogens = frozenset({"F", "Cl", "Br", "I"})
    cycle = {"F": "Cl", "Cl": "Br", "Br": "I", "I": "F"}
    correct_by_id = {
        joined.anonymous_sample_id: joined.process_record["parsed_reference_state"][
            "step1_remove_group_smiles"
        ]
        for joined in _substitution_records()
    }
    halogen_origins = {
        anonymous_sample_id: value
        for anonymous_sample_id, value in correct_by_id.items()
        if value in halogens
    }
    assert {
        value: tuple(halogen_origins.values()).count(value)
        for value in sorted(halogens)
    } == {"Br": 9, "Cl": 15, "F": 8, "I": 2}

    selected: dict[str, str] = {}
    for joined in _substitution_records():
        if joined.anonymous_sample_id not in halogen_origins:
            continue
        _, pool = _invoke(
            SUBSTITUTION_OPERATOR_IDS[5],
            target_node_id="remove_group",
            joined=joined,
        )
        assert pool.candidates
        patch = pool.candidates[0]
        value = patch.new_value.normalized_value
        assert value == cycle[halogen_origins[joined.anonymous_sample_id]]
        assert patch.edit_action is None
        selected[joined.anonymous_sample_id] = value

    assert set(selected.values()) == halogens
    assert all(value in set(correct_by_id.values()) for value in selected.values())
    assert {
        value: tuple(selected.values()).count(value) for value in sorted(halogens)
    } == {"Br": 15, "Cl": 8, "F": 2, "I": 9}
    reversed_results = {}
    for joined in reversed(_substitution_records()):
        if joined.anonymous_sample_id in halogen_origins:
            _, pool = _invoke(
                SUBSTITUTION_OPERATOR_IDS[5],
                target_node_id="remove_group",
                joined=joined,
            )
            reversed_results[joined.anonymous_sample_id] = pool.candidates[
                0
            ].new_value.normalized_value
    assert reversed_results == selected


def test_count_candidates_use_substitution_arithmetic_not_fixed_offsets() -> None:
    heavy_differences: set[int] = set()
    ring_differences: set[int] = set()
    for joined in _substitution_records():
        for operator_id, target, accumulator in (
            (SUBSTITUTION_OPERATOR_IDS[9], "heavy_delta", heavy_differences),
            (SUBSTITUTION_OPERATOR_IDS[10], "ring_delta", ring_differences),
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
            assert all(patch.edit_action is None for patch in pool.candidates)

    assert len(heavy_differences) >= 4
    assert any(abs(delta) > 1 for delta in heavy_differences)
    assert len(ring_differences) >= 2
    assert any(abs(delta) > 1 for delta in ring_differences)


def test_all_origin_operator_pools_honor_cap_and_underfill_stops_selection() -> None:
    generation = OPERATORS_CONFIG.candidate_generation
    underfilled: tuple[PerturbationContext, CandidatePool] | None = None
    selectable: tuple[PerturbationContext, CandidatePool] | None = None
    for joined in _substitution_records():
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
                if pool.candidates[0].edit_action is not None and underfilled is None:
                    underfilled = (context, pool)
                    for patch in pool.candidates:
                        _assert_replayable_patch(context, patch)

    assert underfilled is not None and selectable is not None
    engine = SubstitutionCandidateEngine(operators_config=OPERATORS_CONFIG)
    _ = SubstitutionPerturbator(**_ports(engine))
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


@pytest.mark.parametrize(
    "anonymous_sample_id",
    (
        "mol_edit.substitute_v2.0064",
        "mol_edit.substitute_v2.0123",
        "mol_edit.substitute_v2.0191",
        "mol_edit.substitute_v2.0216",
        "mol_edit.substitute_v2.0271",
        "mol_edit.substitute_v2.0276",
    ),
)
def test_terminal_is_exact_valid_wrong_final_answer(
    anonymous_sample_id: str,
) -> None:
    joined = next(
        record
        for record in _substitution_records()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    context, pool = _invoke(
        SUBSTITUTION_OPERATOR_IDS[11],
        target_node_id="final_answer",
        policy=PropagationPolicy.TERMINAL,
        source=CandidateSourceType.RDKIT,
        joined=joined,
    )
    assert pool.candidates

    request = _candidate_request(context)
    replayed_wrong_products = set()
    for action in substitution_module._graph_actions(request):
        try:
            products = replay_edit_action(request, action)
        except (RuntimeError, TypeError, ValueError):
            continue
        replayed_wrong_products.update(
            product
            for product in products
            if not isomeric_graph_equivalent(product, context.truth.gt_smiles)
        )
    assert replayed_wrong_products
    for patch in pool.candidates:
        assert patch.root_node_id == "final_answer"
        assert patch.old_value == context.reference_graph.value_for("final_answer")
        assert patch.edit_action is None
        candidate = patch.new_value.normalized_value
        assert type(candidate) is str
        parsed = Chem.MolFromSmiles(candidate, sanitize=True)
        assert parsed is not None
        assert not isomeric_graph_equivalent(candidate, context.truth.gt_smiles)
        assert any(
            isomeric_graph_equivalent(candidate, product)
            for product in replayed_wrong_products
        )
        assert parsed.GetNumHeavyAtoms() >= (
            context.truth.source_descriptors.heavy_atom_count
            - len(context.truth.removed_atom_maps)
        )
