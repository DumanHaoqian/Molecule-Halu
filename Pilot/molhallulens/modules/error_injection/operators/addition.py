"""Deterministic, root-only Addition operators and candidate dispatch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from rdkit import Chem, rdBase

from molhallulens.infrastructure.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
    compute_descriptors,
    isomeric_graph_equivalent,
)
from molhallulens.config.models import OperatorsConfig
from molhallulens.core import (
    BondTypeName,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    ClaimValue,
    EditAction,
    EditErrorSubtype,
    EditKind,
    HallucinationType,
    OperatorCapability,
    PropagationPolicy,
    ValueProvenance,
)

from molhallulens.orchestration import CandidateEngine, PerturbationContext
from ..registry import OperatorRegistryError, PerturbatorRegistry, operator

if TYPE_CHECKING:
    from molhallulens.modules.error_planning import CandidateProposal, CandidateRequest

    from . import AdditionPerturbator


ADDITION_OPERATOR_IDS = (
    "mol_edit.add.alternate_anchor_same_element",
    "mol_edit.add.neighborhood_matched_anchor",
    "mol_edit.add.fragment_bucket_swap",
    "mol_edit.add.fragment_attachment_atom",
    "mol_edit.add.attachment_bond_order",
    "mol_edit.add.valid_wrong_site_product",
    "mol_edit.add.valid_regioisomer_product",
    "mol_edit.add.heavy_count_claim",
    "mol_edit.add.ring_count_claim",
    "mol_edit.add.internal_relation_claim",
    "mol_edit.add.terminal_answer",
)

_STRUCTURAL_POLICIES = frozenset(
    {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
)
_CLAIM_POLICIES = frozenset({PropagationPolicy.STOP, PropagationPolicy.PARTIAL})
_STRUCTURAL_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
_COUNT_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
_RELATION_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.HYBRID}
)
_CLAIM_CAPABILITY = frozenset({OperatorCapability.CLAIM_PERTURBATION})
_FRAGMENT_BUCKET = (
    "C",
    "CC",
    "CN",
    "CO",
    "C#N",
    "C(F)(F)F",
    "C1CC1",
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "c1ccc2cc3ccccc3cc2c1",
)
_BOND_ORDERS = (
    BondTypeName.SINGLE,
    BondTypeName.DOUBLE,
    BondTypeName.TRIPLE,
)


class AdditionOperatorMixin:
    """The eleven T019 members; each delegates chemistry to the injected engine."""

    def _addition_candidate_pool(self, context: PerturbationContext[Any]) -> CandidatePool:
        engine = self.candidate_engine  # type: ignore[attr-defined]
        if type(engine) is not AdditionCandidateEngine:
            raise OperatorRegistryError(
                code="ADDITION_CANDIDATE_ENGINE_REQUIRED",
                operator_id=context.recipe.operator_id,
                detail="Addition operators require the deterministic T019 engine",
            )
        return engine._pool_from_member(self, context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[0],
        operator_family="wrong_anchor_site",
        root_fields={"anchor_idx"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ANCHOR_GROUNDING},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_alternate_anchor_same_element(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[1],
        operator_family="wrong_anchor_site",
        root_fields={"anchor_idx"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ANCHOR_GROUNDING},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_neighborhood_matched_anchor(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[2],
        operator_family="wrong_fragment_group",
        root_fields={"add_fragment"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ADD_FRAGMENT_IDENTIFICATION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_fragment_bucket_swap(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[3],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_fragment_attachment_atom(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[4],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_attachment_bond_order(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[5],
        operator_family="wrong_anchor_site",
        root_fields={"product"},
        supported_policies={PropagationPolicy.FULL_CF},
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_valid_wrong_site_product(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[6],
        operator_family="wrong_anchor_site",
        root_fields={"product"},
        supported_policies={PropagationPolicy.FULL_CF},
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_valid_regioisomer_product(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[7],
        operator_family="numeric_count_claim",
        root_fields={"fragment_heavy", "source_heavy", "product_heavy", "heavy_delta"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_COUNT_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.HEAVY_ATOM_COUNT, EditErrorSubtype.HEAVY_ATOM_ARITHMETIC},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_heavy_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[8],
        operator_family="numeric_count_claim",
        root_fields={"source_rings", "product_rings", "ring_delta"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_COUNT_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_ring_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[9],
        operator_family="nl_formal_internal_relation",
        root_fields={"anchor_element"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_RELATION_SOURCES,
        hallucination_types={HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_internal_relation_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)

    @operator(
        operator_id=ADDITION_OPERATOR_IDS[10],
        operator_family="final_answer_identity",
        root_fields={"final_answer"},
        supported_policies={PropagationPolicy.TERMINAL},
        supported_sources=_STRUCTURAL_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
        required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
    )
    def perturb_terminal_answer(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._addition_candidate_pool(context)


@dataclass(frozen=True, slots=True)
class AdditionCandidateDispatcher:
    """Resolve and invoke one exact Addition member before the T018 boundary."""

    operators_config: OperatorsConfig

    def __post_init__(self) -> None:
        if type(self.operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")

    def _registry(self) -> PerturbatorRegistry:
        from . import AdditionPerturbator

        return PerturbatorRegistry.from_perturbator_types(
            (AdditionPerturbator,), operators_config=self.operators_config
        )

    def invoke(
        self,
        perturbator: AdditionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        return self._registry().invoke(perturbator, context)

    def build_member_pool(
        self,
        perturbator: AdditionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        from molhallulens.modules.error_planning import (
            CandidateRejectCode,
            CandidateRequest,
            DeterministicCandidateEngine,
            RDKitCandidateSource,
            RuleCandidateSource,
        )

        resolution = self._registry().resolve(perturbator, context)
        request = CandidateRequest(context=context, resolution=resolution)
        sources = tuple(
            source
            for source in (
                RuleCandidateSource(
                    partial(
                        _enumerate_addition_proposals,
                        source=CandidateSourceType.RULE,
                    )
                ),
                RDKitCandidateSource(
                    partial(
                        _enumerate_addition_proposals,
                        source=CandidateSourceType.RDKIT,
                    )
                ),
            )
            if source.source_type in resolution.registration.spec.supported_sources
        )
        result = DeterministicCandidateEngine(sources).build_pool(request)
        generation = self.operators_config.candidate_generation
        maximum = generation.candidates_per_recipe_max
        candidates = result.pool.candidates[:maximum]
        rejection_codes = set(result.pool.rejection_codes)
        if 0 < len(candidates) < generation.candidates_per_recipe_min:
            rejection_codes.add(CandidateRejectCode.INSUFFICIENT_CANDIDATES.value)
        return CandidatePool(
            request_id=result.pool.request_id,
            candidates=candidates,
            rejection_codes=tuple(sorted(rejection_codes)),
        )


class AdditionCandidateEngine(CandidateEngine[Any]):
    """CandidateEngine port whose production path is registry→member→T018."""

    __slots__ = ("_dispatcher", "_owner")

    def __init__(self, *, operators_config: OperatorsConfig) -> None:
        self._dispatcher = AdditionCandidateDispatcher(operators_config)
        self._owner: AdditionPerturbator | None = None

    @property
    def dispatcher(self) -> AdditionCandidateDispatcher:
        return self._dispatcher

    def bind_owner(self, perturbator: AdditionPerturbator) -> None:
        from . import AdditionPerturbator

        if type(perturbator) is not AdditionPerturbator:
            raise TypeError("AdditionCandidateEngine owner must be AdditionPerturbator")
        if self._owner is not None and self._owner is not perturbator:
            raise RuntimeError("AdditionCandidateEngine is already bound")
        self._owner = perturbator

    def _require_owner(self) -> AdditionPerturbator:
        if self._owner is None:
            raise RuntimeError("AdditionCandidateEngine is not bound")
        return self._owner

    def enumerate_root_patches(self, context: PerturbationContext[Any]) -> CandidatePool:
        return self._dispatcher.invoke(self._require_owner(), context)

    def select_root_patch(
        self,
        context: PerturbationContext[Any],
        pool: CandidatePool,
    ) -> CandidatePatch:
        if not isinstance(context, PerturbationContext):
            raise TypeError("context must be PerturbationContext")
        if type(pool) is not CandidatePool:
            raise TypeError("pool must be CandidatePool")
        if not pool.candidates:
            raise ValueError("cannot select from an empty Addition candidate pool")
        minimum = (
            self._dispatcher.operators_config.candidate_generation.candidates_per_recipe_min
        )
        if (
            len(pool.candidates) < minimum
            or "INSUFFICIENT_CANDIDATES" in pool.rejection_codes
        ):
            raise OperatorRegistryError(
                code="INSUFFICIENT_CANDIDATES",
                operator_id=context.recipe.operator_id,
                detail="candidate pool is below the configured selection minimum",
                evidence={
                    "actual": len(pool.candidates),
                    "minimum": minimum,
                },
            )
        return pool.candidates[0]

    def _pool_from_member(
        self,
        perturbator: AdditionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        if perturbator is not self._require_owner():
            raise RuntimeError("AdditionCandidateEngine owner mismatch")
        return self._dispatcher.build_member_pool(perturbator, context)


def _strict_molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError("strict RDKit parsing failed")
    return molecule


def _source_graph(
    request: CandidateRequest,
) -> tuple[Chem.Mol, dict[int, int], tuple[int, ...]]:
    molecule = _strict_molecule(request.context.record.indexed_smiles)
    mapped: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0 or atom_map in mapped:
            raise ValueError("Addition sources require unique positive atom maps")
        mapped[atom_map] = atom.GetIdx()
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


def _alternate_anchors(
    request: CandidateRequest,
    *,
    neighborhood_matched: bool,
) -> tuple[int, ...]:
    molecule, mapped, ranks = _source_graph(request)
    valid = tuple(
        anchor
        for anchor in request.context.truth.valid_anchor_indices
        if anchor in mapped
    )
    if not valid:
        return ()
    reference_atoms = tuple(molecule.GetAtomWithIdx(mapped[anchor]) for anchor in valid)
    reference_atomic_numbers = {atom.GetAtomicNum() for atom in reference_atoms}
    valid_ranks = {ranks[mapped[anchor]] for anchor in valid}

    def neighborhood_signature(atom: Chem.Atom) -> tuple[Any, ...]:
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
                                atom.GetIdx(), neighbor.GetIdx()
                            ).GetBondType()
                        ),
                    )
                    for neighbor in atom.GetNeighbors()
                )
            ),
        )

    reference_neighborhoods = {
        neighborhood_signature(atom) for atom in reference_atoms
    }
    alternatives = []
    for atom_map, atom_index in sorted(mapped.items()):
        atom = molecule.GetAtomWithIdx(atom_index)
        if (
            atom_map in valid
            or atom.GetAtomicNum() not in reference_atomic_numbers
            or ranks[atom_index] in valid_ranks
        ):
            continue
        if neighborhood_matched and neighborhood_signature(atom) not in reference_neighborhoods:
            continue
        alternatives.append(atom_map)
    return tuple(alternatives)


def _fragment_atom_indices(smiles: str) -> tuple[int, ...]:
    molecule = _strict_molecule(smiles)
    return tuple(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )


def _correct_fragment(request: CandidateRequest) -> str:
    fragment = request.context.truth.add_fragment
    if fragment is None:
        raise ValueError("Addition truth must contain an add fragment")
    return fragment.canonical_smiles


def _reference_bond_types(request: CandidateRequest) -> tuple[BondTypeName, ...]:
    values = tuple(
        sorted(
            {bond.bond_type for bond in request.context.truth.formed_bonds},
            key=lambda item: item.value,
        )
    )
    return values or (BondTypeName.SINGLE,)


def _action(
    *,
    anchor: int,
    fragment: str,
    attachment_atom: int,
    bond_type: BondTypeName,
) -> EditAction:
    return EditAction(
        edit_kind=EditKind.ADDITION,
        source_anchor_index=anchor,
        add_fragment_smiles=fragment,
        fragment_attachment_atom=attachment_atom,
        bond_type=bond_type,
    )


def _replayed_product(
    request: CandidateRequest,
    action: EditAction,
) -> str | None:
    from molhallulens.modules.error_planning import replay_edit_action

    try:
        products = replay_edit_action(request, action)
    except (RuntimeError, TypeError, ValueError):
        return None
    return products[0] if products else None


def _reference_parameters(
    request: CandidateRequest,
) -> tuple[tuple[int, int, BondTypeName], ...]:
    fragment = _correct_fragment(request)
    matches: list[tuple[int, int, BondTypeName]] = []
    fallback: list[tuple[int, int, BondTypeName]] = []
    for anchor in request.context.truth.valid_anchor_indices:
        for attachment_atom in _fragment_atom_indices(fragment):
            for bond_type in _reference_bond_types(request):
                parameters = (anchor, attachment_atom, bond_type)
                fallback.append(parameters)
                product = _replayed_product(
                    request,
                    _action(
                        anchor=anchor,
                        fragment=fragment,
                        attachment_atom=attachment_atom,
                        bond_type=bond_type,
                    ),
                )
                if product is not None and isomeric_graph_equivalent(
                    product, request.context.truth.canonical_gt_smiles
                ):
                    matches.append(parameters)
    return tuple(matches or fallback)


def _canonical_fragment_bucket(request: CandidateRequest) -> tuple[str, ...]:
    correct = _correct_fragment(request)
    fragments = {
        canonicalize_smiles(
            fragment,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        for fragment in _FRAGMENT_BUCKET
    }
    return tuple(
        sorted(
            fragment
            for fragment in fragments
            if not isomeric_graph_equivalent(fragment, correct)
        )
    )


def _graph_actions(
    request: CandidateRequest,
) -> Iterator[EditAction]:
    operator_id = request.operator_id
    correct_fragment = _correct_fragment(request)
    reference_parameters = _reference_parameters(request)
    reference_attachment_bonds = tuple(
        sorted({(attachment, bond) for _, attachment, bond in reference_parameters})
    )

    if operator_id in ADDITION_OPERATOR_IDS[:2]:
        alternatives = _alternate_anchors(
            request,
            neighborhood_matched=operator_id == ADDITION_OPERATOR_IDS[1],
        )
        for anchor in alternatives:
            for attachment_atom, bond_type in reference_attachment_bonds:
                yield _action(
                    anchor=anchor,
                    fragment=correct_fragment,
                    attachment_atom=attachment_atom,
                    bond_type=bond_type,
                )
        return

    if operator_id == ADDITION_OPERATOR_IDS[2]:
        for anchor in request.context.truth.valid_anchor_indices:
            for fragment in _canonical_fragment_bucket(request):
                for attachment_atom in _fragment_atom_indices(fragment):
                    for bond_type in _reference_bond_types(request):
                        yield _action(
                            anchor=anchor,
                            fragment=fragment,
                            attachment_atom=attachment_atom,
                            bond_type=bond_type,
                        )
        return

    if operator_id == ADDITION_OPERATOR_IDS[3]:
        for anchor in request.context.truth.valid_anchor_indices:
            for attachment_atom in _fragment_atom_indices(correct_fragment):
                for bond_type in _reference_bond_types(request):
                    yield _action(
                        anchor=anchor,
                        fragment=correct_fragment,
                        attachment_atom=attachment_atom,
                        bond_type=bond_type,
                    )
        return

    if operator_id == ADDITION_OPERATOR_IDS[4]:
        reference_bonds = set(_reference_bond_types(request))
        for anchor in request.context.truth.valid_anchor_indices:
            for attachment_atom in _fragment_atom_indices(correct_fragment):
                for bond_type in _BOND_ORDERS:
                    if bond_type not in reference_bonds:
                        yield _action(
                            anchor=anchor,
                            fragment=correct_fragment,
                            attachment_atom=attachment_atom,
                            bond_type=bond_type,
                        )
        return

    if operator_id in ADDITION_OPERATOR_IDS[5:7]:
        alternatives = _alternate_anchors(
            request,
            neighborhood_matched=operator_id == ADDITION_OPERATOR_IDS[5],
        )
        for anchor in alternatives:
            for attachment_atom, bond_type in reference_attachment_bonds:
                yield _action(
                    anchor=anchor,
                    fragment=correct_fragment,
                    attachment_atom=attachment_atom,
                    bond_type=bond_type,
                )
        return

    if operator_id == ADDITION_OPERATOR_IDS[10]:
        alternatives = _alternate_anchors(request, neighborhood_matched=False)
        for anchor in alternatives:
            for attachment_atom, bond_type in reference_attachment_bonds:
                yield _action(
                    anchor=anchor,
                    fragment=correct_fragment,
                    attachment_atom=attachment_atom,
                    bond_type=bond_type,
                )
        for anchor in request.context.truth.valid_anchor_indices:
            for fragment in _canonical_fragment_bucket(request):
                for attachment_atom in _fragment_atom_indices(fragment):
                    yield _action(
                        anchor=anchor,
                        fragment=fragment,
                        attachment_atom=attachment_atom,
                        bond_type=BondTypeName.SINGLE,
                    )


def _claim(
    old: ClaimValue,
    value: Any,
    source: CandidateSourceType,
) -> ClaimValue:
    provenance = {
        CandidateSourceType.RULE: ValueProvenance.RULE,
        CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
    }[source]
    return replace(
        old,
        raw_value=value,
        normalized_value=value,
        provenance=provenance,
        locally_valid=True,
        oracle_match=False,
        confidence=1.0,
    )


def _candidate_identity(
    request: CandidateRequest,
    source: CandidateSourceType,
    value: Any,
    action: EditAction | None,
) -> str:
    payload = {
        "operator_id": request.operator_id,
        "source": source.value,
        "target": request.resolution.target_node_id,
        "value": value,
        "action": None
        if action is None
        else {
            "anchor": action.source_anchor_index,
            "fragment": action.add_fragment_smiles,
            "attachment": action.fragment_attachment_atom,
            "bond": None if action.bond_type is None else action.bond_type.value,
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"add:{source.value.lower()}:{digest}"


def _proposal(
    request: CandidateRequest,
    source: CandidateSourceType,
    *,
    value: Any,
    action: EditAction | None = None,
    product: str | None = None,
) -> CandidateProposal:
    from molhallulens.modules.error_planning import (
        CandidateDifficultyFeatures,
        CandidateProposal,
    )

    old = request.context.reference_graph.value_for(request.resolution.target_node_id)
    candidate_id = _candidate_identity(request, source, value, action)
    return CandidateProposal(
        proposal_id=f"proposal:{candidate_id}",
        patch=CandidatePatch(
            candidate_id=candidate_id,
            root_node_id=request.resolution.target_node_id,
            old_value=old,
            new_value=_claim(old, value, source),
            edit_action=action,
            source=source,
            metadata={
                "generator": "addition_t019",
                "operator_id": request.operator_id,
            },
        ),
        candidate_product_smiles=product,
        difficulty_features=CandidateDifficultyFeatures(
            source_score=1.0 if source is CandidateSourceType.RDKIT else 0.75
        ),
    )


def _count_values(request: CandidateRequest) -> tuple[int, ...]:
    root = request.resolution.target_node_id
    truth = request.context.truth
    fragment_descriptors = tuple(
        compute_descriptors(
            fragment,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        for fragment in (*_FRAGMENT_BUCKET, _correct_fragment(request))
    )
    fragment_heavy = {item.heavy_atom_count for item in fragment_descriptors}
    fragment_rings = {item.ring_count for item in fragment_descriptors}
    if root == "fragment_heavy":
        values = fragment_heavy
    elif root == "source_heavy":
        values = fragment_heavy | {
            truth.product_descriptors.heavy_atom_count,
        }
    elif root == "product_heavy":
        values = {
            truth.source_descriptors.heavy_atom_count + count
            for count in fragment_heavy
        }
    elif root == "heavy_delta":
        values = fragment_heavy
    elif root == "source_rings":
        values = fragment_rings | {truth.product_descriptors.ring_count}
    elif root == "product_rings":
        values = {
            truth.source_descriptors.ring_count + count for count in fragment_rings
        }
    elif root == "ring_delta":
        values = fragment_rings
    else:
        raise ValueError("unsupported Addition count root")
    old = request.context.reference_graph.value_for(root).normalized_value
    return tuple(sorted(value for value in values if value != old and value >= 0))


def _relation_values(request: CandidateRequest) -> tuple[str, ...]:
    molecule, _, _ = _source_graph(request)
    observed = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    observed.update({"C", "N", "O", "S"})
    old = request.context.reference_graph.value_for("anchor_element").normalized_value
    return tuple(sorted(value for value in observed if value != old))


def _enumerate_addition_proposals(
    request: CandidateRequest,
    *,
    source: CandidateSourceType,
) -> Iterable[CandidateProposal]:
    """Enumerate operator-owned proposals; T018 remains the acceptance gate."""

    from molhallulens.modules.error_planning import CandidateRequest

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if source not in {CandidateSourceType.RULE, CandidateSourceType.RDKIT}:
        raise TypeError("Addition proposal source must be RULE or RDKIT")
    root = request.resolution.target_node_id
    operator_id = request.operator_id
    if operator_id in ADDITION_OPERATOR_IDS[7:9]:
        for value in _count_values(request):
            yield _proposal(request, source, value=value)
        return
    if operator_id == ADDITION_OPERATOR_IDS[9]:
        for value in _relation_values(request):
            yield _proposal(request, source, value=value)
        return

    for action in _graph_actions(request):
        product = _replayed_product(request, action)
        if product is None:
            continue
        if root == "anchor_idx":
            value = action.source_anchor_index
        elif root == "add_fragment":
            value = action.add_fragment_smiles
        elif root in {"product", "final_answer"}:
            value = product
        else:
            raise ValueError("unsupported structural Addition root")
        if value is None:
            continue
        try:
            yield _proposal(
                request,
                source,
                value=value,
                action=action,
                product=product,
            )
        except ValueError:
            continue


__all__ = [
    "ADDITION_OPERATOR_IDS",
    "AdditionCandidateDispatcher",
    "AdditionCandidateEngine",
    "AdditionOperatorMixin",
]
