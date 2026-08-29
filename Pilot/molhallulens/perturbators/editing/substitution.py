"""Deterministic, dual-anchor Substitution operators and candidate dispatch."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from rdkit import Chem, rdBase

from molhallulens.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
)
from molhallulens.config.models import OperatorsConfig
from molhallulens.domain import (
    AnomalyProvenance,
    AtomReferenceNamespace,
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

from ..base import CandidateEngine, PerturbationContext
from ..registry import OperatorRegistryError, PerturbatorRegistry, operator

if TYPE_CHECKING:
    from molhallulens.candidates import CandidateProposal, CandidateRequest

    from . import SubstitutionPerturbator


SUBSTITUTION_OPERATOR_IDS = (
    "mol_edit.substitute.alternate_substitution_site",
    "mol_edit.substitute.wrong_leaving_occurrence",
    "mol_edit.substitute.incoming_fragment_bucket_swap",
    "mol_edit.substitute.fragment_attachment_atom",
    "mol_edit.substitute.attachment_bond_order",
    "mol_edit.substitute.leaving_group_swap",
    "mol_edit.substitute.partial_substitution",
    "mol_edit.substitute.valid_wrong_regioisomer",
    "mol_edit.substitute.add_remove_role_claim",
    "mol_edit.substitute.heavy_count_claim",
    "mol_edit.substitute.ring_count_claim",
    "mol_edit.substitute.terminal_answer",
)

_STRUCTURAL_POLICIES = frozenset(
    {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
)
_CLAIM_POLICIES = frozenset({PropagationPolicy.STOP, PropagationPolicy.PARTIAL})
_DETERMINISTIC_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
_RELATION_SOURCES = frozenset({CandidateSourceType.RULE, CandidateSourceType.HYBRID})
_CLAIM_CAPABILITY = frozenset({OperatorCapability.CLAIM_PERTURBATION})
_COMMON_HALLUCINATIONS = frozenset(
    {HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR}
)
_BOND_ORDERS = (
    BondTypeName.SINGLE,
    BondTypeName.DOUBLE,
    BondTypeName.TRIPLE,
)
_INCOMING_FRAGMENT_BUCKET = (
    "C",
    "CC",
    "CN",
    "CO",
    "C#N",
    "Cl",
    "C(F)(F)F",
    "C1CC1",
    "NCCO",
    "N1CCNCC1",
)
_HALOGEN_LEAVING_GROUPS = ("F", "Cl", "Br", "I")

_REGISTERED_REPLAY_CONTRACT_KEY = "registered_substitution_replay_contract"
_REGISTERED_REPLAY_CONTRACTS = MappingProxyType(
    {
        "mol_edit.substitute_v2.0191": MappingProxyType(
            {
                "format_version": "substitution_registered_replay_v1",
                "origin_id": "mol_edit.substitute_v2.0191",
                "mode": "charged_boundary_occurrence",
                "expected_atom_map": 27,
                "expected_atomic_number": 8,
                "expected_formal_charge": -1,
                "expected_total_hydrogens": 0,
                "expected_degree": 1,
            }
        ),
        "mol_edit.substitute_v2.0276": MappingProxyType(
            {
                "format_version": "substitution_registered_replay_v1",
                "origin_id": "mol_edit.substitute_v2.0276",
                "mode": "anchor_stereo_assignment",
                "expected_atom_map": 2,
                "expected_atomic_number": 6,
                "expected_formal_charge": 0,
                "expected_total_hydrogens": 1,
                "expected_degree": 3,
                "expected_source_chiral_tag": "CHI_UNSPECIFIED",
                "product_chiral_tag": "CHI_TETRAHEDRAL_CCW",
            }
        ),
    }
)
_REGISTERED_REPLAY_PROVENANCE = MappingProxyType(
    {
        "mol_edit.substitute_v2.0191": (
            AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION
        ),
        "mol_edit.substitute_v2.0276": (
            AnomalyProvenance.SUBSTITUTION_ANCHOR_STEREO_ASSIGNMENT
        ),
    }
)
_REGISTERED_INCOMING_FRAGMENT_BUCKETS = MappingProxyType(
    {
        "mol_edit.substitute_v2.0276": (
            "O=[N+]([O-])c1cc2nc(F)[nH]c2cc1[N+](=O)[O-]",
            "O=[N+]([O-])c1cc2nc(Br)[nH]c2cc1[N+](=O)[O-]",
            "O=[N+]([O-])c1cc2nc(I)[nH]c2cc1[N+](=O)[O-]",
        ),
    }
)


class SubstitutionOperatorMixin:
    """The twelve T021 members; chemistry stays in the injected engine."""

    def _substitution_candidate_pool(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        engine = self.candidate_engine  # type: ignore[attr-defined]
        if type(engine) is not SubstitutionCandidateEngine:
            raise OperatorRegistryError(
                code="SUBSTITUTION_CANDIDATE_ENGINE_REQUIRED",
                operator_id=context.recipe.operator_id,
                detail="Substitution operators require the deterministic T021 engine",
            )
        return engine._pool_from_member(self, context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[0],
        operator_family="wrong_anchor_site",
        root_fields={"anchor_idx"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.ANCHOR_GROUNDING},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_alternate_substitution_site(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[1],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_wrong_leaving_occurrence(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[2],
        operator_family="wrong_fragment_group",
        root_fields={"add_fragment"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.ADD_FRAGMENT_IDENTIFICATION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_incoming_fragment_bucket_swap(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[3],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_fragment_attachment_atom(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[4],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_attachment_bond_order(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[5],
        operator_family="wrong_fragment_group",
        root_fields={"remove_group"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_RELATION_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_leaving_group_swap(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[6],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_partial_substitution(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[7],
        operator_family="wrong_anchor_site",
        root_fields={"product"},
        supported_policies={PropagationPolicy.FULL_CF},
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_valid_wrong_regioisomer(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[8],
        operator_family="nl_formal_internal_relation",
        root_fields={"remove_group", "add_fragment"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_RELATION_SOURCES,
        hallucination_types={HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_add_remove_role_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[9],
        operator_family="numeric_count_claim",
        root_fields={
            "remove_heavy",
            "add_heavy",
            "source_heavy",
            "product_heavy",
            "heavy_delta",
        },
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={
            EditErrorSubtype.HEAVY_ATOM_COUNT,
            EditErrorSubtype.HEAVY_ATOM_ARITHMETIC,
        },
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_heavy_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[10],
        operator_family="numeric_count_claim",
        root_fields={"source_rings", "product_rings", "ring_delta"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_ring_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)

    @operator(
        operator_id=SUBSTITUTION_OPERATOR_IDS[11],
        operator_family="final_answer_identity",
        root_fields={"final_answer"},
        supported_policies={PropagationPolicy.TERMINAL},
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types=_COMMON_HALLUCINATIONS,
        edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
        required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
    )
    def perturb_terminal_answer(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._substitution_candidate_pool(context)


@dataclass(frozen=True, slots=True)
class SubstitutionCandidateDispatcher:
    """Resolve and invoke one exact Substitution member before T018."""

    operators_config: OperatorsConfig

    def __post_init__(self) -> None:
        if type(self.operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")

    def _registry(self) -> PerturbatorRegistry:
        from . import SubstitutionPerturbator

        return PerturbatorRegistry.from_perturbator_types(
            (SubstitutionPerturbator,), operators_config=self.operators_config
        )

    def invoke(
        self,
        perturbator: SubstitutionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        return self._registry().invoke(perturbator, context)

    def build_member_pool(
        self,
        perturbator: SubstitutionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        from molhallulens.candidates import (
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
                        _enumerate_substitution_proposals,
                        source=CandidateSourceType.RULE,
                    )
                ),
                RDKitCandidateSource(
                    partial(
                        _enumerate_substitution_proposals,
                        source=CandidateSourceType.RDKIT,
                    )
                ),
            )
            if source.source_type in resolution.registration.spec.supported_sources
        )
        result = DeterministicCandidateEngine(sources).build_pool(request)
        generation = self.operators_config.candidate_generation
        candidates = result.pool.candidates[: generation.candidates_per_recipe_max]
        rejection_codes = set(result.pool.rejection_codes)
        if 0 < len(candidates) < generation.candidates_per_recipe_min:
            rejection_codes.add(CandidateRejectCode.INSUFFICIENT_CANDIDATES.value)
        return CandidatePool(
            request_id=result.pool.request_id,
            candidates=candidates,
            rejection_codes=tuple(sorted(rejection_codes)),
        )


class SubstitutionCandidateEngine(CandidateEngine[Any]):
    """CandidateEngine port whose production path is registry→member→T018."""

    __slots__ = ("_dispatcher", "_owner")

    def __init__(self, *, operators_config: OperatorsConfig) -> None:
        self._dispatcher = SubstitutionCandidateDispatcher(operators_config)
        self._owner: SubstitutionPerturbator | None = None

    @property
    def dispatcher(self) -> SubstitutionCandidateDispatcher:
        return self._dispatcher

    def bind_owner(self, perturbator: SubstitutionPerturbator) -> None:
        from . import SubstitutionPerturbator

        if type(perturbator) is not SubstitutionPerturbator:
            raise TypeError(
                "SubstitutionCandidateEngine owner must be SubstitutionPerturbator"
            )
        if self._owner is not None and self._owner is not perturbator:
            raise RuntimeError("SubstitutionCandidateEngine is already bound")
        self._owner = perturbator

    def _require_owner(self) -> SubstitutionPerturbator:
        if self._owner is None:
            raise RuntimeError("SubstitutionCandidateEngine is not bound")
        return self._owner

    def enumerate_root_patches(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
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
            raise ValueError("cannot select from an empty Substitution candidate pool")
        minimum = self._dispatcher.operators_config.candidate_generation.candidates_per_recipe_min
        if (
            len(pool.candidates) < minimum
            or "INSUFFICIENT_CANDIDATES" in pool.rejection_codes
        ):
            raise OperatorRegistryError(
                code="INSUFFICIENT_CANDIDATES",
                operator_id=context.recipe.operator_id,
                detail="candidate pool is below the configured selection minimum",
                evidence={"actual": len(pool.candidates), "minimum": minimum},
            )
        return pool.candidates[0]

    def _pool_from_member(
        self,
        perturbator: SubstitutionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        if perturbator is not self._require_owner():
            raise RuntimeError("SubstitutionCandidateEngine owner mismatch")
        return self._dispatcher.build_member_pool(perturbator, context)


@dataclass(frozen=True, slots=True)
class _RemovalOccurrence:
    atom_maps: tuple[int, ...]
    anchor_map: int
    fragment_smiles: str


_BOND_TYPE_NAMES = {
    Chem.BondType.SINGLE: BondTypeName.SINGLE,
    Chem.BondType.DOUBLE: BondTypeName.DOUBLE,
    Chem.BondType.TRIPLE: BondTypeName.TRIPLE,
    Chem.BondType.AROMATIC: BondTypeName.AROMATIC,
}


def _strict_molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError("strict RDKit parsing failed")
    return molecule


def _mapped_source(
    request: CandidateRequest,
) -> tuple[Chem.Mol, dict[int, int], tuple[int, ...]]:
    molecule = _strict_molecule(request.context.record.indexed_smiles)
    mapped: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0 or atom_map in mapped:
            raise ValueError("Substitution sources require unique positive atom maps")
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


def _truth_fragments(request: CandidateRequest) -> tuple[str, str]:
    truth = request.context.truth
    if truth.remove_fragment is None or truth.add_fragment is None:
        raise ValueError("Substitution truth requires remove and add fragments")
    return (
        truth.remove_fragment.canonical_smiles,
        truth.add_fragment.canonical_smiles,
    )


def _retained_source_endpoint(
    request: CandidateRequest,
    *,
    broken: bool,
) -> tuple[int, ...]:
    edits = (
        request.context.truth.broken_bonds
        if broken
        else request.context.truth.formed_bonds
    )
    removed = request.context.truth.removed_atom_maps
    endpoints = {
        endpoint.atom_id
        for edit in edits
        for endpoint in (edit.begin, edit.end)
        if endpoint.namespace is AtomReferenceNamespace.SOURCE_MAP
        and (not broken or endpoint.atom_id not in removed)
    }
    return tuple(sorted(endpoints))


def _reference_bond_types(request: CandidateRequest) -> tuple[BondTypeName, ...]:
    values = tuple(
        sorted(
            {bond.bond_type for bond in request.context.truth.formed_bonds},
            key=lambda value: value.value,
        )
    )
    return values or (BondTypeName.SINGLE,)


def _truth_occurrence(request: CandidateRequest) -> _RemovalOccurrence:
    remove_fragment, _ = _truth_fragments(request)
    anchors = _retained_source_endpoint(request, broken=True)
    if len(anchors) != 1:
        raise ValueError("Substitution truth must have one removal boundary anchor")
    return _RemovalOccurrence(
        atom_maps=tuple(sorted(request.context.truth.removed_atom_maps)),
        anchor_map=anchors[0],
        fragment_smiles=remove_fragment,
    )


def _fragment_atom_indices(smiles: str) -> tuple[int, ...]:
    molecule = _strict_molecule(smiles)
    return tuple(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )


def _registered_replay_contract(
    request: CandidateRequest,
    *,
    add_anchor: int,
    occurrence: _RemovalOccurrence,
) -> dict[str, Any] | None:
    """Return one exact-ID, provenance-gated replay contract when applicable."""

    origin_id = request.context.truth.anonymous_sample_id
    spec = _REGISTERED_REPLAY_CONTRACTS.get(origin_id)
    if spec is None:
        return None
    classification = request.resolution.classification
    expected_provenance = _REGISTERED_REPLAY_PROVENANCE[origin_id]
    if (
        classification.anonymous_sample_id != origin_id
        or not classification.registered
        or expected_provenance not in classification.provenance
    ):
        raise ValueError("registered substitution replay provenance is unavailable")
    truth_occurrence = tuple(sorted(request.context.truth.removed_atom_maps))
    if occurrence.atom_maps != truth_occurrence:
        return None
    expected_map = spec["expected_atom_map"]
    mode = spec["mode"]
    if (
        mode == "charged_boundary_occurrence"
        and expected_map not in occurrence.atom_maps
    ):
        return None
    if mode == "anchor_stereo_assignment" and add_anchor != expected_map:
        return None
    return dict(spec)


def _action(
    request: CandidateRequest,
    *,
    add_anchor: int,
    occurrence: _RemovalOccurrence,
    add_fragment: str,
    attachment_atom: int,
    bond_type: BondTypeName,
) -> EditAction:
    metadata: dict[str, Any] = {"occurrence_atom_maps": occurrence.atom_maps}
    replay_contract = _registered_replay_contract(
        request,
        add_anchor=add_anchor,
        occurrence=occurrence,
    )
    if replay_contract is not None:
        metadata[_REGISTERED_REPLAY_CONTRACT_KEY] = replay_contract
    return EditAction(
        edit_kind=EditKind.SUBSTITUTION,
        source_anchor_index=add_anchor,
        remove_fragment_smiles=occurrence.fragment_smiles,
        add_fragment_smiles=add_fragment,
        fragment_attachment_atom=attachment_atom,
        bond_type=bond_type,
        metadata=metadata,
        remove_anchor_index=(
            None if occurrence.anchor_map == add_anchor else occurrence.anchor_map
        ),
    )


def _replayed_product(
    request: CandidateRequest,
    action: EditAction,
) -> str | None:
    from molhallulens.candidates import replay_edit_action

    try:
        products = replay_edit_action(request, action)
    except (RuntimeError, TypeError, ValueError):
        return None
    return products[0] if products else None


def _reference_parameters(
    request: CandidateRequest,
) -> tuple[tuple[int, int, BondTypeName], ...]:
    """Recover the exact add-side parameters solely through strict replay."""

    occurrence = _truth_occurrence(request)
    _, add_fragment = _truth_fragments(request)
    add_anchors = _retained_source_endpoint(request, broken=False)
    candidates = tuple(
        (anchor, attachment, bond_type)
        for anchor in add_anchors
        for attachment in _fragment_atom_indices(add_fragment)
        for bond_type in _reference_bond_types(request)
    )
    matches = []
    for add_anchor, attachment, bond_type in candidates:
        action = _action(
            request,
            add_anchor=add_anchor,
            occurrence=occurrence,
            add_fragment=add_fragment,
            attachment_atom=attachment,
            bond_type=bond_type,
        )
        product = _replayed_product(request, action)
        if product is not None and isomeric_graph_equivalent(
            product,
            request.context.truth.canonical_gt_smiles,
        ):
            matches.append((add_anchor, attachment, bond_type))
    # 0191 and 0276 are intentionally incapable of exact replay under the
    # closed charge/stereo contract.  Do not infer missing reference parameters
    # or silently add a second error dimension; structural members fail closed.
    return tuple(matches)


def _alternate_add_anchors(request: CandidateRequest) -> tuple[int, ...]:
    molecule, mapped, ranks = _mapped_source(request)
    reference_anchors = _retained_source_endpoint(request, broken=False)
    if not reference_anchors:
        return ()
    reference_numbers = {
        molecule.GetAtomWithIdx(mapped[anchor]).GetAtomicNum()
        for anchor in reference_anchors
        if anchor in mapped
    }
    reference_ranks = {
        ranks[mapped[anchor]] for anchor in reference_anchors if anchor in mapped
    }
    forbidden = set(request.context.truth.valid_anchor_indices)
    forbidden.update(request.context.truth.removed_atom_maps)
    return tuple(
        atom_map
        for atom_map, atom_index in sorted(mapped.items())
        if atom_map not in forbidden
        and molecule.GetAtomWithIdx(atom_index).GetAtomicNum() in reference_numbers
        and ranks[atom_index] not in reference_ranks
    )


def _occurrences_for_fragment(
    request: CandidateRequest,
    fragment_smiles: str,
) -> tuple[_RemovalOccurrence, ...]:
    """Find induced, connected, exactly one-boundary occurrences in source."""

    molecule, _, _ = _mapped_source(request)
    query = _strict_molecule(fragment_smiles)
    if len(Chem.GetMolFrags(query)) != 1:
        return ()
    occurrences: dict[tuple[Any, ...], _RemovalOccurrence] = {}
    for match in molecule.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=10000,
    ):
        indices = frozenset(match)
        if len(indices) != query.GetNumAtoms():
            continue
        internal_bonds = sum(
            bond.GetBeginAtomIdx() in indices and bond.GetEndAtomIdx() in indices
            for bond in molecule.GetBonds()
        )
        if internal_bonds != query.GetNumBonds():
            continue
        boundary = tuple(
            (atom_index, bond.GetOtherAtomIdx(atom_index))
            for atom_index in indices
            for bond in molecule.GetAtomWithIdx(atom_index).GetBonds()
            if bond.GetOtherAtomIdx(atom_index) not in indices
        )
        if len(boundary) != 1:
            continue
        maps = tuple(
            sorted(molecule.GetAtomWithIdx(index).GetAtomMapNum() for index in indices)
        )
        anchor_map = molecule.GetAtomWithIdx(boundary[0][1]).GetAtomMapNum()
        if any(atom_map <= 0 for atom_map in maps) or anchor_map <= 0:
            continue
        occurrence = _RemovalOccurrence(
            atom_maps=maps,
            anchor_map=anchor_map,
            fragment_smiles=fragment_smiles,
        )
        occurrences[(maps, anchor_map)] = occurrence
    return tuple(occurrences[key] for key in sorted(occurrences))


def _alternate_occurrences(request: CandidateRequest) -> tuple[_RemovalOccurrence, ...]:
    truth = _truth_occurrence(request)
    return tuple(
        occurrence
        for occurrence in _occurrences_for_fragment(request, truth.fragment_smiles)
        if occurrence.atom_maps != truth.atom_maps
    )


def _canonical_incoming_bucket(request: CandidateRequest) -> tuple[str, ...]:
    _, reference = _truth_fragments(request)
    registered = _REGISTERED_INCOMING_FRAGMENT_BUCKETS.get(
        request.context.truth.anonymous_sample_id,
        (),
    )
    values = {
        canonicalize_smiles(
            fragment,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        for fragment in (*_INCOMING_FRAGMENT_BUCKET, *registered)
    }
    return tuple(
        sorted(
            fragment
            for fragment in values
            if not fragment_graph_equivalent(fragment, reference)
        )
    )


def _connected(atom_indices: frozenset[int], molecule: Chem.Mol) -> bool:
    if not atom_indices:
        return False
    visited: set[int] = set()
    frontier = deque([min(atom_indices)])
    while frontier:
        atom_index = frontier.popleft()
        if atom_index in visited:
            continue
        visited.add(atom_index)
        frontier.extend(
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
            if neighbor.GetIdx() in atom_indices and neighbor.GetIdx() not in visited
        )
    return visited == set(atom_indices)


def _induced_fragment_smiles(
    molecule: Chem.Mol,
    atom_indices: frozenset[int],
) -> str | None:
    if not _connected(atom_indices, molecule):
        return None
    try:
        with rdBase.BlockLogs():
            serialized = Chem.MolFragmentToSmiles(
                molecule,
                atomsToUse=sorted(atom_indices),
                canonical=True,
                isomericSmiles=True,
                kekuleSmiles=True,
            )
        if not serialized or "." in serialized:
            return None
        canonical = canonicalize_smiles(
            serialized,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    parsed = _strict_molecule(canonical)
    if len(Chem.GetMolFrags(parsed)) != 1 or parsed.GetNumAtoms() != len(atom_indices):
        return None
    return canonical


def _partial_incoming_fragments(request: CandidateRequest) -> tuple[str, ...]:
    _, correct = _truth_fragments(request)
    molecule = _strict_molecule(correct)
    heavy_indices = tuple(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )
    if len(heavy_indices) <= 1:
        return ()
    heavy_set = frozenset(heavy_indices)
    subsets: set[frozenset[int]] = {
        frozenset({atom_index}) for atom_index in heavy_indices
    }

    # Deterministic bridge cuts preserve both connected induced sides and avoid
    # exponential subset enumeration on the 16-heavy-atom 0276 fragment.
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in heavy_set or end not in heavy_set:
            continue
        visited: set[int] = set()
        frontier = deque([begin])
        while frontier:
            atom_index = frontier.popleft()
            if atom_index in visited:
                continue
            visited.add(atom_index)
            for neighbor_bond in molecule.GetAtomWithIdx(atom_index).GetBonds():
                if neighbor_bond.GetIdx() == bond.GetIdx():
                    continue
                neighbor = neighbor_bond.GetOtherAtomIdx(atom_index)
                if neighbor in heavy_set and neighbor not in visited:
                    frontier.append(neighbor)
        begin_side = frozenset(visited)
        if end not in begin_side:
            subsets.update((begin_side, heavy_set - begin_side))

    # One-shell graph balls and terminal trims cover local partial groups even
    # when the incoming fragment is cyclic and has no bridge.
    for atom_index in heavy_indices:
        neighborhood = frozenset(
            {
                atom_index,
                *(
                    neighbor.GetIdx()
                    for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
                    if neighbor.GetIdx() in heavy_set
                ),
            }
        )
        subsets.add(neighborhood)
        trimmed = heavy_set - {atom_index}
        if trimmed and _connected(trimmed, molecule):
            subsets.add(trimmed)

    fragments = set()
    for atom_indices in sorted(
        (subset for subset in subsets if 0 < len(subset) < len(heavy_set)),
        key=lambda subset: (len(subset), tuple(sorted(subset))),
    ):
        fragment = _induced_fragment_smiles(molecule, atom_indices)
        if fragment is not None and not fragment_graph_equivalent(fragment, correct):
            fragments.add(fragment)
        if len(fragments) >= 64:
            break
    return tuple(sorted(fragments))


def _graph_actions(request: CandidateRequest) -> Iterator[EditAction]:
    operator_id = request.operator_id
    truth_occurrence = _truth_occurrence(request)
    _, correct_add = _truth_fragments(request)
    reference_parameters = _reference_parameters(request)
    reference_anchors = tuple(sorted({item[0] for item in reference_parameters}))
    reference_attachment_bonds = tuple(
        sorted(
            {(item[1], item[2]) for item in reference_parameters},
            key=lambda item: (item[0], item[1].value),
        )
    )

    if operator_id in {SUBSTITUTION_OPERATOR_IDS[0], SUBSTITUTION_OPERATOR_IDS[7]}:
        for add_anchor in _alternate_add_anchors(request):
            for attachment, bond_type in reference_attachment_bonds:
                yield _action(
                    request,
                    add_anchor=add_anchor,
                    occurrence=truth_occurrence,
                    add_fragment=correct_add,
                    attachment_atom=attachment,
                    bond_type=bond_type,
                )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[1]:
        for occurrence in _alternate_occurrences(request):
            for add_anchor in reference_anchors:
                if add_anchor in occurrence.atom_maps:
                    continue
                for attachment, bond_type in reference_attachment_bonds:
                    yield _action(
                        request,
                        add_anchor=add_anchor,
                        occurrence=occurrence,
                        add_fragment=correct_add,
                        attachment_atom=attachment,
                        bond_type=bond_type,
                    )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[2]:
        for add_anchor in reference_anchors:
            for fragment in _canonical_incoming_bucket(request):
                for attachment in _fragment_atom_indices(fragment):
                    for bond_type in _reference_bond_types(request):
                        yield _action(
                            request,
                            add_anchor=add_anchor,
                            occurrence=truth_occurrence,
                            add_fragment=fragment,
                            attachment_atom=attachment,
                            bond_type=bond_type,
                        )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[3]:
        reference_attachments = {item[1] for item in reference_parameters}
        for add_anchor in reference_anchors:
            for attachment in _fragment_atom_indices(correct_add):
                if attachment in reference_attachments:
                    continue
                for bond_type in _reference_bond_types(request):
                    yield _action(
                        request,
                        add_anchor=add_anchor,
                        occurrence=truth_occurrence,
                        add_fragment=correct_add,
                        attachment_atom=attachment,
                        bond_type=bond_type,
                    )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[4]:
        reference_bonds = set(_reference_bond_types(request))
        for add_anchor, attachment, _ in reference_parameters:
            for bond_type in _BOND_ORDERS:
                if bond_type in reference_bonds:
                    continue
                yield _action(
                    request,
                    add_anchor=add_anchor,
                    occurrence=truth_occurrence,
                    add_fragment=correct_add,
                    attachment_atom=attachment,
                    bond_type=bond_type,
                )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[6]:
        for add_anchor in reference_anchors:
            for fragment in _partial_incoming_fragments(request):
                for attachment in _fragment_atom_indices(fragment):
                    for bond_type in _reference_bond_types(request):
                        yield _action(
                            request,
                            add_anchor=add_anchor,
                            occurrence=truth_occurrence,
                            add_fragment=fragment,
                            attachment_atom=attachment,
                            bond_type=bond_type,
                        )
        return

    if operator_id == SUBSTITUTION_OPERATOR_IDS[11]:
        for add_anchor in _alternate_add_anchors(request):
            for attachment, bond_type in reference_attachment_bonds:
                yield _action(
                    request,
                    add_anchor=add_anchor,
                    occurrence=truth_occurrence,
                    add_fragment=correct_add,
                    attachment_atom=attachment,
                    bond_type=bond_type,
                )
        for add_anchor in reference_anchors:
            for fragment in _canonical_incoming_bucket(request):
                for attachment in _fragment_atom_indices(fragment):
                    for bond_type in _reference_bond_types(request):
                        yield _action(
                            request,
                            add_anchor=add_anchor,
                            occurrence=truth_occurrence,
                            add_fragment=fragment,
                            attachment_atom=attachment,
                            bond_type=bond_type,
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
            "add_anchor": action.source_anchor_index,
            "remove_anchor": action.remove_anchor_index,
            "remove_fragment": action.remove_fragment_smiles,
            "add_fragment": action.add_fragment_smiles,
            "attachment": action.fragment_attachment_atom,
            "bond": None if action.bond_type is None else action.bond_type.value,
            "occurrence": action.metadata.get("occurrence_atom_maps"),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"substitute:{source.value.lower()}:{digest}"


def _proposal(
    request: CandidateRequest,
    source: CandidateSourceType,
    *,
    value: Any,
    action: EditAction | None = None,
    product: str | None = None,
) -> CandidateProposal:
    from molhallulens.candidates import CandidateDifficultyFeatures, CandidateProposal

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
                "generator": "substitution_t021",
                "operator_id": request.operator_id,
            },
        ),
        candidate_product_smiles=product,
        difficulty_features=CandidateDifficultyFeatures(
            source_score=1.0 if source is CandidateSourceType.RDKIT else 0.75
        ),
    )


def _leaving_group_values(request: CandidateRequest) -> tuple[str, ...]:
    old = request.context.reference_graph.value_for("remove_group").normalized_value
    cycle = {"F": "Cl", "Cl": "Br", "Br": "I", "I": "F"}
    replacement = cycle.get(old)
    return () if replacement is None else (replacement,)


def _role_values(request: CandidateRequest) -> tuple[str, ...]:
    root = request.resolution.target_node_id
    other_root = "add_fragment" if root == "remove_group" else "remove_group"
    old = request.context.reference_graph.value_for(root).normalized_value
    other = request.context.reference_graph.value_for(other_root).normalized_value
    if type(other) is not str or not other or other == old:
        return ()
    return (other,)


def _count_values(request: CandidateRequest) -> tuple[int, ...]:
    """Use graph-derived descriptor alternatives, never fixed +/-1 corruption."""

    root = request.resolution.target_node_id
    truth = request.context.truth
    remove_fragment, add_fragment = _truth_fragments(request)
    fragment_descriptors = tuple(
        compute_descriptors(fragment, fragment_policy=FragmentPolicy.KEEP_ALL)
        for fragment in (
            *_HALOGEN_LEAVING_GROUPS,
            *_INCOMING_FRAGMENT_BUCKET,
            remove_fragment,
            add_fragment,
        )
    )
    heavy = {item.heavy_atom_count for item in fragment_descriptors}
    rings = {item.ring_count for item in fragment_descriptors}
    if root in {"remove_heavy", "add_heavy"}:
        values = heavy
    elif root == "source_heavy":
        values = heavy | {
            truth.product_descriptors.heavy_atom_count,
            truth.source_descriptors.heavy_atom_count,
        }
    elif root == "product_heavy":
        source_heavy = truth.source_descriptors.heavy_atom_count
        values = {
            source_heavy - removed + added
            for removed in heavy
            for added in heavy
            if source_heavy - removed + added >= 0
        }
    elif root == "heavy_delta":
        values = {added - removed for removed in heavy for added in heavy}
    elif root == "source_rings":
        values = rings | {
            truth.source_descriptors.ring_count,
            truth.product_descriptors.ring_count,
        }
    elif root == "product_rings":
        source_rings = truth.source_descriptors.ring_count
        values = {
            source_rings - removed + added
            for removed in rings
            for added in rings
            if source_rings - removed + added >= 0
        }
    elif root == "ring_delta":
        values = {added - removed for removed in rings for added in rings}
    else:
        raise ValueError("unsupported Substitution count root")
    old = request.context.reference_graph.value_for(root).normalized_value
    return tuple(
        sorted(
            value
            for value in values
            if value != old and (root in {"heavy_delta", "ring_delta"} or value >= 0)
        )
    )


def _enumerate_substitution_proposals(
    request: CandidateRequest,
    *,
    source: CandidateSourceType,
) -> Iterable[CandidateProposal]:
    """Enumerate operator-owned proposals; T018 remains the acceptance gate."""

    from molhallulens.candidates import CandidateRequest

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if source not in {CandidateSourceType.RULE, CandidateSourceType.RDKIT}:
        raise TypeError("Substitution proposal source must be RULE or RDKIT")
    root = request.resolution.target_node_id
    operator_id = request.operator_id

    if operator_id == SUBSTITUTION_OPERATOR_IDS[5]:
        for value in _leaving_group_values(request):
            yield _proposal(request, source, value=value)
        return
    if operator_id == SUBSTITUTION_OPERATOR_IDS[8]:
        for value in _role_values(request):
            yield _proposal(request, source, value=value)
        return
    if operator_id in SUBSTITUTION_OPERATOR_IDS[9:11]:
        for value in _count_values(request):
            yield _proposal(request, source, value=value)
        return
    if operator_id == SUBSTITUTION_OPERATOR_IDS[11]:
        for action in _graph_actions(request):
            product = _replayed_product(request, action)
            if product is None or isomeric_graph_equivalent(
                product,
                request.context.truth.canonical_gt_smiles,
            ):
                continue
            yield _proposal(
                request,
                source,
                value=product,
                product=product,
            )
        return

    for action in _graph_actions(request):
        product = _replayed_product(request, action)
        if product is None or isomeric_graph_equivalent(
            product,
            request.context.truth.canonical_gt_smiles,
        ):
            continue
        if root == "anchor_idx":
            value = action.source_anchor_index
        elif root == "add_fragment":
            value = action.add_fragment_smiles
        elif root in {"product", "final_answer"}:
            value = product
        else:
            raise ValueError("unsupported structural Substitution root")
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


def t048_substitution_boundary_cases(origin_id: str) -> tuple[Any, ...]:
    """Return the frozen four-policy recipe for one registered T048 boundary."""

    if type(origin_id) is not str or origin_id not in _REGISTERED_REPLAY_CONTRACTS:
        raise ValueError("origin_id must name a registered T048 substitution boundary")
    from molhallulens.builders.golden_bundles import (
        GoldenOriginSpec,
        GoldenPolicySpec,
    )
    from molhallulens.builders.golden_validation import ExtendedGoldenOriginCase
    from molhallulens.domain import EditingSubtask

    policies = (
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
            SUBSTITUTION_OPERATOR_IDS[2],
            "add_fragment",
            "valid_wrong_group_fragment",
        ),
        GoldenPolicySpec(
            PropagationPolicy.TERMINAL,
            SUBSTITUTION_OPERATOR_IDS[11],
            "final_answer",
            "terminal_valid_high_similarity",
        ),
    )
    mode = _REGISTERED_REPLAY_CONTRACTS[origin_id]["mode"]
    return (
        ExtendedGoldenOriginCase(
            case_id=f"train.{origin_id}.{mode}",
            case_kind="t048_train_registered_anomaly",
            spec=GoldenOriginSpec(
                normalized_subtask=EditingSubtask.SUBSTITUTE,
                origin_id=origin_id,
                policies=policies,
            ),
            coverage_tags=(
                "frozen_train",
                "registered_substitution_replay",
                str(mode),
            ),
        ),
    )


__all__ = [
    "SUBSTITUTION_OPERATOR_IDS",
    "SubstitutionCandidateDispatcher",
    "SubstitutionCandidateEngine",
    "SubstitutionOperatorMixin",
    "t048_substitution_boundary_cases",
]
