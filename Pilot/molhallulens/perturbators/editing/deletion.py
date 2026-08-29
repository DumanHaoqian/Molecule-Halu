"""Deterministic, root-only Deletion operators and connected-cut dispatch."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
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

    from . import DeletionPerturbator


DELETION_OPERATOR_IDS = (
    "mol_edit.delete.wrong_group_occurrence",
    "mol_edit.delete.wrong_adjacent_group",
    "mol_edit.delete.group_boundary_contract",
    "mol_edit.delete.group_boundary_expand",
    "mol_edit.delete.partial_deletion",
    "mol_edit.delete.over_deletion",
    "mol_edit.delete.matched_remove_group",
    "mol_edit.delete.alternative_deprotection_product",
    "mol_edit.delete.cross_step_group_identity",
    "mol_edit.delete.heavy_count_claim",
    "mol_edit.delete.ring_count_claim",
    "mol_edit.delete.terminal_answer",
)

_STRUCTURAL_POLICIES = frozenset(
    {PropagationPolicy.STOP, PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
)
_CLAIM_POLICIES = frozenset({PropagationPolicy.STOP, PropagationPolicy.PARTIAL})
_DETERMINISTIC_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.RDKIT, CandidateSourceType.HYBRID}
)
_RELATION_SOURCES = frozenset(
    {CandidateSourceType.RULE, CandidateSourceType.HYBRID}
)
_STRUCTURAL_CAPABILITY = frozenset({OperatorCapability.STRUCTURAL_DELETION})
_CLAIM_CAPABILITY = frozenset({OperatorCapability.CLAIM_PERTURBATION})


class DeletionOperatorMixin:
    """The twelve T020 members; chemistry stays in the injected engine."""

    def _deletion_candidate_pool(self, context: PerturbationContext[Any]) -> CandidatePool:
        engine = self.candidate_engine  # type: ignore[attr-defined]
        if type(engine) is not DeletionCandidateEngine:
            raise OperatorRegistryError(
                code="DELETION_CANDIDATE_ENGINE_REQUIRED",
                operator_id=context.recipe.operator_id,
                detail="Deletion operators require the deterministic T020 engine",
            )
        return engine._pool_from_member(self, context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[0],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_wrong_group_occurrence(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[1],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_wrong_adjacent_group(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[2],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_group_boundary_contract(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[3],
        operator_family="attachment_bond_edit",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_group_boundary_expand(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[4],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_partial_deletion(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[5],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_over_deletion(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[6],
        operator_family="wrong_fragment_group",
        root_fields={"remove_group_step1", "remove_group_step2"},
        supported_policies=_STRUCTURAL_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_matched_remove_group(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[7],
        operator_family="wrong_fragment_group",
        root_fields={"product"},
        supported_policies={PropagationPolicy.FULL_CF},
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.PRODUCT_CONSTRUCTION},
        required_capabilities=_STRUCTURAL_CAPABILITY,
    )
    def perturb_alternative_deprotection_product(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[8],
        operator_family="nl_formal_internal_relation",
        root_fields={"remove_group_step1", "remove_group_step2"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_RELATION_SOURCES,
        hallucination_types={HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_cross_step_group_identity(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[9],
        operator_family="numeric_count_claim",
        root_fields={"remove_heavy", "source_heavy", "product_heavy", "heavy_delta"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.HEAVY_ATOM_COUNT, EditErrorSubtype.HEAVY_ATOM_ARITHMETIC},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_heavy_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[10],
        operator_family="numeric_count_claim",
        root_fields={"source_rings", "product_rings", "ring_delta"},
        supported_policies=_CLAIM_POLICIES,
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.RING_COUNT, EditErrorSubtype.RING_ARITHMETIC},
        required_capabilities=_CLAIM_CAPABILITY,
    )
    def perturb_ring_count_claim(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)

    @operator(
        operator_id=DELETION_OPERATOR_IDS[11],
        operator_family="final_answer_identity",
        root_fields={"final_answer"},
        supported_policies={PropagationPolicy.TERMINAL},
        supported_sources=_DETERMINISTIC_SOURCES,
        hallucination_types={HallucinationType.CONTRADICTION, HallucinationType.REASONING_ERROR},
        edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
        required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
    )
    def perturb_terminal_answer(
        self, context: PerturbationContext[Any]
    ) -> CandidatePool:
        return self._deletion_candidate_pool(context)


@dataclass(frozen=True, slots=True)
class DeletionCandidateDispatcher:
    """Resolve and invoke one exact Deletion member before the T018 boundary."""

    operators_config: OperatorsConfig

    def __post_init__(self) -> None:
        if type(self.operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")

    def _registry(self) -> PerturbatorRegistry:
        from . import DeletionPerturbator

        return PerturbatorRegistry.from_perturbator_types(
            (DeletionPerturbator,), operators_config=self.operators_config
        )

    def invoke(
        self,
        perturbator: DeletionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        return self._registry().invoke(perturbator, context)

    def build_member_pool(
        self,
        perturbator: DeletionPerturbator,
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
                        _enumerate_deletion_proposals,
                        source=CandidateSourceType.RULE,
                    )
                ),
                RDKitCandidateSource(
                    partial(
                        _enumerate_deletion_proposals,
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


class DeletionCandidateEngine(CandidateEngine[Any]):
    """CandidateEngine whose production path is registry→member→T018."""

    __slots__ = ("_dispatcher", "_owner")

    def __init__(self, *, operators_config: OperatorsConfig) -> None:
        self._dispatcher = DeletionCandidateDispatcher(operators_config)
        self._owner: DeletionPerturbator | None = None

    @property
    def dispatcher(self) -> DeletionCandidateDispatcher:
        return self._dispatcher

    def bind_owner(self, perturbator: DeletionPerturbator) -> None:
        from . import DeletionPerturbator

        if type(perturbator) is not DeletionPerturbator:
            raise TypeError("DeletionCandidateEngine owner must be DeletionPerturbator")
        if self._owner is not None and self._owner is not perturbator:
            raise RuntimeError("DeletionCandidateEngine is already bound")
        self._owner = perturbator

    def _require_owner(self) -> DeletionPerturbator:
        if self._owner is None:
            raise RuntimeError("DeletionCandidateEngine is not bound")
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
            raise ValueError("cannot select from an empty Deletion candidate pool")
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
                evidence={"actual": len(pool.candidates), "minimum": minimum},
            )
        return pool.candidates[0]

    def _pool_from_member(
        self,
        perturbator: DeletionPerturbator,
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        if perturbator is not self._require_owner():
            raise RuntimeError("DeletionCandidateEngine owner mismatch")
        return self._dispatcher.build_member_pool(perturbator, context)


@dataclass(frozen=True, slots=True)
class _CutCandidate:
    """One source-induced connected component separated by exactly one bond."""

    occurrence_atom_maps: tuple[int, ...]
    anchor_atom_map: int
    fragment_smiles: str
    bond_type: BondTypeName
    product_smiles: str


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


def _mapped_source(request: CandidateRequest) -> tuple[Chem.Mol, dict[int, int]]:
    molecule = _strict_molecule(request.context.record.indexed_smiles)
    mapped: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0 or atom_map in mapped:
            raise ValueError("Deletion sources require unique positive atom maps")
        mapped[atom_map] = atom.GetIdx()
    return molecule, mapped


def _component_without_bond(
    molecule: Chem.Mol,
    start: int,
    blocked_bond_index: int,
) -> frozenset[int]:
    visited: set[int] = set()
    frontier = deque([start])
    while frontier:
        atom_index = frontier.popleft()
        if atom_index in visited:
            continue
        visited.add(atom_index)
        for bond in molecule.GetAtomWithIdx(atom_index).GetBonds():
            if bond.GetIdx() == blocked_bond_index:
                continue
            neighbor = bond.GetOtherAtomIdx(atom_index)
            if neighbor not in visited:
                frontier.append(neighbor)
    return frozenset(visited)


def _fragment_smiles(
    molecule: Chem.Mol,
    atom_indices: frozenset[int],
) -> str | None:
    map_free = Chem.Mol(molecule)
    for atom in map_free.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        with rdBase.BlockLogs():
            serialized = Chem.MolFragmentToSmiles(
                map_free,
                atomsToUse=sorted(atom_indices),
                canonical=True,
                isomericSmiles=True,
            )
        if not serialized or "." in serialized:
            return None
        canonical = canonicalize_smiles(
            serialized,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    try:
        parsed = _strict_molecule(canonical)
    except ValueError:
        return None
    if len(Chem.GetMolFrags(parsed)) != 1 or parsed.GetNumAtoms() != len(atom_indices):
        return None
    return canonical


def _deletion_action(candidate: _CutCandidate) -> EditAction:
    return EditAction(
        edit_kind=EditKind.DELETION,
        source_anchor_index=candidate.anchor_atom_map,
        remove_fragment_smiles=candidate.fragment_smiles,
        bond_type=candidate.bond_type,
        metadata={"occurrence_atom_maps": candidate.occurrence_atom_maps},
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


def _source_cut_candidates(request: CandidateRequest) -> tuple[_CutCandidate, ...]:
    """Enumerate only induced, connected, one-boundary source components."""

    molecule, _ = _mapped_source(request)
    all_indices = frozenset(range(molecule.GetNumAtoms()))
    candidates: dict[tuple[Any, ...], _CutCandidate] = {}
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_side = _component_without_bond(molecule, begin, bond.GetIdx())
        if end in begin_side:
            continue  # ring edge: removing it does not define a deletion component
        end_side = all_indices - begin_side
        for occurrence, anchor_index in ((begin_side, end), (end_side, begin)):
            if not occurrence or len(occurrence) == molecule.GetNumAtoms():
                continue
            boundaries = tuple(
                candidate_bond
                for atom_index in occurrence
                for candidate_bond in molecule.GetAtomWithIdx(atom_index).GetBonds()
                if candidate_bond.GetOtherAtomIdx(atom_index) not in occurrence
            )
            if len(boundaries) != 1:
                continue
            boundary = boundaries[0]
            if boundary.GetOtherAtomIdx(
                boundary.GetBeginAtomIdx()
                if boundary.GetBeginAtomIdx() in occurrence
                else boundary.GetEndAtomIdx()
            ) != anchor_index:
                continue
            fragment = _fragment_smiles(molecule, occurrence)
            bond_type = _BOND_TYPE_NAMES.get(bond.GetBondType())
            if fragment is None or bond_type is None:
                continue
            maps = tuple(
                sorted(molecule.GetAtomWithIdx(index).GetAtomMapNum() for index in occurrence)
            )
            anchor_map = molecule.GetAtomWithIdx(anchor_index).GetAtomMapNum()
            provisional = _CutCandidate(
                occurrence_atom_maps=maps,
                anchor_atom_map=anchor_map,
                fragment_smiles=fragment,
                bond_type=bond_type,
                product_smiles="C",
            )
            product = _replayed_product(request, _deletion_action(provisional))
            if product is None or isomeric_graph_equivalent(
                product, request.context.truth.canonical_gt_smiles
            ):
                continue
            candidate = replace(provisional, product_smiles=product)
            key = (maps, anchor_map, fragment, bond_type.value, product)
            candidates[key] = candidate
    return tuple(candidates[key] for key in sorted(candidates))


def _truth_removed(request: CandidateRequest) -> frozenset[int]:
    return frozenset(request.context.truth.removed_atom_maps)


def _truth_fragment(request: CandidateRequest) -> str:
    fragment = request.context.truth.remove_fragment
    if fragment is None:
        raise ValueError("Deletion truth must contain a remove fragment")
    return fragment.canonical_smiles


def _candidate_map_set(candidate: _CutCandidate) -> frozenset[int]:
    return frozenset(candidate.occurrence_atom_maps)


def _same_fragment(candidate: _CutCandidate, reference: str) -> bool:
    try:
        return fragment_graph_equivalent(candidate.fragment_smiles, reference)
    except ValueError:
        return False


def _matched_fragment(candidate: _CutCandidate, reference: str) -> bool:
    if _same_fragment(candidate, reference):
        return False
    actual = compute_descriptors(
        candidate.fragment_smiles,
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    expected = compute_descriptors(reference, fragment_policy=FragmentPolicy.KEEP_ALL)
    return (
        actual.heavy_atom_count == expected.heavy_atom_count
        and actual.ring_count == expected.ring_count
        and actual.formal_charge == expected.formal_charge
        and actual.heteroatom_counts == expected.heteroatom_counts
    )


def _structural_cut_candidates(request: CandidateRequest) -> tuple[_CutCandidate, ...]:
    cuts = _source_cut_candidates(request)
    truth_removed = _truth_removed(request)
    truth_fragment = _truth_fragment(request)
    truth_anchors = frozenset(request.context.truth.valid_anchor_indices)
    operator_id = request.operator_id

    if operator_id == DELETION_OPERATOR_IDS[0]:
        selected = tuple(
            cut
            for cut in cuts
            if _candidate_map_set(cut) != truth_removed
            and _same_fragment(cut, truth_fragment)
        )
    elif operator_id == DELETION_OPERATOR_IDS[1]:
        selected = tuple(
            cut
            for cut in cuts
            if _candidate_map_set(cut) != truth_removed
            and (
                bool(_candidate_map_set(cut).intersection(truth_removed))
                or cut.anchor_atom_map in truth_removed
                or bool(truth_anchors.intersection(_candidate_map_set(cut)))
            )
        )
    elif operator_id in {DELETION_OPERATOR_IDS[2], DELETION_OPERATOR_IDS[4]}:
        selected = tuple(
            cut
            for cut in cuts
            if _candidate_map_set(cut) < truth_removed
        )
    elif operator_id in {DELETION_OPERATOR_IDS[3], DELETION_OPERATOR_IDS[5]}:
        selected = tuple(
            cut
            for cut in cuts
            if _candidate_map_set(cut) > truth_removed
        )
    elif operator_id == DELETION_OPERATOR_IDS[6]:
        selected = tuple(
            cut for cut in cuts if _matched_fragment(cut, truth_fragment)
        )
    elif operator_id == DELETION_OPERATOR_IDS[7]:
        selected = cuts
    else:
        selected = ()
    return tuple(
        sorted(
            selected,
            key=lambda cut: (
                len(_candidate_map_set(cut).symmetric_difference(truth_removed)),
                -len(cut.occurrence_atom_maps),
                cut.occurrence_atom_maps,
                cut.anchor_atom_map,
                cut.fragment_smiles,
                cut.product_smiles,
            ),
        )
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
            "fragment": action.remove_fragment_smiles,
            "bond": None if action.bond_type is None else action.bond_type.value,
            "occurrence": action.metadata.get("occurrence_atom_maps"),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"delete:{source.value.lower()}:{digest}"


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
                "generator": "deletion_t020",
                "operator_id": request.operator_id,
            },
        ),
        candidate_product_smiles=product,
        difficulty_features=CandidateDifficultyFeatures(
            source_score=1.0 if source is CandidateSourceType.RDKIT else 0.75
        ),
    )


def _claim_fragment_values(request: CandidateRequest) -> tuple[str, ...]:
    old = request.context.reference_graph.value_for(
        request.resolution.target_node_id
    ).normalized_value
    values = {
        cut.fragment_smiles
        for cut in _source_cut_candidates(request)
        if cut.fragment_smiles != old
    }
    return tuple(sorted(values))


def _count_values(request: CandidateRequest) -> tuple[int, ...]:
    """Use observed graph descriptors, never a fixed +/-1 corruption rule."""

    root = request.resolution.target_node_id
    truth = request.context.truth
    cuts = _source_cut_candidates(request)
    fragment_descriptors = tuple(
        compute_descriptors(
            cut.fragment_smiles,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        for cut in cuts
    )
    product_descriptors = tuple(
        compute_descriptors(
            cut.product_smiles,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
        for cut in cuts
    )
    if root in {"remove_heavy", "source_heavy", "product_heavy", "heavy_delta"}:
        values = {
            truth.source_descriptors.heavy_atom_count,
            truth.product_descriptors.heavy_atom_count,
            *(item.heavy_atom_count for item in fragment_descriptors),
            *(item.heavy_atom_count for item in product_descriptors),
        }
    elif root in {"source_rings", "product_rings", "ring_delta"}:
        values = {
            truth.source_descriptors.ring_count,
            truth.product_descriptors.ring_count,
            *(item.ring_count for item in fragment_descriptors),
            *(item.ring_count for item in product_descriptors),
        }
    else:
        raise ValueError("unsupported Deletion count root")
    # A DELETE_WITH_REPLACEMENT classification forbids the ordinary deprotection
    # identities source-remove and -remove.  Apply the capability policy rather
    # than keying behavior to an opaque origin ID.
    if not request.resolution.classification.allows(
        OperatorCapability.REMOVE_ONLY_DELTA_RULE
    ):
        source_heavy = truth.source_descriptors.heavy_atom_count
        removed_heavy = (
            0
            if truth.remove_fragment is None
            else truth.remove_fragment.descriptors.heavy_atom_count
        )
        forbidden_by_root = {
            "product_heavy": source_heavy - removed_heavy,
            "heavy_delta": -removed_heavy,
        }
        forbidden = forbidden_by_root.get(root)
        if forbidden is not None:
            values.discard(forbidden)
    old = request.context.reference_graph.value_for(root).normalized_value
    return tuple(sorted(value for value in values if value != old and value >= 0))


def _enumerate_deletion_proposals(
    request: CandidateRequest,
    *,
    source: CandidateSourceType,
) -> Iterable[CandidateProposal]:
    """Enumerate operator-owned proposals; T018 remains the acceptance gate."""

    from molhallulens.candidates import CandidateRequest

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if source not in {CandidateSourceType.RULE, CandidateSourceType.RDKIT}:
        raise TypeError("Deletion proposal source must be RULE or RDKIT")
    root = request.resolution.target_node_id
    operator_id = request.operator_id

    if operator_id == DELETION_OPERATOR_IDS[8]:
        for value in _claim_fragment_values(request):
            try:
                yield _proposal(request, source, value=value)
            except ValueError:
                continue
        return
    if operator_id in DELETION_OPERATOR_IDS[9:11]:
        for value in _count_values(request):
            yield _proposal(request, source, value=value)
        return
    if operator_id == DELETION_OPERATOR_IDS[11]:
        # Terminal errors intentionally carry no structural action: this keeps
        # delete-with-replacement origins inside TERMINAL_PERTURBATION policy.
        for cut in _source_cut_candidates(request):
            try:
                yield _proposal(request, source, value=cut.product_smiles)
            except ValueError:
                continue
        return

    for cut in _structural_cut_candidates(request):
        action = _deletion_action(cut)
        value = (
            cut.product_smiles
            if root == "product"
            else cut.fragment_smiles
        )
        try:
            yield _proposal(
                request,
                source,
                value=value,
                action=action,
                product=cut.product_smiles,
            )
        except ValueError:
            continue


__all__ = [
    "DELETION_OPERATOR_IDS",
    "DeletionCandidateDispatcher",
    "DeletionCandidateEngine",
    "DeletionOperatorMixin",
]
