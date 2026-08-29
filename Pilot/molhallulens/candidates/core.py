"""Deterministic, chemistry-checked candidate pooling for T018.

Concrete editing operators own proposal enumeration.  This module owns the
shared fail-closed boundary after enumeration: strict source contracts,
canonical chemistry, reference/symmetry exclusion, semantic de-duplication,
and stable difficulty ranking.  It deliberately does not propagate or render.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from typing import Any, Protocol, runtime_checkable

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from molhallulens.chemistry import (
    MoleculeParseError,
    canonicalize_smiles,
    compute_descriptors,
    isomeric_graph_equivalent,
)
from molhallulens.domain import (
    BondTypeName,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    ClaimValue,
    ComparatorKind,
    EditAction,
    EditKind,
    EditTruth,
    FragmentPolicy,
    FrozenMap,
    ValueProvenance,
    ValueType,
)
from molhallulens.perturbators.base import PerturbationContext
from molhallulens.perturbators.registry import OperatorResolution


class CandidateRejectCode(StrEnum):
    """Stable reasons emitted by the deterministic candidate boundary."""

    SOURCE_FAILED = "SOURCE_FAILED"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNSUPPORTED_SOURCE_MODE = "UNSUPPORTED_SOURCE_MODE"
    ROOT_MISMATCH = "ROOT_MISMATCH"
    REFERENCE_VALUE_MISMATCH = "REFERENCE_VALUE_MISMATCH"
    STRUCTURAL_PRODUCT_MISSING = "STRUCTURAL_PRODUCT_MISSING"
    ATTACHMENT_SEMANTICS_MISSING = "ATTACHMENT_SEMANTICS_MISSING"
    SMILES_INVALID = "SMILES_INVALID"
    REFERENCE_EQUIVALENT = "REFERENCE_EQUIVALENT"
    SYMMETRY_EQUIVALENT = "SYMMETRY_EQUIVALENT"
    DUPLICATE = "DUPLICATE"
    ACTION_PRODUCT_MISMATCH = "ACTION_PRODUCT_MISMATCH"
    INSUFFICIENT_CANDIDATES = "INSUFFICIENT_CANDIDATES"


class CandidateSourceError(RuntimeError):
    """Structured failure at a rule/RDKit proposal-source boundary."""

    def __init__(
        self,
        *,
        code: CandidateRejectCode,
        source: CandidateSourceType,
        detail: str,
    ) -> None:
        if type(code) is not CandidateRejectCode:
            raise TypeError("code must be CandidateRejectCode")
        if type(source) is not CandidateSourceType:
            raise TypeError("source must be CandidateSourceType")
        if type(detail) is not str or not detail:
            raise ValueError("detail must be non-empty text")
        self.code = code
        self.source = source
        self.detail = detail
        super().__init__(f"{code.value} from {source.value}: {detail}")


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    """Exact validated context and T017 resolution used for one pool build."""

    context: PerturbationContext[EditTruth]
    resolution: OperatorResolution

    def __post_init__(self) -> None:
        if not isinstance(self.context, PerturbationContext):
            raise TypeError("context must be PerturbationContext")
        if type(self.resolution) is not OperatorResolution:
            raise TypeError("resolution must be OperatorResolution")
        if type(self.context.truth) is not EditTruth:
            raise TypeError("candidate requests require exact EditTruth")
        recipe = self.context.recipe
        registration = self.resolution.registration
        if not (
            recipe.operator_id == registration.operator_id
            and recipe.policy is self.resolution.policy
            and recipe.candidate_source_mode is self.resolution.candidate_source
            and recipe.target_node_id == self.resolution.target_node_id
            and recipe.target_node_id in registration.spec.root_fields
            and self.context.record.family is registration.task_family
            and self.context.record.normalized_subtask is registration.subtask
            and self.context.truth.normalized_subtask is registration.subtask
            and self.context.truth.anonymous_sample_id == self.context.record.origin_id
            and self.resolution.classification.anonymous_sample_id
            == self.context.record.origin_id
        ):
            raise ValueError("context and operator resolution do not describe one request")

    @property
    def request_id(self) -> str:
        return self.context.recipe.recipe_id

    @property
    def operator_id(self) -> str:
        return self.resolution.registration.operator_id

    @property
    def derived_seed(self) -> int:
        return self.context.recipe.derived_seed


@dataclass(frozen=True, slots=True)
class CandidateDifficultyFeatures:
    """Comparable chemistry/environment features; ``None`` means not applicable."""

    structural_similarity: float | None = None
    heavy_atom_delta: int | None = None
    ring_delta: int | None = None
    formal_charge_delta: int | None = None
    heteroatom_l1_distance: int | None = None
    anchor_element_match: bool | None = None
    anchor_aromaticity_match: bool | None = None
    anchor_degree_match: bool | None = None
    anchor_hybridization_match: bool | None = None
    source_score: float = 0.0

    def __post_init__(self) -> None:
        if self.structural_similarity is not None and (
            type(self.structural_similarity) not in {int, float}
            or not math.isfinite(float(self.structural_similarity))
            or not 0.0 <= float(self.structural_similarity) <= 1.0
        ):
            raise ValueError("structural_similarity must be finite in [0, 1] or None")
        for value, name in (
            (self.heavy_atom_delta, "heavy_atom_delta"),
            (self.ring_delta, "ring_delta"),
            (self.formal_charge_delta, "formal_charge_delta"),
        ):
            if value is not None and type(value) is not int:
                raise TypeError(f"{name} must be an integer or None")
        if self.heteroatom_l1_distance is not None and (
            type(self.heteroatom_l1_distance) is not int
            or self.heteroatom_l1_distance < 0
        ):
            raise ValueError("heteroatom_l1_distance must be non-negative or None")
        for value, name in (
            (self.anchor_element_match, "anchor_element_match"),
            (self.anchor_aromaticity_match, "anchor_aromaticity_match"),
            (self.anchor_degree_match, "anchor_degree_match"),
            (self.anchor_hybridization_match, "anchor_hybridization_match"),
        ):
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be bool or None")
        if type(self.source_score) not in {int, float} or not math.isfinite(
            float(self.source_score)
        ):
            raise ValueError("source_score must be finite")


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """A root-only patch plus private structural evidence used before propagation."""

    proposal_id: str
    patch: CandidatePatch
    candidate_product_smiles: str | None = None
    difficulty_features: CandidateDifficultyFeatures = CandidateDifficultyFeatures()

    def __post_init__(self) -> None:
        if type(self.proposal_id) is not str or not self.proposal_id:
            raise ValueError("proposal_id must be non-empty text")
        if type(self.patch) is not CandidatePatch:
            raise TypeError("patch must be CandidatePatch")
        if self.candidate_product_smiles is not None and (
            type(self.candidate_product_smiles) is not str
            or not self.candidate_product_smiles.strip()
        ):
            raise ValueError("candidate_product_smiles must be non-empty text or None")
        if type(self.difficulty_features) is not CandidateDifficultyFeatures:
            raise TypeError("difficulty_features must be CandidateDifficultyFeatures")


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    """One immutable, plaintext-minimizing candidate rejection ledger row."""

    code: CandidateRejectCode
    proposal_id: str
    operator_id: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.code) is not CandidateRejectCode:
            raise TypeError("code must be CandidateRejectCode")
        for value, name in (
            (self.proposal_id, "proposal_id"),
            (self.operator_id, "operator_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        if any(type(key) is not str or not key for key in self.evidence):
            raise TypeError("evidence keys must be non-empty strings")
        object.__setattr__(self, "evidence", FrozenMap(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "proposal_id": self.proposal_id,
            "operator_id": self.operator_id,
            "evidence": _json_value(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Canonical accepted proposal and its complete deterministic sort key."""

    proposal: CandidateProposal
    canonical_product_smiles: str | None
    canonical_key: str
    difficulty_features: CandidateDifficultyFeatures
    rank_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        if type(self.proposal) is not CandidateProposal:
            raise TypeError("proposal must be CandidateProposal")
        if self.canonical_product_smiles is not None and (
            type(self.canonical_product_smiles) is not str
            or not self.canonical_product_smiles
        ):
            raise ValueError("canonical_product_smiles must be non-empty text or None")
        if type(self.canonical_key) is not str or not self.canonical_key:
            raise ValueError("canonical_key must be non-empty text")
        if type(self.difficulty_features) is not CandidateDifficultyFeatures:
            raise TypeError("difficulty_features must be CandidateDifficultyFeatures")
        object.__setattr__(self, "rank_key", tuple(self.rank_key))


@dataclass(frozen=True, slots=True)
class CandidateBuildResult:
    """CandidatePool plus the structured audit information it cannot itself hold."""

    pool: CandidatePool
    rejections: tuple[CandidateRejection, ...]
    ranked_candidates: tuple[RankedCandidate, ...] = ()

    def __post_init__(self) -> None:
        if type(self.pool) is not CandidatePool:
            raise TypeError("pool must be CandidatePool")
        rejections = tuple(self.rejections)
        ranked = tuple(self.ranked_candidates)
        if any(type(item) is not CandidateRejection for item in rejections):
            raise TypeError("rejections must contain CandidateRejection values")
        if any(type(item) is not RankedCandidate for item in ranked):
            raise TypeError("ranked_candidates must contain RankedCandidate values")
        if ranked and tuple(item.proposal.patch for item in ranked) != self.pool.candidates:
            raise ValueError("ranked_candidates order must exactly match CandidatePool")
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "ranked_candidates", ranked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.pool.request_id,
            "accepted_candidate_ids": tuple(
                candidate.candidate_id for candidate in self.pool.candidates
            ),
            "rejection_codes": self.pool.rejection_codes,
            "rejections": tuple(item.to_dict() for item in self.rejections),
        }


@runtime_checkable
class CandidateSource(Protocol):
    """A deterministic root-proposal source; no global registry is consulted."""

    source_type: CandidateSourceType

    def propose(self, request: CandidateRequest) -> Sequence[CandidateProposal]: ...


ProposalFunction = Callable[[CandidateRequest], Iterable[CandidateProposal]]


def _chemical_action_metadata(action: EditAction) -> dict[str, Any]:
    primary_occurrence = action.metadata.get("remove_atom_maps")
    alternate_occurrence = action.metadata.get("occurrence_atom_maps")
    if (
        primary_occurrence is not None
        and alternate_occurrence is not None
        and primary_occurrence != alternate_occurrence
    ):
        raise ValueError("conflicting chemical occurrence identities")
    occurrence = (
        primary_occurrence
        if primary_occurrence is not None
        else alternate_occurrence
    )
    if occurrence is None:
        return {}
    if isinstance(occurrence, (str, bytes)) or not isinstance(
        occurrence, (tuple, list, frozenset)
    ):
        raise TypeError("occurrence atom maps must be an integer collection")
    normalized = tuple(sorted(occurrence))
    if (
        not normalized
        or any(type(item) is not int or item <= 0 for item in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("occurrence atom maps must contain unique positive atom maps")
    return {"occurrence_atom_maps": normalized}


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_value(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if type(value) in {type(None), bool, int, float, str}:
        return value
    raise TypeError(f"candidate semantic payload contains {type(value).__name__}")


def _canonical_fragment(value: str | None) -> str | None:
    if value is None:
        return None
    return canonicalize_smiles(value, fragment_policy=FragmentPolicy.KEEP_ALL)


def _canonical_fragment_attachment(
    value: str,
    attachment_atom: int,
) -> tuple[str, int]:
    """Canonicalize a fragment and remap its input atom index to output order."""

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value, sanitize=True)
    if molecule is None:
        # Reuse the public utility to raise its stable MoleculeParseError.
        canonicalize_smiles(value, fragment_policy=FragmentPolicy.KEEP_ALL)
        raise AssertionError("unreachable after failed strict canonicalization")
    if attachment_atom < 0 or attachment_atom >= molecule.GetNumAtoms():
        raise ValueError(CandidateRejectCode.ATTACHMENT_SEMANTICS_MISSING.value)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not canonical or not molecule.HasProp("_smilesAtomOutputOrder"):
        raise ValueError(CandidateRejectCode.SMILES_INVALID.value)
    raw_order = molecule.GetProp("_smilesAtomOutputOrder")
    order = tuple(
        int(item.strip())
        for item in raw_order.strip().strip("[]").split(",")
        if item.strip()
    )
    if len(order) != molecule.GetNumAtoms() or attachment_atom not in order:
        raise ValueError(CandidateRejectCode.ATTACHMENT_SEMANTICS_MISSING.value)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    symmetry_ranks = tuple(
        Chem.CanonicalRankAtoms(
            molecule,
            breakTies=False,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    selected_rank = symmetry_ranks[attachment_atom]
    equivalent_positions = tuple(
        order.index(atom_index)
        for atom_index, rank in enumerate(symmetry_ranks)
        if rank == selected_rank
    )
    return canonical, min(equivalent_positions)


def _canonical_claim(claim: ClaimValue) -> ClaimValue:
    if claim.value_type not in {
        ValueType.SMILES,
        ValueType.MOLECULE,
        ValueType.FRAGMENT,
    }:
        return claim
    normalized = claim.normalized_value
    if type(normalized) is not str:
        raise TypeError("molecular ClaimValue normalized payload must be text")
    canonical = canonicalize_smiles(
        normalized,
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    raw = canonical if type(claim.raw_value) is str else claim.raw_value
    return replace(claim, raw_value=raw, normalized_value=canonical)


def _canonical_action(action: EditAction | None) -> EditAction | None:
    if action is None:
        return None
    add_fragment = _canonical_fragment(action.add_fragment_smiles)
    attachment_atom = action.fragment_attachment_atom
    if action.add_fragment_smiles is not None and attachment_atom is not None:
        add_fragment, attachment_atom = _canonical_fragment_attachment(
            action.add_fragment_smiles,
            attachment_atom,
        )
    return replace(
        action,
        remove_fragment_smiles=_canonical_fragment(action.remove_fragment_smiles),
        add_fragment_smiles=add_fragment,
        fragment_attachment_atom=attachment_atom,
    )


def _canonicalize_proposal(proposal: CandidateProposal) -> CandidateProposal:
    patch = proposal.patch
    canonical_patch = replace(
        patch,
        new_value=_canonical_claim(patch.new_value),
        edit_action=_canonical_action(patch.edit_action),
    )
    product = (
        None
        if proposal.candidate_product_smiles is None
        else canonicalize_smiles(
            proposal.candidate_product_smiles,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
    )
    return replace(
        proposal,
        patch=canonical_patch,
        candidate_product_smiles=product,
    )


def canonical_candidate_key(proposal: CandidateProposal) -> str:
    """Return a stable semantic key that retains full attachment semantics."""

    if type(proposal) is not CandidateProposal:
        raise TypeError("proposal must be CandidateProposal")
    canonical = _canonicalize_proposal(proposal)
    action = canonical.patch.edit_action
    action_payload = None
    if action is not None:
        action_payload = {
            "edit_kind": action.edit_kind,
            "source_anchor_index": action.source_anchor_index,
            "remove_fragment_smiles": action.remove_fragment_smiles,
            "add_fragment_smiles": action.add_fragment_smiles,
            "fragment_attachment_atom": action.fragment_attachment_atom,
            "bond_type": action.bond_type,
            "metadata": _chemical_action_metadata(action),
        }
    payload = {
        "root_node_id": canonical.patch.root_node_id,
        "value_type": canonical.patch.new_value.value_type,
        "normalized_value": canonical.patch.new_value.normalized_value,
        "edit_action": action_payload,
        "candidate_product_smiles": canonical.candidate_product_smiles,
    }
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _morgan_similarity(reference_smiles: str, candidate_smiles: str) -> float:
    with rdBase.BlockLogs():
        reference = Chem.MolFromSmiles(reference_smiles, sanitize=True)
        candidate = Chem.MolFromSmiles(candidate_smiles, sanitize=True)
    if reference is None or candidate is None:  # canonical inputs should make this unreachable
        raise ValueError("canonical molecule failed deterministic fingerprint parsing")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(reference),
            generator.GetFingerprint(candidate),
        )
    )


def _atom_for_map(molecule: Chem.Mol, atom_map: int) -> Chem.Atom | None:
    matches = tuple(atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == atom_map)
    return matches[0] if len(matches) == 1 else None


def _anchor_features(
    request: CandidateRequest,
    proposal: CandidateProposal,
) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    patch = proposal.patch
    candidate_anchor: int | None = None
    reference_anchor: int | None = None
    if patch.root_node_id == "anchor_idx":
        if type(patch.new_value.normalized_value) is int:
            candidate_anchor = patch.new_value.normalized_value
        if type(patch.old_value.normalized_value) is int:
            reference_anchor = patch.old_value.normalized_value
    elif patch.edit_action is not None:
        candidate_anchor = patch.edit_action.source_anchor_index
        if len(request.context.truth.valid_anchor_indices) == 1:
            reference_anchor = request.context.truth.valid_anchor_indices[0]
    if candidate_anchor is None or reference_anchor is None:
        return (None, None, None, None)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(request.context.record.indexed_smiles, sanitize=True)
    if molecule is None:
        return (None, None, None, None)
    candidate = _atom_for_map(molecule, candidate_anchor)
    reference = _atom_for_map(molecule, reference_anchor)
    if candidate is None or reference is None:
        return (None, None, None, None)
    return (
        candidate.GetAtomicNum() == reference.GetAtomicNum(),
        candidate.GetIsAromatic() == reference.GetIsAromatic(),
        candidate.GetDegree() == reference.GetDegree(),
        candidate.GetHybridization() == reference.GetHybridization(),
    )


def compute_difficulty_features(
    request: CandidateRequest,
    proposal: CandidateProposal,
) -> CandidateDifficultyFeatures:
    """Compute deterministic chemistry features, preserving source-only score."""

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if type(proposal) is not CandidateProposal:
        raise TypeError("proposal must be CandidateProposal")
    supplied = proposal.difficulty_features
    anchor = _anchor_features(request, proposal)
    if proposal.candidate_product_smiles is None:
        return replace(
            supplied,
            anchor_element_match=(
                anchor[0] if anchor[0] is not None else supplied.anchor_element_match
            ),
            anchor_aromaticity_match=(
                anchor[1]
                if anchor[1] is not None
                else supplied.anchor_aromaticity_match
            ),
            anchor_degree_match=(
                anchor[2] if anchor[2] is not None else supplied.anchor_degree_match
            ),
            anchor_hybridization_match=(
                anchor[3]
                if anchor[3] is not None
                else supplied.anchor_hybridization_match
            ),
        )
    reference_smiles = request.context.truth.canonical_gt_smiles
    product_smiles = proposal.candidate_product_smiles
    reference = compute_descriptors(
        reference_smiles, fragment_policy=FragmentPolicy.KEEP_ALL
    )
    product = compute_descriptors(
        product_smiles, fragment_policy=FragmentPolicy.KEEP_ALL
    )
    reference_hetero = Counter(dict(reference.heteroatom_counts))
    product_hetero = Counter(dict(product.heteroatom_counts))
    hetero_l1 = sum(
        abs(reference_hetero[key] - product_hetero[key])
        for key in set(reference_hetero).union(product_hetero)
    )
    return CandidateDifficultyFeatures(
        structural_similarity=_morgan_similarity(reference_smiles, product_smiles),
        heavy_atom_delta=product.heavy_atom_count - reference.heavy_atom_count,
        ring_delta=product.ring_count - reference.ring_count,
        formal_charge_delta=product.formal_charge - reference.formal_charge,
        heteroatom_l1_distance=hetero_l1,
        anchor_element_match=(
            anchor[0] if anchor[0] is not None else supplied.anchor_element_match
        ),
        anchor_aromaticity_match=(
            anchor[1] if anchor[1] is not None else supplied.anchor_aromaticity_match
        ),
        anchor_degree_match=(
            anchor[2] if anchor[2] is not None else supplied.anchor_degree_match
        ),
        anchor_hybridization_match=(
            anchor[3]
            if anchor[3] is not None
            else supplied.anchor_hybridization_match
        ),
        source_score=supplied.source_score,
    )


def _rank_key(
    request: CandidateRequest,
    proposal: CandidateProposal,
    features: CandidateDifficultyFeatures,
    canonical_key: str,
) -> tuple[Any, ...]:
    matches = (
        features.anchor_element_match,
        features.anchor_aromaticity_match,
        features.anchor_degree_match,
        features.anchor_hybridization_match,
    )
    known_match_count = sum(item is True for item in matches)
    known_mismatch_count = sum(item is False for item in matches)
    seeded_identity = json.dumps(
        {
            "seed": request.derived_seed,
            "canonical_key": canonical_key,
            "source": proposal.patch.source.value,
            "candidate_id": proposal.patch.candidate_id,
            "proposal_id": proposal.proposal_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    seeded_sha256 = hashlib.sha256(seeded_identity.encode("utf-8")).hexdigest()
    return (
        -(features.structural_similarity if features.structural_similarity is not None else -1.0),
        -known_match_count,
        known_mismatch_count,
        abs(features.heavy_atom_delta) if features.heavy_atom_delta is not None else math.inf,
        abs(features.ring_delta) if features.ring_delta is not None else math.inf,
        abs(features.formal_charge_delta)
        if features.formal_charge_delta is not None
        else math.inf,
        features.heteroatom_l1_distance
        if features.heteroatom_l1_distance is not None
        else math.inf,
        -float(features.source_score),
        seeded_sha256,
        canonical_key,
        proposal.patch.source.value,
        proposal.patch.candidate_id,
        proposal.proposal_id,
    )


def rank_candidates(
    request: CandidateRequest,
    proposals: Iterable[CandidateProposal],
) -> tuple[RankedCandidate, ...]:
    """Canonicalize and totally order already-admissible proposals."""

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Iterable):
        raise TypeError("proposals must be a non-string iterable")
    ranked: list[RankedCandidate] = []
    for proposal in proposals:
        if type(proposal) is not CandidateProposal:
            raise TypeError("proposals must contain CandidateProposal values")
        canonical = _canonicalize_proposal(proposal)
        features = compute_difficulty_features(request, canonical)
        key = canonical_candidate_key(canonical)
        ranked.append(
            RankedCandidate(
                proposal=replace(canonical, difficulty_features=features),
                canonical_product_smiles=canonical.candidate_product_smiles,
                canonical_key=key,
                difficulty_features=features,
                rank_key=_rank_key(request, canonical, features, key),
            )
        )
    return tuple(sorted(ranked, key=lambda item: item.rank_key))


def _reject(
    request: CandidateRequest,
    proposal_id: str,
    code: CandidateRejectCode,
    **evidence: Any,
) -> CandidateRejection:
    if not evidence:
        evidence = {"phase": "candidate_validation"}
    return CandidateRejection(
        code=code,
        proposal_id=proposal_id,
        operator_id=request.operator_id,
        evidence=evidence,
    )


_RDKIT_BOND_TYPES = {
    BondTypeName.SINGLE: Chem.BondType.SINGLE,
    BondTypeName.DOUBLE: Chem.BondType.DOUBLE,
    BondTypeName.TRIPLE: Chem.BondType.TRIPLE,
    BondTypeName.AROMATIC: Chem.BondType.AROMATIC,
}
_BOND_VALENCE_UNITS = {
    BondTypeName.SINGLE: 1,
    BondTypeName.DOUBLE: 2,
    BondTypeName.TRIPLE: 3,
    BondTypeName.AROMATIC: 1,
}
_RDKIT_BOND_VALENCE_UNITS = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 1,
}


def _action_product_mismatch() -> ValueError:
    return ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)


def _validate_attachment_semantics(
    request: CandidateRequest,
    proposal: CandidateProposal,
) -> None:
    action = proposal.patch.edit_action
    if action is None:
        if proposal.patch.root_node_id == "product":
            raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
        return
    if proposal.candidate_product_smiles is None:
        raise ValueError(CandidateRejectCode.STRUCTURAL_PRODUCT_MISSING.value)
    if action.source_anchor_index is None or action.bond_type is None:
        raise ValueError(CandidateRejectCode.ATTACHMENT_SEMANTICS_MISSING.value)
    expected_kind = {
        "add": EditKind.ADDITION,
        "delete": EditKind.DELETION,
        "substitute": EditKind.SUBSTITUTION,
    }[request.resolution.registration.subtask.value]
    if action.edit_kind is not expected_kind:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    occurrence_keys = {"remove_atom_maps", "occurrence_atom_maps"}
    if action.edit_kind is EditKind.ADDITION:
        shape_valid = (
            action.remove_fragment_smiles is None
            and action.add_fragment_smiles is not None
            and action.fragment_attachment_atom is not None
            and not occurrence_keys.intersection(action.metadata)
        )
    elif action.edit_kind is EditKind.DELETION:
        shape_valid = (
            action.remove_fragment_smiles is not None
            and action.add_fragment_smiles is None
            and action.fragment_attachment_atom is None
            and bool(occurrence_keys.intersection(action.metadata))
        )
    else:
        shape_valid = (
            action.remove_fragment_smiles is not None
            and action.add_fragment_smiles is not None
            and action.fragment_attachment_atom is not None
            and bool(occurrence_keys.intersection(action.metadata))
        )
    if not shape_valid:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)


def _strict_replay_molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    return molecule


def _source_anchor(
    source: Chem.Mol,
    source_anchor_map: int,
) -> tuple[int, dict[int, int]]:
    map_to_indices: dict[int, list[int]] = {}
    for atom in source.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map > 0:
            map_to_indices.setdefault(atom_map, []).append(atom.GetIdx())
    if any(len(indices) != 1 for indices in map_to_indices.values()):
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    map_to_index = {
        atom_map: indices[0] for atom_map, indices in map_to_indices.items()
    }
    if source_anchor_map not in map_to_index:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    return map_to_index[source_anchor_map], map_to_index


def _consume_explicit_hydrogens(atom: Chem.Atom, count: int) -> None:
    explicit = atom.GetNumExplicitHs()
    if explicit:
        atom.SetNumExplicitHs(max(0, explicit - count))
        atom.UpdatePropertyCache(strict=False)


def _add_explicit_hydrogens(atom: Chem.Atom, count: int) -> None:
    if count <= 0:
        return
    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + count)
    atom.SetNoImplicit(True)
    atom.UpdatePropertyCache(strict=False)


def _canonical_replay_product(molecule: Chem.Mol) -> str:
    product = Chem.Mol(molecule)
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(product)
            Chem.AssignStereochemistry(product, cleanIt=True, force=True)
            serialized = Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=True,
            )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
        ) from error
    try:
        return canonicalize_smiles(
            serialized,
            fragment_policy=FragmentPolicy.KEEP_ALL,
        )
    except MoleculeParseError as error:
        raise ValueError(
            CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
        ) from error


def _combine_and_bond(
    core: Chem.Mol,
    core_anchor_index: int,
    fragment_smiles: str,
    fragment_attachment_atom: int,
    bond_type: BondTypeName,
    *,
    consume_core_hydrogens: int,
) -> Chem.Mol:
    fragment = _strict_replay_molecule(fragment_smiles)
    if not 0 <= fragment_attachment_atom < fragment.GetNumAtoms():
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    combined = Chem.RWMol(Chem.CombineMols(core, fragment))
    fragment_index = core.GetNumAtoms() + fragment_attachment_atom
    _consume_explicit_hydrogens(
        combined.GetAtomWithIdx(core_anchor_index),
        consume_core_hydrogens,
    )
    _consume_explicit_hydrogens(
        combined.GetAtomWithIdx(fragment_index),
        _BOND_VALENCE_UNITS[bond_type],
    )
    try:
        combined.AddBond(
            core_anchor_index,
            fragment_index,
            _RDKIT_BOND_TYPES[bond_type],
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
        ) from error
    return combined.GetMol()


def _occurrence_atom_maps(action: EditAction) -> tuple[int, ...]:
    primary = action.metadata.get("remove_atom_maps")
    alternate = action.metadata.get("occurrence_atom_maps")
    if primary is not None and alternate is not None and primary != alternate:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    raw = primary if primary is not None else alternate
    if isinstance(raw, (str, bytes)) or not isinstance(
        raw, (tuple, list, frozenset)
    ):
        raise _action_product_mismatch()
    atom_maps = tuple(sorted(raw))
    if (
        not atom_maps
        or any(type(item) is not int or item <= 0 for item in atom_maps)
        or len(atom_maps) != len(set(atom_maps))
    ):
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    return atom_maps


def _remove_exact_occurrence(
    source: Chem.Mol,
    anchor_index: int,
    map_to_index: Mapping[int, int],
    action: EditAction,
    *,
    require_boundary_bond_type: bool,
) -> tuple[Chem.Mol, int, Chem.BondType]:
    if action.remove_fragment_smiles is None or action.bond_type is None:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    atom_maps = _occurrence_atom_maps(action)
    if any(atom_map not in map_to_index for atom_map in atom_maps):
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    occurrence = frozenset(map_to_index[atom_map] for atom_map in atom_maps)
    if anchor_index in occurrence:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    query = _strict_replay_molecule(action.remove_fragment_smiles)
    matched_sets = {
        frozenset(match)
        for match in source.GetSubstructMatches(
            query,
            uniquify=False,
            useChirality=True,
            maxMatches=10000,
        )
    }
    if occurrence not in matched_sets or len(occurrence) != query.GetNumAtoms():
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    boundary = []
    for atom_index in occurrence:
        atom = source.GetAtomWithIdx(atom_index)
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtomIdx(atom_index)
            if neighbor not in occurrence:
                boundary.append((atom_index, neighbor, bond.GetBondType()))
    if len(boundary) != 1 or boundary[0][1] != anchor_index:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    removed_bond_type = boundary[0][2]
    if (
        require_boundary_bond_type
        and removed_bond_type != _RDKIT_BOND_TYPES[action.bond_type]
    ):
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)

    editable = Chem.RWMol(source)
    for atom_index in sorted(occurrence, reverse=True):
        editable.RemoveAtom(atom_index)
    shifted_anchor = anchor_index - sum(
        atom_index < anchor_index for atom_index in occurrence
    )
    return editable.GetMol(), shifted_anchor, removed_bond_type


def replay_edit_action(
    request: CandidateRequest,
    action: EditAction,
) -> tuple[str, ...]:
    """Replay one typed graph action using the same strict T018 chemistry gate."""

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if type(action) is not EditAction:
        raise TypeError("action must be EditAction")
    if action.source_anchor_index is None or action.bond_type is None:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    source = _strict_replay_molecule(request.context.record.indexed_smiles)
    anchor_index, map_to_index = _source_anchor(
        source,
        action.source_anchor_index,
    )
    bond_units = _BOND_VALENCE_UNITS[action.bond_type]
    if action.edit_kind is EditKind.ADDITION:
        if action.add_fragment_smiles is None or action.fragment_attachment_atom is None:
            raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
        product = _combine_and_bond(
            source,
            anchor_index,
            action.add_fragment_smiles,
            action.fragment_attachment_atom,
            action.bond_type,
            consume_core_hydrogens=bond_units,
        )
    elif action.edit_kind is EditKind.DELETION:
        product, shifted_anchor, _ = _remove_exact_occurrence(
            source,
            anchor_index,
            map_to_index,
            action,
            require_boundary_bond_type=True,
        )
        editable = Chem.RWMol(product)
        _add_explicit_hydrogens(
            editable.GetAtomWithIdx(shifted_anchor),
            bond_units,
        )
        product = editable.GetMol()
    elif action.edit_kind is EditKind.SUBSTITUTION:
        if action.add_fragment_smiles is None or action.fragment_attachment_atom is None:
            raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
        core, shifted_anchor, removed_bond = _remove_exact_occurrence(
            source,
            anchor_index,
            map_to_index,
            action,
            require_boundary_bond_type=False,
        )
        try:
            removed_units = _RDKIT_BOND_VALENCE_UNITS[removed_bond]
        except KeyError as error:
            raise ValueError(
                CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value
            ) from error
        if removed_units > bond_units:
            editable = Chem.RWMol(core)
            _add_explicit_hydrogens(
                editable.GetAtomWithIdx(shifted_anchor),
                removed_units - bond_units,
            )
            core = editable.GetMol()
        product = _combine_and_bond(
            core,
            shifted_anchor,
            action.add_fragment_smiles,
            action.fragment_attachment_atom,
            action.bond_type,
            consume_core_hydrogens=max(0, bond_units - removed_units),
        )
    else:  # pragma: no cover - EditKind is sealed and exhaustive
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    return (_canonical_replay_product(product),)


def _replay_action_products(
    request: CandidateRequest,
    proposal: CandidateProposal,
) -> tuple[str, ...]:
    action = proposal.patch.edit_action
    if action is None:
        return ()
    if proposal.candidate_product_smiles is None:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    return replay_edit_action(request, action)


def _validate_action_product(
    request: CandidateRequest,
    proposal: CandidateProposal,
) -> None:
    action = proposal.patch.edit_action
    if action is None:
        return
    candidate_product = proposal.candidate_product_smiles
    if candidate_product is None:
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    if proposal.patch.root_node_id == "product":
        root_value = proposal.patch.new_value.normalized_value
        if type(root_value) is not str or not isomeric_graph_equivalent(
            root_value,
            candidate_product,
        ):
            raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)
    replayed = _replay_action_products(request, proposal)
    if not any(
        isomeric_graph_equivalent(candidate_product, product)
        for product in replayed
    ):
        raise ValueError(CandidateRejectCode.ACTION_PRODUCT_MISMATCH.value)


def _reference_equivalent(request: CandidateRequest, proposal: CandidateProposal) -> bool:
    if proposal.candidate_product_smiles is not None and isomeric_graph_equivalent(
        proposal.candidate_product_smiles,
        request.context.truth.canonical_gt_smiles,
    ):
        return True
    node = request.context.state_schema.nodes_by_id[proposal.patch.root_node_id]
    if node.comparator in {
        ComparatorKind.ISOMERIC_GRAPH_EQUIVALENCE,
        ComparatorKind.FRAGMENT_GRAPH_EQUIVALENCE,
    }:
        old = proposal.patch.old_value.normalized_value
        new = proposal.patch.new_value.normalized_value
        if type(old) is str and type(new) is str:
            return isomeric_graph_equivalent(old, new)
    if node.comparator is ComparatorKind.CASE_INSENSITIVE:
        old = proposal.patch.old_value.normalized_value
        new = proposal.patch.new_value.normalized_value
        if type(old) is str and type(new) is str:
            return old.casefold() == new.casefold()
    if node.comparator is ComparatorKind.FLOAT_TOLERANCE:
        old = proposal.patch.old_value.normalized_value
        new = proposal.patch.new_value.normalized_value
        if type(old) in {int, float} and type(new) in {int, float}:
            return math.isclose(float(old), float(new), rel_tol=1e-9, abs_tol=1e-9)
    return proposal.patch.old_value.semantically_equals(proposal.patch.new_value)


def _prevalidation_order_key(proposal: CandidateProposal) -> tuple[str, ...]:
    """Total order even for chemistry-invalid proposals; never uses arrival order."""

    try:
        semantic = canonical_candidate_key(proposal)
    except (KeyError, MoleculeParseError, RuntimeError, TypeError, ValueError):
        fallback = {
            "root": proposal.patch.root_node_id,
            "source": proposal.patch.source.value,
            "candidate_id": proposal.patch.candidate_id,
            "proposal_id": proposal.proposal_id,
            "product_length": (
                None
                if proposal.candidate_product_smiles is None
                else len(proposal.candidate_product_smiles)
            ),
        }
        semantic = hashlib.sha256(
            json.dumps(fallback, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    return (
        proposal.proposal_id,
        semantic,
        proposal.patch.source.value,
        proposal.patch.candidate_id,
    )


def _symmetry_equivalent(request: CandidateRequest, proposal: CandidateProposal) -> bool:
    if request.resolution.registration.operator_family != "wrong_anchor_site":
        return False
    candidate_anchor: int | None = None
    if proposal.patch.root_node_id == "anchor_idx":
        value = proposal.patch.new_value.normalized_value
        if type(value) is int:
            candidate_anchor = value
    elif proposal.patch.root_node_id == "product" and proposal.patch.edit_action is not None:
        candidate_anchor = proposal.patch.edit_action.source_anchor_index
    if candidate_anchor is None:
        return False
    truth = request.context.truth
    if candidate_anchor in truth.valid_anchor_indices:
        return True
    if any(
        candidate_anchor in group
        and any(anchor in group for anchor in truth.valid_anchor_indices)
        for group in truth.symmetry_equivalent_anchors
    ):
        return True

    # EditTruth records audited mapping-level equivalence.  Independently freeze
    # graph automorphism orbits from the map-free, stereochemistry-aware source
    # so an omitted group can never admit a chemically identical alternate site.
    with rdBase.BlockLogs():
        mapped = Chem.MolFromSmiles(request.context.record.indexed_smiles, sanitize=True)
    if mapped is None:
        return False
    map_indices: dict[int, list[int]] = {}
    for atom in mapped.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map > 0:
            map_indices.setdefault(atom_map, []).append(atom.GetIdx())
    if any(len(indices) != 1 for indices in map_indices.values()):
        raise ValueError(CandidateRejectCode.INVALID_PROPOSAL.value)
    atom_by_map = {atom_map: indices[0] for atom_map, indices in map_indices.items()}
    if candidate_anchor not in atom_by_map:
        raise ValueError(CandidateRejectCode.INVALID_PROPOSAL.value)
    present_valid_anchors = tuple(
        anchor for anchor in truth.valid_anchor_indices if anchor in atom_by_map
    )
    map_free = Chem.Mol(mapped)
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
    candidate_rank = ranks[atom_by_map[candidate_anchor]]
    return any(
        ranks[atom_by_map[valid_anchor]] == candidate_rank
        for valid_anchor in present_valid_anchors
    )


@dataclass(frozen=True, slots=True)
class DeterministicCandidateEngine:
    """Merge rule/RDKit proposals into one audited, deterministic pool."""

    sources: tuple[CandidateSource, ...]

    def __post_init__(self) -> None:
        if isinstance(self.sources, (str, bytes)):
            raise TypeError("sources must be a non-string collection")
        sources = tuple(self.sources)
        if not sources:
            raise ValueError("at least one candidate source is required")
        if any(not isinstance(source, CandidateSource) for source in sources):
            raise TypeError("sources must implement CandidateSource")
        allowed = {CandidateSourceType.RULE, CandidateSourceType.RDKIT}
        if any(source.source_type not in allowed for source in sources):
            raise ValueError("T018 accepts only RULE and RDKIT sources")
        if len({source.source_type for source in sources}) != len(sources):
            raise ValueError("candidate source types must be unique")
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(sources, key=lambda source: source.source_type.value)),
        )

    def build_pool(self, request: CandidateRequest) -> CandidateBuildResult:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        source_mode = request.resolution.candidate_source
        supported_modes = {
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
            CandidateSourceType.HYBRID,
        }
        if source_mode not in supported_modes:
            rejection = _reject(
                request,
                f"source:{source_mode.value}",
                CandidateRejectCode.UNSUPPORTED_SOURCE_MODE,
                source_mode=source_mode.value,
            )
            return CandidateBuildResult(
                pool=CandidatePool(
                    request_id=request.request_id,
                    candidates=(),
                    rejection_codes=(rejection.code.value,),
                ),
                rejections=(rejection,),
            )
        admitted_sources = {
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
        } if source_mode is CandidateSourceType.HYBRID else {source_mode}
        configured_sources = frozenset(source.source_type for source in self.sources)
        if not configured_sources.intersection(admitted_sources):
            rejection = _reject(
                request,
                f"source:{source_mode.value}",
                CandidateRejectCode.SOURCE_UNAVAILABLE,
                requested_source=source_mode.value,
                configured_sources=tuple(
                    sorted(source.value for source in configured_sources)
                ),
            )
            return CandidateBuildResult(
                pool=CandidatePool(
                    request_id=request.request_id,
                    candidates=(),
                    rejection_codes=(rejection.code.value,),
                ),
                rejections=(rejection,),
            )
        raw: list[CandidateProposal] = []
        rejections: list[CandidateRejection] = []
        seen_proposal_ids: set[str] = set()
        for source in self.sources:
            if source.source_type not in admitted_sources:
                continue
            try:
                proposed = tuple(
                    sorted(source.propose(request), key=_prevalidation_order_key)
                )
            except (CandidateSourceError, TypeError, ValueError) as error:
                rejections.append(
                    _reject(
                        request,
                        f"source:{source.source_type.value}",
                        CandidateRejectCode.SOURCE_FAILED,
                        source=source.source_type.value,
                        exception_type=type(error).__name__,
                    )
                )
                continue
            for proposal in proposed:
                if type(proposal) is not CandidateProposal:
                    rejections.append(
                        _reject(
                            request,
                            f"source:{source.source_type.value}",
                            CandidateRejectCode.INVALID_PROPOSAL,
                            source=source.source_type.value,
                        )
                    )
                    continue
                if proposal.proposal_id in seen_proposal_ids:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.DUPLICATE,
                            duplicate_scope="proposal_id",
                        )
                    )
                    continue
                seen_proposal_ids.add(proposal.proposal_id)
                patch = proposal.patch
                if patch.source is not source.source_type:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.SOURCE_MISMATCH,
                            expected=source.source_type.value,
                            actual=patch.source.value,
                        )
                    )
                    continue
                expected_provenance = {
                    CandidateSourceType.RULE: ValueProvenance.RULE,
                    CandidateSourceType.RDKIT: ValueProvenance.RDKIT,
                }[source.source_type]
                if patch.new_value.provenance is not expected_provenance:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.SOURCE_MISMATCH,
                            expected_provenance=expected_provenance.value,
                            actual_provenance=patch.new_value.provenance.value,
                        )
                    )
                    continue
                if patch.root_node_id != request.resolution.target_node_id:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.ROOT_MISMATCH,
                        )
                    )
                    continue
                reference = request.context.reference_graph.value_for(patch.root_node_id)
                if patch.old_value != reference:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.REFERENCE_VALUE_MISMATCH,
                        )
                    )
                    continue
                try:
                    _validate_attachment_semantics(request, proposal)
                    # Compare first: canonical replacement would violate
                    # CandidatePatch's non-equality invariant for alternate
                    # serializations of the reference molecule.
                    if _reference_equivalent(request, proposal):
                        rejections.append(
                            _reject(
                                request,
                                proposal.proposal_id,
                                CandidateRejectCode.REFERENCE_EQUIVALENT,
                            )
                        )
                        continue
                    canonical = _canonicalize_proposal(proposal)
                    _validate_action_product(request, canonical)
                    if _symmetry_equivalent(request, canonical):
                        rejections.append(
                            _reject(
                                request,
                                proposal.proposal_id,
                                CandidateRejectCode.SYMMETRY_EQUIVALENT,
                            )
                        )
                        continue
                except MoleculeParseError as error:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.SMILES_INVALID,
                            molecule_error_code=error.code.value,
                            input_length=error.input_length,
                        )
                    )
                    continue
                except ValueError as error:
                    code = (
                        CandidateRejectCode(error.args[0])
                        if error.args and error.args[0] in CandidateRejectCode._value2member_map_
                        else CandidateRejectCode.INVALID_PROPOSAL
                    )
                    evidence: dict[str, Any] = {}
                    if code is CandidateRejectCode.ACTION_PRODUCT_MISMATCH:
                        evidence = {
                            "phase": "action_product_replay",
                            "edit_kind": (
                                None
                                if proposal.patch.edit_action is None
                                else proposal.patch.edit_action.edit_kind.value
                            ),
                        }
                    rejections.append(
                        _reject(request, proposal.proposal_id, code, **evidence)
                    )
                    continue
                except (TypeError, KeyError, RuntimeError) as error:
                    rejections.append(
                        _reject(
                            request,
                            proposal.proposal_id,
                            CandidateRejectCode.INVALID_PROPOSAL,
                            exception_type=type(error).__name__,
                        )
                    )
                    continue
                raw.append(canonical)

        ranked_items: list[RankedCandidate] = []
        for proposal in raw:
            try:
                ranked_items.append(rank_candidates(request, (proposal,))[0])
            except (MoleculeParseError, TypeError, ValueError) as error:
                rejections.append(
                    _reject(
                        request,
                        proposal.proposal_id,
                        CandidateRejectCode.INVALID_PROPOSAL,
                        phase="difficulty_ranking",
                        exception_type=type(error).__name__,
                    )
                )
        ranked = tuple(sorted(ranked_items, key=lambda item: item.rank_key))

        accepted: list[RankedCandidate] = []
        seen_keys: dict[str, RankedCandidate] = {}
        seen_candidate_ids: dict[str, RankedCandidate] = {}
        for candidate in ranked:
            prior = seen_keys.get(candidate.canonical_key)
            if prior is not None:
                rejections.append(
                    _reject(
                        request,
                        candidate.proposal.proposal_id,
                        CandidateRejectCode.DUPLICATE,
                        duplicate_of=prior.proposal.proposal_id,
                        duplicate_scope="semantic",
                    )
                )
                continue
            prior_id = seen_candidate_ids.get(candidate.proposal.patch.candidate_id)
            if prior_id is not None:
                rejections.append(
                    _reject(
                        request,
                        candidate.proposal.proposal_id,
                        CandidateRejectCode.DUPLICATE,
                        duplicate_of=prior_id.proposal.proposal_id,
                        duplicate_scope="candidate_id",
                    )
                )
                continue
            seen_keys[candidate.canonical_key] = candidate
            seen_candidate_ids[candidate.proposal.patch.candidate_id] = candidate
            accepted.append(candidate)

        ordered_rejections = tuple(
            sorted(
                rejections,
                key=lambda item: (item.proposal_id, item.code.value),
            )
        )
        rejection_codes = tuple(sorted({item.code.value for item in ordered_rejections}))
        if not accepted and not rejection_codes:
            rejection_codes = (CandidateRejectCode.INVALID_PROPOSAL.value,)
        pool = CandidatePool(
            request_id=request.request_id,
            candidates=tuple(item.proposal.patch for item in accepted),
            rejection_codes=rejection_codes,
        )
        return CandidateBuildResult(
            pool=pool,
            rejections=ordered_rejections,
            ranked_candidates=tuple(accepted),
        )


__all__ = [
    "CandidateBuildResult",
    "CandidateDifficultyFeatures",
    "CandidateProposal",
    "CandidateRejectCode",
    "CandidateRejection",
    "CandidateRequest",
    "CandidateSource",
    "CandidateSourceError",
    "DeterministicCandidateEngine",
    "ProposalFunction",
    "RankedCandidate",
    "canonical_candidate_key",
    "compute_difficulty_features",
    "rank_candidates",
    "replay_edit_action",
]
