"""Deterministic, read-only chemistry tools for Poe proposal agents.

The schemas in :mod:`molhallulens.providers.poe.schemas` are the trust
boundary.  A dispatcher always validates an argument mapping against the
tool-specific frozen model before it resolves or invokes a handler.  Handlers
perform only local RDKit computations and the T018 replay path; they never
accept a caller-supplied product as evidence that an edit is chemically valid.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import rdkit
from pydantic import BaseModel
from rdkit import Chem, rdBase

from molhallulens.builders.edit_truth import EditTruthBuilder, EditTruthBuildError
from molhallulens.candidates import replay_edit_action_from_source
from molhallulens.chemistry import (
    FragmentPolicy,
    MoleculeParseError,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
    murcko_scaffold_smiles,
)
from molhallulens.domain import BondTypeName, EditAction, EditingSubtask, EditKind

from .schemas import (
    CHEMISTRY_TOOL_NAMES,
    AnalyzeSmilesArgs,
    CheckCandidateSignatureArgs,
    CompareMoleculesArgs,
    ComputeDescriptorsArgs,
    EnumerateAlternateAnchorsArgs,
    EnumerateRemovableGroupsArgs,
    FindGroupAtAnchorArgs,
    InspectAtomsArgs,
    SimulateEditArgs,
    parse_chemistry_tool_call,
    validate_chemistry_tool_arguments,
)

CHEMISTRY_TOOL_RESULT_VERSION = "molhallulens.chemistry_tool_result.v1"
CHEMISTRY_TOOL_CACHE_KEY_VERSION = "molhallulens.chemistry_tool_cache.v1"

_RDKIT_BOND_NAMES = MappingProxyType(
    {
        Chem.BondType.SINGLE: BondTypeName.SINGLE,
        Chem.BondType.DOUBLE: BondTypeName.DOUBLE,
        Chem.BondType.TRIPLE: BondTypeName.TRIPLE,
        Chem.BondType.AROMATIC: BondTypeName.AROMATIC,
    }
)
_FAMILY_EDIT_KINDS = MappingProxyType(
    {
        "add": EditKind.ADDITION,
        "delete": EditKind.DELETION,
        "substitute": EditKind.SUBSTITUTION,
    }
)
_FAMILY_SUBTASKS = MappingProxyType(
    {
        "add": EditingSubtask.ADD,
        "delete": EditingSubtask.DELETE,
        "substitute": EditingSubtask.SUBSTITUTE,
    }
)


class ChemistryToolRejected(ValueError):
    """Stable chemistry rejection without echoing untrusted molecule text."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("rejection code must be non-empty text")
        if type(message) is not str or not message:
            raise ValueError("rejection message must be non-empty text")
        self.code = code
        self.safe_message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ChemistryToolResult:
    """Frozen result whose canonical JSON bytes are the serialization contract."""

    tool: str
    cache_key: str
    _result_json: bytes
    result_version: str = CHEMISTRY_TOOL_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.tool not in CHEMISTRY_TOOL_NAMES:
            raise ValueError("result tool is not allow-listed")
        if (
            type(self.cache_key) is not str
            or len(self.cache_key) != 64
            or any(character not in "0123456789abcdef" for character in self.cache_key)
        ):
            raise ValueError("cache_key must be lowercase SHA256 hex")
        if type(self._result_json) is not bytes:
            raise TypeError("result payload must be canonical JSON bytes")
        decoded = json.loads(self._result_json)
        if type(decoded) is not dict:
            raise ValueError("result payload must be a JSON object")
        if self.result_version != CHEMISTRY_TOOL_RESULT_VERSION:
            raise ValueError("unsupported chemistry tool result version")

    @property
    def result(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the handler result."""

        return cast(dict[str, Any], json.loads(self._result_json))

    @property
    def payload(self) -> dict[str, Any]:
        """Compatibility alias for callers that call tool data a payload."""

        return self.result

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "result_version": self.result_version,
            "tool": self.tool,
            "cache_key": self.cache_key,
            "result": self.result,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_dict())


@dataclass(frozen=True, slots=True)
class _RemovableGroup:
    anchor_map: int
    occurrence_atom_maps: tuple[int, ...]
    fragment_smiles: str
    heavy_atom_count: int
    bond_type: BondTypeName
    fragment_attachment_atom_map: int

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.fragment_smiles,
            self.occurrence_atom_maps,
            self.anchor_map,
            self.bond_type.value,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "anchor_idx": self.anchor_map,
            "occurrence_atom_maps": list(self.occurrence_atom_maps),
            "fragment_smiles": self.fragment_smiles,
            "heavy_atom_count": self.heavy_atom_count,
            "bond_type": self.bond_type.value,
            "fragment_attachment_atom_map": self.fragment_attachment_atom_map,
        }


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(
            "chemistry tool result must contain finite JSON values"
        ) from error


def _cache_key(tool: str, arguments: BaseModel) -> str:
    normalized = arguments.model_dump(mode="json")
    if tool == "inspect_atoms":
        normalized["atom_indices"] = sorted(normalized["atom_indices"])
    identity = {
        "cache_key_version": CHEMISTRY_TOOL_CACHE_KEY_VERSION,
        "rdkit_version": rdkit.__version__,
        "tool": tool,
        "arguments": normalized,
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _reject(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "reject_reasons": [{"code": code, "message": message}],
    }


def _simulate_reject(family: str, code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "family": family,
        "products": [],
        "reject_reasons": [{"code": code, "message": message}],
    }


def _parse_molecule(smiles: str) -> Chem.Mol:
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except (RuntimeError, ValueError) as error:
        raise ChemistryToolRejected(
            "invalid_smiles", "SMILES failed strict RDKit parsing and sanitization"
        ) from error
    if molecule is None:
        raise ChemistryToolRejected(
            "invalid_smiles", "SMILES failed strict RDKit parsing and sanitization"
        )
    return molecule


def _mapped_source(smiles: str) -> tuple[Chem.Mol, dict[int, int]]:
    molecule = _parse_molecule(smiles)
    map_to_index: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0:
            raise ChemistryToolRejected(
                "source_atom_maps_required",
                "source heavy atoms require unique positive atom-map identifiers",
            )
        if atom_map in map_to_index:
            raise ChemistryToolRejected(
                "duplicate_source_atom_map",
                "source heavy-atom map identifiers must be unique",
            )
        map_to_index[atom_map] = atom.GetIdx()
    if not map_to_index:
        raise ChemistryToolRejected(
            "source_atom_maps_required", "source has no mapped heavy atoms"
        )
    return molecule, map_to_index


def _anchor_index(map_to_index: Mapping[int, int], atom_map: int) -> int:
    try:
        return map_to_index[atom_map]
    except KeyError as error:
        raise ChemistryToolRejected(
            "unknown_anchor", "anchor_idx is not a source atom-map identifier"
        ) from error


def _canonical_indexed_smiles(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _component_without_bond(
    molecule: Chem.Mol,
    *,
    start_atom: int,
    blocked_bond: int,
) -> frozenset[int]:
    visited: set[int] = set()
    frontier = deque((start_atom,))
    while frontier:
        atom_index = frontier.popleft()
        if atom_index in visited:
            continue
        visited.add(atom_index)
        for bond in molecule.GetAtomWithIdx(atom_index).GetBonds():
            if bond.GetIdx() == blocked_bond:
                continue
            neighbor = bond.GetOtherAtomIdx(atom_index)
            if neighbor not in visited:
                frontier.append(neighbor)
    return frozenset(visited)


def _fragment_smiles(molecule: Chem.Mol, atom_indices: frozenset[int]) -> str:
    map_free = Chem.Mol(molecule)
    for atom in map_free.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        with rdBase.BlockLogs():
            raw = Chem.MolFragmentToSmiles(
                map_free,
                atomsToUse=sorted(atom_indices),
                canonical=True,
                isomericSmiles=True,
            )
        if not raw or "." in raw:
            raise ValueError("fragment was disconnected")
        result = canonicalize_smiles(raw, fragment_policy=FragmentPolicy.KEEP_ALL)
    except (MoleculeParseError, RuntimeError, TypeError, ValueError) as error:
        raise ChemistryToolRejected(
            "fragment_canonicalization_failed",
            "a source group could not be represented as a sanitized fragment",
        ) from error
    return result


def _enumerate_groups(
    source_smiles: str,
    anchor_map: int,
    *,
    max_heavy_atoms: int,
    max_results: int | None,
) -> tuple[_RemovableGroup, ...]:
    molecule, map_to_index = _mapped_source(source_smiles)
    anchor_index = _anchor_index(map_to_index, anchor_map)
    all_indices = frozenset(range(molecule.GetNumAtoms()))
    groups: dict[tuple[Any, ...], _RemovableGroup] = {}
    for bond in molecule.GetAtomWithIdx(anchor_index).GetBonds():
        neighbor_index = bond.GetOtherAtomIdx(anchor_index)
        occurrence = _component_without_bond(
            molecule,
            start_atom=neighbor_index,
            blocked_bond=bond.GetIdx(),
        )
        if anchor_index in occurrence or not occurrence or occurrence == all_indices:
            continue
        boundary = tuple(
            candidate_bond
            for atom_index in occurrence
            for candidate_bond in molecule.GetAtomWithIdx(atom_index).GetBonds()
            if candidate_bond.GetOtherAtomIdx(atom_index) not in occurrence
        )
        if len(boundary) != 1 or boundary[0].GetIdx() != bond.GetIdx():
            continue
        heavy_atom_count = sum(
            molecule.GetAtomWithIdx(index).GetAtomicNum() > 1 for index in occurrence
        )
        if heavy_atom_count == 0 or heavy_atom_count > max_heavy_atoms:
            continue
        occurrence_maps = tuple(
            sorted(
                molecule.GetAtomWithIdx(index).GetAtomMapNum() for index in occurrence
            )
        )
        if any(atom_map <= 0 for atom_map in occurrence_maps):
            continue
        bond_type = _RDKIT_BOND_NAMES.get(bond.GetBondType())
        if bond_type is None:
            continue
        group = _RemovableGroup(
            anchor_map=anchor_map,
            occurrence_atom_maps=occurrence_maps,
            fragment_smiles=_fragment_smiles(molecule, occurrence),
            heavy_atom_count=heavy_atom_count,
            bond_type=bond_type,
            fragment_attachment_atom_map=(
                molecule.GetAtomWithIdx(neighbor_index).GetAtomMapNum()
            ),
        )
        groups[group.sort_key] = group
    ordered = tuple(groups[key] for key in sorted(groups))
    return ordered if max_results is None else ordered[:max_results]


def _descriptors_json(smiles: str, policy: FragmentPolicy) -> dict[str, Any]:
    descriptors = compute_descriptors(smiles, fragment_policy=policy)
    return {
        "canonical_smiles": descriptors.canonical_smiles,
        "fragment_policy": descriptors.fragment_policy.value,
        "fragment_count": descriptors.fragment_count,
        "heavy_atom_count": descriptors.heavy_atom_count,
        "ring_count": descriptors.ring_count,
        "aromatic_ring_count": descriptors.aromatic_ring_count,
        "formal_charge": descriptors.formal_charge,
        "heteroatom_counts": [list(item) for item in descriptors.heteroatom_counts],
        "molecular_weight": descriptors.molecular_weight,
        "exact_molecular_weight": descriptors.exact_molecular_weight,
        "rotatable_bond_count": descriptors.rotatable_bond_count,
        "hydrogen_bond_donor_count": descriptors.hydrogen_bond_donor_count,
        "hydrogen_bond_acceptor_count": descriptors.hydrogen_bond_acceptor_count,
        "topological_polar_surface_area": (descriptors.topological_polar_surface_area),
        "log_p": descriptors.log_p,
    }


def _reference_json(reference: Any) -> dict[str, Any]:
    return {
        "namespace": reference.namespace.value,
        "atom_id": reference.atom_id,
    }


def _bond_edit_json(edit: Any) -> dict[str, Any]:
    return {
        "begin": _reference_json(edit.begin),
        "end": _reference_json(edit.end),
        "bond_type": edit.bond_type.value,
        "stereo": edit.stereo,
        "aromatic": edit.aromatic,
    }


def _graph_diff(
    *,
    family: str,
    source_smiles: str,
    product_smiles: str,
    remove_hint: str | None,
    add_hint: str | None,
) -> dict[str, Any]:
    try:
        truth = EditTruthBuilder().derive(
            source_smiles,
            product_smiles,
            anonymous_sample_id="chemistry_tool_simulation",
            normalized_subtask=_FAMILY_SUBTASKS[family],
            remove_fragment_hint=remove_hint,
            add_fragment_hint=add_hint,
        )
    except EditTruthBuildError as error:
        raise ChemistryToolRejected(
            "graph_diff_derivation_failed",
            "replayed product failed deterministic graph-difference derivation",
        ) from error
    return {
        "canonical_source_smiles": truth.canonical_source_smiles,
        "canonical_product_smiles": truth.canonical_gt_smiles,
        "valid_anchor_indices": list(truth.valid_anchor_indices),
        "removed_atom_maps": sorted(truth.removed_atom_maps),
        "added_atoms": [
            {
                "reference": _reference_json(atom.reference),
                "atomic_number": atom.atomic_number,
                "element": atom.element,
                "isotope": atom.isotope,
                "formal_charge": atom.formal_charge,
                "aromatic": atom.aromatic,
                "chiral_tag": atom.chiral_tag,
            }
            for atom in truth.added_atoms
        ],
        "broken_bonds": [_bond_edit_json(item) for item in truth.broken_bonds],
        "formed_bonds": [_bond_edit_json(item) for item in truth.formed_bonds],
        "remove_fragment_smiles": (
            None
            if truth.remove_fragment is None
            else truth.remove_fragment.canonical_smiles
        ),
        "add_fragment_smiles": (
            None if truth.add_fragment is None else truth.add_fragment.canonical_smiles
        ),
        "heavy_atom_delta": truth.heavy_atom_delta,
        "mapping_algorithm": truth.mapping_evidence.algorithm,
        "mapping_confidence": truth.mapping_confidence,
    }


def _action_json(action: EditAction) -> dict[str, Any]:
    occurrence = action.metadata.get("occurrence_atom_maps", ())
    return {
        "edit_kind": action.edit_kind.value,
        "source_anchor_index": action.source_anchor_index,
        "remove_anchor_index": action.remove_anchor_index,
        "remove_fragment_smiles": action.remove_fragment_smiles,
        "add_fragment_smiles": action.add_fragment_smiles,
        "fragment_attachment_atom": action.fragment_attachment_atom,
        "bond_type": None if action.bond_type is None else action.bond_type.value,
        "occurrence_atom_maps": list(occurrence),
    }


def _actions_for_simulation(arguments: SimulateEditArgs) -> tuple[EditAction, ...]:
    edit_kind = _FAMILY_EDIT_KINDS[arguments.family]
    if arguments.family == "add":
        return (
            EditAction(
                edit_kind=edit_kind,
                source_anchor_index=arguments.anchor_idx,
                add_fragment_smiles=arguments.add_fragment_smiles,
                fragment_attachment_atom=arguments.fragment_attachment_atom,
                bond_type=BondTypeName(arguments.bond_type),
            ),
        )

    remove_anchor = arguments.remove_anchor_idx or arguments.anchor_idx
    assert arguments.remove_group_smiles is not None
    canonical_remove = canonicalize_smiles(
        arguments.remove_group_smiles,
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    occurrences = tuple(
        group
        for group in _enumerate_groups(
            arguments.source_smiles,
            remove_anchor,
            max_heavy_atoms=128,
            max_results=None,
        )
        if fragment_graph_equivalent(group.fragment_smiles, canonical_remove)
    )
    if not occurrences:
        raise ChemistryToolRejected(
            "remove_group_not_found",
            "remove group is not an induced connected one-boundary source occurrence",
        )
    actions = []
    for occurrence in occurrences:
        if arguments.family == "delete":
            action = EditAction(
                edit_kind=edit_kind,
                source_anchor_index=arguments.anchor_idx,
                remove_fragment_smiles=canonical_remove,
                bond_type=occurrence.bond_type,
                metadata={"occurrence_atom_maps": occurrence.occurrence_atom_maps},
            )
        else:
            action = EditAction(
                edit_kind=edit_kind,
                source_anchor_index=arguments.anchor_idx,
                remove_anchor_index=(
                    None if remove_anchor == arguments.anchor_idx else remove_anchor
                ),
                remove_fragment_smiles=canonical_remove,
                add_fragment_smiles=arguments.add_fragment_smiles,
                fragment_attachment_atom=arguments.fragment_attachment_atom,
                bond_type=BondTypeName(arguments.bond_type),
                metadata={"occurrence_atom_maps": occurrence.occurrence_atom_maps},
            )
        actions.append(action)
    return tuple(actions)


def _simulate_edit(arguments: SimulateEditArgs) -> dict[str, Any]:
    try:
        # Fail early on malformed/unmapped source and an unknown add anchor.
        _, map_to_index = _mapped_source(arguments.source_smiles)
        _anchor_index(map_to_index, arguments.anchor_idx)
        actions = _actions_for_simulation(arguments)
    except (MoleculeParseError, ChemistryToolRejected) as error:
        if isinstance(error, ChemistryToolRejected):
            return _simulate_reject(arguments.family, error.code, error.safe_message)
        return _simulate_reject(
            arguments.family, "invalid_smiles", "SMILES failed strict parsing"
        )

    products: dict[tuple[Any, ...], dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for action in actions:
        try:
            replayed = replay_edit_action_from_source(arguments.source_smiles, action)
            for product_smiles in replayed:
                canonical_product = canonicalize_smiles(
                    product_smiles,
                    fragment_policy=FragmentPolicy.KEEP_ALL,
                )
                graph_diff = _graph_diff(
                    family=arguments.family,
                    source_smiles=arguments.source_smiles,
                    product_smiles=canonical_product,
                    remove_hint=arguments.remove_group_smiles,
                    add_hint=arguments.add_fragment_smiles,
                )
                item = {
                    "product_smiles": canonical_product,
                    "action": _action_json(action),
                    "graph_diff": graph_diff,
                    "descriptors": _descriptors_json(
                        canonical_product, FragmentPolicy.KEEP_ALL
                    ),
                }
                key = (
                    canonical_product,
                    tuple(item["action"]["occurrence_atom_maps"]),
                )
                products[key] = item
        except ChemistryToolRejected as error:
            rejected.append({"code": error.code, "message": error.safe_message})
        except (MoleculeParseError, RuntimeError, TypeError, ValueError):
            rejected.append(
                {
                    "code": "action_product_mismatch",
                    "message": "typed edit failed strict T018 replay or sanitization",
                }
            )
    ordered_products = [products[key] for key in sorted(products)]
    reject_reasons = sorted(
        {(_reason["code"], _reason["message"]) for _reason in rejected}
    )
    if not ordered_products and not reject_reasons:
        reject_reasons = [("no_products", "edit produced no sanitized products")]
    return {
        "ok": bool(ordered_products),
        "family": arguments.family,
        "products": ordered_products,
        "reject_reasons": [
            {"code": code, "message": message} for code, message in reject_reasons
        ],
    }


def _inspect_atoms(arguments: InspectAtomsArgs) -> dict[str, Any]:
    molecule = _parse_molecule(arguments.smiles)
    selected = (
        tuple(sorted(arguments.atom_indices))
        if arguments.atom_indices
        else tuple(range(molecule.GetNumAtoms()))
    )
    if any(index >= molecule.GetNumAtoms() for index in selected):
        raise ChemistryToolRejected(
            "atom_index_out_of_range", "atom_indices contains an out-of-range index"
        )
    atoms = []
    for index in selected:
        atom = molecule.GetAtomWithIdx(index)
        neighbors = []
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            neighbors.append(
                {
                    "atom_index": neighbor.GetIdx(),
                    "atom_map": neighbor.GetAtomMapNum() or None,
                    "bond_type": str(bond.GetBondType()),
                    "aromatic": bond.GetIsAromatic(),
                }
            )
        atoms.append(
            {
                "atom_index": index,
                "atom_map": atom.GetAtomMapNum() or None,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "isotope": atom.GetIsotope(),
                "formal_charge": atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(),
                "degree": atom.GetDegree(),
                "total_hydrogens": atom.GetTotalNumHs(),
                "hybridization": str(atom.GetHybridization()),
                "chiral_tag": str(atom.GetChiralTag()),
                "neighbors": sorted(neighbors, key=lambda item: item["atom_index"]),
            }
        )
    return {
        "ok": True,
        "canonical_smiles": canonicalize_smiles(arguments.smiles),
        "indexed_smiles": _canonical_indexed_smiles(molecule),
        "atom_count": molecule.GetNumAtoms(),
        "atoms": atoms,
        "reject_reasons": [],
    }


def _enumerate_alternate_anchors(
    arguments: EnumerateAlternateAnchorsArgs,
) -> dict[str, Any]:
    molecule, map_to_index = _mapped_source(arguments.source_smiles)
    reference_index = _anchor_index(map_to_index, arguments.reference_anchor_idx)
    reference = molecule.GetAtomWithIdx(reference_index)
    anchors = []
    for atom_map, index in sorted(map_to_index.items()):
        if atom_map == arguments.reference_anchor_idx:
            continue
        atom = molecule.GetAtomWithIdx(index)
        if (
            arguments.same_element_only
            and atom.GetAtomicNum() != reference.GetAtomicNum()
        ):
            continue
        anchors.append(
            {
                "anchor_idx": atom_map,
                "atom_index": index,
                "element": atom.GetSymbol(),
                "aromatic": atom.GetIsAromatic(),
                "degree": atom.GetDegree(),
                "hybridization": str(atom.GetHybridization()),
                "graph_distance": len(
                    Chem.GetShortestPath(molecule, reference_index, index)
                )
                - 1,
            }
        )
    return {
        "ok": True,
        "reference_anchor_idx": arguments.reference_anchor_idx,
        "reference_element": reference.GetSymbol(),
        "anchors": anchors[: arguments.max_results],
        "reject_reasons": [],
    }


def _analyze_smiles(arguments: AnalyzeSmilesArgs) -> dict[str, Any]:
    policy = FragmentPolicy(arguments.fragment_policy)
    molecule = _parse_molecule(arguments.smiles)
    return {
        "ok": True,
        "descriptors": _descriptors_json(arguments.smiles, policy),
        "murcko_scaffold_smiles": murcko_scaffold_smiles(
            arguments.smiles, fragment_policy=policy
        ),
        "input_atom_count": molecule.GetNumAtoms(),
        "input_fragment_count": len(Chem.GetMolFrags(molecule)),
        "reject_reasons": [],
    }


def _find_group_at_anchor(arguments: FindGroupAtAnchorArgs) -> dict[str, Any]:
    groups = _enumerate_groups(
        arguments.source_smiles,
        arguments.anchor_idx,
        max_heavy_atoms=arguments.max_heavy_atoms,
        max_results=None,
    )
    return {
        "ok": True,
        "anchor_idx": arguments.anchor_idx,
        "groups": [group.to_json_dict() for group in groups],
        "reject_reasons": [],
    }


def _enumerate_removable_groups(
    arguments: EnumerateRemovableGroupsArgs,
) -> dict[str, Any]:
    groups = _enumerate_groups(
        arguments.source_smiles,
        arguments.anchor_idx,
        max_heavy_atoms=arguments.max_group_heavy_atoms,
        max_results=arguments.max_results,
    )
    return {
        "ok": True,
        "anchor_idx": arguments.anchor_idx,
        "groups": [group.to_json_dict() for group in groups],
        "reject_reasons": [],
    }


def _compute_descriptors(arguments: ComputeDescriptorsArgs) -> dict[str, Any]:
    return {
        "ok": True,
        "descriptors": _descriptors_json(
            arguments.smiles, FragmentPolicy(arguments.fragment_policy)
        ),
        "reject_reasons": [],
    }


def _compare_molecules(arguments: CompareMoleculesArgs) -> dict[str, Any]:
    left = canonicalize_smiles(arguments.left_smiles)
    right = canonicalize_smiles(arguments.right_smiles)
    if arguments.comparator == "exact":
        equivalent = arguments.left_smiles == arguments.right_smiles
    elif arguments.comparator == "fragment_graph_equivalence":
        equivalent = fragment_graph_equivalent(
            arguments.left_smiles, arguments.right_smiles
        )
    else:
        equivalent = isomeric_graph_equivalent(
            arguments.left_smiles, arguments.right_smiles
        )
    return {
        "ok": True,
        "comparator": arguments.comparator,
        "equivalent": equivalent,
        "canonical_left_smiles": left,
        "canonical_right_smiles": right,
        "reject_reasons": [],
    }


def _check_candidate_signature(
    arguments: CheckCandidateSignatureArgs,
) -> dict[str, Any]:
    try:
        candidate = canonicalize_smiles(arguments.candidate_product_smiles)
    except MoleculeParseError:
        return {
            "ok": True,
            "valid": False,
            "matched_product_smiles": None,
            "graph_diff": None,
            "reject_reasons": [
                {
                    "code": "candidate_invalid_smiles",
                    "message": "candidate product failed strict parsing and sanitization",
                }
            ],
        }
    simulation_args = SimulateEditArgs.model_validate(
        {
            key: value
            for key, value in arguments.model_dump(mode="python").items()
            if key != "candidate_product_smiles"
        },
        strict=True,
    )
    simulation = _simulate_edit(simulation_args)
    matched = next(
        (
            product
            for product in simulation.get("products", [])
            if isomeric_graph_equivalent(candidate, product["product_smiles"])
        ),
        None,
    )
    if matched is None:
        reasons = list(simulation.get("reject_reasons", []))
        reasons.append(
            {
                "code": "candidate_not_replay_product",
                "message": "candidate does not match any strictly replayed product",
            }
        )
        return {
            "ok": True,
            "valid": False,
            "matched_product_smiles": None,
            "graph_diff": None,
            "replayed_product_smiles": [
                product["product_smiles"] for product in simulation.get("products", [])
            ],
            "reject_reasons": reasons,
        }
    return {
        "ok": True,
        "valid": True,
        "matched_product_smiles": matched["product_smiles"],
        "graph_diff": matched["graph_diff"],
        "replayed_product_smiles": [
            product["product_smiles"] for product in simulation["products"]
        ],
        "reject_reasons": [],
    }


_Handler = Callable[[Any], dict[str, Any]]
CHEMISTRY_TOOL_HANDLERS: Mapping[str, _Handler] = MappingProxyType(
    {
        "inspect_atoms": _inspect_atoms,
        "enumerate_alternate_anchors": _enumerate_alternate_anchors,
        "analyze_smiles": _analyze_smiles,
        "find_group_at_anchor": _find_group_at_anchor,
        "enumerate_removable_groups": _enumerate_removable_groups,
        "simulate_edit": _simulate_edit,
        "compute_descriptors": _compute_descriptors,
        "compare_molecules": _compare_molecules,
        "check_candidate_signature": _check_candidate_signature,
    }
)
if tuple(CHEMISTRY_TOOL_HANDLERS) != CHEMISTRY_TOOL_NAMES:  # pragma: no cover
    raise RuntimeError("chemistry tool schema and handler allow-lists differ")


class ChemistryTools:
    """Read-only validated dispatcher with a deterministic in-memory cache."""

    def __init__(self) -> None:
        self._cache: dict[str, ChemistryToolResult] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def dispatch(
        self,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> ChemistryToolResult:
        # This must remain the first semantic operation.  In particular, handler
        # lookup occurs only after the exact tool name and all args are parsed.
        validated = validate_chemistry_tool_arguments(tool, arguments)
        cache_key = _cache_key(tool, validated)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        handler = CHEMISTRY_TOOL_HANDLERS[tool]
        try:
            payload = handler(validated)
        except ChemistryToolRejected as error:
            payload = _reject(error.code, error.safe_message)
        except MoleculeParseError:
            payload = _reject(
                "invalid_smiles", "SMILES failed strict parsing and sanitization"
            )
        payload_json = _canonical_json(payload)
        result = ChemistryToolResult(
            tool=tool,
            cache_key=cache_key,
            _result_json=payload_json,
        )
        self._cache[cache_key] = result
        return result

    def dispatch_call(
        self,
        payload: str | bytes | Mapping[str, Any],
    ) -> ChemistryToolResult:
        parsed = parse_chemistry_tool_call(payload)
        return self.dispatch(
            parsed.tool,
            parsed.arguments.model_dump(mode="python"),
        )


_DEFAULT_CHEMISTRY_TOOLS = ChemistryTools()


def dispatch_chemistry_tool(
    tool: str,
    arguments: Mapping[str, Any],
) -> ChemistryToolResult:
    """Dispatch through the process-local read-only tool instance."""

    return _DEFAULT_CHEMISTRY_TOOLS.dispatch(tool, arguments)


def dispatch_chemistry_tool_call(
    payload: str | bytes | Mapping[str, Any],
) -> ChemistryToolResult:
    """Parse a strict tool envelope, then dispatch its validated arguments."""

    return _DEFAULT_CHEMISTRY_TOOLS.dispatch_call(payload)


__all__ = [
    "CHEMISTRY_TOOL_CACHE_KEY_VERSION",
    "CHEMISTRY_TOOL_HANDLERS",
    "CHEMISTRY_TOOL_RESULT_VERSION",
    "ChemistryToolRejected",
    "ChemistryToolResult",
    "ChemistryTools",
    "dispatch_chemistry_tool",
    "dispatch_chemistry_tool_call",
]
