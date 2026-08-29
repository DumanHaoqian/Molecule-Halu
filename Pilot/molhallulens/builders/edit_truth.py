"""Deterministic, fail-closed graph-difference derivation for molecule edits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS

from molhallulens.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
)
from molhallulens.domain.edit_truth import (
    AtomDescriptor,
    AtomMapping,
    AtomMappingPair,
    AtomReference,
    AtomReferenceNamespace,
    BondEdit,
    EditTruth,
    FragmentSpec,
    MappingEvidence,
)
from molhallulens.domain.enums import (
    BondTypeName,
    EditingSubtask,
    Severity,
    ValidationStage,
)
from molhallulens.domain.errors import ValidationIssue, ValidationReport

from .reference_dag import ReferenceDAGArtifact


_MAX_MATCHES = 20_000
_DEFAULT_MCS_TIMEOUT_SECONDS = 30
_NONE_FRAGMENT_VALUES = frozenset({"", "none", "null", "nil", "n/a"})
_BOND_TYPES = {
    Chem.BondType.SINGLE: BondTypeName.SINGLE,
    Chem.BondType.DOUBLE: BondTypeName.DOUBLE,
    Chem.BondType.TRIPLE: BondTypeName.TRIPLE,
    Chem.BondType.AROMATIC: BondTypeName.AROMATIC,
}


class EditTruthBuildError(RuntimeError):
    """A structured graph-edit derivation failure."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("EditTruthBuildError report must be a ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"EditTruth derivation failed ({codes})")


@dataclass(frozen=True, slots=True)
class _Candidate:
    mapping: tuple[tuple[int, int], ...]
    domain_mapping: AtomMapping
    removed_source_indices: frozenset[int]
    added_product_indices: frozenset[int]
    removed_atom_maps: frozenset[int]
    added_atoms: tuple[AtomDescriptor, ...]
    broken_bonds: tuple[BondEdit, ...]
    formed_bonds: tuple[BondEdit, ...]
    remove_fragment: FragmentSpec | None
    add_fragment: FragmentSpec | None
    anchors: frozenset[int]
    hint_score: tuple[int, int, int]
    signature: tuple[Any, ...]


def _reference_key(reference: AtomReference) -> tuple[str, int]:
    return (reference.namespace.value, reference.atom_id)


def _bond_key(bond: BondEdit) -> tuple[Any, ...]:
    return (
        _reference_key(bond.begin),
        _reference_key(bond.end),
        bond.bond_type.value,
        bond.stereo,
        bond.aromatic,
    )


def _normal_fragment_hint(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("fragment hints must be strings or None")
    stripped = value.strip()
    return None if stripped.lower() in _NONE_FRAGMENT_VALUES else stripped


def _claim_value(artifact: ReferenceDAGArtifact, node_id: str) -> Any:
    return artifact.state_dag.values[node_id].normalized_value


class EditTruthBuilder:
    """Derive an immutable EditTruth without trusting a single RDKit embedding."""

    def __init__(
        self,
        *,
        mcs_timeout_seconds: int = _DEFAULT_MCS_TIMEOUT_SECONDS,
        max_matches: int = _MAX_MATCHES,
    ) -> None:
        if type(mcs_timeout_seconds) is not int or mcs_timeout_seconds <= 0:
            raise ValueError("mcs_timeout_seconds must be a positive integer")
        if type(max_matches) is not int or max_matches <= 1:
            raise ValueError("max_matches must be an integer greater than one")
        self.mcs_timeout_seconds = mcs_timeout_seconds
        self.max_matches = max_matches

    def _raise(
        self,
        anonymous_sample_id: str,
        code: str,
        message: str,
        **evidence: Any,
    ) -> None:
        safe_evidence = {
            key: value for key, value in evidence.items() if value is not None
        }
        raise EditTruthBuildError(
            ValidationReport(
                "molhallulens.edit_truth.v1",
                (
                    ValidationIssue(
                        code=code,
                        severity=Severity.FATAL,
                        stage=ValidationStage.GRAPH_EDIT,
                        node_ids=(anonymous_sample_id,),
                        message=message,
                        evidence=safe_evidence,
                    ),
                ),
            )
        )

    def build(self, artifact: ReferenceDAGArtifact) -> EditTruth:
        """Derive EditTruth from one validated T011 reference-DAG artifact."""

        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("EditTruthBuilder.build requires a ReferenceDAGArtifact")
        values = artifact.state_dag.values
        remove_hint: str | None = None
        add_hint: str | None = None
        if artifact.normalized_subtask is EditingSubtask.ADD:
            add_hint = values["add_fragment"].normalized_value
        elif artifact.normalized_subtask is EditingSubtask.DELETE:
            remove_hint = values["remove_group_step2"].normalized_value
        else:
            remove_hint = values["remove_group"].normalized_value
            add_hint = values["add_fragment"].normalized_value
        return self.derive(
            _claim_value(artifact, "source"),
            _claim_value(artifact, "oracle_gt"),
            anonymous_sample_id=artifact.anonymous_sample_id,
            normalized_subtask=artifact.normalized_subtask,
            trace_anchor_indices=(_claim_value(artifact, "anchor_idx"),),
            remove_fragment_hint=remove_hint,
            add_fragment_hint=add_hint,
        )

    def derive(
        self,
        source_smiles: str,
        gt_smiles: str,
        *,
        anonymous_sample_id: str,
        normalized_subtask: EditingSubtask,
        trace_anchor_indices: Iterable[int] = (),
        remove_fragment_hint: str | None = None,
        add_fragment_hint: str | None = None,
    ) -> EditTruth:
        """Core graph API, also suitable for compact synthetic fixtures."""

        if type(anonymous_sample_id) is not str or not anonymous_sample_id.strip():
            raise ValueError("anonymous_sample_id must be non-empty text")
        if type(normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be an EditingSubtask")
        if type(source_smiles) is not str or type(gt_smiles) is not str:
            raise TypeError("source_smiles and gt_smiles must be strings")
        trace_anchors = tuple(trace_anchor_indices)
        if any(type(value) is not int or value <= 0 for value in trace_anchors):
            raise ValueError("trace_anchor_indices must contain positive integers")
        if len(trace_anchors) != len(set(trace_anchors)):
            raise ValueError("trace_anchor_indices must be unique")
        trace_anchors = tuple(sorted(trace_anchors))
        remove_hint = _normal_fragment_hint(remove_fragment_hint)
        add_hint = _normal_fragment_hint(add_fragment_hint)

        try:
            source = self._parse(source_smiles)
            product = self._parse(gt_smiles)
            canonical_source = canonicalize_smiles(source_smiles)
            canonical_product = canonicalize_smiles(gt_smiles)
        except (TypeError, ValueError, RuntimeError) as error:
            self._raise(
                anonymous_sample_id,
                "MOLECULE_PARSE",
                "source or product failed strict molecular parsing",
                error_type=type(error).__name__,
                source_length=len(source_smiles),
                product_length=len(gt_smiles),
            )

        source_maps = self._source_maps(source, anonymous_sample_id)
        if any(anchor not in source_maps.values() for anchor in trace_anchors):
            self._raise(
                anonymous_sample_id,
                "TRACE_ANCHOR_UNKNOWN",
                "a trace anchor is not a source atom-map identifier",
                trace_anchor_indices=trace_anchors,
            )
        source_clean = Chem.Mol(source)
        for atom in source_clean.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(source_clean, cleanIt=True, force=True)
        product_clean = Chem.Mol(product)
        for atom in product_clean.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(product_clean, cleanIt=True, force=True)

        product_ids = self._canonical_product_ids(
            product_clean, anonymous_sample_id
        )
        raw_mappings, algorithm, mcs_smarts = self._choose_mappings(
            source_clean,
            product_clean,
            normalized_subtask=normalized_subtask,
            remove_hint=remove_hint,
            anonymous_sample_id=anonymous_sample_id,
        )
        candidates = self._analyze_candidates(
            raw_mappings,
            source_clean,
            product_clean,
            source_maps,
            product_ids,
            trace_anchors=trace_anchors,
            remove_hint=remove_hint,
            add_hint=add_hint,
            anonymous_sample_id=anonymous_sample_id,
        )
        best_score = max(candidate.hint_score for candidate in candidates)
        candidates = tuple(
            candidate for candidate in candidates if candidate.hint_score == best_score
        )
        signature_groups: dict[tuple[Any, ...], list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            signature_groups[candidate.signature].append(candidate)
        if len(signature_groups) > 1 and trace_anchors:
            source_symmetry = self._source_symmetry_by_map(source_clean, source_maps)

            def signature_agrees_with_trace(group: list[_Candidate]) -> bool:
                graph_anchor_classes = {
                    source_symmetry[anchor]
                    for candidate in group
                    for anchor in candidate.anchors
                }
                return all(
                    any(trace_anchor in group_members for group_members in graph_anchor_classes)
                    for trace_anchor in trace_anchors
                )

            trace_agreeing_groups = {
                signature: group
                for signature, group in signature_groups.items()
                if signature_agrees_with_trace(group)
            }
            # A trace may select exactly one already graph-derived signature; it
            # can never invent an anchor or suppress graph truth when it agrees
            # with zero/multiple signatures.
            if len(trace_agreeing_groups) == 1:
                signature_groups = trace_agreeing_groups
                candidates = tuple(next(iter(trace_agreeing_groups.values())))
        if len(signature_groups) != 1:
            self._raise(
                anonymous_sample_id,
                "AMBIGUOUS_EDIT_MAPPING",
                "trace constraints leave multiple inequivalent graph-edit signatures",
                optimal_mapping_count=len(candidates),
                inequivalent_edit_signature_count=len(signature_groups),
            )
        selected = sorted(
            candidates,
            key=lambda candidate: (
                not set(trace_anchors).issubset(candidate.anchors),
                tuple(
                    (
                        pair.source.atom_id,
                        pair.product.atom_id,
                    )
                    for pair in candidate.domain_mapping.pairs
                ),
            ),
        )
        representative = selected[0]
        symmetry_groups, valid_anchors = self._anchor_symmetry(
            source_clean,
            source_maps,
            frozenset().union(*(candidate.anchors for candidate in selected)),
            trace_anchors,
        )
        trace_agreement = (
            None
            if not trace_anchors
            else set(trace_anchors).issubset(valid_anchors)
        )
        mapped_heavy = len(representative.mapping)
        source_heavy = source_clean.GetNumHeavyAtoms()
        product_heavy = product_clean.GetNumHeavyAtoms()
        coverage = mapped_heavy / min(source_heavy, product_heavy)
        inequivalent_signatures = 1
        ambiguity_penalty = 1.0 / inequivalent_signatures
        confidence = coverage * ambiguity_penalty
        evidence = MappingEvidence(
            algorithm=algorithm,
            rdkit_version=rdkit.__version__,
            mcs_smarts=mcs_smarts,
            source_heavy_atoms=source_heavy,
            product_heavy_atoms=product_heavy,
            mapped_heavy_atoms=mapped_heavy,
            optimal_mappings=tuple(candidate.domain_mapping for candidate in selected),
            inequivalent_edit_signature_count=inequivalent_signatures,
            coverage=coverage,
            ambiguity_penalty=ambiguity_penalty,
            confidence=confidence,
            trace_anchor_indices=trace_anchors,
            trace_consistent=trace_agreement,
        )
        try:
            return EditTruth(
                anonymous_sample_id=anonymous_sample_id,
                normalized_subtask=normalized_subtask,
                source_smiles=source_smiles,
                gt_smiles=gt_smiles,
                canonical_source_smiles=canonical_source,
                canonical_gt_smiles=canonical_product,
                valid_anchor_indices=tuple(sorted(valid_anchors)),
                symmetry_equivalent_anchors=symmetry_groups,
                removed_atom_maps=representative.removed_atom_maps,
                added_atoms=representative.added_atoms,
                broken_bonds=representative.broken_bonds,
                formed_bonds=representative.formed_bonds,
                remove_fragment=representative.remove_fragment,
                add_fragment=representative.add_fragment,
                source_descriptors=compute_descriptors(
                    source_smiles, fragment_policy=FragmentPolicy.KEEP_ALL
                ),
                product_descriptors=compute_descriptors(
                    gt_smiles, fragment_policy=FragmentPolicy.KEEP_ALL
                ),
                mapping_evidence=evidence,
                mapping_confidence=confidence,
            )
        except (TypeError, ValueError) as error:
            self._raise(
                anonymous_sample_id,
                "EDIT_TRUTH_INVARIANT",
                "derived graph edit violates the immutable EditTruth contract",
                error_type=type(error).__name__,
                mapped_heavy_atoms=mapped_heavy,
                removed_atom_count=len(representative.removed_atom_maps),
                added_atom_count=len(representative.added_atoms),
            )

    @staticmethod
    def _parse(smiles: str) -> Chem.Mol:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            raise ValueError("SMILES strict sanitization failed")
        return molecule

    def _source_maps(
        self, molecule: Chem.Mol, anonymous_sample_id: str
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        seen: set[int] = set()
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            atom_map = atom.GetAtomMapNum()
            if atom_map <= 0:
                self._raise(
                    anonymous_sample_id,
                    "MISSING_SOURCE_ATOM_MAP",
                    "every source heavy atom must carry a positive atom-map identifier",
                    atom_index=atom.GetIdx(),
                )
            if atom_map in seen:
                self._raise(
                    anonymous_sample_id,
                    "DUPLICATE_SOURCE_ATOM_MAP",
                    "source heavy-atom map identifiers must be unique",
                    atom_map=atom_map,
                )
            seen.add(atom_map)
            result[atom.GetIdx()] = atom_map
        if not result:
            self._raise(
                anonymous_sample_id,
                "EMPTY_SOURCE_GRAPH",
                "source must contain at least one mapped heavy atom",
            )
        return result

    def _canonical_product_ids(
        self, molecule: Chem.Mol, anonymous_sample_id: str
    ) -> dict[int, int]:
        ranks = tuple(
            int(value)
            for value in Chem.CanonicalRankAtoms(
                molecule,
                breakTies=True,
                includeChirality=True,
                includeIsotopes=True,
            )
        )
        ids = {
            atom.GetIdx(): ranks[atom.GetIdx()] + 1
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1
        }
        if len(ids.values()) != len(set(ids.values())):
            self._raise(
                anonymous_sample_id,
                "PRODUCT_CANONICAL_ID_COLLISION",
                "canonical product atom identities are not unique",
            )
        return ids

    def _matches(self, target: Chem.Mol, query: Chem.Mol) -> tuple[tuple[int, ...], ...]:
        matches = tuple(
            target.GetSubstructMatches(
                query,
                # Query automorphisms repeat the same target atom set and can
                # inflate an FMCS cross-product by orders of magnitude. Target
                # embeddings remain distinct, preserving real site ambiguity.
                uniquify=True,
                useChirality=False,
                maxMatches=self.max_matches,
            )
        )
        if len(matches) >= self.max_matches:
            raise OverflowError("substructure embedding enumeration reached its bound")
        return matches

    def _choose_mappings(
        self,
        source: Chem.Mol,
        product: Chem.Mol,
        *,
        normalized_subtask: EditingSubtask,
        remove_hint: str | None,
        anonymous_sample_id: str,
    ) -> tuple[tuple[tuple[tuple[int, int], ...], ...], str, str | None]:
        try:
            if normalized_subtask is EditingSubtask.ADD:
                direct = self._matches(product, source)
                if direct:
                    return (
                        tuple(
                            tuple(enumerate(product_match))
                            for product_match in direct
                        ),
                        "direct_source_subgraph",
                        "not_applicable",
                    )
            elif normalized_subtask is EditingSubtask.DELETE:
                direct = self._matches(source, product)
                if direct:
                    return (
                        tuple(
                            tuple(
                                sorted(
                                    (source_idx, product_idx)
                                    for product_idx, source_idx in enumerate(source_match)
                                )
                            )
                            for source_match in direct
                        ),
                        "direct_product_subgraph",
                        "not_applicable",
                    )

            if remove_hint is not None:
                seeded = self._trace_seeded_mappings(source, product, remove_hint)
                if seeded:
                    return seeded, "trace_seeded_retained_core", "not_applicable"

            exact, exact_result = self._fmcs_mappings(
                source, product, bond_typer=rdFMCS.BondCompare.CompareOrderExact
            )
            relaxed: tuple[tuple[tuple[int, int], ...], ...] = ()
            relaxed_result = None
            source_elements = sorted(
                atom.GetAtomicNum()
                for atom in source.GetAtoms()
                if atom.GetAtomicNum() > 1
            )
            product_elements = sorted(
                atom.GetAtomicNum()
                for atom in product.GetAtoms()
                if atom.GetAtomicNum() > 1
            )
            if source_elements == product_elements:
                relaxed, relaxed_result = self._fmcs_mappings(
                    source, product, bond_typer=rdFMCS.BondCompare.CompareAny
                )
            exact_size = len(exact[0]) if exact else 0
            relaxed_size = len(relaxed[0]) if relaxed else 0
            if relaxed and relaxed_size > exact_size:
                if relaxed_result is not None and relaxed_result.canceled:
                    self._raise(
                        anonymous_sample_id,
                        "MCS_TIMEOUT",
                        "bond-order-relaxed FMCS reached its timeout",
                        timeout_seconds=self.mcs_timeout_seconds,
                    )
                return (
                    relaxed,
                    "rdkit_fmcs_compare_any",
                    relaxed_result.smartsString if relaxed_result is not None else None,
                )
            if exact_result.canceled:
                self._raise(
                    anonymous_sample_id,
                    "MCS_TIMEOUT",
                    "exact-bond FMCS reached its timeout",
                    timeout_seconds=self.mcs_timeout_seconds,
                )
            if exact:
                return exact, "rdkit_fmcs_compare_order_exact", exact_result.smartsString
        except EditTruthBuildError:
            raise
        except (OverflowError, RuntimeError, ValueError) as error:
            self._raise(
                anonymous_sample_id,
                "MAPPING_ENUMERATION",
                "graph mapping enumeration failed or exceeded its deterministic bound",
                error_type=type(error).__name__,
                max_matches=self.max_matches,
            )
        self._raise(
            anonymous_sample_id,
            "NO_GRAPH_MAPPING",
            "source and product have no admissible heavy-atom mapping",
        )

    def _trace_seeded_mappings(
        self, source: Chem.Mol, product: Chem.Mol, remove_hint: str
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        hint = self._parse(remove_hint)
        fragment_matches = self._matches(source, hint)
        mappings: set[tuple[tuple[int, int], ...]] = set()
        for fragment_match in fragment_matches:
            removed = set(fragment_match)
            boundary_source_indices = {
                neighbor.GetIdx()
                for removed_idx in removed
                for neighbor in source.GetAtomWithIdx(removed_idx).GetNeighbors()
                if neighbor.GetIdx() not in removed
            }
            editable = Chem.RWMol(source)
            for atom in editable.GetAtoms():
                atom.SetIntProp("_molhallulens_source_idx", atom.GetIdx())
            for atom_idx in sorted(removed, reverse=True):
                editable.RemoveAtom(atom_idx)
            core = editable.GetMol()
            if core.GetNumHeavyAtoms() == 0:
                continue
            for atom in core.GetAtoms():
                if atom.GetIntProp("_molhallulens_source_idx") in boundary_source_indices:
                    # Removing a substituent can leave RDKit's query copy as an
                    # artificial radical/no-implicit-H atom. Relax only the cut
                    # boundary; changing the whole retained core would erase
                    # chemically meaningful valence constraints.
                    atom.SetNumRadicalElectrons(0)
                    atom.SetNoImplicit(False)
                    atom.SetNumExplicitHs(0)
                    atom.UpdatePropertyCache(strict=False)
            # First preserve the original source atom valence/H state. Sanitizing
            # after cutting a leaving group can add an implicit H at the attachment
            # atom, turning the retained core into the wrong query for a newly
            # substituted product (for example aromatic C-Cl -> C-N).
            try:
                product_matches = self._matches(product, core)
            except (RuntimeError, ValueError):
                product_matches = ()
            if not product_matches:
                sanitized_core = Chem.Mol(core)
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(sanitized_core)
                    product_matches = self._matches(product, sanitized_core)
                    core = sanitized_core
                except (RuntimeError, ValueError):
                    product_matches = ()
            for product_match in product_matches:
                mapping = tuple(
                    sorted(
                        (
                            core.GetAtomWithIdx(core_idx).GetIntProp(
                                "_molhallulens_source_idx"
                            ),
                            product_idx,
                        )
                        for core_idx, product_idx in enumerate(product_match)
                    )
                )
                mappings.add(mapping)
        return tuple(sorted(mappings))

    def _fmcs_mappings(
        self,
        source: Chem.Mol,
        product: Chem.Mol,
        *,
        bond_typer: Any,
    ) -> tuple[tuple[tuple[tuple[int, int], ...], ...], Any]:
        parameters = rdFMCS.MCSParameters()
        parameters.Timeout = self.mcs_timeout_seconds
        parameters.MaximizeBonds = True
        parameters.AtomTyper = rdFMCS.AtomCompare.CompareElements
        parameters.BondTyper = bond_typer
        parameters.AtomCompareParameters.MatchFormalCharge = True
        parameters.AtomCompareParameters.MatchChiralTag = False
        parameters.AtomCompareParameters.RingMatchesRingOnly = True
        parameters.BondCompareParameters.RingMatchesRingOnly = True
        parameters.BondCompareParameters.CompleteRingsOnly = True
        result = rdFMCS.FindMCS((source, product), parameters)
        if not result.smartsString:
            return (), result
        query = Chem.MolFromSmarts(result.smartsString)
        if query is None:
            return (), result
        source_matches = self._matches(source, query)
        product_matches = self._matches(product, query)
        if len(source_matches) * len(product_matches) >= self.max_matches:
            raise OverflowError("FMCS embedding cross-product reached its bound")
        mappings = {
            tuple(sorted(zip(source_match, product_match, strict=True)))
            for source_match in source_matches
            for product_match in product_matches
        }
        return tuple(sorted(mappings)), result

    def _analyze_candidates(
        self,
        raw_mappings: tuple[tuple[tuple[int, int], ...], ...],
        source: Chem.Mol,
        product: Chem.Mol,
        source_maps: dict[int, int],
        product_ids: dict[int, int],
        *,
        trace_anchors: tuple[int, ...],
        remove_hint: str | None,
        add_hint: str | None,
        anonymous_sample_id: str,
    ) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        seen_mappings: set[tuple[tuple[int, int], ...]] = set()
        source_heavy = set(source_maps)
        product_heavy = set(product_ids)
        symmetry_by_map = self._source_symmetry_by_map(source, source_maps)
        symmetry_by_product_id = self._product_symmetry_by_id(product, product_ids)
        for raw_mapping in raw_mappings:
            mapping = tuple(
                sorted(
                    (source_idx, product_idx)
                    for source_idx, product_idx in raw_mapping
                    if source_idx in source_heavy and product_idx in product_heavy
                )
            )
            if not mapping or mapping in seen_mappings:
                continue
            if len({item[0] for item in mapping}) != len(mapping) or len(
                {item[1] for item in mapping}
            ) != len(mapping):
                continue
            seen_mappings.add(mapping)
            mapping_dict = dict(mapping)
            inverse_mapping = {product_idx: source_idx for source_idx, product_idx in mapping}
            removed = frozenset(source_heavy - mapping_dict.keys())
            added = frozenset(product_heavy - inverse_mapping.keys())
            broken, formed = self._bond_differences(
                source,
                product,
                mapping_dict,
                source_maps,
                product_ids,
            )
            remove_fragment = self._fragment_spec(
                source,
                removed,
                source_maps,
                broken,
                AtomReferenceNamespace.SOURCE_MAP,
            )
            add_fragment = self._fragment_spec(
                product,
                added,
                product_ids,
                formed,
                AtomReferenceNamespace.PRODUCT_CANONICAL,
            )
            anchors = self._candidate_anchors(
                source,
                product,
                mapping_dict,
                removed,
                added,
                source_maps,
            )
            remove_match = self._hint_matches(remove_fragment, remove_hint)
            add_match = self._hint_matches(add_fragment, add_hint)
            mapping_pairs = tuple(
                sorted(
                    (
                    AtomMappingPair(
                        AtomReference(
                            AtomReferenceNamespace.SOURCE_MAP,
                            source_maps[source_idx],
                        ),
                        AtomReference(
                            AtomReferenceNamespace.PRODUCT_CANONICAL,
                            product_ids[product_idx],
                        ),
                    )
                    for source_idx, product_idx in mapping
                    ),
                    key=lambda pair: (
                        pair.source.atom_id,
                        pair.product.atom_id,
                    ),
                )
            )
            domain_mapping = AtomMapping(mapping_pairs)
            added_atoms = tuple(
                sorted(
                    (
                        self._atom_descriptor(
                            product.GetAtomWithIdx(atom_idx),
                            AtomReference(
                                AtomReferenceNamespace.PRODUCT_CANONICAL,
                                product_ids[atom_idx],
                            ),
                        )
                        for atom_idx in added
                    ),
                    key=lambda item: _reference_key(item.reference),
                )
            )
            anchor_classes = tuple(
                sorted({symmetry_by_map[anchor] for anchor in anchors})
            )
            signature = self._canonical_edit_signature(
                remove_fragment,
                add_fragment,
                added_atoms,
                broken,
                formed,
                anchor_classes,
                symmetry_by_map,
                symmetry_by_product_id,
            )
            candidates.append(
                _Candidate(
                    mapping=mapping,
                    domain_mapping=domain_mapping,
                    removed_source_indices=removed,
                    added_product_indices=added,
                    removed_atom_maps=frozenset(source_maps[index] for index in removed),
                    added_atoms=added_atoms,
                    broken_bonds=broken,
                    formed_bonds=formed,
                    remove_fragment=remove_fragment,
                    add_fragment=add_fragment,
                    anchors=anchors,
                    hint_score=(remove_match + add_match, len(mapping), 0),
                    signature=signature,
                )
            )
        if not candidates:
            self._raise(
                anonymous_sample_id,
                "NO_HEAVY_ATOM_MAPPING",
                "mapping enumeration produced no injective heavy-atom mapping",
            )
        return tuple(candidates)

    @staticmethod
    def _canonical_edit_signature(
        remove_fragment: FragmentSpec | None,
        add_fragment: FragmentSpec | None,
        added_atoms: tuple[AtomDescriptor, ...],
        broken_bonds: tuple[BondEdit, ...],
        formed_bonds: tuple[BondEdit, ...],
        anchor_classes: tuple[tuple[int, ...], ...],
        source_symmetry: dict[int, tuple[int, ...]],
        product_symmetry: dict[int, tuple[int, ...]],
    ) -> tuple[Any, ...]:
        """Collapse embeddings that differ only by graph automorphism.

        Product canonical IDs deliberately do not participate: they are stable
        identities for audit output, but symmetry-equivalent embeddings may assign
        the edited role to different canonical IDs. Fragment graphs, bond kinds,
        atom chemistry, and source symmetry classes define edit equivalence.
        """

        atom_chemistry = tuple(
            sorted(
                (
                    atom.atomic_number,
                    atom.isotope,
                    atom.formal_charge,
                    atom.aromatic,
                    atom.chiral_tag,
                )
                for atom in added_atoms
            )
        )
        def normalized_reference(reference: AtomReference) -> tuple[str, tuple[int, ...]]:
            if reference.namespace is AtomReferenceNamespace.SOURCE_MAP:
                return (reference.namespace.value, source_symmetry[reference.atom_id])
            return (reference.namespace.value, product_symmetry[reference.atom_id])

        def normalized_bond(bond: BondEdit) -> tuple[Any, ...]:
            endpoints = tuple(
                sorted(
                    (
                        normalized_reference(bond.begin),
                        normalized_reference(bond.end),
                    )
                )
            )
            return (endpoints, bond.bond_type.value, bond.stereo, bond.aromatic)

        broken_chemistry = tuple(
            sorted(
                normalized_bond(bond) for bond in broken_bonds
            )
        )
        formed_chemistry = tuple(
            sorted(
                normalized_bond(bond) for bond in formed_bonds
            )
        )
        return (
            None if remove_fragment is None else remove_fragment.canonical_smiles,
            None if add_fragment is None else add_fragment.canonical_smiles,
            atom_chemistry,
            broken_chemistry,
            formed_chemistry,
            anchor_classes,
        )

    @staticmethod
    def _hint_matches(fragment: FragmentSpec | None, hint: str | None) -> int:
        if hint is None:
            return 1 if fragment is None else 0
        if fragment is None:
            return 0
        try:
            return int(fragment_graph_equivalent(fragment.canonical_smiles, hint))
        except (TypeError, ValueError):
            return 0

    def _bond_differences(
        self,
        source: Chem.Mol,
        product: Chem.Mol,
        mapping: dict[int, int],
        source_maps: dict[int, int],
        product_ids: dict[int, int],
    ) -> tuple[tuple[BondEdit, ...], tuple[BondEdit, ...]]:
        broken: list[BondEdit] = []
        formed: list[BondEdit] = []
        handled_product_bonds: set[int] = set()
        inverse_mapping = {
            product_idx: source_idx for source_idx, product_idx in mapping.items()
        }

        def product_reference(product_idx: int) -> AtomReference:
            source_idx = inverse_mapping.get(product_idx)
            if source_idx is not None:
                return AtomReference(
                    AtomReferenceNamespace.SOURCE_MAP, source_maps[source_idx]
                )
            return AtomReference(
                AtomReferenceNamespace.PRODUCT_CANONICAL, product_ids[product_idx]
            )

        for bond in source.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin not in source_maps or end not in source_maps:
                continue
            # Internal fragment bonds are already represented by FragmentSpec.
            if begin not in mapping and end not in mapping:
                continue
            product_bond = None
            if begin in mapping and end in mapping:
                product_bond = product.GetBondBetweenAtoms(mapping[begin], mapping[end])
            if product_bond is not None and self._same_bond(bond, product_bond):
                handled_product_bonds.add(product_bond.GetIdx())
                continue
            broken.append(
                self._bond_edit(
                    bond,
                    {
                        begin: AtomReference(
                            AtomReferenceNamespace.SOURCE_MAP, source_maps[begin]
                        ),
                        end: AtomReference(
                            AtomReferenceNamespace.SOURCE_MAP, source_maps[end]
                        ),
                    },
                )
            )
            if product_bond is not None:
                formed.append(
                    self._bond_edit(
                        product_bond,
                        {
                            product_bond.GetBeginAtomIdx(): product_reference(
                                product_bond.GetBeginAtomIdx()
                            ),
                            product_bond.GetEndAtomIdx(): product_reference(
                                product_bond.GetEndAtomIdx()
                            ),
                        },
                    )
                )
                handled_product_bonds.add(product_bond.GetIdx())
        for bond in product.GetBonds():
            if bond.GetIdx() in handled_product_bonds:
                continue
            if (
                bond.GetBeginAtomIdx() not in product_ids
                or bond.GetEndAtomIdx() not in product_ids
            ):
                continue
            # Internal added-fragment bonds are represented by FragmentSpec.
            if (
                bond.GetBeginAtomIdx() not in inverse_mapping
                and bond.GetEndAtomIdx() not in inverse_mapping
            ):
                continue
            formed.append(
                self._bond_edit(
                    bond,
                    {
                        bond.GetBeginAtomIdx(): product_reference(
                            bond.GetBeginAtomIdx()
                        ),
                        bond.GetEndAtomIdx(): product_reference(
                            bond.GetEndAtomIdx()
                        ),
                    },
                )
            )
        return (
            tuple(sorted(set(broken), key=_bond_key)),
            tuple(sorted(set(formed), key=_bond_key)),
        )

    @staticmethod
    def _same_bond(left: Chem.Bond, right: Chem.Bond) -> bool:
        return (
            left.GetBondType() == right.GetBondType()
            and left.GetIsAromatic() == right.GetIsAromatic()
            and str(left.GetStereo()) == str(right.GetStereo())
        )

    def _bond_edit(
        self, bond: Chem.Bond, references: dict[int, AtomReference]
    ) -> BondEdit:
        try:
            bond_type = _BOND_TYPES[bond.GetBondType()]
        except KeyError as error:
            raise ValueError(f"unsupported bond type: {bond.GetBondType()}") from error
        return BondEdit(
            begin=references[bond.GetBeginAtomIdx()],
            end=references[bond.GetEndAtomIdx()],
            bond_type=bond_type,
            stereo=str(bond.GetStereo()),
            aromatic=bool(bond.GetIsAromatic()),
        )

    @staticmethod
    def _atom_descriptor(atom: Chem.Atom, reference: AtomReference) -> AtomDescriptor:
        return AtomDescriptor(
            reference=reference,
            atomic_number=atom.GetAtomicNum(),
            element=atom.GetSymbol(),
            isotope=atom.GetIsotope(),
            formal_charge=atom.GetFormalCharge(),
            aromatic=bool(atom.GetIsAromatic()),
            chiral_tag=str(atom.GetChiralTag()),
        )

    def _fragment_spec(
        self,
        molecule: Chem.Mol,
        atom_indices: frozenset[int],
        identities: dict[int, int],
        changed_bonds: tuple[BondEdit, ...],
        namespace: AtomReferenceNamespace,
    ) -> FragmentSpec | None:
        if not atom_indices:
            return None
        references = tuple(
            sorted(
                (
                    AtomReference(namespace, identities[index])
                    for index in atom_indices
                ),
                key=_reference_key,
            )
        )
        reference_set = frozenset(references)
        components = self._induced_components(molecule, atom_indices)
        attachment_indices = {
            index
            for index in atom_indices
            if any(
                neighbor.GetIdx() not in atom_indices
                for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            )
        }
        component_smiles = tuple(
            sorted(
                Chem.MolFragmentToSmiles(
                    molecule,
                    atomsToUse=sorted(component),
                    canonical=True,
                    isomericSmiles=True,
                )
                for component in components
            )
        )
        try:
            canonical_smiles = canonicalize_smiles(".".join(component_smiles))
        except ValueError:
            # Aromatic heteroatoms encode their substituted valence in the full
            # product. A raw induced fragment such as triazol-1-yl can therefore
            # be non-kekulizable after its boundary bond is removed. Cap only its
            # attachment atom(s) and retry as a standalone audited fragment.
            component_smiles = tuple(
                sorted(
                    self._capped_component_smiles(
                        molecule,
                        component,
                        attachment_indices.intersection(component),
                    )
                    for component in components
                )
            )
            canonical_smiles = canonicalize_smiles(".".join(component_smiles))
        attachment_atoms = tuple(
            sorted(
                (
                    AtomReference(namespace, identities[index])
                    for index in attachment_indices
                ),
                key=_reference_key,
            )
        )
        boundary_bonds = tuple(
            sorted(
                (
                    bond
                    for bond in changed_bonds
                    if (bond.begin in reference_set) ^ (bond.end in reference_set)
                ),
                key=_bond_key,
            )
        )
        return FragmentSpec(
            canonical_smiles=canonical_smiles,
            component_smiles=component_smiles,
            atom_references=references,
            attachment_atoms=attachment_atoms,
            boundary_bonds=boundary_bonds,
            descriptors=compute_descriptors(
                canonical_smiles, fragment_policy=FragmentPolicy.KEEP_ALL
            ),
        )

    @staticmethod
    def _capped_component_smiles(
        molecule: Chem.Mol,
        component: frozenset[int],
        attachment_indices: set[int],
    ) -> str:
        editable = Chem.RWMol(molecule)
        for atom in editable.GetAtoms():
            atom.SetIntProp("_molhallulens_fragment_source_idx", atom.GetIdx())
        for atom_idx in sorted(
            set(range(molecule.GetNumAtoms())) - set(component), reverse=True
        ):
            editable.RemoveAtom(atom_idx)
        fragment = editable.GetMol()
        for atom in fragment.GetAtoms():
            source_idx = atom.GetIntProp("_molhallulens_fragment_source_idx")
            if source_idx not in attachment_indices:
                continue
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.UpdatePropertyCache(strict=False)
            if (
                atom.GetAtomicNum() == 7
                and atom.GetIsAromatic()
                and atom.GetFormalCharge() == 0
                and atom.GetDegree() == 2
            ):
                atom.SetNumExplicitHs(1)
                atom.SetNoImplicit(True)
                atom.UpdatePropertyCache(strict=False)
        with rdBase.BlockLogs():
            Chem.SanitizeMol(fragment)
        return Chem.MolToSmiles(
            fragment,
            canonical=True,
            isomericSmiles=True,
        )

    @staticmethod
    def _induced_components(
        molecule: Chem.Mol, atom_indices: frozenset[int]
    ) -> tuple[frozenset[int], ...]:
        remaining = set(atom_indices)
        result: list[frozenset[int]] = []
        while remaining:
            start = min(remaining)
            stack = [start]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current not in remaining:
                    continue
                remaining.remove(current)
                component.add(current)
                stack.extend(
                    neighbor.GetIdx()
                    for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors()
                    if neighbor.GetIdx() in remaining
                )
            result.append(frozenset(component))
        return tuple(sorted(result, key=lambda item: tuple(sorted(item))))

    @staticmethod
    def _candidate_anchors(
        source: Chem.Mol,
        product: Chem.Mol,
        mapping: dict[int, int],
        removed: frozenset[int],
        added: frozenset[int],
        source_maps: dict[int, int],
    ) -> frozenset[int]:
        inverse = {product_idx: source_idx for source_idx, product_idx in mapping.items()}
        anchors: set[int] = set()
        for bond in source.GetBonds():
            endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if len(endpoints & removed) == 1:
                retained = next(iter(endpoints - removed))
                if retained in source_maps:
                    anchors.add(source_maps[retained])
            elif not endpoints & removed and endpoints <= mapping.keys():
                mapped = tuple(mapping[index] for index in endpoints)
                product_bond = product.GetBondBetweenAtoms(*mapped)
                if product_bond is None or not EditTruthBuilder._same_bond(bond, product_bond):
                    anchors.update(source_maps[index] for index in endpoints)
        for bond in product.GetBonds():
            endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if len(endpoints & added) == 1:
                retained_product = next(iter(endpoints - added))
                retained_source = inverse.get(retained_product)
                if retained_source in source_maps:
                    anchors.add(source_maps[retained_source])
            elif not endpoints & added and endpoints <= inverse.keys():
                source_endpoints = tuple(inverse[index] for index in endpoints)
                source_bond = source.GetBondBetweenAtoms(*source_endpoints)
                if source_bond is None or not EditTruthBuilder._same_bond(
                    source_bond, bond
                ):
                    anchors.update(source_maps[index] for index in source_endpoints)
        return frozenset(anchors)

    @staticmethod
    def _source_symmetry_by_map(
        source: Chem.Mol, source_maps: dict[int, int]
    ) -> dict[int, tuple[int, ...]]:
        ranks = tuple(
            int(value)
            for value in Chem.CanonicalRankAtoms(
                source,
                breakTies=False,
                includeChirality=True,
                includeIsotopes=True,
            )
        )
        groups: dict[int, list[int]] = defaultdict(list)
        for index, atom_map in source_maps.items():
            groups[ranks[index]].append(atom_map)
        result: dict[int, tuple[int, ...]] = {}
        for members in groups.values():
            group = tuple(sorted(members))
            for atom_map in group:
                result[atom_map] = group
        return result

    @staticmethod
    def _product_symmetry_by_id(
        product: Chem.Mol, product_ids: dict[int, int]
    ) -> dict[int, tuple[int, ...]]:
        ranks = tuple(
            int(value)
            for value in Chem.CanonicalRankAtoms(
                product,
                breakTies=False,
                includeChirality=True,
                includeIsotopes=True,
            )
        )
        groups: dict[int, list[int]] = defaultdict(list)
        for index, canonical_id in product_ids.items():
            groups[ranks[index]].append(canonical_id)
        result: dict[int, tuple[int, ...]] = {}
        for members in groups.values():
            group = tuple(sorted(members))
            for canonical_id in group:
                result[canonical_id] = group
        return result

    def _anchor_symmetry(
        self,
        source: Chem.Mol,
        source_maps: dict[int, int],
        inferred_anchors: frozenset[int],
        trace_anchors: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], frozenset[int]]:
        symmetry = self._source_symmetry_by_map(source, source_maps)
        # The trace is only a disambiguating seed. Graph-derived anchors remain
        # authoritative even when a malformed trace names a removed atom.
        seeds = inferred_anchors
        if not seeds:
            return (), frozenset()
        all_groups = tuple(sorted({symmetry[anchor] for anchor in seeds}))
        valid = frozenset(member for group in all_groups for member in group)
        non_singletons = tuple(group for group in all_groups if len(group) >= 2)
        return non_singletons, valid


def derive_edit_truth(artifact: ReferenceDAGArtifact) -> EditTruth:
    """Derive EditTruth from a validated T011 artifact using frozen defaults."""

    return EditTruthBuilder().build(artifact)


__all__ = [
    "EditTruthBuildError",
    "EditTruthBuilder",
    "derive_edit_truth",
]
