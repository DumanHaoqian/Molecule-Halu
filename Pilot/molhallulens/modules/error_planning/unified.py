"""Single-path planner for configurable, independent multi-point edits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random
from typing import Any

from rdkit import Chem, rdBase

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)
from molhallulens.core import (
    EditingSubtask,
    MutationCategory,
    PlannedMutation,
    UnifiedHallucinationPlan,
    ValueType,
    editing_schema_for,
)
from molhallulens.modules.reference import ReferenceDAGArtifact

from .fragment_pool import FragmentPool
from .smiles_mutation import select_smiles_mutation


class HallucinationPlanningError(RuntimeError):
    """Raised when requested edits cannot be produced under the explicit config."""


@dataclass(frozen=True, slots=True)
class _SemanticTarget:
    semantic_id: str
    node_ids: tuple[str, ...]
    value_type: ValueType
    before: Any


def _stable_seed(global_seed: int, origin_id: str, variant_index: int) -> int:
    payload = f"{global_seed}\0{origin_id}\0{variant_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _candidate_id(origin_id: str, semantic_id: str, operator: str, after: Any) -> str:
    payload = f"{origin_id}\0{semantic_id}\0{operator}\0{after!r}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{origin_id}.mutation.{semantic_id}.{digest}"


def _target_seed(derived_seed: int, semantic_id: str) -> int:
    payload = f"{derived_seed}\0{semantic_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _atom_map_candidates(
    indexed_smiles: str,
    *,
    reference_index: int,
    prefer_same_element: bool,
) -> tuple[tuple[int, str], ...]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(indexed_smiles)
    if molecule is None:
        raise HallucinationPlanningError("indexed source SMILES cannot be parsed")
    mapped = tuple(
        (atom.GetAtomMapNum(), atom.GetSymbol())
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0 and atom.GetAtomMapNum() != reference_index
    )
    if not mapped:
        raise HallucinationPlanningError("source molecule has no alternate mapped atom")
    reference_atom = next(
        (atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == reference_index),
        None,
    )
    if reference_atom is None:
        raise HallucinationPlanningError("reference anchor does not exist in indexed source")
    if prefer_same_element:
        same_element = tuple(
            item for item in mapped if item[1] == reference_atom.GetSymbol()
        )
        if same_element:
            return tuple(sorted(same_element))
    return tuple(sorted(mapped))


class UnifiedHallucinationPlanner:
    """Plan K independently configured semantic edits for one positive sample."""

    def __init__(
        self,
        fragment_pool: FragmentPool,
        config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
    ) -> None:
        if not isinstance(fragment_pool, FragmentPool):
            raise TypeError("fragment_pool must be FragmentPool")
        if type(config) is not HallucinationGenerationConfig:
            raise TypeError("config must be HallucinationGenerationConfig")
        self.fragment_pool = fragment_pool
        self.config = config

    def plan(
        self,
        artifact: ReferenceDAGArtifact,
        *,
        variant_index: int = 0,
    ) -> UnifiedHallucinationPlan:
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("artifact must be ReferenceDAGArtifact")
        if type(variant_index) is not int or variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer")

        derived_seed = _stable_seed(
            self.config.global_seed,
            artifact.anonymous_sample_id,
            variant_index,
        )
        selection_random = Random(derived_seed)
        requested_count = self.config.requested_edit_count(selection_random)
        targets = self._available_targets(artifact)

        targets_by_id = {target.semantic_id: target for target in targets}
        reasoning_ids = sorted(
            semantic_id for semantic_id in targets_by_id if semantic_id != "final_answer"
        )
        selection_random.shuffle(reasoning_ids)
        final_available = "final_answer" in targets_by_id
        final_selected = (
            final_available
            and self.config.include_final_answer
            and selection_random.random() < self.config.final_answer_probability
        )
        target_order = list(reasoning_ids)
        if final_available:
            if final_selected:
                target_order.insert(0, "final_answer")
            else:
                target_order.append("final_answer")

        # Build only the candidates we may actually use. This matters for product
        # SMILES, whose chemically valid candidate enumeration is comparatively
        # expensive on large molecules.
        mutations: list[PlannedMutation] = []
        failures: dict[str, str] = {}
        for semantic_id in target_order:
            if len(mutations) == requested_count:
                break
            try:
                mutations.append(
                    self._propose(
                        artifact,
                        targets_by_id[semantic_id],
                        Random(_target_seed(derived_seed, semantic_id)),
                    )
                )
            except (HallucinationPlanningError, ValueError) as error:
                failures[semantic_id] = str(error)

        if len(mutations) != requested_count:
            raise HallucinationPlanningError(
                "not enough editable targets for configured edit_count; "
                f"requested={requested_count}, generated={len(mutations)}, failures={failures}"
            )
        plan_id = f"{artifact.anonymous_sample_id}__hallucination_{variant_index:03d}"
        return UnifiedHallucinationPlan(
            plan_id=plan_id,
            origin_id=artifact.anonymous_sample_id,
            variant_index=variant_index,
            derived_seed=derived_seed,
            requested_edit_count=requested_count,
            mutations=tuple(mutations),
        )

    def _available_targets(
        self,
        artifact: ReferenceDAGArtifact,
    ) -> tuple[_SemanticTarget, ...]:
        subtask_name = artifact.normalized_subtask.value
        configured = self.config.editable_nodes_by_subtask[subtask_name]
        definition = editing_schema_for(artifact.normalized_subtask)
        graph = artifact.state_dag
        targets = []
        for semantic_id in configured:
            if semantic_id in definition.semantic_state_groups:
                node_ids = tuple(definition.semantic_state_groups[semantic_id])
            elif semantic_id in graph.schema.nodes_by_id:
                node_ids = (semantic_id,)
            else:
                raise HallucinationPlanningError(
                    f"configured semantic target {semantic_id!r} is not in {subtask_name} schema"
                )
            specs = tuple(graph.schema.nodes_by_id[node_id] for node_id in node_ids)
            if any(not spec.mutable for spec in specs):
                raise HallucinationPlanningError(
                    f"configured target {semantic_id!r} contains an immutable node"
                )
            if semantic_id == "final_answer":
                if not self.config.include_final_answer:
                    continue
            elif not self.config.include_reasoning_steps:
                continue
            values = tuple(graph.values[node_id].normalized_value for node_id in node_ids)
            types = tuple(graph.values[node_id].value_type for node_id in node_ids)
            if len(set(values)) != 1 or len(set(types)) != 1:
                raise HallucinationPlanningError(
                    f"semantic target {semantic_id!r} does not share one value/type"
                )
            targets.append(
                _SemanticTarget(
                    semantic_id=semantic_id,
                    node_ids=node_ids,
                    value_type=types[0],
                    before=values[0],
                )
            )
        return tuple(targets)

    def _propose(
        self,
        artifact: ReferenceDAGArtifact,
        target: _SemanticTarget,
        random_source: Random,
    ) -> PlannedMutation:
        value_type = target.value_type
        before = target.before
        operator: str
        after: Any
        magnitude: int | float | None = None
        similarity: float | None = None
        metadata: dict[str, Any] = {}
        category: MutationCategory

        if value_type in {ValueType.COUNT, ValueType.INTEGER}:
            category = MutationCategory.NUMERIC_INTEGER
            deltas = tuple(
                delta
                for delta in self.config.integer_deltas
                if value_type is not ValueType.COUNT
                or before + delta >= self.config.count_min_value
            )
            if not deltas:
                raise HallucinationPlanningError("no configured integer delta is valid")
            magnitude = deltas[random_source.randrange(len(deltas))]
            after = before + magnitude
            operator = "integer_offset"
        elif value_type is ValueType.FLOAT:
            category = MutationCategory.NUMERIC_FLOAT
            if self.config.float_mutation_mode == "absolute":
                delta = self.config.float_absolute_deltas[
                    random_source.randrange(len(self.config.float_absolute_deltas))
                ]
                after = round(before + delta, self.config.float_decimal_places)
                magnitude = delta
                operator = "float_absolute_offset"
            else:
                ratio = self.config.float_relative_changes[
                    random_source.randrange(len(self.config.float_relative_changes))
                ]
                after = round(before * (1.0 + ratio), self.config.float_decimal_places)
                magnitude = ratio
                operator = "float_relative_change"
            if after == before:
                raise HallucinationPlanningError("configured float precision erased the mutation")
        elif value_type is ValueType.ATOM_INDEX:
            category = MutationCategory.ATOM_INDEX
            candidates = _atom_map_candidates(
                artifact.state_dag.values["source"].normalized_value,
                reference_index=before,
                prefer_same_element=self.config.atom_index_prefer_same_element,
            )
            after, after_element = candidates[random_source.randrange(len(candidates))]
            operator = "existing_atom_index_replacement"
            metadata = {
                "replacement_element": after_element,
                "index_distance": after - before,
                "candidate_count": len(candidates),
            }
        elif value_type is ValueType.ELEMENT:
            category = MutationCategory.ELEMENT_SYMBOL
            candidates = tuple(
                value for value in self.config.replacement_elements if value != before
            )
            after = candidates[random_source.randrange(len(candidates))]
            operator = "element_replacement"
            metadata = {"candidate_count": len(candidates)}
        elif value_type in {ValueType.FRAGMENT, ValueType.STRING}:
            category = MutationCategory.MOLECULAR_FRAGMENT
            selection = self.fragment_pool.select_replacement(
                before,
                config=self.config,
                random_source=random_source,
            )
            after = selection.entry.canonical_smiles
            operator = "fragment_pool_replacement"
            similarity = selection.similarity
            metadata = {
                "accepted_pool_size": selection.accepted_pool_size,
                "replacement_heavy_atoms": selection.entry.heavy_atom_count,
                "replacement_rings": selection.entry.ring_count,
                "replacement_formal_charge": selection.entry.formal_charge,
                "donor_origin_ids": selection.entry.source_origin_ids,
            }
        elif value_type in {ValueType.SMILES, ValueType.MOLECULE}:
            category = MutationCategory.SMILES_STRUCTURE
            selection = select_smiles_mutation(
                before,
                config=self.config,
                random_source=random_source,
            )
            after = selection.smiles
            operator = selection.operator
            similarity = selection.similarity
            metadata = {
                **selection.metadata,
                "accepted_pool_size": selection.accepted_pool_size,
            }
        else:
            raise HallucinationPlanningError(
                f"no unified operator for value type {value_type.value}"
            )

        return PlannedMutation(
            mutation_id=_candidate_id(
                artifact.anonymous_sample_id,
                target.semantic_id,
                operator,
                after,
            ),
            semantic_target_id=target.semantic_id,
            target_node_ids=target.node_ids,
            value_type=value_type,
            mutation_category=category,
            operator=operator,
            before=before,
            after=after,
            magnitude=magnitude,
            similarity=similarity,
            metadata=metadata,
        )


__all__ = ["HallucinationPlanningError", "UnifiedHallucinationPlanner"]
