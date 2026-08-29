"""Immutable operator, candidate, and perturbation recipe contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    BondTypeName,
    CandidateSourceType,
    EditKind,
    HallucinationType,
    PropagationPolicy,
)
from .state_dag import ClaimValue, freeze_string_mapping


@dataclass(frozen=True, slots=True)
class EditAction:
    edit_kind: EditKind
    source_anchor_index: int | None = None
    remove_fragment_smiles: str | None = None
    add_fragment_smiles: str | None = None
    fragment_attachment_atom: int | None = None
    bond_type: BondTypeName | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    remove_anchor_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.edit_kind) is not EditKind:
            raise TypeError("EditAction edit_kind must be an EditKind")
        if self.bond_type is not None and type(self.bond_type) is not BondTypeName:
            raise TypeError("EditAction bond_type must be a BondTypeName or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("EditAction metadata must be a mapping")
        for value, name in (
            (self.remove_fragment_smiles, "remove_fragment_smiles"),
            (self.add_fragment_smiles, "add_fragment_smiles"),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"EditAction {name} must be a string or None")
        for index, name in (
            (self.source_anchor_index, "source_anchor_index"),
            (self.fragment_attachment_atom, "fragment_attachment_atom"),
            (self.remove_anchor_index, "remove_anchor_index"),
        ):
            if index is not None and type(index) is not int:
                raise TypeError(f"EditAction {name} must be an integer or None")
            if index is not None and index < 0:
                raise ValueError(f"EditAction {name} cannot be negative")
        object.__setattr__(
            self,
            "metadata",
            freeze_string_mapping(self.metadata, name="EditAction metadata"),
        )
        if self.edit_kind is EditKind.ADDITION and not self.add_fragment_smiles:
            raise ValueError("addition EditAction requires add_fragment_smiles")
        if self.edit_kind is EditKind.DELETION and not self.remove_fragment_smiles:
            raise ValueError("deletion EditAction requires remove_fragment_smiles")
        if (
            self.edit_kind in {EditKind.ADDITION, EditKind.DELETION}
            and self.remove_anchor_index is not None
        ):
            raise ValueError(
                "remove_anchor_index is only valid for substitution EditAction"
            )
        if self.edit_kind is EditKind.SUBSTITUTION and (
            not self.remove_fragment_smiles or not self.add_fragment_smiles
        ):
            raise ValueError(
                "substitution EditAction requires remove_fragment_smiles and add_fragment_smiles"
            )


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    operator_id: str
    root_fields: frozenset[str]
    supported_policies: frozenset[PropagationPolicy]
    supported_sources: frozenset[CandidateSourceType]
    hallucination_types: frozenset[HallucinationType]
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.root_fields, "root_fields"),
            (self.supported_policies, "supported_policies"),
            (self.supported_sources, "supported_sources"),
            (self.hallucination_types, "hallucination_types"),
        ):
            if isinstance(value, (str, bytes)):
                raise TypeError(f"OperatorSpec {name} must be a collection, not text")
        object.__setattr__(self, "root_fields", frozenset(self.root_fields))
        object.__setattr__(self, "supported_policies", frozenset(self.supported_policies))
        object.__setattr__(self, "supported_sources", frozenset(self.supported_sources))
        object.__setattr__(self, "hallucination_types", frozenset(self.hallucination_types))
        if type(self.operator_id) is not str:
            raise TypeError("OperatorSpec operator_id must be a string")
        if type(self.diagnostic_only) is not bool:
            raise TypeError("OperatorSpec diagnostic_only must be bool")
        if any(type(value) is not str or not value for value in self.root_fields):
            raise TypeError("OperatorSpec root_fields must contain non-empty strings")
        if any(type(value) is not PropagationPolicy for value in self.supported_policies):
            raise TypeError("OperatorSpec supported_policies must contain PropagationPolicy values")
        if any(type(value) is not CandidateSourceType for value in self.supported_sources):
            raise TypeError("OperatorSpec supported_sources must contain CandidateSourceType values")
        if any(type(value) is not HallucinationType for value in self.hallucination_types):
            raise TypeError("OperatorSpec hallucination_types must contain HallucinationType values")
        if not self.operator_id:
            raise ValueError("OperatorSpec operator_id cannot be empty")
        for value, name in (
            (self.root_fields, "root_fields"),
            (self.supported_policies, "supported_policies"),
            (self.supported_sources, "supported_sources"),
            (self.hallucination_types, "hallucination_types"),
        ):
            if not value:
                raise ValueError(f"OperatorSpec {name} cannot be empty")


@dataclass(frozen=True, slots=True)
class CandidatePatch:
    candidate_id: str
    root_node_id: str
    old_value: ClaimValue
    new_value: ClaimValue
    edit_action: EditAction | None
    source: CandidateSourceType
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or type(self.root_node_id) is not str:
            raise TypeError("CandidatePatch IDs must be strings")
        if type(self.old_value) is not ClaimValue or type(self.new_value) is not ClaimValue:
            raise TypeError("CandidatePatch old_value and new_value must be ClaimValue values")
        if self.edit_action is not None and type(self.edit_action) is not EditAction:
            raise TypeError("CandidatePatch edit_action must be an EditAction or None")
        if type(self.source) is not CandidateSourceType:
            raise TypeError("CandidatePatch source must be a CandidateSourceType")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("CandidatePatch metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            freeze_string_mapping(self.metadata, name="CandidatePatch metadata"),
        )
        if not self.candidate_id or not self.root_node_id:
            raise ValueError("CandidatePatch IDs cannot be empty")
        if self.old_value.value_type is not self.new_value.value_type:
            raise ValueError("CandidatePatch cannot change a root node's ValueType")
        if self.old_value.semantically_equals(self.new_value):
            raise ValueError("CandidatePatch must change the normalized root value")


@dataclass(frozen=True, slots=True)
class RewriteBudget:
    max_changed_claims: int
    max_added_characters: int
    length_bucket: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_changed_claims, "max_changed_claims"),
            (self.max_added_characters, "max_added_characters"),
        ):
            if type(value) is not int:
                raise TypeError(f"RewriteBudget {name} must be an integer")
        if type(self.length_bucket) is not str:
            raise TypeError("RewriteBudget length_bucket must be a string")
        if self.max_changed_claims < 1:
            raise ValueError("RewriteBudget max_changed_claims must be positive")
        if self.max_added_characters < 0:
            raise ValueError("RewriteBudget max_added_characters cannot be negative")
        if not self.length_bucket:
            raise ValueError("RewriteBudget length_bucket cannot be empty")


@dataclass(frozen=True, slots=True)
class PerturbationRecipe:
    recipe_id: str
    origin_id: str
    operator_id: str
    policy: PropagationPolicy
    target_node_id: str
    candidate_source_mode: CandidateSourceType
    variant_index: int
    derived_seed: int
    rewrite_budget: RewriteBudget
    candidate_difficulty_bucket: str
    renderer_style_id: str
    partial_cut_nodes: frozenset[str] = frozenset()
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.partial_cut_nodes, (str, bytes)):
            raise TypeError("PerturbationRecipe partial_cut_nodes must be a collection")
        object.__setattr__(self, "partial_cut_nodes", frozenset(self.partial_cut_nodes))
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("PerturbationRecipe policy must be a PropagationPolicy")
        if type(self.candidate_source_mode) is not CandidateSourceType:
            raise TypeError(
                "PerturbationRecipe candidate_source_mode must be a CandidateSourceType"
            )
        if type(self.rewrite_budget) is not RewriteBudget:
            raise TypeError("PerturbationRecipe rewrite_budget must be a RewriteBudget")
        if not isinstance(self.constraints, Mapping):
            raise TypeError("PerturbationRecipe constraints must be a mapping")
        if any(type(node) is not str or not node for node in self.partial_cut_nodes):
            raise TypeError("PerturbationRecipe partial_cut_nodes must contain non-empty strings")
        object.__setattr__(
            self,
            "constraints",
            freeze_string_mapping(self.constraints, name="PerturbationRecipe constraints"),
        )
        for value, name in (
            (self.recipe_id, "recipe_id"),
            (self.origin_id, "origin_id"),
            (self.operator_id, "operator_id"),
            (self.target_node_id, "target_node_id"),
            (self.candidate_difficulty_bucket, "candidate_difficulty_bucket"),
            (self.renderer_style_id, "renderer_style_id"),
        ):
            if type(value) is not str:
                raise TypeError(f"PerturbationRecipe {name} must be a string")
            if not value:
                raise ValueError(f"PerturbationRecipe {name} cannot be empty")
        for value, name in (
            (self.variant_index, "variant_index"),
            (self.derived_seed, "derived_seed"),
        ):
            if type(value) is not int:
                raise TypeError(f"PerturbationRecipe {name} must be an integer")
        if self.variant_index < 0 or self.derived_seed < 0:
            raise ValueError("PerturbationRecipe variant_index and derived_seed cannot be negative")
        if self.policy is PropagationPolicy.PARTIAL and not self.partial_cut_nodes:
            raise ValueError("PARTIAL recipes require at least one partial cut node")
        if self.policy is not PropagationPolicy.PARTIAL and self.partial_cut_nodes:
            raise ValueError("only PARTIAL recipes may declare partial cut nodes")
        if self.policy is PropagationPolicy.TERMINAL and self.target_node_id != "final_answer":
            raise ValueError("TERMINAL recipes must target final_answer")


@dataclass(frozen=True, slots=True)
class CandidatePool:
    request_id: str
    candidates: tuple[CandidatePatch, ...]
    rejection_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.candidates, (str, bytes)) or isinstance(
            self.rejection_codes, (str, bytes)
        ):
            raise TypeError("CandidatePool candidates and rejection_codes must be collections")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejection_codes", tuple(self.rejection_codes))
        if type(self.request_id) is not str:
            raise TypeError("CandidatePool request_id must be a string")
        if any(type(candidate) is not CandidatePatch for candidate in self.candidates):
            raise TypeError("CandidatePool candidates must contain CandidatePatch values")
        if any(type(code) is not str or not code for code in self.rejection_codes):
            raise TypeError("CandidatePool rejection_codes must contain non-empty strings")
        if not self.request_id:
            raise ValueError("CandidatePool request_id cannot be empty")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("CandidatePool candidate IDs must be unique")
        if not self.candidates and not self.rejection_codes:
            raise ValueError("an empty CandidatePool must explain why candidates were rejected")
