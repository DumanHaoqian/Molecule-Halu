"""Fail-closed schemas for the frozen MolHalluLens configuration."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


PolicyName = Literal["LOCAL", "PARTIAL", "FULL_CF", "TERMINAL"]
SubtaskName = Literal["add", "delete", "substitute"]

FROZEN_POLICIES = ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL")
FROZEN_FIELD_ORDER = (
    "indexed_smiles",
    "instruction",
    "reasoning_chain",
    "final_answer",
)
FROZEN_SEMANTIC_TYPES = (
    "CONTRADICTION",
    "UNSUPPORTED",
    "REASONING_ERROR",
    "INVALID_CHEMISTRY",
    "CONSTRAINT_VIOLATION",
    "FORMAT_ERROR",
    "OMISSION",
    "UNVERIFIABLE",
)
FROZEN_EDIT_SUBTYPES = (
    "ANCHOR_GROUNDING",
    "REMOVE_OR_LEAVING_GROUP_IDENTIFICATION",
    "ADD_FRAGMENT_IDENTIFICATION",
    "ATTACHMENT_OR_BOND_EDIT",
    "PRODUCT_CONSTRUCTION",
    "HEAVY_ATOM_COUNT",
    "HEAVY_ATOM_ARITHMETIC",
    "RING_COUNT",
    "RING_ARITHMETIC",
    "CHEMICAL_VALIDITY",
    "INSTRUCTION_CONSTRAINT",
    "FINAL_ANSWER_IDENTITY",
    "INTERNAL_INCONSISTENCY",
    "FORMAT_SCHEMA",
    "UNSUPPORTED_NATURAL_CLAIM",
)
FROZEN_CAUSAL_ROLES = (
    "ROOT",
    "PROPAGATED_FALSE",
    "PROPAGATED_CONDITIONAL",
    "TERMINAL",
)
FROZEN_EVIDENCE_RELATIONS = (
    "CONTRADICTS_SOURCE",
    "CONTRADICTS_INSTRUCTION",
    "CONTRADICTS_REFERENCE_STATE",
    "UNSUPPORTED_BY_EVIDENCE",
    "INTERNAL_INCONSISTENCY",
)
FROZEN_OPERATOR_COMPATIBILITY = {
    "wrong_anchor_site": (
        ("LOCAL", "PARTIAL", "FULL_CF"),
        ("RULE", "RDKIT", "LLM", "HYBRID"),
    ),
    "wrong_fragment_group": (
        ("LOCAL", "PARTIAL", "FULL_CF"),
        ("RULE", "RDKIT", "LLM", "HYBRID"),
    ),
    "attachment_bond_edit": (
        ("LOCAL", "PARTIAL", "FULL_CF"),
        ("RULE", "RDKIT", "LLM", "HYBRID"),
    ),
    "numeric_count_claim": (
        ("LOCAL", "PARTIAL"),
        ("RULE", "RDKIT", "HYBRID"),
    ),
    "nl_formal_internal_relation": (
        ("LOCAL", "PARTIAL"),
        ("RULE", "LLM", "HYBRID"),
    ),
    "final_answer_identity": (
        ("TERMINAL",),
        ("RULE", "RDKIT", "LLM", "HYBRID"),
    ),
}
FROZEN_OPERATOR_QUOTAS = {
    "LOCAL": (
        ("anchor_site_grounding", 15),
        ("group_fragment_identity", 15),
        ("attachment_internal_relation", 10),
        ("heavy_ring_count_claim", 10),
    ),
    "PARTIAL": (
        ("entity_partial_propagation", 18),
        ("product_dependency_cross_step", 17),
        ("count_ring_propagation", 10),
        ("nl_formal_internal_relation", 5),
    ),
    "FULL_CF": (
        ("valid_wrong_site_occurrence_regioisomer", 18),
        ("valid_wrong_group_fragment", 15),
        ("wrong_attachment_atom_bond", 10),
        ("alternate_valid_edit_boundary", 7),
    ),
    "TERMINAL": (
        ("terminal_valid_high_similarity", 35),
        ("terminal_stereo_connectivity_regio", 10),
        ("terminal_invalid_format_diagnostic", 5),
    ),
}
FROZEN_QUOTA_BUCKET_MAPPINGS = {
    "anchor_site_grounding": ("wrong_anchor_site",),
    "group_fragment_identity": ("wrong_fragment_group",),
    "attachment_internal_relation": (
        "attachment_bond_edit",
        "nl_formal_internal_relation",
    ),
    "heavy_ring_count_claim": ("numeric_count_claim",),
    "entity_partial_propagation": (
        "wrong_anchor_site",
        "wrong_fragment_group",
        "attachment_bond_edit",
    ),
    "product_dependency_cross_step": (
        "wrong_anchor_site",
        "wrong_fragment_group",
        "attachment_bond_edit",
    ),
    "count_ring_propagation": ("numeric_count_claim",),
    "nl_formal_internal_relation": ("nl_formal_internal_relation",),
    "valid_wrong_site_occurrence_regioisomer": ("wrong_anchor_site",),
    "valid_wrong_group_fragment": ("wrong_fragment_group",),
    "wrong_attachment_atom_bond": ("attachment_bond_edit",),
    "alternate_valid_edit_boundary": (
        "wrong_anchor_site",
        "wrong_fragment_group",
        "attachment_bond_edit",
    ),
    "terminal_valid_high_similarity": ("final_answer_identity",),
    "terminal_stereo_connectivity_regio": ("final_answer_identity",),
    "terminal_invalid_format_diagnostic": ("final_answer_identity",),
}


class FrozenMapping(Mapping[object, object]):
    """A deeply read-only mapping backed by an unexposed mapping proxy."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[object, object]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(value)))

    def __getitem__(self, key: object) -> object:
        return self._data[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("FrozenMapping is immutable")

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"


def _frozen_mapping(value: Mapping[object, object]) -> FrozenMapping:
    return FrozenMapping(value)


class StrictModel(BaseModel):
    """Immutable model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetIdentityConfig(StrictModel):
    version_name: Literal["pilot_v1"]
    global_seed: Literal[20260828]
    origins_total: Literal[150]
    records_total: Literal[1200]


class InputConfig(StrictModel):
    root: Literal["Pilot/Dataset"]
    family: Literal["mol_edit"]
    subtasks: tuple[SubtaskName, ...]
    origins_per_subtask: Mapping[SubtaskName, int]

    @field_validator("origins_per_subtask")
    @classmethod
    def freeze_origins_per_subtask(
        cls, value: Mapping[SubtaskName, int]
    ) -> Mapping[SubtaskName, int]:
        return _frozen_mapping(value)

    @field_serializer("origins_per_subtask")
    def serialize_origins_per_subtask(self, value: Mapping[SubtaskName, int]) -> dict[str, int]:
        return dict(value)

    @model_validator(mode="after")
    def validate_frozen_subtasks(self) -> InputConfig:
        if self.subtasks != ("add", "delete", "substitute"):
            raise ValueError("subtasks must be ordered as add, delete, substitute")
        if self.origins_per_subtask != {"add": 50, "delete": 50, "substitute": 50}:
            raise ValueError("each editing subtask must contain exactly 50 origins")
        return self


class BundleConfig(StrictModel):
    policies: tuple[PolicyName, ...]
    matched_controls: Literal[True]
    records_per_origin: Literal[8]
    hallucinated_per_origin: Literal[4]
    faithful_per_origin: Literal[4]

    @field_validator("policies")
    @classmethod
    def validate_policy_order(cls, value: tuple[PolicyName, ...]) -> tuple[PolicyName, ...]:
        if value != FROZEN_POLICIES:
            raise ValueError(f"policies must be exactly {FROZEN_POLICIES!r}")
        return value


class SplitSubtaskQuota(StrictModel):
    train: int
    validation: int
    test: int


class SplitConfig(StrictModel):
    train_origins: Literal[100]
    validation_origins: Literal[25]
    test_origins: Literal[25]
    train_records: Literal[800]
    validation_records: Literal[200]
    test_records: Literal[200]
    enforce_leakage_groups: Literal[True]
    freeze_before_donor_generation: Literal[True]
    donor_scope: Literal["same_split_only"]
    subtask_origin_quotas: Mapping[SubtaskName, SplitSubtaskQuota]

    @field_validator("subtask_origin_quotas")
    @classmethod
    def freeze_subtask_origin_quotas(
        cls, value: Mapping[SubtaskName, SplitSubtaskQuota]
    ) -> Mapping[SubtaskName, SplitSubtaskQuota]:
        return _frozen_mapping(value)

    @field_serializer("subtask_origin_quotas")
    def serialize_subtask_origin_quotas(
        self, value: Mapping[SubtaskName, SplitSubtaskQuota]
    ) -> dict[str, SplitSubtaskQuota]:
        return dict(value)

    @model_validator(mode="after")
    def validate_subtask_quotas(self) -> SplitConfig:
        expected = {
            "add": (34, 8, 8),
            "delete": (33, 9, 8),
            "substitute": (33, 8, 9),
        }
        actual = {
            name: (quota.train, quota.validation, quota.test)
            for name, quota in self.subtask_origin_quotas.items()
        }
        if actual != expected:
            raise ValueError(f"subtask origin quotas must be {expected!r}")
        return self


class DetectorConfig(StrictModel):
    field_order: tuple[str, ...]
    include_gt_smiles: Literal[False]
    include_reference_only_metadata: Literal[False]
    activation_alignment: Literal["post_token_h_t"]
    ignored_evaluation_segments: tuple[str, ...]
    evaluated_segments: tuple[str, ...]

    @field_validator("field_order")
    @classmethod
    def validate_field_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != FROZEN_FIELD_ORDER:
            raise ValueError(f"detector field_order must be {FROZEN_FIELD_ORDER!r}")
        return value

    @model_validator(mode="after")
    def validate_evaluation_segments(self) -> DetectorConfig:
        if self.ignored_evaluation_segments != (
            "source",
            "instruction",
            "special",
            "padding",
        ):
            raise ValueError("source, instruction, special, and padding must be ignored")
        if self.evaluated_segments != ("reasoning_chain", "final_answer"):
            raise ValueError("only reasoning_chain and final_answer may be evaluated")
        return self


class DatasetConfig(StrictModel):
    schema_version: Literal["1.0"]
    dataset: DatasetIdentityConfig
    input: InputConfig
    bundle: BundleConfig
    split: SplitConfig
    detector: DetectorConfig

    @model_validator(mode="after")
    def validate_count_invariants(self) -> DatasetConfig:
        if self.dataset.origins_total != sum(self.input.origins_per_subtask.values()):
            raise ValueError("origins_total must equal the sum of subtask origins")
        if self.dataset.records_total != (
            self.dataset.origins_total * self.bundle.records_per_origin
        ):
            raise ValueError("records_total must equal origins_total * records_per_origin")
        split_origins = (
            self.split.train_origins
            + self.split.validation_origins
            + self.split.test_origins
        )
        split_records = (
            self.split.train_records
            + self.split.validation_records
            + self.split.test_records
        )
        if split_origins != self.dataset.origins_total:
            raise ValueError("split origin counts must sum to origins_total")
        if split_records != self.dataset.records_total:
            raise ValueError("split record counts must sum to records_total")
        return self


class LabelDefinition(StrictModel):
    id: int
    name: str
    description: str


class EditSubtypeDefinition(StrictModel):
    code: str
    name: str


class CanonicalAnnotationConfig(StrictModel):
    unit: Literal["character_span"]
    interval: Literal["half_open"]
    primary_span: Literal["literal"]
    secondary_span: Literal["claim"]
    token_artifact_role: Literal["tokenizer_specific_derived"]
    activation_alignment: Literal["post_token_h_t"]


class DerivedMaskConfig(StrictModel):
    hallucination_core: tuple[str, ...]
    error_any_excludes: tuple[str, ...]
    include_local_falsehood_mask: Literal[True]
    include_off_task_branch_mask: Literal[True]


class OmissionConfig(StrictModel):
    construct_pure_omission_in_primary_localization: Literal[False]
    insert_missing_sentinel: Literal[False]
    allowed_diagnostic_targets: tuple[str, ...]


class LabelsConfig(StrictModel):
    schema_version: Literal["1.0"]
    multi_label: Literal[True]
    orthogonal_axes: tuple[str, ...]
    canonical_annotation: CanonicalAnnotationConfig
    semantic_types: tuple[LabelDefinition, ...]
    editing_subtypes: tuple[EditSubtypeDefinition, ...]
    causal_roles: tuple[str, ...]
    evidence_relations: tuple[str, ...]
    derived_masks: DerivedMaskConfig
    omission: OmissionConfig

    @model_validator(mode="after")
    def validate_frozen_taxonomy(self) -> LabelsConfig:
        semantic_ids = tuple(item.id for item in self.semantic_types)
        semantic_names = tuple(item.name for item in self.semantic_types)
        subtype_codes = tuple(item.code for item in self.editing_subtypes)
        subtype_names = tuple(item.name for item in self.editing_subtypes)
        if semantic_ids != tuple(range(8)) or semantic_names != FROZEN_SEMANTIC_TYPES:
            raise ValueError("semantic taxonomy must contain the frozen IDs 0-7")
        if subtype_codes != tuple(f"E{index:02d}" for index in range(1, 16)):
            raise ValueError("editing subtype codes must be E01-E15")
        if subtype_names != FROZEN_EDIT_SUBTYPES:
            raise ValueError("editing subtype names do not match the frozen taxonomy")
        if self.causal_roles != FROZEN_CAUSAL_ROLES:
            raise ValueError("causal roles do not match the frozen taxonomy")
        if self.evidence_relations != FROZEN_EVIDENCE_RELATIONS:
            raise ValueError("evidence relations do not match the frozen taxonomy")
        if self.orthogonal_axes != (
            "hallucination_presence",
            "semantic_type",
            "editing_subtype",
            "causal_role",
            "evidence_relation",
        ):
            raise ValueError("orthogonal label axes are incomplete or reordered")
        if self.derived_masks.hallucination_core != ("CONTRADICTION", "UNSUPPORTED"):
            raise ValueError("hallucination_core must be CONTRADICTION OR UNSUPPORTED")
        if self.derived_masks.error_any_excludes != ("UNVERIFIABLE",):
            raise ValueError("error_any must exclude UNVERIFIABLE")
        return self


class OperatorFamilyConfig(StrictModel):
    supported_policies: tuple[PolicyName, ...]
    allowed_candidate_sources: tuple[Literal["RULE", "RDKIT", "LLM", "HYBRID"], ...]


class QuotaItem(StrictModel):
    family: str
    target_per_50: Annotated[int, Field(gt=0, le=50)]


class CandidateGenerationConfig(StrictModel):
    target_mix_percent: Mapping[Literal["RULE_RDKIT", "LLM", "HYBRID"], int]
    candidates_per_recipe_min: Annotated[int, Field(ge=1)]
    candidates_per_recipe_max: Annotated[int, Field(ge=1)]
    minimum_valid_structural_h_percent: Annotated[int, Field(ge=0, le=100)]
    minimum_llm_structural_participation_percent: Annotated[int, Field(ge=0, le=100)]
    accept_first_passing_candidate: Literal[False]

    @field_validator("target_mix_percent")
    @classmethod
    def freeze_target_mix(
        cls,
        value: Mapping[Literal["RULE_RDKIT", "LLM", "HYBRID"], int],
    ) -> Mapping[Literal["RULE_RDKIT", "LLM", "HYBRID"], int]:
        return _frozen_mapping(value)

    @field_serializer("target_mix_percent")
    def serialize_target_mix(
        self, value: Mapping[Literal["RULE_RDKIT", "LLM", "HYBRID"], int]
    ) -> dict[str, int]:
        return dict(value)

    @model_validator(mode="after")
    def validate_candidate_targets(self) -> CandidateGenerationConfig:
        if self.target_mix_percent != {"RULE_RDKIT": 40, "LLM": 40, "HYBRID": 20}:
            raise ValueError("candidate target mix must be 40/40/20")
        if self.candidates_per_recipe_min > self.candidates_per_recipe_max:
            raise ValueError("candidate pool minimum cannot exceed maximum")
        if self.minimum_valid_structural_h_percent < 80:
            raise ValueError("at least 80% of structural H must be chemically valid")
        if self.minimum_llm_structural_participation_percent < 50:
            raise ValueError("LLM must materially participate in at least half of structural H")
        return self


class FallbackConfig(StrictModel):
    within_operator_family_first: Literal[True]
    same_policy_only: Literal[True]
    record_quota_deviation: Literal[True]
    prohibit_silent_phenotype_change: Literal[True]


class OperatorsConfig(StrictModel):
    schema_version: Literal["1.0"]
    policies: tuple[PolicyName, ...]
    families: Mapping[str, OperatorFamilyConfig]
    quota_bucket_mappings: Mapping[str, tuple[str, ...]]
    quotas_per_subtask_policy: Mapping[PolicyName, tuple[QuotaItem, ...]]
    candidate_generation: CandidateGenerationConfig
    fallback: FallbackConfig

    @field_validator("families")
    @classmethod
    def freeze_families(
        cls, value: Mapping[str, OperatorFamilyConfig]
    ) -> Mapping[str, OperatorFamilyConfig]:
        return _frozen_mapping(value)

    @field_validator("quota_bucket_mappings")
    @classmethod
    def freeze_quota_bucket_mappings(
        cls, value: Mapping[str, tuple[str, ...]]
    ) -> Mapping[str, tuple[str, ...]]:
        return _frozen_mapping(value)

    @field_validator("quotas_per_subtask_policy")
    @classmethod
    def freeze_quotas(
        cls, value: Mapping[PolicyName, tuple[QuotaItem, ...]]
    ) -> Mapping[PolicyName, tuple[QuotaItem, ...]]:
        return _frozen_mapping(value)

    @field_serializer("families")
    def serialize_families(
        self, value: Mapping[str, OperatorFamilyConfig]
    ) -> dict[str, OperatorFamilyConfig]:
        return dict(value)

    @field_serializer("quota_bucket_mappings")
    def serialize_quota_bucket_mappings(
        self, value: Mapping[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        return dict(value)

    @field_serializer("quotas_per_subtask_policy")
    def serialize_quotas(
        self, value: Mapping[PolicyName, tuple[QuotaItem, ...]]
    ) -> dict[str, tuple[QuotaItem, ...]]:
        return dict(value)

    @model_validator(mode="after")
    def validate_operator_invariants(self) -> OperatorsConfig:
        if self.policies != FROZEN_POLICIES:
            raise ValueError("operator policies must match the frozen policy order")
        actual_compatibility = {
            name: (family.supported_policies, family.allowed_candidate_sources)
            for name, family in self.families.items()
        }
        if actual_compatibility != FROZEN_OPERATOR_COMPATIBILITY:
            raise ValueError("operator compatibility matrix differs from the frozen plan")
        if self.quota_bucket_mappings != FROZEN_QUOTA_BUCKET_MAPPINGS:
            raise ValueError("quota bucket mappings differ from the frozen plan")
        actual_quotas = {
            policy: tuple((item.family, item.target_per_50) for item in items)
            for policy, items in self.quotas_per_subtask_policy.items()
        }
        if actual_quotas != FROZEN_OPERATOR_QUOTAS:
            raise ValueError("operator-family quotas differ from the frozen plan")
        for policy, items in self.quotas_per_subtask_policy.items():
            for item in items:
                compatible_families = self.quota_bucket_mappings.get(item.family)
                if not compatible_families:
                    raise ValueError(f"quota bucket {item.family!r} has no family mapping")
                for family_name in compatible_families:
                    family = self.families.get(family_name)
                    if family is None:
                        raise ValueError(
                            f"quota bucket {item.family!r} maps to unknown family {family_name!r}"
                        )
                    if policy not in family.supported_policies:
                        raise ValueError(
                            f"quota bucket {item.family!r} maps to family {family_name!r} "
                            f"that does not support {policy}"
                        )
        return self


class ProviderConfig(StrictModel):
    name: Literal["poe"]
    base_url: Literal["https://api.poe.com/v1"]
    api_key_env: Literal["POE_API_KEY"]
    model_id: Literal["gpt-5.4-mini"]
    bot_name: Literal["GPT-5.4-Mini"]
    primary_transport: Literal["responses"]
    fallback_transports: tuple[Literal["chat_completions", "fastapi_poe"], ...]


class ModelDiscoveryConfig(StrictModel):
    endpoint: Literal["/models"]
    require_model_id: Literal["gpt-5.4-mini"]
    require_endpoints: tuple[str, ...]
    require_features: tuple[str, ...]
    fail_closed: Literal[True]
    cache_ttl_hours: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_required_capabilities(self) -> ModelDiscoveryConfig:
        if self.require_endpoints != ("/v1/responses", "/v1/chat/completions"):
            raise ValueError("Poe Responses and Chat Completions endpoints are both required")
        if self.require_features != ("tools",):
            raise ValueError("Poe model discovery must require tool capability")
        return self


class ProposalConfig(StrictModel):
    model_id: Literal["gpt-5.4-mini"]
    reasoning_effort: Literal["medium"]
    retry_reasoning_effort: Literal["high"]
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    parallel_tool_calls: Literal[False]
    web_search: Literal[False]
    max_attempts: Literal[3]
    candidates_per_attempt: Literal[5]
    max_tool_calls: Literal[6]
    structured_output_schema: Literal["proposal_v1"]
    local_schema_validation: Literal[True]


class LLMRendererConfig(StrictModel):
    model_id: Literal["gpt-5.4-mini"]
    reasoning_effort: Literal["low"]
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    label_blind: Literal[True]
    output_mode: Literal["placeholder_template"]
    local_schema_validation: Literal[True]


class RateLimitConfig(StrictModel):
    requests_per_minute: Annotated[int, Field(gt=0, le=450)]
    max_concurrency: Annotated[int, Field(gt=0)]
    retry_base_ms: Annotated[int, Field(gt=0)]
    respect_retry_after: Literal[True]


class UsageConfig(StrictModel):
    check_balance_before_build: Literal[True]
    balance_endpoint: Literal["https://api.poe.com/usage/current_balance"]
    history_endpoint: Literal["https://api.poe.com/usage/points_history"]
    export_private_ledger: Literal[True]


class CacheConfig(StrictModel):
    enabled: Literal[True]
    content_addressed: Literal[True]
    replay_only_for_release: Literal[True]
    include_model_catalog_hash: Literal[True]


class LLMConfig(StrictModel):
    schema_version: Literal["1.0"]
    provider: ProviderConfig
    model_discovery: ModelDiscoveryConfig
    proposal: ProposalConfig
    renderer: LLMRendererConfig
    rate_limit: RateLimitConfig
    usage: UsageConfig
    cache: CacheConfig

    @model_validator(mode="after")
    def validate_single_frozen_model(self) -> LLMConfig:
        configured_models = {
            self.provider.model_id,
            self.model_discovery.require_model_id,
            self.proposal.model_id,
            self.renderer.model_id,
        }
        if configured_models != {"gpt-5.4-mini"}:
            raise ValueError("all Poe roles must use the frozen gpt-5.4-mini model")
        if self.provider.fallback_transports != ("chat_completions", "fastapi_poe"):
            raise ValueError("Poe fallback transport order is frozen")
        return self


class DetectorTemplateConfig(StrictModel):
    field_order: tuple[str, ...]
    delimiters: Mapping[str, str]
    include_gt_smiles: Literal[False]
    include_reference_only_metadata: Literal[False]

    @field_validator("delimiters")
    @classmethod
    def freeze_delimiters(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _frozen_mapping(value)

    @field_serializer("delimiters")
    def serialize_delimiters(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("field_order")
    @classmethod
    def validate_field_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != FROZEN_FIELD_ORDER:
            raise ValueError("rendering field order does not match detector contract")
        return value


class NaturalRendererConfig(StrictModel):
    backends: tuple[Literal["rule", "poe_placeholder"], ...]
    label_blind: Literal[True]
    output_mode: Literal["placeholder_template"]
    locked_slots_are_immutable: Literal[True]
    styles: tuple[str, ...]


class DeterministicRendererConfig(StrictModel):
    formal: Literal[True]
    numeric_claims: Literal[True]
    final_answer: Literal[True]
    placeholder_fill: Literal[True]


class SpanRenderingConfig(StrictModel):
    construct_offsets_during_render: Literal[True]
    allow_text_find_localization: Literal[False]
    interval: Literal["half_open"]
    emit_literal_span: Literal[True]
    emit_claim_span: Literal[True]


class LeakageScanConfig(StrictModel):
    prohibited_inputs: tuple[str, ...]
    prohibited_phrases: tuple[str, ...]
    fail_closed: Literal[True]


class RenderingConfig(StrictModel):
    schema_version: Literal["1.0"]
    detector_template: DetectorTemplateConfig
    natural_renderer: NaturalRendererConfig
    deterministic_renderer: DeterministicRendererConfig
    spans: SpanRenderingConfig
    leakage_scan: LeakageScanConfig

    @model_validator(mode="after")
    def validate_renderer_is_label_blind(self) -> RenderingConfig:
        prohibited = set(self.leakage_scan.prohibited_inputs)
        required = {"gt_smiles", "hallucination_label", "oracle_state", "operator_correctness"}
        if not required <= prohibited:
            raise ValueError("renderer prohibited_inputs must include all private label/oracle fields")
        if set(self.detector_template.delimiters) != set(FROZEN_FIELD_ORDER):
            raise ValueError("every detector field must have exactly one delimiter")
        expected_delimiters = {
            "indexed_smiles": "<MOLECULE>",
            "instruction": "<INSTRUCTION>",
            "reasoning_chain": "<REASONING>",
            "final_answer": "<FINAL_ANSWER>",
        }
        if self.detector_template.delimiters != expected_delimiters:
            raise ValueError("detector delimiters do not match the frozen template")
        if not self.natural_renderer.styles:
            raise ValueError("at least one rendering style must be frozen")
        return self
