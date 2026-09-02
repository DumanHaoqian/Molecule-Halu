"""Canonical character annotations and tokenizer-specific label artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from .enums import (
    CausalRole,
    EditErrorSubtype,
    EvidenceRelation,
    HallucinationType,
    SegmentKind,
)
from .state_dag import FrozenMap, freeze_string_mapping


@dataclass(frozen=True, slots=True, order=True)
class CharSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.start, self.end)):
            raise TypeError("CharSpan boundaries must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("CharSpan must be a non-empty half-open interval")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: CharSpan) -> bool:
        return max(self.start, other.start) < min(self.end, other.end)


@dataclass(frozen=True, slots=True)
class ClaimLabel:
    semantic_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]
    evidence_relations: frozenset[EvidenceRelation]
    causal_role: CausalRole | None
    root_event_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_types", frozenset(self.semantic_types))
        object.__setattr__(self, "edit_subtypes", frozenset(self.edit_subtypes))
        object.__setattr__(self, "evidence_relations", frozenset(self.evidence_relations))
        if any(type(value) is not HallucinationType for value in self.semantic_types):
            raise TypeError("ClaimLabel semantic_types must contain HallucinationType values")
        if any(type(value) is not EditErrorSubtype for value in self.edit_subtypes):
            raise TypeError("ClaimLabel edit_subtypes must contain EditErrorSubtype values")
        if any(type(value) is not EvidenceRelation for value in self.evidence_relations):
            raise TypeError("ClaimLabel evidence_relations must contain EvidenceRelation values")
        if self.causal_role is not None and type(self.causal_role) is not CausalRole:
            raise TypeError("ClaimLabel causal_role must be a CausalRole or None")
        if self.root_event_id is not None and type(self.root_event_id) is not str:
            raise TypeError("ClaimLabel root_event_id must be a string or None")
        if not self.semantic_types:
            raise ValueError("ClaimLabel axes cannot omit semantic_types")
        if HallucinationType.UNVERIFIABLE in self.semantic_types:
            if self.semantic_types != frozenset({HallucinationType.UNVERIFIABLE}):
                raise ValueError("UNVERIFIABLE must be mutually exclusive with adjudicated types")
            if self.edit_subtypes or self.causal_role is not None or self.root_event_id is not None:
                raise ValueError(
                    "UNVERIFIABLE labels cannot carry editing, causal, or root error axes"
                )
        elif (
            not self.edit_subtypes
            or not self.evidence_relations
            or self.causal_role is None
            or not self.root_event_id
        ):
            raise ValueError("adjudicated ClaimLabel axes cannot be empty")


@dataclass(frozen=True, slots=True)
class CharAnnotation:
    span_id: str
    component: SegmentKind
    step_index: int | None
    state_or_edge_id: str
    literal_span: CharSpan
    claim_span: CharSpan
    semantic_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]
    evidence_relations: frozenset[EvidenceRelation]
    causal_role: CausalRole | None
    root_span_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_types", frozenset(self.semantic_types))
        object.__setattr__(self, "edit_subtypes", frozenset(self.edit_subtypes))
        object.__setattr__(self, "evidence_relations", frozenset(self.evidence_relations))
        for value, name in (
            (self.span_id, "span_id"),
            (self.state_or_edge_id, "state_or_edge_id"),
        ):
            if type(value) is not str:
                raise TypeError(f"CharAnnotation {name} must be a string")
        if type(self.component) is not SegmentKind:
            raise TypeError("CharAnnotation component must be a SegmentKind")
        if self.step_index is not None and (
            type(self.step_index) is not int
        ):
            raise TypeError("CharAnnotation step_index must be an integer or None")
        if type(self.literal_span) is not CharSpan or type(self.claim_span) is not CharSpan:
            raise TypeError("CharAnnotation spans must be CharSpan values")
        if any(type(value) is not HallucinationType for value in self.semantic_types):
            raise TypeError("CharAnnotation semantic_types must contain HallucinationType values")
        if any(type(value) is not EditErrorSubtype for value in self.edit_subtypes):
            raise TypeError("CharAnnotation edit_subtypes must contain EditErrorSubtype values")
        if any(type(value) is not EvidenceRelation for value in self.evidence_relations):
            raise TypeError("CharAnnotation evidence_relations must contain EvidenceRelation values")
        if self.causal_role is not None and type(self.causal_role) is not CausalRole:
            raise TypeError("CharAnnotation causal_role must be a CausalRole or None")
        if self.root_span_id is not None and type(self.root_span_id) is not str:
            raise TypeError("CharAnnotation root_span_id must be a string or None")
        if not self.span_id or not self.state_or_edge_id:
            raise ValueError("CharAnnotation IDs cannot be empty")
        if self.component not in {SegmentKind.REASONING, SegmentKind.FINAL_ANSWER}:
            raise ValueError("positive annotations may only occur in reasoning or final answer")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("CharAnnotation step_index cannot be negative")
        if not (
            self.claim_span.start <= self.literal_span.start
            and self.literal_span.end <= self.claim_span.end
        ):
            raise ValueError("CharAnnotation literal_span must be contained by claim_span")
        if not self.semantic_types:
            raise ValueError("CharAnnotation axes cannot omit semantic_types")
        if HallucinationType.UNVERIFIABLE in self.semantic_types:
            if self.semantic_types != frozenset({HallucinationType.UNVERIFIABLE}):
                raise ValueError("UNVERIFIABLE must be mutually exclusive with adjudicated types")
            if self.edit_subtypes or self.causal_role is not None or self.root_span_id is not None:
                raise ValueError(
                    "UNVERIFIABLE annotations cannot carry editing, causal, or root error axes"
                )
        elif (
            not self.edit_subtypes
            or not self.evidence_relations
            or self.causal_role is None
            or not self.root_span_id
        ):
            raise ValueError("adjudicated CharAnnotation axes cannot be empty")
        if self.causal_role in {CausalRole.ROOT, CausalRole.TERMINAL}:
            if self.root_span_id != self.span_id:
                raise ValueError("root/terminal annotations must identify themselves as root")
        elif self.causal_role is not None and self.root_span_id == self.span_id:
            raise ValueError("propagated annotations must refer to a distinct root span")
        if self.causal_role is CausalRole.TERMINAL and self.component is not SegmentKind.FINAL_ANSWER:
            raise ValueError("TERMINAL annotations must be in the final answer")
        if self.causal_role is CausalRole.TERMINAL:
            if self.state_or_edge_id != "final_answer":
                raise ValueError("TERMINAL annotations must target final_answer")
            if EditErrorSubtype.FINAL_ANSWER_IDENTITY not in self.edit_subtypes:
                raise ValueError("TERMINAL annotations must carry FINAL_ANSWER_IDENTITY")
        if self.causal_role is CausalRole.ROOT and self.state_or_edge_id == "final_answer":
            raise ValueError("an independent final_answer error must use the TERMINAL causal role")

    @property
    def is_adjudicated(self) -> bool:
        return any(label is not HallucinationType.UNVERIFIABLE for label in self.semantic_types)


@dataclass(frozen=True, slots=True)
class TokenizerFingerprint:
    tokenizer_name: str
    tokenizer_revision: str
    tokenizer_vocab_hash: str
    special_token_config: Mapping[str, object]
    normalization_config: Mapping[str, object]

    def __post_init__(self) -> None:
        for value, name in (
            (self.tokenizer_name, "tokenizer_name"),
            (self.tokenizer_revision, "tokenizer_revision"),
            (self.tokenizer_vocab_hash, "tokenizer_vocab_hash"),
        ):
            if type(value) is not str:
                raise TypeError(f"TokenizerFingerprint {name} must be a string")
        if not isinstance(self.special_token_config, Mapping) or not isinstance(
            self.normalization_config, Mapping
        ):
            raise TypeError("TokenizerFingerprint configs must be mappings")
        object.__setattr__(
            self,
            "special_token_config",
            freeze_string_mapping(
                self.special_token_config,
                name="TokenizerFingerprint special_token_config",
            ),
        )
        object.__setattr__(
            self,
            "normalization_config",
            freeze_string_mapping(
                self.normalization_config,
                name="TokenizerFingerprint normalization_config",
            ),
        )
        if not self.tokenizer_name or not self.tokenizer_revision or not self.tokenizer_vocab_hash:
            raise ValueError("TokenizerFingerprint identity fields cannot be empty")


def _as_binary_mask(values: tuple[int, ...], *, name: str, length: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) != length:
        raise ValueError(f"{name} length {len(values)} != token length {length}")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain integer 0/1 values")
    if any(value not in {0, 1} for value in values):
        raise ValueError(f"{name} must contain only 0/1")
    return values


@dataclass(frozen=True, slots=True)
class TokenLabelSet:
    activation_alignment: str
    tokenizer_fingerprint: TokenizerFingerprint
    serialized_text_sha256: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    segment_ids: tuple[SegmentKind, ...]
    evaluation_mask: tuple[int, ...]
    hallucination_core_mask: tuple[int, ...]
    error_any_mask: tuple[int, ...]
    semantic_type_masks: Mapping[HallucinationType, tuple[int, ...]]
    edit_subtype_masks: Mapping[EditErrorSubtype, tuple[int, ...]]
    causal_role_masks: Mapping[CausalRole, tuple[int, ...]]
    local_falsehood_mask: tuple[int, ...]
    off_task_branch_mask: tuple[int, ...]
    reasoning_mask: tuple[int, ...]
    answer_mask: tuple[int, ...]
    boundary_ambiguous_mask: tuple[int, ...]
    error_char_fraction: tuple[float, ...]
    matched_target_span: CharSpan | None = None

    def __post_init__(self) -> None:
        if type(self.activation_alignment) is not str:
            raise TypeError("TokenLabelSet activation_alignment must be a string")
        if self.activation_alignment != "post_token_h_t":
            raise ValueError("TokenLabelSet only supports post_token_h_t alignment")
        if type(self.tokenizer_fingerprint) is not TokenizerFingerprint:
            raise TypeError("TokenLabelSet tokenizer_fingerprint must be a TokenizerFingerprint")
        if type(self.serialized_text_sha256) is not str:
            raise TypeError("TokenLabelSet serialized_text_sha256 must be a string")
        if not self.serialized_text_sha256:
            raise ValueError("TokenLabelSet serialized_text_sha256 cannot be empty")
        input_ids = tuple(self.input_ids)
        if any(type(value) is not int for value in input_ids):
            raise TypeError("TokenLabelSet input_ids must contain integers")
        if any(value < 0 for value in input_ids):
            raise ValueError("TokenLabelSet input_ids must be non-negative")
        token_count = len(input_ids)
        if token_count == 0:
            raise ValueError("TokenLabelSet cannot be empty")
        raw_offsets = tuple(self.offset_mapping)
        offsets: list[tuple[int, int]] = []
        for pair in raw_offsets:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError("each token offset must be a two-integer sequence")
            start, end = pair
            if any(type(value) is not int for value in (start, end)):
                raise TypeError("token offset boundaries must be integers")
            offsets.append((start, end))
        segments = tuple(self.segment_ids)
        if any(type(segment) is not SegmentKind for segment in segments):
            raise TypeError("TokenLabelSet segment_ids must contain SegmentKind values")
        if self.matched_target_span is not None and type(self.matched_target_span) is not CharSpan:
            raise TypeError("TokenLabelSet matched_target_span must be a CharSpan or None")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "offset_mapping", tuple(offsets))
        object.__setattr__(self, "segment_ids", segments)
        if len(offsets) != token_count or len(segments) != token_count:
            raise ValueError("offset_mapping and segment_ids must match token length")
        for start, end in self.offset_mapping:
            if start < 0 or end < start:
                raise ValueError("token offsets must be valid half-open intervals")

        binary_fields = (
            "attention_mask",
            "evaluation_mask",
            "hallucination_core_mask",
            "error_any_mask",
            "local_falsehood_mask",
            "off_task_branch_mask",
            "reasoning_mask",
            "answer_mask",
            "boundary_ambiguous_mask",
        )
        for name in binary_fields:
            object.__setattr__(
                self,
                name,
                _as_binary_mask(getattr(self, name), name=name, length=token_count),
            )

        if not isinstance(self.semantic_type_masks, Mapping):
            raise TypeError("TokenLabelSet semantic_type_masks must be a mapping")
        if not isinstance(self.edit_subtype_masks, Mapping):
            raise TypeError("TokenLabelSet edit_subtype_masks must be a mapping")
        if not isinstance(self.causal_role_masks, Mapping):
            raise TypeError("TokenLabelSet causal_role_masks must be a mapping")
        if any(type(key) is not HallucinationType for key in self.semantic_type_masks):
            raise TypeError("semantic_type_masks keys must be HallucinationType values")
        if any(type(key) is not EditErrorSubtype for key in self.edit_subtype_masks):
            raise TypeError("edit_subtype_masks keys must be EditErrorSubtype values")
        if any(type(key) is not CausalRole for key in self.causal_role_masks):
            raise TypeError("causal_role_masks keys must be CausalRole values")

        semantic_masks = FrozenMap(
            {
                key: _as_binary_mask(value, name=f"semantic_type_masks[{key.name}]", length=token_count)
                for key, value in self.semantic_type_masks.items()
            }
        )
        edit_masks = FrozenMap(
            {
                key: _as_binary_mask(value, name=f"edit_subtype_masks[{key.name}]", length=token_count)
                for key, value in self.edit_subtype_masks.items()
            }
        )
        role_masks = FrozenMap(
            {
                key: _as_binary_mask(value, name=f"causal_role_masks[{key.name}]", length=token_count)
                for key, value in self.causal_role_masks.items()
            }
        )
        object.__setattr__(self, "semantic_type_masks", semantic_masks)
        object.__setattr__(self, "edit_subtype_masks", edit_masks)
        object.__setattr__(self, "causal_role_masks", role_masks)
        if set(semantic_masks) != set(HallucinationType):
            raise ValueError("semantic_type_masks must include every HallucinationType")
        if set(edit_masks) != set(EditErrorSubtype):
            raise ValueError("edit_subtype_masks must include every EditErrorSubtype")
        if set(role_masks) != set(CausalRole):
            raise ValueError("causal_role_masks must include every CausalRole")

        fractions = tuple(self.error_char_fraction)
        if any(
            type(value) not in {int, float} for value in fractions
        ):
            raise TypeError("error_char_fraction must contain numeric values")
        if len(fractions) != token_count or any(
            not isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions
        ):
            raise ValueError("error_char_fraction must have one [0,1] value per token")
        object.__setattr__(self, "error_char_fraction", fractions)

        contradiction = semantic_masks[HallucinationType.CONTRADICTION]
        unsupported = semantic_masks[HallucinationType.UNSUPPORTED]
        expected_core = tuple(int(left or right) for left, right in zip(contradiction, unsupported))
        if self.hallucination_core_mask != expected_core:
            raise ValueError("hallucination_core_mask must be CONTRADICTION OR UNSUPPORTED")
        adjudicated = tuple(
            semantic_masks[label]
            for label in HallucinationType
            if label is not HallucinationType.UNVERIFIABLE
        )
        expected_any = tuple(int(any(mask[index] for mask in adjudicated)) for index in range(token_count))
        if self.error_any_mask != expected_any:
            raise ValueError("error_any_mask must OR all semantic types except UNVERIFIABLE")

        for index, segment in enumerate(self.segment_ids):
            visible_segment = segment in {
                SegmentKind.SOURCE,
                SegmentKind.INSTRUCTION,
                SegmentKind.REASONING,
                SegmentKind.FINAL_ANSWER,
            }
            if visible_segment and self.attention_mask[index] != 1:
                raise ValueError("visible text tokens must have attention_mask=1")
            if visible_segment and self.offset_mapping[index][0] == self.offset_mapping[index][1]:
                raise ValueError("visible text tokens must have non-empty character offsets")
            should_evaluate = (
                segment in {SegmentKind.REASONING, SegmentKind.FINAL_ANSWER}
                and self.attention_mask[index] == 1
            )
            if self.evaluation_mask[index] != int(should_evaluate):
                raise ValueError("evaluation_mask does not match token segment")
            if self.reasoning_mask[index] != int(segment is SegmentKind.REASONING):
                raise ValueError("reasoning_mask does not match token segment")
            if self.answer_mask[index] != int(segment is SegmentKind.FINAL_ANSWER):
                raise ValueError("answer_mask does not match token segment")
            semantic_positive = any(mask[index] for mask in semantic_masks.values())
            if (
                semantic_masks[HallucinationType.UNVERIFIABLE][index]
                and self.error_any_mask[index]
            ):
                raise ValueError(
                    "UNVERIFIABLE must be mutually exclusive with adjudicated token types"
                )
            edit_positive = any(mask[index] for mask in edit_masks.values())
            active_roles = tuple(role for role, mask in role_masks.items() if mask[index])
            all_positive = any(
                (
                    semantic_positive,
                    edit_positive,
                    bool(active_roles),
                    bool(self.hallucination_core_mask[index]),
                    bool(self.error_any_mask[index]),
                    bool(self.local_falsehood_mask[index]),
                    bool(self.off_task_branch_mask[index]),
                    bool(self.boundary_ambiguous_mask[index]),
                    fractions[index] > 0,
                )
            )
            if all_positive and not self.evaluation_mask[index]:
                raise ValueError("positive labels may only occur on evaluated, attended tokens")
            if edit_positive != bool(self.error_any_mask[index]):
                raise ValueError("editing subtype masks must cover exactly the error_any tokens")
            if len(active_roles) != self.error_any_mask[index]:
                raise ValueError("each error_any token must have exactly one causal role")
            if CausalRole.TERMINAL in active_roles and segment is not SegmentKind.FINAL_ANSWER:
                raise ValueError("TERMINAL causal labels may only occur in the final answer")
            if (
                CausalRole.TERMINAL in active_roles
                and not edit_masks[EditErrorSubtype.FINAL_ANSWER_IDENTITY][index]
            ):
                raise ValueError("TERMINAL causal labels must carry FINAL_ANSWER_IDENTITY")

            expected_local = bool(
                set(active_roles)
                & {CausalRole.ROOT, CausalRole.PROPAGATED_FALSE, CausalRole.TERMINAL}
            )
            expected_off_task = CausalRole.PROPAGATED_CONDITIONAL in active_roles
            if bool(self.local_falsehood_mask[index]) != expected_local:
                raise ValueError("local_falsehood_mask does not match causal_role_masks")
            if bool(self.off_task_branch_mask[index]) != expected_off_task:
                raise ValueError("off_task_branch_mask does not match causal_role_masks")
            if bool(fractions[index]) != semantic_positive:
                raise ValueError("error_char_fraction must be positive exactly on semantic spans")
            if self.boundary_ambiguous_mask[index] and not semantic_positive:
                raise ValueError("boundary_ambiguous_mask must be a semantic-label subset")

        if self.matched_target_span is not None and not any(
            self.evaluation_mask[index]
            and max(
                0,
                min(self.offset_mapping[index][1], self.matched_target_span.end)
                - max(self.offset_mapping[index][0], self.matched_target_span.start),
            )
            > 0
            for index in range(token_count)
        ):
            raise ValueError("matched_target_span must overlap at least one evaluated token")

    @property
    def has_positive_labels(self) -> bool:
        """Whether any canonical error/uncertainty label is positive."""

        mask_groups = (
            self.semantic_type_masks.values(),
            self.edit_subtype_masks.values(),
            self.causal_role_masks.values(),
        )
        return any(any(mask) for group in mask_groups for mask in group) or any(
            any(mask)
            for mask in (
                self.hallucination_core_mask,
                self.error_any_mask,
                self.local_falsehood_mask,
                self.off_task_branch_mask,
                self.boundary_ambiguous_mask,
            )
        ) or any(fraction > 0 for fraction in self.error_char_fraction)

    @property
    def positive_label_indices(self) -> tuple[int, ...]:
        """Token positions carrying any semantic label, including UNVERIFIABLE."""

        return tuple(
            index
            for index in range(len(self.input_ids))
            if any(mask[index] for mask in self.semantic_type_masks.values())
        )
