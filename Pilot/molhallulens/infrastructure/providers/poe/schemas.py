"""Strict local schemas for the Poe proposal-agent boundary.

Poe structured-output and function-calling flags are transport hints, not a
trust boundary.  Every request, response, and tool call therefore crosses the
frozen Pydantic-v2 models in this module before any semantic processing.  This
module contains no network client and executes no chemistry tool.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from molhallulens.core import CandidateSourceType

if TYPE_CHECKING:
    from molhallulens.modules.release.manifest import VerifiedSplitManifest
    from molhallulens.modules.error_planning import CandidateRequest


PROPOSAL_SCHEMA_VERSION = "1.0"
PROPOSAL_VERSION = "1.0"
FROZEN_DATASET_VERSION = "pilot_v1"
FROZEN_GLOBAL_SEED = 20260828
FROZEN_SPLIT_SEED = 8347206628578381721
PROPOSAL_CANDIDATE_MIN = 3
PROPOSAL_CANDIDATE_MAX = 5


def _validated_text(value: str) -> str:
    if not value or value != value.strip() or "\0" in value:
        raise ValueError("text must be non-empty, trimmed, and NUL-free")
    return value


def _validated_single_line(value: str) -> str:
    value = _validated_text(value)
    if "\r" in value or "\n" in value:
        raise ValueError("text must be a single line")
    return value


NonEmptyText = Annotated[
    str, Field(min_length=1, max_length=8192), AfterValidator(_validated_text)
]
SingleLineText = Annotated[
    str,
    Field(min_length=1, max_length=1024),
    AfterValidator(_validated_single_line),
]
Identifier = Annotated[
    str,
    Field(min_length=1, max_length=512, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]*$"),
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]
NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]

PolicyName = Literal["LOCAL", "PARTIAL", "FULL_CF", "TERMINAL"]
SplitName = Literal["train", "validation", "test"]
ProposalSourceName = Literal["LLM", "HYBRID"]
EditingFamilyName = Literal["add", "delete", "substitute"]
BondTypeName = Literal["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


class StrictFrozenModel(BaseModel):
    """Shared recursive trust-boundary policy."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


def derive_proposal_seed(
    *,
    global_seed: int,
    dataset_version: str,
    origin_id: str,
    operator_id: str,
    policy: PolicyName,
    variant_index: int,
) -> int:
    """Reuse T025's six-field NUL-delimited SHA256 seed contract."""

    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a non-negative exact integer")
    if type(variant_index) is not int or variant_index < 0:
        raise ValueError("variant_index must be a non-negative exact integer")
    for value, name in (
        (dataset_version, "dataset_version"),
        (origin_id, "origin_id"),
        (operator_id, "operator_id"),
        (policy, "policy"),
    ):
        if type(value) is not str or not value or "\0" in value:
            raise ValueError(f"{name} must be non-empty NUL-free text")
    payload = "\0".join(
        (
            str(global_seed),
            dataset_version,
            origin_id,
            operator_id,
            policy,
            str(variant_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class ProposalManifestIdentity(StrictFrozenModel):
    """Immutable identity of the verified T029 split manifest."""

    dataset_version: Literal["pilot_v1"]
    split_seed: Literal[8347206628578381721]
    manifest_sha256: Sha256Hex
    source_origin_audit_sha256: Sha256Hex
    source_split_report_sha256: Sha256Hex


class ProposalConstraints(StrictFrozenModel):
    """Closed proposal allow-list; absent optional matches are unconstrained."""

    sanitized: Literal[True] = True
    not_equivalent_to_reference: Literal[True] = True
    same_attachment_element: StrictBool | None = None
    match_heavy_count: StrictBool | None = None
    match_ring_count: StrictBool | None = None


class ProposalRequest(StrictFrozenModel):
    """One operator/root/split-bound request sent to the Poe proposal agent."""

    schema_version: Literal["1.0"] = PROPOSAL_SCHEMA_VERSION
    request_id: Identifier
    origin_id: Identifier
    operator_id: Identifier
    propagation: PolicyName
    candidate_source_mode: ProposalSourceName
    target_root: Identifier
    constraints: ProposalConstraints
    global_seed: Literal[20260828]
    dataset_version: Literal["pilot_v1"]
    variant_index: NonNegativeStrictInt
    derived_seed: NonNegativeStrictInt
    split: SplitName
    manifest_identity: ProposalManifestIdentity

    @model_validator(mode="after")
    def validate_seed_and_manifest_binding(self) -> ProposalRequest:
        if self.manifest_identity.dataset_version != self.dataset_version:
            raise ValueError("manifest and proposal dataset versions differ")
        expected = derive_proposal_seed(
            global_seed=self.global_seed,
            dataset_version=self.dataset_version,
            origin_id=self.origin_id,
            operator_id=self.operator_id,
            policy=self.propagation,
            variant_index=self.variant_index,
        )
        if self.derived_seed != expected:
            raise ValueError("derived_seed does not match the frozen six-field SHA256")
        return self


class ProposedBondEdit(StrictFrozenModel):
    operation: Literal["BREAK", "FORM"]
    begin_atom: PositiveStrictInt
    end_atom: PositiveStrictInt
    bond_type: BondTypeName

    @model_validator(mode="after")
    def validate_canonical_endpoints(self) -> ProposedBondEdit:
        if self.begin_atom >= self.end_atom:
            raise ValueError("bond endpoints must be distinct and canonically ordered")
        return self


class ProposalReplacement(StrictFrozenModel):
    """Exactly one primary scalar plus optional SMILES attachment identity."""

    smiles: NonEmptyText | None = None
    atom_index: PositiveStrictInt | None = None
    integer: StrictInt | None = None
    text: NonEmptyText | None = None
    attachment_atom: NonNegativeStrictInt | None = None

    @model_validator(mode="after")
    def validate_one_primary_value(self) -> ProposalReplacement:
        primaries = (self.smiles, self.atom_index, self.integer, self.text)
        if sum(value is not None for value in primaries) != 1:
            raise ValueError("replacement must contain exactly one primary value")
        if self.attachment_atom is not None and self.smiles is None:
            raise ValueError("attachment_atom is valid only for a SMILES replacement")
        return self


class ProposalCandidatePatch(StrictFrozenModel):
    """Untrusted proposal only; it intentionally has no accepted/label fields."""

    candidate_id: Identifier
    root_field: Identifier
    replacement: ProposalReplacement
    bond_edits: Annotated[tuple[ProposedBondEdit, ...], Field(max_length=64)] = ()
    minimal_surface_realization: NonEmptyText
    plausibility_reason: NonEmptyText

    @field_validator("bond_edits")
    @classmethod
    def validate_unique_bond_edits(
        cls, value: tuple[ProposedBondEdit, ...]
    ) -> tuple[ProposedBondEdit, ...]:
        identities = tuple(
            (item.operation, item.begin_atom, item.end_atom, item.bond_type)
            for item in value
        )
        if len(identities) != len(set(identities)):
            raise ValueError("bond_edits must be unique")
        return value


class ProposalResponse(StrictFrozenModel):
    proposal_version: Literal["1.0"] = PROPOSAL_VERSION
    request_id: Identifier
    candidates: tuple[ProposalCandidatePatch, ...]
    abstain_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_candidate_or_abstention_shape(self) -> ProposalResponse:
        count = len(self.candidates)
        if count == 0:
            if self.abstain_reason is None:
                raise ValueError("zero candidates require an abstain_reason")
        elif not PROPOSAL_CANDIDATE_MIN <= count <= PROPOSAL_CANDIDATE_MAX:
            raise ValueError(
                "a non-abstaining response requires three to five candidates"
            )
        elif self.abstain_reason is not None:
            raise ValueError("candidate responses cannot also abstain")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a response")
        return self

    def validate_for_request(self, request: ProposalRequest) -> ProposalResponse:
        """Bind dynamic request ID and the request's single canonical root."""

        if type(request) is not ProposalRequest:
            raise TypeError("request must be ProposalRequest")
        if self.request_id != request.request_id:
            raise ValueError("proposal response request_id does not match request")
        wrong_roots = tuple(
            sorted(
                {
                    candidate.root_field
                    for candidate in self.candidates
                    if candidate.root_field != request.target_root
                }
            )
        )
        if wrong_roots:
            raise ValueError(
                f"proposal candidates escape target_root {request.target_root!r}: "
                f"{wrong_roots!r}"
            )
        return self


class InspectAtomsArgs(StrictFrozenModel):
    smiles: NonEmptyText
    atom_indices: Annotated[
        tuple[NonNegativeStrictInt, ...], Field(max_length=512)
    ] = ()

    @field_validator("atom_indices")
    @classmethod
    def unique_atom_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError("atom_indices must be unique")
        return value


class EnumerateAlternateAnchorsArgs(StrictFrozenModel):
    source_smiles: NonEmptyText
    reference_anchor_idx: PositiveStrictInt
    same_element_only: StrictBool = True
    max_results: Annotated[StrictInt, Field(gt=0, le=64)] = 16


class AnalyzeSmilesArgs(StrictFrozenModel):
    smiles: NonEmptyText
    fragment_policy: Literal["keep_all", "largest_heavy"] = "keep_all"


class FindGroupAtAnchorArgs(StrictFrozenModel):
    source_smiles: NonEmptyText
    anchor_idx: PositiveStrictInt
    max_heavy_atoms: Annotated[StrictInt, Field(gt=0, le=128)] = 64


class EnumerateRemovableGroupsArgs(StrictFrozenModel):
    source_smiles: NonEmptyText
    anchor_idx: PositiveStrictInt
    max_group_heavy_atoms: Annotated[StrictInt, Field(gt=0, le=128)] = 64
    max_results: Annotated[StrictInt, Field(gt=0, le=64)] = 16


class SimulateEditArgs(StrictFrozenModel):
    family: EditingFamilyName
    source_smiles: NonEmptyText
    anchor_idx: PositiveStrictInt
    remove_anchor_idx: PositiveStrictInt | None = None
    remove_group_smiles: NonEmptyText | None = None
    add_fragment_smiles: NonEmptyText | None = None
    fragment_attachment_atom: NonNegativeStrictInt | None = None
    bond_type: BondTypeName | None = None

    @model_validator(mode="after")
    def validate_edit_shape(self) -> SimulateEditArgs:
        _validate_tool_edit_shape(
            family=self.family,
            remove_anchor_idx=self.remove_anchor_idx,
            remove_group_smiles=self.remove_group_smiles,
            add_fragment_smiles=self.add_fragment_smiles,
            fragment_attachment_atom=self.fragment_attachment_atom,
            bond_type=self.bond_type,
        )
        return self


class ComputeDescriptorsArgs(StrictFrozenModel):
    smiles: NonEmptyText
    fragment_policy: Literal["keep_all", "largest_heavy"] = "keep_all"


class CompareMoleculesArgs(StrictFrozenModel):
    left_smiles: NonEmptyText
    right_smiles: NonEmptyText
    comparator: Literal[
        "isomeric_graph_equivalence",
        "fragment_graph_equivalence",
        "exact",
    ] = "isomeric_graph_equivalence"


class CheckCandidateSignatureArgs(StrictFrozenModel):
    family: EditingFamilyName
    source_smiles: NonEmptyText
    candidate_product_smiles: NonEmptyText
    anchor_idx: PositiveStrictInt
    remove_anchor_idx: PositiveStrictInt | None = None
    remove_group_smiles: NonEmptyText | None = None
    add_fragment_smiles: NonEmptyText | None = None
    fragment_attachment_atom: NonNegativeStrictInt | None = None
    bond_type: BondTypeName | None = None

    @model_validator(mode="after")
    def validate_edit_shape(self) -> CheckCandidateSignatureArgs:
        _validate_tool_edit_shape(
            family=self.family,
            remove_anchor_idx=self.remove_anchor_idx,
            remove_group_smiles=self.remove_group_smiles,
            add_fragment_smiles=self.add_fragment_smiles,
            fragment_attachment_atom=self.fragment_attachment_atom,
            bond_type=self.bond_type,
        )
        return self


def _validate_tool_edit_shape(
    *,
    family: EditingFamilyName,
    remove_anchor_idx: int | None,
    remove_group_smiles: str | None,
    add_fragment_smiles: str | None,
    fragment_attachment_atom: int | None,
    bond_type: BondTypeName | None,
) -> None:
    if family == "add":
        if (
            remove_anchor_idx is not None
            or remove_group_smiles is not None
            or add_fragment_smiles is None
            or fragment_attachment_atom is None
            or bond_type is None
        ):
            raise ValueError("add tool args require only add fragment/attachment/bond")
    elif family == "delete":
        if (
            remove_anchor_idx is not None
            or remove_group_smiles is None
            or add_fragment_smiles is not None
            or fragment_attachment_atom is not None
            or bond_type is not None
        ):
            raise ValueError("delete tool args require a remove group only")
    elif (
        remove_group_smiles is None
        or add_fragment_smiles is None
        or fragment_attachment_atom is None
        or bond_type is None
    ):
        raise ValueError("substitute tool args require remove/add/attachment/bond")


class InspectAtomsToolCall(StrictFrozenModel):
    tool: Literal["inspect_atoms"]
    arguments: InspectAtomsArgs


class EnumerateAlternateAnchorsToolCall(StrictFrozenModel):
    tool: Literal["enumerate_alternate_anchors"]
    arguments: EnumerateAlternateAnchorsArgs


class AnalyzeSmilesToolCall(StrictFrozenModel):
    tool: Literal["analyze_smiles"]
    arguments: AnalyzeSmilesArgs


class FindGroupAtAnchorToolCall(StrictFrozenModel):
    tool: Literal["find_group_at_anchor"]
    arguments: FindGroupAtAnchorArgs


class EnumerateRemovableGroupsToolCall(StrictFrozenModel):
    tool: Literal["enumerate_removable_groups"]
    arguments: EnumerateRemovableGroupsArgs


class SimulateEditToolCall(StrictFrozenModel):
    tool: Literal["simulate_edit"]
    arguments: SimulateEditArgs


class ComputeDescriptorsToolCall(StrictFrozenModel):
    tool: Literal["compute_descriptors"]
    arguments: ComputeDescriptorsArgs


class CompareMoleculesToolCall(StrictFrozenModel):
    tool: Literal["compare_molecules"]
    arguments: CompareMoleculesArgs


class CheckCandidateSignatureToolCall(StrictFrozenModel):
    tool: Literal["check_candidate_signature"]
    arguments: CheckCandidateSignatureArgs


ChemistryToolCall = Annotated[
    InspectAtomsToolCall
    | EnumerateAlternateAnchorsToolCall
    | AnalyzeSmilesToolCall
    | FindGroupAtAnchorToolCall
    | EnumerateRemovableGroupsToolCall
    | SimulateEditToolCall
    | ComputeDescriptorsToolCall
    | CompareMoleculesToolCall
    | CheckCandidateSignatureToolCall,
    Field(discriminator="tool"),
]

CHEMISTRY_TOOL_ARGUMENT_MODELS = MappingProxyType(
    {
        "inspect_atoms": InspectAtomsArgs,
        "enumerate_alternate_anchors": EnumerateAlternateAnchorsArgs,
        "analyze_smiles": AnalyzeSmilesArgs,
        "find_group_at_anchor": FindGroupAtAnchorArgs,
        "enumerate_removable_groups": EnumerateRemovableGroupsArgs,
        "simulate_edit": SimulateEditArgs,
        "compute_descriptors": ComputeDescriptorsArgs,
        "compare_molecules": CompareMoleculesArgs,
        "check_candidate_signature": CheckCandidateSignatureArgs,
    }
)
CHEMISTRY_TOOL_NAMES = tuple(CHEMISTRY_TOOL_ARGUMENT_MODELS)
_TOOL_CALL_ADAPTER = TypeAdapter(ChemistryToolCall)


def _mapping_as_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("mapping payload must contain finite JSON values") from error


def parse_proposal_response(
    payload: str | bytes | Mapping[str, Any],
    *,
    request: ProposalRequest,
) -> ProposalResponse:
    """Strictly parse and dynamically bind one untrusted Poe response."""

    if type(request) is not ProposalRequest:
        raise TypeError("request must be ProposalRequest")
    if isinstance(payload, (str, bytes)):
        response = ProposalResponse.model_validate_json(payload, strict=True)
    elif isinstance(payload, Mapping):
        response = ProposalResponse.model_validate_json(
            _mapping_as_json(payload), strict=True
        )
    else:
        raise TypeError("proposal payload must be JSON text/bytes or a mapping")
    return response.validate_for_request(request)


def parse_chemistry_tool_call(
    payload: str | bytes | Mapping[str, Any],
) -> ChemistryToolCall:
    """Validate a fixed-allowlist tool envelope before any dispatch."""

    if isinstance(payload, (str, bytes)):
        return _TOOL_CALL_ADAPTER.validate_json(payload, strict=True)
    if isinstance(payload, Mapping):
        return _TOOL_CALL_ADAPTER.validate_json(_mapping_as_json(payload), strict=True)
    raise TypeError("tool payload must be JSON text/bytes or a mapping")


def validate_chemistry_tool_arguments(
    tool: str,
    arguments: Mapping[str, Any],
) -> StrictFrozenModel:
    """Allow-list dispatch that validates arguments without executing a tool."""

    if type(tool) is not str or tool not in CHEMISTRY_TOOL_ARGUMENT_MODELS:
        raise ValueError(f"unknown chemistry tool {tool!r}")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    model_type = CHEMISTRY_TOOL_ARGUMENT_MODELS[tool]
    return model_type.model_validate_json(_mapping_as_json(arguments), strict=True)


def proposal_request_json_schema() -> dict[str, Any]:
    return ProposalRequest.model_json_schema(mode="validation")


def proposal_response_json_schema() -> dict[str, Any]:
    return ProposalResponse.model_json_schema(mode="validation")


def chemistry_tool_call_json_schema() -> dict[str, Any]:
    return _TOOL_CALL_ADAPTER.json_schema(mode="validation")


def proposal_request_from_candidate_request(
    request: CandidateRequest,
    *,
    verified_manifest: VerifiedSplitManifest,
) -> ProposalRequest:
    """Bind a T017/T018 request to one verified T029 manifest row.

    The exact T029 extraction is intentionally centralized here so callers
    cannot invent metadata keys or choose a split independently.
    """

    from molhallulens.modules.release.manifest import VerifiedSplitManifest
    from molhallulens.modules.error_planning import CandidateRequest

    if type(request) is not CandidateRequest:
        raise TypeError("request must be CandidateRequest")
    if type(verified_manifest) is not VerifiedSplitManifest:
        raise TypeError("verified_manifest must be VerifiedSplitManifest")
    recipe = request.context.recipe
    if recipe.candidate_source_mode not in {
        CandidateSourceType.LLM,
        CandidateSourceType.HYBRID,
    }:
        raise ValueError("Poe proposal requests require LLM or HYBRID source mode")
    row = verified_manifest.row_for_origin(recipe.origin_id)
    truth = request.context.truth
    if not (
        row.anonymous_sample_id == recipe.origin_id
        and row.subtask is request.resolution.registration.subtask
        and row.dataset_version == verified_manifest.dataset_version
        and row.split_seed == verified_manifest.split_seed
        and row.canonical_source_hash
        == hashlib.sha256(truth.canonical_source_smiles.encode("utf-8")).hexdigest()
        and row.canonical_gt_hash
        == hashlib.sha256(truth.canonical_gt_smiles.encode("utf-8")).hexdigest()
    ):
        raise ValueError("verified manifest row does not match the operator request")
    manifest_identity = ProposalManifestIdentity(
        dataset_version=verified_manifest.dataset_version,
        split_seed=verified_manifest.split_seed,
        manifest_sha256=verified_manifest.manifest_sha256,
        source_origin_audit_sha256=(verified_manifest.source_origin_audit_sha256),
        source_split_report_sha256=(verified_manifest.source_split_report_sha256),
    )
    proposal_constraints = ProposalConstraints.model_validate(
        dict(recipe.constraints), strict=True
    )
    return ProposalRequest(
        request_id=request.request_id,
        origin_id=recipe.origin_id,
        operator_id=request.operator_id,
        propagation=recipe.policy.dataset_name,
        candidate_source_mode=recipe.candidate_source_mode.value,
        target_root=recipe.target_node_id,
        constraints=proposal_constraints,
        global_seed=FROZEN_GLOBAL_SEED,
        dataset_version=verified_manifest.dataset_version,
        variant_index=recipe.variant_index,
        derived_seed=recipe.derived_seed,
        split=row.split.value,
        manifest_identity=manifest_identity,
    )


__all__ = [
    "CHEMISTRY_TOOL_ARGUMENT_MODELS",
    "CHEMISTRY_TOOL_NAMES",
    "FROZEN_DATASET_VERSION",
    "FROZEN_GLOBAL_SEED",
    "FROZEN_SPLIT_SEED",
    "PROPOSAL_CANDIDATE_MAX",
    "PROPOSAL_CANDIDATE_MIN",
    "PROPOSAL_SCHEMA_VERSION",
    "PROPOSAL_VERSION",
    "AnalyzeSmilesArgs",
    "AnalyzeSmilesToolCall",
    "CheckCandidateSignatureArgs",
    "CheckCandidateSignatureToolCall",
    "ChemistryToolCall",
    "CompareMoleculesArgs",
    "CompareMoleculesToolCall",
    "ComputeDescriptorsArgs",
    "ComputeDescriptorsToolCall",
    "EnumerateAlternateAnchorsArgs",
    "EnumerateAlternateAnchorsToolCall",
    "EnumerateRemovableGroupsArgs",
    "EnumerateRemovableGroupsToolCall",
    "FindGroupAtAnchorArgs",
    "FindGroupAtAnchorToolCall",
    "InspectAtomsArgs",
    "InspectAtomsToolCall",
    "ProposalCandidatePatch",
    "ProposalConstraints",
    "ProposalManifestIdentity",
    "ProposalReplacement",
    "ProposalRequest",
    "ProposalResponse",
    "ProposedBondEdit",
    "SimulateEditArgs",
    "SimulateEditToolCall",
    "StrictFrozenModel",
    "chemistry_tool_call_json_schema",
    "derive_proposal_seed",
    "parse_chemistry_tool_call",
    "parse_proposal_response",
    "proposal_request_from_candidate_request",
    "proposal_request_json_schema",
    "proposal_response_json_schema",
    "validate_chemistry_tool_arguments",
]
