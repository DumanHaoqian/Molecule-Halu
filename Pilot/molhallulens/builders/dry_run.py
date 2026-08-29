"""Deterministic T045 fifteen-origin dry-run build.

The dry run is a release-shaped, frozen offline slice.  It reuses the real
T019--T044 construction path, writes detector-visible records separately from
oracle/state/provenance artifacts, and refuses to publish an incomplete
origin.  The tokenizer backend implements the T042 fast-offset contract but is
explicitly a small offline fixture; no ChemDFM-R weights and no Poe endpoint
are contacted by this module.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.annotation import TokenLabelSetWriter, rebase_char_annotations
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.golden_validation import (
    DELETE_WITH_REPLACEMENT_ORIGIN_ID,
    T044_GOLDEN_ORIGIN_CASES,
    ExtendedGoldenOriginBuild,
    ExtendedGoldenOriginCase,
    build_extended_origin,
)
from molhallulens.builders.leakage_groups import assign_leakage_groups
from molhallulens.builders.origin_audit import audit_origin_split_features
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.builders.split_manifest import (
    VerifiedSplitManifest,
    load_verified_split_manifest,
)
from molhallulens.builders.splitter import build_group_stratified_split
from molhallulens.candidates.donors import SplitDonorPool, load_split_donor_pool
from molhallulens.config import load_config_bundle
from molhallulens.config.loader import ConfigBundle
from molhallulens.domain import (
    EditingSubtask,
    PropagationPolicy,
    TokenizerFingerprint,
    ValidationReport,
    VariantLabel,
)
from molhallulens.validation.chain import (
    ARTIFACT_VALIDATOR_IDS,
    BUNDLE_INTEGRITY_VALIDATOR_ID,
    ArtifactValidationInput,
    ValidatorChain,
)
from molhallulens.validation.reference import OriginValidationInput

T045_DRY_RUN_FORMAT_VERSION = "t045_dry_run_release_v1"
T045_DRY_RUN_ID = "t045_frozen_offline_15_origin_v1"
T045_RECORD_SCHEMA_VERSION = "molhallulens.edit.v1"
T045_REPORT_FORMAT_VERSION = "t045_dry_run_build_report_v1"
T045_ORIGIN_COUNT = 15
T045_ORIGINS_PER_SUBTASK = 5
T045_RECORDS_PER_ORIGIN = 8
T045_RECORD_COUNT = T045_ORIGIN_COUNT * T045_RECORDS_PER_ORIGIN

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "Dataset"
DEFAULT_MANIFEST_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset"
DEFAULT_DONOR_POOL_ROOT = DEFAULT_MANIFEST_ROOT / "donor_pools"
DEFAULT_DRY_RUN_ROOT = DEFAULT_MANIFEST_ROOT / "dry_run"
DEFAULT_REPORT_PATH = DEFAULT_DATASET_ROOT / "reports/t045_dry_run_build.json"

_SUBTASK_ORDER = (
    EditingSubtask.ADD,
    EditingSubtask.DELETE,
    EditingSubtask.SUBSTITUTE,
)
_SPLIT_ORDER = ("train", "validation", "test")
_STATUS_ACCEPTED = "accepted"
_STATUS_REJECTED = "rejected"
_FORBIDDEN_DATASET_KEYS = frozenset(
    {
        "build_provenance",
        "candidate_graph",
        "candidate_state_graph",
        "gt_smiles",
        "oracle",
        "oracle_gt",
        "reference_graph",
        "reference_state_graph",
    }
)
_FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "poe_api_key",
        "secret",
        "set_cookie",
    }
)
_CREDENTIAL_TEXT = re.compile(
    r"(?i)(?:^|\s)(?:authorization\s*:|poe_api_key\s*=|"
    r"(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,})"
)


class DryRunBuildError(RuntimeError):
    """Structured fail-closed T045 build or publication failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("dry-run error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("dry-run error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("dry-run error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": _stable(self.evidence),
        }


def _standard_case(subtask: EditingSubtask) -> ExtendedGoldenOriginCase:
    return next(
        case
        for case in T044_GOLDEN_ORIGIN_CASES
        if case.case_kind == "standard" and case.spec.normalized_subtask is subtask
    )


def _candidate_case(
    subtask: EditingSubtask,
    origin_id: str,
    suffix: str,
    *coverage_tags: str,
) -> ExtendedGoldenOriginCase:
    base = _standard_case(subtask)
    return ExtendedGoldenOriginCase(
        case_id=f"{subtask.value}.{suffix}",
        case_kind="dry_run_candidate",
        spec=replace(base.spec, origin_id=origin_id),
        coverage_tags=coverage_tags,
    )


# The two probe rows are intentionally attempted before reserves.  At the
# frozen corpus revision they fail the real candidate/gate path; the next
# complete origin replaces them.  If a future implementation legitimately
# makes a probe pass, it is simply accepted after the same strict T043 gates.
# Cross-balance choices were made only with train/validation diagnostics.  The
# three test origins are frozen inputs to the final held-out T047 diagnostic.
T045_ORIGIN_CANDIDATES: Mapping[
    EditingSubtask, tuple[ExtendedGoldenOriginCase, ...]
] = MappingProxyType(
    {
        EditingSubtask.ADD: (
            *(
                case
                for case in T044_GOLDEN_ORIGIN_CASES
                if case.spec.origin_id
                in {
                    "mol_edit.add_v2.0022",
                    "mol_edit.add_v2.0071",
                }
            ),
            _candidate_case(
                EditingSubtask.ADD,
                "mol_edit.add_v2.0282",
                "train_cross_balance",
                "dev_only_shortcut_remediation",
            ),
            _candidate_case(
                EditingSubtask.ADD,
                "mol_edit.add_v2.0012",
                "candidate_gap_probe",
                "real_candidate_gate_probe",
            ),
            _candidate_case(
                EditingSubtask.ADD,
                "mol_edit.add_v2.0023",
                "test_reserve",
                "split_coverage",
                "atomic_backfill_reserve",
            ),
            _candidate_case(
                EditingSubtask.ADD,
                "mol_edit.add_v2.0295",
                "train_cross_balance_reserve",
                "dev_only_shortcut_remediation",
            ),
        ),
        EditingSubtask.DELETE: (
            *(
                case
                for case in T044_GOLDEN_ORIGIN_CASES
                if case.spec.normalized_subtask is EditingSubtask.DELETE
            ),
            _candidate_case(
                EditingSubtask.DELETE,
                "mol_edit.delete_v2.0028",
                "validation_reserve",
                "split_coverage",
                "remove_only_capability",
            ),
            _candidate_case(
                EditingSubtask.DELETE,
                "mol_edit.delete_v2.0046",
                "test_reserve",
                "split_coverage",
                "remove_only_capability",
            ),
        ),
        EditingSubtask.SUBSTITUTE: (
            _candidate_case(
                EditingSubtask.SUBSTITUTE,
                "mol_edit.substitute_v2.0150",
                "validation_cross_balance",
                "dev_only_shortcut_remediation",
            ),
            *(
                case
                for case in T044_GOLDEN_ORIGIN_CASES
                if case.spec.origin_id == "mol_edit.substitute_v2.0271"
            ),
            _candidate_case(
                EditingSubtask.SUBSTITUTE,
                "mol_edit.substitute_v2.0232",
                "train_cross_balance",
                "dev_only_shortcut_remediation",
            ),
            _candidate_case(
                EditingSubtask.SUBSTITUTE,
                "mol_edit.substitute_v2.0000",
                "candidate_gap_probe",
                "real_candidate_gate_probe",
            ),
            _candidate_case(
                EditingSubtask.SUBSTITUTE,
                "mol_edit.substitute_v2.0057",
                "test_reserve",
                "split_coverage",
                "atomic_backfill_reserve",
            ),
            _candidate_case(
                EditingSubtask.SUBSTITUTE,
                "mol_edit.substitute_v2.0239",
                "validation_cross_balance_reserve",
                "split_coverage",
                "dev_only_shortcut_remediation",
            ),
        ),
    }
)


class _FrozenOfflineOffsetTokenizer:
    """Small deterministic tokenizer fixture satisfying the T042 API."""

    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        expected = {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "truncation": False,
            "padding": False,
        }
        if kwargs != expected:
            raise ValueError("T045 fixture requires the exact fast-offset call")
        offsets = tuple(
            (match.start(), match.end()) for match in re.finditer(r"\S+", text)
        )
        count = len(offsets)
        return {
            "input_ids": (
                self.bos_token_id,
                *(100 + index for index in range(count)),
                self.eos_token_id,
            ),
            "attention_mask": (1,) * (count + 2),
            "offset_mapping": ((0, 0), *offsets, (0, 0)),
            "special_tokens_mask": (1, *((0,) * count), 1),
            "input_text": text,
        }


T045_TOKENIZER_FINGERPRINT = TokenizerFingerprint(
    tokenizer_name="ChemDFM-R-offset-contract-offline-whitespace-fixture",
    tokenizer_revision="t045-frozen-offline-v1",
    tokenizer_vocab_hash="not-computed-offline-fixture",
    special_token_config={
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "unk_token_id": 3,
    },
    normalization_config={
        "normalizer": "none",
        "offset_unit": "python_char",
        "production_weights_loaded": False,
    },
)


@dataclass(frozen=True, slots=True)
class DryRunAttempt:
    """One accepted or rejected complete-origin attempt."""

    attempt_index: int
    subtask: EditingSubtask
    case_id: str
    origin_id: str
    status: str
    emitted_record_count: int
    error_code: str | None = None
    exception_type: str | None = None
    replacement_origin_id: str | None = None
    replaced_origin_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("attempt subtask must be EditingSubtask")
        for value, name in (
            (self.case_id, "case_id"),
            (self.origin_id, "origin_id"),
            (self.status, "status"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"attempt {name} must be non-empty text")
        if self.status not in {_STATUS_ACCEPTED, _STATUS_REJECTED}:
            raise ValueError("attempt status is invalid")
        replaced = tuple(self.replaced_origin_ids)
        if any(type(value) is not str or not value for value in replaced):
            raise TypeError("replaced_origin_ids must contain non-empty text")
        if len(replaced) != len(set(replaced)):
            raise ValueError("replaced_origin_ids must be unique")
        object.__setattr__(self, "replaced_origin_ids", replaced)
        if self.status == _STATUS_ACCEPTED:
            if (
                self.emitted_record_count != T045_RECORDS_PER_ORIGIN
                or self.error_code is not None
                or self.exception_type is not None
                or self.replacement_origin_id is not None
            ):
                raise ValueError(
                    "accepted attempt must emit exactly eight clean records"
                )
        elif (
            self.emitted_record_count != 0
            or type(self.error_code) is not str
            or not self.error_code
            or type(self.exception_type) is not str
            or not self.exception_type
            or self.replaced_origin_ids
        ):
            raise ValueError(
                "rejected attempt must emit zero records and carry failure identity"
            )
        if self.replacement_origin_id is not None and (
            type(self.replacement_origin_id) is not str
            or not self.replacement_origin_id
        ):
            raise ValueError("replacement_origin_id must be non-empty text or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": f"t045-attempt-{self.attempt_index:03d}",
            "attempt_index": self.attempt_index,
            "subtask": self.subtask.value,
            "case_id": self.case_id,
            "origin_id": self.origin_id,
            "status": self.status,
            "emitted_record_count": self.emitted_record_count,
            "atomic_unit": "complete_origin_bundle_4_pairs_8_records",
            "failure_stage": (
                None if self.status == _STATUS_ACCEPTED else "origin_bundle_build"
            ),
            "error_code": self.error_code,
            "exception_type": self.exception_type,
            "action": (
                "commit_complete_origin"
                if self.status == _STATUS_ACCEPTED
                else "discard_origin_and_replace"
            ),
            "replacement_origin_id": self.replacement_origin_id,
            "replaced_origin_ids": self.replaced_origin_ids,
            "committed": self.status == _STATUS_ACCEPTED,
        }


def _stable(value: Any) -> Any:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=repr)
    raise TypeError(f"unsupported T045 JSON value: {type(value).__qualname__}")


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            _stable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _render_jsonl(values: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            _stable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for value in values
    )


def _span(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [value.start, value.end]


def _report_snapshot(report: ValidationReport) -> dict[str, Any]:
    return {
        "validator_id": report.validator_id,
        "all_pass": report.all_pass,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "stage": issue.stage.value,
                "node_ids": issue.node_ids,
                "message": issue.message,
                "evidence": issue.evidence,
            }
            for issue in report.issues
        ],
    }


def _joined_by_id(
    records: Sequence[JoinedInputRecord],
) -> dict[str, JoinedInputRecord]:
    indexed: dict[str, JoinedInputRecord] = {}
    for record in records:
        if record.anonymous_sample_id in indexed:
            raise DryRunBuildError(
                "DRY_RUN_ORIGIN_DUPLICATE",
                "joined input contains a duplicate anonymous origin",
                evidence={"origin_id": record.anonymous_sample_id},
            )
        indexed[record.anonymous_sample_id] = record
    return indexed


def _load_verified_manifest(
    records: Sequence[JoinedInputRecord],
    manifest_root: Path,
) -> VerifiedSplitManifest:
    validated = []
    for record in records:
        reference = build_reference_dag(record)
        validated.append(
            OriginValidationInput(
                record=record,
                artifact=reference,
                edit_truth=derive_edit_truth(reference),
            )
        )
    audit = audit_origin_split_features(validated).audit
    leakage = assign_leakage_groups(
        audit,
        canonical_source_smiles_by_id={
            item.edit_truth.anonymous_sample_id: (
                item.edit_truth.canonical_source_smiles
            )
            for item in validated
        },
    )
    split = build_group_stratified_split(audit, leakage)
    return load_verified_split_manifest(
        manifest_root / "split_manifest.csv",
        manifest_root / "split_manifest.metadata.json",
        split_result=split,
        audit=audit,
    )


def _load_donor_pools(
    donor_pool_root: Path,
    manifest: VerifiedSplitManifest,
) -> Mapping[str, SplitDonorPool]:
    return MappingProxyType(
        {
            split: load_split_donor_pool(
                donor_pool_root / f"{split}.json",
                manifest=manifest,
                expected_split=split,
            )
            for split in _SPLIT_ORDER
        }
    )


def _nested_values_for_key(value: Any, wanted: str) -> tuple[Any, ...]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() == wanted:
                found.append(item)
            found.extend(_nested_values_for_key(item, wanted))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            found.extend(_nested_values_for_key(item, wanted))
    return tuple(found)


def _execution_for(
    origin: ExtendedGoldenOriginBuild,
    policy: PropagationPolicy,
) -> Any:
    matches = tuple(
        execution
        for execution in origin.golden.executions
        if execution.context.recipe.policy is policy
    )
    if len(matches) != 1:
        raise DryRunBuildError(
            "DRY_RUN_EXECUTION_AMBIGUOUS",
            "origin does not retain exactly one execution for a policy",
            evidence={
                "origin_id": origin.case.spec.origin_id,
                "policy": policy.dataset_name,
                "match_count": len(matches),
            },
        )
    return matches[0]


def _donor_origin_for_execution(execution: Any) -> str | None:
    values = [
        *_nested_values_for_key(execution.selected_patch.metadata, "donor_origin_id"),
    ]
    if execution.selected_patch.edit_action is not None:
        values.extend(
            _nested_values_for_key(
                execution.selected_patch.edit_action.metadata,
                "donor_origin_id",
            )
        )
    unique = tuple(sorted(set(values)))
    if not unique:
        return None
    if len(unique) != 1 or type(unique[0]) is not str or not unique[0]:
        raise DryRunBuildError(
            "DRY_RUN_DONOR_IDENTITY_AMBIGUOUS",
            "candidate provenance contains an invalid donor origin identity",
        )
    return unique[0]


def _validate_donor_edge(
    *,
    recipient_origin_id: str,
    donor_origin_id: str,
    split: str,
    manifest: VerifiedSplitManifest,
    donor_pools: Mapping[str, SplitDonorPool],
) -> None:
    if recipient_origin_id == donor_origin_id:
        raise DryRunBuildError(
            "DRY_RUN_SELF_DONOR",
            "self donors are forbidden",
            evidence={"origin_id": recipient_origin_id},
        )
    try:
        recipient = manifest.row_for_origin(recipient_origin_id)
        donor = manifest.row_for_origin(donor_origin_id)
        manifest.require_same_split(recipient_origin_id, donor_origin_id)
    except Exception as error:
        raise DryRunBuildError(
            "DRY_RUN_CROSS_SPLIT_DONOR",
            "verified manifest rejected the donor edge",
            evidence={
                "recipient_origin_id": recipient_origin_id,
                "donor_origin_id": donor_origin_id,
                "exception_type": type(error).__name__,
            },
        ) from error
    if recipient.split.value != split or donor.split.value != split:
        raise DryRunBuildError(
            "DRY_RUN_CROSS_SPLIT_DONOR",
            "donor edge differs from the record split",
            evidence={
                "record_split": split,
                "recipient_split": recipient.split.value,
                "donor_split": donor.split.value,
            },
        )
    pool = donor_pools[split]
    if not any(
        edge.recipient_origin_id == recipient_origin_id
        and edge.donor_origin_id == donor_origin_id
        for edge in pool.edges
    ):
        raise DryRunBuildError(
            "DRY_RUN_DONOR_EDGE_UNREGISTERED",
            "candidate donor edge is absent from the frozen split-local pool",
            evidence={
                "recipient_origin_id": recipient_origin_id,
                "donor_origin_id": donor_origin_id,
                "split": split,
            },
        )


def _assert_no_forbidden_key(value: Any, forbidden: frozenset[str], code: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in forbidden:
                raise DryRunBuildError(
                    code,
                    "release artifact contains a forbidden key",
                    evidence={"key": str(key)},
                )
            _assert_no_forbidden_key(item, forbidden, code)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _assert_no_forbidden_key(item, forbidden, code)


def _assert_no_secret(value: Any) -> None:
    _assert_no_forbidden_key(value, _FORBIDDEN_SECRET_KEYS, "DRY_RUN_SECRET_KEY")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_secret(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _assert_no_secret(item)
    elif type(value) is str and _CREDENTIAL_TEXT.search(value):
        raise DryRunBuildError(
            "DRY_RUN_SECRET_TEXT",
            "release artifact contains credential-shaped text",
        )


def _identity(
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    draft = artifact.draft
    return {
        "schema_version": T045_RECORD_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "dry_run_id": T045_DRY_RUN_ID,
        "record_id": draft.record_id,
        "origin_id": draft.origin_id,
        "bundle_id": draft.bundle_id,
        "pair_id": draft.pair_id,
        "split": artifact.split.value,
        "leakage_group_id": artifact.leakage_group_id,
    }


def _annotation_records(artifact: ArtifactValidationInput) -> list[dict[str, Any]]:
    rebased = rebase_char_annotations(
        artifact.rendered,
        artifact.serialized,
        artifact.char_annotations,
    )
    return [
        {
            "span_id": item.span_id,
            "component": item.component.value,
            "step_index": item.step_index,
            "state_or_edge_id": item.state_or_edge_id,
            "literal_span": _span(item.literal_span),
            "claim_span": _span(item.claim_span),
            "semantic_types": sorted(label.value for label in item.semantic_types),
            "edit_subtypes": sorted(label.value for label in item.edit_subtypes),
            "evidence_relations": sorted(
                label.value for label in item.evidence_relations
            ),
            "causal_role": (
                None if item.causal_role is None else item.causal_role.value
            ),
            "root_span_id": item.root_span_id,
        }
        for item in rebased.annotations
    ]


def _fallback_snapshot(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "requested_operator_id": value.requested_operator_id,
        "selected_operator_id": value.selected_operator_id,
        "requested_operator_family": value.requested_operator_family,
        "selected_operator_family": value.selected_operator_family,
        "policy": value.policy.dataset_name,
        "candidate_source": value.candidate_source.value,
        "quota_bucket": value.quota_bucket,
        "attempted_operator_ids": value.attempted_operator_ids,
        "quota_deviation": value.quota_deviation,
        "target_change_required": value.target_change_required,
    }


def _dataset_record(
    origin: ExtendedGoldenOriginBuild,
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    draft = artifact.draft
    execution = _execution_for(origin, draft.policy)
    donor_origin_id = _donor_origin_for_execution(execution)
    detector = artifact.serialized.detector_input
    trace_labels = {
        field.name: getattr(artifact.trace_labels, field.name)
        for field in fields(artifact.trace_labels)
    }
    record = {
        **_identity(artifact, dataset_version),
        "family": "mol_edit",
        "subtask": origin.case.spec.normalized_subtask.value,
        "variant": {
            "label": draft.variant_label.value,
            "propagation": draft.policy.dataset_name,
            "matched_record_id": draft.matched_record_id,
        },
        "detector_input": {
            "indexed_smiles": detector.indexed_smiles,
            "instruction": detector.instruction,
            "reasoning_chain": detector.reasoning_chain,
            "final_answer": detector.final_answer,
        },
        "serialized": {
            "text": artifact.serialized.text,
            "sha256": artifact.serialized.sha256,
            "template_version": artifact.serialized.template_version,
            "segments": [
                {
                    "field_name": segment.field_name,
                    "segment_kind": segment.segment_kind.value,
                    "start": segment.start,
                    "end": segment.end,
                }
                for segment in artifact.serialized.segments
            ],
        },
        "mutation": {
            "root_state_id": draft.target_node_id,
            "operator_id": draft.operator_id,
            "operator_family": draft.operator_family,
            "candidate_source": draft.candidate_source.value,
            "candidate_id": execution.selected_patch.candidate_id,
            "donor_origin_id": donor_origin_id,
            "renderer_id": draft.renderer_style_id,
            "seed": execution.context.recipe.derived_seed,
            "fallback": _fallback_snapshot(draft.fallback_decision),
        },
        "trace_labels": trace_labels,
        "spans": _annotation_records(artifact),
        "verification": {
            "rdkit_sanitize": True,
            "graph_edit_verified": True,
            "propagation_verified": True,
            "renderer_verified": True,
            "span_verified": True,
            "token_alignment_verified": True,
            "bundle_integrity_verified": origin.validation.all_pass,
        },
    }
    _assert_no_forbidden_key(
        record,
        _FORBIDDEN_DATASET_KEYS,
        "DRY_RUN_GT_LEAKAGE",
    )
    return record


def _claim_value_snapshot(value: Any) -> dict[str, Any]:
    return {
        "normalized_value": value.normalized_value,
        "value_type": value.value_type.value,
        "provenance": value.provenance.value,
        "locally_valid": value.locally_valid,
        "oracle_match": value.oracle_match,
        "confidence": value.confidence,
        "mention_ids": value.mention_ids,
    }


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "schema_id": state.schema.schema_id,
        "schema_version": state.schema.version,
        "nodes": {
            node_id: _claim_value_snapshot(value)
            for node_id, value in state.values.items()
        },
        "edges": {
            edge_id: _claim_value_snapshot(value)
            for edge_id, value in state.edge_values.items()
        },
    }


def _event_snapshot(event: Any) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "target_kind": event.target_kind.value,
        "target_id": event.node_or_edge_id,
        "before": event.before.normalized_value,
        "after": event.after.normalized_value,
        "causal_role": event.causal_role.value,
        "semantic_types": sorted(label.value for label in event.hallucination_types),
        "edit_subtypes": sorted(label.value for label in event.edit_subtypes),
        "operator_id": event.operator_id,
        "root_event_id": event.root_event_id,
    }


def _oracle_record(
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    draft = artifact.draft
    return {
        **_identity(artifact, dataset_version),
        "gt_smiles": draft.reference_graph.value_for("oracle_gt").normalized_value,
        "reference_state_graph": _state_snapshot(draft.reference_graph),
        "candidate_state_graph": _state_snapshot(draft.locked_state),
        "graph_delta": [_event_snapshot(event) for event in draft.graph_delta.events],
        "visible_to_detector": False,
    }


def _state_record(
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    draft = artifact.draft
    return {
        **_identity(artifact, dataset_version),
        "artifact_scope": "build_only_non_detector",
        "reference": _state_snapshot(draft.reference_graph),
        "locked": _state_snapshot(draft.locked_state),
        "semantic_difference_targets": [
            {"target_kind": kind.value, "target_id": target_id}
            for kind, target_id in sorted(
                draft.locked_state.semantic_differences(draft.reference_graph),
                key=lambda item: (item[0].value, item[1]),
            )
        ],
        "mutation_events": [
            _event_snapshot(event) for event in draft.graph_delta.events
        ],
        "formal_trace": [
            {
                "step_index": step.step_index,
                "step_name": step.step_name,
                "formal_ab": step.formal_ab,
            }
            for step in draft.formal_trace.steps
        ],
    }


def _token_record(
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    labels = artifact.token_labels
    if labels is None:
        raise DryRunBuildError(
            "DRY_RUN_TOKEN_ARTIFACT_MISSING",
            "strict T045 records require token labels",
            evidence={"record_id": artifact.record_id},
        )
    fingerprint = labels.tokenizer_fingerprint
    return {
        **_identity(artifact, dataset_version),
        "activation_alignment": labels.activation_alignment,
        "tokenizer_fingerprint": {
            "tokenizer_name": fingerprint.tokenizer_name,
            "tokenizer_revision": fingerprint.tokenizer_revision,
            "tokenizer_vocab_hash": fingerprint.tokenizer_vocab_hash,
            "special_token_config": fingerprint.special_token_config,
            "normalization_config": fingerprint.normalization_config,
        },
        "serialized_text_sha256": labels.serialized_text_sha256,
        "input_ids": labels.input_ids,
        "attention_mask": labels.attention_mask,
        "offset_mapping": labels.offset_mapping,
        "segment_ids": labels.segment_ids,
        "evaluation_mask": labels.evaluation_mask,
        "hallucination_core_mask": labels.hallucination_core_mask,
        "error_any_mask": labels.error_any_mask,
        "semantic_type_masks": labels.semantic_type_masks,
        "edit_subtype_masks": labels.edit_subtype_masks,
        "causal_role_masks": labels.causal_role_masks,
        "local_falsehood_mask": labels.local_falsehood_mask,
        "off_task_branch_mask": labels.off_task_branch_mask,
        "reasoning_mask": labels.reasoning_mask,
        "answer_mask": labels.answer_mask,
        "boundary_ambiguous_mask": labels.boundary_ambiguous_mask,
        "error_char_fraction": labels.error_char_fraction,
        "matched_target_span": _span(labels.matched_target_span),
    }


def _provenance_record(
    origin: ExtendedGoldenOriginBuild,
    artifact: ArtifactValidationInput,
    dataset_version: str,
) -> dict[str, Any]:
    draft = artifact.draft
    execution = _execution_for(origin, draft.policy)
    trace = execution.to_trace_dict()
    donor_origin_id = _donor_origin_for_execution(execution)
    return {
        **_identity(artifact, dataset_version),
        "artifact_scope": "private_build_provenance",
        "execution_mode": {
            "network_mode": "frozen_offline",
            "live_poe_attempted": False,
            "live_availability_probe_performed": False,
            "network_request_count": 0,
            "provider": None,
            "transport": None,
            "requested_model_id": None,
            "response_model": None,
            "request_ids": [],
            "response_ids": [],
            "cache_keys": [],
            "token_usage": {},
            "cost_points": None,
            "reason": (
                "T045 intentionally replays deterministic local candidate and "
                "rule-renderer paths; live Poe status was not probed"
            ),
        },
        "recipe": {
            "recipe_id": execution.context.recipe.recipe_id,
            "policy": draft.policy.dataset_name,
            "operator_id": draft.operator_id,
            "operator_family": draft.operator_family,
            "quota_bucket": draft.quota_bucket,
            "target_node_id": draft.target_node_id,
            "candidate_source_mode": (
                execution.context.recipe.candidate_source_mode.value
            ),
            "candidate_difficulty_bucket": draft.candidate_difficulty_bucket,
            "derived_seed": execution.context.recipe.derived_seed,
            "rewrite_budget": {
                "max_changed_claims": draft.rewrite_budget.max_changed_claims,
                "max_added_characters": draft.rewrite_budget.max_added_characters,
                "length_bucket": draft.rewrite_budget.length_bucket,
            },
        },
        "candidate_selection": {
            "candidate_engine": execution.candidate_engine_name,
            "request_id": execution.pool.request_id,
            "accepted_pool_size": len(execution.pool.candidates),
            "pool_rejection_codes": execution.pool.rejection_codes,
            "selected_from_complete_pool": True,
            "selected_candidate_id": execution.selected_patch.candidate_id,
            "selected_candidate_source": execution.selected_patch.source.value,
            "selected_rank": execution.pool.candidates.index(execution.selected_patch),
            "trace": trace["candidate_pool"],
        },
        "donor": {
            "donor_origin_id": donor_origin_id,
            "recipient_split": artifact.split.value,
            "donor_split": None if donor_origin_id is None else artifact.split.value,
            "pool_split": artifact.split.value,
            "verified_split_local": True,
        },
        "fallback": _fallback_snapshot(draft.fallback_decision),
        "propagation": trace["propagation_plan"],
        "renderer": {
            "backend": draft.renderer_backend,
            "style_id": draft.renderer_style_id,
            "live_llm_used": False,
        },
        "tokenizer": {
            "backend": "deterministic_fast_offset_offline_fixture",
            "fingerprint": T045_TOKENIZER_FINGERPRINT.tokenizer_revision,
            "production_chemdfm_r_weights_loaded": False,
        },
    }


def _validation_record(
    origin: ExtendedGoldenOriginBuild,
    artifact: ArtifactValidationInput,
    chain: ValidatorChain,
    dataset_version: str,
) -> dict[str, Any]:
    gates = (
        chain.semantic.validate(artifact),
        chain.propagation.validate(artifact),
        chain.renderer.validate(artifact),
        chain.token_alignment.validate(artifact),
    )
    return {
        **_identity(artifact, dataset_version),
        "artifact_chain": _report_snapshot(chain.validate_artifact(artifact)),
        "artifact_gates": [_report_snapshot(report) for report in gates],
        "bundle_validator_id": BUNDLE_INTEGRITY_VALIDATOR_ID,
        "bundle_all_pass": origin.validation.all_pass,
    }


@dataclass(frozen=True, slots=True)
class DryRunBuild:
    """A fully validated in-memory T045 release slice."""

    dataset_version: str
    global_seed: int
    origins: tuple[ExtendedGoldenOriginBuild, ...]
    attempts: tuple[DryRunAttempt, ...]
    split_manifest: VerifiedSplitManifest
    donor_pools: Mapping[str, SplitDonorPool]

    def __post_init__(self) -> None:
        origins = tuple(self.origins)
        attempts = tuple(self.attempts)
        pools = MappingProxyType(dict(self.donor_pools))
        if type(self.dataset_version) is not str or not self.dataset_version:
            raise ValueError("dataset_version must be non-empty text")
        if type(self.global_seed) is not int:
            raise TypeError("global_seed must be an integer")
        if len(origins) != T045_ORIGIN_COUNT or any(
            type(origin) is not ExtendedGoldenOriginBuild for origin in origins
        ):
            raise DryRunBuildError(
                "DRY_RUN_ORIGIN_COUNT",
                "T045 requires exactly fifteen complete origins",
                evidence={"observed": len(origins)},
            )
        if any(type(attempt) is not DryRunAttempt for attempt in attempts):
            raise TypeError("attempts must contain DryRunAttempt values")
        if type(self.split_manifest) is not VerifiedSplitManifest:
            raise TypeError("split_manifest must be loader-verified")
        if set(pools) != set(_SPLIT_ORDER) or any(
            type(pool) is not SplitDonorPool for pool in pools.values()
        ):
            raise TypeError("donor_pools must contain three verified split pools")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "donor_pools", pools)
        self._validate_complete_release()

    @property
    def artifacts(self) -> tuple[ArtifactValidationInput, ...]:
        return tuple(
            artifact for origin in self.origins for artifact in origin.artifacts
        )

    def _validate_complete_release(self) -> None:
        counts = Counter(origin.case.spec.normalized_subtask for origin in self.origins)
        if counts != Counter(
            {subtask: T045_ORIGINS_PER_SUBTASK for subtask in _SUBTASK_ORDER}
        ):
            raise DryRunBuildError(
                "DRY_RUN_SUBTASK_COUNT",
                "T045 requires five complete origins per editing subtask",
                evidence={key.value: value for key, value in counts.items()},
            )
        origin_ids = tuple(origin.case.spec.origin_id for origin in self.origins)
        if len(origin_ids) != len(set(origin_ids)):
            raise DryRunBuildError(
                "DRY_RUN_ORIGIN_DUPLICATE",
                "selected dry-run origins must be unique",
            )
        if DELETE_WITH_REPLACEMENT_ORIGIN_ID in origin_ids:
            raise DryRunBuildError(
                "DRY_RUN_DELETE_CAPABILITY_FORBIDDEN",
                "delete-with-replacement cannot form the T014 remove-only bundle",
            )
        artifacts = self.artifacts
        if len(artifacts) != T045_RECORD_COUNT:
            raise DryRunBuildError(
                "DRY_RUN_RECORD_COUNT",
                "T045 requires exactly 120 complete records",
                evidence={"observed": len(artifacts)},
            )
        record_ids = tuple(artifact.record_id for artifact in artifacts)
        if len(record_ids) != len(set(record_ids)):
            raise DryRunBuildError(
                "DRY_RUN_RECORD_DUPLICATE",
                "dry-run record IDs must be globally unique",
            )
        if Counter(artifact.draft.variant_label for artifact in artifacts) != Counter(
            {VariantLabel.HALLUCINATED: 60, VariantLabel.FAITHFUL: 60}
        ):
            raise DryRunBuildError(
                "DRY_RUN_VARIANT_BALANCE",
                "T045 must contain sixty H and sixty N records",
            )
        accepted = tuple(
            attempt for attempt in self.attempts if attempt.status == _STATUS_ACCEPTED
        )
        if (
            len(accepted) != T045_ORIGIN_COUNT
            or tuple(attempt.origin_id for attempt in accepted) != origin_ids
            or any(
                attempt.emitted_record_count != T045_RECORDS_PER_ORIGIN
                for attempt in accepted
            )
            or any(
                attempt.emitted_record_count != 0
                for attempt in self.attempts
                if attempt.status == _STATUS_REJECTED
            )
        ):
            raise DryRunBuildError(
                "DRY_RUN_BACKFILL_NOT_ATOMIC",
                "attempt ledger contains partial or unbound output",
            )
        for origin in self.origins:
            if not origin.validation.all_pass or len(origin.artifacts) != 8:
                raise DryRunBuildError(
                    "DRY_RUN_STRICT_VALIDATION_FAILED",
                    "selected origin did not pass the complete T043 chain",
                    evidence={"origin_id": origin.case.spec.origin_id},
                )
            policies = Counter(artifact.draft.policy for artifact in origin.artifacts)
            if policies != Counter({policy: 2 for policy in PropagationPolicy}):
                raise DryRunBuildError(
                    "DRY_RUN_POLICY_SET",
                    "origin does not contain one H/N pair per policy",
                    evidence={"origin_id": origin.case.spec.origin_id},
                )
            row = self.split_manifest.row_for_origin(origin.case.spec.origin_id)
            for artifact in origin.artifacts:
                if (
                    artifact.split is not row.split
                    or artifact.leakage_group_id != row.leakage_group_id
                ):
                    raise DryRunBuildError(
                        "DRY_RUN_MANIFEST_BINDING",
                        "artifact split/group differs from verified manifest",
                        evidence={"record_id": artifact.record_id},
                    )
            for execution in origin.golden.executions:
                donor_origin_id = _donor_origin_for_execution(execution)
                if donor_origin_id is not None:
                    _validate_donor_edge(
                        recipient_origin_id=origin.case.spec.origin_id,
                        donor_origin_id=donor_origin_id,
                        split=row.split.value,
                        manifest=self.split_manifest,
                        donor_pools=self.donor_pools,
                    )
        dataset = self.dataset_records()
        for record in dataset:
            _assert_no_forbidden_key(
                record,
                _FORBIDDEN_DATASET_KEYS,
                "DRY_RUN_GT_LEAKAGE",
            )
        for collection in (
            dataset,
            self.oracle_records(),
            self.state_records(),
            self.token_records(),
            self.provenance_records(),
        ):
            _assert_no_secret(collection)

    def dataset_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _dataset_record(origin, artifact, self.dataset_version)
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def oracle_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _oracle_record(artifact, self.dataset_version)
            for artifact in self.artifacts
        )

    def state_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _state_record(artifact, self.dataset_version) for artifact in self.artifacts
        )

    def token_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _token_record(artifact, self.dataset_version) for artifact in self.artifacts
        )

    def provenance_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _provenance_record(origin, artifact, self.dataset_version)
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def validation_records(self) -> tuple[dict[str, Any], ...]:
        chain = ValidatorChain()
        return tuple(
            _validation_record(origin, artifact, chain, self.dataset_version)
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def selection_manifest(self) -> dict[str, Any]:
        return {
            "format_version": T045_DRY_RUN_FORMAT_VERSION,
            "dry_run_id": T045_DRY_RUN_ID,
            "selection_unit": "origin",
            "commit_unit": "complete_origin_bundle_4_pairs_8_records",
            "required_origins_per_subtask": T045_ORIGINS_PER_SUBTASK,
            "selected": [
                {
                    "case_id": origin.case.case_id,
                    "case_kind": origin.case.case_kind,
                    "coverage_tags": origin.case.coverage_tags,
                    "origin_id": origin.case.spec.origin_id,
                    "subtask": origin.case.spec.normalized_subtask.value,
                    "split": origin.artifacts[0].split.value,
                    "leakage_group_id": origin.artifacts[0].leakage_group_id,
                    "record_count": len(origin.artifacts),
                }
                for origin in self.origins
            ],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "excluded_capability_cases": [
                {
                    "origin_id": DELETE_WITH_REPLACEMENT_ORIGIN_ID,
                    "capability": "delete_with_replacement",
                    "required_bundle_capability": "structural_deletion_remove_only",
                    "included": False,
                    "reason": "T014 capability contract forbids replacement deletion",
                }
            ],
        }

    def validation_report(self) -> dict[str, Any]:
        return {
            "format_version": "t045_strict_validation_report_v1",
            "dry_run_id": T045_DRY_RUN_ID,
            "all_pass": True,
            "required_validator_ids": (
                *ARTIFACT_VALIDATOR_IDS,
                BUNDLE_INTEGRITY_VALIDATOR_ID,
            ),
            "artifact_gate_count": len(self.artifacts) * len(ARTIFACT_VALIDATOR_IDS),
            "bundle_gate_count": len(self.origins),
            "records": self.validation_records(),
            "origins": [
                {
                    "origin_id": origin.case.spec.origin_id,
                    "subtask": origin.case.spec.normalized_subtask.value,
                    "record_count": len(origin.artifacts),
                    "all_pass": origin.validation.all_pass,
                    "issue_codes": tuple(
                        issue.code for issue in origin.validation.issues
                    ),
                }
                for origin in self.origins
            ],
        }

    def build_report(self) -> dict[str, Any]:
        artifacts = self.artifacts
        source_counts = Counter(
            execution.selected_patch.source.value
            for origin in self.origins
            for execution in origin.golden.executions
        )
        split_origin_counts = Counter(
            origin.artifacts[0].split.value for origin in self.origins
        )
        split_record_counts = Counter(artifact.split.value for artifact in artifacts)
        rejection_counts = Counter(
            attempt.error_code
            for attempt in self.attempts
            if attempt.status == _STATUS_REJECTED
        )
        donor_edges = sum(
            _donor_origin_for_execution(execution) is not None
            for origin in self.origins
            for execution in origin.golden.executions
        )
        return {
            "format_version": T045_REPORT_FORMAT_VERSION,
            "dry_run_id": T045_DRY_RUN_ID,
            "dataset_version": self.dataset_version,
            "global_seed": self.global_seed,
            "all_pass": True,
            "execution": {
                "network_mode": "frozen_offline",
                "live_poe_attempted": False,
                "live_availability_probe_performed": False,
                "network_request_count": 0,
                "live_llm_material_participation_count": 0,
                "renderer": "natural_rule_v1",
                "tokenizer_backend": "deterministic_fast_offset_offline_fixture",
                "production_chemdfm_r_weights_loaded": False,
                "deterministic_replay": True,
                "content_hashes_added_by_t045": False,
            },
            "summary": {
                "origin_count": len(self.origins),
                "record_count": len(artifacts),
                "hallucinated_record_count": sum(
                    artifact.draft.variant_label is VariantLabel.HALLUCINATED
                    for artifact in artifacts
                ),
                "faithful_record_count": sum(
                    artifact.draft.variant_label is VariantLabel.FAITHFUL
                    for artifact in artifacts
                ),
                "origins_per_subtask": {
                    subtask.value: sum(
                        origin.case.spec.normalized_subtask is subtask
                        for origin in self.origins
                    )
                    for subtask in _SUBTASK_ORDER
                },
                "origin_counts_by_split": {
                    split: split_origin_counts[split] for split in _SPLIT_ORDER
                },
                "record_counts_by_split": {
                    split: split_record_counts[split] for split in _SPLIT_ORDER
                },
                "candidate_sources": dict(sorted(source_counts.items())),
                "attempt_count": len(self.attempts),
                "rejected_origin_attempt_count": sum(rejection_counts.values()),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            },
            "acceptance": {
                "fifteen_complete_origins": len(self.origins) == 15,
                "one_hundred_twenty_records": len(artifacts) == 120,
                "five_origins_per_subtask": True,
                "four_h_four_n_per_origin": True,
                "all_t043_strict_validators_pass": True,
                "cross_split_donor_edge_count": 0,
                "verified_donor_edge_count": donor_edges,
                "gt_or_reference_artifact_leakage_count": 0,
                "secret_leakage_count": 0,
                "rejected_attempt_partial_record_count": 0,
                "atomic_backfill_unit": "complete_origin_bundle_4_pairs_8_records",
                "delete_with_replacement_excluded": True,
            },
            "limitations": [
                "Poe/live availability was not probed and no live request was made.",
                "Token labels use an explicit offline fast-offset fixture; production ChemDFM-R weights are not loaded.",
                "T045 is a pipeline dry run, not the final 1,200-record release.",
            ],
            "artifact_layout": {
                "split_scoped_families": (
                    "records",
                    "oracle",
                    "state_graphs",
                    "tokenized/chemdfm_r",
                    "provenance",
                ),
                "reports": (
                    "build_report.json",
                    "validation_report.json",
                    "backfill_ledger.jsonl",
                    "rejection_counts.json",
                ),
            },
        }

    def dataset_manifest(self) -> dict[str, Any]:
        return {
            "format_version": T045_DRY_RUN_FORMAT_VERSION,
            "record_schema_version": T045_RECORD_SCHEMA_VERSION,
            "dry_run_id": T045_DRY_RUN_ID,
            "dataset_version": self.dataset_version,
            "release_kind": "dry_run",
            "record_count": len(self.artifacts),
            "origin_count": len(self.origins),
            "splits": {
                split: {
                    "record_count": sum(
                        artifact.split.value == split for artifact in self.artifacts
                    ),
                    "origin_count": sum(
                        origin.artifacts[0].split.value == split
                        for origin in self.origins
                    ),
                }
                for split in _SPLIT_ORDER
            },
            "detector_field_order": (
                "indexed_smiles",
                "instruction",
                "reasoning_chain",
                "final_answer",
            ),
            "oracle_visible_to_detector": False,
            "network_mode": "frozen_offline",
            "content_hashes_added_by_t045": False,
        }

    def artifact_payloads(self) -> Mapping[str, str]:
        families = {
            "records": self.dataset_records(),
            "oracle": self.oracle_records(),
            "state_graphs": self.state_records(),
            "tokenized/chemdfm_r": self.token_records(),
            "provenance": self.provenance_records(),
        }
        payloads: dict[str, str] = {}
        for directory, values in families.items():
            for split in _SPLIT_ORDER:
                selected = tuple(
                    sorted(
                        (value for value in values if value["split"] == split),
                        key=lambda value: value["record_id"],
                    )
                )
                payloads[f"{directory}/{split}.jsonl"] = _render_jsonl(selected)
        payloads.update(
            {
                "dataset_manifest.json": _render_json(self.dataset_manifest()),
                "selection_manifest.json": _render_json(self.selection_manifest()),
                "reports/build_report.json": _render_json(self.build_report()),
                "reports/validation_report.json": _render_json(
                    self.validation_report()
                ),
                "reports/backfill_ledger.jsonl": _render_jsonl(
                    tuple(attempt.to_dict() for attempt in self.attempts)
                ),
                "reports/rejection_counts.json": _render_json(
                    {
                        "format_version": "t045_rejection_counts_v1",
                        "dry_run_id": T045_DRY_RUN_ID,
                        "counts": dict(
                            sorted(
                                Counter(
                                    attempt.error_code
                                    for attempt in self.attempts
                                    if attempt.status == _STATUS_REJECTED
                                ).items()
                            )
                        ),
                    }
                ),
            }
        )
        _assert_no_secret(payloads)
        return MappingProxyType(payloads)

    def render_report_json(self) -> str:
        return _render_json(self.build_report())


OriginBuilder = Callable[..., ExtendedGoldenOriginBuild]


def build_t045_dry_run(
    dataset_root: Path | None = None,
    manifest_root: Path | None = None,
    donor_pool_root: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    candidates: Mapping[
        EditingSubtask, Sequence[ExtendedGoldenOriginCase]
    ] = T045_ORIGIN_CANDIDATES,
    origin_builder: OriginBuilder = build_extended_origin,
) -> DryRunBuild:
    """Build fifteen real origins with complete-origin atomic backfill."""

    root = DEFAULT_DATASET_ROOT if dataset_root is None else Path(dataset_root)
    manifest_dir = (
        DEFAULT_MANIFEST_ROOT if manifest_root is None else Path(manifest_root)
    )
    donor_dir = (
        DEFAULT_DONOR_POOL_ROOT if donor_pool_root is None else Path(donor_pool_root)
    )
    loaded_config = load_config_bundle() if config is None else config
    if type(loaded_config) is not ConfigBundle:
        raise TypeError("config must be ConfigBundle or None")
    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a subtask mapping")
    if not callable(origin_builder):
        raise TypeError("origin_builder must be callable")
    records = ChemCoTMolEditAdapter().load(root)
    indexed = _joined_by_id(records)
    manifest = _load_verified_manifest(records, manifest_dir)
    donor_pools = _load_donor_pools(donor_dir, manifest)
    token_writer = TokenLabelSetWriter(
        _FrozenOfflineOffsetTokenizer(),
        T045_TOKENIZER_FINGERPRINT,
    )
    selected: list[ExtendedGoldenOriginBuild] = []
    attempts: list[DryRunAttempt] = []
    attempt_index = 0
    for subtask in _SUBTASK_ORDER:
        queue = tuple(candidates.get(subtask, ()))
        if any(
            type(case) is not ExtendedGoldenOriginCase
            or case.spec.normalized_subtask is not subtask
            for case in queue
        ):
            raise TypeError("candidate queue contains an invalid subtask case")
        accepted_for_subtask = 0
        pending_rejections: list[int] = []
        for case in queue:
            if accepted_for_subtask == T045_ORIGINS_PER_SUBTASK:
                break
            joined = indexed.get(case.spec.origin_id)
            if joined is None:
                error: Exception = DryRunBuildError(
                    "DRY_RUN_ORIGIN_MISSING",
                    "candidate origin is absent from joined inputs",
                    evidence={"origin_id": case.spec.origin_id},
                )
            else:
                try:
                    built = origin_builder(
                        case,
                        config=loaded_config,
                        joined=joined,
                        manifest=manifest,
                        token_writer=token_writer,
                        global_seed=loaded_config.dataset.dataset.global_seed,
                    )
                except (RuntimeError, TypeError, ValueError) as caught:
                    error = caught
                else:
                    if type(built) is not ExtendedGoldenOriginBuild:
                        raise TypeError(
                            "origin_builder must return ExtendedGoldenOriginBuild"
                        )
                    replaced_ids = tuple(
                        attempts[index].origin_id for index in pending_rejections
                    )
                    for rejected_index in pending_rejections:
                        attempts[rejected_index] = replace(
                            attempts[rejected_index],
                            replacement_origin_id=case.spec.origin_id,
                        )
                    pending_rejections.clear()
                    attempts.append(
                        DryRunAttempt(
                            attempt_index=attempt_index,
                            subtask=subtask,
                            case_id=case.case_id,
                            origin_id=case.spec.origin_id,
                            status=_STATUS_ACCEPTED,
                            emitted_record_count=T045_RECORDS_PER_ORIGIN,
                            replaced_origin_ids=replaced_ids,
                        )
                    )
                    attempt_index += 1
                    selected.append(built)
                    accepted_for_subtask += 1
                    continue
            code = getattr(error, "code", type(error).__name__.upper())
            if type(code) is not str or not code:
                code = type(error).__name__.upper()
            attempts.append(
                DryRunAttempt(
                    attempt_index=attempt_index,
                    subtask=subtask,
                    case_id=case.case_id,
                    origin_id=case.spec.origin_id,
                    status=_STATUS_REJECTED,
                    emitted_record_count=0,
                    error_code=code,
                    exception_type=type(error).__name__,
                )
            )
            pending_rejections.append(len(attempts) - 1)
            attempt_index += 1
        if accepted_for_subtask != T045_ORIGINS_PER_SUBTASK:
            raise DryRunBuildError(
                "DRY_RUN_BACKFILL_EXHAUSTED",
                "candidate queue cannot provide five complete origins",
                evidence={
                    "subtask": subtask.value,
                    "accepted_count": accepted_for_subtask,
                    "candidate_count": len(queue),
                    "pending_rejection_count": len(pending_rejections),
                },
            )
    return DryRunBuild(
        dataset_version=loaded_config.dataset.dataset.version_name,
        global_seed=loaded_config.dataset.dataset.global_seed,
        origins=tuple(selected),
        attempts=tuple(attempts),
        split_manifest=manifest,
        donor_pools=donor_pools,
    )


def _existing_publication_matches(
    root: Path,
    payloads: Mapping[str, str],
) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        raise DryRunBuildError(
            "DRY_RUN_ARTIFACT_CONFLICT",
            "dry-run publication target exists but is not a directory",
        )
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed != set(payloads):
        raise DryRunBuildError(
            "DRY_RUN_ARTIFACT_CONFLICT",
            "existing dry-run file inventory differs from the complete release",
            evidence={
                "missing_count": len(set(payloads) - observed),
                "unexpected_count": len(observed - set(payloads)),
            },
        )
    for relative, payload in payloads.items():
        if (root / relative).read_text(encoding="utf-8") != payload:
            raise DryRunBuildError(
                "DRY_RUN_ARTIFACT_CONFLICT",
                "existing dry-run artifact differs from deterministic replay",
                evidence={"relative_path": relative},
            )
    return True


def _publish_directory(root: Path, payloads: Mapping[str, str]) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    if _existing_publication_matches(root, payloads):
        return
    staging = Path(tempfile.mkdtemp(prefix=".t045-staging-", dir=root.parent))
    try:
        for relative, payload in payloads.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        staging.replace(root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _publish_report(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise DryRunBuildError(
                "DRY_RUN_REPORT_CONFLICT",
                "existing T045 report differs from deterministic replay",
                evidence={"filename": path.name},
            )
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def write_t045_dry_run_artifacts(
    *,
    dry_run_root: Path | None = None,
    report_path: Path | None = None,
    build: DryRunBuild | None = None,
) -> DryRunBuild:
    """Atomically publish one validated T045 release slice and build report."""

    value = build_t045_dry_run() if build is None else build
    if type(value) is not DryRunBuild:
        raise TypeError("build must be DryRunBuild or None")
    root = DEFAULT_DRY_RUN_ROOT if dry_run_root is None else Path(dry_run_root)
    report = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    payloads = value.artifact_payloads()
    report_payload = value.render_report_json()
    if report.exists() and (
        not report.is_file() or report.read_text(encoding="utf-8") != report_payload
    ):
        raise DryRunBuildError(
            "DRY_RUN_REPORT_CONFLICT",
            "existing T045 report differs from deterministic replay",
            evidence={"filename": report.name},
        )
    _publish_directory(root, payloads)
    _publish_report(report, report_payload)
    return value


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DONOR_POOL_ROOT",
    "DEFAULT_DRY_RUN_ROOT",
    "DEFAULT_MANIFEST_ROOT",
    "DEFAULT_REPORT_PATH",
    "T045_DRY_RUN_FORMAT_VERSION",
    "T045_DRY_RUN_ID",
    "T045_ORIGINS_PER_SUBTASK",
    "T045_ORIGIN_CANDIDATES",
    "T045_ORIGIN_COUNT",
    "T045_RECORD_COUNT",
    "T045_RECORD_SCHEMA_VERSION",
    "T045_REPORT_FORMAT_VERSION",
    "T045_TOKENIZER_FINGERPRINT",
    "DryRunAttempt",
    "DryRunBuild",
    "DryRunBuildError",
    "build_t045_dry_run",
    "write_t045_dry_run_artifacts",
]
