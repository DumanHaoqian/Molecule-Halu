"""T048 frozen train-split construction and publication.

The builder consumes only origins assigned to ``train`` by the loader-verified
T029 manifest.  Every recipe attempt is provisional until all four reciprocal
H/N pairs pass the T043 chain; a failed attempt therefore emits no records.
Detector-visible, oracle, state, token and provenance artifacts remain separate.

Token projection uses the same T042 fast-offset interface as production, but
the backend in T048 is explicitly an offline contract fixture.  T051 replaces
that family after loading the real ChemDFM-R tokenizer and weights.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.annotation import TokenLabelSetWriter
from molhallulens.modules.release.dry_run import (
    _FORBIDDEN_DATASET_KEYS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_DONOR_POOL_ROOT,
    DEFAULT_MANIFEST_ROOT,
    _assert_no_forbidden_key,
    _assert_no_secret,
    _dataset_record,
    _donor_origin_for_execution,
    _FrozenOfflineOffsetTokenizer,
    _joined_by_id,
    _load_donor_pools,
    _load_verified_manifest,
    _oracle_record,
    _provenance_record,
    _render_json,
    _render_jsonl,
    _state_record,
    _token_record,
    _validate_donor_edge,
    _validation_record,
)
from molhallulens.modules.release.generation import GoldenPolicySpec
from molhallulens.modules.release.record_build import (
    ExtendedGoldenOriginBuild,
    ExtendedGoldenOriginCase,
    build_extended_origin,
)
from molhallulens.modules.release.manifest import VerifiedSplitManifest
from molhallulens.modules.release.splitter import SplitName
from molhallulens.modules.error_planning.donors import SplitDonorPool
from molhallulens.config import load_config_bundle
from molhallulens.config.loader import ConfigBundle
from molhallulens.core import (
    EditingSubtask,
    PropagationPolicy,
    TokenizerFingerprint,
    VariantLabel,
)
from molhallulens.modules.error_injection.operators.addition import ADDITION_OPERATOR_IDS
from molhallulens.modules.error_injection.operators.deletion import (
    REPLACEMENT_DELETION_OPERATOR_ID,
)
from molhallulens.modules.error_injection.operators.substitution import SUBSTITUTION_OPERATOR_IDS
from molhallulens.infrastructure.validation.chain import (
    ARTIFACT_VALIDATOR_IDS,
    BUNDLE_INTEGRITY_VALIDATOR_ID,
    ArtifactValidationInput,
    ValidatorChain,
)

T048_FORMAT_VERSION = "t048_train_release_v1"
T048_REPORT_FORMAT_VERSION = "t048_train_build_report_v1"
T048_VALIDATION_FORMAT_VERSION = "t048_train_strict_validation_v1"
T048_TRAIN_ID = "t048_frozen_train_100_origin_v1"
T048_ORIGIN_COUNT = 100
T048_RECORDS_PER_ORIGIN = 8
T048_RECORD_COUNT = 800
T048_H_COUNT = 400
T048_N_COUNT = 400

from molhallulens.config.paths import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
DEFAULT_RELEASE_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset"
DEFAULT_REPORT_PATH = DEFAULT_PROJECT_ROOT / "Dataset/reports/t048_train_build.json"

_SUBTASK_COUNTS = MappingProxyType(
    {
        EditingSubtask.ADD: 34,
        EditingSubtask.DELETE: 33,
        EditingSubtask.SUBSTITUTE: 33,
    }
)
_STATUS_ACCEPTED = "accepted"
_STATUS_REJECTED = "rejected"

T048_TOKENIZER_FINGERPRINT = TokenizerFingerprint(
    tokenizer_name="ChemDFM-R-offset-contract-offline-whitespace-fixture",
    tokenizer_revision="t048-frozen-train-offline-v1",
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


class TrainSplitBuildError(RuntimeError):
    """Structured fail-closed T048 construction/publication error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("train build error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("train build error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("train build error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


def _t048_record_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace the reused T045 helper identity with the formal T048 build ID."""

    if not isinstance(value, Mapping):
        raise TypeError("serialized T048 record must be a mapping")
    result = dict(value)
    if result.pop("dry_run_id", None) is None:
        raise TrainSplitBuildError(
            "TRAIN_RECORD_IDENTITY",
            "reused serializer omitted its required build identity",
        )
    if "train_build_id" in result:
        raise TrainSplitBuildError(
            "TRAIN_RECORD_IDENTITY",
            "serialized record already contains a train build identity",
        )
    result["train_build_id"] = T048_TRAIN_ID
    return result


@dataclass(frozen=True, slots=True)
class TrainBuildAttempt:
    """One all-or-nothing origin recipe attempt."""

    attempt_index: int
    origin_id: str
    subtask: EditingSubtask
    case_id: str
    status: str
    emitted_record_count: int
    error_code: str | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        for value, name in (
            (self.origin_id, "origin_id"),
            (self.case_id, "case_id"),
            (self.status, "status"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        if self.status == _STATUS_ACCEPTED:
            if (
                self.emitted_record_count != T048_RECORDS_PER_ORIGIN
                or self.error_code is not None
                or self.exception_type is not None
            ):
                raise ValueError("accepted attempt must emit exactly eight records")
        elif self.status == _STATUS_REJECTED:
            if (
                self.emitted_record_count != 0
                or type(self.error_code) is not str
                or not self.error_code
                or type(self.exception_type) is not str
                or not self.exception_type
            ):
                raise ValueError("rejected attempt must emit zero records and an error")
        else:
            raise ValueError("unknown attempt status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": f"t048-attempt-{self.attempt_index:04d}",
            "attempt_index": self.attempt_index,
            "origin_id": self.origin_id,
            "subtask": self.subtask.value,
            "case_id": self.case_id,
            "status": self.status,
            "emitted_record_count": self.emitted_record_count,
            "atomic_unit": "complete_origin_bundle_4_pairs_8_records",
            "action": (
                "commit_complete_origin"
                if self.status == _STATUS_ACCEPTED
                else "discard_complete_origin_attempt"
            ),
            "error_code": self.error_code,
            "exception_type": self.exception_type,
            "committed": self.status == _STATUS_ACCEPTED,
        }


def _standard_case(subtask: EditingSubtask) -> ExtendedGoldenOriginCase:
    from molhallulens.modules.release.record_build import T044_GOLDEN_ORIGIN_CASES

    return next(
        case
        for case in T044_GOLDEN_ORIGIN_CASES
        if case.case_kind == "standard" and case.spec.normalized_subtask is subtask
    )


def _replace_full(
    case: ExtendedGoldenOriginCase,
    *,
    origin_id: str,
    operator_id: str,
    target_node_id: str,
    quota_bucket: str,
    suffix: str,
) -> ExtendedGoldenOriginCase:
    full = GoldenPolicySpec(
        PropagationPolicy.FULL_CF,
        operator_id,
        target_node_id,
        quota_bucket,
    )
    spec = replace(
        case.spec,
        origin_id=origin_id,
        policies=(*case.spec.policies[:2], full, case.spec.policies[3]),
    )
    return ExtendedGoldenOriginCase(
        case_id=f"train.{origin_id}.{suffix}",
        case_kind="t048_train_recipe",
        spec=spec,
        coverage_tags=("frozen_train", "validated_candidate_pool", suffix),
    )


_ADD_FRAGMENT_FULL_ORIGINS = frozenset(
    {
        "mol_edit.add_v2.0003",
        "mol_edit.add_v2.0013",
        "mol_edit.add_v2.0035",
        "mol_edit.add_v2.0044",
        "mol_edit.add_v2.0047",
        "mol_edit.add_v2.0051",
        "mol_edit.add_v2.0052",
        "mol_edit.add_v2.0057",
        "mol_edit.add_v2.0063",
        "mol_edit.add_v2.0071",
        "mol_edit.add_v2.0079",
        "mol_edit.add_v2.0101",
        "mol_edit.add_v2.0114",
        "mol_edit.add_v2.0119",
        "mol_edit.add_v2.0125",
        "mol_edit.add_v2.0148",
        "mol_edit.add_v2.0150",
        "mol_edit.add_v2.0174",
        "mol_edit.add_v2.0176",
        "mol_edit.add_v2.0185",
        "mol_edit.add_v2.0194",
        "mol_edit.add_v2.0214",
        "mol_edit.add_v2.0274",
    }
)
_SUB_ANCHOR_FULL_ORIGINS = frozenset(
    {
        "mol_edit.substitute_v2.0001",
        "mol_edit.substitute_v2.0080",
        "mol_edit.substitute_v2.0122",
        "mol_edit.substitute_v2.0123",
        "mol_edit.substitute_v2.0174",
        "mol_edit.substitute_v2.0222",
        "mol_edit.substitute_v2.0243",
        "mol_edit.substitute_v2.0281",
    }
)
_SUB_FRAGMENT_FULL_ORIGINS = frozenset(
    {
        "mol_edit.substitute_v2.0009",
        "mol_edit.substitute_v2.0134",
        "mol_edit.substitute_v2.0271",
        "mol_edit.substitute_v2.0283",
    }
)


def _special_train_cases(
    origin_id: str,
    subtask: EditingSubtask,
) -> tuple[ExtendedGoldenOriginCase, ...]:
    """Load narrowly scoped anomaly recipes without weakening normal recipes."""

    if origin_id == "mol_edit.delete_v2.0081":
        base = _standard_case(EditingSubtask.DELETE)
        special_policies = (
            GoldenPolicySpec(
                PropagationPolicy.STOP,
                REPLACEMENT_DELETION_OPERATOR_ID,
                "product",
                "group_fragment_identity",
            ),
            GoldenPolicySpec(
                PropagationPolicy.PARTIAL,
                REPLACEMENT_DELETION_OPERATOR_ID,
                "product",
                "product_dependency_cross_step",
                frozenset({"product_heavy", "product_rings"}),
            ),
            GoldenPolicySpec(
                PropagationPolicy.FULL_CF,
                REPLACEMENT_DELETION_OPERATOR_ID,
                "product",
                "valid_wrong_group_fragment",
            ),
            base.spec.policies[3],
        )
        return (
            ExtendedGoldenOriginCase(
                case_id="train.mol_edit.delete_v2.0081.replacement_aware",
                case_kind="t048_train_registered_anomaly",
                spec=replace(
                    base.spec,
                    origin_id=origin_id,
                    policies=special_policies,
                ),
                coverage_tags=(
                    "frozen_train",
                    "delete_with_replacement",
                    "replacement_aware_replay",
                ),
            ),
        )
    if origin_id in {
        "mol_edit.substitute_v2.0191",
        "mol_edit.substitute_v2.0276",
    }:
        try:
            from molhallulens.modules.error_injection.operators.substitution import (
                t048_substitution_boundary_cases,
            )
        except ImportError:
            return ()
        return tuple(t048_substitution_boundary_cases(origin_id))
    return ()


def train_recipe_candidates(
    origin_id: str,
    subtask: EditingSubtask,
) -> tuple[ExtendedGoldenOriginCase, ...]:
    """Return frozen primary and same-origin fallback recipes in audit order."""

    if type(origin_id) is not str or not origin_id:
        raise ValueError("origin_id must be non-empty text")
    if type(subtask) is not EditingSubtask:
        raise TypeError("subtask must be EditingSubtask")
    special = _special_train_cases(origin_id, subtask)
    if special:
        return special
    base = _standard_case(subtask)
    standard = ExtendedGoldenOriginCase(
        case_id=f"train.{origin_id}.standard",
        case_kind="t048_train_recipe",
        spec=replace(base.spec, origin_id=origin_id),
        coverage_tags=("frozen_train", "validated_candidate_pool", "standard"),
    )
    candidates: list[ExtendedGoldenOriginCase] = []
    if subtask is EditingSubtask.ADD:
        if origin_id in _ADD_FRAGMENT_FULL_ORIGINS:
            candidates.append(
                _replace_full(
                    base,
                    origin_id=origin_id,
                    operator_id=ADDITION_OPERATOR_IDS[2],
                    target_node_id="add_fragment",
                    quota_bucket="valid_wrong_group_fragment",
                    suffix="fragment_full",
                )
            )
        elif origin_id == "mol_edit.add_v2.0098":
            candidates.append(
                _replace_full(
                    base,
                    origin_id=origin_id,
                    operator_id=ADDITION_OPERATOR_IDS[0],
                    target_node_id="anchor_idx",
                    quota_bucket="valid_wrong_site_occurrence_regioisomer",
                    suffix="anchor_full",
                )
            )
    elif subtask is EditingSubtask.SUBSTITUTE:
        if origin_id in _SUB_ANCHOR_FULL_ORIGINS:
            candidates.append(
                _replace_full(
                    base,
                    origin_id=origin_id,
                    operator_id=SUBSTITUTION_OPERATOR_IDS[0],
                    target_node_id="anchor_idx",
                    quota_bucket="valid_wrong_site_occurrence_regioisomer",
                    suffix="anchor_full",
                )
            )
        elif origin_id in _SUB_FRAGMENT_FULL_ORIGINS:
            candidates.append(
                _replace_full(
                    base,
                    origin_id=origin_id,
                    operator_id=SUBSTITUTION_OPERATOR_IDS[2],
                    target_node_id="add_fragment",
                    quota_bucket="valid_wrong_group_fragment",
                    suffix="fragment_full",
                )
            )
    candidates.append(standard)
    unique: dict[tuple[GoldenPolicySpec, ...], ExtendedGoldenOriginCase] = {}
    for case in candidates:
        unique.setdefault(case.spec.policies, case)
    return tuple(unique.values())


@dataclass(frozen=True, slots=True)
class TrainSplitBuild:
    """Exactly 100 complete train origins and their immutable attempt ledger."""

    dataset_version: str
    global_seed: int
    origins: tuple[ExtendedGoldenOriginBuild, ...]
    attempts: tuple[TrainBuildAttempt, ...]
    split_manifest: VerifiedSplitManifest
    donor_pool: SplitDonorPool

    def __post_init__(self) -> None:
        origins = tuple(self.origins)
        attempts = tuple(self.attempts)
        if type(self.dataset_version) is not str or not self.dataset_version:
            raise ValueError("dataset_version must be non-empty text")
        if type(self.global_seed) is not int:
            raise TypeError("global_seed must be an integer")
        if type(self.split_manifest) is not VerifiedSplitManifest:
            raise TypeError("split_manifest must be loader verified")
        if type(self.donor_pool) is not SplitDonorPool:
            raise TypeError("donor_pool must be SplitDonorPool")
        if self.donor_pool.split != SplitName.TRAIN.value:
            raise TrainSplitBuildError(
                "TRAIN_DONOR_POOL_SPLIT",
                "T048 requires the verified train donor pool",
            )
        if any(type(item) is not ExtendedGoldenOriginBuild for item in origins):
            raise TypeError("origins must contain ExtendedGoldenOriginBuild values")
        if any(type(item) is not TrainBuildAttempt for item in attempts):
            raise TypeError("attempts must contain TrainBuildAttempt values")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "attempts", attempts)
        self._validate_complete_train()

    @property
    def artifacts(self) -> tuple[ArtifactValidationInput, ...]:
        return tuple(
            artifact for origin in self.origins for artifact in origin.artifacts
        )

    def _validate_complete_train(self) -> None:
        if len(self.origins) != T048_ORIGIN_COUNT:
            raise TrainSplitBuildError(
                "TRAIN_ORIGIN_COUNT",
                "T048 requires exactly 100 complete train origins",
                evidence={"observed": len(self.origins)},
            )
        origin_ids = tuple(origin.case.spec.origin_id for origin in self.origins)
        if len(origin_ids) != len(set(origin_ids)):
            raise TrainSplitBuildError(
                "TRAIN_ORIGIN_DUPLICATE",
                "T048 train origins must be unique",
            )
        expected_rows = tuple(
            row for row in self.split_manifest.rows if row.split is SplitName.TRAIN
        )
        expected_ids = {row.anonymous_sample_id for row in expected_rows}
        if set(origin_ids) != expected_ids or len(expected_rows) != T048_ORIGIN_COUNT:
            raise TrainSplitBuildError(
                "TRAIN_MANIFEST_COVERAGE",
                "built origins must equal the full frozen train assignment",
                evidence={
                    "missing_origin_ids": tuple(sorted(expected_ids - set(origin_ids))),
                    "unexpected_origin_ids": tuple(
                        sorted(set(origin_ids) - expected_ids)
                    ),
                },
            )
        subtask_counts = Counter(
            origin.case.spec.normalized_subtask for origin in self.origins
        )
        if subtask_counts != Counter(_SUBTASK_COUNTS):
            raise TrainSplitBuildError(
                "TRAIN_SUBTASK_COUNTS",
                "T048 subtask counts differ from the verified manifest",
            )
        artifacts = self.artifacts
        if len(artifacts) != T048_RECORD_COUNT:
            raise TrainSplitBuildError(
                "TRAIN_RECORD_COUNT",
                "T048 requires exactly 800 records",
                evidence={"observed": len(artifacts)},
            )
        record_ids = tuple(artifact.record_id for artifact in artifacts)
        if len(record_ids) != len(set(record_ids)):
            raise TrainSplitBuildError(
                "TRAIN_RECORD_DUPLICATE",
                "T048 record identities must be globally unique",
            )
        if Counter(artifact.draft.variant_label for artifact in artifacts) != Counter(
            {
                VariantLabel.HALLUCINATED: T048_H_COUNT,
                VariantLabel.FAITHFUL: T048_N_COUNT,
            }
        ):
            raise TrainSplitBuildError(
                "TRAIN_VARIANT_BALANCE",
                "T048 requires 400 hallucinated and 400 faithful records",
            )
        policy_labels = Counter(
            (artifact.draft.policy, artifact.draft.variant_label)
            for artifact in artifacts
        )
        expected_policy_labels = Counter(
            {
                (policy, label): T048_ORIGIN_COUNT
                for policy in PropagationPolicy
                for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
            }
        )
        if policy_labels != expected_policy_labels:
            raise TrainSplitBuildError(
                "TRAIN_POLICY_BALANCE",
                "each train policy requires exactly 100 H and 100 N records",
            )
        accepted = tuple(
            attempt for attempt in self.attempts if attempt.status == _STATUS_ACCEPTED
        )
        if (
            tuple(attempt.attempt_index for attempt in self.attempts)
            != tuple(range(len(self.attempts)))
            or len(accepted) != T048_ORIGIN_COUNT
            or tuple(attempt.origin_id for attempt in accepted) != origin_ids
            or any(
                attempt.emitted_record_count != 0
                for attempt in self.attempts
                if attempt.status == _STATUS_REJECTED
            )
        ):
            raise TrainSplitBuildError(
                "TRAIN_ATTEMPT_ATOMICITY",
                "attempt ledger contains partial or unbound output",
            )
        accepted_seen: set[str] = set()
        for attempt in self.attempts:
            if attempt.origin_id in accepted_seen:
                raise TrainSplitBuildError(
                    "TRAIN_ATTEMPT_AFTER_COMMIT",
                    "an origin has an attempt after its complete bundle committed",
                    evidence={"origin_id": attempt.origin_id},
                )
            if attempt.status == _STATUS_ACCEPTED:
                accepted_seen.add(attempt.origin_id)
        if accepted_seen != set(origin_ids):
            raise TrainSplitBuildError(
                "TRAIN_ATTEMPT_ATOMICITY",
                "every selected origin must terminate in one accepted attempt",
            )
        for origin in self.origins:
            origin_id = origin.case.spec.origin_id
            row = self.split_manifest.row_for_origin(origin_id)
            if (
                not origin.validation.all_pass
                or len(origin.artifacts) != T048_RECORDS_PER_ORIGIN
                or row.split is not SplitName.TRAIN
            ):
                raise TrainSplitBuildError(
                    "TRAIN_STRICT_VALIDATION",
                    "selected origin failed T043 or train binding",
                    evidence={"origin_id": origin_id},
                )
            if {
                (artifact.split, artifact.leakage_group_id)
                for artifact in origin.artifacts
            } != {(SplitName.TRAIN, row.leakage_group_id)}:
                raise TrainSplitBuildError(
                    "TRAIN_MANIFEST_BINDING",
                    "record split/group differs from verified manifest",
                    evidence={"origin_id": origin_id},
                )
            for execution in origin.golden.executions:
                donor_origin_id = _donor_origin_for_execution(execution)
                if donor_origin_id is not None:
                    _validate_donor_edge(
                        recipient_origin_id=origin_id,
                        donor_origin_id=donor_origin_id,
                        split="train",
                        manifest=self.split_manifest,
                        donor_pools={"train": self.donor_pool},
                    )
        for record in self.dataset_records():
            _assert_no_forbidden_key(
                record,
                _FORBIDDEN_DATASET_KEYS,
                "TRAIN_GT_LEAKAGE",
            )
        for collection in (
            self.dataset_records(),
            self.oracle_records(),
            self.state_records(),
            self.token_records(),
            self.provenance_records(),
        ):
            _assert_no_secret(collection)

    def dataset_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t048_record_identity(
                _dataset_record(origin, artifact, self.dataset_version)
            )
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def oracle_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t048_record_identity(_oracle_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def state_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t048_record_identity(_state_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def token_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t048_record_identity(_token_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def provenance_records(self) -> tuple[dict[str, Any], ...]:
        values = []
        for origin in self.origins:
            for artifact in origin.artifacts:
                value = _t048_record_identity(
                    _provenance_record(origin, artifact, self.dataset_version)
                )
                value["execution_mode"]["reason"] = (
                    "T048 frozen train build uses deterministic local candidate and "
                    "rule-renderer paths; live Poe status was not probed"
                )
                value["tokenizer"]["fingerprint"] = (
                    T048_TOKENIZER_FINGERPRINT.tokenizer_revision
                )
                values.append(value)
        return tuple(values)

    def validation_records(self) -> tuple[dict[str, Any], ...]:
        chain = ValidatorChain()
        return tuple(
            _t048_record_identity(
                _validation_record(origin, artifact, chain, self.dataset_version)
            )
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def selection_manifest(self) -> dict[str, Any]:
        return {
            "format_version": T048_FORMAT_VERSION,
            "train_build_id": T048_TRAIN_ID,
            "selection_split": "train",
            "selection_unit": "origin",
            "commit_unit": "complete_origin_bundle_4_pairs_8_records",
            "validation_or_test_used_for_selection": False,
            "selected": [
                {
                    "origin_id": origin.case.spec.origin_id,
                    "subtask": origin.case.spec.normalized_subtask.value,
                    "case_id": origin.case.case_id,
                    "leakage_group_id": origin.artifacts[0].leakage_group_id,
                    "record_count": len(origin.artifacts),
                    "policy_operator_ids": [
                        policy.operator_id for policy in origin.case.spec.policies
                    ],
                }
                for origin in self.origins
            ],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    def validation_report(self) -> dict[str, Any]:
        return {
            "format_version": T048_VALIDATION_FORMAT_VERSION,
            "train_build_id": T048_TRAIN_ID,
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
        policy_counts = Counter(
            (artifact.draft.policy.dataset_name, artifact.draft.variant_label.value)
            for artifact in artifacts
        )
        return {
            "format_version": T048_REPORT_FORMAT_VERSION,
            "train_build_id": T048_TRAIN_ID,
            "dataset_version": self.dataset_version,
            "global_seed": self.global_seed,
            "all_pass": True,
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
                "origin_counts_by_subtask": {
                    subtask.value: sum(
                        origin.case.spec.normalized_subtask is subtask
                        for origin in self.origins
                    )
                    for subtask in EditingSubtask
                },
                "policy_variant_counts": {
                    f"{policy}:{label}": count
                    for (policy, label), count in sorted(policy_counts.items())
                },
                "candidate_sources": dict(sorted(source_counts.items())),
                "attempt_count": len(self.attempts),
                "rejected_attempt_count": sum(rejection_counts.values()),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            },
            "strict_validation": {
                "artifact_gate_count": len(artifacts) * len(ARTIFACT_VALIDATOR_IDS),
                "bundle_gate_count": len(self.origins),
                "all_pass": True,
            },
            "isolation": {
                "candidate_selection_split": "train",
                "validation_used_for_candidate_selection": False,
                "test_used_for_candidate_selection": False,
                "verified_train_donor_edge_count": donor_edges,
                "cross_split_donor_edge_count": 0,
            },
            "execution": {
                "network_mode": "frozen_offline",
                "live_poe_attempted": False,
                "network_request_count": 0,
                "renderer": "deterministic-formal-v1",
                "tokenizer_backend": "deterministic_fast_offset_offline_fixture",
                "production_chemdfm_r_weights_loaded": False,
                "explicit_digest_or_sha_verification_performed": False,
            },
            "acceptance": {
                "one_hundred_complete_train_origins": len(self.origins) == 100,
                "eight_hundred_records": len(artifacts) == 800,
                "four_hundred_h_four_hundred_n": True,
                "each_policy_one_hundred_h_one_hundred_n": True,
                "all_t043_strict_validators_pass": True,
                "failed_recipe_emitted_record_count": 0,
                "atomic_backfill_unit": "complete_origin_bundle_4_pairs_8_records",
                "full_frozen_train_manifest_coverage": True,
                "gt_or_reference_artifact_leakage_count": 0,
                "secret_leakage_count": 0,
            },
            "limitations": [
                "T048 token labels use an explicit offline fast-offset fixture; production ChemDFM-R weights are loaded only in T051.",
                "No live Poe request or availability probe was performed for this deterministic train build.",
                "No explicit digest or SHA verification was performed.",
            ],
        }

    def artifact_payloads(self) -> Mapping[str, str]:
        payloads = {
            "records/train.jsonl": _render_jsonl(self.dataset_records()),
            "oracle/train.jsonl": _render_jsonl(self.oracle_records()),
            "state_graphs/train.jsonl": _render_jsonl(self.state_records()),
            "tokenized/chemdfm_r/train.jsonl": _render_jsonl(self.token_records()),
            "provenance/train.jsonl": _render_jsonl(self.provenance_records()),
            "reports/train_selection_manifest.json": _render_json(
                self.selection_manifest()
            ),
            "reports/train_validation_report.json": _render_json(
                self.validation_report()
            ),
            "reports/train_backfill_ledger.jsonl": _render_jsonl(
                tuple(attempt.to_dict() for attempt in self.attempts)
            ),
            "reports/train_build_report.json": _render_json(self.build_report()),
        }
        _assert_no_secret(payloads)
        return MappingProxyType(payloads)

    def render_report_json(self) -> str:
        return _render_json(self.build_report())


OriginBuilder = Callable[..., ExtendedGoldenOriginBuild]
RecipeResolver = Callable[[str, EditingSubtask], Sequence[ExtendedGoldenOriginCase]]


def build_t048_train_split(
    dataset_root: Path | None = None,
    manifest_root: Path | None = None,
    donor_pool_root: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    recipe_resolver: RecipeResolver = train_recipe_candidates,
    origin_builder: OriginBuilder = build_extended_origin,
) -> TrainSplitBuild:
    """Build all frozen train origins with complete-origin atomic recipe retry."""

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
    if not callable(recipe_resolver) or not callable(origin_builder):
        raise TypeError("recipe_resolver and origin_builder must be callable")
    records = ChemCoTMolEditAdapter().load(root)
    indexed = _joined_by_id(records)
    manifest = _load_verified_manifest(records, manifest_dir)
    donor_pool = _load_donor_pools(donor_dir, manifest)["train"]
    writer = TokenLabelSetWriter(
        _FrozenOfflineOffsetTokenizer(),
        T048_TOKENIZER_FINGERPRINT,
    )
    train_rows = tuple(
        sorted(
            (row for row in manifest.rows if row.split is SplitName.TRAIN),
            key=lambda row: row.anonymous_sample_id,
        )
    )
    if len(train_rows) != T048_ORIGIN_COUNT:
        raise TrainSplitBuildError(
            "TRAIN_MANIFEST_COUNT",
            "verified manifest does not contain exactly 100 train origins",
        )
    selected: list[ExtendedGoldenOriginBuild] = []
    attempts: list[TrainBuildAttempt] = []
    exhausted: list[dict[str, Any]] = []
    attempt_index = 0
    for row in train_rows:
        cases = tuple(recipe_resolver(row.anonymous_sample_id, row.subtask))
        if not cases or any(
            type(case) is not ExtendedGoldenOriginCase
            or case.spec.origin_id != row.anonymous_sample_id
            or case.spec.normalized_subtask is not row.subtask
            for case in cases
        ):
            raise TrainSplitBuildError(
                "TRAIN_RECIPE_SET_INVALID",
                "recipe resolver returned no exact same-origin recipes",
                evidence={"origin_id": row.anonymous_sample_id},
            )
        joined = indexed.get(row.anonymous_sample_id)
        if joined is None:
            raise TrainSplitBuildError(
                "TRAIN_ORIGIN_MISSING",
                "manifest train origin is absent from joined input",
                evidence={"origin_id": row.anonymous_sample_id},
            )
        accepted = False
        failures: list[str] = []
        for case in cases:
            try:
                built = origin_builder(
                    case,
                    config=loaded_config,
                    joined=joined,
                    manifest=manifest,
                    token_writer=writer,
                    global_seed=loaded_config.dataset.dataset.global_seed,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                code = getattr(error, "code", type(error).__name__.upper())
                if type(code) is not str or not code:
                    code = type(error).__name__.upper()
                attempts.append(
                    TrainBuildAttempt(
                        attempt_index=attempt_index,
                        origin_id=row.anonymous_sample_id,
                        subtask=row.subtask,
                        case_id=case.case_id,
                        status=_STATUS_REJECTED,
                        emitted_record_count=0,
                        error_code=code,
                        exception_type=type(error).__name__,
                    )
                )
                failures.append(code)
                attempt_index += 1
                continue
            if type(built) is not ExtendedGoldenOriginBuild:
                raise TypeError("origin_builder must return ExtendedGoldenOriginBuild")
            attempts.append(
                TrainBuildAttempt(
                    attempt_index=attempt_index,
                    origin_id=row.anonymous_sample_id,
                    subtask=row.subtask,
                    case_id=case.case_id,
                    status=_STATUS_ACCEPTED,
                    emitted_record_count=T048_RECORDS_PER_ORIGIN,
                )
            )
            attempt_index += 1
            selected.append(built)
            accepted = True
            break
        if not accepted:
            exhausted.append(
                {
                    "origin_id": row.anonymous_sample_id,
                    "subtask": row.subtask.value,
                    "recipe_count": len(cases),
                    "error_codes": tuple(failures),
                }
            )
    if exhausted:
        raise TrainSplitBuildError(
            "TRAIN_BACKFILL_EXHAUSTED",
            "same-origin validated recipe set cannot cover the frozen train split",
            evidence={
                "failed_origin_count": len(exhausted),
                "failures": tuple(exhausted),
                "accepted_origin_count": len(selected),
                "rejected_attempt_emitted_record_count": 0,
            },
        )
    return TrainSplitBuild(
        dataset_version=loaded_config.dataset.dataset.version_name,
        global_seed=loaded_config.dataset.dataset.global_seed,
        origins=tuple(selected),
        attempts=tuple(attempts),
        split_manifest=manifest,
        donor_pool=donor_pool,
    )


def _publish_payloads(root: Path, payloads: Mapping[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        target = root / relative
        if target.exists() and (
            not target.is_file() or target.read_text(encoding="utf-8") != payload
        ):
            raise TrainSplitBuildError(
                "TRAIN_ARTIFACT_CONFLICT",
                "existing train artifact differs from deterministic replay",
                evidence={"relative_path": relative},
            )
    missing = tuple(relative for relative in payloads if not (root / relative).exists())
    if not missing:
        return
    staging = Path(tempfile.mkdtemp(prefix=".t048-staging-", dir=root))
    try:
        for relative in missing:
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payloads[relative], encoding="utf-8", newline="\n")
        # The report is the completion marker and is published last.
        ordered = tuple(
            relative
            for relative in missing
            if relative != "reports/train_build_report.json"
        ) + tuple(
            relative
            for relative in missing
            if relative == "reports/train_build_report.json"
        )
        for relative in ordered:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_t048_train_artifacts(
    *,
    release_root: Path | None = None,
    report_path: Path | None = None,
    build: TrainSplitBuild | None = None,
) -> TrainSplitBuild:
    """Publish the complete validated train family without touching held-out files."""

    value = build_t048_train_split() if build is None else build
    if type(value) is not TrainSplitBuild:
        raise TypeError("build must be TrainSplitBuild or None")
    root = DEFAULT_RELEASE_ROOT if release_root is None else Path(release_root)
    report = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    payloads = value.artifact_payloads()
    report_payload = value.render_report_json()
    if report.exists() and (
        not report.is_file() or report.read_text(encoding="utf-8") != report_payload
    ):
        raise TrainSplitBuildError(
            "TRAIN_REPORT_CONFLICT",
            "existing T048 report differs from deterministic replay",
            evidence={"filename": report.name},
        )
    _publish_payloads(root, payloads)
    report.parent.mkdir(parents=True, exist_ok=True)
    if not report.exists():
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{report.name}.",
            dir=report.parent,
            delete=False,
        ) as handle:
            handle.write(report_payload)
            temporary = Path(handle.name)
        temporary.replace(report)
    return value


__all__ = [
    "DEFAULT_RELEASE_ROOT",
    "DEFAULT_REPORT_PATH",
    "T048_FORMAT_VERSION",
    "T048_H_COUNT",
    "T048_N_COUNT",
    "T048_ORIGIN_COUNT",
    "T048_RECORD_COUNT",
    "T048_REPORT_FORMAT_VERSION",
    "T048_TOKENIZER_FINGERPRINT",
    "T048_TRAIN_ID",
    "TrainBuildAttempt",
    "TrainSplitBuild",
    "TrainSplitBuildError",
    "build_t048_train_split",
    "train_recipe_candidates",
    "write_t048_train_artifacts",
]
