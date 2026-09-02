"""T050 final held-out test-split construction and publication.

The held-out split is deliberately built last.  Construction is enabled only
after the T047 shortcut design, T048 train build, and T049 validation build
have all published passing completion reports.  Candidate recipes, operators,
renderer behavior, ordering, and thresholds are never derived from test
outcomes: the builder delegates to the already-frozen development resolver and
permits test validation only as a fail-closed record-acceptance gate.

Every predeclared recipe attempt is provisional until its complete four-pair,
eight-record origin bundle passes the T043 chain.  A failed attempt emits no
records, never changes a frozen rule, and never triggers cross-split backfill.

Token projection remains an explicit offline T042 contract fixture.  T051 is
responsible for replacing it with the real ChemDFM-R tokenizer and weights.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
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
from molhallulens.infrastructure.validation.chain import (
    ARTIFACT_VALIDATOR_IDS,
    BUNDLE_INTEGRITY_VALIDATOR_ID,
    ArtifactValidationInput,
    ValidatorChain,
)

T050_FORMAT_VERSION = "t050_test_release_v1"
T050_REPORT_FORMAT_VERSION = "t050_test_build_report_v1"
T050_VALIDATION_FORMAT_VERSION = "t050_test_strict_validation_v1"
T050_ISOLATION_FORMAT_VERSION = "t050_test_isolation_declaration_v1"
T050_TEST_ID = "t050_frozen_test_25_origin_v1"
T050_ORIGIN_COUNT = 25
T050_RECORDS_PER_ORIGIN = 8
T050_RECORD_COUNT = 200
T050_H_COUNT = 100
T050_N_COUNT = 100

from molhallulens.config.paths import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
DEFAULT_RELEASE_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset"
DEFAULT_REPORT_PATH = DEFAULT_PROJECT_ROOT / "Dataset/reports/t050_test_build.json"
DEFAULT_T047_REPORT_PATH = (
    DEFAULT_PROJECT_ROOT / "Dataset/reports/t047_shortcut_audit.json"
)
DEFAULT_T048_REPORT_PATH = (
    DEFAULT_PROJECT_ROOT / "Dataset/reports/t048_train_build.json"
)
DEFAULT_T049_REPORT_PATH = (
    DEFAULT_PROJECT_ROOT / "Dataset/reports/t049_validation_build.json"
)

_SUBTASK_COUNTS = MappingProxyType(
    {
        EditingSubtask.ADD: 8,
        EditingSubtask.DELETE: 8,
        EditingSubtask.SUBSTITUTE: 9,
    }
)
_STATUS_ACCEPTED = "accepted"
_STATUS_REJECTED = "rejected"

T050_TOKENIZER_FINGERPRINT = TokenizerFingerprint(
    tokenizer_name="ChemDFM-R-offset-contract-offline-whitespace-fixture",
    tokenizer_revision="t050-frozen-test-offline-v1",
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


class TestSplitBuildError(RuntimeError):
    """Structured fail-closed T050 construction/publication error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("test build error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("test build error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("test build error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


def _t050_record_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace the reused T045 identity with the formal T050 test build ID."""

    if not isinstance(value, Mapping):
        raise TypeError("serialized T050 record must be a mapping")
    result = dict(value)
    if result.pop("dry_run_id", None) is None:
        raise TestSplitBuildError(
            "TEST_RECORD_IDENTITY",
            "reused serializer omitted its required build identity",
        )
    if "test_build_id" in result:
        raise TestSplitBuildError(
            "TEST_RECORD_IDENTITY",
            "serialized record already contains a test build identity",
        )
    result["test_build_id"] = T050_TEST_ID
    return result


def _load_json_report(path: Path, task_id: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise TestSplitBuildError(
            "TEST_BUILD_ORDER",
            f"{task_id} completion report must exist before T050 reads test inputs",
            evidence={"task_id": task_id, "filename": path.name},
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TestSplitBuildError(
            "TEST_BUILD_ORDER",
            f"{task_id} completion report is not readable JSON",
            evidence={"task_id": task_id, "exception_type": type(error).__name__},
        ) from error
    if not isinstance(value, Mapping) or value.get("all_pass") is not True:
        raise TestSplitBuildError(
            "TEST_BUILD_ORDER",
            f"{task_id} must report all_pass=true before T050",
            evidence={"task_id": task_id},
        )
    return value


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentPrerequisites:
    """Passing development reports checked before any held-out construction."""

    shortcut_report: Mapping[str, Any]
    train_report: Mapping[str, Any]
    validation_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = (
            (self.shortcut_report, "shortcut_report"),
            (self.train_report, "train_report"),
            (self.validation_report, "validation_report"),
        )
        for value, name in values:
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if value.get("all_pass") is not True:
                raise ValueError(f"{name} must report all_pass=true")
            object.__setattr__(self, name, MappingProxyType(dict(value)))


def _load_frozen_development_prerequisites(
    *,
    shortcut_report_path: Path,
    train_report_path: Path,
    validation_report_path: Path,
) -> FrozenDevelopmentPrerequisites:
    """Validate chronological and selection isolation before test is built."""

    shortcut = _load_json_report(shortcut_report_path, "T047")
    train = _load_json_report(train_report_path, "T048")
    validation = _load_json_report(validation_report_path, "T049")

    protocol = shortcut.get("audit_protocol")
    train_isolation = train.get("isolation")
    validation_isolation = validation.get("isolation")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("test_used_for_model_or_threshold_selection") is not False
        or not isinstance(train_isolation, Mapping)
        or train_isolation.get("test_used_for_candidate_selection") is not False
        or not isinstance(validation_isolation, Mapping)
        or validation_isolation.get("test_used_for_candidate_selection") is not False
    ):
        raise TestSplitBuildError(
            "TEST_ISOLATION_PREREQUISITE",
            "development reports do not prove that test was excluded from selection",
        )
    train_summary = train.get("summary")
    validation_summary = validation.get("summary")
    if (
        not isinstance(train_summary, Mapping)
        or train_summary.get("origin_count") != 100
        or train_summary.get("record_count") != 800
        or not isinstance(validation_summary, Mapping)
        or validation_summary.get("origin_count") != 25
        or validation_summary.get("record_count") != 200
    ):
        raise TestSplitBuildError(
            "TEST_BUILD_ORDER",
            "train and validation must be complete before the final test build",
        )
    return FrozenDevelopmentPrerequisites(shortcut, train, validation)


def frozen_test_recipe_candidates(
    origin_id: str,
    subtask: EditingSubtask,
) -> tuple[ExtendedGoldenOriginCase, ...]:
    """Return the pre-test frozen development recipes without test feedback.

    T049's resolver is finalized before this function is invoked.  T050 only
    relabels audit metadata; it does not add, remove, reorder, or alter policy,
    operator, target, renderer, quota, or candidate-source choices.
    """

    from molhallulens.modules.release.validation import validation_recipe_candidates

    frozen = tuple(validation_recipe_candidates(origin_id, subtask))
    result = []
    for index, case in enumerate(frozen):
        if type(case) is not ExtendedGoldenOriginCase:
            raise TypeError("frozen development resolver returned an invalid case")
        suffix = case.case_id.rsplit(".", maxsplit=1)[-1]
        result.append(
            ExtendedGoldenOriginCase(
                case_id=f"test.{origin_id}.{index:02d}.{suffix}",
                case_kind="t050_test_predeclared_frozen_recipe",
                spec=case.spec,
                coverage_tags=(
                    "frozen_before_test_build",
                    "no_test_feedback",
                    f"recipe_order_{index:02d}",
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TestBuildAttempt:
    """One all-or-nothing predeclared test-origin recipe attempt."""

    attempt_index: int
    origin_id: str
    subtask: EditingSubtask
    case_id: str
    frozen_recipe_index: int
    status: str
    emitted_record_count: int
    error_code: str | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if type(self.frozen_recipe_index) is not int or self.frozen_recipe_index < 0:
            raise ValueError("frozen_recipe_index must be non-negative")
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
                self.emitted_record_count != T050_RECORDS_PER_ORIGIN
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
            "attempt_id": f"t050-attempt-{self.attempt_index:04d}",
            "attempt_index": self.attempt_index,
            "origin_id": self.origin_id,
            "subtask": self.subtask.value,
            "case_id": self.case_id,
            "frozen_recipe_index": self.frozen_recipe_index,
            "recipe_order_mutated_after_test_observation": False,
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


@dataclass(frozen=True, slots=True)
class TestSplitBuild:
    """Exactly 25 complete held-out origins plus immutable attempt evidence."""

    dataset_version: str
    global_seed: int
    origins: tuple[ExtendedGoldenOriginBuild, ...]
    attempts: tuple[TestBuildAttempt, ...]
    split_manifest: VerifiedSplitManifest
    donor_pool: SplitDonorPool
    prerequisites: FrozenDevelopmentPrerequisites

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
        if self.donor_pool.split != SplitName.TEST.value:
            raise TestSplitBuildError(
                "TEST_DONOR_POOL_SPLIT",
                "T050 requires the verified test donor pool",
            )
        if type(self.prerequisites) is not FrozenDevelopmentPrerequisites:
            raise TypeError("prerequisites must be frozen development reports")
        if any(type(item) is not ExtendedGoldenOriginBuild for item in origins):
            raise TypeError("origins must contain ExtendedGoldenOriginBuild values")
        if any(type(item) is not TestBuildAttempt for item in attempts):
            raise TypeError("attempts must contain TestBuildAttempt values")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "attempts", attempts)
        self._validate_complete_test()

    @property
    def artifacts(self) -> tuple[ArtifactValidationInput, ...]:
        return tuple(
            artifact for origin in self.origins for artifact in origin.artifacts
        )

    def _validate_complete_test(self) -> None:
        if len(self.origins) != T050_ORIGIN_COUNT:
            raise TestSplitBuildError(
                "TEST_ORIGIN_COUNT",
                "T050 requires exactly 25 complete test origins",
                evidence={"observed": len(self.origins)},
            )
        origin_ids = tuple(origin.case.spec.origin_id for origin in self.origins)
        if len(origin_ids) != len(set(origin_ids)):
            raise TestSplitBuildError(
                "TEST_ORIGIN_DUPLICATE", "T050 test origins must be unique"
            )
        expected_rows = tuple(
            row for row in self.split_manifest.rows if row.split is SplitName.TEST
        )
        expected_ids = {row.anonymous_sample_id for row in expected_rows}
        if set(origin_ids) != expected_ids or len(expected_rows) != T050_ORIGIN_COUNT:
            raise TestSplitBuildError(
                "TEST_MANIFEST_COVERAGE",
                "built origins must equal the full frozen test assignment",
                evidence={
                    "missing_origin_ids": tuple(sorted(expected_ids - set(origin_ids))),
                    "unexpected_origin_ids": tuple(
                        sorted(set(origin_ids) - expected_ids)
                    ),
                },
            )
        if Counter(
            origin.case.spec.normalized_subtask for origin in self.origins
        ) != Counter(_SUBTASK_COUNTS):
            raise TestSplitBuildError(
                "TEST_SUBTASK_COUNTS",
                "T050 subtask counts differ from the verified manifest",
            )
        artifacts = self.artifacts
        if len(artifacts) != T050_RECORD_COUNT:
            raise TestSplitBuildError(
                "TEST_RECORD_COUNT",
                "T050 requires exactly 200 records",
                evidence={"observed": len(artifacts)},
            )
        record_ids = tuple(artifact.record_id for artifact in artifacts)
        if len(record_ids) != len(set(record_ids)):
            raise TestSplitBuildError(
                "TEST_RECORD_DUPLICATE", "T050 record identities must be unique"
            )
        if Counter(artifact.draft.variant_label for artifact in artifacts) != Counter(
            {
                VariantLabel.HALLUCINATED: T050_H_COUNT,
                VariantLabel.FAITHFUL: T050_N_COUNT,
            }
        ):
            raise TestSplitBuildError(
                "TEST_VARIANT_BALANCE",
                "T050 requires 100 hallucinated and 100 faithful records",
            )
        if Counter(
            (artifact.draft.policy, artifact.draft.variant_label)
            for artifact in artifacts
        ) != Counter(
            {
                (policy, label): T050_ORIGIN_COUNT
                for policy in PropagationPolicy
                for label in (VariantLabel.HALLUCINATED, VariantLabel.FAITHFUL)
            }
        ):
            raise TestSplitBuildError(
                "TEST_POLICY_BALANCE",
                "each policy requires exactly 25 H and 25 N test records",
            )
        accepted = tuple(
            attempt for attempt in self.attempts if attempt.status == _STATUS_ACCEPTED
        )
        if (
            tuple(attempt.attempt_index for attempt in self.attempts)
            != tuple(range(len(self.attempts)))
            or len(accepted) != T050_ORIGIN_COUNT
            or tuple(attempt.origin_id for attempt in accepted) != origin_ids
            or any(
                attempt.emitted_record_count != 0
                for attempt in self.attempts
                if attempt.status == _STATUS_REJECTED
            )
        ):
            raise TestSplitBuildError(
                "TEST_ATTEMPT_ATOMICITY",
                "attempt ledger contains partial or unbound output",
            )
        accepted_seen: set[str] = set()
        previous_recipe_index: dict[str, int] = {}
        for attempt in self.attempts:
            if attempt.origin_id in accepted_seen:
                raise TestSplitBuildError(
                    "TEST_ATTEMPT_AFTER_COMMIT",
                    "an origin has an attempt after its bundle committed",
                )
            expected_index = previous_recipe_index.get(attempt.origin_id, -1) + 1
            if attempt.frozen_recipe_index != expected_index:
                raise TestSplitBuildError(
                    "TEST_FROZEN_RECIPE_ORDER",
                    "test feedback cannot reorder or insert a recipe attempt",
                    evidence={"origin_id": attempt.origin_id},
                )
            previous_recipe_index[attempt.origin_id] = attempt.frozen_recipe_index
            if attempt.status == _STATUS_ACCEPTED:
                accepted_seen.add(attempt.origin_id)
        if accepted_seen != set(origin_ids):
            raise TestSplitBuildError(
                "TEST_ATTEMPT_ATOMICITY",
                "every test origin must terminate in one accepted attempt",
            )
        for origin in self.origins:
            origin_id = origin.case.spec.origin_id
            row = self.split_manifest.row_for_origin(origin_id)
            if (
                not origin.validation.all_pass
                or len(origin.artifacts) != T050_RECORDS_PER_ORIGIN
                or row.split is not SplitName.TEST
            ):
                raise TestSplitBuildError(
                    "TEST_STRICT_VALIDATION",
                    "selected origin failed T043 or test binding",
                    evidence={"origin_id": origin_id},
                )
            if {
                (artifact.split, artifact.leakage_group_id)
                for artifact in origin.artifacts
            } != {(SplitName.TEST, row.leakage_group_id)}:
                raise TestSplitBuildError(
                    "TEST_MANIFEST_BINDING",
                    "record split/group differs from verified manifest",
                    evidence={"origin_id": origin_id},
                )
            for execution in origin.golden.executions:
                donor_origin_id = _donor_origin_for_execution(execution)
                if donor_origin_id is not None:
                    _validate_donor_edge(
                        recipient_origin_id=origin_id,
                        donor_origin_id=donor_origin_id,
                        split="test",
                        manifest=self.split_manifest,
                        donor_pools={"test": self.donor_pool},
                    )
        for record in self.dataset_records():
            _assert_no_forbidden_key(record, _FORBIDDEN_DATASET_KEYS, "TEST_GT_LEAKAGE")
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
            _t050_record_identity(
                _dataset_record(origin, artifact, self.dataset_version)
            )
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def oracle_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t050_record_identity(_oracle_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def state_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t050_record_identity(_state_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def token_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _t050_record_identity(_token_record(artifact, self.dataset_version))
            for artifact in self.artifacts
        )

    def provenance_records(self) -> tuple[dict[str, Any], ...]:
        values = []
        for origin in self.origins:
            for artifact in origin.artifacts:
                value = _t050_record_identity(
                    _provenance_record(origin, artifact, self.dataset_version)
                )
                value["execution_mode"]["reason"] = (
                    "T050 final held-out build replays only pre-test frozen local "
                    "candidate and renderer rules; no live Poe status was probed"
                )
                value["tokenizer"]["fingerprint"] = (
                    T050_TOKENIZER_FINGERPRINT.tokenizer_revision
                )
                value["test_isolation"] = {
                    "recipe_frozen_before_test_build": True,
                    "test_used_to_add_or_reorder_recipes": False,
                    "test_used_to_modify_candidate_rules": False,
                    "test_used_to_modify_renderer": False,
                    "test_used_for_layer_selection": False,
                    "test_used_for_threshold_selection": False,
                    "strict_validation_used_only_for_record_acceptance": True,
                }
                values.append(value)
        return tuple(values)

    def validation_records(self) -> tuple[dict[str, Any], ...]:
        chain = ValidatorChain()
        return tuple(
            _t050_record_identity(
                _validation_record(origin, artifact, chain, self.dataset_version)
            )
            for origin in self.origins
            for artifact in origin.artifacts
        )

    def isolation_declaration(self) -> dict[str, Any]:
        return {
            "format_version": T050_ISOLATION_FORMAT_VERSION,
            "test_build_id": T050_TEST_ID,
            "all_pass": True,
            "build_order": {
                "test_built_last": True,
                "required_predecessors_checked_before_test_construction": [
                    "T047",
                    "T048",
                    "T049",
                ],
                "train_complete_before_test": True,
                "validation_complete_before_test": True,
            },
            "frozen_design": {
                "recipe_resolver": (
                    "molhallulens.modules.release.validation."
                    "validation_recipe_candidates"
                ),
                "recipe_order_frozen_before_test_build": True,
                "operator_rules_frozen_before_test_build": True,
                "candidate_generation_rules_frozen_before_test_build": True,
                "propagation_rules_frozen_before_test_build": True,
                "renderer_rules_frozen_before_test_build": True,
                "thresholds_frozen_before_test_build": True,
                "test_failure_may_add_remove_or_reorder_recipe": False,
                "test_failure_may_change_operator_or_candidate_rule": False,
                "test_failure_may_change_propagation_or_renderer": False,
            },
            "test_usage": {
                "used_for_candidate_rule_selection": False,
                "used_for_candidate_generation_tuning": False,
                "used_for_propagation_layer_selection": False,
                "used_for_detector_layer_selection": False,
                "used_for_detector_threshold_selection": False,
                "used_for_renderer_selection_or_tuning": False,
                "used_for_shortcut_threshold_selection": False,
                "used_for_strict_record_acceptance": True,
                "strict_acceptance_can_mutate_frozen_design": False,
                "diagnostic_results_feed_back_into_build": False,
            },
            "failure_semantics": {
                "commit_unit": "complete_origin_bundle_4_pairs_8_records",
                "failed_attempt_emitted_record_count": 0,
                "cross_split_backfill_allowed": False,
                "same_origin_predeclared_recipe_order_only": True,
                "exhaustion_action": "fail_closed_without_publication",
            },
            "detector_scope": {
                "formal_detector_training_authorized": False,
                "detector_layer_selected": False,
                "detector_threshold_selected": False,
                "pilot_use": "pipeline_schema_smoke_test_and_feasibility_audit_only",
            },
        }

    def selection_manifest(self) -> dict[str, Any]:
        return {
            "format_version": T050_FORMAT_VERSION,
            "test_build_id": T050_TEST_ID,
            "selection_split": "test",
            "selection_unit": "frozen_manifest_origin",
            "commit_unit": "complete_origin_bundle_4_pairs_8_records",
            "candidate_recipe_order_frozen_before_test_build": True,
            "test_used_to_modify_selection_rules": False,
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
            "format_version": T050_VALIDATION_FORMAT_VERSION,
            "test_build_id": T050_TEST_ID,
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
            "format_version": T050_REPORT_FORMAT_VERSION,
            "test_build_id": T050_TEST_ID,
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
                "candidate_selection_split": "pre_test_frozen_development_rules",
                "test_used_to_modify_candidate_rules": False,
                "test_used_to_modify_propagation_layer": False,
                "test_used_to_modify_renderer": False,
                "test_used_for_detector_layer_selection": False,
                "test_used_for_detector_threshold_selection": False,
                "test_validation_used_only_for_strict_record_acceptance": True,
                "diagnostic_feedback_into_build_count": 0,
                "verified_test_donor_edge_count": donor_edges,
                "cross_split_donor_edge_count": 0,
                "declaration_path": "reports/test_isolation_declaration.json",
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
                "twenty_five_complete_test_origins": len(self.origins) == 25,
                "two_hundred_records": len(artifacts) == 200,
                "one_hundred_h_one_hundred_n": True,
                "each_policy_twenty_five_h_twenty_five_n": True,
                "all_t043_strict_validators_pass": True,
                "failed_recipe_emitted_record_count": 0,
                "atomic_failure_unit": "complete_origin_bundle_4_pairs_8_records",
                "full_frozen_test_manifest_coverage": True,
                "gt_or_reference_artifact_leakage_count": 0,
                "secret_leakage_count": 0,
                "test_isolation_declaration_pass": True,
            },
            "limitations": [
                "T050 token labels are an offline fast-offset contract fixture; real ChemDFM-R tokenization and activation extraction are deferred to T051.",
                "No live Poe request or availability probe was performed for this deterministic final split build.",
                "Test strict validation accepts or rejects only predeclared frozen recipes and never changes recipe, operator, renderer, layer, or threshold rules.",
                "No explicit digest or SHA verification was performed.",
            ],
        }

    def artifact_payloads(self) -> Mapping[str, str]:
        payloads = {
            "records/test.jsonl": _render_jsonl(self.dataset_records()),
            "oracle/test.jsonl": _render_jsonl(self.oracle_records()),
            "state_graphs/test.jsonl": _render_jsonl(self.state_records()),
            "tokenized/chemdfm_r/test.jsonl": _render_jsonl(self.token_records()),
            "provenance/test.jsonl": _render_jsonl(self.provenance_records()),
            "reports/test_selection_manifest.json": _render_json(
                self.selection_manifest()
            ),
            "reports/test_validation_report.json": _render_json(
                self.validation_report()
            ),
            "reports/test_isolation_declaration.json": _render_json(
                self.isolation_declaration()
            ),
            "reports/test_backfill_ledger.jsonl": _render_jsonl(
                tuple(attempt.to_dict() for attempt in self.attempts)
            ),
            "reports/test_build_report.json": _render_json(self.build_report()),
        }
        _assert_no_secret(payloads)
        return MappingProxyType(payloads)

    def render_report_json(self) -> str:
        return _render_json(self.build_report())


def build_t050_test_split(
    dataset_root: Path | None = None,
    manifest_root: Path | None = None,
    donor_pool_root: Path | None = None,
    *,
    config: ConfigBundle | None = None,
    shortcut_report_path: Path | None = None,
    train_report_path: Path | None = None,
    validation_report_path: Path | None = None,
) -> TestSplitBuild:
    """Build the final held-out split from already-frozen development rules.

    The public construction API intentionally exposes no recipe, candidate,
    renderer, layer, threshold, or origin-builder injection point.
    """

    root = DEFAULT_DATASET_ROOT if dataset_root is None else Path(dataset_root)
    manifest_dir = (
        DEFAULT_MANIFEST_ROOT if manifest_root is None else Path(manifest_root)
    )
    donor_dir = (
        DEFAULT_DONOR_POOL_ROOT if donor_pool_root is None else Path(donor_pool_root)
    )
    shortcut_path = (
        DEFAULT_T047_REPORT_PATH
        if shortcut_report_path is None
        else Path(shortcut_report_path)
    )
    train_path = (
        DEFAULT_T048_REPORT_PATH
        if train_report_path is None
        else Path(train_report_path)
    )
    validation_path = (
        DEFAULT_T049_REPORT_PATH
        if validation_report_path is None
        else Path(validation_report_path)
    )

    # This gate intentionally precedes loading the adapter, manifest test rows,
    # donor test pool, or any origin content.
    prerequisites = _load_frozen_development_prerequisites(
        shortcut_report_path=shortcut_path,
        train_report_path=train_path,
        validation_report_path=validation_path,
    )
    loaded_config = load_config_bundle() if config is None else config
    if type(loaded_config) is not ConfigBundle:
        raise TypeError("config must be ConfigBundle or None")
    records = ChemCoTMolEditAdapter().load(root)
    indexed = _joined_by_id(records)
    manifest = _load_verified_manifest(records, manifest_dir)
    donor_pool = _load_donor_pools(donor_dir, manifest)["test"]
    writer = TokenLabelSetWriter(
        _FrozenOfflineOffsetTokenizer(),
        T050_TOKENIZER_FINGERPRINT,
    )
    test_rows = tuple(
        sorted(
            (row for row in manifest.rows if row.split is SplitName.TEST),
            key=lambda row: row.anonymous_sample_id,
        )
    )
    if len(test_rows) != T050_ORIGIN_COUNT:
        raise TestSplitBuildError(
            "TEST_MANIFEST_COUNT",
            "verified manifest does not contain exactly 25 test origins",
        )

    selected: list[ExtendedGoldenOriginBuild] = []
    attempts: list[TestBuildAttempt] = []
    exhausted: list[dict[str, Any]] = []
    attempt_index = 0
    for row in test_rows:
        cases = frozen_test_recipe_candidates(row.anonymous_sample_id, row.subtask)
        if not cases or any(
            type(case) is not ExtendedGoldenOriginCase
            or case.spec.origin_id != row.anonymous_sample_id
            or case.spec.normalized_subtask is not row.subtask
            for case in cases
        ):
            raise TestSplitBuildError(
                "TEST_RECIPE_SET_INVALID",
                "frozen resolver returned no exact same-origin recipes",
                evidence={"origin_id": row.anonymous_sample_id},
            )
        joined = indexed.get(row.anonymous_sample_id)
        if joined is None:
            raise TestSplitBuildError(
                "TEST_ORIGIN_MISSING",
                "manifest test origin is absent from joined input",
                evidence={"origin_id": row.anonymous_sample_id},
            )
        accepted = False
        failures: list[str] = []
        for frozen_recipe_index, case in enumerate(cases):
            try:
                built = build_extended_origin(
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
                    TestBuildAttempt(
                        attempt_index=attempt_index,
                        origin_id=row.anonymous_sample_id,
                        subtask=row.subtask,
                        case_id=case.case_id,
                        frozen_recipe_index=frozen_recipe_index,
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
                raise TypeError("origin builder must return ExtendedGoldenOriginBuild")
            attempts.append(
                TestBuildAttempt(
                    attempt_index=attempt_index,
                    origin_id=row.anonymous_sample_id,
                    subtask=row.subtask,
                    case_id=case.case_id,
                    frozen_recipe_index=frozen_recipe_index,
                    status=_STATUS_ACCEPTED,
                    emitted_record_count=T050_RECORDS_PER_ORIGIN,
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
                    "predeclared_recipe_count": len(cases),
                    "error_codes": tuple(failures),
                }
            )
    if exhausted:
        raise TestSplitBuildError(
            "TEST_FROZEN_RECIPES_EXHAUSTED",
            "pre-test frozen same-origin recipes cannot cover the held-out split",
            evidence={
                "failed_origin_count": len(exhausted),
                "failures": tuple(exhausted),
                "accepted_origin_count": len(selected),
                "rejected_attempt_emitted_record_count": 0,
                "frozen_design_mutated": False,
                "publication_performed": False,
            },
        )
    return TestSplitBuild(
        dataset_version=loaded_config.dataset.dataset.version_name,
        global_seed=loaded_config.dataset.dataset.global_seed,
        origins=tuple(selected),
        attempts=tuple(attempts),
        split_manifest=manifest,
        donor_pool=donor_pool,
        prerequisites=prerequisites,
    )


def _publish_payloads(root: Path, payloads: Mapping[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        target = root / relative
        if target.exists() and (
            not target.is_file() or target.read_text(encoding="utf-8") != payload
        ):
            raise TestSplitBuildError(
                "TEST_ARTIFACT_CONFLICT",
                "existing test artifact differs from deterministic replay",
                evidence={"relative_path": relative},
            )
    missing = tuple(relative for relative in payloads if not (root / relative).exists())
    if not missing:
        return
    staging = Path(tempfile.mkdtemp(prefix=".t050-staging-", dir=root))
    try:
        for relative in missing:
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payloads[relative], encoding="utf-8", newline="\n")
        ordered = tuple(
            relative
            for relative in missing
            if relative != "reports/test_build_report.json"
        ) + tuple(
            relative
            for relative in missing
            if relative == "reports/test_build_report.json"
        )
        for relative in ordered:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_t050_test_artifacts(
    *,
    release_root: Path | None = None,
    report_path: Path | None = None,
    build: TestSplitBuild | None = None,
) -> TestSplitBuild:
    """Publish the held-out family without touching train or validation files."""

    value = build_t050_test_split() if build is None else build
    if type(value) is not TestSplitBuild:
        raise TypeError("build must be TestSplitBuild or None")
    root = DEFAULT_RELEASE_ROOT if release_root is None else Path(release_root)
    report = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    payloads = value.artifact_payloads()
    if any(
        relative.endswith(("/train.jsonl", "/validation.jsonl"))
        or relative.startswith(("reports/train_", "reports/validation_"))
        for relative in payloads
    ):
        raise TestSplitBuildError(
            "TEST_PUBLICATION_SCOPE",
            "T050 payloads may not modify train or validation artifacts",
        )
    report_payload = value.render_report_json()
    if report.exists() and (
        not report.is_file() or report.read_text(encoding="utf-8") != report_payload
    ):
        raise TestSplitBuildError(
            "TEST_REPORT_CONFLICT",
            "existing T050 report differs from deterministic replay",
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
    "DEFAULT_T047_REPORT_PATH",
    "DEFAULT_T048_REPORT_PATH",
    "DEFAULT_T049_REPORT_PATH",
    "T050_FORMAT_VERSION",
    "T050_H_COUNT",
    "T050_ISOLATION_FORMAT_VERSION",
    "T050_N_COUNT",
    "T050_ORIGIN_COUNT",
    "T050_RECORD_COUNT",
    "T050_REPORT_FORMAT_VERSION",
    "T050_TEST_ID",
    "T050_TOKENIZER_FINGERPRINT",
    "FrozenDevelopmentPrerequisites",
    "TestBuildAttempt",
    "TestSplitBuild",
    "TestSplitBuildError",
    "build_t050_test_split",
    "frozen_test_recipe_candidates",
    "write_t050_test_artifacts",
]
