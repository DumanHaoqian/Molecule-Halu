"""Deterministic structural-operator candidate coverage audit.

T038 is an audit boundary, not a candidate generator.  It exercises the
operator-owned T019--T021 enumerators through the T017 resolution and T018
validation contracts, then aggregates immutable per-request observations.
Live Poe traffic is deliberately outside this module: deterministic LLM
fixtures are labelled as mocks and can never contribute to live participation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

from molhallulens.adapters import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.builders.edit_truth import derive_edit_truth
from molhallulens.builders.reference_dag import build_reference_dag
from molhallulens.candidates import (
    CandidateBuildResult,
    CandidateProposal,
    CandidateRequest,
    CandidateSourceError,
    DeterministicCandidateEngine,
    RDKitCandidateSource,
    RuleCandidateSource,
)
from molhallulens.config import load_config_bundle
from molhallulens.config.models import OperatorsConfig
from molhallulens.domain import (
    CandidateSourceType,
    EditingSubtask,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    ValueProvenance,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    DeletionPerturbator,
    PerturbationContext,
    SubstitutionPerturbator,
    task_record_from_joined_input,
)
from molhallulens.perturbators.editing.addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateEngine,
    _enumerate_addition_proposals,
)
from molhallulens.perturbators.editing.deletion import (
    DELETION_OPERATOR_IDS,
    DeletionCandidateEngine,
    _enumerate_deletion_proposals,
)
from molhallulens.perturbators.editing.substitution import (
    SUBSTITUTION_OPERATOR_IDS,
    SubstitutionCandidateEngine,
    _enumerate_substitution_proposals,
)
from molhallulens.perturbators.registry import (
    OperatorRegistration,
    OperatorRegistryError,
    PerturbatorRegistry,
)

REPORT_FORMAT_VERSION = "operator_candidate_coverage_v1"
FROZEN_DATASET_VERSION = "pilot_v1"
DEFAULT_REPORT_PATH = Path("Dataset/reports/operator_candidate_coverage.json")
LLM_MATERIAL_PARTICIPATION_TARGET = 0.5

ADDITION_STRUCTURAL_OPERATOR_IDS = ADDITION_OPERATOR_IDS[:7]
DELETION_STRUCTURAL_OPERATOR_IDS = DELETION_OPERATOR_IDS[:8]
SUBSTITUTION_STRUCTURAL_OPERATOR_IDS = SUBSTITUTION_OPERATOR_IDS[:8]
STRUCTURAL_OPERATOR_IDS = (
    *ADDITION_STRUCTURAL_OPERATOR_IDS,
    *DELETION_STRUCTURAL_OPERATOR_IDS,
    *SUBSTITUTION_STRUCTURAL_OPERATOR_IDS,
)

_NON_STRUCTURAL_FAMILIES = frozenset(
    {
        "numeric_count_claim",
        "nl_formal_internal_relation",
        "final_answer_identity",
    }
)


class CoverageBackend(StrEnum):
    RULE = "rule"
    RDKIT = "rdkit"
    HYBRID = "hybrid"
    LLM = "llm"


class CoverageExecutionMode(StrEnum):
    DETERMINISTIC_LOCAL = "deterministic_local"
    DETERMINISTIC_MOCK = "deterministic_mock"
    LIVE = "live"
    OFFLINE_NOT_EXECUTED = "offline_not_executed"


class OperatorCoverageError(RuntimeError):
    """Structured failure when a coverage claim cannot be proven."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("coverage error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("coverage error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("coverage error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def _text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{name} must be non-empty single-line text")
    return value


def _count_mapping(
    value: Mapping[str, int],
    *,
    name: str,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(sorted(value.items()))
    if any(
        type(key) is not str or not key or type(count) is not int or count < 0
        for key, count in result.items()
    ):
        raise ValueError(f"{name} must map non-empty strings to non-negative integers")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class CandidateCoverageObservation:
    """One operator/root/origin/backend request after deterministic validation."""

    operator_id: str
    subtask: EditingSubtask
    operator_family: str
    target_root: str
    origin_id: str
    request_id: str
    backend: CoverageBackend
    execution_mode: CoverageExecutionMode
    raw_proposals: int
    valid_candidates: int
    source_counts: Mapping[str, int] = field(default_factory=dict)
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    failure_code: str | None = None
    llm_materially_participated: bool = False
    fixture_id: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.operator_id, "operator_id"),
            (self.operator_family, "operator_family"),
            (self.target_root, "target_root"),
            (self.origin_id, "origin_id"),
            (self.request_id, "request_id"),
        ):
            _text(value, name)
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        if type(self.backend) is not CoverageBackend:
            raise TypeError("backend must be CoverageBackend")
        if type(self.execution_mode) is not CoverageExecutionMode:
            raise TypeError("execution_mode must be CoverageExecutionMode")
        for value, name in (
            (self.raw_proposals, "raw_proposals"),
            (self.valid_candidates, "valid_candidates"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        sources = _count_mapping(self.source_counts, name="source_counts")
        rejections = _count_mapping(self.rejection_counts, name="rejection_counts")
        if sum(sources.values()) != self.valid_candidates:
            raise ValueError("source_counts must exactly cover valid_candidates")
        if self.valid_candidates > self.raw_proposals:
            raise ValueError("valid_candidates cannot exceed raw_proposals")
        failed = self.valid_candidates == 0
        if failed != (self.failure_code is not None):
            raise ValueError("zero-valid observations require exactly one failure_code")
        if self.failure_code is not None:
            _text(self.failure_code, "failure_code")
        if type(self.llm_materially_participated) is not bool:
            raise TypeError("llm_materially_participated must be an exact bool")
        llm_count = sources.get(CandidateSourceType.LLM.value, 0)
        if self.llm_materially_participated != (llm_count > 0):
            raise ValueError(
                "LLM participation must equal validated LLM source presence"
            )
        if self.execution_mode is CoverageExecutionMode.DETERMINISTIC_MOCK:
            _text(self.fixture_id, "fixture_id")
        elif self.fixture_id is not None:
            raise ValueError("only deterministic mock observations may name a fixture")
        if self.execution_mode is CoverageExecutionMode.OFFLINE_NOT_EXECUTED:
            raise ValueError("offline declarations are report rows, not observations")
        if self.execution_mode is CoverageExecutionMode.DETERMINISTIC_LOCAL:
            allowed = {
                CoverageBackend.RULE: {CandidateSourceType.RULE.value},
                CoverageBackend.RDKIT: {CandidateSourceType.RDKIT.value},
                CoverageBackend.HYBRID: {
                    CandidateSourceType.RULE.value,
                    CandidateSourceType.RDKIT.value,
                },
            }.get(self.backend)
            if allowed is None or not set(sources) <= allowed:
                raise ValueError("deterministic-local backend/source mix is invalid")
        elif (
            self.execution_mode is CoverageExecutionMode.DETERMINISTIC_MOCK
            and self.backend not in {CoverageBackend.LLM, CoverageBackend.HYBRID}
        ):
            raise ValueError("mock LLM fixtures use only llm or hybrid backends")
        object.__setattr__(self, "source_counts", sources)
        object.__setattr__(self, "rejection_counts", rejections)

    @property
    def success(self) -> bool:
        return self.valid_candidates > 0

    @property
    def stable_key(self) -> tuple[str, ...]:
        return (
            self.operator_id,
            self.backend.value,
            self.execution_mode.value,
            self.target_root,
            self.origin_id,
            self.request_id,
        )


@dataclass(frozen=True, slots=True)
class OperatorBackendCoverage:
    backend: CoverageBackend
    execution_mode: CoverageExecutionMode
    execution_status: str
    requests: int
    raw_proposals: int
    valid_candidates: int
    successful_requests: int
    failures: int
    failure_rate: float | None
    request_coverage: float | None
    source_counts: Mapping[str, int]
    rejection_counts: Mapping[str, int]
    target_roots: tuple[str, ...]
    origin_count: int
    llm_material_request_count: int
    mock_counts_toward_live_participation: bool = False

    def __post_init__(self) -> None:
        if type(self.backend) is not CoverageBackend:
            raise TypeError("backend must be CoverageBackend")
        if type(self.execution_mode) is not CoverageExecutionMode:
            raise TypeError("execution_mode must be CoverageExecutionMode")
        if self.execution_status not in {"tested", "offline_not_executed"}:
            raise ValueError("unsupported execution_status")
        for value, name in (
            (self.requests, "requests"),
            (self.raw_proposals, "raw_proposals"),
            (self.valid_candidates, "valid_candidates"),
            (self.successful_requests, "successful_requests"),
            (self.failures, "failures"),
            (self.origin_count, "origin_count"),
            (self.llm_material_request_count, "llm_material_request_count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.requests != self.successful_requests + self.failures:
            raise ValueError("requests must equal successes plus failures")
        if self.execution_status == "offline_not_executed":
            if (
                any(
                    (
                        self.requests,
                        self.raw_proposals,
                        self.valid_candidates,
                        self.successful_requests,
                        self.failures,
                        self.origin_count,
                        self.llm_material_request_count,
                    )
                )
                or self.failure_rate is not None
                or self.request_coverage is not None
            ):
                raise ValueError("offline backend rows cannot claim measurements")
        else:
            if self.requests == 0:
                raise ValueError("tested backend rows require requests")
            expected_failure = self.failures / self.requests
            expected_coverage = self.successful_requests / self.requests
            if self.failure_rate != expected_failure:
                raise ValueError("failure_rate does not match exact counts")
            if self.request_coverage != expected_coverage:
                raise ValueError("request_coverage does not match exact counts")
        if self.mock_counts_toward_live_participation:
            raise ValueError("mock executions must never count as live participation")
        object.__setattr__(
            self,
            "source_counts",
            _count_mapping(self.source_counts, name="source_counts"),
        )
        object.__setattr__(
            self,
            "rejection_counts",
            _count_mapping(self.rejection_counts, name="rejection_counts"),
        )
        roots = tuple(sorted(set(self.target_roots)))
        if any(type(root) is not str or not root for root in roots):
            raise ValueError("target_roots must contain non-empty strings")
        object.__setattr__(self, "target_roots", roots)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "execution_mode": self.execution_mode.value,
            "execution_status": self.execution_status,
            "requests": self.requests,
            "raw_proposals": self.raw_proposals,
            "valid_candidates": self.valid_candidates,
            "successful_requests": self.successful_requests,
            "failures": self.failures,
            "failure_rate": self.failure_rate,
            "request_coverage": self.request_coverage,
            "source_counts": dict(self.source_counts),
            "rejection_counts": dict(self.rejection_counts),
            "target_roots": list(self.target_roots),
            "origin_count": self.origin_count,
            "llm_material_request_count": self.llm_material_request_count,
            "mock_counts_toward_live_participation": False,
        }


@dataclass(frozen=True, slots=True)
class StructuralOperatorCoverage:
    operator_id: str
    subtask: EditingSubtask
    operator_family: str
    root_fields: tuple[str, ...]
    backends: tuple[OperatorBackendCoverage, ...]
    verified_candidate_covered: bool
    verified_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.operator_id, "operator_id")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        _text(self.operator_family, "operator_family")
        roots = tuple(sorted(set(self.root_fields)))
        if not roots:
            raise ValueError("structural operator must expose a root field")
        rows = tuple(
            sorted(
                self.backends,
                key=lambda item: (item.backend.value, item.execution_mode.value),
            )
        )
        identities = tuple((item.backend, item.execution_mode) for item in rows)
        if len(identities) != len(set(identities)):
            raise ValueError("operator backend/mode rows must be unique")
        actual_sources = tuple(
            sorted(
                {
                    source
                    for row in rows
                    if row.execution_mode is CoverageExecutionMode.DETERMINISTIC_LOCAL
                    for source, count in row.source_counts.items()
                    if count > 0
                }
            )
        )
        if self.verified_sources != actual_sources:
            raise ValueError(
                "verified_sources differs from deterministic-local evidence"
            )
        if self.verified_candidate_covered != bool(actual_sources):
            raise ValueError(
                "coverage assertion differs from deterministic-local evidence"
            )
        object.__setattr__(self, "root_fields", roots)
        object.__setattr__(self, "backends", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "subtask": self.subtask.value,
            "operator_family": self.operator_family,
            "root_fields": list(self.root_fields),
            "verified_candidate_covered": self.verified_candidate_covered,
            "verified_sources": list(self.verified_sources),
            "backends": [item.to_dict() for item in self.backends],
        }


@dataclass(frozen=True, slots=True)
class LLMParticipationMeasurement:
    target_fraction: float
    measurement_unit: str
    numerator_definition: str
    denominator_definition: str
    live_execution_status: str
    live_material_variants: int
    live_structural_h_variants: int
    live_fraction: float | None
    target_met: bool | None
    deterministic_mock_material_requests: int
    deterministic_mock_requests: int
    deterministic_mock_fraction: float | None
    mock_counts_toward_live_fraction: bool = False

    def __post_init__(self) -> None:
        if self.target_fraction != LLM_MATERIAL_PARTICIPATION_TARGET:
            raise ValueError("LLM material participation target is frozen at one half")
        for value, name in (
            (self.measurement_unit, "measurement_unit"),
            (self.numerator_definition, "numerator_definition"),
            (self.denominator_definition, "denominator_definition"),
            (self.live_execution_status, "live_execution_status"),
        ):
            _text(value, name)
        for value, name in (
            (self.live_material_variants, "live_material_variants"),
            (self.live_structural_h_variants, "live_structural_h_variants"),
            (
                self.deterministic_mock_material_requests,
                "deterministic_mock_material_requests",
            ),
            (self.deterministic_mock_requests, "deterministic_mock_requests"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.live_structural_h_variants == 0:
            if self.live_fraction is not None or self.target_met is not None:
                raise ValueError(
                    "offline live participation must be unmeasured, not zero"
                )
        else:
            expected = self.live_material_variants / self.live_structural_h_variants
            if self.live_fraction != expected:
                raise ValueError("live_fraction differs from live variant counts")
            if self.target_met != (expected >= self.target_fraction):
                raise ValueError("target_met differs from the measured live fraction")
        if self.deterministic_mock_requests == 0:
            if self.deterministic_mock_fraction is not None:
                raise ValueError("empty mock fixture has no measured fraction")
        elif self.deterministic_mock_fraction != (
            self.deterministic_mock_material_requests / self.deterministic_mock_requests
        ):
            raise ValueError("mock fraction differs from exact fixture counts")
        if self.mock_counts_toward_live_fraction:
            raise ValueError("deterministic mocks must never count as live LLM use")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fraction": self.target_fraction,
            "measurement_unit": self.measurement_unit,
            "numerator_definition": self.numerator_definition,
            "denominator_definition": self.denominator_definition,
            "live_execution_status": self.live_execution_status,
            "live_material_variants": self.live_material_variants,
            "live_structural_h_variants": self.live_structural_h_variants,
            "live_fraction": self.live_fraction,
            "target_met": self.target_met,
            "deterministic_mock_material_requests": (
                self.deterministic_mock_material_requests
            ),
            "deterministic_mock_requests": self.deterministic_mock_requests,
            "deterministic_mock_fraction": self.deterministic_mock_fraction,
            "mock_counts_toward_live_fraction": False,
        }


@dataclass(frozen=True, slots=True)
class OperatorCandidateCoverageReport:
    dataset_version: str
    operator_count: int
    operators: tuple[StructuralOperatorCoverage, ...]
    all_structural_operators_covered: bool
    aggregate_source_counts: Mapping[str, int]
    deterministic_local_source_counts: Mapping[str, int]
    deterministic_mock_source_counts: Mapping[str, int]
    live_source_counts: Mapping[str, int]
    llm_participation: LLMParticipationMeasurement
    report_format_version: str = REPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.dataset_version != FROZEN_DATASET_VERSION:
            raise ValueError("unsupported coverage dataset version")
        if self.report_format_version != REPORT_FORMAT_VERSION:
            raise ValueError("unsupported coverage report version")
        rows = tuple(sorted(self.operators, key=lambda item: item.operator_id))
        if self.operator_count != len(rows):
            raise ValueError("operator_count differs from operator rows")
        if tuple(item.operator_id for item in rows) != tuple(
            sorted(STRUCTURAL_OPERATOR_IDS)
        ):
            raise ValueError(
                "coverage report does not contain exact structural operators"
            )
        actual_all = all(item.verified_candidate_covered for item in rows)
        if self.all_structural_operators_covered != actual_all:
            raise ValueError("aggregate structural coverage assertion is inconsistent")
        if type(self.llm_participation) is not LLMParticipationMeasurement:
            raise TypeError("llm_participation must be LLMParticipationMeasurement")
        for name in (
            "aggregate_source_counts",
            "deterministic_local_source_counts",
            "deterministic_mock_source_counts",
            "live_source_counts",
        ):
            object.__setattr__(
                self,
                name,
                _count_mapping(getattr(self, name), name=name),
            )
        combined = Counter(self.deterministic_local_source_counts)
        combined.update(self.deterministic_mock_source_counts)
        combined.update(self.live_source_counts)
        if dict(sorted(combined.items())) != dict(self.aggregate_source_counts):
            raise ValueError("aggregate source counts do not match execution modes")
        object.__setattr__(self, "operators", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "report_format_version": self.report_format_version,
            "dataset_version": self.dataset_version,
            "audit_execution": {
                "network_requests": 0,
                "secret_reads": 0,
                "live_llm_status": "offline_not_executed",
                "deterministic_mock_is_live": False,
            },
            "structural_operator_definition": {
                "excluded_operator_families": sorted(_NON_STRUCTURAL_FAMILIES),
                "addition_operator_count": len(ADDITION_STRUCTURAL_OPERATOR_IDS),
                "deletion_operator_count": len(DELETION_STRUCTURAL_OPERATOR_IDS),
                "substitution_operator_count": len(
                    SUBSTITUTION_STRUCTURAL_OPERATOR_IDS
                ),
            },
            "operator_count": self.operator_count,
            "all_structural_operators_covered": self.all_structural_operators_covered,
            "aggregate_source_counts": dict(self.aggregate_source_counts),
            "deterministic_local_source_counts": dict(
                self.deterministic_local_source_counts
            ),
            "deterministic_mock_source_counts": dict(
                self.deterministic_mock_source_counts
            ),
            "live_source_counts": dict(self.live_source_counts),
            "llm_participation": self.llm_participation.to_dict(),
            "operators": [item.to_dict() for item in self.operators],
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


def _aggregate_backend(
    backend: CoverageBackend,
    execution_mode: CoverageExecutionMode,
    observations: Sequence[CandidateCoverageObservation],
) -> OperatorBackendCoverage:
    values = tuple(observations)
    if not values:
        return OperatorBackendCoverage(
            backend=backend,
            execution_mode=CoverageExecutionMode.OFFLINE_NOT_EXECUTED,
            execution_status="offline_not_executed",
            requests=0,
            raw_proposals=0,
            valid_candidates=0,
            successful_requests=0,
            failures=0,
            failure_rate=None,
            request_coverage=None,
            source_counts={},
            rejection_counts={},
            target_roots=(),
            origin_count=0,
            llm_material_request_count=0,
        )
    requests = len(values)
    successes = sum(item.success for item in values)
    failures = requests - successes
    sources: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    for item in values:
        sources.update(item.source_counts)
        rejections.update(item.rejection_counts)
    return OperatorBackendCoverage(
        backend=backend,
        execution_mode=execution_mode,
        execution_status="tested",
        requests=requests,
        raw_proposals=sum(item.raw_proposals for item in values),
        valid_candidates=sum(item.valid_candidates for item in values),
        successful_requests=successes,
        failures=failures,
        failure_rate=failures / requests,
        request_coverage=successes / requests,
        source_counts=dict(sources),
        rejection_counts=dict(rejections),
        target_roots=tuple(item.target_root for item in values),
        origin_count=len({item.origin_id for item in values}),
        llm_material_request_count=sum(
            item.llm_materially_participated for item in values
        ),
    )


def _registration_map(
    operators_config: OperatorsConfig,
) -> Mapping[str, OperatorRegistration]:
    registry = PerturbatorRegistry.from_perturbator_types(
        (AdditionPerturbator, DeletionPerturbator, SubstitutionPerturbator),
        operators_config=operators_config,
    )
    values = {
        registration.operator_id: registration
        for registration in registry.registrations_for()
        if registration.operator_family not in _NON_STRUCTURAL_FAMILIES
    }
    if tuple(sorted(values)) != tuple(sorted(STRUCTURAL_OPERATOR_IDS)):
        raise OperatorCoverageError(
            "STRUCTURAL_OPERATOR_SET_DRIFT",
            "registry structural members differ from the frozen T019--T021 set",
            evidence={
                "expected": sorted(STRUCTURAL_OPERATOR_IDS),
                "actual": sorted(values),
            },
        )
    return MappingProxyType(values)


def build_operator_candidate_coverage_report(
    observations: Iterable[CandidateCoverageObservation],
    *,
    operators_config: OperatorsConfig,
    dataset_version: str = FROZEN_DATASET_VERSION,
) -> OperatorCandidateCoverageReport:
    """Aggregate order-independent observations and prove local coverage."""

    if type(operators_config) is not OperatorsConfig:
        raise TypeError("operators_config must be OperatorsConfig")
    if dataset_version != FROZEN_DATASET_VERSION:
        raise ValueError("dataset_version must be pilot_v1")
    registrations = _registration_map(operators_config)
    values = tuple(sorted(observations, key=lambda item: item.stable_key))
    if any(type(item) is not CandidateCoverageObservation for item in values):
        raise TypeError("observations must contain CandidateCoverageObservation values")
    identities = tuple(item.stable_key for item in values)
    if len(identities) != len(set(identities)):
        raise OperatorCoverageError(
            "DUPLICATE_COVERAGE_OBSERVATION",
            "operator coverage observations must have unique request identities",
        )
    unknown = sorted({item.operator_id for item in values} - set(registrations))
    if unknown:
        raise OperatorCoverageError(
            "UNKNOWN_STRUCTURAL_OPERATOR",
            "coverage observations name non-structural operators",
            evidence={"operator_ids": unknown},
        )

    operator_rows: list[StructuralOperatorCoverage] = []
    local_sources: Counter[str] = Counter()
    mock_sources: Counter[str] = Counter()
    live_sources: Counter[str] = Counter()
    for operator_id, registration in sorted(registrations.items()):
        operator_values = tuple(
            item for item in values if item.operator_id == operator_id
        )
        if any(
            item.subtask is not registration.subtask
            or item.operator_family != registration.operator_family
            or item.target_root not in registration.spec.root_fields
            for item in operator_values
        ):
            raise OperatorCoverageError(
                "OPERATOR_METADATA_MISMATCH",
                "coverage observation escaped its registry declaration",
                evidence={"operator_id": operator_id},
            )
        rows: list[OperatorBackendCoverage] = []
        for backend, mode in (
            (CoverageBackend.RULE, CoverageExecutionMode.DETERMINISTIC_LOCAL),
            (CoverageBackend.RDKIT, CoverageExecutionMode.DETERMINISTIC_LOCAL),
            (CoverageBackend.HYBRID, CoverageExecutionMode.DETERMINISTIC_LOCAL),
            (CoverageBackend.LLM, CoverageExecutionMode.DETERMINISTIC_MOCK),
            (CoverageBackend.HYBRID, CoverageExecutionMode.DETERMINISTIC_MOCK),
        ):
            selected = tuple(
                item
                for item in operator_values
                if item.backend is backend and item.execution_mode is mode
            )
            rows.append(_aggregate_backend(backend, mode, selected))
        rows.extend(
            (
                _aggregate_backend(
                    CoverageBackend.LLM,
                    CoverageExecutionMode.LIVE,
                    (),
                ),
                _aggregate_backend(
                    CoverageBackend.HYBRID,
                    CoverageExecutionMode.LIVE,
                    (),
                ),
            )
        )
        verified_sources = tuple(
            sorted(
                {
                    source
                    for row in rows
                    if row.execution_mode is CoverageExecutionMode.DETERMINISTIC_LOCAL
                    for source, count in row.source_counts.items()
                    if count > 0
                }
            )
        )
        operator_rows.append(
            StructuralOperatorCoverage(
                operator_id=operator_id,
                subtask=registration.subtask,
                operator_family=registration.operator_family,
                root_fields=tuple(registration.spec.root_fields),
                backends=tuple(rows),
                verified_candidate_covered=bool(verified_sources),
                verified_sources=verified_sources,
            )
        )
        for item in operator_values:
            target = (
                local_sources
                if item.execution_mode is CoverageExecutionMode.DETERMINISTIC_LOCAL
                else mock_sources
                if item.execution_mode is CoverageExecutionMode.DETERMINISTIC_MOCK
                else live_sources
            )
            target.update(item.source_counts)

    uncovered = tuple(
        item.operator_id
        for item in operator_rows
        if not item.verified_candidate_covered
    )
    if uncovered:
        raise OperatorCoverageError(
            "STRUCTURAL_OPERATOR_UNCOVERED",
            "at least one structural operator has no deterministic verified candidate",
            evidence={"operator_ids": list(uncovered)},
        )
    mock_values = tuple(
        item
        for item in values
        if item.execution_mode is CoverageExecutionMode.DETERMINISTIC_MOCK
    )
    mock_material = sum(item.llm_materially_participated for item in mock_values)
    llm_measurement = LLMParticipationMeasurement(
        target_fraction=LLM_MATERIAL_PARTICIPATION_TARGET,
        measurement_unit="accepted_structural_h_variants",
        numerator_definition=(
            "accepted live structural H variants whose validated selected pool "
            "contains at least one LLM-sourced candidate"
        ),
        denominator_definition="all accepted live structural H variants",
        live_execution_status="offline_not_executed",
        live_material_variants=0,
        live_structural_h_variants=0,
        live_fraction=None,
        target_met=None,
        deterministic_mock_material_requests=mock_material,
        deterministic_mock_requests=len(mock_values),
        deterministic_mock_fraction=(
            None if not mock_values else mock_material / len(mock_values)
        ),
    )
    aggregate = Counter(local_sources)
    aggregate.update(mock_sources)
    aggregate.update(live_sources)
    return OperatorCandidateCoverageReport(
        dataset_version=dataset_version,
        operator_count=len(operator_rows),
        operators=tuple(operator_rows),
        all_structural_operators_covered=True,
        aggregate_source_counts=dict(aggregate),
        deterministic_local_source_counts=dict(local_sources),
        deterministic_mock_source_counts=dict(mock_sources),
        live_source_counts=dict(live_sources),
        llm_participation=llm_measurement,
    )


class _UnusedDownstreamPorts:
    def propagate(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("coverage audit does not propagate candidates")

    def render(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("coverage audit does not render candidates")

    def validate_reference(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("coverage audit does not validate full artifacts")

    def validate_artifact(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("coverage audit does not validate full artifacts")

    def project(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("coverage audit does not project labels")


@dataclass(frozen=True, slots=True)
class _FamilyHarness:
    subtask: EditingSubtask
    marker: str
    perturbator_type: type
    candidate_engine_type: type
    structural_operator_ids: tuple[str, ...]
    enumerator: Callable[..., Iterable[CandidateProposal]]


@dataclass(frozen=True, slots=True)
class _CapturedCandidateSource:
    """Replay one measured source call through the ordinary T018 engine."""

    source_type: CandidateSourceType
    proposals: tuple[CandidateProposal, ...]
    error: CandidateSourceError | None = field(default=None, repr=False)

    def propose(self, request: CandidateRequest) -> tuple[CandidateProposal, ...]:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        if self.error is not None:
            raise self.error
        return self.proposals


_FAMILY_HARNESSES = (
    _FamilyHarness(
        EditingSubtask.ADD,
        ".add_v2.",
        AdditionPerturbator,
        AdditionCandidateEngine,
        ADDITION_STRUCTURAL_OPERATOR_IDS,
        _enumerate_addition_proposals,
    ),
    _FamilyHarness(
        EditingSubtask.DELETE,
        ".delete_v2.",
        DeletionPerturbator,
        DeletionCandidateEngine,
        DELETION_STRUCTURAL_OPERATOR_IDS,
        _enumerate_deletion_proposals,
    ),
    _FamilyHarness(
        EditingSubtask.SUBSTITUTE,
        ".substitute_v2.",
        SubstitutionPerturbator,
        SubstitutionCandidateEngine,
        SUBSTITUTION_STRUCTURAL_OPERATOR_IDS,
        _enumerate_substitution_proposals,
    ),
)


def _perturbator(harness: _FamilyHarness, operators_config: OperatorsConfig) -> Any:
    candidate_engine = harness.candidate_engine_type(operators_config=operators_config)
    unused = _UnusedDownstreamPorts()
    return harness.perturbator_type(
        candidate_engine=candidate_engine,
        propagator=unused,
        renderer=unused,
        validators=unused,
        label_projector=unused,
    )


def _policy(registration: OperatorRegistration) -> PropagationPolicy:
    if PropagationPolicy.STOP in registration.spec.supported_policies:
        return PropagationPolicy.STOP
    if PropagationPolicy.FULL_CF in registration.spec.supported_policies:
        return PropagationPolicy.FULL_CF
    raise OperatorCoverageError(
        "STRUCTURAL_POLICY_UNAVAILABLE",
        "structural coverage requires STOP or FULL_CF",
        evidence={"operator_id": registration.operator_id},
    )


def _context(
    *,
    joined: JoinedInputRecord,
    artifact: Any,
    truth: Any,
    task_record: Any,
    registration: OperatorRegistration,
    target_root: str,
    source_mode: CandidateSourceType,
) -> PerturbationContext[Any]:
    recipe = PerturbationRecipe(
        recipe_id=(
            f"t038:{joined.anonymous_sample_id}:{registration.operator_id}:"
            f"{target_root}:{source_mode.value}"
        ),
        origin_id=joined.anonymous_sample_id,
        operator_id=registration.operator_id,
        policy=_policy(registration),
        target_node_id=target_root,
        candidate_source_mode=source_mode,
        variant_index=0,
        derived_seed=20260829,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=128,
            length_bucket="t038",
        ),
        candidate_difficulty_bucket="coverage",
        renderer_style_id="deterministic-coverage",
    )
    return PerturbationContext(
        record=task_record,
        recipe=recipe,
        state_schema=artifact.state_dag.schema,
        reference_graph=artifact.state_dag,
        truth=truth,
    )


def _result_counts(
    result: CandidateBuildResult,
) -> tuple[dict[str, int], dict[str, int]]:
    sources = Counter(item.source.value for item in result.pool.candidates)
    rejections = Counter(item.code.value for item in result.rejections)
    for code in result.pool.rejection_codes:
        if code not in rejections:
            rejections[code] += 1
    return dict(sources), dict(rejections)


def _failure_code(result: CandidateBuildResult) -> str | None:
    if result.pool.candidates:
        return None
    if result.pool.rejection_codes:
        return min(result.pool.rejection_codes)
    return "NO_VALID_CANDIDATE"


def _mock_llm_proposal(
    proposal: CandidateProposal,
    *,
    fixture_id: str,
) -> CandidateProposal:
    patch = proposal.patch
    new_value = replace(
        patch.new_value,
        provenance=ValueProvenance.LLM,
        locally_valid=None,
        oracle_match=None,
        confidence=None,
    )
    candidate_id = f"mock.{patch.candidate_id}"
    return replace(
        proposal,
        proposal_id=f"llm-fixture:{proposal.proposal_id}",
        patch=replace(
            patch,
            candidate_id=candidate_id,
            new_value=new_value,
            source=CandidateSourceType.LLM,
            metadata={
                **dict(patch.metadata),
                "coverage_fixture_id": fixture_id,
                "execution_mode": CoverageExecutionMode.DETERMINISTIC_MOCK.value,
            },
        ),
    )


def _failure_observation(
    *,
    registration: OperatorRegistration,
    target_root: str,
    joined: JoinedInputRecord,
    backend: CoverageBackend,
    mode: CoverageExecutionMode,
    request_id: str,
    failure_code: str,
    fixture_id: str | None = None,
) -> CandidateCoverageObservation:
    return CandidateCoverageObservation(
        operator_id=registration.operator_id,
        subtask=registration.subtask,
        operator_family=registration.operator_family,
        target_root=target_root,
        origin_id=joined.anonymous_sample_id,
        request_id=request_id,
        backend=backend,
        execution_mode=mode,
        raw_proposals=0,
        valid_candidates=0,
        source_counts={},
        rejection_counts={failure_code: 1},
        failure_code=failure_code,
        fixture_id=fixture_id,
    )


def _probe_deterministic_backend(
    *,
    request: CandidateRequest,
    registration: OperatorRegistration,
    joined: JoinedInputRecord,
    target_root: str,
    backend: CoverageBackend,
    enumerator: Callable[..., Iterable[CandidateProposal]],
) -> tuple[CandidateCoverageObservation, CandidateBuildResult]:
    source_types = {
        CoverageBackend.RULE: (CandidateSourceType.RULE,),
        CoverageBackend.RDKIT: (CandidateSourceType.RDKIT,),
        CoverageBackend.HYBRID: (
            CandidateSourceType.RULE,
            CandidateSourceType.RDKIT,
        ),
    }[backend]
    sources = tuple(
        RuleCandidateSource(partial(enumerator, source=source_type))
        if source_type is CandidateSourceType.RULE
        else RDKitCandidateSource(partial(enumerator, source=source_type))
        for source_type in source_types
    )
    captured_sources: list[_CapturedCandidateSource] = []
    raw = 0
    for source in sources:
        try:
            proposals = tuple(source.propose(request))
        except CandidateSourceError as error:
            captured_sources.append(
                _CapturedCandidateSource(
                    source_type=source.source_type,
                    proposals=(),
                    error=error,
                )
            )
        else:
            raw += len(proposals)
            captured_sources.append(
                _CapturedCandidateSource(
                    source_type=source.source_type,
                    proposals=proposals,
                )
            )
    result = DeterministicCandidateEngine(tuple(captured_sources)).build_pool(request)
    source_counts, rejection_counts = _result_counts(result)
    observation = CandidateCoverageObservation(
        operator_id=registration.operator_id,
        subtask=registration.subtask,
        operator_family=registration.operator_family,
        target_root=target_root,
        origin_id=joined.anonymous_sample_id,
        request_id=request.request_id,
        backend=backend,
        execution_mode=CoverageExecutionMode.DETERMINISTIC_LOCAL,
        raw_proposals=max(raw, len(result.pool.candidates)),
        valid_candidates=len(result.pool.candidates),
        source_counts=source_counts,
        rejection_counts=rejection_counts,
        failure_code=_failure_code(result),
    )
    return observation, result


def _probe_llm_fixture(
    *,
    request: CandidateRequest,
    registration: OperatorRegistration,
    joined: JoinedInputRecord,
    target_root: str,
    deterministic_result: CandidateBuildResult,
    backend: CoverageBackend,
) -> CandidateCoverageObservation:
    fixture_id = "t038.t018_replay_fixture.v1"
    request_id = f"{request.request_id}:mock:{backend.value}"
    if not deterministic_result.ranked_candidates:
        return _failure_observation(
            registration=registration,
            target_root=target_root,
            joined=joined,
            backend=backend,
            mode=CoverageExecutionMode.DETERMINISTIC_MOCK,
            request_id=request_id,
            failure_code="MOCK_FIXTURE_SOURCE_UNAVAILABLE",
            fixture_id=fixture_id,
        )
    from molhallulens.candidates.hybrid_engine import T018CandidateGate

    proposal = _mock_llm_proposal(
        deterministic_result.ranked_candidates[0].proposal,
        fixture_id=fixture_id,
    )
    gated = T018CandidateGate().validate(
        request,
        (proposal,),
        allowed_sources=frozenset({CandidateSourceType.LLM}),
    )
    source_counts, rejection_counts = _result_counts(gated)
    return CandidateCoverageObservation(
        operator_id=registration.operator_id,
        subtask=registration.subtask,
        operator_family=registration.operator_family,
        target_root=target_root,
        origin_id=joined.anonymous_sample_id,
        request_id=request_id,
        backend=backend,
        execution_mode=CoverageExecutionMode.DETERMINISTIC_MOCK,
        raw_proposals=1,
        valid_candidates=len(gated.pool.candidates),
        source_counts=source_counts,
        rejection_counts=rejection_counts,
        failure_code=_failure_code(gated),
        llm_materially_participated=bool(gated.pool.candidates),
        fixture_id=fixture_id,
    )


def audit_structural_operator_candidate_coverage(
    dataset_root: Path,
    *,
    operators_config: OperatorsConfig | None = None,
    max_origins_per_subtask: int | None = None,
) -> OperatorCandidateCoverageReport:
    """Run an ordered real-corpus T038 audit without Poe access.

    The audit makes the canonically ordered 50-origin subtask corpus available
    to each operator/root/backend and stops that search after its first
    verified candidate. ``max_origins_per_subtask`` optionally lowers that
    search bound. Report rows expose exact request and origin counts, so this
    coverage proof cannot be mistaken for an exhaustive success-rate study.
    """

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be pathlib.Path")
    if max_origins_per_subtask is not None and (
        type(max_origins_per_subtask) is not int or max_origins_per_subtask <= 0
    ):
        raise ValueError("max_origins_per_subtask must be a positive integer or None")
    config = (
        load_config_bundle().operators if operators_config is None else operators_config
    )
    if type(config) is not OperatorsConfig:
        raise TypeError("operators_config must be OperatorsConfig or None")
    registrations = _registration_map(config)
    joined_records = tuple(
        sorted(
            ChemCoTMolEditAdapter().load(dataset_root),
            key=lambda item: item.anonymous_sample_id,
        )
    )
    if len(joined_records) != 150:
        raise OperatorCoverageError(
            "CORPUS_SIZE_MISMATCH",
            "T038 requires the exact 150-origin Pilot corpus",
            evidence={"observed": len(joined_records), "expected": 150},
        )
    prepared: dict[str, tuple[Any, Any, Any]] = {}
    for joined in joined_records:
        artifact = build_reference_dag(joined)
        truth = derive_edit_truth(artifact)
        prepared[joined.anonymous_sample_id] = (
            artifact,
            truth,
            task_record_from_joined_input(joined),
        )

    observations: list[CandidateCoverageObservation] = []
    for harness in _FAMILY_HARNESSES:
        family_records = tuple(
            item
            for item in joined_records
            if harness.marker in item.anonymous_sample_id
        )
        if len(family_records) != 50:
            raise OperatorCoverageError(
                "SUBTASK_CORPUS_SIZE_MISMATCH",
                "each editing subtask must contribute exactly 50 origins",
                evidence={
                    "subtask": harness.subtask.value,
                    "observed": len(family_records),
                },
            )
        if max_origins_per_subtask is not None:
            family_records = family_records[:max_origins_per_subtask]
        perturbator = _perturbator(harness, config)
        registry = PerturbatorRegistry.from_perturbator_types(
            (harness.perturbator_type,),
            operators_config=config,
        )
        for operator_id in harness.structural_operator_ids:
            registration = registrations[operator_id]
            for target_root in sorted(registration.spec.root_fields):
                successful_backends: set[CoverageBackend] = set()
                mock_recorded = False
                for joined in family_records:
                    artifact, truth, task_record = prepared[joined.anonymous_sample_id]
                    hybrid_result: CandidateBuildResult | None = None
                    hybrid_request: CandidateRequest | None = None
                    for backend, source_mode in (
                        (CoverageBackend.RULE, CandidateSourceType.RULE),
                        (CoverageBackend.RDKIT, CandidateSourceType.RDKIT),
                        (CoverageBackend.HYBRID, CandidateSourceType.HYBRID),
                    ):
                        if backend in successful_backends:
                            continue
                        context = _context(
                            joined=joined,
                            artifact=artifact,
                            truth=truth,
                            task_record=task_record,
                            registration=registration,
                            target_root=target_root,
                            source_mode=source_mode,
                        )
                        try:
                            resolution = registry.resolve(perturbator, context)
                            request = CandidateRequest(
                                context=context,
                                resolution=resolution,
                            )
                        except OperatorRegistryError as error:
                            observations.append(
                                _failure_observation(
                                    registration=registration,
                                    target_root=target_root,
                                    joined=joined,
                                    backend=backend,
                                    mode=CoverageExecutionMode.DETERMINISTIC_LOCAL,
                                    request_id=context.recipe.recipe_id,
                                    failure_code=error.code,
                                )
                            )
                            continue
                        observation, result = _probe_deterministic_backend(
                            request=request,
                            registration=registration,
                            joined=joined,
                            target_root=target_root,
                            backend=backend,
                            enumerator=harness.enumerator,
                        )
                        observations.append(observation)
                        if observation.success:
                            successful_backends.add(backend)
                        if backend is CoverageBackend.HYBRID:
                            hybrid_result = result
                            hybrid_request = request
                    if (
                        not mock_recorded
                        and hybrid_result is not None
                        and hybrid_request is not None
                        and hybrid_result.pool.candidates
                    ):
                        observations.extend(
                            _probe_llm_fixture(
                                request=hybrid_request,
                                registration=registration,
                                joined=joined,
                                target_root=target_root,
                                deterministic_result=hybrid_result,
                                backend=backend,
                            )
                            for backend in (CoverageBackend.LLM, CoverageBackend.HYBRID)
                        )
                        mock_recorded = True
                    if len(successful_backends) == 3 and mock_recorded:
                        break
                if not mock_recorded:
                    joined = family_records[-1]
                    for backend in (CoverageBackend.LLM, CoverageBackend.HYBRID):
                        observations.append(
                            _failure_observation(
                                registration=registration,
                                target_root=target_root,
                                joined=joined,
                                backend=backend,
                                mode=CoverageExecutionMode.DETERMINISTIC_MOCK,
                                request_id=(
                                    f"t038:{joined.anonymous_sample_id}:"
                                    f"{registration.operator_id}:{target_root}:"
                                    f"mock:{backend.value}"
                                ),
                                failure_code="MOCK_FIXTURE_REQUEST_UNAVAILABLE",
                                fixture_id="t038.t018_replay_fixture.v1",
                            )
                        )
    return build_operator_candidate_coverage_report(
        observations,
        operators_config=config,
    )


def write_operator_candidate_coverage_report(
    report: OperatorCandidateCoverageReport,
    *,
    path: Path = DEFAULT_REPORT_PATH,
) -> None:
    if type(report) is not OperatorCandidateCoverageReport:
        raise TypeError("report must be OperatorCandidateCoverageReport")
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = report.to_json_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    path.write_bytes(payload)


__all__ = [
    "ADDITION_STRUCTURAL_OPERATOR_IDS",
    "DEFAULT_REPORT_PATH",
    "DELETION_STRUCTURAL_OPERATOR_IDS",
    "FROZEN_DATASET_VERSION",
    "LLM_MATERIAL_PARTICIPATION_TARGET",
    "REPORT_FORMAT_VERSION",
    "STRUCTURAL_OPERATOR_IDS",
    "SUBSTITUTION_STRUCTURAL_OPERATOR_IDS",
    "CandidateCoverageObservation",
    "CoverageBackend",
    "CoverageExecutionMode",
    "LLMParticipationMeasurement",
    "OperatorBackendCoverage",
    "OperatorCandidateCoverageReport",
    "OperatorCoverageError",
    "StructuralOperatorCoverage",
    "audit_structural_operator_candidate_coverage",
    "build_operator_candidate_coverage_report",
    "write_operator_candidate_coverage_report",
]
