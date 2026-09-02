"""Aggregation and fail-closed corpus APIs for the four reference gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from molhallulens.modules.ingestion import JoinedInputRecord
from molhallulens.modules.reference.anomaly_registry import (
    AnomalyRegistryError,
    classify_edit_truth,
)
from molhallulens.modules.reference.builder import ReferenceDAGArtifact
from molhallulens.core import (
    EditingSubtask,
    EditTruth,
    OperationSubtype,
    Severity,
    ValidationIssue,
    ValidationReport,
    ValidationStage,
)

from .reference_gates import (
    GRAPH_EDIT_VALIDATOR,
    INPUT_RECORD_VALIDATOR,
    RDKIT_STRUCTURE_VALIDATOR,
    REFERENCE_DAG_VALIDATOR,
    VALIDATION_GATE_IDS,
    VALIDATION_GATE_STAGES,
    GraphEditValidator,
    InputRecordValidator,
    RDKitStructureValidator,
    ReferenceDAGValidator,
)


_COMBINED_VALIDATOR_ID = "molhallulens.reference_validation.v1"
_REPORT_FORMAT_VERSION = "reference_validation_report_v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    return value


def _issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "severity": issue.severity.value,
        "stage": issue.stage.value,
        "anonymous_sample_ids": list(issue.node_ids),
        "message": issue.message,
        "evidence": _json_safe(issue.evidence),
    }


@dataclass(frozen=True, slots=True)
class OriginValidationInput:
    """The three immutable reference artifacts required by the validation chain."""

    record: JoinedInputRecord
    artifact: ReferenceDAGArtifact
    edit_truth: EditTruth

    def __post_init__(self) -> None:
        if type(self.record) is not JoinedInputRecord:
            raise TypeError("record must be a JoinedInputRecord")
        if type(self.artifact) is not ReferenceDAGArtifact:
            raise TypeError("artifact must be a ReferenceDAGArtifact")
        if type(self.edit_truth) is not EditTruth:
            raise TypeError("edit_truth must be an EditTruth")


@dataclass(frozen=True, slots=True)
class OriginValidationReport:
    """One origin's ordered, immutable four-gate ValidationReport ledger."""

    anonymous_sample_id: str
    normalized_subtask: EditingSubtask
    operation_subtype: OperationSubtype | None
    stage_reports: tuple[ValidationReport, ...]

    def __post_init__(self) -> None:
        if type(self.anonymous_sample_id) is not str or not self.anonymous_sample_id:
            raise ValueError("anonymous_sample_id must be non-empty text")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if self.operation_subtype is not None and type(
            self.operation_subtype
        ) is not OperationSubtype:
            raise TypeError("operation_subtype must be OperationSubtype or None")
        reports = tuple(self.stage_reports)
        if any(type(report) is not ValidationReport for report in reports):
            raise TypeError("stage_reports must contain ValidationReport values")
        if tuple(report.validator_id for report in reports) != VALIDATION_GATE_IDS:
            raise ValueError("stage_reports must contain each validation gate exactly once")
        for report in reports:
            expected_stage = VALIDATION_GATE_STAGES[report.validator_id]
            if any(issue.stage is not expected_stage for issue in report.issues):
                raise ValueError("stage report contains an issue from another gate")
            if any(
                issue.node_ids != (self.anonymous_sample_id,)
                for issue in report.issues
            ):
                raise ValueError("stage issue origin IDs must match the report origin")
        object.__setattr__(self, "stage_reports", reports)

    @property
    def report(self) -> ValidationReport:
        return ValidationReport.combine(_COMBINED_VALIDATOR_ID, self.stage_reports)

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        return self.report.issues

    @property
    def issue_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def all_pass(self) -> bool:
        return self.report.all_pass and self.operation_subtype is not None

    @property
    def status(self) -> str:
        return "passed" if self.all_pass else "failed"

    def to_dict(self) -> dict[str, Any]:
        gates = []
        for report in self.stage_reports:
            stage = VALIDATION_GATE_STAGES[report.validator_id]
            gates.append(
                {
                    "validator_id": report.validator_id,
                    "stage": stage.value,
                    "all_pass": report.all_pass,
                    "issues": [_issue_to_dict(issue) for issue in report.issues],
                }
            )
        return {
            "anonymous_sample_id": self.anonymous_sample_id,
            "normalized_subtask": self.normalized_subtask.value,
            "operation_subtype": (
                None
                if self.operation_subtype is None
                else self.operation_subtype.value
            ),
            "status": self.status,
            "gates": gates,
        }


@dataclass(frozen=True, slots=True)
class ReferenceValidationCorpusReport:
    """Deterministic validation ledger with one entry for every input origin."""

    origins: tuple[OriginValidationReport, ...]
    format_version: str = _REPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        origins = tuple(self.origins)
        if any(type(origin) is not OriginValidationReport for origin in origins):
            raise TypeError("origins must contain OriginValidationReport values")
        origins = tuple(sorted(origins, key=lambda item: item.anonymous_sample_id))
        ids = tuple(origin.anonymous_sample_id for origin in origins)
        if len(ids) != len(set(ids)):
            raise ValueError("origin validation reports must use unique IDs")
        if self.format_version != _REPORT_FORMAT_VERSION:
            raise ValueError("unknown reference validation report version")
        object.__setattr__(self, "origins", origins)

    @property
    def attempted(self) -> int:
        return len(self.origins)

    @property
    def passed(self) -> int:
        return sum(origin.all_pass for origin in self.origins)

    @property
    def failed(self) -> int:
        return self.attempted - self.passed

    @property
    def all_pass(self) -> bool:
        return self.attempted > 0 and self.failed == 0

    @property
    def issue_count(self) -> int:
        return sum(len(origin.issues) for origin in self.origins)

    @property
    def counts_by_subtask(self) -> tuple[tuple[EditingSubtask, int], ...]:
        counts = Counter(origin.normalized_subtask for origin in self.origins)
        return tuple((subtask, counts.get(subtask, 0)) for subtask in EditingSubtask)

    @property
    def counts_by_operation_subtype(
        self,
    ) -> tuple[tuple[OperationSubtype, int], ...]:
        counts = Counter(
            origin.operation_subtype
            for origin in self.origins
            if origin.operation_subtype is not None
        )
        return tuple(
            (subtype, counts.get(subtype, 0)) for subtype in OperationSubtype
        )

    @property
    def gate_failure_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(
            report.validator_id
            for origin in self.origins
            for report in origin.stage_reports
            if not report.all_pass
        )
        return tuple((validator_id, counts.get(validator_id, 0)) for validator_id in VALIDATION_GATE_IDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "summary": {
                "attempted": self.attempted,
                "passed": self.passed,
                "failed": self.failed,
                "all_pass": self.all_pass,
                "issue_count": self.issue_count,
                "counts_by_subtask": {
                    subtask.value: count for subtask, count in self.counts_by_subtask
                },
                "counts_by_operation_subtype": {
                    subtype.value: count
                    for subtype, count in self.counts_by_operation_subtype
                },
                "gate_failure_counts": dict(self.gate_failure_counts),
            },
            "origins": [origin.to_dict() for origin in self.origins],
        }


class ReferenceValidationError(RuntimeError):
    """Raised when one validation request cannot pass fail-closed gates."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("ReferenceValidationError report must be ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"reference validation failed ({codes})")


class ReferenceValidationCorpusError(RuntimeError):
    """Raised by strict corpus audit while preserving every origin report."""

    def __init__(self, result: ReferenceValidationCorpusReport) -> None:
        if type(result) is not ReferenceValidationCorpusReport:
            raise TypeError("result must be ReferenceValidationCorpusReport")
        self.result = result
        super().__init__(f"reference validation corpus has {result.failed} failed origin(s)")


@dataclass(frozen=True, slots=True)
class ReferenceValidationPipeline:
    input_validator: InputRecordValidator = INPUT_RECORD_VALIDATOR
    reference_validator: ReferenceDAGValidator = REFERENCE_DAG_VALIDATOR
    rdkit_validator: RDKitStructureValidator = RDKIT_STRUCTURE_VALIDATOR
    graph_validator: GraphEditValidator = GRAPH_EDIT_VALIDATOR

    def __post_init__(self) -> None:
        for validator, expected_type in (
            (self.input_validator, InputRecordValidator),
            (self.reference_validator, ReferenceDAGValidator),
            (self.rdkit_validator, RDKitStructureValidator),
            (self.graph_validator, GraphEditValidator),
        ):
            if type(validator) is not expected_type:
                raise TypeError("pipeline validators must use the four concrete gate types")

    def validate(self, item: OriginValidationInput) -> OriginValidationReport:
        if type(item) is not OriginValidationInput:
            raise TypeError("ReferenceValidationPipeline.validate requires OriginValidationInput")
        stage_reports = (
            self.input_validator.validate(item.record),
            self.reference_validator.validate(item.record, item.artifact),
            self.rdkit_validator.validate(item.record, item.artifact, item.edit_truth),
            self.graph_validator.validate(item.record, item.artifact, item.edit_truth),
        )
        try:
            operation_subtype = classify_edit_truth(item.edit_truth).operation_subtype
        except AnomalyRegistryError:
            operation_subtype = None
        return OriginValidationReport(
            anonymous_sample_id=item.record.anonymous_sample_id,
            normalized_subtask=item.artifact.normalized_subtask,
            operation_subtype=operation_subtype,
            stage_reports=stage_reports,
        )

    def validate_strict(self, item: OriginValidationInput) -> OriginValidationReport:
        report = self.validate(item)
        if not report.all_pass:
            raise ReferenceValidationError(report.report)
        return report

    def audit(
        self,
        items: Iterable[OriginValidationInput],
        *,
        require_all_pass: bool = True,
    ) -> ReferenceValidationCorpusReport:
        if type(require_all_pass) is not bool:
            raise TypeError("require_all_pass must be bool")
        try:
            items = tuple(items)
        except TypeError as error:
            raise TypeError("items must be iterable") from error
        if not items:
            raise ValueError("reference validation corpus cannot be empty")
        if any(type(item) is not OriginValidationInput for item in items):
            raise TypeError("items must contain OriginValidationInput values")
        ids = tuple(item.record.anonymous_sample_id for item in items)
        duplicates = tuple(sorted({origin_id for origin_id in ids if ids.count(origin_id) > 1}))
        if duplicates:
            raise ReferenceValidationError(
                ValidationReport(
                    _COMBINED_VALIDATOR_ID,
                    (
                        ValidationIssue(
                            code="DUPLICATE_VALIDATION_ORIGIN",
                            severity=Severity.FATAL,
                            stage=ValidationStage.INPUT_RECORD,
                            node_ids=duplicates,
                            message="validation corpus contains duplicate origin IDs",
                            evidence={"duplicate_count": len(duplicates)},
                        ),
                    ),
                )
            )
        result = ReferenceValidationCorpusReport(
            tuple(
                self.validate(item)
                for item in sorted(items, key=lambda value: value.record.anonymous_sample_id)
            )
        )
        if require_all_pass and not result.all_pass:
            raise ReferenceValidationCorpusError(result)
        return result


DEFAULT_REFERENCE_VALIDATION_PIPELINE = ReferenceValidationPipeline()


def validate_reference_origin(item: OriginValidationInput) -> OriginValidationReport:
    return DEFAULT_REFERENCE_VALIDATION_PIPELINE.validate(item)


def validate_reference_origin_strict(item: OriginValidationInput) -> OriginValidationReport:
    return DEFAULT_REFERENCE_VALIDATION_PIPELINE.validate_strict(item)


def audit_reference_corpus(
    items: Iterable[OriginValidationInput],
    *,
    require_all_pass: bool = True,
) -> ReferenceValidationCorpusReport:
    return DEFAULT_REFERENCE_VALIDATION_PIPELINE.audit(
        items,
        require_all_pass=require_all_pass,
    )


__all__ = [
    "DEFAULT_REFERENCE_VALIDATION_PIPELINE",
    "OriginValidationInput",
    "OriginValidationReport",
    "ReferenceValidationCorpusError",
    "ReferenceValidationCorpusReport",
    "ReferenceValidationError",
    "ReferenceValidationPipeline",
    "audit_reference_corpus",
    "validate_reference_origin",
    "validate_reference_origin_strict",
]
