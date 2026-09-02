"""Structured validation issues and reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import Severity, ValidationStage
from .state_dag import freeze_string_mapping


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: Severity
    stage: ValidationStage
    node_ids: tuple[str, ...]
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.code) is not str or type(self.message) is not str:
            raise TypeError("ValidationIssue code and message must be strings")
        if type(self.severity) is not Severity:
            raise TypeError("ValidationIssue severity must be a Severity")
        if type(self.stage) is not ValidationStage:
            raise TypeError("ValidationIssue stage must be a ValidationStage")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("ValidationIssue evidence must be a mapping")
        if not isinstance(self.node_ids, (list, tuple)):
            raise TypeError("ValidationIssue node_ids must be a list or tuple")
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        if any(type(node_id) is not str for node_id in self.node_ids):
            raise TypeError("ValidationIssue node_ids must contain strings")
        object.__setattr__(
            self,
            "evidence",
            freeze_string_mapping(self.evidence, name="ValidationIssue evidence"),
        )
        if not self.code or not self.message:
            raise ValueError("ValidationIssue code and message cannot be empty")
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("ValidationIssue node_ids must be unique")
        if any(not node_id for node_id in self.node_ids):
            raise ValueError("ValidationIssue node_ids cannot contain empty strings")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    validator_id: str
    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        if type(self.validator_id) is not str:
            raise TypeError("ValidationReport validator_id must be a string")
        if any(type(issue) is not ValidationIssue for issue in self.issues):
            raise TypeError("ValidationReport issues must contain ValidationIssue values")
        if not self.validator_id:
            raise ValueError("ValidationReport validator_id cannot be empty")

    @property
    def all_pass(self) -> bool:
        return not any(issue.severity in {Severity.ERROR, Severity.FATAL} for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity is Severity.WARNING for issue in self.issues)

    def by_severity(self, severity: Severity) -> tuple[ValidationIssue, ...]:
        if type(severity) is not Severity:
            raise TypeError("by_severity severity must be a Severity")
        return tuple(issue for issue in self.issues if issue.severity is severity)

    @classmethod
    def combine(
        cls,
        validator_id: str,
        reports: Iterable[ValidationReport],
    ) -> ValidationReport:
        reports = tuple(reports)
        if any(type(report) is not ValidationReport for report in reports):
            raise TypeError("ValidationReport.combine reports must contain ValidationReport values")
        return cls(
            validator_id=validator_id,
            issues=tuple(issue for report in reports for issue in report.issues),
        )


class DomainValidationError(ValueError):
    """Raised when an immutable domain object violates a construction invariant."""


class ArtifactValidationError(RuntimeError):
    """Raised when a completed artifact fails one or more deterministic gates."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("ArtifactValidationError report must be a ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"Artifact validation failed ({codes})")
