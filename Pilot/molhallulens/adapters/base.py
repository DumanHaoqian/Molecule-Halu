"""Build-only input adapter contracts and structured failure handling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molhallulens.domain import (
    DomainValidationError,
    Severity,
    ValidationIssue,
    ValidationReport,
    ValidationStage,
)
from molhallulens.domain.state_dag import freeze_string_mapping


_IDENTITY_FIELDS = ("task_family", "subtask", "reporting_task")
_SHARED_AUTHORITATIVE_FIELDS = (*_IDENTITY_FIELDS, "orig_id", "gt_smiles")


def _stable_value_key(value: Any) -> tuple[Any, ...]:
    """Return a fully ordered representation for JSON-like validation evidence."""

    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (str(key), _stable_value_key(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_stable_value_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(_stable_value_key(item) for item in value)))
    if value is None:
        return ("none", "")
    if type(value) is bool:
        return ("bool", "1" if value else "0")
    if type(value) is int:
        return ("int", str(value))
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    return (
        f"{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


@dataclass(frozen=True, slots=True)
class JoinedInputRecord:
    """Namespaced raw/process/template inputs joined only by the anonymous ID."""

    anonymous_sample_id: str
    raw_record: Mapping[str, Any]
    process_record: Mapping[str, Any]
    formal_template: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.anonymous_sample_id) is not str:
            raise DomainValidationError("anonymous_sample_id must be a string")
        if not self.anonymous_sample_id.strip():
            raise DomainValidationError("anonymous_sample_id cannot be empty")
        for value, name in (
            (self.raw_record, "raw_record"),
            (self.process_record, "process_record"),
            (self.formal_template, "formal_template"),
        ):
            if not isinstance(value, Mapping):
                raise DomainValidationError(f"JoinedInputRecord {name} must be a mapping")
            object.__setattr__(
                self,
                name,
                freeze_string_mapping(value, name=f"JoinedInputRecord {name}"),
            )

        for source_name, record in (
            ("raw_record", self.raw_record),
            ("process_record", self.process_record),
        ):
            if record.get("anonymous_sample_id") != self.anonymous_sample_id:
                raise DomainValidationError(
                    f"JoinedInputRecord {source_name} anonymous_sample_id mismatch"
                )
        for field_name in _IDENTITY_FIELDS:
            raw_value = self.raw_record.get(field_name)
            process_value = self.process_record.get(field_name)
            template_value = self.formal_template.get(field_name)
            if any(type(value) is not str or not value.strip() for value in (
                raw_value,
                process_value,
                template_value,
            )):
                raise DomainValidationError(
                    f"JoinedInputRecord {field_name} values must be non-empty strings"
                )
            if not (raw_value == process_value == template_value):
                raise DomainValidationError(
                    f"JoinedInputRecord {field_name} values must agree exactly"
                )
        for field_name in _SHARED_AUTHORITATIVE_FIELDS:
            raw_value = self.raw_record.get(field_name)
            process_value = self.process_record.get(field_name)
            if any(
                type(value) is not str or not value.strip()
                for value in (raw_value, process_value)
            ):
                raise DomainValidationError(
                    f"JoinedInputRecord {field_name} values must be non-empty strings"
                )
            if raw_value != process_value:
                raise DomainValidationError(
                    f"JoinedInputRecord {field_name} values must agree exactly"
                )

    @property
    def task_family(self) -> str:
        return self.raw_record["task_family"]

    @property
    def pilot_subtask(self) -> str:
        return self.raw_record["subtask"]

    @property
    def reporting_task(self) -> str:
        return self.raw_record["reporting_task"]


class InputAdapterError(RuntimeError):
    """Raised when input sources cannot be joined without ambiguity or loss."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("InputAdapterError report must be a ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"Input adapter validation failed ({codes})")


class InputAdapter(ABC):
    """Interface for loading build-only joined input records."""

    @abstractmethod
    def load(self, dataset_root: Path) -> tuple[JoinedInputRecord, ...]:
        """Load, validate, and deterministically join one dataset root."""


def input_issue(
    code: str,
    message: str,
    *,
    anonymous_sample_id: str | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    """Create a fatal input-stage issue without embedding full source records."""

    node_ids = () if anonymous_sample_id is None else (anonymous_sample_id,)
    return ValidationIssue(
        code=code,
        severity=Severity.FATAL,
        stage=ValidationStage.INPUT_RECORD,
        node_ids=node_ids,
        message=message,
        evidence={} if evidence is None else evidence,
    )


def sorted_input_issues(issues: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    """Return input issues in an order independent of source array positions."""

    issues = tuple(issues)
    if any(type(issue) is not ValidationIssue for issue in issues):
        raise TypeError("sorted_input_issues expects ValidationIssue values")
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.node_ids,
                issue.code,
                issue.message,
                _stable_value_key(issue.evidence),
            ),
        )
    )


def raise_for_input_issues(
    validator_id: str,
    issues: Iterable[ValidationIssue],
) -> None:
    """Fail atomically when any collected input issue is present."""

    ordered = sorted_input_issues(issues)
    if ordered:
        raise InputAdapterError(ValidationReport(validator_id, ordered))
