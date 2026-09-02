"""Fail-closed reference-DAG builders for molecule-editing origins."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, ClassVar

from molhallulens.modules.ingestion import (
    DEFAULT_SUBTASK_NORMALIZER,
    JoinedInputRecord,
    SubtaskNormalizationError,
)
from molhallulens.core import (
    ClaimValue,
    DetectorInput,
    EditingSubtask,
    Severity,
    StateDAG,
    ValidationIssue,
    ValidationReport,
    ValidationStage,
    ValueProvenance,
    ValueType,
    Visibility,
    editing_schema_for,
)
from molhallulens.core.state_dag import FrozenMap


_TRACE_FIELDS = frozenset(
    {"step_index", "step_name", "natural_language", "formal_ab", "step_text"}
)
_INTEGER_VALUE_TYPES = frozenset(
    {ValueType.INTEGER, ValueType.ATOM_INDEX, ValueType.COUNT}
)
_VISIBLE_CHANNELS = frozenset({"prompt", "formal", "final_answer"})


def _nonempty_string(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _format_delta(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _safe_evidence(**values: Any) -> dict[str, Any]:
    """Return structural evidence without copying molecule or oracle plaintext."""

    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ReferenceSlotBinding:
    """A logical FORMAL slot occurrence bound to one typed DAG node."""

    source_field: str
    node_id: str
    mention_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_field, "source_field"),
            (self.node_id, "node_id"),
            (self.mention_id, "mention_id"),
        ):
            if type(value) is not str:
                raise TypeError(f"ReferenceSlotBinding {name} must be a string")
            if not value:
                raise ValueError(f"ReferenceSlotBinding {name} cannot be empty")


@dataclass(frozen=True, slots=True)
class ReferenceMention:
    """A stable logical claim occurrence; character spans are assigned later."""

    mention_id: str
    node_id: str
    channel: str
    source_field: str
    step_index: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.mention_id, "mention_id"),
            (self.node_id, "node_id"),
            (self.channel, "channel"),
            (self.source_field, "source_field"),
        ):
            if type(value) is not str:
                raise TypeError(f"ReferenceMention {name} must be a string")
            if not value:
                raise ValueError(f"ReferenceMention {name} cannot be empty")
        if self.channel not in _VISIBLE_CHANNELS:
            raise ValueError(f"unknown reference mention channel: {self.channel!r}")
        if self.step_index is not None and (
            type(self.step_index) is not int or self.step_index <= 0
        ):
            raise ValueError("ReferenceMention step_index must be positive or None")
        if self.channel == "formal" and self.step_index is None:
            raise ValueError("FORMAL mentions require a step_index")
        if self.channel != "formal" and self.step_index is not None:
            raise ValueError("only FORMAL mentions may have a step_index")


@dataclass(frozen=True, slots=True)
class ReferenceTraceStep:
    """An exact natural/FORMAL step payload with typed FORMAL slot bindings."""

    step_index: int
    step_name: str
    natural_language: str
    formal_ab: str
    step_text: str
    slot_bindings: tuple[ReferenceSlotBinding, ...]
    answer_suffix: str | None = None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("ReferenceTraceStep step_index must be positive")
        for value, name in (
            (self.step_name, "step_name"),
            (self.natural_language, "natural_language"),
            (self.formal_ab, "formal_ab"),
            (self.step_text, "step_text"),
        ):
            if not _nonempty_string(value):
                raise ValueError(f"ReferenceTraceStep {name} must be non-empty text")
        object.__setattr__(self, "slot_bindings", tuple(self.slot_bindings))
        if any(type(binding) is not ReferenceSlotBinding for binding in self.slot_bindings):
            raise TypeError("slot_bindings must contain ReferenceSlotBinding values")
        if self.answer_suffix is not None and not _nonempty_string(self.answer_suffix):
            raise ValueError("answer_suffix must be non-empty text or None")
        if self.render(include_answer=True) != self.step_text:
            raise ValueError("step_text does not round-trip from natural/FORMAL components")

    def render(self, *, include_answer: bool) -> str:
        text = (
            f"Step {self.step_index} [{self.step_name}]: {self.natural_language}"
            f"\n  FORMAL: {self.formal_ab}"
        )
        if include_answer and self.answer_suffix is not None:
            text += f"\n\nAnswer: {self.answer_suffix}"
        return text


@dataclass(frozen=True, slots=True)
class ReferenceDAGArtifact:
    """One immutable reference DAG plus its detector-visible trace projection."""

    anonymous_sample_id: str
    legacy_orig_id: str
    legacy_sample_id: int
    normalized_subtask: EditingSubtask
    state_dag: StateDAG
    trace_steps: tuple[ReferenceTraceStep, ...]
    mentions: tuple[ReferenceMention, ...]

    def __post_init__(self) -> None:
        if not _nonempty_string(self.anonymous_sample_id):
            raise ValueError("anonymous_sample_id must be non-empty text")
        if not _nonempty_string(self.legacy_orig_id):
            raise ValueError("legacy_orig_id must be non-empty text")
        if type(self.legacy_sample_id) is not int:
            raise TypeError("legacy_sample_id must be an integer")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be an EditingSubtask")
        if type(self.state_dag) is not StateDAG:
            raise TypeError("state_dag must be a StateDAG")
        if self.state_dag.schema is not editing_schema_for(
            self.normalized_subtask
        ).schema:
            raise ValueError("state_dag schema does not match normalized_subtask")
        if self.state_dag.edge_values:
            raise ValueError("reference-DAG relations must remain unknown at T011")
        object.__setattr__(self, "trace_steps", tuple(self.trace_steps))
        object.__setattr__(self, "mentions", tuple(self.mentions))
        if not self.trace_steps or any(
            type(step) is not ReferenceTraceStep for step in self.trace_steps
        ):
            raise TypeError("trace_steps must contain ReferenceTraceStep values")
        if any(type(mention) is not ReferenceMention for mention in self.mentions):
            raise TypeError("mentions must contain ReferenceMention values")
        mention_ids = tuple(mention.mention_id for mention in self.mentions)
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("artifact mention IDs must be unique")
        nodes = self.state_dag.schema.nodes_by_id
        if any(mention.node_id not in nodes for mention in self.mentions):
            raise ValueError("artifact mention targets must be schema nodes")
        if any(
            nodes[mention.node_id].visibility is Visibility.BUILD_ONLY
            for mention in self.mentions
        ):
            raise ValueError("BUILD_ONLY nodes cannot have detector-visible mentions")
        claim_mention_ids = tuple(
            mention_id
            for claim in self.state_dag.values.values()
            for mention_id in claim.mention_ids
        )
        if sorted(claim_mention_ids) != sorted(mention_ids):
            raise ValueError("claim mention IDs must exactly match artifact mentions")
        if any(
            self.state_dag.values[node.node_id].mention_ids
            for node in self.state_dag.schema.nodes
            if node.visibility is Visibility.BUILD_ONLY
        ):
            raise ValueError("BUILD_ONLY claims cannot carry mention IDs")

    @property
    def reasoning_chain(self) -> str:
        """Return exact reference reasoning without duplicating the final answer."""

        return "\n\n".join(
            step.render(include_answer=False) for step in self.trace_steps
        )

    @property
    def detector_input(self) -> DetectorInput:
        """Project only the four fields admitted by the frozen detector contract."""

        return DetectorInput(
            indexed_smiles=self.state_dag.values["source"].normalized_value,
            instruction=self.state_dag.values["instruction"].normalized_value,
            reasoning_chain=self.reasoning_chain,
            final_answer=self.state_dag.values["final_answer"].normalized_value,
        )

    @property
    def detector_visible_values(self) -> FrozenMap[str, ClaimValue]:
        """Expose a visibility-based projection that excludes every oracle node."""

        nodes = self.state_dag.schema.nodes_by_id
        return FrozenMap(
            {
                node_id: claim
                for node_id, claim in self.state_dag.values.items()
                if nodes[node_id].visibility is not Visibility.BUILD_ONLY
            }
        )


@dataclass(frozen=True, slots=True)
class ReferenceDAGOriginReport:
    anonymous_sample_id: str
    subtask: str
    schema_id: str
    schema_version: str
    status: str
    node_count: int
    edge_count: int
    trace_step_count: int
    natural_round_trip_steps: int
    formal_round_trip_slots: int
    mention_count: int
    raw_answer_gt_equal: bool
    issue_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.anonymous_sample_id, "anonymous_sample_id"),
            (self.subtask, "subtask"),
            (self.schema_id, "schema_id"),
            (self.schema_version, "schema_version"),
            (self.status, "status"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"ReferenceDAGOriginReport {name} must be non-empty")
        if self.status not in {"built", "failed"}:
            raise ValueError("origin report status must be built or failed")
        for value in (
            self.node_count,
            self.edge_count,
            self.trace_step_count,
            self.natural_round_trip_steps,
            self.formal_round_trip_slots,
            self.mention_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("origin report counts must be non-negative integers")
        if type(self.raw_answer_gt_equal) is not bool:
            raise TypeError("raw_answer_gt_equal must be bool")
        object.__setattr__(self, "issue_codes", tuple(self.issue_codes))
        if any(type(code) is not str or not code for code in self.issue_codes):
            raise ValueError("issue_codes must contain non-empty strings")
        if self.status == "built" and self.issue_codes:
            raise ValueError("built origins cannot have issue codes")
        if self.status == "failed" and not self.issue_codes:
            raise ValueError("failed origins require issue codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anonymous_sample_id": self.anonymous_sample_id,
            "subtask": self.subtask,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "status": self.status,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "trace_step_count": self.trace_step_count,
            "natural_round_trip_steps": self.natural_round_trip_steps,
            "formal_round_trip_slots": self.formal_round_trip_slots,
            "mention_count": self.mention_count,
            "raw_answer_gt_equal": self.raw_answer_gt_equal,
            "issue_codes": list(self.issue_codes),
        }


@dataclass(frozen=True, slots=True)
class ReferenceDAGBuildReport:
    origins: tuple[ReferenceDAGOriginReport, ...]
    issues: tuple[ValidationIssue, ...] = ()
    format_version: str = "reference_dag_build_report_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", tuple(self.origins))
        object.__setattr__(self, "issues", tuple(self.issues))
        if any(type(item) is not ReferenceDAGOriginReport for item in self.origins):
            raise TypeError("origins must contain ReferenceDAGOriginReport values")
        if any(type(item) is not ValidationIssue for item in self.issues):
            raise TypeError("issues must contain ValidationIssue values")
        ids = tuple(item.anonymous_sample_id for item in self.origins)
        if ids != tuple(sorted(ids)):
            raise ValueError("origin reports must be sorted by anonymous_sample_id")
        if len(ids) != len(set(ids)):
            raise ValueError("origin reports must use unique anonymous_sample_id values")
        if self.format_version != "reference_dag_build_report_v1":
            raise ValueError("unknown reference-DAG report version")

    @property
    def attempted(self) -> int:
        return len(self.origins)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "built" for item in self.origins)

    @property
    def failed(self) -> int:
        return self.attempted - self.succeeded

    @property
    def all_pass(self) -> bool:
        return self.failed == 0 and not self.issues

    @property
    def counts_by_subtask(self) -> FrozenMap[str, int]:
        return FrozenMap(dict(sorted(Counter(item.subtask for item in self.origins).items())))

    def to_dict(self) -> dict[str, Any]:
        built = tuple(item for item in self.origins if item.status == "built")
        return {
            "format_version": self.format_version,
            "summary": {
                "attempted": self.attempted,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "counts_by_subtask": dict(self.counts_by_subtask),
                "trace_steps": sum(item.trace_step_count for item in built),
                "natural_round_trip_steps": sum(
                    item.natural_round_trip_steps for item in built
                ),
                "formal_round_trip_slots": sum(
                    item.formal_round_trip_slots for item in built
                ),
                "node_values": sum(item.node_count for item in built),
                "schema_edges": sum(item.edge_count for item in built),
                "logical_mentions": sum(item.mention_count for item in built),
                "raw_answer_gt_string_mismatches": sum(
                    not item.raw_answer_gt_equal for item in built
                ),
                "build_only_detector_mentions": 0,
                "issue_count": len(self.issues),
            },
            "origins": [item.to_dict() for item in self.origins],
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "stage": issue.stage.value,
                    "anonymous_sample_ids": list(issue.node_ids),
                    "message": issue.message,
                    "evidence": dict(issue.evidence),
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True, slots=True)
class ReferenceDAGCorpusResult:
    artifacts: tuple[ReferenceDAGArtifact, ...]
    report: ReferenceDAGBuildReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if any(type(item) is not ReferenceDAGArtifact for item in self.artifacts):
            raise TypeError("artifacts must contain ReferenceDAGArtifact values")
        ids = tuple(item.anonymous_sample_id for item in self.artifacts)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("artifacts must be uniquely sorted by anonymous_sample_id")
        if type(self.report) is not ReferenceDAGBuildReport:
            raise TypeError("report must be a ReferenceDAGBuildReport")
        built_ids = tuple(
            item.anonymous_sample_id
            for item in self.report.origins
            if item.status == "built"
        )
        if ids != built_ids:
            raise ValueError("artifact IDs must exactly match successful report entries")


class ReferenceDAGBuildError(RuntimeError):
    """Raised when one origin cannot be built without guessing or fallback."""

    def __init__(self, report: ValidationReport) -> None:
        if type(report) is not ValidationReport:
            raise TypeError("ReferenceDAGBuildError report must be a ValidationReport")
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues) or "unknown"
        super().__init__(f"reference DAG build failed ({codes})")


class ReferenceDAGCorpusError(RuntimeError):
    """Raised by the strict corpus API when any origin has a reported failure."""

    def __init__(self, result: ReferenceDAGCorpusResult) -> None:
        if type(result) is not ReferenceDAGCorpusResult:
            raise TypeError("ReferenceDAGCorpusError result must be a corpus result")
        self.result = result
        super().__init__(
            f"reference DAG corpus has {result.report.failed} failed origin(s)"
        )


@dataclass(frozen=True, slots=True)
class _FormalStepContract:
    step_index: int
    step_name: str
    template: str
    fields: tuple[str, ...]
    signed_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("formal step_index must be positive")
        if not _nonempty_string(self.step_name) or not _nonempty_string(self.template):
            raise ValueError("formal step name/template must be non-empty")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "signed_fields", frozenset(self.signed_fields))
        placeholders = tuple(
            field_name
            for _, field_name, format_spec, conversion in Formatter().parse(self.template)
            if field_name is not None
            and not format_spec
            and conversion is None
        )
        if placeholders != self.fields:
            raise ValueError("formal template placeholders must exactly match fields")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("formal step fields must be unique within one step")
        if not self.signed_fields <= set(self.fields):
            raise ValueError("signed fields must be FORMAL step fields")

    def render(self, state: Mapping[str, Any]) -> str:
        values = {
            name: (
                _format_delta(state[name])
                if name in self.signed_fields
                else str(state[name])
            )
            for name in self.fields
        }
        return self.template.format_map(values)

    def parse(
        self,
        formal_ab: str,
        *,
        state: Mapping[str, Any],
        value_types: Mapping[str, ValueType],
    ) -> Mapping[str, Any] | None:
        pattern_parts: list[str] = []
        for literal, field_name, _, _ in Formatter().parse(self.template):
            pattern_parts.append(re.escape(literal))
            if field_name is None:
                continue
            value_type = value_types[field_name]
            if value_type in _INTEGER_VALUE_TYPES:
                payload = r"[+-]?\d+" if field_name in self.signed_fields else r"-?\d+"
            else:
                payload = r'[^"\r\n]*'
            pattern_parts.append(f"(?P<{field_name}>{payload})")
        match = re.fullmatch("".join(pattern_parts), formal_ab)
        if match is None:
            return None
        parsed: dict[str, Any] = {}
        for field_name in self.fields:
            raw = match.group(field_name)
            value_type = value_types[field_name]
            parsed[field_name] = int(raw) if value_type in _INTEGER_VALUE_TYPES else raw
        if any(
            type(parsed[name]) is not type(state[name]) or parsed[name] != state[name]
            for name in self.fields
        ):
            return None
        if self.render(parsed) != formal_ab or self.render(state) != formal_ab:
            return None
        return MappingProxyType(parsed)


def _contract(
    step_index: int,
    step_name: str,
    template: str,
    *fields: str,
    signed: Sequence[str] = (),
) -> _FormalStepContract:
    return _FormalStepContract(
        step_index=step_index,
        step_name=step_name,
        template=template,
        fields=tuple(fields),
        signed_fields=frozenset(signed),
    )


_ADDITION_CONTRACTS = (
    _contract(
        1,
        "ANCHOR_IDENTIFICATION",
        'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + LEAVING(smiles="{step1_leaving_smiles}")',
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_leaving_smiles",
    ),
    _contract(
        2,
        "FRAGMENT_IDENTIFICATION",
        'INSTRUCTION --> ADD_FRAGMENT(smiles="{step2_frag_smiles}", heavy_atoms={step2_heavy_atoms})',
        "step2_frag_smiles",
        "step2_heavy_atoms",
    ),
    _contract(
        3,
        "PRODUCT_CONSTRUCTION",
        'SMILES + ANCHOR(idx={step1_anchor_idx}) + LEAVING("{step1_leaving_smiles}") + ADD_FRAGMENT(smiles="{step2_frag_smiles}") --> PRODUCT_SMILES("{step3_product_smiles}")',
        "step1_anchor_idx",
        "step1_leaving_smiles",
        "step2_frag_smiles",
        "step3_product_smiles",
    ),
    _contract(
        4,
        "HEAVY_ATOM_VERIFICATION",
        "SMILES[n_heavy={step4_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step4_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step4_heavy_delta})",
        "step4_n_heavy_src",
        "step4_n_heavy_prod",
        "step4_heavy_delta",
        signed=("step4_heavy_delta",),
    ),
    _contract(
        5,
        "RING_VERIFICATION",
        "SMILES[n_rings={step5_n_rings_src}] + PRODUCT_SMILES[n_rings={step5_n_rings_prod}] --> RING_DELTA({step5_ring_delta})",
        "step5_n_rings_src",
        "step5_n_rings_prod",
        "step5_ring_delta",
        signed=("step5_ring_delta",),
    ),
)

_DELETION_CONTRACTS = (
    _contract(
        1,
        "ANCHOR_IDENTIFICATION",
        'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + REMOVE_GROUP(smiles="{step1_remove_group}")',
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_remove_group",
    ),
    _contract(
        2,
        "GROUP_SIZE_VERIFICATION",
        'REMOVE_GROUP(smiles="{step2_remove_smiles}") --> HEAVY_ATOMS({step2_heavy_atoms})',
        "step2_remove_smiles",
        "step2_heavy_atoms",
    ),
    _contract(
        3,
        "PRODUCT_CONSTRUCTION",
        'SMILES + ANCHOR(idx={step1_anchor_idx}) + REMOVE_GROUP(smiles="{step2_remove_smiles}") --> PRODUCT_SMILES("{step3_product_smiles}")',
        "step1_anchor_idx",
        "step2_remove_smiles",
        "step3_product_smiles",
    ),
    _contract(
        4,
        "HEAVY_ATOM_VERIFICATION",
        "SMILES[n_heavy={step4_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step4_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step4_heavy_delta})",
        "step4_n_heavy_src",
        "step4_n_heavy_prod",
        "step4_heavy_delta",
        signed=("step4_heavy_delta",),
    ),
    _contract(
        5,
        "RING_VERIFICATION",
        "SMILES[n_rings={step5_n_rings_src}] + PRODUCT_SMILES[n_rings={step5_n_rings_prod}] --> RING_DELTA({step5_ring_delta})",
        "step5_n_rings_src",
        "step5_n_rings_prod",
        "step5_ring_delta",
        signed=("step5_ring_delta",),
    ),
)

_SUBSTITUTION_CONTRACTS = (
    _contract(
        1,
        "ANCHOR_IDENTIFICATION",
        'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + REMOVE_GROUP(smiles="{step1_remove_group_smiles}") + ADD_FRAGMENT(smiles="{step1_add_fragment_smiles}")',
        "step1_anchor_idx",
        "step1_anchor_element",
        "step1_remove_group_smiles",
        "step1_add_fragment_smiles",
    ),
    _contract(
        2,
        "REMOVE_GROUP_SIZE",
        'REMOVE_GROUP(smiles="{step1_remove_group_smiles}") --> REMOVE_HEAVY({step2_remove_heavy})',
        "step1_remove_group_smiles",
        "step2_remove_heavy",
    ),
    _contract(
        3,
        "ADD_FRAGMENT_SIZE",
        'ADD_FRAGMENT(smiles="{step1_add_fragment_smiles}") --> ADD_HEAVY({step3_add_heavy})',
        "step1_add_fragment_smiles",
        "step3_add_heavy",
    ),
    _contract(
        4,
        "PRODUCT_CONSTRUCTION",
        'SMILES + ANCHOR(idx={step1_anchor_idx}) + REMOVE_GROUP("{step1_remove_group_smiles}") + ADD_FRAGMENT("{step1_add_fragment_smiles}") --> PRODUCT_SMILES("{step4_product_smiles}")',
        "step1_anchor_idx",
        "step1_remove_group_smiles",
        "step1_add_fragment_smiles",
        "step4_product_smiles",
    ),
    _contract(
        5,
        "HEAVY_ATOM_VERIFICATION",
        "SMILES[n_heavy={step5_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step5_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step5_heavy_delta})",
        "step5_n_heavy_src",
        "step5_n_heavy_prod",
        "step5_heavy_delta",
        signed=("step5_heavy_delta",),
    ),
    _contract(
        6,
        "RING_VERIFICATION",
        "SMILES[n_rings={step6_n_rings_src}] + PRODUCT_SMILES[n_rings={step6_n_rings_prod}] --> RING_DELTA({step6_ring_delta})",
        "step6_n_rings_src",
        "step6_n_rings_prod",
        "step6_ring_delta",
        signed=("step6_ring_delta",),
    ),
)


class EditingReferenceDAGBuilder:
    """Binding-driven base builder; concrete classes freeze one trace grammar."""

    normalized_subtask: ClassVar[EditingSubtask]
    trace_contracts: ClassVar[tuple[_FormalStepContract, ...]]

    def _issue(
        self,
        record: JoinedInputRecord,
        code: str,
        message: str,
        **evidence: Any,
    ) -> ReferenceDAGBuildError:
        return ReferenceDAGBuildError(
            ValidationReport(
                "molhallulens.reference_dag.v1",
                (
                    ValidationIssue(
                        code=code,
                        severity=Severity.FATAL,
                        stage=ValidationStage.REFERENCE_DAG,
                        node_ids=(record.anonymous_sample_id,),
                        message=message,
                        evidence=_safe_evidence(**evidence),
                    ),
                ),
            )
        )

    def build(self, record: JoinedInputRecord) -> ReferenceDAGArtifact:
        if type(record) is not JoinedInputRecord:
            raise TypeError("reference-DAG builder requires a JoinedInputRecord")
        try:
            mapping = DEFAULT_SUBTASK_NORMALIZER.normalize(record.pilot_subtask)
        except SubtaskNormalizationError as error:
            raise self._issue(
                record,
                "SUBTASK_CONTRACT",
                "record subtask is not registered",
                actual_type=type(record.pilot_subtask).__name__,
            ) from error
        if mapping.normalized_subtask is not self.normalized_subtask:
            raise self._issue(
                record,
                "SUBTASK_CONTRACT",
                "record was dispatched to the wrong reference-DAG builder",
                expected_subtask=self.normalized_subtask.value,
                actual_subtask=mapping.normalized_subtask.value,
            )

        definition = editing_schema_for(self.normalized_subtask)
        state = record.process_record.get("parsed_reference_state")
        if not isinstance(state, Mapping):
            raise self._issue(
                record,
                "STATE_SHAPE",
                "parsed_reference_state must be a mapping",
                actual_type=type(state).__name__,
            )
        expected_fields = set(definition.legacy_step_field_bindings) | set(
            definition.rdkit_reference_bindings
        )
        actual_fields = set(state)
        if actual_fields != expected_fields or any(type(key) is not str for key in state):
            raise self._issue(
                record,
                "STATE_FIELD_SET",
                "parsed_reference_state fields differ from the typed schema bindings",
                missing_fields=tuple(sorted(expected_fields - actual_fields)),
                extra_fields=tuple(sorted(str(item) for item in actual_fields - expected_fields)),
            )
        if set(record.formal_template.get("step_fields", ())) != set(
            definition.legacy_step_field_bindings
        ) or set(record.formal_template.get("rdkit_reference_fields", ())) != set(
            definition.rdkit_reference_bindings
        ):
            raise self._issue(
                record,
                "SCHEMA_BINDING",
                "formal-template inventory differs from typed schema bindings",
            )

        if record.process_record.get("outcome") is not True or record.process_record.get(
            "verifier_checks", {}
        ).get("all_pass") is not True:
            raise self._issue(
                record,
                "UNCLEAN_REFERENCE_TRACE",
                "process trace is not marked as a verified clean reference",
            )

        trace_payload = record.process_record.get("formal_cot_trace")
        if isinstance(trace_payload, (str, bytes)) or not isinstance(
            trace_payload, Sequence
        ):
            raise self._issue(
                record,
                "TRACE_SHAPE",
                "formal_cot_trace must be a sequence",
                actual_type=type(trace_payload).__name__,
            )
        if len(trace_payload) != len(self.trace_contracts):
            raise self._issue(
                record,
                "TRACE_SHAPE",
                "formal_cot_trace has the wrong number of steps",
                expected_count=len(self.trace_contracts),
                actual_count=len(trace_payload),
            )

        answer_smiles = record.process_record.get("answer_smiles")
        if not _nonempty_string(answer_smiles):
            raise self._issue(
                record,
                "FINAL_ANSWER_SHAPE",
                "answer_smiles must be non-empty text",
                actual_type=type(answer_smiles).__name__,
            )

        value_types = {
            field_name: definition.schema.nodes_by_id[node_id].value_type
            for field_name, node_id in definition.legacy_step_field_bindings.items()
        }
        node_occurrences: Counter[str] = Counter()
        mentions: list[ReferenceMention] = []
        step_objects: list[ReferenceTraceStep] = []
        mention_ids_by_node: dict[str, list[str]] = {
            node.node_id: [] for node in definition.schema.nodes
        }

        for payload, contract in zip(trace_payload, self.trace_contracts, strict=True):
            if not isinstance(payload, Mapping) or set(payload) != _TRACE_FIELDS:
                raise self._issue(
                    record,
                    "TRACE_SHAPE",
                    "trace step fields differ from the frozen contract",
                    step_index=contract.step_index,
                    actual_type=type(payload).__name__,
                )
            if (
                type(payload["step_index"]) is not int
                or payload["step_index"] != contract.step_index
                or payload["step_name"] != contract.step_name
            ):
                raise self._issue(
                    record,
                    "STEP_SEQUENCE",
                    "trace step index/name differs from the frozen contract",
                    expected_step_index=contract.step_index,
                    expected_step_name=contract.step_name,
                )
            for key in ("natural_language", "formal_ab", "step_text"):
                if not _nonempty_string(payload[key]):
                    raise self._issue(
                        record,
                        "TRACE_SHAPE",
                        "trace text fields must be non-empty strings",
                        step_index=contract.step_index,
                        field=key,
                        actual_type=type(payload[key]).__name__,
                    )
            parsed = contract.parse(
                payload["formal_ab"], state=state, value_types=value_types
            )
            if parsed is None:
                raise self._issue(
                    record,
                    "FORMAL_STATE_ROUNDTRIP",
                    "FORMAL slots do not round-trip against parsed_reference_state",
                    step_index=contract.step_index,
                    step_name=contract.step_name,
                )

            slot_bindings: list[ReferenceSlotBinding] = []
            for source_field in contract.fields:
                node_id = definition.legacy_step_field_bindings[source_field]
                node_occurrences[node_id] += 1
                mention_id = (
                    f"{record.anonymous_sample_id}.formal.step{contract.step_index:02d}."
                    f"{node_id}.{node_occurrences[node_id]:02d}"
                )
                slot_bindings.append(
                    ReferenceSlotBinding(source_field, node_id, mention_id)
                )
                mentions.append(
                    ReferenceMention(
                        mention_id=mention_id,
                        node_id=node_id,
                        channel="formal",
                        source_field=source_field,
                        step_index=contract.step_index,
                    )
                )
                mention_ids_by_node[node_id].append(mention_id)

            answer_suffix = (
                answer_smiles
                if contract.step_index == self.trace_contracts[-1].step_index
                else None
            )
            expected_step_text = (
                f"Step {contract.step_index} [{contract.step_name}]: "
                f"{payload['natural_language']}\n  FORMAL: {payload['formal_ab']}"
            )
            if answer_suffix is not None:
                expected_step_text += f"\n\nAnswer: {answer_suffix}"
            if payload["step_text"] != expected_step_text:
                raise self._issue(
                    record,
                    "STEP_TEXT_ROUNDTRIP",
                    "step_text does not round-trip from natural/FORMAL components",
                    step_index=contract.step_index,
                    expected_length=len(expected_step_text),
                    actual_length=len(payload["step_text"]),
                )
            step_objects.append(
                ReferenceTraceStep(
                    step_index=contract.step_index,
                    step_name=contract.step_name,
                    natural_language=payload["natural_language"],
                    formal_ab=payload["formal_ab"],
                    step_text=payload["step_text"],
                    slot_bindings=tuple(slot_bindings),
                    answer_suffix=answer_suffix,
                )
            )

        for field_name, node_id, channel in (
            ("indexed_smiles", "source", "prompt"),
            ("instruction", "instruction", "prompt"),
            ("answer_smiles", "final_answer", "final_answer"),
        ):
            mention_id = (
                f"{record.anonymous_sample_id}.{channel}.{node_id}.01"
            )
            mentions.append(
                ReferenceMention(
                    mention_id=mention_id,
                    node_id=node_id,
                    channel=channel,
                    source_field=field_name,
                )
            )
            mention_ids_by_node[node_id].append(mention_id)

        source_values: dict[str, tuple[Any, ValueProvenance]] = {}
        for field_name, node_id in definition.record_field_bindings.items():
            source = (
                record.process_record if field_name == "answer_smiles" else record.raw_record
            )
            if field_name not in source:
                raise self._issue(
                    record,
                    "MISSING_BOUND_VALUE",
                    "record is missing a schema-bound value",
                    field=field_name,
                    node_id=node_id,
                )
            source_values[node_id] = (source[field_name], ValueProvenance.REFERENCE)
        for field_name, node_id in definition.legacy_step_field_bindings.items():
            source_values[node_id] = (state[field_name], ValueProvenance.REFERENCE)
        for field_name, node_id in definition.rdkit_reference_bindings.items():
            source_values[node_id] = (state[field_name], ValueProvenance.RDKIT)

        values: dict[str, ClaimValue] = {}
        for node in definition.schema.nodes:
            raw_value, provenance = source_values[node.node_id]
            try:
                values[node.node_id] = ClaimValue(
                    raw_value=raw_value,
                    normalized_value=raw_value,
                    value_type=node.value_type,
                    provenance=provenance,
                    mention_ids=tuple(mention_ids_by_node[node.node_id]),
                )
            except (TypeError, ValueError) as error:
                raise self._issue(
                    record,
                    "STATE_VALUE_TYPE",
                    "schema-bound state value has an invalid exact type",
                    node_id=node.node_id,
                    actual_type=type(raw_value).__name__,
                    expected_value_type=node.value_type.value,
                ) from error

        try:
            state_dag = StateDAG(definition.schema, values)
            return ReferenceDAGArtifact(
                anonymous_sample_id=record.anonymous_sample_id,
                legacy_orig_id=record.raw_record["orig_id"],
                legacy_sample_id=record.process_record["sample_id"],
                normalized_subtask=self.normalized_subtask,
                state_dag=state_dag,
                trace_steps=tuple(step_objects),
                mentions=tuple(sorted(mentions, key=lambda item: item.mention_id)),
            )
        except ReferenceDAGBuildError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            raise self._issue(
                record,
                "ARTIFACT_INVARIANT",
                "reference-DAG artifact violates its immutable contract",
                error_type=type(error).__name__,
            ) from error


class AdditionReferenceDAGBuilder(EditingReferenceDAGBuilder):
    normalized_subtask = EditingSubtask.ADD
    trace_contracts = _ADDITION_CONTRACTS


class DeletionReferenceDAGBuilder(EditingReferenceDAGBuilder):
    normalized_subtask = EditingSubtask.DELETE
    trace_contracts = _DELETION_CONTRACTS


class SubstitutionReferenceDAGBuilder(EditingReferenceDAGBuilder):
    normalized_subtask = EditingSubtask.SUBSTITUTE
    trace_contracts = _SUBSTITUTION_CONTRACTS


_REFERENCE_DAG_BUILDERS: Mapping[EditingSubtask, EditingReferenceDAGBuilder] = (
    MappingProxyType(
        {
            EditingSubtask.ADD: AdditionReferenceDAGBuilder(),
            EditingSubtask.DELETE: DeletionReferenceDAGBuilder(),
            EditingSubtask.SUBSTITUTE: SubstitutionReferenceDAGBuilder(),
        }
    )
)


def reference_dag_builder_for(
    subtask: EditingSubtask,
) -> EditingReferenceDAGBuilder:
    if type(subtask) is not EditingSubtask:
        raise TypeError("subtask must be an EditingSubtask")
    return _REFERENCE_DAG_BUILDERS[subtask]


def build_reference_dag(record: JoinedInputRecord) -> ReferenceDAGArtifact:
    """Dispatch one joined record through the exact typed subtask registry."""

    if type(record) is not JoinedInputRecord:
        raise TypeError("build_reference_dag requires a JoinedInputRecord")
    try:
        mapping = DEFAULT_SUBTASK_NORMALIZER.normalize(record.pilot_subtask)
    except SubtaskNormalizationError as error:
        issue = ValidationIssue(
            code="SUBTASK_CONTRACT",
            severity=Severity.FATAL,
            stage=ValidationStage.REFERENCE_DAG,
            node_ids=(record.anonymous_sample_id,),
            message="record subtask is not registered",
            evidence={"actual_type": type(record.pilot_subtask).__name__},
        )
        raise ReferenceDAGBuildError(
            ValidationReport("molhallulens.reference_dag.v1", (issue,))
        ) from error
    return reference_dag_builder_for(mapping.normalized_subtask).build(record)


def _success_report(
    artifact: ReferenceDAGArtifact,
    record: JoinedInputRecord,
) -> ReferenceDAGOriginReport:
    return ReferenceDAGOriginReport(
        anonymous_sample_id=artifact.anonymous_sample_id,
        subtask=artifact.normalized_subtask.value,
        schema_id=artifact.state_dag.schema.schema_id,
        schema_version=artifact.state_dag.schema.version,
        status="built",
        node_count=len(artifact.state_dag.values),
        edge_count=len(artifact.state_dag.schema.edges),
        trace_step_count=len(artifact.trace_steps),
        natural_round_trip_steps=len(artifact.trace_steps),
        formal_round_trip_slots=sum(
            len(step.slot_bindings) for step in artifact.trace_steps
        ),
        mention_count=len(artifact.mentions),
        raw_answer_gt_equal=(
            record.process_record["answer_smiles"] == record.raw_record["gt_smiles"]
        ),
    )


def audit_reference_dag_corpus(
    records: Iterable[JoinedInputRecord],
) -> ReferenceDAGCorpusResult:
    """Build all unambiguous origins and retain every controlled failure in a report."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Iterable):
        raise TypeError("records must be a non-string iterable")
    records = tuple(records)
    if any(type(record) is not JoinedInputRecord for record in records):
        raise TypeError("records must contain JoinedInputRecord values")
    ids = tuple(record.anonymous_sample_id for record in records)
    duplicates = tuple(sorted(item for item, count in Counter(ids).items() if count > 1))
    if duplicates:
        issues = tuple(
            ValidationIssue(
                code="DUPLICATE_ANONYMOUS_ID",
                severity=Severity.FATAL,
                stage=ValidationStage.REFERENCE_DAG,
                node_ids=(anonymous_id,),
                message="corpus contains a duplicate anonymous_sample_id",
                evidence={"count": ids.count(anonymous_id)},
            )
            for anonymous_id in duplicates
        )
        raise ReferenceDAGBuildError(
            ValidationReport("molhallulens.reference_dag.corpus.v1", issues)
        )

    artifacts: list[ReferenceDAGArtifact] = []
    origin_reports: list[ReferenceDAGOriginReport] = []
    issues: list[ValidationIssue] = []
    for record in sorted(records, key=lambda item: item.anonymous_sample_id):
        try:
            artifact = build_reference_dag(record)
        except ReferenceDAGBuildError as error:
            issues.extend(error.report.issues)
            try:
                normalized = DEFAULT_SUBTASK_NORMALIZER.normalize(
                    record.pilot_subtask
                ).normalized_subtask
                definition = editing_schema_for(normalized)
                subtask = normalized.value
                schema_id = definition.schema.schema_id
                schema_version = definition.schema.version
            except SubtaskNormalizationError:
                subtask = "unknown"
                schema_id = "unknown"
                schema_version = "unknown"
            origin_reports.append(
                ReferenceDAGOriginReport(
                    anonymous_sample_id=record.anonymous_sample_id,
                    subtask=subtask,
                    schema_id=schema_id,
                    schema_version=schema_version,
                    status="failed",
                    node_count=0,
                    edge_count=0,
                    trace_step_count=0,
                    natural_round_trip_steps=0,
                    formal_round_trip_slots=0,
                    mention_count=0,
                    raw_answer_gt_equal=(
                        record.process_record.get("answer_smiles")
                        == record.raw_record.get("gt_smiles")
                    ),
                    issue_codes=tuple(issue.code for issue in error.report.issues),
                )
            )
            continue
        artifacts.append(artifact)
        origin_reports.append(_success_report(artifact, record))

    report = ReferenceDAGBuildReport(tuple(origin_reports), tuple(issues))
    return ReferenceDAGCorpusResult(tuple(artifacts), report)


def build_reference_dag_corpus(
    records: Iterable[JoinedInputRecord],
) -> ReferenceDAGCorpusResult:
    """Strict corpus build: raise with the complete report if any origin fails."""

    result = audit_reference_dag_corpus(records)
    if not result.report.all_pass:
        raise ReferenceDAGCorpusError(result)
    return result


__all__ = [
    "AdditionReferenceDAGBuilder",
    "DeletionReferenceDAGBuilder",
    "EditingReferenceDAGBuilder",
    "ReferenceDAGArtifact",
    "ReferenceDAGBuildError",
    "ReferenceDAGBuildReport",
    "ReferenceDAGCorpusError",
    "ReferenceDAGCorpusResult",
    "ReferenceDAGOriginReport",
    "ReferenceMention",
    "ReferenceSlotBinding",
    "ReferenceTraceStep",
    "SubstitutionReferenceDAGBuilder",
    "audit_reference_dag_corpus",
    "build_reference_dag",
    "build_reference_dag_corpus",
    "reference_dag_builder_for",
]


if __name__ == "__main__":
    # Import locally so importing the production builder does not pull the
    # filesystem adapter into normal library use.
    from molhallulens.modules.ingestion import ChemCoTMolEditAdapter

    from molhallulens.config.paths import DEFAULT_DATASET_ROOT

    dataset_root = DEFAULT_DATASET_ROOT
    records = ChemCoTMolEditAdapter().load(dataset_root)
    # import ipdb;ipdb.set_trace()
    requested = (
        ("add", EditingSubtask.ADD),
        ("sub", EditingSubtask.SUBSTITUTE),
        ("del", EditingSubtask.DELETE),
    )

    print(f"Dataset: {dataset_root}")
    print("Examples: [add, sub, del]")

    for label, expected_subtask in requested:
        record = next(
            item
            for item in records
            if DEFAULT_SUBTASK_NORMALIZER.normalize(
                item.pilot_subtask
            ).normalized_subtask
            is expected_subtask
        )

        print("\n" + "=" * 88)
        print(f"EXAMPLE: {label.upper()}")
        print(f"origin_id:   {record.anonymous_sample_id}")
        print(f"raw subtask: {record.pilot_subtask}")
        print(f"instruction: {record.raw_record['instruction']}")

        print("\n[1] Normalize the raw subtask")
        mapping = DEFAULT_SUBTASK_NORMALIZER.normalize(record.pilot_subtask)
        print(
            f"    {record.pilot_subtask!r} -> "
            f"EditingSubtask.{mapping.normalized_subtask.name}"
        )

        print("\n[2] Select the fixed graph schema and FORMAL grammar")
        definition = editing_schema_for(mapping.normalized_subtask)
        builder = reference_dag_builder_for(mapping.normalized_subtask)
        print(f"    schema_id: {definition.schema.schema_id}")
        print(f"    builder:   {type(builder).__name__}")
        print(f"    steps:     {len(builder.trace_contracts)}")

        print("\n[3] Parse each FORMAL step and identify its DAG nodes")
        state = record.process_record["parsed_reference_state"]
        value_types = {
            field_name: definition.schema.nodes_by_id[node_id].value_type
            for field_name, node_id in definition.legacy_step_field_bindings.items()
        }
        trace = record.process_record["formal_cot_trace"]
        for payload, contract in zip(trace, builder.trace_contracts, strict=True):
            parsed = contract.parse(
                payload["formal_ab"],
                state=state,
                value_types=value_types,
            )
            if parsed is None:  # The real builder would fail closed here too.
                raise RuntimeError(
                    f"demo could not parse step {contract.step_index} for "
                    f"{record.anonymous_sample_id}"
                )
            print(f"\n    Step {contract.step_index}: {contract.step_name}")
            print(f"      FORMAL: {payload['formal_ab']}")
            for source_field in contract.fields:
                node_id = definition.legacy_step_field_bindings[source_field]
                rendered_value = repr(parsed[source_field])
                if len(rendered_value) > 96:
                    rendered_value = f"{rendered_value[:93]}..."
                print(
                    f"      field {source_field:<28}"
                    f" -> node {node_id:<24}"
                    f" = {rendered_value}"
                )

        print("\n[4] Call the real builder: StateDAG(schema, ClaimValue nodes)")
        artifact = builder.build(record)
        graph = artifact.state_dag
        print(
            f"    built {len(graph.values)} nodes and "
            f"{len(graph.schema.edges)} directed edges"
        )

        print("\n[5] Final nodes (topological order)")
        for node_id in graph.schema.topological_order():
            spec = graph.schema.nodes_by_id[node_id]
            claim = graph.values[node_id]
            rendered_value = repr(claim.normalized_value)
            if len(rendered_value) > 96:
                rendered_value = f"{rendered_value[:93]}..."
            print(
                f"    {node_id:<24}"
                f" type={spec.value_type.value:<16}"
                f" provenance={claim.provenance.value:<10}"
                f" visibility={spec.visibility.value:<16}"
                f" value={rendered_value}"
            )

        print("\n[6] Final graph edges: source --relation--> target")
        for edge in graph.schema.edges:
            print(
                f"    {edge.source:<24}"
                f" --{edge.relation.value}--> "
                f"{edge.target}"
            )
