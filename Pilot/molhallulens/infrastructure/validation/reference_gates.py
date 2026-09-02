"""The four deterministic validation gates for reference molecule edits."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS

from molhallulens.modules.ingestion import (
    DEFAULT_SUBTASK_NORMALIZER,
    JoinedInputRecord,
    SubtaskNormalizationError,
)
from molhallulens.modules.reference.anomaly_registry import (
    AnomalyRegistryError,
    classify_edit_truth,
)
from molhallulens.modules.reference.truth import EditTruthBuildError, EditTruthBuilder
from molhallulens.modules.reference.builder import (
    ReferenceDAGArtifact,
    ReferenceDAGBuildError,
    build_reference_dag,
)
from molhallulens.infrastructure.chemistry import (
    FragmentPolicy,
    MoleculeParseError,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
    select_main_fragment,
)
from molhallulens.core import (
    AnomalyProvenance,
    EditingSubtask,
    EditTruth,
    Severity,
    ValidationIssue,
    ValidationReport,
    ValidationStage,
    ValueProvenance,
    editing_schema_for,
)
from molhallulens.core.edit_truth import BondEdit, FragmentSpec
from molhallulens.core.molecules import AtomReference, AtomReferenceNamespace


# These are persisted data-contract IDs, not Python import paths.  Keep their
# v1 values stable when implementation modules move.
_INPUT_VALIDATOR_ID = "molhallulens.validation.input_record.v1"
_REFERENCE_VALIDATOR_ID = "molhallulens.validation.reference_dag.v1"
_RDKIT_VALIDATOR_ID = "molhallulens.validation.rdkit_structure.v1"
_GRAPH_VALIDATOR_ID = "molhallulens.validation.graph_edit.v1"

VALIDATION_GATE_IDS = (
    _INPUT_VALIDATOR_ID,
    _REFERENCE_VALIDATOR_ID,
    _RDKIT_VALIDATOR_ID,
    _GRAPH_VALIDATOR_ID,
)

VALIDATION_GATE_STAGES = {
    _INPUT_VALIDATOR_ID: ValidationStage.INPUT_RECORD,
    _REFERENCE_VALIDATOR_ID: ValidationStage.REFERENCE_DAG,
    _RDKIT_VALIDATOR_ID: ValidationStage.RDKIT_STRUCTURE,
    _GRAPH_VALIDATOR_ID: ValidationStage.GRAPH_EDIT,
}

_REQUIRED_RAW_FIELDS = frozenset(
    {
        "anonymous_sample_id",
        "task_family",
        "subtask",
        "reporting_task",
        "orig_id",
        "indexed_smiles",
        "instruction",
        "gt_smiles",
    }
)
_REQUIRED_PROCESS_FIELDS = frozenset(
    {
        "anonymous_sample_id",
        "task_family",
        "subtask",
        "reporting_task",
        "orig_id",
        "sample_id",
        "formal_cot_trace",
        "gt_smiles",
        "answer_smiles",
        "outcome",
        "parsed_reference_state",
        "verifier_checks",
    }
)
_REQUIRED_TEMPLATE_FIELDS = frozenset(
    {
        "task_family",
        "subtask",
        "reporting_task",
        "n_samples",
        "step_fields",
        "rdkit_reference_fields",
        "verifier_fields",
        "notes",
    }
)
_SHARED_FIELDS = (
    "anonymous_sample_id",
    "task_family",
    "subtask",
    "reporting_task",
    "orig_id",
    "gt_smiles",
)
_REPORTING_TASKS = {
    EditingSubtask.ADD: "MolEdit/Add",
    EditingSubtask.DELETE: "MolEdit/Delete",
    EditingSubtask.SUBSTITUTE: "MolEdit/Substitute",
}
_NONE_FRAGMENT_VALUES = frozenset({"", "none", "null", "nil", "n/a"})


@dataclass(frozen=True, slots=True)
class _BoundaryCapAllowance:
    anonymous_sample_id: str
    node_id: str
    claim_smiles: str
    truth_smiles: str
    required_provenance: AnomalyProvenance


_BOUNDARY_CAP_ALLOWANCES = (
    _BoundaryCapAllowance(
        anonymous_sample_id="mol_edit.substitute_v2.0191",
        node_id="remove_group",
        claim_smiles="O",
        truth_smiles="[O-]",
        required_provenance=AnomalyProvenance.RETAINED_BOUNDARY_VALENCE_RELAXATION,
    ),
)


def _stable_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Enum):
        return ("enum", value.value)
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _stable_key(item)) for key, item in value.items())),
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return ("sequence", tuple(sorted(_stable_key(item) for item in value)))
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _issue(
    code: str,
    message: str,
    *,
    stage: ValidationStage,
    anonymous_sample_id: str,
    severity: Severity = Severity.ERROR,
    evidence: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        stage=stage,
        node_ids=(anonymous_sample_id,),
        message=message,
        evidence={} if evidence is None else evidence,
    )


def _report(
    validator_id: str,
    issues: list[ValidationIssue],
) -> ValidationReport:
    return ValidationReport(
        validator_id,
        tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.node_ids,
                    item.code,
                    item.message,
                    _stable_key(item.evidence),
                ),
            )
        ),
    )


def _add(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    *,
    stage: ValidationStage,
    anonymous_sample_id: str,
    severity: Severity = Severity.ERROR,
    **evidence: Any,
) -> None:
    issues.append(
        _issue(
            code,
            message,
            stage=stage,
            anonymous_sample_id=anonymous_sample_id,
            severity=severity,
            evidence={key: value for key, value in evidence.items() if value is not None},
        )
    )


def _molecule_error_evidence(error: MoleculeParseError, *, field: str) -> dict[str, Any]:
    return {
        "field": field,
        "molecule_error_code": error.code.value,
        "input_length": error.input_length,
    }


@dataclass(frozen=True, slots=True)
class InputRecordValidator:
    """Validate the joined input contract before any artifact is trusted."""

    validator_id: ClassVar[str] = _INPUT_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.INPUT_RECORD

    def validate(self, record: JoinedInputRecord) -> ValidationReport:
        if type(record) is not JoinedInputRecord:
            raise TypeError("InputRecordValidator requires a JoinedInputRecord")
        origin_id = record.anonymous_sample_id
        issues: list[ValidationIssue] = []

        for payload, required, source_name in (
            (record.raw_record, _REQUIRED_RAW_FIELDS, "raw_record"),
            (record.process_record, _REQUIRED_PROCESS_FIELDS, "process_record"),
            (record.formal_template, _REQUIRED_TEMPLATE_FIELDS, "formal_template"),
        ):
            missing = tuple(sorted(required - set(payload)))
            if missing:
                _add(
                    issues,
                    "INPUT_REQUIRED_FIELDS_MISSING",
                    "joined input namespace is missing required fields",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=Severity.FATAL,
                    source=source_name,
                    missing_fields=missing,
                )

        for field_name in _SHARED_FIELDS:
            raw_value = record.raw_record.get(field_name)
            process_value = record.process_record.get(field_name)
            if raw_value != process_value:
                _add(
                    issues,
                    "INPUT_SHARED_FIELD_MISMATCH",
                    "raw and process authoritative fields differ",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=Severity.FATAL,
                    field=field_name,
                    raw_type=type(raw_value).__name__,
                    process_type=type(process_value).__name__,
                )
        if (
            record.raw_record.get("anonymous_sample_id") != origin_id
            or record.process_record.get("anonymous_sample_id") != origin_id
        ):
            _add(
                issues,
                "INPUT_ID_MISMATCH",
                "joined, raw, and process anonymous IDs must agree exactly",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )

        mapping = None
        try:
            mapping = DEFAULT_SUBTASK_NORMALIZER.reconcile(
                record.raw_record.get("subtask"),
                record.process_record.get("subtask"),
                record.formal_template.get("subtask"),
            )
        except (SubtaskNormalizationError, TypeError, ValueError):
            _add(
                issues,
                "INPUT_SUBTASK_INVALID",
                "input subtask names do not resolve to one registered mapping",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )
        if mapping is not None:
            expected_reporting = _REPORTING_TASKS[mapping.normalized_subtask]
            if any(
                payload.get("reporting_task") != expected_reporting
                for payload in (
                    record.raw_record,
                    record.process_record,
                    record.formal_template,
                )
            ):
                _add(
                    issues,
                    "INPUT_SUBTASK_INVALID",
                    "reporting task disagrees with normalized subtask",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=Severity.FATAL,
                    expected_reporting_task=expected_reporting,
                )
            expected_prefix = f"mol_edit.{mapping.source_subtask}."
            suffix = origin_id.removeprefix(expected_prefix)
            if (
                not origin_id.startswith(expected_prefix)
                or re.fullmatch(r"[0-9]{4}", suffix) is None
            ):
                _add(
                    issues,
                    "INPUT_ID_MISMATCH",
                    "anonymous ID does not match its explicit family/subtask contract",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=Severity.FATAL,
                    expected_prefix=expected_prefix,
                )
        if any(
            payload.get("task_family") != "mol_edit"
            for payload in (
                record.raw_record,
                record.process_record,
                record.formal_template,
            )
        ):
            _add(
                issues,
                "INPUT_SUBTASK_INVALID",
                "input task family is not the molecule-editing contract",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )

        for field_name, issue_code in (
            ("indexed_smiles", "INPUT_SOURCE_PARSE_FAILED"),
            ("gt_smiles", "INPUT_GT_PARSE_FAILED"),
        ):
            try:
                canonicalize_smiles(record.raw_record.get(field_name))
            except MoleculeParseError as error:
                issues.append(
                    _issue(
                        issue_code,
                        "input molecule failed strict parse/sanitize",
                        stage=self.stage,
                        anonymous_sample_id=origin_id,
                        severity=Severity.FATAL,
                        evidence=_molecule_error_evidence(error, field=field_name),
                    )
                )

        return _report(self.validator_id, issues)


@dataclass(frozen=True, slots=True)
class ReferenceDAGValidator:
    """Validate reference state, dependencies, equivalence, and trace round-trip."""

    validator_id: ClassVar[str] = _REFERENCE_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.REFERENCE_DAG

    def validate(
        self,
        record: JoinedInputRecord,
        artifact: ReferenceDAGArtifact,
    ) -> ValidationReport:
        if type(record) is not JoinedInputRecord:
            raise TypeError("ReferenceDAGValidator record must be JoinedInputRecord")
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("ReferenceDAGValidator artifact must be ReferenceDAGArtifact")
        origin_id = record.anonymous_sample_id
        issues: list[ValidationIssue] = []

        if artifact.anonymous_sample_id != origin_id:
            _add(
                issues,
                "REFERENCE_ID_MISMATCH",
                "reference artifact and input origin IDs differ",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                artifact_id=artifact.anonymous_sample_id,
            )
        if artifact.legacy_orig_id != record.raw_record.get("orig_id") or (
            artifact.legacy_sample_id != record.process_record.get("sample_id")
        ):
            _add(
                issues,
                "REFERENCE_ID_MISMATCH",
                "reference artifact legacy IDs differ from input provenance",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )

        try:
            mapping = DEFAULT_SUBTASK_NORMALIZER.normalize(record.pilot_subtask)
        except (SubtaskNormalizationError, TypeError):
            mapping = None
        if mapping is None or artifact.normalized_subtask is not mapping.normalized_subtask:
            _add(
                issues,
                "REFERENCE_SUBTASK_MISMATCH",
                "reference artifact subtask differs from the input mapping",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                actual_subtask=artifact.normalized_subtask.value,
            )

        definition = editing_schema_for(artifact.normalized_subtask)
        if artifact.state_dag.schema is not definition.schema:
            _add(
                issues,
                "REFERENCE_SCHEMA_MISMATCH",
                "reference DAG does not use the authoritative typed schema",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                schema_id=artifact.state_dag.schema.schema_id,
                schema_version=artifact.state_dag.schema.version,
            )
        try:
            topo = artifact.state_dag.schema.topological_order()
        except (TypeError, ValueError, KeyError):
            topo = ()
        if len(topo) != len(artifact.state_dag.schema.nodes) or artifact.state_dag.edge_values:
            _add(
                issues,
                "REFERENCE_DEPENDENCY_INVALID",
                "reference DAG dependencies are incomplete, cyclic, or prematurely valued",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                topo_node_count=len(topo),
                schema_node_count=len(artifact.state_dag.schema.nodes),
                edge_value_count=len(artifact.state_dag.edge_values),
            )

        parsed_state = record.process_record.get("parsed_reference_state")
        if not isinstance(parsed_state, Mapping):
            parsed_state = {}
        expected_values: dict[str, tuple[Any, ValueProvenance]] = {}
        for field_name, node_id in definition.record_field_bindings.items():
            source = (
                record.process_record
                if field_name == "answer_smiles"
                else record.raw_record
            )
            expected_values[node_id] = (
                source.get(field_name),
                ValueProvenance.REFERENCE,
            )
        for field_name, node_id in definition.legacy_step_field_bindings.items():
            expected_values[node_id] = (
                parsed_state.get(field_name),
                ValueProvenance.REFERENCE,
            )
        for field_name, node_id in definition.rdkit_reference_bindings.items():
            expected_values[node_id] = (
                parsed_state.get(field_name),
                ValueProvenance.RDKIT,
            )
        mismatched_nodes = []
        for node_id, (expected, provenance) in expected_values.items():
            claim = artifact.state_dag.values.get(node_id)
            if claim is None or (
                claim.raw_value != expected
                or claim.normalized_value != expected
                or claim.provenance is not provenance
            ):
                mismatched_nodes.append(node_id)
        if mismatched_nodes:
            _add(
                issues,
                "REFERENCE_STATE_BINDING_MISMATCH",
                "reference claims do not round-trip to their namespaced source fields",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                node_ids=tuple(sorted(mismatched_nodes)),
            )

        gt_smiles = record.raw_record.get("gt_smiles")
        product_field = next(
            (
                field_name
                for field_name, node_id in definition.legacy_step_field_bindings.items()
                if node_id == "product"
            ),
            None,
        )
        process_product = (
            None if product_field is None else parsed_state.get(product_field)
        )
        for values, code, label in (
            (
                (
                    process_product,
                    artifact.state_dag.values.get("product"),
                ),
                "REFERENCE_PRODUCT_GT_MISMATCH",
                "process product",
            ),
            (
                (
                    record.process_record.get("answer_smiles"),
                    artifact.state_dag.values.get("final_answer"),
                ),
                "REFERENCE_ANSWER_GT_MISMATCH",
                "process answer",
            ),
        ):
            comparison_error: str | None = None
            equivalent = True
            for value in values:
                molecule_value = (
                    value.normalized_value if hasattr(value, "normalized_value") else value
                )
                try:
                    value_equivalent = isomeric_graph_equivalent(
                        molecule_value,
                        gt_smiles,
                    )
                except MoleculeParseError as error:
                    value_equivalent = False
                    comparison_error = error.code.value
                except (KeyError, TypeError, AttributeError) as error:
                    value_equivalent = False
                    comparison_error = type(error).__name__
                equivalent &= value_equivalent
            if not equivalent:
                _add(
                    issues,
                    code,
                    f"{label} is not isomeric-graph equivalent to GT",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=(
                        Severity.FATAL
                        if comparison_error is not None
                        else Severity.ERROR
                    ),
                    reason=comparison_error,
                )

        trace_payload = record.process_record.get("formal_cot_trace")
        trace_ok = (
            not isinstance(trace_payload, (str, bytes))
            and isinstance(trace_payload, Sequence)
            and len(trace_payload) == len(artifact.trace_steps)
        )
        if trace_ok:
            for index, (payload, step) in enumerate(
                zip(trace_payload, artifact.trace_steps, strict=True), start=1
            ):
                if not isinstance(payload, Mapping) or any(
                    payload.get(field_name) != getattr(step, field_name)
                    for field_name in (
                        "step_index",
                        "step_name",
                        "natural_language",
                        "formal_ab",
                        "step_text",
                    )
                ):
                    trace_ok = False
                    break
                expected_answer = (
                    record.process_record.get("answer_smiles")
                    if index == len(artifact.trace_steps)
                    else None
                )
                if step.answer_suffix != expected_answer or (
                    step.render(include_answer=True) != step.step_text
                ):
                    trace_ok = False
                    break
                for binding in step.slot_bindings:
                    expected_node = definition.legacy_step_field_bindings.get(
                        binding.source_field
                    )
                    if expected_node != binding.node_id:
                        trace_ok = False
                        break
        if not trace_ok:
            _add(
                issues,
                "REFERENCE_TRACE_ROUND_TRIP_FAILED",
                "natural/FORMAL trace no longer round-trips to process state",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )

        try:
            rebuilt = build_reference_dag(record)
        except ReferenceDAGBuildError as error:
            rebuilt = None
            _add(
                issues,
                "REFERENCE_REBUILD_FAILED",
                "authoritative reference builder rejected this joined input",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                builder_issue_codes=tuple(issue.code for issue in error.report.issues),
            )
        if rebuilt is not None and rebuilt != artifact:
            _add(
                issues,
                "REFERENCE_REBUILD_MISMATCH",
                "reference artifact differs from deterministic reconstruction",
                stage=self.stage,
                anonymous_sample_id=origin_id,
            )

        return _report(self.validator_id, issues)


@dataclass(frozen=True, slots=True)
class RDKitStructureValidator:
    """Recompute strict molecular identity, fragment scope, and descriptors."""

    validator_id: ClassVar[str] = _RDKIT_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.RDKIT_STRUCTURE

    def validate(
        self,
        record: JoinedInputRecord,
        artifact: ReferenceDAGArtifact,
        truth: EditTruth,
    ) -> ValidationReport:
        if type(record) is not JoinedInputRecord:
            raise TypeError("RDKitStructureValidator record must be JoinedInputRecord")
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("RDKitStructureValidator artifact must be ReferenceDAGArtifact")
        if type(truth) is not EditTruth:
            raise TypeError("RDKitStructureValidator truth must be EditTruth")
        origin_id = record.anonymous_sample_id
        issues: list[ValidationIssue] = []

        parsed_state = record.process_record.get("parsed_reference_state")
        if not isinstance(parsed_state, Mapping):
            parsed_state = {}
        definition = editing_schema_for(artifact.normalized_subtask)
        product_field = next(
            (
                field_name
                for field_name, node_id in definition.legacy_step_field_bindings.items()
                if node_id == "product"
            ),
            None,
        )
        molecule_fields = {
            "input.source": record.raw_record.get("indexed_smiles"),
            "input.gt": record.raw_record.get("gt_smiles"),
            "process.gt": record.process_record.get("gt_smiles"),
            "process.product": (
                None if product_field is None else parsed_state.get(product_field)
            ),
            "process.answer": record.process_record.get("answer_smiles"),
            "artifact.product": artifact.state_dag.values["product"].normalized_value,
            "artifact.answer": artifact.state_dag.values["final_answer"].normalized_value,
            "truth.source": truth.source_smiles,
            "truth.gt": truth.gt_smiles,
        }
        canonical: dict[str, str] = {}
        for field_name, smiles in molecule_fields.items():
            try:
                canonical[field_name] = canonicalize_smiles(smiles)
            except MoleculeParseError as error:
                issues.append(
                    _issue(
                        "RDKIT_STRICT_SANITIZE_FAILED",
                        "molecule failed strict RDKit parse/sanitize",
                        stage=self.stage,
                        anonymous_sample_id=origin_id,
                        severity=Severity.FATAL,
                        evidence=_molecule_error_evidence(error, field=field_name),
                    )
                )

        for left, right, relation in (
            ("input.source", "truth.source", "source identity"),
            ("input.gt", "process.gt", "raw/process GT identity"),
            ("input.gt", "process.product", "process product vs GT"),
            ("input.gt", "process.answer", "process answer vs GT"),
            ("input.gt", "artifact.product", "artifact product vs GT"),
            ("input.gt", "artifact.answer", "artifact answer vs GT"),
            ("input.gt", "truth.gt", "EditTruth GT identity"),
        ):
            if left in canonical and right in canonical and canonical[left] != canonical[right]:
                _add(
                    issues,
                    "RDKIT_REFERENCE_EQUIVALENCE_FAILED",
                    "reference molecule identities are not isomeric-graph equivalent",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    relation=relation,
                )

        try:
            source_descriptors = compute_descriptors(record.raw_record.get("indexed_smiles"))
            product_descriptors = compute_descriptors(record.raw_record.get("gt_smiles"))
            select_main_fragment(record.raw_record.get("indexed_smiles"))
            select_main_fragment(record.raw_record.get("gt_smiles"))
        except MoleculeParseError:
            source_descriptors = None
            product_descriptors = None
        if source_descriptors is not None and source_descriptors != truth.source_descriptors:
            _add(
                issues,
                "RDKIT_DESCRIPTOR_MISMATCH",
                "source descriptors do not match strict recomputation",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                scope="source",
            )
        if product_descriptors is not None and product_descriptors != truth.product_descriptors:
            _add(
                issues,
                "RDKIT_DESCRIPTOR_MISMATCH",
                "product descriptors do not match strict recomputation",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                scope="product",
            )

        for descriptor, scope in (
            (truth.source_descriptors, "source"),
            (truth.product_descriptors, "product"),
            (
                None if truth.remove_fragment is None else truth.remove_fragment.descriptors,
                "remove_fragment",
            ),
            (
                None if truth.add_fragment is None else truth.add_fragment.descriptors,
                "add_fragment",
            ),
        ):
            if descriptor is not None and descriptor.fragment_policy is not FragmentPolicy.KEEP_ALL:
                _add(
                    issues,
                    "RDKIT_FRAGMENT_POLICY_MISMATCH",
                    "reference chemistry must retain all disconnected components",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    scope=scope,
                    actual_policy=descriptor.fragment_policy.value,
                )

        for spec, scope in (
            (truth.remove_fragment, "remove_fragment"),
            (truth.add_fragment, "add_fragment"),
        ):
            if spec is None:
                continue
            try:
                recalculated = compute_descriptors(spec.canonical_smiles)
            except MoleculeParseError as error:
                issues.append(
                    _issue(
                        "RDKIT_STRICT_SANITIZE_FAILED",
                        "graph-derived fragment failed strict parse/sanitize",
                        stage=self.stage,
                        anonymous_sample_id=origin_id,
                        severity=Severity.FATAL,
                        evidence=_molecule_error_evidence(error, field=scope),
                    )
                )
            else:
                if recalculated != spec.descriptors:
                    _add(
                        issues,
                        "RDKIT_DESCRIPTOR_MISMATCH",
                        "fragment descriptors do not match strict recomputation",
                        stage=self.stage,
                        anonymous_sample_id=origin_id,
                        scope=scope,
                    )

        fragment_nodes = tuple(
            node_id
            for node_id in (
                "leaving",
                "remove_group_step1",
                "remove_group_step2",
                "remove_group",
                "add_fragment",
            )
            if node_id in artifact.state_dag.values
        )
        for node_id in fragment_nodes:
            fragment = artifact.state_dag.values[node_id].normalized_value
            if type(fragment) is str and fragment.strip().lower() in _NONE_FRAGMENT_VALUES:
                continue
            try:
                canonicalize_smiles(fragment)
            except MoleculeParseError as error:
                issues.append(
                    _issue(
                        "RDKIT_STRICT_SANITIZE_FAILED",
                        "reference fragment failed strict parse/sanitize",
                        stage=self.stage,
                        anonymous_sample_id=origin_id,
                        severity=Severity.FATAL,
                        evidence=_molecule_error_evidence(error, field=node_id),
                    )
                )

        if source_descriptors is not None and product_descriptors is not None:
            anchor_element = None
            try:
                with rdBase.BlockLogs():
                    source_molecule = Chem.MolFromSmiles(
                        record.raw_record.get("indexed_smiles"),
                        sanitize=True,
                    )
                anchor_index = artifact.state_dag.values["anchor_idx"].normalized_value
                if source_molecule is not None:
                    anchor_element = next(
                        (
                            atom.GetSymbol()
                            for atom in source_molecule.GetAtoms()
                            if atom.GetAtomMapNum() == anchor_index
                        ),
                        None,
                    )
            except (RuntimeError, ValueError, TypeError, KeyError):
                anchor_element = None
            expected_claims = {
                "source_heavy": source_descriptors.heavy_atom_count,
                "product_heavy": product_descriptors.heavy_atom_count,
                "heavy_delta": (
                    product_descriptors.heavy_atom_count
                    - source_descriptors.heavy_atom_count
                ),
                "source_rings": source_descriptors.ring_count,
                "product_rings": product_descriptors.ring_count,
                "ring_delta": product_descriptors.ring_count - source_descriptors.ring_count,
                "oracle_source_heavy": source_descriptors.heavy_atom_count,
                "oracle_product_heavy": product_descriptors.heavy_atom_count,
                "oracle_source_rings": source_descriptors.ring_count,
                "oracle_product_rings": product_descriptors.ring_count,
                "oracle_anchor_element": anchor_element,
            }
            if truth.add_fragment is not None:
                expected_claims["oracle_fragment_heavy"] = (
                    truth.add_fragment.descriptors.heavy_atom_count
                )
                expected_claims["oracle_add_heavy"] = (
                    truth.add_fragment.descriptors.heavy_atom_count
                )
            if truth.remove_fragment is not None:
                expected_claims["oracle_remove_heavy"] = (
                    truth.remove_fragment.descriptors.heavy_atom_count
                )
            mismatches = tuple(
                sorted(
                    node_id
                    for node_id, expected in expected_claims.items()
                    if node_id in artifact.state_dag.values
                    and artifact.state_dag.values[node_id].normalized_value != expected
                )
            )
            if mismatches:
                _add(
                    issues,
                    "RDKIT_DESCRIPTOR_MISMATCH",
                    "reference descriptor/count claims differ from strict recomputation",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    node_ids=mismatches,
                )

            process_mismatches: list[str] = []
            for field_name, node_id in definition.rdkit_reference_bindings.items():
                expected = expected_claims.get(node_id)
                if expected is None or parsed_state.get(field_name) != expected:
                    process_mismatches.append(field_name)
            if process_mismatches:
                _add(
                    issues,
                    "RDKIT_DESCRIPTOR_MISMATCH",
                    "process rdkit_* fields differ from direct strict recomputation",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    process_fields=tuple(sorted(process_mismatches)),
                )

        return _report(self.validator_id, issues)


def _fragment_boundary_issues(
    *,
    spec: FragmentSpec,
    graph_bonds: tuple[BondEdit, ...],
    valid_anchors: frozenset[int],
    scope: str,
) -> tuple[tuple[int, ...], bool]:
    """Return retained source anchors and whether fragment-side semantics hold."""

    if set(spec.boundary_bonds) != set(graph_bonds):
        return (), False
    fragment_atoms = set(spec.atom_references)
    attachments = set(spec.attachment_atoms)
    retained: set[int] = set()
    seen_attachments: set[AtomReference] = set()
    for bond in graph_bonds:
        inside = tuple(
            endpoint
            for endpoint in (bond.begin, bond.end)
            if endpoint in fragment_atoms
        )
        outside = tuple(
            endpoint
            for endpoint in (bond.begin, bond.end)
            if endpoint not in fragment_atoms
        )
        if len(inside) != 1 or len(outside) != 1 or inside[0] not in attachments:
            return (), False
        seen_attachments.add(inside[0])
        retained_endpoint = outside[0]
        if (
            retained_endpoint.namespace is not AtomReferenceNamespace.SOURCE_MAP
            or retained_endpoint.atom_id not in valid_anchors
        ):
            return (), False
        retained.add(retained_endpoint.atom_id)
        if scope == "remove" and inside[0].namespace is not AtomReferenceNamespace.SOURCE_MAP:
            return (), False
        if scope == "add" and inside[0].namespace is not AtomReferenceNamespace.PRODUCT_CANONICAL:
            return (), False
    return tuple(sorted(retained)), seen_attachments == attachments


def _boundary_capped_heavy_graph_equivalent(left: str, right: str) -> bool:
    """Compare heavy-atom/bond topology while explicitly ignoring only charge/H caps."""

    try:
        with rdBase.BlockLogs():
            left_molecule = Chem.MolFromSmiles(left, sanitize=True)
            right_molecule = Chem.MolFromSmiles(right, sanitize=True)
    except (RuntimeError, ValueError, TypeError):
        return False
    if left_molecule is None or right_molecule is None:
        return False
    if (
        left_molecule.GetNumHeavyAtoms() != right_molecule.GetNumHeavyAtoms()
        or left_molecule.GetNumBonds() != right_molecule.GetNumBonds()
    ):
        return False
    left_caps = tuple(
        sorted(
            (atom.GetFormalCharge(), atom.GetNumExplicitHs())
            for atom in left_molecule.GetAtoms()
        )
    )
    right_caps = tuple(
        sorted(
            (atom.GetFormalCharge(), atom.GetNumExplicitHs())
            for atom in right_molecule.GetAtoms()
        )
    )
    if left_caps == right_caps:
        return False
    parameters = rdFMCS.MCSParameters()
    parameters.AtomTyper = rdFMCS.AtomCompare.CompareElements
    parameters.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    parameters.AtomCompareParameters.MatchChiralTag = True
    parameters.AtomCompareParameters.MatchIsotope = True
    parameters.BondCompareParameters.MatchStereo = True
    parameters.AtomCompareParameters.RingMatchesRingOnly = True
    parameters.BondCompareParameters.RingMatchesRingOnly = True
    parameters.BondCompareParameters.CompleteRingsOnly = True
    parameters.Timeout = 2
    result = rdFMCS.FindMCS((left_molecule, right_molecule), parameters)
    return (
        not result.canceled
        and result.numAtoms == left_molecule.GetNumHeavyAtoms()
        and result.numAtoms == right_molecule.GetNumHeavyAtoms()
        and result.numBonds == left_molecule.GetNumBonds()
        and result.numBonds == right_molecule.GetNumBonds()
    )


@dataclass(frozen=True, slots=True)
class GraphEditValidator:
    """Validate graph truth, all optimal mappings, fragments, and edit family."""

    validator_id: ClassVar[str] = _GRAPH_VALIDATOR_ID
    stage: ClassVar[ValidationStage] = ValidationStage.GRAPH_EDIT

    def validate(
        self,
        record: JoinedInputRecord,
        artifact: ReferenceDAGArtifact,
        truth: EditTruth,
    ) -> ValidationReport:
        if type(record) is not JoinedInputRecord:
            raise TypeError("GraphEditValidator record must be JoinedInputRecord")
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("GraphEditValidator artifact must be ReferenceDAGArtifact")
        if type(truth) is not EditTruth:
            raise TypeError("GraphEditValidator truth must be EditTruth")
        origin_id = record.anonymous_sample_id
        issues: list[ValidationIssue] = []

        if truth.anonymous_sample_id != origin_id or artifact.anonymous_sample_id != origin_id:
            _add(
                issues,
                "GRAPH_TRUTH_ID_MISMATCH",
                "input, reference artifact, and graph truth IDs differ",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
            )
        if truth.normalized_subtask is not artifact.normalized_subtask:
            _add(
                issues,
                "GRAPH_TRUTH_SUBTASK_MISMATCH",
                "reference artifact and graph truth subtasks differ",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                artifact_subtask=artifact.normalized_subtask.value,
                truth_subtask=truth.normalized_subtask.value,
            )

        try:
            classification = classify_edit_truth(truth)
        except AnomalyRegistryError as error:
            classification = None
            issues.extend(
                ValidationIssue(
                    code=issue.code,
                    severity=issue.severity,
                    stage=self.stage,
                    node_ids=(origin_id,),
                    message=issue.message,
                    evidence=issue.evidence,
                )
                for issue in error.report.issues
            )

        trace_anchor = artifact.state_dag.values["anchor_idx"].normalized_value
        trace_element = artifact.state_dag.values["anchor_element"].normalized_value
        valid_anchors = frozenset(truth.valid_anchor_indices)
        if trace_anchor not in valid_anchors:
            _add(
                issues,
                "GRAPH_ANCHOR_MISMATCH",
                "reference anchor is not one of the graph-derived valid anchors",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                trace_anchor=trace_anchor,
                valid_anchors=tuple(sorted(valid_anchors)),
            )

        source_molecule = None
        try:
            with rdBase.BlockLogs():
                source_molecule = Chem.MolFromSmiles(truth.source_smiles, sanitize=True)
        except (RuntimeError, ValueError):
            source_molecule = None
        map_to_element = (
            {}
            if source_molecule is None
            else {
                atom.GetAtomMapNum(): atom.GetSymbol()
                for atom in source_molecule.GetAtoms()
                if atom.GetAtomMapNum() > 0
            }
        )
        if map_to_element.get(trace_anchor) != trace_element:
            _add(
                issues,
                "GRAPH_ANCHOR_ELEMENT_MISMATCH",
                "reference anchor element differs from the source atom map",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=(
                    Severity.FATAL
                    if source_molecule is None
                    else Severity.ERROR
                ),
                trace_anchor=trace_anchor,
                claimed_element=trace_element,
                actual_element=map_to_element.get(trace_anchor),
                reason=("source_parse_failed" if source_molecule is None else None),
            )

        removed = set(truth.removed_atom_maps)
        added = {atom.reference for atom in truth.added_atoms}
        compatible_mapping_count = 0
        for mapping in truth.mapping_evidence.optimal_mappings:
            source_ids = {pair.source.atom_id for pair in mapping.pairs}
            product_refs = {pair.product for pair in mapping.pairs}
            if (
                valid_anchors.issubset(source_ids)
                and removed.isdisjoint(source_ids)
                and added.isdisjoint(product_refs)
            ):
                compatible_mapping_count += 1
        if compatible_mapping_count == 0:
            _add(
                issues,
                "GRAPH_MAPPING_INCOMPATIBLE",
                "no optimal mapping is compatible with all anchors and atom partitions",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                optimal_mapping_count=len(truth.mapping_evidence.optimal_mappings),
            )

        retained_anchors: set[int] = set()
        boundary_ok = True
        if truth.remove_fragment is not None:
            retained, ok = _fragment_boundary_issues(
                spec=truth.remove_fragment,
                graph_bonds=truth.broken_bonds,
                valid_anchors=valid_anchors,
                scope="remove",
            )
            retained_anchors.update(retained)
            boundary_ok &= ok
        elif truth.removed_atom_maps:
            boundary_ok = False
        if truth.add_fragment is not None:
            retained, ok = _fragment_boundary_issues(
                spec=truth.add_fragment,
                graph_bonds=truth.formed_bonds,
                valid_anchors=valid_anchors,
                scope="add",
            )
            retained_anchors.update(retained)
            boundary_ok &= ok
        elif truth.added_atoms:
            boundary_ok = False
        if (truth.remove_fragment is not None or truth.add_fragment is not None) and (
            trace_anchor not in retained_anchors
        ):
            boundary_ok = False
        if not boundary_ok:
            _add(
                issues,
                "GRAPH_BOUNDARY_MISMATCH",
                "fragment-side attachments and retained source anchors disagree with bond edits",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                retained_anchors=tuple(sorted(retained_anchors)),
            )

        fragment_claims: tuple[tuple[str, FragmentSpec | None], ...]
        if artifact.normalized_subtask is EditingSubtask.ADD:
            fragment_claims = (("add_fragment", truth.add_fragment),)
        elif artifact.normalized_subtask is EditingSubtask.DELETE:
            fragment_claims = (("remove_group_step2", truth.remove_fragment),)
        else:
            fragment_claims = (
                ("remove_group", truth.remove_fragment),
                ("add_fragment", truth.add_fragment),
            )
        for node_id, spec in fragment_claims:
            claim_value = artifact.state_dag.values.get(node_id)
            claim = None if claim_value is None else claim_value.normalized_value
            comparison_failed = claim_value is None
            if spec is None:
                equivalent = type(claim) is str and claim.strip().lower() in _NONE_FRAGMENT_VALUES
            else:
                try:
                    equivalent = fragment_graph_equivalent(
                        claim,
                        spec.canonical_smiles,
                    )
                except (MoleculeParseError, TypeError):
                    equivalent = False
                    comparison_failed = True
                if (
                    not equivalent
                    and not comparison_failed
                    and classification is not None
                    and classification.registered
                    and any(
                        allowance.anonymous_sample_id == origin_id
                        and allowance.node_id == node_id
                        and allowance.claim_smiles == claim
                        and allowance.truth_smiles == spec.canonical_smiles
                        and allowance.required_provenance in classification.provenance
                        for allowance in _BOUNDARY_CAP_ALLOWANCES
                    )
                ):
                    equivalent = _boundary_capped_heavy_graph_equivalent(
                        claim,
                        spec.canonical_smiles,
                    )
            if not equivalent:
                _add(
                    issues,
                    "GRAPH_FRAGMENT_MISMATCH",
                    "reference fragment claim differs from graph-derived fragment",
                    stage=self.stage,
                    anonymous_sample_id=origin_id,
                    severity=(Severity.FATAL if comparison_failed else Severity.ERROR),
                    node_id=node_id,
                )

        expected_delta = (
            truth.product_descriptors.heavy_atom_count
            - truth.source_descriptors.heavy_atom_count
        )
        if (
            truth.heavy_atom_delta != expected_delta
            or artifact.state_dag.values["heavy_delta"].normalized_value != expected_delta
        ):
            _add(
                issues,
                "GRAPH_EDIT_DELTA_MISMATCH",
                "atom partitions and claimed heavy-atom delta disagree",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                graph_delta=truth.heavy_atom_delta,
                descriptor_delta=expected_delta,
            )

        try:
            rederived = EditTruthBuilder().build(artifact)
        except EditTruthBuildError as error:
            rederived = None
            _add(
                issues,
                "GRAPH_EDIT_REDERIVATION_MISMATCH",
                "deterministic graph-diff derivation rejected the reference artifact",
                stage=self.stage,
                anonymous_sample_id=origin_id,
                severity=Severity.FATAL,
                builder_issue_codes=tuple(issue.code for issue in error.report.issues),
            )
        if rederived is not None and rederived != truth:
            _add(
                issues,
                "GRAPH_EDIT_REDERIVATION_MISMATCH",
                "stored graph truth differs from deterministic source-to-product rederivation",
                stage=self.stage,
                anonymous_sample_id=origin_id,
            )

        if classification is None:
            _add(
                issues,
                "GRAPH_EDIT_FAMILY_INVALID",
                "graph edit could not be assigned a declared operation subtype",
                stage=self.stage,
                anonymous_sample_id=origin_id,
            )

        return _report(self.validator_id, issues)


INPUT_RECORD_VALIDATOR = InputRecordValidator()
REFERENCE_DAG_VALIDATOR = ReferenceDAGValidator()
RDKIT_STRUCTURE_VALIDATOR = RDKitStructureValidator()
GRAPH_EDIT_VALIDATOR = GraphEditValidator()


__all__ = [
    "GRAPH_EDIT_VALIDATOR",
    "INPUT_RECORD_VALIDATOR",
    "RDKIT_STRUCTURE_VALIDATOR",
    "REFERENCE_DAG_VALIDATOR",
    "GraphEditValidator",
    "InputRecordValidator",
    "RDKitStructureValidator",
    "ReferenceDAGValidator",
    "VALIDATION_GATE_IDS",
    "VALIDATION_GATE_STAGES",
]
