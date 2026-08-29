"""Stable molecule-editing family/subtask perturbator types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from molhallulens.adapters import JoinedInputRecord
from molhallulens.domain import (
    EditingSubtask,
    EditKind,
    EditTruth,
    StateDAG,
    StateSchema,
    TaskFamily,
    TaskRecord,
    state_schema_for,
)

from ..base import (
    CandidateEngine,
    LabelProjector,
    PerturbationStage,
    Perturbator,
    PerturbatorExecutionError,
    PropagationEngine,
    TraceRenderer,
    ValidatorChain,
)

if TYPE_CHECKING:
    from molhallulens.builders.reference_dag import ReferenceDAGArtifact
    from molhallulens.validation import OriginValidationInput, OriginValidationReport


EDITING_REFERENCE_ENVELOPE_METADATA_KEY = "editing_reference_envelope"


@dataclass(frozen=True, slots=True)
class EditingReferenceEnvelope:
    """Lossless, T015-validated build input retained by a normalized TaskRecord."""

    joined_input_record: JoinedInputRecord
    validation_report: OriginValidationReport

    def __post_init__(self) -> None:
        from molhallulens.validation import OriginValidationReport

        if type(self.joined_input_record) is not JoinedInputRecord:
            raise TypeError("joined_input_record must be JoinedInputRecord")
        if type(self.validation_report) is not OriginValidationReport:
            raise TypeError("validation_report must be OriginValidationReport")
        origin_id = self.joined_input_record.anonymous_sample_id
        if not self.validation_report.all_pass:
            raise ValueError("EditingReferenceEnvelope requires a passing T015 report")
        if not (
            self.validation_report.anonymous_sample_id == origin_id
            and self.validation_report.operation_subtype is not None
        ):
            raise ValueError("EditingReferenceEnvelope report must describe its joined origin")

    @property
    def reference_artifact(self) -> ReferenceDAGArtifact:
        """Deterministically replay T011; artifacts are not copied through metadata."""

        from molhallulens.builders import build_reference_dag

        return build_reference_dag(self.joined_input_record)

    @property
    def edit_truth(self) -> EditTruth:
        """Deterministically replay T013 from the authoritative T011 artifact."""

        from molhallulens.builders import derive_edit_truth

        return derive_edit_truth(self.reference_artifact)


def task_record_from_validated_reference(
    item: OriginValidationInput,
) -> TaskRecord:
    """Strictly validate T015 input and create its normalized pipeline record."""

    from molhallulens.validation import (
        OriginValidationInput,
        validate_reference_origin_strict,
    )

    if type(item) is not OriginValidationInput:
        raise TypeError("item must be OriginValidationInput")
    report = validate_reference_origin_strict(item)
    envelope = EditingReferenceEnvelope(
        joined_input_record=item.record,
        validation_report=report,
    )
    raw = item.record.raw_record
    process = item.record.process_record
    return TaskRecord(
        origin_id=item.record.anonymous_sample_id,
        anonymous_sample_id=item.record.anonymous_sample_id,
        family=TaskFamily.MOLECULE_EDITING,
        source_subtask=item.record.pilot_subtask,
        normalized_subtask=item.artifact.normalized_subtask,
        operation_subtype=report.operation_subtype,
        indexed_smiles=raw["indexed_smiles"],
        instruction=raw["instruction"],
        gt_smiles=raw["gt_smiles"],
        reference_reasoning_chain=item.artifact.reasoning_chain,
        reference_final_answer=process["answer_smiles"],
        parsed_reference_state=process["parsed_reference_state"],
        raw_metadata={EDITING_REFERENCE_ENVELOPE_METADATA_KEY: envelope},
    )


def task_record_from_joined_input(record: JoinedInputRecord) -> TaskRecord:
    """Build, derive, validate, and losslessly envelope one joined Pilot origin."""

    from molhallulens.builders import build_reference_dag, derive_edit_truth
    from molhallulens.validation import OriginValidationInput

    if type(record) is not JoinedInputRecord:
        raise TypeError("record must be JoinedInputRecord")
    artifact = build_reference_dag(record)
    truth = derive_edit_truth(artifact)
    item = OriginValidationInput(record=record, artifact=artifact, edit_truth=truth)
    return task_record_from_validated_reference(item)


class MoleculeEditingPerturbator(Perturbator[EditTruth], ABC):
    """Common molecule-editing type boundary; chemistry stays in injected ports."""

    family: ClassVar[str] = "mol_edit"
    normalized_subtask: ClassVar[EditingSubtask | None] = None

    def __init__(
        self,
        *,
        candidate_engine: CandidateEngine[EditTruth],
        propagator: PropagationEngine[EditTruth],
        renderer: TraceRenderer[EditTruth],
        validators: ValidatorChain[EditTruth],
        label_projector: LabelProjector[EditTruth],
    ) -> None:
        super().__init__(
            candidate_engine=candidate_engine,
            propagator=propagator,
            renderer=renderer,
            validators=validators,
            label_projector=label_projector,
        )

    def state_schema(self) -> StateSchema:
        if self.normalized_subtask is None:
            raise PerturbatorExecutionError(
                code="ABSTRACT_EDITING_SUBTASK",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id="unknown",
                detail="molecule-editing base has no normalized subtask",
            )
        return state_schema_for(self.normalized_subtask)

    def _validate_truth(self, truth: EditTruth, *, origin_id: str) -> None:
        if type(truth) is not EditTruth:
            raise PerturbatorExecutionError(
                code="EDIT_TRUTH_CONTRACT_VIOLATION",
                stage=PerturbationStage.TRUTH_DERIVATION,
                origin_id=origin_id,
                detail="molecule-editing truth hook did not return EditTruth",
            )

    def _reference_envelope(self, record: TaskRecord) -> EditingReferenceEnvelope:
        envelope = record.raw_metadata.get(EDITING_REFERENCE_ENVELOPE_METADATA_KEY)
        if type(envelope) is not EditingReferenceEnvelope:
            raise PerturbatorExecutionError(
                code="EDITING_REFERENCE_ENVELOPE_MISSING",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=record.origin_id,
                detail=(
                    "TaskRecord must be created by the validated joined-input "
                    "converter and retain its frozen reference envelope"
                ),
            )
        joined = envelope.joined_input_record
        raw = joined.raw_record
        process = joined.process_record
        identity_matches = (
            joined.anonymous_sample_id == record.anonymous_sample_id
            and joined.anonymous_sample_id == record.origin_id
            and raw.get("task_family") == record.family.value
            and joined.pilot_subtask == record.source_subtask
            and raw.get("indexed_smiles") == record.indexed_smiles
            and raw.get("instruction") == record.instruction
            and raw.get("gt_smiles") == record.gt_smiles
            and process.get("answer_smiles") == record.reference_final_answer
            and process.get("parsed_reference_state") == record.parsed_reference_state
        )
        if not identity_matches:
            raise PerturbatorExecutionError(
                code="EDITING_REFERENCE_ENVELOPE_MISMATCH",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=record.origin_id,
                detail="validated reference envelope does not match normalized TaskRecord fields",
            )
        if (
            envelope.validation_report.normalized_subtask
            is not record.normalized_subtask
            or envelope.validation_report.operation_subtype
            is not record.operation_subtype
        ):
            raise PerturbatorExecutionError(
                code="EDITING_REFERENCE_ENVELOPE_MISMATCH",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=record.origin_id,
                detail="validated artifacts differ from normalized TaskRecord semantics",
            )
        return envelope

    def build_reference_dag(self, record: TaskRecord) -> StateDAG:
        """Delegate normalized editing reference construction to the T011 builder."""

        envelope = self._reference_envelope(record)
        artifact = envelope.reference_artifact
        if artifact.reasoning_chain != record.reference_reasoning_chain:
            raise PerturbatorExecutionError(
                code="REFERENCE_ARTIFACT_MISMATCH",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=record.origin_id,
                detail="rebuilt reference reasoning differs from normalized TaskRecord",
            )
        return artifact.state_dag

    def derive_truth(self, record: TaskRecord, dag: StateDAG) -> EditTruth:
        """Delegate graph truth derivation to T013 after deterministic DAG replay."""

        from molhallulens.builders import derive_edit_truth

        envelope = self._reference_envelope(record)
        artifact = envelope.reference_artifact
        if artifact.state_dag != dag:
            raise PerturbatorExecutionError(
                code="REFERENCE_DAG_REPLAY_MISMATCH",
                stage=PerturbationStage.TRUTH_DERIVATION,
                origin_id=record.origin_id,
                detail="derive_truth DAG differs from the authoritative rebuilt artifact",
            )
        return derive_edit_truth(artifact)

    @abstractmethod
    def expected_edit_kind(self) -> EditKind:
        """Return the stable graph edit kind admitted by this subtask class."""

    # These public family capabilities are intentionally closed placeholders in
    # T016. Their chemistry and operator behavior belong to later tasks.
    def derive_graph_diff(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("derive_graph_diff")

    def apply_edit_action(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("apply_edit_action")

    def enumerate_attachment_sites(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("enumerate_attachment_sites")

    def enumerate_removable_groups(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("enumerate_removable_groups")

    def compare_molecules(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("compare_molecules")

    def validate_edit_family(self, *_args: Any, **_kwargs: Any) -> Any:
        self._future_capability("validate_edit_family")

    def _future_capability(self, capability: str) -> None:
        raise PerturbatorExecutionError(
            code="EDITING_CAPABILITY_UNAVAILABLE",
            stage=PerturbationStage.CANDIDATE_ENUMERATION,
            origin_id="unknown",
            detail=f"{capability} is intentionally not implemented by the T016 skeleton",
        )


from .addition import (
    ADDITION_OPERATOR_IDS,
    AdditionCandidateDispatcher,
    AdditionCandidateEngine,
    AdditionOperatorMixin,
)


class AdditionPerturbator(AdditionOperatorMixin, MoleculeEditingPerturbator):
    """Concrete normalized ``mol_edit/add`` perturbator type."""

    subtask: ClassVar[str] = "add"
    normalized_subtask: ClassVar[EditingSubtask] = EditingSubtask.ADD
    __molhallulens_operator_member_mixins__ = (AdditionOperatorMixin,)

    def __init__(self, **ports: Any) -> None:
        super().__init__(**ports)
        if type(self.candidate_engine) is AdditionCandidateEngine:
            self.candidate_engine.bind_owner(self)

    def expected_edit_kind(self) -> EditKind:
        return EditKind.ADDITION


class DeletionPerturbator(MoleculeEditingPerturbator):
    """Concrete normalized ``mol_edit/delete`` perturbator type."""

    subtask: ClassVar[str] = "delete"
    normalized_subtask: ClassVar[EditingSubtask] = EditingSubtask.DELETE

    def expected_edit_kind(self) -> EditKind:
        return EditKind.DELETION


class SubstitutionPerturbator(MoleculeEditingPerturbator):
    """Concrete normalized ``mol_edit/substitute`` perturbator type."""

    subtask: ClassVar[str] = "substitute"
    normalized_subtask: ClassVar[EditingSubtask] = EditingSubtask.SUBSTITUTE

    def expected_edit_kind(self) -> EditKind:
        return EditKind.SUBSTITUTION


__all__ = [
    "ADDITION_OPERATOR_IDS",
    "EDITING_REFERENCE_ENVELOPE_METADATA_KEY",
    "AdditionCandidateDispatcher",
    "AdditionCandidateEngine",
    "AdditionOperatorMixin",
    "AdditionPerturbator",
    "DeletionPerturbator",
    "EditingReferenceEnvelope",
    "MoleculeEditingPerturbator",
    "SubstitutionPerturbator",
    "task_record_from_joined_input",
    "task_record_from_validated_reference",
]
