"""Fail-closed Template Method and composition ports for perturbators.

This module deliberately owns orchestration, not chemistry, operator lookup,
propagation rules, rendering algorithms, or tokenization.  Those capabilities
arrive through the five narrow ports below and can therefore be implemented by
later tasks without widening the family/subtask inheritance hierarchy.
"""

from __future__ import annotations

import hashlib
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Generic, Protocol, TypeVar, final, runtime_checkable

from molhallulens.domain import (
    BuildProvenance,
    CandidatePatch,
    CandidatePool,
    CharAnnotation,
    DetectorInput,
    GraphDelta,
    PerturbationRecipe,
    PerturbationResult,
    StateDAG,
    StateSchema,
    TaskRecord,
    TokenLabelSet,
    TraceLabels,
    ValidationReport,
    VariantLabel,
)


TruthT = TypeVar("TruthT")


class PerturbationStage(StrEnum):
    """Stable stages used by structured orchestration failures."""

    INGEST = "ingest"
    REFERENCE_BUILD = "reference_build"
    TRUTH_DERIVATION = "truth_derivation"
    REFERENCE_VALIDATION = "reference_validation"
    CANDIDATE_ENUMERATION = "candidate_enumeration"
    CANDIDATE_SELECTION = "candidate_selection"
    PROPAGATION = "propagation"
    RENDERING = "rendering"
    LABEL_PROJECTION = "label_projection"
    ARTIFACT_VALIDATION = "artifact_validation"
    RESULT_CONSTRUCTION = "result_construction"


class FinalTemplateMethodError(TypeError):
    """Raised when a subclass attempts to replace the sealed Template Method."""


class PerturbatorExecutionError(RuntimeError):
    """A structured, plaintext-minimizing, fail-closed pipeline failure."""

    def __init__(
        self,
        *,
        code: str,
        stage: PerturbationStage,
        origin_id: str,
        detail: str,
        cause: Exception | None = None,
        report: ValidationReport | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("PerturbatorExecutionError code must be non-empty text")
        if type(stage) is not PerturbationStage:
            raise TypeError("PerturbatorExecutionError stage must be PerturbationStage")
        if type(origin_id) is not str or not origin_id:
            raise ValueError("PerturbatorExecutionError origin_id must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("PerturbatorExecutionError detail must be non-empty text")
        if cause is not None and not isinstance(cause, Exception):
            raise TypeError("PerturbatorExecutionError cause must be an Exception or None")
        if report is not None and type(report) is not ValidationReport:
            raise TypeError("PerturbatorExecutionError report must be ValidationReport or None")
        self.code = code
        self.stage = stage
        self.origin_id = origin_id
        self.detail = detail
        self.cause = cause
        self.report = report
        super().__init__(f"{code} at {stage.value} for {origin_id}: {detail}")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "stage": self.stage.value,
            "origin_id": self.origin_id,
            "detail": self.detail,
        }


class PerturbatorConfigurationError(TypeError):
    """Raised when a dependency does not implement its declared port."""


@dataclass(frozen=True, slots=True)
class PerturbationContext(Generic[TruthT]):
    """Immutable inputs shared by every injected strategy."""

    record: TaskRecord
    recipe: PerturbationRecipe
    state_schema: StateSchema
    reference_graph: StateDAG
    truth: TruthT

    def __post_init__(self) -> None:
        if type(self.record) is not TaskRecord:
            raise TypeError("PerturbationContext record must be TaskRecord")
        if type(self.recipe) is not PerturbationRecipe:
            raise TypeError("PerturbationContext recipe must be PerturbationRecipe")
        if type(self.state_schema) is not StateSchema:
            raise TypeError("PerturbationContext state_schema must be StateSchema")
        if type(self.reference_graph) is not StateDAG:
            raise TypeError("PerturbationContext reference_graph must be StateDAG")
        if self.record.origin_id != self.recipe.origin_id:
            raise ValueError("record and recipe origin_id values must match")
        if self.reference_graph.schema != self.state_schema:
            raise ValueError("reference_graph must use state_schema")


@dataclass(frozen=True, slots=True)
class PropagationOutcome:
    """The only state a propagation strategy may return at this boundary."""

    candidate_graph: StateDAG
    graph_delta: GraphDelta

    def __post_init__(self) -> None:
        if type(self.candidate_graph) is not StateDAG:
            raise TypeError("PropagationOutcome candidate_graph must be StateDAG")
        if type(self.graph_delta) is not GraphDelta:
            raise TypeError("PropagationOutcome graph_delta must be GraphDelta")


@dataclass(frozen=True, slots=True)
class RenderedPerturbation:
    """Renderer output before tokenizer projection and final validation."""

    record_id: str
    origin_id: str
    leakage_group_id: str
    bundle_id: str
    pair_id: str
    matched_record_id: str
    variant_label: VariantLabel
    detector_input: DetectorInput
    serialized_text: str
    serialized_text_sha256: str
    char_annotations: tuple[CharAnnotation, ...]
    trace_labels: TraceLabels
    provenance: BuildProvenance

    def __post_init__(self) -> None:
        for value, name in (
            (self.record_id, "record_id"),
            (self.origin_id, "origin_id"),
            (self.leakage_group_id, "leakage_group_id"),
            (self.bundle_id, "bundle_id"),
            (self.pair_id, "pair_id"),
            (self.matched_record_id, "matched_record_id"),
            (self.serialized_text, "serialized_text"),
            (self.serialized_text_sha256, "serialized_text_sha256"),
        ):
            if type(value) is not str:
                raise TypeError(f"RenderedPerturbation {name} must be a string")
            if not value:
                raise ValueError(f"RenderedPerturbation {name} cannot be empty")
        if type(self.variant_label) is not VariantLabel:
            raise TypeError("RenderedPerturbation variant_label must be VariantLabel")
        if type(self.detector_input) is not DetectorInput:
            raise TypeError("RenderedPerturbation detector_input must be DetectorInput")
        if type(self.trace_labels) is not TraceLabels:
            raise TypeError("RenderedPerturbation trace_labels must be TraceLabels")
        if type(self.provenance) is not BuildProvenance:
            raise TypeError("RenderedPerturbation provenance must be BuildProvenance")
        annotations = tuple(self.char_annotations)
        if any(type(annotation) is not CharAnnotation for annotation in annotations):
            raise TypeError(
                "RenderedPerturbation char_annotations must contain CharAnnotation values"
            )
        object.__setattr__(self, "char_annotations", annotations)
        actual_hash = hashlib.sha256(self.serialized_text.encode("utf-8")).hexdigest()
        if self.serialized_text_sha256 != actual_hash:
            raise ValueError("serialized_text_sha256 does not match serialized_text")


@dataclass(frozen=True, slots=True)
class PerturbationDraft(Generic[TruthT]):
    """Complete pre-validation artifact exposed to the validator strategy."""

    context: PerturbationContext[TruthT]
    root_patch: CandidatePatch
    propagation: PropagationOutcome
    rendered: RenderedPerturbation
    token_labels: TokenLabelSet | None
    reference_validation_report: ValidationReport

    def __post_init__(self) -> None:
        if not isinstance(self.context, PerturbationContext):
            raise TypeError("PerturbationDraft context must be PerturbationContext")
        if type(self.root_patch) is not CandidatePatch:
            raise TypeError("PerturbationDraft root_patch must be CandidatePatch")
        if type(self.propagation) is not PropagationOutcome:
            raise TypeError("PerturbationDraft propagation must be PropagationOutcome")
        if type(self.rendered) is not RenderedPerturbation:
            raise TypeError("PerturbationDraft rendered must be RenderedPerturbation")
        if self.token_labels is not None and type(self.token_labels) is not TokenLabelSet:
            raise TypeError("PerturbationDraft token_labels must be TokenLabelSet or None")
        if type(self.reference_validation_report) is not ValidationReport:
            raise TypeError(
                "PerturbationDraft reference_validation_report must be ValidationReport"
            )
        if self.propagation.candidate_graph.schema != self.context.state_schema:
            raise ValueError("candidate_graph must use the context state_schema")
        if self.rendered.origin_id != self.context.record.origin_id:
            raise ValueError("rendered origin_id must match the context record")


@runtime_checkable
class CandidateEngine(Protocol[TruthT]):
    """Enumerate and deterministically select root-only candidate patches."""

    def enumerate_root_patches(
        self, context: PerturbationContext[TruthT]
    ) -> CandidatePool: ...

    def select_root_patch(
        self,
        context: PerturbationContext[TruthT],
        pool: CandidatePool,
    ) -> CandidatePatch: ...


@runtime_checkable
class PropagationEngine(Protocol[TruthT]):
    """Apply a selected root patch under the recipe's propagation policy."""

    def propagate(
        self,
        context: PerturbationContext[TruthT],
        root_patch: CandidatePatch,
    ) -> PropagationOutcome: ...


@runtime_checkable
class TraceRenderer(Protocol[TruthT]):
    """Build the trace representation and render exact text/character spans."""

    def render(
        self,
        context: PerturbationContext[TruthT],
        root_patch: CandidatePatch,
        propagation: PropagationOutcome,
    ) -> RenderedPerturbation: ...


@runtime_checkable
class LabelProjector(Protocol[TruthT]):
    """Project exact character annotations onto tokenizer-aligned labels."""

    def project(
        self,
        context: PerturbationContext[TruthT],
        root_patch: CandidatePatch,
        propagation: PropagationOutcome,
        rendered: RenderedPerturbation,
    ) -> TokenLabelSet | None: ...


@runtime_checkable
class ValidatorChain(Protocol[TruthT]):
    """Validate reference inputs first and the completed draft last."""

    def validate_reference(
        self, context: PerturbationContext[TruthT]
    ) -> ValidationReport: ...

    def validate_artifact(
        self, draft: PerturbationDraft[TruthT]
    ) -> ValidationReport: ...


def _sealed_template(method: Any) -> Any:
    """Mark the one canonical method before ``typing.final`` returns it."""

    setattr(method, "__molhallulens_sealed_template__", True)
    return final(method)


class _FinalTemplateMeta(ABCMeta):
    """Enforce finality against body, grandchild, and pre-MRO-mixin overrides."""

    _canonical_perturb_one: ClassVar[object | None] = None

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> _FinalTemplateMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        resolved = next(
            (
                ancestor.__dict__["perturb_one"]
                for ancestor in cls.__mro__
                if "perturb_one" in ancestor.__dict__
            ),
            None,
        )
        canonical = mcls._canonical_perturb_one
        if canonical is None and getattr(
            resolved, "__molhallulens_sealed_template__", False
        ):
            mcls._canonical_perturb_one = resolved
            canonical = resolved
        elif canonical is not None and resolved is not None and resolved is not canonical:
            raise FinalTemplateMethodError(
                f"{name} cannot override or shadow final Perturbator.perturb_one"
            )
        if canonical is not None and resolved is canonical:
            type.__setattr__(cls, "__molhallulens_template_sealed__", True)
        return cls

    def __setattr__(cls, name: str, value: Any) -> None:
        if name == "perturb_one" and getattr(
            cls, "__molhallulens_template_sealed__", False
        ):
            raise FinalTemplateMethodError(
                f"{cls.__name__}.perturb_one is a final Template Method"
            )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if name == "perturb_one" and getattr(
            cls, "__molhallulens_template_sealed__", False
        ):
            raise FinalTemplateMethodError(
                f"{cls.__name__}.perturb_one is a final Template Method"
            )
        super().__delattr__(name)


class Perturbator(ABC, Generic[TruthT], metaclass=_FinalTemplateMeta):
    """Family-level base whose sealed method owns the deterministic stage order."""

    family: ClassVar[str]
    subtask: ClassVar[str | None] = None

    __slots__ = (
        "_candidate_engine",
        "_propagator",
        "_renderer",
        "_validators",
        "_label_projector",
    )

    def __init__(
        self,
        *,
        candidate_engine: CandidateEngine[TruthT],
        propagator: PropagationEngine[TruthT],
        renderer: TraceRenderer[TruthT],
        validators: ValidatorChain[TruthT],
        label_projector: LabelProjector[TruthT],
    ) -> None:
        requirements = (
            (candidate_engine, CandidateEngine, "candidate_engine"),
            (propagator, PropagationEngine, "propagator"),
            (renderer, TraceRenderer, "renderer"),
            (validators, ValidatorChain, "validators"),
            (label_projector, LabelProjector, "label_projector"),
        )
        for value, protocol, name in requirements:
            if not isinstance(value, protocol):
                raise PerturbatorConfigurationError(
                    f"{name} does not implement the required {protocol.__name__} port"
                )
        self._candidate_engine = candidate_engine
        self._propagator = propagator
        self._renderer = renderer
        self._validators = validators
        self._label_projector = label_projector

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "perturb_one":
            raise FinalTemplateMethodError(
                f"{type(self).__name__}.perturb_one is a final Template Method"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "perturb_one":
            raise FinalTemplateMethodError(
                f"{type(self).__name__}.perturb_one is a final Template Method"
            )
        object.__delattr__(self, name)

    @property
    def candidate_engine(self) -> CandidateEngine[TruthT]:
        return self._candidate_engine

    @property
    def propagator(self) -> PropagationEngine[TruthT]:
        return self._propagator

    @property
    def renderer(self) -> TraceRenderer[TruthT]:
        return self._renderer

    @property
    def validators(self) -> ValidatorChain[TruthT]:
        return self._validators

    @property
    def label_projector(self) -> LabelProjector[TruthT]:
        return self._label_projector

    def parse_record(self, record: TaskRecord) -> TaskRecord:
        """Accept an already-normalized record and enforce the routed class identity."""

        if type(record) is not TaskRecord:
            raise TypeError("record must be a normalized TaskRecord")
        if record.family.value != self.family:
            raise ValueError("record family does not match perturbator family")
        if self.subtask is not None and record.normalized_subtask.value != self.subtask:
            raise ValueError("record normalized_subtask does not match perturbator subtask")
        return record

    def build_reference_dag(self, record: TaskRecord) -> StateDAG:
        """Future-family integration hook; editing overrides it with T011."""

        raise PerturbatorExecutionError(
            code="REFERENCE_DAG_BUILDER_UNAVAILABLE",
            stage=PerturbationStage.REFERENCE_BUILD,
            origin_id=record.origin_id,
            detail="a family reference-DAG integration hook has not been configured",
        )

    def derive_truth(self, record: TaskRecord, dag: StateDAG) -> TruthT:
        """Future-family integration hook; editing overrides it with T013."""

        raise PerturbatorExecutionError(
            code="TRUTH_DERIVER_UNAVAILABLE",
            stage=PerturbationStage.TRUTH_DERIVATION,
            origin_id=record.origin_id,
            detail="a family truth-derivation integration hook has not been configured",
        )

    @abstractmethod
    def state_schema(self) -> StateSchema:
        """Return the authoritative schema for this stable family/subtask class."""

    def _validate_truth(self, truth: TruthT, *, origin_id: str) -> None:
        """Optional family-level runtime check for a generic truth payload."""

    def _call_stage(
        self,
        stage: PerturbationStage,
        code: str,
        origin_id: str,
        operation: Any,
    ) -> Any:
        try:
            return operation()
        except PerturbatorExecutionError:
            raise
        except Exception as exc:
            raise PerturbatorExecutionError(
                code=code,
                stage=stage,
                origin_id=origin_id,
                detail=f"stage dependency raised {type(exc).__name__}",
                cause=exc,
            ) from exc

    def _require_report(
        self,
        report: object,
        *,
        stage: PerturbationStage,
        code: str,
        origin_id: str,
    ) -> ValidationReport:
        if type(report) is not ValidationReport:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=stage,
                origin_id=origin_id,
                detail="validator port did not return ValidationReport",
            )
        if not report.all_pass:
            raise PerturbatorExecutionError(
                code=code,
                stage=stage,
                origin_id=origin_id,
                detail="deterministic validation did not pass",
                report=report,
            )
        return report

    @_sealed_template
    def perturb_one(
        self,
        record: TaskRecord,
        recipe: PerturbationRecipe,
    ) -> PerturbationResult:
        """Run the fixed, non-overridable single-origin Template Method."""

        origin_id = getattr(record, "origin_id", "unknown")
        if type(record) is not TaskRecord:
            raise PerturbatorExecutionError(
                code="INVALID_NORMALIZED_RECORD",
                stage=PerturbationStage.INGEST,
                origin_id=origin_id if type(origin_id) is str and origin_id else "unknown",
                detail="record must be a normalized TaskRecord",
            )
        origin_id = record.origin_id
        if type(recipe) is not PerturbationRecipe:
            raise PerturbatorExecutionError(
                code="INVALID_RECIPE",
                stage=PerturbationStage.INGEST,
                origin_id=origin_id,
                detail="recipe must be PerturbationRecipe",
            )
        if recipe.origin_id != origin_id:
            raise PerturbatorExecutionError(
                code="RECORD_RECIPE_ORIGIN_MISMATCH",
                stage=PerturbationStage.INGEST,
                origin_id=origin_id,
                detail="recipe origin_id does not match record origin_id",
            )

        normalized_record = self._call_stage(
            PerturbationStage.INGEST,
            "RECORD_INGEST_FAILED",
            origin_id,
            lambda: self.parse_record(record),
        )
        if type(normalized_record) is not TaskRecord:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.INGEST,
                origin_id=origin_id,
                detail="parse_record did not return TaskRecord",
            )

        schema = self._call_stage(
            PerturbationStage.REFERENCE_BUILD,
            "STATE_SCHEMA_FAILED",
            origin_id,
            self.state_schema,
        )
        if type(schema) is not StateSchema:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=origin_id,
                detail="state_schema did not return StateSchema",
            )
        reference_graph = self._call_stage(
            PerturbationStage.REFERENCE_BUILD,
            "REFERENCE_DAG_BUILD_FAILED",
            origin_id,
            lambda: self.build_reference_dag(normalized_record),
        )
        if type(reference_graph) is not StateDAG or reference_graph.schema != schema:
            raise PerturbatorExecutionError(
                code="REFERENCE_DAG_CONTRACT_VIOLATION",
                stage=PerturbationStage.REFERENCE_BUILD,
                origin_id=origin_id,
                detail="reference DAG must be StateDAG using the authoritative schema",
            )
        truth = self._call_stage(
            PerturbationStage.TRUTH_DERIVATION,
            "TRUTH_DERIVATION_FAILED",
            origin_id,
            lambda: self.derive_truth(normalized_record, reference_graph),
        )
        self._validate_truth(truth, origin_id=origin_id)
        context = PerturbationContext(
            record=normalized_record,
            recipe=recipe,
            state_schema=schema,
            reference_graph=reference_graph,
            truth=truth,
        )

        reference_report = self._call_stage(
            PerturbationStage.REFERENCE_VALIDATION,
            "REFERENCE_VALIDATOR_FAILED",
            origin_id,
            lambda: self.validators.validate_reference(context),
        )
        reference_report = self._require_report(
            reference_report,
            stage=PerturbationStage.REFERENCE_VALIDATION,
            code="REFERENCE_VALIDATION_FAILED",
            origin_id=origin_id,
        )

        pool = self._call_stage(
            PerturbationStage.CANDIDATE_ENUMERATION,
            "CANDIDATE_ENUMERATION_FAILED",
            origin_id,
            lambda: self.candidate_engine.enumerate_root_patches(context),
        )
        if type(pool) is not CandidatePool:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.CANDIDATE_ENUMERATION,
                origin_id=origin_id,
                detail="candidate engine did not return CandidatePool",
            )
        if not pool.candidates:
            raise PerturbatorExecutionError(
                code="NO_CANDIDATE_ROOT_PATCH",
                stage=PerturbationStage.CANDIDATE_ENUMERATION,
                origin_id=origin_id,
                detail="candidate pool contains no admissible root patch",
            )
        selected = self._call_stage(
            PerturbationStage.CANDIDATE_SELECTION,
            "CANDIDATE_SELECTION_FAILED",
            origin_id,
            lambda: self.candidate_engine.select_root_patch(context, pool),
        )
        if type(selected) is not CandidatePatch:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.CANDIDATE_SELECTION,
                origin_id=origin_id,
                detail="candidate engine did not select CandidatePatch",
            )
        canonical_candidates = {
            candidate.candidate_id: candidate for candidate in pool.candidates
        }
        if (
            selected.candidate_id not in canonical_candidates
            or canonical_candidates[selected.candidate_id] != selected
        ):
            raise PerturbatorExecutionError(
                code="SELECTED_CANDIDATE_OUTSIDE_POOL",
                stage=PerturbationStage.CANDIDATE_SELECTION,
                origin_id=origin_id,
                detail="selected root patch is not an unchanged member of its pool",
            )
        if selected.root_node_id != recipe.target_node_id:
            raise PerturbatorExecutionError(
                code="SELECTED_CANDIDATE_TARGET_MISMATCH",
                stage=PerturbationStage.CANDIDATE_SELECTION,
                origin_id=origin_id,
                detail="selected root patch does not target the recipe node",
            )

        propagation = self._call_stage(
            PerturbationStage.PROPAGATION,
            "PROPAGATION_FAILED",
            origin_id,
            lambda: self.propagator.propagate(context, selected),
        )
        if type(propagation) is not PropagationOutcome:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.PROPAGATION,
                origin_id=origin_id,
                detail="propagator did not return PropagationOutcome",
            )
        if propagation.candidate_graph.schema != schema:
            raise PerturbatorExecutionError(
                code="PROPAGATION_SCHEMA_MISMATCH",
                stage=PerturbationStage.PROPAGATION,
                origin_id=origin_id,
                detail="candidate graph does not use the authoritative schema",
            )

        rendered = self._call_stage(
            PerturbationStage.RENDERING,
            "RENDERING_FAILED",
            origin_id,
            lambda: self.renderer.render(context, selected, propagation),
        )
        if type(rendered) is not RenderedPerturbation:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.RENDERING,
                origin_id=origin_id,
                detail="renderer did not return RenderedPerturbation",
            )
        if rendered.origin_id != origin_id:
            raise PerturbatorExecutionError(
                code="RENDERED_ORIGIN_MISMATCH",
                stage=PerturbationStage.RENDERING,
                origin_id=origin_id,
                detail="renderer changed the origin identity",
            )

        token_labels = self._call_stage(
            PerturbationStage.LABEL_PROJECTION,
            "LABEL_PROJECTION_FAILED",
            origin_id,
            lambda: self.label_projector.project(
                context, selected, propagation, rendered
            ),
        )
        if token_labels is not None and type(token_labels) is not TokenLabelSet:
            raise PerturbatorExecutionError(
                code="PORT_CONTRACT_VIOLATION",
                stage=PerturbationStage.LABEL_PROJECTION,
                origin_id=origin_id,
                detail="label projector did not return TokenLabelSet or None",
            )

        draft = PerturbationDraft(
            context=context,
            root_patch=selected,
            propagation=propagation,
            rendered=rendered,
            token_labels=token_labels,
            reference_validation_report=reference_report,
        )
        artifact_report = self._call_stage(
            PerturbationStage.ARTIFACT_VALIDATION,
            "ARTIFACT_VALIDATOR_FAILED",
            origin_id,
            lambda: self.validators.validate_artifact(draft),
        )
        artifact_report = self._require_report(
            artifact_report,
            stage=PerturbationStage.ARTIFACT_VALIDATION,
            code="ARTIFACT_VALIDATION_FAILED",
            origin_id=origin_id,
        )
        validation_report = ValidationReport.combine(
            "molhallulens.perturbator.pipeline.v1",
            (reference_report, artifact_report),
        )

        try:
            return PerturbationResult(
                record_id=rendered.record_id,
                origin_id=rendered.origin_id,
                leakage_group_id=rendered.leakage_group_id,
                bundle_id=rendered.bundle_id,
                pair_id=rendered.pair_id,
                matched_record_id=rendered.matched_record_id,
                variant_label=rendered.variant_label,
                policy=recipe.policy,
                detector_input=rendered.detector_input,
                serialized_text=rendered.serialized_text,
                serialized_text_sha256=rendered.serialized_text_sha256,
                reference_graph=reference_graph,
                candidate_graph=propagation.candidate_graph,
                graph_delta=propagation.graph_delta,
                char_annotations=rendered.char_annotations,
                token_labels=token_labels,
                trace_labels=rendered.trace_labels,
                validation_report=validation_report,
                provenance=rendered.provenance,
            )
        except Exception as exc:
            raise PerturbatorExecutionError(
                code="RESULT_CONSTRUCTION_FAILED",
                stage=PerturbationStage.RESULT_CONSTRUCTION,
                origin_id=origin_id,
                detail=f"immutable PerturbationResult rejected the draft ({type(exc).__name__})",
                cause=exc,
            ) from exc


__all__ = [
    "CandidateEngine",
    "FinalTemplateMethodError",
    "LabelProjector",
    "PerturbationContext",
    "PerturbationDraft",
    "PerturbationStage",
    "Perturbator",
    "PerturbatorConfigurationError",
    "PerturbatorExecutionError",
    "PropagationEngine",
    "PropagationOutcome",
    "RenderedPerturbation",
    "TraceRenderer",
    "ValidatorChain",
]
