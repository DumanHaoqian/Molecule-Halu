"""T016 inheritance, composition, and normalized factory contracts."""

from __future__ import annotations

import copy
import hashlib
import inspect
from functools import lru_cache
from pathlib import Path

import pytest

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import build_reference_dag, derive_edit_truth
from molhallulens.domain import (
    BuildProvenance,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    ClaimValue,
    ComparatorKind,
    DetectorInput,
    EditKind,
    EditingSubtask,
    GraphDelta,
    NodeRole,
    OperationSubtype,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    StateDAG,
    StateNodeSpec,
    StateSchema,
    TaskFamily,
    TaskRecord,
    TraceLabels,
    ValidationReport,
    ValueProvenance,
    ValueType,
    VariantLabel,
    Visibility,
    state_schema_for,
)
from molhallulens.perturbators import (
    AdditionPerturbator,
    CandidateEngine,
    DeletionPerturbator,
    EDITING_REFERENCE_ENVELOPE_METADATA_KEY,
    EditingReferenceEnvelope,
    LabelProjector,
    MolecularOptimizationPerturbator,
    MoleculeEditingPerturbator,
    MoleculeUnderstandingPerturbator,
    Perturbator,
    PerturbatorExecutionError,
    PerturbatorFactory,
    PerturbationStage,
    PropagationEngine,
    PropagationOutcome,
    ReactionPredictionPerturbator,
    RenderedPerturbation,
    SubstitutionPerturbator,
    TraceRenderer,
    ValidatorChain,
    task_record_from_joined_input,
    task_record_from_validated_reference,
)
from molhallulens.validation import OriginValidationInput


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


class _CandidateEngine(CandidateEngine):
    def enumerate_root_patches(self, context):
        raise AssertionError("T016 hierarchy tests must not enumerate T017 candidates")

    def select_root_patch(self, context, pool):
        raise AssertionError("T016 hierarchy tests must not select T017 candidates")


class _PropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T016 hierarchy tests must not execute T022 propagation")


class _TraceRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T016 hierarchy tests must not render an artifact")


class _ValidatorChain(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T016 hierarchy tests must not execute the pipeline")

    def validate_artifact(self, draft):
        raise AssertionError("T016 hierarchy tests must not execute the pipeline")


class _LabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T016 hierarchy tests must not project labels")


def _ports() -> dict[str, object]:
    return {
        "candidate_engine": _CandidateEngine(),
        "propagator": _PropagationEngine(),
        "renderer": _TraceRenderer(),
        "validators": _ValidatorChain(),
        "label_projector": _LabelProjector(),
    }


def _record(subtask: EditingSubtask) -> TaskRecord:
    source_subtask = {
        EditingSubtask.ADD: "add_v2",
        EditingSubtask.DELETE: "delete_v2",
        EditingSubtask.SUBSTITUTE: "substitute_v2",
    }[subtask]
    return TaskRecord(
        origin_id=f"fixture-{subtask.value}",
        anonymous_sample_id=f"mol_edit.{source_subtask}.0000",
        family=TaskFamily.MOLECULE_EDITING,
        source_subtask=source_subtask,
        normalized_subtask=subtask,
        operation_subtype=OperationSubtype.STANDARD,
        indexed_smiles="[CH3:1][OH:2]",
        instruction="Apply the requested molecular edit.",
        gt_smiles="CO",
        reference_reasoning_chain="A frozen reference trace.",
        reference_final_answer="CO",
        parsed_reference_state={},
    )


@lru_cache(maxsize=None)
def _real_record(
    subtask: EditingSubtask,
) -> tuple[TaskRecord, OriginValidationInput, EditingReferenceEnvelope]:
    joined = next(
        record
        for record in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if f".{subtask.value}_v2." in record.anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    item = OriginValidationInput(
        record=joined,
        artifact=artifact,
        edit_truth=derive_edit_truth(artifact),
    )
    task_record = task_record_from_joined_input(joined)
    assert task_record == task_record_from_validated_reference(item)
    envelope = task_record.raw_metadata[EDITING_REFERENCE_ENVELOPE_METADATA_KEY]
    assert type(envelope) is EditingReferenceEnvelope
    return task_record, item, envelope


def test_hierarchy_has_three_concrete_editors_and_three_future_abstract_families() -> None:
    assert inspect.isabstract(Perturbator)
    assert inspect.isabstract(MoleculeEditingPerturbator)
    assert not inspect.isabstract(AdditionPerturbator)
    assert not inspect.isabstract(DeletionPerturbator)
    assert not inspect.isabstract(SubstitutionPerturbator)

    ports = _ports()
    for perturbator_type, subtask, edit_kind in (
        (AdditionPerturbator, EditingSubtask.ADD, EditKind.ADDITION),
        (DeletionPerturbator, EditingSubtask.DELETE, EditKind.DELETION),
        (SubstitutionPerturbator, EditingSubtask.SUBSTITUTE, EditKind.SUBSTITUTION),
    ):
        perturbator = perturbator_type(**ports)
        assert perturbator.family == "mol_edit"
        assert perturbator.subtask == subtask.value
        assert perturbator.normalized_subtask is subtask
        assert perturbator.expected_edit_kind() is edit_kind
        assert perturbator.state_schema() is state_schema_for(subtask)

    for future_family, family_name in (
        (MolecularOptimizationPerturbator, "mol_opt"),
        (MoleculeUnderstandingPerturbator, "mol_und"),
        (ReactionPredictionPerturbator, "rxn_pred"),
    ):
        assert inspect.isabstract(future_family)
        assert future_family.family == family_name
        with pytest.raises(TypeError):
            future_family(**_ports())


def test_template_method_rejects_direct_and_grandchild_overrides() -> None:
    with pytest.raises(TypeError, match="perturb_one"):

        class _DirectOverride(AdditionPerturbator):
            def perturb_one(self, record, recipe):
                raise AssertionError

    class _Intermediate(AdditionPerturbator):
        pass

    with pytest.raises(TypeError, match="perturb_one"):

        class _GrandchildOverride(_Intermediate):
            def perturb_one(self, record, recipe):
                raise AssertionError


def test_template_method_cannot_be_hijacked_by_multiple_inheritance() -> None:
    class _HijackMixin:
        def perturb_one(self, record, recipe):
            raise AssertionError

    with pytest.raises(TypeError, match="perturb_one"):

        class _UnsafeMRO(_HijackMixin, AdditionPerturbator):
            pass

    class _SafeMRO(AdditionPerturbator, _HijackMixin):
        pass

    assert inspect.getattr_static(_SafeMRO, "perturb_one") is inspect.getattr_static(
        Perturbator,
        "perturb_one",
    )


def test_template_method_rejects_class_level_rebinding_and_deletion() -> None:
    class _Probe(AdditionPerturbator):
        pass

    with pytest.raises(TypeError, match="perturb_one"):
        _Probe.perturb_one = lambda self, record, recipe: None  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="perturb_one"):
        del _Probe.perturb_one

    assert inspect.getattr_static(_Probe, "perturb_one") is inspect.getattr_static(
        Perturbator,
        "perturb_one",
    )


def test_template_method_rejects_instance_shadowing_and_deletion() -> None:
    perturbator = AdditionPerturbator(**_ports())
    canonical = inspect.getattr_static(type(perturbator), "perturb_one")

    with pytest.raises(TypeError, match="perturb_one"):
        perturbator.perturb_one = (  # type: ignore[method-assign]
            lambda record, recipe: None
        )
    with pytest.raises(TypeError, match="perturb_one"):
        del perturbator.perturb_one

    assert inspect.getattr_static(type(perturbator), "perturb_one") is canonical
    assert "perturb_one" not in getattr(perturbator, "__dict__", {})


def test_all_five_composition_dependencies_are_exactly_injected_and_read_only() -> None:
    ports = _ports()
    perturbator = AdditionPerturbator(**ports)

    assert perturbator.candidate_engine is ports["candidate_engine"]
    assert perturbator.propagator is ports["propagator"]
    assert perturbator.renderer is ports["renderer"]
    assert perturbator.validators is ports["validators"]
    assert perturbator.label_projector is ports["label_projector"]

    with pytest.raises(AttributeError):
        perturbator.renderer = _TraceRenderer()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("subtask", "expected_type"),
    (
        (EditingSubtask.ADD, AdditionPerturbator),
        (EditingSubtask.DELETE, DeletionPerturbator),
        (EditingSubtask.SUBSTITUTE, SubstitutionPerturbator),
    ),
)
def test_factory_uses_typed_normalized_subtask_and_forwards_dependencies(
    subtask: EditingSubtask,
    expected_type: type[Perturbator],
) -> None:
    ports = _ports()
    record = _record(subtask)

    first = PerturbatorFactory.from_record(record, **ports)
    second = PerturbatorFactory.from_record(record, **ports)

    assert type(first) is expected_type
    assert type(second) is expected_type
    assert first is not second
    for name, port in ports.items():
        assert getattr(first, name) is port
        assert getattr(second, name) is port


def test_factory_does_not_reparse_a_valid_records_source_subtask_name() -> None:
    record = _record(EditingSubtask.ADD)
    object.__setattr__(record, "source_subtask", "add_non_registry_but_normalized")

    perturbator = PerturbatorFactory.from_record(
        record,
        **_ports(),
    )

    assert type(perturbator) is AdditionPerturbator


@pytest.mark.parametrize(
    ("subtask", "perturbator_type"),
    (
        (EditingSubtask.ADD, AdditionPerturbator),
        (EditingSubtask.DELETE, DeletionPerturbator),
        (EditingSubtask.SUBSTITUTE, SubstitutionPerturbator),
    ),
)
def test_concrete_reference_and_truth_hooks_delegate_to_real_builders(
    subtask: EditingSubtask,
    perturbator_type: type[Perturbator],
) -> None:
    record, item, envelope = _real_record(subtask)
    joined = item.record
    perturbator = perturbator_type(**_ports())

    reference_dag = perturbator.build_reference_dag(record)
    truth = perturbator.derive_truth(record, reference_dag)

    assert record.origin_id == record.anonymous_sample_id
    assert record.origin_id == joined.anonymous_sample_id
    assert record.origin_id != item.artifact.legacy_orig_id
    assert record.source_subtask == joined.pilot_subtask
    assert envelope.joined_input_record == joined
    assert envelope.validation_report.all_pass is True
    assert reference_dag == item.artifact.state_dag
    assert truth == item.edit_truth


def test_concrete_reference_hook_requires_validated_reference_envelope() -> None:
    perturbator = AdditionPerturbator(**_ports())

    with pytest.raises(PerturbatorExecutionError) as captured:
        perturbator.build_reference_dag(_record(EditingSubtask.ADD))

    assert captured.value.code == "EDITING_REFERENCE_ENVELOPE_MISSING"
    assert captured.value.stage is PerturbationStage.REFERENCE_BUILD


def test_factory_rejects_wrong_types_future_families_and_mismatched_records() -> None:
    ports = _ports()
    with pytest.raises(TypeError):
        PerturbatorFactory.from_record(  # type: ignore[arg-type]
            object(),
            **ports,
        )

    future_record = copy.copy(_record(EditingSubtask.ADD))
    object.__setattr__(future_record, "family", TaskFamily.MOLECULAR_OPTIMIZATION)
    with pytest.raises((LookupError, ValueError)):
        PerturbatorFactory.from_record(
            future_record,
            **ports,
        )

    mismatched = copy.copy(_record(EditingSubtask.ADD))
    object.__setattr__(mismatched, "source_subtask", "delete_v2")
    with pytest.raises(ValueError):
        PerturbatorFactory.from_record(
            mismatched,
            **ports,
        )

    invalid_subtask = copy.copy(_record(EditingSubtask.ADD))
    object.__setattr__(invalid_subtask, "normalized_subtask", "add")
    with pytest.raises((TypeError, ValueError)):
        PerturbatorFactory.from_record(
            invalid_subtask,
            **ports,
        )


@pytest.mark.parametrize(
    "field_name",
    ("candidate_engine", "propagator", "renderer", "validators", "label_projector"),
)
def test_dependency_bundle_rejects_missing_or_wrong_ports(field_name: str) -> None:
    missing = _ports()
    del missing[field_name]
    with pytest.raises(TypeError):
        AdditionPerturbator(**missing)

    wrong = _ports()
    wrong[field_name] = None
    with pytest.raises(TypeError):
        AdditionPerturbator(**wrong)


def test_template_stage_order_is_fixed_and_a_failure_stops_all_downstream_work() -> None:
    events: list[str] = []
    old_value = ClaimValue(
        raw_value="old",
        normalized_value="old",
        value_type=ValueType.STRING,
        provenance=ValueProvenance.REFERENCE,
    )
    new_value = ClaimValue(
        raw_value="new",
        normalized_value="new",
        value_type=ValueType.STRING,
        provenance=ValueProvenance.RULE,
    )
    schema = StateSchema(
        schema_id="fixture.t016.stage_order",
        version="1",
        nodes=(
            StateNodeSpec(
                node_id="root",
                value_type=ValueType.STRING,
                step_index=1,
                role=NodeRole.PRIMARY_CLAIM,
                visibility=Visibility.CANDIDATE_OUTPUT,
                mutable=True,
                comparator=ComparatorKind.EXACT,
                renderer_slot="root",
            ),
        ),
        edges=(),
    )
    reference_graph = StateDAG(schema=schema, values={"root": old_value})
    candidate_graph = StateDAG(schema=schema, values={"root": new_value})
    root_patch = CandidatePatch(
        candidate_id="fixture-candidate",
        root_node_id="root",
        old_value=old_value,
        new_value=new_value,
        edit_action=None,
        source=CandidateSourceType.RULE,
    )
    pool = CandidatePool(request_id="fixture-request", candidates=(root_patch,))
    propagation = PropagationOutcome(
        candidate_graph=candidate_graph,
        graph_delta=GraphDelta(()),
    )
    serialized_text = "fixture rendered text"
    rendered = RenderedPerturbation(
        record_id="fixture-record-H",
        origin_id="fixture-add",
        leakage_group_id="fixture-leakage",
        bundle_id="fixture-bundle",
        pair_id="fixture-pair",
        matched_record_id="fixture-record-N",
        variant_label=VariantLabel.HALLUCINATED,
        detector_input=DetectorInput(
            indexed_smiles="[CH3:1][OH:2]",
            instruction="Apply the requested molecular edit.",
            reasoning_chain="A rendered trace.",
            final_answer="CO",
        ),
        serialized_text=serialized_text,
        serialized_text_sha256=hashlib.sha256(
            serialized_text.encode("utf-8")
        ).hexdigest(),
        char_annotations=(),
        trace_labels=TraceLabels(
            hallucination_present=True,
            reasoning_valid=False,
            answer_correct=True,
            chemically_valid=True,
            constraint_satisfied=True,
            format_valid=True,
            answer_complete=True,
        ),
        provenance=BuildProvenance(
            provider="fixture",
            transport=None,
            requested_model_id=None,
            response_model=None,
            model_catalog_entry_sha256=None,
        ),
    )
    passing_report = ValidationReport("fixture.pass", ())

    class _OrderedCandidateEngine:
        def enumerate_root_patches(self, context):
            events.append("enumerate")
            return pool

        def select_root_patch(self, context, candidate_pool):
            events.append("select")
            assert candidate_pool is pool
            return root_patch

    class _OrderedPropagationEngine:
        def propagate(self, context, selected):
            events.append("propagate")
            assert selected is root_patch
            return propagation

    class _OrderedRenderer:
        def render(self, context, selected, outcome):
            events.append("render")
            assert selected is root_patch
            assert outcome is propagation
            return rendered

    class _FailingLabelProjector:
        def project(self, context, selected, outcome, rendered_output):
            events.append("labels")
            raise RuntimeError("controlled label-stage failure")

    class _OrderedValidators:
        def validate_reference(self, context):
            events.append("validate_reference")
            return passing_report

        def validate_artifact(self, draft):
            events.append("validate_artifact")
            return passing_report

    class _OrderedPerturbator(Perturbator[object]):
        family = "mol_edit"
        subtask = "add"

        def parse_record(self, record):
            events.append("parse")
            return super().parse_record(record)

        def state_schema(self):
            events.append("schema")
            return schema

        def build_reference_dag(self, record):
            events.append("build_reference")
            return reference_graph

        def derive_truth(self, record, dag):
            events.append("derive_truth")
            assert dag is reference_graph
            return object()

    ports = {
        "candidate_engine": _OrderedCandidateEngine(),
        "propagator": _OrderedPropagationEngine(),
        "renderer": _OrderedRenderer(),
        "validators": _OrderedValidators(),
        "label_projector": _FailingLabelProjector(),
    }
    record = _record(EditingSubtask.ADD)
    recipe = PerturbationRecipe(
        recipe_id="fixture-recipe",
        origin_id=record.origin_id,
        operator_id="fixture.operator",
        policy=PropagationPolicy.STOP,
        target_node_id="root",
        candidate_source_mode=CandidateSourceType.RULE,
        variant_index=0,
        derived_seed=0,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=0,
            length_bucket="fixture",
        ),
        candidate_difficulty_bucket="fixture",
        renderer_style_id="fixture",
    )

    with pytest.raises(PerturbatorExecutionError) as captured:
        _OrderedPerturbator(**ports).perturb_one(record, recipe)

    assert captured.value.code == "LABEL_PROJECTION_FAILED"
    assert captured.value.stage is PerturbationStage.LABEL_PROJECTION
    assert type(captured.value.cause) is RuntimeError
    assert events == [
        "parse",
        "schema",
        "build_reference",
        "derive_truth",
        "validate_reference",
        "enumerate",
        "select",
        "propagate",
        "render",
        "labels",
    ]
    assert "validate_artifact" not in events
