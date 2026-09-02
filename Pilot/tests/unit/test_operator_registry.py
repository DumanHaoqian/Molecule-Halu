"""T017 operator metadata, registry, compatibility, and fallback contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import pytest
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import build_reference_dag, derive_edit_truth
from molhallulens.config import load_config_bundle
from molhallulens.core import (
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    ClaimValue,
    EditAction,
    EditErrorSubtype,
    EditingSubtask,
    EditKind,
    HallucinationType,
    OperationSubtype,
    OperatorCapability,
    OperatorSpec,
    PerturbationRecipe,
    PropagationPolicy,
    RewriteBudget,
    TaskFamily,
    ValueProvenance,
    ValueType,
)
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    CandidateEngine,
    DeletionPerturbator,
    LabelProjector,
    OperatorFallbackError,
    OperatorRegistration,
    OperatorRegistryError,
    PerturbationContext,
    PerturbatorRegistry,
    PropagationEngine,
    SubstitutionPerturbator,
    TraceRenderer,
    ValidatorChain,
    operator,
    task_record_from_validated_reference,
)
from molhallulens.infrastructure.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
OPERATORS_CONFIG = load_config_bundle().operators


class _CandidateEngine(CandidateEngine):
    def enumerate_root_patches(self, context):
        raise AssertionError("T017 registry tests invoke decorated members directly")

    def select_root_patch(self, context, pool):
        raise AssertionError("T017 registry tests do not perform T018 selection")


class _PropagationEngine(PropagationEngine):
    def propagate(self, context, root_patch):
        raise AssertionError("T017 registry tests do not execute T022 propagation")


class _TraceRenderer(TraceRenderer):
    def render(self, context, root_patch, propagation):
        raise AssertionError("T017 registry tests do not render output")


class _ValidatorChain(ValidatorChain):
    def validate_reference(self, context):
        raise AssertionError("T017 registry tests do not execute the full pipeline")

    def validate_artifact(self, draft):
        raise AssertionError("T017 registry tests do not execute the full pipeline")


class _LabelProjector(LabelProjector):
    def project(self, context, root_patch, propagation, rendered):
        raise AssertionError("T017 registry tests do not project labels")


def _ports() -> dict[str, object]:
    return {
        "candidate_engine": _CandidateEngine(),
        "propagator": _PropagationEngine(),
        "renderer": _TraceRenderer(),
        "validators": _ValidatorChain(),
        "label_projector": _LabelProjector(),
    }


@lru_cache(maxsize=1)
def _joined_corpus() -> tuple[JoinedInputRecord, ...]:
    return ChemCoTMolEditAdapter().load(DATASET_ROOT)


def _origin_id(subtask: EditingSubtask) -> str:
    marker = f".{subtask.value}_v2."
    return next(
        record.anonymous_sample_id
        for record in _joined_corpus()
        if marker in record.anonymous_sample_id
    )


@cache
def _validated_input(anonymous_sample_id: str) -> OriginValidationInput:
    joined = next(
        record
        for record in _joined_corpus()
        if record.anonymous_sample_id == anonymous_sample_id
    )
    artifact = build_reference_dag(joined)
    return OriginValidationInput(
        record=joined,
        artifact=artifact,
        edit_truth=derive_edit_truth(artifact),
    )


def _recipe(
    item: OriginValidationInput,
    operator_id: str,
    *,
    target_node_id: str,
    policy: PropagationPolicy = PropagationPolicy.STOP,
    source: CandidateSourceType = CandidateSourceType.RULE,
) -> PerturbationRecipe:
    return PerturbationRecipe(
        recipe_id=f"recipe:{operator_id}:{policy.dataset_name}",
        origin_id=item.record.anonymous_sample_id,
        operator_id=operator_id,
        policy=policy,
        target_node_id=target_node_id,
        candidate_source_mode=source,
        variant_index=0,
        derived_seed=17,
        rewrite_budget=RewriteBudget(
            max_changed_claims=1,
            max_added_characters=32,
            length_bucket="fixture",
        ),
        candidate_difficulty_bucket="fixture",
        renderer_style_id="fixture",
        partial_cut_nodes=(
            frozenset({target_node_id})
            if policy is PropagationPolicy.PARTIAL
            else frozenset()
        ),
    )


def _context(
    anonymous_sample_id: str,
    operator_id: str,
    *,
    target_node_id: str,
    policy: PropagationPolicy = PropagationPolicy.STOP,
    source: CandidateSourceType = CandidateSourceType.RULE,
) -> PerturbationContext:
    item = _validated_input(anonymous_sample_id)
    record = task_record_from_validated_reference(item)
    return PerturbationContext(
        record=record,
        recipe=_recipe(
            item,
            operator_id,
            target_node_id=target_node_id,
            policy=policy,
            source=source,
        ),
        state_schema=item.artifact.state_dag.schema,
        reference_graph=item.artifact.state_dag,
        truth=item.edit_truth,
    )


def _changed_value(old: ClaimValue, *, provenance: ValueProvenance) -> ClaimValue:
    value = old.normalized_value
    if old.value_type in {ValueType.INTEGER, ValueType.ATOM_INDEX, ValueType.COUNT}:
        changed: Any = value + 1
    elif old.value_type is ValueType.FLOAT:
        changed = value + 1.0
    elif old.value_type is ValueType.BOOLEAN:
        changed = not value
    elif old.value_type is ValueType.ATOM_SET:
        changed = tuple(value) + (max(value, default=-1) + 1,)
    else:
        changed = f"{value}N"
    return ClaimValue(
        raw_value=changed,
        normalized_value=changed,
        value_type=old.value_type,
        provenance=provenance,
    )


def _candidate(
    context: PerturbationContext,
    *,
    root_node_id: str | None = None,
    old_value: ClaimValue | None = None,
    source: CandidateSourceType | None = None,
) -> CandidatePatch:
    root = root_node_id or context.recipe.target_node_id
    reference = context.reference_graph.values[root]
    before = old_value or reference
    return CandidatePatch(
        candidate_id=f"candidate:{context.recipe.operator_id}:{root}",
        root_node_id=root,
        old_value=before,
        new_value=_changed_value(before, provenance=ValueProvenance.RULE),
        edit_action=None,
        source=source or context.recipe.candidate_source_mode,
    )


def _pool(
    context: PerturbationContext,
    *,
    root_node_id: str | None = None,
    old_value: ClaimValue | None = None,
    source: CandidateSourceType | None = None,
) -> CandidatePool:
    return CandidatePool(
        request_id=f"request:{context.recipe.operator_id}",
        candidates=(
            _candidate(
                context,
                root_node_id=root_node_id,
                old_value=old_value,
                source=source,
            ),
        ),
    )


OperatorBehavior = Callable[[PerturbationContext], object]


def _metadata(
    operator_id: str,
    operator_family: str,
    root_fields: set[str],
    *,
    policies: set[PropagationPolicy] | None = None,
    sources: set[CandidateSourceType] | None = None,
    hallucination_types: set[HallucinationType] | None = None,
    edit_subtypes: set[EditErrorSubtype] | None = None,
    required_capabilities: set[OperatorCapability] | None = None,
    diagnostic_only: bool = False,
) -> dict[str, object]:
    return {
        "operator_id": operator_id,
        "operator_family": operator_family,
        "root_fields": root_fields,
        "supported_policies": policies or {PropagationPolicy.STOP},
        "supported_sources": sources or {CandidateSourceType.RULE},
        "hallucination_types": hallucination_types
        or {HallucinationType.CONTRADICTION},
        "edit_subtypes": edit_subtypes or {EditErrorSubtype.ANCHOR_GROUNDING},
        "required_capabilities": required_capabilities
        or {OperatorCapability.CLAIM_PERTURBATION},
        "diagnostic_only": diagnostic_only,
    }


def _operator_method(
    method_name: str,
    metadata: dict[str, object],
    behavior: OperatorBehavior,
) -> Callable[..., object]:
    def member(self, context):
        return behavior(context)

    member.__name__ = method_name
    return operator(**metadata)(member)  # type: ignore[arg-type,return-value]


def _perturbator_type(
    name: str,
    base: type,
    definitions: tuple[tuple[str, dict[str, object], OperatorBehavior], ...],
) -> type:
    namespace = {
        method_name: _operator_method(method_name, metadata, behavior)
        for method_name, metadata, behavior in definitions
    }
    namespace["__module__"] = __name__
    return type(name, (base,), namespace)


def _valid(root: str) -> OperatorBehavior:
    return lambda context: _pool(context, root_node_id=root)


_FAMILY_DEFINITIONS = (
    (
        "wrong_anchor",
        _metadata(
            "mol_edit.add.fixture_wrong_anchor",
            "wrong_anchor_site",
            {"anchor_idx"},
            policies={
                PropagationPolicy.STOP,
                PropagationPolicy.PARTIAL,
                PropagationPolicy.FULL_CF,
            },
            sources=set(CandidateSourceType),
        ),
        _valid("anchor_idx"),
    ),
    (
        "wrong_fragment",
        _metadata(
            "mol_edit.add.fixture_wrong_fragment",
            "wrong_fragment_group",
            {"add_fragment"},
            policies={
                PropagationPolicy.STOP,
                PropagationPolicy.PARTIAL,
                PropagationPolicy.FULL_CF,
            },
            sources=set(CandidateSourceType),
            edit_subtypes={EditErrorSubtype.ADD_FRAGMENT_IDENTIFICATION},
        ),
        _valid("add_fragment"),
    ),
    (
        "attachment",
        _metadata(
            "mol_edit.add.fixture_attachment",
            "attachment_bond_edit",
            {"product"},
            policies={
                PropagationPolicy.STOP,
                PropagationPolicy.PARTIAL,
                PropagationPolicy.FULL_CF,
            },
            sources=set(CandidateSourceType),
            edit_subtypes={EditErrorSubtype.ATTACHMENT_OR_BOND_EDIT},
        ),
        _valid("product"),
    ),
    (
        "numeric",
        _metadata(
            "mol_edit.add.fixture_numeric",
            "numeric_count_claim",
            {"heavy_delta"},
            policies={PropagationPolicy.STOP, PropagationPolicy.PARTIAL},
            sources={
                CandidateSourceType.RULE,
                CandidateSourceType.RDKIT,
                CandidateSourceType.HYBRID,
            },
            edit_subtypes={EditErrorSubtype.HEAVY_ATOM_COUNT},
        ),
        _valid("heavy_delta"),
    ),
    (
        "internal_relation",
        _metadata(
            "mol_edit.add.fixture_internal_relation",
            "nl_formal_internal_relation",
            {"anchor_element"},
            policies={PropagationPolicy.STOP, PropagationPolicy.PARTIAL},
            sources={
                CandidateSourceType.RULE,
                CandidateSourceType.LLM,
                CandidateSourceType.HYBRID,
            },
            hallucination_types={HallucinationType.REASONING_ERROR},
            edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
        ),
        _valid("anchor_element"),
    ),
    (
        "terminal",
        _metadata(
            "mol_edit.add.fixture_terminal",
            "final_answer_identity",
            {"final_answer"},
            policies={PropagationPolicy.TERMINAL},
            sources=set(CandidateSourceType),
            edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
            required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
        ),
        _valid("final_answer"),
    ),
)


_SixFamilyAddition = _perturbator_type(
    "_SixFamilyAddition",
    AdditionPerturbator,
    _FAMILY_DEFINITIONS,
)


def _registry(*types: type) -> PerturbatorRegistry:
    return PerturbatorRegistry.from_perturbator_types(
        types,
        operators_config=OPERATORS_CONFIG,
    )


def test_decorator_metadata_is_typed_frozen_and_keeps_member_binding() -> None:
    registry = _registry(_SixFamilyAddition)
    registration = registry.registration("mol_edit.add.fixture_wrong_anchor")

    assert type(registration) is OperatorRegistration
    assert type(registration.spec) is OperatorSpec
    assert registration.perturbator_type is _SixFamilyAddition
    assert registration.task_family == "mol_edit"
    assert registration.subtask == "add"
    assert registration.operator_family == "wrong_anchor_site"
    assert registration.method_name == "wrong_anchor"
    assert registration.spec.root_fields == frozenset({"anchor_idx"})
    assert registration.spec.supported_policies == frozenset(
        {
            PropagationPolicy.STOP,
            PropagationPolicy.PARTIAL,
            PropagationPolicy.FULL_CF,
        }
    )
    assert registration.edit_subtypes == frozenset(
        {EditErrorSubtype.ANCHOR_GROUNDING}
    )
    assert registration.required_capabilities == frozenset(
        {OperatorCapability.CLAIM_PERTURBATION}
    )

    context = _context(
        _origin_id(EditingSubtask.ADD),
        registration.operator_id,
        target_node_id="anchor_idx",
    )
    perturbator = _SixFamilyAddition(**_ports())
    assert type(perturbator.wrong_anchor(context)) is CandidatePool

    with pytest.raises(FrozenInstanceError):
        registration.method_name = "rebound"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registration.spec.operator_id = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        registration.spec.root_fields.add("oracle_gt")  # type: ignore[attr-defined]


def test_registry_is_exact_unique_deterministic_and_has_no_global_side_effects() -> None:
    unrelated = _perturbator_type(
        "_UnrelatedSubstitution",
        SubstitutionPerturbator,
        (
            (
                "substitute_anchor",
                _metadata(
                    "mol_edit.substitute.fixture_anchor",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                _valid("anchor_idx"),
            ),
        ),
    )
    forward = _registry(_SixFamilyAddition, unrelated)
    reverse = _registry(unrelated, _SixFamilyAddition)
    forward_ids = tuple(item.operator_id for item in forward.registrations_for())
    reverse_ids = tuple(item.operator_id for item in reverse.registrations_for())

    assert forward_ids == tuple(sorted(forward_ids))
    assert reverse_ids == forward_ids
    assert type(forward.registrations_for()) is tuple
    assert not hasattr(forward, "register")
    assert tuple(
        item.operator_id
        for item in _registry(_SixFamilyAddition).registrations_for()
    ) == tuple(
        sorted(metadata["operator_id"] for _, metadata, _ in _FAMILY_DEFINITIONS)
    )

    for unknown in (
        "MOL_EDIT.ADD.FIXTURE_WRONG_ANCHOR",
        " mol_edit.add.fixture_wrong_anchor",
        "mol_edit.add.fixture_wrong_anchor.json",
        "missing.operator",
    ):
        with pytest.raises(OperatorRegistryError) as captured:
            forward.registration(unknown)
        assert captured.value.operator_id == unknown
        assert captured.value.to_dict()["operator_id"] == unknown

    duplicate_a = _perturbator_type(
        "_DuplicateA",
        AdditionPerturbator,
        (
            (
                "first",
                _metadata(
                    "mol_edit.add.fixture_duplicate",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                _valid("anchor_idx"),
            ),
        ),
    )
    duplicate_b = _perturbator_type(
        "_DuplicateB",
        AdditionPerturbator,
        (
            (
                "second",
                _metadata(
                    "mol_edit.add.fixture_duplicate",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                _valid("anchor_idx"),
            ),
        ),
    )
    with pytest.raises(OperatorRegistryError, match="duplicate"):
        _registry(duplicate_a, duplicate_b)


def test_direct_registry_constructor_cannot_bypass_family_or_root_validation() -> None:
    valid = _registry(_SixFamilyAddition).registration(
        "mol_edit.add.fixture_wrong_anchor"
    )
    rogue_family = replace(valid, operator_family="not_in_operators_yaml")
    with pytest.raises(OperatorRegistryError) as captured:
        PerturbatorRegistry(
            {rogue_family.operator_id: rogue_family},
            OPERATORS_CONFIG,
        )
    assert captured.value.code == "UNKNOWN_OPERATOR_FAMILY"

    metadata = _metadata(
        "mol_edit.add.direct_build_only",
        "wrong_anchor_site",
        {"oracle_gt"},
    )
    build_only_type = _perturbator_type(
        "_DirectBuildOnlyAddition",
        AdditionPerturbator,
        (("build_only", metadata, _valid("oracle_gt")),),
    )
    build_only = OperatorRegistration(
        perturbator_type=build_only_type,
        task_family=TaskFamily.MOLECULE_EDITING,
        subtask=EditingSubtask.ADD,
        operator_family="wrong_anchor_site",
        method_name="build_only",
        spec=OperatorSpec(
            operator_id="mol_edit.add.direct_build_only",
            root_fields=frozenset({"oracle_gt"}),
            supported_policies=frozenset({PropagationPolicy.STOP}),
            supported_sources=frozenset({CandidateSourceType.RULE}),
            hallucination_types=frozenset({HallucinationType.CONTRADICTION}),
        ),
        edit_subtypes=frozenset({EditErrorSubtype.ANCHOR_GROUNDING}),
        required_capabilities=frozenset(
            {OperatorCapability.CLAIM_PERTURBATION}
        ),
    )
    with pytest.raises(OperatorRegistryError) as captured:
        PerturbatorRegistry(
            {build_only.operator_id: build_only},
            OPERATORS_CONFIG,
        )
    assert captured.value.code == "INADMISSIBLE_ROOT_NODE"


def test_all_six_config_families_accept_the_frozen_policy_source_contract() -> None:
    registry = _registry(_SixFamilyAddition)
    expected = {
        name: (
            tuple(family.supported_policies),
            tuple(family.allowed_candidate_sources),
        )
        for name, family in OPERATORS_CONFIG.families.items()
    }

    assert {
        registration.operator_family
        for registration in registry.registrations_for(task_family="mol_edit", subtask="add")
    } == set(expected)
    for registration in registry.registrations_for():
        policies, sources = expected[registration.operator_family]
        assert {
            policy.dataset_name for policy in registration.spec.supported_policies
        } == set(policies)
        assert {
            source.value for source in registration.spec.supported_sources
        } == set(sources)

    numeric = registry.registration("mol_edit.add.fixture_numeric")
    assert PropagationPolicy.STOP in numeric.spec.supported_policies
    assert PropagationPolicy.LOCAL is PropagationPolicy.STOP
    assert PropagationPolicy.STOP.dataset_name == "LOCAL"
    assert "heavy_ring_count_claim" in registry.quota_buckets_for(
        numeric.operator_id
    )


@pytest.mark.parametrize(
    ("name", "metadata"),
    (
        (
            "numeric_full",
            _metadata(
                "mol_edit.add.invalid_numeric_full",
                "numeric_count_claim",
                {"heavy_delta"},
                policies={PropagationPolicy.FULL_CF},
            ),
        ),
        (
            "numeric_llm",
            _metadata(
                "mol_edit.add.invalid_numeric_llm",
                "numeric_count_claim",
                {"heavy_delta"},
                sources={CandidateSourceType.LLM},
            ),
        ),
        (
            "relation_rdkit",
            _metadata(
                "mol_edit.add.invalid_relation_rdkit",
                "nl_formal_internal_relation",
                {"anchor_element"},
                sources={CandidateSourceType.RDKIT},
            ),
        ),
        (
            "terminal_local",
            _metadata(
                "mol_edit.add.invalid_terminal_local",
                "final_answer_identity",
                {"final_answer"},
                policies={PropagationPolicy.STOP},
                edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
                required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
            ),
        ),
        (
            "unknown_family",
            _metadata(
                "mol_edit.add.invalid_unknown_family",
                "not_in_operators_yaml",
                {"anchor_idx"},
            ),
        ),
    ),
)
def test_registry_rejects_policy_or_source_expansion_beyond_config(
    name: str,
    metadata: dict[str, object],
) -> None:
    invalid_type = _perturbator_type(
        f"_InvalidConfig_{name}",
        AdditionPerturbator,
        (
            (
                "invalid",
                metadata,
                _valid(next(iter(metadata["root_fields"]))),  # type: ignore[arg-type]
            ),
        ),
    )

    with pytest.raises(OperatorRegistryError):
        _registry(invalid_type)


@pytest.mark.parametrize("root", ("missing", "source", "instruction", "oracle_gt", "remove_group"))
def test_registry_rejects_unknown_immutable_build_only_and_cross_subtask_roots(
    root: str,
) -> None:
    invalid_type = _perturbator_type(
        f"_InvalidRoot_{root}",
        AdditionPerturbator,
        (
            (
                "invalid_root",
                _metadata(
                    f"mol_edit.add.invalid_root_{root}",
                    "wrong_anchor_site",
                    {root},
                ),
                _valid(root),
            ),
        ),
    )

    with pytest.raises(OperatorRegistryError):
        _registry(invalid_type)


def test_resolve_rejects_target_policy_and_source_before_member_invocation() -> None:
    calls = {"count": 0}

    def counted(context: PerturbationContext) -> CandidatePool:
        calls["count"] += 1
        return _pool(context, root_node_id="anchor_idx")

    probe_type = _perturbator_type(
        "_CompatibilityProbeAddition",
        AdditionPerturbator,
        (
            (
                "probe",
                _metadata(
                    "mol_edit.add.compatibility_probe",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                    policies={PropagationPolicy.STOP},
                    sources={CandidateSourceType.RULE},
                ),
                counted,
            ),
        ),
    )
    registry = _registry(probe_type)
    perturbator = probe_type(**_ports())
    origin = _origin_id(EditingSubtask.ADD)

    invalid_contexts = (
        _context(
            origin,
            "mol_edit.add.compatibility_probe",
            target_node_id="add_fragment",
        ),
        _context(
            origin,
            "mol_edit.add.compatibility_probe",
            target_node_id="anchor_idx",
            policy=PropagationPolicy.FULL_CF,
        ),
        _context(
            origin,
            "mol_edit.add.compatibility_probe",
            target_node_id="anchor_idx",
            source=CandidateSourceType.LLM,
        ),
    )
    for context in invalid_contexts:
        with pytest.raises(OperatorRegistryError):
            registry.resolve(perturbator, context)
    assert calls["count"] == 0


def test_invoke_accepts_one_root_patch_bound_to_reference_value() -> None:
    registry = _registry(_SixFamilyAddition)
    perturbator = _SixFamilyAddition(**_ports())
    context = _context(
        _origin_id(EditingSubtask.ADD),
        "mol_edit.add.fixture_wrong_anchor",
        target_node_id="anchor_idx",
        source=CandidateSourceType.RDKIT,
    )

    pool = registry.invoke(perturbator, context)

    assert type(pool) is CandidatePool
    assert len(pool.candidates) == 1
    patch = pool.candidates[0]
    assert patch.root_node_id == context.recipe.target_node_id
    assert patch.old_value == context.reference_graph.values[patch.root_node_id]
    assert patch.source is context.recipe.candidate_source_mode


@pytest.mark.parametrize(
    ("case", "behavior", "metadata"),
    (
        (
            "wrong_root",
            lambda context: _pool(context, root_node_id="add_fragment"),
            _metadata(
                "mol_edit.add.contract_wrong_root",
                "wrong_anchor_site",
                {"anchor_idx"},
            ),
        ),
        (
            "wrong_old",
            lambda context: _pool(
                context,
                old_value=_changed_value(
                    context.reference_graph.values["anchor_idx"],
                    provenance=ValueProvenance.RULE,
                ),
            ),
            _metadata(
                "mol_edit.add.contract_wrong_old",
                "wrong_anchor_site",
                {"anchor_idx"},
            ),
        ),
        (
            "wrong_source",
            lambda context: _pool(context, source=CandidateSourceType.RDKIT),
            _metadata(
                "mol_edit.add.contract_wrong_source",
                "wrong_anchor_site",
                {"anchor_idx"},
                sources={CandidateSourceType.RULE},
            ),
        ),
        (
            "not_pool",
            lambda context: _candidate(context),
            _metadata(
                "mol_edit.add.contract_not_pool",
                "wrong_anchor_site",
                {"anchor_idx"},
            ),
        ),
    ),
)
def test_invoke_rejects_non_root_old_value_source_and_return_contracts(
    case: str,
    behavior: OperatorBehavior,
    metadata: dict[str, object],
) -> None:
    probe_type = _perturbator_type(
        f"_InvocationContract_{case}",
        AdditionPerturbator,
        (("probe", metadata, behavior),),
    )
    registry = _registry(probe_type)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        str(metadata["operator_id"]),
        target_node_id="anchor_idx",
    )

    with pytest.raises(OperatorRegistryError) as captured:
        registry.invoke(probe_type(**_ports()), context)

    assert captured.value.operator_id == metadata["operator_id"]
    assert captured.value.to_dict()["code"] == captured.value.code


@pytest.mark.parametrize(
    "metadata_factory",
    (
        lambda context: {"audit": {"Graph-Delta": "hidden downstream state"}},
        lambda context: {"audit": {"innocent_name": context.reference_graph}},
    ),
)
def test_invoke_recursively_rejects_renamed_keys_and_nested_state_payloads(
    metadata_factory: Callable[[PerturbationContext], dict[str, object]],
) -> None:
    operator_id = "mol_edit.add.contract_nested_metadata"

    def nested_payload(context: PerturbationContext) -> CandidatePool:
        patch = replace(
            _candidate(context),
            metadata=metadata_factory(context),
        )
        return CandidatePool(
            request_id=f"request:{operator_id}",
            candidates=(patch,),
        )

    probe_type = _perturbator_type(
        "_NestedMetadataAddition",
        AdditionPerturbator,
        (
            (
                "nested_payload",
                _metadata(
                    operator_id,
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                nested_payload,
            ),
        ),
    )
    registry = _registry(probe_type)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        operator_id,
        target_node_id="anchor_idx",
    )

    with pytest.raises(OperatorRegistryError) as captured:
        registry.invoke(probe_type(**_ports()), context)

    assert captured.value.code == "CANDIDATE_DOWNSTREAM_PAYLOAD"
    assert captured.value.operator_id == operator_id
    assert captured.value.evidence


def test_invoke_rejects_nested_state_in_edit_action_metadata() -> None:
    operator_id = "mol_edit.add.contract_edit_action_metadata"

    def nested_edit_action(context: PerturbationContext) -> CandidatePool:
        patch = replace(
            _candidate(context),
            edit_action=EditAction(
                edit_kind=EditKind.ADDITION,
                source_anchor_index=0,
                add_fragment_smiles="N",
                fragment_attachment_atom=0,
                metadata={"audit": {"payload": context.reference_graph}},
            ),
        )
        return CandidatePool(
            request_id=f"request:{operator_id}",
            candidates=(patch,),
        )

    probe_type = _perturbator_type(
        "_EditActionMetadataAddition",
        AdditionPerturbator,
        (
            (
                "nested_edit_action",
                _metadata(
                    operator_id,
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                nested_edit_action,
            ),
        ),
    )
    registry = _registry(probe_type)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        operator_id,
        target_node_id="anchor_idx",
    )

    with pytest.raises(OperatorRegistryError) as captured:
        registry.invoke(probe_type(**_ports()), context)

    assert captured.value.code == "CANDIDATE_DOWNSTREAM_PAYLOAD"
    assert captured.value.evidence["payload_path"].startswith(
        "edit_action.metadata"
    )


def test_invoke_rejects_nested_state_in_new_value_raw_payload() -> None:
    operator_id = "mol_edit.add.contract_new_value_raw"

    def nested_new_value(context: PerturbationContext) -> CandidatePool:
        patch = _candidate(context)
        changed = _changed_value(
            patch.old_value,
            provenance=ValueProvenance.RULE,
        )
        poisoned = ClaimValue(
            raw_value={"audit": {"payload": context.reference_graph}},
            normalized_value=changed.normalized_value,
            value_type=changed.value_type,
            provenance=changed.provenance,
        )
        return CandidatePool(
            request_id=f"request:{operator_id}",
            candidates=(replace(patch, new_value=poisoned),),
        )

    probe_type = _perturbator_type(
        "_NewValueRawAddition",
        AdditionPerturbator,
        (
            (
                "nested_new_value",
                _metadata(
                    operator_id,
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                nested_new_value,
            ),
        ),
    )
    registry = _registry(probe_type)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        operator_id,
        target_node_id="anchor_idx",
    )

    with pytest.raises(OperatorRegistryError) as captured:
        registry.invoke(probe_type(**_ports()), context)

    assert captured.value.code == "CANDIDATE_DOWNSTREAM_PAYLOAD"
    assert captured.value.evidence["payload_path"].startswith("new_value.raw_value")


_CapabilityDeletion = _perturbator_type(
    "_CapabilityDeletion",
    DeletionPerturbator,
    (
        (
            "claim",
            _metadata(
                "mol_edit.delete.capability_claim",
                "numeric_count_claim",
                {"heavy_delta"},
                edit_subtypes={EditErrorSubtype.HEAVY_ATOM_COUNT},
                required_capabilities={OperatorCapability.CLAIM_PERTURBATION},
            ),
            _valid("heavy_delta"),
        ),
        (
            "structural",
            _metadata(
                "mol_edit.delete.capability_structural",
                "wrong_fragment_group",
                {"remove_group_step1"},
                edit_subtypes={
                    EditErrorSubtype.REMOVE_OR_LEAVING_GROUP_IDENTIFICATION
                },
                required_capabilities={OperatorCapability.STRUCTURAL_DELETION},
            ),
            _valid("remove_group_step1"),
        ),
        (
            "remove_delta",
            _metadata(
                "mol_edit.delete.capability_remove_delta",
                "numeric_count_claim",
                {"heavy_delta"},
                edit_subtypes={EditErrorSubtype.HEAVY_ATOM_ARITHMETIC},
                required_capabilities={OperatorCapability.REMOVE_ONLY_DELTA_RULE},
            ),
            _valid("heavy_delta"),
        ),
        (
            "terminal",
            _metadata(
                "mol_edit.delete.capability_terminal",
                "final_answer_identity",
                {"final_answer"},
                policies={PropagationPolicy.TERMINAL},
                sources=set(CandidateSourceType),
                edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
                required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
            ),
            _valid("final_answer"),
        ),
    ),
)


def test_delete_with_replacement_capability_policy_is_checked_before_invocation() -> None:
    origin = "mol_edit.delete_v2.0081"
    registry = _registry(_CapabilityDeletion)
    perturbator = _CapabilityDeletion(**_ports())

    claim = registry.resolve(
        perturbator,
        _context(
            origin,
            "mol_edit.delete.capability_claim",
            target_node_id="heavy_delta",
        ),
    )
    terminal = registry.resolve(
        perturbator,
        _context(
            origin,
            "mol_edit.delete.capability_terminal",
            target_node_id="final_answer",
            policy=PropagationPolicy.TERMINAL,
        ),
    )
    assert claim.classification.operation_subtype is OperationSubtype.DELETE_WITH_REPLACEMENT
    assert terminal.classification.operation_subtype is OperationSubtype.DELETE_WITH_REPLACEMENT

    for operator_id, target in (
        ("mol_edit.delete.capability_structural", "remove_group_step1"),
        ("mol_edit.delete.capability_remove_delta", "heavy_delta"),
    ):
        with pytest.raises(OperatorRegistryError) as captured:
            registry.resolve(
                perturbator,
                _context(origin, operator_id, target_node_id=target),
            )
        assert captured.value.operator_id == operator_id
        assert "CAPABILITY" in captured.value.code


_FallbackAddition = _perturbator_type(
    "_FallbackAddition",
    AdditionPerturbator,
    (
        (
            "attachment_a",
            _metadata(
                "mol_edit.add.fallback_attachment_a",
                "attachment_bond_edit",
                {"add_fragment"},
            ),
            _valid("add_fragment"),
        ),
        (
            "attachment_b",
            _metadata(
                "mol_edit.add.fallback_attachment_b",
                "attachment_bond_edit",
                {"add_fragment"},
            ),
            _valid("add_fragment"),
        ),
        (
            "relation_c",
            _metadata(
                "mol_edit.add.fallback_relation_c",
                "nl_formal_internal_relation",
                {"anchor_element"},
                hallucination_types={HallucinationType.REASONING_ERROR},
                edit_subtypes={EditErrorSubtype.INTERNAL_INCONSISTENCY},
            ),
            _valid("anchor_element"),
        ),
        (
            "terminal_z",
            _metadata(
                "mol_edit.add.fallback_terminal_z",
                "final_answer_identity",
                {"final_answer"},
                policies={PropagationPolicy.TERMINAL},
                sources=set(CandidateSourceType),
                edit_subtypes={EditErrorSubtype.FINAL_ANSWER_IDENTITY},
                required_capabilities={OperatorCapability.TERMINAL_PERTURBATION},
            ),
            _valid("final_answer"),
        ),
    ),
)


def test_fallback_is_explicit_deterministic_and_never_changes_phenotype() -> None:
    registry = _registry(_FallbackAddition)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        "mol_edit.add.fallback_attachment_a",
        target_node_id="add_fragment",
    )
    resolution = registry.resolve(_FallbackAddition(**_ports()), context)

    decision = registry.decide_fallback(
        resolution,
        quota_bucket="attachment_internal_relation",
        attempted_operator_ids=("mol_edit.add.fallback_attachment_a",),
    )
    repeated = registry.decide_fallback(
        resolution,
        quota_bucket="attachment_internal_relation",
        attempted_operator_ids=("mol_edit.add.fallback_attachment_a",),
    )
    assert decision == repeated
    assert decision.requested_operator_id == "mol_edit.add.fallback_attachment_a"
    assert decision.selected_operator_id == "mol_edit.add.fallback_attachment_b"
    assert decision.requested_operator_family == "attachment_bond_edit"
    assert decision.selected_operator_family == "attachment_bond_edit"
    assert decision.policy is PropagationPolicy.STOP
    assert decision.candidate_source is CandidateSourceType.RULE
    assert decision.quota_deviation is False
    assert decision.target_change_required is False

    attempted_same_family = (
        "mol_edit.add.fallback_attachment_a",
        "mol_edit.add.fallback_attachment_b",
    )
    with pytest.raises(OperatorFallbackError):
        registry.decide_fallback(
            resolution,
            quota_bucket="attachment_internal_relation",
            attempted_operator_ids=attempted_same_family,
            allow_quota_deviation=False,
        )

    explicit_cross_family = registry.decide_fallback(
        resolution,
        quota_bucket="attachment_internal_relation",
        attempted_operator_ids=attempted_same_family,
        allow_quota_deviation=True,
    )
    assert explicit_cross_family.selected_operator_id == "mol_edit.add.fallback_relation_c"
    assert explicit_cross_family.selected_operator_family == "nl_formal_internal_relation"
    assert explicit_cross_family.policy is resolution.policy
    assert explicit_cross_family.candidate_source is resolution.candidate_source
    assert explicit_cross_family.quota_deviation is True
    assert explicit_cross_family.target_change_required is True

    with pytest.raises(OperatorFallbackError) as captured:
        registry.decide_fallback(
            resolution,
            quota_bucket="attachment_internal_relation",
            attempted_operator_ids=(
                *attempted_same_family,
                "mol_edit.add.fallback_relation_c",
            ),
            allow_quota_deviation=True,
        )
    assert captured.value.operator_id == resolution.registration.operator_id
    assert "mol_edit.add.fallback_terminal_z" not in captured.value.evidence.get(
        "eligible_operator_ids", ()
    )


def test_fallback_rejects_unrelated_bucket_and_requires_requested_attempt() -> None:
    registry = _registry(_FallbackAddition)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        "mol_edit.add.fallback_attachment_a",
        target_node_id="add_fragment",
    )
    resolution = registry.resolve(_FallbackAddition(**_ports()), context)

    with pytest.raises(OperatorFallbackError) as captured:
        registry.decide_fallback(
            resolution,
            quota_bucket="anchor_site_grounding",
            attempted_operator_ids=(resolution.registration.operator_id,),
        )
    assert captured.value.code == "INCOMPATIBLE_QUOTA_BUCKET"

    with pytest.raises(OperatorFallbackError) as captured:
        registry.decide_fallback(
            resolution,
            quota_bucket="attachment_internal_relation",
            attempted_operator_ids=(),
        )
    assert captured.value.code == "REQUESTED_OPERATOR_NOT_ATTEMPTED"


def test_operator_exception_is_preserved_and_does_not_implicitly_run_fallback() -> None:
    calls = {"fallback": 0}

    def explode(context: PerturbationContext) -> CandidatePool:
        raise RuntimeError("controlled operator failure")

    def fallback(context: PerturbationContext) -> CandidatePool:
        calls["fallback"] += 1
        return _pool(context, root_node_id="anchor_idx")

    probe_type = _perturbator_type(
        "_ExceptionFallbackAddition",
        AdditionPerturbator,
        (
            (
                "raises",
                _metadata(
                    "mol_edit.add.exception_primary",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                explode,
            ),
            (
                "fallback",
                _metadata(
                    "mol_edit.add.exception_fallback",
                    "wrong_anchor_site",
                    {"anchor_idx"},
                ),
                fallback,
            ),
        ),
    )
    registry = _registry(probe_type)
    context = _context(
        _origin_id(EditingSubtask.ADD),
        "mol_edit.add.exception_primary",
        target_node_id="anchor_idx",
    )

    with pytest.raises(OperatorRegistryError) as captured:
        registry.invoke(probe_type(**_ports()), context)

    assert captured.value.operator_id == "mol_edit.add.exception_primary"
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "controlled operator failure"
    assert calls["fallback"] == 0

    resolution = registry.resolve(probe_type(**_ports()), context)
    decision = registry.decide_fallback(
        resolution,
        quota_bucket="anchor_site_grounding",
        attempted_operator_ids=("mol_edit.add.exception_primary",),
    )
    assert decision.selected_operator_id == "mol_edit.add.exception_fallback"
    assert calls["fallback"] == 0
