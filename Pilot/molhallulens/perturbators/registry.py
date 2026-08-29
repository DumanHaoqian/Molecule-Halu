"""Typed, immutable operator metadata and compatibility registry.

The registry deliberately stops at root-patch contract validation.  Candidate
generation, ranking, propagation, rendering, and bundle scheduling belong to
later pipeline stages and are not performed here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from molhallulens.config.models import OperatorsConfig
from molhallulens.domain import (
    AnomalyClassification,
    CandidatePatch,
    CandidatePool,
    CandidateSourceType,
    EditErrorSubtype,
    EditingSubtask,
    EditKind,
    EditTruth,
    HallucinationType,
    OperatorCapability,
    OperatorSpec,
    PropagationPolicy,
    TaskFamily,
    Visibility,
    state_schema_for,
)

from .base import PerturbationContext, Perturbator
from .editing import MoleculeEditingPerturbator

_DECLARATION_ATTRIBUTE = "__molhallulens_operator_declaration__"
_MEMBER_MIXINS_ATTRIBUTE = "__molhallulens_operator_member_mixins__"
_RESERVED_CANDIDATE_METADATA_KEYS = frozenset(
    {
        "candidate_graph",
        "downstream_state",
        "graph_delta",
        "propagated_nodes",
        "rendered_text",
        "serialized_text",
        "trace_labels",
        "token_labels",
    }
)


class OperatorRegistryError(RuntimeError):
    """Structured fail-closed operator registration or invocation failure."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        operator_id: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("OperatorRegistryError code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("OperatorRegistryError detail must be non-empty text")
        if operator_id is not None and type(operator_id) is not str:
            raise TypeError("OperatorRegistryError operator_id must be text or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("OperatorRegistryError evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.operator_id = operator_id
        self.evidence = MappingProxyType(dict(evidence or {}))
        label = f" for {operator_id!r}" if operator_id is not None else ""
        super().__init__(f"{code}{label}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "operator_id": self.operator_id,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


class OperatorFallbackError(OperatorRegistryError):
    """Raised when explicit, phenotype-preserving fallback is exhausted."""


@dataclass(frozen=True, slots=True)
class _OperatorDeclaration:
    operator_family: str
    spec: OperatorSpec
    edit_subtypes: frozenset[EditErrorSubtype]
    required_capabilities: frozenset[OperatorCapability]


@dataclass(frozen=True, slots=True)
class OperatorRegistration:
    """One decorated member bound to an exact family/subtask owner."""

    perturbator_type: type[Perturbator[Any]]
    task_family: TaskFamily
    subtask: EditingSubtask
    operator_family: str
    method_name: str
    spec: OperatorSpec
    edit_subtypes: frozenset[EditErrorSubtype]
    required_capabilities: frozenset[OperatorCapability]

    def __post_init__(self) -> None:
        if not inspect.isclass(self.perturbator_type) or not issubclass(
            self.perturbator_type, Perturbator
        ):
            raise TypeError("perturbator_type must be a Perturbator type")
        if type(self.task_family) is not TaskFamily:
            raise TypeError("task_family must be TaskFamily")
        if type(self.subtask) is not EditingSubtask:
            raise TypeError("subtask must be EditingSubtask")
        for value, name in (
            (self.operator_family, "operator_family"),
            (self.method_name, "method_name"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.spec) is not OperatorSpec:
            raise TypeError("spec must be OperatorSpec")
        subtypes = frozenset(self.edit_subtypes)
        capabilities = frozenset(self.required_capabilities)
        if not subtypes or any(type(item) is not EditErrorSubtype for item in subtypes):
            raise TypeError("edit_subtypes must contain EditErrorSubtype values")
        if not capabilities or any(
            type(item) is not OperatorCapability for item in capabilities
        ):
            raise TypeError(
                "required_capabilities must contain OperatorCapability values"
            )
        if HallucinationType.UNVERIFIABLE in self.spec.hallucination_types:
            raise ValueError("UNVERIFIABLE cannot be an operator phenotype")
        object.__setattr__(self, "edit_subtypes", subtypes)
        object.__setattr__(self, "required_capabilities", capabilities)

    @property
    def operator_id(self) -> str:
        return self.spec.operator_id


@dataclass(frozen=True, slots=True)
class OperatorResolution:
    """A pre-invocation compatibility decision for one immutable context."""

    registration: OperatorRegistration
    classification: AnomalyClassification
    policy: PropagationPolicy
    candidate_source: CandidateSourceType
    target_node_id: str
    quota_buckets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """An explicit decision only; it never invokes or rewrites a recipe."""

    requested_operator_id: str
    selected_operator_id: str
    requested_operator_family: str
    selected_operator_family: str
    policy: PropagationPolicy
    candidate_source: CandidateSourceType
    quota_bucket: str
    attempted_operator_ids: tuple[str, ...]
    quota_deviation: bool
    target_change_required: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.requested_operator_id, "requested_operator_id"),
            (self.selected_operator_id, "selected_operator_id"),
            (self.requested_operator_family, "requested_operator_family"),
            (self.selected_operator_family, "selected_operator_family"),
            (self.quota_bucket, "quota_bucket"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"FallbackDecision {name} must be non-empty text")
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("FallbackDecision policy must be PropagationPolicy")
        if type(self.candidate_source) is not CandidateSourceType:
            raise TypeError("FallbackDecision candidate_source must be CandidateSourceType")
        attempted = tuple(self.attempted_operator_ids)
        if any(type(item) is not str or not item for item in attempted):
            raise TypeError("attempted_operator_ids must contain non-empty strings")
        if len(set(attempted)) != len(attempted):
            raise ValueError("attempted_operator_ids must be unique")
        if type(self.quota_deviation) is not bool:
            raise TypeError("quota_deviation must be bool")
        if type(self.target_change_required) is not bool:
            raise TypeError("target_change_required must be bool")
        object.__setattr__(self, "attempted_operator_ids", attempted)


DecoratedT = TypeVar("DecoratedT", bound=Callable[..., object])


def _exact_frozenset(
    values: Iterable[Any],
    *,
    name: str,
    member_type: type[Any],
) -> frozenset[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"operator {name} must be a non-string iterable")
    frozen = frozenset(values)
    if not frozen:
        raise ValueError(f"operator {name} cannot be empty")
    if any(type(item) is not member_type for item in frozen):
        raise TypeError(
            f"operator {name} must contain exact {member_type.__name__} values"
        )
    return frozen


def operator(
    *,
    operator_id: str,
    operator_family: str,
    root_fields: Iterable[str],
    supported_policies: Iterable[PropagationPolicy],
    supported_sources: Iterable[CandidateSourceType],
    hallucination_types: Iterable[HallucinationType],
    edit_subtypes: Iterable[EditErrorSubtype],
    required_capabilities: Iterable[OperatorCapability],
    diagnostic_only: bool = False,
) -> Callable[[DecoratedT], DecoratedT]:
    """Attach typed immutable metadata without mutating a global registry."""

    if type(operator_id) is not str or not operator_id:
        raise ValueError("operator_id must be non-empty text")
    if type(operator_family) is not str or not operator_family:
        raise ValueError("operator_family must be non-empty text")
    if type(diagnostic_only) is not bool:
        raise TypeError("diagnostic_only must be bool")
    roots = _exact_frozenset(root_fields, name="root_fields", member_type=str)
    if any(not item for item in roots):
        raise ValueError("root_fields must contain non-empty node IDs")
    policies = _exact_frozenset(
        supported_policies,
        name="supported_policies",
        member_type=PropagationPolicy,
    )
    sources = _exact_frozenset(
        supported_sources,
        name="supported_sources",
        member_type=CandidateSourceType,
    )
    hallucinations = _exact_frozenset(
        hallucination_types,
        name="hallucination_types",
        member_type=HallucinationType,
    )
    subtypes = _exact_frozenset(
        edit_subtypes,
        name="edit_subtypes",
        member_type=EditErrorSubtype,
    )
    capabilities = _exact_frozenset(
        required_capabilities,
        name="required_capabilities",
        member_type=OperatorCapability,
    )
    if HallucinationType.UNVERIFIABLE in hallucinations:
        raise ValueError("UNVERIFIABLE is diagnostic taxonomy, not an operator phenotype")
    declaration = _OperatorDeclaration(
        operator_family=operator_family,
        spec=OperatorSpec(
            operator_id=operator_id,
            root_fields=roots,
            supported_policies=policies,
            supported_sources=sources,
            hallucination_types=hallucinations,
            diagnostic_only=diagnostic_only,
        ),
        edit_subtypes=subtypes,
        required_capabilities=capabilities,
    )

    def decorate(member: DecoratedT) -> DecoratedT:
        if not inspect.isfunction(member):
            raise TypeError("@operator must decorate an instance method function")
        if hasattr(member, _DECLARATION_ATTRIBUTE):
            raise TypeError("an operator member cannot be decorated more than once")
        parameters = tuple(inspect.signature(member).parameters.values())
        if (
            len(parameters) != 2
            or parameters[0].name != "self"
            or parameters[1].name != "context"
            or any(
                parameter.kind
                is not inspect.Parameter.POSITIONAL_OR_KEYWORD
                for parameter in parameters
            )
            or any(parameter.default is not inspect.Parameter.empty for parameter in parameters)
        ):
            raise TypeError("operator members must have exact signature (self, context)")
        setattr(member, _DECLARATION_ATTRIBUTE, declaration)
        return member

    return decorate


@dataclass(frozen=True, slots=True)
class PerturbatorRegistry:
    """An immutable deterministic registry built by explicit discovery."""

    _registrations: Mapping[str, OperatorRegistration] = field(repr=False)
    _operators_config: OperatorsConfig = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._registrations, Mapping):
            raise TypeError("registrations must be a mapping")
        if type(self._operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")
        ordered = {
            operator_id: self._registrations[operator_id]
            for operator_id in sorted(self._registrations)
        }
        if any(
            type(operator_id) is not str
            or type(registration) is not OperatorRegistration
            or operator_id != registration.operator_id
            for operator_id, registration in ordered.items()
        ):
            raise TypeError("registry entries must map exact IDs to OperatorRegistration")
        for registration in ordered.values():
            self._validate_registration(registration, self._operators_config)
        object.__setattr__(self, "_registrations", MappingProxyType(ordered))

    @classmethod
    def from_perturbator_types(
        cls,
        perturbator_types: Iterable[type[Perturbator[Any]]],
        *,
        operators_config: OperatorsConfig,
    ) -> PerturbatorRegistry:
        if isinstance(perturbator_types, (str, bytes)) or not isinstance(
            perturbator_types, Iterable
        ):
            raise TypeError("perturbator_types must be a non-string iterable")
        if type(operators_config) is not OperatorsConfig:
            raise TypeError("operators_config must be OperatorsConfig")
        types = tuple(perturbator_types)
        if len(set(types)) != len(types):
            raise OperatorRegistryError(
                code="DUPLICATE_PERTURBATOR_TYPE",
                detail="perturbator_types contains a duplicate owner type",
            )
        discovered: dict[str, OperatorRegistration] = {}
        for perturbator_type in sorted(types, key=lambda item: (item.__module__, item.__qualname__)):
            if not inspect.isclass(perturbator_type) or not issubclass(
                perturbator_type, MoleculeEditingPerturbator
            ):
                raise OperatorRegistryError(
                    code="INVALID_OPERATOR_OWNER",
                    detail="T017 operators must belong to a molecule-editing perturbator type",
                )
            try:
                task_family = TaskFamily(perturbator_type.family)
                subtask = EditingSubtask(perturbator_type.subtask)
            except (TypeError, ValueError) as error:
                raise OperatorRegistryError(
                    code="INVALID_OPERATOR_OWNER",
                    detail="operator owner has no exact editing family/subtask identity",
                ) from error
            member_mixins = vars(perturbator_type).get(
                _MEMBER_MIXINS_ATTRIBUTE, ()
            )
            if type(member_mixins) is not tuple or any(
                not inspect.isclass(owner) for owner in member_mixins
            ):
                raise OperatorRegistryError(
                    code="INVALID_OPERATOR_OWNER",
                    detail="operator member mixins must be an exact tuple of classes",
                )
            member_owners = (perturbator_type, *member_mixins)
            visible_names = {
                name for owner in member_owners for name in vars(owner)
            }
            for method_name in sorted(visible_names):
                member = next(
                    vars(owner)[method_name]
                    for owner in member_owners
                    if method_name in vars(owner)
                )
                declaration = getattr(member, _DECLARATION_ATTRIBUTE, None)
                if declaration is None:
                    continue
                if type(declaration) is not _OperatorDeclaration:
                    raise OperatorRegistryError(
                        code="INVALID_OPERATOR_METADATA",
                        detail="decorated member metadata has been altered",
                    )
                registration = OperatorRegistration(
                    perturbator_type=perturbator_type,
                    task_family=task_family,
                    subtask=subtask,
                    operator_family=declaration.operator_family,
                    method_name=method_name,
                    spec=declaration.spec,
                    edit_subtypes=declaration.edit_subtypes,
                    required_capabilities=declaration.required_capabilities,
                )
                cls._validate_registration(registration, operators_config)
                prior = discovered.get(registration.operator_id)
                if prior is not None:
                    raise OperatorRegistryError(
                        code="DUPLICATE_OPERATOR_ID",
                        operator_id=registration.operator_id,
                        detail=(
                            "duplicate operator_id declared by "
                            f"{prior.perturbator_type.__qualname__}.{prior.method_name} and "
                            f"{perturbator_type.__qualname__}.{method_name}"
                        ),
                    )
                discovered[registration.operator_id] = registration
        return cls(discovered, operators_config)

    @staticmethod
    def _validate_registration(
        registration: OperatorRegistration,
        operators_config: OperatorsConfig,
    ) -> None:
        operator_id = registration.operator_id
        family_config = operators_config.families.get(registration.operator_family)
        if family_config is None:
            raise OperatorRegistryError(
                code="UNKNOWN_OPERATOR_FAMILY",
                operator_id=operator_id,
                detail="operator_family is absent from operators.yaml",
            )
        configured_policies = frozenset(
            PropagationPolicy.from_dataset_name(name)
            for name in family_config.supported_policies
        )
        configured_sources = frozenset(
            CandidateSourceType(name)
            for name in family_config.allowed_candidate_sources
        )
        if not registration.spec.supported_policies <= configured_policies:
            raise OperatorRegistryError(
                code="POLICY_CONFIG_EXPANSION",
                operator_id=operator_id,
                detail="operator policies expand beyond operators.yaml family compatibility",
            )
        if not registration.spec.supported_sources <= configured_sources:
            raise OperatorRegistryError(
                code="SOURCE_CONFIG_EXPANSION",
                operator_id=operator_id,
                detail="operator sources expand beyond operators.yaml family compatibility",
            )
        schema = state_schema_for(registration.subtask)
        if not issubclass(registration.perturbator_type, MoleculeEditingPerturbator):
            raise OperatorRegistryError(
                code="INVALID_OPERATOR_OWNER",
                operator_id=operator_id,
                detail="operator owner must be a molecule-editing perturbator type",
            )
        if (
            registration.task_family is not TaskFamily.MOLECULE_EDITING
            or registration.perturbator_type.family != registration.task_family.value
            or registration.perturbator_type.subtask != registration.subtask.value
        ):
            raise OperatorRegistryError(
                code="INVALID_OPERATOR_OWNER",
                operator_id=operator_id,
                detail="registration identity differs from its exact owner type",
            )
        member = inspect.getattr_static(
            registration.perturbator_type, registration.method_name, None
        )
        declaration = getattr(member, _DECLARATION_ATTRIBUTE, None)
        if type(declaration) is not _OperatorDeclaration or not (
            declaration.operator_family == registration.operator_family
            and declaration.spec == registration.spec
            and declaration.edit_subtypes == registration.edit_subtypes
            and declaration.required_capabilities == registration.required_capabilities
        ):
            raise OperatorRegistryError(
                code="INVALID_OPERATOR_BINDING",
                operator_id=operator_id,
                detail="registration does not match immutable decorated member metadata",
            )
        nodes = schema.nodes_by_id
        for root in registration.spec.root_fields:
            node = nodes.get(root)
            if node is None:
                raise OperatorRegistryError(
                    code="UNKNOWN_ROOT_NODE",
                    operator_id=operator_id,
                    detail=f"root {root!r} is not a canonical node ID for the subtask",
                )
            if not node.mutable or node.visibility is not Visibility.CANDIDATE_OUTPUT:
                raise OperatorRegistryError(
                    code="INADMISSIBLE_ROOT_NODE",
                    operator_id=operator_id,
                    detail=f"root {root!r} is not mutable candidate output",
                )
        terminal = PropagationPolicy.TERMINAL in registration.spec.supported_policies
        terminal_shape = (
            registration.operator_family == "final_answer_identity"
            or "final_answer" in registration.spec.root_fields
            or EditErrorSubtype.FINAL_ANSWER_IDENTITY in registration.edit_subtypes
            or OperatorCapability.TERMINAL_PERTURBATION
            in registration.required_capabilities
        )
        if (terminal or terminal_shape) and not (
                registration.operator_family == "final_answer_identity"
                and registration.spec.supported_policies
                == frozenset({PropagationPolicy.TERMINAL})
                and registration.spec.root_fields == frozenset({"final_answer"})
                and registration.edit_subtypes
                == frozenset({EditErrorSubtype.FINAL_ANSWER_IDENTITY})
                and registration.required_capabilities
                == frozenset({OperatorCapability.TERMINAL_PERTURBATION})
        ):
            raise OperatorRegistryError(
                code="INVALID_TERMINAL_OPERATOR",
                operator_id=operator_id,
                detail="terminal operator identity, root, subtype, policy, and capability must agree",
            )

    def registration(self, operator_id: str) -> OperatorRegistration:
        if type(operator_id) is not str:
            raise TypeError("operator_id must be a string")
        try:
            return self._registrations[operator_id]
        except KeyError as error:
            raise OperatorRegistryError(
                code="UNKNOWN_OPERATOR_ID",
                operator_id=operator_id,
                detail="operator_id is not present in this explicit registry",
            ) from error

    def registrations_for(
        self,
        *,
        task_family: TaskFamily | str | None = None,
        subtask: EditingSubtask | str | None = None,
        operator_family: str | None = None,
        policy: PropagationPolicy | None = None,
        candidate_source: CandidateSourceType | None = None,
    ) -> tuple[OperatorRegistration, ...]:
        if task_family is not None:
            if type(task_family) is str:
                try:
                    task_family = TaskFamily(task_family)
                except ValueError as error:
                    raise ValueError("unknown task_family") from error
            elif type(task_family) is not TaskFamily:
                raise TypeError("task_family must be TaskFamily, string, or None")
        if subtask is not None:
            if type(subtask) is str:
                try:
                    subtask = EditingSubtask(subtask)
                except ValueError as error:
                    raise ValueError("unknown editing subtask") from error
            elif type(subtask) is not EditingSubtask:
                raise TypeError("subtask must be EditingSubtask, string, or None")
        if operator_family is not None and (
            type(operator_family) is not str or not operator_family
        ):
            raise TypeError("operator_family must be non-empty text or None")
        if policy is not None and type(policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy or None")
        if candidate_source is not None and type(candidate_source) is not CandidateSourceType:
            raise TypeError("candidate_source must be CandidateSourceType or None")
        return tuple(
            registration
            for registration in self._registrations.values()
            if (task_family is None or registration.task_family is task_family)
            and (subtask is None or registration.subtask is subtask)
            and (
                operator_family is None
                or registration.operator_family == operator_family
            )
            and (policy is None or policy in registration.spec.supported_policies)
            and (
                candidate_source is None
                or candidate_source in registration.spec.supported_sources
            )
        )

    def quota_buckets_for(self, operator_id: str) -> tuple[str, ...]:
        registration = self.registration(operator_id)
        return tuple(
            bucket
            for bucket, families in self._operators_config.quota_bucket_mappings.items()
            if registration.operator_family in families
        )

    def resolve(
        self,
        perturbator: Perturbator[Any],
        context: PerturbationContext[Any],
    ) -> OperatorResolution:
        if not isinstance(perturbator, Perturbator):
            raise TypeError("perturbator must be a Perturbator")
        if not isinstance(context, PerturbationContext):
            raise TypeError("context must be PerturbationContext")
        registration = self.registration(context.recipe.operator_id)
        operator_id = registration.operator_id
        if type(perturbator) is not registration.perturbator_type:
            raise OperatorRegistryError(
                code="OPERATOR_OWNER_MISMATCH",
                operator_id=operator_id,
                detail="operator_id is not owned by the exact perturbator type",
            )
        if (
            context.record.family is not registration.task_family
            or context.record.normalized_subtask is not registration.subtask
        ):
            raise OperatorRegistryError(
                code="OPERATOR_TASK_MISMATCH",
                operator_id=operator_id,
                detail="record family/subtask does not match operator registration",
            )
        if context.recipe.target_node_id not in registration.spec.root_fields:
            raise OperatorRegistryError(
                code="INCOMPATIBLE_ROOT",
                operator_id=operator_id,
                detail="recipe target is not an operator root field",
            )
        if context.recipe.candidate_source_mode not in registration.spec.supported_sources:
            raise OperatorRegistryError(
                code="INCOMPATIBLE_SOURCE",
                operator_id=operator_id,
                detail="recipe candidate source is not supported",
            )
        if context.recipe.policy not in registration.spec.supported_policies:
            raise OperatorRegistryError(
                code="INCOMPATIBLE_POLICY",
                operator_id=operator_id,
                detail="recipe propagation policy is not supported",
            )
        if type(context.truth) is not EditTruth:
            raise OperatorRegistryError(
                code="INVALID_EDIT_TRUTH",
                operator_id=operator_id,
                detail="editing operator resolution requires exact EditTruth",
            )
        from molhallulens.builders.anomaly_registry import classify_edit_truth

        try:
            classification = classify_edit_truth(context.truth)
        except Exception as error:
            raise OperatorRegistryError(
                code="ANOMALY_CLASSIFICATION_FAILED",
                operator_id=operator_id,
                detail=f"classification raised {type(error).__name__}",
            ) from error
        forbidden = tuple(
            sorted(
                (
                    capability
                    for capability in registration.required_capabilities
                    if not classification.allows(capability)
                ),
                key=lambda item: item.value,
            )
        )
        if forbidden:
            raise OperatorRegistryError(
                code="OPERATOR_CAPABILITY_FORBIDDEN",
                operator_id=operator_id,
                detail="origin capability policy forbids this operator",
                evidence={"forbidden_capabilities": tuple(item.value for item in forbidden)},
            )
        return OperatorResolution(
            registration=registration,
            classification=classification,
            policy=context.recipe.policy,
            candidate_source=context.recipe.candidate_source_mode,
            target_node_id=context.recipe.target_node_id,
            quota_buckets=self.quota_buckets_for(operator_id),
        )

    def invoke(
        self,
        perturbator: Perturbator[Any],
        context: PerturbationContext[Any],
    ) -> CandidatePool:
        resolution = self.resolve(perturbator, context)
        registration = resolution.registration
        operator_id = registration.operator_id
        member = getattr(perturbator, registration.method_name)
        try:
            pool = member(context)
        except OperatorRegistryError:
            raise
        except Exception as error:
            raise OperatorRegistryError(
                code="OPERATOR_INVOCATION_FAILED",
                operator_id=operator_id,
                detail=f"operator member raised {type(error).__name__}",
            ) from error
        if type(pool) is not CandidatePool:
            raise OperatorRegistryError(
                code="INVALID_OPERATOR_RETURN",
                operator_id=operator_id,
                detail="operator member must return exact CandidatePool",
            )
        for candidate in pool.candidates:
            self._validate_candidate(candidate, context, registration)
        return pool

    @staticmethod
    def _validate_candidate(
        candidate: CandidatePatch,
        context: PerturbationContext[Any],
        registration: OperatorRegistration,
    ) -> None:
        operator_id = registration.operator_id
        if candidate.root_node_id != context.recipe.target_node_id:
            raise OperatorRegistryError(
                code="CANDIDATE_ROOT_MISMATCH",
                operator_id=operator_id,
                detail="candidate must contain only the recipe root patch",
            )
        if candidate.root_node_id not in registration.spec.root_fields:
            raise OperatorRegistryError(
                code="CANDIDATE_ROOT_UNREGISTERED",
                operator_id=operator_id,
                detail="candidate root is absent from OperatorSpec",
            )
        reference = context.reference_graph.value_for(candidate.root_node_id)
        if candidate.old_value != reference:
            raise OperatorRegistryError(
                code="CANDIDATE_REFERENCE_MISMATCH",
                operator_id=operator_id,
                detail="candidate old_value is not the authoritative reference value",
            )
        recipe_source = context.recipe.candidate_source_mode
        source_compatible = (
            candidate.source in registration.spec.supported_sources
            and (
                recipe_source is CandidateSourceType.HYBRID
                or candidate.source is recipe_source
            )
        )
        if not source_compatible:
            raise OperatorRegistryError(
                code="CANDIDATE_SOURCE_MISMATCH",
                operator_id=operator_id,
                detail="candidate source changes the recipe source mode",
            )
        if candidate.edit_action is not None:
            expected = {
                EditingSubtask.ADD: EditKind.ADDITION,
                EditingSubtask.DELETE: EditKind.DELETION,
                EditingSubtask.SUBSTITUTE: EditKind.SUBSTITUTION,
            }[registration.subtask]
            if candidate.edit_action.edit_kind is not expected:
                raise OperatorRegistryError(
                    code="CANDIDATE_EDIT_KIND_MISMATCH",
                    operator_id=operator_id,
                    detail="candidate EditAction changes the editing phenotype",
                )
        reserved: list[str] = []

        def validate_closed_payload(
            value: Any,
            *,
            path: str,
            require_string_keys: bool,
        ) -> None:
            if type(value) in {type(None), bool, int, float, str}:
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if require_string_keys and type(key) is not str:
                        raise OperatorRegistryError(
                            code="CANDIDATE_DOWNSTREAM_PAYLOAD",
                            operator_id=operator_id,
                            detail="candidate metadata keys must be exact strings",
                            evidence={"payload_path": path},
                        )
                    if type(key) not in {bool, int, float, str}:
                        raise OperatorRegistryError(
                            code="CANDIDATE_DOWNSTREAM_PAYLOAD",
                            operator_id=operator_id,
                            detail="candidate structured claim keys must be scalar",
                            evidence={
                                "payload_path": path,
                                "payload_type": type(key).__qualname__,
                            },
                        )
                    key_text = str(key)
                    child_path = f"{path}.{key_text}" if path else key_text
                    if type(key) is str:
                        normalized = (
                            key.casefold().replace("-", "_").replace(" ", "_")
                        )
                        if normalized in _RESERVED_CANDIDATE_METADATA_KEYS:
                            reserved.append(child_path)
                    validate_closed_payload(
                        item,
                        path=child_path,
                        require_string_keys=require_string_keys,
                    )
                return
            if isinstance(value, (tuple, frozenset)):
                for index, item in enumerate(value):
                    validate_closed_payload(
                        item,
                        path=f"{path}[{index}]",
                        require_string_keys=require_string_keys,
                    )
                return
            raise OperatorRegistryError(
                code="CANDIDATE_DOWNSTREAM_PAYLOAD",
                operator_id=operator_id,
                detail="candidate metadata must use the closed scalar/container schema",
                evidence={"payload_path": path, "payload_type": type(value).__qualname__},
            )

        validate_closed_payload(
            candidate.metadata,
            path="metadata",
            require_string_keys=True,
        )
        if candidate.edit_action is not None:
            validate_closed_payload(
                candidate.edit_action.metadata,
                path="edit_action.metadata",
                require_string_keys=True,
            )
        validate_closed_payload(
            candidate.new_value.raw_value,
            path="new_value.raw_value",
            require_string_keys=False,
        )
        validate_closed_payload(
            candidate.new_value.normalized_value,
            path="new_value.normalized_value",
            require_string_keys=False,
        )
        if reserved:
            raise OperatorRegistryError(
                code="CANDIDATE_DOWNSTREAM_PAYLOAD",
                operator_id=operator_id,
                detail="candidate metadata contains downstream state",
                evidence={"reserved_keys": tuple(sorted(reserved))},
            )

    def decide_fallback(
        self,
        resolution: OperatorResolution,
        *,
        quota_bucket: str,
        attempted_operator_ids: Iterable[str],
        allow_quota_deviation: bool = False,
    ) -> FallbackDecision:
        if type(resolution) is not OperatorResolution:
            raise TypeError("resolution must be OperatorResolution")
        if type(quota_bucket) is not str or not quota_bucket:
            raise ValueError("quota_bucket must be non-empty text")
        if isinstance(attempted_operator_ids, (str, bytes)) or not isinstance(
            attempted_operator_ids, Iterable
        ):
            raise TypeError("attempted_operator_ids must be a non-string iterable")
        attempted = tuple(attempted_operator_ids)
        if any(type(item) is not str or not item for item in attempted):
            raise TypeError("attempted_operator_ids must contain non-empty strings")
        if len(set(attempted)) != len(attempted):
            raise ValueError("attempted_operator_ids must be unique")
        if type(allow_quota_deviation) is not bool:
            raise TypeError("allow_quota_deviation must be bool")
        mapped_families = self._operators_config.quota_bucket_mappings.get(quota_bucket)
        if mapped_families is None:
            raise OperatorFallbackError(
                code="UNKNOWN_QUOTA_BUCKET",
                operator_id=resolution.registration.operator_id,
                detail="quota bucket has no configured operator-family mapping",
            )
        requested = resolution.registration
        if quota_bucket not in resolution.quota_buckets:
            raise OperatorFallbackError(
                code="INCOMPATIBLE_QUOTA_BUCKET",
                operator_id=requested.operator_id,
                detail="quota bucket does not map to the requested operator family",
                evidence={"admissible_quota_buckets": resolution.quota_buckets},
            )
        attempted_set = set(attempted)
        if requested.operator_id not in attempted_set:
            raise OperatorFallbackError(
                code="REQUESTED_OPERATOR_NOT_ATTEMPTED",
                operator_id=requested.operator_id,
                detail="fallback requires an explicit failed attempt of the requested operator",
            )
        for attempted_id in attempted:
            try:
                attempted_registration = self.registration(attempted_id)
            except OperatorRegistryError as error:
                raise OperatorFallbackError(
                    code="INVALID_ATTEMPTED_OPERATOR",
                    operator_id=requested.operator_id,
                    detail="attempted fallback history contains an unknown operator",
                    evidence={"invalid_operator_id": attempted_id},
                ) from error
            related_family = (
                attempted_registration.operator_family == requested.operator_family
                or attempted_registration.operator_family in mapped_families
            )
            if not (
                attempted_registration.task_family is requested.task_family
                and attempted_registration.subtask is requested.subtask
                and resolution.policy in attempted_registration.spec.supported_policies
                and resolution.candidate_source
                in attempted_registration.spec.supported_sources
                and related_family
            ):
                raise OperatorFallbackError(
                    code="UNRELATED_ATTEMPTED_OPERATOR",
                    operator_id=requested.operator_id,
                    detail="attempted fallback history changes task, policy, source, or quota phenotype",
                    evidence={"invalid_operator_id": attempted_id},
                )

        def compatible(registration: OperatorRegistration) -> bool:
            return (
                registration.operator_id not in attempted_set
                and registration.task_family is requested.task_family
                and registration.subtask is requested.subtask
                and resolution.policy in registration.spec.supported_policies
                and resolution.candidate_source in registration.spec.supported_sources
                and all(
                    resolution.classification.allows(capability)
                    for capability in registration.required_capabilities
                )
            )

        same_family = tuple(
            registration
            for registration in self._registrations.values()
            if compatible(registration)
            and registration.operator_family == requested.operator_family
        )
        selected: OperatorRegistration | None = same_family[0] if same_family else None
        quota_deviation = False
        if selected is None and allow_quota_deviation:
            cross_family = tuple(
                registration
                for registration in self._registrations.values()
                if compatible(registration)
                and registration.operator_family != requested.operator_family
                and registration.operator_family in mapped_families
            )
            if cross_family:
                selected = cross_family[0]
                quota_deviation = True
        if selected is None:
            eligible_ids = tuple(
                registration.operator_id
                for registration in self._registrations.values()
                if compatible(registration)
                and (
                    registration.operator_family == requested.operator_family
                    or (
                        allow_quota_deviation
                        and registration.operator_family in mapped_families
                    )
                )
            )
            raise OperatorFallbackError(
                code="BACKFILL_REQUIRED",
                operator_id=requested.operator_id,
                detail="no explicit same-policy/source fallback remains",
                evidence={
                    "quota_bucket": quota_bucket,
                    "eligible_operator_ids": eligible_ids,
                    "attempted_operator_ids": attempted,
                },
            )
        return FallbackDecision(
            requested_operator_id=requested.operator_id,
            selected_operator_id=selected.operator_id,
            requested_operator_family=requested.operator_family,
            selected_operator_family=selected.operator_family,
            policy=resolution.policy,
            candidate_source=resolution.candidate_source,
            quota_bucket=quota_bucket,
            attempted_operator_ids=attempted,
            quota_deviation=quota_deviation,
            target_change_required=(
                resolution.target_node_id not in selected.spec.root_fields
            ),
        )


__all__ = [
    "FallbackDecision",
    "OperatorFallbackError",
    "OperatorRegistration",
    "OperatorRegistryError",
    "OperatorResolution",
    "PerturbatorRegistry",
    "operator",
]
