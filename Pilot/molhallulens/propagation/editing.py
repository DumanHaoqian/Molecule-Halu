"""Strict deterministic propagation for the three molecule-editing DAGs."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

from rdkit import Chem, rdBase

from molhallulens.candidates import replay_edit_action_from_source
from molhallulens.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
)
from molhallulens.domain import (
    CandidatePatch,
    CausalRole,
    ClaimValue,
    EditErrorSubtype,
    EditKind,
    EditTruth,
    GraphDelta,
    HallucinationType,
    MutationEvent,
    MutationTargetKind,
    PropagationPolicy,
    StateDAG,
    ValueProvenance,
    ValueType,
    Visibility,
    state_schema_for,
)
from molhallulens.perturbators.base import PerturbationContext, PropagationOutcome

from .base import (
    DerivationContext,
    DerivationRuleRegistry,
    PropagationError,
    PropagationPlan,
    TypedDerivationRule,
)

_EDITING_SCHEMA_IDS = frozenset(
    {"mol_edit.add", "mol_edit.delete", "mol_edit.substitute"}
)


@dataclass(frozen=True, slots=True)
class _OperatorLabels:
    root_fields: frozenset[str]
    supported_policies: frozenset[PropagationPolicy]
    hallucination_types: frozenset[HallucinationType]
    edit_subtypes: frozenset[EditErrorSubtype]


@cache
def _operator_labels_by_id() -> Mapping[str, _OperatorLabels]:
    from molhallulens.perturbators.editing.addition import AdditionOperatorMixin
    from molhallulens.perturbators.editing.deletion import DeletionOperatorMixin
    from molhallulens.perturbators.editing.substitution import (
        SubstitutionOperatorMixin,
    )

    labels: dict[str, _OperatorLabels] = {}
    attribute = "__molhallulens_operator_declaration__"
    for mixin in (
        AdditionOperatorMixin,
        DeletionOperatorMixin,
        SubstitutionOperatorMixin,
    ):
        for member in vars(mixin).values():
            declaration = getattr(member, attribute, None)
            if declaration is None:
                continue
            operator_id = declaration.spec.operator_id
            if operator_id in labels:
                raise RuntimeError(f"duplicate operator declaration {operator_id!r}")
            labels[operator_id] = _OperatorLabels(
                root_fields=declaration.spec.root_fields,
                supported_policies=declaration.spec.supported_policies,
                hallucination_types=declaration.spec.hallucination_types,
                edit_subtypes=declaration.edit_subtypes,
            )
    return labels


def _propagated_claim(state: StateDAG, output_node: str, value: Any) -> ClaimValue:
    before = state.value_for(output_node)
    return replace(
        before,
        raw_value=value,
        normalized_value=value,
        provenance=ValueProvenance.PROPAGATED,
        locally_valid=True,
        oracle_match=False,
        confidence=1.0,
    )


def _text_value(state: StateDAG, node_id: str) -> str:
    value = state.value_for(node_id).normalized_value
    if type(value) is not str or not value:
        raise PropagationError(
            code="DERIVATION_INPUT_TYPE_MISMATCH",
            detail="derivation input is not non-empty molecular text",
            node_id=node_id,
        )
    return value


def _integer_value(state: StateDAG, node_id: str) -> int:
    value = state.value_for(node_id).normalized_value
    if type(value) is not int:
        raise PropagationError(
            code="DERIVATION_INPUT_TYPE_MISMATCH",
            detail="derivation input is not an integer",
            node_id=node_id,
        )
    return value


def _derive_anchor_element(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    source = _text_value(state, "source")
    anchor_map = _integer_value(state, "anchor_idx")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(source, sanitize=True)
    if molecule is None:
        raise PropagationError(
            code="DERIVATION_FAILED",
            detail="mapped source failed strict RDKit parsing",
            node_id="anchor_element",
        )
    atoms = tuple(
        atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == anchor_map
    )
    if len(atoms) != 1:
        raise PropagationError(
            code="CROSS_FIELD_MISMATCH",
            detail="anchor map is not unique in source",
            node_id="anchor_element",
            evidence={"anchor_map": anchor_map, "matches": len(atoms)},
        )
    return _propagated_claim(state, "anchor_element", atoms[0].GetSymbol())


def _descriptor_value(
    state: StateDAG,
    *,
    input_node: str,
    output_node: str,
    attribute: str,
) -> ClaimValue:
    descriptors = compute_descriptors(
        _text_value(state, input_node),
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    return _propagated_claim(state, output_node, getattr(descriptors, attribute))


def _derive_fragment_heavy(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="add_fragment",
        output_node="fragment_heavy",
        attribute="heavy_atom_count",
    )


def _derive_delete_remove_heavy(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="remove_group_step2",
        output_node="remove_heavy",
        attribute="heavy_atom_count",
    )


def _derive_substitute_remove_heavy(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="remove_group",
        output_node="remove_heavy",
        attribute="heavy_atom_count",
    )


def _derive_add_heavy(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="add_fragment",
        output_node="add_heavy",
        attribute="heavy_atom_count",
    )


def _derive_delete_remove_group_step2(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    value = _text_value(state, "remove_group_step1")
    canonical = canonicalize_smiles(
        value,
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    return _propagated_claim(state, "remove_group_step2", canonical)


def _derive_product(
    state: StateDAG,
    context: DerivationContext,
) -> ClaimValue:
    product = context.candidate_product_smiles
    if product is None:
        raise PropagationError(
            code="STRUCTURAL_ACTION_REQUIRED",
            detail="product derivation requires a replayed structural action",
            node_id="product",
        )
    return _propagated_claim(state, "product", product)


def _derive_product_heavy(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="product",
        output_node="product_heavy",
        attribute="heavy_atom_count",
    )


def _derive_product_rings(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    return _descriptor_value(
        state,
        input_node="product",
        output_node="product_rings",
        attribute="ring_count",
    )


def _derive_heavy_delta(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    value = _integer_value(state, "product_heavy") - _integer_value(
        state, "source_heavy"
    )
    return _propagated_claim(state, "heavy_delta", value)


def _derive_ring_delta(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    value = _integer_value(state, "product_rings") - _integer_value(
        state, "source_rings"
    )
    return _propagated_claim(state, "ring_delta", value)


def _derive_final_answer(
    state: StateDAG,
    _context: DerivationContext,
) -> ClaimValue:
    product = canonicalize_smiles(
        _text_value(state, "product"),
        fragment_policy=FragmentPolicy.KEEP_ALL,
    )
    return _propagated_claim(state, "final_answer", product)


def _rule(
    rule_id: str,
    output_node: str,
    input_nodes: tuple[str, ...],
    input_types: tuple[ValueType, ...],
    output_type: ValueType,
    derive_fn: Any,
    *,
    schema_ids: frozenset[str] = _EDITING_SCHEMA_IDS,
) -> TypedDerivationRule:
    return TypedDerivationRule(
        rule_id=rule_id,
        output_node=output_node,
        input_nodes=input_nodes,
        input_types=input_types,
        output_type=output_type,
        derive_fn=derive_fn,
        schema_ids=schema_ids,
    )


def editing_derivation_rule_registry() -> DerivationRuleRegistry:
    """Return the immutable authoritative T022 editing derivation registry."""

    add = frozenset({"mol_edit.add"})
    delete = frozenset({"mol_edit.delete"})
    substitute = frozenset({"mol_edit.substitute"})
    rules = (
        _rule(
            "editing.anchor_element",
            "anchor_element",
            ("source", "anchor_idx"),
            (ValueType.INDEXED_SMILES, ValueType.ATOM_INDEX),
            ValueType.ELEMENT,
            _derive_anchor_element,
        ),
        _rule(
            "addition.fragment_heavy",
            "fragment_heavy",
            ("add_fragment",),
            (ValueType.FRAGMENT,),
            ValueType.COUNT,
            _derive_fragment_heavy,
            schema_ids=add,
        ),
        _rule(
            "deletion.remove_group_step2",
            "remove_group_step2",
            ("remove_group_step1",),
            (ValueType.FRAGMENT,),
            ValueType.FRAGMENT,
            _derive_delete_remove_group_step2,
            schema_ids=delete,
        ),
        _rule(
            "deletion.remove_heavy",
            "remove_heavy",
            ("remove_group_step2",),
            (ValueType.FRAGMENT,),
            ValueType.COUNT,
            _derive_delete_remove_heavy,
            schema_ids=delete,
        ),
        _rule(
            "substitution.remove_heavy",
            "remove_heavy",
            ("remove_group",),
            (ValueType.FRAGMENT,),
            ValueType.COUNT,
            _derive_substitute_remove_heavy,
            schema_ids=substitute,
        ),
        _rule(
            "substitution.add_heavy",
            "add_heavy",
            ("add_fragment",),
            (ValueType.FRAGMENT,),
            ValueType.COUNT,
            _derive_add_heavy,
            schema_ids=substitute,
        ),
        _rule(
            "addition.product",
            "product",
            ("source", "anchor_idx", "leaving", "add_fragment"),
            (
                ValueType.INDEXED_SMILES,
                ValueType.ATOM_INDEX,
                ValueType.STRING,
                ValueType.FRAGMENT,
            ),
            ValueType.SMILES,
            _derive_product,
            schema_ids=add,
        ),
        _rule(
            "deletion.product",
            "product",
            ("source", "anchor_idx", "remove_group_step2"),
            (
                ValueType.INDEXED_SMILES,
                ValueType.ATOM_INDEX,
                ValueType.FRAGMENT,
            ),
            ValueType.SMILES,
            _derive_product,
            schema_ids=delete,
        ),
        _rule(
            "substitution.product",
            "product",
            ("source", "anchor_idx", "remove_group", "add_fragment"),
            (
                ValueType.INDEXED_SMILES,
                ValueType.ATOM_INDEX,
                ValueType.FRAGMENT,
                ValueType.FRAGMENT,
            ),
            ValueType.SMILES,
            _derive_product,
            schema_ids=substitute,
        ),
        _rule(
            "editing.product_heavy",
            "product_heavy",
            ("product",),
            (ValueType.SMILES,),
            ValueType.COUNT,
            _derive_product_heavy,
        ),
        _rule(
            "editing.heavy_delta",
            "heavy_delta",
            ("source_heavy", "product_heavy"),
            (ValueType.COUNT, ValueType.COUNT),
            ValueType.INTEGER,
            _derive_heavy_delta,
            schema_ids=add | delete,
        ),
        _rule(
            "substitution.heavy_delta",
            "heavy_delta",
            ("source_heavy", "product_heavy", "remove_heavy", "add_heavy"),
            (
                ValueType.COUNT,
                ValueType.COUNT,
                ValueType.COUNT,
                ValueType.COUNT,
            ),
            ValueType.INTEGER,
            _derive_heavy_delta,
            schema_ids=substitute,
        ),
        _rule(
            "editing.product_rings",
            "product_rings",
            ("product",),
            (ValueType.SMILES,),
            ValueType.COUNT,
            _derive_product_rings,
        ),
        _rule(
            "editing.ring_delta",
            "ring_delta",
            ("source_rings", "product_rings"),
            (ValueType.COUNT, ValueType.COUNT),
            ValueType.INTEGER,
            _derive_ring_delta,
        ),
        _rule(
            "editing.final_answer",
            "final_answer",
            ("product",),
            (ValueType.SMILES,),
            ValueType.SMILES,
            _derive_final_answer,
        ),
    )
    return DerivationRuleRegistry(rules)


DEFAULT_EDITING_DERIVATION_RULE_REGISTRY = editing_derivation_rule_registry()


def _mutable_closure(context: PerturbationContext[EditTruth]) -> tuple[str, ...]:
    schema = context.state_schema
    root = context.recipe.target_node_id
    closure = set(schema.dependency_closure((root,)))
    return tuple(
        node_id
        for node_id in schema.topological_order()
        if node_id in closure
        and (
            node_id == root
            or (
                schema.nodes_by_id[node_id].mutable
                and schema.nodes_by_id[node_id].visibility
                is Visibility.CANDIDATE_OUTPUT
            )
        )
    )


def _partial_selected_nodes(
    context: PerturbationContext[EditTruth],
    full_closure: tuple[str, ...],
    registry: DerivationRuleRegistry,
) -> tuple[str, ...]:
    schema = context.state_schema
    root = context.recipe.target_node_id
    cuts = context.recipe.partial_cut_nodes
    if not cuts:
        raise PropagationError(
            code="PARTIAL_NOT_STRICT",
            detail="PARTIAL without a cut is full-equivalent",
        )
    node_ids = set(schema.nodes_by_id)
    unknown = tuple(sorted(cuts - node_ids))
    if unknown:
        raise PropagationError(
            code="PARTIAL_CUT_UNKNOWN",
            detail="partial cut references unknown nodes",
            evidence={"unknown": unknown},
        )
    strict_descendants = set(full_closure) - {root}
    invalid = tuple(
        sorted(
            cut
            for cut in cuts
            if cut not in strict_descendants
            or not registry.has_rule(cut, schema_id=schema.schema_id)
        )
    )
    if invalid:
        raise PropagationError(
            code="PARTIAL_CUT_NOT_DESCENDANT",
            detail="partial cuts must be strict mutable derivable descendants",
            evidence={"invalid": invalid},
        )

    allowed = set(full_closure)
    outgoing: dict[str, tuple[str, ...]] = {
        node_id: tuple(
            sorted(
                edge.target
                for edge in schema.edges
                if edge.source == node_id and edge.target in allowed
            )
        )
        for node_id in allowed
    }
    selected = {root}
    frontier = deque([root])
    while frontier:
        node_id = frontier.popleft()
        if node_id in cuts:
            continue
        for target in outgoing[node_id]:
            if target not in selected:
                selected.add(target)
                frontier.append(target)
    unreached = tuple(sorted(cuts - selected))
    if unreached:
        raise PropagationError(
            code="PARTIAL_NOT_STRICT",
            detail="every partial cut must be reached before an upstream cut stops it",
            evidence={"unreached_cuts": unreached},
        )
    if len(selected) <= 1:
        raise PropagationError(
            code="PARTIAL_NOT_NONTRIVIAL",
            detail="PARTIAL must propagate beyond its root",
        )
    if selected == allowed:
        raise PropagationError(
            code="PARTIAL_NOT_STRICT",
            detail="PARTIAL must be strictly smaller than full closure",
        )
    ineffective = tuple(
        sorted(
            cut
            for cut in cuts
            if not (set(schema.descendants(cut)) & (allowed - selected))
        )
    )
    if ineffective:
        raise PropagationError(
            code="PARTIAL_NOT_STRICT",
            detail="every partial cut must actually stop a downstream branch",
            evidence={"ineffective_cuts": ineffective},
        )
    if not schema.is_connected_downstream_subgraph((root,), selected):
        raise PropagationError(
            code="PARTIAL_NOT_CONNECTED",
            detail="PARTIAL selected nodes are not root-reachable and connected",
        )
    return tuple(
        node_id for node_id in schema.topological_order() if node_id in selected
    )


class EditingPropagationEngine:
    """Execute STOP/PARTIAL/FULL_CF/TERMINAL over typed editing DAGs."""

    __slots__ = ("_rule_registry",)

    def __init__(
        self,
        rule_registry: DerivationRuleRegistry | None = None,
    ) -> None:
        if (
            rule_registry is not None
            and type(rule_registry) is not DerivationRuleRegistry
        ):
            raise TypeError("rule_registry must be DerivationRuleRegistry or None")
        self._rule_registry = (
            DEFAULT_EDITING_DERIVATION_RULE_REGISTRY
            if rule_registry is None
            else rule_registry
        )
        self._validate_registry()

    @property
    def rule_registry(self) -> DerivationRuleRegistry:
        return self._rule_registry

    def _validate_registry(self) -> None:
        from molhallulens.domain import EDITING_STATE_SCHEMAS

        schemas = {
            schema.schema_id: schema for schema in EDITING_STATE_SCHEMAS.values()
        }
        for rule in self._rule_registry.rules:
            scopes = rule.schema_ids or frozenset(
                schema_id
                for schema_id, schema in schemas.items()
                if rule.output_node in schema.nodes_by_id
            )
            unknown_scopes = tuple(sorted(scopes - set(schemas)))
            if unknown_scopes:
                raise PropagationError(
                    code="DERIVATION_RULE_MISSING",
                    detail="rule declares unknown editing schemas",
                    node_id=rule.output_node,
                    evidence={"schema_ids": unknown_scopes},
                )
            if not scopes:
                raise PropagationError(
                    code="DERIVATION_RULE_MISSING",
                    detail="rule output is absent from all editing schemas",
                    node_id=rule.output_node,
                )
            for schema_id in scopes:
                schema = schemas[schema_id]
                output = schema.nodes_by_id.get(rule.output_node)
                if output is None:
                    # Property/custom schemas may deliberately reuse the stable
                    # editing schema_id with an alternate version.  Their exact
                    # node contracts are validated against the runtime schema.
                    continue
                if not output.mutable or output.visibility is Visibility.BUILD_ONLY:
                    raise PropagationError(
                        code="DERIVATION_RULE_MISSING",
                        detail="BUILD_ONLY or immutable nodes cannot be rule outputs",
                        node_id=rule.output_node,
                        evidence={"schema_id": schema_id},
                    )
                if output.value_type is not rule.output_type:
                    raise PropagationError(
                        code="DERIVATION_OUTPUT_TYPE_MISMATCH",
                        detail="rule output type disagrees with schema",
                        node_id=rule.output_node,
                        evidence={"schema_id": schema_id},
                    )
                for input_node, input_type in zip(
                    rule.input_nodes,
                    rule.input_types,
                    strict=True,
                ):
                    input_spec = schema.nodes_by_id.get(input_node)
                    if input_spec is None or input_spec.value_type is not input_type:
                        raise PropagationError(
                            code="DERIVATION_INPUT_TYPE_MISMATCH",
                            detail="rule input signature disagrees with schema",
                            node_id=rule.output_node,
                            evidence={
                                "schema_id": schema_id,
                                "input_node": input_node,
                            },
                        )

    def _validate_root(
        self,
        context: PerturbationContext[EditTruth],
        root_patch: CandidatePatch,
    ) -> _OperatorLabels:
        if (
            not isinstance(context, PerturbationContext)
            or type(context.truth) is not EditTruth
        ):
            raise TypeError(
                "EditingPropagationEngine requires editing PerturbationContext"
            )
        if type(root_patch) is not CandidatePatch:
            raise TypeError("root_patch must be CandidatePatch")
        recipe = context.recipe
        if root_patch.root_node_id != recipe.target_node_id:
            raise PropagationError(
                code="ROOT_PATCH_MISMATCH",
                detail="root patch target differs from recipe target",
            )
        if root_patch.root_node_id not in context.state_schema.nodes_by_id:
            raise PropagationError(
                code="ROOT_PATCH_MISMATCH",
                detail="root patch targets an unknown node",
                node_id=root_patch.root_node_id,
            )
        spec = context.state_schema.nodes_by_id[root_patch.root_node_id]
        if not spec.mutable or spec.visibility is not Visibility.CANDIDATE_OUTPUT:
            raise PropagationError(
                code="ROOT_PATCH_MISMATCH",
                detail="root patch must target mutable candidate output",
                node_id=root_patch.root_node_id,
            )
        reference = context.reference_graph.value_for(root_patch.root_node_id)
        if (
            root_patch.old_value != reference
            or root_patch.new_value.value_type is not spec.value_type
        ):
            raise PropagationError(
                code="ROOT_PATCH_MISMATCH",
                detail="root patch values do not match reference/schema",
                node_id=root_patch.root_node_id,
            )
        metadata_operator = root_patch.metadata.get("operator_id")
        if metadata_operator is not None and metadata_operator != recipe.operator_id:
            raise PropagationError(
                code="ROOT_PATCH_MISMATCH",
                detail="root patch operator metadata differs from recipe",
            )
        labels = _operator_labels_by_id().get(recipe.operator_id)
        if labels is None:
            raise PropagationError(
                code="OPERATOR_DECLARATION_MISSING",
                detail="recipe operator has no decorated declaration",
            )
        if recipe.policy not in labels.supported_policies:
            raise PropagationError(
                code="POLICY_INCOMPATIBLE",
                detail="operator does not support recipe policy",
            )
        if recipe.target_node_id not in labels.root_fields:
            raise PropagationError(
                code="ROOT_FIELD_INCOMPATIBLE",
                detail="operator does not support recipe target root",
                node_id=recipe.target_node_id,
            )
        if recipe.policy is PropagationPolicy.TERMINAL:
            if recipe.target_node_id != "final_answer":
                raise PropagationError(
                    code="TERMINAL_POLICY_VIOLATION",
                    detail="TERMINAL may only mutate final_answer",
                )
        elif recipe.target_node_id == "final_answer":
            raise PropagationError(
                code="TERMINAL_POLICY_VIOLATION",
                detail="final_answer root requires TERMINAL policy",
            )
        return labels

    def plan(
        self,
        context: PerturbationContext[EditTruth],
        root_patch: CandidatePatch,
    ) -> PropagationPlan:
        self._validate_root(context, root_patch)
        recipe = context.recipe
        full_closure = _mutable_closure(context)
        if recipe.policy in {PropagationPolicy.STOP, PropagationPolicy.TERMINAL}:
            selected = (recipe.target_node_id,)
        elif recipe.policy is PropagationPolicy.FULL_CF:
            selected = full_closure
        else:
            selected = _partial_selected_nodes(
                context,
                full_closure,
                self._rule_registry,
            )
        coverage_nodes = (
            full_closure[1:]
            if recipe.policy in {PropagationPolicy.PARTIAL, PropagationPolicy.FULL_CF}
            else selected[1:]
        )
        for node_id in coverage_nodes:
            try:
                rule = self._rule_registry.rule_for(
                    node_id,
                    schema_id=context.state_schema.schema_id,
                )
            except KeyError as error:
                raise PropagationError(
                    code="DERIVATION_RULE_MISSING",
                    detail="selected downstream node has no derivation rule",
                    node_id=node_id,
                ) from error
            self._validate_runtime_rule(context.state_schema, rule)
        return PropagationPlan(
            policy=recipe.policy,
            root_node_id=recipe.target_node_id,
            full_closure=full_closure,
            selected_nodes=selected,
        )

    def _validate_runtime_rule(self, schema: Any, rule: Any) -> None:
        output = schema.nodes_by_id.get(rule.output_node)
        if (
            output is None
            or not output.mutable
            or output.visibility is Visibility.BUILD_ONLY
        ):
            raise PropagationError(
                code="DERIVATION_RULE_MISSING",
                detail="rule output must be mutable runtime candidate state",
                node_id=rule.output_node,
            )
        if output.value_type is not rule.output_type:
            raise PropagationError(
                code="DERIVATION_OUTPUT_TYPE_MISMATCH",
                detail="runtime rule output type disagrees with schema",
                node_id=rule.output_node,
            )
        for input_node, input_type in zip(
            rule.input_nodes,
            rule.input_types,
            strict=True,
        ):
            input_spec = schema.nodes_by_id.get(input_node)
            if input_spec is None or input_spec.value_type is not input_type:
                raise PropagationError(
                    code="DERIVATION_INPUT_TYPE_MISMATCH",
                    detail="runtime rule input signature disagrees with schema",
                    node_id=rule.output_node,
                    evidence={"input_node": input_node},
                )

    def _candidate_product(
        self,
        context: PerturbationContext[EditTruth],
        root_patch: CandidatePatch,
        plan: PropagationPlan,
    ) -> str | None:
        if "product" not in plan.selected_nodes:
            return None
        action = root_patch.edit_action
        if action is None:
            raise PropagationError(
                code="STRUCTURAL_ACTION_REQUIRED",
                detail="a path reaching product requires a typed EditAction",
                node_id="product",
            )
        expected_kind = {
            "mol_edit.add": EditKind.ADDITION,
            "mol_edit.delete": EditKind.DELETION,
            "mol_edit.substitute": EditKind.SUBSTITUTION,
        }[context.state_schema.schema_id]
        if action.edit_kind is not expected_kind:
            raise PropagationError(
                code="ACTION_PRODUCT_MISMATCH",
                detail="EditAction kind differs from editing schema",
                node_id="product",
            )
        try:
            products = replay_edit_action_from_source(
                context.record.indexed_smiles,
                action,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise PropagationError(
                code="ACTION_PRODUCT_MISMATCH",
                detail="typed EditAction failed strict replay",
                node_id="product",
                evidence={"exception_type": type(error).__name__},
            ) from error
        if len(products) != 1:
            raise PropagationError(
                code="ACTION_PRODUCT_MISMATCH",
                detail="typed EditAction replay was not uniquely determined",
                node_id="product",
            )
        product = products[0]
        self._validate_root_action_alignment(root_patch, product)
        return product

    def _validate_root_action_alignment(
        self,
        root_patch: CandidatePatch,
        product: str,
    ) -> None:
        action = root_patch.edit_action
        if action is None:  # pragma: no cover - caller checks
            raise RuntimeError("missing EditAction")
        root = root_patch.root_node_id
        value = root_patch.new_value.normalized_value
        try:
            if root == "anchor_idx":
                aligned = value == action.source_anchor_index
            elif root == "product":
                aligned = type(value) is str and isomeric_graph_equivalent(
                    value, product
                )
            elif root in {"add_fragment"}:
                aligned = (
                    type(value) is str
                    and action.add_fragment_smiles is not None
                    and fragment_graph_equivalent(value, action.add_fragment_smiles)
                )
            elif root in {
                "remove_group",
                "remove_group_step1",
                "remove_group_step2",
            }:
                aligned = (
                    type(value) is str
                    and action.remove_fragment_smiles is not None
                    and fragment_graph_equivalent(value, action.remove_fragment_smiles)
                )
            else:
                aligned = True
        except (RuntimeError, TypeError, ValueError) as error:
            raise PropagationError(
                code=(
                    "ACTION_PRODUCT_MISMATCH"
                    if root == "product"
                    else "CROSS_FIELD_MISMATCH"
                ),
                detail="root claim could not be compared with EditAction semantics",
                node_id=root,
                evidence={"exception_type": type(error).__name__},
            ) from error
        if not aligned:
            raise PropagationError(
                code=(
                    "ACTION_PRODUCT_MISMATCH"
                    if root == "product"
                    else "CROSS_FIELD_MISMATCH"
                ),
                detail="root claim and EditAction semantics disagree",
                node_id=root,
            )

    def _derive_node(
        self,
        state: StateDAG,
        derivation_context: DerivationContext,
        node_id: str,
        policy: PropagationPolicy,
    ) -> tuple[ClaimValue, CausalRole]:
        schema_id = state.schema.schema_id
        try:
            rule = self._rule_registry.rule_for(node_id, schema_id=schema_id)
        except KeyError as error:
            raise PropagationError(
                code="DERIVATION_RULE_MISSING",
                detail="selected node has no derivation rule",
                node_id=node_id,
            ) from error
        for input_node, input_type in zip(
            rule.input_nodes,
            rule.input_types,
            strict=True,
        ):
            if state.value_for(input_node).value_type is not input_type:
                raise PropagationError(
                    code="DERIVATION_INPUT_TYPE_MISMATCH",
                    detail="runtime input claim type differs from rule signature",
                    node_id=node_id,
                    evidence={"input_node": input_node},
                )
        try:
            derived = rule.derive(state, derivation_context)
        except PropagationError:
            raise
        except Exception as error:
            raise PropagationError(
                code="DERIVATION_FAILED",
                detail="derivation rule raised an exception",
                node_id=node_id,
                evidence={
                    "rule_id": rule.rule_id,
                    "exception_type": type(error).__name__,
                },
            ) from error
        if (
            type(derived) is not ClaimValue
            or derived.value_type is not rule.output_type
        ):
            raise PropagationError(
                code="DERIVATION_OUTPUT_TYPE_MISMATCH",
                detail="rule returned a claim with the wrong output type",
                node_id=node_id,
                evidence={"rule_id": rule.rule_id},
            )
        expected: ClaimValue | None = None
        authoritative_schema = state_schema_for(
            derivation_context.context.truth.normalized_subtask
        )
        if state.schema == authoritative_schema:
            try:
                expected_rule = DEFAULT_EDITING_DERIVATION_RULE_REGISTRY.rule_for(
                    node_id,
                    schema_id=schema_id,
                )
                expected = expected_rule.derive(state, derivation_context)
            except KeyError as error:
                raise PropagationError(
                    code="DERIVATION_RULE_MISSING",
                    detail="authoritative editing output has no relation oracle",
                    node_id=node_id,
                ) from error
            except (PropagationError, RuntimeError, TypeError, ValueError) as error:
                raise PropagationError(
                    code="CROSS_FIELD_MISMATCH",
                    detail="candidate-parent relation could not be evaluated",
                    node_id=node_id,
                    evidence={"reason": "relation_unknown"},
                ) from error
        relation_valid = (
            rule.causal_role is CausalRole.PROPAGATED_CONDITIONAL
            if expected is None
            else self._claims_equivalent(state, node_id, derived, expected)
        )
        if node_id == "heavy_delta" and schema_id == "mol_edit.substitute":
            fragment_delta = _integer_value(state, "add_heavy") - _integer_value(
                state, "remove_heavy"
            )
            relation_valid = relation_valid and (
                derived.normalized_value == fragment_delta
            )
        if not relation_valid and policy is PropagationPolicy.FULL_CF:
            raise PropagationError(
                code="CROSS_FIELD_MISMATCH",
                detail="FULL_CF derivation is inconsistent with candidate parents",
                node_id=node_id,
                evidence={"reason": "relation_conflict"},
            )
        causal_role = (
            CausalRole.PROPAGATED_CONDITIONAL
            if relation_valid
            else CausalRole.PROPAGATED_FALSE
        )
        derived = replace(
            derived,
            provenance=ValueProvenance.PROPAGATED,
            locally_valid=relation_valid,
            oracle_match=False,
            confidence=1.0,
            mention_ids=state.value_for(node_id).mention_ids,
        )
        return derived, causal_role

    def _claims_equivalent(
        self,
        state: StateDAG,
        node_id: str,
        actual: ClaimValue,
        expected: ClaimValue,
    ) -> bool:
        comparator = state.schema.nodes_by_id[node_id].comparator.value
        try:
            if comparator == "isomeric_graph_equivalence":
                return (
                    type(actual.normalized_value) is str
                    and type(expected.normalized_value) is str
                    and isomeric_graph_equivalent(
                        actual.normalized_value,
                        expected.normalized_value,
                    )
                )
            if comparator == "fragment_graph_equivalence":
                return (
                    type(actual.normalized_value) is str
                    and type(expected.normalized_value) is str
                    and fragment_graph_equivalent(
                        actual.normalized_value,
                        expected.normalized_value,
                    )
                )
            return actual.semantically_equals(expected)
        except (RuntimeError, TypeError, ValueError) as error:
            raise PropagationError(
                code="CROSS_FIELD_MISMATCH",
                detail="candidate-parent relation could not be compared",
                node_id=node_id,
                evidence={
                    "reason": "relation_unknown",
                    "exception_type": type(error).__name__,
                },
            ) from error

    def propagate(
        self,
        context: PerturbationContext[EditTruth],
        root_patch: CandidatePatch,
    ) -> PropagationOutcome:
        labels = self._validate_root(context, root_patch)
        plan = self.plan(context, root_patch)
        product = self._candidate_product(context, root_patch, plan)
        derivation_context = DerivationContext(context, root_patch, product)
        values = dict(context.reference_graph.values)
        values[plan.root_node_id] = root_patch.new_value
        state = StateDAG(
            context.state_schema,
            values,
            context.reference_graph.edge_values,
        )
        roles: dict[str, CausalRole] = {}
        for node_id in plan.selected_nodes[1:]:
            derived, role = self._derive_node(
                state,
                derivation_context,
                node_id,
                plan.policy,
            )
            reference = context.reference_graph.value_for(node_id)
            values[node_id] = (
                reference if derived.semantically_equals(reference) else derived
            )
            roles[node_id] = role
            state = StateDAG(
                context.state_schema,
                values,
                context.reference_graph.edge_values,
            )

        differences = state.semantic_differences(context.reference_graph)
        if any(kind is not MutationTargetKind.NODE for kind, _ in differences):
            raise PropagationError(
                code="GRAPH_DELTA_MISMATCH",
                detail="T022 propagation cannot mutate edge claims",
            )
        difference_nodes = {node_id for _, node_id in differences}
        if plan.root_node_id not in difference_nodes or not difference_nodes <= set(
            plan.selected_nodes
        ):
            raise PropagationError(
                code="GRAPH_DELTA_MISMATCH",
                detail="semantic differences escape the propagation plan",
                evidence={"differences": tuple(sorted(difference_nodes))},
            )
        if plan.policy is PropagationPolicy.PARTIAL:
            propagated_differences = difference_nodes - {plan.root_node_id}
            if not propagated_differences:
                raise PropagationError(
                    code="PARTIAL_NOT_NONTRIVIAL",
                    detail="PARTIAL must produce a downstream semantic difference",
                )
            if not context.state_schema.is_connected_downstream_subgraph(
                (plan.root_node_id,), difference_nodes
            ):
                raise PropagationError(
                    code="PARTIAL_NOT_CONNECTED",
                    detail="PARTIAL semantic differences are not root-connected",
                    evidence={"differences": tuple(sorted(difference_nodes))},
                )
        root_event_id = f"{context.recipe.recipe_id}.mutation.root"
        root_role = (
            CausalRole.TERMINAL
            if plan.policy is PropagationPolicy.TERMINAL
            else CausalRole.ROOT
        )
        events = [
            MutationEvent(
                event_id=root_event_id,
                target_kind=MutationTargetKind.NODE,
                node_or_edge_id=plan.root_node_id,
                before=context.reference_graph.value_for(plan.root_node_id),
                after=state.value_for(plan.root_node_id),
                causal_role=root_role,
                hallucination_types=labels.hallucination_types,
                edit_subtypes=labels.edit_subtypes,
                operator_id=context.recipe.operator_id,
                root_event_id=root_event_id,
            )
        ]
        derived_index = 0
        for node_id in plan.selected_nodes[1:]:
            if node_id not in difference_nodes:
                continue
            derived_index += 1
            events.append(
                MutationEvent(
                    event_id=(
                        f"{context.recipe.recipe_id}.mutation.{derived_index:02d}.{node_id}"
                    ),
                    target_kind=MutationTargetKind.NODE,
                    node_or_edge_id=node_id,
                    before=context.reference_graph.value_for(node_id),
                    after=state.value_for(node_id),
                    causal_role=roles[node_id],
                    hallucination_types=labels.hallucination_types,
                    edit_subtypes=labels.edit_subtypes,
                    operator_id=context.recipe.operator_id,
                    root_event_id=root_event_id,
                )
            )
        delta = GraphDelta(tuple(events))
        delta_targets = {
            (event.target_kind, event.node_or_edge_id) for event in delta.events
        }
        if delta_targets != differences:
            raise PropagationError(
                code="GRAPH_DELTA_MISMATCH",
                detail="GraphDelta targets differ from candidate graph semantics",
            )
        return PropagationOutcome(candidate_graph=state, graph_delta=delta)


__all__ = [
    "DEFAULT_EDITING_DERIVATION_RULE_REGISTRY",
    "EditingPropagationEngine",
    "editing_derivation_rule_registry",
]
