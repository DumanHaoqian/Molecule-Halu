"""Deterministic FORMAL and final-Answer rendering from locked editing state.

This module deliberately does not render natural language, construct character
offsets, inspect truth objects, or repair chemistry.  Its only fact-bearing
input is an authoritative :class:`StateDAG`; FORMAL text is a byte-stable
projection of its typed normalized values, and the Answer preserves the locked
``final_answer`` serialization exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from string import Formatter
from types import MappingProxyType
from typing import Any

from molhallulens.infrastructure.chemistry import isomeric_graph_equivalent
from molhallulens.core import (
    EditingSubtask,
    PropagationPolicy,
    StateDAG,
    ValueType,
    deep_freeze,
    editing_schema_for,
)
from molhallulens.core.state_dag import FrozenMap

_INTEGER_VALUE_TYPES = frozenset(
    {ValueType.INTEGER, ValueType.ATOM_INDEX, ValueType.COUNT}
)
_STRING_VALUE_TYPES = frozenset(
    {
        ValueType.STRING,
        ValueType.SMILES,
        ValueType.INDEXED_SMILES,
        ValueType.ELEMENT,
        ValueType.FRAGMENT,
        ValueType.MOLECULE,
    }
)


class FormalRenderError(RuntimeError):
    """Structured fail-closed FORMAL/Answer rendering failure."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        schema_id: str | None = None,
        step_index: int | None = None,
        node_id: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("FormalRenderError code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("FormalRenderError detail must be non-empty text")
        for value, name in ((schema_id, "schema_id"), (node_id, "node_id")):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"FormalRenderError {name} must be non-empty or None")
        if step_index is not None and (type(step_index) is not int or step_index <= 0):
            raise ValueError("FormalRenderError step_index must be positive or None")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("FormalRenderError evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.schema_id = schema_id
        self.step_index = step_index
        self.node_id = node_id
        self.evidence = MappingProxyType(dict(evidence or {}))
        location = ""
        if schema_id is not None:
            location += f" schema={schema_id!r}"
        if step_index is not None:
            location += f" step={step_index}"
        if node_id is not None:
            location += f" node={node_id!r}"
        super().__init__(f"{code}{location}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "schema_id": self.schema_id,
            "step_index": self.step_index,
            "node_id": self.node_id,
            "evidence": dict(self.evidence),
        }


def _validate_scalar(value_type: ValueType, value: Any, *, name: str) -> Any:
    if type(value_type) is not ValueType:
        raise TypeError(f"{name} value_type must be ValueType")
    frozen = deep_freeze(value)
    if value_type in _INTEGER_VALUE_TYPES:
        if type(frozen) is not int:
            raise TypeError(f"{name} must be an integer")
        if value_type is ValueType.ATOM_INDEX and frozen <= 0:
            raise ValueError(f"{name} must be one-based and positive")
        if value_type is ValueType.COUNT and frozen < 0:
            raise ValueError(f"{name} cannot be negative")
    elif value_type in _STRING_VALUE_TYPES:
        if type(frozen) is not str:
            raise TypeError(f"{name} must be text")
    else:
        raise TypeError(f"{name} uses unsupported FORMAL value type {value_type.value}")
    return frozen


@dataclass(frozen=True, slots=True)
class FormalSlotValue:
    """One typed occurrence in a rendered FORMAL expression."""

    field_name: str
    node_id: str
    value_type: ValueType
    value: Any

    def __post_init__(self) -> None:
        for value, name in (
            (self.field_name, "field_name"),
            (self.node_id, "node_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"FormalSlotValue {name} must be non-empty text")
        if type(self.value_type) is not ValueType:
            raise TypeError("FormalSlotValue value_type must be ValueType")
        object.__setattr__(self, "value", deep_freeze(self.value))


@dataclass(frozen=True, slots=True)
class RenderedFormalStep:
    """One exact T011 FORMAL expression and its audit-only typed slots."""

    step_index: int
    step_name: str
    formal_ab: str
    slots: tuple[FormalSlotValue, ...]

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("RenderedFormalStep step_index must be positive")
        for value, name in (
            (self.step_name, "step_name"),
            (self.formal_ab, "formal_ab"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"RenderedFormalStep {name} must be non-empty text")
        if "\r" in self.formal_ab or "\n" in self.formal_ab or "\x00" in self.formal_ab:
            raise ValueError("RenderedFormalStep formal_ab must be one NUL-free line")
        slots = tuple(self.slots)
        if any(type(slot) is not FormalSlotValue for slot in slots):
            raise TypeError("RenderedFormalStep slots must contain FormalSlotValue")
        field_names = tuple(slot.field_name for slot in slots)
        if len(field_names) != len(set(field_names)):
            raise ValueError("RenderedFormalStep field occurrences must be unique")
        object.__setattr__(self, "slots", slots)

    @property
    def step_header(self) -> str:
        return f"Step {self.step_index} [{self.step_name}]"

    @property
    def formal_line(self) -> str:
        return f"  FORMAL: {self.formal_ab}"


@dataclass(frozen=True, slots=True)
class RenderedFormalTrace:
    """Byte-stable FORMAL projection of one authoritative editing StateDAG."""

    schema_id: str
    schema_version: str
    normalized_subtask: EditingSubtask
    steps: tuple[RenderedFormalStep, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.schema_id, "schema_id"),
            (self.schema_version, "schema_version"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"RenderedFormalTrace {name} must be non-empty text")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        steps = tuple(self.steps)
        if not steps or any(type(step) is not RenderedFormalStep for step in steps):
            raise TypeError("steps must contain RenderedFormalStep values")
        object.__setattr__(self, "steps", steps)

    @property
    def formal_lines(self) -> tuple[str, ...]:
        return tuple(step.formal_line for step in self.steps)

    @property
    def formal_text(self) -> str:
        return "\n".join(
            line for step in self.steps for line in (step.step_header, step.formal_line)
        )


@dataclass(frozen=True, slots=True)
class ParsedFormalState:
    """Typed scalar state recovered exclusively from FORMAL text."""

    schema_id: str
    schema_version: str
    normalized_subtask: EditingSubtask
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, name in (
            (self.schema_id, "schema_id"),
            (self.schema_version, "schema_version"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"ParsedFormalState {name} must be non-empty text")
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError("normalized_subtask must be EditingSubtask")
        if not isinstance(self.values, Mapping):
            raise TypeError("ParsedFormalState values must be a mapping")
        definition = editing_schema_for(self.normalized_subtask)
        expected_nodes = set(definition.legacy_step_field_bindings.values())
        if set(self.values) != expected_nodes or any(
            type(node_id) is not str for node_id in self.values
        ):
            raise ValueError("ParsedFormalState values must cover exact FORMAL nodes")
        validated = {
            node_id: _validate_scalar(
                definition.schema.nodes_by_id[node_id].value_type,
                value,
                name=node_id,
            )
            for node_id, value in self.values.items()
        }
        object.__setattr__(self, "values", FrozenMap(validated))


@dataclass(frozen=True, slots=True)
class RenderedAnswer:
    """Exact locked final-answer surface plus its relation to locked product."""

    policy: PropagationPolicy
    source_node_id: str
    smiles: str
    product_equivalent: bool

    def __post_init__(self) -> None:
        if type(self.policy) is not PropagationPolicy:
            raise TypeError("RenderedAnswer policy must be PropagationPolicy")
        if self.source_node_id != "final_answer":
            raise ValueError("RenderedAnswer source_node_id must be final_answer")
        if type(self.smiles) is not str or not self.smiles:
            raise ValueError("RenderedAnswer smiles must be non-empty text")
        if type(self.product_equivalent) is not bool:
            raise TypeError("RenderedAnswer product_equivalent must be bool")

    @property
    def answer_matches_product(self) -> bool:
        return self.product_equivalent

    @property
    def answer_line(self) -> str:
        return f"Answer: {self.smiles}"


@dataclass(frozen=True, slots=True)
class _FormalStepSpec:
    step_index: int
    step_name: str
    template: str
    fields: tuple[str, ...]
    signed_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index <= 0:
            raise ValueError("formal step_index must be positive")
        if type(self.step_name) is not str or not self.step_name:
            raise ValueError("formal step_name must be non-empty")
        if type(self.template) is not str or not self.template:
            raise ValueError("formal template must be non-empty")
        fields = tuple(self.fields)
        signed = frozenset(self.signed_fields)
        placeholders = tuple(
            field_name
            for _, field_name, format_spec, conversion in Formatter().parse(
                self.template
            )
            if field_name is not None and not format_spec and conversion is None
        )
        if placeholders != fields or len(fields) != len(set(fields)):
            raise ValueError("formal fields must exactly match unique placeholders")
        if not signed <= set(fields):
            raise ValueError("signed fields must be formal fields")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "signed_fields", signed)

    def render(
        self, values: Mapping[str, Any], value_types: Mapping[str, ValueType]
    ) -> str:
        rendered: dict[str, str] = {}
        for field_name in self.fields:
            value_type = value_types[field_name]
            value = _validate_scalar(value_type, values[field_name], name=field_name)
            if value_type in _INTEGER_VALUE_TYPES:
                rendered[field_name] = (
                    f"+{value}"
                    if field_name in self.signed_fields and value > 0
                    else str(value)
                )
            else:
                if not value or any(
                    marker in value for marker in ('"', "\r", "\n", "\x00")
                ):
                    raise FormalRenderError(
                        "FORMAL_SLOT_MISMATCH",
                        "quoted FORMAL values must be non-empty and need no escaping",
                        step_index=self.step_index,
                        evidence={"field_name": field_name},
                    )
                rendered[field_name] = value
        return self.template.format_map(rendered)

    def parse(
        self,
        formal_ab: str,
        value_types: Mapping[str, ValueType],
    ) -> Mapping[str, Any]:
        pattern_parts: list[str] = []
        for literal, field_name, _, _ in Formatter().parse(self.template):
            pattern_parts.append(re.escape(literal))
            if field_name is None:
                continue
            value_type = value_types[field_name]
            if value_type in _INTEGER_VALUE_TYPES:
                payload = r"[+-]?\d+" if field_name in self.signed_fields else r"-?\d+"
            else:
                payload = r'[^"\r\n\x00]+'
            pattern_parts.append(f"(?P<{field_name}>{payload})")
        match = re.fullmatch("".join(pattern_parts), formal_ab)
        if match is None:
            raise FormalRenderError(
                "FORMAL_PARSE_ERROR",
                "FORMAL text does not match its exact step grammar",
                step_index=self.step_index,
            )
        parsed: dict[str, Any] = {}
        for field_name in self.fields:
            raw = match.group(field_name)
            value_type = value_types[field_name]
            parsed[field_name] = int(raw) if value_type in _INTEGER_VALUE_TYPES else raw
        try:
            canonical = self.render(parsed, value_types)
        except (TypeError, ValueError) as error:
            raise FormalRenderError(
                "FORMAL_PARSE_ERROR",
                "parsed FORMAL scalar violates its typed state contract",
                step_index=self.step_index,
                evidence={"exception_type": type(error).__name__},
            ) from error
        if canonical != formal_ab:
            raise FormalRenderError(
                "FORMAL_PARSE_ERROR",
                "FORMAL scalar serialization is not canonical",
                step_index=self.step_index,
            )
        return MappingProxyType(parsed)


def _spec(
    step_index: int,
    step_name: str,
    template: str,
    *fields: str,
    signed: Sequence[str] = (),
) -> _FormalStepSpec:
    return _FormalStepSpec(
        step_index=step_index,
        step_name=step_name,
        template=template,
        fields=tuple(fields),
        signed_fields=frozenset(signed),
    )


_FORMAL_SPECS: Mapping[EditingSubtask, tuple[_FormalStepSpec, ...]] = MappingProxyType(
    {
        EditingSubtask.ADD: (
            _spec(
                1,
                "ANCHOR_IDENTIFICATION",
                'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + LEAVING(smiles="{step1_leaving_smiles}")',
                "step1_anchor_idx",
                "step1_anchor_element",
                "step1_leaving_smiles",
            ),
            _spec(
                2,
                "FRAGMENT_IDENTIFICATION",
                'INSTRUCTION --> ADD_FRAGMENT(smiles="{step2_frag_smiles}", heavy_atoms={step2_heavy_atoms})',
                "step2_frag_smiles",
                "step2_heavy_atoms",
            ),
            _spec(
                3,
                "PRODUCT_CONSTRUCTION",
                'SMILES + ANCHOR(idx={step1_anchor_idx}) + LEAVING("{step1_leaving_smiles}") + ADD_FRAGMENT(smiles="{step2_frag_smiles}") --> PRODUCT_SMILES("{step3_product_smiles}")',
                "step1_anchor_idx",
                "step1_leaving_smiles",
                "step2_frag_smiles",
                "step3_product_smiles",
            ),
            _spec(
                4,
                "HEAVY_ATOM_VERIFICATION",
                "SMILES[n_heavy={step4_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step4_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step4_heavy_delta})",
                "step4_n_heavy_src",
                "step4_n_heavy_prod",
                "step4_heavy_delta",
                signed=("step4_heavy_delta",),
            ),
            _spec(
                5,
                "RING_VERIFICATION",
                "SMILES[n_rings={step5_n_rings_src}] + PRODUCT_SMILES[n_rings={step5_n_rings_prod}] --> RING_DELTA({step5_ring_delta})",
                "step5_n_rings_src",
                "step5_n_rings_prod",
                "step5_ring_delta",
                signed=("step5_ring_delta",),
            ),
        ),
        EditingSubtask.DELETE: (
            _spec(
                1,
                "ANCHOR_IDENTIFICATION",
                'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + REMOVE_GROUP(smiles="{step1_remove_group}")',
                "step1_anchor_idx",
                "step1_anchor_element",
                "step1_remove_group",
            ),
            _spec(
                2,
                "GROUP_SIZE_VERIFICATION",
                'REMOVE_GROUP(smiles="{step2_remove_smiles}") --> HEAVY_ATOMS({step2_heavy_atoms})',
                "step2_remove_smiles",
                "step2_heavy_atoms",
            ),
            _spec(
                3,
                "PRODUCT_CONSTRUCTION",
                'SMILES + ANCHOR(idx={step1_anchor_idx}) + REMOVE_GROUP(smiles="{step2_remove_smiles}") --> PRODUCT_SMILES("{step3_product_smiles}")',
                "step1_anchor_idx",
                "step2_remove_smiles",
                "step3_product_smiles",
            ),
            _spec(
                4,
                "HEAVY_ATOM_VERIFICATION",
                "SMILES[n_heavy={step4_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step4_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step4_heavy_delta})",
                "step4_n_heavy_src",
                "step4_n_heavy_prod",
                "step4_heavy_delta",
                signed=("step4_heavy_delta",),
            ),
            _spec(
                5,
                "RING_VERIFICATION",
                "SMILES[n_rings={step5_n_rings_src}] + PRODUCT_SMILES[n_rings={step5_n_rings_prod}] --> RING_DELTA({step5_ring_delta})",
                "step5_n_rings_src",
                "step5_n_rings_prod",
                "step5_ring_delta",
                signed=("step5_ring_delta",),
            ),
        ),
        EditingSubtask.SUBSTITUTE: (
            _spec(
                1,
                "ANCHOR_IDENTIFICATION",
                'INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx={step1_anchor_idx}, element="{step1_anchor_element}") + REMOVE_GROUP(smiles="{step1_remove_group_smiles}") + ADD_FRAGMENT(smiles="{step1_add_fragment_smiles}")',
                "step1_anchor_idx",
                "step1_anchor_element",
                "step1_remove_group_smiles",
                "step1_add_fragment_smiles",
            ),
            _spec(
                2,
                "REMOVE_GROUP_SIZE",
                'REMOVE_GROUP(smiles="{step1_remove_group_smiles}") --> REMOVE_HEAVY({step2_remove_heavy})',
                "step1_remove_group_smiles",
                "step2_remove_heavy",
            ),
            _spec(
                3,
                "ADD_FRAGMENT_SIZE",
                'ADD_FRAGMENT(smiles="{step1_add_fragment_smiles}") --> ADD_HEAVY({step3_add_heavy})',
                "step1_add_fragment_smiles",
                "step3_add_heavy",
            ),
            _spec(
                4,
                "PRODUCT_CONSTRUCTION",
                'SMILES + ANCHOR(idx={step1_anchor_idx}) + REMOVE_GROUP("{step1_remove_group_smiles}") + ADD_FRAGMENT("{step1_add_fragment_smiles}") --> PRODUCT_SMILES("{step4_product_smiles}")',
                "step1_anchor_idx",
                "step1_remove_group_smiles",
                "step1_add_fragment_smiles",
                "step4_product_smiles",
            ),
            _spec(
                5,
                "HEAVY_ATOM_VERIFICATION",
                "SMILES[n_heavy={step5_n_heavy_src}] + PRODUCT_SMILES[n_heavy={step5_n_heavy_prod}] --> HEAVY_ATOM_DELTA({step5_heavy_delta})",
                "step5_n_heavy_src",
                "step5_n_heavy_prod",
                "step5_heavy_delta",
                signed=("step5_heavy_delta",),
            ),
            _spec(
                6,
                "RING_VERIFICATION",
                "SMILES[n_rings={step6_n_rings_src}] + PRODUCT_SMILES[n_rings={step6_n_rings_prod}] --> RING_DELTA({step6_ring_delta})",
                "step6_n_rings_src",
                "step6_n_rings_prod",
                "step6_ring_delta",
                signed=("step6_ring_delta",),
            ),
        ),
    }
)


def _definition_for_state(state: StateDAG):
    if type(state) is not StateDAG:
        raise TypeError("deterministic renderer requires a StateDAG")
    matches = tuple(
        subtask
        for subtask in EditingSubtask
        if editing_schema_for(subtask).schema.schema_id == state.schema.schema_id
    )
    if len(matches) != 1:
        raise FormalRenderError(
            "FORMAL_SCHEMA_MISMATCH",
            "state schema is not a registered editing schema",
            schema_id=state.schema.schema_id,
        )
    subtask = matches[0]
    definition = editing_schema_for(subtask)
    if state.schema != definition.schema:
        raise FormalRenderError(
            "FORMAL_SCHEMA_MISMATCH",
            "state schema/version differs from the authoritative editing contract",
            schema_id=state.schema.schema_id,
        )
    return subtask, definition


def _value_types(definition: Any, spec: _FormalStepSpec) -> Mapping[str, ValueType]:
    return MappingProxyType(
        {
            field_name: definition.schema.nodes_by_id[
                definition.legacy_step_field_bindings[field_name]
            ].value_type
            for field_name in spec.fields
        }
    )


class DeterministicFormalRenderer:
    """Render and parse exact T011 FORMAL grammar without state repair."""

    __slots__ = ()

    def render(self, state: StateDAG) -> RenderedFormalTrace:
        subtask, definition = _definition_for_state(state)
        steps: list[RenderedFormalStep] = []
        for spec in _FORMAL_SPECS[subtask]:
            value_types = _value_types(definition, spec)
            try:
                slots = tuple(
                    FormalSlotValue(
                        field_name=field_name,
                        node_id=definition.legacy_step_field_bindings[field_name],
                        value_type=value_types[field_name],
                        value=state.value_for(
                            definition.legacy_step_field_bindings[field_name]
                        ).normalized_value,
                    )
                    for field_name in spec.fields
                )
                values = {slot.field_name: slot.value for slot in slots}
                formal_ab = spec.render(values, value_types)
            except FormalRenderError as error:
                if error.schema_id is not None:
                    raise
                raise FormalRenderError(
                    error.code,
                    error.detail,
                    schema_id=state.schema.schema_id,
                    step_index=error.step_index,
                    node_id=error.node_id,
                    evidence=error.evidence,
                ) from error
            except (TypeError, ValueError) as error:
                raise FormalRenderError(
                    "FORMAL_SLOT_MISMATCH",
                    "locked FORMAL value violates its typed state contract",
                    schema_id=state.schema.schema_id,
                    step_index=spec.step_index,
                    evidence={"exception_type": type(error).__name__},
                ) from error
            steps.append(
                RenderedFormalStep(
                    step_index=spec.step_index,
                    step_name=spec.step_name,
                    formal_ab=formal_ab,
                    slots=slots,
                )
            )
        trace = RenderedFormalTrace(
            schema_id=state.schema.schema_id,
            schema_version=state.schema.version,
            normalized_subtask=subtask,
            steps=tuple(steps),
        )
        self.assert_round_trip(state, trace)
        return trace

    def parse(self, trace: RenderedFormalTrace) -> ParsedFormalState:
        if type(trace) is not RenderedFormalTrace:
            raise TypeError("parse requires RenderedFormalTrace")
        definition = editing_schema_for(trace.normalized_subtask)
        if (
            trace.schema_id != definition.schema.schema_id
            or trace.schema_version != definition.schema.version
        ):
            raise FormalRenderError(
                "FORMAL_SCHEMA_MISMATCH",
                "trace schema metadata differs from its normalized subtask",
                schema_id=trace.schema_id,
            )
        specs = _FORMAL_SPECS[trace.normalized_subtask]
        if len(trace.steps) != len(specs):
            raise FormalRenderError(
                "FORMAL_STEP_MISMATCH",
                "trace has the wrong number of FORMAL steps",
                schema_id=trace.schema_id,
                evidence={"expected": len(specs), "actual": len(trace.steps)},
            )
        node_values: dict[str, Any] = {}
        for step, spec in zip(trace.steps, specs, strict=True):
            if step.step_index != spec.step_index or step.step_name != spec.step_name:
                raise FormalRenderError(
                    "FORMAL_STEP_MISMATCH",
                    "FORMAL step identity/order differs from the exact contract",
                    schema_id=trace.schema_id,
                    step_index=step.step_index,
                )
            value_types = _value_types(definition, spec)
            try:
                parsed = spec.parse(step.formal_ab, value_types)
            except FormalRenderError as error:
                raise FormalRenderError(
                    error.code,
                    error.detail,
                    schema_id=trace.schema_id,
                    step_index=spec.step_index,
                    evidence=error.evidence,
                ) from error
            parsed_slots = tuple(
                FormalSlotValue(
                    field_name=field_name,
                    node_id=definition.legacy_step_field_bindings[field_name],
                    value_type=value_types[field_name],
                    value=parsed[field_name],
                )
                for field_name in spec.fields
            )
            if step.slots != parsed_slots:
                raise FormalRenderError(
                    "FORMAL_SLOT_MISMATCH",
                    "audit slots differ from values parsed from FORMAL text",
                    schema_id=trace.schema_id,
                    step_index=spec.step_index,
                )
            for slot in parsed_slots:
                if slot.node_id in node_values and (
                    type(node_values[slot.node_id]) is not type(slot.value)
                    or node_values[slot.node_id] != slot.value
                ):
                    raise FormalRenderError(
                        "FORMAL_SLOT_MISMATCH",
                        "repeated FORMAL mentions disagree for one state node",
                        schema_id=trace.schema_id,
                        step_index=spec.step_index,
                        node_id=slot.node_id,
                    )
                node_values[slot.node_id] = slot.value
        try:
            return ParsedFormalState(
                schema_id=trace.schema_id,
                schema_version=trace.schema_version,
                normalized_subtask=trace.normalized_subtask,
                values=node_values,
            )
        except (TypeError, ValueError) as error:
            raise FormalRenderError(
                "FORMAL_SLOT_MISMATCH",
                "parsed FORMAL text does not cover the exact typed state projection",
                schema_id=trace.schema_id,
                evidence={"exception_type": type(error).__name__},
            ) from error

    def assert_round_trip(
        self,
        state: StateDAG,
        trace: RenderedFormalTrace,
    ) -> None:
        subtask, definition = _definition_for_state(state)
        if (
            trace.normalized_subtask is not subtask
            or trace.schema_id != state.schema.schema_id
            or trace.schema_version != state.schema.version
        ):
            raise FormalRenderError(
                "FORMAL_SCHEMA_MISMATCH",
                "trace cannot round-trip against a different state schema",
                schema_id=trace.schema_id,
            )
        parsed = self.parse(trace)
        expected = {
            node_id: state.value_for(node_id).normalized_value
            for node_id in set(definition.legacy_step_field_bindings.values())
        }
        if dict(parsed.values) != expected or any(
            type(parsed.values[node_id]) is not type(value)
            for node_id, value in expected.items()
        ):
            raise FormalRenderError(
                "FORMAL_ROUND_TRIP_MISMATCH",
                "FORMAL parse does not recover exact locked normalized state",
                schema_id=state.schema.schema_id,
            )


class DeterministicAnswerRenderer:
    """Render the exact locked Answer and audit its graph relation to product."""

    __slots__ = ()

    def render(
        self,
        state: StateDAG,
        *,
        policy: PropagationPolicy,
    ) -> RenderedAnswer:
        if type(policy) is not PropagationPolicy:
            raise TypeError("policy must be PropagationPolicy")
        _, definition = _definition_for_state(state)
        answer_spec = definition.schema.nodes_by_id.get("final_answer")
        product_spec = definition.schema.nodes_by_id.get("product")
        if (
            answer_spec is None
            or product_spec is None
            or answer_spec.value_type is not ValueType.SMILES
            or product_spec.value_type is not ValueType.SMILES
        ):
            raise FormalRenderError(
                "ANSWER_STATE_MISMATCH",
                "editing state lacks typed product/final_answer nodes",
                schema_id=state.schema.schema_id,
            )
        answer = state.value_for("final_answer").normalized_value
        product = state.value_for("product").normalized_value
        if (
            type(answer) is not str
            or not answer
            or type(product) is not str
            or not product
        ):
            raise FormalRenderError(
                "ANSWER_STATE_MISMATCH",
                "product and final_answer must be non-empty locked SMILES",
                schema_id=state.schema.schema_id,
            )
        try:
            product_equivalent = isomeric_graph_equivalent(answer, product)
        except (RuntimeError, TypeError, ValueError) as error:
            raise FormalRenderError(
                "ANSWER_INVALID_SMILES",
                "locked product/final_answer cannot be compared as strict molecules",
                schema_id=state.schema.schema_id,
                evidence={"exception_type": type(error).__name__},
            ) from error
        if policy is PropagationPolicy.FULL_CF and not product_equivalent:
            raise FormalRenderError(
                "ANSWER_PRODUCT_MISMATCH",
                "FULL_CF final_answer must be graph-equivalent to candidate product",
                schema_id=state.schema.schema_id,
                node_id="final_answer",
            )
        try:
            faithful_surface = isomeric_graph_equivalent(answer, answer)
        except (RuntimeError, TypeError, ValueError) as error:
            raise FormalRenderError(
                "ANSWER_INVALID_SMILES",
                "locked final_answer is not strict parseable SMILES",
                schema_id=state.schema.schema_id,
                node_id="final_answer",
                evidence={"exception_type": type(error).__name__},
            ) from error
        if not faithful_surface:  # pragma: no cover - reflexive comparator contract
            raise FormalRenderError(
                "ANSWER_SERIALIZATION_MISMATCH",
                "rendered Answer changed the locked molecular graph",
                schema_id=state.schema.schema_id,
                node_id="final_answer",
            )
        return RenderedAnswer(
            policy=policy,
            source_node_id="final_answer",
            smiles=answer,
            product_equivalent=product_equivalent,
        )


_FORMAL_RENDERER = DeterministicFormalRenderer()
_ANSWER_RENDERER = DeterministicAnswerRenderer()


def render_formal(state: StateDAG) -> RenderedFormalTrace:
    return _FORMAL_RENDERER.render(state)


def parse_formal(trace: RenderedFormalTrace) -> ParsedFormalState:
    return _FORMAL_RENDERER.parse(trace)


def render_answer(
    state: StateDAG,
    *,
    policy: PropagationPolicy,
) -> RenderedAnswer:
    return _ANSWER_RENDERER.render(state, policy=policy)


__all__ = [
    "DeterministicAnswerRenderer",
    "DeterministicFormalRenderer",
    "FormalRenderError",
    "FormalSlotValue",
    "ParsedFormalState",
    "RenderedAnswer",
    "RenderedFormalStep",
    "RenderedFormalTrace",
    "parse_formal",
    "render_answer",
    "render_formal",
]
