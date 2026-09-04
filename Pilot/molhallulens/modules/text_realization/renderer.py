"""Render complete reasoning text and exact hallucination mentions from one graph."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from molhallulens.core import (
    EditingSubtask,
    InjectedHallucination,
    RenderedHallucination,
    RenderedMention,
    StateDAG,
)
from molhallulens.modules.reference import ReferenceDAGArtifact

from .poe_agent import (
    PoeRewriteRequest,
    PoeStepRewriteInput,
    PoeStepTextAgent,
)


_FORMAL_MARKER = "\n  FORMAL: "
_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}", flags=re.ASCII)


@dataclass(frozen=True, slots=True)
class _Literal:
    node_id: str
    signed: bool = False


def _value_text(graph: StateDAG, literal: _Literal) -> str:
    value = graph.values[literal.node_id].normalized_value
    if literal.signed and type(value) is int and value > 0:
        return f"+{value}"
    return str(value)


def _add_step(
    *,
    origin_id: str,
    graph: StateDAG,
    edited_nodes: frozenset[str],
    step_index: int,
    parts: tuple[str | _Literal, ...],
    reasoning_offset: int,
    occurrence_counts: dict[str, int],
) -> tuple[str, tuple[RenderedMention, ...]]:
    buffer = ""
    mentions = []
    for part in parts:
        if type(part) is str:
            buffer += part
            continue
        value = _value_text(graph, part)
        start = reasoning_offset + len(buffer)
        buffer += value
        end = reasoning_offset + len(buffer)
        occurrence_counts[part.node_id] = occurrence_counts.get(part.node_id, 0) + 1
        mentions.append(
            RenderedMention(
                mention_id=(
                    f"{origin_id}.reasoning.step{step_index:02d}."
                    f"{part.node_id}.{occurrence_counts[part.node_id]:02d}"
                ),
                component="reasoning_chain",
                node_id=part.node_id,
                step_index=step_index,
                start=start,
                end=end,
                value=value,
                hallucinated=part.node_id in edited_nodes,
            )
        )
    return buffer, tuple(mentions)


def _templates(subtask: EditingSubtask) -> tuple[tuple[str | _Literal, ...], ...]:
    L = _Literal
    if subtask is EditingSubtask.ADD:
        return (
            (
                "Step 1 [ANCHOR_IDENTIFICATION]: The attachment atom is ",
                L("anchor_idx"),
                " (",
                L("anchor_element"),
                ") and the leaving group is ",
                L("leaving"),
                ".\n  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=",
                L("anchor_idx"),
                ', element="',
                L("anchor_element"),
                '") + LEAVING(smiles="',
                L("leaving"),
                '")',
            ),
            (
                "Step 2 [FRAGMENT_IDENTIFICATION]: The incoming fragment is ",
                L("add_fragment"),
                " and contains ",
                L("fragment_heavy"),
                " heavy atoms.\n  FORMAL: INSTRUCTION --> ADD_FRAGMENT(smiles=\"",
                L("add_fragment"),
                '\", heavy_atoms=',
                L("fragment_heavy"),
                ")",
            ),
            (
                "Step 3 [PRODUCT_CONSTRUCTION]: Attach ",
                L("add_fragment"),
                " to atom ",
                L("anchor_idx"),
                " after removing ",
                L("leaving"),
                ", producing the product encoded below.\n  FORMAL: SMILES + ANCHOR(idx=",
                L("anchor_idx"),
                ') + LEAVING("',
                L("leaving"),
                '") + ADD_FRAGMENT(smiles="',
                L("add_fragment"),
                '") --> PRODUCT_SMILES("',
                L("product"),
                '")',
            ),
            (
                "Step 4 [HEAVY_ATOM_VERIFICATION]: The source has ",
                L("source_heavy"),
                " heavy atoms and the product has ",
                L("product_heavy"),
                ", so the claimed delta is ",
                L("heavy_delta", signed=True),
                ".\n  FORMAL: SMILES[n_heavy=",
                L("source_heavy"),
                "] + PRODUCT_SMILES[n_heavy=",
                L("product_heavy"),
                "] --> HEAVY_ATOM_DELTA(",
                L("heavy_delta", signed=True),
                ")",
            ),
            (
                "Step 5 [RING_VERIFICATION]: The source has ",
                L("source_rings"),
                " rings and the product has ",
                L("product_rings"),
                ", so the claimed delta is ",
                L("ring_delta", signed=True),
                ".\n  FORMAL: SMILES[n_rings=",
                L("source_rings"),
                "] + PRODUCT_SMILES[n_rings=",
                L("product_rings"),
                "] --> RING_DELTA(",
                L("ring_delta", signed=True),
                ")",
            ),
        )
    if subtask is EditingSubtask.DELETE:
        return (
            (
                "Step 1 [ANCHOR_IDENTIFICATION]: The retained anchor is atom ",
                L("anchor_idx"),
                " (",
                L("anchor_element"),
                ") and the group to remove is ",
                L("remove_group_step1"),
                ".\n  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=",
                L("anchor_idx"),
                ', element="',
                L("anchor_element"),
                '") + REMOVE_GROUP(smiles="',
                L("remove_group_step1"),
                '")',
            ),
            (
                "Step 2 [GROUP_SIZE_VERIFICATION]: The removed group ",
                L("remove_group_step2"),
                " contains ",
                L("remove_heavy"),
                " heavy atoms.\n  FORMAL: REMOVE_GROUP(smiles=\"",
                L("remove_group_step2"),
                '\") --> HEAVY_ATOMS(',
                L("remove_heavy"),
                ")",
            ),
            (
                "Step 3 [PRODUCT_CONSTRUCTION]: Remove ",
                L("remove_group_step2"),
                " next to atom ",
                L("anchor_idx"),
                ", producing the product encoded below.\n  FORMAL: SMILES + ANCHOR(idx=",
                L("anchor_idx"),
                ') + REMOVE_GROUP(smiles="',
                L("remove_group_step2"),
                '") --> PRODUCT_SMILES("',
                L("product"),
                '")',
            ),
            (
                "Step 4 [HEAVY_ATOM_VERIFICATION]: The source has ",
                L("source_heavy"),
                " heavy atoms and the product has ",
                L("product_heavy"),
                ", so the claimed delta is ",
                L("heavy_delta", signed=True),
                ".\n  FORMAL: SMILES[n_heavy=",
                L("source_heavy"),
                "] + PRODUCT_SMILES[n_heavy=",
                L("product_heavy"),
                "] --> HEAVY_ATOM_DELTA(",
                L("heavy_delta", signed=True),
                ")",
            ),
            (
                "Step 5 [RING_VERIFICATION]: The source has ",
                L("source_rings"),
                " rings and the product has ",
                L("product_rings"),
                ", so the claimed delta is ",
                L("ring_delta", signed=True),
                ".\n  FORMAL: SMILES[n_rings=",
                L("source_rings"),
                "] + PRODUCT_SMILES[n_rings=",
                L("product_rings"),
                "] --> RING_DELTA(",
                L("ring_delta", signed=True),
                ")",
            ),
        )
    return (
        (
            "Step 1 [ANCHOR_IDENTIFICATION]: At atom ",
            L("anchor_idx"),
            " (",
            L("anchor_element"),
            "), replace ",
            L("remove_group"),
            " with ",
            L("add_fragment"),
            ".\n  FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=",
            L("anchor_idx"),
            ', element="',
            L("anchor_element"),
            '") + REMOVE_GROUP(smiles="',
            L("remove_group"),
            '") + ADD_FRAGMENT(smiles="',
            L("add_fragment"),
            '")',
        ),
        (
            "Step 2 [REMOVE_GROUP_SIZE]: The removed group ",
            L("remove_group"),
            " contains ",
            L("remove_heavy"),
            " heavy atoms.\n  FORMAL: REMOVE_GROUP(smiles=\"",
            L("remove_group"),
            '\") --> REMOVE_HEAVY(',
            L("remove_heavy"),
            ")",
        ),
        (
            "Step 3 [ADD_FRAGMENT_SIZE]: The incoming fragment ",
            L("add_fragment"),
            " contains ",
            L("add_heavy"),
            " heavy atoms.\n  FORMAL: ADD_FRAGMENT(smiles=\"",
            L("add_fragment"),
            '\") --> ADD_HEAVY(',
            L("add_heavy"),
            ")",
        ),
        (
            "Step 4 [PRODUCT_CONSTRUCTION]: Replace ",
            L("remove_group"),
            " at atom ",
            L("anchor_idx"),
            " with ",
            L("add_fragment"),
            ", producing the product encoded below.\n  FORMAL: SMILES + ANCHOR(idx=",
            L("anchor_idx"),
            ') + REMOVE_GROUP("',
            L("remove_group"),
            '") + ADD_FRAGMENT("',
            L("add_fragment"),
            '") --> PRODUCT_SMILES("',
            L("product"),
            '")',
        ),
        (
            "Step 5 [HEAVY_ATOM_VERIFICATION]: The source has ",
            L("source_heavy"),
            " heavy atoms and the product has ",
            L("product_heavy"),
            ", so the claimed delta is ",
            L("heavy_delta", signed=True),
            ".\n  FORMAL: SMILES[n_heavy=",
            L("source_heavy"),
            "] + PRODUCT_SMILES[n_heavy=",
            L("product_heavy"),
            "] --> HEAVY_ATOM_DELTA(",
            L("heavy_delta", signed=True),
            ")",
        ),
        (
            "Step 6 [RING_VERIFICATION]: The source has ",
            L("source_rings"),
            " rings and the product has ",
            L("product_rings"),
            ", so the claimed delta is ",
            L("ring_delta", signed=True),
            ".\n  FORMAL: SMILES[n_rings=",
            L("source_rings"),
            "] + PRODUCT_SMILES[n_rings=",
            L("product_rings"),
            "] --> RING_DELTA(",
            L("ring_delta", signed=True),
            ")",
        ),
    )


def _split_step_parts(
    parts: tuple[str | _Literal, ...],
) -> tuple[tuple[str | _Literal, ...], tuple[str | _Literal, ...]]:
    natural: list[str | _Literal] = []
    formal: list[str | _Literal] = []
    found = False
    for part in parts:
        if not found and type(part) is str and _FORMAL_MARKER in part:
            before, after = part.split(_FORMAL_MARKER, 1)
            if before:
                natural.append(before)
            if after:
                formal.append(after)
            found = True
        elif found:
            formal.append(part)
        else:
            natural.append(part)
    if not found or not natural or not formal:
        raise ValueError("step template must contain one FORMAL boundary")
    return tuple(natural), tuple(formal)


def _placeholder_template(parts: tuple[str | _Literal, ...]) -> str:
    return "".join(
        part if type(part) is str else "{{" + part.node_id + "}}"
        for part in parts
    )


def _render_value_text(parts: tuple[str | _Literal, ...], graph: StateDAG) -> str:
    return "".join(
        part if type(part) is str else _value_text(graph, part)
        for part in parts
    )


def _literal_flags(parts: tuple[str | _Literal, ...]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for part in parts:
        if type(part) is str:
            continue
        previous = result.setdefault(part.node_id, part.signed)
        if previous != part.signed:
            raise ValueError(f"node {part.node_id!r} mixes signed and unsigned rendering")
    return result


def build_poe_rewrite_request(
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
) -> PoeRewriteRequest:
    """Build the label-free Poe input while keeping modified FORMAL locally owned."""

    _validate_render_input(artifact, injected)
    graph = injected.candidate_graph
    templates = _templates(artifact.normalized_subtask)
    if len(templates) != len(artifact.trace_steps):
        raise ValueError("renderer template count does not match the reference trace")
    steps = []
    for trace_step, parts in zip(artifact.trace_steps, templates, strict=True):
        natural_parts, formal_parts = _split_step_parts(parts)
        full_natural_template = _placeholder_template(natural_parts)
        prefix = f"Step {trace_step.step_index} [{trace_step.step_name}]: "
        if not full_natural_template.startswith(prefix):
            raise ValueError("natural template has an unexpected fixed step prefix")
        natural_template = full_natural_template[len(prefix) :]
        placeholder_counts = Counter(_PLACEHOLDER.findall(natural_template))
        flags = _literal_flags(natural_parts)
        placeholder_values = {
            node_id: _value_text(graph, _Literal(node_id, signed=signed))
            for node_id, signed in flags.items()
        }
        steps.append(
            PoeStepRewriteInput(
                step_index=trace_step.step_index,
                step_name=trace_step.step_name,
                original_natural_language=trace_step.natural_language,
                original_formal_ab=trace_step.formal_ab,
                modified_formal_ab=_render_value_text(formal_parts, graph),
                natural_template_draft=natural_template,
                placeholder_values=placeholder_values,
                required_placeholder_counts=placeholder_counts,
            )
        )
    return PoeRewriteRequest(
        origin_id=artifact.anonymous_sample_id,
        subtask=artifact.normalized_subtask.value,
        indexed_smiles=artifact.state_dag.values["source"].normalized_value,
        instruction=artifact.state_dag.values["instruction"].normalized_value,
        steps=tuple(steps),
    )


def _validate_render_input(
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
) -> None:
    if type(artifact) is not ReferenceDAGArtifact:
        raise TypeError("artifact must be ReferenceDAGArtifact")
    if type(injected) is not InjectedHallucination:
        raise TypeError("injected must be InjectedHallucination")
    if not artifact.state_dag.semantically_equals(injected.reference_graph):
        raise ValueError("artifact and injection must share the same reference graph")


def _render_placeholder_body(
    *,
    template: str,
    graph: StateDAG,
    signed_by_node: dict[str, bool],
    origin_id: str,
    edited_nodes: frozenset[str],
    step_index: int,
    reasoning_offset: int,
    occurrence_counts: dict[str, int],
) -> tuple[str, tuple[RenderedMention, ...]]:
    buffer = ""
    mentions = []
    cursor = 0
    for match in _PLACEHOLDER.finditer(template):
        buffer += template[cursor : match.start()]
        node_id = match.group(1)
        if node_id not in signed_by_node:
            raise ValueError(f"natural template contains unknown placeholder {node_id!r}")
        value = _value_text(graph, _Literal(node_id, signed_by_node[node_id]))
        start = reasoning_offset + len(buffer)
        buffer += value
        end = reasoning_offset + len(buffer)
        occurrence_counts[node_id] = occurrence_counts.get(node_id, 0) + 1
        mentions.append(
            RenderedMention(
                mention_id=(
                    f"{origin_id}.reasoning.step{step_index:02d}."
                    f"{node_id}.{occurrence_counts[node_id]:02d}"
                ),
                component="reasoning_chain",
                node_id=node_id,
                step_index=step_index,
                start=start,
                end=end,
                value=value,
                hallucinated=node_id in edited_nodes,
            )
        )
        cursor = match.end()
    buffer += template[cursor:]
    return buffer, tuple(mentions)


def _assemble_rendered_text(
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
    natural_templates: tuple[str, ...],
    realization: dict[str, Any],
) -> RenderedHallucination:
    _validate_render_input(artifact, injected)
    graph = injected.candidate_graph
    edited_nodes = frozenset(injected.plan.edited_node_ids)
    step_definitions = _templates(artifact.normalized_subtask)
    if len(natural_templates) != len(step_definitions):
        raise ValueError("natural template count does not match the trace")

    step_texts: list[str] = []
    mentions: list[RenderedMention] = []
    occurrence_counts: dict[str, int] = {}
    reasoning_length = 0
    for trace_step, parts, natural_template in zip(
        artifact.trace_steps,
        step_definitions,
        natural_templates,
        strict=True,
    ):
        if step_texts:
            reasoning_length += 2
        natural_parts, formal_parts = _split_step_parts(parts)
        prefix = f"Step {trace_step.step_index} [{trace_step.step_name}]: "
        natural_body, natural_mentions = _render_placeholder_body(
            template=natural_template,
            graph=graph,
            signed_by_node=_literal_flags(natural_parts),
            origin_id=artifact.anonymous_sample_id,
            edited_nodes=edited_nodes,
            step_index=trace_step.step_index,
            reasoning_offset=reasoning_length + len(prefix),
            occurrence_counts=occurrence_counts,
        )
        natural_text = prefix + natural_body
        formal_text, formal_mentions = _add_step(
            origin_id=artifact.anonymous_sample_id,
            graph=graph,
            edited_nodes=edited_nodes,
            step_index=trace_step.step_index,
            parts=formal_parts,
            reasoning_offset=(
                reasoning_length + len(natural_text) + len(_FORMAL_MARKER)
            ),
            occurrence_counts=occurrence_counts,
        )
        step_text = natural_text + _FORMAL_MARKER + formal_text
        step_texts.append(step_text)
        mentions.extend(natural_mentions)
        mentions.extend(formal_mentions)
        reasoning_length += len(step_text)

    reasoning_chain = "\n\n".join(step_texts)
    final_answer = str(graph.values["final_answer"].normalized_value)
    mentions.append(
        RenderedMention(
            mention_id=f"{artifact.anonymous_sample_id}.final_answer.final_answer.01",
            component="final_answer",
            node_id="final_answer",
            step_index=None,
            start=0,
            end=len(final_answer),
            value=final_answer,
            hallucinated="final_answer" in edited_nodes,
        )
    )
    return RenderedHallucination(
        reasoning_chain=reasoning_chain,
        final_answer=final_answer,
        step_texts=tuple(step_texts),
        mentions=tuple(mentions),
        realization=realization,
    )


class DeterministicTextRenderer:
    """Offline validator fixture; production dataset generation uses PoeTextRenderer."""

    def render(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
    ) -> RenderedHallucination:
        request = build_poe_rewrite_request(artifact, injected)
        return _assemble_rendered_text(
            artifact,
            injected,
            tuple(step.natural_template_draft for step in request.steps),
            {
                "backend": "deterministic_test_renderer",
                "provider": None,
                "network_request_count": 0,
            },
        )


class PoeTextRenderer:
    """Production renderer: Poe rewrites prose; local code owns values and FORMAL."""

    def __init__(self, agent: PoeStepTextAgent) -> None:
        if type(agent) is not PoeStepTextAgent:
            raise TypeError("agent must be PoeStepTextAgent")
        self.agent = agent

    def render(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
    ) -> RenderedHallucination:
        request = build_poe_rewrite_request(artifact, injected)
        result = self.agent.rewrite(request)
        return _assemble_rendered_text(
            artifact,
            injected,
            result.natural_templates,
            {
                "backend": "poe_agent",
                "provider": "poe",
                "bot_name": result.bot_name,
                "api_key_environment_variable": result.api_key_environment_variable,
                "prompt_sha256": result.prompt_sha256,
                "response_sha256": result.response_sha256,
                "network_request_count": result.network_request_count,
                "cache_hit": result.cache_hit,
            },
        )


__all__ = [
    "DeterministicTextRenderer",
    "PoeTextRenderer",
    "build_poe_rewrite_request",
]
