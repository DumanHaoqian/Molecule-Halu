"""Render complete reasoning text and exact hallucination mentions from one graph."""

from __future__ import annotations

import re
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
    AffectedNodeClaim,
    FORMAL_MARKER,
    PoeRewriteRequest,
    RequiredHallucinationOccurrence,
    PoeStepRewriteInput,
    PoeStepTextAgent,
    PoeTextRealizationError,
    StepRewriteMode,
    parse_hallucination_markers,
    strip_hallucination_markers,
    validate_rewritten_step_text,
)
from .occurrence_audit import (
    loose_occurrence_spans,
    requires_derivation_rewrite,
)


_FORMAL_MARKER = FORMAL_MARKER


@dataclass(frozen=True, slots=True)
class _Literal:
    node_id: str
    signed: bool = False


@dataclass(frozen=True, slots=True)
class _StepRewriteContract:
    mode: StepRewriteMode
    occurrences: tuple[RequiredHallucinationOccurrence, ...]
    affected_claims: tuple[AffectedNodeClaim, ...]


def _value_text(graph: StateDAG, literal: _Literal) -> str:
    value = graph.values[literal.node_id].normalized_value
    if literal.signed and type(value) is int and value > 0:
        return f"+{value}"
    return str(value)


def _add_step(
    *,
    origin_id: str,
    graph: StateDAG,
    causal_roles_by_node: dict[str, Any],
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
                hallucinated=part.node_id in causal_roles_by_node,
                causal_role=causal_roles_by_node.get(part.node_id),
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


def _captured_spans(pattern: str, text: str) -> set[tuple[int, int]]:
    return {
        match.span("value")
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }


def _original_occurrence_spans(
    node_id: str,
    before_text: str,
    natural_body: str,
) -> tuple[tuple[int, int], ...]:
    """Locate node-specific natural-language mentions in ChemCoT's fixed steps."""

    escaped = re.escape(before_text)
    numeric_nodes = {
        "anchor_idx",
        "source_heavy",
        "product_heavy",
        "heavy_delta",
        "source_rings",
        "product_rings",
        "ring_delta",
        "fragment_heavy",
        "remove_heavy",
        "add_heavy",
    }
    needle = rf"(?<!\d){escaped}" if node_id in numeric_nodes else escaped
    spans: set[tuple[int, int]] = set()
    if node_id == "anchor_idx":
        spans |= _captured_spans(
            rf"\b(?:atom|idx|index|map)\s*(?:=|:)?\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:is|atom|anchor)\s+[A-Z][a-z]?"
            rf"(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\[\s*[A-Za-z]+[^\]]*:\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
    elif node_id == "anchor_element":
        spans |= _captured_spans(
            rf"\belement\s*(?:=|:)?\s*[\"']?(?P<value>{needle})(?![a-z])",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\(\s*(?P<value>{needle})\s*\)",
            natural_body,
        )
    elif node_id in {"source_heavy", "product_heavy"}:
        subject = "source" if node_id == "source_heavy" else "product"
        spans |= _captured_spans(
            rf"\b{subject}\b[^,.;\n]{{0,120}}?\b(?:has|contains)\s+"
            rf"(?P<value>{needle})\s+heavy\s+atoms?",
            natural_body,
        )
        if node_id == "source_heavy":
            spans |= _captured_spans(
                rf"(?:max(?:imum)?\s+(?:atom[- ]?map|map)|map)\s*(?:number)?\s*:"
                rf"?\s*(?P<value>{needle})(?!\d)",
                natural_body,
            )
        equation_context = (
            r"(?:HEAVY_ATOM_DELTA|net\s+change|change\s+in\s+heavy\s+atoms|"
            r"check)"
        )
        if node_id == "product_heavy":
            spans |= _captured_spans(
                rf"{equation_context}[^.\n]{{0,120}}?(?:is|=|:)\s*"
                rf"(?P<value>{needle})\s*-",
                natural_body,
            )
        else:
            spans |= _captured_spans(
                rf"{equation_context}[^.\n]{{0,120}}?(?:is|=|:)\s*"
                rf"\d+\s*-\s*(?P<value>{needle})\s*=",
                natural_body,
            )
    elif node_id in {"source_rings", "product_rings"}:
        subject = "source" if node_id == "source_rings" else "product"
        spans |= _captured_spans(
            rf"\b{subject}\b[^,.;\n]{{0,120}}?\b(?:has|contains)\s+"
            rf"(?P<value>{needle})\s+rings?",
            natural_body,
        )
        equation_context = r"(?:RING_DELTA|net\s+change(?:\s+in\s+rings)?|check)"
        if node_id == "product_rings":
            spans |= _captured_spans(
                rf"{equation_context}[^.\n]{{0,120}}?(?:is|=|:)\s*"
                rf"(?P<value>{needle})\s*-",
                natural_body,
            )
        else:
            spans |= _captured_spans(
                rf"{equation_context}[^.\n]{{0,120}}?(?:is|=|:)\s*"
                rf"\d+\s*-\s*(?P<value>{needle})\s*=",
                natural_body,
            )
    elif node_id in {"heavy_delta", "ring_delta"}:
        label = "HEAVY_ATOM_DELTA" if node_id == "heavy_delta" else "RING_DELTA"
        spans |= _captured_spans(
            rf"\b{label}\b[^.\n]*?=\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:claimed\s+delta|net\s+change|change\s+in\s+"
            rf"(?:heavy\s+atoms|rings))\b[^.\n]*?=\s*"
            rf"(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:expected\s+(?:difference|change)|matches|cross-check)\b"
            rf"[^.\n]*?=\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:claimed\s+delta|net\s+change(?:\s+in\s+rings)?)\b"
            rf"[^.\n]{{0,80}}?\bis\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
    elif node_id == "fragment_heavy":
        spans |= _captured_spans(
            rf"\b(?:ADD_FRAGMENT|fragment)\b[^.\n]{{0,180}}?"
            rf"(?:has|contains|consists\s+of)\s+(?P<value>{needle})\s+"
            rf"heavy\s+atoms?",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:fragment\s+heavy(?:-atom)?\s+count|total\s+k|k)\b"
            rf"\s*(?:=|is|:)\s*(?P<value>{needle})(?!\d)",
            natural_body,
        )
    elif node_id == "remove_heavy":
        spans |= _captured_spans(
            rf"\b(?:REMOVE_HEAVY|k_remove)\b\s*(?:=|\()\s*"
            rf"(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\bremoved\s+group\b\s*\(\s*(?P<value>{needle})\s*\)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:matches|cross-check|equals|difference)\b[^.\n]{{0,120}}?"
            rf"\d+\s*-\s*(?P<value>{needle})(?=\s*(?:=|[).,]))",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:leaving\s+group|REMOVE_GROUP)\b[^.\n]{{0,180}}?"
            rf"(?:has|contains|consists\s+of(?:\s+exactly)?)\s+"
            rf"(?P<value>{needle})\s+heavy\s+atoms?",
            natural_body,
        )
    elif node_id == "add_heavy":
        spans |= _captured_spans(
            rf"\b(?:ADD_HEAVY|k_add)\b\s*(?:=|\()\s*"
            rf"(?P<value>{needle})(?!\d)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\badded\s+fragment\b\s*\(\s*(?P<value>{needle})\s*\)",
            natural_body,
        )
        spans |= _captured_spans(
            rf"\b(?:matches|cross-check|equals|difference)\b[^.\n]{{0,120}}?"
            rf"(?P<value>{needle})\s*-\s*\d+",
            natural_body,
        )
    else:
        spans |= {
            match.span()
            for match in re.finditer(re.escape(before_text), natural_body)
        }
    return tuple(sorted(spans))


def _step_rewrite_contract(
    *,
    original_step_text: str,
    prefix: str,
    step_name: str,
    natural_parts: tuple[str | _Literal, ...],
    reference_graph: StateDAG,
    candidate_graph: StateDAG,
    changed_nodes: frozenset[str],
) -> _StepRewriteContract:
    """Audit strict recall and choose COPY, PATCH, or full derivation rewrite."""

    original_head, _ = original_step_text.split(_FORMAL_MARKER, 1)
    natural_body = original_head[len(prefix) :]
    literal_flags = _literal_flags(natural_parts)
    # Audit every changed node against every step. Templates define canonical
    # rendering and signed surface forms; they must never limit recall.
    nodes_to_scan = changed_nodes
    requirements = []
    affected_claims = []
    incomplete_coverage = False
    for node_id in sorted(nodes_to_scan):
        if node_id not in reference_graph.values or node_id not in candidate_graph.values:
            continue
        signed = literal_flags.get(node_id, node_id in {"heavy_delta", "ring_delta"})
        before = _value_text(reference_graph, _Literal(node_id, signed=signed))
        after = _value_text(candidate_graph, _Literal(node_id, signed=signed))
        if before == after:
            continue
        strict_spans = set(_original_occurrence_spans(node_id, before, natural_body))
        loose_spans = strict_spans | set(
            loose_occurrence_spans(
                node_id,
                before,
                natural_body,
                step_name=step_name,
            )
        )
        if not loose_spans:
            continue
        affected_claims.append(
            AffectedNodeClaim(
                node_id=node_id,
                before_text=before,
                after_text=after,
            )
        )
        if loose_spans - strict_spans:
            incomplete_coverage = True
        for occurrence_index, (start, end) in enumerate(
            sorted(strict_spans),
            start=1,
        ):
            requirements.append(
                RequiredHallucinationOccurrence(
                    occurrence_id=f"{node_id}.{occurrence_index:02d}",
                    node_id=node_id,
                    before_text=natural_body[start:end],
                    after_text=after,
                    original_start=start,
                    original_end=end,
                )
            )
    affected_node_ids = frozenset(item.node_id for item in affected_claims)
    ordered_requirements = sorted(
        requirements,
        key=lambda item: (item.original_start, item.original_end, item.node_id),
    )
    for previous, current in zip(
        ordered_requirements,
        ordered_requirements[1:],
    ):
        if current.original_start < previous.original_end:
            # One source substring cannot carry nested markers for two DAG nodes.
            # Route it to derivation rewrite, whose canonical body can render one
            # explicit occurrence per affected claim.
            incomplete_coverage = True
    if affected_claims and (
        incomplete_coverage
        or requires_derivation_rewrite(natural_body, affected_node_ids)
    ):
        return _StepRewriteContract(
            mode=StepRewriteMode.DERIVATION_REWRITE,
            occurrences=(),
            affected_claims=tuple(affected_claims),
        )
    if requirements:
        return _StepRewriteContract(
            mode=StepRewriteMode.OCCURRENCE_PATCH,
            occurrences=tuple(requirements),
            affected_claims=tuple(affected_claims),
        )
    return _StepRewriteContract(
        mode=StepRewriteMode.COPY,
        occurrences=(),
        affected_claims=(),
    )


def build_poe_rewrite_request(
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
) -> PoeRewriteRequest:
    """Pair every original complete step_text with its locally modified FORMAL."""

    _validate_render_input(artifact, injected)
    graph = injected.candidate_graph
    changed_nodes = frozenset(injected.changed_node_ids)
    templates = _templates(artifact.normalized_subtask)
    if len(templates) != len(artifact.trace_steps):
        raise ValueError("renderer template count does not match the reference trace")
    steps = []
    for trace_step, parts in zip(artifact.trace_steps, templates, strict=True):
        natural_parts, formal_parts = _split_step_parts(parts)
        original_step_text = trace_step.render(include_answer=False)
        prefix = f"Step {trace_step.step_index} [{trace_step.step_name}]: "
        contract = _step_rewrite_contract(
            original_step_text=original_step_text,
            prefix=prefix,
            step_name=trace_step.step_name,
            natural_parts=natural_parts,
            reference_graph=artifact.state_dag,
            candidate_graph=graph,
            changed_nodes=changed_nodes,
        )
        steps.append(
            PoeStepRewriteInput(
                step_index=trace_step.step_index,
                step_name=trace_step.step_name,
                original_step_text=original_step_text,
                modified_formal_ab=_render_value_text(formal_parts, graph),
                required_hallucination_occurrences=contract.occurrences,
                rewrite_mode=contract.mode,
                affected_node_claims=contract.affected_claims,
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


def _rewrite_contract_metadata(request: PoeRewriteRequest) -> list[dict[str, Any]]:
    """Release the per-step routing decision for later dataset audits."""

    return [
        {
            "step_index": step.step_index,
            "step_name": step.step_name,
            "mode": step.rewrite_mode.value,
            "affected_node_claims": [
                item.to_prompt_dict() for item in step.affected_node_claims
            ],
            "required_occurrence_ids": [
                item.occurrence_id
                for item in step.required_hallucination_occurrences
            ],
        }
        for step in request.steps
    ]


def _render_marked_natural_body(
    *,
    marked_natural_body: str,
    origin_id: str,
    step_index: int,
    reasoning_offset: int,
    occurrence_counts: dict[str, int],
    expected: PoeStepRewriteInput,
    causal_roles_by_node: dict[str, Any],
) -> tuple[str, tuple[RenderedMention, ...]]:
    """Strip Poe markers and convert their exact locations into mentions."""

    parsed_markers = parse_hallucination_markers(marked_natural_body)
    clean_body = strip_hallucination_markers(marked_natural_body)

    # Do not mistake an old value embedded in the planned replacement itself
    # (for example ``5`` inside ``15``) for an unmodified occurrence.  Mask all
    # correctly marked replacements first, then search the remaining prose for
    # every node's old surface value using the same node-aware inventory rules.
    stale_search_body = list(clean_body)
    for _, _, start, end in parsed_markers:
        stale_search_body[start:end] = " " * (end - start)
    stale_search_body_text = "".join(stale_search_body)

    claims_to_check = list(expected.affected_node_claims)
    if not claims_to_check:
        claims_to_check = [
            AffectedNodeClaim(
                node_id=requirement.node_id,
                before_text=requirement.before_text,
                after_text=requirement.after_text,
            )
            for requirement in expected.required_hallucination_occurrences
        ]
    checked: set[tuple[str, str]] = set()
    for claim in claims_to_check:
        key = (claim.node_id, claim.before_text)
        if key in checked:
            continue
        checked.add(key)
        if loose_occurrence_spans(
            claim.node_id,
            claim.before_text,
            stale_search_body_text,
            step_name=expected.step_name,
        ):
            raise PoeTextRealizationError(
                "rewritten natural language retained a stale value for "
                f"{claim.node_id!r}"
            )
    mentions = []
    requirements = {
        item.occurrence_id: item
        for item in expected.required_hallucination_occurrences
    }
    for occurrence_id, value, local_start, local_end in parsed_markers:
        node_id = occurrence_id.rsplit(".", 1)[0]
        if (
            expected.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH
            and requirements[occurrence_id].node_id != node_id
        ):
            raise PoeTextRealizationError("marker namespace disagrees with its DAG node")
        start = reasoning_offset + local_start
        end = reasoning_offset + local_end
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
                hallucinated=True,
                causal_role=causal_roles_by_node[node_id],
            )
        )
    return clean_body, tuple(mentions)


def _render_deterministic_derivation_body(
    expected: PoeStepRewriteInput,
    natural_parts: tuple[str | _Literal, ...],
    graph: StateDAG,
) -> str:
    """Render a coherent fixture body with one marker per claim occurrence."""

    prefix = f"Step {expected.step_index} [{expected.step_name}]: "
    claims = {claim.node_id: claim for claim in expected.affected_node_claims}
    occurrence_counts: dict[str, int] = {}
    marked_head = ""
    for part in natural_parts:
        if type(part) is str:
            marked_head += part
            continue
        value = _value_text(graph, part)
        claim = claims.get(part.node_id)
        if claim is None:
            marked_head += value
            continue
        if value != claim.after_text:
            raise ValueError("canonical derivation value disagrees with affected claim")
        occurrence_counts[part.node_id] = occurrence_counts.get(part.node_id, 0) + 1
        occurrence_id = f"{part.node_id}.{occurrence_counts[part.node_id]:02d}"
        marked_head += f"[[HALLU:{occurrence_id}]]{value}[[/HALLU]]"
    if not marked_head.startswith(prefix):
        raise ValueError("canonical natural template changed the Step header")
    marked_body = marked_head[len(prefix) :]
    for node_id, claim in sorted(claims.items()):
        if occurrence_counts.get(node_id, 0):
            continue
        occurrence_counts[node_id] = 1
        marked_body += (
            f"\n  The value of {node_id} is "
            f"[[HALLU:{node_id}.01]]{claim.after_text}[[/HALLU]]."
        )
    return marked_body


def _assemble_rendered_text(
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
    request: PoeRewriteRequest,
    rewritten_step_texts: tuple[str, ...],
    realization: dict[str, Any],
) -> RenderedHallucination:
    _validate_render_input(artifact, injected)
    graph = injected.candidate_graph
    changed_nodes = frozenset(injected.changed_node_ids)
    causal_roles_by_node = dict(injected.causal_roles_by_node)
    step_definitions = _templates(artifact.normalized_subtask)
    if request.origin_id != artifact.anonymous_sample_id:
        raise ValueError("Poe request does not belong to this reference artifact")
    if not (
        len(rewritten_step_texts)
        == len(request.steps)
        == len(step_definitions)
        == len(artifact.trace_steps)
    ):
        raise ValueError("rewritten step count does not match the trace")

    step_texts: list[str] = []
    mentions: list[RenderedMention] = []
    occurrence_counts: dict[str, int] = {}
    reasoning_length = 0
    for trace_step, parts, expected, rewritten_step_text in zip(
        artifact.trace_steps,
        step_definitions,
        request.steps,
        rewritten_step_texts,
        strict=True,
    ):
        if step_texts:
            reasoning_length += 2
        _, formal_parts = _split_step_parts(parts)
        prefix = f"Step {trace_step.step_index} [{trace_step.step_name}]: "
        validate_rewritten_step_text(rewritten_step_text, expected)
        marked_head, returned_formal = rewritten_step_text.split(_FORMAL_MARKER, 1)
        locally_rendered_formal = _render_value_text(formal_parts, graph)
        if returned_formal != locally_rendered_formal:
            raise ValueError("validated Poe FORMAL differs from the candidate DAG")
        marked_natural_body = marked_head[len(prefix) :]
        natural_body, natural_mentions = _render_marked_natural_body(
            marked_natural_body=marked_natural_body,
            origin_id=artifact.anonymous_sample_id,
            step_index=trace_step.step_index,
            reasoning_offset=reasoning_length + len(prefix),
            occurrence_counts=occurrence_counts,
            expected=expected,
            causal_roles_by_node=causal_roles_by_node,
        )
        natural_text = prefix + natural_body
        formal_text, formal_mentions = _add_step(
            origin_id=artifact.anonymous_sample_id,
            graph=graph,
            causal_roles_by_node=causal_roles_by_node,
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
            hallucinated="final_answer" in changed_nodes,
            causal_role=causal_roles_by_node.get("final_answer"),
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
        graph = injected.candidate_graph
        rewritten_steps = []
        for expected, parts in zip(
            request.steps,
            _templates(artifact.normalized_subtask),
            strict=True,
        ):
            natural_parts, _ = _split_step_parts(parts)
            prefix = f"Step {expected.step_index} [{expected.step_name}]: "
            if expected.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
                marked_body = _render_deterministic_derivation_body(
                    expected,
                    natural_parts,
                    graph,
                )
            else:
                marked_head = expected.original_step_text.split(_FORMAL_MARKER, 1)[0]
                marked_body = marked_head[len(prefix) :]
                for occurrence in sorted(
                    expected.required_hallucination_occurrences,
                    key=lambda item: item.original_start,
                    reverse=True,
                ):
                    if (
                        marked_body[occurrence.original_start : occurrence.original_end]
                        != occurrence.before_text
                    ):
                        raise ValueError(
                            "required occurrence no longer matches original text"
                        )
                    marker = (
                        f"[[HALLU:{occurrence.occurrence_id}]]"
                        f"{occurrence.after_text}[[/HALLU]]"
                    )
                    marked_body = (
                        marked_body[: occurrence.original_start]
                        + marker
                        + marked_body[occurrence.original_end :]
                    )
            marked_head = prefix + marked_body
            rewritten_steps.append(
                marked_head + _FORMAL_MARKER + expected.modified_formal_ab
            )
        return _assemble_rendered_text(
            artifact,
            injected,
            request,
            tuple(rewritten_steps),
            {
                "backend": "deterministic_test_renderer",
                "provider": None,
                "network_request_count": 0,
                "rewrite_mode": "audited_step_routing",
                "step_rewrite_modes": [
                    step.rewrite_mode.value for step in request.steps
                ],
                "step_rewrite_contracts": _rewrite_contract_metadata(request),
                "annotation_protocol": "temporary_hallu_markers",
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
            request,
            result.rewritten_step_texts,
            {
                "backend": "poe_agent",
                "provider": "poe",
                "bot_name": result.bot_name,
                "api_key_environment_variable": result.api_key_environment_variable,
                "prompt_sha256": result.prompt_sha256,
                "response_sha256": result.response_sha256,
                "network_request_count": result.network_request_count,
                "cache_hit": result.cache_hit,
                "rewrite_mode": "audited_step_routing",
                "step_rewrite_modes": [
                    step.rewrite_mode.value for step in request.steps
                ],
                "step_rewrite_contracts": _rewrite_contract_metadata(request),
                "annotation_protocol": "temporary_hallu_markers",
            },
        )


__all__ = [
    "DeterministicTextRenderer",
    "PoeTextRenderer",
    "build_poe_rewrite_request",
]
