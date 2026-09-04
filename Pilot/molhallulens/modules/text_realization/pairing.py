"""Construct a truth-valued control from an already validated H rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from molhallulens.core import InjectedHallucination, RenderedHallucination, RenderedMention
from molhallulens.modules.reference import ReferenceDAGArtifact

from .occurrence_audit import arithmetic_violations, loose_occurrence_spans
from .poe_agent import (
    AffectedNodeClaim,
    FORMAL_MARKER,
    PoeRewriteRequest,
    PoeStepRewriteInput,
    PoeStepTextAgent,
    PoeTextRealizationError,
    RequiredHallucinationOccurrence,
    StepRewriteMode,
    parse_hallucination_markers,
    strip_hallucination_markers,
)
from .renderer import build_poe_rewrite_request


class PairAlignment(StrEnum):
    """How a negative step was aligned to its hallucinated partner."""

    BYTE_IDENTICAL = "byte_identical"
    REGENERATED = "regenerated"


@dataclass(frozen=True, slots=True)
class StepPairAlignment:
    step_index: int
    step_name: str
    rewrite_mode: StepRewriteMode
    pair_alignment: PairAlignment

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "rewrite_mode": self.rewrite_mode.value,
            "pair_alignment": self.pair_alignment.value,
        }


@dataclass(frozen=True, slots=True)
class MatchedRenderedPair:
    """Validated H/N renderings plus per-step alignment provenance."""

    hallucinated: RenderedHallucination
    negative: RenderedHallucination
    step_pair_alignment: tuple[StepPairAlignment, ...]

    def __post_init__(self) -> None:
        if type(self.hallucinated) is not RenderedHallucination:
            raise TypeError("hallucinated must be RenderedHallucination")
        if type(self.negative) is not RenderedHallucination:
            raise TypeError("negative must be RenderedHallucination")
        alignments = tuple(self.step_pair_alignment)
        if len(alignments) != len(self.hallucinated.step_texts):
            raise ValueError("every reasoning step requires pair alignment metadata")
        if any(type(item) is not StepPairAlignment for item in alignments):
            raise TypeError("step_pair_alignment contains an invalid value")
        object.__setattr__(self, "step_pair_alignment", alignments)


@dataclass(frozen=True, slots=True)
class _LocalControl:
    mention_id: str
    node_id: str
    step_index: int | None
    start: int
    end: int
    value: str


def _surface_truth(artifact: ReferenceDAGArtifact, mention: RenderedMention) -> str:
    value = artifact.state_dag.values[mention.node_id].normalized_value
    if (
        mention.node_id in {"heavy_delta", "ring_delta"}
        and type(value) is int
        and value > 0
    ):
        return f"+{value}"
    return str(value)


def _replace_step_mentions(
    step_text: str,
    *,
    reasoning_start: int,
    mentions: tuple[RenderedMention, ...],
    artifact: ReferenceDAGArtifact,
) -> tuple[str, tuple[_LocalControl, ...]]:
    """Replace H values in one step and retain shifted local control offsets."""

    pieces: list[str] = []
    controls: list[_LocalControl] = []
    cursor = 0
    output_length = 0
    for mention in sorted(mentions, key=lambda item: (item.start, item.end)):
        local_start = mention.start - reasoning_start
        local_end = mention.end - reasoning_start
        if local_start < cursor or step_text[local_start:local_end] != mention.value:
            raise PoeTextRealizationError(
                "H mention offsets overlap or do not round-trip during pair construction"
            )
        prefix = step_text[cursor:local_start]
        pieces.append(prefix)
        output_length += len(prefix)
        truth = _surface_truth(artifact, mention)
        start = output_length
        pieces.append(truth)
        output_length += len(truth)
        controls.append(
            _LocalControl(
                mention_id=mention.mention_id,
                node_id=mention.node_id,
                step_index=mention.step_index,
                start=start,
                end=output_length,
                value=truth,
            )
        )
        cursor = local_end
    pieces.append(step_text[cursor:])
    return "".join(pieces), tuple(controls)


def _masked_text(text: str, controls: tuple[_LocalControl, ...]) -> str:
    characters = list(text)
    for control in controls:
        characters[control.start : control.end] = " " * (control.end - control.start)
    return "".join(characters)


def _natural_h_controls(
    *,
    step_text: str,
    reasoning_start: int,
    mentions: tuple[RenderedMention, ...],
    prefix: str,
) -> tuple[_LocalControl, ...]:
    """Project global H mention offsets into one natural-language body."""

    head = step_text.split(FORMAL_MARKER, 1)[0]
    controls = []
    for mention in mentions:
        step_start = mention.start - reasoning_start
        step_end = mention.end - reasoning_start
        if step_start >= len(head):
            continue
        if step_start < len(prefix):
            raise PoeTextRealizationError(
                "H natural-language mention overlaps the locked Step header"
            )
        controls.append(
            _LocalControl(
                mention_id=mention.mention_id,
                node_id=mention.node_id,
                step_index=mention.step_index,
                start=step_start - len(prefix),
                end=step_end - len(prefix),
                value=mention.value,
            )
        )
    return tuple(controls)


def _step_validation_errors(
    *,
    artifact: ReferenceDAGArtifact,
    injected: InjectedHallucination,
    expected: PoeStepRewriteInput,
    hallucinated_step: str,
    hallucinated_controls: tuple[_LocalControl, ...],
    step_text: str,
    controls: tuple[_LocalControl, ...],
) -> tuple[str, ...]:
    """Validate a swapped/regenerated truth step without weakening H checks."""

    errors: list[str] = []
    prefix = f"Step {expected.step_index} [{expected.step_name}]: "
    if not step_text.startswith(prefix) or step_text.count(FORMAL_MARKER) != 1:
        return ("step_header_or_formal_boundary",)
    head, formal = step_text.split(FORMAL_MARKER, 1)
    reference_formal = artifact.trace_steps[expected.step_index - 1].formal_ab
    if formal != reference_formal:
        errors.append("reference_formal_mismatch")
    natural_body = head[len(prefix) :]
    natural_controls = tuple(
        _LocalControl(
            mention_id=item.mention_id,
            node_id=item.node_id,
            step_index=item.step_index,
            start=item.start - len(prefix),
            end=item.end - len(prefix),
            value=item.value,
        )
        for item in controls
        if len(prefix) <= item.start < len(head)
    )
    search_body = _masked_text(natural_body, natural_controls)
    h_head = hallucinated_step.split(FORMAL_MARKER, 1)[0]
    h_natural_body = h_head[len(prefix) :]
    h_search_body = _masked_text(h_natural_body, hallucinated_controls)
    for node_id in injected.changed_node_ids:
        candidate_value = injected.candidate_graph.values[node_id].normalized_value
        candidate_text = str(candidate_value)
        if (
            node_id in {"heavy_delta", "ring_delta"}
            and type(candidate_value) is int
            and candidate_value > 0
        ):
            candidate_text = f"+{candidate_value}"
        remaining = loose_occurrence_spans(
            node_id,
            candidate_text,
            search_body,
            step_name=expected.step_name,
        )
        baseline = loose_occurrence_spans(
            node_id,
            candidate_text,
            h_search_body,
            step_name=expected.step_name,
        )
        if len(remaining) > len(baseline):
            errors.append(f"stale_candidate_value:{node_id}")
    if arithmetic_violations(natural_body):
        errors.append("false_arithmetic")
    return tuple(errors)


def _reverse_request(
    *,
    artifact: ReferenceDAGArtifact,
    original_request: PoeRewriteRequest,
    hallucinated: RenderedHallucination,
    target_step_index: int,
    target_h_mentions: tuple[RenderedMention, ...],
) -> PoeRewriteRequest:
    """Build one reverse-direction request while preserving the original mode."""

    h_step_starts: dict[int, int] = {}
    offset = 0
    for step_index, step_text in enumerate(hallucinated.step_texts, start=1):
        h_step_starts[step_index] = offset
        offset += len(step_text) + 2

    steps: list[PoeStepRewriteInput] = []
    for expected, h_step, reference_step in zip(
        original_request.steps,
        hallucinated.step_texts,
        artifact.trace_steps,
        strict=True,
    ):
        if expected.step_index != target_step_index:
            steps.append(
                PoeStepRewriteInput(
                    step_index=expected.step_index,
                    step_name=expected.step_name,
                    original_step_text=h_step,
                    modified_formal_ab=reference_step.formal_ab,
                    required_hallucination_occurrences=(),
                    rewrite_mode=StepRewriteMode.COPY,
                )
            )
            continue

        claims_by_node = {
            item.node_id: item for item in expected.affected_node_claims
        }
        natural_mentions_by_node: dict[str, list[RenderedMention]] = defaultdict(list)
        head = h_step.split(FORMAL_MARKER, 1)[0]
        step_start = h_step_starts[expected.step_index]
        for mention in sorted(target_h_mentions, key=lambda item: item.start):
            local_start = mention.start - step_start
            if local_start < len(head):
                natural_mentions_by_node[mention.node_id].append(mention)

        if expected.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH:
            prefix = f"Step {expected.step_index} [{expected.step_name}]: "
            requirements: list[RequiredHallucinationOccurrence] = []
            for node_id, mentions in sorted(natural_mentions_by_node.items()):
                claim = claims_by_node[node_id]
                for occurrence_index, mention in enumerate(mentions, start=1):
                    start = mention.start - step_start - len(prefix)
                    requirements.append(
                        RequiredHallucinationOccurrence(
                            occurrence_id=f"{node_id}.{occurrence_index:02d}",
                            node_id=node_id,
                            before_text=claim.after_text,
                            after_text=claim.before_text,
                            original_start=start,
                            original_end=start + len(claim.after_text),
                        )
                    )
            reverse_claims = tuple(
                AffectedNodeClaim(
                    node_id=claim.node_id,
                    before_text=claim.after_text,
                    after_text=claim.before_text,
                )
                for claim in expected.affected_node_claims
            )
        elif expected.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
            requirements = []
            reverse_claims = tuple(
                AffectedNodeClaim(
                    node_id=claim.node_id,
                    before_text=claim.after_text,
                    after_text=claim.before_text,
                    required_occurrence_count=len(
                        natural_mentions_by_node[claim.node_id]
                    ),
                )
                for claim in expected.affected_node_claims
            )
        else:
            raise PoeTextRealizationError(
                "a COPY step cannot require matched-negative regeneration"
            )
        steps.append(
            PoeStepRewriteInput(
                step_index=expected.step_index,
                step_name=expected.step_name,
                original_step_text=h_step,
                modified_formal_ab=reference_step.formal_ab,
                required_hallucination_occurrences=tuple(requirements),
                rewrite_mode=expected.rewrite_mode,
                affected_node_claims=reverse_claims,
            )
        )

    return PoeRewriteRequest(
        origin_id=(
            f"{original_request.origin_id}__matched_negative_step{target_step_index:02d}"
        ),
        subtask=original_request.subtask,
        indexed_smiles=original_request.indexed_smiles,
        instruction=original_request.instruction,
        steps=tuple(steps),
    )


def _regenerated_step(
    *,
    artifact: ReferenceDAGArtifact,
    original_request: PoeRewriteRequest,
    hallucinated: RenderedHallucination,
    expected: PoeStepRewriteInput,
    h_mentions: tuple[RenderedMention, ...],
    direct_step: str,
    direct_controls: tuple[_LocalControl, ...],
    agent: PoeStepTextAgent,
) -> tuple[str, tuple[_LocalControl, ...], int]:
    reverse = _reverse_request(
        artifact=artifact,
        original_request=original_request,
        hallucinated=hallucinated,
        target_step_index=expected.step_index,
        target_h_mentions=h_mentions,
    )
    result = agent.rewrite(reverse)
    marked_step = result.rewritten_step_texts[expected.step_index - 1]
    prefix = f"Step {expected.step_index} [{expected.step_name}]: "
    marked_head, formal = marked_step.split(FORMAL_MARKER, 1)
    marked_body = marked_head[len(prefix) :]
    parsed = parse_hallucination_markers(marked_body)
    clean_body = strip_hallucination_markers(marked_body)
    clean_step = prefix + clean_body + FORMAL_MARKER + formal

    h_head = hallucinated.step_texts[expected.step_index - 1].split(FORMAL_MARKER, 1)[0]
    h_step_start = sum(
        len(item) + 2
        for item in hallucinated.step_texts[: expected.step_index - 1]
    )
    h_natural_by_node: dict[str, list[RenderedMention]] = defaultdict(list)
    for mention in sorted(h_mentions, key=lambda item: item.start):
        if mention.start - h_step_start < len(h_head):
            h_natural_by_node[mention.node_id].append(mention)

    parsed_by_node: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for occurrence_id, value, start, end in parsed:
        node_id = occurrence_id.rsplit(".", 1)[0]
        parsed_by_node[node_id].append((value, start, end))
    controls: list[_LocalControl] = []
    for node_id, h_node_mentions in sorted(h_natural_by_node.items()):
        regenerated = parsed_by_node[node_id]
        if len(regenerated) != len(h_node_mentions):
            raise PoeTextRealizationError(
                "regenerated control markers do not map one-to-one to H mentions"
            )
        for h_mention, (value, start, end) in zip(
            h_node_mentions,
            regenerated,
            strict=True,
        ):
            controls.append(
                _LocalControl(
                    mention_id=h_mention.mention_id,
                    node_id=node_id,
                    step_index=expected.step_index,
                    start=len(prefix) + start,
                    end=len(prefix) + end,
                    value=value,
                )
            )

    direct_formal_start = direct_step.index(FORMAL_MARKER) + len(FORMAL_MARKER)
    regenerated_formal_start = clean_step.index(FORMAL_MARKER) + len(FORMAL_MARKER)
    shift = regenerated_formal_start - direct_formal_start
    for control in direct_controls:
        if control.start < direct_formal_start:
            continue
        controls.append(
            _LocalControl(
                mention_id=control.mention_id,
                node_id=control.node_id,
                step_index=control.step_index,
                start=control.start + shift,
                end=control.end + shift,
                value=control.value,
            )
        )
    return clean_step, tuple(sorted(controls, key=lambda item: item.start)), result.network_request_count


def _replace_by_pair_id(
    text: str,
    *,
    base_offset: int,
    h_mentions: tuple[RenderedMention, ...],
    controls_by_id: dict[str, RenderedMention],
) -> str:
    pieces: list[str] = []
    cursor = 0
    for mention in sorted(h_mentions, key=lambda item: item.start):
        local_start = mention.start - base_offset
        local_end = mention.end - base_offset
        control = controls_by_id[mention.mention_id]
        pieces.extend((text[cursor:local_start], control.value))
        cursor = local_end
    pieces.append(text[cursor:])
    return "".join(pieces)


class MatchedNegativeTextBuilder:
    """Create N from H, regenerating only steps that cannot be safely swapped."""

    def __init__(self, agent: PoeStepTextAgent | None = None) -> None:
        if agent is not None and type(agent) is not PoeStepTextAgent:
            raise TypeError("agent must be PoeStepTextAgent or None")
        self.agent = agent

    def build(
        self,
        artifact: ReferenceDAGArtifact,
        injected: InjectedHallucination,
        hallucinated: RenderedHallucination,
    ) -> MatchedRenderedPair:
        if type(artifact) is not ReferenceDAGArtifact:
            raise TypeError("artifact must be ReferenceDAGArtifact")
        if type(injected) is not InjectedHallucination:
            raise TypeError("injected must be InjectedHallucination")
        if type(hallucinated) is not RenderedHallucination:
            raise TypeError("hallucinated must be RenderedHallucination")

        request = build_poe_rewrite_request(artifact, injected)
        h_reasoning_mentions = tuple(
            item
            for item in hallucinated.hallucination_spans
            if item.component == "reasoning_chain"
        )
        n_steps: list[str] = []
        local_controls_by_step: list[tuple[_LocalControl, ...]] = []
        alignments: list[StepPairAlignment] = []
        h_offset = 0
        regeneration_network_requests = 0
        for expected, h_step in zip(request.steps, hallucinated.step_texts, strict=True):
            prefix = f"Step {expected.step_index} [{expected.step_name}]: "
            h_step_mentions = tuple(
                item
                for item in h_reasoning_mentions
                if item.step_index == expected.step_index
            )
            h_natural_controls = _natural_h_controls(
                step_text=h_step,
                reasoning_start=h_offset,
                mentions=h_step_mentions,
                prefix=prefix,
            )
            direct_step, direct_controls = _replace_step_mentions(
                h_step,
                reasoning_start=h_offset,
                mentions=h_step_mentions,
                artifact=artifact,
            )
            errors = _step_validation_errors(
                artifact=artifact,
                injected=injected,
                expected=expected,
                hallucinated_step=h_step,
                hallucinated_controls=h_natural_controls,
                step_text=direct_step,
                controls=direct_controls,
            )
            alignment = PairAlignment.BYTE_IDENTICAL
            n_step = direct_step
            controls = direct_controls
            if errors:
                if self.agent is None:
                    raise PoeTextRealizationError(
                        "matched-negative step requires Poe regeneration but no agent "
                        f"was provided: step={expected.step_index}, errors={list(errors)}"
                    )
                n_step, controls, network_requests = _regenerated_step(
                    artifact=artifact,
                    original_request=request,
                    hallucinated=hallucinated,
                    expected=expected,
                    h_mentions=h_step_mentions,
                    direct_step=direct_step,
                    direct_controls=direct_controls,
                    agent=self.agent,
                )
                regeneration_network_requests += network_requests
                alignment = PairAlignment.REGENERATED
                regenerated_errors = _step_validation_errors(
                    artifact=artifact,
                    injected=injected,
                    expected=expected,
                    hallucinated_step=h_step,
                    hallucinated_controls=h_natural_controls,
                    step_text=n_step,
                    controls=controls,
                )
                if regenerated_errors:
                    raise PoeTextRealizationError(
                        "regenerated matched-negative step failed local validation: "
                        f"step={expected.step_index}, errors={list(regenerated_errors)}"
                    )
            n_steps.append(n_step)
            local_controls_by_step.append(controls)
            alignments.append(
                StepPairAlignment(
                    step_index=expected.step_index,
                    step_name=expected.step_name,
                    rewrite_mode=expected.rewrite_mode,
                    pair_alignment=alignment,
                )
            )
            h_offset += len(h_step) + 2

        n_reasoning = "\n\n".join(n_steps)
        n_mentions: list[RenderedMention] = []
        n_offset = 0
        for controls, n_step in zip(local_controls_by_step, n_steps, strict=True):
            for control in controls:
                n_mentions.append(
                    RenderedMention(
                        mention_id=control.mention_id,
                        component="reasoning_chain",
                        node_id=control.node_id,
                        step_index=control.step_index,
                        start=n_offset + control.start,
                        end=n_offset + control.end,
                        value=control.value,
                        hallucinated=False,
                    )
                )
            n_offset += len(n_step) + 2

        h_final_mentions = tuple(
            item
            for item in hallucinated.hallucination_spans
            if item.component == "final_answer"
        )
        n_final, final_controls = _replace_step_mentions(
            hallucinated.final_answer,
            reasoning_start=0,
            mentions=h_final_mentions,
            artifact=artifact,
        )
        for control in final_controls:
            n_mentions.append(
                RenderedMention(
                    mention_id=control.mention_id,
                    component="final_answer",
                    node_id=control.node_id,
                    step_index=None,
                    start=control.start,
                    end=control.end,
                    value=control.value,
                    hallucinated=False,
                )
            )
        reference_answer = str(
            artifact.state_dag.values["final_answer"].normalized_value
        )
        if n_final != reference_answer:
            raise PoeTextRealizationError(
                "matched-negative final answer does not equal reference truth"
            )

        n_rendered = RenderedHallucination(
            reasoning_chain=n_reasoning,
            final_answer=n_final,
            step_texts=tuple(n_steps),
            mentions=tuple(n_mentions),
            realization={
                "backend": "matched_negative",
                "source_backend": hallucinated.realization.get("backend"),
                "step_pair_alignment": [item.to_dict() for item in alignments],
                "regeneration_network_request_count": regeneration_network_requests,
            },
        )

        h_ids = {item.mention_id for item in hallucinated.hallucination_spans}
        controls_by_id = {item.mention_id: item for item in n_rendered.mentions}
        if len(controls_by_id) != len(n_rendered.mentions) or set(controls_by_id) != h_ids:
            raise PoeTextRealizationError(
                "N control mentions must map one-to-one to H hallucination mentions"
            )

        h_offset = 0
        for alignment, h_step, n_step in zip(
            alignments,
            hallucinated.step_texts,
            n_rendered.step_texts,
            strict=True,
        ):
            if alignment.pair_alignment is PairAlignment.BYTE_IDENTICAL:
                h_step_mentions = tuple(
                    item
                    for item in h_reasoning_mentions
                    if item.step_index == alignment.step_index
                )
                reconstructed = _replace_by_pair_id(
                    h_step,
                    base_offset=h_offset,
                    h_mentions=h_step_mentions,
                    controls_by_id=controls_by_id,
                )
                if reconstructed != n_step:
                    raise PoeTextRealizationError(
                        "byte-identical pair invariant failed for reasoning step "
                        f"{alignment.step_index}"
                    )
            h_offset += len(h_step) + 2
        reconstructed_final = _replace_by_pair_id(
            hallucinated.final_answer,
            base_offset=0,
            h_mentions=h_final_mentions,
            controls_by_id=controls_by_id,
        )
        if reconstructed_final != n_rendered.final_answer:
            raise PoeTextRealizationError("byte-identical pair invariant failed for final answer")
        return MatchedRenderedPair(
            hallucinated=hallucinated,
            negative=n_rendered,
            step_pair_alignment=tuple(alignments),
        )


__all__ = [
    "MatchedNegativeTextBuilder",
    "MatchedRenderedPair",
    "PairAlignment",
    "StepPairAlignment",
]
