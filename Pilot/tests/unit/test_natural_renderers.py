"""T040 label-blind rule and Poe placeholder renderer contracts."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from molhallulens.domain import MutationTargetKind
from molhallulens.providers.poe.response_cache import (
    CacheMode,
    PoeResponseCache,
    PoeResponseCacheError,
)
from molhallulens.rendering.natural_llm import (
    LLMNaturalRenderer,
    NarrativeRewriteRequest,
    validate_narrative_template,
)
from molhallulens.rendering.natural_rule import (
    LockedFinalAnswer,
    LockedNaturalStep,
    NaturalRenderError,
    NaturalRenderRequest,
    RuleNaturalRenderer,
    render_natural_rule,
)
from molhallulens.rendering.trace_ast import (
    ClaimNode,
    LiteralNode,
    SequenceNode,
)

GOOD_TEMPLATE = (
    "The incoming {{fragment_smiles}} fragment contains {{heavy_atoms}} heavy atoms."
)


def _literal(
    mention_id: str,
    target_id: str,
    value: str,
    *,
    target_kind: MutationTargetKind = MutationTargetKind.NODE,
) -> LiteralNode:
    return LiteralNode(
        mention_id=mention_id,
        state_or_edge_id=target_id,
        value=value,
        target_kind=target_kind,
    )


def _narrative_slots(prefix: str = "natural") -> dict[str, LiteralNode]:
    return {
        "fragment_smiles": _literal(f"{prefix}.fragment", "add_fragment", "N1CCCC1"),
        "heavy_atoms": _literal(f"{prefix}.heavy", "fragment_heavy", "5"),
    }


def _narrative_claim(prefix: str = "natural") -> ClaimNode:
    return ClaimNode.from_template(
        f"claim.{prefix}",
        "The incoming {fragment_smiles} fragment contains {heavy_atoms} heavy atoms.",
        _narrative_slots(prefix),
    )


def _formal() -> tuple[SequenceNode, str]:
    text = 'add_frag="N1CCCC1", frag_heavy=5'
    claim = ClaimNode.from_template(
        "claim.formal",
        'add_frag="{fragment_smiles}", frag_heavy={heavy_atoms}',
        _narrative_slots("formal"),
    )
    return SequenceNode((claim,)), text


def _step(narrative: ClaimNode | None = None) -> LockedNaturalStep:
    formal, formal_text = _formal()
    return LockedNaturalStep(
        step_index=1,
        step_name="ADD_FRAGMENT_SIZE",
        narrative_claims=(narrative or _narrative_claim(),),
        formal_content=formal,
        formal_text=formal_text,
    )


def _rule_request(narrative: ClaimNode | None = None) -> NaturalRenderRequest:
    return NaturalRenderRequest(
        steps=(_step(narrative),),
        final_answer=LockedFinalAnswer(
            _literal("answer.fragment", "final_answer", "N1CCCC1")
        ),
    )


def _rewrite_request(**overrides: object) -> NarrativeRewriteRequest:
    values: dict[str, object] = {
        "claim_id": "claim.llm.narrative",
        "step_name": "ADD_FRAGMENT_SIZE",
        "locked_slots": _narrative_slots("llm"),
        "style_id": "style_07",
        "original_style_excerpt": "The incoming fragment contains several heavy atoms.",
    }
    values.update(overrides)
    return NarrativeRewriteRequest(**values)  # type: ignore[arg-type]


@dataclass
class _MockPoeProducer:
    response: object
    calls: int = 0
    payloads: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.payloads = []

    def propose(self, request: dict[str, Any]) -> object:
        self.calls += 1
        assert self.payloads is not None
        self.payloads.append(request)
        return self.response


def test_rule_renderer_preserves_exact_formal_numbers_answer_and_offsets() -> None:
    rendered = render_natural_rule(_rule_request())

    assert rendered.detector_text == (
        "Step 1 [ADD_FRAGMENT_SIZE]: The incoming N1CCCC1 fragment contains "
        "5 heavy atoms.\n"
        '  FORMAL: add_frag="N1CCCC1", frag_heavy=5\n\n'
        "Answer: N1CCCC1"
    )
    assert rendered.detector_text[
        rendered.segment_spans["reasoning.step.01"].start : rendered.segment_spans[
            "reasoning.step.01"
        ].end
    ].endswith('FORMAL: add_frag="N1CCCC1", frag_heavy=5')
    answer_span = rendered.segment_spans["final_answer"]
    assert rendered.detector_text[answer_span.start : answer_span.end] == (
        "Answer: N1CCCC1"
    )
    assert len(rendered.node_mentions["add_fragment"]) == 2
    assert len(rendered.node_mentions["fragment_heavy"]) == 2
    all_fragment_occurrences = tuple(
        mention for mention in rendered.mentions if mention.literal_text == "N1CCCC1"
    )
    assert len(all_fragment_occurrences) == 3
    assert len({mention.literal_span for mention in all_fragment_occurrences}) == 3


def test_rule_renderer_is_deterministic_and_unicode_offsets_survive_transitions() -> (
    None
):
    unicode_claim = ClaimNode.from_template(
        "claim.unicode",
        "引入片段 {fragment_smiles}，其重原子数为 {heavy_atoms}。",
        _narrative_slots("unicode"),
    )
    first = RuleNaturalRenderer().render(_rule_request(unicode_claim))
    second = RuleNaturalRenderer().render(_rule_request(unicode_claim))

    assert first.to_dict() == second.to_dict()
    mention = first.mention("unicode.fragment")
    prefix = first.detector_text[: mention.literal_span.start]
    assert "引入片段" in prefix
    assert mention.literal_span.start == len(prefix)
    assert len(prefix.encode("utf-8")) > len(prefix)


def test_locked_formal_drift_unlocked_numbers_and_leakage_fail_closed() -> None:
    formal, formal_text = _formal()
    with pytest.raises(NaturalRenderError) as drift:
        LockedNaturalStep(
            1,
            "ADD_FRAGMENT_SIZE",
            (_narrative_claim(),),
            formal,
            formal_text + " ",
        )
    assert drift.value.code == "LOCKED_FORMAL_CHANGED"

    unlocked_number = ClaimNode.from_template(
        "claim.unlocked",
        "version 2 contains {fragment_smiles}",
        {"fragment_smiles": _narrative_slots()["fragment_smiles"]},
    )
    with pytest.raises(NaturalRenderError) as number_error:
        _step(unlocked_number)
    assert number_error.value.code == "UNLOCKED_NUMBER"

    leaked = ClaimNode.from_template(
        "claim.leaked",
        "The hallucinated fragment is {fragment_smiles}",
        {"fragment_smiles": _narrative_slots()["fragment_smiles"]},
    )
    with pytest.raises(NaturalRenderError) as leakage:
        _step(leaked)
    assert leakage.value.code == "LABEL_LEAKAGE"

    raw_formal_number = SequenceNode(
        (
            ClaimNode.from_template(
                "claim.bad.formal",
                "count=5, fragment={fragment_smiles}",
                {"fragment_smiles": _narrative_slots("bad")["fragment_smiles"]},
            ),
        )
    )
    with pytest.raises(NaturalRenderError) as formal_number:
        LockedNaturalStep(
            1,
            "ADD_FRAGMENT_SIZE",
            (_narrative_claim(),),
            raw_formal_number,
            "count=5, fragment=N1CCCC1",
        )
    assert formal_number.value.code == "UNLOCKED_NUMBER"


def test_final_answer_is_a_program_owned_final_answer_node() -> None:
    with pytest.raises(ValueError, match="target final_answer"):
        LockedFinalAnswer(_literal("answer", "product", "CCN"))
    with pytest.raises(ValueError, match="target final_answer"):
        LockedFinalAnswer(
            _literal(
                "answer",
                "final_answer",
                "CCN",
                target_kind=MutationTargetKind.EDGE,
            )
        )


def test_good_poe_template_reuses_exact_locked_literal_identities() -> None:
    request = _rewrite_request()
    producer = _MockPoeProducer({"template": GOOD_TEMPLATE})
    result = LLMNaturalRenderer(producer=producer).render(request)

    assert producer.calls == 1
    assert producer.payloads == [
        {
            "step_name": "ADD_FRAGMENT_SIZE",
            "locked_slots": {
                "fragment_smiles": "N1CCCC1",
                "heavy_atoms": "5",
            },
            "style_id": "style_07",
            "original_style_excerpt": (
                "The incoming fragment contains several heavy atoms."
            ),
        }
    ]
    assert result.template == GOOD_TEMPLATE
    assert result.claim.claim_id == request.claim_id
    claim_literals = tuple(
        child for child in result.claim.content.children if type(child) is LiteralNode
    )
    assert tuple(literal.mention_id for literal in claim_literals) == (
        "llm.fragment",
        "llm.heavy",
    )
    assert tuple(literal.state_or_edge_id for literal in claim_literals) == (
        "add_fragment",
        "fragment_heavy",
    )

    rendered = render_natural_rule(_rule_request(result.claim))
    assert "The incoming N1CCCC1 fragment contains 5 heavy atoms." in (
        rendered.detector_text
    )
    assert rendered.mention("llm.fragment").literal_text == "N1CCCC1"


@pytest.mark.parametrize(
    ("template", "code"),
    (
        (
            "The incoming {{fragment_smiles}} fragment is present.",
            "LOCKED_FACT_OMITTED",
        ),
        (
            "{{fragment_smiles}} has {{heavy_atoms}} atoms and {{charge}} charge.",
            "UNKNOWN_PLACEHOLDER",
        ),
        (
            "{{fragment_smiles}} and {{fragment_smiles}} have {{heavy_atoms}} atoms.",
            "PLACEHOLDER_REPEATED",
        ),
        (
            "{{fragment_smiles}} has {{heavy_atoms}} atoms in version 2.",
            "UNLOCKED_NUMBER",
        ),
        (
            "{{fragment_smiles}} has six or {{heavy_atoms}} atoms.",
            "UNLOCKED_NUMBER",
        ),
        (
            "The hallucinated {{fragment_smiles}} has {{heavy_atoms}} atoms.",
            "LABEL_LEAKAGE",
        ),
        (
            "FORMAL: {{fragment_smiles}} has {{heavy_atoms}} atoms.",
            "RESERVED_STRUCTURE",
        ),
        (
            "The {fragment_smiles} has {{heavy_atoms}} atoms.",
            "PLACEHOLDER_SYNTAX",
        ),
        (
            "{{fragment_smiles}} does not contain {{heavy_atoms}} atoms.",
            "LOCKED_FACT_SEMANTIC_DRIFT",
        ),
    ),
)
def test_invalid_poe_templates_are_rejected_locally(template: str, code: str) -> None:
    with pytest.raises(NaturalRenderError) as captured:
        validate_narrative_template(_rewrite_request(), template)
    assert captured.value.code == code


def test_renderer_response_cannot_override_claim_or_literal_identity() -> None:
    response = {
        "template": GOOD_TEMPLATE,
        "claim_id": "provider-controlled-identity",
    }
    with pytest.raises(NaturalRenderError) as captured:
        LLMNaturalRenderer(producer=_MockPoeProducer(response)).render(
            _rewrite_request()
        )
    assert captured.value.code == "RENDER_RESPONSE_SCHEMA"


def test_locked_slot_mapping_is_defensively_frozen_before_producer() -> None:
    slots = _narrative_slots("frozen")
    request = _rewrite_request(locked_slots=slots)
    slots["fragment_smiles"] = _literal("changed", "changed", "O")

    class _MutatingProducer:
        def propose(self, payload: dict[str, Any]) -> dict[str, str]:
            payload["locked_slots"]["fragment_smiles"] = "provider mutation"
            return {"template": GOOD_TEMPLATE}

    result = LLMNaturalRenderer(producer=_MutatingProducer()).render(request)
    assert request.locked_slots["fragment_smiles"].value == "N1CCCC1"
    assert result.claim.content.children[1] == request.locked_slots["fragment_smiles"]
    with pytest.raises(TypeError):
        request.locked_slots["new"] = _literal("new", "new", "N")  # type: ignore[index]


def test_render_cache_hit_and_frozen_replay_never_reinvoke_poe(
    tmp_path: Path,
) -> None:
    request = _rewrite_request()
    producer = _MockPoeProducer({"template": GOOD_TEMPLATE})
    cache = PoeResponseCache(
        tmp_path,
        clock=lambda: "2026-08-30T05:00:00+00:00",
    )
    renderer = LLMNaturalRenderer(producer=producer, cache=cache)

    first = renderer.render(request)
    second = renderer.render(request)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert producer.calls == 1
    assert first.claim == second.claim

    never = _MockPoeProducer({"template": "must not run"})
    replay = LLMNaturalRenderer(
        producer=never,
        cache=PoeResponseCache(tmp_path, mode=CacheMode.FROZEN_REPLAY),
    ).render(request)
    assert replay.cache_hit is True
    assert never.calls == 0

    miss = replace(request, style_id="style_08")
    with pytest.raises(PoeResponseCacheError) as captured:
        LLMNaturalRenderer(
            producer=never,
            cache=PoeResponseCache(tmp_path, mode=CacheMode.FROZEN_REPLAY),
        ).render(miss)
    assert captured.value.code == "CACHE_MISS_FROZEN"
    assert never.calls == 0


def test_public_renderer_inputs_have_no_truth_label_or_correctness_fields() -> None:
    forbidden = {
        "gt_smiles",
        "oracle",
        "label",
        "operator_id",
        "operator_correctness",
        "correctness",
        "hallucination",
    }
    assert set(NaturalRenderRequest.__dataclass_fields__).isdisjoint(forbidden)
    assert set(NarrativeRewriteRequest.__dataclass_fields__).isdisjoint(forbidden)
    assert set(inspect.signature(RuleNaturalRenderer.render).parameters) == {
        "self",
        "request",
    }
    assert set(inspect.signature(LLMNaturalRenderer.render).parameters) == {
        "self",
        "request",
    }
    assert set(_rewrite_request().provider_payload()) == {
        "step_name",
        "locked_slots",
        "style_id",
        "original_style_excerpt",
    }
