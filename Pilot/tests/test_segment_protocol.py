"""Native v15 wire tests. No legacy fixture adapter and no real Poe calls."""
import json
from collections import Counter
from dataclasses import replace

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.text_realization import (
    AffectedNodeClaim, PoeStepRewriteInput, PoeRewriteRequest, PoeStepTextAgent,
    PoeTextRealizationError, StepRewriteMode, RequiredHallucinationOccurrence,
    PoeTextRenderer, MatchedNegativeTextBuilder,
)
from molhallulens.modules.text_realization.segments import compile_segments, SegmentContractError, step_payload
from molhallulens.modules.text_realization.enumeration_plan import enumeration_plan


def step(index, node="product_rings", before="4", after="5"):
    return PoeStepRewriteInput(
        step_index=index, step_name="RING_VERIFICATION",
        original_step_text=f"Step {index} [RING_VERIFICATION]: The product has {before} rings.\n  FORMAL: VALUE({before})",
        modified_formal_ab=f"VALUE({after})", required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(AffectedNodeClaim(node, before, after),),
    )


def request(steps):
    return PoeRewriteRequest("test_origin", "add", "[CH4:1]", "Synthetic edit.", tuple(steps))


def native_segments(payload):
    """Construct legal fixture segments from the actual wire catalogue."""
    if payload["rewrite_mode"] == "occurrence_patch":
        head = payload["original_natural_body"]
        segments, cursor = [], 0
        for item in sorted(payload["required_hallucination_occurrences"], key=lambda o: o["original_span"]):
            a, b = item["original_span"]
            if a > cursor:
                segments.append({"text": head[cursor:a]})
            segments.append({"occurrence_ref": item["occurrence_id"]})
            cursor = b
        if cursor < len(head):
            segments.append({"text": head[cursor:]})
        return segments
    segments = []
    for claim in payload["affected_node_claims"]:
        if claim["parent_node_id"]:
            continue
        segments.extend([{"text": f"{claim['node_id']}="}, {"claim_ref": claim["node_id"]}, {"text": ". "}])
    for enum in payload["enumeration_blocks"]:
        segments.extend([{"enumeration_ref": enum["enumeration_ref"]}, {"text": " "}])
    if segments and segments[-1] == {"text": " "}:
        segments.pop()
    elif segments and segments[-1] == {"text": ". "}:
        segments[-1] = {"text": "."}
    return segments


@pytest.mark.parametrize("bad", [
    [{"claim_ref": "unknown"}], [{"claim_ref": "product_rings", "value": "9"}],
    [{"claim_ref": "product_rings", "surface": "name"}],
    [{"text": "[[HALLU:product_rings.01]]5[[/HALLU]]"}],
    [{"text": "FORMAL: VALUE(8)"}], [{"text": "Answer: 8"}],
    [{"text": "Step 1 [RING_VERIFICATION]: "}, {"claim_ref": "product_rings"}],
    [{"text": "bad\x00"}], [{"occurrence_ref": "product_rings.01"}],
    [{"enumeration_ref": "enum_01"}], [{"text": "Nothing changed."}],
    [{"text": "The product has 4 rings. "}, {"claim_ref": "product_rings"}],
    [{"text": "The product has 5 rings. "}, {"claim_ref": "product_rings"}],
    [{"text": "2 + 2 = "}, {"claim_ref": "product_rings"}],
])
def test_invalid_segments_fail_closed(bad):
    with pytest.raises((SegmentContractError, PoeTextRealizationError)):
        compile_segments(bad, step(1))


def test_exact_values_aliases_and_automatic_numbering():
    expected = step(1, "anchor_element", "C", "S")
    rendered = compile_segments([
        {"text": "The ANCHOR is the "}, {"claim_ref": "anchor_element", "surface": "name"},
        {"text": " atom (element "}, {"claim_ref": "anchor_element", "surface": "symbol"}, {"text": ")."},
    ], expected)
    assert "[[HALLU:anchor_element.01]]sulfur[[/HALLU]]" in rendered
    assert "[[HALLU:anchor_element.02]]S[[/HALLU]]" in rendered
    assert rendered.endswith("FORMAL: VALUE(S)")
    for before, after in (("2", "-1"), ("1.5", "2.75"), ("CCO", "CCN")):
        assert f"]]{after}[[/HALLU]]" in compile_segments([{"claim_ref": "count"}], step(1, "count", before, after))


def test_wire_prompt_has_mode_specific_examples_and_body_only_context():
    from molhallulens.modules.text_realization.poe_agent import _user_prompt
    patch = replace(step(1), rewrite_mode=StepRewriteMode.OCCURRENCE_PATCH,
        required_hallucination_occurrences=(RequiredHallucinationOccurrence(
            "product_rings.01", "product_rings", "4", "5", 16, 17),))
    rewrite = step(2)
    prompt = _user_prompt(request((patch, rewrite)))
    prefix, encoded = prompt.split("\nINPUT:\n", 1)
    shape = json.loads(prefix.split("RESPONSE_SHAPE:\n", 1)[1])
    payload = json.loads(encoded)
    assert "claim_ref is FORBIDDEN" in prefix
    assert shape["steps"][0]["segments"] == [{"patch_ref": "original_occurrences"}]
    assert not any("claim_ref" in segment for segment in shape["steps"][0]["segments"])
    assert shape["steps"][1]["segments"] == [{"draft_ref": "complete_derivation"}]
    for expected, row in zip((patch, rewrite), shape["steps"], strict=True):
        compile_segments(row["segments"], expected)
    for item in payload["steps"] + payload["context_steps"]:
        assert "original_step_text" not in item
        assert not item["original_natural_body"].startswith("Step ")
        assert "FORMAL:" not in item["original_natural_body"]
    with pytest.raises(SegmentContractError) as caught:
        compile_segments([{"claim_ref": "product_rings"}], patch)
    assert caught.value.code == "wrong_reference_type"
    assert caught.value.expected == ["product_rings.01"]


def test_patch_ref_is_explicit_exclusive_and_fully_validated():
    from molhallulens.modules.text_realization.segments import response_segments_example
    patch = replace(step(1), rewrite_mode=StepRewriteMode.OCCURRENCE_PATCH,
        required_hallucination_occurrences=(RequiredHallucinationOccurrence(
            "product_rings.01", "product_rings", "4", "5", 16, 17),))
    assert compile_segments([{"patch_ref":"original_occurrences"}], patch) == compile_segments(response_segments_example(patch), patch)
    for bad in ([{"patch_ref":"unknown"}], [{"patch_ref":"original_occurrences"}, {"text":" extra"}]):
        with pytest.raises(SegmentContractError):
            compile_segments(bad, patch)
    with pytest.raises(SegmentContractError):
        compile_segments([{"patch_ref":"original_occurrences"}], step(1))
    assert "[[HALLU:product_rings.01]]5[[/HALLU]]" in compile_segments([{"draft_ref":"complete_derivation"}], step(1))
    with pytest.raises(SegmentContractError):
        compile_segments([{"draft_ref":"complete_derivation"}], patch)
    with pytest.raises(SegmentContractError):
        compile_segments([{"draft_ref":"complete_derivation"}, {"text":"extra"}], step(1))


def test_complete_examples_all_150_maximum_pairs(all_references, fragment_pool, tmp_path):
    from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
    from molhallulens.modules.error_injection import UnifiedHallucinationInjector
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    def transport(system, user, bot, temperature):
        # Model must explicitly return the candidate. Production never calls
        # this fake or substitutes an example after an invalid response.
        return user.split("RESPONSE_SHAPE:\n", 1)[1].split("\nINPUT:\n", 1)[0]
    agent = PoeStepTextAgent(config, transport=transport, environment={}, cache_directory=tmp_path)
    for reference in all_references:
        injected = UnifiedHallucinationInjector(config).apply(reference.state_dag, planner.plan(reference, variant_index=0))
        rendered = PoeTextRenderer(agent).render(reference, injected)
        pair = MatchedNegativeTextBuilder(agent).build(reference, injected, rendered)
        assert all(s.pair_alignment.value == "byte_identical" for s in pair.step_pair_alignment)


def test_enumeration_local_expansion_and_no_deletion():
    parent = AffectedNodeClaim("fragment_heavy", "9", "10")
    body = "The fragment contains 9 heavy atoms (1 sulfur, 2 oxygens, 3 carbons, and 3 fluorines)."
    clauses, children = enumeration_plan(body, (parent,), step_name="FRAGMENT_IDENTIFICATION")
    expected = PoeStepRewriteInput(1, "FRAGMENT_IDENTIFICATION",
        "Step 1 [FRAGMENT_IDENTIFICATION]: " + body + "\n  FORMAL: COUNT(9)", "COUNT(10)", (),
        StepRewriteMode.DERIVATION_REWRITE, (parent, *(AffectedNodeClaim(**c) for c in children)), clauses)
    rendered = compile_segments([{"enumeration_ref": "enum_01"}], expected)
    assert "1 sulfur" in rendered and "2 oxygens" in rendered and "3 carbons" in rendered
    assert "]]4[[/HALLU]] fluorines" in rendered
    assert "[[HALLU:" not in json.dumps(step_payload(expected))
    for bad in ([{"claim_ref": "fragment_heavy"}], [{"enumeration_ref": "enum_01", "items": []}],
                [{"enumeration_ref": "enum_01"}, {"enumeration_ref": "enum_01"}]):
        with pytest.raises(SegmentContractError):
            compile_segments(bad, expected)


def test_retry_only_failed_steps_and_cache_complete_result(tmp_path):
    copy = replace(step(1), affected_node_claims=(), rewrite_mode=StepRewriteMode.COPY)
    expected = request((copy, step(2), step(3)))
    calls = []
    def transport(system, user, bot, temperature):
        payload = json.loads(user.split("\nINPUT:\n")[1])
        calls.append(payload)
        indices = [s["step_index"] for s in payload["steps"]]
        assert indices == ([2, 3] if len(calls) == 1 else [3])
        if len(calls) == 2:
            assert payload["repair"][0]["step_index"] == 3
            assert "unlisted" in payload["repair"][0]["response_excerpt"]
        return json.dumps({"steps": [{"step_index": i, "segments": [{"claim_ref": "unlisted" if i == 3 and len(calls) == 1 else "product_rings"}]} for i in indices]})
    agent = PoeStepTextAgent(transport=transport, environment={}, cache_directory=tmp_path)
    result = agent.rewrite(expected)
    assert result.network_request_count == 2
    assert [s["attempts"] for s in result.step_execution] == [0, 1, 2]
    assert result.step_execution[0]["backend"] == "local_copy"
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0]["observed"] == "unlisted"
    assert result.rewritten_step_texts[0].split("\n  FORMAL:")[0] == copy.original_step_text.split("\n  FORMAL:")[0]
    cached = PoeStepTextAgent(environment={}, cache_directory=tmp_path).rewrite(expected)
    assert cached.cache_hit and cached.network_request_count == 0
    assert cached.rewritten_step_texts == result.rewritten_step_texts


def test_copy_requires_no_transport_or_key(tmp_path):
    copy = replace(step(1), affected_node_claims=(), rewrite_mode=StepRewriteMode.COPY)
    result = PoeStepTextAgent(environment={}, cache_directory=tmp_path).rewrite(request((copy,)))
    assert result.network_request_count == 0 and not result.cache_hit


def test_prose_diff_feedback_and_diagnostic_config(tmp_path):
    expected = replace(step(1), rewrite_mode=StepRewriteMode.OCCURRENCE_PATCH,
        required_hallucination_occurrences=(RequiredHallucinationOccurrence("product_rings.01", "product_rings", "4", "5", 16, 17),))
    calls = []
    def transport(system, user, bot, temperature):
        payload = json.loads(user.split("\nINPUT:\n")[1])
        calls.append(payload)
        if len(calls) == 2:
            diagnostic = payload["repair"][0]
            assert diagnostic["error_code"] == "unapproved_prose_change"
            assert any("aromatic" in d["after"] for d in diagnostic["text_differences"])
        return json.dumps({"steps": [{"step_index": 1, "segments": [
            {"text": "The product has "}, {"occurrence_ref": "product_rings.01"},
            {"text": " rings and is aromatic." if len(calls) == 1 else " rings."},
        ]}]})
    config = replace(DEFAULT_HALLUCINATION_CONFIG, poe_save_diagnostics=False)
    result = PoeStepTextAgent(config, transport=transport, environment={}, cache_directory=tmp_path).rewrite(request((expected,)))
    assert result.diagnostics and result.diagnostic_path is None
    assert not (tmp_path / "diagnostics").exists()
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError):
            replace(config, poe_diagnostic_max_characters=invalid)
    with pytest.raises(ValueError):
        replace(config, poe_save_diagnostics="yes")


def test_transport_exception_cannot_leak_headers(tmp_path):
    secret = "transport-test-secret-987654"
    def transport(*args):
        raise RuntimeError("Authorization: Bearer " + secret)
    agent = PoeStepTextAgent(transport=transport, environment={"POE_API_KEY": secret}, cache_directory=tmp_path)
    with pytest.raises(PoeTextRealizationError) as caught:
        agent.rewrite(request((step(1),)))
    assert secret not in str(caught.value)
    assert caught.value.diagnostics[0]["error_code"] == "transport_error"
    assert agent.telemetry()["network_request_count"] == 1
    for path in tmp_path.rglob("*.jsonl"):
        assert secret not in path.read_text()


def test_diagnostics_redact_and_preserve_each_attempt(tmp_path):
    secret = "private-test-credential-918273"
    calls = []
    def transport(system, user, bot, temperature):
        assert secret not in user
        calls.append(user)
        return json.dumps({"steps": [{"step_index": 1, "segments": [{"text": secret if len(calls) == 1 else "Authorization: Bearer other-private-secret"}]}]})
    agent = PoeStepTextAgent(transport=transport, environment={"POE_API_KEY": secret}, cache_directory=tmp_path)
    with pytest.raises(PoeTextRealizationError) as caught:
        agent.rewrite(request((step(1),)))
    assert len(caught.value.diagnostics) == 2
    assert [d["attempt"] for d in caught.value.diagnostics] == [1, 2]
    assert secret not in str(caught.value)
    files = list(tmp_path.rglob("*"))
    assert not list(tmp_path.glob("*/*.json"))  # No success cache for failures.
    for path in files:
        if path.is_file():
            assert secret not in path.read_text() and "other-private-secret" not in path.read_text()


@pytest.mark.parametrize("row", [
    {"step_index": True, "segments": [{"claim_ref": "product_rings"}]},
    {"step_index": 2, "segments": [{"claim_ref": "product_rings"}]},
    {"step_index": 1, "rewritten_natural_language": "old protocol"},
])
def test_bad_wire_shape_has_structured_failure(row, tmp_path):
    agent = PoeStepTextAgent(transport=lambda *args: json.dumps({"steps": [row]}), environment={}, cache_directory=tmp_path)
    with pytest.raises(PoeTextRealizationError) as caught:
        agent.rewrite(request((step(1),)))
    assert len(caught.value.diagnostics) == 2


def test_all_150_maximum_pairs_through_native_wire(all_references, fragment_pool, tmp_path):
    from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
    from molhallulens.modules.error_injection import UnifiedHallucinationInjector
    from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
    from molhallulens.modules.release import UnifiedRecordBuilder
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    calls, alignments, execution = [], Counter(), Counter()
    def transport(system, user, bot, temperature):
        payload = json.loads(user.split("\nINPUT:\n")[1])
        assert all(s["rewrite_mode"] != "copy" for s in payload["steps"])
        calls.append(payload["origin_id"])
        return json.dumps({"steps": [{"step_index": s["step_index"], "segments": native_segments(s)} for s in payload["steps"]]})
    agent = PoeStepTextAgent(config, transport=transport, environment={}, cache_directory=tmp_path)
    annotator, builder = UnifiedHallucinationAnnotator(), UnifiedRecordBuilder()
    for reference in all_references:
        injected = UnifiedHallucinationInjector(config).apply(reference.state_dag, planner.plan(reference, variant_index=0))
        rendered = PoeTextRenderer(agent).render(reference, injected)
        execution.update(s["backend"] for s in rendered.realization["step_execution"])
        pair = MatchedNegativeTextBuilder(agent).build(reference, injected, rendered)
        positive = annotator.annotate(rendered, injected)
        negative = annotator.annotate_negative(pair.negative, positive)
        h, n = builder.build_pair(reference, injected, pair, positive, negative)
        controls = {s["pair_occurrence_id"]: s for s in n.data["control_spans"]}
        text = h.data["serialized"]["text"]
        for span in sorted(h.data["hallucination_spans"], key=lambda s: s["serialized_span"][0], reverse=True):
            start, end = span["serialized_span"]
            text = text[:start] + controls[span["pair_occurrence_id"]]["text"] + text[end:]
        assert text == n.data["serialized"]["text"], reference.anonymous_sample_id
        alignments.update(s.pair_alignment.value for s in pair.step_pair_alignment)
    assert len(calls) == 147 and alignments == {"byte_identical": 800}
    assert execution == {"local_copy": 221, "poe_segments": 579}
    assert agent.telemetry()["retry_count"] == 0
