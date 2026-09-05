from __future__ import annotations

from conftest import structured_fixture_transport

from dataclasses import replace

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.text_realization import (
    AffectedNodeClaim,
    DeterministicTextRenderer,
    PoeStepRewriteInput,
    PoeTextRealizationError,
    StepRewriteMode,
    validate_rewritten_step_text,
)
from molhallulens.modules.text_realization.occurrence_audit import (
    enumeration_violations,
)
from molhallulens.modules.text_realization.enumeration_plan import enumeration_plan
from molhallulens.modules.text_realization.poe_agent import strip_hallucination_markers


@pytest.mark.parametrize("separator", (", while ", "; ", " and ", "\n"))
def test_source_enumeration_and_product_claim_can_share_sentence(all_references, separator, tmp_path):
    reference = next(r for r in all_references if r.anonymous_sample_id == "mol_edit.add_v2.0015")
    step = reference.trace_steps[4]
    claims = (AffectedNodeClaim("product_rings", "4", "6"), AffectedNodeClaim("ring_delta", "0", "+2"))
    clauses, children = enumeration_plan(step.natural_language, claims, step_name=step.step_name)
    assert not children  # The source's enumeration must remain unchanged.
    assert len(clauses) == 1
    expected = PoeStepRewriteInput(
        step_index=5, step_name=step.step_name,
        original_step_text=step.render(include_answer=False),
        modified_formal_ab=step.formal_ab.replace("PRODUCT_SMILES[n_rings=4]", "PRODUCT_SMILES[n_rings=6]").replace("RING_DELTA(0)", "RING_DELTA(+2)"),
        required_hallucination_occurrences=(), rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=claims, preserved_enumerations=clauses,
    )
    source = "The source molecule contains 4 rings (two phenyl rings and one quinoline ring system which counts as 2 rings)"
    product = "the product also contains [[HALLU:product_rings.01]]6[[/HALLU]] rings"
    delta = "The difference is [[HALLU:ring_delta.01]]+2[[/HALLU]]."
    def complete(body):
        return "Step 5 [RING_VERIFICATION]: " + body + "\n  FORMAL: " + expected.modified_formal_ab
    for body in (source + separator + product + ". " + delta, product + separator + source + ". " + delta):
        validate_rewritten_step_text(complete(body), expected)
        with pytest.raises(PoeTextRealizationError):
            validate_rewritten_step_text(complete(body.replace("contains 4 rings", "contains 5 rings")), expected)
        with pytest.raises(PoeTextRealizationError):
            validate_rewritten_step_text(complete(body.replace("two phenyl rings", "two phenyl rings and one pyrrole ring")), expected)
        with pytest.raises(PoeTextRealizationError):
            validate_rewritten_step_text(complete(body.replace("]6[[/HALLU]]", "]7[[/HALLU]]")), expected)
    with pytest.raises(PoeTextRealizationError):
        # An explicit subject *inside* the breakdown cannot hide an extra claim.
        validate_rewritten_step_text(complete(source[:-1] + "; " + product + "). " + delta), expected)
    # Exercise the actual agent boundary: the compound sentence succeeds on
    # its first fake response, rather than consuming both validation attempts.
    import json
    from molhallulens.modules.text_realization import PoeRewriteRequest, PoeStepTextAgent
    steps = tuple(PoeStepRewriteInput(
        step_index=t.step_index, step_name=t.step_name,
        original_step_text=t.render(include_answer=False), modified_formal_ab=t.formal_ab,
        required_hallucination_occurrences=(),
    ) for t in reference.trace_steps[:4]) + (expected,)
    request = PoeRewriteRequest(
        origin_id=reference.anonymous_sample_id, subtask="add",
        indexed_smiles=reference.state_dag.values["source"].normalized_value,
        instruction=reference.state_dag.values["instruction"].normalized_value,
        steps=steps,
    )
    calls = []
    def fake_transport(*args):
        calls.append(args)
        return json.dumps({"steps": [{"step_index": 5, "segments": [
            {"enumeration_ref": "enum_01"}, {"text": " The product contains "},
            {"claim_ref": "product_rings"}, {"text": " rings. The difference is "},
            {"claim_ref": "ring_delta"}, {"text": "."},
        ]}]})
    result = PoeStepTextAgent(transport=fake_transport, environment={}, cache_directory=tmp_path).rewrite(request)
    assert result.network_request_count == 1
    assert len(calls) == 1


def test_component_error_is_preserved_marked_and_reversible():
    original = "The fragment has 9 heavy atoms (1 sulfur, 2 oxygens, 3 carbons, and 3 fluorines)."
    parent = AffectedNodeClaim("fragment_heavy", "9", "10")
    clauses, derived = enumeration_plan(original, (parent,), step_name="FRAGMENT_IDENTIFICATION")
    assert len(derived) == 1
    child = AffectedNodeClaim(**derived[0])
    assert (child.before_text, child.after_text, child.parent_node_id) == ("3", "4", "fragment_heavy")
    expected = PoeStepRewriteInput(
        step_index=1, step_name="FRAGMENT_IDENTIFICATION",
        original_step_text="Step 1 [FRAGMENT_IDENTIFICATION]: " + original + "\n  FORMAL: COUNT(9)",
        modified_formal_ab="COUNT(10)", required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(parent, child), preserved_enumerations=clauses,
    )
    marked = clauses[0]
    complete = "Step 1 [FRAGMENT_IDENTIFICATION]: " + marked + "\n  FORMAL: COUNT(10)"
    validate_rewritten_step_text(complete, expected)
    total_marker = "[[HALLU:fragment_heavy.01]]10[[/HALLU]]"
    child_marker = f"[[HALLU:{child.node_id}.01]]4[[/HALLU]]"
    for body in (
        f"The fragment contains {total_marker} heavy atoms: 1 sulfur, 2 oxygens, 3 carbons, and {child_marker} fluorines.",
        f"The {total_marker} heavy atoms comprise {child_marker} fluorines,\n 3 carbons, 2 oxygens, and 1 sulfur.",
        f"1 sulfur; 2 oxygen; 3 carbon; {child_marker} fluorine, totaling {total_marker} heavy atoms.",
    ):
        validate_rewritten_step_text(
            "Step 1 [FRAGMENT_IDENTIFICATION]: " + body + "\n  FORMAL: COUNT(10)", expected,
        )
    for changed in (
        marked.replace("2 oxygens, ", ""),
        marked.replace("3 carbons", "3 fluorines").replace(" fluorines =", " carbons ="),
        marked.replace(" = ", ", 1 nitrogen = "),
        marked.replace("1 sulfur", "2 sulfur"),
        marked.replace("1 sulfur", "-1 sulfur"),
        marked.replace("1 sulfur", "1 sulfur ring"),
    ):
        with pytest.raises(PoeTextRealizationError, match="enumeration_preservation"):
            validate_rewritten_step_text(
                "Step 1 [FRAGMENT_IDENTIFICATION]: " + changed + "\n  FORMAL: COUNT(10)", expected,
            )
    truth = marked
    for claim in (parent, child):
        truth = truth.replace(f"]]{claim.after_text}[[/HALLU]]", f"]]{claim.before_text}[[/HALLU]]")
    assert not enumeration_violations(strip_hallucination_markers(truth))
    assert "3 fluorines" in strip_hallucination_markers(truth)
    with pytest.raises(PoeTextRealizationError, match="enumeration_preservation"):
        validate_rewritten_step_text(complete.replace("3 carbons", "4 carbons"), expected)
    for unmarked in ("3 fluorines", "4 fluorines"):
        with pytest.raises(PoeTextRealizationError, match="enumeration_unmarked_component"):
            validate_rewritten_step_text(
                complete.replace("\n  FORMAL:", f" Also {unmarked}.\n  FORMAL:"), expected,
            )
    with pytest.raises(PoeTextRealizationError):
        validate_rewritten_step_text(complete.replace(f"[[HALLU:{child.node_id}.01]]4[[/HALLU]]", "4"), expected)
    with pytest.raises(PoeTextRealizationError):
        validate_rewritten_step_text(
            "Step 1 [FRAGMENT_IDENTIFICATION]: "
            f"[[HALLU:fragment_heavy.01]]10[[/HALLU]], [[HALLU:{child.node_id}.01]]4[[/HALLU]]."
            "\n  FORMAL: COUNT(10)", expected,
        )


def test_derived_enumeration_spans_have_truth_controls(all_references, fragment_pool):
    from molhallulens.modules.text_realization import MatchedNegativeTextBuilder
    from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
    reference = next(item for item in all_references if item.anonymous_sample_id == "mol_edit.add_v2.0016")
    config = replace(DEFAULT_HALLUCINATION_CONFIG, max_edit_count=3)
    plan = UnifiedHallucinationPlanner(fragment_pool, config).plan(reference, variant_index=0)
    injected = UnifiedHallucinationInjector(config).apply(reference.state_dag, plan)
    rendered = DeterministicTextRenderer().render(reference, injected)
    pair = MatchedNegativeTextBuilder().build(reference, injected, rendered)
    annotator = UnifiedHallucinationAnnotator()
    h = annotator.annotate(rendered, injected)
    n = annotator.annotate_negative(pair.negative, h)
    children = [span for span in h.spans if span.parent_node_id]
    assert children
    controls = {span.pair_occurrence_id: span for span in n.control_spans}
    for child in children:
        assert child.causal_role.value == "propagated_error"
        assert child.propagation_event_id.startswith("text:")
        assert child.text != controls[child.mention_id].text
    reconstructed = rendered.reasoning_chain
    for span in sorted((s for s in h.spans if s.component == "reasoning_chain"), key=lambda s: s.start, reverse=True):
        reconstructed = reconstructed[:span.start] + controls[span.mention_id].text + reconstructed[span.end:]
    assert reconstructed == pair.negative.reasoning_chain
    # A later reverse-Poe retry must retain the same component inventory.
    from molhallulens.modules.text_realization import build_poe_rewrite_request
    from molhallulens.modules.text_realization.pairing import _reverse_request
    request = build_poe_rewrite_request(reference, injected)
    step_index = children[0].step_index
    reverse = _reverse_request(
        artifact=reference, original_request=request, hallucinated=rendered,
        target_step_index=step_index,
        target_h_mentions=tuple(m for m in rendered.hallucination_spans if m.step_index == step_index),
    )
    reverse_step = reverse.steps[step_index - 1]
    assert reverse_step.preserved_enumerations
    assert not enumeration_violations(" ".join(
        strip_hallucination_markers(clause) for clause in reverse_step.preserved_enumerations
    ))
    assert any(claim.parent_node_id for claim in reverse_step.affected_node_claims)


@pytest.mark.parametrize(
    ("text", "invalid"),
    (
        (
            "It contains 1 sulfur, 2 oxygens, 3 carbons, and 3 fluorines, "
            "totaling 10 heavy atoms.",
            True,
        ),
        (
            "It contains 1 sulfur, 2 oxygens, 3 carbons, and 3 fluorines, "
            "totaling 9 heavy atoms.",
            False,
        ),
        (
            "The fragment has 4 heavy atoms (one carbonyl carbon, one carbonyl "
            "oxygen, one methyl carbon).",
            True,
        ),
        (
            "The fragment has three heavy atoms (one carbonyl carbon, one "
            "carbonyl oxygen, one methyl carbon).",
            False,
        ),
        (
            "Count heavy atoms: 2 carbons, 1 sulfur, and 2 oxygens = 6 heavy atoms.",
            True,
        ),
        (
            "Count heavy atoms: two carbons, one sulfur, and two oxygens = five "
            "heavy atoms.",
            False,
        ),
        (
            "It has one pyrazole, one pyridine, and two benzene rings, totaling "
            "five rings.",
            True,
        ),
        (
            "Heavy atoms: 5 (methylene carbon, carbonyl carbon, and two oxygens).",
            True,
        ),
        (
            "Heavy atoms: four (methylene carbon, carbonyl carbon, and two oxygens).",
            False,
        ),
    ),
)
def test_enumeration_validator_covers_corpus_forms_and_number_words(text, invalid):
    assert bool(enumeration_violations(text)) is invalid


def test_all_raw_natural_language_has_no_enumeration_false_positive(all_references):
    violations = []
    for reference in all_references:
        for step in reference.trace_steps:
            found = enumeration_violations(step.natural_language)
            if found:
                violations.append(
                    (reference.anonymous_sample_id, step.step_index, found)
                )
    assert violations == []


def test_poe_contract_rejects_a_marked_total_with_false_enumeration():
    expected = PoeStepRewriteInput(
        step_index=1,
        step_name="FRAGMENT_IDENTIFICATION",
        original_step_text=(
            "Step 1 [FRAGMENT_IDENTIFICATION]: The fragment has 9 heavy atoms."
            "\n  FORMAL: FRAGMENT(heavy_atoms=9)"
        ),
        modified_formal_ab="FRAGMENT(heavy_atoms=10)",
        required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(
            AffectedNodeClaim(
                node_id="fragment_heavy",
                before_text="9",
                after_text="10",
            ),
        ),
    )
    rewritten = (
        "Step 1 [FRAGMENT_IDENTIFICATION]: It contains 1 sulfur, 2 oxygens, "
        "3 carbons, and 3 fluorines, totaling "
        "[[HALLU:fragment_heavy.01]]10[[/HALLU]] heavy atoms."
        "\n  FORMAL: FRAGMENT(heavy_atoms=10)"
    )
    with pytest.raises(PoeTextRealizationError, match="enumeration"):
        validate_rewritten_step_text(rewritten, expected)


def test_all_deterministic_variant_zero_text_passes_enumeration_audit(
    all_references,
    fragment_pool,
):
    # Keep the historical three-root variant envelope so this regression covers
    # the same 150 fixtures used by the original occurrence-recall review.
    config = replace(DEFAULT_HALLUCINATION_CONFIG, max_edit_count=3)
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injector = UnifiedHallucinationInjector(config)
    for reference in all_references:
        plan = planner.plan(reference, variant_index=0)
        injected = injector.apply(reference.state_dag, plan)
        rendered = DeterministicTextRenderer().render(reference, injected)
        assert not enumeration_violations(rendered.reasoning_chain), (
            reference.anonymous_sample_id,
            enumeration_violations(rendered.reasoning_chain),
        )
