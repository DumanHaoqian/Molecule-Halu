from __future__ import annotations

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
