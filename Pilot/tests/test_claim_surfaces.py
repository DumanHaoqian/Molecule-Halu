from conftest import structured_fixture_transport
"""Semantic-binding regressions from the maximum-edits failure investigation."""

import json
from collections import Counter
from dataclasses import replace

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.text_realization import (
    AffectedNodeClaim, DeterministicTextRenderer, MatchedNegativeTextBuilder,
    PoeStepRewriteInput, PoeStepTextAgent, PoeTextRenderer, PoeTextRealizationError,
    RequiredHallucinationOccurrence, StepRewriteMode, build_poe_rewrite_request,
    validate_rewritten_step_text,
)
from molhallulens.modules.text_realization.claim_surfaces import (
    anchor_element_spans, claim_surface_pairs, patch_prose_signature,
)
from molhallulens.modules.text_realization.occurrence_audit import (
    enumeration_violations, loose_occurrence_spans,
)
from molhallulens.modules.text_realization.enumeration_plan import enumeration_plan
from molhallulens.modules.text_realization.poe_agent import FORMAL_MARKER, strip_hallucination_markers


def test_anchor_names_are_bound_but_smiles_branches_are_not():
    text = ('The ANCHOR is the carbon atom, atom 25 (element C). '
            'ADD_FRAGMENT = "N(C)CCS(=O)(=O)O". Fragment N(C) is unchanged.')
    assert [text[a:b] for a, b in anchor_element_spans("C", text)] == ["carbon", "C"]
    assert not anchor_element_spans("C", 'The ANCHOR receives ADD_FRAGMENT = "N(C)CCO".')
    assert not anchor_element_spans("C", 'The ANCHOR receives N(C)CCO.')
    assert not anchor_element_spans("C", "The fragment contains carbon and oxygen.")
    assert not anchor_element_spans("C", "The ANCHOR receives a fragment containing carbon.")
    text = "The ANCHOR is the phenolic oxygen (element O)."
    assert [text[a:b] for a, b in anchor_element_spans("O", text)] == ["oxygen", "O"]
    assert claim_surface_pairs("anchor_element", "C", "S")[1] == ("carbon", "sulfur")


def test_element_name_marker_is_reversible_and_stale_alias_rejected():
    expected = PoeStepRewriteInput(
        step_index=1, step_name="ANCHOR_IDENTIFICATION",
        original_step_text='Step 1 [ANCHOR_IDENTIFICATION]: The ANCHOR is the carbon atom (element C).\n  FORMAL: ANCHOR(C)',
        modified_formal_ab="ANCHOR(S)", required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(AffectedNodeClaim("anchor_element", "C", "S"),),
    )
    valid = ('Step 1 [ANCHOR_IDENTIFICATION]: The ANCHOR is the '
             '[[HALLU:anchor_element.01]]sulfur[[/HALLU]] atom '
             '(element [[HALLU:anchor_element.02]]S[[/HALLU]]).\n  FORMAL: ANCHOR(S)')
    assert validate_rewritten_step_text(valid, expected) == valid
    for bad in (valid.replace("sulfur", "oxygen"),
                valid.replace("[[HALLU:anchor_element.01]]sulfur[[/HALLU]]", "carbon"),
                valid.replace("[[HALLU:anchor_element.01]]sulfur[[/HALLU]]", "sulfur")):
        with pytest.raises(PoeTextRealizationError):
            validate_rewritten_step_text(bad, expected)


def test_patch_permits_scoped_wording_and_layout_not_new_facts():
    body = 'The product has 4 rings. Fragment = "N(C)CO".'
    start = body.index("4")
    expected = PoeStepRewriteInput(
        step_index=1, step_name="RING_CHECK",
        original_step_text="Step 1 [RING_CHECK]: " + body + FORMAL_MARKER + "COUNT(4)",
        modified_formal_ab="COUNT(5)",
        required_hallucination_occurrences=(RequiredHallucinationOccurrence(
            "product_rings.01", "product_rings", "4", "5", start, start + 1,
        ),),
        affected_node_claims=(AffectedNodeClaim("product_rings", "4", "5"),),
    )
    valid = ("Step 1 [RING_CHECK]: " + body.replace("has 4", "contains\n  [[HALLU:product_rings.01]]5[[/HALLU]]")
             + FORMAL_MARKER + "COUNT(5)")
    validate_rewritten_step_text(valid, expected)
    for bad in (valid.replace("contains", "does not contain"), valid.replace(" rings.", " atoms."),
                valid.replace("N(C)CO", "N(N)CO"), valid.replace(" rings.", " rings and is aromatic.")):
        with pytest.raises(PoeTextRealizationError):
            validate_rewritten_step_text(bad, expected)
    assert patch_prose_signature('Fragment = "N( C)CO".') != patch_prose_signature('Fragment = "N(C)CO".')
    assert patch_prose_signature("The product has 1 ring.") == patch_prose_signature("The product molecule contains 1 rings.")


def test_cross_step_fragment_count_and_symbolic_breakdown():
    body = 'REMOVE_GROUP = "O" (1 heavy atom). ADD_FRAGMENT = "Cl" (1 heavy atom).'
    spans = loose_occurrence_spans("remove_heavy", "1", body, step_name="ANCHOR_IDENTIFICATION")
    assert len(spans) == 1 and spans[0][0] < body.index("ADD_FRAGMENT")
    body = 'Heavy atoms: N + C + C + C + C + O = 6. ADD_HEAVY = 6.'
    spans = loose_occurrence_spans("add_heavy", "6", body, step_name="ADD_FRAGMENT_SIZE")
    assert len(spans) == 2
    assert not loose_occurrence_spans("add_heavy", "6", "ADD_HEAVY = 6.5.", step_name="ADD_FRAGMENT_SIZE")
    assert not loose_occurrence_spans("ring_delta", "0", "The claimed delta is 0.", step_name="HEAVY_ATOM_VERIFICATION")
    assert loose_occurrence_spans("ring_delta", "0", "RING_DELTA = 0.", step_name="HEAVY_ATOM_VERIFICATION")
    assert enumeration_violations(body.replace("= 6.", "= 5.", 1))
    parent = AffectedNodeClaim("add_heavy", "6", "5")
    clauses, children = enumeration_plan(body, (parent,), step_name="ADD_FRAGMENT_SIZE")
    assert len(clauses) == len(children) == 1
    assert children[0]["parent_node_id"] == "add_heavy"
    assert (children[0]["before_text"], children[0]["after_text"]) == ("1", "0")
    clean = strip_hallucination_markers(clauses[0])
    assert "0 O" in clean and not enumeration_violations(clean)
    assert clean.count("1 C") == 4  # No component was deleted.
    from molhallulens.modules.text_realization.enumeration_plan import validate_enumeration_inventory
    validate_enumeration_inventory(clauses[0], clauses)
    with pytest.raises(ValueError):
        validate_enumeration_inventory(clauses[0].replace("1 C", "1 c", 1), clauses)


def test_enumeration_connectors_and_repeated_totals_keep_binding():
    from molhallulens.modules.text_realization.enumeration_plan import validate_enumeration_inventory
    clauses = ('1 sulfur, 2 oxygens = [[HALLU:fragment_heavy.01]]3[[/HALLU]] heavy atoms.',)
    body = ('The fragment contains [[HALLU:fragment_heavy.01]]3[[/HALLU]] heavy atoms, '
            'comprising 1 sulfur and 2 oxygens, totaling '
            '[[HALLU:fragment_heavy.02]]3[[/HALLU]] heavy atoms.')
    validate_enumeration_inventory(body, clauses)
    for bad in (body.replace('.02]]3', '.02]]4'),
                body.replace('fragment_heavy.02', 'remove_heavy.02'),
                body.replace('1 sulfur', '1 sulfur and 1 nitrogen')):
        with pytest.raises(ValueError):
            validate_enumeration_inventory(bad, clauses)


def test_real_maximum_examples_and_name_pairing(all_references, fragment_pool, tmp_path):
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    for origin in ("mol_edit.substitute_v2.0005", "mol_edit.substitute_v2.0035", "mol_edit.substitute_v2.0274"):
        reference = next(r for r in all_references if r.anonymous_sample_id == origin)
        injected = UnifiedHallucinationInjector(config).apply(reference.state_dag, planner.plan(reference, variant_index=0))
        request = build_poe_rewrite_request(reference, injected)
        fixture = DeterministicTextRenderer().render(reference, injected)
        if origin.endswith("0274"):
            assert request.steps[2].rewrite_mode is StepRewriteMode.COPY
            assert 'N(C)CCS(=O)(=O)O' in fixture.step_texts[2]
        if origin.endswith("0035"):
            assert request.steps[2].rewrite_mode is StepRewriteMode.DERIVATION_REWRITE
            assert request.steps[2].preserved_enumerations
        if origin.endswith("0005"):
            assert {c.node_id for c in request.steps[0].affected_node_claims} >= {"anchor_element", "remove_heavy"}
            # Inject a genuine name occurrence through the production renderer.
            from conftest import preserve_enumerations
            def fake_transport(*args):
                steps = []
                for step in request.steps:
                    if step.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
                        marked = "; ".join(
                            f"{c.node_id}=[[HALLU:{c.node_id}.01]]{c.after_text}[[/HALLU]]"
                            for c in step.affected_node_claims if not c.parent_node_id
                        ) + "."
                        if step.step_index == 1:
                            marked += " The ANCHOR is the [[HALLU:anchor_element.02]]sulfur[[/HALLU]] atom."
                        marked = preserve_enumerations(marked, step.to_prompt_dict())
                    else:
                        marked = step.original_step_text.split(FORMAL_MARKER)[0].split(": ", 1)[1]
                        for item in sorted(step.required_hallucination_occurrences, key=lambda i: i.original_start, reverse=True):
                            marked = marked[:item.original_start] + f"[[HALLU:{item.occurrence_id}]]{item.after_text}[[/HALLU]]" + marked[item.original_end:]
                    steps.append({"step_index": step.step_index, "rewritten_natural_language": marked})
                return json.dumps({"steps": steps})
            agent = PoeStepTextAgent(config=config, transport=structured_fixture_transport(fake_transport), environment={}, cache_directory=tmp_path)
            fixture = PoeTextRenderer(agent).render(reference, injected)
        pair = MatchedNegativeTextBuilder().build(reference, injected, fixture)
        if origin.endswith("0005"):
            assert "sulfur atom" in pair.hallucinated.reasoning_chain
            assert "carbon atom" in pair.negative.reasoning_chain
        # Independent character-level pair invariant across the complete chain.
        truth_by_id = {m.mention_id: m for m in pair.negative.mentions}
        swapped = fixture.reasoning_chain
        for mention in sorted((m for m in fixture.mentions if m.component == "reasoning_chain" and m.hallucinated), key=lambda m: m.start, reverse=True):
            control = truth_by_id[mention.mention_id]
            swapped = swapped[:mention.start] + control.value + swapped[mention.end:]
        assert swapped == pair.negative.reasoning_chain


def test_maximum_all_origins_preserve_pair_and_remove_stale_claims(all_references, fragment_pool):
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    modes, alignment = Counter(), Counter()
    for reference in all_references:
        injected = UnifiedHallucinationInjector(config).apply(reference.state_dag, planner.plan(reference, variant_index=0))
        request = build_poe_rewrite_request(reference, injected)
        rendered = DeterministicTextRenderer().render(reference, injected)
        pair = MatchedNegativeTextBuilder().build(reference, injected, rendered)
        modes.update(s.rewrite_mode.value for s in request.steps)
        alignment.update(s.pair_alignment.value for s in pair.step_pair_alignment)
        controls = {m.mention_id: m for m in pair.negative.mentions}
        swapped = rendered.reasoning_chain
        for m in sorted((m for m in rendered.mentions if m.component == "reasoning_chain" and m.hallucinated), key=lambda m: m.start, reverse=True):
            swapped = swapped[:m.start] + controls[m.mention_id].value + swapped[m.end:]
        assert swapped == pair.negative.reasoning_chain, reference.anonymous_sample_id
        for step, text in zip(request.steps, rendered.step_texts, strict=True):
            body = text.split(FORMAL_MARKER)[0].split(": ", 1)[1]
            for node in injected.changed_node_ids:
                value = reference.state_dag.values[node].normalized_value
                before = f"+{value}" if node in {"heavy_delta", "ring_delta"} and type(value) is int and value > 0 else str(value)
                assert not loose_occurrence_spans(node, before, body, step_name=step.step_name), (reference.anonymous_sample_id, step.step_index, node, body)
    assert alignment == {"byte_identical": 800}
    print(json.dumps({"maximum_origins": len(all_references), "modes": modes, "alignment": alignment, "stale_claims": 0}))
