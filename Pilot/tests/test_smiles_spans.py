from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from rdkit import Chem

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.release import UnifiedRecordBuilder
from molhallulens.modules.text_realization import (
    DeterministicTextRenderer,
    MatchedNegativeTextBuilder,
)
from molhallulens.modules.text_realization.smiles_diff import molecular_text_diff


def _only_product_targets():
    return {name: ("product",) for name in ("add", "delete", "substitute")}


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def test_pure_insertion_and_deletion_have_nonempty_reconstructable_spans():
    for reference, candidate in (("CCO", "CCNO"), ("CCNO", "CCO")):
        difference = molecular_text_diff(reference, candidate)
        h_value = candidate[difference.candidate_start : difference.candidate_end]
        n_value = reference[difference.reference_start : difference.reference_end]
        assert h_value
        assert n_value
        reconstructed = (
            candidate[: difference.candidate_start]
            + n_value
            + candidate[difference.candidate_end :]
        )
        assert reconstructed == reference


def test_one_atom_product_edit_has_small_span_and_full_smiles_context(
    all_references,
    fragment_pool,
):
    reference = next(
        item
        for item in all_references
        if item.anonymous_sample_id == "mol_edit.add_v2.0003"
    )
    config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        editable_nodes_by_subtask=_only_product_targets(),
        smiles_mutation_operators=("smiles_atom_replacement",),
    )
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injected = UnifiedHallucinationInjector(config).apply(
        reference.state_dag,
        planner.plan(reference, variant_index=0),
    )
    rendered = DeterministicTextRenderer().render(reference, injected)
    positive = UnifiedHallucinationAnnotator().annotate(rendered, injected)
    product_spans = [
        span
        for span in positive.spans
        if span.node_id == "product"
        and span.operator == "smiles_atom_replacement"
    ]
    assert product_spans
    for span in product_spans:
        assert len(span.text) < 5
        assert span.diff_opcodes
        component = (
            rendered.reasoning_chain
            if span.component == "reasoning_chain"
            else rendered.final_answer
        )
        context = component[span.context_start : span.context_end]
        assert _canonical(context) == _canonical(
            str(injected.candidate_graph.values[span.node_id].normalized_value)
        )

    pair = MatchedNegativeTextBuilder().build(reference, injected, rendered)
    negative = UnifiedHallucinationAnnotator().annotate_negative(
        pair.negative,
        positive,
    )
    h_record, n_record = UnifiedRecordBuilder().build_pair(
        reference,
        injected,
        pair,
        positive,
        negative,
    )
    controls = {
        item["pair_occurrence_id"]: item
        for item in n_record.data["control_spans"]
    }
    for span in h_record.data["hallucination_spans"]:
        if not span["diff_opcodes"]:
            continue
        context_start, context_end = span["serialized_context_span"]
        h_context = h_record.data["serialized"]["text"][context_start:context_end]
        control = controls[span["pair_occurrence_id"]]
        n_start, n_end = control["serialized_context_span"]
        n_context = n_record.data["serialized"]["text"][n_start:n_end]
        assert _canonical(h_context) == _canonical(
            str(injected.candidate_graph.values[span["node_id"]].normalized_value)
        )
        assert n_context == str(
            reference.state_dag.values[span["node_id"]].normalized_value
        )


def test_molecular_operator_mean_spans_are_below_fifteen_characters(
    all_references,
    fragment_pool,
):
    measured = defaultdict(list)
    planner = UnifiedHallucinationPlanner(fragment_pool)
    injector = UnifiedHallucinationInjector()
    annotator = UnifiedHallucinationAnnotator()
    operators = {
        "smiles_atom_replacement",
        "smiles_terminal_atom_deletion",
        "final_answer_to_product",
        "product_to_final_answer",
    }
    for reference in all_references:
        plan = planner.plan(reference, variant_index=0)
        injected = injector.apply(reference.state_dag, plan)
        rendered = DeterministicTextRenderer().render(reference, injected)
        for span in annotator.annotate(rendered, injected).spans:
            if span.operator in operators:
                measured[span.operator].append(len(span.text))

    bond_config = replace(
        DEFAULT_HALLUCINATION_CONFIG,
        edit_count_mode="fixed",
        fixed_edit_count=1,
        include_final_answer=False,
        editable_nodes_by_subtask=_only_product_targets(),
        smiles_mutation_operators=("smiles_bond_order_change",),
        smiles_similarity_min=0.0,
    )
    bond_planner = UnifiedHallucinationPlanner(fragment_pool, bond_config)
    bond_injector = UnifiedHallucinationInjector(bond_config)
    for reference in all_references:
        try:
            plan = bond_planner.plan(reference, variant_index=0)
            injected = bond_injector.apply(reference.state_dag, plan)
        except ValueError:
            continue
        rendered = DeterministicTextRenderer().render(reference, injected)
        for span in annotator.annotate(rendered, injected).spans:
            if span.operator == "smiles_bond_order_change":
                measured[span.operator].append(len(span.text))
        if len(measured["smiles_bond_order_change"]) >= 30:
            break

    for operator in (*sorted(operators), "smiles_bond_order_change"):
        assert measured[operator], operator
        assert sum(measured[operator]) / len(measured[operator]) < 15, operator
