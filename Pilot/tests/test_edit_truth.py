"""Graph-derived EditTruth acceptance tests and real-corpus audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import (
    EditTruthBuildError,
    EditTruthBuilder,
    build_reference_dag,
    derive_edit_truth,
)
from molhallulens.core import EditingSubtask


DATASET_ROOT = Path(__file__).resolve().parents[1] / "Dataset"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "edit_truth" / "synthetic_cases.json"


def _fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["format_version"] == "edit_truth_fixture_v1"
    return payload


def _case(case_id: str) -> dict[str, Any]:
    return next(item for item in _fixture()["cases"] if item["case_id"] == case_id)


def _derive(case: dict[str, Any]):
    return EditTruthBuilder().derive(
        case["source_smiles"],
        case["gt_smiles"],
        anonymous_sample_id=f"fixture.{case['case_id']}",
        normalized_subtask=EditingSubtask(case["normalized_subtask"]),
        trace_anchor_indices=tuple(case["trace_anchor_indices"]),
        remove_fragment_hint=case["remove_fragment_hint"],
        add_fragment_hint=case["add_fragment_hint"],
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "addition_at_unique_oxygen",
        "deletion_of_ester_methyl",
        "halogen_substitution",
    ),
)
def test_synthetic_atom_and_fragment_diffs_are_graph_derived(case_id: str) -> None:
    case = _case(case_id)
    expected = case["expected"]
    truth = _derive(case)

    assert truth.valid_anchor_indices == tuple(expected["valid_anchor_indices"])
    assert truth.removed_atom_maps == frozenset(expected["removed_atom_maps"])
    assert tuple(sorted(atom.atomic_number for atom in truth.added_atoms)) == tuple(
        expected["added_atomic_numbers"]
    )
    assert len(truth.broken_bonds) == expected["broken_bond_count"]
    assert len(truth.formed_bonds) == expected["formed_bond_count"]
    assert (
        truth.source_descriptors.heavy_atom_count
        == expected["source_heavy_atoms"]
    )
    assert (
        truth.product_descriptors.heavy_atom_count
        == expected["product_heavy_atoms"]
    )
    assert truth.mapping_confidence == truth.mapping_evidence.confidence
    assert 0.0 <= truth.mapping_confidence <= 1.0
    assert truth.mapping_evidence.optimal_mapping_count == len(
        truth.mapping_evidence.optimal_mappings
    )
    assert truth.mapping_evidence.coverage == (
        truth.mapping_evidence.mapped_heavy_atoms
        / min(
            truth.mapping_evidence.source_heavy_atoms,
            truth.mapping_evidence.product_heavy_atoms,
        )
    )
    assert truth.mapping_evidence.ambiguity_penalty == (
        1.0 / truth.mapping_evidence.inequivalent_edit_signature_count
    )
    assert truth.mapping_evidence.confidence == (
        truth.mapping_evidence.coverage
        * truth.mapping_evidence.ambiguity_penalty
    )
    assert truth.mapping_evidence.confidence_formula.endswith("_v1")
    json.dumps(truth.to_json_dict(), sort_keys=True)

    expected_fragment_presence = {
        "addition_at_unique_oxygen": (False, True),
        "deletion_of_ester_methyl": (True, False),
        "halogen_substitution": (True, True),
    }[case_id]
    assert (truth.remove_fragment is not None) is expected_fragment_presence[0]
    assert (truth.add_fragment is not None) is expected_fragment_presence[1]
    if truth.remove_fragment is not None:
        assert (
            truth.remove_fragment.descriptors.heavy_atom_count
            == expected["remove_fragment_heavy_atoms"]
        )
    if truth.add_fragment is not None:
        assert (
            truth.add_fragment.descriptors.heavy_atom_count
            == expected["add_fragment_heavy_atoms"]
        )


def test_bond_order_change_is_one_broken_and_one_formed_bond() -> None:
    case = _case("double_to_single_bond_order_change")
    expected = case["expected"]
    truth = _derive(case)

    assert truth.removed_atom_maps == frozenset()
    assert truth.added_atoms == ()
    assert tuple(bond.bond_type.value for bond in truth.broken_bonds) == tuple(
        expected["broken_bond_types"]
    )
    assert tuple(bond.bond_type.value for bond in truth.formed_bonds) == tuple(
        expected["formed_bond_types"]
    )
    assert {
        frozenset((bond.begin, bond.end)) for bond in truth.broken_bonds
    } == {
        frozenset((bond.begin, bond.end)) for bond in truth.formed_bonds
    }


def test_symmetric_sites_are_preserved_as_an_equivalence_class() -> None:
    case = _case("symmetric_ethane_terminal_addition")
    expected = case["expected"]

    first = _derive(case)
    second = _derive(case)

    assert first == second
    assert first.valid_anchor_indices == tuple(expected["valid_anchor_indices"])
    assert first.symmetry_equivalent_anchors == tuple(
        tuple(group) for group in expected["symmetry_equivalent_anchors"]
    )
    assert first.mapping_evidence.inequivalent_edit_signature_count == 1
    assert first.mapping_evidence.optimal_mapping_count > 1
    assert len(first.mapping_evidence.optimal_mappings) > 1


@pytest.mark.parametrize("invalid_case", _fixture()["invalid_source_maps"])
def test_source_atom_maps_must_be_complete_and_unique(
    invalid_case: dict[str, Any],
) -> None:
    with pytest.raises(EditTruthBuildError) as captured:
        EditTruthBuilder().derive(
            invalid_case["source_smiles"],
            invalid_case["gt_smiles"],
            anonymous_sample_id=f"fixture.{invalid_case['case_id']}",
            normalized_subtask=EditingSubtask(invalid_case["normalized_subtask"]),
        )

    assert captured.value.report.issues
    assert all(issue.node_ids == (f"fixture.{invalid_case['case_id']}",) for issue in captured.value.report.issues)
    assert tuple(issue.code for issue in captured.value.report.issues) == (
        invalid_case["expected_error_code"],
    )


def test_trace_anchor_hint_cannot_overwrite_graph_truth() -> None:
    case = _fixture()["adversarial_trace_hint"]
    truth = _derive(case)

    assert case["graph_truth_anchor"] in truth.valid_anchor_indices
    assert case["untrusted_trace_anchor"] not in truth.valid_anchor_indices
    assert truth.mapping_evidence.trace_anchor_indices == (
        case["untrusted_trace_anchor"],
    )
    assert truth.mapping_evidence.trace_anchor_agreement is False


def test_all_150_real_origins_derive_reconciled_graph_truth() -> None:
    records = ChemCoTMolEditAdapter().load(DATASET_ROOT)
    truths = []

    for record in records:
        artifact = build_reference_dag(record)
        try:
            truth = derive_edit_truth(artifact)
        except Exception as error:
            raise AssertionError(
                f"EditTruth derivation failed for {record.anonymous_sample_id}"
            ) from error
        state = record.process_record["parsed_reference_state"]

        assert truth.source_descriptors.heavy_atom_count == state["rdkit_src_heavy"]
        assert truth.product_descriptors.heavy_atom_count == state["rdkit_prod_heavy"]
        assert truth.source_descriptors.ring_count == state["rdkit_src_rings"]
        assert truth.product_descriptors.ring_count == state["rdkit_prod_rings"]
        assert 0.0 <= truth.mapping_confidence <= 1.0
        assert truth.mapping_evidence.trace_anchor_agreement is True
        assert (
            truth.removed_atom_maps
            or truth.added_atoms
            or truth.broken_bonds
            or truth.formed_bonds
        )
        truths.append((record.anonymous_sample_id, truth))

    assert len(truths) == 150
    assert tuple(item[0] for item in truths) == tuple(sorted(item[0] for item in truths))


def test_direct_embeddings_preserve_trace_selected_non_equivalent_site() -> None:
    record = next(
        item
        for item in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if item.anonymous_sample_id == "mol_edit.add_v2.0071"
    )
    truth = derive_edit_truth(build_reference_dag(record))

    assert truth.mapping_evidence.algorithm == "direct_source_subgraph"
    assert truth.mapping_evidence.trace_anchor_indices == (30,)
    assert truth.mapping_evidence.trace_anchor_agreement is True
    assert truth.valid_anchor_indices == (30,)
    assert truth.mapping_evidence.optimal_mapping_count > 1


def test_delete_v2_0081_contains_both_removal_and_addition_graph_edits() -> None:
    record = next(
        item
        for item in ChemCoTMolEditAdapter().load(DATASET_ROOT)
        if item.anonymous_sample_id == "mol_edit.delete_v2.0081"
    )
    truth = derive_edit_truth(build_reference_dag(record))

    assert len(truth.removed_atom_maps) == 24
    assert tuple(sorted(atom.atomic_number for atom in truth.added_atoms)) == (6, 7)
    assert len(truth.broken_bonds) == 1
    assert len(truth.formed_bonds) == 1
    assert 11 in truth.valid_anchor_indices
    assert truth.remove_fragment is not None
    assert truth.add_fragment is not None
    assert truth.remove_fragment.descriptors.heavy_atom_count == 24
    assert truth.add_fragment.descriptors.heavy_atom_count == 2
