"""RDKit acceptance checks over all 150 molecule-editing origins."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import rdkit

from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.infrastructure.chemistry import (
    FragmentPolicy,
    canonicalize_smiles,
    compute_descriptors,
    isomeric_graph_equivalent,
    murcko_scaffold_smiles,
)


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


def test_pinned_rdkit_parses_and_reconciles_all_150_origins() -> None:
    assert rdkit.__version__ == "2025.09.6"
    records = ChemCoTMolEditAdapter().load(DATASET_ROOT)
    raw_drift_ids: list[str] = []

    for record in records:
        process = record.process_record
        state = process["parsed_reference_state"]
        source = record.raw_record["indexed_smiles"]
        gt = record.raw_record["gt_smiles"]
        answer = process["answer_smiles"]
        product_field = next(
            field
            for field in state
            if field.endswith("_product_smiles")
        )
        product = state[product_field]

        # Every public structure is strictly parsed and canonicalized.
        for smiles in (source, gt, answer, product):
            assert canonicalize_smiles(smiles)
        assert isomeric_graph_equivalent(answer, gt)
        assert isomeric_graph_equivalent(product, answer)
        if answer != gt:
            raw_drift_ids.append(record.anonymous_sample_id)

        source_descriptors = compute_descriptors(source)
        product_descriptors = compute_descriptors(product)
        assert source_descriptors.heavy_atom_count == state["rdkit_src_heavy"]
        assert product_descriptors.heavy_atom_count == state["rdkit_prod_heavy"]
        assert source_descriptors.ring_count == state["rdkit_src_rings"]
        assert product_descriptors.ring_count == state["rdkit_prod_rings"]

        fragment_bindings = {
            "add_pilot_origin": (
                ("step2_frag_smiles", "rdkit_frag_heavy"),
            ),
            "delete_pilot_origin": (
                ("step2_remove_smiles", "rdkit_group_heavy"),
            ),
            "substitute_pilot_origin": (
                ("step1_remove_group_smiles", "rdkit_remove_heavy"),
                ("step1_add_fragment_smiles", "rdkit_add_heavy"),
            ),
        }[record.pilot_subtask]
        for fragment_field, count_field in fragment_bindings:
            assert compute_descriptors(
                state[fragment_field]
            ).heavy_atom_count == state[count_field]

    assert len(records) == 150
    assert len(raw_drift_ids) == 7
    assert raw_drift_ids == sorted(raw_drift_ids)


def test_canonical_source_and_murcko_inventories_are_deterministic() -> None:
    records = ChemCoTMolEditAdapter().load(DATASET_ROOT)
    canonical_groups: dict[str, list[str]] = defaultdict(list)
    scaffold_groups: dict[str | None, list[str]] = defaultdict(list)
    for record in records:
        source = record.raw_record["indexed_smiles"]
        canonical_groups[canonicalize_smiles(source)].append(
            record.anonymous_sample_id
        )
        scaffold_groups[
            murcko_scaffold_smiles(
                source,
                fragment_policy=FragmentPolicy.LARGEST_HEAVY,
            )
        ].append(record.anonymous_sample_id)

    canonical_duplicates = sorted(
        tuple(sorted(ids)) for ids in canonical_groups.values() if len(ids) > 1
    )
    scaffold_duplicates = sorted(
        tuple(sorted(ids)) for ids in scaffold_groups.values() if len(ids) > 1
    )
    assert len(canonical_groups) == 146
    assert len(canonical_duplicates) == 3
    assert sum(map(len, canonical_duplicates)) == 7
    # Strict Bemis-Murcko is broader than exact canonical-source grouping.
    assert len(scaffold_groups) == 143
    assert len(scaffold_duplicates) == 6
    assert sum(map(len, scaffold_duplicates)) == 13
