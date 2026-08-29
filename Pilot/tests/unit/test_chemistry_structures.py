"""Tests for strict RDKit molecular identity and descriptor utilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from molhallulens.chemistry import (
    FragmentPolicy,
    MoleculeErrorCode,
    MoleculeParseError,
    canonicalize_smiles,
    compute_descriptors,
    fragment_graph_equivalent,
    isomeric_graph_equivalent,
    murcko_scaffold_smiles,
    select_main_fragment,
)


def test_canonicalization_removes_atom_maps_but_preserves_chemistry() -> None:
    mapped = "[CH3:7][OH:2]"
    assert canonicalize_smiles(mapped) == canonicalize_smiles("CO")
    assert canonicalize_smiles("[OH:91][CH3:4]") == canonicalize_smiles("CO")
    assert ":7" not in canonicalize_smiles(mapped)
    assert mapped == "[CH3:7][OH:2]"
    assert canonicalize_smiles(canonicalize_smiles(mapped)) == canonicalize_smiles(
        mapped
    )

    isotope = canonicalize_smiles("[13CH3:7][OH:2]")
    assert isotope == canonicalize_smiles("[13CH3]O")
    assert isotope != canonicalize_smiles("CO")
    assert canonicalize_smiles("C[NH3+]") != canonicalize_smiles("CN")
    assert canonicalize_smiles("c1ccccc1") == canonicalize_smiles("C1=CC=CC=C1")


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("CCO", "OCC"),
        ("C(O)C", "CCO"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("C1CCCCC1", "C2CCCCC2"),
        ("CCO.[Na+]", "[Na+].OCC"),
        ("[CH3:8][OH:2]", "CO"),
    ),
)
def test_equivalent_noncanonical_isomeric_graphs_are_recognized(
    left: str, right: str
) -> None:
    assert left != right
    assert isomeric_graph_equivalent(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("CCO", "COC"),
        ("CN", "C[NH3+]"),
        ("CC(=O)C", "CC(O)=C"),
        ("CCO", "CCO.[Na+]"),
    ),
)
def test_chemically_distinct_graphs_are_not_equivalent(left: str, right: str) -> None:
    assert not isomeric_graph_equivalent(left, right)


def test_stereochemistry_is_part_of_identity() -> None:
    assert isomeric_graph_equivalent("N[C@@H](C)C(=O)O", "C[C@H](N)C(O)=O")
    assert not isomeric_graph_equivalent("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O")
    assert not isomeric_graph_equivalent("N[C@@H](C)C(=O)O", "NC(C)C(=O)O")
    assert not isomeric_graph_equivalent("F/C=C/F", "F/C=C\\F")
    assert not isomeric_graph_equivalent("F/C=C/F", "FC=CF")


@pytest.mark.parametrize(
    ("invalid", "code"),
    (
        (None, MoleculeErrorCode.INVALID_INPUT_TYPE),
        (1, MoleculeErrorCode.INVALID_INPUT_TYPE),
        ("", MoleculeErrorCode.EMPTY_SMILES),
        ("   ", MoleculeErrorCode.EMPTY_SMILES),
        ("C1CC", MoleculeErrorCode.SMILES_PARSE_FAILED),
        ("[CH5]", MoleculeErrorCode.SMILES_SANITIZE_FAILED),
    ),
)
def test_invalid_or_unsanitized_smiles_fail_with_stable_codes(
    invalid: object, code: MoleculeErrorCode
) -> None:
    with pytest.raises(MoleculeParseError) as captured:
        canonicalize_smiles(invalid)  # type: ignore[arg-type]
    assert captured.value.code is code
    assert captured.value.input_length == (len(invalid) if type(invalid) is str else None)
    assert "[CH5]" not in str(captured.value)


def test_equivalence_never_conflates_invalid_input_with_non_equivalence() -> None:
    with pytest.raises(MoleculeParseError) as captured:
        isomeric_graph_equivalent("CCO", "[CH5]")
    assert captured.value.code is MoleculeErrorCode.SMILES_SANITIZE_FAILED


def test_main_fragment_policy_is_explicit_order_invariant_and_deterministic() -> None:
    sodium_salt = select_main_fragment("CCO.[Na+]")
    assert sodium_salt.canonical_smiles == canonicalize_smiles("CCO")
    assert sodium_salt.heavy_atom_count == 3
    assert sodium_salt.contains_carbon
    assert sodium_salt.input_fragment_count == 2
    assert select_main_fragment("[Na+].CCO") == sodium_salt

    chloride_salt = select_main_fragment("CC[NH3+].[Cl-]")
    assert chloride_salt.canonical_smiles == canonicalize_smiles("CC[NH3+]")
    assert select_main_fragment("[Cl-].CC[NH3+]") == chloride_salt

    # Equal heavy-atom and organic status uses ascending canonical lexical form.
    assert select_main_fragment("CN.CC").canonical_smiles == "CC"
    assert canonicalize_smiles("CCO.[Na+]") != canonicalize_smiles("CCO")
    assert canonicalize_smiles(
        "CCO.[Na+]", fragment_policy=FragmentPolicy.LARGEST_HEAVY
    ) == canonicalize_smiles("CCO")
    with pytest.raises(TypeError, match="FragmentPolicy"):
        canonicalize_smiles("CCO", fragment_policy="keep_all")  # type: ignore[arg-type]


def test_descriptors_use_frozen_explicit_definitions() -> None:
    ethanol = compute_descriptors("CCO")
    assert ethanol.canonical_smiles == "CCO"
    assert ethanol.fragment_policy is FragmentPolicy.KEEP_ALL
    assert ethanol.fragment_count == 1
    assert ethanol.heavy_atom_count == 3
    assert ethanol.ring_count == 0
    assert ethanol.aromatic_ring_count == 0
    assert ethanol.formal_charge == 0
    assert ethanol.heteroatom_counts == ((8, 1),)
    assert ethanol.molecular_weight == pytest.approx(46.069, abs=1e-6)
    assert ethanol.exact_molecular_weight == pytest.approx(46.041864812, abs=1e-9)
    assert ethanol.rotatable_bond_count == 0
    assert ethanol.hydrogen_bond_donor_count == 1
    assert ethanol.hydrogen_bond_acceptor_count == 1
    assert ethanol.topological_polar_surface_area == pytest.approx(20.23, abs=1e-9)
    assert compute_descriptors("[H]O[H]").heavy_atom_count == 1
    assert compute_descriptors("c1ccccc1").ring_count == 1
    assert compute_descriptors("c1ccc2ccccc2c1").ring_count == 2
    assert compute_descriptors("[CH3:7][OH:2]") == compute_descriptors("CO")
    with pytest.raises(FrozenInstanceError):
        ethanol.heavy_atom_count = 4  # type: ignore[misc]


def test_descriptor_fragment_scope_never_silently_drops_counterions() -> None:
    whole = compute_descriptors("CCO.[Na+]", fragment_policy=FragmentPolicy.KEEP_ALL)
    main = compute_descriptors(
        "CCO.[Na+]", fragment_policy=FragmentPolicy.LARGEST_HEAVY
    )
    assert whole.fragment_count == 2
    assert whole.heavy_atom_count == 4
    assert whole.formal_charge == 1
    assert main.fragment_count == 1
    assert main.heavy_atom_count == 3
    assert main.formal_charge == 0


def test_murcko_scaffold_is_map_free_scoped_and_none_for_acyclic_molecules() -> None:
    benzene = canonicalize_smiles("c1ccccc1")
    assert murcko_scaffold_smiles(
        "Cc1ccccc1", fragment_policy=FragmentPolicy.KEEP_ALL
    ) == benzene
    assert murcko_scaffold_smiles(
        "Oc1ccccc1", fragment_policy=FragmentPolicy.KEEP_ALL
    ) == benzene
    assert murcko_scaffold_smiles(
        "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1",
        fragment_policy=FragmentPolicy.KEEP_ALL,
    ) == benzene
    assert murcko_scaffold_smiles(
        "CCO", fragment_policy=FragmentPolicy.KEEP_ALL
    ) is None
    assert murcko_scaffold_smiles(
        "Cc1ccccc1.[Na+]", fragment_policy=FragmentPolicy.LARGEST_HEAVY
    ) == benzene
    with pytest.raises(TypeError, match="required keyword-only"):
        murcko_scaffold_smiles("c1ccccc1")  # type: ignore[call-arg]


def test_fragment_comparator_is_graph_based_but_not_attachment_aware() -> None:
    assert fragment_graph_equivalent("C(C)=O", "CC=O")
    assert not fragment_graph_equivalent("Br", "Cl")
