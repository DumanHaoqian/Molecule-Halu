from __future__ import annotations

import json
from types import MappingProxyType

import pytest
from pydantic import ValidationError

import molhallulens.infrastructure.providers.poe.chemistry_tools as chemistry_tools_module
from molhallulens.infrastructure.providers.poe.chemistry_tools import (
    CHEMISTRY_TOOL_HANDLERS,
    CHEMISTRY_TOOL_RESULT_VERSION,
    ChemistryTools,
)
from molhallulens.infrastructure.providers.poe.schemas import CHEMISTRY_TOOL_NAMES

MAPPED_BROMOETHANE = "[CH3:1][CH2:2][Br:3]"


def test_fixed_allow_list_matches_the_nine_strict_t031_models() -> None:
    assert tuple(CHEMISTRY_TOOL_HANDLERS) == CHEMISTRY_TOOL_NAMES
    assert CHEMISTRY_TOOL_NAMES == (
        "inspect_atoms",
        "enumerate_alternate_anchors",
        "analyze_smiles",
        "find_group_at_anchor",
        "enumerate_removable_groups",
        "simulate_edit",
        "compute_descriptors",
        "compare_molecules",
        "check_candidate_signature",
    )
    with pytest.raises(TypeError):
        CHEMISTRY_TOOL_HANDLERS["network_lookup"] = lambda _: {}  # type: ignore[index]


def test_unknown_or_invalid_args_never_invoke_a_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def spy(arguments: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"ok": True, "reject_reasons": []}

    handlers = dict(CHEMISTRY_TOOL_HANDLERS)
    handlers["inspect_atoms"] = spy
    monkeypatch.setattr(
        chemistry_tools_module,
        "CHEMISTRY_TOOL_HANDLERS",
        MappingProxyType(handlers),
    )
    tools = ChemistryTools()

    with pytest.raises(ValueError, match="unknown chemistry tool"):
        tools.dispatch("shell", {})
    with pytest.raises(ValidationError):
        tools.dispatch("inspect_atoms", {"smiles": "CC", "unexpected": True})
    with pytest.raises(ValidationError):
        tools.dispatch("inspect_atoms", {"smiles": "CC", "atom_indices": [True]})
    assert calls == 0

    tools.dispatch("inspect_atoms", {"smiles": "CC"})
    assert calls == 1


def test_result_and_cache_key_are_frozen_canonical_contracts() -> None:
    tools = ChemistryTools()
    first = tools.dispatch(
        "inspect_atoms",
        {"smiles": MAPPED_BROMOETHANE, "atom_indices": [2, 0]},
    )
    second = tools.dispatch(
        "inspect_atoms",
        {"atom_indices": [0, 2], "smiles": MAPPED_BROMOETHANE},
    )

    assert first is second
    assert tools.cache_size == 1
    assert first.result_version == CHEMISTRY_TOOL_RESULT_VERSION
    assert len(first.cache_key) == 64
    assert first.to_json_bytes() == second.to_json_bytes()
    assert json.loads(first.to_json_bytes()) == first.to_json_dict()

    detached = first.result
    detached["atoms"] = []
    assert len(first.result["atoms"]) == 2
    with pytest.raises((AttributeError, TypeError)):
        first.cache_key = "0" * 64  # type: ignore[misc]


def test_atom_anchor_and_group_inspection_is_deterministic() -> None:
    tools = ChemistryTools()
    inspected = tools.dispatch("inspect_atoms", {"smiles": MAPPED_BROMOETHANE}).result
    assert inspected["ok"] is True
    assert [atom["atom_map"] for atom in inspected["atoms"]] == [1, 2, 3]
    assert inspected["atoms"][1]["element"] == "C"

    anchors = tools.dispatch(
        "enumerate_alternate_anchors",
        {
            "source_smiles": MAPPED_BROMOETHANE,
            "reference_anchor_idx": 1,
        },
    ).result
    assert [item["anchor_idx"] for item in anchors["anchors"]] == [2]

    found = tools.dispatch(
        "find_group_at_anchor",
        {"source_smiles": MAPPED_BROMOETHANE, "anchor_idx": 2},
    ).result
    enumerated = tools.dispatch(
        "enumerate_removable_groups",
        {"source_smiles": MAPPED_BROMOETHANE, "anchor_idx": 2},
    ).result
    assert found["groups"] == enumerated["groups"]
    assert {item["fragment_smiles"] for item in found["groups"]} == {"Br", "C"}
    bromine = next(item for item in found["groups"] if item["fragment_smiles"] == "Br")
    assert bromine["occurrence_atom_maps"] == [3]
    assert bromine["bond_type"] == "SINGLE"


def test_analysis_descriptors_and_comparators_reuse_strict_chemistry() -> None:
    tools = ChemistryTools()
    analysis = tools.dispatch(
        "analyze_smiles", {"smiles": "CCO.[Na+]", "fragment_policy": "largest_heavy"}
    ).result
    descriptors = analysis["descriptors"]
    assert analysis["ok"] is True
    assert descriptors["canonical_smiles"] == "CCO"
    assert descriptors["heavy_atom_count"] == 3
    assert descriptors["fragment_policy"] == "largest_heavy"

    direct = tools.dispatch(
        "compute_descriptors", {"smiles": "OCC", "fragment_policy": "keep_all"}
    ).result
    assert direct["descriptors"]["canonical_smiles"] == "CCO"
    assert direct["descriptors"]["hydrogen_bond_donor_count"] == 1

    graph = tools.dispatch(
        "compare_molecules",
        {"left_smiles": "CCO", "right_smiles": "OCC"},
    ).result
    exact = tools.dispatch(
        "compare_molecules",
        {"left_smiles": "CCO", "right_smiles": "OCC", "comparator": "exact"},
    ).result
    assert graph["equivalent"] is True
    assert exact["equivalent"] is False
    assert graph["canonical_left_smiles"] == graph["canonical_right_smiles"]


@pytest.mark.parametrize(
    ("arguments", "expected_product", "removed", "added_count"),
    (
        (
            {
                "family": "add",
                "source_smiles": "[CH3:1][CH3:2]",
                "anchor_idx": 2,
                "add_fragment_smiles": "Cl",
                "fragment_attachment_atom": 0,
                "bond_type": "SINGLE",
            },
            "CCCl",
            [],
            1,
        ),
        (
            {
                "family": "delete",
                "source_smiles": MAPPED_BROMOETHANE,
                "anchor_idx": 2,
                "remove_group_smiles": "Br",
            },
            "CC",
            [3],
            0,
        ),
        (
            {
                "family": "substitute",
                "source_smiles": MAPPED_BROMOETHANE,
                "anchor_idx": 2,
                "remove_group_smiles": "Br",
                "add_fragment_smiles": "Cl",
                "fragment_attachment_atom": 0,
                "bond_type": "SINGLE",
            },
            "CCCl",
            [3],
            1,
        ),
    ),
)
def test_simulate_edit_strictly_replays_and_returns_graph_diff_and_descriptors(
    arguments: dict[str, object],
    expected_product: str,
    removed: list[int],
    added_count: int,
) -> None:
    result = ChemistryTools().dispatch("simulate_edit", arguments).result
    assert result["ok"] is True
    assert result["reject_reasons"] == []
    assert [item["product_smiles"] for item in result["products"]] == [expected_product]
    product = result["products"][0]
    assert product["descriptors"]["canonical_smiles"] == expected_product
    assert product["graph_diff"]["canonical_product_smiles"] == expected_product
    assert product["graph_diff"]["removed_atom_maps"] == removed
    assert len(product["graph_diff"]["added_atoms"]) == added_count


def test_simulate_edit_rejects_unmapped_sources_and_non_occurrences() -> None:
    tools = ChemistryTools()
    unmapped = tools.dispatch(
        "simulate_edit",
        {
            "family": "add",
            "source_smiles": "CC",
            "anchor_idx": 1,
            "add_fragment_smiles": "Cl",
            "fragment_attachment_atom": 0,
            "bond_type": "SINGLE",
        },
    ).result
    missing = tools.dispatch(
        "simulate_edit",
        {
            "family": "delete",
            "source_smiles": MAPPED_BROMOETHANE,
            "anchor_idx": 2,
            "remove_group_smiles": "I",
        },
    ).result
    assert unmapped["ok"] is False
    assert unmapped["reject_reasons"][0]["code"] == "source_atom_maps_required"
    assert missing["ok"] is False
    assert missing["products"] == []
    assert missing["reject_reasons"][0]["code"] == "remove_group_not_found"


def test_candidate_signature_is_proven_only_against_replayed_products() -> None:
    tools = ChemistryTools()
    base = {
        "family": "substitute",
        "source_smiles": MAPPED_BROMOETHANE,
        "anchor_idx": 2,
        "remove_group_smiles": "Br",
        "add_fragment_smiles": "Cl",
        "fragment_attachment_atom": 0,
        "bond_type": "SINGLE",
    }
    valid = tools.dispatch(
        "check_candidate_signature",
        {**base, "candidate_product_smiles": "ClCC"},
    ).result
    invented = tools.dispatch(
        "check_candidate_signature",
        {**base, "candidate_product_smiles": "CCC"},
    ).result
    malformed = tools.dispatch(
        "check_candidate_signature",
        {**base, "candidate_product_smiles": "C("},
    ).result

    assert valid["valid"] is True
    assert valid["matched_product_smiles"] == "CCCl"
    assert valid["graph_diff"]["removed_atom_maps"] == [3]
    assert invented["valid"] is False
    assert invented["reject_reasons"][-1]["code"] == "candidate_not_replay_product"
    assert malformed["valid"] is False
    assert malformed["reject_reasons"][0]["code"] == "candidate_invalid_smiles"


def test_dispatch_call_strictly_parses_envelope_before_execution() -> None:
    tools = ChemistryTools()
    result = tools.dispatch_call(
        {
            "tool": "compute_descriptors",
            "arguments": {"smiles": "CCO"},
        }
    )
    assert result.result["descriptors"]["canonical_smiles"] == "CCO"
    with pytest.raises(ValidationError):
        tools.dispatch_call(
            {
                "tool": "compute_descriptors",
                "arguments": {"smiles": "CCO"},
                "accepted": True,
            }
        )
