"""Unit contracts for deterministic T027 leakage grouping."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from molhallulens.builders.leakage_groups import (
    LeakageGroupIndex,
    LeakageIdentity,
    LeakageReason,
    stable_leakage_group_id,
)
from molhallulens.chemistry import (
    FragmentPolicy,
    generic_murcko_scaffold_smiles,
    murcko_scaffold_smiles,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(
    origin_id: str,
    *,
    source: str,
    gt: str,
    murcko: str | None,
    generic: str | None,
) -> LeakageIdentity:
    return LeakageIdentity(
        anonymous_sample_id=origin_id,
        canonical_source_sha256=_digest(source),
        canonical_gt_sha256=_digest(gt),
        murcko_scaffold_sha256=None if murcko is None else _digest(murcko),
        generic_murcko_scaffold_sha256=(None if generic is None else _digest(generic)),
    )


def _synthetic_identities() -> tuple[LeakageIdentity, ...]:
    return (
        _identity(
            "origin.a",
            source="shared-source",
            gt="gt-a",
            murcko="murcko-a",
            generic="shared-generic",
        ),
        _identity(
            "origin.b",
            source="shared-source",
            gt="shared-gt",
            murcko="murcko-b",
            generic="shared-generic",
        ),
        _identity(
            "origin.c",
            source="source-c",
            gt="shared-gt",
            murcko="murcko-c",
            generic="generic-c",
        ),
        _identity(
            "origin.none-1",
            source="source-none-1",
            gt="gt-none-1",
            murcko=None,
            generic=None,
        ),
        _identity(
            "origin.none-2",
            source="source-none-2",
            gt="gt-none-2",
            murcko=None,
            generic=None,
        ),
    )


def test_group_id_uses_full_sha256_of_version_and_nul_joined_sorted_ids() -> None:
    expected = hashlib.sha256(b"pilot_v1\0origin.a\0origin.z").hexdigest()
    assert stable_leakage_group_id("pilot_v1", ("origin.z", "origin.a")) == expected
    assert len(expected) == 64

    with pytest.raises(ValueError):
        stable_leakage_group_id("pilot_v1", ("origin.a", "origin.a"))
    with pytest.raises(ValueError):
        stable_leakage_group_id("pilot\0v1", ("origin.a",))


def test_union_find_uses_transitive_closure_and_aggregates_edge_reasons() -> None:
    index = LeakageGroupIndex(
        dataset_version="pilot_v1",
        identities=_synthetic_identities(),
    )
    groups = {group.anonymous_sample_ids: group for group in index.groups}
    transitive = groups[("origin.a", "origin.b", "origin.c")]
    assert transitive.leakage_group_id == stable_leakage_group_id(
        "pilot_v1", transitive.anonymous_sample_ids
    )
    assert set(transitive.reasons) == {
        LeakageReason.CANONICAL_SOURCE,
        LeakageReason.CANONICAL_GT,
        LeakageReason.GENERIC_MURCKO_SCAFFOLD,
    }

    edge_ab = next(
        edge
        for edge in transitive.trigger_edges
        if (edge.left_origin_id, edge.right_origin_id) == ("origin.a", "origin.b")
    )
    assert edge_ab.reasons == (
        LeakageReason.CANONICAL_SOURCE,
        LeakageReason.GENERIC_MURCKO_SCAFFOLD,
    )
    edge_bc = next(
        edge
        for edge in transitive.trigger_edges
        if (edge.left_origin_id, edge.right_origin_id) == ("origin.b", "origin.c")
    )
    assert edge_bc.reasons == (LeakageReason.CANONICAL_GT,)


def test_none_scaffolds_are_absence_not_one_shared_identity() -> None:
    index = LeakageGroupIndex(
        dataset_version="pilot_v1",
        identities=_synthetic_identities(),
    )
    assignments = {
        item.identity.anonymous_sample_id: item for item in index.assignments
    }
    for origin_id in ("origin.none-1", "origin.none-2"):
        assignment = assignments[origin_id]
        assert assignment.leakage_group_size == 1
        assert assignment.leakage_reasons == ()
    assert (
        assignments["origin.none-1"].leakage_group_id
        != assignments["origin.none-2"].leakage_group_id
    )


def test_input_order_cannot_change_edges_components_assignments_or_ids() -> None:
    identities = _synthetic_identities()
    forward = LeakageGroupIndex(dataset_version="pilot_v1", identities=identities)
    reverse = LeakageGroupIndex(
        dataset_version="pilot_v1", identities=reversed(identities)
    )
    assert forward == reverse

    with pytest.raises(FrozenInstanceError):
        forward.groups[0].leakage_group_id = "0" * 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="unique origin IDs"):
        LeakageGroupIndex(
            dataset_version="pilot_v1",
            identities=(*identities, identities[0]),
        )


def test_generic_murcko_merges_atom_and_bond_analogues_only_explicitly() -> None:
    benzene = "c1ccccc1"
    pyridine = "c1ccncc1"
    assert murcko_scaffold_smiles(
        benzene, fragment_policy=FragmentPolicy.LARGEST_HEAVY
    ) != murcko_scaffold_smiles(pyridine, fragment_policy=FragmentPolicy.LARGEST_HEAVY)
    assert generic_murcko_scaffold_smiles(
        benzene, fragment_policy=FragmentPolicy.LARGEST_HEAVY
    ) == generic_murcko_scaffold_smiles(
        pyridine, fragment_policy=FragmentPolicy.LARGEST_HEAVY
    )
    assert (
        generic_murcko_scaffold_smiles(
            "CCO", fragment_policy=FragmentPolicy.LARGEST_HEAVY
        )
        is None
    )
    with pytest.raises(TypeError):
        generic_murcko_scaffold_smiles("c1ccccc1")  # type: ignore[call-arg]
