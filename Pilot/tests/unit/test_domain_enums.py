"""Tests for the stable MolHalluLens enum vocabulary."""

from __future__ import annotations

import pytest

from molhallulens.core.enums import (
    CausalRole,
    DependencyType,
    EditErrorSubtype,
    EvidenceRelation,
    HallucinationType,
    NodeRole,
    PropagationPolicy,
    Visibility,
)
from molhallulens.config.loader import load_config_bundle


def test_frozen_taxonomy_ids_and_axes_are_complete() -> None:
    assert [item.value for item in HallucinationType] == list(range(8))
    assert [item.value for item in EditErrorSubtype] == [
        f"E{index:02d}" for index in range(1, 16)
    ]
    assert set(CausalRole) == {
        CausalRole.ROOT,
        CausalRole.PROPAGATED_FALSE,
        CausalRole.PROPAGATED_CONDITIONAL,
        CausalRole.TERMINAL,
    }
    assert len(EvidenceRelation) == 5
    assert NodeRole.INTERNAL_TRUTH.value == "internal_truth"
    assert Visibility.BUILD_ONLY.value == "build_only"
    assert DependencyType.EDIT_PRODUCES.value == "edit_produces"
    with pytest.raises(TypeError, match="immutable"):
        CausalRole.ROOT.sidecar = []  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        CausalRole.ROOT.__dict__["sidecar"] = []
    with pytest.raises(TypeError, match="immutable"):
        del CausalRole.ROOT._sort_order_


def test_local_and_stop_are_the_same_policy_with_stable_dataset_names() -> None:
    assert PropagationPolicy.LOCAL is PropagationPolicy.STOP
    assert [
        policy.dataset_name
        for policy in (
            PropagationPolicy.STOP,
            PropagationPolicy.PARTIAL,
            PropagationPolicy.FULL_CF,
            PropagationPolicy.TERMINAL,
        )
    ] == ["LOCAL", "PARTIAL", "FULL_CF", "TERMINAL"]
    assert PropagationPolicy.from_dataset_name("LOCAL") is PropagationPolicy.STOP
    with pytest.raises(ValueError, match="Unknown propagation"):
        PropagationPolicy.from_dataset_name("FULL")


def test_domain_enums_match_the_frozen_configuration() -> None:
    config = load_config_bundle()

    assert [(item.id, item.name) for item in config.labels.semantic_types] == [
        (item.value, item.name) for item in HallucinationType
    ]
    assert [(item.code, item.name) for item in config.labels.editing_subtypes] == [
        (item.value, item.name) for item in EditErrorSubtype
    ]
    assert config.labels.causal_roles == tuple(item.value for item in CausalRole)
    assert config.labels.evidence_relations == tuple(item.value for item in EvidenceRelation)
    assert config.dataset.bundle.policies == tuple(
        item.dataset_name
        for item in (
            PropagationPolicy.STOP,
            PropagationPolicy.PARTIAL,
            PropagationPolicy.FULL_CF,
            PropagationPolicy.TERMINAL,
        )
    )
