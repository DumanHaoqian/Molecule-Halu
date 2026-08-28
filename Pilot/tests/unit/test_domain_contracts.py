"""Cross-module acceptance checks for the public immutable domain API."""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from molhallulens import domain


CORE_DATACLASSES = (
    domain.BuildProvenance,
    domain.CandidatePatch,
    domain.CandidatePool,
    domain.CharAnnotation,
    domain.CharSpan,
    domain.ClaimLabel,
    domain.ClaimValue,
    domain.DetectorInput,
    domain.EditAction,
    domain.GraphDelta,
    domain.MutationEvent,
    domain.OperatorSpec,
    domain.OriginBundle,
    domain.PerturbationRecipe,
    domain.PerturbationResult,
    domain.RewriteBudget,
    domain.StateDAG,
    domain.StateEdge,
    domain.StateNodeSpec,
    domain.StateSchema,
    domain.TaskRecord,
    domain.TokenLabelSet,
    domain.TokenizerFingerprint,
    domain.TraceLabels,
    domain.ValidationIssue,
    domain.ValidationReport,
)


@pytest.mark.parametrize("contract", CORE_DATACLASSES, ids=lambda value: value.__name__)
def test_core_domain_contracts_are_frozen_and_slotted(contract: type[object]) -> None:
    assert is_dataclass(contract)
    assert contract.__dataclass_params__.frozen
    assert hasattr(contract, "__slots__")


def test_public_domain_api_exports_every_core_contract() -> None:
    assert {contract.__name__ for contract in CORE_DATACLASSES} <= set(domain.__all__)
