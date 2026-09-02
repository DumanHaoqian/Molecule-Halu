"""T038 deterministic structural-operator coverage contracts."""

from __future__ import annotations

import json
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from molhallulens.modules.release.operator_coverage import (
    FROZEN_DATASET_VERSION,
    REPORT_FORMAT_VERSION,
    STRUCTURAL_OPERATOR_IDS,
    CandidateCoverageObservation,
    CoverageBackend,
    CoverageExecutionMode,
    audit_structural_operator_candidate_coverage,
    build_operator_candidate_coverage_report,
)
from molhallulens.config import load_config_bundle
from molhallulens.core import CandidateSourceType
from molhallulens.modules.error_injection import (
    AdditionPerturbator,
    DeletionPerturbator,
    PerturbatorRegistry,
    SubstitutionPerturbator,
)

REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Dataset"
    / "reports"
    / "operator_candidate_coverage.json"
)
OPERATORS_CONFIG = load_config_bundle().operators


@cache
def _registrations():
    registry = PerturbatorRegistry.from_perturbator_types(
        (AdditionPerturbator, DeletionPerturbator, SubstitutionPerturbator),
        operators_config=OPERATORS_CONFIG,
    )
    return {
        item.operator_id: item
        for item in registry.registrations_for()
        if item.operator_id in STRUCTURAL_OPERATOR_IDS
    }


def _observation_fixture() -> tuple[CandidateCoverageObservation, ...]:
    values: list[CandidateCoverageObservation] = []
    for operator_id, registration in sorted(_registrations().items()):
        root = min(registration.spec.root_fields)
        common = {
            "operator_id": operator_id,
            "subtask": registration.subtask,
            "operator_family": registration.operator_family,
            "target_root": root,
            "origin_id": f"fixture.{registration.subtask.value}",
        }
        for backend, source in (
            (CoverageBackend.RULE, CandidateSourceType.RULE),
            (CoverageBackend.RDKIT, CandidateSourceType.RDKIT),
            (CoverageBackend.HYBRID, CandidateSourceType.RULE),
        ):
            values.extend(
                (
                    CandidateCoverageObservation(
                        **common,
                        request_id=f"{operator_id}:{backend.value}:failure",
                        backend=backend,
                        execution_mode=CoverageExecutionMode.DETERMINISTIC_LOCAL,
                        raw_proposals=0,
                        valid_candidates=0,
                        rejection_counts={"NO_VALID_CANDIDATE": 1},
                        failure_code="NO_VALID_CANDIDATE",
                    ),
                    CandidateCoverageObservation(
                        **common,
                        request_id=f"{operator_id}:{backend.value}:success",
                        backend=backend,
                        execution_mode=CoverageExecutionMode.DETERMINISTIC_LOCAL,
                        raw_proposals=2,
                        valid_candidates=1,
                        source_counts={source.value: 1},
                        rejection_counts={"DUPLICATE": 1},
                    ),
                )
            )
        for backend in (CoverageBackend.LLM, CoverageBackend.HYBRID):
            values.append(
                CandidateCoverageObservation(
                    **common,
                    request_id=f"{operator_id}:{backend.value}:mock",
                    backend=backend,
                    execution_mode=CoverageExecutionMode.DETERMINISTIC_MOCK,
                    raw_proposals=1,
                    valid_candidates=1,
                    source_counts={CandidateSourceType.LLM.value: 1},
                    llm_materially_participated=True,
                    fixture_id="t038.fixture.v1",
                )
            )
    return tuple(values)


def test_aggregation_is_order_independent_and_failure_rates_are_exact() -> None:
    observations = _observation_fixture()
    report = build_operator_candidate_coverage_report(
        observations,
        operators_config=OPERATORS_CONFIG,
    )
    reversed_report = build_operator_candidate_coverage_report(
        reversed(observations),
        operators_config=OPERATORS_CONFIG,
    )

    assert report.to_json_bytes() == reversed_report.to_json_bytes()
    assert report.operator_count == len(STRUCTURAL_OPERATOR_IDS) == 23
    assert report.all_structural_operators_covered
    for operator in report.operators:
        local = tuple(
            row
            for row in operator.backends
            if row.execution_mode is CoverageExecutionMode.DETERMINISTIC_LOCAL
        )
        assert len(local) == 3
        assert all(row.requests == 2 for row in local)
        assert all(row.failures == 1 for row in local)
        assert all(row.failure_rate == 0.5 for row in local)
        assert all(row.request_coverage == 0.5 for row in local)

    participation = report.llm_participation
    assert participation.deterministic_mock_fraction == 1.0
    assert participation.live_fraction is None
    assert participation.target_met is None
    assert not participation.mock_counts_toward_live_fraction


def test_observation_rejects_mock_or_failure_accounting_drift() -> None:
    registration = next(iter(_registrations().values()))
    common = {
        "operator_id": registration.operator_id,
        "subtask": registration.subtask,
        "operator_family": registration.operator_family,
        "target_root": min(registration.spec.root_fields),
        "origin_id": "fixture.invalid",
        "request_id": "fixture.invalid.request",
        "backend": CoverageBackend.LLM,
        "execution_mode": CoverageExecutionMode.DETERMINISTIC_MOCK,
        "raw_proposals": 1,
        "valid_candidates": 1,
        "source_counts": {CandidateSourceType.LLM.value: 1},
        "fixture_id": "t038.fixture.v1",
    }
    with pytest.raises(ValueError, match="LLM participation"):
        CandidateCoverageObservation(**common)
    with pytest.raises(ValueError, match="zero-valid"):
        CandidateCoverageObservation(
            **{
                **common,
                "raw_proposals": 0,
                "valid_candidates": 0,
                "source_counts": {},
                "llm_materially_participated": False,
            }
        )


def test_frozen_real_corpus_report_has_bounded_coverage_and_no_fake_live_llm() -> None:
    raw = REPORT_PATH.read_bytes()
    payload = json.loads(raw)

    assert raw.endswith(b"\n")
    assert payload["report_format_version"] == REPORT_FORMAT_VERSION
    assert payload["dataset_version"] == FROZEN_DATASET_VERSION
    assert payload["operator_count"] == 23
    assert payload["all_structural_operators_covered"] is True
    assert tuple(item["operator_id"] for item in payload["operators"]) == tuple(
        sorted(STRUCTURAL_OPERATOR_IDS)
    )
    assert payload["audit_execution"]["network_requests"] == 0
    assert payload["audit_execution"]["secret_reads"] == 0
    assert payload["audit_execution"]["deterministic_mock_is_live"] is False

    for operator in payload["operators"]:
        assert operator["verified_candidate_covered"] is True
        assert operator["verified_sources"]
        roots = set(operator["root_fields"])
        rows = operator["backends"]
        local = [row for row in rows if row["execution_mode"] == "deterministic_local"]
        mocked = [row for row in rows if row["execution_mode"] == "deterministic_mock"]
        live = [row for row in rows if row["execution_mode"] == "offline_not_executed"]
        assert {row["backend"] for row in local} == {"rule", "rdkit", "hybrid"}
        assert {row["backend"] for row in mocked} == {"llm", "hybrid"}
        assert {row["backend"] for row in live} == {"llm", "hybrid"}
        assert any(row["valid_candidates"] > 0 for row in local)
        for row in local:
            assert 1 <= row["requests"] <= 50 * len(roots)
            assert 1 <= row["origin_count"] <= 50
            assert row["raw_proposals"] >= row["valid_candidates"]
            assert set(row["target_roots"]) == roots
            assert row["failures"] == row["requests"] - row["successful_requests"]
            assert row["failure_rate"] == row["failures"] / row["requests"]
            assert row["request_coverage"] == (
                row["successful_requests"] / row["requests"]
            )
        for row in mocked:
            assert row["execution_status"] == "tested"
            assert row["source_counts"].get(CandidateSourceType.LLM.value, 0) > 0
            assert row["mock_counts_toward_live_participation"] is False
        for row in live:
            assert row["execution_status"] == "offline_not_executed"
            assert row["requests"] == row["valid_candidates"] == 0
            assert row["source_counts"] == {}

    combined = Counter(payload["deterministic_local_source_counts"])
    combined.update(payload["deterministic_mock_source_counts"])
    combined.update(payload["live_source_counts"])
    assert dict(sorted(combined.items())) == payload["aggregate_source_counts"]
    participation = payload["llm_participation"]
    assert participation["target_fraction"] == 0.5
    assert participation["live_execution_status"] == "offline_not_executed"
    assert participation["live_fraction"] is None
    assert participation["target_met"] is None
    assert participation["deterministic_mock_fraction"] == 1.0
    assert participation["mock_counts_toward_live_fraction"] is False


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_real_corpus_search_bound_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        audit_structural_operator_candidate_coverage(
            Path("Dataset"),
            max_origins_per_subtask=value,
        )
