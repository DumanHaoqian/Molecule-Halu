"""Frozen-corpus acceptance checks for both T026 audit artifacts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import (
    audit_origin_split_features,
    build_origin_split_audit,
    build_reference_dag,
    derive_edit_truth,
)
from molhallulens.validation import OriginValidationInput

DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
AUDIT_PATH = DATASET_ROOT / "reports" / "origin_split_audit.json"
DISTRIBUTION_PATH = DATASET_ROOT / "reports" / "origin_split_feature_distribution.json"


@lru_cache(maxsize=1)
def _inputs() -> tuple[OriginValidationInput, ...]:
    values = []
    for record in ChemCoTMolEditAdapter().load(DATASET_ROOT):
        artifact = build_reference_dag(record)
        values.append(
            OriginValidationInput(
                record=record,
                artifact=artifact,
                edit_truth=derive_edit_truth(artifact),
            )
        )
    return tuple(values)


def test_real_150_origin_audit_matches_frozen_artifacts() -> None:
    result = build_origin_split_audit(DATASET_ROOT)

    assert result.audit.to_dict() == json.loads(AUDIT_PATH.read_text())
    assert result.feature_distribution.to_dict() == json.loads(
        DISTRIBUTION_PATH.read_text()
    )
    assert result.audit.to_json_bytes() == AUDIT_PATH.read_bytes()
    assert result.feature_distribution.to_json_bytes() == DISTRIBUTION_PATH.read_bytes()
    assert (
        result.feature_distribution.source_audit_sha256
        == hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
    )


def test_input_reordering_produces_byte_identical_typed_results() -> None:
    forward = audit_origin_split_features(_inputs())
    reverse = audit_origin_split_features(reversed(_inputs()))

    assert forward == reverse
    assert forward.audit.to_json_bytes() == reverse.audit.to_json_bytes()
    assert (
        forward.feature_distribution.to_json_bytes()
        == reverse.feature_distribution.to_json_bytes()
    )
