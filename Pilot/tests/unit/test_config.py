"""Tests for frozen, fail-closed project configuration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molhallulens.config.loader import (
    _read_yaml_mapping,
    default_config_directory,
    load_config_bundle,
)
from molhallulens.config.models import DatasetConfig, LLMConfig, OperatorsConfig


def _load_yaml(name: str) -> dict[str, object]:
    with (default_config_directory() / name).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_default_bundle_encodes_frozen_dataset_contract() -> None:
    bundle = load_config_bundle()

    assert bundle.dataset.dataset.origins_total == 150
    assert bundle.dataset.dataset.records_total == 1200
    assert (
        bundle.dataset.split.train_origins,
        bundle.dataset.split.validation_origins,
        bundle.dataset.split.test_origins,
    ) == (100, 25, 25)
    assert bundle.dataset.bundle.policies == ("LOCAL", "PARTIAL", "FULL_CF", "TERMINAL")
    assert bundle.dataset.detector.activation_alignment == "post_token_h_t"
    assert bundle.dataset.detector.include_gt_smiles is False
    assert bundle.llm.provider.name == "poe"
    assert bundle.llm.provider.model_id == "gpt-5.4-mini"
    assert bundle.rendering.leakage_scan.prohibited_inputs == (
        "gt_smiles",
        "hallucination_label",
        "oracle_state",
        "operator_correctness",
    )


def test_configuration_models_are_immutable() -> None:
    bundle = load_config_bundle()

    with pytest.raises(ValidationError, match="frozen"):
        bundle.dataset.dataset.origins_total = 149  # type: ignore[misc]
    with pytest.raises(TypeError):
        bundle.dataset.input.origins_per_subtask["add"] = 49  # type: ignore[index]
    with pytest.raises(TypeError):
        bundle.rendering.detector_template.delimiters["indexed_smiles"] = "<LEAKY>"  # type: ignore[index]
    assert not isinstance(bundle.dataset.input.origins_per_subtask, dict)
    assert not isinstance(bundle.rendering.detector_template.delimiters, dict)


def test_configuration_models_remain_json_serializable() -> None:
    bundle = load_config_bundle()

    for model in (
        bundle.dataset,
        bundle.labels,
        bundle.operators,
        bundle.llm,
        bundle.rendering,
    ):
        json.dumps(model.model_dump(mode="json"))
        json.loads(model.model_dump_json())


def test_dataset_rejects_gt_visibility_and_unknown_keys() -> None:
    data = _load_yaml("dataset.yaml")
    detector = data["detector"]
    assert isinstance(detector, dict)
    detector["include_gt_smiles"] = True
    data["unexpected"] = "silent configuration drift"

    with pytest.raises(ValidationError) as error:
        DatasetConfig.model_validate(data)

    message = str(error.value)
    assert "include_gt_smiles" in message
    assert "unexpected" in message


def test_dataset_schema_is_strict_and_rejects_coercion() -> None:
    data = _load_yaml("dataset.yaml")
    dataset = data["dataset"]
    assert isinstance(dataset, dict)
    dataset["origins_total"] = "150"

    with pytest.raises(ValidationError, match="origins_total"):
        DatasetConfig.model_validate(data)


def test_llm_config_rejects_unapproved_model_fallback() -> None:
    data = _load_yaml("llm.yaml")
    provider = data["provider"]
    assert isinstance(provider, dict)
    provider["model_id"] = "another-model"

    with pytest.raises(ValidationError, match="gpt-5.4-mini"):
        LLMConfig.model_validate(data)


def test_operator_quota_must_sum_to_fifty() -> None:
    data = _load_yaml("operators.yaml")
    quotas = data["quotas_per_subtask_policy"]
    assert isinstance(quotas, dict)
    local = quotas["LOCAL"]
    assert isinstance(local, list)
    assert isinstance(local[0], dict)
    local[0]["target_per_50"] = 14

    with pytest.raises(ValidationError, match="quotas differ from the frozen plan"):
        OperatorsConfig.model_validate(data)


def test_operator_compatibility_matrix_is_exact() -> None:
    original = _load_yaml("operators.yaml")

    rogue = deepcopy(original)
    rogue_families = rogue["families"]
    assert isinstance(rogue_families, dict)
    rogue_families["rogue"] = {
        "supported_policies": ["LOCAL"],
        "allowed_candidate_sources": ["RULE"],
    }

    empty_sources = deepcopy(original)
    empty_families = empty_sources["families"]
    assert isinstance(empty_families, dict)
    assert isinstance(empty_families["wrong_anchor_site"], dict)
    empty_families["wrong_anchor_site"]["allowed_candidate_sources"] = []

    expanded_policy = deepcopy(original)
    expanded_families = expanded_policy["families"]
    assert isinstance(expanded_families, dict)
    assert isinstance(expanded_families["numeric_count_claim"], dict)
    expanded_families["numeric_count_claim"]["supported_policies"].append("FULL_CF")

    for invalid in (rogue, empty_sources, expanded_policy):
        with pytest.raises(ValidationError, match="compatibility matrix"):
            OperatorsConfig.model_validate(invalid)


def test_quota_buckets_have_frozen_compatible_family_mappings() -> None:
    original = _load_yaml("operators.yaml")

    missing_mapping = deepcopy(original)
    mappings = missing_mapping["quota_bucket_mappings"]
    assert isinstance(mappings, dict)
    del mappings["anchor_site_grounding"]

    policy_mismatch = deepcopy(original)
    mismatch_mappings = policy_mismatch["quota_bucket_mappings"]
    assert isinstance(mismatch_mappings, dict)
    mismatch_mappings["anchor_site_grounding"] = ["final_answer_identity"]

    for invalid in (missing_mapping, policy_mismatch):
        with pytest.raises(ValidationError, match="quota bucket mappings"):
            OperatorsConfig.model_validate(invalid)


def test_poe_capability_requirements_cannot_be_removed() -> None:
    original = _load_yaml("llm.yaml")
    for key in ("require_endpoints", "require_features"):
        invalid = deepcopy(original)
        discovery = invalid["model_discovery"]
        assert isinstance(discovery, dict)
        discovery[key] = []
        with pytest.raises(ValidationError):
            LLMConfig.model_validate(invalid)


def test_loader_requires_all_five_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        load_config_bundle(tmp_path)


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: '1.0'\nschema_version: '2.0'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        _read_yaml_mapping(path)


def test_yaml_loader_rejects_embedded_secrets(tmp_path: Path) -> None:
    path = tmp_path / "secret.yaml"
    path.write_text("api_key: sk-example-secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Embedded secret"):
        _read_yaml_mapping(path)


def test_committed_configs_contain_no_secret_value() -> None:
    for path in default_config_directory().glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "Authorization:" not in text
        assert "POE_API_KEY=" not in text
