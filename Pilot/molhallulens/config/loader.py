"""Load and cross-validate the frozen MolHalluLens YAML configuration."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from .models import (
    DatasetConfig,
    LabelsConfig,
    LLMConfig,
    OperatorsConfig,
    RenderingConfig,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
_FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "authorization",
    "openai_api_key",
    "password",
    "poe_api_key",
    "secret",
    "token",
}
_SECRET_VALUE_PATTERN = re.compile(r"(?i)^(?:bearer\s+|sk-[a-z0-9_-]{8,})")


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class ConfigPaths:
    dataset: Path
    labels: Path
    operators: Path
    llm: Path
    rendering: Path

    @classmethod
    def from_directory(cls, directory: Path) -> ConfigPaths:
        directory = directory.resolve()
        return cls(
            dataset=directory / "dataset.yaml",
            labels=directory / "labels.yaml",
            operators=directory / "operators.yaml",
            llm=directory / "llm.yaml",
            rendering=directory / "rendering.yaml",
        )


@dataclass(frozen=True)
class ConfigBundle:
    dataset: DatasetConfig
    labels: LabelsConfig
    operators: OperatorsConfig
    llm: LLMConfig
    rendering: RenderingConfig

    def validate_cross_file_invariants(self) -> None:
        dataset_policies = tuple(self.dataset.bundle.policies)
        if tuple(self.operators.policies) != dataset_policies:
            raise ValueError("operators.yaml policies differ from dataset.yaml")
        if self.rendering.detector_template.field_order != self.dataset.detector.field_order:
            raise ValueError("rendering.yaml field order differs from dataset.yaml")
        if (
            self.labels.canonical_annotation.activation_alignment
            != self.dataset.detector.activation_alignment
        ):
            raise ValueError("labels.yaml alignment differs from dataset.yaml")
        if self.rendering.detector_template.include_gt_smiles:
            raise ValueError("renderer must never include gt_smiles")
        if self.llm.renderer.label_blind is not self.rendering.natural_renderer.label_blind:
            raise ValueError("LLM and rendering label-blind settings disagree")

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset.dataset.version_name,
            "origins_total": self.dataset.dataset.origins_total,
            "records_total": self.dataset.dataset.records_total,
            "split_origins": {
                "train": self.dataset.split.train_origins,
                "validation": self.dataset.split.validation_origins,
                "test": self.dataset.split.test_origins,
            },
            "policies": list(self.dataset.bundle.policies),
            "activation_alignment": self.dataset.detector.activation_alignment,
            "include_gt_smiles": self.dataset.detector.include_gt_smiles,
            "provider": self.llm.provider.name,
            "model_id": self.llm.provider.model_id,
        }


def default_config_directory() -> Path:
    return Path(__file__).resolve().parent


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required configuration file: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.load(handle, Loader=UniqueKeySafeLoader)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration file must contain a YAML mapping: {path}")
    _reject_embedded_secrets(value, path=path)
    return value


def _reject_embedded_secrets(value: Any, *, path: Path, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_SECRET_KEYS:
                raise ValueError(f"Embedded secret key {key!r} is forbidden in {path}:{location}")
            _reject_embedded_secrets(nested, path=path, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_embedded_secrets(nested, path=path, location=f"{location}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE_PATTERN.match(value):
        raise ValueError(f"Embedded secret value is forbidden in {path}:{location}")


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(_read_yaml_mapping(path))


def load_config_bundle(directory: Path | None = None) -> ConfigBundle:
    paths = ConfigPaths.from_directory(directory or default_config_directory())
    bundle = ConfigBundle(
        dataset=_load_model(paths.dataset, DatasetConfig),
        labels=_load_model(paths.labels, LabelsConfig),
        operators=_load_model(paths.operators, OperatorsConfig),
        llm=_load_model(paths.llm, LLMConfig),
        rendering=_load_model(paths.rendering, RenderingConfig),
    )
    bundle.validate_cross_file_invariants()
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(load_config_bundle(args.config_dir).summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
