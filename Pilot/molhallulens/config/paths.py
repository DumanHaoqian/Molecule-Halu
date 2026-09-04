"""Repository-local default paths.

All defaults are derived from the installed source tree.  Callers may still
override them explicitly; no module should embed a developer-specific path.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset"
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "GeneratedDataset"


__all__ = ["DEFAULT_DATASET_ROOT", "DEFAULT_GENERATED_ROOT", "PROJECT_ROOT"]
