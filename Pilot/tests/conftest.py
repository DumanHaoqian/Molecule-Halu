from __future__ import annotations

from pathlib import Path

import pytest

from molhallulens.modules.error_planning import FragmentPool
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def references(project_root: Path):
    origins = ChemCoTMolEditAdapter().load(project_root / "Dataset")
    artifacts = tuple(build_reference_dag(origin) for origin in origins)
    return {artifact.normalized_subtask.value: artifact for artifact in artifacts}


@pytest.fixture(scope="session")
def all_references(project_root: Path):
    origins = ChemCoTMolEditAdapter().load(project_root / "Dataset")
    return tuple(build_reference_dag(origin) for origin in origins)


@pytest.fixture(scope="session")
def fragment_pool(all_references):
    return FragmentPool.from_reference_artifacts(all_references)
