"""Package-level smoke tests."""

from __future__ import annotations

import molhallulens


def test_package_version_is_exposed() -> None:
    assert molhallulens.__version__ == "0.1.0"
