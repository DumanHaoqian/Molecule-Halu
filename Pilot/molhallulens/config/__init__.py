"""Versioned project configuration resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .loader import ConfigBundle

__all__ = ["ConfigBundle", "load_config_bundle"]


def __getattr__(name: str) -> Any:
    if name == "ConfigBundle":
        from .loader import ConfigBundle

        return ConfigBundle
    if name == "load_config_bundle":
        from .loader import load_config_bundle

        return load_config_bundle
    raise AttributeError(name)
