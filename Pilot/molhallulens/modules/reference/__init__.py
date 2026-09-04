"""Reference DAG construction and executable molecule-editing truth.

Exports are loaded lazily so individual builders remain executable with
``python -m`` and the package has no import-time dependency cycle.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    **{
        name: ".builder"
        for name in (
            "AdditionReferenceDAGBuilder",
            "DeletionReferenceDAGBuilder",
            "EditingReferenceDAGBuilder",
            "ReferenceDAGArtifact",
            "ReferenceDAGBuildError",
            "ReferenceDAGBuildReport",
            "ReferenceDAGCorpusError",
            "ReferenceDAGCorpusResult",
            "ReferenceDAGOriginReport",
            "ReferenceMention",
            "ReferenceSlotBinding",
            "ReferenceTraceStep",
            "SubstitutionReferenceDAGBuilder",
            "audit_reference_dag_corpus",
            "build_reference_dag",
            "build_reference_dag_corpus",
            "reference_dag_builder_for",
        )
    },
}


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORTS)
