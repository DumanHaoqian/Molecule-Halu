"""Dataset assembly, split publication, tokenization, and release QA.

Import concrete entry points from their focused modules (for example,
``molhallulens.modules.release.train``).  Keeping this package initializer
lightweight prevents release concerns from leaking into earlier pipeline stages.
"""

__all__ = [
    "assembly",
    "chemdfm",
    "dry_run",
    "dry_run_review",
    "generation",
    "leakage",
    "manifest",
    "origin_audit",
    "qa",
    "record_build",
    "shortcut_audit",
    "splitter",
    "test",
    "train",
    "validation",
]
