# MolHalluLens Pilot

MolHalluLens converts ChemCoTBench-V2 molecule-editing traces into a validated
reference graph, injects controlled errors, propagates their consequences,
renders detector-visible text, aligns labels, and builds release artifacts.

The code follows one visible pipeline:

```text
ingestion → reference → error_planning → error_injection
          → trajectory → text_realization → annotation → release
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the responsibility and input/output
contract of every module.

## Repository layout

```text
Pilot/
├── molhallulens/          # Python source
│   ├── core/              # shared immutable data contracts
│   ├── config/            # configuration and repository-local paths
│   ├── modules/           # the eight pipeline stages
│   ├── infrastructure/    # RDKit, validators, external providers
│   ├── cli/               # thin command-line entry points
│   └── pipeline.py        # stage orchestration only
├── Dataset/               # bounded ChemCoTBench pilot inputs and reports
├── HallucinationDataset/  # generated/released artifacts
└── tests/                 # unit, property, and integration tests
```

`Dataset/` and `HallucinationDataset/` intentionally remain top-level artifact
roots. They are data products, not importable source modules.

## Environment and tests

Use Python 3.11–3.13:

```bash
conda activate molhallulens
python -m pip install -r requirements-dev.lock
python -m pytest
```

All commands should be run from this `Pilot` directory. Package imports should
use `python -m ...` or the installed console commands; do not execute a package
file by its filesystem path.

## Main entry points

Build a deterministic pilot subset from a local ChemCoTBench-V2 checkout:

```bash
molhallulens-build-subset --source-root /path/to/ChemCoTBench-V2
```

Inspect the three reference-DAG examples directly:

```bash
python -m molhallulens.modules.reference.builder
```

Audit the frozen origins:

```bash
molhallulens-audit-origins
```

ChemDFM commands accept `--checkpoint-path`. You may also set the local default
without editing source code:

```bash
export MOLHALLULENS_CHEMDFM_CHECKPOINT=/path/to/ChemDFM-R-14B
```

The refactor preserves the existing release semantics. It does not yet remove
the legacy N controls or implement the proposed cumulative-error data model.
