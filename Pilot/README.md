# MolHalluLens Pilot

This package builds and validates the MolHalluLens Molecule Editing hallucination pilot described in `Blueprint/MOLHALLULENS_MOLEDIT_HALLUCINATION_IMPLEMENTATION_PLAN.md`.

Use Python 3.11–3.13. Runtime and development environments are reproducible from the compiled lock files:

```bash
python -m pip install -r requirements.lock
python -m pip install -r requirements-dev.lock
```

Regenerate locks only after intentionally updating `pyproject.toml` and the matching `.in` files:

```bash
python -m piptools compile --resolver=backtracking --output-file requirements.lock requirements.in
python -m piptools compile --resolver=backtracking --output-file requirements-dev.lock requirements-dev.in
```

Run tests from this directory with `python -m pytest` after installing the development lock.
