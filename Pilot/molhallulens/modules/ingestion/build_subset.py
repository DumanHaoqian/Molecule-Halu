#!/usr/bin/env python3
"""Build a reproducible pilot subset across ChemCoTBench-V2 tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from molhallulens.config.paths import DEFAULT_DATASET_ROOT

DEFAULT_SEED = 42
DEFAULT_VERSION_NAME = "pilot_origin"

# Global suffix for every generated subtask and file name.
# Examples: add_v2 -> add_pilot_origin; forward -> forward_pilot_origin.
# Each item is: (source subtask name, number of samples to draw).
# Family, reporting_task, and source paths are resolved from manifest.json.
DEFAULT_SUBTASKS = (
    ("add_v2", 50),
    ("delete_v2", 50),
    ("substitute_v2", 50),
)

MANIFEST_FIELDS = (
    "family",
    "subtask",
    "reporting_task",
    "n_samples",
    "raw_file",
    "process_file",
)


def load_json(path: Path) -> list[dict] | dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: list[dict] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_config(
    *, version_name: str, subtasks: Sequence[tuple[str, int]]
) -> None:
    if not version_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version_name):
        raise ValueError(
            "version_name must be a non-empty filename-safe suffix containing only "
            "letters, digits, underscores, or hyphens"
        )

    names = [name for name, _ in subtasks]
    if len(names) != len(set(names)):
        raise ValueError("SUBTASKS contains duplicate subtask names")
    for source_subtask, n_samples in subtasks:
        if not source_subtask or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"Invalid SUBTASKS entry: {(source_subtask, n_samples)!r}")


def build_manifest_lookup(
    source_root: Path,
    subtasks: Sequence[tuple[str, int]],
) -> dict[str, dict]:
    manifest = load_json(source_root / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Source manifest.json does not contain a valid 'files' list")

    lookup: dict[str, dict] = {}
    duplicates: set[str] = set()
    for entry in manifest["files"]:
        name = entry["subtask"]
        if name in lookup:
            duplicates.add(name)
        lookup[name] = entry

    requested_duplicates = duplicates.intersection(name for name, _ in subtasks)
    if requested_duplicates:
        raise ValueError(
            "These subtask names occur in multiple families and are ambiguous: "
            f"{sorted(requested_duplicates)}"
        )
    return lookup


def output_subtask_name(source_subtask: str, *, version_name: str) -> str:
    """Replace a terminal version such as _v2/_v3 with the global suffix."""
    base_name = re.sub(r"_v\d+$", "", source_subtask)
    return f"{base_name}_{version_name}"


def rng_for_subtask(source_subtask: str, *, seed: int) -> random.Random:
    """Use an order-independent deterministic random stream per subtask."""
    digest = hashlib.sha256(f"{seed}:{source_subtask}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_subset(
    *,
    source_root: Path,
    output_root: Path = DEFAULT_DATASET_ROOT,
    subtasks: Sequence[tuple[str, int]] = DEFAULT_SUBTASKS,
    version_name: str = DEFAULT_VERSION_NAME,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Build one deterministic subset and return its CSV manifest path."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    subtasks = tuple(subtasks)
    validate_config(version_name=version_name, subtasks=subtasks)
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    source_entries = build_manifest_lookup(source_root, subtasks)
    manifest_rows: list[dict] = []

    for source_subtask, n_samples in subtasks:
        if source_subtask not in source_entries:
            raise KeyError(f"Subtask {source_subtask!r} is not present in source manifest.json")

        entry = source_entries[source_subtask]
        family = entry["family"]
        reporting_task = entry["reporting_task"]
        pilot_subtask = output_subtask_name(
            source_subtask,
            version_name=version_name,
        )

        raw_source = source_root / entry["raw_file"]
        process_source = source_root / entry["process_file"]
        template_source = source_root / "formal_templates" / family / f"{source_subtask}.json"

        raw_records = load_json(raw_source)
        process_records = load_json(process_source)
        if not isinstance(raw_records, list) or not isinstance(process_records, list):
            raise TypeError(f"Raw/process data for {source_subtask} must both be JSON lists")
        if n_samples > len(raw_records):
            raise ValueError(
                f"Requested {n_samples} samples from {source_subtask}, "
                f"but only {len(raw_records)} are available"
            )

        process_by_id = {record["anonymous_sample_id"]: record for record in process_records}
        selected_raw = rng_for_subtask(source_subtask, seed=seed).sample(
            raw_records,
            n_samples,
        )
        missing_ids = [
            record["anonymous_sample_id"]
            for record in selected_raw
            if record["anonymous_sample_id"] not in process_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"{source_subtask} has {len(missing_ids)} sampled raw records without process matches"
            )
        selected_process = [process_by_id[record["anonymous_sample_id"]] for record in selected_raw]

        # Preserve anonymous_sample_id/original IDs for provenance and pairing.
        selected_raw = [{**record, "subtask": pilot_subtask} for record in selected_raw]
        selected_process = [{**record, "subtask": pilot_subtask} for record in selected_process]

        raw_rel = Path("raw_benchmark_data") / family / f"{pilot_subtask}.json"
        process_rel = Path("process_evaluation_data") / family / f"{pilot_subtask}.json"
        save_json(output_root / raw_rel, selected_raw)
        save_json(output_root / process_rel, selected_process)

        if not template_source.exists():
            raise FileNotFoundError(f"Missing formal template: {template_source}")
        template = load_json(template_source)
        if not isinstance(template, dict):
            raise TypeError(f"Formal template for {source_subtask} must be a JSON object")
        template["subtask"] = pilot_subtask
        template["n_samples"] = n_samples
        save_json(
            output_root / "formal_templates" / family / f"{pilot_subtask}.json",
            template,
        )

        manifest_rows.append(
            {
                "family": family,
                "subtask": pilot_subtask,
                "reporting_task": reporting_task,
                "n_samples": n_samples,
                "raw_file": raw_rel.as_posix(),
                "process_file": process_rel.as_posix(),
            }
        )

    manifest_path = output_root / "active_benchmark_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Built {sum(n for _, n in subtasks)} samples across {len(subtasks)} subtasks "
        f"at {output_root} (version_name={version_name!r})"
    )
    return manifest_path


def _subtask_argument(value: str) -> tuple[str, int]:
    try:
        name, raw_count = value.rsplit("=", 1)
        count = int(raw_count)
    except ValueError as error:
        raise ValueError("subtask must use NAME=COUNT, for example add_v2=50") from error
    if not name or count < 1:
        raise ValueError("subtask must use a non-empty name and positive count")
    return name, count


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="ChemCoTBench-V2 directory containing manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"output dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--version-name", default=DEFAULT_VERSION_NAME)
    parser.add_argument(
        "--subtask",
        action="append",
        type=_subtask_argument,
        help="repeatable NAME=COUNT selection; defaults to add/delete/substitute=50",
    )
    args = parser.parse_args(argv)
    build_subset(
        source_root=args.source_root,
        output_root=args.output_root,
        subtasks=tuple(args.subtask or DEFAULT_SUBTASKS),
        version_name=args.version_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
