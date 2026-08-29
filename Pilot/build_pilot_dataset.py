#!/usr/bin/env python3
"""Build a reproducible pilot subset across ChemCoTBench-V2 tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path


SOURCE_ROOT = Path("/home/haoqian/Data/Molecule/datasets/ChemCoTBench-V2")
OUTPUT_ROOT = Path("/home/haoqian/Data/Molecule/Pilot/Dataset")
SEED = 42

# Global suffix for every generated subtask and file name.
# Examples: add_v2 -> add_pilot_origin; forward -> forward_pilot_origin.
version_name = "pilot_origin"

# Each item is: (source subtask name, number of samples to draw).
# Family, reporting_task, and source paths are resolved from manifest.json.
SUBTASKS = [
    ("add_v2", 50),
    ("delete_v2", 50),
    ("substitute_v2", 50),
    # ("fg_detect", 20),
    # ("forward", 20),
    # ("logp", 20),
]

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


def validate_config() -> None:
    if not version_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version_name):
        raise ValueError(
            "version_name must be a non-empty filename-safe suffix containing only "
            "letters, digits, underscores, or hyphens"
        )

    names = [name for name, _ in SUBTASKS]
    if len(names) != len(set(names)):
        raise ValueError("SUBTASKS contains duplicate subtask names")
    for source_subtask, n_samples in SUBTASKS:
        if not source_subtask or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"Invalid SUBTASKS entry: {(source_subtask, n_samples)!r}")


def build_manifest_lookup() -> dict[str, dict]:
    manifest = load_json(SOURCE_ROOT / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Source manifest.json does not contain a valid 'files' list")

    lookup: dict[str, dict] = {}
    duplicates: set[str] = set()
    for entry in manifest["files"]:
        name = entry["subtask"]
        if name in lookup:
            duplicates.add(name)
        lookup[name] = entry

    requested_duplicates = duplicates.intersection(name for name, _ in SUBTASKS)
    if requested_duplicates:
        raise ValueError(
            "These subtask names occur in multiple families and are ambiguous: "
            f"{sorted(requested_duplicates)}"
        )
    return lookup


def output_subtask_name(source_subtask: str) -> str:
    """Replace a terminal version such as _v2/_v3 with the global suffix."""
    base_name = re.sub(r"_v\d+$", "", source_subtask)
    return f"{base_name}_{version_name}"


def rng_for_subtask(source_subtask: str) -> random.Random:
    """Use an order-independent deterministic random stream per subtask."""
    digest = hashlib.sha256(f"{SEED}:{source_subtask}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def main() -> None:
    validate_config()
    source_entries = build_manifest_lookup()
    manifest_rows: list[dict] = []

    for source_subtask, n_samples in SUBTASKS:
        if source_subtask not in source_entries:
            raise KeyError(f"Subtask {source_subtask!r} is not present in source manifest.json")

        entry = source_entries[source_subtask]
        family = entry["family"]
        reporting_task = entry["reporting_task"]
        pilot_subtask = output_subtask_name(source_subtask)

        raw_source = SOURCE_ROOT / entry["raw_file"]
        process_source = SOURCE_ROOT / entry["process_file"]
        template_source = SOURCE_ROOT / "formal_templates" / family / f"{source_subtask}.json"

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
        selected_raw = rng_for_subtask(source_subtask).sample(raw_records, n_samples)
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
        save_json(OUTPUT_ROOT / raw_rel, selected_raw)
        save_json(OUTPUT_ROOT / process_rel, selected_process)

        if not template_source.exists():
            raise FileNotFoundError(f"Missing formal template: {template_source}")
        template = load_json(template_source)
        if not isinstance(template, dict):
            raise TypeError(f"Formal template for {source_subtask} must be a JSON object")
        template["subtask"] = pilot_subtask
        template["n_samples"] = n_samples
        save_json(
            OUTPUT_ROOT / "formal_templates" / family / f"{pilot_subtask}.json",
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

    manifest_path = OUTPUT_ROOT / "active_benchmark_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Built {sum(n for _, n in SUBTASKS)} samples across {len(SUBTASKS)} subtasks "
        f"at {OUTPUT_ROOT} (version_name={version_name!r})"
    )


if __name__ == "__main__":
    main()
