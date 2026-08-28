#!/usr/bin/env python3
"""Build the 150-sample Molecule Editing pilot benchmark."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


SOURCE_ROOT = Path("/home/haoqian/Data/Molecule/datasets/ChemCoTBench-V2")
OUTPUT_ROOT = Path("/home/haoqian/Data/Molecule/Pilot/Dataset")
SEED = 42
N_PER_SUBTASK = 50

SUBTASKS = (
    ("add_v2", "add_pilot_origin", "MolEdit/Add"),
    ("delete_v2", "delete_pilot_origin", "MolEdit/Delete"),
    ("substitute_v2", "substitute_pilot_origin", "MolEdit/Substitute"),
)


def load_json(path: Path) -> list[dict] | dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: list[dict] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    rng = random.Random(SEED)
    manifest_rows: list[dict] = []

    for source_subtask, pilot_subtask, reporting_task in SUBTASKS:
        raw_source = SOURCE_ROOT / "raw_benchmark_data" / "mol_edit" / f"{source_subtask}.json"
        process_source = SOURCE_ROOT / "process_evaluation_data" / "mol_edit" / f"{source_subtask}.json"
        template_source = SOURCE_ROOT / "formal_templates" / "mol_edit" / f"{source_subtask}.json"

        raw_records = load_json(raw_source)
        process_records = load_json(process_source)
        assert isinstance(raw_records, list) and isinstance(process_records, list)

        process_by_id = {record["anonymous_sample_id"]: record for record in process_records}
        selected_raw = rng.sample(raw_records, N_PER_SUBTASK)
        selected_process = [process_by_id[record["anonymous_sample_id"]] for record in selected_raw]

        # Keep stable source IDs for provenance, but expose the pilot subtask consistently.
        selected_raw = [{**record, "subtask": pilot_subtask} for record in selected_raw]
        selected_process = [{**record, "subtask": pilot_subtask} for record in selected_process]

        raw_rel = Path("raw_benchmark_data") / "mol_edit" / f"{pilot_subtask}.json"
        process_rel = Path("process_evaluation_data") / "mol_edit" / f"{pilot_subtask}.json"
        save_json(OUTPUT_ROOT / raw_rel, selected_raw)
        save_json(OUTPUT_ROOT / process_rel, selected_process)

        template = load_json(template_source)
        assert isinstance(template, dict)
        template["subtask"] = pilot_subtask
        template["n_samples"] = N_PER_SUBTASK
        save_json(
            OUTPUT_ROOT / "formal_templates" / "mol_edit" / f"{pilot_subtask}.json",
            template,
        )

        manifest_rows.append(
            {
                "family": "mol_edit",
                "subtask": pilot_subtask,
                "reporting_task": reporting_task,
                "n_samples": N_PER_SUBTASK,
                "raw_file": raw_rel.as_posix(),
                "process_file": process_rel.as_posix(),
            }
        )

    manifest_path = OUTPUT_ROOT / "active_benchmark_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("family", "subtask", "reporting_task", "n_samples", "raw_file", "process_file"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
