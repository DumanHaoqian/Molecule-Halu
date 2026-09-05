"""Retry an existing failure manifest without overwriting previous artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molhallulens.config.hallucination_generation import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.generate_dataset import generate_dataset
from molhallulens.modules.text_realization.poe_agent import POE_RENDERER_VERSION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, required=True, help="Existing failure manifest to retry; historical manifests may have been cleaned up.")
    args = parser.parse_args()
    if not os.environ.get("POE_API_KEY", "").strip():
        raise SystemExit("POE_API_KEY is not loaded; run the poe alias in this shell first.")
    source = args.failures.resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    total = len(rows)
    if not rows or any(row["variant_index"] != 0 for row in rows):
        raise SystemExit("This runner requires a nonempty variant-0 failure manifest.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = source.with_name(f"retry_{POE_RENDERER_VERSION.rsplit('_', 1)[-1]}_{stamp}.jsonl")
    log_path = output.with_suffix(".progress.jsonl")
    summary_path = output.with_suffix(".summary.json")
    for path in (output, log_path, summary_path, output.with_suffix(".failures.jsonl")):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path.name}")
    started = time.monotonic()
    config = replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        def progress(event):
            elapsed = time.monotonic() - started
            event = dict(event, timestamp=datetime.now().astimezone().isoformat(),
                         elapsed_seconds=round(elapsed, 2), total=total)
            processed = event.get("successful", 0) + event.get("failed", 0)
            if processed:
                event["remaining_seconds_estimate"] = round(elapsed / processed * (total - processed))
            line = json.dumps(event, ensure_ascii=False, default=str)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        progress({"event": "run_start", "model": config.poe_bot_name,
                  "protocol": POE_RENDERER_VERSION, "edit_mode": "maximum",
                  "output": str(output), "input": str(source), "input_sha256": source_hash})
        summary = generate_dataset(output_path=output, retry_failures_path=source,
                                   config=config, progress_callback=progress)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        payload = dict(asdict(summary), elapsed_seconds=round(time.monotonic() - started, 2),
                       input_manifest=str(source), input_sha256=source_hash,
                       model=config.poe_bot_name, protocol=POE_RENDERER_VERSION)
        with summary_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        progress({"event": "run_complete", "successful": summary.successful_variant_count,
                  "failed": summary.failed_variant_count, "summary_path": str(summary_path),
                  "network_requests": summary.poe_network_request_count})
    raise SystemExit(1 if summary.failed_variant_count else 0)
