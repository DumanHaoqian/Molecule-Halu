"""Generate the complete unified molecule-editing hallucination dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    HallucinationGenerationConfig,
)
from molhallulens.config.paths import DEFAULT_DATASET_ROOT, DEFAULT_GENERATED_ROOT
from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import FragmentPool, UnifiedHallucinationPlanner
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag
from molhallulens.modules.release import UnifiedRecordBuilder
from molhallulens.modules.text_realization import (
    MatchedNegativeTextBuilder,
    PoeStepTextAgent,
    PoeTextRealizationError,
    PoeTextRenderer,
)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    output_path: Path
    failure_manifest_path: Path
    origin_count: int
    successful_origin_count: int
    failed_origin_count: int
    attempted_variant_count: int
    successful_variant_count: int
    failed_variant_count: int
    record_count: int
    variants_per_origin: int
    fragment_pool_size: int
    subtask_counts: dict[str, int]
    edit_count_distribution: dict[int, int]
    variant_label_counts: dict[str, int]
    pair_alignment_distribution: dict[str, int]
    same_char_length_count: int
    paired_span_count: int
    same_char_length_ratio: float
    rewrite_mode_distribution: dict[str, int]
    poe_rewrite_call_count: int
    poe_uncached_request_count: int
    poe_cache_hit_count: int
    poe_network_request_count: int
    poe_retry_count: int
    poe_step_retry_count: int
    local_copy_step_count: int
    poe_requests_with_retry: int
    poe_retry_rate: float
    poe_validation_rejection_counts: dict[str, int]
    failures: tuple[dict[str, Any], ...]


def _default_failure_manifest_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".jsonl"
    stem = (
        output_path.name[: -len(output_path.suffix)]
        if output_path.suffix
        else output_path.name
    )
    return output_path.with_name(f"{stem}.failures{suffix}")


def _write_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _telemetry_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    scalar_keys = (
        "rewrite_call_count",
        "uncached_request_count",
        "cache_hit_count",
        "network_request_count",
        "retry_count",
        "requests_with_retry",
        "step_retry_count",
        "local_copy_step_count",
    )
    delta = {key: after[key] - before[key] for key in scalar_keys}
    rejection_keys = set(before["validation_rejection_counts"]) | set(
        after["validation_rejection_counts"]
    )
    delta["validation_rejection_counts"] = {
        key: (
            after["validation_rejection_counts"].get(key, 0)
            - before["validation_rejection_counts"].get(key, 0)
        )
        for key in sorted(rejection_keys)
        if (
            after["validation_rejection_counts"].get(key, 0)
            - before["validation_rejection_counts"].get(key, 0)
        )
    }
    return delta


def generate_dataset(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path = DEFAULT_GENERATED_ROOT / "example.jsonl",
    failure_manifest_path: Path | None = None,
    variants_per_origin: int = 1,
    max_origins: int | None = None,
    retry_failures_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
    poe_agent: PoeStepTextAgent | None = None,
) -> GenerationSummary:
    """Run the explicit modules for every origin and write one JSONL file."""

    if type(variants_per_origin) is not int or variants_per_origin < 1:
        raise ValueError("variants_per_origin must be a positive integer")
    if max_origins is not None and (
        type(max_origins) is not int or max_origins < 1
    ):
        raise ValueError("max_origins must be a positive integer or None")
    if type(config) is not HallucinationGenerationConfig:
        raise TypeError("config must be HallucinationGenerationConfig")
    if poe_agent is not None and type(poe_agent) is not PoeStepTextAgent:
        raise TypeError("poe_agent must be PoeStepTextAgent or None")

    # A: read and validate all raw/process/template triples.
    origins = ChemCoTMolEditAdapter().load(Path(dataset_root))

    # B: construct all reference DAGs once and build a corpus-level fragment pool.
    all_references = tuple(build_reference_dag(origin) for origin in origins)
    fragment_pool = FragmentPool.from_reference_artifacts(all_references)
    references = (
        all_references if max_origins is None else all_references[:max_origins]
    )
    selected_variants: set[tuple[str, int]] | None = None
    if retry_failures_path is not None:
        if max_origins is not None:
            raise ValueError("retry_failures_path and max_origins are mutually exclusive")
        selected_variants = set()
        known_ids = {reference.anonymous_sample_id for reference in all_references}
        for line in Path(retry_failures_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            origin_id, variant = row["origin_id"], row["variant_index"]
            if origin_id not in known_ids or type(variant) is not int or not 0 <= variant < variants_per_origin:
                raise ValueError("retry manifest contains an unknown origin or out-of-range variant")
            key = (origin_id, variant)
            if key in selected_variants:
                raise ValueError("retry manifest contains duplicate origin/variant pairs")
            selected_variants.add(key)
        if not selected_variants:
            raise ValueError("retry manifest is empty")
        selected_ids = {origin_id for origin_id, _ in selected_variants}
        references = tuple(r for r in all_references if r.anonymous_sample_id in selected_ids)

    output_path = Path(output_path)
    manifest_path = (
        _default_failure_manifest_path(output_path)
        if failure_manifest_path is None
        else Path(failure_manifest_path)
    )
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("output_path and failure_manifest_path must be different")
    if retry_failures_path is not None and Path(retry_failures_path).resolve() in {
        output_path.resolve(), manifest_path.resolve()
    }:
        raise ValueError("retry input manifest must not be overwritten")

    # C-G: each variant independently selects, applies, renders and annotates edits.
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injector = UnifiedHallucinationInjector(config)
    agent = PoeStepTextAgent(config) if poe_agent is None else poe_agent
    renderer = PoeTextRenderer(agent)
    pair_builder = MatchedNegativeTextBuilder(agent)
    annotator = UnifiedHallucinationAnnotator()
    record_builder = UnifiedRecordBuilder()
    subtask_counts = Counter(reference.normalized_subtask.value for reference in references)
    edit_counts: Counter[int] = Counter()
    variant_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    rewrite_mode_counts: Counter[str] = Counter()
    record_count = 0
    same_length_count = 0
    paired_span_count = 0
    successful_variant_count = 0
    failed_variant_count = 0
    successful_origin_count = 0
    failed_origin_count = 0
    failures: list[dict[str, Any]] = []
    telemetry_before = agent.telemetry()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("w", encoding="utf-8") as output_handle,
        manifest_path.open("w", encoding="utf-8") as failure_handle,
    ):
        for reference in references:
            origin_failed = False
            for variant_index in range(variants_per_origin):
                if selected_variants is not None and (reference.anonymous_sample_id, variant_index) not in selected_variants:
                    continue
                if progress_callback is not None:
                    progress_callback({"event": "start", "origin_id": reference.anonymous_sample_id, "variant_index": variant_index})
                stage = "C_ERROR_PLANNING"
                try:
                    plan = planner.plan(reference, variant_index=variant_index)
                    stage = "D_ERROR_INJECTION"
                    injected = injector.apply(reference.state_dag, plan)
                    stage = "E_TEXT_REALIZATION"
                    rendered = renderer.render(reference, injected)
                    stage = "F_ANNOTATION"
                    positive = annotator.annotate(rendered, injected)
                    if config.emit_matched_negative:
                        stage = "E_MATCHED_NEGATIVE"
                        rendered_pair = pair_builder.build(
                            reference,
                            injected,
                            rendered,
                        )
                        stage = "F_ANNOTATION"
                        negative = annotator.annotate_negative(
                            rendered_pair.negative,
                            positive,
                        )
                        stage = "G_RELEASE"
                        released_records = record_builder.build_pair(
                            reference,
                            injected,
                            rendered_pair,
                            positive,
                            negative,
                        )
                    else:
                        stage = "G_RELEASE"
                        released_records = (
                            record_builder.build(reference, injected, positive),
                        )
                except Exception as error:
                    origin_failed = True
                    failed_variant_count += 1
                    failure = {
                        "origin_id": reference.anonymous_sample_id,
                        "subtask": reference.normalized_subtask.value,
                        "variant_index": variant_index,
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "error_message": str(error) or type(error).__name__,
                        "diagnostics": list(getattr(error, "diagnostics", ())),
                        "diagnostic_path": getattr(error, "diagnostic_path", None),
                    }
                    failures.append(failure)
                    _write_json_line(failure_handle, failure)
                    failure_handle.flush()
                    if progress_callback is not None:
                        progress_callback({"event": "failure", "origin_id": reference.anonymous_sample_id,
                                           "variant_index": variant_index, "stage": stage,
                                           "successful": successful_variant_count, "failed": failed_variant_count})
                    continue

                # H: write a complete H/N pair together, then flush so an
                # interrupted later origin cannot erase completed work.
                for released in released_records:
                    _write_json_line(output_handle, released.to_dict())
                    record_count += 1
                    edit_counts[released.data["edit_count"]] += 1
                    variant_counts[released.data["variant_label"]] += 1
                output_handle.flush()
                successful_variant_count += 1
                if progress_callback is not None:
                    progress_callback({"event": "success", "origin_id": reference.anonymous_sample_id,
                                       "variant_index": variant_index,
                                       "successful": successful_variant_count, "failed": failed_variant_count})
                h_record = released_records[0]
                alignment_counts.update(
                    item["pair_alignment"]
                    for item in h_record.data["pair_alignment"]
                )
                rewrite_mode_counts.update(
                    h_record.data["text_realization"].get(
                        "step_rewrite_modes",
                        (),
                    )
                )
                positive_spans = h_record.data["hallucination_spans"]
                paired_span_count += len(positive_spans)
                same_length_count += sum(
                    item["same_char_length"] for item in positive_spans
                )
            if origin_failed:
                failed_origin_count += 1
            else:
                successful_origin_count += 1

    telemetry = _telemetry_delta(telemetry_before, agent.telemetry())
    return GenerationSummary(
        output_path=output_path.resolve(),
        failure_manifest_path=manifest_path.resolve(),
        origin_count=len(references),
        successful_origin_count=successful_origin_count,
        failed_origin_count=failed_origin_count,
        attempted_variant_count=successful_variant_count + failed_variant_count,
        successful_variant_count=successful_variant_count,
        failed_variant_count=failed_variant_count,
        record_count=record_count,
        variants_per_origin=variants_per_origin,
        fragment_pool_size=len(fragment_pool),
        subtask_counts=dict(sorted(subtask_counts.items())),
        edit_count_distribution=dict(sorted(edit_counts.items())),
        variant_label_counts=dict(sorted(variant_counts.items())),
        pair_alignment_distribution=dict(sorted(alignment_counts.items())),
        same_char_length_count=same_length_count,
        paired_span_count=paired_span_count,
        same_char_length_ratio=(
            same_length_count / paired_span_count if paired_span_count else 0.0
        ),
        rewrite_mode_distribution=dict(sorted(rewrite_mode_counts.items())),
        poe_rewrite_call_count=telemetry["rewrite_call_count"],
        poe_uncached_request_count=telemetry["uncached_request_count"],
        poe_cache_hit_count=telemetry["cache_hit_count"],
        poe_network_request_count=telemetry["network_request_count"],
        poe_retry_count=telemetry["retry_count"],
        poe_step_retry_count=telemetry["step_retry_count"],
        local_copy_step_count=telemetry["local_copy_step_count"],
        poe_requests_with_retry=telemetry["requests_with_retry"],
        poe_retry_rate=(
            telemetry["requests_with_retry"] / telemetry["uncached_request_count"]
            if telemetry["uncached_request_count"]
            else 0.0
        ),
        poe_validation_rejection_counts=telemetry[
            "validation_rejection_counts"
        ],
        failures=tuple(failures),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GENERATED_ROOT / "example.jsonl",
    )
    parser.add_argument("--variants-per-origin", type=int, default=1)
    parser.add_argument("--retry-failures", type=Path, default=None,
                        help="Retry only exact origin/variant pairs from a failure manifest, retaining the full fragment pool.")
    parser.add_argument(
        "--max-edits",
        action="store_true",
        help="Use each origin's maximum non-conflicting root edit count (no random K).",
    )
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=None,
        help="Failure JSONL path (default: <output-stem>.failures.jsonl).",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Generate only the first N origins while keeping the full fragment pool.",
    )
    args = parser.parse_args()
    try:
        summary = generate_dataset(
            dataset_root=args.dataset_root,
            output_path=args.output,
            failure_manifest_path=args.failure_manifest,
            variants_per_origin=args.variants_per_origin,
            max_origins=args.max_origins,
            retry_failures_path=args.retry_failures,
            config=(
                replace(DEFAULT_HALLUCINATION_CONFIG, edit_count_mode="maximum")
                if args.max_edits else DEFAULT_HALLUCINATION_CONFIG
            ),
        )
    except PoeTextRealizationError as error:
        raise SystemExit(str(error)) from None
    print(f"wrote {summary.record_count} records to {summary.output_path}")
    print(
        f"origins: {summary.successful_origin_count} succeeded, "
        f"{summary.failed_origin_count} failed, {summary.origin_count} attempted; "
        f"variants/origin: {summary.variants_per_origin}"
    )
    print(
        f"variants: {summary.successful_variant_count} succeeded, "
        f"{summary.failed_variant_count} failed"
    )
    print(f"fragment pool: {summary.fragment_pool_size}")
    print(f"subtasks: {summary.subtask_counts}")
    print(f"edit-count distribution: {summary.edit_count_distribution}")
    print(f"variant labels: {summary.variant_label_counts}")
    print(f"pair alignments: {summary.pair_alignment_distribution}")
    print(f"rewrite modes: {summary.rewrite_mode_distribution}")
    print(
        "Poe: "
        f"{summary.poe_network_request_count} network attempts, "
        f"{summary.poe_retry_count} retries, retry rate {summary.poe_retry_rate:.2%}, "
        f"{summary.poe_step_retry_count} retried steps, {summary.local_copy_step_count} local COPY steps, "
        f"rejections {summary.poe_validation_rejection_counts}"
    )
    print(
        "same-character-length controls: "
        f"{summary.same_char_length_count}/{summary.paired_span_count} "
        f"({summary.same_char_length_ratio:.2%})"
    )
    print(f"failure manifest: {summary.failure_manifest_path}")
    if summary.failures:
        print(
            "failed items: "
            + ", ".join(
                f"{item['origin_id']}[v{item['variant_index']}]@{item['stage']}"
                for item in summary.failures
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["GenerationSummary", "generate_dataset", "main"]
