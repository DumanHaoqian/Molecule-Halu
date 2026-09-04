"""Generate the complete unified molecule-editing hallucination dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
from molhallulens.modules.release import UnifiedRecordBuilder, write_jsonl
from molhallulens.modules.text_realization import (
    MatchedNegativeTextBuilder,
    PoeStepTextAgent,
    PoeTextRealizationError,
    PoeTextRenderer,
)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    output_path: Path
    origin_count: int
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


def generate_dataset(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path = DEFAULT_GENERATED_ROOT / "example.jsonl",
    variants_per_origin: int = 1,
    max_origins: int | None = None,
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

    # C-G: each variant independently selects, applies, renders and annotates edits.
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injector = UnifiedHallucinationInjector(config)
    agent = PoeStepTextAgent(config) if poe_agent is None else poe_agent
    renderer = PoeTextRenderer(agent)
    pair_builder = MatchedNegativeTextBuilder(agent)
    annotator = UnifiedHallucinationAnnotator()
    record_builder = UnifiedRecordBuilder()
    records = []
    for reference in references:
        for variant_index in range(variants_per_origin):
            plan = planner.plan(reference, variant_index=variant_index)
            injected = injector.apply(reference.state_dag, plan)
            rendered = renderer.render(reference, injected)
            positive = annotator.annotate(rendered, injected)
            if config.emit_matched_negative:
                rendered_pair = pair_builder.build(reference, injected, rendered)
                negative = annotator.annotate_negative(
                    rendered_pair.negative,
                    positive,
                )
                records.extend(
                    record_builder.build_pair(
                        reference,
                        injected,
                        rendered_pair,
                        positive,
                        negative,
                    )
                )
            else:
                records.append(record_builder.build(reference, injected, positive))

    # H: release exactly the records made by the unified path.
    write_jsonl(records, Path(output_path))
    subtask_counts = Counter(reference.normalized_subtask.value for reference in references)
    edit_counts = Counter(record.data["edit_count"] for record in records)
    variant_counts = Counter(record.data["variant_label"] for record in records)
    alignment_counts = Counter(
        item["pair_alignment"]
        for record in records
        if record.data["variant_label"] == "H"
        for item in record.data["pair_alignment"]
    )
    positive_spans = [
        span
        for record in records
        if record.data["variant_label"] == "H"
        for span in record.data["hallucination_spans"]
    ]
    same_length_count = sum(item["same_char_length"] for item in positive_spans)
    return GenerationSummary(
        output_path=Path(output_path).resolve(),
        origin_count=len(references),
        record_count=len(records),
        variants_per_origin=variants_per_origin,
        fragment_pool_size=len(fragment_pool),
        subtask_counts=dict(sorted(subtask_counts.items())),
        edit_count_distribution=dict(sorted(edit_counts.items())),
        variant_label_counts=dict(sorted(variant_counts.items())),
        pair_alignment_distribution=dict(sorted(alignment_counts.items())),
        same_char_length_count=same_length_count,
        paired_span_count=len(positive_spans),
        same_char_length_ratio=(
            same_length_count / len(positive_spans) if positive_spans else 0.0
        ),
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
            variants_per_origin=args.variants_per_origin,
            max_origins=args.max_origins,
        )
    except PoeTextRealizationError as error:
        raise SystemExit(str(error)) from None
    print(f"wrote {summary.record_count} records to {summary.output_path}")
    print(f"origins: {summary.origin_count}; variants/origin: {summary.variants_per_origin}")
    print(f"fragment pool: {summary.fragment_pool_size}")
    print(f"subtasks: {summary.subtask_counts}")
    print(f"edit-count distribution: {summary.edit_count_distribution}")
    print(f"variant labels: {summary.variant_label_counts}")
    print(f"pair alignments: {summary.pair_alignment_distribution}")
    print(
        "same-character-length controls: "
        f"{summary.same_char_length_count}/{summary.paired_span_count} "
        f"({summary.same_char_length_ratio:.2%})"
    )


if __name__ == "__main__":
    main()


__all__ = ["GenerationSummary", "generate_dataset", "main"]
