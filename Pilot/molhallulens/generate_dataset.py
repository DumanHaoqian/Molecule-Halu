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


def generate_dataset(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path = DEFAULT_GENERATED_ROOT / "example.jsonl",
    variants_per_origin: int = 1,
    config: HallucinationGenerationConfig = DEFAULT_HALLUCINATION_CONFIG,
) -> GenerationSummary:
    """Run the explicit modules for every origin and write one JSONL file."""

    if type(variants_per_origin) is not int or variants_per_origin < 1:
        raise ValueError("variants_per_origin must be a positive integer")
    if type(config) is not HallucinationGenerationConfig:
        raise TypeError("config must be HallucinationGenerationConfig")

    # A: read and validate all raw/process/template triples.
    origins = ChemCoTMolEditAdapter().load(Path(dataset_root))

    # B: construct all reference DAGs once and build a corpus-level fragment pool.
    references = tuple(build_reference_dag(origin) for origin in origins)
    fragment_pool = FragmentPool.from_reference_artifacts(references)

    # C-G: each variant independently selects, applies, renders and annotates edits.
    planner = UnifiedHallucinationPlanner(fragment_pool, config)
    injector = UnifiedHallucinationInjector(config)
    renderer = PoeTextRenderer(PoeStepTextAgent(config))
    annotator = UnifiedHallucinationAnnotator()
    record_builder = UnifiedRecordBuilder()
    records = []
    for reference in references:
        for variant_index in range(variants_per_origin):
            plan = planner.plan(reference, variant_index=variant_index)
            injected = injector.apply(reference.state_dag, plan)
            rendered = renderer.render(reference, injected)
            annotated = annotator.annotate(rendered, injected)
            records.append(record_builder.build(reference, injected, annotated))

    # H: release exactly the records made by the unified path.
    write_jsonl(records, Path(output_path))
    subtask_counts = Counter(reference.normalized_subtask.value for reference in references)
    edit_counts = Counter(record.data["edit_count"] for record in records)
    return GenerationSummary(
        output_path=Path(output_path).resolve(),
        origin_count=len(references),
        record_count=len(records),
        variants_per_origin=variants_per_origin,
        fragment_pool_size=len(fragment_pool),
        subtask_counts=dict(sorted(subtask_counts.items())),
        edit_count_distribution=dict(sorted(edit_counts.items())),
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
    args = parser.parse_args()
    try:
        summary = generate_dataset(
            dataset_root=args.dataset_root,
            output_path=args.output,
            variants_per_origin=args.variants_per_origin,
        )
    except PoeTextRealizationError as error:
        raise SystemExit(str(error)) from None
    print(f"wrote {summary.record_count} records to {summary.output_path}")
    print(f"origins: {summary.origin_count}; variants/origin: {summary.variants_per_origin}")
    print(f"fragment pool: {summary.fragment_pool_size}")
    print(f"subtasks: {summary.subtask_counts}")
    print(f"edit-count distribution: {summary.edit_count_distribution}")


if __name__ == "__main__":
    main()


__all__ = ["GenerationSummary", "generate_dataset", "main"]
