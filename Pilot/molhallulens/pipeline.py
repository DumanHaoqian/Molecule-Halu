"""Explicit A-to-H pipeline for the single multi-point hallucination design."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def _plain_data(value: Any) -> Any:
    """Convert immutable project objects into complete JSON-printable data."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_data(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return value


def _print_stage(stage: str, input_value: Any, output_value: Any) -> None:
    print("\n" + "=" * 100)
    print(stage)
    print("-" * 100)
    print("FULL INPUT")
    print(json.dumps(_plain_data(input_value), ensure_ascii=False, indent=2))
    print("\nFULL OUTPUT")
    print(json.dumps(_plain_data(output_value), ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one configurable multi-point hallucination record."
    )
    parser.add_argument(
        "--origin-id",
        default="mol_edit.add_v2.0003",
        help="ChemCoTBench-V2 anonymous origin ID.",
    )
    parser.add_argument(
        "--variant-index",
        type=int,
        default=0,
        help="Non-negative deterministic variant number.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; default: Pilot/GeneratedDataset/example.jsonl.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run without the interactive Enter pauses.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 支持两种运行方式：
    #   Pilot/ 下运行:       python -m molhallulens.pipeline
    #   molhallulens/ 下运行: python pipeline.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from molhallulens.config.hallucination_generation import (
        DEFAULT_HALLUCINATION_CONFIG,
    )
    from molhallulens.modules.annotation import UnifiedHallucinationAnnotator
    from molhallulens.modules.error_injection import UnifiedHallucinationInjector
    from molhallulens.modules.error_planning import (
        FragmentPool,
        UnifiedHallucinationPlanner,
    )
    from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
    from molhallulens.modules.reference import build_reference_dag
    from molhallulens.modules.release import UnifiedRecordBuilder, write_jsonl
    from molhallulens.modules.text_realization import (
        PoeStepTextAgent,
        PoeTextRealizationError,
        PoeTextRenderer,
        build_poe_rewrite_request,
    )

    args = _parse_args()
    dataset_root = project_root / "Dataset"
    output_path = args.output or project_root / "GeneratedDataset" / "example.jsonl"
    pause = not args.no_pause

    print("MolHalluLens unified multi-point hallucination pipeline")
    print("Every output is one configurable, always-positive hallucination sample")
    print(f"origin_id: {args.origin_id}")
    print(f"variant_index: {args.variant_index}")

    # A. 原始 Dataset -> 经过校验并按 ID join 的 150 条输入记录。
    ingestion_module = ChemCoTMolEditAdapter()
    all_records = ingestion_module.load(dataset_root)
    selected_record = next(
        (item for item in all_records if item.anonymous_sample_id == args.origin_id),
        None,
    )
    if selected_record is None:
        raise SystemExit(f"unknown origin ID: {args.origin_id}")
    _print_stage(
        "A INGESTION — load and join raw/process/template data",
        {"dataset_root": dataset_root, "origin_id": args.origin_id},
        selected_record,
    )
    if pause:
        input("\nPress Enter to continue to B REFERENCE...")

    # B. Joined records -> Reference DAG；同时从全语料建立 fragment pool。
    all_reference_artifacts = tuple(build_reference_dag(item) for item in all_records)
    reference_artifact = next(
        item
        for item in all_reference_artifacts
        if item.anonymous_sample_id == args.origin_id
    )
    fragment_pool = FragmentPool.from_reference_artifacts(all_reference_artifacts)
    reference_output = {
        "reference_artifact": reference_artifact,
        "fragment_pool_size": len(fragment_pool),
        "fragment_pool": fragment_pool.entries,
    }
    _print_stage(
        "B REFERENCE — build DAG and corpus fragment pool",
        selected_record,
        reference_output,
    )
    if pause:
        input("\nPress Enter to continue to C ERROR PLANNING...")

    # C. Reference DAG -> 一份包含 K 个直接修改点的统一 plan。
    planning_module = UnifiedHallucinationPlanner(
        fragment_pool=fragment_pool,
        config=DEFAULT_HALLUCINATION_CONFIG,
    )
    hallucination_plan = planning_module.plan(
        reference_artifact,
        variant_index=args.variant_index,
    )
    _print_stage(
        "C ERROR PLANNING — select edit count, targets, operators, and magnitudes",
        reference_output,
        hallucination_plan,
    )
    if pause:
        input("\nPress Enter to continue to D ERROR INJECTION...")

    # D. Plan -> 只应用 plan 中显式列出的节点修改；没有隐式传播。
    injection_module = UnifiedHallucinationInjector()
    injected = injection_module.apply(
        reference_artifact.state_dag,
        hallucination_plan,
    )
    _print_stage(
        "D ERROR INJECTION — apply every planned semantic edit",
        hallucination_plan,
        injected,
    )
    if pause:
        input("\nPress Enter to continue to E TEXT REALIZATION...")

    # E. 代码锁定 FORMAL；Poe 只重写自然语言；再拼成完整 step_text。
    poe_request = build_poe_rewrite_request(reference_artifact, injected)
    _print_stage(
        "E1 POE REQUEST — original context + modified FORMAL + locked placeholders",
        injected,
        poe_request,
    )
    if pause:
        input("\nPress Enter to send this request to Poe...")
    text_module = PoeTextRenderer(PoeStepTextAgent(DEFAULT_HALLUCINATION_CONFIG))
    try:
        rendered = text_module.render(reference_artifact, injected)
    except PoeTextRealizationError as error:
        raise SystemExit(str(error)) from None
    _print_stage(
        "E2 TEXT REALIZATION — Poe prose + locally locked FORMAL",
        poe_request,
        rendered,
    )
    if pause:
        input("\nPress Enter to continue to F ANNOTATION...")

    # F. Rendered text -> 每个被修改语义点对应的全部文本 span。
    annotation_module = UnifiedHallucinationAnnotator()
    annotated = annotation_module.annotate(rendered, hallucination_plan)
    _print_stage(
        "F ANNOTATION — label every occurrence of every edited node",
        rendered,
        annotated,
    )
    if pause:
        input("\nPress Enter to continue to G RECORD ASSEMBLY...")

    # G. Graph + text + spans -> 单一格式的 positive hallucination record。
    record_module = UnifiedRecordBuilder()
    released_record = record_module.build(
        reference_artifact,
        injected,
        annotated,
    )
    _print_stage(
        "G RECORD ASSEMBLY — create the new always-hallucinated schema",
        annotated,
        released_record,
    )
    if pause:
        input("\nPress Enter to continue to H RELEASE...")

    # H. 最终 record -> JSONL 文件。
    write_jsonl((released_record,), output_path)
    _print_stage(
        "H RELEASE — write one JSONL record",
        released_record,
        {
            "output_path": output_path,
            "record_count": 1,
            "record_id": released_record.data["record_id"],
        },
    )

    print("\n" + "=" * 100)
    print("PIPELINE FINISHED")
    print(f"output: {output_path}")
    print(f"record_id: {released_record.data['record_id']}")
    print(f"edit_count: {released_record.data['edit_count']}")
    print(
        "edited semantic points: "
        + ", ".join(
            item["semantic_target_id"]
            for item in released_record.data["mutation_events"]
        )
    )
    print(
        "hallucination spans: "
        f"{len(released_record.data['hallucination_spans'])}"
    )
