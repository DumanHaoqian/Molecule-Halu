"""Local Gradio walkthrough with backend annotation merged into the E-stage UI."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any


# Make ``python molhallulens/demo.py`` and ``python demo.py`` both work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
)
from molhallulens.config.paths import DEFAULT_DATASET_ROOT
from molhallulens.core import InjectedHallucination, UnifiedHallucinationPlan
from molhallulens.modules.annotation import (
    AnnotatedHallucination,
    UnifiedHallucinationAnnotator,
)
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import (
    FragmentPool,
    UnifiedHallucinationPlanner,
)
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter, JoinedInputRecord
from molhallulens.modules.reference import ReferenceDAGArtifact, build_reference_dag
from molhallulens.modules.release import (
    ReleasedHallucinationRecord,
    UnifiedRecordBuilder,
)
from molhallulens.modules.text_realization import (
    PoeStepTextAgent,
    PoeTextRealizationError,
    PoeTextRenderer,
)


_JS_SAFE_INTEGER = 9_007_199_254_740_991


@dataclass(frozen=True, slots=True)
class DemoCorpus:
    """The validated Pilot corpus reused by every UI event."""

    records_by_id: Mapping[str, JoinedInputRecord]
    references_by_id: Mapping[str, ReferenceDAGArtifact]
    fragment_pool: FragmentPool


@dataclass(frozen=True, slots=True)
class LocalDemoRun:
    """A-to-D outputs for one deterministic origin/variant pair."""

    source_record: JoinedInputRecord
    reference: ReferenceDAGArtifact
    plan: UnifiedHallucinationPlan
    injected: InjectedHallucination


@dataclass(frozen=True, slots=True)
class TextDemoRun:
    """E-to-F outputs after Poe text realization and local annotation."""

    local: LocalDemoRun
    annotated: AnnotatedHallucination
    released: ReleasedHallucinationRecord


def _plain(value: Any) -> Any:
    """Convert project objects into browser-safe JSON without rounding large seeds."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if type(value) is int and abs(value) > _JS_SAFE_INTEGER:
        return str(value)
    return value


def _display_value(value: Any, *, limit: int | None = None) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True)
    if limit is not None and len(text) > limit:
        return text[: max(1, limit - 1)] + "…"
    return text


@lru_cache(maxsize=1)
def load_demo_corpus() -> DemoCorpus:
    """Load all origins and reference DAGs once for a responsive local UI."""

    records = ChemCoTMolEditAdapter().load(DEFAULT_DATASET_ROOT)
    references = tuple(build_reference_dag(record) for record in records)
    return DemoCorpus(
        records_by_id={item.anonymous_sample_id: item for item in records},
        references_by_id={item.anonymous_sample_id: item for item in references},
        fragment_pool=FragmentPool.from_reference_artifacts(references),
    )


def prepare_local_run(origin_id: str, variant_index: int) -> LocalDemoRun:
    """Execute the same real A-to-D modules used by ``pipeline.py``."""

    if type(origin_id) is not str or not origin_id:
        raise ValueError("origin_id must be non-empty text")
    if type(variant_index) is not int or variant_index < 0:
        raise ValueError("variant_index must be a non-negative integer")
    corpus = load_demo_corpus()
    if origin_id not in corpus.records_by_id:
        raise ValueError(f"unknown origin ID: {origin_id}")
    source_record = corpus.records_by_id[origin_id]
    reference = corpus.references_by_id[origin_id]
    plan = UnifiedHallucinationPlanner(
        corpus.fragment_pool,
        DEFAULT_HALLUCINATION_CONFIG,
    ).plan(reference, variant_index=variant_index)
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    return LocalDemoRun(source_record, reference, plan, injected)


def complete_text_run(
    local: LocalDemoRun,
    *,
    agent: PoeStepTextAgent | None = None,
) -> TextDemoRun:
    """Execute E-to-F with Poe prose rewriting and exact local span annotation."""

    if type(local) is not LocalDemoRun:
        raise TypeError("local must be LocalDemoRun")
    poe_agent = PoeStepTextAgent(DEFAULT_HALLUCINATION_CONFIG) if agent is None else agent
    rendered = PoeTextRenderer(poe_agent).render(local.reference, local.injected)
    annotated = UnifiedHallucinationAnnotator().annotate(rendered, local.injected)
    released = UnifiedRecordBuilder().build(local.reference, local.injected, annotated)
    return TextDemoRun(local, annotated, released)


def _dag_payload(reference: ReferenceDAGArtifact, *, candidate: bool) -> dict[str, Any]:
    graph = reference.state_dag
    if candidate:
        raise ValueError("candidate graph must be supplied through _candidate_payload")
    return {
        "schema_id": graph.schema.schema_id,
        "schema_version": graph.schema.version,
        "nodes": {
            node.node_id: {
                "step_index": node.step_index,
                "role": node.role.value,
                "visibility": node.visibility.value,
                "mutable": node.mutable,
                "value_type": node.value_type.value,
                "value": _plain(graph.values[node.node_id].normalized_value),
                "provenance": graph.values[node.node_id].provenance.value,
            }
            for node in graph.schema.nodes
        },
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation.value,
            }
            for edge in graph.schema.edges
        ],
    }


def _candidate_payload(injected: InjectedHallucination) -> dict[str, Any]:
    graph = injected.candidate_graph
    return {
        "schema_id": graph.schema.schema_id,
        "root_node_ids": list(injected.plan.edited_node_ids),
        "changed_node_ids": list(injected.changed_node_ids),
        "propagation_events": _plain(injected.propagation_events),
        "edge_audit": _plain(injected.edge_audit),
        "violated_edge_ids": list(injected.violated_edge_ids),
        "nodes": {
            node.node_id: {
                "step_index": node.step_index,
                "role": node.role.value,
                "visibility": node.visibility.value,
                "mutable": node.mutable,
                "value_type": node.value_type.value,
                "value": _plain(graph.values[node.node_id].normalized_value),
                "provenance": graph.values[node.node_id].provenance.value,
            }
            for node in graph.schema.nodes
        },
    }


def _dag_html(
    artifact: ReferenceDAGArtifact,
    *,
    graph: Any | None = None,
    changed_nodes: Sequence[str] = (),
) -> str:
    """Render the actual StateDAG as a dependency-free, responsive SVG."""

    dag = artifact.state_dag if graph is None else graph
    schema = dag.schema
    node_ids = schema.topological_order()
    depth = {node_id: 0 for node_id in node_ids}
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in schema.edges:
        incoming[edge.target].append(edge.source)
    for node_id in node_ids:
        if incoming[node_id]:
            depth[node_id] = max(depth[parent] + 1 for parent in incoming[node_id])

    columns: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        columns[depth[node_id]].append(node_id)
    max_rows = max(len(items) for items in columns.values())
    node_width, node_height = 190, 66
    x_gap, y_gap = 92, 24
    margin_x, margin_y = 45, 38
    width = margin_x * 2 + (max(columns) + 1) * node_width + max(columns) * x_gap
    height = margin_y * 2 + max_rows * node_height + max(0, max_rows - 1) * y_gap
    positions: dict[str, tuple[float, float]] = {}
    for column, items in sorted(columns.items()):
        column_height = len(items) * node_height + max(0, len(items) - 1) * y_gap
        start_y = (height - column_height) / 2
        x = margin_x + column * (node_width + x_gap)
        for row, node_id in enumerate(items):
            positions[node_id] = (x, start_y + row * (node_height + y_gap))

    changed = set(changed_nodes)
    edge_svg = []
    for edge in schema.edges:
        source_x, source_y = positions[edge.source]
        target_x, target_y = positions[edge.target]
        x1, y1 = source_x + node_width, source_y + node_height / 2
        x2, y2 = target_x, target_y + node_height / 2
        bend = max(30, (x2 - x1) * 0.45)
        relation = html.escape(edge.relation.value)
        edge_svg.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {x1 + bend:.1f} {y1:.1f}, '
            f'{x2 - bend:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" '
            'fill="none" stroke="#94a3b8" stroke-width="1.6" '
            'marker-end="url(#dag-arrow)"><title>' + relation + "</title></path>"
        )

    specs = schema.nodes_by_id
    node_svg = []
    for node_id in node_ids:
        spec = specs[node_id]
        claim = dag.values[node_id]
        x, y = positions[node_id]
        if node_id in changed:
            fill, stroke, stroke_width = "#ecfccb", "#65a30d", "3"
        elif spec.visibility.value == "build_only":
            fill, stroke, stroke_width = "#f1f5f9", "#94a3b8", "1.5"
        elif not spec.mutable:
            fill, stroke, stroke_width = "#ecfeff", "#0891b2", "1.5"
        else:
            fill, stroke, stroke_width = "#eff6ff", "#3b82f6", "1.5"
        step_label = "global" if spec.step_index is None else f"Step {spec.step_index}"
        value = _display_value(claim.normalized_value, limit=24)
        title = html.escape(
            f"{node_id}\nvalue={_display_value(claim.normalized_value)}\n"
            f"type={spec.value_type.value}\nrole={spec.role.value}"
        )
        node_svg.append(
            f'<g><title>{title}</title><rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{node_width}" height="{node_height}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
            f'<text x="{x + 12:.1f}" y="{y + 20:.1f}" fill="#0f172a" '
            f'font-size="13" font-weight="700">{html.escape(node_id)}</text>'
            f'<text x="{x + 12:.1f}" y="{y + 39:.1f}" fill="#475569" '
            f'font-size="11">{html.escape(step_label)} · {html.escape(spec.value_type.value)}</text>'
            f'<text x="{x + 12:.1f}" y="{y + 56:.1f}" fill="#334155" '
            f'font-size="11">{html.escape(value)}</text></g>'
        )

    return (
        '<div class="dag-shell"><div class="dag-legend">'
        '<span class="legend input"></span>输入/证据 '
        '<span class="legend editable"></span>可编辑节点 '
        '<span class="legend oracle"></span>仅构建时可见 '
        '<span class="legend changed"></span>本次修改</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(schema.schema_id)} DAG">'
        '<defs><marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>'
        + "".join(edge_svg)
        + "".join(node_svg)
        + "</svg></div>"
    )


def _reference_node_rows(reference: ReferenceDAGArtifact) -> list[list[Any]]:
    graph = reference.state_dag
    specs = graph.schema.nodes_by_id
    rows = []
    for node_id in graph.schema.topological_order():
        spec = specs[node_id]
        claim = graph.values[node_id]
        rows.append(
            [
                node_id,
                "global" if spec.step_index is None else spec.step_index,
                spec.role.value,
                spec.value_type.value,
                spec.mutable,
                spec.visibility.value,
                _display_value(claim.normalized_value),
                claim.provenance.value,
            ]
        )
    return rows


def _parsed_formal_rows(reference: ReferenceDAGArtifact) -> list[list[Any]]:
    rows = []
    for step in reference.trace_steps:
        for binding in step.slot_bindings:
            rows.append(
                [
                    step.step_index,
                    step.step_name,
                    binding.source_field,
                    binding.node_id,
                    _display_value(
                        reference.state_dag.values[binding.node_id].normalized_value
                    ),
                ]
            )
    return rows


def _plan_rows(plan: UnifiedHallucinationPlan) -> list[list[Any]]:
    return [
        [
            mutation.semantic_target_id,
            ", ".join(mutation.target_node_ids),
            mutation.mutation_category.value,
            mutation.operator,
            _display_value(mutation.before),
            _display_value(mutation.after),
            mutation.magnitude,
            mutation.similarity,
        ]
        for mutation in plan.mutations
    ]


def _comparison_rows(local: LocalDemoRun) -> list[list[Any]]:
    mutation_by_node = {
        node_id: mutation
        for mutation in local.plan.mutations
        for node_id in mutation.target_node_ids
    }
    root_rows = [
        [
            "root_hallucination",
            mutation_by_node[node_id].semantic_target_id,
            node_id,
            local.reference.state_dag.values[node_id].value_type.value,
            _display_value(local.injected.reference_graph.values[node_id].normalized_value),
            _display_value(local.injected.candidate_graph.values[node_id].normalized_value),
            mutation_by_node[node_id].operator,
        ]
        for node_id in local.plan.edited_node_ids
    ]
    propagated_rows = [
        [
            "propagated_error",
            event.root_semantic_target_id,
            event.target_node_id,
            local.reference.state_dag.values[event.target_node_id].value_type.value,
            _display_value(event.before),
            _display_value(event.after),
            event.rule_id,
        ]
        for event in local.injected.propagation_events
    ]
    return root_rows + propagated_rows


_DIFF_TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z0-9_]+|.", re.DOTALL)


def _inline_diff_html(original: str, modified: str) -> tuple[str, str]:
    """Render a readable, escaped token diff for the two comparison columns."""

    original_tokens = _DIFF_TOKEN_PATTERN.findall(original)
    modified_tokens = _DIFF_TOKEN_PATTERN.findall(modified)
    matcher = difflib.SequenceMatcher(
        None,
        original_tokens,
        modified_tokens,
        autojunk=False,
    )
    original_html: list[str] = []
    modified_html: list[str] = []
    for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        left_text = html.escape("".join(original_tokens[left_start:left_end]))
        right_text = html.escape("".join(modified_tokens[right_start:right_end]))
        if operation == "equal":
            original_html.append(left_text)
            modified_html.append(right_text)
        else:
            if left_text:
                original_html.append(
                    '<mark class="diff-before" title="原始文本中被替换或删除的内容">'
                    + left_text
                    + "</mark>"
                )
            if right_text:
                modified_html.append(
                    '<mark class="diff-after" title="Poe 重写后新增或替换的内容">'
                    + right_text
                    + "</mark>"
                )
    return "".join(original_html), "".join(modified_html)


def _comparison_card(title: str, original: str, modified: str) -> str:
    original_html, modified_html = _inline_diff_html(original, modified)
    return (
        '<section class="step-compare"><h3>'
        + html.escape(title)
        + '</h3><div class="compare-grid"><article><h4>原始 ChemCoTBench-V2</h4><pre>'
        + original_html
        + '</pre></article><article><h4>Poe 重写 + 本地锁定 FORMAL</h4><pre>'
        + modified_html
        + "</pre></article></div></section>"
    )


def _step_comparison_html(run: TextDemoRun) -> str:
    original_steps = tuple(
        step.render(include_answer=False) for step in run.local.reference.trace_steps
    )
    modified_steps = run.annotated.rendered.step_texts
    cards = [
        '<div class="diff-legend">'
        '<span><i class="diff-before-swatch"></i>左侧：被删除或替换的原文</span>'
        '<span><i class="diff-after-swatch"></i>右侧：新增或替换后的文本</span>'
        '<span class="label-note">颜色展示文字差异；下方 span 表才是精确训练标签。</span>'
        "</div>"
    ]
    for index, (original, modified) in enumerate(
        zip(original_steps, modified_steps, strict=True),
        start=1,
    ):
        cards.append(_comparison_card(f"Step {index}", original, modified))
    original_answer = _display_value(
        run.local.reference.state_dag.values["final_answer"].normalized_value
    )
    modified_answer = run.annotated.rendered.final_answer
    cards.append(_comparison_card("Final answer", original_answer, modified_answer))
    return "".join(cards)


def _local_outputs(local: LocalDemoRun) -> tuple[Any, ...]:
    record = local.source_record
    reference = local.reference
    plan = local.plan
    injected = local.injected
    token = {
        "origin_id": reference.anonymous_sample_id,
        "variant_index": plan.variant_index,
    }
    a_summary = (
        f"### A · 原始数据\n加载并按 `{reference.anonymous_sample_id}` 连接 raw、"
        f"process-evaluation 和 FORMAL template；原始推理共有 "
        f"**{len(reference.trace_steps)} steps**。"
    )
    a_data = {
        "anonymous_sample_id": record.anonymous_sample_id,
        "raw_record": _plain(record.raw_record),
        "process_record": _plain(record.process_record),
        "formal_template": _plain(record.formal_template),
    }
    graph = reference.state_dag
    b_summary = (
        f"### B · Reference DAG\n`{graph.schema.schema_id}` 包含 "
        f"**{len(graph.schema.nodes)} nodes / {len(graph.schema.edges)} directed edges**。"
        "下方节点表包含所有 DAG 值；FORMAL 解析表显示每个原字段如何绑定到 node。"
    )
    c_targets = "、".join(item.semantic_target_id for item in plan.mutations)
    c_summary = (
        f"### C · 修改计划\n本条记录计划修改 **{len(plan.mutations)} 个语义点**："
        f"`{c_targets}`。derived seed 为 `{plan.derived_seed}`。"
    )
    d_summary = (
        "### D · 注入与传播结果\n绿色荧光边框包含根错误和确定性传播结果："
        f"**{len(plan.edited_node_ids)} 个 root nodes + "
        f"{len(injected.propagation_events)} 个 propagated nodes**。"
    )
    return (
        token,
        (
            f"✅ 已完成本地 A–D：`{reference.anonymous_sample_id}` / "
            f"variant `{plan.variant_index}`。尚未调用 Poe。"
        ),
        a_summary,
        a_data,
        b_summary,
        _dag_html(reference),
        _reference_node_rows(reference),
        _parsed_formal_rows(reference),
        c_summary,
        _plan_rows(plan),
        _plain(plan),
        d_summary,
        _dag_html(
            reference,
            graph=injected.candidate_graph,
            changed_nodes=injected.changed_node_ids,
        ),
        _comparison_rows(local),
        _candidate_payload(injected),
    )


def _text_outputs(run: TextDemoRun) -> tuple[Any, ...]:
    rendered = run.annotated.rendered
    realization = dict(rendered.realization)
    cache_text = "命中缓存" if realization.get("cache_hit") else "调用了 Poe"
    span_rows = [
        [
            span.step_index if span.step_index is not None else "final_answer",
            span.semantic_target_id,
            span.node_id,
            span.operator,
            span.causal_role.value,
            span.start,
            span.end,
            span.text,
        ]
        for span in run.annotated.spans
    ]
    e_status = (
        f"### E · 文本对比 + Hallucination span\n✅ `{cache_text}`；network request count = "
        f"`{realization.get('network_request_count')}`。Poe 根据 modified `formal_ab` 重写"
        "受影响的原始 step_text；临时 HALLU markers 经本地校验、移除并转成"
        "精确 span，FORMAL 始终等于 modified `formal_ab`。左右栏直接高亮所有文本差异；"
        f"下方列出 **{len(run.annotated.spans)} 个精确训练标签 span**。"
    )
    return (
        e_status,
        _step_comparison_html(run),
        span_rows,
        _plain(realization),
        _plain(run.released.to_dict()),
    )


def _coerce_variant(value: Any) -> int:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError("variant index 必须是非负整数")
    converted = int(value)
    if converted < 0 or float(value) != converted:
        raise ValueError("variant index 必须是非负整数")
    return converted


def _run_local_ui(origin_id: str, variant_index: Any) -> tuple[Any, ...]:
    try:
        return _local_outputs(prepare_local_run(origin_id, _coerce_variant(variant_index)))
    except Exception as error:
        raise gr.Error(f"A–D 运行失败：{error}") from None


def _run_poe_ui(run_token: Mapping[str, Any] | None) -> tuple[Any, ...]:
    if not isinstance(run_token, Mapping):
        raise gr.Error("请先点击“运行 A–D（本地）”生成同一条样本的修改计划。")
    try:
        local = prepare_local_run(
            str(run_token["origin_id"]),
            _coerce_variant(run_token["variant_index"]),
        )
        return _text_outputs(complete_text_run(local))
    except PoeTextRealizationError as error:
        raise gr.Error(f"Poe 阶段失败：{error}") from None
    except Exception as error:
        raise gr.Error(f"E 运行失败：{error}") from None


_CSS = """
.gradio-container { max-width: 1540px !important; }
.hero { padding: 10px 2px 4px; }
.hero h1 { letter-spacing: -0.035em; margin-bottom: 0.25rem; }
.control-note { color: #475569; font-size: 0.94rem; }
.dag-shell { overflow-x: auto; border: 1px solid #dbe3ee; border-radius: 14px;
             background: #fff; padding: 12px; }
.dag-shell svg { display: block; width: 100%; min-width: 1050px; height: auto; }
.dag-legend { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
              color: #475569; font-size: 12px; margin: 0 0 8px 4px; }
.legend { width: 13px; height: 13px; border-radius: 4px; display: inline-block; margin-left: 8px; }
.legend.input { background: #ecfeff; border: 1px solid #0891b2; }
.legend.editable { background: #eff6ff; border: 1px solid #3b82f6; }
.legend.oracle { background: #f1f5f9; border: 1px solid #94a3b8; }
.legend.changed { background: #ecfccb; border: 2px solid #65a30d; }
.step-compare { border: 1px solid #dbe3ee; border-radius: 14px; padding: 14px;
                margin: 0 0 14px; background: #fff; }
.step-compare h3 { margin: 0 0 10px; color: #0f172a; }
.compare-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.compare-grid article { min-width: 0; }
.compare-grid h4 { margin: 0 0 7px; color: #475569; font-size: 13px; }
.compare-grid pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0;
    border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; padding: 13px;
    font-size: 12px; line-height: 1.55; color: #172033; }
.diff-legend { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin: 0 0 14px;
               padding: 10px 12px; border: 1px solid #dbe3ee; border-radius: 10px;
               background: #fff; color: #334155; font-size: 12px; }
.diff-legend span { display: inline-flex; align-items: center; gap: 6px; }
.diff-legend .label-note { color: #64748b; }
.diff-before-swatch, .diff-after-swatch { width: 14px; height: 14px; border-radius: 3px;
                                         display: inline-block; }
.diff-before-swatch, mark.diff-before { background: #fecaca; box-shadow: 0 0 0 1px #f87171; }
.diff-after-swatch, mark.diff-after { background: #ccff00; box-shadow: 0 0 0 1px #84cc16; }
mark.diff-before, mark.diff-after { color: #111827; padding: 1px 2px; border-radius: 3px;
                                    font-weight: 800; }
@media (max-width: 850px) { .compare-grid { grid-template-columns: 1fr; } }
"""


def build_demo() -> gr.Blocks:
    """Build the Gradio UI without launching a server."""

    corpus = load_demo_corpus()
    origin_ids = tuple(corpus.records_by_id)
    default_origin = (
        "mol_edit.add_v2.0003"
        if "mol_edit.add_v2.0003" in corpus.records_by_id
        else origin_ids[0]
    )
    token_available = bool(os.environ.get(DEFAULT_HALLUCINATION_CONFIG.poe_api_key_env))
    token_message = (
        "已检测到 `POE_API_KEY`。"
        if token_available
        else "当前进程未检测到 `POE_API_KEY`；缓存命中仍可运行，新请求需要先在 terminal 中 export。"
    )

    with gr.Blocks(title="MolHalluLens · A–E Pipeline Demo") as app:
        run_state = gr.State(value=None)
        gr.Markdown(
            "# MolHalluLens · A–E 可视化流水线\n"
            "从 ChemCoTBench-V2 原始记录到 Reference DAG、修改计划，以及带精确 span 的文本对比。",
            elem_classes="hero",
        )
        gr.Markdown(
            "先运行免费的本地 A–D；检查修改计划后，再明确点击 Poe 按钮。" + token_message,
            elem_classes="control-note",
        )
        with gr.Row():
            origin_input = gr.Dropdown(
                choices=origin_ids,
                value=default_origin,
                label="Origin ID",
                info=f"已加载 {len(origin_ids)} 条通过校验的 molecule-editing origins",
                scale=5,
            )
            variant_input = gr.Number(
                value=0,
                precision=0,
                minimum=0,
                step=1,
                label="Variant index",
                info="相同 origin + variant + config 会产生相同 plan",
                scale=1,
            )
        with gr.Row():
            local_button = gr.Button("① 运行 A–D（本地，不调用 Poe）", variant="primary")
            poe_button = gr.Button("② 调用 Poe，完成 E（含 span 标签）", variant="secondary")
        overall_status = gr.Markdown("尚未运行。")

        with gr.Tabs():
            with gr.Tab("A · 原始数据"):
                a_summary = gr.Markdown()
                a_json = gr.JSON(label="Joined input：raw + process evaluation + FORMAL template")

            with gr.Tab("B · Reference DAG"):
                b_summary = gr.Markdown()
                b_graph = gr.HTML(label="Reference DAG")
                with gr.Accordion("全部 DAG nodes", open=False):
                    b_nodes = gr.Dataframe(
                        headers=[
                            "node_id",
                            "step",
                            "role",
                            "value_type",
                            "mutable",
                            "visibility",
                            "parsed value",
                            "provenance",
                        ],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Accordion("formal_ab → DAG node 解析明细", open=True):
                    b_parsed = gr.Dataframe(
                        headers=["step", "step_name", "source_field", "node_id", "parsed value"],
                        interactive=False,
                        wrap=True,
                    )

            with gr.Tab("C · 修改 Plan"):
                c_summary = gr.Markdown()
                c_table = gr.Dataframe(
                    headers=[
                        "semantic target",
                        "physical node(s)",
                        "category",
                        "operator",
                        "before",
                        "after",
                        "magnitude",
                        "similarity",
                    ],
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion("完整 plan JSON", open=False):
                    c_json = gr.JSON()

            with gr.Tab("D · 修改前后"):
                d_summary = gr.Markdown()
                d_graph = gr.HTML(label="Candidate DAG（绿色为修改节点）")
                d_table = gr.Dataframe(
                    headers=[
                        "causal role",
                        "root semantic target",
                        "node_id",
                        "type",
                        "before",
                        "after",
                        "operator / propagation rule",
                    ],
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion("完整 candidate DAG JSON", open=False):
                    d_json = gr.JSON()

            with gr.Tab("E · 文本对比 + Span"):
                e_status = gr.Markdown("请先运行 A–D，再点击 Poe 按钮。")
                e_compare = gr.HTML(label="原始 / 修改后 step_text")
                gr.Markdown("#### 精确训练标签（已并入 E）")
                e_spans = gr.Dataframe(
                    headers=[
                        "step/component",
                        "semantic target",
                        "node_id",
                        "operator",
                        "causal role",
                        "start",
                        "end",
                        "text",
                    ],
                    interactive=False,
                    wrap=True,
                )
                with gr.Accordion("Poe realization provenance", open=False):
                    e_provenance = gr.JSON()
                with gr.Accordion("完整最终 record（未写入文件）", open=False):
                    e_record = gr.JSON()

        local_button.click(
            fn=_run_local_ui,
            inputs=[origin_input, variant_input],
            outputs=[
                run_state,
                overall_status,
                a_summary,
                a_json,
                b_summary,
                b_graph,
                b_nodes,
                b_parsed,
                c_summary,
                c_table,
                c_json,
                d_summary,
                d_graph,
                d_table,
                d_json,
            ],
            api_name="run_local_stages",
        )
        poe_button.click(
            fn=_run_poe_ui,
            inputs=run_state,
            outputs=[
                e_status,
                e_compare,
                e_spans,
                e_provenance,
                e_record,
            ],
            api_name="run_poe_stages",
            concurrency_limit=1,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        show_error=True,
        css=_CSS,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DemoCorpus",
    "LocalDemoRun",
    "TextDemoRun",
    "build_demo",
    "complete_text_run",
    "load_demo_corpus",
    "main",
    "prepare_local_run",
]
