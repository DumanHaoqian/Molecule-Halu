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
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any


# Make ``python molhallulens/demo.py`` and ``python demo.py`` both work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import tiktoken
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molhallulens.config.hallucination_generation import (
    DEFAULT_HALLUCINATION_CONFIG,
    DEMO_TOKEN_ENCODING,
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


@lru_cache(maxsize=512)
def maximum_root_edit_count(origin_id: str) -> int:
    """Return this origin's exact safe maximum under current propagation rules."""

    corpus = load_demo_corpus()
    if origin_id not in corpus.references_by_id:
        raise ValueError(f"unknown origin ID: {origin_id}")
    return UnifiedHallucinationPlanner(
        corpus.fragment_pool,
        DEFAULT_HALLUCINATION_CONFIG,
    ).maximum_edit_count(corpus.references_by_id[origin_id])


def prepare_local_run(
    origin_id: str,
    variant_index: int,
    edit_count: int | None = None,
) -> LocalDemoRun:
    """Execute the same real A-to-D modules used by ``pipeline.py``."""

    if type(origin_id) is not str or not origin_id:
        raise ValueError("origin_id must be non-empty text")
    if type(variant_index) is not int or variant_index < 0:
        raise ValueError("variant_index must be a non-negative integer")
    run_config = DEFAULT_HALLUCINATION_CONFIG
    if edit_count is not None:
        if type(edit_count) is not int:
            raise TypeError("edit_count must be an integer")
        minimum = DEFAULT_HALLUCINATION_CONFIG.min_edit_count
        maximum = maximum_root_edit_count(origin_id)
        if not minimum <= edit_count <= maximum:
            raise ValueError(
                f"edit_count must be between {minimum} and {maximum}, inclusive"
            )
        run_config = replace(
            DEFAULT_HALLUCINATION_CONFIG,
            edit_count_mode="fixed",
            fixed_edit_count=edit_count,
        )
    corpus = load_demo_corpus()
    if origin_id not in corpus.records_by_id:
        raise ValueError(f"unknown origin ID: {origin_id}")
    source_record = corpus.records_by_id[origin_id]
    reference = corpus.references_by_id[origin_id]
    plan = UnifiedHallucinationPlanner(
        corpus.fragment_pool,
        run_config,
    ).plan(reference, variant_index=variant_index)
    injected = UnifiedHallucinationInjector(run_config).apply(reference.state_dag, plan)
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


def _source_molecule_html(reference: ReferenceDAGArtifact) -> str:
    """Draw the unedited input molecule, never the candidate or final answer."""
    smiles = str(reference.state_dag.values["source"].normalized_value)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("原始分子的 SMILES 无法解析，无法绘制结构图")
    # Atom maps belong to the indexed input; omit them for a readable structure.
    # This molecule is a new local object, so the reference DAG is untouched.
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 420)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg[svg.index("<svg"):]
    return (
        '<section class="source-molecule step-compare"><h3>原始分子（编辑前）</h3>'
        '<p>来源：' + html.escape(reference.anonymous_sample_id)
        + ' · 原始输入结构，不是修改后的产物。</p>'
        + svg
        + '<details><summary>查看原始 SMILES</summary><pre>'
        + html.escape(smiles) + '</pre></details></section>'
    )


def _step_comparison_html(run: TextDemoRun) -> str:
    original_steps = tuple(
        step.render(include_answer=False) for step in run.local.reference.trace_steps
    )
    modified_steps = run.annotated.rendered.step_texts
    cards = [
        _source_molecule_html(run.local.reference),
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
        "edit_count": plan.requested_edit_count,
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


@lru_cache(maxsize=1)
def _demo_encoding():
    return tiktoken.get_encoding(DEMO_TOKEN_ENCODING)


def _token_coverage(text: str, spans, encoding) -> dict[str, Any]:
    """Project character spans onto full-text tokens using exact UTF-8 offsets."""
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    covered = bytearray(offsets[-1])
    for start, end in spans:
        if not (0 <= start < end <= len(text)):
            raise ValueError("token coverage received invalid character span")
        a, b = offsets[start], offsets[end]
        covered[a:b] = b"\x01" * (b - a)
    token_ids = encoding.encode_ordinary(text)
    raw = text.encode("utf-8")
    cursor = 0
    hallucinated = 0
    for token_id in token_ids:
        piece = encoding.decode_single_token_bytes(token_id)
        end = cursor + len(piece)
        if raw[cursor:end] != piece:
            raise ValueError("tokenizer byte offsets do not round-trip")
        hallucinated += int(any(covered[cursor:end]))
        cursor = end
    if cursor != len(raw):
        raise ValueError("tokenizer did not cover the complete text")
    total = len(token_ids)
    return {
        "encoding": encoding.name,
        "scope": "serialized.text",
        "overlap_rule": "any_overlap_deduplicated",
        "hallucination_tokens": hallucinated,
        "total_tokens": total,
        "percentage": 100 * hallucinated / total if total else 0.0,
    }


def _text_outputs(run: TextDemoRun) -> tuple[Any, ...]:
    rendered = run.annotated.rendered
    realization = dict(rendered.realization)
    record = run.released.to_dict()
    coverage = _token_coverage(
        record["serialized"]["text"],
        [span["serialized_span"] for span in record["hallucination_spans"]],
        _demo_encoding(),
    )
    cache_text = "命中缓存" if realization.get("cache_hit") else "调用了 Poe"
    rewrite_modes = realization.get("step_rewrite_modes", ())
    mode_summary = ", ".join(
        f"{mode}={rewrite_modes.count(mode)}"
        for mode in ("copy", "occurrence_patch", "derivation_rewrite")
    )
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
        f"`{realization.get('network_request_count')}`；`{mode_summary}`。Poe 根据覆盖审计"
        "结果逐 occurrence 修改或整段重写；临时 HALLU markers 经旧值、算术和 FORMAL"
        "及枚举校验后移除。局部 patch 和 derivation rewrite 都逐 claim 标注，"
        "枚举传播产生的错误也单独标注，不再把整段正文标为幻觉；"
        f"下方共列出 **{len(run.annotated.spans)} 个训练标签 span**。"
        f"\n\n**幻觉 token 占比：{coverage['hallucination_tokens']} / "
        f"{coverage['total_tokens']} = {coverage['percentage']:.2f}%**\n\n"
        f"参考 tokenizer：`{coverage['encoding']}`。统计完整 `serialized.text`"
        "（原分子、指令、推理含 FORMAL、最终答案及分隔标记）；"
        "token 与幻觉 span 有任何重叠即计入，重复命中只计一次。"
        "不包含额外的模型 chat template/BOS/EOS；换用下游模型 tokenizer 后比例可能不同。"
    )
    return (
        e_status,
        _step_comparison_html(run),
        span_rows,
        _plain({**realization, "demo_token_coverage": coverage}),
        _plain(run.released.to_dict()),
    )


def _coerce_variant(value: Any) -> int:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError("variant index 必须是非负整数")
    converted = int(value)
    if converted < 0 or float(value) != converted:
        raise ValueError("variant index 必须是非负整数")
    return converted


def _coerce_edit_count(value: Any) -> int:
    minimum = DEFAULT_HALLUCINATION_CONFIG.min_edit_count
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"修改 node 数量必须是大于或等于 {minimum} 的整数")
    converted = int(value)
    if float(value) != converted or converted < minimum:
        raise ValueError(f"修改 node 数量必须是大于或等于 {minimum} 的整数")
    return converted


def _edit_count_slider_update(origin_id: str, current_value: Any) -> gr.Slider:
    """Update the selectable maximum when the user changes the origin."""

    try:
        maximum = maximum_root_edit_count(origin_id)
        try:
            current = _coerce_edit_count(current_value)
        except ValueError:
            current = DEFAULT_HALLUCINATION_CONFIG.fixed_edit_count
        return gr.Slider(
            minimum=DEFAULT_HALLUCINATION_CONFIG.min_edit_count,
            maximum=maximum,
            value=min(current, maximum),
            step=1,
            label="修改 root node 数量",
            info=f"当前样本的无冲突范围是 1–{maximum}；传播产生的下游 node 不计入这里",
        )
    except Exception as error:
        raise gr.Error(f"无法计算当前样本的最大修改数量：{error}") from None


def _run_all_ui(origin_id: str, variant_index: Any, edit_count: Any):
    """One click, one local plan, then Poe; clear stale E results before work."""
    cleared_e = ("正在运行，旧 E 结果已清空。", "", [], None, None)
    yield ("正在运行 A–D…", "", None, "", "", [], [], "", [], None, "", "", [], None) + cleared_e
    try:
        local = prepare_local_run(origin_id, _coerce_variant(variant_index), _coerce_edit_count(edit_count))
        local_outputs = _local_outputs(local)[1:]
        yield ("A–D 已完成，正在准备 tokenizer 并调用 Poe…",) + local_outputs[1:] + cleared_e
        _demo_encoding()  # Fail before a paid Poe request if tokenization is unavailable.
        text_run = complete_text_run(local)
        text_outputs = _text_outputs(text_run)
        yield (f"✅ A–E 已完成：`{origin_id}` / variant `{local.plan.variant_index}`。",) + local_outputs[1:] + text_outputs
    except Exception as error:
        message = f"A–E 运行失败：{error}"
        yield (message,) + (gr.skip(),) * 13 + (message, "", [], None, None)
        raise gr.Error(message) from None


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
.source-molecule svg { display: block; width: 100%; max-width: 1000px;
                       height: auto; margin: 0 auto; }
.source-molecule p { color: #475569; font-size: 13px; }
.source-molecule pre { white-space: pre-wrap; overflow-wrap: anywhere; }
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
    default_maximum_edit_count = maximum_root_edit_count(default_origin)
    token_available = bool(os.environ.get(DEFAULT_HALLUCINATION_CONFIG.poe_api_key_env))
    token_message = (
        "已检测到 `POE_API_KEY`。"
        if token_available
        else "当前进程未检测到 `POE_API_KEY`；缓存命中仍可运行，新请求需要先在 terminal 中 export。"
    )

    with gr.Blocks(title="MolHalluLens · A–E Pipeline Demo") as app:
        gr.Markdown(
            "# MolHalluLens · A–E 可视化流水线\n"
            "从 ChemCoTBench-V2 原始记录到 Reference DAG、修改计划，以及带精确 span 的文本对比。",
            elem_classes="hero",
        )
        gr.Markdown(
            "点击一次运行 A–E：先执行本地 A–D，再自动调用 Poe（可能消耗 API 点数）。" + token_message,
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
            edit_count_input = gr.Slider(
                minimum=DEFAULT_HALLUCINATION_CONFIG.min_edit_count,
                maximum=default_maximum_edit_count,
                value=min(
                    DEFAULT_HALLUCINATION_CONFIG.fixed_edit_count,
                    default_maximum_edit_count,
                ),
                step=1,
                label="修改 root node 数量",
                info=(
                    f"当前样本的无冲突范围是 "
                    f"{DEFAULT_HALLUCINATION_CONFIG.min_edit_count}–"
                    f"{default_maximum_edit_count}；传播产生的下游 node 不计入这里"
                ),
                scale=2,
            )
        with gr.Row():
            run_button = gr.Button("运行 A–E（含 Poe、Span 标签和 token 占比）", variant="primary")
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
                e_status = gr.Markdown("点击上方“运行 A–E”按钮，自动完成所有步骤。")
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

        run_button.click(
            fn=_run_all_ui,
            inputs=[origin_input, variant_input, edit_count_input],
            outputs=[
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
                e_status,
                e_compare,
                e_spans,
                e_provenance,
                e_record,
            ],
            api_name="run_all_stages",
            concurrency_limit=1,
            trigger_mode="once",
        )
        origin_input.change(
            fn=_edit_count_slider_update,
            inputs=[origin_input, edit_count_input],
            outputs=edit_count_input,
            api_name="update_edit_count_limit",
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
