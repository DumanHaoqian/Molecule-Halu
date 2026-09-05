"""Read-only Gradio replay of successfully released H/N pairs; never calls Poe."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

# Also support ``cd scripts && python gendemo.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from molhallulens.core import StateDAG, ValueProvenance
from molhallulens.config.paths import DEFAULT_DATASET_ROOT
from molhallulens.demo import (
    _CSS, _comparison_card, _dag_html, _demo_encoding, _display_value,
    _parsed_formal_rows, _plain, _reference_node_rows, _source_molecule_html,
    _token_coverage,
)
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag

DEFAULT_RECORDS = PROJECT_ROOT / "GeneratedDataset" / "maximum_edits_complete.jsonl"
CSS = _CSS + """
mark.saved-hallu { background: #ccff00; color: #172033; border-radius: 2px; }
mark.saved-control { background: #bae6fd; color: #172033; border-radius: 2px; }
.pair-warning { color: #9a3412; padding: 8px; background: #fff7ed; }
"""


def _replace_spans(text, spans, controls, *, base=0, field="span"):
    for span in sorted(spans, key=lambda s: s[field][0], reverse=True):
        start, end = (value - base for value in span[field])
        if text[start:end] != span["text"]:
            raise ValueError("saved span does not match text")
        text = text[:start] + controls[span["pair_occurrence_id"]]["text"] + text[end:]
    return text


def _step_offsets(record):
    chain = record["detector_input"]["reasoning_chain"]
    cursor = 0
    offsets = []
    for step in record["step_texts"]:
        start = chain.index(step, cursor)
        offsets.append(start)
        cursor = start + len(step)
    return offsets


def validate_pair(h, n):
    """Reject damaged/incomplete releases; never regenerate missing content."""
    if (h["pair_id"], h["origin_id"], h["variant_index"]) != (
        n["pair_id"], n["origin_id"], n["variant_index"]
    ):
        raise ValueError("H/N identity mismatch")
    if h["matched_record_id"] != n["record_id"] or n["matched_record_id"] != h["record_id"]:
        raise ValueError("H/N record links mismatch")
    if (h["labels"]["hallucination_present"] is not True
            or n["labels"]["hallucination_present"] is not False
            or not h["hallucination_spans"] or n["hallucination_spans"]):
        raise ValueError("invalid H/N labels")
    for record in (h, n):
        text = record["serialized"]["text"]
        if hashlib.sha256(text.encode()).hexdigest() != record["serialized"]["sha256"]:
            raise ValueError("saved text SHA256 mismatch")
        for span in record["hallucination_spans"] + record["control_spans"]:
            for field, value in (("serialized_span", text),
                                 ("span", record["detector_input"][span["component"]])):
                start, end = span[field]
                if not 0 <= start < end <= len(value) or value[start:end] != span["text"]:
                    raise ValueError("saved span offset/text mismatch")
    controls = {s["pair_occurrence_id"]: s for s in n["control_spans"]}
    ids = [s["pair_occurrence_id"] for s in h["hallucination_spans"]]
    if len(controls) != len(n["control_spans"]) or len(set(ids)) != len(ids) or set(ids) != set(controls):
        raise ValueError("H spans and N controls must correspond one-to-one")
    for span in h["hallucination_spans"]:
        control = controls[span["pair_occurrence_id"]]
        same = len(span["text"]) == len(control["text"])
        if span["same_char_length"] != same or control["same_char_length"] != same:
            raise ValueError("same_char_length mismatch")
    alignments = h["pair_alignment"]
    if (alignments != n["pair_alignment"] or len(h["step_texts"]) != len(n["step_texts"])
            or [a["step_index"] for a in alignments] != list(range(1, len(h["step_texts"]) + 1))):
        raise ValueError("step alignment mismatch")
    bases = _step_offsets(h)
    for alignment in alignments:
        mode = alignment["pair_alignment"]
        if mode not in {"byte_identical", "regenerated"}:
            raise ValueError("unknown pair alignment")
        index = alignment["step_index"]
        if mode == "byte_identical":
            spans = [s for s in h["hallucination_spans"] if s["step_index"] == index]
            restored = _replace_spans(h["step_texts"][index - 1], spans, controls, base=bases[index - 1])
            if restored != n["step_texts"][index - 1]:
                raise ValueError("byte-identical step invariant failed")
    if all(a["pair_alignment"] == "byte_identical" for a in alignments):
        restored = _replace_spans(h["serialized"]["text"], h["hallucination_spans"], controls, field="serialized_span")
        if restored != n["serialized"]["text"]:
            raise ValueError("full serialized pair invariant failed")


def load_pairs(path=DEFAULT_RECORDS):
    """Load the successful JSONL only; the failure manifest is never a data source."""
    grouped = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                label = record["variant_label"]
                pair = grouped.setdefault(record["pair_id"], {})
                if label not in {"H", "N"} or label in pair:
                    raise ValueError("unknown or duplicate variant label")
                pair[label] = record
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid saved record at line {line_number}: {error}") from error
    if not grouped:
        raise ValueError("No saved successful pairs found")
    for pair_id, pair in grouped.items():
        if set(pair) != {"H", "N"}:
            raise ValueError(f"Incomplete saved pair: {pair_id}")
        validate_pair(pair["H"], pair["N"])
    return grouped


def replay_graph(reference, record):
    """Restore saved before/after values, without running planner or propagation."""
    values = dict(reference.state_dag.values)
    rows, changed = [], set()
    root_events = [
        (node, event, "root_hallucination", event["semantic_target_id"], event["operator"])
        for event in record["mutation_events"] for node in event["target_node_ids"]
    ]
    downstream = [
        (event["target_node_id"], event, "propagated_error", event["root_semantic_target_id"], event["rule_id"])
        for event in record["propagation_events"]
    ]
    for node_id, event, role, target, operator in root_events + downstream:
        if node_id in changed or node_id not in values:
            raise ValueError(f"Duplicate/unknown saved mutation node: {node_id}")
        original = values[node_id]
        if original.normalized_value != event["before"]:
            raise ValueError(f"Reference data no longer matches saved before value: {node_id}")
        values[node_id] = replace(original, raw_value=event["after"], normalized_value=event["after"],
                                  provenance=ValueProvenance.RULE, locally_valid=None, oracle_match=False)
        changed.add(node_id)
        rows.append([role, target, node_id, original.value_type.value,
                     _display_value(event["before"]), _display_value(event["after"]), operator])
    return StateDAG(reference.state_dag.schema, values), rows, sorted(changed)


def _marked_html(text, spans, *, base=0, control=False):
    cursor, pieces = 0, []
    for span in sorted(spans, key=lambda s: s["span"][0]):
        start, end = (value - base for value in span["span"])
        if start < cursor or text[start:end] != span["text"]:
            raise ValueError("Cannot highlight overlapping or invalid saved spans")
        css = "saved-control" if control else "saved-hallu"
        title = html.escape(f"{span['node_id']} | {span['pair_occurrence_id']}", quote=True)
        pieces.extend([html.escape(text[cursor:start]),
                       f'<mark class="{css}" title="{title}">{html.escape(text[start:end])}</mark>'])
        cursor = end
    return "".join(pieces) + html.escape(text[cursor:])


def pair_comparison(h, n):
    """Use saved offsets, not a text diff, to highlight actual training labels."""
    h_offsets, n_offsets = _step_offsets(h), _step_offsets(n)
    cards = []
    for index in range(len(h["step_texts"]) + 1):
        final = index == len(h["step_texts"])
        step_id = None if final else index + 1
        mode = "final_answer" if final else h["pair_alignment"][index]["pair_alignment"]
        title = "Final answer" if final else f"Step {index + 1} · {mode}"
        columns = []
        for record, offsets, control in ((n, n_offsets, True), (h, h_offsets, False)):
            text = record["detector_input"]["final_answer"] if final else record["step_texts"][index]
            spans = [s for s in record["control_spans" if control else "hallucination_spans"] if s["step_index"] == step_id]
            marked = _marked_html(text, spans, base=0 if final else offsets[index], control=control)
            label = "N · 真值对照（蓝色 control spans）" if control else "H · 幻觉版本（荧光训练标签）"
            columns.append(f"<article><h4>{label}</h4><pre>{marked}</pre></article>")
        warning = ('<p class="pair-warning">此步骤为 regenerated：标注之外的文字可能不同。</p>'
                   if mode == "regenerated" else "")
        cards.append(f'<section class="step-compare"><h3>{title}</h3>{warning}'
                     + '<div class="compare-grid">' + "".join(columns) + "</div></section>")
    return "".join(cards)


class SavedDemo:
    def __init__(self, path=DEFAULT_RECORDS, dataset_root=DEFAULT_DATASET_ROOT):
        self.path = Path(path)
        self.pairs = load_pairs(self.path)
        ids = {pair["H"]["origin_id"] for pair in self.pairs.values()}
        self.sources = {r.anonymous_sample_id: r for r in ChemCoTMolEditAdapter().load(Path(dataset_root))
                        if r.anonymous_sample_id in ids}
        if set(self.sources) != ids:
            raise ValueError("Some successful origins are missing from the source dataset")
        self.references = {key: build_reference_dag(value) for key, value in self.sources.items()}

    def outputs(self, pair_id):
        pair = self.pairs[pair_id]
        h, n = pair["H"], pair["N"]
        reference = self.references[h["origin_id"]]
        source = self.sources[h["origin_id"]]
        graph, changes, changed_ids = replay_graph(reference, h)
        if len(reference.trace_steps) != len(h["step_texts"]):
            raise ValueError("Reference step count differs from the saved record")
        events = h["mutation_events"]
        plan_rows = [[e["semantic_target_id"], ", ".join(e["target_node_ids"]), e["mutation_category"],
                      e["operator"], _display_value(e["before"]), _display_value(e["after"]),
                      e["magnitude"], e["similarity"]] for e in events]
        coverage = _token_coverage(h["serialized"]["text"],
                                   [s["serialized_span"] for s in h["hallucination_spans"]], _demo_encoding())
        provenance = h["text_realization"]
        regenerated = sum(a["pair_alignment"] == "regenerated" for a in h["pair_alignment"])
        status = (f"### E · 已保存文本 + 精确 Span\n本次 Poe 请求 **0**；历史模型："
                  f"`{provenance.get('bot_name', '未记录')}`。共 **{len(h['hallucination_spans'])} 个幻觉 span**。"
                  f"\n\n**幻觉 token 占比：{coverage['hallucination_tokens']} / {coverage['total_tokens']}"
                  f" = {coverage['percentage']:.2f}%**。参考 tokenizer：`{coverage['encoding']}`；"
                  "范围为完整 `serialized.text`，任意重叠计入一次，不含额外 chat template/BOS/EOS。"
                  f"\n\n本对有 **{regenerated} 个 regenerated 步骤**，这些步骤不保证标注之外逐字节一致。")
        original_cards = [_comparison_card(f"Step {i}", step.render(include_answer=False), saved)
                          for i, (step, saved) in enumerate(zip(reference.trace_steps, h["step_texts"], strict=True), 1)]
        original_cards.append(_comparison_card("Final answer", str(reference.state_dag.values["final_answer"].normalized_value),
                                              h["detector_input"]["final_answer"]))
        controls = {s["pair_occurrence_id"]: s for s in n["control_spans"]}
        span_rows = [[s["step_index"] or "final_answer", s["node_id"], s["causal_role"], s["text"],
                      controls[s["pair_occurrence_id"]]["text"], str(s["span"]), str(s["serialized_span"]),
                      s["same_char_length"], s["pair_occurrence_id"]] for s in h["hallucination_spans"]]
        return (
            f"✅ 已读取 `{h['origin_id']}` · variant {h['variant_index']} · {h['edit_count']} 个 root 修改；只读回放，未生成新数据。",
            _plain({"raw_record": source.raw_record, "process_record": source.process_record, "formal_template": source.formal_template}),
            _dag_html(reference), _reference_node_rows(reference), _parsed_formal_rows(reference),
            plan_rows, _plain({"origin_id": h["origin_id"], "derived_seed": h["derived_seed"], "edit_count": h["edit_count"],
                               "mutation_events": events}),
            _dag_html(reference, graph=graph, changed_nodes=changed_ids), changes,
            _plain({"replay_note": "按保存的事件还原节点值；没有重新运行规划、传播或 edge audit。",
                    "node_values": {key: value.normalized_value for key, value in graph.values.items()},
                    "propagation_events": h["propagation_events"], "edge_audit": h["edge_audit"]}),
            status, _source_molecule_html(reference), "".join(original_cards), pair_comparison(h, n), span_rows,
            _plain({"historical_H": provenance, "historical_N": n["text_realization"], "viewer_poe_request_count": 0,
                    "pair_alignment": h["pair_alignment"], "demo_token_coverage": coverage}), _plain(h), _plain(n),
        )


def _navigate_pair(pair_ids, pair_id, offset=0):
    """Use dropdown order; stay at the boundary rather than wrapping around."""
    index = pair_ids.index(pair_id) if pair_id is not None else 0
    index = max(0, min(len(pair_ids) - 1, index + offset))
    return pair_ids[index], index > 0, index < len(pair_ids) - 1, f"第 {index + 1} / {len(pair_ids)} 题"


def build_demo(path=DEFAULT_RECORDS, dataset_root=DEFAULT_DATASET_ROOT):
    viewer = SavedDemo(path, dataset_root)
    pair_ids = tuple(viewer.pairs)
    first = pair_ids[0]
    initial = viewer.outputs(first)
    counts = Counter(p["H"]["subtask"] for p in viewer.pairs.values())
    choices = [(f"{p['H']['origin_id']} · v{p['H']['variant_index']} · roots={p['H']['edit_count']}", key)
               for key, p in viewer.pairs.items()]
    with gr.Blocks(title="MolHalluLens · 已生成数据只读回放") as app:
        gr.Markdown(f"# MolHalluLens\n已加载 **{len(viewer.references)} 题 / {len(viewer.pairs)} 对 H/N / "
                    f"{2 * len(viewer.pairs)} 条记录**：{dict(counts)}。")
        gr.Markdown("仅展示已保存的数据；无需 POE_API_KEY、不调用 Poe、不修改 JSONL。"
                    "A/B 来自对应原题；C/D 来自保存的修改事件；E 使用原样保存的文本与标签。"
                    "首次加载参考 tokenizer 可能下载公开词表，不上传数据。")
        with gr.Row():
            previous = gr.Button("← 上一个", interactive=False, scale=0)
            selection = gr.Dropdown(choices=choices, value=first, label="成功的 Origin / Pair", interactive=True, scale=5)
            following = gr.Button("下一个 →", interactive=len(pair_ids) > 1, scale=0)
        position = gr.Markdown(f"第 1 / {len(pair_ids)} 题")
        gr.Markdown("## 原始 vs H · 文字差异")
        gr.Markdown("差异颜色不等于训练标签；精确标签见下方 E 页的 H/N 对比与表格。")
        original_comparison = gr.HTML(initial[12])
        outputs = [gr.Markdown(initial[0])]
        with gr.Tabs():
            with gr.Tab("A · 原始数据"):
                outputs.append(gr.JSON(initial[1], label="原题 raw / process evaluation / FORMAL template"))
            with gr.Tab("B · Reference DAG"):
                outputs.append(gr.HTML(initial[2]))
                outputs.append(gr.Dataframe(initial[3], headers=["node", "step", "role", "type", "mutable", "visibility", "value", "provenance"], interactive=False, wrap=True))
                outputs.append(gr.Dataframe(initial[4], headers=["step", "name", "source field", "node", "parsed value"], interactive=False, wrap=True))
            with gr.Tab("C · 保存的修改 Plan"):
                gr.Markdown("读取当时的 mutation_events；不根据当前配置重新采样。")
                outputs.append(gr.Dataframe(initial[5], headers=["target", "nodes", "category", "operator", "before", "after", "magnitude", "similarity"], interactive=False, wrap=True))
                outputs.append(gr.JSON(initial[6], label="保存的根修改事件与 seed"))
            with gr.Tab("D · 修改前后"):
                gr.Markdown("按保存的 root / propagation events 还原；绿色为修改节点，edge audit 为历史结果。")
                outputs.append(gr.HTML(initial[7]))
                outputs.append(gr.Dataframe(initial[8], headers=["causal role", "root target", "node", "type", "before", "after", "operator / rule"], interactive=False, wrap=True))
                outputs.append(gr.JSON(initial[9], label="还原节点值与历史传播 / 审计"))
            with gr.Tab("E · 文本对比 + Span"):
                outputs.append(gr.Markdown(initial[10]))
                outputs.append(gr.HTML(initial[11]))
                # Keep callback output order while displaying this component
                # above the stage tabs, directly below the question counter.
                outputs.append(original_comparison)
                gr.Markdown("### N vs H · 精确训练标签")
                outputs.append(gr.HTML(initial[13]))
                outputs.append(gr.Dataframe(initial[14], headers=["step", "node", "causal role", "H text", "N truth", "component span", "serialized span", "same length", "pair occurrence"], interactive=False, wrap=True))
                with gr.Accordion("历史模型、配对方式与 token 统计", open=False):
                    outputs.append(gr.JSON(initial[15]))
                with gr.Accordion("完整已保存 H / N（只读）", open=False):
                    outputs.append(gr.JSON(initial[16], label="H"))
                    outputs.append(gr.JSON(initial[17], label="N"))

        def load_saved(pair_id):
            try:
                _, has_previous, has_next, label = _navigate_pair(pair_ids, pair_id)
                return (*viewer.outputs(pair_id), gr.update(interactive=has_previous),
                        gr.update(interactive=has_next), label)
            except Exception as error:
                # Clear the previous selection's details instead of showing stale data.
                return (f"❌ 加载失败：{error}", None, "", [], [], [], None, "", [], None,
                        "加载失败，没有展示上一题的文本。", "", "", "", [], None, None, None,
                        gr.update(interactive=False), gr.update(interactive=False), "请选择有效题目")

        def navigate(pair_id, offset):
            target, has_previous, has_next, label = _navigate_pair(pair_ids, pair_id, offset)
            return target, gr.update(interactive=has_previous), gr.update(interactive=has_next), label

        # Changing the dropdown programmatically triggers exactly one .change
        # render. Navigation itself never regenerates or rewrites any record.
        previous.click(lambda pair_id: navigate(pair_id, -1), selection,
                       [selection, previous, following, position], api_name="previous_saved_pair", queue=False)
        following.click(lambda pair_id: navigate(pair_id, 1), selection,
                        [selection, previous, following, position], api_name="next_saved_pair", queue=False)
        display_outputs = [*outputs, previous, following, position]
        selection.change(load_saved, selection, display_outputs, api_name="select_saved_pair", concurrency_limit=1,
                         concurrency_id="saved_pair_render",
                         trigger_mode="always_last")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    build_demo(args.records).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
        show_error=True, css=CSS, share=False,
    )


if __name__ == "__main__":
    main()
