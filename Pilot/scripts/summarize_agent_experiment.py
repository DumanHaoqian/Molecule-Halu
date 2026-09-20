#!/usr/bin/env python3
"""Render a source-backed Markdown readout of the isolated agent experiment."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_GENERATION = ROOT / "Experiments/agent_corruption_v13/development18"
MODEL_DIRS = {"ChemDFM-R-14B": "chemdfm_r14b", "Chem-R-8B": "chem_r8b"}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def percent(value) -> str:
    return "—" if value is None else f"{value:.2%}"


def number(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def interval(value) -> str:
    return "—" if value is None else f"[{percent(value[0])}, {percent(value[1])}]"


def paired_stats(summary: dict, comparison: str, metric: str):
    current = summary.get("causal_comparisons", {}).get(comparison, {}).get(metric)
    if current is not None:
        return current
    if metric != "primary_match":
        return None
    # Legacy summaries contain exact contingency counts for primary accuracy.
    key, reverse = {"H_vs_empty": ("B_vs_A", False), "H_vs_N": ("C_vs_B", True)}[comparison]
    counts = summary.get("paired_primary_comparisons", {}).get(key)
    if counts is None:
        return None
    from molhallulens.modules.agent_generation.metrics import paired_comparison
    both_right, both_wrong = counts["both_correct"], counts["both_wrong"]
    harmful = counts["lhs_only_correct" if reverse else "rhs_only_correct"]
    rescue = counts["rhs_only_correct" if reverse else "lhs_only_correct"]
    if both_right + both_wrong + harmful + rescue != counts["n"]:
        raise ValueError(f"Inconsistent paired contingency counts: {key}")
    baseline = [True] * both_right + [False] * both_wrong + [True] * harmful + [False] * rescue
    treatment = [True] * both_right + [False] * both_wrong + [False] * harmful + [True] * rescue
    return paired_comparison(baseline, treatment, baseline_label="C" if reverse else "A", treatment_label="B")


def count_summary(values: list[int]) -> str:
    if not values:
        return "无记录"
    return f"总计 {sum(values)}；均值 {statistics.mean(values):.2f}；范围 {min(values)}–{max(values)}；覆盖 {len(values)} 个样本"


def review_status_lines(accepted_rows: list[dict]) -> list[str]:
    reviews = [row.get("review_status") if isinstance(row.get("review_status"), dict) else {} for row in accepted_rows]
    present = sum(isinstance(row.get("review_status"), dict) for row in accepted_rows)
    lines = ["", f"模型审查计数以 results 内 {len(accepted_rows)} 条已接受记录为分母；其中 {present} 条有 review_status 对象。",
             "模型意见只作诊断，不替代程序检查，也不因样本已接受而将争议改记为通过。", "",
             "| 模型审查 | 一致 agreed | 争议 disputed | 缺失 / 其他 |",
             "| --- | ---: | ---: | ---: |"]
    for key, label in (("reference_status", "参考审查"), ("conformance_status", "构造合规审查")):
        agreed = sum(review.get(key) == "agreed" for review in reviews)
        disputed = sum(review.get(key) == "disputed" for review in reviews)
        lines.append(f"| {label} | {agreed} | {disputed} | {len(reviews) - agreed - disputed} |")
    consensus = sum(review.get("agent_consensus") is True for review in reviews)
    nonconsensus = sum(review.get("agent_consensus") is False for review in reviews)
    lines += ["", f"参考与构造合规两项同时一致：{consensus}；未同时一致：{nonconsensus}；缺失/其他：{len(reviews) - consensus - nonconsensus}。此字段不包含盲审。", "",
              "| 盲审观察 | true：模型报疑 | false：模型未报疑 | 缺失 / 其他 |",
              "| --- | ---: | ---: | ---: |"]
    flagged = sum(review.get("blind_leakage_flag") is True for review in reviews)
    unflagged = sum(review.get("blind_leakage_flag") is False for review in reviews)
    lines += [f"| 盲审泄漏标记 | {flagged} | {unflagged} | {len(reviews) - flagged - unflagged} |", "",
              "false 仅表示该次模型未报疑，不证明无泄漏；true 也不自动等同于已确认泄漏。缺失、非布尔值及争议保留为待解释诊断，不能自动消除。"]
    for key, expected, label in (("reference_status", "disputed", "参考争议"),
                                 ("conformance_status", "disputed", "构造合规争议"),
                                 ("blind_leakage_flag", True, "盲审泄漏报疑")):
        origins = [str(row.get("origin_id", "未知")) for row, review in zip(accepted_rows, reviews)
                   if (review.get(key) is True if expected is True else review.get(key) == expected)]
        if origins:
            lines += ["", f"{label} origin：" + "、".join(f"`{origin}`" for origin in origins)]
    return lines


def severity_lines(generation: Path, accepted_rows: list[dict], protocol: str | None = None) -> list[str]:
    """Describe only selected accepted plans, preserving missing-value coverage."""
    plans = []
    for row in accepted_rows:
        record = read_json(generation / "origins" / str(row["origin_id"]) / "accepted.json")
        if isinstance(record, dict) and isinstance(record.get("plan"), dict):
            plans.append(record["plan"])
    severities = [plan["severity"] for plan in plans if isinstance(plan.get("severity"), dict)]
    denominator = len(accepted_rows)

    def fraction_summary(values, *, as_percent=False):
        values = [value for value in values if type(value) in (int, float)
                  and math.isfinite(value) and 0 <= value <= 1]
        if not values:
            return f"无记录；覆盖 0/{denominator}"
        fmt = percent if as_percent else number
        return (f"均值 {fmt(statistics.mean(values))}；中位数 {fmt(statistics.median(values))}；"
                f"范围 {fmt(min(values))}–{fmt(max(values))}；覆盖 {len(values)}/{denominator}")

    title = "### 已选源状态干预幅度" if protocol == "agent_source_state_v1" else "### 已选结构干预幅度"
    lines = ["", title, "",
             f"accepted.plan 记录覆盖 {len(plans)}/{denominator}；severity 记录覆盖 {len(severities)}/{denominator}。只统计 summary 中已接受 origin 的最终计划，不汇总候选池或被拒绝样本。",
             "缺失或非法数值不补为 0，各指标单独显示有效覆盖；每题等权。", "",
             "| 幅度指标 | 统计 |", "| --- | --- |",
             "| H−GT 完整产物 Tanimoto | " + fraction_summary([s.get("product_tanimoto") for s in severities]) + " |",
             "| 原始分子重原子编辑覆盖率 | " + fraction_summary([s.get("source_footprint_ratio") for s in severities], as_percent=True) + " |"]
    root_lists = [plan["roots"] for plan in plans if isinstance(plan.get("roots"), list)]
    root_types = [("structural", "structural 根"), ("source_state", "source_state 根"),
                  ("numeric_claim", "numeric_claim 根")]
    if protocol == "agent_task_intent_diagnostic_v1":
        root_types.append(("task_interpretation", "task_interpretation 根"))
    known_root_types = {kind for kind, _ in root_types}
    for kind, label in [*root_types, (None, "其他 / 未标类型的根")]:
        counts = []
        for roots in root_lists:
            kinds = [root.get("type") if isinstance(root, dict) else None for root in roots]
            counts.append(sum(value == kind if kind else value not in known_root_types for value in kinds))
        description = (f"总计 {sum(counts)}；均值 {statistics.mean(counts):.2f}；范围 {min(counts)}–{max(counts)}；覆盖 {len(counts)}/{denominator}"
                       if counts else f"无记录；覆盖 0/{denominator}")
        lines.append(f"| {label} | {description} |")
    for name in MODEL_DIRS:
        ratios = [s.get("text", {}).get("tokenizers", {}).get(name, {}).get("error_token_fraction") for s in severities]
        lines.append(f"| {name} 独立 H 错误 token 比例 | {fraction_summary(ratios, as_percent=True)} |")
    lines += ["", "Tanimoto 比较已执行 H 产物与程序验证等价于 GT 的参考产物：完整分子、全部组分、去映射、无手性的 Morgan 指纹（半径 2、2048 bits）；数值越低表示指纹差异越大，不表示目标模型错误率。",
              "编辑覆盖率的分母是原始分子的重原子数；分子保留、原子属性、原有原子间键或新增组分连接发生变化的原始重原子计入分子。该值不是 MCS，也不是按根错误数加权的分数；CIP 变化可能来自配体优先级变化。",
              "根类型直接读取 plan.roots；不根据版本名称把未知类型补记为 structural。独立 H token 比例仅在 severity.text.tokenizers 保存该比例时统计，不与完整提示词 token 比例混用。",
              f"来源：`{generation / 'origins'}/<origin>/accepted.json` 的 `plan.severity` 与 `plan.roots`。"]
    if protocol == "agent_source_state_v1":
        lines += ["", "源状态协议的产物幅度以真实 S 上的联合执行与干净参考执行比较；感知源分子 S′ 上的分阶段执行另存并核对同一产物。编辑覆盖率分母仍是真实 S 的重原子，不把 S′ 当成原始输入。",
                  "source_state 根定位整条源图重建声明；整个错误 SMILES 值的语义标签不意味着其中每个原子或字符都错误，也不把长分子串当成多个独立根。"]
    return lines


def generation_lines(generation: Path, summary: dict | None, protocol: str | None = None) -> list[str]:
    if summary is None:
        return ["## 数据生成", "", "尚无生成汇总。", ""]
    planned, processed, accepted, rejected = (summary.get(key) for key in ("planned", "processed", "accepted", "rejected"))
    named_binding = protocol in {"agent_named_binding_diagnostic_v1", "agent_anchored_connections_diagnostic_v1", "agent_task_intent_diagnostic_v1"}
    diagnostic = protocol in {"agent_structural_ablation_v1", "agent_edit_order_diagnostic_v1", "agent_named_binding_diagnostic_v1", "agent_anchored_connections_diagnostic_v1", "agent_task_intent_diagnostic_v1"}
    if named_binding:
        outside = summary.get("outside_scope", "未提供")
        lines = ["## 配对诊断范围", "",
                 f"计划 {planned} 个 origin；已处理 {processed}；本诊断纳入 {accepted}/{planned}；诊断范围外 {outside}。",
                 "范围外记录使用 rejected 状态仅为文件协议兼容，不是化学失败，也不是目标模型行为失败；不把范围纳入比例称为生产接受率。",
                 f"本诊断生产验收计数：{summary.get('production_accepted', '未提供')}；四个固定样本只用于机制原型。",
                 "密度仅描述，不根据根/节点/token 阈值筛除已选样本或添加额外数值错误。", "",
                 "| 未达到参考阈值的项目 | origin 数 |", "| --- | ---: |"]
        violations = summary.get("density_violation_counts")
        for key, label in (("min_roots", "最少根错误数"), ("min_nodes", "最少错误节点数"), ("min_tokens", "最少错误 token 数")):
            count = violations.get(key, 0) if isinstance(violations, dict) else "未提供"
            lines.append(f"| {label} | {count} |")
        density_origins = summary.get("density_violation_origins", [])
        if density_origins:
            lines += ["", "密度例外 origin：" + "、".join(f"`{origin}`" for origin in density_origins) + "；全部保留，没有据此筛例。"]
        calls = summary.get("actual_api_calls")
        if type(calls) is int and calls == 0:
            reviewed_scope = {"agent_anchored_connections_diagnostic_v1": "新增连接块",
                              "agent_task_intent_diagnostic_v1": "任务解读文本"}.get(protocol, "native 新文本")
            lines += ["", f"本诊断新增 API 调用数为 0，未进行新的 LLM 审查。原 Agent 意见属于历史记录，不是对 {reviewed_scope}的审查通过。"]
        else:
            lines += ["", f"本诊断新增 API 调用数：{calls if type(calls) is int else '未提供'}；不由调用次数或 accepted 状态推断模型审查通过。历史审查与新审查必须分别记录。"]
    elif diagnostic:
        metadata = read_json(generation / "manifest.json") or {}
        lines = ["## 配对诊断覆盖", "",
                 f"预定 {planned} 个 origin；已处理 {processed}；纳入配对诊断 {accepted}；未纳入 {rejected}。"]
        if planned and accepted is not None:
            lines.append(f"配对纳入覆盖：{accepted}/{planned}（{accepted / planned:.2%}）；这是固定父集合的保留率，不是生产验收率。")
        lines += [f"源批次计划 {metadata.get('source_planned', '未提供')}；程序接受 {metadata.get('source_accepted', '未提供')}；拒绝 {metadata.get('source_rejected', '未提供')}。",
                  f"本消融生产验收计数：{summary.get('production_accepted', '未提供')}；本协议不授予新的生产质量通过状态。", "",
                  "| 未达到父版本阈值的项目 | origin 数 |", "| --- | ---: |"]
        violations = summary.get("parent_threshold_violation_counts", {})
        missing_violation_count = "未提供"
        if protocol == "agent_edit_order_diagnostic_v1" and "density_violation_counts" in summary:
            violations = summary["density_violation_counts"]
            missing_violation_count = 0
        for key, label in (("min_roots", "最少根错误数"), ("min_nodes", "最少错误节点数"), ("min_tokens", "最少错误 token 数")):
            lines.append(f"| {label} | {violations.get(key, missing_violation_count)} |")
        density_origins = summary.get("density_violation_origins", [])
        if density_origins:
            lines += ["", "错误节点/token 密度未达父版本标准的 origin：" + "、".join(f"`{origin}`" for origin in density_origins) + "。这些样本仍保留，未据此筛选或补采。"]
        lines += ["", "未进行新的 LLM 审查；父版本的模型意见仅作为历史记录保存，不能作为本消融的新审查通过或否决。"]
    else:
        lines = ["## 数据生成", "",
                 f"计划 {planned} 个 origin；已处理 {processed}；接受 {accepted}；拒绝 {rejected}。"]
        if planned and accepted is not None:
            lines.append(f"接受数 / 计划数：{accepted}/{planned}（{accepted / planned:.2%}）；未处理样本保留在计划分母。")
        if processed and accepted is not None:
            lines.append(f"已处理样本接受率：{accepted}/{processed}（{accepted / processed:.2%}）。")
    results = summary.get("results", [])
    accepted_rows = [row for row in results if row.get("status") == "accepted"]
    if protocol in {"agent_corruption_v11", "agent_corruption_v12", "agent_source_state_v1"}:
        lines += review_status_lines(accepted_rows)
    count_scope = ("以下计数来自四个固定诊断样本的程序重算记录：" if named_binding else
                   "以下计数来自已纳入消融样本的程序重算记录：" if diagnostic else "以下计数仅来自已接受样本的生成审计记录：")
    lines += ["", count_scope, "",
              "| 项目 | 统计 |", "| --- | --- |"]
    for key, label in (("roots", "根错误"), ("distinct_wrong_nodes", "每题不同错误语义节点"), ("error_spans", "错误字符跨度")):
        values = [row["error_counts"][key] for row in accepted_rows if key in row.get("error_counts", {})]
        lines.append(f"| {label} | {count_summary(values)} |")
    for name in MODEL_DIRS:
        values = [row["error_counts"]["tokens"][name] for row in accepted_rows
                  if name in row.get("error_counts", {}).get("tokens", {})]
        lines.append(f"| {name} 生成阶段独立 CoT 错误 token | {count_summary(values)} |")
    lines += ["", "生成阶段 token 数对应独立 CoT 分词；评估中的完整提示词 token 数另列，二者的边界 token 可能不同。"]
    if protocol in {"agent_corruption_v12", "agent_source_state_v1", "agent_edit_order_diagnostic_v1", "agent_named_binding_diagnostic_v1", "agent_anchored_connections_diagnostic_v1", "agent_task_intent_diagnostic_v1"}:
        lines += severity_lines(generation, accepted_rows, protocol)
    reasons = collections.Counter(row.get("reason", "未提供原因") for row in results if row.get("status") == "rejected")
    if reasons:
        lines += ["", "未纳入诊断的范围记录：" if named_binding else "拒绝原因："]
        for reason, count in sorted(reasons.items()):
            lines.append(f"- {count} 个：{str(reason).replace(chr(10), ' ')}")
    lines += ["", f"来源：`{generation / 'summary.json'}`", ""]
    return lines


def model_lines(name: str, directory: Path) -> list[str]:
    summary = read_json(directory / "summary.json")
    manifest = read_json(directory / "manifest.json")
    lines = [f"## {name}", ""]
    if summary is None:
        lines += ["答案评估尚无汇总。", ""]
    else:
        status = "已完成" if summary.get("complete") else "部分结果"
        lines += [f"答案评估：{status}；完成 {summary.get('n_completed', '—')}/{summary.get('n_expected', '—')} 条请求。", "",
                  "| 组别 | n | 主片段 Accuracy | 完整分子精确率 | 分子相似度 | 有效 / 缺失 / 截断 |",
                  "| :---: | ---: | ---: | ---: | ---: | ---: |"]
        for group in "ABCDE":
            row = summary.get("groups", {}).get(group)
            if row:
                validity = " / ".join(str(row.get(key, "—")) for key in ("valid_smiles", "missing_answers", "truncated"))
                lines.append(f"| {group} | {row['n']} | {percent(row.get('primary_accuracy'))} | {percent(row.get('normalized_exact_accuracy'))} | {number(row.get('mean_fts'))} | {validity} |")
            else:
                lines.append(f"| {group} | — | 未评估 | 未评估 | 未评估 | — |")
        lines += ["", "有效为可解析分子数；缺失为未提取出答案数；截断为达到生成长度上限数，后两项可能重叠。",
                  "配对差值为 B 减去对照；转错率的分母仅包括对照回答正确的样本。", "",
                  "| 配对 | 指标 | 配对 n | Accuracy 差值（百分点） | 对照正确→B 错误 | 转错率 | Wilson 95% 区间 | McNemar 精确双侧 p |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for comparison, label in (("H_vs_empty", "B−A"), ("H_vs_N", "B−C")):
            for metric, metric_label in (("primary_match", "主片段"), ("normalized_exact_match", "完整分子")):
                stats = paired_stats(summary, comparison, metric)
                if stats is None:
                    continue
                flip = stats["flip_to_wrong"]
                difference = stats.get("accuracy_difference")
                difference_text = "—" if difference is None else f"{100 * difference:+.2f}"
                p_value = stats.get("mcnemar_exact_p")
                p_text = "—" if p_value is None else f"{p_value:.6g}"
                lines.append(f"| {label} | {metric_label} | {stats['n']} | {difference_text} | {flip['count']}/{flip['denominator']} | {percent(flip['rate'])} | {interval(flip['ci95'])} | {p_text} |")
        if summary.get("repeated_origins"):
            lines += ["", "同一 origin 存在重复变体；以上按 pair 的统计检验和区间仅作描述，不将变体视为独立样本。"]
        lines += ["", f"来源：`{directory / 'summary.json'}`", ""]
    if manifest:
        protocol = manifest.get("agent_evaluation")
        if protocol:
            lines += [f"实际评估协议：`{protocol['version']}`，placement=`{protocol['placement']}`，groups=`{protocol['groups']}`。", ""]
        else:
            lines += ["此来源为旧版评估汇总；不将其提示词解释为 新 prefix 结果。", ""]
    labels = read_jsonl(directory / "prompt_token_labels.jsonl")
    h_labels = [row for row in labels if row.get("group") == "B"]
    if h_labels:
        values = [row["error_token_count"] for row in h_labels if row.get("status") == "available"]
        lines += [f"实际完整 H 提示词的错误 token：{count_summary(values)}；H 标注覆盖 {len(values)}/{len(h_labels)}。", ""]
    probe = read_json(directory / "probe_summary.json")
    if probe:
        status = "已完成" if probe.get("complete") else "部分结果"
        lines += [f"GT teacher-forcing：{status}，{probe['n_completed']}/{probe['n_expected']} 条请求。", "",
                  "| 条件差 | 配对 n | 每 GT token 的平均 logprob 差（nats） | GT 总 logprob 的平均差（nats） |",
                  "| --- | ---: | ---: | ---: |"]
        for key in ("H_minus_N", "H_minus_empty", "N_minus_empty", "irrelevant_minus_empty"):
            value = probe.get("contrasts", {}).get(key)
            if value:
                lines.append(f"| {key} | {value['n']} | {number(value.get('mean_delta_mean_logprob'), 6)} | {number(value.get('mean_delta_sum_logprob'), 6)} |")
        lines += ["", "每个条件续接相同的 GT SMILES token 序列；统计范围不包括结束标签、EOS 或 D 组。负差值表示该 GT token 事件的概率下降。",
                  f"来源：`{directory / 'probe_summary.json'}`", ""]
    else:
        lines += ["GT teacher-forcing 尚无结果。", ""]
    entropy = read_json(directory / "entropy_summary.json")
    if entropy:
        sampling = entropy["sampling"]
        lines += [f"采样熵：每题 n={sampling['n']}，temperature={sampling['temperature']}，top_p={sampling['top_p']}。", "",
                  "| 组别 | 已评估问题数 | 平均 canonical-SMILES 熵（bits） | 有效分子覆盖率 |",
                  "| --- | ---: | ---: | ---: |"]
        for group, row in sorted(entropy.get("groups", {}).items()):
            lines.append(f"| {group} | {row['n']} | {number(row['mean_entropy_bits'])} | {percent(row['mean_valid_coverage'])} |")
        lines += ["", "无效和缺失输出分别成类且保留在分母。该熵只由采样答案的分子类别计算，不使用 GT 正误；有限采样下的低熵不代表答案正确。",
                  f"来源：`{directory / 'entropy_summary.json'}`", ""]
    return lines


def summarize(args) -> bool:
    generation = read_json(args.generation_dir / "summary.json")
    has_evaluation = any(any((args.evaluation_dir / folder / filename).exists()
                             for filename in ("summary.json", "probe_summary.json", "entropy_summary.json"))
                         for folder in MODEL_DIRS.values())
    if generation is None and not has_evaluation:
        print("No source results are available; report was not created or replaced.", flush=True)
        return False
    manifest = read_json(args.generation_dir / "manifest.json")
    protocol = manifest.get("protocol") if manifest else None
    lines = ["# Agent 化学推理干预实验", "",
             f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "A：空推理直接回答；B：本题 H；C：本题 N；D：模型自行推理；E：借用其他题目的 N。",
             "目标协议采用 prefix placement：A/B/C/E 的 system 与原题相同，B/C/E 的推理放在 assistant 前缀；D 保留自行推理设置。", ""]
    if protocol == "agent_structural_ablation_v1":
        if manifest.get("diagnostic_only") is not True:
            raise ValueError("Structural ablation manifest must explicitly declare diagnostic_only=true")
        lines += ["**仅限配对诊断：本消融的 accepted 表示纳入固定配对比较，纳入不等于生产质量验收通过。**",
                  "生成协议：`agent_structural_ablation_v1`。保留父版本全部 15 个已接受 origin、相同 N 及错误结构编辑计划，仅移除数值类根错误并重新推导文本与标注；不按目标模型结果筛选样本。",
                  "部分样本不再达到父版本的根错误数、错误节点数或错误 token 数要求，仍按固定配对集合保留并明确报告。",
                  "未进行新的 LLM 审查；父版本的参考、构造合规与盲审意见是历史记录，不是对新文本的审查结论。",
                  "原始 question/instruction 保持不变，ABCDE 的提示词规则、采样参数与引擎配置沿用父实验。", ""]
    elif protocol == "agent_edit_order_diagnostic_v1":
        if manifest.get("diagnostic_only") is not True or manifest.get("order") not in {"account_last", "edit_last"}:
            raise ValueError("Edit-order manifest requires diagnostic_only=true and a supported order")
        lines += ["**仅限配对诊断：accepted 表示保留父版本样本进行次序对照，不表示新的生产质量验收通过。**",
                  f"生成协议：`agent_edit_order_diagnostic_v1`；当前次序：`{manifest['order']}`。",
                  "固定 v12 的全部 14 个已接受 origin、结构根错误、编辑计划与产物；原批次 4 个拒绝仍保留在 18 个计划样本中。",
                  "两个条件均无局部产物窗口；account_last 以原子/环计数结束，edit_last 以具体编辑结束。双方化学声明相同，仅块次序与步骤编号不同。",
                  "N/H 同时采用各自条件的次序；原问题、instruction、GT、模型 system 和解码参数不变。",
                  "这是同一批 origin 的重复测量，两个条件或重复的 A 不能合并当作独立新样本；跨次序行为差异需按 origin 配对比较。",
                  "未进行新的 LLM 审查；原 Agent 意见仅作历史证据。文本经程序重算和检查，保留 v12 标签的参考相对语义定义。",
                  "本诊断不按目标模型输出筛题，不自动授权留出集扩展，也不单凭次序效应证明内部推理机制。", ""]
    elif protocol == "agent_named_binding_diagnostic_v1":
        if manifest.get("diagnostic_only") is not True or manifest.get("style") not in {"ledger", "native"}:
            raise ValueError("Named-binding manifest requires diagnostic_only=true and a supported style")
        lines += ["**仅限四题配对诊断：accepted 表示纳入固定表达对照，不表示新的生产质量验收或一般成功。**",
                  f"生成协议：`agent_named_binding_diagnostic_v1`；当前样式：`{manifest['style']}`。",
                  "固定 add0098、add0172、add0193、add0194 四题的正确源锚点和参考操作，仅将入射片段替换为 C(=O)c1ccc(C(F)(F)F)cc1；两个样式使用相同错误计划和执行产物。",
                  "native 保留原始名称及源官能团到羰基的连接语言，故意将原始名称绑定到错误片段图；名称本身仍引用题目符号，不单独标成改错值。根类型沿用 structural，绑定关系另由 binding_evidence 解释，不虚构新根类型。",
                  "ledger 使用无局部产物窗口的类型化账本；native N 是原始 N 去除完整产物/答案后的投影，native H 只应用已记录、独立检查的依赖跨度补丁。两者不是同一篇文本的纯措辞替换。",
                  "只有条件于错误绑定的执行与数值传播保持一致；名称与错误图之间的关系本身是化学错误，不能把整段称为化学上无矛盾。未改变值不加错误标签，重复提及不增加根数。",
                  "独立氢封端片段 C8H5F3O 与实际附着酰基 C8H4F3O 属于不同计数范围，不能混用；原题、instruction、源分子、GT 和 ABCDE 设置保持不变。",
                  "两个样式比较整套表达差异，不能将结果单独归因于名称、纯措辞或单一命名关系；同四个 origin 的重复测量不构成八个独立问题。",
                  "不按目标模型输出筛题或选择样式赢家；密度只报告不筛例。四题诊断不能自动授权 heldout 扩展，亦不替代一般 Agent 构造协议的验证。", ""]
    elif protocol == "agent_anchored_connections_diagnostic_v1":
        if manifest.get("diagnostic_only") is not True or manifest.get("style") not in {"native", "native_with_connections"}:
            raise ValueError("Anchored-connection manifest requires diagnostic_only=true and a supported style")
        lines += ["**仅限四题配对诊断：accepted 表示纳入固定连接表示对照，不表示生产质量验收或一般成功。**",
                  f"生成协议：`agent_anchored_connections_diagnostic_v1`；当前样式：`{manifest['style']}`。",
                  "native 逐字保留 v15 native N/H，作为本轮重新运行的并行对照；native_with_connections 在同一 N/H 原文末尾对称追加各自的产物侧连接表示。",
                  "固定相同四个 origin、原题、instruction、源分子、GT、错误计划与执行产物；化学编辑幅度不变，只有显式部分产物信息及其表达格式不同。",
                  "连接图保留正确源锚点和全部入射片段，用开放端口 [*:k] 表示通向被省略的原源原子 k 的连接；不是氢封端，也不是将真实元素替换为 dummy。保留实原子的产物氢数、电荷、价态与键。",
                  "边界按固定编辑操作定义，不按目标结果挑选半径。补回保存的外部源上下文应恢复对应完整执行产物及已指定立体结构；不能由裁剪图推断完整产物的环数、分子式或全部外部立体信息。",
                  "原生 N/H 的既有注释保留；H 新增连接值是原 fragment 的 propagated 后果，仍为一个 fragment 根，不把正确锚点、端口或重复显示另算独立错误。整个图值的关系标签不表示每个原子都错。",
                  "完整答案字符串及去 dummy 后等于完整产物/完整组分的表示均须排除；这不等于没有答案信息，源分子与完整编辑操作本已支持重建产物。必须承认显式部分产物的信息与格式优势。",
                  "这仍是同四个 origin 的重复测量，不是八个独立问题。比较同条件 B−A/B−C、跨条件 B/C 与重跑 A/D；局部连接采纳若源映射不唯一或源图未保留，应记为不可判定，不能当作未采纳。",
                  "不按目标输出选题或挑选条件赢家；长度与密度如实报告，不填充数值错误或据此筛例。无新的模型审查通过声明，本诊断不能单独授权 heldout 扩展。", ""]
    elif protocol == "agent_task_intent_diagnostic_v1":
        if manifest.get("diagnostic_only") is not True or manifest.get("style") not in {"binding", "intent"}:
            raise ValueError("Task-intent manifest requires diagnostic_only=true and a supported style")
        lines += ["**仅限四题配对诊断：accepted 表示纳入固定任务解读对照，不表示生产质量验收或一般成功。**",
                  f"生成协议：`agent_task_intent_diagnostic_v1`；当前样式：`{manifest['style']}`。",
                  "binding 逐字保留 v16 native_with_connections 的 N/H；intent 的 N 逐字不变，H 仅将官能团名称及既有任务复述统一为冻结错误片段的实际名称。物理编辑计划与执行产物不变，源锚点、计数及连接图不变。",
                  "原始 instruction 不变，源分子、问题与 GT 不变；同模型跨样式的 A/C/D/E 完整请求必须相同，仅 B 的推理前缀变化。",
                  "每题仍为一个根，但语义本体不同：binding 的 plan.roots 声明 structural / fragment；intent 声明 task_interpretation / task_group，fragment 转为 task_group 的 propagated 后果。根统计直接读取该声明，不将物理计划相同误写为根字典相同。",
                  "本操作消除 H 内部名称与图的矛盾；CoT 对任务的错误解读与原问题的外部冲突仍然存在。它不是自然错误分布，也不证明整段化学叙述全局无矛盾。",
                  "继承的产物侧连接图仍是部分产物信息：开放端口 [*:k] 不是完整产物，完整答案字符串排除不等于没有答案信息；dummy 草稿复制与完整错误产物采纳应分开统计。",
                  "固定四个 origin 的两种表达是重复测量，不合并为八个独立问题。比较同条件 B−A/B−C、跨条件 B 以及不变的 A/C/D/E 重跑控制；图幅度不扩大，不按目标输出筛题或挑选条件赢家。",
                  "未进行新的 LLM 审查，历史 Agent 意见不能作为任务解读文本的新审查通过。密度如实报告且不筛例；本诊断不能单独授权 heldout 扩展。", ""]
    elif protocol == "agent_source_state_v1":
        lines += ["**源状态协议的接受仅表示通过已定义的程序构造检查，不表示所有 Agent 一致认可或模型提出的争议已经解决。**",
                  "生成协议：`agent_source_state_v1`；发布契约：`program_verified_source_state_v1`。N 独立展示真实源分子 S，H 展示错误重建的感知源分子 S′；两者都应用同一个参考编辑 I。原始 instruction、indexed SMILES、问题与评分 GT 保持不变。",
                  "干净执行 I(S) 必须等价于 GT；H 的 I(S′) 另与真实 S 上的联合执行核对。程序检查覆盖受保护编辑区域、感知源图可执行性、相同参考编辑、联合执行一致性、条件算术、语义/token 标签、密度和完整产物/控制信息排除。",
                  "参考审查、构造合规审查和盲审意见保留为非否决诊断；下面分别报告 agreed/disputed 和盲审标记，不由 accepted 状态推断审查通过。",
                  "N/H 使用相同模板；N 是带显式源表示的规范化参考，与 N_raw 及早期版本 N 都不保证逐字相同。完整 GT 与 H 产物只用于内部检查，不作为推理中的产品字符串。",
                  "候选选择只使用程序可行的源重建与 Agent 批准结果，再按执行产物指纹差异选择；不使用目标模型输出筛选候选或样本。实际阈值以冻结 manifest 为准。",
                  "该实验检验错误源状态的影响；完整源分子复制是可能机制和混杂因素。准确率下降不能单独证明模型执行了参考编辑或采用了错误推理，需要区分输出与 S、S′、I(S′) 和 GT 的匹配。它不是自然错误分布的估计，也不等同于局部片段或错误编辑计划干预。", ""]
    elif protocol in {"agent_corruption_v11", "agent_corruption_v12"}:
        version = protocol.rsplit("_", 1)[-1]
        lines += [f"**{version} 接受仅表示通过程序化构造检查，不表示所有 Agent 一致认可，也不表示模型提出的语义争议已解决。**",
                  "程序硬检查覆盖可执行化学编辑、确定性语义文本、错误标注及答案/控制信息泄漏规则；模型参考审查、构造合规审查和盲审意见另行保留，不作为该版本的发布否决条件。", "",
                  f"生成协议：`{protocol}`。N 使用冻结 benchmark 编辑及程序计算形成的规范化语义参考；程序核对参考编辑产物与 benchmark GT 图一致，不声称 N 与 N_raw 逐字一致。",
                  "原始 instruction 保持不变；评估 prepare 将 H/N 的 instruction 与原始 benchmark 逐条核对。", ""]
        if protocol == "agent_corruption_v12":
            lines += ["v12 构造策略：扩大结构替换、连接位置及移除集合的变化；不再加入数值 +1 根错误。Agent 从程序可行候选中批准至多 6 个，控制器按实际 H−GT 产物 Tanimoto 最小值选择一个 H，并以候选 ID 打破平局；不使用目标模型输出选择样本。",
                      "根数、错误节点数和两种 tokenizer 的 token 密度阈值以冻结生成配置为准；幅度统计是构造诊断，不保证 B<A 或 B<C。", ""]
    elif protocol in {"agent_corruption_v9", "agent_corruption_v10"}:
        lines += [f"生成协议：`{protocol}`。数据采用规范化语义参考：N 是依据化学语义规范化后的正确参考，不声称与原始 N_raw 逐字一致；H 从该冻结参考派生。",
                  "原始 instruction 保持不变；评估 prepare 将 H/N 的 instruction 与原始 benchmark 逐条核对。", ""]
    else:
        lines += [f"生成协议：`{protocol or '尚无 manifest'}`。N 语义规范化声明仅适用于确认的新 Agent 协议数据；旧版结果保持其原协议解释。", ""]
    lines += ["主指标为去原子映射后的主片段分子相等；另列完整分子精确率。分子相似度为完整分子 Morgan 指纹（半径 2、2048 bits）的 Tanimoto 均值，无效或缺失答案记 0。",
              "这些是固定样本上的实测结果，不预设或保证 B<A/B<C；配对 p 值未作多重比较校正。", ""]
    lines += generation_lines(args.generation_dir, generation, protocol)
    for name, folder in MODEL_DIRS.items():
        lines += model_lines(name, args.evaluation_dir / folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(args.output)
    print(f"Report updated: {args.output}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/agent_corruption_result.md")
    args = parser.parse_args()
    args.evaluation_dir = args.evaluation_dir or args.generation_dir / "evaluation"
    summarize(args)


if __name__ == "__main__":
    main()
