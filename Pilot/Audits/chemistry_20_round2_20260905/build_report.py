"""Round-two evidence report and inspectable executed plain-Python notebook.

Retains the prior audit's Markdown delivery: small exact-lookup evidence tables
are more useful here than a chart, which the portable report schema mandates.
The notebook uses a sequential Python cell runner, not an IPython kernel.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    evidence = json.loads((HERE / "evidence.json").read_text())
    review = json.loads((HERE / "review.json").read_text())
    records = {r["origin_id"]: r for r in evidence["records"]}
    parts = ["""# ChemCoTBench-V2：第二轮20题化学审计

## 结论：产物能重建，仍有4题正文认错化学结构或官能团

从上一轮未抽中的130题里，再分层随机抽20题：add 7、delete 7、substitute 6；与上一轮重叠0题。审读了106个原始step。

**20题全部通过246项结构/数值检查，并重建出参考产物；但4题正文存在明确的骨架命名或官能团分类错误，另1题输入未指定新片段的立体构型，GT却选定了两个手性中心。** 后者不表示GT分子本身不成立，而是题面不足以唯一推出它。其余15题未发现同等级明确问题，不代表全部token已被认证为真。

本轮是AI辅助化学审读与独立RDKit计算，不是独立人类专家金标准。本轮未修改原始数据、生产代码、第一轮审计结果，也未调用Poe。

## 本轮计数口径

| 结论 | 题数 / 20 | 对使用数据的影响 |
| --- | ---: | --- |
| 正文明确错误 | 4 | GT可以保留；错误正文不能作为无幻觉负标签 |
| 题面立体信息不足 | 1 | 需补充构型条件或放宽立体评分口径；不是判GT分子错误 |
| 未发现同等级明确问题 | 15 | 包含另列的1处低等级术语歧义；不是全面真值保证 |

这三类按origin计且互不重叠。额外的8条rxn_cls口径警告单独记录，可能与上表重叠，不可相加为“13题错误”。这只是本地子集的第二轮抽样，不是上游完整数据集的错误率估计，也不是质量随时间变化的趋势。
"""]
    for f in review["findings"]:
        parts.append(f"\n## {f['id']} · {f['origin_id']}\n\n")
        parts.append(f"涉及Step {', '.join(map(str,f['steps']))}；类型：{review['categories'][f['category']]}。\n\n")
        if f.get("instruction_quote"):
            parts.append("题目：\n\n> " + f["instruction_quote"] + "\n\n")
        parts.append("原始正文：\n\n> " + f["quote"] + "\n\n")
        parts.append(f["explanation"] + "\n\n建议：" + f["recommendation"] + "\n\n")
        if f["sources"]:
            parts.append("对照依据：" + "；".join(f"[{s['label']}]({s['url']})" for s in f["sources"]) + "。\n")
        parts.append("上述错误判断基于本地结构与对照结构/定义的比较；网页只提供化学对照，不对整条样本背书。\n" if f["sources"] else "这是基于本地输入缺少立体限定、输出却新增立体标记作出的判断；4个候选异构体保存在semantic_checks.json中，不表示它们在任意实验条件下都等概率生成。\n")
    parts.append("""
## 另外两类问题不混入“4题明确正文错误”

### Boc中心碳的术语歧义

delete_v2.0036的Step2写“quaternary C + 3 methyls”。原结构中心C2连接3个C和1个O：若按碳邻居数分类，不是季碳；某些NMR语境会把无氢碳称为quaternary。原文没有说明口径，建议直接写“central non-protonated carbon”。本轮不据此增加明确错误题数。

### rxn_cls不能未经确认当作本次编辑类型

以下8题的字段与当前净编辑类型不一致。未找到本地字段定义，可能是继承的原反应标签；原因尚未证实。风险是用它分层、训练或解释当前任务时会引入混淆，而不是直接证明这8题的推理都错。

| Origin | 原rxn_cls | 本题净编辑 |
| --- | --- | --- |
""")
    for row in review["metadata_notes"]["observations"]:
        parts.append(f"| {row['origin_id']} | {row['stored']} | {row['observed_edit']} |\n")
    parts.append("\n## 全部20题逐题结果\n\n数值表示源→产物。结构重建20/20匹配，保留已指定的立体信息；这不证明实验可行性或答案唯一性。语义结论来自AI阅读原题和完整step_text，不是从原数据的all_pass推断。完整原文见sample_review.txt，机器证据见evidence.json。\n\n| Origin | 重原子 | 环数 | 逐题化学核对 |\n| --- | --- | --- | --- |\n")
    for row in review["records"]:
        r = records[row["origin_id"]]
        a, b = r["source_computed"], r["answer_computed"]
        parts.append(f"| {row['origin_id']} | {a['heavy_atoms']}→{b['heavy_atoms']} | {a['rings']}→{b['rings']} | {row['review']} |\n")
    parts.append("""
## 两轮合起来说明了什么

| 指标 | 第一轮 | 第二轮 | 累计 |
| --- | ---: | ---: | ---: |
| 不重复题目 | 20 | 20 | 40 |
| 审读step | 106 | 106 | 212 |
| 结构/数值检查通过 | 246 | 246 | 492 |
| 参考产物重建匹配 | 20 | 20 | 40 |
| 已识别的明确正文错误origin | 1 | 4 | 5 |

第一轮另有1题模板未实例化、2题机理解释需限定；第二轮另发现1题立体信息不足。不同轮次检查逐步深化，不能据此宣称各类问题已被穷尽，或把第二轮的4/20与第一轮1/20解读为质量“变差”。

共同信号是：**当前机械核验更擅长检查计数与净图编辑，正文中“这是什么骨架、属于哪类官能团”的断言仍可能错误。** 第二轮还表明，重建出GT可以是因为FORMAL本身已经带入题目没给出的立体信息。因此，产物重建通过与题目充分、答案唯一是不同命题。

对H/N的影响：即使两侧非标注位置逐字节相同，原文共同携带的错误仍会污染N；而题目未指定的构型差异，不应未经定义就标成幻觉。配对控制风格与化学真值审核必须分别做。

## 方法与复现边界

- 审计日期2026-09-05；随机种子20260906只是种子，不表示日期。先排除第一轮20个ID，再按add→delete→substitute从排序后的43/43/44个剩余ID抽7/7/6；固定选题后不替换。
- 复用第一轮独立audit.py的RDKit检查函数，不调用项目builder、planner、propagation或validator。六份raw/process源文件SHA-256与上一轮一致，原数据all_pass不作为证据。
- 核对anchor map/元素、片段重原子、源/产物计数与delta、各处答案分子等价性，并按声明编辑枚举片段连接位点重建产物。该自动检查仍以原FORMAL编辑为输入；指令是否支持该编辑另行语义审读。
- 环计数采用当前RDKit CalcNumRings口径；立体比较先去掉atom-map标签，避免把标记编号误当化学差异。N原子间距离、参照骨架子结构匹配、四种立体候选均另存回验。
- verify_round2.py确认5条发现的原文引用、原始字段、源哈希、上一轮依赖哈希；故意注入错误计数、错误答案和错误anchor元素的3个测试均检出。
- 审计不会保证所有反应条件下的选择性、收率、产物唯一性；不验证每个化学名称的完整位次命名，AI审读仍有漏检可能。无真实Poe调用、无生产测试套件或生成过程改动。

在本目录、已有molhallulens环境下执行：

```bash
python audit_round2.py
python verify_round2.py
python build_report.py
```

audit.ipynb保存可复跑的Python单元和实际stdout。当前环境缺Jupyter内核包，采用普通Python共享命名空间顺序执行，不是nbclient/IPython内核验证，没有安装依赖。报告沿用第一轮Markdown证据表；可视化模板强制要求chart，本次不为小样本审计添加多余统计图。

## 下一步与仍待确认的问题

1. 先复核并修订这4题正文的骨架/官能团名称，不必因此删掉正确的题目和产物；本轮只审计，不直接修源数据。
2. 对substitute_v2.0064，确认上游是否有未进入本地instruction的构型限定。若没有，补充输入或明确评估是否忽略立体；不要把另一个同连接关系的异构体直接打成幻觉。
3. 确认rxn_cls的上游含义后再用于分层；在此之前按实际编辑或add/delete/substitute切分更可解释。
4. 当前累计40题，剩余110题未审；这40题也不是人类专家金标准。若要得出探针有效、固定format损害性能等研究结论，还需要目标模型自然作答的独立、盲审测试集和对照实验。
""")
    (HERE / "REPORT.md").write_text("".join(parts))

    cells = []
    def md(text):
        cells.append({"cell_type":"markdown", "id":f"cell-{len(cells)}", "metadata":{}, "source":text})
    def code(text):
        cells.append({"cell_type":"code", "id":f"cell-{len(cells)}", "metadata":{}, "source":text, "execution_count":None, "outputs":[]})
    md("# 第二轮20题原始化学审计\n\n分层抽样、不与第一轮重复；无Poe调用。AI语义审读不是人类金标准。用本目录及安装RDKit的Python运行；保存输出来自普通Python单元顺序执行器，不是Jupyter内核。")
    code("import json\nfrom audit_round2 import HERE, run\nfrom verify_round2 import verify\nevidence = run()\n")
    code("verification = verify()\n")
    md("## 全部20题：输入、计算结果与逐题审读\n\n原数据all_pass不作为真值证明；语义结论来自原始题目、结构和完整106步的AI审读。")
    code("review = json.loads((HERE / 'review.json').read_text())\nby_id = {r['origin_id']:r for r in review['records']}\nfor r in evidence['records']:\n    print(r['origin_id'], r['raw']['instruction'])\n    print('SOURCE', r['raw']['indexed_smiles'])\n    print('GT', r['raw']['gt_smiles'])\n    print('COUNTS', r['source_computed'], r['answer_computed'])\n    print('REVIEW', by_id[r['origin_id']])\n")
    md("## 具体发现与结构反例\n\n计数正确≠骨架认对；GT可重建≠题目唯一确定立体构型。")
    code("print(json.dumps(review['findings'], ensure_ascii=False, indent=2))\nprint((HERE / 'semantic_checks.json').read_text())\nprint(json.dumps(review['metadata_notes'], ensure_ascii=False, indent=2))\n")
    md("## 限制\n\n不估计上游总体错误率，不验证实验收率或反应选择性；对rxn_cls仅报告与本题编辑不符，继承字段的真实定义尚未确认。完整建议与来源链接见REPORT.md和review.json。")
    namespace, number, previous_dir = {}, 0, Path.cwd()
    try:
        os.chdir(HERE)
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            number += 1
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exec(compile(cell["source"], f"audit.ipynb/cell-{number}", "exec"), namespace)
            cell["execution_count"] = number
            cell["outputs"] = [{"output_type":"stream", "name":"stdout", "text":stdout.getvalue()}]
    finally:
        os.chdir(previous_dir)
    notebook = {"cells":cells, "nbformat":4, "nbformat_minor":5,
                "metadata":{"kernelspec":{"name":"python3", "language":"python", "display_name":"Python (RDKit required)"},
                            "language_info":{"name":"python"}, "execution_method":"Shared-namespace Python cell runner, not Jupyter/nbclient"}}
    (HERE / "audit.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n")
    print(f"REPORT.md saved; {number} companion notebook Python cells executed successfully.")


if __name__ == "__main__":
    build()
