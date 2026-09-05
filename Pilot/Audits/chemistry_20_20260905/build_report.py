"""Produce a source-backed report and executed companion notebook.

The notebook's plain Python cells are executed in order by this script because
the current environment has no Jupyter kernel packages. This is not nbclient
execution; no IPython-only syntax is used. Does not install dependencies.
"""
from collections import Counter
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
    counts = Counter(r["category"] for r in review["records"])
    title = "ChemCoTBench-V2：20题原始化学审计"
    intro = f"""# {title}

日期：2026-09-05。审核性质：AI辅助的化学审读 + 独立RDKit核算，**不是人类专家金标准**。

## 结论

这20题的目标产物均能由所述净图编辑重建，重原子、环数、差值及各处答案的一致性检查全部通过。但是，**正确的产物和FORMAL不能保证整段step_text正确**：找到1题明确的杂环命名错误、1题未实例化的自然语言模板，另有2题需要限定“图编辑”与“反应机制”的区别。剩余16题在本次范围内未发现明确问题，不代表每个token都已被证明为真。

不能据此说整个ChemCoTBench-V2不可用；也不能把原始正文直接视为无幻觉负样本。本轮没有发现需要判错的目标产物，也没有做实验反应条件、收率或竞争反应的可行性认证。

## 抽样与口径

- 母体：本地Pilot/Dataset中的add、delete、substitute各50题，共150题；不是上游完整数据集。
- 抽样：先固定随机种子20260905，按add→delete→substitute顺序，分别从排序后的ID用同一Random对象抽7、7、6题，抽样后不替换题目。
- 审读20道题、106个原始推理step；没有抽取我们生成的H/N，没有按Poe失败记录富集。
- 原始150题全部带outcome=True、verifier_checks.all_pass=True。这是源文件元数据，不被当作本审计结论。抽样只代表这一已筛选的本地子集。
- 单位为origin；下面四类记录在本次样本中不重叠。机制疑义不计入确定的事实错误；未填写推理不计入幻觉。

| 审计结论 | 题数 / 20 | 含义 |
| --- | ---: | --- |
| 明确正文化学事实错误 | {counts['factual_error']} | 产物正确，但不能将相关正文标为无幻觉 |
| 自然语言推理未实例化 | {counts['incomplete_reasoning']} | FORMAL有值，正文仍是指令模板 |
| 图编辑与机制解释需限定 | {counts['mechanism_caveat']} | 不判GT错误，不能用作确定机理标签 |
| 本轮未发现明确问题 | {counts['no_material_issue_found']} | 仅在声明的审计范围内 |

## 独立检查做了什么

audit.py不导入molhallulens的builder、planner、propagation或validator；直接读取六个原始JSON文件。独立解析SMILES，核对anchor的map及元素、重原子与环数、片段计数、delta，以及process答案/FORMAL产物与原始GT的分子等价性。共246项检查通过。

随后按anchor切除指定的边界片段、调整氢价态、枚举传入片段的连接原子，重建产物，与GT的isomeric canonical SMILES比较：20/20匹配。**这证明GT可由指定图编辑得到，不证明GT是所有化学条件下唯一产物**；该检查仍读取原FORMAL描述的编辑，指令语义与目标位点另由AI逐题审读。

初版审计器只处理单键离去，漏掉substitute_v2.0043醛基氧的双键；补充价态处理后成功重建。该过程是审计器覆盖修正，不计作数据错误。canonical片段第一个原子不必是连接原子，例如add_v2.0022需连S而非C。

环数按RDKit {evidence['rdkit_version']} 的CalcNumRings口径核对，并检查具体AtomRings；对复杂笼状结构，SSSR与对称化环集可能不同，本样本结论不能推广到所有环定义。见[RDKit ring finding说明](https://www.rdkit.org/docs/RDKit_Book.html#ring-finding-and-sssr)。
"""
    sections = [intro]
    for finding in review["findings"]:
        step_label = ", ".join(map(str, finding["steps"]))
        text = f"\n## {finding['id']} · {finding['origin_id']}\n\n"
        text += f"类型：{review['categories'][finding['category']]}。涉及Step {step_label}。\n\n"
        text += "原文：\n\n> " + finding["quote"] + "\n\n"
        text += finding["explanation"] + "\n\n建议：" + finding["recommended_action"] + "\n"
        if finding.get("reference_url"):
            label = "PubChem CID 11636795" if finding["id"] == "F1" else "OpenStax《Organic Chemistry》21.6"
            text += f"\n化学依据：[{label}]({finding['reference_url']})。这是对结构/机理区别的支持，不是对整条数据正确性的背书。\n"
        sections.append(text)
    sections.append("\n## 逐题核对表\n\n“结构匹配”包含已标注立体信息的分子图一致性；不等于实验验证。完整原始题目、答案与106个step见sample_review.txt；完整JSON、数值、anchor邻居、环成员和重建结果见evidence.json。\n\n| Origin | 重原子 源→产物 | 环数 源→产物 | 结构匹配 | AI化学审读与结论 |\n| --- | --- | --- | --- | --- |\n")
    for reviewed in review["records"]:
        r = records[reviewed["origin_id"]]
        a, b = r["source_computed"], r["answer_computed"]
        sections.append(f"| {r['origin_id']} | {a['heavy_atoms']}→{b['heavy_atoms']} | {a['rings']}→{b['rings']} | 是 | {reviewed['review']} |\n")
    sections.append("""
## 对当前项目意味着什么

1. **保留已有读取和DAG工具，但把结构正确与正文正确分开。** F1表明只覆盖图节点、计数和算术的验证器会漏掉杂环名称错误；all_pass不是全文真值证明。
2. **H/N配对只能控制改写风格，不会自动净化共同前缀。** 如果原文存在F1那样未改动的错误，两侧共享它，N仍不是全真值。配对前需要原文审查；本轮仅提出待修项，未修改源数据或重发标签。
3. **F2适合分子编辑题库，不适合直接当自然完成的推理。** 可暂时移出自然推理评测，或补全后重新审核。补全也是合成数据，需记录来源。
4. **F3/F4应明确任务是净分子图编辑。** 如果要研究反应机理，需另外加入条件、反应步骤和可信的原子来源标注，不能从简化编辑图直接推出。
5. **本轮不回答固定format是否降低LLM性能。** 那需要同题、同模型、同预算的自然输出/固定格式对照实验。这20题也不是用于探针泛化评估的自然错误样本。

## 不确定性与下一步

这是一个小型AI审计，语义判断不是独立人类复核；抽样母体还经过本地筛选，不能将1/20当作上游错误率或已知漏检率。未验证每项系统命名、实验收率、试剂兼容性、对映选择性、所有可能编辑位点或最终探针性能；未发现问题的16题仍可能有漏检。

建议先人工复核F1与F4的具体结构/机制证据，决定是否修订这四题的用途和正文；再扩展到其余130题的同类正文审查。若要检测模型自发错误，另取题目让目标模型自由作答，盲审实际出现的正确/错误claim，作为独立测试集，并按origin切分防止泄漏。不要把现有106步直接宣传为人类审核的自然推理金标准。

## 复现与验证

在本目录、已有molhallulens环境下运行：

```bash
python audit.py
python verify_audit.py
python build_report.py
```

实际结果：20题、106步、246项结构/数值检查通过、20次产物重建匹配；4条原文证据逐字匹配；6个源文件SHA-256未变。另以故意错误的产物计数、答案SMILES、anchor元素测试审计器，3项均检出。骨架反例的环内N数也独立回验通过。没有运行Poe，没有修改生成配置、原数据或生产代码。

audit.ipynb包含可复跑的普通Python单元及实际stdout。本环境缺少Jupyter内核包，因此build_report.py按序在共享Python命名空间执行这些单元，不安装依赖；这不是nbclient/IPython内核验证。若用Jupyter打开，请选择装有RDKit的molhallulens内核。

文件索引：audit.py（独立计算）、verify_audit.py（证据与检错回验）、evidence.json（机器证据及源文件哈希）、review.json（20题AI语义审读与4项发现）、sample_review.txt（完整原文）、audit.ipynb（可复跑伴随分析）。没有为小样本添加总体错误率图；使用逐题证据表。
""")
    (HERE / "REPORT.md").write_text("".join(sections))
    cells = []
    def markdown(text):
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": text})
    def code(text):
        cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "metadata": {}, "source": text,
                      "execution_count": None, "outputs": []})
    markdown("# " + title + "\n\nAI审读，不是人类专家金标准。仅审原始数据；无Poe调用。\n\n在本文件目录用已安装RDKit的Python运行。保存输出由普通Python顺序执行器生成，未通过Jupyter内核。")
    code("from pathlib import Path\nimport json\nfrom collections import Counter\nfrom audit import run, HERE\nfrom verify_audit import verify\nevidence = run()\n")
    code("verification = verify()\n")
    markdown("## 全部20题数值证据\n\n图重建读取原FORMAL指定的编辑，不能单独证明instruction语义；后者对应独立审读记录。")
    code("for r in evidence['records']:\n    print(r['origin_id'], 'source=', r['source_computed'], 'answer=', r['answer_computed'], 'reconstruction=', r['graph_reconstruction'])\n")
    markdown("## 逐题AI语义审读\n\n这些判断来自原题、结构和完整正文的审读，不是由PASS标记自动推断。剩余16题未发现明确问题，不是逐token真值认证。")
    code("review = json.loads((HERE / 'review.json').read_text())\nprint(json.dumps(review, ensure_ascii=False, indent=2))\nprint('Observed categories, not population error rates:', dict(Counter(r['category'] for r in review['records'])))\n")
    markdown("## 限制与来源\n\n仅20/150本地已筛选题；没有上游错误率估计或固定format性能实验。\n\n[PubChem骨架参照](https://pubchem.ncbi.nlm.nih.gov/compound/Pyrazolo_1_5-a_pyrimidine)；[OpenStax酯化与水解](https://openstax.org/books/organic-chemistry/pages/21-6-chemistry-of-esters)；[RDKit环计数](https://www.rdkit.org/docs/RDKit_Book.html#ring-finding-and-sssr)。完整边界与处置建议见REPORT.md。")
    namespace, count = {}, 0
    previous = Path.cwd()
    try:
        os.chdir(HERE)
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            count += 1
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exec(compile(cell["source"], f"audit.ipynb/cell-{count}", "exec"), namespace)
            cell["execution_count"] = count
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": buffer.getvalue()}]
    finally:
        os.chdir(previous)
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (RDKit required)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "execution_method": "shared-namespace Python cell runner; not Jupyter/nbclient"},
                "nbformat": 4, "nbformat_minor": 5}
    (HERE / "audit.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n")
    (HERE / "verification.json").write_text(json.dumps(namespace["verification"], indent=2)+"\n")
    print(f"Wrote REPORT.md and audit.ipynb; {count} Python cells executed successfully.")


if __name__ == "__main__":
    build()
