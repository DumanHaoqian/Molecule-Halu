# ChemCoTBench-V2：20题原始化学审计

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
| 明确正文化学事实错误 | 1 | 产物正确，但不能将相关正文标为无幻觉 |
| 自然语言推理未实例化 | 1 | FORMAL有值，正文仍是指令模板 |
| 图编辑与机制解释需限定 | 2 | 不判GT错误，不能用作确定机理标签 |
| 本轮未发现明确问题 | 16 | 仅在声明的审计范围内 |

## 独立检查做了什么

audit.py不导入molhallulens的builder、planner、propagation或validator；直接读取六个原始JSON文件。独立解析SMILES，核对anchor的map及元素、重原子与环数、片段计数、delta，以及process答案/FORMAL产物与原始GT的分子等价性。共246项检查通过。

随后按anchor切除指定的边界片段、调整氢价态、枚举传入片段的连接原子，重建产物，与GT的isomeric canonical SMILES比较：20/20匹配。**这证明GT可由指定图编辑得到，不证明GT是所有化学条件下唯一产物**；该检查仍读取原FORMAL描述的编辑，指令语义与目标位点另由AI逐题审读。

初版审计器只处理单键离去，漏掉substitute_v2.0043醛基氧的双键；补充价态处理后成功重建。该过程是审计器覆盖修正，不计作数据错误。canonical片段第一个原子不必是连接原子，例如add_v2.0022需连S而非C。

环数按RDKit 2025.09.6 的CalcNumRings口径核对，并检查具体AtomRings；对复杂笼状结构，SSSR与对称化环集可能不同，本样本结论不能推广到所有环定义。见[RDKit ring finding说明](https://www.rdkit.org/docs/RDKit_Book.html#ring-finding-and-sssr)。

## F1 · mol_edit.add_v2.0150

类型：明确的正文化学事实错误，不代表产物错误。涉及Step 5。

原文：

> Source has 6 rings (pyrazolo[1,5-a]pyrimidine core, pyrrolidine, indazole core, and phenyl). Product has 6 rings. RING_DELTA = 0.

总环数6与产物均正确，但第一处稠合杂环名称不符。原结构六元环 maps=[2,31,6,5,4,3] 含3个N；与之稠合的五元环 maps=[13,7,6,31,14] 含1个N。pyrazolo[1,5-a]pyrimidine 则是六元环2N、五元环2N；两者总N都可为3，所以只检查总元素数或总环数会漏掉这个错误。

建议：保留已核对的产物与计数；修订该正文中的骨架描述并重新审查负标签。可先用无歧义描述：a fused six-membered C3N3 ring and five-membered C4N ring；本次不猜测完整稠环位次名称。

化学依据：[PubChem CID 11636795](https://pubchem.ncbi.nlm.nih.gov/compound/Pyrazolo_1_5-a_pyrimidine)。这是对结构/机理区别的支持，不是对整条数据正确性的背书。

## F2 · mol_edit.delete_v2.0056

类型：正文未实例化，不属于已断言的错误事实。涉及Step 1, 2, 3, 4, 5。

原文：

> State which atom in the Indexed SMILES (give the ":n" map number) is the ANCHOR, and its element.

五步自然语言仍在要求读者找anchor、数原子、构建产物，而没有给出本题实际推导。FORMAL已经填入具体值，故不是字段缺失，也不是产物错误。将整段当作已经完成的自然推理会混淆任务说明与推理输出。

建议：在自然推理评测集中暂时排除，或另行补全并审核正文；题目与产物仍可用于分子编辑。不要把它标成化学幻觉。

## F3 · mol_edit.add_v2.0185

类型：净图编辑可成立，但不能据此确定实验机制或原子来源。涉及Step 1。

原文：

> This is an O-alkylation of the carboxylic acid, where the new ethyl group bonds to the hydroxyl oxygen.

题目只要求羧酸变乙酯，未指定试剂。把乙基连到原OH氧上是可行的净图编辑表示，但酯化也可经其他途径，不能由未映射的产物唯一推断为O-alkylation。

建议：限定为 in this graph-edit representation；不作为确定的机理标签，也不据此判产物错误。

化学依据：[OpenStax《Organic Chemistry》21.6](https://openstax.org/books/organic-chemistry/pages/21-6-chemistry-of-esters)。这是对结构/机理区别的支持，不是对整条数据正确性的背书。

## F4 · mol_edit.delete_v2.0255

类型：净图编辑可成立，但不能据此确定实验机制或原子来源。涉及Step 1, 3。

原文：

> After hydrolysis, this O becomes part of the free carboxylic acid -C(=O)OH.

把甲酯O2连接的甲基删去，可以得到正确的未映射羧酸结构。但普通皂化的酰氧键断裂中，原烷氧基氧通常随醇离去；不能把为净图编辑保留O2直接解释为真实反应中该氧留在酸中。题目无条件、产物无原子映射，不能据此判GT结构错误。

建议：明确区分编辑anchor与机制原子去向；若评价机理，需要条件与原子映射支持。

化学依据：[OpenStax《Organic Chemistry》21.6](https://openstax.org/books/organic-chemistry/pages/21-6-chemistry-of-esters)。这是对结构/机理区别的支持，不是对整条数据正确性的背书。

## 逐题核对表

“结构匹配”包含已标注立体信息的分子图一致性；不等于实验验证。完整原始题目、答案与106个step见sample_review.txt；完整JSON、数值、anchor邻居、环成员和重建结果见evidence.json。

| Origin | 重原子 源→产物 | 环数 源→产物 | 结构匹配 | AI化学审读与结论 |
| --- | --- | --- | --- | --- |
| mol_edit.add_v2.0022 | 36→42 | 4→4 | 是 | N36为目标一级胺，丙基磺酰基通过S连接，新增6个重原子，环数不变。不能把canonical片段的第一个C误当作连接点。 |
| mol_edit.add_v2.0079 | 41→45 | 4→4 | 是 | O3羟基甲磺酰化，通过S形成O–S键；增加4个重原子、环数不变。未验证试剂条件、选择性或产物稳定性。 |
| mol_edit.add_v2.0101 | 31→34 | 4→4 | 是 | O31接入2-氟乙基，形成O–CH2–CH2–F；新增3个重原子，原有标注立体结构在重建产物中保留。 |
| mol_edit.add_v2.0150 | 31→34 | 6→6 | 是 | 吡咯烷N11乙酰化与+3重原子正确；Step5稠合杂环名称错误，详见F1。 |
| mol_edit.add_v2.0185 | 39→41 | 4→4 | 是 | 羧酸O30乙基化的净结构及+2重原子正确；不能由目标乙酯唯一推出O-alkylation机制，详见F3。 |
| mol_edit.add_v2.0274 | 39→44 | 5→5 | 是 | 酚氧O39接入CH2CH2N(CH3)2，新增5个重原子；喹唑啉2环、苯1环、吲唑2环，共5环。 |
| mol_edit.add_v2.0295 | 37→44 | 4→5 | 是 | 哌啶N33苄基化，接CH2Ph，增加7个重原子与1个苯环。片段SMILES按氢封端片段理解，不是游离苄基自由基。 |
| mol_edit.delete_v2.0028 | 52→42 | 4→3 | 是 | N22的Cbz脱保护形成一级胺；去除10个重原子及1个苯环，保留其余标注立体结构。 |
| mol_edit.delete_v2.0056 | 41→31 | 4→3 | 是 | N15末端胺Cbz脱保护，41→31重原子、4→3环，结构正确；五步自然语言尚是操作模板，详见F2。 |
| mol_edit.delete_v2.0107 | 39→32 | 6→5 | 是 | N29脱苄基，叔胺变仲胺；移除CH2Ph的7个重原子，少1个苯环。 |
| mol_edit.delete_v2.0126 | 41→34 | 6→5 | 是 | N27脱苄基，移除7个重原子；原6环变5环，剩余稠环、联苯和哌啶与结构相符。 |
| mol_edit.delete_v2.0186 | 38→31 | 6→5 | 是 | O20脱苄基形成酚；去除7个重原子与1环，保留其余苯环、稠杂环及环丙烷。 |
| mol_edit.delete_v2.0255 | 47→46 | 5→5 | 是 | 甲酯水解的目标羧酸正确，47→46重原子；O2留存是图编辑约定，不等于实际水解中的氧来源，详见F4。 |
| mol_edit.delete_v2.0287 | 49→39 | 6→5 | 是 | 哌嗪N10的Cbz脱保护形成NH；减少10个重原子和1个苯环，其余环系统与计数一致。 |
| mol_edit.substitute_v2.0005 | 32→32 | 4→4 | 是 | CH2OH的C25所接O26换成Cl；重原子总数不变、环数不变，邻近已标注手性中心保留。 |
| mol_edit.substitute_v2.0009 | 34→35 | 4→4 | 是 | 磺酰S18上的Cl21换成NHCH3；去1加2净+1重原子，得到对应N-甲基磺酰胺。 |
| mol_edit.substitute_v2.0043 | 26→42 | 4→7 | 是 | 醛C12=O13与目标哌啶胺的还原胺化：去O1、接含17重原子的胺片段，净+16；醛碳由CH=O成为CH2–N，新增3环。 |
| mol_edit.substitute_v2.0065 | 33→43 | 4→5 | 是 | 羧酸C12上的OH14换成3-乙氧羰基哌啶的N；去1加11净+10重原子，新增1个哌啶环。 |
| mol_edit.substitute_v2.0078 | 35→43 | 4→6 | 是 | 芳基C16的F17被环丙基哌嗪N替换；去1加9净+8重原子，新增哌嗪与环丙烷2环；未据此保证具体SNAr实验条件。 |
| mol_edit.substitute_v2.0239 | 51→57 | 5→5 | 是 | 三嗪C13的Cl14被N(CH2CH2OH)2替换；去1加7净+6重原子，没有新环，其余取代基保持。 |

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
