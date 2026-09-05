# ChemCoTBench-V2：第二轮20题化学审计

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

## R2-F1 · mol_edit.add_v2.0111

涉及Step 5；类型：正文存在明确的骨架命名或官能团分类错误；不是GT图结构错误。

原始正文：

> The source contains 6 rings (phenyl, pyridine, thieno[3,2-c]pyridine (2 rings), thiazole, and piperidine).

thieno[3,2-c]pyridine应为thieno[3,2-b]pyridine。该稠环包含map10–18，吡啶N13直接邻接稠合C14；PubChem的b型结构匹配这些原子，c型结构不匹配。两种骨架同分子式、同环数，故重原子数和环数校验无法区分。

建议：修正Step5骨架名称；保留N30乙酰化产物与36→39重原子、6→6环的计数。

对照依据：[PubChem CID 12210218: thieno[3,2-b]pyridine](https://pubchem.ncbi.nlm.nih.gov/compound/Thienopyridine)；[PubChem CID 67500: thieno[3,2-c]pyridine](https://pubchem.ncbi.nlm.nih.gov/compound/Thieno_3_2-c_pyridine)。
上述错误判断基于本地结构与对照结构/定义的比较；网页只提供化学对照，不对整条样本背书。

## R2-F2 · mol_edit.delete_v2.0134

涉及Step 3；类型：正文存在明确的骨架命名或官能团分类错误；不是GT图结构错误。

原始正文：

> The ANCHOR nitrogen gains an implicit H to satisfy its valence, becoming a primary amine (-NH2).

脱Cbz后N3确实变成NH2，但它剩下的重原子邻居是磺酰S4，不是碳：H2N–S(=O)2–NH–Ar。这是sulfamide的末端NH2，不是普通primary amine。不能只看氢数就把所有NH2归为一级胺。原题的sulfamide nitrogen反而正确。

建议：改为the deprotected terminal sulfamide nitrogen bears two hydrogens；不要改动正确的GT和计数。

对照依据：[IUPAC Gold Book: amines](https://goldbook.iupac.org/terms/view/A00274)。
上述错误判断基于本地结构与对照结构/定义的比较；网页只提供化学对照，不对整条样本背书。

## R2-F3 · mol_edit.substitute_v2.0122

涉及Step 6；类型：正文存在明确的骨架命名或官能团分类错误；不是GT图结构错误。

原始正文：

> one isoindolinone core which has 2 rings

原结构是oxindole / indolin-2-one型内酰胺，不是isoindolinone。五元环map11–10–30–25–13中，N10直接连接芳环融合C30和羰基C11；独立子结构匹配命中oxindole、未命中isoindolin-1-one。两者各有2环且分子式相同，计数检查不能发现。

建议：将isoindolinone改为oxindole (indolin-2-one)。同句的chroman-like只是类比；实际含氧五元环是map14–13–21–16–15，建议改用精确的五元环描述，但不另增加一个确定错误计数。

对照依据：[PubChem CID 321710: oxindole](https://pubchem.ncbi.nlm.nih.gov/compound/Oxindole)；[PubChem CID 10199: isoindolin-1-one](https://pubchem.ncbi.nlm.nih.gov/compound/Isoindolin-1-one)。
上述错误判断基于本地结构与对照结构/定义的比较；网页只提供化学对照，不对整条样本背书。

## R2-F4 · mol_edit.substitute_v2.0150

涉及Step 6；类型：正文存在明确的骨架命名或官能团分类错误；不是GT图结构错误。

原始正文：

> The source molecule has 4 rings (pyrimidinone, 2-fluorophenyl, pyrazole, phenyl).

第一个含氮六元环中N5与N14直接相连，为1,2-diazine（pyridazinone型）骨架；pyrimidinone要求1,3排列，两个N之间应隔一个C。该差别不是互变异构或SMILES遍历顺序造成的。Br替换为2-oxopyrrolidin-1-yl的产物及4→5环仍然正确。

建议：将pyrimidinone改为pyridazinone型骨架描述，或直接写明含相邻两个N的六元羰基杂环；本次不指定完整位次名称。

对照依据：[PubChem CID 9260: pyrimidine, N atoms separated by one carbon](https://pubchem.ncbi.nlm.nih.gov/compound/Pyrimidine)。
上述错误判断基于本地结构与对照结构/定义的比较；网页只提供化学对照，不对整条样本背书。

## R2-F5 · mol_edit.substitute_v2.0064

涉及Step 1, 4；类型：题面没有指定新片段构型，答案却选定立体异构体；不判该异构体本身错误。

题目：

> Please substitute the chloro group on the pyridine ring with a 4-hydroxy-2-(hydroxymethyl)pyrrolidin-1-yl group.

原始正文：

> The SMILES for this fragment is "N1C[C@H](O)C[C@H]1CO".

instruction没有R/S、cis/trans或新片段的带立体SMILES。原分子中也没有这两个中心，但FORMAL和GT选定两个手性中心。固定连接关系后RDKit可枚举4个立体异构体，GT仅是其中之一。这不是证明GT分子不成立，而是其立体选择无法从本地输入唯一推出。

建议：补充明确的立体指令/片段结构，或明确按非立体连接关系评分、允许相应异构体；在规则确定前不要把其他构型自动标成化学幻觉。

这是基于本地输入缺少立体限定、输出却新增立体标记作出的判断；4个候选异构体保存在semantic_checks.json中，不表示它们在任意实验条件下都等概率生成。

## 另外两类问题不混入“4题明确正文错误”

### Boc中心碳的术语歧义

delete_v2.0036的Step2写“quaternary C + 3 methyls”。原结构中心C2连接3个C和1个O：若按碳邻居数分类，不是季碳；某些NMR语境会把无氢碳称为quaternary。原文没有说明口径，建议直接写“central non-protonated carbon”。本轮不据此增加明确错误题数。

### rxn_cls不能未经确认当作本次编辑类型

以下8题的字段与当前净编辑类型不一致。未找到本地字段定义，可能是继承的原反应标签；原因尚未证实。风险是用它分层、训练或解释当前任务时会引入混淆，而不是直接证明这8题的推理都错。

| Origin | 原rxn_cls | 本题净编辑 |
| --- | --- | --- |
| mol_edit.add_v2.0035 | C-N Bond Formation | O–S成键 |
| mol_edit.add_v2.0172 | C-N Bond Formation | O–C(=O)成键 |
| mol_edit.add_v2.0229 | C-N Bond Formation | N–S成键 |
| mol_edit.substitute_v2.0032 | C-C Bond Formation | C–Cl换为C–N |
| mol_edit.substitute_v2.0064 | C-C Bond Formation | C–Cl换为C–N |
| mol_edit.substitute_v2.0122 | Heterocycle Synthesis | 羧酸OH换Cl，未成环 |
| mol_edit.substitute_v2.0150 | C-C Bond Formation | C–Br换为C–N |
| mol_edit.substitute_v2.0281 | Heterocycle Synthesis | 烯丙位OH换Cl，未成环 |

## 全部20题逐题结果

数值表示源→产物。结构重建20/20匹配，保留已指定的立体信息；这不证明实验可行性或答案唯一性。语义结论来自AI阅读原题和完整step_text，不是从原数据的all_pass推断。完整原文见sample_review.txt，机器证据见evidence.json。

| Origin | 重原子 | 环数 | 逐题化学核对 |
| --- | --- | --- | --- |
| mol_edit.add_v2.0013 | 32→35 | 4→4 | N2是目标仲脂肪胺而非N16芳胺；乙酰基含2C+O，通过羰基C成键，+3重原子。喹唑啉2环、苯1环、吡啶1环正确；已有立体标记保留。 |
| mol_edit.add_v2.0035 | 34→38 | 4→4 | O8为CH2OH的氧，接SO2NH2的S而非N，新增N1S1O2共4重原子；环戊烷、两苯与吡啶共4环。rxn_cls另有口径问题。 |
| mol_edit.add_v2.0111 | 36→39 | 6→6 | N30哌啶乙酰化正确，+3重原子、6环不变；Step5将thieno[3,2-b]pyridine误写为c型，见R2-F1。 |
| mol_edit.add_v2.0119 | 33→37 | 5→5 | N1接甲氧羰基生成甲基氨基甲酸酯；新增C2O2共4原子。两苯、独立吡啶和呋喃并吡啶共5环。 |
| mol_edit.add_v2.0172 | 40→59 | 4→4 | 仲醇O28与硬脂酰羰基C成键；18C+1O共19重原子，40→59。芴骨架3环加苄酯苯1环，保持4环；已有立体标记保留。 |
| mol_edit.add_v2.0183 | 33→39 | 5→5 | 目标是芳胺N29，而非酰胺NH2的N22；新增异丙基氨甲酰C4NO共6原子生成脲。苯、咪唑及咔唑三环共5环。 |
| mol_edit.add_v2.0229 | 54→67 | 7→8 | N33为仲胺，N21/N54为酰胺氮；连接磺酰S。噻唑片段含两个S（一个芳香s）、C6N2O3，共13原子，新增1环。 |
| mol_edit.delete_v2.0016 | 39→32 | 5→4 | O8断苄基C9形成酚；去7原子和1苯环，39→32重原子、5→4环。其余3苯与噁唑匹配。 |
| mol_edit.delete_v2.0036 | 39→32 | 5→5 | N8去Boc的7原子后得到苄胺NH2，环数5不变；苯并噻吩计2环。quaternary C说法另列术语歧义，不计入明确错误。 |
| mol_edit.delete_v2.0108 | 42→32 | 5→4 | N25去Cbz，去10原子和1苯环，形成苯胺；四氢喹啉型稠环与已有手性结构保持。 |
| mol_edit.delete_v2.0134 | 44→34 | 6→5 | N3去Cbz正确，44→34原子、6→5环；产物末端是sulfamide NH2而不是一级胺，见R2-F2。 |
| mol_edit.delete_v2.0273 | 56→49 | 7→6 | O49去苄基C50，-7原子、-1环。3个哌啶、1个七元二氮环、3个苯环的源枚举匹配；保留手性。 |
| mol_edit.delete_v2.0275 | 62→52 | 3→2 | 糖环C41上的N42脱Cbz变NH2；去10原子和1苯环，62→52、3→2环。糖环及已指定的多个立体中心保留。 |
| mol_edit.delete_v2.0299 | 44→37 | 5→4 | O37去苄基C38，得到羧酸，-7原子、-1环；Boc与三苯甲基保留。题目没有明确要求水解，不把苄酯脱保护的O留存误判成上一轮的水解机制疑义；实际选择性未验证。 |
| mol_edit.substitute_v2.0032 | 29→37 | 4→5 | C22的Cl23换成N-乙酰哌嗪游离N端，片段6C2NO共9原子，净+8、+1环；原苯/吡啶/嘧啶/吡唑枚举匹配。 |
| mol_edit.substitute_v2.0057 | 28→35 | 4→5 | 酰氯C21的Cl23由NCCN1CCCC1的末端一级胺N替换成酰胺；加C6N2共8、去1，净+7原子、+1吡咯烷环。 |
| mol_edit.substitute_v2.0064 | 35→42 | 3→4 | C25–Cl换C–N的连接关系及+7原子、+1环正确；题面却未指定新吡咯烷片段的两个中心构型，见R2-F5。 |
| mol_edit.substitute_v2.0122 | 31→31 | 6→6 | C2的OH3换Cl生成酰氯正确，31原子、6环不变；原文把oxindole型骨架认成isoindolinone，见R2-F3。 |
| mol_edit.substitute_v2.0150 | 28→33 | 4→5 | C10的Br11换为2-吡咯烷酮N，净+5原子、+1环正确；原六元杂环N5–N14相邻，不是pyrimidinone，见R2-F4。 |
| mol_edit.substitute_v2.0281 | 46→46 | 5→5 | 烯丙位CH2的C5去O6加Cl；重原子46、环数5保持，原E/Z与标注手性保留。环上6条键均单键、双键外环，不把central cyclohexane ring判错；未验证具体反应条件。 |

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
