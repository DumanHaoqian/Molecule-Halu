# Agent CoT 幻觉构造与因果评估工作流

最近完成的是 **v17 固定物理图任务解读诊断**，使用独立启动器 [run_task_intent_exp.sh](../scripts/run_task_intent_exp.sh)。同四个酰化题的 N、原题和错误产物不变，只将 intent H 的名称及既有任务复述统一为实际错误片段名称。80 个贪心答案已完成；ChemDFM B 从 3/4 降为 2/4，Chem-R 从 2/4 升为 3/4，两样式均未达到双模型 H<A。没有完整 P′ 或可判定的错误局部连接匹配，未推进 heldout。

保留的完整生成协议为 **v13 源状态诊断**（目录 `agent_corruption_v13`，协议字段 `agent_source_state_v1`）。当前生成入口是 [generate_source_agent_pairs.py](../scripts/generate_source_agent_pairs.py)，评估入口仍为 [evaluate_agent_corruption.py](../scripts/evaluate_agent_corruption.py)；无参数 [run_agent_exp.sh](../scripts/run_agent_exp.sh) 已指向 v13。下文命令按当前协议说明；旧 [generate_agent_pairs.py](../scripts/generate_agent_pairs.py) 保留 v12 severe 构造，不是 v13 入口。

v13 两模型的答案、GT probe 与 n=8 熵诊断已全部完成并归档；**两模型均为 B>A，双模型开发门槛失败，未推进 heldout**。下文分开报告 v13 与 v12 的实测结果；两版本接受集合与公开 N 不同，不能把跨版本差异单独归因于干预类型。v13 的假设与版本边界见[实施计划](superpowers/plans/2026-09-20-source-state-diagnostic.md)。

## v17 完整任务解读对照结果

监督进程于 **2026-09-20 06:21:51 +08:00，status=0** 结束。四组各 20/20、共 80 个贪心答案完成；全部可解析且无 dummy，缺失和截断均为 0。本轮没有 probe 或熵。两样式均纳入固定 4/18 题，14 题为诊断范围外，生产验收与新增 API 审查计数均为 0。[化学独立审计](../Experiments/agent_corruption_v17/review/independent_task_intent_chemistry.md)通过 199 项检查，[标签审计](../Experiments/agent_corruption_v17/review/task_intent_label_audit.md)通过 13,800 项，[完整提示词预检](../Experiments/agent_corruption_v17/review/evaluator_preflight.md)通过 165,797 项，独立核对 32,722 条 token 记录。

binding 精确保留 v16 增强 N/H；intent 的 N 逐字相同，只更改 H 的任务名称及 add0194 原有任务复述。当前根由 `structural/fragment` 改为 `task_interpretation/task_group`，fragment 变为传播后果；仍是每题一个根，物理计划、产物、计数和连接图不变。关闭的是 H 内部名称与图的矛盾，与原始 instruction 的外部冲突仍在。已有部分产物信息也仍在，不把该表达称为自然错误分布。

| 模型 | 样式 | A | B | C | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | binding | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| ChemDFM-R-14B | intent | 2/4 | 2/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | binding | 3/4 | 2/4 | 4/4 | 2/4 | 2/4 |
| Chem-R-8B | intent | 3/4 | 3/4 | 4/4 | 2/4 | 2/4 |

跨样式 B 的配对变化为 ChemDFM −25 个百分点：add0194 转错，1/3 个原正确样本，Wilson 95% 区间 [6.15%, 79.23%]，McNemar 精确双侧 p=1；Chem-R +25 个百分点：add0172 恢复正确，转错 0/2，p=1。两模型 B 的完整分子平均相似度分别从 0.935714→0.873214、0.903991→0.950658。完整手性准确率与表中主准确率相同。每个条件使用本轮 A/C，未替换为历史基线：DFM binding/intent 的 B−A 为 +25/0、B−C 为 0/−25 个百分点；Chem-R 为 −25/0、−50/−25 个百分点。

A/C/D/E 的完整请求跨样式均相同。两模型 A/C/E 各四题的完整生成文本、解析答案、完整分子图和正确性均 4/4 相同；D 的完整生成文本仅各 1/4 相同，但解析答案和完整分子图仍各 4/4 相同。这里能够检查的是本轮输出稳定性，不由 D 文本差异推断原因。

两模型两样式的 B 均无完整 P′ 匹配（手性与无手性都是 0/4），也无仅 GT 手性差异。局部分类器沿用冻结 v16 规则：DFM binding/intent 可判定 4/4、4/4，局部 N/H 为 3/0、2/0；Chem-R 为 3/4、4/4，局部 N/H 为 2/0、3/0。Chem-R binding 的 add0172 为 `unknown_source_not_intact`，不能算作未采纳 H。所有 B 均无开放端口草稿复制。source intact 只指重原子身份与诱导内部连接保持，不保证远端氢数、取代或立体正确；局部匹配也不证明完整计划或内部推理路径。

无密度例外且不筛例；错误节点 binding 为 8/10/7/8，intent 为 9/11/8/9，11 处任务名称跨度共用每题同一个根。两样式均未达到双模型 H<A，四题探索诊断也不能单独授权 heldout 扩展。全部逐例输出、相似度、配对区间与机制状态见[完整对照报告](../Experiments/agent_corruption_v17/review/task_intent_comparison.md)和[JSON](../Experiments/agent_corruption_v17/review/task_intent_comparison.json)；[精确日志](../Experiments/agent_corruption_v17/review/run_task_intent_exp.log)及[终态验证](../Experiments/agent_corruption_v17/review/terminal_validation.json)已归档。

## v16 完整开放端口连接对照结果

两条件保持原问题、instruction、GT、四条错误计划、执行产物和单个 fragment 根不变。`native` 逐字复用 v15 native N/H；`native_with_connections` 只在各自原文末尾追加同格式的源锚点＋全部入射片段连接表示，用 `[*:k]` 表示通向被省略原源原子 k 的开放端口。它不是氢封端或完整产物，也不能据此推断完整分子的环环境；显式部分产物仍携带答案信息。两条件各保留全部四个指定 origin、14 个范围外记录，不构成八个独立问题。

监督进程于 2026-09-20 05:54:40 +08:00 以 status=0 结束，四组各 20/20 个答案完成。80 个输出均可解析且不含 dummy，缺失与截断均为 0；本轮没有 probe 或熵。[独立化学核查](../Experiments/agent_corruption_v16/review/independent_connection_chemistry.md)、[标签审计](../Experiments/agent_corruption_v16/review/anchored_connection_label_audit.md)及[完整提示词预检](../Experiments/agent_corruption_v16/review/evaluator_preflight.md)均通过。标签审计为 11,906 项检查；提示词预检为 150,861 项检查，覆盖 80 个请求和 29,781 条实际提示词 token 记录。没有新增 API 审查或生产验收。

| 模型 | 条件 | A | B（H） | C（N） | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | native | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| ChemDFM-R-14B | native_with_connections | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | native | 3/4 | 3/4 | 2/4 | 2/4 | 2/4 |
| Chem-R-8B | native_with_connections | 2/4 | 2/4 | 4/4 | 2/4 | 2/4 |

所有主组分与完整含立体分子正确性相同。跨条件 B 差异：DFM 为 0；Chem-R 为 −25 个百分点，add0172 转错 1/3，Wilson 95% 区间 [6.15%, 79.23%]、精确 McNemar p=1。Chem-R 的 C 则由 2/4 升至 4/4，救回两题，p=0.5。各条件内部采用本轮自己的 A/C：DFM 两次 B−A 都为 +25 个百分点、B−C 为 0；Chem-R 两次 B−A 都为 0，native B−C 为 +25、增强条件为 −50 个百分点（add0098、add0172 转错 2/4，Wilson [15.00%, 85.00%]，p=0.5）。四题探索性结果未作多重比较校正；两个条件均未满足双模型 B<A。

**重复 A 不完全稳定：DFM A 为 4/4 原文与分子重复，Chem-R 为 3/4。** Chem-R add0172 在 native 轮 A 完整正确，增强轮 A 只缺失立体指定，去立体后仍匹配 GT；两次请求完全相同。B 在同一 origin 的转错则改变了连通图，不能把 A/B 的错误类型混同，也不能将 A 的波动归因于不存在于 A 提示词中的追加块。D 两模型都只有 1/4 完整生成文本相同，但提取答案、完整图和正确性全部 4/4 相同。历史 v15 A 不替代本轮观测。

全部 16 个 B 对完整 P′ 的含立体/去立体匹配均为 0，B 中没有仅手性不同的错误。80 个答案均无 dummy，也没有任何 N/H 开放端口草稿的完整复制。附加局部判定在全部有效 source 嵌入下要求同一预测锚点及同一带原端口身份的连接投影；4096 上限触顶、源图不完整、锚点/投影歧义和不支持边界都单列不可判定。冻结 GT/P′ 的八个阳性控制正确恢复 N/H 连接，允许的远端对称嵌入数分别为 16/8/2/2。

| 模型 | 条件 | B 局部可判 | 正确 N 连接 | 错误 H 连接 | 不可判定 |
| --- | --- | ---: | ---: | ---: | --- |
| ChemDFM-R-14B | native | 4/4 | 3 | 0 | 0 |
| ChemDFM-R-14B | native_with_connections | 4/4 | 3 | 0 | 0 |
| Chem-R-8B | native | 4/4 | 3 | 0 | 0 |
| Chem-R-8B | native_with_connections | 3/4 | 2 | 0 | add0172：source 不完整 |

表中 N/H 数只来自可判定输出；剩余可判定错误属于其他连接，不能把增强 Chem-R 的 add0172 记成“未采用 H”。输出匹配也不证明内部推理过程。每题的 dummy、草稿复制、GT/P′ 手性分类、局部状态和配对翻转均见[完整报告](../Experiments/agent_corruption_v16/review/anchored_connection_comparison.md)及[逐例 JSON](../Experiments/agent_corruption_v16/review/anchored_connection_comparison.json)。判定规则在目标运行前固定于[本轮计划](superpowers/plans/2026-09-20-anchored-connection-diagnostic.md)，主分母始终为四个 origin。

两条件均无密度例外：错误节点从 7/9/6/7 增为 8/10/7/8，每题只新增一个 propagated 节点；DFM 完整提示词错误 token 从 80–85 增为 104–115，Chem-R 从 70–74 增为 92–101。根和图干预不变，不把更长的表达改称更大的化学错误。结果没有授权 heldout 扩展或按输出选取条件/样本；精确[执行日志](../Experiments/agent_corruption_v16/review/run_anchored_connection_exp.log)与[终态记录](../Experiments/agent_corruption_v16/review/terminal_validation.json)已保存。

## v15 完整命名片段表达对照结果

本轮在原定 18 个 development origin 中固定纳入 add0098、add0172、add0193、add0194 四个酰化题，其余 14 个记录为诊断范围外，不是化学失败；没有按模型输出筛题。两种表达使用完全相同的正确源锚点、参考操作和错误入射片段 `C(=O)c1ccc(C(F)(F)F)cc1`，原问题、instruction、GT 均不变。native N 是原始 N 去完整答案后的投影，native H 使用已审计的声明式跨度补丁；ledger 不展示原片段名称。两条件的错误节点与 token 数不同，因此不能把差异单独归因于名称或纯措辞。

监督进程于 2026-09-20 05:28:28 +08:00 以 status=0 结束。四组各 20/20 条 ABCDE 答案全部完成，80 个输出全部有效，无缺失或截断；未运行本轮 GT probe 或熵。每组是同四个 origin 的重复测量，不能视为八个独立问题。[独立化学审计](../Experiments/agent_corruption_v15/review/independent_chemistry_review.md)完成 899 项检查、零失败；[独立标签审计](../Experiments/agent_corruption_v15/review/named_binding_label_audit.md)完成 8,924 项检查及 67 个 native 补丁重建；[完整提示词预检](../Experiments/agent_corruption_v15/review/evaluator_preflight.md)完成 104,513 项检查，核对 80 个请求与 20,553 条实际提示词 token 记录。没有新的 API/LLM 文本审查，production_accepted 为 0。

| 模型 | 表达 | A | B（H） | C（N） | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | ledger | 2/4 | 2/4 | 2/4 | 2/4 | 2/4 |
| ChemDFM-R-14B | native | 2/4 | 3/4 | 3/4 | 2/4 | 3/4 |
| Chem-R-8B | ledger | 2/4 | 2/4 | 3/4 | 2/4 | 2/4 |
| Chem-R-8B | native | 2/4 | 3/4 | 2/4 | 2/4 | 2/4 |

本轮所有主组分与完整含立体分子正确性相同。跨表达 B(native)−B(ledger) 在两模型均为 +25 个百分点，均救回一题、转错 0/2，精确 McNemar p=1；DFM 救回 add0194，Chem-R 救回 add0172。同条件 B−A：ledger 均为 0，native 均为 +25 个百分点；全部转错率为 0/2，Wilson 95% 区间 [0%, 65.76%]。B−C：DFM 两条件都为 0；Chem-R ledger 为 −25 个百分点（add0098 转错 1/3，Wilson [6.15%, 79.23%]，p=1），native 为 +25 个百分点（救回 add0172，p=1）。四题的不确定性很大；这些探索性 p 值未作多重比较校正，不能推断一般稳定改善或破坏。

**全部 16 个 B 输出均未匹配完整错误计划产物 P′，去掉立体要求仍为零。** B 中也没有仅因立体不同而错的答案。DFM ledger 的错误为 add0172、add0194，native 只剩 add0172；Chem-R ledger 的错误为 add0098、add0172，native 只剩 add0098。这些都是其他有效完整分子图，而不是 P′；逐例 GT/P′ 含立体与去立体分类及原始预测见[配对报告](../Experiments/agent_corruption_v15/review/named_binding_comparison.md)和[完整 JSON](../Experiments/agent_corruption_v15/review/named_binding_comparison.json)。完整 P′ 未匹配不证明模型内部忽略 H，也不排除部分结构变化。

**A 的重复控制在两模型均为 4/4 完整文本、提取 SMILES、分子图和正确性完全相同。** D 两模型都只有 1/4 完整文本相同，3/4 含立体分子相同，但 4/4 正确性相同：DFM 的 add0194 连通图发生变化，两次都错；Chem-R 的 add0172 仅立体指定不同，两次都属于 GT 连通图匹配但完整立体不符。这些实际重复运行差异保留在报告，不为其编造原因，也不拿历史 A 替代本轮基线。

两条件都是每题一个 structural 根；ledger 错误节点数依次为 7/7/5/7，native 为 7/9/6/7。ledger add0193 的五节点密度例外被保留；DFM H 错误 token 为 ledger 59–65、native 80–85，Chem-R 为 47–53、70–74。无按阈值补数值错误或筛例。两条件均未达到双模型 B<A，本四题诊断也不能单独授权 heldout 扩展，保留两个表达的全部结果。

本轮[终态与日志说明](../Experiments/agent_corruption_v15/review/terminal_validation.json)记录了一个仅影响 `MODEL DONE` 样式文本的 Bash 动态作用域问题：ledger 阶段结束标签误写 native。实际 `MODEL START`、冻结数据/输出路径、请求和各目录完整结果均正确；分析依据这些文件而非该标签。执行脚本和[日志](../Experiments/agent_corruption_v15/review/run_named_binding_exp.log)已精确归档，后续修复不改本轮快照。设计边界见[v15 计划](superpowers/plans/2026-09-20-named-fragment-binding-diagnostic.md)。

## v14 完整次序对照结果

这轮仅运行贪心 ABCDE；未运行 v14 GT probe 或熵。监督进程于 2026-09-20 04:46:53 +08:00 以 status=0 结束，四组各 70/70 条答案全部完成，280 个输出全部有效，无缺失或截断。每组仍是同一批 14 个 origin，不能将两个条件当成 28 个独立问题。原始 18 个计划样本中的 4 个父版本拒绝仍保留；没有按目标结果补采或筛选。

[独立构造审计](../Experiments/agent_corruption_v14/review/independent_order_audit.md)覆盖两个条件共 3,962 项检查，未发现失败；[CPU 评估准备审计](../Experiments/agent_corruption_v14/review/evaluator_preflight.md)核对全部 280 个请求及 53,124 条实际提示词 token 记录。原题与输入、14 条结构编辑计划、27 个根及执行产物均冻结；同一条件 N/H 次序一致，反向置换后两条件文本逐字相同。没有新的 Agent 文本审查，也没有密度例外。日志已[逐字节归档](../Experiments/agent_corruption_v14/review/run_edit_order_exp.log)，[终态核验](../Experiments/agent_corruption_v14/review/terminal_validation.json)保存完成数与退出状态。

| 模型 | 次序 | A | B（H） | C（N） | D | E |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | account_last | 11/14 | 12/14 | 12/14 | 11/14 | 12/14 |
| ChemDFM-R-14B | edit_last | 11/14 | 12/14 | 12/14 | 11/14 | 12/14 |
| Chem-R-8B | account_last | 8/14 | 7/14 | 12/14 | 9/14 | 10/14 |
| Chem-R-8B | edit_last | 8/14 | 8/14 | 11/14 | 9/14 | 9/14 |

主组分与完整含立体分子的正确计数相同。跨次序 B(edit_last)−B(account_last)：DFM 无转错或救回；Chem-R 为 +7.14 个百分点，救回 delete0255 一题、转错 0/7，精确 McNemar p=1。account_last 中该题相对 A 的唯一转错同时改变 GT 连通性，edit_last 恢复了完整正确产物。结果未支持“错误编辑放最后会使答案更差”。

各条件内部均使用本轮实际 A/C：DFM 两次 B−A 都为 +7.14 个百分点、B−C 为 0；Chem-R account_last 的 B−A 为 −7.14、B−C 为 −35.71（5/12 转错，p=0.0625），edit_last 分别为 0、−21.43（3/11 转错，p=0.25）。这些是 14 题的探索性配对结果，p 值未作多重比较校正。两个条件均未满足双模型 B<A；本诊断本身也不授权 heldout 扩展，没有挑选某个次序作为“赢家”后重算分母。

**重复控制：两模型的 A 在两个条件间均为 14/14 完整生成文本、提取 SMILES 和正确性完全相同。** D 的完整生成文本相同数分别为 DFM 12/14、Chem-R 10/14，但提取 SMILES、含立体完整图和正确性仍全部 14/14 相同。因此 D 的文字变化不是答案变化。Chem-R 本轮 A 固定为 8/14，与历史 v12 的 9/14 不同；历史正确率不替代本轮基线，此处不为跨轮差异编造原因。

两条件、两模型 B 对完整 H 计划产物均为 0/14 匹配，去掉立体要求仍为零。Chem-R B 的 GT 完整含立体/去立体匹配为 account_last 7/8、edit_last 8/10：add0101 在两次都仅有立体差异，edit_last 的 add0172 也属于此类。完整 H 产物不匹配不能证明内部忽略 CoT，亦不穷举部分结构采纳。全部配对统计、转错/救回 origin 和重复控制逐题证据见[次序对照报告](../Experiments/agent_corruption_v14/review/order_comparison.md)及其 JSON；生成假设和版本边界见[v14 计划](superpowers/plans/2026-09-20-edit-recency-diagnostic.md)。

## 当前 v13 生成与独立审计

[v13 生成汇总](../Experiments/agent_corruption_v13/development18/summary.json) 已终结全部 18 个 development origin：**17 个接受、1 个拒绝**（add 5/6、delete 6/6、substitute 6/6）。`add_v2.0287` 没有满足保护区域与远端含环分支条件的候选，保存为 `no_feasible_source_state`；没有替换成更容易的题。`pairs.jsonl` 的 34 行是 17 对 N/H，不是 34 个独立问题。

[独立化学审计](../Experiments/agent_corruption_v13/development18/review/independent_chemistry_review.md) 检查了全部 17 条已选计划的分阶段图、氢/键收支、保留源原子对应、参考编辑与立体化学，未发现阻断问题。[独立标签与幅度审计](../Experiments/agent_corruption_v13/development18/review/source_state_label_audit.md) 对两套冻结 tokenizer 完成 34 次完整 H 编码重建，逐项核对 ID、词表 token、偏移、原文切片、全部跨度重叠与多标签边界，全部通过。每条 H 均为 1 个 `source_graph` 根、6 个实际错误语义节点、16 次已标注值提及；不含产品局部窗口。

程序接受不表示 Agent 一致同意。17 条记录中，参考审查为 2 agreed/15 disputed，符合性审查为 6 agreed/11 disputed，同时 agreed 为 0。原始异议、独立核查处置与 `review_status` 全部保留；模型意见不是化学证明，异议本身也不是已确认的额外缺陷。

已选 H 遗漏原源图重原子的比例最小/中位/最大为 **13.04% / 32.26% / 48.94%**；考虑边界存活原子属性与键变化后的 source footprint 为 **15.22% / 35.48% / 51.28%**。完整产物无手性 Morgan 相似度为 **0.4079 / 0.6076 / 0.7846**。这三项不能互相替代，更不是目标模型错误概率。两种 tokenizer 的语义错误 token 范围分别为 52–86、42–73；字面变化与语义标签的比例另列于审计报告。

v12/v13 有 13 个共同接受的 origin，其公开 N 全部不同：v13 增加完整真实源 SMILES/分子式前言，并移除产品局部窗口。**N 只在同一 v13 构造内独立冻结，不能声称 v12→v13 的 N 逐字不变，也不能把跨版本效果差异单独归因于 H 幅度。**

## v13 完整结果

监督进程于 2026-09-20 04:29:47 +08:00 以 status=0 结束；每模型完成 85/85 条 ABCDE 答案、68/68 条 GT probe、68/68 条 ABCE 熵请求（每请求 8 次，每模型 544 个采样输出）。[原始日志](../Experiments/agent_corruption_v13/development18/evaluation/archive/run_agent_exp.log)、[自动报告](../Experiments/agent_corruption_v13/development18/evaluation/archive/agent_corruption_result.md)和启动器已按字节归档；[终态核验](../Experiments/agent_corruption_v13/development18/evaluation/archive/terminal_validation.json)保留协议、目录与完成数证据。

| 模型 | A | B（H） | C（N） | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | 14/17 | 15/17 | 14/17 | 15/17 | 15/17 |
| Chem-R-8B | 9/17 | 12/17 | 11/17 | 8/17 | 8/17 |

主组分与完整含立体分子计数相同。DFM 的 85 个答案均有效；Chem-R 为 83/85，有 2 个非法分子分别在 A、D，所有组均无缺失或截断。DFM B−A 为 +5.88 个百分点，转错 0/14、救回 1，精确 McNemar p=1；Chem-R 为 +17.65 个百分点，转错 1/9（Wilson 95%：[1.99%, 43.50%]）、救回 4，p=0.375。两模型 B−C 均为 +5.88 个百分点，p=1。[v13 门槛](../Experiments/agent_corruption_v13/development18/v13briefgate.json)保留失败决定；全部 17 个接受 origin 均纳入，没有按目标答案筛题。

[源状态输出核查](../Experiments/agent_corruption_v13/development18/review/source_state_mechanisms.md)比较完整实际源 S、感知源 S′、GT P 与感知源上同一参考编辑所得 P′。两模型 B 与 S′、P′ 的匹配均为 0/17，去除立体要求也为零；DFM B 有 1 个输出等于真实源 S。Chem-R 的唯一 A 正确→B 错误 origin 为 add0172，输出连通性仍不等于 GT；另有一个 B 错误仅为立体差异。完整匹配未出现不能证明内部忽略了 CoT，也不穷举部分结构采纳。

固定 GT SMILES 的平均 token 对数概率 H−N / H−A：DFM −0.001893 / −0.011506 nats，Chem-R −0.004225 / −0.007227 nats。每个对比包含全部 17 个 origin；各条件续接同一 GT token 序列，仅计 GT token，不含结束标签或 EOS。这是单一正确字符串的条件概率，不能等同于全部化学正确答案的概率，也不替换贪心准确率。

| 模型 | 指标 | A | B | C | E |
| --- | --- | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | 平均经验熵（bits） | 0.126802 | 0.118571 | 0.154268 | 0.156096 |
| Chem-R-8B | 平均经验熵（bits） | 0.707731 | 0.464940 | 0.455785 | 0.522414 |
| ChemDFM-R-14B | 逐题平均采样准确率 | 83.09% | 88.24% | 84.56% | 86.03% |
| Chem-R-8B | 逐题平均采样准确率 | 51.47% | 68.38% | 63.97% | 49.26% |

采样为 temperature=0.8、top_p=0.95；先计算每题 correct/8，再等权平均 17 题，不将每组 136 次采样视为独立问题，不做 best-of-eight 筛选。DFM 的 544 个采样均有效；Chem-R 的 A 有 7 个非法分子及 1 个缺失、C 有 2 个非法、E 有 3 个非法；所有采样均无截断，非法与缺失保留在分母。熵不使用 GT，采样准确率使用 GT；两者均为次要诊断。逐题与逐采样记录见[采样准确率](../Experiments/agent_corruption_v13/development18/review/sampled_accuracy.md)。

## v12 完整结果

[v12 生成汇总](../Experiments/agent_corruption_v12/development18/summary.json) 为 14/18 程序接受（add 6、delete 5、substitute 3），拒绝 4 个：3 个局部环境不可用，另 1 个候选选择无效/为空。独立构造审查覆盖全部 14 对、1,752 项检查，未发现程序检查失败。参考模型意见 9 agreed/5 disputed，符合性意见 2 agreed/12 disputed，同时 agreed 仅 1 个；14 个盲审 false 只表示模型未报疑。不得把程序接受称为一致语义认证。

运行于 2026-09-20 03:51:05 +08:00 以 status=0 结束。每个模型完成 70/70 条 ABCDE 答案、56/56 条 GT probe 和 56/56 条 ABCE 熵请求（每请求 8 次、每模型 448 个采样输出）。原始日志和自动报告已按字节核对并保存为 [v12 归档日志](../Experiments/agent_corruption_v12/development18/evaluation/run_agent_exp.log)、[v12 归档报告](../Experiments/agent_corruption_v12/development18/evaluation/agent_corruption_result.md)。

| 模型 | A | B：H | C：N | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | 11/14 | 12/14 | 12/14 | 11/14 | 12/14 |
| Chem-R-8B | 9/14 | 7/14 | 10/14 | 9/14 | 9/14 |

主组分与完整分子准确率计数相同，140 个答案全部可解析，无缺失或截断。ChemDFM B−A 为 +7.14 个百分点，转错 0/11，精确 McNemar p=1；Chem-R 为 −14.29 个百分点，转错 2/9（Wilson 95%：[6.32%, 54.74%]），p=0.5。Chem-R B−C 为 −21.43 个百分点，p=0.25。方向变化不是显著性结论；[v12 门槛记录](../Experiments/agent_corruption_v12/development18/v12briefgate.json) 保留失败决定，概率和采样诊断没有替换主门槛。

两模型 B 与完整 H 产物及任一有效非空根子集产物均为 0/14 匹配；去掉立体要求也无非空子集匹配。全部 64 个根子集组合中，62 个可执行、2 个因锚点被删而无效，均有记录。尤其 Chem-R 两个 A 正确→B 错误的样本仍具有正确产物连通性：add0172 的源中心 4 从 S 变 R，原未指定中心 27 被新增指定为 R；delete0078 的源中心 27 丢失 R 指定。它们在正式含立体指标下仍然算错，但不能称为采纳了错误片段/删除根。详见[部分计划与立体核查](../Experiments/agent_corruption_v12/development18/review/partial_plan_following.md)；此有限子集库不穷举任意部分子结构或其他错误，也不能证明内部忽略或修复机制。

GT 每 token 平均 logprob 的 H−N / H−A 差分别为 ChemDFM **+0.013088 / +0.009280 nats**、Chem-R **−0.022130 / −0.028577 nats**。这些值只度量同一固定 GT token 事件，不能当成所有化学正确答案的概率。

下表各格为“平均 canonical-SMILES 熵（bits） / 次要采样准确率”。准确率先按每个 origin 的 correct/8 计算，再在 14 个 origin 上等权平均；不把每组 112 个输出视为 112 个独立问题，不做逐采样 McNemar/Wilson，也不是 best-of-8：

| 模型 | A | B：H | C：N | E |
| --- | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | 0.068174 / 81.25% | 0.238597 / 84.82% | 0.077652 / 86.61% | 0.038826 / 86.61% |
| Chem-R-8B | 0.746833 / 57.14% | 0.619613 / 56.25% | 0.232373 / 81.25% | 0.323300 / 63.39% |

采样使用 temperature=0.8、top_p=0.95，与主实验 temperature=0 不同。ChemDFM 的采样 B−A 为 +3.57 个百分点、B−C 为 −1.79；Chem-R 为 −0.89、−25.00。Chem-R A 的 112 次输出中有 1 个非法分子和 1 个缺失答案，其余所有采样输出有效；全体无截断，非法/缺失均保留在分母。经验熵不使用 GT，次要准确率才使用 GT；低熵不等于正确，有限采样结果不替代贪心解码结论。逐 origin 的 correct/8 和逐采样判定见[次要采样准确率](../Experiments/agent_corruption_v12/development18/review/sampled_accuracy.md)。

v12 含 27 个结构根、0 个数值根；实际错误节点为 8–14，每种 tokenizer 的最小错误 token 数分别为 84、69。已执行 H−GT 产物 Tanimoto 均值为 0.3959，原始重原子编辑覆盖率均值为 35.29%。这些是有意选取较大结构差异后的构造幅度，不是自然错误分布，也不证明下游模型采用该计划。

与 v11 的同 origin 14 题对照，N 全部逐字相同。产物 Tanimoto 中位数从 0.7464 降为 0.3744，原图受影响比例中位数从 5.00% 增为 39.02%。13/14 题结构幅度增大；`substitute_v2.0272` 是保留的反例。字面 token 改动与整项语义值的错误 token 覆盖分别统计，不能混用。扩大幅度的程序与回归检查共 252 项通过，生成完成后实现与冻结快照仍一致。

## v11 已归档运行状态

依据 [v11 生成汇总](../Experiments/agent_corruption_v11/development18/summary.json)，18 个 development origin 均已终结：程序接受 15 个，拒绝 3 个，接受率为 15/18（83.3%）。15 对 N/H 对应 `pairs.jsonl` 中 30 条记录，不是 30 个独立问题。

| 子任务 | 原计划 | 程序接受 | 拒绝 |
| --- | ---: | ---: | ---: |
| add | 6 | 6 | 0 |
| delete | 6 | 5 | 1 |
| substitute | 6 | 4 | 2 |
| 合计 | 18 | 15 | 3 |

3 个拒绝均为 `local_environment_unavailable`：`delete_v2.0033`、`substitute_v2.0005`、`substitute_v2.0009`。原因是局部候选过短、超过大小限制或需在已定义立体中心处补氢，不能当作原题化学不成立。

**v11 的“接受”表示通过程序构造检查，不表示模型审查一致同意。** 15 个接受样本中，参考审查为 9 个 agreed、6 个 disputed；符合性审查为 4 个 agreed、11 个 disputed；两者均 agreed 的交集只有 3 个。原始意见及 `review_status` 全部保留，没有将分歧改写为通过，也不能把这 15 个样本称为经过一致语义认证的数据。盲审另存，不计入该两角色一致率。

两个评估目录各有 15 题、75 个 ABCDE 请求，全部完成；另各完成 60 个 A/B/C/E GT probe 请求。各有 45 条 B/C/E 完整提示词标签，30 条 A/D 因没有外供推理而记为 `no_supplied_trace`。15 个 E 均来自不同 origin、相同子任务。[独立化学审查](../Experiments/agent_corruption_v11/development18/review/independent_chemistry_review.json) 对 15 题的图、声明和标签完成 1,412 项检查，未发现阻断问题；另独立重建了 30 个 N/H 局部图。接受样本含 16 个结构根、14 个数值根，保留 9 个已定义 H 产物四面体中心。这些有界检查不等于所有化学语义的形式证明。

v11 主组分准确率如下，本批完整 normalized exact 计数相同。每格分母为全部 15 个程序接受 origin；150 个答案均可解析，无缺失或截断：

| 模型 | A | B：H | C：N | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| [ChemDFM-R-14B](../Experiments/agent_corruption_v11/development18/evaluation/chemdfm_r14b/summary.json) | 11/15 | 12/15 | 12/15 | 12/15 | 12/15 |
| [Chem-R-8B](../Experiments/agent_corruption_v11/development18/evaluation/chem_r8b/summary.json) | 8/15 | 10/15 | 12/15 | 10/15 | 9/15 |

两模型 B−A 分别为 +6.67、+13.33 个百分点，基线正确后转错分别为 0/11、0/8；Wilson 95% 区间分别为 [0%, 25.88%]、[0%, 32.44%]，精确双侧 McNemar p 分别为 1、0.5。**v11 开发门槛失败，未据此推进 heldout 132 题。** B 与执行 H 产物的完整/主组分匹配均为 0/15；该输出匹配诊断没有观察到采用完整错误计划的答案，但不能证明模型内部忽略或修复了 H。详见 [门槛记录](../Experiments/agent_corruption_v11/development18/v11briefgate.json) 与 [计划匹配核查](../Experiments/agent_corruption_v11/development18/review/plan_following.md)。

GT 每 token 平均 logprob 的 H−N / H−A 差分别为 ChemDFM −0.022193 / −0.026112 nats，Chem-R −0.025823 / −0.020392 nats。它们衡量固定 GT token 事件，不能把其下降改称为自由生成准确率下降。v11 未运行熵诊断。完整配对统计和概率结果保存在 [v11 归档报告](../Experiments/agent_corruption_v11/development18/evaluation/agent_corruption_result.md)，精确运行日志保存在 [v11 归档日志](../Experiments/agent_corruption_v11/development18/evaluation/run_agent_exp.log)；`scripts/` 下的当前日志和报告会供后续版本使用。

版本变化必须与覆盖率一起阅读：

| 版本 | development 接受/计划 | 已核实状态与限制 |
| --- | ---: | --- |
| [v9](../Experiments/agent_corruption_v9/development18/summary.json) | 10/18 | 无局部环境步骤；3 个参考审查拒绝、5 个符合性审查拒绝；两模型行为已完成 |
| [v10](../Experiments/agent_corruption_v10/development18/summary.json) | 2/18 | 增加 N/H 对称局部结构；3 个局部环境不可用、5 个参考审查拒绝、8 个符合性审查拒绝 |
| [v11](../Experiments/agent_corruption_v11/development18/summary.json) | 15/18 | 模型审查改为非否决诊断；两模型答案与 GT probe 完成，均 B>A，开发门槛失败 |
| [v12](../Experiments/agent_corruption_v12/development18/summary.json) | 14/18 | 扩大纯结构干预；两模型答案、GT probe 与熵完成，仅 Chem-R B<A，双模型开发门槛失败 |
| [v13](../Experiments/agent_corruption_v13/development18/summary.json) | 17/18 | 源状态重构后应用同一正确编辑；完整答案、GT probe 与熵已归档；两模型 B>A，开发门槛失败 |
| [v14](../Experiments/agent_corruption_v14/derivation_summary.json) | 每条件保留 14/18 | 固定 v12 计划的两种无局部窗口次序；280 个贪心答案完成，双模型方向均未达到，无 heldout |
| [v15](../Experiments/agent_corruption_v15/derivation_summary.json) | 每表达固定纳入 4/18 | 14 个范围外；命名关系原文表达与无名账本使用同一错误片段；80 个贪心答案完成，两模型 B 都由 2/4 升至 3/4，无完整 P′ 匹配，无 heldout |
| [v16](../Experiments/agent_corruption_v16/derivation_summary.json) | 每条件固定纳入 4/18 | 同一 native 原文末尾增加开放端口连接信息；80 答案完成，DFM B 不变、Chem-R B 降一题但 A 也变化；无完整 P′ 或已判定局部 H 连接采纳，无 heldout |
| [v17](../Experiments/agent_corruption_v17/derivation_summary.json) | 每条件固定纳入 4/18 | 同物理图仅 H 任务名称闭包，N 和 A/C/D/E 请求不变；80 答案完成，DFM B −1、Chem-R B +1；无完整 P′ 或已判定局部 H 匹配，无 heldout |

v10 审查记录中有解释承认“符合批准值/不是缺陷”却仍列入缺陷的输出。v11 因此明确改变了发布契约；接受率上升不是同一门槛下算法质量提高的证据。所有旧失败与快照仍在原目录，不能跨版本累加独立样本。

v9 的主组分准确率如下，完整 normalized exact 在这批结果中相同。每格分母均为 10 个接受 origin：

| 模型 | A | B：H | C：N | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| [ChemDFM-R-14B](../Experiments/agent_corruption_v9/development18/evaluation/chemdfm_r14b/summary.json) | 9/10 | 10/10 | 10/10 | 9/10 | 10/10 |
| [Chem-R-8B](../Experiments/agent_corruption_v9/development18/evaluation/chem_r8b/summary.json) | 5/10 | 5/10 | 9/10 | 6/10 | 6/10 |

两模型 B 输出与所执行 H 产物的完整/主组分匹配均为 0/10，见 [v9 计划匹配核查](../Experiments/agent_corruption_v9/development18/review/plan_following.md)。这说明尚未观察到输出采用 H 计划，不能推断内部推理机制。错误 token 数多也不保证行为受损。v10/v11 的策略修订发生在看到 v9 开发结果之后，v12 的幅度修订发生在看到 v11 开发结果之后；“不使用目标输出选择本题候选”不等于整个开发过程从未参考目标模型结果。

另有固定保留 v11 全部 15 题的纯数值根移除消融，仅作为 CPU 诊断材料保存于 `agent_corruption_v11/development18_structural_ablation`，没有运行 GPU。用户随后选择优先探索幅度更大的结构干预；不能把该消融与 v12 当作已完成的对照实验。

历史 `pilot3/summary.json` 已核对：v1–v3 各为 0/3，v4–v6 各为 1/3，v7 为 3/3，v8 为 2/3。它们复用开发题且修改协议，不能作为受控算法比较或累加为新样本。

## 数据与当前 v13 源状态协议

原题来自 `Dataset/raw_benchmark_data/mol_edit/*_pilot_origin.json`，原始推理和解析字段来自 `Dataset/process_evaluation_data/mol_edit/*_pilot_origin.json`，按 `anonymous_sample_id` 对齐。原始 instruction 与 indexed SMILES 保持不变；评估准备阶段再次与原始 benchmark 对照。修改的仅是外供 CoT，不把原题改成适合错误源图的任务。

| 字段 | 含义 |
| --- | --- |
| `N_raw` | 原始 `formal_cot_trace` 拼接文本，留作来源与审查证据 |
| `N_source_visible` | 从原始文本去掉明确产品答案子句后的历史来源视图 |
| `N_visible` | 当前协议确定性渲染、在构造 H 前冻结的规范化 N |

**公开 N 不是原始 N 的逐字副本。** v13 N 先说明真实源 S 的完整非映射 SMILES 与分子式，再从经验证的正确参考编辑 I 渲染步骤 1–3。H 使用同一模板，先错误地重构源 S′，随后正确地把同一个 I 应用于 S′，得到条件产物 P′；正确产物记为 P。`reference_projection=typed_reference_with_explicit_source_state_v1` 与生成协议号是不同字段。

当前流程如下：

1. 本地验证原始参考、片段与 FORMAL 数值，执行参考编辑 `S + I → P` 并核对完整 GT。`render_source_state_reference()` 不接收任何 H 候选；其文本、绑定、节点和事实先写入 `reference_render.json`。干净参考 Agent 只看原题、N 与正确工具事实。
2. 从真实 S 枚举含环的远端连通分支遗漏：只跨可切断的单桥键，保留至少一半活性组分重原子，保护参考编辑涉及的原子及其直接邻居，也不在保护原子上补氢。保留独立旁观组分。最多检查 128 个提议，返回至多 12 个已验证候选；S′ 必须满足连通组分、价态、氢与已指定立体信息的约束。这不是稳定可分离分子或自然误读频率的证明。
3. 每个候选执行两条路径：`S → S′ → P′` 和从 S 执行合并计划直接得到 P′。要求完整、含立体的产物一致，并验证保留源原子对应。可见 H 中的编辑 I 不变；额外遗漏仅属于前置源状态重构。
4. 程序核对六类错误值、条件算术、完整产品泄露、语义绑定及两套 token 标签。Poe `gpt-5.4-mini` 的选择角色只批准至多 6 个已有、可解释为漏看源取代基的候选，不创建新操作或读取目标模型输出。在批准的可行池内，按 P 与 P′ 的无手性 Morgan Tanimoto 最小者选定计划，平局按 candidate ID。`sampling_policy=max_graph_distance_among_agent_approved_feasible_source_states` 只指这个有限池，不是全化学空间的最远距离或 seed 均匀抽样。
5. 盲审只看原题与 H；符合性审查只看 H、批准的 after 值、同一编辑、绑定与标注，不接收 N、before 值、原题或盲审结论。它检查 H 是否符合批准的错误源状态，而不是把预期错误根本身重新判作额外缺陷。参考/符合性模型意见仍是非否决诊断；程序检查、传输或解析失败按终态保存。每个审查角色调用一次，不反复请求直至同意。

当前固定 **1 个 source-state 根**，默认至少 6 个不同错误节点、两种 tokenizer 各至少 40 个错误 token（`--min-nodes 6 --min-tokens 40`）。CLI 不提供 `--candidate-mode` 或 `--min-roots`；不添加虚假数值根补足密度。全部 17 条实际 H 恰有以下六类错误：

| 节点 | N 的真实值 | H 的错误值与来源 |
| --- | --- | --- |
| `source_graph` | S 的完整非映射 SMILES | S′ 的完整非映射 SMILES；唯一根 `r1` |
| `source_formula` | S 的分子式 | S′ 的分子式；传播自 `r1` |
| `source_heavy` | S 的重原子数 | S′ 的重原子数；传播自 `r1` |
| `source_rings` | S 的环数 | S′ 的环数；传播自 `r1` |
| `product_heavy` | P 的重原子数 | P′ 的重原子数；传播自 `r1` |
| `product_rings` | P 的环数 | P′ 的环数；传播自 `r1` |

这些值均相对于原题/正确参考实际错误，但在错误前提 S′ 下条件自洽。增加/删除片段与编辑增量仍按同一 I 计算；`heavy_delta`、`ring_delta` 的共同源遗漏抵消，值不变、不标错。`root_ids` 记录计算祖先，不代表每个后继都有非零边际因果效应。

**v13 的 N/H 都没有产品局部窗口或 Step 4。** 更小的产物规模可能迫使局部描述改用较小半径，却仍是正确局部描述；这种表示变化不应增加幻觉标签。该步骤在首次 v13 API 生成前对称移除，未修改冻结的 v12 模块或数据。

完整 S/S′ 作为源声明允许出现，完整 GT/P′ 不进入 N/H。S/S′ 与完整产物等价时拒绝；共享旁观离子不等于暴露完整答案。源重述引入了**完整源复制这一可能机制/混杂**。后续行为核查应分别匹配 S、S′、P 与 P′；准确率下降本身不能证明采纳了错误源状态，也不能仅凭完整源字符串出现就称其为答案泄露。

### 分阶段 atom map 与执行证据

| 证据 | 起点与终点 | map 的解释范围 |
| --- | --- | --- |
| `reference.json.execution` | S 经 I 得 P | 原始 S |
| `plan.source_plan` / `perceived_source_execution` | S 经遗漏得 S′ | 原始 S 的保留原子 |
| `plan.edit_plan` / `execution` | S′ 经原封不动的 I 得 P′ | 保存的 S′ 阶段 |
| `plan.joint_plan` / `joint_execution` | 原始 S 经合并操作得同一 P′ | 原始 S，用于源对应与 severity |

`accepted.json.source_contexts.N/H` 各保存源 SMILES、事实、编辑和执行。S′ 删除原来的较大 map 后，随后新增原子可能重用已经被删除的原始编号；相同数字不自动表示跨阶段同一个原子。实际 5 对（add0098、add0172、add0193、add0194、sub0009）存在这种重用，独立审计均验证了正确对应。比较原始源的保留/影响范围必须使用从 S 出发的 `joint_execution`，不能把 S′ 阶段的新 map 错认成原始 S 的存活原子。

所有角色目前使用同一个 Poe 模型，上下文隔离不等于错误统计独立。v13 是在观察到 v12 开发结果后修订的探索性诊断；构造本题时不读取目标输出，不意味着整个开发过程没有参考旧版本结果。历史 v12 severe 修改的是编辑中的结构变量，v9–v11 还曾添加数值声明根；这些历史干预与当前源状态机制应分别解释。

## 幅度指标与 v10–v12 局部结构历史

v12/v13 的 `plan.severity` 分开保存 `product_tanimoto`、`source_footprint_ratio` 与文本变化/错误标签比例，不把它们相加为一个分数。Tanimoto 比较完整产物、全部组分，去原子映射，使用无手性 Morgan 指纹（半径 2、2048 bits）；较低值仅表示该指纹下差异更大，不是目标模型答错概率。原始重原子编辑覆盖率记录保留、属性、原有原子间键或新组分连接发生变化的原始重原子占比；不是 MCS，也不按根数加权。CIP 属性差异可能来自配体优先级变化，不能直接叫作立体反转。

独立 H 的 token 比例在最终分词标注后写入冻结计划：语义错误 token 比例与字符插入/替换覆盖的 token 比例含义不同，后者只度量字面改变；删除的 N 字符另计。评估阶段仍须重新在真实完整提示词上分词投影。缺失值不能补成零，报告应给出每项覆盖分母。选择结构差异较大的合格 H 是经过有意筛选的严重干预分布，不拟合自然错误率，也不表示自然模型经常产生此类错误。

以下局部环境规则仅解释 v10–v12 归档，v13 不渲染该窗口。v10–v12 对 N 与 H 分别从各自执行产物提取局部环境。中心优先取保留的添加锚点，否则取删除集合的原图保留邻居；按固定规则选择最小 map。搜索半径为 1–3，再递归补全相交的完整环和跨边界多重键。只允许在单键边界补氢，且不能在已定义立体中心处作有歧义的补氢。候选必须连通、无自由基、可由 RDKit 解析、至少 6 个 SMILES 字符，至多 16 个重原子且不超过实际产物重原子数的一半；取合格候选中重原子最多者。

程序还排除与完整正确产物、H 执行产物或其任一完整组分等价的片段。GT 只用于程序检查，完整 GT/H 产物 SMILES 不进入 N/H 文本。完整图、局部 atom maps、半径和 `cap_hydrogens` 保存在节点的 `execution_evidence`。此前切断 S=O 或 C≡N 会产生误导性投影；v10 的 13 项局部模块测试覆盖了完整多重键保留及反例。

文本明确称其为 **hydrogen-capped standalone representation**，不是完整基团、稳定试剂或真实反应中间体。例如 add0287 的 `[SH](=O)=O` 来自切断单键 S–C 后补 H，两个 S=O 均保留；长链可被截短，片段外 N-methyl 也可被 H 代替。不能据此把局部短链解读为整条原始链，也不能只从局部片段反推完整编辑。

v11 的 15 对中有 8 对半径不同、9 对中心不同。这是按各自计划独立选择的结果，意味着 N/H 对比同时改变局部覆盖范围和长度，并非只替换同一固定邻域中的少数原子。整个改变的局部 SMILES 作为一个语义节点标注；其中的字符不都对应独立错误原子。这一混杂限制应与行为结果一并报告。

## 当前生成命令与数据划分

以下命令从仓库根目录运行，使用现有环境。联网生成前，在同一交互式 shell 执行用户已有的 `poe` 命令；程序从环境读取 `POE_API_KEY`，不打印密钥或 alias 定义。v13 通过 `spawn` 的 `ProcessPoolExecutor` 隔离 RDKit 工作，当前批次为 3 个进程；生成只加载本地 tokenizer，不加载目标模型权重。

```bash
cd /mnt_nas1/haoqian/Data/Molecule
source /home/haoqian/.venvs/chemllm-outcome/bin/activate
poe

python Pilot/scripts/generate_source_agent_pairs.py \
  --output Pilot/Experiments/agent_corruption_v13/development18 \
  --split development --workers 3 --seed 20260920 \
  --min-nodes 6 --min-tokens 40 --offline
```

`--offline` 仅生成/检查 manifest，不调用 Poe、不构造样本。对相同配置去掉该参数即可执行生成或读取已有终态：

```bash
python Pilot/scripts/generate_source_agent_pairs.py \
  --output Pilot/Experiments/agent_corruption_v13/development18 \
  --split development --workers 3 --seed 20260920 \
  --min-nodes 6 --min-tokens 40
```

上述目录的 18 题已全部终结；相同命令不会重新选择已终态计划。新策略须使用新版本/目录，不能覆盖这一批。需要独立小批次时，可对新目录使用 `--limit 3`；它按任务轮流取样，包含三类任务各一个。不能把同一目录的 `pilot3` manifest 原地扩成 18 题。

默认 seed 下，add/delete/substitute 各 6 个进入 development，共 18 个；其余 132 个进入 heldout。同一 origin 的所有版本留在同一侧。开发批次目标为两模型在全部接受 origin 上均 B<A，这是探索性扩展门槛，不是显著性结论或样本接收规则。通过后才冻结策略并评估全体 132 个 heldout origin，继续保留失败分母；未通过则记录失败并修订假设。`--split all` 不能绕过实验隔离，若依据 heldout 结果继续调整，该集合就不再是未见测试集。

输入、阈值、角色提示、协议、完整源码或 tokenizer 快照变化时须新建目录。`worker_config.json` 还冻结进程模型、进程数与 tokenizer 并发设置；已有目录不能随意换 `--workers`。每个 origin 的 seed 是基础 seed 加有序列表位置；当前最终选择按批准池内的指纹距离排序。跨新目录的 Poe 批准输出不保证确定复现，相同 seed 不保证重生成相同计划。

v13 每个 origin 的 Poe 客户端预算上限为 **6 次实际请求**，正常接受记录使用干净参考、选择、盲审和符合性四个角色。请求发送前持久化，超时和失败也占预算。终态 `rejected.json` 在普通续跑中直接返回，不静默重试到成功或换计划。

旧 v12 生成入口和归档 launcher 保留用于核查：

```bash
# 历史 v12 配置；不是当前 v13 命令。
python Pilot/scripts/generate_agent_pairs.py \
  --output Pilot/Experiments/agent_corruption_v12/development18 \
  --split development --workers 3 --seed 20260920 \
  --candidate-mode severe --min-roots 1 --min-nodes 6 --min-tokens 40
```

v12 当次评估脚本的精确副本见 [v12 归档 launcher](../Experiments/agent_corruption_v12/development18/evaluation/run_agent_exp.sh)。当前 `Pilot/scripts/run_agent_exp.sh` 不再指向 v12；源码快照不匹配时，旧目录会拒绝续跑，不应为绕过检查而重写归档。

## 输出、错误定位与复现

| 路径，相对于生成输出目录 | 内容 |
| --- | --- |
| `manifest.json` | origin 划分、选中题目、协议、阈值和基础 seed |
| `tokenizer_snapshots.json` | 两个 tokenizer 的完整 backend、chat template、特殊 token 配置及类名 |
| `implementation_snapshot.json` | `modules/agent_generation/*.py` 与当前 `generate_source_agent_pairs.py` 的完整源码快照 |
| `worker_config.json` | spawn 进程池、进程数与 tokenizer 并发设置 |
| `origins/<origin_id>/input.json` | 本题完整输入、角色提示及生成配置 |
| `reference_checks.json`、`reference.json`，位于上述 origin 目录 | 参考事实检查、参考编辑和执行证据 |
| `reference_render.json`，位于上述 origin 目录 | 在 H 候选出现前冻结的 N、语义绑定、节点和真实源事实 |
| `candidate_pool.json`、`candidate_selection.json`、`fixed_plan.json`，位于上述 origin 目录 | 全部枚举候选、Agent 选择、候选拒绝原因、可行池及冻结计划 |
| `audits.json`，位于上述 origin 目录 | 盲审与受控干预合规审查 |
| `accepted.json` 或 `rejected.json`，位于上述 origin 目录 | 本题终态；失败有 `error_type` 和 `reason`；接受含 `release_contract`、`review_status` 与原审查意见 |
| `plan_calls/calls.jsonl`，位于上述 origin 目录 | 实际尝试日志，包括失败请求的预算占用 |
| `plan_calls/<request_id>.json` / `*.invalid.json` | 完整请求、响应、实际返回的 model/usage；后者保留无效 JSON 响应 |
| `pairs.jsonl` | 仅包含接受样本的公开 N/H 配对 |
| `summary.json` | 全部计划题目的处理、接受、拒绝状态 |

干净参考 Agent 审查保存在 `plan_calls/clean_reference.json`；接受记录另含 `reference_review`，没有单独的 `reference_review.json` 文件。失败可能发生在建池或选定计划之前，因此后续文件不保证存在。当前失败类型并非统一枚举：同时检查 `error_type` 和 `reason`。v13 包括 `no_feasible_source_state`、`invalid_or_empty_source_selection`、`source_state_invalid_execution`、`source_state_answer_leakage`、`insufficient_error_density` 和 `conditional_claims_invalid`；候选内失败存于 `candidate_pool.json.rejected`。v11/v12 历史程序失败包括 `reference_invalid`、`local_environment_unavailable`、`no_feasible_plan`、`renderer_claims_invalid`、`answer_leakage` 等；历史 `reference_semantic_uncertain` 和 `audit_unresolved_defects` 曾是拒绝原因，v11 模型意见分歧改存 `review_status`；预算或 HTTP 问题也保留实际错误类型。不要删除失败记录来提高接受率。

协议没有新增 SHA 字段、摘要缓存或 SHA 完整性依赖：续跑直接比较完整 JSON 请求、配置、tokenizer 和模块源码快照。评估另冻结 manifest、requests、GT 与提示词标签。实际模型文件仍应保持原样，不要在相同路径替换权重或 tokenizer 后混用旧结果。费用只依据接口实际返回的 usage 等记录；没有返回的信息不估作已知成本。

报告应同时给出计划总数、处理数、各阶段拒绝原因、程序接受数及审查一致/分歧分布。不能只展示模型审查 agreed 的样本而隐去 disputed 的程序接受样本；任何分层结果都须给出自己的分母。模型行为结果是**接受子集上的条件结果**；拒绝样本没有 H，不能假造其准确率，也不能从总体分母中隐去这些失败。产物改变、Agent 合理性筛选、错误节点数和 token 门槛都会影响接受分布。

## 字符标签、token 标签和语义绑定

字符偏移统一为 Python Unicode 字符的半开区间 `[start, end)`，不是 UTF-8 字节位置。`root_error` 与 `propagated_error` 带 `node_id` 和可包含多个根的 `root_ids`。确定性 renderer 的每个变量槽还有 `bindings`，记录其来自批准节点还是已执行计划，以及该槽是否实际标为错误。依赖片段表示顺序的连接原子序号可在 N/H 中变化，但本身仍是局部正确坐标；这种情况明确记录语义说明，不因此添加独立根错误。

**标签以干净参考的语义值为基准。** v13 的六类已标注值还逐项核对了真实 S/P 与错误 S′/P′：它们对原题确实错误，同时可在 S′ 条件下算术自洽。完整源 SMILES 是一个根语义值，不能把其中保留的原子、字符或标点都解释成独立错误。一个根的多次后继提及不增加根数；抵消后的不变值不标错，`root_ids` 仍可保留计算祖先。

历史 v11/v12 的错选片段组成、局部环境 SMILES 或移除集合也按整项语义值标注；其中条件正确的片段描述可能偏离干净参考，不能将这些旧标签泛化为任意自然文本的绝对事实真假标签。v13 已去掉产品局部窗口，避免把仍正确但范围改变的局部表示新增为错误节点。

`unchanged` 表示 token 中有字符位于错误标注之外，**不保证这些字符逐字等于原始 N，也不表示语义已经被全面证明正确**。一个 token 可以同时携带 root、propagated 和 unchanged 标签。重复提及同一错误仍保留全部字符/token 跨度。

生成目录中的 `token_labels` 对应独立 CoT 文本，用于生成门槛；ChemDFM 与 Chem-R 各用自己的 tokenizer，数量和边界不能互换。这些是程序按同一语义标注投影的标签，不是两份独立人工标注。评估目录的 `prompt_token_labels.jsonl` 才对应完整 chat template、问题和 assistant prefix 拼接后的真实提示词；它记录完整 `prompt_token_ids`、推理字符区间以及该区间内的 token ID、文本、偏移和标签。边界 token 可以同时覆盖推理和相邻控制字符，不能把独立 CoT 的 token 下标平移后当作实际输入标签。

N 与 E 不携带注入 H 的错误标签。E 是另一题的 N；没有注入错误标签不表示它适用于当前问题。H/N 标签、计划、审计和绑定信息均放在 sidecar。兼容配对文件中的 `detector_input.final_answer` 是评分用 GT，评估器不会把它或整个 `detector_input` 对象直接交给模型。

## A/B/C/D/E 评估

默认 `--placement prefix --groups ABCE`：A/B/C/E 使用相同 system 与原题。A 的 think 内容为空；B 放 H；C 放规范化 N；E 放另一 origin 的 N。`--groups ABCDE` 另增加 D，让模型自行生成推理。E 在完整输入数据集上确定性选取不同 origin，优先相同子任务，至少需要两个不同 origin；E 与目标文本没有长度匹配。

当前批量入口为 `bash Pilot/scripts/run_agent_exp.sh`，不接收命令行参数，固定指向 `agent_corruption_v13/development18`，并要求 manifest 协议为 `agent_source_state_v1`、生成全部终结且至少有两个接受 origin。实际配置以脚本为准：GPU 4–7、TP=4、显存比例 0.72、batch size=8，依次运行 ChemDFM、Chem-R 的 ABCDE 答案，再运行两模型 GT probe 和 n=8 熵诊断（`RUN_ENTROPY=1`、`ENTROPY_SAMPLES=8`）。v13 的这些阶段均已实测完成并归档；v17 四题诊断使用独立的 `run_task_intent_exp.sh`，不改变此 v13 主入口。

后台 worker 继承专用锁，父进程等待 PID 就绪；显存不足时等待，不终止外部作业。日志、PID 和当前自动报告在 `Pilot/scripts/`，后续启动会更新这些当前位置。旧 v12 的日志、报告与 launcher 已另存归档；不要用当前文件冒充旧运行证据。已有工作运行时不要重复启动同一目录的 GPU 作业；脚本不会自动生成 heldout 数据。

下面以已审查且非空的 `development18/pairs.jsonl` 为手动输入示例，输出路径与批量入口一致。生成成功前不要把空配对文件当成可评估数据。

```bash
python Pilot/scripts/evaluate_agent_corruption.py prepare \
  --dataset Pilot/Experiments/agent_corruption_v13/development18/pairs.jsonl \
  --annotations Pilot/Experiments/agent_corruption_v13/development18 \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B \
  --placement prefix --groups ABCDE \
  --tensor-parallel-size 4 --batch-size 8 --gpu-memory-utilization 0.72

CUDA_VISIBLE_DEVICES=4,5,6,7 python Pilot/scripts/evaluate_agent_corruption.py run \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B

python Pilot/scripts/evaluate_agent_corruption.py report \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B
```

Chem-R 使用同样三条命令，将 `--model` 换成 `Chem-R-8B`、输出目录换成 `development18/evaluation/chem_r8b`，并指定 `--model-path chemical_models/Chem-R-8B`。同一组 GPU 上依次运行两个模型。ChemDFM 默认权重路径为 `/mnt_nas1/shared/ChemDFM-R-14B`，Chem-R 为仓库下 `chemical_models/Chem-R-8B`；可以用 `--model-path` 指定路径，但 prepare/run/report 必须一致。

准备阶段不加载模型权重，会读取本地 tokenizer 生成完整提示词标签；若未传 `--annotations`，尝试读取 dataset 同级目录下的 `origins/<origin_id>/accepted.json`。检查 manifest 的 `prompt_label_status`，不能把 `unavailable` 记录解释成已完成标注。`prepare` 要求新目录；断点续跑重复 `run`，`run --limit 4` 仅处理至多 4 个未完成请求。`report --allow-partial` 允许临时统计，但不构成完整实验结论。

默认行为解码为 temperature=0、top_p=1、repetition_penalty=1.05、最多 2048 个输出 token。`--placement supplied` 是另一个实验：B/C/E 把推理放到用户消息并使用要求遵循推理的 system 提示；应保存到新目录，不能与 prefix 结果混报。

`predictions.jsonl` 保存模型输出和请求，`outcome_records.jsonl` 保存抽取答案与分数，`summary.json`/`summary.csv` 保存汇总。缺失或非法答案计为错误；达到长度上限的响应保留并统计为截断。主指标是去 atom map 后的主组分匹配，可能忽略辅助组分；同时报告完整分子的 normalized exact 匹配。FTS 是不编码手性的 Morgan 指纹相似度，不能替代立体化学正确性判断。B−A、B−C 的准确率差使用同题配对；flip-to-wrong 的分母是相应基线答对的题数，区间为 Wilson 95%，显著性为精确双侧 McNemar 检验。

若同一 origin 有多个变体，pair 不能自动视为独立样本；当前报告会标识 repeated origins，此时 pair 级区间和检验仅作描述。开发集上追求“两模型均 B<A”是可修订策略的目标，不是数据质量接收规则，也不是预先保证的结论。最终应在冻结策略后的 heldout 上报告真实结果。

## 概率与熵诊断

`probe` 对原始 GT SMILES 的固定 token 序列做 teacher forcing。前缀与答案分别分词后按 token ID 拼接，保证 A/B/C/E 使用相同答案事件，并避免字符边界分词改变被评分的答案 token。结果单位是自然对数概率，包含 GT token 的和与均值，不包括闭合 answer 标签或 EOS，也不包含 D。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python Pilot/scripts/evaluate_agent_corruption.py probe \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B

python Pilot/scripts/evaluate_agent_corruption.py probe-report \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B

CUDA_VISIBLE_DEVICES=4,5,6,7 python Pilot/scripts/evaluate_agent_corruption.py entropy \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B --entropy-samples 8

python Pilot/scripts/evaluate_agent_corruption.py entropy-report \
  --output Pilot/Experiments/agent_corruption_v13/development18/evaluation/chemdfm_r14b \
  --model ChemDFM-R-14B --model-path chemical_models/ChemDFM-R-14B --entropy-samples 8
```

`probe_config.json`、`probe_records.jsonl`、`probe_summary.json` 与行为输出分开。GT log-probability 只衡量**一个固定 GT 字符串所对应 token 序列**的条件似然，不是全部化学等价正确答案的总概率，也不替代自由生成后的准确率或化学等价性。概率下降不能单独证明模型最终会答错。

熵诊断默认为每个 A/B/C/E 提示词抽样 8 次，temperature=0.8、top_p=0.95，其他采样参数继承冻结 manifest。`entropy_config.json`、`entropy_records.jsonl`、`entropy_summary.json` 独立保存并可续跑。按去 atom map 的完整 canonical SMILES 聚类；非法答案放入同一 invalid 类，缺失答案另成 missing 类，所有抽样都保留在分母，另报告有效 coverage。它是有限样本的经验熵，不是精确分布熵；低熵也可能意味着稳定地答错。

梯度归因、注意力分析和训练功能尚未实现，不能把概率、熵或字符标签描述成这些分析的结果。

## 本地检查

```bash
cd /mnt_nas1/haoqian/Data/Molecule/Pilot
/home/haoqian/.venvs/chemllm-outcome/bin/python -m pytest \
  tests/test_agent_chemistry_tools.py tests/test_agent_candidate_tools.py \
  tests/test_agent_quality.py tests/test_agent_orchestrator.py \
  tests/test_agent_renderer.py tests/test_agent_local_environment.py \
  tests/test_agent_labels_metrics.py \
  tests/test_agent_severity.py tests/test_agent_severe_candidates.py \
  tests/test_agent_source_state_candidates.py tests/test_agent_source_state_renderer.py \
  tests/test_agent_source_state_orchestrator.py \
  tests/test_agent_evaluation.py tests/test_agent_launcher.py tests/test_agent_report.py \
  tests/test_chemllm_outcome.py -q
```

验证记录分开保存：v9 检查点为 191 项；v11 实现检查点为 214 项，之后 launcher/harness 修改仍需各自新检查。不得把旧检查数当成所有后续改动已验证。

本地测试验证实现契约与已覆盖样例，不替代接受样本的化学抽检，也不证明目标模型的行为目标已经达到。
