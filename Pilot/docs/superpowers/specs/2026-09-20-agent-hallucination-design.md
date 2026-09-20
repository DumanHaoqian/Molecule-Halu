# 基于 Agent 的 N→H 化学推理幻觉构造方案

日期：2026-09-20。状态：保留初始方案及版本修订；生成器与评估器已实现，v9/v11 已完成开发评估且未达行为目标。当前 v12 配置扩大纯结构干预，本文不声称已获得 v12 目标模型、熵或 heldout 结果。实际操作以 [工作流](../../agent_corruption_workflow.md) 与 [实施记录](../plans/2026-09-20-agent-corruption-implementation.md) 为准。

## 当前 v12 契约与已完成证据

v11 完成全部 15 个程序接受 origin 的两模型 ABCDE 评估，各 75 条答案和 60 条 GT probe。主组分与完整分子准确率计数在本批相同：

| 模型 | A | B：H | C：N | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChemDFM-R-14B | 11/15 | 12/15 | 12/15 | 12/15 | 12/15 |
| Chem-R-8B | 8/15 | 10/15 | 12/15 | 10/15 | 9/15 |

两模型均 B>A，严格的“双模型 B<A”开发门槛失败。B 与实际执行 H 产物的完整/主组分匹配均为 0/15；零输出匹配不证明内部忽略或修复机制。结果、概率诊断、准确日志与门槛保存在 [v11 报告](../../../Experiments/agent_corruption_v11/development18/evaluation/agent_corruption_result.md)、[日志](../../../Experiments/agent_corruption_v11/development18/evaluation/run_agent_exp.log)、[计划匹配](../../../Experiments/agent_corruption_v11/development18/review/plan_following.md) 和 [门槛记录](../../../Experiments/agent_corruption_v11/development18/v11briefgate.json)。没有推进 heldout 132 题；v11 没有熵结果。

v12 的当前构造契约如下，覆盖下方初始方案中已被修订的部分：

- 原始 instruction、indexed SMILES、GT 与 N_raw 不变。公开 N 是从经检查的语义字段与参考执行图确定性渲染的规范化参考，不是原始 N 的逐字拷贝；N/H 同模板及局部投影规则，完整正确/H 产物只用于内部核查。
- 默认 `--candidate-mode severe --min-roots 1 --min-nodes 6 --min-tokens 40`。候选可组合 1–3 个结构维度：片段、锚点、移除集合；多个分支的一个移除集合仍为一个根。不再用数值 +1 填充根数。阈值是构造条件，不是自然错误分布的估计，具体批次以冻结 manifest 为准。
- 最多检查 768 个不同提议，返回至多 36 个图可执行候选；经文本、标签、密度、泄漏等程序检查后，Agent 批准至多 6 个。控制器选择其中实际 H 与 GT 等价参考产物的无手性 Morgan Tanimoto 最低者，平局按候选 ID。每个 origin 只发布一个 H；没有按目标模型结果选择候选或删题。
- `max_graph_distance_among_agent_approved_feasible_candidates` 仅描述有限批准池中的指纹距离选择，不是全局化学图编辑距离最大化。`plan.severity` 分开记录完整产物 Tanimoto、原始重原子编辑覆盖率及文本与 token 比例，根类型从 `plan.roots` 另行统计。数值范围与缺失覆盖必须分别报告，不合成按根数加权的分数。选择严重的结构变化不代表自然模型经常产生这些错误。
- v11/v12 发布契约为程序验证构造：参考执行等价于 GT、可执行 H、声明/标签一致、密度及完整答案泄漏检查必须通过。参考/符合性模型审查为单次非否决诊断，保留 agreed/disputed 与原始意见；程序接受不等于 Agent 一致认可，未解决意见不改成通过。盲审 false 也不证明无泄漏。
- 文本由确定性 renderer 生成，不让 Agent 自由写文或反复投票到同意。参考审查、候选选择、盲审、符合性审查各为隔离上下文，使用同一 Poe 模型不构成统计独立验证。每题实际 HTTP 请求预算仍为 10，失败也计入并持久化；终态拒绝不静默补采。
- A/B/C/E 保持相同原题和 system，H/N 放在 assistant 原生 think 前缀；D 自行推理。引擎和答案解码设置保持原样。v12 launcher 默认 GPU 4–7、TP4、显存比例 0.72，顺序运行两模型答案、两模型 GT probes，再两模型 ABCE 熵（n=8、temperature=0.8、top_p=0.95）。熵已启用但本文检查点尚未运行；配置不等于结果。
- 18 题 development / 132 题 heldout 划分保持不变。v12 是看到 v11 开发失败后的全局策略修订；不使用目标输出选择本题候选，不代表开发过程从未查看目标结果。策略冻结及门槛决定前不把 heldout 用于调参。

## 初始方案存档（以下第 1–12 节）

以下保留最初的备选设计与动机，不作为当前执行说明。截断 Poisson / fixed-K 抽样、Agent 自由最小改写、模型意见的一票否决、固定计划自动修复、每题 K=1/K=2 双计划等并未按此稿用于 v12；当前实现由上方契约及版本快照定义。下文“本轮”“第一版”“拟实施”等均指初始设计检查点。

目标：以原始正确推理 N 为不可变来源，按明确采样计划引入根错误，执行受影响的化学操作和后续推导，生成可追溯的 H。化学构造与独立校验均使用 Poe bot `gpt-5.4-mini`。程序拥有采样、工具执行、校验和发布的最终控制权。

本轮已在交互式 bash 执行用户提供的 `poe`，确认 `POE_API_KEY` 非空；未输出其值。这只验证环境加载，不代表已验证 API 额度或模型调用成功。

## 1. 方案选择与范围

推荐“程序控制器 + 化学构造 Agent + 独立校验 Agent + 本地化学工具”。两个 Agent 是不同上下文和任务契约，可以使用同一个模型；其判断不视为统计独立证据。

纯规则方案适合计数和显式字段，但难以覆盖自然语言化学操作。纯 Agent 自由改写难以固定根错误数量和范围。本方案保留规则的可控性，让 Agent 负责结构化化学理解、有限候选提议和最小文本改写。

第一版范围是能够表示成有限原子/键编辑的 add、delete、substitute 推理，生成“错误前提下可执行、后续自洽”的 H。无法执行、歧义过大或化学语义未获确认的样本记录为失败，不冒充已经验证。这个范围不是对所有自然幻觉的覆盖。

不以诱导 ChemDFM/Chem-R 答错作为接收条件。不为了得到 B<C 的结果挑选 H。H 可以包含真实错误而最终产物未变，必须分类报告。

## 2. 对现有代码的复用与必要调整

- 复用 `modules/ingestion/` 和 `modules/reference/builder.py` 的原始 CoT、字段绑定、Reference DAG、语义节点及来源信息。原始成功标记是筛选线索，不能替代 N 的化学审查。
- 复用 `modules/error_planning/fragment_pool.py` 的候选片段，以及可用的局部候选枚举。
- 复用 `modules/text_realization/segments.py` 的结构化文本片段编译和本地 FORMAL 拼装；复用 occurrence/arithmetic/enumeration 审计与脱敏诊断。
- 现有 `error_injection/propagation.py` 主要传播计数和重复字段。新版本必须增加“锚点/片段/操作 → 实际产品图 → 描述符”的正向执行，不能用产品原子数倒推片段原子数来代替化学操作。
- 现有 `text_realization/pairing.py` 从 H 恢复 N。新主流程必须改为冻结原始 N，单向派生 H；不允许通过生成 H 后再重写 N 来满足配对。
- 现有 planner 禁止传播范围交叠，`PropagationEvent` 只记一个根来源。新方案允许多个根共享后代：统一应用根干预，拓扑计算后代一次，记录多个 `root_mutation_ids`。
- 旧流程/旧实验结果保留，新增独立的生成协议版本和输出目录，避免在同一 JSONL 混用语义。
- 新协议、ID、缓存、检查与输出不增加 SHA 依赖；不直接沿用旧 Poe 封装中基于摘要的缓存。复用其传输逻辑，缓存使用持久化请求 ID，并逐字段比较完整请求与版本。

## 3. 数据流与角色

```text
原始 N 和原题
  → 冻结 N，审查参考状态和语义依赖
  → 构造受约束的可用 corruption 候选集
  → 程序采样并冻结根错误计划
  → 化学构造 Agent 生成结构化编辑提议
  → 本地工具统一执行，正向计算所有受影响节点
  → 化学构造 Agent 按已批准状态最小改写 N
  → 本地硬校验
  → 独立校验 Agent：盲审、再核对计划
  → 接收 / 固定计划下修复 / 拒绝
  → 发布 N/H 配对与单独的审计数据
```

### 程序控制器

控制 seed、根错误数量/位置/类型/替代值、请求次数、工具许可、状态迁移和接收条件。Agent 不能自行加根错误、降低 K、改原题、变更 N 或把失败计划替换成更容易的计划。

### 化学构造 Agent

使用 `gpt-5.4-mini`，建议 temperature=0.2（实验配置，非真实性保证）。分三个任务上下文执行：

1. 原题/参考 N 不够结构化时，提出节点、依赖边、原子映射和有限候选；候选由工具检查后才能进入采样池。
2. 收到冻结计划后，提出原子/键编辑及受影响节点。不得直接以一条自己生成的产品 SMILES 代替编辑过程。
3. 收到程序执行后的状态，修改受影响的推理句子和引用；无关文本本地复制。文本不能补出隐藏标准答案、完整目标产物或额外根错误。

生成上下文默认不额外提供标准答案。原始 N 已经含有的标准答案提示应由参考审查标记；不能假装没有这类信息。

### 独立校验 Agent

使用新的 `gpt-5.4-mini` 会话，建议 temperature=0。两个阶段隔离上下文：

1. 盲审：只给原题、H 和必要工具访问，不给采样计划、生成 Agent 的自评或目标错误数。报告实际观察到的错误、证据跨度和不确定项，避免只复述计划。
2. 对照审查：提供冻结 N、计划、工具执行结果和盲审报告，判断根错误是否实现、传播是否一致、是否修正了指定错误、是否新添错误、是否存在过期的正确/错误描述。

返回 `pass`、`fail` 或 `uncertain`，每个结论附节点/文本证据。校验任务是验证“按计划构造错误”，不是把 H 修成正确推理。`uncertain` 进入人工审查，不自动通过。

## 4. 错误采样模型

根错误定义在语义变量或操作上，文本中多次提及同一个错误仍只算一个根错误。传播导致的错误和改写跨度单独计数。

初始可配置基线：

```text
count_distribution = truncated_poisson
lambda = 1.0
K ∈ {1, ..., min(3, 本题可实现的联合根计划容量)}
P(K=k) 与 lambda^k / k! 成正比，在允许集合上归一化
```

lambda=1.0 和上限 3 是明确的工程起点，未经自然 CoT 校准，不称为 LLM 的真实错误分布。不采用把超限样本强行裁成上限的方式。还提供 fixed-K 模式用于调试和分层对照。

候选类型分开统计：锚点/位点混淆、加入/移除片段混淆、成键/断键操作错误、计数/验证错误。初始按可用类型均匀分配权重，再按类型内可用候选采样；权重和实际条件概率写入计划。物理/化学不可执行的候选不进入第一版的可执行子集。

两个根是否冲突由联合约束决定：同一字段要求两个不兼容值应拒绝；共同影响 product 并不构成冲突。根覆盖祖先/后代导致无法区分新增错误来源时，应排除或由明确的操作顺序定义，不能无记录地覆盖。

先建立确定的可行候选集，再采样；Agent 候选提议自身会造成选择偏差，也必须记录候选集和拒绝原因。采样后失败不能静默换计划。每个重新采样的计划分配新 ID，并保留前一个失败计划。

诊断阶段可均衡采样错误类型，以获得覆盖；这应标记为诊断分布，不能与自然分布模拟混报。

## 5. 工具与化学传播

Poe 调用沿用仓库现有 `fastapi_poe.get_bot_response_sync` 传输形式。Agent 输出符合本地 schema 的 JSON action；Python 校验、执行白名单工具，再将结果传回模型。不要求 Poe 原生 function calling，不允许 Agent 输出并执行任意 Python 或 shell。

| 工具 | 输入/输出与约束 |
| --- | --- |
| inspect_source | 读取 indexed SMILES，返回原子映射、元素、键、组分、立体信息 |
| describe_fragment | 解析候选片段，返回有效性、连接原子、重原子数、形式电荷等 |
| enumerate_edit_candidates | 基于任务、参考状态和已有池枚举有限编辑候选；返回候选 ID 和不可执行原因 |
| apply_edit_plan | 在原分子图上应用全部已批准编辑，返回中间图、产物和逐项执行证据 |
| describe_product | 从执行产物计算计数、环数等；不接收 Agent 自报数字作为真值 |
| compare_molecules | 规范化后比较等价性、立体化学和组分；用于原始正确产物与错误产物比较 |
| audit_dependencies | 检查每个派生字段是否等于其父节点在当前状态下的正确计算 |

编辑表示必须包含 atom-map ID、bond endpoint、bond order、片段连接原子、移除原子集和组分保留规则。新增原子映射由程序分配。删除操作需明确边界和保留组分，不能简单保留“最大碎片”而丢失无关组分。

先执行所有根干预，再拓扑重算后代；一个节点可有多个根来源。所有重算写入 `node_evaluations`，即使多根影响抵消后结果与 N 相同，也保留计算路径和根来源。仅实际变化的节点产生文本变更事件，并非每个后代都必须改变。枚举在一定限额内仍有多个合理结果时返回 `ambiguous_edit`，不自动选最能降低目标模型准确率的产品。

必须分离 `molecular_state`（程序实际执行的分子图和描述符）与 `claim_state`（CoT 声称的值）。结构类根错误改变前者，再推导后者；计数类根错误只改变指定 claim，不能反向修改分子来配合错误数字。例如产品实际有 6 个重原子，却声称有 5 个，是一个计数根错误；基于 5 计算的差值可以是传播错误。工具仍记录实际值 6，审计把差异定位到已指定的根，不能强行覆盖成 6，也不能把正确执行的减法算作新增根错误。

执行顺序上先冻结 root 的 claim 值；拓扑重算跳过这些干预节点原本的赋值公式，只对非 root 后代执行当前 claim 父节点对应的规则。`molecular_state` 的事实计算独立进行，根处预期的事实差异记录为干预证据，不能借一致性检查覆盖根值。

任何关键边没有实现可用检查时，返回 `unknown`，不能把缺少检查当成 pass。预期的事实违背和非预期的条件推导错误分开审计，不能通过关闭全部一致性检查来容纳 H。

RDKit sanitization 和计数检查只能证明结构形式与程序性质，不能独立证明指令的全部化学语义。化学操作和意图匹配仍需独立审查。

## 6. 一个完整示例

```text
原分子：乙醇，CCO
原指令：用丙酰基酰化羟基
N 的片段：C(=O)CC
采样根错误：把丙酰基误识别成乙酰基
H 的片段：C(=O)C
保持不变：原分子、用户指令、原羟基连接位点
```

程序用错误片段执行编辑，得到内部校验产物乙酸乙酯；其重原子数是 6，而正确产物丙酸乙酯为 7。两者环数均为 0，因此不应强行改变环数。

H 文本更新片段识别、相应连接描述及计数。原用户指令仍是丙酰化，不能把题目改成乙酰化来让 H 变正确。错误片段在推理中的所有相关引用须一致。完整产品 SMILES 只留在私有审计区，不重新加入已删除的 PRODUCT_SMILES 答案子句。

## 7. 参考 N、文本与标签契约

- 冻结原始 `N_raw`，记录原始路径/行号/origin。参考审查失败标记 `reference_invalid` 或 `reference_ambiguous`，不自动修 N 后仍称为原始 N。
- 明确定义 `N_visible`：沿用既有删除产品答案子句的公开视图。如需进一步去除 ground-truth 提示，必须先定义、版本化并对双方对称执行同一规则；不能只清洗 H。
- 冻结前也检查自然语言里的完整产物 SMILES、明确报出最终产物名称的答案句以及 oracle 来源提示。按预先定义的答案跨度规则投影，不能全局删除碰巧相同的分子子串或正常片段名称。无法无歧义处理则标记 `reference_leakage` 并隔离；不能在生成 H 后只删除 H 一侧的内容。
- H 从 `N_visible` 单向派生。未受影响文本原样保留；需要重写的句子记录原因和语义绑定。
- 不要求复杂语义改写后仍具有一对一字符跨度。保留 root/propagated 语义标签与 step 对齐，文本跨度允许一对多、多对多。
- 公开的 model-visible 内容不含 H/N 标签、错误数量、计划、审计意见或错误标记；这些信息放在 sidecar。
- 内部同时保留 `reference_product`、`corrupted_product`、`ground_truth_final_answer` 的不同含义。不能将现有 `final_answer` 字段悄悄改义，因为 outcome runner 当前将其与原题标准答案核对。
- 新 schema 显式区分 `root_error`、`propagated_error`、`unchanged`。在后续研究恢复模式时再增加 `explicit_repair` 与 `unexplained_recovery`；第一版的严格传播不能代表自然恢复率为零。
- 输出成功只表示通过既定校验。生成/校验同用一种模型的共同盲点，须通过人工抽检和工具证据评估。

建议 sidecar 字段：

```text
protocol_version, origin_id, pair_id, variant_index, plan_id
source_n_reference, N_raw, N_visible
seed, draw_index, sampling_policy, sampled_K, feasible_candidates
root_mutations[{id, node, type, before, after, selection_probability}]
edit_operations, affected_node_ids
node_evaluations[{node, parents, root_mutation_ids, result, changed, rule}]
propagation_events[{node, parents, root_mutation_ids, before, after, rule}]
reference_product, corrupted_product, product_changed
tool_checks, blind_audit, plan_audit
attempts, rejection_codes, model, prompt_version, generation_parameters
network_calls, elapsed_seconds, reported_usage_if_available
```

ID 用 UUID 或持久化顺序 ID。seed/draw manifest 一次写入并复用；续跑按 plan_id 和完整输入内容比对，版本或输入改变必须新建计划。没有 SHA 字段或完整性校验依赖。

## 8. 接收标准与失败处理

接收必须同时满足：

1. 参考 N 已审查，无未解决的原题/参考答案冲突。
2. 原题与公开 N 没有被改写；全部指定根错误已实现，根错误数与计划一致。
3. 除指定根错误外，后代由当前父节点和操作合理推导；没有未登记的新错误。
4. 计划要求执行的化学编辑有真实工具执行记录；实际描述符由产物计算。计数根错误及其派生 claim 与实际值不同是允许的，但必须精确归因到固定计划，不能将这些 claim 冒充工具返回值。
5. 受影响文本引用一致，无过期描述、额外产品答案子句或生成者自评泄露。
6. 本地校验通过，独立审查无未解决的 fail/uncertain。盲审未发现某根错误不单独构成拒绝或通过理由，应在对照审查中结合工具证据解决；分歧未解决则进入人工审查。

`product_changed` 是结果分类，默认不是接收条件。计数错误可能不改变结构；若单独发布“产物改变”子集，要披露筛选规则和接收率，不能用于声称总体自然错误率。

重试建议：语义/文本修复最多 2 轮，根计划不变；请求超时或限流采用有上限的退避重试。每个 variant/plan 最多 10 次实际 Poe 请求，包含失败、HTTP/SDK 重试、修复和修复后的重新审查；这个总预算优先于最多 2 轮修复。每个 origin 的共享参考解析/审查另限 4 次实际请求。每轮最多 8 次工具执行，超限进入失败列表。18 origin/36 plan 的初始 smoke 批次总上限为 432 次实际请求，不能自动增加计划来填补失败样本。参考解析/审查可按 origin 缓存；缓存需保留实际响应，不能假定 temperature=0 就完全可重复。

错误分类至少包括 reference_invalid、reference_leakage、no_feasible_plan、ambiguous_edit、invalid_structure、plan_drift、unexpected_root、stale_claim、answer_leakage、audit_uncertain、transport_error、budget_exhausted。失败原因按计划统计，禁止只汇总成功样本。

没有经官方接口核实的 token/费用数据不猜测；先记录请求数、耗时和接口实际返回的 usage，并设置批次调用上限。

## 9. Agent 消息契约摘要

构造 Agent 的 system 核心要求：

```text
You construct a controlled counterfactual chemistry rationale from a frozen reference.
The orchestrator's root mutation plan is fixed. Do not change the source question,
reference N, root count, root targets, or sampled replacement values.
Use only approved tools for molecular edits and numeric properties.
Propagate consequences of the specified false premises; do not repair those premises.
Do not introduce additional unsupported claims or reveal a full product answer in prose.
Return schema-valid actions or the requested structured text segments.
If the plan cannot be executed coherently, return a typed failure with evidence.
```

对照校验 Agent 的 system 核心要求：

```text
Audit whether this candidate implements exactly the provided corruption plan.
Distinguish truth relative to the original question from validity relative to the
corrupted premises. Do not repair intended root errors or accept unplanned ones.
Ground every finding in node IDs, text spans, or tool execution evidence.
Return pass, fail, or uncertain with explicit findings and bounded repair requests.
```

盲审使用独立提示，不暴露上述计划或 H 标签，仅要求检查给定原题与推理的错误和一致性。

## 10. 实施模块建议

| 路径 | 职责 |
| --- | --- |
| `core/agent_generation.py`（新增） | 固定计划、编辑操作、多根传播、审计结果与发布版本的数据契约 |
| `modules/agent_generation/poe_transport.py`（新增） | 环境密钥读取、Poe 调用、脱敏与请求预算；无摘要缓存依赖 |
| `modules/agent_generation/chemistry_tools.py`（新增） | 白名单分子编辑和可复查的工具结果 |
| `modules/agent_generation/agents.py`（新增） | 构造、盲审和对照审查的隔离上下文 |
| `modules/agent_generation/orchestrator.py`（新增） | 状态机、固定计划修复、持久化和断点续跑 |
| `modules/error_planning/`（扩展） | 截断计数采样、候选概率记录和联合计划检查 |
| `modules/error_injection/`（扩展） | 真实编辑操作与共享后代的正向传播 |
| `modules/text_realization/`（适配） | 从冻结 N 最小改写；语义跨度对齐 |
| `modules/release/`（适配） | 原始 N 配对、私有审计与公开输入分离 |
| `scripts/generate_agent_pairs.py`（新增） | 离线计划检查、有限 smoke、批次生成与续跑入口 |

以上是拟实施文件，不声称这些新文件已经存在。第一阶段不引入额外 Agent 框架，复用项目 Python/Poe/RDKit 依赖和结构化 JSON 协议。

## 11. 分阶段验证与验收

1. 离线：用固定 N 和模拟 Poe 响应验证计划约束、联合传播、失败隔离、N 不变、公开数据无答案泄露和恢复行为。不以复制实现细节的测试替代真实图编辑验证。
2. 实例：验证乙醇丙酰化/乙酰化例子，以及锚点改变、删除组分、立体化学保持、多根共享 product、不可执行计划等案例。
3. 小批真实调用：18 个 origin（add/delete/substitute 各 6），每个计划生成 K=1 与 K=2 两类；若某题 K=2 不可行则如实失败，不换成 K=1。最多 36 对，逐对人工审查。
4. 在固定计划上比较原流程与 Agent 流程，统计计划实现率、意外根错误率、文字残留、审查分歧、接收率、调用成本；有机会时比较在自然 D 组轨迹中的错误类型和位置。
5. 质量检查后再扩展到全部 150 origin 和多种 K/错误类型，使用新输出目录；按 origin 划分开发与保留评估集，不能让同题多个变体跨集合泄露。

每次报告至少包括：计划与接受的 K 分布、计划与接受的类型分布、根/传播节点数、产品改变率、失败原因、盲审/对照审查分歧，以及人工发现的错标。原始 N 保持不变、不可执行计划不得假成功，是发布硬条件；没有宣称代理审查达到 100% 化学正确率。

后续 A–E 实验采用相同问题、相同 N 和相同 E 借用策略；B 使用新版 H，其他条件固定。自然生成校准与“要求遵循 H”的响应实验分别解释，最终准确率不能用于反推自然根错误分布。

## 12. 终端与运行边界

用户终端运行方式为先输入 `poe`，再在同一 shell 启动将来的生成命令。alias 属于 shell，会话结束后不会为其他进程自动持久化；后台进程需在已经加载密钥的 shell 中启动并继承环境。程序只检查变量是否存在，不打印 alias 定义或变量值。

初始设计检查点当时只交付设计，尚未做 API 连通性调用、真实化学 smoke 或批量生成。此后已完成的调用、生成与 v9/v11 评估记录见本文开头和实施记录；不得把这条历史状态当作当前运行状态。
