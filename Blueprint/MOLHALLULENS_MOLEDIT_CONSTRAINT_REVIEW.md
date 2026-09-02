# MolHalluLens Molecule Editing Pilot 约束审阅记录

> 审阅日期：2026-08-28
> 审阅对象：`MOLHALLULENS_MOLEDIT_HALLUCINATION_IMPLEMENTATION_PLAN.md` v0.2
> 实现范围：ChemCoTBench-V2 Molecule Editing Pilot
> 状态：批准作为实现基线

## 1. 已批准的实现范围

本轮实现严格限于 150 个 Molecule Editing origins，包含 addition、deletion 和 substitution 三个 subtask。目标数据规模为 1,200 records，按 leakage group 拆分为 100/25/25 origins，对应 train/validation/test 800/200/200 records。

实现包括 typed State DAG、严格的 RDKit/graph-edit validation、四类 propagation、四组 H/N matched pairs、char-first multi-axis annotation、ChemDFM-R post-token derived labels、Poe 候选 agent、label-blind renderer、split-local donors、完整 provenance/cache、shortcut audit 和 release QA。

本 Pilot 只为 `mol_opt`、`mol_und`、`rxn_pred` 预留抽象接口，不实现它们的具体 operators。不得增加 pre-token target，不得把 LLM 自报结论作为标签或接受依据，不得使用全文 string diff 生成 mask，也不得使用裸 SMILES 字符串相等判断分子身份。

## 2. 冻结决定的实现与验证归属

| # | 冻结决定 | 实现归属 | 强制验证归属 |
|---:|---|---|---|
| 1 | Detector 可见顺序固定为 indexed SMILES → instruction → reasoning → final answer | T005 serializer；T023/T040 renderer | T005 golden string；T043 renderer validator；T052 release QA |
| 2 | `gt_smiles` 仅为隐藏 oracle | T004 visibility/types；T005 serializer；T011 reference DAG；T052 artifact separation | T005 接口测试；T043 leakage scanner；T052 secret/oracle boundary scan |
| 3 | 仅 post-token `h_t` 对齐 | T006 alignment contract；T042 token projection；T051 activation extraction | T006/T042 等长与 ignore-mask 测试；T051 hidden-state alignment assertion |
| 4 | typed State DAG 上扰动，之后再渲染 | T004/T009/T010/T011 DAG；T017 root patch；T022 propagation；T039 AST | T015 reference validation；T043 state/render round-trip；T044 golden snapshots |
| 5 | LOCAL/PARTIAL/FULL_CF/TERMINAL 四类等额 | T003 config；T022 propagation；T024 scheduler | T022 property tests；T043 phenotype validator；T048–T050 split counts |
| 6 | 每 origin 四组 H/N，共 8 records | T024 matched bundle builder | T024 bundle tests；T043 integrity validator；T048–T050 exact counts |
| 7 | 150×8=1,200；100/25/25 origins | T003 config；T028 split solver；T029 manifest | T028 quota proof；T048–T050 counts；T052 final integrity |
| 8 | char span 为 canonical，token mask 为 derived artifact | T039/T041 char annotations；T042 token projection | T039 offset tests；T042 tokenizer fingerprint/hash tests；T043 validator |
| 9 | token label 使用正交多标签轴 | T004/T003 taxonomy；T041 annotation；T042 token labels | T041 taxonomy tests；T043 subset/causal-role invariants；T052 release QA |
| 10 | family/subtask 继承，其他能力 composition/Strategy | T016 hierarchy/factory；T017 registry；T018/T022/T040/T043 injected components | T016 factory/Template Method tests；代码审阅与 T044 integration tests |
| 11 | Rule/RDKit + LLM 提候选，deterministic validator 唯一裁决 | T018 deterministic sources；T032 tools；T037 hybrid agent | T015/T037 chemistry gates；T038 reject/coverage report；T052 provenance audit |
| 12 | split 在 donor 和 LLM generation 前冻结；donor 仅同 split | T027/T028/T029 split；T030 donor pools；T037 candidate request | T029 immutable hash；T030 cross-split rejection；T043/T052 leakage assertions |
| 13 | GPT 仅经 Poe，固定 `gpt-5.4-mini`，key 仅来自环境 | T003 config；T033 registry；T034 clients；T035 ledger | T033 capability probes；T034 fallback tests；T052 model/provenance/secret scan |

每项冻结决定至少同时拥有一个实现任务和一个自动或可审计的验证任务。任何实现发现无法维持上述归属时，必须停止并在任务 JSON 中登记 blocker，不得静默绕过。

## 3. Benchmark 使用决定（风险 6）

采用实施计划风险 6 的方案 1：

> 当前 150 个 ChemCoTBench-V2 origins 仅用于 MolHalluLens pipeline/schema smoke test、数据构建验证和方法可行性审计，不进入正式 detector training，不用于选择 detector layer 或调节最终 threshold。

因此：

- 本 Pilot 即使生成 train/validation/test 文件，这些名称也表示数据构建管线内部的 counterfactual split，不授权将其中 train/validation 用于正式 detector 拟合或模型选择。
- 与这些 origins 对应的 ChemCoTBench-V2 slice 继续按 blind evaluation material 管理。
- 所有统计、bootstrap 和配对评测以 origin/leakage group 为独立单位。
- T050 必须验证 test 没有用于 candidate、layer 或 threshold 选择。
- T052 的 dataset card 和 known limitations 必须明确披露该限制；如未来改用方案 2，必须由项目负责人显式批准、更新本文、实施计划相关说明、dataset manifest 和 dataset card。

## 4. 实现边界与 fail-closed 规则

以下情况必须 fail closed：

- serializer 尝试接收或输出 `gt_smiles`、reference-only state 或 correctness metadata；
- propagation changed set 不满足对应 phenotype；
- H 候选与 reference 图等价，或 structural candidate 未通过 deterministic chemistry/graph-edit validation；
- donor 跨 split，或 split manifest 尚未冻结即开始 donor/LLM candidate generation；
- Poe model ID、endpoint 或 tool capability 与冻结配置不符；
- 输出缺少可验证 char span、token offset、provenance 或完整 matched pair；
- 任何 artifact、日志或 traceback 包含 `POE_API_KEY` 或 Authorization header。

不得用降低 sanitize/valence 标准、改变 propagation、切换模型、改变 H/N 比例或留下不完整 bundle 的方式解决失败。

## 5. Phase 0 退出解释

本文只完成了计划审阅和范围冻结。Phase 0 仍需 T002–T006 完成目录/依赖、配置、domain schema、detector golden string 和 post-token alignment tests 后才可退出；本文不得被解释为代码、数据或完整 Phase 0 已完成。
