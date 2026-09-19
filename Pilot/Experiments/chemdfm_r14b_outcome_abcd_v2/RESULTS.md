# ChemDFM-R-14B 四组 Outcome 实验（v2）

本实验替代 `chemdfm_r14b_outcome_abcd`。旧结果的主要异常来自实验实现，不能用于判断 hallucinated reasoning 对最终产物预测的因果影响：B/C reasoning 都直接包含正确 `PRODUCT_SMILES`；A/D 缺少 benchmark 原本提供的 plain source SMILES；主指标标签与 benchmark 的实际主片段判定不一致；此外，8,192 token 上限和无 repetition penalty 导致 A/D 出现重复生成和截断。

## 修正后的设计

- 四组共享完全相同的 plain source、atom-indexed source 和 edit instruction。
- A：直接回答。
- B：读取 H reasoning 后直接回答。
- C：读取 N reasoning 后直接回答。
- D：模型自行生成 reasoning，再回答。
- B/C 中的最终产物字符串已删除；准备阶段同时检查所有引号内的有效 SMILES，拒绝任何与 GT 主片段等价的产物泄漏。
- 主指标为去 atom-map 后的 main-fragment molecular equality；同时保留 normalized exact、raw exact 和 FTS。
- greedy decoding，seed 42，repetition penalty 1.05，最多 2,048 output tokens。
- 仅使用物理 GPU 6、7，tensor parallel size 2。

样本为同一批 150 个配对问题，add/delete/substitute 各 50；每组 150 条，共 600 条。

## 主要结果

| 组别 | 输入方式 | 主指标 | 平均 FTS | 有效 SMILES | 平均输出 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Question → direct | 121/150 (80.67%) | 0.9562 | 150/150 | 52.88 |
| B | Question + redacted H reasoning → direct | 132/150 (88.00%) | 0.9682 | 150/150 | 52.97 |
| C | Question + redacted N reasoning → direct | 131/150 (87.33%) | 0.9662 | 150/150 | 53.03 |
| D | Question → self-CoT → answer | 117/150 (78.00%) | 0.9431 | 150/150 | 249.86 |

本次所有预测均无 atom-map，因而主指标、normalized exact 和 raw exact 的正确数相同。

### 子任务

| 子任务 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| add | 37/50 (74%) | 39/50 (78%) | 40/50 (80%) | 36/50 (72%) |
| delete | 48/50 (96%) | 48/50 (96%) | 47/50 (94%) | 44/50 (88%) |
| substitute | 36/50 (72%) | 45/50 (90%) | 44/50 (88%) | 37/50 (74%) |

### 配对比较

双侧 exact McNemar 检验基于同一题的成败变化，未做多重比较校正。

| 比较 | 准确率差 | 前者独对 / 后者独对 | p |
| --- | ---: | ---: | ---: |
| B vs A | +7.33 pp | 15 / 4 | 0.0192 |
| C vs A | +6.67 pp | 16 / 6 | 0.0525 |
| D vs A | -2.67 pp | 4 / 8 | 0.3877 |
| C vs B | -0.67 pp | 3 / 4 | 1.0000 |

B/C 的 150 对输出中，143 对最终生成文本完全相同。C 相对 B 只有 3 题从错变对、4 题从对变错，因此没有证据表明 N reasoning 优于 H reasoning，也没有证据表明 H reasoning 会降低 outcome。B/C 相对 A 的提升集中在 substitute：B +18 pp（9/0 discordant，p=0.0039），C +16 pp（9/1，p=0.0215）。合理解释是两类 reasoning 都提供了大量相同且有用的编辑线索；这不能证明错误的中间推理无害。

D 没有改善直接回答，且平均生成约 250 tokens；在该模型和数据上，自生成 CoT 增加计算量但没有可靠的 outcome 收益。

这些结果只覆盖一个模型、150 道题和一次确定性解码。若要识别 hallucination 的因果效应，下一步应对同一条正确 reasoning 做最小化、单因素错误注入，并增加多个 decoding seeds 或模型。

## 完整性与运行记录

- 600 个 request ID、prediction 和 outcome record 一一对应；600 个 request hash 全部匹配。
- 600/600 `finish_reason=stop`，均由 `</answer>` 终止。
- 600/600 有效 SMILES；0 missing，0 truncated，0 atom-mapped。
- A/B/C/D 的最大输出长度分别为 118/118/118/409 tokens；没有输出超过 1,000 tokens。
- 首次长跑在 280/600 时 vLLM engine 异常退出。该批 8 条被标为 `abort`，已从正式预测中移除并以同一冻结配置重生成；原文件保存在 `predictions.pre_abort_repair.jsonl`。
- 后续新实验使用 `Pilot/scripts/run_chemdfm_outcome_v3.py`；该版本不会把 `abort` 等非终止输出写成完成记录，并会显式退出以便从未完成 request 续跑。当前 v2 结果仍由目录内冻结脚本复现。
- 正式预测 SHA256：`b321e281143e032a7a196b05a94659dfb44b5dbb657ef830d822b8d828b2c92d`。
- outcome records SHA256：`27113fb3782f41d8792d353891075d86badcc1d6adce263c71a5c05fed8010d2`。
- summary SHA256：`8db83dd51bef2fec7be1f3f2b73ab6e174ab4ba8479972cf53403b550f99f844`。
- 冻结脚本 SHA256：`0a36526a67e4a7ecc21c906faf358b5055dd2449156537cec9180ae323819e17`。

机器可读结果见 `summary.json`、`summary.csv` 和 `outcome_records.jsonl`；完整输入见 `requests.jsonl`，配置与输入哈希见 `manifest.json`，推理环境与三次断点运行见 `runtime_history.jsonl`。
