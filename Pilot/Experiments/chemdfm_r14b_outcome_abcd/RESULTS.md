# ChemDFM-R-14B 四组 Outcome 实验结果

已完成 600/600 条预测：同一批 150 个 Question，每组各 150 题，
add/delete/substitute 各 50 题。模型为 `/mnt_nas1/shared/ChemDFM-R-14B`，
BF16、greedy decoding、seed=42、每条最多生成 8,192 tokens。
使用原始 benchmark 的 `gt_smiles`，仅评测最终产物。

| 实验 | 输入/输出方式 | 原 benchmark 严格匹配 | 去 atom-map 后匹配（辅助） | 平均 FTS |
| --- | --- | ---: | ---: | ---: |
| A | Question → 直接答案 | 0/150 (0.00%) | 17/150 (11.33%) | 0.3551 |
| B | Question + H reasoning → 直接答案 | 86/150 (57.33%) | 92/150 (61.33%) | 0.8185 |
| C | Question + N reasoning → 直接答案 | 85/150 (56.67%) | 91/150 (60.67%) | 0.8291 |
| D | Question → 自己 CoT → 答案 | 0/150 (0.00%) | 19/150 (12.67%) | 0.4226 |

本次主片段匹配率与严格匹配率相同。严格匹配调用 ChemCoTBench-V2
`smiles_match_exact`，该函数会保留 atom-map 编号；模型虽然被要求输出
无映射编号的 SMILES，仍有不少输出带编号。辅助指标仅清除预测中的
atom-map 编号，再比较分子；保留连接关系、立体化学、电荷和同位素。
A/D 的严格 0% 因而不等于所有预测在分子结构上都错误。

## 子任务结果

每个单元格为严格匹配率 / 去 atom-map 后匹配率，每组每项 50 题。

| 子任务 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| add | 0% / 10% | 64% / 66% | 66% / 68% | 0% / 12% |
| delete | 0% / 4% | 42% / 42% | 38% / 38% | 0% / 4% |
| substitute | 0% / 20% | 66% / 76% | 66% / 76% | 0% / 22% |

## 结果解释与输出完整性

- B/C 严格匹配只相差 1 题。两组都正确 82 题，仅 B 正确 4 题，
  仅 C 正确 3 题，两组都错误 61 题；本次结果没有显示 H 导致明显的
  Outcome 下降，也不能据此认定 H 优于 N。
- 150 条 H reasoning 和 150 条 N reasoning **都包含正确产物的
  PRODUCT_SMILES**。B/C 的结果包含利用或复制已给产物的可能，不能
  用这组结果证明模型验证了中间推理，或抵抗了错误产物提示。
- 按用户定义完整保留 H/N reasoning；没有在输入中额外加入
  `final_answer`、GT 标签或 H/N 身份信息。
- A/B/C 均没有生成新的 `<think>` 标签；D 使用模型自己的 CoT。
  对 D 的推理文本未进行 Layer 2/3 或幻觉评分。
- 所有题都保留在分母中，没有根据正确性重试、筛除或选择多次答案。
  缺少最终答案的输出均达到长度上限，计为失败。

| 实验 | 有效 SMILES | 有效输出中带 atom-map 编号 | 无完整最终答案 / 长度截断 |
| --- | ---: | ---: | ---: |
| A | 80/150 | 67 | 22 |
| B | 138/150 | 47 | 2 |
| C | 139/150 | 51 | 2 |
| D | 91/150 | 84 | 10 |

完整性检查通过：600 个 request ID 与输入一一对应、请求哈希吻合、
四组各 150 题、每组子任务各 50 题、没有超出生成 token 上限。

## 可复查材料

- [summary.json](summary.json)：完整机器可读指标。
- [summary.csv](summary.csv)：总体及分子任务表。
- [outcome_records.jsonl](outcome_records.jsonl)：原始生成、提取答案、GT 和逐题分数。
- [requests.jsonl](requests.jsonl)：全部实际输入及 assistant prefill。
- [manifest.json](manifest.json)：固定实验配置和数据哈希。
- [README.md](README.md)：推理、计分和复跑方法。
- 生成脚本：`Pilot/scripts/run_chemdfm_outcome.py`。

执行期间将批量落盘改为逐条落盘并续跑，保留了已完成的 16 条输出，
没有改变输入或解码参数。运行环境、初始/续跑元数据和日志均已保留。
