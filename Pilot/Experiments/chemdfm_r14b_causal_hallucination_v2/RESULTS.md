# ChemDFM-R-14B 因果 Hallucination 实验结果

## 结论

新的 wrong-fragment 数据可以稳定测到 hallucinated reasoning 对最终产物的负面影响。推荐主分析子集包含 57 个 add/substitute 样本，正确与错误片段具有相同重原子数和相同首连接元素：

- N（正确 fragment reasoning）：49/57，85.96%。
- H（仅 fragment 被替换）：31/57，54.39%。
- H − N：-31.58 percentage points。
- 配对变化：0 题 H 独对，18 题 N 独对；双侧 exact McNemar `p=7.63e-6`。

这说明旧 H/N 数据没有效应并不代表 CoT hallucination 无影响。原 H 中的错误多为外围计数或错误 anchor，模型容易忽略；当错误落在决定产物结构的 ADD/REMOVE fragment 上时，outcome 明显下降。

## 实验设计

每个样本来自同一个原始问题，四组共享 plain source、atom-indexed source、instruction、system prompt 和 greedy decoding 配置。

| 组别 | 输入 |
| --- | --- |
| A | 只有问题 |
| N | 上游专家的简洁正确 reasoning，最终产物已删除 |
| H | 与 N 相同，只把一个决定产物的 ADD_FRAGMENT 或 REMOVE_GROUP 换成错误 fragment |
| L | H 加上一个权威语境下的错误 no-edit 产物诱饵（原 source molecule） |

N/H 的 anchor、操作、文本结构和其他字段保持相同。错误 fragment 从同一子任务的其他样本中确定性选择，优先匹配重原子数、首连接元素和字符串长度。GT 从 reasoning 中完全删除。

共有 135 个问题：add/delete/substitute 各 45；四组共 540 次生成。85/135 个错误 fragment 与正确 fragment 重原子数相同，124/135 首连接元素相同。

## 完整结果

| 组别 | 正确 | Accuracy | Mean FTS | 错误 source lure 匹配 |
| --- | ---: | ---: | ---: | ---: |
| A | 108/135 | 80.00% | 0.9534 | 2/135 |
| N | 116/135 | 85.93% | 0.9622 | 2/135 |
| H | 91/135 | 67.41% | 0.9239 | 2/135 |
| L | 97/135 | 71.85% | 0.9297 | 2/135 |

H 相对 N 下降 18.52 points：25 题 N 对/H 错，0 题 H 对/N 错，`p=5.96e-8`。H 也比无 reasoning 的 A 低 12.59 points（2/19 discordant，`p=2.21e-4`）。N 相对 A 提升 5.93 points，但本样本下未达到 0.05（13/5，`p=0.0963`）。

| 子任务 | N | H | H − N | 配对 p |
| --- | ---: | ---: | ---: | ---: |
| add | 36/45 (80.00%) | 28/45 (62.22%) | -17.78 pp | 0.0078 |
| delete | 42/45 (93.33%) | 41/45 (91.11%) | -2.22 pp | 1.0000 |
| substitute | 38/45 (84.44%) | 22/45 (48.89%) | -35.56 pp | 3.05e-5 |

H/N 有 104/135 对生成文本完全相同。剩余 31 对中有 25 对发生正确到错误的单向变化，说明效应集中在模型真正响应错误 fragment 的样本，而不是整体随机波动。

错误 fragment 的直接子结构采用率并不高：在可明确判定的 add 题中 H 为 2/34，在 substitute 中为 7/33。多数受影响输出没有逐字复制错误片段，而是因 instruction 与错误 reasoning 冲突而生成其他错误结构。因此当前指标证明的是错误 reasoning 会扰乱 outcome，不能把全部错误解释为 literal copying。

L 组的 source lure 只匹配 2/135；相对 H 没有额外伤害。no-edit source 对模型来说过于明显，不适合作为后续主要诱饵。

## 数据质量等级

`hallucination_dataset.jsonl` 共 135 条，每条包含原问题、N/H rationale、正确/错误 fragment、source、GT 和 `quality_tier`。

| 等级 | 数量 | 用途 |
| --- | ---: | --- |
| `primary_matched_fragment` | 57 | 推荐主 benchmark；add/substitute，重原子数和首元素均匹配 |
| `secondary_fragment` | 33 | add/substitute 扩展集；至少一个匹配条件不满足 |
| `diagnostic_delete_present` | 18 | 错误 REMOVE_GROUP 确实存在于 source，可作探索分析 |
| `diagnostic_delete_conflict` | 27 | 错误 REMOVE_GROUP 不存在于 source，只测冲突鲁棒性，不应进入主结果 |

delete 的整体效应弱，主要原因是 donor REMOVE_GROUP 在 27/45 个 source 中不存在。下一版本若要把 delete 纳入主 benchmark，应从当前分子中枚举另一条真实 pendant branch，生成可执行的错误删除操作。

## 完整性

- 540/540 request、prediction、outcome 一一对应，request hashes 全部匹配。
- 540/540 `finish_reason=stop`，540/540 为有效 SMILES。
- 0 missing、0 truncated、0 atom-map 输出；最大生成长度 118 tokens。
- N/H 平均输入长度分别为 550.48/550.40 tokens；配对差的中位数为 0，排除了系统性长度差。
- 推理仅使用物理 GPU 6、7。

详细统计见 `summary.json`、`causal_analysis.json` 和逐题 `outcome_records.jsonl`。
