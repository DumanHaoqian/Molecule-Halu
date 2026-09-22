# ChemDFM-R Layer 26 SAE 语料检查

检查日期：2026-09-21。源目录 `/home/haoqian/Data/past/Molecule`；只读检查，未抽取 activations、未训练、未下载额外语料。

**结论：现有语料足以启动小到中等规模的 SAE 探索实验。优先解决去重、评估集隔离和 CoT 分布匹配；不需要为了凑文本量立即下载大型通用语料。是否足以让某个 SAE 收敛，仍需结合字典宽度、稀疏度和验证曲线判断。**

## 统计口径

样本数为逐文件全量解析的精确计数。token 为本地 ChemDFM-R-14B tokenizer 对固定种子 20260921 的 reservoir 样本估算：一般每文件 1,024 条；ChemCoT 字段回退版 512 条，小文件全量。M=百万。统计不含 system prompt/chat-template 的固定开销，未做长度截断；未扣除全局重复、评估重合与质量过滤，所以不能当作最终可训练 activation 数。每文件采样误差和长度分位数见 profile.json / chemcot_selected.json。

## 候选数据

| 数据 | 本地记录数 | 回答/推理 token 估计 | 用途与限制 |
|---|---:|---:|---|
| ChemCoT | 23,223 | 33.92M | 推理语料优先；raw_cot → struct_cot → cot_result 选择一个视图 |
| Mol-LLaMA-Instruct 四类 | 284,743 | 133.42M | 自然语言化学描述/问答；不是 ChemDFM 自生成 CoT |
| PubChem 分子描述 | 315,396 | 原始描述 24.92M；扩写 108.68M | 两个替代视图，与 Mol-LLaMA 有分子重合；不能都当独立新增数据 |
| OpenMolIns xlarge | 1,200,000 | 答案 SMILES 34.74M；含题目 76.58M | 无 CoT；主要用作 ChemDFM 自生成推理的题目池 |
| MoleculeQA train | 49,993 | 0.15M | 多项选择短答案；适合作为生成解释的题目池 |
| ChEBI-20 train | 26,407 | 2.37M | 分子描述补充，需与 Pilot 隔离 |
| 已整理 Datasets/Train | 68,748 | 不另加总 | 上述原料的变换/子集，含大量非文本 world-model 字段 |

Mol-LLaMA 四类共 194,915 个不同 CID、672,289 个问答轮次；含用户输入约 163.52M tokens。PubChem 与这四类共享 194,915 个 CID，共享分子不等于全文重复。

主要入口：

- `Downloads/Train/Chemcot/`
- `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/`
- `Downloads/Train/OpenMolIns/xlarge/train.csv`
- `Molecule/Baselines/Mol-LLaMA/data/moleculeqa/train.json`
- `Molecule/Baselines/Mol-LLaMA/data/ChEBI-20/train.txt`

## 已证实的数据风险

1. **不同任务的 CoT 字段不同。** 13,597 条使用 raw_cot，6,484 条需要 struct_cot，3,142 条需要 cot_result。只读 raw_cot 会漏掉 9,626 条；不能把两个推理视图拼起来重复计数。反应 major-product 的 2,788 条只有 1,164 个不同的题目+选定推理文本对。
2. **答案提示。** 选定推理中 3,254 条命中 ground truth / reference answer / correct answer is；这是需要复核的文本信号，不代表这些全部存在泄漏。已有样例明确提到已知真值。建议隔离或在只给问题、不给参考答案的条件下重新生成。
3. **重复副本。** 已对 ChemCoT add、OpenMolIns xlarge、Mol-LLaMA comprehensive 的原料/Construction 副本验证 SHA-256 相同。其他同名副本尚未逐一做字节级验证。OpenMolIns 的 light/small/medium/large 文本对全部包含在 xlarge 中，只保留 xlarge 即可。
4. **模态与字段。** Mol-LLaMA 的 `<mol>` 需要替换为实际 SMILES；PubChem 的 4.42 GB 包含原子和坐标，不能视为 4.42 GB 文本。加工版的 latent slots、subgraphs 和目标分子字段不能整体序列化为提示；有些 molecule 字段是目标结构，误用会把答案放进输入。
5. **评估集重合。** OpenMolIns xlarge 与 Pilot 有 10 道完全相同的题目文本，其中 4 道属于 test；Mol-LLaMA 与 Pilot caption 任务共享 42 个 CID，涉及 43 个 task/question 组合（30 train、8 validation、5 test）。ChEBI-20 train 另命中 1 道 Pilot train 题目。这里是按规范化题目文本/CID 的保守检查，未进行全量 canonical-SMILES 或语义近重复检索，实际重合可能更多。
6. **训练集与测试集用途。** 优先训练来源，并排除 Pilot 全部 600 道题及对应分子/反应组；至少严格隔离其 validation/test。原目录 Testset 和外部 benchmark 的评估 split 不作为 SAE 默认训练池。

## 是否需要补充

**数量上：第一版不需要先补充外部数据。分布上：建议优先补充 ChemDFM-R 自己生成的化学推理。** Mol-LLaMA 的 GPT-4o 描述、ChemCoT 的其他模型推理通过 teacher forcing 仍能用于学习 ChemDFM 的激活字典，但不能等同于 ChemDFM 自身 CoT 的分布。

建议在不重合的新题目上按当前 12 个任务分层，用 ChemDFM-R 生成 2–5 万条新回答；假设每条保留约 1,000 个 CoT/回答 token，即新增约 20–50M tokens（规划假设，不是已测量结果）。不需要幻觉标签，也不必只保留正确回答；需要排除破损文本、异常循环和无上下文输出，保留正常生成中的错误模式。应按 token 配比，避免某类长 CoT 独占训练。

若需要补领域，再考虑：

- **更新 ChemCoTDataset 的缺失子任务**：官方 2026-01-19 更新记录加入逆合成、机理等；本地 16 文件中没有 retro / mech_sel / nepp。只补独立训练语料，先与我们的评估题去重。[官方数据卡](https://huggingface.co/datasets/IDEA-AI4S/ChemCoTDataset)
- **ChemPile 的 chemistry reasoning / 教育文本小子集**：用于弥补通用化学解释覆盖，不必下载全量。其不同 config 有相同题目的多种模板，不可直接累加。官方卡片的总 token 统计与文件/行规模不易相符，不据此规划容量，应下载选定 split 后重新测量。[官方数据卡](https://huggingface.co/datasets/jablonkagroup/chempile-reasoning)

## 激活获取建议

沿用前一实验 ChemDFM-R-14B：第 26 个 transformer block 的输出（零索引 25），post-token 表示，维度 5,120。冻结语言模型，以真实输入/对话作为上下文；主要采样 assistant 的 CoT/回答 token，并保留 region、source、task、token offset。padding 和模板控制符不作为主要 SAE 样本。不得将标签、标注原因或 reference 内容额外加入模型实际不可见的提示。

先用 10–20M 个去重后有效 tokens 做流程和小字典试验，再向 50–100M 扩展；这是实验预算建议，非已证实的最低需求。另按分子/题目组留出 SAE 验证语料。用重建误差/解释方差、稀疏度、dead-feature 比例、特征稳定性及独立 Pilot 验证表现判断是否需要更多数据；最终测试集不用于选择数据配比或超参数。

按 BF16/FP16 2 bytes 缓存：1M tokens = 10.24 GB；10M = 102.4 GB；50M = 512 GB；100M = 1.024 TB（均为十进制、仅激活张量，不含索引等开销）。磁盘容量已复核更正：先前 df -h 曾返回可用 643G；用户反馈后再次检查同一挂载点 /mnt_nas1，当前为可用 9.8T、使用率 87%，df -B1 给出的可用空间为 10,674,734,628,864 bytes（约 9.71 TiB / 10.67 TB）。两次读数变化原因未查明。按当前容量，100M tokens 的 1.024 TB 激活缓存可以容纳；之前“放不下”的判断不再适用。仍建议分片缓存以便恢复和管理，但当前容量不要求必须采用流式方案。

SAE 需求取决于字典宽度和稀疏度，并应多维度评估，而不是设一个通用 token 数门槛。参考：[Gao et al., Scaling and evaluating sparse autoencoders](https://arxiv.org/html/2406.04093v1)

## 逐文件数据

ChemCoT 表使用字段回退后的推理；PubChem second 是 enriched_description；processed 表仅为字段诊断，缺少对话 history/分子上下文，不可直接视为完整输入。

| 路径（相对源目录） | 行数 | 文件内不同文本对 | second 字段 token 估计 |
|---|---:|---:|---:|
| `Downloads/Train/Chemcot/mol_edit/add.json` | 1,499 | 1,499 | 4.16M |
| `Downloads/Train/Chemcot/mol_edit/delete.json` | 1,499 | 1,499 | 4.12M |
| `Downloads/Train/Chemcot/mol_edit/sub.json` | 1,499 | 1,499 | 4.30M |
| `Downloads/Train/Chemcot/mol_opt/drd.json` | 1,500 | 1,500 | 1.69M |
| `Downloads/Train/Chemcot/mol_opt/gsk.json` | 724 | 724 | 0.97M |
| `Downloads/Train/Chemcot/mol_opt/jnk.json` | 180 | 180 | 0.27M |
| `Downloads/Train/Chemcot/mol_opt/logp.json` | 1,394 | 1,394 | 0.46M |
| `Downloads/Train/Chemcot/mol_opt/qed.json` | 377 | 377 | 0.43M |
| `Downloads/Train/Chemcot/mol_opt/solubility.json` | 1,412 | 1,412 | 0.45M |
| `Downloads/Train/Chemcot/mol_und/Murcko_scaffold.json` | 1,600 | 1,600 | 2.85M |
| `Downloads/Train/Chemcot/mol_und/fg_count.json` | 1,519 | 1,519 | 2.38M |
| `Downloads/Train/Chemcot/mol_und/ring_count.json` | 1,600 | 1,600 | 5.11M |
| `Downloads/Train/Chemcot/mol_und/ring_system_scaffold.json` | 1,600 | 1,600 | 3.68M |
| `Downloads/Train/Chemcot/rxn/fs_by_product.json` | 890 | 890 | 0.56M |
| `Downloads/Train/Chemcot/rxn/fs_major_product.json` | 2,788 | 1,164 | 1.35M |
| `Downloads/Train/Chemcot/rxn/rcr.json` | 3,142 | 3,142 | 1.13M |
| `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/detailed_structural_descriptions.json` | 77,239 | 77,239 | 35.30M |
| `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/structure2chemical_features_relationships.json` | 73,712 | 73,712 | 36.90M |
| `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/structure2biological_features_relationships.json` | 73,645 | 73,645 | 35.43M |
| `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/comprehensive_conversations.json` | 60,147 | 60,147 | 25.80M |
| `Molecule/Baselines/Mol-LLaMA/data/Mol-LLaMA-Instruct/pubchem-molecules.json` | 315,396 | 315,396 | 108.68M |
| `Downloads/Train/OpenMolIns/xlarge/train.csv` | 1,200,000 | 1,200,000 | 34.74M |
| `Downloads/Train/OpenMolIns/large/train.csv` | 90,000 | 90,000 | 2.64M |
| `Downloads/Train/OpenMolIns/medium/train.csv` | 45,000 | 45,000 | 1.32M |
| `Downloads/Train/OpenMolIns/small/train.csv` | 18,000 | 18,000 | 0.54M |
| `Downloads/Train/OpenMolIns/light/train.csv` | 4,500 | 4,500 | 0.13M |
| `Molecule/Baselines/Mol-LLaMA/data/moleculeqa/train.json` | 49,993 | 49,968 | 0.15M |
| `Molecule/Baselines/Mol-LLaMA/data/ChEBI-20/train.txt` | 26,407 | 26,407 | 2.37M |
| `Datasets/Train/01Conversation/conversation_sft_with_worldmodel.jsonl` | 10,292 | 10,292 | 0.65M |
| `Datasets/Train/02PubChem/latent_world_modeling.jsonl` | 17,243 | 17,243 | 5.58M |
| `Datasets/Train/03MolEdit/mol_edit_with_worldmodel.jsonl` | 4,497 | 4,491 | 0.35M |
| `Datasets/Train/04TomgBench/train_small.jsonl` | 17,999 | 17,999 | 1.31M |
| `Datasets/Train/05MolOpt/mol_opt_with_worldmodel.jsonl` | 5,587 | 5,572 | 0.76M |
| `Datasets/Train/06MolUnd/mol_und_with_worldmodel.jsonl` | 6,310 | 6,226 | 0.07M |
| `Datasets/Train/07Rxn/rxn_with_worldmodel.jsonl` | 6,820 | 5,176 | 0.18M |

OpenMolIns 与 xlarge 的文本对重合：

```json
[
  {
    "size": "large",
    "unique_pairs": 90000,
    "shared_with_xlarge": 90000
  },
  {
    "size": "medium",
    "unique_pairs": 45000,
    "shared_with_xlarge": 45000
  },
  {
    "size": "small",
    "unique_pairs": 18000,
    "shared_with_xlarge": 18000
  },
  {
    "size": "light",
    "unique_pairs": 4500,
    "shared_with_xlarge": 4500
  }
]
```

复核文件：`inventory.json`、`profile.json`、`chemcot_selected.json`、`overlap.json`。复现代码：`audit.py`、`check_chemcot_fallback.py`、`check_overlap.py`；可交互查看 `AUDIT.ipynb`。
