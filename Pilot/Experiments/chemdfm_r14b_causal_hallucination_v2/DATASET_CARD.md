# Causal MolEdit Hallucination v2 Dataset Card

## 目的

该数据集测量上游 reasoning 中一个产物决定性 fragment 错误，是否会改变下游模型的最终分子编辑结果。它采用同题配对、最小字段替换和产品泄漏检查，适合 outcome-level causal evaluation。

## 推荐使用方式

主结果只使用 `quality_tier == "primary_matched_fragment"` 的 57 条记录。每条记录分别构造：

- N：`question + normal_rationale`
- H：`question + hallucinated_rationale`

两组使用同一个 system prompt、assistant prefill、模型和 decoding 参数。主指标为去 atom-map 后的 main-fragment molecular equality，并用配对 exact McNemar 检验比较 N/H。

不要把 `gt_smiles` 或记录标签暴露给被测模型。`L` 的 no-edit lure 仅为诊断条件，不建议作为主要 hallucination 类型。

## 字段

`hallucination_dataset.jsonl` 每行包括：

- `pair_id`, `origin_id`, `subtask`
- `quality_tier`
- `question`
- `normal_rationale`, `hallucinated_rationale`
- `intervention_field`
- `correct_fragment`, `wrong_fragment`
- `same_heavy_atom_count`, `same_first_element`
- `source_smiles`, `gt_smiles`

`interventions.jsonl` 额外保存选择错误 fragment 的全部构造元数据。`requests.jsonl` 保存本次四组实验的实际输入和哈希。

## 构造约束

- H 相对 N 只替换一个 `ADD_FRAGMENT` 或 `REMOVE_GROUP` 字段。
- 错误 fragment 必须是有效 SMILES，且与正确 fragment 化学结构不同。
- 选择 donor 时依次优先：相同重原子数、相同首连接元素、最小重原子数差、最小字符长度差。
- 正确最终产物从 N/H reasoning 中删除；准备阶段拒绝引号内任何与 GT 主片段等价的 SMILES。
- 所有请求、GT、intervention 文件和模型配置均在 `manifest.json` 中冻结哈希。

## 限制

- 目前只在 ChemDFM-R-14B 上验证，且使用 deterministic greedy decoding。
- fragment 首原子不总是实际连接原子；`same_first_element` 是匹配启发式，并非反应模板证明。
- delete 中只有 18/45 个错误 removal fragment 实际存在于 source，所以 delete 默认属于 diagnostic。
- 子结构采用率不能覆盖所有错误传播路径；输出错误可能表现为其他无效编辑，而不是逐字采用 donor fragment。

## 复现

在 `/home/haoqian/Data/Molecule` 下运行：

```bash
/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_causal_hallucination_v2.py self-check

/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_causal_hallucination_v2.py prepare \
  --output Pilot/Experiments/chemdfm_r14b_causal_hallucination_v2_new

CUDA_VISIBLE_DEVICES=6,7 NCCL_P2P_DISABLE=1 \
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
/home/haoqian/Data/miniconda3/envs/skillopt/bin/python -u \
  Pilot/scripts/run_chemdfm_causal_hallucination_v2.py run \
  --output Pilot/Experiments/chemdfm_r14b_causal_hallucination_v2_new

/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_causal_hallucination_v2.py summarize \
  --output Pilot/Experiments/chemdfm_r14b_causal_hallucination_v2_new
```
