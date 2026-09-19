# ChemDFM-R-14B Outcome A/B/C/D v2

本目录保存修正后的 150-pair、600-request 实验。`RESULTS.md` 给出结论和配对统计；`manifest.json` 固定输入、模型元数据、生成参数和脚本哈希。

## 文件

- `requests.jsonl`：600 条实际模型输入。
- `ground_truth.jsonl`：150 条配对 GT。
- `predictions.jsonl`：600 条正式生成结果。
- `predictions.pre_abort_repair.jsonl`：基础设施中断后的原始备份，仅用于审计，不参与正式评分。
- `outcome_records.jsonl`：答案抽取、GT 和逐题指标。
- `summary.json` / `summary.csv`：总体和子任务汇总。
- `runtime.json` / `runtime_history.jsonl`：运行环境与断点历史。
- `run_chemdfm_outcome_v2.frozen.py`：与 manifest 哈希一致的源码快照。脚本按自身路径定位仓库；如工作脚本丢失，请先把快照复制回 `Pilot/scripts/run_chemdfm_outcome_v2.py` 再执行。

## 复跑

在 `/home/haoqian/Data/Molecule` 下执行。`prepare` 只允许写入全新的 output 目录；`run` 和 `summarize` 会验证脚本、requests 和 GT 哈希。推理命令强制要求 `CUDA_VISIBLE_DEVICES=6,7`。

```bash
/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_outcome_v2.py self-check

/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_outcome_v2.py prepare \
  --output Pilot/Experiments/chemdfm_r14b_outcome_abcd_v2_new

CUDA_VISIBLE_DEVICES=6,7 \
NCCL_P2P_DISABLE=1 \
TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=4 \
/home/haoqian/Data/miniconda3/envs/skillopt/bin/python -u \
  Pilot/scripts/run_chemdfm_outcome_v2.py run \
  --output Pilot/Experiments/chemdfm_r14b_outcome_abcd_v2_new

/home/haoqian/Data/miniconda3/bin/python3 \
  Pilot/scripts/run_chemdfm_outcome_v2.py summarize \
  --output Pilot/Experiments/chemdfm_r14b_outcome_abcd_v2_new
```

正式目录已经完成，无需再次执行 `run`。若复跑中出现 `finish_reason=abort`，不要把该行视作完成结果；保留备份、删除对应 abort 行，再以相同冻结配置续跑。

新建实验应优先使用 `Pilot/scripts/run_chemdfm_outcome_v3.py`。它保持同一实验设计，并增加非终止输出保护：`abort` 等记录不会落入正式 predictions，进程会报错退出，下一次启动会自动从这些 request 续跑。当前目录必须继续使用 SHA256 为 `0a36526a...` 的 `Pilot/scripts/run_chemdfm_outcome_v2.py`；目录内的 frozen 文件用于核对或恢复该源码。
