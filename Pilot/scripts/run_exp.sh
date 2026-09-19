#!/usr/bin/env bash
# Run without arguments: bash /home/haoqian/Data/Molecule/Pilot/scripts/run_exp.sh
set -Eeuo pipefail

SCRIPT=/home/haoqian/Data/Molecule/Pilot/scripts/run_exp.sh
LOG=/home/haoqian/Data/Molecule/Pilot/scripts/run_exp.log
REPORT=/home/haoqian/Data/Molecule/Pilot/scripts/run_exp_result.md
PYTHON=/home/haoqian/.venvs/chemllm-outcome/bin/python
RUNNER=/home/haoqian/Data/Molecule/Pilot/scripts/run_chemLLM_outcome.py
DATASET=/home/haoqian/Data/Molecule/Pilot/GeneratedDataset/maximum_edits_complete.jsonl
MODEL_DFM=/home/haoqian/Data/Molecule/chemical_models/ChemDFM-R-14B
MODEL_R=/home/haoqian/Data/Molecule/chemical_models/Chem-R-8B
OUT_DFM=/home/haoqian/Data/Molecule/Pilot/Experiments/chemdfm_r14b_outcome_abcde
OUT_R=/home/haoqian/Data/Molecule/Pilot/Experiments/chem_r8b_outcome_abcde
GPUS=4,5,6,7
TENSOR_PARALLEL=4
BATCH_SIZE=8
LOCK=/tmp/chemllm-outcome-${UID}.lock

if (( $# != 0 )); then
    echo "No arguments are accepted; edit the settings in run_exp.sh." >&2
    exit 2
fi

if [[ ${CHEMLLM_EXP_WORKER:-0} != 1 ]]; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo "An experiment is already running. See $LOG" >&2
        exit 1
    fi
    # The detached worker inherits this lock until the sequential run finishes.
    : > "$LOG"
    nohup setsid env CHEMLLM_EXP_WORKER=1 bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    worker_pid=$!
    echo "Started PID=$worker_pid; log=$LOG"
    echo "Stop both models and workers: kill -TERM -- -$worker_pid"
    exit 0
fi

# setsid makes this shell the process-group leader; all model workers belong to it.
printf 'PID=%s\n' "$$"
printf 'STOP: kill -TERM -- -%s\n' "$$"
printf 'FORCE STOP (if needed): kill -KILL -- -%s\n' "$$"
printf 'Started: %s\nGPU: %s\nOrder: ChemDFM-R-14B -> Chem-R-8B\n\n' "$(date -Is)" "$GPUS"

finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    printf '\nExperiment supervisor exited: status=%s time=%s\n' "$status" "$(date -Is)"
    # Also stop any remaining vLLM children if a stage failed or we were cancelled.
    kill -TERM -- "-$$" 2>/dev/null || true
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="/home/haoqian/.venvs/chemllm-outcome/bin:$PATH"
unset TRANSFORMERS_CACHE
cd /home/haoqian/Data/Molecule/Pilot/scripts

write_report() {
    "$PYTHON" - "$REPORT" "$OUT_DFM" "$OUT_R" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

report = Path(sys.argv[1])
lines = ["# 五组分子编辑实验结果", "", f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
         "A：直接回答；B：加入 H 幻觉推理；C：加入本题 N 正确推理；D：自行推理后回答；E：加入其他题目的 N 正确推理。", "",
         "Accuracy 沿用评估脚本的 primary_accuracy：去除原子映射后，比较预测与标准答案的主片段。",
         "分子相似度为去除原子映射后的完整分子 Morgan 指纹 Tanimoto 相似度均值（0–1）。无效或缺失答案记为 0，计入分母。", ""]
for name, directory in zip(("ChemDFM-R-14B", "Chem-R-8B"), sys.argv[2:]):
    source = Path(directory) / "summary.json"
    summary = json.loads(source.read_text()) if source.exists() else None
    status = ("已完成" if summary["complete"] else "部分结果") if summary else "待完成"
    lines += [f"## {name}", "", f"状态：{status}", "",
              "| 组别 | 已评估样本数 | Accuracy | 分子相似度 |",
              "| :---: | ---: | ---: | ---: |"]
    for group in "ABCDE":
        stats = summary["groups"].get(group) if summary else None
        if stats:
            lines.append(f"| {group} | {stats['n']} | {stats['primary_accuracy']:.2%} | {stats['mean_fts']:.4f} |")
        else:
            lines.append(f"| {group} | — | 待完成 | 待完成 |")
    lines += ["", f"数据来源：`{source}`", ""]
temp = report.with_suffix(".md.tmp")
temp.write_text("\n".join(lines) + "\n")
temp.replace(report)
print(f"Report updated: {report}", flush=True)
PY
}

run_model() {
    local name=$1 model_path=$2 output=$3
    local -a common=(--model "$name" --model-path "$model_path" --output "$output")
    printf '\nMODEL START: %s at %s\nOutput: %s\n' "$name" "$(date -Is)" "$output"
    if [[ ! -f "$output/manifest.json" ]]; then
        "$PYTHON" -u "$RUNNER" prepare "${common[@]}" \
            --dataset "$DATASET" --batch-size "$BATCH_SIZE" --tensor-parallel-size "$TENSOR_PARALLEL"
    else
        echo "Resuming prepared experiment."
        "$PYTHON" - "$output/manifest.json" "$TENSOR_PARALLEL" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest["engine"]["tensor_parallel_size"] != int(sys.argv[2]):
    raise SystemExit("Existing experiment has a different GPU configuration; choose a new output directory in run_exp.sh")
PY
    fi
    "$PYTHON" -u "$RUNNER" run "${common[@]}"
    "$PYTHON" -u "$RUNNER" summarize "${common[@]}"
    write_report
    printf 'MODEL DONE: %s at %s\n' "$name" "$(date -Is)"
}

write_report
"$PYTHON" -u "$RUNNER" self-check
run_model ChemDFM-R-14B "$MODEL_DFM" "$OUT_DFM"
run_model Chem-R-8B "$MODEL_R" "$OUT_R"
echo "ALL EXPERIMENTS COMPLETE"
