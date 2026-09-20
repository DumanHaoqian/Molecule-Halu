#!/usr/bin/env bash
# No arguments: bash Pilot/scripts/run_agent_exp.sh
set -Eeuo pipefail

ROOT=/mnt_nas1/haoqian/Data/Molecule
SCRIPT="$ROOT/Pilot/scripts/run_agent_exp.sh"
LOG="$ROOT/Pilot/scripts/run_agent_exp.log"
PID_FILE="$ROOT/Pilot/scripts/run_agent_exp.pid"
REPORT="$ROOT/Pilot/scripts/agent_corruption_result.md"
PYTHON=/home/haoqian/.venvs/chemllm-outcome/bin/python
RUNNER="$ROOT/Pilot/scripts/evaluate_agent_corruption.py"
AGGREGATOR="$ROOT/Pilot/scripts/summarize_agent_experiment.py"
GENERATION="$ROOT/Pilot/Experiments/agent_corruption_v13/development18"
DATASET="$GENERATION/pairs.jsonl"
EVALUATION="$GENERATION/evaluation"
MODEL_DFM="$ROOT/chemical_models/ChemDFM-R-14B"
MODEL_R="$ROOT/chemical_models/Chem-R-8B"
OUT_DFM="$EVALUATION/chemdfm_r14b"
OUT_R="$EVALUATION/chem_r8b"
GPUS=4,5,6,7
TENSOR_PARALLEL=4
GPU_MEMORY_UTILIZATION=0.72
GPU_FREE_MARGIN_MIB=1024
BATCH_SIZE=8
RUN_ENTROPY=1
ENTROPY_SAMPLES=8
STARTUP_ATTEMPTS=150
STARTUP_POLL_SECONDS=0.1
LOCK="/tmp/agent-corruption-${UID}.lock"
# The older launcher uses this lock for the same allocation; respect its jobs.
GPU_LOCK="/tmp/chemllm-outcome-${UID}.lock"

if (( $# != 0 )); then
    echo "No arguments are accepted; edit the settings in run_agent_exp.sh." >&2
    exit 2
fi

if [[ ${AGENT_CORRUPTION_WORKER:-0} != 1 ]]; then
    if [[ ! -s "$DATASET" || ! -f "$GENERATION/summary.json" ]]; then
        echo "Generation data is not ready: $GENERATION" >&2
        exit 1
    fi
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo "The agent experiment is already running. See $LOG" >&2
        exit 1
    fi
    exec 8>"$GPU_LOCK"
    if ! flock -n 8; then
        echo "Another chemistry experiment owns the GPU allocation; its job was left running." >&2
        exit 1
    fi
    # The detached worker and its children inherit both locks.
    : > "$LOG"
    rm -f "$PID_FILE"
    nohup setsid env AGENT_CORRUPTION_WORKER=1 bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    worker_pid=$!
    # Keep the launching shell alive until setsid and the worker's PID write
    # have completed. Returning immediately can lose the child in exec clients.
    for (( attempt=0; attempt<STARTUP_ATTEMPTS; attempt++ )); do
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "Detached worker exited before readiness; see $LOG" >&2
            exit 1
        fi
        ready_pid=""
        if [[ -r "$PID_FILE" ]]; then
            read -r ready_pid < "$PID_FILE" || true
        fi
        if [[ "$ready_pid" == "$worker_pid" ]]; then
            printf 'Started PID=%s; log=%s\n' "$worker_pid" "$LOG"
            printf 'Stop this experiment and its workers: kill -TERM -- -%s\n' "$worker_pid"
            exit 0
        fi
        sleep "$STARTUP_POLL_SECONDS"
    done
    echo "Detached startup readiness timed out; see $LOG" >&2
    kill -TERM -- "-$worker_pid" 2>/dev/null || kill -TERM "$worker_pid" 2>/dev/null || true
    exit 1
fi

printf 'PID=%s\n' "$$"
printf 'STOP: kill -TERM -- -%s\n' "$$"
printf 'FORCE STOP (if needed): kill -KILL -- -%s\n' "$$"
printf 'Started: %s\nGPU: %s\nOrder: ChemDFM-R-14B -> Chem-R-8B; then GT probes\n' "$(date -Is)" "$GPUS"
printf 'Dataset: %s\nReport: %s\n\n' "$DATASET" "$REPORT"
printf '%s\n' "$$" > "$PID_FILE"

write_report() {
    "$PYTHON" "$AGGREGATOR" --generation-dir "$GENERATION" \
        --evaluation-dir "$EVALUATION" --output "$REPORT"
}

finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    write_report || true
    printf '\nAgent experiment exited: status=%s time=%s\n' "$status" "$(date -Is)"
    rm -f "$PID_FILE"
    # Only this setsid process group is addressed. External jobs are untouched.
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
cd "$ROOT/Pilot"

# Freeze evaluation only after the planned generation batch is terminal.
"$PYTHON" - "$GENERATION/summary.json" "$GENERATION/manifest.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
if summary["processed"] != summary["planned"]:
    raise SystemExit("Generation is still running; evaluation was not prepared")
if summary["accepted"] < 2:
    raise SystemExit("At least two accepted origins are required for the E control")
if manifest.get("protocol") != "agent_source_state_v1":
    raise SystemExit("Expected the agent_source_state_v1 generation protocol")
PY

wait_for_gpu_capacity() {
    while ! "$PYTHON" - "$GPUS" "$GPU_MEMORY_UTILIZATION" "$GPU_FREE_MARGIN_MIB" <<'PY'
import csv, io, subprocess, sys
gpus = [int(value) for value in sys.argv[1].split(",")]
fraction, margin = float(sys.argv[2]), int(sys.argv[3])
output = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"
], text=True)
available = {int(row[0]): (int(row[1]), int(row[2])) for row in csv.reader(io.StringIO(output))}
waiting = []
for gpu in gpus:
    total, free = available[gpu]
    required = int(total * fraction) + margin
    if free < required:
        waiting.append(f"GPU {gpu}: free={free}MiB required={required}MiB")
if waiting:
    print("Waiting for external jobs to release memory: " + "; ".join(waiting), flush=True)
    raise SystemExit(1)
PY
    do
        sleep 30
    done
}

run_model() {
    local name=$1 model_path=$2 output=$3
    local -a common=(--model "$name" --model-path "$model_path" --output "$output")
    printf '\nMODEL START: %s at %s\nOutput: %s\n' "$name" "$(date -Is)" "$output"
    if [[ ! -f "$output/manifest.json" ]]; then
        "$PYTHON" -u "$RUNNER" prepare "${common[@]}" --dataset "$DATASET" \
            --groups ABCDE --placement prefix --batch-size "$BATCH_SIZE" \
            --tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    else
        "$PYTHON" - "$output/manifest.json" "$DATASET" "$TENSOR_PARALLEL" "$GPU_MEMORY_UTILIZATION" <<'PY'
import json, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1]))
if (manifest["agent_evaluation"]["placement"] != "prefix"
        or manifest["agent_evaluation"]["groups"] != "ABCDE"
        or Path(manifest["dataset"]).resolve() != Path(sys.argv[2]).resolve()
        or manifest["engine"]["tensor_parallel_size"] != int(sys.argv[3])
        or manifest["engine"]["gpu_memory_utilization"] != float(sys.argv[4])):
    raise SystemExit("Prepared experiment configuration differs; select a fresh evaluation directory")
PY
    fi
    wait_for_gpu_capacity
    "$PYTHON" -u "$RUNNER" run "${common[@]}"
    "$PYTHON" -u "$RUNNER" report "${common[@]}"
    write_report
    printf 'MODEL DONE: %s at %s\n' "$name" "$(date -Is)"
}

run_diagnostic() {
    local stage=$1 name=$2 model_path=$3 output=$4
    printf '\n%s START: %s at %s\n' "$stage" "$name" "$(date -Is)"
    wait_for_gpu_capacity
    "$PYTHON" -u "$RUNNER" "$stage" --model "$name" --model-path "$model_path" \
        --output "$output" --entropy-samples "$ENTROPY_SAMPLES"
    write_report
}

write_report
run_model ChemDFM-R-14B "$MODEL_DFM" "$OUT_DFM"
run_model Chem-R-8B "$MODEL_R" "$OUT_R"
run_diagnostic probe ChemDFM-R-14B "$MODEL_DFM" "$OUT_DFM"
run_diagnostic probe Chem-R-8B "$MODEL_R" "$OUT_R"
if (( RUN_ENTROPY == 1 )); then
    run_diagnostic entropy ChemDFM-R-14B "$MODEL_DFM" "$OUT_DFM"
    run_diagnostic entropy Chem-R-8B "$MODEL_R" "$OUT_R"
fi
echo "ALL AGENT EXPERIMENTS COMPLETE"
