#!/usr/bin/env bash
# No arguments. Full150 exploratory ABC task-interpretation experiment.
set -Eeuo pipefail

ROOT=/mnt_nas1/haoqian/Data/Molecule
SCRIPT="$ROOT/Pilot/scripts/run_full150_exp.sh"
LOG="$ROOT/Pilot/scripts/run_full150_exp.log"
PID_FILE="$ROOT/Pilot/scripts/run_full150_exp.pid"
EXPERIMENT="$ROOT/Pilot/Experiments/agent_corruption_v18/full150"
PYTHON=/home/haoqian/.venvs/chemllm-outcome/bin/python
RUNNER="$ROOT/Pilot/scripts/evaluate_agent_corruption.py"
AGGREGATOR="$ROOT/Pilot/scripts/summarize_full150_exp.py"
GPUS=4,5,6,7
TENSOR_PARALLEL=4
GPU_MEMORY_UTILIZATION=0.72
GPU_FREE_MARGIN_MIB=1024
BATCH_SIZE=8
LOCK="/tmp/agent-corruption-${UID}.lock"
GPU_LOCK="/tmp/chemllm-outcome-${UID}.lock"

if (( $# != 0 )); then
    echo 'No arguments are accepted; settings are written in the script.' >&2
    exit 2
fi

if [[ ${FULL150_WORKER:-0} != 1 ]]; then
    "$PYTHON" "$AGGREGATOR" --check-only --generation-dir "$EXPERIMENT"
    exec 9>"$LOCK"
    if ! flock -n 9; then
        echo 'Another agent experiment is running; its job was left running.' >&2
        exit 1
    fi
    exec 8>"$GPU_LOCK"
    if ! flock -n 8; then
        echo 'Another chemistry experiment owns the GPU allocation.' >&2
        exit 1
    fi
    : > "$LOG"
    rm -f "$PID_FILE"
    nohup setsid env FULL150_WORKER=1 bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    worker_pid=$!
    for (( attempt=0; attempt<150; attempt++ )); do
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
        sleep 0.1
    done
    echo "Detached startup readiness timed out; see $LOG" >&2
    kill -TERM -- "-$worker_pid" 2>/dev/null || kill -TERM "$worker_pid" 2>/dev/null || true
    exit 1
fi

printf 'PID=%s\nSTOP: kill -TERM -- -%s\n' "$$" "$$"
printf 'Started: %s\nGPUs: %s\nOrder: ChemDFM -> Chem-R\n' "$(date -Is)" "$GPUS"
printf '%s\n' "$$" > "$PID_FILE"

write_reports() {
    local report_slug report_name
    for report_slug in chemdfm_r14b chem_r8b; do
        report_name=ChemDFM-R-14B
        if [[ "$report_slug" == chem_r8b ]]; then report_name=Chem-R-8B; fi
        if [[ -f "$EXPERIMENT/evaluation/$report_slug/manifest.json" ]]; then
            "$PYTHON" "$RUNNER" report --model "$report_name" --model-path "$ROOT/chemical_models/$report_name" --output "$EXPERIMENT/evaluation/$report_slug" --allow-partial || true
        fi
    done
    "$PYTHON" "$AGGREGATOR" --generation-dir "$EXPERIMENT"
}
finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    write_reports || true
    printf '\nFull150 experiment exited: status=%s time=%s\n' "$status" "$(date -Is)"
    rm -f "$PID_FILE"
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

"$PYTHON" "$AGGREGATOR" --check-only --generation-dir "$EXPERIMENT"

wait_for_gpu_capacity() {
    while ! "$PYTHON" - "$GPUS" "$GPU_MEMORY_UTILIZATION" "$GPU_FREE_MARGIN_MIB" <<'PY'
import csv, io, subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(',')]
fraction, margin = float(sys.argv[2]), int(sys.argv[3])
output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'], text=True)
available = {int(row[0]):(int(row[1]),int(row[2])) for row in csv.reader(io.StringIO(output))}
waiting = [f'GPU {gpu}:free={available[gpu][1]}MiB' for gpu in gpus
           if available[gpu][1] < int(available[gpu][0]*fraction)+margin]
if waiting:
    print('Waiting for external jobs: '+'; '.join(waiting), flush=True)
    raise SystemExit(1)
PY
    do
        sleep 30
    done
}

run_model() {
    local name=$1 model_path=$2 slug=$3
    local dataset="$EXPERIMENT/pairs.jsonl"
    local output="$EXPERIMENT/evaluation/$slug"
    local -a common=(--model "$name" --model-path "$model_path" --output "$output")
    printf '\nMODEL START: %s at %s\n' "$name" "$(date -Is)"
    if [[ ! -f "$output/manifest.json" ]]; then
        "$PYTHON" -u "$RUNNER" prepare "${common[@]}" --dataset "$dataset" \
            --groups ABC --placement prefix --batch-size "$BATCH_SIZE" \
            --tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    fi
    "$PYTHON" "$AGGREGATOR" --check-only --prepared-folder "$slug"
    "$PYTHON" - "$output/manifest.json" "$dataset" "$name" "$model_path" \
        "$ROOT/Pilot/Experiments/agent_corruption_v12/development18/evaluation/$slug/manifest.json" <<'PYCODE'
import json, sys
from pathlib import Path
m, parent = [json.load(open(p)) for p in (sys.argv[1],sys.argv[5])]
if (Path(m['dataset']).resolve() != Path(sys.argv[2]).resolve()
        or m['model']['name'] != sys.argv[3] or Path(m['model']['path']).resolve() != Path(sys.argv[4]).resolve()):
    raise SystemExit('Prepared model/dataset differs; use a fresh directory')
for key in ('sampling', 'engine', 'batch_size'):
    if m[key] != parent[key]:
        raise SystemExit('Prepared '+key+' differs from frozen v12 runtime')
PYCODE
    wait_for_gpu_capacity
    "$PYTHON" -u "$RUNNER" run "${common[@]}"
    "$PYTHON" -u "$RUNNER" report "${common[@]}"
    write_reports
    printf 'MODEL DONE: %s at %s\n' "$name" "$(date -Is)"
}

write_reports
run_model ChemDFM-R-14B "$ROOT/chemical_models/ChemDFM-R-14B" chemdfm_r14b
run_model Chem-R-8B "$ROOT/chemical_models/Chem-R-8B" chem_r8b
echo 'ALL FULL150 ABC EXPERIMENTS COMPLETE'
