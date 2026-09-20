#!/usr/bin/env bash
# No arguments. Fixed paired CoT-order diagnostic, all v12 accepted parents.
set -Eeuo pipefail

ROOT=/mnt_nas1/haoqian/Data/Molecule
SCRIPT="$ROOT/Pilot/scripts/run_edit_order_exp.sh"
LOG="$ROOT/Pilot/scripts/run_edit_order_exp.log"
PID_FILE="$ROOT/Pilot/scripts/run_edit_order_exp.pid"
EXPERIMENT="$ROOT/Pilot/Experiments/agent_corruption_v14"
PYTHON=/home/haoqian/.venvs/chemllm-outcome/bin/python
RUNNER="$ROOT/Pilot/scripts/evaluate_agent_corruption.py"
AGGREGATOR="$ROOT/Pilot/scripts/summarize_agent_experiment.py"
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

if [[ ${EDIT_ORDER_WORKER:-0} != 1 ]]; then
    for order in account_last edit_last; do
        if [[ ! -s "$EXPERIMENT/$order/pairs.jsonl" || ! -f "$EXPERIMENT/$order/summary.json" ]]; then
            echo "Generation data is not ready: $EXPERIMENT/$order" >&2
            exit 1
        fi
    done
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
    nohup setsid env EDIT_ORDER_WORKER=1 bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
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
printf 'Started: %s\nGPUs: %s\nOrder: ChemDFM both conditions -> Chem-R both conditions\n' "$(date -Is)" "$GPUS"
printf '%s\n' "$$" > "$PID_FILE"

write_reports() {
    local report_order
    for report_order in account_last edit_last; do
        "$PYTHON" "$AGGREGATOR" --generation-dir "$EXPERIMENT/$report_order" \
            --evaluation-dir "$EXPERIMENT/$report_order/evaluation" \
            --output "$EXPERIMENT/$report_order/evaluation/agent_corruption_result.md"
    done
}
finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    write_reports || true
    printf '\nEdit-order experiment exited: status=%s time=%s\n' "$status" "$(date -Is)"
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

# Paired CPU preflight
"$PYTHON" - "$EXPERIMENT" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
conditions = []
for order in ('account_last', 'edit_last'):
    directory = base / order
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if manifest.get('protocol') != 'agent_edit_order_diagnostic_v1' or manifest.get('order') != order:
        raise SystemExit('Unexpected diagnostic protocol/order')
    if any(summary.get(k) != n for k,n in [('planned',18),('processed',18),('accepted',14),('rejected',4)]):
        raise SystemExit('Expected all18 parents,14 paired diagnostics and4 inherited rejections')
    pairs = [json.loads(line) for line in (directory/'pairs.jsonl').read_text().splitlines() if line]
    indexed = {(p['origin_id'],p['variant_label']):p for p in pairs}
    if len(indexed) != 28 or len(pairs) != 28 or {label for _,label in indexed} != {'N','H'}:
        raise SystemExit('Expected14 unique N/H pairs')
    origins = {origin for origin,_ in indexed}
    if len(origins) != 14 or any((origin,label) not in indexed for origin in origins for label in ('N','H')):
        raise SystemExit('Every origin must have both N and H')
    conditions.append(indexed)
left, right = conditions
if left.keys() != right.keys():
    raise SystemExit('Paired order origin coverage differs')
for key, a in left.items():
    b = right[key]
    if a['pair_id'] != b['pair_id']:
        raise SystemExit('Pair IDs differ across orders')
    for field in ('instruction','indexed_smiles','final_answer'):
        if a['detector_input'][field] != b['detector_input'][field]:
            raise SystemExit('Immutable question or GT differs across orders')
print('Paired preflight passed:18 planned,14 shared accepted origins; no question changes')
PY

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
    local name=$1 model_path=$2 slug=$3 order=$4
    local dataset="$EXPERIMENT/$order/pairs.jsonl"
    local output="$EXPERIMENT/$order/evaluation/$slug"
    local -a common=(--model "$name" --model-path "$model_path" --output "$output")
    printf '\nMODEL START: %s order=%s at %s\n' "$name" "$order" "$(date -Is)"
    if [[ ! -f "$output/manifest.json" ]]; then
        "$PYTHON" -u "$RUNNER" prepare "${common[@]}" --dataset "$dataset" \
            --groups ABCDE --placement prefix --batch-size "$BATCH_SIZE" \
            --tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    fi
    "$PYTHON" - "$output/manifest.json" "$dataset" "$name" "$model_path" \
        "$ROOT/Pilot/Experiments/agent_corruption_v12/development18/evaluation/$slug/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m = json.load(open(sys.argv[1]))
parent = json.load(open(sys.argv[5]))
if (Path(m['dataset']).resolve() != Path(sys.argv[2]).resolve()
        or m['model']['name'] != sys.argv[3] or Path(m['model']['path']).resolve() != Path(sys.argv[4]).resolve()
        or m['agent_evaluation']['placement'] != 'prefix' or m['agent_evaluation']['groups'] != 'ABCDE'
        or m['engine']['tensor_parallel_size'] != 4 or m['engine']['gpu_memory_utilization'] != 0.72):
    raise SystemExit('Prepared experiment configuration differs; use a fresh directory')
for key in ('sampling', 'engine', 'batch_size'):
    if m[key] != parent[key]:
        raise SystemExit('Prepared '+key+' differs from the frozen v12 runtime control')
PY
    wait_for_gpu_capacity
    "$PYTHON" -u "$RUNNER" run "${common[@]}"
    "$PYTHON" -u "$RUNNER" report "${common[@]}"
    write_reports
    printf 'MODEL DONE: %s order=%s at %s\n' "$name" "$order" "$(date -Is)"
}

write_reports
for order in account_last edit_last; do
    run_model ChemDFM-R-14B "$ROOT/chemical_models/ChemDFM-R-14B" chemdfm_r14b "$order"
done
for order in account_last edit_last; do
    run_model Chem-R-8B "$ROOT/chemical_models/Chem-R-8B" chem_r8b "$order"
done
echo 'ALL EDIT-ORDER EXPERIMENTS COMPLETE'
