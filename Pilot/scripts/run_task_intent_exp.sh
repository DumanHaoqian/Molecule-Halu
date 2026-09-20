#!/usr/bin/env bash
# No arguments. Fixed four-origin task-interpretation closure diagnostic.
set -Eeuo pipefail

ROOT=/mnt_nas1/haoqian/Data/Molecule
SCRIPT="$ROOT/Pilot/scripts/run_task_intent_exp.sh"
LOG="$ROOT/Pilot/scripts/run_task_intent_exp.log"
PID_FILE="$ROOT/Pilot/scripts/run_task_intent_exp.pid"
EXPERIMENT="$ROOT/Pilot/Experiments/agent_corruption_v17"
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

if [[ ${TASK_INTENT_WORKER:-0} != 1 ]]; then
    for style in binding intent; do
        if [[ ! -s "$EXPERIMENT/$style/pairs.jsonl" || ! -f "$EXPERIMENT/$style/summary.json" ]]; then
            echo "Generation data is not ready: $EXPERIMENT/$style" >&2
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
    nohup setsid env TASK_INTENT_WORKER=1 bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
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
    local report_style
    for report_style in binding intent; do
        "$PYTHON" "$AGGREGATOR" --generation-dir "$EXPERIMENT/$report_style" \
            --evaluation-dir "$EXPERIMENT/$report_style/evaluation" \
            --output "$EXPERIMENT/$report_style/evaluation/agent_corruption_result.md"
    done
}
finish() {
    status=$?
    trap - EXIT
    trap '' TERM INT
    write_reports || true
    printf '\nTask-intent experiment exited: status=%s time=%s\n' "$status" "$(date -Is)"
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
parent_directory = base.parent/'agent_corruption_v16/native_with_connections'
expected = {'mol_edit.add_v2.'+suffix for suffix in ('0098','0172','0193','0194')}
conditions, plans = [], []
for style in ('binding', 'intent'):
    directory = base / style
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (manifest.get('protocol') != 'agent_task_intent_diagnostic_v1'
            or manifest.get('style') != style or manifest.get('diagnostic_only') is not True):
        raise SystemExit('Unexpected task-intent diagnostic protocol/style')
    if Path(manifest.get('parent_generation', '')).resolve() != parent_directory.resolve():
        raise SystemExit('Parent generation differs from frozen v16 connection context')
    if any(summary.get(k) != n for k,n in [('planned',18),('processed',18),('accepted',4),('rejected',14),('outside_scope',14),('production_accepted',0)]):
        raise SystemExit('Expected 18 planned, 4 paired diagnostic origins and 14 outside scope')
    pairs = [json.loads(line) for line in (directory/'pairs.jsonl').read_text().splitlines() if line.strip()]
    indexed = {(p['origin_id'],p['variant_label']):p for p in pairs}
    if len(indexed) != 8 or len(pairs) != 8 or set(indexed) != {(o,v) for o in expected for v in ('N','H')}:
        raise SystemExit('Expected exactly the four fixed origins with both N and H')
    selected_plans = {}
    for origin in expected:
        pair_id = origin+'__agent_task_intent_diagnostic_v1'
        n, h = indexed[origin,'N'], indexed[origin,'H']
        if n['pair_id'] != pair_id or h['pair_id'] != pair_id:
            raise SystemExit('Unexpected condition-neutral pair ID')
        for field in ('instruction','indexed_smiles','final_answer'):
            if n['detector_input'][field] != h['detector_input'][field]:
                raise SystemExit('Immutable question or GT differs between N and H')
        accepted = json.loads((directory/'origins'/origin/'accepted.json').read_text())
        if accepted['origin_id'] != origin or accepted['pair_id'] != pair_id:
            raise SystemExit('Accepted plan identity differs from dataset')
        selected_plans[origin] = {key:accepted['plan'][key] for key in ('edit_plan','execution')}
        roots = accepted['plan']['roots']
        if len(roots) != 1 or roots[0].get('id') != 'r1':
            raise SystemExit('Each style must retain exactly one root')
        expected_root = ('structural','fragment') if style == 'binding' else ('task_interpretation','task_group')
        if (roots[0].get('type'), roots[0].get('node_id')) != expected_root:
            raise SystemExit('Declared root ontology differs from the fixed style contract')
        saved = {p['variant_label']:p for p in accepted['pairs']}
        if saved != {'N':n,'H':h}:
            raise SystemExit('Accepted views differ from dataset')
    conditions.append(indexed)
    plans.append(selected_plans)
left, right = conditions
for key, a in left.items():
    b = right[key]
    for field in ('instruction','indexed_smiles','final_answer'):
        if a['detector_input'][field] != b['detector_input'][field]:
            raise SystemExit('Immutable question or GT differs across styles')
if plans[0] != plans[1]:
    raise SystemExit('Fixed wrong plan or executed product differs across styles')
parent_rows = [json.loads(line) for line in (parent_directory/'pairs.jsonl').read_text().splitlines() if line.strip()]
parent = {(p['origin_id'],p['variant_label']):p for p in parent_rows}
if len(parent_rows) != 8 or set(parent) != set(left):
    raise SystemExit('Frozen v16 binding control coverage differs')
for origin in expected:
    record = json.loads((parent_directory/'origins'/origin/'accepted.json').read_text())
    if plans[0][origin] != {key:record['plan'][key] for key in ('edit_plan','execution')}:
        raise SystemExit('Fixed physical plan differs from frozen v16 parent')
    saved = json.loads((base/'binding/origins'/origin/'accepted.json').read_text())
    if saved['plan']['roots'] != record['plan']['roots']:
        raise SystemExit('Binding control roots differ from frozen parent')
for key, binding in left.items():
    if binding['detector_input'] != parent[key]['detector_input']:
        raise SystemExit('Binding control differs from exact frozen v16 input')
    if key[1] == 'N' and right[key]['detector_input'] != binding['detector_input']:
        raise SystemExit('Intent N changed from the fixed paired control')
    if key[1] == 'H' and right[key]['detector_input']['reasoning_chain'] == binding['detector_input']['reasoning_chain']:
        raise SystemExit('Intent H did not change its task interpretation')
print('Paired preflight passed: 18 planned, 4 shared origins, 14 outside scope; same question/GT and wrong plans')
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
    local name=$1 model_path=$2 slug=$3 style=$4
    local dataset="$EXPERIMENT/$style/pairs.jsonl"
    local output="$EXPERIMENT/$style/evaluation/$slug"
    local -a common=(--model "$name" --model-path "$model_path" --output "$output")
    printf '\nMODEL START: %s style=%s at %s\n' "$name" "$style" "$(date -Is)"
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
        or m['n_pairs'] != 4 or m['n_origins'] != 4 or m['n_requests'] != 20
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
    printf 'MODEL DONE: %s style=%s at %s\n' "$name" "$style" "$(date -Is)"
}

prepare_controls() {
    local name=$1 model_path=$2 slug=$3 condition dataset output
    for condition in binding intent; do
        dataset="$EXPERIMENT/$condition/pairs.jsonl"
        output="$EXPERIMENT/$condition/evaluation/$slug"
        if [[ ! -f "$output/manifest.json" ]]; then
            "$PYTHON" -u "$RUNNER" prepare --model "$name" --model-path "$model_path" --output "$output" \
                --dataset "$dataset" --groups ABCDE --placement prefix --batch-size "$BATCH_SIZE" \
                --tensor-parallel-size "$TENSOR_PARALLEL" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        fi
    done
    # Fixed non-B request validation
    "$PYTHON" - "$EXPERIMENT" "$slug" "$name" "$model_path" \
        "$ROOT/Pilot/Experiments/agent_corruption_v12/development18/evaluation/$slug/manifest.json" <<'PY'
import json, sys
from pathlib import Path
base, slug, name, model_path, runtime_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
runtime = json.loads(runtime_path.read_text())
expected = {'mol_edit.add_v2.'+n for n in ('0098','0172','0193','0194')}
prepared = []
for style in ('binding','intent'):
    directory = base/style/'evaluation'/slug
    m = json.loads((directory/'manifest.json').read_text())
    if (Path(m['dataset']).resolve() != (base/style/'pairs.jsonl').resolve()
            or m['model']['name'] != name or Path(m['model']['path']).resolve() != model_path.resolve()
            or m['n_pairs'] != 4 or m['n_origins'] != 4 or m['n_requests'] != 20
            or m['agent_evaluation']['groups'] != 'ABCDE' or m['agent_evaluation']['placement'] != 'prefix'):
        raise SystemExit('Prepared paired model configuration differs')
    if any(m[key] != runtime[key] for key in ('sampling','engine','batch_size')):
        raise SystemExit('Paired runtime differs from frozen v12 configuration')
    rows = [json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines() if line.strip()]
    indexed = {(r['origin_id'],r['group']):r for r in rows}
    if len(rows) != 20 or set(indexed) != {(o,g) for o in expected for g in 'ABCDE'}:
        raise SystemExit('Prepared paired request coverage differs')
    prepared.append(indexed)
for key, left in prepared[0].items():
    if key[1] in 'ACDE' and left != prepared[1][key]:
        raise SystemExit('Fixed non-B request changed across task-interpretation styles')
print('Prepared A/C/D/E complete requests are identical across styles; all 40 requests match frozen runtime')
PY
}

write_reports
prepare_controls ChemDFM-R-14B "$ROOT/chemical_models/ChemDFM-R-14B" chemdfm_r14b
for style in binding intent; do
    run_model ChemDFM-R-14B "$ROOT/chemical_models/ChemDFM-R-14B" chemdfm_r14b "$style"
done
prepare_controls Chem-R-8B "$ROOT/chemical_models/Chem-R-8B" chem_r8b
for style in binding intent; do
    run_model Chem-R-8B "$ROOT/chemical_models/Chem-R-8B" chem_r8b "$style"
done
echo 'ALL TASK-INTENT EXPERIMENTS COMPLETE'
