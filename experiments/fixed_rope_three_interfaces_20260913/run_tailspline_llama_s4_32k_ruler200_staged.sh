#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
source_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_full500/assets/full13_32k_extra490_parallel/parts
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
data_manifest=${base_root}/assets/ppl46/manifest.json
batch_size=${LLAMA_RULER200_BATCH_SIZE:-2}

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
arms=(tailspline mrpro)

mkdir -p "${experiment_root}/assets/parts" "${experiment_root}/runs" \
  "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_for_task_asset() {
  local task=$1
  local manifest=${source_root}/${task}/manifest.json
  while ! /root/miniconda3/bin/python - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    value = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(not (value.get("status") == "COMPLETE" and int(value.get("rows", 0)) >= 190))
PY
  do
    sleep 15
  done
}

prepare_task_slice() {
  local task=$1
  local source=${source_root}/${task}/rows.jsonl
  local target=${experiment_root}/assets/parts/${task}_extra190.jsonl
  if [[ ! -f "${target}" ]] || [[ $(wc -l < "${target}") != 190 ]]; then
    head -n 190 "${source}" > "${target}.incomplete"
    test "$(wc -l < "${target}.incomplete")" = 190
    mv "${target}.incomplete" "${target}"
  fi
  /root/miniconda3/bin/python - "${target}" "${base_root}/assets/full13/rows.jsonl" "${task}" <<'PY'
import json
import sys

target_path, base_path, task = sys.argv[1:]
target = [json.loads(line) for line in open(target_path) if line.strip()]
base = [
    json.loads(line) for line in open(base_path) if line.strip()
]
base = [row for row in base if row["task"] == task and int(row["length_cap"]) == 32768]
target_prompts = {row["prompt_sha256"] for row in target}
base_prompts = {row["prompt_sha256"] for row in base}
if len(target) != 190 or len(target_prompts) != 190 or {row["task"] for row in target} != {task}:
    raise ValueError(f"invalid staged RULER-200 slice: {task}")
if {int(row["length_cap"]) for row in target} != {32768} or len(base) != 10:
    raise ValueError(f"invalid staged/base length coverage: {task}")
if target_prompts & base_prompts:
    raise ValueError(f"staged RULER-200 prompts overlap the frozen base10: {task}")
PY
}

run_task_arm() {
  local task=$1
  local arm=$2
  local panel=${experiment_root}/assets/parts/${task}_extra190.jsonl
  local run_dir=${experiment_root}/runs/${arm}_${task}_extra190
  local status=${run_dir}/status.json
  if [[ -f "${status}" ]] && /root/miniconda3/bin/python - "${status}" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 190, "lm_rows": 0})
PY
  then
    printf 'SKIP_COMPLETE %s %s\n' "${task}" "${arm}"
    return
  fi
  printf 'START %s %s batch=%s %s\n' "${task}" "${arm}" "${batch_size}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size "${batch_size}" \
    --static-table-json "${base_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_32k_ruler200_${arm}_${task}" \
    --out "${run_dir}" \
    --execute >"${experiment_root}/logs/${arm}_${task}_extra190.log" 2>&1
  printf 'COMPLETE %s %s %s\n' "${task}" "${arm}" "$(date -u +%FT%TZ)"
}

# A manually started first stage may still own the GPU when this supervisor starts.
if [[ -f "${experiment_root}/gpu_stage.pid" ]]; then
  first_pid=$(cat "${experiment_root}/gpu_stage.pid")
  while ps -p "${first_pid}" -o args= | grep -q recovery_v2_eval; do
    sleep 15
  done
fi

for task in "${tasks[@]}"; do
  wait_for_task_asset "${task}"
  prepare_task_slice "${task}"
  for arm in "${arms[@]}"; do
    run_task_arm "${task}" "${arm}"
  done
done

validation=${experiment_root}/reports/validation.json
/root/miniconda3/bin/python - "${base_root}" "${experiment_root}" "${validation}" "${tasks[@]}" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

base_root = Path(sys.argv[1])
root = Path(sys.argv[2])
output = Path(sys.argv[3])
tasks = sys.argv[4:]
prompt_sets = {}
records = {}
for arm in ("tailspline", "mrpro"):
    base_path = base_root / f"runs/{arm}/generations.jsonl"
    base_all = [json.loads(line) for line in base_path.read_text().splitlines() if line]
    combined = [row for row in base_all if int(row["length_cap"]) == 32768]
    stage_hashes = {}
    for task in tasks:
        run_dir = root / f"runs/{arm}_{task}_extra190"
        status = json.loads((run_dir / "status.json").read_text())
        if status != {"status": "COMPLETE", "rows": 190, "lm_rows": 0}:
            raise ValueError(f"incomplete staged arm: {arm}/{task}/{status}")
        raw = run_dir / "generations.jsonl"
        values = [json.loads(line) for line in raw.read_text().splitlines() if line]
        if len(values) != 190 or {row["task"] for row in values} != {task}:
            raise ValueError(f"staged task identity drift: {arm}/{task}")
        combined.extend(values)
        stage_hashes[task] = hashlib.sha256(raw.read_bytes()).hexdigest()
    counts = Counter(row["task"] for row in combined)
    prompts = {row["prompt_sha256"] for row in combined}
    if len(combined) != 2600 or len(prompts) != 2600:
        raise ValueError(f"RULER-200 coverage drift: {arm}")
    if counts != Counter({task: 200 for task in tasks}):
        raise ValueError(f"RULER-200 task count drift: {arm}/{counts}")
    prompt_sets[arm] = prompts
    records[arm] = {
        "base_raw_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "stage_raw_sha256": stage_hashes,
        "table_sha256_float32": json.loads((base_root / f"tables/{arm}.json").read_text())["table_sha256_float32"],
    }
if prompt_sets["tailspline"] != prompt_sets["mrpro"]:
    raise ValueError("RULER-200 arms do not use exactly paired prompts")
record = {
    "status": "TAILSPLINE_LLAMA_32K_RULER200_VALIDATED_V1",
    "tasks": tasks,
    "lengths": [32768],
    "rows_per_arm": 2600,
    "rows_per_task": 200,
    "records": records,
}
temporary = output.with_name(output.name + ".incomplete")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(record, sort_keys=True))
PY

report=${experiment_root}/reports/tailspline_vs_mrpro_full13_32k_200_per_task.json
report_args=()
for arm in "${arms[@]}"; do
  report_args+=(--source "${arm}=${base_root}/runs/${arm}/generations.jsonl")
  for task in "${tasks[@]}"; do
    report_args+=(--source "${arm}=${experiment_root}/runs/${arm}_${task}_extra190/generations.jsonl")
  done
done
/root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  "${report_args[@]}" \
  --candidate tailspline \
  --baseline mrpro \
  --length 32768 \
  --out "${report}"

printf 'RULER200_COMPLETE %s\n' "$(date -u +%FT%TZ)"
