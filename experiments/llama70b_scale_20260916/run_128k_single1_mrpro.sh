#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY: run only the 10 matched 128K MrPro niah_single_1 rows and compare with completed TailSpline rows\n'
  exit 0
fi

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
assets=${LLAMA128K_ASSET_ROOT:-${plan}/tailspline_llama_s16_128k_gate}
root=${LLAMA70B_128K_ROOT:-${plan}/llama3_70b_s16_128k_direct_reuse}
model=${LLAMA70B_MODEL:-/root/autodl-tmp/models/llama-3-70b-Instruct-bnb-4bit}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
tailspline_run=${root}/niah/runs/tailspline
mrpro_run=${root}/niah_single1/runs/mrpro
report=${root}/reports/niah_single1_128k.json

mkdir -p "${mrpro_run%/*}" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

exec 9>"${gpu_lock}"
flock -n 9 || { printf 'REFUSE: GPU lock is held\n' >&2; exit 73; }
test -f "${assets}/assets/ready.json" || { printf 'REFUSE: frozen 128K assets are missing\n' >&2; exit 76; }
test -f "${tailspline_run}/generations.jsonl" || { printf 'REFUSE: completed TailSpline generations are missing\n' >&2; exit 76; }
test -f "${tailspline_run}/status.json" || { printf 'REFUSE: completed TailSpline status is missing\n' >&2; exit 76; }

total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${total_mib}" -lt 90000 ]]; then
  printf 'REFUSE: 70B 128K execution requires at least 90,000 MiB; found %s MiB\n' "${total_mib}" >&2
  exit 75
fi

device_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
runtime_report=${root}/runtime/prefill_128k_${device_uuid}.json
test -f "${runtime_report}" || { printf 'REFUSE: validated 128K prefill report is missing\n' >&2; exit 76; }
generation_chunk=$("${python_bin}" - "${runtime_report}" "${tailspline_run}/status.json" "${tailspline_run}/generations.jsonl" <<'PY'
import json
import sys

runtime_path, status_path, generations_path = sys.argv[1:]
runtime = json.load(open(runtime_path))
chunk = runtime.get("recommended_generation_chunk")
if chunk is None:
    raise SystemExit("validated runtime has no safe generation chunk")
status = json.load(open(status_path))
if status != {"status": "COMPLETE", "rows": 80, "lm_rows": 0}:
    raise SystemExit(f"TailSpline NIAH-8 run is not complete: {status}")
rows = [json.loads(line) for line in open(generations_path) if line.strip()]
single = [row for row in rows if row.get("task") == "niah_single_1"]
prompts = {row.get("prompt_sha256") for row in single}
if len(single) != 10 or len(prompts) != 10 or None in prompts:
    raise SystemExit("TailSpline niah_single_1 slice is not exactly 10 unique prompts")
print(int(chunk))
PY
)

if [[ ! -f ${mrpro_run}/status.json ]] || ! "${python_bin}" - "${mrpro_run}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={"status":"COMPLETE","rows":10,"lm_rows":0} else 1)
PY
then
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${assets}/assets/ppl10/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${assets}/assets/full13/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 131072 --prefill-chunk-size "${generation_chunk}" --batch-size 1 \
    --task niah_single_1 --static-table-json "${assets}/tables/mrpro.json" \
    --table-label llama3_70b_nf4_s16_128k_single1_mrpro --out "${mrpro_run}" --execute \
    >"${root}/logs/niah_single1_mrpro.log" 2>&1
fi

"${python_bin}" - "${mrpro_run}/status.json" "${mrpro_run}/generations.jsonl" <<'PY'
import json
import sys

status_path, generations_path = sys.argv[1:]
status = json.load(open(status_path))
if status != {"status": "COMPLETE", "rows": 10, "lm_rows": 0}:
    raise SystemExit(f"MrPro single1 run is not complete: {status}")
rows = [json.loads(line) for line in open(generations_path) if line.strip()]
prompts = {row.get("prompt_sha256") for row in rows}
if len(rows) != 10 or len(prompts) != 10 or None in prompts:
    raise SystemExit("MrPro niah_single_1 output is not exactly 10 unique prompts")
PY

if [[ ! -f ${report} ]]; then
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_point_report \
    --source "tailspline=${tailspline_run}/generations.jsonl" \
    --source "mrpro=${mrpro_run}/generations.jsonl" \
    --candidate tailspline --baseline mrpro --length 131072 --task niah_single_1 \
    --out "${report}"
fi
printf 'LLAMA3_70B_NIAH_SINGLE1_128K_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/single1_complete.txt"
