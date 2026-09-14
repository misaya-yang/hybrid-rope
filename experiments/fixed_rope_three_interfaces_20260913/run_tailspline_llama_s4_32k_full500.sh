#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_full500
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
extra_rows="${experiment_root}/assets/full13_32k_extra490/rows.jsonl"
data_manifest="${base_root}/assets/ppl46/manifest.json"
batch_size=${LLAMA_FULL500_BATCH_SIZE:-1}

if [[ "${batch_size}" != "1" && "${batch_size}" != "2" ]]; then
  printf 'REFUSE LLAMA_FULL500_BATCH_SIZE must be 1 or 2\n' >&2
  exit 1
fi

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_llama_32k_full500_assets.sh"

run_arm() {
  local arm=$1
  local run_dir="${experiment_root}/runs/${arm}_extra490"
  if [[ -f "${run_dir}/status.json" ]] && /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 6370, "lm_rows": 0})
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  printf 'START %s batch=%s %s\n' "${arm}" "${batch_size}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${extra_rows}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size "${batch_size}" \
    --static-table-json "${base_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_32k_full500_${arm}_extra490" \
    --out "${run_dir}" \
    --execute >"${experiment_root}/logs/${arm}_extra490.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro

validation="${experiment_root}/reports/validation.json"
/root/miniconda3/bin/python - "${base_root}" "${experiment_root}" "${validation}" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

base_root, root, output = map(Path, sys.argv[1:])
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
prompt_sets = {}
records = {}
for arm in ("tailspline", "mrpro"):
    base_path = base_root / f"runs/{arm}/generations.jsonl"
    base_all = [json.loads(line) for line in base_path.read_text().splitlines() if line]
    base = [row for row in base_all if int(row["length_cap"]) == 32768]
    extra_run = root / f"runs/{arm}_extra490"
    extra_path = extra_run / "generations.jsonl"
    extra = [json.loads(line) for line in extra_path.read_text().splitlines() if line]
    status = json.loads((extra_run / "status.json").read_text())
    combined = base + extra
    counts = Counter(row["task"] for row in combined)
    prompts = {row["prompt_sha256"] for row in combined}
    if status != {"status": "COMPLETE", "rows": 6370, "lm_rows": 0}:
        raise ValueError(f"incomplete Llama 32K extra490 arm: {arm}/{status}")
    if len(combined) != 6500 or len(prompts) != 6500:
        raise ValueError(f"Llama 32K full500 prompt coverage drift: {arm}")
    if counts != Counter({task: 500 for task in tasks}):
        raise ValueError(f"Llama 32K full500 task count drift: {arm}/{counts}")
    contract = json.loads((extra_run / "contract.json").read_text())
    receipt = json.loads((base_root / f"tables/{arm}.json").read_text())
    if contract.get("static_table") != receipt.get("table"):
        raise ValueError(f"Llama 32K full500 installed table drift: {arm}")
    prompt_sets[arm] = prompts
    records[arm] = {
        "base_rows": len(base), "extra_rows": len(extra),
        "base_raw_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "extra_raw_sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        "table_sha256_float32": receipt["table_sha256_float32"],
        "batch_size": contract.get("batch_size"),
    }
if prompt_sets["tailspline"] != prompt_sets["mrpro"]:
    raise ValueError("Llama 32K full500 arms are not paired on exactly the same prompts")
record = {
    "status": "TAILSPLINE_LLAMA_32K_FULL500_VALIDATED_V1",
    "tasks": list(tasks), "lengths": [32768],
    "rows_per_arm": 6500, "rows_per_task": 500,
    "records": records,
}
temporary = output.with_name(output.name + ".incomplete")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(record, sort_keys=True))
PY

report="${experiment_root}/reports/tailspline_vs_mrpro_full13_32k_500_per_task.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source tailspline="${base_root}/runs/tailspline/generations.jsonl" \
    --source tailspline="${experiment_root}/runs/tailspline_extra490/generations.jsonl" \
    --source mrpro="${base_root}/runs/mrpro/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_extra490/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --length 32768 \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
