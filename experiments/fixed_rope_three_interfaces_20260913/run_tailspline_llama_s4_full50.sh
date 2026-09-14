#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_full50
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
extra_rows="${experiment_root}/assets/full13_extra40/rows.jsonl"
data_manifest="${base_root}/assets/ppl46/manifest.json"

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_llama_full50_assets.sh"

run_arm() {
  local arm=$1
  local run_dir="${experiment_root}/runs/${arm}_extra40"
  if [[ -f "${run_dir}/status.json" ]] && /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 1560, "lm_rows": 0})
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${extra_rows}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 8192 \
    --length-cap 16384 \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${base_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_full50_${arm}_extra40" \
    --out "${run_dir}" \
    --execute >"${experiment_root}/logs/${arm}_extra40.log" 2>&1
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
lengths = (8192, 16384, 32768)
prompt_sets = {}
records = {}
for arm in ("tailspline", "mrpro"):
    base_path = base_root / f"runs/{arm}/generations.jsonl"
    extra_run = root / f"runs/{arm}_extra40"
    extra_path = extra_run / "generations.jsonl"
    status = json.loads((extra_run / "status.json").read_text())
    if status != {"status": "COMPLETE", "rows": 1560, "lm_rows": 0}:
        raise ValueError(f"incomplete Llama extra40 arm: {arm}/{status}")
    base = [json.loads(line) for line in base_path.read_text().splitlines() if line]
    extra = [json.loads(line) for line in extra_path.read_text().splitlines() if line]
    combined = base + extra
    counts = Counter((row["task"], int(row["length_cap"])) for row in combined)
    expected = Counter({(task, length): 50 for task in tasks for length in lengths})
    prompts = {row["prompt_sha256"] for row in combined}
    if len(combined) != 1950 or len(prompts) != 1950 or counts != expected:
        raise ValueError(f"Llama full50 coverage drift: {arm}")
    contract = json.loads((extra_run / "contract.json").read_text())
    receipt = json.loads((base_root / f"tables/{arm}.json").read_text())
    if contract.get("static_table") != receipt.get("table"):
        raise ValueError(f"Llama full50 installed table drift: {arm}")
    prompt_sets[arm] = prompts
    records[arm] = {
        "base_rows": len(base), "extra_rows": len(extra),
        "base_raw_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "extra_raw_sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        "table_sha256_float32": receipt["table_sha256_float32"],
    }
if prompt_sets["tailspline"] != prompt_sets["mrpro"]:
    raise ValueError("Llama full50 arms are not paired on exactly the same prompts")
record = {
    "status": "TAILSPLINE_LLAMA_FULL50_VALIDATED_V1",
    "tasks": list(tasks), "lengths": list(lengths),
    "rows_per_arm": 1950, "rows_per_task_length": 50,
    "records": records,
}
temporary = output.with_name(output.name + ".incomplete")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(record, sort_keys=True))
PY

report="${experiment_root}/reports/tailspline_vs_mrpro_full13_50_per_cell.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source tailspline="${base_root}/runs/tailspline/generations.jsonl" \
    --source tailspline="${experiment_root}/runs/tailspline_extra40/generations.jsonl" \
    --source mrpro="${base_root}/runs/mrpro/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_extra40/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
