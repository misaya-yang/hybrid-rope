#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_full50
table_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_unified_full324/tables
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
panel="${experiment_root}/assets/full13_50/rows.jsonl"

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_qwen_full50_assets.sh"

run_arm() {
  local arm=$1
  local run_dir="${experiment_root}/runs/${arm}"
  if [[ -f "${run_dir}/status.json" ]] && /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 1300, "lm_rows": 0})
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
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --length-cap 65536 \
    --prefill-chunk-size 8192 \
    --batch-size 2 \
    --static-table-json "${table_root}/${arm}.json" \
    --table-label "qwen25_3b_s2_full50_${arm}" \
    --out "${run_dir}" \
    --execute >"${experiment_root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro

validation="${experiment_root}/reports/validation.json"
/root/miniconda3/bin/python - "${experiment_root}" "${table_root}" "${validation}" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

root, table_root, output = map(Path, sys.argv[1:])
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
lengths = (32768, 65536)
prompt_sets = {}
records = {}
for arm in ("tailspline", "mrpro"):
    run = root / f"runs/{arm}"
    raw = run / "generations.jsonl"
    status = json.loads((run / "status.json").read_text())
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line]
    counts = Counter((row["task"], int(row["length_cap"])) for row in rows)
    expected = Counter({(task, length): 50 for task in tasks for length in lengths})
    prompts = {row["prompt_sha256"] for row in rows}
    if status != {"status": "COMPLETE", "rows": 1300, "lm_rows": 0}:
        raise ValueError(f"incomplete Qwen full50 arm: {arm}/{status}")
    if len(rows) != 1300 or len(prompts) != 1300 or counts != expected:
        raise ValueError(f"Qwen full50 coverage drift: {arm}")
    contract = json.loads((run / "contract.json").read_text())
    receipt = json.loads((table_root / f"{arm}.json").read_text())
    if contract.get("static_table") != receipt.get("table"):
        raise ValueError(f"Qwen full50 installed table drift: {arm}")
    prompt_sets[arm] = prompts
    records[arm] = {
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "table_sha256_float32": receipt["table_sha256_float32"],
    }
if prompt_sets["tailspline"] != prompt_sets["mrpro"]:
    raise ValueError("Qwen full50 arms are not paired on exactly the same prompts")
record = {
    "status": "TAILSPLINE_QWEN_FULL50_VALIDATED_V1",
    "tasks": list(tasks), "lengths": list(lengths),
    "rows_per_arm": 1300, "rows_per_task_length": 50,
    "records": records,
}
temporary = output.with_name(output.name + ".incomplete")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(record, sort_keys=True))
PY

report="${experiment_root}/reports/tailspline_vs_mrpro_full13_50_per_cell_32k64k.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source tailspline="${experiment_root}/runs/tailspline/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
