#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_first
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/llama_low108/screen.jsonl

mkdir -p "${experiment_root}/tables" "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python - "${experiment_root}/tables" "${panel}" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
panel = Path(sys.argv[2])
receipts = {name: json.loads((root / f"{name}.json").read_text()) for name in ("tailspline", "mrpro", "yarn", "bm")}
if {tuple(receipt["band_envelope"]) for receipt in receipts.values()} != {(18, 35)}:
    raise ValueError("Llama TailSpline comparison lacks the canonical common band")
if {float(receipt["gain"]).hex() for receipt in receipts.values()} != {(1.0 + 0.1 * math.log(4.0)).hex()}:
    raise ValueError("Llama TailSpline comparison lacks the common S4 gain")
if receipts["tailspline"]["table"]["construction"].get("fitted_coefficients") != 0:
    raise ValueError("TailSpline is not the zero-fit exact construction")
rows = [json.loads(line) for line in panel.read_text().splitlines() if line.strip()]
selected = [row for row in rows if int(row["length_cap"]) in (8192, 32768)]
counts = Counter((row["task"], int(row["length_cap"])) for row in selected)
if len(selected) != 72 or set(counts.values()) != {6} or len(counts) != 12:
    raise ValueError("the historical llama_low108 panel does not yield the declared 72-row 8K/32K diagnostic")
print(json.dumps({
    "status": "TAILSPLINE_LLAMA_S4_8K32K_DIAGNOSTIC_READY_V1",
    "band": [18, 35], "gain": receipts["tailspline"]["gain"],
    "rows_per_arm": len(selected), "lengths": [8192, 32768],
    "not_main_evidence": True,
}))
PY

run_arm() {
  local arm=$1
  local output_dir="${experiment_root}/runs/${arm}"
  if [[ -f "${output_dir}/status.json" ]] && grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
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
    --length-cap 8192 \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${experiment_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_unified_${arm}" \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro
run_arm yarn
run_arm bm

/root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source tailspline="${experiment_root}/runs/tailspline/generations.jsonl" \
  --source mrpro="${experiment_root}/runs/mrpro/generations.jsonl" \
  --source yarn="${experiment_root}/runs/yarn/generations.jsonl" \
  --source bm="${experiment_root}/runs/bm/generations.jsonl" \
  --candidate tailspline \
  --baseline mrpro \
  --baseline yarn \
  --baseline bm \
  --out "${experiment_root}/reports/tailspline_vs_mrpro_yarn_bm_8k32k_diagnostic.json"

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
