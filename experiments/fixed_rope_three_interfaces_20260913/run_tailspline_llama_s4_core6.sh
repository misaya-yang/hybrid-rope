#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp
experiment_root=/root/autodl-tmp
data_manifest=/root/autodl-tmp
model_dir=/root/autodl-tmp
panel=/root/autodl-tmp

mkdir -p "${experiment_root}/tables" "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3 - "${experiment_root}/tables" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipts = {name: json.loads((root / f"{name}.json").read_text()) for name in ("tailspline", "mrpro", "yarn", "bm")}
if {tuple(receipt["band_envelope"]) for receipt in receipts.values()} != {(18, 35)}:
    raise ValueError("Llama TailSpline comparison lacks the canonical common band")
if {float(receipt["gain"]).hex() for receipt in receipts.values()} != {(1.0 + 0.1 * math.log(4.0)).hex()}:
    raise ValueError("Llama TailSpline comparison lacks the common S4 gain")
if receipts["tailspline"]["table"]["construction"].get("fitted_coefficients") != 0:
    raise ValueError("TailSpline is not the zero-fit exact construction")
print(json.dumps({"status": "TAILSPLINE_LLAMA_S4_TABLES_READY_V1", "band": [18, 35], "gain": receipts["tailspline"]["gain"]}))
PY

run_arm() {
  local arm=$1
  local output_dir="${experiment_root}/runs/${arm}"
  if [[ -f "${output_dir}/status.json" ]] && grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3 -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 8192 \
    --length-cap 16384 \
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

/root/miniconda3 -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source tailspline="${experiment_root}/runs/tailspline/generations.jsonl" \
  --source mrpro="${experiment_root}/runs/mrpro/generations.jsonl" \
  --source yarn="${experiment_root}/runs/yarn/generations.jsonl" \
  --source bm="${experiment_root}/runs/bm/generations.jsonl" \
  --candidate tailspline \
  --baseline mrpro \
  --baseline yarn \
  --baseline bm \
  --out "${experiment_root}/reports/tailspline_vs_mrpro_yarn_bm_8k16k32k.json"

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
