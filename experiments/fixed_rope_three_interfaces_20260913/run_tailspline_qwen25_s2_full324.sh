#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_unified_full324
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
panel=/root/autodl-tmp/band_mini_20260913/qwen_s2/frozen_324/screen.jsonl

mkdir -p "${experiment_root}/tables" "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

make_table() {
  local name=$1
  local method=$2
  local role=$3
  local output="${experiment_root}/tables/${name}.json"
  if [[ -e "${output}" ]]; then
    return
  fi
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${model_dir}/config.json" \
    --method "${method}" \
    --scale 2 \
    --candidate-id "qwen25_3b_s2_unified_${name}" \
    --model-id Qwen2.5-3B-Instruct \
    --role "${role}" \
    --changed-variable exponent_allocation \
    --out "${output}"
}

make_table tailspline tailspline candidate
make_table mrpro mrpro baseline
make_table yarn yarn baseline
make_table bm bm baseline

/root/miniconda3/bin/python - "${experiment_root}/tables" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipts = {name: json.loads((root / f"{name}.json").read_text()) for name in ("tailspline", "mrpro", "yarn", "bm")}
gains = {float(receipt["gain"]) for receipt in receipts.values()}
bands = {tuple(receipt["band_envelope"]) for receipt in receipts.values()}
if len(gains) != 1 or len(bands) != 1:
    raise ValueError(f"unified TailSpline comparison must share gain/band, got gains={gains}, bands={bands}")
if receipts["tailspline"]["role"] != "candidate":
    raise ValueError("TailSpline receipt is not the frozen candidate")
print(json.dumps({"status": "TAILSPLINE_UNIFIED_TABLES_READY_V1", "gain": next(iter(gains)), "band": next(iter(bands))}))
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
    --length-cap 32768 \
    --length-cap 49152 \
    --length-cap 65536 \
    --prefill-chunk-size 8192 \
    --batch-size 2 \
    --static-table-json "${experiment_root}/tables/${arm}.json" \
    --table-label "qwen25_3b_s2_unified_${arm}" \
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
  --out "${experiment_root}/reports/tailspline_vs_mrpro_yarn_bm_32k48k64k.json"

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
