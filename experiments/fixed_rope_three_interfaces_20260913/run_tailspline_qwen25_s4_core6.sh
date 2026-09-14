#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp
experiment_root=/root/autodl-tmp
data_manifest=/root/autodl-tmp
model_dir=/root/autodl-tmp
near_panel=/root/autodl-tmp
far_panel=/root/autodl-tmp

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
  /root/miniconda3 -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${model_dir}/config.json" \
    --method "${method}" \
    --scale 4 \
    --candidate-id "qwen25_3b_s4_unified_${name}" \
    --model-id Qwen2.5-3B-Instruct \
    --role "${role}" \
    --changed-variable exponent_allocation \
    --out "${output}"
}

make_table tailspline tailspline candidate
make_table mrpro mrpro baseline
make_table yarn yarn baseline
make_table bm bm baseline

/root/miniconda3 - "${experiment_root}/tables" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipts = {name: json.loads((root / f"{name}.json").read_text()) for name in ("tailspline", "mrpro", "yarn", "bm")}
gains = {float(receipt["gain"]) for receipt in receipts.values()}
bands = {tuple(receipt["band_envelope"]) for receipt in receipts.values()}
if gains != {1.0 + 0.1 * math.log(4.0)} or bands != {(23, 40)}:
    raise ValueError(f"S4 TailSpline comparison is not unified: gains={gains}, bands={bands}")
if receipts["tailspline"]["table"]["construction"].get("fitted_coefficients") != 0:
    raise ValueError("TailSpline is not the zero-fit exact construction")
print(json.dumps({"status": "TAILSPLINE_S4_UNIFIED_TABLES_READY_V1", "gain": next(iter(gains)), "band": next(iter(bands))}))
PY

run_part() {
  local arm=$1
  local part=$2
  local panel=$3
  local batch_size=$4
  shift 4
  local output_dir="${experiment_root}/runs/${arm}_${part}"
  if [[ -f "${output_dir}/status.json" ]] && grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
    printf 'SKIP_COMPLETE %s_%s\n' "${arm}" "${part}"
    return
  fi
  printf 'START %s_%s %s\n' "${arm}" "${part}" "$(date -u +%FT%TZ)"
  /root/miniconda3 -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    "$@" \
    --prefill-chunk-size 8192 \
    --batch-size "${batch_size}" \
    --static-table-json "${experiment_root}/tables/${arm}.json" \
    --table-label "qwen25_3b_s4_unified_${arm}_${part}" \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/${arm}_${part}.log" 2>&1
  printf 'COMPLETE %s_%s %s\n' "${arm}" "${part}" "$(date -u +%FT%TZ)"
}

run_arm() {
  local arm=$1
  run_part "${arm}" near "${near_panel}" 2 --length-cap 32768 --length-cap 65536
  run_part "${arm}" far "${far_panel}" 1 --length-cap 131072
}

run_arm tailspline
run_arm mrpro
run_arm yarn
run_arm bm

/root/miniconda3 -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source tailspline="${experiment_root}/runs/tailspline_near/generations.jsonl" \
  --source tailspline="${experiment_root}/runs/tailspline_far/generations.jsonl" \
  --source mrpro="${experiment_root}/runs/mrpro_near/generations.jsonl" \
  --source mrpro="${experiment_root}/runs/mrpro_far/generations.jsonl" \
  --source yarn="${experiment_root}/runs/yarn_near/generations.jsonl" \
  --source yarn="${experiment_root}/runs/yarn_far/generations.jsonl" \
  --source bm="${experiment_root}/runs/bm_near/generations.jsonl" \
  --source bm="${experiment_root}/runs/bm_far/generations.jsonl" \
  --candidate tailspline \
  --baseline mrpro \
  --baseline yarn \
  --baseline bm \
  --out "${experiment_root}/reports/tailspline_vs_mrpro_yarn_bm_32k64k128k.json"

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
