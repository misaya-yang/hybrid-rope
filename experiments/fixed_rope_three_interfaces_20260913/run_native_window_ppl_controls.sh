#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914

cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_native() {
  local label=$1
  local model=$2
  local data=$3
  local length=$4
  local out=$5
  local log=$6
  local status="${out}/status.json"
  if [[ -f "${status}" ]] && /root/miniconda3/bin/python - "${status}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
raise SystemExit(value.get("status") != "COMPLETE" or value.get("rows") != 0 or value.get("lm_rows") != 46)
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${label}"
    return
  fi
  mkdir -p "$(dirname "${log}")"
  printf 'START %s %s\n' "${label}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data}" \
    --model "${model}" \
    --arm Native \
    --only-extra-panels \
    --lm-length-cap "${length}" \
    --out "${out}" \
    --execute >"${log}" 2>&1
  printf 'COMPLETE %s %s\n' "${label}" "$(date -u +%FT%TZ)"
}

run_native \
  llama_native_8k \
  /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct \
  "${root}/tailspline_llama_s4_classic/assets/ppl46/manifest.json" \
  8192 \
  "${root}/tailspline_llama_s4_classic/runs/native_ppl_8k" \
  "${root}/tailspline_llama_s4_classic/logs/native_ppl_8k.log"

run_native \
  olmo_native_4k \
  /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  "${root}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json" \
  4096 \
  "${root}/tailspline_olmo_s4_classic/runs/native_ppl_4k" \
  "${root}/tailspline_olmo_s4_classic/logs/native_ppl_4k.log"

printf 'NATIVE_PPL_CONTROLS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
