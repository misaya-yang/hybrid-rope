#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s FULL13_PREP_PID PPL46_PREP_PID\n' "$0" >&2
  exit 2
fi

full_pid=$1
ppl_pid=$2
repo_dir=/root/autodl-tmp/hybrid-rope
classic_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
full_manifest=${classic_root}/assets/full13/manifest.json
ppl_manifest=${classic_root}/assets/ppl46/manifest.json
prepare=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_llama_classic_assets.sh
runner=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_classic.sh

while true; do
  full_ready=false
  ppl_ready=false
  if [[ -f "${full_manifest}" ]] && grep -q '"status": "COMPLETE"' "${full_manifest}"; then
    full_ready=true
  fi
  if [[ -f "${ppl_manifest}" ]] && grep -q '"status": "COMPLETE"' "${ppl_manifest}"; then
    ppl_ready=true
  fi
  if [[ "${full_ready}" == true && "${ppl_ready}" == true ]]; then
    break
  fi
  if ! kill -0 "${full_pid}" 2>/dev/null && [[ "${full_ready}" != true ]]; then
    printf 'REFUSE: Full-13 preparation exited without a complete manifest\n' >&2
    exit 1
  fi
  if ! kill -0 "${ppl_pid}" 2>/dev/null && [[ "${ppl_ready}" != true ]]; then
    printf 'REFUSE: PPL46 preparation exited without a complete manifest\n' >&2
    exit 1
  fi
  sleep 5
done

cd "${repo_dir}"
bash "${prepare}"
exec bash "${runner}"
