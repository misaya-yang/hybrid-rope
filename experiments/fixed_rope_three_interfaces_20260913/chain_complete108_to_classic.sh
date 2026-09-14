#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s COMPLETE108_PID ASSET_PREP_PID\n' "$0" >&2
  exit 2
fi

complete_pid=$1
asset_pid=$2
repo_dir=/root/autodl-tmp/hybrid-rope
completion_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_complete108
classic_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
classic_runner=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_classic.sh

if [[ ! -r "/proc/${complete_pid}/cmdline" ]] || \
   ! tr '\0' ' ' < "/proc/${complete_pid}/cmdline" | \
     grep -q 'run_tailspline_llama_s4_complete108.sh'; then
  printf 'REFUSE: PID %s is not the expected complete108 runner\n' "${complete_pid}" >&2
  exit 1
fi
if [[ -r "/proc/${asset_pid}/cmdline" ]] && \
   ! tr '\0' ' ' < "/proc/${asset_pid}/cmdline" | \
     grep -q 'prepare_tailspline_llama_classic_assets.sh'; then
  printf 'REFUSE: PID %s is not the expected classic asset preparer\n' "${asset_pid}" >&2
  exit 1
fi

while true; do
  completion_ready=true
  for arm in tailspline mrpro yarn bm; do
    status="${completion_root}/runs/${arm}_16k/status.json"
    if [[ ! -f "${status}" ]] || \
       ! grep -q '"status": "COMPLETE"' "${status}" || \
       ! grep -q '"rows": 36' "${status}"; then
      completion_ready=false
    fi
  done
  assets_ready=false
  if [[ -f "${classic_root}/assets/full13/manifest.json" ]] && \
     [[ -f "${classic_root}/assets/ppl46/manifest.json" ]] && \
     grep -q '"status": "COMPLETE"' "${classic_root}/assets/full13/manifest.json" && \
     grep -q '"status": "COMPLETE"' "${classic_root}/assets/ppl46/manifest.json"; then
    assets_ready=true
  fi
  if [[ "${completion_ready}" == true && "${assets_ready}" == true ]]; then
    break
  fi
  if ! kill -0 "${complete_pid}" 2>/dev/null && [[ "${completion_ready}" != true ]]; then
    printf 'REFUSE: complete108 runner exited before all four 16K arms completed\n' >&2
    exit 1
  fi
  if ! kill -0 "${asset_pid}" 2>/dev/null && [[ "${assets_ready}" != true ]]; then
    printf 'REFUSE: classic asset preparation exited before both manifests completed\n' >&2
    exit 1
  fi
  sleep 5
done

cd "${repo_dir}"
exec bash "${classic_runner}"
