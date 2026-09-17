#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: start CPU asset preparation in parallel, then begin the Native GPU queue as soon as mechanism/capture assets exist.'
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/native_research_20260916
mkdir -p "${root}/logs"
cd "${repo}"

bash experiments/native_enhancement_oral_20260915/prepare_server_cpu.sh \
  >"${root}/logs/cpu_prepare_supervisor.log" 2>&1 &
cpu_pid=$!

deadline=$((SECONDS + 600))
while [[ ! -f "${root}/assets/mechanism_v2/manifest.json" \
      || ! -f "${root}/assets/capture96/manifest.json" ]]; do
  if ! kill -0 "${cpu_pid}" 2>/dev/null; then
    wait "${cpu_pid}"
    printf '%s\n' 'REFUSE: CPU preparation ended before mechanism/capture assets were ready.' >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    printf '%s\n' 'REFUSE: mechanism/capture CPU preparation exceeded 10 minutes.' >&2
    exit 1
  fi
  sleep 2
done

gpu_status=0
bash experiments/native_enhancement_oral_20260915/run_server_gpu.sh --execute \
  >"${root}/logs/gpu_queue_supervisor.log" 2>&1 || gpu_status=$?
cpu_status=0
wait "${cpu_pid}" || cpu_status=$?
if [[ "${gpu_status}" != 0 || "${cpu_status}" != 0 ]]; then
  printf 'Native queue failed: cpu_status=%s gpu_status=%s\n' "${cpu_status}" "${gpu_status}" >&2
  exit 1
fi
printf '%s\n' 'NATIVE_RESEARCH_QUEUE_COMPLETE'
