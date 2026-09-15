#!/usr/bin/env bash
set -euo pipefail

plan_root=${PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
repo_root=${REPO_ROOT:-/root/autodl-tmp/hybrid-rope}

file_state() {
  local label=$1 path=$2
  if [[ -f "${path}" ]]; then
    printf '  PRESENT   %-24s %s\n' "${label}" "${path}"
  else
    printf '  MISSING   %-24s %s\n' "${label}" "${path}"
  fi
}

row_count() {
  local path=$1
  if [[ -f "${path}" ]]; then
    wc -l <"${path}" | tr -d ' '
  else
    printf '0'
  fi
}

full20=${plan_root}/tailspline_llama_s4_mrrope_niah_heatmap_full20
queue_pid_file=${full20}/queue.pid
queue_state=no_marker_or_pid_file
queue_pid=none
if [[ -f "${full20}/complete.txt" ]]; then
  queue_state=completion_marker_present
elif [[ -f "${queue_pid_file}" ]]; then
  queue_pid=$(tr -d '[:space:]' <"${queue_pid_file}")
  if [[ -n "${queue_pid}" ]] && kill -0 "${queue_pid}" 2>/dev/null; then
    queue_state=pid_alive_unverified
  else
    queue_state=pid_not_alive_no_completion_marker
  fi
fi

printf 'OBSERVED MARKERS, PID LIVENESS AND ROW COUNTS\n'
printf '  NIAH Full20: state=%s pid=%s TailSpline=%s/720 MrPro=%s/720\n' \
  "${queue_state}" "${queue_pid}" \
  "$(row_count "${full20}/runs/tailspline/generations.jsonl")" \
  "$(row_count "${full20}/runs/mrpro/generations.jsonl")"

printf '\nREPORT FILE PRESENCE (content/completion not validated here)\n'
file_state 'Llama classic S4' "${plan_root}/tailspline_llama_s4_classic/reports/tailspline_vs_mrpro_classic.json"
file_state 'OLMo classic S4' "${plan_root}/tailspline_olmo_s4_classic/reports/tailspline_vs_mrpro_classic.json"
file_state 'Qwen 32K/64K' "${plan_root}/tailspline_qwen25_s2_32k64k/reports/tailspline_vs_mrpro_32k64k.json"
file_state 'Llama clean 32K' "${plan_root}/tailspline_llama_s4_32k_ruler200_clean/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json"
file_state 'Llama clean 16K' "${plan_root}/tailspline_llama_s4_16k_ruler50_clean/reports/tailspline_vs_mrpro_full13_16k_50_per_task_clean.json"
file_state 'Natural-QA631' "${plan_root}/tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json"
file_state 'Native-Z5 V1' "${plan_root}/olmo_native_z5_enhancement/reports/native_vs_z5.json"
file_state 'Native-Z5 consensus' "${plan_root}/olmo_native_z5_enhancement/reports/native_z5_consensus.json"
file_state 'NIAH pilot' "${plan_root}/tailspline_llama_s4_mrrope_niah_heatmap/reports/tailspline_vs_mrpro_niah_heatmap.json"
file_state 'ProofPile32 PPL' "${plan_root}/tailspline_llama_s4_mrrope_niah_heatmap/reports/proofpile32_ppl_curve.json"
file_state 'Prefill 32K/32GB' "${plan_root}/prefill_chunk_benchmarks/llama32k_32gb_v2.json"
file_state 'NIAH Full20' "${full20}/reports/tailspline_vs_mrpro_niah_heatmap_full20.json"

printf '\nREQUIRES 48GB OR MORE\n'
highmem=${plan_root}/tailspline_llama_s16_128k_gate
if [[ -f "${highmem}/assets/ready.json" ]]; then
  printf '  PRESENT   Llama S16 ready marker %s\n' "${highmem}/assets/ready.json"
else
  printf '  MISSING   Llama S16 128K assets   run prepare_llama_s16_128k_assets.sh\n'
fi
printf '  LAUNCHER  %s/experiments/iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh\n' "${repo_root}"
printf '  CONTRACT  refuses GPUs below 45000 MiB; prefill strategies are selected on the destination GPU\n'

printf '\nDO NOT LAUNCH AS A QUEUE\n'
printf '  run_original_gpu_queue.sh and run_clone_gpu_queue.sh are dated sprint wrappers, not current priorities.\n'
printf '  Completed directories and canaries remain evidence sources; their presence is not pending work.\n'
