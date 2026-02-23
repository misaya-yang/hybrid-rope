#!/usr/bin/env bash
set -euo pipefail

# Reviewer-oriented downstream autopilot:
# - keep GPU busy with fair 8B downstream evidence generation
# - auto-resume pending methods if a run stops unexpectedly
# - queue extra robustness seeds to produce reviewer-facing stability evidence

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/dfrope/hybrid-rope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_v2_longbench_stable_20260223_0150}"
LONGBENCH_LOCAL_DATA_DIR="${LONGBENCH_LOCAL_DATA_DIR:-/root/autodl-tmp/dfrope/ms_datasets/LongBench/data}"

WATCH_HOURS="${WATCH_HOURS:-6}"
POLL_SECONDS="${POLL_SECONDS:-90}"

TASKS_REVIEWER="${TASKS_REVIEWER:-qasper,hotpotqa,2wikimqa,multi_news,gov_report,narrativeqa}"
NIAH_LENGTHS="${NIAH_LENGTHS:-4096,8192,16384}"
PASSKEY_LENGTHS="${PASSKEY_LENGTHS:-1024,2048,4096,8192,16384}"

PRIMARY_EVAL_ROOT="${PRIMARY_EVAL_ROOT:-${SUITE_OUTPUT_ROOT}/downstream_eval_autorun}"
SEED123_EVAL_ROOT="${SEED123_EVAL_ROOT:-${SUITE_OUTPUT_ROOT}/downstream_eval_seed123_reviewer}"
SEED7_EVAL_ROOT="${SEED7_EVAL_ROOT:-${SUITE_OUTPUT_ROOT}/downstream_eval_seed7_reviewer}"

METHODS=(baseline pi yarn sigmoid anchored_sigmoid)

log() {
  echo "[$(date '+%F %T %Z')] $*"
}

ensure_repo_latest() {
  if [[ ! -d "${REPO_ROOT}/.git" ]]; then
    log "fatal: repo not found at ${REPO_ROOT}"
    exit 1
  fi
  cd "${REPO_ROOT}"
  # Best effort: stay non-interactive and do not block autopilot on transient network issues.
  git fetch origin --quiet || true
  git pull --ff-only --quiet || true
  log "repo_head=$(git rev-parse --short HEAD)"
}

ensure_longbench_data() {
  cd "${REPO_ROOT}"
  log "ensure LongBench local jsonl (source=auto, fallback includes dashscope/modelscope)"
  if ! "${PYTHON_BIN}" scripts/prepare_longbench_local_data.py \
    --tasks "${TASKS_REVIEWER}" \
    --output_dir "${LONGBENCH_LOCAL_DATA_DIR}" \
    --source auto; then
    log "auto source failed; retry with dashscope(modelscope)"
    "${PYTHON_BIN}" scripts/prepare_longbench_local_data.py \
      --tasks "${TASKS_REVIEWER}" \
      --output_dir "${LONGBENCH_LOCAL_DATA_DIR}" \
      --source dashscope
  fi
}

eval_running() {
  pgrep -af "run_sota_downstream_eval.py|eval_longbench.py|eval_niah_recall.py|eval_passkey_teacher_forcing.py" \
    | grep -v "pgrep -af" >/dev/null 2>&1
}

method_complete() {
  local eval_root="$1"
  local method="$2"
  [[ -s "${eval_root}/niah/${method}/niah_recall_results.json" ]] \
    && [[ -s "${eval_root}/longbench/${method}.json" ]] \
    && [[ -s "${eval_root}/passkey_tf/${method}/passkey_tf_summary.json" ]]
}

pending_methods_csv() {
  local eval_root="$1"
  local pending=()
  local m
  for m in "${METHODS[@]}"; do
    if ! method_complete "${eval_root}" "${m}"; then
      pending+=("${m}")
    fi
  done
  local IFS=,
  echo "${pending[*]}"
}

launch_eval_profile() {
  local eval_root="$1"
  local seed="$2"
  local max_samples="$3"
  local niah_trials="$4"
  local passkey_trials="$5"
  local tasks="$6"
  local pending
  pending="$(pending_methods_csv "${eval_root}")"
  if [[ -z "${pending}" ]]; then
    return 1
  fi

  mkdir -p "${eval_root}"
  cd "${REPO_ROOT}"

  log "launch profile root=${eval_root} seed=${seed} pending_methods=${pending}"
  nohup "${PYTHON_BIN}" -u scripts/run_sota_downstream_eval.py \
    --suite_output_root "${SUITE_OUTPUT_ROOT}" \
    --eval_root "${eval_root}" \
    --methods "${pending}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --longbench_local_data_dir "${LONGBENCH_LOCAL_DATA_DIR}" \
    --longbench_tasks "${tasks}" \
    --longbench_max_samples "${max_samples}" \
    --niah_lengths "${NIAH_LENGTHS}" \
    --niah_trials_per_cell "${niah_trials}" \
    --passkey_lengths "${PASSKEY_LENGTHS}" \
    --passkey_trials_per_cell "${passkey_trials}" \
    --seed "${seed}" \
    >> "${eval_root}/orchestrator.log" 2>&1 &

  local pid="$!"
  echo "${pid}" > "${eval_root}/orchestrator.pid"
  log "profile_pid=${pid} log=${eval_root}/orchestrator.log"
  return 0
}

all_profiles_done() {
  local profile
  for profile in \
    "${PRIMARY_EVAL_ROOT}" \
    "${SEED123_EVAL_ROOT}" \
    "${SEED7_EVAL_ROOT}"; do
    if [[ -n "$(pending_methods_csv "${profile}")" ]]; then
      return 1
    fi
  done
  return 0
}

launch_fallback_stress_if_needed() {
  local stress_root="${SUITE_OUTPUT_ROOT}/reviewer_stress_niah32k_anchored"
  local result_json="${stress_root}/niah_recall_results.json"
  local adapter="${SUITE_OUTPUT_ROOT}/anchored_sigmoid/final_lora"
  local custom_inv="${SUITE_OUTPUT_ROOT}/anchored_sigmoid/artifacts/custom_inv_freq.pt"
  mkdir -p "${stress_root}"

  if [[ -s "${result_json}" ]]; then
    return 1
  fi
  if [[ ! -d "${adapter}" ]]; then
    log "fallback skipped: adapter missing at ${adapter}"
    return 1
  fi

  cd "${REPO_ROOT}"
  log "launch fallback stress NIAH@32K for anchored_sigmoid"
  nohup "${PYTHON_BIN}" -u scripts/eval_niah_recall.py \
    --base_model_path "${BASE_MODEL_PATH}" \
    --adapter_path "${adapter}" \
    --output_dir "${stress_root}" \
    --lengths "32768" \
    --depths "0,10,20,30,40,50,60,70,80,90,100" \
    --trials_per_cell 4 \
    --needles_per_prompt 1 \
    --prompt_mode qa \
    --attn_implementation sdpa \
    --seed 42 \
    --variant custom \
    --custom_inv_freq_path "${custom_inv}" \
    >> "${stress_root}/run.log" 2>&1 &
  log "fallback_pid=$! log=${stress_root}/run.log"
  return 0
}

main() {
  local started_ts
  started_ts="$(date +%s)"
  local deadline_ts=$(( started_ts + WATCH_HOURS * 3600 ))
  log "reviewer_eval_autopilot start watch_hours=${WATCH_HOURS} poll_seconds=${POLL_SECONDS}"

  ensure_repo_latest
  ensure_longbench_data

  while true; do
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts >= deadline_ts )); then
      log "watch window reached; exit autopilot"
      break
    fi

    if eval_running; then
      local running_line
      running_line="$(pgrep -af "run_sota_downstream_eval.py|eval_longbench.py|eval_niah_recall.py|eval_passkey_teacher_forcing.py" | head -n 1 || true)"
      local gpu_apps
      gpu_apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' '; ' || true)"
      log "active_eval=${running_line} gpu_apps=${gpu_apps}"
      sleep "${POLL_SECONDS}"
      continue
    fi

    local launched=0
    launch_eval_profile "${PRIMARY_EVAL_ROOT}" 42 80 2 24 "${TASKS_REVIEWER}" && launched=1 || true
    if (( launched == 0 )); then
      launch_eval_profile "${SEED123_EVAL_ROOT}" 123 80 3 32 "${TASKS_REVIEWER}" && launched=1 || true
    fi
    if (( launched == 0 )); then
      launch_eval_profile "${SEED7_EVAL_ROOT}" 7 60 3 32 "${TASKS_REVIEWER}" && launched=1 || true
    fi

    if (( launched == 0 )) && all_profiles_done; then
      launch_fallback_stress_if_needed && launched=1 || true
    fi

    if (( launched == 0 )); then
      log "no runnable pending profile detected; sleeping"
      sleep "${POLL_SECONDS}"
    else
      sleep 20
    fi
  done
}

main "$@"

