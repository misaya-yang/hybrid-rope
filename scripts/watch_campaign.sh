#!/usr/bin/env bash
set -euo pipefail

# Lightweight watchdog for plan-v2 campaign execution.
# It does NOT modify experiments; it only reports health/progress.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INTERVAL_SEC="${INTERVAL_SEC:-60}"
MAX_IDLE_SEC="${MAX_IDLE_SEC:-900}"
MODEL_TAG="${MODEL_TAG:-meta_llama_3_8b_instruct}"
CTX_TAG="${CTX_TAG:-32k}"
SEED="${SEED:-1337}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
CAMPAIGN_PID="${CAMPAIGN_PID:-}"
LAUNCH_LOG="${LAUNCH_LOG:-}"
OUT_LOG="${OUT_LOG:-artifacts/logs/watch_campaign_${DATE_TAG}_${CTX_TAG}_seed${SEED}.log}"

mkdir -p "$(dirname "$OUT_LOG")"

declare -a RUNS=(
  "e2_${MODEL_TAG}_${CTX_TAG}_yarn_${SEED}"
  "e2_${MODEL_TAG}_${CTX_TAG}_hybrid_${SEED}"
  "e1_${MODEL_TAG}_${CTX_TAG}_baseline_native_${SEED}"
  "e1_${MODEL_TAG}_${CTX_TAG}_pi_${SEED}"
  "e1_${MODEL_TAG}_${CTX_TAG}_yarn_${SEED}"
  "e1_${MODEL_TAG}_${CTX_TAG}_hybrid_${SEED}"
)

run_path() {
  local short="$1"
  echo "runs/${DATE_TAG}_${short}"
}

run_status() {
  local short="$1"
  local p
  p="$(run_path "$short")"

  if [[ -f "${p}/summary.json" ]]; then
    local status
    status="$(grep -m1 '"status"' "${p}/summary.json" 2>/dev/null | sed -E 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' || true)"
    [[ -n "$status" ]] && echo "done:${status}" || echo "done:unknown"
    return
  fi

  if [[ -f "${p}/stdout.log" || -f "${p}/metrics.jsonl" ]]; then
    echo "running"
    return
  fi

  echo "pending"
}

last_activity_epoch() {
  local newest=0
  for short in "${RUNS[@]}"; do
    local p
    p="$(run_path "$short")"
    for f in "${p}/stdout.log" "${p}/eval_longbench.log" "${p}/metrics.jsonl"; do
      if [[ -f "$f" ]]; then
        local t
        t="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
        if (( t > newest )); then
          newest="$t"
        fi
      fi
    done
  done
  echo "$newest"
}

grep_recent_errors() {
  local short="$1"
  local p
  p="$(run_path "$short")"
  local f="${p}/stdout.log"
  if [[ ! -f "$f" ]]; then
    return
  fi
  grep -nE "CUDA out of memory|Traceback|RuntimeError|ERROR" "$f" | tail -n 2 || true
}

all_done() {
  for short in "${RUNS[@]}"; do
    local p
    p="$(run_path "$short")"
    [[ -f "${p}/summary.json" ]] || return 1
  done
  [[ -f artifacts/results/prior_fit.json ]] || return 1
  [[ -f artifacts/tables/table1_main.csv ]] || return 1
  return 0
}

{
  echo "===== watchdog start $(date '+%F %T') ====="
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "INTERVAL_SEC=${INTERVAL_SEC}"
  echo "MAX_IDLE_SEC=${MAX_IDLE_SEC}"
  echo "CAMPAIGN_PID=${CAMPAIGN_PID:-<none>}"
  echo "LAUNCH_LOG=${LAUNCH_LOG:-<none>}"
} >> "$OUT_LOG"

while true; do
  now_epoch="$(date +%s)"
  now_human="$(date '+%F %T')"

  {
    echo
    echo "----- ${now_human} -----"
    if [[ -n "${CAMPAIGN_PID}" ]]; then
      if ps -p "${CAMPAIGN_PID}" >/dev/null 2>&1; then
        echo "campaign_pid=${CAMPAIGN_PID} alive"
      else
        echo "campaign_pid=${CAMPAIGN_PID} dead"
      fi
    fi

    echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null | tr '\n' '; ' || true)"
    echo "gpu_procs:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

    echo "run_status:"
    for short in "${RUNS[@]}"; do
      printf "  %s -> %s\n" "$short" "$(run_status "$short")"
    done

    echo "recent_errors:"
    for short in "${RUNS[@]}"; do
      err="$(grep_recent_errors "$short")"
      if [[ -n "${err}" ]]; then
        echo "  ${short}"
        echo "${err}" | sed 's/^/    /'
      fi
    done

    if [[ -n "${LAUNCH_LOG}" && -f "${LAUNCH_LOG}" ]]; then
      echo "launch_tail:"
      tail -n 8 "${LAUNCH_LOG}" | sed 's/^/  /'
    fi
  } >> "$OUT_LOG"

  last_act="$(last_activity_epoch)"
  if [[ "$last_act" != "0" ]]; then
    idle_sec=$(( now_epoch - last_act ))
    if (( idle_sec > MAX_IDLE_SEC )); then
      {
        echo "[WARN] idle_for=${idle_sec}s exceeds MAX_IDLE_SEC=${MAX_IDLE_SEC}"
      } >> "$OUT_LOG"
    fi
  fi

  if all_done; then
    {
      echo "[DONE] all expected outputs detected at ${now_human}"
      echo "===== watchdog end $(date '+%F %T') ====="
    } >> "$OUT_LOG"
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
