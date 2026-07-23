#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
  local candidate="$SCRIPT_DIR"
  while [[ "$candidate" != "/" ]]; do
    if [[ -f "$candidate/AGENTS.md" && -f "$candidate/scripts/lib/rope/schedules.py" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    candidate="$(dirname "$candidate")"
  done
  return 1
}
REPO_ROOT="$(find_repo_root)" || {
  echo "Repository root not found from $SCRIPT_DIR" >&2
  exit 2
}
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/.local/reviewer27be_shape_base}"
DATA_MANIFEST="${DATA_MANIFEST:-${WORK_DIR}/data/data_manifest.json}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python interpreter not found" >&2
    exit 2
  fi
fi
RUNNER="rebuttal.rebuttal_0723.reviewer27be_shape_base.run_experiment"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORK_DIR}/torchinductor_cache}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${WORK_DIR}/logs" "${TORCHINDUCTOR_CACHE_DIR}"

exec 9>"${WORK_DIR}/reviewer27be.lock"
if ! flock -n 9; then
  echo "Another reviewer27be launcher owns ${WORK_DIR}/reviewer27be.lock" >&2
  exit 1
fi

run_logged() {
  local label="$1"
  shift
  "$@" 2>&1 | tee "${WORK_DIR}/logs/${label}.log"
}

preflight() {
  run_logged preflight_cpu \
    "${PYTHON_BIN}" -m "${RUNNER}" preflight \
    --data_manifest "${DATA_MANIFEST}" \
    --full_hash_check
}

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L; then
    echo "GPU mode is not active; refusing to start paid-run commands." >&2
    exit 2
  fi
  "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see CUDA"
assert torch.cuda.is_bf16_supported(), "GPU does not support BF16"
p = torch.cuda.get_device_properties(0)
memory = getattr(p, "total_memory", getattr(p, "total_mem", 0))
assert memory >= 30 * 2**30, f"need >=30 GiB, found {memory / 2**30:.1f}"
print({"gpu": p.name, "memory_gib": round(memory / 2**30, 1), "torch": str(torch.__version__)})
PY
}

run_one() {
  local suite="$1"
  local arm="$2"
  local seed="$3"
  local compile_mode="${R27_COMPILE_MODE:-default}"
  local run_dir="${WORK_DIR}/runs/${suite}/${arm}/seed${seed}"
  if [[ ! -s "${run_dir}/train_result.json" ]]; then
    run_logged "train_${suite}_${arm}_seed${seed}" \
      "${PYTHON_BIN}" -m "${RUNNER}" train \
      --suite "${suite}" --arm "${arm}" --seed "${seed}" \
      --data_manifest "${DATA_MANIFEST}" --work_dir "${WORK_DIR}" \
      --compile_mode "${compile_mode}" --num_workers 8
  fi
  if [[ ! -s "${run_dir}/eval_test.json" ]]; then
    run_logged "eval_test_${suite}_${arm}_seed${seed}" \
      "${PYTHON_BIN}" -m "${RUNNER}" evaluate \
      --suite "${suite}" --arm "${arm}" --seed "${seed}" --split test \
      --data_manifest "${DATA_MANIFEST}" --work_dir "${WORK_DIR}"
  fi
}

probe_suite() {
  local suite="$1"
  if [[ ! -s "${WORK_DIR}/gpu_probe_${suite}.json" ]]; then
    run_logged "gpu_probe_${suite}" \
      "${PYTHON_BIN}" -m "${RUNNER}" probe-gpu \
      --suite "${suite}" --data_manifest "${DATA_MANIFEST}" \
      --work_dir "${WORK_DIR}"
  fi
}

run_shape() {
  local seed42=(
    std_geo paper_geo evq_tau1 evq_tau2 evq_tau3 evq_tau4
    evq_tau5 evq_rule evq_tau6 evq_tau7 uniform_span_matched
    power_matched exp_matched
  )
  local core=(paper_geo uniform_span_matched evq_rule power_matched exp_matched)
  local arm
  probe_suite shape_l128
  for arm in "${seed42[@]}"; do
    run_one shape_l128 "${arm}" 42
    case "${arm}" in
      paper_geo|evq_tau1|evq_tau2|evq_tau3|evq_tau4|evq_tau5|evq_rule|evq_tau6|evq_tau7)
        local selection="${WORK_DIR}/runs/shape_l128/${arm}/seed42/eval_selection.json"
        if [[ ! -s "${selection}" ]]; then
          run_logged "eval_selection_shape_l128_${arm}_seed42" \
            "${PYTHON_BIN}" -m "${RUNNER}" evaluate \
            --suite shape_l128 --arm "${arm}" --seed 42 --split selection \
            --data_manifest "${DATA_MANIFEST}" --work_dir "${WORK_DIR}"
        fi
        ;;
    esac
  done
  if [[ ! -s "${WORK_DIR}/tau_selection.json" ]]; then
    run_logged select_tau \
      "${PYTHON_BIN}" -m "${RUNNER}" select-tau --work_dir "${WORK_DIR}"
  fi
  local seed
  for seed in 137 256; do
    for arm in "${core[@]}"; do
      run_one shape_l128 "${arm}" "${seed}"
    done
  done
  run_logged summarize_shape \
    "${PYTHON_BIN}" -m "${RUNNER}" summarize \
    --suite shape_l128 --work_dir "${WORK_DIR}"
}

run_heldout() {
  local arm seed
  probe_suite heldout_b1m_d128
  for seed in 42 137 256; do
    for arm in paper_geo evq_rule; do
      run_one heldout_b1m_d128 "${arm}" "${seed}"
    done
  done
  run_logged summarize_heldout \
    "${PYTHON_BIN}" -m "${RUNNER}" summarize \
    --suite heldout_b1m_d128 --work_dir "${WORK_DIR}"
}

cd "${REPO_ROOT}"
case "${MODE}" in
  preflight)
    preflight
    ;;
  shape)
    require_gpu
    preflight
    run_shape
    ;;
  heldout)
    require_gpu
    preflight
    run_heldout
    ;;
  all)
    require_gpu
    preflight
    run_shape
    run_heldout
    ;;
  *)
    echo "usage: $0 {preflight|shape|heldout|all}" >&2
    exit 64
    ;;
esac
