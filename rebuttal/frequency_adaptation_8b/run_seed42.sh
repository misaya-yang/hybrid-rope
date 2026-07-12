#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${EVQ_FREQ_MODEL:-}"
MODEL_MANIFEST="${EVQ_FREQ_MODEL_MANIFEST:-}"
FILLER_DIR="${EVQ_FREQ_FILLER_DIR:-}"
WORK_DIR="${EVQ_FREQ_WORK_DIR:-}"
LORA_R="${EVQ_FREQ_LORA_R:-64}"

usage() {
  cat <<'EOF'
Required environment:
  EVQ_FREQ_MODEL        local LLaMA-3-8B-Instruct directory
  EVQ_FREQ_MODEL_MANIFEST precomputed full-byte model manifest (create before GPU rental)
  EVQ_FREQ_FILLER_DIR   frozen plain-text folder with train.pt, validation.pt, manifest.json
  EVQ_FREQ_WORK_DIR     external work directory for data/checkpoints/evaluations

Optional environment:
  PYTHON_BIN, EVQ_FREQ_LORA_R (64; alpha is fixed to 2r)

Before renting a GPU, create the model manifest with:
  python experiments/lora_evq_v2/prepare_legacy_model_manifest.py --model_dir "$EVQ_FREQ_MODEL" --output MODEL_MANIFEST.json

Commands (training auto-evaluates that checkpoint; no command advances to the next gate):
  run_seed42.sh prepare
  run_seed42.sh preflight
  run_seed42.sh warmup
  run_seed42.sh train transition geo|evq
  run_seed42.sh train exact_8k geo|evq
  run_seed42.sh train exact_16k geo|evq
  run_seed42.sh eval-base warmup|transition|exact_8k|exact_16k
  run_seed42.sh eval warmup geo
  run_seed42.sh eval transition|exact_8k|exact_16k geo|evq
EOF
}

require_common() {
  : "${MODEL:?Set EVQ_FREQ_MODEL}"
  : "${MODEL_MANIFEST:?Set EVQ_FREQ_MODEL_MANIFEST}"
  : "${WORK_DIR:?Set EVQ_FREQ_WORK_DIR outside the repository}"
}

require_prepared_data() {
  for phase in warmup transition exact_8k exact_16k; do
    [[ -f "${WORK_DIR}/data/train_${phase}.pt" ]] || {
      echo "Missing prepared train bundle for ${phase}; run 'run_seed42.sh prepare' first" >&2
      return 2
    }
    [[ -f "${WORK_DIR}/data/eval_${phase}.pt" ]] || {
      echo "Missing prepared eval bundle for ${phase}; run 'run_seed42.sh prepare' first" >&2
      return 2
    }
  done
  [[ -f "${WORK_DIR}/data/manifest.json" ]] || {
    echo "Missing prepared data manifest; run 'run_seed42.sh prepare' first" >&2
    return 2
  }
}

cpu_preflight() {
  require_common
  require_prepared_data
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/curriculum.py" \
    "${SCRIPT_DIR}/prepare_data.py" \
    "${SCRIPT_DIR}/train.py" \
    "${SCRIPT_DIR}/evaluate.py"
  "${PYTHON_BIN}" -m pytest "${REPO_ROOT}/tests/test_frequency_adaptation_8b.py" -q
  "${PYTHON_BIN}" -m rebuttal.frequency_adaptation_8b.train \
    --model-name "${MODEL}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --phase warmup \
    --arm geo \
    --data "${WORK_DIR}/data/train_warmup.pt" \
    --output-dir "${WORK_DIR}/preflight/dryrun_warmup_geo" \
    --seed 42 \
    --lora-r "${LORA_R}" \
    --dry-run
}

checkpoint_path() {
  local phase="$1"
  local arm="$2"
  if [[ "${phase}" == "warmup" ]]; then
    printf '%s/checkpoints/warmup_geo' "${WORK_DIR}"
  else
    printf '%s/checkpoints/%s_%s' "${WORK_DIR}" "${phase}" "${arm}"
  fi
}

prior_checkpoint() {
  local phase="$1"
  local arm="$2"
  case "${phase}" in
    transition) checkpoint_path warmup geo ;;
    exact_8k) checkpoint_path transition "${arm}" ;;
    exact_16k) checkpoint_path exact_8k "${arm}" ;;
    *) echo "No prior checkpoint mapping for phase ${phase}" >&2; return 2 ;;
  esac
}

train_phase() {
  local phase="$1"
  local arm="$2"
  local output
  output="$(checkpoint_path "${phase}" "${arm}")"
  local command=(
    "${PYTHON_BIN}" -m rebuttal.frequency_adaptation_8b.train
    --model-name "${MODEL}"
    --model-manifest "${MODEL_MANIFEST}"
    --phase "${phase}"
    --arm "${arm}"
    --data "${WORK_DIR}/data/train_${phase}.pt"
    --output-dir "${output}"
    --seed 42
    --lora-r "${LORA_R}"
  )
  if [[ "${phase}" != "warmup" ]]; then
    command+=(--adapter-from "$(prior_checkpoint "${phase}" "${arm}")")
  fi
  "${command[@]}" --dry-run
  "${command[@]}"
  eval_checkpoint "${phase}" "${arm}"
}

eval_checkpoint() {
  local phase="$1"
  local arm="$2"
  local checkpoint
  checkpoint="$(checkpoint_path "${phase}" "${arm}")"
  "${PYTHON_BIN}" -m rebuttal.frequency_adaptation_8b.evaluate \
    --model-name "${MODEL}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --adapter-dir "${checkpoint}" \
    --phase "${phase}" \
    --arm "${arm}" \
    --data "${WORK_DIR}/data/eval_${phase}.pt" \
    --output "${WORK_DIR}/evaluations/${phase}_${arm}.json" \
    --seed 42
}

eval_base_checkpoint() {
  local phase="$1"
  "${PYTHON_BIN}" -m rebuttal.frequency_adaptation_8b.evaluate \
    --model-name "${MODEL}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --base-only \
    --phase "${phase}" \
    --arm geo \
    --data "${WORK_DIR}/data/eval_${phase}.pt" \
    --output "${WORK_DIR}/evaluations/${phase}_base_geo.json" \
    --seed 42
}

command="${1:-help}"
case "${command}" in
  prepare)
    require_common
    : "${FILLER_DIR:?Set EVQ_FREQ_FILLER_DIR}"
    "${PYTHON_BIN}" -m rebuttal.frequency_adaptation_8b.prepare_data \
      --tokenizer "${MODEL}" \
      --filler-dir "${FILLER_DIR}" \
      --output-dir "${WORK_DIR}/data" \
      --seed 42 \
      --value-tokens 12 \
      --eval-groups 64
    ;;
  preflight)
    cpu_preflight
    ;;
  warmup)
    require_common
    train_phase warmup geo
    ;;
  train)
    require_common
    phase="${2:-}"
    arm="${3:-}"
    case "${phase}" in transition|exact_8k|exact_16k) ;; *) usage; exit 2 ;; esac
    case "${arm}" in geo|evq) ;; *) usage; exit 2 ;; esac
    train_phase "${phase}" "${arm}"
    ;;
  eval-base)
    require_common
    phase="${2:-}"
    case "${phase}" in warmup|transition|exact_8k|exact_16k) ;; *) usage; exit 2 ;; esac
    eval_base_checkpoint "${phase}"
    ;;
  eval)
    require_common
    phase="${2:-}"
    arm="${3:-}"
    case "${phase}" in warmup|transition|exact_8k|exact_16k) ;; *) usage; exit 2 ;; esac
    case "${arm}" in geo|evq) ;; *) usage; exit 2 ;; esac
    if [[ "${phase}" == "warmup" && "${arm}" != "geo" ]]; then
      echo "warmup has only the shared Geo arm" >&2
      exit 2
    fi
    eval_checkpoint "${phase}" "${arm}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
