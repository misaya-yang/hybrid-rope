#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_instruct_pro6000_v1}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"
MODEL="${MODEL:-$ASSET_ROOT/models/OLMo-2-0425-1B-Instruct}"
RULER_ROOT="${RULER_ROOT:-$ASSET_ROOT/sources/RULER}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT/data/ruler_fresh_s20260726_n128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ASSET_ROOT/runs/ruler_fresh_s20260726_n128}"
READY="${READY:-$ASSET_ROOT/receipts/pro6000_ruler_matrix_ready.json}"
PYTHON="${PYTHON:-/root/olmo2_venv/bin/python}"
SEED="${SEED:-20260726}"
ORIGINAL_FORMAL_PID="${ORIGINAL_FORMAL_PID:-3180}"
ORIGINAL_EVAL_PID="${ORIGINAL_EVAL_PID:-33663}"

NATIVE_STAGE_A="$ASSET_ROOT/runs/instruct_native_stage_a_4k_20m_s20260725/adapter.pt"
EVQ_STAGE_A="$ASSET_ROOT/runs/instruct_evq_stage_a_4k_20m_s20260725/adapter.pt"
NATIVE_FINAL="$ASSET_ROOT/runs/instruct_native_counterfactual_routing_4k_300_s20260725/adapter.pt"
EVQ_FINAL="$ASSET_ROOT/runs/instruct_evq_counterfactual_routing_4k_300_s20260725/adapter.pt"
MODULE="rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity"

export PYTHONPATH="$ASSET_ROOT/python_deps:$CODE_ROOT"
export NLTK_DATA="$ASSET_ROOT/nltk_data"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export TOKENIZERS_PARALLELISM=false

prepare() {
  if [[ ! -f "$RULER_ROOT/scripts/data/synthetic/json/PaulGrahamEssays.json" ]]; then
    (
      cd "$RULER_ROOT/scripts/data/synthetic/json"
      CUDA_VISIBLE_DEVICES="" ionice -c3 nice -n19 \
        "$PYTHON" download_paulgraham_essay.py
    )
  fi
  mkdir -p "$DATA_ROOT"
  for task in niah_single_1 niah_single_2; do
    output="$DATA_ROOT/$task"
    if [[ ! -e "$output" ]]; then
      CUDA_VISIBLE_DEVICES="" ionice -c3 nice -n19 "$PYTHON" \
        -m "$MODULE.prepare_instruct_ruler_screen" \
        --ruler-root "$RULER_ROOT" \
        --checkpoint "$MODEL" \
        --output "$output" \
        --task "$task" \
        --samples-per-length 128 \
        --seed "$SEED"
    fi
  done
}

preflight() {
  if [[ -e "$READY" ]]; then
    echo "READY already exists: $READY" >&2
    return 20
  fi
  mkdir -p "$(dirname "$READY")" "$OUTPUT_ROOT"
  CUDA_VISIBLE_DEVICES="" ionice -c3 nice -n19 "$PYTHON" \
    -m "$MODULE.preflight_instruct_ruler_matrix" \
    --checkpoint "$MODEL" \
    --code-root "$CODE_ROOT" \
    --ruler-root "$RULER_ROOT" \
    --data-single-1 "$DATA_ROOT/niah_single_1" \
    --data-single-2 "$DATA_ROOT/niah_single_2" \
    --native-stage-a "$NATIVE_STAGE_A" \
    --evq-stage-a "$EVQ_STAGE_A" \
    --native-final "$NATIVE_FINAL" \
    --evq-final "$EVQ_FINAL" \
    --output-root "$OUTPUT_ROOT" \
    --receipt "$READY" \
    --seed "$SEED"
}

require_idle_gpu() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | tr -d '[:space:]')"
  if [[ -n "$pids" ]]; then
    echo "STOP: GPU still has compute processes: $pids" >&2
    return 20
  fi
}

evaluate_cell() {
  local task="$1"
  local arm="$2"
  shift 2
  local frequency adapter
  case "$arm" in
    native)
      frequency="native"
      adapter="$NATIVE_FINAL"
      ;;
    evq)
      frequency="evq"
      adapter="$EVQ_FINAL"
      ;;
    *)
      echo "unknown arm: $arm" >&2
      return 20
      ;;
  esac
  output="$OUTPUT_ROOT/${task}_${arm}"
  "$PYTHON" -m "$MODULE.evaluate_instruct_ruler_screen" \
    --checkpoint "$MODEL" \
    --ready-receipt "$READY" \
    --data-root "$DATA_ROOT/$task" \
    --output "$output" \
    --task "$task" \
    --frequency "$frequency" \
    --adapter "$adapter" \
    --rank 64 \
    --alpha 128 \
    --limit-per-length 128 \
    "$@"
}

strict_score() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["results"]["cells"][sys.argv[2]]["first_number_exact"])
PY
}

run_matrix() {
  test -f "$READY"
  require_idle_gpu
  export CUDA_VISIBLE_DEVICES=0

  evaluate_cell niah_single_1 native --lengths 4096 8192 16384
  evaluate_cell niah_single_1 evq --lengths 4096 8192 16384

  evaluate_cell niah_single_2 native --lengths 4096
  evaluate_cell niah_single_2 evq --lengths 4096
  native_4k="$(strict_score \
    "$OUTPUT_ROOT/niah_single_2_native/results.json" 4096)"
  evq_4k="$(strict_score \
    "$OUTPUT_ROOT/niah_single_2_evq/results.json" 4096)"
  "$PYTHON" - "$native_4k" "$evq_4k" <<'PY'
import sys
native, evq = map(float, sys.argv[1:])
if native < 0.25 and evq < 0.25:
    raise SystemExit(
        "STOP: both single_2 arms are below the 4K competence gate"
    )
PY
  evaluate_cell niah_single_2 native --lengths 4096 8192 16384
  evaluate_cell niah_single_2 evq --lengths 4096 8192 16384
}

wait_and_run() {
  while kill -0 "$ORIGINAL_FORMAL_PID" 2>/dev/null; do sleep 20; done
  while kill -0 "$ORIGINAL_EVAL_PID" 2>/dev/null; do sleep 20; done
  test -f /root/autodl-tmp/olmo2_1b_evq_run/evq_1000/step-001000-full/trainer_state.json
  test -f /root/autodl-tmp/olmo2_1b_evq_run/evaluation/evq1000/results.json
  test -f /root/autodl-tmp/olmo2_1b_evq_run/evaluation/retrieval_evq1000/results.json
  require_idle_gpu
  run_matrix
}

case "$MODE" in
  prepare) prepare ;;
  preflight) preflight ;;
  run) run_matrix ;;
  wait-and-run) wait_and_run ;;
  *)
    echo "usage: $0 {prepare|preflight|run|wait-and-run}" >&2
    exit 2
    ;;
esac
