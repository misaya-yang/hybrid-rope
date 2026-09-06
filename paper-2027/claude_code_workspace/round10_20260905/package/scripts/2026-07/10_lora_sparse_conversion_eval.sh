#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${REPO_DIR:?set REPO_DIR}"
: "${MODEL_NAME:?set MODEL_NAME}"
: "${MODEL_MANIFEST:?set MODEL_MANIFEST}"
: "${TRAINING_MANIFEST:?set TRAINING_MANIFEST}"
: "${GEO_ADAPTER:?set GEO_ADAPTER}"
: "${EVQ_ADAPTER:?set EVQ_ADAPTER}"
: "${PASSKEY_ROOT:?set PASSKEY_ROOT}"
: "${SUITE_ROOT:?set SUITE_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"

stage=${1:?usage: $0 phase0|phase1-pilot|phase1-full|phase1-suite}
entry="$REPO_DIR/experiments/lora_evq_v2/eval_sparse_conversion.py"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RESULT_ROOT"

common=(
  --model-name "$MODEL_NAME"
  --model-manifest "$MODEL_MANIFEST"
  --training-data-manifest "$TRAINING_MANIFEST"
)

run_phase1() {
  local selection=$1
  local dataset=$2
  local data_root=$3
  local prefix=$4
  "$PYTHON_BIN" -u "$entry" phase1 "${common[@]}" \
    --adapter-dir "$GEO_ADAPTER" --substrate native_geo \
    --phase0-gate "$RESULT_ROOT/phase0_summary.json" \
    --dataset "$dataset" --data-root "$data_root" --selection "$selection" \
    --output "$RESULT_ROOT/${prefix}_geo.json"
  "$PYTHON_BIN" -u "$entry" phase1 "${common[@]}" \
    --adapter-dir "$EVQ_ADAPTER" --substrate evq_cosh \
    --phase0-gate "$RESULT_ROOT/phase0_summary.json" \
    --dataset "$dataset" --data-root "$data_root" --selection "$selection" \
    --output "$RESULT_ROOT/${prefix}_evq.json"
  "$PYTHON_BIN" -u "$entry" summarize-phase1 \
    --geo "$RESULT_ROOT/${prefix}_geo.json" \
    --evq "$RESULT_ROOT/${prefix}_evq.json" \
    --output "$RESULT_ROOT/${prefix}_summary.json"
}

case "$stage" in
  phase0)
    "$PYTHON_BIN" -u "$entry" phase0 "${common[@]}" \
      --adapter-dir "$GEO_ADAPTER" --substrate native_geo \
      --passkey-root "$PASSKEY_ROOT" --output "$RESULT_ROOT/phase0_geo.json"
    "$PYTHON_BIN" -u "$entry" phase0 "${common[@]}" \
      --adapter-dir "$EVQ_ADAPTER" --substrate evq_cosh \
      --passkey-root "$PASSKEY_ROOT" --output "$RESULT_ROOT/phase0_evq.json"
    "$PYTHON_BIN" -u "$entry" summarize-phase0 \
      --geo "$RESULT_ROOT/phase0_geo.json" \
      --evq "$RESULT_ROOT/phase0_evq.json" \
      --output "$RESULT_ROOT/phase0_summary.json"
    ;;
  phase1-pilot)
    run_phase1 pilot passkey "$PASSKEY_ROOT" phase1_passkey_pilot
    ;;
  phase1-full)
    run_phase1 full passkey "$PASSKEY_ROOT" phase1_passkey_full
    ;;
  phase1-suite)
    run_phase1 pilot retrieval-suite "$SUITE_ROOT" phase1_retrieval_suite
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
