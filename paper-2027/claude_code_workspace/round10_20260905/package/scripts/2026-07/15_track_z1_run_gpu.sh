#!/usr/bin/env bash
set -euo pipefail

: "${READY_FILE:?set READY_FILE to the no-GPU READY receipt}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"

shutdown_now() {
  sync
  if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
    shutdown -h now || poweroff || true
  fi
}

failed() {
  status=$?
  trap - EXIT
  echo "Track Z GPU run failed; shutting down. Re-enter no-GPU mode before fixing anything." >&2
  shutdown_now
  exit "$status"
}
trap failed EXIT

if [[ ! -r "$READY_FILE" ]]; then
  echo "missing no-GPU READY receipt: $READY_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$READY_FILE"
if [[ "${TRACK_Z_READY_VERSION:-}" != "1" ]]; then
  echo "incompatible no-GPU READY receipt" >&2
  exit 1
fi

entry="$REPO_DIR/experiments/lora_evq_v2/eval_sparse_conversion.py"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting Track Z association swap: Geo (1/2)"
"$PYTHON_BIN" -u "$entry" association-swap-trace \
  --model-name "$MODEL_NAME" \
  --model-manifest "$MODEL_MANIFEST" \
  --training-data-manifest "$TRAINING_MANIFEST" \
  --adapter-dir "$GEO_ADAPTER" \
  --substrate native_geo \
  --cases-root "$CASES_ROOT" \
  --output "$SWAP_GEO_OUTPUT" \
  2>&1 | tee "$(dirname "$READY_FILE")/logs/swap_native_geo.log"

echo "Starting Track Z association swap: EVQ (2/2)"
"$PYTHON_BIN" -u "$entry" association-swap-trace \
  --model-name "$MODEL_NAME" \
  --model-manifest "$MODEL_MANIFEST" \
  --training-data-manifest "$TRAINING_MANIFEST" \
  --adapter-dir "$EVQ_ADAPTER" \
  --substrate evq_cosh \
  --cases-root "$CASES_ROOT" \
  --output "$SWAP_EVQ_OUTPUT" \
  2>&1 | tee "$(dirname "$READY_FILE")/logs/swap_evq_cosh.log"

printf 'complete\n' > "$GPU_COMPLETE_FILE.incomplete"
mv "$GPU_COMPLETE_FILE.incomplete" "$GPU_COMPLETE_FILE"
trap - EXIT
echo "Track Z GPU collection complete; shutting down before analysis."
shutdown_now
