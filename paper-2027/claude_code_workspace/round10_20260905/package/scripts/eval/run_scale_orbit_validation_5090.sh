#!/usr/bin/env bash
# Frozen driver for the scale-orbit validation panel.  GPU modes require an
# explicit environment guard and are intentionally resumable by the evaluators.
set -euo pipefail

MODE="${1:-}"
ROOT="${EVQ_REPO_ROOT:?set EVQ_REPO_ROOT}"
PYTHON="${EVQ_PYTHON:-/root/miniconda3/bin/python}"
WORK="${EVQ_SCALE_ORBIT_ROOT:?set EVQ_SCALE_ORBIT_ROOT}"
ASSETS="${EVQ_SCALE_ORBIT_ASSETS:?set EVQ_SCALE_ORBIT_ASSETS}"
CHECKPOINT="${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}"
TOKEN_MANIFEST="${EVQ_TOKEN_MANIFEST:?set EVQ_TOKEN_MANIFEST}"
RULER_DATA="${EVQ_RULER_DATA:?set EVQ_RULER_DATA}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

require_gpu() {
  if [[ "${SCALE_ORBIT_GPU_AUTHORIZED:-}" != "YES" ]]; then
    echo "set SCALE_ORBIT_GPU_AUTHORIZED=YES after explicit GPU authorization" >&2
    exit 3
  fi
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(torch.cuda.get_device_name(0))
PY
}

table_value() {
  local name="$1" key="$2"
  "$PYTHON" - "$ASSETS/manifest.json" "$name" "$key" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1]))
print(manifest["tables"][sys.argv[2]][sys.argv[3]])
PY
}

run_external_formal() {
  local name="$1" limit="$2" output="$3"; shift 3
  "$PYTHON" "$ROOT/scripts/eval/target_free_formal_eval.py" \
    --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
    --method external_table_static --tasks pg19 \
    --multipliers "$@" --limit-per-cell "$limit" \
    --table "$ASSETS/$(table_value "$name" path)" --table-name "$name" \
    --table-support "$(table_value "$name" table_support)" --factor 4 \
    --long-attention-scaling "$(table_value "$name" attention_scaling)" \
    --expected-active-sha256 "$(table_value "$name" float32_sha256)" \
    --output "$output"
}

run_external_ruler() {
  local name="$1" limit="$2" output="$3"; shift 3
  "$PYTHON" "$ROOT/scripts/eval/target_free_ruler_smoke.py" \
    --checkpoint "$CHECKPOINT" --data-root "$RULER_DATA" \
    --method external_table_static --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --lengths "$@" --limit-per-cell "$limit" --native-context-length 4096 \
    --table "$ASSETS/$(table_value "$name" path)" --table-name "$name" \
    --table-support "$(table_value "$name" table_support)" --table-factor 4 \
    --long-attention-scaling "$(table_value "$name" attention_scaling)" \
    --expected-active-sha256 "$(table_value "$name" float32_sha256)" \
    --expected-native-sha256 "$(table_value native float32_sha256)" \
    --output "$output"
}

case "$MODE" in
  preflight)
    for name in same_support_geometric_s4 legacy_u_p2_log_s4 min_q_exact_chain_s4 min_q_ulp_jitter_s4; do
      run_external_formal "$name" 1 "$WORK/preflight/$name" 1 4 --preflight-only
    done
    ;;
  gpu-smoke)
    require_gpu
    for name in min_q_exact_chain_s4 min_q_ulp_jitter_s4; do
      run_external_formal "$name" 1 "$WORK/smoke/$name" 1
    done
    ;;
  gpu-primary-formal)
    require_gpu
    for name in min_q_exact_chain_s4 min_q_ulp_jitter_s4; do
      run_external_formal "$name" 20 "$WORK/formal/$name" 1 4
    done
    ;;
  gpu-primary-ruler)
    require_gpu
    for name in min_q_exact_chain_s4 min_q_ulp_jitter_s4; do
      run_external_ruler "$name" 20 "$WORK/ruler/$name" 4096 16384
    done
    ;;
  gpu-breadth-formal)
    require_gpu
    for name in same_support_geometric_s4 legacy_u_p2_log_s4; do
      run_external_formal "$name" 20 "$WORK/formal/$name" 1 4
    done
    ;;
  gpu-breadth-ruler)
    require_gpu
    for name in same_support_geometric_s4 legacy_u_p2_log_s4; do
      run_external_ruler "$name" 20 "$WORK/ruler/$name" 4096 16384
    done
    ;;
  summarize)
    "$PYTHON" "$ROOT/scripts/analysis/scale_orbit_validation.py" summarize \
      --manifest "$ASSETS/manifest.json" --results-root "$WORK" \
      --output "$WORK/summary.json"
    ;;
  *)
    echo "usage: $0 {preflight|gpu-smoke|gpu-primary-formal|gpu-primary-ruler|gpu-breadth-formal|gpu-breadth-ruler|summarize}" >&2
    exit 2
    ;;
esac
