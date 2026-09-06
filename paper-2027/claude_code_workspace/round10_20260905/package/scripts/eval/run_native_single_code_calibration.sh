#!/usr/bin/env bash
# One pre-registered Native instrument repair. Does not open confirmation.
set -euo pipefail
: "${P0_CHECKPOINT:?exact local checkpoint required}"
: "${P0_PARENT_DATA:?frozen original P0 input directory required}"
: "${P0_OUTPUT:?fresh output root required}"
: "${P0_WEIGHT_SHA:?frozen weight identity required}"
: "${P0_NATIVE_SHA:?runtime Native frequency identity required}"
P0_PYTHON="${P0_PYTHON:-python3}"
P0_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$P0_REPO"

"$P0_PYTHON" scripts/data/prepare_native_single_code_calibration.py \
  --checkpoint "$P0_CHECKPOINT" --parent-data-root "$P0_PARENT_DATA" \
  --output "$P0_OUTPUT/data"
p0_manifest_receipt="$(sha256sum "$P0_OUTPUT/data/manifest.json")"
"$P0_PYTHON" -u scripts/eval/eval_native_reference_calibration.py \
  --checkpoint "$P0_CHECKPOINT" --expected-weight-sha256 "$P0_WEIGHT_SHA" \
  --expected-native-sha256 "$P0_NATIVE_SHA" \
  --data-root "$P0_OUTPUT/data" \
  --expected-data-manifest-sha256 "${p0_manifest_receipt%% *}" \
  --split calibration --batch-tokens 16384 --max-batch-size 8 \
  --output "$P0_OUTPUT/calibration"

if "$P0_PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "P0_NATIVE_REFERENCE_EVAL_COMPLETE")' \
    "$P0_OUTPUT/calibration/results.json"; then
  "$P0_PYTHON" scripts/analysis/summarize_native_reference_calibration.py \
    --examples "$P0_OUTPUT/calibration/examples.jsonl" \
    --data-manifest "$P0_OUTPUT/data/manifest.json" \
    --output "$P0_OUTPUT/calibration_decision.json" --phase calibration
fi
