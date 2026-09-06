#!/usr/bin/env bash
# P1 fixed controls and NLL; no profile selection or new parameters.
set -euo pipefail
: "${P1_CHECKPOINT:?}"
: "${P1_ROOT:?}"
: "${P1_OLD_TABLE:?}"
: "${P1_WEIGHT_SHA:?}"
: "${P1_DATA_SHA:?}"
P1_PYTHON="${P1_PYTHON:-python3}"
P1_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$P1_REPO"
read -r p1_yarn_hash p1_yarn_gain p1_native_hash < <(
  "$P1_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); t=m["tables"]["official_equation_yarn"]; print(t["tensor_sha256"],t["attention_scaling"],m["checkpoint"]["native_sha256_float32"])' \
    "$P1_ROOT/baselines_s4/manifest.json"
)
P1_COMMON=(--checkpoint "$P1_CHECKPOINT" --data-root "$P1_ROOT/ruler_holdout"
  --method external_table_static --table-support native_div_factor
  --expected-weight-sha256 "$P1_WEIGHT_SHA" --expected-native-sha256 "$p1_native_hash"
  --expected-data-manifest-sha256 "$P1_DATA_SHA"
  --native-context-length 8192 --profile-target-length 16384
  --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt --limit-per-cell 20)

"$P1_PYTHON" -u scripts/eval/target_free_ruler_smoke.py "${P1_COMMON[@]}" \
  --table "$P1_ROOT/baselines_s4/official_equation_yarn_s4.npy" \
  --table-name reference_corrected_official_yarn_s4 --table-factor 4 \
  --expected-active-sha256 "$p1_yarn_hash" --long-attention-scaling "$p1_yarn_gain" \
  --lengths 4096 8192 16384 --output "$P1_ROOT/ruler_s4/official_yarn"

"$P1_PYTHON" -u scripts/eval/target_free_ruler_smoke.py "${P1_COMMON[@]}" \
  --table "$P1_OLD_TABLE" --table-name old_config_reference_s2_at_same_16k_target \
  --table-factor 2 --expected-active-sha256 fbd2f80a462f3271e65a8cdc9f3acb81b46c4afcf79c3be73509c36ac1f0bf0b \
  --long-attention-scaling 1.0512928913614359 \
  --lengths 16384 --output "$P1_ROOT/misreferenced_control_16k"

"$P1_PYTHON" -u scripts/eval/eval_reference_coupling_nll.py \
  --checkpoint "$P1_CHECKPOINT" --expected-weight-sha256 "$P1_WEIGHT_SHA" \
  --data-root "$P1_ROOT/natural_holdout" \
  --coupling-manifest "$P1_ROOT/coupling_s4/manifest.json" \
  --baseline-manifest "$P1_ROOT/baselines_s4/manifest.json" \
  --lengths 4096 8192 16384 --output "$P1_ROOT/nll_s4"
