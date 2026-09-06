#!/usr/bin/env bash
# One frozen K64 C2-s2 profile after the fixed YaRN2 long resolver passes.
set -euo pipefail
: "${P2_CHECKPOINT:?}"
: "${P2_DATA_ROOT:?}"
: "${P2_LONG_DATA_ROOT:?}"
: "${P2_OUTPUT_ROOT:?}"
P2_PYTHON="${P2_PYTHON:-python3}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
read -r p2_hash p2_gain < <(
  "$P2_PYTHON" - "$P2_OUTPUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
r = json.loads((root / "official_yarn_65536/results.json").read_text())
assert r["status"] == "TARGET_FREE_RULER_SMOKE_COMPLETE"
assert r["results"]["macro_official_task_score"] > 0
m = json.loads((root / "coupling_s2/manifest.json").read_text())
assert m["checkpoint"]["pairs"] == 64 and m["checkpoint"]["native_length"] == 32768
assert m["law"]["scale"] == 2 and m["law"]["x_high"] == .7382780681078285
assert m["law"]["x_low"] == .366403835112904 and m["gain"]["coefficient"] == .074
print(m["tables"]["dimensionless_x"]["tensor_sha256"], m["gain"]["attention_scaling"])
PY
)
for p2_length in 32768 65536; do
  p2_data_root="$P2_DATA_ROOT"
  if [[ "$p2_length" == 65536 ]]; then p2_data_root="$P2_LONG_DATA_ROOT"; fi
  p2_data=$("$P2_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$p2_data_root/manifest.json")
  "$P2_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
    --checkpoint "$P2_CHECKPOINT" --data-root "$p2_data_root" \
    --method external_table_static --table-support native_div_factor \
    --table "$P2_OUTPUT_ROOT/coupling_s2/dimensionless_x_s2.npy" \
    --table-name frozen_k64_c2_s2_unique_profile --table-factor 2 \
    --native-context-length 32768 --profile-target-length 65536 \
    --expected-weight-sha256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee \
    --expected-native-sha256 138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e \
    --expected-data-manifest-sha256 "$p2_data" --expected-active-sha256 "$p2_hash" \
    --long-attention-scaling "$p2_gain" \
    --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --lengths "$p2_length" --limit-per-cell 20 \
    --output "$P2_OUTPUT_ROOT/c2_s2_${p2_length}"
done
