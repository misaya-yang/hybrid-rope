#!/usr/bin/env bash
# Two frozen arms for the independently seeded K128 coordinate confirmation.
set -euo pipefail

: "${K128_CHECKPOINT:?}"
: "${K128_DATA_ROOT:?}"
: "${K128_OUTPUT_ROOT:?}"
: "${K128_PHYSICAL_TABLE:?}"
: "${K128_INDEX_TABLE:?}"
K128_PYTHON="${K128_PYTHON:-python3}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

k128_data_hash=$("$K128_PYTHON" - "$K128_DATA_ROOT/manifest.json" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
manifest = json.load(open(path))
assert manifest["status"] == "INSTRUCT_RULER_TRANSFER_PREPARED"
assert manifest["seed"] == 202609028
assert manifest["samples_per_cell"] == 80
assert manifest["lengths"] == [16384]
assert manifest["tasks"] == [
    "niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt",
]
print(hashlib.sha256(open(path, "rb").read()).hexdigest())
PY
)

common=(
  --checkpoint "$K128_CHECKPOINT"
  --data-root "$K128_DATA_ROOT"
  --method external_table_static
  --table-support native_div_factor
  --native-context-length 8192
  --profile-target-length 16384
  --table-factor 4
  --expected-weight-sha256 584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933
  --expected-native-sha256 cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15
  --expected-data-manifest-sha256 "$k128_data_hash"
  --long-attention-scaling 1.102585782722872
  --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt
  --lengths 16384
  --limit-per-cell 80
)

mkdir -p "$K128_OUTPUT_ROOT"
for arm in physical_x normalized_index; do
  if [[ "$arm" == physical_x ]]; then
    table="$K128_PHYSICAL_TABLE"
    expected=be5c2b3b4ce01d7fe6020cb01d9041e10aad93b9cdb03e989e64b8fa17561423
  else
    table="$K128_INDEX_TABLE"
    expected=1b908f90aebccc006521b3840b5662c217caa040d173c974527d33c7ea9e9849
  fi
  test ! -e "$K128_OUTPUT_ROOT/$arm"
  "$K128_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
    "${common[@]}" \
    --table "$table" \
    --table-name "gemma_k128_${arm}_s4_independent_confirmation" \
    --expected-active-sha256 "$expected" \
    --output "$K128_OUTPUT_ROOT/$arm"
done
