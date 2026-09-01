#!/usr/bin/env bash
# One fixed arm of the independently seeded 80-row K32 crossing confirmation.
set -euo pipefail
: "${K32_CHECKPOINT:?}"
: "${K32_DATA_ROOT:?}"
: "${K32_OUTPUT_ROOT:?}"
: "${K32_ARM:?}"
K32_PYTHON="${K32_PYTHON:-python3}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
k32_data_hash=$("$K32_PYTHON" - "$K32_DATA_ROOT/manifest.json" <<'PY'
import hashlib, json, sys
p = sys.argv[1]
m = json.load(open(p))
assert m["status"] == "INSTRUCT_RULER_TRANSFER_PREPARED"
assert m["seed"] == 202609026 and m["samples_per_cell"] == 80
assert m["lengths"] == [32768, 65536]
assert m["tasks"] == ["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt"]
print(hashlib.sha256(open(p, "rb").read()).hexdigest())
PY
)
K32_COMMON=(--checkpoint "$K32_CHECKPOINT" --data-root "$K32_DATA_ROOT"
  --native-context-length 32768 --profile-target-length 65536 --table-factor 2
  --expected-weight-sha256 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
  --expected-native-sha256 6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3
  --expected-data-manifest-sha256 "$k32_data_hash"
  --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt
  --lengths 32768 65536 --limit-per-cell 80)
if [[ "$K32_ARM" == native ]]; then
  K32_METHOD=(--method native)
else
  : "${K32_TABLE:?}"
  case "$K32_ARM" in
    physical_x) k32_hash=b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb ;;
    normalized_index) k32_hash=8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f ;;
    *) exit 2 ;;
  esac
  K32_METHOD=(--method external_table_static --table "$K32_TABLE"
    --table-name "k32_${K32_ARM}_s2_independent_confirmation"
    --table-support native_div_factor --expected-active-sha256 "$k32_hash"
    --long-attention-scaling 1.0512928913614359)
fi
"$K32_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
  "${K32_COMMON[@]}" "${K32_METHOD[@]}" --output "$K32_OUTPUT_ROOT/$K32_ARM"
