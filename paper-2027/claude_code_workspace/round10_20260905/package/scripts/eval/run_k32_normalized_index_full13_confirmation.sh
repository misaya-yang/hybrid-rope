#!/usr/bin/env bash
# New-seed full-RULER confirmation of the frozen K32 engineering representative.
set -euo pipefail

: "${K32_CHECKPOINT:?}"
: "${K32_DATA_ROOT:?}"
: "${K32_OUTPUT_ROOT:?}"
: "${K32_INDEX_TABLE:?}"
: "${K32_YARN_TABLE:?}"
K32_PYTHON="${K32_PYTHON:-python3}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

k32_data_hash=$("$K32_PYTHON" - "$K32_DATA_ROOT/manifest.json" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
manifest = json.load(open(path))
assert manifest["status"] == "INSTRUCT_RULER_TRANSFER_PREPARED"
assert manifest["seed"] == 202609027
assert manifest["samples_per_cell"] == 20
assert manifest["lengths"] == [32768, 65536]
assert manifest["tasks"] == [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_1", "qa_2",
]
print(hashlib.sha256(open(path, "rb").read()).hexdigest())
PY
)

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
common=(
  --checkpoint "$K32_CHECKPOINT"
  --data-root "$K32_DATA_ROOT"
  --native-context-length 32768
  --profile-target-length 65536
  --table-factor 2
  --expected-weight-sha256 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
  --expected-native-sha256 6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3
  --expected-data-manifest-sha256 "$k32_data_hash"
  --tasks "${tasks[@]}"
  --lengths 32768 65536
  --limit-per-cell 20
)

mkdir -p "$K32_OUTPUT_ROOT"
test ! -e "$K32_OUTPUT_ROOT/native"
"$K32_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
  "${common[@]}" --method native --output "$K32_OUTPUT_ROOT/native"

test ! -e "$K32_OUTPUT_ROOT/normalized_index"
"$K32_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
  "${common[@]}" --method external_table_static \
  --table-support native_div_factor --table "$K32_INDEX_TABLE" \
  --table-name k32_normalized_index_s2_full13_confirmation \
  --expected-active-sha256 8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f \
  --long-attention-scaling 1.0512928913614359 \
  --output "$K32_OUTPUT_ROOT/normalized_index"

test ! -e "$K32_OUTPUT_ROOT/official_yarn"
"$K32_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
  "${common[@]}" --method external_table_static \
  --table-support native_div_factor --table "$K32_YARN_TABLE" \
  --table-name qwen_k32_official_yarn_s2_full13_confirmation \
  --expected-active-sha256 d9eb5ac0185e84f2afa85997f10e4c51de97e3a2f937325769dd45ff86a0ea59 \
  --long-attention-scaling 1.0693147180559945 \
  --output "$K32_OUTPUT_ROOT/official_yarn"
