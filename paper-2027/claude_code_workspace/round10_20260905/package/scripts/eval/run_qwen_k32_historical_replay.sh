#!/usr/bin/env bash
# Fixed 24-row decoded-output replay; never a new profile or score-selection set.
set -euo pipefail
: "${P2_CHECKPOINT:?}"
: "${P2_DATA_ROOT:?}"
: "${P2_PHYSICAL_TABLE:?}"
: "${P2_INDEX_TABLE:?}"
: "${P2_OUTPUT_ROOT:?}"
P2_PYTHON="${P2_PYTHON:-python3}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
P2_COMMON=(--checkpoint "$P2_CHECKPOINT" --data-root "$P2_DATA_ROOT"
  --native-context-length 32768 --profile-target-length 65536 --table-factor 2
  --expected-weight-sha256 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
  --expected-native-sha256 6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3
  --expected-data-manifest-sha256 231535101268c6798057fd458aa1891cd7632afe73aaae11f590801e01c28090
  --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt
  --lengths 32768 65536 --limit-per-cell 1)
"$P2_PYTHON" -u scripts/eval/target_free_ruler_smoke.py "${P2_COMMON[@]}" \
  --method native --output "$P2_OUTPUT_ROOT/native"
"$P2_PYTHON" -u scripts/eval/target_free_ruler_smoke.py "${P2_COMMON[@]}" \
  --method external_table_static --table-support native_div_factor \
  --table "$P2_PHYSICAL_TABLE" --table-name historical_k32_physical_s2_replay \
  --expected-active-sha256 b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb \
  --long-attention-scaling 1.0512928913614359 --output "$P2_OUTPUT_ROOT/physical_x"
"$P2_PYTHON" -u scripts/eval/target_free_ruler_smoke.py "${P2_COMMON[@]}" \
  --method external_table_static --table-support native_div_factor \
  --table "$P2_INDEX_TABLE" --table-name historical_k32_index_s2_replay \
  --expected-active-sha256 8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f \
  --long-attention-scaling 1.0512928913614359 --output "$P2_OUTPUT_ROOT/normalized_index"
