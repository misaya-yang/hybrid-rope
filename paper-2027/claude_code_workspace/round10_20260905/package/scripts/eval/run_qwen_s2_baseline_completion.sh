#!/usr/bin/env bash
# Fixed P2 YaRN2 completion: long resolver first, then the same profile at Native.
set -euo pipefail
: "${P2_CHECKPOINT:?}"
: "${P2_DATA_ROOT:?}"
: "${P2_NATIVE_RESULT:?}"
: "${P2_OUTPUT_ROOT:?}"
P2_PYTHON="${P2_PYTHON:-python3}"
P2_LONG_DATA_ROOT="${P2_LONG_DATA_ROOT:-$P2_DATA_ROOT}"
P2_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$P2_REPO"
read -r p2_weight p2_native < <(
  "$P2_PYTHON" - "$P2_NATIVE_RESULT" "$P2_CHECKPOINT/config.json" <<'PY'
import hashlib, json, sys
r, c = (json.load(open(p)) for p in sys.argv[1:3])
assert r["status"] == "TARGET_FREE_RULER_SMOKE_COMPLETE"
assert r["method"]["method"] == "native"
assert r["method"]["attention_scaling"] == 1
assert r["results"]["macro_official_task_score"] > 0
assert r["protocol"]["lengths"] == [32768]
assert c["model_type"] == "qwen2" and c["max_position_embeddings"] == 32768
assert c.get("use_sliding_window", False) is False
assert c.get("rope_scaling") in (None, {})
print(r["protocol"]["checkpoint_sha256"], r["method"]["active_sha256_float32"])
PY
)
mkdir -p "$P2_OUTPUT_ROOT"
"$P2_PYTHON" scripts/analysis/export_runtime_native_rope.py \
  --config "$P2_CHECKPOINT/config.json" --expected-native-sha256 "$p2_native" \
  --output "$P2_OUTPUT_ROOT/native_inv.npy"
"$P2_PYTHON" scripts/analysis/export_static_rope_baselines.py \
  --config "$P2_CHECKPOINT/config.json" --native-inv "$P2_OUTPUT_ROOT/native_inv.npy" \
  --factor 2 --verify-transformers --output "$P2_OUTPUT_ROOT/baselines_s2"
read -r p2_yarn p2_gain < <(
  "$P2_PYTHON" -c 'import json,sys; t=json.load(open(sys.argv[1]))["tables"]["official_equation_yarn"]; print(t["tensor_sha256"],t["attention_scaling"])' \
    "$P2_OUTPUT_ROOT/baselines_s2/manifest.json"
)
for p2_length in 65536 32768; do
  p2_data_root="$P2_DATA_ROOT"
  if [[ "$p2_length" == 65536 ]]; then p2_data_root="$P2_LONG_DATA_ROOT"; fi
  p2_data=$("$P2_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$p2_data_root/manifest.json")
  "$P2_PYTHON" -u scripts/eval/target_free_ruler_smoke.py \
    --checkpoint "$P2_CHECKPOINT" --data-root "$p2_data_root" \
    --method external_table_static --table-support native_div_factor \
    --table "$P2_OUTPUT_ROOT/baselines_s2/official_equation_yarn_s2.npy" \
    --table-name qwen_same_family_official_yarn_s2 --table-factor 2 \
    --native-context-length 32768 --profile-target-length 65536 \
    --expected-weight-sha256 "$p2_weight" --expected-native-sha256 "$p2_native" \
    --expected-data-manifest-sha256 "$p2_data" --expected-active-sha256 "$p2_yarn" \
    --long-attention-scaling "$p2_gain" \
    --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --lengths "$p2_length" --limit-per-cell 20 \
    --output "$P2_OUTPUT_ROOT/official_yarn_${p2_length}"
  if [[ "$p2_length" == 65536 ]]; then
    "$P2_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="TARGET_FREE_RULER_SMOKE_COMPLETE"; assert r["results"]["macro_official_task_score"]>0, "YaRN2 long resolver failed; no reinterpretation or tuning"' \
      "$P2_OUTPUT_ROOT/official_yarn_65536/results.json"
  fi
done
