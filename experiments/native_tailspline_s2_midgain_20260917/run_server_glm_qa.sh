#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "PLAN_ONLY: prepare GLM-tokenized Native-32K Natural-QA60, then run Native and NTS2."
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/native_tailspline_s2_midgain_glm_20260917
model=/root/models/GLM-4-9B-0414
python=/root/miniconda3/bin/python
source=/root/autodl-tmp/hybrid-rope-target-free-real-data-v3/longbench
exclude=${plan}/tailspline_olmo_s4_naturalqa631/assets/inputs.jsonl
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
table=${root}/tables/nts2.json

cd "$repo"
mkdir -p "$root"/{assets,tables,runs,logs,reports}
if [[ ! -f "$root/assets/qa/manifest.json" ]]; then
  "$python" -m experiments.native_enhancement_oral_20260915.prepare_native_naturalqa \
    --archive "$source/data.zip" --config-root "$source/official_config" \
    --model "$model" --exclude-panel "$exclude" --out "$root/assets/qa" \
    --native-length 32768 --rows-per-task 20 >"$root/logs/prepare_qa.out" 2>&1
fi
"$python" -m experiments.native_tailspline_s2_midgain_20260917.prepare \
  --config "$model/config.json" --out "$table" >"$root/logs/prepare_table.out" 2>&1

for arm in native nts2; do
  args=(--data "$data" --model "$model" --arm Native
        --extra-panel "$root/assets/qa/inputs.jsonl" --only-extra-panels --skip-lm
        --prefill-chunk-size 32768 --batch-size 1 --out "$root/runs/$arm")
  if [[ "$arm" == nts2 ]]; then
    args+=(--static-table-json "$table" --table-label nts2_glm_native32k_qa)
  fi
  "$python" -m experiments.olmo_recovery_20260912.recovery_v2_eval "${args[@]}" --execute \
    >"$root/logs/${arm}.out" 2>&1
done

"$python" -m experiments.native_tailspline_s2_midgain_20260917.report_glm_qa --root "$root"
echo GLM_NTS2_QA_COMPLETE
