#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "PLAN_ONLY: reuse frozen Native/NCP baselines; add NTS2 RULER130, QA99, and LM128 only."
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
base=${plan}/native_research_20260916
root=${plan}/native_tailspline_s2_midgain_20260917
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
python=/root/miniconda3/bin/python
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
table=${root}/tables/nts2.json

cd "$repo"
mkdir -p "$root"/{tables,runs,logs,reports}
"$python" -m experiments.native_tailspline_s2_midgain_20260917.prepare \
  --config "$model/config.json" --out "$table" >"$root/logs/prepare.out" 2>&1

run_generation() {
  local suite="$1" panel="$2" out="$3"
  "$python" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "$data" --model "$model" --arm Native \
    --extra-panel "$panel" --only-extra-panels --skip-lm \
    --prefill-chunk-size 4096 --batch-size 1 \
    --static-table-json "$table" --table-label "nts2_${suite}" \
    --out "$out" --execute >"$root/logs/${suite}.out" 2>&1
}

run_generation ruler "$base/assets/ruler_confirm_13x10/panels/4096/inputs.jsonl" "$root/runs/ruler" &
pids=("$!")
run_generation qa "$base/assets/naturalqa_3x80/inputs.jsonl" "$root/runs/qa" &
pids+=("$!")
"$python" -m experiments.native_enhancement_oral_20260915.run_native_lm \
  --model "$model" --manifest "$base/assets/lm128/manifest.json" \
  --tokens "$base/assets/lm128/tokens_128x4097.npy" \
  --candidate-table "$table" --candidate-label native_tailspline_s2_midgain_v1 \
  --arm candidate --allow-paired-parallel --out "$root/runs/lm" --execute \
  >"$root/logs/lm.out" 2>&1 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "NTS2_SUBRUN_FAILED; completed raw rows are preserved" >&2
  exit 1
fi

"$python" -m experiments.native_tailspline_s2_midgain_20260917.report \
  --root "$root" --baseline-root "$base" --out "$root/reports/decision.json"
echo NTS2_COMPLETE
