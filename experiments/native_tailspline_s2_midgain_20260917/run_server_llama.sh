#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "PLAN_ONLY: reuse Llama Native Full-13x10 and PPL46; add only the frozen NTS2 arm."
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
baseline=${plan}/ca_ncp_llama_native_20260917
ppl_baseline=${plan}/tailspline_llama_s4_classic/runs/native_ppl_8k
root=${plan}/native_tailspline_s2_midgain_llama_20260917
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
python=/root/miniconda3/bin/python
data=${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json
table=${root}/tables/nts2.json

cd "$repo"
mkdir -p "$root"/{tables,runs,logs,reports}
"$python" -m experiments.native_tailspline_s2_midgain_20260917.prepare \
  --config "$model/config.json" --out "$table" >"$root/logs/prepare.out" 2>&1

"$python" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "$data" --model "$model" --arm Native \
  --extra-panel "$baseline/assets/pilot/inputs.jsonl" --only-extra-panels --skip-lm \
  --prefill-chunk-size 8192 --batch-size 1 \
  --static-table-json "$table" --table-label nts2_llama_native8k_ruler \
  --out "$root/runs/ruler" --execute >"$root/logs/ruler.out" 2>&1

"$python" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "$data" --model "$model" --arm Native --only-extra-panels \
  --lm-length-cap 8192 --lm-prefill-chunk-size 0 \
  --static-table-json "$table" --table-label nts2_llama_native8k_ppl \
  --out "$root/runs/ppl" --execute >"$root/logs/ppl.out" 2>&1

"$python" -m experiments.native_tailspline_s2_midgain_20260917.report_llama \
  --root "$root" --baseline-root "$baseline" --ppl-baseline "$ppl_baseline" \
  --out "$root/reports/decision.json"
echo LLAMA_NTS2_COMPLETE
