#!/usr/bin/env bash
set -Eeuo pipefail

key=${1:-}
execute=${2:-}
if [[ "$execute" != "--execute" || ! "$key" =~ ^(llama|llama_confirm|qwen3)$ ]]; then
  echo "Usage: run_server_model_qa.sh {llama|llama_confirm|qwen3} --execute"
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
python=/root/miniconda3/bin/python
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
case "$key" in
  llama)
    root=${plan}/native_tailspline_s2_midgain_llama_qa_20260917
    model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
    native_length=8192
    model_label=Meta-Llama-3-8B-Instruct
    ;;
  llama_confirm)
    root=${plan}/native_tailspline_s2_midgain_llama_qa_confirm_20260917
    model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
    native_length=8192
    model_label=Meta-Llama-3-8B-Instruct
    ;;
  qwen3)
    root=${plan}/native_tailspline_s2_midgain_qwen3_qa_20260917
    model=/root/autodl-tmp/rope_qwen_baseline_20260907/model
    native_length=32768
    model_label=Qwen2.5-3B-Instruct
    ;;
esac
table=${root}/tables/nts2.json
test -f "$root/assets/qa/manifest.json"
cd "$repo"
mkdir -p "$root"/{tables,runs,logs,reports}
"$python" -m experiments.native_tailspline_s2_midgain_20260917.prepare \
  --config "$model/config.json" --out "$table" >"$root/logs/prepare_table.out" 2>&1
for arm in native nts2; do
  args=(--data "$data" --model "$model" --arm Native
        --extra-panel "$root/assets/qa/inputs.jsonl" --only-extra-panels --skip-lm
        --prefill-chunk-size "$native_length" --batch-size 1 --out "$root/runs/$arm")
  if [[ "$arm" == nts2 ]]; then
    args+=(--static-table-json "$table" --table-label "nts2_${key}_native_qa")
  fi
  "$python" -m experiments.olmo_recovery_20260912.recovery_v2_eval "${args[@]}" --execute \
    >"$root/logs/${arm}.out" 2>&1
done
"$python" -m experiments.native_tailspline_s2_midgain_20260917.report_paired_qa \
  --root "$root" --model "$model_label" --native-length "$native_length"
echo "${key}_NTS2_QA_COMPLETE"
