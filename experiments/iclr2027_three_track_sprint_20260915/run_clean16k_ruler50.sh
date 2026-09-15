#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s4_16k_ruler50_clean
classic=${plan}/tailspline_llama_s4_classic
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=${root}/assets/inputs.jsonl
python_bin=/root/miniconda3/bin/python
parts=${root}/source_parts
nonqa_tasks=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe
qa_tasks=qa_1,qa_2

mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prepare_shard() {
  local shard=$1
  local tasks=$2
  local output=${root}/assets/${shard}
  "${python_bin}" -m \
    experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_32k_ruler200_clean \
    --source-parts "${parts}" --model "${model}" --out "${output}" \
    --length 16384 --rows-per-task 50 --tasks "${tasks}"
}

run_shard() {
  local arm=$1
  local shard=$2
  local expected=$3
  local shard_panel=${root}/assets/${shard}/inputs.jsonl
  local run=${root}/runs/${arm}_${shard}
  if [[ -f "${run}/status.json" ]] && "${python_bin}" - "${run}/status.json" "${expected}" <<'PY'
import json, sys
expected=int(sys.argv[2])
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": expected, "lm_rows": 0})
PY
  then
    return
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${shard_panel}" --only-extra-panels --skip-lm --length-cap 16384 \
    --prefill-chunk-size 8192 --batch-size 1 \
    --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_16k_ruler50_clean_${arm}_${shard}" \
    --out "${run}" --execute >"${root}/logs/${arm}_${shard}.log" 2>&1
}

prepare_shard nonqa11 "${nonqa_tasks}"
for arm in tailspline mrpro; do run_shard "${arm}" nonqa11 550; done

while [[ ! -f "${parts}/qa_1/manifest.json" ]] || \
      [[ ! -f "${parts}/qa_2/manifest.json" ]] || \
      ! grep -q '"status": "COMPLETE"' "${parts}/qa_1/manifest.json" || \
      ! grep -q '"status": "COMPLETE"' "${parts}/qa_2/manifest.json"; do
  sleep 10
done
prepare_shard qa2 "${qa_tasks}"
for arm in tailspline mrpro; do run_shard "${arm}" qa2 100; done

report=${root}/reports/tailspline_vs_mrpro_full13_16k_50_per_task_clean.json
if [[ ! -f "${report}" ]]; then
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source "tailspline=${root}/runs/tailspline_nonqa11/generations.jsonl" \
    --source "tailspline=${root}/runs/tailspline_qa2/generations.jsonl" \
    --source "mrpro=${root}/runs/mrpro_nonqa11/generations.jsonl" \
    --source "mrpro=${root}/runs/mrpro_qa2/generations.jsonl" \
    --candidate tailspline --baseline mrpro --length 16384 --out "${report}"
fi

printf 'CLEAN16K_RULER50_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
