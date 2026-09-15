#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914/olmo_native_z5_enhancement
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
classic=/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic
consensus=${root}/consensus

mkdir -p "${consensus}" "${root}/logs" "${root}/reports" "${root}/runs"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${root}/complete.txt" ]]; then
  printf 'Original Native-Z5 must complete before the consensus comparison\n' >&2
  exit 1
fi

if [[ ! -f "${consensus}/status.json" ]]; then
  /root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.consensus_direction \
    --assets "${root}/assets" --model "${model}" \
    --native-optimization "${root}/optimization" --out "${consensus}" \
    >"${root}/logs/consensus_direction.log" 2>&1
fi

advance=$(/root/miniconda3/bin/python - "${consensus}/status.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
if value.get("status") != "COMPLETE": raise SystemExit("invalid consensus status")
print("1" if value.get("advance") else "0")
PY
)
if [[ "${advance}" != 1 ]]; then
  printf 'NATIVE_Z5_CONSENSUS_STOPPED_BEFORE_TASKS %s\n' "$(date -u +%FT%TZ)" | tee "${consensus}/complete.txt"
  exit 0
fi

run_arm() {
  local arm=$1
  local table=$2
  local output=${root}/runs/${arm}
  if [[ -f "${output}/status.json" ]] && /root/miniconda3/bin/python - "${output}/status.json" <<'PY'
import json, sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 229, "lm_rows": 0})
PY
  then
    return
  fi
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${root}/assets/ruler4k_full13x10.jsonl" \
    --extra-panel "${root}/assets/natural4k_three_task.jsonl" \
    --only-extra-panels --skip-lm --length-cap 4096 --prefill-chunk-size 4096 \
    --batch-size 4 --static-table-json "${table}" --table-label "olmo2_1b_${arm}" \
    --out "${output}" --execute >"${root}/logs/${arm}.log" 2>&1
}

run_arm consensus_plus "${consensus}/table_plus.json"
run_arm consensus_minus "${consensus}/table_minus.json"

/root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.consensus_report \
  --assets "${root}/assets" --native-optimization "${root}/optimization" \
  --consensus "${consensus}" --native-run "${root}/runs/native" \
  --z5-run "${root}/runs/native_z5" --plus-run "${root}/runs/consensus_plus" \
  --minus-run "${root}/runs/consensus_minus" \
  --out "${root}/reports/native_z5_consensus.json"

printf 'NATIVE_Z5_CONSENSUS_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${consensus}/complete.txt"
