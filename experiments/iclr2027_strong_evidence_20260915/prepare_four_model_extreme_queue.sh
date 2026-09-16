#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
upstream=${RULER_UPSTREAM:-/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a}
long_sources=${LONG_SOURCES:-/root/autodl-tmp/nongeometric_screen_20260909/long_sources}
suite=${plan}/four_model_128k_extreme

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

conditions=(
  "qwen25_3b_64k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|2|65536|20263001|5100"
  "qwen25_3b_128k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|4|131072|20263101|5200"
  "qwen25_1p5b_128k|/root/autodl-tmp/qwen25_1p5b_32k|qwen25_1p5b|4|131072|20263201|5300"
  "olmo2_1b_64k|/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct|olmo2_1b|16|65536|20263301|5400"
)
niah_tasks=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery

prepare_one() {
  local spec=$1 name model model_id scale target seed offset root
  IFS='|' read -r name model model_id scale target seed offset <<<"${spec}"
  root=${suite}/${name}
  mkdir -p "${root}/tables" "${root}/logs" "${root}/reports"
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer \
    --model "${model}" --model-id "${model_id}" --data-root "${upstream}" \
    --out "${root}/assets" --scale "${scale}" --lengths "${target}" \
    --rows-per-task 5 --seed "${seed}" --qa-offset "${offset}" --tasks "${niah_tasks}" \
    >"${root}/logs/prepare_ruler.log" 2>&1
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_128k_ppl \
    --model "${model}" --model-id "${model_id}" --source-root "${long_sources}" \
    --source-manifest "${long_sources}/sources.json" --out "${root}/ppl" \
    --length "${target}" --documents 5 --seed "${seed}" \
    >"${root}/logs/prepare_ppl.log" 2>&1
  for arm_method_role in "tailspline tailspline candidate" "mrpro mrpro baseline"; do
    read -r arm method role <<<"${arm_method_role}"
    output=${root}/tables/${arm}.json
    if [[ ! -f "${output}" ]]; then
      "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
        --config "${model}/config.json" --method "${method}" --scale "${scale}" \
        --candidate-id "extreme_${model_id}_s${scale}_${arm}" --model-id "${model_id}" \
        --role "${role}" --changed-variable internal_frequency_allocation --out "${output}"
    fi
  done
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.validate_extreme_condition \
    --root "${root}" --model "${model}" --model-id "${model_id}" \
    --target "${target}" --scale "${scale}" --rows-per-task 5 --ppl-documents 5 \
    >"${root}/logs/validate.log" 2>&1
}

running=()
for spec in "${conditions[@]}"; do
  prepare_one "${spec}" &
  running+=("$!")
  if [[ ${#running[@]} -ge 3 ]]; then
    wait "${running[0]}"
    running=("${running[@]:1}")
  fi
done
for pid in "${running[@]}"; do wait "${pid}"; done

"${python_bin}" - "${suite}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
names=("qwen25_3b_64k","qwen25_3b_128k","qwen25_1p5b_128k","olmo2_1b_64k")
records={name:json.loads((root/name/'ready.json').read_text()) for name in names}
expected={
 "qwen25_3b_64k":(32768,65536,2),"qwen25_3b_128k":(32768,131072,4),
 "qwen25_1p5b_128k":(32768,131072,4),"olmo2_1b_64k":(4096,65536,16),
}
for name,(native,target,scale) in expected.items():
 r=records[name]
 if r['status']!='EXTREME_CONDITION_READY_V1' or (r['native_length'],r['target_length'],r['scale'])!=(native,target,scale):
  raise ValueError(f'condition identity drift: {name}')
payload={'status':'FOUR_MODEL_EXTREME_ASSETS_READY_V1','conditions':records,
         'llama_reuses':'tailspline_llama_s16_128k_gate','gpu_execution':False}
tmp=root/'ready.json.incomplete';tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(root/'ready.json')
print(json.dumps({'status':payload['status'],'conditions':list(records)}))
PY

printf 'FOUR_MODEL_EXTREME_ASSETS_READY %s\n' "$(date -u +%FT%TZ)"
