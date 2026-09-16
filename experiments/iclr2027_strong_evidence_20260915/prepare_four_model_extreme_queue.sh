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
  "qwen25_3b_64k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|2|65536|20263001|5100|5"
  "qwen25_3b_128k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|4|131072|20263101|5200|5"
  "qwen25_3b_256k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|8|262144|20263151|5250|2"
  "qwen25_1p5b_128k|/root/autodl-tmp/qwen25_1p5b_32k|qwen25_1p5b|4|131072|20263201|5300|5"
  "olmo2_1b_64k|/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct|olmo2_1b|16|65536|20263301|5400|5"
)
niah_tasks=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery

prepare_one() {
  local spec=$1 name model model_id scale target seed offset ppl_documents root
  IFS='|' read -r name model model_id scale target seed offset ppl_documents <<<"${spec}"
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
    --length "${target}" --documents "${ppl_documents}" --seed "${seed}" \
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
    --target "${target}" --scale "${scale}" --rows-per-task 5 --ppl-documents "${ppl_documents}" \
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

"${python_bin}" - "${suite}/qwen25_3b_256k" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);panel=root/'assets/panels/262144'
source=panel/'inputs.jsonl';target=panel/'inputs_single.jsonl'
tasks=('niah_single_1','niah_single_2','niah_single_3')
rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
selected=[row for row in rows if row.get('task') in tasks]
if len(selected)!=15 or {row['task'] for row in selected}!=set(tasks): raise ValueError('256K single-needle subset drift')
temporary=target.with_name(target.name+'.incomplete')
with temporary.open('w') as stream:
 for row in selected: stream.write(json.dumps(row,sort_keys=True)+'\n')
os.replace(temporary,target)
manifest={'status':'QWEN25_3B_256K_SINGLE_NIAH_ASSETS_READY_V1','tasks':list(tasks),
          'rows_per_task':5,'rows':15,'length':262144,
          'inputs_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'source_inputs_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
out=panel/'single_manifest.json';tmp=out.with_name(out.name+'.incomplete')
tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.replace(tmp,out)
print(json.dumps(manifest,sort_keys=True))
PY

"${python_bin}" - "${suite}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
names=("qwen25_3b_64k","qwen25_3b_128k","qwen25_3b_256k","qwen25_1p5b_128k","olmo2_1b_64k")
records={name:json.loads((root/name/'ready.json').read_text()) for name in names}
expected={
 "qwen25_3b_64k":(32768,65536,2),"qwen25_3b_128k":(32768,131072,4),
 "qwen25_3b_256k":(32768,262144,8),
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
