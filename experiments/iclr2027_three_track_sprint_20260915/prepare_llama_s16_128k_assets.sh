#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s16_128k_gate
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
long_sources=/root/autodl-tmp/nongeometric_screen_20260909/long_sources
python_bin=/root/miniconda3/bin/python
tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

mkdir -p "${root}/source_parts" "${root}/assets" "${root}/tables" "${root}/logs"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

running=()
wait_one() {
  local pid=${running[0]}
  wait "${pid}"
  running=("${running[@]:1}")
}
for index in "${!tasks[@]}"; do
  task=${tasks[$index]}
  part=${root}/source_parts/${task}
  if [[ -f "${part}/manifest.json" ]] && grep -q '"status": "COMPLETE"' "${part}/manifest.json"; then
    continue
  fi
  "${python_bin}" -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model}" --upstream "${upstream}" --out "${part}" \
    --stage H --contract planb --tasks "${task}" --caps 131072 \
    --counts-by-cap 131072:10 --selection-mode source-order \
    --qa-base-offset 5800 --seed "$((20262001 + index))" \
    >"${root}/logs/prepare_${task}.log" 2>&1 &
  running+=("$!")
  if [[ ${#running[@]} -ge 4 ]]; then wait_one; fi
done
while [[ ${#running[@]} -gt 0 ]]; do wait_one; done

"${python_bin}" -m \
  experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_32k_ruler200_clean \
  --source-parts "${root}/source_parts" --model "${model}" \
  --out "${root}/assets/full13" --length 131072 --rows-per-task 10

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.prepare_llama_s16_ppl10 \
  --model "${model}" --source-root "${long_sources}" \
  --source-manifest "${long_sources}/sources.json" --out "${root}/assets/ppl10"

make_table() {
  local name=$1 method=$2 role=$3 output=${root}/tables/$1.json
  if [[ -f "${output}" ]]; then return; fi
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${model}/config.json" --method "${method}" --scale 16 \
    --candidate-id "llama3_8b_s16_128k_${name}" --model-id Meta-Llama-3-8B-Instruct \
    --role "${role}" --changed-variable exponent_allocation --out "${output}"
}
make_table tailspline tailspline candidate
make_table mrpro mrpro baseline

"${python_bin}" - "${root}" <<'PY'
from collections import Counter
import hashlib, json, math, sys
from pathlib import Path
root=Path(sys.argv[1])
full=json.loads((root/'assets/full13/manifest.json').read_text())
ppl=json.loads((root/'assets/ppl10/manifest.json').read_text())
rows=[json.loads(line) for line in (root/'assets/full13/inputs.jsonl').read_text().splitlines()]
tables={name:json.loads((root/f'tables/{name}.json').read_text()) for name in ('tailspline','mrpro')}
tasks=("niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2","niah_multikey_3","niah_multivalue","niah_multiquery","vt","cwe","fwe","qa_1","qa_2")
if full.get('rows')!=130 or Counter(row['task'] for row in rows)!=Counter({task:10 for task in tasks}): raise ValueError('Full-13 128K drift')
if len({row['prompt_sha256'] for row in rows})!=130 or any(len(row['prompt_ids'])+row['max_new_tokens']>131072 for row in rows): raise ValueError('128K prompt identity drift')
if ppl.get('documents')!=10 or ppl.get('lengths')!=[131072]: raise ValueError('PPL10 128K drift')
if {tuple(value['band_envelope']) for value in tables.values()}!={(18,35)}: raise ValueError('S16 band drift')
if {float(value['gain']).hex() for value in tables.values()}!={(1+0.1*math.log(16)).hex()}: raise ValueError('S16 gain drift')
receipt={'status':'TAILSPLINE_LLAMA_S16_128K_ASSETS_READY_V1','ruler_rows':130,'ppl_documents':10,'length':131072,'band':[18,35],'gain':tables['tailspline']['gain'],'table_sha256':{name:value['table_sha256_float32'] for name,value in tables.items()},'inputs_sha256':full['inputs_sha256'],'lm_array_sha256':ppl['lm_array_sha256'],'gpu_execution':False}
(root/'assets/ready.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY

printf 'LLAMA_S16_128K_ASSETS_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/assets_complete.txt"
