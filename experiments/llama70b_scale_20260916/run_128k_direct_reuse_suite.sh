#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY: reuse frozen Llama-3 128K PPL10 and NIAH-8x10 assets with the 70B model\n'
  exit 0
fi

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
assets=${LLAMA128K_ASSET_ROOT:-${plan}/tailspline_llama_s16_128k_gate}
root=${LLAMA70B_128K_ROOT:-${plan}/llama3_70b_s16_128k_direct_reuse}
model=${LLAMA70B_MODEL:-/root/autodl-tmp/models/llama-3-70b-Instruct-bnb-4bit}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
niah_tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery
)

mkdir -p "${root}/ppl/runs" "${root}/niah/runs" "${root}/logs" "${root}/reports" "${root}/runtime"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

exec 9>"${gpu_lock}"
flock -n 9 || { printf 'REFUSE: GPU lock is held\n' >&2; exit 73; }
test -f "${assets}/assets/ready.json" || { printf 'REFUSE: frozen 128K assets are missing\n' >&2; exit 76; }
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${total_mib}" -lt 90000 ]]; then
  printf 'REFUSE: 70B 128K execution requires at least 90,000 MiB; found %s MiB\n' "${total_mib}" >&2
  exit 75
fi

device_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
runtime_report=${root}/runtime/prefill_128k_${device_uuid}.json
if [[ ! -f ${runtime_report} ]]; then
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${model}" --table "${assets}/tables/tailspline.json" \
    --panel "${assets}/assets/full13/inputs.jsonl" \
    --lm-array "${assets}/assets/ppl10/lm.npy" \
    --length 131072 --chunks 32768 --minimum-free-fraction 0.03 \
    --out "${runtime_report}" >"${root}/logs/prefill_benchmark.log" 2>&1
fi
read -r generation_chunk lm_chunk < <("${python_bin}" - "${runtime_report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
g=x.get('recommended_generation_chunk');m=x.get('recommended_lm_chunk')
if g is None or m is None: raise SystemExit('no safe 128K prefill path')
print(int(g),int(m))
PY
)

run_ppl() {
  local arm=$1 out=${root}/ppl/runs/$1
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':0,'lm_rows':10} else 1)
PY
  then return; fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${assets}/assets/ppl10/manifest.json" --model "${model}" --arm Native \
    --only-extra-panels --lm-length-cap 131072 --lm-limit-documents 10 \
    --lm-prefill-chunk-size "${lm_chunk}" --batch-size 1 \
    --static-table-json "${assets}/tables/${arm}.json" \
    --table-label "llama3_70b_nf4_s16_128k_ppl10_${arm}" --out "${out}" --execute \
    >"${root}/logs/ppl_${arm}.log" 2>&1
}

run_niah() {
  local arm=$1 out=${root}/niah/runs/$1 task_args=()
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':80,'lm_rows':0} else 1)
PY
  then return; fi
  for task in "${niah_tasks[@]}"; do task_args+=(--task "${task}"); done
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${assets}/assets/ppl10/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${assets}/assets/full13/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 131072 --prefill-chunk-size "${generation_chunk}" --batch-size 1 \
    "${task_args[@]}" --static-table-json "${assets}/tables/${arm}.json" \
    --table-label "llama3_70b_nf4_s16_128k_niah8x10_${arm}" --out "${out}" --execute \
    >"${root}/logs/niah_${arm}.log" 2>&1
}

run_ppl tailspline
run_ppl mrpro
"${python_bin}" - "${root}" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);arms={}
for arm in ('tailspline','mrpro'):
    rows=[json.loads(x) for x in (root/f'ppl/runs/{arm}/lm_rows.jsonl').read_text().splitlines() if x]
    if [(r['document'],r['length']) for r in rows] != [(i,131072) for i in range(10)]: raise ValueError(arm)
    loss=sum(r['whole_loss_sum'] for r in rows);count=sum(r['whole_target_count'] for r in rows)
    arms[arm]={'nll':loss/count,'ppl':math.exp(loss/count),'documents':10,'target_tokens':count}
out={'status':'COMPLETE','contract':'LLAMA3_70B_NF4_PPL10_128K_V1','arms':arms,
     'delta_nll_tailspline_minus_mrpro':arms['tailspline']['nll']-arms['mrpro']['nll']}
p=root/'reports/ppl10_128k.json';tmp=p.with_name(p.name+'.incomplete');tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');tmp.replace(p)
PY
printf 'LLAMA3_70B_PPL10_128K_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/ppl_complete.txt"

run_niah tailspline
run_niah mrpro
"${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source "tailspline=${root}/niah/runs/tailspline/generations.jsonl" \
  --source "mrpro=${root}/niah/runs/mrpro/generations.jsonl" \
  --candidate tailspline --baseline mrpro --length 131072 \
  --out "${root}/reports/niah8x10_128k.json"
printf 'LLAMA3_70B_128K_DIRECT_REUSE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
