#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY: reuse existing RULER200/PPL46/Natural-QA631 assets with the 70B model path\n'
  exit 0
fi

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
root=${LLAMA70B_ROOT:-${plan}/llama3_70b_s4_32k_direct_reuse}
model=${LLAMA70B_MODEL:-/root/autodl-tmp/models/llama-3-70b-Instruct-bnb-4bit}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
classic=${plan}/tailspline_llama_s4_classic
clean=${plan}/tailspline_llama_s4_32k_ruler200_clean
qa=${plan}/tailspline_llama_s4_naturalqa631
canary=${root}/runtime/canary.json
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

mkdir -p "${root}/ppl/runs" "${root}/ruler/runs" "${root}/qa/runs" "${root}/logs" "${root}/reports" "${root}/runtime"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec 9>"${gpu_lock}"
flock -n 9 || { printf 'REFUSE: GPU lock is held\n' >&2; exit 73; }
test -f "${canary}" || { printf 'REFUSE: validated 70B canary is missing\n' >&2; exit 76; }
read -r generation_chunk lm_chunk < <("${python_bin}" - "${canary}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status')!='LLAMA3_70B_NF4_CANARY_COMPLETE_V1': raise SystemExit('invalid canary')
print(int(x['selected_generation_prefill_chunk']),int(x['selected_lm_prefill_chunk']))
PY
)

run_ppl() {
  arm=$1; out=${root}/ppl/runs/${arm}
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':0,'lm_rows':5} else 1)
PY
  then return; fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --only-extra-panels --lm-length-cap 32768 --lm-limit-documents 5 \
    --lm-prefill-chunk-size "${lm_chunk}" \
    --batch-size 1 --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_70b_nf4_s4_32k_ppl5_${arm}" --out "${out}" --execute \
    >"${root}/logs/ppl_${arm}.log" 2>&1
}

run_ruler() {
  arm=$1; out=${root}/ruler/runs/${arm}
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':130,'lm_rows':0} else 1)
PY
  then return; fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${clean}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 32768 --limit-per-cell 10 --prefill-chunk-size "${generation_chunk}" \
    --batch-size 1 --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_70b_nf4_s4_32k_ruler10_${arm}" --out "${out}" --execute \
    >"${root}/logs/ruler_${arm}.log" 2>&1
}

run_qa() {
  arm=$1; out=${root}/qa/runs/${arm}
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':631,'lm_rows':0} else 1)
PY
  then return; fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${qa}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 32768 --prefill-chunk-size "${generation_chunk}" --batch-size 1 \
    --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_70b_nf4_s4_32k_qa631_${arm}" --out "${out}" --execute \
    >"${root}/logs/qa_${arm}.log" 2>&1
}

run_ppl tailspline
run_ppl mrpro
"${python_bin}" - "${root}" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);arms={}
for arm in ('tailspline','mrpro'):
 rows=[json.loads(x) for x in (root/f'ppl/runs/{arm}/lm_rows.jsonl').read_text().splitlines() if x]
 if [(r['document'],r['length']) for r in rows] != [(i,32768) for i in range(5)]: raise ValueError(arm)
 loss=sum(r['whole_loss_sum'] for r in rows);count=sum(r['whole_target_count'] for r in rows)
 arms[arm]={'nll':loss/count,'ppl':math.exp(loss/count),'documents':5,'target_tokens':count}
out={'status':'COMPLETE','contract':'LLAMA3_70B_NF4_PPL5_32K_V1','arms':arms,
     'delta_nll_tailspline_minus_mrpro':arms['tailspline']['nll']-arms['mrpro']['nll']}
p=root/'reports/ppl5_32k.json';tmp=p.with_name(p.name+'.incomplete');tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');tmp.replace(p)
PY
printf 'LLAMA3_70B_PPL5_32K_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/ppl_complete.txt"

run_ruler tailspline
run_ruler mrpro
run_qa tailspline
run_qa mrpro

if [[ ! -f ${root}/reports/ruler13x10.json ]]; then
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source "tailspline=${root}/ruler/runs/tailspline/generations.jsonl" \
    --source "mrpro=${root}/ruler/runs/mrpro/generations.jsonl" \
    --candidate tailspline --baseline mrpro --length 32768 \
    --out "${root}/reports/ruler13x10.json"
fi
if [[ ! -f ${root}/reports/naturalqa631.json ]]; then
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report \
    --panel "${qa}/assets/inputs.jsonl" \
    --candidate "${root}/qa/runs/tailspline/generations.jsonl" \
    --baseline "${root}/qa/runs/mrpro/generations.jsonl" \
    --out "${root}/reports/naturalqa631.json"
fi
printf 'LLAMA3_70B_DIRECT_REUSE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
