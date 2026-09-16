#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=${plan}/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl
ppl=${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json
generation_root=${plan}/tailspline_llama_s4_32k_ruler200_clean/runs
lm_root=${plan}/tailspline_llama_s4_classic/runs
table=${plan}/official_yarn_quick/llama3_8b_s4_32k/tables/yarn.json
root=${plan}/official_yarn_llama_niah200_ppl46
tasks=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery)

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${root}/logs"

exec 9>"${gpu_lock}"
flock -n 9 || { echo "REFUSE: GPU lock is owned" >&2; exit 73; }

if [[ ! -f "${root}/run/status.json" ]]; then
  command=("${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval
    --data "${ppl}" --model "${model}" --arm Native
    --extra-panel "${panel}" --only-extra-panels
    --length-cap 32768 --lm-length-cap 32768
    --prefill-chunk-size 8192 --lm-prefill-chunk-size 0
    --batch-size 1 --longest-first
    --static-table-json "${table}"
    --table-label llama3_8b_s4_32k_official_static_yarn_niah200_ppl46
    --out "${root}/run" --execute)
  for task in "${tasks[@]}"; do command+=(--task "${task}"); done
  "${command[@]}" >"${root}/logs/run.log" 2>&1
fi

report_command=("${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.matched_three_method_quick_report
  --condition llama3_8b_s4_32k_niah200_ppl46
  --target-length 32768 --rows-per-task 200 --ppl-documents 46
  --panel "${panel}" --ppl-manifest "${ppl}"
  --arm tailspline "${generation_root}/tailspline/generations.jsonl" "${lm_root}/tailspline/lm_rows.jsonl" "${generation_root}/tailspline/contract.json" "${lm_root}/tailspline/contract.json"
  --arm mrpro "${generation_root}/mrpro/generations.jsonl" "${lm_root}/mrpro/lm_rows.jsonl" "${generation_root}/mrpro/contract.json" "${lm_root}/mrpro/contract.json"
  --arm yarn "${root}/run/generations.jsonl" "${root}/run/lm_rows.jsonl" "${root}/run/contract.json" "${root}/run/contract.json"
  --out "${root}/report.json")
for task in "${tasks[@]}"; do report_command+=(--task "${task}"); done
"${report_command[@]}" >"${root}/logs/report.log" 2>&1

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);status=json.loads((root/'run/status.json').read_text());report=json.loads((root/'report.json').read_text())
if status!={'status':'COMPLETE','rows':1600,'lm_rows':46}: raise SystemExit(f'run incomplete: {status}')
if report.get('status')!='COMPLETE' or report.get('identity',{}).get('paired_generation_rows')!=1600 or report.get('identity',{}).get('ppl_documents')!=46: raise SystemExit('report incomplete')
owners=[]
for name in ('run/generations.jsonl','run/lm_rows.jsonl','run/contract.json','report.json'):
 p=root/name;owners.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'status':'LLAMA3_8B_S4_OFFICIAL_STATIC_YARN_NIAH200_PPL46_COMPLETE_V1','method_identity':'official static YaRN, zero-training installation','niah_tasks':8,'rows_per_task':200,'generation_rows':1600,'ppl_documents':46,'owners':owners}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
