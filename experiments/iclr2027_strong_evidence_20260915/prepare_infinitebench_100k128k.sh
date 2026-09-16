#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
data_root=${INFINITEBENCH_ROOT:-/root/autodl-tmp/InfiniteBench}
root=${plan}/tailspline_llama_s16_infinitebench_100k128k

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

prepare_task() {
  local short=$1 task=$2
  local out=${root}/${short}/assets
  if [[ -f "${out}/manifest.json" ]]; then
    "${python_bin}" - "${out}/manifest.json" "${task}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if (x.get('status')!='COMPLETE' or x.get('benchmark')!='infinitebench'
    or x.get('model_id')!='llama3_8b' or x.get('scale')!=16
    or x.get('lengths')!=[131072] or x.get('rows_per_task')!=100
    or x.get('tasks')!=[sys.argv[2]] or x.get('minimum_input_tokens')!=100000):
 raise SystemExit('existing InfiniteBench asset contract differs')
PY
    return
  fi
  mkdir -p "${root}/${short}/logs"
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_natural_long \
    --benchmark infinitebench --model "${model}" --model-id llama3_8b \
    --data-root "${data_root}" --out "${out}" --scale 16 --lengths 131072 \
    --rows-per-task 100 --task "${task}" --minimum-input-tokens 100000 \
    >"${root}/${short}/logs/prepare.log" 2>&1
}

prepare_task en_dia longdialogue_qa_eng &
pid_dia=$!
prepare_task en_qa longbook_qa_eng &
pid_qa=$!
wait "${pid_dia}"
wait "${pid_qa}"

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);conditions={
 'en_dia':'longdialogue_qa_eng','en_qa':'longbook_qa_eng',
};receipt={}
for short,task in conditions.items():
 p=root/short/'assets/manifest.json';x=json.loads(p.read_text())
 selected=int(x['summary']['selected_rows'])
 if selected<=0 or selected>100 or x['tasks']!=[task]: raise ValueError(short)
 rows=[json.loads(line) for line in (root/short/'assets/inputs.jsonl').read_text().splitlines() if line.strip()]
 lengths=[int(row['input_tokens']) for row in rows]
 if len(rows)!=selected or min(lengths)<100000 or max(lengths)>131072: raise ValueError(f'{short} selected length range')
 receipt[short]={'task':task,'selected_rows':selected,
                 'minimum_input_tokens':min(lengths),
                 'maximum_input_tokens':max(lengths),
                 'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
out={'status':'INFINITEBENCH_100K128K_ASSETS_READY_V1','conditions':receipt,
     'scale':16,'target_length':131072,'model_id':'llama3_8b','gpu_execution':False}
tmp=root/'ready.json.incomplete';tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');tmp.replace(root/'ready.json')
print(json.dumps(out,sort_keys=True))
PY
