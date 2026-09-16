#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
root=${plan}/tailspline_llama_s16_infinitebench_100k128k

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${root}/ready.json" ]]; then
  printf 'REFUSE: InfiniteBench 100K-128K assets are not ready\n' >&2
  exit 1
fi

run_task() {
  local short=$1
  local assets=${root}/${short}/assets out=${root}/${short}/evaluation
  if [[ -f "${out}/status.json" ]]; then
    "${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));
if x.get('status')!='COMPLETE' or x.get('arms')!=['tailspline','mrpro'] or int(x.get('rows_per_arm',0))<=0:
 raise SystemExit('existing InfiniteBench run is not complete')
PY
    return
  fi
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_natural_long \
    --model "${model}" --model-id llama3_8b --data-root "${assets}" --out "${out}" \
    --scale 16 --lengths 131072 --rows-per-task 100 \
    --data-manifest "${assets}/manifest.json" --python "${python_bin}" \
    --benchmark infinitebench --prefill-chunk-size 65536 --execute \
    >"${root}/${short}/logs/run.log" 2>&1
}

# Each task is its own resumable result owner.  En.Dia runs first because the
# paper reports a larger MrRoPE/YaRN separation on dialogue utilization.
run_task en_dia
run_task en_qa

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);reports={}
for short in ('en_dia','en_qa'):
 p=root/short/'evaluation/report.json';s=root/short/'evaluation/status.json'
 if json.loads(s.read_text()).get('status')!='COMPLETE': raise ValueError(short)
 reports[short]={'report':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
out={'status':'INFINITEBENCH_100K128K_COMPLETE_V1','reports':reports}
tmp=root/'complete.json.incomplete';tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');tmp.replace(root/'complete.json')
print(json.dumps(out,sort_keys=True))
PY
