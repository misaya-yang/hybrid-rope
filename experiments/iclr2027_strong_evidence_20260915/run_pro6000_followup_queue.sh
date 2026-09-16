#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
llama=${plan}/tailspline_llama_s16_128k_gate
extreme=${plan}/four_model_128k_extreme
infinite=${plan}/tailspline_llama_s16_infinitebench_100k128k
queue=${plan}/pro6000_followup_queue

mkdir -p "${queue}/logs"
cd "${repo}"
export PYTHONPATH=.

# This is a dependency check, not a scientific score gate: follow-ups start for
# either Llama outcome once both frozen arms and their report exist.
"${python_bin}" - "${llama}" "${extreme}" <<'PY'
import json,sys
from pathlib import Path
llama,extreme=map(Path,sys.argv[1:])
for arm in ('tailspline','mrpro'):
 status=json.loads((llama/f'runs/{arm}/status.json').read_text())
 if status!={'status':'COMPLETE','rows':130,'lm_rows':10}: raise ValueError(f'llama {arm}: {status}')
if not (llama/'reports/tailspline_vs_mrpro_s16_128k_gate.json').is_file(): raise FileNotFoundError('Llama report')
if json.loads((extreme/'ready.json').read_text()).get('status')!='FOUR_MODEL_EXTREME_ASSETS_READY_V1': raise ValueError('extreme assets')
PY

bash experiments/iclr2027_strong_evidence_20260915/run_four_model_extreme_queue.sh \
  >"${queue}/logs/four_model_extreme.log" 2>&1
bash experiments/iclr2027_strong_evidence_20260915/run_infinitebench_100k128k.sh \
  >"${queue}/logs/infinitebench.log" 2>&1

"${python_bin}" - "${extreme}" "${infinite}" "${queue}" <<'PY'
import hashlib,json,sys
from pathlib import Path
extreme,infinite,queue=map(Path,sys.argv[1:])
owners=[extreme/'complete.json',infinite/'complete.json']
for path in owners:
 if not path.is_file(): raise FileNotFoundError(path)
receipt={'status':'PRO6000_FOLLOWUP_QUEUE_COMPLETE_V1','owners':[
 {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in owners
]}
tmp=queue/'complete.json.incomplete';tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(queue/'complete.json')
print(json.dumps(receipt,sort_keys=True))
PY
