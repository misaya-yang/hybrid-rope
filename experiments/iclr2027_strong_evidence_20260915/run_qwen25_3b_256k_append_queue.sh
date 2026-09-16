#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
root=${plan}/four_model_128k_extreme/qwen25_3b_256k
qa_assets=${root}/infinitebench_en_qa_assets
qa_out=${root}/infinitebench_en_qa_evaluation

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash experiments/iclr2027_strong_evidence_20260915/run_qwen25_3b_256k_ppl_niah.sh \
  >"${root}/logs/run_256k_ppl_niah.log" 2>&1

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
runtime=${root}/runtime/prefill_262144_${gpu_uuid}.json
gen_chunk=$("${python_bin}" - "${runtime}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));g=x.get('recommended_generation_chunk')
if x.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2' or g is None: raise SystemExit('missing 256K runtime receipt')
results={int(item['chunk']):item for item in x.get('results',[])}
direct=results.get(0,{}).get('generation',{});chosen=results.get(int(g),{}).get('generation',{})
if int(g) and direct.get('stable') and chosen.get('stable') and float(chosen['seconds']) >= .98*float(direct['seconds']): g=0
print(int(g))
PY
)

if [[ ! -f "${qa_out}/status.json" ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_natural_long \
    --model "${model}" --model-id qwen25_3b --data-root "${qa_assets}" --out "${qa_out}" \
    --scale 8 --lengths 262144 --rows-per-task 50 \
    --data-manifest "${qa_assets}/manifest.json" --python "${python_bin}" \
    --benchmark infinitebench --prefill-chunk-size "${gen_chunk}" --execute \
    >"${root}/logs/run_infinitebench_en_qa.log" 2>&1
fi

"${python_bin}" - "${root}" "${qa_out}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,qa=map(Path,sys.argv[1:])
base=root/'complete_256k.json';status=qa/'status.json';report=qa/'report.json'
if json.loads(base.read_text()).get('status')!='QWEN25_3B_256K_PPL_NIAH_COMPLETE_V1': raise ValueError('PPL/NIAH incomplete')
s=json.loads(status.read_text())
if s.get('status')!='COMPLETE' or s.get('arms')!=['tailspline','mrpro'] or int(s.get('rows_per_arm',0))<=0: raise ValueError('QA incomplete')
receipt={'status':'QWEN25_3B_256K_APPEND_QUEUE_COMPLETE_V1',
         'ppl_niah_receipt_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
         'qa_rows_per_arm':int(s['rows_per_arm']),
         'qa_report':str(report),'qa_report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
out=root/'append_complete.json';tmp=out.with_name(out.name+'.incomplete')
tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(out)
print(json.dumps(receipt,sort_keys=True))
PY
