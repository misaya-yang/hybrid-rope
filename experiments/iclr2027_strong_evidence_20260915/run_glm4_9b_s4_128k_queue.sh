#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${GLM_MODEL:-/root/models/GLM-4-9B-0414}
root=${plan}/glm4_9b_s4_128k
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${root}/runtime" "${root}/runs" "${root}/reports" "${root}/logs"

exec 9>"${gpu_lock}"
flock -n 9 || { echo "REFUSE: GPU lock is owned" >&2; exit 73; }

"${python_bin}" - "${model}" "${root}" <<'PY'
import json,sys
from pathlib import Path
model,root=map(Path,sys.argv[1:])
if json.loads((model/'DOWNLOAD_RECEIPT.json').read_text()).get('status')!='DOWNLOAD_COMPLETE_VERIFIED': raise SystemExit('GLM download incomplete')
a=json.loads((root/'assets/manifest.json').read_text());q=json.loads((root/'en_qa_assets/manifest.json').read_text());p=json.loads((root/'ppl5/manifest.json').read_text())
if a.get('rows')!=65 or a.get('rows_per_task')!=5 or a.get('lengths')!=[131072]: raise SystemExit('RULER assets drift')
if q.get('status')!='COMPLETE' or q.get('scale')!=4 or q.get('lengths')!=[131072] or q.get('summary',{}).get('selected_rows',0)<=0: raise SystemExit('En.QA assets drift')
if p.get('status')!='COMPLETE' or p.get('documents')!=5 or p.get('lengths')!=[131072]: raise SystemExit('PPL assets drift')
for arm in ('tailspline','mrpro'):
 d=json.loads((root/f'tables/{arm}.json').read_text())
 if d.get('model_geometry',{}).get('pairs')!=32 or d.get('band_envelope')!=[17,30] or d.get('scale')!=4: raise SystemExit(f'{arm} table drift')
PY

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
runtime=${root}/runtime/prefill_131072_${gpu_uuid}.json
if [[ ! -f "${runtime}" ]]; then
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${model}" --table "${root}/tables/tailspline.json" \
    --panel "${root}/assets/panels/131072/inputs.jsonl" --lm-array "${root}/ppl5/lm.npy" \
    --length 131072 --chunks 0,65536,32768 --minimum-free-fraction 0.06 --out "${runtime}" \
    >"${root}/logs/prefill_benchmark.log" 2>&1
fi
read -r gen_chunk lm_chunk < <("${python_bin}" - "${runtime}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));g=d.get('recommended_generation_chunk');m=d.get('recommended_lm_chunk')
if d.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2' or g is None or m is None: raise SystemExit('no safe GLM prefill strategy')
print(int(g),int(m))
PY
)

for arm in tailspline mrpro; do
  run=${root}/runs/${arm}
  if [[ ! -f "${run}/status.json" ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${root}/ppl5/manifest.json" --model "${model}" --arm Native \
      --extra-panel "${root}/assets/panels/131072/inputs.jsonl" --only-extra-panels \
      --length-cap 131072 --lm-length-cap 131072 --lm-limit-documents 5 \
      --prefill-chunk-size "${gen_chunk}" --lm-prefill-chunk-size "${lm_chunk}" \
      --batch-size 1 --longest-first --static-table-json "${root}/tables/${arm}.json" \
      --table-label "glm4_9b_s4_${arm}" --out "${run}" --execute \
      >"${root}/logs/${arm}.log" 2>&1
  fi
done

"${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
  --source "tailspline=${root}/runs/tailspline/generations.jsonl" \
  --source "mrpro=${root}/runs/mrpro/generations.jsonl" \
  --candidate tailspline --baseline mrpro --length 131072 \
  --out "${root}/reports/tailspline_vs_mrpro_full13.json"

if [[ ! -f "${root}/en_qa_evaluation/status.json" ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_natural_long \
    --model "${model}" --model-id glm4_9b_0414 --data-root "${root}/en_qa_assets" \
    --out "${root}/en_qa_evaluation" --scale 4 --lengths 131072 --rows-per-task 50 \
    --data-manifest "${root}/en_qa_assets/manifest.json" --python "${python_bin}" \
    --benchmark infinitebench --prefill-chunk-size "${gen_chunk}" --execute \
    >"${root}/logs/en_qa.log" 2>&1
fi

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);owners=[]
for p in (root/'reports/tailspline_vs_mrpro_full13.json',root/'en_qa_evaluation/report.json'):
 if not p.is_file(): raise FileNotFoundError(p)
 owners.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'status':'GLM4_9B_S4_128K_QUEUE_COMPLETE_V1','owners':owners,'ppl_rows_per_arm':5}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
