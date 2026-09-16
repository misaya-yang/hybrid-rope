#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
root=${plan}/four_model_128k_extreme/qwen25_3b_256k
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
target=262144
scale=8
tasks=(niah_single_1 niah_single_2 niah_single_3)
panel=${root}/assets/panels/${target}/inputs_single.jsonl
panel_manifest=${root}/assets/panels/${target}/single_manifest.json
ppl_root=${root}/ppl5_256k

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
  exit 73
fi

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.validate_extreme_condition \
  --root "${root}" --model "${model}" --model-id qwen25_3b \
  --target "${target}" --scale "${scale}" --rows-per-task 5 --ppl-documents 2 \
  >"${root}/logs/validate_gpu_entry.log" 2>&1
"${python_bin}" - "${panel}" "${panel_manifest}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.load(open(sys.argv[2]))
rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
tasks=['niah_single_1','niah_single_2','niah_single_3']
if (m.get('status')!='QWEN25_3B_256K_SINGLE_NIAH_ASSETS_READY_V1' or m.get('tasks')!=tasks
    or m.get('rows')!=15 or len(rows)!=15 or {row.get('task') for row in rows}!=set(tasks)
    or hashlib.sha256(p.read_bytes()).hexdigest()!=m.get('inputs_sha256')):
    raise SystemExit('256K single-needle panel drift')
PY
"${python_bin}" - "${ppl_root}/manifest.json" "${ppl_root}/lm.npy" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
m=json.load(open(sys.argv[1]));p=Path(sys.argv[2]);a=np.load(p,mmap_mode='r',allow_pickle=False)
if (m.get('status')!='COMPLETE' or m.get('contract')!='MODEL_TOKENIZED_LONG_CONTEXT_PPL_V1'
    or m.get('datasets')!=['infinitebench_longbook']
    or m.get('documents')!=5 or m.get('lengths')!=[262144]
    or list(a.shape)!=[5,262145] or str(a.dtype)!='int64'
    or hashlib.sha256(p.read_bytes()).hexdigest()!=m.get('lm_array_sha256')):
    raise SystemExit('InfiniteBench LongBook 256K PPL5 drift')
PY

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
mkdir -p "${root}/runtime" "${root}/runs" "${root}/reports"
runtime=${root}/runtime/prefill_${target}_${gpu_uuid}.json
if [[ ! -f "${runtime}" ]]; then
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${model}" --table "${root}/tables/tailspline.json" \
    --panel "${panel}" --lm-array "${ppl_root}/lm.npy" \
    --length "${target}" --chunks 0,131072,65536 --minimum-free-fraction 0.06 \
    --out "${runtime}" >"${root}/logs/prefill_benchmark_256k.log" 2>&1
fi
read -r gen_chunk lm_chunk < <("${python_bin}" - "${runtime}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));g=x.get('recommended_generation_chunk');m=x.get('recommended_lm_chunk')
if x.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2' or g is None or m is None: raise SystemExit('no safe 256K prefill strategy')
results={int(item['chunk']):item for item in x.get('results',[])}
direct=results.get(0,{}).get('generation',{});chosen=results.get(int(g),{}).get('generation',{})
if int(g) and direct.get('stable') and chosen.get('stable') and float(chosen['seconds']) >= .98*float(direct['seconds']): g=0
print(int(g),int(m))
PY
)
batch=1
batch_report=${root}/runtime/batch_${target}_${gpu_uuid}.json
if [[ "${gen_chunk}" -eq 0 ]]; then
  if [[ ! -f "${batch_report}" ]]; then
    "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch \
      --model "${model}" --table "${root}/tables/tailspline.json" \
      --panel "${panel}" --length "${target}" \
      --batch-sizes 2 --minimum-free-fraction 0.06 --minimum-speedup 1.02 \
      --out "${batch_report}" >"${root}/logs/batch_benchmark_256k.log" 2>&1
  fi
  batch=$("${python_bin}" - "${batch_report}" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1])).get('recommended_batch_size',1)))
PY
  )
fi

for arm in tailspline mrpro; do
  run=${root}/runs/${arm}
  if [[ -f "${run}/status.json" ]]; then
    "${python_bin}" - "${run}/status.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))!={'status':'COMPLETE','rows':15,'lm_rows':5}: raise SystemExit('incomplete 256K run')
PY
    continue
  fi
  command=("${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval
    --data "${ppl_root}/manifest.json" --model "${model}" --arm Native
    --extra-panel "${panel}" --only-extra-panels
    --length-cap "${target}" --lm-length-cap "${target}" --lm-limit-documents 5
    --limit-per-cell 5 --prefill-chunk-size "${gen_chunk}" --lm-prefill-chunk-size "${lm_chunk}"
    --batch-size "${batch}" --longest-first --static-table-json "${root}/tables/${arm}.json"
    --table-label "extreme_qwen25_3b_s8_${arm}" --out "${run}" --execute)
  for task in "${tasks[@]}"; do command+=(--task "${task}"); done
  "${command[@]}" >"${root}/logs/${arm}_256k.log" 2>&1
done

report=${root}/reports/tailspline_vs_mrpro_256k_ppl_niah.json
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.extreme_condition_report \
  --root "${root}" --model-id qwen25_3b --native 32768 --target "${target}" --scale "${scale}" \
  --rows-per-task 5 --ppl-documents 5 --ppl-dataset-label infinitebench_longbook \
  --task niah_single_1 --task niah_single_2 --task niah_single_3 \
  --out "${report}"
"${python_bin}" - "${report}" "${root}/complete_256k.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);out=Path(sys.argv[2]);d=json.loads(p.read_text())
receipt={'status':'QWEN25_3B_256K_PPL_NIAH_COMPLETE_V1','report':str(p),
         'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'rows_per_arm':15,'ppl_documents_per_arm':5,'ppl_dataset':'infinitebench_longbook',
         'scale':8,'target_length':262144}
tmp=out.with_name(out.name+'.incomplete');tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(out)
print(json.dumps(receipt,sort_keys=True))
PY
