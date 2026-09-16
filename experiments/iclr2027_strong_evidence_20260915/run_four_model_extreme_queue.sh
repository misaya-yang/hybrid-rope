#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
suite=${plan}/four_model_128k_extreme
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

if [[ "${HYBRID_ROPE_GPU_LOCK_HELD:-0}" != 1 ]]; then
  exec 9>"${gpu_lock}"
  if ! flock -n 9; then
    printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
    exit 73
  fi
fi

if [[ ! -f "${suite}/ready.json" ]]; then
  printf 'REFUSE: prepare_four_model_extreme_queue.sh has not completed\n' >&2
  exit 1
fi

gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
tasks=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery)
conditions=(
  "qwen25_3b_64k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|32768|2|65536|2,4,8"
  "qwen25_3b_128k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|qwen25_3b|32768|4|131072|2,4,8"
  "qwen25_1p5b_128k|/root/autodl-tmp/qwen25_1p5b_32k|qwen25_1p5b|32768|4|131072|2,4,8"
  "olmo2_1b_64k|/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct|olmo2_1b|4096|16|65536|2,4"
)

run_condition() {
  local spec=$1 name model model_id native scale target batch_candidates root chunks runtime gen_chunk lm_chunk batch report
  IFS='|' read -r name model model_id native scale target batch_candidates <<<"${spec}"
  root=${suite}/${name}
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.validate_extreme_condition \
    --root "${root}" --model "${model}" --model-id "${model_id}" \
    --target "${target}" --scale "${scale}" --rows-per-task 5 --ppl-documents 5 \
    >"${root}/logs/validate_gpu_entry.log" 2>&1
  if [[ "${target}" -eq 131072 ]]; then chunks=0,65536,32768; else chunks=0,32768,16384; fi
  runtime=${root}/runtime/prefill_${target}_${gpu_uuid}.json
  mkdir -p "${root}/runtime" "${root}/runs" "${root}/reports"
  if [[ ! -f "${runtime}" ]]; then
    "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
      --model "${model}" --table "${root}/tables/tailspline.json" \
      --panel "${root}/assets/panels/${target}/inputs.jsonl" --lm-array "${root}/ppl/lm.npy" \
      --length "${target}" --chunks "${chunks}" --minimum-free-fraction 0.06 \
      --out "${runtime}" >"${root}/logs/prefill_benchmark.log" 2>&1
  fi
  read -r gen_chunk lm_chunk < <("${python_bin}" - "${runtime}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));g=x.get('recommended_generation_chunk');m=x.get('recommended_lm_chunk')
if x.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2' or g is None or m is None: raise SystemExit('no safe prefill strategy')
results={int(item['chunk']):item for item in x.get('results',[])}
direct=results.get(0,{}).get('generation',{});chosen=results.get(int(g),{}).get('generation',{})
if int(g) and direct.get('stable') and chosen.get('stable') and float(chosen['seconds']) >= .98*float(direct['seconds']): g=0
print(int(g),int(m))
PY
  )
  batch=1
  report=${root}/runtime/batch_${target}_${gpu_uuid}.json
  if [[ "${gen_chunk}" -eq 0 ]]; then
    if [[ ! -f "${report}" ]]; then
      "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch \
        --model "${model}" --table "${root}/tables/tailspline.json" \
        --panel "${root}/assets/panels/${target}/inputs.jsonl" --length "${target}" \
        --batch-sizes "${batch_candidates}" --minimum-free-fraction 0.06 --minimum-speedup 1.02 \
        --out "${report}" >"${root}/logs/batch_benchmark.log" 2>&1
    fi
    batch=$("${python_bin}" - "${report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));print(int(x.get('recommended_batch_size',1)))
PY
    )
  fi
  for arm in tailspline mrpro; do
    run=${root}/runs/${arm}
    if [[ -f "${run}/status.json" ]]; then
      if "${python_bin}" - "${run}/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={'status':'COMPLETE','rows':40,'lm_rows':5} else 1)
PY
      then continue; fi
      printf 'REFUSE: incomplete status exists for %s/%s\n' "${name}" "${arm}" >&2; exit 1
    fi
    command=("${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval
      --data "${root}/ppl/manifest.json" --model "${model}" --arm Native
      --extra-panel "${root}/assets/panels/${target}/inputs.jsonl" --only-extra-panels
      --length-cap "${target}" --lm-length-cap "${target}" --lm-limit-documents 5
      --limit-per-cell 5 --prefill-chunk-size "${gen_chunk}" --lm-prefill-chunk-size "${lm_chunk}"
      --batch-size "${batch}" --longest-first --static-table-json "${root}/tables/${arm}.json"
      --table-label "extreme_${model_id}_s${scale}_${arm}" --out "${run}" --execute)
    for task in "${tasks[@]}"; do command+=(--task "${task}"); done
    "${command[@]}" >"${root}/logs/${arm}.log" 2>&1
  done
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.extreme_condition_report \
    --root "${root}" --model-id "${model_id}" --native "${native}" --target "${target}" \
    --scale "${scale}" --rows-per-task 5 --ppl-documents 5 \
    --out "${root}/reports/tailspline_vs_mrpro.json"
}

for spec in "${conditions[@]}"; do run_condition "${spec}"; done

"${python_bin}" - "${suite}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);names=('qwen25_3b_64k','qwen25_3b_128k','qwen25_1p5b_128k','olmo2_1b_64k')
reports={}
for name in names:
 p=root/name/'reports/tailspline_vs_mrpro.json';d=json.loads(p.read_text())
 if d.get('status')!='EXTREME_NIAH_PPL_REPORT_V1': raise ValueError(name)
 reports[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
receipt={'status':'FOUR_MODEL_EXTREME_GPU_QUEUE_COMPLETE_V1','reports':reports,
         'llama_reuses':'tailspline_llama_s16_128k_gate/reports/tailspline_vs_mrpro_s16_128k_gate.json'}
tmp=root/'complete.json.incomplete';tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(root/'complete.json')
print(json.dumps(receipt,sort_keys=True))
PY
