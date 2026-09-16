#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
root=${plan}/tailspline_llama_s16_128k_gate
model=${LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

if [[ "${HYBRID_ROPE_GPU_LOCK_HELD:-0}" != 1 ]]; then
  exec 8>"${gpu_lock}"
  if ! flock -n 8; then
    printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
    exit 73
  fi
fi

mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${total_mib}" -lt 45000 ]]; then
  printf 'REFUSE: Llama S16 128K gate requires >=45,000 MiB VRAM; found %s MiB\n' "${total_mib}" >&2
  exit 75
fi
if [[ ! -f "${root}/assets/ready.json" ]]; then
  printf 'REFUSE: run prepare_llama_s16_128k_assets.sh before GPU execution\n' >&2
  exit 1
fi

device_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
if [[ -z "${device_uuid}" ]]; then
  printf 'REFUSE: cannot identify the destination GPU UUID\n' >&2
  exit 1
fi
runtime_report=${root}/runtime/prefill_128k_${device_uuid}.json
if [[ ! -f "${runtime_report}" ]]; then
  mkdir -p "${root}/runtime"
  chunks=0,8192,16384,32768
  minimum_free_fraction=0.10
  if [[ "${total_mib}" -ge 90000 ]]; then
    chunks=0,65536,32768
    # A 96 GB Pro 6000 may use roughly 90 GB at peak.  Selection remains
    # throughput-based; this only avoids rejecting the fastest safe candidate
    # because of the more conservative 48 GB headroom policy.
    minimum_free_fraction=0.06
  fi
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${model}" --table "${root}/tables/tailspline.json" \
    --panel "${root}/assets/full13/inputs.jsonl" --lm-array "${root}/assets/ppl10/lm.npy" \
    --length 131072 --chunks "${chunks}" \
    --minimum-free-fraction "${minimum_free_fraction}" --out "${runtime_report}" \
    >"${root}/logs/prefill_benchmark.log" 2>&1
fi
read -r selected_generation_chunk selected_lm_chunk < <("${python_bin}" - "${runtime_report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
g=x.get('recommended_generation_chunk');m=x.get('recommended_lm_chunk')
if g is None or m is None: raise SystemExit('no safe prefill strategy passed the canary')
print(g,m)
PY
)
generation_prefill_chunk_size=${GENERATION_PREFILL_CHUNK_SIZE:-${selected_generation_chunk}}
lm_prefill_chunk_size=${LM_PREFILL_CHUNK_SIZE:-${selected_lm_chunk}}
# On a >=90 GB GPU, a sub-2% chunking win is timing noise compared with the
# opportunity to test direct-prefill batching.  Chunked generation remains the
# choice when it produces a material win or direct prefill is not stable.
if [[ "${total_mib}" -ge 90000 && -z "${GENERATION_PREFILL_CHUNK_SIZE+x}" ]]; then
  generation_prefill_chunk_size=$("${python_bin}" - "${runtime_report}" "${generation_prefill_chunk_size}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));selected=int(sys.argv[2])
results={int(item['chunk']):item for item in x.get('results',[])}
direct=results.get(0,{}).get('generation',{})
chosen=results.get(selected,{}).get('generation',{})
if (selected and direct.get('stable') and chosen.get('stable')
        and float(chosen['seconds']) >= 0.98*float(direct['seconds'])):
    selected=0
print(selected)
PY
  )
fi
generation_batch_size=${GENERATION_BATCH_SIZE:-1}
batch_report=${root}/runtime/batch_128k_${device_uuid}.json
if [[ "${total_mib}" -ge 90000 && "${generation_prefill_chunk_size}" -eq 0 \
      && -z "${GENERATION_BATCH_SIZE+x}" ]]; then
  if [[ ! -f "${batch_report}" ]]; then
    "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch \
      --model "${model}" --table "${root}/tables/tailspline.json" \
      --panel "${root}/assets/full13/inputs.jsonl" --length 131072 \
      --batch-sizes 2,4 \
      --minimum-free-fraction 0.06 --minimum-speedup 1.05 --out "${batch_report}" \
      >"${root}/logs/batch_benchmark.log" 2>&1
  fi
  generation_batch_size=$("${python_bin}" - "${batch_report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));value=int(x.get('recommended_batch_size',1))
if x.get('status')!='GENERATION_BATCH_BENCHMARK_COMPLETE_V2' or value not in (1,2,4):
    raise SystemExit('invalid generation batch benchmark')
print(value)
PY
  )
fi

run_arm() {
  local arm=$1 run=${root}/runs/$1
  if [[ -f "${run}/status.json" ]]; then
    if "${python_bin}" - "${run}/status.json" "${run}/contract.json" \
      "${root}/tables/${arm}.json" "${generation_prefill_chunk_size}" \
      "${lm_prefill_chunk_size}" "${generation_batch_size}" <<'PY'
import json,sys
status=json.load(open(sys.argv[1]));contract=json.load(open(sys.argv[2]));receipt=json.load(open(sys.argv[3]))
table=receipt.get('table',receipt);active=contract.get('static_table') or {}
valid=(status=={"status":"COMPLETE","rows":130,"lm_rows":10}
       and contract.get('prefill_chunk_size')==int(sys.argv[4])
       and contract.get('lm_prefill_chunk_size')==int(sys.argv[5])
       and contract.get('generation_prefill_strategy') in ('direct_generate_v1','dynamic_cache_lower_right_v1')
       and contract.get('lm_execution_strategy') in ('direct_no_cache_v1','dynamic_cache_exact_nll_v1')
       and isinstance(contract.get('runtime_versions'),dict)
       and contract.get('batch_size')==int(sys.argv[6])
       and contract.get('generation_length_caps')==[131072]
       and contract.get('lm_lengths')==[131072]
       and active.get('values_float32')==table.get('values_float32')
       and active.get('gain')==table.get('gain'))
raise SystemExit(0 if valid else 1)
PY
    then return; fi
    printf 'REFUSE: completed %s arm does not match requested runtime/table contract\n' "${arm}" >&2
    exit 1
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${root}/assets/ppl10/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${root}/assets/full13/inputs.jsonl" --only-extra-panels \
    --length-cap 131072 --lm-length-cap 131072 \
    --prefill-chunk-size "${generation_prefill_chunk_size}" \
    --lm-prefill-chunk-size "${lm_prefill_chunk_size}" \
    --batch-size "${generation_batch_size}" \
    --static-table-json "${root}/tables/${arm}.json" \
    --table-label "llama3_8b_s16_128k_${arm}" --out "${run}" --execute \
    >"${root}/logs/${arm}.log" 2>&1
}
run_arm tailspline
run_arm mrpro

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.llama_s16_128k_report \
  --run "tailspline=${root}/runs/tailspline" --run "mrpro=${root}/runs/mrpro" \
  --out "${root}/reports/tailspline_vs_mrpro_s16_128k_gate.json"
printf 'LLAMA_S16_128K_GATE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
