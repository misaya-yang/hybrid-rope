#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
llama_model=${LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
qwen_model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
llama_root=${plan}/tailspline_llama_s16_128k_gate
qwen_root=${plan}/tailspline_qwen25_s4_64k128k_clean
queue_root=${plan}/pro6000_128k_queue
gpu_lock=/tmp/hybrid-rope-gpu0.lock

mkdir -p "${queue_root}/logs" "${qwen_root}/runtime"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# One owner for the entire expensive queue.  The generic Qwen runner receives
# a private nested lock because this process already owns the global GPU lock.
exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
  exit 73
fi

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.pro6000_128k_preflight \
  --plan-root "${plan}" --qwen-model "${qwen_model}" --llama-model "${llama_model}" --check-gpu \
  --minimum-vram-mib 80000 --out "${queue_root}/pro6000_preflight.json"

device_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | head -1 | tr -cd 'A-Za-z0-9_-')
if [[ -z "${device_uuid}" ]]; then
  printf 'REFUSE: cannot identify the Pro 6000 GPU UUID\n' >&2
  exit 1
fi

monitor_log=${queue_root}/logs/hardware_${device_uuid}.csv
(
  printf 'timestamp,utilization_gpu_pct,memory_used_mib,memory_total_mib,sm_clock_mhz,power_w,temperature_c\n'
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu \
      --format=csv,noheader,nounits || true
    sleep 2
  done
) >>"${monitor_log}" 2>&1 &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Phase 1 is the cheap, decisive Llama S=16 gate.  Its own canary independently
# chooses generation and LM prefill strategies on this exact GPU.
HYBRID_ROPE_GPU_LOCK_HELD=1 HYBRID_ROPE_REPO="${repo}" \
HYBRID_ROPE_PLAN_ROOT="${plan}" LLAMA_MODEL="${llama_model}" PYTHON_BIN="${python_bin}" \
bash experiments/iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh \
  >"${queue_root}/logs/llama_s16_128k.log" 2>&1

# Phase 2 keeps the frozen BF16 scientific path.  On a 96 GB card the runtime
# can use up to roughly 90 GB, and direct/64K/32K prefill compete by wall time.
qwen_runtime=${qwen_root}/runtime/prefill_128k_${device_uuid}.json
if [[ ! -f "${qwen_runtime}" ]]; then
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${qwen_model}" --table "${qwen_root}/tables/tailspline.json" \
    --panel "${qwen_root}/assets/panels/131072/inputs.jsonl" \
    --length 131072 --chunks 0,65536,32768 --skip-lm \
    --minimum-free-fraction 0.06 --out "${qwen_runtime}" \
    >"${queue_root}/logs/qwen_prefill_benchmark.log" 2>&1
fi
qwen_chunk=$("${python_bin}" - "${qwen_runtime}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
chunk=value.get('recommended_generation_chunk')
if value.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2' or chunk is None:
    raise SystemExit('no safe Qwen 128K prefill strategy passed')
print(int(chunk))
PY
)
qwen_batch_size=1
qwen_batch_report=${qwen_root}/runtime/batch_128k_${device_uuid}.json
if [[ "${qwen_chunk}" -eq 0 ]]; then
  if [[ ! -f "${qwen_batch_report}" ]]; then
    "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_generation_batch \
      --model "${qwen_model}" --table "${qwen_root}/tables/tailspline.json" \
      --panel "${qwen_root}/assets/panels/131072/inputs.jsonl" --length 131072 \
      --batch-sizes 2,4,8 \
      --minimum-free-fraction 0.06 --minimum-speedup 1.05 --out "${qwen_batch_report}" \
      >"${queue_root}/logs/qwen_batch_benchmark.log" 2>&1
  fi
  qwen_batch_size=$("${python_bin}" - "${qwen_batch_report}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));value=int(x.get('recommended_batch_size',1))
if x.get('status')!='GENERATION_BATCH_BENCHMARK_COMPLETE_V2' or value not in (1,2,4,8):
    raise SystemExit('invalid Qwen generation batch benchmark')
print(value)
PY
  )
fi

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_clean_matrix \
  --model "${qwen_model}" --model-id qwen25_3b \
  --data-root "${qwen_root}/assets" --out "${qwen_root}" --scale 4 \
  --lengths 65536 131072 --rows-per-task 50 \
  --data-manifest "${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json" \
  --python "${python_bin}" --batch-size "${qwen_batch_size}" \
  --prefill-chunk-size "${qwen_chunk}" \
  --gpu-lock "${queue_root}/qwen_nested_gpu.lock" --longest-first --execute \
  >"${queue_root}/logs/qwen_s4_64k128k.log" 2>&1

"${python_bin}" - "${plan}" "${device_uuid}" "${qwen_chunk}" "${qwen_batch_size}" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan=Path(sys.argv[1]);uuid=sys.argv[2];qwen_chunk=int(sys.argv[3]);qwen_batch=int(sys.argv[4])
llama=plan/'tailspline_llama_s16_128k_gate'
qwen=plan/'tailspline_qwen25_s4_64k128k_clean'
queue=plan/'pro6000_128k_queue'
llama_report=llama/'reports/tailspline_vs_mrpro_s16_128k_gate.json'
qwen_report=qwen/'reports/tailspline_vs_mrpro_65536_131072.json'
for arm in ('tailspline','mrpro'):
    ls=json.loads((llama/f'runs/{arm}/status.json').read_text())
    qs=json.loads((qwen/f'runs/{arm}/status.json').read_text())
    if ls!={'status':'COMPLETE','rows':130,'lm_rows':10}:
        raise ValueError(f'incomplete Llama arm: {arm}: {ls}')
    if qs!={'status':'COMPLETE','rows':1300,'lm_rows':0}:
        raise ValueError(f'incomplete Qwen arm: {arm}: {qs}')
for path in (llama_report,qwen_report):
    if not path.is_file(): raise FileNotFoundError(path)
receipt={
    'status':'PRO6000_128K_QUEUE_COMPLETE_V1',
    'gpu_uuid':uuid,
    'qwen_generation_prefill_chunk_size':qwen_chunk,
    'qwen_generation_batch_size':qwen_batch,
    'llama':{
        'scale':16,'lengths':[131072],'rows_per_arm':130,'lm_documents_per_arm':10,
        'report_sha256':hashlib.sha256(llama_report.read_bytes()).hexdigest(),
    },
    'qwen':{
        'scale':4,'lengths':[65536,131072],'rows_per_arm':1300,
        'report_sha256':hashlib.sha256(qwen_report.read_bytes()).hexdigest(),
    },
    'conditional_followups_started':False,
}
temporary=queue/'complete.json.incomplete'
temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
temporary.replace(queue/'complete.json')
print(json.dumps(receipt,sort_keys=True))
PY
