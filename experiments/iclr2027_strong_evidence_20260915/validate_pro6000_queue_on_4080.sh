#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
llama_model=${LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
qwen_model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
qwen_root=${plan}/tailspline_qwen25_s4_64k128k_clean
llama_root=${plan}/tailspline_llama_s16_128k_gate
out=${plan}/pro6000_128k_queue/4080_validation
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p "${out}"
exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
  exit 73
fi

# This validates code/table/model integration on the existing 32 GB 4080 vGPU.
# A standard 16 GB physical 4080 is intentionally refused.  This is not 128K
# evidence and writes outside every formal run directory.
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.pro6000_128k_preflight \
  --plan-root "${plan}" --qwen-model "${qwen_model}" --llama-model "${llama_model}" --check-gpu \
  --minimum-vram-mib 30000 --allow-non-blackwell --out "${out}/environment.json"

qwen_canary=${out}/qwen32k_engineering_input.jsonl
"${python_bin}" - "${qwen_root}/assets/panels/65536/inputs.jsonl" "${qwen_canary}" <<'PY'
import hashlib,json,sys
source,out=sys.argv[1:]
row=json.loads(next(line for line in open(source) if line.strip()))
budget=max(8,int(row.get('max_new_tokens',8)))
prompt=list(row['prompt_ids'])[-(32768-budget):]
row.update(
    row_id='engineering_qwen32k_'+str(row.get('row_id','row0')),
    length_cap=32768,prompt_ids=prompt,input_tokens=len(prompt),actual_length=len(prompt),
    max_new_tokens=budget,
    prompt_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),
)
with open(out,'w') as stream:stream.write(json.dumps(row)+'\n')
PY

for arm in tailspline mrpro; do
  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${qwen_model}" --table "${qwen_root}/tables/${arm}.json" \
    --panel "${qwen_canary}" --length 32768 --chunks 0,8192 \
    --max-new-tokens 8 --skip-lm --minimum-free-fraction 0.05 \
    --out "${out}/qwen_s4_32k_${arm}.json"

  "${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
    --model "${llama_model}" --table "${llama_root}/tables/${arm}.json" \
    --panel "${plan}/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl" \
    --lm-array "${plan}/tailspline_llama_s4_classic/assets/ppl46/lm.npy" \
    --length 32768 --chunks 0,8192 --max-new-tokens 8 \
    --minimum-free-fraction 0.05 --out "${out}/llama_s16_32k_${arm}.json"
done

"${python_bin}" - "${out}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
reports={name:json.loads((root/f'{name}.json').read_text()) for name in (
    'qwen_s4_32k_tailspline','qwen_s4_32k_mrpro',
    'llama_s16_32k_tailspline','llama_s16_32k_mrpro')}
for name,report in reports.items():
    if report.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2':
        raise ValueError(f'{name} benchmark incomplete')
    chunks={int(item['chunk']):item for item in report.get('results',[])}
    if set(chunks)!={0,8192}:
        raise ValueError(f'{name} did not execute direct and chunked paths')
    for chunk,item in chunks.items():
        generation=item.get('generation',{})
        if generation.get('status')!='ok' or generation.get('stable') is not True:
            raise ValueError(f'{name}/{chunk} generation failed exact parity or memory checks')
        if len(generation.get('generated_ids',[]))<2:
            raise ValueError(f'{name}/{chunk} did not exercise cached decode beyond the first token')
        if name.startswith('llama_'):
            lm=item.get('lm',{})
            if lm.get('status')!='ok' or lm.get('finite') is not True:
                raise ValueError(f'{name}/{chunk} LM did not execute with finite losses')
    if name.startswith('llama_'):
        stable_lm=[item['lm'] for item in chunks.values() if item['lm'].get('stable') is True]
        if not stable_lm or report.get('recommended_lm_chunk') is None:
            raise ValueError(f'{name} has no LM strategy within the frozen numerical/memory tolerance')
receipt={
    'status':'PRO6000_QUEUE_32GB_CODEPATH_VALIDATED_V2',
    'scientific_result':False,
    'scope':'32GB-vGPU two-model/two-table multi-token generation and Llama LM integration; not a 128K capacity or Blackwell validation.',
    'reports':{
        name:hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()
        for name in reports
    },
}
(root/'complete.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY
