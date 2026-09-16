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

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${out}"

# This validates code/table/model integration on the cheaper card.  It is not
# 128K evidence and writes outside every formal run directory.
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.pro6000_128k_preflight \
  --plan-root "${plan}" --qwen-model "${qwen_model}" --check-gpu \
  --minimum-vram-mib 20000 --allow-non-blackwell --out "${out}/environment.json"

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
  --model "${qwen_model}" --table "${qwen_root}/tables/tailspline.json" \
  --panel "${qwen_root}/assets/panels/65536/inputs.jsonl" \
  --length 65536 --chunks 8192 --max-new-tokens 1 --skip-lm \
  --minimum-free-fraction 0 --out "${out}/qwen_s4_64k_chunk8k.json"

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.benchmark_prefill_chunks \
  --model "${llama_model}" --table "${llama_root}/tables/tailspline.json" \
  --panel "${plan}/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl" \
  --length 32768 --chunks 8192 --max-new-tokens 1 --skip-lm \
  --minimum-free-fraction 0.05 --out "${out}/llama_s16_32k_chunk8k.json"

"${python_bin}" - "${out}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
reports={name:json.loads((root/f'{name}.json').read_text()) for name in (
    'qwen_s4_64k_chunk8k','llama_s16_32k_chunk8k')}
for name,report in reports.items():
    if report.get('status')!='PREFILL_CHUNK_BENCHMARK_COMPLETE_V2':
        raise ValueError(f'{name} benchmark incomplete')
    if report.get('recommended_generation_chunk')!=8192:
        raise ValueError(f'{name} did not pass the fixed chunked generation canary')
receipt={
    'status':'PRO6000_QUEUE_4080_CODEPATH_VALIDATED_V1',
    'scientific_result':False,
    'scope':'Cheaper-card model/table/chunked-generation integration only; not a 128K capacity or Blackwell validation.',
    'reports':{
        name:hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()
        for name in reports
    },
}
(root/'complete.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY
