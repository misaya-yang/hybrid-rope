#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
source_root=${LONGBBOOK256_ROOT:-/root/autodl-tmp/infinitebench_longbook256_pool}
source_file=${INFINITEBENCH_LONGBOOK:-/root/autodl-tmp/InfiniteBench/data/longbook_qa_eng.jsonl}
root=${plan}/four_model_128k_extreme/qwen25_3b_256k

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_infinitebench_longbook_pool \
  --source "${source_file}" --out "${source_root}/longest32" --documents 32
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_128k_ppl \
  --model "${model}" --model-id qwen25_3b \
  --source-root "${source_root}/longest32" \
  --source-manifest "${source_root}/longest32/sources.json" \
  --dataset infinitebench_longbook \
  --out "${root}/ppl5_256k" --length 262144 --documents 5 --seed 20263501

"${python_bin}" - "${root}/ppl5_256k/manifest.json" "${root}/ppl5_256k/lm.npy" <<'PY'
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
print(json.dumps({'status':'QWEN25_3B_256K_PPL5_READY_V1','documents':5,
                  'dataset':'infinitebench_longbook',
                  'array_shape':list(a.shape),'lm_array_sha256':m['lm_array_sha256']},sort_keys=True))
PY
