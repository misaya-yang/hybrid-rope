#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${gpu_lock}"; flock -n 9 || { echo "REFUSE: GPU lock is owned" >&2; exit 73; }

# The strict orchestrator keeps Qwen2.5-1.5B excluded, reuses Llama/OLMo
# T/P, resumes Qwen T/P, and adds only GLM assets10 rows 5..9 to T/P.
# Before reporting, it validates row_id, prompt hash, table, gain and length.
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.four_model_yarn_full13 \
  --plan "${plan}" --python "${python_bin}" --gpu-lock "${gpu_lock}" --execute
