#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
upstream_pid=${UPSTREAM_PID:?set UPSTREAM_PID to the Pro6000 heavy queue}
root=${plan}/glm4_9b_s4_128k
assets=${root}/en_qa_second_books_assets
evaluation=${root}/en_qa_second_books_evaluation

while [[ -d "/proc/${upstream_pid}" ]]; do
  state=$(ps -o stat= -p "${upstream_pid}" 2>/dev/null | tr -d ' ' || true)
  command=$(ps -o cmd= -p "${upstream_pid}" 2>/dev/null || true)
  [[ -z "${state}" || "${state}" == Z* ]] && break
  [[ "${command}" == *run_pro6000_heavy_yarn_queue.sh* ]] || { echo "REFUSE: upstream PID reused" >&2; exit 74; }
  sleep 30
done
test -f "${plan}/official_yarn_pro6000_heavy/complete.json" || { echo "REFUSE: heavy queue incomplete" >&2; exit 1; }

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${evaluation}/status.json" ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_natural_long \
    --model /root/models/GLM-4-9B-0414 --model-id glm4_9b_0414 \
    --data-root "${assets}" --data-manifest "${assets}/manifest.json" \
    --out "${evaluation}" --scale 4 --lengths 131072 --rows-per-task 100 \
    --benchmark infinitebench --prefill-chunk-size 65536 --python "${python_bin}" --execute
fi

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa \
  --condition glm4_9b_second_books --plan-root "${plan}" --python "${python_bin}" --execute

test -f "${root}/en_qa_second_books_yarn_a1/complete.json" || {
  echo "REFUSE: second-book GLM QA incomplete" >&2
  exit 1
}
