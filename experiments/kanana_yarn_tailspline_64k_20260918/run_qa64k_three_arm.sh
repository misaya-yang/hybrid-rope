#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
qa=${root}/qa64k

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY model=%s target=65536 benchmark=InfiniteBench-En.QA arms=tailspline,official_yarn,mrpro\n' "${model}"
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${KANANA_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${KANANA_MKL_NUM_THREADS:-8}
mkdir -p "${qa}/runs" "${qa}/logs" "${qa}/reports"

for required in "${qa}/assets/manifest.json" "${qa}/assets/inputs.jsonl" \
  "${root}/tables/tailspline.json" "${root}/tables/official_yarn.json" "${root}/tables/mrpro.json"; do
  [[ -s ${required} ]] || { printf 'REFUSE: missing %s\n' "${required}" >&2; exit 1; }
done
expected=$("${python_bin}" - "${qa}/assets/inputs.jsonl" <<'PY'
import sys
print(sum(bool(line.strip()) for line in open(sys.argv[1])))
PY
)
[[ ${expected} -gt 0 ]] || { printf 'REFUSE: QA panel is empty\n' >&2; exit 1; }

exec 8>"${gpu_lock}"
flock 8

for arm in tailspline official_yarn mrpro; do
  out=${qa}/runs/${arm}
  complete=0
  if [[ -f ${out}/status.json ]]; then
    complete=$("${python_bin}" - "${out}/status.json" "${expected}" <<'PY'
import json,sys
print(1 if json.load(open(sys.argv[1])) == {"status":"COMPLETE","rows":int(sys.argv[2]),"lm_rows":0} else 0)
PY
    )
  fi
  if [[ ${complete} != 1 ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${root}/minimal_eval_manifest.json" --model "${model}" --arm Native \
      --extra-panel "${qa}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
      --length-cap 65536 --batch-size 1 --unmasked-unpadded-generate \
      --static-table-json "${root}/tables/${arm}.json" \
      --table-label "kanana_64k_qa_${arm}" --out "${out}" --execute \
      >"${qa}/logs/${arm}.log" 2>&1
  fi
done

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report_qa_three_arm \
  --panel "${qa}/assets/inputs.jsonl" --target-length 65536 \
  --tailspline "${qa}/runs/tailspline/generations.jsonl" \
  --official-yarn "${qa}/runs/official_yarn/generations.jsonl" \
  --mrpro "${qa}/runs/mrpro/generations.jsonl" \
  --out "${qa}/reports/three_arm.json"
printf 'KANANA_64K_QA_TRIARM_COMPLETE %s\n' "$(date -u +%FT%TZ)"
