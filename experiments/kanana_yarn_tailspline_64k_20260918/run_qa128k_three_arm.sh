#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
qa=${root}/qa128k
skip_canary=${KANANA_SKIP_CANARY:-0}
parallel_arms=${KANANA_PARALLEL_ARMS:-auto}

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY target=131072 rows=118 clusters=23 arms=tailspline,official_yarn,mrpro canary=longest\n'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${KANANA_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${KANANA_MKL_NUM_THREADS:-8}
mkdir -p "${qa}/runs" "${qa}/logs" "${qa}/reports" "${qa}/canary"

for required in "${qa}/assets/manifest.json" "${qa}/assets/inputs.jsonl" \
  "${qa}/tables/tailspline.json" "${qa}/tables/official_yarn.json" "${qa}/tables/mrpro.json"; do
  [[ -s ${required} ]] || { printf 'REFUSE: missing %s\n' "${required}" >&2; exit 1; }
done
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
[[ ${total_mib} -ge 45000 ]] || { printf 'REFUSE: QA128K requires a 48GB-class GPU; found %s MiB\n' "${total_mib}" >&2; exit 75; }
if [[ ${parallel_arms} == auto ]]; then
  if [[ ${total_mib} -ge 90000 ]]; then parallel_arms=1; else parallel_arms=0; fi
fi
[[ ${parallel_arms} == 0 || ${parallel_arms} == 1 ]] || {
  printf 'REFUSE: KANANA_PARALLEL_ARMS must be 0, 1 or auto\n' >&2
  exit 1
}

"${python_bin}" - "${qa}/assets/inputs.jsonl" "${qa}/canary/longest.jsonl" <<'PY'
import json,os,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
longest=max(rows,key=lambda row:int(row["input_tokens"])); path=sys.argv[2]
payload=json.dumps(longest,sort_keys=True)+"\n"
if os.path.exists(path) and open(path).read()!=payload: raise SystemExit("canary drift")
if not os.path.exists(path): open(path,"w").write(payload)
print(json.dumps({"row_id":longest["row_id"],"input_tokens":longest["input_tokens"]},sort_keys=True))
PY

exec 8>"${gpu_lock}"
flock 8

if [[ ${skip_canary} != 1 ]]; then
  # Optional longest-row memory canary. It is separate from the frozen full
  # runs and is never included in the benchmark score.
  canary_out=${qa}/canary/tailspline
  if [[ ! -f ${canary_out}/status.json ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${root}/minimal_eval_manifest.json" --model "${model}" --arm Native \
      --extra-panel "${qa}/canary/longest.jsonl" --only-extra-panels --skip-lm \
      --length-cap 131072 --batch-size 1 --unmasked-unpadded-generate \
      --static-table-json "${qa}/tables/tailspline.json" \
      --table-label kanana_128k_qa_tailspline_canary \
      --out "${canary_out}" --execute >"${qa}/logs/canary.log" 2>&1
  fi
  "${python_bin}" - "${canary_out}/generations.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
if len(rows)!=1 or rows[0].get("empty") or not rows[0].get("output_text","").strip():
    raise SystemExit("QA128K longest-row canary is unhealthy")
print(json.dumps({k:rows[0].get(k) for k in ("row_id","ended_eos","hit_cap","output_text")},ensure_ascii=False))
PY
fi

run_arm() {
  local arm=$1
  out=${qa}/runs/${arm}
  complete=0
  if [[ -f ${out}/status.json ]]; then
    complete=$("${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
print(1 if json.load(open(sys.argv[1])) == {"status":"COMPLETE","rows":118,"lm_rows":0} else 0)
PY
    )
  fi
  if [[ ${complete} != 1 ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${root}/minimal_eval_manifest.json" --model "${model}" --arm Native \
      --extra-panel "${qa}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
      --length-cap 131072 --batch-size 1 --unmasked-unpadded-generate \
      --static-table-json "${qa}/tables/${arm}.json" \
      --table-label "kanana_128k_qa_${arm}" --out "${out}" --execute \
      >"${qa}/logs/${arm}.log" 2>&1
  fi
}

if [[ ${parallel_arms} == 1 ]]; then
  run_arm tailspline & first=$!
  run_arm official_yarn & second=$!
  wait "${first}"
  wait "${second}"
  run_arm mrpro
else
  run_arm tailspline
  run_arm official_yarn
  run_arm mrpro
fi

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report_qa_three_arm \
  --panel "${qa}/assets/inputs.jsonl" --target-length 131072 \
  --tailspline "${qa}/runs/tailspline/generations.jsonl" \
  --official-yarn "${qa}/runs/official_yarn/generations.jsonl" \
  --mrpro "${qa}/runs/mrpro/generations.jsonl" \
  --out "${qa}/reports/three_arm.json"
printf 'KANANA_128K_QA_TRIARM_COMPLETE %s\n' "$(date -u +%FT%TZ)"
