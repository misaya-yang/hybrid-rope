#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
out=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_full500/batch2_canary

cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${out}/status.json" ]] && /root/miniconda3/bin/python - "${out}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 2, "lm_rows": 0})
PY
then
  printf 'SKIP_COMPLETE llama32_batch2_canary\n'
  exit 0
fi

/root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${base_root}/assets/ppl46/manifest.json" \
  --model /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct \
  --arm Native \
  --extra-panel "${base_root}/assets/full13/rows.jsonl" \
  --only-extra-panels \
  --skip-lm \
  --length-cap 32768 \
  --task niah_single_1 \
  --limit-per-cell 2 \
  --batch-size 2 \
  --static-table-json "${base_root}/tables/tailspline.json" \
  --table-label llama3_8b_s4_32k_batch2_canary \
  --out "${out}" \
  --execute

printf 'BATCH2_CANARY_COMPLETE %s\n' "$(date -u +%FT%TZ)"
