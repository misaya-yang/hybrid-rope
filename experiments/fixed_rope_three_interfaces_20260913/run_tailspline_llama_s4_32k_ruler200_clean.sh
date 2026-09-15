#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200_clean
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=${root}/assets/inputs.jsonl

mkdir -p "${root}/assets" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_32k_ruler200_clean \
  --source-parts /root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_full500/assets/full13_32k_extra490_parallel/parts \
  --qa1-tail10 "${root}/qa1_tail10_source" \
  --model "${model}" \
  --out "${root}/assets"

for arm in tailspline mrpro; do
  run_dir=${root}/runs/${arm}
  if [[ -f "${run_dir}/status.json" ]] && /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 2600, "lm_rows": 0})
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    continue
  fi
  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${base_root}/assets/ppl46/manifest.json" \
    --model "${model}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${base_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_32k_ruler200_clean_${arm}" \
    --out "${run_dir}" \
    --execute >"${root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
done

/root/miniconda3/bin/python - "${panel}" "${root}" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

panel_path, root = Path(sys.argv[1]), Path(sys.argv[2])
panel = [json.loads(line) for line in panel_path.read_text().splitlines() if line]
expected = {row["prompt_sha256"] for row in panel}
if len(panel) != 2600 or Counter(row["task"] for row in panel) != Counter({task: 200 for task in set(row["task"] for row in panel)}):
    raise ValueError("clean RULER-200 panel coverage drift")
for arm in ("tailspline", "mrpro"):
    run = root / "runs" / arm
    status = json.loads((run / "status.json").read_text())
    rows = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line]
    if status != {"status": "COMPLETE", "rows": 2600, "lm_rows": 0} or len(rows) != 2600:
        raise ValueError(f"incomplete clean RULER-200 arm: {arm}")
    if {row["prompt_sha256"] for row in rows} != expected:
        raise ValueError(f"unpaired clean RULER-200 prompts: {arm}")
PY

report=${root}/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json
if [[ ! -f "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source "tailspline=${root}/runs/tailspline/generations.jsonl" \
    --source "mrpro=${root}/runs/mrpro/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --length 32768 \
    --out "${report}"
fi
printf 'RULER200_CLEAN_COMPLETE %s\n' "$(date -u +%FT%TZ)"
