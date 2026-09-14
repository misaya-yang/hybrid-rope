#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
source_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_first
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_complete108
data_manifest=/root/autodl-tmp/llama3_planb_20260911/data/P/manifest.json
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/llama_s4_core6_fresh6_16k32k_seed20260917_v2/screen.jsonl
table_dir="${source_root}/tables"

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python - "${source_root}" "${panel}" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

source, panel_path = map(Path, sys.argv[1:])
for arm in ("tailspline", "mrpro", "yarn", "bm"):
    run = source / "runs" / arm
    status = json.loads((run / "status.json").read_text())
    rows = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line]
    cells = Counter((row["task"], row["length_cap"]) for row in rows)
    expected = Counter({(task, length): 6 for task in (
        "niah_single_2", "niah_multikey_2", "niah_multiquery", "vt", "fwe", "qa_1"
    ) for length in (8192, 32768)})
    if status != {"status": "COMPLETE", "rows": 72, "lm_rows": 0} or cells != expected:
        raise ValueError(f"source 72-row arm is not complete and balanced: {arm}/{status}/{cells}")
panel = [json.loads(line) for line in panel_path.read_text().splitlines() if line]
selected = [row for row in panel if row["length_cap"] == 16384]
cells = Counter((row["task"], row["length_cap"]) for row in selected)
if len(selected) != 36 or set(cells.values()) != {6} or len(cells) != 6:
    raise ValueError(f"frozen 16K completion panel is not Core-6 x 6: {cells}")
source_prompts = {
    json.loads(line)["prompt_sha256"]
    for line in (source / "runs" / "tailspline" / "generations.jsonl").read_text().splitlines()
    if line
}
if source_prompts & {row["prompt_sha256"] for row in selected}:
    raise ValueError("16K completion panel overlaps the preserved 8K/32K prompts")
print(json.dumps({"status": "LLAMA_CORRECT108_COMPLETION_READY_V1", "new_rows_per_arm": 36}))
PY

run_arm() {
  local arm=$1
  local output_dir="${experiment_root}/runs/${arm}_16k"
  if [[ -f "${output_dir}/status.json" ]] && \
     grep -q '"rows": 36' "${output_dir}/status.json" && \
     grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  printf 'START %s_16k %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 16384 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${table_dir}/${arm}.json" \
    --table-label "llama3_8b_s4_unified_${arm}_16k_completion" \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/${arm}_16k.log" 2>&1
  printf 'COMPLETE %s_16k %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro
run_arm yarn
run_arm bm

report="${experiment_root}/reports/tailspline_vs_mrpro_yarn_bm_core6_8k16k32k6.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m \
    experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source tailspline="${source_root}/runs/tailspline/generations.jsonl" \
    --source tailspline="${experiment_root}/runs/tailspline_16k/generations.jsonl" \
    --source mrpro="${source_root}/runs/mrpro/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_16k/generations.jsonl" \
    --source yarn="${source_root}/runs/yarn/generations.jsonl" \
    --source yarn="${experiment_root}/runs/yarn_16k/generations.jsonl" \
    --source bm="${source_root}/runs/bm/generations.jsonl" \
    --source bm="${experiment_root}/runs/bm_16k/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --baseline yarn \
    --baseline bm \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
