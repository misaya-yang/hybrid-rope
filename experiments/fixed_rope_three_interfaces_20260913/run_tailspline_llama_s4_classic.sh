#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
full13_dir="${experiment_root}/assets/full13"
ppl_dir="${experiment_root}/assets/ppl46"
table_dir="${experiment_root}/tables"

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in \
  "${full13_dir}/manifest.json" "${full13_dir}/rows.jsonl" \
  "${ppl_dir}/manifest.json" "${ppl_dir}/lm.npy" \
  "${table_dir}/tailspline.json" "${table_dir}/mrpro.json"; do
  if [[ ! -f "${required}" ]]; then
    printf 'REFUSE missing prepared classic asset: %s\n' "${required}" >&2
    exit 1
  fi
done

/root/miniconda3/bin/python - \
  "${full13_dir}/manifest.json" "${full13_dir}/rows.jsonl" \
  "${ppl_dir}/manifest.json" "${ppl_dir}/lm.npy" "${table_dir}" <<'PY'
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

full_path, rows_path, ppl_path, lm_path, table_root = map(Path, sys.argv[1:])
full = json.loads(full_path.read_text())
ppl = json.loads(ppl_path.read_text())
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
lengths = (8192, 16384, 32768)
expected = Counter({(task, length): 10 for task in tasks for length in lengths})
got = Counter((row.get("task"), int(row.get("length_cap", -1))) for row in rows)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if (
    full.get("status") != "COMPLETE"
    or full.get("contract") != "tailspline-classic"
    or full.get("rows") != 390
    or full.get("rows_sha256") != sha(rows_path)
    or got != expected
    or len({row.get("prompt_sha256") for row in rows}) != 390
):
    raise ValueError("refusing GPU launch: Full-13 asset is not the frozen 390-row contract")
wanted_depths = Counter({(0.1,): 2, (0.3,): 2, (0.5,): 2, (0.7,): 2, (0.9,): 2})
for task in tasks[:6]:
    for length in lengths:
        observed = Counter(
            tuple(row.get("depth_target") or []) for row in rows
            if row["task"] == task and row["length_cap"] == length
        )
        if observed != wanted_depths:
            raise ValueError(f"refusing GPU launch: depth contract drift {task}/{length}")
if (
    ppl.get("status") != "COMPLETE"
    or ppl.get("contract") != "TAILSPLINE_LLAMA_PPL46_V1"
    or ppl.get("documents") != 46
    or ppl.get("lengths") != list(lengths)
    or ppl.get("lm_array_sha256") != sha(lm_path)
):
    raise ValueError("refusing GPU launch: PPL46 asset drift")
receipts = {
    name: json.loads((table_root / f"{name}.json").read_text())
    for name in ("tailspline", "mrpro")
}
expected_sources = {
    "tailspline": "analytic:tailspline", "mrpro": "analytic:mrpro",
}
for name, value in receipts.items():
    if value.get("source") != expected_sources[name]:
        raise ValueError(f"refusing GPU launch: {name} construction identity drift")
    expected_role = "candidate" if name == "tailspline" else "baseline"
    if value.get("role") != expected_role:
        raise ValueError(f"refusing GPU launch: {name} role drift")
if {tuple(value["band_envelope"]) for value in receipts.values()} != {(18, 35)}:
    raise ValueError("refusing GPU launch: table band drift")
if {float(value["gain"]).hex() for value in receipts.values()} != {
    (1.0 + 0.1 * math.log(4.0)).hex()
}:
    raise ValueError("refusing GPU launch: table gain drift")
if len({value["table_sha256_float32"] for value in receipts.values()}) != 2:
    raise ValueError("refusing GPU launch: TailSpline and MrPro tables are not distinct")
print(json.dumps({"status": "TAILSPLINE_LLAMA_CLASSIC_GPU_READY_V1", "rows": 390, "lm_rows": 138}))
PY

run_arm() {
  local arm=$1
  local output_dir="${experiment_root}/runs/${arm}"
  local status_path="${output_dir}/status.json"
  if [[ -f "${status_path}" ]] && /root/miniconda3/bin/python - "${status_path}" <<'PY'
import json
import sys
status = json.load(open(sys.argv[1]))
if status != {"status": "COMPLETE", "rows": 390, "lm_rows": 138}:
    raise SystemExit(1)
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi

  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${ppl_dir}/manifest.json" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${full13_dir}/rows.jsonl" \
    --only-extra-panels \
    --length-cap 8192 \
    --length-cap 16384 \
    --length-cap 32768 \
    --lm-length-cap 8192 \
    --lm-length-cap 16384 \
    --lm-length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${table_dir}/${arm}.json" \
    --table-label "llama3_8b_s4_classic_${arm}" \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro

/root/miniconda3/bin/python -m \
  experiments.fixed_rope_three_interfaces_20260913.build_mrrope_baseline_registry \
  --root /root/autodl-tmp \
  --out /root/autodl-tmp/mrrope_baselines

report="${experiment_root}/reports/tailspline_vs_mrpro_classic.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m \
    experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report \
    --run tailspline="${experiment_root}/runs/tailspline" \
    --run mrpro="${experiment_root}/runs/mrpro" \
    --receipt tailspline="${table_dir}/tailspline.json" \
    --receipt mrpro="${table_dir}/mrpro.json" \
    --ppl-manifest "${ppl_dir}/manifest.json" \
    --candidate tailspline \
    --baseline mrpro \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
