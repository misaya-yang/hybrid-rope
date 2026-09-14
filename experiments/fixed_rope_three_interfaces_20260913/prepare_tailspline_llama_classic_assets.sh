#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
long_sources=/root/autodl-tmp/nongeometric_screen_20260909/long_sources
full13_dir="${experiment_root}/assets/full13"
ppl_dir="${experiment_root}/assets/ppl46"
table_dir="${experiment_root}/tables"

mkdir -p "${experiment_root}/assets" "${table_dir}" "${experiment_root}/logs"
cd "${repo_dir}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${full13_dir}/manifest.json" ]] || \
   ! grep -q '"status": "COMPLETE"' "${full13_dir}/manifest.json"; then
  /root/miniconda3/bin/python -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model_dir}" \
    --upstream "${upstream}" \
    --out "${full13_dir}" \
    --stage H \
    --contract tailspline-classic \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
    --caps 8192,16384,32768 \
    --counts-by-cap 8192:10,16384:10,32768:10 \
    --depth-targets 0.10,0.30,0.50,0.70,0.90 \
    --seed 20260924 >"${experiment_root}/logs/prepare_full13.log" 2>&1
fi

if [[ ! -f "${ppl_dir}/manifest.json" ]] || \
   ! grep -q '"status": "COMPLETE"' "${ppl_dir}/manifest.json"; then
  /root/miniconda3/bin/python -m \
    experiments.fixed_rope_three_interfaces_20260913.prepare_llama_ppl46 \
    --model "${model_dir}" \
    --source-root "${long_sources}" \
    --source-manifest "${long_sources}/sources.json" \
    --out "${ppl_dir}" >"${experiment_root}/logs/prepare_ppl46.log" 2>&1
fi

make_table() {
  local name=$1
  local method=$2
  local role=$3
  local output="${table_dir}/${name}.json"
  if [[ -e "${output}" ]]; then
    return
  fi
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${model_dir}/config.json" \
    --method "${method}" \
    --scale 4 \
    --candidate-id "llama3_8b_s4_classic_${name}" \
    --model-id Meta-Llama-3-8B-Instruct \
    --role "${role}" \
    --changed-variable exponent_allocation \
    --out "${output}"
}

make_table tailspline tailspline candidate
make_table mrpro mrpro baseline

/root/miniconda3/bin/python - \
  "${full13_dir}/manifest.json" "${full13_dir}/rows.jsonl" \
  "${ppl_dir}/manifest.json" "${table_dir}" <<'PY'
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

full_manifest_path, rows_path, ppl_manifest_path, table_root = map(Path, sys.argv[1:])
full = json.loads(full_manifest_path.read_text())
ppl = json.loads(ppl_manifest_path.read_text())
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
counts = {8192: 10, 16384: 10, 32768: 10}
expected = Counter({(task, length): count for task in tasks for length, count in counts.items()})
got = Counter((row.get("task"), int(row.get("length_cap", -1))) for row in rows)
if (
    full.get("status") != "COMPLETE"
    or full.get("contract") != "tailspline-classic"
    or full.get("rows") != 390
    or got != expected
    or len({row.get("prompt_sha256") for row in rows}) != 390
):
    raise ValueError("Full-13 prepared asset violates the frozen 390-row contract")
depths = Counter({(0.1,): 2, (0.3,): 2, (0.5,): 2, (0.7,): 2, (0.9,): 2})
for task in tasks[:6]:
    for length in counts:
        observed = Counter(
            tuple(row.get("depth_target") or []) for row in rows
            if row["task"] == task and row["length_cap"] == length
        )
        if observed != depths:
            raise ValueError(f"missing depth coverage: {task}/{length}/{observed}")
if (
    ppl.get("status") != "COMPLETE"
    or ppl.get("contract") != "TAILSPLINE_LLAMA_PPL46_V1"
    or ppl.get("documents") != 46
    or ppl.get("lengths") != [8192, 16384, 32768]
):
    raise ValueError("PPL46 prepared asset violates its frozen contract")
receipts = {
    name: json.loads((table_root / f"{name}.json").read_text())
    for name in ("tailspline", "mrpro")
}
if {tuple(receipt["band_envelope"]) for receipt in receipts.values()} != {(18, 35)}:
    raise ValueError("classic comparison tables do not share canonical band [18,35]")
if {float(receipt["gain"]).hex() for receipt in receipts.values()} != {
    (1.0 + 0.1 * math.log(4.0)).hex()
}:
    raise ValueError("classic comparison tables do not share canonical S4 gain")
if receipts["tailspline"]["table"]["construction"].get("fitted_coefficients") != 0:
    raise ValueError("TailSpline receipt is not the zero-fit exact construction")
print(json.dumps({
    "status": "TAILSPLINE_LLAMA_CLASSIC_ASSETS_READY_V1",
    "full13_rows": len(rows), "ppl_documents": ppl["documents"],
    "lengths": ppl["lengths"], "band": [18, 35],
    "gain": receipts["tailspline"]["gain"],
}, sort_keys=True))
PY

printf 'ASSETS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
