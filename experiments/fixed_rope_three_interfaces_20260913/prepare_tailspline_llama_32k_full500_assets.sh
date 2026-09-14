#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_full500
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
extra_dir="${experiment_root}/assets/full13_32k_extra490"

mkdir -p "${experiment_root}/assets" "${experiment_root}/logs"
cd "${repo_dir}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${extra_dir}/manifest.json" ]] || \
   ! grep -q '"status": "COMPLETE"' "${extra_dir}/manifest.json"; then
  /root/miniconda3/bin/python -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model_dir}" \
    --upstream "${upstream}" \
    --out "${extra_dir}" \
    --stage H \
    --contract planb \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
    --caps 32768 \
    --counts-by-cap 32768:490 \
    --depth-targets 0.10,0.30,0.50,0.70,0.90 \
    --qa-base-offset 6000 \
    --seed 20260929 >"${experiment_root}/logs/prepare_full13_32k_extra490.log" 2>&1
fi

/root/miniconda3/bin/python - \
  "${base_root}/assets/full13/rows.jsonl" \
  "${extra_dir}/manifest.json" "${extra_dir}/rows.jsonl" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

base_path, manifest_path, rows_path = map(Path, sys.argv[1:])
base_all = [json.loads(line) for line in base_path.read_text().splitlines() if line]
base = [row for row in base_all if int(row["length_cap"]) == 32768]
manifest = json.loads(manifest_path.read_text())
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
counts = Counter(row["task"] for row in rows)
if (
    manifest.get("status") != "COMPLETE"
    or manifest.get("rows") != 6370
    or manifest.get("caps") != [32768]
    or manifest.get("qa_base_offset") != 6000
    or manifest.get("rows_sha256") != hashlib.sha256(rows_path.read_bytes()).hexdigest()
    or counts != Counter({task: 490 for task in tasks})
):
    raise ValueError("Llama 32K extra490 asset violates official full500 completion contract")
base_prompts = {row["prompt_sha256"] for row in base}
extra_prompts = {row["prompt_sha256"] for row in rows}
if len(base) != 130 or len(base_prompts) != 130 or len(extra_prompts) != 6370:
    raise ValueError("Llama 32K base10 or extra490 prompt identity/count drift")
if base_prompts & extra_prompts:
    raise ValueError("Llama 32K base10 and extra490 prompts overlap")
base_qa = {row.get("qa_source_index") for row in base if row["task"].startswith("qa_")}
extra_qa = {row.get("qa_source_index") for row in rows if row["task"].startswith("qa_")}
if None in extra_qa or base_qa & extra_qa:
    raise ValueError("Llama 32K base10 and extra490 QA source identities overlap")
print(json.dumps({
    "status": "TAILSPLINE_LLAMA_32K_FULL500_EXTRA490_READY_V1",
    "base_rows": len(base), "extra_rows": len(rows),
    "combined_rows": len(base) + len(rows),
    "rows_per_task": 500,
}, sort_keys=True))
PY

printf 'ASSETS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
