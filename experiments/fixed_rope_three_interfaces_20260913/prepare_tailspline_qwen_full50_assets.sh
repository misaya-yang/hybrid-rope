#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_full50
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
panel_dir="${experiment_root}/assets/full13_50"

mkdir -p "${experiment_root}/assets" "${experiment_root}/logs"
cd "${repo_dir}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${panel_dir}/manifest.json" ]] || \
   ! grep -q '"status": "COMPLETE"' "${panel_dir}/manifest.json"; then
  /root/miniconda3/bin/python -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model_dir}" \
    --model-contract generic \
    --upstream "${upstream}" \
    --out "${panel_dir}" \
    --stage H \
    --contract planb \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
    --caps 32768,65536 \
    --counts-by-cap 32768:50,65536:50 \
    --depth-targets 0.10,0.30,0.50,0.70,0.90 \
    --qa-base-offset 8000 \
    --seed 20260928 >"${experiment_root}/logs/prepare_full13_50.log" 2>&1
fi

/root/miniconda3/bin/python - "${panel_dir}/manifest.json" "${panel_dir}/rows.jsonl" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

manifest_path, rows_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
tasks = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
lengths = (32768, 65536)
counts = Counter((row["task"], int(row["length_cap"])) for row in rows)
expected = Counter({(task, length): 50 for task in tasks for length in lengths})
sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
if (
    manifest.get("status") != "COMPLETE"
    or manifest.get("rows") != 1300
    or manifest.get("caps") != list(lengths)
    or manifest.get("qa_base_offset") != 8000
    or manifest.get("rows_sha256") != sha
    or counts != expected
    or len({row["prompt_sha256"] for row in rows}) != 1300
):
    raise ValueError("Qwen Full-13 x 32/64K x 50 asset violates the frozen contract")
print(json.dumps({
    "status": "TAILSPLINE_QWEN_FULL50_READY_V1",
    "rows": len(rows), "rows_per_task_length": 50,
    "lengths": list(lengths), "rows_sha256": sha,
}, sort_keys=True))
PY

printf 'ASSETS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
