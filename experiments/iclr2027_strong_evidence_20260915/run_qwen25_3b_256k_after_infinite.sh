#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
upstream_pid=${UPSTREAM_PID:?set UPSTREAM_PID to the active Pro6000 follow-up driver}
infinite=${plan}/tailspline_llama_s16_infinitebench_100k128k
root=${plan}/four_model_128k_extreme/qwen25_3b_256k

mkdir -p "${root}/logs"

# Wait only for the declared upstream queue. A reused PID is not accepted as the
# same dependency, and a zombie counts as finished.
while [[ -d "/proc/${upstream_pid}" ]]; do
  state=$(ps -o stat= -p "${upstream_pid}" 2>/dev/null | tr -d ' ' || true)
  command=$(ps -o cmd= -p "${upstream_pid}" 2>/dev/null || true)
  [[ -z "${state}" || "${state}" == Z* ]] && break
  if [[ "${command}" != *run_pro6000_followup_queue.sh* ]]; then
    printf 'REFUSE: PID %s was reused by %s\n' "${upstream_pid}" "${command}" >&2
    exit 74
  fi
  sleep 30
done

cd "${repo}"
export PYTHONPATH=.

"${python_bin}" - "${infinite}" "${root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

infinite, root = map(Path, sys.argv[1:])
complete = json.loads((infinite / "complete.json").read_text())
if complete.get("status") != "INFINITEBENCH_100K128K_COMPLETE_V1":
    raise SystemExit("InfiniteBench upstream queue did not complete")

panel_path = root / "assets/panels/262144/inputs_single.jsonl"
panel = json.loads((panel_path.with_name("single_manifest.json")).read_text())
if (
    panel.get("status") != "QWEN25_3B_256K_SINGLE_NIAH_ASSETS_READY_V1"
    or panel.get("rows") != 15
    or hashlib.sha256(panel_path.read_bytes()).hexdigest() != panel.get("inputs_sha256")
):
    raise SystemExit("Qwen 256K single-needle assets are not frozen")

ppl = json.loads((root / "ppl5_256k/manifest.json").read_text())
if (
    ppl.get("contract") != "MODEL_TOKENIZED_LONG_CONTEXT_PPL_V1"
    or ppl.get("datasets") != ["infinitebench_longbook"]
    or ppl.get("documents") != 5
    or ppl.get("lengths") != [262144]
):
    raise SystemExit("Qwen 256K LongBook PPL5 assets are not frozen")

qa = json.loads((root / "infinitebench_en_qa_assets/manifest.json").read_text())
if (
    qa.get("status") != "COMPLETE"
    or qa.get("tasks") != ["longbook_qa_eng"]
    or qa.get("summary", {}).get("selected_rows") != 50
):
    raise SystemExit("Qwen 256K Natural-QA assets are not frozen")
PY

if [[ -f "${root}/append_complete.json" ]]; then
  printf 'Qwen 256K append queue is already complete\n'
  exit 0
fi

bash experiments/iclr2027_strong_evidence_20260915/run_qwen25_3b_256k_append_queue.sh
