#!/usr/bin/env bash
# Serial paper-grade official-static-YaRN natural-QA queue. Plan-only by default.
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
module=experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa
conditions=(llama3_8b qwen25_3b olmo2_1b glm4_9b)

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: official static YaRN, zero-training installation.'
  printf '%s\n' 'Models: Llama-3-8B, Qwen2.5-3B, OLMo-2-1B, GLM-4-9B.'
  printf '%s\n' 'Qwen2.5-1.5B is explicitly excluded.'
  printf '%s\n' 'Execution first preflights every model; no GPU arm starts if any Tail/Mr prerequisite is missing.'
  printf '%s\n' 'Reports remain per model and are never pooled across benchmark families.'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Finish every read-only dependency check before the first new generation.
for condition in "${conditions[@]}"; do
  "${python_bin}" -m "${module}" --condition "${condition}" \
    --plan-root "${plan}" --python "${python_bin}" --check-ready
done

exec 9>"${gpu_lock}"
flock -n 9 || { echo 'REFUSE: another process owns the shared GPU lock' >&2; exit 73; }

# The existing Llama launcher is the frozen owner for its 631-row YaRN arm.
ROPE_PLAN_ROOT="${plan}" ROPE_PYTHON="${python_bin}" \
  bash experiments/iclr2027_three_track_sprint_20260915/run_naturalqa_yarn.sh --execute
"${python_bin}" -m "${module}" --condition llama3_8b \
  --plan-root "${plan}" --python "${python_bin}" --finalize-only

# The remaining runners derive runtime settings from each model's completed
# TailSpline/MrPro contracts and generate only the missing YaRN arm.
for condition in olmo2_1b qwen25_3b glm4_9b; do
  "${python_bin}" -m "${module}" --condition "${condition}" \
    --plan-root "${plan}" --python "${python_bin}" --execute
done

"${python_bin}" - "${plan}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

plan = Path(sys.argv[1])
reports = {
    "llama3_8b": plan / "tailspline_llama_s4_naturalqa631_yarn_a1/reports/official_static_yarn_naturalqa_triarm.json",
    "qwen25_3b": plan / "four_model_128k_extreme/qwen25_3b_128k/infinitebench_en_qa_yarn_a1/reports/official_static_yarn_naturalqa_triarm.json",
    "olmo2_1b": plan / "tailspline_olmo_s4_naturalqa631_yarn_a1/reports/official_static_yarn_naturalqa_triarm.json",
    "glm4_9b": plan / "glm4_9b_s4_128k/en_qa_yarn_a1/reports/official_static_yarn_naturalqa_triarm.json",
}
owners = []
for condition, path in reports.items():
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE" or value.get("condition") != condition:
        raise SystemExit(f"incomplete report: {condition}")
    if value.get("cross_model_pooling_allowed") is not False:
        raise SystemExit(f"cross-model pooling guard missing: {condition}")
    owners.append({
        "condition": condition,
        "report": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
receipt = {
    "status": "FOUR_MODEL_OFFICIAL_STATIC_YARN_NATURAL_QA_COMPLETE_V1",
    "method_identity": "official static YaRN, zero-training installation",
    "models": ["Llama-3-8B", "Qwen2.5-3B", "OLMo-2-1B", "GLM-4-9B"],
    "excluded_models": ["Qwen2.5-1.5B-Instruct"],
    "cross_model_pooling_allowed": False,
    "owners": owners,
}
out = plan / "official_yarn_naturalqa_four_model_complete.json"
temporary = out.with_name(out.name + ".incomplete")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, out)
print(json.dumps(receipt, sort_keys=True))
PY
