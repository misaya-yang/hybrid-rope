#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s CURRENT_RUNNER_PID\n' "$0" >&2
  exit 2
fi

current_pid=$1
repo_dir=/root/autodl-tmp/hybrid-rope
source_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_first
next_runner=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_complete108.sh

if [[ ! -r "/proc/${current_pid}/cmdline" ]] || \
   ! tr '\0' ' ' < "/proc/${current_pid}/cmdline" | \
     grep -q 'run_tailspline_llama_s4_core6.sh'; then
  printf 'REFUSE: PID %s is not the expected Llama diagnostic runner\n' "${current_pid}" >&2
  exit 1
fi

while kill -0 "${current_pid}" 2>/dev/null; do
  sleep 5
done

/root/miniconda3/bin/python - "${source_root}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for arm in ("tailspline", "mrpro", "yarn", "bm"):
    status = json.loads((root / "runs" / arm / "status.json").read_text())
    if status != {"status": "COMPLETE", "rows": 72, "lm_rows": 0}:
        raise ValueError(f"diagnostic arm did not finish cleanly: {arm}/{status}")
PY

cd "${repo_dir}"
exec bash "${next_runner}"
