#!/usr/bin/env bash
set -euo pipefail

old_root=/root/autodl-tmp/today_rope_plan_20260914/qwen25_s2_full324
old_run="${old_root}/runs/mix075_b2"
old_supervisor=$(<"${old_root}/logs/supervisor.pid")
current_child=$1

supervisor_args=$(ps -p "${old_supervisor}" -o args=)
child_args=$(ps -p "${current_child}" -o args=)
if [[ "${supervisor_args}" != *run_qwen25_s2_full324.sh* ]]; then
  printf 'REFUSE unexpected old supervisor: %s\n' "${supervisor_args}" >&2
  exit 1
fi
if [[ "${child_args}" != *qwen25_s2_full324/runs/mix075_b2* ]]; then
  printf 'REFUSE expected live mix child, got: %s\n' "${child_args}" >&2
  exit 1
fi

# Freeze only the old queue shell while its already-running mix child finishes.
# This closes the race in which the shell could fork MrPro before this handoff
# validates mix and starts TailSpline. SIGSTOP is not sent to the child.
kill -STOP "${old_supervisor}"

while ps -p "${current_child}" -o args= | grep -q 'qwen25_s2_full324/runs/mix075_b2'; do
  sleep 2
done

/root/miniconda3/bin/python - "${old_run}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
status = json.loads((run / "status.json").read_text())
if status != {"status": "COMPLETE", "rows": 324, "lm_rows": 0}:
    raise ValueError(f"superseded mix arm did not finish cleanly: {status}")
if sum(1 for line in (run / "generations.jsonl").open() if line.strip()) != 324:
    raise ValueError("superseded mix arm lacks 324 preserved raw rows")
PY

kill -KILL "${old_supervisor}"
if pgrep -af 'qwen25_s2_full324/runs/(mrpro|yarn|bm)' >/dev/null; then
  printf 'REFUSE legacy baseline child appeared during handoff\n' >&2
  exit 1
fi
exec /root/autodl-tmp/hybrid-rope/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_qwen25_s2_full324.sh
