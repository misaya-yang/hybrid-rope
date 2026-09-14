#!/usr/bin/env bash
set -euo pipefail

qwen_root=/root/autodl-tmp
qwen_run="${qwen_root}/runs/tailspline"
qwen_supervisor=$(<"${qwen_root}/logs/supervisor.pid")
qwen_child=$1

supervisor_args=$(ps -p "${qwen_supervisor}" -o args=)
child_args=$(ps -p "${qwen_child}" -o args=)
if [[ "${supervisor_args}" != *run_tailspline_qwen25_s2_full324.sh* ]]; then
  printf 'REFUSE unexpected Qwen supervisor: %s\n' "${supervisor_args}" >&2
  exit 1
fi
if [[ "${child_args}" != *tailspline_qwen25_s2_unified_full324/runs/tailspline* ]]; then
  printf 'REFUSE expected live Qwen TailSpline child, got: %s\n' "${child_args}" >&2
  exit 1
fi

kill -STOP "${qwen_supervisor}"
while ps -p "${qwen_child}" -o args= | grep -q 'tailspline_qwen25_s2_unified_full324/runs/tailspline'; do
  sleep 2
done

/root/miniconda3 - "${qwen_run}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
status = json.loads((run / "status.json").read_text())
rows_path = run / "generations.jsonl"
rows = sum(1 for line in rows_path.open() if line.strip())
if status != {"status": "COMPLETE", "rows": 324, "lm_rows": 0} or rows != 324:
    raise ValueError(f"Qwen TailSpline arm did not finish cleanly: status={status}, rows={rows}")
receipt = {
    "status": "QWEN_TAILSPLINE_SINGLE_ARM_ARCHIVED_V1",
    "reason": "author redirected all remaining TailSpline evaluations to Llama",
    "rows": rows,
    "generations_path": str(rows_path),
    "generations_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    "contract_sha256": hashlib.sha256((run / "contract.json").read_bytes()).hexdigest(),
    "excluded_followups": ["qwen_mrpro", "qwen_yarn", "qwen_bm", "qwen_s4"],
}
out = run.parent.parent / "archive_after_single_arm.json"
tmp = out.with_suffix(".json.incomplete")
tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
tmp.replace(out)
print(json.dumps(receipt, sort_keys=True))
PY

kill -KILL "${qwen_supervisor}"
if pgrep -af 'tailspline_qwen25_s2_unified_full324/runs/(mrpro|yarn|bm)' >/dev/null; then
  printf 'REFUSE stale Qwen baseline child appeared during handoff\n' >&2
  exit 1
fi
exec /root/autodl-tmp
