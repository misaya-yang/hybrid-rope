#!/usr/bin/env bash
set -euo pipefail

s2_root=/root/autodl-tmp
s4_script=/root/autodl-tmp

while :; do
  if [[ -f "${s2_root}/runs/bm/status.json" ]] \
      && grep -q '"status": "COMPLETE"' "${s2_root}/runs/bm/status.json"; then
    break
  fi
  supervisor=$(<"${s2_root}/logs/supervisor.pid")
  if ! ps -p "${supervisor}" -o args= | grep -q 'run_tailspline_qwen25_s2_full324.sh'; then
    exit 3
  fi
  sleep 2
done

/root/miniconda3 - "${s2_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for arm in ("tailspline", "mrpro", "yarn", "bm"):
    run = root / "runs" / arm
    status = json.loads((run / "status.json").read_text())
    if status != {"status": "COMPLETE", "rows": 324, "lm_rows": 0}:
        raise ValueError(f"S2 arm {arm} is incomplete: {status}")
    if sum(1 for line in (run / "generations.jsonl").open() if line.strip()) != 324:
        raise ValueError(f"S2 arm {arm} lacks 324 raw rows")
PY

exec "${s4_script}"
