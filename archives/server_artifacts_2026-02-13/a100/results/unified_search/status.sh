#!/usr/bin/env bash
set -euo pipefail
split="A"
log="/opt/dfrope/results/unified_search/log_${split}.txt"
res="/opt/dfrope/results/unified_search/results_${split}.json"
python="/usr/local/miniconda3/envs/py312/bin/python"

echo "[${split}] $(date -Is)"
pid="$(pgrep -f "/opt/dfrope/unified_search.py ${split}" | head -n 1 || true)"
if [ -n "$pid" ]; then
  echo "pid $pid"
else
  echo "pid NONE"
fi
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 || true

echo ""
echo "results file: $res"
if [ -f "$res" ]; then
  $python - <<PY
import json
p="$res"
obj=json.load(open(p))
best=None
for k,v in obj.items():
  p16=v.get("16384")
  if p16 is None:
    continue
  if best is None or p16<best[1]:
    best=(k,p16)
print("completed", len(obj))
print("best@16384", best)
PY
else
  echo MISSING
fi

echo ""
echo "tail log: $log"
tail -n 30 "$log" 2>/dev/null || true
