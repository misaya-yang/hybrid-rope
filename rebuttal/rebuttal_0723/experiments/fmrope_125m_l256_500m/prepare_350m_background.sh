#!/usr/bin/env bash
set -euo pipefail

: "${FMR_WORK_DIR:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/AGENTS.md" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/AGENTS.md" ]] || {
  echo "repository root not found" >&2
  exit 2
}
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=""
export FMR_MODEL_TIER=350m
export FMR_SEED=42
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

RUNNER="rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_5090.sh"
bash "$RUNNER" prepare-350m
bash "$RUNNER" preflight-350m

"$PYTHON_BIN" - "$FMR_WORK_DIR/data/data_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
segments = manifest["train"]["segments"]
expected = [
    {"name": "A", "token_start": 0, "token_stop": 499_974_144, "tokens": 499_974_144},
    {"name": "B", "token_start": 499_974_144, "token_stop": 999_948_288, "tokens": 499_974_144},
]
if segments != expected:
    raise SystemExit(f"unexpected training segments: {segments}")
print("1B non-overlap segment gate: PASS")
PY

printf 'READY\n' > "$FMR_WORK_DIR/READY"
