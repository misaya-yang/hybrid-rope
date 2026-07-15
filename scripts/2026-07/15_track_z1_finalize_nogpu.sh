#!/usr/bin/env bash
set -euo pipefail

: "${READY_FILE:?set READY_FILE to the completed no-GPU READY receipt}"

if [[ ! -r "$READY_FILE" ]]; then
  echo "missing no-GPU READY receipt: $READY_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$READY_FILE"
if [[ "${TRACK_Z_READY_VERSION:-}" != "1" ]]; then
  echo "incompatible no-GPU READY receipt" >&2
  exit 1
fi
if "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "refusing Track Z analysis while CUDA is available; switch to no-GPU mode" >&2
  exit 1
fi

required=(
  "$GPU_COMPLETE_FILE"
  "$CAUSAL_GEO_MANIFEST"
  "$CAUSAL_EVQ_MANIFEST"
  "$SWAP_GEO_OUTPUT/manifest.json"
  "$SWAP_EVQ_OUTPUT/manifest.json"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing completed GPU output: $path" >&2
    exit 1
  fi
done

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -u "$REPO_DIR/scripts/analysis/readout_conversion.py" \
  --causal-manifest "$CAUSAL_GEO_MANIFEST" \
  --causal-manifest "$CAUSAL_EVQ_MANIFEST" \
  --swap-manifest "$SWAP_GEO_OUTPUT/manifest.json" \
  --swap-manifest "$SWAP_EVQ_OUTPUT/manifest.json" \
  --output-dir "$LINCHPIN_OUTPUT"

"$PYTHON_BIN" - "$LINCHPIN_OUTPUT/summary.json" "$LINCHPIN_OUTPUT/linchpin_causal_delta_rank.png" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
figure_path = Path(sys.argv[2])
text = summary_path.read_text(encoding="utf-8")
summary = json.loads(text)

if summary.get("measurement_labels") != {
    "association_swap": "oracle-diagnostic",
    "causal_delta_rank": "oracle-diagnostic",
}:
    raise SystemExit("summary measurement labels are invalid")
if summary.get("single_seed_supporting") is not True:
    raise SystemExit("summary is not marked single-seed supporting")
if summary.get("paper_claim") is not False:
    raise SystemExit("summary incorrectly permits a paper claim")
for private_field in (
    "passkey_text",
    "prompt_ids",
    "prompt_sha256",
    "gold_token_ids",
    "candidate_first_token_ids",
    "generation",
):
    if private_field in text:
        raise SystemExit(f"private field leaked into summary: {private_field}")
if not figure_path.is_file() or figure_path.stat().st_size == 0:
    raise SystemExit("linchpin figure is missing or empty")

test_swap = next(
    row
    for row in summary["association_swap_final_layer"]
    if row["substrate"] == "evq_cosh" and row["split"] == "test"
)
test_scalar = next(
    row
    for row in summary["oracle_scalar_feasibility"]
    if row["substrate"] == "evq_cosh" and row["split"] == "test"
)
ci_low = float(test_swap["mean_ci95"][0])
infeasible = float(test_scalar["infeasible_fraction"])
if ci_low <= 0.0:
    gate = "STOP_TRACK_Z_RECOMMEND_S1"
elif infeasible >= 0.9:
    gate = "STOP_AMPLIFICATION_RECOMMEND_S1"
else:
    gate = "NEXT_AVAILABLE_ZT0_ALPHA1_AND_ONE_DEV_SELECTED_ZTCAL"

print(json.dumps({
    "gate": gate,
    "evq_test_swap_ci95_low": ci_low,
    "evq_test_oracle_scalar_infeasible_fraction": infeasible,
    "causal_rank_gate": "inspect_linchpin; no numeric near-random threshold was preregistered",
    "figure": str(figure_path),
}, indent=2, sort_keys=True))
PY
