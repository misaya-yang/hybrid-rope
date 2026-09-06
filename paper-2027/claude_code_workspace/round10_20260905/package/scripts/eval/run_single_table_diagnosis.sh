#!/usr/bin/env bash
# E0/E1 only. E2 uses train_single_table_native_constrained.py with qualified assets.
set -euo pipefail
ACTION="${1:-help}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${EVQ_PYTHON:-python3}"
if [[ "$ACTION" == help ]]; then
  echo 'E0/E1 stages: prepare preflight smoke controls native nll-screen retention select seal far8 far16 far32'
  echo 'Fixed arms: N Z G Y (EVQ_ARM defaults to Z). No scale/gain/movement search.'
  echo 'E2: scripts/train/train_single_table_native_constrained.py --help'
  echo 'Required: EVQ_CHECKPOINT EVQ_WORK_DIR; prepare: EVQ_NATIVE_TABLE EVQ_LOG_P2_S4_TABLE; retention: EVQ_RETENTION_MANIFEST'
  echo 'GPU stages: SINGLE_TABLE_GPU_AUTHORIZED=YES; EVQ_MAX_SECONDS defaults to 3600 per stage.'
  exit 0
fi
CHECKPOINT="${EVQ_CHECKPOINT:?set EVQ_CHECKPOINT}"
WORK="${EVQ_WORK_DIR:?set EVQ_WORK_DIR to a new experiment directory}"
ENTRY="$ROOT/scripts/experiments/single_table_generation.py"
DATA="$WORK/source_twins"
CONTROLS="$WORK/fixed_controls"
ARM="${EVQ_ARM:-Z}"
FOLD="${EVQ_RETENTION_FOLD:-selection}"
ARGS=(--checkpoint "$CHECKPOINT" --data "$DATA" --seed 20260904)
GPU=(--authorized --max-seconds "${EVQ_MAX_SECONDS:-3600}" --min-headroom-gib 1)
TOKENIZER_ARGS=(--asset-tokenizer-root "${EVQ_ASSET_TOKENIZER_ROOT:-$CHECKPOINT}")
require_gpu() {
  [[ "${SINGLE_TABLE_GPU_AUTHORIZED:-}" == YES ]] || {
    echo 'GPU stage not authorized: set SINGLE_TABLE_GPU_AUTHORIZED=YES for the chosen machine/time cap.' >&2
    exit 3
  }
}
require_resolved() {
  "$PYTHON" "$ENTRY" summarize --output "$1"
  "$PYTHON" - "$1" <<'PY'
import json,sys
from pathlib import Path
s=json.loads((Path(sys.argv[1])/'summary.json').read_text())
if not s.get('diagnoses') or any(not v.startswith('RESOLVED_NONZERO') for v in s['diagnoses'].values()):
    raise SystemExit('Prior length did not resolve with nonzero exact+EOS; inspect raw failure before expansion.')
PY
}
method_args() {
  mapfile -t VALUES < <("$PYTHON" - "$CONTROLS/manifest.json" "$ARM" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); a=json.loads(p.read_text())['arms'][sys.argv[2]]
print(p.parent/a['path']); print(a['rotary_amplitude'])
PY
  )
  [[ ${#VALUES[@]} == 2 ]] || { echo 'Unknown or malformed fixed arm' >&2; exit 4; }
  METHOD=(--table "${VALUES[0]}" --gain "${VALUES[1]}")
  [[ "$ARM" != N ]] || METHOD=()
}
case "$ACTION" in
 prepare)
  mkdir -p "$WORK"
  "$PYTHON" "$ROOT/scripts/analysis/export_single_table_controls.py" --checkpoint "$CHECKPOINT" \
    --native-table "${EVQ_NATIVE_TABLE:?set EVQ_NATIVE_TABLE}" \
    --reference-log-s4 "${EVQ_LOG_P2_S4_TABLE:?set EVQ_LOG_P2_S4_TABLE}" --output "$CONTROLS"
  "$PYTHON" "$ENTRY" prepare "${ARGS[@]}" --eval-pairs 24 --output "$DATA"
  ;;
 preflight)
  "$PYTHON" "$ENTRY" preflight "${ARGS[@]}" "${TOKENIZER_ARGS[@]}" \
    --retention-manifest "${EVQ_RETENTION_MANIFEST:?set EVQ_RETENTION_MANIFEST}" --output "$WORK/preflight"
  ;;
 smoke)
  require_gpu; method_args
  "$PYTHON" "$ENTRY" smoke "${ARGS[@]}" "${METHOD[@]}" "${GPU[@]}" --output "$WORK/smoke/$ARM"
  ;;
 controls)
  require_gpu
  "$PYTHON" "$ENTRY" evaluate "${ARGS[@]}" "${GPU[@]}" --factors 1 --audit-margins --output "$WORK/controls_native"
  ;;
 native)
  require_gpu
  "$PYTHON" "$ENTRY" retention "${ARGS[@]}" "${GPU[@]}" "${TOKENIZER_ARGS[@]}" \
    --retention-manifest "${EVQ_RETENTION_MANIFEST:?set EVQ_RETENTION_MANIFEST}" \
    --retention-fold "$FOLD" --output "$WORK/native_$FOLD"
  ;;
 retention|nll-screen)
  require_gpu; method_args
  EXTRA=(); DEST="$WORK/retention_$FOLD/$ARM"
  if [[ "$ACTION" == nll-screen ]]; then EXTRA+=(--nll-only); DEST="$WORK/nll_$FOLD/$ARM"; fi
  "$PYTHON" "$ENTRY" retention "${ARGS[@]}" "${METHOD[@]}" "${GPU[@]}" "${EXTRA[@]}" "${TOKENIZER_ARGS[@]}" \
    --retention-manifest "${EVQ_RETENTION_MANIFEST:?set EVQ_RETENTION_MANIFEST}" \
    --retention-fold "$FOLD" --baseline "$WORK/native_$FOLD" --output "$DEST"
  ;;
 select)
  require_gpu; method_args
  "$PYTHON" "$ENTRY" evaluate "${ARGS[@]}" "${METHOD[@]}" "${GPU[@]}" \
    --factors 1 4 --audit-margins --output "$WORK/selection/$ARM"
  ;;
 seal)
  mkdir -p "$WORK/sealed"
  "$PYTHON" "$ENTRY" seal --selection-result "$WORK/selection/$ARM" \
    --retention-result "$WORK/retention_selection/$ARM" --output "$WORK/sealed/$ARM.json"
  ;;
 far8|far16|far32)
  require_gpu; method_args
  FACTOR="${ACTION#far}"
  if [[ "$FACTOR" == 16 ]]; then require_resolved "$WORK/blind/$ARM/x8"; fi
  if [[ "$FACTOR" == 32 ]]; then require_resolved "$WORK/blind/$ARM/x16"; fi
  "$PYTHON" "$ENTRY" evaluate "${ARGS[@]}" "${METHOD[@]}" "${GPU[@]}" \
    --split blind --factors "$FACTOR" --frozen-selection "$WORK/sealed/$ARM.json" \
    --output "$WORK/blind/$ARM/x$FACTOR"
  ;;
 train-ce|train-source|train-qkvo|acquire-format|acquire-controls|adapter-retention|adapter-select)
  echo 'Superseded prototype. Use the Native-constrained E2 engine and qualified natural/Native data; see current preflight.' >&2
  exit 5
  ;;
 *) echo "Unknown stage: $ACTION" >&2; exit 2 ;;
esac
