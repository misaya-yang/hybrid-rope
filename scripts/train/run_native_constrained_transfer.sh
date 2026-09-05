#!/usr/bin/env bash
# One explicitly chosen stage. Never starts the next arm/seed or powers off a host.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?choose preflight/cache/cache-resume/smoke/smoke-resume/compare-smoke/train/resume/evaluate/native/review}
: "${EVQ_CHECKPOINT:?exact registered local checkpoint}"
: "${EVQ_WORK_DIR:?private output root}"
: "${TASK_MANIFEST:?qualified natural manifest; no synthetic substitute}"
: "${NATIVE_POOL:?independent Native replay/calibration/validation manifest}"
py=${EVQ_PYTHON:-python3}
arm=${E2_ARM:-N}
seed=${E2_SEED:-42}
label=${E2_LABEL:-${arm}_s${seed}_first}
out="$EVQ_WORK_DIR/e2_$label"
common=(--checkpoint "$EVQ_CHECKPOINT" --tasks "$TASK_MANIFEST" --native-pool "$NATIVE_POOL")
engine=scripts/train/train_single_table_native_constrained.py
table=()
if [[ "$arm" != N ]]; then
  [[ "$arm" == Z || "$arm" == Y ]] || { echo 'E2_ARM must be N/Z/Y' >&2; exit 2; }
  gain=$("$py" - "$EVQ_WORK_DIR/fixed_controls/manifest.json" "$arm" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); assert m['status']=='FIXED_NZGY_CONTROLS_FROZEN_V1'
print(m['arms'][sys.argv[2]]['rotary_amplitude'])
PY
)
  table=(--table "$EVQ_WORK_DIR/fixed_controls/$arm.npy" --gain "$gain")
fi
if [[ "$stage" == preflight ]]; then
  cached=(); [[ ! -f "${TEACHER_CACHE:-}/manifest.json" ]] || cached=(--teacher-cache "$TEACHER_CACHE")
  exec "$py" "$engine" preflight "${common[@]}" "${cached[@]}" --output "${out}_preflight"
fi
if [[ "$stage" == review ]]; then
  : "${E2_ADAPTER:?saved step directory}"
  : "${E2_NATIVE_BASELINE:?original-Native fresh validation directory}"
  : "${E2_TASK_BASELINE:?original-Native validation directory}"
  exec "$py" scripts/analysis/review_native_constrained_transfer.py \
    --checkpoint "$EVQ_CHECKPOINT" --adapter "$E2_ADAPTER" \
    --native-baseline "$E2_NATIVE_BASELINE" --native-candidate "${out}_native" \
    --task-baseline "$E2_TASK_BASELINE" --task-candidate "${out}_validation" --output "${out}_review.json"
fi
if [[ "$stage" == compare-smoke ]]; then
  : "${E2_REFERENCE_RUN:?uninterrupted smoke directory}"
  : "${E2_RESUMED_RUN:?resumed smoke directory}"
  exec "$py" "$engine" compare-smoke "${common[@]}" \
    --reference-run "$E2_REFERENCE_RUN" --resumed-run "$E2_RESUMED_RUN" --output "${out}_resume_parity.json"
fi
[[ "${SINGLE_TABLE_GPU_AUTHORIZED:-}" == YES ]] || { echo 'Choose the exact machine/time cap and authorize this GPU stage first.' >&2; exit 2; }
gpu=(--authorized --max-seconds "${EVQ_MAX_SECONDS:-3600}" --seed "$seed")
case "$stage" in
  cache|cache-resume)
    : "${TEACHER_CACHE:?new or interrupted original-Native cache directory}"
    resume=(); [[ "$stage" != cache-resume ]] || resume=(--resume-cache)
    exec "$py" "$engine" cache-native --checkpoint "$EVQ_CHECKPOINT" --native-pool "$NATIVE_POOL" \
      "${gpu[@]}" "${resume[@]}" --output "$TEACHER_CACHE" ;;
  smoke|smoke-resume|train|resume)
    : "${TEACHER_CACHE:?completed teacher cache}"
    action=$stage; resume=()
    if [[ "$stage" == resume ]]; then
      : "${E2_RESUME:?verified saved step directory}"
      : "${E2_STOP_STEP:?next saved step 96 or 128}"
      action=train; resume=(--resume "$E2_RESUME")
    elif [[ "$stage" == smoke-resume ]]; then
      : "${E2_RESUME:?step_001 of the completed same-arm smoke}"
      action=smoke; resume=(--resume "$E2_RESUME")
    fi
    exec "$py" "$engine" "$action" "${common[@]}" "${table[@]}" "${gpu[@]}" "${resume[@]}" \
      --teacher-cache "$TEACHER_CACHE" --arm "$arm" --stop-after-step "${E2_STOP_STEP:-64}" --output "$out" ;;
  evaluate)
    adapter=(); [[ -z "${E2_ADAPTER:-}" ]] || adapter=(--adapter "$E2_ADAPTER")
    exec "$py" "$engine" evaluate "${common[@]}" "${table[@]}" "${gpu[@]}" "${adapter[@]}" \
      --split validation --lengths 2048 16384 --output "${out}_validation" ;;
  native)
    adapter=(); [[ -z "${E2_ADAPTER:-}" ]] || adapter=(--adapter "$E2_ADAPTER")
    exec "$py" "$engine" native-evaluate "${common[@]}" "${table[@]}" "${gpu[@]}" "${adapter[@]}" \
      --split validation --output "${out}_native" ;;
  *) echo 'Unknown stage; no GPU action started.' >&2; exit 2 ;;
esac
