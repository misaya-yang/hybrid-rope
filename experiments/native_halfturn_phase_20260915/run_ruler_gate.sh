#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/olmo_native_halfturn_phase
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
panel=${root}/assets/ruler/panel/inputs.jsonl
v1=${plan}/olmo_native_z5_enhancement/optimization/table.json
python_bin=/root/miniconda3/bin/python

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: OLMo Native-4K RULER-13x60, arms Native/contract/reverse/V1.'
  printf '%s\n' 'No model is loaded without --execute.'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"

exec 9>/tmp/hybrid-rope-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'REFUSE: another Hybrid-RoPE GPU task holds the global lock' >&2
  exit 1
fi

"${python_bin}" - "${root}" "${v1}" <<'PY'
from pathlib import Path
import hashlib,json,sys
import numpy as np
root=Path(sys.argv[1]);v1=Path(sys.argv[2])
manifest=json.loads((root/'assets/ruler/manifest.json').read_text())
if manifest.get('contract')!='native-halfturn-ruler4k-source-order-v1' or manifest.get('rows')!=780:
    raise SystemExit('RULER asset contract drift')
audit=json.loads((root/'tables/cpu_audit.json').read_text())
if audit.get('status')!='NATIVE_HALFTURN_CPU_AUDIT_V1' or not all(audit['checks'].values()):
    raise SystemExit('CPU table audit is incomplete')
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()
for arm in ('native','contract','reverse'):
    table=json.loads((root/f'tables/{arm}.json').read_text())
    values=np.asarray(table['values_float32'],dtype=np.float32)
    if sha(values)!=audit['table_sha256_float32'][arm] or float(table['gain'])!=1.0:
        raise SystemExit(f'table identity drift: {arm}')
v=json.loads(v1.read_text())
if v.get('table_sha256_float32')!=audit['historical_v1_comparison']['identity']['table_sha256_float32']:
    raise SystemExit('historical V1 identity drift')
print('NATIVE_HALFTURN_RULER_GATE_PREFLIGHT_OK')
PY

validate_run() {
  local arm=$1
  "${python_bin}" - "${root}/runs/${arm}" <<'PY'
from pathlib import Path
import json,sys
run=Path(sys.argv[1])
status=run/'status.json';rows=run/'generations.jsonl'
if not status.exists() or not rows.exists():print('INCOMPLETE');raise SystemExit(0)
if json.loads(status.read_text())!={'status':'COMPLETE','rows':780,'lm_rows':0}:
    raise SystemExit('run status drift')
if sum(1 for line in rows.open() if line.strip())!=780:
    raise SystemExit('generation row count drift')
print('COMPLETE')
PY
}

run_arm() {
  local arm=$1 table=$2 label=$3 state
  state=$(validate_run "${arm}")
  if [[ "${state}" == "COMPLETE" ]]; then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  if [[ "${state}" != "INCOMPLETE" ]]; then
    printf 'REFUSE: unexpected run state %s/%s\n' "${arm}" "${state}" >&2
    exit 1
  fi
  args=(
    --data "${data}" --model "${model}" --arm Native
    --extra-panel "${panel}" --only-extra-panels --skip-lm --length-cap 4096
    --prefill-chunk-size 4096 --batch-size 1
    --out "${root}/runs/${arm}" --execute
  )
  if [[ -n "${table}" ]]; then
    args+=(--static-table-json "${table}" --table-label "${label}")
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval "${args[@]}" \
    >"${root}/logs/${arm}.log" 2>&1
  [[ "$(validate_run "${arm}")" == "COMPLETE" ]]
}

run_arm native '' olmo2_1b_native
run_arm contract "${root}/tables/contract.json" olmo2_1b_native_halfturn_contract
run_arm reverse "${root}/tables/reverse.json" olmo2_1b_native_halfturn_reverse
run_arm v1 "${v1}" olmo2_1b_native_z5_v1_reference

"${python_bin}" -m experiments.native_halfturn_phase_20260915.report_ruler \
  --assets "${root}/assets/ruler" \
  --run "native=${root}/runs/native" \
  --run "contract=${root}/runs/contract" \
  --run "reverse=${root}/runs/reverse" \
  --run "v1=${root}/runs/v1" \
  --out "${root}/reports/ruler4k_four_arm.json"

printf 'OLMO_NATIVE_HALFTURN_RULER_GATE_COMPLETE %s\n' "$(date -u +%FT%TZ)" \
  | tee "${root}/ruler_gate_complete.txt"

