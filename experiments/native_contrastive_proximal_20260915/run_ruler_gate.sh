#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
source_root=${plan}/olmo_native_halfturn_phase
root=${plan}/olmo_native_contrastive_proximal
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
assets=${source_root}/assets/ruler
panel=${assets}/panel/inputs.jsonl
native_run=${source_root}/runs/native
python_bin=/root/miniconda3/bin/python

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: reuse the completed 780-row Native run and generate only 780 NCP rows.'
  printf '%s\n' 'No model is loaded without --execute.'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
mkdir -p "${root}/runs" "${root}/logs" "${root}/reports" "${root}/tables"

"${python_bin}" -m experiments.native_contrastive_proximal_20260915.tables \
  --config "${model}/config.json" --model-id olmo2_1b --out "${root}/tables"

"${python_bin}" - "${assets}" "${native_run}" "${root}/tables/cpu_audit.json" <<'PY'
from pathlib import Path
import json,sys

assets,native_run,audit_path=map(Path,sys.argv[1:])
manifest=json.loads((assets/'manifest.json').read_text())
if manifest.get('contract')!='native-halfturn-ruler4k-source-order-v1' or manifest.get('rows')!=780:
    raise SystemExit('frozen RULER panel drift')
status=json.loads((native_run/'status.json').read_text())
rows=native_run/'generations.jsonl'
if status!={'status':'COMPLETE','rows':780,'lm_rows':0} or sum(1 for line in rows.open() if line.strip())!=780:
    raise SystemExit('the reusable Native run is incomplete')
audit=json.loads(audit_path.read_text())
if audit.get('status')!='NATIVE_CONTRASTIVE_PROXIMAL_CPU_AUDIT_V1' or not all(audit['checks'].values()):
    raise SystemExit('NCP CPU table audit is incomplete')
print('NATIVE_CONTRASTIVE_PROXIMAL_PREFLIGHT_OK')
PY

exec 9>/tmp/hybrid-rope-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'REFUSE: another Hybrid-RoPE GPU task holds the global lock' >&2
  exit 1
fi

validate_run() {
  "${python_bin}" - "${root}/runs/ncp" <<'PY'
from pathlib import Path
import json,sys
run=Path(sys.argv[1]);status=run/'status.json';rows=run/'generations.jsonl'
if not status.exists() or not rows.exists():print('INCOMPLETE');raise SystemExit(0)
if json.loads(status.read_text())!={'status':'COMPLETE','rows':780,'lm_rows':0}:
    print('INCOMPLETE');raise SystemExit(0)
if sum(1 for line in rows.open() if line.strip())!=780:
    raise SystemExit('completed NCP generation row count drift')
print('COMPLETE')
PY
}

state=$(validate_run)
if [[ "${state}" == "COMPLETE" ]]; then
  printf '%s\n' 'SKIP_COMPLETE ncp'
elif [[ "${state}" == "INCOMPLETE" ]]; then
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data}" --model "${model}" --arm Native \
    --extra-panel "${panel}" --only-extra-panels --skip-lm --length-cap 4096 \
    --prefill-chunk-size 4096 --batch-size 1 \
    --static-table-json "${root}/tables/ncp.json" \
    --table-label olmo2_1b_native_contrastive_proximal_v1 \
    --out "${root}/runs/ncp" --execute \
    >"${root}/logs/ncp.log" 2>&1
  [[ "$(validate_run)" == "COMPLETE" ]]
else
  printf 'REFUSE: unexpected NCP run state %s\n' "${state}" >&2
  exit 1
fi

"${python_bin}" -m experiments.native_contrastive_proximal_20260915.report_ruler \
  --assets "${assets}" \
  --native-run "${native_run}" \
  --ncp-run "${root}/runs/ncp" \
  --table-audit "${root}/tables/cpu_audit.json" \
  --out "${root}/reports/ncp_vs_native_ruler4k.json"

printf 'OLMO_NATIVE_CONTRASTIVE_PROXIMAL_COMPLETE %s\n' "$(date -u +%FT%TZ)" \
  | tee "${root}/complete.txt"

