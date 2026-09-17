#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: Native research queue: 288 four-arm behavior -> Native Q/K/V capture/replay -> 64 fixed-block interventions -> independent confirmation.'
  printf '%s\n' 'No model is loaded without the explicit --execute argument.'
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/native_research_20260916
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
python_bin=/root/miniconda3/bin/python
data=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json
ncp=${plan}/olmo_native_contrastive_proximal/tables/ncp.json
halfturn=${plan}/olmo_native_halfturn_phase/tables/contract.json
native_table=${plan}/olmo_native_halfturn_phase/tables/native.json
v1=${plan}/olmo_native_z5_enhancement/optimization/table.json
mechanism=${root}/assets/mechanism_v2
capture=${root}/assets/capture96

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${root}/runs" "${root}/reports" "${root}/logs"

exec 8>/tmp/hybrid-rope-native-queue.lock
if ! flock -n 8; then
  printf '%s\n' 'REFUSE: another Native research queue is active.' >&2
  exit 1
fi
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${active}" ]]; then
  printf 'REFUSE: GPU already has active compute process(es): %s\n' "${active}" >&2
  exit 1
fi

"${python_bin}" -m experiments.native_enhancement_oral_20260915.run \
  --repo "${repo}" --model "${model}" --panel "${mechanism}/inputs.jsonl" \
  --data "${data}" --out "${root}/runs/mechanism" \
  --arm native --arm halfturn --arm ncp --arm v1 --execute \
  >"${root}/logs/mechanism_four_arm.log" 2>&1

for arm in halfturn ncp v1; do
  report=${root}/reports/mechanism_${arm}_vs_native.json
  if [[ ! -f "${report}" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.report \
      --panel "${mechanism}/inputs.jsonl" \
      --run native="${root}/runs/mechanism/native" \
      --run "${arm}=${root}/runs/mechanism/${arm}" --out "${report}"
  fi
done

capture_run=${root}/runs/qkv_capture96
"${python_bin}" -m experiments.checkpoint_attention_replay_20260913.capture_checkpoint \
  --model "${model}" --model-id olmo2_1b_native4k \
  --panel "${capture}/inputs.jsonl" --out "${capture_run}" \
  --layer 3 --layer 7 --layer 11 --layer 15 --row-limit 96 \
  --queries-per-row 4 --query-mode annotated --max-input-tokens 4096 --execute \
  >"${root}/logs/qkv_capture96.log" 2>&1

if [[ ! -f "${root}/reports/signed_phase_response/report.json" ]]; then
  "${python_bin}" -m experiments.checkpoint_attention_replay_20260913.signed_phase_response \
    --capture-index "${capture_run}/index.json" \
    --table halfturn="${halfturn}" --table ncp="${ncp}" --table v1="${v1}" \
    --out "${root}/reports/signed_phase_response" \
    >"${root}/logs/signed_phase_response.log" 2>&1
fi

intervention_panel=${capture}/intervention_inputs.jsonl
canary_panel=${capture}/override_canary_inputs.jsonl
"${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${data}" --model "${model}" --arm Native \
  --extra-panel "${canary_panel}" --only-extra-panels --skip-lm --batch-size 1 \
  --out "${root}/runs/intervention_canary/native" --execute \
  >"${root}/logs/intervention_canary_native.log" 2>&1
"${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${data}" --model "${model}" --arm Native \
  --extra-panel "${canary_panel}" --only-extra-panels --skip-lm --batch-size 1 \
  --layer-override-table-json "${native_table}" --layer-override-range 12:16 \
  --layer-override-label native_with_native_final_quarter_canary \
  --out "${root}/runs/intervention_canary/native_override" --execute \
  >"${root}/logs/intervention_canary_override.log" 2>&1
"${python_bin}" - \
  "${root}/runs/intervention_canary/native/generations.jsonl" \
  "${root}/runs/intervention_canary/native_override/generations.jsonl" <<'PY'
import json,sys
def rows(path): return [json.loads(line) for line in open(path) if line.strip()]
left,right=rows(sys.argv[1]),rows(sys.argv[2])
if len(left)!=2 or [row['row_id'] for row in left]!=[row['row_id'] for row in right]:
 raise SystemExit('layer override parity canary row identity drift')
if any(a['generated_ids']!=b['generated_ids'] for a,b in zip(left,right)):
 raise SystemExit('Native table in the selected layer block changed generated tokens')
print('LAYER_OVERRIDE_NATIVE_PARITY_PASS')
PY

"${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${data}" --model "${model}" --arm Native \
  --extra-panel "${intervention_panel}" --only-extra-panels --skip-lm \
  --batch-size 1 --layer-override-table-json "${ncp}" --layer-override-range 12:16 \
  --layer-override-label native_with_ncp_final_quarter \
  --out "${root}/runs/intervention/native_with_ncp_final_quarter" --execute \
  >"${root}/logs/intervention_native_with_ncp.log" 2>&1

"${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${data}" --model "${model}" --arm Native \
  --extra-panel "${intervention_panel}" --only-extra-panels --skip-lm \
  --batch-size 1 --static-table-json "${ncp}" --table-label ncp_base \
  --layer-override-table-json "${native_table}" --layer-override-range 12:16 \
  --layer-override-label ncp_with_native_final_quarter \
  --out "${root}/runs/intervention/ncp_with_native_final_quarter" --execute \
  >"${root}/logs/intervention_ncp_with_native.log" 2>&1

if [[ ! -f "${root}/reports/final_quarter_intervention.json" ]]; then
  "${python_bin}" -m experiments.native_enhancement_oral_20260915.report_intervention \
    --panel "${intervention_panel}" \
    --native "${root}/runs/mechanism/native/generations.jsonl" \
    --ncp "${root}/runs/mechanism/ncp/generations.jsonl" \
    --native-with-ncp-block "${root}/runs/intervention/native_with_ncp_final_quarter/generations.jsonl" \
    --ncp-with-native-block "${root}/runs/intervention/ncp_with_native_final_quarter/generations.jsonl" \
    --out "${root}/reports/final_quarter_intervention.json"
fi

asset_deadline=$((SECONDS + 1800))
while [[ ! -f "${root}/cpu_ready.json" ]]; do
  if (( SECONDS >= asset_deadline )); then
    printf '%s\n' 'REFUSE: independent-confirmation CPU assets were not ready within 30 minutes.' >&2
    exit 1
  fi
  sleep 5
done
ruler=${root}/assets/ruler_confirm_13x10/panels/4096/inputs.jsonl
qa=${root}/assets/naturalqa_3x80/inputs.jsonl
for suite in ruler qa; do
  panel_var=${suite}
  panel_path=${!panel_var}
  for arm in native ncp; do
    args=(--data "${data}" --model "${model}" --arm Native --extra-panel "${panel_path}"
          --only-extra-panels --skip-lm --batch-size 1 --out "${root}/runs/confirm/${suite}/${arm}")
    if [[ "${arm}" == ncp ]]; then
      args+=(--static-table-json "${ncp}" --table-label "olmo_native_confirm_${suite}_ncp")
    fi
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval "${args[@]}" --execute \
      >"${root}/logs/confirm_${suite}_${arm}.log" 2>&1
  done
done

if [[ ! -f "${root}/reports/independent_confirmation.json" ]]; then
  "${python_bin}" -m experiments.native_enhancement_oral_20260915.report_confirmation \
    --ruler-panel "${ruler}" \
    --ruler-native "${root}/runs/confirm/ruler/native/generations.jsonl" \
    --ruler-ncp "${root}/runs/confirm/ruler/ncp/generations.jsonl" \
    --qa-panel "${qa}" \
    --qa-native "${root}/runs/confirm/qa/native/generations.jsonl" \
    --qa-ncp "${root}/runs/confirm/qa/ncp/generations.jsonl" \
    --out "${root}/reports/independent_confirmation.json"
fi

"${python_bin}" -m experiments.native_enhancement_oral_20260915.run_native_lm \
  --model "${model}" --manifest "${root}/assets/lm128/manifest.json" \
  --tokens "${root}/assets/lm128/tokens_128x4097.npy" --ncp-table "${ncp}" \
  --out "${root}/runs/confirm/lm" --execute \
  >"${root}/logs/confirm_lm.log" 2>&1

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
reports={
 'mechanism_ncp':root/'reports/mechanism_ncp_vs_native.json',
 'signed_response':root/'reports/signed_phase_response/report.json',
 'block_intervention':root/'reports/final_quarter_intervention.json',
 'independent_confirmation':root/'reports/independent_confirmation.json',
 'lm_confirmation':root/'runs/confirm/lm/report.json',
}
if any(not path.is_file() for path in reports.values()): raise SystemExit('Native queue reports incomplete')
receipt={'status':'NATIVE_RESEARCH_GPU_QUEUE_COMPLETE_V1','reports':{
 name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for name,path in reports.items()}}
path=root/'gpu_complete.json';tmp=path.with_name(path.name+'.incomplete')
tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(path)
print(json.dumps(receipt,sort_keys=True))
PY
