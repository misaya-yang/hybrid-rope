#!/usr/bin/env bash
set -euo pipefail

repo=${CA_NCP_REPO:-/root/autodl-tmp/hybrid-rope}
root=${CA_NCP_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_native_20260917}
model=${CA_NCP_MODEL:-/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct}
ncp=${CA_NCP_TABLE:-/root/autodl-tmp/today_rope_plan_20260914/olmo_native_contrastive_proximal/tables/ncp.json}
pilot_reuse=${CA_NCP_PILOT_REUSE:-/root/autodl-tmp/today_rope_plan_20260914/native_research_20260916/assets/ruler_confirm_13x10/manifest.json}
python_bin=${CA_NCP_PYTHON:-/root/miniconda3/bin/python}
unlabeled_manifest=""
execute=0

while (($#)); do
  case "$1" in
    --execute) execute=1 ;;
    --unlabeled-manifest) shift; unlabeled_manifest=${1:?missing manifest path} ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

if [[ ${execute} != 1 ]]; then
  printf '%s\n' 'PLAN_ONLY: CPU tables + Full-13x10 pilot + optional unlabeled token/pair assets. No model load and no GPU execution.'
  exit 0
fi

cd "${repo}"
mkdir -p "${root}/logs"
"${python_bin}" -m experiments.ca_ncp_native_20260917.build_alignment \
  --model "${model}" --ncp-table "${ncp}" --tables-only \
  --out "${root}/construction" \
  >"${root}/logs/build_tables.log" 2>&1

"${python_bin}" -m experiments.ca_ncp_native_20260917.prepare_pilot \
  --model "${model}" --reuse-manifest "${pilot_reuse}" --rows-per-task 10 \
  --out "${root}/assets/pilot" \
  >"${root}/logs/prepare_pilot.log" 2>&1

"${python_bin}" -m experiments.ca_ncp_native_20260917.reuse_baselines \
  --root "${root}" \
  >"${root}/logs/reuse_baselines.log" 2>&1

if [[ -n "${unlabeled_manifest}" ]]; then
  "${python_bin}" -m experiments.ca_ncp_native_20260917.prepare_statistics \
    --model "${model}" --corpus-manifest "${unlabeled_manifest}" \
    --fit-documents 32 --report-documents 8 --pairs-per-document 512 \
    --out "${root}/assets/statistics" \
    >"${root}/logs/prepare_statistics.log" 2>&1
  statistics_status=CPU_PREPARED
else
  statistics_status=MISSING_FORMAL_PG19_PROOFPILE_TRAIN_MANIFEST
fi

"${python_bin}" -c '
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); status=sys.argv[2]
payload={
  "status":"CA_NCP_CPU_PREPARATION_COMPLETE" if status=="CPU_PREPARED" else "CA_NCP_CPU_PARTIAL_ASSET_READY",
  "gpu_execution":False,
  "method_receipt":str(root/"construction/METHOD_RECEIPT.json"),
  "pilot_manifest":str(root/"assets/pilot/manifest.json"),
  "baseline_reuse_receipt":str(root/"BASELINE_REUSE_RECEIPT.json"),
  "statistics_assets":status,
  "next_step":"capture_statistics --execute only after a GPU is attached and formal unlabeled statistics assets exist",
}
path=root/"CPU_READINESS.json"; tmp=path.with_name(path.name+".incomplete")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
print(json.dumps(payload,sort_keys=True))
' "${root}" "${statistics_status}"
