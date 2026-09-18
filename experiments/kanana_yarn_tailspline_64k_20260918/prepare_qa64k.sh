#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
data_root=${INFINITEBENCH_ROOT:-/root/autodl-tmp/InfiniteBench/data}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
assets=${root}/qa64k/assets

cd "${repo}"
export PYTHONPATH=.
source_file=${data_root}/longbook_qa_eng.jsonl
[[ -s ${source_file} ]] || { printf 'REFUSE: missing %s\n' "${source_file}" >&2; exit 1; }

if [[ ! -f ${assets}/manifest.json ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_natural_long \
    --benchmark infinitebench --model "${model}" --model-id kanana_1p5_8b \
    --data-root "${data_root}" --out "${assets}" \
    --scale 2 --lengths 65536 --rows-per-task 1000 \
    --task longbook_qa_eng --minimum-input-tokens 32769 \
    --maximum-input-tokens 65496
fi

"${python_bin}" - "${assets}/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
selected=int(x.get("summary",{}).get("selected_rows",0))
if x.get("status")!="COMPLETE" or selected < 1:
    raise SystemExit(f"QA64K assets invalid or empty: {x}")
print(json.dumps({"status":x["status"],"rows":selected,"summary":x["summary"]},sort_keys=True))
PY
