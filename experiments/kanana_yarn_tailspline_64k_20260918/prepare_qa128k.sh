#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
data_root=${INFINITEBENCH_ROOT:-/root/autodl-tmp/InfiniteBench/data}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
assets=${root}/qa128k/assets

cd "${repo}"
export PYTHONPATH=.
mkdir -p "${root}"
"${python_bin}" - "${root}/minimal_eval_manifest.json" <<'PY'
import json,os,sys
path=sys.argv[1]
expected="{}\n"
if os.path.exists(path) and json.load(open(path)) != {}:
    raise SystemExit("minimal evaluation manifest drift")
if not os.path.exists(path):
    temporary=path+".incomplete"
    open(temporary,"w").write(expected)
    os.replace(temporary,path)
PY
source_file=${data_root}/longbook_qa_eng.jsonl
[[ -s ${source_file} ]] || { printf 'REFUSE: missing %s\n' "${source_file}" >&2; exit 1; }

if [[ ! -f ${assets}/manifest.json ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_natural_long \
    --benchmark infinitebench --model "${model}" --model-id kanana_1p5_8b \
    --data-root "${data_root}" --out "${assets}" \
    --scale 4 --lengths 131072 --rows-per-task 1000 \
    --task longbook_qa_eng --minimum-input-tokens 32769 \
    --maximum-input-tokens 131032
fi

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.prepare_qa128k_tables \
  --model "${model}" --root "${root}"

"${python_bin}" - "${assets}/manifest.json" "${assets}/inputs.jsonl" <<'PY'
import hashlib,json,sys
manifest=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
expected_sha="7f53a3aabd5445fa29f168914f13569c348406cfc88a3e495f3f2c13a0966ca4"
source=manifest.get("source_files",{}).get("longbook_qa_eng",{})
clusters={row["source_cluster_id"] for row in rows}
lengths=[int(row["input_tokens"]) for row in rows]
if (
    manifest.get("status")!="COMPLETE" or len(rows)!=118 or len(clusters)!=23
    or min(lengths)!=69680 or max(lengths)!=130852
    or source.get("sha256")!=expected_sha
    or hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest()!=manifest.get("inputs_sha256")
    or any(int(row["length_cap"])!=131072 or len(row["prompt_ids"])+40>131072 for row in rows)
):
    raise SystemExit("Kanana QA128K frozen asset identity drift")
print(json.dumps({"status":"COMPLETE","rows":len(rows),"clusters":len(clusters),"range":[min(lengths),max(lengths)],"inputs_sha256":manifest["inputs_sha256"]},sort_keys=True))
PY
