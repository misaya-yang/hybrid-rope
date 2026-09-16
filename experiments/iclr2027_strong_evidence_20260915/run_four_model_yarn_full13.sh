#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${gpu_lock}"; flock -n 9 || { echo "REFUSE: GPU lock is owned" >&2; exit 73; }

# Frozen transfer set: Llama-3-8B, Qwen2.5-3B, OLMo-2-1B, GLM-4-9B.
# Qwen2.5-1.5B is not scheduled.
conditions=(
  "llama3_8b_s4_32k|/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct|32768|8192|${plan}/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl|${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json|${plan}/official_yarn_quick/llama3_8b_s4_32k/tables/yarn.json"
  "qwen25_3b_s4_128k|/root/autodl-tmp/rope_qwen_baseline_20260907/model|131072|65536|${plan}/tailspline_qwen25_s4_64k128k_clean/assets/panels/131072/inputs.jsonl|${plan}/four_model_128k_extreme/qwen25_3b_128k/ppl/manifest.json|${plan}/four_model_128k_extreme/qwen25_3b_128k/official_yarn/tables/yarn.json"
  "olmo2_1b_s4_16k|/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct|16384|8192|${plan}/tailspline_olmo_s4_16k_ruler200_clean/assets/panels/16384/inputs.jsonl|${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json|${plan}/official_yarn_quick/olmo2_1b_s4_16k/tables/yarn.json"
  "glm4_9b_s4_128k|/root/models/GLM-4-9B-0414|131072|65536|${plan}/glm4_9b_s4_128k/assets10/panels/131072/inputs.jsonl|${plan}/glm4_9b_s4_128k/ppl5/manifest.json|${plan}/official_yarn_quick/glm4_9b_s4_128k/tables/yarn.json"
)

for spec in "${conditions[@]}"; do
  IFS='|' read -r name model target chunk panel data table <<<"${spec}"
  root=${plan}/official_yarn_full13/${name}; mkdir -p "${root}/logs"
  if [[ ! -f "${root}/run/status.json" ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${data}" --model "${model}" --arm Native --extra-panel "${panel}" \
      --only-extra-panels --skip-lm --length-cap "${target}" --limit-per-cell 10 \
      --prefill-chunk-size "${chunk}" --batch-size 1 --longest-first \
      --static-table-json "${table}" --table-label "${name}_official_static_yarn_full13" \
      --out "${root}/run" --execute >"${root}/logs/run.log" 2>&1
  fi
  "${python_bin}" - "${root}/run/status.json" "${name}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d!={'status':'COMPLETE','rows':130,'lm_rows':0}: raise SystemExit(f'{sys.argv[2]}: {d}')
PY
done

"${python_bin}" - "${plan}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])/'official_yarn_full13';names=('llama3_8b_s4_32k','qwen25_3b_s4_128k','olmo2_1b_s4_16k','glm4_9b_s4_128k');owners=[]
for name in names:
 p=root/name/'run/generations.jsonl';owners.append({'condition':name,'rows':130,'generations_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'status':'FOUR_MODEL_YARN_FULL13_10_COMPLETE_V1','models':['Llama-3-8B-Instruct','Qwen2.5-3B-Instruct','OLMo-2-0425-1B-Instruct','GLM-4-9B-0414'],'excluded_models':['Qwen2.5-1.5B-Instruct'],'owners':owners}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
