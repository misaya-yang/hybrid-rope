#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
tasks=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery)
# Frozen transfer set: Llama-3-8B, Qwen2.5-3B, OLMo-2-1B, GLM-4-9B.
# Qwen2.5-1.5B is historical evidence only and is deliberately excluded here.
qwen_report=${plan}/four_model_128k_extreme/qwen25_3b_128k/official_yarn/report.json

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${gpu_lock}"; flock -n 9 || { echo "REFUSE: GPU lock is owned" >&2; exit 73; }

conditions=(
  "llama3_8b_s4_32k|/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct|llama3_8b|4|32768|8192|0|${plan}/tailspline_llama_s4_32k_ruler200_clean/assets/inputs.jsonl|${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json"
  "olmo2_1b_s4_16k|/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct|olmo2_1b|4|16384|8192|0|${plan}/tailspline_olmo_s4_16k_ruler200_clean/assets/panels/16384/inputs.jsonl|${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json"
  "glm4_9b_s4_128k|/root/models/GLM-4-9B-0414|glm4_9b_0414|4|131072|65536|0|${plan}/glm4_9b_s4_128k/assets/panels/131072/inputs.jsonl|${plan}/glm4_9b_s4_128k/ppl5/manifest.json"
)

for spec in "${conditions[@]}"; do
  IFS='|' read -r name model model_id scale target gen_chunk lm_chunk panel ppl <<<"${spec}"
  root=${plan}/official_yarn_quick/${name}; mkdir -p "${root}/tables" "${root}/logs"
  table=${root}/tables/yarn.json
  if [[ ! -f "${table}" ]]; then
    "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
      --config "${model}/config.json" --method yarn --scale "${scale}" \
      --candidate-id "${name}_official_static_yarn" --model-id "${model_id}" --role baseline \
      --changed-variable internal_frequency_allocation --out "${table}" >"${root}/logs/table.log"
  fi
  if [[ ! -f "${root}/run/status.json" ]]; then
    command=("${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval
      --data "${ppl}" --model "${model}" --arm Native --extra-panel "${panel}" --only-extra-panels
      --length-cap "${target}" --lm-length-cap "${target}" --lm-limit-documents 5 --limit-per-cell 5
      --prefill-chunk-size "${gen_chunk}" --lm-prefill-chunk-size "${lm_chunk}" --batch-size 1 --longest-first
      --static-table-json "${table}" --table-label "${name}_official_static_yarn" --out "${root}/run" --execute)
    for task in "${tasks[@]}"; do command+=(--task "${task}"); done
    "${command[@]}" >"${root}/logs/run.log" 2>&1
  fi
done

"${python_bin}" - "${plan}" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan=Path(sys.argv[1]);root=plan/'official_yarn_quick';names=('llama3_8b_s4_32k','olmo2_1b_s4_16k','glm4_9b_s4_128k');owners=[]
for name in names:
 p=root/name/'run/status.json';d=json.loads(p.read_text())
 if d!={'status':'COMPLETE','rows':40,'lm_rows':5}: raise SystemExit(f'{name}: {d}')
 owners.append({'condition':name,'status_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
qwen=plan/'four_model_128k_extreme/qwen25_3b_128k/official_yarn/report.json'
if not qwen.is_file(): raise SystemExit(f'missing Qwen2.5-3B YaRN report: {qwen}')
owners.insert(1,{'condition':'qwen25_3b_s4_128k','report_sha256':hashlib.sha256(qwen.read_bytes()).hexdigest(),'reused':True})
out={'status':'FOUR_MODEL_YARN_QUICK_COMPLETE_V1','models':['Llama-3-8B-Instruct','Qwen2.5-3B-Instruct','OLMo-2-0425-1B-Instruct','GLM-4-9B-0414'],'excluded_models':['Qwen2.5-1.5B-Instruct'],'owners':owners}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
