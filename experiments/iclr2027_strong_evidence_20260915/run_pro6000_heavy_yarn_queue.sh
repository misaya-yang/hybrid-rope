#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
module=experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa
root=${plan}/official_yarn_pro6000_heavy
mkdir -p "${root}/logs"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Only the two 128K conditions stay on the 96GB Blackwell host.
for condition in qwen25_3b glm4_9b; do
  "${python_bin}" -m "${module}" --condition "${condition}" \
    --plan-root "${plan}" --python "${python_bin}" --execute \
    >>"${root}/logs/naturalqa_${condition}.log" 2>&1
done

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.four_model_yarn_full13 \
  --plan "${plan}" --python "${python_bin}" \
  --condition qwen25_3b_s4_128k --condition glm4_9b_s4_128k --execute \
  >>"${root}/logs/full13_qwen_glm.log" 2>&1

"${python_bin}" - "${plan}" "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan,root=map(Path,sys.argv[1:])
owners=[]
for path in (
 plan/'four_model_128k_extreme/qwen25_3b_128k/infinitebench_en_qa_yarn_a1/complete.json',
 plan/'glm4_9b_s4_128k/en_qa_yarn_a1/complete.json',
 plan/'official_yarn_full13/complete__qwen25_3b_s4_128k__glm4_9b_s4_128k.json',
):
 if not path.is_file(): raise FileNotFoundError(path)
 owners.append({'path':str(path.relative_to(plan)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out={'status':'PRO6000_QWEN_GLM_128K_YARN_QUEUE_COMPLETE_V1','conditions':['qwen25_3b_s4_128k','glm4_9b_s4_128k'],'owners':owners}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
