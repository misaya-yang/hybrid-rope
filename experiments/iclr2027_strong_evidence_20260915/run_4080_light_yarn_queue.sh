#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
root=${plan}/official_yarn_4080_light

if [[ "${1:-}" != "--execute" ]]; then
  echo 'PLAN_ONLY: Llama/OLMo <=32GB official-static-YaRN queue.'
  echo 'Runs Llama NIAH-8x200+PPL46, Llama/OLMo Natural-QA631, and Llama/OLMo Full-13x10.'
  echo 'Qwen/GLM 128K remain on the 96GB Pro6000 host.'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${root}/logs"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No devices/d' | head -1 || true)
[[ -n "${gpu_name}" ]] || { echo 'REFUSE: no CUDA GPU is attached' >&2; exit 75; }

ensure_yarn_table() {
  local name=$1 model=$2 model_id=$3 out=${plan}/official_yarn_quick/$1/tables/yarn.json
  if [[ ! -f "${out}" ]]; then
    mkdir -p "$(dirname "${out}")"
    "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
      --config "${model}/config.json" --method yarn --scale 4 \
      --candidate-id "${name}_official_static_yarn" --model-id "${model_id}" \
      --role baseline --changed-variable internal_frequency_allocation --out "${out}"
  fi
}

ensure_yarn_table llama3_8b_s4_32k \
  /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct llama3_8b
ensure_yarn_table olmo2_1b_s4_16k \
  /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct olmo2_1b

# Synthetic retrieval and PPL use the complete existing evidence pool.
bash experiments/iclr2027_strong_evidence_20260915/run_llama_yarn_niah200_ppl46.sh \
  >"${root}/logs/llama_niah200_ppl46.log" 2>&1

# The legacy Llama launcher has no lock of its own; hold the shared lock in a
# subshell so it cannot overlap another GPU owner.
(
  exec 9>"${gpu_lock}"
  flock -n 9 || { echo 'REFUSE: GPU lock is owned' >&2; exit 73; }
  ROPE_PLAN_ROOT="${plan}" ROPE_PYTHON="${python_bin}" \
    bash experiments/iclr2027_three_track_sprint_20260915/run_naturalqa_yarn.sh --execute
)
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa \
  --condition llama3_8b --plan-root "${plan}" --python "${python_bin}" --finalize-only \
  >"${root}/logs/naturalqa_llama3_8b.log" 2>&1

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa \
  --condition olmo2_1b --plan-root "${plan}" --python "${python_bin}" --execute \
  >"${root}/logs/naturalqa_olmo2_1b.log" 2>&1

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.four_model_yarn_full13 \
  --plan "${plan}" --python "${python_bin}" \
  --condition llama3_8b_s4_32k --condition olmo2_1b_s4_16k --execute \
  >"${root}/logs/full13_llama_olmo.log" 2>&1

"${python_bin}" - "${plan}" "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan,root=map(Path,sys.argv[1:])
owners=[]
for path in (
 plan/'official_yarn_llama_niah200_ppl46/complete.json',
 plan/'tailspline_llama_s4_naturalqa631_yarn_a1/complete.json',
 plan/'tailspline_olmo_s4_naturalqa631_yarn_a1/complete.json',
 plan/'official_yarn_full13/complete__llama3_8b_s4_32k__olmo2_1b_s4_16k.json',
):
 if not path.is_file(): raise FileNotFoundError(path)
 owners.append({'path':str(path.relative_to(plan)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out={'status':'RTX4080_LLAMA_OLMO_YARN_QUEUE_COMPLETE_V1','conditions':['llama3_8b_s4_32k','olmo2_1b_s4_16k'],'owners':owners}
p=root/'complete.json';t=p.with_name(p.name+'.incomplete');t.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');t.replace(p);print(json.dumps(out,sort_keys=True))
PY
