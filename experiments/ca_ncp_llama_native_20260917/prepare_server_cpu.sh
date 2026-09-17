#!/usr/bin/env bash
set -euo pipefail

repo=${CA_NCP_REPO:-/root/autodl-tmp/hybrid-rope}
root=${CA_NCP_LLAMA_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_llama_native_20260917}
model=${CA_NCP_LLAMA_MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
source=${CA_NCP_LLAMA_SOURCE:-/root/autodl-tmp/today_rope_plan_20260914/strong_evidence/llama_s4_clean_native_x5}
corpus_manifest=${CA_NCP_CORPUS_MANIFEST:-/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_native_20260917/source_pool/formal_pg19_proofpile_train.jsonl}
python_bin=${CA_NCP_PYTHON:-/root/miniconda3/bin/python}

cd "${repo}"
"${python_bin}" -m experiments.ca_ncp_llama_native_20260917.prepare \
  --model "${model}" \
  --runtime-matrix experiments/native_enhancement_oral_20260915/reports/runtime_matrix.json \
  --source-assets "${source}/assets/manifest.json" \
  --source-native-run "${source}/runs/native" \
  --out "${root}"

"${python_bin}" -m experiments.ca_ncp_native_20260917.prepare_statistics \
  --model "${model}" --corpus-manifest "${corpus_manifest}" \
  --fit-documents 32 --report-documents 8 --pairs-per-document 512 \
  --native-length 8192 --rotary-pairs 64 --method-id CA_NCP_LLAMA3_8B_NATIVE_V1 \
  --out "${root}/assets/statistics"

"${python_bin}" - "${root}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
payload={
  "status":"CA_NCP_LLAMA_CPU_READY",
  "gpu_execution":False,
  "model":"Meta-Llama-3-8B-Instruct",
  "native_length":8192,
  "new_gpu_arms":["C0","P0","N1","P1"],
  "reused_gpu_arms":["N0"],
  "next_step":"Run only if the OLMo CA-NCP gate is positive: capture statistics, solve alignment, then execute the five-arm runner with the Llama source manifest.",
}
path=root/"CPU_READINESS.json"; tmp=path.with_name(path.name+".incomplete")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
print(json.dumps(payload,sort_keys=True))
PY
