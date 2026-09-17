#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/native_research_20260916
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
python_bin=/root/miniconda3/bin/python
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
longbench_root=/root/autodl-tmp/hybrid-rope-target-free-real-data-v3/longbench
old_qa=${plan}/tailspline_olmo_s4_naturalqa631/assets/inputs.jsonl
old_ppl=${plan}/tailspline_olmo_s4_classic/assets/ppl46/manifest.json

mkdir -p "${root}/assets" "${root}/reports" "${root}/logs"
cd "${repo}"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=''
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mechanism=${root}/assets/mechanism_v2
capture=${root}/assets/capture96
ruler=${root}/assets/ruler_confirm_13x100
qa=${root}/assets/naturalqa_3x80
lm=${root}/assets/lm128

prepare_existing() {
  if [[ ! -f "${root}/reports/existing_five_arm_reanalysis.json" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.reanalyze_existing \
      --run native="${plan}/olmo_native_halfturn_phase/runs/native/generations.jsonl" \
      --run halfturn="${plan}/olmo_native_halfturn_phase/runs/contract/generations.jsonl" \
      --run reverse="${plan}/olmo_native_halfturn_phase/runs/reverse/generations.jsonl" \
      --run v1="${plan}/olmo_native_halfturn_phase/runs/v1/generations.jsonl" \
      --run ncp="${plan}/olmo_native_contrastive_proximal/runs/ncp/generations.jsonl" \
      --out "${root}/reports/existing_five_arm_reanalysis.json" \
      >"${root}/logs/reanalyze_existing.log" 2>&1
  fi
}

prepare_mechanism_and_capture() {
  if [[ ! -f "${mechanism}/manifest.json" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.prepare \
      --model "${model}" --out "${mechanism}" \
      >"${root}/logs/prepare_mechanism.log" 2>&1
  fi
  if [[ ! -f "${capture}/manifest.json" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.prepare_capture \
      --panel "${mechanism}/inputs.jsonl" --panel-manifest "${mechanism}/manifest.json" \
      --model "${model}" --out "${capture}" \
      >"${root}/logs/prepare_capture96.log" 2>&1
  fi
}

prepare_ruler() {
  if [[ ! -f "${ruler}/manifest.json" ]]; then
    "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer \
      --model "${model}" --model-id olmo2_1b_native_confirm --data-root "${upstream}" \
      --out "${ruler}" --scale 1 --lengths 4096 --rows-per-task 100 \
      --seed 20261216 --qa-offset 6000 \
      >"${root}/logs/prepare_ruler_confirm.log" 2>&1
  fi
}

prepare_qa() {
  if [[ ! -f "${qa}/manifest.json" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.prepare_native_naturalqa \
      --archive "${longbench_root}/data.zip" --config-root "${longbench_root}/official_config" \
      --download-config --model "${model}" --exclude-panel "${old_qa}" --out "${qa}" \
      >"${root}/logs/prepare_naturalqa.log" 2>&1
  fi
}

prepare_lm() {
  if [[ ! -f "${lm}/manifest.json" ]]; then
    "${python_bin}" -m experiments.native_enhancement_oral_20260915.prepare_native_lm \
      --pg19-parquet /root/autodl-tmp/longtext/pg19test.parquet \
      --proofpile-root /root/autodl-tmp/nongeometric_screen_20260909/long_sources \
      --exclude-manifest "${old_ppl}" --model "${model}" --out "${lm}" \
      >"${root}/logs/prepare_lm128.log" 2>&1
  fi
}

prepare_existing & p1=$!
prepare_mechanism_and_capture & p2=$!
prepare_ruler & p3=$!
prepare_qa & p4=$!
prepare_lm & p5=$!
failed=0
for pid in "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != 0 ]]; then
  printf '%s\n' 'Native CPU preparation failed; inspect the five preparation logs.' >&2
  exit 1
fi

"${python_bin}" -m experiments.native_enhancement_oral_20260915.lm_context canary \
  >"${root}/logs/lm_context_canary.json"

"${python_bin}" - "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
required={
 "existing_reanalysis":root/'reports/existing_five_arm_reanalysis.json',
 "mechanism":root/'assets/mechanism_v2/manifest.json',
 "capture":root/'assets/capture96/manifest.json',
 "ruler":root/'assets/ruler_confirm_13x100/manifest.json',
 "naturalqa":root/'assets/naturalqa_3x80/manifest.json',
 "lm":root/'assets/lm128/manifest.json',
}
if any(not path.is_file() for path in required.values()):
 raise SystemExit('native CPU preparation is incomplete')
receipt={
 'status':'NATIVE_RESEARCH_CPU_PREPARED_V1',
 'gpu_execution':False,
 'artifacts':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
              for name,path in required.items()},
 'next_entry':'experiments/native_enhancement_oral_20260915/run_server_gpu.sh --execute',
}
path=root/'cpu_ready.json';tmp=path.with_name(path.name+'.incomplete')
tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');tmp.replace(path)
print(json.dumps(receipt,sort_keys=True))
PY
