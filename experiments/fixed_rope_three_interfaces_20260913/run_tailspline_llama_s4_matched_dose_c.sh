#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
classic=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_matched_dose_c
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
table=${root}/tables/llama_s4_tailspline_dose_control.json

mkdir -p "${root}/tables" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${table}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${model}/config.json" \
    --method tailspline_dose_control \
    --scale 4 \
    --candidate-id llama3_8b_s4_tailspline_dose_control_c \
    --model-id meta_llama3_8b_instruct \
    --role control \
    --changed-variable interior_allocation_shape_at_fixed_total_log_displacement \
    --out "${table}"
fi

/root/miniconda3/bin/python - "${classic}/tables/tailspline.json" "${table}" <<'PY'
import json
import math
import sys

tailspline, control = (json.load(open(path)) for path in sys.argv[1:])
for value in (tailspline, control):
    if value["band_envelope"] != [18, 35] or float(value["gain"]).hex() != (1.0 + 0.1 * math.log(4.0)).hex():
        raise ValueError("matched-dose band/gain identity drift")
if abs(float(tailspline["sum_m"]) - float(control["sum_m"])) > 2e-6:
    raise ValueError("deployed FP32 TailSpline/C exponent sums are not matched")
if tailspline["table_sha256_float32"] == control["table_sha256_float32"]:
    raise ValueError("TailSpline and C unexpectedly share one table")
print(json.dumps({
    "status": "MATCHED_DOSE_C_READY_V1",
    "tailspline_sum_m": tailspline["sum_m"],
    "control_sum_m": control["sum_m"],
    "control_table_sha256_float32": control["table_sha256_float32"],
}))
PY

run=${root}/runs/dose_control_c
if [[ ! -f "${run}/status.json" ]] || ! /root/miniconda3/bin/python - "${run}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 390, "lm_rows": 138})
PY
then
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" \
    --model "${model}" \
    --arm Native \
    --extra-panel "${classic}/assets/full13/rows.jsonl" \
    --only-extra-panels \
    --length-cap 8192 --length-cap 16384 --length-cap 32768 \
    --lm-length-cap 8192 --lm-length-cap 16384 --lm-length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 2 \
    --static-table-json "${table}" \
    --table-label llama3_8b_s4_classic_dose_control_c \
    --out "${run}" \
    --execute >"${root}/logs/dose_control_c.log" 2>&1
fi

report=${root}/reports/tailspline_vs_dose_control_c_classic.json
if [[ ! -f "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report \
    --run "tailspline=${classic}/runs/tailspline" \
    --run "dose_control_c=${run}" \
    --receipt "tailspline=${classic}/tables/tailspline.json" \
    --receipt "dose_control_c=${table}" \
    --ppl-manifest "${classic}/assets/ppl46/manifest.json" \
    --candidate tailspline \
    --baseline dose_control_c \
    --out "${report}"
fi
printf 'MATCHED_DOSE_C_COMPLETE %s\n' "$(date -u +%FT%TZ)"
