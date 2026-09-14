#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
source_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_unified_full324
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_32k64k
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
panel=/root/autodl-tmp/band_mini_20260913/qwen_s2/frozen_324/screen.jsonl

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python - "${source_root}" "${panel}" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

root, panel = map(Path, sys.argv[1:])
status = json.loads((root / "runs/tailspline/status.json").read_text())
if status != {"status": "COMPLETE", "rows": 324, "lm_rows": 0}:
    raise ValueError(f"archived TailSpline arm is incomplete: {status}")
raw = root / "runs/tailspline/generations.jsonl"
rows = [json.loads(line) for line in raw.read_text().splitlines() if line]
counts = Counter((row["task"], int(row["length_cap"])) for row in rows)
tasks = sorted({row["task"] for row in rows})
expected = Counter({(task, length): 18 for task in tasks for length in (32768, 49152, 65536)})
if len(tasks) != 6 or counts != expected or len({row["prompt_sha256"] for row in rows}) != 324:
    raise ValueError("archived TailSpline raw is not the frozen Core-6 x 3 x 18 panel")
panel_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
if panel_hash != "22a0eb40d7abcd663f8887076da47d0ad81fdc967cb110c46399f4b8810c69e0":
    raise ValueError(f"Qwen panel hash drift: {panel_hash}")
for name, source in (("tailspline", "analytic:tailspline"), ("mrpro", "analytic:mrpro")):
    receipt = json.loads((root / f"tables/{name}.json").read_text())
    if (
        receipt.get("source") != source
        or receipt.get("band_envelope") != [23, 40]
        or receipt.get("gain") != 1.0693147180559945
        or receipt.get("scale") != 2.0
    ):
        raise ValueError(f"Qwen {name} table identity drift")
print(json.dumps({
    "status": "QWEN_TAILSPLINE_32K64K_REUSE_READY_V1",
    "reused_tailspline_rows": 216, "tasks": tasks, "panel_sha256": panel_hash,
}))
PY

run_dir="${experiment_root}/runs/mrpro"
if [[ ! -f "${run_dir}/status.json" ]] || ! /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 216, "lm_rows": 0})
PY
then
  printf 'START mrpro %s\n' "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --length-cap 65536 \
    --prefill-chunk-size 8192 \
    --batch-size 2 \
    --static-table-json "${source_root}/tables/mrpro.json" \
    --table-label qwen25_3b_s2_32k64k_mrpro \
    --out "${run_dir}" \
    --execute >"${experiment_root}/logs/mrpro.log" 2>&1
  printf 'COMPLETE mrpro %s\n' "$(date -u +%FT%TZ)"
fi

validation="${experiment_root}/reports/validation.json"
/root/miniconda3/bin/python - "${source_root}" "${experiment_root}" "${validation}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_root, root, output = map(Path, sys.argv[1:])
candidate_path = source_root / "runs/tailspline/generations.jsonl"
baseline_path = root / "runs/mrpro/generations.jsonl"
candidate = [json.loads(line) for line in candidate_path.read_text().splitlines() if line]
candidate = [row for row in candidate if int(row["length_cap"]) in {32768, 65536}]
baseline = [json.loads(line) for line in baseline_path.read_text().splitlines() if line]
candidate_prompts = {row["prompt_sha256"] for row in candidate}
baseline_prompts = {row["prompt_sha256"] for row in baseline}
if len(candidate) != 216 or len(baseline) != 216 or candidate_prompts != baseline_prompts:
    raise ValueError("Qwen 32K/64K arms are not exactly paired on 216 prompts")
contract = json.loads((root / "runs/mrpro/contract.json").read_text())
receipt = json.loads((source_root / "tables/mrpro.json").read_text())
if contract.get("static_table") != receipt.get("table"):
    raise ValueError("Qwen MrPro installed table differs from frozen receipt")
record = {
    "status": "QWEN_TAILSPLINE_MRPRO_32K64K_VALIDATED_V1",
    "tasks": sorted({row["task"] for row in candidate}),
    "lengths": [32768, 65536], "rows_per_arm": 216,
    "candidate_raw": str(candidate_path),
    "candidate_raw_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    "baseline_raw": str(baseline_path),
    "baseline_raw_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    "table_sha256_float32": {
        "tailspline": json.loads((source_root / "tables/tailspline.json").read_text())["table_sha256_float32"],
        "mrpro": receipt["table_sha256_float32"],
    },
}
temporary = output.with_name(output.name + ".incomplete")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(record, sort_keys=True))
PY

report="${experiment_root}/reports/tailspline_vs_mrpro_32k64k.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source tailspline="${source_root}/runs/tailspline/generations.jsonl" \
    --source mrpro="${run_dir}/generations.jsonl" \
    --candidate tailspline \
    --baseline mrpro \
    --length 32768 \
    --length 65536 \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
