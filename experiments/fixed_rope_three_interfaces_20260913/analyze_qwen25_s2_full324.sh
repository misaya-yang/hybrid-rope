#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/qwen25_s2_full324
table_dir=/root/autodl-tmp/cross_model_mix075_20260914/tables
report="${experiment_root}/reports/mix075_vs_mrpro_yarn_bm_32k48k64k.json"
validation="${experiment_root}/reports/validation.json"

mkdir -p "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.

/root/miniconda3/bin/python - "${experiment_root}" "${table_dir}" "${validation}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

experiment_root = Path(sys.argv[1])
table_dir = Path(sys.argv[2])
validation_path = Path(sys.argv[3])
receipts = {
    "mix075": table_dir / "mix075.json",
    "mrpro": table_dir / "mrpro.json",
    "yarn": table_dir / "yarn.json",
    "bm": table_dir / "bm_band22_39_midgain.json",
}
prompt_sets = {}
records = {}
for arm, receipt_path in receipts.items():
    run = experiment_root / "runs" / f"{arm}_b2"
    status = json.loads((run / "status.json").read_text())
    if status != {"status": "COMPLETE", "rows": 324, "lm_rows": 0}:
        raise ValueError(f"{arm} is not a complete 324-row generation run: {status}")
    contract = json.loads((run / "contract.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    installed = receipt.get("table", receipt)
    if contract.get("static_table") != installed:
        raise ValueError(f"{arm} contract table differs from its frozen receipt")
    if contract.get("batch_size") != 2 or contract.get("prefill_chunk_size") != 8192:
        raise ValueError(f"{arm} execution settings differ from the frozen queue")
    rows = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line.strip()]
    prompts = {row.get("prompt_sha256") for row in rows}
    if len(rows) != 324 or len(prompts) != 324 or None in prompts:
        raise ValueError(f"{arm} row or prompt coverage is invalid")
    if any(not isinstance(row.get("generated_ids"), list) or not row["generated_ids"] for row in rows):
        raise ValueError(f"{arm} has a missing generated token sequence")
    if any(not math.isfinite(float(row.get("ruler_official_score", float("nan")))) for row in rows):
        raise ValueError(f"{arm} has a missing official score")
    prompt_sets[arm] = prompts
    raw_path = run / "generations.jsonl"
    records[arm] = {
        "rows": len(rows),
        "raw_path": str(raw_path),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "table_path": str(receipt_path),
        "table_sha256_float32": receipt.get("table_sha256_float32"),
        "gain": installed["gain"],
    }
reference = prompt_sets["mix075"]
if any(prompts != reference for prompts in prompt_sets.values()):
    raise ValueError("Qwen S2 arms are not paired on exactly the same prompts")
output = {
    "status": "QWEN25_S2_FULL324_VALIDATED_V1",
    "paired_prompts": len(reference),
    "records": records,
    "excluded_partial_run": str(experiment_root / "runs" / "mix075"),
    "excluded_partial_reason": "batch4 OOM at 114/324; preserved but not merged",
}
temporary = validation_path.with_suffix(".json.incomplete")
temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
temporary.replace(validation_path)
print(json.dumps({"status": output["status"], "paired_prompts": len(reference)}))
PY

if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source mix075="${experiment_root}/runs/mix075_b2/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_b2/generations.jsonl" \
    --source yarn="${experiment_root}/runs/yarn_b2/generations.jsonl" \
    --source bm="${experiment_root}/runs/bm_b2/generations.jsonl" \
    --candidate mix075 \
    --baseline mrpro \
    --baseline yarn \
    --baseline bm \
    --out "${report}"
fi
