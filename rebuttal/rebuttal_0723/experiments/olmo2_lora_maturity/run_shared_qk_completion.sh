#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ASSET_BASE SCREEN_CONTROLLER_PID" >&2
  exit 2
fi

asset_base=$1
screen_pid=$2
script_path=$(realpath "${BASH_SOURCE[0]}")
code_root=${CODE_ROOT:-"$asset_base/code_general_qk_20260727"}
python_bin=${PYTHON_BIN:-python3}
checkpoint="$asset_base/models/OLMo-2-0425-1B-Instruct"
checkpoint_receipt="$asset_base/receipts/instruct_4k_conversion_ready.json"
downstream_ready="$asset_base/receipts/general_qk_downstream_eval_ready_s20260727.json"
qa_data="$asset_base/data/2wiki_phase_4k_s20260728"
ruler_data="$asset_base/data/ruler_full_merged_n20_s20260802"
runs="$asset_base/runs"
registered_root="$runs/general_qk_downstream_eval_s20260727"
completion_receipt="$runs/general_qk_shared_completion_receipt.json"
strict_validation_receipt="$runs/general_qk_strict_validation_receipt_s20260730.json"
validator_path="$code_root/rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/validate_qk_eval_result.py"

evq_shared_run="$runs/general_qk_evq_shared_task_phase500_s20260730"
native_shared_run="$runs/general_qk_native_shared_task_phase500_s20260730"
evq_generic_run="$runs/general_qk_evq_600_s20260727"
native_generic_run="$runs/general_qk_native_600_s20260727"

rm -f -- \
  "$completion_receipt" \
  "$completion_receipt.incomplete" \
  "$strict_validation_receipt" \
  "$strict_validation_receipt.incomplete"

for required in \
  "$checkpoint_receipt" \
  "$downstream_ready" \
  "$evq_generic_run/adapter.pt" \
  "$native_generic_run/adapter.pt"; do
  test -f "$required"
done

screen_qa_evq_root="$runs/screen_2wiki_general_qk_evq_shared_phase500_s20260730"
screen_qa_native_root="$runs/screen_2wiki_general_qk_native_shared_phase500_s20260730"
screen_ruler_evq_root="$runs/screen_ruler_general_qk_evq_shared_phase500_s20260730"
screen_ruler_native_root="$runs/screen_ruler_general_qk_native_shared_phase500_s20260730"
screen_roots=(
  "$screen_qa_evq_root"
  "$screen_qa_native_root"
  "$screen_ruler_evq_root"
  "$screen_ruler_native_root"
)
gate_json="$runs/general_qk_shared_task_screen_gate_s20260730.json"

screen_complete() {
  local root
  for root in "${screen_roots[@]}"; do
    [[ -f "$root/results.json" ]] || return 1
  done
}

if kill -0 "$screen_pid" 2>/dev/null; then
  screen_cmdline=$(tr '\0' ' ' <"/proc/$screen_pid/cmdline")
  [[ "$screen_cmdline" == *general_qk_shared_task* ]]
fi

wait_seconds=0
while ! screen_complete && kill -0 "$screen_pid" 2>/dev/null; do
  if (( wait_seconds >= 7200 )); then
    echo "SCREEN_CONTROLLER_TIMEOUT pid=$screen_pid" >&2
    exit 3
  fi
  sleep 3
  wait_seconds=$((wait_seconds + 3))
done

cd "$code_root"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$code_root"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t identities < <(
  "$python_bin" - "$evq_shared_run/results.json" \
    "$native_shared_run/results.json" "$downstream_ready" <<'PY'
import json
import pathlib
import sys

evq_path, native_path, ready_path = map(pathlib.Path, sys.argv[1:])
evq = json.loads(evq_path.read_text())
native = json.loads(native_path.read_text())
ready = json.loads(ready_path.read_text())
if evq.get("status") != "OLMO2_4K_PHASE_ADAPTATION_COMPLETE_V1":
    raise RuntimeError("EVQ shared training result is incomplete")
if native.get("status") != "OLMO2_4K_PHASE_ADAPTATION_COMPLETE_V1":
    raise RuntimeError("Native shared training result is incomplete")
if evq.get("checkpoint_sha256") != native.get("checkpoint_sha256"):
    raise RuntimeError("shared checkpoint identity drift")
for key in (
    "selection_stream_sha256",
    "position_stream_sha256",
    "exposure_stream_sha256",
    "family_steps",
    "position_bucket_counts",
    "task_position_bucket_counts",
    "processed_input_tokens",
):
    if evq["training"].get(key) != native["training"].get(key):
        raise RuntimeError(f"shared training matched field drift: {key}")
gate = ready["matched_training_gate"]
if (
    evq["adapter_metadata"]["parent_adapter_sha256"]
    != gate["evq_adapter_sha256"]
    or native["adapter_metadata"]["parent_adapter_sha256"]
    != gate["native_adapter_sha256"]
):
    raise RuntimeError("shared parent adapter identity drift")
print(evq["checkpoint_sha256"])
print(evq["adapter_sha256"])
print(native["adapter_sha256"])
print(gate["evq_adapter_sha256"])
print(gate["native_adapter_sha256"])
print(ready["inputs"]["two_wiki_manifest_sha256"])
print(ready["inputs"]["ruler_manifest_sha256"])
print(ready["bound_code_sha256"]["two_wiki_evaluator"])
print(ready["bound_code_sha256"]["ruler_evaluator"])
PY
)

if [[ ${#identities[@]} -ne 9 ]]; then
  echo "identity extraction failed" >&2
  exit 4
fi
checkpoint_sha=${identities[0]}
evq_shared_sha=${identities[1]}
native_shared_sha=${identities[2]}
evq_generic_sha=${identities[3]}
native_generic_sha=${identities[4]}
qa_manifest_sha=${identities[5]}
ruler_manifest_sha=${identities[6]}
qa_evaluator_sha=${identities[7]}
ruler_evaluator_sha=${identities[8]}

validator=(
  "$python_bin" -m
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.validate_qk_eval_result
)

validate_2wiki() {
  local root=$1
  local adapter_sha=$2
  local training_frequency=$3
  local frequency=$4
  local limit=$5
  local role=$6
  local yarn_factor=$7
  shift 7
  local lengths=("$@")
  local command=(
    "${validator[@]}"
    --root "$root"
    --benchmark 2wiki
    --checkpoint-sha256 "$checkpoint_sha"
    --adapter-sha256 "$adapter_sha"
    --adapter-training-frequency "$training_frequency"
    --frequency "$frequency"
    --lengths "${lengths[@]}"
    --limit "$limit"
    --data-manifest-sha256 "$qa_manifest_sha"
    --evaluator-sha256 "$qa_evaluator_sha"
    --adaptation qk_answer
    --rank 64
    --alpha 128
    --expected-role "$role"
    --fill-to-budget
  )
  if [[ "$yarn_factor" != none ]]; then
    command+=(--yarn-factor "$yarn_factor")
  fi
  "${command[@]}"
}

validate_ruler() {
  local root=$1
  local adapter_sha=$2
  local training_frequency=$3
  local frequency=$4
  local limit=$5
  local yarn_factor=$6
  shift 6
  local lengths=("$@")
  local command=(
    "${validator[@]}"
    --root "$root"
    --benchmark ruler
    --checkpoint-sha256 "$checkpoint_sha"
    --adapter-sha256 "$adapter_sha"
    --adapter-training-frequency "$training_frequency"
    --frequency "$frequency"
    --lengths "${lengths[@]}"
    --limit "$limit"
    --data-manifest-sha256 "$ruler_manifest_sha"
    --evaluator-sha256 "$ruler_evaluator_sha"
    --adaptation qk_answer
    --rank 64
    --alpha 128
  )
  if [[ "$yarn_factor" != none ]]; then
    command+=(--yarn-factor "$yarn_factor")
  fi
  "${command[@]}"
}

ensure_2wiki() {
  local root=$1
  local adapter=$2
  local adapter_sha=$3
  local training_frequency=$4
  local frequency=$5
  local limit=$6
  local role=$7
  local yarn_factor=$8
  shift 8
  local lengths=("$@")
  if [[ ! -f "$root/results.json" ]]; then
    echo "START $root"
    local command=(
      "$python_bin" -m
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_2wiki_phase_adaptation
      --checkpoint "$checkpoint"
      --checkpoint-ready-receipt "$checkpoint_receipt"
      --data-root "$qa_data"
      --adapter "$adapter"
      --output "$root"
      --role "$role"
      --frequency "$frequency"
      --adaptation qk_answer
      --rank 64
      --alpha 128
      --budgets "${lengths[@]}"
      --limit "$limit"
      --fill-to-budget
    )
    if [[ "$yarn_factor" != none ]]; then
      command+=(--yarn-factor "$yarn_factor")
    fi
    "${command[@]}"
  fi
  validate_2wiki \
    "$root" "$adapter_sha" "$training_frequency" "$frequency" \
    "$limit" "$role" "$yarn_factor" "${lengths[@]}"
  echo "VALID_COMPLETE $root"
}

ensure_ruler() {
  local root=$1
  local adapter=$2
  local adapter_sha=$3
  local training_frequency=$4
  local frequency=$5
  local limit=$6
  local yarn_factor=$7
  shift 7
  local lengths=("$@")
  if [[ ! -f "$root/results.json" ]]; then
    echo "START $root"
    local command=(
      "$python_bin" -m
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer
      --checkpoint "$checkpoint"
      --ready-receipt "$checkpoint_receipt"
      --data-root "$ruler_data"
      --adapter "$adapter"
      --output "$root"
      --frequency "$frequency"
      --adaptation qk_answer
      --rank 64
      --alpha 128
      --lengths "${lengths[@]}"
      --limit-per-cell "$limit"
    )
    if [[ "$yarn_factor" != none ]]; then
      command+=(--yarn-factor "$yarn_factor")
    fi
    "${command[@]}"
  fi
  validate_ruler \
    "$root" "$adapter_sha" "$training_frequency" "$frequency" \
    "$limit" "$yarn_factor" "${lengths[@]}"
  echo "VALID_COMPLETE $root"
}

shared_gate_passed=0
if screen_complete; then
  validate_2wiki \
    "$screen_qa_evq_root" "$evq_shared_sha" evq evq 50 \
    evq_shared none 4096 8192
  validate_2wiki \
    "$screen_qa_native_root" "$native_shared_sha" native native 50 \
    native_shared none 4096 8192
  validate_ruler \
    "$screen_ruler_evq_root" "$evq_shared_sha" evq evq 2 none \
    4096 8192
  validate_ruler \
    "$screen_ruler_native_root" "$native_shared_sha" native native 2 none \
    4096 8192

  if "$python_bin" - \
    "$screen_qa_evq_root/results.json" \
    "$screen_qa_native_root/results.json" \
    "$screen_ruler_evq_root/results.json" \
    "$screen_ruler_native_root/results.json" \
    "$evq_shared_run/results.json" \
    "$native_shared_run/results.json" \
    "$gate_json" <<'PY'
import hashlib
import json
import pathlib
import sys

qa_evq, qa_native, ruler_evq, ruler_native, evq_train_path, \
    native_train_path, output = map(
    pathlib.Path, sys.argv[1:]
)


def load(path):
    with path.open() as handle:
        return json.load(handle)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qa_metric(payload, length, name):
    return float(payload["results"]["cells"][str(length)][name])


def ruler_macro(payload, length):
    scores = [
        float(task_cells[str(length)]["official_task_score"])
        for task_cells in payload["results"]["cells"].values()
    ]
    return sum(scores) / len(scores)


qe = load(qa_evq)
qn = load(qa_native)
re = load(ruler_evq)
rn = load(ruler_native)
evq_train = load(evq_train_path)
native_train = load(native_train_path)
metrics = {
    "qa": {
        f"{arm}_{length}_{metric}": qa_metric(payload, length_value, metric)
        for arm, payload in (("evq", qe), ("native", qn))
        for length, length_value in (("4k", 4096), ("8k", 8192))
        for metric in ("mean_token_f1", "normalized_exact", "terminal_eos")
    },
    "ruler": {
        f"{arm}_{length}_macro": ruler_macro(payload, length_value)
        for arm, payload in (("evq", re), ("native", rn))
        for length, length_value in (("4k", 4096), ("8k", 8192))
    },
}
checks = {
    "qa_8k_evq_f1_gt_native": (
        metrics["qa"]["evq_8k_mean_token_f1"]
        > metrics["qa"]["native_8k_mean_token_f1"]
    ),
    "qa_8k_evq_exact_gt_native": (
        metrics["qa"]["evq_8k_normalized_exact"]
        > metrics["qa"]["native_8k_normalized_exact"]
    ),
    "qa_8k_evq_terminal_eos_at_least_80pct": (
        metrics["qa"]["evq_8k_terminal_eos"] >= 0.80
    ),
    "qa_4k_evq_within_5pp_native": (
        metrics["qa"]["evq_4k_mean_token_f1"]
        >= metrics["qa"]["native_4k_mean_token_f1"] - 0.05
    ),
    "ruler_8k_evq_gt_native": (
        metrics["ruler"]["evq_8k_macro"]
        > metrics["ruler"]["native_8k_macro"]
    ),
    "ruler_4k_evq_within_10pp_native": (
        metrics["ruler"]["evq_4k_macro"]
        >= metrics["ruler"]["native_4k_macro"] - 0.10
    ),
}
passed = all(checks.values())
receipt = {
    "status": "SHARED_QK_SCREEN_GATE_V2",
    "passed": passed,
    "metrics": metrics,
    "checks": checks,
    "screen_results": {
        "qa_evq": {"path": str(qa_evq), "sha256": sha(qa_evq)},
        "qa_native": {"path": str(qa_native), "sha256": sha(qa_native)},
        "ruler_evq": {"path": str(ruler_evq), "sha256": sha(ruler_evq)},
        "ruler_native": {
            "path": str(ruler_native),
            "sha256": sha(ruler_native),
        },
    },
    "lineage": {
        "checkpoint_sha256": evq_train["checkpoint_sha256"],
        "evq_adapter_sha256": evq_train["adapter_sha256"],
        "native_adapter_sha256": native_train["adapter_sha256"],
        "evq_training_result_sha256": sha(evq_train_path),
        "native_training_result_sha256": sha(native_train_path),
    },
}
temporary = output.with_suffix(".json.incomplete")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 10)
PY
  then
    shared_gate_passed=1
  fi
else
  "$python_bin" - "$gate_json" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
receipt = {
    "status": "SHARED_QK_SCREEN_GATE_V2",
    "passed": False,
    "reason": "screen controller ended without all four result artifacts",
}
temporary = output.with_suffix(".json.incomplete")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
fi

if [[ "$shared_gate_passed" -eq 1 ]]; then
  ensure_2wiki \
    "$runs/full_2wiki_general_qk_evq_shared_phase500_s20260730" \
    "$evq_shared_run/adapter.pt" "$evq_shared_sha" evq evq 200 \
    evq_shared none 4096 8192 16384
  ensure_2wiki \
    "$runs/full_2wiki_general_qk_native_shared_phase500_s20260730" \
    "$native_shared_run/adapter.pt" "$native_shared_sha" native native 200 \
    native_shared none 4096 8192 16384
  ensure_ruler \
    "$runs/full_ruler_general_qk_evq_shared_phase500_s20260730" \
    "$evq_shared_run/adapter.pt" "$evq_shared_sha" evq evq 20 none \
    4096 8192 16384
  ensure_ruler \
    "$runs/full_ruler_general_qk_native_shared_phase500_s20260730" \
    "$native_shared_run/adapter.pt" "$native_shared_sha" native native 20 none \
    4096 8192 16384
else
  echo "SHARED_GATE_FAILED_SKIP_FULL"
fi

# Resume the exact registered raw-EVQ 2Wiki job. Existing 4K/8K rows are
# identity-checked by the evaluator; only the missing 16K rows are generated.
ensure_2wiki \
  "$registered_root/2wiki_general_qk_evq_raw" \
  "$evq_generic_run/adapter.pt" "$evq_generic_sha" evq evq 200 \
  2wiki_general_qk_evq_raw none 4096 8192 16384

# Complete the only still-needed operator control: the same generic adapters,
# test rows, and factor-2 target under the repository fixed ramp.
ensure_ruler \
  "$registered_root/ruler_general_qk_native_repo_fixed_ramp_f2" \
  "$native_generic_run/adapter.pt" "$native_generic_sha" native \
  repo_fixed_ramp 20 2.0 4096 8192
ensure_ruler \
  "$registered_root/ruler_general_qk_evq_repo_fixed_ramp_f2" \
  "$evq_generic_run/adapter.pt" "$evq_generic_sha" evq \
  evq_repo_fixed_ramp 20 2.0 4096 8192

# Revalidate every promoted component under the strengthened validator. This
# block performs no inference when results.json already exists.
validate_2wiki \
  "$registered_root/2wiki_base_native_4k" none native native 200 \
  2wiki_base_native_4k none 4096
validate_2wiki \
  "$registered_root/2wiki_general_qk_native_raw" \
  "$native_generic_sha" native native 200 \
  2wiki_general_qk_native_raw none 4096 8192 16384
validate_2wiki \
  "$registered_root/2wiki_general_qk_evq_raw" \
  "$evq_generic_sha" evq evq 200 \
  2wiki_general_qk_evq_raw none 4096 8192 16384
validate_ruler \
  "$registered_root/ruler_base_native_4k" none native native 20 none 4096
validate_ruler \
  "$registered_root/ruler_general_qk_native_raw" \
  "$native_generic_sha" native native 20 none 4096 8192 16384
validate_ruler \
  "$registered_root/ruler_general_qk_evq_raw" \
  "$evq_generic_sha" evq evq 20 none 4096 8192 16384
validate_ruler \
  "$registered_root/ruler_general_qk_native_official_yarn_f2" \
  "$native_generic_sha" native official_yarn 20 2.0 4096 8192
validate_ruler \
  "$registered_root/ruler_general_qk_evq_official_yarn_f2" \
  "$evq_generic_sha" evq evq_official_yarn 20 2.0 4096 8192
validate_ruler \
  "$registered_root/ruler_general_qk_native_repo_fixed_ramp_f2" \
  "$native_generic_sha" native repo_fixed_ramp 20 2.0 4096 8192
validate_ruler \
  "$registered_root/ruler_general_qk_evq_repo_fixed_ramp_f2" \
  "$evq_generic_sha" evq evq_repo_fixed_ramp 20 2.0 4096 8192

"$python_bin" - \
  "$gate_json" \
  "$registered_root" \
  "$evq_shared_run/results.json" \
  "$native_shared_run/results.json" \
  "$script_path" \
  "$validator_path" \
  "$strict_validation_receipt" \
  "$completion_receipt" <<'PY'
import hashlib
import json
import pathlib
import sys

gate_path = pathlib.Path(sys.argv[1])
registered_root = pathlib.Path(sys.argv[2])
evq_train_path = pathlib.Path(sys.argv[3])
native_train_path = pathlib.Path(sys.argv[4])
script_path = pathlib.Path(sys.argv[5])
validator_path = pathlib.Path(sys.argv[6])
strict_output = pathlib.Path(sys.argv[7])
output = pathlib.Path(sys.argv[8])
gate = json.loads(gate_path.read_text())
paths = [
    gate_path,
    evq_train_path,
    native_train_path,
    registered_root / "2wiki_general_qk_evq_raw" / "results.json",
    registered_root
    / "ruler_general_qk_native_repo_fixed_ramp_f2"
    / "results.json",
    registered_root
    / "ruler_general_qk_evq_repo_fixed_ramp_f2"
    / "results.json",
]
component_roots = {
    "2wiki_base_native_4k": registered_root / "2wiki_base_native_4k",
    "2wiki_general_qk_native_raw": (
        registered_root / "2wiki_general_qk_native_raw"
    ),
    "2wiki_general_qk_evq_raw": (
        registered_root / "2wiki_general_qk_evq_raw"
    ),
    "ruler_base_native_4k": registered_root / "ruler_base_native_4k",
    "ruler_general_qk_native_raw": (
        registered_root / "ruler_general_qk_native_raw"
    ),
    "ruler_general_qk_evq_raw": (
        registered_root / "ruler_general_qk_evq_raw"
    ),
    "ruler_general_qk_native_official_yarn_f2": (
        registered_root / "ruler_general_qk_native_official_yarn_f2"
    ),
    "ruler_general_qk_evq_official_yarn_f2": (
        registered_root / "ruler_general_qk_evq_official_yarn_f2"
    ),
    "ruler_general_qk_native_repo_fixed_ramp_f2": (
        registered_root / "ruler_general_qk_native_repo_fixed_ramp_f2"
    ),
    "ruler_general_qk_evq_repo_fixed_ramp_f2": (
        registered_root / "ruler_general_qk_evq_repo_fixed_ramp_f2"
    ),
}
initial_evq_qa_log = registered_root / "2wiki_general_qk_evq_raw.log"
resume_log = (
    registered_root.parent
    / "general_qk_2wiki_evq_resume_frozen_s20260730.log"
)
initial_log_text = initial_evq_qa_log.read_text(errors="replace")
resume_log_text = resume_log.read_text(errors="replace")
required_initial_markers = (
    "2wiki_general_qk_evq_raw L=8192 200/200",
    "torch.OutOfMemoryError: CUDA out of memory",
)
required_resume_markers = (
    f"START {component_roots['2wiki_general_qk_evq_raw']}",
    "2wiki_general_qk_evq_raw L=16384 1/200",
    "OLMO2_2WIKI_PHASE_EVALUATION_COMPLETE_V1",
)
if not all(marker in initial_log_text for marker in required_initial_markers):
    raise RuntimeError("initial EVQ 2Wiki failure log contract drift")
if not all(marker in resume_log_text for marker in required_resume_markers):
    raise RuntimeError("EVQ 2Wiki resume log contract drift")
artifacts = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths
}
components = {
    name: {
        filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for filename in ("results.json", "examples.jsonl", "run_manifest.json")
    }
    for name, root in component_roots.items()
}
strict_receipt = {
    "status": "GENERAL_QK_SELECTED_MATRIX_STRICT_VALIDATION_COMPLETE_V1",
    "component_count": len(components),
    "validator_sha256": hashlib.sha256(
        validator_path.read_bytes()
    ).hexdigest(),
    "controller_script_sha256": hashlib.sha256(
        script_path.read_bytes()
    ).hexdigest(),
    "validated_contract": (
        "checkpoint, adapter metadata, training length, independently "
        "anchored realized frequency, recorded evaluator/helper or bound-code "
        "identity, data manifest, cell coverage, unique raw rows, physical "
        "token budget, aggregate recomputation, and artifact hashes"
    ),
    "components": components,
    "recovery": {
        "component": "2wiki_general_qk_evq_raw",
        "initial_launcher_sha256": hashlib.sha256(
            (registered_root / "2wiki_launcher.jsonl").read_bytes()
        ).hexdigest(),
        "initial_failure_log_sha256": hashlib.sha256(
            initial_evq_qa_log.read_bytes()
        ).hexdigest(),
        "resume_controller_log_sha256": hashlib.sha256(
            resume_log.read_bytes()
        ).hexdigest(),
        "contract": (
            "The initial log reaches 8K row 200 and then records CUDA OOM. "
            "The recovery log starts the same output root at 16K row 1 and "
            "records final completion. The evaluator's resume path requires "
            "run-manifest equality and completed-row identity, while the "
            "final strict gate verifies 600 unique rows."
        ),
    },
}
temporary = strict_output.with_suffix(".json.incomplete")
temporary.write_text(
    json.dumps(strict_receipt, indent=2, sort_keys=True) + "\n"
)
temporary.replace(strict_output)
long_checks = (
    "qa_8k_evq_f1_gt_native",
    "qa_8k_evq_exact_gt_native",
    "qa_8k_evq_terminal_eos_at_least_80pct",
    "ruler_8k_evq_gt_native",
)
receipt = {
    "status": "SHARED_QK_COMPLETION_PIPELINE_COMPLETE_V2",
    "shared_single_path_gate_passed": bool(gate["passed"]),
    "length_route_screen_passed": all(
        gate.get("checks", {}).get(name) is True
        for name in long_checks
    ),
    "artifacts": {
        **artifacts,
        str(strict_output): hashlib.sha256(
            strict_output.read_bytes()
        ).hexdigest(),
    },
    "controller_script_sha256": hashlib.sha256(
        script_path.read_bytes()
    ).hexdigest(),
    "validator_sha256": hashlib.sha256(
        validator_path.read_bytes()
    ).hexdigest(),
}
temporary = output.with_suffix(".json.incomplete")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "SHARED_QK_COMPLETION_PIPELINE_COMPLETE"
