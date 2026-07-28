#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ASSET_BASE PRIOR_CONTROLLER_PID" >&2
  exit 2
fi

asset_base=$1
prior_pid=$2
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
screen_gate="$runs/general_qk_shared_task_screen_gate_s20260730.json"
evq_run="$runs/general_qk_evq_shared_task_phase500_s20260730"
native_run="$runs/general_qk_native_shared_task_phase500_s20260730"
script_path=$(realpath "${BASH_SOURCE[0]}")
validator_path="$code_root/rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/validate_qk_eval_result.py"
completion_controller_path="$code_root/rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/run_shared_qk_completion.sh"

if kill -0 "$prior_pid" 2>/dev/null; then
  prior_cmdline=$(tr '\0' ' ' <"/proc/$prior_pid/cmdline")
  [[ "$prior_cmdline" == *run_shared_qk_completion* ]]
fi
while kill -0 "$prior_pid" 2>/dev/null; do
  sleep 3
done
test -f "$completion_receipt"
test -f "$strict_validation_receipt"
test -f "$screen_gate"

cd "$code_root"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$code_root"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t identities < <(
  "$python_bin" - "$evq_run/results.json" "$native_run/results.json" \
    "$downstream_ready" "$screen_gate" "$completion_receipt" \
    "$strict_validation_receipt" "$validator_path" \
    "$completion_controller_path" <<'PY'
import hashlib
import json
import pathlib
import sys

evq_path, native_path, ready_path, gate_path, completion_path, \
    strict_path, validator_path, completion_controller_path = map(
    pathlib.Path, sys.argv[1:]
)
evq = json.loads(evq_path.read_text())
native = json.loads(native_path.read_text())
ready = json.loads(ready_path.read_text())
gate = json.loads(gate_path.read_text())
completion = json.loads(completion_path.read_text())
strict = json.loads(strict_path.read_text())
gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
evq_train_sha = hashlib.sha256(evq_path.read_bytes()).hexdigest()
native_train_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
strict_sha = hashlib.sha256(strict_path.read_bytes()).hexdigest()
validator_sha = hashlib.sha256(validator_path.read_bytes()).hexdigest()
completion_controller_sha = hashlib.sha256(
    completion_controller_path.read_bytes()
).hexdigest()
if (
    completion.get("status")
    != "SHARED_QK_COMPLETION_PIPELINE_COMPLETE_V2"
    or completion.get("artifacts", {}).get(str(gate_path)) != gate_sha
):
    raise RuntimeError("prior completion receipt or screen-gate hash drift")
if gate.get("status") != "SHARED_QK_SCREEN_GATE_V2":
    raise RuntimeError("screen-gate status drift")
if completion.get("shared_single_path_gate_passed") != bool(
    gate.get("passed")
):
    raise RuntimeError("completion/screen-gate decision drift")
if completion.get("length_route_screen_passed") is not True:
    raise RuntimeError("completion receipt did not pass length-route screen")
if (
    completion.get("artifacts", {}).get(str(strict_path)) != strict_sha
    or completion.get("validator_sha256") != validator_sha
    or completion.get("controller_script_sha256")
    != completion_controller_sha
):
    raise RuntimeError("completion strict-validator lineage drift")
if (
    strict.get("status")
    != "GENERAL_QK_SELECTED_MATRIX_STRICT_VALIDATION_COMPLETE_V1"
    or strict.get("component_count") != 10
    or strict.get("validator_sha256") != validator_sha
    or strict.get("controller_script_sha256")
    != completion_controller_sha
):
    raise RuntimeError("strict-validation receipt drift")
for path, digest in (
    (evq_path, evq_train_sha),
    (native_path, native_train_sha),
):
    if completion.get("artifacts", {}).get(str(path)) != digest:
        raise RuntimeError(f"completion training-result drift: {path}")
expected_lineage = {
    "checkpoint_sha256": evq.get("checkpoint_sha256"),
    "evq_adapter_sha256": evq.get("adapter_sha256"),
    "native_adapter_sha256": native.get("adapter_sha256"),
    "evq_training_result_sha256": evq_train_sha,
    "native_training_result_sha256": native_train_sha,
}
for name, expected in expected_lineage.items():
    if gate.get("lineage", {}).get(name) != expected:
        raise RuntimeError(f"screen-gate lineage drift: {name}")
screen_expectations = {
    "qa_evq": evq.get("adapter_sha256"),
    "ruler_evq": evq.get("adapter_sha256"),
    "qa_native": native.get("adapter_sha256"),
    "ruler_native": native.get("adapter_sha256"),
}
for name, expected_adapter in screen_expectations.items():
    artifact = gate.get("screen_results", {}).get(name)
    if not isinstance(artifact, dict):
        raise RuntimeError(f"missing screen artifact: {name}")
    path = pathlib.Path(artifact["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise RuntimeError(f"screen artifact hash drift: {name}")
    payload = json.loads(path.read_text())
    if (
        payload.get("checkpoint_sha256") != evq.get("checkpoint_sha256")
        or payload.get("adapter", {}).get("sha256") != expected_adapter
    ):
        raise RuntimeError(f"screen artifact lineage drift: {name}")
long_checks = (
    "qa_8k_evq_f1_gt_native",
    "qa_8k_evq_exact_gt_native",
    "qa_8k_evq_terminal_eos_at_least_80pct",
    "ruler_8k_evq_gt_native",
)
if not all(gate.get("checks", {}).get(name) is True for name in long_checks):
    raise RuntimeError("shared adapter did not pass the long-route screen")
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
        raise RuntimeError(f"shared matched field drift: {key}")
print(evq["checkpoint_sha256"])
print(evq["adapter_sha256"])
print(native["adapter_sha256"])
print(ready["inputs"]["two_wiki_manifest_sha256"])
print(ready["inputs"]["ruler_manifest_sha256"])
print(ready["bound_code_sha256"]["two_wiki_evaluator"])
print(ready["bound_code_sha256"]["ruler_evaluator"])
PY
)

if [[ ${#identities[@]} -ne 7 ]]; then
  echo "identity extraction failed" >&2
  exit 4
fi
checkpoint_sha=${identities[0]}
evq_sha=${identities[1]}
native_sha=${identities[2]}
qa_manifest_sha=${identities[3]}
ruler_manifest_sha=${identities[4]}
qa_evaluator_sha=${identities[5]}
ruler_evaluator_sha=${identities[6]}

validator=(
  "$python_bin" -m
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.validate_qk_eval_result
)

validate_qa() {
  local root=$1
  local adapter_sha=$2
  local training_frequency=$3
  local frequency=$4
  local role=$5
  shift 5
  "${validator[@]}" \
    --root "$root" \
    --benchmark 2wiki \
    --checkpoint-sha256 "$checkpoint_sha" \
    --adapter-sha256 "$adapter_sha" \
    --adapter-training-frequency "$training_frequency" \
    --frequency "$frequency" \
    --lengths "$@" \
    --limit 200 \
    --data-manifest-sha256 "$qa_manifest_sha" \
    --evaluator-sha256 "$qa_evaluator_sha" \
    --adaptation qk_answer \
    --rank 64 \
    --alpha 128 \
    --expected-role "$role" \
    --fill-to-budget
}

validate_ruler() {
  local root=$1
  local adapter_sha=$2
  local training_frequency=$3
  local frequency=$4
  shift 4
  "${validator[@]}" \
    --root "$root" \
    --benchmark ruler \
    --checkpoint-sha256 "$checkpoint_sha" \
    --adapter-sha256 "$adapter_sha" \
    --adapter-training-frequency "$training_frequency" \
    --frequency "$frequency" \
    --lengths "$@" \
    --limit 20 \
    --data-manifest-sha256 "$ruler_manifest_sha" \
    --evaluator-sha256 "$ruler_evaluator_sha" \
    --adaptation qk_answer \
    --rank 64 \
    --alpha 128
}

ensure_qa() {
  local root=$1
  local adapter=$2
  local adapter_sha=$3
  local training_frequency=$4
  local frequency=$5
  local role=$6
  shift 6
  local lengths=("$@")
  if [[ ! -f "$root/results.json" ]]; then
    echo "START $root"
    "$python_bin" -m \
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_2wiki_phase_adaptation \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$checkpoint_receipt" \
      --data-root "$qa_data" \
      --adapter "$adapter" \
      --output "$root" \
      --role "$role" \
      --frequency "$frequency" \
      --adaptation qk_answer \
      --rank 64 \
      --alpha 128 \
      --budgets "${lengths[@]}" \
      --limit 200 \
      --fill-to-budget
  fi
  validate_qa \
    "$root" "$adapter_sha" "$training_frequency" "$frequency" "$role" \
    "${lengths[@]}"
}

ensure_ruler() {
  local root=$1
  local adapter=$2
  local adapter_sha=$3
  local training_frequency=$4
  local frequency=$5
  shift 5
  local lengths=("$@")
  if [[ ! -f "$root/results.json" ]]; then
    echo "START $root"
    "$python_bin" -m \
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer \
      --checkpoint "$checkpoint" \
      --ready-receipt "$checkpoint_receipt" \
      --data-root "$ruler_data" \
      --adapter "$adapter" \
      --output "$root" \
      --frequency "$frequency" \
      --adaptation qk_answer \
      --rank 64 \
      --alpha 128 \
      --lengths "${lengths[@]}" \
      --limit-per-cell 20
  fi
  validate_ruler \
    "$root" "$adapter_sha" "$training_frequency" "$frequency" \
    "${lengths[@]}"
}

qa_evq_root="$runs/full_long_2wiki_general_qk_evq_shared_phase500_s20260730"
qa_native_root="$runs/full_long_2wiki_general_qk_native_shared_phase500_s20260730"
ruler_evq_root="$runs/full_long_ruler_general_qk_evq_shared_phase500_s20260730"
ruler_native_root="$runs/full_long_ruler_general_qk_native_shared_phase500_s20260730"

ensure_qa \
  "$qa_evq_root" "$evq_run/adapter.pt" "$evq_sha" evq evq \
  evq_shared_long 8192 16384
ensure_qa \
  "$qa_native_root" "$native_run/adapter.pt" "$native_sha" native native \
  native_shared_long 8192 16384
ensure_ruler \
  "$ruler_evq_root" "$evq_run/adapter.pt" "$evq_sha" evq evq \
  8192 16384
ensure_ruler \
  "$ruler_native_root" "$native_run/adapter.pt" "$native_sha" native native \
  8192 16384

# Validate the exact untouched-Native short branch on the same frozen data.
validate_qa \
  "$registered_root/2wiki_base_native_4k" none native native \
  2wiki_base_native_4k 4096
validate_ruler \
  "$registered_root/ruler_base_native_4k" none native native 4096

"$python_bin" - \
  "$registered_root/2wiki_base_native_4k/results.json" \
  "$qa_evq_root/results.json" \
  "$qa_native_root/results.json" \
  "$registered_root/ruler_base_native_4k/results.json" \
  "$ruler_evq_root/results.json" \
  "$ruler_native_root/results.json" \
  "$evq_run/results.json" \
  "$native_run/results.json" \
  "$downstream_ready" \
  "$screen_gate" \
  "$completion_receipt" \
  "$strict_validation_receipt" \
  "$script_path" \
  "$validator_path" \
  "$runs/length_routed_shared_qk_metrics_s20260730.json" <<'PY'
import hashlib
import json
import pathlib
import sys

qa_base_path, qa_evq_path, qa_native_path, ruler_base_path, \
    ruler_evq_path, ruler_native_path, evq_train_path, native_train_path, \
    ready_path, gate_path, completion_path, strict_path, script_path, \
    validator_path, output = map(pathlib.Path, sys.argv[1:])


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_set(result_path):
    root = result_path.parent
    return {
        "results_sha256": sha(result_path),
        "examples_sha256": sha(root / "examples.jsonl"),
        "run_manifest_sha256": sha(root / "run_manifest.json"),
    }


def qa_cell(payload, length):
    cell = payload["results"]["cells"][str(length)]
    return {
        "token_f1": float(cell["mean_token_f1"]),
        "normalized_exact": float(cell["normalized_exact"]),
        "terminal_eos": float(cell["terminal_eos"]),
        "examples": int(cell["examples"]),
    }


def ruler_cell(payload, length):
    cells = payload["results"]["cells"]
    scores = [
        float(task_cells[str(length)]["official_task_score"])
        for task_cells in cells.values()
    ]
    return {
        "official_macro": sum(scores) / len(scores),
        "families": len(scores),
        "examples_per_family": int(
            next(iter(cells.values()))[str(length)]["examples"]
        ),
    }


def ruler_families(base_payload, evq_payload):
    return {
        task: {
            "4096": float(
                base_payload["results"]["cells"][task]["4096"][
                    "official_task_score"
                ]
            ),
            "8192": float(
                evq_payload["results"]["cells"][task]["8192"][
                    "official_task_score"
                ]
            ),
            "16384": float(
                evq_payload["results"]["cells"][task]["16384"][
                    "official_task_score"
                ]
            ),
        }
        for task in sorted(base_payload["results"]["cells"])
    }


def native_long_families(payload):
    return {
        task: {
            length: float(cells[length]["official_task_score"])
            for length in ("8192", "16384")
        }
        for task, cells in sorted(payload["results"]["cells"].items())
    }


qa_base = load(qa_base_path)
qa_evq = load(qa_evq_path)
qa_native = load(qa_native_path)
ruler_base = load(ruler_base_path)
ruler_evq = load(ruler_evq_path)
ruler_native = load(ruler_native_path)
evq_train = load(evq_train_path)
native_train = load(native_train_path)
ready = load(ready_path)
screen = load(gate_path)
receipt = {
    "status": "OLMO2_REQUEST_LENGTH_ROUTED_SHARED_EVAL_COMPLETE_V1",
    "policy": {
        "selection": "registered total context budget before prefill",
        "short": {
            "condition": "budget <= 4096",
            "frequency": "untouched Native RoPE",
            "adapter": "disabled",
        },
        "long": {
            "condition": "evaluated budgets 8192 or 16384",
            "frequency": "EVQ-Cosh",
            "adapter": "shared 2Wiki+RULER13 Q/K phase adapter",
        },
        "boundary": (
            "Post-hoc deterministic two-path deployment policy; not a "
            "single pure-EVQ configuration and not unseen-task transfer."
        ),
    },
    "routed": {
        "2wiki": {
            "4096": qa_cell(qa_base, 4096),
            "8192": qa_cell(qa_evq, 8192),
            "16384": qa_cell(qa_evq, 16384),
        },
        "ruler": {
            "4096": ruler_cell(ruler_base, 4096),
            "8192": ruler_cell(ruler_evq, 8192),
            "16384": ruler_cell(ruler_evq, 16384),
        },
        "ruler_by_family": ruler_families(ruler_base, ruler_evq),
    },
    "matched_long_native_control": {
        "2wiki": {
            "8192": qa_cell(qa_native, 8192),
            "16384": qa_cell(qa_native, 16384),
        },
        "ruler": {
            "8192": ruler_cell(ruler_native, 8192),
            "16384": ruler_cell(ruler_native, 16384),
        },
        "ruler_by_family": native_long_families(ruler_native),
    },
    "lineage": {
        "checkpoint_sha256": evq_train["checkpoint_sha256"],
        "data_manifest_sha256": {
            "2wiki": ready["inputs"]["two_wiki_manifest_sha256"],
            "ruler": ready["inputs"]["ruler_manifest_sha256"],
        },
        "evaluator_sha256": {
            "2wiki": ready["bound_code_sha256"]["two_wiki_evaluator"],
            "ruler": ready["bound_code_sha256"]["ruler_evaluator"],
        },
        "screen_gate_sha256": sha(gate_path),
        "completion_receipt_sha256": sha(completion_path),
        "strict_validation_receipt_sha256": sha(strict_path),
        "route_script_sha256": sha(script_path),
        "validator_sha256": sha(validator_path),
        "screen_long_checks": {
            name: value
            for name, value in screen["checks"].items()
            if "8k" in name
        },
        "shared_training": {
            "native_adapter_sha256": native_train["adapter_sha256"],
            "evq_adapter_sha256": evq_train["adapter_sha256"],
            "native_parent_adapter_sha256": native_train[
                "adapter_metadata"
            ]["parent_adapter_sha256"],
            "evq_parent_adapter_sha256": evq_train[
                "adapter_metadata"
            ]["parent_adapter_sha256"],
            "selection_stream_sha256": evq_train["training"][
                "selection_stream_sha256"
            ],
            "position_stream_sha256": evq_train["training"][
                "position_stream_sha256"
            ],
            "exposure_stream_sha256": evq_train["training"][
                "exposure_stream_sha256"
            ],
            "family_steps": evq_train["training"]["family_steps"],
            "processed_input_tokens": evq_train["training"][
                "processed_input_tokens"
            ],
            "native_training_result_sha256": sha(native_train_path),
            "evq_training_result_sha256": sha(evq_train_path),
        },
    },
    "component_artifacts": {
        str(path.parent): artifact_set(path)
        for path in (
            qa_base_path,
            qa_evq_path,
            qa_native_path,
            ruler_base_path,
            ruler_evq_path,
            ruler_native_path,
        )
    },
}
temporary = output.with_suffix(".json.incomplete")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "LENGTH_ROUTED_SHARED_EVAL_COMPLETE"
