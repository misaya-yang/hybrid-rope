#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
code_root="${CODE_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
run_root="${DIRECT_Z_ROOT:?set DIRECT_Z_ROOT to a persistent output directory}"
checkpoint="${DIRECT_Z_CHECKPOINT:?set DIRECT_Z_CHECKPOINT}"
ready="${DIRECT_Z_READY:?set DIRECT_Z_READY}"
natural_tensor="${DIRECT_Z_NATURAL_TENSOR:-}"
natural_receipt="${DIRECT_Z_NATURAL_RECEIPT:-}"
token_manifest="${DIRECT_Z_TOKEN_MANIFEST:-}"
ruler_data="${DIRECT_Z_RULER_DATA:-}"
longbench_zip="${DIRECT_Z_LONGBENCH_ZIP:-}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN to the experiment Python executable}"
table="$run_root/direct_z_fixed_support.npy"
optimization_result="$run_root/direct_z_result.json"

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

die() { echo "ERROR: $*" >&2; exit 2; }

require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

require_python() {
  [[ -x "$python_bin" ]] || die "PYTHON_BIN is not executable: $python_bin"
}

require_common_assets() {
  require_python
  require_dir "$checkpoint"
  require_file "$ready"
}

authorize_gpu() {
  [[ "${DIRECT_Z_RUN_AUTHORIZED:-}" == "YES" ]] || {
    die "set DIRECT_Z_RUN_AUTHORIZED=YES only after explicit authorization for this run"
  }
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export DIRECT_Z_GPU_AUTHORIZED=YES
  mkdir -p "$run_root" "${TORCHINDUCTOR_CACHE_DIR:-$run_root/torchinductor_cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$run_root/torchinductor_cache}"
}

table_hash() {
  require_file "$table"
  require_file "$optimization_result"
  "$python_bin" - "$optimization_result" "$table" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

result_path, table_path = map(pathlib.Path, sys.argv[1:3])
result = json.loads(result_path.read_text())
table = np.load(table_path, allow_pickle=False)
active = hashlib.sha256(np.ascontiguousarray(table, dtype="<f4").tobytes()).hexdigest()
if result.get("status") != "OLMO_FIXED_SUPPORT_DIRECT_Z_COMPLETE_V1":
    raise SystemExit("full direct-z result required; smoke is not promotable")
if result.get("passed") is not True or result.get("scientific_passed") is not True:
    raise SystemExit("direct-z optimization did not pass its held-out gates")
protocol = result.get("protocol", {})
receipt = result.get("table", {})
if (
    protocol.get("target_ratio") != 2
    or protocol.get("same_table_at_1x_and_2x") is not True
    or protocol.get("model_weight_updates") != 0
    or receipt.get("fixed_support") is not True
    or receipt.get("candidate_differs_from_native") is not True
    or receipt.get("active_sha256_float32") != active
):
    raise SystemExit("direct-z result identity drift")
print(active)
PY
}

require_bound_result() {
  local directory="$1"
  local expected_hash="$2"
  local expected_kind="$3"
  require_file "$directory/run_manifest.json"
  require_file "$directory/results.json"
  "$python_bin" - "$directory/run_manifest.json" "$expected_hash" "$optimization_result" "$expected_kind" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
optimization = json.loads(pathlib.Path(sys.argv[3]).read_text())
if (
    manifest.get("table_sha256_float32") != sys.argv[2]
    or manifest.get("checkpoint_sha256")
    != optimization.get("checkpoint", {}).get("composite_sha256")
    or manifest.get("method") != "external_table_static"
    or manifest.get("table_support") != "native"
    or manifest.get("long_attention_scaling") != 1.0
):
    raise SystemExit("screen result belongs to a different table")
kind = sys.argv[4]
if kind == "pg19" and (
    manifest.get("tasks") != ["pg19"]
    or manifest.get("multipliers") != [1, 2]
    or manifest.get("factor") != 2.0
    or manifest.get("limit_per_cell") != 5
):
    raise SystemExit("PG-19 screen protocol drift")
if kind == "ruler" and (
    manifest.get("tasks")
    != ["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt"]
    or manifest.get("lengths") != [8192]
    or manifest.get("table_factor") != 2.0
    or manifest.get("limit_per_cell") != 5
):
    raise SystemExit("RULER screen protocol drift")
if kind == "formal" and (
    manifest.get("tasks") != ["2wikimqa", "qasper"]
    or manifest.get("multipliers") != [1, 2]
    or manifest.get("factor") != 2.0
    or manifest.get("limit_per_cell") != 20
):
    raise SystemExit("formal natural protocol drift")
PY
}

reuse_bound_result() {
  local directory="$1"
  local expected_hash="$2"
  local expected_kind="$3"
  if [[ ! -e "$directory/run_manifest.json" && ! -e "$directory/results.json" ]]; then
    return 1
  fi
  require_bound_result "$directory" "$expected_hash" "$expected_kind"
  echo "REUSE: completed identity-matched result at $directory" >&2
  return 0
}

require_smoke() {
  local directory="$run_root/smoke"
  require_file "$directory/direct_z_fixed_support.npy"
  require_file "$directory/direct_z_result.json"
  require_file "$directory/progress.jsonl"
  "$python_bin" - "$directory/direct_z_result.json" "$directory/direct_z_fixed_support.npy" "$directory/progress.jsonl" <<'PY'
import hashlib, json, math, pathlib, sys
import numpy as np
result_path, table_path = map(pathlib.Path, sys.argv[1:3])
progress_path = pathlib.Path(sys.argv[3])
result = json.loads(result_path.read_text())
table = np.load(table_path, allow_pickle=False)
progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
active = hashlib.sha256(np.ascontiguousarray(table, dtype="<f4").tobytes()).hexdigest()
if (
    result.get("status") != "OLMO_FIXED_SUPPORT_DIRECT_Z_SMOKE_COMPLETE_V1"
    or result.get("passed") is not True
    or result.get("protocol", {}).get("steps") != 1
    or result.get("protocol", {}).get("model_weight_updates") != 0
    or result.get("table", {}).get("fixed_support") is not True
    or result.get("table", {}).get("active_sha256_float32") != active
    or result.get("initialization", {}).get("maximum_native_nll_parity_delta", 1.0) > 1e-4
    or len(progress) != 1
    or progress[0].get("step") != 1
    or not math.isfinite(float(progress[0].get("gradient_norm", float("nan"))))
):
    raise SystemExit("one-step direct-z smoke identity drift")
PY
}

require_full_result() {
  local directory="$1"
  local expected_hash="$2"
  local expected_task="$3"
  require_file "$directory/results.json"
  "$python_bin" - "$directory/results.json" "$expected_hash" "$expected_task" "$optimization_result" <<'PY'
import json, pathlib, sys
result = json.loads(pathlib.Path(sys.argv[1]).read_text())
optimization = json.loads(pathlib.Path(sys.argv[4]).read_text())
method = result.get("method", {})
protocol = result.get("protocol", {})
if (
    result.get("checkpoint_sha256")
    != optimization.get("checkpoint", {}).get("composite_sha256")
    or method.get("active_sha256_float32") != sys.argv[2]
    or method.get("method") != "external_static_fixed_support_z"
    or protocol.get("task") != sys.argv[3]
    or protocol.get("frequency") != "external_static"
    or protocol.get("factor") != 2.0
    or protocol.get("nominal_length") != 8192
    or protocol.get("rows") != 200
):
    raise SystemExit("full-task result identity drift")
PY
}

reuse_full_result() {
  local directory="$1"
  local expected_hash="$2"
  local expected_task="$3"
  [[ -e "$directory/results.json" ]] || return 1
  require_full_result "$directory" "$expected_hash" "$expected_task"
  echo "REUSE: completed identity-matched result at $directory" >&2
  return 0
}

case "$action" in
  optimize-smoke)
    require_common_assets
    require_file "$natural_tensor"
    require_file "$natural_receipt"
    if [[ -e "$run_root/smoke/direct_z_result.json" || -e "$run_root/smoke/direct_z_fixed_support.npy" || -e "$run_root/smoke/progress.jsonl" ]]; then
      require_smoke
      echo "REUSE: successful one-step smoke at $run_root/smoke" >&2
      exit 0
    fi
    authorize_gpu
    "$python_bin" -m scripts.eval.optimize_olmo_fixed_support_z \
      --authorize --smoke --steps 1 \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$ready" \
      --natural-2x-tensor "$natural_tensor" \
      --natural-2x-receipt "$natural_receipt" \
      --output-table "$run_root/smoke/direct_z_fixed_support.npy" \
      --output-result "$run_root/smoke/direct_z_result.json" \
      --progress "$run_root/smoke/progress.jsonl"
    ;;
  optimize)
    require_common_assets
    require_file "$natural_tensor"
    require_file "$natural_receipt"
    require_smoke
    if [[ -e "$optimization_result" || -e "$table" || -e "$run_root/progress.jsonl" ]]; then
      hash="$(table_hash)"
      echo "REUSE: completed identity-matched optimization, table=$hash" >&2
      exit 0
    fi
    authorize_gpu
    "$python_bin" -m scripts.eval.optimize_olmo_fixed_support_z \
      --authorize --steps 10 \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$ready" \
      --natural-2x-tensor "$natural_tensor" \
      --natural-2x-receipt "$natural_receipt" \
      --output-table "$table" \
      --output-result "$optimization_result" \
      --progress "$run_root/progress.jsonl"
    ;;
  pg19-screen)
    require_common_assets
    require_file "$token_manifest"
    hash="$(table_hash)"
    if reuse_bound_result "$run_root/pg19_screen" "$hash" pg19; then exit 0; fi
    authorize_gpu
    "$python_bin" -m scripts.eval.target_free_formal_eval \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$ready" --skip-checkpoint-rehash \
      --token-manifest "$token_manifest" \
      --method external_table_static \
      --tasks pg19 --multipliers 1 2 --factor 2 --limit-per-cell 5 \
      --table "$table" --table-name direct_z_fixed_support_r2 \
      --table-support native --long-attention-scaling 1.0 \
      --expected-active-sha256 "$hash" \
      --output "$run_root/pg19_screen"
    ;;
  ruler-screen)
    require_common_assets
    require_dir "$ruler_data"
    hash="$(table_hash)"
    require_bound_result "$run_root/pg19_screen" "$hash" pg19
    if reuse_bound_result "$run_root/ruler_screen" "$hash" ruler; then exit 0; fi
    authorize_gpu
    "$python_bin" -m scripts.eval.target_free_ruler_smoke \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$ready" --skip-checkpoint-rehash \
      --data-root "$ruler_data" --lengths 8192 --limit-per-cell 5 \
      --method external_table_static \
      --table "$table" --table-name direct_z_fixed_support_r2 \
      --table-support native --table-factor 2 \
      --long-attention-scaling 1.0 \
      --expected-active-sha256 "$hash" \
      --output "$run_root/ruler_screen"
    ;;
  formal-natural)
    require_common_assets
    require_file "$token_manifest"
    [[ "${DIRECT_Z_PROMOTE:-}" == "YES" ]] || die "inspect both screens, then set DIRECT_Z_PROMOTE=YES"
    hash="$(table_hash)"
    require_bound_result "$run_root/pg19_screen" "$hash" pg19
    require_bound_result "$run_root/ruler_screen" "$hash" ruler
    if reuse_bound_result "$run_root/formal_natural" "$hash" formal; then exit 0; fi
    authorize_gpu
    "$python_bin" -m scripts.eval.target_free_formal_eval \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$ready" --skip-checkpoint-rehash \
      --token-manifest "$token_manifest" \
      --method external_table_static \
      --tasks qasper 2wikimqa --multipliers 1 2 --factor 2 --limit-per-cell 20 \
      --table "$table" --table-name direct_z_fixed_support_r2 \
      --table-support native --long-attention-scaling 1.0 \
      --expected-active-sha256 "$hash" \
      --output "$run_root/formal_natural"
    ;;
  qasper-full|2wiki-full)
    require_common_assets
    require_file "$longbench_zip"
    [[ "${DIRECT_Z_PROMOTE:-}" == "YES" ]] || die "formal promotion requires DIRECT_Z_PROMOTE=YES"
    hash="$(table_hash)"
    require_bound_result "$run_root/formal_natural" "$hash" formal
    task=qasper
    output="$run_root/qasper_full200"
    if [[ "$action" == "2wiki-full" ]]; then
      require_full_result "$run_root/qasper_full200" "$hash" qasper
      task=2wikimqa
      output="$run_root/2wiki_full200"
    fi
    if reuse_full_result "$output" "$hash" "$task"; then exit 0; fi
    authorize_gpu
    "$python_bin" -m rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.evaluate_frozen_2wiki \
      --checkpoint "$checkpoint" --ready-receipt "$ready" \
      --longbench-zip "$longbench_zip" \
      --task "$task" --length 8192 --factor 2 \
      --frequency external_static \
      --table "$table" --table-name direct_z_fixed_support_r2 \
      --expected-table-sha256 "$hash" --long-attention-scaling 1.0 \
      --limit 200 --output "$output"
    ;;
  *)
    echo "usage: $0 {optimize-smoke|optimize|pg19-screen|ruler-screen|formal-natural|qasper-full|2wiki-full}" >&2
    exit 2
    ;;
esac
