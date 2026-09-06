#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"; shift || true
ROOT="${EVQ_REPO_ROOT:?set EVQ_REPO_ROOT}"
PYTHON="${EVQ_PYTHON:-/root/miniconda3/bin/python}"
WORK="${LOG_P2_PROGRAM_ROOT:?set LOG_P2_PROGRAM_ROOT}"
CHECKPOINT="${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}"
TOKEN_MANIFEST="${EVQ_TOKEN_MANIFEST:?set EVQ_TOKEN_MANIFEST}"
RULER_DATA="${EVQ_RULER_DATA:?set EVQ_RULER_DATA}"
NATIVE_TABLE="${EVQ_NATIVE_TABLE:?set EVQ_NATIVE_TABLE}"
REFERENCE_S4="${EVQ_LOG_P2_S4_TABLE:?set EVQ_LOG_P2_S4_TABLE}"
REPLAY_DATA="${EVQ_IDENTIFIABLE_4K_DATA:?set EVQ_IDENTIFIABLE_4K_DATA}"
NEAR_DATA="${EVQ_IDENTIFIABLE_8K_DATA:?set EVQ_IDENTIFIABLE_8K_DATA}"
FAR_DATA="${EVQ_IDENTIFIABLE_16K_DATA:?set EVQ_IDENTIFIABLE_16K_DATA}"
GRID="$WORK/grid"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

require_gpu() {
  [[ "${LOG_P2_PROGRAM_GPU_AUTHORIZED:-}" == "YES" ]] || {
    echo "set LOG_P2_PROGRAM_GPU_AUTHORIZED=YES after explicit GPU authorization" >&2
    exit 3
  }
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 CUDA is unavailable")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
print(torch.cuda.get_device_name(0))
PY
}

manifest_value() {
  "$PYTHON" - "$GRID/manifest.json" "$1" "$2" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))["arms"][sys.argv[2]][sys.argv[3]]
print(value)
PY
}

table_value() {
  local arm="$1" key="$2" table
  table="$(manifest_value "$arm" table)"
  "$PYTHON" - "$GRID/manifest.json" "$table" "$key" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["tables"][sys.argv[2]][sys.argv[3]])
PY
}

external_formal_args() {
  local arm="$1"
  printf '%s\n' \
    --method external_table_static \
    --table "$GRID/$(table_value "$arm" path)" \
    --table-name "$arm" \
    --table-support native_div_factor \
    --factor "$(manifest_value "$arm" factor)" \
    --long-attention-scaling "$(manifest_value "$arm" attention_scaling)" \
    --expected-active-sha256 "$(table_value "$arm" float32_sha256)"
}

external_ruler_args() {
  local arm="$1"
  printf '%s\n' \
    --method external_table_static \
    --table "$GRID/$(table_value "$arm" path)" \
    --table-name "$arm" \
    --table-support native_div_factor \
    --table-factor "$(manifest_value "$arm" factor)" \
    --long-attention-scaling "$(manifest_value "$arm" attention_scaling)" \
    --expected-active-sha256 "$(table_value "$arm" float32_sha256)"
}

formal() {
  local label="$1" output="$2"; shift 2
  "$PYTHON" "$ROOT/scripts/eval/target_free_formal_eval.py" \
    --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
    --label "$label" --output "$output" "$@"
}

ruler() {
  local output="$1"; shift
  "$PYTHON" "$ROOT/scripts/eval/target_free_ruler_smoke.py" \
    --checkpoint "$CHECKPOINT" --data-root "$RULER_DATA" \
    --native-context-length 4096 --output "$output" "$@"
}

require_retention_artifacts() {
  local stage="$1" arm="$2" cell
  for cell in "${arm}_pg19" "${arm}_tasks" "${arm}_ruler"; do
    [[ -f "$WORK/$stage/$cell/results.json" ]] || {
      echo "missing retention result: $WORK/$stage/$cell/results.json" >&2
      exit 5
    }
  done
}

case "$ACTION" in
  prepare-ruler)
    RULER_SOURCE="${EVQ_RULER_SOURCE:?set EVQ_RULER_SOURCE}"
    "$PYTHON" -m \
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer \
      --ruler-root "$RULER_SOURCE" --checkpoint "$CHECKPOINT" --output "$RULER_DATA" \
      --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths 4096 8192 16384 32768 65536 131072 \
      --samples-per-cell 20 --workers "${RULER_PREP_WORKERS:-1}" --seed 20260904
    ;;
  build)
    "$PYTHON" "$ROOT/scripts/analysis/export_log_p2_factor_frontier.py" \
      --native-table "$NATIVE_TABLE" --reference-log-s4 "$REFERENCE_S4" \
      --factors 4 5 6 7 8 --gain-coefficients 0.05 0.074 0.10 \
      --output "$GRID"
    ;;
  preflight-retention)
    [[ $# -ge 1 ]] || { echo "usage: $0 preflight-retention <arm> [...]" >&2; exit 2; }
    for arm in "$@"; do
      mapfile -t formal_args < <(external_formal_args "$arm")
      mapfile -t ruler_args < <(external_ruler_args "$arm")
      formal "$arm" "$WORK/preflight/${arm}_pg19" \
        --tasks pg19 --multipliers 1 --limit-per-cell 1 --preflight-only "${formal_args[@]}"
      formal "$arm" "$WORK/preflight/${arm}_tasks" \
        --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa \
        --multipliers 1 --limit-per-cell 1 --preflight-only "${formal_args[@]}"
      ruler "$WORK/preflight/${arm}_ruler" \
        --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
        --lengths 4096 --limit-per-cell 1 --preflight-only "${ruler_args[@]}"
    done
    ;;
  retention-native)
    require_gpu
    formal native "$WORK/retention/native_pg19" \
      --method native --tasks pg19 --multipliers 1 --limit-per-cell 20
    formal native "$WORK/retention/native_tasks" \
      --method native --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa \
      --multipliers 1
    ruler "$WORK/retention/native_ruler" \
      --method native --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths 4096 --limit-per-cell 20
    ;;
  retention-arm)
    [[ $# -eq 1 ]] || { echo "usage: $0 retention-arm <arm>" >&2; exit 2; }
    require_gpu
    arm="$1"; mapfile -t formal_args < <(external_formal_args "$arm")
    mapfile -t ruler_args < <(external_ruler_args "$arm")
    formal "$arm" "$WORK/retention/${arm}_pg19" \
      --tasks pg19 --multipliers 1 --limit-per-cell 20 "${formal_args[@]}"
    formal "$arm" "$WORK/retention/${arm}_tasks" \
      --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa \
      --multipliers 1 "${formal_args[@]}"
    ruler "$WORK/retention/${arm}_ruler" \
      --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths 4096 --limit-per-cell 20 "${ruler_args[@]}"
    ;;
  zero-far)
    [[ $# -eq 1 ]] || { echo "usage: $0 zero-far <arm>" >&2; exit 2; }
    require_gpu
    arm="$1"; mapfile -t args < <(external_ruler_args "$arm")
    require_retention_artifacts retention "$arm"
    [[ "${LOG_P2_ZERO_RETENTION_VERIFIED:-}" == "$arm" ]] || {
      echo "set LOG_P2_ZERO_RETENTION_VERIFIED=$arm after verifying every 1x gate" >&2
      exit 5
    }
    ruler "$WORK/zero_far/$arm" \
      --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths 8192 16384 32768 --profile-target-length 32768 \
      --limit-per-cell 20 "${args[@]}"
    ;;
  train-preflight|train-smoke|train)
    [[ $# -eq 1 ]] || { echo "usage: $0 $ACTION <arm>" >&2; exit 2; }
    arm="$1"
    table="$GRID/$(table_value "$arm" path)"
    scaling="$(manifest_value "$arm" attention_scaling)"
    output="$WORK/${ACTION%-preflight}/$arm"
    steps=300
    lora_rank="${LORA_RANK:-64}"
    lora_alpha="${LORA_ALPHA:-128}"
    lora_modules=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})
    extra=()
    if [[ "$ACTION" == train-preflight ]]; then output="$WORK/train/$arm"; extra+=(--preflight); fi
    if [[ "$ACTION" == train-smoke ]]; then require_gpu; output="$WORK/smoke/$arm"; steps=1; extra+=(--authorized); fi
    if [[ "$ACTION" == train ]]; then
      require_gpu
      [[ -f "$WORK/smoke/$arm/receipt.json" ]] || { echo "missing completed smoke receipt" >&2; exit 4; }
      extra+=(--authorized)
    fi
    "$PYTHON" "$ROOT/scripts/train/train_log_p2_phase_transfer_lora.py" \
      --checkpoint "$CHECKPOINT" --table "$table" --attention-scaling "$scaling" \
      --replay-data "$REPLAY_DATA" --near-data "$NEAR_DATA" --far-data "$FAR_DATA" \
      --steps "$steps" --rank "$lora_rank" --alpha "$lora_alpha" --target-modules "${lora_modules[@]}" \
      --output "$output" "${extra[@]}"
    ;;
  eval-adapted-retention)
    [[ $# -eq 1 ]] || { echo "usage: $0 eval-adapted-retention <arm>" >&2; exit 2; }
    require_gpu
    arm="$1"; mapfile -t formal_args < <(external_formal_args "$arm")
    mapfile -t ruler_args < <(external_ruler_args "$arm")
    lora_rank="${LORA_RANK:-64}"
    adapter_name="${ADAPTER_NAME:-${arm}_physical_2x4x_r${lora_rank}}"
    adapter=(--adapter "$WORK/train/$arm/adapter" --adapter-name "$adapter_name")
    formal "$arm" "$WORK/adapted_retention/${arm}_pg19" \
      --tasks pg19 --multipliers 1 --limit-per-cell 20 "${adapter[@]}" "${formal_args[@]}"
    formal "$arm" "$WORK/adapted_retention/${arm}_tasks" \
      --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa \
      --multipliers 1 "${adapter[@]}" "${formal_args[@]}"
    ruler "$WORK/adapted_retention/${arm}_ruler" \
      --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths 4096 --limit-per-cell 20 "${adapter[@]}" "${ruler_args[@]}"
    ;;
  eval-far)
    [[ $# -eq 1 ]] || { echo "usage: $0 eval-far <arm>" >&2; exit 2; }
    require_gpu
    arm="$1"
    require_retention_artifacts adapted_retention "$arm"
    [[ "${LOG_P2_ADAPTED_RETENTION_VERIFIED:-}" == "$arm" ]] || {
      echo "set LOG_P2_ADAPTED_RETENTION_VERIFIED=$arm after verifying every 1x gate" >&2
      exit 5
    }
    mapfile -t args < <(external_ruler_args "$arm")
    lora_rank="${LORA_RANK:-64}"
    adapter_name="${ADAPTER_NAME:-${arm}_physical_2x4x_r${lora_rank}}"
    # 32K and 64K run on 16GB VRAM; 131K (32x) requires 32GB+ VRAM (RTX 5090)
    far_lengths=(${FAR_LENGTHS:-32768 65536 131072})
    max_len="${far_lengths[-1]}"
    ruler "$WORK/adapted_far/$arm" \
      --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
      --lengths "${far_lengths[@]}" --profile-target-length "$max_len" \
      --limit-per-cell 20 --adapter "$WORK/train/$arm/adapter" \
      --adapter-name "$adapter_name" "${args[@]}"
    ;;
  *)
    echo "usage: $0 {prepare-ruler|build|preflight-retention|retention-native|retention-arm|zero-far|train-preflight|train-smoke|train|eval-adapted-retention|eval-far}" >&2
    exit 2
    ;;
esac
