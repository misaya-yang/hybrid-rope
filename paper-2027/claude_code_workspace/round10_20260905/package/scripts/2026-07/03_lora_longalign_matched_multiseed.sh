#!/usr/bin/env bash
# Protocol-matched fallback for the historical LongAlign LoRA experiment.
# No phase launches all six arms accidentally.  See the package README.
set -Eeuo pipefail

PHASE="${1:-}"
case "$PHASE" in
  preflight|prepare-data|geo-control|baseline|seed42|remaining-seeds|eval|summarize) ;;
  *)
    echo "usage: $0 {preflight|prepare-data|geo-control|baseline|seed42|remaining-seeds|eval|summarize}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${EVQ_LORA_PYTHON:-python}"
BASE_DIR="${EVQ_LORA_BASE_DIR:?set EVQ_LORA_BASE_DIR to an external runtime directory}"
MODEL="${EVQ_LORA_MODEL:?set EVQ_LORA_MODEL to the complete LLaMA-3-8B-Instruct directory}"
RAW_LONGALIGN="${EVQ_LEGACY_LONGALIGN_JSONL:?set EVQ_LEGACY_LONGALIGN_JSONL to verified long.jsonl}"
RAW_LONGALIGN_SHA256="${EVQ_LEGACY_LONGALIGN_SHA256:?set EVQ_LEGACY_LONGALIGN_SHA256}"
RAW_WIKITEXT="${EVQ_LEGACY_WIKITEXT_PARQUET:?set EVQ_LEGACY_WIKITEXT_PARQUET to the pinned WT2 test parquet}"
RAW_WIKITEXT_SHA256="${EVQ_LEGACY_WIKITEXT_SHA256:?set EVQ_LEGACY_WIKITEXT_SHA256}"
ROOT="${EVQ_LEGACY_ROOT:-$BASE_DIR/legacy_longalign_lora_v2}"
DATA_DIR="$ROOT/data/longalign"
EVAL_DATA_DIR="$ROOT/data/wikitext2"
MODEL_MANIFEST="$ROOT/data/model_manifest.json"
CHECKPOINTS="$ROOT/checkpoints"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
TELEMETRY="$ROOT/telemetry"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/compile_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/compile_cache/triton}"

mkdir -p "$DATA_DIR" "$EVAL_DATA_DIR" "$CHECKPOINTS" "$RESULTS" "$LOGS" "$TELEMETRY" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

require_file() {
  test -s "$1" || { echo "required file missing: $1" >&2; exit 1; }
}

run_name() {
  local method="$1" seed="$2"
  if [[ "$method" == native_geo ]]; then
    echo "geo_longalign_s${seed}"
  else
    echo "evq_longalign_tau1414_s${seed}"
  fi
}

adapter_dir() {
  echo "$CHECKPOINTS/$(run_name "$1" "$2")"
}

record_gpu() {
  local label="$1"
  nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader > "$TELEMETRY/${label}_preflight.csv"
}

train_arm() {
  local method="$1" seed="$2" name out log telemetry_pid=""
  name="$(run_name "$method" "$seed")"
  out="$(adapter_dir "$method" "$seed")"
  log="$LOGS/train_${name}.log"
  if "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/validate_legacy_lora_artifact.py" \
      --adapter_dir "$out" --expected_method "$method" --expected_seed "$seed" >/dev/null 2>&1; then
    echo "[skip] validated final adapter: $name"
    return
  fi
  record_gpu "$name"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader --loop-ms=5000 > "$TELEMETRY/${name}.csv" &
  telemetry_pid=$!
  trap 'if [[ -n "${telemetry_pid:-}" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' RETURN
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/train_evq_lora.py" \
    --model_name "$MODEL" \
    --output_dir "$out" \
    --rope_method "$method" \
    --tau 1.414 \
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --lora_targets q_proj,k_proj,v_proj,o_proj \
    --max_steps 300 --per_device_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --warmup_steps 60 --weight_decay 0.01 --max_grad_norm 1.0 \
    --max_seq_len 8192 --max_samples 8000 --save_steps 100 \
    --seed "$seed" --bf16 --no_4bit --compile --compile_mode default \
    --strict_legacy_protocol \
    --prepared_data_manifest "$DATA_DIR/manifest.json" \
    --model_manifest "$MODEL_MANIFEST" \
    --resume_from_checkpoint auto 2>&1 | tee -a "$log"
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=""
  "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/validate_legacy_lora_artifact.py" \
    --adapter_dir "$out" --expected_method "$method" --expected_seed "$seed"
}

eval_variant() {
  local variant="$1" method="${2:-}" seed="${3:-}" args=()
  if [[ -n "$method" ]]; then
    args+=(--adapter_dir "$(adapter_dir "$method" "$seed")")
  fi
  if "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
      --model_name "$MODEL" --model_manifest "$MODEL_MANIFEST" \
      --eval_manifest "$EVAL_DATA_DIR/manifest.json" --variant "$variant" \
      --output_dir "$RESULTS" "${args[@]}" --validate_only >/dev/null 2>&1; then
    echo "[skip] validated evaluation: $variant"
    return
  fi
  record_gpu "eval_${variant}"
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
    --model_name "$MODEL" \
    --model_manifest "$MODEL_MANIFEST" \
    --eval_manifest "$EVAL_DATA_DIR/manifest.json" \
    --variant "$variant" \
    --output_dir "$RESULTS" \
    "${args[@]}" 2>&1 | tee -a "$LOGS/eval_${variant}.log"
}

preflight() {
  command -v "$PYTHON" >/dev/null
  require_file "$MODEL/config.json"
  require_file "$RAW_LONGALIGN"
  require_file "$RAW_WIKITEXT"
  printf '%s  %s\n' "$RAW_LONGALIGN_SHA256" "$RAW_LONGALIGN" | sha256sum -c -
  printf '%s  %s\n' "$RAW_WIKITEXT_SHA256" "$RAW_WIKITEXT" | sha256sum -c -
  "$PYTHON" - <<'PY'
import importlib
for name in ("torch", "transformers", "peft", "datasets"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY
  if command -v nvidia-smi >/dev/null; then nvidia-smi -L; fi
}

run_geo_control() {
  [[ "${EVQ_LEGACY_UNLOCK_GEO_CONTROL:-NO}" == YES ]] || {
    echo "geo-control is cost-gated; set EVQ_LEGACY_UNLOCK_GEO_CONTROL=YES explicitly" >&2
    exit 1
  }
  preflight
  require_file "$MODEL_MANIFEST"
  require_file "$DATA_DIR/manifest.json"
  require_file "$EVAL_DATA_DIR/manifest.json"
  eval_variant base_geo
  train_arm native_geo 42
  eval_variant geo_longalign_s42 native_geo 42
  "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/summarize_legacy_geo_control.py" \
    --results_dir "$RESULTS" --output "$ROOT/legacy_geo_control_summary.json"
}

case "$PHASE" in
  preflight)
    preflight
    ;;
  prepare-data)
    preflight
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/prepare_legacy_model_manifest.py" \
      --model_dir "$MODEL" --output "$MODEL_MANIFEST"
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/prepare_legacy_longalign_data.py" \
      --raw_jsonl "$RAW_LONGALIGN" --expected_raw_sha256 "$RAW_LONGALIGN_SHA256" \
      --tokenizer "$MODEL" --output_dir "$DATA_DIR"
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/prepare_legacy_wikitext.py" \
      --parquet "$RAW_WIKITEXT" --expected_raw_sha256 "$RAW_WIKITEXT_SHA256" \
      --tokenizer "$MODEL" --output_dir "$EVAL_DATA_DIR"
    ;;
  geo-control)
    run_geo_control
    ;;
  baseline)
    require_file "$MODEL_MANIFEST"
    require_file "$EVAL_DATA_DIR/manifest.json"
    eval_variant base_geo
    eval_variant base_evq_tau1414
    echo "Inspect the verified Base-Geo row before setting EVQ_LEGACY_UNLOCK_SEED42=YES."
    ;;
  seed42)
    [[ "${EVQ_LEGACY_UNLOCK_SEED42:-NO}" == YES ]] || {
      echo "seed42 is cost-gated; inspect baseline results, then set EVQ_LEGACY_UNLOCK_SEED42=YES" >&2
      exit 1
    }
    require_file "$RESULTS/eval_base_geo.json"
    train_arm evq_cosh 42
    eval_variant evq_longalign_tau1414_s42 evq_cosh 42
    train_arm native_geo 42
    eval_variant geo_longalign_s42 native_geo 42
    echo "Inspect the matched seed-42 pair before unlocking seeds 43/44."
    ;;
  remaining-seeds)
    [[ "${EVQ_LEGACY_UNLOCK_REMAINING:-NO}" == YES ]] || {
      echo "remaining seeds are cost-gated; set EVQ_LEGACY_UNLOCK_REMAINING=YES explicitly" >&2
      exit 1
    }
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
      --model_name "$MODEL" --model_manifest "$MODEL_MANIFEST" \
      --eval_manifest "$EVAL_DATA_DIR/manifest.json" --variant geo_longalign_s42 \
      --adapter_dir "$(adapter_dir native_geo 42)" --output_dir "$RESULTS" --validate_only
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
      --model_name "$MODEL" --model_manifest "$MODEL_MANIFEST" \
      --eval_manifest "$EVAL_DATA_DIR/manifest.json" --variant evq_longalign_tau1414_s42 \
      --adapter_dir "$(adapter_dir evq_cosh 42)" --output_dir "$RESULTS" --validate_only
    for seed in 43 44; do
      train_arm native_geo "$seed"
      train_arm evq_cosh "$seed"
    done
    ;;
  eval)
    eval_variant base_geo
    eval_variant base_evq_tau1414
    for seed in 42 43 44; do
      eval_variant "geo_longalign_s${seed}" native_geo "$seed"
      eval_variant "evq_longalign_tau1414_s${seed}" evq_cosh "$seed"
    done
    ;;
  summarize)
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/summarize_legacy_lora_matched.py" \
      --results_dir "$RESULTS" --output "$ROOT/legacy_lora_multiseed_summary.json"
    ;;
esac
