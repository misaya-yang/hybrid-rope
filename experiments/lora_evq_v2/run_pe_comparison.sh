#!/bin/bash
# =============================================================
# PE Baseline Comparison — 一键跑全部训练 + 评测
# =============================================================
# 用法:
#   bash run_pe_comparison.sh train    # 跑全部训练 (9 runs, ~18h)
#   bash run_pe_comparison.sh eval     # 跑全部评测 (~1h)
#   bash run_pe_comparison.sh all      # 训练 + 评测
# =============================================================
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
DATA="/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl"
WIKI="/root/autodl-tmp/data/wikitext2/wikitext2_test.txt"
CKPT="${SCRIPT_DIR}/checkpoints"
RESULT="${SCRIPT_DIR}/results"

mkdir -p "${RESULT}" "${SCRIPT_DIR}/logs"

# Pre-flight
for f in "${MODEL}/config.json" "${DATA}" "${WIKI}"; do
    if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 1; fi
done

train_all() {
    echo "================================================"
    echo "TRAINING: 3 methods × 3 seeds = 9 runs"
    echo "================================================"

    # ---- Geometric (τ=0) × 3 seeds ----
    for SEED in 42 43 44; do
        DIR="${CKPT}/geo_s${SEED}"
        if [ -f "${DIR}/adapter_model.safetensors" ]; then
            echo "[SKIP] ${DIR} already exists"
            continue
        fi
        echo ""
        echo "=== Geometric seed=${SEED} ==="
        python "${SCRIPT_DIR}/train_evq_lora.py" \
            --model_name "${MODEL}" \
            --output_dir "${DIR}" \
            --tau 0 \
            --local_data_path "${DATA}" \
            --seed ${SEED} \
            --max_steps 300 \
            2>&1 | tee "${SCRIPT_DIR}/logs/geo_s${SEED}.log"
    done

    # ---- EVQ (τ=1.414) × 3 seeds ----
    # seed42 already exists as evq_r64_tau1414
    for SEED in 42 43 44; do
        if [ "${SEED}" = "42" ]; then
            # Reuse existing checkpoint
            if [ -f "${CKPT}/evq_r64_tau1414/adapter_model.safetensors" ]; then
                echo "[SKIP] EVQ seed=42 already exists (evq_r64_tau1414)"
                continue
            fi
        fi
        DIR="${CKPT}/evq_s${SEED}"
        if [ -f "${DIR}/adapter_model.safetensors" ]; then
            echo "[SKIP] ${DIR} already exists"
            continue
        fi
        echo ""
        echo "=== EVQ seed=${SEED} ==="
        python "${SCRIPT_DIR}/train_evq_lora.py" \
            --model_name "${MODEL}" \
            --output_dir "${DIR}" \
            --tau 1.414 \
            --local_data_path "${DATA}" \
            --seed ${SEED} \
            --max_steps 300 \
            2>&1 | tee "${SCRIPT_DIR}/logs/evq_s${SEED}.log"
    done

    # ---- YaRN (factor=2) × 3 seeds ----
    for SEED in 42 43 44; do
        DIR="${CKPT}/yarn_s${SEED}"
        if [ -f "${DIR}/adapter_model.safetensors" ]; then
            echo "[SKIP] ${DIR} already exists"
            continue
        fi
        echo ""
        echo "=== YaRN seed=${SEED} ==="
        python "${SCRIPT_DIR}/train_yarn_lora.py" \
            --model_name "${MODEL}" \
            --output_dir "${DIR}" \
            --yarn_factor 2.0 \
            --local_data_path "${DATA}" \
            --seed ${SEED} \
            --max_steps 300 \
            2>&1 | tee "${SCRIPT_DIR}/logs/yarn_s${SEED}.log"
    done

    echo ""
    echo "================================================"
    echo "ALL TRAINING COMPLETE"
    echo "================================================"
}

eval_all() {
    echo "================================================"
    echo "EVALUATION: Positional PPL for all checkpoints"
    echo "================================================"

    # Base (no adapter)
    echo ""
    echo "=== Base (no adapter) ==="
    python "${SCRIPT_DIR}/eval_positional_ppl.py" \
        --model_name "${MODEL}" \
        --base_only \
        --wikitext_path "${WIKI}" \
        --output "${RESULT}/ppl_base.json"

    # Geometric
    for SEED in 42 43 44; do
        DIR="${CKPT}/geo_s${SEED}"
        if [ ! -f "${DIR}/adapter_model.safetensors" ]; then continue; fi
        echo ""
        echo "=== Geo seed=${SEED} ==="
        python "${SCRIPT_DIR}/eval_positional_ppl.py" \
            --model_name "${MODEL}" \
            --adapter_dir "${DIR}" \
            --method geo \
            --wikitext_path "${WIKI}" \
            --output "${RESULT}/ppl_geo_s${SEED}.json"
    done

    # EVQ
    for SEED in 42 43 44; do
        if [ "${SEED}" = "42" ]; then
            DIR="${CKPT}/evq_r64_tau1414"
        else
            DIR="${CKPT}/evq_s${SEED}"
        fi
        if [ ! -f "${DIR}/adapter_model.safetensors" ]; then continue; fi
        echo ""
        echo "=== EVQ seed=${SEED} ==="
        python "${SCRIPT_DIR}/eval_positional_ppl.py" \
            --model_name "${MODEL}" \
            --adapter_dir "${DIR}" \
            --method evq \
            --wikitext_path "${WIKI}" \
            --output "${RESULT}/ppl_evq_s${SEED}.json"
    done

    # YaRN
    for SEED in 42 43 44; do
        DIR="${CKPT}/yarn_s${SEED}"
        if [ ! -f "${DIR}/adapter_model.safetensors" ]; then continue; fi
        echo ""
        echo "=== YaRN seed=${SEED} ==="
        python "${SCRIPT_DIR}/eval_positional_ppl.py" \
            --model_name "${MODEL}" \
            --adapter_dir "${DIR}" \
            --method yarn \
            --yarn_factor 2.0 \
            --wikitext_path "${WIKI}" \
            --output "${RESULT}/ppl_yarn_s${SEED}.json"
    done

    # Summary
    echo ""
    echo "================================================"
    echo "SUMMARY"
    echo "================================================"
    python -c "
import json, glob, os, numpy as np

results_dir = '${RESULT}'
methods = {}
for f in sorted(glob.glob(os.path.join(results_dir, 'ppl_*.json'))):
    with open(f) as fh:
        d = json.load(fh)
    label = d.get('label', os.path.basename(f))
    method = d.get('method', 'unknown')
    if method not in methods:
        methods[method] = {}
    for window, r in d.get('results', {}).items():
        if window not in methods[method]:
            methods[method][window] = []
        methods[method][window].append(r['ppl'])

print(f\"{'Method':<12s}  {'0-4K':>12s}  {'4K-8K':>12s}  {'8K-12K':>12s}  {'12K-16K':>12s}\")
print('-' * 62)
for method in ['base', 'geo', 'evq', 'yarn']:
    if method not in methods:
        continue
    row = f'{method:<12s}'
    for w in ['0-4K', '4K-8K', '8K-12K', '12K-16K']:
        vals = methods[method].get(w, [])
        if vals:
            mean = np.mean(vals)
            std = np.std(vals) if len(vals) > 1 else 0
            row += f'  {mean:6.1f}±{std:4.1f}'
        else:
            row += f'  {\"N/A\":>12s}'
    print(row)
"

    echo ""
    echo "All results in: ${RESULT}/"
}

case "${1:-all}" in
    train) train_all ;;
    eval)  eval_all ;;
    all)   train_all; eval_all ;;
    *)     echo "Usage: bash run_pe_comparison.sh [train|eval|all]"; exit 1 ;;
esac
