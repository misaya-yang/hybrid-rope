#!/usr/bin/env bash
# Weekend τ theory sweep — 50M model, 4×L × 7×τ × 3 seeds = 84 runs
#
# Purpose: generate empirical evidence for three NeurIPS rebuttal questions:
#   Q1 basin flatness  — PPL(τ) curves at multiple L
#   Q2 L-exponent fit  — τ_opt(L) log-log regression
#   Q3 (after sweep)   — long-range attention diffuseness on best ckpt
#
# Each L is called separately with run_evq_sweep.py's built-in multi-τ loop
# and --resume flag so failed runs can be re-launched without losing progress.
#
# Usage:
#   cd ~/neurIPS-2026/hybrid-rope
#   bash scripts/m4_max_36gb/weekend_tau_theory_sweep.sh [TOKENS_PER_RUN]
#
# Default TOKENS_PER_RUN=15000000 (15M). Adjust based on pilot timing.

set -u  # don't fail on empty grep results but catch undefined vars

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

TOKENS_PER_RUN="${1:-25000000}"
BATCH_SIZE="${2:-8}"  # M4 Max safe batch; default auto-adjust (bs=64 at L=1024) triggers MPS OOM
BASE=500000
SEEDS="42,43,44"
# TinyStories (1.8GB cached locally) — FineWeb-Edu download is slow
# from user's network. TinyStories matches paper Table 18's 50M setup.
DATASET="tinystories"
OUT="results/weekend_sweep"
LOGS="$OUT/logs"

mkdir -p "$OUT" "$LOGS"

# r values (τ / τ*(L)); applied uniformly across all L
R_VALUES=(0.00 0.25 0.50 0.75 1.00 1.30 1.70)

# L values: smaller L first (cheaper per run, catches bugs early)
L_VALUES=(256 512 1024 2048)

# τ*(L) = 64 / √L  (d_head = 64 for 50m tier)
declare -A TAU_STAR=(
    [256]=4.0000
    [512]=2.8284
    [1024]=2.0000
    [2048]=1.4142
)

export PYTHONPATH="$REPO_ROOT/scripts/supporting_eval:$REPO_ROOT/scripts/core_text_phases:${PYTHONPATH:-}"
# Force HF offline — prevent streaming=True from validating against the hub
# (TinyStories is fully cached at ~/.cache/huggingface/datasets/roneneldan___tiny_stories/)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Unbuffered stdout so logs flush in real time (without this, Python buffers
# block-wise when writing to file, making tail -f useless)
export PYTHONUNBUFFERED=1
PY_CMD="conda run -n aidemo --no-capture-output python -u"

echo "=============================================================="
echo " Weekend τ Theory Sweep — 50M model"
echo " Started: $(date)"
echo " Tokens per run: ${TOKENS_PER_RUN}"
echo " Dataset: ${DATASET}"
echo " Base: ${BASE}"
echo " Seeds: ${SEEDS}"
echo " L values: ${L_VALUES[*]}"
echo " r values (τ/τ*): ${R_VALUES[*]}"
echo " Total runs: $(( ${#L_VALUES[@]} * ${#R_VALUES[@]} * 3 ))"
echo "=============================================================="

START_TIME=$(date +%s)
RUN_COUNT=0

for L in "${L_VALUES[@]}"; do
    TAU_STAR_VAL="${TAU_STAR[$L]}"
    WORK_DIR="$OUT/L${L}"
    LOG_FILE="$LOGS/L${L}.log"
    mkdir -p "$WORK_DIR"

    # Build τ list = r × τ*(L)
    TAU_LIST=""
    for r in "${R_VALUES[@]}"; do
        TAU_VAL=$(python3 -c "print(f'{$r * $TAU_STAR_VAL:.4f}')")
        TAU_LIST="${TAU_LIST}${TAU_VAL},"
    done
    TAU_LIST="${TAU_LIST%,}"

    echo ""
    echo "--------------------------------------------------------------"
    echo " [L=${L}] τ*(L)=${TAU_STAR_VAL}, τ values: ${TAU_LIST}"
    echo " work_dir=${WORK_DIR}"
    echo " log=${LOG_FILE}"
    echo " started: $(date)"
    echo "--------------------------------------------------------------"

    # call run_evq_sweep.py with --resume so re-starts skip completed runs
    ${PY_CMD} scripts/core_text_phases/run_evq_sweep.py \
        --tier 50m \
        --taus "${TAU_LIST}" \
        --seeds "${SEEDS}" \
        --dataset "${DATASET}" \
        --base "${BASE}" \
        --train_tokens "${TOKENS_PER_RUN}" \
        --seq_len "${L}" \
        --batch_size "${BATCH_SIZE}" \
        --work_dir "${WORK_DIR}" \
        --resume \
        2>&1 | tee -a "${LOG_FILE}"

    RUN_COUNT=$((RUN_COUNT + ${#R_VALUES[@]} * 3))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    HOURS=$(python3 -c "print(f'{$ELAPSED/3600:.2f}')")
    echo ""
    echo " [progress] L=${L} done. Total runs: ${RUN_COUNT}, elapsed: ${HOURS}h"
    echo ""
done

echo ""
echo "=============================================================="
echo " Weekend sweep complete: $(date)"
TOTAL=$(( $(date +%s) - START_TIME ))
echo " Total elapsed: $(python3 -c "print(f'{$TOTAL/3600:.2f}')")h"
echo " Run count: ${RUN_COUNT}"
echo "=============================================================="
