#!/usr/bin/env bash
# quick_eval_a1_vs_a2.sh — Monitor A2 training completion, then run lb6 eval for A1 vs A2.
# Usage: bash scripts/isolated/longinst/quick_eval_a1_vs_a2.sh
set -euo pipefail

# ═══════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════
BASE_MODEL="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
LONGBENCH_DATA="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"
EVAL_SCRIPT="/root/autodl-tmp/dfrope/hybrid-rope/scripts/eval_longbench.py"
ARTIFACT_ROOT="/root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1"

# A1: geometric (tau=0), checkpoint-400 from the 800-step run
A1_RUN_DIR="${ARTIFACT_ROOT}/train/A1_geometric_tau0p00_r32_s800_seed42__20260228_115150"
A1_ADAPTER="${A1_RUN_DIR}/checkpoint-400"
A1_INV_FREQ="${A1_RUN_DIR}/artifacts/custom_inv_freq.pt"

# A2: EVQ-Cosh (tau=1.5), auto-detect from training
A2_RUN_DIR="${ARTIFACT_ROOT}/train/A2_evq_cosh_tau1p50_r32_s400_seed42__20260228_135110"
A2_TRAIN_LOG="${A2_RUN_DIR}/train_log.jsonl"

# Output directory for eval results
EVAL_OUT="/root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1/quick_eval_a1_vs_a2"
mkdir -p "${EVAL_OUT}"

# ═══════════════════════════════════════════════════════
# PHASE 1: Wait for A2 training to finish
# ═══════════════════════════════════════════════════════
echo "[MONITOR] Waiting for A2 training to reach step 400..."
echo "[MONITOR] Polling ${A2_TRAIN_LOG} every 60s"

while true; do
    if [ ! -f "${A2_TRAIN_LOG}" ]; then
        echo "[MONITOR] $(date +%H:%M:%S) train_log.jsonl not found yet, waiting..."
        sleep 60
        continue
    fi

    # Get the last logged step
    LAST_STEP=$(tail -1 "${A2_TRAIN_LOG}" 2>/dev/null | /root/miniconda3/bin/python -c "
import sys, json
try:
    d = json.loads(sys.stdin.read().strip())
    print(d.get('step', 0))
except:
    print(0)
" 2>/dev/null || echo 0)

    echo "[MONITOR] $(date +%H:%M:%S) A2 at step ${LAST_STEP}/400"

    if [ "${LAST_STEP}" -ge 400 ]; then
        echo "[MONITOR] A2 training complete!"
        break
    fi

    # Also check if training process is still alive
    if ! pgrep -f "A2_evq_cosh" > /dev/null 2>&1; then
        # Process gone — check if it completed or crashed
        if [ "${LAST_STEP}" -ge 390 ]; then
            echo "[MONITOR] A2 process exited near completion (step ${LAST_STEP}). Proceeding."
            break
        else
            echo "[MONITOR] WARNING: A2 process not found and only at step ${LAST_STEP}. Checking for checkpoint..."
            if [ -f "${A2_RUN_DIR}/checkpoint-400/adapter_model.safetensors" ]; then
                echo "[MONITOR] checkpoint-400 exists. Proceeding."
                break
            fi
            echo "[MONITOR] No checkpoint-400 found. A2 may have crashed. Waiting 120s..."
            sleep 120
            continue
        fi
    fi

    sleep 60
done

# Wait a bit for any final file writes
sleep 10

# ═══════════════════════════════════════════════════════
# PHASE 2: Locate A2 checkpoint and inv_freq
# ═══════════════════════════════════════════════════════
echo ""
echo "[EVAL] Locating A2 artifacts..."

# Find the best checkpoint (prefer checkpoint-400, fallback to latest)
if [ -d "${A2_RUN_DIR}/checkpoint-400" ]; then
    A2_ADAPTER="${A2_RUN_DIR}/checkpoint-400"
elif [ -d "${A2_RUN_DIR}/checkpoint-200" ]; then
    A2_ADAPTER="${A2_RUN_DIR}/checkpoint-200"
    echo "[WARN] Using checkpoint-200 (checkpoint-400 not found)"
else
    # Find latest checkpoint
    A2_ADAPTER=$(ls -dt "${A2_RUN_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -z "${A2_ADAPTER}" ]; then
        echo "[ERROR] No checkpoint found in ${A2_RUN_DIR}"
        exit 1
    fi
    echo "[WARN] Using latest checkpoint: ${A2_ADAPTER}"
fi

# Find A2 inv_freq
A2_INV_FREQ="${A2_RUN_DIR}/artifacts/custom_inv_freq.pt"
if [ ! -f "${A2_INV_FREQ}" ]; then
    echo "[ERROR] A2 custom_inv_freq.pt not found at ${A2_INV_FREQ}"
    exit 1
fi

echo "[EVAL] A1 adapter: ${A1_ADAPTER}"
echo "[EVAL] A1 inv_freq: ${A1_INV_FREQ}"
echo "[EVAL] A2 adapter: ${A2_ADAPTER}"
echo "[EVAL] A2 inv_freq: ${A2_INV_FREQ}"

# Verify all files exist
for f in "${A1_ADAPTER}/adapter_model.safetensors" "${A1_INV_FREQ}" \
         "${A2_ADAPTER}/adapter_model.safetensors" "${A2_INV_FREQ}"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing file: $f"
        exit 1
    fi
done
echo "[EVAL] All artifacts verified."

# ═══════════════════════════════════════════════════════
# PHASE 3: Run lb6 eval for A1 (geometric baseline)
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[EVAL] Running LB6 eval for A1 (geometric tau=0.0)..."
echo "═══════════════════════════════════════════════════════"

/root/miniconda3/bin/python "${EVAL_SCRIPT}" \
    --base_model_path "${BASE_MODEL}" \
    --adapter_path "${A1_ADAPTER}" \
    --custom_inv_freq_path "${A1_INV_FREQ}" \
    --model_alias "A1_geometric" \
    --variant custom \
    --task_set lb6 \
    --max_input_tokens 8192 \
    --batch_size 4 \
    --max_batch_input_tokens 32768 \
    --attn_implementation sdpa \
    --skip_base_unfinetuned \
    --score_scale pct \
    --longbench_local_data_dir "${LONGBENCH_DATA}" \
    --output_json "${EVAL_OUT}/a1_geometric_lb6.json" \
    --save_per_sample_traces 0 \
    2>&1 | tee "${EVAL_OUT}/a1_eval.log"

echo "[EVAL] A1 eval done."

# ═══════════════════════════════════════════════════════
# PHASE 4: Run lb6 eval for A2 (EVQ-Cosh tau=1.5)
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[EVAL] Running LB6 eval for A2 (EVQ-Cosh tau=1.5)..."
echo "═══════════════════════════════════════════════════════"

/root/miniconda3/bin/python "${EVAL_SCRIPT}" \
    --base_model_path "${BASE_MODEL}" \
    --adapter_path "${A2_ADAPTER}" \
    --custom_inv_freq_path "${A2_INV_FREQ}" \
    --model_alias "A2_evq_cosh" \
    --variant custom \
    --task_set lb6 \
    --max_input_tokens 8192 \
    --batch_size 4 \
    --max_batch_input_tokens 32768 \
    --attn_implementation sdpa \
    --skip_base_unfinetuned \
    --score_scale pct \
    --longbench_local_data_dir "${LONGBENCH_DATA}" \
    --output_json "${EVAL_OUT}/a2_evq_cosh_lb6.json" \
    --save_per_sample_traces 0 \
    2>&1 | tee "${EVAL_OUT}/a2_eval.log"

echo "[EVAL] A2 eval done."

# ═══════════════════════════════════════════════════════
# PHASE 5: Compare results side-by-side
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[COMPARE] A1 (geometric) vs A2 (EVQ-Cosh) on LB6"
echo "═══════════════════════════════════════════════════════"

/root/miniconda3/bin/python - "${EVAL_OUT}/a1_geometric_lb6.json" "${EVAL_OUT}/a2_evq_cosh_lb6.json" <<'PYEOF'
import json, sys
from pathlib import Path

a1_path, a2_path = sys.argv[1], sys.argv[2]

def load_scores(path):
    data = json.loads(Path(path).read_text())
    # The eval script writes results under a model key
    scores = {}
    for model_key, model_data in data.items():
        if isinstance(model_data, dict) and "tasks" in model_data:
            for task_name, task_data in model_data["tasks"].items():
                if isinstance(task_data, dict):
                    scores[task_name] = task_data.get("score", 0.0)
            scores["__avg__"] = model_data.get("avg_score", 0.0)
            break
    # Fallback: try flat structure
    if not scores:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                scores[k] = v
            elif isinstance(v, dict) and "score" in v:
                scores[k] = v["score"]
    return scores

a1 = load_scores(a1_path)
a2 = load_scores(a2_path)

all_tasks = sorted(set(list(a1.keys()) + list(a2.keys())) - {"__avg__"})

print(f"\n{'Task':<20} {'A1 (geo)':>10} {'A2 (EVQ)':>10} {'Delta':>10} {'Winner':>8}")
print("-" * 62)

a1_total, a2_total, count = 0.0, 0.0, 0
for t in all_tasks:
    s1 = a1.get(t, 0.0)
    s2 = a2.get(t, 0.0)
    delta = s2 - s1
    winner = "A2" if delta > 0.5 else ("A1" if delta < -0.5 else "~tie")
    print(f"{t:<20} {s1:>10.2f} {s2:>10.2f} {delta:>+10.2f} {winner:>8}")
    a1_total += s1
    a2_total += s2
    count += 1

if count > 0:
    a1_avg = a1_total / count
    a2_avg = a2_total / count
    delta_avg = a2_avg - a1_avg
    winner_avg = "A2" if delta_avg > 0.5 else ("A1" if delta_avg < -0.5 else "~tie")
    print("-" * 62)
    print(f"{'AVERAGE':<20} {a1_avg:>10.2f} {a2_avg:>10.2f} {delta_avg:>+10.2f} {winner_avg:>8}")

# Also print stored averages if available
if "__avg__" in a1 or "__avg__" in a2:
    a1a = a1.get("__avg__", 0.0)
    a2a = a2.get("__avg__", 0.0)
    print(f"{'(stored avg)':<20} {a1a:>10.2f} {a2a:>10.2f} {a2a-a1a:>+10.2f}")

print()
if count > 0 and delta_avg > 1.0:
    print(">>> RESULT: A2 (EVQ-Cosh tau=1.5) shows meaningful improvement over A1 (geometric).")
    print(">>> RECOMMENDATION: Proceed with B1/B2 stability testing.")
elif count > 0 and delta_avg < -1.0:
    print(">>> RESULT: A1 (geometric) outperforms A2 (EVQ-Cosh). Investigate before continuing.")
else:
    print(">>> RESULT: Performance roughly comparable. Check per-task patterns for insights.")
PYEOF

echo ""
echo "[DONE] Results saved to ${EVAL_OUT}/"
echo "[DONE] A1 results: ${EVAL_OUT}/a1_geometric_lb6.json"
echo "[DONE] A2 results: ${EVAL_OUT}/a2_evq_cosh_lb6.json"
