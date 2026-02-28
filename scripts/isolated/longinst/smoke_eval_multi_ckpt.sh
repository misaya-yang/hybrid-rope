#!/usr/bin/env bash
# smoke_eval_multi_ckpt.sh — Multi-checkpoint smoke test: A1 vs A2 at 400/600/800 steps
# Runs lb6 eval on each checkpoint pair and produces comparison tables.
# Usage: bash scripts/isolated/longinst/smoke_eval_multi_ckpt.sh
set -euo pipefail

cd /root/autodl-tmp/dfrope/hybrid-rope

BASE_MODEL="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
LONGBENCH_DATA="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"
EVAL_SCRIPT="scripts/eval_longbench.py"
ARTIFACT_ROOT="artifacts/llama8k_theory_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUT="${ARTIFACT_ROOT}/smoke_multi_ckpt_${TIMESTAMP}"
mkdir -p "${EVAL_OUT}"

# A1 run dir (800-step geometric)
A1_DIR="${ARTIFACT_ROOT}/train/A1_geometric_tau0p00_r32_s800_seed42__20260228_115150"
A1_INV="${A1_DIR}/artifacts/custom_inv_freq.pt"

# A2 run dir (800-step EVQ) — auto-detect latest
A2_DIR=$(ls -dt ${ARTIFACT_ROOT}/train/A2_evq_cosh_tau1p50_r32_s800_seed42__* 2>/dev/null | head -1)
A2_INV="${A2_DIR}/artifacts/custom_inv_freq.pt"

if [ -z "${A2_DIR}" ]; then
    echo "[ERROR] No A2 800-step run dir found"
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "[SMOKE] Multi-checkpoint A1 vs A2 comparison"
echo "[SMOKE] A1: ${A1_DIR}"
echo "[SMOKE] A2: ${A2_DIR}"
echo "═══════════════════════════════════════════════════════"

# Common eval args
EVAL_COMMON="--base_model_path ${BASE_MODEL} \
  --variant custom \
  --task_set lb6 \
  --max_input_tokens 8192 \
  --batch_size 4 \
  --max_batch_input_tokens 32768 \
  --attn_implementation sdpa \
  --skip_base_unfinetuned \
  --score_scale pct \
  --longbench_local_data_dir ${LONGBENCH_DATA} \
  --save_per_sample_traces 0"

# Checkpoints to evaluate
CKPTS="400 600 800"

for STEP in ${CKPTS}; do
    A1_CKPT="${A1_DIR}/checkpoint-${STEP}"
    A2_CKPT="${A2_DIR}/checkpoint-${STEP}"

    # Skip if checkpoint doesn't exist
    if [ ! -d "${A1_CKPT}" ]; then
        echo "[SKIP] A1 checkpoint-${STEP} not found"
        continue
    fi
    if [ ! -d "${A2_CKPT}" ]; then
        echo "[SKIP] A2 checkpoint-${STEP} not found"
        continue
    fi

    echo ""
    echo "─────────────────────────────────────────────"
    echo "[EVAL] Checkpoint ${STEP}: A1 (geometric)"
    echo "─────────────────────────────────────────────"
    /root/miniconda3/bin/python ${EVAL_SCRIPT} \
      ${EVAL_COMMON} \
      --adapter_path "${A1_CKPT}" \
      --custom_inv_freq_path "${A1_INV}" \
      --model_alias "A1_geo_${STEP}" \
      --output_json "${EVAL_OUT}/a1_geo_${STEP}.json" \
      2>&1 | tee "${EVAL_OUT}/a1_${STEP}.log"

    echo ""
    echo "─────────────────────────────────────────────"
    echo "[EVAL] Checkpoint ${STEP}: A2 (EVQ-Cosh)"
    echo "─────────────────────────────────────────────"
    /root/miniconda3/bin/python ${EVAL_SCRIPT} \
      ${EVAL_COMMON} \
      --adapter_path "${A2_CKPT}" \
      --custom_inv_freq_path "${A2_INV}" \
      --model_alias "A2_evq_${STEP}" \
      --output_json "${EVAL_OUT}/a2_evq_${STEP}.json" \
      2>&1 | tee "${EVAL_OUT}/a2_${STEP}.log"
done

# ═══════════════════════════════════════════════════════
# Summary comparison across all checkpoints
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[SUMMARY] Multi-checkpoint comparison"
echo "═══════════════════════════════════════════════════════"

/root/miniconda3/bin/python - "${EVAL_OUT}" <<'PYEOF'
import json, sys, os
from pathlib import Path

eval_dir = Path(sys.argv[1])

def load_scores(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    scores = {}
    for model_key, model_data in data.items():
        if isinstance(model_data, dict) and "tasks" in model_data:
            for task_name, task_data in model_data["tasks"].items():
                if isinstance(task_data, dict):
                    scores[task_name] = task_data.get("score", 0.0)
            scores["__avg__"] = model_data.get("avg_score", 0.0)
            break
    if not scores:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                scores[k] = v
            elif isinstance(v, dict) and "score" in v:
                scores[k] = v["score"]
    return scores

checkpoints = [400, 600, 800]
results = {}

for step in checkpoints:
    a1_path = eval_dir / f"a1_geo_{step}.json"
    a2_path = eval_dir / f"a2_evq_{step}.json"
    a1 = load_scores(a1_path)
    a2 = load_scores(a2_path)
    if a1 or a2:
        results[step] = (a1, a2)

if not results:
    print("No results found!")
    sys.exit(0)

# Collect all tasks
all_tasks = set()
for step, (a1, a2) in results.items():
    all_tasks.update(k for k in list(a1.keys()) + list(a2.keys()) if k != "__avg__")
all_tasks = sorted(all_tasks)

# Print per-task table across checkpoints
steps_with_data = sorted(results.keys())
header = f"{'Task':<20}"
for s in steps_with_data:
    header += f" {'A1@'+str(s):>8} {'A2@'+str(s):>8} {'Δ':>7}"
print(header)
print("-" * len(header))

for t in all_tasks:
    row = f"{t:<20}"
    for s in steps_with_data:
        a1, a2 = results[s]
        s1, s2 = a1.get(t, 0.0), a2.get(t, 0.0)
        d = s2 - s1
        row += f" {s1:>8.2f} {s2:>8.2f} {d:>+7.2f}"
    print(row)

print("-" * len(header))
row = f"{'AVERAGE':<20}"
for s in steps_with_data:
    a1, a2 = results[s]
    tasks_here = [t for t in all_tasks if t in a1 and t in a2]
    if tasks_here:
        a1_avg = sum(a1[t] for t in tasks_here) / len(tasks_here)
        a2_avg = sum(a2[t] for t in tasks_here) / len(tasks_here)
        row += f" {a1_avg:>8.2f} {a2_avg:>8.2f} {a2_avg-a1_avg:>+7.2f}"
    else:
        row += f" {'N/A':>8} {'N/A':>8} {'N/A':>7}"
print(row)

# Verdict
print()
for s in steps_with_data:
    a1, a2 = results[s]
    tasks_here = [t for t in all_tasks if t in a1 and t in a2]
    if tasks_here:
        a1_avg = sum(a1[t] for t in tasks_here) / len(tasks_here)
        a2_avg = sum(a2[t] for t in tasks_here) / len(tasks_here)
        delta = a2_avg - a1_avg
        wins = sum(1 for t in tasks_here if a2[t] > a1[t] + 0.5)
        losses = sum(1 for t in tasks_here if a1[t] > a2[t] + 0.5)
        print(f"  @{s}: A2-A1={delta:+.2f}  A2 wins {wins}/{len(tasks_here)} tasks, A1 wins {losses}/{len(tasks_here)}")

print()
# Overall recommendation
all_deltas = []
for s in steps_with_data:
    a1, a2 = results[s]
    tasks_here = [t for t in all_tasks if t in a1 and t in a2]
    if tasks_here:
        all_deltas.append(sum(a2[t] - a1[t] for t in tasks_here) / len(tasks_here))
if all_deltas:
    avg_delta = sum(all_deltas) / len(all_deltas)
    if avg_delta > 1.0:
        print(f"VERDICT: EVQ-Cosh shows consistent improvement (avg delta={avg_delta:+.2f}). Proceed to B1/B2.")
    elif avg_delta < -1.0:
        print(f"VERDICT: Geometric baseline outperforms (avg delta={avg_delta:+.2f}). Investigate.")
    else:
        print(f"VERDICT: Roughly comparable (avg delta={avg_delta:+.2f}). Check per-task patterns.")

# Save summary JSON
summary = {"checkpoints": {}, "all_deltas": all_deltas}
for s in steps_with_data:
    a1, a2 = results[s]
    summary["checkpoints"][s] = {"a1": a1, "a2": a2}
(eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary saved to {eval_dir / 'summary.json'}")
PYEOF

echo ""
echo "[DONE] All results in ${EVAL_OUT}/"
