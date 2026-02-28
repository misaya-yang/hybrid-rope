#!/usr/bin/env bash
# launch_a1_resume_a2_fresh_800.sh
# A1: resume from checkpoint-600 → 800 steps (~9 min)
# A2: fresh 800 steps with proper cosine schedule (~35 min)
# Then: lb6 smoke test comparison
set -euo pipefail

cd /root/autodl-tmp/dfrope/hybrid-rope

BASE_MODEL="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
LONGALPACA="/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"
WIKITEXT="/root/autodl-tmp/wikitext_data/train.txt"
LONGBENCH_DATA="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data"
QWEN42="/root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json"
QWEN1337="/root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json"
TRAIN_SCRIPT="scripts/isolated/longinst/new_lora_longinst_train_v1.py"
EVAL_SCRIPT="scripts/eval_longbench.py"
ARTIFACT_ROOT="artifacts/llama8k_theory_v1"

# Common training args
COMMON_ARGS="--base_model_path ${BASE_MODEL} \
  --output_root ${ARTIFACT_ROOT} \
  --max_steps 800 \
  --lora_rank 32 \
  --mix_long_ratio 0.7 \
  --mix_wiki_ratio 0.1 \
  --synthetic_ratio 0.2 \
  --min_supervised_tokens 48 \
  --max_seq_len 8192 \
  --attn_implementation sdpa \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_steps 50 \
  --save_steps 200 \
  --longalpaca_path ${LONGALPACA} \
  --longqa_path '' \
  --wikitext_train_path ${WIKITEXT} \
  --mixed_dataset_split train \
  --longbench_local_data_dir ${LONGBENCH_DATA} \
  --qwen_seed42_json ${QWEN42} \
  --qwen_seed1337_json ${QWEN1337} \
  --morning_reference_json ${QWEN42} \
  --eval_batch_size 8 \
  --max_batch_input_tokens 98304 \
  --max_input_tokens_eval 8192 \
  --require_offset_boundary"

A1_RUN_DIR="${ARTIFACT_ROOT}/train/A1_geometric_tau0p00_r32_s800_seed42__20260228_115150"
A1_CKPT600="${A1_RUN_DIR}/checkpoint-600"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ═══════════════════════════════════════════════════════
# PHASE 1: A1 resume from checkpoint-600 → 800
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[A1] Resuming geometric (tau=0) from checkpoint-600 → 800 steps"
echo "═══════════════════════════════════════════════════════"

if [ ! -d "${A1_CKPT600}" ]; then
    echo "[ERROR] A1 checkpoint-600 not found at ${A1_CKPT600}"
    echo "[ERROR] Will run A1 from scratch instead."
    A1_RESUME_FLAG=""
else
    echo "[A1] Found checkpoint-600, resuming..."
    A1_RESUME_FLAG="--resume_from_checkpoint ${A1_CKPT600}"
fi

/root/miniconda3/bin/python ${TRAIN_SCRIPT} \
  ${COMMON_ARGS} \
  --run_name A1_geometric_tau0p00_r32_s800_seed42__20260228_115150 \
  --seed 42 \
  --rope_schedule evq_cosh \
  --evq_tau 0.0 \
  --evq_beta 3.0 \
  ${A1_RESUME_FLAG} \
  --allow_existing_run_dir \
  --run_full_eval \
  2>&1 | tee "${A1_RUN_DIR}/resume_log_${TIMESTAMP}.txt"

echo "[A1] Training complete (800 steps)."
echo ""

# ═══════════════════════════════════════════════════════
# PHASE 2: A2 fresh 800 steps (EVQ-Cosh tau=1.5)
# ═══════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════"
echo "[A2] Fresh training EVQ-Cosh (tau=1.5) 800 steps"
echo "═══════════════════════════════════════════════════════"

A2_RUN_NAME="A2_evq_cosh_tau1p50_r32_s800_seed42__${TIMESTAMP}"

/root/miniconda3/bin/python ${TRAIN_SCRIPT} \
  ${COMMON_ARGS} \
  --run_name ${A2_RUN_NAME} \
  --seed 42 \
  --rope_schedule evq_cosh \
  --evq_tau 1.5 \
  --evq_beta 3.0 \
  --run_full_eval \
  2>&1 | tee "${ARTIFACT_ROOT}/train/${A2_RUN_NAME}/train_stdout.txt"

echo "[A2] Training complete (800 steps)."
echo ""

# ═══════════════════════════════════════════════════════
# PHASE 3: LB6 smoke test — A1 vs A2
# ═══════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════"
echo "[EVAL] LB6 smoke test: A1 (geometric) vs A2 (EVQ-Cosh)"
echo "═══════════════════════════════════════════════════════"

# Locate best checkpoints
A1_ADAPTER="${A1_RUN_DIR}/checkpoint-800"
A1_INV="${A1_RUN_DIR}/artifacts/custom_inv_freq.pt"

A2_DIR=$(ls -dt ${ARTIFACT_ROOT}/train/A2_evq_cosh_tau1p50_r32_s800_seed42__* 2>/dev/null | head -1)
A2_ADAPTER="${A2_DIR}/checkpoint-800"
A2_INV="${A2_DIR}/artifacts/custom_inv_freq.pt"

# Fallback: use latest checkpoint if 800 doesn't exist
if [ ! -d "${A1_ADAPTER}" ]; then
    A1_ADAPTER=$(ls -dt ${A1_RUN_DIR}/checkpoint-* 2>/dev/null | head -1)
    echo "[WARN] A1 checkpoint-800 not found, using $(basename ${A1_ADAPTER})"
fi
if [ ! -d "${A2_ADAPTER}" ]; then
    A2_ADAPTER=$(ls -dt ${A2_DIR}/checkpoint-* 2>/dev/null | head -1)
    echo "[WARN] A2 checkpoint-800 not found, using $(basename ${A2_ADAPTER})"
fi

EVAL_OUT="${ARTIFACT_ROOT}/smoke_eval_a1_vs_a2_${TIMESTAMP}"
mkdir -p "${EVAL_OUT}"

echo "[EVAL] A1 adapter: ${A1_ADAPTER}"
echo "[EVAL] A2 adapter: ${A2_ADAPTER}"

# Verify files
for f in "${A1_ADAPTER}/adapter_model.safetensors" "${A1_INV}" \
         "${A2_ADAPTER}/adapter_model.safetensors" "${A2_INV}"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f"
        exit 1
    fi
done

# A1 eval
echo "[EVAL] Running A1 (geometric)..."
/root/miniconda3/bin/python ${EVAL_SCRIPT} \
  --base_model_path ${BASE_MODEL} \
  --adapter_path "${A1_ADAPTER}" \
  --custom_inv_freq_path "${A1_INV}" \
  --model_alias "A1_geometric_800" \
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

# A2 eval
echo "[EVAL] Running A2 (EVQ-Cosh)..."
/root/miniconda3/bin/python ${EVAL_SCRIPT} \
  --base_model_path ${BASE_MODEL} \
  --adapter_path "${A2_ADAPTER}" \
  --custom_inv_freq_path "${A2_INV}" \
  --model_alias "A2_evq_cosh_800" \
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

# ═══════════════════════════════════════════════════════
# PHASE 4: Comparison table
# ═══════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "[COMPARE] A1@800 (geometric) vs A2@800 (EVQ-Cosh)"
echo "═══════════════════════════════════════════════════════"

/root/miniconda3/bin/python - "${EVAL_OUT}/a1_geometric_lb6.json" "${EVAL_OUT}/a2_evq_cosh_lb6.json" <<'PYEOF'
import json, sys
from pathlib import Path

a1_path, a2_path = sys.argv[1], sys.argv[2]

def load_scores(path):
    data = json.loads(Path(path).read_text())
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

a1 = load_scores(a1_path)
a2 = load_scores(a2_path)
all_tasks = sorted(set(list(a1.keys()) + list(a2.keys())) - {"__avg__"})

print(f"\n{'Task':<20} {'A1(geo)':>10} {'A2(EVQ)':>10} {'Delta':>10} {'Win':>6}")
print("-" * 60)
a1_t, a2_t, cnt = 0.0, 0.0, 0
for t in all_tasks:
    s1, s2 = a1.get(t, 0.0), a2.get(t, 0.0)
    d = s2 - s1
    w = "A2" if d > 0.5 else ("A1" if d < -0.5 else "~")
    print(f"{t:<20} {s1:>10.2f} {s2:>10.2f} {d:>+10.2f} {w:>6}")
    a1_t += s1; a2_t += s2; cnt += 1

if cnt > 0:
    a1a, a2a = a1_t/cnt, a2_t/cnt
    da = a2a - a1a
    wa = "A2" if da > 0.5 else ("A1" if da < -0.5 else "~")
    print("-" * 60)
    print(f"{'AVERAGE':<20} {a1a:>10.2f} {a2a:>10.2f} {da:>+10.2f} {wa:>6}")
    print()
    if da > 1.0:
        print("VERDICT: A2 (EVQ-Cosh) shows improvement. Proceed with B1/B2.")
    elif da < -1.0:
        print("VERDICT: A1 (geometric) better. Investigate EVQ config before continuing.")
    else:
        print("VERDICT: Roughly comparable. Check per-task patterns.")
PYEOF

echo ""
echo "[DONE] All results in ${EVAL_OUT}/"
echo "[DONE] Total pipeline complete."
