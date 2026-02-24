#!/usr/bin/env bash
set -euo pipefail

# Parallel campaign runner for plan-v2 experiments.
# Designed for single-GPU systems where MIG may be unavailable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-base}"
MODEL="${MODEL:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
SEED="${SEED:-1337}"
MAIN_CTX="${MAIN_CTX:-}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LAUNCH_GAP_SEC="${LAUNCH_GAP_SEC:-240}"
PPL_MAX_CHUNKS="${PPL_MAX_CHUNKS:-20}"
RUN_NOTES="${RUN_NOTES:-campaign_parallel_v1}"

if [[ -z "${MAIN_CTX}" ]]; then
  if [[ -f "artifacts/results/main_ctx.txt" ]]; then
    MAIN_CTX="$(tr -d '[:space:]' < artifacts/results/main_ctx.txt)"
  else
    MAIN_CTX="32768"
  fi
fi

if [[ "${MAIN_CTX}" != "32768" && "${MAIN_CTX}" != "65536" ]]; then
  echo "Invalid MAIN_CTX=${MAIN_CTX}. Expected 32768 or 65536." >&2
  exit 1
fi

run_py() {
  "${CONDA_EXE}" run -n "${CONDA_ENV}" python "$@"
}

DATE_TAG="$(date +%Y-%m-%d)"
MODEL_TAG="meta_llama_3_8b_instruct"
LOG_DIR="artifacts/logs/campaign_${DATE_TAG}_ctx${MAIN_CTX}_seed${SEED}"
mkdir -p "${LOG_DIR}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL=${MODEL}"
echo "MAIN_CTX=${MAIN_CTX}"
echo "SEED=${SEED}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "LAUNCH_GAP_SEC=${LAUNCH_GAP_SEC}"
echo "PPL_MAX_CHUNKS=${PPL_MAX_CHUNKS}"
echo "LOG_DIR=${LOG_DIR}"

echo "[1/5] Cleanup old campaign rows (E1/E2/TEST) from registry..."
run_py - <<'PY'
import json
from pathlib import Path

registry = Path("artifacts/registry.jsonl")
if not registry.exists():
    print("registry missing, skip")
    raise SystemExit(0)

drop_exps = {"E1", "E2", "TEST"}
kept = []
dropped = 0
for line in registry.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        kept.append(line)
        continue
    if str(row.get("exp", "")).upper() in drop_exps:
        dropped += 1
        continue
    kept.append(json.dumps(row, ensure_ascii=False))

registry.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
print(f"dropped_rows={dropped}")
PY

echo "[2/5] Cleanup old run directories for E1/E2/TEST..."
rm -rf "runs/${DATE_TAG}_e1_${MODEL_TAG}_"*"_${SEED}" || true
rm -rf "runs/${DATE_TAG}_e2_${MODEL_TAG}_"*"_${SEED}" || true
rm -rf "runs/${DATE_TAG}_test_${MODEL_TAG}_"*"_${SEED}" || true

declare -a JOBS=(
  "E2 yarn ppl,longbench_full"
  "E2 hybrid ppl,longbench_full"
  "E1 baseline_native ppl,longbench_full,needle"
  "E1 pi ppl,longbench_full,needle"
  "E1 yarn ppl,longbench_full,needle"
  "E1 hybrid ppl,longbench_full,needle"
)

echo "[3/5] Launch run_eval jobs with parallelism=${MAX_PARALLEL}..."
declare -a PIDS=()
declare -a NAMES=()

for spec in "${JOBS[@]}"; do
  read -r exp method suite <<<"${spec}"
  name="${exp}_${method}_ctx${MAIN_CTX}_seed${SEED}"
  log="${LOG_DIR}/${name}.log"

  while [[ "$(jobs -pr | wc -l | tr -d '[:space:]')" -ge "${MAX_PARALLEL}" ]]; do
    wait -n || true
  done

  (
    echo "[START] ${name} suite=${suite} time=$(date '+%F %T')"
    run_py scripts/run_eval.py \
      --exp "${exp}" \
      --model "${MODEL}" \
      --method "${method}" \
      --ctx "${MAIN_CTX}" \
      --seed "${SEED}" \
      --suite "${suite}" \
      --ppl_max_chunks "${PPL_MAX_CHUNKS}" \
      --notes "${RUN_NOTES}"
    echo "[DONE] ${name} time=$(date '+%F %T')"
  ) > "${log}" 2>&1 &

  pid=$!
  PIDS+=("${pid}")
  NAMES+=("${name}")
  echo "launched pid=${pid} name=${name} log=${log}"

  # Stagger job starts so peak-memory PPL stages do not overlap.
  sleep "${LAUNCH_GAP_SEC}"
done

echo "Waiting for all run_eval jobs..."
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  name="${NAMES[$idx]}"
  if wait "${pid}"; then
    echo "[OK] ${name}"
  else
    echo "[FAILED] ${name}"
  fi
done

echo "[4/5] Run E3-lite..."
run_py scripts/run_attn_hist.py \
  --exp E3 \
  --model "${MODEL}" \
  --ctx 8192 \
  --seed "${SEED}" \
  --N 32 \
  --layers "2,16,30" \
  --heads "0,1,2,3,4,5,6,7" \
  --bins 256 \
  > "${LOG_DIR}/E3.log" 2>&1

echo "[5/5] Summarize tables..."
run_py scripts/summarize.py \
  --registry artifacts/registry.jsonl \
  --out artifacts/tables \
  > "${LOG_DIR}/summarize.log" 2>&1

echo "Campaign finished. Logs: ${LOG_DIR}"
