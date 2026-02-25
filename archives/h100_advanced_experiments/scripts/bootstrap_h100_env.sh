#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-dftorch}"
OUT_DIR="${OUT_DIR:-/opt/dfrope/results/env_checks}"

mkdir -p "$OUT_DIR"

echo "[info] conda: $CONDA_BIN"
echo "[info] env:   $ENV_NAME"
echo "[info] out:   $OUT_DIR"

nvidia-smi | tee "$OUT_DIR/nvidia_smi.txt"

"$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY' | tee "$OUT_DIR/python_stack.txt"
import platform
import torch
import importlib

def ver(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return "NOT_INSTALLED"

print(f"python: {platform.python_version()}")
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    prop = torch.cuda.get_device_properties(0)
    print(f"gpu_mem_gb: {prop.total_memory / 1e9:.1f}")
    print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
for pkg in ["transformers", "datasets", "accelerate", "numpy", "matplotlib"]:
    print(f"{pkg}: {ver(pkg)}")
PY

echo "[done] Environment snapshot saved under $OUT_DIR"
