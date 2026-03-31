#!/bin/bash
# =============================================================
# Server Setup for LoRA EVQ-Cosh v2
# =============================================================
# Run on GPU server (无卡模式 OK) to prepare everything.
#
# Usage:
#   bash server_setup.sh              # full setup: download + dryrun
#   bash server_setup.sh download     # download model + data only
#   bash server_setup.sh dryrun       # dryrun validation only
#   bash server_setup.sh verify       # verify everything ready
# =============================================================

set -e

# Fix PATH for miniconda
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "LoRA EVQ-Cosh v2 — Server Setup"
echo "Time: $(date)"
echo "Python: $(python --version 2>&1)"
echo "============================================"

# Step 0: Verify all imports
verify_deps() {
    echo ""
    echo "[DEPS] Checking imports ..."
    python -c "
import torch, transformers, datasets, peft, bitsandbytes, numpy, accelerate, modelscope
print(f'  torch={torch.__version__}, transformers={transformers.__version__}')
print(f'  peft={peft.__version__}, bnb={bitsandbytes.__version__}')
print(f'  datasets={datasets.__version__}, modelscope={modelscope.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('  ✅ All imports OK')
"
}

# Step 1: Download model + data
do_download() {
    echo ""
    echo "[DOWNLOAD] Starting model + data download ..."
    python "${SCRIPT_DIR}/download_model_data.py" 2>&1 | tee "${LOG_DIR}/download.log"
}

# Step 2: Dryrun validation
do_dryrun() {
    echo ""
    echo "[DRYRUN] Running validation ..."

    # Read paths from download output
    LOCAL_MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
    LOCAL_DATA="/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl"

    python "${SCRIPT_DIR}/dryrun_validate.py" \
        --tau 1.414 \
        --lora_r 64 \
        --lora_alpha 128 \
        --max_seq_len 8192 \
        --local_data_path "${LOCAL_DATA}" \
        2>&1 | tee "${LOG_DIR}/dryrun.log"
}

# Step 3: Verify
do_verify() {
    echo ""
    echo "[VERIFY] Checking all components ..."
    python "${SCRIPT_DIR}/download_model_data.py" --verify_only
}

# Main
case "${1:-all}" in
    download)   verify_deps; do_download ;;
    dryrun)     verify_deps; do_dryrun ;;
    verify)     verify_deps; do_verify ;;
    all)
        verify_deps
        do_download
        do_dryrun
        do_verify
        echo ""
        echo "============================================"
        echo "✅ Setup complete! Ready for GPU mode."
        echo "   Next: switch to 有卡模式, then:"
        echo "   bash ${SCRIPT_DIR}/run.sh"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash server_setup.sh [download|dryrun|verify|all]"
        exit 1
        ;;
esac
