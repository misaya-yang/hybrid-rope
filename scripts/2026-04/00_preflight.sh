#!/bin/bash
# 上机后第一个跑的脚本：检查环境，不做任何训练
set -e
echo "=== Preflight Check ==="

# GPU
echo "[1/5] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "NO GPU"

# Python
echo "[2/5] Python:"
/root/miniconda3/bin/python --version 2>/dev/null || echo "NO PYTHON"

# 模型
echo "[3/5] Model:"
ls /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct/config.json 2>/dev/null && echo "OK" || echo "MISSING"

# 数据
echo "[4/5] Data:"
ls /root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl 2>/dev/null && echo "LongAlign OK" || echo "LongAlign MISSING"
ls /root/autodl-tmp/data/wikitext2/wikitext2_test.txt 2>/dev/null && echo "WikiText OK" || echo "WikiText MISSING (optional)"

# 代码
echo "[5/5] Code sync:"
cd /root/autodl-tmp/hybrid-rope 2>/dev/null && git log --oneline -1 || echo "REPO MISSING - run: git clone"

# 已有checkpoint
echo ""
echo "=== Existing LoRA Checkpoints ==="
CKPT="/root/autodl-tmp/hybrid-rope/experiments/lora_evq_v2/checkpoints"
for d in "${CKPT}"/*/; do
    [ -d "$d" ] && echo "  $(basename $d): $(ls $d/adapter_model.* 2>/dev/null | head -1 || echo 'INCOMPLETE')"
done

echo ""
echo "=== Done. 如果所有检查通过，开始训练。 ==="
