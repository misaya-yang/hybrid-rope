#!/bin/bash
# 生成Stage2检索训练数据（S-NIAH + MK-NIAH + KV-Retr）
# 不需要GPU，几秒完成
set -e
export PATH=/root/miniconda3/bin:$PATH

LORA="/root/autodl-tmp/lora_evq_v2"
OUTPUT="${LORA}/retrieval_mix.jsonl"

if [ -f "${OUTPUT}" ]; then
    echo "retrieval_mix.jsonl exists ($(wc -l < ${OUTPUT}) samples)"
    echo "删除后重新生成: rm ${OUTPUT}"
    exit 0
fi

echo ">>> Generating retrieval mix data..."
/root/miniconda3/bin/python -u "${LORA}/gen_retrieval_mix.py" \
    --output "${OUTPUT}" \
    --n_samples 500
echo ">>> Done: $(wc -l < ${OUTPUT}) samples -> ${OUTPUT}"
