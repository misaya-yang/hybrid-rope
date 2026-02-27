#!/bin/bash
# =============================================================
# EVQ τ-sweep 一键执行脚本 — 5090 32GB
# SSH: ssh -p 13275 root@connect.bjb2.seetacloud.com
# =============================================================
set -e

echo "============================================"
echo "  Step 0: 环境检查"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 检查必要依赖
python3 -c "from transformers import AutoTokenizer; print('transformers OK')" 2>/dev/null || pip install transformers -q
python3 -c "import datasets; print('datasets OK')" 2>/dev/null || pip install datasets -q
python3 -c "import matplotlib; print('matplotlib OK')" 2>/dev/null || pip install matplotlib -q

echo ""
echo "============================================"
echo "  Step 1: 上传实验脚本"
echo "============================================"

# 创建工作目录
mkdir -p /root/evq_sweep/scripts
WORK_DIR="/root/evq_sweep"
RESULTS_DIR="${WORK_DIR}/results"
mkdir -p ${RESULTS_DIR}

# 如果脚本不存在，提示用户上传
if [ ! -f "${WORK_DIR}/scripts/run_evq_sweep.py" ]; then
    echo ""
    echo "请先把以下两个文件上传到服务器:"
    echo "  scripts/m4_evq_sweep/run_evq_sweep.py  →  ${WORK_DIR}/scripts/run_evq_sweep.py"
    echo "  scripts/m4_evq_sweep/evq_analysis.py    →  ${WORK_DIR}/scripts/evq_analysis.py"
    echo ""
    echo "可以用 scp 上传:"
    echo "  scp -P 13275 scripts/m4_evq_sweep/run_evq_sweep.py root@connect.bjb2.seetacloud.com:${WORK_DIR}/scripts/"
    echo "  scp -P 13275 scripts/m4_evq_sweep/evq_analysis.py root@connect.bjb2.seetacloud.com:${WORK_DIR}/scripts/"
    echo ""
    echo "上传完成后重新运行本脚本。"
    exit 1
fi

echo "脚本已就位: ${WORK_DIR}/scripts/"

echo ""
echo "============================================"
echo "  Step 2: 运行 50M τ-sweep (预计 ~40 分钟)"
echo "============================================"

cd ${WORK_DIR}
python3 scripts/run_evq_sweep.py \
    --tier 50m \
    --taus 0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0 \
    --seeds 42 \
    --base 500000.0 \
    --work_dir ${RESULTS_DIR}/50m \
    --resume

echo ""
echo "============================================"
echo "  Step 3: 运行 125M τ-sweep (预计 ~100 分钟)"
echo "============================================"

python3 scripts/run_evq_sweep.py \
    --tier 125m \
    --taus 0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0 \
    --seeds 42 \
    --base 500000.0 \
    --work_dir ${RESULTS_DIR}/125m \
    --resume

echo ""
echo "============================================"
echo "  Step 4: 生成分析图表"
echo "============================================"

python3 scripts/evq_analysis.py --input ${RESULTS_DIR}/50m/results_final.json --out_dir ${RESULTS_DIR}/50m/figures
python3 scripts/evq_analysis.py --input ${RESULTS_DIR}/125m/results_final.json --out_dir ${RESULTS_DIR}/125m/figures

echo ""
echo "============================================"
echo "  DONE! 结果位置:"
echo "    50M:  ${RESULTS_DIR}/50m/results_final.json"
echo "    125M: ${RESULTS_DIR}/125m/results_final.json"
echo "    图表: ${RESULTS_DIR}/*/figures/"
echo "============================================"
