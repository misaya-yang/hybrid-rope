#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Local runtime artifacts should not live in repo root.
# Default locations can be overridden via env vars for compatibility.
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/variational_sparse_attention}"
PID_FILE="${PID_FILE:-${LOG_DIR}/exp_pid.txt}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/full_experiment.log}"

OUTPUT_GLOB="${OUTPUT_GLOB:-${REPO_ROOT}/outputs/var_attn/*/gpt2/}"

echo "=== 实验状态检查 ==="
echo "时间: $(date)"
echo "repo_root: ${REPO_ROOT}"
echo "log_dir: ${LOG_DIR}"
echo ""

pid=""
if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
fi

if [[ -n "${pid}" ]] && ps -p "${pid}" > /dev/null 2>&1; then
    echo "✅ 实验进程运行中 (PID: ${pid})"
    echo ""
    echo "最近进度:"
    if [[ -f "${LOG_FILE}" ]]; then
        tail -20 "${LOG_FILE}" | grep -E "(Variant|Phase|Loss|PPL|Sparsity|===)" | tail -10 || true
    else
        echo "(log missing) ${LOG_FILE}"
    fi
else
    echo "✅ 实验已完成或停止"
    echo ""
    echo "最终输出:"
    if [[ -f "${LOG_FILE}" ]]; then
        tail -50 "${LOG_FILE}" || true
    else
        echo "(log missing) ${LOG_FILE}"
    fi
fi

echo ""
echo "=== 输出文件 ==="
ls -lt ${OUTPUT_GLOB} 2>/dev/null | head -15 || echo "等待输出目录生成..."
