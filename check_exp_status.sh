#!/bin/bash
echo "=== 实验状态检查 ==="
echo "时间: $(date)"
echo ""

# 检查进程
if ps -p $(cat exp_pid.txt 2>/dev/null) > /dev/null 2>&1; then
    echo "✅ 实验进程运行中 (PID: $(cat exp_pid.txt))"
    echo ""
    echo "最近进度:"
    tail -20 full_experiment.log 2>/dev/null | grep -E "(Variant|Phase|Loss|PPL|Sparsity|===)" | tail -10
else
    echo "✅ 实验已完成或停止"
    echo ""
    echo "最终输出:"
    tail -50 full_experiment.log 2>/dev/null | tail -50
fi

echo ""
echo "=== 输出文件 ==="
ls -lt outputs/var_attn/*/gpt2/ 2>/dev/null | head -15 || echo "等待输出目录生成..."
