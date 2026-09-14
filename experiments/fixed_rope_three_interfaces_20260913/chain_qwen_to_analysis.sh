#!/usr/bin/env bash
set -euo pipefail

qwen_root=/root/autodl-tmp/today_rope_plan_20260914/qwen25_s2_full324
qwen_pid=$(<"${qwen_root}/logs/supervisor.pid")

while ps -p "${qwen_pid}" -o args= | grep -q 'run_qwen25_s2_full324.sh'; do
  sleep 5
done

grep -q '^QUEUE_COMPLETE ' "${qwen_root}/logs/supervisor_b2.log"
exec /root/autodl-tmp/hybrid-rope/experiments/fixed_rope_three_interfaces_20260913/analyze_qwen25_s2_full324.sh
