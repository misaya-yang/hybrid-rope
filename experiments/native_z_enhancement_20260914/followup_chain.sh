#!/usr/bin/env bash
set -euo pipefail

plan=/root/autodl-tmp/today_rope_plan_20260914
while [[ ! -f "${plan}/stable_accept_ready_queue_complete.txt" ]]; do
  sleep 15
done
cd /root/autodl-tmp/hybrid-rope
experiments/native_z_enhancement_20260914/run.sh
