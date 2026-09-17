#!/usr/bin/env bash
set -u

wait_pid=${WAIT_PID:-}
if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
fi

cd /root/autodl-tmp/hybrid-rope || exit 1
failed=0
for model in llama qwen3; do
  bash experiments/native_tailspline_s2_midgain_20260917/run_server_model_qa.sh "$model" --execute || failed=1
done
if (( failed )); then
  echo NTS2_QA_QUEUE_PARTIAL_FAILURE >&2
  exit 1
fi
echo NTS2_QA_QUEUE_COMPLETE
