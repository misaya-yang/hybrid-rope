#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
manifest=${root}/qa64k/assets/manifest.json

for _ in $(seq 1 180); do
  [[ -s ${manifest} ]] && break
  if ! pgrep -f 'prepare_natural_long.*kanana_1p5_8b' >/dev/null; then
    printf 'REFUSE: QA asset preparation exited without a manifest\n' >&2
    exit 1
  fi
  sleep 10
done
[[ -s ${manifest} ]] || { printf 'REFUSE: QA asset preparation timed out\n' >&2; exit 1; }

cd "${repo}"
bash experiments/kanana_yarn_tailspline_64k_20260918/prepare_qa64k.sh
bash experiments/kanana_yarn_tailspline_64k_20260918/run_qa64k_three_arm.sh --execute
