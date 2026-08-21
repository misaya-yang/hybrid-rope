#!/usr/bin/env bash
set -euo pipefail

action="dry-run"
if [[ "$#" -ge 1 ]]; then
  action="$1"
fi
if [[ "$action" != "dry-run" ]]; then
  echo "STOP: R4' preparation permits only the no-GPU dry-run; action=$action" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$repo_root"
exec python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.dry_run
