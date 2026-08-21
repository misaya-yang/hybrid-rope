#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  echo "usage: $0 dry-run --r0-json PATH --output PATH [options]" >&2
  echo "       $0 show-matrix --r0-json PATH" >&2
  echo "       $0 preflight|probe|train|evaluate [train_custom.py options]" >&2
}

if [[ "${MODE}" == "preflight" || "${MODE}" == "probe" || "${MODE}" == "train" || "${MODE}" == "evaluate" ]]; then
  shift
  cd "${REPO_ROOT}"
  exec "${PYTHON_BIN}" -m rebuttal.rebuttal_0723.experiments.demand_companding_5090.train_custom "${MODE}" "$@"
fi

if [[ "${MODE}" != "dry-run" && "${MODE}" != "show-matrix" ]]; then
  usage
  exit 64
fi
shift
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m rebuttal.rebuttal_0723.experiments.demand_companding_5090.run_experiment "${MODE}" "$@"
