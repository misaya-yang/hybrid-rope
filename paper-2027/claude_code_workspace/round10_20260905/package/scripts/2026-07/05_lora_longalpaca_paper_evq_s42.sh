#!/usr/bin/env bash
# Thin EVQ-Cosh seed-42 wrapper around the exact paper-lineage LongAlpaca driver.
# The shared driver owns every training argument; this wrapper changes only the
# frequency method and deliberately cannot launch evaluation.
set -Eeuo pipefail

case "${1:-}" in
  preflight|train) ;;
  *)
    echo "usage: $0 {preflight|train}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVQ_PAPER_ROPE_METHOD=evq_cosh
exec "$SCRIPT_DIR/04_lora_longalpaca_paper_geo_s42.sh" "$@"
