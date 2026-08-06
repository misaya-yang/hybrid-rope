#!/usr/bin/env bash
# Reproduce the full layer-resolved causal readout decomposition.
#
# CPU only.  Loads no model, uses no GPU, downloads nothing, and writes only
# inside this directory.  Reads ~540 MB of frozen tensors; runs in ~1 minute.
#
# Requires a python with torch + numpy + scipy + matplotlib.  The system python3
# on this machine has none of them; the conda env below does (torch 2.9.1).
# Override with:  PYTHON=/path/to/python ./run_all.sh
set -euo pipefail

PYTHON="${PYTHON:-/Users/misaya.yanghejazfs.com.au/miniconda3/envs/ai_gateway/bin/python}"
cd "$(dirname "$0")"

echo "== 1/4  control gate (hard stop if it fails) =="
"$PYTHON" controls.py

echo
echo "== 2/4  decomposition =="
"$PYTHON" decompose.py

echo
echo "== 3/4  independent validation against the RTX 5090 rank_16k_v1 run =="
"$PYTHON" validate.py

echo
echo "== 4/4  figures =="
"$PYTHON" figures.py

echo
echo "done. outputs:"
ls -1 ./*.csv ./*.json figures/*.png
