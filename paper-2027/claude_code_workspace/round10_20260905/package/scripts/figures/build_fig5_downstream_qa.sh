#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ROOT}/paper/build_figures"
SOURCE="${SCRIPT_DIR}/fig5_downstream_qa_nll.tex"
PDF_OUT="${ROOT}/paper/figs/fig5_downstream_qa.pdf"
PNG_STEM="${ROOT}/paper/figs/fig5_downstream_qa"

TECTONIC="${TECTONIC:-$(command -v tectonic || true)}"
PDFTOPPM="${PDFTOPPM:-$(command -v pdftoppm || true)}"
if [ -z "${TECTONIC}" ]; then
    echo "tectonic is required" >&2
    exit 1
fi
if [ -z "${PDFTOPPM}" ]; then
    echo "pdftoppm is required" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
"${TECTONIC}" -X compile "${SOURCE}" --outdir "${BUILD_DIR}"
cp "${BUILD_DIR}/fig5_downstream_qa_nll.pdf" "${PDF_OUT}"
"${PDFTOPPM}" -png -singlefile -r 300 "${PDF_OUT}" "${PNG_STEM}"

echo "Generated ${PDF_OUT} and ${PNG_STEM}.png"
