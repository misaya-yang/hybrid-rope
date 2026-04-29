#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build_aidemo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

run_pdflatex() {
  TEXINPUTS="${BUILD_DIR}:.:${TEXINPUTS:-}" \
    conda run --no-capture-output -n aidemo \
      pdflatex -interaction=nonstopmode -halt-on-error \
      -output-directory="${BUILD_DIR}" main.tex
}

run_pdflatex > "${BUILD_DIR}/pdflatex1.console.log" 2>&1
BIBINPUTS=".:refs:${BIBINPUTS:-}" \
  conda run --no-capture-output -n aidemo \
    bibtex "${BUILD_DIR}/main" > "${BUILD_DIR}/bibtex.console.log" 2>&1
run_pdflatex > "${BUILD_DIR}/pdflatex2.console.log" 2>&1
run_pdflatex > "${BUILD_DIR}/pdflatex3.console.log" 2>&1
run_pdflatex > "${BUILD_DIR}/pdflatex4.console.log" 2>&1

if command -v pdffonts >/dev/null 2>&1 && pdffonts "${BUILD_DIR}/main.pdf" | grep -q ' Type 3 '; then
  echo "ERROR: Type 3 fonts detected in ${BUILD_DIR}/main.pdf" >&2
  exit 1
fi

if [ "${COPY_MAIN:-0}" = "1" ]; then
  cp "${BUILD_DIR}/main.pdf" main.pdf
fi

echo "Wrote ${BUILD_DIR}/main.pdf"
