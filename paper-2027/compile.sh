#!/usr/bin/env bash
# Build the ICML 2027 submission PDF.
#   ./compile.sh            full build (pdflatex x2 + bibtex + pdflatex)
#   ./compile.sh quick      single pdflatex pass
#   ./compile.sh clean      remove build artefacts
set -euo pipefail
cd "$(dirname "$0")"
MAIN=main

clean() { rm -f "$MAIN".{aux,bbl,blg,log,out,brf,fls,fdb_latexmk,synctex.gz}; }

require() { command -v "$1" >/dev/null || { echo "missing command: $1" >&2; exit 1; }; }

for tool in pdflatex bibtex pdfinfo pdffonts kpsewhich; do require "$tool"; done
for dep in forloop.sty pcrr7t.tfm; do
  kpsewhich "$dep" >/dev/null || {
    echo "missing TeX dependency: $dep (install a full TeX Live/MacTeX distribution)" >&2
    exit 1
  }
done

case "${1:-full}" in
  clean) clean; echo "cleaned"; exit 0 ;;
  quick) pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null; ;;
  full)
    pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
    bibtex "$MAIN" >/dev/null
    pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
    pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
    ;;
  *) echo "usage: $0 [full|quick|clean]"; exit 1 ;;
esac

echo "=================== BUILD REPORT ==================="
FAIL=0

# --- 1. main-body page limit (ICML: 8 pages, refs/impact/appendix excluded) ---
BODYEND=$(grep -o 'newlabel{page:bodyend}{{[^}]*}{[0-9]*}' $MAIN.aux \
          | grep -o '{[0-9]*}$' | tr -d '{}' || echo "?")
echo -n "main body ends on page : $BODYEND   "
if [ "$BODYEND" != "?" ] && [ "$BODYEND" -le 8 ]; then
  echo "[OK, limit 8]"
else
  echo "[OVER LIMIT - CUT]"
  FAIL=1
fi

# --- 2. undefined references / citations ---
UNDEF=$( { grep -oE "(Reference|Citation) .[^']*. undefined" "$MAIN.log" || true; } | sort -u | wc -l | tr -d " ")
echo "undefined refs/cites   : $UNDEF"
{ grep -o "Reference .[^']*. undefined" "$MAIN.log" || true; } | sort -u | sed 's/^/    /'
{ grep -o "Citation .[^']*. undefined"  "$MAIN.log" || true; } | sort -u | sed 's/^/    /'
[ "$UNDEF" -eq 0 ] || FAIL=1

# --- 3. layout overflow ---
WORST=$( { grep "Overfull .hbox" "$MAIN.log" || true; } \
          | sed -nE 's/.*\(([0-9.]+)pt too wide\).*/\1/p' \
          | sort -rn | head -1 )
 WORST=${WORST:-0}
echo "worst overfull hbox    : ${WORST:-0}pt  (aim < 5pt)"
awk -v worst="$WORST" 'BEGIN { exit !(worst > 5) }' && FAIL=1 || true

# --- 4. anonymity ---
echo -n "anonymity scan         : "
if grep -rqiE "github\.com/[a-z0-9_-]+|acknowledg(e|ment)|/Users/|/root/|autodl|Author action" \
    main.tex sections/ appendix/ tables/ refs/ 2>/dev/null; then
  echo "[CHECK - possible deanonymising string]"
  FAIL=1
else
  echo "[OK]"
fi

# --- 5. total pages ---
PAGES=$(pdfinfo "$MAIN.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
PAGE_SIZE=$(pdfinfo "$MAIN.pdf" | awk -F': *' '/^Page size/{print $2}')
PDF_AUTHOR=$(pdfinfo "$MAIN.pdf" | awk -F': *' '/^Author/{print $2}')
TYPE3=$(pdffonts "$MAIN.pdf" | grep -c 'Type 3' || true)
NOT_EMBEDDED=$(pdffonts "$MAIN.pdf" | awk 'NR>2 && $5 != "yes" {n++} END {print n+0}')
echo "total pages            : $PAGES"
echo "page size              : $PAGE_SIZE"
echo "PDF bytes              : $(wc -c < "$MAIN.pdf" | tr -d ' ')  (limit 52428800)"
echo "PDF author             : $PDF_AUTHOR"
echo "Type 3 fonts           : $TYPE3"
echo "fonts not embedded     : $NOT_EMBEDDED"

case "$PAGE_SIZE" in
  "612 x 792 pts"*) ;;
  *) echo "[BLOCK] PDF is not US Letter" >&2; FAIL=1 ;;
esac
[ "$TYPE3" -eq 0 ] || FAIL=1
[ "$NOT_EMBEDDED" -eq 0 ] || FAIL=1
if [ -n "$PDF_AUTHOR" ] && ! printf '%s' "$PDF_AUTHOR" | grep -qi '^anonymous'; then
  echo "[BLOCK] PDF author metadata is not anonymous" >&2
  FAIL=1
fi

if grep -RniE "Author action|TODO|FIXME|TBD" main.tex sections/ appendix/ tables/ refs/; then
  echo "[BLOCK] unresolved manuscript marker found" >&2
  exit 1
fi

BYTES=$(wc -c < "$MAIN.pdf" | tr -d ' ')
[ "$BYTES" -le 52428800 ] || { echo "[BLOCK] PDF exceeds 50 MiB" >&2; exit 1; }
echo "===================================================="
[ "$FAIL" -eq 0 ] || { echo "[BLOCK] submission checks failed" >&2; exit 1; }
