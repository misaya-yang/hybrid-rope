#!/usr/bin/env bash
# Build the ICLR 2027 submission PDF and run the hard format gates.
#   ./compile.sh            full build + compliance report
#   ./compile.sh clean      remove build artefacts
#
# Engine: pdflatex if present, else tectonic.
# NOTE on tectonic: it runs XeTeX, so line breaking can differ by a line or two
# from a pdflatex build.  Treat a body ending exactly on page 9 as tight, not
# safe; re-check on a full TeX Live install before submitting.
set -euo pipefail
cd "$(dirname "$0")"
MAIN=main
LIMIT=9   # ICLR 2027: 9 pages of main text at submission (10 at rebuttal/CR)

clean() { rm -f "$MAIN".{aux,bbl,blg,log,out,brf,fls,fdb_latexmk,synctex.gz}; }
case "${1:-full}" in clean) clean; echo "cleaned"; exit 0 ;; esac

if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
  ENGINE=pdflatex
  pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
  bibtex "$MAIN" >/dev/null
  pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
  pdflatex -halt-on-error -interaction=nonstopmode "$MAIN.tex" >/dev/null
elif command -v tectonic >/dev/null; then
  ENGINE=tectonic
  ok=0
  for _ in 1 2 3 4; do
    if tectonic "$MAIN.tex" --keep-intermediates --keep-logs >/tmp/tectonic_build.log 2>&1; then ok=1; break; fi
    sleep 2   # tectonic fetches packages on demand; retry transient network errors
  done
  [ "$ok" -eq 1 ] || { echo "tectonic build failed:" >&2; tail -20 /tmp/tectonic_build.log >&2; exit 1; }
else
  echo "no LaTeX engine found (need pdflatex+bibtex, or tectonic)" >&2; exit 1
fi

echo "=================== BUILD REPORT (engine: $ENGINE) ==================="
FAIL=0

# --- 1. main-body page limit (main text + statements must end on <= 9)
BODYEND=$(grep -o 'newlabel{page:bodyend}{{[^}]*}{[0-9]*}' $MAIN.aux \
          | grep -o '{[0-9]*}$' | tr -d '{}' || echo "?")
echo -n "main body ends on page : $BODYEND   "
if [ "$BODYEND" != "?" ] && [ "$BODYEND" -le "$LIMIT" ]; then
  echo "[OK, limit $LIMIT]"
else
  echo "[OVER LIMIT - CUT]"; FAIL=1
fi

BIBSTART=$(grep -o 'newlabel{page:bibstart}{{[^}]*}{[0-9]*}' $MAIN.aux \
           | grep -o '{[0-9]*}$' | tr -d '{}' || echo "?")
echo -n "references start on page: $BIBSTART   "
if [ "$BIBSTART" != "?" ] && [ "$BIBSTART" -gt "$LIMIT" ]; then
  echo "[OK, clean page break after main body]"
elif [ "$BIBSTART" != "?" ] && [ "$BIBSTART" -le "$LIMIT" ]; then
  echo "[OK, within page limit]"
else
  echo "[CHECK BIBSTART]"; FAIL=1
fi

# --- 2. required ICLR statements present ---
for req in "AI use statement" "Ethics statement" "Reproducibility statement"; do
  if grep -q "$req" "$MAIN.tex"; then
    echo "statement present       : $req"
  else
    echo "[BLOCK] missing required section: $req" >&2; FAIL=1
  fi
done

# --- 3. undefined references / citations ---
UNDEF=$(grep -Ec '^LaTeX Warning: (Reference|Citation).*undefined' "$MAIN.log" || true)
UNDEF=${UNDEF:-0}
echo "undefined refs/cites   : $UNDEF"
grep -E '^LaTeX Warning: (Reference|Citation).*undefined' "$MAIN.log" \
  | sort -u | sed 's/^/    /' || true
[ "$UNDEF" -eq 0 ] || FAIL=1

# --- 4. layout overflow ---
WORST=$( { grep "Overfull .hbox" "$MAIN.log" || true; } \
          | sed -nE 's/.*\(([0-9.]+)pt too wide\).*/\1/p' | sort -rn | head -1 )
WORST=${WORST:-0}
echo "worst overfull hbox    : ${WORST}pt  (aim < 5pt)"
awk -v w="$WORST" 'BEGIN { exit !(w > 5) }' && FAIL=1 || true

# --- 5. anonymity (ICLR is double blind; de-anonymised PDFs are desk rejected)
echo -n "anonymity scan         : "
USER_HOME_PATTERN='/''Users/'
if grep -rqiE "github\.com/[a-z0-9_-]+|acknowledg(e|ment)|${USER_HOME_PATTERN}|/root/|autodl" \
    main.tex sections/ appendix/ tables/ refs/ 2>/dev/null; then
  echo "[CHECK - possible deanonymising string]"; FAIL=1
else
  echo "[OK]"
fi
if grep -q '^\\iclrfinalcopy' "$MAIN.tex"; then
  echo "[BLOCK] \\iclrfinalcopy is uncommented - submission must stay anonymous" >&2; FAIL=1
fi

# --- 6. PDF hygiene ---
if command -v pdfinfo >/dev/null; then
  PAGES=$(pdfinfo "$MAIN.pdf" | awk '/^Pages/{print $2}')
  PAGE_SIZE=$(pdfinfo "$MAIN.pdf" | awk -F': *' '/^Page size/{print $2}')
  PDF_AUTHOR=$(pdfinfo "$MAIN.pdf" | awk -F': *' '/^Author/{print $2}')
  echo "total pages            : $PAGES"
  echo "page size              : $PAGE_SIZE"
  echo "PDF author             : ${PDF_AUTHOR:-<none>}"
  case "$PAGE_SIZE" in "612 x 792 pts"*) ;; *) echo "[BLOCK] not US Letter" >&2; FAIL=1 ;; esac
  if [ -n "$PDF_AUTHOR" ] && ! printf '%s' "$PDF_AUTHOR" | grep -qi '^anonymous'; then
    echo "[BLOCK] PDF author metadata is not anonymous" >&2; FAIL=1
  fi
fi
if command -v pdffonts >/dev/null; then
  TYPE3=$(pdffonts "$MAIN.pdf" | grep -c 'Type 3' || true)
  NOT_EMBEDDED=$(pdffonts "$MAIN.pdf" | awk 'NR>2 && $5 != "yes" {n++} END {print n+0}')
  echo "Type 3 fonts           : $TYPE3"
  echo "fonts not embedded     : $NOT_EMBEDDED"
  [ "$TYPE3" -eq 0 ] || FAIL=1
  [ "$NOT_EMBEDDED" -eq 0 ] || FAIL=1
fi
echo "PDF bytes              : $(wc -c < "$MAIN.pdf" | tr -d ' ')  (limit 52428800)"
[ "$(wc -c < "$MAIN.pdf" | tr -d ' ')" -le 52428800 ] || { echo "[BLOCK] PDF exceeds 50 MiB" >&2; FAIL=1; }

if grep -RniE "TODO|FIXME|TBD" main.tex sections/ appendix/ tables/ refs/; then
  echo "[BLOCK] unresolved manuscript marker found" >&2; FAIL=1
fi

echo "===================================================="
[ "$FAIL" -eq 0 ] || { echo "[BLOCK] submission checks failed" >&2; exit 1; }
echo "all ICLR 2027 format gates passed"
