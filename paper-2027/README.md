# Beyond the Base: Exponent Allocation in RoPE

`main.tex` is the active ICLR 2027 manuscript. The paper studies exponent
allocation through positional geometry, EVQ-Cosh construction, trained-model
comparisons, and adjustments to frozen mature models.

- Read the paper: [main.pdf](main.pdf) and [main.tex](main.tex).
- Build the PDF: `bash paper-2027/compile.sh` from the repository root.
- Rebuild the revision figures from recorded results:
  `python3 paper-2027/figs/make_exponent_revision_figures.py`.
- Package the active source and all TeX dependencies:
  `python3 paper-2027/package_source.py`.
- Current revision scope: [REVISION_BRIEF.md](REVISION_BRIEF.md).
- Manuscript state and verification: [HANDOFF.md](HANDOFF.md).
- Writing choices: [NARRATIVE_GUIDE.md](NARRATIVE_GUIDE.md).

The source archive is `exponent-allocation-source.zip`. Experimental result
owners remain in the repository and are identified in the internal evidence
map. The complete earlier manuscript is available at `main_0726`; the NeurIPS
source in `../paper/` retains its historical identity.
