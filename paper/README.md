# NeurIPS Submission Skeleton

Anonymous submission scaffold for the EVQ-Cosh paper.

## Compile

Submission mode is the default. Use the official style file in this directory.

```bash
cd paper_draft/submission
~/miniconda3/bin/conda run -n aidemo latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
cd paper_draft/submission
~/miniconda3/bin/conda run -n aidemo pdflatex main.tex
~/miniconda3/bin/conda run -n aidemo bibtex main
~/miniconda3/bin/conda run -n aidemo pdflatex main.tex
~/miniconda3/bin/conda run -n aidemo pdflatex main.tex
```

## Current intent

- keep the body focused on `Figure 2` and `Figure 3`
- keep `Figure 1` and phase9f/phase15 expansions in the appendix
- keep the review version anonymous and submission-mode compliant

