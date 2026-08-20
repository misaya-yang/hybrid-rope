# ICLR 2027 submission checklist

Official sources checked on 2026-08-19: the ICLR 2027 Author Guidelines and AI
Policy for Authors. `./compile.sh` runs the mechanical gates; unchecked items
require an author decision or final manual review.

## Format and build

- [x] Official `iclr-2027-style-files.zip` files are byte-identical to the
      copies used by this package.
- [x] Anonymous submission uses `iclr2027_conference` and leaves
      `\\iclrfinalcopy` commented.
- [x] Main text is at most **9 pages**; bibliography and appendices follow the
      exempt statements and do not count.
- [x] AI use statement is present before the bibliography.
- [x] Ethics and reproducibility statements are present before the bibliography.
- [x] Undefined citations/references, overfull boxes above 5 pt, non-Letter
      pages, Type 3 fonts, unembedded fonts, PDF size above 50 MiB, and obvious
      anonymity leaks are hard failures in `compile.sh`.
- [ ] Final submission-day build: clean temporary directory, both LaTeX passes,
      BibTeX, source/PDF hash receipt, and visual review of all pages.

## Scientific consistency

- [x] The central claim is fixed-range interior allocation as an independent
      training-time variable; EVQ-Cosh is a closed-form constructive instance,
      not a universal optimum.
- [x] Static full-RoPE geometry is described as positional-basis geometry, not
      an LM-quality or extrapolation predictor.
- [x] Exact-range, M4, 50M table-by-weights, OLMo-2, and LLaMA protocols remain
      separate; no cross-protocol averaging or seed splicing.
- [x] Teacher-forced NLL/PPL, strict autoregressive exact, 2Wiki, RULER, and
      causal source-use endpoints remain distinct.
- [x] Exact-range three-seed aggregate is not labelled raw-backed or used with
      unsupported confidence intervals; the body uses raw-backed seed 42 plus
      the independent raw-backed M4 factorial.
- [x] The transplant theorem is limited to exact, fixed,
      position-independent invertible Q/K compensation.
- [x] LeRoPE is positioned as learned/fixed-table evidence; no claim says it
      validates EVQ, is approximated by EVQ, or is dominated by EVQ.
- [x] LLaMA 32K RULER Native macro is not reported because three shards never
      started.
- [ ] Author reviews every number against the named owner one final time after
      layout freezes.

## Anonymity and release package

- [x] No author names, affiliations, acknowledgements, private machine paths,
      host names, or identifying repository links appear in the manuscript.
- [x] Anonymous code archive removes identities, credentials, private paths,
      host names, checkpoints, caches, and ignored raw results.
- [x] Anonymous code archive includes the frequency initializer, analysis
      scripts, evaluation contracts, and exact-range configs needed for claims.
      The `iclr2027` packager profile passed its leak scan, ZIP integrity test,
      isolated paper build, RULER reanalysis, and 142 focused tests on
      2026-08-20.

## Author actions before submission

- [x] Author confirmed on 2026-08-19 that the AI use statement is complete and
      literally true for every required and recommended category it lists.
- [x] Author confirmed on 2026-08-19 that all authors have current OpenReview
      profiles and satisfy the ICLR 2027 reciprocal-reviewing requirements.
- [x] Dual-submission timing and current-draft distinctness audited on
      2026-08-19. ICLR expressly permits an abstract while the NeurIPS decision
      is pending; NeurIPS notifies on Sep 24, before the ICLR full-paper
      deadline on Sep 25. The current title, central theory, identification
      controls, co-adaptation result, and mature-scale evidence are materially
      different from `../paper/`; see `CHANGES_FROM_NEURIPS2026.md` §2.1.
- [ ] If NeurIPS accepts, cite the accepted paper in third person and state the
      old/new contribution boundary before the ICLR full-paper upload. No such
      citation action is needed if NeurIPS rejects.
- [x] Official deadlines checked on 2026-08-19: abstract Sep 18 and full paper
      Sep 25, 2026 AoE. Recheck the live page immediately before submission.
- [ ] OpenReview title and abstract exactly match the final PDF.
