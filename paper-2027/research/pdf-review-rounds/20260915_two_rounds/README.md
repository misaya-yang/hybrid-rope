# Two independent PDF review and optimization cycles

## Correction after author challenge (2026-09-15)

The parent withdraws its assertion that the historical MLA cache-building and
document-exclusion records were not retained. The first PDF reviewer requested
a more specific evaluation description. The parent inspected cache-loader code
but did not establish the absence of historical records; it nevertheless added
that assertion to the manuscript. The second PDF reviewer then treated the
parent-added assertion as an evidence gap. This is not independent corroboration.
The added appendix paragraph and Figure 3 warning have been removed; the actual
evaluation protocol, seed scores and aggregates remain. The A09 interpretation
and current claim map are corrected. The frozen reviews/dispositions below
preserve the history and must not be treated as a verified provenance finding
or an instruction to rerun this experiment. New narrative edits begun during
the author discussion were paused and restored from the preceding source bundle.


The author requested two review/optimization cycles. Each reviewer receives only
one frozen manuscript PDF and a neutral ICLR-style rubric, with a fresh agent
context. Reviewers must not read repository sources, author discussions, prior
reviews, memory, or outside literature. PDF text extraction and rendering are
allowed. The integrator separately inspects sources, verifies findings, reuses
existing evidence and edits the manuscript. No new GPU experiment is authorized
by this review workflow.

The rubric follows the current
[ICLR 2027 reviewer guidance](https://iclr.cc/Conferences/2027/ReviewerGuidelines):
question and motivation, supported correctness, significance/new knowledge,
experimental rigor, reproducibility and clarity. SOTA is not required. Internal
1–10 scores are reviewer judgments, not official ratings or calibrated acceptance
probabilities. The [author rules](https://iclr.cc/Conferences/2027/AuthorGuidelines)
allow nine main pages, with references and appendices exempt.

| Cycle | Frozen input | Reviewer | Integration |
|---|---|---|---|
| 1 | [input.pdf](r01/input.pdf), [identity](r01/input_manifest.json) | [Review](r01/review.md), internal 6/10 | [Disposition](r01/disposition.md), [optimized PDF](r01/optimized.pdf), [validation](r01/validation.json) |
| 2 | [input.pdf](r02/input.pdf), [identity](r02/input_manifest.json) | [Review](r02/review.md), internal 6/10 | [Disposition](r02/disposition.md), [optimized PDF](r02/optimized.pdf), [validation](r02/validation.json) |

Author constraints retained in editing: no numerical results in the abstract,
no figure on page one, allocation as the central research contribution, and
direct benefit/trade-off language. The native task difference is the measured
2.33% relative difference on its specific panel. Statistical uncertainty is
reported with the corresponding estimate. Review suggestions are judged against
the actual claim and evidence, rather than adopted indiscriminately.

## Final result

Both cycles are complete. The main text remains nine pages; the complete paper
is 61 pages. Both independent reviewers gave weak-accept recommendations on
their own inputs (internal 6/10; confidence 3/5 and 4/5). These are not ratings
of the final post-optimization PDF or calibrated acceptance probabilities.

The final paper uses the explicit 151.9M crossing in Figure 1, adds the complete
M4 summary, clarifies the frequency-space guarantees and empirical hypotheses,
reports E1 and Native uncertainty next to their claims, and adds TailSpline's
complete Natural-QA631 result as main Table 2. The new QA report was completed
by the existing queue during review and independently rescored by the integrator.
All question/cluster strata and task means are retained.

The remaining empirical improvements are the runtime-matched C arm and an
explicitly disjoint-corpus evaluation of the existing MLA checkpoints; optional
clean YaRN and clean-native comparisons address deployment breadth/precision.
They are documented future evaluations, not outcomes claimed by this review.

Final delivery checks: 139 source-package manifest entries verified, key figures
and clean/natural tables regenerated in an independent extraction, and full
LaTeX build passed with nine main pages, zero undefined references/citations,
zero overfull boxes, anonymous metadata and embedded fonts. Documentation checks
retain only the two pre-existing historical snapshot mismatches recorded in the
validation JSON; no new navigation error was introduced.
