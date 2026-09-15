# Five further Astra/Sol review and optimization rounds

**Status: COMPLETE.** Five sequential review/integration cycles, ten independent model contexts.

Within each round, Astra and Sol received the same frozen PDF and identical prompts.
They did not receive prior reviews, repository sources, desired scores or author discussions.
R03/R04 use single reviews; R05–R07 use four perspectives plus AC synthesis in each model.
Those perspectives are simulated within a context, not four independent agents.

| Round | Frozen input | Astra | Sol | Integration |
|---|---|---|---|---|
| R03 | [PDF](r03/input.pdf) | [6/10](r03/astra.md) | [7/10](r03/sol.md) | [Disposition](r03/disposition.md), [output](r03/optimized.pdf) |
| R04 | [PDF](r04/input.pdf) | [6/10](r04/astra.md) | [7/10](r04/sol.md) | [Disposition](r04/disposition.md), [output](r04/optimized.pdf) |
| R05 | [PDF](r05/input.pdf) | [AC 7/10](r05/astra.md) | [AC 7/10](r05/sol.md) | [Disposition](r05/disposition.md), [output](r05/optimized.pdf) |
| R06 | [PDF](r06/input.pdf) | [AC 7/10](r06/astra.md) | [AC 7/10](r06/sol.md) | [Disposition](r06/disposition.md), [output](r06/optimized.pdf) |
| R07 | [PDF](r07/input.pdf) | [AC 7/10](r07/astra.md) | [AC 8/10](r07/sol.md) | [Disposition](r07/disposition.md), [output](r07/optimized.pdf) |

R05 changed the review format at the author's request. Score changes across that
boundary cannot be interpreted as a controlled estimate of manuscript improvement.
All scores apply to their frozen inputs, not subsequent edits or real acceptance probabilities.

## Four perspectives and AC

| Round/model | Novelty | Theory | Experiments | Narrative | AC |
|---|---:|---:|---:|---:|---:|
| R05 astra | 7 | 7 | 6 | 7 | 7 |
| R05 sol | 7 | 7 | 6 | 7 | 7 |
| R06 astra | 7 | 6 | 6 | 7 | 7 |
| R06 sol | 7 | 7 | 6 | 7 | 7 |
| R07 astra | 7 | 7 | 6 | 7 | 7 |
| R07 sol | 8 | 8 | 7 | 8 | 8 |

## What the integrator changed

- Centered quality within the intended context range, with z as the research object.
- Made TailSpline the primary method; Cosh remains a supporting extrapolation transport.
- Added the completed clean16K report (650 pairs), alongside clean32K, with complete task means.
- Promoted complete-pair geometry and its ordering counterexample into the main argument.
- Clarified that total displacement is a statistic of allocation at fixed native table and endpoints.
- Restored the concrete750M continuation results from existing evidence.
- Moved detailed control classification and classic curves into the corresponding appendices.
- Preserved exact results and intervals; rejected redundant defensive prose and out-of-scope requirements.

## Calibration and reviewer errors

[Review protocol](review_protocol.md) records source restrictions and the panel prompt.
The requested fresh Astra single review of MrRoPE Markdown gave4/10:
[report](reference_single/astra.md), [integrator assessment](reference_single/assessment.md).
Its source format differs from PDF and full figures were unavailable. It is not a valid
conversion from simulated score to conference tier, nor a reason to dismiss all criticisms.

Concrete rejected mistakes include Sol R04 inverting EOS/non-EOS counts and Sol R07
suggesting use of the exempt page10 for extra substantive main text. Repeated wishes for
more baselines, mechanism isolation and wider natural tasks are separated from actual
errors in the stated allocation results.

## Final validation

Final manuscript:9 main pages,63 total pages; abstract151 words with no numerical results;
no figure on page1. Five main figures and both main tables are within the body.
Compilation reports zero undefined references/citations and zero overfull boxes.
All fonts embedded, anonymous metadata. Final main pages1–9 visually inspected.
The142-file source payload independently compiles after extraction.
Frozen inputs and input/output lineage were verified. No GPU run was launched.
