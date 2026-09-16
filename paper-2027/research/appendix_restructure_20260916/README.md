# Appendix reorganization and main-text reading pass, 2026-09-16

Status: completed. The manuscript is **29 pages**, down from69; the main text
ends on page9. Six thematic appendices replace twelve accumulated sections.
The abstract remains158 words without digits; the first page contains no figure.
No model evaluation or remote job was started by this revision.

## Editorial policy and conference rule

The author's target is **at most35 total pages**, with40 as the maximum.
These are editorial constraints. The [ICLR2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
(accessed2026-09-16) impose a nine-page submission main-text limit, allow unlimited
references and appendices, and place supplementary text after the references.
Reviewers are not required to read appendices. The main text therefore carries
the motivation, contribution, constructions and decisive comparisons itself.
The build script enforces the main-text limit and author's total-page maximum.

## Retained argument and disposition

| Material | Current PDF location | Retention |
|---|---|---|
| Full-pair Gram, basis invariance and effective-rank identity | A.1 | Formula and proof |
| Slow collapse and finite-window geometry | A.1–A.2 | Asymptotic derivation, coefficient and finite numerical table |
| Static ordering examples, integer kernel equivalence and content matching | A.3–A.5 | Explicit examples, hypotheses and proofs |
| Rank versus task quality | A.6 | Matched frequency-table ranks and existing task results |
| TailSpline/BM and equal-displacement C | B.1–B.2 | Finite-grid minimizer, uniqueness, boundary interpretation and exact identities |
| Cosh | B.3 | Objective, existence/uniqueness argument, solution and actual finite installation |
| Scale response and finite phase intervals | B.4–B.5 | Identities, reference definitions and finite comparison quantities |
| Fixed-support training, range interaction and crossing | C.2–C.3 | Architecture, data, training/evaluation recipe and all paired-seed contrasts |
| Factorial, frozen equal-support/shape and slot interventions | C.4–C.6 | Controls, scoring units and results; repeated displays consolidated |
| Clean Llama8/16/32K and OLMo16K | D.1–D.2 | Shared protocol, complete13-task table, main intervals and output statistics |
| Equal-displacement T/C | D.3 | Both lengths, full task breakdown and intervals |
| Natural QA and LongBench-v2 | D.4 | Sample units, scoring, clustering, task/domain tables and length strata |
| Classic length curves and native references | D.5 | Original protocol, task curves, PPL results and controls |
| NCP | E | Reference risk, constraints, numerical construction and full task results |
|432M MLA,750M continuation and matched OLMo learning | F | Architectures, recipes, metric definitions, results and position schedule |
| Video, additional architecture studies, earlier profile families and peripheral theory | `extended-records/` | Historical sources retained outside the submission PDF |
| Duplicate figures and repeated protocol/result explanations | Consolidated above | One current location per topic |

The compact PDF retains the evidence needed for its current main claims, rather
than every historical study. No experiment output was deleted. The immutable
[previous69-page source snapshot](previous_69page_source.zip) preserves the
previous full document. The new anonymous source package also carries the
[extended records](../../extended-records/README.md) with their historical labels.
Those files are not active TeX dependencies.

## Information checks and repairs

The second pass restored two protocol details that were too compressed:

- OLMo's two-stage position schedule now follows the recorded band quotas,
  offset clipping and realized-gap construction. The compressed draft's
  shorthand schedule was replaced; the bundled checker reproduces all268/88
  recorded offsets/targets.
- The factorial again specifies validation splitting, text packing/repetition,
  chunk selection and the scored next-token targets.

Shared gain application and reuse of the already fixed tables are explicit.
The main-text classic-result pointer and duplicate D.2/D.4 pointers were repaired.
The classic figure caption now describes the task curves actually plotted;
PPL remains in the adjacent numerical text. None of these changes alters a
reported result.

## Main-text narrative

The opening now introduces quality throughout a target window, then asks what
interior frequency placement contributes beyond the chosen range. The argument
continues through controlled identification, positional structure and learned
use, explicit construction, and task evidence. TailSpline and NCP remain separate
contributions; Cosh remains supporting learning/extrapolation evidence. The
conclusion was shortened without removing these contributions.

## Verification

See [validation.json](validation.json). The checks include:

- successful local and extracted-source builds, both9 main/29 total pages;
- zero undefined citations/references, zero overfull boxes and embedded fonts;
- visual inspection of all main pages and the reorganized appendices;
-117 consolidated task-table cells checked against stored per-input scores;
-19,722 stored-score rows and exact NCP table checked with the existing verifier;
- finite-grid optimization, full-pair geometry and scale-response CPU checks;
- exact reproduction of the two historical routing schedules;
- historical archive comparison (only whitespace normalization in two text files);
- source manifest verification and identical extracted PDF text after rebuilding.
