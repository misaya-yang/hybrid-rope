# Astra / Sol PDF-only comparative review

Both agents received identical prompts and the same frozen PDFs, with no repository,
TeX, report, web or cross-review access. Four perspectives are viewpoints within
each independent agent, not eight separately sampled reviewers. See identity.json.

## Initial candidate A versus pre-edit B (v2)

Both found stronger evidence and no central mathematical or pairing error. A still
contained the interim T2 native-deployment results. They questioned the breadth of
native-task wording, asked for clearer panel separation and distinguished whole
T2 deployment from pure allocation. Subsequent author steering and additional QA
results removed the entire T2 study from this manuscript, preserving it in research
records pending broader confirmation. NCP's established content was retained.

## C/D versus B: completed comparison

| Perspective | Astra | Sol |
|---|---|---|
| Novelty/significance | Same central allocation question, stronger practical breadth | Formulation, controls and expanded validation strengthen the paper |
| Theory/method | No central regression; finite-series convexity certificate strengthens NCP | No printed mathematical error after visual correction |
| Experiments | Paired direct baselines, 70B NF4, separate GLM book pools, same-target LM are useful | Comparisons remain rowwise and matched; no evidence loss identified |
| Narrative | NCP fixed-support content retained; method roles clear | Native LM claim is better attributed; panel labels resolve ambiguity |

Astra: no remaining material error, mismatched central comparison or substantive
regression. Sol: no remaining material error or evidence regression; lean accept.
These are internal reviews, not an acceptance-probability estimate.

## Adopted repairs

- Mark the main table's distinct panels (a)/(b) and state comparisons are within rows.
- Give the observed 70B QA scores directly.
- Preserve NCP's controlled native task results and add paired native LM evidence.
- Identify the native LM corpora, document/window counts and score construction.
  Hash sorting and window-iteration implementation details remain in code, following
  the author's request to keep methodological prose proportional to its value.
- Bound the implemented NCP scalar objective's curvature; the certificate refers to
  its 32-mode approximation, not universal Transformer task quality.

## Verified PDF-extraction false positive

Sol initially reported the bound in Appendix E.1 as 4+36/x²+16/x²≤3.5,
calling it a mathematical regression. The printed PDF has a square root over
4+36/x². Direct high-resolution visual inspection gave sqrt(6.25)+1=3.5 at x=4;
Sol explicitly withdrew the error and its conditional-accept language. Astra
independently read the correct bound. No theorem was changed to accommodate the
extraction error.

## Opinions not treated as required changes

Both versions already give the native s=4 scores and complete adverse task table.
The request to replace the author's 'slight reduction' wording with a prominent
failure description is an emphasis preference, not hidden data or a new regression.
No new generic caveat, universal-optimality demand or experiment campaign was added.
The statement page is exempt and unchanged in role; compressing it has no claim value.

The E candidate removed implementation-order prose from the native LM protocol,
retaining its corpus names, counts and paired scoring. The subsequent module audit
identified targeted prose repairs, completed in F.

## F versus B: final module repairs

Both agents independently read the same F/B PDFs under prompt_F.md. Both found
the question-to-evidence-to-explanation-to-construction progression clearer,
with no printed mathematical error or substantive evidence loss. They highlighted
the restored structural question, the native-table design motivation and the
interpretation of TailSpline's solution before proof details.

Astra found no essential repair. Sol retained its concern about the PDF not
specifying document/window selection order, while acknowledging that it was not
a demonstrated validity error. Both suggested more explicit uncertainty wording
around observed natural-QA leads; Sol also preferred 'additional' to 'new' for
the native task panel. These were assessed against the actual text and source
package; no generic caveats or execution-order prose were added. See
[the disposition](../MODULE_REPAIRS_20260917.md) for each decision.

The delivered main.pdf is identical to F. Baseline B is identical to history/v2.pdf.
