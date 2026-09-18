# Beyond the Base：当前论证与证据安放表

## Current manuscript (2026-09-18)

Current revision: [Kanana evidence and design-explanation integration](revision_20260918/README.md).
The evidence-integration baseline is [v4](../history/v4.pdf); later chapter/abstract passes used their own pre-edit PDFs. Main text: 9 pages; total: 35. The title and 167-word abstract are frozen.
Title and section order are retained. The abstract contains no numbers; the first page has no figure.

The argument studies z through identification, positional structure, construction and measured quality.
TailSpline provides the frozen extension construction. NCP retains the native fixed-support test,
now supported by same-target language modeling; Cosh retains its training/adaptation role.
NTS2 is a research-stage tradeoff study outside this manuscript.
For future native enhancement, real downstream QA is primary; RULER and LM cannot replace it.

| Claim | Evidence | Location |
|---|---|---|
| Allocation matters at fixed support | A01,A08; paired training and frozen controls | Section 3, Appendix C (Figure 1 now explains allocation and distance scaling) |
| Positional structure and learned use differ | A03-A06,A48; full pairs, crossing, reassignment, rank/quality | Section 4, Figure 2, Appendix A |
| TailSpline has an exact finite-grid construction | A37,A42; stated one-sided minimum-bending objective, closed form and CPU audit | Section 5.1, Figure 3, Appendix B.1 |
| One s=4 deployment serves L/2L/4L inputs | A39,A46,A52; large clean Llama panels, original Native reference retained | Section 6.1, Table 1(a), Figure 4, Appendix D.2 |
| Cross-family direct comparisons | A49,A59,A60,A65; five-family Full-13, large Llama/OLMo NIAH and PPL | Section 6.1, Table 1(b), Appendix D.6 |
| Natural-task quality | A45,A50,A53,A57,A59,A60,A65; QA631, LongBench v2, GLM book pools and Kanana complete-book QA | Table 2, Appendices D.4,D.7 |
| Publisher-recommended runtime deployment | A65; Kanana 64K Full-13 and 128K complete-context English QA versus official runtime YaRN | Section 6.1, Tables 1-2, Appendix D |
| Frozen transfer to 70B NF4 | A61; same public 8B tables, paired 70B Full-13/QA/PPL | Tables 1-2, Appendices D.6-D.8 |
| Shape effects at equal displacement | A51; clean T-C at16/32K | Sections 3.1,6.1, Appendix D.3 |
| Native allocation at fixed support | A54,A62; NCP original780/new130 task panels, LM128 and context benefit | Section 6.2 and Appendix E; fixed-support control |
| Cosh supports learned extrapolation | A09-A12,A19; 432M/750M/1.485B, Llama LoRA, OLMo adaptation | Section 6.3, Figure 5, Appendix F |
| Higher-factor task profiles | A58,A61; Llama/Qwen mixed endpoints and 70B completed PPL | Appendix D.8 |

The [portable inputs](../figs/revision_evidence_inputs.json) and [generator](../figs/make_revision_evidence.py)
retain score and protocol identities. Separate panels are never pooled as independent repetitions.
Native-QA question-weighted primary scores and source-equal sensitivity remain different estimands.
The fixed-support 151.9M study belongs to the Cosh training series, not a separate FMRoPE experiment.
The full task chart and 432M curve remain in the main paper.

Figure 1 now illustrates YaRN/MrPro/TailSpline allocation and distance responses. Fixed-support scores remain in Section 3 and Appendix C; Cosh and NCP retain their supporting scientific roles.


## Historical maps

Earlier chapter/figure placements and dated interpretations are preserved in the [navigation archive](../../docs/archive/navigation_20260918/index.md). They do not override the current map above.
