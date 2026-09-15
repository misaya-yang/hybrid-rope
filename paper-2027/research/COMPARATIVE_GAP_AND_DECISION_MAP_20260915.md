# Field-gap revision and experiment decisions

Status: implemented manuscript revision and CPU preparation; GPU follow-ups are
assigned below but were not launched by this revision. The author requested an
abstract with **no numerical results** and direct, constructive writing. Explain
the problem, identification, constructions and contribution. Frame the Native
comparison as a small trade-off: **2.33% relative (2.14pp), below 3%**, alongside
the clean long-context gain. Retain precise uncertainty in the experiment details. The first page is text-only: problem, approach and contributions; the first figure belongs to the identification section.

Guidance: author-supplied `Hybrid_RoPE_Field_Gap_Codex_Plan_20260915.md`, reviewed
against starting HEAD `8235d2e74fca2b5a2af0a5d1d6bb1d5fdeada9bb`, rather than
assuming its older repository snapshot is current. The live result owner is
[clean confirmation and controls](../../docs/research/next_stage_20260912/TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md).

## Frozen argument and outline

Core contribution: internal frequency placement is a useful, identifiable RoPE
design choice. Support, total displacement, spectrum and coordinate assignment
are different controlled objects. Cosh and TailSpline make this freedom useful
under different learning/deployment objectives; neither is claimed task-optimal.

| Main-text component | Question answered | Evidence / boundary |
|---|---|---|
| Abstract / introduction | What is the design freedom? | No numbers in abstract; fixed-support training and same-spectrum assignment |
| Objects and control table | What does each intervention fix? | Actual endpoints, added span, dose, spectrum, gain; centroid is redundant at fixed dose |
| Identification / Fig. 1 | Is placement useful and distinct from coordinate use? | Three-seed control, BM–Uni, crossings/slot permutations; local costs retained |
| Geometry / discrete corollary | What changes structurally? | Full rotary pairs; no-alias integer kernel equivalence, not a task ranking theorem |
| Constructions / Fig. 2 | How are tables constructed? | Separate Cosh density and TailSpline boundary priors; explicit zero-sum T/C exchange |
| Learning / Fig. 3 | Can a model use the constructed basis? | Three-seed MLA; unanchored endpoints named, full continuation/adaptation tables in appendix |
| Frozen results / Fig. 4 | Does TailSpline work on a larger clean panel? | Complete 2,600 T/P pairs, classic curves kept separate; Native and E1 limits |
| Related work / conclusion | What is distinct and what remains open? | Four scientific groups; strongest unresolved comparison is natural QA plus YaRN |

The abstract now describes the larger source-order confirmation qualitatively.
It does not imply YaRN superiority, or successful mechanism
mediation. BM natural QA remains BM evidence. Native-Z5 remains calibrated.

## Current decisions from actual results

- Clean T/P is complete and raw-recomputed: +11.72pp, 95% interval
  [10.32, 13.11]pp; 12/13 positive task means. It is the current main deployment
  result, not a replacement for the classic 32K curve point.
- E1 T/C: −0.41pp, reported interval [−2.63, +1.82]pp; cross-batch and
  incomplete legacy runtime metadata. No equivalence margin; no positive
  residual-shape conclusion. The audit emits `QUALIFIED_ONLY` and
  `cross_runtime_diagnostic` even after E0.
- E0: 39/39 scores unchanged, six generated sequences changed. This is finite
  probe evidence, not complete executor equivalence.
- Original Native-8K LM: T PPL 5.295283 versus Native 5.275919; cost reported.
  The completed 130-prompt Native task reference scores T/Native 89.74/91.88%; T−Native −2.14pp, interval [−6.14,+1.92]pp; a 2.33% relative native task trade-off.
- Natural-QA631 is now complete and integrated: T/P 41.08/40.88% F1, +0.20pp [−1.53,+1.89]pp. Optional clean/classic YaRN and Native-Z5 are not reported outcomes of this revision. Existing processes retain their owners; current SSH endpoint alone
  does not establish which disk/queue is the original or clone.

## Execution order and acceptance

| Item | Next action | Scope / owner / output |
|---|---|---|
| E0 | Reuse completed report | Current endpoint; never replay it to unlock E1 |
| E1 | Preserve current diagnostic; if stronger shape attribution is needed, only run C at batch 1 | 390 generations + at most 138 LM rows; reuse T and any separately proven equivalent LM identity; supplement legacy runtime provenance before promotion |
| E2 | Reuse complete clean T/P | Original clean input owner; add optional YaRN only when an owner and budget are assigned; never regenerate T/P |
| E3 | Reuse completed Natural-QA631 V2 | Complete whole 631 plus native-within316 / extended315; 524 source-context clusters; primary +0.20pp [−1.53,+1.89] |
| E4 | Arrange clean/classic YaRN when assigned | Optional and currently unscheduled per the current execution owner; a joint T>P and T>Y claim needs two-comparison simultaneous inference |
| E5 | Read completed Native task reference | Complete 130 paired inputs; original frequencies AND original gain; noninferiority not established |
| A1 | Add static YaRN to the exact natural QA631 | **Prepared, not launched**; 631 generations, maximum 41,056 generated tokens; reuse T/P |
| Native-Z5 | Continue separate registered calibration | design16 / selection16 / confirm18, 40 steps, five position parameters; no TailSpline retuning |
| M1 | Hold | Current E1 does not establish the required effect; needs clean E1 direction and a separately accepted mechanism budget |

A1 has a concrete standalone launcher:
[run_naturalqa_yarn.sh](../../experiments/iclr2027_three_track_sprint_20260915/run_naturalqa_yarn.sh).
Its default invocation prints the planned action and executes no model. With
`--execute`, it validates existing 631 inputs and T/P completion, runs only YaRN,
and writes new T/P and T/Y reports with Bonferroni familywise intervals. Old
reports are preserved. The launcher is not inserted into either running queue.
Before assigning the GPU, compare the current model/tokenizer/runtime with the
T/P owner; missing legacy identity is not repaired by matching names.

The existing [natural-QA reporter](../../experiments/fixed_rope_three_interfaces_20260913/matched_naturalqa_report.py)
retains question weighting within task and source-context cluster bootstrap,
adds generated-length health and document-equal sensitivity, and supports a
predeclared two-comparison family. Native strata and task decompositions remain
secondary. Input token total is printed from the manifest by the A1 preparation;
hours must use measured prefill/decode throughput, not a model-name estimate.

### Exact C-only repair command (after assigning its owner)

Use a fresh output directory; never alter the completed cross-batch C raw.
The following paths are relative to the existing remote experiment layout:

```bash
python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "$CLASSIC/assets/ppl46/manifest.json" --model "$MODEL" --arm Native \
  --extra-panel "$CLASSIC/assets/full13/rows.jsonl" --only-extra-panels \
  --length-cap 8192 --length-cap 16384 --length-cap 32768 \
  --lm-length-cap 8192 --lm-length-cap 16384 --lm-length-cap 32768 \
  --prefill-chunk-size 8192 --batch-size 1 \
  --static-table-json "$DOSE/tables/llama_s4_tailspline_dose_control.json" \
  --table-label llama3_8b_s4_dose_control_c_batch1 \
  --out "$DOSE/runs/dose_control_c_batch1" --execute
```

Primary E1 endpoint stays Full-13 task-equal log-length T−C. Preserve the original
reported interval as historical; audit source-blueprint identity across lengths
before stronger confirmatory uncertainty. A zero-crossing interval remains
unresolved. Do not invent a post-result equivalence threshold or rerun to
significance. A confirmed C>T would reject the one-sided ranking prediction,
not the existing T/P method result or all allocation effects.

### Conditional M1, frozen scope

If E1 and an additional budget support it, use only Llama S4, unchanged T/C,
`niah_single_2`, `niah_multikey_2`, `niah_multiquery`. Design: 30 blueprints;
confirmation: 120 independent blueprints, each at 16K/32K. Select at most eight
(layer, query-head) pairs on the design block using direct evidence log-odds,
then freeze. Preserve exact evidence spans and GQA mapping. Capture pre-RoPE K;
patch only the final prompt query and decoding queries. Six conditions: C, T,
C+T-reader, C+shuffled-delta, T+C-reader, T+shuffled-delta. No-op hooks must
recover baseline logits/tokens on the chosen runtime. Confirmation is 1,440
generations; paired blueprint-cluster simultaneous inference tests both
rescue-over-shuffle and damage-over-shuffle directions. No extra heads, tasks
or samples after a null result. No implementation platform is built while this
branch is inactive. Success supports a local readout contribution, not tail-only
mediation or a complete Transformer mechanism.

No new BM GPU, model, gain grid, HELMET download or sparse/MoE/mHC method is
part of this revision. HELMET-RAG requires a separately frozen official-data
contract and resource allocation after the direct comparisons.

## Literature verification and positioning

Primary sources checked during this revision:

- [Round and Round](https://arxiv.org/html/2410.06205v3), §6 / Appendix D:
  frequency-function study and Gemma training; no frozen-installation equivalence.
- [FoPE](https://arxiv.org/html/2412.17739): component changes and learning protocols.
- [STRING](https://arxiv.org/html/2502.02562), Theorems 3.2–3.4:
  differentiability, identity, relative law, orthogonal basis; bounded universality.
- [Massive Values](https://arxiv.org/html/2502.01563): Q/K interventions, not V.
- [GRAPE](https://arxiv.org/html/2512.07805), Corollary 2.1:
  frequency–norm coupling, not a new claim here.
- [LongRoPE2](https://arxiv.org/html/2502.20082): search and mixed-context training.
- [PINE](https://arxiv.org/html/2407.01100): document masks/order and invariance.
- [Selective RoPE](https://arxiv.org/html/2511.17388): learned input-dependent rotation.
- [MrRoPE](https://arxiv.org/html/2601.22181): nearest static mixed-radix comparator.
- [LeRoPE](https://arxiv.org/html/2607.10134v1), §3.2:
  frequency–attention–value gradients and fixed learned tables already exist.
- [HoPE](https://aclanthology.org/2025.acl-long.1123/): high-frequency ACL work,
  distinguished from namesakes.
- [PRoPE](https://arxiv.org/html/2507.10496): camera geometry;
  [RePo](https://arxiv.org/html/2512.14391): learned position dependence.
- [HELMET](https://princeton-nlp.github.io/HELMET/): application coverage;
  current natural-QA pool is not HELMET or complete LongBench.

“Million-Token Context Scaling” remains insufficiently identified. It was not
replaced by a guessed paper or cited as verified.

## Delivery milestones

[Official ICLR author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
and [call for papers](https://iclr.cc/Conferences/2027/CallForPapers) were checked:
abstract September 18 and paper September 25, 2026, both 23:59 AoE; main text at
most nine pages. Dates do not authorize submission or publication.

1. Current revision: freeze the no-number abstract and complete-evidence main
   paper, discrete proof/examples, result owner, claim mapping, source package.
2. Before abstract deadline: integrate only newly completed, verified evidence;
   the abstract is already substantive without pending A1/M1.
3. Next comparison block: existing natural T/P work and optional YaRN, then A1 on its assigned
   GPU. M1 stays conditional; do not delay required comparisons for it.
4. Forty-eight hours before paper deadline: freeze numbers, plots and claims;
   review the rendered PDF as a standalone reader.
5. Last day: independent source-package build, anonymity and format checks.

Build/reproduction commands remain the repository's existing toolchain:
`bash paper-2027/compile.sh`, `python3 paper-2027/package_source.py`.
The additional CPU checker is
[verify_discrete_kernel_equivalence.py](../../experiments/iclr2027_three_track_sprint_20260915/verify_discrete_kernel_equivalence.py);
[verify_field_gap.py](../figs/verify_field_gap.py) recomputes clean point estimates,
checks bootstrap agreement and regenerates the complete task table.

## Final delivery receipt

[Machine-readable verification](FIELD_GAP_REVISION_VALIDATION_20260915.json):
scientific body 9 pages, complete PDF 59 pages; no undefined citations/references
or overfull boxes; anonymous metadata and embedded fonts. The final first page
contains the complete introduction and contributions with no figure. Main pages
and changed proof/result appendices were visually inspected; the overlapping
plot annotations found during review were repaired.

The 133-entry anonymous source package passed its manifest and an independent
LaTeX build. Its clean score-only checker and discrete kernel examples also
passed with the declared NumPy dependency. The original 15 operator checks and
six evidence/reporting regression tests passed. No model experiment was started
by this revision. All 2,600 clean answer scores were recomputed from existing
raw, and T/C task points and complete LM-row coverage were checked.

Documentation validation reports no new link or source issues from this revision.
It retains two pre-existing historical snapshot mismatches in the finite-window
receipt and fixed-u result; their identities were not rewritten. These are
recorded explicitly in the verification JSON.

## Title and abstract refinement for submission

Recommended and installed title: **Beyond the Base: Frequency Allocation in RoPE**.
The abstract is 154 whitespace-delimited words, with no numerical results.
The first page remains text-only. [Submission text](../title_abstract.txt)
contains the exact title and plain-text abstract for copying into the submission form.

| Title considered | Decision | Reason |
|---|---|---|
| Beyond the Base: Frequency Allocation in RoPE | Selected | Names the physical design object directly; retains the central range/allocation distinction and accommodates both learning and frozen deployment |
| Beyond the Base: Exponent Allocation in RoPE | Previous version | Accurate after the normalized exponent is defined; less immediately clear to readers encountering the work for the first time |
| TailSpline: Training-Free Context Extension through Frequency Allocation | More narrowly method-focused | Would shift the paper identity toward one construction and underrepresent the controlled allocation and learning contributions |

The revision replaces a section-by-section inventory with a direct sequence:
fixed-range allocation finding, controlled identification, positional structure,
explicit constructions, and observed learning/deployment benefits. Technical
terms such as the frequency multiset and integer-position kernel criterion
remain in the body rather than appearing as a list of abstract keywords.
Cosh and TailSpline have separate roles. The task claim names the measured
MrRoPE-Pro comparator; pending TailSpline natural QA and optional YaRN outcomes
are not implied. The small native-window trade-off is retained succinctly.

The author uses September 17 AoE as the target for finalization. The official
[ICLR author page](https://iclr.cc/Conferences/2027/AuthorGuidelines), checked
in this revision, lists September 18, 2026 at 23:59 AoE for abstract submission.
No submission was performed.

## Completed two-cycle review integration

[Two independent PDF-only reviews and dispositions](pdf-review-rounds/20260915_two_rounds/README.md)
are preserved with frozen input PDFs. Both reviewers gave internal 6/10 weak-accept
recommendations; those scores describe their inputs. The final optimization adds
the completed Natural-QA631 result, independently rescored and cluster-bootstrap
reproduced, to the main text and appendix. It promotes the well-specified 151.9M
crossed-table result to Figure 1 and the full M4 comparison to the main evidence
summary. Native uncertainty, the E1 diagnostic and historical MLA/permutation
provenance are stated at their corresponding claims. No new GPU experiment was
launched by this review task.
