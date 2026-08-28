# ICLR 2027 submission narrative and experiment plan

> **PARTIALLY SUPERSEDED (2026-08-27):** positioning and the revision roadmap
> are now owned by [`REVISION_BRIEF.md`](../REVISION_BRIEF.md) v2. This memo
> remains the authority on what may and may not be run before submission and on
> the post-submission headroom inventory.

- **Date:** 2026-08-26
- **Status:** submission decision memo, updated after the completed dose and
  Native-4K diagnostics
- **Role:** decides what the paper's single claim is, which owned numbers carry
  it, which objection is pre-empted in the body, and what may and may not be
  run before submission
- **Objective:** acceptance. Not method completeness, not a long-context
  system, not closing every open method question

Mechanism, constraints, and the post-submission headroom inventory are owned by
[`attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`](attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md).
This file holds only the submission decisions.

---

## 1. The one sentence

> A RoPE head spends a third of its frequency budget on duplicate positional
> directions. Reallocating it at fixed support, with no learned parameter,
> takes a released 1.5B model from `0.0056` to `0.6047` on 16K retrieval.

That is what a reviewer should be able to repeat to the AC. Everything in the
paper either sets it up or follows from it.

The diagnosis of the current draft is not that its evidence is thin. It is that
the abstract introduces eleven objects in eleven sentences, so a reader's first
ninety seconds produce "a collection of experiments around a decomposition"
rather than one result. **The largest available score movement is editorial,
not experimental.**

---

## 2. The two numbers that must move to the front

### 2.1 The hook — currently sentence 2

`46` nominal dimensions carrying block-whitened Renyi-2 effective rank `2.00`.
Concrete, verifiable, counterintuitive. It should be the first thing after the
problem statement and it deserves a figure panel. This is the "huh" moment and
it is the only place the paper gets one cheaply.

### 2.2 The ace — currently one bland clause

The abstract now says "Frozen fixed-support interventions on OLMo and Qwen show
that allocation remains actionable after pretraining." The number behind that
clause is:

| Installed table (OLMo-2 1B, unseen-nine RULER @16K, 20 rows/task) | Macro |
| --- | ---: |
| same-support geometric | `0.0056` |
| derived allocation | **`0.6047`** |
| nearest movement-profile ramp | `0.6104` |

`derived - geometric = +0.5992`, paired row bootstrap `[+0.5488, +0.6480]`.
Owner: [`attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) §3.2.

This is a hundredfold effect with a tight interval, on a third-party released
checkpoint, zero training, zero learned parameters, and **both arms are our own
controls** — it is a controlled intervention, not a beat-the-baseline claim,
which makes it the hardest number in the paper to attack. It is currently
undersold relative to "432M MLA lowers 16K PPL by 31.1%", and a task score
going `0.006 -> 0.605` is far more visceral than a 31% PPL reduction.

**Action: promote both numbers into the abstract with their values, and give
§2.2 its own subsection in the experiments section.**

### 2.3 The defensive asset hiding inside 2.2

The coarse ramp scores `0.6104`, statistically indistinguishable from the
derived profile. Read defensively this pre-empts "did you tune the profile?" —
no, a coarse ramp does as well. **The axis matters; the fine shape does not.**
That is clear, memorable, and hard to attack. State it in our own voice rather
than leaving it in the appendix to be discovered.

---

## 3. The objection that must be pre-empted in the body

**R3.** Target-matched support reverses the direction: `+0.026 / +0.060 /
+0.227 / +0.460`, `0/3` seeds, monotone in length. It is currently in an
appendix. A reviewer who finds it unaided concludes "this only wins when the
baseline is untuned", and that reading is rejection-grade.

It cannot be left to be discovered. The answer already exists and turns it into
a strength: **support and allocation are interacting coordinates, not additive
ones.** The `0.0056 -> 0.6047` contrast is measured *on an already-moved
support*. So the honest and favourable framing is:

> Support decides whether the model functions at a length; allocation decides
> how well it functions there. Given the same moved support, geometric spacing
> scores `0.0056` and non-geometric `0.6047`.

That converts R3 from "your comparison was unfair" into "these are two
composing axes and we identify the second one". It is a section-ordering and
one-paragraph change.

---

## 4. Structure

Five steps, one line of causation, every existing evidence layer routed to
exactly one of them.

| Step | Claim | Owner |
| --- | --- | --- |
| 1 Problem | a third of the pairs are duplicate positional directions; rank `2.00` over 46 dims | full-RoPE report |
| 2 Tool | `x = a + Rz` separates support from allocation — **this is the novelty**, it is what makes allocation separately manipulable | causal-variables owner |
| 3 Cause | endpoints bitwise fixed, 30 interior frequencies move, 3/3 seeds improve every OOD length | exact-range three-seed |
| 4 Construction | EVQ-Cosh, closed form, zero learned parameters | research synthesis |
| 5 Consequence | scale and systems, then: on a released 1.5B at fixed support, `0.0056 -> 0.6047` with frozen weights | MLA / 454M / 750M / 1.485B / 8B, then same-support owner |

**Do not delete scale evidence — route it.** Nine evidence layers without a
throughline read as a kitchen sink; the same nine with one sentence each saying
what that layer tests read as thorough. The fix is routing, not subtraction.

**Zero-training is step 5, not an appendix.** It is the direct cash-out of the
step-2 decomposition on a mature model. Filing it as an "engineering fallback"
demotes the paper's strongest capability evidence. Per the mechanism analysis,
the frozen-table form is *motivated* by the co-adaptation result and the
transplant obstruction — not proven necessary by them — so it belongs on the
main line as the lowest-cost route that leaves model weights untouched.

---

## 5. What not to report

- **LongBench 2Wiki at 8K/16K from the recovery runs.** Both arms sit at
  `F1 ~ 0.075` with 176-178 of 200 rows at zero F1 and degenerate repetition
  output; Native support on ~7K prompts is a known-dead configuration. Do not
  report it, and equally do not stamp it "unresolved" in reviewer-facing text.
  Simply do not lean on it. The 4K cell (`-0.00654` F1) is real and stays
  internal as a direction signal.
- **"We beat YaRN 0.60 versus 0.08."** Official YaRN scoring `0.0794` at 16K
  invites scrutiny that costs more than the sentence earns. Keep the locked
  nomenclature separation and lead with our own same-support control instead.
- **Internal negatives, failed probes, and plans**, per `AGENTS.md` §2. This
  file and the mechanism analysis are where they live.

---

## 6. Experiment plan

### 6.1 Completed submission experiment

The 128-document fixed-support dose response is complete:
[`attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md`](attention-aware-retrofit/results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md).
The registered analytic Path A shows an interior 16K-tail minimum but no point
passes its joint `+0.01` 4K guard; the static-`r2` location prediction fails.
The empirical oracle direction gives a graded tail effect at small 4K cost,
but inherits one learned run and is mechanism evidence rather than a new
zero-training method. The experiment stays internal and does not replace the
stronger three-seed and same-support manuscript owners.

### 6.2 Before submission — no further compute

No new seeds. No larger models. No `tau`, band, amplitude, `p`, or routing
sweeps. No per-head arm. The dose and Native-4K diagnostic above close the
authorised window; do not add capability runs from the same evaluation pool.

The binding reason is not cost. **Tuning `p`, `c`, or the profile now, on
evaluation sets that have already been read, converts a clean controlled result
into a tuned one.** That is the only remaining way to damage the current
position. In particular the measured `c=0.12` improvement stays unadopted.

### 6.3 Completed diagnostic and corrected next question

Native 4K core-four is `1.00/0.85/0.60/0.03`. Because each length uses
different generated rows, and another frozen policy reaches VT `0.62` at 8K,
the proposed one-number capability-versus-position decision is invalid. The
result owner is
[`attention-aware-retrofit/results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md`](attention-aware-retrofit/results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md).

### 6.4 After submission

The next clean experiment reuses identical prompt content and decoding while
changing only position IDs/phase exposure. Only after that identification
should per-task, per-head, or per-layer allocation be trained. Amplitude and
profile values already inspected on the evaluation tasks remain frozen.

---

## 7. Answering the two questions this plan exists to settle

**"Can we validate capability on released models?"** It is already done and
under-claimed. OLMo-2-0425-1B-Instruct and Qwen2.5-1.5B are third-party
releases; core-4 RULER and official LongBench 2Wiki are capability endpoints,
not perplexity; the frozen operator scores `0.5825 / 0.4000` against Native
`0.0000 / 0.0000`, and 2Wiki stays within `+0.0023 / -0.0017` of the Native 4K
score. The gap is only that this is 1.5B rather than 70B, and no reviewer
expects a 70B ablation from an academic paper. The work needed is promotion,
not measurement.

**"Is zero-training finished?"** The current evidence is sufficient for the
paper's bounded identification and construction claim. The method frontier is
not closed: the analytic dose misses its in-window gate, and per-task headroom
cannot be localised without a matched-content phase intervention.

---

## Claim boundary

This file is not a numerical owner. Every number is quoted from the linked
owner and inherits that owner's scope.
