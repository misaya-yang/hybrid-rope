# ICLR 2027 submission narrative and experiment plan

- **Date:** 2026-08-26
- **Status:** plan; changes no number and no manuscript source by itself
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
| 5 Consequence | scale and systems, then: on a released 1.5B at fixed support, `0.0056 -> 0.6047`, zero training, downstream QA preserved | MLA / 454M / 750M / 1.485B / 8B, then same-support + length-conditioned |

**Do not delete scale evidence — route it.** Nine evidence layers without a
throughline read as a kitchen sink; the same nine with one sentence each saying
what that layer tests read as thorough. The fix is routing, not subtraction.

**Zero-training is step 5, not an appendix.** It is the direct cash-out of the
step-2 decomposition on a mature model. Filing it as an "engineering fallback"
demotes the paper's strongest capability evidence. Per the mechanism analysis,
the frozen-table form is *forced* by the co-adaptation result, so it belongs on
the main line.

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

### 6.1 Before submission — one experiment, thirty-five minutes

**The dose-response curve.** Native → anchored EVQ-Cosh at fixed support, on
the frozen released checkpoint, zero training. It is the figure that answers
R4 ("the theorem is decorative") by showing the theory's axis predicts
behaviour on a model we did not train.

Preregistration:
[`attention-aware-retrofit/preflights/ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md`](attention-aware-retrofit/preflights/ALLOCATION_DOSE_RESPONSE_PREFLIGHT_20260826.md).
Code and the sixteen frozen tables are already staged and CPU-verified.

```
bash scripts/eval/run_allocation_dose_grid_5090.sh screen
```

Reporting, not gating: full NLL, tail NLL, and the per-1024-position-bin
profile, per row. A flat in-window cost with a growing tail gain up to a knee
is the figure; anything else is recorded and the construction changes. **This
preflight carries no route-closing clause** — a marginal curve is a reason to
change the construction, never to close the axis.

Optional second cell, fifteen minutes: re-run the matched self-consistent
comparison on the existing 128-document holdout instead of the four-document
views, because that number is destined for the paper and the 128-row version
lands directly beside the published Native row `2.7538 / 7.0023 / 7.2703`.

### 6.2 Before submission — nothing else

No new seeds. No larger models. No `tau`, band, amplitude, `p`, or routing
sweeps. No per-head arm. No capability re-runs.

The binding reason is not cost. **Tuning `p`, `c`, or the profile now, on
evaluation sets that have already been read, converts a clean controlled result
into a tuned one.** That is the only remaining way to damage the current
position. In particular the measured `c=0.12` improvement stays unadopted.

### 6.3 Immediately after submission — the gating diagnostic

**Variable tracking at 4K, in-window, same checkpoint.** Ten minutes. It
decides whether the remaining headroom is positional or is a model ceiling, and
every later method choice depends on it. Rationale and the two readings:
mechanism analysis §6.

### 6.4 After that — headroom, conditional on §6.3

Ordered in the mechanism analysis §7: per-task profile optimisation, per-head /
per-layer allocation (`0.89` versus `0.09` repairable fraction, code complete,
never trained), position-profile repair, and only then the amplitude
coefficient.

---

## 7. Answering the two questions this plan exists to settle

**"Can we validate capability on industrial models?"** It is already done and
under-claimed. OLMo-2-0425-1B-Instruct and Qwen2.5-1.5B are third-party
releases; core-4 RULER and official LongBench 2Wiki are capability endpoints,
not perplexity; the frozen operator scores `0.5825 / 0.4000` against Native
`0.0000 / 0.0000`, and 2Wiki stays within `+0.0023 / -0.0017` of the Native 4K
score. The gap is only that this is 1.5B rather than 70B, and no reviewer
expects a 70B ablation from an academic paper. The work needed is promotion,
not measurement.

**"Is zero-training finished?"** For this paper, yes — the claim it supports is
identification and construction, and that claim is fully evidenced. For the
method, no: the per-task decomposition shows range is saturated while
long-range resolution is not, and the headroom is concentrated in multikey-3
and variable tracking. Those are §6.3 and §6.4, after submission.

---

## Claim boundary

This file contains no new measurement. Every number is quoted from the owner
named beside it and inherits that owner's scope. It is a plan, and a plan is
not a result.
