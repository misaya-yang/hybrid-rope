# Why a frozen table is the minimal zero-training intervention

- **Date:** 2026-08-26
- **Status:** internal mechanism analysis, updated after the registered dose
  response and Native-4K diagnostic
- **Role:** explains why a frozen-table intervention is attractive, records
  two design hypotheses, and separates measured headroom from speculation
- **Not:** a manuscript claim, a method proposal, or an action queue

Every number below is owned elsewhere and cited to its owner. The two
quantities computed during this analysis are marked **[computed 2026-08-26]**
and carry their derivation inline.

---

## 0. Summary

The frozen-table route is the smallest zero-training intervention because it
changes the positional code without changing model weights. Co-adaptation and
the exact transplant obstruction motivate that choice but do not prove it is
the only possible retrofit. Sections 2 and 3 therefore state design hypotheses,
not universal constraints. The completed 4K diagnostic in Section 6 also shows
that a cross-length task score cannot by itself separate model capability from
position coding.

---

## 1. Why start with a table change

For one head, the RoPE attention logit between query `i` and key `j` is

```
z_ij = sum_k a_k cos(theta_k + omega_k * d),    d = i - j
```

where `a_k, theta_k` come from the Q/K weights and `omega_k` from the rotary
table. Content lives in `theta`; position lives in `omega * d`.

**Fact 1 — the weights are co-adapted to one specific table.** Swapping the
table under frozen weights moves 50M PPL `7.14 -> 76.20`; the reverse cell is
`23.05 -> 7.16`
([`../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)).
The 151.9M two-seed crossing replicates it: FMRoPE-trained weights prefer their
own derived table `3.426` versus `5.776`, Cosh-trained weights prefer theirs
`3.479` versus `4.455`
([`../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) §4).

**Fact 2 — beyond the training length two distinct failures start.** Pairs with
`omega * L < 2*pi` never completed a turn during training and enter phases the
weights have never seen. Pairs that already wrapped begin to alias: lag `d` and
lag `d + 2*pi/omega` become indistinguishable. Both are properties of `omega`,
not of the weights.

**Fact 3 — exact post-hoc Q/K compensation is obstructed.** For unequal
frequency multisets there is no fixed invertible Q/K map preserving all
relative-position logits
([`../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`](../../../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md)).

Together these **motivate** the design, and the strength of that word matters.
The failure is a property of `omega`, so it is repairable by changing `omega`
or by changing `theta`. Changing `theta` means retraining, which by Fact 1
perturbs the co-adaptation the released checkpoint was built on, and by Fact 3
cannot afterwards be repaired by any *fixed invertible linear* Q/K map.
Changing `omega` costs nothing and touches no weight.

**This is an argument for preferring the frozen-table route, not a proof that
it is the only one.** Fact 3 obstructs exact post-hoc linear compensation for
unequal frequency multisets; it says nothing against approximate retraining or
against operators outside that family, and `AGENTS.md`'s frozen-retrofit claim
ceiling states exactly that boundary. Any sentence of the form "the
intervention *must* be a pure table change" exceeds the theorem and must not
enter the manuscript. What the evidence supports is the weaker and still useful
claim: the frozen-table route avoids modifying the learned weights, and no
retraining route has yet established the same downstream-retention outcome at
comparable cost on this checkpoint.

This may help explain why NLL and capability dissociate under adaptation: NLL
is token averaged, whereas multi-hop retrieval and instruction following can
depend on narrow learned circuits. That is a mechanism hypothesis, not a
consequence of the obstruction theorem, and a frozen table does not guarantee
capability preservation.

---

## 2. Design constraint A — preserve the relative-code symmetry

`z_ij` depends on `i` and `j` only through `d = i - j`. The Q/K weights encode
content phases that are meaningful only inside that relative code. An operator
that makes the phase depend on absolute position destroys it: for a pair
straddling a boundary, `theta(i) - theta(j)` is no longer a function of
`i - j`, so the same physical lag maps to different code values depending on
where the pair sits.

**Measured counterexample.** The target-free continuous-boundary-slope operator
in [`../../../../scripts/lib/rope/target_free.py`](../../../../scripts/lib/rope/target_free.py)
keeps Native phase up to `L_native` and continues at reduced slope beyond it.
It preserves the window exactly and needs no target length, so on paper it is
the ideal operator. It scores **`0.0000` core-4 RULER macro at both 8K and
16K**, against `0.7175 / 0.4075` for the routed frozen-table policy on the same
data and harness (raw owners
`iclr_next_runs/target_free_20260823/ruler_smoke_target_free_{8k,16k}` and
`.../ruler_smoke_session_binary_s4_{8k,16k}`, read 2026-08-25).

This falsifies the tested position-dependent continuation on this checkpoint
and harness. Translation-symmetry breaking is a plausible mechanism, but one
failed operator does not prove that every absolute-position-dependent operator
must fail.

---

## 3. Design prior B — protect the fast band

Pairs with `omega * L >> 2*pi` are phase-saturated inside the training window,
while pairs with `omega * L <= 1` form the measured redundant
block: 23 of 64 pairs occupy 46 nominal dimensions at block-whitened Renyi-2
effective rank `2.00` ([`../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)).

This motivates holding the fast endpoint and spending most movement on slower
bands. It does not prove that fast channels never move in an optimum, nor that
the slow block carries no content. A deployment factor supplies one practical
horizon; target-free construction remains an open objective.

The measured decomposition confirms the shape is load-bearing on both parts:
frequency-only scores `0.4000 / 0.1150` at 8K/16K, amplitude-only leaves Native
at `0.0000 / 0.0000`, and the joint operator reaches `0.5825 / 0.4000`
([`../results/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md`](../results/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md)).

---

## 4. What has been bought, and what has not

The macro score hides the structure. Per task, for the frozen operator (same
owner as above, 20 rows per task):

| length | single-key | multikey-2 | multikey-3 | variable tracking |
| --- | ---: | ---: | ---: | ---: |
| 8K | **1.00** | 0.80 | 0.50 | **0.03** |
| 16K | **1.00** | 0.55 | **0.00** | **0.05** |

Read this as three separate statements.

1. **Single-key retrieval is saturated on these rows.** It reaches `1.00` at
   both tested lengths, so this task contributes no visible headroom to the
   reported macro.
2. **Long-range resolution is not.** multikey-3 collapses to `0.00` at 16K and
   variable tracking sits at `0.03` at 8K — *at the same length where
   single-key is perfect*. Finding one distant item works; discriminating among
   several does not.
3. **The macro is therefore a poor optimisation target.** Roughly a quarter of
   it is a saturated task. A profile change that improves multikey-3 can be
   invisible in the macro.

Statement 3 also reinterprets an existing negative. The same-support study found
the coarse fixed-index ramp indistinguishable from the derived profile
(`0.6104` versus `0.6047`, interval `[-0.0464, +0.0345]`) and concluded that
profile detail is not identified. That conclusion is correct **for the macro**.
It has never been tested per task, and the tasks where the operator actually
fails are exactly the ones the macro dilutes.

Given a support move, interior allocation can strongly affect the resulting
score: at fixed `(a, R)`, amplitude, checkpoint and rows, same-support
geometric scores `0.0056` and the derived allocation `0.6047` at OLMo 16K,
interval `[+0.5488, +0.6480]`
([`../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) §3.2).
The macro contrast does not localise that effect to multikey-3 or variable
tracking; a per-task fixed-support comparison would be required.

---

## 5. Historical four-document position decomposition

The registered 128-document dose result now supersedes this four-document
calculation for outward use; see
[`../results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md`](../results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md).
It confirms graded full-versus-tail redistribution but rejects static `r2` as
a selector for the useful dose. The calculation below is retained only as the
hypothesis that motivated per-position logging.

The 2026-08-25 co-adaptive oracle gives a second, independent view of the same
axis at a very small displacement (`max|dz| = 0.0012975`). Its matched
self-consistent contrast is owned by
[`../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`](../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md):
`+0.00098` held-out 4K, `+0.03844 / -0.03866` at 8K full/tail, and
`+0.02190 / -0.08767` at 16K full/tail.

Because attention is causal, positions `0..4095` of an 8K forward pass are the
identical computation to a 4K forward pass on the same document. Assuming the
delta over that first segment equals the measured 4K delta, the full-sequence
mean decomposes as:

| L | positions 0-4095 | middle band | final 512 |
| --- | ---: | ---: | ---: |
| 8K | `+0.001` | **`+0.092`** | `-0.039` |
| 16K | `+0.001` | **`+0.034`** | `-0.088` |

The conclusion is insensitive to the assumption: raising the first-segment
delta twentyfold to `+0.02` still leaves the 8K middle band at `+0.071`.

Per-row signs, extracted from the raw run on 2026-08-26, are `4/4` positive on
full and `4/4` negative on tail at both lengths — eight of eight, under two
independently trained adapters.

**So the effect is not a smooth "full traded for tail". It changes sign with
position: a penalty band just past the native window, then a gain in the far
tail.** The phase-shell attribution shows the same non-monotonicity through a
completely different protocol (table-only deltas `+0.078 / -0.213 / +0.273` at
offsets `L / 3L / 15L`).

Two protocols suggested that the allocation effect could be non-monotone in
lag. The evaluator in
[`../../../../scripts/eval/eval_allocation_dose_grid.py`](../../../../scripts/eval/eval_allocation_dose_grid.py)
emits per-1024-position-bin NLL; the completed dose run owns those measurements.

**Scope.** Four documents per length. Reproducible across runs is not the same
as generalising across documents.

---

## 6. Native 4K diagnostic: useful result, invalid binary test

Variable tracking is `0.03` at 8K while single-key is `1.00`. Two readings are
consistent with that:

- **positional** — chained dereference needs sharper long-range attention than
  single retrieval, and the table does not yet provide it. Headroom is real and
  large.
- **capability** — a 1.5B model cannot do chained dereference at any length.
  No table will fix it, and the operator is closer to its ceiling than the
  macro suggests.

The diagnostic is complete. Native 4K scores are `1.00/0.85/0.60/0.03` for
single-key, multikey-2, multikey-3, and variable tracking; see
[`../results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md`](../results/NATIVE_4K_RULER_DIAGNOSTIC_RESULT_20260826.md).
Low VT at 4K does not prove a model ceiling: different nominal lengths use
different generated rows, and an existing frozen policy reaches `0.62` on VT
at 8K. The proposed one-number decision rule is therefore rejected. A clean
test must hold token content fixed and change only phase/position exposure.

---

## 7. Headroom inventory

Ordered by expected value, all **post-submission** (see the submission plan for
why none of it may be touched before then).

| # | Direction | Why it might pay | Status |
| --- | --- | --- | --- |
| 1 | matched-content phase shift | identical prompts and decoding, only position IDs/phases change; this is the missing capability-versus-position identification | not yet designed |
| 2 | per-task fixed-support comparison | tests whether the macro hides profile differences on multikey tasks | blocked on #1 |
| 3 | per-head / per-layer allocation | `0.89` versus `0.09` repairable fraction at equal parameter count ([`RETROFIT_AXIS_FALSIFICATION_20260822.md`](RETROFIT_AXIS_FALSIFICATION_20260822.md) §5); code complete, never trained (`heterogeneous_rope_5090`) | never run |
| 4 | amplitude coefficient | `c=0.12` measured better than the frozen `c=0.10` at both lengths (`0.5850/0.4275` versus `0.5825/0.4000`) | **must not be adopted** — see below |

On #4: `c=0.10` was frozen as the matched reference value and was *not*
selected on these tasks. Adopting `0.12` after reading the scores converts a
controlled result into a tuned one. The measured value stays in the sensitivity
row. Discipline and strategy agree here.

---

## Claim boundary

One released 1.485B checkpoint and one 1.5B Qwen checkpoint, deterministic task
and document subsets, row bootstraps conditioning on those. The §5 position
decomposition rests on four documents per length and one stated assumption.
Nothing here changes the three-seed fixed-support identification owner, and
none of it is a manuscript claim.
