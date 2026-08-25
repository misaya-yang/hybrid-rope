# Why zero-training is the necessary form, and where its ceiling is

- **Date:** 2026-08-26
- **Status:** internal mechanism analysis; no new experiment, no new number of
  its own except the two derivations marked as computed here
- **Role:** explains *why* the frozen-table route works, states the two hard
  constraints any mature-checkpoint operator must satisfy, and locates the
  remaining headroom by task rather than by macro score
- **Not:** a manuscript claim, a method proposal, or an action queue

Every number below is owned elsewhere and cited to its owner. The two
quantities computed during this analysis are marked **[computed 2026-08-26]**
and carry their derivation inline.

---

## 0. Summary

The frozen-table route is not a fallback that happened to work. It is the only
form the intervention can take, and that follows from three facts the
repository already owns. Section 1 derives it. Sections 2 and 3 state the two
constraints that kill every operator which violates them, each with a measured
counterexample. Section 4 shows, from the per-task RULER decomposition, that
the current operator has bought **range** and has not bought **long-range
resolution**, and that the remaining headroom is concentrated in exactly two
tasks. Section 5 gives the one cheap diagnostic that decides whether that
headroom is real or is a model-capability ceiling.

---

## 1. Why the intervention must be a table change

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

Together these force the design. The failure is a property of `omega`, so it is
repairable by changing `omega` or by changing `theta`. Changing `theta` means
retraining, which by Fact 1 dissolves the co-adaptation the released checkpoint
was built on, and by Fact 3 cannot be repaired exactly afterwards. Changing
`omega` costs nothing and touches no weight. **Therefore the intervention must
be a pure frequency-table change, and the method's shape is a consequence of
the co-adaptation result, not an engineering convenience.**

This is also why NLL and capability dissociate under adaptation. NLL is a
token-averaged smooth quantity that a small adapter restores quickly;
multi-hop retrieval and instruction following live in the co-adapted structure
and do not. The frozen route preserves capability *by construction* because it
never perturbs that structure.

---

## 2. Constraint A — the relative code must stay translation invariant

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

This is the sharpest negative in the mature-checkpoint line and it is currently
unrouted. It belongs in the closed-route ledger: **position-dependent phase
continuation on a frozen checkpoint is falsified, with a first-principles
reason.**

---

## 3. Constraint B — the fast band must not move

Pairs with `omega * L >> 2*pi` carry all in-window discrimination and are
already phase-saturated; they have no aliasing problem to fix. Moving them buys
nothing and costs the window. Pairs with `omega * L <= 1` are the redundant
block: 23 of 64 pairs occupy 46 nominal dimensions at block-whitened Renyi-2
effective rank `2.00` ([`../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)).

Any admissible operator therefore has the same shape: **hold the fast endpoint,
move only the redundant slow block, and move it just far enough not to alias
out to the deployment horizon.** "Just far enough" needs the horizon, which is
why a factor appears and why the request-length branch appears. That is
information the operator genuinely requires; it is not a defect of the
construction.

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

1. **Range is solved.** Single-key retrieval is saturated at both lengths.
   Nothing on the frequency axis can improve it.
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

Given a support move, interior allocation is the variable that governs
resolution: at fixed `(a, R)`, amplitude, checkpoint and rows, same-support
geometric scores `0.0056` and the derived allocation `0.6047` at OLMo 16K,
interval `[+0.5488, +0.6480]`
([`../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) §3.2).
So the remaining headroom is on the allocation axis, and it is located at
multikey-3 and variable tracking.

---

## 5. The allocation effect is non-monotone in position **[computed 2026-08-26]**

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

Two protocols independently indicate that the allocation effect is a
non-monotone function of lag. No owner currently measures that profile
directly. The evaluator in
[`../../../../scripts/eval/eval_allocation_dose_grid.py`](../../../../scripts/eval/eval_allocation_dose_grid.py)
emits per-1024-position-bin NLL from the same forward pass, so the profile
costs nothing beyond a run that is already planned.

**Scope.** Four documents per length. Reproducible across runs is not the same
as generalising across documents.

---

## 6. The one diagnostic that decides whether the headroom is real

Variable tracking is `0.03` at 8K while single-key is `1.00`. Two readings are
consistent with that:

- **positional** — chained dereference needs sharper long-range attention than
  single retrieval, and the table does not yet provide it. Headroom is real and
  large.
- **capability** — a 1.5B model cannot do chained dereference at any length.
  No table will fix it, and the operator is closer to its ceiling than the
  macro suggests.

These are separated by one number: **variable tracking at 4K, in-window, on the
same checkpoint.** If it is also near `0.05`, the ceiling is the model. If it is
`0.4+`, positional encoding is eating it.

The RULER data already contains the `L4096` cell
(`iclr_next_runs/far_pass_chord_20260821/data/ruler_eval_core4_v3_s20260822_n20/L4096/`).
The smoke harness restricts `allowed_lengths` to `{2L, 4L}` and would need that
restriction relaxed for the in-window cell. This is the cheapest decision-
relevant measurement available on this line and it gates every later method
choice.

---

## 7. Headroom inventory

Ordered by expected value, all **post-submission** (see the submission plan for
why none of it may be touched before then).

| # | Direction | Why it might pay | Status |
| --- | --- | --- | --- |
| 1 | per-task profile optimisation | the macro is diluted by a saturated task; profile detail has only ever been tested on the macro | blocked on §6 |
| 2 | per-head / per-layer allocation | `0.89` versus `0.09` repairable fraction at equal parameter count ([`RETROFIT_AXIS_FALSIFICATION_20260822.md`](RETROFIT_AXIS_FALSIFICATION_20260822.md) §5); code complete, never trained (`heterogeneous_rope_5090`) | never run |
| 3 | position-profile repair | the `+0.09` near-OOD penalty band in §5 is unexamined; if the deployed operator has a similar band it is a concrete failure mode | new |
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
