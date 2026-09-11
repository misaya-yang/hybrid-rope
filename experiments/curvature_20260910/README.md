# Frequency allocation as a constrained optimization — solved, not searched

Frozen Qwen2.5-3B-Instruct, MrRoPE as the base table, no weight updates.

## The claim, in one paragraph

Every method in this programme — YaRN's NTK-by-parts ramp, MrRoPE's quadratic
radial family, EVQ's companding curve — pre-specifies a *transport path* for the
64 log-frequencies and then reports how it did. None of them is derived from a
statement of what the table is for. The statement is available: keep the output
distribution on the distribution the checkpoint was trained for, and buy
long-range capacity with the budget that leaves. Written down, that is

    minimise   L_long(nu)                      the long-range objective
    s.t.       D_N(nu - nu_native) <= eps      native output drift

and its local form is a trust-region step in the native output-Fisher metric
with a closed-form solution. Nothing about it is a search, a scan, or a
hyper-parameter. The three bands are what the solution *produces*; they are not
an assumption it starts from.

The load-bearing number is `G = g^T P_F g`. It has exactly two readings:

| | meaning |
|---|---|
| `G ≈ 0` | the long-range gradient has no component the native metric can pay for. **MrRoPE is a KKT point of the stated problem** — the theory *explains* it, and no local step beats it at any affordable native price. |
| `G > 0` | there is a measured direction that buys long-range improvement at a pre-computable native cost. The theory *beats* it, by a predicted amount `sqrt(2 eps G)` fixed before the forwards are run. |

Both are results. Neither is a null outcome, and neither depends on a curve
search over a hyper-parameter.

**But the EXPLAINS reading is conditional on `eps`, and this is the one place the
formulation can quietly lie to itself.** `D_N` is measured *from the native
table*, so every real method — MrRoPE included — has a large `D_N`. If `eps` is
set far below `D_N(MrRoPE − native)`, then the feasible set does not contain
MrRoPE at all, and `G ≈ 0` would be asserting optimality about a point the
stated problem excludes. The claim is only meaningful at a budget MrRoPE could
afford, so `s1b` measures that budget and prints it:

    cost(MrRoPE) = D_N(MrRoPE − native)   <- the calibrated budget

`eps` itself stays pre-registered on the command line and is **not** re-chosen
after seeing anything; what `s1b` decides is which of the two readings a given
run is entitled to. At `eps < cost(MrRoPE)` the solve answers *"what is the best
direction at this budget"*, which is a legitimate question and the one a
deployment would actually ask, because a deployment does not have to pay
MrRoPE's native drift to get long-range ability. At `eps = cost(MrRoPE)` it also
answers the KKT question. Report both, and say which one the number is.

The budget-free half of the KKT reading is the residual `lambda_j = -g_Lj/g_Nj`
across the free slots: its *spread* is a property of the base point and does not
depend on `eps` at all. If MrRoPE is a KKT point the spread vanishes. That is
the test to lead with, because it cannot be tuned by a budget choice.

## Why this is not the same as the other unified accounts

MrRoPE's success is *inverted exactly*, not fitted. In the compression
coordinate `m_j = ln(omega_j / nu_j) / ln 4` (native `m=0`, one full compression
`m=1`), MrRoPE is `m_q = q(q+1)/(N(N+1))` with `q = clip(j-23, 0, 17)`. Two
facts follow arithmetically and are the whole content of "it is a KKT point":

* its increments along the bridge are **arithmetic in q**: `m_{q+1} - m_q =
  (q+1)/153`, so the log-frequency gap exceeds the native gap by exactly
  `ln4 * (q+1)/153`. The first one is `0.009060747`, and the deployed table
  gives `0.00906075` — agreement to eight digits, and to 6e-17 in m-units
  across the whole bridge;
* `N = 17` is not chosen by a search. It is the member of the family whose
  transition width equals YaRN's own `dim*ln(beta/alpha)/(2 ln theta) = 16.06`.
  And since `sum m = 40.6667 - (2/3)N`, `N = 17` is also the *smallest-budget*
  member — MrRoPE spends `29.333` where YaRN spends `30.104` over the same
  support with the same endpoints and the same `gain`.

That arithmetic is the explain branch arriving early. What the GPU work below
adds is whether the *residual* is zero — i.e. whether the spread of
`lambda_j = -g_Lj / g_Nj` across the free slots vanishes — and if it does not,
the spread itself is a measured direction to move in.

## Layout

| file | role | GPU |
|---|---|---|
| `tables.py` | the m-coordinate and every named family in it | no |
| `preflight.py` | stage 0: free checks that catch a wrong algebra, a drifted env, or a probe corpus that is also an evaluation corpus | no |
| `local_probe.py` | constraint side: output KL and the Fisher it implies | yes |
| `arms.py` | the constraint model vs the 16 arms the panel already scored, and the `E1_s28_less` veto test — free | no |
| `long_grad.py` | objective side: `d(long loss)/d(log freq)`, forward only | yes |
| `solve_kkt.py` | the closed-form step, and the explain-vs-beat judge | no |
| `forward_check.py` | real forwards on the step, and the pre-registered gate | yes |
| `panel_jobs.py` | queue the solved table on the existing frozen-weight harness: registration + RULER panel + 64K/128K `long_eval` | no |
| `driver.sh` | staged runner, gates wired in | — |
| `RUNBOOK.md` | the boot-day operational checklist: exact commands, what each stage should print, resuming, recording | — |

Run modules from the repository root: `python -m experiments.curvature_20260910.X`.

## How to run it

```bash
bash experiments/curvature_20260910/driver.sh s0        # free, no GPU — always first
bash experiments/curvature_20260910/driver.sh all       # s0..s5 with gates
bash experiments/curvature_20260910/driver.sh status    # what has run, and disk
```

`RUNBOOK.md` is the same thing with the boot-day detail: the deploy layout, what
each stage should print and what to do when it does not, how to resume, and what
must be recorded in each of the three outcome branches.

Every stage writes one file under `runs/` and is skipped if that file exists, so
an interrupted chain resumes for free. Delete the file to force a re-run.
`touch runs/STOP` halts at the next stage boundary.

Paths are environment-overridable (`MODEL`, `NPY`, `BM`, `HARNESS`, `TABLES`);
the defaults are the ones recorded in the archived runs. `driver.sh s0` prints
a check per assumption and names what each failure invalidates.

## Cost, from this host's own 640 archived rows

Measured on the same checkpoint and the same inputs: a 32K forward is 4.1 s
median, a 128K forward 33.9 s median (max 100.7 s), peak allocation 21–27 GiB.

| stage | what | forwards | estimated |
|---|---|---|---|
| s0 | preflight | 0 | seconds |
| s1 | Fisher diagonal, 64 slots @32K fp32 + 8 pairs + scaling check | ~90 @32K | 8–12 min |
| s1b | the constraint model vs the panel's own arms; no forwards | 0 | seconds |
| s2c | in-window gradient, `wide` @32K bf16 | 60 @32K | 5 min |
| s2a | long gradient, bridge @128K bf16 | 36 @128K | 20 min |
| s2b | long gradient, wide @128K bf16 | 60 @128K | 34 min |
| s3 | solve | 0 | seconds |
| s4 | trust sweep + 3 controls + screen | 12 @128K, 20 @32K | 12 min |
| s5 | queue the two panel jobs | 0 | seconds |
| harness A | registration: NLL 16 docs × 3 lengths + 36 RULER rows | ~48 + generation | ~25 min |
| harness B | `long_eval`: 4 docs × 2 lengths × 2 methods | 16 @64K/128K | ~15 min |

Core chain (s0–s4) is **under 90 minutes** of GPU. s2b is the only expensive
optional stage: `s2a` alone is enough to solve, and `s2b` widens the slot set
from 18 to 30. Skip it if the session is short — the driver falls back to the
bridge gradient automatically. The archived `long_eval` job took 177 s for
MrPro alone at 2 docs × 2 lengths; the paired run adds the candidate's rows.

Disk: probe receipts are kilobytes; the harness writes token-level NLL rows
(512 floats × 48 records ≈ 1 MB per candidate). The 21 GiB free on
`/root/autodl-tmp` is not a constraint for this work. *Model weights are not
touched, nothing is trained, and no checkpoint is written.*

## The protocol, and why it is in this order

**Reuse before re-measure.** The archived MrPro panel rows are the baseline.
`panel_jobs.py` reads the harness's own `tables.json` and copies that file's
gain, so the candidate differs from MrRoPE in the frequency table and nothing
else. Five.6-pro's constraint is honoured literally: **gain is fixed at
MrRoPE's value and is not co-optimised.** A candidate that quietly changed the
gain could not be read against the archived rows at all.

**Screen target-length PPL before releasing a downstream test.** That is the
standing protocol in `experiments/nongeometric_screen/README.md`, and s4
implements it: tail-token NLL at 8K/32K/64K/128K for the candidate against the
base table in the same model load.

**The harness has two dispatch paths, and the candidate needs both.** Verified
against the harness source, not assumed:

| path | inputs | lengths | what it gives |
|---|---|---|---|
| `action: evaluate` | `prepared_nll_01` (32769 tok/doc) | 8192 / 16384 / 32768 | RULER panel + NLL comparable to the archived `run_nll_01/MrPro.jsonl`; writes `results/<name>/contract.json` |
| `module: long_eval` | `long_inputs` (131073 tok/doc) | 65536 / 131072 | the actual extrapolation measurement, paired against MrPro in one load |

`long_eval` reads each method's spec from `results/<method>/contract.json`, so
the registration job must run first. The worker takes
`sorted(queue/*.json)[0]`, and `panel_jobs.py` names its outputs so the
registration job sorts first — it raises rather than write a pair that would run
out of order. **A 128K request on the `evaluate` path would silently truncate**:
`data[:131072]` on a 32769-token document returns the whole document and
`data[130561:131073]` returns an empty tail. The measured length is stated in
every row so a truncation cannot pass unnoticed, but the fix is to use the right
corpus, not to check afterwards.

**The probe corpus is held out from the panel.** The panel's `long_eval` takes
the first `docs_per_dataset=2` documents per source — `proofpile` 001364/001901
and `pg19` 28988/30312. The probes run on `pg19_test_37702`, which is not among
them. A gradient measured on the evaluation documents would be fitting the test
set one forward at a time.

Archived MrPro long baseline (from `done/022_long_nll_baseline.json`, n=2 per
cell, tail-512 NLL):

| | 65536 | 131072 |
|---|---|---|
| pg19 | 2.5698 | **2.3077** |
| proofpile | 0.6222 | **1.1030** |

**Pre-register the number.** `eps`, the native output-KL budget, is set on the
command line **before** the long gradient is measured. The predicted gain
`sqrt(2 eps G)` is then a prediction, and `rho_trust` is a falsification test
rather than a summary of what happened. `--eps-sweep` exists and is marked
descriptive-only for the reason that a budget chosen after seeing the curve is
not a budget.

**One delta for everything.** `d*` is a single number per slot. There is no
schedule, no per-layer variant, no second hyper-parameter to tune — because
there is no second hyper-parameter in the problem.

## The gate, and what each failure means

`s4` computes `rho_trust = (measured long-loss drop) / (predicted drop)` and
requires `0.5 <= rho_trust <= 1.5`, plus a fitted native-KL exponent in
`[1.7, 2.3]`. **If it fails, everything downstream stops.** That is not a
judgement call made after seeing the numbers; it is the stopping rule
five.6-pro stated and it is wired into `driver.sh`.

Attribution, in the order to check it:

| symptom | diagnosis | what to change |
|---|---|---|
| KL exponent ≪ 2 | the probes sit outside the region where `F_N` describes the model — cubic terms or forward rounding dominate | shrink `eps`; if it is still flat at `1e-4`, the probe resolution is the limit, not the theory — report `delta`-independence as the finding |
| KL exponent ≫ 2 | the response is below the forward's rounding floor at small alpha (the measured KL is noise) | raise `--delta` in `local_probe` and re-check with `fisher_scaling`; the diagnosed "free" slots in `solve_kkt` are the ones where this happened |
| rho low, exponent fine | `g_L` is over-stated — the long loss is not linear over the step | re-measure `long_grad` without `--one-sided`, and read its `curvature` field: the second difference is the local model's own error bar |
| rho high | `g_L` under-stated, or the baseline loss is off | check `base_loss` in the `long_grad` receipt against the archived MrRoPE tail NLL for the same doc; a mismatch is an input-identity problem, not a theory problem |
| solver does not beat the controls | direction carries no information beyond the budget | **this is a positive finding.** Budget alone determines the outcome, which means every geometric family is a budget choice wearing a shape, and the band structure is a consequence of the endpoints, not of the interior profile |
| `G <= 0` (degenerate receipt) | no payable direction exists at this budget | the **EXPLAINS** branch. Write it up; do not force a step |

## Contingencies

Every stage has a cheaper fallback and a sharper follow-up, chosen so that the
session ends with something absorbable in every branch.

**If `s1` OOMs in fp32 at 32K.** Re-run with `--dtype bf16`; the bf16 rounding
floor is why fp32 is the default, so record the dtype in the receipt and treat
any negative `F_jj` as "not priced" rather than "free". `solve_kkt` already
pins unpriced slots by default.

**If the long corpus is shorter than 131072.** `preflight` reports it and
`--screen-lengths` drops the longest entry. Run at the longest length available
and record the shortening in the receipt — a 64K measurement is still a
measurement, and the protocol's target-length screen is defined relative to
whatever length the baseline rows exist at.

**If `G > 0` but `rho_trust` is at the bottom of the band (0.5–0.7).** The
direction is right and the magnitude is over-stated. Apply a `0.6x` shrink and
ship that: the *ordering* of slots is what the theory contributes, and the
linear model's scale error is a known, bounded defect, not a refutation.

**If `step_diag_share` is low** (the full-matrix run only): the answer depends on
cross-frequency coupling that a per-slot benefit/cost reading cannot see. That
is the concrete, quotable reason to prefer this solver over every ratio-based
allocation rule, and it is worth its own measurement — run `s6` for the exact
64×64 `F_N` and re-solve with `--fisher` pointing at it.

**If the panel result is flat against MrRoPE.** Then the local model is right
and the *objective* is the wrong one: `g_L` was measured against tail-token
NLL, and the downstream panel measures retrieval. The follow-up is
`long_grad --long-loss gold`, which scores the binding answer span instead of
the average token. Same machinery, one flag, and it is the honest test of
whether the two objectives agree.

**If a solved table stops being monotone.** `solve_kkt` reports it rather than
clipping it, `panel_jobs` refuses to queue it. A non-monotone table is still a
legal rotation grid but is no longer a frequency ordering, and every band
statement about it becomes unreadable. Re-solve with the ordering active — that
is a different problem (an inequality-constrained KKT system) and the receipt
will say which constraint bit.

## The vetoed route this package is adjacent to, and why it is not that route

This has to be stated plainly rather than left implicit, because the shape is
close enough to be dangerous. The corpus carries an explicit dead zone:

> **V-E2** 重启 18 样本/64 自由度的 **margin-gradient 能力优化路线** — 已失败，"不能据此
> 重新启动"；**共享频率响应工具只作局部诊断**。  (`REVIEW-0907 行 178`)

and the review that drew it says exactly where the line is:

> 该量只描述冻结 hidden-state 框架中的**局部块响应**；**它不是整网 LM 风险的梯度**，
> 不能据此重新启动历史上已失败的 18 样本/64 自由度 margin-gradient 路线。

This package is 64 free coordinates and it is a gradient. It is not the vetoed
route, and the distinction is the one that sentence draws:

| | the vetoed route | this package |
|---|---|---|
| what is differentiated | a **local block response** in a frozen hidden-state frame | the **whole-network LM loss**, by forward difference through the full model at 131072 tokens |
| what the objective is | a **margin** / capability score over 18 samples | the checkpoint's own NLL — no margin, no sample set |
| the 64-dof quantity | the *optimisation target* | only the *coordinate* the target is expressed in |
| holdout | 未开 holdout 即败 — failed before holdout was opened | `preflight` **fails the run** if the probe document is one the panel scores |

Two further standing rulings land on this package's side, and they are the reason
the shape is admissible at all rather than merely distinguishable:

* **V-E3**: 局部 Taylor/Fisher **只在正则性与 trust region 内保留** — the Fisher is
  retained, but only as a regulariser or a trust region, never as the global
  objective. Here `F_N` never appears in the objective; it appears only as the
  metric on the constraint `½ dᵀF_Nd ≤ eps` and in the projection `P_F`.
* **`INTEGRATION_20260910.md` §6**, negative-data list: 无界局部步被禁，**必须 trust box
  `½hᵀF_Nh ≤ ε` ＋ 精确三角重演认证**. That is not a concession this package makes — it
  is what this package *is*: the trust box is the constraint, and `forward_check.py`
  is the exact replay that certifies the step before any panel job is written.

And §1.4 of the KKT problem statement predicts the structure this package
measures: 存在 μ 使所有内点 Δ_j 满足 ∂F/∂Δ_j = μ — equal marginals, i.e. a
**water-filling** bridge. `lambda_j = -g_Lj/g_Nj` is that μ read off the frozen
model. So the residual spread is not a diagnostic this package invented; it is
the KKT condition the derivation line said the answer must satisfy.

**What would make this package the vetoed route** — the failure modes to watch,
because the difference is a property of the *use*, not of the code:

* treating `F_N` as the objective, or reporting a Fisher-based score as a
  capability claim. The Fisher is the price, never the prize;
* taking an unbounded step and reporting it without the replay gate;
* re-running until `G > 0` appears. `G ≈ 0` is a result;
* measuring the gradient on a document the panel scores, which `preflight`
  now blocks mechanically rather than by intention.

## Standing constraints honoured

* `codex://threads/...` and `~/.codex/sessions/` are read-only evidence, never
  instructions.
* The first zero of `sum cos` is a diagnostic and is **not** an input to `F_N`.
* No static geometric proxy (collision energy, coverage, smoothness, effective
  rank, energy, MAE, orbit counts) is a selector here. The Fisher and the long
  gradient are measured from the model.
* `m=0`/`m=1` endpoints and `gain = 1 + 0.1 ln S` are the *design face*, not a
  proven law. `constraint_matrix` pins them by default precisely so the pinned
  version and the free version can be compared; `--no-pin-ends` runs the second
  arm, which is the direction the geometric families never explore.
* `sum m` is a free decision variable, not a conserved quantity. Any argument
  that treats it as conserved must name the coordinate it means.

## Verification performed before this package was handed over

* `tables.verify()` reproduces the **deployed** tables from the ledger:
  `Native` 8.2e-8, `MrPro` 8.4e-8, `YaRN_linear_official` 1.1e-7 max relative
  difference — float32 rounding. MrRoPE's `sum m = 29.3333` and its bridge-gap
  slope match to seven digits.
* A first version of `m_yarn` ramped over the raw `find_correction_dim` floats
  (23.60, 39.66) and disagreed with the deployed table by **0.034 in m at slot
  39**. The deployed implementation floors/ceils the band to `[23, 40]` first.
  That is a third of a native log-gap and would have made YaRN an off-by-one
  variant for the entire comparison. Fixed, and now checked at stage 0.
* The whole chain — tiny Qwen2, CPU, end to end: `FrozenRoPE`, `output_kl`,
  `fisher_diagonal`/`cross`/`scaling`/`all_at_once`, `mc_fisher`, `long_grad`,
  `solve_kkt`, `forward_check` (including gate evaluation and budget-matched
  controls), and `panel_jobs` against a synthetic harness tree. Three real bugs
  were found this way and fixed: a float32 log round trip that returned
  **negative Fisher diagonals**, a detached autograd graph in `mc_fisher`, and
  `base.m` attribute access on what is a dict.
* The job contract was checked against the **live harness on the server**, not
  against the queue files that were already understood to be one shape. Two
  things came out of that and both were wrong in the first version of this
  package: (i) the 128K path is `module: long_eval` reading `long_inputs/`, while
  the `spec`-shaped path caps out at 32768 because `prepared_nll_01` documents
  are 32769 tokens — a 131072 request there truncates silently and scores an
  empty tail; (ii) the emitted `long_eval` job now matches
  `done/022_long_nll_baseline.json` field for field. The probe corpus was also
  moved to a document the panel does not score on, and stage 0 now fails if it
  is one.

## What this does not claim

* No independent confirmation: the panel is the historical development set and
  the receipts say so.
* No causal band attribution. The step moves 18–30 slots at once; the solver
  says *where* budget should go, not that band *j* is *the* cause of any
  observed change.
* No claim that the frozen-weight optimum transfers to training. `m` is the
  compression relative to the trained grid; a from-scratch arm would re-derive
  it, and the two are different problems.
