# KKT analytic solution of the frequency-allocation problem

**Date:** 2026-09-10 · **Author:** recon agent (kkt-derivation) · **Status:** derivation complete, all numbers machine-computed
**Companion code:** `/tmp/kkt_verify.py`, `/tmp/kkt_verify3.py` (runs 1–3 below); repo module `experiments/curvature_20260910/tables.py`.

Every identity marked VERIFIED was checked numerically with numpy to machine precision (stated) or with exact `Fraction` arithmetic. Every number in this report is one I computed, not one I recalled. Provenance labels at the end follow LESSONS.md L1–L4 discipline.

---

## 0. Objects and notation

Slot `j = 0..63`, native frequency `omega_j = theta^(-2j/128)`, deployed `nu_j = omega_j * S^(-m_j)` with `S = 4`. Band `[lo, hi]`, `n = hi - lo`; increments `eps_k = m_{lo+k} - m_{lo+k-1}`, `k = 1..n`, with `eps_0 = eps_{n+1} = 0` (both sides of the band untouched) and `sum_k eps_k = 1` (the band spans exactly one radix compression). `L_D` = tridiagonal Dirichlet Laplacian (2 on diagonal, -1 off), acting on the interior nodes `k = 1..n`. `U` = lower-triangular cumulative-sum matrix, `(U eps)_q = sum_{k<=q} eps_k = m_q`.

- **MrRoPE:** `eps_k = 2k/(n(n+1))` (m quadratic: `m_q = q(q+1)/(n(n+1))`).
- **BM:** `eps_k = 6k(n+1-k)/(n(n+1)(n+2))` (m cubic: `m_q = q(q+1)(3n+2-2q)/(n(n+1)(n+2))`).
- Qwen band `[23,40]`, `n = 17`; OLMo band `[14,32]`, `n = 18`.

---

## 1. The constraint side: is the Dirichlet energy a damage model?

### 1.1 What first principles give

The only first-principles damage model available is the **output-KL / Fisher form**: to second order in the frequency perturbation, the in-window drift of the frozen checkpoint is

```
D(delta m) = (1/2) sum_j w_j (delta log nu_j)^2 = (ln S)^2/2 * sum_j w_j (delta m_j)^2,
```

`w_j` = the diagonal output-Fisher weight of slot `j` (the quadratic form `solve_kkt.py` already uses, `D_N(delta) = (1/2) delta^T F_N delta`). Rewritten on increments inside the band:

```
D(eps) = (1/2) eps^T K_W eps,   K_W = U^T diag(w) U,
(K_W)_{ij} = sum_{q >= max(i,j)} w_q      (tail sums),
```

so **for any weights `w >= 0`, `K_W` depends on `(i,j)` only through `max(i,j)`** — in particular `(K_W)_{12} = (K_W)_{22}`. VERIFIED numerically for a random positive `w` (n = 17). This is the entire content of the damage axis; everything below compares the BM objective against it.

### 1.2 The Dirichlet energy is NOT a Fisher damage form

**PROVEN.** `L_D` has `(L_D)_{12} = -1` and `(L_D)_{22} = 2`, violating the `max(i,j)`-only structure that *every* nonnegative weight profile imposes. No choice of slot sensitivities `w_j >= 0` makes the BM objective `sum_i (eps_{i+1} - eps_i)^2 = eps^T L_D eps` equal to an output-KL damage budget. The two quadratic forms are different objects.

### 1.3 The flat-sensitivity special case: the Dirichlet form is the *inverse* metric

For flat weights `W = I`, `K_flat = U^T U` (Brownian-covariance kernel, entries `n+1-max(i,j)`).

- **VERIFIED** (n = 17 and 18, max diff ~1e-14): `K_flat^{-1}` = the second-difference operator with diagonal `(1, 2, ..., 2)` and off-diagonal `-1` — i.e. the Dirichlet energy *up to one boundary corner*. The smoothness penalty appears in the Fisher machinery only as the **inverse precision of the flat model**, not as its damage.
- **VERIFIED** (`|K_flat^{-1} 1 - e_n| ~ 1e-15`): the flat-damage minimizer subject to `sum eps = 1`, `eps >= 0` is the **extreme back-loaded step** `eps = e_n` — all compression in the last band slot. PROVEN independently: `eps^T K eps = ||U eps||^2 >= (U eps)_n^2 = 1 = ||U e_n||^2`.
- Flat damage ranks the b-family **oppositely to the BM prior** (computed, span fixed):

| table | flat damage Σ m_q² (n=17) | (n=18) |
|---|---|---|
| step e_n | **1.000** | 1.000 |
| b=0 (MrRoPE) | 4.015 | 4.214 |
| b=0.25 | 4.715 | 4.961 |
| b=0.5 | 5.411 | 5.703 |
| b=1 (BM) | 6.705 | 7.075 |
| b=2 | 8.737 | 9.223 |

BM costs 1.67× (n=17) / 1.68× (n=18) the flat damage of MrRoPE for the same span.

### 1.4 Honest verdict

The smoothness penalty on the log-frequency profile has **no derivation from damage first principles**. It is a shape prior — the minimum-curvature completion of the transition. Its one defensible reading: among all monotone profiles with the same band, endpoints and span, BM is the completion the flat-sensitivity damage model can *least sharply exclude* (its inverse metric is the Dirichlet form, i.e. it is the max-uncertainty default). That is an epistemic default, not a damage model. The BM file's own docstring agrees: "No theorem of task-score improvement." Anything stronger would be the L1 error (local quantity → capability), repeated 10+ times in this project's history.

The one structural fact the Dirichlet objective genuinely encodes: the only difference between the two deployed tables is the **arrival jump** `eps_n`. With both ends pinned, MrRoPE pays `eps_n^2 = (1/9)^2 = 0.01235` for crashing into full compression; that single term is **94.4%** of its roughness score (exact: roughness(Mr) = `2/153`; the `eps_n^2` share = `1156/1224`).

---

## 2. The KKT system

**Problem (P):** `min g^T eps  s.t.  (1/2) eps^T L_D eps <= delta,  eps >= 0`  [variants: hard sum constraint `1^T eps = 1`; bounds; monotonicity]

Here `g_k = d(LongLoss)/d eps_k` (negative = compression at increment `k` reduces long-range loss). Lagrangian `g^T eps + lambda((1/2) eps^T L eps - delta) - mu^T eps`, `mu >= 0`:

```
stationarity:   g + lambda L eps - mu = 0
complementarity: mu_k eps_k = 0,  lambda >= 0,  lambda((1/2) eps^T L eps - delta) = 0
```

### 2.1 Interior solution and the multiplier

On the inactive set (`eps_k > 0`, `mu_k = 0`):

```
eps = -(1/lambda) L_D^{-1} g                      (Green's function of the metric)
lambda = sqrt( g^T L_D^{-1} g / (2 delta) )       (budget binds)
optimal gain:  g^T eps = -sqrt(2 delta * g^T L_D^{-1} g)     (Cauchy–Schwarz, not a model choice)
```

**Yes — the solution is the long-range gradient smoothed by the Green's function of the chosen metric.** Conditions: (i) the budget binds — always, for any `g != 0`, since the linear objective is otherwise unbounded below in the `-g` direction; (ii) no nonnegativity constraint active — **sufficient condition `g <= 0` componentwise**, since `L_D^{-1} >= 0` entrywise (M-matrix); (iii) no other constraints (sum, `m <= 1`, monotonicity) active. VERIFIED numerically on a random `g < 0`: budget saturates to `1e-15`, gain matches `-sqrt(2 delta G)` to 9 digits, `eps >= 0` exactly.

With the hard span constraint `1^T eps = 1` added (the deployed setting), stationarity becomes `eps = -(1/lambda) L_D^{-1}(g + nu * 1)` — the same Green smoothing applied to `g` shifted by a constant `nu`; the two constants are fixed by the two constraints.

### 2.2 Special case g = const → BM, exactly

`L_D^{-1} 1` has the closed form `f_k = k(n+1-k)/2` (PROVEN: applying `L_D` to `f` gives 1 identically; VERIFIED to 1e-14). Normalized to `sum = 1`:

```
eps_k = k(n+1-k) / [sum_i i(n+1-i)] = 6k(n+1-k) / (n(n+1)(n+2))     = BM exactly.
```

VERIFIED to `2.8e-17`. The prior-only problem (`g = 0`, `min (1/2) eps^T L eps s.t. 1^T eps = 1`) gives the same shape, with multiplier `nu = 12/(n(n+1)(n+2))` — and the minimal roughness value equals the multiplier:

```
R_BM = nu = 12/(n(n+1)(n+2))          R_Mr = 4/(n(n+1))          R_BM/R_Mr = 3/(n+2)
n=17:  R_BM = 2/969 = 2.064e-3,  R_Mr = 2/153 = 1.307e-2,  ratio = 3/19 = 0.1579   (exact Fractions)
n=18:  ratio = 3/20 = 0.1500
```

(constant `g` means: every increment buys the same long-range benefit — the uniform/agnostic prior.)

### 2.3 What produces MrRoPE's ramp: an end-spike forcing

**VERIFIED** to `2.8e-17`: `L_D^{-1} e_n` normalized is exactly `eps_k = 2k/(n(n+1))` = MrRoPE. Equivalently `L_D eps_Mr = (2/n) e_n` — the ramp is the Dirichlet Green response to a forcing **concentrated entirely in the last increment**: *all long-range benefit at the slow edge of the band*.

The tempting boundary-condition story ("free slow end gives the ramp") is **false** — I checked. With `eps_{n+1}` free and unpenalized, the minimum-roughness profile subject to `sum = 1` is **uniform increments** `eps_k = 1/n` (m linear — the YaRN-like ramp), energy `1/n^2`, strictly lower than the ramp's `2/153`. MrRoPE is not the minimizer of any pure Dirichlet/Neumann boundary-condition variant; it is the end-spike response. The *correct* boundary-condition reading is one-sided: BM pins both ends (`eps_0 = eps_{n+1} = 0`); the arrival jump `eps_n` is the single term whose presence (BM) or forcing-concentration (MrRoPE) flips the profile.

### 2.4 Active set for a measured g

For sign-changing `g` the solution clamps: `eps_k = 0` where the smoothed gradient has the wrong sign, and the problem is re-solved on the reduced set (`mu >= 0` on the clamped set). VERIFIED against scipy SLSQP on `g = linspace(1.5, -1.5)` (n = 17): free set = increments 6..17 (positive `g` = "compression harms long range" at the front → clamped), budget 1.0, stationarity residual 1.6e-7, `mu >= 0` on the active set. Since `L_D` is tridiagonal, `L_D^{-1} g` is an O(n) solve (Thomas algorithm), so the general solution for a measured `g` is cheap and **not generally in the b-family** — the family is a 1-D path through the n-D solution space.

### 2.5 Continuum limits

`n -> inf`: BM → smoothstep `m(t) = 3t^2 - 2t^3`, `eps(t) = 6t(1-t)`; MrRoPE → `m(t) = t^2`, `eps(t) = 2t`. The b-family is `eps(t) ~ t(1-t)^b`, peak at `t = 1/(1+b)`, i.e. peak position `k_peak = (n+1)/(1+b)`: b<1 explores the back half of the band, b>1 the front half, b=2 peaks at k=6 (n=17).

### 2.6 The implied forcing of each b-arm (computed, n=17 and n=18)

`g_implied(b) = L_D eps(b)` — the long-range forcing for which that arm is the exact interior KKT solution:

| arm | peak at k (n=17 / n=18) | implied forcing |
|---|---|---|
| b=0 | 17 / 18 | pure spike at k=n: `(2/n) e_n`, interior < 1e-16 |
| b=0.25 | 14 / 15 | spike + tiny front mass (front sum 0.0019 vs back 0.0773) |
| b=0.5 | 12 / 13 | spike + small front mass (0.0047 vs 0.0477) |
| b=1 | 9 / 9 | **exact constant** `12/(n(n+1)(n+2)) = 2/969` (deviation 1.4e-17) |
| b=2 | 6 / 6 | front-concentrated **with negative tail** (back-6 sum = -0.0103) |

The b=2 arm implicitly assumes the slow end of the band is **over-compressed** (further compression there *hurts* long-range). This is a falsifiable structural claim embedded in the arm, visible before the GPU results land.

### 2.7 The derived interpolation family (not the b-family)

A 2-mixture forcing `g = c*1 + d*e_n` (uniform + end-spike) has the closed-form response `eps ∝ k(C-k)` with `C = n+1 + 2d/(c(n+1)) ∈ [n+1, ∞)`: C = 18 gives BM (VERIFIED, diff 0.0), C → ∞ gives MrRoPE, peak at C/2. **The running sweep `k(n+1-k)^b` does not contain this family between its endpoints.** If the true forcing is a mixture, the sweep's interior arms are off-family and can only approximate the optimum.

---

## 3. Numerical verification summary (all computed)

1. `L_D^{-1} 1` normalized = BM closed form: **2.8e-17**.
2. `L_D^{-1} e_n` normalized = MrRoPE closed form: **2.8e-17**.
3. Exact roughness: BM `2/969` (2.064e-3) vs Mr `2/153` (1.307e-2), ratio **0.1579** (n=17), **0.1500** (n=18); 94.4% of Mr's score is the arrival-jump term `eps_n^2`; `R_BM < R_Mr` holds for all n = 2..30 (exact Fractions).
4. `K_flat^{-1}` = second-difference operator, diagonal (1,2,...,2): **1.1e-14**; `K_flat^{-1} 1 = e_n`: **9e-16**.
5. Flat damage along b: 4.015 → 8.737 (n=17), 4.214 → 9.223 (n=18), step = 1.000; strictly monotone.
6. Interior KKT: budget 1.0 (1e-15), gain = `-sqrt(2δG)` (9 digits), `eps >= 0` for `g <= 0`.
7. Active-set example vs scipy: free set {6..17}, budget 1.0, stationarity 1.6e-7, `mu >= 0` ✓.
8. Endpoint increments: BM moves the first band slot **2.68×** (n=17) / **2.70×** (n=18) more than MrRoPE; MrRoPE's arrival increment is **6.3×** / **6.7×** larger than BM's.
9. `m_incr_beta(1)` = BM closed form to **2.2e-16**; `m_incr_beta(0)` vs `m_mrpro(17)`: **not bitwise** — 3 of 64 entries differ by 1 ULP (2.78e-17) (docstring claims bit-for-bit; discrepancy flagged per L9). `m_incr_power(1)` IS bitwise.
10. Repo's `m_smoothstep` (q/n-grid discretization) differs from the exact-parabola BM by up to **0.0148 in m**; increments correlate 1.0. The KKT derivation reproduces the exact parabola, i.e. the BM-file object; the deployed-counter object is its smoothstep discretization. Which of the two is "the deployed table" should be pinned in the receipts.

---

## 4. Falsifiable predictions for the running b-sweep

### 4.1 The in-window axis is off the table

Round 1 (34 arms): every arm within 4e-3 nats in-window. L6b: the frequency allocation's *total* in-window contribution vs Native is ~7e-4 nats (MrRoPE's 0.0509 gap to Native decomposes as 0.0502 gain + 0.0007 allocation). **Prediction (testable): in-window NLL spread along b stays ≤ ~1e-3 nats on both models; any spread ≥ 5e-3 falsifies "the damage constraint is slack along b".** Consequence: `b* = argmin_b g^T eps(b)` — the sweep is decided entirely on the long-range axis.

### 4.2 Three theories, disjoint orderings

- **A (smoothness = damage):** b=1 optimal for every model (BM is the minimizer of its own proxy by construction — PROVEN within the family). **Already falsified** by Qwen 128K: b=0 (78.13%) > b=1 (70.83%).
- **B (flat-Fisher damage, uniform benefit):** b=0 optimal for every model (closest member to the step `e_n`). **Already falsified** by OLMo 16K: b=1 (51.32%) ≫ b=0 (2.78%).
- **C (measured-metric KKT):** `b*` model-dependent, via `g_implied` of the winner.

### 4.3 Turn-universality is already refuted  ⚠ **[CORRECTED 2026-09-11]**

> **更正**：本节的负载论据是「两个模型的端点排序相反」。对归档 Qwen 逐行数据做配对检验后，
> Qwen 侧 **4W/4L/16T，bootstrap CI 含 0**，**排序在噪声内**。该论证因此**不再是硬证据**。
> 结论（b* 模型相关）可能仍对，但必须换论证。见 `CORRECTION_QWEN_20260911.md`。

**DERIVED-using-measured-facts:** if both the damage weights `w` and the long-range gradient `g` were functions of turn count `tau_j = W omega_j/(2pi)` only, the KKT problem in turn coordinates would be identical for OLMo and Qwen (same turn band [1,32], same objective) and the two models would order b **identically**. The measured opposite endpoint orderings refute universality. **Therefore the derivation predicts: the optimum b is model-dependent; the two models will not share a peak in the sweep.**

### 4.4 Concrete per-model predictions

- **OLMo** (endpoints imply `g ≈ const`): score peaks at b=1 with monotone falloff toward b=0. **The b=2 arm is the critical discriminator:** under A, b=2 < b=1; if b=2 > b=1 on OLMo, the forcing is front-concentrated and the front-loading regime (native-gradient translation below) wins there.
- **Qwen** (endpoints imply `g ≈ end-spike`): monotone decrease, 0 ≥ 0.25 ≥ 0.5 ≥ 1 ≥ 2.
- **If any interior arm beats both endpoints on either model:** the forcing is a genuine `e_n + c·1` mixture → the derived family `eps ∝ k(C-k)` is the correct interpolation, and the running b-family misses the optimum between its endpoints; screen C next.
- **Native-gradient translation exercise** (LABELED: assumes the native-point gradient is representative of the MrRoPE-point gradient — it is not established): chain rule `g_incr(k) = sum_{j >= lo+k} g_m(j)` on the reported Phase-0 summary gives a forcing front-concentrated by a factor 8.09 (k=1 vs k=17) → predicts b=2 best on Qwen. **This contradicts Qwen's measured b=0 > b=1** → the native-point gradient does not transfer to the operating point (or first-order reasoning breaks). Real negative result for the naive translation; resolution: measure `g` at the MrRoPE base (the curvature package's `long_grad` does exactly this).

### 4.5 What to compute when the sweep lands

**The inversion recipe.** `g_implied = L_D eps(b_winner)` — closed forms: constant `12/(n(n+1)(n+2))` for b=1, spike `(2/n) e_n` for b=0, tabulated for the other arms in §2.6. Compare against a freshly measured `g` at the winner's operating point. Equality ⇒ the KKT program explains the winner (a first for this project — every local-quantity bridge so far has failed, per LESSONS L1). Inequality ⇒ the metric or the first-order approximation is wrong, and that is also a result. Until this comparison is made, no claim beyond "the b-ordering is what it is".

### 4.6 What the derivation cannot predict

A numeric `b*` per model **without measured `W` and `g` at the operating point** — the KKT solution for a measured `g` is not generally in the b-family at all, so the family optimum is a projection of `eps* = K_W^{-1}(g + nu*1)` onto the family. Saying so plainly, as required.

---

## 5. Provenance

| Statement | Status |
|---|---|
| Output-KL damage = `(1/2) eps^T K_W eps`, `K_W = U^T W U`, `(K_W)_{ij}` depends on `max(i,j)` only | PROVEN (chain rule + 2nd-order KL; verified numerically for arbitrary W) |
| Dirichlet energy is not a Fisher damage form for any `W >= 0` | PROVEN (structure contradiction `K_12 = K_22` vs `L_12 = -1`) |
| `K_flat^{-1}` = 2nd-difference operator (diag 1,2,…,2); flat-damage minimizer = step `e_n` | PROVEN + VERIFIED |
| BM = `L_D^{-1} 1` normalized = exact parabola; roughness(BM) = `12/(n(n+1)(n+2))`; ratio `3/(n+2)` | PROVEN + VERIFIED (2.8e-17, exact Fractions) |
| MrRoPE = `L_D^{-1} e_n` normalized = ramp; free-end minimizer is uniform increments, not the ramp | PROVEN + VERIFIED (2.8e-17) |
| Interior KKT: `eps = -(1/lambda)L^{-1}g`, `lambda = sqrt(g^T L^{-1} g/(2δ))`, conditions (g ≤ 0 suffices; budget binds) | PROVEN + VERIFIED |
| Active-set structure for sign-changing g | PROVEN + VERIFIED (scipy cross-check) |
| "Smoothness = damage ⇒ b=1 optimal everywhere" | FALSIFIED by Qwen 128K measurement |
| "Flat-Fisher ⇒ b=0 optimal everywhere" | FALSIFIED by OLMo 16K measurement |
| "Objective is turn-universal ⇒ identical b-orderings" ⇒ b* is model-dependent | DERIVED using measured endpoint orderings |
| Per-model b-orderings in §4.4 | SPECULATION (hypothesis with explicit test: the running sweep + the inversion recipe) |
| Native-gradient translation predicts b=2 on Qwen | DERIVED-assuming (native ≈ operating-point gradient); already in tension with measurement |
| In-window NLL spread along b ≤ ~1e-3 nats | DERIVED from Round 1 + L6b; testable |

**Discrepancies flagged (L9 discipline):** `m_incr_beta(0)` vs `m_mrpro(17)` is 1-ULP-equal in 3/64 entries, not bitwise as its docstring claims; `m_smoothstep` (deployed-counter discretization) differs from the exact-parabola BM by ≤ 0.0148 in m — the KKT derivation reproduces the exact-parabola object, and the receipts should pin which object is "the deployed table".
