# Sol15 — claim audit and a defensible EVQ/MrRoPE unification statement

## Result

The manuscript's current central claim survives: exponent allocation is a real design variable, fixed-range allocation changes learned behavior, and frozen checkpoints can prefer different native-relative displacement profiles. What does **not** survive is any stronger reading that EVQ-Cosh, the exact EVQ collision kernel, MrRoPE-Pro, or a geometry-only score is a universal optimizer of language-model quality.

The mathematically defensible common object is a **role-conditioned source-versus-competitor log-partition margin over a finite frequency table**. EVQ and MrRoPE enter as restricted signal/noise models or feasible parameterizations of that object. Training from scratch and frozen deployment are separate optimizations because the learned coefficients change with the table in the first regime and are slot-labeled compatibility constraints in the second.

This report fully ingested all 48 assigned files (1,107,672 bytes), including all 6,144 JSONL rows in 52 contiguous pages, and then read Astra01–06 in full. No GPU/model job, table proposal, source edit, or paper edit was made.

## 1. What the actual manuscript claims

The active manuscript is substantially more careful than the historical theory note and some archived reviews.

* `paper-2027/sections/01_intro.tex:13–25` asks three separate questions, reports the fixed-range result and the range-retargeting reversal together, and says weights-by-table crossings expose interaction with learned representation.
* `paper-2027/sections/03_theory.tex:74–90` explicitly **chooses** the convex criterion
  \[
  \mathcal C_{\rm app}[\rho]=\frac\alpha2\int\rho^2+\frac\beta2\int S_\rho^2
  \]
  as specified design preferences. Its Cosh theorem at `:92–115` is exactly scoped to this functional. It does not say Cosh minimizes language-model loss or the exact finite-window collision kernel.
* `paper-2027/sections/budget_intro.tex:30–44` expressly says the surrogate supplies a reproducible construction, does not establish optimality for the rotary-budget objective, and leaves the budget interaction experiment pending.
* `paper-2027/sections/02_identification.tex:4–24` supports only the causal fixed-endpoint statement: across three paired seeds, Cosh lowers tail NLL at 512/1K/2K, while separately retargeting each support reverses the ordering. This establishes allocation activity and allocation-by-range interaction, not a universal Cosh preference.
* `paper-2027/sections/02_identification.tex:28–38` weakens functional specificity further: the matched exponential is unresolved from the reference Cosh, and preferred strengths vary by configuration.
* `paper-2027/sections/03_findings.tex:14–20` gives the decisive regime split: Geo-trained/Geo-runtime versus Geo-trained/Cosh-runtime is 7.14 versus 76.20 PPL; the reverse swap is 7.16 versus 23.05, with a larger replication. An unlabeled frequency measure cannot describe frozen compatibility.
* `paper-2027/sections/04_mature.tex:136–160` reports checkpoint-dependent frozen preferences: BM is much better for OLMo, whereas MrPro is better at 128K for both Qwen checkpoints. This rules out a checkpoint-independent middle-band optimum.

The Round-2 PDF review independently read the contribution the same way and specifically retained the bounded Cosh and geometry-to-construction language (`paper-2027/research/pdf-review-rounds/20260909/r02/review.md:5–14`). The older Qwen panel warning that the abstract needed the target-matched reversal is now resolved in the active introduction (`01_intro.tex:21–25`); its archived Major Revision label is not a present scientific verdict.

## 2. The common exact object

For a query, let \(S_r\) be the role-qualified useful keys and \(D_r\) the relevant competitors. With existing labeled rotary slots \(j=1,\dots,K\), frequencies \(\nu_j\), signed lags \(d_t\), and content coefficients \(A_{tj},B_{tj}\), write

\[
z_t(\nu)=c_t+\sum_j\{A_{tj}\cos(\nu_jd_t)+B_{tj}\sin(\nu_jd_t)\},
\]
\[
M_r(\nu)=\log\sum_{t\in S_r}e^{z_t(\nu)}-
          \log\sum_{t\in D_r}e^{z_t(\nu)}.
\tag{1}
\]

Equation (1) is exact for a fixed row state. It retains source labels, hard-key competition, signed sine/cosine phases, slot identity, gain through the coefficients, and distractor count. For one source, \(p(S_r\mid S_r\cup D_r)=\sigma(M_r)\). It is therefore stronger than an unsigned collision energy, first-zero horizon, pairwise SNR, or frequency multiset.

For a random role/example population, a useful probability certificate is based on the joint margin MGF. If the useful source score is deterministic \(\mu(\nu)\) and each distractor obeys

\[
\mathbb E e^{D_t(\nu)}\le e^{b_t(\nu)+v_t(\nu)/2},
\]

then Markov's inequality gives, for desired source mass \(q\in(0,1)\),

\[
\Pr[p_*<q]\le \min\left\{1,
\frac{q}{1-q}\sum_t e^{b_t+v_t/2-\mu}\right\}.
\tag{2}
\]

No independence across distractor keys is needed. If the source is random, (2) requires \(\mathbb E e^{D_t-S}\); replacing \(S\) by its mean is invalid. In the equal zero-mean/variance case, the correct scalar objective is \(v/2-\mu\), with the unavoidable \(\log |D_r|\) threshold. Pairwise SNR \(\mu/\sqrt v\) alone can improve while softmax source mass worsens, and multiple pairwise SNRs do not determine joint retrieval probability.

## 3. Exactly when EVQ appears

EVQ's exact collision energy is

\[
\langle\rho,K\rho\rangle
=\mathbb E_\Delta\left[\int\cos(\omega(x)\Delta)\rho(x)dx\right]^2.
\tag{3}
\]

It is a squared coherent aggregate response. It follows from (2) only under a specific nuisance model: the useful mean is protected and allocation-independent, while each distractor has a shared scalar loading across frequencies, so its variance is the squared summed-cosine response. Independent isotropic sine/cosine noise per channel is rotation-invariant and produces no frequency-dependent collision term. Likewise, independent channel errors yield a variance linear in the density, scaled by \(1/K\), rather than \(\alpha\int\rho^2\).

The Cosh functional needs the additional covariance approximation

\[
C_{\rm app}=\alpha I+\beta G,\qquad G(x,y)=\min(x,y).
\]

Here \(\alpha I\) must represent a resolved shared frequency-local noise field or a declared finite-bin model. Point evaluation of continuous white noise at atomic channel frequencies is undefined, and the finite-channel and white-noise limits do not commute. Under constant useful signal, the inverse-covariance law \(\rho\propto C^{-1}{\bf1}\) gives the Cosh density. With frequency-dependent useful signal \(h\), the active-set law is \(Cw\ge\lambda h\), \(w\ge0\), with equality on occupied support; there is no universal Cosh curve.

Astra02's exact finite-window result is compatible with this conclusion and must remain separate. For the continuous positive lag prior on a finite interval, the exact analytic kernel has a unique **finite atomic** probability-measure minimizer. The delta-ridge surrogate introduces the anti-concentration term that creates a smooth density. The atomic theorem does not directly transfer to a finite integer-lag Gram, where strict positive definiteness and uniqueness can fail. It also does not say an LM should repeat a handful of frequencies.

Thus four objects must not be conflated:

1. the exact continuous-lag collision measure optimum, which is finite atomic under Astra02's assumptions;
2. the smooth surrogate density optimum, which is Cosh under constant signal and shared local-plus-nested noise;
3. the finite \(K\), equal-amplitude table, whose atom masses must be integer channel counts;
4. the frozen labeled table, where slot-specific learned coefficients prevent exchangeable-count reduction in general.

The historical `docs/theory/EVQ_COSH_THEORY.tex:105–125` is therefore stronger than the active manuscript in two places. A nonzero multiple of the identity is not Hilbert–Schmidt in infinite-dimensional \(L^2\), so literal continuous Hilbert–Schmidt projection onto \(\alpha I+\beta G\) is undefined. A finite Galerkin resolution must be declared. Its `:216–223` “dominant Cosh component” is a heuristic scale argument, not a consequence for task risk. Its \(\tau\approx d_{\rm head}/\sqrt L\) section correctly labels the law conjectural; `scripts/verify_tau_unified.py` does not upgrade it. The script's hand-entered 15 anchors have mean relative error 9.6%, maximum 33.3%, and 10/15 below its chosen 15% threshold. Calling this `PASS` is a script convention, not a theorem or independent validation.

## 4. Finite channel counts: the strongest constructive special case

Astra06 supplies an exact count allocator under a declared finite-resolution covariance model. Divide exponent space into \(B\) bins of width \(\Delta\), put \(n_i\in\mathbb Z_+\) equal-amplitude channels in bin \(i\), and set \(\sum_i n_i=K\). With

\[
C_{ij}=\frac\alpha\Delta\mathbf1\{i=j\}+\beta\min(x_i,x_j),
\qquad T_i=\sum_{j\ge i}n_j,
\]

and useful unit-channel mean \(h_i\), (2)'s equal-variance objective becomes exactly

\[
\min_n\sum_i\left[
\frac{\alpha}{2\Delta}n_i^2+
\frac{\beta\Delta}{2}T_i^2-Kh_in_i
\right].
\tag{4}
\]

The backward recurrence

\[
F_i(t)=\frac{\beta\Delta}{2}t^2+
\min_{0\le n\le t}\left\{
\frac{\alpha}{2\Delta}n^2-Kh_in+F_{i+1}(t-n)
\right\}
\tag{5}
\]

solves (4) globally in \(O(BK^2)\), and backtracking emits integer counts. This is the clearest answer to the density-versus-channel-count problem. For constant \(h\), the real relaxation satisfies a discrete Cosh recurrence and converges to the continuum Cosh law in the resolved-density limit. At finite \(K\), the integer DP is the optimizer.

Its assumptions are substantive. Noise is shared within bins and has nested Brownian covariance; \(B\) is physical correlation resolution, not an arbitrary candidate grid. Repeated frequencies are allowed. If strict distinctness, minimum spacing, endpoint occupancy, or slot-specific covariance is required, the feasible problem changes. For frozen ordered slots, a DP remains possible only in the special case where covariance depends on bins/counts and slot-specific means \(h_{j,i}\) are additive over consecutive assignments. General heterogeneous cross-slot covariance destroys the recurrence.

## 5. What MrRoPE contributes, and what it does not

MrRoPE supplies a useful native-relative coordinate and feasible scaffold. With \(\nu_j=\omega_jS^{-m_j}\), the transition can be expressed through nonnegative edge increments whose sum is one. MrUni and MrPro are fixed allocations of that dilation budget. This coordinate unifies many frequency scalings algebraically, as the manuscript correctly states at `paper-2027/sections/04_mature.tex:12–34`.

The MrRoPE paper does not derive the progressive arithmetic profile from its positive-cosine or first-zero analysis. It explicitly **assumes** an arithmetic progression for the radix exponents before obtaining its closed form (local Markdown lines 420–440), and its Qwen boundaries are empirically selected (Appendix B). Its first-zero argument concerns positive coherent matched-content response, whereas EVQ suppresses squared coherent nuisance response. These become two terms of (2) or (1) only after source and distractor populations are declared.

Astra05's projected mixed-mode retiming

\[
\nu=(I-P_R)\omega+S^{-1}P_R\omega
\tag{6}
\]

is mathematically exact: it retimes every linear frequency relation in row\((R)\), preserves every orthogonal relation, minimizes Euclidean frequency displacement, and composes over scale while \(R\) is fixed. It is a genuine finite transport outside independent compression ramps. But (6) is an exact solution **conditional on a role-qualified relation set \(R\)**; it is not a way to discover \(R\). A large bias harmonic, long beat period, projection norm, or unsigned mode magnitude does not establish that a relation carries useful source-versus-distractor computation. Order can also fail, and some exact beat-preserving moves accelerate individual frequencies. The projected transport is therefore a candidate generator after role identification, not a universal unification law.

Astra04's stitched-block KL allocator is similarly conditional. It gives an exact position-stretch distillation loss and a correct extra-distractor term, but cached native states are not a 128K forward, zero-padding the teacher asserts distractor irrelevance, and equal block weighting specifies an evaluation prior. It can rank a fixed-state intervention; it cannot establish whole-model transfer.

## 6. Evidence checks against stronger interpretations

The assigned evidence repeatedly enforces the claim ceiling.

* The 6,144-row sparse-memory baseline has 2,048 paired examples. Both relation worlds are solved perfectly (4,096/4,096; mean margin 9.1994), while the content world is only 399/2,048 correct (19.48%; mean margin −0.87595). All three rows are correct in exactly 399 pairs. This cleanly demonstrates role heterogeneity: a high aggregate accuracy or relation margin can coexist with failure of the content role. It is one synthetic TP baseline, not an EVQ/MrRoPE comparison.
* `FRESH_FINEWEB_S4_RESULTS_20260824.json` supports one-checkpoint frozen natural-text tail-NLL and fixed-support effects, with its own explicit claim boundary. It does not validate a universal selector.
* `ALLOCATION_DOSE_RESPONSE_RESULTS_20260826.json` reports monotone pieces and a failed primary joint gate; it explicitly says it is not a table selector or checkpoint-population result.
* The two-parameter OLMo movement fit has MAE 0.001223, but its own metadata says reconstruction only and no downstream/Qwen fit. Small profile error is not task sufficiency.
* The seed-42 EVQ LoRA raw probes collapse from perfect 8K passkey to zero at 16K/32K; RULER top-1 also largely collapses. This is a real negative for that adapter/table/protocol, not a universal refutation of scratch allocation or EVQ.
* The operator-family cache counterexample shows row/fixed schedules can be partition-invariant while call-time scheduling is not, but that experiment reads every token and concerns dense per-token representation compression. It does not test sparse token selection.
* BM's OLMo gains and Qwen long losses are both real within their reported scopes. Smooth_MrBudget improving all audited unsigned geometry costs while losing task score rules out those costs as sufficient selectors. P2 and E1 preserve conditional development signals and should not be erased by that failure.

## 7. Improved statement suitable for integration

The strongest accurate unification statement is:

> A RoPE design allocates a finite set of equal-amplitude rotary channels over frequency while choosing which positional score relations to preserve or retime. For a declared source/competitor population, the operative quantity is the signed log-partition margin generated by the table and its learned content coefficients. In an exchangeable scratch-design model with allocation-independent useful signal and shared coherent local-plus-nested nuisance, minimizing a softmax failure bound reduces to the EVQ surrogate and yields Cosh in the continuum; its finite resolved equal-count version is an integer allocation problem. In a frozen checkpoint, channel labels and learned coefficients must be retained, so allocation is a constrained finite transport problem. MrRoPE supplies a structured native-relative dilation family, while role-qualified harmonic constraints can justify exact mixed-mode retiming. None of these restricted reductions establishes a checkpoint-independent LM-optimal curve.

This statement is stronger than “both methods alter frequencies” because it identifies the shared task object, the content/noise assumptions that recover EVQ, the feasible deployment structure supplied by MrRoPE, the finite-count solver, and the regime boundary created by co-adaptation.

## 8. Constructive next rule for Qwen2.5-3B 32K→128K

Use MrPro as the installed reference and preserve its evidenced fast/slow endpoints and gain. On a benchmark-isolated calibration set, identify actual source-dependent operations and hard competitors, then measure signed log-partition margins under MrPro, Smooth, P2, and the one-slot E1 direction using either real long prefills or a separately tested transport envelope. The selector must first reproduce at least one existing ordering/failure contrast; if it still ranks Smooth above MrPro on the failing rows, the chosen row state or source labels are not explanatory.

Then choose exactly one of two constructive routes based on the measured structure:

1. If statistics are approximately exchangeable within resolved frequency bins and the shared local-plus-nested covariance is empirically adequate, solve the finite-count DP (5), with ordered labeled assignment if its additive special case holds.
2. If one or a few signed joint relations dominate normalized source-versus-competitor mass, use their qualified row span in the exact projected retiming (6), followed by exact finite log-partition evaluation and an equal-size orthogonal perturbation control.

Freeze one output, compare it with reused MrPro on independent long rows, and retain native-window task outcomes. A calibration-margin gain without generated-answer gain falsifies transfer/readout sufficiency; it does not validate another proxy. No new table is justified by this report alone because the required Qwen role-conditioned statistics are not present in the assigned corpus.

## Bottom-line verdict

**Supported:** allocation geometry as a real variable; fixed-range causal effect; range and weight co-adaptation; Cosh as the unique optimizer of the declared smooth surrogate; MrRoPE as a native-relative displacement coordinate and empirical baseline; conditional role-margin, finite-count DP, and exact mixed-mode transport constructions under explicit assumptions.

**Unsupported:** universal Cosh or MrPro optimality; deriving \(\tau=d/\sqrt L\) from current theory; treating iid channel noise as the EVQ \(L^2\) term; treating the exact finite-window kernel optimum as a smooth density; transplanting an unlabeled density to a frozen checkpoint; selecting a table by smoothness, effective rank, residual energy, SNR alone, or unqualified harmonic magnitude; claiming task success from fixed-state attention or geometry.
