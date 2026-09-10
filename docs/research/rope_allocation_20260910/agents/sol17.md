# Sol17 — scale-transport audit and a label-preserving finite optimizer

## Result

The surviving evidence does not support a universal optimizer built from source-window observability, unsigned phase distortion, Q/K projection norms, a null band, or a common-descent direction. Those quantities can constrain or diagnose a table, but they do not say whether a phase response is useful source signal or coherent distractor interference. The strongest old counterexample is `Smooth_MrBudget`: it improves several geometry and Q/K-weighted distortion measures relative to MrPro and nevertheless has worse recorded long-task development behavior. A source label and signed comparison are therefore indispensable.

A constructive restricted unification is available. For scratch allocation, a constant useful signal plus a declared shared frequency-local/nested nuisance covariance reduces to the finite count form of EVQ and admits an exact integer dynamic program. For a frozen checkpoint, the same program must retain slot labels and use role-conditioned signed source contributions. It is not valid to transplant a scratch density into unlabeled frozen slots. The exact finite-window collision optimum may be atomic, but without semantic slot labels it is only a geometry lower bound/initializer, not a frozen deployment rule.

The root's planned weights-frozen, frequency-only answer-CE calibration on 32K source rows whose positions are stretched to 128K is a useful supervised oracle. It does not by itself test this new theory. It is structurally a LeRoPE-style learned-frequency optimization with frozen non-frequency weights. Native-versus-stretch loss constraints protect the fitted rows; they do not establish the signed-margin/MGF mechanism, dense-128K extra-key competition, or the upstream state distribution of a true 128K prefill.

## 1. What survives from the old optimizers

### Source observability is a constraint, not an allocation objective

`scripts/analysis/rope_transport/tables.py:177-219` assigns displacement from a normalized conditional-uniqueness scalar. Its definition is target-free and useful for finding source-window redundancy, but it is unsigned and table-only. It cannot distinguish a redundant nuisance direction from a redundant-looking direction that participates in a learned cancellation. The same-multiset permutation evidence and the exact Q/K-frequency gauge audit show why: fixed-Q/K table permutation changes the learned computation, while joint Q/K-frequency relabeling is an identity (`paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md:25-40`).

Native-only observability also does not identify target-range behavior. Rephasing a stored native state is exact for that state, but a true long prefill changes preceding-layer states and the number/content of keys. The scale-transport runners themselves repeatedly scope their replays to fixed selected Q/K/V and explicitly deny model-quality identification.

### Null-band logic does not survive joint modes

The old null-band intuition treats a weak or wrapped marginal pair as freely movable. Mixed modes invalidate that implication: a weak individual marginal can participate in a strong beat, curvature, or cancellation. The relevant object is a signed joint relation, not the per-slot marginal energy. Astra05's exact projector construction makes the counterexample constructive: retiming one relation while preserving its orthogonal carrier can require alternating frequency changes, including acceleration, which a compression-only ramp excludes. A null band remains a diagnostic only after proving that no qualified useful relation uses it.

### MGDA is mathematically valid but answers only a local question

`scripts/analysis/verify_general_rope_allocation.py:24-42` implements the classical minimum-norm convex-hull solver. Its own checks show both cases: compatible gradients yield a common descent direction, while opposing gradients return zero (`:127-142`). This is useful for a local multi-loss step. It neither supplies missing source labels nor certifies a finite scale-four endpoint. The same file uses backtracking and an ordered-table check (`:98-126`), so the result is explicitly a checked local descent, not an analytic allocation law. Calling it a unified optimizer would recycle a generic local method and obscure the substantive statistical assumptions.

### Old tau/transport rules overstate what their moments establish

The static tau audit found that pure collision/coherence/conditioning objectives prefer extreme tau around 11–14 and have near-zero length exponent, while the self-consistent surrogate reached only about `L^-0.17`. The later stiffness note obtains exponents near `-0.5` by choosing an f-divergence exponent or by conditioning on the desired empirical scaling. Those calculations can motivate a family; they do not identify the real nuisance covariance or derive a task-optimal Qwen profile. The exact-kernel result in Astra02 sharpens the issue: at a finite continuous lag window, the exact EVQ measure optimum is a unique finite atomic equilibrium, not a corrected Cosh density. Cosh requires the additional anti-concentration/local-noise model.

### Geometry improvements are not sufficient evidence

The assigned weighted-subspace code says exactly what it computes: independent unit-second-moment Q/K inputs, not measured activations or task loss. `Smooth_MrBudget` improves the available unsigned geometry and Q/K-weighted weak-subspace criteria yet loses on the recorded 128K development rows. The old `operator_family` output gives another objective-transfer warning: a fitted operator can reduce an internal error and still miss the answer. Likewise, the 125M diagnostic improves long NLL while retrieval collapses. These are not arguments against all geometry; they veto geometry as a sufficient selector.

## 2. Shared mathematical object

For frozen slot `j`, query/source `+`, and matched hard distractor `-`, retain the actual signed Q/K quadratures

\[
A_{rj}^{\pm}=q_{rj,0}k^{\pm}_{rj,0}+q_{rj,1}k^{\pm}_{rj,1},\qquad
B_{rj}^{\pm}=q_{rj,1}k^{\pm}_{rj,0}-q_{rj,0}k^{\pm}_{rj,1}.
\]

For candidate frequency `nu_j` and signed lags `d_r^+`, `d_r^-`, the exact conditional source margin is

\[
M_r(\nu)=\gamma\sum_j\big[A_{rj}^+\cos(\nu_jd_r^+)+B_{rj}^+\sin(\nu_jd_r^+)
-A_{rj}^-\cos(\nu_jd_r^-)-B_{rj}^-\sin(\nu_jd_r^-)\big],
\]

with the fixed gain included in `gamma`. Over a declared role/lag population, define signed mean `mu(nu)` and either the full covariance of these margins or a unit-argument log-MGF bound. This preserves cancellation and source identity. Projection Frobenius energy, marginal phase coverage, and unsigned attention displacement discard this information.

For one useful key and `N` competing keys, if the source score is deterministic `mu` and each distractor satisfies

\[
\mathbb E e^{D_t}\le e^{b_t+v_t/2},
\]

then Markov's inequality gives

\[
\Pr[p_*<\rho]\le {\rho\over1-\rho}\sum_{t=1}^N e^{b_t+v_t/2-\mu}.
\]

No independence across distractors is required. If the source is random, the required object is the joint margin MGF `E exp(D_t-S)`; replacing it by the mean source score is invalid. This is stronger than vague “unknown moments”: mean/covariance alone gives only Cantelli, and a sample covariance is not a population PSD upper bound. A Gaussian or sub-Gaussian tail must be declared and checked; it cannot be inferred from products of Gaussian-looking Q/K activations.

## 3. Exact finite count DP and its frozen-label version

Declare `B` resolved log-frequency bins with width `Delta`, `K` equal-amplitude rotary pairs, integer occupancies `n_i`, and tail counts `T_i=sum_{l>=i} n_l`. Assume the explicit covariance

\[
C_{il}={\alpha\over\Delta}\mathbf 1\{i=l\}+\beta\min(x_i,x_l),\qquad \alpha>0,\ \beta\ge0.
\]

This represents shared bin noise plus shared nested slow-tail noise. It is not iid channel noise: iid per-channel noise would give a density-linear contribution and would not recover EVQ's squared-density cost. Let `h_i` be a deterministic useful signal contribution. Minimizing the softmax mass bound is equivalent, up to constants, to

\[
\min_{n_i\in\mathbb Z_+,\ \sum n_i=K}
\sum_i\left[{\alpha\over2\Delta}n_i^2+{\beta\Delta\over2}T_i^2-Kh_in_i\right].
\tag{1}
\]

The exact backward recurrence is

\[
F_i(t)={\beta\Delta\over2}t^2+
\min_{0\le n\le t}\left\{{\alpha\over2\Delta}n^2-Kh_in+F_{i+1}(t-n)\right\},
\]

with `F_{B+1}(0)=0` and infinity otherwise. Backtracking from `F_1(K)` emits the globally optimal integer counts under the declared model in `O(BK^2)` time. Occupancy caps and required endpoints enter by restricting `n`. Duplicate frequencies are part of this exact model; spreading them after solving changes the problem.

For a frozen ordered table, counts alone are inadmissible. Preserve slot labels and assign consecutive labels monotonically to bins. Replace `h_i n` by

\[
\sum_{j=K-t}^{K-t+n-1}h_{j,i},
\]

where `h_{j,i}` is the signed useful contribution if original slot `j` is assigned bin `i`. Precomputed prefix sums keep the same complexity. Per-label native constraints can forbid assignments whose certified native role margin or MGF bound falls below its frozen threshold. This yields an exact DP for the special covariance above while respecting learned labels. With general heterogeneous cross-slot covariance the recurrence no longer applies; use the certified finite-margin conic solver from Astra03 or return that the special model is unsupported.

Constant `h` makes the linear term constant and recovers the discrete Cosh recurrence in the continuous relaxation. Oscillatory, source-aligned `h_{j,i}` yields a Mr-like preference for positive remote response. MrPro's arithmetic radix increments are still heuristic: neither the first-zero argument nor (1) derives its exact quadratic cumulative schedule without further assumptions.

The exact finite-kernel atomic optimum from Astra02 is complementary. It supplies a collision lower bound and an exchange/certificate algorithm, but its unlabeled measure cannot choose a frozen Qwen slot assignment. Integer count DP is the appropriate bridge only when the covariance resolution and label-conditioned signal are explicitly defended.

## 4. Counterexample checks

1. If all signed means vanish, reducing covariance improves only a bound and supplies no positive source signal. The optimizer must report no certified retrieval direction.
2. With iid isotropic sine/cosine distractor coefficients, score variance is invariant to frequency. An EVQ collision term disappears; any derivation retaining it is using an unstated coherent covariance.
3. Two candidates can share mean/covariance and have different nonlinear failure probabilities. Cantelli-bound ordering is not exact performance ordering without a common coupling, Gaussian model, or another stochastic dominance condition.
4. Better pairwise standardized margins need not imply better multi-key joint success because dependence among margins matters. The softmax MGF bound remains sufficient but can be loose.
5. If a joint Q/K-frequency permutation is applied, the computation is unchanged. A frozen table-only permutation is not; any frozen optimizer that drops labels contradicts this gauge check.
6. If a true 128K prefill moves upstream activations outside the calibrated envelope, a fixed-state certificate does not apply even when its local finite-phase arithmetic is exact.

## 5. Evaluation of the root's current calibration plan

The plan, as communicated, uses Qwen2.5-3B at native window 32K, freezes all ordinary weights, optimizes only frequencies against answer cross-entropy on source-task rows with positions stretched to 128K, imposes exact Native-versus-stretch loss constraints, then evaluates 128K PPL/passkey and held-out generation.

What it does test:

- whether a supervised, frequency-only table can improve answer likelihood under the chosen sparse position-stretch intervention while respecting fitted native losses;
- a stronger endpoint than fixed-state geometry or attention-map Frobenius loss, because the complete frozen model and requested answer tokens participate;
- whether the resulting direction transfers from fitting rows to held-out rows and later to real 128K endpoints.

What it does not test:

- the new role-conditioned margin/MGF theory, unless the theory produces a preregistered ranking or table independently of answer CE;
- dense 128K extra-key competition, because 32K tokens spread over a 128K coordinate range still contain only 32K keys;
- the state distribution of a contiguous 128K prefill; stretched positions change phases but do not supply the missing intervening content or its upstream influence;
- a general frozen deployment rule, because answer CE directly learns task labels and the native constraints are conditional on the fitted rows;
- scratch EVQ's count-density claim, because the weights are frozen and slots are already labeled.

The closest methodological description is “supervised frozen-backbone per-frequency calibration,” in the same broad family as LeRoPE's learned frequencies, with a different frozen-weight and constrained-loss protocol. Calling it a derivation or a direct test of EVQ/Mr unification would overstate it. It is valuable as an oracle ceiling and as a source of a candidate direction.

The phrase “exact loss constraints” also needs scope. Exact numerical evaluation of both losses at each candidate is good. It does not make the constraint population-level, and it does not isolate source routing from output-format or decoder effects. The native and stretched rows must be disjoint from the final holdout, and a source-swap or source-deletion pair is needed to show that improved CE follows the lawful source rather than a shortcut.

## 6. Better minimal experiment

Before optimizing any new table, use the same frozen source rows and calculate one preregistered mechanism diagnostic for the already available `MrPro`, `Smooth_MrBudget`, `P2`, and `E1` tables:

1. For each correct source, choose matched content-confusable distractors and preserve the source identity under a paired position stretch. Compute the exact full-row source-versus-distractor log odds, signed mean margin, empirical covariance, and unit-argument log-MGF cost at native and stretched positions. Keep task/lag cells separate.
2. Require the role-conditioned statistic to rank the known `Smooth_MrBudget` loss correctly on these development rows. If it still prefers Smooth, stop the theoretical claim; inspect source labels, values/readout, upstream state drift, or generation-prefix effects. Do not add another unsigned term.
3. If the diagnostic survives, solve one frozen-label DP table under a declared covariance, or the certified finite-margin solver when empirical covariance is heterogeneous. Fit/calibration rows determine the statistical inputs; answer CE is not used to select the table. Freeze one table and its mirrored equal-size direction.
4. On an untouched distance-paired set, compare `MrPro`, the theory table, and the mirror. Measure answer CE, exact source-following under world/source swaps, and complete generation with EOS. This tests whether the selected signed role matters rather than whether any small table perturbation helps.
5. Only after that mechanism comparison should the selected table enter real contiguous 128K PPL, passkey, and held-out generation. These remain the requested outcome endpoints.

This is cheaper and more discriminating than immediately fitting 64 frequencies to answer CE. It tests the theory against an existing falsifier before paying for a new table. If the theory fails but the CE optimizer succeeds, report a supervised frequency-calibration result rather than evidence for the unification. If both succeed and their directions agree, the CE optimizer becomes an independent oracle comparison.

## 7. Claim boundary

Supported: role-conditioned signed source/interference is a common formal object; under the declared shared-bin/nested covariance, scratch EVQ and a frozen-label finite allocation share an exact integer DP; the current CE calibration is a useful supervised oracle; the proposed diagnostic directly separates the new theory from generic learned-frequency fitting.

Unresolved: the real Qwen role-conditioned covariance/MGF envelope, whether the DP's structured covariance is adequate, whether any theory-derived table beats MrPro, and whether sparse position-stretch transfer survives dense contiguous 128K prefill and generation.

No GPU/model job, frequency table, runtime source, or paper source was created or modified in this assignment.
