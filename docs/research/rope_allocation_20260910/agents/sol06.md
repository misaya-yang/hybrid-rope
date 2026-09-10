# Sol06: allocation math audit and constructive regime-conditioned unification

## Decisive result

There is a clean common allocation variable, but there is no evidence-grounded common **proxy objective** for scratch training and frozen deployment. The common variable is the extra log-gap budget. The objective must change with whether the coefficients can coadapt.

Let native frequencies be `omega_j`, deployed frequencies be

\[
\nu_j=\omega_j e^{-x_j}=\omega_j S^{-m_j},\qquad 0\le m_0\le\cdots\le m_{K-1}\le1.
\]

In the frozen Qwen deployment, the presently evidenced feasible face is
`m_j=0` for `j<=23`, `m_j=1` for `j>=40`, with monotonicity in between. Equivalently, writing deployed log-period gaps as

\[
a_i=\log(T_{i+1}^{dep}/T_i^{dep})=a_i^0+e_i,
\]

the transition variables obey `e_i>=-a_i^0` (frequency order) and

\[
\sum_{i=23}^{39}e_i=\log S.
\]

This is the rigorous useful content of the generalized gap-budget view: it parameterizes a simplex/polytope. It does not determine where the budget should go. The document itself correctly says the budget centroid is not an objective (`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:14-25`).

I propose the following **relation-targeted gap allocation (RTGA)** construction. It fills in the missing coupled quadratic anticipated by the subspace derivation, while keeping its performance claim appropriately conditional.

## Constructive objective and solution

For a finite set of native interactions `n in Z^K`, define the mixed frequency

\[
\kappa_n(\nu)=n^T\nu.
\]

These are not invented features: the exact exponential-softmax expansion contains every such integer combination, with content-dependent Bessel coefficient (`ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md:19-46`). Assign each retained interaction a desired clock `t_n in [0,1]`:

\[
\kappa_n^{tar}=S^{-t_n}\kappa_n(\omega).
\]

`t_n=0` preserves a native-scale local operation; `t_n=1` retimes a long-scale operation by PI; intermediate values express a genuinely mixed requirement. Around a reference `x^r`, linearize

\[
\kappa_n(x^r+v)\simeq \kappa_n(x^r)-a_n^Tv,
\quad (a_n)_j=n_j\nu_j^r.
\]

Set `b_n=kappa_n(x^r)-kappa_n^tar`. With rows `a_n^T` in `A`, weights `W=diag(w_n)`, and a positive semidefinite stabilizer `Q`, solve exactly one constrained quadratic program:

\[
\boxed{
v^*=\arg\min_{v:\ x^r+v\in\mathcal F}
\frac12\|W^{1/2}(Av-b)\|_2^2+
\frac\zeta2 v^TQv+c^Tv
}
\]

where `F` is the endpoint, monotonicity, and `[0,log S]` feasible polytope. `Q` may be the source-subspace commutator/transport quadratic, but only as a stabilizer or constraint, not as the capability selector. `c` is a signed task-risk derivative and is zero only when coadaptation makes compatibility irrelevant.

Before inequality constraints, for `zeta>0` and positive definite `A^TWA+zeta Q`, the constructive solution is

\[
v^*=(A^TWA+\zeta Q)^{-1}(A^TWb-c).
\]

With constraints it is a convex QP with a global optimum; active-set/KKT conditions give the finite table. Evaluate the **exact** nonlinear residuals `kappa_n(nu)-kappa_n^tar` afterward and relinearize only if needed. This is an algorithm that emits a table, not an unspecified `argmin` over an unknown utility and not a hand-built grid.

The relation matrix also gives the right coupling. For a native cancellation such as `n=(1,-1)` or a higher difference, `a_na_n^T` has off-diagonal terms. Adjacent smoothing cannot reproduce this in general. The source-subspace calculation independently obtains graph coupling through `sum ||P_ij||^2(m_i-m_j)^2` (`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:121-143`).

## What fixes `w_n`, `t_n`, and `c`

These quantities are regime dependent; pretending otherwise recreates the failed proxy jump.

### Frozen deployment

Weights cannot coadapt, so preserve the interactions the checkpoint actually uses. Obtain a finite relation set and signed importance from native conditional content, preferably on a frozen, benchmark-isolated calibration corpus:

1. For each head/layer and sampled attention row, retain the largest measured Bessel/Fourier interaction coefficients (or a controlled low-order set with a certified coefficient-mass tail).
2. Label a relation native (`t_n=0`) or retimed (`t_n=1`) by whether intervention on that relation improves the declared short/long self-supervised risk, not by individual turn count. If evidence conflicts, retain `t_n` as a calibrated continuous target or place the interaction in both objectives.
3. Let `c` be the full-model signed gradient of the declared frozen calibration losses with respect to `x`, including changed prefixes and readout. The repository already derives the exact per-frequency signed logit derivative and explains why detached Q/K norms cannot replace a whole-network gradient (`ROPE_GENERAL_ALLOCATION_DERIVATION_20260907.md:74-112`).
4. Use the endpoint constraints supported by HighGapToLong/MrUni/E2/E8 as this checkpoint's currently tested feasible face, rather than universal laws.

For multiple short/long calibration cells, replace `c` by the MGDA minimum-norm convex combination of their gradients. This gives a common first-order descent direction when one exists, followed by exact finite-step acceptance checks. It avoids arbitrary scalar weights and is already mathematically derived in the project (`ROPE_GENERAL_ALLOCATION_DERIVATION_20260907.md:97-132`). RTGA adds the missing relation-preservation curvature and the exact gap constraints around that signed direction.

### Training from scratch or sufficiently strong adaptation

Compatibility with fixed native coefficients is not the same problem. Optimize

\[
\min_{W,x}\;R_{train/val}(W,x)+\lambda J_{rep}(x),
\]

where `J_rep` can be EVQ's density/gap functional. In gap coordinates EVQ is a representation prior of the form

\[
J[h]=\tfrac12\int\{\alpha/h+\beta(1-u)^2h\}\,du,
\quad \int h=1,
\]

whose stationarity yields the Cosh family. This is a legitimate regularizer when `W` can relearn the basis; it is not a frozen capability theorem. For a local frequency step, RTGA is recovered with `c=nabla_x R`, curvature from the loss or retained interactions, and `t_n` learned implicitly by the data rather than fixed from the native checkpoint. Equal training exposure and update sets are required when comparing allocation families (`COSH_REDESIGN_EVIDENCE_REVIEW.md:155-171`).

Thus EVQ and MrRoPE unify as follows: both allocate the same log-gap mass; scratch EVQ prices representation density because coefficients can move, while frozen MrRoPE prices compatibility with learned native and retimed computations. The latter naturally produces two coherent blocks and a coupled bridge when interaction targets separate. A hard native prefix plus a common `/S` suffix is an exact special case: all interactions supported entirely in either block satisfy their target clock exactly; only cross-block relations enter the bridge QP. This follows from the exact mixed-frequency retiming identity (`ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md:130-149`).

## Counterexamples and corrections to recent theories

1. **No nonnegative combination of the audited geometry costs is a sufficient selector.** Smooth MrBudget has lower unweighted unresolved exposure and local distortion than MrPro but worse 128K behavior (`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:192-203`). It also has lower Q/K-weighted operator MSE at every listed lag range (`ibid.:240-255`) and lower Q/K-weighted unresolved response at every cutoff, in all 36 layers (`ibid.:258-276`). Therefore any objective monotonically increasing in just these audited costs ranks Smooth at least as good as MrPro, contrary to the measured 128K ordering. Adding positive weights or averaging more layers cannot repair it. A signed, content- and task-conditioned term is necessary.

2. **Individual arc horizons do not imply a hard failure boundary.** `D_j=W S^{m_j}` is algebraically the largest distance whose phase magnitude is no larger than that channel's native maximum. But periodic functions do not have an ordered notion of an unseen arc after multiple turns, and useful computation includes mixed `n^Tnu`. The project's own row evidence has MrPro successes beyond the proposed danger band and failures are not determined by distance alone (`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:58-71`). So `D_j` is a risk feature, not proof that every slot 36--39 must reach `m=1`.

3. **The filter-bank/arc-clock dichotomy is an empirical hypothesis, not a derived binary law.** The extrapolation review correctly describes high/low bands as asymptotic regimes and the middle as a practical transition, not a third mathematical module (`ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md:27-50`). Slots 28/29 complete roughly 12.37/9.97 native turns yet have conditional interventions, already defeating a one-turn account (`ibid.:51-69`). RTGA handles this by testing relations and signed use rather than imposing a universal rotation threshold.

4. **Mixed-frequency existence is insufficient.** The 28-2(29)+30 mode has a roughly 70K period, but its relative amplitude ranges from about 0.00049 at assumed amplitude .3 to .294 at amplitude 2. The derivation explicitly warns that content varies and upstream states change (`ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md:123-128,159-170`). Hence relation selection must use measured native content and tail bounds; a low `|n^Tnu|` alone is not a cost.

5. **Exact preservation and extrapolation conflict in the universal-content limit.** Preserving every displacement-one bilinear operation forces the original rotation, while changing long-distance behavior requires moving frequencies (`ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md:123-134`). RTGA does not claim universal preservation: it selects a finite distribution-weighted set and exposes residual conflict through the QP value.

6. **The Cosh formula proves an optimum only for its selected functional.** Its form follows once constant-coefficient local square and min-kernel terms are assumed; improved surrogate fit does not validate task selection (`COSH_REDESIGN_EVIDENCE_REVIEW.md:47-58`). The deployment tau also failed to select the best nearby point in the recorded multi-seed comparison (`ibid.:62-82`). RTGA therefore treats EVQ as a scratch regularizer, not the frozen target.

## Minimal decision-sufficient validation

For frozen Qwen2.5-3B, form one RTGA direction from held-out self-supervised native/long calibration content. Compare `+v` to its norm- and endpoint-matched mirror `-v`, plus MrPro and the strongest existing P2/E1 baselines. This single mirror test asks whether the signed construction predicts direction; the project already recommends this control (`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:123-126`). Report exact nonlinear relation residuals, short-window calibration risk, 32K/128K task outcomes, and independent holdout outcomes. Geometry scores remain diagnostics.

For scratch, compare EVQ-regularized joint training to the same architecture/data/budget with the RTGA/MGDA frequency update but no frozen compatibility targets, plus deformation-matched exponential and Cosh. Evaluate in-window capability before extrapolation. This separates representation prior from frozen compatibility instead of claiming one formula is optimal in both regimes.

## Claim boundary

The QP has an actual constructive solution under stated finite interaction targets and yields the intended native-prefix/common-retimed-suffix structure in the separated case. It does **not** prove that those targets improve language-model capability. The audited data establish that allocation matters and rule out several sufficient geometry objectives; they do not yet identify universal interaction weights or a universal bridge width. Those must be measured and validated by the mirror/holdout comparison.

