# Astra03 — finite-scale allocation that preserves labeled computations

## Result

A concrete frozen allocation criterion is available once the checkpoint's **signed source-versus-distractor coefficient distribution** is specified. Optimize a lower bound on the standardized source margin at the actual finite target lags, with native-operation probability requirements as hard constraints. The variables are the existing labeled frequency slots; means and covariances retain slot identity. There is no geometric utility bonus, quadratic native tether, or arbitrary mixture coefficient.

This is a conditional allocation algorithm, not a universal table or a demonstrated Qwen improvement. Its conditional guarantee is a genuine probability bound for selecting the intended source. A complete-answer guarantee requires an additional downstream-readout condition, stated below. The empirical coefficients needed to emit a defensible new Qwen table have not been measured in this subtask. The prior Q/K projection-matrix Gram is not that measurement.

The principal mechanism is **signal and interference have different content conditioning**. A response that is harmful as a coherent distractor can be useful as matched-content signal. EVQ's collision quadratic is obtainable as an interference covariance under exchangeable co-adaptation assumptions. MrRoPE's positive matched-content response concerns the signal mean. It is invalid to minimize one aggregate response in one regime and maximize it in another without specifying which population generates it. Astra01 owns the continuum/co-adaptation derivation; this report supplies the frozen finite-slot rule and its solver.

## 1. Evidence that the rule must respect

All 44 assigned files, totaling 1,107,664 bytes, were read in full. A concatenation of 1,098,005 characters was delivered in 37 contiguous pages of at most 30,000 characters. Two initial oversized outputs were truncated and then superseded by those complete pages. The coverage JSON gives each file's hash, line count, and exact page coverage.

Three findings matter for the theory:

1. The manuscript records Geo-trained/Geo-runtime versus Geo-trained/Cosh-runtime PPL of 7.14 versus 76.20, while positional rank increases from 4.57 to 12.54. The opposite weights-by-table crossing also fails. See `paper-2027/DOCUMENT_TEXT_MAP.md:1795–1864`; the 151.9M replication starts at line 2515. This supports preserving the learned association between coefficients and slots. It does not identify which slots have which semantic roles.
2. `docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:145–203` gives Smooth/MrPro remote weak-subspace energy 0.049435/0.235928 and native-range unweighted distortion 35.3793/42.5180. The same file, lines 215–278, reports lower Smooth Q/K operator MSE at every listed lag range and lower weighted weak-response exposure in every layer at all three cutoffs. The explicit premise is independent unit-second-moment input vectors; these are projection-weight calculations, not the checkpoint's content-conditioned activation distribution.
3. The fully read `results/nongeometric_screen_20260909/development_summary.md:1–31` reports Smooth at −9.79 percentage points versus the identical historical MrPro inputs at 128K, with 36 combined short/long rows. E1 slot28 is +5.21 points on that development panel; P2 is +3.54 long but −14.31 short. These are development observations, not independent confirmation, and I did not re-audit the generation JSONL. They prohibit treating any positive combination of the audited unsigned geometry costs as a sufficient selector.

The assigned historical evaluation code also repeatedly distinguishes NLL-gap retrieval from autoregressive correctness. The manuscript's 750M comparison explicitly has both methods at 100% NLL-gap retrieval at 8K but 0%/77.5% strict autoregressive exact match. A head-routing theorem must not be renamed an answer-generation theorem.

## 2. The exact labeled bilinear object

For a rotary pair j, take the signed relative position d=key position minus query position. Let

\[
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix},\quad
A_j=q_{j0}k_{j0}+q_{j1}k_{j1},\quad
B_j=q_{j1}k_{j0}-q_{j0}k_{j1}.
\]

Then q_jᵀR(ν_jd)k_j=A_j cos(ν_jd)+B_j sin(ν_jd). The head logit has scale γ=g²/√d_head when g multiplies both cosine/sine tables. Nonrotary contributions can be included as one additional constant feature.

For a labeled operation r, identify query, intended source +, distractor −, and their signed distances d_r+ and d_r−. Stack

\[
c_r=\gamma(A^+_{r1},B^+_{r1},A^-_{r1},B^-_{r1},\ldots),
\]
\[
f_r(x)=\big(\cos u^+_{r1},\sin u^+_{r1},-\cos u^-_{r1},-\sin u^-_{r1},\ldots\big),
\quad u^\pm_{rj}=d_r^\pm\omega_j e^{-x_j}.
\]

The logit contrast is exactly Z_r(x)=c_rᵀf_r(x). Here x_j=log(ω_j/ν_j) is a labeled displacement. A table-only permutation changes f without permuting c, generally changing Z. A joint permutation of complete slots in both objects leaves Z unchanged. This is the relevant label-preservation property, not merely preserving frequency order or the multiset.

In a complete transformer, c_r itself depends on the installed table through previous layers and prefixes. Freezing c_r for every x is an assumption, not an identity. The next section makes the required replacement explicit.

## 3. An explicit generative assumption and a nonvacuous guarantee

For each operation type and lag cell r, assume that under all admissible tables in a declared region:

* its coefficient mean obeys ||E c_r−μ_r||₂≤ε_r;
* its covariance is bounded in PSD order by Σ_r=L_rL_rᵀ;
* desired source labels continue to denote correct evidence after the declared content/layout transport.

These are **conditional moment-envelope assumptions**, not claims that native samples establish long-context invariance. They permit correlated slots, correlated sine/cosine coefficients, and source/distractor correlation. They can include upstream changes if the envelope is established for them. A source-native coefficient sample alone does not establish that envelope.

Define

\[
\underline M_r(x)=\mu_r^Tf_r(x)-\epsilon_r\|f_r(x)\|_2,
\quad S_r(x)=\|L_r^Tf_r(x)\|_2,
\quad z_r(x)=\frac{\underline M_r(x)-\eta_r}{S_r(x)}.
\]

η_r is an operational logit-margin requirement, rather than a preference weight. For hard source ranking, η_r=0. For softmax source mass at least 1−a among M keys, the sufficient pairwise margin is

\[
\eta_r=\log\frac{(M-1)(1-a)}a.
\]

The ratios below presume S_r>0. At zero variance, treat a strictly positive certified numerator as deterministic success, a strictly negative numerator as failure, and equality as unresolved because ties require a rule. Native constraints in the conic solver use nonnegative required reliability; a negative native baseline does not qualify an operation for a positive-reliability guarantee.

For positive z, Cantelli gives the distribution-free guarantee

\[
\Pr[Z_r\le\eta_r]\le\frac1{1+z_r^2}.
\]

Under the stronger centered-coefficient sub-Gaussian MGF envelope

\[
E\exp(t^T(c_r-Ec_r))\le\exp(t^T\Sigma_rt/2),
\]

the bound improves to exp(−z_r²/2). If c_r is exactly Gaussian with known moments, the exact pairwise failure probability is Φ(−z_r), with ε_r=0. Gaussian coefficients are an explicit statistical model; arbitrary products of Gaussian Q and K need not be Gaussian. Conditioning on a fixed query and Gaussian keys is one exact realization.

For M−1 distractors, apply a union bound. At z≥6 and M−1=1000, the sub-Gaussian bound guarantees source ranking with probability at least 1−1000 exp(−18)>0.9999847. Thus the guarantee is not formally true but numerically vacuous by construction. Whether Qwen's measured z reaches an informative value is unknown. Cantelli is often too conservative at 128K; that must be reported rather than replaced by an unverified Gaussian tail.

For a full generated answer, suppose additionally that conditioned on the correct prefix, each required routing event writes a value whose downstream decoder preserves the correct next token whenever the event's margin threshold holds. If T required events have failure bounds b_t, induction on correct prefixes and a union bound give complete-output success at least 1−Σ_t b_t. This is a restricted copying/readout theorem. General language-model values, MLPs, and EOS behavior do not satisfy that premise automatically.

## 4. The finite allocation rule

For a predeclared target-role set R_L and native-role set R_N, solve

\[
\boxed{\max_{x\in\mathcal F}\ \min_{r\in R_L}z_r(x)
\quad\text{subject to }z_r(x)\ge\bar z_r\ (r\in R_N).}
\tag{F}
\]

The native thresholds can equal the certified native-table values when requiring no degradation of the same bound, or be derived from an explicitly requested failure tolerance. This is a max-min reliability objective with hard requirements; no mixing coefficient trades native against target losses. Its output can be infeasible, which exposes incompatibility among the declared requirements.

The general feasible set fixes sampled endpoints and enforces ω_je^(−x_j)≥ω_(j+1)e^(−x_(j+1)). In log variables this is linear. Optional x_j∈[0,log S] excludes frequency acceleration. The presently evidenced Qwen face uses x_j=0 for j≤23 and x_j=log4 for j≥40; the 16 interior entries j=24,…,39 are variables. Keeping that face is a scoped continuation of MrPro evidence, not a universal 32-turn/one-turn theorem.

In adjacent mixed-radix coordinates e_j=x_(j+1)−x_j, total added span is Σe_j=log4 on the transition. Uniform and progressive MrRoPE are particular budget assignments. Problem (F) chooses that assignment by the labeled computations' reliability. It does **not** derive MrPro's quadratic cumulative profile from arbitrary data. Such an exact-profile derivation requires additional assumptions on the means/covariances; reverse-engineering them to make MrPro optimal would be circular.

The mean/covariance criterion genuinely differs from the previous scalar proxies. μ has signs and source labels. Σ retains coherent nuisance relations rather than just projection norms. The relevant reliability can decrease when both an unsigned distortion and an unsigned weak-subspace energy decrease.

## 5. A practical convex inner solver with finite-change certification

An unspecified nonconvex argmax would not be a usable rule. Here is a concrete solver which needs only stored coefficient statistics and elementary CPU arrays. It makes no global-optimum claim.

At an installed reference x⁰, consider a box |v_j|≤r_j and write

\[
\widehat f_r(v)=f_r(x^0)+J_rv.
\]

For one sine/cosine pair with u=dωe^(−x), the derivative is (u sin u,−u cos u). Its second derivative has Euclidean norm √(u²+u⁴). Let U_rj±=|d_r±|ω_j exp(−x_j⁰+r_j), and

\[
B_{rj}=\sqrt{(U^+_{rj})^2+(U^+_{rj})^4+(U^-_{rj})^2+(U^-_{rj})^4},
\quad Q_r(v)=\tfrac12\sum_j B_{rj}v_j^2.
\]

Taylor's integral remainder and the triangle inequality give the global-on-this-box bound

\[
\|f_r(x^0+v)-\widehat f_r(v)\|_2\le Q_r(v).
\]

For a fixed candidate reliability t≥0, impose the following constraint for each target cell:

\[
\boxed{\mu_r^T\widehat f_r(v)-\eta_r\ \ge
\epsilon_r\|\widehat f_r(v)\|_2
+t\|L_r^T\widehat f_r(v)\|_2
+(\|\mu_r\|_2+\epsilon_r+t\|L_r\|_2)Q_r(v).}
\tag{I}
\]

Use t=bar z_r for a native cell. The right side is convex and the left side affine for fixed t. Together with x⁰+v∈F and the box, this is a convex quadratically constrained conic feasibility problem. Epigraphs for the norms and diagonal quadratic make it representable by second-order cones. Bisection over t solves the maximal certified reliability inside this local inner approximation.

Why it certifies the finite table: the first remainder term lower-bounds the exact mean, the ε terms cover mean-estimation/transport uncertainty, and the final t||L||Q term upper-bounds the exact standard deviation through the triangle inequality. Hence (I) implies underline M_r(x⁰+v)−η_r≥tS_r(x⁰+v), using exact phases, even when S=4. No infinitesimal statement is being substituted for a finite update.

Algorithm:

1. Start from the recorded MrPro table, with amplitude fixed. Calculate exact z_r and native feasibility under the declared envelopes.
2. Choose radii from the remainder resolution needed by the current positive margins, not a candidate grid. For example ensure each maximal Q_r on the box consumes at most one quarter of the current positive surplus; halve radii if necessary. This is a numerical approximation tolerance, not a new scientific acceptance threshold.
3. Bisection-solve (I), with the previous certified t as a feasible lower bound. Select the minimum-Euclidean-norm feasible v only to resolve solver ties; this is lexicographic, not a utility regularizer.
4. Evaluate the exact trigonometric margins at x⁰+v. Record exact and certified t. Replace the reference and repeat while a strictly positive certified improvement is obtained.
5. Freeze one resulting table. If the starting point is not feasible, first solve the same conic constraints for minimal maximum violation. Do not claim safety until violation is zero. If no feasible positive t is found, return the incompatibility/missing-envelope result rather than fabricating a useful frequency curve.

If v=0 is feasible at a current t, every stage has a feasible incumbent. The certified target reliability is therefore nondecreasing while all stated native bounds remain satisfied. Local convergence, global optimality, and actual language-model task improvement are distinct claims. This algorithm establishes the first monotonicity property; it does not assert the other two.

One-dimensional special case is even more explicit. If a learned operation has mean A cos(νD−θ), constant nuisance standard deviation σ, and threshold η, achieving reliability t requires

\[
\nu\in\bigcup_{n\in\mathbb Z}
\left[\frac{\theta+2\pi n-\arccos((\eta+t\sigma)/A)}D,
\frac{\theta+2\pi n+\arccos((\eta+t\sigma)/A)}D\right],
\]

when the arccos argument lies in [−1,1]. Intersect these finite branch intervals with the positive frequency range and native-role intervals, then bisect t. This exact finite-scale allocation illustrates why individual first-zero horizons are not universal: a slot's correct content phase θ and admissible phase branch matter.

## 6. Measuring the inputs without relabeling a geometry score

Use a benchmark-isolated finite collection of qualified source-dependent operations. Existing qualified natural transport assets are a possible source, but their current availability was not checked. In the assigned `scripts/train/train_single_table_native_constrained.py`, `read_tasks`, `validate_answer_worlds`, and `evaluate_tasks` explicitly preserve lawful answers across layouts and distinguish original-native qualification, full trajectory, and EOS. Those are useful semantics; the script's fixed quotas and training recipe are not adopted as instructions here.

For each query, source, and matched hard distractor, collect **pre-RoPE** Q/K activations after the checkpoint's actual normalization and GQA sharing. Form the signed A/B vectors above. Keep query, source, layer, head, relation type, and actual lag labels. Estimate μ and full Σ within these cells, pooling only where exchangeability is an explicit hypothesis. Hard distractors must include content-confusable keys, not only independent random keys.

Native teacher top-attention edges may supply cheap candidate operations, but they are not automatically correct evidence. Qualify source dependence using a correct compact answer plus a source intervention/content fork. Carry unchanged lawful labels to long layouts. This is calibration supervision, possibly self-supervised when the source establishes the answer; it must not be called label-free.

For transported Q/K statistics, use either actual long forwards or declare the invariance/envelope hypothesis and test it separately. Retiming detached native activations is a useful inexpensive conditional calculation, but it cannot validate whole-network co-adaptation. Record source-template estimates, observed long drift, and uncertainty radii separately. Native-to-long distribution mismatch cannot be repaired by covariance shrinkage alone.

The one decisive diagnostic before proposing a new table is to calculate **signed mean margin and margin variance separately for the existing MrPro, Smooth, and P2 tables**, with the same operation labels. If those measurements still rank Smooth better on the failing source-dependent rows, this theory does not explain the failure at the chosen head/operation level: investigate wrong operation labels, value/readout behavior, upstream activation changes, or generation-prefix effects. Do not add more unsigned geometry terms to force the ranking.

For moment-estimation confidence, use independent calibration and validation units at the source/group level. A sample covariance is not a population PSD upper bound. Any claimed probability guarantee must state how Σ and ε cover estimation error and transport drift, or explicitly be a model-conditional guarantee with known moments. The CPU optimizer cannot create that evidence.

## 7. A concrete counterexample using the actual Smooth/Mr tables

This is an **analytic counterexample chosen to test the logic**, not a post-hoc explanation of the observed Qwen tasks and not a new deployment candidate.

The actual Qwen native slot28 frequency is ω=10^(−6·28/64)=0.0023713737056616554. From the fully read `planned_controls/p2_gap_comparison.json`, the MrPro slot28 period is 3035.3259630259545, giving ν_M=0.002070019952952862. The full queue specification `queue/0440_Smooth_MrBudget.json` gives ν_S=0.0022761875297874212.

Construct a learned template with native source lag d₀=26590 and target lag D=4d₀=106360. Its fixed labeled mean coefficients are (cos(ωd₀),sin(ωd₀)), so its expected target response is cos(νD−ωd₀). Direct CPU arithmetic gives

| Table | Native-template matched target mean |
|---|---:|
| MrPro | +0.9994600706262113 |
| Smooth | −0.9995409168024664 |

Add independent isotropic coefficient noise with standard deviation 0.1; the projected noise standard deviation remains 0.1 because cos²+sin²=1. Mr has standardized margin about +9.995 and Smooth about −9.995. Under Gaussian noise, their ranking probabilities are almost one and almost zero. Under only moments, Cantelli certifies Mr success above 99%; it gives no useful success bound for Smooth's negative mean. The same fixed full tables retain the published unsigned distortion ordering favoring Smooth.

The lag was selected by exhaustive integer arithmetic within d₀∈[8192,32767] to exhibit a maximally clear counterexample. It was not a frozen experiment prediction. The construction proves that the observed unsigned ordering is compatible with a reversed label-preserving computational ordering, including within the actual 32K→128K finite scale.

The exact one-slot optimum on the nearby phase branch n=25 is

\[
\nu^*=\frac{\omega d_0+2\pi\cdot25}{106360}
=0.0020697109769935414,
\]

or m*=0.09814685885 compared with MrPro m≈0.09803918082. It gives target mean 1 and native lag26 template mean 0.9999692420. This explicitly demonstrates an allocation calculation from declared semantic phase and noise assumptions. It is not evidence that this tiny adjustment helps Qwen, nor an endorsement of its direction over E1.

Degenerate checks:

* If every operation is pure long content matching with no local/ordering requirement, zero frequencies or full PI may win; the absence of local roles is substantive, not an optimizer bug.
* If all coefficient means vanish, no positive-margin theorem follows from reduced covariance alone.
* If the noise covariance is isotropic and independent across all sine/cosine pairs, frequency changes do not reduce its variance at fixed lag. A collision reduction then needs coherent cross-slot or lag-conditioned nuisance structure; it cannot be inferred from independent isotropic distractors.
* If a permutation leaves every μ and Σ invariant, slot labels are genuinely exchangeable in that model. Frozen label sensitivity then cannot be claimed from that model.
* If the actual table changes upstream activations outside the envelope, the probability statement no longer applies. A smaller fixed-coefficient residual does not rescue it.

## 8. EVQ/MrRoPE connection without an arbitrary regularizer

At scratch initialization with enough co-adaptation, one can assume a field of exchangeable learnable channels whose useful normalized signal is constant across allocations. If nuisance consists of independent loading noise plus shared nested slow-tail noise, the variance of the density-weighted response is

\[
\alpha\int\rho^2+\beta\int S_\rho(t)^2dt,
\quad S_\rho(t)=\int_t^1\rho.
\]

Maximizing reliability with constant positive signal is exactly minimizing that variance, and yields the EVQ Cosh density under the manuscript's assumptions. This explains the criterion as a generative covariance model; it does not prove that real scratch-trained transformers have that covariance. The delta term needs a finite-resolution/white-noise interpretation and must not be evaluated on atomic quantiles directly. Astra01 treats that issue.

After freezing, useful signal is μᵀf(x), not a constant. Matched query/key content can give positive mean cosine response over a desired remote lag, while distractors contribute a different covariance. The first-zero surrogate discards the slot-specific coefficient means, quadrature phases, hard distractor structure, and downstream values. It is recovered only under restrictive aligned equal-mean assumptions. MrRoPE provides an important structured feasible family and empirical baseline; its exact progressive profile does not follow universally from this reliability principle.

Thus the substantive shared principle is **allocate finite positional channels to preserve useful conditional signal relative to interference, accounting for whether their content coefficients are allowed to co-adapt**. Scratch EVQ is a special constant-signal covariance model. Frozen allocation is the labeled finite transport problem (F), solved by (I). This connection changes the objective, measurements, and solver; it is more than putting both methods into log-frequency coordinates.

## 9. Check of the root's softmax information-projection identity

Fix a frequency table and fixed logit features f_j. Let p_c(j)=exp(cᵀf_j+b_j−A(c)). If a finite unconstrained c* minimizes KL(p||p_c), then E_p f=E_p* f and

\[
\mathrm{KL}(p\|p_{c_0})=
\mathrm{KL}(p\|p_{c_*})+
\mathrm{KL}(p_{c_*}\|p_{c_0}).
\]

Indeed, subtracting the right side from the left leaves (c*−c₀)ᵀ(E_p f−E_p* f)=0. No feature linear independence is needed, but existence of a finite minimizer and fixed offsets/support is needed. Under a convex constraint on c with c₀ feasible, the equality becomes the corresponding ≥ Pythagorean inequality by the first-order optimality condition. In separable cases the infimum may occur only at infinite coefficients; a limiting argument is necessary.

This exactly separates representational fit and fixed-coefficient compatibility in a softmax exponential family. It is a better decomposition than an arbitrary squared-logit fit when softmax is the actual operator. It does not by itself choose a frozen frequency table: the desired p must encode lawful source selection, and features and upstream states change with the table. A native same-position p at long length is not automatically the correct transported target. Use the identity as explanation or a fit diagnostic, not as a replacement for problem (F)'s signed target.

## Deliverable boundary

Produced: a labeled finite-scale optimizer, explicit convex inner implementation, a source-selection probability guarantee under stated moment/tail assumptions, a direct arithmetic counterexample using the existing Smooth/Mr frequencies, and an empirical coefficient-measurement contract. A CPU check of the Taylor remainder used 1,000 random 16-slot perturbations, signed lags −106360/−70000, radii 0.01, and seed 7303; the largest exact-remainder/bound ratio was 0.8346442448. This checks the stated numerical bound on those draws, separately from its analytic proof. No model/GPU run, runtime edit, manuscript edit, or additional agent was launched. No claim is made that the new optimizer has already emitted a validated Qwen table or explained the observed Smooth failures causally.
