# Astra01 — Signal and interference, rather than curve coordinates, unify the allocation problem

Status: mathematical derivation and CPU checks complete; no model run or GPU use. All 89 assigned files plus the full MrRoPE Markdown and EVQ theory note were ingested. Historical reports are evidence of their stated analyses, not authoritative scientific verdicts. In particular, old declarations of an airtight universal non-identifiability theorem or a permanent prohibition on new tables are not adopted here.

## 1. Decisive distinction in the primary sources

EVQ defines the exact cosine kernel

\[
K(x,y)=\mathbb E_{\Delta\sim D}\cos(\omega(x)\Delta)\cos(\omega(y)\Delta),\quad \omega(x)=b^{-x}.
\]

Consequently its quadratic energy is exactly

\[
\langle\rho,K\rho\rangle=\mathbb E_D B_\rho(\Delta)^2,
\qquad B_\rho(\Delta)=\int\cos(\omega(x)\Delta)\rho(x)dx.
\]

This is a squared aggregate response, not a generic measure of useful positional signal. MrRoPE §4.4 instead invokes positive \(B_\theta(\Delta)=\sum_k\cos(\theta_k\Delta)\) as an advantage for similar tokens over a random token. It seeks to postpone the first zero of that coherent signal. Suppressing the square of this response and keeping this response positive are different objectives.

Sources: `docs/theory/EVQ_COSH_THEORY.tex:87–114,162–174,225–245`; `/Users/misaya.yanghejazfs.com.au/Downloads/RoPE_Papers/Markdown/5551_MrRoPE_Mixed_radix_Rotary.md:782–812`. MrPro's arithmetic progressive increments are explicitly assumed at lines 420–440; they are not derived from the first-zero analysis. Appendix B also reports empirical boundary selection, including Qwen's (23,40), rather than a universally derived boundary.

A same-support counterexample removes possible coordinate excuses. At lag \(\Delta=2\pi\), use three frequencies

\[
A=(1,1/2,1/4),\qquad B=(1,0.9,1/4).
\]

Both are positive, strictly ordered, and share endpoints. Their aggregate cosines are \(B_A=0\) and \(B_B=1.8090169944\), so the exact point-lag EVQ quadratic prefers A: 0 versus 3.2725424859.

* For a matched-content key at this remote lag and a zero-mean Gaussian wrong-key score with standard deviation \(\sqrt3\), the mean margin is \(B\). Pairwise error is 0.5 under A and 0.1481417538 under B.
* For a zero-lag positional target against a same-content competitor displaced by this lag, with the same Gaussian score uncertainty, the mean margin is \(3-B\). Pairwise error is 0.0416322583 under A and 0.2458478318 under B.

These are exact Gaussian score-model calculations, not LM outcomes. The Gaussian wrong-key construction is realizable by fixing unit-norm query pairs and using isotropic Gaussian distractor keys. In the positional comparison the Gaussian term is added distractor uncertainty. Changing the semantic role of the same lag reverses the correct ordering. This is the central reason a single unsigned cosine energy cannot be a universal allocation objective.

## 2. A common physical object: role-conditioned score margins

For one query, one required source and one competing key, write the exact rotary contribution at log frequency x as

\[
Z_e(x)=\operatorname{Re}\{A_{+,e}(x)e^{i\omega(x)\Delta_{+,e}}
-A_{-,e}(x)e^{i\omega(x)\Delta_{-,e}}\}.
\]

Here e is the example/role/lag random variable, and complex A includes both signed sine and cosine content coefficients. A source-versus-distractor label is essential: energy or an attention-distance histogram alone does not provide it. For a prescribed response field define

\[
h(x)=\mathbb E Z_e(x),\quad
C(x,y)=\operatorname{Cov}(Z_e(x),Z_e(y)).
\]

For a probability measure \(\eta\) of allocated frequencies, the normalized margin is

\[
M_\eta=\int Z_e(x)d\eta(x),\quad
\mu_\eta=\int h\,d\eta,\quad
v_\eta=\iint C(x,y)d\eta(x)d\eta(y).
\tag{1}
\]

Equation (1) is exact for this stipulated response-field model. The covariance is positive semidefinite by construction; it includes signed cross-frequency cancellation and reinforcement, not just diagonal energies. A fixed desired margin threshold a can be handled by replacing h with h−a because \(\eta(I)=1\). The constant normalization by K does not change score ordering; it is bookkeeping, not a gain intervention.

For \(\mu>0\), the common dimensionless quantity is \(t=\mu/\sqrt v\). It has an interpretable consequence:

* If the margin itself is Gaussian, \(\Pr(M\le0)=\Phi(-t)\) exactly.
* Without a distributional assumption, Cantelli gives \(\Pr(M\le0)\le v/(v+\mu^2)=1/(1+t^2)\). To prove it, apply Markov to \((M-\mu-a)^2\) on \(M-\mu\le-\mu\) and optimize a, or use the one-sided variance inequality directly.
* When \(v=0,\mu>0\), the margin is positive almost surely. When \(\mu\le0\), neither the positive-mean guarantee nor squared-SNR maximization is appropriate.
* Gaussianity conditional on a lag does not imply Gaussianity after mixing lags. Use \(\mathbb E_\Delta\Phi(-t_\Delta)\) for the conditional Gaussian model, or Cantelli on unconditional moments. Do not replace it silently by \(\Phi(-\mu/\sigma)\).
* For multiple distractors, a union bound sums the pairwise bounds. It can be loose, but positive mean against one random key is plainly insufficient against 128K competitors. Multi-key source identity requires its own contrast classes.

This object explains both intuitions: local positional discrimination needs low sidelobes at wrong lags, while remote content retrieval needs a positive target response at the correct long lag. The covariance controls how reliably the two remain distinguishable.

## 3. Constructive continuum allocation law

Consider the equal-channel population-design approximation in which the response field h,C is prescribed independently of the allocation. Let C be a positive coercive covariance operator on the chosen function space, and require a nonnegative density. Maximize positive standardized margin:

\[
\max_{\rho\ge0,\,\int\rho=1,\,\langle h,\rho\rangle>0}
\frac{\langle h,\rho\rangle}{\sqrt{\langle\rho,C\rho\rangle}}.
\tag{2}
\]

Homogeneity converts (2) into the convex problem

\[
\min_{w\ge0,\,\langle h,w\rangle=1}\frac12\langle w,Cw\rangle,
\qquad \rho=w/\int w.
\tag{3}
\]

If \(C^{-1}h\ge0\) and its integral is positive, the exact law is

\[
\boxed{\rho^*(x)=\frac{[C^{-1}h](x)}{\int[C^{-1}h](y)dy}}.
\tag{4}
\]

Cauchy–Schwarz in the C inner product proves optimality and gives \(t_*^2=\langle h,C^{-1}h\rangle\). If the inverse has negative components, clipping it is generally wrong. The actual positive active-set law is

\[
Cw\ge\lambda h,\quad w\ge0,\quad
w(Cw-\lambda h)=0,\quad\langle h,w\rangle=1.
\tag{5}
\]

The sign follows from the multiplier of the nonnegativity constraint. On the occupied frequencies marginal covariance cost equals scaled useful signal; outside that set covariance cost is at least as large. In a finite resolution model, solve the linear system on its active set and check the inequalities off that set. This is a generalized matched-filter allocation law, a classical mathematical structure, not a claimed new optimizer. The established lineage includes Capon's 1969 *High-Resolution Frequency-Wavenumber Spectrum Analysis* (DOI 10.1109/PROC.1969.7278; [original report record and abstract](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD696880.xhtml), checked in this session; full Capon paper not ingested). Its substantive contribution here is identifying the correct role-conditioned h,C and the assumptions that justify allocating channel count with it. It is not an arbitrary LM loss plus a regularizer or a local Taylor-gradient QP.

If several required role/lag classes r matter, use their own \(h_r,C_r\) and maximize the smallest positive standardized margin. For a fixed proposed t, the constraints

\[
\|C_r^{1/2}\rho\|\le\langle h_r,\rho\rangle/t
\]

are convex together with mass and nonnegativity. Feasibility bisection yields the global optimum of this continuum surrogate, without an arbitrary candidate-curve grid. A fixed protected part of the frequency measure contributes fixed mean and covariance cross terms; the norm representation remains affine in the remaining measure and the same convex feasibility formulation applies. One closed inverse is no longer generally available. The risk thresholds express required discrimination, rather than a tunable smoothing preference.

## 4. Precisely how EVQ is recovered, and what is additional

First use the exact EVQ kernel. A concrete nuisance model is

\[
Z_e(x)=h_0+\xi_e\cos(\omega(x)\Delta_e),\quad
\mathbb E\xi=0,\quad\mathbb E\xi^2=1,
\]

with ξ independent of Δ. The protected target contribution is constant across frequency; the wrong-key nuisance shares a random signed amplitude across the frequency field. Then h=h0 and C=K_exact. Maximizing standardized margin exactly minimizes EVQ's quadratic energy. This is a restricted model of a protected target against nuisance sidelobes, not a general model of a remote content source. More generally h constant suffices; no Gaussian assumption is necessary for the Cantelli interpretation.

Now assume the additional broadband covariance approximation used by EVQ,

\[
C_{\rm app}=\alpha I+\beta G,\qquad G(x,y)=\min(x,y),\quad x\in[0,1].
\]

The constant-signal law (4) solves \(\alpha\rho+\beta G\rho=\lambda\). Differentiating twice and using \((G\rho)''=-\rho\) gives

\[
\rho''-\tau^2\rho=0,\quad\rho'(1)=0,\quad\int\rho=1,
\qquad \tau^2=\beta/\alpha,
\]

and therefore

\[
\rho^*(x)=\tau\cosh(\tau(1-x))/\sinh\tau.
\]

Equivalently let \(T(x)=\int_x^1\rho(y)dy\). The variance action is

\[
v=\alpha\int(T')^2+\beta\int T^2,
\quad T(0)=1,T(1)=0.
\]

The screened harmonic solution is \(T(x)=\sinh(\tau(1-x))/\sinh\tau\). This identifies what the two EVQ terms mean under the stated nuisance covariance, rather than simply renaming its quantile curve. For nonconstant signal, on an active interval the equation becomes

\[
\alpha\rho''-\beta\rho=\lambda h'',\qquad
\alpha\rho'(1)=\lambda h'(1),
\]

with active-set/free-boundary conditions when positivity binds. Thus keeping the coherent signal produces a forced allocation rather than automatically another Cosh curve. The coefficients and forcing must come from the specified role-conditioned population; inventing them to recover a desired profile is circular.

Three limitations are unavoidable:

1. The exact oscillatory K is not automatically the Brownian-plus-identity C. Historical independent calculations found exact-kernel optima with truncated densities and collision-optimal strengths outside the trained basin (`rebuttal/STRONG_MODEL_THEORY_VERDICT_20260720.md:27–40,50–75,194–223`; the assigned numerics script independently implements those calculations). This report does not restore those withdrawn equivalences.
2. \(\tau=\sqrt{\beta/\alpha}\) is the covariance ratio in this model. It is not the empirical \(d/\sqrt L\) rule unless that ratio is separately demonstrated. The historical audit documents a large mismatch and normalization dependence.
3. Neither the protected-constant-signal assumption nor the frequency-stationary covariance assumption follows from generic RoPE. The law is testable precisely because changing target lag/role changes h and can reverse its prediction.

## 5. White noise is not iid channel noise

The \(\alpha I\) operator needs a precise interpretation. A generalized white-noise field W in frequency space has

\[
\operatorname{Var}\!\left(\int\rho(x)dW(x)\right)=\alpha\int\rho^2.
\]

This is a frequency-field integral, not K independent noises attached to K channels. Evaluating white noise at atomic frequencies is undefined. If K independent channels each have conditional variance v(x), their average instead contributes

\[
\frac1{K^2}\sum_k v(x_k)\simeq\frac1K\int v(x)\rho(x)dx,
\]

which is constant v0/K when v is constant. It cannot generate \(\alpha\int\rho^2\).

A realizable finite-resolution alternative uses the PSD triangular covariance

\[
k_\epsilon(x-y)=\epsilon^{-1}(1-|x-y|/\epsilon)_+,
\quad C_\epsilon=\alpha k_\epsilon+\beta\min.
\]

This is a shared frequency-local nuisance field, with the triangular kernel coming from overlapping box filters of white noise. For smooth densities and \(\epsilon\to0\), its quadratic form tends to the EVQ action. For K equal atoms its diagonal contribution is \(\alpha/(K\epsilon)\): taking \(\epsilon\to0\) at fixed K diverges. The finite-channel and white-noise limits do not commute. A density approximation needs resolved ridge width, roughly many channels per correlation width, not just a fitted α.

An equivalent coarse-bin model has bin width ε, shared independent bin noise variance α/ε, and bin occupancy p_r. Its variance is \(\alpha\sum_r p_r^2/\epsilon\). Quantized channel counts n_r implement \(p_r=n_r/K\) exactly at bin level. This supplies a clear interpretation, but whether real nuisance coefficients have this shared-bin structure is an empirical question. It must not be attributed to generic independent channel noise. The smooth-kernel finite-K guarantee below does not apply directly to an ideal δ kernel.

## 6. Equal-channel realization and a quantitative finite-K bound

The law (4) is interpreted as a **sampling density of frequency locations**. It is not deployed as amplitudes multiplying fixed channels. On interval I=[a,b] of width R, take

\[
x_k=F^{-1}((k+1/2)/K),\quad
\eta_K=K^{-1}\sum_{k=0}^{K-1}\delta_{x_k},\quad
\omega_k=e^{-x_k}
\]

(or use \(b^{-x_k}\) when x is the normalized exponent). Every pair retains the same multiplicative contribution; only locations and their counts change. For any probability law on I, the midpoint-quantile coupling gives

\[
W_1(\eta_K,\rho dx)\le R/(2K).
\tag{6}
\]

Proof: partition quantile u into K equal cells, couple Q(u) to its cell midpoint Q. On each half-cell, monotonicity bounds the integrated distance by half the cell width times the corresponding endpoint variation. Sum the variations to at most R. No positive lower bound on ρ is needed. For endpoint-pinned equal weights at \(u_k=k/(K-1)\), the empirical CDF differs by at most 1/K, giving the safe bound \(W_1\le R/K\). This is a different quantization convention from EVQ's original k/K; it must be declared explicitly.

Suppose h is L_h-Lipschitz, C is L_C-Lipschitz in each argument, and the same response field describes the allocated channels. Put d=W1. Then

\[
|\mu_K-\mu|\le L_h d=:e_\mu,
\qquad |v_K-v|\le2L_C d=:e_v.
\tag{7}
\]

The second statement follows by replacing one marginal of the product measure at a time. If \(\mu>e_\mu\),

\[
\boxed{t_K\ge\frac{\mu-e_\mu}{\sqrt{v+e_v}}}.
\tag{8}
\]

This supplies the finite-K Cantelli upper bound

\[
\Pr(M_K\le0)\le
\frac{v+e_v}{v+e_v+(\mu-e_\mu)^2},
\]

and the corresponding Gaussian bound when that finite margin is Gaussian. If \(v>e_v\), the reverse ratio using \(\mu+e_\mu\) and \(v-e_v\) bounds t_K above. The same bounds hold per contrast class.

This is a genuine equal-count quantile realization bound, but it can be weak at K=64 and 128K: phase derivatives grow with lag, and the narrow-ridge covariance has \(L_C\) of order \(\alpha/\epsilon^2+\beta\). No small error is claimed without evaluating the actual constants or direct finite-K moments. Independent variance v(x)/K adds a separate quadrature error at most \(L_v d/K\); it does not change into a density-squared term.

At fixed finite K the unconstrained optimal spectral measure can have atoms. Repeated quantiles then duplicate frequencies. Enforcing strict spacing, maximum occupancy, or exact endpoints is an additional architecture constraint, not something guaranteed by the inverse-C law. Reoptimizing under declared physical constraints is legitimate; silently spreading atoms for a prettier table is not.

## 7. Training from scratch versus a frozen labeled checkpoint

The same exact object can be written on the product space (slot label k, candidate frequency x):

\[
Z_{k,e}(x),\quad h_k(x),\quad C_{k\ell}(x,y),\qquad
\eta_\Omega=K^{-1}\sum_k\delta_{(k,x_k)}.
\]

Equation (1) then remains exact for frozen incoming activations, with each slot having exactly 1/K mass. This is the appropriate finite labeled design. A density on x alone discards the learned slot association. The label constraint prevents replacing one useful trained slot with multiple nominally equivalent frequencies. The frozen labeled optimization and calibration details are being handled independently by Astra03.

From scratch, symmetry of initialization makes channel-population design a plausible modeling approximation, but it does not prove that the **trained** h,C are independent of the chosen allocation. After co-adaptation they can change with ρ. Thus (4) is a constructive design rule for a prescribed role-response model and a candidate initial spectrum; it is not a closed-form solution of deep-model training. The actual from-scratch objective remains a training-algorithm-dependent risk, and the trained-table crossings are direct evidence against transplanting its response law to another allocation.

For the frozen Qwen2.5-3B target, keep actual fast/slow anchors and the labeled slots. Native-window Q/K measurements can estimate signed short-role coefficients and finite-phase replays on the same incoming states. They cannot identify all long-context h,C from native-only observations. Two response families can agree for every lag/history up to W and have opposite useful-source coefficient sign on histories beyond W; every native-only selector then sees the same input yet the long-optimal allocations differ. Full known weights plus an actual long forward can of course distinguish them. The obstruction concerns inference from native-only response samples, not computational unknowability of a known network.

Rephasing a short activation cache to 128K is exact for that cache's local rotary contraction. It does not reproduce the preceding layers' content formation under a 128K history. Any extrapolation of h,C therefore needs an explicit stationarity/transport assumption, independent target-range evidence, or a bound on the resulting discrepancy. The native-window test in the assigned `tests/test_native_windows.py:17–33` proves a cache intervention preserves later window formation under its special construction, not equivalence to full contiguous long prefill.

## 8. Why MrPro is a heuristic member rather than another exact optimizer

MrPro enforces native fast-band behavior, full low-band scale, and arithmetic radix increments. In its simplified similar-token model, slow content carriers preserve positive remote h; avoiding excessive early compression preserves fast local discrimination. Those are sensible competing requirements in (1)–(3). But neither positive B nor a first-zero endpoint selects the exact arithmetic increment profile. They provide no unique covariance C, no role-conditioned h across multi-key tasks, and no objective proving quadratic cumulative compression optimal.

To make Pro exactly solve (4), one could manufacture h=Cρ_Pro. That would be an inverse-optimal-control identity with no predictive content; this report explicitly rejects it. Likewise an arbitrary stiffness chosen proportional to inverse slot index can manufacture its increments but is not derived from RoPE. The useful unification is **one discriminative margin problem with different signal and nuisance regimes**, yielding an actual inverse-covariance allocation law under stated conditions, and exposing what MrPro leaves heuristic. It is not a theorem that both published formulas solve the same optimization.

## 9. Evidence checks, CPU checks, and the next scientific decision

The source corpus supplies several relevant limits:

* Fixed-support from-scratch allocation is causally active, while support retargeting reverses the ranking: `paper-2027/sections/02_identification.tex:4–24`.
* Trained table crossings directly expose learned basis dependence: `paper-2027/sections/03_findings.tex:14–20`.
* BM's smoother increments raise every strict-middle compression exponent, so they do not preserve all native damage budgets: `docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md:38–61`. Its OLMo benefit and Qwen long loss are both recorded in `paper-2027/sections/04_mature.tex:136–158`. Smoothness alone cannot supply h.
* Q/K adaptation can improve NLL without improving generated capability: `paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md:18–30,47–99`.
* The exact known generic MGDA calculation is explicitly synthetic and not novel in `docs/research/ROPE_GENERAL_ALLOCATION_CPU_20260907.json:45–67`; this proposal does not rebrand it.
* Older review assertions of maximal risk divergence from low-dimensional metrics do not constitute a universal theorem about all possible informative frequency rules. The useful obstruction is missing role/label/learned-response information, which our counterexample exhibits directly.

CPU checks performed in this session:

1. The three-frequency sign counterexample above, including both Gaussian errors.
2. A separate 800-point midpoint discretization of \(I+4G\), direct linear solve and normalization, reproduced the analytic τ=2 Cosh density with maximum error 2.986e−7.
3. For a smooth finite covariance \(C(x,y)=0.3e^{-|x-y|/0.15}+\min(x,y)\), h=1+0.4x, and a Cosh density, K=16/64 equal quantiles gave mean errors 0.000337/0.000255 and variance errors 0.001333/0.000503 against dense quadrature. Conservative analytic bounds were 0.0125/0.003125 for mean and 0.1875/0.046875 for variance. Dense numerical quadrature itself is approximate; these checks validate formulas, not model quality.

Decision consequence: the next allocation should be selected by **positive, role-conditioned source-versus-distractor margins and their covariance**, with frozen slot identity preserved and native retention treated as a real constraint. For a population design, (4) or its active-set version gives the actual frequency-count density and quantiles; for Qwen frozen deployment, use the labeled finite phase version, not the density quantile table. First test whether the prescribed margin moments distinguish the already completed MrPro / P2 / BM / Smooth_MrBudget outcomes on independent examples. A rule that fails this existing behavioral contrast has not earned a new frequency proposal merely by improving its own objective. No successful Qwen128K allocation, universal profile, or task-success theorem is claimed by the present derivation.
