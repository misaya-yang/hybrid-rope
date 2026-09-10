# Astra05: role-conditioned nonlinear modes and exact mode-selective allocation

Date: 2026-09-10. Scope: mathematical derivation and CPU arithmetic; no GPU, model run, candidate queue, source modification, or empirical success claim. All assigned full texts were ingested; see the accompanying coverage receipt. The reason-research-theory skill was applied.

## Result

EVQ's squared cosine objective and MrRoPE's positive cosine objective can occur as different terms of one **signal-versus-distractor log-partition margin**. They are not competing universal instructions for the same kernel at the same lag. A coherent match benefits from a positive mean score; zero-mean coherent nuisance incurs an exponential-moment cost governed, to second order, by its squared kernel. Crucially, independent isotropic distractor coordinates do **not** produce that squared-kernel cost: their score variance is rotation-invariant. The content assumptions are part of the unification, not dispensable details.

A constructive consequence of the mixed-mode calculation is the exact allocation

\[
\boxed{\nu(S)=(I-P_R)\omega+S^{-1}P_R\omega,}
\]

where the rows of R are measured, role-qualified integer frequency relations and P_R projects onto their row span. It stretches the selected joint modes exactly, keeps every orthogonal relation exactly, and obeys a scale-composition law. This is a finite transport, not a first-order QP. For the active second difference on slots 28–30 it changes three frequencies by approximately −0.471%, +1.169%, −0.726%, while extending that mode's period fourfold. Uniform PI changes each by −75%. This opens a genuinely different allocation direction, but no existing evidence identifies this particular mode as causally useful in Qwen. The largest bias-only coefficient at layer 27/head 8 cannot supply that identification because its normalized far mass is negligible.

## 1. The exact object before approximating a spectrum

At one fixed query/head/layer hold pre-RoPE Q/K fixed. Let d_t be the signed relative position of key t and write

\[
z_t(\nu)=b_t+\sum_j\{a_{tj}\cos(\nu_jd_t)+b_{tj}\sin(\nu_jd_t)\}.
\]

The coefficients already include the attention scale and any fixed Q/K amplitude. (The separate b_t is any nonrotary score.) For explicit useful-source and distractor sets S,D,

\[
Z_A(\nu)=\sum_{t\in A}e^{z_t(\nu)},\quad
M(\nu)=\log Z_S(\nu)-\log Z_D(\nu),\quad
p(S\mid S\cup D)=\sigma(M).
\]

No Fourier stationarity, independence, low-amplitude, or small-table-change assumption is needed for these identities. If S∪D omits other keys, this is a conditional mass; global p(S) additionally depends on every omitted key. This distinction matters for a mode with tiny global attention weight.

For an LM, useful source is not automatically the set of keys with highest attention. Its identity must come from the task's source relation or an intervention demonstrating source use. Even then M is an attention observable, not a theorem about emitted answers: V, W_O, other heads, later layers, and decoder margins remain.

The repository already implements exact explicit-span log odds in `scripts/analysis/compare_decision_traces.py:142` onward. Its warning about local replay versus whole-model attribution is correct. `scripts/analysis/native_attention_kl.py` keeps the finite trigonometric displacement and softmax normalization, which is the appropriate numerical foundation. Neither should be replaced by an uncalibrated local Hessian extrapolation.

## 2. A precise EVQ/Mr bridge, and its necessary limitation

Normalize Kν(d)=K^−1 Σ_j cos(ν_j d). Consider a stylized retrieval row with one deterministic signal score μKν(d_s), μ>0. Let distractor distances be i.i.d. from p_D, and their score be ξ_t Kν(d_t), with ξ_t∼N(0,σ²) independent across keys. This is **shared coherent nuisance within each key**: the same scalar multiplies the entire frequency sum.

Then

\[
\mathbb E e^{z_D}=\mathbb E_{d\sim p_D}\exp\{\tfrac12\sigma^2 K_\nu(d)^2\}.
\]

For N distractors, define

\[
\overline M(\nu)=\mu K_\nu(d_s)-\log N
-\log\mathbb E_{p_D}\exp\{\tfrac12\sigma^2K_\nu(d)^2\}.
\]

This is the log signal mass minus log **expected** distractor mass; it is not E log odds. With independent finite-moment distractors, the empirical partition approaches its expectation as N increases. Moreover Jensen gives E log Z_D ≤ log E Z_D, so E M ≥ Mbar for deterministic signal. This expectation inequality does not guarantee a particular realization or accuracy.

For weak coherent noise, using |K|≤1,

\[
-\overline M=\log N-\mu K_\nu(d_s)
+\tfrac12\sigma^2\mathbb E_D K_\nu(d)^2+O(\sigma^4).
\]

Thus:

* The **Mr-like term** maximizes the positive matched-content cosine mean at the signal distance. Mr's root diagnostic is discussed in the full paper's §4.4 (local markdown lines 782–795). Its actual progressive rule assumes arithmetic radix increments (§3.2.2, line 420); the positive cosine argument does not derive those increments uniquely.
* The **EVQ-like term** suppresses coherent distractor collisions under their lag measure. It is squared **after summing** the frequencies. The lag measure, content coherence and signal constraint must be specified.
* The −log N term is distractor multiplicity. Retiming phases alone does not cancel it. Gain can help only for favorable signal/distractor margins and can amplify wrong maxima.

This derivation does not turn the current Cosh surrogate into the exact optimizer of Mbar. `paper-2027/sections/03_theory.tex:74–110` explicitly chooses the density criterion α∫ρ²/2+β∫Sρ²/2 and derives its Cosh minimizer. Its α and β are not identified by the bridge above without further kernel approximation and content assumptions.

### Counterexample to the tempting generic-noise explanation

If distractor coefficients instead have independent Gaussian quadratures,

\[
z_D(d)=\sum_j[A_j\cos(\nu_jd)+B_j\sin(\nu_jd)],\quad
A_j,B_j\stackrel{iid}{\sim}N(0,\sigma^2/K),
\]

then Var z_D(d)=σ² and E exp z_D(d)=exp(σ²/2), independent of ν and d. An EVQ-style squared sum is absent. Likewise, independent isotropic Q/K vectors give a rotation-invariant score distribution. Therefore “random distractors produce squared cosine collisions” is false unless a coherent covariance structure is stated.

For general Gaussian coefficients u with mean m and covariance C,

\[
\log\mathbb E e^{u^Tx_\nu(d)}=m^Tx_\nu(d)+\tfrac12x_\nu(d)^TCx_\nu(d).
\]

The coherent rank-one covariance gives K²; isotropic C gives a constant. Intermediate and learned C produce the actual joint sums/differences, with sine phases and cross-slot covariance. Q/K projection Frobenius norms alone do not determine C under real activations.

## 3. Actual nonlinear joint modes retain roles and normalization

For a small selected rotary block B, write its per-key contribution as Σ_{j∈B} r_tj cos(ν_jd_t−φ_tj), leaving the exact rest score z_{t,−B}. Then

\[
e^{z_t}=e^{z_{t,-B}}\sum_{n\in\mathbb Z^{|B|}}
 c_{t,n}e^{i(n^T\nu_B)d_t},\qquad
c_{t,n}=\prod_{j\in B}I_{n_j}(r_{tj})e^{-in_j\varphi_{tj}}.
\]

This is valid with position-varying coefficients: it is a separate expansion for every key, not a stationary Fourier model of a natural prompt. Define the actual role transforms

\[
H_{A,n}(\kappa)=\sum_{t\in A}e^{z_{t,-B}}c_{t,n}e^{i\kappa d_t},\qquad
Z_A=\sum_nH_{A,n}(n^T\nu_B).
\]

Here weights and phases are supplied by the stored computation; no positive relation importance weights are invented. The exact mode contribution to the frequency derivative of M is

\[
\nabla_{\nu_B}M=\operatorname{Re}\sum_n n
\left\{\frac{H'_{S,n}(n^T\nu_B)}{Z_S}
-\frac{H'_{D,n}(n^T\nu_B)}{Z_D}\right\}.
\]

This derivative describes sensitivity, not a finite-change predictor. For a proposed finite transport, directly recompute Z_S and Z_D. Crucially, the same harmonic can increase or decrease the margin depending on its **signed role contrast**, content phase, and occupied positions. Magnitude |c_n| or a long period alone cannot choose whether to preserve, stretch, or suppress it.

In the stationary special case H_{A,n}=c_{A,n} Σ_{t∈A}e^{iκd_t}. For an interval this is the exact Dirichlet kernel, including aliases. If a slow harmonic dominates role differences after fast averaging, the lower-dimensional role partition—not raw mode amplitude—determines its function. An error ε_A in truncated Z_A is controlled relative to the actual positive Z_A. If |error|≤η_A Z_A with η_A<1, the induced log-margin error is at most −log(1−η_S)−log(1−η_D). This provides a numerical truncation rule; it is not a learned capability threshold.

The exact expansion and fast-averaging obstruction were already established in `docs/research/ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md:9–94`; the new step here is the role-resolved partition and explicit finite allocation below.

## 4. A finite, constructive allocation from the joint modes

Suppose source evidence establishes that relations n_1,…,n_r carry a computation in a coarse coordinate that must stretch by S; let R contain these independent rows. All other directions should be retained unless the selected relations mathematically force a change. Choose the **exact least-displacement** solution

\[
\min_\nu\|\nu-\omega\|_2^2\quad\text{subject to }R\nu=R\omega/S.
\]

This is not a local proxy fit: constraints are exact harmonic retiming, and the objective explicitly minimizes raw frequency displacement. Solving once gives

\[
P_R=R^T(RR^T)^\dagger R,\quad
\nu=(I-P_R)\omega+P_R\omega/S.
\]

Every n in row(R) has n^Tν=n^Tω/S. Every q orthogonal to that row space has q^Tν=q^Tω. Also T_R(S)T_R(U)=T_R(SU), so repeated application with the **same fixed R** has no scale-path ambiguity. A changed R or content distribution breaks that composition interpretation.

This declares what is preserved and what is stretched without assigning a free desired scale or weight to each individual frequency. It does not claim Euclidean frequency displacement is model damage; it merely selects the unique smallest intervention that realizes the specified mode transform. Native functional cost must still be evaluated exactly.

### Two-frequency envelope transport

For n=(1,−1), write c=(ω_1+ω_2)/2 and g=ω_1−ω_2. Then

\[
\nu_1=c+g/(2S),\quad \nu_2=c-g/(2S).
\]

The beat stretches while the sum/carrier is retained. One frequency moves upward. A compression-only box ν_j≤ω_j excludes this exact carrier-preserving solution. Under that box, the minimum-displacement solution is instead ν_2=ω_2, ν_1=ω_2+g/S (for ω_1>ω_2). It stretches the beat but shifts the carrier. Hence a blanket ban on any acceleration is a substantive mechanism restriction, not merely a harmless implementation convention.

### Three-frequency curvature transport

For n=(1,−2,1) and κ=ω_j−2ω_{j+1}+ω_{j+2},

\[
\nu_B=\omega_B-(1-S^{-1})\kappa(1,-2,1)/6.
\]

The block mean and its first linear slope are unchanged; curvature/second-difference frequency is divided by S. This gives a middle allocation whose changes alternate sign. It is not generic smoothing and need not be well represented by a monotone movement-exponent ramp.

CPU float64 formula-grid check, b=10^6,K=64,S=4 (not the stored FP32 initializer):

| Relation | Old period | New period | Relative slot changes | Norm(change) / norm(PI change) |
|---|---:|---:|---|---:|
| slots36−37 | 76740.5661 | 306962.2644 | −7.2809%, +9.0352% | .10690 |
| slots28−2×29+30 | 70286.2105 | 281144.8420 | −.471216%, +1.169499%, −.725638% | .01069 |

Both blocks remain strictly decreasing in this check. Adjacent untouched slots, positivity, and every finite functional cost must be checked for any full table; no general order guarantee follows from a projector. The stored Qwen period is 70285.94, a small initializer-rounding difference already documented in the source owner at lines 187–190.

Reproducer for the arithmetic above:

```python
import numpy as np
w = 1e6 ** (-np.arange(64) / 64)
for slots, n in [([36,37],[1,-1]), ([28,29,30],[1,-2,1])]:
    n = np.array(n); v = w[slots]; k = n @ v
    u = v - .75 * k * n / (n @ n)
    print(2*np.pi/k, 2*np.pi/(n@u), u/v-1,
          np.linalg.norm(u-v)/np.linalg.norm(v/4-v))
```

Uniform PI is recovered when row(R) spans the full frequency space; selective block PI is recovered when it spans a coordinate block. Thus coherent block compression and sparse relation transport are members of one exact finite family. If R has full rank, the freedom to keep carriers disappears. Significant interactions spanning the whole space can therefore force global PI, exposing a genuine local-versus-long conflict.

## 5. How to identify R without inventing relation weights

A valid construction needs a source computation and its spatial transformation. The minimal observational contract is: fixed pre-RoPE Q/K, explicit source relation S, the full competing key set D, and the source-to-target coarse coordinate mapping. On a synthetic source-use assay these are known from generation rules; on natural text they require defensible source annotations or counterfactuals. Source-only conditions in `scripts/experiments/source_only_generation_guard.py` are useful for this purpose, while attention magnitudes by themselves are not.

For each small interaction block, evaluate its exact role transforms above. Qualify a relation only when (a) it has material normalized contribution after the rest-of-row partition is included, (b) its sign/phase supports source-versus-distractor separation in source-only paired worlds, and (c) the intended coordinate extension actually calls for its retiming. A relation merely present in Q/K biases fails this contract. Use exact finite replay to compare the one resulting projector transport against its reference. This is an operational construction from observed roles, not an automatic capability guarantee. It still requires empirical role information that the supplied source-geometry and projection-weight audits do not contain.

If no relation satisfies the contract, the outcome is **no justified mixed-mode intervention**, not an arbitrary triple selected because its period is appealing. Do not pick a harmonic, cutoff, sign, or S by maximizing the already exposed 128K answers.

A decision-sufficient mechanism comparison is one role-qualified transport and one equal-size orthogonal intervention, with exact normalized role odds and then actual generated answers. An orthogonal perturbation preserves n^Tν while changing carriers; the rank-one intervention changes that relation with minimal movement. This distinguishes “the selected beat matters” from “any local table perturbation helps.” Native/local controls are required because exact harmonic retiming does not preserve all raw logits. No such model intervention was run here.

## 6. Why the existing failures constrain this construction

`docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md:151–204` reports that Smooth improves source-weak exposure and unweighted distortion relative to MrPro yet has worse 128K development outputs. Its lines 208–281 further establish that projection-operator weighting and weighted source-weak energy do not repair the ranking; Smooth is better on the latter in every layer at three cutoffs. This rules out choosing R or a table simply by reducing total source-weak energy, operator distance, or a frequency norm.

The source weak projector N is still useful as a **constraint diagnostic**: after the finite transport, evaluate tr(NG_far(ν)) and exact local feature/logit behavior. It is not an objective whose minimizer is automatically useful. A small source eigenvalue does not authorize independently moving participating channels.

The bias-only audit `results/nongeometric_screen_20260909/planned_controls/bias_harmonic_head27_8_audit.json:2–34` chooses layer27/head8 by the largest native (1,−2,1) bias coefficient, but Native has effectively all mass in the first target quarter; MrPro places only about 2.576e−8 beyond it. Smooth's mass beyond the first quarter is about 2.055e−7. Large coefficients inside a separately normalized block therefore do not establish global far attention. Local KL differences in that head may still matter, but the audit does not identify an actual far evidence pathway.

The carrier-removal pilot (`docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md`, full text) records a 73.85% decrease of its background derivative objective alongside severe long VT/UUID degradation. Preserving frequency differences while altering the carrier was not enough there. The new projector family therefore cannot be promoted solely because it preserves a selected beat. The full role partition and the competing preserved/damaged modes must be checked.

Historical agent claims of universal closure or “all vetoes watertight” in assigned handoffs are not adopted. A concentrated residual does not demonstrate alignment with a measured dominant loss-Hessian eigenvector; the supplied challenger text asserts that causal alignment without measuring it. Likewise, one-turn marginal phase coverage never proves joint phase safety. The old `rope_transport/nullband.py` opening assertion that a wrapped channel is unconditionally safe wherever moved is explicitly contradicted by mixed modes and by the later joint-trajectory section of the same file.

## 7. Training from scratch versus frozen deployment

For scratch training, the relevant objective is evaluated after learning coefficients and circuit roles under ν. Formally ν→θ*(ν)→task risk; the learned mean/covariance and active relation span can change. An allocation that reduces nuisance redundancy may free channels for useful computation; the Cosh criterion is a tractable design choice for that regime, not the solved risk. Source-window rank alone cannot identify the unseen-length optimum.

For frozen deployment, slot identities, conditional coefficients and the existing circuit are fixed initially, so mode retiming has a direct meaning. At deeper layers they still change through upstream state propagation. The projector is therefore a conditional transport hypothesis; a full-model run must test whether its preserved relation continues to represent the same function. Allowing LoRA introduces an additional learning problem and cannot retroactively validate the zero-training conditional model.

The strongest supported result is the unified role-dependent partition framework plus an exact, sparse mixed-mode transport that current independent compression ramps may exclude. Selecting its active relations for Qwen and proving any advantage over MrRoPE remain uncompleted empirical questions; neither follows from the supplied geometry or bias data.


## 8. Requested continuation: all adjacent lowest-order transition candidates

Implemented `.agents/rope_unification_20260910/code/joint_mode_candidates.py` and generated `.agents/rope_unification_20260910/joint_mode_candidates.json`. This continuation is the explicitly requested family, not an additional arbitrary grid: 15 adjacent differences plus 14 adjacent second differences supported entirely within slots24–39. Every candidate starts at the verified actual FP32 MrPro table and targets the corresponding actual Native clock divided by4:

`nu_c = nu_M + n * (n^T omega_native / 4 - n^T nu_M) / (n^T n)`.

Native tensor SHA `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e`; Mr tensor SHA `33cbe3a40994ac2a79126a14ce30282867bd6d49b74c1ada4d4cce8a7e76016f`. Both were read from actual run contracts. Gain1.138629436111989 is unchanged. Mr sum(m)=29.333333268998935. All29 candidates are finite, positive and strictly decreasing, preserve every outside slot and both endpoints bitwise, and pass FP64 projection identities plus explicit FP32 rounding bounds. Independent standard-library exact-rational arithmetic on the source FP32 numbers agrees with the generator's ideal frequencies within 0. Maximum deployed relative relation-clock error is 1.0494258591124796e-05.

**Important correction to naive clock reasoning:** none of these Mr relations already retimes Native by4. Early middle first differences are faster than Native despite each raw frequency being slowed. Second-difference periods mostly also become shorter. Thus this family changes actual joint clocks and is not a disguised reproduction of the existing Mr ramp. That observation establishes the intervention, not its utility.

| Candidate | Mr/native period ratio | Delta sum(m) | Maximum absolute128K phase change | Within0≤m≤1 |
|---|---:|---:|---:|---|
| JointMode_d1_s24_25 | 0.939106 | -0.009311633 | 58.305359 | no |
| JointMode_d1_s25_26 | 0.924643 | -0.010063031 | 47.945312 | no |
| JointMode_d1_s26_27 | 0.919962 | -0.010831734 | 38.892059 | no |
| JointMode_d1_s27_28 | 0.924824 | -0.011615592 | 31.126892 | yes |
| JointMode_d1_s28_29 | 0.939298 | -0.012411491 | 24.580612 | yes |
| JointMode_d1_s29_30 | 0.963758 | -0.013215000 | 19.151062 | yes |
| JointMode_d1_s30_31 | 0.998901 | -0.014020074 | 14.717461 | yes |
| JointMode_d1_s31_32 | 1.045774 | -0.014818063 | 11.151443 | yes |
| JointMode_d1_s32_33 | 1.105830 | -0.015596872 | 8.325500 | yes |
| JointMode_d1_s33_34 | 1.181003 | -0.016339572 | 6.118835 | yes |
| JointMode_d1_s34_35 | 1.273807 | -0.017021916 | 4.421074 | yes |
| JointMode_d1_s35_36 | 1.387481 | -0.017608583 | 3.134420 | yes |
| JointMode_d1_s36_37 | 1.526170 | -0.018048214 | 2.174410 | yes |
| JointMode_d1_s37_38 | 1.695176 | -0.018264693 | 1.469765 | yes |
| JointMode_d1_s38_39 | 1.901282 | -0.018143720 | 0.961573 | yes |
| JointMode_d2_s24_25_26 | 1.004304 | +0.000366836 | 6.906677 | yes |
| JointMode_d2_s25_26_27 | 0.944590 | +0.000440662 | 6.035522 | yes |
| JointMode_d2_s26_27_28 | 0.900320 | +0.000522615 | 5.176788 | yes |
| JointMode_d2_s27_28_29 | 0.869231 | +0.000613092 | 4.364197 | yes |
| JointMode_d2_s28_29_30 | 0.849782 | +0.000711943 | 3.619705 | yes |
| JointMode_d2_s29_30_31 | 0.840963 | +0.000819363 | 2.955734 | yes |
| JointMode_d2_s30_31_32 | 0.842223 | +0.000934877 | 2.377342 | yes |
| JointMode_d2_s31_32_33 | 0.853410 | +0.001057884 | 1.883957 | yes |
| JointMode_d2_s32_33_34 | 0.874740 | +0.001187531 | 1.471115 | yes |
| JointMode_d2_s33_34_35 | 0.906799 | +0.001322201 | 1.131840 | yes |
| JointMode_d2_s34_35_36 | 0.950576 | +0.001459941 | 0.857769 | yes |
| JointMode_d2_s35_36_37 | 1.007489 | +0.001597458 | 0.640005 | yes |
| JointMode_d2_s36_37_38 | 1.079487 | +0.001730576 | 0.469763 | yes |
| JointMode_d2_s37_38_39 | 1.169148 | +0.001853444 | 0.338795 | yes |

The three early pair candidates accelerate one slot beyond Native:25,26,27 respectively. They remain positive and ordered but fall outside the optional compression-only box. All triples stay inside that box. The zero-sum relation preserves the raw frequency sum, **not** the sum of log-compression exponents. Therefore these candidates are not same-compression-budget controls; every delta is reported rather than silently corrected with an additional slot.

Each JSON entry supplies `delta_log_period = -log(nu_c/nu_M)` for contraction with the parent's `gradient_log_period`. The resulting first-order predicted loss is only a direction filter. Pair changes reach58.305radians and triple changes6.907radians at128K; the gradient cannot forecast their finite loss reliably. Exact full-model finite answer CE and actual generation must adjudicate any selected test. Repeating the intervention against independent orthogonal directions is needed for a later causal-harmonic attribution; this construction alone does not establish one. No GPU was used in this continuation.
