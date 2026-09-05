# RoPE optimality: identifiability limit and conditional equations

- **Date:** 2026-09-03
- **Status:** `COMPLETE DERIVED RESULT / PRIMARY-LITERATURE AUDIT / NO MODEL OR GPU EXECUTION`
- **Evidence labels:** the counterexamples, exact-preservation obstruction, and
  conditional first-order equations below are **Derived results** under their
  stated assumptions. Repository numbers are used only with their current
  owner labels. Literature comparisons and paper-value estimates are
  **Interpretations**, not novelty certificates.
- **Exact questions:** Can RoPE/attention structure alone determine (i) the
  optimal fixed-support interior allocation `z`, (ii) the optimal frequency
  system, or (iii) the optimal attention-aware movement of a mature frozen
  checkpoint? If not, what single mathematical object is missing, and what
  exact equations become available once it is supplied?
- **Inputs:** current routed theory, experiments, failures, and corrections in
  [`INDEX.md`](../../../INDEX.md); the historical
  [true-objective audit](../../../rebuttal/rebuttal_0723/theory_results/EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md);
  and the primary sources listed in Section 9.
- **Execution boundary:** source reading, proof checking, primary-literature
  search, and deterministic CPU algebra only. No training, model inference,
  remote compute, GPU, paid compute, curve search, or new experiment.

> **Scope correction (2026-09-03).** This report answers the distribution-free
> question in its title. It does not answer which form is selected after the
> completed repository experiments are admitted, and it must not be used to
> close the zero-training or LoRA objective. In particular, request-level
> Native/long routing does not answer the single-static-table question. The
> completed-history selection and its same-substrate LoRA boundary are owned by
> [`SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md`](../attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md).

## 1. Verdict

There is no task-independent behavioural optimum for any of the three
questions. This is not merely an absence of a successful heuristic: Section 4
gives strict counterexamples even when the operator, number of rotary pairs,
and physical frequency endpoints are fixed.

The one missing mathematical object is the **training--deployment risk
functional**

\[
\boxed{\mathscr R_\kappa(\Omega)} ,
\]

where `kappa` fixes the model/operator class, data and labels, training
algorithm and budget, deployment workloads, losses, decoder/scorer, randomness,
and any endpoint weights or constraints. For a mature checkpoint, the training
map is replaced by the fixed checkpoint, but deployment data, losses, and
tolerances remain part of `kappa`.

Globally, the whole functional (or a risk vector when no scalar endpoint
weights are justified) is required. Under differentiability and a valid local
quadratic approximation, its first and second variations `(g,H)` are sufficient
for a principled local direction. RoPE algebra gives the Jacobian from frequency
to attention score; it does **not** give the signed loss adjoints that turn this
Jacobian into `g`, nor the co-adaptation response that enters the full
training-time gradient and curvature.

Consequently:

1. **Optimal `z`:** non-identifiable from RoPE structure alone; given
   `R_kappa`, it obeys an ordered KKT system.
2. **Optimal frequency system:** non-identifiable until a target kernel,
   measure, task risk, and model class are specified; classical Fourier/frame
   results then give conditional optima or bounds, not an LM-universal grid.
3. **Mature-checkpoint movement:** checkpoint tensors alone cannot determine a
   behavioural direction. Given a deployment risk, the exact signed spectral
   gradient is available; given positive curvature, the local optimum is an
   `H`-metric projection onto the feasible movement cone.
4. **Universal exact short no-harm:** a nontrivial stationary table is
   impossible under standard RoPE and universal content/position equality.

No new frequency curve or method follows from this result.

## 2. One coordinate system for all three questions

For `K` rotary pairs, write physical log frequencies as

\[
x_k=-\log\omega_k=a+Rz_k,
\qquad
0=z_0\le z_1\le\cdots\le z_{K-1}=1.
\]

Here `(a,R)` fixes sampled support and `z` is the normalized interior
allocation. This removes the base--exponent gauge: `B^{-u}` is unchanged by
`B -> B^c, u -> u/c`, while `x=-log omega` is physical.

A convenient closed feasible set is

\[
\mathcal Z_\delta=
\{z:z_0=0,z_{K-1}=1,
z_{k+1}-z_k\ge\delta\},
\quad 0\le\delta\le\frac1{K-1}.
\]

For a mature checkpoint and requested scale `S`, an often-used movement
coordinate is

\[
\omega'_k(m)=\omega_k S^{-m_k},
\qquad m\in\mathcal C,
\]

where `C` must state its order, bounds, sharing, support, and zero-frequency
rules. The semigroup identity of this parametrization constrains form only; it
does not select `m`.

For an experiment instance

\[
\kappa=(M,O,K,a,R,\mathcal A,T,P_{\rm train},
\{Q_j,\ell_j,D_j,w_j\}_{j=1}^J,\Xi),
\]

let \(W_T(\Omega;\mathcal S,\xi)\) be the weights produced by the declared
finite training algorithm on sample \(\mathcal S\). A scalar
training--deployment risk is

\[
\mathscr R_\kappa(\Omega)=
\mathbb E_{\mathcal S\sim P_{\rm train},\xi\sim\Xi}\sum_{j=1}^J w_j
\mathbb E_{(q,y)\sim Q_j}
\ell_j\!\left(
D_j[f_{W_T(\Omega;\mathcal S,\xi),O_j(\Omega)}(q)],y
\right).
\]

If endpoint weights `w_j` are not scientifically justified, replace the scalar
by the vector `(R_1,...,R_J)` and report its Pareto set. A threshold chosen by
an author is a constraint or utility choice, not a discontinuity supplied by
RoPE mathematics.

For a frozen mature checkpoint `W_0`, set `W_T= W_0`. This special case still
requires `Q_j`, `ell_j`, decoder/scorer, length distribution, and tolerances.

## 3. What repository history identifies -- and what it does not

The intersection of the current valid owners is unusually sharp:

| Historical result | Valid information | What it cannot identify |
| --- | --- | --- |
| [fixed-support 151.9M three-seed result](../evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) | changing only `z` can change trained behaviour | a universal or unique optimum |
| [M4 exact-range factorial](../../../rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md) | non-Cosh matched shapes can remain competitive | Cosh-shape necessity |
| [full sin/cos / weights-by-table crossing](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | static geometry and weight co-adaptation are distinct; unsigned rank can reverse LM quality | a static behavioural ranker |
| [ordered frequency permutation collapse](../attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | the learned object includes slot-to-frequency pairing | an unordered optimal spectrum or a task-free ordering |
| [static-selector failures](../attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md), [direct-`z` negative](../attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md), and [analytic-table negatives](../attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) | those exact proxies/candidates do not prospectively rank the registered behaviour | impossibility of every task-conditioned objective |
| [allocation dose response](../attention-aware-retrofit/results/adaptation-coadaptation/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md) | analytic Cosh direction misses no-harm; a learned local direction need not improve long full NLL | a monotone dose law or global basin |
| [Native-isotonic result](../attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) | one Native-derived table trades better natural likelihood against worse fresh structured 4K/8K capability | a checkpoint-latent movement law |
| [Selective-31 six-arm result](../attention-aware-retrofit/results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) | the exact head/slot selection fails against registered random/reverse controls | the whole class of attention-conditioned risks |
| [Native/s4 request policy](../attention-aware-retrofit/results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | a completed two-mode policy can preserve exact short sessions and support long use in its OLMo protocol | one static universally optimal table |

Thus the experiments establish that allocation is causal and
lifecycle-dependent, while failures repeatedly reject unsigned or
content-blind surrogates as universal selectors. They do not supply the missing
population risk functional by accumulation: different tasks, lengths,
checkpoints, training paths, and metrics are different `kappa` values.

## 4. Non-identifiability theorems

### 4.1 Fixed-support interior allocation

**Theorem 1 (no task-independent `z`).** Assume the feasible set contains at
least two values for one interior coordinate and the corresponding physical
frequency lies in `(0,pi)`. Even with fixed endpoints, standard RoPE, fixed
content, and one active interior pair, no allocation can minimize all valid
teacher-attention cross-entropy tasks.

**Proof.** Let the free coordinate be `u`, with
`omega(u)=exp[-(a+Ru)] in (0,pi)`. Use two keys at relative distances `0,1`
and logits

\[
s_u=(1,\cos\omega(u)),\qquad p_u=\operatorname{softmax}(s_u).
\]

Because

\[
\frac{d}{du}\cos\omega(u)
=R\omega(u)\sin\omega(u)>0,
\]

`u -> p_u` is injective. For any feasible `v`, define the teacher target to be
`p_v`. Its cross entropy is

\[
H(p_v,p_u)=H(p_v)+D_{\rm KL}(p_v\|p_u),
\]

which has the unique minimum `u=v`. Two distinct teachers `v_1` and `v_2`
therefore have mutually incompatible unique optima. All other rotary pairs,
including both fixed endpoint pairs, can be made content-inactive. QED.

This refutes the quantifier

\[
\exists F(K,a,R,L)\ \forall\kappa:\quad
F(K,a,R,L)\in\arg\min\mathscr R_\kappa.
\]

It does not refute an instance-specific optimum after `kappa` is fixed.

### 4.2 The training algorithm is part of the estimand

**Theorem 2 (finite-training rank reversal).** Consider

\[
f_{a,\omega}=a\cos\omega,
\quad L=\tfrac12(a\cos\omega-1)^2,
\quad a_0=0.
\]

After one gradient step of size `eta`,

\[
R_\eta(\omega)=\tfrac12(\eta\cos^2\omega-1)^2.
\]

For any two candidates with distinct nonzero `cos^2(omega)`, choosing
`eta=1/cos^2(omega_A)` makes `A` exact and leaves positive risk for `B`;
choosing `eta=1/cos^2(omega_B)` reverses the result. For example,
`omega_A=0.2` and `omega_B=0.8` satisfy the condition. At convergence both can
reach zero. Hence representation, finite-training optimum, and converged ERM
optimum are different questions.

### 4.3 Mature checkpoint alone cannot rank movements

**Theorem 3 (checkpoint-only decision dichotomy).** Fix a checkpoint and two
frequency movements `A,B`. If their output distributions differ on some input
`q`, then two ordinary hard-label cross-entropy tasks strictly reverse their
ranking. If they never differ, their behaviour cannot identify a unique choice.

**Proof.** If `p_A(.|q) != p_B(.|q)`, normalization implies that there is a
token `y_+` with `p_A(y_+|q)>p_B(y_+|q)` and a token `y_-` with the opposite
inequality. The point-mass task `(q,y_+)` strictly prefers `A` under
`-log p(y|q)`; `(q,y_-)` strictly prefers `B`. If the two output distributions
coincide for every input, every output-based risk ties them. QED.

The theorem does not say natural-language tasks are arbitrary. It says that
checkpoint tensors and RoPE algebra do not contain the missing deployment
semantics needed for a distribution-free ranking.

### 4.4 Exact Native no-harm closes the nontrivial static class

For one rotary plane, universal equality

\[
q^TR(\omega'\Delta)k=q^TR(\omega\Delta)k
\quad\text{for all }q,k
\]

at integer `Delta=1` gives `R(omega')=R(omega)`. Inside `(0,pi)`, this forces
`omega'=omega`. Allowing fixed invertible Q/K compensation still requires the
frequency multiset to match up to sign/permutation by similarity of the
rotation-group generators, as proved by the
[transplant obstruction](../../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md).
Therefore a non-Native stationary table cannot
guarantee the exact Native function for all short contents while also changing
long geometry. Approximate no-harm is a risk constraint and returns to
`R_kappa`. The full scoped statement is owned by the
[static no-harm note](../attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md).

### 4.5 Correction to the historical PI-arc proof

The 2026-09-02 working memo's PI-arc conclusion is repairable, but its written
arc-length proof is incomplete: for per-slot scaling `rho`, the candidate
speed is `||rho elementwise omega||`, not a scalar `rho ||omega||`; equality of
arc lengths alone cannot prove uniform scaling.

Under the memo's finite-arc injectivity assumptions, equality or containment of
the two embedded arcs first forces their tangent rays at the common origin to
coincide. Hence

\[
\rho\odot\omega=c\omega.
\]

All `omega_k>0`, so every `rho_k=c`. Arc length then gives `c=1/S` for
set equality and `c<=1/S` for containment. The conclusion survives; this
tangent-space argument, not the old scalar-speed line, owns it.

## 5. Exact conditional optimality equations

### 5.1 Ordered `z` KKT system

Let \(\mathscr R(z)\) be one declared differentiable scalar risk, with
endpoints fixed, and call the log-frequency span \(R_{\rm span}\) in this
subsection.
For gap constraints

\[
c_k(z)=\delta+z_k-z_{k+1}\le0,
\]

there exist multipliers `lambda_k >= 0` such that every internal coordinate
at a regular local optimum satisfies

\[
-R_{\rm span}\omega_j\frac{\partial\mathscr R}{\partial\omega_j}
+\lambda_j-\lambda_{j-1}=0,
\qquad
\lambda_k c_k(z)=0.
\]

Equivalently, `0` belongs to the risk gradient plus the normal cone of
`Z_delta`. RoPE supplies `d omega_j/d z_j=-R omega_j`; it does not supply the
signed derivative of the risk with respect to that frequency.

If `Z_delta` is nonempty and compact and `R` is lower-semicontinuous and
proper, a global minimizer exists. Neither uniqueness nor a closed form follows.
With a risk vector, positive scalarizations give supported Pareto points; they
do not manufacture scientifically justified endpoint weights.

### 5.2 Exact signed attention-frequency derivative

For one band and relative lag `Delta`, write its pre-softmax contribution as

\[
s_k(\Delta)=c_k\cos(\omega_k\Delta)
+d_k\sin(\omega_k\Delta).
\]

In log-frequency coordinate `x_k=-log omega_k`,

\[
\frac{\partial s_k}{\partial x_k}
=\omega_k\Delta
\left[c_k\sin(\omega_k\Delta)-d_k\cos(\omega_k\Delta)\right].
\]

Holding rotary amplitude and every non-frequency operator field fixed, the
exact direct gradient of a frozen-checkpoint risk is

\[
g_k=
\mathbb E_{Q}
\sum_{\ell,h,i,j}
\underbrace{\frac{\partial\ell}
{\partial s_{\ell hij}}}_{\text{signed task adjoint}}
\omega_k\Delta_{ij}
\left[c_{\ell hij,k}\sin(\omega_k\Delta_{ij})
-d_{\ell hij,k}\cos(\omega_k\Delta_{ij})\right].
\]

End-to-end differentiation includes every layer and lets the adjoints carry all
downstream activation paths. An unsigned Q/K norm, Gram rank, Fisher magnitude,
or attention displacement can measure sensitivity/cost, but cannot determine
the sign of `g`. This agrees with the repository's
[local functional audit](../attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md)
and with LeRoPE's independently published per-frequency gradient.

For `omega'_k=omega_k S^{-m_k}`, `x'_k=x_k+m_k log S`, hence

\[
\frac{\partial\mathscr R}{\partial m_k}
=(\log S)g_k.
\]

This is the closed-form **directional equation**, not a closed-form movement:
its coefficients are precisely the missing risk-dependent object.

### 5.3 Strongest local movement statement

At a baseline `m_0`, let `g=grad R(m_0)` and `H=grad^2 R(m_0)`. If `H` is
positive definite and the feasible set has a closed convex tangent cone
`T_C(m_0)`, the quadratic local optimum is

\[
\delta m^*=
\arg\min_{v\in T_\mathcal C(m_0)}
\left(g^Tv+\tfrac12v^THv\right)
=\Pi^{H}_{T_\mathcal C(m_0)}(-H^{-1}g).
\]

If no constraint is active, this reduces to the Newton direction `-H^{-1}g`.
If `H` is indefinite, there is no convex local optimum of this form and a
trust-region or higher-order specification is part of the objective. This is
the strongest curve-free local principle available from ordinary optimization.

A particularly explicit short-no-harm/long-benefit specialization is possible.
Let the local Native-output divergence and long risk be

\[
C_N(h)=\tfrac12h^TF_Nh+o(\|h\|^2),
\qquad
L_{\rm long}(h)=L_{\rm long}(0)+g_L^Th+o(\|h\|).
\]

If `F_N` is positive definite, `g_L` is nonzero, `epsilon>0`, and there are no
active order/bound constraints, then

\[
h^\star
=\arg\min_{h:\,\frac12h^TF_Nh\le\epsilon} g_L^Th
=-
\sqrt{\frac{2\epsilon}{g_L^TF_N^{-1}g_L}}
F_N^{-1}g_L.
\]

With active constraints this becomes a cone KKT problem. This closed-form
direction is exact for the stated local quadratic program, not for the original
nonlinear endpoint: `F_N` must be a full-model, data-weighted functional
curvature and `g_L` a signed long-risk gradient, and Taylor remainders still
bound its valid radius.

### 5.4 Training-time co-adaptation term

For finite updates `W_{t+1}=Psi_t(W_t,z,xi)`, define
`J_t=dW_t/dz`. Then

\[
J_{t+1}=\partial_W\Psi_t J_t+\partial_z\Psi_t,
\]

and

\[
\nabla_z\mathscr R_{\rm out}
=\partial_z\mathscr R_{\rm out}
+J_T^T\nabla_W\mathscr R_{\rm out}.
\]

At a differentiable isolated inner optimum with invertible training Hessian,

\[
\frac{dW^*}{dz}=-H_{WW}^{-1}H_{Wz},
\qquad
\nabla_z\mathscr R_{\rm out}
=R_z-R_W H_{WW}^{-1}H_{Wz}.
\]

The second term is the weight--frequency co-adaptation response. Any static
frequency objective omitting it answers a different estimand.

## 6. Why established Fourier/frame theories do not close the gap

One exact simplified frequency-system principle makes the missing object
especially visible. Let `H=L2(mu)`, and let a random target phase function `F`
have trace-class second-moment operator (or covariance after explicit
centering)

\[
C=\mathbb E[F\otimes F].
\]

Let

\[
V_\Omega=\operatorname{span}\{\cos(\omega_k\Delta),
\sin(\omega_k\Delta)\}_{k=1}^K.
\]

After optimizing the linear readout in `V_Omega`, projection gives

\[
\mathbb E\inf_{v\in V_\Omega}\|F-v\|_H^2
=\operatorname{Tr}C-\operatorname{Tr}(P_\Omega C).
\]

Thus the exact conditional frequency principle is

\[
\Omega^*\in\arg\max_\Omega\operatorname{Tr}(P_\Omega C).
\]

For arbitrary subspaces of the same dimension, Ky Fan's principle chooses the
top eigenspace of `C`; restricting the subspace to RoPE sinusoidal atoms makes
this a nonlinear best-subspace problem, generally without a closed form. If
two distinct equal-dimensional candidate spans are compared, choosing target
covariance in either span reverses their ordering; if the spans agree, the
frequency descriptions are unidentifiable in this model. Here the missing
objects are `mu` and `C`, which are the linear-projection specialization of the
general `R_kappa`.

These theories map rigorously after their own target object is supplied:

| Theory / declared objective | Conditional result | Missing bridge to LM behaviour |
| --- | --- | --- |
| uniform scalar quantization on log frequency | `D=(1/12) sum gap^3`; equal gaps uniquely minimize it, giving geometric spacing | why uniform log-frequency coverage is the risk |
| weighted high-rate quantization | `rho*(phi) proportional to w(phi)^(1/3)` | a positive task-conditioned distortion weight `w`; it is not supplied by RoPE geometry or interchangeable with the signed risk gradient |
| quadratic potential / integral operator | `(K rho)(phi)+V(phi)=constant` on support, at least that constant off support | why the chosen kernel `K` and field `V` equal the LM objective |
| EVQ Brownian-kernel surrogate | its chosen `alpha delta + beta min(phi,psi)` operator yields the Cosh ODE and unique density | the surrogate-to-trained-risk identification |
| frame potential / tight-frame design | whitened Gram energy is minimized by a tight frame; harmonic lattices can attain equality on a specified window | a tight frame can be exactly periodic/aliased outside that window |
| Slepian--Pollak concentration | prolate eigenfunctions optimally concentrate a bandlimited subspace in a stated time window | the optimum functions are broadband eigenfunctions, not a unique set of RoPE atoms; the task loss is absent |
| Landau density theory | necessary asymptotic sampling/interpolation densities | no finite-node ordering, content weights, or deployment utility |
| Bochner / random Fourier features / Gaussian quadrature | approximate a **given** stationary kernel or spectral measure, sometimes with data-dependent leverage sampling | the target kernel/spectral measure itself |
| D-optimal frequency design | with lag measure `mu`, `G_kl=mu_hat(omega_k-omega_l)` and `d log det G/d omega_k=tr(G^{-1}dG/d omega_k)=0` | changing `mu`, horizon, regularizer, or endpoint loss changes the solution |
| isotonic regression | unique projection onto an ordered cone for a supplied target and norm | neither target nor norm is identified by the checkpoint |

The contradiction among these conditional optima is expected: each theorem is
correct for a different functional. Choosing one functional because it yields
an attractive curve is exactly the surrogate substitution that the repository
failures warn against.

## 7. What is genuinely new here, and what is already known

### Already known or independently published

- Fourier characters, frame potentials, prolate concentration, sampling
  density, optimal design, scalar quantization, isotonic projection, and
  bilevel implicit differentiation are established mathematics.
- LeRoPE learns one shared log-frequency scalar per band and derives an exact
  loss gradient whose coefficients include relative Q/K phase, attention,
  values, and downstream signed gradients. Thus the signed local derivative
  is not a new invention of this report.
- *How Data Shapes RoPE Frequency Usage* explicitly introduces a
  data-dependent positional dependency kernel. Its `theta*=pi/W` theorem is
  conditional on a single width and a field-admissibility constraint; the
  paper explicitly treats natural language as a mixture rather than predicting
  one universal frequency.
- AdaRoPE learns head-specific frequencies and attention scales. It supports
  heterogeneity of the operative object, but changes a broader model class than
  fixed-support shared `z`.
- FourierLearner-Transformers learn a spectral representation of relative
  positional encodings inside a linear-attention model class. This is broader
  than standard fixed-support RoPE, but further rules out any priority claim
  for learning an “optimal” positional spectrum in general.
- Random-Fourier-feature and quadrature results optimize approximation of a
  supplied kernel or risk; data-dependent leverage schemes make the dependence
  on distribution and regularization explicit.

### Narrow contribution not located as one theorem in the searched sources

The defensible RoPE-specific contribution is the unified statement that:

1. fixed-support `z`, the full frequency system, and mature movement are three
   restrictions of the same risk-functional problem;
2. a fixed-support teacher-attention construction and checkpoint-output
   dichotomy formally rule out a distribution-free selector;
3. exact short-function preservation separately closes all nontrivial static
   movements under standard assumptions; and
4. after supplying the missing risk, the ordered KKT, signed spectral
   derivative, training hypergradient, and local `H`-projection state exactly
   what can be solved.

This is a synthesis and boundary theorem, not a claim to have invented
frequency learning, Fourier optimal design, or task-dependent RoPE.

## 8. Success probability and paper value

These are calibrated research judgments, not measured probabilities.

| Target | Assessment | Reason |
| --- | --- | --- |
| derive one universal behavioural optimum from RoPE structure alone | `0` under Theorems 1--3's quantifiers | formally refuted, not merely unlikely |
| produce a rigorous self-contained theory conclusion without new GPU evidence | `85--95%` likely to survive expert checking; the result itself is completed here | counterexamples and conditional equations are analytic; novelty is a separate question |
| derive a guaranteed better practical static table without supplying/estimating `R_kappa` | not identifiable | would reintroduce an unstated surrogate |
| standalone “new optimal RoPE method” headline | `0--10%` | no method follows, and learned/task-dependent frequencies already exist |
| standalone theory novelty at a top venue | `15--30%` | ingredients are classical and recent RoPE papers already expose task-conditioned gradients/dependencies |
| paper boundary theorem / appendix proposition | `65--80%` useful | it protects the causal claim, explains negative selectors, and prevents Cosh from being overclaimed |
| main-paper value if it displaces the fixed-support empirical story | low | the empirical identification is stronger and more distinctive than a broad no-free-lunch claim |

The best paper role is therefore narrow: one proposition stating no
task-independent optimum, one equation showing the signed task-conditioned
gradient, and a proof appendix. It strengthens the paper's claim discipline:
`z` is a real causal design axis; EVQ-Cosh is a closed-form point selected by a
declared surrogate, not the behavioural optimum. It should not be marketed as
a solved frequency-design method.

## 9. Primary-source ledger

- Karypis et al., [LeRoPE: Learnable RoPE Frequencies Improve Language
  Modeling](https://arxiv.org/abs/2607.10134), especially Eq. 6 and its
  task/content-conditioned spectral-gradient derivation.
- Wu et al., [How Data Shapes RoPE Frequency Usage: From Positional Scale
  Matching to Length Generalization](https://arxiv.org/abs/2607.07678),
  especially the dependency-kernel definition, field-constrained theorem, and
  self-similarity condition.
- Wang et al., [AdaRoPE: Not All Attention Heads Should Rotate and Scale
  Equally](https://arxiv.org/abs/2607.19363).
- Choromanski et al., [Learning a Fourier Transform for Linear Relative
  Positional Encodings in
  Transformers](https://proceedings.mlr.press/v238/choromanski24a.html).
- Slepian, [Prolate Spheroidal Wave Functions, Fourier Analysis, and
  Uncertainty--V: The Discrete Case](https://doi.org/10.1002/j.1538-7305.1978.tb02104.x).
- Landau, [Necessary Density Conditions for Sampling and Interpolation of
  Certain Entire Functions](https://doi.org/10.1007/BF02395039).
- Benedetto and Fickus, [Finite Normalized Tight
  Frames](https://doi.org/10.1023/A:1021323312367).
- Dao, De Sa, and Re, [Gaussian Quadrature for Kernel
  Features](https://arxiv.org/abs/1709.02605).
- Avron et al., [Random Fourier Features for Kernel Ridge Regression:
  Approximation Bounds and Statistical
  Guarantees](https://proceedings.mlr.press/v70/avron17a.html).
- Li et al., [Towards a Unified Analysis of Random Fourier
  Features](https://proceedings.mlr.press/v97/li19k.html).

## 10. Supported and unsupported claims

### Supported

- RoPE/attention algebra alone does not identify a universal optimal `z`,
  frequency system, or mature-checkpoint movement.
- The unique global missing object is the declared training--deployment risk
  functional; its signed gradient and curvature are the local missing data.
- Given that object, ordered KKT, exact spectral-gradient, hypergradient, and
  local quadratic-projection equations follow without inventing a curve.
- Classical Fourier/frame/approximation results give conditional solutions and
  useful impossibility/stability boundaries, not a language-model optimum.
- Exact universal Native preservation permits no nontrivial stationary table
  under the stated standard-RoPE assumptions.
- This conclusion is useful as a paper boundary and claim correction, not as a
  new method.

### Unsupported

- any specific new `z`, density, frequency table, head mask, gain, or movement;
- any claim that the repository's current best table is globally or universally
  optimal;
- any claim that an unsigned attention/geometry statistic can substitute for
  the signed task risk;
- any empirical improvement, model-quality prediction, or GPU-run conclusion
  from the analytic equations;
- priority or novelty beyond the bounded primary-source search above.

This owner ends at the theory conclusion. It authorizes no experiment or RoPE
method recommendation.
