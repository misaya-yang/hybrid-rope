# Astra07 — Adaptation is a learning operator, and free coefficients do not identify a density

Status: all 90 assigned files (1,107,652 bytes) were read in full; the coverage receipt records paths, hashes, and complete character ranges. Mathematical derivation and synthetic CPU checks are complete. No GPU, model evaluation, manuscript edit, runtime edit, or extra agent was used. The research theory skill was applied. Archived instructions and reviewer verdicts were treated as historical text.

## 1. Result and the correction that matters most

The common useful object is a signed source-versus-distractor margin with a specified content population. The lifecycle enters through the actual map that learns its coefficients. At zero updates, inherited labeled coefficients matter. After adaptation, coefficient means and covariances must be recalculated from that learning map. They cannot be treated as a scalar penalty attached to an otherwise unchanged geometric optimum.

Two rigorous results make this operational:

1. In an affine-logit softmax family, information projection gives an exact approximation-plus-compatibility decomposition, provided the optimum moment-matches the same target and context measure. Convex adaptation constraints add an explicit nonnegative KKT slack. General nonlinear Transformer or LoRA parameterizations do not inherit this equality.
2. In a Gaussian signed-margin model, the logarithm of an exponential selection-error bound is a quadratic **derived from the margin distribution**. Its finite-step adaptation has a closed form with no regularizer. This produces explicit lifecycle crossovers and shows why more permitted adaptation can hurt a different long deployment population.

There is also a substantive obstruction to overstating the proposed EVQ connection: if both a continuum frequency density and its coefficient function are freely adjustable, their product is the observable object. An arbitrary positive density can be compensated by its coefficients. Therefore unrestricted scratch adaptation does not select a unique Cosh density. A Cosh count-allocation derivation needs the equal-loading/shared-noise assumptions stated by Astra01, or another concrete finite-channel/optimization constraint. Initialization exchangeability by itself is insufficient.

The constructive next rule is to compute signed margin means/covariances after a declared finite update operator, and then use a positive-margin allocation under hard native constraints. In the special equal-loading population model this reduces to inverse-covariance allocation and, conditionally, EVQ. In frozen deployment it reduces to the labeled finite-phase solver in Astra03. The exact published MrPro polynomial remains an empirical construction, not a generally derived optimizer.

## 2. Evidence from the fully read assignment

The following are current-file observations, not endorsements of old prose:

- `paper-2027/appendix/a5_identification.tex:8–57` specifies 30 changed interiors, matched support, and matched training. Lines 97–126 report all fixed-support OOD signs favoring Cosh but all retargeted OOD signs favoring FMRoPE. Lines 168–184 give the two-seed trained-weights/runtime-table interaction, 3.400/3.251 NLL at 1024. These directly require distinct training and deployment objects.
- `paper-2027/sections/03_theory.tex:73–110` explicitly **chooses** the convex Cosh criterion. It does not claim a generic end-to-end task-loss equivalence. The alpha/beta ratio is not identified by those equations.
- `paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md:12–44` records anchored Cosh and protected-band Cosh improving 2x NLL while costing +3.9780/+0.6922 at 1x. Protecting one wavelength band reduces one shock but does not prove functional preservation.
- `paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md:12–41` records a small Cosh displacement improving the 16K tail while missing its native guard, and a learned direction with a better local trade-off. Long **full** NLL worsens at every nonzero point; the static-rank location prediction fails. Thus replacing task endpoints by tail loss or rank would repeat an existing error.
- `paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md:12–57` records a 62-effective-degree table fitted on two documents, with a held-out mean improvement but a single held-out regression violating the declared robustness requirement. This is an identifiability failure of that calibration protocol, not a theorem against every learned allocation.
- The full `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json` confirms 192 completed runs, 180 main plus 12 extreme, and the rule-minus-exponential mean +0.0007397504 with p=0.83642578125. It does not distinguish Cosh's exact shape from the matched alternative. Its old `best_cosh_multiplier_counts` field totals only three, while the explicit structural rows contain twelve configurations; I rely on the actual contrast rows and manuscript table, not that summary field.
- The full official/derived-YaRN pilot JSONs under `rebuttal/pre_rebuttal/seed42_lora_eval_20260713/raw/` show repeated malformed or repetitive generated outputs despite finite likelihoods. The shared training metadata fixes rank 64, alpha 128, Q/K/V/O targets and 300 steps. The old `phase_transition_safe` metadata flag is not a proof that this rank can repair a frequency transplant.
- `rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json` explicitly separates numeric NIAH full-string-plus-EOS success (100/98/60 out of 100 for EVQ at 4K/8K/16K), native RULER regressions, physical-token cap, and long position exposure. Neither the positive nor negative part is transferable to arbitrary adaptation recipes.

The full external cross-audit (`paper-2027/research/external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md:191–377`) already gives finite-step least-squares coefficient dynamics and distinguishes training from deployment. The contribution below is not rediscovery of that matrix sum: it supplies an exact softmax decomposition with correct constraint/distribution conditions, a signed-margin generative objective replacing arbitrary ridge fitting, and the density/loading non-identifiability correction.

## 3. Exact softmax decomposition, including the missing terms

Let c denote a fixed evaluation context with a finite candidate set. Fix feature vectors f_Omega(c,k), offsets b_Omega(c,k), and a context measure P. Define

    q_Omega,u(k|c) = exp[b_Omega(c,k) + u^T f_Omega(c,k) - A_Omega,c(u)].
    A_Omega(u) = E_P A_Omega,c(u).
    m_Omega = E_P sum_k p*(k|c) f_Omega(c,k).

The target p* is fixed across candidate tables in its semantic meaning. Its sufficient-statistic moment changes with the features. p* may be a lawful source distribution for pointer attention, or a teacher/token target. These are different targets and must not be interchanged.

Apart from target entropy and the offset expectation, the expected forward KL is A_Omega(u)-m_Omega^T u. If a **finite unconstrained minimizer** u* exists, then

    grad A_Omega(u*) = m_Omega.

For any inherited or partially adapted coefficient u0,

    E_P KL(p* || q_u0)
      = E_P KL(p* || q_u*) + E_P KL(q_u* || q_u0).                 (S1)

Proof: subtract the right side from the left. The difference is

    E_P sum_k (p* - q_u*) log(q_u*/q_u0)
      = (u*-u0)^T [m_Omega-grad A_Omega(u*)] = 0.

Offsets cancel because both distributions belong to the **same candidate family**. No feature linear independence is needed. Softmax additive-constant directions are harmless gauge directions, although the coefficient optimum need not be unique.

For a shared coefficient across many contexts, moment matching is aggregate. The per-context identity generally fails. The CPU check has aggregate residual 9.88e-15 but maximum per-context residual 0.38397. A paper must not turn the aggregate equation into a rowwise certificate.

### Restricted affine adaptation

Write u=u0+Uv with fixed U. Replace f by U^T f and b by b+u0^T f. Then (S1) holds for the optimum over the unrestricted v-space and every v in that same affine space. U describes actual permitted functional directions. Its value is not specified solely by the nominal LoRA rank.

For a closed convex feasible coefficient set C and u0 in C, if u* minimizes the KL over C, the exact identity is

    KL_P(p*||q_u0)
      = KL_P(p*||q_u*) + KL_P(q_u*||q_u0)
        + (u0-u*)^T [grad A(u*)-m].                             (S2)

The last term is nonnegative by the convex first-order condition. It vanishes only when the feasible chord u0-u* is orthogonal to the residual moment, as at an interior affine optimum. This is the precise Pythagorean inequality, with its slack retained.

If optimization is stopped at u_t, one may use u_t as the comparator u0 in (S1), but cannot replace u* by u_t and retain equality: the remaining moment residual contributes a signed term. Empirical moment matching also gives an identity only on its empirical measure, not on new long contexts.

### Nonlinear adaptation does not inherit the identity

For general logits eta(theta), stationarity gives J_eta(theta*)^T(q*-p*)=0. The KL residual is instead

    E_P (q*-p*)^T [eta(theta0)-eta(theta*)].                     (S3)

A tangent is not a finite chord. This term need not vanish or be nonnegative on a curved model manifold. The CPU example eta(u)=(u,u^2,0), p*=(1/3,.38,.2866667), has a strict stationary local optimum u*=0 and a nonzero residual -0.0116667 at u=.5. Real Q/K products, changing hidden states, and low-rank factorized updates have precisely the kind of nonlinear geometry for which the affine equality is unavailable.

If the infimum is attained only at diverging logits (e.g. separable one-hot targets), a finite u* does not exist. One must either work with a bounded reachable set, prove an appropriate limiting result, or report the absent finite projection. Arbitrarily adding a ridge to create an optimum changes the problem and is not a consequence of softmax.

### What S1 can and cannot establish

It gives a clean candidate-family approximation term plus an inherited/finite-adaptation compatibility term. The former may be smaller for a richer allocation while the latter is larger, allowing an ordering reversal. It does not imply that either term is known from frequencies alone. It does not justify the same decomposition under a different long-context target or context measure. It cannot prove that a source-attention fit is sufficient for generated answers.

## 4. A non-arbitrary finite-adaptation model of the signed margin

For a specified operation/lag population, let X_Omega in R^K collect the signed per-channel useful-key-minus-distractor contributions under a finite table. Its entries retain actual cosine/sine phases. A real loading a produces the margin

    M = a^T X_Omega.

Assume X_Omega is Gaussian with mean mu_Omega and covariance C_Omega. One exact realization is a fixed query and Gaussian key/coefficient uncertainty; products of arbitrary Gaussian Q and K are not themselves Gaussian. Under this model

    K_Omega(a) := log E exp(-M)
                 = -mu_Omega^T a + (1/2) a^T C_Omega a.        (G1)

Markov gives P(M<=0)<=min(1,exp K). The bound is not exact error, but its origin is explicit signed selection error. It has no geometric bonus or regularizer. Under a valid sub-Gaussian envelope, G1 is an upper bound on log MGF instead of equality. Conditional Gaussianity by lag does not imply an unconditional Gaussian mixture; use a separate log-sum-exp of class MGFs when needed.

For a desired margin eta, use eta+K in the exponent. Against N distractors, sum their bounds or use their class-specific log-sum-exp; a single random-key mean is insufficient at 128K.

Training on population A with a=a0+Uv and v0=0 yields

    H = U^T C_A U,
    d = U^T(mu_A-C_A a0),
    v_(t+1) = (I-eta_step H) v_t + eta_step d.

For n actual gradient steps,

    v_n = f_n(H)d,
    f_n(lambda) = [1-(1-eta_step lambda)^n]/lambda,
    f_n(0) = n eta_step,
    a_n = a0+U v_n.                                           (G2)

This is an explicit learning operator. It is not a norm ball standing in for unknown adaptation. The safe fixed-step stability condition on positive curvature is 0<eta_step lambda_max(H)<2. Finite n is still algebraically defined outside that region, but stability must not be claimed.

Deploy with potentially different features/table/population D:

    K_D(n,Omega_A,Omega_D) = -mu_D^T a_n + (1/2)a_n^T C_D a_n. (G3)

This is computed without re-fitting the long target. The lifecycle cases are n=0 (frozen), fixed U (restricted coefficient adaptation), U=I (all coefficient directions), and common initial a_init with declared A/Omega_A (scratch in this simplified model). These are coefficient-model cases; none is a theorem that a deep model trains like fixed-feature GD or that free gain adaptation equals full Q/K/V/O adaptation.

At the unconstrained optimum for the **same** population and positive definite C,

    a*=C^-1 mu,
    K(a)=-(1/2)mu^T C^-1 mu + (1/2)(a-a*)^T C(a-a*).           (G4)

Thus maximal fitted standardized margin and minimal Gaussian exponential bound have the same direction C^-1 mu. G4 is the Gaussian quadratic counterpart of S1. Nonnegative loadings require an active-set solution; clipping C^-1 mu generally does not solve the constrained problem.

### Finite-step allocation derivative

A concrete outer optimization must differentiate the learning map, not just its terminal geometry. For a table parameter x_j, with fixed a0,U, set V_tj=partial v_t/partial x_j. Then

    V_(t+1),j = (I-eta_step H)V_tj - eta_step(partial_j H)v_t
                 + eta_step partial_j d,
    V_0j=0.

Consequently

    partial_j K_D = -(partial_j mu_D)^T a_n
                    +(1/2)a_n^T(partial_j C_D)a_n
                    +(C_D a_n-mu_D)^T U V_nj.                (G5)

For RoPE x_j=-log omega_j, partial_x cos(omega d)=omega d sin(omega d), and partial_x sin(omega d)=-omega d cos(omega d). These supply finite-feature moment derivatives. If a0 or U themselves depend on x, their derivatives also belong in G5. Native retention is a hard constraint evaluated with its own mu_N,C_N or exact native task margins; it is not automatically preserved because the training objective decreases.

G2–G5 make the candidate-selection model executable: supplied common source operations determine mu_A,C_A; supplied target transport determines mu_D,C_D; declared updates give a_n; finite frequency variables are then selected using G3 and actual native constraints. Continuous frequency optimization remains nonconvex. Astra03's certified finite-phase inner solver is appropriate for the n=0 case. Do not call a local stationary table globally optimal or claim that unknown target transport has been inferred by this algebra.

## 5. Explicit lifecycle predictions and counterexamples

### A better fitted representation can be worse when frozen

Take inherited a0=(1,0). Representation A has mu_A=(1,0), C_A=I. Representation B has mu_B=(0,1), C_B=I/4. These are two synthetic signed-feature populations, not measured RoPE tables. With U=I and step size 1/2,

    K_A(n)=-1/2,
    K_B(n)=-2+(17/8)(7/8)^(2n).

At n=0, A=-.5 while B=.125. At n=1, B=-.373047 remains worse; at n=2, B=-.754364 becomes better. Its fully fitted value is -2. The crossover occurs when (7/8)^(2n)<12/17. The example is not calibrated to the project; it proves that compatibility and approximation can give an explicit finite-budget ordering reversal without a made-up regularizer.

### More adaptation need not improve long deployment

Take C_A=C_D=I, a0=(1,0), mu_A=(1,1), mu_D=(1,-1). Updating both directions learns a_n=(1,1-2^-n); updating only the first direction leaves a0 unchanged. Training K decreases under the larger update set, but long K becomes

    -1/2 + b_n + b_n^2/2,

which increases for b_n>0. At four steps it is .876953 versus -.5 for the restricted update. Thus a larger accessible space improves the **best achievable same-objective optimum** but a prescribed finite learning trajectory may worsen a different deployment objective. This distinction directly limits claims that all-linear LoRA or full parameter training must repair a particular allocation.

### Training length and physical length are separate variables

Sparse position IDs can change mu_A,C_A by changing phase exposure while leaving the number of competing keys small. A real long sequence changes the competition population as well. Equal maximum position does not equate their learning operators. The assignment's query-gap lineage and cross-audit explicitly preserve this distinction; the new theory must too.

## 6. Density, channel count, and learned amplitude cannot be collapsed into one variable

Suppose the continuum response model is

    M = integral rho(x)a(x)Z(x) dx,
    rho>=0, integral rho=1.

If a is unrestricted and rho1,rho2 are positive on the same support, choose

    a2(x)=rho1(x)a1(x)/rho2(x).

Then every realized margin is identical, not just its mean or variance. Thus arbitrary positive density changes are a gauge of this unrestricted continuum model. There is no unique count-allocation optimum. Bounds or regularity on a can break the invariance, but must be physically justified; finite K also breaks it because the sampled feature dictionaries differ. The implication is not that real allocation is inert—the matched experiments prove otherwise—but that the overly free continuum model erased the very resource it hoped to explain.

### Independent channel noise gives a different count law

Divide frequency space into narrow bins of width dx. Let K rho(x)dx independent channels occupy the bin. Suppose their aggregate useful loading must be w(x)dx and each channel's independent noise variance is sigma(x)^2. Equal sharing within the bin minimizes its noise and gives total variance

    V_ind = (1/K) integral sigma(x)^2 w(x)^2/rho(x) dx.        (C1)

Cauchy–Schwarz proves

    V_ind >= (1/K)[integral sigma(x)|w(x)| dx]^2,
    rho*(x)=sigma(x)|w(x)| / integral sigma|w|.                (C2)

This is a concrete Neyman-type allocation of channel count. It is not EVQ. It includes no alpha integral rho^2. If sigma or w vanishes, zero occupancy is possible; strict positive densities then achieve only an infimum unless lower occupancy is imposed. Finite K requires integer counts and exact finite variance after rounding.

The CPU four-bin example has count fractions [.102564,.410256,.025641,.461538] and variance 1.48535 versus 2.33203 for uniform, with K=64. It is a continuous count relaxation, not an integer K-channel optimum.

A shared frequency-local noise field instead contributes an expression quadratic in aggregate loading, w^T C_shared w. If w itself is fixed, reallocating count cannot reduce that shared component. If one assumes equal per-channel loading, then w is proportional to rho, and a shared-bin covariance can indeed produce alpha integral rho^2 plus beta integral S_rho^2. **Equal loading is the critical restriction** that links the covariance objective to a frequency-count density.

## 7. The conditional EVQ and MrRoPE special cases

### EVQ case: constant protected signal and specified shared noise

Adopt Astra01's explicitly stated equal-loading population model: learned useful normalized signal is h0>0 independent of the allocation; nuisance is a frequency-local shared field plus nested slow-tail noise; and the population model remains valid under the permitted allocation. Let

    C=alpha I + beta G, G(x,y)=min(x,y).

Then maximizing positive margin reliability minimizes

    alpha integral rho^2 + beta integral S_rho^2,

and the normalized inverse-covariance response C^-1 1 gives

    rho_tau(x)=tau cosh[tau(1-x)]/sinh tau,
    tau=sqrt(beta/alpha).

This is mathematically valid under those hypotheses. It does not follow from generic iid channel noise, free coefficient adaptation, or initialization symmetry. Ideal white-noise fields cannot be evaluated at atoms; a declared finite bin or finite correlation-width model is needed before K-channel realization. Astra02 further shows why an exact smooth finite-window collision kernel can have a finite atomic equilibrium, so its delta approximation is not innocuous.

Under nonconstant signal h, the active-interval equation is alpha rho''-beta rho=lambda h'', together with active-set boundaries. The forcing is actual signed useful signal; inventing h=C rho_Mr to reproduce a desired curve would be circular.

### Frozen/Mr case: retain the learned long match

At n=0, every source/distractor coefficient remains attached to its slot. Under the special aligned positive-match model, the source mean contains sum_j cos(nu_j D). Keeping it positive and sufficiently large is then beneficial. A different population of coherent wrong-key contributions produces a covariance cost involving a squared aggregate response. Thus the Mr-like signal and EVQ-like interference terms are different parts of one discriminative problem.

The exact progressive arithmetic radix increment in MrPro does not follow from positivity or a first zero alone. It supplies a useful feasible table and baseline. A mixed local/long task determines where frequency compression sacrifices desired native margins; those margins, rather than an imposed smoothing preference, should decide the middle.

### Intermediate adaptation

Use G2 with U and n fixed by the actual update rule. Native and long moment constraints are evaluated after that update. This gives an honest interpolation among lifecycle regimes; it need not interpolate linearly between Geo and Cosh curves. Since learned coefficients can change their roles, a frozen calibration law cannot simply be carried unchanged through large adaptation.

## 8. What the combined executable rule should require for Qwen32K→128K

The full Qwen output is not available from this subtask. The mathematical rule can be implemented once the following inputs are supplied, without inventing a curve or hidden regularizer:

1. Known lawful source operations and hard distractor identities, preserving paired answer worlds and EOS semantics. Top attention alone is not a source label.
2. Actual post-normalization, pre-RoPE Q/K coefficients, plus their slot/layer/head and lag labels; a declared bound or measurement for long-state transport. Projection-weight Frobenius norms are not these statistics.
3. A declared lifecycle: frozen n=0, or a fixed coefficient approximation U, optimizer and number of steps. For a real LoRA/full run, record the actual learning map and treat G2 as a diagnostic approximation until checked.
4. Native requirements stated in actual decision or source-margin units. If using an exponential-moment or Cantelli bound, keep its assumptions and numerical looseness visible.
5. One resulting finite labeled frequency table, followed by exact finite trigonometric evaluation and the requested generated task endpoint. A continuum density with tunable amplitudes is not a substitute for that table.

Use the already completed MrPro/P2/Smooth/E1 contrasts to discriminate whether signed moments explain the behavioral ordering. If the signed conditional source metric still favors the failing Smooth rows, the missing component is likely upstream transport, the chosen operation/role, or value/readout behavior. Do not append another unsigned geometry cost to force agreement. If actual long-state sampling is necessary, that is an empirical input gap, not something resolved by information projection.

The smallest useful new lifecycle test, if the parent decides a model test is warranted, is the same target table with frozen versus a single specified adaptation budget, retaining the corresponding native table pair and measuring both source margins and generated answers. A claim about adaptation requires an actual change in those quantities, not merely convergence of a fitted coefficient model.

## 9. CPU evidence and claim boundary

Reproducer: `.agents/rope_unification_20260910/code/astra07_lifecycle.py` (NumPy only, run successfully).

- Exact aggregate softmax Pythagorean residual: 9.88e-15.
- Convex-constraint slack: .259319883, matching the KKT expression.
- Curved-family residual: -.011666667 despite stationary positive local curvature.
- Finite-step closed form versus direct updates: max error 2.22e-15.
- Explicit compatibility crossover: representation B overtakes A after two declared updates.
- Larger adaptation space under shifted deployment: worse log exponential bound .876953 versus -.5.
- Density/loading gauge equality: max realized-margin difference 0.
- Independent-noise count allocation: C2 equality and strict gain over uniform checked.

These verify mathematics in constructed finite models. They establish no new Qwen table, no causal explanation of the existing Qwen outputs, no task-success theorem for a general LLM, and no universal Cosh/Mr optimum. The useful completed result is an exact lifecycle decomposition, a specified learning operator, and an explicit warning about which coefficient/count assumptions are needed for the EVQ special case to retain predictive content.
