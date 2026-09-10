# Astra06 — adversarial audit and an exact finite-channel signal/interference allocation

Status: all 18 assigned full texts were ingested in bounded contiguous pages, including original reports and their explicit retraction. No GPU, model execution, new agent, runtime edit, or manuscript edit. The coverage receipt is `astra06_coverage.json`. The reason-research-theory skill was applied. The following results are conditional mathematical constructions, not a validated Qwen table.

## 1. Concrete result

A stronger shared criterion than pairwise SNR is available when the requested computation is **softmax source mass**. With a deterministic useful source score μ and Gaussian coherent distractor score variance v, the actual allocation should minimize

    J = v/2 − μ.

This supplies a finite-context probability bound without assuming independent distractor keys. Under the declared shared-bin plus nested covariance model, the criterion has an **exact integer dynamic program that allocates K equal channels**, rather than reweighting fixed channels. Constant protected μ recovers a discrete Cosh law; a coherent remote source contributes the signed cosine μ that motivates MrRoPE. The same dynamic program handles the signed source term, so this is an actual allocation rule rather than a formal objective with an unspecified optimizer.

This connects the objectives in a restricted but nonvacuous computation. It does not derive MrPro's arithmetic radix progression, identify the real Qwen signal/covariance, or establish that scratch-trained response statistics are invariant to allocation. Those are separate missing premises. Astra01/03 correctly retain these distinctions; Astra02's continuous-kernel equilibrium is not Cosh.

## 2. Exact softmax probability statement

Consider one attention row with one useful source at score μ(ν), N competing keys with scores D_t(ν), and all relevant keys included. For a desired mass 0<ρ<1,

    p_* = 1 / [1 + Σ_t exp(D_t−μ)].

Assume μ is deterministic under the stipulated population and each distractor has a valid exponential-moment envelope

    E exp(D_t) ≤ exp(b_t + v_t/2).

A Gaussian D_t of mean b_t and variance v_t is one exact realization. More generally this is a unit-argument MGF bound; no Gaussian approximation is required if the envelope is independently established. Markov's inequality gives

    P[p_* < ρ] ≤ min{1, [ρ/(1−ρ)] Σ_t exp(b_t+v_t/2−μ)}.       (1)

There is **no independence assumption across keys** in (1). Shared query randomness and arbitrary key dependence are permitted provided the stated source-score and MGF assumptions hold. If the source score is random, it is the joint margin MGF E exp(D_t−S), not E exp(D_t)−E S, that must be bounded. Replacing a random source by its mean would invalidate the proof.

If all distractors have b_t=0 and common variance v, minimizing v/2−μ exactly minimizes the right side before clipping at one. A sufficient condition for mass failure ≤δ is

    μ − v/2 ≥ log N + log[ρ/(1−ρ)] + log(1/δ).                (2)

Equation (2) contains distractor multiplicity and absolute logit scale. Squared-SNR does not. In particular, increasing μ/√v by shrinking both μ and v can reduce useful softmax mass: the threshold in a mass theorem must shift the margin by the required log N term.

An explicit output consequence is possible for a **declared linear readout**. Suppose each useful value has correct-versus-competitor readout contrast at least a>0, every distractor value has contrast at least −b, b≥0, and all remaining residual contributions have contrast at least r. Then the output contrast is at least r+a p_*−b(1−p_*). Thus (1) certifies that output when r+aρ−b(1−ρ)>0. This is a precise sufficient condition; actual Transformer residuals, nonlinear MLPs, other heads, and EOS must not be assumed to satisfy it without evidence. A copying construction with one-hot useful values is a nonvacuous instance.

## 3. The actual channel-count rule

Declare B resolved frequency bins with spacing Δ=1/B, representatives x_i=iΔ for i=1,…,B, and ω_i=exp(−x_i) (or b^(−x_i) for normalized exponents). A slot placed in bin i contributes the same amplitude as every other slot. Let n_i be the nonnegative integer count of slots in the bin, with Σ_i n_i=K. The corresponding empirical count mass is p_i=n_i/K.

The nuisance field has covariance

    C_ij = (α/Δ) 1{i=j} + β min(x_i,x_j),   α>0, β≥0.        (3)

A concrete realization is independent Gaussian bin noises of variance α/Δ plus Brownian increments of variance βΔ shared cumulatively between bins. This is **shared frequency-local noise**, not independent noise for individual channels. All channels in the same bin observe the same bin nuisance. Brownian covariance is likewise shared across occupied frequencies. These are testable modeling assumptions, not consequences of RoPE.

Let h_i be the deterministic useful-source score contribution of a unit channel in bin i. For a protected source h_i=A is constant. For an aligned remote source at lag D, h_i=A cos(ω_i D−θ_i), retaining the actual signed phase convention. General deterministic nonrotary or sine/cosine source contributions can be included in h_i. Define tail counts T_i=Σ_{j≥i}n_j. Then exactly

    μ(n)=K^−1 Σ_i h_i n_i,
    v(n)=K^−2 [(α/Δ) Σ_i n_i² + βΔ Σ_i T_i²].             (4)

The min-kernel identity in (4) follows by expanding min(i,j)=Σ_r 1{r≤i}1{r≤j}. Consequently the allocation minimizing (1) solves

    min_{n_i∈Z_+, Σn_i=K} Σ_i [(α/(2Δ))n_i² + (βΔ/2)T_i² − K h_i n_i]. (5)

This integer optimum is constructive. Set F_{B+1}(0)=0 and F_{B+1}(t>0)=∞. Working backward,

    F_i(t) = (βΔ/2)t² + min_{n=0,…,t}
             { (α/(2Δ))n² − K h_i n + F_{i+1}(t−n) }.     (6)

Store the minimizing n and backtrack from F_1(K). Complexity is O(BK²), memory O(BK), and there are no local minima or unknown continuous solver conditions. Equal-bin ties may be broken lexicographically. Required occupied endpoints, occupancy caps, and per-bin minimum channel counts are imposed by restricting the n range, without changing the proof. The representative support in this example is [Δ,1]; if exact support [0,1] is required use a node at zero and its correct zero Brownian-increment weight. Do not silently call the right-endpoint bin rule endpoint-pinned RoPE.

All K frequency locations are emitted by repeating ω_i exactly n_i times. Repeated frequencies retain their K separate content coordinate pairs; they do not delete channels, although they reduce the number of distinct rotations. If distinct frequencies or minimum spacing are required, (5)'s feasible set changes. Spreading duplicates afterward is not an exact solution of (5). B is a stated physical covariance resolution, not an arbitrary curve-candidate sweep. Refining B at fixed α changes the local-noise model; it is not automatically an innocuous numerical refinement.

### Frozen labeled channels

Counts alone are admissible only for exchangeable channel-response models or scratch design. For frozen slots j, replace h_i by h_{j,i}; keep each original slot label. Under a stipulated monotone assignment to the same bins, a DP state (i,t) assigns the next n consecutive labels K−t,…,K−t+n−1 to bin i, and replaces −K h_i n by

    −K Σ_{j=K−t}^{K−t+n−1} h_{j,i}.

Forbid n when any assigned slot's frequency constraints are violated. Under covariance (3), which is bin-dependent and label-independent, the covariance cost still depends only on n and t, so this is again exact O(BK²) with precomputed source-cost prefixes. Actual heterogeneous slot covariance generally destroys this special decomposition. The frozen DP is therefore a concrete special-case algorithm, not a claim that arbitrary Qwen statistics enjoy its structure. It preserves labels explicitly rather than transplanting the scratch count density.

## 4. Discrete Cosh and continuum Cosh are now distinguishable

For constant h, (5)'s linear term is constant. In the continuous count-mass relaxation p_i≥0, Σp_i=1, the optimum solves Cp=λ1. Subtracting neighboring equations twice gives

    p_{i+1}−(2+βΔ²/α)p_i+p_{i−1}=0.

The outer condition is p_{B+1}=p_B. Hence

    p_i ∝ cosh[κ(B+1/2−i)],
    κ=arcosh(1+βΔ²/(2α)).                                (7)

This relaxed solution is positive. As Δ→0 with a resolved density regime, κ/Δ→sqrt(β/α), and (7)/Δ tends to the Cosh density. The exact finite K result remains the integer DP, not (7). This distinction prevents amplitude weights from masquerading as physical counts.

With a nonconstant h, the relaxed KKT equation becomes Cp−h=λ1 on occupied bins, with Cp−h≥λ on empty bins. In the continuous formal limit,

    α ρ''−βρ=h'',     α ρ'(1)=h'(1),

together with normalization and free-boundary conditions if positivity binds. There is no universal Cosh profile when the useful signal changes across frequencies. The coefficient of h differs from the standardized-margin problem because here the optimized quantity is the exponential-moment mass bound, not a homogeneous SNR ratio.

This yields a principled distinction between the two named intuitions: EVQ's constant useful signal minimizes nuisance variance; Mr-like remote coherent content makes h oscillatory and sometimes favors preserving positive low-frequency signal. Neither identity forces a three-band middle transition or the exact t(t+1)/[n(n+1)] exponent.

## 5. CPU arithmetic and a genuinely nonvacuous case

One declared math check used B=16,K=64,α=1,β=4, so τ=2. It is not a Qwen candidate search.

For constant h_i=20, DP returns

    n=[8,7,6,6,5,4,4,4,3,3,3,3,2,2,2,2].

Variance is 2.20953369140625; direct quadratic evaluation matches the DP exactly. Uniform counts give 2.4609375. The real-valued optimum is 2.2038757819141113; (7) agrees with a direct C inverse solve to 2.78e−17. An independent exhaustive B=3,K=4 enumeration gives counts (2,1,1), variance 2.875, as expected from the same recurrence.

At N=131071,ρ=.9,μ=20, equation (1) gives failure bound **0.007339282913290652**, so useful mass is at least .9 with probability at least .9926607 in this stated model. The theorem is not inherently vacuous at 128K; its empirical usefulness depends on real logit margins and a valid tail envelope.

Changing only the declared source to h_i=20 cos(2exp(−x_i)) gives

    n=[0,0,0,0,0,0,0,0,0,0,4,7,9,12,15,17],
    μ=13.507423599659706, v=6.4801025390625.

The bound is now >1 before clipping, hence uninformative at N=131071. This output is honest: optimization has not conjured sufficient capacity. It also shows why constant-signal Cosh and remote signal preservation demand different allocations under the same correlated nuisance model.

Reproducer:

```python
import numpy as np
B,K,alpha,beta=16,64,1.,4.
d=1/B; x=(np.arange(B)+1)*d
C=alpha/d*np.eye(B)+beta*np.minimum.outer(x,x)
h=np.full(B,20.)  # or 20*np.cos(2*np.exp(-x))
F=np.full((B+1,K+1),np.inf);F[B,0]=0
choice=np.zeros((B,K+1),int)
for i in range(B-1,-1,-1):
    for t in range(K+1):
        vals=np.array([alpha*n*n/(2*d)-K*h[i]*n+F[i+1,t-n]
                       for n in range(t+1)])
        choice[i,t]=vals.argmin(); F[i,t]=beta*d*t*t/2+vals.min()
t=K; counts=[]
for i in range(B):
    n=choice[i,t];counts.append(int(n));t-=n
p=np.array(counts)/K
print(counts,p@C@p,p@h,F[0,K]/K**2)
```

## 6. Strong counterexamples that the integration must survive

**Pairwise SNR improvement is not multikey dominance.** Two Gaussian margins with mean .8, variance one, and independent noise give all-positive probability Φ(.8)²=.6211719. Two perfectly correlated Gaussian margins with mean .5 and variance one give Φ(.5)=.6914625. Thus every pairwise standardized margin is better in the first model, but joint retrieval is worse. Astra01/03 correctly phrase union bounds as guarantees, not exact task ranking. A stronger sufficient condition for monotone actual ranking is a common coupling M_A=m_A+D Z, M_B=m_B+D Z with the same diagonal scale D and same joint noise Z and m_A≥m_B componentwise. Then the success event is nested samplewise. This is substantial structure, not implied by covariance summaries alone.

**Independent isotropic channel noise supplies no EVQ collision objective.** If A_j,B_j are iid zero-mean Gaussian quadratures, Var Σ_j[A_j cos(ν_jd)+B_j sin(ν_jd)] is independent of ν,d. Reducing a squared aggregate cosine then cannot be justified as reducing that noise. For K independent per-channel errors, variance of the channel average is K^−1∫v(x)ρ(x)dx, linear in density. It is not α∫ρ². Shared finite-resolution noise or another explicit correlation model is necessary.

**Identical mean/covariance does not determine nonlinear task failure.** Cantelli is an upper bound, not an exact ordering, unless additional distributional structure is imposed. A smaller failure upper bound does not prove the true error of one candidate is lower than another's. Claim improved certification, or establish stochastic dominance/exact Gaussian single-margin conditions.

**Joint torus containment is stronger than marginal phase coverage.** Keeping each slow phase within its native marginal interval need not keep the joint vector (ω_1d,…,ω_Kd) on the original one-parameter orbit. Common PI exactly preserves that orbit on retimed lags; independent compressions generally do not. Even orbit preservation does not preserve dense-prefill states, number of distractors, or values.

**Transplant rigidity is conditional.** The invertible full-space identity AᵀR_new(d)B=R_old(d) over an open lag interval implies matching spectra. It does not preclude compensation on the actually used low-rank content subspace, changes that leave unused channels irrelevant, or learned adaptation accessing new hidden directions. It also does not logically mandate a specific displacement schedule: every positive table can be expressed as native-relative displacements.

**Integer-lag distinction in Astra02.** Its unique finite atomic equilibrium proof is correct for a continuous distance prior positive on an interval. A finite integer-lag Gram has finite rank on signed measures and is not strictly positive definite there, so its uniqueness proof does not directly transfer. For q=0, a finite-moment vector has an atomic representation with at most T+1 points by convex-hull geometry, but uniqueness may fail; a uniform measure on [0,π] has zero cosine moments at every positive integer lag and is a diffuse minimizer for that particular support/prior. This example is an assumption counterexample, not the usual RoPE [1/base,1] interval.

## 7. Audit of the assigned historical evidence

The important current owner is `docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`, fully read in addition to all assigned texts. Its §6 explicitly retracts the broad old non-identifiability theorem and incorrect gate, Fisher, carrier, and phase claims. The retraction at `.agents/worker_remediation_1/report.md:1–16` overrides the confidence rhetoric in the preserved historical body as scientific evidence.

Specific errors found independently in the assigned reports:

* `.agents/challenger_2/report.md` says the cosine Taylor series converges only for small phase. Cosine is entire; finite truncation error is the issue. Its cosine numbers are not Qwen loss measurements.
* `.agents/reviewer_1/report.md` endorses a fabricated condition number above 10^4 and reinterprets 81.2% curve-fit residual as measured Hessian curvature. Neither follows from the original C2 receipt. It calls a first/second-order expansion an exact risk law.
* `.agents/explorer_survey_1/report.md` calls FullLagP2 U a checkpoint content statistic, but its construction is native sine/cosine geometry without learned Q/K. Geometry-based success does not prove U's proposed mechanism.
* `.agents/worker_empirical_r3/r3_empirical_defense.md` and the blueprint assert that fixed channel count makes simultaneous short/long improvement information-theoretically impossible. Conservation of count alone gives no task-risk conservation law. The same report reverses the actual Cosh density direction in one paragraph.
* The blueprint's claim that choosing the native table restores Native-LoRA's 6.82 PPL while keeping EVQ-LoRA weights contradicts the weight/table-crossing evidence. Exact routing preservation requires the same complete weights, table, gain, and execution state, not merely switching one table between independently adapted checkpoints.
* The BM diagnostic is two selected instances. P success/L failure excludes the sufficiency of that late mask on those cached states; it does not uniquely identify a general prefill contamination mechanism, deny all positional mechanisms, or derive optimal frequencies.
* “Same sum of squared norms after regrouping” verifies algebra, not independent observables' equivalence, full-model measurement, or absence of value/readout effects.
* The .qoder material is an older harness inventory. Its embedded suggested commands, policies, and source-count heuristics are historical data, not current research instructions or frequency evidence.

The directly relevant live numerical owner, `docs/research/ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`, was read in full. Smooth MrBudget has lower remote unresolved response (.0494352 vs .235928), C26 (.000159185 vs .000240986), and C32768 (35.3793 vs 42.5180) than MrPro. It also has lower projection-operator MSE at all four listed ranges and lower weighted remote weak response in all 36 layers at three cutoffs. Yet its recorded development task result is worse. This forbids any claim that a positive combination of those audited costs alone selects the better frozen table. It does not refute a valid conditional continuity bound with actual margin slack.

The full `ROPE_RESEARCH_FAILURE_REVIEW_20260907.md` further records shared-frequency versus independent phase energy ratios .918–8.728 and finite-Jacobian errors 71.15–468.12 on the old p2 intervention. It explicitly warns that signed local responses alone are not a capability optimizer; this report's useful-score/MGF model adds the role and distribution assumptions needed for its own narrow guarantee.

## 8. What to use next

Use (1) as a stronger alternative to unsigned energy or plain SNR when the actual target is attention mass. Use (6) when a finite shared-bin/nested nuisance model is justified: it gives exact equal-channel allocations with or without a signed remote source term. Preserve original labels for frozen deployment; the source term and covariance have to describe the actual conditional computation or a separately defended transport model. No unspecified loss-gradient QP is necessary for this special case.

The remaining Qwen question is empirical identification of useful signed means and nuisance MGFs/covariance under real long prefill, or discovery of a simple general rule whose actual held-out outputs improve without those measurements. The mathematical construction does not make such measurements mandatory for every useful heuristic. It does prevent claiming an unmeasured heuristic as a theorem. Existing MrPro/P2/E1/Smooth evidence must retain its task, length, and development scope.
