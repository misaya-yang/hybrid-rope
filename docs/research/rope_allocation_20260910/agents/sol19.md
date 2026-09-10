# sol19 — historical closure audit and constructive allocation rule

## Decision

The historical record does **not** prove a universal impossibility theorem for frequency allocation, and it does not close the causal loop with six necessary-and-sufficient observables. It supports narrower, useful conclusions:

1. an unordered spectral summary cannot certify a frozen checkpoint, because rotary slots are labeled by the learned Q/K coordinates;
2. the discrete scale-orbit count is not a stable physical selector;
3. small unweighted table error does not certify behavior;
4. fixed-support allocation and support retargeting are coupled experimentally;
5. static geometry is a diagnostic, not a task-success objective.

The positive route is therefore a **role-conditioned signed-margin allocation problem**. EVQ is recovered in a scratch/co-adapted special case with exchangeable channels, constant useful signal, and a declared interference covariance. MrRoPE becomes a structured frozen-checkpoint feasible set or initialization. Frozen deployment requires a labeled finite-slot optimization using task-conditioned source-versus-distractor means and covariance. This is constructive, but it does not yet authorize a new Qwen table: the signed calibration object has not been measured on the decisive rows.

## 1. Claim ledger: what survives and what must be retracted

### 1.1 Universal non-identifiability: retract; retain class-specific counterexamples

`worker_falsification_1/report.md:447-456` quantifies over “any structural metric” in a named family and then states three kinds of non-identifiability. The proof does not support that quantifier.

* The permutation construction at `:462-482` proves only that **permutation-invariant** maps (multisets, spectra, symmetric frame potentials) cannot identify frozen behavior. An ordered map can distinguish the two tables. More strongly, the scalar map `Phi(Omega)=R(Omega;M,D)` identifies the chosen risk by definition; it is model/data dependent, but it is a counterexample to the literal universal claim.
* The ULP construction at `:486-498` refutes the discontinuous orbit-count statistic. It says nothing about continuous ordered maps.
* One observed C2 approximation does not construct the claimed continuous epsilon-to-zero family in `:456`. The report assumes such a family and an eigenspectrum rather than measuring or proving it.

Correct theorem boundary: **no member of the actually tested model-blind class is a sufficient frozen-deployment certificate on the observed interventions**. The evidence does not rule out ordered, checkpoint-conditioned, task-conditioned low-dimensional statistics, nor an exact risk functional.

### 1.2 Slot-19 Hessian and Fisher claims: retract

The record repeatedly converts “81.2% of C2 squared residual is in slot 19” into “81.2% of curvature is in slot 19.” These are different quantities. The unsupported conversion appears in `worker_falsification_1/report.md:507-517`; the same document earlier calls the number residual concentration. No assigned artifact provides a measured Fisher matrix, Hessian eigenvalues, `kappa(F)>10^4`, `lambda_1/lambda_2>10^2`, or alignment of its top eigenvector with slot 19.

There are two further mathematical errors:

* `:506` identifies the empirical Fisher with the loss Hessian. Equality needs regularity, correct specification, and expectation under the model distribution; it is not automatic for empirical next-token loss.
* `:517` treats movement MAE `0.001223` as an L1 norm and asserts `Delta L=0.004615`. MAE differs from L1 by a factor of K, while the reported gate scores do not establish that quadratic decomposition.

The remediation record correctly retires local Taylor gating after phase shifts of 22.74 and 90.97 radians (`challenger_remediation_1/report.md:23-25`, `:85-105`). Its replacement path integral is an identity, not a cheap pre-hoc predictor: evaluating gradients along the whole intervention path still requires model/data computation. The finite rotation bound is safe but too loose to rank candidates after saturation.

### 1.3 “Necessary and sufficient six observables”: retract

`worker_falsification_1/report.md:539` calls six objects necessary and sufficient without a sufficiency proof. The list mixes:

* local derivatives (`H`, `J`, Fisher), which do not determine finite interventions;
* moment summaries that need distributional assumptions;
* a softmax approximation based on an average distractor and LLN, which does not control correlated extremes;
* `T(a,R,z)`, an input parameterization rather than an observable.

The remediation itself concedes two problems: it unifies `H` and `J` as contractions of one Jacobian Gram object and reclassifies `T` as coordinates (`challenger_remediation_1/report.md:26-28`). That is a correction, not “zero remaining theoretical defects.” Even the full distribution of all layer activations under all prefixes plus decoder dynamics would be closer to sufficient; the six summaries are useful diagnostics only.

### 1.4 Support-retargeting “mechanism”: observation survives, explanation retracts

The reversal is strong evidence that range and interior allocation cannot be treated independently. But `worker_falsification_1/report.md:600-603` says multiplying every interval by the same enlarged R “destroys” only nonlinear allocations and that uniform spacing preserves relative harmonic ratios. Multiplication by a common scalar preserves the ratios of all adjacent log-intervals for **both** uniform and nonuniform z. The algebra does not imply a sign reversal or privilege uniform z. It restates that absolute phase differences change.

Valid claim: the controlled reversal demonstrates interaction between `(R,z)` and learned/trained behavior. Its mechanism remains empirical and may include training support, coefficient co-adaptation, finite-lag phase coincidences, and task distribution.

### 1.5 What the historical success/failure evidence actually establishes

* Same-multiset collapse is decisive evidence for labeled slots in frozen checkpoints, not for scratch training where weights may co-adapt.
* FullLagP2 is a conditional local success on Qwen-1.5B at 64K and a mixed/failing transfer elsewhere. The old report's causal stories about “premature phase saturation” and architecture “scrambling” (`worker_falsification_1/report.md:152`) are hypotheses, not identified mechanisms.
* MrRoPE-Pro is an empirically robust heuristic in tested settings. Its fast/slow cutoffs and progressive transition are not derived optima. The remediation correctly calls it “satisficing” (`challenger_remediation_1/report.md:121-145`).
* The manuscript audit correctly warns that rank and whitening omit learned coefficient magnitudes (`critic_r1/r1_reviewer2_audit.md:47-78`) and that the geometry-to-loss chain contains unbounded approximation steps (`:141-164`).
* The historical “VICTORY CONFIRMED” audit is unreliable as a closure certificate. It claimed zero hallucinated mathematics (`auditor_victory_document_2/report.md:9-17,215-224`) while coexisting with the unsupported Fisher spectrum and universal theorem above.
* `teamwork_preview_document_2/DOCUMENT_REVIEW_REPORT.md:860-868` is internally inconsistent: it says there is no broken mathematics, then relies on mechanisms such as capacity conservation and KV interference that its own earlier critique does not establish as general causal explanations.

## 2. One framework, two regimes

For an operation/example role `r`, source token `s`, hard distractor `d`, head/layer `h`, and table `nu=(nu_0,...,nu_{K-1})`, define the signed contrast

```
Z_r(nu) = sum_{h,k} [ A^c_{rhk} (cos(nu_k Delta_s)-cos(nu_k Delta_d))
                    + A^s_{rhk} (sin(nu_k Delta_s)-sin(nu_k Delta_d)) ] .
```

`A^c,A^s` are the actual content-conditioned Q/K coefficients, with source and distractor labels retained. Let

```
mu_r(nu)=E[Z_r(nu)],    Sigma_r(nu)=Cov(feature contrasts),
psi_r(t;nu)=log E exp(t(Z_r-mu_r)).
```

For `N_r` distractors, a union/Chernoff certificate has the form

```
P(any distractor beats source) <= N_r exp[-t mu_r(nu)+psi_r(t;nu)] .       (1)
```

The finite-MGF version is preferable when available. Under a centered sub-Gaussian envelope `psi <= t^2 v/2`, optimizing t yields the standardized-margin criterion

```
J_r(nu)=mu_r(nu)^2 / (2 v_r(nu)) - log N_r .                              (2)
```

This directly represents the requested behavior: useful signed source margin, coherent interference, and distractor multiplicity. It also exposes failure cleanly: zero or wrong-sign mean cannot be repaired by reducing covariance.

### 2.1 Training from scratch / joint co-adaptation

Before specialization, suppose channels assigned to frequency bin b are exchangeable, each contributes the same protected useful mean `a_b>0`, has independent variance `sigma_b^2`, and shares a coherent bin nuisance `gamma_b^2`. With `n_b` channels,

```
mu_b = n_b a_b,
v_b  = n_b sigma_b^2 + n_b^2 gamma_b^2,
SNR_b(n_b)= n_b a_b^2/(sigma_b^2+n_b gamma_b^2).                           (3)
```

This utility is increasing and concave:

```
d SNR_b/dn = a_b^2 sigma_b^2/(sigma_b^2+n gamma_b^2)^2 > 0.
```

Allocate the integer budget by diminishing returns:

```
maximize sum_b q_b SNR_b(n_b),  n_b in Z_+,  sum_b n_b=K,                 (4)
```

with endpoint/coverage constraints. A one-unit greedy allocator is exact for separable discrete-concave utilities; general finite-lag covariance uses the DP below. In the continuum limit, constant `a`, exchangeable channels, and a covariance operator with diagonal ridge plus nested-tail/Green component reduce to the EVQ inverse-covariance problem. The Cosh density is therefore a **conditional relaxation**, not a universal law. If `a_b` varies or signed means differ, the KKT condition becomes an active-set inverse-covariance rule and is generally non-Cosh.

This is the regime in which unordered density reasoning can be legitimate: slot identities may be permuted together with learned weights during training. It does not predict frozen table swaps.

### 2.2 Frozen deployment

In a frozen checkpoint, k is a permanent label. Estimate `mu_{r,k,b}` and joint covariance for assigning existing slot k to one of a predeclared ordered bins b using held-out calibration operations. Protect native behavior with hard constraints, rather than an arbitrary weighted penalty:

```
min_r J_r(nu) >= eta_r,
P(native loss/margin violation for role r) <= alpha_r,
nu_0 >= ... >= nu_{K-1}, endpoints fixed.                                 (5)
```

MrRoPE supplies a safe structured domain: retain fast native slots, bound slow-slot phase, and distribute the finite log-compression budget through the transition. These are constraints/initialization supported by history, not fixed truths.

For a discretized set of allowed labeled moves `b in B_k`, define a cost from the exact empirical log-MGF certificate plus infeasibility flags. With local transition costs (or an augmented state carrying sufficient covariance statistics), solve

```
D[k,b,c] = cost(k,b,c) + min_{b'<=b,c'} D[k-1,b',c'] ,                    (6)
```

where c is used compression/count budget. Backtracking gives an exact finite-K monotone assignment for the declared model. Full dense cross-slot covariance destroys this simple separability; then use the exact integer quadratic/conic problem or retain covariance sufficient statistics in state. Do not diagonalize merely to preserve DP convenience.

## 3. Counterexample checks

1. **Permutation.** Equation (5) changes when nu values are permuted because `mu_{r,k,b}` retains k. Scratch objective (4) is invariant only under its explicit exchangeability assumption.
2. **Zero useful signal.** If all `a_b=0`, EVQ-style covariance minimization cannot certify selection. The optimizer must report no positive guarantee.
3. **Coherent shared noise.** From (3), SNR saturates at `a_b^2/gamma_b^2`; piling channels into one slow bin has diminishing value. IID-noise reasoning would miss this.
4. **Negative covariance.** Cancellation may improve a joint assignment. A diagonal score can reverse the correct ordering; retain the signed covariance/MGF.
5. **Support retargeting.** Recompute the actual finite-lag features and moments after changing R. Scaling normalized z is not a causal proof and cannot inherit the fixed-support ordering.
6. **Finite phases.** Use exact sine/cosine responses or a finite MGF. Local Taylor curvature is inadmissible when `|delta nu| Delta` is not small.
7. **Geometry/task disagreement.** A table can reduce collision/rank/response distortion and still lower signed source margin. Geometry is allowed as a diagnostic or constraint only after the task margin has been defined.

## 4. Decision-sufficient next step

No broad grid is needed. On the already generated, identical development inputs for MrPro, Smooth_MrBudget, E1 slot28, and P2, compute `Z_r`, its signed mean, joint finite-MGF (or a justified upper envelope), and `log N_r` separately for source-dependent long rows and native/short rows. Freeze labels, prefixes, decoding, ordering, initial state, and table hashes. Do not use generation correctness to fit the moments.

The rule earns one frozen candidate only if, before looking at held-out generation labels, it:

* rejects Smooth_MrBudget where unsigned geometry preferred it;
* preserves E1's slight long-panel direction without claiming confirmation from the tiny sample;
* represents P2's long/short tradeoff rather than averaging it away;
* satisfies the native-role constraints in (5).

If the signed Q/K margin still ranks Smooth above MrPro on the failing rows, stop. The missing mechanism is then downstream of that measurement level: source labeling, upstream prefix formation, value transport, head/output mixing, or autoregressive decoder trajectory. Adding another frequency-shape proxy would not resolve it.

## 5. Evidence boundary and corpus notes

All 18 assigned files were read end-to-end. The two large JSON artifacts were parsed across every record; their role here is coverage and boundary checking, not evidence for a frequency selector. `development_rows.jsonl` contains 1,536 per-row outputs (1,024 relation, 512 content) but no table-arm field, so it cannot by itself attribute an allocation effect. `surrogate_boundary_scan.json` contains 280 geometry-grid points and reports 218 flagged failures (77.86%); this shows the surrogate's broad boundary behavior, not task success.

The Qoder harness/canvas files document prior workflow and presentation claims. They provide no independent mathematical validation. Compilation and manuscript-verification reports establish build/data transcription facts only; they cannot certify causal derivations.

## Final corrected boundary

There is no historical proof that constructive frequency allocation is impossible. There is strong proof that a model-blind unordered scalar cannot certify frozen deployment, and strong evidence that geometry-only rankings can fail. A defensible unification is conditional: EVQ is the exchangeable co-adaptation covariance limit; MrRoPE is a frozen labeled feasibility structure; the common objective is a signed relevant-versus-distractor finite-MGF margin with explicit native-operation constraints. The missing evidence is the calibrated, source-labeled moment object at the layer/operation level and its transport stability, not another hand-designed frequency curve.
