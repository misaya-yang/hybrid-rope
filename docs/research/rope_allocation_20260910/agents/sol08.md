# Sol08 — failure-record audit and surviving EVQ–MrRoPE allocation theory

## Verdict

The assigned record supports a common **coordinate system and conditional optimization problem**, but it does not support closure of an EVQ–MrRoPE behavioral family.  The physical object is the ordered log-frequency vector

\[
x_k=-\log \omega_k=a+Rz_k,\qquad 0=z_0<\cdots<z_{K-1}=1.
\]

`(a,R)` is sampled support, `z` is interior allocation, and slot identity remains part of the mature object.  Training-time exact-range evidence identifies `z` as causal, while the frozen crossing and later permutation failures show that the same numerical table cannot be interpreted independently of learned Q/K coordinates.  These conclusions are explicitly separated in the foundations (`ATTENTION_AWARE_ALLOCATION_THEORY_STATE_20260826.md:23-43,58-65`; `TIMELINE.md:79-96,115-129`).

The strongest constructive conclusion is therefore:

> **EVQ and MrRoPE may supply different structural priors over the same ordered log-frequency allocation, but the final allocation must be selected by the signed training/deployment risk appropriate to its lifecycle.**

For training from scratch this is a bilevel risk with weight co-adaptation.  For frozen Qwen deployment it is a native-compatibility-constrained signed long-risk movement.  Static geometry is a regularizer and diagnostic only.

I found no assigned primary source that proves the details or optimality of MrRoPE's mixed-radix construction.  The assigned corpus only classifies it as a distinct training-free conversion (`ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`, family table) and records its citation identity.  Consequently, this report does not claim that EVQ is a limit of MrRoPE, that MrRoPE is a discretization of EVQ, or that their published rules form a closed optimal family.

## What EVQ actually proves empirically

1. **A causal coordinate, not a universal curve.**  At fixed sampled endpoints and log span, the three-seed 151.9M intervention changes only the interior frequencies and gives Cosh-minus-uniform NLL `+0.026/-0.281/-0.176/-0.146` at 256/512/1K/2K.  This identifies allocation `z` during matched training (`ATTENTION_AWARE...:58-60`).
2. **The direction is broader than Cosh.**  The M4 record says a deformation-matched exponential is competitive, so the observations do not identify Cosh-shape necessity (`TIMELINE.md:67-77`; `ICLR2027_RESEARCH_SYNTHESIS_20260819.md`, Sections 4.1 and 7).
3. **Support and allocation interact.**  When both grids are retargeted, the OOD ordering reverses (`+0.060/+0.227/+0.460`).  Thus `z` and `(a,R)` do not contribute additively (`ATTENTION_AWARE...:59-60`).
4. **Behavior is co-adapted.**  Frozen 50M PPL is `7.14/76.20/23.05/7.16` for Geo/Geo, Geo/EVQ, EVQ/Geo, EVQ/EVQ.  Static rank rises from `4.57` to `12.54` in the worst cell while PPL collapses (`FULL_ROPE...:357-370`).  This directly rejects static-rank selection.
5. **Frozen sensitivity survives, detailed-profile necessity does not.**  Same-support mature OLMo changes from `0.56%` to `60.47%`, but a coarse ramp reaches `61.04%`; that identifies frozen sensitivity to ordered `z`, not a unique Cosh or uniqueness curve (`ATTENTION_AWARE...:61-63`).

The mathematically exact EVQ result is narrower: the Cosh density uniquely minimizes its declared convex surrogate.  Full-RoPE geometry gives the exact static identity

\[
r_2(R)=\frac{2K}{1+(K-1)\bar c},
\]

and low-frequency subspaces collapse toward `span{1,Delta}` (`FULL_ROPE...:140-214`).  Neither result orders language-model behavior.  Fourier-comb optima, length-order reversal, and exact periodic aliasing are explicit counterexamples (`FULL_ROPE...:255-316`).

## Failure record: what is closed and what survives

### Closed as selectors

- Cosine-only collision, full-subspace rank/logdet, and any monotone combination of unsigned geometry statistics.
- The attention-measure `kappa` rule: its first-order and finite-swap rankings disagree, triggering preregistered Branch C (`KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md:27-74`).
- The `rho proportional to w^(1/3)` LeRoPE oracle under structural softmax curvature.  It lies beyond EVQ in the direction away from LeRoPE (`alpha=-0.957`) and requires a signed trajectory-aware risk instead (`LEROPE_PROFILE_ORACLE_AUDIT_20260820.md:124-175`).
- A universal density derived from an assumed distance prior.  Additive per-frequency utility collapses all channels to the same frequency unless interference is added, while candidate rankings change with the chosen kernel (`DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md`).
- The scratch-note arcsine claim.  Its kernel-to-potential argument is invalid and the numerical shape is not U-shaped (`optimization_notes.md:155-169`).
- Exact universal short no-harm from a changed static table.  Under standard RoPE, this permits only the Native frequency multiset up to the stated aliases; approximate no-harm is a risk constraint (`ROPE_OPTIMALITY...`, Sections 4.4 and 5.3).
- Any inference from emulation rank bounds to re-adaptation.  The corrected notes explicitly state that the former does not bound the latter without a loss-local model and a joint Q/K budget (`optimization_notes.md:173-204`).

### Surviving causal coordinates

1. **Ordered physical table** `x`, not a free `base/exponent` textual pair.
2. **Support** `(a,R)` and **interior allocation** `z`, distinct by intervention but interacting in outcome.
3. **Slot association** between learned rotary subspaces and frequency/dilation.  Same-multiset permutation collapse means an unordered spectrum is insufficient (`TIMELINE.md:115-129`).
4. **Lifecycle** `scratch`, `frozen`, or `adapted`; these have different gradients.
5. **Signed task/data risk**, including endpoint weights or a Pareto vector when no scalar weights are defensible.
6. **Native compatibility budget** for frozen deployment, plus ordering/support constraints.
7. **Operator fields** such as attention gain and routing, which must remain separate if changed.

## Constructive unification

### 1. One prior in ordered log-frequency space

Let `q_E` be an EVQ structural prior over normalized log-frequency coordinate `z` (for example its declared Cosh density), and let `q_M` be a **declared** MrRoPE relation-scale prior induced by its radix place values after mapping them into the same physical `x=-log omega` coordinate.  Without assuming either is behaviorally correct, define

\[
q_\eta(t)=(1-\eta)q_E(t)+\eta q_M(t),\qquad 0\le\eta\le1.
\]

A deterministic finite initialization is the ordered midpoint quantile rule

\[
z_k^{(0)}=F_{q_\eta}^{-1}\!\left(\frac{k+1/2}{K}\right),
\]

followed by endpoint anchoring or an order-preserving projection into `Z_delta`.  This is the only safe sense in which EVQ and MrRoPE are unified before observing risk: they are alternative priors/demand measures quantized into the same finite ordered table.  `eta` must be fixed by the scientific workload or treated as a comparison, not tuned on the final benchmark.  The quantile construction is an initialization, not a task-success theorem.

### 2. Scratch-training rule

For training algorithm `W_T(x;S,xi)`, choose

\[
x_{\rm scratch}^*\in\arg\min_{x\in\mathcal X}
\mathbb E_{S,\xi}\,\mathcal L_{\rm deploy}(W_T(x;S,\xi),x)
+\lambda D(x,x^{(0)}),
\]

where `D` is only a weak prior penalty and may be zero.  The exact hypergradient is

\[
\nabla_x\mathcal R_{\rm out}
=\partial_x\mathcal R_{\rm out}+J_T^T\nabla_W\mathcal R_{\rm out},
\qquad J_{t+1}=\partial_W\Psi_tJ_t+\partial_x\Psi_t.
\]

At an isolated differentiable inner optimum this becomes

\[
R_x-R_WH_{WW}^{-1}H_{Wx}.
\]

The second term is the co-adaptation response and is indispensable (`ROPE_OPTIMALITY...:408-435`).  Thus an EVQ/MrRoPE prior may initialize scratch training, but selection must use trained risk.  This is exactly the regime in which the fixed-support EVQ evidence is informative.

### 3. Frozen Qwen2.5-3B, 32K to 128K rule

Start from the verified Native ordered vector `x^N` and define `h=x-x^N`.  Estimate on frozen, predeclared calibration rows:

- `g_L`, the signed gradient of a declared 128K long-risk endpoint;
- `F_N`, the full-model curvature of Native-window output divergence;
- optionally `H_L`, a positive semidefinite long-risk curvature estimate.

The exact band derivative contains the signed task adjoint:

\[
g_k=\mathbb E\sum_{\ell,h,i,j}
\frac{\partial\ell}{\partial s_{\ell hij}}\,
\omega_k\Delta_{ij}
[c_{\ell hij,k}\sin(\omega_k\Delta_{ij})
-d_{\ell hij,k}\cos(\omega_k\Delta_{ij})],
\]

so unsigned Q/K norms, Fisher magnitudes, or Gram statistics cannot replace it (`ROPE_OPTIMALITY...:316-362`).

Construct exactly one candidate by solving the convex local program

\[
\begin{aligned}
\min_h\quad &g_L^Th+\tfrac12h^TH_Lh
+\tfrac{\beta_E}{2}\|h-h_E\|_{W_E}^2
+\tfrac{\beta_M}{2}\|h-h_M\|_{W_M}^2\\
\text{s.t.}\quad &\tfrac12h^TF_Nh\le\epsilon,\\
&x^N+h\in\mathcal X_{\rm ordered},
\end{aligned}
\]

where `h_E` and `h_M` are the endpoint/support-matched EVQ and MrRoPE prior movements.  Set `beta_E=beta_M=0` unless the priors are predeclared; otherwise they regularize only directions weakly identified by calibration.  If `H_L=0` and no order bound is active, the risk-selected solution reduces exactly to

\[
h^*=-\sqrt{\frac{2\epsilon}{g_L^TF_N^{-1}g_L}}F_N^{-1}g_L.
\]

With order constraints, solve the corresponding cone KKT projection (`ROPE_OPTIMALITY...:364-406`).  This is a concrete, non-grid rule.  It protects Native behavior explicitly and moves slots only where the signed long benefit justifies compatibility cost.  The final nonlinear 32K and 128K endpoints remain the acceptance test.

## Counterexample checks required before family promotion

1. **Geometry/behavior discordance:** a candidate may improve `r2`, collision, phase coverage, or smoothness and still lose behavior.  The 50M crossing proves this pattern; the coordination-level Smooth_MrBudget result is consistent with it but was not independently verifiable from Sol08's assigned corpus.
2. **Support reversal:** rerun the interpretation with support held fixed and then target-retargeted; a reversal means the result is conditional, not invalid.
3. **Slot permutation:** preserve the multiset and permute slot assignment.  Collapse falsifies any unordered-spectrum theory.
4. **Sign mirror:** compare the local candidate `h*` with the endpoint/norm-matched `-h*`.  Failure to predict direction rejects the signed local model.
5. **Lifecycle crossing:** for a scratch claim, compare self-consistent trained/table cells; for a frozen claim, keep weights fixed.  Do not pool them.
6. **Endpoint dissociation:** require both strict long capability and Native-window retention; tail NLL alone is insufficient (`TIMELINE.md:98-113`).
7. **Locality radius:** verify the nonlinear risk along the single selected direction at the predicted trust radius; curvature only justifies a local candidate.

The coordination prompt also mentions P2 conditional benefit and an E1 slot-28 tiny-development positive.  No assigned owner file contained those artifacts, so they are not evidence in this report.  They can at most initialize the sign or active-set hypothesis after their rows, scorer, checkpoint, and holdout identity are verified.

## Claim boundary

Supported: a common physical coordinate; allocation causality during matched training; strong lifecycle/slot co-adaptation; impossibility of a distribution-free behavioral optimum; exact conditional KKT, signed gradient, hypergradient, and local trust-region equations.

Unsupported: a universal EVQ–MrRoPE curve; optimal mixed-radix allocation from the assigned corpus; task success from geometry; exact Native preservation after any nontrivial static move; a monotone family law from P2 or slot 28; family closure from a single Qwen protocol.

## Coverage statement

All 25 files listed in `assignments/sol08.md` were read completely.  The 1,006-line TeX note was read as contiguous lines 1–500 and 501–1006; the 415-line evidence audit was re-read in bounded contiguous chunks after a combined read truncated.  No assigned lines were omitted.  I did not read unassigned MrRoPE primary-paper text, Pro attachments, raw JSON/JSONL experiment owners, or other agents' reports as evidence.  Accidental search-result snippets naming other reports were ignored.
