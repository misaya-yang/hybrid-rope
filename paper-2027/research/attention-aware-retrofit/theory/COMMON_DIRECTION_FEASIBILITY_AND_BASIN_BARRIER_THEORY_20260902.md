# Common-direction feasibility, likelihood-to-winner barrier, and the Native--long basin structure

- **Date:** 2026-09-02
- **Status:** PARTIALLY SUPERSEDED (same day, later fact update). The user-supplied
  latest fact state (2026-09-02, recorded in
  [`FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902`](FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md)
  §0) voids the anchors of this note as follows:
  - **Voided:** the framing of the §7.3 margin-gradient common-direction candidate as
    "the only consistent choice" and as a registered entrance gate. The 18-sample
    64-D behavioral-gradient route failed its unopened holdout with inconsistent
    8K/16K behavior and has exited the main line (fact F). Pillar 1's convex-hull
    algebra remains valid as mathematics, but its object (behavioral gradients) is
    no longer an authorized or promising measurement, and no conclusion here may
    be quoted as support for that route.
  - **Challenged:** Pillar 3's bimodal-basin hypothesis modeled only the
    Native-vs-long split and does not account for the task-dependent radius in the
    latest natural QA facts (same `log_s4` strong on RULER 16K, weak on Hotpot 16K,
    recovered at Hotpot 8K; fact D). It is retained only as an audited conjecture.
  - **Retained:** Pillar 2 (likelihood is smooth, generation is piecewise constant;
    gain preserves key argmax) is independent of the gradient route and remains
    compatible with the updated facts, subject to the new memo's review.
  This note authorizes nothing and is no longer an anchor for next steps.
- **Original status:** internal theory note. Establishes one derivation, one decomposition,
  and one labeled hypothesis. **Authorizes no compute, opens no queue, proposes no
  new allocation formula, gain, ramp, or boundary.**
- **Scope:** mature-checkpoint static pure-`z` retrofit; addresses the two open
  method gaps recorded in `INDEX.md` §3.5 — the Native--long basin bridge and the
  capability-conversion barrier — and gives the registered §7.3 common-direction
  candidate of
  [`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902`](../results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md)
  a closed-form decision rule.
- **Relation to closed classes:** this is not a static score functional of a shared
  table (`INDEX.md` §3.4 items 1–11). Its objects are behavioral gradients measured
  on task cells, and its conclusion is a feasibility statement about directions, not
  a new content-blind predictor. Escape route is the one already registered in the
  two-day synthesis §7.4.

## 1. Summary of results

1. **Theorem (common direction = convex-hull geometry).** The registered robust
   common-direction problem
   \[
   \max_{d,\gamma}\ \gamma-\tfrac{\lambda}{2}\lVert d\rVert^2
   \quad\text{s.t.}\quad g_j^\top d+\gamma\le0,\ j=1,\dots,J
   \]
   has the closed-form solution
   \[
   \bar g^*=\arg\min_{c\in\operatorname{conv}\{g_1,\dots,g_J\}}\lVert c\rVert,
   \qquad
   d^*=-\bar g^*/\lambda,\qquad
   \gamma^*=\lVert\bar g^*\rVert^2/\lambda.
   \]
   In particular **a strictly improving common direction exists iff
   \(0\notin\operatorname{conv}\{g_j\}\)**, independently of \(\lambda\);
   \(\lambda\) sets only the step size. When \(0\in\operatorname{conv}\{g_j\}\),
   any representing convex combination is a Carathéodory conflict certificate
   naming exactly which task/scale cells are mutually obstructing. This is the
   multi-gradient (Pareto-stationarity) condition known from multi-objective
   optimization; the content here is the measurement problem and the
   interpretation for frequency tables, not the optimization.
2. **Decomposition (likelihood is smooth, generation is piecewise constant).**
   Teacher-forced NLL is a smooth function of the table; greedy generation is a
   piecewise-constant function of it, changing only when some visited decoding
   state crosses a zero-margin boundary. Therefore an arbitrarily large canonical
   likelihood improvement can be behaviorally invisible, and converting it into
   generation requires per-visited-state margin injections that exceed the local
   competitor gaps. This locates the 2026-09-02 capability-conversion barrier
   exactly: source signal entered canonical logits, but not above the winner
   margin on the visited path. It also explains, without new runs, why both
   readout rescues and free headwise gain failed (§4).
3. **Hypothesis (bimodal basin along dilation).** Under three labeled
   assumptions — a resolvability band per frequency slot, a smooth Native tax
   weighted by learned slot usage, and thresholded long-horizon benefit per slot —
   the constrained objective along any shared dilation coordinate is bimodal: a
   Native-compatible basin near \(d=0\) and a long-capable basin near the
   transported point, separated by a region worse than both. The teacher-forced
   gradient at the Native end is predicted to be concentrated on near-threshold
   slots and approximately zero on deeply aliased slots, which would make the
   long basin unbootstrappable from exact Native by scalar-only local
   optimization. This is a hypothesis with registered CPU-falsifiable
   predictions (§5), not an established result.

Taken together, the three parts convert the two vague bottlenecks into sharp
statements: the basin conflict is a **convex-hull containment question about
measurable behavioral gradients**, and the QA breakpoint is a **margin-vs-mean
functional mismatch** of the current objective class.

## 2. Pillar 1 — closed form of the common-direction gate

### 2.1 Problem

Let table space be \(\mathbb R^K\) (one exponent increment \(d_k\) per frequency
slot, \(\omega'_k=\omega_k S^{-(m_{0,k}+d_k)}\)). For each calibration cell
\(j\) (task \(\times\) scale \(\times\) source-counterfactual cell as in the
two-day synthesis §7.2), let
\[
g_j=\nabla_d\,\mathcal L^{\rm margin}_j\big|_{d=0}
\]
be the behavioral gradient of the cell's margin loss (softplus of
\(1-[\ell_y-\max_{v\ne y}\ell_v]\), plus the suffix/EOS and source-preference
terms) evaluated at the current normalized-index table \(m_0\). The registered
candidate seeks one direction improving all cells:
\[
\text{(P)}\qquad \max_{d,\gamma}\ \gamma-\tfrac{\lambda}{2}\lVert d\rVert^2
\quad\text{s.t.}\quad g_j^\top d+\gamma\le0\ \ \forall j.
\]

### 2.2 Theorem and proof

**Theorem.** (P) has a unique optimal \(d^*\), and with
\(\bar g^*=\arg\min_{c\in\operatorname{conv}\{g_j\}}\lVert c\rVert\)
(unique since the norm is strictly convex and the convex hull compact),
\[
d^*=-\bar g^*/\lambda,\qquad \gamma^*=\lVert\bar g^*\rVert^2/\lambda,\qquad
\text{optimal value}=\lVert\bar g^*\rVert^2/(2\lambda).
\]
Consequently \(\gamma^*>0\iff 0\notin\operatorname{conv}\{g_j\}\).

**Proof.** The objective is strictly concave in \(d\), constraints are affine,
and Slater holds (\(d=0,\gamma=-1\)), so KKT is necessary and sufficient. With
multipliers \(\mu_j\ge0\), the Lagrangian is
\[
L=\gamma-\tfrac\lambda2\lVert d\rVert^2-\sum_j\mu_j(g_j^\top d+\gamma).
\]
Stationarity in \(d\) gives \(d=-(1/\lambda)\sum_j\mu_jg_j\); stationarity in
\(\gamma\) gives \(\sum_j\mu_j=1\), so \(\mu\) ranges over the simplex and
\(\bar g=\sum_j\mu_jg_j\) ranges over the convex hull. Substituting, the dual
function is
\[
q(\mu)=-\frac{\lVert\bar g\rVert^2}{2\lambda}
+\frac1\lambda\sum_j\mu_jg_j^\top\bar g
=\frac{\lVert\bar g\rVert^2}{2\lambda},
\]
so the dual is \(\min_{\mu\in\Delta}\lVert\bar g(\mu)\rVert^2/(2\lambda)\),
whose solution is the minimum-norm point \(\bar g^*\) of the convex hull.
Strong duality gives the optimal value. For \(\gamma^*\): the constraint binds
as \(\gamma^*=\min_j(-g_j^\top d^*)=(1/\lambda)\min_jg_j^\top\bar g^*\). The
projection KKT on the simplex gives \(g_j^\top\bar g^*=\lVert\bar g^*\rVert^2\)
for \(j\) in the active set and \(\ge\) off it, hence
\(\min_jg_j^\top\bar g^*=\lVert\bar g^*\rVert^2\) and
\(\gamma^*=\lVert\bar g^*\rVert^2/\lambda\). Finally,
\(\gamma^*>0\iff\bar g^*\ne0\iff0\notin\operatorname{conv}\{g_j\}\). ∎

### 2.3 Consequences

- **C1 — \(\lambda\) separates feasibility from step size.** The existence of a
  common improving direction is a pure geometric statement about the measured
  gradients; \(\lambda\) only rescales \(d^*\). Finite-step concerns are handled
  by the separately frozen step bound of the registered protocol, not by
  re-tuning \(\lambda\).
- **C2 — conflict certificates.** If \(0\in\operatorname{conv}\{g_j\}\), the
  dual minimizer supplies \(\mu^*\in\Delta\) with \(\sum_j\mu^*_jg_j=0\); by
  Carathéodory, at most \(K+1\) cells suffice. The support of \(\mu^*\) names
  the conflicting cells. For the present project the predicted informative
  certificate is \(\{\text{Native-retention cells},\ \text{long-QA margin
  cells}\}\): if that is exactly what \(\mu^*\) supports, then within the
  linearized class a single shared static table provably cannot improve both
  sides, which closes the §6.2 first research goal with a structural negative
  rather than another sweep. If instead \(\mu^*\) is supported on redundant
  cells, \(d^*\) is the bridge direction and the gate passes. Either outcome is
  a result; the theorem fixes the report format in advance:
  \((\bar g^*,\gamma^*,\operatorname{supp}\mu^*)\).
- **C3 — headwise factorization changes the hull, not the law.** With per-head
  increments the gradients become block vectors in \(\mathbb R^{KH}\) and the
  convex hull lives in a higher-dimensional space, where containing the origin
  is harder. This is consistent with
  [`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](../results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md):
  per-head coordinates improved long QA (cells escape the shared-table conflict)
  while the retention gate still failed, because the bounded two-axis
  construction applies the same frozen directions to every head and cannot
  realize the per-head retention-saving directions at the heads where they
  conflict with the long margin.
- **C4 — first-order scope.** The theorem speaks about directions at \(m_0\).
  It says nothing about finite steps, curvature, or path effects; that boundary
  is the same one already stated for linearized re-adaptation arguments
  (`INDEX.md` §2.3 O6). A measured \(\gamma^*>0\) is an entrance condition for
  a frozen finite-step table, not a guarantee of endpoint behavior.

## 3. Pillar 2 — the likelihood-to-winner decomposition

### 3.1 Two functionals with different continuity

Fix the model and vary only the table \(d\). The teacher-forced
canonical-answer loss \(L_{\rm TF}(d)\) is smooth in \(d\). Greedy generation
\(G(d)\) is piecewise constant: at each visited decoding state, the output
token is \(\arg\max_v\ell_v(d)\), which is invariant under all table changes
that preserve the ordering at that state. Hence:

> **Generation invariance.** Any table change that preserves the logit argmax
> at every state reachable along the greedy path leaves the complete generated
> sequence exactly unchanged, no matter how much canonical likelihood it adds.

The 2026-09-02 bridge observation is an instance: index moved canonical answer
NLL from `5.123` to `3.695` nats (far source-use `+1.438` CI-positive), while
greedy QA stayed statistically flat. There is no contradiction; the two
quantities measure different functionals.

### 3.2 Budget form

At a visited state \(t\) let the Native competitor gap be
\(\delta_t=\max_{v\ne y_t}\ell_v(0)-\ell_{y_t}(0)\ge0\) (how far the correct
token sits below the leader). Flipping that state requires injected margin
\(\Delta M_t(d)>\delta_t\). If the total positive margin injection over visited
states is \(B(d)=\sum_t[\Delta M_t(d)]_+\), then the number of newly correct
visited states is at most \(B(d)/\bar\delta\) for any lower bound
\(\bar\delta\) on the gaps at wrong states. Canonical-likelihood gains do not
lower-bound \(B(d)\): they may concentrate on states already won, where they
are behaviorally redundant. This is the quantitative shape of the barrier —
the bridge effect is an O(1) sum over a whole answer, i.e. O(0.05–0.1) per
token, while leader gaps at wrong states are typically O(1) or more.

### 3.3 Path dependence compounds the mismatch

The margin condition of §3.2 is stated on the Native-visited path. Once \(d\)
flips any visited state, all downstream states move off the canonical prefix,
where no teacher-forced measurement at \(d\) was taken. A sufficient condition
for complete-answer correctness is therefore the pathwise margin condition
under the self-induced prefix — a moving, non-convex feasibility set in \(d\).
First-order certificates on the canonical path (including Pillar 1 applied to
answer-margin cells) are necessary for generation improvement, never
sufficient. EOS is one more visited state in the same condition: stopping
requires the EOS token to beat all continuations at the induced stopping
position.

### 3.4 Why the three attempted fixes had to fail (postdiction)

- **Source-contrast decoding** \(\tilde\ell=(1+\alpha)\ell_{\rm index}
  -\alpha\ell_{\rm Native}\) can flip an argmax only at states where the two
  arms' leaders differ. Where index and Native share the same wrong leader —
  the generic case when the table moved likelihood mass without moving the
  leader identity — the contrast at the decisive tokens is near zero. Observed
  \(`+0.00079\) is consistent with near-total leader agreement.
- **Existing-candidate rerank by source score** ranks by a likelihood contrast
  (source present vs ablated), again a mean-type functional; a candidate can
  be source-dependent and wrong or source-independent and right. The oracle
  macro `0.18656` shows the winner was in the set; the ranking statistic is
  simply not the winner-margin functional.
- **Free headwise gain** scales attention scores \(s\mapsto cs\). For any
  \(c>0\), \(\arg\max\) of the key softmax is unchanged: **gain provably cannot
  alter which key is attended, only the concentration of the mixture around the
  existing ranking.** Retrieval errors are ranking errors; gain cannot move
  mass across keys. Teacher-forced loss rewards concentration whenever the
  top key is already right (mostly true on short in-distribution data — the
  observed `3.56→2.31`), which is a pure sharpening shortcut; at generation,
  sharpening amplifies whatever leads, including wrong continuations, matching
  the observed EOS collapse `178→108` and longer wrong answers.

### 3.5 Consequence for objectives

Any objective that reads only canonical-prefix likelihoods (natural NLL,
teacher-forced CE, answer NLL, source-contrast likelihood) belongs to the
smooth functional class of §3.1 and cannot, by construction, control the
piecewise-constant generation map except through explicit margin and stopping
terms on visited states. This retroactively justifies the margin-plus-EOS
objective of the registered §7.3 candidate as the only consistent choice inside
pure-`z`, and predicts in advance that any future gain/rerank/decoder variant
that still does not touch internal winner margins will fail the same way.

## 4. Pillar 3 — bimodal basin structure along dilation (labeled hypothesis)

This section is a model-based hypothesis, not an established result. It exists
to turn "the basins are disconnected" into falsifiable finite-difference
predictions compatible with the CPU-identifiability entrance requirement of
`INDEX.md` §6.5.

### 4.1 Assumptions

- **A1 (resolvability band).** Slot \(k\) separates sources at token lag
  \(\Delta\) only when \(\omega'_k\Delta=\Omega(1)\), and keeps a decodable
  phase over a context of length \(L\) only while the wrap count
  \(\omega'_k L/2\pi\) stays within what the trained decoder can disambiguate.
  This is the standard wavelength picture behind PI/YaRN-style transport,
  stated here as a definition, not as new physics.
- **A2 (smooth Native tax).** Q/K readouts are co-adapted to the Native phase
  velocities (the ordered-coupling object of
  [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md)).
  Under per-slot log shift \(u_k=-(m_{0,k}+d_k)\log S\), retention loss grows
  smoothly — locally quadratically — in \(|u_k|\), weighted by learned usage of
  slot \(k\). Same-multiset permutation collapse rules out reassignment as a
  compensation channel, so the tax is per-slot and cannot be redistributed.
- **A3 (thresholded long benefit).** Slot \(k\)'s contribution to horizon
  \(L_{\rm target}\) turns on when \(\omega'_k L_{\rm target}\) crosses below
  the aliasing wrap threshold — a sigmoid in \(\log\omega'_k\) with O(1) width
  — because below threshold the slot's phase pattern over the long context
  becomes distinguishable for the first time.

### 4.2 Consequences

Under A1–A3, along any shared coordinate \(d_k\equiv\delta\):

1. Long capability \(C(\delta)\) is a **staircase**: slots cross their
   thresholds one by one, in an order determined by \(m_0\) (the
   normalized-index ordering is exactly the threshold-crossing order). Native
   retention \(R(\delta)\) is **smooth and decreasing** (A2). The constrained
   problem \(\max C\) s.t. \(R\ge R_{\rm gate}\) is therefore generically
   bimodal: a feasible Native basin at \(\delta\approx0\) with \(C\approx0\),
   and a long basin near the transported point, feasible only if enough
   thresholds are crossed before \(R\) passes the gate.
2. **Disconnection is a quantitative spectral condition, not a universal
   law**: it depends on where the thresholds sit relative to the retention-tax
   curve. This is consistent with the repository's standing refusal to claim
   universality, and with the fact that official YaRN-4 fails the same 4K
   retention gate (`0.6588`) in the headwise owner.
3. **Bootstrapping impossibility at the Native end.** At \(u=0\), the gradient
   of the long-context teacher-forced loss with respect to a deeply aliased
   slot's log-frequency is predicted to be near zero: its phase wraps many
   times across the context, the trained weights carry no readout for those
   wraps, and the slot's contribution phase-averages (Riemann–Lebesgue-type
   cancellation). Slots far below threshold are also insensitive. The gradient
   concentrates on the near-threshold band, which is sparse at the Native
   point. Scalar-only local optimization from exact Native therefore cannot
   see a direction toward the long basin — consistent with the Native-start
   arm's stagnation (mean \(\alpha=0.0966\), 16K loss still `6.75`, retention
   `1.0462`), which on this account is gradient starvation, not optimizer
   failure.
4. **No smooth descent from the long end either.** From the transported point,
   the retention gradient is smooth and nonzero (A2), and every retention-
   improving step dismantles threshold crossings discretely (A3), so a local
   optimizer cannot stop at an intermediate point that keeps both. The ridge
   between basins is traversed by neither initialization, matching the
   headwise owner's verdict that initialization selects the side of the
   frontier.
5. **Historical consistency.** Sharp transitions along dilation coordinates
   have been observed before in this project's history (Video DiT: behavior
   flipping between neighboring \(\tau\) values). Those are internal
   consistency evidence for A3-style thresholding, not proof.

### 4.3 Registered falsifiable predictions (not executed)

- **P1 — gradient concentration.** At the Native table, per-slot finite-
  difference gradients of a 16K teacher-forced loss concentrate on
  near-threshold slots and are approximately zero on deeply aliased slots.
  Measured at both the Native and the transported table, the concentration
  pattern must shift with the threshold band.
- **P2 — non-convex continuation.** Continuing the Native-start arm along
  shared \(\alpha\), 16K loss is non-convex with a high plateau, while 4K
  retention degrades smoothly and monotonically.
- **P3 — midpoint is worse than both basins.** A midpoint table
  (\(\alpha\approx0.5\)-equivalent along the frozen movement direction) is
  worse than both endpoints on long tasks and no better than the transported
  point on retention.

P1–P3 are CPU/cheap diagnostics in principle, but they are registered here, not
authorized; running them is a separate decision under `INDEX.md` §6.5. If P1
fails (aliased slots carry usable gradient at Native), the bootstrapping-
impossibility mechanism is wrong and the Native--long conflict must be
re-attributed, even if the conflict itself remains.

## 5. Claim table

### Established by derivation (no new empirical input)

- Closed form of (P): \(d^*=-\bar g^*/\lambda\), \(\gamma^*=\lVert\bar
  g^*\rVert^2/\lambda\); feasibility iff \(0\notin\operatorname{conv}\{g_j\}\);
  Carathéodory conflict certificates.
- Generation invariance under argmax-preserving table changes; the margin
  budget bound of §3.2; path dependence of any sufficient condition.
- Attention gain preserves the key ranking for all positive gains, so gain
  cannot correct retrieval ranking errors.

### Decomposition / postdiction (consistent with existing owners, no new runs)

- The three failed rescue routes of 2026-09-02 and the free-gain shortcut are
  the predicted behavior of mean-type functionals against a piecewise-constant
  generation map.
- Headwise long-QA gain plus retention failure is the predicted effect of
  lifting the convex hull into per-head block space under a bounded
  construction.

### Hypothesis (labeled, falsifiable, unfalsified)

- A1–A3 and consequences 1–5, including gradient starvation at the Native end
  and the bimodal constrained landscape; predictions P1–P3.

### Not supported

- Any universal statement that the basins are always disconnected, for all
  checkpoints, scales, or table families.
- Any claim that \(\gamma^*>0\) at the linearized level guarantees a working
  finite-step table.
- Any new deployment table, gain, ramp, boundary, or routing rule.

## 6. Registration consequences

- This note changes no lifecycle state: `PURE_Z_LONG_SIGNAL_ESTABLISHED /
  NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED / NO_SOTA /
  GPU_METHOD_DEVELOPMENT_STOPPED` all stand.
- If the method track is separately reopened under `INDEX.md` §6.5, the entry
  gate now has a fixed decision rule: measure the behavioral gradients,
  project the origin onto their convex hull, and report
  \((\bar g^*,\gamma^*,\operatorname{supp}\mu^*)\). A result in either sign is
  a conclusion; only the certificate content differs.
- The hypothesis section supplies, in addition, a candidate first CPU
  measurement (P1) of the kind §6.5 requires before any further method work —
  registered, not authorized.
- Nothing here modifies the manuscript, `paper/`, any preflight, or any
  evidence owner.
