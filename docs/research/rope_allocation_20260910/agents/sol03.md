# Sol 03 — audit of the Pro first-principles proposals

## Scope and evidence status

I read all three assigned files in full (3,091 lines total). They are analysis/plan documents, not primary run artifacts. Accordingly, I use their algebra when it can be checked directly and treat all reported model outcomes as owner-level claims requiring manifest/raw-output verification. I did not inspect server artifacts, run a model, or infer a universal rule from the reported scores.

The three files agree on a crucial distinction but do not finish the requested unification:

* EVQ-Cosh is an allocation rule for training or co-adaptation under a declared positional surrogate; it is not a mature-checkpoint transplant rule (`HYBRID_ROPE_FIRST_PRINCIPLES_SYNTHESIS.md:76-78`, `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md:1329-1334`).
* MrRoPE is a close training-free, non-uniform scaling baseline and a mixed-radix structural prior, but these documents do not derive it from EVQ or show that its geometry predicts task utility (`...dossier...:554-559`, `:1364-1369`).
* A frozen model attaches learned content coordinates to ordered rotary slots. Therefore a frequency multiset, positional Gram, or mixed-radix coverage alone is insufficient (`...dossier...:140-175`; `HYBRID_ROPE_FIRST_PRINCIPLES_SYNTHESIS.md:157-199`).

This means a correct unification is a shared **allocation framework with lifecycle-specific evidence**, not one curve claimed optimal in both regimes.

## Assumptions that survive, and assumptions that do not

### Surviving assumptions

1. Work in normalized log-frequency coordinates

   \[
   x_k=-\log\omega_k=a+Rz_k,\qquad 0=z_0\le z_1\le\cdots\le z_{K-1}=1.
   \]

   Fixing the actually sampled endpoints \(a,a+R\), pair count \(K\), rotary operator, and gain isolates interior allocation. This is the clean causal object (`...dossier...:166-175`).

2. Mixed radix is exactly a statement about adjacent log gaps. Define

   \[
   r_k=\frac{\omega_k}{\omega_{k+1}}=\exp(x_{k+1}-x_k),\qquad
   \prod_{k=0}^{K-2}r_k=e^R.
   \]

   Thus a MrRoPE radix schedule and an EVQ frequency allocation live in the same feasible simplex

   \[
   \Delta x_k\ge0,\qquad \sum_k\Delta x_k=R.
   \]

   This is the exact constructive bridge: MrRoPE describes the *factorization of span* into adjacent radices; EVQ supplies a criterion for how much span each factor receives.

3. The full sin/cos Gram and slow-collapse results remain valid within their stated positional surrogate. The documents correctly retain those results while warning that the strongest fixed-support experiment is not wholly in the deep \(\omega L\ll1\) limit (`HYBRID_ROPE_FIRST_PRINCIPLES_SYNTHESIS.md:157-161`, `:830-833`; `HYBRID_ROPE_TWO_DIRECTION_THEORY_AUDIT_20260905.md:329-333`).

4. For a frozen checkpoint, useful allocation is signed and content-conditioned. For a task margin \(m\),

   \[
   \frac{\partial m}{\partial s_{ij}}
   =\alpha_{ij}\langle\nabla_{o_i}m,v_j-o_i\rangle,
   \]

   so more evidence attention is beneficial only when the written value points in the correct downstream decision direction (`...TWO_DIRECTION...:88-103`).

### Assumptions rejected or narrowed

1. **Position-function redundancy implies cheap frozen slots.** False. If \(K\) pairs all use the same \(\omega\), the positional function span is only two-dimensional, while with \(Q=K=I_{2K}\), the content kernel \(\operatorname{diag}(R(\omega\Delta),\ldots,R(\omega\Delta))\) has rank \(2K\). Hence even perfectly redundant positional functions may carry independent content channels (`...FIRST_PRINCIPLES...:163-181`).

2. **A slow pair is suppressed by softmax.** False in general. Softmax removes a rowwise constant, not a term that is nearly distance-independent but varies with key content. The leading slow-frequency term \(q_k^\top k_k\) may be semantically decisive (`...FIRST_PRINCIPLES...:183-199`).

3. **A static scalar gain resolves local/long phase conflict.** False: gain changes logit scale, not the simultaneous phase constraints. The no-wrap interval condition in the first-principles file makes this explicit (`...FIRST_PRINCIPLES...:247-264`).

4. **Lower geometric distortion implies higher long utility.** Unsupported. The dossiers themselves state that stable rank, phase coverage, condition number, and frequency movement cannot be renamed task utility (`...dossier...:207-238`; `...TWO_DIRECTION...:79-125`). This also explains why the externally reported Smooth_MrBudget counterexample, once verified from its raw owner artifacts, is decisive against geometry-only ranking.

5. **The same analytic allocation should be used from scratch and after freezing.** Unsupported and contradicted by the ordered-slot coupling and installation-history decomposition (`...TWO_DIRECTION...:258-310`).

## Constructive unified rule: radix-factor allocation with lifecycle-specific marginal value

Let \(u_k=\Delta x_k=\log r_k\), so \(u\) is a nonnegative allocation of the fixed log span \(R\). Let \(J_{\rm pos}(u;\mu)\) be the declared EVQ positional surrogate under a relation-distance measure \(\mu\). It may be the existing full-sin/cos Gram/Cosh surrogate; its exact published definition must be preserved rather than replaced by an invented proxy. Let \(\mathcal U(u;\theta,\mathcal D)\) be a task functional such as paired full-trajectory margin or strict utility. The common allocation problem is

\[
\boxed{
\min_{u\ge0,\;\mathbf1^\top u=R}
J_{\rm pos}(u;\mu)
-\gamma\,\mathcal U(u;\theta,\mathcal D)
+\lambda\,D_N(u;\theta)
}
\tag{1}
\]

subject to fixed sampled endpoints, ordered slots, fixed operator/gain, and any declared minimum-gap constraint. Equation (1) is a framework, not a claim that all three terms are observable in every lifecycle.

The KKT condition for an interior gap is the concrete marginal-allocation law

\[
\boxed{
-\partial_{u_k}J_{\rm pos}
+\gamma\,\partial_{u_k}\mathcal U
-\lambda\,\partial_{u_k}D_N
=\nu,
}
\tag{2}
\]

with the usual complementary inequalities at \(u_k=0\). Each extra unit of log radix is assigned until its net marginal value is equal across active gaps. This is the direct EVQ–MrRoPE unification: EVQ defines positional marginal value, MrRoPE supplies the multiplicative radix coordinates and span conservation.

### Training from scratch / co-adaptation

At initialization the slot semantics are not yet installed. Set \(\lambda=0\), allow \(\theta\) to co-adapt, and use a predeclared training relation measure \(\mu_{\rm train}\):

\[
u^{\rm scratch}
=\arg\min_{u\in\mathcal S_R}J_{\rm EVQ}(u;\mu_{\rm train}),
\qquad
\theta^*=\arg\min_\theta L_{\rm train}(\theta,u^{\rm scratch}).
\tag{3}
\]

If the existing Cosh solution is the analytic minimizer of that declared surrogate, retain it exactly. Its radices are simply \(r_k^*=e^{u_k^*}\). This gives a precise sense in which EVQ yields a mixed-radix code: it chooses the radix factors by equalizing surrogate marginal value. It does **not** make the Cosh solution task-universal. The relation measure and surrogate are explicit assumptions, and end-to-end training remains the behavioral test.

A stronger scratch variant jointly optimizes \(\theta,u\) on training loss with the EVQ term as a regularizer:

\[
(\theta^*,u^*)=\arg\min_{\theta,u\in\mathcal S_R}
L_{\rm train}(\theta,u)+\beta J_{\rm EVQ}(u;\mu_{\rm train}).
\tag{4}
\]

This is mathematically legitimate because content coordinates adapt to the chosen slot labels. It should be compared with fixed-support controls; it cannot establish frozen deployment compatibility.

### Frozen mature deployment

For Qwen2.5-3B, native \(W=32768\), target 128K, keep the original ordered slots and define \(x=x^0+\delta x\). MrRoPE/EVQ supplies only a structural prior or reference direction \(d_0\); checkpoint evidence supplies the actual choice. Around the native point, solve one constrained trust-region problem:

\[
\boxed{
\delta x^*=\arg\max_{\delta x\in\mathcal C}
g_L^\top\delta x
-\frac{\lambda}{2}\delta x^\top H_N\delta x
-\frac{\rho}{2}\|B\delta x-d_0\|_2^2
}
\tag{5}
\]

where:

* \(g_L\) is the signed derivative of a predeclared long *functional* target, preferably paired content-fork/full-trajectory margins on development instances, not a geometry score;
* \(H_N\) is the pullback Fisher or a directly estimated quadratic approximation to actual native-output KL at the native checkpoint;
* \(B\delta x\) maps slot frequency changes to adjacent log-radix changes;
* \(d_0\) is the MrRoPE/EVQ structured radix proposal for 4x extension;
* \(\mathcal C\) fixes the chosen endpoint policy, preserves order, respects a declared trust radius, and retains one global static table.

Ignoring active inequalities, the stationarity equation is

\[
(\lambda H_N+\rho B^\top B)\delta x
=g_L+\rho B^\top d_0.
\tag{6}
\]

With inequalities, project the solution in the same positive-definite metric. This is one deterministic candidate, not a grid. If long labeled development data is forbidden, set \(g_L=0\) and call the result a **prior-constrained compatibility candidate**, not a predicted long optimum. If no native calibration is allowed either, use the external MrRoPE table unchanged as a baseline; the optimum is underdetermined.

Equation (5) correctly separates the two regimes. In scratch training, EVQ geometry can choose an allocation because semantics co-adapt. In frozen deployment, the only defensible movement is an ordered, trust-region modification whose long benefit and native cost are measured on the installed function.

## Counterexample and falsification checks

1. **Permutation check.** Frequency-only permutation must generally change behavior; joint permutation of complete rotary pairs plus the corresponding Q/K output coordinates and normalization parameters must preserve logits. Failure of joint parity is an implementation bug, not evidence for slot sensitivity (`...dossier...:177-201`).

2. **Repeated-frequency check.** Apply the rank-2 versus rank-\(2K\) construction above. Any proposed frozen score assigning zero cost solely because positional columns coincide fails this test.

3. **Row-shift check.** Add a constant to every key logit in a row; allocation loss based on raw logit MSE changes, attention does not. Functional/native costs should be invariant to this shift.

4. **Phase-conflict check.** For an active pair with local distance \(d_N\), long scale \(s=4\), and lifted-phase tolerances, verify whether

   \[
   3|\omega|\le4\eta_N/d_N+\eta_L/d_L.
   \]

   If violated, no single frequency meets both exact phase targets; the rule must rely on other slots, functional margin, or accept a trade-off.

5. **Geometry/task discordance check.** Compare the single candidate from (5) against native, official MrRoPE, and the EVQ-only prior using locked strict long outcomes. If EVQ/Mr geometry improves but task margin worsens, reject the geometry-to-utility claim while retaining the allocation framework. Smooth_MrBudget should be entered here only after its exact table, model, rows, and scorer are verified.

6. **Lifecycle crossing check.** For existing co-adapted weights, complete the fixed-weight support-by-shape factorial proposed in `HYBRID_ROPE_TWO_DIRECTION_THEORY_AUDIT_20260905.md:262-310`. A diagonal reversal without fixed-weight crossover supports installation-history dependence, not a universal runtime rule.

## Concrete minimal next comparison

For the target Qwen2.5-3B 32K→128K deployment, do not launch a candidate grid. Freeze one development set of legal paired compact/far relations and one native calibration set. Construct exactly one candidate with (5), using the official MrRoPE 4x table as \(d_0\) and the existing EVQ surrogate only as the smooth structural penalty. Compare four fixed arms: native, official MrRoPE, EVQ-prior-only, and the single function-calibrated solution. The decision endpoint is native-gated strict paired long generation, with full trajectory/EOS; geometry is diagnostic. P2 and E1-slot28 are useful priors only after their tiny-sample conditionality and exact owner artifacts are confirmed; they must not create extra search arms.

The constructive claim available before this experiment is narrow but real: **EVQ and MrRoPE are unified exactly as allocations of conserved log-span across multiplicative radices; the marginal allocation criterion must be positional during co-adaptation and checkpoint-functional during frozen deployment.** No assigned evidence supports a task-independent universal frequency curve.

## Coverage and omissions

All assigned texts were read linearly in contiguous bounded pages. No assigned lines were omitted. I did not ingest the underlying paper PDF, MrRoPE full paper, server JSON/JSONL, or failure transcripts because they were outside Sol 03's assigned full-file list. References to Smooth_MrBudget, P2, and E1 slot28 therefore remain coordination-level claims, not independently verified observations in this report.
