# Sol 05 — Audit and constructive correction for 6Pro full-row calibration

## Decision

The 6Pro proposal contains three sound pieces: the log-frequency decomposition, the exact EVQ change of variables, and the ridge-regression decomposition of representational fit versus frozen-weight compatibility. The proposed full-row calibration is also a material improvement over geometry-only proxies because it uses signed, content-conditioned Q/K coefficients and the complete softmax denominator.

It is not yet a valid allocation rule for preserving both local and long dependencies. Two issues are decisive:

1. A single full-row KL weights each subset of keys by its *native attention mass*. A long-distance group with native mass \(\pi_L\ll1\) may have its internal ordering destroyed while contributing only \(O(\pi_L)\) to the objective. Full-row coverage does not imply long-dependency coverage.
2. The fixed block map \(T(p)=SM\lfloor p/M\rfloor+(p\bmod M)\), combined with only the last query position of each document, aliases locality to one arbitrary block origin. It preserves distances within the query's final block but never tests a local relation that crosses an internal block boundary. Such a pair can be mapped from distance 1 to \((S-1)M+1=12289\) for \(S=4,M=4096\).

The constructive correction is a **constrained, stratum-balanced transported-row rule**. Optimize transported long-range conditional distributions and their competition against local keys, while explicitly constraining native/local compatibility. Compute the first update by a small projected quadratic program in log-frequency coordinates, then accept a step only after exact nonlinear losses and monotonicity are checked. This yields an actual candidate table and a matched reverse-direction control without claiming task success from the calibration proxy.

## What is mathematically reusable

Let \(x_j=-\log(\nu_j/\nu_{\rm ref})\), with gaps \(a_i=x_i-x_{i-1}>0\). Then \((x_0,A,\{a_i/A\})\), where \(A=x_{K-1}-x_0\), separates translation, support width, and internal allocation. This is correctly stated in the newest 6Pro proposal (attachment lines 19–68). MrRoPE's \(\nu_j=\omega_jS^{-m_j}\) gives

\[
a_i=a_i^{(0)}+(m_i-m_{i-1})\log S,
\]

so its radix increments are precisely added log-gap allocation. Fixing \(\sum_jx_j\) is an additional centroid/compression control, not a consequence of fixed endpoints (attachment lines 70–119; review lines 40–42).

The EVQ substitution \(h(u)=\phi'(u)\), \(\rho(\phi(u))=1/h(u)\), is exact and gives

\[
J[h]=\frac12\int_0^1\left[\frac{\alpha}{h(u)}+\beta(1-u)^2h(u)\right]du,
\qquad \int_0^1h(u)du=1.
\]

Its Cosh optimizer is therefore optimal for this stated interval-cost functional, not for Transformer behavior (attachment lines 180–260; independent numerical review at `docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:44-55`). This is the correct bridge: EVQ supplies an analytic allocation prior; empirical content-conditioned calibration supplies a checkpoint-specific correction.

The quadratic decomposition

\[
J_r(c_0,\nu)=J_r(c_r^*,\nu)+(c_0-c_r^*)^\top(G_r+\zeta I)(c_0-c_r^*)
\]

is valid for \(\zeta>0\). It cleanly distinguishes a basis that could fit a relation from the cost of making an existing checkpoint use that basis (attachment lines 406–483). It is not a theorem that EVQ or a calibrated table improves a deep model; 6Pro itself correctly limits that claim at lines 485–495.

## Audit of the current full-row objective

For a captured query/head row, the implementation stores raw Q, every raw K/V, native exact and runtime log-probabilities, and checks exact replay (`experiments/nongeometric_screen/pro_block_calibration.py:48-80`). It captures all 32,768 visible keys rather than reconstructing a row from selected keys (`...:25-33`, `...:70-78`). Natural documents and no forced answer trajectory are appropriate (`...:1-5`, `...:31-33`). Using Native frequencies at the shared MrPro gain is also a legitimate matched-gain isolation, provided it is never called the operational Native baseline (`...:22-30`; review lines 68–72).

However, the current capture takes only \(q=32767\) (`...:43-46`) and four heads per document (`...:43-45`). Four fit documents rotate across the 16 query heads, but each head is represented by only one fit document and every row has the same query offset at the end of a 4096-token block. Layers provide many rows, yet their hidden states are causally dependent observations from the same four documents. This is much thinner content and offset coverage than “all heads and full rows” might suggest.

### Why ordinary full-row KL suppresses long relations

Partition keys into groups \(G\), such as local \(L\) and remote \(R\). For teacher row \(P\) and candidate row \(Q\), the KL chain rule is exact:

\[
\mathrm{KL}(P\|Q)
=\mathrm{KL}(P_G\|Q_G)
+\sum_g P_G(g)\,\mathrm{KL}(P(\cdot\mid g)\|Q(\cdot\mid g)).
\]

Thus the remote conditional geometry is multiplied by \(P_G(R)\). If the native row assigns remote keys mass \(10^{-3}\), reversing a 90/10 conditional split in that group has conditional KL \(1.7578\) nats but contributes only \(0.00176\) nats to the full-row loss. This is not a corner case: local-dominated natural-text rows are expected to have precisely this imbalance.

The group-mass term does retain whether remote keys compete globally, but it does not force useful discrimination *within* the remote group. Conversely, equal conditional weighting alone would preserve remote ordering but could allow all remote mass to collapse. Both terms are required.

### The transport map is a hypothesis with a severe boundary artifact

Write \(p=bM+r\), \(0\le r<M\), and \(T(p)=SMb+r\). For two positions,

\[
T(p)-T(q)=SM(b_p-b_q)+(r_p-r_q).
\]

This is identity only within the same block and is not globally \(S(p-q)\). A pair straddling a boundary by one token changes from distance 1 to \((S-1)M+1\). At \(S=4,M=4096\), that is 12,289. For a randomized block origin, a pair at distance \(d<M\) crosses a boundary with probability \(d/M\); its expected mapped distance is \(Sd\), but the distribution is a mixture of \(d\) and \((S-1)M+d\), not a smooth scale transport. Matching only the expectation is inadequate for periodic rotary features.

The proposal openly describes the map as an assumption rather than a theorem (attachment lines 511–525; review line 64). The present sampling nevertheless hides its most important failure mode: with every query at 32767, all keys in the final 4096-token block preserve local distance, and there are no query rows immediately across each earlier boundary. A candidate can therefore look locally compatible under calibration while disrupting the same short dependency at another absolute offset.

Randomizing origins alone is insufficient: it exposes the discontinuity but does not turn the map into a realistic natural-language transport. It should be used as a robustness nuisance variable, paired with identity constraints and multiple query offsets.

### Frozen-state and target-content limits

The exact Q/K reconstruction is exact only conditional on the captured hidden states. Installing a new table changes earlier-layer states, so later-layer Q/K are no longer the captured Q/K. The proposal and review correctly acknowledge this (`pasted-text.txt:568-584`; review lines 55 and 66). The calibration can nominate a direction; only a full prefill on held-out content tests whether that direction survives the intervention.

Natural text avoids answer leakage, but it does not identify which low-mass remote keys carry a task dependency. The target is “preserve the checkpoint's native attention behavior under a chosen coordinate transport,” not “create missing long-range capability.” This distinction is especially important because the existing project evidence shows improved geometry without improved long-task behavior, and because total distractor count changes softmax requirements (major revision lines 193–209).

## Corrected allocation rule

### 1. Capture design

Retain full keys and signed Q/K coefficients. For each fit document/layer, sample query positions with offsets stratified relative to the 4096 boundaries: at minimum immediately before, immediately after, the block middle, and the document tail. Rotate the sampled heads so every head appears across more than one document and more than one offset. Keep documents 4–5 fully held out as already planned.

Use the identity map plus block-stretch maps with at least two origins fixed before fitting, one being the current origin and one shifted by \(M/2\). This is not a candidate grid: origin is a nuisance variable used to prevent a rule from exploiting block alignment. Report performance by origin; do not average away a sign reversal. If capture cost permits only one added origin, \(M/2\) is the maximally distinct deterministic choice.

### 2. Decompose every row into protected and transported terms

Let \(g(k)\in\{\text{local},\text{remote}\}\), with local defined by the model assumption \(|d_k|\le M\), and define teacher/candidate group masses \(P_g,Q_g\). Use:

\[
C_g(x;T)=\mathrm{KL}\big(P(\cdot\mid g)\,\|\,Q_x^T(\cdot\mid g)\big),
\]

\[
B_g(x;T)=\left[\operatorname{logit}Q_g^T-\operatorname{logit}P_g\right]^2.
\]

The conditional term preserves which keys win within a scale; the group-odds term preserves competition between that scale and the rest. When a group has zero or numerically negligible teacher mass, omit its conditional term for that row and record the omission; do not fabricate a uniform target.

Define protected losses

\[
L_{\rm nat}(x)=E[C_{\rm local}(x;I)+B_{\rm local}(x;I)+C_{\rm remote}(x;I)+B_{\rm remote}(x;I)],
\]

\[
L_{\rm locT}(x)=E_{T,o}[C_{\rm local}(x;T_o)+B_{\rm local}(x;T_o)],
\]

and the transported long objective

\[
L_{\rm longT}(x)=E_{T,o}[C_{\rm remote}(x;T_o)+B_{\rm remote}(x;T_o)].
\]

Average each displayed component over eligible rows with equal row/head weight, rather than weighting remote conditionals by their native mass. Keep the four components separate in receipts. The formula deliberately avoids claiming semantic relevance for every remote token; it preserves what the native checkpoint distinguishes within remote strata and whether that stratum remains competitive.

### 3. Produce the update with a constrained projected step

Start at the exact MrPro log-frequency vector \(x^M\). Let \(g_T=\nabla L_{\rm longT}(x^M)\), \(g_N=\nabla L_{\rm nat}(x^M)\), and \(g_L=\nabla L_{\rm locT}(x^M)\), computed by autograd through exact sin/cos and softmax.

Find a first-order direction by the convex QP

\[
\begin{aligned}
\delta^*=\arg\min_\delta\;&g_T^\top\delta+\frac{\mu}{2}\|\delta\|_2^2\\
\text{s.t. }&\delta_0=\delta_{K-1}=0,\quad \mathbf1^\top\delta=0,\\
&g_N^\top\delta\le0,\quad g_L^\top\delta\le0.
\end{aligned}
\]

The equal-sum constraint preserves the current compression centroid for the causal test; endpoints preserve support. The two gradient inequalities prevent first-order degradation of native and transported-local compatibility. \(\mu\) only sets direction norm; normalize \(\delta^*\) afterward, so it is not a performance-tuned hyperparameter.

If this QP has no direction with \(g_T^\top\delta<0\), the evidence is a useful obstruction: under the current rows/map and resource constraints, there is no local Pareto improvement over MrPro. Do not force a candidate by changing weights.

Otherwise choose the largest step \(\eta\), starting from the monotonicity limit and halving deterministically, such that exact nonlinear evaluation on fit rows satisfies

\[
x_{j+1}^+-x_j^+>0,\quad
L_{\rm nat}(x^+)\le L_{\rm nat}(x^M),\quad
L_{\rm locT}(x^+)\le L_{\rm locT}(x^M),
\]

and \(L_{\rm longT}(x^+)<L_{\rm longT}(x^M)\), where \(x^+=x^M+\eta\delta^*\). Then construct \(x^-=x^M-\eta\delta^*\), shrinking the shared \(\eta\) only if needed for ordering. This preserves equal endpoints, equal \(\sum x\), and matched perturbation norm, exactly the directional control requested in the proposal (attachment lines 628 onward).

Do not iterate a high-dimensional optimizer to convergence on four documents. One projected direction plus a deterministic line search is the smallest rule that tests whether the calibrated gradient predicts behavior and sharply limits overfitting. A later iteration is justified only if held-out calibration components agree in sign.

### 4. Acceptance and interpretation

Before any task result is opened, require the held-out documents and nuisance origins to show:

- no increase in native/local protected components beyond replay/numerical tolerance;
- a decrease in transported-long conditional KL and group-odds error separately, not merely their sum;
- no head or layer subset with a large protected-loss regression hidden by the mean.

Then evaluate MrPro, \(x^+\), and \(x^-\) at common gain on full prefills. The first decisive behavioral test should contain paired local relations at both sides of a block boundary and the same remote binding at controlled expanded distances. Follow with real dense 128K inputs so added-key competition is present. A calibration win with no held-out full-prefill or generation win rejects the fixed-state transport rule; it does not reject frequency allocation generally.

## Counterexample checks and predictions

1. **Mass-blindness:** a remote group with tiny native mass can undergo an arbitrarily large conditional permutation with vanishing full-row KL contribution. The stratum-balanced conditional term detects it.
2. **Boundary locality:** the current last-token capture predicts good local preservation inside the final block even if a one-token dependency across an internal boundary is catastrophically stretched. Multi-offset queries and shifted origins detect it.
3. **Identity-only solution:** minimizing only native-map KL returns or favors Native frequencies, while stretch-only fitting may sacrifice native behavior. The constrained formulation makes this tradeoff explicit rather than hiding it in an equal-weight average.
4. **All remote keys are not semantic dependencies:** a lower remote conditional loss may still fail tasks. This is why the report calls it a direction predictor and requires paired full-model tests.
5. **Gain interaction:** common gain isolates the comparison input, but frequency and gain remain non-orthogonal through softmax. The result applies only at the declared gain; it does not establish gain-independent allocation quality (review line 26).

## Consequence for scratch versus frozen deployment

For training from scratch, the compatibility constraints around \(x^M\) are not fundamental because content projections co-adapt. EVQ or another allocation prior should be judged by matched training curves and held-out task behavior under fixed support. The finite-feature Gram analysis can predict learning difficulty only within its fixed-content approximation (major revision lines 118–181).

For frozen deployment, the checkpoint-conditioned Q/K row losses are directly relevant, and the corrected projected rule is appropriate as a one-shot calibration method. For LoRA or full adaptation, replace the frozen-row gradient by the effective post-adaptation local quadratic (the Schur-complement idea at major revision lines 165–181) only after the actual trainable Jacobian is measured. Nominal LoRA rank alone does not specify the reachable function subspace.

The unifying statement supported by the mathematics is therefore conditional: EVQ and MrRoPE allocate finite log-frequency gaps; scratch training evaluates representational/learnability consequences after co-adaptation, while frozen deployment must solve a constrained compatibility problem around an existing frequency–subspace coupling. The proposed corrected rule supplies a concrete local solution to the latter. It remains an empirical hypothesis until held-out full-prefill and target behavior agree.

## Evidence and scope

All five assigned files were read in full. No GPU job was launched and no paper/runtime source was edited. The only computation was a CPU check of block-boundary distances and the KL-chain-rule weighting example. The review document records that capture had started but optimization and model benefit were not yet complete (`docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md:57-72`), so this report does not state or imply a completed 6Pro result.
