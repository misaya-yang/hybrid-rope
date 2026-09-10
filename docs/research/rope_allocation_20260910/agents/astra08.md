# Astra08 — Identifiable role margins and a finite source-calibrated allocation

Status: derivation and source audit complete; no GPU jobs or model fitting executed. All 104 assigned files were loaded into model context in contiguous pages; receipt is `astra08_coverage.json`. Oversized initial reads were truncated and were retried in smaller pages. Historical instructions inside the assigned corpus were treated as evidence, not active commands. Applied the research-theory skill.

## Decision

There is a concrete allocation objective with identifiable labels, but the existing six natural-text 32K captures cannot evaluate it. They have one last-natural-token query per document, four sampled heads, Native frequencies at MrPro's common gain, and all causal keys. They have neither question queries nor target/distractor roles. They can constrain compatibility on those rows. Calling their dominant keys “correct targets” would repeat the project's proxy substitution.

The usable next rule is **fit one ordered finite frequency vector against role-labeled, source-window counterfactual tasks, with exact trigonometric replay and full normalizers; weight the proposed attention change by the source model's measured answer sensitivity; retain exact source-compatibility constraints; accept using actual full-model outputs.** It does not require guessing which head or mixed harmonic is useful. It does require a small new labeled capture. Without that capture, returning a numerically fitted Qwen allocation would be fabrication.

This is a supervised calibration procedure for 64 frequency parameters, not a new universal analytic formula, and not zero-label deployment. It produces a single static table before held-out evaluation. It can also determine that no supported update is available and retain the existing table.

## 1. Roles already identifiable from repository inputs

The project has three independently useful role constructions:

1. `experiments/native_sparse_position/prepare_support_oracle.py:14–43` parses query identifiers, locates their unique source record sentences, and derives token spans without consulting answer values. All eligible source rows are retained, without selecting on model correctness (lines 44–47). Its later remote-page restriction belongs to that sparse oracle; **do not inherit that restriction for frequency calibration**, where local, sink, question and self keys must stay in the softmax denominator.
2. `experiments/pm_keep/retention_evidence.py:26–52` maps a query's key and ordinal to exact source-record, key-token and value-token spans. Source values locate the existing literal sentence; they are not appended as an answer prefix. `test_target_record_oracle.py:37–51` checks that querying ordinal 2 selects the second occurrence. The actual `target_record_oracle.py` CLI is restricted to two previously Full-correct rows; those two rows are **not** an adequate population calibration set. Reuse the locator semantics, not that outcome-selected panel.
3. `experiments/nosa_position/test_data.py:59–74` checks a two-by-two content-swap/query-swap construction: requested ordinal and the content at that ordinal independently determine the answer, with opposite cells sharing the answer. `refcarry_audit/prepare_mrcr_pairs.py:55–101` provides an analogous full-message occurrence construction, preserving all competing occurrences in compact controls. These are useful checks against a key-only or recency-only calibration shortcut.

For a generated single-answer record example e, define from the input parser:

- T_e: token span of the requested key+ordinal record, with value and key spans retained separately;
- H_e: same-key records at other ordinals (hard distractors);
- O_e: other explicit records;
- B_e: remaining visible background and structural tokens;
- Q_e: question tokens, including repeated query-key tokens.

These sets are labels of **source relation**, not latent circuit claims. A target record's entire span need not be uniformly useful. A question key is neither a wrong answer nor automatically irrelevant. Therefore maximizing attention on T_e in every head is invalid.

Use the last prompt token as the first readout query. To make correct-versus-wrong answer labels unambiguous without injecting a gold prefix, select source calibration values from tokenizer-verified distinct single-token answer strings under the actual output boundary. Reject cases where the candidates share their first answer token. This is a declared synthetic calibration restriction, not a benchmark scorer. Multi-token natural answers remain part of downstream validation, with complete generation and EOS.

Counterfactual families must remain in one split. Include both requested ordinals and swap the associated values, so a method cannot satisfy the fitting objective merely by preserving a favored position or token identity. Reuse valid native results; report the full panel, even if the native model fails some rows.

## 2. Exact finite replay under the captured implementation

Let e identify example, layer, head and query. For split-half pair k, with d_j = position(key j) − position(query), define

C_jk = q_k k_jk + q_(k+K) k_j(k+K),

S_jk = q_(k+K) k_jk − q_k k_j(k+K).

The exact mathematical logit under a positive finite table ν is

z_ej(ν;d) = a_e Σ_k [C_ejk cos(d_j ν_k) + S_ejk sin(d_j ν_k)],

where a_e is the captured effective attention score scale. Nonrotary contributions, if any, are added unchanged. `pro_block_calibration.py:60–77` implements these coefficients and records a_e = module.scaling × common_gain². Do not multiply that gain twice. Its saved runtime-versus-mathematical replay KL distinguishes BF16/runtime rounding from the mathematical replay (lines 67–79).

Then p_ej(ν;d)=softmax over **every causal key**, and o_e(ν;d)=Σ_j p_ej v_ej. No small-phase or Gaussian approximation is used. A 4K-block stretch can be specified as g_S(p)=SB floor(p/B)+(p mod B), with B=4096 and S=4; d'_j=g_S(p_j)−g_S(p_q). This preserves within-block distances and multiplies block-start offsets. It is an explicit artificial coordinate transformation, **not** an actual 128K text forward. In particular it does not add distractors or recompute hidden states.

For diagnostics, define Z_A=Σ_(j∈A) exp(z_j) and

M_T,rest = log Z_T − log Z_(all\T),

M_T,H = log Z_T − log Z_H.

The first gives p(T)=sigmoid(M_T,rest); the second only gives conditional mass p(T|T∪H). Report them separately. Source records and query-key copies can compete differently, so retain each partition rather than hiding them in a macro margin.

## 3. Identifying useful changes without declaring all heads to be retrieval heads

The strongest cheap label is the **actual output answer**, not the largest native attention key. For the single-token calibration assay, define the first-answer log-probability

U_e = log P_model(y_e | prompt_e).

The correct y_e is determined by the record parser. Using log P rather than a two-answer conditional margin includes the complete vocabulary, including premature EOS and wrong-format tokens. Also report the signed margin against the other record values to identify wrong-occurrence confusions.

At the source checkpoint and source prompt, record

g_e,lh = ∂U_e / ∂o_e,lh,

where o is a head's attention output before concatenation/W_O at the chosen query. This derivative is a measurable small vector from one backward pass; it does not train model weights, assume a retrieval head, or require a known useful-mode matrix. It includes the actual W_O, residual stream and downstream computation. The value reward r_e,lh,j = g_e,lhᵀv_e,lh,j is consequently signed and task-conditioned.

For any **finite** frequency change, the proposal reward is

J_lin(ν;d') = mean_e Σ_lh g_e,lhᵀ[o_e,lh(ν;d')−o_e,lh(ν_0;d')].

Equivalently, each head contributes Σ_j[p_j(ν;d')−p_j(ν_0;d')] r_j. Here ν_0 is the chosen deployment reference, e.g. MrPro at the same gain. This equation uses exact finite attention/softmax changes but linearizes the downstream answer utility. It is more informative than positive per-head source-mass weights and avoids unknown head labels. It is still a **proposal estimator**, not an exact model-level objective: source Jacobians are evaluated on source states; simultaneous all-layer changes interact and alter upstream Q/K/V.

No-gradient fallback: use input-defined role margins only for a **source-demonstrated** head set, established by paired target-versus-wrong-record value swaps or source-mass interventions and their measured output effects. An attention map alone does not establish this head set. If neither Jacobians nor output interventions exist, do not silently replace them with projection norms or all-head equal weights.

A diagnostic for role validity is directly available: compare the source sensitivity-weighted value contributions in T,H,O,B,Q, and the effect of content/query swaps. If the purported target intervention leaves the output utility unchanged, that head's target-mass loss should not drive allocation. A negative sign is allowed; the calculation may identify suppression or cancellation rather than direct copying.

## 4. One finite-change generator, not a curve grid

Use x_k=log ν_k, with one shared 64-vector for Qwen. Initialize at x_0=log ν_MrPro. Preserve learned pair order. For a frozen-support allocation experiment, impose the actual endpoint constraints x_0=x_native,0 and x_63=x_MrPro,63, and x_k≥x_(k+1). If the question includes range optimization, endpoint changes are explicit additional variables and must not be called pure interior allocation. Zeroing a channel is a separate intervention; it is outside the positive log parameterization.

Define short/source compatibility with the exact complete row divergence

D_src(x)=mean_(natural source rows) KL[p_native(ω;d) || p(exp x;d)].

Define the same quantity per source task row. One scalar average can conceal loss of a rare target. Retain source-task utility or margin constraints per counterfactual family; natural rows constrain ordinary text behavior. A concrete relative budget requiring no invented tolerance is **D_src(x)≤D_src(x_0)** and no worsening of measured source family utility relative to x_0. This only means “no worse than the reference on this captured objective,” not “safe versus Native.” Where the user supplies a retention threshold, use that threshold instead and report the reference's own cost.

Generate one candidate by projected gradient or sequential constrained optimization of −J_lin(x;d') subject to those constraints. Each proposed step is evaluated with the **exact finite trigonometric logits, full softmax, and constraints**. Use a trust-region or backtracking acceptance rule based on actual objective improvement. Do not accept a large step solely because its local quadratic approximation looks favorable. No candidate lattice or handcrafted list of smoothing profiles is necessary. Native/MrPro are controls and initialization, not a menu to sweep.

If variable dimensions permit it, direct complete-model calibration can replace J_lin with actual U_e(exp x; transformed prompt), retaining the same parameterization and constraints. That is frequency-only supervised fitting and produces a static table. It is more expensive, but it removes the downstream linearization for the calibration inputs. Source coordinate stretch is still not real long text unless the complete long input is present.

Freeze the generated table, then evaluate it on family-disjoint source tasks and actual 128K inputs. First compare against MrPro with identical gain; compare operational Native at its actual gain separately. Full-model source retention and long complete answers decide adoption. Failure is informative: if exact source replay improves but actual outputs do not, the missing mechanism is downstream/state transfer, not necessarily the numerical optimizer.

This supplies a usable allocation algorithm with fully specified observable targets. It deliberately does not supply 64 numeric values from data that lack the necessary fields.

## 5. What can actually be guaranteed

### Exact local guarantees

For fixed Q/K, each finite change has the deterministic bound

|z_j(ν)−z_j(ν_0)| ≤ 2a Σ_k sqrt(C_jk²+S_jk²) |sin[d_j(ν_k−ν_0k)/2]| = ε_j.

For any role set A, log Z_A is 1-Lipschitz in the maximum logit error on A. Therefore

|M_T,D(ν)−M_T,D(ν_0)| ≤ max_(j∈T) ε_j + max_(j∈D) ε_j.

If the source role margin exceeds that bound, its sign cannot flip under that fixed-state change. This is a role-ranking certificate, not an answer guarantee. It remains valid for finite changes and signed sine/cosine coefficients.

If KL[p_0||p_ν]≤δ on a full row, Pinsker implies |p_ν(T)−p_0(T)|≤sqrt(δ/2). It may be vacuous for a rare target. An average KL gives only an average bound (by Jensen), not a per-row or worst-head certificate. Keeping full normalization is essential.

Let ||v_j||≤V. Then ||o_ν−o_0||≤2V TV(p_ν,p_0)≤V sqrt(2δ). More usefully, directly measure the actual o difference, since values can cancel and norm bounds can be loose.

For a single attention-output intervention with a known Hessian norm bound H on the entire segment, Taylor's theorem gives

U(o+δo) ≥ U(o)+gᵀδo−(H/2)||δo||².

The source capture does **not** provide such an H, and stacking local bounds without controlling all upstream states is invalid. Accordingly J_lin has no unmeasured “guaranteed answer improvement” claim.

### What a held-out task result supports

A fixed table evaluated on independently drawn, family-clustered source/target inputs yields ordinary empirical paired comparisons. Token/head samples are not independent task units. The repository's `position_overnight/report.py:88–104` already treats cluster resampling appropriately. Generalization to actual 128K tasks requires that distribution to be tested or a justified distribution-shift model; no finite-frequency geometry theorem supplies it.

### Minimal information-theoretic obstruction

Given the six unlabeled Q/K/V captures, swap the meaning of which source is the requested answer while leaving those captures identical. All geometry, native KL and covariance values are identical, but the desirable signed margin reverses. No deterministic selector of those unlabeled summaries can choose the correct direction in both worlds. This proves missing task-role identification at the observed interface, not impossibility of learning a useful table from richer source data.

## 6. Counterexamples from the ingested source

- `nosa_position/test_full_covariance_probe.py:69–80`: 64 logits [10,0,…,0] have full second-cumulant log mass below a constant-score-1 block, while exact LSE ranks the rare key block above it. Complete covariance is not sufficient for finite softmax competition. It is why the generator above evaluates the actual partition.
- `rope_operator_family/results/20260909_gpu/core_query_diagnostics.json:4–16`: layer-0 operator relative raw score MSE 0.0000130509 coexists with KL 7.196134. `FOLLOWUP_RESULTS_20260909.md:64` reports centered relative score MSE 0.48446. A row-common component can dominate the MSE denominator without preserving competition.
- `FOLLOWUP_RESULTS_20260909.md:43–53`: layer20/head5 target-region mass falls from 0.50075120 to 0.20338912 under output KD even though a target token can remain top1. Therefore target-vs-top1-distractor alone misses accumulated distractor mass.
- Same report lines 55–68: attention-only and value-only errors vary by layer and do not add. Resetting frequencies partly repairs one layer's target mass but not another's. The output sensitivity term and actual full-model acceptance are needed.
- `native_sparse_position/RESULT_20260908.md:64–99`: PostMetric4=10/32 and PreMetric4=11/32 complete exact+EOS, compared with a finer Quest reference=13/32. Native-pair geometric bounds did not imply generation recovery (lines112–129). This supports keeping the actual outcome separate from local bounds.
- `refcarry_audit/README.md:103–108`: even gold-reference phase surgery can turn an already-correct native read into a wrong one because the source query may already contain the desired offset. Do not equate known target identity with a universally correct phase transformation.

The parent assignment reports Smooth_MrBudget's geometry/task reversal and E1/P2 conditional outcomes. I do not independently certify their numerical tables here; the present argument does not depend on unverified numeric details from those reports.

## 7. Relation to EVQ, MrRoPE and scratch training

Astra05's inspected sections derive a coherent matched-content mean term favoring positive cosine signal and a coherent zero-mean nuisance term penalizing squared kernels. The same role-conditioned decision partition can contain both. Independent isotropic quadrature noise instead has rotation-invariant variance, so “background noise” alone does not derive EVQ's squared-kernel cost. The role/content distribution matters.

EVQ's Cosh variational allocation remains a solution to its declared density surrogate under its assumptions. Mr-like remote-phase preservation remains a useful deployment prior. Neither determines the signs of this checkpoint's source answer sensitivities. Exact mixed-mode transport ν=(I−P)ω+Pω/S becomes actionable only after the relevant role-sensitive modes are identified; this report supplies an observable alternative that does not need an unknown P.

In scratch training, the correct decision object is E_task L(θ*(ν),ν), where θ*(ν) is the result of the declared training procedure. In frozen deployment it is E_task L(θ_0,ν). Source Jacobians from θ_0 cannot identify scratch-optimal frequency density, because changing ν changes the learned representation. A matched scratch comparison may use the density prior and then evaluate the resulting trained models; it cannot inherit a frozen-model compatibility theorem. Training cost, support, K and initialization must match the claimed attribution.

The constructive common framework is **task-role decision loss under the actual regime of parameter adaptation**. The finite replay estimator is useful for frozen deployment proposals; the Cosh analytic result is an assumption-dependent design prior. The framework unifies the objects being optimized while preserving the empirical distinction between from-scratch allocation and a mature-model transplant.

## 8. Immediate handoff

No extra large benchmark or arbitrary curve grid is needed to make the next choice. The missing artifact is a source-window labeled capture containing raw Q/all K/V, correct single-token answer label, explicit source roles and source answer-output sensitivities for family-paired record tasks. Existing natural full rows can be reused for the compatibility constraint. Fit one shared ordered table, record exact finite objective changes and constraints, and verify the generated table on actual full-model source and 128K outcomes. If the source sensitivity estimator fails its own held-out finite single-layer interventions, do not proceed as if it had identified a frequency mechanism.

## CPU arithmetic executed in this turn

A standard-library calculation checked the exact finite score bound on 1,000 random five-pair changes with signed distances up to 131,072 and arbitrary finite old/new frequencies. No violations occurred; maximum observed |score change| minus bound was −0.04345434. This is arithmetic verification of the bound, not empirical calibration. The rare-key comparison independently gives exact log mass 10.00285611, second-cumulant 5.08417605, and constant-score-1 competitor 5.15888308, reproducing the strict ranking reversal.
