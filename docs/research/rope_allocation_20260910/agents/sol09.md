# Sol09 — failure-transcript audit and allocation principle

## Bottom-line finding

The failure record rules out a universal frequency-only ranking rule for frozen checkpoints. The strongest surviving principle is **marginal-utility allocation under a mixed-radix budget**: represent the extension as nonnegative log-radix increments whose sum is fixed, preserve rotary-pair identities, and allocate each next unit of log-scale only where a finite, signed, end-to-end margin intervention shows benefit under the current regime. EVQ supplies a useful smooth allocation prior; MrRoPE supplies a useful feasible budget and phase-safety scaffold. Neither supplies the missing utility by itself.

This gives one mathematical framework but two estimators:

- **Training from scratch:** utility is endogenous because weights co-adapt to the table. Jointly optimize weights and allocation on training data, and select the allocation on held-out target-window loss/capability. Geometry may regularize the allocation but cannot certify task performance.
- **Frozen deployment:** weights and slot semantics are fixed. Estimate utility at the exact checkpoint with finite-path, task-signed interventions, subject to Native-retention and ordering constraints. Do not transplant a curve, reorder a frequency multiset, or use a geometry proxy as if the weights were exchangeable.

## Formal allocation object

Let native rotary-pair frequencies be \(\omega_0>\cdots>\omega_{K-1}>0\), target extension factor \(s\), and nonnegative radix increments \(b_i\):

\[
b_i\ge0,\qquad \sum_{i=1}^{K-1}b_i=\log s,
\qquad m_j=\frac{1}{\log s}\sum_{i\le j}b_i,
\qquad \nu_j=\omega_j s^{-m_j}.
\]

The feasible set \(\mathcal M_s\) additionally requires \(0\le m_j\le1\), ordered positive deployed frequencies, fixed slot labels, and declared endpoint/support conventions. MrRoPE-Pro is one hand-designed point in this simplex, with increasing \(b_i\) in its transition band. EVQ is another way to induce a nonuniform allocation, through a continuous density/quantile construction. This parameterization exposes their common object: **where a fixed total log-scale budget is spent**.

For regime \(r\in\{\text{scratch},\text{frozen}\}\), define task loss through signed answer margins, with short-context protection:

\[
\mathcal J_r(m,\theta)=
\mathbb E_{(x,y)\sim D_{\rm target}}
\operatorname{softplus}\!\left(\gamma-[\ell_y(x;m,\theta)-\max_{c\ne y}\ell_c(x;m,\theta)]\right)
+\lambda\,[\mathcal L_{\rm short}(m,\theta)-\mathcal L_{\rm short}(0,\theta)-\epsilon]_+^2
+\rho\,\Omega(m).
\]

Here \(\Omega\) is only a regularizer: for example distance to an EVQ density, curvature of the radix increments, or distance from MrPro. It is not the capability objective. A practical finite-path marginal utility for increment \(i\) is

\[
U_i=-\int_0^1 \frac{\partial}{\partial b_i}
\mathcal J_r\bigl(m(b^{(0)}+t\,\delta b),\theta_t\bigr)\,dt.
\]

For frozen deployment, \(\theta_t=\theta_0\). For scratch training, \(\theta_t\) follows the jointly trained or inner-optimized weights. At an interior optimum, the KKT condition equalizes marginal utility per unit budget across active increments; increments at bounds have the corresponding one-sided inequalities. This is the constructive unification: **EVQ/MrRoPE propose priors and constraints; finite task utility decides the allocation.**

### Concrete frozen rule for Qwen2.5-3B, 32K to 128K

1. Start at the exact MrPro table and its declared gain, keeping slot identities.
2. Use the existing completed paired rows; select correct-answer versus actual competitor margins, not unsigned geometry or gold-only NLL.
3. For each admissible increment transfer \(\delta b=\eta(e_p-e_q)\) that preserves \(\mathcal M_s\), evaluate the full model at a small finite set including the endpoint. Score the paired target-margin change minus the registered short-context penalty. This is a finite intervention, not a long-horizon Taylor extrapolation.
4. Move budget only when the same signed direction is supported across the relevant task strata and the Native-retention constraint holds. Otherwise keep the MrPro increment.
5. Stop when no admissible transfer has positive lower-confidence utility. Freeze the resulting table and evaluate once on independent tasks/rows. Gain remains fixed during allocation attribution; a separate 2-by-2 table-by-gain comparison is required if gain changes.

This rule is conservative by design. The transcript shows one-slot or local successes can be real but conditional; it does not support a broad coordinate grid. The minimal next informative candidate should therefore be the single best supported budget transfer from current evidence (the existing slot-28/E1 clue may nominate a transfer, but must be rechecked on the exact Qwen-3B margin panel before use), against MrPro at identical gain.

## Non-redundant failure ledger

| Failure | What was proposed / implemented / tested | Correction and surviving constraint |
|---|---|---|
| Proxy-first theory | Smoothness, effective rank, collision, transport, movement MAE, and phase-safety quantities were repeatedly treated as candidate selectors. Some were only derived or CPU-reconstructed; some were tested as frozen-table diagnostics. | C2 reconstructed movement with MAE 0.001223 but failed the Native operating point; an identical frequency multiset under slot permutation changed OLMo NLL 3.10423→6.86493 and Qwen core-4 0.70→0. Geometry is a regularizer/diagnostic, never the utility. [CPU coupling owner, lines 38 and 61](../../../paper-2027/research/attention-aware-retrofit/results/coupling-transfer/CPU_LOW_DIM_COUPLING_LAW_20260901.md#L38). |
| Cosh specificity overclaim | The Cosh family was correctly derived as the minimizer of a chosen convex surrogate. Reviews then sometimes narrated this as the performance mechanism or a unique allocation law. | Fixed-endpoint training identifies an interior-allocation effect; it does not identify Cosh specificity. The surrogate is specified, not derived from full-subspace geometry or LM loss, and its strength remains externally chosen. |
| Frozen/scratch conflation | Frozen table swaps and overlays were used to reason about a table that might train well from scratch, while scratch outcomes were used to suggest frozen compatibility. | The same trained pair changes ordering across deployment policies: fixed-s4 overlays favor Cosh, target-s8 overlays reverse it ([ROI memo, lines 70–75](../../../docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md#L70)). Scratch and frozen share the budget variable but require different utility estimators. |
| Slot exchangeability | Curves/multisets were treated as if only their sorted frequencies mattered. | Same-multiset permutation collapse proves learned Q/K content is bound to rotary slot labels. All frozen rules must preserve slot identity and evaluate label-specific interventions. |
| Scale stationarity | A 64K local success was promoted toward a scale law. FullLagP2 improved Qwen-1.5B 64K on the small panel, then hit an MK2 floor at 128K and transferred with mixed task signs to Qwen-3B. | Utility is conditioned on target scale and task distribution. A 2x result cannot determine 4x/8x allocation. |
| Numerical alias masquerading as mechanism | The old p2 used 2,048 rounded lag centroids and produced isolated movement spikes and order crossings. The corrected full-lag version removed those artifacts. | Numerical construction identity is part of the method. Do not infer mechanism from an aliased table; preserve exact arrays/hashes and distinguish arithmetic correction from downstream validation. |
| Local Taylor extrapolation | Long-horizon effects were summarized with a local quadratic/Fisher approximation even when phase shifts exceeded multiple radians. | Use exact finite-path replay/integrated gradients or direct endpoints. The archived audit reports extreme Taylor errors for slot 19 at long horizons; the bounded sinusoidal operator makes unconstrained quadratic growth invalid. |
| Unsigned score proxy | Raw score MSE looked tiny for one operator fit. The full tool record shows layer-0 relative score MSE \(1.305\times10^{-5}\), but row-centered relative MSE 0.48446; attention KL was separately large. | Softmax ignores row constants and answer success depends on signed competition. Optimize correct-vs-competitor margins or attention/output effects, not raw score magnitude. |
| NLL/probability substituted for generation | Multiple adaptation/compression variants improved teacher-forced NLL or local reconstruction but did not recover strict autoregressive answers. FAR-pass variants and the operator-family study are direct examples. | NLL is a valid likelihood endpoint only. Retrieval/generation claims require full answer, EOS, and task scorer. The zero-training budgeted owner explicitly keeps frequency-only, gain-only, and joint results separate ([lines 52–71](../../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md#L52)). |
| Operator-family premature closure | The first comparison tested one compressed structure with score fitting and output KD, then initially suggested the GPU could be released. Later analysis found attention and value damage, calibration mismatch, and untested alternatives. | A failed objective is not a failed method class. The exact native answer was deterministic and correct; the compressed models lost target attention/value pathways. On layer20 output-KD, answer mass fell 0.50075→0.20339 and attention/value-only output NMSE were 0.18309/0.23950; on layer26 answer mass fell 0.04830→0.0000485. |
| Calibration-distribution mismatch | Operator compression used 2K natural-text states and replayed position phases, but the target was real 8K retrieval among hundreds of random records. | Rephasing fixed states does not create true long-context hidden states or distractor competition. Calibration must match the claim’s content, length, and competitor structure. |
| Assay floors and noisy pilots | Small panels, weak Native cells, and partial-credit scorers were sometimes treated as model or method ceilings. The historical QuALITY report originally blamed capacity, but its own later correction says the design did not identify capacity versus alignment/input distribution; n=200 apparent gains shrank at n=2,086 ([lines 1–5 and 68–76](../../../docs/exp/2026-03/2026-03-12_phase21b_454m_full_eval_report.md#L1)). | A floor means the assay cannot rank candidates. Use tasks the unchanged checkpoint solves and preserve uncertainty at the correct unit. |
| Bundled table, gain, routing | Table changes were compared with different gains or routing and then discussed as allocation effects. | Hold gain fixed for the allocation contrast. The budgeted OLMo result shows frequency-only 0.400/0.115 versus joint 0.5825/0.400 at 8K/16K, so gain interaction is material, not bookkeeping ([lines 52–76](../../../paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md#L52)). |
| Invented operational hard gate | E2 was blocked by a 30.66 GiB single-arm storage estimate and a >90 GiB three-arm narrative. The implementation had chosen three weight milestones plus optimizer/RNG resume state. | This was a retention-policy choice, not a scientific requirement. The user corrected it to final-only, and then further corrected the scientific scope: train only the project’s method, keep YaRN/MrPro frozen. The durable correction is recorded at [ROI memo lines 106–112](../../../docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md#L106). |
| Scope inflation after correction | Even after fixing storage, the plan still proposed full-adapting YaRN/MrPro/Z, although the actual objective was to train the project method and reuse frozen opponents. | Separate proposed, implemented, launched, and tested. Old three-arm E2 remained NOT_STARTED and was superseded by Z-only; no training result exists for it. |
| GPU utilization as objective | Historical video work chose a 300M model for MNIST-like data, repeatedly OOM-tuned batches, and confused memory fill with throughput. | Use the least costly experiment that distinguishes the claim. Benchmark first and optimize samples/sec; the failure record is explicit at [AI handoff lines 5–21](../../../scripts/video_temporal/AI_HANDOFF.md#L5). |

## Counterexample checks for the proposed rule

1. **Permutation:** the rule is slot-labeled, so equal multisets with different assignments can receive different utility. It does not predict invariance contradicted by the collapse.
2. **Smooth_MrBudget-style proxy improvement:** if geometry improves but signed margins worsen, \(\mathcal J\) rejects it. This directly handles the known geometry/task misranking.
3. **P2 conditional gain:** the rule can retain a positive 64K task-specific marginal utility without calling it universal; 128K and Qwen-3B utilities are separately estimated.
4. **Training from scratch:** if a table initially hurts frozen weights but trains better after co-adaptation, the scratch inner optimization can accept it while the frozen estimator rejects it. This is intended, not a contradiction.
5. **Gain interaction:** fixing gain during allocation prevents table utility from absorbing temperature effects; a later factorial can estimate the interaction.
6. **Task conflict:** if MK improves while VT/FWE worsen, no scalar universal victory is manufactured. The loss reports the task-stratified Pareto trade-off or uses a declared task weighting chosen before confirmation.

## What is mathematically established versus empirical

- Exact: the radix-budget simplex, cumulative mapping to \(m_j\), KKT marginal-utility condition for the stated constrained optimization, and the need to retain slot labels in the frozen objective.
- Evidence-supported: allocation affects behavior at fixed endpoints in paired training; frozen outcomes depend on checkpoint/table pairing, target support, scale, gain, and task; low-dimensional or geometric proxies can mis-rank tested behavior.
- Proposed: signed finite-path marginal utility is a better selection rule than EVQ geometry or MrPro schedule alone. It is not yet validated as a universal predictor and must be assessed by the one minimal matched transfer described above.

## Coverage and tool-output receipt

The assigned dialogue file was loaded in full: 136/136 JSONL records, preserving each complete message and source mapping. All 114 additional assignment-listed files were loaded in full as a single ordered boundary-marked stream: 1,043,561 characters including boundary markers, with no omissions. One attempted 30,000-character page was truncated by the display ceiling and was reloaded in two complete 15,000-character pages; it is not counted as read from the truncated attempt. The detailed machine-readable receipt is `sol09_coverage.json`.

Full relevant archived tool-output records inspected, rather than merely indexed:

- `tool_outputs_003.jsonl` records 33–38: original E2 plan, 1,528-step budget, disk calculation, and the training/checkpoint-retention implementation. Record 39 was displayed truncated and is not relied on beyond the complete code already loaded from assigned files.
- `tool_outputs_019.jsonl` records 8–10: complete target-span attention/value decomposition, phase-restoration diagnostics, and progressive-KD generation/NLL result.

No unviewed tool output is represented here as read.
