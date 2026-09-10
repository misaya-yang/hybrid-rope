# Sol11 — failure-transcript audit and constructive allocation principle

## Verdict

The failure record rules out a task-independent allocation rule chosen from frequency geometry, smoothness, local NLL, or isolated slot gains. It does **not** rule out frequency allocation. The consistent object is the ordered dilation field

\[
\omega'_j=\omega_j s^{-m_j},\qquad 0\le m_j\le1,
\]

together with its slot identity, gain, checkpoint weights, and lifecycle. EVQ and MrRoPE can be unified only as different priors on this field. The final rule must be selected by **signed competitive transport in the full model**: preserve the learned Native computation while improving the target-length evidence-versus-distractor margin and complete generation.

The lifecycle distinction is essential:

- **Training from scratch:** weights can co-adapt to a new allocation. EVQ-Cosh is a defensible structured initialization or regularizer, and its strongest evidence is matched training, including the three-seed 454M fixed-scaler comparison and the 151.9M fixed-support intervention. It is not a frozen deployment rule.
- **Frozen deployment:** changing a frequency changes the meaning of a learned rotary slot. Start from a robust MrRoPE-style compatibility prior, then admit only a small, ordered perturbation whose direction is chosen by the checkpoint's signed full-model response under the actual target-length competition. Static geometry may constrain the move; it cannot select its sign.
- **Adaptation:** optimize weights and allocation jointly or keep the table fixed while adapting the readout, and assess the final product. Tail NLL or source use alone does not establish successful answer binding.

This is consistent with the frozen 50M crossing, where the runtime swap can increase effective rank while PPL changes from `7.14` to `76.20`, and with the mature co-adaptation result, where a learned table improves long-tail NLL but not full-sequence NLL or full-200 2Wiki capability (`.agents/critic_r1/handoff.md:17-25`; `paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md:94-122`).

## Non-redundant failure ledger

| Failure | Proposed | Implemented / tested | Actual result | Correction and constraint |
|---|---|---|---|---|
| Objective substitution | Improve the broad high/mid/low allocation that could unify EVQ and MrRoPE. | Several turns narrowed the problem to one middle-frequency role, smoothness, a local bridge, or a score. | The user corrected this twice: the middle band is part of the problem, not the whole problem. | A candidate must specify the complete ordered field and how high, middle, and low roles trade under one budget. |
| Smoothness as selector | Smooth(MrBudget) fixed MrPro's cumulative compression budget and minimized roughness. | Same checkpoint, gain, 36-row historical development panel, plus 48 NLL measurements. | 32K tied MrPro; 128K fell from `78.125%` to `68.333%`. NLL worsened by `+.00098/+.00192/+.00342` at 8K/16K/32K. Full archived output inspected at `corpus/tool_outputs_055.jsonl:103`. | Budget and roughness are insufficient. A smoother cumulative dilation field can damage evidence competition. |
| Group phase shift confused with allocation | LongBridge moved four long-period slots together. | Slower and Faster directions, with within-group gaps preserved. | Slower: `-6.67 pp` at 32K, `+1.94 pp` at 128K, driven mainly by VT; Faster did not supply a symmetric improvement. Full Slower output inspected at `corpus/tool_outputs_056.jsonl:30`. | A common group shift tests phase origin/direction. It does not test how resolution is distributed inside the group. |
| Conditional P2 win promoted too quickly | FullLagP2 repaired old lag sampling and was expected to transfer its 1.5B signal. | Qwen2.5-3B, same 36-row panel; compared against both official-gain and matched-gain MrPro. | 32K `72.92%` versus matched-gain MrPro `98.33%`; 128K `81.67%`, `+3.54 pp` versus official MrPro and `+6.32 pp` versus matched gain. At 128K QA/VT rose while multikey fell. Full output inspected at `corpus/tool_outputs_056.jsonl:91`. | P2 is a Pareto crossing on a small reused development panel, not a generally better rule. It is useful as a direction hypothesis only. |
| Arbitrary EVQ borrowing | Move high-frequency log-gap budget into a long-period band, by analogy with EVQ. | One fixed transfer candidate, then a same-budget middle-band recipient control. | Long-recipient candidate lost `17.08 pp` at 32K and `10.76 pp` at 128K, with `0/36` improvements. The middle-recipient run also lost (`70.14%/67.36%` versus `87.22%/78.13%`) and had `0` wins, `7` losses; its full output is `corpus/tool_outputs_056.jsonl:130`. | “Borrow from high, give to low” is not a rule until the donor and recipient are justified by the checkpoint's signed computation. |
| Gain confounded with allocation | Some short gains were initially credited to the table. | Four-cell gain comparison: MrPro and BM at original and matched gain. | Dialogue record: MrPro `87.22/78.13`, BM `91.67/70.83`, MrPro-g074 `98.33/75.35`, BM-g074 `100/70` at 32K/128K. Independent BM transfer owner confirms BM is `+4.44 pp` at 32K and `-7.29 pp` at 128K with matched gain (`docs/research/ROPE_BM_TRANSFER_RESULT_20260908.md:11-28`). | Treat gain as a separate operator coordinate. A table comparison without a gain cross cannot attribute the change to allocation. |
| Single-slot additivity | E1 isolated slight benefits from decompression near slot 28. | Individual slot interventions, followed by their combined table. | The combined intervention lost `4.17 pp` at 128K while tying 32K; the supposed answer-binding counterfactual was wrong in both worlds. | Local slot effects interact through softmax and subsequent layers. Compose only through a jointly evaluated candidate, never by summing per-slot scores. |
| Proxy-to-task bridge | Geometry, native-QK KL, NLL, source use, or teacher-forced fork margins would select task improvements. | Many exact CPU identities and real-model diagnostics were implemented; some lowered the named proxy. | Smooth(MrBudget), frozen rank/PPL crossing, Llama QA, and co-adaptive full/tail results give direct counterexamples. The project itself records that low long PPL can coexist with failed answer binding (`docs/exp/2026-07/2026-07-15_lora_qa16k_three_arm_results.md:81-111`). | Proxies may reject unsafe candidates or diagnose failures. The requested outcome must be tested by complete target task generation and EOS. |
| Wrong “unseen arc” theory | A GLM/horizon story treated slots 36–39 as entering a new phase regime only at long context. | Post-hoc circle-count and tiny-sample correlations. | Those slots already exceed a full turn inside the Native window; FWE evidence distance was inferred from repeated answer tokens, not a valid source locator. | Derive phase regimes from the actual native table and source/query positions. Do not infer evidence distance from answer-string occurrences. |
| Mechanical panels after target changed | Continue running 32K and 128K for every candidate. | Repeated dual-length panels and full 36-row runs. | Several candidates already had target-length or NLL negatives; the user explicitly redirected to 128K. | Reuse existing 32K baselines. Screen on target-length natural NLL/passkey, then run the smallest downstream subset that can decide the claim. |
| Frozen/scratch conflation | Use EVQ's from-scratch successes as evidence that a related frozen table should work. | Frozen swaps and short LoRA conversions. | Frozen allocation is strongly checkpoint-relative; the same BM construction wins OLMo 16K but loses Qwen 3B/7B 128K (`paper-2027/research/EXPERIMENT_ASSETS_TOP15_20260909.md:19`). | Scratch evidence supports substrate–scaler complementarity; frozen deployment requires compatibility calibration on that checkpoint. |

The ledger also exposes two operational failures. First, candidate generation repeatedly preceded a sufficiently precise mechanism: the superficial high-gap transfer entered GPU evaluation before its donor/recipient logic was established. Second, useful controls were sometimes delayed: matched gain, actual target length, and complete task generation should precede broad confirmation. These are experiment-order errors, not reasons to erase the resulting negative evidence.

## What survives

### 1. A conserved log-dilation budget is the right coordinate

Let edge increments be

\[
p_i=m_i-m_{i-1},\qquad p_i\ge0,\qquad \sum_i p_i=1.
\]

Then the cumulative budget satisfies

\[
B=\sum_{q=1}^{N-1}m_q=N-\sum_i i p_i.
\]

Thus total compression and the center of mass of added log gaps are coupled. This identity explains what Smooth(MrBudget) controlled, but the negative result proves that the budget's *placement* must be weighted by learned computation. MrRoPE's increasing radix is one prior over `p`; EVQ's exponent-density quantiles are another.

### 2. Ordered slot identity is part of the model

The successful OLMo log-s4 owner reports that permuting the interior final-frequency multiset raises 1x PG-19 NLL by `+3.760692`; the 16K endpoint was correctly left unopened (`paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md:181-195`). Any unification over an unordered frequency density is therefore incomplete for frozen weights.

### 3. Full-model competition, not an isolated phase feature, is the target

For a query and relevant/distractor sets, define

\[
G(m,g)=\operatorname{LSE}_{r\in R}\ell_r(m,g)
-\operatorname{LSE}_{d\in D}\ell_d(m,g)
=\operatorname{logit} A_R.
\]

This quantity includes softmax competition. Under literal replication of the distractor exponential sum by factor `q`, it drops by exactly `\log q`; the project records this as a conditional identity rather than an extrapolation law (`paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md:270-300`). A useful target-length rule must preserve or improve this signed margin on actual long prefixes and must still pass complete decoding.

## Concrete allocation principle: Compatible Competitive Transport (CCT)

CCT uses one common parameterization and two lifecycle-specific estimators.

### Common parameterization

Use `m` or its nonnegative increments `p` to construct

\[
\omega'_j=\omega_j s^{-m_j}.
\]

Keep the original slot ordering, fixed `K`, standard RoPE computation, and a separately declared scalar gain `g`. High/middle/low are consequences of learned sensitivity and target competition, not hard-coded semantic labels.

Define two structural priors in this coordinate:

- `p_E`: EVQ's endpoint-matched exponent-density allocation;
- `p_M`: MrRoPE-Pro's cumulative radix allocation.

They are priors, not candidates to average by an arbitrary coefficient. Their role is to regularize directions weakly identified by the lifecycle-specific risk.

### Scratch rule

For scratch training, choose an initialization `p_0` from EVQ's maximum-entropy/convex allocation prior and train weights jointly with that table. If allocation is learned, solve the bilevel problem

\[
\min_p\;R_{\rm deploy}(W_T(p),p)+\lambda D(p,p_E),
\]

where `W_T(p)` is the result of the fixed training algorithm. The hypergradient contains the weight-response term; omitting it reduces the problem to the same frozen proxy that the crossing results refute. The 454M result supports only this scoped conclusion: under a shared fixed-index scaler, EVQ-trained weights reached 8K teacher-forced passkey `100\pm0%` versus Geo-trained `61\pm3%`, and 16K PPL `107.5` versus `157.7` (`docs/exp/2026-03/2026-03-03_passkey_mix_results.md:77-98`).

### Frozen Qwen2.5-3B 32K→128K rule

Start from `m_M`, because it encodes the strongest existing high-frequency preservation and complete-tail scaling prior. On predeclared calibration rows, construct one low-dimensional ordered perturbation `h` in cumulative-dilation space. A defensible basis is the span of:

1. the endpoint-matched EVQ-minus-Mr direction;
2. the FullLagP2-minus-Mr direction, retained because it produced a real 128K Pareto signal;
3. the constant-gain coordinate, kept separate from frequency movement.

This is a mechanism-grounded basis, not an arbitrary table grid. Estimate, using the complete frozen model:

- `D_N(h,g)`: Native-window divergence, preferably next-token KL plus strict generation retention on independent short rows;
- `G_T(h,g)`: target-length relevant-versus-distractor log-sum-exp margin on real 128K prefixes, using source-located tasks where `R` and `D` are known;
- `R_T(h,g)`: target-length complete task risk.

Choose exactly one perturbation by

\[
\begin{aligned}
\max_{h,g}\quad &\widehat G_T(h,g)-\lambda_E\|h-h_E\|^2-\lambda_M\|h\|^2\\
\text{s.t.}\quad &\widehat D_N(h,g)\le\epsilon,\\
&0\le m_M+h\le1,\quad p(m_M+h)\ge0.
\end{aligned}
\]

The regularizers stabilize the fit and must be fixed before target task outcomes are opened. The output is one ordered table and one gain. It is then accepted or rejected by the actual 128K generation endpoint; the optimization objective is not a success theorem.

If labeled relevant sets are unavailable, build paired long prefixes by inserting the same native text and answer continuation into independently sampled distractors, and distill the Native-window teacher distribution on the answer tokens. This preserves a known computation while exposing the full long-context softmax denominator. A Native-QK-only KL cache is insufficient because the prepared finite-KL preflight itself notes that it omits candidate hidden-state feedback, values, and downstream layers (`paper-2027/research/attention-aware-retrofit/preflights/coupling-transfer/NATIVE_QK_FINITE_KL_PREFLIGHT_20260901.md:75-90`).

### Adaptation rule

When weight updates are allowed, keep the selected table fixed and optimize QKVO/readout on physical target-scale data with Native replay. Use answer-plus-EOS and same-target source counterfactual loss, as already implemented in `scripts/train/train_log_p2_phase_transfer_lora.py:125-174`. The adapter is part of the method product; compare it to the frozen table and a matched Native-table adapter. Do not infer an adaptation success from training loss.

## Precise assumptions and counterexample checks

CCT depends on these assumptions:

1. The calibration distribution contains the relevant target competition: evidence distances, distractor counts, and prompt formation must resemble the declared 128K claim.
2. The selected perturbation lies inside a region where Native divergence is meaningfully controlled. A local Fisher or QK KL alone is not a finite-radius bound.
3. Source-located relevant sets are valid for the diagnostic. Repeated answer strings are not source locators.
4. The final task endpoint is independent of table/gain selection.
5. The gain is fixed or jointly crossed; it is never silently inherited.

Before promotion, require:

- **sign mirror:** compare `h` with an equal-norm `-h`; failure to predict direction rejects the signed response model;
- **combination check:** evaluate the jointly selected table, not the sum of isolated slot improvements;
- **distractor replication:** hold evidence fixed and increase distractor blocks; the measured `G` degradation should track the actual exponential-sum ratio, not nominal length alone;
- **slot permutation:** preserve the frequency multiset and scramble slot assignment; a collapse confirms checkpoint coupling and prevents density-only claims;
- **gain cross:** table × gain four-cell comparison;
- **lifecycle cross:** for scratch, compare self-consistent weights/table cells; for frozen, keep weights fixed;
- **complete endpoint:** strict target task output plus EOS and Native retention, with NLL and attention diagnostics secondary.

## Minimal next test

For the target Qwen2.5-3B deployment, reuse all current MrPro, matched-gain, Smooth, LongBridge, P2, and high-gap rows. Do not run another broad candidate matrix.

1. Freeze a compact calibration set containing paired 128K source-located multikey/variable-tracking examples with added independent distractor blocks and a small Native 32K retention set.
2. Compute full-model signed finite differences only along the two grounded directions `h_E` and `h_P2`, plus gain. Fit the constrained three-coordinate CCT program.
3. Materialize one table/gain pair.
4. Run target-length natural tail NLL and the smallest independent multikey/VT generation panel. Open the wider task set only if the candidate improves the declared task cluster without breaching Native retention.

Falsification is direct: if the sign mirror is not ordered correctly on calibration, stop the local model; if the selected candidate does not improve the independent 128K cluster or violates Native retention, stop this CCT basis. Do not respond by adding slots, curves, or a new proxy. A failure closes this concrete low-dimensional rule, not all frequency allocation.

## Evidence boundary

Every assigned dialogue record and all 116 additional assigned project files were read in full. Four relevant unique archived tool-output records were also read in full. The exact long-recipient `0/36` output was not separately recoverable by name from the four inspected records; its result is retained as a dialogue-level report, while the separately inspected middle-recipient output is explicitly identified above. No GPU job was launched and no extra agent was spawned.
