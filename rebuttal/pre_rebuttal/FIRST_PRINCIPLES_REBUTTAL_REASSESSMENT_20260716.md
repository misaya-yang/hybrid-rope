# EVQ-Cosh first-principles rebuttal reassessment

Date: 2026-07-16

Status: `author-internal / triage-only / not an OpenReview response`

Decision: `high-risk trust repair; core claims must be narrowed`

This note consolidates four independent audits: rebuttal strategy, frequency-allocation optimality, the \(\tau\) rule, and the mechanism/experiment roadmap. It is subordinate to the raw-artifact provenance manifest for numerical facts and to `rebuttal_playbook.md` for operational decisions. It does not change the submitted paper, any reported number, training code, or experiment configuration.

## 0. Executive decision

The rebuttal cannot credibly be organized around “the submitted paper is mostly correct.” The defensible strategy is:

1. make one concise, consolidated integrity disclosure for material comparator/theory errors;
2. withdraw interpretations that depend on faithful YaRN, faithful DAPE, native-RoPE dominance, ordinary-KL optimality, or `c_coll` calibration;
3. defend only the finite-spectral-budget design view, the exact cosh minimizer of the stated convex surrogate, the zero-learned-parameter construction, and protocol-bounded empirical signals;
4. answer LoRA/downstream and supporting-family questions only when a real reviewer triggers them, always including adverse results;
5. run an experiment only when it discriminates a score-changing reviewer hypothesis. No experiment can repair a method-identity or mathematical error.

The scientific outcome is narrower than the submitted spine: **EVQ-Cosh is a structured allocation family with an exact surrogate theorem and limited mechanism evidence, not a universal RoPE optimum, a faithful YaRN/DAPE comparison, or a demonstrated downstream-capability improvement.**

Evidence labels used below:

- **Strict**: follows from the stated mathematical conditions or a directly audited fact.
- **Conditional**: follows only after listed modeling assumptions.
- **Empirical**: observed in the cited protocol; no broader implication is claimed.
- **Unknown**: the repository does not currently close the question.

## 1. Task 1 — rebuttal strategy after the integrity audit

### 1.1 What a PDF-only reviewer can detect

| Issue | PDF-only visibility | Why | Required disposition |
| --- | --- | --- | --- |
| Ordinary-KL order | **Direct** | The main theory calls the \(O(\tau^2)\) term a KL utility, while the appendix's own local KL expansion is quadratic in the logit perturbation; see `paper/sections/03_theory.tex:90-108` and `paper/appendix/a1_proofs.tex:453-501`. | Proactively correct; ordinary KL is \(O(\tau^4)\). |
| No demonstrated 8B task lift | **Direct, but only the submitted boundary** | The PDF already reports single-seed LoRA, an 8K cost, and no RULER gain in `paper/appendix/a4_supporting_experiments.tex:32-51`. The stronger 2026-07-15 QA failure is post-submission and not PDF-visible. | Do not use LoRA as a defense; answer fully if downstream utility is raised. |
| Midpoint rather than native Geo | **Expert-inferable** | The deployed formula explicitly uses \(u_k=(k+1/2)/K\), while native RoPE uses an endpoint grid; see `paper/sections/03_theory.tex:52-81`. | Proactively relabel; withdraw standard/native-RoPE dominance. |
| `c_coll=1.171` | **Numerically reproducible** | The claimed objective and calibration are stated in the PDF. A reviewer can independently optimize them even without code. | Proactively withdraw minimizer/CV/closure claims. |
| `YaRN` is a fixed-ramp scaler | **Not provable from PDF** | The PDF omits the operator equations needed to identify the 20%--90% fixed-index ramp. The implementation is explicit in `scripts/lib/rope/schedules.py:168-179`; the full audit records the lineage. | Proactively relabel because it changes Primary I's comparator identity. |
| `DAPE (32p)` is shared learnable `inv_freq` | **Suspicious but not provable from PDF** | A DAPE expert may question the 32-parameter row, but the PDF omits the forward path. The historical implementation and full audit establish the actual shared-frequency control. | Proactively relabel because it changes Primary II's comparator identity. |
| Phase16 reporting/protocol errors | **Mostly code/log dependent** | The PDF states 27 configurations and near-optimality; the manifest shows 99 runs over 9 configurations with selected confirmation. | Withdraw the 27/all-\(<1\%\)/near-optimal interpretation. |
| Primary II/III protocol facts | **Code/log dependent** | Model size, learning rates, batches, `head_dim`, and the ad-hoc `d_eff` convention are not recoverable from the PDF alone. | Correct if these rows remain part of the core defense. |

Low discoverability is not a reason for silence. YaRN/DAPE identity and the KL/`c_coll` chain determine the meaning of the central claims. If later code or AC inspection shows that the authors knew and omitted them, the trust cost is larger than an immediate, bounded correction.

### 1.2 One consolidated integrity disclosure

This is a research-integrity recommendation, not a special process explicitly prescribed by the NeurIPS handbook. It should be one concise AC-visible disclosure, not copied into every review response and not diluted with new results.

Minimum content:

1. **Comparator identity**
   - `YaRN` \(\rightarrow\) repository-defined fixed-index smooth-ramp scaler without the official correction range or mscale;
   - `DAPE (32p)` \(\rightarrow\) layer-shared learnable inverse-frequency vector;
   - `Geo` \(\rightarrow\) midpoint-discretized geometric grid, not native endpoint RoPE.
2. **Theory/reporting correction**
   - ordinary baseline-to-perturbed KL starts at \(O(\tau^4)\), not \(O(\tau^2)\);
   - `c_coll=1.171` was not obtained by optimizing the stated collision objective;
   - Phase16 was a 9-configuration pilot/selected-confirmation study, not 27 comparable configurations with all outcomes within 1%.
3. **Protocol facts if Primary II/III remain core**
   - Primary II is about 151.9M parameters with the audited optimizer/batch settings and a seed-42 headline;
   - Primary III has actual `head_dim=64`, `d_rope=32`, heterogeneous batch sizes, and `tau=1.414` as an empirical setting rather than a validated `d_eff` theorem.
4. **Disposition words, not euphemisms**
   - identify which raw-backed numbers are unchanged;
   - identify what is relabeled;
   - identify what interpretation is withdrawn;
   - identify what remains unsupported.

Do not call these “cosmetic naming” or “implementation details.” The official 2026 rules allow textual responses but no revised paper or supplement; the original submission remains the basis of the decision. A future corrected manuscript therefore cannot be written as if it were already the reviewed artifact.

### 1.3 Reviewer-triggered, not proactive

| Real trigger | Response boundary |
| --- | --- |
| 8B, downstream, or practical utility | State that there is no demonstrated capability gain. If the post-submission result is used, report the full registered negative QA result, the \(\le8\)K deficit, the \(>8\)K floor, and the PPL/task mismatch together. Do not call the observed protocol-specific 8K NLL degradation universal “catastrophic forgetting.” |
| Faithful YaRN | Correct the submitted identity first. Only then, if useful, report the single-seed component follow-up: official frequency correction erases most of the abundant-MHA gap, while a scarce-channel aggressive-scale gap survives. It cannot restore the submitted complementarity claim. |
| Faithful DAPE | Correct the identity first. The repository's DAPE-ish Kerple+MLP result does not show EVQ beating DAPE; do not hide this if asked. More seeds cannot repair an identity error. |
| PK or retrieval ability | Define PK as teacher-forced NLL-gap. Keep autoregressive exact match, answer rank, and task F1 separate. |
| Supporting video/750M/QuALITY/old LoRA | Give the exact provenance and adverse boundary only if the reviewer cites that family. Never use it as a replacement for a failed core control. |

### 1.4 The narrow survivor claim

> RoPE exposes a finite set of rotary-frequency channels, making training-time allocation a legitimate design variable. EVQ-Cosh is a closed-form, zero-learned-parameter allocation family whose continuous cosh density uniquely minimizes one explicitly stated convex surrogate. Within the audited protocols, changing the schedule produces measurable PPL and teacher-forced positional effects, particularly under finite-channel stress. We do not establish a task-independent optimum, faithful YaRN/DAPE superiority, native-RoPE dominance, or downstream capability gains.

This changes the paper's identity from “optimal allocation plus broad comparator complementarity” to “allocation-family proposal plus bounded mechanism evidence.” That may lower the score ceiling. If a reviewer's acceptance criterion depends on a withdrawn comparator or task-capability claim, the honest conclusion is that rebuttal cannot close it.

## 2. Task 2 — is geometric allocation already optimal?

### 2.1 There is no task-independent RoPE optimum

For rotary pair \(k\) and relative displacement \(\Delta\), the exact logit contribution has the form

\[
\ell_k(\Delta)=a_k\cos(\omega_k\Delta)+b_k\sin(\omega_k\Delta),
\]

where \(a_k,b_k\) depend on content, layer, head, and trained Q/K weights. A frequency table is therefore a finite Fourier dictionary. Its optimum depends on the task kernel, distance distribution, trained amplitudes, channel budget, spectral range, training regime, and inference scaler. Without these objects, neither “geometric is optimal” nor “cosh is optimal” is a complete proposition.

### 2.2 Conditions under which geometric is strictly optimal

Let \(\phi=-\log_b\omega\in[0,1]\). Geometric RoPE is uniform in \(\phi\), not in \(\omega\).

**Strict A — uniform log-frequency quantization.** Under a uniform \(\phi\) prior, midpoint geometric is the unique minimizer of

\[
\int_0^1\min_k|\phi-\phi_k|^2\,d\phi.
\]

The cell distortion is \(\sum_k \ell_k^3/12\); strict convexity and \(\sum_k\ell_k=1\) force equal cells. The same grid uniquely minimizes worst-case log-frequency coverage radius.

**Strict B — maximum entropy / no-information prior.** Uniform density uniquely maximizes \(-\int\rho\log\rho\) and uniquely minimizes \(D_{\mathrm{KL}}(\rho\|1)\).

**Strict C — symmetric positive quadratic objectives.** For

\[
J[\rho]=\tfrac12\langle\rho,T\rho\rangle,\qquad \int\rho=1,
\]

uniform is the unique minimizer if \(T1\) is constant and \(T\) is strictly positive on the zero-mean subspace. A positive, translation-invariant kernel on a periodic log-frequency domain is a representative case.

**Conditional D — preservation of a heavily pretrained model.** If adaptation risk is locally

\[
J_{\mathrm{task}}(\phi)+\lambda\|\phi-\phi_{\mathrm{Geo}}\|^2
\]

with a small task gradient at Geo and positive total Hessian, the pretrained Geo grid is a strict local optimum. The 15T+ LLaMA pretraining history makes this plausible, but the repository has not measured the required gradient/Hessian.

### 2.3 What the cosh theorem does and does not prove

For the submitted functional

\[
\mathcal C_{\mathrm{app}}[\rho]
=\frac\alpha2\int\rho^2
+\frac\beta2\iint\rho(\phi)\rho(\psi)\min(\phi,\psi),
\]

with \(\alpha>0,\beta\ge0\), the existence, strict convexity, positivity, normalization, and unique cosh solution are valid; see `paper/appendix/a1_proofs.tex:15-50`. For \(\beta>0\), uniform is not even stationary because \(T1\) is nonconstant. Thus the calculus is not the failure point.

The gap is the objective identity: the theorem does not show that this surrogate is the exact RoPE sin/cos kernel, trained attention, LM loss, or task risk. In fact, direct optimization of the stated collision diagnostic drives \(\tau\) far above the deployed range. In a representative internal scalar scan at \(K=32,b=500\mathrm K,L=512\), the best point found is near \(\tau\approx13\), versus deployed \(2.83\); it is not claimed as a certified global minimum. The much better feasible point is already a counterexample to using the reported `c_coll` as a task operating-point selector, not a recommendation to deploy \(\tau=13\).

### 2.4 A previously undercounted finite-grid confound

The practical map is

\[
\phi_\tau(u)=1-\tau^{-1}\operatorname{asinh}((1-u)\sinh\tau).
\]

For every \(u\in(0,1)\) and \(\tau>0\), convexity of `sinh` gives \(\phi_\tau(u)<u\). Therefore all sampled frequencies move upward. With midpoint quantization, the finite table also changes its realized extrema and span.

For \(K=32,b=500\mathrm K,\tau=4\), recomputation from `scripts/lib/rope/schedules.py:94-140` gives:

- highest sampled frequency: about \(1.17\times\) midpoint Geo;
- lowest sampled frequency: about \(3.17\times\) midpoint Geo;
- realized natural-log span: about \(12.71\rightarrow11.71\).

The submitted comparison is therefore a controlled **schedule intervention under a shared quantile convention**, but not a fixed-extrema/fixed-span pure-shape experiment. It is not merely “changing base,” yet the current evidence does not isolate density shape from range/span and active-channel effects.

### 2.5 When nonuniform allocation can be better

Nonuniform allocation is conditionally favored when:

- the task's log-frequency sensitivity is nonuniform; high-rate quantization gives \(\rho^*(\phi)\propto w(\phi)^{1/3}\), and geometric is recovered only for constant \(w\);
- finite training length makes many low-frequency features nearly collinear;
- rotary channels are scarce and extrapolation is aggressive;
- a range scaler does not already substitute for the reallocation;
- training supplies gradients for the dependencies that the reallocated channels should represent.

The present evidence supports only a regime-specific version: official YaRN frequency correction largely erases the abundant-MHA substrate gap, while a single-seed scarce-MLA gap survives at aggressive scales (`EVQ_YARN_COMPONENT_ABLATION_20260714.md:53-102`). It does not support “nonuniform is generally better,” and the 8B negative task result does not prove the converse “geometric is universally best.”

## 3. Task 3 — first-principles status of \(\tau=d_{\mathrm{eff}}/\sqrt L\)

### 3.1 The task-local optimum

Set \(\theta=\tau^2\). Around Geo,

\[
\phi_\tau(u)=u-\theta\frac{u(1-u)(2-u)}6+O(\theta^2),
\]

so RoPE logits have \(z(\theta)=z_0+\theta g+O(\theta^2)\). For a fixed trained model and a specified task risk,

\[
R(\theta)=R_0+A_{\mathrm{task}}\theta
+\tfrac12B_{\mathrm{task}}\theta^2+o(\theta^2).
\]

If \(B_{\mathrm{task}}>0\), the constrained local optimum is

\[
\boxed{\tau_*^2=[-A_{\mathrm{task}}/B_{\mathrm{task}}]_+.}
\]

If \(A_{\mathrm{task}}\ge0\), Geo is locally optimal; if \(A_{\mathrm{task}}<0\), a nonzero local warp can help. When weights are retrained, the effective second derivative also contains the weight-adaptation Schur complement

\[
R_{\theta\theta}-R_{\theta W}H_W^{-1}R_{W\theta}.
\]

This is why data, initialization, training budget, LoRA parameterization, and model scale can change the best \(\tau\). The repository has not estimated \(A_{\mathrm{task}}/B_{\mathrm{task}}\).

### 3.2 What the conditional proxy actually yields

Under diffuse attention, channel additivity, small \(\tau\), a linear phase-displacement utility, and the chosen Pearson/load normalization, one may define

\[
F(\theta)=\frac{\theta^2}{90d_S}
-\lambda\frac{M}{L}Q_1(L,b)\theta,
\]

which gives

\[
\tau_*^2=\frac{45\lambda Q_1(L,b)M d_S}{L}.
\]

The published-looking \(\tau\propto d/\sqrt L\) requires the additional identifications \(M\propto d\) and \(d_S\propto d\). With consistent per-channel averaging, the dimension factor can disappear. For compressed RoPE, \(M=d_{\mathrm{rot}}/2\) is not automatically `d_head`; `d_eff=d_head` is therefore an architecture convention, not a theorem.

For sparse or local attention with effective support \(m\ll L\), the diffuse factor should conditionally depend on \(m\), or more generally on a directional softmax-Jacobian quantity, rather than nominal context length. If \(m\) saturates, the proxy predicts that \(\tau\) may also saturate instead of continuing as \(L^{-1/2}\).

### 3.3 Ordinary KL and finite-\(\tau\) limitations

For \(p_\theta=\operatorname{softmax}(z_0+\theta g+\cdots)\),

\[
D_{\mathrm{KL}}(p_0\|p_\theta)
=\tfrac12\theta^2g^\top(\operatorname{diag}p_0-p_0p_0^\top)g
+O(\theta^3)=O(\tau^4).
\]

The \(O(\tau^2)\) quantity can only be a separately defined linear transport/capacity proxy. It cannot be called ordinary KL and cannot be balanced against an \(O(\tau^4)\) stiffness to prove a nonzero KL optimum.

The leading small-\(\tau\) stiffness is also quantitatively weak at deployed values: the audited relative error is about 27% at \(\tau=1\), 97% at \(\tau=2\), and 252% at \(\tau=4\). It cannot quantitatively justify the short-context \(\tau\approx4\)--5 regime.

### 3.4 What the observed sweep points support

| Observation | Audited interpretation |
| --- | --- |
| \(L=128\), best tested \(\tau=5\) | Right-censored at the sweep boundary. It supports “the best tested point was 5,” not a located optimum above 5. |
| \(L=1024\), \(\tau\approx2\) | Coarse grid; supports a useful neighborhood, not a precise optimum. |
| \(L=2048\), \(\tau\approx1.5\) | The most informative of the three, but the denser sweep is single-seed and the larger-model check compares only Geo with the selected value. |
| Phase16 | At most a fallible basin prior: the common three-seed comparison is 7/9 wins and 2/9 losses, with selected-confirmation bias. |

The points use different models, data, token budgets, and extrapolation ratios. They are compatible with “useful \(\tau\) tends to decrease as training length grows,” but they do not validate exponent \(-1/2\), linear dimension scaling, or unit prefactor.

Reviewer-safe status:

> `tau=d_eff/sqrt(L)` is a parameter-free operating rule motivated by a conditional diffuse-attention phase-capacity proxy and observed to land in useful basins in several audited settings. It is not the universal optimizer of collision, attention, language-model loss, or downstream task risk.

## 4. Task 4 — extrapolation mechanism and experiment route

### 4.1 Three necessary gates

**Gate 1 — phase representation.** The frequency table selects a finite sin/cos dictionary and changes resolution, aliasing, and conditioning. This is where the exact surrogate theorem and model-free kernel diagnostics live. Passing this gate says only that distances can be represented more distinctly under a stated prior.

**Gate 2 — routing and competition.** Trained Q/K amplitudes must turn those features into a sufficient source-versus-distractor logit margin. With more comparable distractors, the required margin grows approximately as \(\log N\) in the equal-distractor model. Sparse attention changes the candidate support and softmax normalization, but it can substitute for or erase an allocation effect; it is not automatically complementary.

**Gate 3 — value/readout/generation.** RoPE directly rotates Q/K, not V. The selected source still has to be transmitted by V/O, preserved through residual/MLP blocks, bound to the requested answer, and promoted by the LM head. Better PPL, gold-token NLL, source-block rank, or attention mass does not guarantee top-1 generation.

The current 8B evidence locates the failure:

- runtime/adapter cross-swaps show that training-time co-adaptation dominates the runtime frequency tensor (`docs/exp/2026-07/2026-07-14_lora_retrieval_conversion_probe.md:48-64`);
- EVQ improves 16K target-block hit@16 (64.06% versus 18.75%) and removing the gold block on all heads worsens EVQ NLL by 1.5055 while barely affecting Geo (`:118-184`);
- forced gold inclusion improves EVQ by only 0.0341 NLL, and the correct first token remains around rank 2,043 rather than top-1 (`:187-220`);
- sparse selection helps Geo more and converts neither arm (`:135-156`);
- on 303 QA examples, EVQ-LoRA is significantly worse overall, with the deficit concentrated at \(\le8\)K; all arms are near floor above 8K (`docs/exp/2026-07/2026-07-15_lora_qa16k_three_arm_results.md:52-97`).

Thus EVQ can improve a source-routing signal in this checkpoint while failing the readout gate. The present dense-dilution/sparse-conversion hypothesis is rejected for this setup.

### 4.2 Minimal discriminating experiments

These are a gated queue, not a precommitted rebuttal campaign.

#### Stage 0 — no training; highest information per cost

1. **Range/span decomposition.** Construct three schedules at the same \(K\):
   - original midpoint Geo;
   - a uniform log-frequency grid matched to EVQ's realized minimum and maximum frequencies;
   - current EVQ.

   `Geo-original -> Geo-span-matched` isolates extrema/span; `Geo-span-matched -> EVQ` isolates nonuniform shape. Evaluate the full sin/cos feature Gram under uniform, causal/local-heavy, and task-observed distance priors. If shape advantage disappears after matching span, withdraw the shape-specific mechanism.

2. **Frozen task-risk landscape.** On existing native- and EVQ-trained checkpoints, evaluate the same paired samples under a small fixed grid of \(\theta=\tau^2\) and the span-matched controls. Estimate the sign of \(A_{\mathrm{task}}\), local curvature, NLL, correct-token margin/rank, and generation endpoint. This directly tests whether the model-free proxy and actual task loss agree in sign.

3. **QA adapter \(\times\) runtime-frequency cross.** Reuse a fixed subset of the registered \(\le8\)K and 8--12K QA examples. Compare Native-LoRA and EVQ-LoRA under native and EVQ runtime frequencies, with base/no-adapter arms if budget allows. This is inference-only and separates pretrained/readout sensitivity from adapter co-adaptation.

Use Stage 0 in a response only if a reviewer asks the corresponding causal question. A negative result is still decisive and must be reported.

#### Stage 1 — low-cost mechanism localization, only after a relevant trigger

- **Association swap:** change query-to-answer identity while preserving source position and format; ask whether the causal logit delta follows the correct answer identity.
- **Layerwise causal trajectory:** track correct-token margin after source removal/restoration across layers. Continue only if a specific layer range gains and then loses answer information.
- **Position-factor control:** at fixed 16K total length, compare early versus late gold position to separate relative-distance OOD, absolute source-position OOD, and total/query-length OOD.

Stop if the source intervention never carries answer identity. No decoding heuristic, sparse budget, or temperature search can manufacture a missing trained readout.

#### Stage 2 — one controlled training test, only if Stage 0/1 identifies a trainable bottleneck

Use true-16K, source-dependent, association-matched supervision. Open only one module family chosen by the diagnostic:

- no answer-identity signal after routing \(\rightarrow\) V/O path;
- mid-layer signal that disappears late \(\rightarrow\) late O/MLP/readout path;
- parameter-matched Q/K arm as the control.

First gate on one seed's correct-token margin learning curve and exact match. Only after stable top-1/EM conversion should the recipe be frozen and compared across Geo/EVQ and the span-matched control, followed by additional seeds. This is a new capability-learning question; it cannot retrospectively prove zero-shot extrapolation in the submitted setup.

### 4.3 Explicit stop/defer list

Do not:

- repeat the failed 8K/50-step retrieval recipe;
- scan sparse budgets, block counts, selectors, temperature, beam, top-p, YaRN settings, or a broad \(\tau\) grid;
- launch another broad 8B LoRA or LongBench suite before a small causal gate succeeds;
- use PPL or hidden-state matching as a substitute for the task endpoint;
- claim that true-16K supervised training validates 8K-to-16K zero-shot extrapolation;
- invent a new \(\tau\) theorem during rebuttal.

## 5. Review-conditioned “last stand” strategy

| Review state on 2026-07-22 | Action |
| --- | --- |
| Positive reviews, errors not noticed | Still make one bounded integrity disclosure. Keep reviewer replies question-specific; do not duplicate the entire audit. |
| Baseline/theory concern is score-driving | Correction first; state unchanged numerics; withdraw the invalid interpretation; defend only the survivor claim. Do not lead with excuses or new experiments. |
| Practical/downstream concern is score-driving | Accept that no task gain is demonstrated. Use PPL/routing only as mechanism evidence and disclose the registered negative QA result if invoked. |
| Reviewer gives a precise causal criterion | Select at most one Stage 0/1 test that answers it regardless of sign. Predefine the endpoint and stop rule; do not create a benchmark campaign. |
| Reviews correctly identify structural flaws as blocking | Treat the goal as preserving the scientific record rather than forcing a score flip. Explicitly state which claims should not be used as an acceptance basis. |

The strongest honest rebuttal is not a larger pile of results. It is a precise separation between what is mathematically exact, what is conditional, what was empirically observed, and what has now been falsified.

## 6. Unresolved risks

1. No theorem connects the chosen surrogate to a shared minimizer of the full sin/cos RoPE kernel, trained attention, or task loss.
2. The finite-grid intervention changes density shape, extrema, span, and active-channel behavior together.
3. The task-sensitive \(A_{\mathrm{task}}/B_{\mathrm{task}}\) and the correct sparse-attention effective support are unmeasured.
4. The original Primary I/II comparator identities cannot be repaired by new results.
5. Official YaRN erases the abundant-channel gap, and DAPE-ish attention adaptation removes the plain EVQ advantage in the audited follow-ups.
6. The current 8B route improves long-position NLL/routing diagnostics without downstream conversion and substantially harms registered \(\le8\)K QA.
7. After corrections, reviewers may reasonably judge the empirical scope too narrow for acceptance. This is a real score risk, not a wording problem.

## 7. Source index

- Current method/theory/protocol boundaries:
  `../rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md`
- Long mathematical audit: `THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`
- Official-YaRN component follow-up: `EVQ_YARN_COMPONENT_ABLATION_20260714.md`
- DAPE identity and DAPE-ish follow-up: `real_dape_compare/FINDINGS.md`
- 8B retrieval, cross-swap, sparse, and causal readout diagnostics: `../docs/exp/2026-07/2026-07-14_lora_retrieval_conversion_probe.md`
- Registered 303-example QA result: `../docs/exp/2026-07/2026-07-15_lora_qa16k_three_arm_results.md`
- \(\tau\) historical sweeps: `../docs/exp/2026-02/2026-02-26_full_experiment_report.md`, `../docs/exp/2026-02/2026-02-27_evq_tau_sweep_results.md`
- Canonical schedule implementation: `../scripts/lib/rope/schedules.py`
- Submitted theory/proofs: `../paper/sections/03_theory.tex`, `../paper/appendix/a1_proofs.tex`
