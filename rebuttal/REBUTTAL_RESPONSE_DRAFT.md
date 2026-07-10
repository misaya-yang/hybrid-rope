# EVQ-Cosh Rebuttal Response Draft

**Package status:** `draft_with_placeholders`

**Scope:** response text for Original Questions 1, 8, 11, and 14

**Claim boundary:** EVQ-Cosh is a zero-learned-parameter, training-time frequency-allocation method. We defend a matched-scale substrate effect in the tested regimes, not universal long-context superiority or replacement of inference-time scaling.

We thank the reviewers for identifying places where the original presentation did not separate the evidence levels sharply enough. Our response below makes four corrections or clarifications. First, we explicitly separate the surrogate-derived cosh family from the calibrated operating-point rule. Second, we narrow the base claim to the high-base, finite-channel regime that motivated EVQ and retain the reported negative regimes. Third, we report the recovered autoregressive exact-match results separately from teacher-forced retrieval. Fourth, we correct the QuALITY figure provenance and use only the full \(n=2086\) gold-answer-NLL evaluation.

## Original Question 1 — Why are allocation shape and operating scale derived separately?

**Reviewer concern.** The cosh allocation shape is derived from a variational collision surrogate, whereas the practical rule for \(\tau\) comes from a separate stiffness--utility model. Does this split make the theory post-hoc?

**Response.** We agree that the original presentation could make these two epistemic layers appear more unified than they are. We have therefore lowered and clarified the theoretical claim.

The variational result answers a **shape question**: conditional on the broadband collision surrogate \(\mathcal C_{\mathrm{app}}\), what normalized frequency density minimizes the surrogate? Its stationary solution is the one-parameter cosh family. We do not claim that this surrogate is the exact trained-transformer objective or a pointwise approximation to the full oscillatory RoPE kernel.

The stiffness--utility model answers a different **scale question**: where should a practical model operate within that one-parameter family? Under the stated diffuse-softmax and small-\(\tau\) assumptions, it derives the structural dependence \(\tau\propto d_{\mathrm{head}}L^{-1/2}\). The unit prefactor is calibrated against a broad, flat empirical basin; it is not claimed to be a globally optimal constant. Thus the method is deliberately semi-analytic: the first stage selects a simple closed-form family, and the second selects a robust operating region within that family.

This separation is useful rather than costless unification. It gives an elementary inverse CDF and a single deployable parameter while keeping each assumption auditable. The resulting intervention changes only the initialized inverse-frequency table: it adds no learned parameters, auxiliary loss, new attention operator, optimizer state, or inference-time computation. We now state this as a conditional derivation plus empirical calibration, not as a first-principles solution of trained attention.

**Revision status.** The theory section now explicitly distinguishes surrogate-derived shape, semi-analytic scale, convention-dependent prefactor, and unmeasured trained-attention assumptions. It describes \(\tau=d_{\mathrm{eff}}/\sqrt L\) as an operating-point or basin selector, not a theorem of global optimality.

## Original Question 8 — Could a tuned geometric base explain the gains?

**Reviewer concern.** EVQ may benefit only because the geometric RoPE base was not tuned. Could a smaller base recover the same behavior?

**Response.** We agree that a matched training-time base sweep is the nearest one-knob control, and we do not claim that EVQ beats geometric RoPE for every base or sequence length. The relevant distinction is between **range scale** and **allocation shape**. Changing the geometric base moves the endpoints and uniformly rescales the log-frequency grid; EVQ changes the density of a fixed finite set of channels at the same base. These are related but non-equivalent interventions.

Importantly, our evidence is not filtered to show only favorable regimes. At \(b=10\mathrm{K}, L=4096\), the paper's analysis places the deployed bare prefactor 25% away from \(c_{\mathrm{pred}}=0.80\), marginally outside the validated \(\pm20\%\) basin. Our existing 350M experiment record also reports an underperforming bare-rule case in this regime. Because that trained row is not currently represented by a curated JSON, we use it only as a disclosed negative boundary, not as primary quantitative evidence. The appendix recommends the explicit \(c_{\mathrm{pred}}(L,b)\) correction at small \(b\) and large \(L\), rather than unconditional use of the bare rule. This boundary is consistent with the mechanism: when the geometric grid already provides sufficient usable phase coverage, aggressive reallocation can impose a waterbed cost without enough spectral headroom to repay it.

The video-DiT base scan supplies an independent negative control. When all temporal channels are already alive (base \(=100\)), geometric RoPE wins by approximately \(20\%\). As the base increases and dead channels emerge, EVQ becomes more useful and substantially more stable across the tested grid. The same scan therefore contains both the failure regime and the predicted benefit regime; it does not support base-independent dominance.

Our primary text setting targets a different, increasingly common regime. Modern long-context models such as LLaMA-3 use a large RoPE base (500K). A large base reduces rapid phase wrapping over long ranges, but at a finite training length it also leaves more low-frequency channels with negligible observed rotation. EVQ is designed for this finite-spectral-budget regime: it reallocates the existing channels so that fewer remain effectively inactive. This is why the primary claim is narrow: EVQ is a training-time allocation remedy for high-base, finite-channel long-context models, not a universal replacement for geometric RoPE or base tuning.

We also retain the raw-backed \(151.9\)M, \(L_{\mathrm{train}}=512\), seed-42 pilot showing that the improvement direction occurs at both base 10K and 500K. This supporting pilot does not replace a fully tuned geometric grid, and it is a different protocol from the reported \(b=10\mathrm{K}, L=4096\) boundary. We will not conflate the two.

**Revision status.** The paper now makes the high-base scope explicit, discloses the negative base regimes, and gives a conditional practitioner rule. We do not claim that the current evidence exhausts all tuned geometric bases.

## Original Question 11 — Teacher-forced retrieval versus autoregressive exact match

**Reviewer concern.** The reported passkey score is teacher-forced and may not reflect actual autoregressive generation.

**Response.** We agree. The earlier shorthand was too easy to read as generation accuracy. We now name the original metric explicitly as **teacher-forced NLL-gap retrieval** and report the recovered **autoregressive exact-match rate** as a separate endpoint.

The original Primary-I payload contains both fields under the same matched-scale evaluation. At 8K, each seed contains 50 passkey trials:

| Method at 8K | Seed 42 AR exact | Seed 123 AR exact | Seed 7 AR exact | Mean AR exact | Mean TF retrieval |
|---|---:|---:|---:|---:|---:|
| Geo+YaRN (\(s=8\)) | 0.0% | 0.0% | 0.0% | **0.0%** | 61.3% |
| EVQ+YaRN (\(s=8\)) | 58.0% | 18.0% | 98.0% | **58.0%** | 100.0% |

The conclusion is stronger and more precise than the teacher-forced result alone: in this matched-scale 8K stress test, Geo+YaRN retains a positive teacher-forced ranking signal but fails to convert it into a correct autoregressive answer in every seed, whereas EVQ+YaRN preserves non-zero generation accuracy in every seed and reaches 58.0% on average. The large \(18\%\)--\(98\%\) seed range also prevents overstatement; we treat the result as a material generation-level separation in this protocol, not as a universal task-accuracy claim or a formal significance result at \(n=3\) seeds.

This distinction also clarifies why we retain both metrics. Teacher-forced NLL-gap retrieval is a sensitive positional diagnostic, while autoregressive exact match tests whether that signal survives decoding. Reporting them side by side exposes, rather than hides, the gap between ranking and generation.

**Revision status.** The rebuttal will report both metrics with their full names and seedwise values. No teacher-forced score will be described as autoregressive accuracy.

## Original Question 14 — QuALITY figure/table inconsistency

**Reviewer concern.** The QuALITY figure and table used inconsistent sample counts and appeared to report different endpoints.

**Response.** We agree that this was a real figure-provenance error. The previous figure used an obsolete 200-sample, accuracy-only pilot, while the retained table used the later full evaluation. We have removed the pilot from the evidentiary chain and replaced the figure with the same four gold-answer-NLL rows used by the full \(n=2086\) evaluation.

The corrected protocol is now consistent throughout: the 454M checkpoints were initialized at 2K, continued and fine-tuned at 4K, and evaluated at 4K, 8K, and 16K on all 2,086 QuALITY validation examples. The obsolete 32K pilot point is excluded. Accuracy remains near the 25% random baseline and has no stable directional gain, so we do not use QuALITY accuracy as positive evidence. The retained observation is the supporting probability-space result, including the 8K raw gold-answer-NLL change from 3.2021 to 2.2392 (\(-30.1\%\)).

No experimental value was changed to create this correction. We synchronized the figure, caption, table, sample count, checkpoint history, and source artifact to the recovered full-evaluation aggregate. The curated record is raw-JSON-backed and explicitly marks the 200-sample pilot as superseded.

**Revision status.** Figure and table now use one source of truth: the \(n=2086\) full gold-answer-NLL evaluation. The old 200-sample pilot is excluded from both the figure and the rebuttal claim.

## Evidence anchors

- Original Primary-I payload and recomputed TF rows: `data/curated/primary1_evq_yarn_10pct_raw.json`
- Theory shape/scale stratification: `paper/sections/03_theory.tex`
- Base-dependent basin and explicit correction: `paper/appendix/a1_proofs.tex`
- DiT base sweep and dead-channel audit: `paper/appendix/a2_experiment_details.tex`
- Corrected QuALITY full evaluation: `data/curated/quality_454m_full_eval.json`
- Corrected QuALITY table and figure text: `paper/appendix/a3_supporting_results.tex`

## 中文核对（作者内部，不提交）

- Q1：必须保留“shape 来自 surrogate、scale 来自另一 stiffness--utility 模型、常数项经验校准”的承认；不能再写成统一的 full-attention 第一性原理推导。
- Q8：防御重点是“适用域明确且负例可预测”，不是声称已完成所有 tuned-Geo controls。`b=10K, L=4096` 的 350M 负例当前缺少 curated JSON，只能作为边界观察；它与 `L_train=512` 的 raw-backed base pilot 属于不同协议，不能混写。
- Q11：AR 数据可直接使用，但必须同时给出三种子离散度；不要把 58.0% 写成所有种子都稳定达到约 58%。
- Q14：主动承认旧图来源错误；只保留 \(n=2086\) Gold-NLL 结果，QuALITY accuracy 仍是 inconclusive。
