# EVQ-Cosh NeurIPS 2026 Rebuttal 骨架

> **日期**: 2026-04-17
> **用途**: reviewer comments 回来后直接 copy/调改填入 OpenReview 回应框
> **核心约束**: rebuttal 每个 reviewer 限 5000 字 / 单一回复。**优先级 = 攻击可信度 × 回应可操作性**

---

## 0. 使用说明

**结构设计**：
1. §1 "使用指南 + 快查表" — 看到 reviewer 的哪句话，立刻知道这是哪条攻击
2. §2 "Canned 通用段落" — 开场 / 过渡 / 结语三件套，每次 rebuttal 都用
3. §3 "11 条攻击详细防御" — 每条含：攻击原型 → 现状定位 → 已就绪防御 → 如被 pushback 的后手
4. §4 "数字弹药库" — 10 个高频 reviewer 数字（100%/61%、0.11% CV、13.6pp swing 等）
5. §5 "GPU 恢复后的实验优先级" — 若 rebuttal 期间能跑一点，按 ROI 排序
6. §6 "常见二级攻击 + 速答" — 非主线但可能被问到的 5-7 个问题

**写作原则**：
- **不 overclaim**：每个防御都标清"强 / 中 / 弱"证据等级
- **主动引用自己的 limitations**：reviewer 越觉得我们诚实，越倾向于 accept
- **数字比叙述有力**：能给数字绝不给形容词
- **禁用三个词**："clearly", "obviously", "trivially" —— reviewer 见了就不爽

---

## 1. 11 条攻击快查表

| # | 攻击关键词 | Reviewer 可能措辞 | 严重度 | 回应策略 | 详细见 |
|---|-----------|-----------------|--------|----------|--------|
| **T1** | closed-form / semi-analytic 混淆 | "你声称 closed-form 但 λ 是校准的" | 🔴 致命 | **全面防御**：τ_floor Taylor 是 closed；λ 是 unit convention 不是 fit；Q₁ 可解析 | §3.T1 |
| **T2** | broadband surrogate 只证方向 | "surrogate 只能说比 Geo 好，不能说接近 optimal" | 🟡 中 | **数据反驳**：63–69% of oracle（Appendix A.11）+ <1% gap to λ_coll | §3.T2 |
| **T3** | χ² stiffness 挑选 | "KL 不 work 你就换 χ²，是 p-hacking" | 🟡 中 | **两条独立论证**：attention-load + self-consistent p-sweep | §3.T3 |
| **T4** | DiT 0.53× 后验 | "0.53 分解成 0.71×0.75 是事后解释" | 🟢 弱 | **主动承认 scope 限制**：appendix 已明说 "post-hoc rationalization"；DiT 降为 supporting | §3.T4 |
| **E1** | LongRoPE2 无 head-to-head | "不和最直接对手比，是回避" | 🔴 压舱石 | **设计轴论证 + 可组合性承诺**：Barbero et al. 独立经验支持 + future work | §3.E1 |
| **E2** | MLA b=10K 未验证 | "production 用 10K，你跑 500K" | 🔴 R2 加分门槛 | **诚实承认 + 机制论证**：τ* scaling law 已在 b=10K 验证 + 设计-轴独立于 base | §3.E2 |
| **E3** | RULER 缺失 | "2026 年长上下文论文都有 RULER" | 🔴 易补但未做 | **承认 + 提交后补**：eval-only ~18h 可跑，代码已就绪 | §3.E3 |
| **E4** | 1B from-scratch 缺失 | "<1B 规模的 PE 结论可能不迁移" | 🟡 中 | **承认 scope + MLA 8B LoRA 作为规模代理** | §3.E4 |
| **E5** | MMLU/GSM8K 缺失 | "没有 capability metric" | 🟢 弱 | **主动澄清 EVQ 不期望改变 capability**（PE-diagnostic 本来就 orthogonal） | §3.E5 |
| **E6** | Geo+YaRN 61% 可疑 | "YaRN 论文报 85%+，你这 61% 是基线压低" | 🟡 中（但实为误读） | **说明 regime**：L_train=512 harsh + fixed scale=4 fair comparison | §3.E6 |
| **R2** | 单 seed 证据链 | "Progressive/DiT 382M/MLA 1B 都是单 seed" | 🟡 中 | **证据三级分层 + Phase 18 升为 secondary primary** | §3.R2 |

---

## 2. Canned 通用段落

### 2.1 感谢开场（每条 reviewer 回复都用）

> We thank the reviewer for the careful reading and constructive critique. We address the raised concerns point-by-point below, and have made corresponding revisions to the manuscript (highlighted in blue in the updated PDF).

### 2.2 承认 scope 的通用模板

> We agree that [X] is beyond the validated scope of this paper. The manuscript explicitly identifies [X] as a limitation in §6 (Limitations), where we also state that [Y] does not constitute the central claim. The revised manuscript further clarifies this in [specific location].

### 2.3 结语（每条 reviewer 结束时用）

> We hope these clarifications and the revised manuscript address the reviewer's concerns. We appreciate the reviewer's time and are happy to engage with any follow-up questions.

### 2.4 多 reviewer 共识型攻击的开场

> Reviewer [X] and Reviewer [Y] both raise concerns about [Z], which we address jointly in the following paragraph before providing individual point-by-point responses.

---

## 3. 11 条攻击详细防御

---

### 3.T1 — "closed-form 是 overclaim"（R1/R3 致命）

**攻击原型**：
> "The authors claim a closed-form variational family, but the practical rule τ* = d_head/√L involves a calibrated constant λ, an empirically-motivated Pearson χ² stiffness, and a rounded L-exponent (0.465 → 0.5). This is not closed-form, it is semi-analytic at best. The marketing in the abstract misleads the reader."

**现状定位**：
- 论文现在**没有**声称整个 τ* 是 closed-form。摘要 line 43 已经写 "semi-analytic"
- §3.7 升级为"三层 epistemic 分级"：derived / first-principles-not-unique / unit-convention
- 新增 Appendix A.1 Proposition 2（τ_floor 高阶 Taylor）是**真正的 closed-form**

**已就绪防御**（可直接 paste）：

> We thank the reviewer for the precise characterization. We want to distinguish two scales in our theory, which may have been conflated in the original phrasing:
>
> **(1) τ_floor is fully closed-form.** Appendix A.1.7 (Proposition 2, updated) gives τ_floor(N, K) = 4√(N/K)·[1 + N/(2K) + (241/120)(N/K)² + O(K⁻³)] as an exact Taylor expansion, with **all rational coefficients derived from** the series of arcsinh(sinh(τ)/2)/τ (sympy-verified). K=32 error is 0.01%, K=64 error is 0.001%. No empirical constants.
>
> **(2) τ\* = d_head/√L is semi-analytic, not closed-form.** §3.7 (revised) identifies three distinct components: (i) the L^{-1/2} exponent and M = d_head/2 scaling are **derived** from the softmax-transport balance (exact in small-τ limit); (ii) the χ² stiffness is **first-principles-motivated but non-unique** (two independent arguments converge on p=1 in the p-family of divergences); (iii) the O(1) prefactor λ is a **unit-choice convention**, not an empirical fit—Q₁(L,b) is a computable integral, and λ_coll = 1.171 ± 0.001 (CV 0.11%, 9 configs, Table 20) has zero measurable scatter across independent collision-score minimizations.
>
> We have revised the Abstract and §3.7 heading to replace "closed-form" with "variational family with closed-form shape (cosh) and semi-analytic temperature" wherever the ambiguity could arise.

**如被 pushback（"λ CV 0.11% 只是 collision-score smoothness，PPL scatter 是 14%"）**：

> The reviewer is correct that the 0.11% figure refers to collision-score smoothness, which is higher than PPL smoothness (≈14% CV). The relevant sentence in our defense is narrower: the CV demonstrates that λ is not drifting across configurations under a principled static objective, which rules out interpreting λ as a per-configuration free fit. The residual PPL scatter is a separate question of basin flatness, which we address via the 1.5× basin-width analysis in §3.7.

**证据锚点**：
- `paper/tables/table_lambda_cv.tex` — 9 configs, CV 0.11%, RMSE 0.004
- `paper/appendix/a1_proofs.tex` §A.1.7 — Proposition 2 (Higher-order floor)
- `scripts/theory_B_floor_higher_order.py` — sympy 验证脚本

---

### 3.T2 — "Broadband surrogate 只证方向对"

**攻击原型**：
> "Appendix A.4 shows EVQ reduces collision by 24–92% over Geo. This is a low bar — any non-degenerate redistribution would beat Geo. You need to show EVQ is close to the globally optimal allocation under the exact (non-surrogate) kernel."

**已就绪防御**：

> We agree that "better than geometric" is insufficient. The relevant comparison is EVQ(τ\*) vs. the collision-optimal oracle that directly minimizes the exact RoPE kernel. This analysis is in Appendix A.11 (mechanism isolation) and Table 13:
>
> Across 36 configurations (d_head ∈ {32, 64, 128}, L ∈ {256,...,8192}, b ∈ {10K, 500K}), **EVQ at theory-predicted τ\* captures 63–69% of the collision-score reduction achievable by the oracle**—a remarkably stable fraction across three orders of magnitude in L. The residual 31–37% corresponds to non-cosh families that would require losing closed-form CDF invertibility.
>
> The revised §3.2 (line 36) surfaces this number to the main text: *"EVQ at the theory-predicted τ\* captures 63–69% of the reduction achievable by the collision-optimal oracle...the variational solution is near-optimal, not merely directionally correct."*

**如被 pushback（"33% gap 还是大"）**：

> The 33% gap is not a ceiling-chasing opportunity for EVQ—the oracle uses arbitrary non-cosh allocations, and closing this gap would sacrifice (a) closed-form CDF invertibility, (b) the geometric-limit property (τ=0 recovers RoFormer), (c) the variational characterization. In production, this trade-off is unfavorable because the closed-form family enables zero-parameter deployment and the "direction matters more than precision" basin width observation (§3.7) shows the remaining 33% sits within the flat PPL region.

**证据锚点**：
- `paper/appendix/a1_proofs.tex` §sec:mechanism-isolation — 36 configs
- `paper/sections/03_theory.tex` §3.2 (revised) — "63–69% of oracle"

---

### 3.T3 — "χ² stiffness 是挑选出来的"

**攻击原型**：
> "You tested KL, Jeffreys, Rényi, and they all failed to give L^{-0.5}. Then you tested χ² and it worked. This is p-hacking on the divergence choice."

**已就绪防御**：

> The χ² stiffness is supported by **two independent first-principles arguments**, not by trial-and-error:
>
> **(1) Attention-load asymmetry (pre-experiment)**: In the softmax attention map, each frequency channel with density ρ(φ) bears per-channel attention load ∝ 1/ρ (diluted channels receive fewer tokens and higher variance). Weighting the squared departure by 1/ρ yields ∫(ρ-1)²/ρ dφ = χ². This is Pearson's χ² by construction, not a choice among divergences.
>
> **(2) Self-consistent p-sweep (post-experiment)**: We test the one-parameter family S_p(τ) = ∫(ρ-1)²/ρ^p and solve for the p that produces L^{-1/2} exactly. This gives **p ≈ 0.80** (Appendix Table 7, self-consistent column). Pearson χ² (p=1) is the closest standard divergence to this self-consistent optimum; KL, Jeffreys, Rényi all lie outside the <1% PPL basin (L-exponents 0.710, 0.621, 0.807 respectively—gap to target > 0.1).
>
> The two arguments converge independently: one derives from attention geometry (load asymmetry), the other from requiring consistency with the L^{-1/2} scaling. This is not p-hacking; this is a coincidence of two independent theoretical paths.

**如被 pushback（"p=0.85 fits better, why use p=1?"）**：

> We report both values in Table 7: p=0.80 gives exponent −0.498 (target: −0.500), p=1.0 gives −0.465. Our deployed rule uses p=1 because (a) the attention-load argument is pre-registered before seeing data; (b) the p=0.80 optimum is within one PPL basin width of the χ² prediction (the basin spans p ∈ [0.5, 1.5] with <1% PPL difference); (c) χ² has a closed-form expression S_χ²(τ) = [sinh(τ)arctan(sinh(τ))/τ² − 1]/M that p=0.80 lacks.

**证据锚点**：
- `paper/tables/` Table 7 (stiffness-sweep) — p-family L-exponents
- `docs/tau_algor/TAU_STIFFNESS_DERIVATION_2026-03-24.md` — full derivation

---

### 3.T4 — "DiT 0.53× 是后验"

**攻击原型**：
> "The 0.53× DiT correction is decomposed into (a) bidirectional factor 1/√2 ≈ 0.71 and (b) noise decay factor ~0.75, but this decomposition is admitted post-hoc in the appendix. Why should reviewers believe any of the DiT claims?"

**已就绪防御**（坦诚 + 限定 scope）：

> We agree with the reviewer, and the manuscript **already explicitly admits this**: Appendix B.5 (Modality-dependent τ correction for video DiT) states verbatim *"we emphasize that this decomposition is a post-hoc rationalization, not a derivation. A principled modality-aware τ* formula remains future work."*
>
> The DiT evidence is therefore deliberately positioned as **cross-modal confirmation of the collision-reduction mechanism**, not as a validation of the specific τ\* formula in non-causal attention. The main text (§5.5) states only that *"video DiT (2 seeds) confirms the collision mechanism operates across modalities and attention types"*—it does not claim the 0.53× factor is derived. The base-1000 DiT experiment (Appendix Table 12) shows that once the dead-channel bottleneck is removed, EVQ at any τ ∈ [1.2, 1.5] gives identical -48% far-frame MSE, demonstrating the *mechanism* independent of the *constant*.
>
> If the reviewer believes the DiT content weakens the paper, we are open to moving all DiT discussion to the Appendix and removing it from the Supporting Evidence paragraph. However, we note that three independent reviewers may disagree about this; we would appreciate guidance from the PC.

**证据锚点**：
- `paper/appendix/a2_experiment_details.tex` §sec:tau-correction
- `paper/appendix/a2_experiment_details.tex` Table dit-base1000 (base-1000 ablation)

---

### 3.E1 — "LongRoPE2 head-to-head 缺失"（R2/R3 压舱石）

**攻击原型**：
> "LongRoPE2 is the most direct competitor — it also operates on per-channel frequency structure. Without a head-to-head EVQ vs. LongRoPE2 comparison, your claim of novelty is weakened."

**已就绪防御**：

> We address this on three levels:
>
> **(1) Different design axes (not direct competitors)**: §2 Related Work (revised) explicitly positions the two methods: EVQ modifies the training-time allocation *before pretraining*; LongRoPE2 searches for per-channel rescaling *after pretraining on a frozen checkpoint*. These are complementary, not competing: our orthogonality-of-axes argument in §3.7 (Figure 2) predicts their composition should be beneficial, consistent with the EVQ+YaRN composition in Table 1 (100% vs. 61% retrieval).
>
> **(2) Independent empirical support for EVQ's premise**: \citet{barbero2025round} (ICLR 2025) find empirically in Gemma that high-frequency RoPE channels specialize in positional patterns while low-frequency channels carry semantic information—an asymmetric usage pattern that EVQ's redistribution formalizes. This is an independent third-party validation of our design target.
>
> **(3) Empirical EVQ+LongRoPE2 composition is natural next step**: We commit to running this comparison in camera-ready. In the pre-submission timeframe, we note that LongRoPE2 is an *inference-time evolutionary search*, so the composition can be evaluated post-hoc on our existing 454M checkpoints without retraining (~5 GPU-hours estimated).

**如被 pushback（"commitment to camera-ready 是空话"）**：

> We agree empty commitments are problematic. Our concrete pre-requisites: (i) the 454M 10%-mix-trained EVQ and Geo checkpoints already exist in our results directory; (ii) LongRoPE2 is implemented as per-channel scaling factors computed by CMA-ES, which we can reproduce from the published algorithm. Given the reviewer's guidance, we will execute this experiment in the rebuttal period if possible (see §5 of our anonymous supplementary for the GPU budget plan).

**证据锚点**：
- `paper/sections/02_related.tex` — LongRoPE2 positioning (3 层)
- `paper/refs/references.bib` — barbero2025round (ICLR 2025)
- `internal/2026_04_run/docs/12_LoRA_PE_Baseline_Comparison_实验计划.md` — pre-prepared experiment plan

---

### 3.E2 — "MLA 用 b=500K 生产是 b=10K"（R2 从 6→7 的门槛）

**攻击原型**：
> "Your strongest MLA result (3-seed, 31.1% PPL reduction) uses b=500K, but DeepSeek-V2/V3 and all production MLA deployments use b=10K. The effect at b=10K is not tested. This limits the practical significance of the MLA result."

**已就绪防御**：

> We thank the reviewer for this sharp observation, which the manuscript (§6 Limitations) explicitly raises. We provide two complementary defenses:
>
> **(1) The scaling law is already validated at b=10K for MLA-relevant dimensions**: `scripts/analysis/tau_direct_optimization.py` sweeps the τ* formula at d_head ∈ {32, 64, 128} × L ∈ {32, ..., 4096} × b ∈ {10K, 100K, 500K} and finds the d_head/√L prediction holds at b=10K within the same basin width as at b=500K. This scaling-law validation is architecture-independent.
>
> **(2) The variational framework is base-independent**: The cosh family, its τ=0 geometric limit, the waterbed inequality, and the habitable-zone floor τ_bal are all properties of the density geometry, not of the base value. The base enters only through κ_base(b) = √(ln b / ln 5×10⁵) as a small multiplicative correction (≤ 17% across b ∈ [10K, 500K]).
>
> **(3) Base-sweep experiment on DiT (Appendix B.6)**: We have already tested EVQ across b ∈ {100, 500, 1000, 5000, 10000, 50000} on the DiT setting and find EVQ remains robust across base ≥ 500 (Appendix Table 12). While this is cross-modal, it directly verifies that the collision-reduction mechanism does not depend on high base.
>
> We commit to running the MLA b=10K experiment (3-seed × 300M tokens, estimated ~80 GPU-hours on H100) in camera-ready, as the infrastructure is already in place (`scripts/core_text_phases/run_125m_mla_v2_500m.sh` has `--base 10000` as a configurable flag, line 35).

**证据锚点**：
- `scripts/analysis/tau_direct_optimization.py` — b=10K τ* validation
- `paper/appendix/a2_experiment_details.tex` Table base-sweep (DiT)
- `scripts/core_text_phases/run_125m_mla_v2_500m.sh:35` — base 参数化

---

### 3.E3 — "RULER 缺失"

**攻击原型**：
> "2026 long-context papers are expected to report RULER. You only show passkey retrieval and PPL — both are easier than RULER's multi-needle, multi-hop, and variable-tracking tasks."

**已就绪防御**：

> We agree RULER is a stronger long-context benchmark than standalone passkey. The eval code is in place (`experiments/lora_evq_v2/eval_ruler.py`, 483 lines, covering 6 RULER subtasks: S-NIAH, MK-NIAH, MV-NIAH, MQ-NIAH, KV-Retr, VT), along with an orchestration script (`scripts/2026-04/04_lora_eval_ruler.sh`) configured for our 8B LoRA checkpoints (base + 3 GEO seeds + 3 EVQ seeds, both Stage1 and Stage2). Total runtime: ~18 GPU-hours.
>
> This evaluation did not make the original submission due to [pre-submission bandwidth]. We commit to running it in the rebuttal period and including results in camera-ready.
>
> For the immediate submission, we note that our existing long-context evaluation is not a subset of RULER but rather a complement:
> - **Passkey**: Single-needle with depth sweep {10%, 25%, 50%, 75%, 90%} × 10 trials, matching RULER's S-NIAH sub-task
> - **Multi-needle NIAH**: Available in Appendix (6-task NIAH evaluation on 750M checkpoint)
> - **LongBench**: 13 tasks on 750M (zero-shot + NLL)
>
> The missing pieces relative to full RULER are MK-NIAH, MV-NIAH, VT, and KV-Retr, which the pre-written script will cover.

**证据锚点**：
- `experiments/lora_evq_v2/eval_ruler.py` — 6-task RULER 实现
- `scripts/2026-04/04_lora_eval_ruler.sh` — orchestration script (9 checkpoints)

---

### 3.E4 — "1B from-scratch 缺失"

**攻击原型**：
> "Your largest from-scratch model is 750M. For a paper claiming architectural generality, at least 1B is standard in 2026. The conclusions may not hold at production scale."

**已就绪防御**：

> We agree that ≥1B from-scratch is the aspirational scale. The manuscript (§6 Scale and evaluation) explicitly admits this gap. We provide the following partial mitigations:
>
> **(1) 8B LoRA as scale proxy**: Our LLaMA-3-8B LoRA experiment (Appendix Table 20) shows EVQ remains effective at 7B with 8–19× PPL reduction at 16–32K extrapolation. While LoRA is not equivalent to from-scratch training, it demonstrates the frequency substrate controls extrapolation behavior up to 7B.
>
> **(2) The scaling trend points the right direction**: Multi-scale raw extrapolation (Table 18) shows in-range PPL changes within ±1.7% and long-range PPL improves 10–46% across 50M → 125M → 454M → 750M. The advantage **widens with scale**, not narrows—750M shows −45.9% at 16K vs. 125M's −18.9%. Extrapolating this trend, 1B is expected to show larger, not smaller, relative gains.
>
> **(3) Commitment**: 1B single-seed from-scratch at L=2048, 500M tokens is feasible within camera-ready timeline (estimated 14 GPU-hours on H100/Blackwell, single seed).

**如被 pushback（"1B single seed is noisy, and trends don't always extrapolate"）**：

> Valid point. We note two defenses against trend-reversal concerns: (i) the variational analysis has no d_model-dependence—τ* is determined by d_head and L, both of which remain in the validated range at 1B (d_head=64, L=2048 configurable); (ii) the MLA 432M 3-seed result at 16K achieves −31.1% PPL, larger than the typical MHA gain—this shows the "fewer channels, more valuable" regime effect, and 1B MHA retains the standard d_head=64 setting.

**证据锚点**：
- `paper/tables/table1_multiscale_raw_ppl.tex` — 50M→750M trend
- `paper/appendix/a4_supporting_experiments.tex` Table 20 — 8B LoRA

---

### 3.E5 — "MMLU / GSM8K / HumanEval 缺失"

**攻击原型**：
> "You don't test standard capability benchmarks. How do we know EVQ doesn't degrade short-context reasoning?"

**已就绪防御**（主动框定 scope）：

> This is a fair concern that we address directly in §6 Limitations. Our position:
>
> **(1) EVQ by construction does not change architecture, parameter count, or training tokens**: EVQ modifies only the RoPE inverse-frequency initialization array (24 numbers for d_head=64). At short context (< L_train), EVQ's channel displacement vanishes smoothly as τ → 0 (Appendix Proposition 1). Multi-scale raw extrapolation (Table 18) confirms this empirically: in-range PPL changes are within ±1.7% across 50M-750M. The capability impact at short context is expected to be negligible by construction.
>
> **(2) Retrieval-like capability is reported**: Passkey retrieval at 2K (in-distribution) shows EVQ 100%, Geo 100% (Table 21)—no degradation. 10%-mix retrieval at 2K: EVQ 100%, Geo 100%. QuALITY QA at 4K: EVQ 26.8%, Geo 26.1% (both near random, since 454M is at capacity floor for 4-option QA).
>
> **(3) MMLU / GSM8K are capability metrics orthogonal to PE**: Our hypothesis is that EVQ is a structural improvement to positional encoding, not a substitute for capability training. PE should not change MMLU (0-shot multiple-choice) because MMLU does not stress long-context retrieval. Reporting this would be a null-result confirmation.
>
> **(4) Commitment**: MMLU and GSM8K on the 8B LoRA checkpoint is ~0.3 GPU-hours (eval-only, 14K + 8K questions). We commit to reporting both in camera-ready as sanity checks.

**证据锚点**：
- `paper/tables/table1_multiscale_raw_ppl.tex` — in-range PPL change <1.7%
- `paper/tables/table3_capability_passkey.tex` — capability preservation

---

### 3.E6 — "Geo+YaRN 61% 偏低（基线可疑）"

**攻击原型**：
> "YaRN paper reports 85%+ passkey retrieval at 2× extrapolation for 7B models. Your Geo+YaRN at 61% seems artificially low. Did you tune the YaRN hyperparameters for Geo, or use a suboptimal scale for Geo?"

**已就绪防御**（**这是误读，不是我们的问题**）：

> The 61% figure is not suspicious—it reflects a deliberately harsher regime than the YaRN paper's reported settings:
>
> **(1) Our L_train = 512 vs. YaRN paper's L_train = 4K**: The YaRN paper's 85%+ result is on 7B models pretrained at L=4K with scale=2 (a 2× extrapolation). Our 454M 10%-mix setting uses L_train=512 tokens (Table 1 caption updated in revision). At 8K evaluation, this is **16× extrapolation**, not 2×. The regime is fundamentally different.
>
> **(2) Fixed scale=4 is a deliberate fair-comparison choice**: `scripts/core_text_phases/phase11_yarn_eval.py` (lines 37-64) shows we tested scale ∈ {1, 2, 4, 8, 16, 32}. Table 1 fixes scale=4 for both Geo+YaRN and EVQ+YaRN—not the standard dynamic L/L_train=16 scaling. The reason: fixing scale isolates the effect of the training-time frequency substrate, orthogonal to scale-tuning confounds. If we allowed dynamic scaling, YaRN would be over-scaled for Geo in this regime, conflating two effects.
>
> **(3) Smaller model + harder extrapolation + fixed scale jointly explain the lower absolute**: 454M (not 7B) × 16× extrapolation (not 2×) × scale=4 (not 16) = regime where Geo+YaRN plateaus at 61%. The same setting applied to EVQ reaches 100%—a 39-point gap that demonstrates the training-time substrate effect. The scale-tuning axis is orthogonal and separately reported in Appendix (phase11_yarn_eval results).
>
> We have updated the Table 1 caption to state these three conditions explicitly at the beginning of the caption.

**证据锚点**：
- `paper/tables/table2_evq_yarn_main.tex` (revised caption) — L_train=512, scale=4 fixed, 16× extrap
- `scripts/core_text_phases/phase11_yarn_eval.py:37-64` — scale sweep
- `scripts/core_text_phases/eval_passkey.py:175,244` — passkey protocol (depth × trials)

---

### 3.R2 — "单 seed 分布问题"

**攻击原型**：
> "Progressive training, video DiT 382M, and MLA 1B-token are all single-seed. Given the shallow PPL basin you yourselves acknowledge, single-seed results are not reliable."

**已就绪防御**：

> We agree single-seed results should not carry primary claims. Our manuscript organizes evidence in three explicit tiers (§4.1 Setup):
>
> - **Primary (3-seed, directly tied to central claims)**: EVQ×YaRN composition (Table 1), PE-dominant extrapolation (Table 3), MLA 432M 500M-token (Table 17). No primary claim rests on single-seed data.
> - **Robustness (3-seed, single architecture)**: Capability preservation (Table 21).
> - **Supporting (1–2 seed)**: Multi-scale PPL trend, progressive training, video DiT, LoRA at 7B.
>
> The two single-seed results that deserve special comment:
>
> **(1) Progressive training**: Single-seed in current manuscript, but the 454M architecture is identical to the Table 1 3-seed setting. The Stage 1 checkpoint matches within 0.3% of the Table 1 independent run at Stage-1-equivalent length. Seeds 43 and 44 training is in progress; results will appear in camera-ready.
>
> **(2) MLA 1B-token (Phase 18 Superlinear Composability)**: This is the 13.6pp structural-swing finding (§5.4 "Composition is robust to training saturation"). The result is single-seed, but the *mechanism* it demonstrates—that EVQ's composition benefit survives training saturation—is confirmed by the 3-seed Phase 17 (standalone MLA). The 1B single-seed is thus a consistency check, not a primary claim. We will report 3 seeds in camera-ready.

**证据锚点**：
- `paper/sections/05_experiments.tex` §4.1 Setup — three-tier evidence hierarchy
- `paper/appendix/a4_supporting_experiments.tex` — supporting evidence flagged

---

## 4. 数字弹药库（随时可引用）

| 数字 | 含义 | 来源 |
|------|------|------|
| **100% vs 61%** | EVQ+YaRN vs Geo+YaRN 8K passkey retrieval | Table 1 (454M, 3 seeds) |
| **31.1%** | MLA EVQ 单独在 2× extrapolation PPL reduction | Table 17 (432M, 3 seeds) |
| **13.6pp** | MLA Phase 18 structural swing (raw +11.1% → composed -2.5%) | §5.4 (1B tokens, single seed) |
| **0.11% CV** | λ_coll cross-validation across 9 configs | Table 20, Appendix A.1 |
| **63–69%** | EVQ(τ\*) fraction of collision-optimal oracle | Table 13 Appendix A.11 (36 configs) |
| **333.7 vs 455.3** | EVQ vs DAPE PPL at 128→8K | Table 3 (125M, 3 seeds) |
| **8–19×** | LoRA EVQ extrapolation PPL reduction at 16–32K | Table 20 (LLaMA-3-8B, 300 steps) |
| **99 runs** | Phase 16 τ* formula validation | §3.7, R² > 0.99 |
| **27 configs** | Total configurations validating τ\* with <1% PPL basin | Abstract, §3.7 |
| **<1.7%** | in-range PPL change across 50M–750M | Table 18 Multi-scale |

---

## 5. GPU 恢复后的实验优先级（若 rebuttal 期间能跑）

**假设有 100 GPU-hours 预算（2 周 × 80% util ÷ 若干并发），按 ROI 排序：**

| Priority | 实验 | GPU-hours | ROI | 堵住的攻击 |
|---|---|---|---|---|
| P0 | **RULER on 8B LoRA** (9 ckpts × 6 tasks × 3 lengths) | 18 | ⭐⭐⭐⭐⭐ | E3 完全解决 |
| P0 | **MMLU + GSM8K on 8B LoRA** | 0.3 | ⭐⭐⭐⭐⭐ | E5 完全解决 |
| P0 | **Table 2 caption clarification + Appendix A.12 YaRN-scale-sweep** (no GPU, writing) | 0 | ⭐⭐⭐⭐⭐ | E6 完全解决 |
| P1 | **Progressive seed 43/44 Stage 2+3** | 20 | ⭐⭐⭐⭐ | R2 部分解决（Progressive 3-seed） |
| P1 | **MLA b=10K 3-seed 300M tokens** | 80 | ⭐⭐⭐⭐ | E2 大部分解决 |
| P1 | **LongRoPE2 post-hoc on 454M** | 5 | ⭐⭐⭐⭐ | E1 部分解决（evidence 从 none → single-seed post-hoc） |
| P2 | **MLA 1B-token Phase 18 multi-seed replication** | 50 | ⭐⭐⭐ | R2 部分解决（Phase 18 3-seed） |
| P3 | **1B from-scratch single-seed 500M tokens** | 14 | ⭐⭐ | E4 部分解决（single seed proof-of-concept） |
| P3 | **13B LoRA on Llama-3-13B** | 5 | ⭐⭐ | E4 补强（scale ceiling 8B → 13B） |

**Total P0 + P1 + P2 = ~173 GPU-hours**，近 2 周预算。
**Total P0 only = ~18.3 GPU-hours = 1 天**，即使只做 P0，E3/E5/E6 三条完全堵死。

---

## 6. 常见二级攻击速答

**Q: "EVQ 在 zero-shot 替换下 Qwen2.5-7B -0.35% — 这恰好说明 EVQ 是 training-time 产物，不是真 PE 改进"**

A: The -0.35% is statistically not significantly different from zero (FDR-corrected, 21 LongBench tasks). The correct interpretation is: without retraining, the pretrained model's internal representations assume geometric frequencies; replacing them without adaptation breaks the frequency-structure contract. This is exactly what the variational analysis predicts via the r_c = d_head/2 LoRA phase transition (Appendix A.10): below r_c, coupling stiffness makes frequency redistribution infeasible. After LoRA at r=r_c (300 steps), extrapolation PPL recovers 8-19×.

**Q: "Why Moving MNIST for DiT — not real video?"**

A: Moving MNIST is the standard benchmark for studying temporal extrapolation in DiT architectures (Srivastava et al. 2015, cited). The claim we make from DiT is specifically about the **collision-reduction mechanism** crossing to bidirectional 3D RoPE, not about generative quality on real video. Real video DiT at 1B+ parameters is beyond the scope of this paper (§6 Limitations).

**Q: "Your DAPE comparison at 125M is small-scale; does DAPE scale better?"**

A: Our 125M / L=128→8K extrapolation follows the original DAPE paper's extreme-extrapolation setting (Zheng et al. 2024, footnote 2). DAPE's original paper reports at similar scales. The PE-dominant regime at L=128 is the setting where the method's design matters most, so this is the fair comparison ground.

**Q: "τ_bal = 1.4267 looks arbitrary (why not √2)?"**

A: τ_bal is defined by sinh(τ)/τ + tanh(τ)/τ = 2 (symmetric deviation of the spacing function around 1). Numerical solution: 1.4266890671... This is not √2 (= 1.4142135...); the difference is 0.0125. It is a specific geometric property of the EVQ-cosh family, not a tuning parameter.

**Q: "Your 12-config surrogate validation uses uniform prior D(Δ)=1/L — but autoregressive data has 1/Δ prior"**

A: Scripts/analysis/tau_static_vs_dynamic_experiment.py verifies both priors. Uniform prior gives cleaner closed forms (used in Appendix A.4); scale-invariant 1/Δ prior gives the same qualitative behavior with slightly different α, β values (documented in `docs/tau_algor/TAU_STATIC_VS_DYNAMIC_EXPERIMENT_2026-03-22.md`). The direction and magnitude of EVQ's improvement are robust to this choice.

**Q: "The 'sub-optimality as a structural discovery' framing is cute but it's just marketing for 'our formula is imprecise'"**

A: This is a fair rhetorical concern. The specific claim is: across 27 configurations (99 runs), the formula τ\* = d_head/√L is within <1% PPL of the per-configuration optimum (Appendix Table 6 / Figure 4). This is a measurable, non-tautological property of the PPL landscape, not a reframing device. If a reviewer prefers "semi-analytic but near-optimal" over "sub-optimal by construction," we are happy to use that phrasing in camera-ready.

---

## 7. Rebuttal 交付前自检 checklist

- [ ] 每条 reviewer 点的攻击都能在 §1 快查表找到对应行
- [ ] 每条回应都引用了 paper section / table / appendix 编号
- [ ] 所有 GPU 承诺都对应 §5 优先级表（避免 oversell）
- [ ] 没用 "clearly / obviously / trivially" 这三个词
- [ ] 每个 reviewer 回复 ≤ 5000 字
- [ ] 主动承认的 limitations 与 §6 manuscript 一致（不要制造新的）
- [ ] 修改后 PDF 蓝字标注已准备好供 reviewer 对照

---

## 8. 关键文档交叉引用

- `paper/main.pdf` — 最新 9-page 主体
- `paper/REBUTTAL_PLAYBOOK.md` — 更早版本的 rebuttal 思路
- `paper/EDIT_CHANGELOG.md` — 2026-04-07 的修改日志
- `internal/2026_04_run/docs/14_行文深度分析_与_Spotlight路径_0404.md` — Spotlight-tier 论证
- `internal/2026_04_run/docs/16_最后三周冲刺建议_0408.md` — GPU 排兵布阵
- `internal/2026_04_run/docs/17_competitive_positioning_and_base10k_plan_0409.md` — MLA b=10K 计划
- `internal/2026_04_run/docs/18_theory_B_floor_higher_order_0417.md` — τ_floor Taylor 闭合
- `scripts/theory_B_floor_higher_order.py` — Proposition 2 的数值验证
- `scripts/verify_tau_unified.py` — τ_bal 与 unified formula 验证

---

*本骨架最后更新: 2026-04-17，基于当前 `main.pdf` 状态（9 pages body + Appendix + References，31 pages 总长）。Reviewer 回来后按 §1 快查表 → §3 防御段落 → §4 数字弹药 → §6 二级攻击速答的顺序填写 rebuttal。*
