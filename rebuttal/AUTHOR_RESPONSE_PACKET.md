# EVQ-Cosh Author Response Packet

日期：2026-06-10

用途：把当前 rebuttal 准备压成最终 author response 的写作入口。本文不是第二篇论文，也不是实验 wishlist；它只组织已经核实的回应、条件回应、以及不能写的句子。

## 0. Decision Switch

正式 response 写作前先选路径：

| Path | 何时使用 | LoRA 怎么写 | 风险 |
| --- | --- | --- | --- |
| A: Controlled LoRA path | 已有 Base / Geo+LoRA / EVQ-LoRA exact numbers、seed scope、same checkpoint/data/rank/steps | 作为最强补充控制，回答 LoRA confound 和 pretrained-checkpoint undertraining concern | 不能夸成 from-scratch industrial validation |
| B: Concession path | 当前 workspace 状态：没有 Geo+LoRA exact result table | 承认旧 LoRA row 是 supporting/post-hoc，不能隔离频率注入 | 放弃 LoRA 作为主防线，主力转到 provenance、scope、1B limitation、Figure fix |

当前证据状态：使用 Path B，除非用户补入 Geo+LoRA exact table。

## 1. Non-Negotiable Opening

Use this as the first paragraph or AC-facing summary:

> We thank the reviewers for the detailed comments. We agree that several rows should be read as mechanism evidence rather than production-scale validation. The central claim is narrower than universal long-context SOTA: EVQ-Cosh changes the training-time RoPE frequency substrate as a finite-spectral-budget allocation, and inference-time range scaling can act differently on that substrate. We have clarified this scope, corrected presentation issues that could reduce trust, added token/seed provenance, and relabeled supporting rows that were too broadly described.

If space is tight, use the short version:

> We clarify the scope as a PE mechanism study: EVQ-Cosh is a zero-learned-parameter training-time frequency allocation, complementary to inference-time range scaling, not a universal long-context SOTA or production deployment claim.

Never open with:

- “The reviewers misunderstood the paper.”
- “All concerns are solved by new experiments.”
- “EVQ replaces YaRN/LongRoPE/DAPE/FIRE.”

## 2. AC Change Log

Ready now:

> We made four concrete corrections/clarifications. First, we added total token budgets and seed scope for the primary anchors: Primary I uses 100M tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7; Primary II Table 4 is the 128-token, 15M-token seed-scoped DAPE-style diagnostic; Phase 11B is a separate 256-token, 100M-token supporting protocol. Second, we relabeled the 1B MLA row as a schedule-sensitivity check rather than robustness to training saturation. Third, we corrected the stale QuALITY figure so Figure 8 now plots Gold-answer NLL, consistent with Table 21. Fourth, we clarified that PK is teacher-forced NLL-gap retrieval unless separately marked AR exact.

Add only under Path A:

> We also added a matched Geo+LoRA control under the same pretrained checkpoint, data, rank, and adaptation schedule, so only the Geo+LoRA -> EVQ-LoRA difference is attributed to EVQ frequency injection.

Do not add the LoRA sentence under Path B.

## 3. R2 Empirical Response

R2 is the reviewer most likely to decide accept/reject. Answer in this order.

### 3.1 LoRA Confound

Path A only:

> The reviewer is right that the original two-row LoRA table could not isolate EVQ frequency injection from LoRA/LongAlign adaptation. We added a matched Geo+LoRA control using the same pretrained checkpoint, data, rank, and 300-step schedule. We now report Base, Geo+LoRA, and EVQ-LoRA side by side. The Base -> Geo+LoRA change measures adaptation cost; only the Geo+LoRA -> EVQ-LoRA difference is attributed to EVQ frequency injection.

Path A table template:

| Model | Adaptation | 8K PPL | 16K PPL | 32K PPL | Seed scope |
| --- | --- | ---: | ---: | ---: | --- |
| Base | none | `[BASE_8K]` | `[BASE_16K]` | `[BASE_32K]` | `[BASE_SCOPE]` |
| Geo+LoRA | same LoRA/LongAlign | `[GEO_LORA_8K]` | `[GEO_LORA_16K]` | `[GEO_LORA_32K]` | `[GEO_SCOPE]` |
| EVQ-LoRA | same LoRA/LongAlign + EVQ | `[EVQ_LORA_8K]` | `[EVQ_LORA_16K]` | `[EVQ_LORA_32K]` | `[EVQ_SCOPE]` |

Path B now:

> We agree that the original LoRA row is supporting and cannot by itself isolate EVQ frequency injection from LoRA/LongAlign adaptation. We therefore will not use the two-row Base vs EVQ-LoRA table as primary evidence. The row remains a post-hoc adaptation observation with explicit 8K cost, and a matched Geo+LoRA control is the required attribution test.

Do not write:

- “LoRA proves industrial-scale training.”
- “The full Base -> EVQ-LoRA gain is EVQ-specific.”
- “The +30% 8K cost is modest.”

### 3.2 Training Budget / Undertraining

Ready now:

> We added total token budgets and seed scope next to the primary anchors. Primary I uses 100M training tokens at \(L_{\mathrm{train}}=2048\) with seeds 42/123/7. Primary II Table 4 is a 128-token, 15M-token PE-dominant diagnostic; the later Phase 11B curves are a separate \(L_{\mathrm{train}}=256\), 100M-token supporting protocol and are not mixed with Table 4. Primary III MLA uses 500M tokens at \(L_{\mathrm{train}}=8192\) with three seeds.

Ready now:

> We also avoid describing Chinchilla-style token counts as “overtraining.” The narrower empirical point is that the EVQ signal is not explained by a simple undertraining-only story: in the MLA progression, the long-range gap grows while the in-range cost shrinks, and the 750M continuation row shows a large 16K gap despite low in-range PPL. These do not establish trillion-token from-scratch durability, but they rule out the simplest undertraining-only explanation.

Under Path A, append:

> The controlled LoRA result further tests a heavily pretrained checkpoint under a matched adaptation budget.

Do not write:

- “9B tokens is overtraining.”
- “Training longer cannot remove EVQ.”
- “The 1B row proves saturation robustness.”

### 3.3 YaRN / Training-Free Scaler Baseline

If no new sweep/eval is available:

> We agree that Table 2 is a matched-scale substrate comparison, not a tuned-scaler leaderboard. The question is whether the same range-scaling operation has different leverage on a Geo-trained versus EVQ-trained frequency substrate. We will make this scope explicit and avoid claiming dominance over best-tuned Geo+YaRN, Dynamic NTK, LongRoPE, or LongRoPE2.

If Geo+YaRN sweep is available:

> We added a Geo+YaRN scale sweep over `[SCALE_SET]`. The purpose is not to claim EVQ dominates all rescalers, but to test whether the matched-scale result is explained by an obviously mistuned Geo baseline. The best Geo+YaRN result is `[BEST_GEO_RESULT]`, while EVQ+YaRN is `[EVQ_RESULT]`.

If LoRA training-free scaler eval is available:

> We also include a training-free Geo+[Dynamic NTK/YaRN] reference at 16K/32K to separate EVQ-LoRA from the default eval-only context-extension baseline.

Do not write:

- “EVQ beats tuned YaRN.”
- “Training-free scaling is irrelevant.”

### 3.4 PK vs AR Exact

If Primary I AR exact is not available:

> We clarify the metric definition throughout: PK denotes teacher-forced NLL-gap retrieval unless explicitly labeled AR exact. We use PK as a positional-encoding diagnostic endpoint and do not use it alone as evidence that the model can generate the key.

If Primary I AR exact is available:

> We now report AR exact separately from teacher-forced PK. The original PK endpoint remains an NLL-gap diagnostic, not a claim of exact generation.

Do not write:

- “PK means exact retrieval.”
- “Teacher-forced metrics are equivalent to generation.”

### 3.5 Primary II Seed Scope

Ready now:

> We will make the seed scope explicit. The DAPE-style comparison is a PE-dominant diagnostic stress test, not the sole statistical anchor of the paper. Geo, DAPE, and EVQ are reported under the retained seed-42 protocol, while the learnable-tau row reports a 3-seed mean/std. We do not present this row as comprehensive DAPE or learned-PE dominance.

If extra seeds are available:

> We added additional seeds under the same 128-token protocol and report mean/std in the revised table: `[FILL]`.

Do not write:

- “Primary II is fully 3-seed for all rows.”
- “This proves broad learned-PE dominance.”

### 3.6 1B MLA Reversal

Ready now:

> We agree that the 1B MLA row should not be described as robustness to training saturation. It is a single-seed schedule-sensitivity stress check in a scarce-channel MLA regime, not a same-configuration token-scaling ablation. We have relabeled it accordingly and discuss it as a limitation motivating fixed-length continuation or stage-wise re-warp/adaptation. The primary MLA evidence remains the 8K/500M, three-seed scarce-channel stress test.

Optional mechanism sentence:

> This also explains why the progressive MHA row need not contradict the MLA reversal: the MLA setup has far fewer rotary channels, and K-dependent distortion terms make allocation mismatch more severe in scarce-channel regimes.

Do not write:

- “The reversal is noise.”
- “EVQ is robust to saturation.”

### 3.7 Figure 8 / Table 21

Ready now:

> We thank the reviewer for catching the stale/mislabeled QuALITY figure. The table values and text use gold-answer NLL; the figure panel was an older accuracy visualization and should not have been captioned as NLL. We have replaced the figure with a Gold-NLL plot consistent with Table 21, and we do not use QuALITY accuracy as a primary claim.

Do not write:

- “The reviewer misread the figure.”
- “Accuracy and NLL deltas are equivalent.”

## 4. R1 Theory Response

### 4.1 Shape vs Scale

Ready now:

> We agree that the paper should separate the two levels more explicitly. The cosh density is derived for the stated broadband surrogate, while \(\tau\) is used as an operating-point selector rather than a theorem of global optimality for trained attention. We will revise the text to avoid implying a single unified optimum and to state that the contribution is the shape-plus-calibrated-scale allocation rule.

If A.15 measurement is available:

> We also added a measurement-based check following the A.15 protocol: `[MEASUREMENT_SUMMARY]`. This ties the empirical active-band estimate to the selected scale without treating \(\tau\) as a learned parameter.

Do not write:

- “We derive the globally optimal tau.”
- “The surrogate is the trained-transformer objective.”

### 4.2 Learnable Tau

Ready now:

> This is an important negative result. The training objective only observes the in-range loss, while the extrapolation benefit is out of range and the in-range waterbed cost is immediate. Therefore gradient-based tau learning is biased toward the in-range basin and does not reliably discover the extrapolation allocation. We will add this interpretation and, if logs are included, the learned-tau trajectory.

If trajectory is available:

> In our runs, learned tau `[DRIFTED_TO_SMALLER / OSCILLATED / STAYED_FLAT]`, consistent with the training-loss signal being weak or myopic for extrapolation allocation.

Do not write:

- “The learned tau result automatically validates EVQ.”
- “Learnable tau failure is irrelevant.”

### 4.3 NTK-Aware Reversal

Ready now:

> We agree and will sharpen the wording. The primary composition claim is matched-scale EVQ+YaRN substrate/range complementarity, not universal monotonic compatibility with every inference-time rescaler. The NTK-aware row is useful precisely because it shows that composition depends on the downstream scaler.

Do not write:

- “EVQ helps any scaler.”
- “NTK-aware is a bad baseline.”

## 5. R3 Systems / Practicality Response

### 5.1 Practical Relevance

Ready now:

> We agree that production-scale from-scratch validation remains future work. The systems relevance is narrower: EVQ-Cosh is a zero-learned-parameter schedule change, the MLA experiment tests a scarce-rotary-channel regime relevant to compressed-attention designs, and the dead-channel audit exposes a diagnostic failure mode that can be applied independently of EVQ adoption.

Under Path A only:

> The controlled LoRA experiment adds an industrial-checkpoint adaptation anchor, not a from-scratch production-scale validation.

Do not write:

- “Production ready.”
- “Industrial scale proven.”

### 5.2 MLA Tau Convention / DeepSeek Scope

Ready now:

> We agree and will clarify the wording. The MLA experiment is a production-relevant scarce-channel stress test, not a production-identical DeepSeek validation. The \(d_{\mathrm{eff}}=d_{\mathrm{head}}\) rule is an empirical operating convention for this architecture; we will either add the \(\tau=d_{\mathrm{rope}}/\sqrt{L}\) sanity ablation or mark it as a limitation.

Do not write:

- “This is the DeepSeek configuration.”
- “\(d_{\mathrm{eff}}=d_{\mathrm{head}}\) is derived by the theory.”

### 5.3 Downstream Benchmarks

Ready now:

> We do not use QuALITY accuracy as a primary benchmark claim. At 454M parameters, accuracy is capacity-limited and near random; we report it as supporting/non-regression context while the probability-space NLL shows the directional effect. The primary evidence remains diagnostic mechanism tests rather than downstream leaderboard performance.

Do not write:

- “Benchmarks are irrelevant.”
- “QuALITY proves downstream improvement.”

## 6. AC Closing

Ready now:

> Across the reviews, the common concern is not whether frequency allocation is interesting, but whether the current evidence overstates its scope. We agree with that distinction. We therefore narrow the claim to training-time RoPE frequency allocation as a finite-spectral-budget design axis, add controls or provenance for the main empirical confounds where available, and relabel supporting stress checks that were too broadly described. The revised paper will not claim universal long-context SOTA or production-scale validation.

Path A final sentence:

> The matched Geo+LoRA control is the main new attribution evidence: it separates LongAlign/LoRA adaptation from EVQ frequency injection under the same adaptation budget.

Path B final sentence:

> Without the matched Geo+LoRA table, the LoRA row remains supporting only and should not carry the main rebuttal.

## 7. Word-Budget Priority

If the response budget is tight, spend words in this order:

1. AC change log: scope, token/provenance, Figure fix, 1B relabel.
2. R2 LoRA path choice: Path A table or Path B concession.
3. R2 undertraining + 1B limitation.
4. R2 YaRN matched-scale scope.
5. R1 shape/scale scope.
6. R3 systems relevance.
7. Optional extras only if exact results exist.

Cut first:

- VideoRoPE comparison discourse.
- Broad benchmark apology.
- New theory details.
- Long explanations of every appendix table.

## 8. Final Send Checklist

- [ ] Remove every `[FILL]`, `[SCALE_SET]`, `[BEST_GEO_RESULT]`, `[EVQ_RESULT]`, `[MEASUREMENT_SUMMARY]` placeholder.
- [ ] Choose Path A or Path B explicitly for LoRA; do not mix them.
- [ ] If using Path A, verify every LoRA number against `TABLE23_LORA_WORKSHEET.md`.
- [ ] If using Path B, remove all “controlled LoRA” upgrade language.
- [ ] Keep PK as teacher-forced NLL-gap unless AR exact is actually reported.
- [ ] Keep YaRN claim as matched-scale unless a scale sweep is actually reported.
- [ ] Keep 1B as schedule-sensitivity limitation.
- [ ] Keep QuALITY as NLL/supporting, not accuracy benchmark.
- [ ] Do not call EVQ a replacement for YaRN, LongRoPE, DAPE, FIRE, or learned PE.
- [ ] Do not call 9B tokens overtraining.
