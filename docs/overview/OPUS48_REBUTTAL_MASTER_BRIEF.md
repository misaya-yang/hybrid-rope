# Opus 4.8 Rebuttal Strategy Brief

用途：这是给导师/合作者快速制定 rebuttal 策略的单一 Markdown 文档。

发送建议：只发这个文件即可。老师不需要先看其它审计文档；文中出现的代码/结果文件路径只是为了说明证据来源，方便之后内部核查。

本文不新增实验结果，也不试图把所有 supporting evidence 都塞进 rebuttal。目标是帮助老师判断三件事：

1. 哪些核心 claim 可以 defend。
2. 哪些 reviewer attack 必须 concede / scope down。
3. 有限时间里，最值得补哪一个 artifact 或实验。

## 0. 一句话结论

论文仍然可以 defend，但必须收缩为机制论文：

> EVQ-Cosh 不是通用 long-context SOTA，也不是 YaRN/LongRoPE 替代品；它提出并验证了一个训练期 RoPE frequency allocation 设计轴。最强证据是 matched-scale EVQ x YaRN 和 8K/500M 3-seed MLA scarce-channel stress test。1B/4K MLA 反向结果必须主动承认并作为 limitation/root-cause target 处理。

如果 rebuttal 试图证明“EVQ 一定随训练更久更强”“EVQ+YaRN 全面打败 tuned Geo+YaRN”“1B row 支持 durability”，会被 Opus4.8 直接打穿。

## A. 需要老师拍板的 6 个策略问题

| 问题 | 推荐选择 | 原因 |
| --- | --- | --- |
| Rebuttal 主 claim 要不要收缩？ | 收缩为 mechanism / design-axis claim。 | 当前证据强在机制 stress test，不强在 production SOTA。 |
| 1B/4K MLA 反向结果怎么处理？ | 主动承认 limitation，解释不是 same-config。 | 隐藏或强辩都会被抓；当前缺 exact JSON/checkpoint/data hash。 |
| 如果只能补一件事，补实验还是补 provenance？ | 先补 artifact provenance。 | 最快把“报告线索”升级为 reviewer-grade evidence；也能澄清 1B freq/data 问题。 |
| MLA `tau=1.414` 怎么 defend？ | 承认 empirical convention，最好补 direct tau ablation。 | 这是 Opus4.8 对 strongest MLA result 的硬攻击。 |
| Table 2 EVQ+YaRN 要不要说打败 YaRN/LongRoPE？ | 不要。只说 matched-scale complementarity。 | 没有 tuned Geo+YaRN / LongRoPE-style dominance evidence。 |
| LoRA/video/progressive/750M 要不要主动展开？ | 不主动展开。 | 都是 supporting/exploratory，主动讲会增加攻击面。 |

如果老师时间只有 10 分钟，建议只看：`0`、`A`、`3`、`4`、`7`、`10`。

## B. 给老师的快速阅读版

### B.1 这篇 rebuttal 的胜负手

真正要保的是一个窄但干净的 claim：

> Training-time RoPE frequency allocation is an independent design axis. EVQ-Cosh
> is a closed-form allocation that changes the learned frequency substrate, and
> matched inference-time scaling can have higher leverage on that substrate.

不要让 rebuttal 变成：

- 长上下文 SOTA 争夺。
- YaRN/LongRoPE replacement 争夺。
- 1B token durability 争夺。
- downstream accuracy 争夺。

如果 reviewer 是 harsh / Opus4.8 风格，他会抓的不是“EVQ 完全没用”，而是：

1. 你把机制 stress test 写得像 broad validation。
2. 你把 single-seed/supporting rows 用得太重。
3. 你把 1B 反向结果藏到 appendix 或解释成支持。
4. 你没有把 PK metric、MLA tau convention、baseline gap 讲清楚。

### B.2 我们现在最强的三根柱子

| Pillar | 能支撑什么 | 不能支撑什么 |
| --- | --- | --- |
| 454M EVQ x YaRN matched-scale table | EVQ substrate makes fixed YaRN scale more effective in this setting。 | 不能说打败 tuned Geo+YaRN / LongRoPE / LongRoPE2。 |
| 8K/500M MLA 3-seed scarce-channel stress test | 在 scarce rotary channels 下 allocation quality matters。 | 不能说 production-identical DeepSeek MLA，也不能解决 tau convention attack。 |
| PE-dominant 128-to-8K diagnostic | 在极端 PE-dominant setting 中，closed-form EVQ 比 Geo/DAPE seed-42 row 更好。 | 不能说完整 3-seed learned-PE dominance。 |

### B.3 现在最该主动承认的三件事

1. 1B/4K MLA 是真实 limitation，不是同配置 token-scaling ablation。
2. MLA `tau=1.414` 是 empirical convention，直接 ablation 仍缺。
3. Table 2 PK 是 teacher-forced NLL-gap，不能当 AR exact retrieval。

### B.4 推荐的 rebuttal 风格

强度要像这样：

> We agree with the reviewer that several rows should be read as mechanism
> evidence rather than production-scale validation. We revised the framing to
> separate primary stress tests from supporting rows, and we do not use the 1B/4K
> MLA row as primary support. The main claim is narrower: EVQ changes the
> training-time frequency substrate, and matched range scaling acts differently
> on that substrate.

不要像这样：

> Our results show EVQ is generally superior for long context and scales with
> more training.

第二种会被 1B row、LoRA control、tuned-scaler gap 一起打穿。

## C. Evidence Tiering For Rebuttal

这部分是给老师判断“哪些数字可以放 rebuttal 主段落，哪些只能被动回答”的。

| Tier | Evidence | Use in rebuttal | Risk |
| --- | --- | --- | --- |
| Primary I | 454M EVQ x YaRN, fixed matched YaRN scale, 3 seeds per method, PK teacher-forced NLL-gap。 | 主动引用。用来证明 substrate/range complementarity。 | reviewer 会问 tuned Geo+YaRN / AR exact。 |
| Primary II | PE-dominant 128-to-8K diagnostic；Geo/DAPE/EVQ seed 42；learnable tau row multi-seed。 | 可以引用，但必须写 diagnostic / seed-scoped。 | reviewer 会攻击 single-seed primary。 |
| Primary III | 8K/500M MLA 3-seed scarce-channel stress test。 | 主动引用，是最强 systems evidence。 | reviewer 会攻击 tau convention 和 production mismatch。 |
| Supporting | 1B/4K MLA, LoRA, video DiT, progressive, 750M continuation, downstream NLL。 | 除非被问，不主动展开；可作为 scope/future-work/context。 | single-seed、control 缺失、artifact gap。 |
| Do-not-overuse | 旧 docs/exp 里的 early reports、paper-ready 旧标签、historical launch wrappers。 | 不作为 rebuttal 权威证据。 | 容易触发 provenance attack。 |

## D. Opus4.8 攻击面和推荐回应

| Reviewer attack | 我们的真实状态 | 推荐回应 |
| --- | --- | --- |
| Novelty只是换一个低频 schedule。 | 缺 rebased-Geo / fixed-interpolation training-time control。 | 承认 simple-schedule baseline 是 open control；强调当前贡献是 closed-form allocation axis，不 claim 排除所有 schedules。 |
| Theory 是 fitted surrogate，不是 full attention derivation。 | 是。变分结论 conditional on broadband surrogate。 | 说 surrogate-derived and functionally validated。不要说完整 attention theorem。 |
| `tau=d_eff/sqrt(L)` 不是严格理论。 | 是 operating default / basin selector。 | 承认；强调 tau basin 和 non-geometric allocation，而非唯一最优 tau。 |
| Table 2 PK 太软。 | 是 teacher-forced NLL-gap。 | 主动说清 metric；不把它说成 generation exact match。 |
| EVQ alone 在 454M passkey 不强，主要是 EVQ+YaRN。 | 是。 | 这正好支持 substrate/range complementarity，不支持 raw EVQ universal win。 |
| matched-scale YaRN 不等于 tuned baseline。 | 是。 | 承认，不 claim tuned-scaler dominance。 |
| PE-dominant primary 是 seed 42。 | 是。 | 改成 diagnostic，尽量不要作为唯一主证据。 |
| MLA tau convention underived。 | 是硬点。 | 承认 empirical convention；补 ablation 或作为 limitation。 |
| 1B raw EVQ 反转。 | 是最危险 limitation。 | 主动承认，不作为支持。解释不是 same-config；需要 artifact recovery。 |
| LoRA 缺 Geo+LoRA。 | 是。 | supporting only。 |
| Video/progressive/750M single-seed or domain-specific。 | 多数是 supporting。 | 不主动展开。 |
| downstream task signal weak。 | accuracy 弱，NLL/diagnostic 更强。 | 说 capacity-limited downstream check；主 claim 不靠 downstream SOTA。 |
| artifact/script drift。 | 过去有 drift，当前已整理。 | 用 manifest/audit scripts；不要引用 historical wrapper 作为 canonical reproduction。 |

## E. 1B/4K MLA 法医结论

### E.1 老师应该知道的核心事实

用户最初担心的是：1B 为什么出现反向？是不是代码错？是不是 freq 凭空来的？是不是之前 AI 找错代码？

当前仓库审计结论：

- 不是“没有代码来源”。训练 launcher 和 YaRN+FT eval code 都存在。
- 不是“当前代码一定会凭空构造 freq”。当前 eval scripts 已经改成必须从 checkpoint 读取 `inv_freq`，clone loaded table，再 apply YaRN。
- 但 1B/4K row 在 compact repo 里不是 reviewer-grade evidence，因为 exact JSON、checkpoint hash、data hash 缺失。
- 1B/4K row 和 primary 8K/500M MLA 不是只差 token count；不能当 longer-training ablation。

### E.2 1B row 现在能说什么？

能说：

- 这是一个真实 limitation signal。
- 它提示 EVQ 在 sparse 4K frequency-window regime 可能 raw extrapolation 失败。
- 它不推翻 finite-spectral-budget 机制；反而说明 allocation shape 不是 universally helpful，错窗口会 hurt。
- 它需要 artifact-level recovery 才能升级。

不能说：

- 1B 证明 EVQ durability。
- 1B 证明 EVQ 只要加 YaRN 就 always win。
- 1B 是 primary 8K/500M 的 longer-training continuation。
- 1B 能直接解释所有 scale behavior。

### E.3 1B 反向的可能原因排序

| Cause | 当前证据 | 置信度 |
| --- | --- | --- |
| Not same-config: 4K train length vs 8K primary。 | 文档/报告链支持。 | High |
| Sparse MLA-32/K16/base500K frequency window。 | Phase22/23 诊断和 schedule analysis 支持。 | Medium-high |
| Dataset/cache provenance drift。 | 缺 data hash；不能排除。 | Medium |
| Exact checkpoint/eval JSON missing。 | compact repo 缺失。 | High |
| Seed outlier。 | 1B row 当前报告 seed 42；缺多 seed。 | Medium |
| YaRN freq 使用局部构造而非 checkpoint。 | 当前代码已修；历史行为需 checkpoint audit。 | Low-to-medium |
| Label swapped / table copy error。 | 没有直接证据；缺 exact JSON 使其不能完全排除。 | Low-to-medium |
| True over-training effect。 | 可能，但当前不是 same-config scan，不能下结论。 | Unknown |

### E.4 如果老师问“这个 1B 到底能不能 defend？”

建议回答：

> It should be defended only as a limitation and diagnostic. We should not defend
> it as support. The correct move is to say we identified it, scoped it, and
> require exact artifacts before promoting it.

## F. 论文文本需要怎么改或怎么说

### F.1 Introduction / Main claim

建议：

> We study training-time RoPE frequency allocation as a mechanism axis
> complementary to inference-time range scaling.

避免：

> EVQ provides a general long-context solution.

### F.2 Experiments

建议：

- Primary I: matched-scale EVQ x YaRN。
- Primary II: PE-dominant diagnostic, seed-scoped。
- Primary III: MLA scarce-channel stress test。

避免：

- 把 supporting rows 写成 primary。
- 把 Table 2 PK 写成 generation ability。
- 把 1B row 放在“scale-up success”旁边。

### F.3 Limitations

必须放进去：

- 1B/4K MLA raw reversal。
- MLA tau convention still needs direct ablation。
- tuned Geo+YaRN/LongRoPE-style baselines missing。
- LoRA/video/progressive are supporting。

如果篇幅很紧，至少写：

> The 1B/4K MLA supporting row shows a raw EVQ reversal under a different
> train-length/data/provenance regime. We therefore do not treat it as a
> same-configuration token-scaling result; it is a limitation and motivates
> artifact-level and same-config follow-up.

## G. 给老师的取舍建议

### G.1 如果 rebuttal 空间很小

保留：

1. Scope correction。
2. Table 2 matched-scale complementarity。
3. MLA 3-seed scarce-channel result。
4. 1B limitation explanation。
5. Artifact/tau ablation follow-up。

删掉或压到一句：

- LoRA。
- video。
- progressive。
- 750M。
- LongBench/Qwen/LLaMA。
- 旧的 5-scale story。

### G.2 如果 rebuttal 空间中等

增加：

- PE-dominant seed-scope clarification。
- PK teacher-forced clarification。
- tuned baseline concession。
- frequency-source code patch / checkpoint audit plan。

### G.3 如果 rebuttal 空间充足

增加：

- 1B frequency-window explanation。
- exact artifact recovery status。
- minimal experiment plan。
- revised claim table: defend / scope / future work。

## H. 老师需要帮忙决定的最终方案

### Option 1: 保守强防守

策略：

- 主动承认所有 P0 scope。
- 把 paper 定位成机制论文。
- 不承诺新实验，只承诺 artifact recovery 和 revised wording。

优点：最稳，不容易被追杀。

缺点：rebuttal 说服力主要靠 claim calibration，不能完全反转 weak reject。

### Option 2: 小补实验 + 强收缩

策略：

- 加 MLA tau ablation 或恢复 1B/primary artifacts。
- 同时收缩 broad claims。

优点：能直接回应 Opus4.8 最硬问题。

缺点：需要时间/算力/外部机器。

### Option 3: 继续强攻 broad evidence

策略：

- 强调 5-scale、LoRA、video、progressive、750M。

不推荐。理由：supporting evidence 质量不均，容易把 rebuttal 变成 reviewer 的靶场。

推荐给老师的方案：Option 2 如果能补 artifact 或 tau ablation；否则 Option 1。

## 1. 老师最需要判断的三件事

### 1.1 是否保住核心贡献？

可以。核心贡献应写成：

- RoPE frequency table 是 finite spectral budget，不只是位置算子的一部分。
- EVQ-Cosh 是 closed-form、zero-learned-parameter 的训练期 allocation。
- EVQ 改变训练得到的 frequency substrate，matched inference-time scaling 在这个 substrate 上作用不同。

不要写成：

- EVQ 是 universal long-context recipe。
- EVQ 替代 YaRN/LongRoPE/LongRoPE2。
- EVQ 在所有 tuned range-scaling baseline 上占优。

### 1.2 最大危险点是什么？

最大危险点不是“没有代码”，而是证据等级混淆：

- 1B/4K MLA 有代码线和 Markdown 报告，但当前 compact repo 没有 exact JSON、checkpoint `inv_freq` hash、data hash。
- 1B/4K 和 primary MLA 8K/500M 不是同配置 longer-training ablation。
- MLA `tau=1.414` 是 empirical `d_eff=128` convention，不是从 released code `head_dim=64` 或 `d_rope=32` 推导出来的定理。
- Table 2 PK 是 teacher-forced NLL-gap，不是 autoregressive exact retrieval。
- PE-dominant Geo/DAPE/EVQ 是 seed 42 diagnostic，不应被讲成完整 3-seed primary。

### 1.3 Rebuttal 应该怎么打？

主线：

1. 主动承认 scope：机制论文，不是 production SOTA。
2. 用 Table 2 defend EVQ x YaRN 的 matched-scale complementarity。
3. 用 8K/500M MLA defend scarce-channel setting 下 allocation matters。
4. 用 PE-dominant result 作为 diagnostic，而不是强统计主证据。
5. 主动解释 1B/4K：不是 same-config，更像 sparse frequency-window failure mode；当前只能 limitation。
6. 明确后续补救：恢复 artifact manifest / 做 tau ablation / 加 tuned scaler 或 rebased-Geo control。

## 2. 可 defend 的主张

| 主张 | 当前证据 | Rebuttal 口径 |
| --- | --- | --- |
| EVQ-Cosh 是训练期 RoPE frequency allocation。 | `scripts/lib/rope/schedules.py`；paper theory；RoPE core tests。 | 作为机制和实现主张 defend。 |
| EVQ x YaRN 是 matched-scale substrate/range complementarity。 | `data/curated/table2_evq_yarn_454m_passkey_10pct.json`；`paper/tables/table2_evq_yarn_main.tex`。 | 只 defend fixed matched YaRN scale；不 claim tuned-scaler dominance。 |
| 8K/500M MLA 是最强系统证据。 | `results/eval_3seeds_full_results.json`；`scripts/core_text_phases/eval_extended_3seeds.py`。 | defend 为 3-seed scarce-channel stress test。 |
| 当前 YaRN eval 代码不会“凭空造 freq”。 | `scripts/core_text_phases/eval_extended_3seeds.py`、`scripts/core_text_phases/yarn_finetune_eval.py` 已强制 checkpoint `inv_freq`、clone loaded buffer、打印 hash。 | defend 当前代码路径；历史 checkpoint 仍需离线审计。 |
| PK metric 已经被正确限定。 | paper/table/docs 已改为 teacher-forced NLL-gap。 | 承认比 AR exact 软；不要把它包装成 exact retrieval。 |

## 3. 必须 concede / scope 的点

| 攻击点 | 正确处理 |
| --- | --- |
| 1B/4K MLA raw EVQ reversal | 主动承认是真 limitation，不隐藏。 |
| 1B 是否证明 longer training 下 EVQ 失效 | 不这么解读。它不是 same-config token-scaling ablation。 |
| 1B 是否可作为支持证据 | 当前不行。只能 code-backed/report-backed limitation。 |
| MLA tau convention | 承认 `tau=1.414` 是 empirical convention；需要 direct ablation 才能更强 defend。 |
| tuned Geo+YaRN / LongRoPE-style baseline | 承认没有证明 tuned baseline dominance。 |
| LoRA 8B | supporting only，缺 matched Geo+LoRA。 |
| video / progressive / 750M | supporting only，不作为主证据。 |
| PE-dominant primary | diagnostic and seed-scoped，不能过度统计化。 |

## 4. 1B/4K MLA 的最终说法

### 4.1 当前证据状态

| 层级 | 状态 |
| --- | --- |
| 训练/eval 代码 | 存在。训练 launcher 调 shared MLA entrypoint；YaRN+FT eval script 会写 JSON。 |
| Markdown report | 存在：`results/PHASE18_YARN_FT_REPORT.md`。 |
| exact baseline JSON | 当前 compact repo 缺失。 |
| exact YaRN+FT JSON | 当前 compact repo 缺失。 |
| checkpoint `inv_freq` hash | 当前 compact repo 缺失。 |
| data hash / cache manifest | 当前 compact repo 缺失。 |
| same-config token scaling | 不成立。 |

### 4.2 为什么不是 same-config longer-training ablation？

当前审计认为它至少在这些字段上和 primary MLA 不一致或未证明一致：

- train length：primary MLA 是 8K/500M；1B row 是 4K/1B。
- seed：primary MLA 是 3 seeds；1B row 当前报告 seed 42。
- artifact completeness：primary 有 compact result JSON；1B 只有 Markdown report。
- data/provenance：1B data/cache hash 不在 compact repo。
- sparse window：1B row 使用旧 MLA-32/K16/base500K sparse frequency substrate，4K 到 8K 的窗口更容易暴露 early extrapolation failure。

### 4.3 Rebuttal 可用短答

> We do not hide the 1B/4K MLA row; we treat it as the main limitation and a root-cause target. It is code-backed and report-backed, but not compact-JSON/checkpoint/data-hash backed. It is also not a same-configuration continuation of the primary 8K/500M MLA experiment. Therefore we do not use it as primary support. Its pattern is consistent with a sparse frequency-window failure mode: raw EVQ can improve in-window PPL but fail early extrapolation when too few channels cover the 4K-to-8K bridge.

### 4.4 如果想把 1B 从 limitation 升级为 reviewer-grade row

必须先恢复：

- exact result JSON files such as `results.json`
- exact YaRN+FT JSON files such as `yarn_ft_s*_seed*_results.json`
- checkpoint SHA256
- checkpoint-loaded `inv_freq` SHA256
- train/eval data cache hash
- seed list and command/config notes

恢复路径见 `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md`。

## 5. P0/P1 问题总表

| ID | 问题 | 当前状态 | Rebuttal 动作 |
| --- | --- | --- | --- |
| O48-06 | PK 是 teacher-forced NLL-gap，不是 AR exact。 | wording 已收紧。 | 主动承认 metric scope。 |
| O48-09 | PE-dominant Geo/DAPE/EVQ seed 42 only。 | wording 已收紧。 | 不讲成全 3-seed。 |
| O48-11 | MLA tau convention 和 code field 曾经不一致。 | wording 已修正。 | 说 empirical `d_eff=128` convention。 |
| O48-12 | 缺 direct MLA `tau=d_rope/sqrt(L)` ablation。 | open。 | 承认或补实验。 |
| O48-13 | 1B raw EVQ reversal。 | open limitation。 | 主动承认，不当支持证据。 |
| O48-14 | 1B provenance 不够 reviewer-grade。 | evidence-gated。 | 恢复 exact artifacts 或降级。 |
| O48-16 | LoRA 缺 Geo+LoRA control。 | open。 | supporting only。 |
| O48-22 | 缺 tuned LongRoPE2/CoPE/tuned YaRN baselines。 | experiment-gated。 | 承认 baseline gap。 |
| O48-24 | Rebuttal playbook 曾经过度进攻。 | 已收紧。 | 使用新版，不粘贴旧话术。 |
| O48-25 | script/artifact drift。 | partly fixed。 | 用 manifest，不直接引用历史 wrapper。 |

其余 P1/P2 问题可以在 rebuttal 中按需回答，不必主动展开：constant-alpha modeling choice、LoRA phase-transition phenomenology、video base sensitivity、progressive single-seed、downstream accuracy limited、appendix-heavy 等。

## 6. 可以不要主动放进 rebuttal 的内容

除非 reviewer 点名，否则不建议主动展开：

- LoRA phase-transition theory。
- video DiT 大量 supporting tables。
- progressive training 的 -81.2% 单 seed 数字。
- 750M AR exact 单 seed 数字。
- LongBench/Qwen/LLaMA supporting rows。
- 旧 docs/exp 中早期 “paper-ready” 或 “recipe” 口吻。

理由：这些内容容易把 rebuttal 从主线机制证据拉到 single-seed/supporting evidence 上，给 reviewer 更多攻击面。

## 7. 如果 rebuttal 只能补一件事

优先级最高：恢复 artifact manifest。

低成本最有效：

1. 在外部机器上对 1B 和 primary MLA run dirs 跑 `scripts/core_text_phases/make_artifact_manifest.py`。
2. 对 exact checkpoint 跑 `scripts/core_text_phases/audit_rope_checkpoint.py`。
3. 对 train/eval cache 跑 `scripts/core_text_phases/audit_training_artifacts.py`。
4. 把 sanitized JSON manifest 加入 repo，并更新 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`。

这不能直接证明 EVQ 更强，但能把“代码/报告线索”升级成 reviewer 可接受的 provenance。

## 8. 如果 rebuttal 能补实验

低成本优先：

| 实验 | 目的 |
| --- | --- |
| MLA `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablation | 回答最强 MLA tau convention attack。 |
| Rebased-Geo / fixed-interpolation training-time control | 回答“只是简单低频 schedule”攻击。 |
| Tuned Geo+YaRN control | 回答 matched-scale vs tuned-scaler attack。 |
| PE-dominant extra seeds | 加强 Primary II。 |
| Geo+LoRA matched control | 让 LoRA 从 anecdote 变成可用 supporting evidence。 |

理想但更贵：

- Same-config 8K MLA token-scaling scan：500M、1B、2B checkpoints，Geo/EVQ，same data/seeds/eval。
- 目标问题：EVQ 是稳定 spectral allocation advantage，还是 early-training optimization advantage。

## 9. 建议 rebuttal 总段落

可作为 rebuttal 开头或老师讨论草稿：

> We thank the reviewer for pressing on scope and provenance. Our intended claim is not that EVQ-Cosh is a universal long-context recipe or a replacement for range scaling. The claim is that RoPE frequency allocation is a finite spectral-budget design axis, and that a closed-form training-time allocation changes the substrate on which matched inference-time scaling acts. We have therefore revised the framing around three evidence tiers: the 454M matched-scale EVQ x YaRN result, the seed-scoped PE-dominant diagnostic, and the 3-seed 8K/500M MLA scarce-channel stress test. We explicitly distinguish teacher-forced NLL-gap PK from autoregressive exact retrieval, and we do not use supporting LoRA/video/progressive rows as primary evidence.
>
> We also agree that the 1B/4K MLA row is the most important limitation. It is not a same-configuration continuation of the 8K/500M MLA result, and in the compact repository it is code-backed/report-backed but not exact-JSON/checkpoint/data-hash backed. We therefore treat it as a root-cause target rather than primary support. The correct interpretation is narrower: EVQ can fail in a sparse 4K frequency-window regime, while the primary matched-scale rows still support the mechanism that training-time allocation changes how inference-time scaling acts on the learned frequency substrate.

## 9.1 可直接改写的分问题回应

### Q1: Does the 1B reversal invalidate the paper?

Draft:

> It is a real limitation, but not a same-configuration refutation of the primary
> result. The 1B/4K row differs from the 8K/500M MLA stress test in train length,
> artifact completeness, and seed/provenance status. We therefore do not use it
> as primary support. Instead, we interpret it as evidence that sparse
> frequency-window regimes can make raw EVQ fail, which is compatible with the
> finite-spectral-budget view and motivates a same-config token-scaling follow-up.

### Q2: Is EVQ just helping YaRN rather than helping by itself?

Draft:

> Our claim is exactly about the substrate on which range scaling acts. EVQ is a
> training-time allocation; YaRN is an inference-time scaling rule. In the
> matched-scale rows, YaRN has higher leverage on the EVQ-trained substrate. We
> do not claim this proves dominance over tuned Geo+YaRN or every range-scaling
> baseline.

### Q3: Is PK too soft?

Draft:

> Yes, PK in Table 2 is teacher-forced NLL-gap retrieval, not autoregressive exact
> generation. We have clarified this and use it as a PE diagnostic signal rather
> than as a full task-success claim.

### Q4: Why use `tau=1.414` in MLA?

Draft:

> In MLA we treat `tau=1.414` as an empirical operating convention corresponding
> to an effective `d_eff=128`, not as a theorem forced by the released code fields
> `head_dim=64` or `d_rope=32`. We agree that direct ablations with
> `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` are the right sanity checks.

### Q5: Why no tuned LongRoPE/Geo+YaRN baseline?

Draft:

> The current evidence is matched-scale rather than tuned-scaler dominance. We
> scope the claim accordingly: EVQ changes the training-time substrate, and under
> the tested fixed scaling rule that substrate gives higher leverage. Tuned
> Geo+YaRN/LongRoPE-style controls are important future or rebuttal additions.

### Q6: What can be removed from the rebuttal if space is tight?

Remove or compress:

- LoRA phase-transition explanation.
- video DiT detail tables.
- progressive training chain.
- 750M single-seed AR exact.
- broad cross-model transfer claims.

Keep:

- mechanism claim,
- matched-scale EVQ x YaRN,
- MLA 3-seed scarce-channel result,
- 1B limitation/scope,
- MLA tau and tuned-baseline concessions.

## 9.2 Suggested One-Paragraph Abstract For The Rebuttal

> We thank the reviewer for identifying places where our evidence should be more
> carefully scoped. We have revised the framing to emphasize that EVQ-Cosh is a
> mechanism study of training-time RoPE frequency allocation, not a universal
> long-context recipe or a replacement for inference-time scaling. The primary
> evidence supports a narrower claim: a closed-form allocation changes the
> learned frequency substrate, and matched range scaling can act more effectively
> on that substrate. We now explicitly distinguish teacher-forced NLL-gap
> passkey evaluation from autoregressive exact retrieval, treat the PE-dominant
> result as seed-scoped diagnostic evidence, and identify the 1B/4K MLA reversal
> as a limitation rather than primary support. The 1B row is not a
> same-configuration continuation of the 8K/500M MLA result and lacks compact
> exact-JSON/checkpoint/data-hash provenance, so our rebuttal does not rely on it
> as positive evidence.

## 10. Final Recommendation

给老师的决策建议：

- 可以继续投/回，但必须按机制论文 defend。
- Rebuttal 不要试图把 1B 说成支持；要主动承认并解释为什么不是 same-config。
- 如果有时间，优先恢复 artifacts，其次做 MLA tau ablation。
- 如果没有时间，至少确保论文和 rebuttal 中所有强 claim 都降级到当前证据能支撑的范围。

当前文档层已经可交付；实验层仍有 P0 gate，不能在 rebuttal 中假装已经解决。
