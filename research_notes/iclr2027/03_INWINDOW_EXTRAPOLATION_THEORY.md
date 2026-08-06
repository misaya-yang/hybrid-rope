# 核心问题:保持 in-window 的同时加大外推 —— 理论缺口与补全路线

创建:2026-08-06 · 状态:`HYPOTHESIS_AND_PLAN`(本文档所有"预测"均未验证;所有"事实"均给 owner)
性质:venue 无关的研究核心文档,不依赖 2026-09-24 NeurIPS 决定结果

---

## 1. 问题陈述

当前微调(retrofit)实验的实然状态:**外推收益和 in-window 保持是二选一**。

| 观测 | 数值 | Owner |
| --- | --- | --- |
| OLMo-2-1B full-EVQ 换表 LoRA:8K NIAH 严格 AR | Native 0/100 → EVQ 69/100、67/100(双 seed) | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| 同一设置的 in-window 代价:4K RULER macro | 82.16 → 37.51 | 同上系列(`LEROPE_CONCURRENT_WORK_NOTE` §7.0 引用) |
| 4K held-out UUID 任务 | 55% → 0% | 审计 §2 link5 |
| LLaMA-8B matched LoRA @8K RULER | Native-LoRA 94.4% vs EVQ 77.6% | `LLAMA8B_RULER_16K_*`(审计 §1.3) |
| selective Q/K phase adaptation:2Wiki 4K 已拉平 | 22.0%/21.5%(Native/EVQ),但 RULER 4K 仍 72.19/42.44 | `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` |

理论缺口:我们尚未给出一个**能同时解释这两侧、并指出何时可以两者兼得**的表述。本文档给出三层理论升级(T1–T3)、可证伪预测(P1–P6)与判定树,把"保 in-window + 增外推"从愿望变成有 gate 的实验命题。

## 2. 已验证的事实基座(全部有 owner,分层引用)

1. **成本二分**:in-window 代价是两个不同的东西——
   - *分配内禀代价*:从头训 OLMo-2 1.485B @4K NLL +0.0381(EVQ vs Geo)。量级小,与 τ=2 下 L-尺度频带仅 0.79× 的轻度抽稀一致。
   - *换表适配失配*:成熟 Geo 预训模型直接换 EVQ 表后 RULER 崩到 37.51。量级大,且**不是分配的内禀属性**,而是权重-频率 co-adaptation 被破坏。
   Owner:`LEROPE_CONCURRENT_WORK_NOTE_20260728.md` §7.0(撤回 B)。
2. **Co-adaptation 主导**:运行时把 EVQ 频率注入 native 权重 → 全灭;把 Geo 频率喂给 EVQ-adapted 权重 → 部分保留。收益走"频率改变可学回路"路径,不是运行时几何。Owner:审计 §3-A(counterfactual swap,8B)。
3. **障碍事实**:静态位置无关的 Q/K 线性映射不能在所有距离上精确共轭两个不同 RoPE 生成元(63/64 对频率同时改变)。Owner:`OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`。
4. **删除可以变好**:held-out interference 剪枝——native-Geo 删 16 个干扰对后 8K NLL 2.827→2.336;EVQ 也有 18 个干扰对。说明存在"对 in-window 近乎无贡献、对长程有害"的通道子集。Owner:151.9M `EXPERIMENT_REPORT.md` §11.2(审计 §3-B)。
5. **Waterbed 约束**:纯重分配在 surrogate 意义下不可能处处占优(`paper/sections/03_theory.tex` waterbed 命题)。任何"无损"主张必须绕开它,而不是违反它。
6. **集中性外证**:LeRoPE leave-one-out——dominant band 0.762 nats vs 次名 0.069 nats(11×)。位置功能可以高度集中于少数频带,这使"保护少数重要通道"有先验合理性。Owner:`LEROPE_CONCURRENT_WORK_NOTE_20260728.md` §4。
7. **外推失效二分**(框架,待经验分离):λ≪d 的通道"行为良好但有歧义"(aliasing),λ≳L_train 的通道"无歧义但相位未受训"(OOD phase)。EVQ/p-RoPE/LeRoPE/FMRoPE 可在此框架内统一安置。Owner:同上 §7.1。

## 3. 理论升级

### T1 — 微调側:成本分解与保护原理

把换表后的 in-window 损失分解为

\[
\Delta\mathcal{L}_{\mathrm{in}}(\rho') \;=\;
\underbrace{\Delta\mathcal{L}_{\mathrm{alloc}}(\rho')}_{\text{分配内禀,受 waterbed 约束,实测 } \sim 0.038}
\;+\;
\underbrace{\Delta\mathcal{L}_{\mathrm{adapt}}(\rho' ,\theta_{\mathrm{pre}})}_{\text{co-adaptation 失配,实测巨大}}
\]

事实 3(障碍)⟹ \(\Delta\mathcal{L}_{\mathrm{adapt}}\) 不能被任何静态 Q/K 线性适配子精确消为零——**除了在频率不变的坐标上**:若通道对 \(p\) 的频率保持 native 值且 LoRA 在该坐标直接输出为零,则该坐标的失配严格为 0(构造性保证,已实现于 `native_protected_evq.py` 的 mask)。

**保护原理(命题级,待 P1-1 检验)**:若 native 模型的 in-window 位置功能集中于小的通道子集 \(F\)(H-PROTECT),则"保护 \(F\) 的频率 + 只在补集上重分配"能把 \(\Delta\mathcal{L}_{\mathrm{adapt}}\) 压到修复训练可闭合的量级,同时把补集(以冗余慢通道为主,事实 4)交给 EVQ 换取外推收益。

这不与 waterbed 冲突:waterbed 约束的是 surrogate 全局;保护方案赚的是 \(\Delta\mathcal{L}_{\mathrm{adapt}}\)(waterbed 之外)加上补集里"对 in-window 近似常数的重复慢通道"(其 in-window 边际贡献 ≈ 0,事实 4、Green 核 \(\min(\varphi,\psi)\) 论证)。

### T2 — Scratch 側:约束分配(无损分配的形式化)

把"保 in-window + 增外推"写成约束变分问题:

\[
\min_{\rho}\; \mathcal{C}_{\mathrm{ext}}(\rho; [L_{\mathrm{train}}, L_{\mathrm{tgt}}])
\quad \text{s.t.}\quad
\mathcal{C}_{\mathrm{in}}(\rho; [0, L_{\mathrm{train}}]) \le (1+\varepsilon)\,\mathcal{C}_{\mathrm{in}}(\rho_{\mathrm{geo}}),
\]

其中 \(\mathcal{C}_{\mathrm{ext}}\) 由二分框架给出双项结构(aliasing 项 + OOD-phase 项,事实 7)。预期解族是 **two-regime 分配**:in-window 关键尺度上保留(或钉住)通道,其余预算按 cosh 尾部压向"已受训相位"侧。两个闭式候选(均零参数、只依赖 \(L_{\mathrm{train}}\),不引入 FMRoPE 式 \(L_{\mathrm{tgt}}\) 依赖):

- **pinned-λ\***:1 个通道解析钉在 \(\lambda = c\cdot L_{\mathrm{train}}\)(c 由 P0-2 标定,LeRoPE 的 2.205 只是起点),其余 \(K-1\) 通道 cosh;
- **protected-band two-regime**:保留几何表在 \([\lambda_1,\lambda_2]\ni O(L_{\mathrm{train}})\) 区间的通道,补集在剩余 \(\varphi\) 测度上做 sub-budget cosh 分位构造 \(\varphi_j = 1-\operatorname{arcsinh}((1-u_j)\sinh\tau')/\tau'\)。

ε→∞ 退化为 EVQ-Cosh;保护集=全体退化为 geometric——EVQ 与 native 成为同一族的两个端点,论文叙事因此闭合。

### T3 — 外推目标的结构:aliasing vs OOD-phase

P1-4 若把两条消融曲线分开(λ≪d 组 vs λ≳L_train 组随 d 的边际贡献走向不同),则 \(\mathcal{C}_{\mathrm{ext}}\) 的双项结构获得直接经验支撑,且四方法(EVQ / p-RoPE / LeRoPE / FMRoPE)在同一图上各居一角——这张图是新版理论章的核心图。若不分离,T2 仍可用(退回 collision 叙事),但理论章语气收缩。

## 4. 可证伪预测与 gates(全部先注册后观测)

| # | 预测 | 检验 | 通过线 | 失败含义 |
| --- | --- | --- | --- | --- |
| P1 | H-PROTECT:native 4K 注意力功能集中于 ≤16/64 对,split 稳定 | P1-1 Stage D(已注册五条判据,不得改) | 注册文档 §4 原文 | 保护假设死亡,走 restoration-only 或放弃 |
| P2 | 保护因果性:protected 臂过 4K gates 而 matched full-EVQ control 不过 | P1-1 Stage G/R + control | 注册文档 §7 gates(2Wiki −5pt、RULER macro −10pt、NLL +0.10、MCQA 无崩) | 两臂都过 → restoration objective 足够,保护非必要(也是可发表结论) |
| P3 | 外推保留:protected 臂 8K NIAH 严格 AR ≥ full-EVQ 转换量的 50%(即 ≥ ~35/100,native=0 基线) | P1-1 冻结 8K/16K 评测 | ≥50%(**已注册阈值,2026-08-06 定稿,作者委托代定**) | 保护把外推也保没了 → 保护集过大/错位,报告为 trade-off 曲线上的一点 |
| P4 | scratch 无损:pinned/two-regime 消除 ≥2/3 的 +0.0381 in-window 差距,同时保住 ≥80% 的 2× 外推 NLL 收益 | P1-2(3 seeds) | 两条件同时成立 | 内禀代价不可分配性消除 → waterbed 在任务损失层也咬合,论文如实报 |
| P5 | λ* 对准:21 配置的经验 τ 选择与 λ* 接近度秩相关 | P0-2 | Spearman 显著且方向一致(阈值在脚本里先写死) | 排除"τ 摆动=对准 λ*"解释,c(Π) 维持纯经验 |
| P6 | 二分分离:aliasing 组与 OOD 组消融曲线随 d 走向不同 | P1-4 | 方向性分离 + bootstrap 区间不重叠(先注册) | 二分降级为动机性叙述 |

## 5. 微调协议(不重复注册文档,只给指针与角色)

执行严格按 `OLMO2_NATIVE_IMPORTANCE_PROTECTED_EVQ_HYPOTHESIS_20260728.md` §4–§9:
Stage D 诊断(exact pair-ablation KL,无训练)→ Stage E0 免训练筛(2Wiki 50 行 / RULER 13×5 / NLL 64 行)→ 一次 144-step masked-Q/K-LoRA restoration(全层 attention KL + A@V MSE,无 CE)→ §7 能力 gates → 冻结后 8K/16K。护栏:阈值先注册不得改;所有频率张量独立 clone + hash(`I-HYBRID-ALIAS` 教训);protected 臂不过 4K gate 则不启动 control 臂。

角色分工:P1-1 回答"微调侧能否两者兼得";P1-2 回答"scratch 侧内禀代价能否设计掉";两者共用 T1/T2 叙事但**证据独立,互不背书**。

## 6. 判定树(每个出口都可写进论文)

```
Stage D 诊断
├─ 失败 → 保护假设死亡;selective-QK(已有)为微调侧最好结果;
│         论文按"部分恢复 + 开放问题"写,支柱iii 降级        [出口A]
└─ 通过 → E0 免训练筛
    ├─ 直接过 4K gates → 跳过训练,冻结评 8K/16K            [出口B,最便宜]
    └─ 不过 → 144-step restoration
        ├─ 4K gates 不过 → 保护+修复不足,分布式交互主导;
        │                   报告为障碍加强                    [出口C]
        └─ 过 → 跑 matched full-EVQ control
            ├─ control 不过 → 保护是因果的(P2 成立)
            │   ├─ 8K/16K 外推保留(P3)→ 完整正结果:
            │   │   "保 in-window + 增外推"成立(2× 域内)      [出口D,最优]
            │   └─ 外推没保住 → trade-off 曲线上的新点        [出口E]
            └─ control 也过 → restoration objective 足够,
                              保护非必要;方法简化             [出口F]
```

出口 A/C/E/F 均为可发表内容(机制主线不受损);只有出口 D 允许在摘要级写"同时保持与提升"。

## 7. 边界与禁语(对齐 AGENTS §4 与注册文档 §10)

- 一切主张限 **2× 域内**(4K→8K;8B 侧 8K→16K 部分);32K/4× 负结果必须毗邻呈现(G-2X-CEILING、32K RULER 全零)。
- P-EVQ 成功后的名称:*Native-importance-protected hybrid EVQ retrofit*;禁称 pure EVQ-Cosh、zero-parameter、function-preserving by construction。
- 不得声称验证 2.205L 常数(c 是待标定量);不得写 LeRoPE/FMRoPE 可比性结论(无共同基准)。
- NLL 改善永不当能力证据(dissociation:8B 32K PPL 3.71 vs 5.35 同时 RULER 0/20 的先例)。
- task-family adaptation(LongAlign→NIAH/RULER 家族)与 unseen-task transfer 分开陈述;held-out UUID 0% 与 niah_single_2 0.70→0.55 回归必须保留在正文或紧邻附录。
