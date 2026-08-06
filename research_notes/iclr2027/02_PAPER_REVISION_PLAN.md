# 论文修改计划(条件启动:2026-09-24 NeurIPS 决定日分支)

创建:2026-08-06 · 修订:2026-08-06(改为条件启动)· 状态:`PLANNING_ONLY`
基线稿:`paper/`(NeurIPS 11628 提交版,只读,一字不动)。

## 0. 启动条件(先于一切)

- **D = 2026-09-24(NeurIPS 作者通知,AoE)。D 之前:不创建新稿目录、不写任何正文、不向任何 venue 提交。**
- **Accept 分支**:本文件降级为 camera-ready 增强清单 —— 只执行不改论文结构的行(§2 中 FMRoPE related-work 补充与对照表、RULER 补表、basin 图、机制章图),遵守 NeurIPS camera-ready 规则。
- **Reject 分支**:启动完整转投。目标 venue 默认 **ICML 2027**(未官宣,第三方估计 2027-01 中下旬,以官网为准);ICLR 2027 不可行的日历原因见 00 §1。新稿目录届时创建,命名 `paper_icml2027/`(或按最终 venue)。
- §1–§4 的内容 venue 无关,两分支共用;§5 里程碑以 D-相对周表示。

---

## 1. 重定位

**旧论点(NeurIPS 版)**:RoPE 频率表是独立设计轴;EVQ-Cosh 是闭式零参数分配,τ=d_eff/√L 操作规则;PPL/TF-passkey 上有收益,与 YaRN 互补,MLA 稀通道下更重要。

**新论点(下一版,venue 无关)**:

> 训练期频率分配是一个**基底(substrate)**:它决定模型能学到什么样的长程寻址回路。我们给出 (i) 因果机制链证据 —— 分配改变 QK 寻址与远端信息的因果使用,并在匹配训练下转化为 native 学不出的 2× 检索能力;(ii) 权衡结构 —— 外推失效的 aliasing/OOD-phase 二分 + in-window 成本二分(分配内禀代价小,换表适配失配大);(iii) 由此导出的**保护性部分频谱替换**,在成熟模型微调中保持 in-window 能力的同时获得外推收益。

三个支柱对应三类已有/计划证据:
- 支柱 i:OLMo matched routing(0/100 vs 69/100 & 67/100,双 EVQ seed)、8B causal suite(gold-drop +1.51 vs ≈0、swap、oracle-null)、750M continuation(0%→77.5% AR)—— 全部已完成。
- 支柱 ii:LeRoPE note §7 的二分框架 + P1-4 分离度量 + §11 interference-pruning —— 框架已有,关键图待 P1-4。
- 支柱 iii:P-EVQ(P1-1)+ scratch 无损变体(P1-2)—— 待跑,是新版最大增量。

**先写後跑的保险**:若 P1-1/P1-2 失败或不及预期,支柱 iii 降级为"负结果 + 开放问题",论文仍以支柱 i+ii 成立(mechanism-clarification 主线不依赖新方法成功)。

## 2. 审稿关切 → 修改动作映射

| 关切 ID | 内容 | 新版动作 | 依赖实验 |
| --- | --- | --- | --- |
| `AC.1` `RzWsa.1` `RzWsa.2` | FMRoPE/dead-channel 新颖性;要 matched 对照 | Related work 正面引用 Oka et al. + 对照表;正文报双向结果:matched exact-range 内部分配 Cosh 优(−0.32/−0.20/−0.17),target-aware range transport FMRoPE 更强(G-FMR-DEPLOY);定位为"allocation shape ⊥ range control 两个旋钮" | P2-3(owner 升级) |
| `AC.2` `RzWsa.4` `RDz6s.1` `R27bE.2` | 规模小、诊断偏重、要 1B–7B + 真实基准 | 成熟模型证据前置成主实验:OLMo-2-1B(RULER 家族 + NIAH 严格 AR,双 seed)、LLaMA-3-8B(matched LoRA + RULER 16K);明确 endpoint 分类学(NLL/TF/AR/downstream 分层呈现) | 已有;P2-1 增强 |
| `RzWsa.3` | 要 RULER | RULER 全矩阵进正文(含 32K 全零负结果与 held-out task 零分,按 boundary 呈现) | 已有 |
| `AC.3` `RDz6s.3` `R27bE.1` | 理论链只部分验证;surrogate/cosh/τ 混层 | 理论章三层分离(exact 定理 / surrogate 检验 / trained 结果)保留并强化;新增 aliasing/OOD 二分作为统一框架;τ 规则降级为"basin selector + 实测 basin 图"(含右边界) | P1-3、P1-4 |
| `R27bE.4` | 独立调 τ + matched-τ 非 cosh schedule | basin 补全 + 已有 native-attention-shape 多 seed 套件(`native_attention_shape_l128_results_20260724.json`)+ Cosh 非普适负结果如实报(G-COSH-NOT-UNIVERSAL) | P1-3 |
| `R27bE.3` | DAPE 混淆 allocation 与参数化 | 修正方法身份表述;fixed-schedule same-operator 对照为主;引 LeRoPE Fixed-63.6% 作为外部独立证据("收益主要在表,不在学习过程") | 已有 + 引用 |
| `R27bE.5` | held-out base + 更大规模预注册 run | 若 P2-2 完成则给 base/head_dim 小 factorial;否则 limitation 直说 | P2-2 |
| `AC.4` | 什么能改变决定 | 摘要与 intro 直接以机制链+能力转换开题;负结果表贴主结果旁 | — |

## 3. 逐章计划

| 章节 | 动作 |
| --- | --- |
| Title/Abstract | 重写:substrate + mechanism + protected retrofit;放弃"closed-form allocation"作为主卖点(降为构造手段) |
| 1 Intro | 三轴分类扩为四类:positional operator / inference-time range scaling / **training-time allocation(解析)** / **learned-searched frequency(LeRoPE、LongRoPE)**;贡献列表按三支柱重写 |
| 2 Related work | 新增 FMRoPE(Oka et al. ICLR 2026,两篇)、LeRoPE、p-RoPE、DAPE、retrieval-head 文献;给"允许说/禁止说"对照(见 §4) |
| 3 Theory | 保留:\(\mathcal{C}_{\mathrm{app}}\) 定理 + waterbed + 收敛检验(24–92% collision 降低);新增:aliasing/OOD-phase 二分(把 EVQ/p-RoPE/LeRoPE/FMRoPE 四法安置进同一框架);新增:in-window 成本二分(+0.0381 scratch vs 适配失配)与约束分配(03 号文档 T2);τ 小节改写为 operating rule + 完整 basin |
| 4 Method | EVQ-Cosh 构造(不变)+ **P-EVQ**:importance 诊断(pair-ablation KL)、保护集选择、restoration 目标(若 P1-1 成功;否则本节只保留诊断作为分析工具) |
| 5 Experiments | 重排:5.1 机制链(P0-1/P0-3 图 + gold-drop/swap/oracle 表)→ 5.2 能力转换(OLMo 双 seed、750M、8B RULER)→ 5.3 scratch 多 seed(Primary I–III 收缩为验证性小节)→ 5.4 P-EVQ 保 in-window 主表 → 5.5 FMRoPE 定位 → 5.6 负结果与边界(32K 全零、held-out 零、in-window 代价、gap 结构、τ 可错) |
| 6 Limitations | 单 seed 机制证据、task-family adaptation ≠ 通用能力、4× 未解决、c(Π) 经验标定 —— 全部保留并前移 |
| Appendix | 保留 surrogate 验证/mechanism-isolation;新增 trace 分解细节、P-EVQ gates 全文、per-seed 表 |

**砍/降级**:视频 DiT、QuALITY、LongBench NLL 等 supporting 全部压缩进附录一表;TF-passkey 不再作为"retrieval"主指标(AR exact 前置,TF 只作机制读数并明确标注)。

## 4. Claims 纪律(沿用 AGENTS §4 + 新增)

**保留**:有限谱预算框架;分配为独立设计轴;与 YaRN 互补(matched-scale 语气);MLA 稀通道敏感性(附 post-sub K=8 负结果毗邻)。

**降级/改写**:
- "EVQ 改善长上下文" → "EVQ 改变可学回路;能力需 exploit-training,且距离受限(gap 结构)";
- τ 规则 → basin selector,c(Π) 经验;
- passkey → AR exact 为能力口径,TF-NLL-gap 为机制口径,永不混用。

**新增(须实验落地后才可写)**:
- P-EVQ 保持 in-window(gate 全过才可称);
- aliasing/OOD 二分(P1-4 两曲线分离才升为"结果",否则为"框架/动机")。

**禁止**:EVQ 表 ≈ LeRoPE 学出表;LeRoPE 验证了 EVQ 外推;比 FMRoPE/LeRoPE 好(无共同基准);P-EVQ = zero-parameter pure EVQ-Cosh;高 attention 权重 = 因果使用(用 gold-drop 语言);NLL 改善 = 能力(dissociation 证据太强)。

## 5. 写作里程碑(D-相对;仅 Reject 分支启动)

前提:实验与素材(图、证据总表、related-work 笔记)已按 01 在 D 前备好;写作期不新开实验,只允许收尾中的 P2 项滚入。

| 阶段 | 交付 |
| --- | --- |
| D+0–1 周 | 转投决定确认;创建 `paper_icml2027/`(从 `paper/` 拷出为起点,源树只读);骨架与 related work 定稿 |
| D+1–3 周 | 理论章(二分框架 + 成本二分)与机制章初稿 |
| D+3–6 周 | 实验章 5.1–5.6 + limitations 全稿 v1 |
| D+6–8 周 | send-gate 式自查(每数字有 owner/protocol/seed 口径;matched 声明核对;负结果毗邻)+ 内审 |
| D+8 周–截止前 | 精修、图表、多轮打磨(ICML 估计截止 2027-01 中下旬,窗口充裕) |

Accept 分支:按 NeurIPS camera-ready 官方截止执行 §0 所列子集,不套用本表。

匿名与卫生:沿用现有规则(无内部路径/机器名/身份);supplement 用 `scripts/package_supplement.py`,不打包仓库根。
