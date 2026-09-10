# digest paper-state

任务：读论文当前状态（文本源，不读大 PDF），输出论文当前声明与 09-10 统一预算理论的关系——哪些新理论内容可以并入、哪些必须等 holdout（0450/0451）；统一方案若发表，最小改动路径是什么。

日期基准：2026-09-10。当前分支 09_09，HEAD `a6e3aba`（Workspace snapshot 2026-09-10 07:49），工作树除 `analysis/` 外干净。

证据等级标记规则：[已验证]=本轮直接读到文件/git/哈希级证据；[部分证据]=有读数但范围或复现受限（开发面板、单次、代理指标）；[假设]=机制解释或未完成计划。

---

## 1. 来源清单（文件路径、大小、行数）

### 论文文本源（paper-2027/，全部为本轮直接 Read）

| 文件 | 大小 (B) | 行数 | mtime | 说明 |
|---|---:|---:|---|---|
| paper-2027/HANDOFF.md | 1,903 | 31 | 09-09 01:09 | 唯一"live state"文件（自身已显旧，见 §2-G） |
| paper-2027/REVISION_BRIEF.md | 1,732 | 30 | 09-09 01:28 | 09 重构周期作者批准范围 |
| paper-2027/main.tex | 3,763 | 124 | 09-09 00:32 | ICLR 2027 模板；标题 `Beyond the Base: Exponent Allocation in RoPE`；venue 注释：abstract 截止 2026-09-18 AoE、全文 2026-09-25 AoE、正文 ≤9 页硬约束（compile.sh 强制） |
| paper-2027/sections/00_abstract.tex | 1,258 | 16 | 09-09 23:30 | 摘要全文（数字见 §2-E） |
| paper-2027/sections/01_intro.tex | 3,708 | 63 | 09-09 23:28 | 三问 + Contributions(i)(ii)(iii) |
| paper-2027/sections/02_exponents.tex | 1,887 | 35 | 09-09 01:06 | 定义段：ω=b^−φ、x=a+Rz 分解 (eq:table-decomposition) |
| paper-2027/sections/02_identification.tex | 2,175 | 38 | 09-09 00:32 | 151.9M 固定范围 + retarget 反转 + 50.9M factorial |
| paper-2027/sections/03_findings.tex | 1,729 | 34 | 09-09 23:30 | 权重×表交叉（7.14→76.20 / 7.16→23.05） |
| paper-2027/sections/03_theory.tex | 5,743 | 125 | 09-09 23:30 | 主定理段：子空间/c、r2 恒等式、Prop collapse、C_app、Thm Cosh、warp 公式 |
| paper-2027/sections/04_experiments.tex | 4,546 | 94 | 09-09 23:30 | Cosh 训练/适配：432M MLA、750M、8B LoRA、RULER 续训、454M 组合 |
| paper-2027/sections/04_mature.tex | 7,126 | 161 | 09-09 01:52 | 冻结段：d_k 坐标、固定范围 RULER、静态表、BM vs MrPro（128K 数字） |
| paper-2027/sections/02_related.tex | 2,868 | 50 | 09-08 23:11 | 相关工作（已含 LeRoPE、Data-Shapes、CoPE 引用） |
| paper-2027/sections/05_discussion.tex | 1,304 | 23 | 09-08 23:58 | 结论 + sparse attention 方向 |
| paper-2027/sections/budget_{intro,theory,method,experiments,related,abstract,discussion}.tex | 共 ~16.3 KB | — | 09-08 21:12 | **未被 main.tex 引用**的"频段预算"框架残段（budget_theory.tex 87 行含 prop:finite-budget 有限预算命题）——并入统一理论的原材料 |
| paper-2027/appendix/a1_proofs.tex | 27,434 | 561 | 09-09 01:52 | 全部证明：cross-Gram 闭式、budget identity、collapse 四阶系数 19/12600、parity-lattice、两静态反例、移植障碍定理、Cosh ODE、self-consistency、τ 参考规则（99 runs）、离散传输界 |
| paper-2027/appendix/a2_experiment_details.tex | 9,496 | — | 09-09 01:52 | 协议总表（含 1.485B 命名） |
| paper-2027/appendix/a5_identification.tex | 11,822 | — | 09-09 01:52 | 识别实验细节 |
| paper-2027/appendix/a6_mature_scale.tex | 16,451 | 326 | 09-09 01:52 | OLMo-2 1.485B、temporal holdout 24 packs、FineWeb-Edu holdout-512 |
| paper-2027/appendix/a7_exponent_adjustments.tex | 8,406 | 179 | 09-09 01:58 | normalized-index 公式（ξ_H=0.7382780681078285, ξ_L=0.366403835112904）、BM 三次式推导 (l,h,N)、gain 公式 g=1+0.1·ln4=1.1386294361 |
| paper-2027/appendix/a4_supporting_experiments.tex | 1,206 | — | 08-26 06:10 | **孤儿文件**：main.tex 只 input a1/a2/a5/a6/a7/a3，a4 与全部 budget_*.tex 同样未被引用（本轮 grep 全仓 .tex/.py/.sh 无 input 命中）[已验证] |
| paper-2027/DOCUMENT_TEXT_MAP.md | 159,078 | 3,393 | **09-10 07:50** | 全文本源+refs.bib+被引 tables 的拼接快照（`<!-- FILE: ... -->` 33 个标记）；09-10 唯一新增论文侧文件 |
| paper-2027/NARRATIVE_GUIDE.md | 1,810 | 29 | 09-09 01:28 | 六条叙事纪律 |
| paper-2027/CHANGES_FROM_NEURIPS2026.md | 2,121 | 30 | 09-09 01:09 | NeurIPS→exponent-allocation 六维对照 |
| paper-2027/README.md | 1,255 | 21 | 09-09 01:28 | 入口文档 |
| paper-2027/AUTHOR_VERDICTS_20260828.md | 2,944 | 63 | 08-30 21:31 | 8 月周期墓碑（5 条 durable 决定）；完整原件 `git show 8bfcd3b:paper-2027/AUTHOR_VERDICTS_20260828.md` |
| paper-2027/SUBMISSION_CHECKLIST.md | 8,699 | 175 | 08-30 21:31 | Gate A 09-17 内部冻结 / Gate B 09-18 abstract / Gate C 09-25 全文 |
| paper-2027/research/EXPONENT_REVISION_REPORT_20260909.md | 3,343 | — | 09-09 01:36 | 重构报告（9 页正文/38 页、五图像素一致验证、审稿进度） |
| paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md | 7,562 | — | 09-09 01:09 | 主张↔公式↔证据↔引用内部表 |
| paper-2027/research/pdf-review-rounds/20260909/{r01..r05}/ | — | — | 09-09 | r01–r04 各含 review.md+SHA 快照；r05 只有 paper.pdf+identity.json（SHA `53b0cb96…`，created_utc 2026-09-09T06:00:48Z），**无 review**；r06–r10 不存在 |
| paper-2027/main.pdf / main.log | 798,603 B / 34.5 KB | — | 09-10 07:50 / 03:01 | 09-10 重编译；`git diff 951f51e..HEAD -- paper-2027/` 显示 09-10 净改动仅 DOCUMENT_TEXT_MAP.md(新增)+main.pdf(重编译,字节大小不变) |

### 09-10 统一理论与同日核查文档（docs/research/，全部本轮直接 Read）

| 文件 | 大小 (B) | 行数 | mtime |
|---|---:|---:|---|
| UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md | 36,013 | 143 | 09-10 07:50 |
| BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md | 6,284 | 63 | 09-10 07:50 |
| ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md | 7,408 | 72 | 09-10 07:50 |
| ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md | 12,726 | 281 | 09-10 07:50 |
| EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md | 6,186 | 138 | 09-10 07:50 |
| ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md | 9,322 | ~230 | 09-10 07:50 |
| ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md | 11,731 | 219 | 09-10 07:50 |
| COSH_REDESIGN_EVIDENCE_REVIEW.md | 15,563 | 183 | 09-10 07:50 |
| PC2_FAILURE_AND_CLAIM_AUDIT_20260910.md | 14,106 | 130 | 09-10 07:50 |
| PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md | 17,596 | 127 | 09-10 07:50 |
| NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md | 37,360 | 630 | 09-10 07:50 |
| NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md | 43,169 | 326 | 09-10 07:50 |
| ACTIVE_RESEARCH_GOAL.md | 6,838 | 66 | 09-08 21:12 |
| paper-2027/refs/references.bib（抽查 wu2026datashapes/li2026copeclipped/karypis2026lerope 三条目） | — | — | 09-10 03:01 |

### git 证据（本轮 `git log` / `git diff` / `git status`）

- 8 月周期关闭：`93d7eac paper: complete ICLR 2027 revision cycle (9-page body, …)`（2026-08-28 07:41）。
- 9 月重构相关：`09876d5 checkpoint: upgrade ICLR paper and freeze narrative`、`8e4d567 paper: reframe narrative around support-allocation decomposition`（均 08-29 时间戳显示，内容属重构前旧叙事）；`00_abstract.tex` 最后一次内容变更出现在 `b8466de Hourly workspace snapshot 2026-09-09 02:28`。
- **09-10 之后 paper-2027 无任何 .tex claim 修改 [已验证]**：`git diff 951f51e..HEAD -- paper-2027/` = `DOCUMENT_TEXT_MAP.md | 3393 +++++` + `main.pdf | Bin 798603 -> 798603`；`find paper-2027 -newermt 2026-09-10` 仅命中编译产物与 TEXT_MAP。
- sections/*.tex mtime 09-09 23:28–23:30 但内容与 22:37 快照一致（重写未改内容）。

---

## 2. 任务时间线

### A. 2026-08 修订周期（前史，已关闭）
- 目标：完成 ICLR 投稿稿修订周期。方案：qwen 五席 panel（Major Revision，全 6/10）→ 作者裁决 → 执行。
- 结果：成功关闭于 `93d7eac`（08-28）。冻结裁决（memory + AUTHOR_VERDICTS 墓碑）：Prop 2 正确、1.485B canonical record、off-by-one section map、MLA scope、factorial 双标准。08-30 `8bfcd3b` 文档清理后 AUTHOR_VERDICTS 退役为墓碑，5 条 durable 决定迁至 AGENTS.md / NARRATIVE_GUIDE.md / INDEX.md / HANDOFF.md / REVISION_BRIEF.md。**注意：这些裁决针对 8 月稿；9 月重构（任务 B）已在其上重排全文，Prop 编号等结构不再一一对应，但 1.485B 命名、"结果只出现一次"等纪律在当前文本源仍可见。**

### B. 2026-09-09 论文重构（当前稿件的直接来源）
- 目标（REVISION_BRIEF.md 原文）："Author-approved scope, 2026-09-09: reconstruct the complete paper from `main_0726` and the August frozen manuscript, using existing evidence. The title is **Beyond the Base: Exponent Allocation in RoPE**."（不新增任何模型运行。）
- 方案：五个 Luna agent 资产盘点 → 主代理源检查 + Top-15 裁决 → 手稿/图重建（`figs/make_exponent_revision_figures.py`）→ 一次背景知情 Sol 二审 → 十轮全新 Sol 仅-PDF 审稿。
- 结果 [已验证]：正文 9 页 / 共 38 页，无未解析引用/overfull，嵌入字体、匿名元数据；五张新图与源包像素一致、四表 TeX 一致、8 个几何量对独立积分偏差 ≤3.55e-14。
- 审稿进度：**r01 borderline slightly favorable、r02 borderline leaning accept、r03 borderline、r04 accept leaning（均"未发现推翻性数学错误"）；r05 已生成不可变 PDF 快照（09-09）但没有 review.md；r06–r10 未执行。**r04 处置（review.md §Author-directed prose pass）记录："After this review, the author reaffirmed the exponent-distribution research question and prohibited defensive writing. The primary agent … removed repeated inventories of unclaimed universal optima, missing theory targets, evidence ceilings, and capability caveats."→ 这就是 09-09 深夜那轮 sections/*.tex 重写（当前文本）。
- 状态滞后项：HANDOFF.md（09-09 01:09 写）仍说"Rounds 1 and 2 have completed; round 3 is reviewing"，与目录实况（4 轮完成/5 轮快照）不符；十轮循环被 09-09 晚起的非几何屏幕与 09-10 理论冲刺中断 [部分证据——推断自 mtime 与后续文档，无显式中断记录]。

### C. 2026-09-09 非几何候选计划（论文 §6 冻结段的实验前线延续）
- 目标：零训练胜过 MrRoPE-Pro。方案：E1–E10 十候选 + 宽松混合 RULER 筛选（NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md，326 行）；面板固定 Qwen2.5-3B-Instruct（revision aa8e7253…，36 层、16Q/2KV、head_dim128、64 对、Native 32768、theta 1e6），对手 = 完整 MrRoPE-Pro 静态 S4（边界 23/40、gain 1+.1·ln4）。
- 结果：E1(s28_less)/LBS(s29 方向)/E2/E3/E7/E8 等完成或部分完成（数字见 §5）；两次用户纠正纳入（见 §6）；09-09 用户指令"提交并推送已有报告与代码，停止继续寻找方法"（ACTIVE_RESEARCH_GOAL.md 末节）。

### D. 2026-09-10 统一预算理论冲刺（GPU 关闭，纯理论+CPU）
- D1. UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md（自称"纯理论 + 论文策略分析"）：把低/中/高频分配形式化为守恒预算放置问题（I1 bank 恒等、I2 尾部精确 ÷S、"额外 ln S=1.386 放哪里"）；filter bank vs arc clocks 二分法；分配规则"bank 恒等+危险区完成+最小桥"；用 36 行开发面板统一读法（§3 表）；新算证据距离（`planned_controls/evidence_distances_20260910.json`）；提出 Claim 三层结构 + Figure 1 相图 + Core-A/B/C 审稿人实验设计；§6 EVQ 和解（sink 保留/source 反转）；§8 整合外部"姐夫"稿（三操作分解、EVQ 变分改写、J_r 配方分解 + ±v 镜像控制、三条件因子实验），并做 novelty 核实（arXiv 2607.07678 / 2607.10134 / 2602.05258 真实存在，"统一预算视角确实不能单独当 headline"）；§9 冻结 GPU 队列 0446→0448→0449→0450→0451。结果：**当日即被 D2 同日复核推翻多条核心表述**（见 §3、§4）。
- D2. ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md：逐条纠错——(1) "Σε=1 ⇒ 一个自由度"错：17 个实际 log-gap 总和 ≈5.056039（原生部分 ≈3.669745），16 自由度，桥宽是外加设计限制；(2) D_j=W·S^{m_j} 不是"未见相位边界"：Native 槽 36–39 在 W 内已转 2.199/1.772/1.428/1.151 圈，单频圆周已覆盖；(3) "所有赢家同方向移预算"与 ε 质心表矛盾（E1 后移、LBS/P2 前移）；(4) LBS 并未"完成到 m=1"（m36–39=.625169/.729476/.846534/.979914，D≈77954/90082/105953/127473），E1 根本没动槽 36–39 却修好 89K 行；(5) gain×频率效应不正交；(6) 证据距离数据质量：FWE 距离定义不成立（答案词全文出现 10785/4793 次）、VT 只取 max 丢链、MQ 一行含 77 与 120934 双距离——"36 行并非同一种距离—准确率样本，不能据此将 75–112K 宣称为已验证的危险带"。同时确认可保留数学：三操作分解、EVQ 变量替换 J[h]（α=1,β=4 复核目标差 1.6e-11、KKT 残差 1.1e-15）、ridge 配方分解。结果：6Pro 校准验证开跑（用户 11:06 授权 GPU；RTX 4080 SUPER 32760 MiB；采集计划 §验证中的实际约束，入口 `experiments/nongeometric_screen/pro_block_calibration.py`），**记录明言"本节记录的是当前计划与已开始的采集，不是已完成的优化或模型收益"**。
- D3. CPU 数学三连：ROPE_ALLOCATION_SUBSPACE_DERIVATION（联合弱子空间传输 U、Selective-PI 恒等式、Q/K 算子 Gram 加权；关键反例：Smooth MrBudget 的 U 与加权 U_H 在**全部 36 层三个 cutoff** 都优于 MrPro，但其 128K 开发分更差 ⇒ 几何量不是充分选表器）；EVQ_NONLOCAL_KERNEL_CORRECTION（旧 cos-collision 的 delta 近似丢弃非局部 log ridge，exact 核 K_L 分解 + CPU 五点界验证全过，误差 9.9e-5~4.5e-3；定位："specific correction to a modeling step, not evidence that a sharp transition improves a frozen model"）；ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION（exp(s) 的 Bessel 展开 ⇒ softmax 产生整数组合混合频率；ω28−2ω29+ω30 周期 70285.94 token；半周期格点 q=0/1/2/3 → 最早 j=43/36/28/21；幅度比 2I1²I2/I0³ 在 a=.3/1/2 为 .000488/.0427/.294 —— "长混合周期本身不足以选规则"）。均无模型执行、无新候选入队。
- D4. ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS：失败机制总账（三带不是定理、见圆≠充分、竞争键的 logsumexp 压力、精确局部保持 ⇒ ν=ω mod 2π、相位码无一致分离、冻结权重加兼容性层）。记录 E1 跨模型迁移**混合**：Qwen7B 32K 平 / 128K −1.667pp；OLMo 4K −3.333 / 16K +3.819pp（OLMo 仍低于其 BM 对照）。
- D5. PARALLEL_NONGEOMETRIC_20X10_AUDIT：独立审 PARALLEL_PLAN，纠正三处过强解释——E7 的"160×"是 BF16 实现（NMSE 7.937e-8→1.281e-5），不是线性化失败；gain 只乘幅度不动相位弧，BM@gain 结果不能当弧机制独立确认；弦距对压缩非单调（Δν=2π 压缩一半反而最大分离），s28_less 的逐距离差 +0.162/−0.365/+0.532/−1.528/−0.405 实证无一致方向。同时确认 E1 目前最可信解释具体化："changing slot 28 improves both prefix formation and subsequent reading enough to move a competing-answer margin across the greedy decision boundary on one multikey case"。
- D6. COSH_REDESIGN_EVIDENCE_REVIEW：对论文 Cosh 主张的内部复核——Phase16 99 全量重算（公式−Geo 7/9 配置、18/27 seed、−0.0133 NLL；公式−邻点 3/9、8/18、**+0.0221**）；τ 解析缺口直接反证（surrogate √(β/α)=6.244/5.704 vs 部署规则 1.414/1.000，4.4×/5.7×）；**三臂 303 题 QA 重算：Base-Native/Native-LoRA/EVQ-LoRA 宏 F1 = 23.09/21.10/11.26%，exact 33/25/4，>8K 子集 exact 全部 0/109**——"当前 Cosh 问题不是从来没做下游，而是已有部分下游失败…尚未形成与醒目 PPL 收益相称的稳定能力证据"；与 YaRN 训练量对齐（YaRN 1.68B-token 64K 真序列 vs 我们 300 步 8K rank-64 LoRA）。
- D7. 论文侧 09-10 动作 [已验证]：生成 DOCUMENT_TEXT_MAP.md（07:50）+ 重编译 main.pdf（log 03:01 / pdf 07:50，字节大小与 09-09 相同）。**没有任何 claim 级 tex 修改。**

### E. 当前论文声明的关键数字（正文实际值，[已验证] 逐条对应文件）

- 摘要（00_abstract.tex）：三 seed 三长度固定范围全改善；retarget 改变排序；frozen weights 偏好训练表；"After matched Llama-3-8B adaptation at 8K, 32K perplexity falls from **991.5 to 127.9**"；"A boundary-matched adjustment improves OLMo's mean token F1 over MrRoPE-Pro from **21.62% to 25.44%** across five natural-QA tasks"。摘要**不含任何 128K/Qwen3B 数字**。
- 固定范围（02_identification.tex）：151.9M、L=256、K=32、499,974,144 tokens、Geo 臂 b=L_train（FMRoPE 规则）、Cosh τ=4 端点锚定、仅 30 个内部指数不同；Cosh−Geo NLL @256/512/1024/2048 = **+0.026/−0.281/−0.176/−0.146**；retarget 臂 Cosh−Geo @512/1K/2K = **+0.060/+0.227/+0.460**（反转）。50.9M factorial：参考强度 7/12、1.25× 强度 10/12、matched exponential 9/12；Cosh−exp 差 +0.00074 [−0.006, 0.008]。
- 交叉（03_findings.tex）：50M Geo 权重换 Cosh 表 PPL 7.14→76.20；反向 7.16→23.05；151.9M 两 seed 重复。
- 理论（03_theory.tex + a1_proofs.tex）：r2(Γ)=2K/[1+(K−1)c̄]；b=500000,K=64,L=4096 慢段 23 pair→名义 46 维、r2=2.00013；collapse 四阶系数 19/12600（解析/数值比 1.00058 @x=.05,y=.10）；parity-lattice K_par=⌈⌊L/π⌋/2⌉；两个静态排序反例（cos-only: r2 6.29478 vs 7.99971；跨长度反转 C_L 0.539 vs 0.632 → C_4L 0.345 vs 0.034）；移植障碍定理（A^⊀R_{Ω'}B=R_Ω ⇒ 多重集相同）；Cosh Thm（ρ_τ=τcosh(τ(1−φ))/sinhτ，warp 逆 CDF）；self-consistency τ²T2+T1=τcothτ；离散传输界 W1≤sinhτ/4Kτ；参考规则 τ=c·d_head/√L_train（c=1），99 runs 7/9 配置均值、18/27 配对 seed 胜 Geo（注意：与 D6 重算的"公式−邻点 3/9、+0.0221"并存——论文对邻点比较只说"order varies by configuration"）。
- 训练/适配（04_experiments.tex）：432M MLA（16 对、500M tok、8K 训）16K PPL 138.8→95.6（8K 35.4/35.8）；750M 续训 16K PPL 45.1→24.4、passkey AR 0→77.5%（40 trials）；**Llama-3-8B：8/16/32K PPL 6.82/10.07、108.96/24.07、991.48/127.91（Native-LoRA/EVQ-LoRA）**，target-block hit@16 中位 18.75%→64.06%，去远程 gold 注意力 NLL +1.5055（EVQ）vs −0.0095（Native）；RULER 续训 516 步：16K macro 0.30%→14.03%（8K 94.44/77.60），16K whole-response exact 0/1.54%；454M 三 seed 共享 smooth-ramp 后 16K PPL Geo 157.7 / Cosh 107.5。
- 冻结（04_mature.tex + a7）：固定范围 RULER：OLMo-1B Geo 0.56 / Coarse ramp 61.04 / Residual-guided 60.47（held-out 九任务 16K）；Qwen-1.5B 57.75/64.00/66.50（四任务 dev 64K）；Gemma-1.1-2B(K=128) 16K 79.00 vs 72.81（+6.19 [2.81,9.63]）；Qwen-0.5B 静态 13 任务：Native 54.78/22.05、YaRN-2 55.94/45.37、NormIdx-2 55.92/**51.46**（64K +6.09 [2.76,9.58]，32K −0.0256 [−3.26,3.22] 近平）；**BM vs MrPro 六任务（同一 36 行式面板设计）：OLMo 4/16K 短/长：MrPro 37.85/2.78、BM 81.81/51.32、MrUni 76.88/32.12、YaRN 54.38/6.94；Qwen-3B 32/128K：MrPro 87.22/78.13、BM 91.67/70.83；Qwen-7B 32/128K：MrPro 83.33/84.44、BM 80.00/71.11**；自然 QA：631 条 >4096-token 五任务 F1 21.62→25.44 [+1.32, 6.29]，短层 +2.0028 [−3.56, 7.47]；结论句（正文）"BM gives the higher OLMo scores and Qwen-3B's higher 32K score; MrRoPE-Pro gives the higher 128K scores on both Qwen checkpoints… differences connect the preferred exponent shape to the checkpoint and operating length."
- 讨论（05_discussion.tex）：设计两形态（训练前显式准则 / 训练后 native-relative）+ "checkpoint-dependent preferences"；下一步 = sparse attention。

### F. 上一任务交代核对（任务说明中"panel 冻结裁决"现状）
- "Prop 2 正确"：8 月稿命题 2 的编号在 9 月重构中不存在直接对应物；当前稿命题/定理为 prop:collapse、prop:parity-lattice、thm:budget、thm:obstruction、thm:ode、thm:self-consistency、lem:budget-crossing（编号共享 theorem 计数器）。r01–r04 四轮独立审稿均"未找到推翻性数学错误" [部分证据——审稿是模型意见，非形式验证机]。
- "1.485B canonical record"：当前文本源一致使用 OLMo-2-0425-1B-Instruct "(1.485B actual"（a6:14）与 "750M/1.485B"（a2:35,38）[已验证]。
- off-by-one section map、MLA scope、factorial 双标准：重构后对应物为 02_exponents 的 k/K vs k/(K−1) 明说（eq 后段落）、04_experiments 的 MLA 段、02_identification 的 7/12 vs 10/12 双标准——裁决精神保留 [已验证 文本级]。

### G. HANDOFF 与实况差异清单（并入前需知）
- HANDOFF "round 3 is reviewing" ↔ 目录 r01–r05（4 完成 5 无评论）。
- HANDOFF "Current build has 9 body pages and 38 total pages" ↔ 与 09-10 07:50 PDF 同源（tex 未变，页数不变，[假设] 编译确定性未逐项复核，但字节大小相同支持不变）。
- HANDOFF "Final review and source-archive verification remain in progress" ↔ 09-10 无人续做 r05–r10。

---

## 3. 理论主张表（主张 | 证据等级 | 出处 | 后续是否被纠正/推翻）

### 3.1 论文当前主张（main text 声明）

| # | 主张 | 证据等级 | 出处 | 纠正状态 |
|---|---|---|---|---|
| P1 | 指数分配 = (a,R) 范围 + 归一化形状 z；几何表 z 恒等距 | [已验证]（定义） | 02_exponents eq:(table-decomposition) | 与 09-10 "三操作分解"同构（UNIFIED §8.1 采纳项即此，无冲突） |
| P2 | r2 恒等式 r2=2K/[1+(K−1)c̄] 及闭式 cross-Gram | [已验证]（数学，trace 恒等式+数值 4e-14 内） | 03_theory eq:(budget-identity) + a1:cross-gram | 未被推翻 |
| P3 | 慢频共享 span{1,Δ}；四阶系数 19/12600 | [已验证] | prop:collapse + a1 slow-collapse | 未被推翻；EVQ_NONLOCAL 补：该结论依赖测度选择（softmax 度量下常数方向被 F=diag(p)−pp^⊤ 湮灭，见 a1 末段） |
| P4 | Cosh 是给定凸准则唯一 minimizer（闭式） | [已验证]（数学） | thm:ode + a1 self-consistency | **范围被内部复核钉住**：COSH_REDESIGN §1.2 "一旦选定常系数局部平方项和 min-kernel，无论怎样拟合 α、β，都会返回 Cosh 族…不能作为独立证明真实 RoPE 应当采用 Cosh"；EVQ_NONLOCAL：若从 exact cos-collision 核推，delta 近似 over-罚窄变化（ridge 乘子 <A0）——Cosh 是"设计偏好的解"，论文措辞（"specified convex criterion"/"tractable surrogate"）已如此限定 |
| P5 | 移植障碍：任何位置无关线性映射不能等价换表 | [已验证]（定理） | thm:obstruction + a1 | ROPE_EXTRAPOLATION §6 防过度解读："it does not imply that beneficial finite-distribution cold swaps are impossible. BM, P2, and E1's measured conditional gains must be retained as counterweights" |
| P6 | 固定端点下内部指数改变学习行为（3 seed × 3 长度全降） | [已验证]（配对训练、原始收据） | 02_identification | 未变 |
| P7 | retarget 反转排序 ⇒ 范围与形状要一起评 | [已验证] | 02_identification | 未变 |
| P8 | 权重与表共适应（7.14→76.20 等） | [已验证]（50M/151.9M 两规模） | 03_findings | 未变；为"冻结阶段用 native-relative 位移"提供动机（04_mature 开头明引） |
| P9 | Cosh 在 432M MLA / 750M / 8B LoRA 的外推 PPL/NLL 收益 | [已验证]（读数即表格值） | 04_experiments + a6 | **能力上限警示（不入论文正文但约束表述）**：COSH_REDESIGN §4 三臂 303 题 QA 中 EVQ-LoRA 宏 F1 11.26%、exact 4/303、>8K exact 0/109；论文只主张 NLL/PPL/RULER-macro/来源依赖，摘要句式"PPL falls from 991.5 to 127.9"未越界，但任何"能力恢复/长文 QA 能力"式改写都会越界 |
| P10 | 8B RULER 续训长度迁移（16K macro 0.30→14.03%） | [部分证据]（13 家族、516 步、8K 回退 94.44/77.60 如实披露） | 04_experiments | 未变 |
| P11 | 静态表对：NormIdx-2 64K +6.09pp 胜 YaRN、32K 平 | [已验证]（新 seed、bootstrap CI、外部版本指针） | 04_mature tab:static-index-main + a7 | 未变 |
| P12 | BM：同 band 端点/总量下平滑 radix 增量（唯一最优 ε_q 解） | [已验证]（推导+数组逐项断言） | a7 eq:(bm-exponents) | 未变；09-10 的 N′ 族（N=17→16→15）是**同坐标的窄桥变体**，非新框架 |
| P13 | BM 胜 MrPro：OLMo 全部 + Qwen3B 32K；**Qwen 两 checkpoints 128K 由 MrPro 胜** | [已验证]（表值）+ 面板小（3B 2 短/4 长每任务） | 04_mature tab:bm-models | 未变——这正是统一理论要改进的现状：论文当前诚实结论 = "preferred shape 依赖 checkpoint 与长度"，没有 128K 赢家 |
| P14 | 自然 QA F1 21.62→25.44 [+1.32,6.29]（OLMo） | [已验证]（631 条、配对 bootstrap、全任务正差） | 04_mature + a7 | 未变 |
| P15 | τ 参考规则 c=1 与"99 runs 7/9、18/27" | [部分证据]（论文如实限定"runner-defined weighted extrapolation NLL"、邻点 order varies） | a1:tau-scaling | COSH_REDESIGN §2 加数字：公式−邻点 3/9、8/18、+0.0221 与 τ 解析缺口 4.4×/5.7× —— 支持论文不主张 τ 最优，但警告"near-optimal law"式升级不可行 |

### 3.2 09-10 统一理论主张（并入候选，逐条独立定级）

| # | 主张 | 证据等级 | 出处 | 当日/后续纠正 |
|---|---|---|---|---|
| U1 | 端点不变量 I1：j≤23 恒等（动高频段全败） | [部分证据]（3 个独立失败读数：HighGapToLong 32K −17.1/128K −10.8、36 行 0 提升；MrUni 全表 ÷4 → 32K 64.6；E8 单槽 → −13.9pp 级） | UNIFIED §1 / BUDGET_MODEL §1 | GLM 复核钉范围："少数失败干预不能证明所有高频改动均失败"——是**方向证据**，不是定理；E8 的失败还有 format/截断症状（PARALLEL_AUDIT） |
| U2 | 端点不变量 I2：j≥40 精确 ÷S（E2 推到 ÷4.93 → 128K 崩 54.7，12 行面板） | [部分证据] | UNIFIED §1 | 同上：单点失败读数；"改平台水平破坏位置重参数化精确性"是理论解释 [假设] |
| U3 | 守恒：两端固定后过渡段"额外 ln S"总量锁定 → 一个自由度 | [已验证] 修正 | UNIFIED §1 | **被 GLM §1 推翻原表述**：Σε_i=1 固定的是**增量和**；17 个实际 log-gap 和 ≈5.056 含原生 3.67；分布仍有 16 自由度；"桥宽唯一参数"是外加设计限制。可并入的只能是弱化版（水床路径形式对**增量**成立）|
| U4 | 识别地平线 D_j=W·S^{m_j} 作为"未训练弧段"边界 | [假设] | UNIFIED §1.1 | **被 GLM §2 推翻因果读法**：Native 槽 36–39 在 W 内已转 >1 圈，超 D_j ≠ 单频未见相位；联合关系越轨需联合模型论证。D_j 可作为定义量保留 |
| U5 | filter bank vs arc clocks 二分法（锐利整数对齐 vs 平滑慢函数） | [假设]（与 E3 gain、E2/E8、I1 相容的机制叙事） | UNIFIED §2 | 未被证伪，也未检验；PARALLEL_AUDIT 把最可信成功解释具体化为 margin-跨阈值，而非通用二分法 |
| U6 | 危险区槽 36–39 + MrPro 欠完成（D=75–112K < 128K） | [部分证据]→[假设] | UNIFIED §2 | GLM：MrPro m36–39 数字属实，但"危险带"作为因果带未验证（36 行混质）；"MrPro 长端损失来源"**归因未成立** |
| U7 | 方向规律："所有赢家右移预算（桥变窄/危险区完成）" | — | UNIFIED §3 | **被 GLM §4 推翻**（ε 质心表：E1 右移 34.699 vs 34.667，LBS 左移 34.440、P2 左移 29.821）；"不太可能是噪声"失效。E1 与 LBS/P2 是**两个独立机制方向**（前端保真 vs 后端完成，BUDGET_MODEL §2） |
| U8 | LBS = "危险区完成手术" | — | UNIFIED §3 | GLM §5 修正：LBS m39=0.9799 未完成；且 E1 不动危险区也修了 89K 行 |
| U9 | 证据距离-失败定位（mk_2@89K、vt_0@106K 落入地平线带） | [部分证据]（距离可算、MrPro 特异性失败行属实） | UNIFIED §4 | GLM §证据距离复核：FWE 定义不成立、max 距离丢链、MQ 双距离——行级 n 太小且样本异质；"75–112K 已验证危险带"不可写；论文级版本 = Core-C holdout + 固定内容配对距离设计 |
| U10 | 零训练不需要训练=两精确态不产生新计算（恒等 + 整块 ÷S 重参数化） | [假设]（重参数化数学本身 [已验证]：块内成对旋转=原生 ¼ 位置成对旋转） | UNIFIED §5 | 交叉缓存证据支持解释但不证明（UNIFIED §7 自己声明 factorization 近似需如实承认） |
| U11 | EVQ 和解：sink 共享、source 反转（预算来自桥的渐进性，不从高频密度） | [部分证据]（HighGapToLong 证伪字面 source 操作；P2 实际只取高频 0.0008） | UNIFIED §6 | 作为对论文 Cosh 讨论的**重新解释**未发表验证；与论文冻结段无冲突（论文从不在冻结模型上做高频压缩） |
| U12 | 4× 窗口只需 +10.19% 对数跨度（13.60→14.99）——"贵的是放置不是跨度" | [已验证]（CPU 算术） | UNIFIED §8.1 | 可并入的定义级事实 |
| U13 | 联合弱子空间 U / Selective-PI / 加权 U_H 作为分配约束 | [已验证]（数学+CPU 恒等式）；但作为**选表器不成立** | ROPE_ALLOCATION_SUBSPACE | 文内自我反例：Smooth 全面优于 MrPro（U 与全 36 层 U_H）但 128K 开发分更差；Q/K Gram 加权也救不回 → 只能当诊断/解释候选，不能当预测主张 |
| U14 | softmax 混合频率机制（慢包络可由快频组合产生；共同 ÷S 精确重定时全部谐波） | [已验证]（数学：Bessel 展开+尾界；CPU 演示 60.96%→50.03%） | ROPE_SOFTMAX_MIXED | 无模型证据归因任何已测成败；"discriminating next measurement"未排 |
| U15 | 三带非定理、外推无上界向量爆炸、局部精确保持⇒恒等表、有限相位码无一致分离、logsumexp 竞争压 | [已验证]（数学）/ 定位=分析 | ROPE_EXTRAPOLATION | 直接可并入的"limits"材料（与论文反例、obstruction 一脉） |

---

## 4. 失败机制清单（试过什么、为什么失败、复盘对复发模式的警告）

| 机制尝试 | 结果 | 失败/限制原因 | 复发警告（来源） |
|---|---|---|---|
| 用旧 cos-collision delta 近似论证 Cosh 必须平滑 | 数学上不闭合 | exact ridge 是**非局部、惩罚随变化变窄而次线性**（h_c 乘子 ≤A0 且递减）；delta 近似在每一非零波数 over-penalize；且先验曾被反向选择抬 R²（09-03 记录：24,000 配置扫出 886 个 R²>0.99） | "拟合 R² 高 ≠ 需求服从该条件"；代理核 ≠ 任务（COSH_REDESIGN §1.1、EVQ_NONLOCAL） |
| τ=c·d/√L 公式选点 | 99 runs 重排后对邻点净负（+0.0221）；解析反证 4.4–5.7× | 变分解不解析确定部署 τ；旧"near-optimal law"混用 stage 评分 | 任何"理论⇒部署点"跳接需独立预注册点验证（COSH_REDESIGN §2） |
| PPL 改善当能力代理 | 8B EVQ-LoRA：PPL 991.5→127.9 但 303 题宏 F1 21.10→11.26、exact 4/303、>8K exact 0 | PPL 度量的是所测文本平均条件概率，不是指令遵循/绑定/生成 | 铁律"代理≠能力"的本地实例；报告时保留各自指标名（COSH_REDESIGN §4；NARRATIVE_GUIDE 第 5 条） |
| 固定态代理选表（U、加权 U、E8 分数） | Smooth 在 U、36 层 U_H、operator-MSE 三项全胜 MrPro 但 128K 68.3 vs 78.1；E8 代理最优而实际 −13.9pp | 代理不含"有用计算的损失"项；min U 的解是 full PI | "U 与本地失真不是充分 selector"——写进论文的机制段前必须有能力级验证（SUBSPACE_DERIVATION §What…prevents、PARALLEL_AUDIT） |
| 单步导数当有限扰动 | E7 的"161×"是 BF16 实现误差（NMSE 7.94e-8→1.28e-5），不是理论失真 | 三问要分开：导数预测理想干预 / 部署算术实现理想 / 局部准则预测全模型 | 数值层、理想层、下游层分开归因（PARALLEL_AUDIT §1） |
| gain 当独立确认 | BM@gain0.074 32K +12.8/128K −8.1：包不是单因子 | gain 只乘幅度 g² 不动弧；2×2 表因子里 BM@gain1 NLL 改善 0.02–0.035 而 128K −12.0 ⇒ 弧安全独立于 gain，但 gain×表有交互 | 因子实验前先固定 factorial（PARALLEL_AUDIT §2、GLM §6） |
| "最小位移/近 Native"当选表器 | gap-capped 更近 Native 而 128K −15.97pp、36 条 0 胜（09-09 历史） | 位移距离不是收益轴 | 已在 09-09 计划 §2 固化为"排除了什么" |
| 手搭局部再分配（HighGap、G1-G3、BM_ScaleTaper） | 从活动队列撤回（09-10 candidate-quality critique 后） | "没有足够理由期待优于已有效分配"；未测 ≠ 否证（措辞纪律） | 后续候选生成必须从有效规则出发+给具体机制（MECHANISM_TRANSFER priority correction 段） |
| 开发面板复用当泛化 | 36 行历史 = development（09-09 计划 §5.2 明文）；E1 跨模型迁移混合（7B −1.67、OLMo −3.3/+3.8） | 同面板多轮选择偏差 | 任何 128K 胜 MrPro 主张先过 0450/0451 新样本 |
| 距离-准确率曲线（36 行版） | 不能发布 75–112K 危险带 | FWE 距离定义不成立等 | 机制升级需要固定内容配对距离的专用批（GLM §证据距离） |
| PC2（稀疏注意力线，另一论文） | 十候选无整体胜 B0；"Cascade 没有完整生成实验，因此尚未被证明任务质量失败" | 集合差异 ≠ 答案差异；候选遗漏与分母误差两个独立瓶颈 | 双向纪律：既不得以代理恢复率冒称能力，也不得把未测写成否证（PC2_FAILURE_AUDIT §2、§4） |

---

## 5. 频率表/方法定义清单（名称、构造规则、32K/128K 得分）

统一坐标：原生表 ω^N、安装表 ω′=ω^N·e^{−d}=ω^N·s^{−m}；论文正文 eq:(movement-allocation)/eq:(adjustment-families) 与 09-10 文档 m 坐标同构（d_k/log b = φ′−φ）。除注明外，Qwen 冻结面板 = Qwen2.5-3B-Instruct 六任务（NIAH single-2/multikey-2/multiquery/VT/FWE/QA-1）、官方 gain 1.1386294361、36 行 = **development**。

| 名称 | 构造规则 | 32K / 64K | 128K | 状态与出处 |
|---|---|---|---|---|
| **Geo/标准 RoPE** | φ_k=k/K（等价 z_k=k/(K−1)） | 基准 | — | 02_exponents |
| **Cosh/EVQ**（ρ_τ，warp 逆 CDF φ_k(τ)） | argmin C_app=α/2∫ρ²+β/2∫S²；τ=√(β/α)；训练期用（151.9M τ=4 锚定、432M/750M/8B τ 按 tab:allocation-protocols；参考 τ=d/√L_train） | 8B LoRA：32K PPL 991.48→127.91 | —（不适用） | 论文主构造 [已验证 数学；能力=PPL/NLL 级] |
| **YaRN**（频率混合） | d_k=−log(1−w_k+w_k/s)；g=1+0.1 log s | Qwen-0.5B：55.94@32K / 45.37@64K；OLMo 六任务 54.38/6.94 | — | 04_mature/04_mature tab:static-index-main |
| **MrRoPE-Uni** | m_q=q/N（mixed-radix 均匀） | Qwen3B —；OLMo 76.88 短/32.12 长 | — | 04_mature tab:bm-models 段 |
| **MrRoPE-Pro (MrPro)** | m_q=q(q+1)/[N(N+1)]，N=17（Qwen l,h=23,40） | **87.22** | **78.13** | 论文表 + UNIFIED §3 一致 [已验证] |
| **BM** | m_q=q(q+1)(3N+2−2q)/[N(N+1)(N+2)]（min Σ(ε_{q+1}−ε_q)² 唯一解） | **91.67**（+4.44） | **70.83**（−7.30） | 论文 tab:bm-models；冻结段当前"32K 赢/128K 输"方 |
| **Normalized-index 静态表** | 64 对参考网格 ξ_j 裁剪插值（ξ_H=.738278, ξ_L=.366404），ω′=ω^N s^{−m} | Qwen-0.5B 55.92@32K / **51.46@64K** | — | 04_mature + a7 |
| **Coarse ramp / Residual-guided**（冻结固定范围屏） | 残差投影能量引导；ramp 近似无标签 | OLMo 61.04@16K / Qwen-1.5B 64.00、66.50@64K | — | 04_mature tab:frozen-exponent-main |
| **s28_less（E1）** | MrPro ⊕ 槽 28 回退到前驱指数（m28 .098→.065359，挪 0.045 到 gap27/28 间） | 87.2（平） | **83.333（+5.208pp）**，2 行升 0 行降 | [部分证据：36 行开发面板；新样本 = 0450 未跑] UNIFIED §3、MECHANISM_TRANSFER、PARALLEL_AUDIT |
| **s29_more（E1 对偶）** | 槽 29 换后继指数 | 95.5（+8.333，全来自 1 个 QA 行） | 77.9（−0.208，VT +1 与 MQ −1 相抵） | 同上，开发级 |
| **LBS（LongBridgeSlower）** | 集中 gap35–38（m36–39=.625/.729/.847/.980） | 80.6 | 80.1（+1.97） | [部分证据 开发面板；"危险区完成手术"标签被 GLM 修正] |
| **P2 / FullLagP2** | 因果 lag 条件残差→集中 gap29（1.088 巨洞）、31 槽完成 | 72.9 | 81.7（+3.5） | [部分证据 开发面板 + 128K passkey 4/4 复用 + tail-512 PPL 5.3575 vs official 5.5035（−2.65%）/ same-gain 5.3705（−0.24%）]；0451 = 长端 VT/QA×16 新样本，未跑 |
| **Smooth(MrBudget)** | 挪向中段、末 gap 削至 .042 | 87.2 | 68.3（−9.8） | 负例 [已验证 开发面板]；U/加权 U 却优于 MrPro（反例，见 §4） |
| **pair(28+29)** | gap28 双倍集中（洞 1.46×） | 87.2 | 74.0 | 负例；同位置叠加不稳健 |
| **HighGapToLong** | 从高频段抽 0.216 log 单位给长程间隔（EVQ 字面操作） | 70.1（−17.1） | 67.3（−10.8） | 负例，36 行 0 提升 [已验证 开发面板]；COSH 和解的"source"证伪件 |
| **MrUni（冻结全表 PI 臂）** | 同过渡区线性累计压缩 | 64.6（−22.6） | — | 负例；GLM 更正其身份（不是全表 PI） |
| **E2 尾部过冲** | 平台推到 ÷4.93 | — | 54.7（12 行面板） | I2 端点不变量证据 [部分证据] |
| **E8 零频槽 51** | ω51=0 | 前 12 行 −13.889@128K | — | 固定态代理高分 vs 实际大跌的反例 [部分证据，12 行] |
| **E3/gain074 因子** | BM 表 + g=1+0.074 log s 或 gain c=0.074 | 98.3（+12.8 vs MrPro 表同 gain 设计内） | 75.3（−8.1） | 幅度/相位因子件；"正交"读法被 GLM/PARALLEL_AUDIT 修正 |
| **N′ 族（0448/0449）** | m_q=q(q+1)/272（N′=16，39 槽完成→D=131K）；N′=15（38 槽完成）；公式族误差 ≤4.3e-8 | 未测（计划 36 行+48 NLL） | 未测 | [未执行] BUDGET_MODEL §3；判定规则已先行冻结 |
| **StackFrontBack（0446）** | MrPro ⊕ s28_less(槽28) ⊕ LBS(槽36–39) 拼接（D 顶到 127.5K） | 未测 | 未测 | [未执行]；检验两机制可加性 |
| **ScaleTaper（OLMo 跨模型）** | 按周期 (l,h,S) 定义的同族几何检查 | 通过几何检查，未测分 | — | [未执行] UNIFIED §7/§9 |

---

## 6. 用户指令与纠正（原文引用）

1. （REVISION_BRIEF.md，09-09，作者批准范围）"Author-approved scope, 2026-09-09: reconstruct the complete paper from `main_0726` and the August frozen manuscript, using existing evidence. The title is **Beyond the Base: Exponent Allocation in RoPE**."
2. （REVISION_BRIEF.md）"A missing local raw mirror is a provenance limitation, not evidence that a documented experiment was fabricated. Preserve valid results and state the actual computation and comparison identity."
3. （pdf-review-rounds/20260909/r04/review.md，Author-directed prose pass）"After this review, the author reaffirmed the exponent-distribution research question and **prohibited defensive writing**. The primary agent … removed repeated inventories of unclaimed universal optima, missing theory targets, evidence ceilings, and capability caveats. Experiment identities, equations, actual scores, selection procedures and statistical units remain."
4. （NARRATIVE_GUIDE.md 第 1/3 条）"Write the positive argument: question, controlled finding, analysis, design, interpretation." / "The first figure should make the intervention and its empirical claim visible."
5. （NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md §1，转述用户纠正）"用户后续纠正已经纳入：**NLL 和 passkey 是初筛与能力维度，不具有单独否决轻微退化候选的权力。既有混合 RULER 是核心评价。**" / "第二项用户纠正也已经纳入：**研究要服务于一篇读者能理解、证据能支撑的论文，不要求打赢所有模型、所有任务。**"；同文件 §1 并引用户授权："E9/E10 放宽了单表旋转算子，依据用户'**只要能赢 MrRoPE，可以重构理论**'的明确许可保留，成功也不能冒充纯静态网格贡献。"
6. （ACTIVE_RESEARCH_GOAL.md 末节，09-09）"用户最新指令为提交并推送已有报告与代码，停止继续寻找方法。……只有用户后续明确恢复研究时再继续。"（同日更早）"2026-09-09: GPU-off research resumed by the user. The user has closed the GPU and explicitly provided sufficient time to find useful methods."
7. （ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md）"用户 11:06 明确告知 GPU 恢复并允许验证。"——且限定"不自动执行 GLM 原来的五个候选"。
8. （PC2_FAILURE_AND_CLAIM_AUDIT_20260910.md）"用户要求停止定时任务：已有 `pc2-pm-keep` 已通过应用接口设为 PAUSED"；"用户授权关机，但在补备份时 SSH 被关闭……尚未成功发出关机命令，不能以断连推断已关机。"；"现在撤销自动成本否决；耗时仍是需要报告的取舍，不代替任务质量判定。"
9. （UNIFIED_BUDGET 文件头）"2026-09-10。GPU 已关闭，本文为纯理论 + 论文策略分析。"；（MECHANISM_TRANSFER）"Latest protocol correction: the user's subsequent instruction supersedes the old automatic full-panel completion rule. … **No new 32K experiments or automatic 36-row completion are queued.**"
10. （任务说明铁律，与 AGENTS.md 一致）"不得把代理指标说成能力结果；不得把未测写成否证；每个结论标注证据等级；引用文件/转录时给出路径或时间戳。"
11. （memory：panel 与 brief）"qwen five-seat panel (Major Revision, all 6/10)；revision cycle CLOSED 2026-08-28 @93d7eac；AUTHOR_VERDICTS retired — 作者 KEPT 全部未执行的 accepted 项，等待新周期（9/18 abstract 截止前）。"

---

## 7. 未决问题（论文 × 统一理论合并的决策面）

1. **十轮仅-PDF 审稿断在 r04/r05 之间**：r05 快照已建无评审；r06–r10 未做；HANDOFF 自称 live 但滞后（§2-G）。并入任何改动前需决定：补完十轮还是带 4 轮记录进入冻结门。
2. **0450/0451（+0446/0448/0449）全部 [未执行]**：GPU 队列冻结在 seetacloud :27741，恢复后 6–7 小时可完成（BUDGET_MODEL §4）。在它们回来之前，128K 胜 MrPro 的任何主张只能引用开发面板（s28_less +5.2 / LBS +1.97 / P2 +3.5，均 36 行 development）——论文语言不允许把这级证据写成方法胜利。
3. **距离-能力曲线（Core-C）的前置缺陷**：证据距离定义需先按 GLM 复核重造（FWE 不成立、VT/MQ 逐链/逐子问题计分），否则机制主张会被同一批评点打回。
4. **E1 跨模型迁移混合**（7B 128K −1.667pp、OLMo −3.33/+3.82pp）：规则若要进论文，必须 (a) 以物理量 (l,h,S)/r_j/D_j 表述且 OLMo ScaleTaper 实测，或 (b) 明说模型特定（Qwen3B 开发面板）+ 如实保留负迁移。当前 (b) 是唯一诚实口径。
5. **Cosh 论文侧能力缺口**（COSH_REDESIGN §4/§6）：8B "PPL 改善 vs 真实问答退化（11.26% 宏 F1）"未进论文；如果统一理论并入导致审稿聚焦 8B 段，这是暴露面。是否补一句范围限定与 09-09"禁止防御性写作"裁决存在张力——留给作者决定。
6. **budget_*.tex 残段（prop:finite-budget 有限预算命题 + 因果权重 Gram）**未并入当前稿也未删除：它是统一理论"守恒预算"在论文坐标系里最接近的现成数学，需要裁决复活或归档。
7. **6Pro 校准验证**（pro_block_calibration，32K 全文档采集 + 固定态 KL + 方向匹配控制）当日仅"计划与已开始的采集"；其结果决定"冻结兼容成本符号预测"能否成为 claim (b)（UNIFIED §8.4）。
8. **摘要 09-17/09-18 冻结 vs 全文 09-25**：统一方案最早可能进全文窗口的条件是 holdout 在 ~09-20 前回来（GPU 恢复时间未知）；摘要不得含未支持结果（Gate A 明文），所以摘要策略 = 现在锁 Cosh+BM 口径，128K 结果只在全文里以"若成立才写"的方式处理。
9. sparse-attention 讨论段与 PC2 线（另一研究目标）的关系：论文 discussion 已预告方向，但 PC2 审计表明该线无可用主张——引用论文"future direction"措辞时不得暗示已有证据。

---

## 8. 并入判定与最小改动路径（本 digest 的任务产出）

### 8.1 哪些统一理论内容**现在**可并入（不需 0450/0451）——全部为定义/恒等式/负结果/范围限定

1. **守恒预算的水床路径形式（U3 弱化版）**：端点与 m∈[0,1] 单调固定后，"额外对数跨度是放置问题"。零风险，因为它就是论文 eq:(movement-allocation)+eq:(adjustment-families) 的求和改写；**必须避开 GLM 已推翻的两个措辞**（"一个自由度"→ 16 自由度；"ln S = 17 个 gap 之和"→ 只对增量和成立）。落点：§6.1 一句 remark 或 A7 首段；可顺带把 Uni/Pro/BM 三行写成同一 Σε 约束下的三个 ε 分布（论文 A7 已有此推导雏形——BM 的 min Σ(Δε)² 已是"桥内预算放置"的一个变分选择，**与 Cosh 的 C_app 显式构成同一方法论的两端，训练/冻结两段因此真正闭环**）。
2. **MrPro 的缺陷 = N=17 线性均布放置 + bank 边缘压缩**（U6 的事实半部）：m_q=q(q+1)/[N(N+1)] 论文已作为他方方法写出；把它读作"过渡带宽最大化的放置点"是描述性重述。落点：§6.2/§6.3 一段。
3. **EVQ source 反转的负证据（U11/HighGapToLong）**：冻结模型上"抽高频给慢端"36 行 0 提升。对论文价值双向：(a) 在 §4.3 Cosh 段防误读（训练期准则 ≠ 冻结期手术）；(b) 支撑冻结段"native-relative"的动机。属负结果+机制限定，可写 [部分证据：单一开发面板负例，写明条件即可]。
4. **"4× 窗口 +10.19% 对数跨度"（U12）**：一句话量级直觉（"贵的是放置不是跨度"），纯算术。落点：§6 开头或 intro 第二段。
5. **limits 三连（U15 部分）**：三带非定理、见圆≠充分、logsumexp 竞争压——与论文两静态反例、obstruction 定理同族，可作 Discussion 1–2 句或 A1 remark；强化"我们没主张三带机制"的防性边界但不构成防御性堆叠（NARRATIVE_GUIDE 允许"necessary scope beside the governed claim"）。
6. **softmax 混合频率（U14）与联合弱子空间（U13）**：数学完备、有 CPU 演示，但**目前只能以"未闭合的解释缺口"身份进 Discussion/A1**——论文正文主张均为线性子空间量，Bessel 谐波表明位置对象在 softmax 后不止线性谱；作为"下一层理论"点到即止。若占篇幅超一句，暂缓。
7. **不能现在并入的**：U4/U5/U7/U8 的机制叙事（bank/arc 二分法、危险带 75–112K、方向梯度、LBS=完成手术）——GLM 同日复核各打掉其一个核心支点；U9 距离曲线；任何"N′ 更优 / 桥变窄单调改善"（0448/0449 未跑）。统一文档自己的判定成立："这五个（0446/0448/0449/0450/0451）正是 Core-A/B/C 的验证件……是把'方向一致的梯度'升级为'论文级证据'的最后一步。"

novelty 面：三件威胁引用（Data-Shapes 2607.07678= wu2026datashapes、LeRoPE 2607.10134 = karypis2026lerope、Clipped RoPE 2602.05258 = li2026copeclipped）**已在论文 bib 与 02_related 中** [已验证]，并入时只需保证相关段落把"数据驱动尺度匹配 vs 冻结守恒放置"的区分句保留，无需新增条目。

### 8.2 必须等 0450/0451（及 0446/0448/0449）的 claim 清单

| 候选 claim | 所需证据 | 现有最强读数（等级） |
|---|---|---|
| 单槽 bank 保真修复（s28_less +5.2pp@128K） | 0450（multikey/multiquery/VT/QA ×16 新样本） | 36 行开发面板，2 升 0 降 [部分证据] |
| P2 长端优势（+3.5pp@128K、PPL −2.65%） | 0451（VT/QA16 新样本） | 开发面板 + 4 文档 tail-512 PPL [部分证据] |
| 双机制可加（StackFrontBack 128K ≥83 且 32K ≥80） | 0446 | 无（未跑） |
| 单参数窄桥族单调梯度（论文核心图候选） | 0448/0449 | 无（未跑）；判定规则已预注册 [好实践，须在附录引用] |
| 距离-准确率机制曲线（危险带因果验证） | Core-C holdout 批 + 距离定义重造 | 36 行混质信号 [假设] |
| 跨模型规则表述（"bank 恒等+危险区完成+最小桥"普适性） | OLMo ScaleTaper 实测 + 7B 复测 | 几何检查通过；E1 迁移混合 [假设→待测] |

### 8.3 统一方案若发表：最小改动路径（按页预算与 9 页硬约束排序）

前提：0446/0448/0449/0450/0451 至少支持一条可发表主线（推荐以 **N′ 族或 Stack**为方法主张，s28_less/LBS 为机制件；若只有单件成立，则以 §8.4 备选口径）。

1. **Figure 1 改造（净增 0）**：把 §6 的 `fig_bm_exponent_profiles`（已是 Uni/Pro/BM 三相 cumulative displacement 曲线）扩为**相图版**：横轴槽位/log-周期、纵轴 m 或周期(log)，画 bank/桥/÷S 三区域 + MrPro/BM/N′/YaRN 的放置曲线 + 128K 参考线。UNIFIED §7 的设计意图（"一眼看到这是一张相图，所有前人是相图上的点"）零新图实现。正文图 3 张可压缩。
2. **§6.2 末 + §6.3（04_mature.tex）加一段（~10 行 TeX）+ 一张 5 行小表**：新臂（N′16/15 或 Stack）与 s28/LBS 在 dev+holdout 六任务的 32K/128K 表；句式沿用现有 tab:bm-models caption 的匹配声明（"Both methods use static s=4…"），扩为 tab:allocation-protocols 新行。**删除/压缩对象**：§6.2 的 Gemma 段（保数字删两句散文，细节已在 A7:coordinate-confirmation）与 Qwen-0.5B 段的重复协议句。
3. **摘要一处最小改写（09-17 冻结前决定）**：现摘要冻结段结尾停在 OLMo 自然 QA，对 Qwen 128K 只字未提（intro 有"favor a different intermediate profile"句）。若 holdout 成立：intro 该句改为"…a narrower-bridge placement derived from the same budget identity gives Qwen-3B improvements at both tested lengths（+X.X pp at 128K on newly generated inputs）"；摘要**可不动**（摘要本就不含 Qwen 数字，Gate A 风险最低）。
4. **A7 增两小节（附录不占页数）**：(a) N′ 族公式 m_q=q(q+1)/[N′(N′+1)] 与完成槽定义、箱/单调约束、预注册判定规则；(b) 冻结队列回执（0446–0451 的输入 manifest、SHA、seeds）——沿用 REVISION_BRIEF "state the actual computation and comparison identity" 与 r04 处置 6 的外部版本指针惯例。
5. **§4.3/Discussion 各一句**（§8.1 项 1+3 的并入）：C_app 与 BM 的 ε-平滑变分写成同一"指定准则→闭式放置"的两例；加 HighGapToLong 负结果一句钉住"训练期密度直觉不可移植到冻结表"。
6. **不做的**：不动 main.tex 结构与定理编号；不复活 budget_*.tex 整段（只取 prop:finite-budget 若审稿人问有限预算下界，作为 A1 remark 一行）；不把 U5/U6 机制语言写进正文（最多 Discussion 的 hypothesis 从句，带"as a hypothesis"字样）；不加防御性免责堆叠（r04 作者禁令）。
7. **流程 gate**：合并任何文字后必须 (i) `bash paper-2027/compile.sh` 过 9 页闸；(ii) 重生成 `exponent-allocation-source.zip` + package_verification（上次验证记录：五图像素一致/四表一致/几何偏差 3.55e-14）；(iii) 从当前 PDF 重开剩余审稿轮次或正式记录"r05–r10 终止于 09-10 理论冲刺"于 HANDOFF——否则 09-17 冻结文档与实况继续背离。

**若 holdout 失败/缺席（GPU 到 09-25 未回）**：执行纯 §8.1 版（零新结果、零新主张）——水床路径形式一句 + EVQ-source 负结果一句 + limits 从句两处 + A7 的 Uni/Pro/BM/N′ 关系段（N′ 族可定义而不测，标注 "untested placement variants of the same family"——**不得**写成方法主张）。该版本论文当前声明全部保持 [已验证/部分证据] 原状，统一理论以"设计空间读法"身份并入，与 93d7eac→b8466de 已冻结的主线零冲突。
