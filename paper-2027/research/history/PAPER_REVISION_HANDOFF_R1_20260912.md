# 论文修订交接总档（给改稿代理的单一入口）

**建立：2026-09-11 · 范围：全仓库（2026-02 → 2026-09-11，约 10 个月）· 用途：ICLR 2027 稿修订时免重探仓库**

> **用法**：改稿前只读本文档 + 它指向的"权威文件"。本文档是地图与判读，不是第四权威——
> 每个数字以它标注的来源文件为准；冲突时按 §0.2 权威顺序裁决。
> **纪律**：审计/验收步骤除本文档外不写任何文件；改稿只改 `paper-2027/` 下源码（`verify_explicit_geometry.py` 复跑会刷新 `figs/explicit_geometry_examples.json`，属授权范围）；数字必须可溯源到 §8 的索引。
> **执行者身份**：本仓库分工纪律为 Codex 改稿、Claude 只审稿/验证（`paper-2027/claude_code_workspace/README.md`:7–8）；本文档面向的"改稿代理"默认=Codex。

---

## R. 修订记录

**R1（2026-09-12）：20 个只读子代理全库对照审计后的合并修订。** 逐处修改均注证据路径；审计中确认无误的数字不再罗列。要点：

计数口径（R1 终检）：实错修正 **28 处**、高价值补录 **约 30 项**、措辞/守卫强化 **约 22 处**——逐条证据内联在对应小节，下表为抽样。

| 类 | 处 | 代表项（证据） |
|---|---|---|
| 事实错误修正 | 28 | "十轮 Sol 评审"→实际 r01–r04（`pdf-review-rounds/20260909/` 实物+分支速览 L28）；T-amp"内点 t≥5.6"→t≥2.2（`verdicts/DOSE_RESULT` §一）；Qwen 迁移行 128K/32K 归属（`QWEN3_SURVIVOR_RESULT` §一）；`FULL_PAPER_INTEGRITY_AUDIT`"不在库"→git 可取（`git show main:rebuttal/pre_rebuttal/…`）；T₁"须改"→现稿已是修正形（`appendix/a1_proofs.tex`:478 数值复算）；0.45n_int 系数无 owner 删（COVERAGE §5.3 原文）；"16× 重设振幅"→S 预算重设（EVIDENCE §2.2）；§7.2 五身份指针→EVIDENCE §1.2；454M legacy 路径→`schedules.py`；交叉矩阵/2.00013 三处来源指针（foundations §5.2/§209）；`validation_readout.json` 与 3 个 `run_*/` 仅服务器（`llama3_60m/history_registry.jsonl`:20 [ABSENT]）；verdicts 29→28；"0% EOS"→仅 BM 臂（S8_RESULT §三）；2 月"4 份"→5 份文件；§5.2/§5.3 出处改挂 TONIGHT；"phase1 全部就绪"矛盾修+9/12 凌晨阻塞解除实测（ssh 核验 3087467144 字节、device_count=1） |
| 高价值补录 | ~30 | Algorithm 1 盲测失败+softplus 死区（`docs/exp/2026-02/2026-02-24_128tok_baseline_report.md`）；GQA/MLA 压缩消融 Tier 0；阶段① provenance 断链警示与 git 恢复路径；THEORY_STANCE_CONSOLIDATED/ADVERSARIAL_REVIEW 两份在库判决文档登记；8B 非同量化器强制句；τ=d/√L 正面辩护；BM 归档重跑 24/350 caveat；falsification_benchmark 仅存 main_0726；"YaRN 顶点"→MrRoPE-Uni；(0,0) 四角 stale 清单；NeurIPS 结局库内零记录；§8 补 6 行 owner |
| 措辞/守卫强化 | ~22 | "非单调"限定到具体网格；跨面板拼接限定；RATIO 全距 1.43–37.18；"任何频率表"；T3 标定区 [9,15]；T-maxis 覆盖区间解读；0.30%→0.29%；canonical EVQ 臂身份统一；§12 末条与 §9 死锁解开 |

（原 R0=2026-09-11 建档。）

---

## 0. 阅读规则

### 0.1 改稿的最小必读集

| 层 | 文件 | 回答什么 |
|---|---|---|
| 本文档 | `paper-2027/research/PAPER_REVISION_HANDOFF_20260911.md` | 四阶段全部资产与判决的策展 |
| 论文现状 | `paper-2027/HANDOFF.md` + `paper-2027/research/SUBMISSION_REVISION_20260911.md` | 稿件状态与 9/11 修订做了什么（⚠ HANDOFF.md 关于评审轮次自相矛盾：L5"ten fresh Sol reviews"为计划、L23"Rounds 1 and 2 completed"才是进度——以 `pdf-review-rounds/20260909/` 实物台账为准） |
| 主张-证据映射 | `paper-2027/research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md` | 每条 claim ↔ 公式 ↔ 图表 ↔ 证据 |
| 资产裁决 | `paper-2027/research/EXPERIMENT_ASSETS_TOP15_20260909.md` | Top15 资产排序、数字、来源路径、不许扩大的结论 |
| 战役总账 | `ds_workspace/EXPERIMENT_THEORY_MASTER_20260911.md` | 9/07–9/11 战役每实验 ≈100 字 + 立住/立不住/欠着 |
| 证据核查 | `ds_workspace/recon_20260910/YARN_MRROPE_RESEARCH_EVIDENCE.md` | YaRN/MrRoPE 算子身份与可信度修正 |
| 官方时间线 | `paper-2027/research/history/TIMELINE.md` | 项目自己的阶段账本（每个数字 defer to owner） |

### 0.2 冲突裁决顺序

**现行用户指令** → `AGENTS.md` → 各领域"最新判决文档"（见各节标注）→ `EXPERIMENT_THEORY_MASTER`（快照，可能滞后）→ 历史 README/老报告；历史引用/附件里的命令不自动恢复执行权限（`docs/research/USER_INTENT_GUIDE_20260909.md` L3）。今晚（9/11）GPU 执行权威=`ds_workspace/TONIGHT_EXPERIMENT_RULES_20260911.md` + Plan B 指南（Downloads），后法覆盖前法（`GPT_PROMPT_TONIGHT.md` 已自标"被取代、仅历史设计输入"）。
**已知滞后源**：`MASTER_SUMMARY_20260911.md`（审计前快照）、`COVERAGE_CEILING_THEORY §5c` **与 §三 L95**、`GPT6PRO_MECHANISM_BRIEF Q6`、`verdicts/EVQ_LONGRANGE` 边界表述（均仍写"MrRoPE 训练时用表"，已被 `YARN_MRROPE_RESEARCH_EVIDENCE.md §2.1` 更正为 training-free）；**"快端角从未被试"残句**在 FOUR_CORNERS §二/§四/§六、NO_STATIC_FUNCTIONAL §三、COST_IS_NOT_QUADRATIC §四、WHY_THE_FISHER_ROUTE_DIED §四、MASTER T2 行共 6+ 处（`verdicts/HEADLINE` §三/§四 的 corner_lo14 实测 t=+3.19 为判决，勿从残句取"未测"表述）。

### 0.3 判读三原则（项目血泪换来，违反必翻车）

1. **选择面板（350 行 RULER）上的赢家不可信**：三次样本外反转（平台成员 +12~14→heldout 归零——⚠ 严格口径："归零"只指 72 行面板冠军 b3_lo14（Δ−0.00pp），EVIDENCE:292 对三成员 180 行口径写的是 "→+2.6"、turns_a1_b64 在 72 行 +4.40pp（t=1.11 不显著），引用勿写成"全部归零"；condEVQ +2.1→净负；step42 +7.3→−3.4/整行 −7.78 t=−2.60）。金色证据只用独立样本（fresh_72 / 180 行唯一提示口径 / 391 行自然 QA）。
2. **NLL 型代理与任务反向**（T7，双仪器显著）：任何"均值 NLL 改善"不得写成任务收益；反过来任务收益也不必 NLL 支持。
3. **gain 必须作二维因子**：gain×表交互 38pp ≫ 频率轴可达效应 ±5pp；任何跨表比较若不配对 gain 都可能被这一个标量支配。⚠ 拼接口径（`verdicts/GAIN_TABLE_2x2_FINAL` §五.2 自己声明）：+38.19pp 测于 350 行面板、−19.9/+18.4pp 测于 180 行面板，**不是同一面板、严格不能相乘对照**；180 行 native 信号主要来自 60 行 @4096。

---

## 1. 四阶段总览

> 项目问题演进（`TIMELINE.md` 原文归纳）：
> **τ/方法搜索 →（阶段一）→ 固定支持分配识别 →（阶段二）→ 表/权重共适应 → 成熟检查点自然生成迁移 →（阶段二末）→ 形状 vs 增益 / "超越 MrRoPE" →（阶段三）→ 跨模型泛化与终稿 →（阶段四）**

| 阶段 | 时间窗 | 目标 | 主战场 | 结局 |
|---|---|---|---|---|
| **① EVQ 起源与 NeurIPS** | 2026-02-24 ~ 2026-07-22 | 闭式非均匀有限频率表（EVQ-Cosh）能否改善外推；τ 能否解析选定 | 50M→1.485B 训练线、432M MLA、8B LoRA、rebuttal 战术室 | NeurIPS 投稿 ~borderline——**分数仅用户口述（3/6 量级），库内无任何真实评审/决定记录**（`AUTHOR_RESPONSE_20260722.md` 从未入库；勿与 ICLR 内审 3/6 混，见 `research/EXPONENT_SECOND_REVIEW_20260909.md`）；内部审计砍掉 Cosh 变分唯一性等过强主张；方法定位改为"第三个 PE 设计轴" |
| **② z-分配主线（Beyond the Base）** | 2026-07-23 ~ 2026-09-06 | 只动归一化内部指数 z（固定端点）的因果识别；冻结成熟模型迁移 | 151.9M 三种子 exact-range、交叉矩阵、Gemma/Qwen/OLMo 冻结干预 | 因果核心建立（3/3 种子同向）+ 共适应机制 + ICLR 稿重建（十轮 Sol 盲评为**计划**；实际有评审记录仅 r01–r04，r05 有快照无评审、r06–r10 未跑——`paper-2027/research/pdf-review-rounds/20260909/` 实物（r05 仅 identity.json+paper.pdf）、`docs/research/USER_INTENT_GUIDE_20260909.md` 分支速览段 L28、`analysis/unify_20260910/digests/digest_thread-0909-am.md`:108 明文禁令） |
| **③ MrRoPE 战役** | 2026-09-07 ~ 2026-09-11 | 零训练打赢 MrRoPE-Pro（README 阶段目标，后被判为框架错误） | OLMo-1B 冻结 350/180/72 行 RULER + 连续 NLL，数十个表臂（唯一显式合计=`_reports/LEDGER_20260911.md` §7：71 臂-测量，口径含 Qwen 61；OLMo 散臂另加约 80–95 量级，无单一审批出处） | BM 族四块独立证据立住；形状扫描的样本外泛化三连败；gain 轴 +38pp 成最大杠杆；理论 T3/T4 建立并被边界化 |
| **④ Llama-3 泛化与终稿** | 2026-09-11 ~ | 冻结规则在 Llama-3-8B 上锁定并跨族泛化；ICLR 终稿 | llama3_60dir/llama3_60m 两实现包 + Plan B 队列 | 写作时（9/11）未跑 GPU（无卡+权重缺失阻塞；**9/12 凌晨只读 ssh 核验：safetensors 已落盘且字节精确匹配、device_count=1，两项阻塞均解除，实验仍未跑**，见 §5.1）；论文 9/11 修订完成（去虚 stub、重写摘要/贡献） |

**分支地图（被遮盖的历史在这里）**：

| 分支 | 内容 | 何时看 |
|---|---|---|
| `main` | NeurIPS 时代正典分支（顶端 rebuttal: 前缀**五连**提交+1 条正典化 docs 提交（`485514f`），`8c425f4` 止，07-27）；`09_09` 完全包含它 | 查 NeurIPS 期证据链 |
| `main_0726`（只读） | 精简前的旧 NeurIPS 稿与全部历史材料（tip `6bca5ab`，9/06），`git show main_0726:<path>` 查阅；**大量被 09_09 瘦身删除的原件在此**（`falsification_benchmark/`、旧 `paper/` 树、`research_notes/FABLE5_*` 与 `opus5_readout_decomposition/`、curated 5 件） | 查被删段落与旧数字 |
| `main_0726_09_06` | 9/06 分叉、内容至 9/08（tip `3dc3527`）：含 3 次傍晚 hourly workspace snapshot + "官方 3B BM smoke + 128K 失败诊断"（`0177e6d`） | 查 9 月初过渡态 |
| `backup/main-restored-paper-20260726` | 恢复的投稿基线（tip `7754486`） | 查投稿时点状态 |
| `09_09`（当前） | main 之后 +352 commits（`git rev-list --count main..09_09`=352 已核）；关键瘦身 commit：**6b636e5**（9/06，删 paper//data 部分/research_notes 等）、**875a8be**（8/25，删 REPO_MAP.md）、**093a605**（7/24，rebuttal 控制室移位/删） | 全部当前工作 |

---

## 2. 阶段①：EVQ 起源与 NeurIPS（2026-02-24 ~ 07-22）

### 2.1 起点（2 月）：τ 是有限网格操作先验，不是连续最优证书

50M/125M 有限 EVQ-Cosh τ 网格从头训练（`docs/exp/2026-02/`，5 份 .md/4 组实验：`128tok_baseline`(report+results)、`phase6`、`full_experiment`、`evq_tau_sweep`）。
**发现**：50M 八点扫描呈锯齿（τ=0.4 −0.1%→0.6 +14.0%→1.5 −10.9%→2.0 +7.0%，`evq_tau_sweep_results.md`；⚠ "非单调"只对该网格成立——phase6/128tok/full_experiment 在各自 τ 轴上报**单调下降**，最强非单调证据在 3 月 PHASE22_23）；有用点不是从小 τ 平滑延拓出来的（扰动带 τ∈[0.2,1.0] 无一优于 baseline，Claim 4）；跨种子方向比效应量更稳（seed42 −18.9% vs seed137 −5.8%，方向 6/6 一致）。
**同月两个必带教训**（`2026-02-24_128tok_baseline_report.md` §2/§9.2、:61/:123/:259）：**Algorithm 1 盲测失败**——broadband 分解残差 35.6%→48%，α∝1/n_grid 离散化伪影致 τ*=40.96 发散，降级为理论动机、由 3-5 点 mini-sweep 取代（这是"τ 非连续最优证书"论点的直接起源，审稿必问）；**softplus 死区**——learnable τ init=0.01 塌到 0.003（sigmoid(ψ)≈0 梯度饿死），协议"init τ≥0.5"，回应"为什么不直接学 τ"。另：waterbed 不等式被 125M 双种子反驳（2K/16K 同改善，`evq_tau_sweep_results.md` Claim 2/3）；"−10.9%→−18.9% 随规模放大"声明与 3 月 Phase18b/19（1B tokens 下 GEO 胜）有张力，引用须带训练量限定。
**判读变化**：τ 从"连续最优"降为"可错的有限网格工作点"——这条降级是后续所有 τ 声明的祖先。

### 2.2 3 月：规模、任务、组合与"失败转换"（`docs/exp/2026-03/`，18 份）

| 资产 | 数字 | 来源 | 现状 |
|---|---|---|---|
| **432M MLA 三种子**（EVQ-Cosh vs Geo，d_rope=32,K=16） | 窗内 PPL +0.9%；16K **−31.1%**；EVQ+YaRN 16K **−48.8%** / 32K −26.9%；优势 50% 训练进度即出现 | `results/350m_mla32_evq_report.md`（2026-03-21）+ `data/curated/table18_mla_3seed_aggregate.json` | 论文 Top15 #5 的前身；⚠ 论文现用口径 138.8→95.6 来自 `data/curated/table18_mla_3seed_aggregate.json`（三种子均值） |
| 750M 续训（2K→4K，500M tokens） | 16K PPL 45.1→24.4；passkey AR exact 0/40→31/40 | `data/curated/phase15_750m_continue_result_20260306.json` | Top15 #8 |
| 454M EVQ×YaRN 四臂（10% passkey mix） | 同 R8 后 16K PPL 157.7→107.5；teacher-forced PK 61±3→100±0% | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | Top15 #10；⚠ 算子身份是 repo fixed-index smooth-ramp（`scripts/lib/rope/schedules.py`:236-247 `repo_fixed_ramp`，20%–90% smoothstep；`official_yarn.py` 是官方公式的钉扎实现，引用须带函数名），**不是官方 YaRN**，论文 A2 已如实命名；⚠ "100±0%"是 8K 饱和天花板口径——151.9M 上另有 PK 度量反转（EVQ teacher-forced PK 46.0% vs Geo 80.67%，`rebuttal/ADVERSARIAL_REVIEW_FINDINGS_20260720.md` E3，引用必并排） |
| GQA/MLA 125M 压缩消融 | KV 压缩越激进 EVQ 增益越大（MLA-16 仅 8 频率 −47.8%@8K）——本表判读句"杠杆依赖 substrate"的直接出处 | `docs/exp/2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md`（自标 "Tier 0 — 论文核心数据"） | 历史证据；数字未走 curated，引用前按 §9 找 owner |
| Phase16-23（formula sweep 99 run、τ 扫描、MLA 架构对齐） | 旧 MLA 配置频率覆盖不足致 EVQ pattern 反转；对齐 DeepSeek-V2/V3 标准（d_rope=64,K=32,base=1e4）后 τ=2.5 恢复 | `results/PHASE18_YARN_FT_REPORT.md`、`PHASE19_TAU1_vs_GEO_REPORT.md`、`PHASE22_23_MLA_TAU_SWEEP_REPORT.md` | 历史证据；τ 公式 `x(L,b)=1−ln(L/2π)/ln(b)` 与 `d/√L` 的争论见 §2.4 Q2；⚠ 99-run 的 top-2/top-3 排名已被 7/24 同一原始记录重分析纠正（更正在 `git show main_0726:rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`） |
| Video DiT（129.6M，32→128 frames） | 远端去噪 MSE −35%（单种子） | `paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md` | 跨模态附录（Top15 之外） |

**本阶段判读变化**：长 PPL / teacher-forced 检索 / 自回归 exact / 下游 QA **互不自动转换**；同一 range 算子的杠杆依赖 substrate（MHA vs MLA vs 稀缺通道）。

### 2.3 4–6 月：投稿加固与 provenance 修复

`docs/overview/`（`RESULT_PROVENANCE_MANIFEST.md` 为数字 provenance 最高权威；另有 `EXPERIMENT_INVENTORY/REGISTRY`、`PAPER_CLAIMS_MAP`、`METHODOLOGY`）。
**教训（TIMELINE 原文）**：多个历史标签夸大了实现身份或种子范围；"计划/脚本/检查点清单/论文行"都不是结果所有者。

### 2.4 7 月：NeurIPS rebuttal 战术室（`rebuttal/`，入口 `rebuttal/README.md`）

统一立场（README 原文）：**"EVQ-Cosh 把 training-time frequency allocation 作为 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。这不是 universal long-context SOTA。"**

**⚠ 阶段① provenance 断链（R1 审计发现，引用任何阶段①数字前先读）**：(a) E1 的权威源 `rebuttal/NATIVE_ROPE_EVQ_150M_500M_RESULT_20260713.md` 与其 curated JSON 均已被瘦身提交删出工作树（`git show 093a605:rebuttal/pre_rebuttal/NATIVE_ROPE_EVQ_150M_500M_RESULT_20260713.md` 可取回；数字经 git 历史核对一致，在库旁证=`docs/exp/2026-07/2026-07-14_repo_fixed_ramp_mechanism_probe.md`:42/:45 的 −0.3293/−0.4127）；(b) `data/curated/` 中被 6b636e5 删除的阶段① owner 共 5 件（native_rope_evq_150m、primary1、fig3_extreme_128、eval_3seeds_full、phase16_99run_manifest），`main` 分支仍完整（`git show main:data/curated/<file>`），而 `RESULT_PROVENANCE_MANIFEST.md`:60-71 仍带 SHA 指向它们；(c) E3/E6/E7/E8 的 raw 目录从未入库（REVIEW §2.3 自记 provenance gap）；(d) rebuttal 控制室 10+ 文件删于 093a605（`git show 093a605^:rebuttal/` 可取），`rebuttal/README.md` 导航 13 条中 11 条悬空；(e) 判决文档里的 `a1_proofs.tex:行号` 全部指**旧 NeurIPS paper/ 树**（`main_0726:paper/appendix/`），与当前 `paper-2027/appendix/` 同名不同物。

**理论审计结论（三份文件交叉：`EXPERIMENT_THEORY_REVIEW_20260720.md` §1、`STRONG_MODEL_THEORY_VERDICT_20260720.md` Q1–Q8；引用的 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 已删出工作树但在库可取（`git show main:rebuttal/pre_rebuttal/FULL_PAPER_INTEGRITY_AUDIT_20260713.md`）——结论仍以上述两份在库文件为据）**：

| 审计项 | 裁决 | 对改稿的含义 |
|---|---|---|
| Q1 Cosh 定理内容量 | **成立（须降级）**：surrogate 的 Green-kernel 结构使任何 (α,β) 拟合只能输出 cosh——"拟合家族≈cosh"零证据量；exact-kernel 二次型自己的最优是**截断型**（L=2048 置零 15 通道），不是 cosh | 定理只能写成"闭式可解性/表示结果"；RoPE 特异支持必须来自 surrogate 之外的检验 |
| Q2 τ_surr vs τ_deploy | **成立**：τ_surr=√(β/α)≈6.24/5.70 vs 部署 1.41/1.00（4.4×/5.7×；⚠ 配置上下文必带：这是 **d=64、b=500K、L=2048/4096 的理论网格**，非 151.9M/50M 训练档；支撑数 ‖ρ_surr−ρ_dep‖₁≈0.90/0.93、EVQ@τ_surr 降幅 97–100%>部署点 24–93%）；无 gauge 能把 √(β/α) 变成 d/√L | `d/√L` 只能作"参考标度"，但 VERDICT §9.2 给了最强可辩护写法 "proxy-motivated, coarsely d-and-L-consistent empirical default"（三支撑：条件化 stationarity、Phase16 九配置 c∈[0.75,1.5] 且 τ=0 从不是最优、d-sweep 粗支持 d¹ 排除 √d）——正面材料，不止禁止面；`1−ln(L/2π)/ln b` 禁令系本档自 Phase18/19 外推（两源字面只裁 d/√L）；全部判决数字可在库脚本 `rebuttal/strong_model_verdict_numerics_20260720.py` 复算 |
| Q3/Q5(b)/Q6/Q7 | 部分成立（前提有误但残余问题真实） | 见 `STRONG_MODEL_THEORY_VERDICT` 行内 |
| Q4 "KL gain" 命名 | **成立（须删）**：ordinary KL 是 O(τ⁴)，τ² 项是线性相位方差 transport proxy，非 KL | 正文名不得出现 KL-gain 因果句 |
| Q8 24–92% collision 降低 | **成立（无鉴别力）**：最优指数单调密度 12/12 配置 ≈100% 降低（≥EVQ） | 不得作为 Cosh 形状支持引用 |
| `c_coll=1.171` 校准 | 无效：独立优化 argmin τ≈13.3（c≈4.7），论文点高出族内 collision 最小值 ~670×（=C 分数比 28.183/0.042≈671，**不是** c 值比——c 比≈4×） | c_coll 全撤；24–92% 按源文件 "What survives" **降级保留为 directional diagnostic**（VERDICT Q8 动作=改名限缩非删除），不得作形状/τ/surrogate 验证；存活件 c_pred=√(45Q₁)≈1.19 是真实 surrogate 计算、勿与 1.171 混；⚠ `scripts/analysis/verify_c_coll.py` 仍在库且 docstring 自称 "verification of c_coll=1.171"（循环验证：τ_coll 作输入；其引用的 `table_lambda_cv.tex` 已不存在）——勿复活 |
| 新发现 | `T₁` 闭式 typo（指 NeurIPS 旧稿 `main_0726:paper/appendix/a1_proofs.tex`:258；恒等式与 T₂ 不受影响）；collision-only 最优 τ≈13 与 trained-PPL basin（c≈1）发散——可转为诚实机制答案（collision 是冗余诊断非目标） | **已修**：现稿 `paper-2027/appendix/a1_proofs.tex`:478 已是 τ² 修正形（数值复算 τ=0.5/2 → 1.0013/1.1894 命中真值），只需防回退；VERDICT Q6 另有未登记新发现：验证表 video 行 d_eff=K 与 text 行 d_eff=2K 混用未披露（保留须逐行标 τ） |

**E1–E8 实验裁决（`EXPERIMENT_THEORY_REVIEW_20260720.md` §2 前言 :97-99 原话：全部 single-seed supporting、"none may be upgraded to primary"、"None modifies a paper number"）**：⚠ 编号冲突——`rebuttal/ADVERSARIAL_REVIEW_FINDINGS_20260720.md` 另有一套 E1–E10（全局析取闭合："EVQ allocation matters" 的 MHA/fixed-ramp/MLA/8B 四分支被自家数据全部关闭——§10.1 Cosh 去中心重排的原始论据；E6=印刷 8B 表隐藏 +47.7% matched in-distribution 回退；E9=τ* 规则不可证伪），两表 "E5/E7" 不同物，引用必须带文件名。

| # | 实验 | 结果 | 处置 |
|---|---|---|---|
| E1 | 151.9M/500M 六格（Native vs EVQ × raw/official-derived-YaRN/repo-fixed-ramp） | raw EVQ PPL −8.6/−14.3/−16.2%（4/8/16K）；official/derived YaRN 拉平两臂（−0.5/−0.3/−0.7%）；repo fixed-ramp 强负交互（EVQ-favoring −0.329@8K/−0.413@16K NLL） | 支持撤回"富通道 MHA 的 official-YaRN 互补"；fixed-ramp 互补仅 repo 局部；协议细节（2.0177% PK 混入训练、500M tok=3.29/param **不得称充分训练**）在 `ai-handoff.md` §0；owner 断链见上（provenance (a)） |
| E2 | MHA 四算子分解 | gap 塌缩由**频率校正**造成（freq_only gap→0.0055/0.0124/0.0330，移除 94/92/81% 原始 gap），非 mscale（mscale 侧 gap 保持 0.091/0.177/0.206） | E1 的机制归因；使用条件："仅 P1 official-YaRN 触发下引用"（REVIEW E2） |
| E3 | MLA K=16 四臂 | 稀缺通道下 substrate gap **挺过** full YaRN（freq_only +0.0492/+0.2389）；EVQ+full 71.55 vs Native+full 85.46 @8K | 稀缺通道方向的机制支持；**非互补恢复**；⚠ 必带 caveats：single seed、L_train=512、τ=1.414 未按 L=512 重推、PPL-only；被反驳臂（EVQ+mscale 519.1、EVQ-raw vs Native-full 412.0 vs 85.5）见 REVIEW §3.2 第 3 行 |
| E4 | repo fixed-ramp 机制探针 | 宽 `scale^r` smoothstep 承载几乎全部交互（16K −0.3917 of −0.4127）；official 线性 shared-index 控制**反转**排序（gap −0.1415） | 互补性特定于该 ramp；无新 novelty 声明 |
| E5 | seed-42 8B LoRA capability | EVQ+LoRA PPL 10.07/24.07/127.91 vs Geo 6.82/108.96/991.48 @8/16/32K（PPL owner=curated longalpaca/causal 两 JSON）；但 S-NIAH 56.67→3.33→0%、passkey 100→0→0%（**归零数字唯一在库 owner=REVIEW E5 行**，原始 REPORT.md 不在库） | **PPL 稳定 ≠ 检索能力**，两个事实必须绑定陈述；⚠ **REVIEW §2.3 强制句（每个 8B 句必带）**：8B pair 基底是 native-endpoint Geo vs midpoint 量化 EVQ（**非同量化器**），测的是预训练 native 的 conversion，非干净形状对比；⚠ TOP15 #2 只登记 PPL+ΔNLL——代价口径 owner 是 REVIEW §3.2 披露表，引 TOP15 不带 E5/E7 = 违规；另 official-YaRN ×2/×4 pilot 什么都没恢复 |
| E6 | 检索转换探针 + sparse pilot + 因果分解 | 50 步微调无转换（0%）；EVQ hit@16 中位 64.06% vs Geo 18.75%（10/10 配对胜）；sparse pilot 负（DiD −0.666）；gold-drop-all-heads +1.5055 NLL（×4.51 似然）vs Geo −0.0095≈0；首 token rank Geo 33,775 vs EVQ 2,043（中位数，注意臂序与单元格前文相反） | **最强新机制资产**；引用 hit@16/rank/causal 必须同行"0% exact、forced-gold inclusion 近零（−0.034）、sparse 负、chat-wrap/decode-parity 双 null"（REVIEW §3.2 第 2 行）；⚠ 旧 KV 生成 bug：名义 16K 实为 6,827 token——旧 Geo"95% KV@16K"**永久不可引用** |
| E7 | **QA16K 三臂（registered gate 负）** | task-macro F1：EVQ-LoRA 0.1126 < Native-LoRA 0.2110 < Base 0.2309；EVQ−Native=−0.0984 CI[−0.1297,−0.0697] | 谈 LoRA/下游必强制披露；杀死"该 adapter 是更好的 16K QA 模型"；分层：赤字集中 ≤8K（−0.334），>8K 全臂地板（+0.018 CI 跨 0）——"long-range substrate mechanism unresolved"（源文件自己的话） |
| E8 | Residual-RoPE pilot v6 | 10 步残差分支训练后三臂逐字节相同（Qasper F1 0.5534；passkey 亦三臂同：8K first-value 100%/strict 0%） | inert，不可作任何一方证据；0.5534 唯一在库 owner=REVIEW E8 行（原始 summary.json 不在库） |

**投稿后一周的在库 owner（07-26~29，晚于阶段①窗口终点 07-22，归属按"rebuttal 时代延伸"）**（`rebuttal/rebuttal_0723/theory_results/`）：M4 exact-range factorial（`m4_exact_range_factorial_evidence_20260726.json`，Top15 #7：12 配置×3 种子，**预指定 1.25×Cosh 臂**改善 10/12、deformation-matched Exp 9/12；cosh_rule 臂直测是 7/12——别把 10/12 安到它头上。TOP15 #7 原话"Cosh 与 Exp 差异**未分离**"——引用时用原话，"唯一性不成立"只能经 Q1"拟合家族≈cosh 零证据量"到达、不得挂 M4 原话）、1.485B OLMo released-rope（Top15 #13：16K PPL 182.73→159.64，126/128 docs 同向）、llama8b matched ruler mix（Top15 #15：16K Native/EVQ 0.29/14.03%；516 步独立 adapter）、OLMo selective Q/K 适配（Top15 #14：2Wiki 8K 0.07→21.48%、16K 0→8.57%）、`evq_query_gap_realized_eos32_20260728/`（4/8/16K AR Native 95/18/0% vs EVQ 100/98/60%；claim_boundaries：单训练 seed、同 numeric-NIAH 任务族非 clean transfer、16K 仍 distance-sensitive——TOP15 有、此前未索引）。
**两份未登记判决文档（R1 补录，§9 禁止清单只是前者子集）**：`rebuttal/THEORY_STANCE_CONSOLIDATED_20260720.md`——三层幸存集（每层比提交稿窄一档）+"No longer writable"7 条（含 ∫w/ρ²、"controlled forcing residual"、"½ factors absorbed into λ"、无条件 structural d_head）+ A1–A15 逐条改写台账（**现成合规英文替换段**）+ §3B"已让步但仍印在提交 PDF"披露目录（waterbed C_app[1]≠0、T₁ "verified<1e-15" 假陈述、YaRN "orthogonal/additive"×7、Primary II 超参失实、compute checklist、QuALITY 4K 反向）；`rebuttal/ADVERSARIAL_REVIEW_FINDINGS_20260720.md`（上段已注编号冲突；另 E2=base=100 全活通道下 **Geo 胜 EVQ ~20%**——"cosh 只在病态配置有益"的自家证据）。REVIEW §1.1 的 safe-to-defend 八项（Cosh 定理对 stated surrogate Holds、min-kernel PSD、自洽恒等式、S_χ²+τ⁴/45、waterbed ∫M²=2τ⁴/945、c_pred≈1.19、floor 4√(N/K)、midpoint-Geo 因子 0.6636/0.8146/0.9026）与 §3.1 三 gate 叙事（signal preservation PASS-causal / addressing PASS-directional / readout FAIL-registered）是**可正面引用**资产。Z0 readout-conversion 线"分析落地前不得引用"维持（`docs/exp/2026-07/2026-07-15_*plan.md`；persisted logits 不存在）；其悬案后被 `main_0726:research_notes/opus5_readout_decomposition/` 关闭（瓶颈=模型从不进入 answer mode，非证据衰减——oracle 诊断、单 seed、5 case，不可升 paper claim）。

**阶段①被审计否决、永久不可用**：
- 28 条 direct-hybrid 零分收据 = Native/EVQ in-place buffer alias 产物——对 partial-pair/blend/per-head hybrid **无效证据**，无重跑（TIMELINE:72-74）；
- 旧 Geo "95% KV@16K"（E6 bug；REVIEW:111 "must never be cited"）；
- Phase16 "27 配置全 <1%" 的旧表述（安全表述=99-run/9-config、7/9 胜 2/9 负；⚠ 旧实物仍在 `docs/tau_algor/TAU_SCALING_DERIVATION.md`:224 等，历史文档非 owner）；
- SFT Λ₀ 恢复曲线（70%@5 与 95%@10 不同 Λ₀=154.5 vs 2380，自不一致，REVIEW:63-70 手算复核确认）→ REMOVE_FROM_DEFENSE；
- 论文行 `0.61-turn/−0.046 NLL`、`−0.023 NLL/−10.6 task` 等单槽机制叙述（9/11 修订已删，找不到原始测量记录；保留 SUBMISSION_REVISION:20 原句"**不据此认定这些实验从未发生**"。⚠ 被删的 −0.023/−10.6 与 T7 立住的 −0.0234/−10.56pp 数值极近但**不同对象不同 owner**——严禁互相顶替或当彼此的复活依据）。

---

## 3. 阶段②：z-分配主线 "Beyond the Base"（2026-07-23 ~ 09-06）

### 3.1 理论重构（8/19）：有限谱基 + 训练共适应

`paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`（ICLR 核心理论的 canonical internal report）四点结论：
1. cosine-only collision kernel 遗漏每 pair 的完整二维 span{cos,sin}；白化 cross-Gram canonical correlations 给相位不变冗余度量，与 block-whitened stable rank 有**严格恒等式**（正文 Eq. r_2=2K/(1+(K−1)c̄)）；
2. geometric 极慢带存在真实 spectral collapse：普通度量收敛 span{1,Δ}；softmax 度量去常数后为 centered span{Δ−EΔ, Δ²−EΔ²}；
3. 最大化静态 rank/logdet 产生近 Fourier harmonic comb 而非多尺度表（collision 与 full rank 排序**相反**的反例已构造：cosine 序反转 C_cos=2.62e−9<1.44e−8 而 r_2=64.48<128.00；另有 L↔2L/4L 第二反例与 Fourier 表 L 外精确 aliasing 论证）⟹ collision reduction 不蕴含外推改善；
4. 50M 2×2 weights-by-table counterfactual：LM loss 主要由 **table×weights 交互**决定（I_{T×W}=−3.5367，CI[−5.165,−3.039]，约为两主效应 5.9 倍；⚠ 报告 §6 限定：受限 scalar-base geometric control 能恢复大部分 frozen mismatch，但非训练期归因）。
**定位句（改稿可直接用）**："EVQ-Cosh 应降为闭式、零学习参数的 constructive instance，而不是通用最优解。"

### 3.2 因果核心（8/19–20）：151.9M 三种子 exact-range

`paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.{json,md}`
- 协议：B=W=256、K=32、每臂 499,974,144 tokens、seeds 42/137/256、固定端点 (a,R) 只动 30 个内部指数。
- 结果：固定范围 Cosh−Geo NLL 在 512/1K/2K 为 **−0.281/−0.176/−0.146**（9/9 seed×length 同向）；256 处 +0.026。（⚠ owner 命名是 "Cosh minus **FMRoPE/uniform**"——基线实体即标准几何网格 base-256，"Geo" 实质不错，但入稿沿用 owner 名，防与同实验族 `paper_geo_base500k` 臂混淆。）
- 部署分析：runtime range retargeting 使排序**反转**（target-matched +0.060/+0.227/+0.460、0/3 favor Cosh）；权重×表交叉矩阵两组——**50M PPL [[7.14,76.20],[23.05,7.16]] 的 owner 是 §3.1 foundations 报告 §5.2**（其数值 JSON 只写到 /tmp 未入库；行=weights/列=runtime table）；**151.9M tail-NLL [[3.426,5.776],[4.455,3.479]] 的 owner 是 `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`:small_model_crossing.mean_tail_nll_two_seed_length1024**（@1024、seeds 137/256 两种子均值——引用必带口径）。两组数都不在 EXACT_RANGE 文件里。
- ✅ 数学校验【已闭合 2026-09-11；原警告登记于 `external-reviews/pro-materials-20260908/RoPE_ICLR2027_Major_Revision_20260906.md` §5.2】：核心实验 min ωW = 256·256^(−31/32) = **1.1892 > 1**（无通道 ωW≤1），慢频渐近确实不直接覆盖；但**有限窗精确计算已补齐**——b=256,K=32,L=256 最慢 8 对（ωW∈[1.19,4]）块白化 r₂=**2.1147**（最慢 4 对 2.0036；b=500K 锚点复现 2.00013），塌缩是**有限窗子空间重叠现象**，ωL→0 渐近只是其极端情形。定义/脚本/全曲线见 `paper-2027/research/evidence/FINITE_WINDOW_SLOW_RANK_RECEIPT_20260911.md`；`03_theory.tex` slow-limit 段已插入对应句。改稿引用一律用 finite-window overlap 措辞，不得退回"ωL≪1 才塌缩"或把渐近当作核心实验的解释。

### 3.3 成熟检查点干预阶梯（8/21–9/03，owner 在 `paper-2027/research/attention-aware-retrofit/`；⚠ 例外：8/21–26 行的"Video DiT 确认"owner 在 `research/evidence/` + `data/curated/video_dit_*.json`，见 §2.2）

| 日期 | 内容 | 判决 |
|---|---|---|
| 8/21–26 | phase-chord from-training 控制、released-model residual/table 干预、同支持 Qwen/OLMo 控制、fresh FineWeb NLL、静态单表门、共适应恢复、剂量响应、Video DiT 确认 | 纯同支持 z 改变冻结成熟行为；NLL-only 正结果 ≠ 能力/部署正结果；**表/权重兼容性与端点特异读出支配静态几何分数**（TIMELINE L111-112 另一半判读） |
| 8/27–31 | 预注册 success-first 组合（后于 8/31 RETIRED）+ scale-consistent exponent 表 + gain 控制 + 跨检查点 transport + **同多重集置换** | 一张静态表通过 OLMo 1× 门并改善更长端点、长能力可转移到 Qwen（仅 long-capability transfer，无 Qwen 1× 门）；**保频率多重集、只置换内部槽指派（端点/多重集/gain/seed 保持）⟹ 决定性崩溃** ⟹ 成熟对象是"学习到的旋转子空间×频率/伸缩"的**有序配对**，不是无序谱；⚠ TIMELINE 同条第二句必带："一张成功表仍是一个检查点/协议的结果，不是普适定律"；SCALE_CONSISTENT §8.5：证的是**非可交换性**，不是唯一性/最优性，不授权 64 维搜索表 |
| 9/01 | Qwen K32/K64、Gemma K128 交叉确认；参考长度协议修复 | normalized index 是最佳已测（best-supported）**且非 canonical/非普适**的 transport 坐标（K128 上 +6.19pp CI[2.81,9.63]）；K 与 checkpoint 共变，**无 K 因果**；⚠ 旧 Gemma 16K 零分用了错误 8K 参考长度（config 上限误作参考；已修正为 validated 4K 并取代旧天花板解读） |
| 9/02 | 38 行"16K Hotpot" stress **判定无效**（四条理由：选择短正确行、机械尾部边界、非官方 filler 分布、无 raw owner）；headwise 探索 report-only；第一性 retrofit memo 的 T4/T5/T7 **未过审计** | 门控与构型效度是解读前提 |
| 9/03 | Native-isotonic 与 Selective-31 负结果（⚠ 严格口径：Selective-31 纯负；Native-isotonic 是 **endpoint tradeoff**——natural likelihood/retention 更好、fresh core-4 4K/8K 更差，勿写成纯负；"distribution-free" 措辞仅 TIMELINE 有、具体定理文件用 universal exact-no-harm，入稿前回到定理陈述核字面） | 完成-历史选择保留 64 槽 legacy-u p2 mask + log-s4 + c=.074（owner：`theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md`；⚠ 该 c=.074 与战役 gain 1.1386=0.1·ln4+1 是两个系数：.074 给 g*=1+0.074·log4=1.102585782722872） |

### 3.4 论文重建（9/06–09/09）

- 分支 `main_0726_09_06`（含 `0177e6d` 官方 3B BM smoke + 128K 失败诊断，分叉于 9/06、内容至 9/08）→ 9/06–09/09 **重建跨阶段②/③窗口**："从既有证据完整重建稿件 + Sol 盲评**计划十轮、实际落地 r01–r04**（r05 有 immutable PDF 快照、无 review.md；r06–r10 未跑）"（实物=`research/pdf-review-rounds/20260909/`；`HANDOFF.md` L5 与 L23 自身矛盾，不得引"十轮完成"；digest 明文禁令在 `analysis/unify_20260910/digests/digest_thread-0909-am.md`:108）。
- 资产裁决 `EXPERIMENT_ASSETS_TOP15_20260909.md`（15 项资产，含完整数字与来源路径，§8 索引引用之；⚠ 其 #2"数值来源"里 `pre_rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` 是死链，真实 owner=`data/curated/lora_longalpaca_temporal_s42_20260712.json`）；
- 主张-证据映射 `EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md`（公式/图表契约/文献关系，含 FMRoPE 链接）；
- 交叉审计 `external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md`；BM 闭式来源 `external-reviews/MRPRO_BOUNDARY_MATCHED_SOURCE_20260908.md`（外部 review 给出最小粗糙度闭式 m\*=3x²−2x³；⚠ 源文件用词 "minimum-gap-roughness"，"Dirichlet" 系本档归纳词——两处对齐，入稿择一）；
- 审计存档 `research/audits/`（ICLR2027 全稿只读评审、引用新颖性审计 8/26、叙事反思 8/21 等）。

### 3.5 阶段②的 Top15 资产速查（全文见 TOP15 文档；此处为改稿取数入口）

1. 151.9M 三种子固定范围（§3.2）；2. Llama-3-8B 匹配适配（8/16/32K PPL 6.82/108.96/991.48→10.07/24.07/127.91 + 来源干预 ΔNLL −0.0095/+1.5055；`data/curated/llama8b_causal_source_use_s42_20260714.json`；**TOP15 #2 未含代价，引用必配 E5/E7 负面与 §2.4 非同量化器句**）；3. 成熟冻结同范围对照（OLMo held-out 9×20：geo/ramp/residual 0.56/61.04/60.47%；⚠ "residual" 是别名，owner 臂名=derived/corrected_derived；`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`，共同 curated=`data/curated/frozen_fixed_support_mature_20260823.json`）；4. Qwen-0.5B full13（64K index/YaRN 51.46/45.37%，+6.09pp CI[2.76,9.58]——两臂振幅 1.0513/1.0693 不同，系 table–amplitude **联合**对比、非纯 shape；`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json`）；5. 432M MLA 三种子（8K 35.4/35.8 → 16K **138.8/95.6**；另 16K YaRN(s=4) 对 117.9→71.1；`table18_mla_3seed_aggregate.json`）；6. OLMo BM 五任务自然 QA（778 inputs/631 长输入，F1 21.62→25.44，+3.82pp CI[1.32,6.29]；`docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`）；7–15 见 TOP15 文档（M4 factorial、750M、交叉、454M、三模型匹配、Gemma K128、1.485B、selective-QK、8B RULER-mix）。

---

## 4. 阶段③：MrRoPE 战役（2026-09-07 ~ 09-11）

> 阶段目标"零训练打赢 MrRoPE-Pro"来自 `README.md` 阶段节，后被判为**框架错误**（novelty 不冲突，见 §10.3）；但战役顺手产出的资产大部分独立成立。

### 4.1 算子与身份（`YARN_MRROPE_RESEARCH_EVIDENCE.md` §1，改稿引用任何"YaRN/MrRoPE"前必读）

| 对象 | 定义 | ε 方向 | OLMo sum_m |
|---|---|---|---|
| 官方 YaRN 代码（jquesnelle@995db5b） | 通道索引 ramp `t=(j−lo)/(hi−lo)` | **递增（后置）** | 38.4918 |
| 论文 A.2.1 YaRN（MrRoPE 叙述中的 "regressive"） | 转数混合 | λ 递减（**前置**） | 42.9117 |
| MrRoPE-Uni（Eq.13） | ε 均匀 ⟺ m 线性 | 均匀 | 40.50 |
| MrRoPE-Pro（Eq.14） | ε_j=2(1+j−dl)/((1+dh−dl)(dh−dl)) 等差；本地等价 `m_q=q(q+1)/(n(n+1))` | 递增（更后置） | 37.6667 |
| BM（本项目自创，非文献方法） | ε∝k(n+1−k)，Dirichlet 最小粗糙度；`scripts/lib/rope/boundary_matched.py` | 帐形（中置） | 40.50 |

关键事实：MrRoPE 原论文（arXiv 2601.22181）是 **training-free**（§4.1 no fine-tuning）——与我方零训练同设定；其报告边际温和（PPL 0.03–0.3；RULER LLaMA3 128K 86.6 vs 79.9；Qwen2.5-3B 53.2 vs 50.1，⚠ 其"上界 RoPE Bound 1K→28K vs YaRN 6K"两个数是恒等计算的上界、**我方未复算**，引用须标）；最大收益格在窗内（LLaMA2@4K 5.72 vs 6.02）；16× 靠**预算重设**（S 按目标重设，慢端 m=log₄16=2——"振幅"一词在本档保留给 gain，§7.6）。作者代码从未获得，全部 MrRoPE 臂为本地复现（公式已对原文逐条核对）。
**三条必带身份注记（R1 补录，owner=EVIDENCE §1.2/§1.6/§2.2）**：(i) 论文 Eq.19–25 的证明依赖转数几何恒等式 r_j²=r_{j+1}r_{j−1}，**只对论文转数混合版成立、不描述官方代码**——引论文对 YaRN 的一切定性先查这条适用性；(ii) FOUR_CORNERS 旧标"(0,0) YaRN 顶点"已更正为 **MrRoPE-Uni**（commit 92c39a9；所有"YaRN 顶点"引用改读 Uni，真 YaRN 前置 OLMo 零训练从未单独测、P5 未跑）；(iii) **归档 BM 与 9/11 同名重跑差 24/350 分数、210/350 文本**（表字典 SHA 同、重建张量 20 槽差 ≤5.96e-8）——引 0.4167 等 350 行归档数必须带"旧文本非精确重放目标"；(iv) 带边界规则三家相同（j=head_dim·ln(W/(turns·2π))/(2lnθ)，floor/ceil），但**论文记法 α=32/β=1 系我方 (1,32) 的交换**；Qwen 频带 (23,40)/n=17 与论文附录 B 逐位一致。sum_m 的 owner=EV 附录 B 命令 #1 `formula_audit`（§8 已登记）。

### 4.2 立住的实测（金色证据，可入论文）

| 结论 | 数字 | 来源 |
|---|---|---|
| BM ≫ MrRoPE-Pro（冻结 OLMo@16K，四块独立；⚠ 上游 EVIDENCE §4.1 同主张记为"三块面板+两仪器"，NLL 算仪器不算块——引用时统一口径） | 72 行 51.32 vs 2.78；350 行 .4167 vs .0709（引归档值带 §4.1 注记(iii) 的 24/350 重跑差异 caveat）；NLL 16/16 篇 2.862 vs 3.688（Δ−0.826 CI[−1.08,−0.62]）；fresh_72 .4823 vs .1354（−34.7pp se 4.8） | 在库：`docs/research/ROPE_OLMO_BM_RESULT_20260908.{json,md}`（72 行权威实体）、`work/jsonl/archive/`（350 行，审计代理已逐行重算 0.416714/0.070857 吻合）、`results/olmo_fast_screen_20260908/run_nll_01/`（NLL，已重算吻合）+ `docs/research/ROPE_OLMO_BM_NLL_RESULT_20260908.json`；**仅服务器**：fresh_72 原始 `validation_readout.json`（在 `/root/autodl-tmp/rope_decision_20260911/`；FRESH72 声称的本地副本被 `experiments/llama3_60m/history_registry.jsonl`:20 记 [ABSENT]+CONFLICT——fresh_72 数字的本地依据=EVIDENCE:144/216 旁证+表内算术，无逐行数据） |
| 官方 YaRN ≈ Pro（勘误级） | 72 行 6.94；fresh_72 yarn_index .1406 vs mrpro .1354（−0.5pp se 1.4） | 72 行在库实体=`docs/research/ROPE_OLMO_BM_RESULT_20260908.json` existing_controls（0.0694444 已核）；原引 `run_controls_01/` **本地已缺失**（EVIDENCE:141 的"本地+服务器"声明过期）；fresh_72 仅服务器 |
| gain 轴：YaRN 解析 mscale=0.1·ln4+1 实测最优 | 1.0/1.05/1.10/1.1386 → .0371/.2559/.4104/.4190（+38.19pp t=15.05）；1.20 未跑完（59→83/350 两处记录=GSWEEP 与 FRESH72 时点差，非矛盾） | `verdicts/GSWEEP_RESULT`（在库）；`olmo_gain/`、`olmo_gsweep/` 为**服务器**目录（/root/autodl-tmp/phase1_20260910/；本地同步副本=`work/jsonl/gain/`；gsweep 连服务器都无 rows.jsonl） |
| gain×表乘法交互 | 180 行非地板面板：表效应 @g1.0 **−19.9pp** ↔ @gY **+18.4pp**；gain 对 native **+0.10pp（t=0.08，非地板）**；350 行上 native 双 gain 全 0 是地板不可读 | `verdicts/GAIN_TABLE_2x2_FINAL` |
| 宽带族长度交易（三块独立） | ① holdout180 唯一提示口径 b4wide 16K **+7.92pp（SE 3.81）**/4K −6.16（3.45）；② holdout180 **全提示**分数值口径 +9.61/4K −4.5~−6.4（第三块，`verdicts/NLL_VS_TASK` §八，EVIDENCE:226 的"三块 +7.9~+9.6"即指此）；③ fresh_72 16K +9.1（se 4.7，仅 EVIDENCE:226 区间旁证）/4K +2.4（se 3.0，与①②的 4K 端符号不一致须并排报） | `audit/pro_decision_20260911/check_results.json.unique_holdout`、NLL_VS_TASK §八、fresh_72（仅服务器） |
| NLL-任务反向律（T7，双仪器） | 走线窗内 NLL −0.0234（t=−2.88）↔ 4096 整行 **−10.56pp（t=−2.45）**；三张独立构造表排序一致；1025 token 处中性（−0.0008）⟹ 长上下文效应 | `verdicts/NLL_VS_TASK` |
| 选择面板无效（三次反转 + 量化） | 任何**频率**表翻动 15–20% 行（gain 翻 39.5% 不在此区间）；净方向面板决定；RATIO：gain 1.03 vs 频率 2.65–37.18（⚠ 系 SYNTHESIS 表口径；DPATTERN 全量还有 b3_lo14 1.43、wide_b4 1.64 两点，全距应 1.43–37.18） | `verdicts/DPATTERN`、`theory/SYNTHESIS_20260911.md`（注意在 theory/）、`verdicts/STEP42_RESULT` |
| 8× 振幅盒子 | 32768 上 BM/b3/wide_b1/wide_b4 全 **0/48**（复读式退化；EOS：BM 臂 0%、wide_b1 实测 **4%**——"全 0%EOS"系源文件行文与自身表格矛盾的夸大，已修）；4096 侧 0.66–0.76（⚠ 区间出处=EVIDENCE:193 汇总，S8_RESULT 只显式给 0.734/0.734/0.740，两端点原始数据在服务器 s8_out） | `verdicts/S8_RESULT`（+EVIDENCE:193） |
| 同谱置换崩溃 | 保频率多重集只置换**内部**槽指派（端点/多重集/gain 保持） ⟹ OLMo tail NLL 3.10423→6.86493；Qwen 64K 0.7000→0.0000 | owner=`attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`（L132/:274/:175/:186；论文落点 `sections/03_theory.tex`:46 只是转述处，非数字 owner） |
| C42/C42V24（唯一跨仪器二阶效应） | 同 S=42 差 **+10.73pp（t=5.47）** + NLL −0.1109（t=−3.91）；Var(ε) 族内 R²=0.009 ⟹ 受控内因果、族内不可推广 | `verdicts/HEADLINE` §二 |
| fresh_72 校准判决 | decision_calibrated @16K 38.85%（vs 初始化 b4wide **−18.44pp** CI[−29.34,−7.53]）；gain_only vs 初始化 −0.69pp CI[−6.04,+4.65]、vs BM +8.37pp（se 5.3） | `verdicts/FRESH72_COMPLETED` |

### 4.3 立不住/已关闭（改稿严禁再引用为支持）

| 假设 | 否定证据 |
|---|---|
| S=Σm 是操作变量 | C42 对同 S 差 10.73pp |
| 12 个静态低维表泛函 | 三层受控对全灭（`NO_STATIC_FUNCTIONAL`） |
| Fisher 二次代价 | 符号错（F₂₄=7842 恒正 vs 实测 ΔNLL −0.0150） |
| 逐槽梯度选表 | SNR<1 |
| NLL 代理选表 | T7 反向 |
| 频谱空洞一阶杀手 | step42/step25 同 7.76× 空洞（owner=`code/coverage_theory_results.json` gap_max=7.761），+7.3（STEP42_RESULT）vs −30.5pp（HEADLINE）——三数出处分散，勿当单文件可核 |
| 平滑必要论 | step42 选择面板赢 BM（但样本外反转——降级为"平滑非先验必要"） |
| EVQ/midpoint 位移零训练替换 | 旧 `evq_shift` 臂（= canonical 中点网格全槽 ×θ^{1/2K}=×1.108，身份见 §7.4 与 EVIDENCE:91/236——**三个 0 不代表 canonical EVQ**，literal canonical 网格臂只有构造、无任务面板结果，GPU 竞争失败未跑）τ=0.5/1/2 全 **0.0000**（W/L/T=0/174/176——非 bug，全零下胜负全由 BM 决定；−41.67pp，t=−16.47）；condEVQ +2.13pp t=1.11 净行 −1（owner=`verdicts/PRO_TABLES_RESULT`；从未跑 held-out）；本判决对从头训练 EVQ/EVQ+微调没有任何陈述（§9 禁止扩大） |
| 覆盖越大越好 | interp（m≡1）不可用 |
| OLMo 赢家跨模型迁移 | run_qwen3_02 无任何臂击败 MrRoPE：a1_b64T 主臂 Δ128K **−16.6pp**（NO_LONG_GAIN）；wide_b4T 次臂 Δ128K **+0.3pp 平手**、Δ32K −10.6pp（TRADEOFF）——旧版把 −10.6 错挂 128K，已按 `verdicts/QWEN3_SURVIVOR_RESULT` §一修正 |
| 65 参数任务校准可部署 | fresh_72 −18.44pp vs 自身初始化 |
| 慢端 ν 微调（LongBridge） | 面板 +11.83pp 但 65% 集中伪影任务；held-out slower−BM **−3.31pp（t=−1.47）**，预注册判据链失败 |
| 覆盖序跨模型迁移 | Qwen 分层 ρ=−0.147（n=14）/−0.473（n=13）；合并 +0.487 是 **Simpson 伪影**（`audit/verify_coverage_20260911/REPORT.md`） |
| **N（可读窗计数）是能力量** | 释放平台"六表"=5 新臂（rel_bm/rel_b3/rel_b4w/**Cslow**/Tstar，N=22–26）+ 部署 BM 锚点行（N=17）——源文件 §六标题与行数（非 BM 行实为 5）本身歧义，引用照数据表；五臂全部差于 BM（连续 NLL +0.005~+0.175）；native N=21 本身是反例（"全场最高"系 §四 写作时说法，后被 N=22–26 超越、反例逻辑不变）。**结构成立、能力性证伪**（`theory/RELEASE_AXIS` §六） |
| 快端缺角（四角第四顶点） | corner_lo14 连续仪器显著差于 BM（t=+3.19）（`verdicts/HEADLINE` §四） |
| "两模型族方向相反" | 36 行 SE≈13pp 无判别力 + 读数器符号错误已修 |
| KNIFE"平台不承重" | KNIFE −2.30pp；正确速率释放同样更差 ⟹ m=1 平台在慢端确实承重 |

### 4.4 理论资产现状（改稿引用时的措辞边界）

| 理论 | 状态 | 引用边界 |
|---|---|---|
| T1 m-坐标表代数 | ✅✅（与部署张量 bit 级核验） | 可作统一记账坐标；**不得**包装成"统一理论"卖点 |
| T2 Lε forcing/四角单纯形 | 结构机器精度验证；快端角已测负（⚠ theory/ 六处 stale"快端从未被试"残句见 §0.2 滞后源清单，不得取用） | "BM=常数 forcing、MrRoPE=**慢端**（末端）点源"可写（防与快端角 +0.111,0,…,0 混）；缺角方向已关；旧"(0,0) YaRN 顶点"已改读 MrRoPE-Uni（§4.1 注记(ii)） |
| T3 覆盖/天花板 | ✅✅ 族内（17 臂 rho 0.926 经 verify_coverage 独立复算吻合；8× 塌零半样本外；step 位置阶） | **必须带 "OLMo 族" 限定**（Qwen 分层 ρ 为负）；标定区限定（`audit/verify_coverage` §二）：单调趋势只在 n_int∈[9,15] 成立，**b4_wide 覆盖最高 18.26 却不是最高分**——上端不得外推 |
| T4 EVQ 可读窗条带 | 结构定理成立（[证]：条带宽 3、斜率 ≡ lnθ/(2K·ln2)=0.147903——**θ=5×10⁵、K=64，即 OLMo-2-0425-1B-Instruct 自身几何**（head_dim 128），非 θ=1e4/K=32；误设常识参数会算出 0.2076 而误判文档造假、五类槽分解）；**族内 11/11 排序是 [量] 弱证据**（11 点无重复、可重建 11/12，owner=`_reports/EVQ_LIMIT` §6.1/6.6——与结构定理分列） | N 作为能力量已证伪（六表=5 新臂+BM 锚点行，§4.3）；论文 a8 附录的 12 臂 N 拟合只能写"exploratory summary"，完整口径=−20.4532+1.19068·N、66 对中 39 对计数不同且同向、27 对平、0 反 |
| T5 Fisher / T6 带内免费 | T5 死；T6 被限定（"免费"仅 NLL 口径，任务口径有代价） | 引用 T6 必须带 T7 限定 |
| T7 NLL-任务反向律 | ✅✅ | 对全文献有效的仪器警告，可作方法学贡献 |
| T-maxis 一根轴 | 推导+穷举核验（EVQ m<0 / 三段式 m≥0；战役连续非退化覆盖 [−0.16,+1.00]——⚠ 此区间指有梯度信息的覆盖，EVQ 深负臂 m 至 −1.740 测过但全塌零无信息，引用带限定否则 ONE_AXIS 自己的表可戳） | "负方向必差"**未干净确立**（纯负样本仅 EVQ 族且有竞争解释（慢槽 OOD）；逐槽小幅加速从未单独测；HGL 是混合干预分不开） |
| T-amp 振幅处方 | 纯推导零实测；1.13 是 **reach-only 上界**（窗内覆盖随 m_p 单调降 28.04→23.83；reach 峰实为平台 ≈1.13–1.20；PREREG §七自证"最优是 1.13"措辞过强已撤回） | 唯一可写形态："真最优在 ≥1.0 一侧（E-dose：**NLL 口径**，族内单调、内点全部显著更差 **t≥+2.2**，端点 a=0 对比 \|t\|≥5.6——旧版"内点被 t≥5.6 拒绝"系 ONE_AXIS:59 讹挂，已按 `verdicts/DOSE_RESULT` §一/§三更正；a*≡1 不得搬去预测 RULER）"；判决在 amp4x 臂 |
| 振幅条件性 | 所有 ramp 结论是"m_p=1.0 下"的条件结论（含剂量曲线、C42 对照、平台成员排序、b 轴扫描——内部排序仍有效） | 论文若保留形状对比必须带此前缀 |

### 4.5 就绪未跑臂

汇总执行顺序见 §5.3（phase 1 A–E + 预注册 P5–P12 + EVQ_LIMIT W2p/W3）。

---

## 5. 阶段④：Llama-3 泛化与 9/11 终稿（2026-09-11 ~ ）

### 5.1 两个实现包（均未跑 GPU）

| 包 | 内容 | 状态 |
|---|---|---|
| `experiments/llama3_60dir_20260911/` | Plan B 实现：`operators.py`（60 配置 CPU 参考编译器，selftest **108/108** 恒等式过）、锚点核验（Llama 频带 `find_correction_range(32,1,128,5e5,8192)=(18,35)`（审计代理独立复算 ✓）、`m_mrpro` 与 curvature tables 位相同（独立复验 ✓）、gain 位相同（✓）、官方 YaRN 公式吻合 3.5e-18（README:60 自述，比对工件未归档））、`plan_arms.py` 收缩臂集、`phase1.py`（A–E 注册+preflight，**另注册 C2=CY8/CM8 强方法控制臂**）、`llama_runner.py` | **候选收缩 60 → RUN 19 / HOLD 5 / DROP 36；Llama=5 控制 + 16 候选 = 21 臂**（控制：Native/OfficialYaRN/MR/BM/ResonanceYaRN；候选：D01a、D03a/b、D04a、D05a/b/c、D06a/b/c、D07a/b/c、D13a/b/c；⚠ RUN19 与候选 16 差额 3=D12a/b/c，RUN 但不入 Llama 面板，见 `priority_correction_20260911.json` llama3_stage） |
| `experiments/llama3_60m/` | 60 规则实现（M01–M09 无 C 规则 + M10–M60），`selftest.py` 66/66 过（README 自述服务器实测，输出工件未入库），checkpoint 身份断言过（32 层/32 头/8 KV/d_head 128/θ=5e5/W=8192/无 rope_scaling；`core.py:184-187`+`selftest.py:57` 代码级证实） | 构造/适配/统计层已建；C 采集与 S/V/H 需卡 |

**判据修复（`execution_order.json`，比原计划强）**：
- F1：原 PASS 判据有反例洞（A(YARN)=10, A(MR)=8, A(M)=9 时 G_new>0 且 C_upgrade≥0 但 M 连 YaRN 未赢）⟹ binding gate 改为 **LCB[A(M) − max(A(MR),A(YARN),A(BM))] > 0**（已实现为 `test_baseline_hole`）；
- F2：Hero 门槛 3pp → **5pp**（3pp 记小改进；依据 SE≈4.56pp(S) 与"频率轴效应 ≲5pp 近地板"）。

**接线时发现的静默缺陷（`llama3_60dir_20260911/README.md` §C 列三条；`phase1.py` docstring 实列四条，R1 补全为五条）**：
1. `--m-file` 是普通 store 参数，重复传**只保留最后一个**——RUN_READY §B 的示例命令正是错误写法（1.13 臂会悄悄不跑；**`AMP4X_PRESCRIPTION_PREREG`:127-128 也是同样的双传错误写法**）；正确写法为单 flag 分号分隔（`phase1.py:98-105`、`patch_mfile_v2.py` split(";")）；
2. `--dry-run` 在 `olmo_beta.py:243` return，而 m-file 注入块在 :314 ⟹ dry-run **从不覆盖注入路径**，preflight 必须校验 argv 本身（⚠ 行号指**服务器已打补丁的 phase1 runner**：本地 zerotrain 副本 dry-run return 在 :200 且无 --m-file；audit 快照副本 :243 吻合但 :314 只在 v1 补丁后成立——两份本地副本都无法同时复核，勿拿本地文件对行号）；
3. 文档 PYTHONPATH 错误；可用根是**服务器路径** `/root/autodl-tmp/olmo_fast_screen_20260908/code`（`scripts.*`）+ `/root/autodl-tmp/nongeometric_screen_20260909/code`（`experiments.*`）——两个都是远端实例目录，仓库内对应物只有 `experiments/nongeometric_screen/`（无日期后缀），仓库 `results/` 下无 nongeometric 目录；
4. （phase1.py 第 3 条）`--m-file` 丢失 gain、与 `--gains` 互斥 ⟹ E-C 的 2×2（表×gain）不可表达——需 `patch_mgain.py` 加 `--m-gain`（phase1 C/C2 臂正因此注册了 `--m-gain`）；
5. （R1 新发现）`experiments/zerotrain_20260910/olmo_beta.py:361-366` 的 `--evq` 分支仍缺省 QWEN θ=1e6、**无 GEOMETRY GUARD**（带 `_evq_cfg=dict(OLMO)` 修正的只有 `ds_workspace/recon_20260910/audit/pro_decision_20260911/olmo_beta.py:477-486`）——今天从 zerotrain 副本重跑 `--evq` 会**静默复现 ×1.0528 mis-scaled EVQ**（实跑臂 sum_m 数值指纹证明当时跑的已是修正版）。

**preflight 阻塞（记录于 9/11；R1 审计更新）**：9/11 状态=`torch.cuda.device_count()=0`（无卡模式）+ `qwen25_1p5b_32k/model.safetensors` 缺失（期望 **3087467144** 字节；9/11 归档恢复时静默丢失，Birth 9/11 18:01 vs mtime 8/23 的 tar 恢复签名；恢复命令在 `RUN_READY` §A，落盘后按字节数核验，勿信返回码）。**9/12 凌晨只读 ssh 实测：两项均已解除**——safetensors 已在盘且 stat=3087467144 精确匹配（RUN_READY:155 记载的重下已完成），device_count=1（已开卡）。⚠ 引用的 RUN_READY §A / MASTER §8 / GPT_PROMPT 新内容均为**未提交改动**，切到已提交状态会对不上。

**Llama 运行器两条硬约定**：Q/K 配对布局是 **(i, i+K) half-split**（`rotate_half`），配错会得到能跑但不同的算子；D04/D05 的 `sum_m` **无定义**（负/零频率），运行器报 `m_coordinate_defined: false` 而非传 nan。

### 5.2 Plan B 主实验设计（主档：`/Users/yang/Downloads/RoPE_Integrated_Experiment_Guide_Codex_20260911.md` §5–6，本地 Downloads 不在仓库、sha256 已被 TONIGHT §0.1 钉扎；**今晚/纪律条款的 owner 是 `ds_workspace/TONIGHT_EXPERIMENT_RULES_20260911.md`**，R1 更正了出处错挂）

- **今晚只跑 Llama-3-8B-Instruct**（TONIGHT §0.5）；Qwen/Llama-2 仅在成功锁定规则后按 **TONIGHT §7** 泛化（各自重推槽号，禁硬抄；注意 Plan B 自己的 §7 是"原 20×60 的处理"、其泛化条款在 §10——引"§7"必带文件名）。**TONIGHT §0.5 同时把 Plan B §5 的 OLMo/Qwen 历史闭环移出今晚 GPU 队列**（"仅作背景与以后单独核查的资产"）——这是 §5.2"今晚只跑 Llama"与 §5.3"phase 1 先于 Llama"表面矛盾的调和依据（时序：execution_order 12:10 → Plan B 12:29 → TONIGHT 12:54，后法覆盖前法）。
- 历史闭环五项（E-A/E-B/E-C/E-E/E-G）：A=Qwen4x BM 臂（非任务迁移判决）；**B=amp4x 加左侧对照 a=0.87**（Δ+/Δ−/Δ_outer 三 contrast；"1.13 显著优"只证 tested point；全无显著差但 CI 允许 2–3pp 记 UNDERPOWERED）；C=8× 实际 scale×gain 2×2（C00/C10/C01/C11 + CY8/CM8 强方法控制；`4^{-1.5r}=8^{-r}` 无新 novelty）；E=官方 YaRN 同 panel（RULER 与自然 QA 两个 estimand 不混数）；G=table×gain 完整 4×2（I_{BM,MR} contrast）。
- Llama 主实验 L0–L7：L0 强基线+native 守门（controlled_geometry.csv 与 deployment_policy.csv **两张表分列**）；L1 形状×振幅 2×4（I_shape,a 交互）；L2 高频入口×低频平台边界联合（I_HL，⚠ 边界改变同时改 Σm，不能称独立槽身份效应）；L3 D03 同谱通道对应+gauge 负控制；L4 D05 慢尾 DC vs DROP vs SIGN（CPU 已核查：Llama j≥35 槽在 MR 下已是 ω/4，p≤4W 内 pν≤Wω，**不能预先解释成"超出训练弧段"**）；L5 D06 交错双尺度+AREA 总量控制（MR 中段 Σm=16/3，全表 34⅓ vs D06 全表 34 ⟹ 直接 D06−MR 混合了总量差）；L6 D07 谱增益（排除只是整体温度）；L7 D13 query 温度+静态 endpoint 控制。
- 纪律（owner=TONIGHT §3.2/§4/§5/§6.2–6.3，非 Plan B——Plan B 只有四元组且无 heartbeat/NEAR_MR）：MrRoPE-Pro 基线每"严格匹配合同"只跑一次并复用；每项写 requires/parent/statistic/decision/on_fail/next；heartbeat 15 分钟；NEAR_MR=±5pp 灰区须写机制明确的优化版提案；成功=赢全部登记强基线+守门+跨长度非劣（此条 Plan B §10 同有）。

### 5.3 执行顺序（`experiments/llama3_60dir_20260911/execution_order.json`）

- **phase 1（execution_order 原案排在 Llama 搜索前；"就绪"仅指协议/代码——9/11 写作时 A 臂权重与无卡两项阻塞执行，均 9/12 凌晨解除见 §5.1；且 TONIGHT §0.5 已把整个 OLMo/Qwen 闭环移出今晚队列）**：A=qwen4x BM 臂；B=amp4x（**振幅因子 a=1.13/1.30**——`m(s)=s·m_wideBM` 的 s 值，与冻结 gain=1.1386 是两个正交轴、数值近撞勿混；vs BM 锚点 0.4167；四行判据已写死于 `prereg_protocols/AMP4X_PRESCRIPTION_PREREG_20260911.md`，读数器 `amp4x_read.py` 已用发表数字自测 −34.59pp/t=−13.18、W/L/T=9/156/185——自测记录在 RUN_READY §F，非脚本内嵌）；C=scale8x_wide（8× 72 行；phase1.py 实际注册为 E-C 2×2 四格+C2 控制臂，粒度见 §5.2）；D=官方 YaRN same-panel（表条目已过约定守卫：反推 MrPro 得 sum_m=37.6667 已知签名，守卫曾以 1e-9 真拒绝过一次）；E=g2x2 的 MR/b3 列补完（18/350 孤儿）。
- **phase 2**：Llama 21 臂（§5.1）。
- **phase 3**：⚠ 出处拆分（R1 更正）——execution_order 的 phase_3 原文是"**剩余 GPU 才扩展 60 库**"（本档此前未登记）；"泛化确认（Qwen/Llama-2 各自重建边界/profile/振幅/gain）"出自 **TONIGHT §7**。
- ⚠ RUN_READY（`ds_workspace/recon_20260910/RUN_READY_20260911.md`）的字母编号与 execution_order 不同（RUN_READY: A=qwen4x/B=amp4x/C=amp8x/D=续跑/E=YaRN 基线，另有 §F 读数器）；引用时以内容为准，并注意 §5.1 的接线缺陷修正了 RUN_READY 的 B 命令。
- 另有未跑预注册：P5（Uni≈BM 0.32–0.48；prereg 标"未排队"）、P6（step 阶梯——prereg 实际未跑臂=**hi19/20/21**；hi22/25 已测、hi23/24 属 SATPRO P11，旧版"hi19–24"沿 EVIDENCE:242 的宽写法）、P7（Qwen 检索式仪器）、P8（OLMo-2-7B 剥规模）、P9–P12（P9 盲预测已被 fresh_72 部分超越：yarn_index 0.1406 落在预测区间 [0.46,0.60] 外；⚠ 严格判分需"选择面板+holdout 双段"、待决策，且 **P9 臂=论文 A.2.1 转数混合（前置，Σm=42.91）、fresh_72 的 yarn_index=官方索引（后置，Σm=38.49）——两个不同算子**，"落在区间外"是跨算子比较，见 EVIDENCE:242/SATPRO §2）；EVQ_LIMIT 的 W2p/W3 臂（**先验已低**：Cslow 同族已在连续仪器证伪（2.8878 vs BM 2.8627），W2p 与 MrPro 只差 Σm=0.168/0.4%、Σm 机制预测在噪声内不可判，只在 RULER 任务面板上仍有微弱判决价值）。

### 5.4 论文 9/11 修订（`SUBMISSION_REVISION_20260911.md`，改稿的基线状态）

已完成：重写摘要/引言/贡献/结论（主线=固定支持训练、范围重设、权重-表交叉、实际方法收益）；Figure 1 合并四联图；§4.3 混合"不可能性定理"替换为置换补偿+已测干预；锚定 Cosh 有限表性质（逆 CDF 凸性 ⟹ z_τ,k ≤ k/(K−1)）；§6.4 只留可证明内容；432M midpoint 与 750M anchored 协议分列；+6.09 明确为 table–amplitude 联合配置。
已删除/纠正：全部 `not yet run`/planned 残留；Fisher/Hessian 条件数、slot-19 最敏感方向等无记录断言；0.61-turn 等单槽机制叙述；orbit-jitter 修正为 −0.000013 CI[−0.003682,+0.003715]；12 臂逐行重算（−20.4532+1.19068N；39/66 计数不同且同向——**不得称完整排序预测**）；AI-use 声明去全称自证。
验证状态：`bash paper-2027/compile.sh` 通过（正文 9 页、总 41 页、0 未定义引用、0 overfull）；`verify_explicit_geometry.py` 8 项最大差 2.71e−14；`verify_profile_diagnostics.py` 12 表重建一致；源码包 `exponent-allocation-source.zip` 62 文件 SHA256 清单过。

---

## 6. 旁线与被遮盖历史（不进正文，但改稿可能被问到）

### 6.1 旁线实验目录（`experiments/`，按目的归类；证据状态以各自 README 为准）

| 目录 | 目的（首行自述） | 与主线关系 |
|---|---|---|
| `curvature_20260910/` | 表代数参考实现（m 坐标、全部构造器、verify CLI） | **战役基础设施**，表公式的权威源 |
| `zerotrain_20260910/` | OLMo runner（olmo_beta.py；350/72/180/391 面板；--m-file 经 **`patch_mfile_v2.py`**——v1 在 `ds_workspace/recon_20260910/code/patch_mfile.py`，v2 文件头明言"v1 CORRUPTED the runner … must not be used"，禁用；另注意 zerotrain 副本 `--evq` 仍带未修几何缺省，见 §5.1 缺陷 5） | 战役基础设施 |
| `rope_decision_20260911/` | 新算子（官方索引 vs 转数 YaRN、literal EVQ 两版）+ fresh_72 + read_validation.py | 决策校准线 |
| `nongeometric_screen/` | Qwen 26 臂筛选 → 汇入 `analysis/unify_20260910/tables/ground_truth_tables.json` 38 方法（**原始 26 臂 results 只在远端服务器，仓库/各分支均无 `results/nongeometric_screen_20260909/`**） | 提供全家族表；E1_s28/s29 族配对 \|t\|≤1.23（⚠ 若读成"该筛选全部臂"就继承了源 verdict 的宽断言错误——`verdicts/HISTORY_HAS_CANDIDATES_20260911.md` 自己的表里 MrUni/HighGap/E3_BM_gain1 达 \|t\|=1.61–1.84，只是同样不显著） |
| `analysis/…`（根 `analysis/unify_20260910/`；同级还有未被此前登记的 `analysis/kkt_20260910/`（推导过程件）与 `analysis/p0_gradients/`（仅 full_model_response.jsonl）） | 9/10 两线 62 代理统一推导工作台：KKT 问题模板、四问推导 D1–D4、T2/T3、V×4 终审席（math/veto/predictions/tables）、G1 地面真值 38 表、13 候选表 | 理论历史；**注意**：其推导结论后来被战役实测部分推翻（如 N 机制），引用需回到 §4.4 现状 |
| `joint_kkt_20260910/`、`kld_v2/`、`twotrack_20260911/` | KKT 边界/接受-拒绝设计；KLD v2 审查修正推导+最小实验；EVQ 与 YaRN/MrRoPE 两线工作台 | 推导期产物，非证据 owner |
| `evq_recovery/` | LoRA+真实长输入监督能否恢复 EVQ 能力；Cosh vs 非几何网格 | 阶段①延伸（开卡前准备态） |
| `native_rope_evq_150m/`、`rotary_budget/`、`position_observability/`、`position_overnight/`、`nosa_position/`、`native_sparse_position/`、`deepseek_mini_position/`、`broad_position_eval/`、`pm_keep/`、`refcarry_audit/`、`rope_operator_family/` | 8–9 月旁线：Native/EVQ 151.9M 六格、谱预算理论与核心实验、核心效应搜索、过夜线、稀疏位置、全向量混合 vs 乘积包络、DeepSeek-mini、跨文本分布真实生成、PM 保持、RefCarry 审查、算子族压缩（attention 输出蒸馏归因对照） | 与指数分配主线**正交**；TIMELINE/MASTER 均标注"证据未重核"；不得进正文主张，可在 discussion 引作独立研究资产 |
| `llama3_60dir_20260911/`、`llama3_60m/` | 见 §5.1 | 阶段④主战场（未跑） |

### 6.2 早期月度档案（`docs/exp/2026-02|03|04|07/`）

阶段①②的原始月度报告（2 月起源 **5 份 .md（4 组实验**，128tok_baseline 占 report+results 两份**）**、3 月 18 份、4 月 1 份（仅 thinking-token 计划文档，无结果）、7 月 10 份（7 份对应 E1–E7 + 3 份计划：Z0 readout 线悬置"落地前不得引用"、128K 工业计划=阶段④ 7 月祖先、M4 计划）；无 2026-05/06/08/09 目录）；正文数字不得直接从这些历史报告取——一律走 §8 索引的 curated/owner 文件（TIMELINE 明确："a plan, script, checkpoint inventory, or paper row is not a result owner"；⚠ 阶段①部分 curated owner 已删出工作树，恢复路径见 §2.4 provenance 断链）。

### 6.3 被遮盖/被否决历史速查（审稿人或未来会话可能翻出）

| 项 | 状态 | 出处 |
|---|---|---|
| 28 条 direct-hybrid 零分收据 | buffer alias 产物，无效 | TIMELINE 2026-07 "Corrected" |
| 旧 Geo 95% KV@16K | KV 生成 bug（名义 16K 实为 6,827 tok），永久不可引 | E6 |
| 38 行"16K Hotpot" stress | 效度审计无效 | TIMELINE 9/02 |
| Gemma 16K 零分天花板 | 参考长度用错（8K→4K），被取代 | TIMELINE 9/01 |
| 第一性 retrofit memo T4/T5/T7 | 未过审计，working history | TIMELINE 9/02 |
| "YaRN 递减 vs MrPro 递增" | 禁用表述（只对论文转数变体成立） | unify `STARTING_POINT` F8 + EVIDENCE §1.2 |
| n_int 与 Σm 族内共线 | rho 0.926 vs 0.924（旧行首的"0.45"系数**全仓库无 owner**，R1 删——COVERAGE §5.3 原文只有两个 rho） | COVERAGE §5.3 |
| MRCR/reference_position、minicpm41、supporting_video/cross_model、core_text、legacy | 旁线数据/历史权重隔离区（staged_training 是单报告文件非目录） | `results/` 各目录；防误引点：`results/legacy/phase14c_REPORT.txt`（旧 fig2 前身，已被 454M Primary I 取代）、`legacy/paper_ready/`（advisor 镜像副本） |
| `falsification_benchmark/`（RoPE 理论证伪盲测基准：16 episodes、visible/hidden 分离、leakage audit、确定性打分器） | 9/02 交付、**9/06 被瘦身整删，仅存 main_0726**（`git show main_0726:falsification_benchmark/README.md`）；TIMELINE:156 的链接已断 | 审稿问"理论做过盲测证伪吗"的唯一资产 |
| rebuttal 控制室 10+ 文件（FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT/THEORY_FREQUENCY_OPTIMALITY_20260716、REVIEWER_TRIAGE_PLAYBOOK、simulated_reviews/ 等）与阶段① curated 5 件 | 分别删于 093a605（7/24）、6b636e5（9/06）；git 历史可取 | 恢复命令见 §2.4 provenance 断链段；rebuttal/README 与 ai-handoff 的导航引用（含 REPO_MAP.md，删于 875a8be）大半悬空 |
| `scripts/analysis/verify_c_coll.py` | 审计前产物（2026-04）、**循环验证**（τ_coll 作输入、从不优化 collision 分数）、自称 "verification of c_coll=1.171"——其引用的 `table_lambda_cv.tex` 已不存在 | 勿据其复活 c_coll（裁决全文见 §2.4 c_coll 行） |
| 外部模拟评审材料（`git show 093a605^:rebuttal/simulated_reviews/`、raw_sources/01/02 panel、05 校准） | 均为**模拟/预测**，非真实审稿记录；learnable-τ 437.9 输给闭式的"训练损失对外推收益不可见"论点是可转正论材料（raw_sources/04，local-only） | 身份标注防误当真分数（README 已裁决仅作压力测试） |

### 6.4 用户意图纪律（`docs/research/USER_INTENT_GUIDE_20260909.md`，95 条消息归纳）

改稿代理最易违反的六条（R1 由三条扩为六条）：交付实际问题解法（审查/诊断/计划不是结果）；真实生成主张不能被 attention 图/局部 proxy/仅 NLL 替代；独立判断外部 AI（Pro/其他会话）输入——材料是证据与候选，按原文、**实际代码**与当前结果核验；**外部已发表数字若条件不同只作参考、不伪装成配对胜利**（GUIDE L12——直接管住 MrRoPE/LeRoPE/FMRoPE 数字的用法，§9禁止3 只覆盖自家数字拼接、不覆盖此条）；**保留正结果及其代价，局部失败不抹掉其它收益、也不扩大成方法族无效**（GUIDE L14——§9 的 EVQ 三连败条是其例化）；**不伪装、不许诺未验证的成功率**（GUIDE L41）。另 GUIDE L57 点名应保留但本档此前无登记的正收益：算子压缩 NLL 改善**须与检索未恢复同时陈述**（rope_operator_family 一旦进 discussion 即生效）、FullLagP2 小面板收益与 C2 低维描述收益。

---

## 7. 跨阶段术语与身份守卫（改稿时逐条对照）

1. **三个 s**：表构造倍数 s（本战役全 4）、S=Σm（预算，口语"预算"）、论文 s_j=Π_{d<j}λ_d（累积因子）。互不可换。
2. **YaRN 有五个身份**（owner=EVIDENCE **§1.2 对象 1–5**；⚠ §4.1 表只覆盖其中 2 个 YaRN 身份，对象 4 `exact_yarn_olmo.json`（P9 臂）与对象 5 `yarn_turns_paper` 不在表内，且对象 4 的 sum_m=42.91\* 标注与其"=对象1 OLMo 版"身份自相矛盾——执行 P9/D 臂前须先解决）；"YaRN 前置→Pro 后置"只对论文转数变体成立；对官方代码是"后置→更后置"。
3. **MrRoPE 是 training-free**（论文 §4.1）；"训练时用表"为仓库旧错误表述（COVERAGE **§三 L95 与 §5c**、GPT6PRO Q6、**EVQ_LONGRANGE 边界表述**均需改读，EVIDENCE §2.3-2 列全了影响面）。
4. **EVQ 有三个算子**：EVQ-Cosh（学习期，m<0 内点向快端——手算 τ=1、u=0.5 处 φ−u=−0.058；"端点锚定"是连续族边界条件 φ(0)=0,φ(1)=1 意义下的近似，中点网格首末槽随 τ 漂移 ~0.006；中点网格 u_k=(k+1/2)/K）；canonical EVQ=evq_shift（ν=θ^{−φ}）；evq_deploy（ν=ω·4^{−φ}，项目变体、压缩方向，代码注释 "NOT this"）。**零训练三 τ 全 0 的臂身份=旧 evq_shift（midpoint 位移装 endpoint-native 网格，全槽 ×1.108/×1.114），不是 canonical EVQ**——literal canonical 网格（evq_midpoint_t1）从未有任务面板结果（EVIDENCE:91/236；§4.3 已同步更正）。
5. **BM 命名**：350 行面板上"MrProBM/BM/部署表"同名同物；`beta_b1_BM`/`beta_b1p0` 在不同面板是同一构造；12 臂拟合对象是部署 BM（archived 端点输出）。
6. **gain**：全战役冻结 1.138629436111989=0.1·ln4+1；作用在 cos/sin 上（logit multiplier g²）；Qwen native 参照臂带此 gain（"表 vs native −0.17 nats"是同 gain 频率对照，非纯 native）。
7. **面板口径**：350=选择面板（dev）；180 实为 120 唯一提示（60 重复组）；fresh_72 与旧 2227 行 token 级零重叠；391 自然 QA 87% 地板行。显著性一律配对 + 唯一提示 + 分层 SE。
8. **伪影任务**：niah_single_3 贡献选择面板 +32~46pp 且不在任何 held-out——凡含它的合计分解须单列。
9. **坐标恒等式**：`4^{−1.5r}=8^{−r}`（8× 包装无新意；`llama3_60m/build_ec_tables.py`:131 有机器检查）；条带斜率≡ν 退化斜率≡lnθ/(2K·ln2)=**0.147903=ln(5×10⁵)/(64·ln4)**（θ=5e5、K=64=**OLMo-2-0425-1B-Instruct 自身几何**，滑行⟺ν 常数；勿按 OLMo-1 代常识 θ=1e4/K=32 复算得 0.2076 而误判）；S≡64−μ_ε 与 31+Σ(19−i)ε_i 同式（⚠ 31/19 是**带特异常数**：OLMo 带 [14,32]、n=18、满平台 31 槽；换带换常数，Llama 带 (18,35)/n=17 应为 28+Σ(18−i)εᵢ）。
10. **Llama 特有**：配对布局 (i,i+K) half-split；K=64 时频带 (18,35)@8K；MR 中段 j=19..34 Σm=16/3、全表 34⅓。
11. **Qwen 特有**：θ=1e6、带 (23,40)/n=17（与 MrRoPE 论文附录 B 最佳逐位一致）；旧 evq_shift 臂在 Qwen 上 ×1.114。
12. **带边界规则与 α/β 记法**：三家（官方 YaRN/论文 Eq.26/我方）同一 j=head_dim·ln(W/(turns·2π))/(2lnθ)、YaRN 取 floor/ceil；但**论文记法 α=32/β=1 是我方 (1,32) 的交换**——引论文超参必查。
13. **同名两棵树**：判决文档中 `a1_proofs.tex:行号`/`03_theory.tex:行号` 指 NeurIPS 旧稿（`main_0726:paper/…`）；当前 `paper-2027/appendix|sections/` 同名不同物（行号全不同，如 T₁ 旧稿 :258 vs 新稿 :478）。

---

## 8. 数字溯源索引（论文常用数字 → 权威文件）

| 数字 | 权威来源（相对仓库根） |
|---|---|
| 151.9M 三种子 −0.281/−0.176/−0.146；+0.026 | `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json` |
| 50M 交叉矩阵 [[7.14,76.20],[23.05,7.16]] | `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5.2（其数值 JSON 只写到 /tmp 未入库，报告即库内 owner） |
| 151.9M 交叉矩阵 [[3.426,5.776],[4.455,3.479]] | `paper-2027/research/attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json`（:small_model_crossing.mean_tail_nll_two_seed_length1024，@1024、seeds 137/256；勿用 Qwen K32 receipt） |
| 432M MLA 138.8→95.6（GEO→EVQ@16K）/ 117.9→71.1（**GEO+YaRN(s=4)→EVQ+YaRN(s=4)@16K**；−48.8% 口径=71.1 对纯 GEO 138.8，两"→"勿并排误读） | `data/curated/table18_mla_3seed_aggregate.json`（"midpoint 协议"系 repo 代码旁证之推断，JSON 本身未标） |
| 750M 45.1→24.4、PK 0/40→31/40（单 seed-42，JSON 自带 claim_ceiling） | `data/curated/phase15_750m_continue_result_20260306.json` |
| Llama-8B 适配 6.82/108.96/991.48→10.07/24.07/127.91 | `data/curated/llama8b_causal_source_use_s42_20260714.json`（300 步 pair；PPL 另一 owner=`data/curated/lora_longalpaca_temporal_s42_20260712.json`（991.475 出处）；勿与 516 步 RULER-mix `llama8b_matched_ruler_mix_20260726.json`（8K 94.44/77.60%、16K 0.29/14.03%）混） |
| 成熟冻结 held-out geo/ramp/residual 0.56/61.04/60.47%（OLMo）+57.75/64.00/66.50%（Qwen） | `data/curated/frozen_fixed_support_mature_20260823.json` + `attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json` |
| 454M 157.7→107.5、PK 61±3→100±0（饱和天花板口径） | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` |
| 三模型 BM 匹配 Qwen3B 78.13/70.83、Qwen7B 84.44/71.11 | `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json`、`ROPE_QWEN7_BM_RESULT_20260908.json` |
| selective-QK 2Wiki 8K 0.07→21.48/16K 0→8.57、RULER 8K 2.02→31.63 | `rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json` |
| BM 五任务 QA 21.62→25.44（+3.82 CI[1.32,6.29]，778/631） | `docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json` |
| Qwen-0.5B +6.09pp CI[2.76,9.58] | `paper-2027/research/attention-aware-retrofit/evidence/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json` |
| Gemma K128 +6.19pp CI[2.81,9.63]（上界 raw=9.625，half-up 舍入） | 同目录 `K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json` |
| r_2=2.00013、23 慢 pair（标准 u_k=k/K、K=64 网格） | `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`:209 + 正文 `sections/03_theory.tex`:33（claim map 在 `paper-2027/research/`非 docs/research/；a1_proofs 只有 23 slow pairs） |
| M4 factorial 10/12、9/12（10/12=预指定 1.25× 臂；cosh_rule 直测 7/12） | `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json` |
| 1.485B 182.73→159.64 | `rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| 战役全部 350/180/fresh72/NLL 数字 | `ds_workspace/recon_20260910/verdicts/*` + `work/jsonl/*`（本地重算命令见 EVIDENCE 附录 B；sum_m 恒等式 owner=附录 B #1 `formula_audit`；**fresh_72 原始读数与 72 行 run_holdout_01/run_controls_01 仅服务器**，本地旁证=EVIDENCE:141-144+`docs/research/ROPE_OLMO_BM_RESULT_20260908.{json,md}`，见 §4.2 标注） |
| 12 臂 N 拟合 −20.4532+1.19068N | `paper-2027/figs/profile_diagnostic_inputs.json` + `verify_profile_diagnostics.py` + `appendix/a8_profile_diagnostics.tex` |
| 审计修正（唯一提示等） | `ds_workspace/recon_20260910/audit/pro_decision_20260911/{REPORT.md,check_results.json}` |
| 覆盖理论全部量 | `ds_workspace/recon_20260910/code/coverage_theory_20260911.py`（纯 numpy） |
| EVQ_LIMIT 理论构造 | `ds_workspace/recon_20260910/code/evq_limit_20260911.py`（N 机制/Q̂/条带候选）；**W2p 无代码 owner**——显式 m 向量只在 `_reports/EVQ_LIMIT_20260911.md` §5.1（Σm=37.835 手算可复核）；**实跑 Cslow** 构造=`experiments/zerotrain_20260910/patch_agent_tables.py` `_cslow()`（0.98× 合法化斜率版，与 evq-limit 原版有意不同），结果 owner=`theory/RELEASE_AXIS` §六 |
| BM 平滑 radix 增量三次 m_q 公式 + N18/N17 逐项指数数组 | `docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json`（fig_bm_exponent_profiles 直接依赖） |

---

## 9. 修订硬规则（do / don't）

**禁止**：
1. 恢复任何 9/11 修订删除的内容（未执行实验残留、Fisher/Hessian 断言、单槽机制叙述、全称自证）——删除清单见 §5.4；
2. 把 350 行选择面板赢家写成结论；把均值 NLL 改善写成任务收益；把 T3/T4 写成跨模型定律；把 1.13 写成精确最优（它是 reach-only 上界）；把 EVQ 零训练三连败扩大到"EVQ 无效"（学习期正结果仍在）；把 collision/24–92%/c_coll 当 Cosh 支持（Q1/Q8 已否）；把 `d/√L`、`1−ln(L/2π)/ln b` 写成推导最优；以 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 作直接依据（已删出工作树、`git show main:rebuttal/pre_rebuttal/…` 可取——读可以，但结论以两份 07-20 在库文件为据）。
3. 跨协议拼接数字（如把 2K-raw-trained 六格表当 YaRN 论文续训复现；把 300 步与 516 步 pair 混用；把 FWE/VT 部分分称 exact accuracy）。

**必须**：
1. 新增任何数字前在 §8 索引找到 owner；找不到就不写；
2. 形状对比全部带"振幅=1.0 条件"前缀（T-amplitude-conditionality），或等待 amp4x 结果；
3. Qwen/跨模型句式用"未证明/UNDERPOWERED/响应平坦"，禁止"方向相反"（Simpson 已否定）；
4. 每次编译 `bash paper-2027/compile.sh` + 跑 `verify_profile_diagnostics.py`/`verify_explicit_geometry.py`；
5. 遵守匿名/字体/页边检查（现稿已过，别弄坏）；
6. 结论分层照 `EXPERIMENT_ASSETS_TOP15` 口径写（其实质=第 2 列"实验资产与可成立的主张"+第 4 列"裁决与稿件位置"内嵌禁止性限定；无字面"不能扩大成"独立列——引用时按这两列）。

---

## 10. 重排与定位建议（上轮深析的结论，供改稿采纳）

### 10.1 单脊柱重排（解决"两 regime 合订"风险）
引言一句话主线：**内点频率分配是独立设计变量，学习期被共适应支配、部署期被振幅×形状支配**。顺序：几何（§3 现状保留）→ 学习期识别（三种子+重设+交叉，全文心脏）→ EVQ-Cosh（Q1 口径：闭式构造）→ 冻结部署压成一节（BM 四块证据 + 官方 YaRN 基线 + table×gain 2×2）→ 讨论（覆盖几何作解释框架、标 OLMo 族；NLL-任务反向一段作仪器警告）。撤下（留第二篇）：覆盖理论作为独立理论主张、四角、LongBridge、自然 QA 零效应、选择效应方法学（留一段诚实声明）。

### 10.2 新增图（单位成本回报最高）
1. **"形状=赌注"机制图**：四张 m 剖面（Pro/YaRN-index/Uni/BM）× 两个任务矩阵（近重型 vs 均匀检索型）的正反对照——同时解释"我们赢 Pro"与"Pro 在 LLaMA 赢 YaRN"，零矛盾叙事。
2. E2 机制遥测一图（1 行×4 表逐槽 logit 归因，解析精确，近乎零 GPU；step25 vs step42 只差槽 22–24 可逐槽判决）。

### 10.3 与 MrRoPE/LeRoPE 的 novelty 关系（外部已核实）
- MrRoPE（arXiv [2601.22181](https://arxiv.org/abs/2601.22181)，training-free 已双重证实；**"ICLR 2026" venue 未在 arXiv abs 页证实**——引用前自行核 OpenReview）：主题相邻、贡献正交。处理=引用让渡统一参数化（Eq.(7) 已把 mixed-radix cast 为指数位移），把贡献从"提出另一张表"上移到"设计变量识别+机制+方法学"。
- LeRoPE（arXiv [2607.10134](https://arxiv.org/abs/2607.10134)）：逐频可学标量、52M–2.5B 从头训练、窗内 LM 质量（2.5B 省 3.4% FLOPs）——机制与我们相反互补（他们中低频向下 vs 我们内点向上）⟹ 转化为"分配是真设计变量、无普适方向"的两端互证。
- FMRoPE（[Oka et al., ICLR 2026](https://openreview.net/forum?id=PR1PPxvG9Q)）：base×窗长定频带、归一化指数保持等距——与"固定端点只动内点"正交互补。
- 文献空档（可写成 claim）：振幅×位置×形状分解下形状=任务直方图赌注的事前预测与判决；N 条带结构定理（结构成立部分）；权重-表共适应配对识别；YaRN-index/turns 身份分歧（勘误级）；倍率饱和公式（YaRN 中频 1/(1−u) 封顶 vs Pro 幂律 s^m，4×↔16× 符号翻转可验）。

### 10.4 上卡三臂（堵最贵审稿洞；集合与 phase1 一致，**列序系"堵洞价值"自创序**——RUN_READY 实际建议队列=0 核验 A 臂权重→B（最便宜）→A→E→D(lb_hold)，execution_order phase_1 列序=A,B,C,D,E）
E（官方 YaRN same-panel，堵"checkpoint mismatch"）、B（amp4x，振幅条件性从 warning 变 finding）、A（qwen4x BM，跨模型从"未证明"变"有功率判决"）。可选第 4 臂：P5 Uni（覆盖归因 vs 剖面形状归因——"平滑归因"无原文出处，prereg 原话是"失败=凸剖面覆盖空洞"；**P5 在所有队列中均"未排队"，系本档建议**）。

---

## 11. 仓库地图（按四阶段）

```
阶段①  docs/exp/2026-02|03|04|07/       月度原始报告（历史证据，非数字 owner）
        results/350m_mla32_*, PHASE18/19/22_23   432M MLA 线
        data/curated/                   现 7 份 curated JSON（数字 owner；⚠ 阶段①另有 5 件被
                                        6b636e5 删出工作树，`git show main:data/curated/<file>` 取回，见 §2.4）
        rebuttal/                       战术室：README/playbook/EXPERIMENT_THEORY_REVIEW/
                                        STRONG_MODEL_THEORY_VERDICT/THEORY_STANCE_CONSOLIDATED/
                                        ADVERSARIAL_REVIEW_FINDINGS/rebuttal_0723/theory_results/
                                        （⚠ README 导航 13 条中 11 条已删，见 §2.4 provenance (d)）
        docs/overview/                  provenance 权威（RESULT_PROVENANCE_MANIFEST 等；
                                        其对已删 curated 件的 SHA 登记已断链）
        ai-handoff.md                   NeurIPS 期入口（2026-07-20 止；其阅读顺序引用的
                                        REPO_MAP.md（删于 875a8be）与 paper/README.md 已失效）
阶段②  paper-2027/                     活动论文（main.tex + sections/ + appendix/）
        paper-2027/research/foundations/        谱基+共适应 canonical report
        paper-2027/research/evidence/           151.9M 三种子、Video DiT
        paper-2027/research/attention-aware-retrofit/   成熟干预阶梯（evidence/results/theory/preflights）
        paper-2027/research/{TOP15, CLAIM_MAP, audits/, external-reviews/, pdf-review-rounds/}
        paper-2027/research/history/TIMELINE.md  官方阶段账本
        分支 main_0726 / main_0726_09_06 / backup/…   被遮盖历史
阶段③  ds_workspace/EXPERIMENT_THEORY_MASTER_20260911.md   战役总账（⚠ 未提交新增 §8 就绪队列，§5.1/§5.3 所引含未落盘内容）
        ds_workspace/recon_20260910/{index.md, YARN_MRROPE_RESEARCH_EVIDENCE.md,
          verdicts/(28), theory/(13), _reports/(8), prereg_protocols/(14),
          audit/{pro_decision_20260911,verify_coverage_20260911}, code/, work/jsonl/}
        ds_workspace/{TONIGHT_EXPERIMENT_RULES_20260911.md（今晚执行权威）,
          GPT_PROMPT_TONIGHT.md（已被取代、仅历史设计输入）, LESSONS}.md
        analysis/unify_20260910/        62 代理统一推导工作台（理论历史；同级 kkt_20260910/、p0_gradients/ 为过程件）
        experiments/{curvature_20260910, zerotrain_20260910, rope_decision_20260911,
          nongeometric_screen}/        战役基础设施
        results/{olmo_fast_screen_20260908, bm_transfer_20260908}/
        （远端实例，非仓库）/root/autodl-tmp/{olmo_fast_screen_20260908/code,
          nongeometric_screen_20260909/code, rope_decision_20260911, phase1_20260910}
阶段④  /Users/yang/Downloads/RoPE_Integrated_Experiment_Guide_Codex_20260911.md  Plan B（不在库）
        /Users/yang/Downloads/LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md
        /Users/yang/Downloads/EVQ_MrRoPE_Separate_Theories_and_Experiments_20260911.md  上轮四问分析（含 P1–P23 文献索引）
        experiments/llama3_60dir_20260911/   Plan B 实现+修正（execution_order.json 为执行权威）
        experiments/llama3_60m/              60 规则第二实现
        ds_workspace/recon_20260910/RUN_READY_20260911.md   就绪清单（注意 §5.1 五缺陷修正其命令；§A 权重恢复已执行完毕）
        paper-2027/research/SUBMISSION_REVISION_20260911.md 9/11 修订说明
旁线    experiments/{nosa_position, native_sparse_position, rotary_budget, position_overnight,
          broad_position_eval, deepseek_mini_position, evq_recovery, joint_kkt_20260910,
          kld_v2, native_rope_evq_150m, pm_keep, position_observability, refcarry_audit,
          rope_operator_family, twotrack_20260911}    正交线（证据未重核，不入正文）
        internal/, outputs/, results/{legacy, core_text, supporting_*}   历史隔离区
```

---

## 12. 修订完成验收清单

- [ ] 每个新增/保留数字在 §8 索引有 owner；grep 论文新数字能回到文件
- [ ] 无 §9"禁止"清单内容复活（对照 SUBMISSION_REVISION 删除清单逐条 grep）
- [ ] 形状结论全部带振幅条件前缀；跨模型句式合规（§9.3）
- [ ] 摘要/贡献与正文证据分层一致（TOP15"可成立/不能扩大"口径）
- [ ] `bash paper-2027/compile.sh` 通过：0 未定义引用、0 overfull、正文 9 页内、匿名
- [ ] `verify_explicit_geometry.py`、`verify_profile_diagnostics.py` 复跑通过
- [ ] 引用核对：MrRoPE=training-free 表述、YaRN-index/turns 区分、FMRoPE/LeRoPE/LongRoPE2 定位准确
- [ ] 若引用战役数字：带判决文档路径 + 已知修正（唯一提示口径、符号翻转修正、Simpson 边界、服务器/在库可达性分层见 §4.2）
- [ ] 摘要数字密度对照作者历史要求（8/26 评审日志 `paper-2027/research/archive/2026-08/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`:1496 "fewer abstract numbers"）确认现行授权——9/11 重写后摘要含 ≥5 组数字，库内暂无推翻记录
- [ ] 除 `paper-2027/` 与本文档外未修改任何仓库文件（R1 修正原表述与 §0 纪律"改稿只改 paper-2027/ 源码"的字面死锁；verify 脚本刷新 `figs/` 属授权）
