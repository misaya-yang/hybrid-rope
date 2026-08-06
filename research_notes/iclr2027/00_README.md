# 转投预案(条件启动)— 总入口

创建:2026-08-06 · 修订:2026-08-06(改为条件性预案)· 状态:`PLANNING_ONLY`
性质:内部规划文档(untracked,不进 supplement,不是 evidence owner,不构成任何 GPU 启动授权)
定位:**条件性预案** —— 2026-09-24 NeurIPS 决定公布前,只做实验与素材,不启动任何新稿写作(用户指示,2026-08-06)

## 0. 本目录文件

| 文件 | 内容 |
| --- | --- |
| `00_README.md` | 本文件:时间线、决策摘要、硬约束 |
| `01_EXPERIMENT_PLAN.md` | 实验计划(P0 零成本 → P1 本地 5090 → P2 视预算),每项含 concern 映射、最小方案、stop condition |
| `02_PAPER_REVISION_PLAN.md` | 论文修改计划:重定位、逐章动作、审稿关切→修改映射、claims 保留/降级/新增 |
| `03_INWINDOW_EXTRAPOLATION_THEORY.md` | 核心问题文档:微调理论缺口 —— 如何保持 in-window 同时加大外推(成本二分、约束分配、P-EVQ、可证伪预测) |

前置阅读(本目录所有结论的事实底稿):
`research_notes/FABLE5_EVQ_MECHANISM_AUDIT.md`(2026-08-06 机制审计)。

## 1. 关键日历(2026-08-06 官网核实)与场馆判断

| 节点 | 日期(AoE) | 来源 |
| --- | --- | --- |
| NeurIPS 2026 讨论期结束 | 2026-08-03(已过) | 官网 |
| **决定日 D:NeurIPS 作者通知** | **2026-09-24** | 官网 |
| ICLR 2027 abstract 截止 | 2026-09-18 | 官网 |
| ICLR 2027 全文截止 | 2026-09-25 | 官网 |
| ICML 2027 截止(默认回退) | 未官宣;第三方估计 2027-01 中下旬 | 待官宣核实 |

**场馆判断**:ICLR 2027 abstract 截止(09-18)早于 NeurIPS 决定日(09-24)。不撤稿而在 09-18 提交 ICLR 即构成双投(同时违反 NeurIPS 与 ICLR 政策);为赶截止提前撤掉一篇 AC 明言"证据可改变推荐"的 borderline 稿,不是本预案的默认动作。因此:

- **直接转投 ICLR 2027:默认不可行。** 仅当作者在 09-17 前主动决定撤稿才重新评估;该决定只能由作者本人做出。
- **默认回退:ICML 2027**(D 后约 16–17 周窗口,时间充裕)。备选:AISTATS 2027(约 2026-10 上旬截止,若官宣时间允许可作快速通道,契合度较弱)。
- 目录名 `iclr2027/` 保留历史命名;实际目标 venue 由 D 日分支决定。

## 2. 决策摘要(一段话)

无论 09-24 结果是 accept(camera-ready 增强)还是 reject(转投),科学主线相同 —— 重定位为 **mechanism clarification + 无损分配(no-harm allocation)** 论文:
(a) 保留并前置已完成的因果机制链证据(OLMo-2-1B matched routing 0/100 vs 69/100 & 67/100;8B gold-drop/swap/oracle 因果分解;750M matched continuation 0%→77.5% AR);
(b) 用「in-window 成本二分」(分配内禀代价 +0.038 NLL,小 vs 换表适配失配,大)和「aliasing/OOD-phase 二分」重写理论叙事;
(c) 新方法轴 = **保护性部分频谱替换(P-EVQ)**:微调时保留 Native 重要通道频率、只重分配冗余/OOD 通道 —— 直接回答"保 in-window + 增外推";
(d) 正面处理 FMRoPE(Oka et al., ICLR 2026)与 LeRoPE(arXiv 2607.10134)定位,含 matched 对照与诚实的双向结果。

依据:NeurIPS 11628 panel(Dz6s 4 / 27bE 3 / zWsa 2,AC 结论 = 新颖性 vs FMRoPE + 规模/基准 + 理论链三大缺口),全部映射见 `02_PAPER_REVISION_PLAN.md` §2。

## 3. 硬约束(全程有效)

1. `paper/` 整树只读(AGENTS §3.1)。新稿目录仅在 **D 日分支为转投后** 创建,按届时 venue 命名(如 `paper_icml2027/`);在此之前不写任何正文。绝不在 `paper/` 上改。
2. 任何 GPU 训练/推理/付费实例,须用户**逐项显式授权**;5090 之外的付费 GPU 先出 READY receipt(AGENTS §2.1–2.2)。本目录所有实验条目均为提案。
3. Evidence tier 纪律不变:计划≠结果;`DESIGN_ONLY` 的东西(attention restoration、P-EVQ、spectral-frame)在论文文本中只能以完成后的 owner 为准引用。
4. 已注册实验的 gates/阈值(如 P-EVQ 的 Stage D 五条判据)**不得在看到结果后调整**。
5. Claim identity:P-EVQ 成功后叫 "Native-importance-protected hybrid retrofit",不叫 pure EVQ-Cosh、不叫 zero-parameter schedule(`OLMO2_NATIVE_IMPORTANCE_PROTECTED_EVQ_HYPOTHESIS_20260728.md` §10)。
6. **双投红线**:2026-09-24 通知前,本工作仍处 NeurIPS 评审期,不得向任何其他 venue 提交(含 abstract)。

## 4. 时间盒与决定日分支(详细排期见 01 §5)

实验轨道(venue 无关,双用途:accept → camera-ready 增强;reject → 转投底稿):

| 周 | 窗口 | 主线 |
| --- | --- | --- |
| W1 | 08/06–08/12 | P0 零成本分析全部完成;related-work 素材笔记(FMRoPE/LeRoPE,不写正文) |
| W2 | 08/13–08/19 | P1-1 P-EVQ 诊断→筛→训练(需授权);P1-3 τ basin 补全启动 |
| W3 | 08/20–08/26 | P1-2 scratch 无损变体;P1-4 aliasing/OOD 分离;P1-5 OLMo QK 探针 |
| W4 | 08/27–09/02 | P2 选择性执行;冻结主结果表 v1 |
| W5–W7 | 09/03–09/24 | 结果整理、图表素材、P2 未完项继续(已无会前硬截止);**不写正文** |

决定日 D = 2026-09-24 分支:

| 结果 | 动作 |
| --- | --- |
| Accept | 02 §0 降级执行:camera-ready 增强清单(不改结构,按官方规则加新证据) |
| Reject | 启动 02 完整转投计划,目标默认 ICML 2027;里程碑按 02 §5 的 D-相对周执行 |

叙事关键实验(不论 venue):P0-1、P0-2、P1-1。其余为增强项;W4 未完成的 P2 项可在 W5–W7 继续。
