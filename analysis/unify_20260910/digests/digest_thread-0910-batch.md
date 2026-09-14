# digest thread-0910-batch（09-10 07:40–07:52 批量启动的 16 个 codex 短会话）

提取时间：2026-09-10 约 08:07（本地，America/New_York；rollout 内时间戳为 UTC，本地 = UTC−4）。
提取方法：python3 逐行解析 JSONL，仅保留 `type=response_item & payload.type=message & role∈{user,assistant}` 的文本，跳过 tool/reasoning/token_count；另用事件类型统计与 `task_complete` 事件判定终止状态。纯文本转录存于 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/analysis/unify_20260910/raw/batch_t07-*.txt`（16 个分文件）与 `batch_all.txt` / `batch_clean.txt`（合并、去 boilerplate 版，clean 版 485 行，已全文通读）。

## 0. 总体判定（先结论）

1. **这 16 个会话不是"失败/中断对话"，而是一个仍在运行的多智能体研究蜂群（swarm）的两波子会话。** 第一波 sol01–sol08（07:40:42–07:41:44 启动）全部有 `event_msg/task_complete`，即正常完成。第二波 astra01–astra05、sol09–sol11（07:46:00–07:52:16 启动）在我最初 07:57 列目录时 mtime 全为 07:57，看似"被杀"，但复查发现 rollout 仍在增长：到 08:04 时 astra01/02/03/04/05、sol09 均已完成（task_complete），sol10、sol11 截至 08:05–08:07 最后时间戳仍在推进（无 task_complete，状态=进行中，非终止）。08:00 后又出现第三波 sol12、astra06、astra07、sol13、sol14。"16 个短会话被批量杀掉"的先前假设**不成立** [已验证：task_complete 事件计数 + 每文件最后时间戳，见 §2 表]。
2. **任务本体**：统一 EVQ 与 MrRoPE 的零训练频率分配规则推导。由一个 root 协调会话（parent_thread_id=`01a0806f-3df5-74b1-bc56-bf00d89d238e`，visualization 目录日期 2026-09-08，即该会话 09-08 已存在）按 `team_plan.json` 计划派出 20 个 gpt-5.6-sol + 10 个 gpt-6-astra 子代理，并发上限 8。用户指令经 COMMON.md 转达（原文见 §6）。
3. **这批是否重复产出已否决方向：没有发现直接重犯。** 16 个会话中所有完成的报告（14 份）都显式引用 Smooth_MrBudget 反例（几何失真度量下降但长任务结果更差）作为约束，把几何量降格为"正则/诊断"，并在各自的 "counterexample checks / evidence boundary" 节自我限制；无一报告把代理指标当能力结果宣称。存在的是**有意的重叠分层**（sol02 与 astra02 都做有限窗口 cosh 修正，astra01/03/05 独立收敛到"signed source-vs-distractor margin"，astra03 明确写 "Astra01 owns the continuum/co-adaptation derivation; this report supplies the frozen finite-slot rule"）——这是收敛增强而非重复否决项 [已验证：报告原文路径见 §3]。
4. **产出物**：`.agents/rope_unification_20260910/reports/` 下 14 份 agent 报告 + coverage 回执（sol01–sol09、astra01–astra05；sol10/sol11 当时未出报告）。任务分配文件 `.agents/rope_unification_20260910/assignments/` 下 sol01–sol20、astra01–astra10 共 30 个 md（sol12–20、astra06–10 当时 pending 或部分已启动）。

## 1. 来源清单

### 1.1 rollout 文件（16 个）与代理映射

目录 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.codex/sessions/2026/09/10/`。所有子会话共享 parent_thread_id `01a0806f-3df5-74b1-bc56-bf00d89d238e`；cwd 均为项目根。agent_path / 行数 / 提取消息数 / 终止状态（截至 ~08:07 本地）：

| # | 文件（前 30 字符） | agent | 行数 | 消息(user+assistant) | 起止(本地) | 终止状态 |
|---|---|---|---|---|---|---|
| 1 | rollout-2026-09-10T07-40-42-01a08b1e-be61 | /root/sol01 | 139 | 5 | 07:40–07:47 | task_complete 07:47:16 |
| 2 | rollout-2026-09-10T07-40-51-01a08b1e-ddf4 | /root/sol02 | 91 | 4 | 07:40–07:45 | task_complete 07:45:43 |
| 3 | rollout-2026-09-10T07-41-00-01a08b1f-040b | /root/sol03 | 162 | 4 | 07:41–07:47 | task_complete 07:47:19 |
| 4 | rollout-2026-09-10T07-41-11-01a08b1f-2bf1 | /root/sol04 | 127 | 4 | 07:41–07:46 | task_complete 07:46:35 |
| 5 | rollout-2026-09-10T07-41-18-01a08b1f-4776 | /root/sol05 | 154 | 6 | 07:41–07:48 | task_complete 07:48:38 |
| 6 | rollout-2026-09-10T07-41-25-01a08b1f-6454 | /root/sol06 | 92 | 5 | 07:41–07:45 | task_complete 07:45:43 |
| 7 | rollout-2026-09-10T07-41-36-01a08b1f-8d94 | /root/sol07 | 204 | 5 | 07:41–07:51 | task_complete 07:51:56 |
| 8 | rollout-2026-09-10T07-41-44-01a08b1f-adb8 | /root/sol08 | 261 | 6 | 07:41–07:49 | task_complete 07:49:53 |
| 9 | rollout-2026-09-10T07-46-00-01a08b23-986e | /root/astra01 | (增长至 7.25MB) | 7+ | 07:46–08:03 | task_complete 08:03:50 |
| 10 | rollout-2026-09-10T07-46-10-01a08b23-bc88 | /root/astra02 | (6.73MB) | 5+ | 07:46–08:00 | task_complete 08:00:41 |
| 11 | rollout-2026-09-10T07-46-55-01a08b24-6e0b | /root/astra03 | (6.65MB) | 6+ | 07:46–08:03 | task_complete 08:03:32 |
| 12 | rollout-2026-09-10T07-47-46-01a08b25-3514 | /root/sol09 | (9.66MB) | 8+ | 07:47–08:01 | task_complete 08:01:58 |
| 13 | rollout-2026-09-10T07-47-58-01a08b25-61cd | /root/astra04 | (6.20MB) | 6+ | 07:47–07:59 | task_complete 07:59:43 |
| 14 | rollout-2026-09-10T07-49-19-01a08b26-9e7e | /root/sol10 | (7.93MB) | 5+ | 07:49–08:07 | **进行中**（无 task_complete，最后事件 08:07:15） |
| 15 | rollout-2026-09-10T07-50-28-01a08b27-af26 | /root/sol11 | (9.28MB) | 6+ | 07:50–08:05 | **进行中**（含 1 次 history `compacted` 事件，最后事件 08:05:33） |
| 16 | rollout-2026-09-10T07-52-16-01a08b29-5312 | /root/astra05 | (6.40MB) | 5+ | 07:52–08:04 | task_complete 08:04:19 |

边界说明（不属于本批但同日相关）：08:00–08:04 又启动第三波 /root/sol12、astra06、astra07、sol13、sol14（截至 08:05 均 RUNNING）；01-53-54、01-59-00、02-12-40 三个更早会话按任务说明归前一任务处理。

### 1.2 磁盘工件（本批的输入与输出）

- `.agents/rope_unification_20260910/COMMON.md`（1,635 B，07:40）：全体子代理共享指令头。
- `.agents/rope_unification_20260910/team_plan.json`（2,877 B，07:54 更新）：`requested {gpt-5.6-sol:20, gpt-6-astra:10}, concurrency_limit_subagents:8`；07:54 快照状态 sol01–08=report_complete、sol09–11 & astra01–05=started、sol12–20 & astra06–10=pending。
- `.agents/rope_unification_20260910/assignments/{sol01..sol20,astra01..astra10}.md`（2.2KB–16.9KB 每个）：每代理任务书（COMMON 全文 + YOUR ID + YOUR TASK + FULL FILE LIST）。
- `.agents/rope_unification_20260910/corpus/`：`solNN_full_dialogue.jsonl`（sol09–14 分片语料，供 transcript-audit 任务）、`tool_outputs_001..077.jsonl`（归档完整工具输出）、`project_fulltext_inventory.json`、`CORPUS_SCOPE.json`。
- `.agents/rope_unification_20260910/reports/`：sol01–sol09、astra01–astra05 各 `.md` 报告（10.3–32.2KB）+ `*_coverage.json` 回执。sol10/sol11 报告截至提取时未出现。

## 2. 任务时间线（每任务：目标 → 方案 → 结果 + 关键数字）

### 第一波（07:40:42–07:41:44 启动，理论/审计任务，全部完成）

- **sol01 — MrRoPE 原论文全量重建**（3 文件：RoPE_Papers/5551_MrRoPE…md + 项目 ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS/TRANSITION_REVIEW 20260908）。方案：全文通读 + 附录逐步重建 + CPU 数学检查。结果：**成功**。关键结论（对话 11:41:43Z、11:44:06Z + 报告）：(a) MrRoPE 的 "mixed-radix" 存在形式缺口——RoPE 坐标是连续相位 mod 2π，真混合进制数字需要 flooring 与逐维进位模数，论文未给出该定理 [已验证:报告推导]；(b) 论文"YaRN 恒 regressive"证明隐含 `c=β−Sα≥0`，目标参数 S=4,α=1,β=32 下成立，但不对所有 S 普适、c<0 时不等式依赖反转 [已验证:报告]；(c) 建设性规则：把 MrRoPE 当累积对数频率坐标系 `ν_j=ω_j S^{-m_j}, m_j=Σ_{d<j} ε_d`，冻结部署时在该坐标下优化"精确有限窗口、源加权 EVQ 碰撞泛函 + 实测的 native-logit 损伤预算"，拒绝把任一几何单独当精度 [假设/待 GPU 验证]。coverage: 3 files / 68,828 B，零遗漏。
- **sol02 — EVQ 推导与论文证明审计**（4 文件：EVQ_COSH_THEORY.tex、a1_proofs.tex、budget_proofs.tex、EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md）。结果：**成功**。决定性发现（对话 11:42:32Z + 报告）：cosh 密度唯一最小化的是**投影代理泛函** `C_app[ρ]=α/2∫ρ² + β/2∬ρ(φ)ρ(ψ)min(φ,ψ)`，不是原有限窗口碰撞泛函；有限 L 下原核保留非局部对数脊，精确分配问题是积分/有限表优化而非同一局部 ODE [已验证:变分推导]；尖锐 slot 跳变被局部 δ-脊替换过度惩罚 → "密度平滑"不是定理支持的部署准则。coverage: 4 files 全读，零 GPU。
- **sol03 — Pro 提案组1 审计**（3 文件：FIRST_PRINCIPLES_SYNTHESIS、iclr2027 dossier 20260904、TWO_DIRECTION_THEORY_AUDIT）。共 3,091 行全读 + SHA-256 回执。结果：**成功**。边界结论（对话 11:43:00Z）：位置基几何可指导从零训练，但**不能**单独给冻结 checkpoint 的 slot 改动排序——学习内容方向绑定 slot 标签；"一个分配框架、两个显式不同的估计器"。coverage: 3,091 行零遗漏。
- **sol04 — Pro 提案组2 审计**（3 文件：Exponent_Allocation_Unified_Plan、Research_Guidance、SINGLE_TABLE_FFN_REPORT_AUDIT）。2,175 行全读。结果：**成功**。构造：可辨识性归一化 `ω_k=exp[−(a+R z_k)]`（base 与未归一化指数否则不可辨识），统一目标 `z*=argmin D_demand(z;μ_Δ)+λ_S·D_compat(W_S,z;μ_N)+γΩ(z)`，λ_S≈0（初始化）/大（冻结）/部分（LoRA）；给出 Qwen2.5-3B 32K→128K 的 "native-KL-budgeted water-filling" 具体分配 [假设:未跑模型]。
- **sol05 — Pro 提案组3 + newest6Pro 附件审计**（5 文件含 pasted-text 附件与 ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md；2,399 行全读）。结果：**成功**。决定性审计（对话 11:45:11Z + 报告 Decision 节）：full-row KL 是 sound 兼容性诊断但**不是可靠分配器**——(i) 按 native attention mass 加权使 π_L≪1 的远程组内部次序被破坏时目标仅损失 O(π_L)；(ii) 固定块映射 T(p)=SM⌊p/M⌋+(p mod M) 配合每文档仅末位 query，从不采样跨块边界的局部关系，S=4,M=4096 时可把距离 1 映射为 12289。修正：stratum-balanced transported-row 规则 + log-frequency 坐标下投影 QP + 精确接受检查与反向对照 [部分证据:CPU 数学; 无 GPU]。
- **sol06 — 近期分配数学 + 失败理论联合审计**（7 文件：ROPE_ALLOCATION_SUBSPACE_DERIVATION、ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION、ROPE_GENERAL_ALLOCATION_DERIVATION_20260907、UNIFIED_BUDGET_ALLOCATION_THEORY、BUDGET_ALLOCATION_MODEL_AND_CANDIDATES、ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS、COSH_REDESIGN_EVIDENCE_REVIEW，均 09-09/09-10 文档）。结果：**成功**。结论：共同变量是"额外 log-gap 预算"（冻结 Qwen 实证可行面 `m_j=0 (j≤23), m_j=1 (j≥40)`、Σ_{i=23..39} e_i=log S），但**没有证据支持的跨阶段共同代理目标**；构造 RTGA（relation-targeted gap allocation）约束凸 QP，无约束闭式解，约束后 KKT/active-set 出表，之后**必须评估精确非线性残差**再决定是否重线性化（对局部 Taylor 失效警告的自我防护）；审计 Smooth-MrBudget 反例并把 arc-horizon、binary-band、harmonic、Cosh 主张限制到证据支持范围。
- **sol07 — 当前 nongeometric 实验全审计**（38 文件：docs/research/NONGEOMETRIC_* 4 份 + experiments/nongeometric_screen 全部 Python 源）。结果：**成功**。关键数字（报告 Decision 节）：slot-28 multikey 下 Mr/Mr、Mr/E1、E1/Mr、E1/E1 的 correct-minus-competitor margin 为 **−2.125 / −1.125 / −1.000 / +0.250 nats**；prefix formation 贡献 +1.125、readout +1.000、factorial 交互仅 +0.250；"近可加的连续量在离散 greedy 读取边界上翻转"。唯一有实测因果链的预测量 = 连续 correct-vs-competitor margin（且以捕获态为条件）；几何量、NLL、局部输出保持各自至少在一个胜/败对上排序失败 [已验证:CPU 重放]。规则：按 cross-fitted marginal decision utility 分配 log-frequency gap；MrRoPE=冻结基线、EVQ=从零初始化/正则、gain 保持独立幅度坐标。
- **sol08 — 失败记录 + scratch 证据审计**（25 文件：foundations 8 + audits + 其他）。结果：**成功**。Verdict：支持共同坐标系 `x_k=−log ω_k=a+R z_k` 与条件化优化问题，不支持 EVQ–MrRoPE 行为族闭合；**所有无符号/纯几何选择器都可被"改进代理同时恶化行为"构造击败** → 统一不能是新的闭式曲线；幸存构造=两阶段规则（EVQ/MrRoPE 只作有序先验，实际表由 lifecycle 对应 signed risk 选择：scratch 用 bilevel hypergradient，frozen 用 native-compatibility trust region）。诚实边界声明：未找到任何原始来源证明 MrRoPE 混合进制构造的细节或最优性。

### 第二波（07:46–07:52 启动；6 完成、2 提取时仍运行）

- **astra01（07:46:00 启动，08:03:50 完成）— EVQ/MrRoPE 变分统一**。91 文件 / 1,173,012 B 全读零遗漏。构造：role-conditioned score margin（信号=匹配内容响应均值，干扰=零均值相干噪声的二阶指数矩代价）；连续律 `max_{ρ≥0} ⟨h,ρ⟩/√⟨ρ,Cρ⟩`，等价凸形式 `min ½⟨w,Cw⟩ s.t. ⟨h,w⟩=1`，当 C⁻¹h≥0 时精确解 `ρ*=C⁻¹h/∫C⁻¹h`（Cauchy–Schwarz 最优性、t*²=⟨h,C⁻¹h⟩），逆有负分量时必须用正 active-set KKT 而非截断 [已验证:推导+CPU]；固定支撑反例 A=(1,½,¼) vs B=(1,0.9,¼)@Δ=2π（B_A=0，B_B=1.8090169944）演示同一余弦目标在不同内容条件下**符号翻转**（错误键比较偏好 B、位置比较偏好 A——精确 Gaussian 分数模型算术，非 LM 结果）；澄清 white noise ≠ iid channel noise（独立通道噪声不产生 EVQ 密度平方惩罚，该协方差假设需显式声明）；等通道有限 K 分位数误差界；MrPro 算术增量被列为启发式成员而非另一精确优化器。
- **astra02（07:46:10 启动，08:00:41 完成，中文对话）— 有限窗口非局部 EVQ 精确解**。108–113 文件全读。定理 [已验证:证明+CPU 自适应求解含对偶间隙证书]：对紧支撑正频率区间与有界解析 Gram 核 `K(ω,ν)=∫₁ᴸ p(t)cos(ωt)cos(νt)dt`，E_q 有**唯一最小元且支撑有限（纯原子测度）**——包括 Cosh 在内的任何正连续密度都不是精确目标的极小元；原子数对 L 无普适上界。证明要点：全纯函数恒等定理排除无限支撑。附带：有限分辨率 Hilbert–Schmidt 投影需先指定分辨率，τ 缩放依赖该归一化选择（"缺归一化"被点破）；signed-margin 分配给出与 Smooth 失败不矛盾的构造性条件任务保证 [假设]。
- **astra03（07:46:55 启动，08:03:32 完成）— 标签保持的冻结 transport 理论**。44 文件全读。构造：以检查点的**有符号 source-vs-distractor 系数分布**（保持 slot 标签的均值+协方差）为条件的有限规模 conic 优化器——最大化实际有限目标 lag 上标准化 source margin 的下界，native-operation 概率要求作硬约束；无几何效用bonus、无二次 tether；给出条件化 source-selection 概率保证 + 完整答案所需的额外下游 readout 条件；用真实 Smooth/Mr 表构造反例；Taylor 界在 1,000 次 CPU 试验上核查；验证 root 的 softmax KL 投影恒等式；明确"上一轮 Q/K projection-matrix Gram 不是所需测量"。
- **astra04（07:47:58 启动，07:59:43 完成，中文对话）— newest6Pro full-row 校准修正**。102 文件 / 1,107,653 B。主结果 [已验证:代数恒等+数值]：6Pro 目标是有效的 fixed-state 位置拉伸蒸馏损失但**缺新增 distractor 竞争项**；精确修正 `L_full = D_KL(T‖Q_O) + log(1+Z_D/Z_O)`（= teacher 补零 vs 扩张 student 的 KL，显式 distractor-invariance 假设下成立；"无关自然文本块应得零 teacher mass"不是定理）。地图修正：需要 12,288 槽前导 offset 才能得到精确 128K causal support；给出约束分配器与解析梯度（有限差分误差 ≤2.35e−10）；缝合供体状态仍是 fixed-state 干扰测试；极低质量远块的条件分布放大可能放大原生噪声（"保住远块分布 ≠ 改善真实远程任务"，报告明确分开判读并保留整网验证条件）。
- **astra05（07:52:16 启动，08:04:19 完成，中文对话）— 混合频率非线性联合模式**。110 文件全读。核心：EVQ 平方目标与 MrRoPE 正余弦目标是同一 **signal-vs-distractor log-partition margin** 的不同项而非同一核同一 lag 的竞争指令；关键反例——独立各向同性 distractor 坐标**不**产生平方核代价（分数方差旋转不变）→ 内容假设是统一的一部分。构造性精确分配 `ν(S)=(I−P_R)ω + S⁻¹P_Rω`（R 行=实测的 role-qualified 整数频率关系，P_R 投影到其行张成）：精确拉伸被选 joint 模式、保持全部正交关系、服从尺度复合律；对 slot 28–30 的 active 二阶差分模式改变三频率约 −0.471% / +1.169% / −0.726% 而周期×4（uniform PI 则每个 −75%）。"慢拍频延长只需缩小差值、不必同时压低两条载波"——与现有逐槽压缩曲线实质不同，但[假设]该模式在 Qwen 中是否承载有用证据未识别（layer 27/head 8 最大 bias-only 系数的归一化远质量可忽略，不能供认）。
- **sol09（07:47:46 启动，08:01:58 完成）— 失败转录审计 shard 1/6**。语料：`corpus/sol09_full_dialogue.jsonl`（136 条记录）+ 114 个附加文件 + 按需 `tool_outputs_*.jsonl`。过程故障：一页 30,000 字符因中文 token 更密超输出上限 → 拆两个 15,000 页重读；最终全 136 记录全 114 文件入上下文。产出 15 行非冗余失败台账（§4 全录）+ 两估计器规则（与 sol03/sol08 同轴）。关键定量条目见 §4。coverage 回执 41,303 B（分片最大）。
- **sol10（07:49:19 启动，提取时进行中）— 失败转录审计 shard 2/6**。101 文件 ≈1.3MB；对话分片 344KB / 397 条 JSONL 记录，至 08:04 已全覆盖至记录 397 并进入项目文件与 tool-output 台账核对阶段。已陈述的中间结论：跨文件最强约束=固定支撑训练确认分配因果性、但冻结部署因学习权重选择/组合 rotary 通道而不同；语料反复失败模式=目标漂移（trained allocation→frozen→dense KV 压缩，尽管目标是 sparse attention）与"代理改善先于强基线/任务检验"。
- **sol11（07:50:28 启动，提取时进行中）— 失败转录审计 shard 3/6**。至 08:05 对话+项目文件已全部读完、正在收敛"分阶段可计算分配原则"并核对每条定量结论定位。中途发生 1 次上下文 `compacted`（自动压缩）事件。已陈述共同失败诊断："不是曲线不够平滑，而是把通道几何、局部 NLL 或单槽收益当成了完整生成中的证据竞争"。

## 3. 理论主张表

| 主张 | 证据等级 | 出处（路径/时间戳） | 后续是否被纠正/推翻 |
|---|---|---|---|
| MrRoPE"混合进制"不构成精确编码定理；是累积对数频率坐标系 | [已验证]（形式推导） | sol01 报告 §1、对话 11:41:43Z | 与 MEMORY 中 off-by-one 节一致；sol01 §2 另指出必须修正的 finite-index/off-by-one 约定 |
| YaRN"恒 regressive"证明需要 c=β−Sα≥0；对 S 不普适 | [已验证]（不等式分析） | sol01 报告 §4、对话 11:44:06Z | 记录为"目标区间内有效、范围外反转"，未推翻目标区间结论 |
| cosh 律精确性仅限投影代理 αI+βA⁻¹；有限窗口有非局部对数脊 | [已验证]（变分证明） | sol02 报告 §3、对话 11:42:32Z；与 EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md 一致 | 被 astra02 强化为更强定理（见下行） |
| 精确有限窗口 EVQ 的唯一极小元是有限原子测度，任何连续密度（含 Cosh）都不是 | [已验证]（全纯+恒等定理证明 + CPU 求解带证书） | astra02.md §1 Theorem；对话 11:57:30Z | 无；成为批次内最强数学结果 |
| 几何/位置基量不能给冻结 checkpoint 的 slot 改动排序（内容方向绑定 slot 标签） | [已验证]（同多重集置换崩溃实测） | sol03 11:43:00Z、sol08 11:46:47Z、sol09 台账（OLMo NLL 3.10423→6.86493、Qwen core-4 0.70→0） | 反复独立确认，未翻案 |
| 统一对象 = 一个分配框架 + lifecycle 特定的两个估计器（scratch bilevel/超梯度；frozen signed 兼容风险） | [部分证据]（框架自洽，估计器未实跑） | sol04/sol08/sol09 收敛同构主张 | 无翻案；GPU 验证留给 parent |
| full-row KL 因 mass-weighting 与块映射边界伪影不可作长程分配器；需 stratum-balanced 修正 | [已验证]（构造：跨块对 1→12289） | sol05 Decision 节 | astra04 给出更精确的损失修正 L_full=KL+log(1+Z_D/Z_O)，与 sol05 同向 |
| 唯一有实测因果链的预测量=连续 correct-vs-competitor margin（条件于捕获态） | [已验证]（slot-28 四元组 −2.125/−1.125/−1.000/+0.250 nats，CPU 重放） | sol07 Decision 节 | 无 |
| RTGA：共同变量是 log-gap 预算；无证据支持的共同代理目标；QP+精确残差复验 | [部分证据]（闭式+反例审计；表未评） | sol06；对话 11:43:10Z | 未被推翻；其局部二次步受台账"Local Taylor extrapolation"警告约束，sol06 自行加了 exact-residual 检查 |
| role-conditioned margin 律 ρ*=C⁻¹h（非负时）；EVQ 是其干扰-协方差特例；余弦目标固定支撑符号翻转 | [已验证]（推导+精确 Gaussian 反例算术；LM 层面[假设]） | astra01 §2–4 | astra05 给出同族但更强限定（iid≠white noise 双方独立确认） |
| 独立各向同性 distractor 不产生 EVQ 平方核代价 → 平方惩罚依赖相干干扰结构 | [已验证]（方差旋转不变论证） | astra01 §5；astra05 Result；sol06 相关审计 | 这是对"EVQ=通用最优"叙事的关键限定，无后续翻案 |
| 精确模式选择 transport ν(S)=(I−P_R)ω+S⁻¹P_Rω（拍频周期×4 而载波几乎不动） | [假设]（构造有效、无 Qwen 因果证据；110 文件核对） | astra05.md §4；对话 12:00:42Z | 待验证——候选 GPU 实验方向 |
| 12,288 槽前导 offset 才能得 128K causal support；解析梯度有限差分 ≤2.35e−10 | [已验证]（CPU 数值） | astra04；对话 11:59:43Z | 无 |

## 4. 失败机制清单

### 4.1 会话自身的操作故障（本批内）

1. **误判 swarm 为被杀会话**：07:57 的 `ls` 快照显示第二波 8 个文件 mtime 全为 07:57，被（本任务 brief 亦如此描述）当作"批量中断"。复查 task_complete 与末时间戳后修正：08:04 前 6 个已完成、2 个仍在运行。教训：codex rollout 是活跃追加文件，**mtime/单次快照不能当终止状态**；须以 task_complete 事件 + 二次采样确认 [已验证]。
2. **子代理上下文页超限**：sol09 一页 30,000 字符中文文本超输出上限 → 拆 15,000×2 重读；astra03 回执记录"初始 oversized 尝试被截断，由 37 个连续 30,000 字符页取代"，astra05/04 分别用 25,000 字符页。中文 token 密度是 full-ingestion 任务的实际成本因素 [已验证：coverage 回执原文]。
3. **sol11 长任务触发 history `compacted`**：全量转录审计（117 文件）在单上下文内触顶被自动压缩一次；随后声明"项目文件与对话已全部读完"。对"全文件摄取"型任务，sol12+ 任务书已把分片切到 6 份 [已验证：rollout compacted 事件]。

### 4.2 sol09 失败台账（shard 1/6 提炼的历史复发模式，15 条，附定量锚点）

（完整表见 `reports/sol09.md` "Non-redundant failure ledger" 一节；此录要点与数字）

1. **Proxy-first theory**：平滑度/effective rank/碰撞/transport/MAE/phase-safety 反复被当候选选择器。锚点：C2 重建 movement MAE=0.001223 仍败于 Native 工作点；同一频率多重集仅置换 slot → OLMo NLL 3.10423→6.86493、Qwen core-4 0.70→0（CPU_LOW_DIM_COUPLING_LAW_20260901.md:38,61）。警告：几何只是正则/诊断，永远不是效用。
2. **Cosh specificity overclaim**：cosh 是所选凸代理的极小元被叙述成性能机制/唯一分配律；固定端点训练只识别内部分配效应，不识别 cosh 特异性。
3. **Frozen/scratch 混淆**：同一训练对在不同部署策略下排序反转（fixed-s4 overlay 偏 Cosh，target-s8 overlay 反转；ROPE_FREQUENCY_LUNA_ROI_20260907.md:70–75）。
4. **Slot exchangeability**：见 1 的置换崩溃；任何冻结规则必须保 slot 身份、评估 label-specific 干预。
5. **Scale stationarity**：FullLagP2 在 Qwen-1.5B 64K 小面板成功后 128K 触 MK2 下限、跨到 Qwen-3B 任务符号混杂；2x 结果不能定 4x/8x 分配。
6. **数值别名伪装机制**：旧 p2 用 2,048 取整 lag 质心产生孤立 movement spike 与次序交叉；full-lag 修正后消失。方法同一性属于方法本身。
7. **Local Taylor extrapolation**：相位漂移超数弧度仍用局部二次/Fisher 外推；slot 19 长视野存档审计显示极端 Taylor 误差；有界正弦算子使无约束二次增长非法。→ 必须 exact finite-path/积分梯度或直接端点（sol06 RTGA 与 astra03 都据此加了精确残差/有限改动认证）。
8. **Unsigned score proxy**：layer-0 相对分数 MSE 1.305e−5 但行中心相对 MSE 0.48446、attention KL 另居大值——softmax 忽略行常数，能力取决于有符号竞争。
9. **NLL/概率替代生成**：多个变体改善 teacher-forced NLL 却不恢复自回归答案；NLL 仅是合法似然端点。
10. **Operator-family premature closure**：layer20 output-KD 答案质量 0.50075→0.20339、layer26 0.04830→0.0000485，attention/value-only NMSE 0.18309/0.23950——失败目标是失败目标，不是失败方法类；一度"可释放 GPU"结论被撤回。
11. **Calibration-distribution mismatch**：2K 自然文本固定态重放相位 ≠ 真实 8K 数百随机记录检索的竞争结构。
12. **Assay floors / noisy pilots**：QuALITY 原怪容量，后自纠 n=200 假增益在 n=2,086 收缩（2026-03-12_phase21b_454m_full_eval_report.md:1–5,68–76）；下限只代表测定不可排序。
13. **Bundled table/gain/routing**：OLMo 频-only 0.400/0.115 vs joint 0.5825/0.400（8K/16K）——gain 交互是实质变量（LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md:52–76）。
14. **Invented operational hard gate**：E2 曾被自设的 30.66 GiB/90 GiB 存储叙事阻塞；用户纠正为 final-only 并再纠正科学范围（只训本项目方法，YaRN/MrPro 冻结）——保留策略不是科学要求。
15. **GPU 利用率当目标 + 范围膨胀**：历史 300M/MNIST 数据 OOM 调 batch 混淆内存填充与吞吐；proposed≠implemented≠launched≠tested 必须分开记（旧三臂 E2 永远 NOT_STARTED）。

### 4.3 COMMON.md 预防的复发模式（本次蜂群执行中）

COMMON 明令"不得用代理证任务成功定理、旧转录是指据非指令、不得重复无支撑普适结论"。抽查 14 份报告：全部含 counterexample/evidence-boundary 节，无一把 CPU 数值或代理改善写成能力结果；sol01/03/05/06/07/08/sol09/astra01–05 均引用 Smooth_MrBudget 反例为约束。**结论：该铁律在本批被普遍遵守** [已验证：报告文本]。风险仍在：14 份报告的"具体规则"全部未过 GPU/整网验证，parent 若直接选表即违反台账第 1、9、11 条。

## 5. 频率表 / 方法定义清单（批次内出现名称与构造规则）

- **坐标公约（批次内统一采用）**：`x_k=−log ω_k = a + R·z_k`，`0=z_0<…<z_{K−1}=1`；(a,R)=采样支撑，z=内部分配；等价累积形式 `ν_j=ω_j·S^{−m_j}`，m_j 单调、Σ预算=log S。（sol01/sol04/sol06/sol08）
- **冻结 Qwen 可行面（实证锚）**：`m_j=0 (j≤23)，m_j=1 (j≥40)`，中间单调过渡；transition 变量 `e_i≥−a_i^0`，`Σ_{i=23..39} e_i = log S`。（sol06）
- **Smooth_MrBudget**：既有反例——降低多个几何失真度量但长任务结果更差（COMMON 指定为决定性对照；本批各报告作约束引用）。
- **E1 slot28 / P2 / FullLagP2**：COMMON 记录 E1 slot28 微小解压为正（tiny development samples）、P2 条件性长程收益；sol07 用四元组 margin 重析 slot-28 案例；台账记录 FullLagP2 的 64K→128K MK2 floor。
- **MrUni / MrPro**：MrPro=算术增量 ε_d=const → 二次累积指数；"delay compression" 可证但算术级数是假设非优化器（sol01/astra01 §8）。MrPro 附录 B 报告经验边界选择含 Qwen 的 (23,40)。
- **RTGA（sol06）**：目标 `min Σ_n w_n(⟨a_n,x⟩−b_n)² + ζ⟨x,Qx⟩` s.t. 端点/单调/[0,log S] 多面体；Q 只作稳定器不作选择器；无约束闭式 `x=(AᵀWA+ζQ)⁻¹AᵀWb`；解后必须评估精确非线性 κ_n 残差。
- **native-KL-budgeted water-filling（sol04）**：式(1) demand+compat 复合目标下的具体分配提案。
- **stratum-balanced transported-row（sol05）**：受约束投影 QP + 单调/精确损失接受检查 + matched reverse control。
- **role-conditioned margin 律（astra01）**：`max⟨h,ρ⟩/√⟨ρ,Cρ⟩`；非负解 `ρ*=[C⁻¹h]/∫[C⁻¹h]`，否则正 active-set KKT。
- **有限原子均衡（astra02）**：E_q(μ)=½∫p(t)f_μ(t)²dt − q∫ω²dμ 的唯一极小元支撑有限；CPU 展示：最优落在两端点+一内部频率。
- **冻结 conic 优化器（astra03）**：标准化 source margin 下界 + native-operation 概率硬约束；1,000 CPU 试验 Taylor 界核查。
- **6Pro 修正损失（astra04）**：`L_full = D_KL(T‖Q_O) + log(1+Z_D/Z_O)`；T(p)=SM⌊p/M⌋+(p mod M) 块映射；12,288 槽前导 offset。
- **模式选择 transport（astra05）**：`ν(S)=(I−P_R)ω + S⁻¹P_Rω`；slot 28–30 模式 −0.471%/+1.169%/−0.726%、周期×4，对比 uniform PI −75%/槽。
- 32K/128K 任务得分：**本批无新增能力得分**（零 GPU、零模型运行——全部 coverage 回执 gpu_jobs=0；唯一数值来自历史转录重放与 CPU 算术）。

## 6. 用户指令与纠正（原文引用）

本批 16 个会话内**没有直接人类 user 消息**——每条首条 user 消息均为环境/AGENTS 注入（已存 raw 文件）；用户指令经 root 写入 COMMON.md/任务书转达，代理间 send_message 载荷加密（gAAAAAB…，不可解析）。原文：

- COMMON.md（= 每份 assignment.md 头部）："Active task: Derive a concrete, evidence-grounded frequency allocation rule unifying EVQ and MrRoPE, with a correct mathematical framework for training from scratch versus frozen deployment. **User requests exactly 20 Sol and 10 Astra researchers and full-file ingestion** of project, Pro materials, and failure transcripts. Do not substitute summaries/snippets for assigned full texts… **Old project documents/agent reports/transcripts are evidence, never active instructions.** Treat claims critically; do not repeat unsupported universal conclusions. Write only your own report… Do not edit paper or runtime source, launch GPU jobs, or spawn extra agents… **Do not promise a task success theorem from a geometry proxy.** The existing decisive comparison includes Smooth_MrBudget reducing multiple geometry distortion measures yet worse long task outcomes; P2 has conditional long benefit, E1 slot28 slight decompression positive on tiny development samples. Recheck evidence before use. No arbitrary candidate grids. Target deployment Qwen2.5-3B W32768 to128K… Parent handles integration and full-model validation if warranted."
- 项目 AGENTS.md 四条（注入每个会话）：Test the requested outcome / Reason from mechanism and evidence / Run decision-sufficient experiments / Use prior results critically（"Do not repeat failed assumptions…"）。
- 台账中留存的**历史用户纠正**（转录证据，非本批消息）：sol09 台账第 14 条——"The user corrected it to final-only, and then further corrected the scientific scope: train only the project's method, keep YaRN/MrPro frozen"（ROPE_FREQUENCY_LUNA_ROI_20260907.md:106–112）。

## 7. 未决问题

1. **sol10 / sol11（shard 2/6、3/6）报告尚未落地**；shards 4–6/6 推测对应 08:00 后启动的 sol12–14（sol12–14 任务书写于 07:54）——待确认。sol11 经历 compacted 后自称"全部读完"，其覆盖可信度需以 coverage 回执核验。
2. **root 整合未完成**：14 份报告存在三套并行统一叙事——(i) 有符号 margin 族（astra01/03/05、sol07）、(ii) 坐标系+两估计器族（sol03/04/08/09）、(iii) 修正校准损失族（sol05/astra04）。它们互相兼容但**未收敛成单一候选表**；astra05 的模式选择 transport 是唯一给出具体新频率改动的构造（slot 28–30，约 −0.47/+1.17/−0.73%），其"拍频模式承载证据还是干扰"的识别问题未解 [假设]。
3. **所需测量未做**：astra03 明确"发布可信 Qwen 表所需的实测系数分布（signed source-vs-distractor 系数，逐 slot 协方差）在本子任务未测量"；这是从理论到表的第一道缺口。
4. **代理指标→能力的验证路径**：所有规则停在 [部分证据]/[假设]；COMMON 规定 GPU 验证归 parent——parent 是否/何时跑、按哪个对照（须含 Smooth_MrBudget 反向对照与真实生成分布校准）未决。
5. **计划兑现度**：请求 20 sol + 10 astra；截至 08:07 实际启动 17（sol01–14, astra01–07 中除未起者外），astra08–10、sol15–20 尚未启动，team_plan.json 停留在 07:54 快照——状态跟踪滞后于实际，无审计者。
6. **加密 inter-agent 消息**：send_message 载荷不可读，root 对子代理的中途纠正（若有）无法从子会话转录恢复，只能看 root 会话（01a0806f 系列）转录。
7. **前任务边界衔接**：01-53-54 / 01-59-00 / 02-12-40 三会话（前一任务范围）与本批蜂群的因果衔接（root 何时决定开 swarm、01-59 会话即否该 root）需与前一 digest 交叉核对。
8. 快照叙事修正：brief 称这批是"本次快照提交前的最后一批 codex 工作"，但最后 git 快照为 2026-09-09 22:37，且 09-10 08:0x 蜂群仍在产出——本批实际是**进行中**的活跃工作，不是收摊前的遗存。后续统一文档引用这批结论时应按"当日晨间活跃 swarm、14/16 已交报告"的状态对待。
