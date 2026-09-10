# digest pro-materials

范围：项目收到的全部"pro"外部评审/外部分析材料（gpt56-sol-pro、opus、qwen-panel、pro-materials-20260908 九份、四份方法/审计文档、发给 Pro 的决策请求、以及"姐夫稿"=6Pro 分析），对照项目方裁决文档 `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md`（§8 为 6Pro 评估）与 `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md` 逐条给出：核心主张 | 采纳/否决裁决 | 尚未消化的可用点。证据标注遵循铁律：[已验证]（有回执/复算）/[部分证据]/[假设]；不得把代理指标说成能力结果；不得把未测写成否证。

## 1. 来源清单（文件路径/大小/行数）

根目录 `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/` 下（wc -c -l 实测，2026-09-10）：

**路由文件**
- `paper-2027/research/external-reviews/README.md` — 3,616 B / 24 行。全部外部材料的"当前用途"权威表。关键句（原文）："External reviews are source material, not validated research owners. Their `VERIFIED`, `mandatory`, `decision` or `priority` labels do not grant project scientific authority or execution permission."

**gpt56-sol-pro-20260825/**（单文件）
- `README.md` — 81,525 B / 2,552 行

**opus-20260823/**（6 md + 1 evidence JSON）
- `01_VERIFIED_STATE_20260823.md` 14,266 B / 229 行
- `02_THEORY_SUPPORT_VS_ALLOCATION.md` 13,569 B / 250 行
- `03_EXPERIMENT_PLAN.md` 10,019 B / 214 行
- `04_MANUSCRIPT_PLAN.md` 7,992 B / 150 行
- `05_PROBLEMS_AND_RISKS.md` 9,280 B / 171 行
- `README.md` 4,333 B / 74 行
- `evidence/OPUS_RECOMPUTATION_20260823.json` 44,461 B / 1,546 行（重算回执）

**qwen-panel-20260826/**（8 文件）
- `README.md` 2,949 B / 50 行；`01_journal_fit_review.md` 17,418/59；`02_methodology_review.md` 19,387/79；`03_perspective_review.md` 24,428/84；`04_domain_review.md` 24,790/91；`05_devils_advocate.md` 22,354/104；`06_editorial_decision.md` 25,627/175；`07_post_review_delta_note.md` 4,267/37

**pro-materials-20260908/**（9 文件，GPT-6-pro 系产物）
- `hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md` 113,719 B / 1,734 行（用户确认"这是我让GPT6pro生成的一份详细的计划"，`docs/research/USER_PROMPT_TRANSCRIPT_20260909.md:996`）
- `RoPE_Exponent_Allocation_Unified_Plan_20260905.md` 63,679 / 897
- `HYBRID_ROPE_NEXT_DAY_PLAN_20260906.md` 42,969 / 645
- `RoPE_ICLR2027_Research_Guidance_20260905.md` 50,398 / 579
- `HYBRID_ROPE_FIRST_PRINCIPLES_SYNTHESIS.md` 48,483 / 877
- `SINGLE_TABLE_FFN_REPORT_AUDIT_AND_CODEX_GUIDANCE_20260904.md` 41,061 / 699
- `RoPE_Round12_YaRN_Limits_and_Experiments_20260906.md` 38,132 / 590
- `RoPE_ICLR2027_Major_Revision_20260906.md` 23,331 / 333
- `HYBRID_ROPE_TWO_DIRECTION_THEORY_AUDIT_20260905.md` 35,339 / 480

**方法/审计文档（位于 `paper-2027/research/external-reviews/` 直下）**
- `MRPRO_BOUNDARY_MATCHED_SOURCE_20260908.md` 8,000 B / 382 行（BM 闭式来源）
- `ROPE_ICLR2027_CROSS_AUDIT_20260906.md` 52,273 B / 1,016 行
- `ROPE_LOWFREQ_CARRIER_METHOD_20260907.md` 13,183 B / 268 行
- `ROPE_SCALE_TRANSPORT_METHOD_AND_CODEX_20260907.md` 29,764 B / 476 行
- 相关：`paper-2027/research/CROSS_AUDIT_EXPERIMENT_PROTOCOL_20260907.md` 10,237 / 152

**发起方文件**
- `docs/research/ROPE_PRO_DECISION_REQUEST_20260907.md` 10,924 B / 78 行（项目发给 Pro 的决策请求，含当日事实表）

**裁决参照（项目方）**
- `docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` 16,571 B / 143 行（§1–7 = "GLM"统一预算模型；§8 = 6Pro/姐夫稿评估）
- `docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md` 7,486 B / 72 行（同日复核，部分推翻 GLM）
- `docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md` 24,928 / 248（对 pro-materials 的裁决文档）
- `docs/research/ROPE_SOURCE_COVERAGE_20260908.json` 7,063 / 237（9 份 pro-materials 的 SHA 台账，状态 "SOURCE_INPUTS_NOT_VERIFIED_CLAIMS_OR_EXECUTION_AUTHORITY"）

**姐夫稿（6Pro）原文：未归档 [已验证的缺席]**。`grep -rn "姐夫" docs paper-2027` 唯一命中 `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:113`（§8 标题"外部分析（'姐夫稿'）"）。`find -iname "*glm*" -o -iname "*jiefu*"` 无独立原文文件。其内容只存在于两处二手记录：UNIFIED §8（采纳清单）与 GLM_6PRO_REVIEW（复核+验证计划）。这是溯源缺口，digest 全部 6Pro 主张均标注出处为这两份文件。

## 2. 任务时间线（目标→方案→结果+关键数字）

通用设定 [已验证]：Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`；64 rotary 槽；W=32,768；S=4；L=131,072；BF16 Flash SDPA；greedy；官方 gain = 1+0.1·ln4 = 1.1386294361（g²=1.2964769928）。36 行 dev 面板（13 任务族×长度格）。

1. **2026-08-23 opus 审计**。目标：核查"support vs allocation"理论底座。方案：重算 R/log-gap/等效基缩放，产出 6 文件+证据 JSON。结果 [已验证]（数学部分）：OLMo 全部拉伸臂共享同一 R=14.303621（精确）；Qwen 快端点被 stride-16 混叠位移 0.17%（DEFECT-3）；同支撑几何 ≡ NTK-aware base scaling θ'=θ·s^{d/(d−2)}=θ·4.088994（OLMo θ′=2,044,497.12、Qwen 4,088,994.24）；budgeted 表 ≈ 2 参数 YaRN ramp（平均 |log₂ 偏差| 0.00876/0.01171）。发现 DEFECT-1（Qwen binary 臂 180 行从不走 Native 分支）、DEFECT-5（Qasper 全 200 binary 0.2457 vs YaRN 0.1803，+0.0654 CI[+0.022,+0.110] p=0.002，无 owner）。RULER-13：binary 0.6772@8K vs YaRN 0.2382；0.5440@16K vs 0.0588；Qwen 64K macro 0.6700/0.6025/0.5450；128K 样本 n=5→20 后均值移动 −0.0725（探针偏置示例）。
2. **2026-08-25 gpt56-sol-pro 备忘**。目标：给 ICLR 方向。方案：Direction A 使用条件离散谱组合（J=2–4 codebooks）、Direction B 双基谱桥+切向传输（gate 日程 Q/R/S 损失）、实验 0 指标混战决策表、7 条停做清单、若全败的退路定理。结果：未执行；README 裁定用途 "None — frozen audit provenance only"。其中"水床权衡归因不成立""固定 support 是实验控制不是自然定律"两条后来被 GLM_6PRO_REVIEW 以不同方式部分吸收。
3. **2026-08-26/27 qwen 五席 panel**。目标：投稿前评审。结果 [已验证]：Major Revision，五席均 6/10；DA-1 成立（目标匹配反转 +0.060/+0.227/+0.460 nats 未披露）、DA-2 部分成立、DA-3 纠正后成立（Cosh 平局 p=0.836、规则点 p=0.125、1.25× p=0.027 多重性）；Proposition 2 的"因子 2"主张被**驳倒**（19/12600 常数，比值 1.00058）；32/73 参考文献未引用；"split rule"全文无定义。修订周期 2026-08-28 关闭 @93d7eac（记忆：author 保留全部未执行的 accepted 项 A2/A7/A8/A9/A12/A13/A14/A15/Step-1）。
4. **2026-09-04～06 pro-materials 九份入库**。用户投喂 GPT-6pro 产物（dossier 1,734 行等）。结果：2026-09-08 `ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md` 做 10 来源/7,310 行全读裁决：根因是**推理链跳步不是曲线**；谱系表 10 行（shape/theta 37648/62474、Phase16 99 运行、tau=1 ω≈0.72967 rad/token、QuALITY、YaRN identity、stride-16 混叠、静态几何→生成、Native compact、rank≈2000、oracle routing）；ALS 假收敛修正 2.848193371→2.836061317（43 轮）；LoRA 容量上界主张撤回；不可辨识定理收窄到可证版本。
5. **2026-09-06 Cross-Audit（1,016 行）**。统一风险模型 u_n=ηΣ(I−ηH)^t d、R_D(n)=q_D−2b^Tc_n+c^TG_Dc_n；KL 判定边界 κ(p)=a·log(2a/(2a…))（p=(0.5001,0.4999)→KL 8e−8 即翻转 argmax）；γ=1/2 守护损失；秩界 Σ_{j>r_Q+r_K}σ_j(E)²；统计口径（McNemar p=0.25/0.375、5pp 功效需 n≈627、3-seed 符号检验 p=0.25）；E0–E5 共约 100 GPU-h；论文三形态 A/B/C；ICLR 截稿 abstract 9/18、全文 9/25 AoE。结果：设计输入，E0–E5 大部未执行。
6. **2026-09-07 scale transport + carrier 两方法**。Scale transport（`ROPE_SCALE_TRANSPORT_METHOD_AND_CODEX_20260907.md`）：逐槽 β_j，χ_ij=p_i²a_ij²‖W_O(v_i−o)‖² 响应距离分布、分位数 LS 得 r_j、有界保序回归、五点 λ 网格+D_N/B_− 回放守护。试点结果 [已验证]：**仅 λ=0 通过 → REFERENCE_ONLY**，即无新表；`docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md` 记载"F1 增量全部来自两行的措辞/词面差异"、最大实测输入 65,408（不得声称 128K 结论）。Carrier removal（`ROPE_LOWFREQ_CARRIER_METHOD_20260907.md`）：低频带 B={j:ω_j·L0≤π/2}（槽 47–63）ν_j=(ω_j−c)/s，闭式 c 来自相干复背景响应、允许负频率、|S_new|=|S| 恒等、相位约束 cL0/s≤3π/32≈0.295 rad。试点 [已验证]：dev 对比完成、队列停止（`ROPE_CARRIER_REMOVAL_PILOT_20260907.md`；"EOS14/50 是结束次数"非分数）；未产生被采纳的频率表。同日 `ROPE_PRO_DECISION_REQUEST_20260907.md` 把当日事实（λ=0 唯一守护通过、DN 22.44445/23.44834、5.38381→5.28431、QA Native 短 F1 0.498465 vs 提案 0.425/长 0.222024、尾频+CoPE 检索短 2/2、128K 8/8 但 UUID 1/2 与 0/8、VT 长格未测≠0、OLMo E1 near Z6/32 vs MrPro0/32、far Z1/32 vs 0/32、裁切控制 β≈0.91970 vs 真实 0.82778、j=24 位移 0.0065→0.912）发给 Pro 求解。
7. **2026-09-08 BM（Boundary-Matched）**。`MRPRO_BOUNDARY_MATCHED_SOURCE_20260908.md` 给闭式：min Σ(a_{i+1}−a_i)² s.t. Σa_i=1 → a_i=6i(N+1−i)/[N(N+1)(N+2)]，m_q=q(q+1)(3N+2−2q)/[N(N+1)(N+2)]，连续极限 3x²−2x³。dev 36 行 [已验证]：S4_GPU_COMPLETE、通过（协议 `ROPE_MRPRO_BM_PROTOCOL_20260908.md`，作者纠正入档："总体仍是最多10个有独立理论依据且经反向审查的候选……5分钟是预期成本，不是截断评测的时限"；OLMo revision `48d788eca847d4d7548f375ad03d3c9312f6139e`、FP32 SHA `fc0f443b1c58…`）。128K 扩大战 [已验证]：**BM 70.83% vs MrPro 78.13%（−7.3pp）**。诊断（`ROPE_BM_128K_DIAGNOSIS_20260908.md`）：O/C/P/L 四格 0%/20%、100%/100%、100%/100%、0%/60%；单 key 编辑（token74100 bizarre-slime→neutral-slime）后 BM 仍错（4068207）；相位差槽28 最大 31.25 rad、VT 例 41.49 rad 但 P 满分 → 相位差大非失败充分条件。同 Gap-capped/收窄 MrUni [已验证]：84.44%/62.15%，36 行 0 胜 7 负 29 平，COMPLETE/NO_LONG_GAIN（`ROPE_GAP_CAPPED_RESULT_20260908.md`，899.87s、21.89GB）。
8. **2026-09-10 UNIFIED/GLM + 6Pro 复核**。`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` 提出端点不变量模型（见 §3）并 §8 采纳/修正 6Pro；同日 `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md` 推翻 GLM 因果论证、确认 6Pro 数学、启动校准验证（`experiments/nongeometric_screen/pro_block_calibration.py`，服务器 `/root/autodl-tmp/rope_pro_block_calibration_20260910`；用户 11:06 告知 GPU 恢复并允许验证）。验证仍是采集计划，无结果（该文 §末："本节记录的是当前计划与已开始的采集，不是已完成的优化或模型收益"）。

## 3. 理论主张表（主张 | 证据等级 | 出处 | 后续是否被纠正/推翻）

| # | 主张 | 出处 | 证据等级 | 项目方裁决（对照 UNIFIED §8 / GLM_6PRO_REVIEW / README / synthesis） |
|---|---|---|---|---|
| 1 | 外部评审的 VERIFIED/mandatory 标签不授予执行权 | external-reviews/README.md | [已验证]（治理事实） | 全项目采纳为路由规则 |
| 2 | 固定支撑（support）下"拉伸"与"重分配"分解：(a,R,z)；全部 OLMo 拉伸臂 R 精确相等 14.303621 | opus 02 | [已验证]（数学重算，evidence JSON） | 采纳为分析语言；opus 自己声明"固定 support 是实验控制不是自然定律" |
| 3 | 同支撑几何 ≡ NTK-aware base scaling θ'=θ·4.088994 | opus 02 | [已验证]（等式代数） | 采纳（论文可用等价式）；未被任何 GPU 实验反驳或证实为性能主张——它不是性能主张 |
| 4 | budgeted 表≈2 参数 YaRN ramp（|log₂| 0.00876/0.01171）；band-split 等效旋转数 YaRN 29–30 段 vs 16.27/7.16（OLMo）、15.35/6.47（Qwen） | opus 02 | [部分证据]（描述性拟合） | 未裁决；三臂判别预测（same_support_geometric_s4 / nearest_yarn_ramp_s4 / converged_budgeted_s4）**从未执行** |
| 5 | 谱三分解：高频端位置 x₀、总跨度 A、归一化间隔分配 {aᵢ/A}；4× 窗口只需 +10.19% log-span（13.60→14.99） | 6Pro，经 UNIFIED §8.1 + GLM_6PRO_REVIEW | [已验证]（定义+算术） | **采纳**（统一坐标；Σm 只是可选额外控制，不等于总跨度） |
| 6 | EVQ 变量替换 h=φ′ 把目标精确写成间隔预算泛函 J[h]=½∫[α/h+β(1−u)²h]du，cosh 解；复核 1.0373147207 双坐标差 1.6e−11、KKT 残差 1.1e−15、端点 .104825834/.391244468 | 6Pro §EVQ，GLM_6PRO_REVIEW 复算 | [已验证]（复算通过，须固定离散约定） | **采纳为数学**；其"高频冗余→低频"直觉在端点约束下**被 HighGapToLong 否证**（32K−17.1/128K−10.8，36 行 0 提升），有效成分重述为过渡段预算位置问题（BUDGET §5） |
| 7 | J_r(c₀,ν)=频谱表示质量+(c₀−c*)ᵀ(G_r+ζI)(c₀−c*)（ridge 关系拟合配方分解） | 6Pro，GLM_6PRO_REVIEW | [已验证]（正定情形代数成立） | 采纳为组织工具（scratch/冻结替换/适配三分）；6Pro 自限"未证明 EVQ 是真实整网目标特例"被保留 |
| 8 | 校准-KL 规则生成器：真实 Q/K 完整 softmax-row 在块内固定尺度假设下可生成方向性表修正 | 6Pro §6（原文未归档） | [假设]（验证刚启动，无结果） | UNIFIED §8 **降级为假设生成器**；GLM_6PRO_REVIEW 批准做一次完整验证（4 步协议），旧 calibration/ 数据不可用（32K MK2 层27 仅存 221/32,123 keys） |
| 9 | ±v 镜像对照 + 三条件 position/interference 因子实验 | 6Pro，UNIFIED §8.2 | [假设]（未执行） | 采纳为协议（两项实验设计入可用清单），从未跑 |
| 10 | GLM 端点不变量 I1（j≤23 m=0）/I2（j≥40 m=1）由"零容忍定理"推出 | UNIFIED §1–2 | I1/I2 作为设计约束 [已验证]（HighGapToLong、MrUni 64.6@32K、E2 崩至 54.7 为支持性干预证据）；"必须"部分被推翻 | GLM_6PRO_REVIEW 第 3 条：少数失败干预不能证明所有高频改动均失败；MrUni 不是全表 PI、MrPro 增量非均布 |
| 11 | 守恒量 ln S=1.386 → 桥宽单参数 | UNIFIED §1 | 守恒 Σ ε_i=1 [已验证]（代数）；"17 log-gap 总和=ln4" [已被推翻]（实际 5.056039，原生部分 3.669745） | GLM_6PRO_REVIEW 第 1 条：16 个自由度；桥宽单参数是设计限制不是守恒推论 |
| 12 | D_j=W·S^{m_j} 是"未见相位弧"边界，36–39 危险区（r∈[1.15,2.2]、D=75–112K<128K）必须完成 | UNIFIED §1 | [已被推翻为因果机制]：native 36–39 已在 W 内转 2.199/1.772/1.428/1.151 圈，圆周相位已覆盖；E1 s28_less 完全未动 36–39 的 D 却修复 89K 行；LBS 实际 m .625169/.729476/.846534/.979914 → D 77,954/90,082/105,953/127,473 也非"完成" | GLM_6PRO_REVIEW 第 2/5 条；D_j 保留为描述性风险因子非门控 |
| 13 | "所有赢家都在把预算右移"（单一方向梯度） | UNIFIED §3 | [已被推翻]：质心表 MrPro Σm=29.333333/质心34.666667；E1 29.300653/34.699347（后移）；LBS 29.560179/34.439821、P2 34.178915/29.821085（前移） | GLM_6PRO_REVIEW 第 4 条：不同方向都可有条件收益 |
| 14 | gain 与相位正交（四格对照证明互不影响） | UNIFIED/PRO_DECISION | [已被推翻为效应独立]：softmax(g²z_ν) 依赖两者，存在非零交互；E3 gain074 98.3/75.3 只支持"长端不由 gain 得救" | GLM_6PRO_REVIEW 第 6 条；gain 作为可分离**控制量**仍可用 |
| 15 | 36 行 = 一条"距离—准确率"曲线，75–112K 危险带 [部分证据→否证宣称] | UNIFIED 引用 | [已被推翻]：MK2 参考距离≈88,726 仅是单绑定标签；VT min=20,563/max=105,880；MQ 一行同时含 77 与 120,934 token 目标距离；FWE 距离定义不成立（答案词 cyuvqn 出现 10,785 次、lobxbq 4,793 次遍布全文） | GLM_6PRO_REVIEW 证据距离复核节：不能据此宣称已验证危险带；需固定内容配对改距离 |
| 16 | BM 平滑性目标（min ΣΔa²）选表 | MRPRO_BOUNDARY + 实测 | [已验证为负]：dev 通过、128K 70.83 vs 78.13；诊断显示失败经由保留 KV 状态（P 100%/L 0%） | 闭式正确但"光滑度不足以保证部署"入档；不支持继续扫 cap/挪边界/表插值 |
| 17 | 响应距离分位数可反推逐槽所需缩放 | SCALE_TRANSPORT 方法文档 | [已验证为负（试点范围）]：仅 λ=0 通过 → REFERENCE_ONLY；最大实测 65,408 | 未采纳任何非平凡表；"不得把未测写成 128K 否证"仍适用于全谱 |
| 18 | 低频载波移除保 |S_new|=|S| 恒等、闭式 c 防背景漂移 | LOWFREQ_CARRIER | 恒等式 [已验证]（推导）；性能 [假设]：dev 队列停止，无采纳表；尾频+CoPE 检索短 2/2、128K 8/8 但 UUID 1/2 与 0/8（PRO_DECISION 当日表） |
| 19 | qwen-panel：Major Revision；Prop 2 因子 2 主张 | qwen-panel 06/05 | Prop2 [已被推翻]（19/12600 常数、比值 1.00058）；DA-1 nats 未披露 [已验证成立并纠正] | 采纳：修订周期 08-28 关闭 @93d7eac；unexecuted accepted 项由 author 保留待新周期 |
| 20 | 统一风险模型/KL 判定边界/γ=1/2 守护/秩界 eq(11) | CROSS_AUDIT | [已验证]（数学）/ 性能含义 [假设]（E0–E5 未跑） | 用途 "Historical design input"；统计口径（n≈627、McNemar、3-seed p=0.25）被后续冻结队列采用 |
| 21 | 使用条件离散谱组合（Direction A）与双基谱桥+切向传输（Direction B） | gpt56 | [假设]（未执行） | 冻结为 provenance；A 与零训练约束的相容性从未检验；B 需要可训练 gate，超出当前赛道 |
| 22 | FFN/微调可能是"阿克琉斯之瞳" | 用户假设（transcript:1006）+ SINGLE_TABLE_FFN 审计 | [假设]（无零训练反证、亦无微调实验） | 未裁决；构成 §7 未决问题之一 |

## 4. 失败机制清单（试过什么、为何失败、复发模式警告）

1. **代理→能力跳步（首要根因）** [已验证的模式性结论]。`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`：根因是推理链跳步不是曲线。实例：ALS 假收敛（2.848193371→2.836061317，43 轮修正）；MAE/KL 均值当能力预言；oracle routing 冒充部署算法；E8 单槽置零 fixed-state 代理过度预言（代理预测远差于实际 −13.9pp@128K 50.6，或反向）——固定状态指标与完整 prefill 之间从未建立迁移证明。警告：任何 proxy 最优解不得自动上 GPU（Gap-capped 文档明文："不能把又一个 proxy 最优解自动送到 GPU"）。
2. **Y2 gain 恒等式错误** [已验证已纠正]：√a=1.0670658068 变体是 non-faithful；官方 gain 唯一为 1.1386294361。复发风险：任何对比若 gain 不同即无效（GLM_6PRO_REVIEW 验证协议特意"所有评估使用相同 gain"）。
3. **采样规模不对称** [已验证]：opus 记录 128K n=5→20 使均值移动 −0.0725——探针偏置足以制造 pp 级假象。冻结队列因此配 36 行面板+NLL 48+确认实验（0450/0451）。
4. **同支撑混叠缺陷** [已验证]：DEFECT-3 stride-16 混叠位移 Qwen 快端点 0.17%；DEFECT-1 使 180 行 binary 从未路由 Native 分支（该数据集结论作废）。警告：路由分支与混叠校验必须是每轮预检项（synthesis"三项执行问题"：hash/smoke 脱离主线、实验前没审代码、判断标准混乱）。
5. **BM 的教训：目标函数正确 ≠ 部署优越** [已验证]。闭式与独立 LP 验证均对，候选满足更小原生频率位移与原最大 gap 限制，仍 −7.3pp@128K；Gap-capped 同样几何受限仍 −15.97pp。诊断证明失败经由"完整长背景形成的保留 KV 状态"参与（O 0%/C 100%/P 100%/L 0%；VT 20%→60% 说明晚期竞争也有贡献），不是单纯相位差（31.25 rad 例 P 仍满分）。复发警告：**不要为解释单一分数临时指定一个几何量为唯一原因**（"不能将最大相位差的槽 28 自动称最敏感槽"）。
6. **Scale transport λ=0** [已验证]：守护（D_N/B_− 回放）把所有非零 λ 拦下；随后发现"F1 增量全部来自两行的措辞/词面差异"。警告：先查分数字符串学出的增量，再谈机制。
7. **MrUni/HighGapToLong/E2 端点干预全崩** [已验证]：全表÷4→32K 64.6；从高频借预算→32K−17.1/128K−10.8 且 36 行 0 提升；平台推到÷4.93→128K 54.7。这三条是 UNIFIED 端点约束仅存的直接实验支柱（其"定理"表述已被 GLM_6PRO_REVIEW 降为设计约束）。
8. **结论没有退出（qwen-panel 判词 + 本地复盘同名）** [已验证]：修订周期关闭后 unexecuted accepted 项滞留（A2/A7/A8/A9/A12/A13/A14/A15/Step-1）；pro 材料入库后无退出机制，靠 README 路由表人工裁定。
9. **危险带因果链断裂** [已验证为推翻]：75–112K 危险带宣称的三类证据（MK2@89K、VT@106K 及 FWE/MQ 距离）全部被证据距离复核削弱（见 §3#12/15）；mk_1@96K 与 vt_1@127K 是现成反例（地平线是风险因子非门）。
10. **旧 calibration 数据不可复用** [已验证]：每行仅 top/recent/target/support 有限 keys（221/32,123），保存原 LSE 不能重建未存 keys 改位置后的 logits——6Pro 校准验证必须全新采集。

## 5. 频率表/方法定义清单（名称、构造规则、32K/128K 得分）

官方 dev 面板（36 行、gain 1.13863、Qwen3B、S=4、W=32768、L=131072），除注明外数字来自 BUDGET/UNIFIED/GAP_CAPPED/BM 各文档回执 [已验证]：

| 名称 | 出处 | 构造规则 | 32K | 128K | 状态 |
|---|---|---|---:|---:|---|
| MrPro (MrRoPE-Pro) | pro/既有基线 | m_q=q(q+1)/[N(N+1)]，边界 j=23/40，N=17（m(x)≈x²；终端谱 kink gap 0.370=1.71× native 0.216，S=16 时 2.43×） | 87.22 | 78.13 | 基线 |
| s28_less (E1) | 本地 | MrPro 上 gap27→gap28 挪 0.045（m28 0.098→0.065） | 87.2 | **83.3** | dev 赢家；修复 89K multikey 绑定 |
| LBS | 本地 | 集中 gap35–38；m36–39=.625169/.729476/.846534/.979914 | 80.6 | 80.1 | dev 赢家；修 106K VT（0.2→1.0） |
| P2 | 本地 | 集中 gap29（洞 2.97×/另一记载 1.088 巨洞） | 72.9 | 81.7 | 短端崩；长端高 |
| pair(28+29) | 本地 | gap28 双倍集中 | 87.2 | 74.0 | 2.9–4.2K 洞超调 1.46× |
| Smooth(MrBudget) | 本地 | 挪中段、末 gap 削 0.042 | 87.2 | 68.3 | 后端抽预算→更不安全 |
| E3 gain074 | 本地 | 表=MrPro，gain 改 1.074 | 98.3 | 75.3 | 反证 gain 不救长端 |
| MrUni | pro | 全表线性压缩（÷4，同过渡区线性累计） | 64.6 | — | 崩（端点 I1 证据） |
| HighGapToLong | pro | 从高频间隔借预算给低频 | −17.1 | −10.8（差值） | 36 行 0 提升（否证 EVQ 直觉的跨端实现） |
| E2 | 本地 | 平台推到÷4.93 | — | 54.7 | 崩（端点 I2 证据） |
| E8 单槽置零 | 本地 | 某槽 ν=0 | — | 50.6（−13.9pp） | fixed-state 代理过度预言案例 |
| BM | MRPRO_BOUNDARY | a_i=6i(N+1−i)/[N(N+1)(N+2)]；m_q=q(q+1)(3N+2−2q)/[N(N+1)(N+2)]；连续 3x²−2x³ | dev 通过 | 70.83 | 采纳→128K 诊断负例（−7.3pp） |
| Gap-capped/收窄MrUni | 本地 | 闭式+独立 LP，限最大 gap | 84.44 | 62.15 | 0胜7负29平，NO_LONG_GAIN |
| Carrier removal | LOWFREQ_CARRIER | 槽 47–63 ν_j=(ω_j−c)/s，c 闭式（相干复背景）、cL0/s≤3π/32 | dev 对比完成 | 未测 | 队列停止，无采纳 |
| Scale transport | SCALE_TRANSPORT | ν_j=ω_j·S^{−β_j}，β_j 由 χ_ij 分位数 LS+保序回归，λ 网格 | — | — | 仅 λ=0（=MrPro 参照）REFERENCE_ONLY |
| N′ 族（0448/0449） | BUDGET §3 | m_q=q(q+1)/[N′(N′+1)]，N′=16/15（0448: /272，m28=0.110，m36–39=.669/.772/.882/1.0，D=131K，洞 1.76×；0449: m28=0.125，.758/.875/1.0/1.0，洞 1.80×） | 未测 | 未测 | 冻结队列 |
| 0446 StackFrontBack | BUDGET §3 | MrPro⊕s28_less(槽28,m=0.065)⊕LBS(槽36–39→D=127.5K) | 未测 | 未测 | 双机制叠加判据已冻结（128K≥83% 且 32K≥80%） |
| YaRN（faithful 参照） | opus/gpt56 | α/β ramp 29–30 onset、0.92–0.93 全插值段 | 见 RULER-13：0.2382@8K、0.0588@16K（binary 0.6772/0.5440） | — | 历史描述性 |
| CoPE/clipped 对照 | PRO_DECISION | 尾频+CoPE 检索 | 短 2/2；128K 8/8 | UUID 1/2、0/8 | 混合证据，未采纳 |

队列（seetacloud :27741）：0446→0448→0449（各 36 行+48 NLL，45–55 分钟/张）→0450 E1 新样本确认（multikey/multiquery/VT/QA×16）→0451 P2 长端确认；0441 P2_QA128_pilot4 收尾中；约 6–7 小时；基线复用历史 MrPro 回执零重算。公式 vs 数组误差 ≤4.3e-8（fp32）。

## 6. 用户指令与纠正（原文引用，出处带行号）

来自 `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md`：
- L196："你要牢记我的实验规则……MrPro已经有了baseline……目前BM在3B模型上输了，128K为什么有问题……他一定不是sota……一定有方法超过他"
- L328："我之前给你的两个文档全是pro模型的思路，你不用完全按照他们的走"
- L400："谁让你收窄研究的？……pro的不对，你就自己想"
- L627："为什么我们一直在你或者pro给一个看似牛逼的理论和实验，然后实验失败，目前仓库失败的报告，没有五十也有100了？"
- L680：实验原则——先和 MrPro 比零训练、单臂、省算力
- L690：三合一顺序（统一理论→候选→确认）
- L770："有没有阅读pro模型所有思考研究 我们所有失败范式……我要得到一个最终绝对能成功的方案。"
- L996："这是我让GPT6pro生成的一份详细的计划……hybrid_rope_iclr2027_theory_experiment_dossier_20260904.md"
- L1006："pro分析的我们之前没微调FFN可能是阿克琉斯之瞳"
其他书面纠正：
- `ROPE_MRPRO_BM_PROTOCOL_20260908.md`（作者纠正）："总体仍是最多10个有独立理论依据且经反向审查的候选……5分钟是预期成本，不是截断评测的时限"
- `ROPE_GAP_CAPPED_RESULT_20260908.md`："用户进一步明确不要进入'改善一个代理、破坏另一项、十分钟换一个猜想'的循环"
- `ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`："用户 11:06 明确告知 GPU 恢复并允许验证"（且明确**不**自动执行 GLM 原来的五个候选）
- `external-reviews/README.md`（治理）："External reviews are source material, not validated research owners. Their `VERIFIED`, `mandatory`, `decision` or `priority` labels do not grant project scientific authority or execution permission."

## 7. 未决问题

1. **冻结队列 0446/0448/0449/0450/0451 无回执**：双机制是否独立可叠加、N′ 族是否单调（洞代价项是否活跃）、s28_less/LBS 的 dev 收益能否外推——三者全败则 dev 收益不可外推的确认实验优先级上升（BUDGET §3 判据）。
2. **姐夫稿/6Pro 原文未归档**：repo 只有 §8 摘要与复核；如用户可提供原文，需补一次全文比对（当前 §3#5–9 全部依赖二手记录）[部分证据]。
3. **6Pro 校准-KL 完整行验证进行中无结果**：pro_block_calibration 采集（前 6 篇 32K 文档、4 拟合 2 保留、每篇每层 4 head 轮换 16 head）；4K 块尺度是**声明的假设**；虚拟位置跨度 118,784，真实 128K 文本未测；固定态 KL 改善不得替代完整模型结果。
4. **opus 三臂判别预测从未执行**：same_support_geometric_s4 vs nearest_yarn_ramp_s4 vs converged_budgeted_s4——这是"band-split≈YaRN ramp"假设的唯一能杀它的实验，未跑即两者都未否证。
5. **±v 镜像对照与三条件 position/interference 因子实验未执行**（§3#9）。
6. **危险带的合格检验未做**：固定内容/绑定结构、配对改距离、逐子问题计分的距离因果实验（GLM_6PRO_REVIEW 尾段要求）。
7. **centroid 与"右移"叙事的最终处置**：GLM_6PRO_REVIEW 已用质心表推翻单向梯度宣称，但 UNIFIED §3 文档正文尚未回改（同日两份文档并存，以复核版为准，需正式勘误）。
8. **FFN/微调"阿克琉斯之瞳"假设未裁决**：与零训练主线冲突，若验证需要训练资源，从未立项（用户 L1006 原话仍是悬置问题）。
9. **carrier removal 的 128K 成绩**：dev 完成后队列停止——不得写成否证（未测≠无效），亦无采纳证据。
10. **qwen-panel 未执行 accepted 修订项**（A2/A7/A8/A9/A12/A13/A14/A15/Step-1）在下个周期是否仍有效；ICLR abstract 9/18 临近，BUDGET §5 论文口径与 GLM_6PRO_REVIEW 推翻条目的措辞须同步。
11. **DEFECT-5（Qasper binary +0.0654 p=0.002）无 owner**：该效应属于谁（方法 or 数据缺陷）仍未认领。

## 附：逐材料"尚未消化的可用点"速记

- gpt56：实验 0 指标混战决策表（与 36 行面板互补的度量合法性检查）；退路定理（若几何线全败的论文形态）；Direction A 的"使用条件"提法可迁移为 QA/VT 分任务表选择的风险声明语言。
- opus：θ' 等价式可直接进论文相关工作一节；band-split 旋转数计数法是可复用的描述工具；三臂判别实验（§7#4）是零训练可做的最便宜裁决。
- qwen-panel：主题 A★–F 路线图与统计功效口径（n≈627、McNemar）仍是审稿应对的现成弹药。
- pro-materials 九份：dossier 的实验矩阵多数格从未跑（1,734 行只被 synthesis 裁决了主张层）；Round12 的 YaRN limits 代数、TWO_DIRECTION 审计的双方向分解在 0910 理论整合中只吸收了标题层。
- BM 系：诊断脚本 diagnose.py 的 O/C/P/L 协议是现成的"保留 KV 状态 vs 位置编号"分离器，可复用于 s28_less/LBS 的负例。
- CROSS_AUDIT：κ(p) KL 判定边界、秩界 eq(11)、γ=1/2 守护在分析层未被使用过（只有统计口径被吸收）。
- 6Pro：EVQ-Cosh 的间隔预算语言已并入 BUDGET 模型；其"关系函数及尺度需求不由配方自动给出"的自限提醒仍有效。
