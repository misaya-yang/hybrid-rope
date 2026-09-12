# unify_20260910 —— 文件权威索引（INDEX）

2026-09-10 重建（取代此前短索引；本文件只作地图，一切数字以其行内引用出处为准）。

## 阅读顺序与状态基线

1. 阅读顺序（`INTEGRATION_20260910.md` 卷首"阅读优先级"）：**INTEGRATION_20260910.md → NEXT_DERIVATION_KKT_PROBLEM.md → STARTING_POINT_YARN_VS_MRPRO.md → answers/（D1–D4、T2、T3）→ digests/**；冲突裁决：本文 < 单条 digest 之上是 INTEGRATION，一手报告＋当场重算终裁。
2. 队列基线（任务书 0910 GROUND facts；`V4_FINAL_VERDICTS.md` §7 同口径不扩述）：0441 在跑；0446→0448→0449→0450→0451 **未排队**，需用户显式重授权（与 NONGEOMETRIC:9-31「no-new-32K」协议冲突未解除）。
3. 时效提示〔刷新 2026-09-10〕：`checks/V4_FINAL_VERDICTS.md` 是 **11:08 快照**（当时仅 tables 席官方在册）；其后三席重跑报告已全部落盘——`answers/V_predictions.md`（11:20，110 行）、`answers/V_veto.md`（11:30，124 行）、`answers/V_math.md`（11:50，165 行）——**V×4 四席官方裁决全部在册**，V4 台账 §4/§5「未产出/[重建-非官方]」口径已被取代（终裁以四份 `V_*.md` 为准，V4 只作历史台账＋[已复核] 复算层）。UNCONFIRMED 九项对账见 `INTEGRATION_20260910.md` FLAG-11 更新行。

## 根级文档

- `INTEGRATION_20260910.md` —— 两线 62 代理资产总整合（本方 32 代理＋codex thread 01a0806f 30 代理）。§1 四子问题裁决、§3 地面事实、§5 死亡机制登记册、§7 冲突裁决表（FLAG-④＝0450↔080 队列号互指；FLAG-10＝D2/D4 对 N′ 倾向相反，0448/0449 为预注册对赌判决点）、§9 交付三件套诚实状态表。
- `NEXT_DERIVATION_KKT_PROBLEM.md` —— 下一次推导（wf3）唯一入口：RoPE 频率分配＝非均匀傅里叶采样 KKT 问题的最终模板、K1–K6 推导任务、§5 红线 1–6。
- `STARTING_POINT_YARN_VS_MRPRO.md` —— YaRN vs MrPro 算子级已验证对比 F1–F9（三方交叉：用户附件核查＋`digests/digest_mrrope-evq.md` §D8〔Qwen S=4 上 HF-YaRN vs MrPro max|Δm|=0.034〕＋codex 30 代理归档）；F8 与 root-veto 红线 1 互证，"YaRN 递减 vs MrPro 递增"禁用。
- `verify_math.md` / `verify_feasibility.md` / `verify_evidence.md` —— workflow-1 提案 A/B 三席对抗验证（载体＝docs/research 的 BUDGET＋UNIFIED，A/B 正文未落盘）。数学席：代数内核全过、缺陷集中在"恒等式→经验"映射层（`verify_math.md` §0 结论速览）；三席裁决对应的修订已随主循环 commit 7702fb8 落地（任务书）。
- `timeline_thread-core.md` —— codex 核心会话（thread 01a07996，`raw/raw_thread-core.txt` 5303 行/468 消息）分块时间线笔记；`read_thread-core_clean.txt` ＝其去 boilerplate 可读源。

## answers/ —— workflow-2 四问推导（D1–D4）＋验证（T2/T3）＋ V×4 终审席

- `D1_high_freq_redundancy.md` —— 问①：弧覆盖冗余 100% 但兑现价值 0；唯一自由预算＝MrPro 自压的 Σm(24..28)=0.2288 lnS（已兑现 0.0326）。〔更新 2026-09-10：D1:22 "÷4.93" 残留旧值**已更正**为 4^m40=4.9638（`V4_FINAL_VERDICTS.md` §1-7 传导完成，另 D1:12 缠绕数底数笔误随 V_math §4-11 修复）〕。
- `D2_mid_freq.md` —— 问②：中频带 24–39＝三重合；双端风险泛函 R=U+αH+βΦ，可行窗 ρ0∈[1.370,1.386]、p∈{0.5,0.55}（`T2_validation.md:83`，转引 `V4_FINAL_VERDICTS.md` §2 c1），发表点 (1.372, 0.5, 3.920, 1.234)（任务书 GROUND/T2）；留出 τ=+0.576、幅度不外推。复算链脚本：`D2_mid_freq.py/.out`、`D2_fit2–4.py`（否决路径）、`D2_fit5.py/.out`（发表拟合）。
- `D3_low_freq.md` —— 问③：低频要的是"完成到 m=1 的资格"非预算（尾部 log-gap 增量实测 −6.5e−9）；E2 过冲 −9.7pp、E8 单槽删频 −13.9pp（12 行口径）。
- `D4_transport_rule.md` —— 问④：盲目标＋argmin 作废为错设证据（盲解=MrUni 64.58）；右载 ramp 方向被导出；推荐 N′=16（0448）/次选 N′=15（0449），§5 交付 N′=16 全表，§6.3 决策分支（ramp16⊕notch）。过程脚本在 `d4_scripts/`。〔更新 2026-09-10：§4 行标签"mid（g29–36/37）"**已改写为"mid（g29–35）"**并带更正注——V4 以 g29–35 带逐位复现发表值、文档自洽，争议系首发 math 席带选取 off-by-one（`V4_FINAL_VERDICTS.md` §2 末段、§3-②；传导完成）〕。V_math §2/§4 另有 7 处末位/标签 slip 更正，已随 V-blocking 修复吸收（其 §2 盲 LP 全局界、字面退化合句判死见 `answers/V_math.md`）。
- `T2_validation.md` ＋ `t2_work/T2_validate.py/.out` —— T1 候选表恒等式自检/覆盖翻转/面板回归/预注册判决；R 矩阵发表段 :115-118：Stack 47.65 < MrPro 48.60 < N16 52.64 < N15 53.04 < N16notch 54.32（V4 §2 9/9 逐位复现；notch 新洞 ρ28=1.3882>线、s28 底座洞 1.3710 恰在线上）；死刑判据 K1–K9 见本文 (d) 节与 `INTEGRATION_20260910.md` FLAG §7（任务书）；重跑输出与 .out diff 一致（`V_predictions.md` §0-1）。
- `T3_unification.md` —— 统一泛函 J=A+H+P 与四方法（EVQ/YaRN/NTK/MrRoPE）极限定理；J 自限为"假设组织器"。〔更新 2026-09-10：四定理推导步、EVQ→cosh 极限运算、ε_j 和=1↔守恒换算**已经 math 席独立复核**——V_math §5：Thm1 全链（waterbed、h↔ρ、O(τ²) 收缩、μ_F 局域化）[已验证/边界条件诚实自标]；Thm2a [已验证，记号撞车警示（斜率 b 与底数 b=10⁶）]；Thm3 [已验证＋跨文档 YaRN 张量冲突注]；Thm4 [部分证据，随 D4 表述修正]；无阻断性数学失败〕。
- `V_tables.md` —— V×4 tables 席官方裁决（首发唯一在册；其后三席重跑补齐，见 V4 时效提示）：8 项 claim/verdict＋2 项 blocking（① 0446 同号异表＝已撤回臂 `deferred_queue/20260909 批 0446_HighGapToMid.json`；② 0446/0448/0449 本地无契约）。逐项被 V4 独立复算（§1）。
- `V_predictions.md` —— V×4 predictions 席报告（11:20 落盘，晚于 V4 快照，补上其 §4 缺失的 (d)(e) 两节）。核心判定：行级"修复/受损"集由已降级的暴露计数代理驱动（mk_1@96K MrPro 1.0 vs LBS 0.0，同暴露集反例）；附**六项"跑队列前必须修复清单"**（K6 命中条件与 0449 预测反向、K1 阈值过松、§6.3 分支重叠等）。
- `V_veto.md` —— V×4 veto 席报告（11:30 落盘；V4 §5"实质零产出"口径已被取代）。逐条核 D1–D4/T2/T3 的"由 X 推 Y"：总裁定 **3 处 contradicted、4 处 overstated、其余主体合规**；触线点含 D3 §3.3 把覆盖矩阵升格为行级因果。
- `V_math.md` —— V×4 math 席官方报告（重跑席落盘 11:50，165 行；过程件 `v_math_work/V_part{1,2,3}*.py` 可重跑）。三焦点结论：① 守恒两层口径钉死——望远镜式 ln4·(m40−m23) 无条件 30/30、字面 Σ=ln4 仅 24 张端点固定＋6 例外（Native/NTK/E2/HGL/HGM/P2），GR §4-1"17 张"[被否定]；② D4 盲 LP z*=1/14 [已验证-限定定义]、字面 F_arc 退化合句 [被否定-记号滑移]、BL=1.071429 等 7 处末位/标签 slip 入错误目录；③ T3 四定理复核结论见其行内条。其 §4 更正目录已全部传导（D1–D4/T2/T3/CANDIDATE/GR，主循环 2026-09-10）。

## tables/ —— G1 地面真值与 T1 候选表

- `ground_truth_tables.json`（G1）—— 38 条目 × {nu_j, m_j, T_j, D_j, r_j, gap_j, 水床和, 洞比率, 分数, sha256, 对账}（`GROUND_README.md:6-7`）；18/18 公式重建与部署张量逐位一致；126 锚点=121 MATCH/2 NOTE/3 MISMATCH。〔更新 2026-09-10〕 字段 `endpoint_delta_m.fast_band_bitwise_equal_native` 在**本 JSON 内**恒 False＝旧生成器 shape 死旗标（24 切片 vs 64 全数组），不得作表体正确旁证；生成器该式**已在脚本修复**（`f32(nu[:24])` vs `f32(NATIVE[:24])`）但**未重生成**——修复＋sources 行号更正＋HGM 构造回填共同登记为 G1 重生成批次（重哈希须对账 glue，待授权，INTEGRATION §8-7）。真实 fast 段例外四臂={HighGapToLong, HighGapToMid, FullLagP2_Transfer3B, NTK_static}（`V4_FINAL_VERDICTS.md` §3-①，含现算；GR §4-5 已按 26/30 全量重算改写）。
- `rebuild_ground_truth_tables.py` —— G1 生成器（会写仓库文件，审阅场合勿跑；"重跑字节全等"已经 V_tables 独立复算 [已验证]（V_tables:9），**但 2026-09-10 脚本已做三处更正性编辑**（:111 死旗标修复、sources 行号 45–47→48–50、max 洞声称历史化），当前脚本重跑 ≠ committed JSON——属预期，重生成与重哈希并入 INTEGRATION §8-7 G1 批次，未授权前勿跑）。
- `GROUND_README.md` —— G1 说明书＋勘误：§6＝BUDGET §3 max 洞更正与 Stack argmax @g35→@g38 位置笔误（已落实）；§4-5 fast 段例外清单已改 26/30 全量口径（补 HGL/HGM、删 YaRN，V4 §3-① 落实）；§4-1 守恒已按 V_math §1.1 重写（望远镜 30/30＋24/30＋6 例外）；§5-9 HGM 升入复原清单（V_tables §5-3 低报修复）；§6 附 0446 同号异表治理警示（Blocking-①）。
- `CANDIDATE_TABLES.md/.json/.csv`（T1）—— 13 表 × 64 槽全表（D4 产出族＋对照），§5.2 29 checks ALL MATCH、G1 errata m28:=m27=0.06535945（Stack）（任务书 GROUND；更强旁证＝`checks/v_tables_check.py` 261/0，V4 §7）；§2/§5 含预注册判据（其缺陷修复清单见 `V_predictions.md` 附录）。
- `generate_candidate_tables.py` —— T1 生成器（幂等）。

## checks/

- `V4_FINAL_VERDICTS.md` —— wf2 终审 V×4 席位裁决回收＋独立复算台账（[已复核]/[未复核] 两级）：§0 官方状态总览（在册维度＝math/veto/predictions/tables，非任务书所述四席名）、§1 tables 席逐项复算、§2/§4/§5 三席转录重建、§3 两项新发现（死旗标、D4 带界裁决）、§6 UNCONFIRMED 清单、§7 GROUND 闭合（V×4 三条 mid-stream 证据全部升格已复核）。
- `v_tables_check.py` —— V_tables 席独立验证器（不复用 G1 生成器代码路径）；V4 重跑 exit=0、261 OK / 0 FAIL（`V4_FINAL_VERDICTS.md` §1）。
- `feas_mj_tables_check.py` —— workflow-1 可行性席的 CPU 检查（按 BUDGET §3 公式重造 0446/0448 的 m 表对账）；**非 V×4 产物**（10:10 早于 10:43 批次，V4 §0 时间戳对照）。
- `g1_ground_truth_acceptance_20260910.md` —— G1 v1 独立验收记录（09:2x，完全独立重算核 PASS 项表）。
- `max_new_tokens_reconciliation_20260910.md` —— astra10 线索结案：OLMo LongBench 自然文本顶层 max_new_tokens=16 为 HF config 回显、实际 cap=64（两臂对称 45/778 行触顶）；登记为口径事实，不作废结果。

## digests/（本方线 16 份，人读提取件）

- 线程转录 7：`digest_thread-core/-main/-0907/-0908-night/-0909-am/-0909-pm/-0910-batch.md` —— codex 各会话 user/assistant 全文通读提取（源＝根级与 `raw/` 的 raw_* 文件；映射表见 `digest_thread-0908-night.md`、`digest_thread-0909-am.md`）。
- 材料/结果 9：`digest_mrrope-evq.md`（MrRoPE 论文精读＋EVQ 对表）、`digest_panel-results.md`（36 行面板地面真值 A1–A17，面板数字权威）、`digest_failure-records.md`（可机检否决清单 V-A1..F12）、`digest_theory-core.md`（docs/theory 四文件）、`digest_theory-0910.md`（0910 新增 8 份理论文档）、`digest_evq-code.md`、`digest_nongeo-code.md`、`digest_pro-materials.md`（外部评审采纳/否决对照）、`digest_paper-state.md`（论文并入路径）。

## digests_codex/（codex 线 7 份）

- `.agents/rope_unification_20260910/reports/`（sol20/astra10 等 30 代理归档）的全量摘要：`digest_astra-evq-finite.md`、`digest_astra-margin-lineage.md`、`digest_calibration.md`、`digest_constructive.md`、`digest_failure-audits-1/2.md`、`digest_transport-operator.md`；仓库权威副本 `docs/research/rope_allocation_20260910/`（codex commit 525dc15，`INTEGRATION_20260910.md` 卷首）。

## 源转录提取（只读素材层）

- `raw/`（28 件）—— 0910 批量提取作业区：`batch_t07-40-42…batch_t07-52-16.txt` 16 份（0910 07:40–07:52 本地批量 16 会话逐份提取）＋ `batch_all/batch_clean.txt` 合并版；`raw_s1–s7.txt`、`raw_thread-core.txt` 的根级同批副本；`rollout-0909-2144.jsonl`、`rollout-0910-0159.jsonl` ＝ codex thread 01a0806f 两份原始 rollout。
- 根级 `raw_thread-core.txt` / `raw_thread-main.txt` / `raw_thread-0909-pm.txt`（各带 `_tools.txt`）—— thread 01a0806f 核心/主线程/0909-PM 两子线程转录（消息与工具调用分文件）。
- 根级 `raw_s1-005047.txt … raw_s7-073814.txt` —— 0910 凌晨 7 子会话提取（digest_thread-0909-am 的源）。
- 根级 `raw_night01–08.txt`（＋`_tools.txt`）与同批命名提取 `raw_224930_argument_reader_check.txt`、`raw_2314xx_asset_*.txt` 5 份（foundations/scale_architecture/llama_adaptation/frozen_models/mechanisms）、`raw_234623_second_manuscript_review.txt`、`raw_235944_pdf_review_r01.txt` —— 09-08 夜间（本地 22:49–23:59）8 会话双命名套提取，会话↔文件映射以 `digest_thread-0908-night.md` 表为准。
- 根级 `raw_0819.txt`（0819 rollout 无消息注记）、`raw_0907-monitor.txt`、`raw_0908a/b/c.txt` —— 0819/0907/0908 更早会话提取。

## Agent 工作目录（过程件，只读）

- `d1_work/`（D1 复算脚本 d1_analysis.py、d1_final.py）、`answers/d4_scripts/`（an*.py、dom*.py、fam.py、tables.py 9 件）、`answers/t2_work/`（T2_validate.py/.out）、`answers/v_math_work/`（V×4 math 席过程件，官方报告 `V_math.md` 已落盘）。
