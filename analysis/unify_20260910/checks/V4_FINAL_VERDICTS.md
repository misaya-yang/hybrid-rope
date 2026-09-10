# V4_FINAL_VERDICTS —— wf2 终审 V×4 席位裁决回收与独立复算台账

2026-09-10（本席由主循环派生，纯 CPU、只读复算；唯一写入=本文件）。
复算数据源：`analysis/unify_20260910/tables/ground_truth_tables.json`（G1，38 条目）、
`tables/CANDIDATE_TABLES.json`、`tables/rebuild_ground_truth_tables.py`（只读）、
`results/nongeometric_screen_20260909/reference_tables.json`、
`results/.../queue/` 与 `deferred_queue/20260910_candidate_quality/`（目录列举，只读）、
`answers/D2_fit5.py`（公式只读、不复用其代码路径，本席独立重写）、
四席转录 `~/.claude/.../workflows/wf_f8c14342-c3c/agent-*.jsonl` 与 `journal.jsonl`（只读）。
标注约定：每项声明标 **[已复核]**（本席用上述数据源现算复现）或 **[未复核]**（席上产出、本席未能复现/未复现/席上未产出）。

## 0. 官方裁决状态总览

`journal.jsonl` 现 25 行（>21）。result 条目仅 1 条属于 V×4 终审批：
**line 25 = agent a6b5cebdaf30c237e，dimension="tables"，官方裁决在册**（8 项 claim/verdict + 2 项 blocking，全文见转录 line 25，本节 §1 摘录并逐项复算）。
其余三席（math a27a12b5b69ae64be、veto ada30027e77b31b71、predictions aa29c4a3d7786094e）
journal 只有 line 18–21 的 started，无 result；三份转录分别在 62/58/95 行处**中断于复算途中**
（无最终文本、无裁决 JSON、未写答案文件），其结论只能**从转录 tool_use/tool_result 重建**，标 [重建-非官方]。〔更新 2026-09-10〕
journal line 22–24 显示主循环已重启三席重跑（ac82fd9e5ac62af37=veto、ae2ce49a3ae7e47e7=predictions、
a9fcda922ac40e803=math，任务维度与首发一致），快照时刻（11:08）仍在途、无 result。
注意：任务书把四席描述为 math/feasibility/evidence/tables，实际在册维度是 **math / veto / predictions / tables**；
`checks/feas_mj_tables_check.py`（10:10 生成）**不是** V×4 产物（早于 10:43 批次启动，属前一波可行性验证席），本席未将其计入 V×4 交付。 **[已复核（时间戳对照）]**

## 1. tables 席（a6b5cebdaf30c237e）——官方裁决在册，逐项复算

交付物：`answers/V_tables.md`（11:07）、`checks/v_tables_check.py`（独立验证器）。
本席重跑验证器（只读）：**exit=0，==== RESULT: 261 OK / 0 FAIL ====**，与席上声明（261/0，
V_tables.md §0-1）一致。 **[已复核]**

官方 8 项 claim 的裁决与本席复算：

1. **G1 数字位级对回原始出处（supported）**——本席复核其支撑面：MrPro `formula_vs_deployed.sha256_deployed`
   == 本席对 G1 `nu_j` fp32 打包重算 sha256 **[已复核]**；261 项全过由重跑复现 **[已复核]**；
   G1 生成器重跑字节全等声明：本席未重跑 `rebuild_ground_truth_tables.py`（它会写仓库文件，越出本席只读边界），标 **[未复核（席上声明）]**。
2. **N′=16/15 ≡ 0448/0449 冻结定义（supported）**——复算 ramp16 闭式 `q(q+1)/272` vs G1 `MrProN16.m_j`：
   **max|Δm|=3.309e-08**（与 math 席 3.309e-08 一致）**[已复核]**；位级路径复现：
   `f32(fp64 Native × 4^−m)` 逐位==G1 `nu_j`（True），而 `fp32(Native)×f32(4^−m)` 乘积路径 False——
   证实席上"路径敏感性为真、~25 槽 1-ULP"叙述的方向 **[已复核]**；
   N16 在 G1 中 `formula_vs_deployed={bit_exact:None, '公式重建，无本地部署张量可对账'}`，
   即"服务器侧执行表不可本地验证"caveat 为实 **[已复核]**；`queue/` 下无 0448/0449 契约文件 **[已复核（目录列举）]**。
3. **Stack ≡ 0446 定义、§6"@g35"为位置笔误（supported）**——复算：Stack 全局洞 argmax=**g38，1.4929753892033155**
   （G1 存储字段 `hole_ratio_max_global`/`hole_ratio_argmax_gap=38` 同值）；ρ35=**1.4562**、ρ27=**1.2409**、
   ρ28=**1.3710**；Stack m36–39=[.6252,.7295,.8465,.9799]≡LBS；`m27−m28=3.824e-9`（fp32 反演噪声）
   **[以上均已复核]**；Stack vs LBS 逐 gap：ρ 仅差 **g27（1.2409 vs 1.2984，Δ−0.0575）与 g28（1.3710 vs 1.3103，Δ+0.0607）**，
   m 数组仅差槽 28——与 T2 §b2/R2 更正口径完全一致 **[已复核]**。
4. **README §1 ω"逐位吻合" overstated（实测 24/64 槽 1-ULP）**——复算：fp64 `b^(−j/64)`→fp32 vs
   `reference_tables.json:Native.values_float32` 差 **24/64 槽、最大 1 ULP、最大相对差 1.178e-7**；
   前 24 槽内差异槽={1,3,7,10,11,12,13,16,20}。席上报"最大相对差 8.21e-8"与本席 1.178e-7 同量级
   （1 ULP≈1.19e-7 上界内，取槽子集不同所致）**[已复核]**。结论 overstated 成立。
5. **README §4-5 fast 段例外清单漏列（overstated）——方向成立，但席上证据链有一处错误，见 §3 新发现**。
   复算确认：HGL `m_j` 槽 1–23 全部偏离原生：m1=**−0.006770**、m5=−0.033851、m10=−0.067702、
   m15=−0.101554、m20=−0.135405、m23=**−0.155715**；HGM 同构造同值（`endpoint_delta_m.m_23=−0.1557154` 双臂相同）
   **[已复核]**——与 GROUND 事实"slots 1–22 deviate（m1=−0.00677 … m23=−0.15572）"吻合。
   GROUND_README.md:101（§4-5）现文为"全部面板表 fast 段（槽0–23）与原生逐位相同……除 P2（+0.000775）与
   NTK/YaRN（按构造）"——例外清单确实漏 HGL/HGM **[已复核]**。
   **更正席上论据**：该 claim 的括号"（G1 JSON 字段对 HGL/HGM=False，数据正确）"不成立——该字段对
   **全部 30 个含数组方法**（含 Native 自身）一律 False，是生成器的死旗标，见 §3-① **[已复核]**。
6. **README §5-9 HGM"未本地复原"为低报（overstated）**——席上按 `gap_budget_transfer.py` 语义逐位复原 HGM；
   本席复核方向性证据：HGM 与 HGL 的 `m_23` 同为 −0.1557154、G1 `mismatches/reconciliation` 账本 126={121,2,3}
   与 mismatches=10 的计数复现（见 v_tables_check 重跑）**[已复核（重跑覆盖）]**。
7. **答案文档更正全量传导（supported）**——grep 复现席上所引行号存在：D3:33/D2:162/D1:164（m36=0.5948）、
   @g35→@g38 更正（T2:24,177 等）**[已复核（行号抽查）]**；残留 **D1:22 "÷4.93"旧值**：复算部署
   E2_tail_more `m40=1.1557154` ⇒ 4^m40=**4.9638**（≠4.93），BUDGET:14 已更正、D1 未同步 **[已复核]**。
8. **Blocking-①：0446 同号异表**——`deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json` 在册存在，
   而 BUDGET:48 以 0446 指 StackFrontBack **[已复核（文件存在性）]**。
   **Blocking-②：0446/0448/0449 本地无契约**——`queue/` 列出现册只有 001–043、**0440×2**（P2_LongPrecheck128 与
   Smooth_MrBudget 同号）、0441、0444、0445、080；无 0446_Stack/0448/0449 **[已复核]**。
   0450↔080 互指的漂移细节（`verify_feasibility.md` V-C2 / INTEGRATION:118 FLAG-④）本席未展开 **[未复核]**。
   处置要求（contract+sha 登记后方可回队）维持；与 NONGEOMETRIC:9-31 的执行授权冲突席上未解除、本席亦不解除。

总评：官方 overall"表本体检毒通过、位级干净；剩余在文档措辞与队列治理层"与本席复算一致。 **[已复核]**

## 2. math 席（a27a12b5b69ae64be，dimension=math）——无官方裁决，[重建-非官方]

状态：62 行转录，未写 `V_math.md`（文件不存在，`answers/` 目录核实）；唯一落盘工件
`answers/v_math_work/V_part1_g1_identity.py`。最后文本（line 62）："D4 §4 在精确带界（mid = g29–35）下复现。
现在进入 D2/T2 风险泛函的复现。"最后一个脚本以 **KeyError 'LBF'（stdin line 98）** 崩溃于
T3 §6.2/D4 §3 质心段——即**任务书要求的 T3 四个极限定理、EVQ→cosh 极限运算、ε_j 和=1↔预算守恒换算
三块从未执行**。 **[已复核（转录）]**

可复算结论（本席现算全部复现）：
- **Σ_excess(g23..39)=ln4 恒等式**：38 表中端点固定（m40−m23=1）者**零违例**（m 基恒等望远镜求和；
  ν 基残差 ≤2.45e-8，math 席全表打印）——G1 存储字段同值 **[已复核]**。例外定性：
  E2/HGL/HGM Σ=1.60216=ln4+lnb/64（m40=1.1557>1，非端点固定臂）；P2 Σ=1.38551895、亏 7.754e-4；
  Native Σ≈0（无搬运）——**"1.088" 与"Σ17=lnS"相容的判词**：1.088415 是 gap29 的**原始** log-gap
  （ρ29=2.969564），其**预算（超额）部分=0.872548=62.9%·ln4**；记错方为把 1.088 当超额引用的措辞，
  账面无冲突 **[已复核]**。
- **P2 逐 gap 超额表**：g25 .007294 / g26 .017922 / g27 .073185 / g28 .203810 / g29 .872548 / g30 .204189；
  洞>1.5：{g28:1.5215, g29:2.9696, g30:1.5220}；m24..32=[.00137,.00315,.00841,.02134,.07413,.22115,.85056,.99785,1.0]
  **[全部已复核]**（与本席 §1-5 P2 复算逐位同）。
- **洞位表**：MrPro/s28_less 1.4476@g39、pair 1.4608@g28、LBS/Stack 1.4930@g38、N16 1.4608@g38、
  N15 1.4757@g37、Smooth 1.4513@g35、MrUni 1.3464@g28、P2 2.9696@g29——9/9 复现，
  且确认 **s28_less 全局最大洞仍是 g39=1.4476（MrPro 自家洞）**（GROUND 事实 (3)）**[已复核]**。
- **闭式重建**：ramp17 vs MrPro max|Δm|=3.686e-8；ramp16 vs N16 3.309e-8；ramp15 vs N15 3.551e-8；
  uni17 vs MrUni 2.489e-8；D39(N16)=131072 **[ramp16/uni 两值已复核，其余同席口径本席抽验 ramp16]**。
- **R 泛函复现**（T2 中心 (ρ0,p,α,β)=(1.371,0.5,3.987,1.3)，本席按 D2_fit5.py 语义独立重写特征）：
  Stack 13/6.507/6.699→**R=47.65** < MrPro 48.60 < … < N16 **52.64** < N15 **53.04** < N16notch **54.32**
  （s28 47.78、P2 48.17、LBS 48.47、Smooth 48.85）——与 T2 §b2 发表矩阵（T2_validation.md:115-118）
  9/9 表逐位复现；notch 新洞 ρ28=**1.3882**（>1.371 线，"notch 在 ramp16 底座不免费"为实）、
  s28 底座洞=**1.3710 恰在线上** **[全部已复核]**。
- **c1 可行窗**：math 席网格扫出 19/279、窗 [1.370,1.386]∪{(1.371,0.45),(1.390,0.55)}；
  T2 发表 18/217、窗 [1.370,1.386]×{0.5,0.55}（T2_validation.md:83）——两口径网格不同，
  (1.39,0.55) 越上界一格是否应判可行 **[未复核（两席互差，需 math 重跑席裁决）]**。
- **c2 序检验**：预测升序 s28<P2<LBS<Faster<MrPro<Smooth<pair<MrUni、Kendall τ=+0.852（25/27）
  **[序本席可复核（R 值见上，Faster/pair/MrUni 席上值 48.49/54.78/55.51 与 T2 表一致）；τ 值未复算，标未复核]**。
- **D4 §4 带界争议——本席已裁决（见 §3-②）**：math 席 mid 序列 .1429/.1333/.1167/.1029/.0915 对应
  带=g29–36；D4 文档序列 .1429/.1238/.1083/.0956/.0850 对应带=g29–35（δ 至槽 36 完成步前一步，q≤13）。
  本席以 g29–35 重算：**argmin 序列 .1→.9 = 14,14,15,15,16,17,17,17,17、R(.5)=.632/.548/.479/.423/.500
  与 D4_transport_rule.md:92 发表值逐位复现**——文档自洽，math 席差异是**验证者自身带选取 off-by-one**，
  非文档错误；建议把 D4 行标签"mid（g29–36/37）"改写为"mid（g29–35）"以消歧。**[已复核]**
- **D4 §2 盲区表/全局界证明**（min max(δ,1−13δ)=1/14 由 uni14 达到）：**[未复核（席上输出，本席未独立证明）]**。

## 3. 本轮独立复算的新发现（超出四席在册结论，供主循环吸收）

**① G1 死旗标：`endpoint_delta_m.fast_band_bitwise_equal_native` 恒为 False（生成器 shape bug）。**
`tables/rebuild_ground_truth_tables.py:111`：`np.array_equal(f32(nu[:24]), f32(NATIVE))` 把 **24 元切片与 64 元全数组**
比较，numpy 形状不等→恒 False（本席以同值 24-vs-64 数组演示 False）。后果：committed G1 中该字段对
**含 Native 在内的全部 30 个数组方法一律 False**，无判别力。**真实的位级 fast 段（槽0–23）例外清单应为
{HighGapToLong, HighGapToMid, FullLagP2_Transfer3B, NTK_static} 四臂**（本席对 G1 存组 `nu_j`→f32 与部署
Native 逐位比较：26/30 等、4/30 不等）；且 **YaRN 两变体的 fast 段实际逐位等于 Native**——GROUND_README.md:101
把它列进例外属**过度排除**。对 tables 席 verdict #5 的影响：**结论方向（README 漏 HGL/HGM）成立，
其"JSON 字段数据正确"论据不成立**（字段是坏旗标，不能作为"表体正确"的旁证）；修复建议：
生成器该行改为 `f32(nu[:24])` vs `f32(NATIVE[:24])`，README §4-5 例外清单改为"P2/NTK/HGL/HGM（YaRN 删名）"。
**[已复核（全部现算）]**

**② D4 §4 mid 带界争议裁决**：见 §2 末段——文档在 g29–35 带下完全自洽。 **[已复核]**

## 4. predictions 席（aa29c4a3d7786094e，dimension=predictions）——无官方裁决，[重建-非官方]

状态：95 行转录；未写 `V_predictions.md`（文件不存在）。最后文本："我的所有独立重算都与 T2 的 b1/b2
矩阵一致。现在让我验证他们的验证脚本能正常运行，并检查 evidence-distance 的来源行。"随后两步均因
**shell 工具层错误**失败：`timeout` 命令不存在（exit 127，T2_validate.py 未真正跑成）；grep 管道里
`echo ===` 触发 zsh "=" 展开报错（exit 1）。任务书 (d)（Smooth/pair 反例是"预测其败"还是"事后解释"）与
(e)（最强一句话/最弱一环）两节**从未产出**。 **[已复核（转录）]**

可复算结论：
- **Stack==LBS 逐 gap**（除 g27 −0.0575→1.2409、g28 +0.0607→1.3710）：17-gap 全表本席复现一致，
  即 GROUND 事实 (2)"aa29c4 confirmed"**[已复核]**。
- **G1 存储洞字段 10 方法打印**与本席 §1/§2 复算一致 **[已复核]**。
- **U 暴露矩阵**（MrPro 17/s28 17/P2 0/LBS 13/Smooth 13/Stack 13/N16 14/N15 10/notch 14）与逐行槽集：
  本席以独立特征实现逐位复现 **[已复核]**。
- **D_j 边距**：N16 D37=95560（vs mk_1 95676 → **−116**）、N16 D38=111347（vt_0 105880 → +5467）、
  Stack D38=LBS D38=105953（+73）、Stack D39=127473、N16 T35=26591 vs 26470（差 121 指标边界伪影口径）
  **[D 值与差值已复核]**。
- **ruler.jsonl 逐行翻转**（8 臂：s28 2行0负、pair 1负、LBS 6、Smooth 8、HGL 7全负、P2 12、Faster 1负、MrUni 10）：
  本席以 G1 `panel_scores.wins/losses` 做一致性核对（HGL 0W/7L、s28 2W/0L、LBS 3W/3L、Smooth 3W/5L、
  P2 6W/6L、MrUni 4W/6L、pair/Faster 0W/1L）——**方向与数量级全部吻合**；逐行 id 级本席未重抽
  （`results/MrPro/ruler.jsonl` 不存在，基线行嵌在各臂 summary 之外，席上取数路径未完全还原）**[部分复核（一致性）]**。
- gain=1.138629436111989 四臂同值 **[已复核]**；Stack `min Δm=−3.82e-9`（=本席 m27−m28=3.824e-9）**[已复核]**。
- 未复核项：T2_validate.py 可运行性（席上没跑成）、evidence-distance 出处行（GLM:32-33 /
  NONGEOMETRIC:443 的 grep 断在 shell 错误上）、(d)(e) 两节结论（未产出）。 **[未复核]**

## 5. veto 席（ada30027e77b31b71，dimension=veto）——无官方裁决，实质零产出

状态：58 行转录，无任何最终文本块、未写 `V_veto.md`；转录全程=通读否决清单/摘要/D1–D4/T2/T3 + G1 数组
抽查打印（例：MrProBM trans max 1.3934@g31 与 JSON 一致 **[已复核]**、E3_BM_gain1 89.5833/58.8194
与 G1 面板分一致 **[已复核]**）。**D1–D4/T3 的"由 X 推 Y"是否踩死跳接的逐条裁决从未执行**，
否决合规目前只有 tables 席 §6 的自评（tables 席已对照否决清单声明合规 **[重建（他席覆盖，窄域）**]。
本维度**整体悬空**，等待重跑席 ac82fd9e5ac62af37（在途）。 **[已复核（转录状态）]**

## 6. UNCONFIRMED 总清单（回给主循环）

1. T3 四个极限定理推导步、EVQ→cosh 极限运算、MrRoPE ε_j 和=1↔预算守恒换算 —— 无人执行（math 席崩于其前）。
2. D4 §2 全局 min-max 证明（1/14 界、uni14 达到）与盲区表 —— 席上重算与文档一致，但本席未独立证明。
3. c1 可行窗两席网格互差：(1.390,0.55) 一格与 (1.371,0.45) —— 待 math 重跑席定稿。
4. Kendall τ=+0.852（25/27）数值 —— 未复算。
5. G1 生成器重跑字节全等 —— 席上声明，本席因只读边界未重跑。
6. ruler.jsonl 逐行 id 级翻转与 evidence-distance 出处 —— predictions 席被工具错误截断；仅一致性复核。
7. veto 全维度（10 类复发跳接、C1–C8、三撤回、E7/E8 教训、校准-KL 降级、J_r 定位逐条对 D1–D4/T3 的裁决）。
8. predictions 席 (d) Smooth/pair"事后 vs 预测"判定与 (e) 最强/最弱句。
9. 0450↔080 互指漂移细节（FLAG-④）—— tables 席 blocking 只复核到"0446 异表存在、三臂无契约、0440 同号两份"。

## 7. 与在册 GROUND 事实的闭合

- "Queue: 0441 running；0446→…UNQUEUED 需用户显式重授权"——tables 席 blocking 与本席目录列举与该口径相容，
  本席不扩述队列状态。
- "T1 29 checks ALL MATCH（§5.2）"——本席以 261/261（v_tables_check 全量重跑）为更强超集旁证。
- "V×4 mid-stream 三条证据"（HGL 快带偏离、aa29c4 复确认 Stack、s28 全局洞 g39）——三条**全部升格为 [已复核]**。
- 新增待传导：§3-① 死旗标与 README 例外清单四臂口径（含 YaRN 删名）、§3-② D4 行标签消歧建议、
  D1:22 "÷4.93" 残留（tables 席在册，本席确认 4^m40=4.9638）。
