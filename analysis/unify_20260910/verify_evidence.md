# 对抗验证 · 维度=经验证据 · proposal="A+B"

日期：2026-09-10。验证者：evidence-verifier（workflow-1 三验证器之经验证据席）。

## 0. 范围声明（先于一切结论）

被检提案正文（workflow-1 提案 A/B）**尚未落盘**（任务 #1 仍为 pending；分派给我的"被检文件"字段为空，全仓 `grep 提案A/Proposal A` 无载体文件）。因此本验证按分派说明中的经验数字清单逐项建地面真值，载体取提案必然继承的规范文本：
`docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md`（下称 UNIFIED）、
`docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md`（下称 BUDGET，其 §3 自带 A/B/C 候选标签）、
`analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md`（下称 KKT-P）、
`analysis/unify_20260910/INTEGRATION_20260910.md` §3。
**提案正文落盘后必须按原文措辞复跑一次本表**；凡本表只能核验"数字是否真实"而不能核验"提案如何措辞"的条目，verdict 记 unverifiable 并在此说明。

方法：所有数字直接由本地原始 JSON 重算（python3，非引用 digest），证据基座文件全部实际打开：`analysis/unify_20260910/digests/digest_panel-results.md`、`digest_failure-records.md`、`digest_pro-materials.md`、`digest_theory-0910.md`、`docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`、`docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`（下称 6Pro）、`docs/theory/EVQ_COSH_THEORY.tex`、`docs/theory/THEORY_IRONCLAD.md`（其头部自标"历史文档，不再是权威参考"）、`results/nongeometric_screen_20260909/**`、`analysis/unify_20260910/tables/ground_truth_tables.json` + `GROUND_README.md`。

## 1. 地面真值重算（本席独立重做，不依赖 digest 转抄）

面板逐宏分（`results/nongeometric_screen_20260909/results/<m>/summary.json`，candidate/baseline 双栏逐条重读）：

- MrPro 锚点：**87.2222 / 78.1250**（= 所有 36 行方法 summary 的 baseline 字段，score_sum 29.2167；另 `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json` models/qwen3/arms/MrPro 同值）。78.13 为四舍五入。✓
- E1_s28_less 87.2222/**83.3333**，W/L=2/0；逐行不一致恰 2 行：`niah_multikey_2_131072_2` 0→1、`niah_multiquery_131072_3` 0.75→1（ruler.jsonl 全列）。
- LongBridgeSlower 80.5556/80.0694（3W/3L，六行不一致：vt_131072_0 .2→1.0、mk_2_131072_2 0→1、vt_32768_1 .8→1.0，**同时** mk_2_32768_1 1→0、mk_2_131072_1 1→0、fwe_131072_3 回退）。
- LongBridgeFaster 87.2222/73.9583（0W/1L）；E1_pair28_29 同分异构（引用陷阱，digest §4.4）。
- FullLagP2_Transfer3B 72.9167/81.6667（6W/6L；+3.5417 vs MrPro、+6.3195 vs Control_Mr_gain074=75.3472 算术成立）。
- Control_Mr_gain074 **98.3333/75.3472**；E3_BM_gain074 **100.0/70.0**；E3_BM_gain1 89.5833/58.8194。
- Smooth_MrBudget 87.2222/68.3333（末 gap 重算 0.0425 ✓ 声称 0.042）。
- HighGapToLong 70.1389/67.3611（−17.0833/−10.7639，0W/7L ✓ "36 行 0 提升"）。
- MrUni 64.5833/73.3333。
- E2_tail_more（12 行）100/54.7222（基线 100/64.4444，0W/2L）；E8_zero51（12 行）100/50.5556（0W/2L）。
- 冻结表逐位反推 m（contract.json `spec.table.values_float32`，ω_j=10^(−6j/64)）：s28_less m28=.0654、余槽=MrPro；LBS m36–39=.6252/.7295/.8465/.9799（D_39=32768·4^.9799≈127.5K ✓）；Faster m36–39=.5656/.6455/.727/.8081（与 LBS 同端点同增益镜像）；**MrUni m23=0、m24=.0588、m28=.2941、m39=.9412、m40=1**；HighGapToLong **m23=−0.1557**；E2 **m40=1.155715**；E8 ν51=0；P2 m36–39 全=1、m23=.0006。守恒 Σm(24..40)=1.0000 ✓。gain 1.1386294 = 1+.1·ln4 ✓；P2 gain 1.1025858 = 1+.074·ln4 ✓。

## 2. 判定表（claim → verdict → 证据）

### C1 "36 行面板" 口径 — **supported**
6 任务×(32K 2 行+128K 4 行)=36，任务集 {niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1}；`results/*/ruler.jsonl` 行数与 row_id 模式直验；12 行早期子集=每任务 `*_0`（基线 100/64.4444）。摘要：digest_panel-results §2.3、§5。

### C2 "MrPro 87.22/78.13" — **supported**
精确 87.2222/78.1250；三独立本地源同值（§1 上文）。提案任何以此为基线的差值算术均先过此锚。

### C3 "四不变量实验（MrUni/HighGap/E2/E8 违反 I1/I2 → 崩，端点不变量证据）" — **overstated，两处 contradicted**
逐位重算的归因矩阵：
- HighGapToLong：m23=−0.1557 <0，**唯一真正违反 I1（j≤23 m=0）的实验**；数字全对。supported。
- E2：m40=1.1557>1，违反 **I2**；supported（12 行口径必须注明；且"÷4.93"重算为 ÷**4.9638**，常数小幅 contradicted，实质成立）。
- E8：置零槽 51∈尾部带（j≥40）→ 违反 **I2**，非 I1。UNIFIED §1（:13）与 KKT-P §1.3 把 E8 列入 **I1** 证据 = **contradicted（错位归因）**。digest_panel-results §4.5 也将 E8 归入"破坏精确 ÷S 端点"（I2 侧）。
- MrUni：部署表位级核验 m23=0、m40=1（`ground_truth_tables.json` mismatches："过渡段内部线性斜坡 m_j=(j−23)/17，段外与 MrPro 逐位相同，bit_exact=True"；本席重算同）。它**两个端点不变量都不违反**——32K 64.6 是"桥内过度压缩"（L_near）证据，不是端点证据。UNIFIED §1 把它记作"I1 证据 + 全表 ÷4 定义"，**双重 contradicted**（定义被 6Pro §3 纠正后 UNIFIED §1 正文仍未回改，digest_pro-materials §7-7 同记）。
- 汇总口径："I1/I2 由四个独立实验钉死"（UNIFIED §8.3.2；digest_theory-0910 U1 把三实验全记 I1 名下）→ 实际 I1=1 个违例实验、I2=2 个、MrUni=0 个；且 6Pro §3 判定强度为"强基线设计约束、非零容忍定理"。**钉死措辞 overstated**。KKT-P §1.3 已吸收该修正，UNIFIED §1/§3 未回改。

### C4 E2/E8 数字 — **supported（带强制口径）**
54.7222、50.5556 均 12 行面板、基线 64.4444、各 0W/2L；"崩"相对本面板成立。**禁止**与 78.125 锚直比（digest_panel-results §4.10 列为最高复发错误）。注意 INTEGRATION §3"钉死数字"清单写"E2 54.7；E8 50.6"**未带面板列**——本席判定该处违反自家规范（digest §7-8 建议强制带列），若提案从 INTEGRATION 抄数会继承此隐患。

### C5 holdout 未跑 / dev-holdout 混淆 — **supported（且提案必须据此自我降格）**
- `results/.../holdout_results/mixed_s20260910/`：仅 qa_1@128K；MrPro 4 行（1/4，唯一正确行=qa_1_131072_3）、Control 3 行 0/3、P2 3 行 0/3，**缺的恰是 MrPro 唯一正确行**→ 截断偏置，非确认非否证（digest §2.4）。
- 0446/0448/0449/0450/0451 **无回执**（UNIFIED §9"GPU 已关、队列冻结"；`ground_truth_tables.json` methods.StackFrontBack 得分 null）。
- 每个 summary.json 自带 `scope: "Historical development inputs; not independent confirmation"`——原始数据自证全部为开发集。
推论：提案 A/B 的一切效果断言（含 BUDGET §3 判定规则"A 若 128K≥83% 且 32K≥80%…"、"B/C 单调则得主线"）目前是**预测**；凡措辞为已验证 → contradicted by status。另 UNIFIED §7 claim-3 "跨任务/跨输入方向一致"仅在开发面板 36 行内成立，holdout 唯一信号（P2 QA 方向）尚未确认（0/3 缺行）。**循环性**：s28_less 与 LBS 由同一固定态选择器在同一面板上选出（UNIFIED §8.3.1 自认"循环"），再用其分数论证机制——本席按规则记：若 A+B 以组件分支撑复合效应"已验证"，记 **contradicted（循环论证）**。BUDGET §2 "两个互相独立的长端机制由此析出"＋§3 "A 用已验证的双端值拼接"正处该风险位置：两手术各自的分数=已验证（dev），"独立可叠加"=0446 要检验的假设，且 pair(28+29) 同族双手术已给出 −4.17pp 反前科（INTEGRATION §3"逐槽可加性已死"）。

### C6 ±v 镜像控制是否真实存在于所引证据 — **overstated**
- **存在**：LBS/Faster 是货真价实的 36 行、同 gain、同端点镜像对（ν 逐槽 ±1/131072 rad/token；m 重算 .6252…/.9799 vs .5656/…/.8081）。Slower +1.9444 / Faster −4.1667 ✓。这给"方向有预测力"一个真实单对证据。
- **半失效**："s28 双向对照持平"（UNIFIED §8.2）——reverse_matched 确在（`results/E1_s28_reverse_matched/`）但 (i) 只有 **12 行**子集，而 s28_less 的两个增益行都是 `_2/_3` 行，**在 12 行子集上正向本身 0/0 持平**，镜像持平不构成对该方向的检验；(ii) 反演构造式未复原（部署 m28=.132270 ≠ 候选式 2m28−m27=.130719，`ground_truth_tables.json` mismatches），扰动范数"同"仅在近似意义成立。
- **协议层**：±v 作为标准协议 **未执行**——digest_pro-materials §3#9 "[假设]（未执行）…从未跑"、digest_theory-0910 U12"待 0455/0456"。提案若声称"规则已按 ±v 协议验证" → contradicted。

### C7 行级 n 过小处的解读 — 逐项
- s28_less "+5.21pp"：全部来自 2 行（1 全翻+1 翻 1/4），36 行开发集、单 seed greedy、选择器循环——**分数 supported、"修 89K multikey 绑定"的因果= 部分证据**（digest A2/A7；6Pro §5 还指出 E1 完全没动 36–39 槽的 D 却"修好了举作地平线证据的行"）。
- LBS "+1.94pp"：6 行不一致 3W/3L，净增全靠 vt_0 一行；且同表 mk_2@96K 被**改坏**（1→0，UNIFIED §4 带星号，BUDGET §2 未提）。BUDGET §2 "VT 75→95" 单报正例 → **overstated**。
- P2 "+6.32pp matched"：算术 ✓，但 6W/6L、短端 −14.3/−25.4pp；pilot n=3 不可判（digest §2.4）。
- E2/E8 各 0W/2L（12 行）→ "违反端点就崩"支持；"端点全守恒必最优"不支持。
- "75–112K 已验证危险带"：**contradicted**（6Pro §证据距离复核：FWE 距离定义不成立、VT 只是链 max、MQ 行混双距离、两行翻盘样本过小；digest_theory-0910 U9：0452 判死"未训练弧"语义，36–39 原生已覆盖全部相位；vt_1@127K 连文档出处都没有——digest_panel-results §1.4）。
- "所有赢家右移/8 构造单向梯度"（UNIFIED §3 结论；KKT-P 2.4 旧表述）：**contradicted**（6Pro §4 质心表：s28 右移、LBS/P2 左移各有收益；digest A7 判"被推翻"；digest §7-5 要求重写）。

### C8 提案引用的其余具体数字
- BUDGET §2 行 "E3 gain074 | 改 gain 非表 | 98.3/75.3"：**contradicted（标签错位）**——98.3333/75.3472 是 `Control_Mr_gain074`（MrPro 表@.074）；E3_BM_gain074=100.0/70.0（digest A17，本席 raw 重算同）。"gain 反证弧安全独立于 gain" overstated（6Pro §6：softmax(g²z_ν) 存在非零交互；四格只支持"长端不由 gain 得救"）。
- 候选设计参数：官方 gain ✓；"公式↔部署 ≤4.3e-8" supported 且实际更强（逐位 0 误差，GROUND_README:128）；LBS D=127.5K ✓；**max 洞 1.86×/1.76×/1.80× contradicted**——按 UNIFIED 自家 ρ=T_{g+1}/T_g 定义重算为 1.493/1.4608/1.4757（`ground_truth_tables.json` mismatches 前三条；文档未给"洞"公式，二者不可能同时成立）。
- "27 个已测面板"（UNIFIED §3，提案证据基数）：**unverifiable**——本地恰 26 个方法目录（§1 列目录实测），第 27 个身份未钉死（digest §7-1）。
- "OLMo ScaleTaper 构造检查已通过"（BUDGET §5）：**supported 但语义受限**——`planned_controls/scale_taper_cross_model_geometry.json` 自带 scope："CPU construction geometry only; no OLMo model run is queued or implied"。
- "7B/OLMo E1 迁移已有"（UNIFIED §7 预问答）：**supported 存在、方向不利**——7B 18 行长端 −1.67pp（1W/1L）、OLMo −3.33/+3.82（`transfer/*/summary.json` 直读）；只能作"单槽规则不可直接跨 checkpoint"的证据（digest A14），不得作泛化支持。
- 距离联表 89K/106K/96K/117K：得分列本地全对（digest §5.4），距离列依赖服务器端 `evidence_distances_20260910.json`（本机缺失，6Pro :30 声明已读并核对）→ **部分证据**；127K 一行 **unverifiable**。

## 3. 否决清单对照（SYNTHESIS/digest_failure-records VETO 交叉）

提案若含以下形态即违 veto：把固定态/代理分数接成能力（V-A6：E8 恰是反例）；小样本升格（V-F6：s28 2 行）；未测写成否证/已验证（V-F10、C5）；单因素归因（V-B5：gain 四格）；"改名复活"（右移梯度→"减少累计压缩"须换证据不换名字，digest §4.8）。本次重算未发现任何数字造假迹象（126 锚点 121 MATCH），失分点全部在**归因强度与状态标注**。

## 4. 结论（证据席）

条件通过（数字可信、标签须改）：
1. 全部面板/表格分数与守恒、端点重算一致，可作提案的合法地面真值。
2. 提案 A/B 的**效果**部分是预测（0446/0448/0449/0450/0451 未执行，holdout 缺行），措辞必须停在"预测+判据预注册"。
3. 继承 UNIFIED §1/§3 时必改四处：E8→I1 错位、MrUni 定义+端点归因、"四实验钉死"强度、"赢家右移"梯度（已被质心表推翻）。
4. ±v：只可写"LBS/Faster 单对镜像 + 12 行 reverse 持平（子集对正向不敏感）"，协议整体未跑。
5. 洞比率三数与 ground-truth 冲突未澄清前，A/B 候选的设计表不得进论文图表。
6. 落盘提案原文后需按 §0 复跑一次（本表未核措辞）。
