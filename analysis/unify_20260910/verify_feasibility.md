# 对抗验证 · 维度=方案可行性与否决清单 · proposal="A+B"

日期：2026-09-10。验证者：feasibility-verifier（workflow-1 三验证器之可行性席）。
载体说明：与经验证据席相同，**提案 A/B 正文未落盘**（分派"被检文件"字段为空）。本席按分派口径取提案必然继承的规范载体：
`docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md`（BUDGET，§3 自带 A/B/C 候选与判定规则）、
`docs/research/UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md`（UNIFIED，§8.4 novelty、§9 冻结队列）、
以及 `analysis/unify_20260910/tables/ground_truth_tables.json`+`GROUND_README.md`（G1）、
`analysis/unify_20260910/digests/digest_failure-records.md`（否决清单）、
`digest_panel-results.md`（面板数字地面真值）、
`docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`（§2/§7 否决谱系原文，本席直读 :29-50、:206-248 复核 digest 转述属实）、
`docs/research/ROPE_GLM_6PRO_REVIEW_AND_VALIDATION_20260910.md`（6Pro/GLM 同日冻结裁决）、
`docs/theory/THEORY_IRONCLAD.md`、`docs/theory/THEORY_MATH_VALIDATION.md`（两文件均自标"历史文档，不再是权威参考"，本提案未直接引用其被降级结论，无冲突）、
`docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md`（用户协议纠正 :9-31）、
`results/nongeometric_screen_20260909/**`（本地镜像：contract/ruler/nll/queue/deferred_queue）。
**提案正文落盘后，凡措辞与本表核验对象不同者必须复跑。**

## 0. 本席实际运行的 CPU 检查（全部可复现）

脚本：`analysis/unify_20260910/checks/feas_mj_tables_check.py`（`python3` 直接运行，纯 stdlib）。
它从 BUDGET §3 声明的公式独立重建 A=0446 StackFrontBack 与 B=0448 MrProN16 的 64 槽 m_j/ν_j 表，
再对部署 fp32 张量（`reference_tables.json`、`results/*/contract.json`）与恒等式做自检。关键输出（原文数字）：

- 公式 vs 部署：native 最大相对差 **8.205e-08**；MrPro 公式 m_q=q(q+1)/306 **8.371e-08**（fp64 对 fp32 存储 ≈1 ulp；G1 用部署同序代码路径做到 18/18 逐位一致）。BUDGET:49 声称 ≤4.3e-8——量级成立，本席口径下未精确复现该常数（记 supported，不加分）。
- 部署反推复核：E1_s28_less 仅槽 28 与 MrPro 不同，m28=**0.065359**=m27（"前驱指数"属实）；LBS 仅槽 36–39 不同，构造 `ν−1/131072` 逐槽成立（ok=True），m36–39=**0.625169/0.729476/0.846534/0.979914**。
- **A 表生成成功**（fp64 重建 与 部署数组拼接 relerr 5.2e-8；与 G1 `StackFrontBack.nu_j` relerr **0.000e+00**，两条独立重建互相印证）。恒等式：ν 严格递减（min diff 7.475e-08>0）；m 单调非降（min diff −1.078e-07，纯 fp32 反演噪声，1e-6 容差下通过）；m23=0、m40=1（噪声内）；**水床和 Σ_{g=23..39}(gap−lnθ/64)=1.386294386 vs ln4=1.386294361，误差 2.45e-08**；17-gap 总和 5.056039（与 GLM 复核稿 :7 的 5.056039 一致）；Σm=29.5275、过渡段 Σm=5.5275；D36–39=77954/90082/105953/**127473**（doc "127.5K" ✓）。
- **B 表生成成功**（纯公式 fp64，恒等式全部 **精确**：水床和误差 **0.00e+00**；m 单调 min-diff 0.0；ν 递减 min 7.475e-08；box [0,1] True；m23=0/m40=1 True）。m28=**0.110294**（doc .110 ✓）；m36–39=**0.6691/0.7721/0.8824/1.0000**（doc ✓）；**D39=131,072=L 精确**（"39 槽完成" ✓）；Σm=30.0000。与 G1 `MrProN16` relerr 1.06e-07（m maxdiff 3.3e-08，同为 fp32 存储差）。
- **洞比率复算（用提案自己的定义 ρ=T_{g+1}/T_g=ν_g/ν_{g+1}，UNIFIED §1-3/BUDGET §2"相邻部署周期比"）**：A max 洞=**1.4930**（@gap38；BUDGET 声称 **1.86×**）；B max 洞=**1.4608**（@gap38；声称 **1.76×**）。与 G1 mismatches #1（1.493/1.461/1.476 vs 1.86/1.76/1.80，穷举 2-gap 跨度/D 域比/×原生比均不吻合）独立同判：**不可复现**。原生比 1.2410 复现 ✓。
- 方向审计：B 的 m24–28 相对 MrPro 全为**正增量**（+0.000817…+0.012255）——前端比 MrPro 更压，**反于 s28_less 方向**；BUDGET:49 自己承认（"B/C 前端 28 槽比 MrPro 更压——反向于 s28_less"），诚实标注，supported。
- 端点违例归因复算：MrUni m23=−0.000000（噪声→0）、m40=1.000000、m24=0.058824=1/17、m28=0.294118=5/17——部署表是**过渡段内线性斜坡、段外逐位同 MrPro**，**两个端点不变量都不违反**；HighGapToLong m23=**−0.155715**（真 I1 违例）；E2 m40=**1.155715**（真 I2 违例；有效尾比 4^1.155715=**÷4.9638**，非 "÷4.93"）；E8 ν51=0（j=51∈尾部带→**I2 侧**违例，UNIFIED §1:13 把 E8 列在 I1 名下=错位）。
- 耗时实测（吞吐核算，答 V-F8）：`results/*/ruler.jsonl` 逐行 `elapsed_seconds`：36 行面板（如 E1_s28_less，24 行 128K + 12 行 32K）生成合计 **14.8 分钟**（128K 行均值 34.8s、max 37.9s；32K 行均值 4.5s）；全部 25 个有计时面板的 36 行方法均为 14.6–14.9 min。0441 pilot 12 代 268.7s（NONGEOMETRIC:39）。NLL 48 条为纯前向（16 doc×3 长度≈0.92M token），RTX 4080 SUPER（runtime.json 实载）上量级分钟。
  → "约 45–55 分钟/张（36 行+48 NLL）"**高于实测下限（生成 15 min + NLL + 装载开销）**，属保守估计；3×~50 + 0450（3 法×4 任务×16 行 128K≈192 行×35s≈1.9h）+ 0451（≈20–35 min）→ **总 4–7 h，"约 6–7 小时"在实测吞吐内成立**；"基线复用历史 MrPro 回执"与面板 baseline 字段复用一致（digest_panel §2.3，score_sum 29.2167 三源同值）。

**结论(1)：两提案的 m_j 表 CPU 上现在即可完整生成，恒等式自检通过（B 精确、A 至 fp32 噪声 2.5e-8 内）；唯一数值失陷是候选表的 max 洞列（1.86/1.76 不可复现）。**

## 1. 否决清单逐条对照（digest_failure-records §4 全表 46 条 + SYNTHESIS §2/§7）

对提案文本（BUDGET 全文 + UNIFIED §1-3、§5、§7-9）逐组过表。**未发现 V-A（代理⇒能力）与 V-E（死区复活）违例**：判定门全部挂在任务分（128K/32K macro），不挂 PPL/NLL/Gram；`±v 镜像`、三条件因子、校准-KL 降级（UNIFIED §8.2/§8.3）是对否决清单的正确吸收；V-D21（"training-free+static+nongeometric 无人占据"）未被声称；V-E5 三类产出形式上满足（候选解+可区分预测+预注册判定）。违例/超界集中于**归因强度与状态标注**：

| VETO | 提案处 | 判定 |
|---|---|---|
| V-D20（家族级关闭禁语）+ 6Pro #3 | BUDGET §1"j≤23 恒等/任何压缩都直接付短端"、"改平台水平=破坏这个精确性质"、"必须被推到 m→1"被写成端点**定律**；实际反例=每个不变量各 1–2 个被测违例（I1 仅 HighGap，我实测；I2=E2/E8），6Pro 已裁"强基线设计、非零容忍定理" | **contradicted（普适措辞）** |
| V-D16（边际覆盖不够）+ 6Pro #2 | "D_j 弧安全→进入训练从未覆盖的相位弧段"（UNIFIED §1-1）对槽 36–39 不成立（原生 W 内已转 2.199/1.772/1.428/1.151 圈，圆周已全覆盖）；四个独立 D_j 不能替代联合模型证明；"75–112K 已验证危险带"6Pro/digest A8 判死 | **contradicted**（UNIFIED §4 自留"风险因子非门"不改变 BUDGET §1 的"必须"句性质） |
| V-B5/单因素 | BUDGET §2"E3 gain074 98.3/75.3 反证相位/弧安全**独立于** gain"——行标签错位（98.3333/75.3472=**Control_Mr_gain074**，E3_BM_gain074=100/70，本席与 digest A17 双重核）；且"独立"被 6Pro #6 数学否定（softmax(g²z_ν) 通常有非零交互） | **contradicted**（标签）+ **overstated**（独立性） |
| V-C4（口径混用） | BUDGET §1:12"E2…128K 崩至 54.7%"无面板列（基线实为 12 行 64.4444，非 78.125 锚；UNIFIED §1 带了列）；§2 表同病："MrUni 全表 ÷4→32K 64.6"定义已死（本席部署复算=过渡段斜坡） | **contradicted（定义）**+ **overstated（缺列）** |
| V-B1/V-B4（因果未隔离） | §2"两个互相独立长端机制由此析出"；实测 LBS 同样翻转 mk_2@89K（0→1）且**改坏** mk_2@96K（1→0）、Faster 单行翻转；s28 完全没动 D36–39 却修好被举作地平线证据的行（6Pro #5）；"所有赢家右移/8 构造单向梯度"（UNIFIED §3 结构结论，BUDGET §5 规则继承）被 6Pro #4 质心表推翻（LBS/P2 左移） | **contradicted**（梯度句，digest A7 判"被推翻"）；行级一一对应= **overstated** |
| V-F2/V-F10（预注册分支完整+禁"崩"字滥用） | 判定规则标"先行冻结"✓，但 A 分支不完备：128K∈[78.125,83) 或 (≥83 且 32K<80) 无动作；B/C"短端可控"无阈值；"三者全败"才升级确认 | **overstated**（分支表未闭合全空间；主终点先行固定这一半成立） |
| V-F6（样本单位/功效） | A 通过线 83% 锚在 s28 的**2 行**开发翻盘（+5.208pp 全部来自 mk_2_131072_2 与 mq_131072_3 两行）；无功效估计 | **overstated** |
| V-C2（身份混淆） | 队列号复用：`deferred_queue/20260910_candidate_quality/0446_HighGapToMid.json`（被撤回臂）与 BUDGET 新候选"0446 StackFrontBack"同号；0450 在本地队列的真实契约是 `queue/080_E1_holdout16.json`（编号漂移） | **blocking（治理风险）**，非既成违例 |
| V-D22（"继承已验证优势"禁语） | BUDGET §3"A 用**已验证**的双端值拼接"——组件分数[已验证-开发面板]成立，但拼接效应＝0446 待检假设，且 pair(28+29) 同族双手术前科 −4.17pp、`summary.json.scope` 自带 "Historical development inputs; not independent confirmation" | **overstated→若以 dev 面板通过宣称"叠加成立"则 contradicted（循环）**：同一 36 行固定输入同时供证据、阈值与裁决场，属"提案引用待证结论"边界情形，本席按任务规则记 **contradicted（弱形式循环）**并给降格措辞（见 §3） |
| V-A/V-E/V-D5/V-D13/V-D14/V-D17/V-D18/V-D19 | 未发现对应被禁形式（洞/Σm/D_j 在 G1/提案中作被测量与构造动机；判定门为任务分；E7-BF16 教训被正确引用约束 J_r；margin 可加性未违反） | 无违例 |

## 2. 判定实验 × 冻结队列 0446/0448/0449/0450/0451

- **自洽性**：BUDGET §4 与 UNIFIED §9 队列内容/顺序逐字一致；两者都写明"未执行、GPU 已关、恢复后自动按序执行"，digest_panel §1.4 确认五者无回执——提案没有把未执行写成结果，这点守规矩（supported）。
- **与用户记录的协议冲突——contradicted（阻塞项）**：同日生效实验记录 NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md（:9-31，且 :13 "This also supersedes the immediate-expansion priority"）写明：① "**No new 32K experiments or automatic 36-row completion are queued**"；② 128K 预检放行 pilot 时"**does not release the deferred 16-row expansions or any 32K work**"；③ "**The immediate GPU priority is independent confirmation of full P2's measured long VT/QA gains**"（即 0451 应先行）。而 BUDGET §4 队列=三张**新 36 行全面板（含 32K）+16 行扩展确认，且排在 0450/0451 确认之前**。本地无任何后续用户授权记录覆盖这三条（6Pro :61 的 11:06 授权只允许 GLM 校准"验证"，且明确"不自动执行 GLM 原来的五个候选"）。候选的**构造**符合 candidate-quality 纠正（从有效规则出发、有具体机制、非手设任意搬动）——被违的是**执行范围与优先序**。恢复 GPU 前需作者/用户显式重授权，否则队列与设计文档必须改为 long-only、确认先行。
- **最小成本真实吗**：真实且有实测支撑（§0 末段）：45–55 min/张 ≥ 实测 15 min 生成 + NLL + 开销；总 6–7 h 落在实测吞吐 4–7 h 带内；基线复用零重复计算有 baseline 字段三源同值佐证。RTX 4080 SUPER 单卡常驻 worker（runtime.json）与"无 GPU 空转"排程相容。
- **通过/失败动作闭环**：部分闭合（overstated）。缺：A 中间带动作、B/C"短端可控"阈值、以及 **A 通过分支"立即安排新样本确认"在冻结队列里无对应作业**（0450 契约只含 MrPro/s28_less/s29_more——测的是组件与新样本，**不含 Stack、不含 LBS**；Stack/LBS 的 holdout 完全开放）。另 `evidence_distances_20260910.json` 本地缺失（Core-C 曲线依赖它+0450 分桶），127K 距离一行连文档出处都没有（digest §1.4/§7-2）——Core-C 在恢复服务器同步前不可判。

## 3. novelty 威胁切割（arXiv:2607.07678 / 2607.10134 / 2602.05258）

- 三篇**真实存在**：`paper-2027/refs/references.bib:560`（wu2026datashapes，标题逐字=UNIFIED 所引）、`:524`（karypis2026lerope=2607.10134）、`:431`（li2026copeclipped=2602.05258，**主标题 CoPE: Clipped RoPE…**）；均已在 `02_related.tex:26/33/41` 被引——UNIFIED §8.4"必须引用并明确切割"的要求实际已半落地（引用在，区分句按 digest_paper-state :259 只需保留）。存在性 supported（bib+两处转录链接 arxiv.org/html/2607.07678v1、2607.10134v1，非凭记忆）。
- **2602.05258 的 provenance 错误——contradicted（仅身份标注）**：UNIFIED 称"**检索新增**"，但同一 ID 早在 09-07 已是本项目已知先例并做过干预（ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md:195 引 CoPE 原文+官方代码 commit f8957a1、:104 CoPE 实现事实；digest_failure-records §5 CoPE 行：Mr+CoPE 组合 128K 检索 8/8、UUID 0/8；V-D11 已因此禁"光滑幅度窗⇒更好"）。把已测先例写成"检索新增"低估了其威胁面；且其引语"低频成分同时支配外推 OOD 行为"本地无出处可核（unverifiable 引文本体）。切割论点本身（裁剪视角 vs 完成+守恒放置）成立。
- 2607.10134/2607.07678 的切割（训练线正交、数据涌现 vs 冻结放置）与 thread-core :1138-1141、TEN_CANDIDATE_PLAN:295 的既有裁决一致，**诚实，supported**。一处残留风险：thread-core :2898 明示"**LeRoPE 已经做了窗口内多干扰项检索，'我们也做 in-window' 本身没有新颖性**"——A/B 判定实验的核心终点恰是 multikey/multiquery 类面板，§8.4 未把这一重叠切进 LeRoPE 条（overstated 缺口）。
- §8.4 总结论"统一预算视角不能单独当 headline"+三条差异化（方向规则±v/冻结兼容符号预测/EVQ source 证伪）自我降格到位，无虚宣称。

## 4. 总判定

**构造层：完全可行**——两表 CPU 可生成、恒等式自检通过、成本与硬件匹配、判定规则先行冻结、否决清单的 A/E 大组无违例。
**证据措辞与授权层：不可原样入队**——(a) 端点"定律"、危险区"必须"、右移梯度、E3 标签、MrUni 定义、洞列 1.86/1.76 六处与冻结复核/本席重算冲突；(b) 队列范围/优先序与用户当日三条记录在案的协议纠正正面冲突且无覆盖授权；(c) A 的通过分支把开发面板级结论写满（83% 阈值锚 2 行 dev 翻盘、确认作业不覆盖 Stack/LBS），存在"以发展集场子验证由发展集析出的机制独立论"的弱循环——通过措辞必须降为"开发面板不崩塌，叠加假设存活，待 holdout"。

**若作者按下述修，A+B 可执行**：① 队列前显式记录用户对"36 行含 32K + 新候选先于 P2 确认"的重新授权，或改 long-only、0451 先行；② 候选表洞列以 G1 §6 重算值（1.493/1.461/1.476）替换并声明构造不依赖该列；③ BUDGET §1/§2 与 UNIFIED §1/§3 吸收 6Pro #1–6 与 E8→I2/MrUni→无违例的归因修正；④ 判定规则补全分支+给 A 通过分支挂上真实存在的 holdout 作业（含 Stack/LBS）；⑤ novelty 条把"检索新增"改为"已知先例（09-07 已测 CoPE 臂）的补充定位"，并补 LeRoPE in-window 重叠句。
