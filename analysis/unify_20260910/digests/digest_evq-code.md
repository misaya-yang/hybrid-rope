# digest evq-code

任务：EVQ 恢复（experiments/evq_recovery/）与三个 CPU 审计脚本（QK 算子 Gram、softmax 混频、QK 加权源子空间）的代码级复盘——EVQ 相关管线现在到底能算什么、不能算什么；QK 加权 Gram 排序失败的具体数字表；无 GPU 条件下还能推进哪些检验。
证据等级标注约定：[已验证]=本地代码/收据文件/已测数字直接支持；[部分证据]=有测量但范围/样本受限；[假设]=机制解释或未测推断。所有文件路径相对仓库根 `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/`。

---

## 1. 来源清单

### 1.1 EVQ 恢复代码包 `experiments/evq_recovery/`（15 个文件，全部在提交 a6e3aba "Workspace snapshot 2026-09-10 07:49" 首次入库）

| 文件 | 字节 | 行数 | 作用 |
|---|---:|---:|---|
| `README.md` | 7778 | 84 | 研究问题、基座、对照集、数据、限制声明 |
| `__init__.py` | 78 | 1 | 包说明："Controlled non-geometric grids under real-length all-linear adaptation" |
| `acquire.py` | 6064 | 134 | PG19/LongAlign/QASPER 下载，Range 断点续传、MD5/SHA256 核验 |
| `data.py` | 2778 | 80 | 流式原语：JsonlIndex（字节偏移）、supervised_chat、QA F1/exact 归一化 |
| `tables.py` | 3598 | 90 | 五臂频率表构造：anchored_cosh / exponential / hybrid / match_deformation / 官方 YaRN |
| `prepare.py` | 15023 | 287 | QASPER 评测池、PG19 16K 窗口、LongAlign SFT、Native replay、错位重叠筛查 |
| `prepare_ruler.py` | 3376 | 63 | 固定 upstream RULER 生成器（c3f5e3b…），niah_single_1 / niah_multikey_3 / vt |
| `make_plan.py` | 2951 | 45 | 从本地资产生成 plan.json（含解释纪律字段 interpretation/teacher/hardware_note） |
| `validate.py` | 4934 | 89 | CPU 读回全部数组/行/哈希/频率表，输出 readiness.json |
| `runtime.py` | 5093 | 101 | GPU 加载入口：静态 FP32 RoPE 安装、LoRA/full、三步合并 loss |
| `train.py` | 6738 | 121 | 可续训 LoRA/full 训练，smoke 模式，合同哈希防漂移 |
| `evaluate.py` | 7439 | 138 | 完整生成评测（QA/RULER/native/LM NLL），contract 防改输入续跑 |
| `compare.py` | 2879 | 57 | 配对、按 source 聚类的 bootstrap 对比，禁止跨指标合并 |
| `launch.py` | 2309 | 50 | 相位编排（smoke/recovery/shapes），默认只打印命令 |
| `test_recovery.py` | 5922 | 133 | 12 个 CPU 单测（tiny OLMo2，hidden=32、2 层） |

全部 `.py` 通过 `python3 -m py_compile` 语法检查（本地 CPU，2026-09-10 会话）。

### 1.2 审计脚本 `scripts/analysis/`（同一提交入库）

| 文件 | 字节 | 行数 | 作用 |
|---|---:|---:|---|
| `rope_qk_operator_gram.py` | 5599 | 115 | 冻结 Q/K 投影（含 bias）的旋转双线性算子精确 Frobenius Gram，纯 CPU、不加载模型 |
| `rope_softmax_harmonic_audit.py` | 3494 | 73 | 条件 softmax 混频（Bessel 展开）恒等式与人工行例子的 CPU 验算 |
| `rope_weighted_subspace_audit.py` | 3369 | 70 | 用 Gram 加权源弱方向的远程响应能量，比较 8 张表 × 3 个截断阈值 |

### 1.3 结果收据（`results/nongeometric_screen_20260909/`，文件 mtime 即计算时刻，2026-09-10）

| 文件 | 字节 | mtime | 内容 |
|---|---:|---|---|
| `planned_controls/source_subspace_transport_audit.json` | 3755 | 05:51 | 13 张表的未加权远程 U + 局部失真 C26/C256/C2048/C32768 |
| `planned_controls/qk_operator_gram.npz` | 4038895 | 06:09 | 每层 Gram `per_layer(36,128,128)`+`mean` |
| `planned_controls/weighted_source_subspace_audit.json` | 33742 | 06:24 | Gram 加权 U_H，3 阈值 × 8 方法 × 36 层 |
| `planned_controls/regularized_worst_direction_audit.json` | 1290 | 06:28 | 正则化最差方向 Rayleigh 商（ridge 0.1→1e-8） |
| `planned_controls/remote_PI_distortion_audit.json` | 965 | 06:35 | 各表对 PI 的远程失真，未加权 vs QK 加权 |
| `planned_controls/softmax_harmonic_audit.json` | 2277 | 06:46 | Bessel 恒等式验算 + Qwen 28−2·29+30 混频周期 |
| `planned_controls/qk_operator_gram_with_means.npz/.json` | 5723658/578 | 06:55 | 增加 matched/independent content mean（Qwen2.5-3B 36 层 16 头 2kv head_dim=128） |
| `planned_controls/bias_harmonic_head27_8_audit.json` | 2519 | 06:58 | 按 (1,−2,1) 混频系数最大选出的 layer27/head8 的 bias-bias 位置项 |
| `planned_controls/bias_prior_local_KL_all_heads.json` | 117837 | 07:00 | 全 576 头 bias-bias 条件 softmax 对 Native 的局部 KL |
| `runtime.json` | 747 | 02:36 | 评测模型 = Qwen/Qwen2.5-3B-Instruct rev aa8e725…，GPU=RTX 4080 SUPER |
| `development_summary.json/.md` | 45684/2496 | 05:03 | 36 行开发面板汇总 |
| `results/{Smooth_MrBudget,LongBridgeSlower,LongBridgeFaster,FullLagP2_Transfer3B,HighGapToLong}/summary.json` | — | 09-10 | 每臂候选 vs MrPro baseline（score_sum 29.2167，32K macro 0.8722，128K macro 0.78125） |

### 1.4 理论文档（docs/research，2026-09-10 提交）

| 文件 | 字节/行数 | 关系 |
|---|---|---|
| `ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md` | 12726/281 | 两个 Gram 脚本的推导文档；加权排序失败记录所在 |
| `EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md` | 6186/138 | 对 `docs/theory/EVQ_COSH_THEORY.tex`（378 行）cos-only 碰撞核的精确修正 |
| `ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md` | 9322/197 | harmonic 审计脚本的推导文档 |
| `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` | 16571/143 | 36 行面板 32K/128K 得分总表 + EVQ"和解"节 |
| `BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md` | 6284/— | 规则表述；HighGapToLong 证伪 EVQ 字面处方 |
| `NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md` | 37360/630 | 同批机制文档（未直接引用 Gram 收据） |

### 1.5 辅助/历史来源

- `experiments/nongeometric_screen/smooth_budget.py`（weighted subspace 脚本 import 的 `construct`；KKT 认证最小粗糙度分配）。
- `docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json`（P2 表来源，7169B，09-07 21:16）。
- `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md`、`analysis/unify_20260910/raw/raw_thread-core.txt`（用户指令原文）。
- `results/fmrope_evq2_500m_s42_20260724/`（356KB，2026-07-24 的 EVQ-2/500M 历史 rebuttal：固定范围 Cosh τ=4 在 L=512/1024/2048 的 exp(NLL) 88.6/278.7/504.1 vs target-matched 29.8/42.2/132.3；单种子，仅 rebuttal 诊断）。
- 服务器侧（本地不存在，README 引用）：`/root/autodl-tmp/evq_recovery_20260910/` 准备根目录及其 `PREPARATION.json`、`tables.json`、旧失败日志。[本地未验证]

---

## 2. 任务时间线

### T1 · 2026-07-24：EVQ-2 500M rebuttal（历史，已完成）
- 目标：核对 Range-anchored Cosh τ=4 固定范围 vs target-matched 范围的小模型 NLL。
- 结果：[成功-诊断性] 固定范围在 L≥512 后 NLL 爆炸（88.6→504.1）；单种子、L=256 内两者相同。出处 `results/fmrope_evq2_500m_s42_20260724/evaluation/summary.md`。
- 限制（自带）："matched rebuttal diagnostic, not a general performance claim"。

### T2 · 2026-09-07/08：EVQ 服务器（无卡模式）资产与纪律
- 目标（用户）：不启动新实验、最多 3 个、不隐藏多候选扫描；EVQ 服务器只读缓存与输出投影权重（`USER_PROMPT_TRANSCRIPT_20260909.md` T3-P01，2026-09-07T13:49:40Z）。
- 结果：[中断→改道] 下载剩 3.7GiB 时用户令"转无卡"（`raw_thread-core.txt` 行 1747–1766）；无卡实例只有 2GiB 内存，双进程分词加载被 OOM 杀（行 4169 上文），改单进程顺序生成[成功-修复]；开卡前发现 transformers 新聊天模板返回 dict、旧代码把 dict 长度当 token 数导致评测样本全被过滤（行 1894）[成功-开卡前拦截]。
- 用户同时撤掉"逐 pair 测 benefit 再设计表"的方案："上一条……仍然是梦话，撤掉"（行 4169，2026-09-08T09:47:55Z）。

### T3 · 2026-09-09 深夜→09-10 上午：36 行非几何面板与源子空间计算链（CPU+短 GPU）
- 23:46 qualification → 04:05 reference_tables.json（Native/MrPro/MrProBM）→ 05:03 development_summary（26 臂面板测完）。
- 05:51 未加权源子空间传输审计[成功]；06:09 QK 算子 Gram（Qwen2.5-3B 切片，CPU，恒等式误差 1.776e-14）[成功]；06:24 QK 加权 U_H 审计[成功——得出"加权不救排序"结论]；06:28 正则化最差方向审计[成功]；06:35 remote-PI 加权失真[成功]。
- 关键失败发现（见 §4 F3）：Smooth_MrBudget 在所有未加权/加权线性指标上都优于 MrPro，但 128K 开发得分 68.3 vs MrPro 78.1（−9.8pp，score_sum 26.87 vs 29.22，`results/Smooth_MrBudget/summary.json`）[已验证，n=36 开发面板]。

### T4 · 2026-09-10 06:46–07:00：softmax 混频机制验算（CPU）
- 06:46 `softmax_harmonic_audit.json`[成功]：Bessel 截断乘积与 exp(s) 逐点一致（误差 2.96e-9 / 3.12e-9 < 绝对尾部界 3.135e-9）；两快频人工行证明"单频圈数多≠粗平均"（首半部注意力质量 60.96% vs 50.03%）；Qwen 真实表 28−2·29+30 混频周期 70285.94 token。
- 06:55–07:00 Gram 升级为含均值版 + bias-harmonic（layer27/head8 按系数选出，"not by outcomes"）+ 全头 bias 局部 KL[成功-数学层]。机制是否为模型真实原因：[假设]，文档明言未排队任何新模型干预。

### T5 · 2026-09-10 上午：EVQ 非局部核修正（CPU 数学）
- `EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md`[成功]：精确核 K_L 分解出非局部对数脊 h_c；delta 近似以 A0=π²/(12c) 在一切非零波数上过罚；数值检验 5 点比率（0.9916@π … 0.3556@64）+ 5 组 (φ,ψ) 核值全部满足导出误差界（最大实测误差 0.004477 < 界 0.015378），SciPy 1.18.1 CPU。
- 对旧 EVQ/Cosh 变分论证的后果："变分论证不证明锐利中段过渡本质坏"；这是对建模步骤的修正，不是能力证据。

### T6 · 2026-09-10（当日）：EVQ 恢复实验包 = 开卡前准备（未执行 GPU）
- 目标（README 行 3）："包含 FFN 的 LoRA 在足够的真实长输入监督下能否恢复能力，以及 Cosh 相对其他非几何网格是否有优势。"
- 方案：OLMo-2-0425-1B-Instruct（实测约 1.485B，原生 4096），五臂表（§5.2），每更新 = 16K PG19 密集 CE + 长指令答案+EOS CE + 0.25×短 replay CE；8M/32M/64M/128M CPT token 固定调度；12 个 CPU 单测；全部入口默认 dry-run。
- 结果：[未执行-GPU 待开] 代码与 CPU 检查就绪；GPU smoke、基线能力、训练结果均未测（`validate.py` 行 76–77）。无任何本地训练/评测输出目录。README 行 84 记载 CPU 阶段已处理：下载截断（Range 补齐+PG19 云 MD5）、LongAlign SHA256 与官方 LFS 一致、生成器依赖缺失、初次数据字段错误；"旧失败日志保留于准备材料"（服务器侧）。
- GPU 队列（`UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §9）：GPU 已关；服务器冻结队列 0446 StackFrontBack → 0448 MrProN16 → 0449 MrProN15 → 0450 E1 holdout16 → 0451 P2 长端确认（本地 `results/nongeometric_screen_20260909/queue/` 只见 ≤0445 + 080_E1_holdout16，0446–0451 在服务器端）[部分证据-本地快照不齐全]。

---

## 3. 理论主张表

| # | 主张 | 证据等级 | 出处 | 后续是否被纠正/推翻 |
|---|---|---|---|---|
| C1 | 旋转双线性算子的精确 Frobenius Gram 可由 Q/K 行内积闭式获得（half-split 配对、GQA 共享、bias 增广） | [已验证]（恒等式检查误差 1.78e-14 < 1e-10；npz 形状 (36,128,128)） | `scripts/analysis/rope_qk_operator_gram.py` `check_formula`; `qk_operator_gram_with_means.json` | 未被推翻；但其"能排序表"的含义被 C5 否定 |
| C2 | Gram 成本 = 独立单位二阶矩输入下的算子 MSE | [已验证]（定义即如此，代码 docstring 明示） | 同上 | 边界被反复强调：**不是**激活统计、**不是** LM loss、**不是** softmax KL |
| C3 | 未加权远程源弱能量 U 与局部失真可选出更好的表（P2/Smooth 规则） | [部分证据→被否证为充分判据] | `source_subspace_transport_audit.json`; `ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md` L192–196 | **否证**：Smooth 的 U 与 C32768 都优于 MrPro，但 128K 得分 68.3 < 78.1；E1_s28 的 U 与 MrPro 几乎相同而任务输出不同（同文档；`results/Smooth_MrBudget/summary.json`） |
| C4 | 用学习的 Q/K 算子加权可以补上线性判据的缺口 | [部分证据→被否证] | `weighted_source_subspace_audit.json` | **否证（限定方式）**：Smooth 在三个阈值（1e-6/1e-8/1e-10）的加权 U_H 上都低于 MrPro，且**在全部 36 层都低**（`smooth_less_than_mr_layers: 36` ×3）；"merely weighting the same source-weak-energy criterion" 这条修复路径被文档关闭（DERIVATION L276–277） |
| C5 | 条件 softmax 中两个快频可产生慢混频包络；逐频圈数条件不足以保证粗平均 | [已验证-数学层]（Bessel 恒等式 + 人工行 60.96%/50.03%）；[假设-模型层]（未测任何真实内容系数） | `softmax_harmonic_audit.json`; `ROPE_SOFTMAX_MIXED_FREQUENCY_DERIVATION_20260910.md` L160–171 | 未被推翻；文档自限："does not establish that any particular harmonic caused the existing E1, P2, or Smooth outcomes" |
| C6 | EVQ 原始碰撞核的 delta/局部近似在每个非零密度波数上过罚变化，锐利中段过渡不"本性坏" | [已验证-数学层]（正斜率展开 −π⁴/(720c³)‖ρ′‖²；5 点数值界全过） | `EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md` CPU checks 节 | 是对 `docs/theory/EVQ_COSH_THEORY.tex` 局部近似步骤的**修正**；不是能力否证 |
| C7 | EVQ 字面处方"压高频侧给低频侧"在冻结模型上可行 | [已验证-失败] | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §6；`HighGapToLong/summary.json`（实测 32K macro 70.1、128K 67.4，score_sum 24.58，36 行 0 提升） | **已被实验证伪**（用户/文档语："HighGapToLong 已证伪"，`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md` L63）；有效成分被改写为"过渡段预算位置"问题 |
| C8 | FFN-LoRA + 真实 16K 监督能在 OLMo-2-1B 上恢复能力；Cosh 相对 Exponential/Hybrid 有优势 | [未执行]（仅 CPU 准备+单测；无一条 GPU 数字） | `experiments/evq_recovery/README.md` | 不得预先承诺（README L37："首轮可以直接发现 Cosh 参考点可被替代"）；不得把 loss 降低当能力恢复（make_plan `interpretation` 字段） |
| C9 | 端点不变量 I1（高频 bank 恒等）/I2（尾部精确 ÷S） | [部分证据]（各有 3–4 个独立面板失败实验支持；36 行开发面板、历史输入） | `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` §1 | 当前被当作硬约束使用；未见后续纠正 |
| C10 | P2 表长端 +3.54pp vs MrPro（128K），短端大幅付费 | [已验证-开发面板]（`full_lag_p2_confirmation.json`；macro 0.7292/0.8167 实测） | 同文件 | 明示为开发信号非确认；32 新输入长行确认件在冻结队列（0447→0451） |

---

## 4. 失败机制清单

**F1 · 代理指标 ≠ 能力结果（本包最高复发风险）。**
线性/加权算子族（cos 碰撞核→U→QK-Gram 加权 U_H→最差方向 Rayleigh→bias-KL）逐级加码，Smooth-vs-MrPro 反例在每一级都不翻转（详见 §5.4 三张表）。文档复盘语："U and unweighted local distortion are not a sufficient selector"；"even its Q/K-weighted version does not rank all observed tables correctly"（EVQ_NONLOCAL L97–98）。用户早在 09-09 就点名该循环："为什么每次都是 理论看似完整-实验失败循环"（`USER_PROMPT_TRANSCRIPT_20260909.md` T2-P31，2026-09-09T00:56:04Z）。**复发警告：任何新"判据"若在开发面板上给出与已知任务排序一致的顺序才允许登记；给出与已知排序矛盾的"更优"表时，先当反例证据处理，不当候选推广。**

**F2 · 阈值/谱约定混用。**
早先草稿把 1e-10 作用于 Gram 特征值，而设计矩阵奇异值 1e-10 对应 Gram 谱 1e-20；其"large regularized amplification numbers"被弃用，改 direct SVD（DERIVATION L179–182）。正则化最差方向表（0.1→1e-8 放大 5–15 个数量级）保留为诊断、不作选择器。**复发警告：判据的数值分辨率声明必须绑定到具体矩阵（设计 vs Gram）。**

**F3 · 局部可加性幻觉。**
逐 pair/逐槽有限差分不能相加成整表改动（槽共同经过 softmax 与后续网络；局部 Jacobian 不能预测完整改动）——用户直接否决："仍然是梦话，撤掉"（`raw_thread-core.txt` 行 4169）。三张审计表都是"整表联合特征"计算，正是对该否决的执行。

**F4 · EVQ source 方向反转失败。**
HighGapToLong 把 EVQ 字面操作做出：从高频抽 0.216 log 单位补给长程间隔 → 两端全输（32K −17.1 / 128K −10.8，vt@32K 直接 0.0）。P2 实际只从高频取 0.0008（少 280 倍），赢的机制被重读为"桥的渐进性"。**复发警告：把'表征条件数好'当冻结模型的目标函数是本项目一个月的核心失败模式。**

**F5 · 无卡模式的工程失败（均已修复并留痕）。**
①2GiB 内存下双进程分词 OOM→单进程顺序（transcript 行 7870）；②聊天模板 dict 误当 token 列表→评测样本全被过滤，开卡前发现（行 1894）；③下载截断→Range 断点+MD5 核验（evq_recovery README L84）；④RULER upstream 多 key 行不满名义长度→按实际 prompt+decode 长度归桶（`prepare_ruler.py` 行 42–49）；⑤多答案 RULER 若按 exact 计分会把"命中其一"冒充全对→`official_recall`+`exact=None`（`evaluate.py` 行 40–46，`test_recovery.py::test_multiple_required_ruler_items_are_not_alternative_exact_answers`）。

**F6 · 命名/范围纪律导致的失败前科。**
"不把仍位于 Qwen 原生 32K 内的结果叫作外推"（README L7）；"Cosh 最好"至多指实际受测范围与预算（README L37）；τ=2 胜匹配对照不证明 Cosh 族最优。QASPER 16–32K 桶只有开发 8 题/测试 2 题——若用它做强外推结论即统计性失败（README L55）。

---

## 5. 频率表/方法定义清单

### 5.1 Qwen2.5-3B 非几何面板（`results/nongeometric_screen_20260909/`）

| 名称 | 构造规则 | 32K / 128K 开发得分（macro %，6 任务×6 行） | 出处 |
|---|---|---|---|
| Native | 原生几何表，gain 1 | （超出窗口，不参与长端） | `reference_tables.json`（sha 138c99b1…） |
| MrPro（MrRoPE 部署表） | 过渡均布 N=17，gain 1.138629 | **87.22 / 78.13**（baseline score_sum 29.2167） | `runtime/reference_tables`; 各 summary baseline 字段 |
| MrProBM | MrPro 同 gain 的 BM 变体 | 面板记录在 `done/001–002` | 同上 |
| PI | native/4 全表压缩 | 长端崩（I1/I2 证据组） | UNIFIED §1 |
| MrUni | MrPro 平台段全 ÷4 | 32K 64.6 | UNIFIED §1 |
| Smooth_MrBudget | 槽 24–39 内非负最小粗糙度、固定累积预算、KKT 证书（`smooth_budget.py::construct(tables,23,40)`），其余=MrPro | 87.2 / **68.3**（反例） | `results/Smooth_MrBudget/summary.json`（0.8722/0.6833 实测） |
| FullLagP2（P2） | Qwen1.5B 历史 log-p2 规则整表冻结迁移，gain .074 | 72.9 / 81.7 | `full_lag_p2_transfer.json`; summary 实测 0.7292/0.8167 |
| LongBridgeSlower/Faster | MrPro 上周期∈[32K,128K] 的槽各 ∓/± 1/131072 | 80.6/80.1（LBS）；LBF 0.8722/0.7396 | `weighted_subspace` fs 定义; summary 实测 |
| HighGapToLong | EVQ 字面：高频抽 0.216 log 单位给长程 | 70.1 / 67.4，36 行 0 提升 | `HighGapToLong/summary.json` 实测 |
| E1_s28_less | 单槽 28 回原生 | 87.2 / 83.3 | UNIFIED §3 |
| step_30/32/34/36 | Native 前缀 0..29 + PI 从槽 30 起 | 见 `source_subspace_transport_audit.json`（CPU 构造，非排队候选） | DERIVATION L160–177 |

### 5.2 EVQ 恢复五臂（OLMo-2-0425-1B，K=64 对；`experiments/evq_recovery/tables.py`）

| 臂 | 构造 | 参数 | 得分 |
|---|---|---|---|
| Native | 原始几何表 | — | 未测 |
| Cosh | midpoint 分位数 `1−arcsinh((1−u)sinh τ)/τ`，归一化+锚定 Native 两端点，gain 1 | τ=2 | 未测（README：不称最优 τ） |
| Exponential | `−log1p(−u·(−expm1(−λ)))/λ` 匹配 Cosh 的 RMS 形变 | λ≈1.47547 | 未测 |
| Hybrid | 前 16 高频 pair 保 Native + 低频 Cosh 重分配（r16 历史对照） | 局部 τ≈2.83714 | 未测 |
| YaRN | 官方算子（仓库 parity 测试过的 `official_yarn_on_inv_freq`），scale=4 | gain 1.138629；范围/幅度不同，纯参照 | 未测 |
| 共同约束 | 三非几何臂 RMS 归一化形变 ≈0.129239；四臂端点=Native、gain=1；Hybrid 保持高频一半时达不到该形变量→固定为 r16 历史对照 | | |

### 5.3 QK 加权 Gram 排序失败的三张具体表（本 digest 核心数字）

**(a) 未加权远程 U + 局部失真**（`source_subspace_transport_audit.json`；source=整数滞后 0..32767，target=32768..131071，design SVD 相对阈 1e-10，84 保留/44 丢弃，Native 未解能量 4.9246e-21）：

| 表 | Remote U | C26 | C32768 |
|---|---:|---:|---:|
| PI | 4.4282e-21 | 23.9139 | 90.7246 |
| FullLagP2 | 7.6285e-19 | 8.3378e-4 | 37.1821 |
| BM | 1.6802e-3 | 6.8578e-4 | 42.9997 |
| **Smooth** | 4.9435e-2 | 1.5918e-4 | **35.3793** |
| **MrPro** | 2.3593e-1 | 2.4099e-4 | **42.5180** |
| LongBridgeFaster | 5.2524e-1 | 2.3850e-4 | 42.7174 |
| Native(0..29)+PI(30) | 1.5900e-20 | 8.0842e-4 | 30.7749 |

排序失败 1：PI/P2 的 U 最优但任务端（PI=MrUni 32K 64.6；P2 短端 72.9）并不最优；Smooth 双指标均优于 MrPro，但 128K 实测 68.3 < 78.1。[已验证-开发面板 n=36]

**(b) QK 加权局部算子失真**（DERIVATION L242–247，因果滞后窗均值加权成本）：

| 表 | 26 | 256 | 2048 | 32768 |
|---|---:|---:|---:|---:|
| MrPro | 3.5e-5 | 4.192e-3 | 0.204290 | 6.043652 |
| P2 | 8.9e-5 | 9.705e-3 | 0.612880 | 5.394220 |
| Smooth | 2.4e-5 | 2.630e-3 | 0.174953 | **5.263579** |
| Native0..29+PI30 | 8.5e-5 | 9.057e-3 | 0.616595 | 4.795435 |

Smooth 仍全面优于 MrPro；"merely replacing uniform phase weights by projection-matrix norms does not close the explanatory gap"。另：step 例在 26/256/2048 尺度上不如 P2——不能从"32768 更优"升格为通用更优。

**(c) QK 加权未解远程响应 U_H**（`weighted_source_subspace_audit.json`，方法内 weighted_mean；三阈值下升序均为 PI < P2 < BM < Smooth < Slower < MrPro < Faster < Native）：

| 相对奇异值阈 | 丢弃方向数 | MrPro U_H | Smooth U_H | P2 U_H | smooth<mr 层数 |
|---|---:|---:|---:|---:|---:|
| 1e-6 | 47 | 1.12494981 | 0.749846431 | 1.8321e-12 | 36/36 |
| 1e-8 | 45 | 0.427494108 | 0.172023636 | 6.8833e-17 | 36/36 |
| 1e-10 | 44 | 0.359961033 | 0.144080138 | 3.9362e-18 | 36/36 |

（unweighted trace 同序：如 1e-10 处 MrPro 0.235928 / Smooth 0.0494352 / Native 14.1944。）
辅证：`remote_PI_distortion_audit.json` QK 加权 PI 失真 MrPro 6.308 vs Smooth 5.895；`regularized_worst_direction_audit.json` ridge .1 处 MrPro 7.33 vs Smooth 6.37（PI 恒 ~1.30–1.34、P2 1.31→8.67 随 ridge 收紧而恶化）。

**结论（本包最重要的一句）**：加权版关闭了"用学习的 Q/K 算子给源弱能量判据补权重"这条修复路线（DERIVATION L276–277），但**没有**否定联合关系机制本身，也没有给出能复现任务排序的替代判据；下一步要求判据能区分"有用计算被改动"与"无方向算子距离"（L251–255），当前候选解释是 bank/arc 二分法 + 证据距离（UNIFIED §2/§4，[假设]→[部分证据]级）。

### 5.4 softmax 混频审计常数表（`softmax_harmonic_audit.json`）

| 项 | 值 | 含义 |
|---|---|---|
| slow_difference (.51,.5103) | 各自 664.94/665.33 圈、差相 2.4576 rad、首半部质量 0.6096 | 两快频差分成慢包络 |
| rapid_difference (.51,.53) | 差相 163.84 rad、首半部 0.5003 | 对照：快差→均匀 |
| Bessel 截断 (|n|≤9) 误差 | 2.957e-9 / 3.120e-9 < 界 3.135e-9 | 展开+尾部界成立 |
| Qwen 28−2·29+30 | 角频 8.9395e-5，周期 70285.94 token，32768 处相位 2.9293 rad | 原生表确有慢混频 |
| 混频系数比 2I₁(a)²I₂(a)/I₀(a)³ | a=.3→4.878e-4；a=1→0.04273；a=2→0.29430 | 振幅依赖极强，"长周期存在"不足以选规则 |

---

## 6. 用户指令与纠正（原文引用）

1. （`docs/research/USER_PROMPT_TRANSCRIPT_20260909.md` T3-P01，2026-09-07T13:49:40.930Z）"不要再进入'猜曲线—跑失败—换猜想'的循环……当前不启动新实验。先继续论文、代码和已有数据分析；后续最多 3 个实验，不能隐藏多候选扫描或重置原有预算。EVQ SSH：[隐去]……最后检查为关机；完整 Q/K/V 缓存在服务器，访问需恢复。"
2. （同上 T2-P31，2026-09-09T00:56:04.461Z）"为什么每次都是 理论看似完整- 实验失败循环， 我们近1个月以来，相比于evq几乎没有任何突破"
3. （同上 T2-P32，2026-09-09T00:58:45.890Z）"为什么没法从理论层面解决问题？ 我们有pro模型，有那么多失败，有别人的成功，这些不足以作为先验吗？"——本日三份理论推导文档是对该指令的直接响应。
4. （`raw_thread-core.txt` 行 4169，2026-09-08T09:47:55Z）"无卡模式你只能处理代码以及下载事宜……**上一条'测每个 pair 的 benefit 再设计表'仍然是梦话，撤掉。**有限频率改动在长距离上会绕相多圈，局部 Jacobian 已被实际数据证明无法预测完整改动；逐槽有限消融又不能相加成整张表，因为槽间会共同经过 softmax 和后续网络。"
5. （`USER_PROMPT_TRANSCRIPT_20260909.md` 行 43）"不要因为我们已有 EVQ / frequency-allocation 结果就强行围绕它展开；只有它真正增强新主线时才复用。"（行 90）"明确指出哪些旧 EVQ / RoPE 实验值得保留，哪些应该彻底下沉附录或删除"。
6. （同上 T3-P05，2026-09-07T14:00:32.804Z）"你可以同时理论分析，并利用GPU， CPU， 让GPU挂着高ROI实验，不然会很浪费。"
7. （`raw_thread-core.txt` 行 2983）"你先准备好实验代码，审查好，规划好，然后你自己操作关机，然后有卡模式自己开机……你全权自主完成这个任务，不要中断，不要询问"（授权上下文，含自主开关卡）。
8. （`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md` L3–5 转述的作者指令）"derive a better allocation rule rather than continue scoring hand-built tables"——三个审计脚本均按"不生成候选表、不入队"实现。
9. 铁律复述（AGENTS.md/项目纪律，README/make_plan 落实为代码字段）：loss↓≠能力恢复；有限训练终点≠LoRA 通用极限；不造教师监督；短 replay 不冒称从未被查看；CPU 准备不能证明 GPU 执行。

---

## 7. EVQ 相关代码"能算/不能算"总表（代码自我限制汇编）

### 能算（本机 CPU，已验证输入齐备）
- **三个审计脚本全部本地可复算**：`rope_softmax_harmonic_audit.py` 与 `rope_weighted_subspace_audit.py` 所需输入（`reference_tables.json`、`ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json`、`qk_operator_gram.npz`、`smooth_budget.construct`）全部在本地存在，本次会话已验证 import 与构造成功（Smooth 64 维数组本地重建）。[已验证]
- Gram 本身只需 checkpoint safetensors **权重切片**（`safe_open(..., device='cpu')`），无模型前向——对任意新 checkpoint 可在 CPU 重做（本地无 Qwen2.5-3B 权重缓存，需从服务器拷 q/k 投影分片）。[部分证据-切片拷贝路径未实测]
- EVQ 恢复包 CPU 面：数据下载核验、全套准备、五臂表构造（含形变匹配二分）、12 个 tiny-model 单测（密集 CE 数值/梯度一致、FFN 梯度存在、adapter 保存重载、超窗 KV-cache 静态 RoPE、续训顺序、错位重叠检测、多答案语义、LM 读出=训练 CE）、`validate.py` 哈希读回、`compare.py` 对已有 generations 的配对分析、`launch.py` dry-run 命令展开。[已验证-代码/单测层面；单测本轮未重跑 pytest，py_compile 通过]
- 混频/EVQ 核的纯数学验算（Bessel、Ci、界检查）。[已验证]

### 不能算（代码/README 明示）
- **不能加载真实模型**：`runtime.load` 无 CUDA 直接 `RuntimeError('GPU unavailable; preparation does not load full model weights')`（runtime.py 行 68）。
- **不能给任何能力/显存/吞吐结论**：validate.py 输出固定 `GPU_training_executed: False, GPU_memory_and_speed_verified: False`，limitation 字段"Actual GPU smoke, baseline capability and trained outcomes remain unmeasured"；README"CPU检查覆盖……完整模型的GPU内存、吞吐和能力仍必须在开卡后测量"、make_plan `hardware_note`"CPU preparation cannot certify GPU execution"。
- **smoke 不是结果**：README L32"真实16K训练smoke只验证更新、内存和耗时，丢弃其adapter；不能作为能力结果"。
- **不能把共同地板归因于某网格**：README L7 保留 Native + YaRN 同配方对照。
- **不能证明语义去重**：QASPER 32-word exact-passage 重叠"不等于语义去重证明"（README L46）；screen scope 字段"not semantic paraphrase detection"（prepare.py 行 277）。
- **不能合并指标**：compare.py 禁跨 QA/retrieval/LM 池化，bootstrap 区间"not training-seed uncertainty or universal model guarantees"。
- **Gram 不能**：说激活分布、说 softmax KL、说任务损失、说语义信息损失（脚本 scope 字段与 DERIVATION L236–238）。
- **16–32K QASPER 桶不能**承担强自然任务外推结论（开发 8/测试 2 题）。
- **EVQ 局部核不能**：全局"只优化脊项"的分配（在近似最不被处被使用）；对角层被 log 近似替代；锐利过渡定罪（非局部核证明其收费更低）。
- **混频审计不能**：拟合参数、生成候选表、作为模型证据（docstring 行 2–4）。

### 无 GPU 可推进的检验清单（建议，按依赖排序）
1. 本地重跑 3 个审计脚本做确定性回归（输出应与 06:24/06:46/06:55 收据一致；重写同路径，需先备份或改输出路径）。
2. 把 Gram 判据迁移到 **OLMo-2-0425-1B 的 Q/K 切片**（CPU）：检验"加权不救排序"是否跨模型复现；只需从服务器拷投影权重分片，不需开卡。若复现，反例不再是 Qwen3B 面板噪声。[当前为开放]
3. 对 Smooth/MrPro 反例做**判据级**新构造：bank/arc 二分法的 CPU 化（逐槽原生圈数 r_j、地平线 D_j、洞比率 ρ_g——UNIFIED §1/§2 已定义，纯算术可算），并检查其能否在 36 行开发面板上复现已知排序（这是对"新判据必须先过排序关"协议的 CPU 版执行）。
4. `experiments/evq_recovery/test_recovery.py` 全套 pytest（tiny 模型，纯 CPU）；`tables.py` 五臂表在 OLMo 配置上的重建与 sha 对照（make_plan 干跑到临时 root）。
5. 服务器无卡模式（2GiB 内存限制）：只允许代码/下载/小模型检查（用户指令 4）；`acquire.py`/`prepare.py`/`validate.py` 可在开卡前完成——这正是 README"开卡前准备"的定位。
6. 混频机制的**可测化设计**（DERIVATION/SOFTMAX 文档指出的"判别测量"：保留 native 条件内容系数、interaction-preserving vs interaction-breaking 等强度干预）——设计可在 CPU 写死，执行需 GPU/缓存激活。

## 8. 未决问题

1. **能复现任务排序的判据是什么？** 加权 Gram 已被关闭，剩余路线（UNIFIED §8.1 J_r 分解、耦合二次型分配）都停在"成本项/交互选择仍未导出"（DERIVATION L204–212）。[假设]
2. Smooth@128K 68.3 的失败行级归因是否稳定？需要 holdout 批（队列 0450/0451）把 36 行开发信号升为确认。[未执行-GPU]
3. EVQ 恢复：16K 训练能否恢复 Native 地板之上的能力？Cosh 是否会被 Exponential/Hybrid 替代？全参数对照的显存是否可行？——全部待开卡，代码已就绪。[未执行]
4. `qk_operator_gram_with_means.npz`（06:55）里新加的 matched-content mean / content correlation advantage 三个数组**尚未被任何审计消费**——是否用于"内容均值≠二阶矩"的下一步判据？未见后续脚本。[开放]
5. `bias_harmonic_head27_8_audit.json` 与 `bias_prior_local_KL_all_heads.json`（06:58/07:00）在仓库内**没有对应生成脚本**（grep 无果）——来源为一次性内联计算，provenance 需在复算时补脚本，否则违反可复现纪律。[部分证据-收据存在、代码缺失]
6. 服务器 `/root/autodl-tmp/evq_recovery_20260910` 的 `PREPARATION.json`、`tables.json`、旧失败日志本地无副本；开卡前应在无卡模式读回核验一次。[开放]
7. 0446–0451 冻结队列只在文档中列出，本地 `queue/` 目录未见对应 json——恢复 GPU 前需确认服务器队列文件仍完整。[部分证据]
8. EVQ 恢复 README 声明 Hybrid 形变量不足"因此在模型输出产生前固定为历史 r16 对照"——该匹配的精确数组需与服务器 `tables.json` 对 sha 后才可引用。[开放]
