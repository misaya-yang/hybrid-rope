# T6 — codex_tools_a.txt 凭据层 digest（线程前半段"实际执行过的命令 + 输出"）

- **输入文件**：`/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/mine/extracts/codex_tools_a.txt`，共 **12808 行**（`wc -l` 实测）。
- **本 digest 的行号约定**：
  - `E:NNNN` = 该 extract 文件自身的行号（用 `Read`/`grep -n` 直接读到）。
  - `S:NNNN` = 文件内 `===== [CMD] line NNNN =====` 标记里的数字，即**源线程**的行号。两者不同，都需保留。
- **角色边界**：本任务只做挖掘与整理，不做推导。以下所有内容均为"线程做过什么"的证据，不构成对当前推导的指令。
- **红线遵守声明**：本 digest 不把任何静态几何代理量（Σcos 首根、碰撞能、覆盖率、平滑度、有效秩、Gram、能量）当作 F 的分项或选择子；不把 Σm 当守恒量；凡涉及"守恒"均点名坐标；原报告中的 VICTORY / 已闭合 / 已证明 类结论一律降级标注。

---

## 0. 覆盖与读取说明

| 区间（extract 行） | 内容 | 状态 |
|---|---|---|
| 1 – 5477 | BM 协议、C1 ψ 投影、gap_capped、layer_screen、Qwen3B/7B 迁移、RULER/LongBench 面板 | 已读（前序会话） |
| 5478 – 6224 | chunked_mlp、natural_nll、BM 五任务 QA | 已读 |
| 6224 – 8224 | NLL 汇总、Qwen7B OOM 与 run_screen_02、cross_cache stage_run_01 失败 | 已读 |
| 8224 – 9224 | bias_position、native_windows 及其测试 | 已读 |
| 9224 – 10324 | OLMo BM 结果记录、extra RULER、Qwen7B NO_LONG_GAIN 筛 | 已读 |
| 10324 – 11424 | paper-2027 正文、151M 三 seed 记录、KLD v2 审计、rotary_budget 建表 | 已读 |
| **11424 – 12808（EOF）** | theory_check 几何核对、prepare_data 数据重建、train_budget、launch_probe、summarize_budget、native_sector carrier、gap_capped | **本次补读完成** |

**跳过**：纯 `cat`/`sed`/`ls`/`rg --files`/`git status` 类文件浏览命令；`~/.codex` 下的路径只读不写。

---

## 1. 本线程写/改的脚本与产物（"跑过什么、写到哪里"）

### 1.1 旋转频率表构造

| 文件 | 出处 | 性质 |
|---|---|---|
| `scripts/lib/rope/boundary_matched.py` | E:5870–5922 | BM 闭式实现；含 `formula='m_q=q*(q+1)*(3*N+2-2*q)/(N*(N+1)*(N+2))'` 自描述 |
| `scripts/lib/rope/gap_capped.py` + `tests/test_gap_capped.py` | E:2024–2054 | 参数自由全局表；`python3 -m pytest tests/test_gap_capped.py` → **2 passed in 1.00s**（E:2035） |
| `scripts/analysis/build_boundary_matched_mrpro.py` | E:1389 | BM 生成 + 独立求解核对 |
| `scripts/analysis/project_mrpro_transition.py` | E:1646 | C1 ψ 投影计算 |
| `experiments/rotary_budget/build_tables.py` | E:11296–11340 | 六臂表构造（G32/E32/G16/E16/P16/U16） |
| `experiments/rotary_budget/theory_check.py` | E:11425–11471 | 有限网格 QR 几何核对 |
| `experiments/rotary_budget/eval_inputs.py` | E:11472–11489 | 同目标输入构造 |
| `experiments/rotary_budget/tests/test_rotary_budget.py` | E:11490–11507（截断） | 端点/极限/初值一致性 |
| `experiments/rotary_budget/{train_budget.py, prepare_data.py, launch_probe.py, summarize_budget.py, REPORT.md}` | E:11971–12045, 11783–11868, 12565–12586, 12619–12678, 11515–11568 | 训练/数据/探针/汇总 |
| `experiments/kld_v2/audit.py` + `RESEARCH_PLAN.md` | E:12048–12109, 12512–12562 | KLD 代数独立核对 |

### 1.2 评测与迁移

`scripts/experiments/olmo_fast_screen/` 下的 `natural_nll.py`、`prepare_ruler.py`（E:1475–1484+）、`layer_screen.py`、`chunked_mlp.py`、`bias_position.py`、`native_windows.py`、`cross_cache.py`；`scripts/analysis/summarize_nll_screen.py`（E:6406–6468）、`summarize_natural_screen`（E:6363 调用）。

### 1.3 论文侧（paper-2027）

`paper-2027/sections/budget_abstract.tex`、`budget_intro.tex`（E:11897–11958）、`budget_related.tex`（E:12428–12460）、`07_reproducibility.tex`（E:12484–12491）；引用条目 `ji2025mha2mla` 追加到 `paper-2027/refs/references.bib`（E:12463–12470）。

---

## 2. 可作 F 泛函零件的公式 / 定义 / 约束（kkt_parts）

### 2.1 BM（Boundary-Matched）闭式分配 —— 本线程的核心解析件

**出处**：E:5870–5922（代码）；E:1371–1379（协议文档原文）；E:1397–1400（理论核对段）。

- 逐槽指数增量：

  ε_i = 6·i·(N+1−i) / [N(N+1)(N+2)]，i = 1..N

- 累计压缩指数（闭式）：

  **m_q = q(q+1)(3N+2−2q) / [N(N+1)(N+2)]**

- 频率映射：ω′_j = ω_j · S^(−m_{j−l})，中段 q = j−l 取 0..N。
- 高频槽（j ≤ l）与低频槽（j ≥ h）**逐位复制** MrPro 值；gain、base、support 端点、单表 RoPE 算子均与 MrPro 同；无新窗口、无长度映射、无 gain 选择、无可调曲率（E:1379–1380）。
- 中段边界由**圈数**定义（native 网格上 >32 圈 与 <1 圈）：

  `turns = w * reference_length / (2π)`；`fast = {j: turns_j > 32}`；`slow = {j: turns_j < 1}`；`low, high = fast[-1], slow[0]`；`n = high − low`（E:5880–5886）。

- gain：**gain = 1 + 0.1·ln(S)**（E:5907）。OLMo 实例值 **1.138629436111989**（E:1384、E:1698）。
- 网格校验：`expected = 1/(base**(arange(0,dim,2)/dim))`，rtol 3e-7（E:5910 附近）。
- 证据等级：**[已验证]** 作为构造闭式与实现（有代码 + 独立精确求解核对）；**[假设]** 作为"最优语言模型频率表"（文档 E:1409、E:1418 自述 R 是几何目标、不是任务损失；最小粗糙度闭式正确不自动证明其最佳）。

**约束/边界条件（可作为 F 的可行域）**：
- Σ ε = 1（E:1406 表格；两端固定 ⇒ 中段增量总和为 1）。**注意**：红线明确 Σm 不是守恒量 —— 文档 E:1413–1414 自己写明"Σ ε 相同不表示所有槽的 exponent 总和相同"，OLMo 的 exponent 总和**增加约 2.833333**。所以 Σε=1 是**设计约束**，不是能力预算守恒。
- 严格中段有 m_q^BM − m_q^Mr = 2q(q+1)(N−q)/[N(N+1)(N+2)] > 0（E:1412）：**每一中频槽都增加压缩**（方向性代价）。
- 一维离散 Laplacian 粗糙度 R = εᵀLε，L 在零端点、正定；在 Σε=1 下 Lε 为常数向量给出**唯一**极小值；解 Lz=1 再归一化即得上述闭式（E:1397–1399）。任意非零可行扰动 v（Σv=0）满足 R(ε+v) − R(ε) = vᵀLv > 0（E:1400）。
  - **⚠ 红线提示**：文档把 R 明确写成"**几何目标，不是任务损失**"（E:1409）。因此 R 只能作为**约束/正则**来源，不得直接当 F 的分项或候选选择子。
- Y 方向上的"守恒必须点名坐标"实例：C1 文档（E:1706–1708）明确区分两种预算事实——两端 exponent 之差仍为 1（累计 radix 乘积仍为 4）；对称槽位 ψ 和为 0（理想实数 exponent 总和不变）。FP32 实现后 exponent 总和误差 **−1.85e−8（Qwen）/ −2.55e−8（OLMo）**。原文自己声明"**不能被解释成能力预算守恒**"。**[已验证-代码+文档自陈]**

### 2.2 MrPro / YaRN 中段参数化（本线程出现的形式）

- MrPro：ν_j^M = ω_j S^(−m_q)，m_q = q(q+1)/[N(N+1)]（二次累计压缩）—— 见 E:1641 附近的对照语境。证据等级 **[部分证据]**（本 extract 未给出该式的独立推导，只在 C1 文档与 unify 起点文档中以既成事实出现）。
- 标准 YaRN（维度线性混合）：ν_j^Y = ω_j(1−t+t/S)，t=q/N —— 本 extract 未见完整写式，见权威文档 `STARTING_POINT_YARN_VS_MRPRO.md` §1。**[叙事-未验证于本 extract]**

### 2.3 C1 ψ 投影（被否决的候选，但给出一个可复用的最小二乘骨架）

**出处**：E:1655–1683。

- x_j = clip((j−l)/(h−l), 0, 1)；**ψ_j = x_j(1−x_j)(2x_j−1)**
- "真实 exponent"定义：m_j = −log(ω_j^deployed / ω_j^Native) / log S
- 唯一无截距、无约束、等权最小二乘系数：

  **a\* = Σ_{j=l+1}^{h−1} ψ_j (m_j^P2 − m_j^Mr) / Σ_{j=l+1}^{h−1} ψ_j²  = 2.1477690111869907**

- m_j^C1 = m_j^Mr + a\*ψ_j；ω_j^C1 = ω_j^Mr · S^(−a\*ψ_j)
- 唯一性来自"非零基向量的严格凸一维二次目标"，**与任务能力最优性无关**（E:1683 自陈）。**[已验证-数学]** 作为投影最优；**[已否决]** 作为候选（见 §4）。

### 2.4 Native-sector（去载波）载波裁剪闭式

**出处**：E:1910–1946。

- 保持 B = {j : ω_j L0 ≤ π/2}（Qwen 上为 47..63），S = 4，要求所有因果距离 0 ≤ δ ≤ S·L0 满足
  **0 ≤ ((ω_j − c)/S)·δ ≤ ω_j·L0**。
- 在 c ≥ 0 下该条件**当且仅当 c ≤ min_B ω_j**；原背景目标 A − 2Nc + Dc²（D>0）加此区间后唯一极小值为 **clip(N/D, 0, min_B ω)**。
- 用已冻结 Native 均值（不重算背景、不读答案）得 **c = min_B ω = 1.2409377632138785e−6**（E:1940）。这是**解析投影，不是从多个效果值中选出的裁剪阈值**。**[已验证-数学]**
- 新表在 B 内为 (ω_j − c)/4；最末槽为 0；组内频差仍为 Native 的 /4；相对同底座最大新增相位 **32K 为 0.010166 rad，128K 为 0.040663 rad**（E:1944）。**[已验证-数值]**

### 2.5 Qwen2 Q/K 仿射 bias 分解（augmented Flash 头的代数）

**出处**：本线程 `bias_position.py` + 其测试（E:8171–8330 区间）。

- s_H = a²/√D · [ qᵀR_B(d)k + b·qᵀ(R_M(d) − R_B(d))·bk ]
- 实现：Q_aug/K_aug 扩到 **192 宽**，V 补齐；`phases()` / `rotate_bias()` / `augment()`。
- 测试：复数旋转一致性 rtol/atol **1e−10**，含 GQA 与 **131072** 位置。**[已验证-数值精度]**

### 2.6 KLD v2 的两槽非扩张递推（可作 F 的约束骨架）

**出处**：E:12056–12109、E:12691–12701、E:10324–11420 区间的 KLD v2 审计文档。

- 递推（`recurrence`）：s ← d_s⊙s；read = kᵀs；s ← s + β·outer(k, v − read)；h ← d_h⊙h + β·outer(k, read − kᵀh)。z 变体：s ← s + β·outer(k, v − (1−z)·read)。
- 关键等价：**H_t = ∂_z S_t(z)|₀**（导数即额外状态）。
- chunk 形式（`chunk`）：A = I + L；A·E^S = V − X·S_in；A·E^H = X(S_in − H_in) + L·E^S；L[t,s] = β_s·Σ k_t ⊙ (∏_{s+1..t} decay) ⊙ k_s。
- 有限差分收敛：eps = 0.01/0.001/0.0001 → ‖(x−s)/eps − h‖ = **0.0051362230 / 0.0005141273 / 5.141777212e−05**（一阶线性收敛）。
- 线性容量：capacity = tr(pinv(C Cᵀ + σ) C Cᵀ)，断言 0 ≤ capacity ≤ 4；实测 **3.966585557648783**（首版）/ **3.9304255356400377**（加入非零初值后）。
- 随机干扰表：n=128 → S = **0.3664377159220373**，H = **0.3693230522678801**；(127/128)^n 与 n/128·(127/128)^(n−1)；n=512 → S = **0.018030205213609263**，H = **0.07268870133360585**。
- 状态预算：**16·128·128·4 = 1048576 bytes/序列（fp32）**。
- 全部误差量级：derivative_max_abs **1.3877787807814457e−16**；chunk_s_max_abs **4.614364446098307e−16**；chunk_h_max_abs **1.942890293094024e−16**；非零初值版 nonzero_chunk_s **3.391384395534658e−16**、nonzero_chunk_h **3.0531133177191805e−16**、nonzero_derivative **1.6653345369377348e−16**（断言 < 1e−12 全过）。
- **限制声明（原文自带）**：`"limitation": "No trained checkpoint, full-network generation, or GPU-kernel performance tested."`；状态字符串 `CPU_ALGEBRA_ONLY`；docstring "These do not establish trained-model utility."。**[已验证-数学] / [未验证-任务效用]**

### 2.7 Qwen3.5 GatedDeltaNet 官方递推（作为外部参照，本 extract 仅读代码）

**出处**：E:12330–12399（读 `transformers/models/qwen3_5/modeling_qwen3_5.py`）。

- `torch_recurrent_gated_delta_rule`：q = q/√d_k；D = exp(g)；`last_recurrent_state *= g_t`；`kv_mem = Σ_k state·k`；`delta = (v − kv_mem)·β`；`state += k ⊗ delta`；`out = Σ_k state·q`。
- β = sigmoid(b)；g = −exp(A_log)·softplus(a + dt_bias)；可选 q/k L2 归一化（eps 1e−6）。
- 版本：本机 Transformers **5.15.1**（E:12329）；官方 config 24 层、18 线性 / 6 全注意力、k/v 头各 16、宽 128（E:12537–12541）。最后一个线性层零基索引 **22**。**[已验证-代码阅读]**

### 2.8 Rotary-budget（EVQ-Cosh）六臂设计 —— 与 KKT 问题直接相关的另一条线

**出处**：E:11296–11415（`build_tables.py`、`protocol.yaml`）、E:11425–11471（theory_check）。

- 臂：ARMS = ('G32','E32','G16','E16','P16','U16')；**TAU = √2**；**BASE = 500000.0**。
- EVQ 形状：
  ```
  u = (arange(k)+.5)/k
  q = 1 − arcsinh((1−u)·sinh(tau))/tau
  z = (q − q[0])/(q[-1] − q[0])
  return exp(−(31/32)·log(BASE)·z)
  ```
- U16 = full[::2] 映到前 16 个固定对；P16 = full[:16]；非活跃对为**零频率**（E:11530–11532）。
- protocol.yaml：expected_parameters **151898880**；pair_layout `split_half_fixed_pairs_j_j_plus_32`；**7629 updates**；512 docs；**8193-token anchors**；remote_replaced 用前 3584/4096；main_length 4096；**practical_margin_nat 0.02**。
- 预注册统计量：D16、D32、J、H、P、U、R_recovered（E:12661–12663）。
- 训练配置常量：CONFIG = vocab 50304 / hidden 768 / 12 层 / 12 头 / head_dim 64 / intermediate 3072 / max_position 8192；`count != 151898880` 直接抛错（E:11985、E:12032）。
- **theory_check 的 32 组有限网格几何核对**（E:11425–11470）：构造 root·[1, t] 与 root·[cos(x t), t·sinc(x t/π)] 的 QR；r0 = 2·(K−m+1)（m>0 时）；断言 `tail ≤ eta + 1e−10`、`|Σspectrum − 2K| < 1e−9`、T1 界 `e ≤ x⁴h²/(σ − .25²h)² + 1e−12`。
  - 关键实测（length=2048、prior=causal，**E:12595–12604**）：

    | arm | slow_pairs | r0 | count(λ≥0.01) | eta |
    |---|---:|---:|---:|---:|
    | G32 | 10 | 46 | 31 | 4.986517321429179e−06 |
    | E32 | 7 | 52 | 37 | 8.79227314706136e−07 |
    | G16 | 5 | 24 | 18 | 1.2485622509051058e−06 |
    | E16 | 4 | 26 | 21 | 1.1158946682258811e−06 |
    | P16 | 0 | 32 | 30 | 0.0 |
    | U16 | 5 | 24 | 18 | 4.177193158517813e−06 |
    | G8 | 3 | 12 | 10 | 3.182118600360797e−06 |
    | E8 | 2 | 14 | 12 | 2.413647929317503e−08 |
  - 输出落盘 `experiments/rotary_budget/theory_checks.json`，status = **CPU_GEOMETRY_ONLY**，打印 "32 finite-grid configurations: T1/T2 checked; no neural-model claim"（E:11469–11470）。
  - **⚠ 红线提示**：这是**几何/谱**量（`spectrum`、`tail`、`eta`、阈值计数），原文自我标注"**This is geometry, not LM evidence**"（E:11542–11543）。**不得作为 F 的分项或选择子**；只能作为"这六张表在给定先验下确实不同"的实现证据。

---

## 3. 已验证的数字与事实（verified_numbers，逐条带出处）

### 3.1 BM 相对 MrPro/MrUni/官方 YaRN 的端到端分

- **OLMo 72 条独立复核（16K 六任务等权均分）**：BM **51.32%**，MrPro **2.78%**，MrUni **32.12%**，官方 YaRN **6.94%**；4K BM **81.81%**（E:136）。原文自述这是"**局部能力收益，不是完整 RULER、跨模型或 SOTA 结论**"。
- **静态 S8 @32K 地板**：BM **6.94%**，MrPro **0.69%**；同 32K 用 S4 BM **0%**；S8 频率 + S4 gain = **5.56%**；两个恢复方案**均不晋级**（E:138）。
- **Qwen2.5-3B-Instruct 冒烟（72 次生成，约 29.44 分钟）**：32K BM **91.67%** vs MrPro **87.22%**；128K BM **70.83%** vs MrPro **78.13%**；长端 4 胜 4 负 16 平，均分下降 **7.29 pp**，**未晋级全量**（E:143、E:277、E:287）。
- **1.5B 线**：MrPro 进行到 19/36 条时按用户要求中止，**不作方法结论**（E:143）。
- **Qwen3B 中段频率量级**：BM 在零基槽 24–39 增加压缩，相对 MrPro 频率下降约 **1.5%–31.5%**，最大在槽 34；高/低两端保持（E:227）。
- **128K 退化诊断中的逐例数**（E:265–268）：
  - 原选错记录的 `bizarre-slime` → `neutral-slime` 单点编辑，只有 token **74100** 改变，长度 **130871**，其他 token 与位置完全相同。MrPro 仍正确（**3954314**）；BM 从 9289114 换成另一个错误数字 **4068207**（属于原文 strange-alien 记录）。
  - 正确多键记录始于 token **35173**，原 query 到该处 lag **95697**，BM 相对 MrPro 最大相位差约 **31.25 rad（槽 28）**；VT 起始赋值 token **3986**，lag **127052**，最大约 **41.49 rad**。
  - **结论（原文自带）**：相位差大本身**不是**失败的充分条件（P 仍能满分）；**不能**把最大相位差的槽 28 自动称最敏感槽。
- **Qwen3B 上 BM 优于 MrPro 的行**（E:1466–1471，`run_qwen3_01` 配对比较）：vt_32768_1 0.8→1.0；fwe_32768_1 0.667→1.0；niah_multiquery_131072_3 0.75→1.0；vt_131072_0 0.2→0.8；vt_131072_2 0.8→1.0；fwe_131072_2 0.667→1.0。**[部分证据]**（只列上升行，未列全表）。
- **机制行逐条生成**（E:949–954、E:1030–1037、E:1120–1123、E:1151–1158）：
  - `niah_multikey_2_131072_1`：MrPro/MrPro O = 1.0（输出 `3954314.`，token 序列 [220,18,24,20,19,18,16,19,13,151645]）；MrProBM/MrProBM O = **0.0**（输出 `9289114.`）。cross_table_cached_query 与 O 完全同分、同 token —— 说明**不是缓存/取表问题**。
  - `vt_131072_1`：MrPro 全模式 correct = **1.0**；MrProBM 全模式 correct = **0.2**（6 token 后 EOS，未耗尽 30-token 预算）。另一 VT 行 BM **20%→80%**（E:229）。
  - 该行 4 行 cross-table 诊断：`{keys_bitwise_equal: True, diagonal_matches_original: True, original_matches_saved: True}`（E:1079）。**[已验证-逐 token 级]**

### 3.2 BM 的尾部 NLL（teacher-forced，配对）

**出处**：E:6469–6488，脚本 `scripts/analysis/summarize_nll_screen.py`（E:6406–6468），产物 `docs/research/ROPE_OLMO_BM_NLL_RESULT_20260908.json`。

- 设置：**16 篇冻结文档**，末 **512** 个 next-token NLL，文档配对 bootstrap（`rng = np.random.default_rng(20260908)`，10000 次重采样，取 [.025,.975] 分位）。
- 4096 长度：Native **2.835046201944351**、MrPro **3.210620880126953**、MrProBM **2.9550598710775375**；BM − MrPro = **−0.2555610090494156**，**lower_nll_docs = 16**（16/16 全下降）。
- 校验断言（脚本硬编码）：receipt COMPLETE + raw_sha256 一致；`len(token_nll) == 512`；`|mean(token_nll) − nll| < 1e−5`；三方法 `input_sha256` / `target_ids` 完全相同；重复行报错。**[已验证-带完整性校验]**
- scope 声明（脚本内）：`'16 frozen natural prefixes, final 512 next-token NLL per length. Document-paired exploratory intervals; not full corpus, whole-string generation or long retrieval capability.'`

### 3.3 OLMo BM 五任务自然 QA（LongBench 分桶）

**出处**：E:6363–6386；产物 `docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`。

- **rows_per_arm = 778**（三 run：`run_natural_01`、`run_natural_02`、`run_natural_extra_01`）。
- **extended 桶**（超过原生长度）：MrPro **0.2162430008958048**，BM **0.2544369190250243**，delta **+0.03819391812921949**，配对 bootstrap 95% CI **[0.013160525890820804, 0.06291977669603137]**（不跨 0）。
- **within_native_length 桶**：MrPro **0.4251144144169148**，BM **0.44514270274697254**，delta **+0.020028288330057764**，95% CI **[−0.03561781342696248, 0.07474193004265728]**（跨 0）。
- interval_scope 原文：`'Resample paired rows within each fixed task; exploratory uncertainty within the selected pool, not unseen-model generalization.'` **[验证等级：探索性区间]**

### 3.4 Qwen7B 筛（NO_LONG_GAIN）

- 权重 13G 下载完成，`model-0000{2,4}-of-00004.safetensors` 状态 = `OFFICIAL_SHA_MATCHED`（E:6510–6512）。
- run_screen_01 结果：**0 wins / 3 losses** → 判定 NO_LONG_GAIN（E:7060 附近区间；OOM 后以 `--reuse-generations` 复用 6 条已完成行重跑为 run_screen_02）。
- OOM 原文：`torch.OutOfMemoryError` at Qwen2RMSNorm："Tried to allocate 1.75 GiB. GPU 0 has a total capacity of 31.47 GiB of which 734.62 MiB is free... 24.38 GiB is allocated by PyTorch, and 6.08 GiB is reserved"（E:7060–7068）。修复：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。**[已验证-工程事实]**

### 3.5 KLD v2 审计数字

见 §2.6（E:12109 与 E:12701 两次完整 JSON 输出）。两次运行的差异仅在 `linear_capacity`（3.966585557648783 → 3.9304255356400377）与新增三个 nonzero_* 字段。**[已验证-float64 代数]**

### 3.6 数据重建与哈希（rotary_budget 的 prepare_data）

**出处**：E:11783–11868。

- FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`；shard `000` SHA256 `b1ba7b2c…78e871`，shard `004` SHA256 `33557ddd…d8748b`。
- **TOKENS = 499974144**；tokenizer `tokenizer.json` SHA256 = `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`。
- 重建后的 token 前缀必须等于 `66ee82396750d2c2fe9ab0a678092383a46ad28290983d091bd83895d5f83e60`，否则抛 `ValueError('Rebuilt token prefix differs from original exact-range stream')`。
- 验证集：从 shard004 取 **8193** token 起的文档，去重后 **512** 篇（`{"status": "READY", "documents": 512}`，E:12685）。
- 下载进度实测：prepare 进程 16258 在 03:18–03:25 之间完成两个 parquet（各 2.1G）与 train.partial.npy（954M → 1.3G→…）后产出 manifest.json（13K）（E:11963–11967、E:12118–12121、E:12685）。
- **[已验证-哈希约束]**；⚠ 这是**重建**，不是原 parquet 的再验证（原服务器 parquet 已被磁盘清理删除，E:11575）。

### 3.7 服务器清理与资源事实（影响后续可复现性）

- 2026-09-08 清理：删 2392 文件，释放 **约 39.64 GiB**；System 15.98→27.73 GiB（+11.75），Data 5.60→33.49 GiB（+27.89）。删除清单 **排除模型权重与当前打开文件**；19 个权重文件在删前删后核对 size/inode/mtime（E:2087–2114）。cleanup plan SHA256 = `8e1e25c1f9b0f0a5c6704b630916392de0a6abc8af6959a797b82b1aeab04dda`。
- **副作用**：`/root/autodl-tmp/fineweb_edu/sample/10BT/` 已不存在（E:11575）→ 直接导致 `natural_nll.py prepare` 失败（见 §4）。
- RTX 4080 SUPER 实测 **32760 MiB**、compute capability **8.9**（E:2142）。

### 3.8 151M 旋转预算训练的支撑数字

- 参数量 **151898880**（E:11985、E:12032）；训练 token **499,974,144**；**7629** updates；micro-batch 8、全局 32、seq 2048 ⇒ input_tokens = updates × 65536。
- 论文侧数字（E:10473–10541、E:11225–11273 区间）：**151.9M** 固定支撑 vs target-aware；anchored EVQ 减 FMRoPE 的 NLL = **+0.026 @256**，**−0.281 / −0.176 / −0.146 @512/1K/2K**（固定支撑）；target-aware 反转 = **+0.060 / +0.227 / +0.460**。
- 优化器常量（E:12508）：AdamW，betas [.9,.95]，weight_decay .01，peak_lr **6e−4**，min_lr **6e−5**，warmup **762**（updates=7629 时），schedule `original_cosine_final_step_floor`，clipping 1.0；precision `FP32_master_BF16_autocast`。**[部分证据-常量来自代码，未见结果引用]**

### 3.9 其它被引用的历史数字

- 早期 shape/theta 的 sigmoid 短/长 PPL 为 **37648 / 62474**，因比值小而误写"更稳定、PASSED"；已撤回（E:426）。理由：低崩溃比可以来自短端已经坏掉，不能作能力收益。
- LoRA 容量的错误界已撤回：rank-r 全局更新的每头切片可以各有 rank-r，受行列维数限制，**不能直接除以头数**（E:1347–1348）。
- HEADWISE 消融表（E:1449–1450）：per-layer α = 16 个参数 → 3.8554 / 3.1170 / 4.3476，`alpha = 0.8955–1.0962`；另一行 per-layer α 0.22437 / 0.135 / 171。**[部分证据-上下文截断]**
- 历史候选的 tensor SHA：Native-sector 冻结数组 `5e51ee46d1bce4adf8b81e6e1594df233f97022ea2fb5ed13fcc658dbee6ec7f`（E:1974）；plan SHA `ef27e120…3663d4`；候选 JSON SHA `6ad1d1b7…0e4ec09`（E:1979–1980）；完整 13 项队列首个 plan SHA `58ad6048…3b7560c3`（E:2005）。
- GapCapped 表 SHA（E:2054）：Qwen3B（N=17）`d6708ca473e2d1695f3d9a0299c2a93701da66f6bd76a623cf8eabc0ef8c25fd`；OLMo（N=18）`24332b8034bd723a5ca513247c50f17c5e64d9adb520f87ca185d3139872abae`。
- OLMo 实际 FP32 频率 SHA256 `fc0f443b1c58601d51209adb7e2b26df7ba10058a9dbdf193eb1de49116d446a`（E:1392）。

### 3.10 BM 几何量对照表（OLMo, N18）

| 量 | MrPro | BM | 出处 |
|---|---:|---:|---|
| radix 粗糙度 R | 0.01169590643 | 0.001754385965 | E:1404 |
| 最末中频增量 | 0.1052631579 | 0.01578947368 | E:1405 |
| Σ ε | 1 | 1 | E:1406 |

- 粗糙度与终端增量**均缩至原来的 3/(N+2) = 15%**；但**有限槽位下终端增量仍非零 ⇒ 是减小边界跳变，不是完全消除**（E:1408–1409）。**[已验证-数学]，⚠ 但 R 是几何目标，不得当 F 分项。**

### 3.11 C1 的两模型核验表

| 检查 | Qwen 源几何 | OLMo 目标几何 | 出处 |
|---|---|---|---|
| 高频段逐位保留 | j ≤ 23，通过 | j ≤ 14，通过 | E:1694 |
| 低频段逐位保留 | j ≥ 40，通过 | j ≥ 32，通过 | E:1695 |
| 全部频率有限正严格递减 | 通过 | 通过 | E:1696 |
| exponent 范围 | [−0.162752403, 1] | [−0.163779770, 1] | E:1699 |
| 负 exponent 槽 | 24–29 | 15–20 | E:1700 |
| 最大频率 / Native 频率 | 1.253102790 | 1.254888813 | E:1701 |
| 有效改动槽 | 24–39（16 槽） | 15–22, 24–31（16 槽） | E:1702 |

- 差异能量解释比例：P2−Mr 的中频内部 exponent 总和 = **+4.844619781**，而 ψ 投影总和 = 0 ⇒ 一维投影只解释差异能量的 **13.1846%**；残差平方和从 **2.831579713** 降至 **2.458247109**（E:1722–1725）。
- 逐槽反证表（E:1712–1720）：槽 26 C1 exponent = **−0.162752**（越过 Native 变升频）；槽 29 → **−0.007008**（与 P2 方向相反）；槽 30 → 0.091203；槽 31 → 0.203819（均与主要差异方向相反）；槽 32 → 0.325593；槽 36 → 0.799362；槽 39 → 0.993807。**[已验证-数值]**

---

## 4. 死路：已证伪 / 已失败、不得再试的机制（dead_ends）

> 每条给出失败原因与出处。**这些条目一经记录，不再重启**。

1. **C1 ψ 投影候选（a\* = 2.1477690111869907）** —— **不推荐进入 GPU 队列**，原系数与原数组保留、不后台改表。
   - 失败原因（三条独立理由，E:1722–1734）：(a) 目标差异主要不是这个零均值方向（P2−Mr 的 exponent 总和中频 +4.8446，ψ 投影总和为 0；只解释 13.1846% 的差异能量）；(b) 最关键两槽（30/31）方向反了，P2 在这里需要相对 Mr **增加** ~.668/.763，C1 反而**减少** ~.092/.031；(c) 保持两端**没有**保持全部原生频率关系，部分中频升到 Native 之上约 25%。
   - **不得**从这些数组给出"失败概率 80%"之类的数字；也**没有**证明实际得分必然下降。裁决只针对这一固定投影，不能扩展成"所有中频分配或所有简单改进均无效"（E:1737–1739）。**[已否决-设计层]**

2. **gap_capped 参数自由全局表（GapCapped, 36 全层同一张表）** —— 已生成两张表（Qwen3B N=17、OLMo N=18）并启动 `gap_capped_run_01`（pid 2991，E:2065–2082）。
   - 机制动因（外部审查原文，E:2020，S:580 附近）："MrRoPE-Pro 把整个 transition 里最大的频谱间距，恰好放在 low-frequency boundary 前一格，然后下一格瞬间掉回 native gap。"
   - **⚠ 本 extract 的 12808 行内未见该运行的结论**；不得据此宣称成功或失败。**[未闭合-状态未知]**
   - 相关旁证：`scripts/analysis/verify_signed_lag_kway_gap.py:295` 记录 "softmax-gap image: PCA1~Cosh, **0.854** inside null p99"（E:2019）—— 这是**模拟零假设下的几何量**，⚠ 红线禁用为 F 分项或选择子。

3. **Native-sector（去载波）载波裁剪候选** —— 改表**没有**解决短 UUID 损失。
   - 实测（E:1989–1994）：UUID/4K n2 新候选 **50** vs 已有 Mr **100**；VT/4K n2（128-token）90 vs 90；UUID/128K n8 12.5 vs 12.5；VT/128K n8 70 vs 62.5。
   - 长 UUID 逐行分数**完全相同**；长 VT 配对增量 = **[20,20,20,0,0,0,0,0]**（E:1996）。
   - 原文结论（E:1997–1998）："改表没有解决短 UUID 损失，**不能宣布 Native 无损、超过 Mr 或 SOTA**。" 长 VT 前 30-token 分数 65、短 VT 前 30-token 80。

4. **同表 key 重建在 bitwise 意义上不可行** —— `cross_cache.py` 第 157 行抛
   `ValueError('same-table key reconstruction is not bitwise exact')`（E:7843–7845），导致 `stage_run_01` **FAILED**。
   - 修复路径：重写 runner + 新增 `--matched-cached-baseline` → `stage_run_02` 建立"受约束的 cached-query 基线"。
   - **不得**再试图对**已舍入的 BF16 缓存 key**求逆来重建同表 key。**[已证伪-数值精度]**

5. **`layer_screen_01` 状态 FAILED / `"error": "operator stop"`**（E:1752–1760），traceback 落在 `layer_screen.py` 第 90 行 `raise RuntimeError('operator stop')`。属操作者主动停止，不是机制失败——但该次运行**未产出可用结果**。**[未产出]**

6. **prepare_ruler 的 7 个新任务在远程缺 NLTK `punkt_tab`** → 失败；修复为 `nltk.download("punkt_tab", download_dir="/root/nltk_data")` 后以 `prepared_ruler_newtasks_02` 重新启动。**[已修复]**

7. **`natural_nll.py prepare` 失败**：`FileNotFoundError: '/root/autodl-tmp/fineweb_edu/sample/10BT/000_00000.parquet'`（服务器磁盘清理删除）。
   - 修复：把 prepare() 改为解码冻结的 Qwen token 文件、再为目标模型重新分词，并加 `--source-tokenizer`。
   - **明确记录**：这**不是**对原 parquet 的再验证。**[已修复但削弱了证据等级]**

8. **`summarize_candidate_screen` 在 `run_ruler_newtasks_01` 上失败**：先 `FileNotFoundError: MrProBM.json`，再 `ValueError: incomplete task/length panel`（`ruler_bench.verdict` 要求恰好 2 个 cap 且固定 6 个 TASKS）。修复：加 `--frozen-panel`（E:5346–5458）。**[已修复]**

9. **Qwen7B `run_screen_01` OOM**（E:7060–7068）→ 加 `expandable_segments:True` 并复用 6 条已完成行，重跑为 `run_screen_02`；最终 **0 wins / 3 losses** = NO_LONG_GAIN。**[已验证-负结果]**

10. **`python` 不在 PATH**：`zsh:1: command not found: python`（E:11880 与更早的 E:383 附近），远程 `python`（非 miniconda 全路径）同样 127。全线程统一改用 `python3` 或 `/root/miniconda3/bin/python`。**[已修复-环境]**

11. **`scp` 花括号展开失败**：`scp ...:{protocol.py,run_experiment.py}` → "No such file or directory"（E:10896）；改成两条显式远程路径后成功（E:10985）。**[已修复-工具用法]**

12. **KLD v2 的一个重要降级（不是失败但是限制）**：额外的一个普通 delta 状态 + 独立可控的 erase/write，**可以用有限差分逼近同一个导数**，因此"导数解释"**不建立**一个普遍更优的记忆族；下一次实验必须把"可用的条件优势（信息/条件数/算力）"与"单纯把内存翻倍"区分开。原文状态 `CPU_ALGEBRA_ONLY`，`limitation` 字段自陈无 checkpoint、无全网生成、无 GPU kernel 测试。**[降级-叙事→限制声明]**

---

## 5. 与权威文档的冲突 / 矛盾 / 定义不一致

### 5.1 与 `/Users/yang/projects/hybrid-rope/analysis/unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md`

- **本文档内部一致（互相印证）**：
  - unify 文档 F1/F2 明确证伪"YaRN 递减 vs MrPro 递增"作为机制解释；本 extract 里 **未出现**任何以此为核心的断言（线程走的路线是 BM / C1 / GapCapped / carrier，不是增量单调性）。两边**无冲突**。
  - unify 文档 F3 的"Σ(ν^M−ω)²/Σ(ν^Y−ω)² = 0.4841"与本 extract 里 BM 的"m_q^BM − m_q^Mr > 0（每槽都增加压缩）"是**不同对象**（前者 YaRN vs MrPro，后者 BM vs MrPro），**不可混用**。⚠ 需点名坐标/点名比较对象。
  - unify 文档 §1 给出 MrPro 的 m_q = q(q+1)/[N(N+1)]；本 extract 里 BM 的 m_q = q(q+1)(3N+2−2q)/[N(N+1)(N+2)]。两者**分母不同、形式不同**，是**两个不同的表**，不要合并。

### 5.2 与 `NEXT_DERIVATION_KKT_PROBLEM.md` / `INTEGRATION_20260910.md` 的关系

- 本 extract 里的 **rotary_budget 六臂设计（G32/E32/G16/E16/P16/U16 + τ=√2 + base 500k）** 与 **C1/BM 的闭式分配**属于**两条独立线**：前者是"固定旋转宽度的分配问题"，后者是"YaRN 升级 / 零训练中频改造"。两者都触及"分配"二字，但优化对象不同（前者优化紧支撑下的频率间距形状，后者优化中频累计压缩指数）。⚠ **不要把 theory_check 的谱尾界当成为 BM 提供的理论依据**：原文自己写 "This is geometry, not LM evidence"（E:11542–11543）。
- 本 extract 中 **未见**任何把"窗内损失最小化 + 外推最大化"写成单一 KKT 系统的正式表述；最接近的是 C1 文档 §4 提到的截图里的"双 cost 公式"（"Native 关系损伤" + "长距相位外推代价"），文档明确写：**"目前给出的只是名称，尚非可直接求解并宣称理论增益的损失函数。"**（E:1741–1745）。**[叙事-未验证，且原文自我降级]** —— 这对本 workflow 的目标函数 F 是一个**空白点，也是一个反面证据**：该线程**没有**解出 KKT。

### 5.3 定义不一致

- **"中频边界"在两条线里用不同定义**：BM/carrier 用**圈数**边界（>32 圈 / <1 圈，E:5880–5886）；C1/transition 用**零基槽号** l/h（Qwen l=23,h=40；OLMo l=14,h=32，E:1656、E:1688）；rotary_budget 用 **d_rope = 2K 个旋转维**。三者不可互换。
- **"gain"** 在两处出现：BM 的 `1 + 0.1·ln(S)`（E:5907）= OLMo 的 1.138629436111989；而 rotary_budget 里 gain 不是参数（表直接给频率）。⚠ 同名不同物。
- **OLMo 的参数量**：本文档同时出现 **1,484,916,736**（E:1382、E:1686）与 **1.485B**（E:136）。前者是实测计数，后者是标签。
- **"守恒"用词**：C1 文档（E:1706–1708）与 BM 协议（E:1413–1414）**都**明确拒绝"能力预算守恒"的说法，但措辞上仍出现"两端 exponent 之差仍为 1""累计 radix 乘积仍为 4"。按红线，**必须点名坐标**：这里守恒的是"两端 exponent 之差"与"累计 radix 乘积"，**不是** Σm，也**不是**任何能力/损伤预算。

### 5.4 VICTORY / 已闭合 / 已证明 类结论的降级

- BM 协议文档标题级别写 `S4_GPU_COMPLETE / 开发与独立seed复核均优于MrPro`（E:1360）。**降级处理**：同一文档紧接着写"总体仍是最多 10 个候选，当前只有 BM 准备完成"（E:1361–1362），且 3B 128K **输了 7.29 pp**、**未晋级全量**（E:277）。所以这是**一个面板上的局部胜负**，不是能力结论。
- 128K 诊断的原文结论本身就是降级式（E:277）："没有复现 OLMo 的长端收益，不建议据此直接进入全量……**本轮仅判断固定 BM 在该小面板的迁移**，不宣称所有 BM 变体无效，也不能单独归因于模型容量。"
- KLD v2 的 `RESEARCH_PLAN.md` 自述："**It does not establish a new universally superior memory family**"（E:12521–12526）。

---

## 6. 开放问题（open_questions）

1. **`gap_capped_run_01`（Qwen3B，pid 2991）的最终结论在本 extract 中缺失**（E:2082 之后无回读）。它是本线程唯一一个"参数自由全局表"候选，其成败直接影响"最大频谱间距应该放在哪"这一问题的答案。**出处**：E:2038–2082。
2. **BM 在 16K/32K 的正面结果与 128K 的负面结果如何统一？** 原文把机制嫌疑放在"中段普遍加大压缩破坏原生/局部处理""槽间光滑性不代表学到的 Q/K 敏感性""较早压缩降低所需距离分辨率"（E:1425–1427），但**没有**隔离任何一个。128K 诊断只排除了"那一个相似 key"是唯一原因（E:265）。
3. **"最大相位差"与"最敏感槽"的关系未建立**：E:267–268 明确写"相位差大本身不是失败的充分条件……不能把最大相位差的槽 28 自动称最敏感槽"。→ 需要 **attention 贡献的符号**层面的证据（E:227："该数学事实不自动决定注意力贡献的符号"）。
4. **C1 文档 §4 提到的"双 cost 公式"是否可写成可解的优化问题？** 原文只说"目前只是名称"（E:1745）。这是最接近 KKT 目标函数 F 的一条线索，**未被任何人形式化**。
5. **rotary_budget 六臂实验是否有结论？** 本 extract 只见预注册协议与探针失败（`validate_cuda_runtime` 抛 `PyTorch build lacks native sm_89; compiled for ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`，E:12715）。**探查探针 FAILED，returncode 1**（E:12687）。→ 六臂训练**未开始**（截至本 extract）。
6. **BM 的 R 最小化与任务损失之间是否存在可证的关系？** 原文只写"R 是几何目标，不是任务损失"（E:1409）、"最小粗糙度闭式正确不自动证明它是最佳语言模型频率表"（E:1418）。这是一个**明确留白**。
7. **C1 的 a\* 跨模型迁移（Qwen 系数 → OLMo）缺乏依据**（E:1732–1734）：OLMo 原生长度、base、学到的 Q/K 与 Qwen 不同。原文声明这是"作者指定小模型评测下明确记录的设计"，不得借用 Qwen 旧分数当 OLMo 的证据。**这也提示：任何 KKT 闭式若含模型相关量，跨模型迁移都需要重新参数化。**

---

## 7. 被"后来引用"的数字（可追溯链）

| 数字 | 首次算出/记录 | 后来在何处被引用 |
|---|---|---|
| BM 16K 51.32% / MrPro 2.78% / MrUni 32.12% / YaRN 6.94% | E:136 | 线程 README/状态行、paper-2027 相关段落 |
| 3B 128K BM 70.83% vs MrPro 78.13%（−7.29pp） | E:143、E:277、E:287 | 作为"未晋级全量"的门控依据；E:1895 的用户规则条目 |
| BM 中段相对 MrPro 下降 1.5%–31.5%（最大槽 34） | E:227 | 128K 退化诊断 |
| tail-512 NLL：MrPro 3.21062 / MrProBM 2.95506 / Native 2.83505 @4K | E:6470–6488 | `docs/research/ROPE_OLMO_BM_NLL_RESULT_20260908.json` |
| 五任务 QA extended delta +0.03819，CI [0.01316, 0.06292] | E:6363–6386 | `docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json` |
| BM 协议闭式 m_q | E:1373–1376 | E:5880–5922 的实现；`scripts/lib/rope/boundary_matched.py` |
| C1 a\* = 2.1477690111869907 | E:1673 | 后续"不推荐 GPU"的裁决（E:1736–1739） |
| carrier c = 1.2409377632138785e−6 | E:1940 | 新表构造与 20 条开发比较（E:1989–1994） |
| 151898880 参数量 | E:11985 | E:12032 的硬断言；protocol.yaml `expected_parameters` |
| 499974144 tokens | E:11796 | E:11854 的重建哈希门槛；REPORT.md E:11555 附近 |
| 512 篇 8193-token 验证文档 | E:11867、E:12685 | `eval_inputs.make_example` 的硬前置（E:11478） |
| KLD 干扰 S/H（128: 0.3664/0.3693；512: 0.01803/0.07269） | E:12109 | E:12701 二次运行复现（数值完全一致） |
| τ = √2、BASE = 500000、TAU | E:11296–11340 | protocol.yaml 冻结、`theory_checks.json` |

---

## 8. 给下游的可用件清单（按"能否进 F"分类）

**可以直接进 F 的零件（约束/定义层）**
- BM 闭式：ε_i、m_q、ω′_j = ω_j S^(−m_{j−l})，及约束 Σε = 1（E:1371–1379）
- 中段边界用圈数定义（>32 / <1，E:5880–5886）；gain = 1 + 0.1·ln(S)（E:5907）
- 载波区间约束：0 ≤ ((ω_j − c)/S)·δ ≤ ω_j·L0 ⇔ c ≤ min_B ω_j；闭式解 clip(N/D, 0, min_B ω)（E:1932–1940）
- C1 的一维最小二乘骨架：ψ_j = x(1−x)(2x−1)、a\* 的闭式比（E:1658–1673）
- KLD 的两槽递推与非扩张结构、H_t = ∂_z S_t(z)|₀（E:12056–12065、E:10324–11420 区间的审计文档）
- Qwen2 bias 分解：s_H = a²/√D [qᵀR_B k + b qᵀ(R_M − R_B) bk]（E:8171–8330 区间）

**只能当"实现存在性/表差异"证据、不得进 F 的量（红线）**
- 一切谱尾界、`spectrum`、`tail`、`eta`、阈值计数（E:11425–11470）
- Laplacian 粗糙度 R（E:1404；原文自称"几何目标，不是任务损失"，E:1409）
- PCA1~Cosh 0.854 的 softmax-gap 零假设量（E:2019）
- 覆盖率 / 有效秩 / Gram / 碰撞能 / 平滑度 / 能量：**本 extract 中未出现被用作选择子的情形**，保持禁令。

**明确不可用的（已证伪或未闭合）**
- C1（E:1736–1739）；同表 bitwise key 重建（E:7843–7845）；carrier 短 UUID 恢复（E:1996–1998）；Qwen7B NO_LONG_GAIN（E:7060 附近）；六臂训练（探针 FAILED，E:12687）。
