# T7 挖掘摘要：codex 线程后半段（tools_b）+ 落盘清单 + 子代理调度

来源文件：
- `/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/mine/extracts/codex_tools_b.txt`（12808 行，806KB，397 条 `[CMD]`）
- `/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/mine/extracts/codex_filechanges.txt`（303 行，50 条 `[FILECHANGE]`）
- `/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/mine/extracts/codex_subagents.txt`（101 行，29 条 `[SUBAGENT]`）
- 交叉核对：`/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/extract_codex.py`（只读抽取器）

## 0. 行号约定（必读，否则引用会错位）

`extract_codex.py:75-76` 的 `line(label, ln, txt)` 把 **源 JSONL 的行号** 写进标题。因此三份 extract 里的 `line N` 全部是
`/Users/yang/.codex/sessions/2026/09/08/rollout-2026-09-08T09-59-44-01a08151-...jsonl` 的行号，**不是 extract 文件自己的行号**。
本文所有出处的写法：`codex_tools_b.txt [CMD] line N`，其中 N 是源 JSONL 行号；必要时附 extract 文件行号（写作 `ext#`）。

`extract_codex.py:22` 给出 PATH；`extract_codex.py:18` 明确 "NEVER writes to ~/.codex"——本轮挖掘全程只读，未触碰 `~/.codex`。

分段边界（由 header grep 实测）：
- `codex_tools_b.txt` 覆盖 JSONL **3449–6774**，397 条 `[CMD]`。
- `codex_filechanges.txt` 中 JSONL **115–2804** 属 tools_a（T6 范围），**4347–6070** 属 tools_b（本任务范围）。[已验证，来源：codex_filechanges.txt 全部 header 及 codex_tools_b.txt 首尾 header]

**本线程与 KKT 主题的距离（重要判断）**：对 `codex_tools_b.txt` 全文 grep `KKT|lagrang|dual` 无任何命中；grep `tau` 仅命中无关词（"tau=1" 出现在被引用的 Phase18–23 复盘文字里，JSONL 12761）。也就是说，这条线程**不是 KKT/EVQ 推导线**，而是 "sparse position / RefCarry / rotary budget" 研究线。它对 KKT 交付物的价值是三块：精确代数恒等式与反例、已被实测钉死的经验锚点、以及一份异常明确的失败台账。[已验证]

---

## 1. 该线程落盘产出了哪些文件

### 1.1 属本任务（tools_b）范围：JSONL 4347–6070

| 路径 | 动作 | 用途 |
|---|---|---|
| `experiments/native_sparse_position/prepare_natural.py` | add @4347 | 从 LongBench data.zip 生成 natural 长 QA 输入（chat-template 冻结身份） |
| `experiments/native_sparse_position/CORE_DIAGNOSIS.md` | add @4392 | 核心诊断文档：PSR 相位/内容方差分解表、mean-summary 臂、InfLLM-V2 共享头选择机制 |
| `experiments/native_sparse_position/summarize_natural.py` | add @4432 | natural 续写结果汇总 |
| `experiments/native_sparse_position/phase_content_audit.py` | add @4464 | 从已存激活离线做相位/内容方差分解（无新前向） |
| `experiments/native_sparse_position/download_pinned.py` | add @4585 | MiniCPM4.1-8B 等 pinned 资产下载 |
| `experiments/native_sparse_position/normalizer_audit.py` | add @4732 | InfLLM-V2 细/粗归一化子审计 |
| `experiments/native_sparse_position/minicpm_native.py` | add @4843 | 原生 MiniCPM sparse 复现 |
| `experiments/native_sparse_position/qualify_minicpm_kernel.py` | add @4985 | 原生 kernel 资格验证 |
| `experiments/native_sparse_position/orbit_regression.py` | add @5107 | 中心化轨道投影回归（后被证伪，见 §3） |
| `tests/test_orbit_regression.py` | add @5107 | 上述回归的测试 |
| `experiments/native_sparse_position/ORBIT_DERIVATION.md` | add @5155 | 轨道回归推导 + 反例记录 |
| `experiments/native_sparse_position/prepare_support_oracle.py` | add @5204 | support/wrong 块 oracle 输入制备 |
| `experiments/native_sparse_position/generate_primary_ruler.py` | add @5501，update @5687 | 主 RULER 生成器 |
| `experiments/native_sparse_position/quest_fine_native.py` | update @6042、@6070 | QuestFine32 原生实现 |

此外 tools_b 期间还写入（由 CMD 中的 heredoc 可见，部分在 .gitignore 之外故未出现在 FileChange 之外的路径）：
- `experiments/native_sparse_position/pair_envelope.py`、`envelope_native.py`、`envelope_ruler.py`、`summarize_envelope.py`、`support_ruler.py`、`mean_native.py`、`oracle_native.py`、`groups.py`、`DESIGN_SOURCE.md`、`RESULT_20260908.md`、`METHOD_INDEPENDENT_20260908.md`
- `tests/test_pair_envelope.py`
- `experiments/refcarry_audit/{README.md, results.json, checks.py}`、`experiments/native_sparse_position/aux_refcarry_math_checks.py`
- `experiments/rotary_budget/runtime.py`、`experiments/rotary_budget/evidence/probe_mb8_02.json`
- `docs/research/POSITION_AFTER_SPARSE_RESEARCH_20260908.md`、`docs/research/ACTIVE_RESEARCH_GOAL.md`、`paper-2027/HANDOFF.md`（多次）
[证据等级：文件路径与动作 [已验证]（来自 FileChange 与 heredoc）；"用途"列 [部分证据]（依据文件内 docstring 与调用上下文推断）]

### 1.2 属 tools_a（T6）范围：JSONL 115–2804（列出以划定"不必重做"的边界）

`scripts/experiments/olmo_fast_screen/` 下 cross_cache.py / layer_policy.py / layer_screen.py / causal_gain.py / prepare_natural.py / prepare_ruler.py / download_qwen7b.py / chunked_mlp.py / bias_position.py / native_windows.py；
`scripts/lib/rope/gap_capped.py`；`scripts/analysis/summarize_candidate_screen.py`、`summarize_natural_screen.py`；
`tests/` 下 test_cross_cache.py / test_layer_policy.py / test_causal_gain.py；
`docs/research/` 下 ROPE_BM_CROSS_CACHE_20260908.md / ROPE_OVERNIGHT_RESEARCH_20260908.md / ROPE_GAP_CAPPED_PROTOCOL_20260908.md / ROPE_GAP_CAPPED_RESULT_20260908.md / ROPE_OLMO_BM_NATURAL_TRANSFER_20260908.md。
[已验证，来源：codex_filechanges.txt line 115–2804]

> 含义：**"gap_capped 协议 + OLMo BM 自然迁移 + cross-cache 屏幕"这三条已被 T6 覆盖**，不要重复挖。本线程独有的增量是 `native_sparse_position/` 整条线 + `refcarry_audit/`。

---

## 2. 数学提取：可直接作为 F 零件的公式 / 定义 / 约束

### 2.1 群矩恒等式与"最小充分状态"（**最高价值**，是唯一自带的、可证明的线性代数结构）

- **群矩因子化恒等式**。设 `ρ: Z → O(d_r)` 为平移群的正交表示（RoPE 是其二维旋转块特例），前一次 read 给出地址分布 μ。则
  `s_ij^ref = (M_μ q_i)^T ρ(p_j) k_j`，其中 `M_μ = E_{a~μ} ρ(p_a)`。RoPE 特例下第 k 个频率通道 `m_k = Σ_a μ(a) e^{iω_k p_a}`。
  出处：`codex_tools_b.txt [CMD] line 6443`（ext#11308-11308 起，用户粘贴的 RefCarry 提案正文）。[已验证：代数恒等式]
- **数值见证**：`group_logit_factorization_max_abs_error: 2.1094237467877974e-15`（100 次随机试验）。出处：`[CMD] line 6541`（ext#11671）。[已验证]
- **最小充分 sketch 维数**。`simplex_rank_counterexample`：`rank_Phi = 2`，但充分线性 sketch 维数为 **1**；正确的一般维数写作
  `rank([ones_row; Phi]) - 1, with known unit mass and an affine decoder`。出处：`[CMD] line 6541`（ext#11636、ext#11715）。[已验证：反例 + 修正后的通式] —— 这是**能直接进 F 的约束式**（"状态维数下界 = 增广矩阵秩减一"）。
- **矩不决定边缘读**：`same_moments_different_reader_mixtures`（moments 相等，reader mixture 不同）、`softmax_counterexample`（logits 均值 → attention [1/3,1/3,1/3]；attention 均值 → [0.49998, 0.49998, 4.54e-05]；`total_variation: 0.33328793546472446`）。出处同上。[已验证]
- **固定 key 的 TAPE 式算子恒等式误差 = 0.0**（`fixed_key_TAPE_style_operator_identity_error`）。出处同上。[已验证]
- **零熵门抵消反例**：`zero_entropy_gate_cancellation.query_norm_ratio = 6.123233995736766e-17`。出处同上。[已验证]
- **native query 重基准反例**：current 10、gold anchor 6、desired offset −1 → native argmax 5，gold-reference argmax 1（`native_query_rebasing_counterexample`）。出处同上。[已验证]
- **reference 平移反例**：`shift_reference_counterexample.selected_positions = [7,7,7,7]`，slope 0.0。出处同上。[已验证]
- **非线性残差地址切换**：有界域上误差 `2.22e-16`（`nonlinear_residual_address_switch`）。出处同上。[已验证]
- 用户粘贴的 beta 等价式：`beta = [mean(conj(z)*y) - conj(m)*mu] / sqrt(1-|m|^2)`，来源 `[CMD] line 5183`（ext#6700）。[部分证据：出现在 METHOD_INDEPENDENT 讨论中，未见独立数值验证]

### 2.2 RoPE 频率扰动律（**直接可作 F 的约束项**）

- `‖R_{ν+δν}(Δ) − R_ν(Δ)‖₂ = 2|sin(Δδν/2)|`。
- 实例：Qwen K=64、base=10^6、`ω_j = 10^(−6j/64)`；高频边缘 `j=23` 的 `ω ≈ 0.0069783`；对 ω 改 1% 时，Δ=128 上额外相位 ≈ **0.00893 rad**，Δ=32768 上 ≈ **2.28665 rad**。
出处：`codex_tools_b.txt [CMD] line 6767`（ext#12706-12711）。[已验证：公式为代数恒等式；数字为代入计算]

### 2.3 稀疏路由的精确关系式（可作 F 的"检索"侧约束）

- 保留下界：`min_h r_h ≥ max(0, H·r̄ − H + 1)`，且该界可达。
- 逐头偏差：`TV(p_h^S, p_h) = 1 − r_h(S)`，`D_KL(p_h^S ‖ p_h) = −log r_h(S)`。
- 其中 `p_hb = Σ_{j∈b} exp(q_h^T R(p_j−p_q) k_j/√d) / Σ_{j∈legal} exp(...)`，`r_h(S) = Σ_{b∈L∪S} p_hb`，`|S| = K`。
出处：`codex_tools_b.txt [CMD] line 6259`（`docs/research/ORACLE_FIRST_SHARED_ROUTING_PLAN_20260908.md`，见 ext#~6770–6847 段）。[已验证：恒等式；`min_h` 界在文中标注 attainable]

- **InfLLM-V2 共享头打分的精确恒等式**：
  `Σ_h exp(s[h,c]) / Zhat[h] = Σ_h (Z[h] / Zhat[h]) · softmax(s[h,:])[c]`，
  其中 `Z[h]` 为精确细粒度配分、`Zhat[h]` 为 128-token/64-stride 粗估计。文中指出：公共倍数在块排序中相消，**头相关因子不相消**（改变头权重），因此 within-head top-k 对该机制不变——"这解释了为什么 per-head 诊断无法识别该机制"。
  出处：`codex_tools_b.txt [CMD] line 5741`（`CORE_DIAGNOSIS.md` 尾部，ext#8746-8756）；同一段亦出现在 ext#4888 附近。实现细节旁证：`[CMD] line 5741` 同段写明"numerator 用 32/16-strides 均值，normalizer 用 128/64"。 [已验证：恒等式；但对 checkpoint 是否更优"是代数，不是新定理"——原文自己降级]

- **FP64 初探**：8 query heads / KV group 下，有效头数 **7.049（RoPE） vs 7.283（NoPE）**。出处：`[CMD] line 5741`（ext#8765-8767）。[部分证据：文中自己标 "preliminary FP64 probe"，且该句在 extract 中被 clip 掉尾部 378 字符]

### 2.4 上界描述符（RPEE / Quest）的数学形状

`experiments/native_sparse_position/pair_envelope.py`：
- `U_pair(q) = q·c + a|q·u| + b|q·v|`（矩形分支）；各向同性时退化为 `q·mu + r‖q‖`（圆盘分支）。
  ```python
  rectangle = a * qu.abs() + b * qv.abs()
  disk = a * qnorm
  spread = torch.where(cache.isotropic.unsqueeze(1), disk, rectangle)
  value += (center_term + spread).sum(dim=-1)
  ```
- `U_b(q) = Σ_pairs U_pair(q) + nonrotary_terms`，文档明确其为 **max-token logits 的上界，不是块对数质量的估计**（原文 "The mathematical descriptor bounds max-token logits, NOT block log mass."）。
- 缓存字节：每块 `2D + d_rot` 个数（满旋转 = 3D），介于 PSR 的 4D 与 Quest 的 2D 之间。
- FP32 安全上界填充策略：`factor = 8.0 * (dim + 8) * finfo(float32).eps; upper = value + factor * magnitude; nextafter(upper, +inf)`。
- 配对定义：native 为 `(i, i+K)`；random 用 `RANDOM_PAIR_SEED = 20260908`。
出处：`codex_tools_b.txt [CMD] line 5546`（ext#7763-7796）与 `[CMD] line 5636`（ext#8257-8337）。[已验证：代码原文]

三条可证结构（原文自列）：(A) 对当前 query 是合法上界；(B) 共同旋转下等变（原始 Quest 的轴对齐盒**不具备**该性质）；(C) 共线 key 时精确等于支撑函数。出处：`[CMD] line 6637` 附近（`METHOD_INDEPENDENT_20260908.md`，JSONL 6979–7043 段）。[部分证据：文档自述；CPU 反例见 §3]

### 2.5 RMBC（Reader-Metric Block Coreset）—— 已被主代理自己否掉

`Fhat_b(qbar) = logsumexp_r(log(n_r) + qbar·mu_r)`；`d(s,t)^2 = ‖y_s−y_t‖₂^2`；per-KV-head farthest-first R=4；`mu_r = mean(y_j, j∈C_r)`，`n_r` 为整数计数。
- 单调性：`Σ_r n_r exp(qbar·mu_r) ≤ Σ_j exp(qbar·y_j)`。
- 每组 gap 界：`rho_r = max_{j∈C_r}‖y_j − mu_r‖₂`，`a_r = ‖qbar‖₂ ρ_r`，则 `0 ≤ J_r ≤ min(‖qbar‖²ρ_r²/2, ‖qbar‖ρ_r)`。
- R=4 约为 K cache 的 6.25%（全 K+V 的约 3.125%）。
- **原文自带降级**："rho_r 只用于诊断和误差上界，不是第一版部署必须增加的自适应精排流程。高维情况下该上界可能松；不能用它宣称已经获得实用的无漏选证书。BF16/FP4 舍入也需要额外误差包络。"
出处：`codex_tools_b.txt [CMD] line 4072`（ext#1225-1240，即 `DESIGN_SOURCE.md` / 用户原始 plan）与 `[CMD] line 4415`（ext#2586-2601），以及 `[CMD] line 5183`（ext#6652-6700，`METHOD_INDEPENDENT_20260908.md`）。[已验证：公式与自我降级句为原文]

### 2.6 相位/内容方差分解恒等式（可作诊断，**不是能力指标**）

对块均值 key 在原始位置上旋转构造 phase-only 分量，与保存的 native key 之差为 content residual（含 BF16 舍入）。分数方差**精确分解**为
`native = phase + content_residual + 2 * covariance`（代码内 `residual = totals['native'] - totals['phase'] - totals['content_residual'] - totals['twice_covariance']`，并 `assert abs(residual).max() <= 1e-8`）。
出处：`codex_tools_b.txt [CMD] line 4415`（ext#2586 段、`CORE_DIAGNOSIS.md`）与 `[CMD] line 5741`（ext#8713-8714），实现在 `[CMD] line 6410` 附近（ext#9664-9678）。[已验证：恒等式与断言]

数值表（均匀 over eligible query/block pairs）：

| Model / partition | Native variance | Phase | Residual | Twice covariance |
|---|---:|---:|---:|---:|
| Qwen3.5 / whole block | 2.952803 | .023275 | 2.928698 | .000830 |
| Qwen3.5 / PSR | 2.457900 | .013475 | 2.443798 | .000626 |
| Qwen3.5 / contiguous | 2.363582 | .011247 | 2.352000 | .000335 |
| Qwen2.5 / whole block | 4.899932 | 1.937103 | 2.963351 | −.000522 |
| Qwen2.5 / PSR | 3.886419 | 1.352520 | 2.533197 | .000702 |
| Qwen2.5 / contiguous | 3.886309 | 1.350432 | 2.537490 | −.001613 |

出处：`codex_tools_b.txt [CMD] line 4415`（ext#4894-4899）。[已验证：数值为文档原文]
原文自己的判词："phase 项在采样的 Qwen3.5 query 上极小；Qwen2.5 上显著，但 PSR 并未降到 contiguous 对照之下。这是待查的诊断解释，不是'什么主导全部模型注意力'的证明。"（[已验证：原文措辞]）

---

## 3. 已被证伪 / 已失败 / 不得再试的机制（dead ends）

每条附失败原因与出处。

1. **块均值摘要整族（RoPEMean / NoPEMean / 任何 mean-summary）**。冻结 32 例：RoPEMean **0/32**，Dense 17/32。失败原因：把整块压成均值后，query 与块内单 token 的相对相位信息被抹掉，selector 无法恢复块内最大值位置。出处：`[CMD] line 6074`（ext#10463-10533，`RESULT_20260908.md`）与 `[CMD] line 6107` 段。**这是本线程最硬的负结果**。[已验证：原始 32 例计数]

2. **中心化轨道投影回归（orbit regression）**。显式反例：一个 64 点单位范数 key 块，其中心化轨道投影把 key MSE 降约 **67%**，却造出沿实轴的虚拟最大值 **1.273**，大于所有真实点的最大值 1。query = 20 时该块排序高于常数 1.1 块，而精确排序恰好相反。
   数值：虚拟实轴最大 `1.272983871`；中心化能量 `0.574743502 → 0.189267751`；块 A 真实对数质量 `22.525078826` vs 回归式 `27.450586177`；块 B（常数 1.1）两者均为 `26.158883083`；真实 top-1 = B，回归 top-1 = A。
   失败原因：**降低平均重构残差不控制支撑函数的上确界**；均值型目标会奖励"虚拟极值"。出处：`codex_tools_b.txt [CMD] line 5183`（ext#6712-6716）与 `[CMD] line 5300` 附近（`ORBIT_DERIVATION.md` 追加的证伪段，JSONL 7136–7155）。[已验证：反例数字为原文]

3. **"pair 边缘足以决定 joint 支撑函数"**。新测试 `test_native_pair_marginals_cannot_identify_full_token_maximum`：两个块 A/B 的**全部 pair 边缘完全相同**（`score_pair_envelope` 输出逐元素相等），但真实最大 logits 分别为 **0** 与 **2**（`(a@q[0]).max()==0` vs `(b@q[0]).max()==2`）。测试结果 `27 passed in 0.91s`。
   失败原因：pair 边缘是 joint 分布的**投影**，不决定支撑函数。出处：`codex_tools_b.txt [CMD] line 6074` 前段（ext#10390-10416）。[已验证：测试断言原文 + pytest 计数]

4. **"矩（M_μ）本身就是充分地址状态"**。反例给出 same moments 但 reader mixture 不同；attention 均值与 logits 均值给出不同结果（TV = 0.33328793546472446）。失败原因：softmax 与求和不交換，矩只决定 logits 的一阶聚合。出处：`[CMD] line 6541`（ext#11665）。[已验证]

5. **"gold-reference 重基准可作为恢复上界"**。反例显示 native argmax 与 gold-reference argmax 落在不同位置（5 vs 1），偏移方向甚至相反。失败原因：把 query 原点移到 gold anchor 并不保证精确 reader 的 argmax 跟着移动。出处：`[CMD] line 6541`（ext#11665 `native_query_rebasing_counterexample`）。[已验证]

6. **PSR（Phase-Separated Representation）静态相位分组**。Qwen2.5 上 PSR 的 phase variance 1.352520 **不低**于 contiguous 对照 1.350432；Qwen3.5 上 phase 项本身就只有 .011–.023，几乎没有可利用信号。失败原因：纯 offset 相位度量没有利用决定组内 score 分布的大量信息（content residual 占方差主体 2.44–2.96）。出处：`[CMD] line 4415`（ext#4894-4903）与 `[CMD] line 5183`（ext#6649）。[已验证：数值；"[叙事] 这是否等于整条 PSR 路线失败" 原文自己否认——"这不是认为 PSR 全路线失败"]

7. **`min_h r_h` 的平均速率式推理**。`min_h r_h ≥ max(0, H·r̄ − H + 1)` 是**下界**，只给平均保持率；把它当作逐头保持的充分条件会高估路由质量。出处：`[CMD] line 6259` 段（ORACLE_FIRST 计划）。[部分证据]

8. **RMBC 作为首选方法**。被主代理自己在第二轮中取代（superseded），理由写在文档里：主任务的真正要求是让 `‖y_s−y_t‖` 小，而不是只控制它的一个三角上界项；且高维下上界可能很松。出处：`[CMD] line 5183`（ext#6649）与 `[CMD] line 5577` 前后（`METHOD_INDEPENDENT_20260908.md` 第二轮）。[部分证据：文档自述；未见独立复算]

9. **旋转不变的"静态几何/响应代理量"整族**。权威复盘直接列表裁决："Gram、曲率、重构残差、phase risk、覆盖、平滑性变好 ⇒ 生成变好 —— 已有直接反例；18 样本 64 维行为梯度也已失败，不重新包装成'功能需求恢复'"。出处：`codex_tools_b.txt [CMD] line 6774`（ext#12765）。[已验证：原文裁决] —— 这一条**与用户红线完全一致**，是本线程与 KKT 交付物的强对齐点。

10. **窗口内训练白送外推**（来自被引用的记忆/复盘，非本线程新证）："ZC vs ZF p=0.0076" 已证伪。出处：`[CMD] line 6762` 附近引用的记忆条目（ext#12600-12616 区域）。[部分证据：属被引用材料的转述]

11. **Phase18–23 的"tau 公式换一个再试"循环**。裁决：长度、训练量、数据与架构同时变化，不能作单因素归因；"tau=1 时最高频率仍约 **.72967 rad/token**，'高频几乎没有'与实际数组不符"。出处：`[CMD] line 6767`（ext#12761）。[已验证：原文裁决] —— 对 KKT 交付物是**红线级提示**：不要用"高频没动/动了"的定性叙事替代频率-距离的定量约束。

12. **YaRN 身份混淆**。官方 cos/sin 幅度为 `a = 1 + .1·ln(s)`，logits 乘 `a²`；Y2 是变体，不能用它的零分证明官方方法上限。出处：`[CMD] line 6774`（ext#12763）。[已验证：原文] —— 与 memory 中 ramp=dim 空间线性、边界 [23,40]、attn_factor 作用 cos/sin 的记录一致。

---

## 4. 经验锚点（可直接进 F 的约束标定 / 对照基线）

出处统一：`codex_tools_b.txt [CMD] line 6762`（ext#12603-12616）与 `[CMD] line 6767`（ext#12760-12800），均引自 `paper-2027/HANDOFF.md`、`~/.codex/memories/MEMORY.md`、`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`、`ROPE_RESEARCH_FAILURE_REVIEW_20260907.md`、`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`。

- **151.9M 精确区间（同支持、纯 z）**：Cosh−FMRoPE NLL = `−0.28073 / −0.17599 / −0.14571`（512 / 1K / 2K），**3/3 seeds 偏向 Cosh**；target-matched 均值反向为 `+0.06032 / +0.22720 / +0.45959`。同支持冻结对照：**+59.92 RULER（OLMo）**、**+8.75（Qwen）**。[已验证：三项独立记忆源互相印证]
- **BM（OLMo-2-0425-1B-Instruct，静态 S4）**：16K **51.32%** vs MrPro 2.78%、MrUni 32.12%、官方 YaRN 6.94%；4K BM 81.81%。[已验证]
- **BM（Qwen2.5-3B-Instruct）**：32K **91.67%** vs MrPro 87.22%；**128K 70.83% vs MrPro 78.13%**（即 128K 子上**输**）。[已验证] —— 这是用户"BM lost the Qwen2.5-3B 128K subset"的出处；复盘明确要求"通过 MrRoPE 实际方法视角分析该结果，而不是假设理论优越或改进不可能"。
- **粗 ramp vs 细 derived**：本地 FineWeb 复算两者只差约 `.000511 / .000560` NLL；而**同端点 geometric 在 16K 上比 derived 差 4.455995**。→ 支持"粗分配结构有用"，未证明精细曲线必需。[已验证]
- **ALS 虚假收敛**：旧首轮 `2.848193371`（`previous=inf` 导致 `inf<=inf`）；修后 43 轮 `2.836061317`，独立显式旋转重算一致。示例 `src=[1,.4,.1]`、`dst=[.95,.16,.45]`… 出处 ext#12775-12779。[已验证]
- **j=24 累计压缩指数**：由 Mr 约 `0.0065` 变成提案约 `0.912`（说明改动不是围绕有效分配的小修补）。[已验证：原文]
- **Qwen 尺度构造五点 guard 只有 λ=0 通过**；原提案短 QA F1 0.425 / Native 0.498465 / 恢复 Mr 中段后 0.523214。[已验证：原文表]
- **OLMo E1**：Z 单证据 near 6/32、far 1/32；MrPro 0/32、0/32。[已验证：原文表]
- **KLD v2 结构**：Qwen3.5-0.8B，24 层（18 linear / 6 full attention），key/value heads 16 × width 128，最后一个 linear 层索引 **22**，layer 23 为 full attention；该层多加一个 FP32 状态 = `16*128*128*4 = 1,048,576` bytes/sequence。`beta = sigmoid(b)`、`g = −exp(A_log)*softplus(a+dt_bias)`、`D = exp(g)`。出处：`[CMD] line 6578`（ext#11901 段、ext#11792）。[已验证]
- **F7 判定阈值**：`beta=(0.9,0.95)`、weight decay 0.01、peak/minimum rates。出处：`[CMD] line 6619`（ext#12445）。[部分证据]

### 4.1 原生 sparse 选择器的实测对照（本线程自产）

`RESULT_20260908.md` 固定开发例（native Qwen2.5-3B，`niah_multiquery_32768_0`，四数字输出空间分隔）：
Dense `2608476 9403234 4105180 3192420`；RoPEMean `2608471 …`；Privileged source-page repair = Dense；同预算 wrong-page repair `2631010 …`；Exact block logmass = Dense；Exact block maximum = Dense；Quest64 参考 `26081234 94032010 4105180 3192420`；Native rotary-pair envelope `2608476 4102310 3192420 3192420`；同字节 random-pair envelope `2608476 4105180 3192420 3192420`；QuestSplit32 `2608476 9405180 4105180 3192420`；PostMetric4 = Dense；PreMetric4 = Dense；MatchedContiguous4 `2608476 9403234 3187540 4192730`。

冻结 32 例：**Dense 17/32，RoPEMean 0/32，Quest64 3/32，QuestSplit32 8/32**。
描述符字节（每 64-token KV page，D=128）：RoPEMean 512、Quest64 1024、QuestSplit32 2048。

QuestFine32 结果：**13/32** raw exact+EOS，**32/32 EOS**，`210.4937837831676` s，outputs SHA256 `a642b957a04671baf68db790cc8e55049b2bd31f90e9fe493496050570ff58d8`。
配对比较 `cross_run_comparison.json`：PostMetric4 3 胜 / 6 负 / 23 平，未校正 sign-test **p = 0.5078125**；PreMetric4 3/5/24，p = 0.7265625；Dense 7/3/22，p = 0.34375；QuestSplit32 1/6/25，p = 0.125。
`verify_generation_records.py`：mixture_frozen_01 224 行 / 50 正确，sha `216ea464d40dc0aca0193251c767302d4e92d40b3499340650a8b8ef678ab8db`；quest_fine_frozen_01 32 行 / 13 正确。
HANDOFF 最终有界结论：**"finished224 outputs: Dense17, Mean0, Quest643, QuestSplit328, PostMetric410, PreMetric411, count-matched-contiguous1 correct"**。
出处：`codex_tools_b.txt [CMD] line 6074`（ext#10463-10533）、`[CMD] line 6107`（ext#10420-10433）、`[CMD] line 6196`~`line 6205`（ext#10839-10872）、`[CMD] line 6220`（ext#10890），以及 HANDOFF 重写段。[已验证：原始计数与 SHA]

### 4.2 oracle_causal_02 诊断（唯一跑通的原生 envelope 对照）

`summarize_envelope.py envelope_causal_02 --reference support_ruler_02`：
- output_sha256 `d1a7cd94aab5d2ba7a50221632e920ea5744821e6c2af873c3b9cd6def511b90`
- 5 个方法各 n=1，raw_full_exact_and_eos **全为 0**，eos 全为 1（即 1 例全部答错）。
- 诊断（同一条失败 RoPEMean 轨迹，5 个位置 × 首/中/尾层，per-head source cells）：`eligible_source_pairs = 1200`、`source_max_rank_top16 = 57`（所有方法相同）；`source_selected`：RoPEMean 65、Quest 55、PairEnvelope 69、RandomPair 39、QuestSplit32 56；`inflations_mean`：Quest 25.302、PairEnvelope 25.131、RandomPair 25.656、QuestSplit32 21.955，`competitor_inflations_mean` 一律更高（26.80 / 26.38 / 27.37 / 23.25）。
- reference replay 校验：`same_prefix: true`、`raw_generated_ids_equal: true`。
- 文档自设边界："same failed RoPEMean query trajectory … these are correlated diagnostics, not independent task wins"。
出处：`codex_tools_b.txt [CMD] line 5792`（ext#8894-8984）。[已验证：数字为原文 JSON；n=1，统计上不可作能力结论——原文自己注明]

---

## 5. 子代理调度记录

来源：`codex_subagents.txt`（29 条事件，覆盖 JSONL 2006–6563）。

| 项 | 内容 |
|---|---|
| 子代理 1 | `agent_path: /root/mechanism_candidate`，`agent_thread_id: 01a081df-8ff7-7402-ada8-338367b2d63c` |
| 子代理 1 事件 | `started` @2006；`interacted` @2182 / 2295 / 2593 / 3010；`completed` @2068 / 2215 / 2344 / 2648 |
| 子代理 2 | `agent_path: /root/independent_method`，`agent_thread_id: 01a08316-2d5d-7963-bbf7-8c440a88da68` |
| 子代理 2 事件 | `started` @5081；`interacted` @5115 / 5196 / 5342 / 5445 / 5492 / 5647 / 5661 / 5718 / 5751 / 5779 / 5834 / 5880 / 5957 / 6426 / 6464 / 6515；`completed` @5173 / 5302 / 5414 / 5553 / 5732 / 5948 / 6003 / 6563 |

事实观察（本节全部为 [已验证]，直接来自 `codex_subagents.txt` 文本）：
1. **事件类型只有 `started` / `interacted` / `completed` 三种**。在整个 extract 中 grep 不到 `failed` / `interrupted` / `cancelled` / `error` 任何一种；**没有失败或中断记录**。这只能证明"记录里没有失败事件"，**不能证明"没有发生过失败"**——失败可能是以随后一条 `interacted` 的形式被主代理纠正的。
2. **同一个 `agent_thread_id` 上出现多个不同的 `completed` id**（子代理 1 有 4 个，子代理 2 有 8 个），且每个 `completed` 后面还继续出现 `interacted`。这说明这是**同一条被反复唤醒的长线程**，而不是 4 个或 8 个一次性子代理；`completed` 语义更接近"这一轮回答结束/交回主代理"，而非"代理终止"。
3. **调度呈"两阶段 + 长尾"结构**：子代理 1 覆盖 JSONL 2006–3010（早期机制候选筛查）；子代理 2 从 5081 起一直活到 6563（本段末尾），横跨 native sparse 实现、envelope、RULER、RefCarry 审计的**整个后半段**。
4. **主代理与子代理是并行独立工作的**：`paper-2027/HANDOFF.md` 的 2026-09-09 重写（`codex_tools_b.txt [CMD] line 6431`，ext#11280-11304）明确写 "One existing independent method agent is examining the actual compressed shared-KV reader and its nearest positional work; the primary agent is independently examining sparse indexing and alternative mechanisms."；同一段还写明 "No new GPU matrix, coefficients or model-switch queue is authorized."（即：**该阶段被显式禁止再开 GPU 矩阵**）。
5. **产出归属**：`METHOD_INDEPENDENT_20260908.md` 的内容（RMBC → RPEE）署名给 independent method agent；`ORBIT_DERIVATION.md` 的 64 点反例（`codex_tools_b.txt [CMD] line 5300` 附近，JSONL 7136–7155）摘要里挂在该 agent 名下。也就是说，**本线程最有价值的两个数学产出（RPEE 上界、轨道回归反例）来自子代理，而不是主代理**。

---

## 6. 未解问题与潜在冲突

### 6.1 与 `/analysis/unify_20260910/` 权威文档的关系

对 `INTEGRATION_20260910.md`、`NEXT_DERIVATION_KKT_PROBLEM.md`、`STARTING_POINT_YARN_VS_MRPRO.md`：**在 `codex_tools_b.txt` 中没有任何一处引用或提及这三个文件**（grep 无命中）。因此**不存在正面冲突**，但存在两处**需要人工确认的潜在张力**：

- **张力 A（框架层）**：本线程的整个研究对象是 sparse selector / 块描述符 / 读取接口，与"YaRN 型 vs EVQ 的频段分配"几乎不重叠。若权威文档把 KKT 问题限定在频率分配上，则本线程的多数内容属于**外围**；但它提供的 `TV/KL ↔ 保持率` 恒等式与 `rank([1^T;Φ])−1` 最小状态维数是**与频段无关的结构性约束**，可以直接进 F。出处：本线程无 KKT/EVQ 关键词（§0 的 grep 结果），恒等式见 §2.1/§2.3。
- **张力 B（一条可能被误用的"已解决问题"）**：`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md` 与 `ROPE_RESEARCH_FAILURE_REVIEW_20260907.md` 里有强烈的"尚无已证实的方法突破、超越 MrRoPE 的结果或可直接升格的论文主贡献"判断（`[CMD] line 6767`，ext#12645）。按用户红线，这类 VICTORY/已闭合 型结论**必须降级**——本 digest 只把它当作**证据记录**，不作为定论。

### 6.2 未解问题（本线程内明示或由证据直接产生）

1. **"1% 的频率改动能动多少距离"缺一张距离加权表**。复盘只给了 Δ=128 与 Δ=32768 两个点（0.00893 / 2.28665 rad），并明确"正确研究变量是距离加权下的可用频率改变量，不能以同一相对百分比代表全谱的同等干预"（ext#12711）。**这张表在本线程里没有被做出来**。这直接对应用户说的"压缩高频侧 log 间距、把间距让给低频侧"的预算理论——它是把 EVQ 与 KKT 接上的最短路径。[部分证据]
2. **`upper_bound_inflation` 的定义不在 extract 内**。汇总脚本（`summarize_envelope.py`，ext#8830-8832）读 `s['upper_bound_inflation']`，但生成它的代码在那段被 clip 的 `envelope_native.py` 里。因此 §4.2 里 "inflations_mean 25.3" 的**量纲与参照系不明**，不可直接解释为"上界虚高了多少"。[未解：extract 覆盖缺口]
3. **InfLLM-V2 粗归一化的因果判别实验未执行**。原文设计了判别测试（保留细分数与 reader，只把粗归一化换成精确细归一化），但明确"First qualify the original native kernel and baseline, then decide"（ext#8758-8763）——本轮只走到 "qualify" 阶段（minicpm_native.py / qualify_minicpm_kernel.py 落地、flash-attn 2.8.3 编译），**判别实验本身没跑**。[部分证据]
4. **RefCarry 全线的模型级证据为零**。`refcarry_audit/{README.md,results.json,checks.py}` 与 `aux_refcarry_math_checks.py` 明确自述 "These are finite algebraic witnesses, not pretrained-model capability evidence."（ext#12128-12132）。11 个数值见证全是有限维反例/恒等式，**没有一个是在真实 checkpoint 上的生成结果**。[已验证：文件自述]
5. **`count-matched-contiguous` 只有 1 正确**，且 `MatchedContiguous4` 在开发例上给出与 Dense 完全不同的输出（`3187540 4192730` 两个数字 vs Dense 的四个）——说明"距离/计数配对的错误块对照"本身就不稳，用它做因果归因需要额外设计。[部分证据：n 很小]
6. **本线程的所有 selector 结论 n 都极小**：oracle_causal_02 是 n=1；冻结集是 32 例；sign-test p 值全在 0.12–0.73 区间，**无一显著**。因此 §4.1 的排名只能当**方向性证据**，不能作为能力声明。[已验证：p 值与 n 为原文]

### 6.3 覆盖缺口（内容被 clip，不是文件不存在）

`codex_tools_b.txt` 中以下块被打上 `...[clipped N chars]`，全文不可得（需另读仓库/服务器副本）：
`envelope_native.py` 主体（clip 10123 / 10631 / 9497 chars）、`METHOD_INDEPENDENT_20260908.md`（clip 4288）、`ORACLE_FIRST_SHARED_ROUTING_PLAN_20260908.md`（clip 3097）、`RESULT_20260908.md`（clip 18629）、`ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md` 与 `EXACT_RANGE_151M_3SEED_RESULT_20260820.md`（line 6767 处 clip 29374 chars）、`summarize_candidate_screen` 相关（line 6762 处 clip 4241 chars）。
其中 **clip 29374 那一段（ext#12716 截断）包含了 "过渡应写成相邻 log-frequency 间距，而不是先选光滑曲线" 的完整推导**——这是与 EVQ gap-budget 最直接相关的一段，**必须在下一步从仓库原文补齐**。[未解]

---

## 7. 覆盖度（Coverage）

**已读**（`codex_tools_b.txt`，按 extract 行号）：
- ext#1–3400：命令头全扫（`awk` 提取 397 条 header 的第一行命令），确认 3449–6774 的命令流形态
- ext#4880–4913、#7697–7937、#8014–8116、#8200–8370、#8701–9000、#10200–12500（定点读）
- ext#11280–11430、#11537–11700、#11757–11800、#12019–12170、#12599–12808
- 全文 grep：`KKT|lagrang|dual`、`tau`、`upper_bound_inflation`、关键数字（`1.272983871`、`22.525078826`、`0.574743502`、`simplex_rank`、`group_logit_factorization`、`beta =`）

**已读**（`codex_filechanges.txt`）：全文 303 行 / 50 条 FileChange。
**已读**（`codex_subagents.txt`）：全文 101 行 / 29 条 SubAgentActivity。
**已读**（`extract_codex.py`）：全文 176 行（确认行号约定与只读保证）。

**未读 / 跳过**：
- `codex_tools_b.txt` ext#3400–4880、#5000–7690 的**大部分正文**（只读了命令头，未逐条读输出）。这约 4000 行主要是 rotary_budget CUDA 资格验证探针的输出、Qwen2.5/Qwen3.5/MiniCPM4.1 模型下载与 runtime 身份、activation audit 的原始日志——**判为低价值**（重复的 GPU 状态轮询与下载进度），但**若需要 MiniCPM 原生 kernel 的资格数字，此处需回读**。
- ext#9000–10200 未读（envelope 后续的 mixture / oracle_target / 冻结协议段落）。§4.1 的数字来自 ext#10200+ 的汇总，若需中间过程需回读。
- ext#12170–12599 未读（paper-2027 审读中间段）。
- 本线程之外的 4 个 context compaction（`extract_codex.py:94` 的 `compaction` view）未渲染。
- 服务器端（westc / AutoDL）的 `results/` 原始目录未直接访问；所有服务器侧数字均**经 extract 转述**。

**方法学限制**：本 digest 的所有数字都来自 extract 的**转述**，未回读 `.codex` 原始 JSONL，也未回读仓库/服务器原件。带 `[已验证]` 的含义是"该数字在 extract 文本中原样出现，且（在多数情况下）有第二个独立来源交叉印证"，**不等于**"我从原始数据复算过"。

---

## 8. 给 KKT 交付物的三点可执行提示（仅为挖掘结论，不是对上游 researcher 的指令）

1. **唯一可无痛搬运的部件是 §2.1 与 §2.3 的恒等式族**：群矩因子化（误差 2.1e-15）、`rank([1^T;Φ])−1` 最小状态维数、`TV = 1−r`、`D_KL = −log r`、`min_h r_h ≥ max(0, H r̄ − H + 1)`。它们**不含任何被红线的静态几何代理量**，且带精确反例。
2. **唯一可无痛搬运的经验约束是 §2.2 的 `2|sin(Δδν/2)|` 扰动律 + j=23 的 `ω≈0.0069783` 数字**。它把"高频能不能动"化成一个可以用距离做参数的定量问题——正是把 EVQ 的 gap-budget 接上 KKT 的最短路径。
3. **本线程最值得警惕的一课是 §3 第 2 条与第 11 条**：降低平均重构残差（−67%）反而造出虚拟极值（1.273 > 真实最大值 1）并反转排序。任何以"平均/能量/重构"为目标的新 F 分项，先要过这个反例。
