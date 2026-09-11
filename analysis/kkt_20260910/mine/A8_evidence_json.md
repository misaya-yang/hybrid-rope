# A8 — `docs/research/rope_allocation_20260910/evidence/` 六文件挖掘 digest

挖掘者：A8（只读）。日期：2026-09-10。
被读目录：`/Users/yang/projects/hybrid-rope/docs/research/rope_allocation_20260910/evidence/`
（CORPUS_SCOPE.json / full_model_response_native.jsonl / joint_mode_candidates.json /
project_fulltext_inventory.json / session_inventory.json / team_plan_original.json）

**纪律声明**：本文件是"证据—出处—等级"的清单，不是推导。所有数字均来自文件本体或我从文件本体重算；
凡重算，我在括号里写明是重算。[已验证] = 我在本机复算/直接读出且可复现；[部分证据] = 文件自述但未独立复核；
[假设] = 材料里的假设；[叙事-未验证] = 材料里的解释性说法，无载体。

**关键前置**：这六份文件是 2026-09-10 那批 **30 代理（20 sol + 10 astra）** 工作的归档语料，
归档清单 `../archive_manifest.json`（104 条）。归档方给它们的 kind 标签本身就是重要的证据分级：
- `full_model_response_native.jsonl` → `completed_native_gradient_measurement`
- `joint_mode_candidates.json` → `cpu_candidates_only`
- `session_inventory.json` / `project_fulltext_inventory.json` → `input_inventory_not_read_proof`（**目录不等于读过**）
- `team_plan_original.json` → `stale_historical_dispatch_plan`

---

## 0. 先放三条"同源互证"结论（后面每个文件都在验证它们）

这三条是本批文件相对当前权威文档最有价值的部分——**它们是可独立复算的坐标事实**，不是叙事。

| # | 事实 | 载体出处 | 权威文档互证 | 等级 |
|---|---|---|---|---|
| C1 | Qwen2.5-3B 原生表 = 几何表 `ω_j = b^(−j/64)`，`b = 10⁶`，64 槽 | `joint_mode_candidates.json` 的 `relation.native_clock`（我用它反解 θ 得 1000000.334，相对误差 3.3e-7） | `NEXT_DERIVATION_KKT_PROBLEM.md:21` 同式 | [已验证] |
| C2 | MrPro 的压缩指数 `m` 精确等于 **二次族** `m_q = q(q+1)/(N(N+1))`，`N=17`，`q=1..17` 覆盖槽 24..40，外侧取 0 / 1 | `joint_mode_candidates.json` 的 `source.mr_m_float64`（我按上式重算，max｜diff｜= **3.7e-8**，即 fp32 舍入量级） | `STARTING_POINT_YARN_VS_MRPRO.md:25` 给同式；`NEXT_DERIVATION_KKT_PROBLEM.md:29` 给 `m24=.0065…m39=.8889` 与我读出的 `0.0065359…0.8888889` 逐位吻合 | [已验证] |
| C3 | MrPro 的 `gain = 1 + 0.1·ln S`，S=4 ⇒ `1.138629436111989` | 同上 `source.mr_gain` / `table.gain` | `digests_codex/digest_transport-operator.md:193` 记同一合同 | [已验证] |

**Σm 的读法（红线相关）**：`source.mr_sum_m = 29.333333268998935`。**这不是守恒量**——
同一文件里 29 个候选的 `delta_sum_m ∈ [−0.01826, +0.00185]`（我逐条复算，与 `INTEGRATION_20260910.md:30`
引用值完全一致）。**"Σm 守恒"在本材料里是被直接证伪的**，与红线一致。

---

## 1. `full_model_response_native.jsonl`（8,673 B，4 行）

### 1.1 行结构与字段（抄录，非转述）

每行 8 个字段：
`row_id, prompt_sha256, answer_ids, loss, per_token_loss, gradient_log_period, seconds, peak_allocated`

- `answer_ids`：Qwen 分词器对 `' ' + ', '.join(row['references'])` 的编码（**不含 EOS**）。
- `per_token_loss`：与 answer 等长的逐 token CE。
- `gradient_log_period`：**长度恒为 64**的浮点向量，全部 4 行都是 64（我核过长度）。
- `prompt_sha256`：对 `prompt_ids` 列表做的 digest（`experiments/nongeometric_screen/worker.py:66-68` 有校验），
  即**输入 token 输入可与历史面板逐条对账**——这是这份 jsonl 最硬的溯源点。

### 1.2 生成器即协议（`../code/full_model_response.py`，逐行）

| 行 | 内容 | 含义 |
|---|---|---|
| 1-2 | docstring：`Whole-model signed frequency response; weights frozen, exact target answer CE. This measures a proposal direction. It is not a claim of successful extension.` | **文件自述=方向测量，不是扩展成功** |
| 11 | `w.apply({'table': w.tables['MrPro']})` | 起点是 **MrPro 表**，不是原生表 |
| 13-14 | `base = MrPro.values_float32`；`delta = nn.Parameter(zeros(64))` | 唯一可训参 = 64 维 δ，初值 0 |
| 19-27 | 覆写 `rotary_emb.forward`：`freq = base·exp(−δ)`；`phase = inv @ pos`；`emb = cat(phase,phase)`；`co = cos·gain, si = sin·gain` | **δ 就是 log-period**：`log P_j = δ_j + const`，故 `gradient_log_period[j] = ∂L/∂δ_j` |
| 24 | 相位计算在 `autocast(enabled=False)` 内、fp32 | 但 27 行把 cos/sin `.to(x.dtype)` = **bf16** 输出 |
| 29-32 | gradient checkpointing (non-reentrant) + `model.train()` + **dropout 断言为零** | 冻结态可复现 |
| 33 | `length_cap==32768 and task in (niah_multikey_2, niah_multiquery) and row_id.endswith(('_0','_1'))` | 4 行 = 32K 窗口下这两个任务的 `_0/_1` 两个 split |
| 46-48 | `logits_to_keep=len(ans)`；`CE(out, target, reduction='none').mean()`；`loss.backward()` | teacher-forced 全前缀反向 |
| 51 | `peak_allocated` 取自 `torch.cuda.max_memory_allocated()` | — |

模型侧（`experiments/nongeometric_screen/worker.py:88-92`）：
`Qwen2.5-3B-Instruct, dtype=bf16, attn_implementation='sdpa'(flash), self.model.requires_grad_(False)`；
`install_table`（同文件 39-50）要求 `rotary.rope_type == 'default'`，直接把 `inv_freq` 换成 64 槽表、
`attention_scaling = gain`。
**因此 `rot=model.model.rotary_emb` 是单一共享模块 ⇒ 这 64 维梯度是"同一 δ 施加到全部 36 层"的和梯度**
（层特异分配在本测量里没有信号）。模型身份出处：`docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md:75`
（`Qwen/Qwen2.5-3B-Instruct`，rev `aa8e72537993ba99e69dfaafa59ed015b17504d1`，36 层 / 16 Q / 2 KV / head_dim 128 / 64 旋转对 / Native 32768 / theta 1e6）。

### 1.3 4 行关键读数（我全量抄出 + 复算）

| row_id | loss | ‖g‖₂（重算） | ‖g‖₁（重算） | ｜g｜最大槽 | seconds | peak_allocated |
|---|---:|---:|---:|---|---:|---:|
| `niah_multikey_2_32768_0` | 0.013202 | **50.897** | 117.921 | 槽0 = −40.169 | 13.95 | 16,460,931,584 (15.33 GiB) |
| `niah_multikey_2_32768_1` | 0.110649 | **688.661** | 1433.110 | 槽0 = **−640.927** | 13.48 | 16,469,561,856 (15.34 GiB) |
| `niah_multiquery_32768_0` | 0.308742 | **70.667** | 221.745 | 槽6 = +47.414 | 13.44 | 同上 |
| `niah_multiquery_32768_1` | 0.474598 | **460.502** | 1164.139 | 槽1 = **−382.669** | 13.48 | 同上 |

- ‖g‖₂ 区间 **50.9 – 688.7**，与 `INTEGRATION_20260910.md:52` 引用值逐位一致 [已验证]。
- `loss` 与我用 `per_token_loss` 重算的均值逐位一致（4/4 行）[已验证]。
- 每行第 1 个答案 token 的 CE 是全场最大项：mk2_0 = 0.0536；mk2_1 = 0.8614；mq_0 = **4.2593**；
  mq_1 = **7.3610**。mq_1 行另有一个 6.7518 的 token。**其余 token 大量是 0 / 1e-6 / −0.0**
  （−0.0 说明 fp32 下 p 已饱和到 1.0）。
  ⇒ **"4 行测量" 的有效信息量实际上来自每行 1–2 个硬 token，不是 8/35 个 token 的平均。** [已验证-重算]

### 1.4 响应集中在哪些槽 —— 本文件最重要的一条

我按四个频段算了 ｜g｜₁ 占比（重算）：

| 段（零基槽） | mk2_0 | mk2_1 | mq_0 | mq_1 |
|---|---:|---:|---:|---:|
| 0–15 | **0.956** | **0.894** | **0.932** | **0.924** |
| 16–23 | 0.033 | 0.044 | 0.050 | 0.070 |
| 24–39（MrPro 过渡带） | 0.011 | **0.061** | 0.018 | 0.005 |
| 40–63（MrPro 已 ÷4 的尾） | **0.001** | **0.001** | **0.001** | **0.001** |

- 只看 0–6（被 MrPro 冻结、m=0 的快槽）也占 ｜g｜₁ 的 **69.2% / 79.2% / 83.9% / 73.6%**（重算）。
  ⇒ 支持 `INTEGRATION_20260910.md:52` 的"响应集中于快槽 0–6" [已验证-重算]；
  但该句后半"过渡槽 24–39 几乎无响应"对 mk2_1 行是 6.1%（含槽24 ≈ −14.2、槽27 ≈ −20.9、槽28 ≈ +13.0），
  说"几乎无"偏强，建议口径写成"≤6.1%"。
- **槽 40–63 在全部 4 行都只占 0.1%** ⇒ 在真实 32K 位置上，答案 CE 对"低频 ÷S 那一段"的
  一阶敏感度实质为零。这是给 KKT 问题的硬结构事实：**窗口内损失在低频尾上是退化的
  （一阶不可辨识）**，任何"窗口内损失 + 外推能力"的 KKT 系统，其低频侧的内点条件
  **不能由窗口内数据钉死**。[已验证-重算；这一条是本 digest 对 KKT 最直接的输入]
- 该量的**符号不稳定**：4 行 `Σ_j g_j`（即 ∂L/∂(总 log-period)）重算得 **−16.416 / −822.033 / +29.658 / −121.533**；
  64 槽中 4 行同号的只有 **12 槽**（其中槽 0–23 内只有 3 个：2、3、11）。
  ⇒ 想从这 4 行里读出一条"预算搬运方向"是**不被支持的**。[已验证-重算]

### 1.5 一阶损失预测（可复用的零件，我在本条内给出算式与结果）

因为 `δ_j = log P_j + const`（1.2 节已证），候选文件的 `delta_log_period = −log(ν_c/ν_M)`
正是 `Δδ`，于是**一阶预测** `ΔL ≈ gᵀ Δδ = Σ_j g_j·delta_log_period_j`。
`digests_codex/digest_astra-margin-lineage.md:91` 也是这么说的（"for contraction with parent gradient_log_period"）。
我对 29 个候选 × 4 行全算了（重算，脚本逻辑即上式）：

- 量级：**|ΔL| ≤ 3.35**，绝大多数 < 0.1（对比 loss 本身 0.013–0.47）。
- 唯一 4 行同号（全负 = 预测降损）的候选是 **`JointMode_d1_s24_25`**：
  ΔL = **−0.0546 / −0.3502 / −0.0472 / −0.2109**。
- 其余候选符号随行翻转（例：`d1_s27_28` = −0.013 / **−3.350** / −0.021 / +0.099）。
- **该预测的适用边界写死在文件里**：`qualification`（`joint_mode_candidates.py:138`）
  "Use actual whole-model long/native gradient sign only as a direction filter…；
  no linear extrapolation guarantee"；且候选的 `max_abs_delta_phase_at_128k` 高达 **58.31 rad**——
  一阶式在这些候选上**没有外推保证**。所以它是方向过滤器，不是 F 的分项。

### 1.6 协议与局限（本文件的自带边界 + 我补的）

**(a) 文件/代码自述的边界**
1. `full_model_response.py:2` — "不是扩展成功的断言"。归档方也标 `completed_native_gradient_measurement`，非 result。
2. `full_model_response.py:35` 的 manifest scope 原文：
   `Historical development examples; measurement, not independent holdout or success of a new allocation`
   —— **但 `response_manifest.json` 没有进归档**，归档里只有 4 行 jsonl。
   即"这 4 行是开发样本、非 holdout"目前只有代码 docstring + `digests_codex` 转述，**manifest 原件不在本地**。[部分证据]
3. `JointMode` 文件的 `limitations`（`joint_mode_candidates.py:139-143`）虽非本文件的，但同批：
   关系时钟正确 ≠ 该模式承载有用计算。

**(b) 我读出的协议局限（按重要性）**
1. **窗口 = 32768 = native 训练窗**。文件名 `_native` 与 1.2 节第 33 行的 `length_cap==32768` 都确认：
   **这不是外推测量**。它只对 KKT 的 `L_near` 一侧直接有用。
2. **起点是 MrPro，不是原生**：δ=0 处即 MrPro 点。所以 g 是 **MrPro 处**的窗口内梯度；
   要谈"原生附近"的边际代价需另测。[部分证据-由代码 11/13 行推定]
3. **无 EOS**：target = `ans`，不含 EOS。所以既不测"会不会停"，也不测 sol18/sol16 提案里的
   "answer+EOS CE"。`digests_codex/digest_transport-operator.md` §7 提议的答案+EOS 口径在这份材料里**不存在**。
4. **层不可分**：单一共享 `rotary_emb` ⇒ 只有全层和梯度。
5. **bf16 输出**：cos/sin 在 fp32 算完被 cast 到 bf16（第 27 行）。bf16 相对步长约 2⁻⁸ ≈ 3.9e-3，
   所以 Δδ ≲ 1e-3 量级的一阶预测落在量化台阶以下；Δδ ~ 0.1（即 29 个候选的量级）仍在其上。
   这条限制是"一阶预测能用到多小步长"的定量界。[已验证-由代码行推出，量化界为标准 bf16 值]
6. **只有 4 行、且 2 个任务**（niah_multikey_2 / niah_multiquery）。mk2 行 loss 已到 0.013/0.111 = CE 天花板，
   与其说是"检索成功"不如说是"这 4 行不区分能力"（`digest_transport-operator.md` §5(c) 说了同一件事）。
7. **128K 侧状态未知**：`INTEGRATION_20260910.md:52` 明确"128K response 终态 **UNKNOWN——使用前必须先收回执**"
   （progress 文档纪律）。本目录里**没有任何 128K 的 response 文件**。[已验证-缺文件]

---

## 2. `joint_mode_candidates.json`（298,810 B，29 个候选）

### 2.1 顶层结构（全文抄录键名）

```
status  = "CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION"
formula = "nu_c = nu_M + n * (n^T omega_native / 4 - n^T nu_M) / (n^T n)"
coordinate = "actual runtime inverse frequency; global64 slots; no sorting or clamping"
scale = 4.0 ;  transition_slots_zero_based = [24, 39] ;  orders = [1, 2]
candidate_count = 29 ;  valid_candidates = 29
source = {native_contract / mr_contract 的路径 + sha256(文件) + tensor sha256, mr_gain, mr_sum_m, mr_m_float64[64]}
generator_sha256 = 8d1f0a85…（= 归档的 joint_mode_candidates.py 的 sha256，我核过一致）
numpy_version = "2.4.1"
candidates[29] ;  qualification ;  limitations[5]
```

每个候选 17 个键：`name, status, table{values_float32[64], tensor_sha256, gain}, relation{...}, checks{...},
delta_frequency_float64, delta_frequency_float32, delta_log_period, delta_log_frequency, delta_m,
m_float64, sum_m, delta_sum_m, compression_box_0_to1, slots_faster_than_native, slots_faster_than_mr,
max_abs_delta_phase_at_128k, max_abs_delta_log_period`。

### 2.2 构造规则（`../code/joint_mode_candidates.py`）

- 18-22 行：`SCALE = 4.0`，`FIRST, LAST = 24, 39`；只允许改中段。
- 48-52 行 `project(ref, native, n)`：
  `target = (n @ native)/SCALE`；`delta_clock = target − (n @ reference)`；
  `candidate = reference + n · delta_clock/(n @ n)`。
  ⇒ **每个候选 = 把一条"低阶关系"的时钟 `nᵀν` 精确投影到 `nᵀω/4` 的最近点，只动 2 或 3 个相邻槽。**
- 63 行：两族 —— `order=1, pattern=[1,−1]`（起点 24..38，15 个）与
  `order=2, pattern=[1,−2,1]`（起点 24..37，14 个），共 **29**。
- 61 行：`source_m = −log(mr/native)/log(SCALE)`，即 `m_j = −log(ν_j^M/ω_j)/ln S`。
- 127-128 行：断言必须恰 29 个且 tensor_sha 互异（脚本级硬门）。
- 每个候选的 `checks` **9 项全 True**（我逐条核过 29/29）：`finite_positive, strictly_decreasing,
  crossing_left_indices=[], unchanged_outside_relation, endpoints_bitwise_equal, gain_exactly_preserved,
  exact_projection_identity_pass, exact_orthogonal_complement_pass, float32_relation_within_rounding_bound`。
  ⇒ **这是一个断言门控的精确构造族，不是启发式搜索**。[已验证]

### 2.3 三个例子（讲清构造）

| 量 | `JointMode_d1_s24_25` | `JointMode_d2_s26_27_28` | `JointMode_d1_s37_38` |
|---|---|---|---|
| order / slots / coef | 1 / [24,25] / [1,−1] | 2 / [26,27,28] / [1,−2,1] | 1 / [37,38] / [1,−1] |
| `native_clock` = nᵀω | 0.00109182950 | 0.00013766089 | 6.5978878e-05 |
| `mr_clock` = nᵀν_M | 0.00116262678 | 0.00015290221 | 3.8921557e-05 |
| `target_clock` = nᵀω/4 | 0.00027295738 | 3.4415221e-05 | 1.64947196e-05 |
| `mr_clock/native_clock` | **1.0648**（MrPro 让它**更快**） | 1.1107 | **0.5899**（MrPro 让它**更慢**） |
| `mr_effective_period_extension` = ω/ν 比 | 0.9391（周期被拉长 6.1%，**不是 4×**） | 0.9003 | 1.6952 |
| 实际改动（fp32） | 槽24 −4.4483e-4 / 槽25 +4.4483e-4 | 槽26 −1.975e-5、槽27 +3.95e-5、槽28 −1.975e-5 | 槽37 −1.121e-5 / 槽38 +1.121e-5 |
| `delta_log_period`（=Δδ） | 槽24 +0.0832、槽25 −0.0961 | +0.00573 / −0.01459 / +0.00959 | +0.08931 / −0.11463 |
| `delta_m`（Δ 压缩指数） | +0.06 / −0.0693 | +0.00413 / −0.01052 / +0.00691 | +0.06443 / −0.08269 |
| `sum_m` / `delta_sum_m` | 29.32402 / **−0.00931** | 29.33370 / **+0.000367** | 29.31507 / **−0.018265**（族内最负） |
| `compression_box_0_to1` | **false** | true | true |
| `slots_faster_than_native` / `_than_mr` | [25] / [25] | [] / [27] | [] / [38] |
| `max_abs_delta_phase_at_128k` | **58.31 rad** | 6.91 rad | 1.47 rad |

读法（三条）：
1. **零和只在"原始频率"坐标里成立**：`delta_frequency` 的非零项严格成对/成三和为零
   （`raw_frequency_sum_delta = 0.0`，`_float32` 版最大 2.3e-10）。所以**Σν 守恒**；
   但 `Σ log-period` / `Σm` 都**不守恒**（上表 delta_sum_m 一栏）。这就是
   `INTEGRATION_20260910.md:30` 的"守恒必须先点名坐标"的原始载体。[已验证]
2. **"加速到超过原生"只发生在 3 个 order-1 首段候选**（s24_25, s25_26, s26_27 的右槽
   超过原生速度 ⇒ `compression_box_0_to1 = false`）。我复算：`box=false` 恰 3/29，
   与 `digests_codex/digest_transport-operator.md` 的 "26/29 在盒内、3 对越界" 一致。[已验证]
3. **目标"×4 重定时"根本没被 MrPro 实现**：`mr_already_retimes_native_by4` 在 **0/29** 上为真
   （重算）；`mr_clock_relative_error_to_target ∈ [1.104, 3.756]`。
   ⇒ 在"相邻槽拍频"这个低阶关系坐标里，MrPro 的斜坡**几乎没有重定时任何东西**。[已验证]

### 2.4 族级读数（29 个候选的结构，按起点槽排列；全部重算）

`mr_effective_period_extension`（= `native_clock/mr_clock`，>1 表示 MrPro 已把关系时钟拉长）：

| order=1 起点 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ext | 0.939 | 0.925 | 0.920 | 0.925 | 0.939 | 0.964 | 0.999 | 1.046 | 1.106 | 1.181 | 1.274 | 1.388 | 1.526 | 1.695 | **1.901** |

| order=2 起点 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ext | 1.004 | 0.945 | 0.900 | 0.869 | 0.850 | **0.841** | 0.842 | 0.853 | 0.875 | 0.907 | 0.951 | 1.008 | 1.080 | 1.169 |

**这是一条干净的新事实**：以低阶关系时钟看 MrPro，
**order-1 在槽 ≈30/31 处穿越 1**（左段"更慢"、右段"更快"），**order-2 在槽 ≈35 处穿越 1**；
最深处（order-1 s38）ext = 1.901、目标却是 4.0。
这与 `STARTING_POINT_YARN_VS_MRPRO.md` §4（F5）的"交点随 S 移动"是**同一个对象在关系坐标下的表述**，
并且可以写成 KKT 可用的约束型：`nᵀν = nᵀω/S` 是**线性等式**（48-52 行的投影就是它）。
[已验证-重算]

### 2.5 局限

- 顶层 status 自述：`CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION`
  ⇒ 这一族**没有经过任何能力或角色资格认证**，是纯坐标合法点。
- `limitations`（`joint_mode_candidates.py:139-143`）：(i) 关系时钟正确 ≠ 该模式承载有用计算；
  (ii) 全模型里相位系数与上层隐状态会变；(iii) **零和 `n` 保 Σν，但 log-压缩指数之和不定**；
  (iv) fp32 部署破坏精确投影恒等式（有逐条报告的舍入界）；(v) 候选**刻意允许相对 MrPro 的加速**并报告越界。
- `max_abs_delta_phase_at_128k ∈ [0.34, 58.31]` rad ⇒ `qualification` 明写 **no linear extrapolation guarantee**。
- 归档 kind = `cpu_candidates_only`。

---

## 3. 30 代理：规划 vs 交付（`team_plan_original.json` / `CORPUS_SCOPE.json` / `session_inventory.json` / `project_fulltext_inventory.json`）

### 3.1 规划（`team_plan_original.json`，2,877 B，归档 kind = `stale_historical_dispatch_plan`）

- `requested: {gpt-5.6-sol: 20, gpt-6-astra: 10}`；`concurrency_limit_subagents: 8`。
- 30 个 agent 状态：**sol01–sol08 = `report_complete`；sol09/10/11 + astra01–astra05 = `started`（共 8 个）；其余 14 个 = `pending`**。
- **与 CORPUS_SCOPE 的关系（可对齐，非冲突）**：`started` 的正好 8 个 = 并发上限 8 =
  `CORPUS_SCOPE.current_team_sessions_in_snapshot = 8`。两份文件是**同一时刻的两种记法**：
  sol01–08 是更早完成的一批，其余是运行中的 8 席。[已验证-对齐]

### 3.2 语料范围（`CORPUS_SCOPE.json`，752 B）

```
sessions 38 ; raw_bytes 313,837,981
unique_dialogue_records 1530 ; unique_dialogue_characters 553,260
unique_tool_outputs 3727 ; unique_tool_characters 115,038,479 ; tool_parts 87
excluded_from_dialogue: duplicate event mirrors / system+developer scaffolding / private reasoning / binary payloads
full_dialogue_ingestion_assigned_to: "sol09-sol14"
tool_output_ingestion_status: "archived complete unique outputs, not yet model-ingested"
historical_sessions 30 ; current_team_sessions_in_snapshot 8
snapshot_note: 8 名研究员启动后抓拍；their startup messages are NOT independent historical evidence
```

**核对**：`session_inventory.json` 恰 **38** 条，`Σbytes = 313,837,981`（**逐位等于** `raw_bytes`）[已验证]。
日期直方图：2026/02/26、03/10、03/11、04/24、04/27、05/07、08/19 各 1；09/06 1；09/07 1；09/08 12；09/09 8；09/10 **9**。
⇒ `current_team_sessions_in_snapshot = 8` 对应 09/10 的 **07:40–07:41 那一簇 8 个 session**
（`…01a08b1e-be61`、`-ddf4`、`-040b`、`-2bf1`、`-6454`、`-8d94`、`-adb8`、`…01a08b1f-4776`），
第 9 个（`01a089e5-e681`，09/10 01:59）归入 historical 30。[已验证-重算]

**最大的一条覆盖缺口（红线相关）**：`tool_output_ingestion_status = not yet model-ingested` ——
**115,038,479 字符的工具输出（3,727 条，87 个 part）被归档但从未进入任何代理的上下文**。
所以"30 代理读过全项目"这个说法必须限定为：**读过 27.34 MB 项目文本 + 6 份对话切片，
基本没读 115 MB 工具输出**。

### 3.3 交付对照表（我逐项在本地文件系统核过存在性）

- 规划：30 席（sol20 + astra10）；`assignments/` 有 **30 份 + COMMON.md**。
- 原报告：`agents/` **28 份**（astra01–09、sol01–19）+ `recovered/` **2 份**（astra10、sol20）。
- 回执：`coverage/` **27 份**（+ `astra09_pages.json` 辅助件）。

| 组 | 人数 | 任务内容（`assignments/*.md` 的 YOUR TASK 原句摘要） | 原报告 | 回执 |
|---|---:|---|:--:|:--:|
| sol01–sol08 | 8 | 主纸/EVQ 推导/Pro 材料 3 组/失败记录/当前非几何实验/剩余评审 | 8/8 | 8/8 |
| sol09–sol14 | 6 | "Full failure-transcript audit shard k/6"，逐字加载分配的对话 JSONL + 全部唯一工具输出 | 6/6 | 6/6 |
| sol15–sol19 | 5 | 剩余外部评审+手稿+历史论文、规范 RoPE 库/训练/相位源码、分析+scale-transport、剩余实验实现、历史 .agents 报告 | 5/5 | **4/5（缺 sol18）** |
| sol20 | 1 | 剩余项目文档 + 研究回顾 + 全部支撑源 | **0（仅 `recovered/sol20.md`，1,426 B）** | **0** |
| astra01–astra09 | 9 | 变分统一 / 有限窗非局部 EVQ / 标签保持冻结传输 / 6Pro 全行标定 / 混合频率 softmax / 对抗证明审计 / 训练 vs 冻结 / 决策论 margin / 独立搜索 | 9/9 | 9/9 |
| astra10 | 1 | 独立综合与证伪，把所有 30 代理证据收敛成最强理论 | **0（仅 `recovered/astra10.md`，2,294 B）** | **0** |

**有 / 无回执的精确名单**（`archive_manifest.json` 的 `missing_original_reports` / `missing_original_read_receipts` 字段互证）：
- `missing_original_reports = ["sol20", "astra10"]`
- `missing_original_read_receipts = ["sol18", "sol20", "astra10"]`
- `archive_status = "28_original_reports_plus_2_labeled_recoveries"`

`recovered/*.md` 头部自述（**重要，别把恢复稿当原报告**）：
"原综合报告和阅读回执未完成落盘。本文件由主任务根据已回传内容整理，**不是该代理的原报告**，也不是新一轮代理工作。"
其中 `recovered/astra10.md` 保留的唯一实质点是数学条件（`[C⁻¹h]_+` 不能替代 active-set 求解等）；
`recovered/sol20.md` 只有 OLMo-2-0425-1B 16K 350 条的一张表（7 任务×2 臂），**不是 sol20 自己算的**，
是主任务对既有结果 JSON 摘要字段的复核。[部分证据]

**回执里的省略（诚实记录）**：
- `astra03_coverage.json`：`omissions = ["MrRoPE primary full paper（astra01/root 负责）",
  "Underlying full-model Smooth/Mr generation JSONL（只读了开发摘要，未复审原始生成）"]`
- `astra04_coverage.json`：`omissions = ["remote native_full_rows tensors 未下载",
  "未查阅外部原始论文"]`
- 其余 25 份回执 `omissions` 为空。

### 3.4 语料切分方式：**按字节预算切，不是按语义切**（我重算出来的）

`project_fulltext_inventory.json` = 1,747 条 `{path, bytes, assigned_to}`，`Σbytes = 27,339,984`（27.34 MB），
30 席**互斥划分**（我按 `assigned_to` 聚合，无重叠）。每席字节数：

| 席 | 文件数 | 字节 | 席 | 文件数 | 字节 |
|---|---:|---:|---|---:|---:|
| sol01 | 3 | 68,828 | astra01 | 89 | 1,107,633 |
| sol02 | 4 | 50,943 | astra02 | 108 | 1,107,657 |
| sol03 | 3 | 197,541 | astra03 | 44 | 1,107,664 |
| sol04 | 3 | 155,138 | astra04 | 102 | 1,107,653 |
| sol05 | 5 | 139,364 | astra05 | 110 | 1,107,664 |
| sol06 | 7 | 83,963 | astra06 | 18 | 1,110,528 |
| sol07 | 38 | 271,166 | astra07 | 90 | 1,107,652 |
| sol08 | 25 | 448,727 | astra08 | 104 | 1,107,665 |
| sol09 | 115 | 1,303,413 | astra09 | 6 | 1,120,284 |
| sol10 | 116 | 1,451,784 | astra10 | 6 | 1,112,625 |
| sol11 | 117 | 1,323,640 | sol15 | 48 | 1,107,672 |
| sol12 | 117 | 1,376,268 | sol16 | 15 | 1,107,653 |
| sol13 | 116 | 1,340,601 | sol17 | 100 | 1,107,668 |
| sol14 | 117 | 1,339,090 | sol18 | 101 | 1,107,673 |
| | | | sol19 | 18 | 1,115,850 |
| | | | sol20 | 2 | 1,145,977 |

**读法**：astra01–08 / sol15–18 这一大簇的字节数被钉在 **≈1.1076 MB**（差 <0.3%），
而文件数在 **6–110** 之间摆动 ⇒ 切分器按**字节预算填装**，不按文档主题聚簇。
⇒ **30 代理的"覆盖"是字节覆盖率，不是主题覆盖率**；任何"某代理读全了某专题"的说法都不能从这份 inventory 推出。
同时 `kind = input_inventory_not_read_proof` 已明写：**目录不等于读过**。[已验证-重算]

### 3.5 口径不一致（不是矛盾，但会误读）

1. **文件计数基准不同**：inventory 给 sol12 = 117 个文件；`coverage/sol12_coverage.json` 自报
   `assigned_paths_total = 121`、`assigned_bytes_total = 1,479,196`（含对话流与追加文件）。
   sol14 同理（120 vs 117）。**引用"每人读了几个文件"必须说明口径。** [已验证]
2. **对话切片记录数只有 4/6 份可核**：sol09=136、sol10=397、sol12=257、sol14=312 条；
   sol11 / sol13 的回执没有给出记录数。`CORPUS_SCOPE` 说唯一对话记录 = 1,530 条，
   可核的 4 片合计 1,102 ⇒ 剩余 428 条落在 sol11/sol13 两片（数量级自洽，但**未独立核到**）。[部分证据]
3. **9 个 astra + sol15–18 的字节数异常一致**（3.4 节）本身是一个"必须解释的发现"：
   它意味着这些代理的**上下文是等长的、内容随机的字节块**，其报告之间的差异更多来自模型而非材料。

---

## 4. 矛盾与冲突清单

### 4.1 与权威文档的冲突

| # | 冲突点 | 本材料出处 | 权威文档出处 | 判断 |
|---|---|---|---|---|
| D1 | `digests_codex/digest_transport-operator.md:199` 写 "MrPro's effective period extension is only `mr_clock/native-clock ≈ 0.85–1.00×`" | 我重算全族：`mr_effective_period_extension = native/mr ∈ [0.841, 1.901]`，`mr_clock/native ∈ [0.526, 1.189]`（还有**方向反了**：该句把 native/mr 写成 mr/native） | `NEXT_DERIVATION_KKT_PROBLEM.md`、`STARTING_POINT_YARN_VS_MRPRO.md` §4 未给此数 | **digest 错**：区间过窄且比值倒置。真值应以 JSON 的 `relation` 字段为准（题 0/C 表外的第 4 条可复算事实） |
| D2 | `INTEGRATION_20260910.md:52` "过渡槽 24–39 几乎无响应" | 4 行占 ｜g｜₁ 比例 = 1.1% / **6.1%** / 1.8% / 0.5% | 同左 | **口径过强**：对 mk2_1 行 6.1%（含槽24/27/28 三项 >8）。建议改写为 "≤6.1%，且各段符号不稳定" |
| D3 | `INTEGRATION_20260910.md:52` "响应集中于被 MrPro 面冻结的快槽 0–6" | 重算 0–6 段占 ｜g｜₁ = 83.9% / 79.2% / 69.2% / 73.6% | 同左 | **一致**（虽"冻结"一词须理解为 m=0 即表值等于原生） |
| D4 | `NEXT_DERIVATION_KKT_PROBLEM.md:29` "MrPro m24=.0065…m39=.8889" | JSON `source.mr_m_float64` 读出同值（且给出完整公式 `q(q+1)/(N(N+1))`, N=17） | 同左 | **一致，且本材料把出处补全为可执行公式** |
| D5 | `INTEGRATION_20260910.md:30` "零和频移 ΔΣm≠0，joint_mode 实测 −0.01826…+0.00185" | 我逐条重算 `delta_sum_m`，极值与之逐位一致 | 同左 | **一致** |
| D6 | `archive_manifest.json` 说归档是"available 30-agent outputs"，`CORPUS_SCOPE` 说 `full_dialogue_ingestion_assigned_to = sol09-sol14` | 两份 recovered 文档明说"不是该代理的原报告" | `INTEGRATION_20260910.md:133` 记 "sol20/astra10 回传未落盘=整合席位" | **不冲突，但引用时须写"28 原报告 + 2 恢复稿"**，不得说成 30 份原报告 |

### 4.2 材料内部的口径不一致

- **窗口命名**：`joint_mode_candidates.json` 的 `transition_slots_zero_based = [24, 39]`（**被改动的槽**）vs
  KKT 文档 §1.3 的 Δ 索引 `(Δ₂₃,…,Δ₃₉)`（**17 个 gap**，`ΣΔ = 1`）。两者相容但不是同一个下标集，
  引用时若只写"[24,39] 的 17 槽"会自相矛盾（24..39 只有 16 槽）。[已验证]
- **m 公式的作用域**：`q(q+1)/(N(N+1))` 在槽 24..**40** 上取 q=1..17 才闭合（槽 40 得 m=1），
  而文件把"过渡带"写成到 39 为止。**范围 24..40 是我复算出来的，文件没写。** [已验证-重算]

---

## 5. 死路（本材料直接/间接证伪，不得再试）

1. **用静态几何/时钟量做候选选择子**。`joint_mode_candidates.py:138` 的 `qualification` 原文：
   "Not selected by clocks, amplitudes, geometry or native gradients alone. Use actual whole-model
   long/native gradient sign only as a **direction filter**, then exact finite full-model losses and
   generated task endpoints."（与 `INTEGRATION_20260910.md:84` 的"几何-无符号类（全灭）"同向）
2. **"Σm 守恒"/"水床守恒"作为未命名坐标的论证**。JSON 自带的 `delta_sum_m ≠ 0`（`limitations` 第 3 条明写
   "sum of log-compression exponents is not fixed"）直接否掉 Σm 不变式。
3. **把 29 个零和候选当"已按 ×4 重定时"**。`mr_already_retimes_native_by4` = false **29/29**；
   `mr_clock_relative_error_to_target` 1.10–3.76。⇒ "MrPro 已经完成了中段关系时钟的 ×4 搬运"是**错的**。
4. **线性/一阶外推到 128K**。`qualification` 末句 + `max_abs_delta_phase_at_128k` 最大 **58.31 rad**。
   一阶预测（§1.5）只能做方向过滤。
5. **用 32K 行（无论拉伸与否）去校 128K 过渡带**。4 行实测：过渡带 ｜g｜₁ 占比 ≤6.1%、
   尾段 0.1%。`digests_codex/digest_transport-operator.md` 与 `INTEGRATION_20260910.md:103`（FLAG-5）
   都据此判 "拉伸行 = 方向发生器/oracle 上限，模型级判定必须真实连续 128K"。[已验证-重算]
6. **把"4 行 32K 梯度"当成"低频/远处的边际代价"**。第 5 条的直接推论：槽 40–63 的一阶敏感度 = 0.1%，
   **窗口内损失在低频尾上是退化方向**。
7. **把 `tool_output_ingestion_status` 读成"工具输出已被吸收"**。115 MB 未入模型上下文；
   而 4 行梯度、29 个候选都大量依赖历史工具输出里的面板数字。**任何以"全量证据已读"为前提的结论都要降级。**
8. **把 `recovered/sol20.md`、`recovered/astra10.md` 当原代理结论引用**。

---

## 6. 未解问题（材料自带的我未闭口处）

1. `response_manifest.json`（含 `weights_updated / frequency_updated / scope` 原文）**未归档**——
   "4 行是开发样本、非 holdout"目前只有代码 docstring 与 digest 转述。要拿回执必须先回到
   `.agents/rope_unification_20260910/` 的服务器副本或 codex 会话。
2. **128K response 是否存在、是否跑完，未知**（`INTEGRATION_20260910.md:52` 已把它列为 UNKNOWN）。
   本目录 0 个 128K 文件。→ 开工第一件事是收这份回执。
3. **layer-wise 梯度不可得**：共享 `rotary_emb` ⇒ 只有全层和梯度。若 KKT 解需要"哪些层"，
   本测量给不出信号，必须改测量协议（per-layer δ）。
4. **EOS/终止口径缺失**：target 不含 EOS；与 sol18 协议里的 "answer+EOS CE" 不可比。
5. **一阶预测的可用下限未定标**：bf16 输出把 Δδ ≲ 1e-3 的预测埋进量化噪声；这个阈值
   需要一次 fp32 对照才能钉死（本材料没有 fp32 对照）。
6. **29 个候选从未跑过模型**（归档 kind = `cpu_candidates_only`；`archive_manifest.json` notes:
   "No new Qwen allocation was selected or validated from the 29 CPU-created joint-mode tables"）。
   其中 `JointMode_d1_s24_25` 是唯一个一阶预测 4 行同号（全负）的候选，且**恰好是唯一越出
   [0,1] 压缩盒的三个候选之一**——"一阶最有利"与"违反已知可行域"重合，这个张力没人解决。
7. **字节预算切分 vs 语义切分**：astra01–08 / sol15–18 拿到 ≈1.1076 MB 的等长随机块（§3.4）。
   这些报告的**独立性因此可疑**（材料重叠、边界随机），未有人做过重复性检查。

---

## 7. 覆盖度

**读完的**（全量，逐字节）：
- `CORPUS_SCOPE.json`（752 B）、`team_plan_original.json`（2,877 B）、`full_model_response_native.jsonl`
  （8,673 B，4/4 行全字段抄出并复算）、`session_inventory.json`（8,583 B，38 条全表）、
  `joint_mode_candidates.json` 的**全部顶层元数据 + 全部 29 个候选的标量字段**
  （`relation` / `checks` / `sum_m` / `delta_sum_m` / `box` / `slots_faster_*` / `max_abs_delta_phase_at_128k`
  等，29/29 逐条读；`table.values_float32` 与 `delta_frequency_*` 只对 3 个样例全读，
  其余按非零项提取，未逐槽打印 64 长数组 × 29）。

**为解释上述文件而额外读的**（均在仓库内，只读）：
- `docs/research/rope_allocation_20260910/code/full_model_response.py`（全文 57 行）
- `docs/research/rope_allocation_20260910/code/joint_mode_candidates.py`（全文 162 行）
- `docs/research/rope_allocation_20260910/archive_manifest.json`（104 条 entries 全表 + 顶层元数据）
- `experiments/nongeometric_screen/worker.py`（1–140 行）
- `docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`（第 75 行等关键行）
- `docs/research/rope_allocation_20260910/assignments/COMMON.md` + 全部 30 份 assignments 的 YOUR TASK 句
- `docs/research/rope_allocation_20260910/coverage/*_coverage.json`（27 份的标量字段 + 3 份全文结构）
- `docs/research/rope_allocation_20260910/recovered/{astra10,sol20}.md`（全文）
- `analysis/unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md`（全文 116 行）、
  `NEXT_DERIVATION_KKT_PROBLEM.md`（1–110 行）、`INTEGRATION_20260910.md`（关键行 30 / 52 / 84 / 103 / 133）、
  `digests_codex/digest_transport-operator.md`（§4–§7，190–300 行）、`digests_codex/digest_astra-margin-lineage.md`（第 91 行）

**明确没读 / 跳过**：
- `project_fulltext_inventory.json` 的 1,747 条**逐条路径**——只做聚合统计（按 agent / 字节 / 前缀分布）与
  极值抽查（最大 8 个文件），未逐条浏览。
- `joint_mode_candidates.json` 的 29×64 长数组未逐槽打印（只对比了 3 个样例的完整 64 槽 + 全族的非零项位置）。
- 本目录外的 30 份 `agents/*.md` 报告正文（约 46 万字节）——那是 A1–A6 的范围，本 digest 只用了它们的
  标题行与 `archive_manifest` 元数据。
- 任何 `~/.codex` 路径（严格只读且未访问）。
- 未跑任何模型 / GPU；本 digest 的全部数字来自文件读出或纯 CPU 复算（numpy 2.4.1 语义）。
