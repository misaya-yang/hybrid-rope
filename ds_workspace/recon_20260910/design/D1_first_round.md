# D1 — 首轮实验设计（三个）

依据：`/Users/yang/Downloads/rope_sparse_attention_kkt_research.md`（2026-09-10，612 行）。
地基：本文件所有能力与数字均来自仓库内实读或已标注的侦察文件（`analysis/route_read_20260910/recon/R1|R2|R3`）。
本机无 GPU，远端已关机（`docs/research/ACTIVE_RESEARCH_GOAL.md:54`：*"The user has closed the GPU..."*）。本文件不启动任何运行。

---

## §0 框架与被遵守的约束

### 0.1 文档对本轮的直接指令

文档 §7 开头定了顺序，不是我的选择：

> **首轮不需要追求更大的外推倍率。应先在模型已训练的长窗口内，用同等资源证明更好的信息利用。**

以及主结果形态（§7 末）：

> 相同计算／KV 预算 ⇒ 更高的窗口内最坏位置能力，且不损害必要次序能力。**平均分必须一起报告**，避免把两端也压低以后得到一条"更平的坏曲线"。

文档 §7 指定了三条核心行：位置搬移＋顺序对照、选择与读取拆分、架构条件化对照；压缩架构另须保留"原始证据已无法从压缩状态恢复"的对照。

### 0.2 第三条核心行本轮不能跑（诚实结论）

§7 第三行要求"同训练或适配预算，对比原生 PE、分支化 LeRoPE、联合位置／gain 规则"。这需要**一个可训练的、含 ≥2 种注意力运算的 LM**。侦察结论：

- 仓库全部从零训练史都是 **dense attention**；生产 harness 全部终止于 `F.scaled_dot_product_attention(..., is_causal=True)`（R2 §1）。
- 稀疏注意力**存在**，但只是**冻结预训练检查点上的推理期参考实现**：`experiments/nosa_position/runtime.py`（compress+select+local，`from_pretrained` 加载）、`experiments/native_sparse_position/*_native.py`（经 `ALL_ATTENTION_FUNCTIONS` hook Qwen2）、`experiments/deepseek_mini_position/runtime.py`（monkeypatch `DeepseekV4Attention.forward`）。`native_sparse_position/RESULT_20260908.md:15` 自陈 *"No training."*
- **per-layer local/global 调度在全仓库可运行代码中不存在**（R2 §2 "Not found"）。
- 150M 从零训练器是 `experiments/rotary_budget/train_budget.py`，其 line 46 导入的包在本 checkout 缺失，但完整存活于 `main_0726`（10 个文件的闭包，属**恢复**而非重写）。**但 150M 在 seq_len ≥ 4096 的吞吐全仓库无任何记录**（R2 §5 "Not recorded"）。
- R2 的判决：*"a sparse-attention 150M four-arm comparison cannot be run with what exists in-repo today... That is a build task of weeks, not a configuration task."*

结论：**第三行降级为本轮的设计项与预备项，不申请 GPU**。本文件的三个实验全部落在前两行 + §2/§5.2 的理论判据上，且全部是冻结模型、零权重的因果对照。

### 0.3 三个实验各自测文档的哪一条

| # | 实验 | 测的文档主张 | §7 行 |
|---|---|---|---|
| **D1** | 位置搬移＋顺序对照（原子臂） | §3.3 第四条 trade-off（位置不变性—顺序能力）是定律还是可选设计；§5.2"零频是原子不是无限慢的频率" | 第一行 |
| **D2** | 增量次序反演（multiset/端点/跨度/Σm/gain 全固定） | §2.1 的 O(N⁻²) 边界扰动机制；§2.4"不能延伸成最优分配定律" | 第一行的形状面 |
| **D3** | 选择／读取拆分＋压缩不可恢复对照 | §4.2 三种失败模式；§4.3 压缩的新位置问题；§7 第二行 | 第二行 |

### 0.4 已核对的硬约束（逐条遵守，违反即计划作废）

1. **最多 3 个实验，无隐藏多候选扫描，不重置既有预算** → 每个实验只声明**一个**自变量，臂数预先固定并在开跑前哈希。
2. **不得以新名字重启已否决路线**：
   - **V-E2**（`docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:178`，"18 样本/64 自由度 margin-gradient"）——本轮三个实验**都不做梯度步、不拟合、不设目标函数**；`b̃` 只是固定设计点上的读出。
   - **V-E4**（`analysis/unify_20260910/digests/digest_failure-records.md:203`，"用频率置换重新发现同一个 multiset 限制"）——D2 必须机器化地声明区别：V-E4 确立的是 **multiset** 限制，D2 **把 multiset 固定住**、只问排列。此声明做成**测试门**而非段落（仿 `experiments/joint_kkt_20260910/selftest.py:978` 在缺声明时令运行失败）。
   - **`sum cos` 的首零点只是诊断，不得进入目标**；**静态几何代理**（collision energy、coverage、smoothness、effective rank、MAE、orbit counts）不得作为选择器；**gain 是设计面不是定律** → 三个实验一律**冻结 gain = 1.138629436111989**（= 1 + 0.1·ln 4，MrPro 自身的值，见 `analysis/unify_20260910/tables/rebuild_ground_truth_tables.py:352` 的逐字规格串），不联合优化。
3. **必须能产出机制性主张，不能只是诊断** → 每个实验的成功/失败两枝都写成"某条机制被确立或被排除"。
4. **GPU 不得空转**：三个实验的 build 全部是 CPU（本机或服务器 CPU），GPU 会话只跑注册好的评测链。吞吐一律先跑**放弃式探针**（`docs/overview/RTX5090_BLACKWELL_PROFILE.md:175-178` 要求）。

### 0.5 一条影响设计的重要事实：Qwen 上没有未微调 YaRN 基线

`analysis/unify_20260910/tables/GROUND_README.md:74` 中 **YaRN_linear（官方）** 表是 `bit=True`（与部署载体逐位一致），但分数列是 **`—` / `—`**。即：**"各模型在自己的 2×/4× 上赢未微调 YaRN"这条既有成功标准，在 Qwen 上目前无法评估**（R1 §3.4，独立复核一致）。

因此本文件的判据**不使用**"赢未微调 YaRN"，改用两类可当场核验的量：(a) 该 harness 自身已展示过的分辨力；(b) 仓库内唯一一次真实 margin 测量所跨过的位移量。理由见各实验的阈值段。

---

# D1 — 位置搬移＋顺序对照：位置不变性可以用原子买到吗？

**成本最低、最先跑。** 它同时建起 D2、D3 都要用的读出仪，并先用小代价验证该仪器是否可用。

## 1. 问题与所测主张

**一句话**：把证据在存储位置上搬动（10%/50%/90%）而不改变答案，金标答案的连续 margin 会不会随存储位置变化；若会，能否用"把最慢频带置零"（§5.2 的零频原子）压低这种依赖，而**不**付出顺序能力的代价？

测的是 §3.3 第四条 trade-off（位置不变性—顺序能力）**是不是一条必须沿走的曲线**，还是 §4.1 所说"可以通过结构分工被改变"的性质；以及 §5.2 的 `-log 0 = ∞`：有限 log-频率区间上的连续密度**不能**表示真正的零频原子。

## 2. 精确构造

- **模型**：冻结 **Qwen2.5-3B-Instruct**，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`（与 `docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json` 的 `runtime.model_id`/`revision` 逐字一致）。**零权重更新。** 原生窗口 32768，故全部长度**窗内**（4096/16384/32768），符合 §7 首轮指令。经 `experiments/curvature_20260910/model.py:63 FrozenRoPE` 加载，唯一写入口 `install(values, gain, track_grad=False)`（`:111`）。
- **语料**：`scripts/experiments/niah_retention_canary.py` 的 canary 族。选它的理由是可验证的：它是仓库**唯一**记录证据精确 token 跨度的语料——`source_block=[position,len(needle)]`（记录构造处 `:63-66`），且 `position = round(depth*(size-len(needle)))` 是**算出来的不是猜出来的**，深度取 (.1,.5,.9)，长度取 (4096,16384,32768)，8 组 haystack × 2 world × 10 格 = **160 行**；脚本自带断言"长度不得超过该检查点的原生窗口"（`prepare()` 首段 `checkpoint_contract(a)['native_context_length']`）。RULER 行**不存**针位（R1/R3 一致），用 RULER 就得另写 span locator——本轮刻意避开。
- **顺序对照**：另设必须依赖次序的行集，取 `scripts/experiments/olmo_fast_screen/bench.py:10` 的 `FAMILIES = ('lookup','linked_lookup','latest_update','attribute_binding')` 中的 **`latest_update`**（"最后一次更新"正是 §3.3 第四条点名的次序敏感任务）。注意：本机现存的 prepared 镜像是 `results/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl`（350 行，任务为 niah_*/cwe/qa_2）与 `prepared_natural_extra_01/screen.jsonl`（211 行，multifieldqa_en/narrativeqa）——**都不含 `latest_update`**，故该行集需由 `prepare.py` 重新生成并加量（见 §7 待建项 4）。
- **臂（4 个，预先固定）**：
  | 臂 | 定义 |
  |---|---|
  | `native` | 检查点原生 64 频 |
  | `mrpro` | MrPro 表（当前零训练最优，36 行面板 32K/128K = 0.872222 / 0.78125） |
  | `atom_N24` | **主臂**：MrPro 的槽 40–63（m=1 的 24 槽）频率置 **ν = 0 精确零**，其余 40 槽与 gain 完全不变 |
  | `atom_N40` | **预注册的升级点**：仅当主臂效应 CI 排除 0 且顺序对照未退化时启用；范围为槽 24–63 |
- **精确自变量**：**接到零频原子上的槽数**（0 / 24，必要时 40）。不是频率重标定、不是 budget 搬运——`ν = 0` 不在闭包内。
- **固定不变的是什么**：输入 token id 逐行相同；总长、内容、干扰量、答案 token 宽度（canary 自己断言两 world 的 needle 宽度相等，`:59`）；模型/dtype/生成配置；**gain 冻结 1.138629**；槽 0–23（原生带）与槽 24–39（严格内区）逐槽不变；端点 m(23)=0、m(40)=1 不变。
- **种子（预先钉死）**：corpus 生成 rng = `random.Random(20260905)`（canary 自身，`:47`）；解码为 greedy（确定性）；paired-bootstrap 重采样种子 = **20260910**（沿用 `experiments/curvature_20260910/local_probe.py:70 mc_fisher(..., seed=20260910)` 的仓库既有约定）。三者在开跑前写入 receipt。

## 3. 测量与"为什么是这台仪器"

| 测量 | 类型 | 为什么非它不可 |
|---|---|---|
| `b̃_e` = `bound.smoothed_bound`（`experiments/joint_kkt_20260910/bound.py:103`）在答案位 | **连续、金标寻址、逐行** | 仓库**唯一**的金标连续量。它带**已证**的关系 `1{greedy ≠ ref} ≤ softplus(b_e)/log 2`（`bound.py:8-11`），且光滑代理被**精确夹住**：`b ≤ b̃ ≤ b + η`，η = `ETA_DEFAULT = 0.02`（`bound.py:51`）并在 `selftest.py` 里数值校验。160 行 × 3 深度的比较只有连续量才有统计力；二值命中率没有。**它不是 V-E2 路线**：无步长、无拟合、无目标函数，只是固定设计点上的读出，并按 `risk.py:361` 的既有做法在 receipt 报 V-E2 ratio。 |
| 二值任务分（官方 substring recall，`scripts/experiments/olmo_fast_screen/ruler_bench.py:5-9 score()`） | **二值** | 它就是 §7 的"最坏位置能力"轴的操作定义。**必须与 `b̃` 同时报**：仓库已有一次实测，二值分数跨过阈值时底层连续 margin 只移动约 1 nat，而"强非线性隐藏态交互"的解释被近似可加的 margin 分解否证（V-D19；`docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:129-142`）。只报二值正是造成那次误读的原因。 |
| 尾 512 token NLL @4096/16384/32768（16 docs） | **标量** | §7 明文要求平均分同报，防止"把两端也压低以后得到一条更平的坏曲线"。 |
| `output_kl` D_N(arm − native)（`model.py:188`） | **连续（约束侧）** | 分离"改动了原生计算"与"只改动了长程部分"。若 `atom_N24` 的 D_N ≈ `mrpro` 的而深度斜率不同，效应不是原生漂移假象。 |

**能分开文档三种失败模式中的哪些？诚实回答：一个都分不开。** 文档 §4.2 的三模式（位置相关选择偏差 / 跨层传播偏差 / 进入读取后未被利用）全是**路由**主张，而上述四台仪器全在输出侧。D1 分的是另一个划分：**(位置依赖 vs 顺序能力)**、**(连续 margin vs 阈值跨越)**、**(金标跨度 vs 平均 token)**、**(原生漂移 vs 仅长程)**。这正是 D3 存在的理由——不要把它说成它能做的事。

## 4. 算力与墙钟（每个数字都指名出处）

**已记录的锚点**（按 `RUNBOOK.md:38` 的告诫，凡来自 curvature 包的一律标为**投影**，该包**未在 GPU 上跑过任何东西**）：

- **A1** 32K 前向 = **4.1 s** 中位；128K = **33.9 s** 中位（max 100.7 s）；峰值 **21–27 GiB**。来源 `experiments/curvature_20260910/README.md:117-118`，该节自述读自"本机自己的 **640 条归档行**"。**投影。**
- **A2** 冻结 Qwen2.5-3B：48 行 @ 8192/16384/32768 用时 **297.4696 s**，峰值 **8.29 GiB**。来源 `docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json` → `cost.elapsed_seconds`、`cost.peak_allocated_bytes = 8898835456`。**实读。**⇒ **6.20 s/行**。
- **A3** harness A（"registration: NLL 16 docs × 3 lengths + 36 RULER rows"）≈ **25 min**；harness B（`long_eval` 4 docs × 2 lengths × 2 methods = 16 前向 @64K/128K）≈ **15 min**。来源 `experiments/curvature_20260910/README.md:131-133`。**投影。**
- **A4** canary 自带的墙钟上限 `--max-seconds` 默认 **900 s**（`scripts/experiments/niah_retention_canary.py:132`），生成预算 32（记录 `'generation_budget':32`）。**实读。**

**算术（D1）**：

| 步骤 | 前向数 | 依据 | 时间 |
|---|---|---|---|
| 放弃式吞吐探针（强制） | — | profile 文档 `:175-178` | 10 min |
| 仪器门 `bound_summary`（真空率） | 40 行 × 1 臂 = 40 | A1/A2 | 40 × 4.1 = 164 s ≈ 3 min |
| 教师强制 `b̃` margin（canary） | 160 行 × 3 臂 = 480 | A1（全按 32K 保守上界） | 480 × 4.1 = **1968 s = 33 min** |
| 顺序对照 margin | 48 行 × 3 臂 = 144 | A1 | 144 × 4.1 = 590 s ≈ **10 min** |
| 尾 512 NLL | 16 docs × 3 长度 × 4 表 = 192 | A2 实读 6.20 s/行 | 192 × 6.20 = 1190 s ≈ **20 min** |
| 二值任务分（生成） | canary+对照 = 208 行 × 4 表 | A4（全套 ≤15 min/表） | 4 × 15 = **60 min** |
| D_N / Fisher 对角（每槽 1 前向，仿 `model.py:219`） | 64 + 8 对 = 72 @32K | A1 | 72 × 4.1 = 295 s ≈ **5 min** |

**合计 ≈ 141 min ≈ 2.4 h，计 20% 开销 ⇒ ≈ 3 GPU-小时**，单卡。
**显存**：32K 处峰值 21–27 GiB（A1）。**与"最少超过 28gb"的用户规则（`analysis/kkt_20260910/mine2/ALL_USER_OPS.txt:48656`）存在张力**——27 < 28。反面规则在同一天被写下：`docs/overview/RTX5090_BLACKWELL_PROFILE.md:85-86`"以预计到完成的总 GPU 时间/成本选优；显存只需稳定可行，不设固定占用率或固定空余比例目标"，`:68`"低显存占用不能推出 GPU 没有吃满"，`:74-76`"不得为了显存数字好看而中断健康的 registered run"。**处理方式**：探针若测到 <28 GiB，则提高 micro-batch / 增大 `keep`，或改走 131073 token 的 `long_inputs` 路径，并**记录实际峰值**——不靠填充数字凑占用率。

## 5. 预注册的成败杀

**阈值不取整数，取该 harness 自己展示过的分辨力**：

- 已实测：16 docs 的 paired bootstrap 在 **16384** 分辨出均值差 **0.004249852** NLL（区间 `[0.00086487, ...]` 排除 0），而在 **8192** 对 **0.002482511** **分辨不出**（区间 `[-0.00116681, ...]` 含 0）。两处同源：`docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json` → `summary.16384` / `summary.8192` 的 `BM_minus_MrPro.paired_bootstrap_95_interval`。
  ⇒ **NLL 侧最小可报效应 = 0.0043**（n=16 上该 harness 史上展示过的最小可分辨量）。低于此一律写"未分辨"，**不写"无效应"**。
- margin 侧：仓库唯一一次真实 margin 测量中，**1.000 nat** 与 **1.125 nat** 的位移正好翻转了 greedy 判决（`docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:129-142`）。
  ⇒ **margin 侧最小可报效应 = 1.0 nat**，且 paired bootstrap（10,000 次，种子 20260910）区间须排除 0。

**主判据**：`Δ_depth = mean(b̃ @ depth .9) − mean(b̃ @ depth .1)`，逐臂计算（`b̃` 为 `max_wrong − right`，**越负越好**）。

- **成功**：`|Δ_depth(atom_N24)| ≤ |Δ_depth(mrpro)| − 1.0 nat`，配对 bootstrap CI 排除 0；**且**顺序对照（`latest_update`）的 margin 未退化超过同一 1.0 nat 带；**且** 32768 尾 NLL 未恶化超过 0.0043。
  → 买到了位置不变性而**没有**付顺序能力 ⇒ §3.3 第四条 trade-off 可被结构改变（§4.1 的主张成立），且 §5.2 的零频原子是模型真正想要的资源。
- **失败**：`|Δ_depth(atom_N24)|` 落在 `mrpro` 的 ±1.0 nat 内且 CI 含 0。
  → 该带上原子没有收益。结合 D_N 读数，这是**修正文档**而非否证参数化：§5.2 保持数学评注地位，本轮**不再授权任何 α₀ 臂**（这就是杀）。
- **杀 1（仪器门，先跑）**：`bound_summary`（`bound.py:197`）的 `vacuum_rate > 0.5` 或 `informative_0p1` 为假 ⇒ `b̃` 无信号。`bound.py:32-40` 明说此事并指出这**是首先该测的**：若如此，立即转 §4 的原生 NLL 表述，**不得**继续做深度扫描。
- **杀 2（一阶性门）**：`atom_N24` 的原生 KL 对数-对数指数落在 **[1.7, 2.3]** 之外 ⇒ 该臂已在冻结模型自身响应不再是一阶的区域，臂没在测模型。先例与措辞：`experiments/curvature_20260910/RUNBOOK.md:158-161`（"**Do not proceed past a failed s4. That stopping rule is pre-registered.**"）。

## 6. 反事实

- **成功** ⇒ 位置资源空间不是 log-频率线上的一段密度，而是"一个原子 + 一段密度"；**所有只参数化间距的方法（YaRN / MrRoPE / EVQ）都在一个排除了模型真正想要的点的集合上做优化**。设计后果是 §5.2 的 `μ_r = α_{0,r}δ_0 + (1−α_{0,r})μ_r^rotary`，并且**无需训练即可测得**。
- **失败** ⇒ 原子在最慢带上不是机制；文档 §5.2 的数学陈述正确但没有可测对应物，第二轮**不得**在 α₀ 上花预算。附带收益：`b̃` 仪器在该语料上被证可用或证不可用，这是 D2/D3 的前置条件，而代价只有 ~2 GPU-小时。

## 7. 必须先建什么（诚实尺寸）

1. **位置索引的 margin 仪器**（R1 G3，标注 BLOCKING）。`bound.py` 本身存在且已自测；缺的是**分箱 + 配对 bootstrap** 的联结层。**~1 个模块 + 测试。**
2. **原子表构造器**。注意：它**无法**住在 `tables.py` 的 m-坐标里——`m_j = ln(ω_j/ν_j)/ln S` 在 `ν=0` 处发散，**这正是 §5.2 的论点**。必须绕过 m-坐标，经 `FrozenRoPE.install`（`model.py:111`）直接写 `inv_freq`，并加守卫（现有 `install` 已校验 64 个有限非负值）。**~30 行 + 测试。** 这个"写不进去"本身就是 receipt。
3. **不变式门**（机器检查，非段落）：multiset 相等、端点相等、Σm 精确相等、gain 相等、原生/压缩带槽位身份；外加 V-E2/V-E4 声明的存在性检查，缺失即令运行失败（仿 `experiments/joint_kkt_20260910/selftest.py:978`）。**~1 个模块。**
4. **顺序对照顾虑行集**。`latest_update` **只以生成器代码存在**（`scripts/experiments/olmo_fast_screen/bench.py:10`）；本机两个 prepared 镜像都不含它。需重跑 `prepare.py` 并加量到 ~48 行。**小（CPU）。**
5. **canary 的两个输入本机不存在**：`--fineweb`（需带 `text`/`url` 的 parquet）与 `--native-pool`（json，其父目录含 `rows_path`）——`find` 无命中。本机 `data/fineweb_val_cache/val_fineweb-edu_5000000_strict.pt`（38 MB）是 `.pt` 缓存，**不是** canary 读的格式。要么恢复服务器路径，要么改写 loader。
6. **`FrozenRoPE` 与两个实验包都是未跟踪状态**（`git ls-files experiments/curvature_20260910/` 为空；R1 G6）。在其上盖东西之前先钉住（R1 建议：显式决定是 build on it 还是先 freeze）。

---

# D2 — 增量次序反演：形状有因果内容，还是只有预算？

**次低，紧接 D1（复用同一仪器与语料，可同一次开机）。**

## 1. 问题与所测主张

**一句话**：在**乘集、端点、跨度、Σm、gain 全部逐位固定**、只有 17 个增量的**排列**改变时，模型的窗内长程能力与原生漂移会不会变？

测 §2.1 给出的**具体机制**：MrPro 第一个内区槽的扰动是 `1 − S^{-2/[N(N+1)]} ≈ 2 log S/(N(N+1)) = O(N⁻²)`，而 YaRN 式是 `(1−1/S)/N = O(N⁻¹)`——"进入旋转算子的频率扰动在前端小了一个阶次"，因此 MrPro "更容易保住一部分已经训练好的局部运算"。反演臂把这个 O(N⁻¹) 的扰动放到高频边界，**同样的 multiset、同样的预算**。同时测 §2.4 的告诫（*"不能把'更保护前端、更压缩后端'无限延伸成最优分配定律"*）与 curvature 包已写死的读法（`README.md:218`：*"**this is a positive finding.** Budget alone determines the outcome, which means every geometric family is a budget choice wearing a shape"*）。

**为什么这是全仓库最干净的 shape-vs-budget 判别**：排列**自动**保持 Σm 与端点。仓库现有唯一的形状比较（MrPro Σm=29.333 vs MrUni Σm=32.000）把形状与总预算混在一起（R1 G7）；本设计由构造消除该混淆。且 R1 判定：**次序问题从未被测量过**——反向序列 `eps_q = 2(N+1−q)/(N(N+1))` 在整个语料中不存在（R1 Q2 结论 (c) NOT FOUND，三次独立搜索）。

## 2. 精确构造

- **模型/语料/种子**：与 D1 完全相同（冻结 Qwen2.5-3B-Instruct，canary + `latest_update` 对照，全部 ≤32768 窗内，种子同上）。
- **臂（4 个，预先固定并哈希）**：
  | 臂 | 定义 |
  |---|---|
  | `native` | 原生 64 频 |
  | `prog` | MrPro：`eps_q = 2q/306`，q=1..17（= 已部署表，`tables.py:71-83 m_mrpro`） |
  | `regr` | **主对照**：`eps_q = 2(18−q)/306`，q=1..17 |
  | `perm` | **单一**预注册随机排列，种子 20260910，抽取一次后连同 SHA-256 写入 receipt |
- **固定不变**（由构造保证并机器校验）：17 个增量的**乘集** `{2,4,…,34}/306` 完全相同；`Σ_{q=1}^{17} eps_q = 2·153/306 = 1`，故 **Σm = 29.3333 逐位相同**；端点 m(23)=0、m(40)=1 相同；跨度（槽 23→40）相同；**gain = 1.138629 冻结**；槽 0–23 与槽 40–63 的槽位身份与取值相同。**只有 eps 的排列变。**
- **精确自变量**：eps 序列的**排列**。
- **为什么不是隐藏扫描**：排列数在开跑前固定为 2 个有信息点（prog/regr 是次序轴的两端）+ 1 个次序不可知的对照（种子钉死、一次性抽取）。不增臂、不删臂。

## 3. 测量与"为什么是这台仪器"

与 D1 同一套四台仪器（`b̃` 连续、二值任务分、尾 NLL 标量、D_N 连续约束侧），此处只说明**新增的判别力**：

- **`b̃`（连续）** 在这里是主仪器，因为本实验的效应预期是**小**的（排列只改形状）：二值分数在 160 行上的分辨力是一个格 ≈ 0.00625，而 `b̃` 的分辨力是 nat 级连续量。同一次实测告诫（V-D19）适用。
- **D_N（连续约束侧）是因果条款的核心**：`regr` 与 `prog` 的 D_N（相对 native）若在 CI 内相同，则两臂付出的**原生漂移预算相同**，长程差异只能归因于排列。若 D_N 不同，则本实验退化为又一次 budget 比较，必须照此报告。这不是附加检查，是"因果"二字的兑现方式。
- **能分开三种失败模式中的哪些？同样一个都分不开**（同一理由：输出侧仪器）。它分的是 **形状 vs 预算**。

## 4. 算力与墙钟

同 D1 的锚点 A1–A4。前向数按 4 臂重算：

| 步骤 | 前向数 | 时间 |
|---|---|---|
| `b̃` margin（canary） | 160 × 4 = 640 | 640 × 4.1 = **44 min** |
| 顺序对照 margin | 48 × 4 = 192 | 192 × 4.1 = **13 min** |
| 尾 NLL | 16 × 3 × 4 = 192 | 192 × 6.20 = **20 min** |
| D_N × 3 非原生表 | 72 × 3 = 216 | 216 × 4.1 = **15 min** |
| 二值任务分（生成） | 208 × 4 表 | 4 × 15 = **60 min** |
| **窗内小计** | | **≈ 152 min ≈ 2.5 h** |
| 128K 长端 margin（若纳入） | 24 行 × 4 表 = 96 | 96 × 33.9 = **54 min** |

**窗内 ≈ 2.5 h；含 128K 长端 ≈ 3.5 h；若再走归档的 36 行 32K+128K 面板（A3，harness A ≈25 min/臂）另加 1.7 h ⇒ ≈ 5–6 GPU-小时。**

## 5. 预注册的成败杀

- **成功（形状有因果内容）**：`regr` 与 `prog` 的窗内 NLL 差 > **0.0043** 且配对 bootstrap CI 排除 0，**同时** D_N 差值 CI 含 0（预算相同）。
  → §2.1 的边界扰动机制获得支持：**排列本身**（不只是预算）携带因果内容；"更保护前端"不是措辞而是机制。
- **失败（只有预算）**：`regr` 与 `prog` 的差落在 ±0.0043 且 D_N 匹配。
  → curvature 包 `README.md:218` 预先写下的"正面结论"被证实：**每个几何族都是穿着形状的预算选择，带结构是端点的后果而非内部剖面的后果**。这**杀死"更好的内部剖面"这条纲领**——在花 D3 的 build 之前知道这件事，价值极高。
- **杀 1（不变式门，先跑，零前向）**：第 7 节第 3 项的不变式门 + V-E2/V-E4 声明检查任一失败 ⇒ 不跑任何前向。
- **杀 2**：任一臂的原生 KL 对数-对数指数落在 [1.7, 2.3] 之外 ⇒ 该臂不在一阶区，停止，不外延到 128K（同 D1 杀 2 的先例）。

## 6. 反事实

- **成功** ⇒ 位置分配的内部**次序**是可优化的自由度，"间距预算"有因果内容；EVQ 的 spacing-budget 资产（文档 §6.3）保住，且第二轮可以在**窗内**而不是外推上继续用它。
- **失败** ⇒ 内部剖面不携带信息，则文档 §6.3 的"间隔预算搬运"在现代架构上只剩端点/带宽两个自由度；D3 之后的一切位置分配工作必须改成"改端点与带宽"，不能改成"改剖面"。**这是一条能省下整轮预算的否证。**

## 7. 必须先建什么

1. **次序/排列表构造器 + 不变式门**（与 D1 第 7 节第 3 项同一件东西）。注意 `tables.py:213 from_eps` 已存在（施加 `ν_j → ν_j·exp(−d_j)` 的一个求解步），但**没有**"固定乘集重排"的入口，需要新写。**~1 个模块。**
2. 其余全部复用 D1 已建/已列项（margin 仪器、canary 语料、`latest_update` 行集、未跟踪包的钉扎）。

---

# D3 — 选择与读取拆分：中间信息是没进候选，还是进了没用？

**价值最高、成本最高、最后跑。它是唯一触及文档真正新主题（稀疏/压缩注意力）的实验。**

## 1. 问题与所测主张

**一句话**：在**相同的 top-k／块数／缓存预算**下，分别改变 **selector 的位置编码**与 **reader 的位置编码**，能不能把"中间信息失败"分解成 §4.2 的三种模式——没进入候选集（位置相关选择偏差）、进入但没被利用、被读了但没传播下去？

测 §4.2、§4.3（压缩的独立位置问题）与 §7 第二行 + 压缩不可恢复对照。

## 2. 精确构造

- **模型**：冻结 **Qwen2.5-3B-Instruct**（同上），挂一个**已存在的**推理期选择+压缩参考：
  - `experiments/nosa_position/runtime.py` — compress + top-k select + local window。`AttentionSettings`（`:40`）里 `dense: bool` 已是干净的逐模型开关；**关键**：RoPE 在 forward 内部、**选择之前**施加（`forward` 中 `q, k = apply_rope(...)` 早于 `selector(context)`，约 `:345`）。所以 **selector-PE 是一个可拨的开关**。
  - 或 `experiments/native_sparse_position/*_native.py`（对缓存前缀做 block-top-k，经 `ALL_ATTENTION_FUNCTIONS` 挂 Qwen2；其中 `oracle_native.py` 自述 *"explicitly NOT an efficient method"*，正合 oracle 对照之用）。注意这两个 runtime 的 RoPE 位置**不同**（前者在 attention forward 内、选择前；后者由宿主模型在接口前施加）——这正是 selector-PE 可被独立改变的证据。
- **2×2（4 格，全部预注册，不增不减）**，固定：选择块数/top-k、块大小、缓存预算、backbone、行集、gain。
  | | reader-PE = 现状（选中的 K/V 原样） | reader-PE = 读出携带相对位置（§4.4 形式） |
  |---|---|---|
  | **selector-PE = 带位置**（旋转后的 q/k 打分） | 格 A（现状） | 格 B |
  | **selector-PE = 纯内容**（未旋转的 q/k 打分） | 格 C | 格 D |
  格 B/D 的形式直接取文档 §4.4 的恒等式：`R(−p_i)Σ_j a_ij R(p_j) v_j = Σ_j a_ij R(p_j − p_i) v_j`——读出携带"被取内容与 query 的相对位置关系"。
- **压缩不可恢复对照（§7 强制，同批测，不是另加一臂）**：在同一行集上另测 (i) 仅压缩、禁止细粒度访问的格，与 (ii) 可完美访问原始 token 的 **oracle** 格。二者之差**上界**任何部署方法能达到的收益。若某位置臂在仅压缩格失败，失败应归因于"信息早已被丢掉"，**不是**位置；若 oracle 格也没有收益，则该行对本问题是**无信息的**，照此报告，不报成 null。
- **精确自变量**：位置由 selector 还是 reader 承载（4 格）；每格相对邻格只差**一个**位置旋钮，选择预算与缓存逐位相同。

## 3. 测量与"为什么是这台仪器"（这是三模式分离的核心）

| 测量 | 类型 | 分开 §4.2 的哪一模 |
|---|---|---|
| **选择集成员**（selector 自己记录的 `selected` 块） | **二值**，逐 (行, 层, query) | **没进入候选集**：query 处证据跨度所在块不在 `selected` 内 |
| **证据跨度注意力质量**，在选定的 (层, query) 上以 fp32 在前向内重算，带奇偶校验 | **连续/分布** | **进入但没被利用**：证据跨度质量 > 0，而 `b̃ ≥ 0`（有错 token 压过参考 token） |
| **`b̃_e`**（`bound.smoothed_bound`），答案位 | **连续、金标寻址** | **读了但没传播**：证据质量与逐层探针都正常，而输出端 `b̃` 差 |
| 二值任务分 | **二值** | §7 的 Pareto 纵轴 |

**硬约束（决定仪器形态）**：`attn_implementation="eager"` 与 `output_attentions=True` 在这些长度上**不可用**。Qwen2.5-3B 为 36 层、**16 Q heads / 2 KV heads**、head_dim 128（锚点：`analysis/unify_20260910/digests/digest_thread-0909-pm.md:99`）；HF eager 会把 KV 重复到 16 头，单层 bf16 分数张量在 32K 为 **32.0 GiB**、128K 为 **512.0 GiB**（R3 §3.4 推导）。故仪器必须**在层与 query 上选择性采样、并在前向内重算**，"先把注意力矩阵 dump 下来离线分析"这条路是死的。

**否决条必须机器化声明（R3 §6）**：
- **V-A7**（`digest_failure-records.md:143`）禁止"attention top-1 / 质量 ≥0.5 ⇒ 生成正确"；
- **V-A6**（`:142`）禁止"冻结态选择器分数 ⇒ 全模型任务改善"。
⇒ 预先钉死主张层级：**注意力质量只是描述性路由量**；它与答案正确性的联系**只能由单独测量的 `b̃` 承载，绝不断言**。检查形态照抄 `experiments/joint_kkt_20260910/risk.py:369-379`（对逐行目标直接 raise）与 `span_guard` 报出零空间维数、receipt 打 `development_only` 的既有做法。

## 4. 算力与墙钟

**锚点 A5（实读）**：注意力原语的唯一实测记录 = 层 0/18/35 × **2 行** × 16301 token → **46.6 s，峰值 7.88 GB**（`results/position_observability_20260908/psr_qwen25_activation_01/result.json`，R3 §3.1）。⇒ 6 个 (层,行) 单元 / 46.6 s ⇒ **7.8 s/(层·行) @16K**。

| 步骤 | 数量 | 时间 |
|---|---|---|
| 前向（每行每格一次） | 24 行 × 4 格 = 96 @≤32K | 96 × 4.1 = **7 min** |
| 选择性 (层,query) 重算 | 12 层 × 24 行 × 4 格 = 1152 单元 | 1152 × 7.8 = **2.5 h** |
| 二值任务分（生成） | 48 行 × 4 格 = 192 | 4 × 15 ≈ **60 min** |
| 仅压缩 / oracle 两格 | 追加 2 格 | **+1.3 h** |

**合计 ≈ 5–7 GPU-小时**（重算项随 层×query 网格线性增长，是真正的约束）。**显存** 21–27 GiB @32K（A1）。

## 5. 预注册的成败杀

本实验是**分解**，主判据是**比例**而非效应量：

- **成功**：在基线格下未通过顺序无关任务的行中，三模式各占比例可被分开，且 (a) **没有任何单一模式 ≥ 0.9**（若有一个占优，说明该语料里另两模式不存在——这是一个真实且可报告的发现，但要照实说，不能说成"分解成功"）；(b) 至少两个模式非空且 bootstrap CI 排除 0。
- **失败**：仪器自身的奇偶/量化下限超过效应 ⇒ 三模式分不开。此时**报出该下限**，并据此判定冻结检查点 + 参考选择器这条栈能否胜任路由问题。
- **杀 1（先跑，且在大 build 之前）**：`bound_summary` 的 `vacuum_rate > 0.5` ⇒ 输出侧读出无信号，**必须**先改走 §4 的原生 NLL 表述再谈注意力工作（`bound.py:32-40` 明确要求先测这个）。**这一条把 D3 的失败代价从"2 周 build + 8 GPU-小时"压到"~2 GPU-小时"，也是 D1 排在 D3 前面的核心理由。**
- **杀 2**：奇偶校验失败（`activation_audit.py:104` 的 `raise RuntimeError('Raw capture / rotary reconstruction parity failed')` 模式）⇒ 捕获的 Q/K 不是真正被用的那份，整条链作废，不许"近似通过"。

## 6. 反事实

- **成功** ⇒ 文档 §4.2 的三分解被实测，且**设计后果由比例直接给出**：以"没进入候选集"为主 ⇒ 修选择器的位置处理（文档 §5 表"内容选择／路由：检查无关存放位置是否改变选中机会"）；以"进入未利用"为主 ⇒ 修读出的关系信息（§4.4 的 value 路径）；以"传播"为主 ⇒ 改最后一层 RoPE 无关紧要，§4.2 的告诫本身成为结论。
- **失败** ⇒ 冻结 Qwen + 参考选择器这条栈无法分辨路由，则该栈就此退出该问题；第三行（架构条件化）若要上马，必须有**别的**理由，不能靠"反正冻结栈不行"。

## 7. 必须先建什么（这是三个里最大的 build）

1. **(层, query) 扫描驱动**：把 `experiments/native_sparse_position/activation_audit.py` 从硬写死的 **3 层 × 最后 8 个 query × 2 行 @16K**（`:85`、`:105`、`:131-132`）推广到声明的网格，**保留**其奇偶断言（`:104`）与 `ALL_ATTENTION_FUNCTIONS` 注册方式（`:99-129`）。R3 的判决：*"What is missing is the sweep, not the primitive"*——但 sweep 是新驱动，不是改配置。**中等。**
2. **证据跨度注意力质量的联结**：**没有任何现存代码对任何语料计算证据跨度质量**（R3 §3.2/§(c)3）。可**规避**：用 canary 已存的 `source_block`，而不是为 RULER 写 span locator（RULER 不存针位）。若非要用 RULER，则需一个"把金标字符串映回 `prompt_ids` 内 token 下标、且失败时报告而非猜测"的定位器——**非平凡 build**。
3. **4 格的位置旋钮接线**：selector-PE 在 `nosa_position/runtime.py` 里是**一行的位置问题**（旋转前后取打分源）；但 **reader-PE 不存在**——§4.4 式的输出端逆旋转在该文件中没有，必须新写。**中等。**
4. **机器化的 V-A7/V-A6 门** + `development_only` 打标 + 零空间维数上报（照 `risk.py:369-379` 与 `span_guard`）。**小。**
5. 复用：`FrozenRoPE`、`tables.py`、`mean_answer_nll`、`bound.py` 全部、`attn_hist.py:13 accumulate_distance_histogram`（按距离累积注意力质量的现成原语，目前只被一个单测练过）、canary 语料。**R3 §(c)7 的原话**：*"The gap is a span-labelled, per-(layer, query) attention readout joined to a teacher-forced margin — nothing less, and (for once) not much more."*

---

# §4 排序（按 价值/成本）与"只跑一个"的选择

| 序 | 实验 | build | GPU | 价值 | 比值 |
|---|---|---|---|---|---|
| **1** | **D1** 位置搬移＋顺序对照（原子臂） | ~天（仪器 + 构造器 + 门） | ~3 h | 高：测 §3.3 第四条 trade-off + §5.2 原子；**并建起 D2/D3 都要用的读出仪** | 最高 |
| **2** | **D2** 增量次序反演 | ~天（复用 D1 仪器与语料） | ~3–6 h | 高：全仓库最干净的 shape-vs-budget 判别；两个方向都能杀死一条纲领 | 次高 |
| **3** | **D3** 选择／读取拆分 + 压缩对照 | ~1–2 周（最大） | ~5–7 h | **最高**：文档真正的新主题（稀疏/压缩），三模式分解 | 最低（但价值最高） |

**若只能跑一个，我跑 D1。** 理由不是它最便宜，而是它是**唯一一个能在 ~2 GPU-小时内同时（a）验证读出仪、（b）检验一条承重假设、（c）为另外两个实验解锁前置条件**的实验。具体地：D3 价值最高，但它的成功完全依赖 `b̃` 在冻结 Qwen 上**非真空**（`bound.py:32-40`：若某些答案位上错 token 压过参考 token，该界的效力不超过平凡的"≤1"）。假如 `b̃` 在这套语料上是真空的，D3 要烧掉 1–2 周 build 加 5–7 GPU-小时才发现输出侧读出根本没信号；D1 用 ~2 GPU-小时把这件事先问掉（杀 1 就是这个门）。同时 D1 用 canary 的已存 `source_block` 绕开了仓库里唯一一个非平凡 build（span locator），使 D1 成为三者中**唯一一个不依赖任何未验证新组件**的实验。D2 紧随其后是因为它复用 D1 的全部资产、可以并入同一次开机、并且它的失败会**杀死一条纲领**（内部剖面无因果内容）——那种信息在花 D3 的钱之前拿到最值钱。

---

# §5 总算力估算

| 项 | GPU-小时 |
|---|---|
| D1（含强制放弃式探针） | ≈ 3 |
| D2（窗内 + 128K 长端；不含归档 36 行面板则 ≈3，含则 ≈6） | ≈ 3–6 |
| D3（4 格 + 仅压缩/oracle 两格） | ≈ 5–7 |
| **合计** | **≈ 11–16 GPU-小时** |

不包含各实验的 build 时间（CPU，本机或服务器 CPU）。口径说明：以上**不含**任何 150M 级训练；150M@≥4096 的吞吐全仓库无记录（A6/A7 只覆盖 50.1M@4096 与 151.9M@L=256，**不可外推**），故第三行（架构条件化）本轮**不报 GPU 预算**——无成本依据的数字就是编的。
**"GPU 不得空转"的满足方式**：三个实验的 build 全部 CPU；GPU 会话只跑预先注册的评测链，且每次开机的链路都足够长（3–7 h）以覆盖 2 分钟空转禁令所需的最小连续负载。

---

# §6 不存在、必须建的东西（直白清单）

1. **位置索引的 margin 仪器**（分箱 + 配对 bootstrap 联结 `bound.smoothed_bound`）。`bound.py` 存在且自测；联结层不存在。**BLOCKING（D1/D2）。**
2. **原子表构造器**——不能住在 m-坐标（`m = ln(ω/ν)/ln S` 在 ν=0 发散，这正是 §5.2 的论点），须直写 `inv_freq`。**D1。**
3. **次序/排列表构造器 + 不变式门**（含 V-E2/V-E4 的机器化声明检查）。`tables.py:213 from_eps` 存在，但没有"固定乘集重排"入口。**D2。**
4. **顺序敏感对照行集**：`latest_update` 只有生成器代码，本机 prepared 镜像不含它。**D1/D2。**
5. **canary 的语料输入**：`--fineweb` parquet 与 `--native-pool` json **本机 not found**。**D1/D2。**
6. **(层, query) 扫描驱动**（推广 `activation_audit.py`）。**D3，中等。**
7. **§4.4 式 reader-PE（输出端逆旋转）**：`nosa_position/runtime.py` 里**不存在**，须新写。**D3，中等。**
8. **证据跨度注意力质量的计算与联结**：无任何现存代码对任何语料计算它。**D3**（可用 canary 的 `source_block` 规避 span locator）。
9. **机器化的 V-A7/V-A6 门**。**D3。**
10. **可训练的稀疏/混合注意力层**：**不存在于任何规模**。稀疏注意力在仓库中只是冻结检查点上的推理期参考。**per-layer local/global 调度在可运行代码中完全不存在。** ⇒ §7 第三行本轮不能跑。
11. **150M 在 ≥4096 的吞吐记录**：无。⇒ 即使是 `main_0726` 上可恢复的 150M 训练器路径，在稀疏注意力需要的长度上也**没有成本依据**。
12. **`FrozenRoPE` 与两个实验包未跟踪**（`git ls-files` 为空）。在其上盖东西前先钉住或先冻结。

**未找到（明说，不猜）**：`deepseek_mini_position` 工作台的参数量/吞吐/显存记录（**not found**，且其模型是"downloaded random-init Mini"，`test_runtime.py:24` 在缺原模型时 SkipTest——**随机初始化模型不能承载"训练后的位置机制"这类行为主张**，故 §6.4 两条缺失导数的实验在本轮**被我排除**，理由记录在此）；本机无任何 `ruler_*.jsonl`；Qwen 面板输入行（`prepared_qwen3_01/screen.jsonl`）仅在服务器；Qwen 上无 YaRN 分数（§0.5）；`experiments/joint_kkt_20260910/` 无 README/RUNBOOK/driver.sh，且其代码三处引用的 `FINAL_PLAN.md` 在树中不存在。

---

# §7 我没有核验的东西（边界）

- A1、A3 出自 `experiments/curvature_20260910/README.md`，该包 `RUNBOOK.md:38` 自陈**未在 GPU 上跑过任何东西**，故一律按**投影**对待；只有 A2、A4、A5 与全部面板/NLL 数字是本机文件实读。
- 128K 处 3B 的峰值显存本机**无凭据**（`ROPE_QWEN3_BM_NLL_RESULT_20260908.json` 只到 32768）；21–27 GiB 是 A1 的投影。
- D3 的 7.8 s/(层·行)@16K 由 A5 的一个实测点（3 层 × 2 行 × 16301 token = 46.6 s）线性外推，**未实测**；开跑前须以放弃式探针取代该外推。
- 三个实验都只测**窗内**（Qwen2.5-3B 原生 32768）。128K 只作为 D2 的可选长端出现，且不在成功判据里。
