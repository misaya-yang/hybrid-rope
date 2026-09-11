# T5 — `codex_reasoning.txt` 挖掘报告：推理清单中的数学推导与建模尝试

任务来源：workflow-3 / T5。目标文件：`/Users/yang/projects/hybrid-rope/analysis/kkt_20260910/mine/extracts/codex_reasoning.txt`（146KB，5287 行）。
权威对照：`analysis/unify_20260910/{INTEGRATION_20260910.md, NEXT_DERIVATION_KKT_PROBLEM.md, STARTING_POINT_YARN_VS_MRPRO.md}`。
写作纪律：只读；每条带出处；证据分级标 `[已验证]` / `[部分证据]` / `[假设]` / `[叙事-未验证]`。静态几何代理量不得作为 F 分项（红线），本文只记录"某条推导曾是几何量"这一事实，不据此推荐。

---

## 0. 最重要的发现（先读这一条）

**`codex_reasoning.txt` 里没有任何推导正文。它只有 Codex 自动生成的推理"小标题"（headline）。**

证据链：

1. 抽取脚本 `analysis/kkt_20260910/extract_codex.py` 第 118–125 行的 `reasoning` 视图逻辑是 `s = it.get("summary_text") or it.get("raw_content")`，即**优先取 `summary_text`**。`summary_text` 在 Codex rollout 里就是那几行粗体小标题。
2. 我以只读方式直接核对了源 rollout `/Users/yang/.codex/sessions/2026/09/08/rollout-2026-09-08T09-59-44-01a08151-4eb4-7572-8865-b321956dda94.jsonl`：1243 条 `reasoning` response_item，其字段集为 `['encrypted_content', 'id', 'internal_chat_message_metadata_passthrough', 'summary', 'type']`——**没有 `raw_content` 字段，`raw_content` 非空条数 = 0，总字符数 = 0**。推理正文只存在于 `encrypted_content`（加密，不可读）。
3. 文件内最长行 73 字符，非空非标记行 3212 行、去重后 376 条标题，全部形如 `**Deriving MrRoPE's Pareto bound**`。`[已验证]`

**推论（对本 workflow 直接有用）：**

- 不存在"从 `codex_reasoning.txt` 里提取公式"的可能。文件里的 `Deriving ...` / `Formalizing ...` 是**尝试的目录**，不是推导。
- 但这份清单仍有三重价值：(a) 它给出了该线程**做过哪些推导尝试、在哪一行、什么顺序**；(b) 它把"仅在脑内完成、从未落盘"的推导与"落盘的推导"区分开——后者可在 `codex_tools.txt` 里被 `cat/sed` 读回（agent 用 shell 读文档时把正文带进了工具输出）；(c) 它暴露了线程的**子项目边界**（见 §2.0），避免把别的课题的定理误当 F 的零件。
- 因此本报告 §3 的公式**全部来自 `codex_tools.txt` 中被读回的文档正文**（agent 的 `cat/sed/rg` 输出），每条都标注了它在 `codex_tools.txt` 的文件行号、对应的源会话行号、以及该正文**原本所在的文件路径**。这些是可以直接去仓库里复核的一手位置。

---

## 1. 文件统计与结构

| 量 | 值 | 说明 |
|---|---:|---|
| 文件行数 | 5287 | 含块分隔与空行 |
| `[REASONING]` 块数 | 1038 | 每块 = 一行源 reasoning item |
| 块内标题行总数 | 3212 | 同一推理连续多行时标题会累积重复 |
| **唯一标题数** | **376** | 真正的信息量 |
| 源会话行号范围 | 103 – 6574 | `line N` 是源 JSONL 行号，与 `codex_tools.txt` 的 `line N` **同一坐标系**，可直接对表 |

块内标题的重复模式：Codex 的 `summary` 是**累积式**的（同一段推理的多行 do 会重复列出前面几条标题），所以"标题出现次数"（如 `**Diagnosing Native/M mismatch**` 出现 58 次）**不代表工作量**，只代表该推理段的行数。不能用频次做任何排序。`[已验证]`

按首词分类（376 条唯一标题）：

| 类别 | 条数 | 类别 | 条数 |
|---|---:|---|---:|
| E-准备 / 实现 / 运行等工程类 | ~190 | D-推导 (Deriving) | 13 |
| T-检查 (Checking) | 45 | T-测试 (Testing) | 23 |
| D-精化 (Refining) | 6 | D-定义 (Defining) | 6 |
| D-形式化 (Formalizing) | 6 | D-中文推导 (正在…) | 6 |
| D-设计 (Designing) | 5 | T-分析 (Analyzing) | 5 |
| D-起草证明 (Drafting proof) | 3 | T-计算 (Computing) | 3 |

**结论：这是一条以工程执行为主的线程（GPU 跑批、传输、核对），理论推导只占约 10%（~40 条标题）。** `[已验证]`

---

## 2. 主题归类

### 2.0 先划边界：线程覆盖 4 个不同子项目

这是本次挖掘的第二个重要发现。标题的空间分布按源行号分段，**段与段之间是不同课题**：

| 源行号段 | 子项目 | 判定依据（工具输出中的实际文件） |
|---|---|---|
| 103 – ~2250 | **RoPE 频率分配候选线**（BM / Gap-capped / P2 / C1，OLMo+Qwen7B 迁移） | `docs/research/ROPE_MRPRO_BM_*`、`ROPE_GAP_CAPPED_*`、`ROPE_QWEN7_BM_*` `[已验证]` |
| ~2900 – ~3300 | **native sparse attention / 位置兼容摘要（PSR、NativePrefix、PCW）** | `exp`/`docs/research/POSITION_AFTER_SPARSE_*` `[已验证]` |
| ~3400 – ~6100 | **position observability / 稀疏位置可观测性 / RefCarry 复核** | `results/position_observability_20260908`、`experiments/refcarry_audit/` `[已验证]` |
| ~6400 – 6574 | **RefCarry「Position Is a Retrieved State」审稿**（压缩位置接口的秩下界、软最大矩反例） | `docs/research/REFCARRY_INTERFACE_AUDIT_20260909.md`、`experiments/refcarry_audit/` `[已验证]` |

**与本次 KKT 目标（RoPE 频率分配）直接相关的只有第一段（源行 103–~2250）。** 第二、三、四段的"界/秩/最小状态/信息屏障"等定理**属于别的课题**（稀疏注意力的位置兼容、压缩位置的地址接口），**不得作为 F 的零件**。这点必须写进任何后续引用的来源标注里。`[已验证]`

（另有源行 ~3800–4000 的 `Deriving repeated-tail decay` / `Checking cap growth model`，工具输出显示是远程 SSH 上的训练 log 与 range 下载，未见推导正文；落在第三段内。`[部分证据]`）

---

### 2.1 水床 / 守恒

| 标题 | 源行 | 同段工具动作 | 判断 |
|---|---:|---|---|
| `Continuing theory work` / `Deriving MrRoPE's Pareto bound` | 897–902 | 同段最近的命令是 src891 `git diff --check`；段内无任何文档被读回 | **只是想法**——Pareto 界的推导只在脑内，正文未落盘（`encrypted_content`）。标题本身只证明"曾尝试给 MrRoPE 推一个 Pareto 界"。`[叙事-未验证]` |
| `Checking cap growth model` | 2969 | src2965 段：cache/native-window 工程 | 属于第二段子项目，非水床 |

**真正落盘的守恒相关推导不在本文件**，而在 `codex_tools.txt` 中被读回的文档正文，见 §3 的 F1/F2（`Σm` 的精确值）——那里明确写了 **"它不是能力预算或损失的充分统计量"**，与 KKT 文档 §4 的 `Σm（质心）不是守恒量` 红线一致。`[已验证]`

→ **水床不等式本身（`∫ln E ≥ ln b − ln c`，取等 iff 均匀）在 `codex_reasoning.txt` 中没有任何对应推导标题。** 该式只在权威文档 `NEXT_DERIVATION_KKT_PROBLEM.md §1.2` 中以 "EVQ tex 原形" 被引用。本文件对这一主题**零贡献**。`[已验证]`

---

### 2.2 密度 / 采样（NUDFT 采样密度 h(u)）

| 标题 | 源行 | 判断 |
|---|---:|---|
| （无直接标题） | — | 文件里**没有任何**标题含 density / sampling / quantile / 分位数。 |

密度层的推导全部在 `codex_tools.txt` 读回的 `paper-2027/appendix/a1_proofs.tex` 里（§3 F6）。`codex_reasoning.txt` 中与之最近的是 `Computing LP cross-check`（585）与 `Preparing candidate table input`（585），其工具动作是 `rg` 搜 `max.?gap|gap.?cap|radix|延后压缩|最大间距`（src580）——**即 Gap-capped 候选的 box 约束 LP**，不是密度泛函。`[部分证据]`

---

### 2.3 KKT / 单纯形约束

| 标题 | 源行 | 判断 |
|---|---:|---|
| `Computing LP cross-check` | 585 | **可复用线索**：Gap-capped 构造被当作一个 LP 交叉核对过。正文落盘在 `docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md`（见 §3 F7），是一个**真的闭式约束优化解**。`[已验证]` |
| `Clarifying the scaling gap` / `Testing a frequency-shift formula` | 562–563 | 工具动作是 `mkdir -p ~/.codex/memories/extensions/ad_hoc/notes`（src556）——只写了条笔记，正文不可见。`[叙事-未验证]` |
| `Requesting bounded theory audit` / `Deriving local frequency constraints` | 519–524 | 同段命令是 `pytest tests/test_cross_cache.py ...`；"local frequency constraints" 的产物未见落盘。`[叙事-未验证]` |

**结论：本文件里唯一有落盘实体的 KKT 型推导是 Gap-capped 的 box-constrained LP（F7）。** 其余 KKT 语言（三段结构、等边际、μ 乘子）在 `codex_reasoning.txt` 中**完全不出现**。`[已验证]`

---

### 2.4 损失代理（L_near / L_far）

| 标题 | 源行 | 同段工具动作 | 判断 |
|---|---:|---|---|
| `Deriving local frequency constraints` | 519 | pytest | 想法，无落盘 |
| `Analyzing phase scaling` / `Deriving dual-regime RoPE` | 668–677 | src663 读 `ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md` + `ROPE_QWEN15_FULL_LAG_P2_RESULT_*.md`；src682 读 gap_capped run 状态 | **只是想法**（dual-regime 的推导正文没被读回，也不在这两个文件里） |
| `Deriving zero-training invariance` / `Deriving phase-alignment formula` | 657–660 | src649 读 gap_capped `status.json` | **只是想法** |
| `Defining the frequency boundary` / `Deriving boundary phase continuity` | 695–700 | src682 读 gap_capped `live.json` | **只是想法**；但同段的实体（boundary 锁定）落盘在 Gap-capped 协议里 |
| `Checking candidate boundary` / `Testing fractional endpoints` / `Deriving mean-distance bounds` | 716–723 | src710 `cat ROPE_ZERO_TRAINING_MRPRO_STEP1_20260908.md` + `ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md` | **部分可复用**：BM 构造分析落盘（§3 F1） |
| `Deriving causal invariants` / `Analyzing cross-model constraints` | 847–856 | src836 跑 `summarize_candidate_screen.py --method GapCapped` | **只是想法** |
| `Computing QK operator Gram` / `Constructing attention signature` / `Diagnosing attention sink drift` | 879–888 | src874 `mkdir results/.../code_gap_capped_01` | **只是想法**；且 `Gram` 属红线禁项 |
| `Deriving causal block value` | 5034 | src5015 读 `Downloads/native_sparse_position_research_plan_20260908.md` §430-700 | 属第三段子项目；但同段文本给出了**相位差精确恒等式**（§3 F9），是"内容/相位分离"的可复用件 |
| `Defining same-token contrast` / `Checking signed-contrast conditions` | 915–924 | src911–924 重跑 prefix Q/K traces | **只是想法** |

**关键判读：** 这一簇（源行 519–924）是本次任务最关心的 L_near/L_far 建模尝试集中地，但**全部停留在脑内推理**——它们的工具动作要么是看远程 GPU 状态、要么是跑打分脚本，没有一条把公式写进文件。这些推导的正文随 `encrypted_content` 一起丢失。`[已验证]`

唯一被读回的相邻实体是 **BM（最小粗糙度）构造**和 **Gap-capped（box-LP）构造**，两者都是"构造式候选"而非"目标泛函推导"。这一点很重要：**该线程没有留下任何 L_near/L_far 的参数化形式**，与 `NEXT_DERIVATION_KKT_PROBLEM.md §1.4` 把 L_near/L_far 列为"部分证据/待闭合"是一致的。`[已验证]`

---

### 2.5 频率间隔 / 间隔重排

| 标题 | 源行 | 判断 |
|---|---:|---|
| `Comparing cyclic RoPE designs` / `Checking phase-allocation rationale` / `Checking native-phase bounds` | 569–576 | 同段命令是读 gap_capped 状态（src556–566 区间无文档读回）。**只是想法** |
| `Computing LP cross-check` | 585 | 见 F7，**可复用** |
| `Refining length correction` | 780 | 同段在等远程 final counts。**只是想法** |
| `Hedging remaining ranges` | 3822 | 属第三段。**只是想法** |

间隔重排的可复用实体同样只有 F1（BM 闭式）与 F7（Gap-capped 闭式）。`[已验证]`

---

### 2.6 能力建模

这是本文件里**确实有落盘、且与"能力"有关**的部分，但全部属于第二/三段子项目（稀疏位置兼容），**不是 RoPE 频率分配的能力模型**。列出是为了让后续引用能正确归档：

| 标题 | 源行 | 落盘实体 | 判断 |
|---|---:|---|---|
| `Formalizing phase-orbit proof` / `Deriving normalized rotary basis` / `Fitting phase coherence` | 5040–5057 | `Downloads/native_sparse_position_research_plan_20260908.md` §4.4（工具读回在 src5015） | **可复用推导**（作为"相位距离度量"，见 F9），但**属另一子项目** |
| `Deriving query-adaptive summaries` / `Checking clustering precedents` | 5022–5028 | 同上 §4.3/§4.5 | **可复用推导**（Jensen gap 区间，F10） |
| `Constructing correlated-coordinate bounds` / `Evaluating orbit-tube bounds` / `Refining RPEE reconstruction` | 5821–5832 | 未在本段工具输出中找到对应正文 | **只是想法** |
| `Revising information-barrier proof` / `Checking rank lower bound` / `Fixing rank formula` | 6449–6462 | src6443 `cat ~/.codex/attachments/07256e86-.../pasted-text.txt`（RefCarry 原文）；正文结论落在 `docs/research/REFCARRY_INTERFACE_AUDIT_20260909.md` | **可复用推导但属另一课题**（见 F12） |
| `Formalizing witness lower bounds` | 6242 | src6235 读远程 status.json | **只是想法** |
| `Testing softmax moment example` / `Building RoPE counterexample` / `Drafting RoPE audit` | 6525–6538 | src6496 建 `experiments/refcarry_audit/checks.py`；src6541 改脚本 | **可复用反例**（F11），属另一课题 |
| `正在校正最小状态定理，秩条件已核查` / `正在核对混合位置状态的门控含义，乘积池化已推导` | 6504–6508 | 同上 RefCarry 段 | **可复用但属另一课题** |

**判读：** 该线程在"能力建模"上的真实产出是 **平均 logit 摘要的误差上界 + 相位距离度量 + 矩不可识别反例**（F9–F12）。它**建模的是"压缩摘要能否重现 softmax 读出"**，不是"频率分配如何换外推能力"。若要复用到 KKT 的 L_far，必须先做一次**对象翻译**，且这一步尚无任何证据支撑。`[假设]`

---

### 2.7 标度律

| 标题 | 源行 | 判断 |
|---|---:|---|
| `Analyzing phase scaling` | 668 | **只是想法** |
| `Checking cap growth model` | 2969 | 属第二段；"cap growth" 指 cache，非长度标度 |
| `Selecting progressive RoPE schedule` | 2944 | 属第二段 |
| `Deriving the decay-clock criterion` | 3508 | 属第三段。同段工具是远程训练 log（`{"arm":"G32","seed":42,"updates":7629,...}`）与写 `POSITION_AFTER_SPARSE_RESEARCH_20260908.md`（src3517）。**只是想法** |
| `Deriving repeated-tail decay` | 3836 | 属第三段；同段是 range 下载。**只是想法** |

**结论：`codex_reasoning.txt` 对标度律（τ* 随训练长度下降、τ* 与模型尺寸无关）零贡献。** `[已验证]`

---

## 3. 可复用的公式（全部来自 `codex_tools.txt` 读回的文档正文）

> 读法：`codex_tools.txt:NNNN` = 该摘录文件的文件行号（agent 的 shell 输出）；`src N` = 源会话行号，可用来在 reasoning 清单里定位同一时刻的标题；"原本位置" = 该正文在真实仓库/本地的文件路径（**建议直接去那里引用，而不是引用摘录**）。

### F1 — 最小粗糙度（BM / boundary-matched）闭式 —— **可复用推导**
```
ε_i = 6i(N+1−i) / [N(N+1)(N+2)],  i = 1..N
m_q = q(q+1)(3N+2−2q) / [N(N+1)(N+2)]
```
- 出处：`codex_tools.txt:1373–1376`（src 369）；实现代码 `codex_tools.txt:5927`（src 1751，Python `m = q*(q+1)*(3*n+2-2*q)/(n*(n+1)*(n+2))`）。
- 原本位置：`docs/research/ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md`；生成程序 `scripts/analysis/build_boundary_matched_mrpro.py`。
- 推导：令 `L` 为零端点的一维离散 Laplacian，粗糙度 `R = εᵀLε`，`L` 正定；在 `Σε = 1` 下 `Lε = 常数向量` 给出唯一极小值；解 `Lz = 1` 再归一化即得闭式。程序用 `Fraction` 精确高斯消元独立求解。
- **性质**：`m_q^B > m_q^M` 逐分量（BM 每个内部中频槽都比 MrPro 更慢）；`N=2..100` 精确分数复算成立。`[已验证-推导+代码+精确复算]`
- **注意（KKT 红线直接相关）**：该构造原则就是"最小化粗糙度"，而 `NEXT_DERIVATION_KKT_PROBLEM.md §5.1` 把"平滑度"列为**不得作为 F 分项或选择子**的代理量。这里的关系是：**平滑度可以是构造式候选的生成原则（BM），但不能作为 F 的目标项或排序子**。两者不矛盾但极易被混用，引用时必须分开写。`[已验证]`

### F2 — `Σm` 的精确值（"Σm 不是守恒量"的正面证据）—— **可复用推导**
```
Σ_{q=1}^{N−1} m_q^M = (N−1)/3        (MrPro)
Σ_{q=1}^{N−1} m_q^B = (N−1)/2        (BM)
```
- 出处：`codex_tools.txt:5929–5932`（src 1751）。
- 原文紧接着写："**它不是能力预算或损失的充分统计量。** BM 的该总量恰与同边界 MrUni 相同，但两者分配不同。"
- 数值对账：Qwen `N=17` → MrPro `16/3 = 5.333…`；OLMo `N=18` → MrPro `17/3 = 5.6667`、BM `17/2 = 8.5`。OLMo 面板文档独立给出 "MrUni 和 BM 的中段累计 exponent 总和均约 8.5（MrPro 约 5.6667）"，与公式吻合。`[已验证-公式+面板对账]`
- 另：`NEXT_DERIVATION_KKT_PROBLEM.md §4/§5` 的 `Σm（质心）不是守恒量` 红线，与此处 F2 是同一条事实的两个来源。`[已验证-双源]`

### F3 — Gap-capped：box 约束下的逐分量最小解 —— **可复用推导（本文件里最接近 KKT 的实体）**
原约束（来自 MrPro 的设计不变式）：
```
0 ≤ ε_i ≤ c,   Σ_{i} ε_i = 1,   c = 2/(N+1)
（MrPro 原式 ε_i = 2i/[N(N+1)]，最大单步额外 log-gap = c·ln S）
```
对任意 `q`，剩 `N−q` 个槽最多承担 `(N−q)c`，故
```
m_q ≥ max(0, 1 − (N−q)c)
```
取
```
m*_q = max(0, 1 − 2(N−q)/(N+1))
```
它同时取到每个累计量的下界，因此是**唯一的逐分量最小值**；对任何"对累计改动严格递增"的目标都最优。保持增量非递减。
**精确身份（独立复核补充）**：
```
m*_q = clip( (q − (N−1)/2) / ((N+1)/2), 0, 1 )
```
即 **一个收窄过渡区的 MrUni**。Qwen `N=17` 有效区间 31→40；OLMo `N=18` 22.5→32。
部署频率 `ω'_j = ω_j · S^(−m*_{j−l})`，两端保持 Native/PI；S=4，gain `1 + 0.1·ln4`。
- 出处：`docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md`（本地仓库，头部 60 行内）。
- **对 KKT 的价值**：这是**一个真实的、闭式的、单纯形 + box 约束最优解**，且它以 `I1/I2` 型端点锁定为约束。KKT 文档 §1.3 的 `I1/I2` 本质上就是 box 约束；F3 给出了"在给定 box 下逐分量极值解存在且唯一"的现成样例。`[已验证-推导+闭式+独立LP核对]`
- **但它是负结果**（见 §5 死路 D3）。

### F4 — 单旋转平面 / 连续距离区间上的最坏算子差
```
sup_{δ∈[0,D]} ‖R(ω'δ) − R(ωδ)‖₂ = 2·sin( min(D|ω'−ω|, π) / 2 )
```
- 出处：`docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md`。
- 原文自带边界："这个结论与实际某一 δ 下的误差大小不同：sin 的周期性允许逐点误差反转；多槽相干、softmax、V、跨层状态和生成决策也不由该界排序。" `[已验证-推导，带自述适用边界]`

### F5 — P2 的双段相位 + 2× 算子搬运条件
```
ρ(L) = max(1, L/L*)
φ_k(d, L) = ω_{N,k}·d           , d ≤ 512
          = ω_{P2,k}·d / ρ(L)   , d > 512
```
- 出处：`codex_tools.txt:7217–7222`（src 2057）。
- 同段三条**已核对**的性质（`[已验证-推导+独立实数/复数复算]`）：
  1. `d>512` 时核逐项与原 P2 相同（不会为处理 128K 先重写 64K 远程相位）。
  2. 对远程 `d` 有 **精确算子恒等式** `R_new,128K(2d) = R_P2,64K(d)`；并明确：**只把 scale 从 4 换成 8 只能保住 `m=1` 的槽，其余槽多出 `2^(1−m_k)` 的相位倍率**，不能完成这个算子的完整搬运。
  3. 近邻（≤512）回到该 checkpoint 自己的原生计算。
- **对 KKT 的价值**：第 2 条给出了"一张表在 2× 长度下精确复现另一张表算子"的**充要形式**，是"能力"侧少见的精确约束，可直接作为 F 的 far 侧候选约束（而非代理量）。`[已验证-推导]`

### F6 — EVQ-Cosh 采样密度：传输界 / 直方图界 / 高分辨率失真 —— **可复用推导（论文级）**
```
W_∞(μ_K, μ_ρ) ≤ 1/(2Km),     W_1(μ_K, μ_ρ) ≤ 1/(4Km)
‖ρ_K − ρ‖_1 ≤ B/(Km),        ‖ρ_K − ρ‖_∞ ≤ B/(Km)
   （假设 0 < m ≤ ρ ≤ M、‖ρ'‖_∞ ≤ B；Q 为 (1/m)-Lipschitz）
EVQ-Cosh 代入：ρ_τ(φ) = τ·cosh(τ(1−φ))/sinh τ,  m_τ = τ/sinh τ,  B_τ = τ²
   W_1(μ_{K,τ}, μ_{ρ_τ}) ≤ sinh τ/(4Kτ)
   ‖ρ_{K,τ} − ρ_τ‖_1  ≤ τ·sinh τ/K = τ²/K + O(τ⁴/K)
核积分误差：|∬K_sm dμ_K dμ_K − ∬K_sm dμ_ρ dμ_ρ| ≤ L_K/(2Km)
高分辨率（Bennett 积分）：D_K[ρ] = (1/(12K²))·∫₀¹ w(φ)/ρ(φ)² dφ + O(K⁻³)
```
- 出处：`codex_tools.txt:12196–12235`（src 3331，agent 用 `sed -n '555,615p' paper-2027/appendix/a1_proofs.tex` 读回）。
- **原本位置（已在库，可直接引用）**：`paper-2027/appendix/a1_proofs.tex:523` 起（我用 `grep` 复核过 `W_\infty(\mu_K,\mu_\rho)` 确实在该行）。`[已验证-文件在库]`
- 原文关键句（可直接复用为 F 的**判据**）："This is the finite-channel mechanism: a shaped ρ can reduce the weighted inverse-density load `∫ w/ρ²`; the absolute benefit scales as `K⁻²` for smooth quantization error and as `K⁻¹` for transport/kernel terms." 并给出 `K=32→16` 时 `K⁻²` 项放大 4×、`K⁻¹` 项放大 2×。
- 大 τ 提醒（原文自带边界）：`m_τ = τ/sinh τ` 变小，`1/(Km_τ)` 增大，"传输界在因子受控时才信息量充足"。
- **这正是 KKT 文档 §4 所说的"EVQ Cosh 的分位数导数等价泛函"的严格离散化版本**——`∫ w/ρ²` 就是 `α/h` 型项（`h = 1/ρ`，`1/ρ² = h²`… 注意指数：`∫ w/ρ²` 对应 `∫ w·h²`，而 KKT 文档 §4 写的是 `∫[α/h + β(1−u)² h]`，**两者不是同一个泛函**，一处是 `h²` 一处是 `h⁻¹`。这是一个**必须核对的口径差异**，见 §6 矛盾 C3。`[部分证据]`

### F7 — 打分与 (a, R, z) 分解（预算实验的定义骨架）
```
s_ij = (1/√d_h)[ Σ_{k=1}^{K} (q_i^{(k)})ᵀ R_{ω_k}(i−j) k_j^{(k)} + (q_i^{N})ᵀ k_j^{N} ]
ω_k = exp[ −(a + R·z_k) ],   z_0 = 0,  z_{K−1} = 1
```
- 出处：`codex_tools.txt:9594–9608`（src 3089）。
- 原本位置：`/Users/yang/Downloads/rotary_budget_theory_and_experiment_20260908.md` §2（本地存在，48KB）。`[已验证-定义]`
- **对 KKT 的价值**：这是"支撑/分配分解 (a,R) 与 z"的**精确记号层**，与记忆红线"读任何 RoPE 表对比前先分解 (a,R) 与 z"完全对应。可作为 F 的坐标声明模板。

### F8 — EVQ-Cosh 分位数表（部署表的精确定义）
```
a = 0,  R = (31/32)·log(500000)          （K_ref = 32, base = 500000）
几何：z^G_k = k/(K−1),           ω^G_k = e^{−R z^G_k}
EVQ ：u_k = (k+1/2)/K
      q_k = 1 − (1/τ)·asinh[(1−u_k)·sinh τ]
      z^E_k = (q_k − q_0)/(q_{K−1} − q_0),   ω^E_k = e^{−R z^E_k}
τ = 64/√2048 = √2   （训练长度 2048 下固定，所有 K 用同一个 τ）
```
- 出处：`codex_tools.txt:9796–9829`（src 3128）。
- 原文自述边界（**重要，防止过度声称**）："不声称 `τ=√2` 是新理论推出的实际模型最优值。该取值也与已有 MLA 使用的约 1.414 强度接近，但不是根据新结果挑出的。" `[已验证-定义+自述边界]`

### F9 — 相位差精确恒等式与相位距离（内容/相位分离）
```
y_s = R_Ω(p_s) u_s
y_s − y_t = R(p_s)(u_s − u_t) + [R(p_s) − R(p_t)] u_t
‖[R(p_s) − R(p_t)] u_t‖² = 4 Σ_k ‖u_t^{(k)}‖² sin²( ω_k(p_s − p_t)/2 )        (12)
d_Ω(s,t)² = 4 Σ_{k=1}^{K} sin²( ω_k(s−t)/2 )                                     (13)
Φ_Ω(s) = [cos ω₁s, sin ω₁s, cos ω₂s, sin ω₂s, …]     （orbit embedding）
Δs_C ≤ ‖q̄‖ [ D_u(C) + U(C)·D_Ω(C) ]
J_C ≤ (‖q̄‖²/8)·[ D_u(C) + U(C)·D_Ω(C) ]²
```
- 出处：`codex_tools.txt:14065–14112`（src 4059）；同一文本第二次出现于 `codex_tools.txt:15434–15475`（src 4415）。
- 原本位置：`/Users/yang/Downloads/native_sparse_position_research_plan_20260908.md` §4.4（本地存在，59KB）。
- 原文自述边界："这保留完整二维 pair，不只看 cos 分量，也不做长距离 Taylor 截断。""**不声称最小化式 (13) 就最小化任务损失。**"
- **对 KKT 的价值**：`(12)` 是**精确恒等式**，可作为"旋转改动对 key 差的贡献"的严格分解；`(13)` 是 orbit 上的欧氏度量。**但注意红线**：`d_Ω` 是静态几何量，**只能作窗口内诊断，不得入 F**。真正可入 F 的是"该量被 softmax 放大后的读出误差"，即经 `(15)` 的桥接——而原文明确说内容项可能抵消它。`[已验证-恒等式；F 可用性=假设]`

### F10 — 均值摘要丢掉的量：精确 KL 与 Jensen 区间 —— **可复用推导**
```
F_b(q̄) − log n − q̄ᵀμ_b = D_KL( U_b ‖ p_b(q̄) ) ≥ 0                     (5)
F_b(q̄) = q̄ᵀμ_0 + H(p_0) + D_KL( p_0 ‖ p_b(q̄) )                        (6)
F_b − F̂_b = log Σ_r π̂_r e^{J_r},   π̂_r = n_r e^{q̄ᵀμ_r}/M̂_b           (8)
0 ≤ J_r ≤ (Δs_r)²/8                                                     (9)
F̂_b ≤ F_b ≤ LSE_r( log n_r + q̄ᵀμ_r + u_r ),   u_r = min(a_r²/2, a_r)    (10)
```
- 出处：`codex_tools.txt:13971–14030`（src 4057）。原本位置同 F9。
- 原文自述边界："不要求随机 token 独立"；"高维情况下该上界可能松；不能用它宣称已经获得实用的无漏选证书"。`[已验证-推导，带自述边界]`
- **对 KKT 的价值**：`(5)` 是"块均值摘要"误差的**精确**（不依赖小相位展开）表达；`(9)` 是 Hoeffding 型非渐近界。若要给 L_far 找"可计算且非静态几何"的载体，`(5)` 的 KL 形式比任何几何代理都更合规（它是分布量，不是几何量）。`[假设-需要对象翻译]`

### F11 — 矩不可识别反例（一阶/二阶摘要不够）—— **可复用反例**
- 一阶：两组 key，四个轴向单位向量 vs 四个对角单位向量；均值都为 0、协方差都为 `I/2`；但沿横轴 query 的 softmax mass 分别是
  ```
  ½(cosh t + 1)   和   cosh(t/√2)
  ```
  不相同。
- 二阶（代码反例）：`experiments/refcarry_audit/checks.py`，用 4 个精确旋转与两个不同 `μ`（`[.5,0,.5,0]` vs `[0,.5,0,.5]`）构造同矩不同读者分布。
- 出处：`codex_tools.txt:15521` 附近（src 4415）；`codex_tools.txt:6541`（src ~6541，改脚本时写入注释 "Same moments do not determine the mixture of actual reader distributions."）。
- 原文自述边界："**不说明 R=4 的 PSR 一定优于实际 COBS。**" `[已验证-反例；外推=未验证]`

### F12 — 压缩位置接口的秩下界（**属另一课题，仅备查**）
```
ker(S|_H) ⊆ ker(Φ|_H)  ⇒  r ≥ rank(Φ|_H)      （中心化秩下界）
反例：原文无条件写 r ≥ rank(Φ) 不成立——地址 {0,1}、频率 1 时未中心化 Φ 秩为 2，
      但只传 μ_1 一个实数即可用 Φ_0 + μ_1(Φ_1 − Φ_0) 恢复全部 moment（CPU 恢复误差 < 1.2e−16）。
```
- 出处：`docs/research/REFCARRY_INTERFACE_AUDIT_20260909.md:47,49`（本地仓库）。
- 原文自述："这是有限维线性代数结论，不应包装成未经查新的新定理。" `[已验证-有限维线代]`
- **红线提示：不得用于 KKT / RoPE 频率分配。** 该定理的对象是"地址 sketch 的维度"，与频率表无关。

---

## 4. 已验证数字（带出处）

> 这些数字都不是从 `codex_reasoning.txt` 里读到的（该文件没有数字），而是从同一线程的工具输出/落盘文档读到的，用于给 workflow-3 提供可对账的锚点。

| 数字 | 出处 | 等级 |
|---|---|---|
| BM 在 Qwen2.5-3B、S=4、36 条面板：**128K 70.83%** vs MrPro **78.13%**（BM 在零基槽 24–39 增加压缩，相对 MrPro 频率下降 1.5%–31.5%，最大在槽 34） | `docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md` §"已经直接观察到的事实" 第 1 条 | `[已验证]` |
| BM 诊断（Qwen3B，128K 两例）：O 原完整输入重放 0% / 20%；C 紧凑位置 100% / 100%；P 原始 128K 位置 + 精简背景 100% / 100%；L 原密集 prefill KV + 读出时限制可见集合 0% / 60% | 同上，结果表 | `[已验证]` |
| P 与 L 保持同一可见 token/位置集合却结果不同 → 从完整长背景形成的保留 KV 状态参与了失败；仅在读出时限制可见集合不足以满分恢复 | 同上，"可支持的机制结论" | `[已验证]` |
| 负例中最大相位差：槽 28 约 **31.25 rad**（lag 95697，多键）；VT 约 **41.49 rad**（lag 127052）。**原文明确"相位差大本身不是失败的充分条件"，"不能将最大相位差的槽 28 自动称最敏感槽"** | 同上，"频率影响的量级" | `[已验证]` |
| Gap-capped 负结果：Qwen2.5-3B 32K **84.44%** vs MrPro 87.22%（−2.78pp）；128K **62.15%** vs 78.13%（−15.97pp）；36 条 0 胜 7 负 29 平 | `docs/research/ROPE_GAP_CAPPED_RESULT_20260908.md` | `[已验证]` |
| OLMo-2-0425-1B-Instruct 零训练：开发 MrPro 37.22%/14.93%（4K/长端），MrProBM **79.44%/49.03%**；独立 seed 复核 MrPro 37.85%/2.78%，MrProBM **81.81%/51.32%**；同输入对照 MrUni 76.88%/32.12%，OfficialYaRN 54.38%/6.94% | `docs/research/ROPE_OLMO_BM_RESULT_20260908.md` 主结果表 | `[已验证]` |
| 上表的胜负计数：开发 BM vs MrPro **24 胜 1 负 11 平**；复核 **44 胜 0 负 28 平**；复核相对 MrUni 22 胜 6 负 44 平；相对官方 YaRN 40 胜 2 负 30 平 | 同上 | `[已验证]`（原文自述"是 prompt 级得失，不是显著性检验"） |
| 中段累计 exponent 总和：MrUni ≈ BM ≈ **8.5**（`(N−1)/2`, N=18），MrPro ≈ **5.6667**（`(N−1)/3`, N=18）；**"BM 高于 MrUni 说明该面板上的差异不能仅归约成指数总和"** | 同上 + `codex_tools.txt:5929–5932` | `[已验证-公式与面板双源吻合]` |
| Qwen `N=17`：第一项增量由 `1/153` 增至 `1/57`（**2.684×**），末项由 `1/9` 减至 `1/57` | `codex_tools.txt:5934` 附近（src 1751） | `[已验证]` |
| 矩阵算子局部扰动比：`Σ(ν^M − ω)² / Σ(ν^Y − ω)² = 0.4841`（Qwen 现配置） | `STARTING_POINT_YARN_VS_MRPRO.md` §2（权威文档，非本文件） | `[已验证-CPU]`（此处仅作对账，非本次挖掘所得） |
| OLMo BM 频率 SHA256：`fc0f443b1c58601d51209adb7e2b26df7ba10058a9dbdf193eb1de49116d446a`；模型 revision `48d788eca847d4d7548f375ad03d3c9312f6139e`，1,484,916,736 参数，Native4096/base500000/K64/S4/l14/h32/N18/gain 1.138629436111989 | `codex_tools.txt:1379–1385`（src 369） | `[已验证-实测]` |
| Qwen P2→C1 无截距等权最小二乘系数 `a* = 2.1477690111869907` | `codex_tools.txt:1671–1676`（src 456） | `[已验证-独立 Fraction 核对]`（原文自述"唯一性来自严格凸一维二次目标，**不是任务能力最优性**"） |

---

## 5. 死路（已证伪 / 已失败，不得再试）

| # | 机制 | 失败原因（原文） | 出处 | 等级 |
|---|---|---|---|---|
| D1 | **Gap-capped / 收窄 MrUni**（在 `0≤ε_i≤c, Σε=1` 下取逐分量最小解，`m*_q = max(0, 1−2(N−q)/(N+1))`，等价于 `clip((q−(N−1)/2)/((N+1)/2),0,1)`） | 闭式与独立 LP 验证正确，候选确实满足"更小原生频率位移 + 原最大 gap 限制"，**但两项都不足以选出更好的部署**。它还压缩了过渡区、增加了大 gap 的数量及粗糙度，改变了长距联合相位关系。"**不能将这些几何事实任一项单独指定为失败的唯一原因。**" 且"不支持继续扫描 cap、挪动边界或把表和 BM 插值以修当前分数。" | `docs/research/ROPE_GAP_CAPPED_RESULT_20260908.md`；`ROPE_GAP_CAPPED_PROTOCOL_20260908.md` | `[已验证-GPU 实测负结果]` |
| D2 | **"从完整长背景形成的 prefill KV 状态可被晚期读出修复"** | P（同 token、同原始 128K 位置、精简背景）100%/100% 而 L（原密集 prefill KV、读出时限制可见集合）仅 0%/60%。"**仅在读出时限制可见集合不足以满分恢复。**" 原文同时明确"这不证明预填充状态不可修复，也不否定其他晚期读取算法" | `docs/research/ROPE_BM_128K_DIAGNOSIS_20260908.md` | `[已验证-受控诊断]` |
| D3 | **"最大相位差的槽就是最敏感槽"** | 槽 28 相位差约 31.25 rad，但 P 条件仍 100%。原文："**相位差大本身不是失败的充分条件。不能将最大相位差的槽 28 自动称最敏感槽。**" | 同上 | `[已验证]` |
| D4 | **"用单点编辑移除竞争 key 就能恢复精确绑定"** | 把 BM 选错的 `bizarre-slime` 改成 `neutral-slime`（仅 token 74100 改变，长度 130871 不变），MrPro 仍正确，BM 从 9289114 换成另一个错误数字 4068207。"**最初那一个相似 key 不是整个失败的唯一原因。**" | 同上 | `[已验证]` |
| D5 | **"导数状态（H_t = ∂_z S_t(z)|₀）能提供普通多状态动力学无法逼近的函数族"** | 原文自我否证："导数表示可能有更好的数值条件或共享计算，**但并没有凭空创造一个普通多状态动力学无法逼近的函数族**。" 并把该状态的精确含义限定为"对减弱擦除的敏感性"，仅在理想写入条件下才退化为"上一条语义记录" | `codex_tools.txt:10792–10820`（src 3196） | `[已验证-推导+自我否证]`（注：属线性注意力记忆子项目） |
| D6 | **"平方根/矩摘要（一阶、二阶）足以恢复 softmax 读出"** | 同矩反例：均值 0、协方差 `I/2` 的两组 key 给出不同 mass `½(cosh t+1)` vs `cosh(t/√2)`。原文限定"**不说明 R=4 的 PSR 一定优于实际 COBS**" | `codex_tools.txt:15521` 附近（src 4415） | `[已验证-反例]` |
| D7 | **"无条件 `r ≥ rank(Φ)`"** | 反例：地址 {0,1}、频率 1，未中心化 Φ 秩为 2，但只传 `μ_1` 一个实数即可恢复全部 moment（CPU 误差 < 1.2e−16）。正确形式是中心化秩 | `docs/research/REFCARRY_INTERFACE_AUDIT_20260909.md:49` | `[已验证-CPU]`（属另一课题） |
| D8 | **"最小粗糙度 ⇒ 更好部署"（作为选择规则）** | 需谨慎：BM（最小粗糙度）在 **OLMo** 上大胜 MrPro（81.81/51.32 vs 37.85/2.78），而 Gap-capped（更粗糙、更集中）在**Qwen3B 128K** 上大败（62.15 vs 78.13），BM 自身在 Qwen3B 128K 也败（70.83 vs 78.13）。**"不能仅归约成指数总和"**。见 §6 矛盾 C1 | 三份文档交叉 | `[已验证-跨模型分歧]` |
| D9 | **"继续扫 cap / 挪边界 / 两表插值 / 改善一个代理再破坏另一项"** | 用户明确指示不得进入该循环；`CausalGain` 仅有 CPU 代码、暂停未运行 | `docs/research/ROPE_GAP_CAPPED_RESULT_20260908.md` "该负结果改变什么" | `[叙事-用户指令，作为边界条件引用]` |

---

## 6. 矛盾与口径不一致

**C1（最重要）— "最小粗糙度"在 OLMo 上是胜负手，在 Qwen3B 128K 上是败因。**
- OLMo 1B：BM（= 最小粗糙度闭式 F1）= 81.81%/51.32%，MrPro = 37.85%/2.78%（`ROPE_OLMO_BM_RESULT_20260908.md`）。
- Qwen2.5-3B 128K：BM = 70.83%，MrPro = 78.13%（`ROPE_BM_128K_DIAGNOSIS_20260908.md`）。
- 而 `NEXT_DERIVATION_KKT_PROBLEM.md §5.1` 把"平滑度"列为例外红线（不得入 F），§3 表又把 `Smooth`（"最小粗糙度重排"）列为 32K/128K = 87.2/**68.3** 的失败点。
- **冲突点**：同一条构造原则（最小化 `εᵀLε`）在一份材料里是 44 胜 0 负的胜因，在另一份材料里被当作"几何更优者任务更差"的反例。这正是 `STARTING_POINT_YARN_VS_MRPRO.md` §5 F6 要求 F 必须**同时容纳**的第 3 条。**任何 F 候选不得回避这个分歧。** `[已验证-双源冲突]`
- 附带口径问题：`NEXT_DERIVATION_KKT_PROBLEM.md §3` 的 `Smooth` 与本文 F1 的 `BM` **是否为同一对象未确认**（Smooth 128K=68.3 vs BM Qwen3B 128K=70.83，接近但不等；面板与文档的模型/长度/评分口径需核对）。**建议交给 `tables/ground_truth_tables.json` 的工作去对账。** `[假设-待核]`

**C2 — `Deriving MrRoPE's Pareto bound`（src 897）没有任何落盘后继。**
- 标题显示推过 Pareto 界，但同段唯一命令是 `git diff --check`（src 891）；此后该线程再未出现 Pareto 相关标题或文档。
- 与 `NEXT_DERIVATION_KKT_PROBLEM.md §4` 把"EVQ / MrRoPE 在同一 KKT 问题中的极限关系"列为 **待证（K4）** 一致：**该推导确实没有完成，也没有留下可用残片。** `[已验证-否定性结论]`

**C3 — `h` 的幂次在两个来源里不一致（需核对）。**
- `NEXT_DERIVATION_KKT_PROBLEM.md §4`：`J[h] = (1/2)∫₀¹[ α/h(u) + β(1−u)² h(u) ] du`，并注 "`h` 的倒数 `1/h` 项 = 间隔 `a_i` 的凸惩罚"。
- `paper-2027/appendix/a1_proofs.tex`（经 `codex_tools.txt:12226`）的高分辨率失真：`D_K[ρ] = (1/(12K²))∫₀¹ w(φ)/ρ(φ)² dφ`，其中 `h = 1/ρ` 是间隔、`ρ` 是密度。
- 代入 `ρ = 1/h`：`w/ρ² = w·h²`。**即一处是 `α·h⁻¹`、另一处是 `w·h²`。** 若两者都声称是同一泛函族的不同写法，则必有一处记号或幂次有误。**这是 K4 对照时必须先钉死的一步。** `[部分证据-记号层冲突]`

**C4 — `Σm` 的口径（已解决，但需保持）。**
- F2 给出 `Σm^M = (N−1)/3`、`Σm^B = (N−1)/2`，两者相差 50%；文档明确"不是守恒量"。
- 与 `NEXT_DERIVATION_KKT_PROBLEM.md §4/§5`（"`Σm`（质心）是自由决策变量"）一致。**本文件不构成冲突，是独立第二来源。** `[已验证-双源一致]`

**C5 — 线程里至少两处"几何量被证伪"的记录，方向相反地支持红线。**
- `codex_tools.txt:432` 与 `:25573`：静态几何/响应（Gram、曲率、重构残差、phase risk、覆盖、平滑性）"变好 ⇒ 生成变好"已被直接反例否定；"18 样本 64 维行为梯度也已失败"。
- `codex_tools.txt:1351`：必须"先定义输出对象再形成 Gram"。
- 与红线 1 一致，**无冲突**。列为确认项。`[已验证]`

---

## 7. 未解问题（本文件留下的空白）

1. **`Deriving local frequency constraints`（src 519）到底推出了什么？** 正文随 `encrypted_content` 丢失，同段无落盘。这是 KKT §1.3 的 `I1/I2` 端点锁定最可能的前身，但**完全不可追**。`[已验证-空白]`
2. **`Deriving dual-regime RoPE`（src 672）、`Deriving zero-training invariance`（src 657）、`Deriving boundary phase continuity`（src 695）、`Deriving mean-distance bounds`（src 720）四条构成了一个连续的推导串（src 657–723）**，工具动作是读 `ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md` 与 gap_capped 运行状态。这个"边界相位连续 + 平均距离界"的组合**是否就是后来 Gap-capped/BM 的理论母体**，无法从本文件判定。建议 workflow-3 去 `docs/research/ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md` 与 `ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md` 里找对应段落。`[假设]`
3. **`Analyzing phase scaling`（src 668）与 `Deriving MrRoPE's Pareto bound`（src 897）之间是否有关联？** 中间没有中间标题，无法判定。
4. **`Fitting phase coherence`（src 5040）/ `Formalizing phase-orbit proof`（src 5046）** 的正文可从 `Downloads/native_sparse_position_research_plan_20260908.md` §4 读到，但**该 §4 是否就是这两条标题的产物**未确认（该文档在 src 5015 被读回，时间上吻合）。`[假设]`
5. **`Constructing correlated-coordinate bounds` / `Evaluating orbit-tube bounds`（src 5821–5832）** 在本段工具输出中找不到对应正文（同段命令是远程 `tail oracle_tar...`）。落盘位置未知。`[已验证-空白]`
6. **`Formalizing witness lower bounds`（src 6242）** 落盘未知。
7. **`Revising information-barrier proof`（src 6455）的"信息屏障"定理** 我在 `REFCARRY_INTERFACE_AUDIT_20260909.md` 里只找到秩下界（F12），**没有找到名为 "information barrier" 的定理**；可能被改名或未落盘。`[部分证据]`
8. **C3 的 `h` 幂次冲突** 未解。
9. **C1 的 Smooth ≡ BM 问题** 未解。

---

## 8. 覆盖度（读了什么 / 没读什么）

**完整通读：**
- `analysis/kkt_20260910/mine/extracts/codex_reasoning.txt` —— 全文 5287 行、1038 块、3212 条标题行，做了首现/末现行号映射与逐词分类（§1、§2 的全部表格基于此）。`[已验证-全文]`
- 权威对照三件：`analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md`（全文）、`STARTING_POINT_YARN_VS_MRPRO.md`（全文）、`INTEGRATION_20260910.md`（**未读**，见下）。

**为定位正文而定向读取（不是通读）：**
- 源 rollout JSONL 的**结构层**：只统计了 `reasoning` response_item 的字段集与非空计数（1243 条），**未读取任何 message 正文**。`[只读]`
- `codex_tools.txt`（1.5MB）：按行号定位读取了 20 余处窗口（12180–12245、13960–14130、15360–15540、10770–10890、9790–9836、10040–10060、9580–9660、9660–9730、5900–5945、7195–7240、1360–1400、1650–1700），以及若干 `src` 行临近命令的抽样。**未通读。**
- 本地仓库文档（只读）：`docs/research/ROPE_GAP_CAPPED_PROTOCOL_20260908.md`、`ROPE_GAP_CAPPED_RESULT_20260908.md`、`ROPE_BM_128K_DIAGNOSIS_20260908.md`、`ROPE_OLMO_BM_RESULT_20260908.md`、`REFCARRY_INTERFACE_AUDIT_20260909.md`（各读了头部/关键节）；`paper-2027/appendix/a1_proofs.tex`（只 grep 定位）。`[已验证]`
- 本地 Downloads：`rotary_budget_theory_and_experiment_20260908.md` 与 `native_sparse_position_research_plan_20260908.md` —— **只读了标题树（header 列表）与经 `codex_tools.txt` 读回的部分正文，未通读正文。** 全文分别 48KB / 59KB。

**明确未读（时间/范围外）：**
- `codex_assistant.txt`（194KB）、`codex_assistant_a/b.txt`、`codex_user*.txt`、`codex_compaction_1..5.txt`、`codex_subagents.txt`、`codex_filechanges.txt` —— 仅做了关键词计数（`Pareto`/`barrier`/`orbit`/`Gram`/`dilution` 等几乎全 0 命中，唯一有命中是 `TAPE`），**未通读**。这也是"这些推导没落进 assistant 输出"这一判断的依据之一（弱证据：只测了关键词）。
- `analysis/unify_20260910/INTEGRATION_20260910.md`（21KB）—— **未读**。因此本报告的"与权威文档冲突"只对照了 `NEXT_DERIVATION_KKT_PROBLEM.md` 与 `STARTING_POINT_YARN_VS_MRPRO.md` 两份。**若 INTEGRATION 里有新结论，§6 的冲突判定可能需修订。**
- `analysis/unify_20260910/digests/` 与 `digests_codex/` 下的全部 digest —— 未读。
- `analysis/kkt_20260910/mine/extracts/` 中除 `codex_reasoning.txt` / `codex_tools.txt` 之外的全部文件。
- 仓库内 0910 的 RoPE 分配文档（`ROPE_ALLOCATION_THEORY_CORE_20260910.md`、`ROPE_ALLOCATION_SUBSPACE_DERIVATION_20260910.md`、`BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md` 等）—— 只列了文件名，**未读**。这些是**别的线程**的产物，与本次指定的 09-08 线程不同源。

**没能读到的东西（客观缺失，不是我的疏漏）：**
- 该线程的**推理正文本身**（`encrypted_content`，1243 条）。这是本任务最大的、不可弥补的缺口。凡 §2 中标注"**只是想法**"的条目，其内容均已永久不可追；只能保留标题作为"曾尝试"的证据。
