# RoPE 变分框架论文实验重构计划书（Codex 执行版，可直接落地）

> 版本：v2（针对“可复制/可执行/可审计”优化版）  
> 目标：仅从**实验设计与执行**角度，重构论文实证证据链，修复 8B 管线可信度问题，补齐基线、统计功效、复现材料与论文级附录产出。  
> 适用对象：Codex / 自动化实验代理 / 实验工程同学  
> 说明：本计划**不扩展理论证明**，但会把理论主张转成可检验的实验假设与 protocol。

---

## 使用说明（你先看这个）

- 你可以把下面的 **「给 Codex 的主提示词」** 直接喂给 Codex。
- Codex 执行时必须按阶段推进，并在每阶段输出：
  - 计划
  - 执行
  - 发现
  - 是否过门槛（Pass/Fail）
  - 下一步
- **任何阶段发现 pipeline 异常，先停下来修 root cause，再继续。**
- 本计划把“论文可发表性”的实证要求拆成：**管线正确性 → 公平性 → 统计性 → 机制性 → 可复现性**。

---

## 一、给 Codex 的主提示词（建议原样复制）

```md
你是该论文的“实验负责人 + 审计员 + 复现工程师”。你的唯一目标是：在不改动论文核心理论主张的前提下，彻底重构并验证其实验部分的科学有效性、统计严谨性和可复现性。

# 总目标
修复并重跑论文实证部分，形成可用于 NeurIPS 级论文的完整证据链：
1) 证明 8B 管线没有坏（尤其是 LongBench / NIAH / LoRA / chat template / scoring）
2) 证明方法比较公平（统一注入路径、统一预算、统一模板）
3) 证明结论不是噪声（全量评测 + 多 seed + 配对统计）
4) 证明论文可复现（规范附录 + 环境锁定 + 工件导出）
5) 证明主张强度与统计证据匹配（显著/方向性/不确定）

# 第一原则（硬约束）
- 禁止 cherry-picking：跑过的任务/方法/seed 都必须记录
- 禁止 preview-only 或 debug split 作为最终结论
- 禁止只报相对提升，必须同时报绝对分数与差值
- 禁止基线崩塌时继续解读相对提升（先修 pipeline）
- 禁止不同方法走不同底层 API 路径产生 confound（除非做等价性验证）
- 禁止只保留 aggregate 指标，必须保存 per-sample traces
- 禁止在论文附录中暴露内部日志/文件名痕迹（EXP_xxx/BAD_xxx/tmp_*.csv 等）

# 核心交付件（必须全部生成）
A. 实验审计报告（pipeline debug + root cause）
B. 完整实验矩阵结果（含失败 run 和重试原因）
C. 统计分析报告（CI, p-value, effect size, robustness）
D. 论文表格/图导出（CSV + LaTeX + PNG/PDF）
E. 复现实验附录（超参表、数据/模型清单、环境、脚本入口）
F. 结论强度分级建议（validated / directional / inconclusive / unsupported）

# 执行阶段（必须顺序执行）
Phase 0: 环境与版本冻结
Phase 1: 8B 管线审计与故障修复（优先级最高）
Phase 2: 严控协议重建（统一注入路径 + 方法规范化）
Phase 3: 全量主实验重跑（LongBench 全任务 + 强基线 + 多seed）
Phase 4: 统计分析与稳健性检验
Phase 5: 机制消融（anchoring / 离散化 / 公平性）
Phase 6: 论文级工件整理与附录生成

# 完成定义（DoD）
仅当以下条件全部满足，才能判定实验重构完成：
- 8B baseline 在完整评测集上恢复到合理绝对水平（非明显失效态）
- 所有主方法在完整 LongBench 上有 per-sample traces
- 至少包含：Geometric / PI / YaRN / LongRoPE / NTK-aware(or equivalent) / 作者主方法
- 8B 至少 2 seeds（最好 3）；小模型主对比至少 3 seeds
- 给出配对 bootstrap CI + permutation/sign-flip p-value + effect size
- 结论文案按统计结果自动分级（significant / directional / unsupported）
- 产出规范附录，不含内部草稿痕迹与临时文件名
- 所有结果可追溯到 config hash + code hash + data hash + model id

# 输出格式要求（每阶段）
请严格按以下结构输出：
1. 本阶段目标
2. 执行计划（具体命令/脚本/配置）
3. 执行结果（表格+关键日志）
4. 异常与根因分析（如有）
5. 修复方案与验证
6. 门槛判定（Pass/Fail）
7. 下一阶段计划
```

---

## 二、实验目标与可验证假设（实验侧）

### 2.1 实验目标（不是刷分，而是建立可信证据链）
本实验重构的目标不是“最大化某个榜单分数”，而是建立以下五层证据：

1. **管线正确性**：8B 的训练与评测流水线工作正常，结果不是乱码/模板错误/打分错误造成的假象  
2. **比较公平性**：所有 RoPE 方法在相同约束下对比，差异能归因于位置频率映射本身  
3. **统计可信性**：结论有足够统计功效，不依赖少数任务或单 seed 运气  
4. **机制一致性**：实验现象与理论预测的 trade-off / anchoring 方向一致  
5. **复现可验证性**：外部读者能按附录与脚本复现实验

---

### 2.2 可验证假设（把理论主张转成实验命题）
> 这些假设由实验支持/反驳，不涉及新增理论证明。

- **H1（管线有效性前提）**：选定的 8B 基座模型在标准完整评测（不训练或合理训练后）应达到“非崩溃”的绝对分数水平；若明显异常偏低，说明 pipeline broken。
- **H2（公平比较命题）**：在统一注入路径、统一预算、统一模板下，各方法性能差异主要反映位置编码方案差异，而非框架/API/模板 confound。
- **H3（方向性机制命题）**：作者方法（尤其 anchored 机制）在长距检索/长上下文任务上展现出与理论一致的方向性 trade-off。
- **H4（稳健性命题）**：结论不依赖单一任务、单一 seed、单一长度、单一 prompt 模板。
- **H5（能力守护命题）**：长上下文优化不会把基础 instruction-following 能力训练崩（或至少该 trade-off 被明确量化并如实报告）。

---

## 三、Phase 0：环境冻结与实验卫生（防止重跑地狱）

### 3.1 环境与版本锁定（必须）
在 `artifacts/repro_manifest/` 中自动生成并版本化以下信息：

- 代码仓主 commit hash
- 子模块 commit hash（如有）
- Python / CUDA / Driver 版本
- PyTorch / transformers / peft / flash-attn / accelerate / vllm（如用）版本
- tokenizer 版本与 chat template 版本标识
- benchmark 版本与数据哈希（LongBench / NIAH / Passkey 等）
- 模型 ID（base vs instruct）与 tokenizer checksum
- 评测脚本、prompt 模板、打分脚本哈希
- 配置文件哈希（每次 run 的 config hash）

**门槛（Pass 条件）**
- 任意实验结果都能回溯到 `code_hash + config_hash + model_id + data_hash`

---

### 3.2 目录与命名规范（杜绝“内部草稿痕迹”污染论文）
建议目录结构如下（Codex必须按此组织）：

```text
artifacts/
  repro_manifest/
  audit/
  configs/
  runs/
  logs/
  traces/
  metrics/
  stats/
  plots/
  tables/
  exports/
paper_exports/
  main_tables/
  appendix_tables/
  main_figures/
  appendix_figures/
  latex/
  captions/
```

**命名规范要求**
- 用 `snake_case` + 日期/哈希，不使用 “final_final_v7”
- 严禁在论文中直接出现内部名：
  - `EXP_*`
  - `BAD_*`
  - `tmp_*`
  - `debug_*`
  - `method_metrics_best_available.csv`
  - `*_integrity_v6.json`
- 论文中的实验命名必须使用规范名称，例如：
  - `8B Controlled LoRA Protocol`
  - `LongBench Full Evaluation (All Tasks)`
  - `Anchoring Ablation`

---

## 四、Phase 1：8B 管线审计（最高优先级，先修再比）

> 这是决定论文生死的阶段。**先证明你的 8B 管线没坏，再做任何方法对比。**

---

### 4.1 先跑“零改动基线”审计（不训练、先验明管线）
#### 4.1.1 必跑实验
- 模型：建议 `Llama-3-8B-Instruct`（作为主 8B 基座；若你坚持 base，也必须同时跑 instruct 做 sanity 对照）
- 设置：**不做 LoRA、不改 RoPE**
- 模板：官方/社区标准 chat template
- 解码：固定一组保守参数（如 temperature=0 / top_p=1 / 合理 max_new_tokens）
- 评测：**完整 LongBench 全任务（全部样本）**
- 输出：必须保存 **per-sample traces**

#### 4.1.2 必须保存的 per-sample 字段
每个样本至少包含：

- `task_name`
- `sample_id`
- `input_length_tokens`
- `context_length_tokens`
- `prompt_hash`
- `generation_args`
- `raw_output`
- `parsed_output`（如有）
- `score`
- `evaluator_status`（ok / parse_fail / timeout / exception）
- `failure_type`（none / empty / truncated / malformed / template_leakage / etc.）

#### 4.1.3 管线损坏判定条件（命中任一则进入深度排查）
- 大量空输出、乱码输出、非法字符
- 模型输出 role tags/system prompt 等模板泄漏
- 早停严重（输出极短且与任务不匹配）
- 长样本普遍被截断至无效长度
- 评分脚本异常率高或 parse_fail 高
- 整体绝对分数落入明显“模型失能区间”

> 注意：这里不要求你“复现榜单最佳”，只要求证明管线自洽、模型未被评测流程弄坏。

---

### 4.2 Prompt / Chat Template / Decoding 审计（高概率 root cause）
这部分请 Codex 自动化检查，并附人工 spot-check：

#### 自动检查项
- 模板是否匹配模型类型（base/instruct）
- role 顺序是否正确（system/user/assistant）
- stop token / eos token 是否冲突导致早停
- 是否把 special tokens 错误拼进纯文本 prompt
- `max_new_tokens` 是否对某些任务过小
- `temperature/top_p/repetition_penalty` 是否导致异常输出
- 长上下文模板膨胀是否触发截断（截断发生位置在哪）

#### 手工 spot-check（必须）
- 至少 20 个样本（覆盖不同任务）
- 人工判断模型是否在“认真答题”
- 人工核对评分是否与输出语义一致（防 evaluator bug）

**输出**
- `artifacts/audit/prompt_template_audit.md`
- `artifacts/audit/scoring_audit.md`
- `artifacts/audit/manual_spotcheck_samples.md`

---

### 4.3 评测脚本审计（LongBench / NIAH）
重点排查：

- 是否误用 preview/debug split
- 任务格式映射是否错误（生成/分类/抽取）
- evaluator 版本与任务定义是否不匹配
- 输出后处理是否过度（如过强正则导致错判）
- ground truth 预处理是否损坏（转义/大小写/空格）
- per-sample traces 是否真完整保存（不是只保存 aggregate）

**必须产出**
- `pipeline_root_cause_report.md`：问题、根因、影响范围、修复方式、回归验证
- `fixed_vs_broken_comparison.md`：修复前后差异（含若干样本）

**Phase 1 门槛（必须通过）**
- 零改动基线恢复到合理绝对水平（不再是明显崩溃态）
- 每任务都有完整 per-sample 结果
- 至少 20 个样本人工核验通过
- 所有审计报告完成

---

## 五、Phase 2：严控协议重建（统一路径 + 方法规范化）

> 目标：把“公平对比”写实，避免审稿人质疑你在 protocol 上偏袒某个方法。

---

### 5.1 方法集合（主矩阵）
至少包含以下方法（8B + 小模型都尽量统一）：

1. **Geometric / 原始 RoPE**（baseline）
2. **PI**
3. **YaRN**
4. **LongRoPE**（关键缺失基线，必须补）
5. **NTK-aware / dynamic scaling**（至少一个代表）
6. **作者主方法（理论离散密度方案）**
7. **Anchored-sigmoid（主打工程方案）**
8. （可选）Hybrid / cosh-inspired 工程近似方案

> 若算力不足，优先保证 1/2/3/4/7 至少齐全，不要牺牲评测完整性。

---

### 5.2 方法规范卡（Method Spec Sheet，必须）
每个方法都必须生成一份规范卡（供附录复现）：

- 方法名称（论文名/常用名）
- 数学定义（连续形式/离散形式；若你理论另写，这里也要写实验实现式）
- 超参数及默认值
- 离散化方式（如 inverse-CDF / fixed formula）
- 注入位置（`inv_freq.copy()` 等）
- 与原始实现差异（若有重实现）
- 调参预算与选择规则
- 实现文件路径与 commit hash

> 这一步会直接修复“核心方法没公式/无法复现”的实证短板。

---

### 5.3 统一注入路径与等价性验证（防 API confound）
你原文强调统一注入路径是正确方向，这里升级为“可审计”的 protocol：

#### 规则
- 所有方法优先统一为同一注入路径（如 `inv_freq.copy()`）
- 若某方法官方实现依赖不同 API：
  - 必须做**等价性验证**（同方法不同注入路径差异测试）
  - 若不等价，需记录并在附录说明

#### 等价性验证最小测试
- 同方法（同 seed、同训练步数、同数据）在路径 A/B 各跑一轮
- 对比：
  - 参数注入是否一致
  - 前向输出差异（小 batch）
  - 关键指标差异是否在误差范围内
- 输出 `injection_path_equivalence.md`

---

### 5.4 公平预算定义（必须写入论文/附录）
不要只写“预算一致”，必须明确以下维度是否一致：

- 训练 token 数
- update steps
- batch size / grad accumulation
- optimizer / weight decay
- scheduler / warmup
- learning rate
- LoRA rank / alpha / dropout
- context length curriculum（如有）
- checkpoint 选择规则（best on val / fixed step / last）
- 数据采样顺序与 seed

**硬约束**
- 不允许作者方法单独拥有更长训练或更大 LoRA rank，除非给所有方法等额 tuning budget。
- 若采用“文献默认超参”，必须如实说明“不保证每方法都最优”，并作为局限性写明。

---

## 六、Phase 3：主实验重跑（全量、无删减、可统计）

---

### 6.1 8B 主实验（核心证据）
#### 6.1.1 主评测（Primary）：LongBench 全量
- 跑 **LongBench 全部子任务 + 全样本**
- 每个方法、每个 seed 均保存 per-sample traces
- 导出：
  - 各任务分数
  - 总分（macro）
  - 任务权重说明（如有）
  - 失败率统计（空输出/parse_fail/timeout）

#### 6.1.2 长距检索诊断（Mechanistic）
至少包含：
- NIAH / Passkey Retrieval（多长度）
- 长度建议：`4K / 8K / 16K / 32K / 64K`（按算力可裁剪）
- 报告：
  - 成功率 vs 长度曲线
  - margin / confidence proxy（若定义稳定）
  - anchored vs non-anchored 的退化形状差异

#### 6.1.3 基础能力守护（必须）
防止审稿人继续质疑“只是把模型训坏了”：

- Instruction-following sanity set（轻量即可）
- 通用 QA/推理 sanity set（轻量即可）
- 输出格式遵循率 / refusal 异常率 / 重复率
- （可选）短上下文任务性能，检查是否为“拆东墙补西墙”

> 目标不是追求通用 SOTA，而是证明长上下文优化未导致基础能力灾难性崩塌。

---

### 6.2 8B 训练设置建议（MVP 可发表版本）
> 你之前 600 step 明显过于脆弱（至少从审稿观感上极危险）。这里给一个更稳妥的实证方案。

#### 推荐最小可发表配置（优先）
- 底座：`Llama-3-8B-Instruct`
- LoRA steps：**3000–5000**
- LoRA rank：**r=64**（并做 `r=32/64/128` 小消融）
- Seeds：**至少 2，最好 3**
- 固定训练 token budget（所有方法一致）
- 固定 checkpoint 选择规则（提前写死）

#### 算力不足时的降级策略（按优先级）
1. 减少方法数量（保留关键基线与主方法）
2. 减少额外消融数量
3. 减少 seed 到 2（不要降到 1）
4. 降低最长长度档位（如先到 32K）
5. **不要**降级成 preview-only 或单任务结论

---

### 6.3 TinyStories 从头训练扩展（机制支撑）
你的小模型实验是加分项，但必须补齐严谨性，使其成为“机制支持”而非“替代 8B”的证据。

#### 必做项
- 规模：`50M / 100M / 350M`
- 主对比方法全覆盖（至少 baseline/PI/YaRN/LongRoPE/作者主方法）
- 每规模 **3 seeds**（至少 baseline 与作者主方法）
- 报告：
  - ID PPL
  - OOD 长度 PPL（多个长度）
  - 训练稳定性（loss 曲线/发散率）
  - 方差（均值±标准差）

#### 强烈建议新增（机制价值高）
构造一个**可控距离先验**的合成任务（例如混合短距+长距依赖）：
- 幂律型距离分布
- 双峰型距离分布
- 对比不同频率密度方案的性能轮廓
- 目的：让实验现象直接呼应理论中的“先验-密度匹配”主张

---

## 七、Phase 4：统计分析（决定你能说多强）

---

### 7.1 统计层级（必须三层）
1. **Per-sample 层**（最高信息量，主分析基础）
2. **Per-task 层**（审稿人最常看）
3. **Aggregate 层**（用于摘要与主表，但不能替代上两层）

---

### 7.2 必做统计分析（最低要求）
#### 主统计
- **配对 bootstrap 置信区间（CI）**
  - 推荐按 task 分层重采样
- **paired permutation / sign-flip test**
  - 对主比较（作者主方法 vs baseline、vs LongRoPE）
- **效应量**
  - 平均差值 + 标准化效应（如 paired Cohen’s d 或稳健替代）
- **多重比较修正**
  - 若任务多、方法多，控制 FDR（如 Benjamini-Hochberg）

#### 稳健性分析（强烈建议）
- Leave-one-task-out：去掉某个任务后结论是否反转
- Seed 敏感性分析：不同 seed 下方向是否一致
- 长度分层分析：短/中/长样本分层看效果

---

### 7.3 增强统计（推荐，能明显提升审稿说服力）
- **混合效应模型（mixed-effects）**
  - 固定效应：方法
  - 随机效应：任务 / 样本（按数据结构可简化）
- **稳健汇总统计**
  - trimmed mean / median-of-means（防个别任务主导）
- **异常值影响分析**
  - 哪些任务对 aggregate 提升贡献最大（防“只靠1-2个任务抬分”）

---

### 7.4 结论强度分级（自动生成文案）
Codex 必须根据统计结果自动给每个主结论打标签：

- **Validated improvement**：显著 + 多 seed 稳定 + 多任务一致
- **Directional evidence**：方向一致但未显著（功效不足或方差较大）
- **Inconclusive**：结果依赖少数任务/seed，证据不充分
- **Unsupported**：无法复现、统计不稳或基线异常

> 这一步能直接修复“措辞过强”的审稿风险。

---

## 八、Phase 5：关键消融实验（补足“为什么有效”）

---

### 8.1 Anchoring 机制消融（最高优先级）
既然审稿一定会盯着 Anchored-sigmoid，这里必须做完整：

#### 消融维度
- 有 anchoring vs 无 anchoring
- 锚点位置 sweep（高频端不同 anchor 位置）
- 锚强度/权重 sweep
- sigmoid 陡峭度（sharpness）sweep
- 与理论离散密度形状相似度（如曲线距离） vs 性能的关系

#### 输出建议
- 密度曲线 + 离散频点可视化
- 参数热力图（anchor位置 × sharpness）
- 长度退化曲线（anchored vs unanchored）
- 任务类别分组收益图（检索/摘要/多跳等）

---

### 8.2 离散化误差（实验版，配合你后续理论版）
你理论会补 quantization bound；实验侧至少给“经验惩罚曲线”：

#### 建议实验
- 在合成任务或小模型设置中改变频点数 `N`（如 32/64/128/256）
- 比较：
  - inverse-CDF 离散化
  - 均匀采样离散化
  - 拟合优化离散化（如有）
- 报告：
  - 密度逼近误差（数值）
  - 下游性能变化
  - `N=64`（Llama 类头维）时是否出现明显量化惩罚

---

### 8.3 公平性与协议敏感性消融（防“protocol偏袒”）
至少做以下几项中的 2-3 项：

- 同方法在不同注入路径下的一致性（若存在替代路径）
- 同方法在统一 base 与 per-method tuned base 下的差异
- 同方法在 base 模型 vs instruct 模型上的相对趋势一致性
- 训练步数 sweep（排除“只是更容易收敛而非上限更好”）

---

### 8.4 “preview vs full” 差异审计（强烈建议做）
专门做一次对比实验（哪怕只对 1-2 方法）：
- preview-only 结果
- full LongBench 结果
- 看结论是否放大/反转

**作用**
- 直接回应“挑选数据/阉割评测”的质疑
- 让你在论文中可以正面解释：最终结论为何必须以 full 为准

---

## 九、Phase 6：论文级工件导出（直接替换原稿）

---

### 9.1 必须导出的主表（CSV + LaTeX）
#### Table A（8B 主结果）
字段建议：
- Method
- LongBench Macro (abs)
- Δ vs Geometric
- Relative Gain (%)
- 95% CI
- p-value
- Effect Size
- #Seeds
- Notes (e.g., directional only)

#### Table B（LongBench 分任务结果）
- 每任务分数（每方法、每 seed 或均值±方差）
- 可在附录给 full table

#### Table C（长距检索）
- NIAH/Passkey 在各长度点结果
- 成功率/准确率/margin（如定义）

#### Table D（基础能力守护）
- instruction-following sanity
- 通用 QA sanity
- 格式遵循率/异常率

#### Table E（TinyStories 多规模）
- 模型规模 × 方法 × PPL × 长度外推指标 × seeds

#### Table F（超参数总表）
- 替代所有内部日志命名痕迹，供附录引用

---

### 9.2 必须导出的图（PNG/PDF）
- 主方法 vs baselines 的长度退化曲线
- Anchoring 消融热力图
- Task-level paired difference 森林图（带CI）
- 密度曲线 + 离散频点图（理论/工程映射可视化）
- 训练稳定性曲线（说明没把模型训崩）
- preview vs full 差异图（如做）

---

### 9.3 复现实验附录（NeurIPS风格）
附录必须包含以下内容（建议按章节）：

1. **Models and Licenses**
   - 模型名称、版本、许可证、用途（训练/评测）
2. **Datasets and Licenses**
   - LongBench、NIAH、TinyStories、合成任务（若有）
3. **Training Configuration**
   - optimizer, lr, scheduler, batch, accumulation, warmup, steps, LoRA rank/alpha/dropout
4. **Evaluation Configuration**
   - prompt 模板、decoding 参数、max_new_tokens、stop rules、scoring script version
5. **Statistical Protocol**
   - bootstrap/permutation、CI、effect size、多重比较修正
6. **Compute Budget**
   - GPU 类型、数量、训练时长、近似算力消耗
7. **Reproduction Entry Points**
   - 脚本入口（命令示例）、配置文件位置、输出目录说明
8. **Failure Cases and Audit Notes**
   - 关键失败模式与修复摘要（可精简版）

> 重点：让外部读者可以不依赖你内部 registry 也能独立跑起来。

---

## 十、推荐实验矩阵（可按算力分层执行）

---

### 10.1 最低可发表版本（MVP）
> 算力紧张时，至少做到这个版本，避免再次出现“证据不足”。

#### 8B（主实验）
- 方法：Geometric / PI / YaRN / LongRoPE / 作者主方法（Anchored）
- Seeds：2
- LongBench：全任务全样本
- NIAH/Passkey：至少到 32K
- 基础能力守护：有
- 统计：bootstrap + permutation + effect size

#### TinyStories
- 规模：50M / 350M
- Seeds：3（至少 baseline + 主方法）
- 主指标：PPL + 长度外推

---

### 10.2 标准可接受版本（Recommended）
#### 8B
- 方法：Geometric / PI / YaRN / LongRoPE / NTK-aware / 作者方法（2种）
- Seeds：3
- LongBench：全量
- NIAH/Passkey：到 64K
- Anchoring 消融：完整 sweep
- 统计：含 mixed-effects 和稳健性分析

#### TinyStories
- 50M / 100M / 350M 全覆盖
- 3 seeds 全覆盖
- 新增可控距离先验合成任务

---

### 10.3 冲刺强接收版本（Stretch）
- 8B 再加一个不同基座（如另一家主流 7B/8B instruct）验证泛化
- 加入“preview vs full 差异审计”
- 加入 protocol 敏感性消融（注入路径/训练步数/base vs instruct）
- 导出完整匿名复现包（可一键运行）

---

## 十一、常见致命坑位清单（Codex 必须逐项规避）

### 11.1 管线类
- 用错 chat template（base/instruct 混用）
- stop token 设置导致早停
- `max_new_tokens` 太小
- 截断发生在问题段而非上下文尾部
- evaluator 版本错配
- 实际跑的是 preview/debug split

### 11.2 比较公平性类
- 不同方法使用不同 API 路径
- 作者方法拥有更大 LoRA rank / 更多训练步数
- checkpoint 选择规则不一致（作者方法挑 best，其他用 last）
- 不同方法 prompt 模板不一致

### 11.3 统计类
- 只报 aggregate，不报 per-task/per-sample
- 只报 p-value 不报效应量与CI
- 多任务多比较不做修正
- 单 seed 结论写成“稳定提升”

### 11.4 论文规范类
- 暴露内部日志命名
- 附录没有完整超参数
- 结果不可追溯到配置哈希
- 用“validate”描述非显著结果（应写 directional evidence）

---

## 十二、结果解释模板（写论文时直接用）

> 下面是你实验重构后可直接套用的措辞模板（按统计强度分级）。

### 12.1 若显著（Validated）
- “在完整 LongBench 全任务评测、统一注入路径和等预算 LoRA 协议下，作者方法相对 Geometric/PI/YaRN/LongRoPE 在宏平均指标上取得统计显著提升（配对 bootstrap 95% CI 不跨 0；permutation test p < ...），且在 2-3 个随机种子下趋势一致。”

### 12.2 若未显著但方向一致（Directional evidence）
- “在当前计算预算与样本规模下，任务级配对检验尚未达到常规显著性阈值；但 across tasks / lengths / seeds 的趋势与理论预测的方向性 trade-off 一致，因此我们将该结果定位为方向性机制证据，而非最终性能定论。”

### 12.3 若部分不支持（Honest reporting）
- “在某些任务/长度区间，作者方法未优于 LongRoPE（或差异不稳定）。这些反例提示理论 surrogate 与工程离散化/优化动态之间仍存在 gap，后续工作需进一步建模。”

---

## 十三、建议的执行优先级（按影响排序）

### P0（决定生死，必须先做）
1. 修复 8B 管线（恢复合理基线）
2. 全量 LongBench（禁止 preview-only）
3. 补 LongRoPE + NTK-aware 基线
4. 8B 至少 2-3 seeds + per-sample traces
5. 统计分析（CI + permutation + effect size）

### P1（显著提升说服力）
6. Anchoring 机制完整消融
7. 基础能力守护测试（证明没训坏模型）
8. TinyStories 主对比补齐 3 seeds + 多规模
9. 预览集 vs 全量集差异审计

### P2（增强论文完成度与审稿体验）
10. 复现实验附录规范化（替换内部日志痕迹）
11. 结果导出自动化（LaTeX 表 + PNG 图）
12. 结论强度自动分级文案生成
13. 计算预算与许可证/数据清单整理

---

## 十四、给你（作者）的一点策略建议（实验侧）

1. **先别急着重跑所有方法**  
   先把 `8B baseline pipeline audit` 做干净。只要这个阶段没过，后面所有对比都不可信。

2. **优先修“绝对分数可信度”，再修“相对提升显著性”**  
   审稿人会先问：模型是不是被你训坏了？只有答清楚这个，后面的 +x% 才有意义。

3. **如果算力不够，宁可少方法，也不要少评测完整性**  
   `5个方法 × 全量评测 × 2 seeds` 通常比 `8个方法 × preview-only × 1 seed` 更有发表价值。

4. **把“方向性证据”写法准备好**  
   即便结果最终仍未显著，只要管线正确、统计透明、机制一致，论文依然能从“可疑实验”升级为“可信的机制验证”。

---

## 十五、最终交付清单（你验收 Codex 用）

### 必须有（缺一不可）
- [ ] `artifacts/audit/pipeline_root_cause_report.md`
- [ ] `artifacts/audit/prompt_template_audit.md`
- [ ] `artifacts/audit/scoring_audit.md`
- [ ] `artifacts/audit/injection_path_equivalence.md`（如适用）
- [ ] `artifacts/repro_manifest/*`（环境、哈希、版本）
- [ ] `artifacts/traces/*`（LongBench 全任务 per-sample）
- [ ] `artifacts/stats/main_stat_report.md`
- [ ] `paper_exports/main_tables/*.tex`
- [ ] `paper_exports/main_figures/*.pdf|png`
- [ ] `paper_exports/appendix_tables/hyperparameter_configurations.tex`
- [ ] `paper_exports/appendix_tables/dataset_model_license_table.tex`
- [ ] `paper_exports/appendix_figures/anchoring_ablation*.pdf|png`
- [ ] `paper_exports/latex/conclusion_strength_wording.md`

### 最终验收标准（简版）
- [ ] 8B baseline 不再处于明显失效态
- [ ] 全量 LongBench、全量 traces、非 preview-only
- [ ] 补齐关键 SOTA 基线（LongRoPE + NTK-aware类至少一个）
- [ ] 多 seed + 配对统计 + 效应量
- [ ] 论文附录可复现、无内部日志痕迹
- [ ] 结论措辞与统计证据强度一致

---

## 十六、你可以直接对 Codex 下的第一条命令（推荐）

```md
请先执行 Phase 1（8B 管线审计），不要跑任何方法对比。目标是确认并修复 LongBench 评测与 LoRA/模板/打分流水线中的 root cause。先给出审计计划（含检查项、脚本入口、预计输出文件），然后执行零改动 8B-Instruct 全量 LongBench 基线评测并保存 per-sample traces。若基线绝对分数异常，进入 root cause 分析并输出修复前后对比报告。只有 Phase 1 门槛通过后，才进入 Phase 2。
```

---

## 十七、附：面向论文文本的实验叙述升级建议（你写作时用）

> 这是为了避免再次被审稿人抓“实验不严谨”。

### 17.1 主实验叙述模板（推荐）
- 先写 **protocol fairness**（统一注入路径、等预算）
- 再写 **pipeline audit**（为什么这次结果可信）
- 再写 **full benchmark coverage**（非预览子集）
- 再写 **statistics and uncertainty**（CI/p/effect size）
- 最后写 **mechanistic consistency**（与理论方向一致/不一致）

### 17.2 禁用表达（建议替换）
- 禁用：“validated” （当统计不显著时）
- 替换为：“provides directional evidence”, “is consistent with”, “suggests”
- 禁用：“SOTA” （如果没有完整强基线和标准协议）
- 替换为：“under the controlled protocol”, “in our constrained setting”

---

## 结束语（实验侧结论）

这份计划的核心不是“把结果修得更好看”，而是把实证部分从“容易被质疑的脆弱证据”升级为“可审计、可复现、可解释的严谨证据”。  
如果严格执行本计划，即使最终提升幅度不大，你的论文实证质量也会显著提升；如果提升同时保持存在，那么论文整体说服力会出现质变。
