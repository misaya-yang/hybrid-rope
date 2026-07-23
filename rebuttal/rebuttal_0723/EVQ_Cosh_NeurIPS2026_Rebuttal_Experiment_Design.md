# EVQ-Cosh NeurIPS 2026 Rebuttal：大规模训练与长上下文评测实验设计

> **2026-07-24 status:** `broad planning draft / not an approved runbook`.
> 本文保留 1.5B、RULER、FMRoPE 与消融的完整想法，但其 96GB RTX Pro
> 6000、每臂 4B-token 和三位 reviewer 映射尚未全部由仓库内原始 review
> 验证，也不匹配当前 5090 窄实验路线。实际优先级与授权以同目录
> `README.md` 为准；不得仅凭本文的 “P0” 标签启动付费实验。

> **2026-07-24 Geo identity correction:** 本文后续出现的论文主对照统一解释为
> `Paper-Geo`，即 \(u_k=(k+\tfrac12)/K\)，不是 standard RoPE 的
> \(u_k=k/K\)。`EVQ-Cosh` 使用同一个 midpoint quantizer；`Std-Geo`
> 只在 15M-token `shape_l128` 中做 seed-42 小消融。精确公式、float32
> 通道值、SHA-256 与历史来源见 `FREQUENCY_DEFINITION_MANIFEST.json`。

> **用途**：直接交给 Codex，在现有 EVQ-Cosh / hybrid-rope 代码仓库中完成实验审计、实现、训练、评测与结果汇总。  
> **实验目标**：以一组严格配对的 1.5B 级从头预训练模型为主证据，补充 RULER、FMRoPE 对照和频率分配消融，集中回应 NeurIPS 2026 三位审稿人关于模型规模、下游任务、方法新颖性和理论—实践归因的质疑。  
> **硬件约束**：单张 NVIDIA RTX PRO 6000 Blackwell，96GB 显存。  
> **第一阶段预算**：Paper-Geo 与 EVQ-Cosh 各训练 4B tokens；必须保存完整 checkpoint，后续可继续至 6B/8B tokens。  
> **核心原则**：除 RoPE inverse-frequency initialization 外，Paper-Geo 与 EVQ 的所有变量必须完全一致。

---

## 0. Codex 的首要任务

在修改代码和启动正式训练之前，先完成以下工作：

1. 审计现有仓库中的模型实现、EVQ initializer、Paper-Geo/Std-Geo baseline、YaRN、数据流水线、checkpoint 恢复逻辑和已有评测脚本。
2. 找到论文提交版本实际使用的：
   - Paper-Geo 与 Std-Geo 指数网格约定；
   - EVQ 的 endpoint / midpoint quantization 约定；
   - `inv_freq` 是否作为 buffer 保存；
   - checkpoint 加载后是否会被模型构造函数重新覆盖；
   - YaRN 推理时是否基于 checkpoint 中真实的 Paper-Geo/EVQ `inv_freq` 进行变换。
3. 不允许为了“代码更漂亮”迁移训练框架。优先复用已经验证过的 trainer、数据管线和优化器配置。
4. 在所有单元测试、配对一致性测试和 1,000-step 预训练 smoke test 通过之前，禁止启动正式 4B-token 训练。
5. 所有实现必须形成：
   - 可复现配置文件；
   - 固定命令；
   - 自动生成的实验 manifest；
   - 结果 CSV/JSON；
   - 可直接用于 rebuttal 的表格和图。

---

# 1. 评审问题与实验映射

## 1.1 需要回答的问题

### Reviewer zWsa

主要质疑：

- 与 FMRoPE 方法和动机高度重合；
- 缺少 FMRoPE 直接比较；
- 模型规模不足；
- 缺少 RULER；
- 需要至少约 1B–7B 规模的证据。

对应实验：

- **E1：1.5B Paper-Geo vs EVQ 从头预训练**
- **E2：1.5B RULER**
- **E3：FMRoPE / EVQ / EVQ+FMRoPE 直接比较**

### Reviewer 27bE

主要质疑：

- `C_app → pure-tether cosh → tau*` 是多层近似/选择，现有实验没有独立归因；
- 实际使用的 `tau≈4` 超出 small-tau 渐近推导的直接适用区域；
- 需要 predicted tau 与 tuned tau 比较；
- 需要 cosh 与非 cosh 解析 schedule 比较；
- 需要 held-out base、较大 `d_head` 和更大模型。

对应实验：

- **E1：1.5B，`d_head=128`，held-out base = 1M**
- **E4：predicted tau vs tuned tau**
- **E5：cosh vs power / exponential / piecewise schedule**
- **E6：surrogate → exact kernel → trained model 三层归因**

### Reviewer Dz6s

主要质疑：

- 真实长上下文任务不足；
- Paper-Geo+YaRN 是否经过充分优化；
- 理论 surrogate、exact collision 和训练结果之间应明确分层。

对应实验：

- **E2：完整 RULER 或 competence-qualified RULER**
- **E7：Raw 与 matched-YaRN 双模式评测**
- **E6：三层归因表**

---

# 2. 实验优先级

## P0：必须完成

1. 1.5B Paper-Geo / EVQ 严格配对训练，至少各 4B tokens。
2. 1B、2B、4B token checkpoint 的验证 PPL 与训练曲线。
3. RULERv1：4K / 8K / 16K / 32K。
4. Raw RoPE 与 matched YaRN 两种推理模式。
5. 防止 `inv_freq` 被错误覆盖的自动测试。
6. 可直接放进 rebuttal 的结果表与 95% 置信区间。

## P1：强烈建议完成

1. FMRoPE / Paper-Geo / EVQ 小规模直接比较，另做一次 Std-Geo 小消融。
2. predicted tau vs independently tuned tau。
3. 至少两种非 cosh schedule 的匹配消融。
4. exact-kernel collision、effective rank、alive-channel count 与训练结果联合报告。

## P2：有余力再做

1. 1.5B 第二个 seed 的 1B–2B token 方向性验证。
2. RULERv2。
3. 1.5B 继续至 6B/8B tokens。
4. 更完整的 base sweep 或 `d_head` sweep。
5. 极轻量通用 format-adaptation SFT，仅在 base 模型无法理解 RULER 格式时启用。

---

# 3. 主实验 E1：1.5B 级严格配对预训练

## 3.1 模型架构

采用 LLaMA-style decoder-only Transformer，但不要引入 GQA、MoE、滑窗注意力等额外变量。

```yaml
model:
  architecture: llama_style_decoder
  num_hidden_layers: 28
  hidden_size: 2048
  num_attention_heads: 16
  num_key_value_heads: 16
  head_dim: 128
  intermediate_size: 5632
  activation: swiglu
  norm: rmsnorm
  norm_eps: 1.0e-5
  pre_norm: true
  attention_bias: false
  mlp_bias: false
  dropout: 0.0
  tie_word_embeddings: true
  max_train_sequence_length: 4096
  rotary_dim: 128
  rope_base: 1000000
```

预期参数量：

- vocab = 32,000 时约 **1.504B**
- vocab ≈ 49K–50K 时约 **1.54B**
- 最终参数量必须由代码真实计算并写入 manifest，不得手工估计。

### 参数量验收

```text
目标区间：1.50B ≤ N_params ≤ 1.56B
```

如果现有 tokenizer 导致参数量超出该区间，优先调整 `intermediate_size`，不要改变：

- `hidden_size=2048`
- `num_heads=16`
- `head_dim=128`
- `num_layers=28`

这三个维度是回应 reviewer 的重要实验条件。

---

## 3.2 RoPE 设置

### Std-Geo、Paper-Geo 与 EVQ-Cosh

令 \(K=d_{\mathrm{head}}/2,\ k=0,\ldots,K-1\)。定义不得再混写：

```text
Std-Geo:   u_k = k/K,         omega_k = b^(-k/K)
Paper-Geo: u_k = (k+1/2)/K,   omega_k = b^(-(k+1/2)/K)
EVQ-Cosh:  same Paper-Geo u_k,
           phi_k = 1 - asinh((1-u_k)sinh(tau))/tau,
           omega_k = b^(-phi_k)
```

Paper-Geo 是投稿主链路的几何对照。相对 Std-Geo，
\(\omega_k^{paper}/\omega_k^{std}=b^{-1/d_{\mathrm{head}}}<1\)，所以是全通道
降频，波长放大 \(b^{1/d_{\mathrm{head}}}\)。它不能等效成一个统一的新 standard
base，因为 standard grid 的第 0 通道对任意 base 都为 1。
若仅对 \(k>0\) 逐通道反解，则
\(b_{\mathrm{eff},k}=b^{1+1/(2k)}\)，它随通道变化，进一步说明不存在
统一的等效 base。

### EVQ-Cosh

保持：

```text
base b = 1,000,000
d_eff = d_head = 128
L_train = 4096
tau* = d_eff / sqrt(L_train) = 2.0
```

即：

```text
omega_k_EVQ = b^(-phi_k(tau))
```

### 必须验证

1. Paper-Geo 与 EVQ 的 `base` 完全相同。
2. 唯一变化是指数位置 `u_k → phi_k(tau)`。
3. 模型参数、优化器参数、数据、随机种子完全相同。
4. `tau→0` 时，EVQ 必须逐字节恢复 Paper-Geo float32 grid。
5. Std-Geo endpoint 版本只能作为附加消融，不能静默替换论文方法。
6. 输出所有频率：
   - `u_k`
   - `phi_k`
   - `omega_k`
   - wavelength
   - phase rotation at 4K / 8K / 16K / 32K

---

## 3.3 配对初始化

禁止分别调用随机初始化后“假设 seed 相同即可”。

正确流程：

1. 固定初始化 seed，例如 `42`。
2. 构造一次基础模型。
3. 保存 `init_state_dict`。
4. Paper-Geo 和 EVQ 都从该文件加载完全相同的可训练权重。
5. 加载后仅设置各自非训练的 RoPE frequency buffer。
6. 输出权重 hash。

必须满足：

```text
hash(trainable_parameters_geo_at_step0)
==
hash(trainable_parameters_evq_at_step0)
```

允许不同的仅有：

```text
RoPE inv_freq / exponent-grid buffers
```

### Step-0 行为测试

- 位置 0 的 logits 应一致到数值误差范围；
- 长度大于 1 时 logits 应出现预期差异；
- 禁止出现除 RoPE 外的 config diff。

---

# 4. 数据设计

## 4.1 主数据集

使用：

```text
HuggingFaceFW/FineWeb-Edu sample-100BT
```

原因：

- 与现有论文主要文本实验保持一致；
- 数据规模足够支持后续继续训练；
- 避免因为新混入多个语料库而产生额外归因问题。

## 4.2 不允许混入的任务

主 1.5B 预训练中禁止加入：

- RULER 模板；
- NIAH / passkey 模板；
- variable tracking；
- common-word / frequent-word extraction；
- RULER 风格 multi-key / multi-value；
- SQuAD / HotpotQA 的任务格式；
- 为评测专门生成的合成长上下文样本。

目的：

> 让 RULER 成为真正的 out-of-distribution 长上下文测评，而不是任务训练效果。

论文原有 passkey-mix 结果可以保留，但本次 1.5B 主实验必须使用纯自然文本。

## 4.3 Tokenizer

优先沿用现有论文训练使用的 tokenizer，避免 tokenizer 成为新变量。

必须保存：

```text
tokenizer name
tokenizer files
vocab size
special token IDs
tokenizer hash
```

Paper-Geo 与 EVQ 必须引用同一个不可变 tokenizer 目录。

## 4.4 数据预处理

推荐先预 tokenize 到本地 NVMe，避免正式训练受网络和 CPU tokenization 限制。

要求：

1. 文档级 hash 划分 train / validation。
2. 同一文档不得跨 split。
3. 保留文档连续性：
   - 长文档优先切成连续 4096-token windows；
   - 短文档可以使用 EOS 拼接；
   - 不改变论文现有的 loss mask 规则；
   - Paper-Geo / EVQ 读取完全相同的 token shard。
4. 固定 shard 顺序与 sample order。
5. 生成 `dataset_manifest.json`，记录：
   - 原始数据版本；
   - shard 列表；
   - 每个 shard 的 token 数；
   - 文档数；
   - tokenizer hash；
   - shuffle seed；
   - manifest hash。

## 4.5 最小去污染

RULERv1 的部分上下文来源包括 Paul Graham essays、SQuAD 和 HotpotQA。

实现以下最低限度的过滤：

1. 下载本次 RULER 使用的源文本。
2. 对源文本构造规范化 64-token 或 128-token shingle hash。
3. FineWeb-Edu 预处理中剔除包含长 exact-match shingle 的文档。
4. 记录：
   - 扫描文档数；
   - 命中文档数；
   - 被移除 token 数；
   - 过滤规则。

不要声称实现了完整语义去污染，只能写：

```text
Exact long-span decontamination against the RULER source passages.
```

---

# 5. 训练超参数

以论文现有 454M / 750M 配置为基础，保持优化器家族一致。

```yaml
training:
  precision: bf16
  optimizer: fused_adamw
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  weight_decay: 0.1
  peak_learning_rate: 2.0e-4
  min_learning_rate: 2.0e-5
  lr_schedule: cosine
  warmup_ratio: 0.02
  gradient_clip_norm: 1.0
  dropout: 0.0
  sequence_length: 4096
  global_batch_tokens: 262144
  target_tokens_stage_a: 4000000000
  flash_attention: true
  activation_checkpointing: true
  fp8: false
```

### 为什么 Stage A 单独按 4B tokens 设计 schedule

第一阶段 rebuttal 的正式比较点就是 4B tokens，因此 Stage A 应在 4B 结束时完成学习率衰减，而不是把 4B 当成 8B schedule 的中点。

后续继续训练时，使用完全匹配的 Stage B continual-pretraining schedule。必须明确：

> 从 4B 继续训练是合法的 matched continuation，但不是“原始 8B 单一 cosine 轨迹”的无缝延伸。

### Stage B 建议

```yaml
continuation:
  start_from_tokens: 4000000000
  target_total_tokens: 8000000000
  peak_learning_rate: 5.0e-5
  min_learning_rate: 1.0e-5
  warmup_tokens: 50000000
  schedule: cosine
```

Paper-Geo 与 EVQ 必须使用完全相同的 continuation schedule。

---

## 5.1 Batch 与显存调优

固定：

```text
global_batch_tokens = 262,144
```

候选：

```text
micro_batch_sequences = 1 or 2
gradient_accumulation_steps =
global_batch_tokens / (micro_batch_sequences × 4096)
```

示例：

- micro batch = 2 sequences
- 每 micro batch = 8192 tokens
- grad accumulation = 32
- global batch = 262,144 tokens

必须优先调整 micro batch 和 gradient accumulation，不得因为 OOM 改变两组的 global batch。

### 显存和性能要求

- 峰值显存目标：≤ 90GB
- 禁止 optimizer CPU offload，除非不 offload 无法训练；
- 禁止在正式主实验中临时切换 FP8；
- `torch.compile` 只能在无编译版本通过正确性测试后启用；
- 数据加载不能成为主要瓶颈。

---

# 6. 吞吐预检与时间决策

两组各 4B tokens，总计 8B tokens。

五天内完成纯训练所需平均吞吐约：

```text
8B / (5 × 86400) ≈ 18.5K tokens/s
```

考虑 checkpoint、验证和故障恢复，正式目标应为：

```text
≥ 22K tokens/s
```

## 6.1 Microbenchmark

先完成：

1. 50-step warmup；
2. 300-step 稳态 benchmark；
3. 不包含数据下载时间；
4. 记录：
   - tokens/s；
   - TFLOP/s 或 MFU；
   - peak VRAM；
   - step time；
   - dataloader wait；
   - GPU utilization；
   - 是否有 loss spike。

## 6.2 Go / No-Go

| 实测吞吐 | 决策 |
|---|---|
| ≥22K tok/s | 按计划跑两组各 4B |
| 18.5K–22K | 可以跑，但减少非关键在线评测并提前生成 RULER 数据 |
| 14K–18.5K | 先保证两组都到 2B，再交替继续到 4B |
| <14K | 必须先优化 kernel/data pipeline；禁止先完整跑完某一组 |

### 关键原则

不要先把 Paper-Geo 跑到 4B，再发现时间不足以完成 EVQ。

推荐阶段性交替：

```text
Paper-Geo 0→0.5B
EVQ 0→0.5B
Paper-Geo 0.5→2B
EVQ 0.5→2B
Paper-Geo 2→4B
EVQ 2→4B
```

这样无论何时截止，都至少拥有相同 token 数的可比 checkpoint。

---

# 7. Checkpoint 与恢复要求

## 7.1 完整 checkpoint 必须包含

- model weights；
- optimizer states；
- LR scheduler states；
- tokens seen；
- optimizer steps；
- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- Torch CUDA RNG；
- dataloader shard / document / token cursor；
- gradient accumulation state；
- config；
- git commit；
- environment info；
- tokenizer hash；
- dataset manifest hash；
- Paper-Geo / EVQ frequency manifest。

## 7.2 保存点

建议：

```text
0.5B：完整 resume checkpoint
1.0B：完整 + HF weights export
2.0B：完整 + HF weights export
3.0B：weights-only
4.0B：完整 + HF weights export
```

同时保留一个 rolling `latest`，避免磁盘爆炸。

## 7.3 恢复一致性测试

在 smoke test 中：

1. 连续训练 1,200 steps；
2. 另一条路径训练 1,000 steps，保存并恢复，再训练 200 steps；
3. 比较 loss、参数和数据游标。

允许 CUDA 非确定性带来的极小误差，但必须确认：

- 没有重复或跳过大段数据；
- scheduler 没有重置；
- optimizer moments 没有丢失；
- `inv_freq` 没有被覆盖。

---

# 8. 主验证指标

## 8.1 训练长度内

在相同 held-out FineWeb-Edu 上报告：

- validation NLL；
- PPL@1K；
- PPL@2K；
- PPL@4K；
- 最后 25% token 的 tail NLL；
- 训练 loss 曲线；
- Paper-Geo–EVQ loss gap 随 tokens seen 的变化。

预注册非破坏性标准：

```text
EVQ PPL@4K 相对 Paper-Geo 的代价 ≤ 2%
```

该阈值是 rebuttal 叙事目标，不得删除不利结果。

## 8.2 长度外推 PPL

准备独立的自然长文档集合，测试：

```text
4K / 8K / 16K / 32K
```

每个长度：

- 使用相同文档和相同 token span；
- 不允许 Paper-Geo / EVQ 使用不同的样本；
- 报告 full-sequence NLL；
- 报告 tail-25% NLL；
- 报告 bootstrap 95% CI；
- 同时给 raw 和 matched YaRN。

---

# 9. 主实验 E2：RULERv1

## 9.1 版本锁定

使用 NVIDIA 官方 RULER 仓库：

```text
branch: rulerv1-ns
```

必须锁定具体 git commit，并写入结果 manifest。

RULERv2 可作为 P2，不要在 rebuttal 主结果中混用两个版本。

## 9.2 13 个任务

完整任务集合：

```text
niah_single_1
niah_single_2
niah_single_3
niah_multikey_1
niah_multikey_2
niah_multikey_3
niah_multivalue
niah_multiquery
vt
cwe
fwe
qa_1
qa_2
```

类别：

1. retrieval；
2. multi-hop tracing；
3. aggregation；
4. QA。

## 9.3 评测长度

```text
4K
8K
16K
32K
```

4K 是训练长度内 anchor；8K/16K/32K 分别是 2×/4×/8× 外推。

## 9.4 推理模式

### Mode A：Raw

- 不改变频率表；
- 只扩大允许的最大位置；
- Paper-Geo 与 EVQ 都使用训练得到的原始 frequency table。

### Mode B：Matched YaRN

- 8K：相同 scale = 2
- 16K：相同 scale = 4
- 32K：相同 scale = 8
- 使用论文现有 YaRN 实现和完全相同的超参数；
- 不允许单独调优 EVQ 或 Paper-Geo。

### 强制测试

在每次评测启动时输出：

```text
checkpoint inv_freq hash
pre-YaRN inv_freq hash
post-YaRN inv_freq hash
```

并确认：

- Paper-Geo+YaRN 基于 Paper-Geo checkpoint；
- EVQ+YaRN 基于 EVQ checkpoint；
- 模型构造函数未重新创建统一的本地 `inv_freq` 覆盖 checkpoint。

这是最高优先级的正确性风险。

---

## 9.5 Base 模型 prompt

模型是纯预训练 base model，不使用 chat template。

优先采用 RULER 官方 base completion template 和官方 one-shot 设置。

规则：

1. prompt 在比较 Paper-Geo / EVQ 之前锁定；
2. 可以使用不属于正式 test set 的少量 development samples 验证格式；
3. 不允许根据 Paper-Geo / EVQ 的胜负分别调 prompt；
4. 两个模型使用完全相同的 prompt 和 stop conditions；
5. greedy decoding：
   - `temperature=0`
   - `do_sample=false`
   - `top_p=1`
6. `max_new_tokens` 使用任务官方默认或按任务固定；
7. 记录 invalid-output rate。

---

## 9.6 样本量

### Smoke test

```text
32 samples / task / length
```

目的：

- 检查 prompt；
- 检查生成格式；
- 检查 32K 是否 OOM；
- 检查 scorer。

### Rebuttal 最低标准

```text
200 samples / task / length
```

### 最终推荐

```text
500 samples / task / length
```

必须为 Paper-Geo 与 EVQ 生成完全相同、固定 seed 的 RULER 样本。

---

## 9.7 4K competence gate

1. 所有 13 个任务均必须完整报告，不能隐藏失败任务。
2. 额外定义一个预注册的 `competence-qualified subset`：

```text
任务在 Paper-Geo 与 EVQ 的 4K 平均分 ≥ 50
且两者 invalid-output rate 均 < 10%
```

3. 该筛选仅基于 4K 短上下文能力，不得查看 8K+ 的胜负。
4. 同时报告：
   - full 13-task macro；
   - competence-qualified macro；
   - category average；
   - per-task score。

目的：

> 区分“模型根本不会做该任务”和“上下文变长后能力退化”。

---

## 9.8 RULER 主指标

### Absolute score

```text
Score(L)
```

### Retention

```text
Retention(L) = Score(L) / Score(4K)
```

### Effective context length

报告两个相对定义：

```text
ECL_90：最长的 Retention ≥ 90% 的长度
ECL_80：最长的 Retention ≥ 80% 的长度
```

不要直接使用适用于 7B 模型的绝对 85.6 阈值来定义本 1.5B base model 的有效长度。

### 统计检验

- 同一批样本做 paired bootstrap；
- 10,000 次 bootstrap；
- 报告差值与 95% CI；
- 训练 seed 只有一个时，必须明确：
  - bootstrap 只反映 evaluation-sample uncertainty；
  - 不代表 training-seed variance。

---

# 10. RULER 失败时的处理

## 情况 A：只是一部分输出格式错误

先改进统一 base prompt：

- 增加一个官方 one-shot demonstration；
- 固定明确的答案前缀；
- 统一 max generation；
- 用 development set 锁定后再跑正式 test。

禁止分别为 Paper-Geo / EVQ 调 prompt。

## 情况 B：两组在 4K 几乎都不会做聚合/QA

1. 仍完整报告结果；
2. retrieval / VT 等有动态范围的任务可以用于主分析；
3. 使用 competence-qualified macro；
4. 不得声称“完整 RULER 优势”。

## 情况 C：所有任务 4K 都接近随机

此时优先级：

1. 检查 tokenizer、prompt、generation 和 scorer；
2. 检查模型是否训练不足；
3. 暂不做 RULER-specific SFT；
4. 若必须做 format adaptation，只能启用 P2 备用方案。

### P2：备用 format-adaptation

仅在明确记录 base 结果后启用：

- Paper-Geo / EVQ 使用同一份通用 instruction / QA 数据；
- 不含 RULER、NIAH、VT、CWE、FWE；
- 不含 SQuAD / HotpotQA；
- 不使用 RULER prompt 模板；
- 相同初始化 checkpoint、数据顺序、步数、LR；
- 训练量控制在 20M–50M tokens；
- base 与 adapted 结果必须同时报告。

当前授权的数据准备实现位于
`experiments/rebuttal_2026/sft_distillation/`。它不是 RULER-specific
训练：程序先生成可解的事实世界与唯一 oracle，DeepSeek 仅自然化报告框架、
问题和无关干扰段落；关键事实由程序逐字插入并复核。100 条审计集未完成
程序门禁和逐条人工确认前，流水线会拒绝生成 3000/400/400 pilot。最终
Paper-Geo 与 EVQ 必须读取相同 `messages/train.jsonl` SHA-256 和顺序。

---

# 11. 主实验 E3：FMRoPE 直接比较

## 11.1 实现原则

Codex 必须先获取和阅读 FMRoPE / Oka et al. 的正式 ICLR 2026 论文与代码。

不得仅根据 reviewer 的描述实现。

需要明确记录：

- 训练时 base 设置；
- 推理时 base / interpolation 设置；
- 是否存在额外搜索；
- 原论文推荐超参数；
- 原论文的上下文长度协议。

## 11.2 最小受控矩阵

当前可执行配置以 `fmrope_125m_l256/SPEC.md` 为准：

```yaml
scale: 125M
train_length: 256
training_tokens: 100M
seed: 42
dataset: FineWeb-Edu
```

方法：

| ID | 方法 | Base | 指数网格 |
|---|---|---:|---|
| F0 | Paper-Geo | 500K | midpoint geometric |
| F1 | FMRoPE | train base=256；infer base=L | Std-Geo endpoint |
| F2 | EVQ-Cosh | 500K | Paper-Geo midpoint + cosh warp |

F0–F2 是完整配对；不预加 EVQ+FMRoPE、best-base sweep 或额外 seed。Std-Geo
只在 `reviewer27be_shape_base/shape_l128` 用 15M tokens、seed 42 做附加消融。

## 11.3 该实验回答的核心

- FMRoPE 是否只是改变全局 base；
- EVQ 是否在与 Paper-Geo 相同的 quantizer/base 下改变指数/通道密度；
- FMRoPE 的 target-length base retarget 是否足以解释同一外推现象。

## 11.4 报告指标

- in-range PPL；
- 2× / 4× / 8× PPL；
- paired final-128-token NLL（primary）；
- full-window NLL（diagnostic）；
- checkpoint / sidecar / runtime `inv_freq` hash。
- alive-channel count；
- 频率表可视化；
- EVQ+FMRoPE 是否优于任一单独方法。

注意：

> 即使 EVQ+FMRoPE 没有叠加提升，也不能伪造“互补”；应改为说明二者参数化不同但经验收益可能重叠。

---

# 12. 主实验 E4：predicted tau vs tuned tau

## 12.1 目的

将两个问题分开：

1. cosh shape 是否有效；
2. `tau*=d_eff/sqrt(L)` 是否能落入优秀 operating basin。

## 12.2 实验设计

固定 cosh family，只改变 tau。

建议 grid：

```text
tau / tau* ∈ {0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
```

其中：

```text
tau=0 为 Paper-Geo
tau*=d_eff/sqrt(L)
```

优先复用论文已有 Phase-16 / 99-run sweep 数据。先审计已有结果，只有缺失时再训练。

## 12.3 结果表达

对每个配置报告：

- predicted tau；
- validation-optimal tuned tau；
- predicted 点与最优点的 PPL gap；
- flat basin 宽度；
- exact collision gap；
- 是否出现 in-range waterbed cost。

主结论应限制为：

```text
The rule selects a robust operating region.
```

不要写：

```text
The rule analytically finds the global optimum.
```

---

# 13. 主实验 E5：cosh 与非 cosh schedule 消融

## 13.1 公平比较原则

不同 schedule 不能直接“使用相同 tau”，因为 tau 是 cosh 特有参数。

应使用共同的 warp 强度进行匹配：

```text
D_warp = sqrt(mean_k[(phi_k - u_k)^2])
```

可额外报告：

```text
mean absolute displacement
Burg / Pearson load divergence
alive-channel count
```

## 13.2 候选 schedule

1. Paper-Geo：
   ```text
   phi(u)=u
   ```

2. EVQ-Cosh：
   ```text
   论文中的 inverse-CDF warp
   ```

3. Power warp：
   ```text
   phi(u)=u^gamma
   ```

4. Exponential warp：
   ```text
   phi(u)=log(1+(exp(a)-1)u)/a
   ```

5. Piecewise-linear warp：
   - 固定 knee，例如 `u=0.5`；
   - 两段斜率；
   - 保持单调和端点一致。

对非 cosh 方法数值求参数，使其 `D_warp` 与 EVQ-Cosh predicted tau 匹配。

## 13.3 控制变量

- 相同 RoPE operator；
- 相同 base；
- 相同端点约定；
- 相同通道数；
- 相同模型；
- 相同初始化；
- 相同数据；
- 相同训练 token；
- 只换固定 frequency schedule。

## 13.4 优先级

该消融不需要放到 1.5B。

推荐：

```text
50M / 125M
200M–500M tokens
seed 42
```

若时间允许，再对前两名 schedule 做 3 seeds。

---

# 14. 主实验 E6：surrogate → exact kernel → trained model

为回应两位 reviewer，需要为每种 schedule 输出同一行三层证据。

## Layer 1：Surrogate

- `C_app`；
- stiffness；
- utility proxy；
- stationary family 信息。

## Layer 2：Exact RoPE kernel diagnostic

- exact collision score；
- collision reduction vs Paper-Geo；
- effective rank；
- phase variance；
- alive/dead channel count；
- per-distance collision curve。

## Layer 3：Trained model

- in-range PPL；
- extrapolation PPL；
- RULER；
- retrieval；
- attention-distance 或相关训练后诊断。

最终生成类似表格：

| Schedule | `C_app` | Exact collision | Effective rank | PPL@4K | PPL@16K | RULER@16K |
|---|---:|---:|---:|---:|---:|---:|
| Paper-Geo | | | | | | |
| EVQ predicted | | | | | | |
| EVQ tuned | | | | | | |
| Power matched | | | | | | |

必须明确：

- `C_app` 下的 cosh 解是条件精确；
- exact kernel 只做功能验证；
- trained Transformer 结果是经验结果；
- 三者不能被写成同一个定理。

---

# 15. 主实验 E7：Paper-Geo+YaRN 是否充分优化

## 15.1 最小比较

在同一 checkpoint 和同一评测长度上：

```text
Paper-Geo raw
EVQ raw
Paper-Geo + matched YaRN
EVQ + matched YaRN
```

## 15.2 可选 scale sweep

若 reviewer 强调 Paper-Geo+YaRN 可能未调优，可在小验证集上统一 sweep：

```text
scale multiplier around nominal:
{0.75, 1.0, 1.25}
```

规则：

- Paper-Geo / EVQ 使用相同 sweep grid；
- 用同一个 validation criterion；
- test set 只运行锁定后的超参数；
- 同时报告 matched-scale 主结果和 independently tuned 辅助结果。

不要只给 EVQ 调优。

---

# 16. 统计和报告规范

## 16.1 大模型单 seed

1.5B 主实验大概率只能做一个训练 seed。

必须通过以下方式增强可信度：

- 完全相同初始化；
- 完全相同数据顺序；
- 多个 token checkpoints；
- 相同 evaluation items；
- paired bootstrap；
- 小模型已有 multi-seed 结果作为独立支撑。

不得把 eval bootstrap CI 描述成训练 seed 方差。

## 16.2 差值

同时报告：

```text
absolute difference
relative difference
95% CI
```

PPL：

```text
relative PPL change = (EVQ - Paper-Geo) / Paper-Geo
```

RULER：

```text
absolute percentage-point change
retention change
```

## 16.3 不允许的分析

- 只挑 EVQ 获胜的任务；
- 根据 test set 调 prompt；
- 根据 8K/16K 结果定义“有效任务”；
- 隐藏 4K 退化；
- 把单 seed 结果写成稳定 scaling law；
- 把 FMRoPE 与 EVQ 的动机差异夸大成“完全无关”。

---

# 17. 自动化目录建议

Codex 应适配现有仓库，而不是强制重构。建议新增：

```text
experiments/rebuttal_2026/
├── README.md
├── manifests/
├── configs/
│   ├── model_1p5b.yaml
│   ├── train_paper_geo_1p5b_4b.yaml
│   ├── train_evq_1p5b_4b.yaml
│   ├── continue_paper_geo_1p5b_8b.yaml
│   ├── continue_evq_1p5b_8b.yaml
│   ├── ruler_raw.yaml
│   ├── ruler_yarn.yaml
│   ├── fmrope_ablation.yaml
│   └── schedule_ablation.yaml
├── scripts/
│   ├── audit_repo.py
│   ├── build_init_checkpoint.py
│   ├── prepare_fineweb_shards.py
│   ├── build_dataset_manifest.py
│   ├── verify_paired_configs.py
│   ├── benchmark_throughput.py
│   ├── train.py
│   ├── export_hf_checkpoint.py
│   ├── run_ppl_eval.py
│   ├── run_ruler.py
│   ├── run_kernel_diagnostics.py
│   ├── aggregate_results.py
│   └── make_rebuttal_tables.py
├── tests/
│   ├── test_geo_evq_only_diff.py
│   ├── test_tau_zero_recovers_geo.py
│   ├── test_inv_freq_checkpoint_roundtrip.py
│   ├── test_yarn_uses_checkpoint_inv_freq.py
│   ├── test_resume_equivalence.py
│   ├── test_dataset_order_pairing.py
│   └── test_ruler_scorer.py
└── results/
    ├── raw/
    ├── tables/
    └── figures/
```

---

# 18. Run ID 规范

```text
scale15b_l4k_b1m_seed42_geo
scale15b_l4k_b1m_seed42_evq_tau2
scale15b_l4k_b1m_seed42_geo_yarn
scale15b_l4k_b1m_seed42_evq_tau2_yarn
fmrope125m_l2k_seed42_geo_b500k
fmrope125m_l2k_seed42_fmrope
fmrope125m_l2k_seed42_evq
fmrope125m_l2k_seed42_evq_fmrope
```

每个 run 自动生成：

```json
{
  "run_id": "...",
  "git_commit": "...",
  "config_hash": "...",
  "init_weight_hash": "...",
  "tokenizer_hash": "...",
  "dataset_manifest_hash": "...",
  "frequency_manifest_hash": "...",
  "hardware": "...",
  "cuda": "...",
  "torch": "...",
  "tokens_seen": 0
}
```

---

# 19. 正式训练前必须通过的测试

## Correctness Gate

- [ ] Paper-Geo / EVQ trainable weight hash 一致
- [ ] config diff 仅包含 RoPE schedule
- [ ] `tau→0` 恢复 Paper-Geo，而非 Std-Geo
- [ ] checkpoint roundtrip 后 `inv_freq` 不变
- [ ] YaRN 使用 checkpoint 的真实 `inv_freq`
- [ ] dataset token order 完全一致
- [ ] resume 不重置 optimizer / scheduler / data cursor
- [ ] RULER scorer 与官方小样例一致
- [ ] 32K 单样本推理不 OOM
- [ ] validation PPL pipeline 对同一 checkpoint 可重复

## Stability Gate

进行两组各 1,000 steps smoke test：

- [ ] 无 NaN / Inf
- [ ] grad norm 合理
- [ ] loss 均下降
- [ ] 两组 early loss 不出现无法解释的大幅分叉
- [ ] tokens/s 达到计划要求
- [ ] peak VRAM ≤ 90GB

只有全部通过，才能启动 4B-token 主训练。

---

# 20. 结果验收标准

这些是分析标准，不是删除负面实验的标准。

## Scale-transfer 成功

满足：

1. 1.5B 模型、`d_head=128`、base=1M；
2. Paper-Geo / EVQ 都完成相同 tokens；
3. PPL@4K 相对代价不超过约 2%；
4. 至少在 8K/16K/32K 中出现随长度增大的 EVQ 优势；
5. 结果与现有小模型机制方向一致。

## RULER 成功

至少满足其一：

- full macro 在 16K 或 32K 有可靠正差；
- competence-qualified macro 有可靠正差；
- retention 显著提高；
- ECL_80 / ECL_90 提升一个长度档位；
- matched YaRN 下 EVQ 保持优势。

## FMRoPE 回应成功

至少完成：

- 正确实现；
- 直接比较；
- 明确展示 base 与 within-span exponent allocation 的不同；
- 检验组合；
- 不依赖纯文字辩解。

## Attribution 成功

至少完成：

- predicted tau 与 tuned tau；
- Paper-Geo 与两个非 cosh schedule；
- surrogate / exact kernel / trained model 三层表。

---

# 21. 风险登记

| 风险 | 严重度 | 检测 | 处理 |
|---|---|---|---|
| Eval 时 checkpoint `inv_freq` 被覆盖 | 致命 | hash 与单元测试 | 先修复，重跑所有相关评测 |
| Paper-Geo / EVQ 数据顺序不同 | 致命 | token shard/sample ID 对照 | 统一 manifest 和 cursor |
| 只保存 weights，无法准确续训 | 高 | checkpoint schema | 保存 optimizer/scheduler/RNG/cursor |
| 4B tokens 下 RULER 4K 仍太弱 | 高 | 32-sample smoke | full + qualified subset；必要时启用 P2 |
| 吞吐不足 | 高 | 300-step benchmark | 配对分阶段；优化 kernel/data |
| 32K MHA 推理慢 | 中 | 单样本 profile | batch=1，预生成数据，分任务执行 |
| FMRoPE 实现依据二手描述 | 致命 | paper/code audit | 必须读取正式论文 |
| 训练 seed 只有一个 | 中 | 实验记录 | 强调配对设计与已有小模型多 seed |
| 非 cosh schedule 匹配不公平 | 高 | warp metric | 按 `D_warp` 匹配并公开参数 |
| 为了 RULER 混入相似训练任务 | 高 | dataset audit | 主训练禁止 benchmark-like data |
| 论文 quantization 与代码不一致 | 致命 | checked-in frequency manifest + checkpoint hash | 主实验固定 Paper-Geo；Std-Geo 仅作小消融 |

---

# 22. 建议执行时间线

## Day 0：审计与准备

- 仓库审计；
- 实现 correctness tests；
- 构建 1.5B config；
- 准备 FineWeb-Edu token shards；
- 锁定 RULER commit；
- 用现有小 checkpoint 调通 RULER；
- 完成 throughput benchmark。

## Day 1–4/5：配对训练

按：

```text
0.5B → 2B → 4B
```

交替推进 Paper-Geo 与 EVQ。

CPU 侧并行完成：

- RULER 数据生成；
- FMRoPE 论文实现审计；
- 表格脚本；
- exact-kernel diagnostics。

## 每个 checkpoint 完成后

立即运行：

- held-out PPL；
- 4K / 8K RULER smoke；
- frequency hash；
- checkpoint integrity test。

## 两组 4B 完成后

优先顺序：

1. 完整 raw RULER；
2. matched YaRN RULER；
3. extrapolation PPL；
4. FMRoPE；
5. schedule ablation；
6. 第二 seed / 继续训练。

---

# 23. Rebuttal 最终需要生成的表

## Table A：1.5B Scale Transfer

| Method | Params | `d_head` | Base | Tokens | PPL@4K | PPL@8K | PPL@16K | PPL@32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Table B：RULER

| Method | Mode | 4K | 8K | 16K | 32K | ECL_90 | ECL_80 |
|---|---|---:|---:|---:|---:|---:|---:|

另外给 category breakdown：

| Method | Length | Retrieval | VT | Aggregation | QA | Full Macro | Qualified Macro |
|---|---:|---:|---:|---:|---:|---:|---:|

## Table C：FMRoPE

| Method | Global Base | Grid Shape | Extra Params | PPL@L | PPL@4L | RULER@4L |
|---|---:|---|---:|---:|---:|---:|

## Table D：Shape / Tau Attribution

| Schedule | Warp Strength | Exact Collision | PPL@L | PPL@4L | Predicted or Tuned |
|---|---:|---:|---:|---:|---|

---

# 24. 最终需要生成的图

1. Paper-Geo / EVQ training loss vs tokens。
2. PPL vs context length，x 轴 log2。
3. RULER score vs context length。
4. RULER retention vs context length。
5. Paper-Geo / EVQ / FMRoPE 的频率指数网格，并单列 Std-Geo 小消融。
6. predicted tau 与 tuned tau 的 basin 图。
7. schedule 的 exact-kernel collision vs trained PPL scatter。
8. raw 与 YaRN 的二维对比图。

所有图必须：

- 自动从 CSV 生成；
- 不手工录入数据；
- 带 bootstrap CI；
- 标明 seed 数；
- 区分 primary 与 supporting evidence。

---

# 25. Codex 的最终交付物

1. `experiments/rebuttal_2026/README.md`
2. 所有 config 与固定命令
3. 审计报告：
   - 当前代码风险；
   - 修复内容；
   - 与论文提交版本的一致性
4. 单元测试与测试结果
5. 训练 manifest
6. 完整 checkpoint 恢复说明
7. RULER 集成
8. 结果 CSV / JSON
9. 自动生成的 LaTeX / Markdown 表格
10. 自动生成的 PNG / PDF 图
11. 每位 reviewer 的证据映射
12. 一页 `RESULTS_STATUS.md`：
    - 已完成；
    - 运行中；
    - 失败；
    - 不可解释；
    - 可用于 rebuttal；
    - 只能作为 supporting。

---

# 26. 禁止事项

- 禁止在主预训练中加入 RULER-like 合成任务。
- 禁止 Paper-Geo / EVQ 使用不同的数据顺序。
- 禁止为 EVQ 单独调优 YaRN，而 Paper-Geo 使用默认值。
- 禁止只保存模型权重而不保存完整训练状态。
- 禁止在未检查 checkpoint buffer 的情况下运行 RULER。
- 禁止把 FMRoPE 简化成未经论文确认的“改 base”实现。
- 禁止因为 4K 结果不利而只展示外推结果。
- 禁止从 test set 选择 prompt。
- 禁止使用不同的 tokenizer、batch、LR 或训练 token 数。
- 禁止把单 seed 结果描述为普遍 scaling law。
- 禁止把 surrogate proof 描述成 trained Transformer objective 的解析最优解。
- 禁止伪造、插值或估算未实际完成的实验结果。

---

# 27. 一句话实验定义

> Train a pre-specified 1.5B-parameter MHA Transformer from scratch on pure FineWeb-Edu at 4K context with `d_head=128` and a held-out RoPE base of 1M, using identical initialization, data order, optimizer, and token budget for Paper-Geo and EVQ-Cosh, changing only the fixed inverse-frequency schedule; then evaluate raw and matched-YaRN behavior on natural-text perplexity and RULER from 4K to 32K, complemented by controlled FMRoPE and schedule-shape ablations at smaller scale, with Std-Geo confined to a small reference ablation.

---

# 28. 参考资料

- NVIDIA RULER 官方仓库：`https://github.com/NVIDIA/RULER`
- RULER 论文：Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, arXiv:2404.06654
- FineWeb-Edu：`https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu`
- Oka et al., ICLR 2026：*How Base Frequency Shapes RoPE: An Analytical Study of Frequency-Band Formation*
- NVIDIA RTX PRO 6000 Blackwell 官方规格：96GB GDDR7，1792 GB/s memory bandwidth
- EVQ-Cosh submission 11628：现有正文、Appendix D、提交代码与实验配置是实现的最终事实来源
