# Llama-3-8B A1/A2 Checkpoint-600 Gate Eval 结果分析

> **日期**: 2026-02-28
> **实验批次**: `llama8k_theory_v1`
> **服务器**: RTX PRO 6000 Blackwell 96GB (AutoDL bjb1)
> **评估脚本**: `scripts/eval_longbench.py` (gate mode: qasper + musique)

---

## 1. 实验配置

| 参数 | A1 (Baseline) | A2 (EVQ) |
|------|--------------|----------|
| rope_schedule | `evq_cosh` (tau=0.0 = geometric) | `evq_cosh` (tau=1.5) |
| 检查点 | checkpoint-600 | checkpoint-600 |
| base model | Meta-Llama-3-8B-Instruct | 同左 |
| LoRA rank / alpha | 32 / 64 | 同左 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj | 同左 |
| max_seq_len | 8192 | 同左 |
| batch_size | 4 | 同左 |
| lr / scheduler | 2e-5 / cosine | 同左 |
| loss | response-only | 同左 |
| 数据集 data_hash | 相同（已验证 SHA256） | 同左 |
| 训练目录 | `A1_geometric_tau0p00_r32_s800_seed42__20260228_115150` | `A2_evq_cosh_tau1p50_r32_s800_seed42__20260228_154206` |

**注意**: A1 的 `tau=0.0` 经过 `build_evq_cosh_inv_freq` 中 `tau <= 1e-8` 分支，输出 `φ(u)=u`，
等价于标准 geometric 频率分配，即 inv_freq 与 Llama-3 原始值完全一致。

---

## 2. Gate 评估结果

### 2.1 原始分数

| 模型配置 | qasper | musique |
|---------|--------|---------|
| base_unfinetuned (A1) | 42.49% | 19.11% |
| base_unfinetuned (A2) | 42.49% | 19.11% |
| A1 hybrid_lora (geometric) | 40.28% | 9.02% |
| A2 hybrid_lora (EVQ τ=1.5) | **3.10%** | **0.99%** |

### 2.2 对比

| 任务 | A1 LoRA | A2 LoRA | Delta |
|------|---------|---------|-------|
| qasper | 40.28% | 3.10% | **-37.18%** |
| musique | 9.02% | 0.99% | **-8.03%** |

### 2.3 Gate 判定

- A2 **未通过** gate 阈值（`qasper_lora >= qasper_base`, `musique_lora >= musique_base - 1.0`）
- A1 也未通过 gate（musique 从 19.11% 降到 9.02%），说明当前训练数据/步数本身也有问题
- A2 属于**灾难性崩溃**，与 A1 的退化性质完全不同

---

## 3. 训练 Loss 对比

### 3.1 Eval Loss

| 检查点 | A1 eval_loss | A2 eval_loss | 倍数 |
|--------|-------------|-------------|------|
| step 400 | — | 0.616 | — |
| step 600 | ~0.188 | 0.578 | **3.1x** |

### 3.2 Train Loss 典型范围

| 指标 | A1 | A2 |
|------|----|----|
| step 610-800 典型 loss | 0.11 ~ 0.48 | 0.19 ~ 1.69 |
| 中位值约 | ~0.25 | ~0.65 |
| 最终状态 | 完成 800 步 | 在 step 770 GPU 进程崩溃 |

### 3.3 关键观察

A2 的 loss 确实 < 1（大部分时间），这说明模型在 teacher forcing 下**确实能部分学习**训练数据。
但 eval 完全崩溃，这看似矛盾。下面解释为什么不矛盾。

---

## 4. 根因分析：为什么 Loss < 1 但 Eval 崩溃？

### 4.1 Teacher Forcing vs 自回归生成的本质区别

**训练 / eval_loss（Teacher Forcing）**:
- 模型每一步都看到**正确的上文 token**
- 即使 EVQ 位置编码扭曲了注意力模式，正确的 key-value 信息仍然可用
- 度量的是：「给定完美上文 + EVQ 位置，能否预测下一个 response token？」
- 这是一种"开卷考试"

**Gate 评估（自回归生成）**:
- 模型逐 token 生成答案
- EVQ 位置编码 → 注意力关注错误位置 → 生成错误 token
- 错误 token 成为下一步的上文 → 更大的偏差 → **指数级误差级联**
- 这是一种"闭卷考试"

因此 eval_loss = 0.578（teacher forcing 下看似还行）可以同时存在于 qasper = 3.1%（自回归生成完全崩溃）。
0.578 的 eval_loss 代表 perplexity = e^0.578 ≈ 1.78，每个 token 位置的预测都有显著不确定性，
在自回归链中这些不确定性成倍累积。

### 4.2 τ=1.5 对频率的扰动量

EVQ-Cosh 公式：`φ(u) = 1 - (1/τ) · arcsinh((1-u) · sinh(τ))`

对 Llama-3-8B（head_dim=128, n=64, base=500000）：

| 频率维度 u | φ_geometric | φ_EVQ(τ=1.5) | inv_freq 比值 (EVQ/Geo) |
|-----------|------------|-------------|----------------------|
| 0.0 (最高频) | 0.000 | 0.000 | 1.0x |
| 0.25 | 0.250 | 0.177 | ~2.5x |
| 0.50 (中频) | 0.500 | 0.382 | **~4.9x** |
| 0.75 (低频) | 0.750 | 0.660 | **~4.2x** |
| 1.0 (最低频) | 1.000 | 1.000 | 1.0x |

**中频维度的旋转速度变为原来的约 5 倍**。
预训练模型认为"相距 10 个位置"的 token，在 EVQ 编码下等效于"相距 ~50 个位置"。

### 4.3 Pretrained Weights 与频率的深度绑定

Llama-3-8B 的 80 亿参数在数万亿 token 的预训练中学会了：
- 每个注意力头在每个频率维度上如何编码相对距离
- Q/K 投影矩阵内化了 geometric 频率下的位置-语义映射

当 EVQ τ=1.5 将中频维度的旋转速度变为 5 倍：
- 模型的 Q·K^T 点积产生的注意力权重分布完全偏移
- 预训练学到的"关注哪些相对位置"的模式被打乱
- Rank-32 LoRA（约占总参数 0.1%）无法重新学习这种全局映射

### 4.4 从头训练 vs 微调的本质差异

50M/125M 从头训练中 τ=1.5 表现出色（PPL@16K -10.9% / -18.9%），因为：
- 模型**从零**学习 EVQ 频率下的位置-注意力映射
- 所有参数都在适应 EVQ，而不只是 0.1% 的 LoRA 参数
- 没有 pretrained weights 与 EVQ 频率的冲突

8B 微调中 τ=1.5 崩溃，因为：
- 80 亿参数已深度绑定 geometric 频率
- LoRA 容量严重不足
- 频率扰动太大（中频 5x），超出 LoRA 的适应范围

---

## 5. 排除的代码 Bug 假设

逐一验证了以下可能的代码问题：

### 5.1 `original_inv_freq` 未同步 — **非根因**

训练脚本 (`inject_inv_freq_copy`) 同时 patch `inv_freq` 和 `original_inv_freq`。
eval 脚本 (`patch_hybrid_rope`) 只 patch `inv_freq`。

但 transformers 5.1.0 中 Llama-3 使用 `rope_type="default"`。
`@dynamic_rope_update` 装饰器仅对 `"dynamic"` 和 `"longrope"` 触发 `original_inv_freq` 重计算。
Default 类型直接跳过，forward 只用 `self.inv_freq`。因此不是问题。

```python
# transformers/modeling_rope_utils.py:82-87
def wrapper(self, x, position_ids):
    if "dynamic" in self.rope_type:     # False for "default"
        dynamic_frequency_update(...)
    elif self.rope_type == "longrope":   # False for "default"
        longrope_frequency_update(...)
    return rope_forward(self, x, position_ids)  # 直接用 self.inv_freq
```

### 5.2 cos/sin 缓存残留 — **非根因**

transformers 5.1.0 的 `LlamaRotaryEmbedding.forward` 每次从 `inv_freq` 实时计算 cos/sin（line 96-106），
不使用旧式 `_cos_cached`/`_sin_cached`。

### 5.3 模型状态泄漏 — **非根因**

`base_unfinetuned` 和 `hybrid_lora` 各自调用 `load_model_and_tokenizer` 全新加载（line 2212），
完全隔离。无状态泄漏。

### 5.4 PeftModel 覆盖 inv_freq — **非根因**

`PeftModel.from_pretrained` 只封装 LoRA target modules (q/k/v/o_proj)，
不触碰 `rotary_emb` 模块。inv_freq 在 LoRA 加载后仍保持 patch 后的值。

### 5.5 inv_freq 注入方式差异 — **非根因但应统一**

| 属性 | 训练脚本 | eval 脚本 |
|------|---------|----------|
| 注入方式 | `.copy_()` (in-place) | 属性替换 `module.inv_freq = new` |
| `original_inv_freq` | 同步 patch | 未 patch |
| 缓存清理 | 清理 4 个 attr | 清理 4 个 attr（列表相同） |

对 default rope_type，两种方式功能等价。但建议统一为 `.copy_()` + 同步 `original_inv_freq`，
以防未来升级 transformers 或使用 dynamic scaling 时出现隐患。

---

## 6. 结论与行动建议

### 6.1 核心结论

τ=1.5 对 Llama-3-8B 微调**完全不可行**。原因是预训练权重与 EVQ 频率不兼容，
LoRA 容量不足以桥接 geometric → EVQ 的映射差异。
训练 loss < 1 是 teacher forcing 的假象，不代表模型真正学会了 EVQ 编码下的位置理解。

### 6.2 下一步实验方案

按优先级排列：

#### 方案 A：降低 τ（推荐首选）

| 参数 | 值 | 理由 |
|------|-----|------|
| tau 候选 | 0.3, 0.5, 0.8 | 中频扰动从 5x 降到 ~1.3x / ~1.7x / ~2.5x |
| 训练步数 | 800 | 保持与 A1 一致 |
| 其余参数 | 全部锁死 | 单变量对照 |
| 评估 | gate (qasper + musique) | 快速判断 |

预期：τ=0.3~0.5 的中频扰动在 1.3x~1.7x 范围，pretrained weights 可能可以通过 LoRA 适应。

#### 方案 B：增大 LoRA 容量

| 参数 | 值 | 理由 |
|------|-----|------|
| LoRA rank | 128 或 256 | 4x-8x 更多参数适应位置编码 |
| tau | 1.5（保持） | 验证容量是否是瓶颈 |
| 显存 | 需评估 96GB 是否足够 | rank=256 可能需要 gradient checkpointing |

#### 方案 C：渐进式频率注入（Warmup Schedule）

| 阶段 | 步数 | tau |
|------|------|-----|
| Phase 1 | 0-200 | 0.0 (geometric) |
| Phase 2 | 200-600 | 线性插值 0.0 → 1.5 |
| Phase 3 | 600-800 | 1.5 (目标) |

需要修改训练脚本，在每 N 步重新注入插值后的 inv_freq。

#### 方案 D：τ 网格搜索 + Gate 快筛

```
tau ∈ {0.3, 0.5, 0.8, 1.0, 1.5}
每个 tau 训练 400 步 → gate eval → 筛选 top-2 → 继续训练到 800 步 → full eval
```

### 6.3 代码改进建议

1. **统一 inv_freq 注入方式**: eval 脚本的 `patch_hybrid_rope` 应改为 `.copy_()` + 同步 `original_inv_freq`，
   与训练脚本保持一致
2. **添加 inv_freq SHA256 校验**: 在 eval 日志中记录 patched inv_freq 的 hash，
   与训练时保存的 hash 交叉验证
3. **添加 τ 扰动预检**: 训练前自动计算中频维度的扰动倍数，超过阈值（如 3x）时打印警告

---

## 7. 文件索引

| 文件 | 说明 |
|------|------|
| `artifacts/llama8k_theory_v1/train/A1_geometric_tau0p00_r32_s800_seed42__20260228_115150/` | A1 训练产物（完成 800 步） |
| `artifacts/llama8k_theory_v1/train/A2_evq_cosh_tau1p50_r32_s800_seed42__20260228_154206/` | A2 训练产物（崩溃于 step 770） |
| `artifacts/llama8k_theory_v1/smoke_eval_ckpt600/a1_geo_ckpt600_gate.json` | A1 gate 评估结果 |
| `artifacts/llama8k_theory_v1/smoke_eval_ckpt600/a2_evq_ckpt600_gate.json` | A2 gate 评估结果 |
| `scripts/eval_longbench.py` | 评估脚本（line 433-474: patch_hybrid_rope, line 538-700: load_model_and_tokenizer） |
| `scripts/isolated/longinst/new_lora_longinst_train_v1.py` | 训练脚本（line 1257-1326: inject_inv_freq_copy） |
