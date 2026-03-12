# AI Handoff: EVQ 实验设计踩坑指南

> 更新: 2026-03-12
> 目的: 给接手的 AI agent (Claude Code / GPT / 任何 LLM) 提供实验设计的硬约束，避免重复踩坑。
> **每次新 session 必读此文件。**

---

## 0. 核心定义速查

```
τ*(L, d_head) = d_head / √L          # 理论最优温度
φ_k(τ) = 1 - (1/τ) arcsinh((1-u_k) sinh(τ))   # EVQ-cosh warp
ω_k = base^(-φ_k(τ))                  # inverse frequency
u_k = k/K  或  (k+0.5)/K              # quantile (两种都在用，注意区分)
K = d_head / 2                        # 频率通道数
```

---

## 1. τ 参数：必须跟着训练长度变

### 规则
**每次训练长度 L 改变，τ 必须重新计算。** τ 不是一个固定的超参数。

### 正确做法
```python
TAU = d_head / math.sqrt(SEQ_LEN)
```

### 历史案例

| 阶段 | L_train | d_head | 正确 τ | 实际用的 τ | 后果 |
|------|---------|--------|--------|-----------|------|
| 454M Stage 1 | 512 | 64 | 2.83 | 2.83 ✓ | 正常 |
| 454M Stage 2 | 1024 | 64 | 2.00 | 2.00 ✓ | 正常 |
| 454M Stage 3 | 2048 | 64 | 1.414 | 1.414 ✓ | 正常 |
| **750M Phase 15 finetune@8192** | **8192** | **128** | **1.414** | **1.5 (经验值)** | ⚠️ 偏差不大但不精确 |
| **750M finetune@8192 (old)** | **8192** | **128** | **1.414** | **0.707 (算错)** | ❌ τ 太小，EVQ≈Geo |

### 踩坑总结
- τ 太小 → EVQ 退化为 Geo（没有重分配），白跑
- τ 太大 → 过度重分配，短程 PPL 恶化
- **切记**: d_head 不是 hidden_size！750M 的 d_head=128（不是 1024）
- **τ 必须用 L_train 计算，不是 L_eval**

---

## 2. YaRN 实现：Progressive vs NTK-aware

### 这是整个项目被坑最多次的地方。

### ❌ 错误：NTK-aware uniform scaling（已确认是 bug）
```python
# phase21b_quality_eval.py 里的错误实现
def apply_yarn_scaling(inv_freq, scale):
    dim = len(inv_freq) * 2
    factor = scale ** (dim / (dim - 2))  # 所有通道同一个 factor
    return inv_freq / factor
```

**为什么错**: 这是 NTK-aware 的公式，对所有频率通道施加相同的缩放因子。它会**摧毁 EVQ 精心设计的频率分配结构**。

### ✅ 正确：Progressive YaRN（per-channel smoothstep ramp）
```python
# phase21b_scrolls_finetune.py 里的正确实现
def apply_yarn_scaling(inv_freq, scale, original_max_pos=2048):
    if scale <= 1.0:
        return inv_freq.clone()
    K = len(inv_freq)
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)      # 前 20% 通道不动（高频保护）
    end = int(0.90 * K)        # 后 10% 通道全量缩放（低频）
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + 0.07 * math.log2(scale)
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return inv_freq / yarn_scale
```

### 区别一览

| 特征 | NTK-aware (❌) | Progressive YaRN (✅) |
|------|---------------|----------------------|
| 缩放方式 | 所有通道 × 同一个 factor | 每个通道不同的 factor |
| 高频通道 | 被缩放（破坏短程精度） | 不动（保护短程） |
| 低频通道 | 和高频一样缩放 | 大幅缩放（释放长程能力） |
| temperature | 无 | `1.0 + 0.07 * log2(scale)` |
| EVQ 兼容性 | ❌ 摧毁 EVQ 结构 | ✅ 保持 EVQ 结构 |
| 代码行数 | 3 行 | 12 行 |

### 检查清单
- [ ] eval 脚本里的 YaRN 和 train 脚本里的 YaRN **必须是同一个实现**
- [ ] 搜索 `factor = scale **` 这类代码，如果看到 uniform 缩放就是 bug
- [ ] YaRN scale 参数 = eval_length / train_length（不是别的比值）

---

## 3. 评估协议：常见陷阱

### 3.1 Distractor padding vs Standard eval

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| Standard (无 padding) | 直接在原文上 eval | **优先使用**，FIRE/SCROLLS 标准做法 |
| Distractor padding | 用无关文本填充到目标长度 | 测试长距离检索能力 |

**踩坑**: Phase 21b 早期同时搞了 standard eval 和 distractor eval，结果混在一起无法对比。**每组实验必须明确标注用的是哪种模式。**

### 3.2 Accuracy vs NLL

| 指标 | 问题 |
|------|------|
| 多选题 accuracy | 200 道题上 variance 巨大（Geo 20.5% vs random 25% 在统计上完全正常，仅 ~1.5σ） |
| NLL (log-likelihood) | 连续值，方差小得多，但不直观 |

**建议**: 优先报 NLL-based metric，accuracy 作为辅助。如果必须用 accuracy，注意样本量足够（>500）或者跑多 seed。

### 3.3 训练和评估的目标必须匹配

Phase 21b 最初的 finetune 是 answer-only loss masking（只算答案 token 的 loss），但 eval 用的是 accuracy（看模型选哪个选项）。这两者不矛盾但需要注意：finetune loss 低不等于 accuracy 高。

---

## 4. Checkpoint 选择：Retrofit vs Progressive

### ❌ Retrofit（在已训练好的 Geo checkpoint 上改 inv_freq）
- 模型已经学会了基于 geometric 频率的 attention pattern
- 硬改 inv_freq 等于破坏已有的 attention 结构
- EVQ 带着先天劣势起步

### ✅ Progressive（从头训练或者从当前阶段的 EVQ checkpoint 继续）
- 模型在每个阶段都是基于当前 τ 的频率分配来学习
- 454M 的 3-stage progressive (512→1024→2048) 是目前最干净的实验

### 踩坑
- 750M Phase 15 早期就是 retrofit（在 Geo pretrain 上叠 EVQ finetune），效果打折
- Phase 17c 证明了 progressive 才是正路

---

## 5. 超参数：不要照抄别的 paper

### FIRE 的教训
- FIRE 用 25K steps × bs128 × L=8192
- 我们最初照抄 25K steps 但忘了 FIRE 的 PE 机制完全不同
- FIRE 是学习式 PE，和我们的 closed-form EVQ 训练特征完全不一样
- **正确做法**: 学习 FIRE 的实验设计思路（4× finetune length, standard eval），不要复制具体数值

### 模型不是"太小学不会"
- FIRE 用 350M 模型在 SCROLLS 上学得很好
- 454M 在 QuALITY 上也学会了
- 如果模型"学不会"，先检查 τ 和 YaRN 是不是写错了，不要急着怪模型

---

## 6. 数值精度

### inv_freq 计算
- 必须用 `float64` 计算 φ_k 和 inv_freq，最后 `.float()` 转回 `float32`
- 低频通道的 inv_freq 可以到 1e-6 量级，float32 中间计算会丢精度

### arcsinh 边界
- 当 τ 很大时，`sinh(τ)` 可能溢出
- 当 τ→0 时，用 Taylor 展开避免 0/0：`φ_k ≈ u_k - (τ²/6) · A_k · (1 - A_k²)`

---

## 7. 正确的参考实现位置

| 功能 | 正确文件 | 行号 |
|------|---------|------|
| EVQ-cosh inv_freq | `scripts/core_text_phases/run_evq_sweep.py` | 141-157 |
| Progressive YaRN | `scripts/core_text_phases/phase21b_scrolls_finetune.py` | 147-169 |
| τ* 公式 | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` | 302-303 |
| Learnable EVQ (带 Taylor) | `scripts/lib/rope/learnable_evq.py` | 95-113 |
| Schedules 模块 | `scripts/lib/rope/schedules.py` | 全文件 |

### ⚠️ 已知 bug 文件（不要从这些文件复制代码）

| 文件 | 问题 |
|------|------|
| `phase21b_quality_eval.py` | YaRN 用了 NTK-aware uniform scaling |

---

## 8. 新实验的 Checklist

每次开始新实验前，过一遍这个清单：

- [ ] **τ 值**: `τ = d_head / √(L_train)` 算对了吗？L_train 是本阶段的训练长度
- [ ] **d_head 值**: 确认是 per-head dimension，不是 hidden_size
- [ ] **YaRN 实现**: 是 progressive（per-channel ramp）还是 NTK-aware（uniform）？必须用前者
- [ ] **YaRN scale**: scale = eval_length / train_length
- [ ] **Checkpoint 来源**: 是 progressive EVQ checkpoint 还是 retrofit？
- [ ] **eval 指标**: NLL / accuracy / retrieval 哪个为主？样本量够吗？
- [ ] **eval 模式**: standard 还是 distractor-padded？和 baseline 一致吗？
- [ ] **train/eval 脚本一致性**: finetune 和 eval 里的 YaRN / loss masking / tokenizer 是同一套吗？
- [ ] **数值精度**: inv_freq 在 float64 下计算了吗？

---

## 9. 当前优先级（2026-03-12）

1. **750M 正确 τ + YaRN eval** — 正在进行
2. **17c multi-seed** — 统计显著性
3. **LaTeX draft** — 把新 figure 集成进去
4. **Reproducibility cleanup** — 移除 hardcode path，统一入口脚本

---

## 10. 联系方式

有问题直接问 Misaya，不要自作主张改超参数。尤其是 τ 和 YaRN 的实现，已经踩了太多坑了。
