# 8B LoRA 实验分析

> 最后更新：2026-02-22（标记旧问题状态、新增公平实验设计）

## 1. 旧实验问题诊断（全部已修复 ✅）

### Issue 1：不公平比较 → ✅ 已修复

| 问题 | 旧做法 | 新做法 |
|------|--------|--------|
| RoPE 实现 | YaRN/PI 用 `rope_scaling`，Hybrid 用 monkey patch | 统一 `inv_freq.copy_()` buffer 覆写 |
| Config 污染 | YaRN 修改 `model.config.rope_scaling` | 强制 `rope_scaling=None` |
| Forward patch | Hybrid 替换 forward 函数 | 禁止任何 forward monkey patch |

### Issue 2：超参不一致 → ✅ 已修复

所有方法现在锁定完全相同的训练超参：

| 参数 | 值 |
|------|-----|
| batch_size × grad_accum | 2 × 2 = 4 |
| learning_rate | 2e-4 |
| max_steps | 600 |
| attention | sdpa（强制） |
| LoRA rank/alpha | 64/128 |
| LoRA targets | q,k,v,o_proj |
| bf16 | True |
| gradient_checkpointing | True (use_reentrant=False) |

### Issue 3：Hybrid 超参未调优 → ✅ 重新设计

旧 Hybrid 使用 `compute_hybrid_inv_freq` 的固定参数（split_ratio=0.5, alpha=0.2, p=3.9）。

新方法 `anchored_hybrid` 采用物理更清晰的设计：
- **rigid_j0=12**：前 12 对高频与 baseline 精确相同（bitwise equal）
- **tail_base = base × scale²**：低频段使用更大 base（严格 > 原 base）
- **cosine ramp + alpha blend**：平滑过渡，无硬拼接

### Issue 4：Device mismatch → ✅ 已修复

验证探针使用 `next(model.parameters()).device` 而非硬编码 `cuda:0`。

### Issue 5：BOS token 丢失 → ✅ 已修复

截断逻辑保留 `[BOS] + tail` 而非纯尾截断。

### Issue 6：inv_freq 注入可能失效 → ✅ 已加防御

运行时探针检测 inv_freq 是否被 forward 真正消费。

---

## 2. 新公平实验设计

### 脚本

- 训练：`2026-02-22/scripts/run_llama8b_fair_suite.py`（830 行，含所有安全检查）
- 流水线：`2026-02-22/scripts/run_overnight_8h.py`（606 行，4-gate 自动化）

### 4 个方法

| 方法 | inv_freq 计算方式 | 特点 |
|------|------------------|------|
| baseline | 原始几何频率 | 对照组 |
| PI | `base_inv / scale` | 位置插值 |
| YaRN | progresssive ramp + temperature | 渐进插值 |
| anchored_hybrid | rigid core + cosine-ramp blend | **我们的方法** |

### 评测

- NIAH 热力图：[4K, 8K, 16K, 32K] × 11 depth × 3 trials
- 训练 loss 曲线
- 基于 baseline（无 LoRA）的参考线

### 安全检查链

1. `model.config.rope_scaling == None` 断言
2. inv_freq shape/dtype 匹配检查
3. rigid core bitwise equal 验证
4. 运行时 logit diff 探针
5. BOS 保留验证

---

## 3. 旧实验结果存档（仅供参考）

| 方法 | train_loss | PPL@16K | PPL@32K | 状态 |
|------|-----------|---------|---------|------|
| YaRN (旧) | 1.7248 | 6.0566 | 6.2702 | ⚠️ 旧协议 |
| PI (旧) | 1.9493 | 6.1369 | 6.3100 | ⚠️ 旧协议 |
| Hybrid (旧) | 2.0565 | 11.8753 | 77.1381 | ❌ 不公平比较 |

**这些数据不应在论文中引用**，等待新公平实验结果替代。

---

*分析完成：2026-02-22 01:50*
