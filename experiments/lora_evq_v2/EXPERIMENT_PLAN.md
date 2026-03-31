# LoRA EVQ-Cosh v2 实验计划

> **日期**: 2026-03-31
> **目标**: 证明 EVQ-cosh LoRA (r=64, τ=1.414) 在长上下文任务上远胜 base Instruct 模型 (≥15%)
> **预算**: ≤ 10 小时 GPU

---

## 1. 问题诊断：为什么 v1 失败

| 问题 | v1 (Qwen) | v1 (LLaMA fair suite) | v2 修正 |
|------|-----------|----------------------|---------|
| RoPE 方法 | hybrid_a0.2_t100k (非 EVQ-cosh) | hybrid inv_freq (非 EVQ-cosh) | **纯 EVQ-cosh** |
| τ 值 | 隐含在 anchor 中 | 隐含 | **τ=1.414 (=128/√8192)** |
| LoRA rank | 未知 | r=16 (r/K=0.25 ❌) | **r=64 (r/K=1.0 ✅)** |
| 训练数据 | TinyStories (短文本) | WikiText (短文本) | **LongAlign-10k (长上下文指令)** |
| 评测 | PPL only | PPL at [16K, 32K] | **LongBench 6-task + PPL** |

### 理论依据 (TAU_REGIME_THEORY)

- **相变条件**: r_c = K = d_head/2 = 64
- **v1 失败**: r=16, r/K=0.25 → 48 个冻结通道无法适应 → PPL 77.1
- **v2 预测**: r=64, r/K=1.0 → 所有通道可适应 → τ*=1.414 应正常工作

---

## 2. 实验参数

### 2.1 模型
- **LLaMA-3-8B-Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`)
- d_head=128, K=64, rope_theta=500K, max_pos=8192
- 选择理由: 8K 原生上下文 → 16K/32K 是清晰的外推区间，有 v1 fair suite 基线可比较

### 2.2 频率分配
- **方法**: EVQ-cosh (纯理论公式)
- **τ**: 1.414 (= d_head/√L = 128/√8192)
- **公式**: φ_k(τ) = 1 - (1/τ)·arcsinh((1-u_k)·sinh(τ)), u_k = (2k-1)/(2K)
- **注意**: 使用中点量化 u_k = (2k-1)/(2K) 而非边界量化 u_k = k/K

### 2.3 LoRA 配置
- **rank**: 64 (匹配 r_c = K = d_head/2)
- **alpha**: 128 (alpha/r = 2)
- **dropout**: 0.05
- **target_modules**: q_proj, k_proj, v_proj, o_proj
- **量化**: 4-bit (nf4, double_quant) - 节省显存

### 2.4 训练数据
- **首选**: `THUDM/LongAlign-10k` (HuggingFace)
  - 10K 长上下文指令数据，最长 64K tokens
  - 包含多种长文档任务类型
- **备选**: `ybelkada/long-form-qa` 或 SlimPajama long docs
- **seq_len**: 8192 tokens

### 2.5 训练超参
- **max_steps**: 600 (匹配 fair suite)
- **batch_size**: 4 (per device)
- **gradient_accumulation**: 2
- **lr**: 2e-4
- **warmup**: 30 steps
- **scheduler**: cosine
- **optimizer**: paged_adamw_8bit
- **gradient_checkpointing**: True

### 2.6 评测

| 评测 | 描述 | 成功标准 |
|------|------|----------|
| PPL@8K/16K/32K | WikiText-2 perplexity | PPL@32K 不爆炸 (< 15) |
| LongBench-6 | 6-task 长上下文基准 | EVQ > Instruct ≥ 15% |
| Passkey@32K | 大海捞针检索 | ≥ 95% |

---

## 3. 预期时间

| 阶段 | 时间估计 |
|------|----------|
| 数据下载 + 预处理 | ~15 min |
| 训练 (600 steps, 4-bit, 1×A100) | ~2-3 hours |
| PPL 评测 (3 lengths) | ~30 min |
| LongBench-6 评测 | ~2-3 hours |
| Passkey 评测 | ~15 min |
| **总计** | **~5-7 hours** |

---

## 4. 文件清单

| 文件 | 用途 |
|------|------|
| `train_evq_lora.py` | 完整训练脚本 (EVQ-cosh 频率计算 + LoRA 训练) |
| `eval_evq_lora.py` | 评测脚本 (PPL + LongBench + Passkey) |
| `dryrun_validate.py` | 无卡验证脚本 (检查频率、config、数据) |
| `EXPERIMENT_PLAN.md` | 本文档 |

---

## 5. 可测试预测

1. **PPL@32K < 15**: EVQ r=64 不会爆炸 (对比 v1 r=16 的 77.1)
2. **LongBench-6 均分 > Instruct × 1.15**: 至少 15% 提升
3. **Passkey@32K ≥ 95%**: 频率分配正确 → 位置检索正常
4. **PPL@8K ≤ Instruct PPL@8K × 1.2**: 短上下文不严重退化
