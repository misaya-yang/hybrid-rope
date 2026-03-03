# Phase 13: LongBench NLL Eval + 750M LoRA — 冲击 Spotlight 的两把刀

> **核心目标**: 用两种互补方式证明 EVQ 在 real downstream tasks 上全面领先，补上论文最后一块短板
> **硬件**: 5090 32GB
> **预计 GPU 时间**: 13A NLL eval ~30min, 13B LoRA ~3-4h training + 1h eval
> **前置**: 750M Geo + Hybrid checkpoints 已完成 (Phase 9F)

---

## 核心论点

目前的证据链:
- ✅ PPL 跨规模一致 (50M-750M)
- ✅ Passkey +40pp (supervised)
- ✅ Retrieval divergence +20pp (750M)
- ✅ EVQ+YaRN 100%@8K (3 seed, zero variance)
- ✅ RULER 单探针领先
- ❌ **Real downstream NLU tasks (LongBench 类)**

Reviewer 攻击面: "PPL 和 passkey 都是代理指标, 模型在真实任务上表现如何？"

本 phase 用两种方式回答:
- **13A**: 不需要生成能力的 NLL-based 评测 (base model 直接跑)
- **13B**: LoRA 指令微调后的生成式评测 (750M from-scratch, LoRA 干净无 confounding)

---

## Phase 13A: LongBench NLL Evaluation (~30 min)

### 方法

把 LongBench 的 (context + question + gold_answer) 拼接, 只在 answer tokens 上计算 NLL。
NLL 越低 = 模型给正确答案分配的概率越高 = 长文本理解越好。

**优势**: 不需要模型生成, 350M/750M base model 直接评测。

### 脚本

脚本已写好: `scripts/m4_evq_sweep/eval_longbench_nll.py`

### 执行命令

```bash
# === 找到 750M checkpoints ===
# Geo checkpoint 和 Hybrid checkpoint 的路径, 从 Phase 9F 的输出目录中找

# === 13A-1: 750M Geo baseline, 3 个上下文长度 ===
for CTX in 2048 4096 8192; do
    python scripts/m4_evq_sweep/eval_longbench_nll.py \
        --model_path <GEO_750M_CKPT_PATH> \
        --tasks qa4 \
        --max_context_len $CTX \
        --max_samples 100 \
        --method_name geo_750m \
        --output_dir results/phase13a_longbench_nll/ \
        --dtype bfloat16
done

# === 13A-2: 750M Hybrid EVQ, 同样 3 个长度 ===
for CTX in 2048 4096 8192; do
    python scripts/m4_evq_sweep/eval_longbench_nll.py \
        --model_path <HYBRID_750M_CKPT_PATH> \
        --tasks qa4 \
        --max_context_len $CTX \
        --max_samples 100 \
        --method_name hybrid_750m \
        --output_dir results/phase13a_longbench_nll/ \
        --dtype bfloat16
done

# === 13A-3 (可选): 350M 3-seed 对比, 统计显著性更强 ===
# 如果 750M 有方向, 再用 350M 3 个 seed 确认
```

### 数据来源

HuggingFace `THUDM/LongBench`。如果网络不通, 提前下载:
```bash
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', 'qasper', split='test')"
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', 'hotpotqa', split='test')"
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', '2wikimqa', split='test')"
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', 'narrativeqa', split='test')"
```

### 预期结果

| 指标 | 预期 | 依据 |
|------|------|------|
| NLL@2K | Geo ≈ Hybrid | 训练内, 两者 PPL 接近 |
| NLL@4K | Hybrid < Geo | 外推区, EVQ PPL 优势 -3% 到 -10% |
| NLL@8K | Hybrid << Geo | 外推放大, PPL 优势 -10%+ |

**关键判断**: 如果 NLL 差距随上下文长度单调扩大 (与 PPL 趋势一致), 就是 spotlight 级别的 downstream evidence。

### 注意事项

1. `max_context_len=8192` 在 750M 上可能 OOM, 试试 `--dtype bfloat16`, 不行就降到 6144
2. `--truncation middle` 保留 context 头尾, 去掉中间 (LongBench 标准做法)
3. 每个 task 100 samples 足够, LongBench 本身每 task 200-500 条
4. 如果某个 task 下载失败, 跳过即可, qa4 里任意 2-3 个 task 有结果就能写论文

---

## Phase 13B: 750M LoRA Instruction Fine-tuning (~4h)

### 为什么 750M LoRA 是干净的

CORE_THEORY.md §11.5 标注 8B LoRA 有 confounding: "模型同时适应新频率 + 学习下游任务"。

**但 750M 不存在这个问题**: 两个模型 (Geo-750M, Hybrid-750M) 都是 from-scratch 训练的, 各自的频率已经是 native 的。LoRA 只在学下游任务, 不需要适应新频率。

对照实验: Geo-750M + LoRA vs Hybrid-750M + LoRA, 同样的 LoRA 配置、同样的微调数据、同样的 steps。唯一差异 = 底座频率分配。完美因果推断。

### 微调配置

```python
# LoRA config
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
}

# Training config
training_config = {
    "dataset": "Yukang/LongAlpaca-12k",  # 长文本指令数据, 12K samples
    "max_seq_len": 4096,       # 2x train length, 测试外推
    "batch_size": 1,
    "gradient_accumulation": 8,
    "lr": 2e-4,
    "epochs": 2,
    "warmup_ratio": 0.03,
    "bf16": True,
}
```

### 执行步骤

```bash
# === Step 1: LoRA 微调 Geo-750M ===
python scripts/train_750m_lora.py \
    --base_model <GEO_750M_CKPT_PATH> \
    --output_dir results/phase13b_lora/geo_750m_lora/ \
    --dataset Yukang/LongAlpaca-12k \
    --max_seq_len 4096 \
    --lora_r 16 --lora_alpha 32 \
    --epochs 2 --lr 2e-4 --bf16

# === Step 2: LoRA 微调 Hybrid-750M ===
python scripts/train_750m_lora.py \
    --base_model <HYBRID_750M_CKPT_PATH> \
    --output_dir results/phase13b_lora/hybrid_750m_lora/ \
    --dataset Yukang/LongAlpaca-12k \
    --max_seq_len 4096 \
    --lora_r 16 --lora_alpha 32 \
    --epochs 2 --lr 2e-4 --bf16

# === Step 3: LongBench 生成式评测 ===
for METHOD in geo_750m_lora hybrid_750m_lora; do
    python scripts/eval_longbench.py \
        --model_path <BASE_MODEL> \
        --adapter_path results/phase13b_lora/${METHOD}/ \
        --task_set lb6 \
        --max_ctx 8192 \
        --output_dir results/phase13b_lora/eval_${METHOD}/
done
```

### ⚠️ 需要新写的脚本

`scripts/train_750m_lora.py` 尚不存在。核心逻辑:
1. 加载 750M from-scratch checkpoint (AutoModelForCausalLM)
2. 应用 LoRA (peft)
3. 加载 LongAlpaca-12k, tokenize 成 max_seq_len=4096
4. 标准 Trainer 训练

这个脚本很简单, 参考 `scripts/train_cross_model_lora_fast_tuned.py` 改一下模型加载即可。

### 预期结果

| 评测 | Geo-750M + LoRA | Hybrid-750M + LoRA | 预期差距 |
|------|----------------|-------------------|---------|
| LongBench QA (F1) @4K | X% | X+5-10% | Hybrid 更好 |
| LongBench QA (F1) @8K | Y% | Y+10-15% | 差距随长度扩大 |
| Passkey @8K (生成) | ~60% | ~80% | 与 Phase 9F 一致 |

---

## 执行优先级

1. **先跑 13A** (30 min), 立刻出结果, 判断方向
2. 13A 有方向 → 同时启动 13B LoRA 训练 (后台跑)
3. 13A 结果写入 CORE_THEORY.md

## 成功标准

- **Green**: NLL@4K+ Hybrid 显著低于 Geo (>2% 差距) → 直接写入论文 Table
- **Yellow**: NLL 差距 <2% → 结果存在但不够强, 论文放 appendix
- **Red**: Geo NLL 更低 → 检查是否 checkpoint 加载错误或 inv_freq 未 patch

## 论文价值

13A + 13B 一旦成功, reviewer 攻击面清零:
- "只有 PPL" → ❌ LongBench NLL 也赢
- "PPL 不等于下游" → ❌ real downstream QA 也赢
- "没有指令微调" → ❌ LoRA 后生成式评测也赢
- "LoRA 不干净" → ❌ from-scratch base, 频率是 native 的

这是从 poster → spotlight 的最后一公里。
