# Phase 22: QA-Mix Pretraining 实验方案

> **核心思路**: 把 QA 能力混入 pretraining（像 passkey-mix 一样），不做 task-specific finetune，直接 eval。PE 信号不会被 finetune adaptation 掩盖。

## 🔴 当前训练进度（2026-03-12）

```
参考: results/staged_training_experiment_report.md

seed 42: ✅ Stage1(512) → ✅ Stage2(1024) → ✅ Stage3(2048) — 全部完成
seed 43: ✅ Stage1(512) → 🔄 Stage2(1024) 进行中 → ❌ Stage3(2048)
seed 44: ✅ Stage1(512) → 🔄 Stage2(1024) 进行中 → ❌ Stage3(2048)
```

**QA-mix 插入点**: 在 seed 43/44 的 Stage 3 (L=2048) 加入 QA-mix。
Stage 1 和 Stage 2 不改（已跑/正在跑），Stage 3 是最后一个阶段，QA-mix 在这里加入。
如果 QuALITY 文章长度问题（median ~5K > 2048），见 §八 备选方案。

---

## ⚠️ Claude Code 必读：开工前 Checklist

**每次修改代码前，先读以下文件确认超参数和实现细节：**

```
必读文件清单:
1. team/AI_HANDOFF_PITFALLS.md              — 全部 10 节，尤其 §1 τ参数、§2 YaRN 实现、§8 新实验 Checklist
2. scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py (line 1-55, 91-178)
                                            — 成功的 pretraining 配置参考
3. scripts/core_text_phases/phase21b_scrolls_finetune.py (line 147-170)
                                            — 正确的 Progressive YaRN 实现
4. scripts/core_text_phases/phase21b_quality_eval_clean.py (line 1-91)
                                            — 正确的 eval 配置参考
5. team/plans/phase22_qa_mix_experiment.md   — 本文件
```

---

## 一、实验设计

### 1.1 与 Passkey-Mix 的对比

| 维度 | Passkey-Mix（已成功） | QA-Mix（本实验） |
|------|---------------------|-----------------|
| 混入内容 | `"The passkey is {N}. ... What is the passkey? {N}"` | `"{article} Q: {question} A: {answer}"` |
| 混入比例 | 5% of training chunks | **5%** of training chunks（与 passkey 一致） |
| Loss 区域 | 全 sequence（包含 filler + passkey） | **只算 answer tokens**（loss masking） |
| 目的 | 教模型在长文中精确检索数字 | 教模型在长文中找信息、回答问题 |
| 测的能力 | Retrieval precision | Reading comprehension（更接近"下游任务"） |

### 1.2 为什么不 finetune

Phase 21 root cause analysis 的核心发现（`team/plans/phase21_scrolls_downstream.md` §Root Cause Analysis）：

> "finetune 步数越多、任务越 specific，模型越能学到 task-specific attention shortcuts 来绕过 PE 差异"

证据链：
- Phase 21a **zero-shot NLL** → 完美 ±4.4% 反转 ✅
- Phase 21b **2000步 task-specific finetune** → accuracy 差异消失 ❌
- Passkey-mix **在 pretraining 学的** → EVQ 100% vs Geo 61% ✅

**规律**：能力在 pretraining 中获得时，PE 差异直接体现；能力在 finetune 中获得时，adaptation 掩盖 PE 差异。

---

## 二、超参数（全部从已验证配置继承）

### 2.1 模型配置

```
来源: phase17c_454m_1024_to_2048_continue.py line 164-178
```

| 参数 | 值 | 来源/验证 |
|------|-----|----------|
| 模型 | 454M (24L/16H/1024d, head_dim=64) | Phase 17c 已验证 |
| vocab_size | 50304 | 同上 |
| θ_base | 500,000 | 全项目统一 |
| init checkpoint | Phase 17b stage2 (L=1024) 的 EVQ/Geo pair | **不是 retrofit** ✅ |

### 2.2 训练配置

```
来源: phase17c (line 92-107, 164-178) — 直接继承 stage 3 配置
```

| 参数 | 值 | 说明 |
|------|-----|------|
| L_train | 2048 | 与 Phase 17c stage 3 完全一致 |
| τ_evq | **d_head / √L_train = 64 / √2048 ≈ 1.4142** | ⚠️ 必须用此公式，不允许硬编码 |
| τ_geo | 0 (geometric, 无 EVQ) | baseline |
| 训练 tokens | 500M | 与 Phase 17c stage 3 一致 |
| LR | 2e-4 (cosine → 10%) | 同上 |
| Warmup | 2% of total steps | 同上 |
| Optimizer | AdamW, β=(0.9,0.95), wd=0.1 | 同上 |
| effective_bs | 20 (micro=5 × accum=4) | 同上 |
| Passkey-mix | 5% | 保留！QA-mix 是额外的 5% |
| **QA-mix** | **5%** | 新增 |
| Seeds | 42, 43, 44 | 与 multi-seed 一致 |

### 2.3 τ 计算验证

```python
# ⚠️ Claude Code: 永远用公式计算，不要硬编码
import math
d_head = 64          # 454M: hidden_size=1024, num_heads=16, 1024/16=64
L_train = 2048       # 本阶段训练长度
TAU = d_head / math.sqrt(L_train)  # = 64 / 45.25... ≈ 1.4142

# 验证: 与 Phase 17c 一致
assert abs(TAU - 1.4142) < 0.001, f"τ 计算错误: got {TAU}"
```

### 2.4 数据混合比例

```
总 chunks = 500M tokens / 2048 tokens_per_chunk ≈ 244K chunks

分配:
  90% = 纯 LM (FineWeb-Edu)         → ~220K chunks
   5% = Passkey-mix (已有实现)        → ~12K chunks
   5% = QA-mix (新增)                → ~12K chunks
```

---

## 三、QA-Mix 数据构造

### 3.1 数据源

**QuALITY train set** (2523 条长文 QA) — 直接用 SCROLLS 格式：

```python
from datasets import load_dataset
ds = load_dataset("tau/scrolls", "quality", split="train", trust_remote_code=True)
# 每条: {"input": "passage + question + options", "output": "answer letter or text"}
```

**为什么选 QuALITY**：
- 文章本身就长 (median ~5K tokens) → 天然需要长距离 attention
- 多选 QA → 有明确的对/错信号
- Eval 时用同一个 dataset 的 validation split → 公平对比
- Phase 21b 已有 eval 代码（`phase21b_quality_eval_clean.py`）

### 3.2 QA 样本格式

```
<QA>
Read the following passage and answer the question.

{article_text}

Question: {question}
Options:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}

Answer: ({correct_letter})
</QA>
```

**Loss masking**: 只在 `Answer: (X)` 这部分算 loss。前面所有 context 只做 teacher forcing，不算 loss。

### 3.3 长度处理

| 情况 | 处理 |
|------|------|
| 文章 + QA < 2048 tokens | 正常使用，尾部 pad 或截断到 seq_len |
| 文章 + QA > 2048 tokens | 截断文章中间部分，保留开头 + 结尾 + Q&A |
| 文章太短 (< 500 tokens) | 跳过（QuALITY 极少出现这种情况） |

⚠️ **关键**: 不要用 distractor padding！这里不是测 distractor retrieval，是教模型读文章回答问题。文章原文就好。

### 3.4 重复利用

QuALITY train 只有 2523 条，但需要 ~12K chunks。处理方式：

- 每条文章有 **多个问题**（QuALITY 每篇文章约 10-20 道题）
- 2523 条 × ~15 个问题/条 ≈ 37K QA pairs → 远超 12K 需求
- 每个 QA pair 作为独立 chunk
- 如果仍不够，shuffle 后重复（passkey-mix 也在重复同一个 passkey 模板）

---

## 四、Eval 方案

### 4.1 两层 Eval

**Layer 1: QA Accuracy（核心指标）**

```
来源: phase21b_quality_eval_clean.py — 使用 options_nll scoring
```

| 长度 | 模式 | YaRN | 预期 |
|------|------|------|------|
| 2K (in-dist) | 标准，无 padding | 无 | EVQ ≈ Geo（waterbed，训练长度内） |
| 4K (2× extrap) | 标准，无 padding | 无 | EVQ 开始赢 |
| 8K (4× extrap) | 标准，无 padding | 无 | **EVQ 显著赢** |
| 8K (4× extrap) | 标准 | +YaRN | **EVQ+YaRN >> Geo+YaRN** |
| 16K (8× extrap) | distractor pad | +YaRN | 同上，更大差距 |

**Layer 2: PPL + Passkey（验证没有 regression）**

```
来源: phase17c 的标准 eval（已有代码）
```

QA-mix 不应该损害 PPL 或 passkey 性能。如果 PPL 恶化超过 2%，说明 QA-mix 比例太高或数据格式有问题。

### 4.2 Eval 指标

| 指标 | 实现 | 说明 |
|------|------|------|
| **options_nll accuracy** | `phase21b_quality_eval_clean.py --scoring_mode options_nll` | 给 4 个选项算 NLL，选最低的 |
| gold_answer_nll | `--scoring_mode gold_answer_nll` | 正确答案的绝对 NLL |
| PPL | Phase 17c 标准 eval | 验证没有 regression |
| Passkey retrieval | Phase 17c 标准 passkey eval | 验证没有 regression |

### 4.3 YaRN 配置

```
来源: phase21b_scrolls_finetune.py line 147-170 (正确的 Progressive YaRN)
```

```python
# ⚠️ 必须使用 Progressive YaRN，不是 NTK-aware
# 检查方法: 代码中必须有 smoothstep ramp 和 per-channel scaling
# 如果看到 `factor = scale ** (dim / (dim - 2))` 就是 BUG

YaRN_scale = eval_length / train_length  # e.g., 8192/2048 = 4.0
# 不是别的比值！
```

---

## 五、代码改动清单

### 5.1 需要修改的文件

**`phase17c_454m_1024_to_2048_continue.py`** — 主训练脚本

改动量: 小 (~50 行)

```
1. 新增 QA 数据加载函数:
   - load_quality_qa_pairs(data_dir) → List[dict]
   - 从 HuggingFace tau/scrolls/quality 或本地 jsonl 加载

2. 新增 QA 样本构造函数:
   - make_qa_training_sample(qa_pair, tokenizer, seq_len, seed) → Tensor
   - 将 article + Q + options + answer 拼成 token sequence
   - 返回 token ids (answer 部分在正确位置)

3. 修改 build_mixed_train_tensor():
   - 增加 QA_MIX_RATIO 环境变量 (默认 0.05)
   - 在 passkey 替换之后，再随机替换 5% 为 QA 样本
   - cache 文件名加入 qa_mix 标识

4. 新增环境变量:
   - PHASE17C_QA_MIX_RATIO (默认 0.05)
   - PHASE17C_QA_DATA_DIR (可选，本地 QuALITY 数据)
```

### 5.2 Loss Masking 实现

```python
# ⚠️ 这是关键改动，需要仔细对齐

# 现有 passkey-mix: 全 sequence 算 loss（因为 passkey 的每个 token 都有意义）
# QA-mix: 只在 answer tokens 上算 loss

# 方案: 使用 label masking（把非 answer 部分的 target 设为 -100）
def make_qa_training_sample(qa_pair, tokenizer, seq_len, seed):
    # 1. 构造 prompt + answer string
    prompt_str = format_qa_prompt(qa_pair)  # article + Q + options + "Answer: "
    answer_str = f"({qa_pair['correct_letter']})"

    # 2. Tokenize
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)

    # 3. 拼接并 pad/truncate to seq_len
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > seq_len:
        # 截断 article 中间部分，保留 Q&A
        ...

    # 4. 创建 label tensor: prompt 部分 = -100, answer 部分 = token_ids
    labels = torch.full((seq_len,), -100, dtype=torch.long)
    answer_start = len(prompt_ids)
    answer_end = answer_start + len(answer_ids)
    labels[answer_start:answer_end] = torch.tensor(full_ids[answer_start:answer_end])

    # 返回 (input_ids, labels) 或编码 labels 到 input tensor 的高位
    return input_ids_tensor, labels_tensor
```

⚠️ **注意**: 现有的 `train_model_ga()` 函数做的是标准 causal LM loss（所有 token 都算 loss）。QA-mix 的 answer-only loss 需要修改训练循环，让它支持 per-token loss mask。

**两种实现方案**：

| 方案 | 改动量 | 推荐 |
|------|--------|------|
| A: 全 sequence 算 loss（包括 article tokens） | 零改动训练循环 | ✅ **推荐** |
| B: 只在 answer tokens 算 loss（label masking） | 改训练循环 | 精确但复杂 |

**推荐方案 A 的理由**:
- Passkey-mix 也是全 sequence 算 loss，证明这样有效
- Article 文本本身就是好的 LM 训练数据（来自 QuALITY 的长文）
- 不需要改训练循环 = 不引入新 bug
- QA pair 的 article + question + answer 整体作为一个 causal LM chunk，模型自然学到"读完文章回答问题"的 pattern

### 5.3 不需要改的文件

| 文件 | 原因 |
|------|------|
| `run_evq_sweep.py` | 底层库，不动 |
| `phase21b_quality_eval_clean.py` | 直接复用做 eval |
| `eval_passkey_scratch.py` | passkey eval 不变 |
| `phase21b_scrolls_finetune.py` | 不需要 finetune 了！ |

---

## 六、实验执行

### 6.1 执行策略（适配当前进度）

**seed 42 已完成 Stage 3，不能重跑**。两个选项：

**选项 A（推荐）: seed 43/44 的 Stage 3 加 QA-mix**
- seed 43/44 正在跑 Stage 2 (L=1024)，Stage 3 还没开始
- 等 Stage 2 跑完后，Stage 3 启用 QA-mix
- 同时用 seed 42 的 Stage 3 checkpoint（无 QA-mix）做对照
- 对比: seed42 (passkey-only) vs seed43/44 (passkey+QA) 可以分离 QA-mix 的增量效果

**选项 B: seed 42 追加一个 Stage 4 (L=2048, +QA-mix)**
- 从 seed 42 Stage 3 checkpoint 继续训 200M tokens，加入 QA-mix
- 快速但引入了额外训练量的混淆变量

**推荐选项 A**：seed 43/44 Stage 3 直接加 QA-mix，最干净。

### 6.2 执行命令 (seed 43/44 Stage 3)

```bash
# 等 Stage 2 完成后执行
# Init: seed43 的 Stage 2 (L=1024) checkpoint
export PHASE17C_QA_MIX_RATIO=0.05
export PHASE17C_QA_DATA_DIR=/path/to/quality_data   # 可选
export PHASE17C_SEED=43
export PHASE17C_RUN_ONLY=evq
export PHASE17C_EVQ_INIT_CKPT=/root/autodl-tmp/evq_phase17b/454m_evq_seed43_continue1024/model.pt

python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

```bash
# Geo baseline (同样加 QA-mix)
export PHASE17C_RUN_ONLY=geo
export PHASE17C_GEO_INIT_CKPT=/root/autodl-tmp/evq_phase17b/454m_geo_seed43_continue1024/model.pt
python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

**训练时间估算**: 500M tokens @ L=2048, effective_bs=20 ≈ Phase 17c stage 3 一样的时间。

### 6.3 Phase 22b: Eval

```bash
# QuALITY eval @2K (in-dist, 无 YaRN)
python scripts/core_text_phases/phase21b_quality_eval_clean.py \
    --model_pt $EVQ_CKPT \
    --tier 454m --rope evq --tau 1.4142 --base 500000 \
    --target_len 2048 --protocol in_dist_nopad \
    --scoring_mode options_nll \
    --eval_samples 200 --data_dir $QA_DATA \
    --output_dir results/phase22/eval/evq_2k_nopad/

# QuALITY eval @8K (4× extrap, +YaRN)
python scripts/core_text_phases/phase21b_quality_eval_clean.py \
    --model_pt $EVQ_CKPT \
    --tier 454m --rope evq --tau 1.4142 --base 500000 \
    --yarn 1 --yarn_scale 4.0 \
    --target_len 8192 --protocol article_pad_extrap \
    --scoring_mode options_nll \
    --eval_samples 200 --data_dir $QA_DATA \
    --output_dir results/phase22/eval/evq_8k_yarn/
```

⚠️ **Eval 时的 τ**: 仍然用 **训练时的 τ=1.4142**，不是 eval 长度对应的 τ。因为模型是在 L=2048 训练的，inv_freq 是固定的。

⚠️ **YaRN scale**: `eval_length / train_length = 8192 / 2048 = 4.0`

### 6.4 对比设计

| Seed | Stage 3 配置 | 用途 |
|------|-------------|------|
| 42 | Passkey 5% only (已完成) | **对照组** — 无 QA-mix |
| 43 | Passkey 5% + QA 5% (待跑) | **实验组** |
| 44 | Passkey 5% + QA 5% (待跑) | **实验组** |

这样每个 method (EVQ/Geo) 都有 1 个对照 + 2 个实验，可以看 QA-mix 的增量效果。

---

## 七、预期结果与判断标准

### 7.1 预期

| 设置 | EVQ 预期 | Geo 预期 | 预期差距 |
|------|---------|---------|---------|
| QA @2K (in-dist) | ~40-50% | ~42-52% | Geo 略赢 (waterbed) |
| QA @4K (2× extrap) | ~35-45% | ~30-40% | **EVQ 赢 ~5pp** |
| QA @8K (4× extrap, raw) | ~25-35% | ~20-25% | **EVQ 赢 ~10pp** |
| QA @8K (+YaRN) | ~40-50% | ~25-35% | **EVQ+YaRN >> Geo+YaRN** |
| PPL @2K (regression check) | ~Phase17c | ~Phase17c | 无退化 |
| Passkey @4K | 100% | ≥80% | 保持 |

### 7.2 判断标准

| 结果 | 行动 |
|------|------|
| EVQ @8K+YaRN accuracy > Geo @8K+YaRN accuracy by ≥5pp | ✅ **成功** — 写入论文 |
| 差距 2-5pp | ⚠️ 方向正确但需要 multi-seed 确认 |
| EVQ ≈ Geo | ❌ 分析原因（可能 QA-mix 5% 不够，或文章长度不够） |
| PPL regression > 2% | ❌ 降低 QA-mix 比例到 2% |

### 7.3 与 Passkey 结果的一致性检查

**必须验证**: QA-mix 没有破坏 passkey 性能。如果 passkey 从 100%→<90%，说明两种 mix 在争抢训练信号，需要调比例。

---

## 八、风险与备选

### 8.1 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| QuALITY 文章 > 2048 tokens，截断后 QA 无意义 | 中 | 选短文章的 QA pairs，或 L=4096 训练 |
| 5% QA-mix 比例不够，模型学不会 QA | 低 | passkey 5% 就够了，QA 格式更接近 LM |
| QA-mix 数据重复太多导致过拟合 | 低 | 37K QA pairs vs 12K chunks，不会严重重复 |
| 训练循环改动引入 bug | 中 | **用方案 A（全 sequence loss），不改训练循环** |

### 8.2 如果 454M @2048 文章长度不够

QuALITY 文章 median ~5K tokens，很多会被截断到 2048。备选方案：

**Phase 22-ALT: 在 Stage 2 (L=1024) 混入短 QA**

用 SQuAD/TriviaQA 这种短文 QA（context < 500 tokens）在 L=1024 的 Stage 2 混入。这样不需要截断，而且模型在更早期就学会 QA 能力。然后 Stage 3 继续正常训练（L=2048，可以加长文 QA）。

**Phase 22-ALT-B: 直接在 L=4096 continue-train 加 QA-mix**

在 Phase 17c (L=2048) 之后再加一个 Stage 4: L=4096 continue-train + QA-mix。τ* = 64/√4096 = 1.0。代价是需要 rechunk 数据到 L=4096。

---

## 九、Claude Code 实现 Checklist

**开始写代码前，逐项确认：**

- [ ] **读了 `team/AI_HANDOFF_PITFALLS.md`** — 尤其 §1, §2, §8
- [ ] **τ = d_head / √L_train** — 用 64/√2048 = 1.4142，不是任何其他值
- [ ] **d_head = 64** (454M: 1024/16)，不是 hidden_size 1024
- [ ] **YaRN 实现** — 从 `phase21b_scrolls_finetune.py` line 147-170 复制，不要自己写
- [ ] **YaRN scale = eval_len / train_len**，不是其他比值
- [ ] **Eval 使用 `phase21b_quality_eval_clean.py`**，不是旧的 `phase21b_quality_eval.py`
- [ ] **Eval scoring_mode = options_nll**，不是首字符匹配
- [ ] **不改 `run_evq_sweep.py`** — 底层库
- [ ] **不改训练循环结构** — 用方案 A (全 sequence loss)
- [ ] **数值精度**: inv_freq 在 float64 下计算
- [ ] **Init checkpoint**: 来自 Phase 17b stage2 (progressive)，不是 retrofit
- [ ] **QA 数据**: 从 `tau/scrolls/quality` 加载，设 `HF_ENDPOINT=https://hf-mirror.com`
- [ ] **环境变量名**: 以 `PHASE17C_` 开头，与现有一致
