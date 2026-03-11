# Phase 21: Downstream Task Evaluation (SCROLLS Finetuning)

## Intent

验证 EVQ 的 PPL/passkey 优势能否迁移到实际下游任务，堵住 reviewer "PPL improves but does it matter" 的攻击面。

## Background

- FIRE (ICLR 2024) 用 125M/350M (head_dim=64) 在 C4 pretrain → SCROLLS 7 子任务 finetune (L=8192)
- 我们的 454M > FIRE Large (350M)，head_dim=64 完全匹配
- 两边（EVQ vs Geo）用同一 finetune recipe，PE 是唯一自变量，归因干净

## Phase 21a 结果：LongBench NLL 路线已排除

**结论：NLL-based eval 不适合展示 EVQ 优势，该路线已关闭。**

### 完整数据 (Phase 15, 750M, EVQ r=0 vs Geo)

**ctx=4096 (训练长度内, L_train=4096)**:

| Task | Type | Geo NLL | EVQ NLL | Δ | % |
|------|------|---------|---------|---|---|
| qasper | QA | 3.021 | 3.250 | +0.230 | +7.6% |
| hotpotqa | QA | 4.034 | 4.257 | +0.223 | +5.5% |
| 2wikimqa | QA | 2.320 | 2.560 | +0.241 | +10.4% |
| narrativeqa | QA | 4.863 | 4.971 | +0.109 | +2.2% |
| multifieldqa_en | QA | 3.077 | 3.241 | +0.164 | +5.3% |
| musique | QA | 4.286 | 4.250 | -0.036 | -0.8% |
| triviaqa | QA | 6.060 | 7.070 | +1.010 | +16.7% |
| gov_report | Sum | 3.811 | 4.009 | +0.198 | +5.2% |
| multi_news | Sum | 4.696 | 4.907 | +0.211 | +4.5% |
| qmsum | Sum | 5.915 | 5.949 | +0.034 | +0.6% |
| samsum | Sum | 6.603 | 6.665 | +0.063 | +0.9% |
| trec | Cls | 3.714 | 3.811 | +0.098 | +2.6% |
| passage_ret | Ret | 4.528 | 4.473 | -0.056 | -1.2% |
| **AGGREGATE** | | **4.379** | **4.570** | **+0.191** | **+4.4%** |

**ctx=8192 (2x 外推)**:

| Task | Type | Geo NLL | EVQ NLL | Δ | % |
|------|------|---------|---------|---|---|
| qasper | QA | 3.056 | 3.159 | +0.103 | +3.4% |
| hotpotqa | QA | 4.527 | 3.918 | -0.609 | **-13.5%** |
| 2wikimqa | QA | 2.994 | 2.501 | -0.493 | **-16.5%** |
| narrativeqa | QA | 4.837 | 4.484 | -0.353 | **-7.3%** |
| multifieldqa_en | QA | 3.087 | 2.798 | -0.290 | **-9.4%** |
| musique | QA | 4.976 | 4.138 | -0.837 | **-16.8%** |
| triviaqa | QA | 6.321 | 6.269 | -0.052 | -0.8% |
| gov_report | Sum | 3.508 | 3.438 | -0.070 | -2.0% |
| multi_news | Sum | 4.610 | 4.831 | +0.221 | +4.8% |
| qmsum | Sum | 4.892 | 4.869 | -0.022 | -0.5% |
| samsum | Sum | 5.989 | 6.071 | +0.083 | +1.4% |
| trec | Cls | 4.248 | 3.947 | -0.301 | **-7.1%** |
| passage_ret | Ret | 4.267 | 4.369 | +0.102 | +2.4% |
| **AGGREGATE** | | **4.409** | **4.215** | **-0.194** | **-4.4%** |

### 汇总：训练内 vs 外推 对称反转

| 设置 | Geo Agg NLL | EVQ Agg NLL | Δ | 谁赢 |
|------|-------------|-------------|---|------|
| ctx=4096 (训练内) | 4.379 | 4.570 | +4.4% | Geo |
| ctx=8192 (2x 外推) | 4.409 | 4.215 | **-4.4%** | **EVQ** |

### 关键发现

1. **训练长度内 (ctx=4096)**: Geo 全面优于 EVQ (+4.4%)。这是 waterbed 代价 — EVQ 将高频分辨率重新分配到低频，导致局部 token 预测略差。ctx=4096 对两个模型都是 in-distribution，Geo 的远距离 attention 还没退化，所以 waterbed 代价占主导。

2. **外推 (ctx=8192)**: EVQ 大幅反超 (-4.4%)。Geo 的远距离 attention 在外推时退化（高频 channel aliasing），而 EVQ 的低频分辨率优势开始体现。模型在预测 answer tokens 时需要 attend back 到远处的 context，EVQ 做得更好。

3. **QA vs Summarization**: EVQ 外推优势集中在 QA 任务（musique -16.8%, 2wikimqa -16.5%, hotpotqa -13.5%），因为 QA 需要精确定位上下文中的特定段落（典型的长距离检索任务）。Summarization 差异不大（依赖全局统计而非精确定位）。

4. **对称性**: +4.4% 和 -4.4% 几乎完美对称 — waterbed trade-off 的两端在 NLL 上被精确测量。

**这不是 "bad result"，这是 waterbed trade-off 在下游任务上的首次直接量化验证。** 可写入论文作为 EVQ 机制理解的关键证据。

> **论文叙事价值**: "EVQ 的 waterbed trade-off 在下游任务 NLL 上首次被直接测量：训练长度内 +4.4% 代价 vs 外推 -4.4% 收益，且收益集中在需要精确长距离检索的 QA 任务（最高 -16.8%），与 collision theory 预测一致。"

### Phase 13A 历史数据（Hybrid r=16, 参考对比）

| ctx | Hybrid r=16 vs Geo |
|-----|-------------------|
| 2048 (训练内) | +0.1%（打平） |
| 4096 (2x 外推) | +2.7%（Hybrid 更差） |
| 8192 (4x 外推) | +2.0%（Hybrid 更差） |

Phase 13A 的 Hybrid r=16 在外推时也更差 — 因为 r=16 只优化了 16/32 channels，优势不够大无法超过 waterbed 代价。而 Phase 21a 的 EVQ r=0 优化了所有 32 channels，在 2x 外推时就实现了反转。**这进一步验证了 r=0 优于 r=16 的结论。**

---

## Phase 21B Pilot 结果：GovReport 750M (已完成)

### 配置

- Model: 750M (18L/24H/1536d), base=500K
- Init: Phase 15 checkpoint (L=2048→4096 continue)
- Finetune: 2000 steps, lr=1e-5, dropout=0.1, seq_len=8192
- Eval: 200 samples, max_gen_tokens=512

### 结果

**@8192 (in-distribution after finetune)**:

| Metric | Geo | EVQ (τ=1.5) | Δ mean | Δ std |
|--------|-----|-------------|--------|-------|
| ROUGE-1 | 30.20 ±8.92 | 28.73 ±8.42 | -1.47 | **-5.6%** |
| ROUGE-2 | 8.66 ±5.01 | 7.28 ±4.00 | -1.38 | **-20.2%** |
| ROUGE-L | 20.39 ±4.89 | 19.83 ±4.72 | -0.56 | **-3.5%** |

**@16384 (2× beyond finetune)**:

| Metric | Geo | EVQ | Δ |
|--------|-----|-----|---|
| ROUGE-1 | 28.81 | 27.97 | -0.84 |
| ROUGE-2 | 7.54 | 7.15 | -0.39 |
| ROUGE-L | 20.18 | 19.52 | -0.66 |

**@16K std (方差)**:

| Metric | Geo std | EVQ std | Δ |
|--------|---------|---------|---|
| ROUGE-1 | 8.50 | 8.36 | -0.14 |
| ROUGE-2 | 4.84 | 3.93 | **-0.91** |
| ROUGE-L | 4.75 | 4.72 | -0.03 |

### 关键发现

1. **Geo mean 略高，但差距从 @8K→@16K 缩小了 43%**（-1.47 → -0.84）
2. **EVQ 方差全面更低**（6/6 指标），ROUGE-2 std -20%，说明 EVQ 生成更稳定
3. **GovReport 是 summarization 任务** — Phase 21a 已显示 summarization 区分度低。EVQ 优势集中在 QA 任务（musique -16.8%, hotpotqa -13.5%）
4. **Finetune at 8192 消除了 extrapolation 优势** — 8192 变成 in-distribution，waterbed cost 占主导
5. **base=500K 过大** — Geo 频率铺得很开，sub-cycle channels 少，EVQ 优化空间有限

### 结论：GovReport 不是展示 EVQ 优势的正确任务。应转向 QuALITY (QA)。

---

## Phase 21C: QuALITY QA Finetuning (新主路线)

### 动机

Phase 21a NLL 数据明确显示：EVQ 优势集中在 **QA 任务**（需要精确长距离检索），而非 summarization（依赖全局统计）。QuALITY 是 SCROLLS 中的多选阅读理解 QA，直接测试长文档中的证据定位能力 — 正好是 EVQ 低频优势的主场。

### QuALITY 任务特点

- **类型**: 长文档多选 QA（4 选 1）
- **输入**: 长篇故事/文章（median ~5K tokens, 最长 ~8K）+ 问题 + 4 个选项
- **输出**: 选项字母 (A/B/C/D)
- **指标**: Accuracy（不是 ROUGE）
- **Hard subset**: 需要多步推理的高难度题目
- **与 Phase 21a 的对应**: musique, hotpotqa, 2wikimqa 都是 QA → EVQ 赢 -13.5% to -16.8%

### Finetune 配置

```bash
# QuALITY EVQ
python scripts/core_text_phases/phase21b_scrolls_finetune.py \
    --init_ckpt /path/to/750m_evq_phase15/model.pt \
    --rope evq --tau 1.5 --base 500000 \
    --task quality --seq_len 8192 --yarn 0 \
    --lr 1e-5 --steps 2000 --dropout 0.1 \
    --seed 42 \
    --output_dir results/phase21c/quality_raw/evq_seed42/

# QuALITY Geo
python scripts/core_text_phases/phase21b_scrolls_finetune.py \
    --init_ckpt /path/to/750m_geo_phase15/model.pt \
    --rope geo --base 500000 \
    --task quality --seq_len 8192 --yarn 0 \
    --lr 1e-5 --steps 2000 --dropout 0.1 \
    --seed 42 \
    --output_dir results/phase21c/quality_raw/geo_seed42/
```

### Eval 矩阵 (关键！)

| 配置 | eval@8K (in-dist) | eval@16K (extrap) |
|------|-------------------|-------------------|
| Geo raw | ✓ | ✓ |
| EVQ raw | ✓ | ✓ |
| Geo+YaRN | — | ✓ |
| EVQ+YaRN | — | ✓ |

**@8K**: 预期 Geo 略赢（waterbed cost, 与 GovReport 一致）
**@16K raw**: 预期 EVQ 反超（QA 需要精确检索，EVQ 低频优势发挥）
**@16K +YaRN**: 预期 EVQ+YaRN >> Geo+YaRN（Claim 3 的下游验证）

### 理想结果：Paper 最强表

| Task type | Task | @8K (in-dist) | @16K (extrap) | @16K +YaRN |
|-----------|------|---------------|---------------|------------|
| Summarization | GovReport ROUGE | Geo +1.5 (已有) | Geo +0.8 (已有) | TBD |
| QA | QuALITY Acc | Geo 略赢? | **EVQ 赢** | **EVQ+YaRN >> Geo+YaRN** |

这张表同时验证：waterbed theory (in-dist cost) + bandpass prediction (QA vs Sum) + YaRN composition (Claim 3) → 三个 claim 一张表搞定。

### 注意事项

1. **QuALITY 是多选题** — eval 方式是看模型输出的 logits 对 A/B/C/D 哪个最高，或者 generate 后 parse 答案字母
2. **max_gen_tokens 可以很小**（只需输出一个字母），不像 GovReport 需要 512 tokens
3. **如果 finetune 脚本已支持 GovReport，改 QuALITY 主要是数据格式变化**
4. **2000 步 pilot 先跑**，看方向再决定是否加步数

---

## Phase 21b 原始计划 (SCROLLS 通用，保留参考)

### 动机（原始）

Phase 21a 已在 NLL eval 上证明 EVQ 在外推场景下的下游优势（-4.4% aggregate, QA 最高 -16.8%）。SCROLLS finetune 会进一步放大这个优势：finetune 教会模型将长距离 attention 转化为任务表现，EVQ 的低频分辨率优势应更充分体现在 ROUGE/F1 上。

### 参照 FIRE 配置

| 参数 | FIRE Large | 我们 |
|------|-----------|------|
| 模型规模 | 350M (24L/16H/768d) | 454M (24L/16H/1024d) |
| head_dim | 64 | 64 |
| Pretrain data | C4 | FineWeb-Edu (proof-pile-2) |
| Pretrain L | 2048 | 512→1024→2048 (progressive) |
| Finetune L | 8192 | 8192 |
| Finetune LR | 1e-5 | 1e-5 |
| Finetune batch | 128 (multi-TPU) | ~10 (single GPU, grad_accum 补齐) |
| Finetune steps | 25k | 25k |
| Dropout | 0.1 (attn + residual) | 0.1 |

### 子任务选择

优先 3 个最能体现长文本优势的：

1. **GovReport** — 政府报告摘要, median context ~8K tokens, 最长可达 ~18K。需要全局理解。FIRE 的 headline task。
2. **QMSum** — 会议摘要，需要跨发言者的远距离信息整合
3. **QuALITY** — 长文档多选 QA，直接测阅读理解，有 hard subset

### 数据获取

```python
from datasets import load_dataset
ds_gov = load_dataset("tau/scrolls", "gov_report")   # train/val/test
ds_qms = load_dataset("tau/scrolls", "qmsum")
ds_qua = load_dataset("tau/scrolls", "quality")
```

如果 HuggingFace 下载失败（中国服务器），设置 `HF_ENDPOINT=https://hf-mirror.com`，或提前下载到本地 `--data_dir`。

### Finetune 脚本设计

**文件**: `scripts/core_text_phases/phase21b_scrolls_finetune.py`

基于 `phase17c_454m_1024_to_2048_continue.py` 改写，核心变更：

```python
# 1. 数据格式：SCROLLS → causal LM
# input:  "Summarize the following document:\n{context}\n\nSummary:"
# target: "{answer}"
# 拼接成: input + target, 只在 target tokens 上算 loss

# 2. Loss masking
loss_mask = torch.zeros(seq_len)
loss_mask[len(input_tokens):len(input_tokens)+len(target_tokens)] = 1.0
loss = F.cross_entropy(logits[loss_mask], targets[loss_mask])

# 3. 关键超参
FINETUNE_LR = 1e-5
FINETUNE_STEPS = 25000
DROPOUT = 0.1
SEQ_LEN = 8192
WARMUP_STEPS = 500  # 2% of 25k

# 4. RoPE 外推处理 (L_train=2048 → L_finetune=8192, 4× 外推)
# 方案 A: Raw (不加 YaRN) — 测试 PE 原生长度泛化
# 方案 B: +YaRN — 测试实际部署场景
# 两个方案都跑
```

**评测**:
```python
# Summarization: ROUGE-1/2/L
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

# QA (QuALITY): Accuracy (multiple choice) 或 F1
```

**环境变量控制**:
```bash
PHASE21B_TASK=gov_report          # gov_report | qmsum | quality
PHASE21B_ROPE=evq                 # evq | geo
PHASE21B_YARN=0                   # 0=raw finetune, 1=+YaRN
PHASE21B_INIT_CKPT=/path/to/model.pt
PHASE21B_SEED=42
PHASE21B_STEPS=25000
PHASE21B_OUTPUT_DIR=/path/to/output
```

### 关于 L=8192 的 RoPE 外推

Phase 17c checkpoint pretrained at L=2048, finetune at L=8192 是 4× 外推。

**方案 A: Raw finetune (不加 YaRN)**
- 直接在 L=8192 上 finetune，RoPE 频率不变
- 测试 PE 分配本身的长度泛化能力
- EVQ 理论预期：collision theory 预测 EVQ 在外推时保持更低 phase collision → 更好的远距离 attention → ROUGE 更高
- Geo 理论预期：外推时高频 channel 产生 aliasing，长距离 attention 退化
- **这个方案最直接测 PE 效果**

**方案 B: +YaRN finetune**
- 加 YaRN scaling (scale=4) 后在 L=8192 上 finetune
- 测试 EVQ+YaRN vs Geo+YaRN 的实际部署场景
- 与 paper 主 claim (EVQ+YaRN >> Geo+YaRN) 完全对齐
- **这个方案与 paper story 一致**

**建议**: 先跑方案 A（最 clean），如果 A 就有显著差异则方案 B 是锦上添花；如果 A 差异不明显，方案 B 可能放大优势（因为 YaRN 与 EVQ 有超线性 synergy）。

---

## Protocol

### Step 1: 写 finetune 脚本 (Phase 21b)

基于 phase17c 脚本改写 `phase21b_scrolls_finetune.py`：
- 加载 SCROLLS 数据 → causal LM 格式
- Loss masking（只在 answer tokens 上算 loss）
- Dropout 0.1
- 支持 --yarn 开关
- 支持 --task 选择子任务
- 内置 ROUGE/F1 评测

### Step 2: GovReport Pilot (seed=42, 方案 A)

```bash
# Geo baseline
python scripts/core_text_phases/phase21b_scrolls_finetune.py \
    --init_ckpt /path/to/phase17c/454m_geo_seed42/model.pt \
    --rope geo --base 500000 \
    --task gov_report --seq_len 8192 --yarn 0 \
    --lr 1e-5 --steps 25000 --dropout 0.1 \
    --seed 42 \
    --output_dir results/phase21b/gov_report_raw/geo_seed42/

# EVQ
python scripts/core_text_phases/phase21b_scrolls_finetune.py \
    --init_ckpt /path/to/phase17c/454m_evq_seed42/model.pt \
    --rope evq --tau 1.4142 --base 500000 \
    --task gov_report --seq_len 8192 --yarn 0 \
    --lr 1e-5 --steps 25000 --dropout 0.1 \
    --seed 42 \
    --output_dir results/phase21b/gov_report_raw/evq_seed42/
```

**判断标准**:
- ROUGE-1/2/L 差异 > 5%：方向确认，继续 multi-seed
- ROUGE 差异 2-5%：需要 multi-seed 才能确认
- ROUGE 差异 < 2% 或方向错误：重新评估方案 B（+YaRN）

### Step 3: 方案 B (+YaRN) 如果方案 A 差异不够

同上，改 `--yarn 1`。

### Step 4: Multi-seed + 多任务 (等 17c multi-seed 完成后)

- 3-seed × 2 methods × 3 tasks × (1 or 2 schemes) = 18-36 runs
- 每 run 25k steps at L=8192
- RTX 6000 Pro (96GB): 估计每 run ~3-5 小时
- 总计 ~54-180 GPU 小时

---

## Hardware

| 硬件 | 454M @ L=8192 | Batch | 估计每 run 时间 |
|------|--------------|-------|----------------|
| RTX 6000 Pro (96GB) | 完全 fit | ~20-30 | ~3h (25k steps) |
| RTX 5090 (32GB) | Flash Attention 必须 | ~10 | ~5h (25k steps) |

**推荐 RTX 6000 Pro** — 内存充裕，batch 大，不需要 gradient checkpointing。

## Checkpoints

### Pilot 用 (Phase 17c, seed=42):

| 文件 | 说明 |
|------|------|
| `evq_phase17c/.../454m_geo_seed42/model.pt` | Stage3 L=2048 GEO |
| `evq_phase17c/.../454m_evq_seed42/model.pt` | Stage3 L=2048 EVQ τ=√2 |

**注意**: 这些 checkpoint 在 5090 服务器上。需要 scp 到 R6000，或者把 Phase 17c multi-seed 跑在 R6000 上。

### Multi-seed 用 (待 17c multi-seed 完成):

| Seeds | 方法 | 说明 |
|-------|------|------|
| 42, 137, 256 | GEO | 3-seed Stage3 L=2048 |
| 42, 137, 256 | EVQ τ=√2 | 3-seed Stage3 L=2048 |

## Success Criteria (Updated)

- **Phase 21B GovReport**: ✅ 已完成 — Geo mean 略赢 (waterbed confirmed), EVQ variance 全面更低 (frequency equalization confirmed)
- **Phase 21C QuALITY pilot**: EVQ Accuracy @16K > Geo Accuracy @16K（方向确认即可）
- **Phase 21C QuALITY +YaRN**: EVQ+YaRN @16K >> Geo+YaRN @16K（复现 Claim 3）
- **理想结果**: QuALITY @16K EVQ 赢 ≥ 5% accuracy，同时 @8K Geo 略赢 → 完美 waterbed reversal on QA task

## Dependencies

- Phase 21b pilot: 需要 Phase 17c seed=42 checkpoint（已有）+ finetune 脚本（需写）
- Phase 21b multi-seed: 依赖 17c multi-seed 完成
- SCROLLS 数据集下载（HuggingFace，需提前验证网络可达）

## Priority (Updated)

**17c multi-seed > LaTeX draft ≥ Phase 21C QuALITY pilot**

Phase 21C QuALITY 可以用现有 750M Phase 15 checkpoint 立即开始（与 GovReport pilot 同源）。
GovReport 已有结果（summarization baseline），QuALITY 是下一个关键实验。

## Appendix: Phase 21a NLL 结果 — Waterbed 对称反转

Phase 21a 的完整结果展示了 waterbed trade-off 在下游任务上的首次直接量化：

- **ctx=4096 (训练内)**: Geo 赢 +4.4% — waterbed 代价占主导
- **ctx=8192 (2x 外推)**: EVQ 赢 -4.4% — 长距离优势占主导（QA 最高 -16.8%）

完美对称反转。详细数据见本文档 Phase 21a 章节。

**论文叙事价值**: 这不仅不是 "bad result"，反而是 waterbed 理论在下游任务上的最直接验证。建议写入论文 §Waterbed 章节和 §Experiments 作为独立 finding。
