# Data Preparation Guide

本文档说明 EVQ-Cosh 实验使用的所有数据集及其获取方式。

---

## 1. FineWeb-Edu (主训练数据)

所有 from-scratch 预训练实验（Phase 8-17）使用 FineWeb-Edu 作为训练数据。

```python
from datasets import load_dataset

# Streaming mode (推荐, 无需下载全部数据)
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)
```

- **来源**: HuggingFace Hub `HuggingFaceFW/fineweb-edu`, config `sample-10BT`
- **加载方式**: Streaming (无需本地存储)
- **Tokenizer**: `EleutherAI/gpt-neox-20b` (vocab_size=50304)
- **预处理**: 在 `run_evq_sweep.py` 中自动完成 (tokenize → chunk to seq_len → shuffle)
- **引用**: 见 `paper/refs/references.bib` 中 FineWeb 条目

## 2. TinyStories (小规模验证)

50M tier 快速验证实验使用 TinyStories。

```python
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train")
```

- **来源**: HuggingFace Hub `roneneldan/TinyStories`
- **用途**: 50M 模型快速 τ-sweep 验证 (~4 hours)
- **引用**: Eldan & Li, 2023

## 3. Passkey Mix (合成检索数据)

Phase 14+ 实验在训练中混入 5-10% passkey 检索样本。

```python
from scripts.supporting_eval.eval_passkey_scratch import (
    MixedDataset,
    make_passkey_training_sample,
)
```

- **生成方式**: `make_passkey_training_sample()` 动态生成，无需预下载。
- **训练混合**: `run_evq_sweep.py` 通过 `maybe_wrap_with_passkey_mix()` 调用 `MixedDataset`；比例由 `--passkey_mix_ratio` 或 `PASSKEY_MIX_RATIO` 控制。
- **评估指标**: 主表的 PK 指标是 teacher-forced NLL-gap retrieval rate；`eval_passkey_scratch.py` 也包含 autoregressive exact-match 评估入口，但二者不能混写。
- **用途**: 训练模型的长距离检索能力，passkey NLL-gap 100% 是 EVQ+YaRN 的标志性结果。

## 4. QuALITY (下游评估)

Phase 21b 使用 QuALITY 长文档理解数据集进行下游评估。

```python
from datasets import load_dataset

ds = load_dataset("emozilla/quality", split="validation")
```

- **来源**: HuggingFace Hub `emozilla/quality`
- **样本量**: n=2086 (validation split)
- **评估指标**: Gold NLL (非准确率，因 454M 处于容量地板)
- **实现**: `scripts/core_text_phases/phase21b_quality_eval_clean.py`

## 5. 预 tokenized 数据 (.pt 文件)

部分实验使用预先 tokenized 的 `.pt` 数据文件以加速训练。这些文件不包含在版本控制中（体积过大），但可通过以下方式重新生成：

```bash
# 生成 FineWeb-Edu tokenized 数据
python scripts/data_prep/prepare_mixed_prior_dataset_v1.py \
    --dataset fineweb-edu \
    --seq_len 2048 \
    --total_tokens 2000000000 \
    --output_dir ./data/

# 生成合成数据
python scripts/data_prep/tokenize_synth.py
```

输出格式: `D_{dataset}_{tokens}_{seq_len}.pt`，存放在 `results/core_text/` 目录下。

---

## 数据集与论文映射

| 数据集 | 论文 Table/Figure | 实验 Phase |
|--------|------------------|-----------|
| FineWeb-Edu | Tables 1-6, Figs 2-7 | Phase 8-17 |
| TinyStories | Table 1 (50M validation) | Phase 8 |
| Passkey Mix | Tables 2-3, Fig 2 | Phase 14+ |
| QuALITY | Fig 5 | Phase 21b |
