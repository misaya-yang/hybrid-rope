# Data Preparation — 数据预处理工具

将原始数据集转换为训练所需的预标记化 (pre-tokenized) 格式。

---

## 脚本清单

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `prepare_mixed_prior_dataset_v1.py` | FineWeb-Edu 预标记化 + Passkey 混合 | HuggingFace streaming | `.pt` 文件 (memmap) |
| `tokenize_synth.py` | 合成数据 (TinyStories) 标记化 | HuggingFace streaming | `.pt` 文件 (memmap) |
| `prepare_longbench_local_data.py` | LongBench QA 子集下载 + 格式化 | HuggingFace Hub | `.jsonl` 文件 |

---

## 数据流

```
HuggingFace Hub
    ↓
prepare_mixed_prior_dataset_v1.py
    ↓
train_fineweb-edu_{tokens}_{seq_len}.pt    ← 核心训练数据
    ↓
scripts/core_text_phases/run_evq_sweep.py  ← 被实验脚本读取
```

## 关键参数

- **Tokenizer**: `EleutherAI/gpt-neox-20b` (vocab_size=50304)
- **FineWeb-Edu**: streaming 模式, 无需预下载, 自动从 HuggingFace 获取
- **Passkey 比例**: 可配置 (默认 3-10%, 用于 passkey retrieval 训练)
- **预标记化格式**: PyTorch memmap `.pt` 文件, 包含 `(num_tokens,)` int64 张量

## 使用示例

```bash
# 准备 2B tokens, seq_len=2048 的训练数据
python scripts/data_prep/prepare_mixed_prior_dataset_v1.py \
    --tokens 2000000000 --seq_len 2048 --output_dir /data/evq/

# 准备 TinyStories 数据
python scripts/data_prep/tokenize_synth.py --output_dir /data/evq/synth/
```

## 参考

- 完整数据源文档: `docs/overview/DATA_PREPARATION.md`
- 复现指南: `docs/overview/REPRODUCE.md`
