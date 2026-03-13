# Supporting Evaluation — 辅助评估脚本

独立的评估工具，用于 LongBench、NIAH 等标准化评测。与 `core_text_phases/eval_*.py` 不同，这些脚本支持 HuggingFace 格式模型且包含更完整的 fair-protocol 控制。

---

## 脚本清单

| 脚本 | 功能 | 适用模型 | 输出 |
|------|------|---------|------|
| `eval_longbench.py` | LongBench generation-style 评估 (含 fair-protocol 控制) | HF 格式 | JSON per-task |
| `eval_multi_needle.py` | Multi-needle NIAH 评估 | from-scratch / HF | JSON recall matrix |
| `eval_niah_heatmap.py` | NIAH heatmap 可视化 (depth × length) | HF 格式 | PDF heatmap |
| `eval_niah_recall.py` | NIAH recall-style 评估 (depth-sweep) | HF 格式 | JSON recall curve |
| `eval_passkey_scratch.py` | Passkey 检索 (from-scratch 模型) | from-scratch | JSON accuracy |

---

## 与 core_text_phases/ 评估脚本的区别

| 特性 | `core_text_phases/eval_*.py` | `supporting_eval/eval_*.py` |
|------|---------------------------|---------------------------|
| 模型格式 | from-scratch GPT (自定义) | HuggingFace + from-scratch |
| RoPE 注入 | 通过 `lib/rope/inject.py` | 原生 HF RoPE 或手动注入 |
| Fair-protocol | 基础控制 | 完整控制 (温度/采样/截断) |
| 用途 | 核心实验流水线内 | 独立评估 / 对比实验 |

## 使用场景

- 在非标准模型上运行 NIAH 评估 → `eval_niah_heatmap.py`
- 对 HuggingFace 模型做 LongBench 生成式评估 → `eval_longbench.py`
- 生成 depth × length 的 NIAH 热力图 → `eval_niah_heatmap.py`
