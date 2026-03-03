# Phase 13: Downstream Task PPL — 验证 EVQ 预训练模型的长文本理解能力

> **核心目标**: 用 Teacher-forcing PPL 在真实下游任务数据上对比 Geo vs Hybrid checkpoint，证明 EVQ 频率分配让模型更好地利用长距离 context
> **方法**: 零 finetune，直接用预训练 checkpoint 计算下游任务的 answer-token PPL
> **硬件**: R6000 96GB（Phase9F 已完成，复用同一台机器）
> **预计 GPU 时间**: ~30min（纯推理，无训练）
> **论文价值**: 回应 reviewer "只看 PPL/passkey 不够" 的质疑，且完全避免 instruction tuning 的 confound

---

## 0. 背景与动机

### 0.1 为什么不做 LoRA finetune + 生成评测？

750M from-scratch 模型没有 instruction following 能力。如果先做 LoRA instruction tuning 再评测，有两个严重问题：

1. **混合数据配比是个大坑**：instruction tuning 数据集的选择、配比、格式都会影响结果，引入大量 confound
2. **归因不清**：如果 Hybrid LoRA 赢了，是因为预训练好还是因为 LoRA 恰好更适配？

**正确做法**：直接用预训练 checkpoint（零 finetune），在下游任务数据上算 answer-token PPL。

**核心逻辑**：给模型 `[long context] + [question] + [answer]`，只计算 answer 部分的 PPL。如果 Hybrid 的 answer PPL 更低，说明 EVQ 预训练让模型更好地"理解"了长距离 context 来预测 answer——这就是频率分配的直接价值。

### 0.2 这个方法的优势

| 对比项 | LoRA + 生成 F1 | Teacher-forcing PPL（本方案）|
|--------|---------------|---------------------------|
| 需要 instruction tuning？ | 需要（大坑） | **不需要** |
| 受生成策略影响？ | 是 | **否** |
| 指标连续性 | F1 离散 | **PPL 连续，统计效力强** |
| 归因清晰度 | 混杂 LoRA 效果 | **纯粹反映预训练质量** |
| 计算成本 | ~2-4h | **~30min** |
| 可复现性 | 依赖训练超参 | **完全确定性** |

### 0.3 可用的 Checkpoint

R6000 路径：`/root/autodl-tmp/evq_phase9/seed42/`

| Checkpoint | 路径 | 说明 |
|-----------|------|------|
| Geo 100% | `geo_750m_2k_1bdata_ckpt/model.pt` | Geometric baseline |
| Hybrid 100% | `hybrid1.5_r16_750m_2k_1bdata_ckpt/model.pt` | EVQ-Cosh τ=1.5 r=16 |

两者架构完全相同（750M GPT, H=1536, L=18, heads=24, head_dim=64），只有 inv_freq 不同。

---

## 1. 服务器信息

- SSH: `sshpass -p 'htG0sD63/yG0' ssh -o StrictHostKeyChecking=no -p 23173 root@connect.bjb1.seetacloud.com`
- GPU: RTX PRO 6000 Blackwell 96GB
- 代码: `/root/autodl-tmp/dfrope/hybrid-rope/scripts/m4_evq_sweep`
- Checkpoint: `/root/autodl-tmp/evq_phase9/seed42/`

## 2. 环境准备

```bash
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# datasets 和 transformers 应该已经装了，确认一下
pip install datasets transformers --break-system-packages
```

---

## 3. 评测方法：Answer-Token Teacher-Forcing PPL

### 3.1 原理

```
输入序列:  [context tokens] [question tokens] [answer tokens]
标签:      [-100 ... -100]  [-100 ... -100]   [answer tokens]
                                                ↑ 只在这部分算 loss
```

对每个样本：
1. 拼接 `context + question + answer` 为一个完整序列
2. 前向传播得到 logits
3. **只在 answer token 位置计算 cross-entropy loss**（prompt 部分标记为 -100）
4. 汇总所有样本的 loss → 计算 PPL = exp(avg_loss)

PPL 越低 → 模型给定长 context 后，对 answer 的预测越准 → 对长距离 context 的利用越好。

### 3.2 为什么这能区分 Geo vs Hybrid？

- 两个模型在 FineWeb-Edu 上训练了相同的 1B tokens
- 短文本语言建模能力几乎相同（in-distribution PPL 差 <0.15%）
- 差异在于**长距离位置信息的编码质量**
- 下游任务的 answer 往往依赖 context 中距离较远的信息
- 如果 EVQ 的频率分配让位置编码更可区分，answer PPL 应该更低

### 3.3 长度分桶评测

关键设计：对每个任务，按 context 实际长度分桶，分别报告 PPL：

| 长度桶 | token 范围 | 预期 |
|--------|-----------|------|
| Short | 512-2048 | 两者接近（in-distribution） |
| Medium | 2048-4096 | Hybrid 开始领先 |
| Long | 4096-8192 | Hybrid 显著领先 |

这样可以画出 **context length vs answer PPL** 的曲线——如果 Hybrid 曲线在长距离端更平缓，就是 EVQ 的直接证据。

---

## 4. 下游任务选择

### 4.1 核心任务（必做，3个）

| 数据集 | 来源 | 特点 | HF 路径 |
|--------|------|------|---------|
| **NarrativeQA** | LongBench | 长篇叙事理解，context 最长 | `THUDM/LongBench` (narrativeqa) |
| **Qasper** | LongBench | 学术论文 QA，结构化长文本 | `THUDM/LongBench` (qasper) |
| **MultiFieldQA-en** | LongBench | 多领域 QA，context 多样 | `THUDM/LongBench` (multifieldqa_en) |

### 4.2 加分任务（如有余力）

| 数据集 | 来源 | 特点 |
|--------|------|------|
| **HotpotQA** | LongBench | 多跳推理 |
| **GovReport** | LongBench | 长文本摘要 |
| **TriviaQA** | LongBench | 开放域问答 |

---

## 5. 代码实现

### 5.1 新脚本: `phase13_downstream_ppl.py`

```python
#!/usr/bin/env python3
"""
Phase 13: Downstream Answer-Token PPL evaluation.
Compare Geo vs Hybrid pretrained checkpoints on long-context tasks.
No finetuning — pure pretrained model evaluation.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import GPT, DEVICE, DTYPE, USE_AUTOCAST, set_seed


# ============================================================
# 1. 数据加载（LongBench 格式）
# ============================================================

def load_longbench_task(task_name: str, tokenizer, max_len: int = 8192):
    """Load a LongBench task and tokenize into (input, labels) pairs.

    LongBench format: each sample has 'input', 'context', 'answers' fields.
    We concatenate context + question as prompt, answer as target.
    Labels = -100 for prompt, actual token ids for answer portion.
    """
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task_name, split="test")
    print(f"  Loaded {task_name}: {len(ds)} raw samples")

    samples = []
    for item in ds:
        context = item.get("context", "")
        question = item.get("input", "")
        answers = item.get("answers", [])
        if not answers:
            continue

        # Format: [context]\n\nQuestion: [question]\nAnswer: [answer]
        prompt_text = f"{context}\n\nQuestion: {question}\nAnswer:"
        answer_text = f" {answers[0]}"

        # Tokenize separately to know boundary
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

        if len(answer_ids) < 2:
            continue  # skip trivially short answers

        # Truncate prompt from LEFT if too long (keep question at end)
        max_prompt = max_len - len(answer_ids)
        if max_prompt < 128:
            continue
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        full_ids = prompt_ids + answer_ids
        # Labels: -100 for prompt, actual ids for answer
        labels = [-100] * len(prompt_ids) + answer_ids

        samples.append({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt_ids),
            "answer_len": len(answer_ids),
            "total_len": len(full_ids),
        })

    # Sort by length for efficient batching
    samples.sort(key=lambda s: s["total_len"])

    print(f"  Valid samples: {len(samples)}")
    if samples:
        lens = [s["total_len"] for s in samples]
        print(f"  Length stats: min={min(lens)}, median={sorted(lens)[len(lens)//2]}, "
              f"max={max(lens)}, mean={np.mean(lens):.0f}")
    return samples


# ============================================================
# 2. Teacher-Forcing PPL 评测
# ============================================================

def eval_answer_ppl(model, samples, length_bins=None):
    """Compute answer-token PPL, optionally bucketed by context length.

    Args:
        model: GPT model
        samples: list of dicts with 'input_ids', 'labels', 'prompt_len'
        length_bins: list of (name, min_len, max_len) tuples for bucketing

    Returns:
        dict with overall PPL and per-bucket PPL
    """
    if length_bins is None:
        length_bins = [
            ("short",  0,    2048),
            ("medium", 2048, 4096),
            ("long",   4096, 16384),
        ]

    model.eval()

    # Per-bucket accumulators
    bucket_loss = defaultdict(float)
    bucket_tokens = defaultdict(int)
    bucket_samples = defaultdict(int)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            labels = sample["labels"].unsqueeze(0).to(DEVICE)
            seq_len = input_ids.size(1)

            # Extend RoPE if needed
            model.extend_rope(seq_len)

            with torch.cuda.amp.autocast(enabled=USE_AUTOCAST, dtype=DTYPE):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction='sum',
                )

            n_answer_tokens = (labels[:, 1:] != -100).sum().item()
            loss_val = loss.item()

            # Overall
            bucket_loss["all"] += loss_val
            bucket_tokens["all"] += n_answer_tokens
            bucket_samples["all"] += 1

            # Per-bucket (based on prompt/context length)
            prompt_len = sample["prompt_len"]
            for bname, bmin, bmax in length_bins:
                if bmin <= prompt_len < bmax:
                    bucket_loss[bname] += loss_val
                    bucket_tokens[bname] += n_answer_tokens
                    bucket_samples[bname] += 1
                    break

            if (i + 1) % 50 == 0:
                running_ppl = math.exp(bucket_loss["all"] / max(bucket_tokens["all"], 1))
                print(f"    [{i+1}/{len(samples)}] running PPL={running_ppl:.2f}")

    # Compute PPL per bucket
    results = {}
    for key in ["all"] + [b[0] for b in length_bins]:
        if bucket_tokens[key] > 0:
            avg_loss = bucket_loss[key] / bucket_tokens[key]
            ppl = math.exp(avg_loss)
            results[key] = {
                "ppl": round(ppl, 4),
                "avg_loss": round(avg_loss, 6),
                "n_tokens": bucket_tokens[key],
                "n_samples": bucket_samples[key],
            }
        else:
            results[key] = {"ppl": None, "n_samples": 0}

    return results


# ============================================================
# 3. 模型加载
# ============================================================

def load_model_from_checkpoint(ckpt_path: str, inv_freq_path: str, cfg: dict):
    """Load pretrained GPT model from Phase9F checkpoint."""
    inv_freq = torch.from_numpy(np.load(inv_freq_path)).float()
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


# ============================================================
# 4. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13: Downstream Answer-Token PPL")
    parser.add_argument("--tasks", type=str, default="narrativeqa,qasper,multifieldqa_en",
                        help="Comma-separated LongBench task names")
    parser.add_argument("--max_len", type=int, default=8192,
                        help="Max sequence length for tokenization")
    parser.add_argument("--work_dir", type=str,
                        default="/root/autodl-tmp/evq_phase13_downstream")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/root/autodl-tmp/evq_phase9/seed42")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    tasks = [t.strip() for t in args.tasks.split(",")]

    # Tokenizer (same as pretraining)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 750M config — max_position_embeddings set to max_len for RoPE extension
    cfg = dict(
        vocab_size=50304,
        hidden_size=1536,
        num_layers=18,
        num_heads=24,
        head_dim=64,
        intermediate_size=6144,
        max_position_embeddings=args.max_len,
        seq_len=2048,
    )

    # Checkpoints to compare
    methods = {
        "geo": {
            "ckpt": str(ckpt_dir / "geo_750m_2k_1bdata_ckpt" / "model.pt"),
            "inv_freq": str(ckpt_dir / "geo_750m_2k_1bdata_ckpt" / "inv_freq.npy"),
        },
        "hybrid": {
            "ckpt": str(ckpt_dir / "hybrid1.5_r16_750m_2k_1bdata_ckpt" / "model.pt"),
            "inv_freq": str(ckpt_dir / "hybrid1.5_r16_750m_2k_1bdata_ckpt" / "inv_freq.npy"),
        },
    }

    # Length bins for analysis
    length_bins = [
        ("short",  0,    2048),   # in-distribution
        ("medium", 2048, 4096),   # mild extrapolation
        ("long",   4096, 16384),  # strong extrapolation
    ]

    all_results = {}

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"  TASK: {task}")
        print(f"{'='*70}")

        # Load data once per task
        samples = load_longbench_task(task, tokenizer, max_len=args.max_len)
        if not samples:
            print(f"  [WARN] No valid samples for {task}, skipping")
            continue

        for method_name, paths in methods.items():
            print(f"\n  --- {method_name.upper()} ---")
            result_key = f"{method_name}_{task}"
            result_file = work_dir / f"{result_key}.json"

            if result_file.exists():
                print(f"  [skip] {result_file} already exists")
                all_results[result_key] = json.loads(result_file.read_text())
                continue

            t0 = time.time()
            model = load_model_from_checkpoint(paths["ckpt"], paths["inv_freq"], cfg)
            ppl_results = eval_answer_ppl(model, samples, length_bins)
            elapsed = time.time() - t0

            result = {
                "method": method_name,
                "task": task,
                "max_len": args.max_len,
                "n_samples": len(samples),
                "elapsed_sec": round(elapsed, 1),
                "ppl_by_bucket": ppl_results,
            }
            result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            all_results[result_key] = result

            del model
            torch.cuda.empty_cache()

            # Print per-bucket results
            print(f"\n  Results for {method_name} on {task}:")
            for bname in ["all", "short", "medium", "long"]:
                r = ppl_results.get(bname, {})
                if r.get("ppl") is not None:
                    print(f"    {bname:>8}: PPL={r['ppl']:.2f}  "
                          f"(n={r['n_samples']}, tokens={r['n_tokens']})")

    # ============================================================
    # Summary comparison table
    # ============================================================
    summary_file = work_dir / "phase13_summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print(f"  PHASE 13 SUMMARY — Answer-Token PPL")
    print(f"{'='*70}")

    for task in tasks:
        geo_key = f"geo_{task}"
        hyb_key = f"hybrid_{task}"
        geo_r = all_results.get(geo_key, {}).get("ppl_by_bucket", {})
        hyb_r = all_results.get(hyb_key, {}).get("ppl_by_bucket", {})

        print(f"\n  Task: {task}")
        print(f"  {'Bucket':>8} | {'Geo PPL':>10} | {'Hybrid PPL':>10} | {'Δ%':>8} | {'Winner':>8}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

        for bname in ["all", "short", "medium", "long"]:
            g = geo_r.get(bname, {})
            h = hyb_r.get(bname, {})
            g_ppl = g.get("ppl")
            h_ppl = h.get("ppl")
            if g_ppl and h_ppl:
                delta = (h_ppl - g_ppl) / g_ppl * 100
                winner = "Hybrid" if h_ppl < g_ppl else "Geo"
                print(f"  {bname:>8} | {g_ppl:>10.2f} | {h_ppl:>10.2f} | "
                      f"{delta:>+7.1f}% | {winner:>8}")
            else:
                print(f"  {bname:>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8}")

    print(f"\n  Full results: {summary_file}")


if __name__ == "__main__":
    main()
```

---

## 6. 执行命令

### Step 1: 环境确认

```bash
ssh -p 23173 root@connect.bjb1.seetacloud.com
# 密码: htG0sD63/yG0

export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

# 确认 checkpoint 存在
ls -la /root/autodl-tmp/evq_phase9/seed42/geo_750m_2k_1bdata_ckpt/{model.pt,inv_freq.npy}
ls -la /root/autodl-tmp/evq_phase9/seed42/hybrid1.5_r16_750m_2k_1bdata_ckpt/{model.pt,inv_freq.npy}

# 确认 datasets 已安装
python -c "from datasets import load_dataset; print('OK')"
```

### Step 2: 运行全部 3 个任务

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope/scripts/m4_evq_sweep
mkdir -p /root/autodl-tmp/evq_phase13_downstream

nohup python -u phase13_downstream_ppl.py \
    --tasks narrativeqa,qasper,multifieldqa_en \
    --max_len 8192 \
    --work_dir /root/autodl-tmp/evq_phase13_downstream \
    > /root/autodl-tmp/evq_phase13_downstream/phase13.log 2>&1 &

tail -f /root/autodl-tmp/evq_phase13_downstream/phase13.log
```

**预计耗时**：3 任务 × 2 方法 × ~3min/方法 ≈ **~20min 总计**

### Step 3（可选）: 扩展到更多任务

```bash
nohup python -u phase13_downstream_ppl.py \
    --tasks hotpotqa,triviaqa,gov_report \
    --max_len 8192 \
    --work_dir /root/autodl-tmp/evq_phase13_downstream \
    > /root/autodl-tmp/evq_phase13_downstream/phase13_extra.log 2>&1 &
```

---

## 7. 分析模板

### 7.1 核心输出表

```
Task: narrativeqa
   Bucket |    Geo PPL |  Hybrid PPL |      Δ% |   Winner
   -------+-----------+-------------+---------+---------
      all |      xx.xx |       xx.xx |  +x.x%  |  ???
    short |      xx.xx |       xx.xx |  +x.x%  |  ???
   medium |      xx.xx |       xx.xx |  +x.x%  |  ???
     long |      xx.xx |       xx.xx |  +x.x%  |  ???
```

### 7.2 论文 Figure: Context Length vs Answer PPL

```
Answer PPL
  ^
  |    Geo ----___
  |               \___   ← Geo 在 >2K 后快速上升
  |    Hybrid ------___
  |                    \  ← Hybrid 上升更缓慢
  +---+----+----+----+---> Context Length
     512  2K   4K   8K
```

如果这个趋势成立，就是论文的 Fig.X，直接支持 "EVQ preserves positional distinguishability at long range" 的 claim。

### 7.3 论文 LaTeX

```latex
\begin{table}[t]
\centering
\caption{Answer-token perplexity on LongBench tasks (750M, zero-shot,
no finetuning). Lower is better. EVQ-Cosh pretrained models achieve lower
answer PPL on long-context questions, indicating better utilization of
distant context for answer prediction.}
\begin{tabular}{ll ccc}
\toprule
Task & Pretrain & Short ($\leq$2K) & Medium (2–4K) & Long ($>$4K) \\
\midrule
NarrativeQA & Geometric & ... & ... & ... \\
NarrativeQA & EVQ-Cosh  & ... & \textbf{...} & \textbf{...} \\
\midrule
Qasper & Geometric & ... & ... & ... \\
Qasper & EVQ-Cosh  & ... & \textbf{...} & \textbf{...} \\
\midrule
MultiFieldQA & Geometric & ... & ... & ... \\
MultiFieldQA & EVQ-Cosh  & ... & \textbf{...} & \textbf{...} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 8. 成功标准与决策树

### 场景 A: Hybrid 在 medium/long 桶上 PPL 更低（最佳）

条件: medium 和 long 桶 Hybrid PPL < Geo PPL，short 桶接近

→ **论文 claim**: "EVQ-Cosh pretrained models better utilize long-range context for downstream task predictions, as evidenced by lower answer-token PPL on questions requiring long-distance information retrieval."
→ 新增 Table + Figure（Context Length vs Answer PPL）
→ 直接支持 capability-preserving proposition

### 场景 B: 所有桶 PPL 接近（可接受）

条件: 差距 <2%

→ **论文 claim**: "Frequency reallocation incurs zero downstream cost: pretrained model quality on real tasks is preserved."
→ 一句话提及，重点在 "zero cost"
→ 结合 passkey +40pp 的结果，故事依然成立

### 场景 C: Hybrid 全面更差（需分析）

条件: Hybrid PPL > Geo PPL 在所有桶

→ 与 Hybrid OOD PPL +5.7% 的趋势一致
→ 说明 Hybrid(r=16) 的 OOD 泛化确实更差
→ **这反而加强 Phase 12 的动机**：需要找到最优 r* 来平衡
→ 可以在 Phase 12 完成后，用最优 r 重新训练再评

---

## 9. 风险与对策

| 风险 | 对策 |
|------|------|
| LongBench 数据集下载失败（GFW） | `HF_ENDPOINT=https://hf-mirror.com` |
| 某些任务 context 超过 8K tokens | `max_len=8192` 截断，从左侧截（保留 question） |
| GPT-2 tokenizer 与 LongBench 格式 | LongBench 是英文数据，GPT-2 tokenizer 完全兼容 |
| inv_freq.npy 不存在 | 从 model.pt 提取：`state["blocks.0.attn.rope.inv_freq"]` |
| 长序列 OOM | 逐样本推理（batch_size=1），96GB 足够处理 8K |
| long 桶样本太少 | 打印每个桶的 n_samples，如果 <10 则合并桶或降低分界 |

---

## 10. ⚠️ Claude Code 执行注意事项

1. **这是纯推理任务**：不需要安装 peft、不需要训练、不需要梯度
2. **先确认 checkpoint 文件存在**：`model.pt` 和 `inv_freq.npy`
3. **如果 inv_freq.npy 不存在**：
   ```python
   state = torch.load("model.pt", map_location="cpu")
   inv_freq = state["blocks.0.attn.rope.inv_freq"]
   np.save("inv_freq.npy", inv_freq.numpy())
   ```
4. **Tokenizer**: GPT-2（与预训练一致）
5. **不要动现有 checkpoint**：只读取
6. **结果保存到** `/root/autodl-tmp/evq_phase13_downstream/`
7. **每个 (method, task) 组合保存独立 JSON**：方便断点续跑
8. **关注 long 桶的样本量**：如果 <10 个样本，统计不可靠，需要在报告中标注

---

## 11. 后续规划

Phase 13 是**低成本快速验证**。根据结果：

- 如果 Hybrid 赢 → 直接写入论文，加强 "practical impact" 叙事
- 如果两者持平 → "zero cost" 也是好结果，一句话带过
- 如果需要更强结果 → **等 poster 90% 确认后**，开 1.5B+ 模型做完整 instruction tuning + 生成评测（Phase 14+，预算另算）
