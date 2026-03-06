# Reviewer Evidence Map (2026-02-24)

## Why this exists
This note maps external high-quality references to concrete experiment actions, so paper claims stay aligned with what NeurIPS reviewers usually challenge.

---

## External references (primary sources)

1. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding  
   - arXiv: https://arxiv.org/abs/2308.14508
2. RULER: What’s the Real Context Size of Your Long-Context Language Models?  
   - arXiv: https://arxiv.org/abs/2404.06654
3. NoLiMa: Long-Context Evaluation Beyond Literal Matching  
   - arXiv: https://arxiv.org/abs/2412.16193
4. NeurIPS paper checklist (official writing/rigor constraints)  
   - https://neurips.cc/Conferences/2025/PaperInformation/PaperChecklist

---

## Reviewer objection -> required evidence

### Objection A: “你的提升可能只是协议或实现差异”
Action:
1. lock paired manifest across methods,
2. keep tokenizer/decode/attn implementation fixed,
3. keep run-level `inv_freq_sha256` and config snapshots.

Pass condition:
- no INVALID run inside final main table.

### Objection B: “统计不显著，可能是随机噪声”
Action:
1. store full `per_sample_scores` (already patched in `eval_longbench.py`),
2. run paired bootstrap/permutation on main comparisons.

Pass condition:
- report CI + p-value explicitly; if non-significant, downgrade claim wording.

### Objection C: “Needle/Passkey 太容易，不能代表真实长推理”
Action:
1. retain LongBench as practical benchmark,
2. add one harder distractor-oriented stress slice (NoLiMa-style pressure),
3. optionally add RULER-style regime stress for context scaling diagnostics.

Pass condition:
- show either robust margin under harder setting or honest failure boundary.

### Objection D: “方法有用性不清楚，算力成本不划算”
Action:
1. add score-vs-cost Pareto plot (GPU-hours or throughput proxy),
2. only compare top-2 methods to control budget.

Pass condition:
- demonstrate at least one Pareto-positive region.

---

## Minimal acceptance-oriented package (budget constrained)

1. **Main table (protocol-locked, 32K)**: baseline/PI/YaRN/hybrid on LongBench+PPL+Needle  
2. **Paired significance**: hybrid vs strongest baseline  
3. **One hard stress test**: distractor density or RULER-style stress slice  
4. **Cost-performance tradeoff**: top-2 methods only

This package is the shortest path to convert “interesting method” into “hard-to-reject evidence”.

