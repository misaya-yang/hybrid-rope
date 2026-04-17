# EVQ-Cosh Competitive Positioning & Base-10K Experiment Plan

> **日期**: 2026-04-09
> **背景**: 基于对 GPT-5.4/Gemini/Qwen3/LLaMA4/DeepSeek-V3 的 1M+ 上下文技术路线调研
> **优先级**: P1（写作部分可立即执行）+ P2（base=10K 实验需 GPU，2-3 周内完成）

---

## 一、核心发现：EVQ 精准击中了一个无人区

### 1.1 工业界长上下文技术路线全景

所有主流厂商冲 1M+ 上下文的技术，**无一例外都在 RoPE 三要素的前两个轴上做文章**：

| 厂商 | 方法 | 第一轴 (base freq) | 第二轴 (inference rescaling) | 第三轴 (频率分配) |
|------|------|:-:|:-:|:-:|
| DeepSeek-V3/V3.1 | YaRN 两阶段 | ✗ (base 不变) | ✅ YaRN s=2→4 | ❌ geometric |
| Qwen2.5-1M / Qwen3 | ABF + YaRN + DCA | ✅ 10K→10M | ✅ YaRN + DCA | ❌ geometric |
| Meta LLaMA 4 Scout | iRoPE (NoPE 层 interleave) | ✗ | ✅ chunked attention | ❌ geometric (RoPE 层) |
| Google Gemini 1.5/2.0 | 未公开（ring attention + MoE） | 未知 | 未知 | 未知 |
| OpenAI GPT-4.1/5.4 | 完全黑盒 | 未知 | 未知 | 未知 |

**结论: 第三轴（频率分配方式）是一个确认的空白区。EVQ-Cosh 是第一个将其 formalize 为变分问题并给出 closed-form 解的工作。**

### 1.2 学术界最近相关工作

| 论文 | 会议 | 与 EVQ 关系 |
|------|------|-------------|
| "Round and Round We Go" | ICLR 2025 | 证实不同 RoPE 频率 channel 功能不同（positional heads vs semantic），**隐含支持 EVQ 的核心论点** |
| LongRoPE2 (Microsoft) | arXiv 2025.02 | Inference-time per-channel rescaling，与 EVQ 的 training-time allocation 正交互补 |
| VideoRoPE | ICML 2025 Oral | Low-frequency Temporal Allocation，精神呼应但是 hand-crafted heuristic 而非 variational |
| Resonance RoPE | 2024 | Critical frequency 对齐，互补方向 |
| RIFLEx | 2025 | Video DiT inference-time frequency correction，论文已 cite |

---

## 二、论文写作改动建议（立即可执行，不需 GPU）

### 2.1 Related Work 补充

在 §2 Related Work 中加入以下内容:

**LLaMA 4 iRoPE**:
```
iRoPE~\citep{llama4} interleaves RoPE layers with NoPE (no positional encoding)
layers, offloading global context to position-free attention. In the layers that
retain RoPE, the geometric allocation is unchanged; EVQ is applicable to these
layers and may be particularly beneficial since they bear the full positional
encoding burden with fewer opportunities for cross-layer correction.
```

**"Round and Round We Go" 作为 independent empirical support**:
```
\citet{roundandround2025} provide independent empirical evidence that RoPE frequency
channels serve functionally distinct roles: highest frequencies construct positional
attention patterns while lowest frequencies carry semantic information. This
functional heterogeneity is precisely the structure that geometric allocation ignores
and EVQ exploits.
```

### 2.2 Intro/Conclusion 中强化 orthogonal composition narrative

在 Intro 的 "Why this matters now" 段落或 Conclusion 中加入一句:

```
EVQ is orthogonal to all existing context-extension methods: Adjusted Base Frequency
(Qwen), YaRN (DeepSeek/Qwen/LLaMA), Dual Chunk Attention (Qwen), and iRoPE
(LLaMA 4) all operate on the first two RoPE axes (base frequency and inference-time
rescaling). EVQ addresses the third axis (training-time frequency layout), and
Table~\ref{tab:evq-yarn} confirms that the corrections compose additively rather
than competing.
```

### 2.3 MLA 段落强化 (针对 DeepSeek-V2/V3 趋势)

在 §5 MLA supporting paragraph 或 Conclusion 加入:

```
The MLA trend---DeepSeek-V2/V3 compress RoPE to 64 channels, our 432M configuration
uses 16---makes principled frequency allocation increasingly consequential. As
channels become scarce, each channel's placement is decisive: EVQ alone outperforms
Geo+YaRN at 16K, a result that cannot be obtained by inference-time methods alone.
```

---

## 三、Base=10K MLA 实验计划（P2，需 GPU，2-3 周内完成）

### 3.1 动机

当前 MLA 实验使用 $b=500K$。但 production DeepSeek-V2/V3 使用 $b=10K$。Reviewer 可能会问：

> "在 $b=10K$ 的 production 设置下，EVQ 的优势是否存在？"

Surrogate validation (Table A.2) 显示 $b=10K$ 时 collision reduction = 45%（vs $b=500K$ 的 24-92%），说明优势会缩小但不会消失。但缺少直接的 training 实验。

### 3.2 实验配置

```python
# MLA 432M @ base=10K，复用现有 MLA 架构
model_config = {
    "scale": "432M",
    "architecture": "MLA",
    "d_rope": 32,         # 16 frequency channels
    "d_head": 128,
    "kv_rank": 256,
    "base": 10_000,       # ★ 改为 production DeepSeek 设置
    "L_train": 8192,
    "tau": 1.414,         # d_head / sqrt(L) = 128/sqrt(8192)
    "dataset": "FineWeb-Edu",
    "tokens": "500M",
    "seeds": [42, 43, 88],  # 复用现有 seed 配置
}
```

### 3.3 Eval 矩阵

| 方法 | PPL@8K | PPL@16K | PPL@24K | PPL@32K |
|------|--------|---------|---------|---------|
| GEO (b=10K) | | | | |
| GEO+YaRN(s=4) (b=10K) | | | | |
| EVQ (b=10K) | | | | |
| EVQ+YaRN(s=4) (b=10K) | | | | |

### 3.4 预期结果 & 叙事

**乐观情况** (70% 概率): EVQ@b=10K 仍然优于 Geo@b=10K，gain 缩小到 15-20%（vs b=500K 的 31.1%），但 EVQ+YaRN 仍然 > Geo+YaRN。

**叙事**: "EVQ's advantage scales with the dead-channel bottleneck severity. At $b=10K$, fewer channels are dead (surrogate predicts $-45\%$ collision reduction vs $-92\%$ at $b=500K$), and the empirical gain scales proportionally. The method remains beneficial but with reduced headroom, consistent with the theoretical prediction."

**保守情况** (25% 概率): EVQ@b=10K 的 raw advantage 很小 (<5%)，但 EVQ+YaRN > Geo+YaRN 仍然成立。

**叙事**: "At $b=10K$ with 16 channels, the geometric allocation already activates most channels (fewer dead channels), reducing EVQ's direct contribution. However, EVQ+YaRN composition still outperforms Geo+YaRN, confirming that even modest frequency redistribution amplifies inference-time methods."

**最坏情况** (5% 概率): EVQ@b=10K 没有优势。

**叙事**: 写入 Limitations，"At low base values where the geometric allocation already activates all channels, EVQ's redistribution provides diminishing returns." 这反而是理论的验证——dead-channel mechanism 预测了这一点。

### 3.5 GPU 预算

- 训练: 432M × 500M tokens × 2 (GEO + EVQ) × 3 seeds = 6 runs
- 每 run ~6h A100 → 总计 ~36h
- Eval: ~4h total
- **总计: ~40h A100**

### 3.6 脚本基础

```bash
# 基于现有 MLA 实验脚本修改
ls scripts/core_text_phases/*mla* scripts/core_text_phases/*phase18*
# 需要修改的参数: rope_base 从 500000 改为 10000
```

---

## 四、EVQ + LongRoPE2 组合实验（P2-1，已在 README 中列出，可选）

### 4.1 理论动机

EVQ 纠正 shape（训练时频率密度），LongRoPE2 纠正 per-channel scale（推理时 evolutionary search）。三层分解：

1. **Shape (EVQ)**: 训练时确定最优频率密度 $\rho(\phi)$
2. **Range (YaRN)**: 推理时 coarse range correction
3. **Per-channel fine-tuning (LongRoPE2)**: 推理时每个 channel 独立缩放

### 4.2 实验设计：2×2 矩阵

| | Geo allocation | EVQ allocation |
|---|---|---|
| **YaRN** | Geo+YaRN ✅ (已有) | EVQ+YaRN ✅ (已有) |
| **LongRoPE2** | Geo+LongRoPE2 (新) | **EVQ+LongRoPE2 (新)** |

### 4.3 预期结论

如果 EVQ+LongRoPE2 >> Geo+LongRoPE2 的 margin ≈ EVQ+YaRN >> Geo+YaRN 的 margin：

→ **Composability 是 EVQ 的 general property，不是与 YaRN 的 specific interaction。**

这直接封堵 "EVQ+YaRN synergy is cherry-picked" 攻击。

### 4.4 实现复杂度

- LongRoPE2 代码需要单独实现 evolutionary search（约 1-2 天工程量）
- 可在 frozen checkpoint 上跑，不需要重新训练
- **建议优先级低于 base=10K 实验**，但如果 progressive 3-seed 顺利完成且有余力，可以做

---

## 五、References 更新（立即可执行）

在 `refs/references.bib` 中补充/确认以下 entries:

```bibtex
@article{roundandround2025,
  title={Round and Round We Go! What Makes Rotary Positional Encodings Useful?},
  author={Barbero, Federico and Banino, Andrea and Kapturowski, Steven and Kumaran, Dharshan and Perez-Nieves, Nicolas and Viri{\'c}, Petar},
  journal={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{longrope2,
  title={LongRoPE2: Near-Lossless LLM Context Window Scaling},
  author={Shang, Yuchen and ...},
  journal={arXiv preprint arXiv:2502.20082},
  year={2025}
}

% LLaMA 4 — 需要确认正式 citation format
@misc{llama4,
  title={Llama 4: Open Multimodal Intelligence},
  author={Meta AI},
  year={2025},
  howpublished={\url{https://ai.meta.com/blog/llama-4-multimodal-intelligence/}}
}
```

---

## 六、与现有冲刺计划的对齐

| 本文档建议 | 对应 16_ 冲刺建议中的编号 | 优先级调整 |
|-----------|--------------------------|-----------|
| §2 Related Work 补充 | 新增（纯写作，0 GPU） | **P0-2 附属，Week 1 同步做** |
| Intro/Conclusion orthogonal narrative | 类似 P0-3 | Week 1 写作线 |
| MLA 强化段落 | 类似 P1-5 | Week 1 写作线 |
| Base=10K MLA 实验 | 新增 | **P1 级别，Week 2-3 GPU 线** |
| EVQ+LongRoPE2 | = P2-1 | 不变，保持 P2 |
| References 更新 | Week 3 final checklist 一部分 | 立即可做 |

---

*本文档基于 2026-04-09 对 GPT-5.4/Gemini/Qwen3/LLaMA4/DeepSeek-V3 技术路线的调研，以及论文全文 + internal/2026_04_run/ 全部文档的阅读。*
