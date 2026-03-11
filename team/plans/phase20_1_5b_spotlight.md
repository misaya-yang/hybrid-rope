# Phase 20: 1.5B Spotlight Experiment Suite

## 目标

Push paper from strong poster → spotlight / oral / best paper. Three pillars:

1. **Progressive training at 1.5B scale** — proves EVQ works with modern training pipelines
2. **MLA d_head=64 regime** — proves EVQ is more valuable when frequency channels are scarce
3. **Video cross-modal generalization** — proves EVQ is a unified theory, not NLP-only

---

## 1.5B 模型配置

```
Model: GPT-2 style (same architecture as existing 50M-750M configs)
vocab_size: 50304
hidden_size: 1536
num_layers: 32
num_heads: 24
head_dim: 64 (= 1536/24, matches MLA regime)
intermediate_size: 6144 (4×hidden)
Total params: ~1.53B
```

### Memory Estimate on H800 (80GB)

- **Model (bf16)**: ~3.1GB
- **Optimizer states (AdamW m+v)**: ~12.4GB
- **Activations (batch 4, seq 2K)**: ~15GB
- **Working buffer**: ~8GB
- **Total**: ~51GB (safe on single H800)
- **At seq 8K with batch 1**: ~72GB (tight but feasible)

---

## Experiment A: 1.5B Progressive Training (核心实验)

### 设计

- **3 stages**: L=2K → L=4K → L=8K
- **2 methods**: Geo (τ=0) vs EVQ (τ*)
- **3 seeds**: 42, 137, 256
- **Total**: 18 training runs

### Stage 详细配置

| Stage | L_train | Tokens | τ_evq | micro_bs | grad_accum | eff_bs |
|-------|---------|--------|-------|----------|------------|--------|
| 0 | 2048 | 1.0B | 1.414 | 4 | 2 | 8 |
| 1 | 4096 | 1.5B | 1.000 | 2 | 4 | 8 |
| 2 | 8192 | 1.0B | 0.707 | 1 | 8 | 8 |

**τ formula**: τ* = d_head / √L_train = 64/√L

### 数据

- **FineWeb-Edu streaming** (same pipeline as existing phases)
- Each stage sees **FRESH tokens** (not rechunked from previous stage)
- **Total**: 3.5B tokens across 3 stages

### Evaluation at each checkpoint (25%/50%/75%/100%)

- **PPL curves**: {2K, 4K, 8K, 16K, 32K, 48K, 64K}
- **Passkey retrieval**: depths {0.1, 0.3, 0.5, 0.7, 0.9} × lengths {2K, 4K, 8K, 16K}
- **Multi-needle**: {2, 4, 8} needles × {4K, 8K, 16K}
- **YaRN overlay**: scale={1,2,4,8} for detecting phase transition

### Final Stage Evaluation (after Stage 2)

- **Position-wise PPL**: 20 position bins × {2K, 4K, 8K, 16K, 32K}
- **Cross-domain PPL**: FineWeb-Edu, SlimPajama, Proof-Pile, C4
- **LongBench NLL**: 12 tasks
- **NIAH heatmap**: 2D (context_length × depth)

### 预期结果

- **EVQ advantage amplifies**: ~25% @16K (Stage 0) → ~50% (Stage 1) → ~80%+ (Stage 2)
- **YaRN phase transition** at Stage 1 or 2: EVQ raw > EVQ+YaRN
- **Training-inference equivalence**: evq_stage(i)+yarn ≈ evq_stage(i+1) raw
- **Position-wise PPL** reveals gains concentrated in mid-sequence positions

### Failure Modes

- **OOM at Stage 2** (L=8K): reduce to 1.2B model or use gradient checkpointing
- **No amplification**: increase token budget per stage
- **YaRN phase transition absent**: model capacity threshold not met → report as "requires sufficient scale"

---

## Experiment B: MLA d_head=64 Regime

Since the 1.5B model already uses d_head=64, Experiment A simultaneously validates the MLA regime.

### Additional Comparison (eval-only, no extra training)

- Build 1.5B variant with **d_head=128** (hidden=1536, heads=12)
- Load weights via interpolation from d_head=64 checkpoint
- Compare: EVQ@64 vs Geo@64 vs Geo@128
- **Expected**: EVQ advantage is larger at d_head=64 (fewer channels = allocation matters more)

---

## Experiment C: Video Cross-Modal Generalization

### C1: Enhanced Bouncing Ball (immediate, zero cost)

- **Upgrade** existing `run_video_temporal.py`
- **Multi-ball** (2-3 balls), variable speed, occlusion
- **Larger model**: 12L, 512h, d_head=64 (~85M params)
- **3 seeds** × {τ=0, 0.5, 1.0, 1.5, 2.0, 3.0}
- **Metrics**: PPL + FVD (I3D features on generated frame sequences)
- **Temporal extrapolation**: train@16 → eval@{16,32,48,64,96,128}

### C2: Real Video DiT (main cross-modal experiment)

- **Model**: Latte-small (~300M) or Open-Sora-Plan mini
- **Replace ONLY temporal RoPE** with EVQ (spatial = geometric, controlled variable)
- **Dataset**: UCF-101 (13k videos, 101 action classes) or WebVid-2M subset
- **Training**: 10-20K steps per config
- **τ sweep**: {0.0, 0.5, 0.7, 1.0, 1.5, 2.0}
- **Seeds**: 2-3
- **Metrics**: FVD (I3D), FID (per-frame Inception), temporal consistency score

### 预期结果

- **EVQ improves FVD** by 3-8% at optimal τ
- **Optimal τ for video** follows same d_head/√L formula
- **Temporal consistency improves** (smoother motion)

### Failure Modes

- **Video model doesn't use RoPE** in temporal dim → need custom 3D RoPE injection
- **FVD not sensitive enough** → use per-frame MSE as backup metric

---

## Experiment D: Enhanced Downstream Evaluation

### D1: Position-wise PPL Analysis

- **20 position bins** across context length
- Reveals **WHERE** EVQ gains concentrate
- **Visualization**: line plot (position → Δ_NLL between EVQ and Geo)

### D2: Cross-domain PPL

- Same **1.5B model**, 4 corpora: FineWeb-Edu, SlimPajama, Proof-Pile, C4
- **Eval lengths**: {2K, 4K, 8K}
- Proves domain-agnostic advantage

### D3: LongBench NLL

- **12 tasks** at 1.5B scale
- **30 samples** per task, 3 seeds
- **Context**: 4K-100K range

### D4: NIAH Multi-frame (video-specific)

- Adapt passkey retrieval for temporal video:
- Insert "target frame" at various temporal depths
- Test recall accuracy at {16, 32, 64, 128} frames

---

## 时间线 & GPU 分配 (4×H800)

| Week | GPU 1 | GPU 2 | GPU 3 | GPU 4 |
|------|-------|-------|-------|-------|
| 1 | Exp A S0 (geo 3seed) | Exp A S0 (evq 3seed) | Exp C1 bouncing ball | Phase 18/19 |
| 2 | Exp A S1 (geo cont.) | Exp A S1 (evq cont.) | Exp C2 video DiT | Eval suite |
| 3 | Exp A S2 (geo cont.) | Exp A S2 (evq cont.) | Exp C2 cont. | Downstream eval |
| 4 | Exp B (eval-only) | Final aggregation | Paper figures | Buffer |

**Total**: ~1,200-1,500 GPU-hours over 3-4 weeks.

**Note**: Each GPU runs 3 seeds sequentially (not parallel) to avoid memory issues.
At ~5-7 days per stage, 3 seeds sequential fits within 1 week per GPU.

---

## Checkpoint & Results 结构

```
team/results/phase20/
├── exp_a_progressive/
│   ├── geo_seed42/stage0/ stage1/ stage2/
│   ├── evq_seed42/stage0/ stage1/ stage2/
│   ├── geo_seed137/...
│   ├── evq_seed137/...
│   ├── geo_seed256/...
│   └── evq_seed256/...
├── exp_b_mla_comparison/
│   ├── eval_d64_geo/ eval_d64_evq/ eval_d128_geo/
├── exp_c_video/
│   ├── bouncing_ball_enhanced/
│   └── video_dit/
├── exp_d_downstream/
│   ├── position_wise_ppl/
│   ├── cross_domain_ppl/
│   └── longbench_nll/
└── summary_aggregate.json
```

---

## Scripts 清单

| Script | Purpose | Lines |
|--------|---------|-------|
| `team/scripts/phase20_1_5b_progressive.py` | 1.5B 3-stage training | ~700 |
| `team/scripts/phase20_eval_suite.py` | Comprehensive eval suite | ~500 |
| `team/scripts/phase21_video_dit.py` | Video DiT experiment | ~500 |
| `team/scripts/phase20_downstream.py` | Downstream tasks | ~300 |

---

## Paper Narrative

### Pillar 1 Narrative

"EVQ-Cosh frequency allocation amplifies its advantage through progressive training — the standard paradigm for modern LLM development. At 1.5B scale with 3-stage progressive extension (2K→4K→8K), EVQ achieves X% lower perplexity at 64K context compared to geometric RoPE. Remarkably, after progressive training, EVQ without auxiliary scaling (YaRN) outperforms the combined EVQ+YaRN approach, demonstrating that the model internalizes the optimal frequency allocation."

### Pillar 2 Narrative

"In the MLA regime (d_head=64), where only 32 frequency pairs are available, EVQ's relative advantage increases from Y% to Z% compared to the standard d_head=128 setting. This validates EVQ's industrial relevance: modern architectures like DeepSeek-V3, GLM-5, and Kimi compress the RoPE subspace, making principled frequency allocation more critical."

### Pillar 3 Narrative

"The variational framework underlying EVQ makes no assumptions specific to language modeling. We validate this by applying EVQ-Cosh to temporal frequencies in a video diffusion transformer, achieving W% lower FVD with smoother temporal consistency. This establishes EVQ as a unified positional encoding principle across modalities."

---

## 风险评估

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 1.5B OOM at L=8K | 30% | High | Gradient checkpointing / reduce to 1.2B |
| No progressive amplification | 10% | Critical | Increase tokens, add more stages |
| Video FVD not improving | 25% | Medium | Fall back to temporal PPL/consistency |
| YaRN phase transition absent | 15% | Medium | Reframe as "complementary" story |
| LongBench data fetch failure | 20% | Low | Pre-download all datasets |
| Training instability at scale | 15% | High | Reduce LR, increase warmup |

---

**Status**: Ready for Phase 20 launch (2026-03-11)
**Next milestone**: Exp A Stage 0 completion → ~1 week
