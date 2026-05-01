# Terms & Protocols

> 最后更新: 2026-03-13
> 本文档定义仓库中所有实验的统一术语、度量指标计算方式和命名规范。

## 1. Terminology

| 术语 | 定义 |
|------|------|
| **Geometric / Geo** | 标准 RoPE 频率分配: ω_k = base^(-2k/d)，等价于 EVQ-Cosh 在 τ=0 的退化极限 |
| **EVQ-Cosh** | 本文提出的闭式频率重分配族: φ_k(τ) = 1 - (1/τ)arcsinh((1-u_k)sinh(τ)) |
| **τ (tau)** | EVQ-Cosh 的唯一控制参数，控制频率从 geometric 向均匀分布偏移的程度 |
| **τ*** | Operating-rule 默认值: τ* = d_head / √L_train；用于选择经验优良 basin，不表示全局最优 |
| **Progressive YaRN** | Per-channel smoothstep ramp 的上下文窗口扩展方法，保护高频通道 |
| **NTK-aware YaRN** | Uniform scaling 的 YaRN 变体，本项目中**禁止使用** |
| **Waterbed trade-off** | EVQ 以有界的短程 PPL 代价换取大幅度长程 PPL 改善 |
| **Passkey mix** | 训练数据中混入 5-10% 合成 passkey 检索样本 |
| **Gold NLL** | 对 gold answer token 计算的 NLL，比准确率更灵敏 |
| **Phase XX** | 实验阶段编号 (Phase 8-21)，详见 `docs/overview/PAPER_CLAIMS_MAP.md` |

### 已废弃术语

| 旧术语 | 替代 | 备注 |
|--------|------|------|
| Hybrid / Anchored Hybrid | EVQ-Cosh | 早期命名，2026-02 前的文档可能仍使用 |
| Phase Collision / D(Δ) | Waterbed trade-off | 理论框架已更新 |
| Sigmoid RoPE | EVQ-Cosh | 早期探索方向 |

## 2. Metric Definitions

### 2.1 PPL (Perplexity)

- 数据: FineWeb-Edu validation (streaming)
- 计算: Causal LM cross-entropy, exp(mean NLL)
- 滑动窗口: 连续 chunk, EVAL_CHUNKS=8-10
- 报告长度: PPL@2K, PPL@4K, PPL@8K, PPL@16K

### 2.2 Passkey Retrieval Accuracy

- 合成 5-digit passkey 嵌入随机文本
- Grid: depths=[0.1, 0.2, 0.5, 0.8, 0.9] × lengths=[2K, 4K, 8K, 12K, 16K]
- Trials: 10 per cell
- Metric: Exact match (binary)

### 2.3 Gold NLL

- 数据: QuALITY validation (n=2086)
- 计算: 仅对 gold answer token position 的 NLL
- 优于准确率: 454M 在 QuALITY 上处于容量地板 (~25%)

### 2.4 Seed Robustness

- 标准 seeds: 42, 123, 7
- 核心结论: 必须 ≥3 seeds 的 mean ± std
- 单 seed 结果: 仅作 supporting evidence，标注 "single-seed"

## 3. Naming Conventions

### 实验报告

格式: `YYYY-MM-DD_slug.md`
位置: `docs/exp/`
示例: `2026-03-03_passkey_mix_results.md`

### 结果数据

格式: `results/{bucket}/{phase_or_tier}/`
Buckets: `core_text/`, `theory/`, `supporting_cross_model/`, `supporting_video/`, `legacy/`

### Checkpoint

格式: `{tier}_{rope}_{tau}_{seed}_step{N}.pt`
示例: `350m_evq_tau1.5_seed42_step50000.pt`
注: Checkpoint 不入版本控制 (.gitignore 排除 *.pt)

## 4. Do-Not-Cite Protocol

实验如违反以下任一条件，禁止在论文正面结论中引用:

1. 不符合公平协议 (使用 monkey patch 而非 inv_freq.copy_())
2. 单 seed 且无统计显著性
3. 使用已知 buggy 脚本 (如 `phase21b_quality_eval.py` 的 NTK-aware YaRN)
4. 超参数不一致 (如 base/lr/tokenizer 与同组实验不同)

降级为 Limitations 或 internal 记录，不入正文。

详细废弃列表见 `docs/overview/EXPERIMENT_REGISTRY.md` 的 "Do Not Cite" section。
