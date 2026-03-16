# EVQ-Cosh AI Agent Handoff — Complete Specification

> **如果你是新接手的 AI agent，按此顺序阅读：**
> 1. 本文件 Part 1-3 (架构 + 超参数 + 禁止事项) → 确保配置正确，避免 GPU 浪费
> 2. `docs/overview/PAPER_CLAIMS_MAP.md` → 论文↔实验↔脚本完整映射
> 3. `paper/main.tex` → 当前论文 LaTeX 源码
> 4. `team/status/WORKFLOW_AND_PAPER_GAPS.md` → P0-P3 优先级矩阵

---

## Part 1: 模型架构规范（禁止偏离）

所有实验必须使用以下精确配置。**任何偏离都会导致结果不可比，浪费 GPU 时间。**

### 架构表

| Tier | Params | Layers | Heads | head_dim | Hidden | FFN | Vocab | 来源 |
|------|--------|--------|-------|----------|--------|-----|-------|------|
| 50M  | ~50M   | 6      | 8     | 64       | 512    | 2048| 50304 | `run_evq_sweep.py:75-89` |
| 125M | ~125M  | 12     | 12    | 64       | 768    | 3072| 50304 | `run_evq_sweep.py:90-104` |
| 350M | ~350M  | 24     | 16    | 64       | 1024   | 4096| 50304 | `run_evq_sweep.py:105-119` |
| 454M | ~454M  | 24     | 16    | 64       | 1024   | 4096| 50304 | `phase21b_quality_eval_clean.py:40-47` |
| 500M | ~500M  | 28     | 16    | 64       | 1024   | 4096| 50304 | `run_evq_sweep.py:120-134` |
| 750M | ~750M  | 18     | 24    | 64       | 1536   | 6144| 50304 | `phase21b_quality_eval_clean.py:48-56` |

⚠️ **关键注意**: 750M 是 **18 层 / 24 头**，不是 24 层 / 16 头！论文 Appendix B.1 中此处有错误需修正。

⚠️ **454M vs 350M**: 454M 使用与 350M 完全相同的架构配置（24L/16H/1024/4096）。参数量差异来自不同的训练设置，实际上共享同一 GPT 类配置。

### 训练超参数

从 `scripts/core_text_phases/run_evq_sweep.py` 精确提取：

| Tier | LR | Batch Size | Seq Len | Total Tokens | Warmup |
|------|----|-----------|---------|-------------|--------|
| 50M  | 6e-4 | 32 | 2048 | 50M | min(200, total_steps//10) |
| 125M | 3e-4 | 16 | 2048 | 100M | min(200, total_steps//10) |
| 350M | 2e-4 | 2 | 2048 | 100M | min(200, total_steps//10) |
| 500M | 1.5e-4 | 4 | 2048 | 500M | min(200, total_steps//10) |

所有 Tier 共用：
- **Optimizer**: AdamW, β1=0.9, β2=0.95, weight_decay=0.1
- **LR Schedule**: Cosine decay to min_lr (通常 1e-5)
- **Tokenizer**: `EleutherAI/gpt-neox-20b` (vocab=50304)
- **Dataset**: FineWeb-Edu (HuggingFace streaming)
- **DTYPE**: bfloat16 (CUDA), float32 (MPS/CPU)
- **RoPE base**: 500,000

### Continued Pretrain 超参数（Phase 11e/15/17 系列）

| 参数 | 值 | 备注 |
|------|---|------|
| LR | 1e-5 | Finetuning/继续预训练用 |
| Warmup | 500 steps | |
| Weight decay | 0.1 | |
| Micro batch | 2 | |
| Grad accum | 5 | Effective batch = 10 |
| 参考脚本 | `phase21b_scrolls_finetune.py` | |

---

## Part 2: EVQ-Cosh 公式规范

### 核心公式

```
φ_k(τ) = 1 - (1/τ) × arcsinh((1 - u_k) × sinh(τ))

其中:
  u_k = (k + 0.5) / K    (midpoint quantization, 论文 Formula 9)
  k = 0, 1, ..., K-1
  K = head_dim / 2 = 32   (当 head_dim=64)

inv_freq_k = base^(-φ_k(τ))
  base = 500,000 (默认)
```

### τ* 最优值公式

```
τ* = d_head / √L_train

示例:
  d_head=64, L_train=2048 → τ* = 64/√2048 ≈ 1.414
  d_head=64, L_train=256  → τ* = 64/√256  = 4.0
  d_head=64, L_train=512  → τ* = 64/√512  ≈ 2.828
```

### 参考实现（规范版本）

文件: `scripts/core_text_phases/run_evq_sweep.py` 第 141-157 行

```python
def evq_cosh_inv_freq(head_dim: int, tau: float, base: float = 500000.0) -> torch.Tensor:
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)          # ⚠️ 必须 float64
    u = (idx + 0.5) / float(K)                           # midpoint quantization
    if abs(tau) < 1e-8:
        phi = u                                           # geometric limit (Theorem 2)
    else:
        sinh_tau = math.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
    inv = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    return inv.float()                                    # 最后转回 float32
```

### 数值稳定性要点

1. **必须用 float64 计算 inv_freq**，低频通道可达 1e-6，float32 精度不够
2. **τ ≤ 1e-8 时回退到 geometric**（Taylor 展开，避免除零）
3. `math.sinh(tau)` 用 Python float64，`torch.arcsinh` 用 tensor float64
4. 最终 `inv.float()` 转回 float32 供模型使用

### 其他实现位置

| 文件 | 用途 | 备注 |
|------|------|------|
| `scripts/lib/rope/schedules.py:180-190` | 通用 RoPE schedule builder | u_k = k/n（无 midpoint），τ 默认 0.5 |
| `scripts/core_text_phases/run_evq_sweep.py:141-157` | **规范版本** | u_k = (k+0.5)/K（有 midpoint），与论文一致 |
| `scripts/lib/rope/learnable_evq.py:95-113` | 可学习 EVQ | Taylor 展开版本 |

⚠️ **schedules.py 与 run_evq_sweep.py 的 u_k 定义不同！** 论文使用 midpoint 版本，以 run_evq_sweep.py 为准。

---

## Part 3: 禁止事项（违反将浪费 GPU 时间）

### ❌ 绝对禁止

**1. ❌ 不要用 NTK-aware YaRN**
- NTK-aware 对所有频率通道施加相同 scale factor: `factor = scale^(dim/(dim-2))`
- 这会破坏 EVQ 精心设计的频率结构
- ✅ 必须用 **Progressive YaRN**: per-channel smoothstep ramp，高频保护，低频缩放
- 参考实现: `phase14c_multiscale_evq_yarn.py` (channel-index ramp, start=0.20×K, end=0.90×K)
- 已知 bug 文件: `phase21b_quality_eval.py` 使用了 NTK-aware ❌

**2. ❌ 不要在已训练的 Geo checkpoint 上直接换 EVQ inv_freq**
- Retrofit（事后替换）不等于从头训练，attention 权重已适配 geometric 频率
- ✅ 必须从头训练（run_evq_sweep.py）或使用 continued-pretrain 流程（phase11e）
- Continued-pretrain: 从 Geo checkpoint 出发，以较低 LR 继续训练时切换 inv_freq

**3. ❌ 不要用 τ=0.707**
- 这是早期错误值，τ=0.707 时 EVQ ≈ Geo，看不出差异
- ✅ 用 τ* = d_head/√L_train 计算，常用值: τ=1.414 (L=2048), τ=4.0 (L=256)

**4. ❌ 不要用 torch.compile(mode="max-autotune") 做首次运行**
- max-autotune 编译时间极长，首次应确认逻辑正确
- ✅ 先用 mode="default" 验证，确认无误后再切换

**5. ❌ eval 时不要改 seq_len 而忘记同步 YaRN scale**
- YaRN scale factor = eval_length / train_length
- 如果训练 L=2048，eval 在 L=16384，则 scale = 8
- ✅ 始终保持 scale = eval_len / train_len

**6. ❌ 不要混用 tokenizer**
- 所有实验统一: `EleutherAI/gpt-neox-20b`，vocab_size=50304
- ✅ 不要用 GPT-2 tokenizer (vocab=50257) 或其他

**7. ❌ 不要在纯文本训练后期望 passkey 100%**
- Passkey 100% 需要在训练中混入 5-10% passkey 数据
- ✅ Phase 14+ 实验使用 passkey mix: `generate_passkey_sample()` 在 run_evq_sweep.py 中

**8. ❌ 不要期望 454M 在 QuALITY 上出现准确率差异**
- 454M 处于 QuALITY 容量地板: ~25% ≈ 随机基线
- EVQ vs Geo 的差异被随机噪声淹没
- ✅ 用 Gold NLL（-30.1% @8K）或 PPL 做指标，不用准确率

**9. ❌ 不要照搬 FIRE 的 25K 步训练量**
- FIRE 使用可学习 PE，25K 步是学习频率所需
- EVQ 是闭式解，不需要学习频率
- ✅ 理解方法论，不要复制超参数

### Blackwell GPU (RTX 5090 / RTX 6000 Pro) 特殊注意

| 要求 | 设置 |
|------|------|
| PyTorch | ≥ 2.7.0, 推荐 2.8+ |
| CUDA | 12.8 (sm_120 required) |
| torch.compile | **必须开启**，否则 GPU 利用率极低 |
| 每步必须调用 | `torch.compiler.cudagraph_mark_step_begin()` |
| 内存配置 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

性能参考 (454M, L=512, RTX 5090 32GB):

| 模式 | ms/step | tok/s | VRAM | ETA (2B tokens) |
|------|---------|-------|------|-----------------|
| eager | 231 | 44K | 25.1GB | 12.6h |
| compile(default) | 165 | 62K | 17.6GB | 8.9h |
| compile(max-autotune) | 183 | 56K | 20.9GB | 9.9h |

### Video DiT 特殊注意 (2026-03-16 新增)

**10. ❌ 不要用跨 run 比较做 DiT 实验结论**
- CUDA 非确定性在 DiT 上可以制造 70%+ 的虚假差异
- 此前 "129.6M GEO 赢 71%" 完全是跨 run 噪声
- ✅ 必须用 **head-to-head**（同一 run 内训练两种方法）
- 参考: `results/video_dit/REPORT_FINAL.md` (v2) Part II

**11. ❌ 不要对 DiT 使用 τ*_AR = K/√T**
- AR 最优 τ=2.83 对 DiT 失败（中频被过度抽空 → 位置指纹崩坏）
- ✅ DiT 使用 τ*_DiT ≈ 0.53 × K_t/√T_train
- 对 K_t=16, T=32: τ*_DiT ≈ 1.5（而非 2.83）
- 已验证: h2h τ=1.5 赢 GEO -21%/-35%

**12. ❌ 不要使用 Power-Shift 族**
- φ_k(α) = 1-(1-u_k)^(1+α) 在理论上有吸引力但实践中灾难性失败
- α=0.25 比 GEO 差 22x，α=0.50 差 6x
- ✅ DiT 仍使用 EVQ-Cosh 族，只是 τ 不同

**13. ❌ 不要忽略 base=10000 的死通道效应**
- base=10000 + K_t=16 + T_train=32 产生"死通道"（θ_k × Δ ≈ 0）
- 这导致 τ 曲线呈离散相变（τ=1.2 差 2.8x，τ=1.5 赢 21%）
- ✅ 如果做 DiT 实验需注意 base_t 选择，小 base_t（~100-1000）可能消除此效应

---

## Part 4: 新实验检查清单

每次启动新实验前，必须逐项确认：

- [ ] **τ 值**: τ = d_head / √L_train，L_train 是当前阶段训练长度
- [ ] **d_head**: 确认是 per-head dimension (64)，不是 hidden_size
- [ ] **YaRN 实现**: Progressive (per-channel ramp)? 不是 NTK-aware (uniform)?
- [ ] **YaRN scale**: scale = eval_length / train_length
- [ ] **Checkpoint 来源**: 是 progressive EVQ checkpoint 还是 retrofit?
- [ ] **eval 指标**: NLL / accuracy / retrieval? 样本量是否充分?
- [ ] **eval 模式**: standard 还是 distractor-padded? 与 baseline 一致?
- [ ] **train/eval 一致性**: YaRN / loss masking / tokenizer 在训练和评估中相同?
- [ ] **数值精度**: inv_freq 是否用 float64 计算?
- [ ] **架构配置**: 是否与上面的架构表完全一致?

### 实验报告模板

每个实验完成后，在 `docs/exp/` 创建报告，命名: `YYYY-MM-DD_slug.md`

```markdown
# [Phase XX] 实验名称

## 配置
- Tier: 350M (24L/16H/1024)
- τ: 1.414 (d_head=64, L_train=2048)
- RoPE base: 500000
- YaRN: Progressive, scale=8
- Seeds: 42, 123, 7
- Tokens: 100M FineWeb-Edu + 5% passkey mix

## 结果
[表格]

## 与论文关联
- 支撑 Claim: C3 (EVQ+YaRN synergy)
- 对应 Figure/Table: Figure 2, Table 2

## 异常与注意事项
[如有]
```

---

## Part 5: 项目状态与证据层级

### 论文三锚点 (Narrative Lock, 2026-03-12)

1. **Closed-form theory**: RoPE 频率分配是变分逆问题的闭式解，geometric RoPE 是 τ→0 退化极限
2. **Extreme extrapolation**: EVQ 在 DAPE-style 极端外推中匹敌或超越可学习 PE
3. **Systems result**: EVQ + Progressive YaRN >> Geo + YaRN (100% vs 61-65% passkey @8×)

### 证据层级

| 级别 | 内容 | 状态 |
|------|------|------|
| **P0** (主锚) | fig2 EVQ+YaRN, fig3 PE-dominant, passkey mix 3-seed | ✅ 完成 |
| **P0.5** | Phase 21 downstream NLL (750M, 13 LongBench tasks) | ✅ 完成 |
| **P1** (强支撑) | Phase 15 750M continue, Phase 9f 750M baseline | ✅ 完成 (single-seed) |
| **P1.5** (新) | Video DiT h2h: τ=1.5 赢 GEO -21%/-35% (129.6M) | ✅ 完成 (h2h validated) |
| **P2** (扩展) | Video AR temporal, cross-model (Llama/Qwen) | ✅ 完成 |
| **P2.5** (新) | DiT τ*_DiT ≈ 0.53 × τ*_AR 架构缩放律 | 🔄 进行中 (需 fine-grained sweep) |

### 单点风险

| 风险 | 严重程度 | 缓解 |
|------|---------|------|
| C4: 454M Stage 2-3 仅 seeds 42-44, single-config | ⚠️ HIGH | 需补充 multi-config |
| C5: 750M continued-pretrain 仅 single-seed | ⚠️ HIGH | 标注为 supporting evidence |
| LongBench 下载失败 | MEDIUM | 已用 NLL 替代 accuracy |

### 关键文件索引

| 用途 | 文件 |
|------|------|
| 论文 LaTeX | `paper/main.tex` |
| 核心叙事线 | `internal/mainstory.md` |
| 论文↔实验映射 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| P0-P3 优先级矩阵 | `team/status/WORKFLOW_AND_PAPER_GAPS.md` |
| τ-sweep 主脚本 | `scripts/core_text_phases/run_evq_sweep.py` |
| RoPE schedule 库 | `scripts/lib/rope/schedules.py` |
| 实验报告目录 | `docs/exp/` |
| 复现指南 | `docs/overview/REPRODUCE.md` |
| Video DiT 报告 (v2) | `results/video_dit/REPORT_FINAL.md` |
| Video DiT 理论 | `DiT_frequency_allocation_analysis.md` (repo root) |
| Video AR 精度 | `results/supporting_video/temporal_precision_report.md` |
| Video DiT 脚本 | `scripts/video_temporal/run_dit_temporal.py` |
