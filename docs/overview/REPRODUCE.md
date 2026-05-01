# Reproducibility Guide

> 最后更新: 2026-04-29
> 本文档提供从环境搭建到论文核心结果复现的完整路径。

---

## 环境配置

### 硬件要求

| 路径 | 最低硬件 | 推荐硬件 | 预计时间 |
|------|---------|---------|---------|
| 快速验证 (50M) | 任意 GPU (4GB+) 或 Apple M-series | NVIDIA GPU 8GB+ | ~4 小时 |
| 核心结果 (125M/454M) | NVIDIA GPU 16GB+ | A100/RTX 4090 | ~24+ 小时 |
| 完整复现 (454M/750M) | NVIDIA GPU 24GB+ | A100 80GB / RTX 5090 | ~72 小时 |

### 安装

```bash
conda create -n evq python=3.10 && conda activate evq
pip install -r requirements-lock.txt
```

`requirements-lock.txt` 固定论文校验环境；`requirements.txt` 保留宽松下界，便于适配新机器。若使用审稿 supplementary ZIP，请运行 `python scripts/package_supplement.py`，避免把本地结果转储、cache、日志、历史 runbook 或私有路径打包进去。

对于 Blackwell GPU (RTX 5090/6000), 需要 PyTorch ≥ 2.7.0 + CUDA 12.8。

---

## 路径一: 快速验证 (50M τ-sweep, ~4 小时)

验证 EVQ-cosh 的核心信号: τ 控制频率重分配，预测的 operating τ 落在外推 PPL 的经验优良 basin 内。

```bash
python scripts/core_text_phases/run_evq_sweep.py \
    --tier 50m \
    --taus 0.0,0.5,1.0,1.5,2.0 \
    --seeds 42 \
    --passkey_mix_ratio 0
```

**预期结果:**
- τ=0.0 (midpoint geometric baseline): 16K PPL 远高于 2K PPL
- τ≈1.4 (best observed sweep basin): 16K PPL 显著下降，但 2K PPL 略升 (waterbed trade-off)
- 输出: 默认写入 `results/core_text/<tier>_sweep/` 下的 JSON + 控制台 summary

**论文对应:** appendix multi-scale raw-PPL table (`paper/tables/table1_multiscale_raw_ppl.tex`; compiled table numbers may shift).

---

## 路径二: 核心结果复现与方向性验证

### 2a. Multi-scale τ-sweep (appendix raw-PPL table / core-script validation)

```bash
# 50M, 3-seed
python scripts/core_text_phases/run_evq_sweep.py --tier 50m --seeds 42,123,7 --strict_dataset --passkey_mix_ratio 0

# 125M, 3-seed
python scripts/core_text_phases/run_evq_sweep.py --tier 125m --seeds 42,123,7 --strict_dataset --passkey_mix_ratio 0

# Optional 350M, 2-seed directional check (受 VRAM 限制, batch_size=2)
python scripts/core_text_phases/run_evq_sweep.py --tier 350m --seeds 42,123 --strict_dataset --passkey_mix_ratio 0
```

这些命令复现 core sweep 方向和脚本路径；论文中的 appendix raw-PPL table also includes curated larger-scale/supporting rows, so use `docs/overview/PAPER_CLAIMS_MAP.md` as the stable source map rather than relying on compiled table numbers.

### 2b. EVQ+YaRN matched-scale leverage (supporting multiscale check)

```bash
python scripts/core_text_phases/phase14c_multiscale_evq_yarn.py
```

- 该脚本复核 50M/125M、5% passkey mix 的 supporting multiscale trend；主论文 Table 2 使用 454M、`L_train=2048`、10% passkey mix、3 seeds/config。
- Supplement includes the curated Table 2 provenance at `data/curated/table2_evq_yarn_454m_passkey_10pct.json`; 不要把该 supporting 脚本误写成 Table 2 的完整重跑入口。
- YaRN 在 Geo 和 EVQ 上使用同一个固定 scale `s=8`，这是 matched-scale control，不是 YaRN 调参评估。
- **预期:** EVQ+YaRN 在 8K 长度达到 100% PK retrieval, Geo+YaRN 约 61%。PK/passkey 表示 teacher-forced NLL-gap retrieval，除非明确标为 autoregressive exact match。

### 2c. PE-Dominant Regime (Table 4, Figure 3)

```bash
python scripts/core_text_phases/phase11b_125m_dape.py
python scripts/core_text_phases/phase11_L256_extrap.py
python scripts/core_text_phases/phase11c_454m_scaling.py
```

---

## 路径三: 论文图表生成

图表生成入口如下；Figure 1 is a static paper asset, and some appendix panels require local result JSONs from the corresponding sweeps:

```bash
# Figure 1: Frequency dynamics (appendix)
python scripts/figures/fig1_neurips.py

# Figure 2: EVQ+YaRN synergy
python scripts/figures/fig2_evq_yarn_orthogonality.py

# Figure 3: PE-dominant scaling
python scripts/figures/fig3_pe_dominant_scaling.py
```

Figure 3 panel (a) has a curated JSON fallback at `data/curated/fig3_extreme_128.json`; panels (b,c) require regenerated local `results/core_text/phase11/results_phase11_{raw,yarn}.json` from the Phase 11 sweep and are not prepackaged in the compact supplement.

输出保存至 `paper/figs/`。

完整的 Figure/Table → Script 映射见 `docs/overview/PAPER_CLAIMS_MAP.md`。

---

## 路径四: 完整复现 (需 GPU 机器)

### 4a. τ* Formula Validation (Figure 6, 99-run)

```bash
python scripts/core_text_phases/phase16_formula_optimality_sweep.py
```

99 runs across multiple (τ, L, tier) 组合，验证 τ*=d_head/√L 作为 operating-rule basin selector 的预测精度。

### 4b. 454M Continued Pretrain (Figure 4)

```bash
# Stage 1: 训练 454M baseline @L=256
python scripts/core_text_phases/phase11c_454m_scaling.py

# Stage 2: Continue @L=1024
python scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py

# Stage 3: Continue @L=2048
python scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py
```

### 4c. Downstream Evaluation (Figure 5)

```bash
python scripts/core_text_phases/phase21b_quality_eval_clean.py \
    --model_pt <454m_checkpoint_path> \
    --tier 454m \
    --rope evq --tau 1.5 \
    --target_len 8192 \
    --data_dir <quality_data_dir> \
    --output_dir <output_dir>
```

---

## 数据获取

详见 `docs/overview/DATA_PREPARATION.md`。

所有训练数据通过 HuggingFace Hub streaming 获取，无需预下载:
- FineWeb-Edu: `HuggingFaceFW/fineweb-edu`
- TinyStories: `roneneldan/TinyStories`
- Passkey: 由训练脚本动态生成
- QuALITY: `emozilla/quality`

---

## 论文编译

```bash
cd paper
bash compile_aidemo.sh
```

---

## 常见问题

**Q: MPS (Apple Silicon) 能运行吗?**
A: 可以运行 50M/125M tier，但必须用 float32 (MPS 不支持 bfloat16)。脚本自动检测并切换。

**Q: 没有 GPU 能验证吗?**
A: 50M tier 可在 CPU 上运行，但会很慢 (~24h)。建议至少使用 Colab 免费 GPU。

**Q: 为什么我的 PPL 数字和论文略有不同?**
A: float32 vs bfloat16 会带来小误差 (通常 <1%)。不同 PyTorch 版本的 random 实现也可能有微小差异。

---

## 复现验证清单

完成复现后，使用此清单验证结果的方向性正确:

| 验证项 | 预期结果 | 论文 |
|--------|---------|------|
| ✅ 50M τ-sweep: τ≈1.4 优于 τ=0 的 8K PPL | PPL 下降 10-20% | appendix raw-PPL table |
| ✅ 125M multi-seed: EVQ 外推 PPL 一致性优于 Geo | 3/3 seeds 方向一致 | appendix raw-PPL table |
| ✅ EVQ+YaRN @8K PK retrieval | 100% (vs Geo+YaRN ~61%; teacher-forced NLL-gap retrieval) | main EVQ×YaRN table |
| ✅ PE-dominant 128→8K diagnostic | seed-42 EVQ row lower than Geo/DAPE-style rows; learnable-tau row is 3-seed | main PE-dominant table |
| ✅ Phase 11 L=256 supporting sweep | EVQ PPL 与 YaRN/τ sweep trend 按文中 seed scope 解释 | Fig 3 |
| ✅ τ* = d_head/√L 预测: best observed sweep basin 与预测接近 | Top-3 ranking in ≥ 8/9 configs | Fig 6 |
| ✅ 454M supporting progressive check | EVQ+YaRN 48K PPL < 3.5; Geo+YaRN PPL >> 10 | Fig 4 |
| ✅ QuALITY Gold NLL: 外推时 EVQ 优于 Geo | NLL 差 10-30% (EVQ 更低) | Fig 5 |

---

## 模型架构速查

| Tier | Layers | Heads | d_model | d_head | FFN | Total Params |
|------|--------|-------|---------|--------|-----|-------------|
| 50M | 6 | 8 | 512 | 64 | 2048 | ~50M |
| 125M | 12 | 12 | 768 | 64 | 3072 | ~125M |
| 350M | 24 | 16 | 1024 | 64 | 4096 | ~350M |
| 454M | 24 | 16 | 1024 | 64 | 4096 | ~454M |
| 500M | 28 | 16 | 1024 | 64 | 4096 | ~500M |
| 750M | 18 | 24 | 1536 | 64 | 6144 | ~750M |

> 注: 所有 tier 的 d_head=64 (这使得 τ* = 64/√L 通用)。454M 与 350M 架构相同，区别在于训练数据量和 continued pretrain 设置。
