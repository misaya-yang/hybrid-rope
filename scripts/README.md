# Scripts — 实验与工具代码

本目录包含 EVQ-Cosh 项目的所有实验脚本、出图脚本、数据准备工具和 RoPE 库。

---

## 目录结构

```
scripts/
├── train.py                    核心训练入口 (from-scratch + continued-pretrain)
├── core_text_phases/           Phase 8–21 主实验链 ⭐
│   ├── README.md               Phase Map + → Paper 映射
│   ├── run_evq_sweep.py        核心 τ-sweep 实验 (50M/125M/350M)
│   ├── phase11_L256_extrap.py  PE-dominant regime
│   ├── phase11b_125m_dape.py   EVQ vs DAPE 对比
│   ├── phase11c_454m_scaling.py  454M scaling
│   ├── phase14c_multiscale_evq_yarn.py  EVQ+YaRN synergy ⭐
│   ├── phase15_750m_*.py       750M continued-pretrain
│   ├── phase16_formula_optimality_sweep.py  99-run τ* validation
│   ├── phase17b_*.py           454M Stage 2 (512→1024)
│   ├── phase17c_*.py           454M Stage 3 (1024→2048) ⭐
│   ├── phase21b_quality_eval_clean.py  QuALITY downstream eval
│   ├── evq_analysis.py         τ-sweep 分析 + waterbed 绘图
│   └── visualize_attention_distance.py  Attention 可视化
├── figures/                    论文图表生成
│   ├── fig1_neurips.py         Fig 1: Frequency dynamics
│   ├── fig2_evq_yarn_orthogonality.py  Fig 2: EVQ×YaRN synergy
│   └── fig3_pe_dominant_scaling.py  Fig 3: PE-dominant scaling
├── data_prep/                  数据预处理
│   ├── prepare_mixed_prior_dataset_v1.py  FineWeb-Edu tokenization
│   └── tokenize_synth.py       合成数据 tokenization
├── supporting_eval/            辅助评估工具
├── lib/rope/                   RoPE 实现库
│   ├── schedules.py            EVQ-Cosh + Geometric 频率计算 + Progressive YaRN
│   └── inject.py               RoPE 注入到 transformer
├── video_temporal/             视频时序外推 (supporting)
├── mac_train/                  M4 Max 本地实验 (legacy)
└── m4_max_36gb/                M4 Max 36GB 实验
```

---

## 核心脚本 → 论文映射

| 脚本 | 论文 Figure/Table | 描述 |
|------|------------------|------|
| `run_evq_sweep.py` | Table 1 | 多尺度 τ-sweep (50M/125M/350M) |
| `phase14c_multiscale_evq_yarn.py` | Table 2-3, Fig 2 | EVQ+YaRN synergy (passkey 100%) |
| `phase11_L256_extrap.py` | Table 4-5, Fig 3 | PE-dominant regime |
| `phase11b_125m_dape.py` | Table 4 | EVQ vs DAPE 对比 |
| `phase11c_454m_scaling.py` | Table 4 | 454M PE-dominant scaling |
| `phase15_750m_*.py` | Table 6 | 750M continued-pretrain |
| `phase16_formula_optimality_sweep.py` | Fig 6 | 99-run τ* formula validation |
| `phase17c_*.py` | Fig 4 | 454M flagship (2K→48K) |
| `phase21b_quality_eval_clean.py` | Fig 5 | QuALITY downstream eval |
| `fig1_neurips.py` | Fig 1, Fig 7 | Frequency dynamics + waterbed |
| `fig2_evq_yarn_orthogonality.py` | Fig 2 | EVQ×YaRN orthogonal synergy |
| `fig3_pe_dominant_scaling.py` | Fig 3 | PE-dominant scaling law |

> 完整的 Figure/Table → Script → Data → Results 追溯地图见 `docs/overview/PAPER_CLAIMS_MAP.md`。

---

## 快速开始

### 环境

```bash
conda create -n evq python=3.10 && conda activate evq
pip install -r requirements.txt
```

### 运行核心实验

```bash
# 50M τ-sweep (~4 小时, 任意 GPU)
python scripts/core_text_phases/run_evq_sweep.py --tier 50m --seeds 42

# 125M τ-sweep (~8 小时, 16GB+ GPU)
python scripts/core_text_phases/run_evq_sweep.py --tier 125m --seeds 42,123,7

# 重新生成论文图表
python scripts/figures/fig1_neurips.py
python scripts/figures/fig2_evq_yarn_orthogonality.py
python scripts/figures/fig3_pe_dominant_scaling.py
```

### RoPE 库使用

```python
from scripts.lib.rope.schedules import evq_cosh_inv_freq

# 计算 EVQ-Cosh 频率 (head_dim=64, τ=1.4)
inv_freq = evq_cosh_inv_freq(head_dim=64, tau=1.4, base=500000.0)

# τ=0.0 等价于 geometric RoPE
inv_freq_geo = evq_cosh_inv_freq(head_dim=64, tau=0.0)
```

---

## 维护规则

- 脚本必须能追溯到论文 Figure/Table 或下一步实验计划，否则不应留在本目录
- 新实验脚本放入 `core_text_phases/`，命名为 `phase{N}_{desc}.py`
- 出图脚本放入 `figures/`，命名为 `fig{N}_{desc}.py`
- 结果输出到 `results/core_text/phase{N}/`
- 实验报告写入 `docs/exp/YYYY-MM-DD_slug.md`
