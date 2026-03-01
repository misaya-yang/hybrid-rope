# NeurIPS 2026 Submission: Reproducible Code & Data

> **Title**: RoPE Scaling as a Variational Inverse Problem: Exact Frequency Allocation and the Waterbed Trade-off
> **Status**: Preparing for submission
> **Deadline**: ~2026年5月中旬

---

## 文件夹结构

```
submission/
├── README.md              ← 本文件
├── code/                  ← 可复现的实验代码（最小集）
│   ├── rope_schedules.py  ← EVQ-Cosh + 所有 baseline 的频率 schedule
│   ├── train.py           ← From-scratch 训练脚本
│   ├── eval_ppl.py        ← PPL 评估
│   ├── eval_passkey.py    ← Passkey retrieval 评估
│   ├── run_evq_sweep.py   ← 一键 τ-sweep 实验
│   └── requirements.txt   ← 依赖
├── data/                  ← 数据说明（不含数据本身）
│   └── DATA_README.md
├── results/               ← 论文中引用的全部实验数据（JSON/CSV）
│   ├── table2_from_scratch_scaling.csv
│   ├── table3_llama8b_longbench.json
│   ├── table4_qwen7b_aggregate.json
│   └── table5_qwen7b_task_family.json
├── figures/               ← 论文中的图
│   └── fig1_evq_warp_curves.pdf
└── paper/                 ← LaTeX 源文件
    └── hybrid_rope_neurips.tex
```

## 复现指南

### 1. From-scratch EVQ τ-sweep (Table 2)
```bash
cd code/
python run_evq_sweep.py --tier 50m --tau-list 0.0,0.4,0.8,1.0,1.5,2.0 --seed 42
python run_evq_sweep.py --tier 125m --tau-list 0.0,1.5 --seed 42,137
python run_evq_sweep.py --tier 500m --tau-list 0.0,1.5 --seed 42
```

### 2. Passkey evaluation
```bash
python eval_passkey.py --checkpoint <path> --lengths 2048,4096,8192,16384
```

### 3. 8B LoRA controlled protocol (Tables 3-5)
见 code/ 中的 LoRA 训练和评估脚本说明。
