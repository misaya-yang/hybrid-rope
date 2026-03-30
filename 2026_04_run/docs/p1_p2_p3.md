# P1/P2/P3 实验简报

---

## P1: QuALITY Downstream 多种子

**目的**: 回应 Weakness 3 (downstream evidence single-seed)

**依赖**: 454M continue@4K checkpoint (seed=43, 44)
- 检查是否已有: `ls results/*454m*continue*4K*seed43*` 或类似路径
- 如果没有，需要先跑 454M continue@4K 训练

**步骤**:
1. 确保 3 个 seed 的 454M continue@4K ckpt 都存在
2. 对每个 ckpt 跑 QuALITY eval (n=2086 full test set)
3. 记录: Gold NLL@4K/8K/16K, Accuracy@4K/8K/16K

**脚本**: 找到之前 Phase 21B 的 eval 脚本
```bash
# 大概是这样的路径:
ls scripts/*quality* scripts/*downstream* scripts/*21b*
```

**预估结果模板**:

| Seed | Gold NLL@8K | Accuracy@8K |
|------|-------------|-------------|
| 42 | -30% | +2.2pp |
| 43 | | |
| 44 | | |

---

## P2: τ* Fine Sweep

**目的**: 回应 Question 2 (τ* sensitivity to λ calibration)

**配置**:
- Model: 125M (快速)
- L_train: 512
- base: 500K
- d_head: 64
- τ* = 64/√512 ≈ 2.828

**τ sweep 范围**: [1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.5]

**每个 τ 跑**:
- From-scratch 125M 训练, 50M tokens
- Eval: PPL@4K/8K/16K (raw + YaRN)

**脚本**: 基于 Phase 16 的 τ sweep 脚本
```bash
ls scripts/*tau_sweep* scripts/*phase16*
```

**产出**:
- fig_tau_sensitivity.pdf: PPL@16K vs τ 碗形曲线
- 量化: 碗底 ±X% τ 范围内 PPL 变化 <1%

---

## P3: LongBench 覆盖 (可选)

**目的**: 回应 Question 4 (broader long-context suites)

**依赖**: 750M continue@4K checkpoint

**步骤**:
1. 在 750M ckpt 上跑 LongBench 全集 (17 subtask)
2. 需要 ≥40GB 显存
3. 记录 per-task accuracy/F1

**脚本**:
```bash
ls scripts/*longbench* scripts/*eval_longbench*
```

**预期**: retrieval-sensitive task 上 EVQ 赢; 非 retrieval task 持平或微弱优势
