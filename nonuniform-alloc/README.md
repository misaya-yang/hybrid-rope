# nonuniform-alloc — 成熟模型上非几何 RoPE 分配的可行域

**主文档：`RESEARCH_MEMO.md`。** `RESEARCH_MEMO_v1_superseded.md` 是被推翻的第一版，保留作记录。

## 三句话

1. **表是绑定约束，不是覆盖。** `SELECTIVE_QK_PHASE_ADAPTATION` 已经做过覆盖匹配对照：
   Native 表不动 + 满相位曝光 300 步，8K RULER 仍只有 **2.02%**，EVQ 同协议 **31.63%**。
   这是你们最强的归因证据，现在只是个 supporting 行。
2. **窗口内代价不是笼统的，它精确落在"多源分辨"上**：单源族平均 −16.67 pp，
   多源族 −40.96 pp（差 24.3 pp），`fwe` 甚至 +10。regime II（`L<λ≤M`，OLMo-2 只有 10 个通道）
   是唯一给单调无歧义距离分辨的带，EVQ 把它压到 0.77×。
3. **窗口内外兼得，在单一共享表下做不到**（三个频谱区都在干活，没有免费预算；
   cosh 是一个旋钮调三个自由度）。能做到的只有 R2（门控残差，窗口内逐位等于 Native）；
   R1（长度路由）是今天就能用的工程近似，你们已经量过 `65.16/25.50/5.48`。

## 脚本（全部零 GPU、纯几何，`python3 <f>` 复算）

| 脚本 | 产出 | 章节 |
|---|---|---|
| `spectral_regimes.py` | 五个真实模型的三区通道计数；EVQ 从哪搬到哪 | §3 |
| `multikey_pattern.py` | **多源 vs 单源的窗口内损失分解（−40.96 vs −16.67 pp）** | §2 |
| `waterbed_lemma.py` | ρ_τ 的三区质量表；最小标尺 λ≥2X | §4 |
| `check_claims.py` | φ_c(τ)、regime II 落在减薄侧的充分条件、FMRoPE 三种 base | §4 |
| `refutation_check.py` | **对抗性复核：BPB 买不到标尺、\|Δsin\| 3.3×、慢通道的 1.05 logit recency 核、base=256 无 dead 带** | §0 |
| `budgeted_phase_bump.py` | ε 汇率表（其结论已被 `refutation_check.py` 推翻，保留作记录） | §0 |
| `cap_design.py` | CAP(G,M,ν) 参数化与各设计的碰撞分数/最小 log 间距 | §0 |
| `coverage_generalization.py` | 覆盖 G 下的 phase-complete 通道数 | §3 |
| `tau_star_vs_coverage.py` | **失败的尝试，保留作记录**：effective rank 当效用求 τ*(G)，无信号 | §7 |

依赖 numpy ≥ 2.0（用了 `np.trapezoid`）。

## 只做三件事

1. 把 `SELECTIVE_QK_PHASE_ADAPTATION` 提到正文当归因证据（零成本）
2. 把多源模式在 8B 的 13 族数据上复核并写进论文（零成本）
3. 跑 E1：leave-one-pair-out KL，**分开报"位置重要性"和"内容重要性"**（分钟级）
