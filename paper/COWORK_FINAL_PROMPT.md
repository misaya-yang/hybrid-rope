# EVQ-Cosh 最终修复 — Cowork Prompt

> 将 `---START---` 到 `---END---` 之间的内容完整粘贴给 cowork。

---START---

你是NeurIPS论文修改专家。对 `/Users/yang/projects/hybrid-rope/paper/` 下的论文进行最终一轮精确修复。

# 硬性约束
1. **只改 .tex 和 .bib**，不增删实验，不改表格中任何原始数值
2. 正文 §1-§7 含表图 **≤9 页**（NeurIPS 2026，neurips_2026.sty）
3. 保持匿名 `Anonymous Authors`

# ═══════════════════════════════════════════
# 第一部分：红色警报（6个，不修必被攻击）
# ═══════════════════════════════════════════

## RED-1 Abstract "beats by 35%" 歧义

**当前** (`main.tex` 第44行):
> "A zero-parameter closed-form initialization beats a 32-parameter learned PE by $35\%$ in extreme extrapolation"

**问题**: 35%是EVQ vs Geo baseline (333.7 vs 513.7)。EVQ vs DAPE(32参数)实际是 -26.7% (333.7 vs 455.3)。当前句法让人理解为"比DAPE好35%"。

**修复**: 改为:
> "A zero-parameter closed-form initialization outperforms a 32-parameter learned PE ($333.7$ vs.\ $455.3$ PPL) and reduces Geo baseline by $35\%$ in extreme extrapolation"

同时检查 `sections/05_experiments.tex` §5.3中类似表述 "333.7 vs. 455.3 PPL, -35% relative to Geo"，确保 "-35% relative to Geo" 不紧跟在 "vs 455.3" 后面造成歧义。建议改为：
> "EVQ achieves $333.7$ PPL ($-35\%$ vs.\ Geo baseline $513.7$), surpassing DAPE ($455.3$) with zero extra parameters."

## RED-2 u_k 论文 vs 代码不一致

**当前** (`sections/03_theory.tex` 第73行):
```latex
u_k = \frac{k}{K},
```

**问题**: 论文写 endpoint quantization `k/K`（使u_0=0），但产生所有实验结果的代码使用 midpoint quantization `(k+0.5)/K`。论文第94行还专门强调 "u_0=0 implies φ_0(τ)=0, anchoring the first channel at ω_0=1"——这在实验中并不严格成立。

**修复方案A（推荐）**: 保留当前公式不变（理论推导基于 k/K 更简洁），但在第94行 "Two properties follow..." 段落末尾加一句：
> "In implementation, midpoint quantization $u_k = (2k{+}1)/(2K)$ is used; for $K \geq 16$ the difference in $\phi_k$ is below $1.6\%$ and does not affect any reported result."

**修复方案B**: 将公式改为 midpoint 版本 `u_k = (2k+1)/(2K)`，相应修改 "u_0=0" 的表述为近似锚定。但这会影响几何极限证明的简洁性。

**选方案A。**

## RED-3 "Three orders of magnitude" 夸大

**当前** (`sections/01_intro.tex` 第22行, Contribution 2):
> "spanning three orders of magnitude in training length"

**问题**: L从32(DiT)到8192(MLA) = 256× = 10^2.4，不到3个数量级。

**修复**: 改为:
> "spanning more than two orders of magnitude in training length (from $L{=}32$ to $L{=}8192$)"

## RED-4 添加 Hyperparameter Table

**问题**: 论文 checklist 声称 "specifies all training and test details" = Yes，但 lr, batch size, warmup, optimizer, weight decay, gradient clipping 等基本参数**全部缺失**。这在 NeurIPS 是可直接 reject 的 reproducibility 缺陷。

**修复**: 在 `appendix/a2_experiment_details.tex` 的 "Reproducibility snapshot" 小节后添加一个 hyperparameter table。从仓库的实验脚本中提取实际使用的参数。

表格结构建议：
```latex
\begin{table}[ht]
\caption{Training hyperparameters for the primary experiments.}
\label{tab:hyperparams}
\centering\small
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Value \\
\midrule
Optimizer & AdamW ($\beta_1{=}0.9$, $\beta_2{=}0.95$, $\epsilon{=}10^{-8}$) \\
Learning rate & ... (从脚本中提取) \\
LR schedule & cosine decay to ... \\
Warmup & ... steps \\
Batch size (tokens) & ... \\
Weight decay & ... \\
Gradient clipping & ... \\
Dropout & ... \\
Sequence packing & ... \\
\bottomrule
\end{tabular}
\end{table}
```

请从以下脚本提取实际参数值（读取代码中的default值）：
- `scripts/core_text_phases/run_evq_sweep.py` — 主实验脚本
- `scripts/core_text_phases/phase18_base_generalization_sweep.py` — MLA
- `scripts/video_temporal/video_dit.py` — DiT

如果不同实验使用不同参数，做成multi-column表格。

## RED-5 缺失引用：XPOS + NTK-aware scaling

**问题1**: XPOS (Sun et al., 2022) 是最早对RoPE通道做非均匀处理的工作（exponential decay per channel），Related Work完全没提。

**修复**: 在 `sections/02_related.tex` 段1 "Resonance RoPE" 之前加一句：
> "XPOS~\citep{sun2022xpos} introduces per-channel exponential decay, acknowledging that different frequency bands warrant different treatment; however, it modifies the \emph{magnitude envelope} rather than the frequency allocation itself."

在 `refs/references.bib` 添加：
```bibtex
@inproceedings{sun2022xpos,
  title={A Length-Extrapolatable Transformer},
  author={Sun, Yutao and Dong, Li and Patra, Barun and Ma, Shuming and Huang, Shaohan and Benhaim, Alon and Chaudhary, Vishrav and Song, Xia and Wei, Furu},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```

**问题2**: NTK-aware scaling 在论文中被多次提到但无 citation。

**修复**: 在 `sections/02_related.tex` 段1 或段2 "NTK-aware scaling" 首次出现处加引用。引用 Code Llama 作为第一个正式使用 NTK-aware scaling 的工作：
```bibtex
@article{roziere2023codellama,
  title={Code Llama: Open Foundation Models for Code},
  author={Rozi{\`e}re, Baptiste and others},
  journal={arXiv preprint arXiv:2308.12950},
  year={2023}
}
```

将段1改为 "NTK-aware scaling~\citep{roziere2023codellama}, YaRN~\citep{peng2024yarn}"。

## RED-6 Limitations 遗漏 MLA base + 下游 benchmark

**当前**: `sections/06_limitations.tex` 第3段没有提及 MLA base=500K 与 production base=10K 的差异，也没有提及所有评估均为 PE-diagnostic metrics。

**修复**: 在 §6 第3段 "Full from-scratch training at ≥1B is deferred." 之后追加：
> "The MLA experiment uses $b{=}500\mathrm{K}$; production MLA deployments (DeepSeek-V2/V3) use $b{=}10\mathrm{K}$, and the effect magnitude at lower base values requires additional validation. All evaluations in this work target PE-diagnostic metrics (extrapolation PPL, passkey retrieval, downstream NLL); standard capability benchmarks (MMLU, HumanEval) are not tested, so EVQ's impact on downstream task accuracy beyond positional quality remains open."

# ═══════════════════════════════════════════
# 第二部分：橙色警告（8个，仔细reviewer会发现）
# ═══════════════════════════════════════════

## ORG-1 Table 2 vs Table 3 PPL@16K 不一致

**问题**: Table 2 Geo PPL@16K = 253.2, Table 3 Geo PPL@16K = 262.0。同一模型，PPL@8K两表一致(161.9)但16K差3.5%。

**修复**: 在 Table 3 caption 中添加解释。如果差异来自 per-document averaging vs full-sequence，应解释为什么只在16K出现偏差（可能是sequence boundary effects在更长context下放大）。如果实际是不同evaluation split或不同seed aggregation，需修正caption。至少加一句：
> "PPL@$16\mathrm{K}$ differs from Table~\ref{tab:evq-yarn} due to [具体原因]; this does not affect the relative ranking."

## ORG-2 Table 1 tier标注与实际seeds矛盾

**问题**: §5.1 定义 Robustness = "3-seed"，但 Table 1 的5行中4行是 single-seed。

**修复二选一**:
- (a) 将 Table 1 的 tier 从 "Robustness" 降为 "Supporting"，§5.1 相应调整
- (b) 重新定义 tier：Robustness = "多scale一致性检查，seed数标注于表中"

**推荐(a)**: 在 §5.1 的 tier 定义中改为：
> "\emph{Robustness} (3-seed, single architecture): capability preservation (\S\ref{sec:exp-robust}). \emph{Supporting} (1--3 seed): multi-scale PPL, progressive training, video DiT, LoRA at 7B."

## ORG-3 MLA τ=1.414 中 d_head 取值需解释

**问题**: MLA d_rope=32, 但 τ=128/√8192=1.414 用了 d_head=128（完整 attention head 维度），不是 d_rope=32。论文从未解释这个选择。

**修复**: 在 `appendix/a3_supporting_results.tex` 第8行（或附近）加：
> "The $\tau^*$ formula uses the complete attention head dimension $d_{\mathrm{head}}{=}128$ rather than $d_{\mathrm{rope}}{=}32$: the frequency allocation must respect the full head's capacity, as the non-RoPE dimensions ($d_{\mathrm{nope}}{=}96$) carry content that interacts with the positional channels through the latent projection."

同时修正括号注释。当前第8行可能写 "i.e., 32/√512" 或类似错误推导——改为：
> "EVQ uses $\tau{=}1.414$ ($d_{\mathrm{head}}/\sqrt{L} = 128/\sqrt{8192}$)."

## ORG-4 Self-consistency proof 中间步骤修正

**问题**: `appendix/a1_proofs.tex` 第170行声称 "both satisfy $y'' = \tau^2 y$ subject to $y(0)=0$"，实际应为 $y'' = \tau^2 y - \tau^2\rho(0)$（非齐次方程）。最终结论 correct，但中间声明有误。

**修复**: 第168-171行改为：
> "Define $f(\phi) = \tau^2 g(\phi) - h(\phi)$. Then $f'' = \tau^2 g'' + \rho'' = -\tau^2\rho + \tau^2\rho = 0$, with $f(0)=0$ and $f'(1)=0$. Hence $f \equiv 0$, yielding $\tau^2 g(\phi) = h(\phi) = \rho(0) - \rho(\phi)$."

## ORG-5 λ CV "<0.2% relative error" 不准确

**问题**: d_head=32, L=512: error=0.005, τ*=1.660, relative=0.301% > 0.2%。

**修复**: `appendix/a1_proofs.tex` 和 `tables/table_lambda_cv.tex` caption 中 "$<0.2\%$ relative" 改为 "$<0.4\%$ relative"。

## ORG-6 Table 4 (PE-dominant) seed数未标注

**问题**: Primary anchor 没标 seed count。

**修复**: Table 4 caption 添加 seed 信息。如果 125M/L=128 实验是 3-seed，写 "(3-seed mean)"；如果是 single-seed，必须降级 tier 或补充说明。

## ORG-7 YaRN scale factor 未报告

**问题**: Table 2 的 YaRN 用了什么 scale？是否对 EVQ 和 Geo 分别优化了？

**修复**: 在 Table 2 caption 或 §5.2 正文中加一句：
> "YaRN uses scale $s{=}[具体值]$ for all configurations; the same scale is applied to both Geo and EVQ baselines."

## ORG-8 Balance Corollary 需验证

**问题**: Appendix 声称 $T_1'(\tau) + \tau^2 T_2'(\tau) = 0$ 对所有 τ>0 成立。但从 self-consistency identity 求导会出现额外的 $2\tau T_2$ 项。

**修复**: 请数值验证这个 corollary。如果不成立，要么修正表述（可能是在 variational balance 点成立而非所有 τ），要么删除该 corollary（它不影响论文的核心结果）。

# ═══════════════════════════════════════════
# 第三部分：黄色提示（7个，提升质量）
# ═══════════════════════════════════════════

## YEL-1 "single, wide basin" → "a wide basin"
Abstract 第43行 "a single, wide basin"，删除 "single"（无法证明不存在其他 basin）。

## YEL-2 "increasingly decisive" → "particularly decisive"
Abstract 第45行和 Conclusion 中 "increasingly" 只有一个数据点(MLA 16ch)。改为 "particularly"。

## YEL-3 "27 configurations" 添加分解表
在 Appendix 添加完整 27 行表格，列出 architecture / d_head / L / base / seeds / τ* / evidence tier。

## YEL-4 LoRA +30% in-dist cost 写入 Limitations
在 §6 第2段 "practical deployment path" 后加：
> "The LoRA path incurs $+30\%$ in-distribution PPL (Table~\ref{tab:lora-8b}), limiting its applicability to settings where extrapolation quality is prioritized over in-distribution performance."

## YEL-5 MLA "50% training" -29.0% 缺 supporting data
Appendix A3 声称 "-29.0% at 16K from 50% training" 但无 table/figure。要么添加简表（4-5个 checkpoint 的 PPL），要么将措辞改为 "the advantage is visible from early training"（不给具体数字）。

## YEL-6 Related Work qualify "has remained at geometric default"
段1 最后一句改为：
> "...has received far less systematic attention, with nearly all production systems retaining the geometric default."

## YEL-7 P ∝ 1/687 → ~1/689
`appendix/a1_proofs.tex` 第100行 "$P \propto 1/687$" 改为 "$P \propto 1/689$"（4ln²(500000) = 688.8）。

# ═══════════════════════════════════════════
# 自检清单
# ═══════════════════════════════════════════

完成所有修改后逐项确认：

## 红色（必须全部通过）
- [ ] Abstract "35%" 歧义已消除
- [ ] u_k midpoint implementation note 已添加
- [ ] "three orders of magnitude" 已改为 "more than two"
- [ ] Hyperparameter table 已添加到 Appendix A2
- [ ] XPOS + NTK-aware scaling 已添加引用和讨论
- [ ] Limitations 已补充 MLA base + 下游 benchmark 声明

## 橙色（尽量通过）
- [ ] Table 2 vs Table 3 PPL@16K 差异已解释
- [ ] Table 1 tier 标注已修正
- [ ] MLA d_head=128 已解释
- [ ] Self-consistency proof 已修正
- [ ] λ CV "<0.2%" 已改为 "<0.4%"
- [ ] Table 4 seed数已标注
- [ ] YaRN scale factor 已报告
- [ ] Balance corollary 已验证或修正

## 黄色（尽量通过）
- [ ] "single basin" → "wide basin"
- [ ] "increasingly" → "particularly"
- [ ] 27-config 分解表已添加
- [ ] LoRA +30% 写入 Limitations
- [ ] MLA 50% training 数据已补充或模糊化
- [ ] Related Work "has remained" 已 qualify
- [ ] P ∝ 1/687 → 1/689

## 格式检查
- [ ] 正文 ≤ 9 页
- [ ] 所有 \ref 无 undefined
- [ ] 所有 \citep/\citet 在 bib 中存在
- [ ] 编译无 error

# 原则
- 效果为王：100% vs 61%、-31.1%@10σ、zero-parameter beats 32-parameter — 这些王牌不动
- 理论是 crown jewel：exact 层大力展现，semi-analytic 层诚实+basin 化解
- 每处修改的目标是**消除 reviewer 攻击面**，而非改变论文的核心叙事

---END---
