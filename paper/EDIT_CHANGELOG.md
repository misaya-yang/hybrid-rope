# EVQ-Cosh 论文完善 — Cowork 提示词

> **使用方式**: 将 `---START---` 到 `---END---` 之间的全部内容复制粘贴给 cowork agent。

---START---

你是一位NeurIPS论文修改专家。对 `/Users/yang/projects/hybrid-rope/paper/` 下的论文进行一轮系统性完善。

# 总体目标

将论文从 Accept 推向 Strong Accept 竞争力。**只修改 .tex 和 .bib 文件，不增删实验数据、不修改任何表格中的原始数字、不生成新图片**。

# 硬性约束

1. **NeurIPS 2026格式**：`neurips_2026.sty`，`[nonatbib]` 选项。正文（§1到§7含正文表/图）**严格≤9页**。References和Appendix不限页。
2. **不修改任何实验数值**：表格中的原始PPL、passkey、MSE等数字一律不动。只修改文字描述、百分比引述、caption措辞。
3. **匿名**：author 保持 `Anonymous Authors`。

# ═══════════════════════════════════════
# 第一部分：必须修复的硬伤（6个）
# ═══════════════════════════════════════

## BUG-1 [编译错误] 缺失图片 `tau_basin.pdf`

`appendix/a1_proofs.tex` ~第217行：
```latex
\includegraphics[width=0.95\linewidth]{tau_basin.pdf}
```
该文件在 `figs/` 中**不存在**，LaTeX会报错。

**修复**：将该 figure 环境注释掉，将 caption 中的关键定量信息（"±20% τ deviation incurs <1% collision increase; leave-one-out max error <0.2%"）融入前后段落的正文中。删除或改写所有 `\ref{fig:tau-basin}` 引用。

## BUG-2 [数值错误] DiT表格均值：正文 vs footer不一致

文件：`appendix/a2_experiment_details.tex`

从 Table `dit-h2h` 的原始数据正确计算：
- Train MSE: mean(-21%, -32%) = **-26%**
- All extrap: mean(-14%, -16%) = **-15%**
- Far extrap: mean(-35%, -30%) = **-32%**

当前错误：
- 正文（~第82行）写 "−19%" for all-extrap → 应改为 **"−15%"**
- 表格 footer（~第101行）写 "−21% / −15% / −32%" → 应改为 **"−26% / −15% / −32%"**

## BUG-3 [数值错误] MLA τ推导括号注释

文件：`appendix/a3_supporting_results.tex` ~第8行：
> "EVQ uses τ=1.414 (i.e., 32/√512)"

但同段明确写 L_train=8192。32/√8192 = 0.354 ≠ 1.414。
括号内的推导路径有误。

**修复**：删除括号内推导注释，改为：
> "EVQ uses $\tau{=}1.414$, the predicted value for this MLA configuration."

如果你能从论文已有内容确认正确推导路径（可能是 d_head 的不同取值），可以恢复括号注释但须保证数值一致。

## BUG-4 [引用错误] Signal gradient引用了错误的表

文件：`appendix/a4_supporting_experiments.tex` ~第37行：
> "raw PPL −52% (Table~\ref{tab:multiscale})"

Table multiscale 最大改善是 -45.9%（750M行）。"-52%" 来自 progressive training Table（Stage 2）。

**修复**：改为 `"raw PPL $-46\%$ (Table~\ref{tab:multiscale})"` 或改为引用正确的表格。

## BUG-5 [歧义] Abstract中MLA baseline不清

Abstract ~第45行和 intro ~第7行：
> "EVQ alone outperforms Geo+YaRN at 2× extrapolation (−31.1% PPL)"

−31.1% 是 EVQ vs **GEO baseline**（95.6 vs 138.8）。EVQ 确实胜过 Geo+YaRN（95.6 < 117.9），但胜过 Geo+YaRN 的幅度是 -18.9%。读者会自然将括号里的 -31.1% 理解为胜过 Geo+YaRN 的幅度。

**修复**：改为：
> "EVQ alone reduces $2\times$ extrapolation PPL by $31.1\%$ relative to Geo baseline, surpassing even Geo+YaRN"

在 intro 中做同样修改。

## BUG-6 [数值精度] MLA in-distribution cost

Appendix a3 ~第27行写 "+0.9% in-distribution cost"。
从表格四舍五入值：(35.8−35.4)/35.4 = +1.13%。

**修复**：改为 "+1.1%" 或 "approximately +1%"。

# ═══════════════════════════════════════
# 第二部分：引用修复（3个问题）
# ═══════════════════════════════════════

## REF-1 缺失模型/数据集引用

以下模型在正文中被提及但**没有bib条目和citation**。请在 `refs/references.bib` 添加条目，在首次提及处加 `\citep`：

| 需引用的 | 出现位置 | 建议bib key |
|---|---|---|
| DeepSeek-V2 | intro §1, appendix a3, a4 | deepseekv2 |
| DeepSeek-V3 | intro §1, appendix a3 | deepseekv3 |
| LLaMA-2 | appendix a2 Table dead-channels | touvron2023llama2 |
| LLaMA-3 | experiments §5, limitations §6, appendix a4 | grattafiori2024llama3 |
| Qwen2.5 | limitations §6 | qwen2024qwen25 |
| CogVideoX-5B | appendix a2 Table dead-channels | yang2024cogvideox |
| Wan-2.1 | appendix a2 Table dead-channels | wan2025wan |
| HunyuanVideo | appendix a2 Table dead-channels | kong2024hunyuanvideo |
| Open-Sora 1.2 | appendix a2 Table dead-channels | opensora2024 |
| Latte-1 | appendix a2 Table dead-channels | ma2024latte |
| Moving MNIST | appendix a2 | srivastava2015unsupervised |

请查找这些模型的正确arxiv/proceedings引用信息填入bib。如果某个模型找不到确切引用，至少加一个arxiv条目。

## REF-2 ICML条目volume不一致

`ding2024longrope`, `zhao2025riflex`, `shang2025longrope2` 缺 `volume` 字段，而 `jin2024selfextend`（volume=235）和 `videorope2025`（volume=267）有。请统一补全，或统一移除 volume（保持一致即可）。

## REF-3 2025年ICML论文状态

`videorope2025`, `zhao2025riflex`, `shang2025longrope2` 标为 ICML 2025 proceedings。如果投稿时（2026年4月）这些已正式出版，保持 `@inproceedings`。否则改为 `@article{..., journal={arXiv preprint arXiv:XXXX.XXXXX}}`。请自行判断。

# ═══════════════════════════════════════
# 第三部分：行文优化（5个修改）
# ═══════════════════════════════════════

核心策略：论文最大优势是 (a) 完整变分理论+闭合解 (b) 碾压性实验数据（100% vs 61%、-31.1%、-35% vs learnable PE）。行文修改的目标是让这两个优势形成一个**不可忽视的narrative**，同时诚实处理theory gap。

## WRITE-1 Abstract重组：三层递进

当前abstract信息密度高但缺乏层次感。重组为：

**第一句（发现）**：频率分配是被忽视的第三设计轴，geometric RoPE是退化极限。
**第二句（方法）**：闭合解族 + 零参数公式 + 次优by construction但match 27 configs。
**第三句（最强结果 — 用一句话串联所有primary evidence）**：建议类似：
> "A zero-parameter closed-form initialization beats a 32-parameter learned method by $35\%$ in extreme extrapolation, composes with YaRN to unlock $100\%$ retrieval where Geo+YaRN plateaus at $61\%$ (3~seeds), and on MLA with only 16 RoPE channels, alone outperforms Geo+YaRN at $2\times$ extrapolation."
**第四句（意义）**：频率分配axis的重要性随channel压缩递增。

保持总长度与当前abstract相当，不要膨胀。

## WRITE-2 "Sub-optimal by construction"从defensive转offensive

**§3.7**（03_theory.tex ~第138行 "Sub-optimality as robustness" 段落）当前是解释性的。

改写核心句为：
> "The most striking finding is not that EVQ improves over geometric RoPE---which the variational analysis predicts---but that a formula with one unclosed constant matches 27 per-setting optima. This implies a structural fact about the frequency allocation landscape: it contains a single, wide basin around the variational solution. The structural correction---redistributing mass from high to low frequencies---does the work; the precision of the rule that selects the operating point is secondary."

关键转变：从"虽然不完美但够用"→"我们发现了landscape的几何结构，公式只是这个发现的证据"。

## WRITE-3 Theory呈现：扬长避短

(a) **§3开头三层epistemic标注**（~第10-16行）已经很好。在第16行后追加一句：
> "The first layer is the paper's primary theoretical contribution; the third is a practical convenience whose imprecision is bounded by the basin width demonstrated in \S\ref{sec:tau-star}."

(b) **§3.7 stiffness选择**（~第123-136行附近）：增加1句承认+立即化解：
> "The stiffness selection is the derivation chain's weakest link: Pearson $\chi^2$ is motivated by attention-load asymmetry and independently supported by the self-consistent analysis ($p \approx 0.85$), but is not uniquely determined. However, the basin width ensures this choice has negligible practical impact: the $L$-exponent could range from $0.465$ to $0.561$ (Table~\ref{tab:stiffness-sweep}) without exiting the ${<}1\%$ PPL basin."

(c) **§3.2 surrogate validation**（~第36行）："Appendix validates this" 太淡了。改为更有力的呈现：
> "This is the theory's sole approximation, validated by the strictest available test: does the allocation derived from $K_{\mathrm{app}}$ also reduce collision under the \emph{exact, oscillatory} kernel? Yes---by $24$--$92\%$ across 12 configurations spanning three orders of magnitude in $L$ and $b$ (Table~\ref{tab:surrogate-validation})."

## WRITE-4 显式evidence hierarchy

在 **§5.1 Setup** 末尾（05_experiments.tex ~第8行后）添加：

> "Evidence is organized in three tiers. \emph{Primary} (3-seed, directly tied to central claims): EVQ$\times$YaRN composition (\S\ref{sec:exp-yarn}), PE-dominant extrapolation (\S\ref{sec:exp-pe}), and MLA (Appendix~\ref{sec:mla-results}). \emph{Robustness} (3-seed, single architecture): capability preservation and multi-scale PPL (\S\ref{sec:exp-robust}). \emph{Supporting} (1--2 seed): progressive training, video DiT, LoRA at 7B (\S\ref{sec:exp-supporting}). No primary claim rests on tier-3 evidence."

## WRITE-5 Conclusion强化

`sections/07_conclusion.tex` 目前是一段。拆为两段：

**第一段**保持总结，但开头改为更有力的开场：
> "This paper establishes that channel-frequency allocation---fixed at the geometric schedule since RoFormer---is a consequential and increasingly important RoPE design axis."

**第二段**新增前瞻（2-3句即可）：
> "The variational framework opens several directions: closing $\lambda$ analytically, extending the modality correction beyond the empirical factor observed for video DiT, and testing whether the wide robustness basin persists beyond 7B scale. The structural insight---that allocation shape matters more than allocation precision---suggests that future RoPE improvements should prioritize density design over per-channel tuning."

# ═══════════════════════════════════════
# 第四部分：MLA scope + 27-config透明化
# ═══════════════════════════════════════

## SCOPE-1 MLA对比的scope calibration

`appendix/a3_supporting_results.tex` ~第6行当前：
> "the attention mechanism used in production-scale models such as DeepSeek-V2/V3"

改为：
> "Multi-head Latent Attention (MLA), the compressed-RoPE attention variant introduced in DeepSeek-V2 \citep{deepseekv2}. Our configuration uses $d_{\mathrm{rope}}{=}32$ and base${=}500\mathrm{K}$; production DeepSeek-V2/V3 uses $d_{\mathrm{rope}}{=}64$ and base${=}10\mathrm{K}$. The shared architectural property---a compressed RoPE subspace with far fewer frequency channels than standard MHA---is what makes allocation quality decisive."

## SCOPE-2 "27 configurations" 透明化

在 Appendix 中（建议在 a1_proofs.tex 的 §tau-scaling 部分末尾）添加一个表格，列出所有27个configuration的：
- Architecture type (MHA / MLA / DiT / LoRA)
- d_head, L_train, base
- Seed count
- τ* formula variant (base formula / 0.53× modality correction)
- Evidence tier (Primary / Robustness / Supporting)

请根据论文中实际报告的所有实验config填写。Video DiT行必须标注使用了0.53×修正系数。如果27个config中有些shared同一model但不同τ sweep点，也要明确。

# ═══════════════════════════════════════
# 第五部分：格式清理
# ═══════════════════════════════════════

## FMT-1 清理未引用labels

以下labels已定义但从未 `\ref`。处理方式二选一：添加引用（如果该内容值得交叉引用），或删除label。

**建议添加引用的**（理论亮点，值得从正文指向）：
- `thm:self-consistency`：在§3正文中提一句 "a self-consistency theorem (Theorem~\ref{thm:self-consistency} in Appendix)"
- `eq:green-identity`：可在waterbed讨论中引用

**建议删除label的**（不需要交叉引用）：
- `fig:freq-dynamics`, `tab:750m-supporting`, `tab:appendix-repro`, `tab:dit-base-sweep`
- `sec:750m-supporting`, `sec:attn-redistribution`, `sec:mechanism-isolation`, `sec:progressive-dynamics`, `sec:quality-downstream`, `sec:self-consistency`
- `eq:T2-closed`, `eq:self-consistency`

## FMT-2 Table 6 小数位统一

`tables/table6_750m_continue_supporting.tex` 使用3位小数（25.922），全文其他表格用1位。统一为1位小数。

## FMT-3 `\citet` vs `\citep`

全文只使用了 `\citep`。在 Related Work（02_related.tex）中，当作者名是句子主语时改用 `\citet`。例如：
- "Resonance RoPE~\citep{wang2024resonance} is the closest precedent" → "Resonance RoPE \citet{wang2024resonance} is the closest precedent" 或更自然地重写为 "\citet{wang2024resonance} (Resonance RoPE) is the closest precedent"

检查全文所有 `\citep`，将句子主语场景改为 `\citet`。

## FMT-4 未使用的图片文件

以下PDF在 `figs/` 中存在但从未被任何 .tex 引用。不影响编译，但会增大submission包。如果方便，将它们移到 `figs/unused/`：
- `attn_distance_heatmap.pdf`, `attn_weight_vs_distance.pdf`
- `fig4a_attn_aggregate.pdf`, `fig4b_attn_perhead.pdf`
- `fig_tau_sweep_collision.pdf`, `fig_tau_sweep_cross_scale.pdf`, `fig_tau_sweep_freq_dist.pdf`, `fig_tau_sweep_ppl.pdf`
- `fig_unification_allocations.pdf`

# ═══════════════════════════════════════
# 第六部分：NeurIPS视角自检
# ═══════════════════════════════════════

完成所有修改后，请逐项自检：

## 编译和格式
- [ ] `tau_basin.pdf` 引用已处理（不再报错）
- [ ] 正文 §1-§7 含表图 **≤9页**（如果超了，精简appendix中被拉入正文的内容）
- [ ] 所有 `\ref` 无 undefined warnings
- [ ] 所有 `\citep`/`\citet` keys 在 bib 中存在
- [ ] 编译无 error

## 数值一致性
- [ ] DiT 均值已修正：正文 -26%/-15%/-32%，footer -26%/-15%/-32%
- [ ] MLA τ=1.414 括号推导已修正或删除
- [ ] Signal gradient "-52%" 引用已修正
- [ ] Abstract MLA -31.1% baseline已消歧
- [ ] MLA "+0.9%" 已改为 "+1.1%" 或 "approximately +1%"

## 引用完整性
- [ ] DeepSeek-V2/V3, LLaMA-2/3, Qwen2.5 已添加引用
- [ ] Dead-channels表中的5个video模型已添加引用
- [ ] Moving MNIST 已添加引用
- [ ] ICML volume字段已统一
- [ ] 无orphaned bib条目（每条都被cite）

## 行文和narrative
- [ ] Abstract已重组为三层递进
- [ ] §3.7 sub-optimality framing是offensive的（landscape结构发现）
- [ ] §3 theory呈现：exact层光芒>semi-analytic层的gap
- [ ] §5.1 evidence hierarchy tier系统已添加
- [ ] Conclusion拆为两段并强化
- [ ] MLA scope已calibrate（注明与DeepSeek的hyperparameter差异）
- [ ] 27-config分解表已添加到Appendix

## NeurIPS reviewer视角最终检查
- [ ] 读完abstract后，能否一句话说出论文为什么重要？
- [ ] 每个primary claim都有3-seed支撑，且在正文中可直接找到对应table？
- [ ] 单seed结果是否全部标注为"supporting evidence"？
- [ ] 理论的epistemic层次是否清晰（exact vs semi-analytic）？
- [ ] 是否有任何overclaim?（特别检查：abstract、intro最后一段、conclusion）
- [ ] Limitations是否覆盖了reviewer可能问的所有scale/generalization问题？

# 修改原则

1. **效果为王**：100% vs 61%、-31.1%@10σ、-35% vs 32-param learned PE——这些是论文的绝对王牌。所有行文修改服务于让这些数字更加不可忽视。
2. **理论是crown jewel**：完整变分框架（ODE→闭合解→几何极限→waterbed→自洽定理→12-config精确kernel验证→36-config mechanism isolation）在PE领域无人匹敌。大力呈现exact层，semi-analytic层（χ²选择、γ rounding）诚实标注+basin化解。
3. **sub-optimality是进攻武器**：零参数公式deliberately不完美但match 27 configs——这证明的是landscape结构，不是公式精度。
4. **epistemic honesty是加分项**：三层标注、supporting evidence显式降级、limitations完整——保持这个风格，不要为了push rating而overclaim。
5. **NeurIPS格式第一**：9页限制、匿名、模板格式不可违反。

---END---
