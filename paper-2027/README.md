# paper-2027 — ICLR 2027 submission package

EVQ-Cosh, rebuilt from the NeurIPS 2026 submission (`../paper/`) plus the
rebuttal-cycle evidence in `../rebuttal/rebuttal_0723/`.

Build: `./compile.sh`（`make -f build.mk` 也行；远程工具写不了名为 `Makefile` 的文件，所以叫 `build.mk`）。产物：`main.pdf`。

---

## 内部研究入口

后续论文工作的默认目录就是 `paper-2027/`。在改核心 claim、理论骨架或实验叙事前，
先读：

- [`research/README.md`](research/README.md) — 研究索引、阅读顺序与 owner 路由
- [`research/ICLR2027_THEORY_ARCHITECTURE.md`](research/ICLR2027_THEORY_ARCHITECTURE.md) — **理论重写蓝图**：attention 第一性原理链、cosh 的准确地位、in-window/外推的可支持措辞、9 页定理清单
- [`research/ICLR2027_REVIEW_AND_EVIDENCE_AUDIT.md`](research/ICLR2027_REVIEW_AND_EVIDENCE_AUDIT.md) — 审稿意见逐条落地审计 + 5 月后证据清单
- [`research/ICML2027_RESEARCH_SYNTHESIS_20260819.md`](research/ICML2027_RESEARCH_SYNTHESIS_20260819.md) — 当前重构决策与中稿优先原则（venue 名沿用历史命名）
- [`research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)

这些文件是内部审计与续作交接，不是可直接复制进正文的 outward-facing 文案。

---

## 会议与格式（已定，2026-08-19 核实）

目标是 **ICLR 2027**：摘要 **2026-09-18 AoE**、正文 **2026-09-25 AoE**。
（ICML 2027 官方 CFP 至今未发布；9/25 这个日期只属于 ICLR。）

| 项 | 要求 | 本包状态 |
|---|---|---|
| 模板 | 官方 `iclr-2027-style-files.zip` | ✅ 原样放在包内（`iclr2027_conference.sty/.bst`、`natbib.sty`、`fancyhdr.sty`、`math_commands.tex`） |
| 版式 | 单栏 | ✅ |
| 正文页数 | ≤ **9** 页（rebuttal/camera-ready 放宽到 10） | ✅ 正文结束于第 9 页（`compile.sh` 硬门禁） |
| 参考文献 / 附录 | 不计页数 | ✅ |
| AI use statement | **必需**，单独一节，不计页数 | ✅ `sections/08_ai_use.tex` |
| Ethics statement | 推荐 | ✅ `sections/06_ethics.tex` |
| Reproducibility statement | 推荐 | ✅ `sections/07_reproducibility.tex` |
| 双盲 | 作者不得出现 | ✅ `\iclrfinalcopy` 保持注释；`compile.sh` 检查 |

**AI use statement 需要作者本人过目定稿**：`sections/08_ai_use.tex` 按 ICLR 2027
AI Policy 的 required / recommended 两类清单写成，逐句都必须对本项目为真。

原 ICML 双栏 8 页模板与旧 `main.tex` 保留在 `venue_icml_fallback/`，正文不依赖它们。

### dual submission 提醒

ICLR 与 NeurIPS 都禁止并行在审。NeurIPS 11628 的作者通知日是 **2026-09-24**，
晚于 ICLR 摘要截稿 **09-18**。在 09-24 之前向 ICLR 提交（含摘要）需要先主动撤回
NeurIPS 稿；这是作者本人的决定，本包不替你做。

## 这一版改了什么（对应审稿意见）

完整对照见 `CHANGES_FROM_NEURIPS2026.md`。三句话版本：

1. **实验主线重排。** 第一块是 **exact-range 识别实验**——钉死最高频、
   最低频、log-span，只动中间 30 个频率。这是回答 AC.1/AC.3/RzWsa.1/R27bE.3
   的那个实验，现在是全文的骨架，标题换成了 *RoPE Has a Spectral Budget*。
   第二块汇总 1.485B / 8B 的 strict generation、2Wiki、RULER 与
   causal source-use；第三块负责 schedule / $\tau$ 归因；第四块汇总
   fixed-scaler substrate leverage 与 MLA scarce-channel evidence。
2. **DAPE 那一行处理掉了。** 原 Table 4 标为 "DAPE" 的实际是 `free_inv_freq`
   （32 参数、layer-shared 可学习 inverse-frequency）。现在：正文里
   **不再承担任何 allocation-shape 归因**，退到附录 E，并按其真实身份标注为
   learned-**table** 家族（LeRoPE 那一类），明确区别于 learned positional
   **operator**（DAPE/FIRE）。归因全部转到 Primary I 的零参数固定 schedule 对照。
3. **YaRN 那一支重命名了。** 全文（含附录、表格）里我们自己那个实现一律写作
   `\rs{}`（渲染为 RAMP），并在 §4.1 和附录 F 明确声明：这是我们自己实现的
   fixed-index smooth-ramp scaler，属于 NTK-by-parts / YaRN 家族但**不是** YaRN
   参考实现，结论只能读作"同一算子在两种训练期频率表上的杠杆差异"，不能读作
   YaRN benchmark。引用 YaRN 论文本身的地方保持不变。

## RAMP 实现身份

App. F 已按真实代码补全：固定 20%–90% channel-index 边界、cubic
smoothstep，并把 `1+0.07 log2(s)` 折入频率缩放；它没有 YaRN 的
wavelength-derived band boundaries，也没有独立 attention-logit mscale。

## 目录

```
main.tex                    正文入口；顶部有 venue note
iclr2027_conference.sty/.bst  官方 ICLR 2027 模板（原样）
venue_icml_fallback/        旧 ICML 模板与旧 main.tex，未被引用
compile.sh / build.mk       构建 + 自动合规检查（页数/未定义引用/溢出/匿名性）
sections/
  01_intro.tex              重写：spectral budget → identification → derivation → scale
  02_related.tex            重写：FMRoPE / LeRoPE / AdaRoPE 准确定位
  03_theory.tex             分层：surrogate theorem → inverse CDF → operating rule
  04_experiments.tex        重写：exact-range → mature endpoints → attribution → composition
  05_discussion.tex         围绕 finite spectral budget 收束
  06_ethics.tex / 07_reproducibility.tex / 08_ai_use.tex  ICLR 声明（不计页数）
tables/
  table_layers.tex          新增：三层参数化
  table_m4.tex              新增：exact-range factorial
  table_mature.tex          新增：1.485B / 8B
  table_ruler.tex           新增：RULER 13-family
  table_evq_ramp.tex        原 table2，YaRN→RAMP
  table_pe_dominant.tex     原 table4，重新标注，移入附录
figs/
  fig_method_overview.pdf   重写：有限预算 → 闭式构造 → 两项关键控制
  make_fig_method_overview.py  生成脚本（含几何计数断言）
  fig_identification.pdf    新增，exact-range 识别主图
  make_fig_identification.py  生成脚本（数字来源写在 docstring 里）
appendix/
  a1_proofs.tex             沿用（已做 YaRN→RAMP 与 additivity 措辞修正）
  a2_experiment_details.tex 沿用（已做 YaRN→RAMP、DAPE-style→PE-dominant）
  a5_identification.tex     新增：识别协议与 provenance + RAMP 真实实现 + learned comparator
  a6_mature_scale.tex       新增：成熟模型协议、2Wiki、RULER、causal source-use
  a3/a4_supporting*.tex     沿用
```

## 构建报告会检查什么

`./compile.sh` 优先用 pdflatex，本机没有时自动回退 tectonic；结束时打印正文结束页（ICLR 上限 9）、
未定义引用数、最严重的 overfull hbox、匿名性、字体和文件大小；任一硬门禁失败
都会非零退出。正文结束页由 `sections/05_discussion.tex`
后的 `\label{page:bodyend}` 定位（现在在 `main.tex` 里）——**别删那一行**。

## 已知的取舍

- 当前正文结束于第 9 页，已用满 ICLR 上限；每新增一段都必须指名替换对象。
- Fig. 2 用的是各 owner 报告里的数字，生成脚本里写了来源；改数字要连脚本一起改。
