# paper-2027 — ICML 2027 submission package

EVQ-Cosh, rebuilt from the NeurIPS 2026 submission (`../paper/`) plus the
rebuttal-cycle evidence in `../rebuttal/rebuttal_0723/`.

Build: `./compile.sh`（`make -f build.mk` 也行；远程工具写不了名为 `Makefile` 的文件，所以叫 `build.mk`）。产物：`main.pdf`。

---

## ⚠️ 先读这条：会议与截稿日期对不上

你给的日期是 **9/18 摘要 + 9/25 正文**。查证结果：

| | ICML 2027 | ICLR 2027 |
|---|---|---|
| CFP 状态 | **未发布**（icml.cc `/Conferences/2027` 返回 404） | 已发布 |
| 摘要截稿 | 未公布 | **2026-09-18 AoE** |
| 正文截稿 | 未公布，按往年节奏约 **2027 年 1 月下旬**（ICML 2026 是 1/28） | **2026-09-25 AoE** |
| 地点 | 南美（Future Meetings 页只写了 "2027 — South America"） | — |
| 正文页数 | 8 页 | 9 页（discussion 阶段可到 10） |
| 必需声明 | Impact Statement | AI use statement |

**你说的 9/18 + 9/25 精确等于 ICLR 2027 的两个截稿日。** 这份包按你的要求做的是
**ICML 格式（8 页 + Impact Statement）**。如果目标其实是 ICLR 2027，需要改的只有
三处：换 `iclr2027_conference.sty`、正文放宽到 9 页、把 `sections/06_impact.tex`
改成 AI use statement。正文内容本身不用动。

另外提醒一句 dual submission：ICML/ICLR 都禁止与其它会议**并行**在审。
NeurIPS 9/24 出结果、9/25 截稿，只有在 9/24 确认被拒之后提交才不构成并行投稿——
摘要阶段（9/18）通常不算，但这条边界建议自己再确认一次。

## ⚠️ 模板是占位版

ICML 2027 官方 style 包还没发布。`icml2027.sty` / `icml2027.bst` 是把**官方
ICML 2026** 文件的年份/卷号字符串替换后得到的；`icml2026.sty` / `icml2026.bst`
原封不动保留在旁边作为权威参照。`icml2027.sty` 里 `\ICML@appearing` 的卷号写的是
`PMLR 3XX`，已加注释标记为 PLACEHOLDER。**官方 icml2027.zip 一发布就整包换掉，
正文不需要任何改动。**

---

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
icml2027.sty / .bst         占位模板（由官方 2026 版派生）
icml2026.sty / .bst         官方原版，作参照与回退
compile.sh / build.mk       构建 + 自动合规检查（页数/未定义引用/溢出/匿名性）
sections/
  01_intro.tex              重写：spectral budget → identification → derivation → scale
  02_related.tex            重写：FMRoPE / LeRoPE / AdaRoPE 准确定位
  03_theory.tex             分层：surrogate theorem → inverse CDF → operating rule
  04_experiments.tex        重写：exact-range → mature endpoints → attribution → composition
  05_discussion.tex         围绕 finite spectral budget 收束
  06_impact.tex             新增（ICML 必需）
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

`./compile.sh` 需要完整 TeX Live/MacTeX；结束时打印正文结束页（ICML 上限 8）、
未定义引用数、最严重的 overfull hbox、匿名性、字体和文件大小；任一硬门禁失败
都会非零退出。正文结束页由 `sections/05_discussion.tex`
末尾的 `\label{page:bodyend}` 定位——**别删那一行**。

## 已知的取舍

- 当前正文标签落在第 8 页，已用满 ICML 上限；不要再增加低杠杆内容。
- Fig. 2 用的是各 owner 报告里的数字，生成脚本里写了来源；改数字要连脚本一起改。
