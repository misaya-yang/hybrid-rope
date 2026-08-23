# paper-2027 — ICLR 2027 submission package

EVQ-Cosh, rebuilt from the NeurIPS 2026 submission (`../paper/`) plus the
rebuttal-cycle evidence in `../rebuttal/rebuttal_0723/`.

Build: `./compile.sh`（`make -f build.mk` 也行；远程工具写不了名为 `Makefile` 的文件，所以叫 `build.mk`）。产物：`main.pdf`。

匿名补充包：`python ../scripts/package_supplement.py --profile iclr2027`。
该 profile 只收录当前论文源码、三张使用中的图、频率实现、关键分析/识别脚本、
最小测试和已清洗的 machine-readable evidence，并在写 ZIP 前执行身份与密钥扫描。

---

## 当前交接

先读 [`HANDOFF.md`](HANDOFF.md)。它是唯一记录当前稿件 hash、验证收据、
worktree 边界和下一步的文档；本 README 只保留稳定的包结构与构建说明。
`HANDOFF.md` 是内部文件，匿名 supplement 会刻意排除；导出包读者可跳过本节。

第一性原理和最高优先级是最大化 ICLR 2027 录用概率。当前阶段不继续做推测性
扩写或无闸门实验。固定支撑成熟模型控制与 151.9M crossing 已完成，但仍是
internal case study，尚未改写正文；当前决策与下一步只看 `HANDOFF.md`。

---

## 内部研究入口

后续论文工作的默认目录就是 `paper-2027/`。在改核心 claim、理论骨架或实验叙事前，
先读：

- [`HANDOFF.md`](HANDOFF.md) — 当前稿件、验证与下一步
- [`research/README.md`](research/README.md) — 研究索引、阅读顺序与 owner 路由
- [`research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md) — 当前 claim 架构、证据路由与否决方向
- [`research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
- [`research/attention-aware-retrofit/README.md`](research/attention-aware-retrofit/README.md) — retrofit 的 results / evidence / analysis / preflight 分层入口
- [`research/audits/README.md`](research/audits/README.md) — 内部审计索引
- [`research/external-reviews/README.md`](research/external-reviews/README.md) — 外部模型复核；不是 canonical evidence

这些文件是内部审计与续作交接，不是可直接复制进正文的 outward-facing 文案。

---

## 会议与格式（2026-08-19 核实；提交前必须实时复核）

目标是 **ICLR 2027**：摘要 **2026-09-18 AoE**、正文 **2026-09-25 AoE**。
官方 Author Guide 与 CFP 已于 2026-08-19 复核。

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

**AI use statement 已由作者于 2026-08-19 确认**：`sections/08_ai_use.tex`
按 ICLR 2027 AI Policy 的 required / recommended 两类清单写成，作者确认
逐句对本项目为真。如此后 AI 使用范围变更，提交前同步更新。

原 ICML 双栏 8 页模板与旧 `main.tex` 保留在 `venue_icml_fallback/`，正文不依赖它们。

### dual submission 提醒

ICLR 2027 FAQ 明确允许 NeurIPS 待定期间先提交 ICLR 摘要，重复投稿检查只针对
全文。NeurIPS 11628 于 **2026-09-24 AoE** 通知，早于 ICLR 全文截止
**2026-09-25 AoE**，因此时间线本身不要求提前撤稿。

当前 ICLR 稿已对 `../paper/` 做逐项差异审计：中心理论、固定范围识别、
table--weights co-adaptation 与成熟模型证据都是新主线，EVQ-Cosh 降为其中一个
构造性实例；当前判定为独立续作，而非 substantially similar 的重投。若
NeurIPS 接收，在 ICLR 全文中以第三人称引用已接收工作并明确新增贡献；
若拒稿，无需这一引用动作。

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
   **operator**（DAPE/FIRE）。归因全部转到 exact-range/M4 的零参数固定
   schedule 对照。
3. **Range-composition 身份写清楚。** 全文用 `\rs{}` 渲染为
   `YaRN-style`，专指仓库固定索引 range operator：它保留高频并渐进缩放低频，
   承担“同一 range 操作在两种训练表上的杠杆差异”这条证据。引用方法始终写
   `YaRN`，两者不混用。

## YaRN-style 实现身份

App. D 按真实代码定义该算子：固定 20%–90% channel-index 边界、cubic
smoothstep，并把 `1+0.07 log2(s)` 折入频率缩放。所有 composition 对照使用
完全相同的 `\rs{}` 实现、cut indices 和 scale。

## 目录

```
main.tex                    正文入口；顶部有 venue note
iclr2027_conference.sty/.bst  官方 ICLR 2027 模板（原样）
venue_icml_fallback/        旧 ICML 模板与旧 main.tex，未被引用
compile.sh / build.mk       构建 + 自动合规检查（页数/未定义引用/溢出/匿名性）
sections/
  00_abstract.tex           fixed-support identification → 432M flagship → mature capability
  01_intro.tex              结果首屏：因果控制 → 训练趋势 → 下游与 trained use
  02_related.tex            重写：FMRoPE / LeRoPE / AdaRoPE 准确定位
  03_theory.tex             full sin/cos geometry → obstruction → closed-form construction
  04_experiments.tex        exact-range → scarce budget/scale → capability → schedule scope
  05_discussion.tex         static basis、trained use、range transport 与 LeRoPE
  06_ethics.tex / 07_reproducibility.tex / 08_ai_use.tex  ICLR 声明（不计页数）
tables/
  table_layers.tex          新增：三层参数化
  table_m4.tex              新增：exact-range factorial
  table_ruler.tex           新增：RULER 13-family
  table_evq_ramp.tex        同一 YaRN-style range 操作的 substrate 交叉
  table_pe_dominant.tex     原 table4，重新标注，移入附录
figs/
  fig_evidence_overview.pdf 三种子控制 → 432M K=16 旗舰 → 下游/8B adapted callout
  make_fig_evidence_overview.py  生成脚本（冻结 owner 数值与断言）
  fig_method_overview.pdf   重写：有限预算 → 闭式构造 → 两项关键控制
  make_fig_method_overview.py  生成脚本（含几何计数断言）
  fig_frequency_geometry.pdf  新增：频率位置与 full-subspace redundancy
  make_fig_frequency_geometry.py  生成脚本（复用 full-RoPE audit）
appendix/
  a1_proofs.tex             核心证明与 operating-rule 支撑推导
  a2_experiment_details.tex supporting scratch / video protocols
  a5_identification.tex     识别协议 + YaRN-style 实现 + learned comparator
  a6_mature_scale.tex       新增：成熟模型协议、2Wiki、RULER、causal source-use
  a3/a4_supporting*.tex     沿用
research/
  README.md                 唯一研究路由与 owner 索引
  attention-aware-retrofit/ results / evidence / analysis / theory / preflights
  audits/                   内部审计与已证伪代理指标
  external-reviews/         外部模型复核，永不直接升级 claim
```

## 构建报告会检查什么

`./compile.sh` 优先用 pdflatex，本机没有时自动回退 tectonic；结束时打印正文结束页（ICLR 上限 9）、
未定义引用数、最严重的 overfull hbox、匿名性、字体和文件大小；任一硬门禁失败
都会非零退出。正文结束页由 `sections/05_discussion.tex`
后的 `\label{page:bodyend}` 定位（现在在 `main.tex` 里）——**别删那一行**。

## 已知的取舍

- 当前正文结束于第 9 页，已用满 ICLR 上限；每新增一段都必须指名替换对象。
- Fig. 2 用的是各 owner 报告里的数字，生成脚本里写了来源；改数字要连脚本一起改。
