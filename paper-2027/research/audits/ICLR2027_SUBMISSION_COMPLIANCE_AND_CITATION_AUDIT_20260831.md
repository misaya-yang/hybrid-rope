# ICLR 2027 投稿合规与引用审计（修订版）— 2026-08-31

对象：`paper-2027/` 下当前 `.tex` 源、`paper-2027/refs/references.bib`、
`paper-2027/main.bbl` 与 `paper-2027/main.pdf`。当前 PDF SHA-256 为
`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`，正文
9 页、参考文献为第 10--13 页、附录从第 14 页开始、总计 31 页。

范围：投稿格式、匿名与声明门禁，以及所有**当前实际被引**文献的存在性、标题、
作者、年份、venue/type、卷页、DOI/官方 URL 和正文归因。本文档是内部审计，不是
论文、实验结果或投稿回执；论文当前 TeX/PDF 和官方文献源优先于本文档。

## 0. 修订原因

旧版 §5 不是实际 42 条被引文献的准确清单：

- TeX 静态提取与 `main.bbl` 各有 42 个 key，二者集合完全一致；
- 旧版第一组声称 24 条并列出 24 条，但混入未被引的 `li2025hope`；
- 旧版第二组声称 18 条，实际列出 21 条，其中
  `vaswani2017attention`、`shaw2018self`、`raffel2020exploring`、
  `press2022alibi`、`su2024roformer` 均未被当前论文引用；
- 旧版漏列实际被引的 `dasigi2021qasper`、`liu2024scaling`、
  `zheng2024dape`；
- 旧版清单并集为 45 个 key，与“42 条实际被引且逐条核实”的表述矛盾。

因此旧版“仅 5 条 P2”结论作废。本版重新以当前 TeX/BBL 的 42-key 集合为唯一范围，
并用官方 proceedings、官方 OpenReview/会议页面、arXiv 当前元数据和 DOI 页面复核。

## 1. 结论

### 1.1 当前论文

- **P0：0。** 未发现虚构文献、冒用 DOI、错误论文身份或错误归因。
- **P1：0。** 42/42 条实际被引文献均真实存在，TeX cite 集合与 BBL bibitem
  集合完全一致，BibTeX 日志 `warning$ = 0`。
- **P2：已修复。** 普通作者列表、正式 venue、已发布卷页、DOI/官方 URL 和两处
  `.bib` 源标题大小写已写回。Llama 3 与 Qwen2.5 的超长作者列表采用显式注释的
  `and others` 例外；RePo/AdaRoPE 尚未发布或尚未定位的最终 PMLR 卷页不作猜测。

### 1.2 ICLR 规则边界

ICLR 2027 Author Guidelines 明确写明：泄露作者身份和正文超过 9 页会 desk reject；
官方指南没有把轻微参考文献字段缺失列为自动 desk-reject 条款。引用造假、错误归因
或严重漏引仍会构成科学诚信与可信度风险，必须按高风险处理。官方指南：
<https://iclr.cc/Conferences/2027/AuthorGuidelines>。

## 2. 当前 42 条实际被引文献

以下集合同时来自当前 TeX 与 `main.bbl`，任何后续引用审计都必须先重新生成并比对
这两个集合，不能从旧审计清单继承：

### 2026（12）

`karypis2026lerope`、`li2026copeclipped`、`li2026repo`、
`movahedi2026selectiverope`、`oka2026frequencyentropy`、`oka2026fmrope`、
`tang2026jetlong`、`tian2026mrrope`、`wang2026adarope`、
`wertheimer2026frayed`、`wu2026datashapes`、`zhang2026grape`。

### 2025（8）

`barbero2025round`、`chen2025hope`、`chiang2025rotary`、`hua2025fope`、
`olmo2furious`、`shang2025longrope2`、`videorope2025`、`zhao2025riflex`。

### 2024（13）

`bai2024longbench`、`deepseekv2`、`ding2024longrope`、
`grattafiori2024llama3`、`hsieh2024ruler`、`li2024fire`、
`liu2024scaling`、`penedo2024fineweb`、`peng2024yarn`、
`qwen2024qwen25`、`wang2024resonance`、`xu2024base`、`zheng2024dape`。

### 1998–2023（9）

`black2022gptneox`、`chen2024position`（条目年份 2023）、
`dasigi2021qasper`、`gray1998quantization`、`ho2020twowiki`、`hu2022lora`、
`kazemnejad2023impact`、`merity2017wikitext`、
`srivastava2015unsupervised`。

## 3. 已写回 `references.bib` 的 P2 修复记录

这些都是当前论文实际被引条目，以下记录保留修复前问题和已写回值。

| Key | 修复前问题 | 已写回值 | 权威来源 |
|---|---|---|---|
| `hua2025fope` | `Hua, Ermo and others`；缺 volume/pages/URL | 补全 10 位作者：Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Youbang Sun, Yuchen Fan, Xuekai Zhu, Biqing Qi, Ning Ding, Bowen Zhou；PMLR 267:24932–24949 | [PMLR](https://proceedings.mlr.press/v267/hua25b.html) |
| `hsieh2024ruler` | 仍写 arXiv preprint | 优先写正式接受版本：First Conference on Language Modeling (COLM 2024)，保留 arXiv 作为 locator；未找到正式 DOI/卷页，不得猜测 | [OpenReview](https://openreview.net/forum?id=kIoBbc76Sy), [arXiv](https://arxiv.org/abs/2404.06654) |
| `bai2024longbench` | 缺正式卷册、页码、DOI、URL | ACL 2024 Long Papers，3119–3137，DOI `10.18653/v1/2024.acl-long.172` | [ACL Anthology](https://aclanthology.org/2024.acl-long.172/) |
| `wang2024resonance` | 缺页码、DOI、URL | Findings of ACL 2024，586–598，DOI `10.18653/v1/2024.findings-acl.32` | [ACL Anthology](https://aclanthology.org/2024.findings-acl.32/) |
| `zheng2024dape` | 缺 DOI、URL | DOI `10.52202/079017-0838`，保留 NeurIPS 37 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f050fa9f0d898e3f265d515f50ae8f9-Abstract-Conference.html) |
| `ding2024longrope` | 缺页码、URL | PMLR 235:11091–11104 | [PMLR](https://proceedings.mlr.press/v235/ding24i.html) |
| `shang2025longrope2` | 缺页码、URL | PMLR 267:54203–54218 | [PMLR](https://proceedings.mlr.press/v267/shang25a.html) |
| `videorope2025` | 缺页码、URL | PMLR 267:66118–66136 | [PMLR](https://proceedings.mlr.press/v267/wei25h.html) |
| `zhao2025riflex` | 缺页码、URL | PMLR 267:77539–77557 | [PMLR](https://proceedings.mlr.press/v267/zhao25m.html) |
| `barbero2025round` | `.bib` 源写 `What Makes ... Useful?`；无官方 URL | 官方标题为 `Round and Round We Go! What makes Rotary Positional Encodings useful?`；当前 BBL 已被 style 转成 sentence case，但源字段仍应统一 | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html) |
| `tian2026mrrope` | `.bib` 源写 `Mixed-Radix` | 官方标题为 `MrRoPE: Mixed-radix Rotary Position Embedding`；当前 BBL 已正确渲染，仍建议修正源字段并换成 proceedings URL | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html) |
| `olmo2furious` | 仅写 `{OLMo Team}`，且 group 名次序与 arXiv 的 `Team OLMo` 不同；缺 locator | 已按 arXiv 当前元数据写入 Team OLMo 加 42 位作者、arXiv DOI 与 URL | [arXiv](https://arxiv.org/abs/2501.00656) |

### 3.1 统一 locator 补全

本轮采用以下最小规则：每条引用至少有一个 canonical locator；已正式发表时优先
官方 proceedings/Anthology/OpenReview，纯 arXiv 条目使用 arXiv URL；有适用的
正式 DOI 时同时写 DOI。以下实际被引条目的 locator 已补齐：

- arXiv/technical report：`chen2024position`、`deepseekv2`、
  `grattafiori2024llama3`、`qwen2024qwen25`、`olmo2furious`、
  `penedo2024fineweb`、`li2026copeclipped`；
- ICLR：`barbero2025round`、`peng2024yarn`、`li2024fire`、`hu2022lora`；
- ACL：`chen2025hope` 已有 DOI，补官方 Anthology URL；
- NeurIPS：`kazemnejad2023impact` 可补 DOI `10.52202/075280-1082`；
- PMLR：使用 §3 的官方页面，不伪造 PMLR 不提供的 venue DOI。

### 3.2 超长作者例外

本轮采用明确例外模式：`grattafiori2024llama3`（数百名作者）和
`qwen2024qwen25`（团队署名及长贡献者列表）保留官方领先作者加 `and others`，并在
Bib 源中用注释登记；不把它们描述为完整作者字段。`olmo2furious` 的 43 个作者和
`hua2025fope` 的 10 个作者均已完整写入。

## 4. 已关闭的旧 P2 与待确认项

| 旧项目 | 修订后裁决 |
|---|---|
| `li2025hope` 作者截断 / NeurIPS 2025 | 论文真实存在且为 NeurIPS 2025，但**当前未被引用**，不属于 42 条引用审计；如保留在 Bib 库，可另行补全 5 位作者，不得计入当前 PDF 的 P2 |
| `li2026copeclipped` 的 `CoPE:` 前缀 | 旧判定错误。arXiv 当前官方标题就是 `CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs`，前缀应保留；来源：<https://arxiv.org/abs/2602.05258> |
| `xu2024base` DOI | 已由 NeurIPS 官方页面直接确认 `10.52202/079017-2773` 正确，不是待确认项 |
| `chen2024position` key 名年份 | key 名含 2024，但渲染 year=2023 正确；建议未来机械重命名时统一，但这不是 reviewer-visible 错误 |
| `li2026repo` / `wang2026adarope` venue | ICML 2026 官方 virtual poster/downloads 与 arXiv comment 均支持已接受；最终 PMLR volume/pages/venue DOI 尚未发布或未定位，投稿前再查一次，不得编造 |
| `zhang2026grape` 作者 | 官方 camera-ready PDF 与 arXiv 支持当前作者链；ICLR proceedings 索引页曾出现冲突作者元数据，应以 camera-ready PDF 与 arXiv 为准，不按异常索引改 Bib |

## 5. 核心 related-work 归因

本轮重新抽查当前 `sections/02_related.tex` 的关键归因，未发现 P0/P1 错误：

| 稿中 claim | 一手证据 | 判定 |
|---|---|---|
| FMRoPE 是 target-aware support/base construction；§6.1 设定 `theta=L_train` 并使用对数均匀指数网格 | ICLR 2026 官方 PDF/OpenReview `PR1PPxvG9Q` §6.1 | 一致 |
| LeRoPE 学习每个频率带的标量，并报告 in-window language-modeling 改善 | arXiv:2607.10134 | 一致；仅作 related evidence，不是 matched comparator |
| AdaRoPE 使用 head-specific learned rotation frequencies 与 head-wise scaling | arXiv:2607.19363；ICML 2026 官方 poster | 一致 |
| `wu2026datashapes` 提供 data-dependent account of learned frequency use | arXiv:2607.07678 | 当前 `learned use of a supplied grid` 易被读成“该方法学习频率”；建议正文改成 `data-dependent accounts of learned frequency use` |
| Merity et al. 用作 WikiText corpus owner | ICLR 2017/OpenReview `Byj72udxe` | 一致；论文标题是 `Pointer Sentinel Mixture Models`，不应改成 `WikiText` |
| Srivastava et al. 用作 Moving MNIST 基础来源 | PMLR 37:843–852 | 基础数据归因成立；`Oscillating Moving MNIST` 应明确是本文变体，不能暗示原论文使用该精确变体 |

## 6. 投稿合规检查

| 检查项 | 结果 | 依据 / 边界 |
|---|---|---|
| 实际 cite key 与 BBL bibitem 集合 | PASS | 42 vs 42，集合完全一致 |
| BibTeX 构建 | PASS | `main.blg`: 42 entries，`warning$ = 0` |
| 引用存在性 / 论文身份 | PASS | 42/42 有权威来源；无虚构、无错误 DOI 归属 |
| 引用字段零容忍 | PASS（有边界） | 当前可确认的作者、venue、卷页、DOI/URL 与标题源字段已写回；未来 PMLR 元数据不猜测 |
| 匿名 | PASS | 当前 PDF/文本无作者身份，`\iclrfinalcopy` 未启用 |
| 正文页数 | PASS | 9 页；References 从第 10 页开始 |
| 模板 / 页眉 / 行号 | PASS | 当前 ICLR 2027 样式与构建回执通过 |
| Ethics / Reproducibility / AI use | PASS | 正文后、References 前；最终表单披露仍是 AUTHOR ACTION |
| PDF 与当前源同步 | PASS（当前 hash） | `37aa6402...167e4`；Bib、BBL 与 PDF 已在同次编译中生成 |
| 双投 | AUTHOR ACTION | 若 NeurIPS 2026 接收则在 ICLR full-paper deadline 前撤回/不提交重复全文；提交前按最新官方政策复核 |

## 7. 修复与重编译回执

- `paper-2027/refs/references.bib` SHA-256：
  `bd09952fa53d8b1acf955bc2aef57a8b282c09f2a0aef6f0e488967ae6736f14`；
- `paper-2027/main.bbl` SHA-256：
  `845224e8f2e7a8f5f61c1625629adff8100653d235bcb0d82fcced9eb1334a58`；
- `paper-2027/main.pdf` SHA-256：
  `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`，
  776,774 bytes；
- `./compile.sh`：全部 ICLR 2027 format gates PASS；正文 9 页，总计 31 页，
  0 undefined refs/cites，最差 overfull hbox 0pt，匿名 PASS，Type 3 字体 0，
  未嵌入字体 0；
- TeX cite 集合与 BBL bibitem 集合：42 vs 42，完全一致；`main.blg warning$=0`；
- References 第 10--13 页已逐页视觉检查，未见截断、重叠、溢出或异常断行；
- `paper/main.pdf` SHA-256 仍为
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`，
  不可变基线未改；
- 未提交、未推送、未上传 OpenReview。引用源变化使现有 supplement archive 继续处于
  source-unsynchronised 状态，最终发布前必须用 curated packager 重建。

## 8. 证据优先级与审计边界

1. 官方 proceedings / ACL Anthology / PMLR / NeurIPS / ICLR 页面；
2. 官方 OpenReview/会议 virtual poster；
3. arXiv 当前 abstract/author metadata 与 arXiv DOI；
4. DOI resolver；
5. DBLP 仅作交叉检查，不覆盖一手来源；
6. GitHub 项目名、搜索摘要和第三方镜像不能单独决定标题、作者或 venue。

本审计证明的是“当前 42 条引用存在且已定位出元数据修复项”，不是对所有正文 claim、
全部附录数字、未来文献版本或投稿平台最终渲染的永久认证。
