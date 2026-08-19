# 投稿前检查清单

`./compile.sh` 会自动跑掉前 5 项。其余需要人工确认。

## 自动（构建报告里）

- [x] **正文 ≤ 8 页**（ICML 上限；references / Impact Statement / appendix 不计）
      — 由 `\label{page:bodyend}` 定位，当前 = 8
- [x] 无未定义引用 / 未定义 citation
- [x] 最严重 overfull hbox < 5pt
- [x] 匿名性扫描（github 链接 / acknowledgement）无命中
- [x] PDF 内 author metadata = "Anonymous Authors"（style 文件自动处理）

## 格式（ICML）

- [x] 用 `\usepackage{icml2027}`，**不带** `[accepted]` / `[preprint]`（盲审版）
- [x] Impact Statement 在 references 之前、正文之后，不计页数
- [x] 单文件提交：正文 + appendix 在同一个 `main.pdf` 里
- [ ] **官方 icml2027.zip 一发布就整包替换 sty/bst**（当前是 2026 版派生的占位）
- [x] 提交文件 ≤ 50MB（当前 `main.pdf` 约 0.62MB）
- [x] 字体全部嵌入（`pdffonts main.pdf`：emb=yes，Type 3 = 0）

## 匿名（双盲）

- [x] 无作者名、无单位、无致谢
- [x] 自引用用第三人称
- [ ] **代码 archive 也要匿名**：删掉作者名、license 里的姓名、内部路径、
      集群主机名；匿名 GitHub 要用不可修改的分支
- [ ] arXiv 预印本可以有，但**审稿期内任何地方都不能说这是 ICML 投稿**
- [x] PDF 里没有残留批注 / todonotes（全文扫描通过）

## 内容（这一版特有，务必逐条过）

- [x] App. F 已按代码写明 RAMP 的固定 index 边界、smoothstep、折入频率的
      temperature 因子，以及缺失的 wavelength boundaries / attention mscale
- [x] 全文我们自己那支 range scaler 一律写作 `\rs{}`（RAMP），不写 YaRN
- [x] 引用 YaRN 论文本身的地方保留（§2、Table 1、App. F）
- [x] 32 参数对照按真实身份标注，退出正文归因，移入 App. E
- [x] 摘要 / intro / related work 里没有「优于 DAPE」类表述
- [x] FMRoPE (Oka et al., ICLR 2026) 已引用并在 Related Work 单列
- [x] dead-channel 观察归因给 Barbero et al.
- [x] LeRoPE 已引用（不再是 concurrent work，2026-07 发布）
- [x] §4.3 每个数字已逐项对上 `theory_results/` 的 owner / raw JSON
      （对照表在 `CHANGES_FROM_NEURIPS2026.md` §4）
- [ ] 落笔任何 "X is worse / did not improve" 之前回 owner 文件确认两臂身份
      （旧坑：OLMo RULER 表里 "Legacy two-seed mean" 和 "New 4K adaptation" 两列都是 EVQ）

## 独立审计后的证据 gate

一个独立 agent 把正文每个数字对着 `theory_results/` 的 owner 逐条核过。
它找到的错误我已经全部改了（见 `CHANGES_FROM_NEURIPS2026.md` §7）。
**剩下三条不是错误，是需要你决定的风险：**

- [x] `E-EXACT-RANGE-3S` 与 `E-HELDOUT` 的精确数字已从公开稿删除；当前头条
      使用 raw-backed seed-42 + 三-seed M4 factorial
- [x] collision prefactor 已统一为 `1.9%` relative / `0.28%` CV；错误的 base
      monotonicity 描述已删除
- [x] 未定位到 endpoint/midpoint `<1% PPL` 的 standalone owner，定量句已删除；
      endpoint-normalised 变体均单独标注
- [x] MLA 不再与 MHA 百分比作 matched 比较；配置已按 executable owner 修正为
      `d_head=64 / d_rope=32 / K=16 / lr=2e-4 / tau=1.414`

## Dual submission

- [ ] **NeurIPS 2026 (#11628) 的结果确认后再提交正文。** ICML/ICLR 都禁止并行在审。
- [ ] 若 NeurIPS 接收 → 本包不能按原样投；需要实质性区分或撤回
- [ ] 若被拒 → 可投，但 arXiv / OpenReview 上的公开旧版会被审稿人搜到，
      建议正文与旧版的差异足够明显（本版换了标题、换了 Primary 排序、换了骨架论证）

## 复现与代码

- [ ] 匿名代码 archive 含：inverse-frequency initializer、collision surrogate、
      τ 分析、评测脚本、Primary I–IV 的配置文件
- [ ] README 写清 exact-range 控制里"钉死"的到底是哪些量（这是本文的核心，
      审稿人一定会看这一条能不能复现）
- [ ] 数据集来源引用齐全（FineWeb-Edu / TinyStories / RULER / QuALITY / PG-19）
- [ ] 不放 checkpoint（体积 + 匿名）

## 提交当天

- [ ] `./compile.sh clean && ./compile.sh full`，构建报告全绿
- [ ] 肉眼翻一遍 PDF 前 8 页（图表位置、表格没出血、公式没截断）
- [ ] OpenReview 上摘要与 PDF 摘要一致
- [ ] 所有作者有 OpenReview profile，摘要截稿后不能加作者
