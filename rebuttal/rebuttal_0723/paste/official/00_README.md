# 讨论期补充 Official Comment — 定稿

2026-08-03 · 截止 8 月 3 日 AOE · 四条,共 2116 词(PC 上限 >10,000 词)

## 发什么

| 序 | 文件 | 发到 | 字符 | 内容 → 对应谁的原话 |
|---|---|---|---:|---|
| 1 | `01_AC.md` | Meta Review 下(公开) | 2175 | 收窄后的主张 + 四条不主张 → `AC.4`;指向四张表的去向 |
| 2 | `02_27bE.md` | Reviewer 27bE | 3851 | **Table 6** 操作点前因子经精确核校验 → `R27bE.1`;**Table 17** base sweep → `R27bE.2`;8B / 1.485B → `R27bE.2/.5` |
| 3 | `03_Dz6s.md` | Reviewer Dz6s | 4849 | **Table 21** QuALITY 2,086 真实文档题 → `RDz6s.1`;**Table 19** YaRN leverage(含不利的 NTK 行)→ `RDz6s.2`;FMRoPE + exact-range → metareview 第一条件 |
| 4 | `04_zWsa.md` | Reviewer zWsa | 2561 | Oka **自己** §6.3 承认 L_target 是 "practical limitation" → `RzWsa.1`;零参数阶梯 → `RzWsa.2` |

**顺序:AC → 27bE → Dz6s → zWsa。** 四个表单可同时开着,先后不影响。

**标题(终稿)**

- AC:`What this cycle leaves us claiming, and what it does not`
- 27bE:`Two results from the submitted version, and the scale evidence`
- Dz6s:`Two submitted tables bearing on your first two concerns`
- zWsa:`Two additions, both checkable against the sources`

标题一律描述内容。**不含 "should have"、"posted where"、"confidential" 等自贬或流程性措辞** —— 那会把我们的疏漏或发帖机制做成头条。

## PC 规则(已核,2026-07 官方澄清)

- "rebuttal" 与 "official comment" 两个按钮**等价**
- **允许多条,合计上限一万词以上**
- metareview 的回复应发在**审稿人能看到**的地方(我们首轮走了 AC-Confidential,故补公开版;但这一点只在正文第一句轻描,不进标题)
- **不得放任何链接**;不得含身份信息
- **AC 在讨论期不回复作者是常态**,其沉默不携带信号

## 四条硬约束(逐项已复检)

1. **不重复已发内容。** 对 AC 用五份合集比对(他读过全部),对审稿人用各自那份。零重复。
2. **不提无人问及的话题。** in-window / 82.16 / +0.0381 / waterbed / unseen-task / 32K 可用性 —— 四份中出现次数全为 0。
3. **数据来源准确。** 引用的每个表号已用作者下载的提交版 PDF 逐个定位到页码。
4. **总-分结构 + 沟通感。** 三份审稿人稿开头均为可核的自陈:回头对着评审重读自己的回复,发现原稿里有该指没指的东西。

## 表号(提交版 PDF 逐个核验)

| 表 | 页 | 内容 |
|---|---:|---|
| 3 | 8 | Training-time and inference-time positional optimization |
| 5 | 16 | Functional surrogate validation(12 配置) |
| 6 | 18 | Collision-score validation of the operating-point prefactor c |
| 12 | 29 | 750M continued pretraining |
| 17 | 32 | Base sweep(video DiT) |
| 18 | 33 | MLA validation(3-seed) |
| 19 | 33 | Matched-scale YaRN leverage at L=256 |
| 20 | 34 | Multi-scale raw length generalization |
| 21 | 35 | QuALITY QA(454M, n=2086, single seed) |
| 23 | 37 | EVQ-Cosh LoRA on LLaMA-3-8B-Instruct |

## ⛔ 红线

- 三种子 exact-range `−0.3159/−0.1949/−0.1674` 台账标 `CONDITIONAL`,**不得外引**。四份中均无。
- **Table 19 的种子数在提交版中未声明**,不得写"three seeds"。已改为"with the dispersions shown there"。
- 不引入任何新实验或未发表对象;不含链接;不要求任何人改分或重新权衡他人评审。
- 32 参数基线的历史标签、YaRN 实现版本:被直接问到才答,见 `../DISCUSSION_PHASE_FOLLOWUP_PLAYBOOK.md` §6。

## 已否决

- **LeRoPE(arXiv 2607.10134)整段已删。** 我们自己的三级阶梯是"每臂零参数、构造上同预算";LeRoPE 的 frozen 臂是 learn-then-freeze,**是更弱的对照**。拿弱证据撑强证据,论证方向反了,且其"全学 100% vs 冻结 63.6%"藏着"学习更强"的反驳。→ camera-ready 必引;若 27bE 再提学习基线 confound,届时作为答问拿出。笔记见 `../theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md`
- **"新颖性标准"类比论证(AdamW/YaRN)** —— 那是在告诉审稿人该用什么标准判断,对信心 5 的审稿人只会激怒
- **waterbed / O(τ⁴) 有界 in-range 代价** —— 原稿 `a1_proofs.tex` §waterbed-proof + Table 20 属实且有力,但防的是没人发动的攻击。转入 playbook 备答

## 交接段(四份统一)

结尾一律为:**这轮工作因这些评审而存在 → 我们守到期限结束 → 有疑问宁可现在答掉,不愿它悬着。**

不含任何责备、催促、或对沉默的暗示。
