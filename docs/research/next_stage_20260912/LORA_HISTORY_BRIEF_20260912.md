# LoRA 微调简报：OLMo-2-1.485B 与 Llama-3-8B（供外部模型研究微调方案）

日期：2026-09-12。自包含文档：外部模型无需访问仓库。所有数字来自仓库内权威 owner（路径与 SHA 见 §5），历史协议按记录时点读取。

---

## 0. 背景与核心问题

**论文背景**：Beyond the Base: Exponent Allocation in RoPE（ICLR 2027 投稿）。核心对象是 RoPE 内部指数分配：EVQ-Cosh 构造把 RoPE 的 inv_freq 表替换为端点锚定的 Cosh 分位数表（或 midpoint 取样），不改变注意力算子、不增加参数。学习期训练（scratch）与成熟模型适配（LoRA）是两条应用线。**适配线是当前瓶颈，也是本简报的主题。**

**历史结果的矛盾**：
- Llama-3-8B LoRA：长端 PPL 大幅改善（32K：991→128），但**灾难性遗忘与下游退步未解决**（检索归零、注册 QA 门负结果）；
- OLMo-2-1.485B（rebuttal 阶段）：用"任务监督 + EOS 终止"的两阶段训练**部分解决读出**（8K 18%→98%、16K 0%→60%），但其协议高度特化（物理 4K + 目标区相位暴露）。

**这条线的论文利害**：当前稿内每句 8B 相关论述都强制携带 E5/E7 负披露（PPL 收益与能力崩坏绑定陈述）；OLMo 正结果则被协议特化（物理 4K + 相位暴露 + 合成任务）限定。若新方案能在 OLMo 上先做到"长端收益 + 能力门全过"，再把协议推广到 8B，适配线就从"机制发现 + 强制披露"升级为可部署的正结果——这是衡量本方案价值的标准。

**待解的三个问题**：
1. **EVQ 表本身在适配场景是否最优**：两个模型的适配表从未扫描，且**两模型身份不同**——8B 全部用 τ=1.414 midpoint Cosh（=128/√8192），OLMo-2 全部用 τ=2、u_k=k/K 网格（=128/√4096）；各自继承经验规则 τ=d/√(物理训练窗)，τ、网格取样（midpoint vs k/K）、anchoring、native-relative 构造在适配场景从未比较；scratch 训练证据（M4 factorial）显示 1.25×Cosh 在 10/12 配置胜过 reference Cosh——适配场景的最优表是开放问题。
2. **微调怎么安排最好**：目标模块（FFN 从未训练）、rank/α、学习率、损失（全序列 vs answer-only 两个极端都试过、各有失败）、训练数据配比、步数——均未系统化。
3. **路径**：先在 OLMo-2-1.485B 上做稳，再扩展到 Llama-3-8B。硬件 RTX 5090 32GB 为主力（可两台分工训练/评测），另有 RTX Pro 6000 96GB 数台辅助。

---

## 1. 硬件与软件环境（实测）

| 项 | 值 | 来源 |
|---|---|---|
| 依赖版本 | torch 2.8.0+cu128、transformers 4.57.6、peft 0.17.1、trl 0.24.0、accelerate 1.10.1、datasets 4.5.0 | requirements-lock.txt |
| 加速组件 | Liger fused linear CE（已用）、FlashAttention only、torch.compile（max-autotune-no-cudagraphs，已用） | RULER-mix 协议 |
| 8B bf16 权重 | ≈16GB；KV ≈128KB/token（GQA 8 KV heads）：32K 上下文 ≈4GB | 模型结构 |
| 8B 评测峰值 | 20.4GB（Pro 6000，32K 前向 + lm_head 1024-chunk） | temporal 三臂评测回执 |
| 8B LoRA 训练吞吐 | **8,383 tokens/s**（micro batch 2 × 8K 物理、flash、liger；33,816,576 tokens / 4,033s） | RULER-mix 训练回执 |
| 已知 OOM | 8B 32K 评测 4 路并发时 32GB 级卡 OOM；**8B 32K 评测必须串行** | RULER-mix execution_incident |
| OLMo-2-1.485B | bf16 权重 ≈3GB；MHA（KV 同为 ≈128KB/token） | 模型结构 |

5090 32GB 推断：8B LoRA 训练 8K 物理（micro 2）可行（同期 rebuttal 实验在 5090 32GB 环境运行：oracle 训练器目录名 `olmo2_allocation_oracle_5090`、E6 记录"5090 causal decomposition"；RULER-mix 的运行卡身份未在其 JSON 中记录，吞吐 8,383 tok/s 为该卡实测）；16K 物理训练需梯度检查点或 micro 1（未测）；8B 32K 训练从未尝试（需 Pro 6000 级验证）。OLMo-2-1.485B 在 32GB 上全流程宽裕（可支持 16K–32K 物理训练与大批量）。

---

## 2. 历史实验档案

### 2.0 历史时间线与问题链（先读这张表）

每一代实验都是为了修上一代暴露的问题；下表是完整迭代链，后续小节按实验展开细节。

| 时间 | 实验 | 得到了什么 | 暴露了什么新问题 |
|---|---|---|---|
| 07-12/15 | 8B 一代：300 步 LongAlpaca **全序列 SFT**（EVQ vs Native 两臂） | 长端 PPL 大赢（32K：1493→991/128） | **能力崩坏**：S-NIAH/passkey 归零、QA16K 注册门负结果；native-vs-midpoint 非同量化器混杂（E5/E7） |
| 07-14/15 | 便宜修复探索 | 机制资产：attention 找得到证据（hit@16 64% vs 19%）、gold 块因果责任 ×4.51 | 50 步 micro、YaRN×2/×4、稀疏转换**全部无效**——便宜修复不存在（E6） |
| 07-26 | 8B 二代：516 步 RULER-mix **answer-only + 自然回放** | 长端 RULER 改善（16K 0→14.0%） | **自然文本遗忘**（8K 7.95→12.27）；匹配 Native-LoRA temporal 未跑 |
| 07-25 | OLMo released-RoPE 零训练换表 | 16K PPL 182.7→159.6（126/128 文档同向） | 无训练仍不够，适配线继续 |
| 07-28 | OLMo EOS 两阶段（物理 4K + 目标区相位暴露 + answer+EOS 监督） | **读出成功**：8K 18%→98%、16K 0%→60% | **协议特化**：合成任务 + 虚拟位置暴露，对通用指令场景的迁移性未知 |
| 07-29 | OLMo Q/K-only selective 续训 | Q/K 承担大部分读出修复（8K F1 0.079→0.215），V/O 冻结逐位不变 | FFN 的作用仍未知（从未训练） |
| 08-25 | OLMo coadaptive oracle（表与 Q/K LoRA 联合学习） | 尾部 NLL 改善 | **gate false**：全序列 NLL 变差、2Wiki 持平——成熟模型"学表"不是免费午餐 |

### 2.1 Llama-3-8B，300 步 LongAlpaca LoRA（全序列 loss）——第一代协议

**训练合同**（`experiments/lora_evq_v2/train_evq_lora.py`，git 历史严格锁版）：
- 模型 Meta-Llama-3-8B-Instruct；max_seq_len 8192；max_samples 8000（**LongAlpaca-12k** 指令数据，instruction/input/output 格式，数据 manifest SHA `1a6108…`）
- LoRA **r=64、α=128、dropout 0.05、targets 仅 q_proj/k_proj/v_proj/o_proj**（FFN 未挂）
- **LR 1e-4**、warmup 60、weight decay 0.01、max_grad_norm 1.0；batch 2×4=8；**max_steps 300**；bf16 全精度（无 4-bit）；compile
- **损失 = 全序列 LM（labels = all non-padding tokens，无 answer masking）**
- 双臂：native_geo（base 500000，端点原生）vs evq_cosh（**τ=1.414 midpoint**，head_dim 128）
- 注意混杂：native 端点 vs midpoint Cosh **非同量化器**，测的是"预训练 native 的 conversion"，非干净形状对比（此披露在论文内为强制句）

**三臂评测**（8/16/32K，arxiv/federal_register/stackoverflow 三域 temporal holdout）：

| 臂 | 8K PPL | 16K PPL | 32K PPL |
|---|---|---|---|
| Geo base（未训） | 7.95 | 150.5 | 1492.9 |
| Geo+LoRA | **6.82** | 108.96 | 991.48 |
| EVQ+LoRA | 10.07 | **24.07** | **127.91** |

**能力崩坏（核心问题）**：
- S-NIAH top-1：56.67%→3.33%→0%；passkey exact：100%→0%→0%（E5 行；归零数字唯一在库 owner 即此行；⚠ 原始 REPORT.md 不在库，臂/长度归属无法复核，引用时保持该行的原样口径）
- 注册 QA16K 三臂门**负结果**（E7，`results/qa16k_three_arm_s42_20260715/summary.json`）：task-macro F1 EVQ-LoRA 0.1126 < Native-LoRA 0.2110 < Base 0.2309；EVQ−Native=−0.0984，CI[−0.1297,−0.0697]；**缺口集中在 ≤8K（−0.334）**，>8K 全臂地板
- official-YaRN ×2/×4 恢复 pilot：**什么都没恢复**

### 2.2 Llama-3-8B，516 步 RULER-mix（answer-only + 自然回放）——第二代协议

**训练合同**（`llama8b_matched_ruler_mix_20260726.json`）：
- 物理长度 8192（无虚拟位置）；训练行 1376 = 13 个 RULER 任务×96 行 + 128 行自然回放；3 epochs；516 optimizer steps；33.8M tokens；全局 batch 8（micro 2×累积 4）；seed 20420726
- LoRA 同 2.1（r64/α128/qkvo）；**损失 = answer_only（RULER 家族）+ natural replay**；liger fused CE
- 训练实测 8,383 tok/s；验证 NLL：native 0.781→0.385、EVQ 1.788→0.409

**RULER official macro**（20 行/格）：

| 臂 | 8K | 16K | 32K |
|---|---|---|---|
| untouched native | 0.918 | 0.000 | 0.000 |
| Native+LoRA | 0.944 | 0.0029 | （32K 并发 OOM 未完成） |
| EVQ+LoRA | 0.776 | **0.140** | 0.000 |

**遗忘证据**：EVQ+LoRA 训后 temporal NLL 8K=2.507（PPL 12.27）vs untouched 2.073（7.95）——**自然文本 8K 域明显退化**；匹配 Native-LoRA 的 temporal 评测未跑。

### 2.3 Llama-3-8B 机制探测（解释"为什么 PPL 好了任务还是崩"，E6）

- 50 步 micro-tune：16K 转换前后均 0%——**便宜修复不存在**
- attention 门控：EVQ hit@16 中位 64.06% vs Geo 18.75%（10/10 配对胜）——**注意力找得到证据，读出失败**
- 因果：移除 gold 块，EVQ answer NLL +1.5055（×4.51 似然比）vs Geo ≈0；首 token gold rank 2,043（EVQ）vs 33,775（Geo）——**EVQ+LoRA 确实在用远端来源**
- 稀疏转换 pilot 负（DiD −0.666 偏 Geo）；forced-gold −0.034（零效应）
- ⚠ 历史 KV 生成器 bug：名义 16K 实为 6,827 tokens，旧"Geo 95% KV@16K"作废，不得引用

### 2.4 OLMo-2-1.485B：Stage-A(QKV/O) → Q/K-only 阶段（2Wiki 读出）

**协议**（`olmo2_qk_phase_adaptation_20260729/metrics.json`；表身份见论文附录 a6）：
- **OLMo-2 EVQ 臂表身份：u_k=k/K 标准网格、τ=2、base 500000，频率张量整个续训期固定**（注意：与 8B 的 τ=1.414 midpoint 不同——跨模型不是同一构造）
- Stage-A：QKV/O LoRA 父适配（r64/α128，300 步）；Stage-B：**Q/K-only 继续**（可训 8.39M 参数，继承 V/O 冻结且逐位不变），300 步、全局 batch 8、**LR 5e-5**、warmup 20
- 数据家族调度 [phase, phase, natural]；**物理训练 ≤4096**，虚拟位置暴露至 16K；answer+immediate-EOS 监督；单 seed

**Held-out 2Wiki**（200 例/格，token F1）：

| 阶段 | 4K | 8K | 16K |
|---|---|---|---|
| Stage-A 父（EVQ） | 0.086 | 0.079 | 0.009 |
| Stage-A 父（Native） | 0.087 | 0.002 | 0.003 |
| QK 相位适配后（EVQ） | **0.248** | **0.215** | **0.086** |
| QK 相位适配后（Native） | 0.260 | 0.0007（59% 空预测） | 0.0（100% 空预测） |

要点：**Q/K-only 继续训练承担了大部分读出修复**，且 EVQ 臂 16K 仍存活（0.086）而 native 臂塌空；terminal-EOS 率修复到 ~1.0。

### 2.5 OLMo-2-1.485B：EOS 两阶段（+100 query-gap → +32 answer+EOS）

**协议**（`evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json`）：
- 每次 backward 物理至多 4096 tokens；**目标区相对相位暴露**（最终 32 步数据中最大观察 position ID 16257）
- 两阶段各锁数据流/种子/优化器/步数；300 步父适配器起步（EVQ/Native 各一套，SHA 在案）
- 主终点 = 严格自回归完整答案串 + 终止 EOS，贪心解码，每长度 100 提示，单训练 seed

**结果**（正确数/100）：

| 臂 | 4K | 8K | 16K |
|---|---|---|---|
| EVQ | 100 | **98** | **60** |
| Native | 95 | 18 | 0 |

失败分类：native 8K 82 例检索错误、16K 100 例全错；EVQ 16K 40 例检索错误（其余正确）；EOS 终止率全格 1.0。

### 2.6 OLMo-2-1.485B：coadaptive oracle（A28，成熟 z 联合学习——负 gate）

- 联合 phase-oracle + z 学习：注册的 all-shells-improve 门 **false**；z 坐标最大移动仅 0.0013
- 学习表 vs native（匹配自洽）：tail512 NLL 改善（8K −0.039、16K −0.088）但**全序列 NLL 变差**（+0.038/+0.022）；完整 200 例 2Wiki ≈0 差异
- 含义：成熟模型上"学表"不是免费午餐——读出/尾部改善与全序列代价并存

### 2.7 对照上限：750M 全参数续训（LoRA 的容量参照）

同起点 2K checkpoint 全参数续训 500M tokens @4K：16K PPL 45.1→24.4；8K passkey answer-token exact 0/40→31/40（协议含 3% passkey + 10% 下游混合）。**全参数 + 大数据量没有遇到 8B LoRA 的能力崩坏**——是判断"LoRA 容量/数据缺口"的关键参照。

---

## 3. 已试 / 未试矩阵

| 维度 | 已试 | 未试 |
|---|---|---|
| 目标模块 | qkvo（8B 两代）、QKV/O→Q/K（OLMo）、Q/K+表联合（oracle，负 gate） | **FFN（gate/up/down）从未训练** |
| rank/α | 64/128 唯一 | 其他 rank、α/r 比例、DoRA/rsLoRA 等 |
| 学习率 | 1e-4（8B 300步）、5e-5（OLMo stage-B） | 系统扫描、调度形状 |
| 损失 | 全序列 SFT（8B 一代，检索崩）、answer-only+replay（8B 二代，自然文本遗忘）、answer+EOS（OLMo，读出成功） | 全域混合比例、NLL+任务联合、格式/EOS 正则的系统组合 |
| 数据 | LongAlpaca 8K 样本；RULER 家族 1376 行；OLMo 合成相位家族 | 数据配比系统化、规模扩展、真实长文档 QA、回放比例 |
| 步数/规模 | 300 / 516 步（33.8M tokens 顶配） | 数亿 token 级 LoRA 训练；多 epoch 曲线 |
| 频率表 | τ=1.414 midpoint 唯一；native 端点对照（非同量化器） | **τ 扫描、anchored vs midpoint、BM/native-relative、同量化器干净对照** |
| 训练物理窗口 | 8K（8B）、4K+相位暴露（OLMo） | 16K/32K 物理训练 |
| 修复 pilot | 50 步 micro（无效）、official-YaRN ×2/×4（无效）、稀疏转换（负） | —（不要重提这三条） |

---

## 4. 关键张力（方案必须回答）

1. **损失两难**：全序列 SFT → 检索崩坏；answer-only+replay → 长端 RULER 有但自然文本 8K 遗忘（7.95→12.27）。OLMo 的 answer+EOS 在特化协议下成功。损失怎么组合才能同时保住：自然文本 PPL、窗内 QA、长端检索、终止行为？
2. **FFN 缺口**：FFN 从未训练是已知缺口（用户判定为 top1 优先级）；但 OLMo 证据显示 Q/K 学习承担大部分读出修复，而 oracle 联合学习显示全序列代价。FFN 加入帮助长端还是加剧遗忘？需要受控对照。
3. **EVQ 是否最优**：适配场景的表是继承默认值且从未扫描——8B 用 τ=1.414 midpoint，OLMo 用 τ=2、k/K 网格（两者均符合 τ=d/√训练窗 经验规则，但互相不可比）；scratch 证据指向 1.25×Cosh 更优；且历史 8B 对照非同量化器（native 保留预训练端点 vs midpoint Cosh）。方案应决定：先扫表还是先稳协议、用什么对照消除量化器混杂。
4. **容量/数据缺口**：LoRA 顶配 33.8M tokens vs 全参数 500M 成功——差 15 倍。这是能力崩坏的候选主因，与 FFN/损失假设竞争。
5. **跨模型外推**：OLMo 成功要素（相位监督、EOS、物理 4K）哪些是任务特异、哪些可迁移到 8B 通用指令场景？

---

## 5. 数据与代码身份

| 对象 | 位置 |
|---|---|
| 8B 三臂评测（300步产物） | `data/curated/lora_longalpaca_temporal_s42_20260712.json`（adapter SHA：evq `8ea042…`、geo `0e7efa…`） |
| 8B 来源使用 | `data/curated/llama8b_causal_source_use_s42_20260714.json` |
| 8B RULER-mix | `rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json` |
| 8B QA16K 注册门 | 结果文档 `docs/exp/2026-07/2026-07-15_lora_qa16k_three_arm_results.md`（原始 `results/qa16k_three_arm_s42_20260715/summary.json` 已在仓库精简时清除，数字经本文档与 REVIEW E7 行双源核对一致）；披露 `rebuttal/EXPERIMENT_THEORY_REVIEW_20260720.md` E5–E7 |
| 8B 训练器（历史锁版） | `experiments/lora_evq_v2/train_evq_lora.py` @ git `6b636e5b^`（strict-legacy 合同内嵌） |
| OLMo EOS 两阶段 | `rebuttal/rebuttal_0723/theory_results/evq_query_gap_realized_eos32_20260728/FINAL_METRICS_AND_LINEAGE.json` |
| OLMo QK selective | `rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json` |
| OLMo oracle | `paper-2027/research/attention-aware-retrofit/evidence/COADAPTIVE_ALLOCATION_ORACLE_RESULTS_20260825.json` |
| OLMo z 联合训练器 | git `main_0726:rebuttal/rebuttal_0723/experiments/olmo2_allocation_oracle_5090/train.py` |

**可复用的评测与数据资产**（新方案的稳定判据应从这里取材，避免新造仪器）：

| 资产 | 说明 |
|---|---|
| Temporal holdout | 三域（arxiv/federal_register/stackoverflow 2026）×8 packs；一次 32K 前向、8/16K 为精确因果前缀聚合（效率合同在案） |
| RULER 面板 | 13 任务 ×20 行/格，8/16/32K 贪心；训练侧混入配方 = 96 行/任务 + 128 行自然回放（二代协议） |
| 2Wiki held-out | 200 例/格，确定性干扰项填充；token F1 / strict exact / terminal EOS 三指标；非 LongBench 榜单口径 |
| EOS 读出终点 | 每长度 100 提示，完整答案串 + 终止 EOS；失败四分类（检索错/不完整/格式/EOS） |
| Passkey / S-NIAH | 一代崩坏所用能力仪器（臂/长度归属警示见 §2.1，复用时先重定义归属） |
| 训练数据流 | LongAlpaca-12k（8K 样本切片）、RULER-mix 配方（1376 行）、query-gap/相位流（SHA 在 lineage JSON） |
| 代码 | 历史锁版训练器（git `6b636e5b^`）、oracle 训练器（`main_0726`）、历史 lora_evq_v2 全套 eval 脚本 |
| 术语 | query-gap = 经 position ID 改变问句—证据相对距离的合成检索任务；phase = 目标区相对相位暴露的数据家族（物理短窗 + 虚拟位置）；natural = 自然文本 |

**明确约束（防重蹈）**：不重提 50 步 micro-tune、official-YaRN ×2/×4、稀疏转换三个已判负的 pilot；不引用旧"Geo 95% KV@16K"（KV bug 作废）；8B 的 native-vs-EVQ 对照若复用必须标注非同量化器或改造成同量化器；OLMo（τ=2、k/K）与 8B（τ=1.414、midpoint）表身份不同，跨模型比较不得说成同一构造；S-NIAH/passkey 归零序列的臂/长度归属不可复核，按 REVIEW E5 原样引用；OLMo 两阶段正结果绑定其特化协议，不得说成通用修复。

---

## 6. 交付目标

一份新的 LoRA 微调方案：**先在 OLMo-2-1.485B 上稳定**（判据明确：自然文本 PPL 保留 + 窗内 QA 不退 + 长端读出改善 + EOS 正常），**验证后扩展到 Llama-3-8B**（同判据 + RULER/自然 QA 注册门）。覆盖：目标模块（含 FFN 的受控引入）、rank/α、LR 与调度、损失组合、数据配比与规模、步数预算、频率表选择（含"EVQ 是否要重扫"的裁决）、5090 32GB 上的显存/吞吐可行性，以及每个选择失败时的回退。
