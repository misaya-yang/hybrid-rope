# 下游评估实验计划（2026-03-04）

> **核心问题**：目前所有 EVQ 证据 = PPL + passkey（合成检索）。Reviewer 会说"只有 PPL 和 toy task"。需要 real NLP downstream evidence。
> **约束**：350M from-scratch 模型，不能做 QA/摘要（需要 instruction tuning）。但 PE 论文通常只用 PPL + 合成任务，我们需要做得比同类论文更多，但不需要做到 7B 级别。
> **核心策略**：利用已有 checkpoint（零额外训练），多维度评估。新训练实验只做最高 ROI 的。

---

## 0. 当前已有证据清单

| 证据 | 强度 | Reviewer 印象 |
|------|------|--------------|
| PPL@8K-16K 6/6 全胜 | A | "PPL 谁都有" |
| EVQ+YaRN 8K=100% (6-seed) | A+ | "passkey 太简单" |
| 750M retrieval divergence | B | "只有 passkey" |
| 128-tok 对标 DAPE 完胜 | A | "PE-dominant regime，实用吗？" |
| Passkey mix +10pp | B- | "幅度不大" |

**缺什么**：多域 PPL 泛化、非 passkey 的检索任务、real text 上的长程依赖证据。

---

## 1. Tier 1：零额外训练，直接用现有 checkpoint 评估

### 1.1 多域 PPL 泛化（⭐ 最高优先级，最容易做）

**目的**：证明 EVQ 的 PPL 改善不只在 FineWeb-Edu 上，跨域泛化。

**数据集**：

| 数据集 | 特点 | 获取方式 |
|--------|------|---------|
| **PG-19** | 长书籍，文档平均 69K tokens | HuggingFace `pg19` |
| **Arxiv** | 学术论文，结构化长文本 | RedPajama 或 DAPE 使用的 split |
| **GitHub Code** | 代码文件，长程结构依赖 | The Stack / StarCoder data |
| **Books3** | 小说/非虚构，DAPE 直接对标 | Pile 子集 |

**方法**：
```
对每个域的 test set：
1. 取 100-200 篇长文档（>8K tokens）
2. 用 sliding window eval（window=2048, stride=2048）
3. 计算 PPL@{2K, 4K, 8K, 16K}
4. 对比 Geo vs EVQ（已有 5% 和 10% checkpoint）
5. 对比 Geo+YaRN vs EVQ+YaRN
```

**预期结果**：如果 PPL 改善在 PG-19/Arxiv/Code 上也成立，reviewer 无法说"只在一个域上 work"。

**工作量**：下载数据 + 写 eval 脚本，~2-3h。纯推理，无需训练。

**用哪些 checkpoint**：
- 5% mix: Geo seed={42,123,7}, EVQ seed={42,123,7} — 共 6 个
- 10% mix: 同上 — 共 6 个
- 选 5% 的就行（更接近真实训练分布）

### 1.2 Key-Value Retrieval（Lost in the Middle 风格）

**目的**：证明 EVQ 的检索优势不限于 passkey format。

**任务设计**：
```
输入：N 个 JSON key-value pairs，每个 key/value 是 random UUID
      "Find the value for key: <target_key>"
      target 的位置随机分布在 context 中

评估：模型在 target 位置的 NLL（teacher-forcing），类似 passkey 方法
```

**长度设置**：
- N = {50, 100, 200, 500} pairs（对应 ~{1K, 2K, 4K, 10K} tokens）
- target position: {front, middle, back}（测 Lost-in-Middle 效应）

**变体**：
- **Multi-key**: 同时查询 3-5 个 key，类似 RULER multi-needle
- **Variable-length values**: value 长度从 1 token 到 50 tokens

**工作量**：生成数据 + NLL eval，~3-4h。纯推理。

### 1.3 RULER-style 合成任务（扩展版）

**目的**：展示 EVQ 在 passkey 之外的合成检索任务上也赢。

**任务列表**（从 RULER benchmark 选取对 350M from-scratch 可行的子集）：

| 任务 | 描述 | 评估方式 |
|------|------|---------|
| **Single NIAH** | 标准 needle-in-haystack | 已有（= passkey）|
| **Multi-NIAH** | 5 个 needle 同时 | 已有（11.6.7），需扩大试次 |
| **Multi-Key KV** | 多对 KV 同时查询 | NLL gap |
| **Variable Tracking** | "X was in city A, then moved to B, then to C. Where is X?" | NLL gap on final city |
| **Common Words** | 统计某个词在长文本中出现次数 | NLL on count token |

**工作量**：~4-5h 数据生成 + eval。

### 1.4 长程 Cloze（Long-Range Dependency）

**目的**：最接近"real NLP"的无 instruction 评估。证明 EVQ 在真实文本的长程依赖上更好。

**方法**：
```
1. 取 PG-19 长文本
2. 找出距离 >4K tokens 的 coreference pairs（e.g., 人名首次出现 → 4K 后再次出现）
3. Mask 第二次出现，计算模型在该位置的 NLL
4. EVQ 应该在长距离 coreference 上 NLL 更低
```

**简化版（更容易实现）**：
```
1. 取长文本前 N tokens 作为 prefix
2. 取 position P 处的 token 作为 target
3. 计算 NLL(target | prefix[:P])
4. 对比不同 P 值下 Geo vs EVQ 的 NLL 衰减曲线
```

这本质是 "positional PPL profile"——不是全局 PPL，而是 per-position PPL。EVQ 在远距离位置的 PPL 应该更低。

**工作量**：~2-3h。纯推理。

---

## 2. Tier 2：需要少量新训练（高 ROI）

### 2.1 QA Mix 训练（类比 Passkey Mix 思路）

**核心思路**：passkey mix 证明了"混合检索信号到训练数据中可以隔离 PE 的泛化效应"。同样的方法可以用于 QA。

**数据设计**：
```
90% FineWeb-Edu（语言建模）+ 10% 合成 QA
QA 格式：
  Context: [长文本，2K tokens]
  Question: [关于文本某个位置的事实性问题]
  Answer: [从文本中可直接提取的短答案]
```

**QA 数据来源**：
- SQuAD-style extractive QA（答案是 context 中的 span）
- 自动生成：用 GPT-4 从 FineWeb 文档生成 QA pairs
- 或更简单：随机选一个句子作为"答案"，prompt 是"What sentence appears at position X?"

**评估**：
- 训练长度 2K，评估 4K/8K/16K 的 QA accuracy
- 对比 Geo vs EVQ vs EVQ+YaRN
- 预期：EVQ 在 OOD 长度的 QA 显著优于 Geo（类似 passkey 的 +40pp 效应）

**工作量**：数据准备 ~4h + 训练 6 runs ~6h = ~10h 总计
**GPU 需求**：3 seeds × 2 methods × 100M tokens = 现有 passkey mix 规模

### 2.2 Code Completion Mix

**思路**：代码有天然的长程依赖（函数定义 → 远处调用）。

```
90% FineWeb-Edu + 10% code（The Stack Python subset）
评估：在 code 文件的 4K/8K 位置处的 token prediction accuracy
```

代码的优势：位置信息对代码生成极端重要（变量名、函数签名需要精确 recall）。

**工作量**：~8h（数据 + 训练 + eval）

---

## 3. Tier 3：高影响但需要较大投入

### 3.1 350M + Instruction Mix（最激进方案）

训练一个 350M 模型：80% FineWeb-Edu + 10% instruction data + 10% passkey。这样模型既能做 language modeling 又能做简单指令跟随，就可以在 LongBench 子集上评估。

**风险**：350M 可能 instruction following 能力不足，即使在 2K 内也做不好 QA。

**工作量**：~20h（数据准备 + 训练 + 大量 eval debugging）

### 3.2 扩大到 1.3B

用更大模型直接说明 EVQ scales。但 GPU 预算可能不够。

---

## 4. 推荐执行顺序

### Phase A：立即执行（醒来就跑，~8h 总工作量，零训练）

| 序号 | 任务 | 时间 | 产出 |
|------|------|------|------|
| A1 | 多域 PPL (PG-19 + Arxiv + Code) | 3h | Table: 4 域 × {2K,4K,8K,16K} × {Geo,EVQ} |
| A2 | 多域 PPL + YaRN | 1h | Table: 4 域 × {Geo+Y, EVQ+Y} |
| A3 | KV Retrieval (Lost-in-Middle) | 3h | Table: N={50,100,200,500} × {front,mid,back} × {Geo,EVQ,+YaRN} |
| A4 | Per-position PPL profile (PG-19) | 1h | Figure: PPL vs position，EVQ 曲线更平 |

**Phase A 产出**：2 个新 Table + 1 个新 Figure，纯推理零训练。

### Phase B：第二优先（需要少量代码，~4h，零训练）

| 序号 | 任务 | 时间 | 产出 |
|------|------|------|------|
| B1 | Multi-NIAH (5 needle, 40+ trials) | 2h | Table: multi-needle acc × length |
| B2 | Variable Tracking | 2h | Table: tracking acc × chain length |

### Phase C：第三优先（需新训练，~12h）

| 序号 | 任务 | 时间 | 产出 |
|------|------|------|------|
| C1 | QA Mix (3 seed × 2 method) | 10h | Table: QA@{2K,4K,8K,16K} 类比 passkey |
| C2 | QA Mix + YaRN eval | 2h | 如果 QA 也有超线性 → spotlight 级 |

---

## 5. 论文整合方案

### 如果 Phase A 全部正面：

正文新增 ~0.5 页（从 Appendix 节省空间）：

> **Section 5.X: Cross-Domain and Retrieval Evaluation**
>
> Table N: Multi-domain PPL extrapolation (PG-19, Arxiv, Code, Books3)
> - EVQ consistently improves PPL@8K-16K across all domains
>
> Table N+1: Key-Value retrieval at various lengths
> - EVQ+YaRN maintains near-perfect retrieval at 8K while Geo+YaRN degrades
>
> Figure N: Per-position PPL profile on PG-19
> - EVQ's PPL decay is significantly slower than Geometric

### 如果 Phase C QA Mix 成功：

这是 spotlight 级别的新结果。在 Section 5 加 0.3 页：

> **Section 5.Y: QA Mix — From Passkey to Question Answering**
>
> Applying the same mix-training paradigm to extractive QA, EVQ+YaRN achieves X% accuracy at 8K (4× extrapolation) vs Y% for Geo+YaRN, demonstrating that the superlinear synergy extends beyond synthetic retrieval to natural language understanding.

---

## 6. 实现注意事项

### 6.1 多域 PPL 的公平比较

- 用同一批 checkpoint（5% mix, 3 seeds）
- Sliding window eval 参数必须一致（window=2048, stride=2048）
- 不同域的 tokenization 可能不同（code vs text），确保用同一 tokenizer
- 报告每个域的 absolute PPL 和 relative Δ%

### 6.2 KV Retrieval 的实现

```python
# 伪代码
def generate_kv_task(n_pairs, target_position, max_len):
    pairs = [(random_uuid(), random_uuid()) for _ in range(n_pairs)]
    target_idx = int(target_position * n_pairs)  # front/mid/back
    target_key, target_value = pairs[target_idx]

    context = format_json(pairs)
    prompt = f"{context}\nThe value for key {target_key} is: "

    # Teacher-forcing: compute NLL of target_value tokens
    return prompt, target_value
```

注意：350M 模型不会"理解"指令。但 teacher-forcing NLL 不需要模型理解——只需要模型在给定 context 后，对 correct answer tokens 的概率是否更高。EVQ 的更好长程注意力 → correct answer NLL 更低。

### 6.3 YaRN 参数

- 保持 scale=8（和 passkey mix 实验一致）
- 对不同 eval 长度：scale=L_eval/L_train，所以 4K→scale=2, 8K→scale=4, 16K→scale=8

### 6.4 Per-position PPL Profile

```python
# 对长文本，计算每个 stride window 的 PPL
for doc in pg19_test:
    for start in range(0, len(doc), stride):
        window = doc[start:start+stride]
        ppl = model.eval_ppl(window, context=doc[:start])
        results[start // stride].append(ppl)

# Plot: x=position (in tokens), y=mean PPL
# EVQ 曲线应该比 Geo 更平（远距离 PPL 衰减更慢）
```

---

## 7. 预期结果与风险

| 实验 | 预期 | 风险 |
|------|------|------|
| 多域 PPL | EVQ 在所有域 -5%~-15% | 某些域可能无改善（如果该域不需要长程依赖） |
| KV Retrieval | EVQ+YaRN 在 N>100 时显著赢 | 350M 可能完全不会做 KV retrieval（连 2K 都不行） |
| Per-position PPL | EVQ 衰减更慢 | 差异可能不够大（需要很长文本） |
| QA Mix | 类比 passkey +30pp | 350M 的 QA 能力可能太弱，即使在 2K 内也接近随机 |

**最大风险**：350M from-scratch 模型在 KV Retrieval 和 QA 上可能完全没有能力（连训练长度内都做不好），这样 OOD 对比就没意义。

**缓解策略**：先跑 Tier 1 的零训练实验（多域 PPL + per-position profile），这些是最安全的。KV Retrieval 先在 2K 内测试，确认基线非随机后再测 OOD。

---

## 8. Prompt：启动 Phase A 实验

```
你需要在我们的 EVQ 研究中执行多域 PPL 和 KV Retrieval 评估。

已有 checkpoint 位置：
- 5% passkey mix, Geo: /root/autodl-tmp/evq_passkey_mix_5pct/350m_tau0_seed{42,123,7}/model.pt
- 5% passkey mix, EVQ: /root/autodl-tmp/evq_passkey_mix_5pct/350m_tau1.5_seed{42,123,7}/model.pt

任务 1: 多域 PPL
- 下载 PG-19 (test split), Arxiv (from RedPajama), Books3 (from Pile)
- 对每个 checkpoint 计算 sliding window PPL@{2K, 4K, 8K, 16K}
- 输出 CSV: domain × method × seed × eval_length → PPL

任务 2: 多域 PPL + YaRN
- 对 EVQ checkpoint 加 YaRN (scale=L_eval/L_train)
- 对 Geo checkpoint 加 YaRN
- 计算同样的 PPL

任务 3: Per-position PPL profile
- 对 PG-19 test 的 50 篇最长文档
- 计算 stride=2048 的 per-window PPL
- 输出: position × method × seed → PPL
- 画图：position vs PPL 曲线，Geo vs EVQ

任务 4: KV Retrieval pilot
- 生成 N={10, 50, 100} 的 KV pairs
- target 位置在 front/middle/back
- 先在 L=2K 内测试确认基线非随机
- 如果基线 OK，扩展到 L={4K, 8K}
```
