# Paper Handover (2026-04-27 evening seal)

**State**: 封板。今天大量 audit 修复 + 晚间补回 §5 Takeaway and limitations 内联段。可以提交。

---

## 当前 PDF 状态

| 项 | 状态 |
|---|---|
| Pages | 41 (主体 9 + Refs/Appendix/Checklist 32) |
| 主体严格 9 页 | ✅ Page 9 末以 "Takeaway and limitations.... natural follow-ups." 完整收尾 |
| File size | 2.1 MB |
| 0 undefined refs/citations | ✅ |
| 0 §?? 断引 | ✅ |
| 0 LaTeX errors | ✅ |
| 0 overfull boxes | ✅ |
| Anonymous | ✅ |
| Bib 完整 | ✅ 44 entries / 44 unique cite keys (zero waste) |

预期 review score: **7.5-8.0 / 10** (weak accept 区间中位~上限)

---

## 今天（2026-04-27）累计完成的工作

### 你白天用 Claude Code 做的（commits a358d6d → 9766136）

#### P0 真 bug（最高 ROI）
- **Table 5 caption metadata drift**: 修正 "3-seed mean" 为 "Geo/DAPE/EVQ seed 42; Learnable τ 3-seed (42/137/256)" — 防止 silent caption-data 不一致
- **Table 4 caption metadata drift**: 修正为 L_train=2048, s=8, 4×/6×/8× extrapolation（与实际 source data 一致）
- **750M LR drift**: Phase 15 = 2e-4 修正（不是 9F 的 3e-4）

#### P1 std insertion（关闭 reviewer 最常用扣分点）
- **Table 4 PK@8K**: 41±5%, 61±3%, 53±8%, 100±0% (3-seed mean±std)
- **Table 5 Learnable τ**: 181.2±1.3 / 437.9±12.2
- **Table phase11-leverage**: 全 6 cell ±std (from results/core_text/phase11/results_phase11_yarn.json)

#### P1 theory consistency
- a1:147 K=d_head/2 → K=d_rot/2 (MLA 通用)
- a1:548 α=1/d_rot cross-ref 扩展 (MHA via tau-scaling, MLA via mla-results)
- a1:189 Normalization-convention disambiguator
- a1:40 β=0 degenerate case 显式

#### P1 caption verbs softening (lower overclaim risk)
- "validation/vanishes" → "mechanism check / no longer present"
- "Waterbed verification" → "Waterbed illustration"
- "dramatic" → "substantial"
- "surpassing Geo+YaRN" → "surpassing matched-scale (s=4) Geo+YaRN" (abstract qualifier)
- checklist Q1 "validated" → "calibrated"
- "clearly" 删除

#### Stage B/C theory hardening
- MLA entry-point reconstruction
- SSH redaction (anonymity hardening)
- a2 propagation tail of P0-2 (L=2048)
- Progressive-section disambiguation
- Broader Impact unnumber (NeurIPS 惯例)

### 我（cowork）晚间补的（封板修复）

**关键问题**: Claude Code commit message 说 "06_limitations.tex header removed; takeaway merged inline at end of §5.4 Robustness paragraph"，但**实际没有内联合并**——06_limitations.tex 改成空 stub，§5 Conclusion + Limitations 内容**完全消失了**。

修复方案：
1. 在 §5 实验末尾（"concentrated at the PE layer"之后）加 `\paragraph{Takeaway and limitations.}` 段
2. 内容: "Allocation shape, not only range, is a third RoPE design axis; theory is exact conditional on C_app and τ* is semi-analytic. Production MLA, ≥1B training, and per-channel head-to-heads are natural follow-ups."
3. 为 fit 9 页边界的连锁压缩：
   - §5.5 Robustness subsection 降级为 \paragraph (saves vertical glue)
   - §5.4 Training-saturation 段第一句 "A standard objection is that..." 删除（meta-context 可省）
   - §5.5 Robustness 详细数字列表压缩为 "raw PPL/NLL/passkey/QA; full breakdown Appendix D"

最终验证：Page 9 末 §5 Takeaway 完整收尾，Page 10 直接 References。

---

## 已知未消除的攻击向量（rebuttal 阶段处理 / 留 GPU 实验）

### 🟢 0-GPU 可榨实验数据（明天上班最高 ROI）

| 项 | 性质 | 价值 |
|---|---|---|
| **每个 paper 数字 trace 到 results/ JSON** | grep + Python 重算 mean | +0.1-0.2，catch 隐藏 bug |
| **REPRODUCTION_GUIDE.md** | Table/Figure → script 映射 | +0.1-0.2 (reproducibility checklist) |
| **arXiv ID 全 bib 验证** | web search | +0.0-0.3 (防意外 reject) |
| **Tables 4/5 PPL 列加 std** | seed-level data 在 results/ 里 | +0.1 (Table 4 caption 已声明 PPL std deferred to camera-ready) |

### 🟠 GPU 路径（你说还有 ~15h budget）

| 实验 | 估 GPU | 收益 | 风险 |
|---|---|---|---|
| **Tuned YaRN best-of-grid (Geo/EVQ at s∈{2,4,8,16,32})** | 3-6h inference (无训练) | +0.5, 闭环"baseline 不公平"攻击 | LOW (评估已有 checkpoint) |
| **Geo+LoRA+LongAlign control** | 2-4h training | +0.3-0.5, 闭环 LoRA 攻击 | LOW |
| **τ=0.354 MLA sanity ablation** | 6-10h training | +0.3-0.5, 验证 d_eff=d_head 经验合理 | MEDIUM |
| **2-seed 1B-MLA replication** | 24-30h training | +0.5-0.8, 闭环 1B reversal single-seed | HIGH (超预算) |

**推荐路径**: tuned YaRN (3h) + Geo+LoRA (3h) + τ=0.354 (6h) = ~12h, buffer 3h。**不推荐 1B-MLA 重跑**。

---

## Repo 结构速览

```
paper/
├── main.tex                 主入口
├── main.pdf                 当前 41 页封板版 (2.1 MB)
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_theory.tex
│   ├── 05_experiments.tex   ⭐ 含 §5 Takeaway and limitations 内联段（晚间补）
│   └── 06_limitations.tex   空 stub (内容已合并入 §5.5)
├── appendix/
│   ├── a1_proofs.tex        含 PSD identity, q(x) 推导, normalization disambiguator
│   ├── a2_experiment_details.tex
│   ├── a3_supporting_results.tex
│   └── a4_supporting_experiments.tex
├── refs/references.bib      44 entries, 含 Hyperbolic-RoPE
└── tables/                  含 std-updated tables

scripts/2026-04/
├── PAPER_HANDOVER_2026-04-27.md           早上版本
└── PAPER_HANDOVER_2026-04-27_evening.md   ⭐ 你正在读的（最新）
```

---

## 最终累计修复总数

**6 轮 + 今天 1 整天 = ~70 处 hardening**

- Round 1-5 (我 cowork): ~38 处
- Round 6 (Claude Code 早间 audit gaps): ~15 处
- Round 7 (Claude Code 全天 audit): ~16 处（caption metadata drift、std insertion、theory consistency、caption verbs）
- Round 8 (我 cowork 晚间封板): 1 处（恢复 §5 Takeaway and limitations）

---

## 提交流程提醒

- Submission portal: contribution type 选 **General**（不要选 Theory）
- Subject area: Deep Learning - Architectures 或 Theory 子类
- Code release: 现勾 "Will be released upon acceptance"
- Final PDF: paper/main.pdf (2.1 MB, 41 pages)

---

如果需要再修，把 commit hash 给我，我重新 git pull + recompile + 全套 audit。
