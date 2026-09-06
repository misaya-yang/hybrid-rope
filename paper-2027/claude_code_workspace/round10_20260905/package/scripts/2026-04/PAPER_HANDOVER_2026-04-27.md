# Paper Handover: EVQ-Cosh NeurIPS 2026 (2026-04-27)

**Status**: 已封板，9 页主体合规，可提交。明天上班继续优化用此文档作为 context 入口（不需要重读对话历史）。

---

## 当前 PDF 状态

| 项 | 状态 |
|---|---|
| Pages | 40 (主体 9 + Refs/Appendix/Checklist 31) |
| Body 严格 9 页 | ✅ Page 9 末以 §5 "natural follow-ups." 收尾 |
| File size | 2.1 MB (≪ 50 MB) |
| 0 undefined refs/citations | ✅ |
| 0 §?? 断引 | ✅ |
| 0 LaTeX errors / overfull | ✅ |
| Anonymous | ✅ Author = "Anonymous Authors", PDF metadata Title/Author 空 |
| NeurIPS 2026 全合规 | ✅ neurips_2026.sty, 无 vspace 压缩, 15 题 checklist |
| 最新 commit | `2b33a59 paper: address audit gaps — c_pred(L,b), W vs C_app, normalization, Hyperbolic-RoPE, typos` |

预期 review score: **7.4-7.7 / 10** (weak accept 区间下限~中位)

---

## 已完成的累计修复（6 轮，~45 处）

### Round 1-2: surface fixes
- 8 处 §?? 断引、Geo+LoRA control 暴露、§4.5 自引用 typo
- §6 Conclusion 独立，9 页边界严格

### Round 3: theory hardening (P0/P1 from GPT-5.5 Pro audit)
- B1 Taylor truncation caveat at τ=4
- B3 q(x) probabilistic derivation (uniform separation prior)
- A1 PSD assumption + Theorem 1 hypotheses
- E1 figure caption 方向性 wording
- D2 plain-language proof ideas
- D1 D_Burg definition
- A4 forced branch metric clarification

### Round 4: A 类真 bug + B 类 sharpening
- A1 PE-dominant τ=4 vs 5.66 上下文歧义 (Table 5 vs Appendix Fig 10)
- A2 Table 3 "4K→8/12/16K" → L_train=512
- A3 §A.3 "20K" typo → 24K
- A4 §A.9 basin "±20% vs 2.3%" 算术冲突
- A5 §3.4 geometric limit 限定为 pure-tether 分支
- A6 Theorem 1 β=0 case 显式说明
- A7 A.16 vs Table 9 S_p 同名两定义 (重命名为 S_p^{mom})
- A8 Abstract "L^{-1/2} derived" 软化
- A9 §3.7 主文加 U(τ,L) 形式化定义
- B5 A.12 LoRA "predicted exactly" → "calibration check"

### Round 5: T1 + T2 batch
- 28-92% → 24-92% (a1:97)
- 13-15% MHA range → -13.3% specific (matched 3-seed)
- "exceeding Geo+YaRN" 加 matched scale s=4 限定
- main.tex 删除空 stub 04_predictions / 07_conclusion
- §6 加 1B reversal/d_rope=32/DiT 0.53× caveats
- §3 加 C_app validated at surrogate level only disclosure
- §A.3 "Two distinct dimensions" 段 (α=1/d_rot vs d_eff=d_head 显式区分)
- §3.7 加 negative control (pure geometric τ=0 falls 10-46% outside)
- §2 加 per-channel-method head-to-head future-work

### Round 6: Claude Code's audit-gap fixes (2b33a59)
- §3.7 + a1 basin: c_pred(L,b) 数值披露 (1.20 at b=500K → 0.80 at b=10K, L=4096)
- §3.4 Waterbed: clarify W vs C_app distinct functionals (W/C_app diverges 0.002→1.59)
- a1 §sec:chi2-load: "Normalization convention" paragraph 统一 S_χ²
- L_eff^J typo fix at a1:553, 565
- §A.3 d_rot → d_rope 一致化
- §2 加 Hyperbolic-RoPE (Dai et al. 2025, arXiv:2509.05218) name-collision disambiguation
- §5.4 raw PPL -46% 标注 single-seed 750M
- 验证 Selective-RoPE (arXiv:2511.17388) 与 EVQ 无关，正确未引

### Round 6 sealing (我做的最终 page-fit 调整)
- §4.5 Robustness 段进一步压缩 (saves 1-2 lines)
- §5 Conclusion 删除"theory exact + semi-analytic" 短句 (该信息已在 §3 epistemic tiers)
- §3.7 Basin 段精简 c_pred 表述 (保留实质)

---

## 已知未消除的攻击向量（rebuttal 阶段处理）

### 🔴 高 ROI experiment (需 GPU, 用户有 ~15h budget)

| 实验 | 估 GPU | 收益 | 风险 |
|---|---|---|---|
| **Tuned YaRN best-of-grid (Geo+YaRN, EVQ+YaRN at s∈{2,4,8,16,32})** | 3-6h inference (无训练) | +0.5 score, 闭环 Attack 5 | LOW (评估已有 checkpoint) |
| **Geo+LoRA+LongAlign control** | 2-4h training | +0.3-0.5, 闭环 LoRA 攻击 | LOW |
| **τ=0.354 MLA sanity ablation** | 6-10h training | +0.3-0.5, 验证 d_eff=d_head 经验合理 | MEDIUM |
| **2-seed 1B-MLA replication** | 24-30h training | +0.5-0.8, 闭环 1B reversal single-seed | HIGH (超预算 + 结果不确定) |

**推荐路径**：tuned YaRN (3h) + Geo+LoRA (3h) + τ=0.354 (6h) = ~12h，buffer 3h。**不推荐 1B-MLA 重跑**（高风险高成本）。

### 🟠 0-GPU 可榨实验数据（明天上班优先）

| 项 | 性质 | 价值 |
|---|---|---|
| **Tables 4, 5 加 mean±std** | 从 results/ seed log 提取，**0 GPU** | **+0.3-0.5** ⭐ 当前最大空缺 |
| **每个 paper 数字 trace 到 results/ JSON** | grep + Python 重算 mean | +0.1-0.2，catch 隐藏 bug |
| **REPRODUCTION_GUIDE.md** | Table/Figure → script 映射 | +0.1-0.2 (reproducibility) |
| **arXiv ID 全 bib 验证** | web search | +0.0-0.3 (防意外 reject) |

**单项最高 ROI: Tables 4, 5 std 提取**——seed-level 数据应该在 results/ 里，应该 1-2 小时可以做完。

---

## 数值验证已得到的核心数据

```
c_pred = √(45·Q_1(L,b)) 跨网格:

L      b         Q_1        c_pred    deployed c=1 偏差
128    10K       0.0317     1.194     -16.3%
1024   10K       0.0241     1.042     -4.0%
4096   10K       0.0141     0.797     +25.4%  ⚠️ 超出 ±20% basin
8192   10K       0.0083     0.611     +63.6%  ⚠️ 严重超出
128    500K      0.0301     1.164     -14.1%  (deployed primary)
2048   500K      0.0305     1.172     -14.7%  (deployed primary)
4096   500K      0.0288     1.138     -12.1%
8192   500K      0.0265     1.091     -8.4%   (MLA primary)
```

**结论**: 我们 deployed primary 都在 b=500K（c_pred 1.09-1.19, c=1 偏差 -8 to -15%, 在 ±20% basin 内）。Claude Code 已经在 §3.7 / §A.10 披露了这个 b-依赖。

---

## 跨机器协作建议

**重点**: Cowork mode (Claude desktop) 不能跨机器共享上下文。Claude Code (dev env) 可以读这个文件作为 context 入口。

**明天工作流**:
1. 公司电脑用 Claude Code agent team
2. 第一步: read this file (PAPER_HANDOVER_2026-04-27.md)
3. 第二步: 跑 0-GPU 数据榨取 (Tables 4, 5 std 提取最优先)
4. 第三步: 决定是否使用剩 GPU 跑 tuned YaRN baseline (推荐) 或 Geo+LoRA control

---

## Repo 结构速览

```
paper/                       LaTeX 源码
├── main.tex                 主入口
├── main.pdf                 当前编译版 (40 pages, 2.1 MB)
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related.tex       (Hyperbolic-RoPE 已加)
│   ├── 03_theory.tex        (b-依赖 disclosure 已加, c_pred 数值)
│   ├── 04_predictions.tex   (空 stub, main.tex 已不引用)
│   ├── 05_experiments.tex   (Tables 4,5 缺 std)
│   ├── 06_limitations.tex   (Conclusion+Limitations 合并)
│   └── 07_conclusion.tex    (空 stub)
├── appendix/
│   ├── a1_proofs.tex        (含 PSD identity, q(x) derivation, normalization paragraph)
│   ├── a2_experiment_details.tex
│   ├── a3_supporting_results.tex  (含 "Two distinct dimensions" MLA 段)
│   └── a4_supporting_experiments.tex
├── tables/                  独立 .tex 表格
├── figs/                    PDF + PNG 图
└── refs/references.bib      含最新 Hyperbolic-RoPE 引用

scripts/                     训练/eval/analysis 脚本
├── 2026-04/                 ⭐ 此文件夹
│   ├── PAPER_HANDOVER_2026-04-27.md  ← 你正在读的
│   ├── 01a-01e_lora_train_*.sh       (Geo+LoRA baseline 脚本，未跑)
│   └── ...
├── core_text_phases/        Primary 实验
└── analysis/                后处理 + verification

results/                     训练日志 / PPL JSON / metadata
```

---

## 联系点

如果你明天用 Claude Code 修完后想让 Cowork 这边再 cross-validate，**只需要把这次的 commit hash 告诉我**——我会重新 git pull、recompile、跑全套 audit。不需要重新解释上下文。
