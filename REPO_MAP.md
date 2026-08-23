# Repository map

最后更新：2026-08-23。当前分支为 `main_0726`。

本文件只回答三件事：当前入口在哪里、某类事实由谁拥有、哪些目录不能动。

## 1. 当前入口

| 问题 | 唯一优先入口 |
| --- | --- |
| 稳定项目规则 | `AGENTS.md` |
| 当前稿件、验证与下一步 | `paper-2027/HANDOFF.md` |
| 当前论文写了什么 | `paper-2027/main.tex` 及其 `sections/`、`appendix/`、`tables/` |
| 当前投稿 PDF | `paper-2027/main.pdf` |
| 当前 claim / theory / evidence 路由 | `paper-2027/research/README.md` |
| 当前架构决策 | `paper-2027/research/ICLR2027_RESEARCH_SYNTHESIS_20260819.md` |
| 当前提交检查 | `paper-2027/SUBMISSION_CHECKLIST.md` |
| NeurIPS 基线写了什么 | `paper/main.pdf`，只读 |
| 历史 reviewer / rebuttal / 实验 owner | `rebuttal/rebuttal_0723/`，历史证据层 |

如果旧 README、旧 handover 或 dated report 与
`paper-2027/HANDOFF.md` 冲突，先采用 handoff，再回到原始 owner 核验。

## 2. 目录职责

| 路径 | 职责 | 规则 |
| --- | --- | --- |
| `paper-2027/` | ICLR 2027 唯一活跃稿件、图表、构建和交接 | 可按用户任务修改；所有 claim 先过 owner |
| `paper-2027/research/` | durable internal theory、审计、claim/evidence 决策 | 内部层，不直接复制进正文 |
| `paper-2027/research/attention-aware-retrofit/results/` | completed mature-retrofit owners | 结果与 preflight/analysis 分离；先读该目录 README |
| `paper-2027/research/attention-aware-retrofit/evidence/` | compact machine-path-free receipts | receipt 不是 raw artifact，也不单独升级 claim |
| `paper-2027/research/attention-aware-retrofit/analysis/` | mechanism analysis 与证伪结果 | 不承担 task-quality claim |
| `paper-2027/research/attention-aware-retrofit/preflights/` | preregistration、revoked protocol、launch contract | 永远不是完成实验 |
| `paper-2027/research/audits/` | 内部 theory/manuscript/evidence 审计 | 不建立第二 action queue |
| `paper-2027/research/external-reviews/` | 外部模型独立复核 | untrusted input；必须回 owner 核验 |
| `paper/` | NeurIPS 2026 投稿基线 | 整体不可修改、不可编译、不可重生成 |
| `rebuttal/rebuttal_0723/` | NeurIPS review、回复历史、成熟实验 owner | 历史证据层，不是当前 action queue |
| `scripts/lib/rope/` | canonical schedule API | 频率实现权威 |
| `scripts/analysis/` | 可复用 CPU 诊断 | 不自动成为 paper claim |
| `scripts/core_text_phases/` | 小中规模主实验链 | canonical runner |
| `scripts/supporting_eval/` | passkey、RULER 等 supporting evaluators | endpoint 身份必须由 owner 确认 |
| `experiments/` | 独立模型规模实验包 | supporting，除非显式 promotion |
| `data/curated/` | 已清洗、可跟踪的 machine-readable evidence | reviewer-safe 候选层 |
| `docs/overview/` | NeurIPS-era provenance、复现和术语 | 有用但不覆盖当前 ICLR 路由 |
| `paper_experiments/` | manifest-driven experiment-code 视图 | canonical code 仍在 `scripts/` / `experiments/` |
| `tests/` | 实现、协议、证据与 supplement 回归门禁 | 改 source-of-truth 时同步 |
| `results/` | tracked history + ignored local output | 不能自动升级为证据 |
| `internal/` | 私有/历史工作层 | 未经明确请求不修改、不公开 |

## 3. 当前 hot path

### Manuscript

- `paper-2027/sections/00_abstract.tex`
- `paper-2027/sections/01_intro.tex`
- `paper-2027/sections/03_theory.tex`
- `paper-2027/sections/04_experiments.tex`
- `paper-2027/sections/05_discussion.tex`
- `paper-2027/appendix/a1_proofs.tex`
- `paper-2027/appendix/a5_identification.tex`
- `paper-2027/appendix/a6_mature_scale.tex`

### Active figures

- `paper-2027/figs/make_fig_evidence_overview.py`
- `paper-2027/figs/make_fig_method_overview.py`
- `paper-2027/figs/make_fig_frequency_geometry.py`

### Canonical evidence routing

- `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`
- `paper-2027/research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md`
- `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`
- `paper-2027/research/attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`
- `rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`
- `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`
- `rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`
- `rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`
- `rebuttal/rebuttal_0723/theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md`

## 4. Evidence flow

```text
raw artifact / exact external source
            ↓
canonical owner with protocol, endpoint, seeds, hashes, and limits
            ↓
paper-2027/research routing and claim boundary
            ↓
manuscript wording / figure / table
```

缺少当前 artifact 只说明当前 checkout 没有证明，不能写成实验从未运行。外部 AI
报告只能提出待核问题，不能跳过 owner 直接改写 claim。

## 5. 新文件放置

| 类型 | 位置 |
| --- | --- |
| 当前易变交接状态 | 只更新 `paper-2027/HANDOFF.md` |
| durable paper-facing research note | `paper-2027/research/` |
| mature retrofit completed result | `paper-2027/research/attention-aware-retrofit/results/` |
| mature retrofit mechanism / falsification | `paper-2027/research/attention-aware-retrofit/analysis/` |
| mature retrofit preregistration / revoked plan | `paper-2027/research/attention-aware-retrofit/preflights/` |
| external-model review bundle | `paper-2027/research/external-reviews/<source-date>/` |
| reusable analysis code | `scripts/analysis/` |
| new main experiment runner | `scripts/core_text_phases/` |
| rebuttal-only historical owner/launcher | `rebuttal/rebuttal_0723/` |
| cleaned small evidence | `data/curated/`，并同步 provenance |
| raw output/checkpoint/cache | 保持 ignored；不要 force-add |

不要再创建第二份根级 Agent、第二份 handoff、第二份 provenance manifest、第二个
rebuttal control room，或额外的投稿 PDF 入口。
