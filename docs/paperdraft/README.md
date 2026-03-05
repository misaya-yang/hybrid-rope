# EVQ-Cosh NeurIPS 2026 — 论文核心文档

> 本文件夹只放论文写作直接需要的文档。Prompt 在 `../prompts/`，历史文档在 `../archive/`。

## 文档清单

| 文件 | 内容 | 读取顺序 |
|------|------|---------|
| `PROMPT_AI_HANDOFF.md` | 项目状态 + 数据路径 + 红线 | **第 1 个读** |
| `CORE_THEORY.md` | 理论精简版（推导链 + 6 个核心理论点） | **第 2 个读** |
| `EVQ_COSH_THEORY.tex` (`EVQ_COSH_THEORY.pdf`) | 纯理论版定理/公式集合（可直接粘贴到论文 Appendix） | 写 Appendix/Proofs 时读 |
| `EXPERIMENT_RESULTS_128TOK.md` | 128-tok paper-ready 结果表 | 写 Section 5 时读 |
| `phase6_report.md` | Phase 6 完整实验报告 | 写 Section 5 时读 |
| `phase7_report.md` | Phase 7 完整实验报告 | 写 Section 5 时读 |
| `EXPERIMENT_REPORT_128TOK.md` | Phase 1-3 mini-sweep 详细报告 | 需要细节时读 |
| `PAPER_ERROR_CORRECTIONS.md` | 论文 7 个已知错误 | 改论文前必读 |
| `LATEX_SNIPPETS.md` | 可粘贴的 LaTeX 段落 | 写论文时参考 |

## 其他文件夹

- `../prompts/` — Claude Code 实验计划（Phase 5-8）+ Gemini 理论提问
- `../archive/` — 历史讨论、旧版文档、Gemini 回复原文
- `../../data/evq_128tok_results/` — 全部实验 JSON 数据
- `../../scripts/m4_evq_sweep/` — 训练 + 评估脚本
