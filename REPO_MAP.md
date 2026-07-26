# Repository Map

最后更新：2026-07-26

当前工作分支为 `main_0726`。`main` 与
`backup/main-restored-paper-20260726` 保留清理前、已恢复投稿版 `paper/` 的快照。

本文件只回答三个问题：去哪里读、哪里可以改、哪一层能支持 reviewer-facing claim。

## 1. 顶层目录职责

| 路径 | 职责 | 状态 / 规则 |
| --- | --- | --- |
| `paper/` | 已恢复的投稿版 LaTeX source、图表、参考文献、唯一 `main.pdf` | 整体只读；不得修改、编译、移动、删减或重新生成 |
| `rebuttal/` | response-only control room | `rebuttal_0723/` 为当前真实审稿周期；`pre_rebuttal/` 为历史准备；不进入 supplement |
| `data/curated/` | 可跟踪、匿名化、reviewer-safe 的小型结果 artifact | reviewer-facing evidence 层；边界由 provenance manifest 决定 |
| `docs/overview/` | 当前 claims、provenance、复现、术语与审计索引 | 决策文档层；先读其中 `README.md` |
| `paper_experiments/` | 94 个 paper experiment source 的 manifest-driven 视图 | 只浏览/导出；canonical code 仍在 `scripts/`、`experiments/` |
| `scripts/` | 主实验、评估、数据准备、出图与 RoPE 库 | canonical implementation |
| `experiments/` | 独立实验包；当前主要是 LLaMA-3-8B LoRA | supporting/rebuttal-only，除非 provenance 显式升级 |
| `tests/` | 科学协议、实现、artifact 和导航回归测试 | 改 source-of-truth 时同步更新 |
| `results/` | 已跟踪历史结果与本地新增结果的混合层 | 新输出默认 ignored；不得自动视为 reviewer-grade |
| `docs/exp/` | 按日期整理的实验报告 | report-backed，不等于 raw-JSON-backed |
| `docs/theory/`, `docs/tau_algor/` | 理论推导、数值检查与历史算法探索 | 当前 rebuttal 理论边界以 theory audit 为准 |
| `internal/` | 历史工作记录、旧稿和本机归档 | 非公开、非当前权威；export-ignore |
| `.codex_tmp/` | 本机忽略的恢复脚本和第三方 scratch | 不能作为 source of truth；删除前先查 provenance |

## 2. Source-of-truth 矩阵

| 需要回答的问题 | 唯一优先入口 |
| --- | --- |
| 论文现在写了什么 | `paper/main.tex` 及其 `sections/`, `appendix/`, `tables/` |
| 最终投稿 PDF 是哪个 | `paper/main.pdf` |
| 论文主张与脚本/数据如何对应 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| 某个数字能否用于 rebuttal | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` |
| 核心实验代码在哪里 | `paper_experiments/MANIFEST.json` → canonical `scripts/` / `experiments/` |
| EVQ-Cosh schedule 的实现 | `scripts/lib/rope/schedules.py` |
| 当前真实 review、AC 与行动状态 | `rebuttal/rebuttal_0723/README.md` → `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` + `01_REBUTTAL_PLAYBOOK.md` |
| 理论批评的历史事实底稿 | `rebuttal/pre_rebuttal/THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md` |
| 当前 AI 应接着做什么 | `rebuttal/rebuttal_0723/README.md` → `01_REBUTTAL_PLAYBOOK.md` |

## 3. 证据层级

```text
reviewer-safe tracked artifact
  data/curated/* + RESULT_PROVENANCE_MANIFEST
          │
          ├── raw JSON backed / byte-exact
          ├── sanitized run manifest
          └── report backed（必须明确降级）

historical/local evidence
  results/*, docs/exp/*, internal/*, .codex_tmp/*
          │
          └── 只能提供线索；不能自动升级 claim
```

缺少当前 artifact 只说明“当前 checkout 未证明”，不能写成“实验从未运行”。

## 4. 热路径

### Paper

- `paper/main.tex`
- `paper/sections/01_intro.tex`
- `paper/sections/03_theory.tex`
- `paper/sections/05_experiments.tex`
- `paper/sections/06_limitations.tex`
- `paper/appendix/a1_proofs.tex`
- `paper/appendix/a2_experiment_details.tex`
- `paper/appendix/a3_supporting_results.tex`
- `paper/appendix/a4_supporting_experiments.tex`

### Core code

- `scripts/lib/rope/schedules.py`
- `scripts/core_text_phases/run_evq_sweep.py`
- `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py`
- `scripts/core_text_phases/phase16_formula_optimality_sweep.py`
- `scripts/supporting_eval/eval_passkey_scratch.py`
- `scripts/package_supplement.py`

### Rebuttal

- `rebuttal/README.md`
- `rebuttal/rebuttal_0723/README.md`
- `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`
- `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`
- `rebuttal/rebuttal_0723/02_RESPONSE_QUESTIONS_AND_OUTCOMES.md`
- `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md`
- `rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`
- `rebuttal/pre_rebuttal/README.md`
- `rebuttal/pre_rebuttal/THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md`
- `rebuttal/pre_rebuttal/THEORY_FREQUENCY_OPTIMALITY_AND_TAU_20260716.md`
- `rebuttal/pre_rebuttal/LORA_GEO_CONTROL_RESULT_AUDIT_20260711.md`

## 5. Data 与 results 放置规则

- 小型、匿名化、可验证且需要 reviewer 使用的 artifact：`data/curated/`。
- 新训练/评估原始输出：`results/<narrow-family>/`，默认不提交。
- 结果解释报告：`docs/exp/YYYY-MM-DD_slug.md`。
- 大型数据、tensor、checkpoint、cache：保持 ignored，不得 force-add。
- 外部 artifact 要进入 reviewer 路径，必须先生成 sanitized manifest 和 SHA256，再更新 provenance manifest。

## 6. Archive 与本地层

- `internal/2026_03_run`, `internal/2026_04_run`：历史工作快照。
- `internal/local_archive`, `internal/local_snapshots`：本机 ignored；可能含大数据或私有路径。
- `.codex_tmp/phase17_454m_L512_2B.py` 等小脚本是 provenance clue，不是 reviewer-grade completion。
- `results/legacy/`：明确不属于当前核心 submission path。

旧代理日志、模拟审稿、重复 playbook、过时 handoff 和历史审计壳已从
`main_0726` 移除；需要追溯时从 `main` 或备份分支读取，不要重新复制回当前入口。

## 7. 新文件该放哪里

| 文件类型 | 位置 |
| --- | --- |
| 新主实验 runner | `scripts/core_text_phases/` |
| rebuttal-only launcher | `rebuttal/rebuttal_0723/experiments/`，并由同目录 README/standalone owner 索引 |
| 复用型 RoPE 实现 | `scripts/lib/rope/` |
| 数据准备 | `scripts/data_prep/` |
| 独立模型实验包 | `experiments/<package>/` |
| 论文图生成脚本 | `scripts/figures/`；不得在本分支写入只读 `paper/figs/` |
| 当前高层文档 | `docs/overview/` |
| 实验报告 | `docs/exp/` |
| 当前 rebuttal 策略与 review | `rebuttal/rebuttal_0723/`；根 `rebuttal/README.md` 只做分流 |

## 8. 不应重新出现的根级内容

- 第二份 provenance manifest；
- 第二个 rebuttal 文件夹；
- paper build 目录或额外根级 PDF；
- server address / token / private absolute path；
- 空 `artifacts/`、`team/` 或工具缓存目录。
