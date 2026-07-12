# AI Handoff — EVQ-Cosh NeurIPS 2026

最后更新：2026-07-12（Asia/Shanghai）
分支：`main`
HEAD：`805878f`（`prepare exact FineWeb 3x1B tensors`）
状态：仓库整理与 rebuttal 准备已完成主要结构工作；工作树仍包含未提交的 rebuttal、paper 清理和独立实验开发改动。

本文件是后续 AI 的**第一入口和状态索引**。它只保存可提交的仓库级信息，不保存服务器地址、凭据、私有绝对路径或实时进程信息。

## 1. 开始工作的固定阅读顺序

1. `AGENTS.md`：科学主张、匿名性、编辑和 Git 硬规则。
2. `ai-handoff.md`：当前状态、未提交工作和已知故障。
3. `REPO_MAP.md`：目录职责、source of truth 和禁止混用的层级。
4. `rebuttal/README.md`：7 月 22 日 rebuttal control room 总入口。
5. `rebuttal/rebuttal_playbook.md`：真实 reviews 到来后的统一 response-only 策略。
6. `docs/overview/RESULT_PROVENANCE_MANIFEST.md`：实验数字与 artifact provenance 的最高权威。
7. `paper/README.md`：论文源码与唯一最终 PDF 的边界。

如果这些材料冲突，优先级是：

`AGENTS.md` → 最新 provenance / theory audit → rebuttal control room → paper 当前源码 → 历史报告。

## 2. 项目身份与不可漂移的主张

EVQ-Cosh 的窄主张是：RoPE 的有限频率表也是 finite spectral budget；training-time frequency allocation 是 operator design 与 inference-time range scaling 之外的第三个 PE 设计轴。

不得把论文改写成：

- universal long-context SOTA；
- YaRN、LongRoPE、DAPE、FIRE 或 learned PE 的替代品；
- `tau=d_eff/sqrt(L)` 的全局最优定理；
- production-scale 或 frontier-model 的完整验证。

当前理论边界：

- exact：给定 convex surrogate 的 cosh density、CDF/inverse CDF 和 geometric limit；
- conditional：diffuse probability-transport / phase-variance proxy 下的局部 scaling motivation；
- empirical：finite-`tau` operating point 与 MLA `d_eff=d_head` convention；
- correction：ordinary baseline-to-perturbed KL 一阶为零，从 `O(tau^4)` 开始。

## 3. 当前最重要的证据

| 证据 | 权威路径 | 可用边界 |
| --- | --- | --- |
| Primary I：EVQ × fixed-scale YaRN | `data/curated/primary1_evq_yarn_10pct_raw.json`; provenance manifest M1 | 454M、3 seeds、teacher-forced NLL-gap PK；不是 tuned-YaRN dominance |
| Primary II：PE-dominant diagnostic | `data/curated/fig3_extreme_128.json`; provenance manifest M2 | Geo/DAPE-style/EVQ 保留 seed 42；不得升级为完整 3-seed learned-PE dominance |
| Primary III：MLA scarce-channel stress test | `data/curated/eval_3seeds_full_results.json`; `table18_mla_3seed_aggregate.json` | 3 seeds；`d_eff` 是 stated operating convention，不是 theorem |
| 99-run basin | `data/curated/phase16_99run_manifest.csv` | empirical basin/rank support，不证明 ordinary KL 或 global optimality |
| 论文提交件 | `paper/main.pdf` | 唯一根级 paper PDF；41 页；不得用历史 PDF 覆盖 |

`data/curated/` 的每个文件都已跟踪；完整用途和 SHA256 以 `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 为准。

## 4. Rebuttal 当前状态

- Preparation：`triage_ready`。
- Response package：`needs_author_input`。
- Mode：`pre-review / triage-only / response-only`。
- 截至 2026-07-12，实际 NeurIPS reviews 尚未收到。
- 真实 reviews 到来后只选择 3–5 个 score-driving concerns；不要恢复多路径 response 草稿。
- `rebuttal/rebuttal_playbook.md` 是统一策略入口；`REVIEWER_TRIAGE_PLAYBOOK.md` 负责稳定 ID 和映射；Master Ledger 与 theory/LoRA audits 是深层索引。
- 模拟审稿只能用于内部压力测试，不能当成 reviewer 原话。

## 5. 当前工作树中必须保护的未提交工作

以下内容来自不同工作流，不能因“清仓库”而丢弃或混成一个未经审计的提交：

1. rebuttal consolidation：旧 `rebuttal_7/` 与重复草稿删除，新 control-room 文档和模拟原文尚未统一提交。
2. paper cleanup：只保留 `paper/main.pdf` 作为最终 PDF，清除了 build artifacts、旧样式、旧 playbook 和未引用历史文件。
3. LongAlpaca paper-lineage launcher：`scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh` 已有未提交修改，另有新的 EVQ wrapper。
4. temporal-holdout 开发：新的 evaluator、data-prep 脚本和两个测试文件尚未提交。

提交前必须按概念拆分审计；不要使用 `git reset --hard`、`git checkout --` 或 `git clean`。

## 6. Known issues / current breakage（已知问题）

全量测试当前结果：`288 passed, 1 failed`。

失败：

```text
tests/test_legacy_lora_multiseed.py::LegacyEvaluationTests::
test_longalpaca_launcher_only_runs_geo42_and_prunes_intermediate_checkpoints
```

最小复现：

```bash
.venv/bin/python -m pytest \
  tests/test_legacy_lora_multiseed.py::LegacyEvaluationTests::test_longalpaca_launcher_only_runs_geo42_and_prunes_intermediate_checkpoints -q
```

原因：测试仍要求 launcher 含字面量 `--rope_method native_geo`，而当前未提交 launcher 使用受控变量 `--rope_method "$ROPE_METHOD"` 支持独立 wrapper。该改动不属于仓库整理，未擅自修复；应由该实验工作流统一决定测试契约。

## 7. 安全验证命令

```bash
# 核心 RoPE / paper workspace / rebuttal evidence
.venv/bin/python -m pytest \
  tests/test_rope_core.py \
  tests/test_paper_experiment_workspace.py \
  tests/test_rebuttal_evidence_bundle.py \
  tests/test_opus48_audit_docs.py -q

# Reviewer supplement
python3 scripts/validate_rebuttal_evidence_bundle.py
python3 scripts/package_supplement.py --output /tmp/evq-cosh-supplement.zip
unzip -t /tmp/evq-cosh-supplement.zip

# Paper PDF identity
pdfinfo paper/main.pdf
shasum -a 256 paper/main.pdf
```

不要把整个仓库直接压缩成 supplement。

## 8. 本地私有快照

旧的 server-heavy handoff 已原样保存到被忽略的：

`internal/local_snapshots/ai-handoff-private-20260711.md`

SHA256：`6c6eff1315e19c19406b0ceccc8838443396d94455d7168c9b4bf02ea2ea312a`。

旧根级 provenance snapshot 已移动到：

`internal/local_snapshots/RESULT_PROVENANCE_MANIFEST_20260614.md`

它们只用于本机历史追溯，不是 main branch 的 source of truth，不得提交、打包或复制其私有路径。当前 provenance 只能使用 `docs/overview/RESULT_PROVENANCE_MANIFEST.md`。

## 9. 下一位 AI 的停止条件

遇到以下情况先停止扩张：

- 真实 reviewer 原文尚未到达，却准备写完整 author response；
- 准备用 supporting/单-seed/trace-only 结果升级 primary claim；
- 缺 artifact 就推断实验从未运行；
- 想删除 `internal/`、`results/`、`data/` 或 `.codex_tmp` 中尚未完成 provenance 判断的内容；
- 想创建新分支；用户已明确要求直接维护 `main`。
