# AI Handoff — EVQ-Cosh NeurIPS 2026

最后更新：2026-07-12（Asia/Shanghai）
分支：`main`
仓库整理 checkpoint：`1b97fdc`（`handoff: consolidate repository and rebuttal index [01]`）
Temporal holdout checkpoint：`5bdec86`（`checkpoint: add temporal holdout evaluation`）
整理前实验基线：`805878f`（`prepare exact FineWeb 3x1B tensors`）
状态：仓库结构、paper 清理与 rebuttal control room 已在 `main`；LongAlpaca seed-42 temporal-holdout 结果已完成本地审计并进入当前 Codex 分支，未修改论文指标。

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

## 5. Rebuttal-triggered 实验准备

仓库整理、rebuttal consolidation 和 paper cleanup 保存在 `1b97fdc`；以下实验准备保存在 `5bdec86`：

1. `scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh` 是受控的 Geo/EVQ seed-42 shared driver；`05_lora_longalpaca_paper_evq_s42.sh` 是只允许 preflight/train 的 EVQ wrapper。
2. `scripts/data_prep/prepare_temporal_holdout_2026.py` 生成冻结、带 hash manifest 的 2026 temporal holdout；不把下载语料或 tokenized packs 提交进仓库。
3. `eval_temporal_holdout_matched.py` 从同一次 32K forward 汇总 matched 8K/16K prefixes；`eval_temporal_holdout_three_arm.py` 比较 Geo base、Geo+LoRA、EVQ+LoRA 三臂。
4. `scripts/2026-07/06_lora_temporal_three_arm_eval.sh` 对 adapter、语料 manifest、输出覆盖和 GPU 锁 fail closed。
5. seed-42 结果保存在 `data/curated/lora_longalpaca_temporal_s42_20260712.json`，NLL 解释与边界保存在 `rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`；不要写成“48% PPL tradeoff”。
6. `scripts/2026-07/07_lora_longalpaca_evq_remaining_seeds.sh` 只训练 EVQ+LoRA seeds 43/44，复用 Geo+LoRA seed 42 作为固定评测参照，并共享持久化 compile cache。

这些结果仍是 single-seed supporting evidence，不是论文已报告的 primary claim。EVQ seeds 43/44 完成后可报告 EVQ 三 seed 稳定性，但不得写成三 seed paired Geo/EVQ 对照。

## 6. Known issues / current breakage and validation

2026-07-12 在 `.venv` 中新鲜验证：`295 passed, 1 warning`。

唯一告警来自本机 Python 的 LibreSSL 2.8.3 与 urllib3 v2 兼容提示；不是测试失败或实验逻辑告警。

```text
urllib3 v2 only supports OpenSSL 1.1.1+; ssl module uses LibreSSL 2.8.3
```

同一轮还通过：

- temporal Python entrypoints `py_compile`；
- rebuttal shell entrypoints `bash -n`；
- reviewer supplement build 与 ZIP integrity；
- `git diff --check` 和新增内容敏感信息扫描。

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
