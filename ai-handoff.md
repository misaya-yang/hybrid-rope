# AI Handoff — EVQ-Cosh NeurIPS 2026

最后更新：2026-07-24（Asia/Shanghai）
分支：`main`
仓库整理 checkpoint：`1b97fdc`（`handoff: consolidate repository and rebuttal index [01]`）
Temporal holdout checkpoint：`5bdec86`（`checkpoint: add temporal holdout evaluation`）
整理前实验基线：`805878f`（`prepare exact FineWeb 3x1B tensors`）
状态：Reviewer 27bE 正式 review 已归档；当前 rebuttal 入口已迁移到
`rebuttal/rebuttal_0723/`，历史准备位于 `rebuttal/pre_rebuttal/`。
151.9M/500M single-seed 机制诊断仍是 supporting evidence；**论文指标与主表数字未改**。

本文件是后续 AI 的**第一入口和状态索引**。它只保存可提交的仓库级信息，不保存服务器地址、凭据、私有绝对路径或实时进程信息。

## 0A. 2026-07-24 最新结果与下一实验

- 当前完整实验结论入口：
  `rebuttal/rebuttal_0723/EXPERIMENT_REPORT_20260724.md`。
- native Std-RoPE / matched shape / attention-derived shape 的三 seed 结果已完成；
  聚合 raw-backed JSON 为
  `rebuttal/rebuttal_0723/native_attention_shape_l128_results_20260724.json`。
  结果支持 allocation shape 轴，但明确不支持 Cosh 唯一最优。
- 下一项 MLA scarcity 包位于
  `rebuttal/rebuttal_0723/mla_scarcity_5090/`，目前只有冻结代码和本地测试，
  没有训练结果。它固定 50.1M 参数、`d_rope=64`、`d_nope=0`，只把 active
  frequency pairs 从 32 改为 8；inactive pair 使用零频率恒等旋转，避免
  legacy `d_rope` 变化同时改变参数量和 key projection path。
- seed 42 是 selection-only 六臂 gate；只有 PASS 才运行 seeds 43/88。
  primary 是 raw tail NLL 的 range/shape decomposition；YaRN 仅为身份清楚的
  secondary diagnostic。
- 存储规则：训练低于 8 GiB free 直接拒绝；100M 权重立即剪除；200M/300M
  只有在匹配 evaluation JSON 和 checkpoint hash 通过后才逐 run 删除。原始
  JSON/JSONL、cleanup receipt、schedule sidecar 与终态前共享 compile cache
  必须保留。当前实验服务器不可连接，未执行远端删除。
- 本轮新鲜门禁：相关 MLA/shape/core 共 `165 passed`，`py_compile`、
  `bash -n`、`git diff --check` 与新增内容泄漏扫描通过。尚未提交或推送。

## 0. 2026-07-13 151.9M / 500M single-seed 结果

**性质：** RTX 5090 小模型机制诊断；不是 RTX Pro 6000 的 LLaMA/LoRA 轨道，也不是 Primary I 多 seed 复现。
**协议：** Native endpoint RoPE 与 endpoint EVQ (`tau=1.5`)；seed 42；训练长度 2K；499,998,720 tokens；151,898,880 参数；FineWeb-Edu + 2.0177% 显式 Passkey tokens。两臂 initial-trainable hash 与 row-order hash 一致。

### 0.1 已闭合结果

- 完整六格评测已落盘：`{raw, official/derived YaRN, repo fixed-ramp} × {Native, EVQ}`，2K/4K/8K/16K natural-text NLL/PPL + 100-case Passkey NLL-gap。
- Raw EVQ 对 Native 的 PPL：4K `-8.6%`、8K `-14.3%`、16K `-16.2%`；4K–16K 均为 8/8 offsets 更低 NLL。
- Official native-grid YaRN / EVQ virtual-coordinate derived YaRN 把两臂拉到接近相同 PPL；NLL difference-in-differences 为正，因此本次 **不支持 official YaRN 超加性交互**。
- 该六格表是 2K raw-trained checkpoints 上的 inference-only / non-fine-tuned 算子测试，不是 YaRN 论文的长上下文继续训练复现。Official YaRN 还包含 `mscale` attention-temperature；scale 8 时 rotary amplitude 为 1.208、QK logit factor 约 1.459，可能是两臂都接近饱和的重要原因，需 checkpoint-only ablation 才能拆分。
- Repo fixed-ramp 是另一个有效 range scaler，且与 EVQ 在 8K/16K 呈明显负 interaction（`-0.329/-0.413` NLL，负值代表 EVQ 获益更多）。它不得称为 official YaRN，且尚未证明相对其他 ramp/by-parts scaler 的新颖性。
- Passkey 在 2K 训练中被显式监督；其 teacher-forced gap 只能作为机制诊断。Official/derived YaRN 两臂均达 100% sign rate，已饱和；fixed-ramp 为 Native 77% / EVQ 87%。
- 500M 对 151.9M 仅约 3.29 tokens/parameter：当前 MHA 协议未出现 reversal，但不能声称充分训练或解决训练饱和问题。

权威结果：`rebuttal/pre_rebuttal/NATIVE_ROPE_EVQ_150M_500M_RESULT_20260713.md`。
清洗后的 raw-backed artifact：`data/curated/native_rope_evq_150m_s42_500m_20260713.json`。

### 0.2 评测修复与当前边界

- 首次自动评测只因 PyTorch 2.8 的 `TorchVersion` metadata 被 `weights_only=True` 拒绝而在模型推理前停止；修复仅 allowlist 该安全类型，并让新 checkpoint 保存 plain-string 版本号。模型权重、算子、数据、offset 与 metric 均未改变。
- 用户已取消后续数据准备；本轮没有启动新的 tokenization、训练或 GPU 任务。
- 新结果保持 single-seed supporting diagnostic；没有修改 `paper/` 或论文表格数字。

---

## 1. 开始工作的固定阅读顺序

1. `AGENTS.md`：科学主张、匿名性、编辑和 Git 硬规则。
2. `ai-handoff.md`：当前状态、未提交工作和已知故障。
3. `REPO_MAP.md`：目录职责、source of truth 和禁止混用的层级。
4. `Agent.md`：当前 reviewer、实验授权与 GPU 硬边界。
5. `rebuttal/README.md`：当前/历史两层目录分流。
6. `rebuttal/rebuttal_0723/README.md`：当前真实审稿周期的唯一操作入口。
7. `rebuttal/rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md`：当前唯一逐字正式 review。
8. `docs/overview/RESULT_PROVENANCE_MANIFEST.md`：实验数字与 artifact provenance 的最高权威。
9. `paper/README.md`：论文源码与唯一最终 PDF 的边界。

如果这些材料冲突，优先级是：

`AGENTS.md` → `Agent.md` → 正式 review → 最新 provenance / theory audit →
paper 当前源码 → pre-rebuttal 历史报告。

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

- Preparation：`official_review_received / triage_only`。
- Response package：`needs_author_input`。
- Mode：`post-review / triage-only / response-only`。
- Reviewer 27bE 的评分 3、置信度 4 review 已逐字归档；仓库内尚无其他 reviewer 的逐字 source。
- 当前只围绕 `R27bE.1`–`R27bE.5` 组织回答；补齐其他 review 后再做跨 reviewer 排序。
- `rebuttal/rebuttal_0723/README.md` 是统一策略入口；`pre_rebuttal/` 中的 playbook、ledger 与 theory/LoRA audits 只作历史事实底稿。
- 模拟审稿只能用于内部压力测试，不能当成 reviewer 原话。

## 5. Rebuttal-triggered 实验准备

仓库整理、rebuttal consolidation 和 paper cleanup 保存在 `1b97fdc`；以下实验准备保存在 `5bdec86`：

1. `scripts/2026-07/04_lora_longalpaca_paper_geo_s42.sh` 是受控的 Geo/EVQ seed-42 shared driver；`05_lora_longalpaca_paper_evq_s42.sh` 是只允许 preflight/train 的 EVQ wrapper。
2. `scripts/data_prep/prepare_temporal_holdout_2026.py` 生成冻结、带 hash manifest 的 2026 temporal holdout；不把下载语料或 tokenized packs 提交进仓库。
3. `eval_temporal_holdout_matched.py` 从同一次 32K forward 汇总 matched 8K/16K prefixes；`eval_temporal_holdout_three_arm.py` 比较 Geo base、Geo+LoRA、EVQ+LoRA 三臂。
4. `scripts/2026-07/06_lora_temporal_three_arm_eval.sh` 对 adapter、语料 manifest、输出覆盖和 GPU 锁 fail closed。
5. seed-42 结果保存在 `data/curated/lora_longalpaca_temporal_s42_20260712.json`，NLL 解释与边界保存在 `rebuttal/pre_rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`；不要写成“48% PPL tradeoff”。
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
