# AI Handoff — EVQ-Cosh NeurIPS 2026

最后更新：2026-07-24（Asia/Shanghai）
分支：`main`
仓库整理 checkpoint：`1b97fdc`（`handoff: consolidate repository and rebuttal index [01]`）
Temporal holdout checkpoint：`5bdec86`（`checkpoint: add temporal holdout evaluation`）
整理前实验基线：`805878f`（`prepare exact FineWeb 3x1B tensors`）
状态：Reviewer 27bE 正式 review 与 AC metareview 已分文件归档；当前
rebuttal 入口为 `rebuttal/rebuttal_0723/`，历史准备位于
`rebuttal/pre_rebuttal/`。
151.9M/500M single-seed 机制诊断仍是 supporting evidence；**论文指标与主表数字未改**。

本文件是后续 AI 的**第一入口和状态索引**。它只保存可提交的仓库级信息，不保存服务器地址、凭据、私有绝对路径或实时进程信息。

## 0A. 2026-07-24 最新结果与下一实验

- 当前完整实验结论入口：
  `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md`。
- 当前 reviewer-facing evidence/action ledger：
  `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`。
- Reviewer 与 AC 统一入口为
  `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`。Reviewer 部分有
  OpenReview 来源哈希；AC 文本由作者提供，当前没有独立 URL/hash。
- Phase16 本机 99-run raw 已重新核对：
  `rebuttal/rebuttal_0723/theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`。
  Formula tau 对 midpoint-Geo 为 7/9 配置均值获胜，但对 pilot-selected
  neighbor 的 held-out 比较仅 3/9 获胜；只能称 fallible operating prior。
- native Std-RoPE / matched shape / attention-derived shape 的三 seed 结果已完成；
  聚合 raw-backed JSON 为
  `rebuttal/rebuttal_0723/theory_results/native_attention_shape_l128_results_20260724.json`。
  结果支持 allocation shape 轴，但明确不支持 Cosh 唯一最优。
- MLA scarcity 包位于
  `rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/`。50.1M 固定架构的 seed-42
  六臂、selection/test 和 YaRN 诊断均已完成；汇总为
  `mla_scarcity_seed42_result_20260724.json`。
- 原注册 gate 形式上 PASS，但 K=8/8K 的 range control 从 native 3.9915
  崩到 6.2195 NLL；EVQ 虽恢复到 4.6437，仍比 native 差 0.6522 NLL
  （约 1.92x PPL）。K=32/8K 也差 0.0382 NLL。独立 test 与 selection
  一致，因此冻结 `DO_NOT_EXPAND_SEEDS`，未运行 seed 43/88。
- YaRN secondary 在 K=8 呈长外推方向性增益（16K -0.1161 NLL、32K
  -0.7797），但 native 使用 active-grid official equations，EVQ 使用
  virtual-coordinate derived transform；只能称 single-seed deployment
  diagnostic，不能恢复 raw 或 official-YaRN complementarity claim。
- 六臂共训练 73.81 分钟，平均 406.3K tokens/s。12 个 200M/300M 权重均在
  test JSON/checkpoint hash 通过后删除；终态 checkpoint/incomplete 均为 0。
  终态曾回传 99 个结果文件并生成 manifest，实例已关机；当前本机只能定位到
  sanitized aggregate 和 manifest hash，原 99-file bundle 已不可发现，不能
  声称当前仍可逐文件访问。
- RTX 5090 的已验证性能基线已记录到
  `docs/overview/RTX5090_BLACKWELL_PROFILE.md`。后续训练默认复用
  BF16、probe/receipt 选出的 `torch.compile` 模式、Flash-only SDPA、fused AdamW、
  `expandable_segments` 和持久 TorchInductor cache，但每个新 workload
  仍须先做 discarded probe，不能照搬 batch size。
- 本轮 MLA 包服务器测试为 `16 passed`；本地 `py_compile`、`bash -n`、
  `git diff --check`、聚合 JSON/原始结果数字 parity 与新增内容泄漏扫描通过。
  这些结果与验证记录已进入当前仓库历史。
- 当前最直接对应 `R27bE.1/.4` 与 `AC.1/.3` 的候选是
  `rebuttal/rebuttal_0723/theory_results/ROPE_RANGE_SHAPE_MAPPING_THEORY_AND_5090_PLAN_20260724.md`：
  复用既有 FMR range baseline 的 seed-42 Anchored-Cosh matched-range arm
  已完成。fixed range 下 Cosh 在 512/1K/2K 改善
  0.478/0.205/0.113 NLL；target-retarget 后反而差
  0.061/0.182/0.279 NLL。Anchored-Exp 不扩；matched-range 多 seed
  确认已于 2026-07-24 启动，运行状态见下节。
- `rebuttal/rebuttal_0723/theory_results/TRAINING_FREE_TAU_SELECTOR_20260724.md`
  已完成 exact finite-K Gram collision selector：Phase16 九配置平均 PPL
  regret 5.44%，旧公式 4.69%，top-2 basin 1/9 vs 5/9。历史 gate 失败，
  因此未运行三个 prospective 配置，也不提出新的经验 tau 公式。
- 较早的
  `rebuttal/rebuttal_0723/theory_results/MLA_YARN_OPERATOR_PARITY_5090_PLAN.md` 及 runner
  保留为 operator-parity 备选，但尚无 fresh anchors/READY，不应与上述
  range/shape 计划并行抢占 GPU。

## 0B. Exact-range 多 seed 与 350M 自动队列 handoff

**注册协议：**
`rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/SPEC.md`。
本节记录可同步的运行状态；服务器地址、凭据和私有绝对路径不得写入仓库。

- 151.9M 确认组：训练 seeds `137/256`，每个 seed 只跑
  `fmrope_base256` 与 `anchored_cosh_tau4_fmrope_range` 两臂；完成后与
  已有 seed-42 comparison 聚合为三训练 seed 结果。
- 350M scale-transfer：seed 42、`350,112,000` 参数、相同两臂。训练流实际
  消费 `999,948,288` tokens，严格拆成不重叠的
  `[0,499974144)` 与 `[499974144,999948288)` 两段；两臂使用完全相同的
  A→B 顺序。
- 2026-07-24 19:43（Asia/Shanghai）运行快照：seed 137 的 FMR 臂正在训练，
  约 `167.8K tokens/s`、GPU 利用率约 `98%`；1B 数据在 CPU/网络侧并行准备。
- 远端队列顺序已经固定：四个 151.9M 缺失臂 → seed 137/256 评测 →
  三 seed 聚合 → 等待 350M `READY` → 350M 两臂与评测。小模型运行目录的
  code fingerprint 已冻结；在其评测结束前不得向该运行副本同步代码。
- 时间预算：从上述快照起，小模型剩余约 `3–4 h`；350M 双臂按参数量缩放
  预计另需 `8–11 h`（含首次编译与评测）。建议自动关机设为从快照起
  `16 h`，即约 2026-07-25 11:45，给吞吐波动留出约一小时以上余量。
- 完成标志：
  `fmrope_exact_range_multiseed_20260724/multiseed/summary.json` 以及
  `fmrope_exact_range_350m_s42_1b_20260724/evaluation/results.json`。
  先检查这两个产物和队列日志，再复制回本地、写解释报告。
- 故障边界：runner 会拒绝覆盖任何非空 arm 目录，不能在失败后直接重跑整个
  队列。应先确认哪些 arm 已有完整 `model.pt`、metadata 和 evaluation，再只
  重跑缺失臂；不删除已完成 checkpoint，不把 incomplete 目录当成结果。

换机后的最小验证：

```bash
python -m py_compile \
  rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/{protocol,prepare,run_experiment}.py
bash -n \
  rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/{run_5090,run_full_queue_5090,prepare_350m_background}.sh
python tests/test_fmrope_125m_l256_500m.py
```

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
4. `rebuttal/README.md`：当前/历史两层目录分流。
6. `rebuttal/rebuttal_0723/README.md`：当前真实审稿周期的唯一操作入口。
7. `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`：Reviewer 27bE 原文、评分、AC 原文与来源边界。
8. `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`：精选证据、回答路线与重要实验状态。
9. `docs/overview/RESULT_PROVENANCE_MANIFEST.md`：实验数字与 artifact provenance 的最高权威。
10. `paper/README.md`：论文源码与唯一最终 PDF 的边界。

如果这些材料冲突，优先级是：

`AGENTS.md` → rebuttal principles → 正式 review → 最新 provenance / theory
audit → paper 当前源码 → pre-rebuttal 历史报告。

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
| 99-run basin | `data/curated/phase16_99run_manifest.csv`; current raw reanalysis | Formula-vs-Geo 7/9，但 held-out neighbor 仅 3/9；不证明 near-optimality、ordinary KL 或 global optimum |
| 论文提交件 | `paper/main.pdf` | 唯一根级 paper PDF；41 页；不得用历史 PDF 覆盖 |

`data/curated/` 的每个文件都已跟踪；完整用途和 SHA256 以 `docs/overview/RESULT_PROVENANCE_MANIFEST.md` 为准。

## 4. Rebuttal 当前状态

- Preparation：`official_review_received / triage_only`。
- Response package：`needs_author_input`。
- Mode：`post-review / triage-only / response-only`。
- Reviewer 27bE 的评分 3、置信度 4 review 已逐字归档；仓库内尚无其他 reviewer 的逐字 source。
- AC metareview 已按作者提供文本单独归档；当前缺独立 source URL/hash。
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
