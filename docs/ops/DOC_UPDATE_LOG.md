# 文档系统性优化升级日志 (Documentation Refactoring Log)

> 更新时间: 2026-02-26 01:20 CST
> 本轮更新用于对齐 “24小时 NeurIPS 证据加固计划” 的代码接口、审计字段和交接文档。

## 0. 2026-02-26 本轮新增

- **新增 handoff 包**：`handoff_2026-02-26/`，沿用现有 dated-handoff 结构，包含：
  - `0_README.md`, `1_PROTOCOL_LOCK.md`, `2_ASSET_MAP.md`, `3_RUNBOOK.md`
  - `01_IMPLEMENTED_SCOPE.md`, `02_VALIDATION_SNAPSHOT.md`, `03_DEEP_REVIEW_FINDINGS.md`
- **更新 AI 接手入口**：`AI_HANDOFF.md` 新增 2026-02-26 包索引。
- **更新脚本索引**：`scripts/README.md` 纳入：
  - `plan_b_eval_longbench.py`
  - `plan_b_train_anchored_v2.py`
  - `prepare_long_instruction_mix.py`
  - `build_model_registry.py`
- **更新实验库存说明**：`docs/exp/EXPERIMENT_INVENTORY.md` 补充 Plan B 代码就绪状态。
- **更新审计清单**：`docs/exp/plan_b_audit_manifest.md` 补充统一结果 schema 与接口变更状态。

## 0.1 代码/协议对齐摘要（与文档关联）

- LoRA 资产合同：
  - 训练产物支持 `root_adapter + final_lora` 双布局，并在 `summary.json` 记录 `adapter_layout`。
- Registry 可评测判定：
  - `build_model_registry.py` 支持 `root_adapter|final_lora`，输出 `adapter_resolved_path`。
- Plan B 评测控制面：
  - 可选 `lb6/lb21/custom tasks`，并支持 NIAH/Passkey 压测参数。
- 统一结果 schema：
  - `eval_longbench.py`, `eval_niah_recall.py`, `eval_passkey_teacher_forcing.py` 均写入：
    - `protocol_lock`
    - `manifest_json`
    - `per_sample_scores_raw`
    - `inv_sha256`

> 更新时间: 2026-02-22
> 本更新日志总结了为满足 "可直接写论文/可复现实验/可被导师复核" 标准而对核心文档层进行的重构。

## 1. 核心产物：实验事实表 (The Registry)
- **新增 `docs/EXPERIMENT_REGISTRY.md`**：作为整个代码库中论文数据的唯一集线器。包含了从 50M -> 350M 的从零训练、124M Phase 4的极化崩溃测试，到 8B LoRA 级别微调任务的详细指令、参数与底层 JSON 指针。
- **废弃标记**：明确阻隔了过去由于 monkey patch API 碰撞导致的旧 8B 实验数据流入论文正文之中，规避了重大学术风险。

## 2. 协议与评价规范 (Protocols)
- **新增 `docs/TERMS_AND_PROTOCOLS.md`**：正式化定义 PPL 评估切片滑动窗口与 `inv_freq.copy_()` 的严格评估规约。
- **重构 `docs/METHODOLOGY.md`**：去除随意的代码记事本风格，转录为严密的学术公式及约束定义。
- **新增 `docs/EXPERIMENT_INDEX_CN.md`**：重定向到 EXPERIMENT_REGISTRY，消除多源信息悖论。

## 3. 知识库升级 (Knowledge Base)
- **更新 `knowledge_base/01_已完成实验核心数据.md`**：化繁为简，只提炼了 Paper-Ready 级别的核心 50/100/350M 训练主线，移除已过时或待 overnight 洗刷的数据。全面重定向至 registry。
- **更新 `knowledge_base/08_8b_experiment_analysis.md`**：套用 TL;DR, Claims, Evidence Map 与 Failure & Fix 格式。明确了实验不公平的实施根源。
- **更新 `knowledge_base/00_项目与结论总览.md` & `ALL_IN_ONE.md`**：拔高至 Claims 级别，为撰写 Intro 与 NeurIPS 投递提供了模块化理论支撑地图。

## 4. 论文展示材料图谱 (Paper Draft & Readmes)
- **重构 `docs/README` & `docs/RESULTS` & `docs/REPRODUCE`**: 划分了 Priority 与 Legacy 边界，提供了直接复现三条代码的核心指令。
- **新增 `paper_draft/FIGURE_TABLE_PLAN.md`**：为 LaTeX 列出了 4 张主图与 2 张主表的明确结构及客观数据来源路径。
- **新增 `paper_draft/06_limitations_and_discussion.md`**：为论文扩写了深度的 Limitations 章节，包括训练依赖性 (Training Dependency)、高频崩溃下的信息水床效应乃至工程上的实施海市蜃楼 (Implementation Mirages)。

---
### 自检报告 (Self-Check Report)
- ✅ **链接校验**：`docs/README.md` 与各类知识库之间的相对路由更新完毕并有效，旧指标档案退居 Legacy。
- ✅ **溯源校验** (随机抽查 5 个数字全过校验)：
  1. 50M Hybrid 17.32 $\pm$ 0.36 -> `results/evidence_chain_50m_3cfg3seed/results.json`
  2. 100M Hybrid 9.42 (-13.5%) -> `artifacts/a100.../100m_scaling/`
  3. 124M Sigmoid 32K 147.5 -> `sigmoid_rope_experiments/data/ppl_vs_length.csv`
  4. Geometric Geo 10k 崩溃比 22x -> `results/anchored_sigmoid_v3_followup/`
  5. 8B 禁用 -> `08_8b_experiment_analysis.md` 明确声明。
- ✅ **公平锁**：已严厉限制 8B 旧版非公平实验流出，保护前沿声明严格可靠。
