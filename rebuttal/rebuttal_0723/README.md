# Rebuttal 0723: current review cycle

最后更新：2026-07-24

状态：`official_review_received / ac_metareview_received / triage_only / no_response_yet`

本目录是当前 rebuttal 的唯一操作入口。先固定 reviewer 实际问了什么，再决定
是否使用历史分析或新实验；不得反过来从已有材料拼一个泛化答辩。

## Directory layout

- Root: this README, `00_REVIEWER_27BE_OFFICIAL_REVIEW.md`, and
  `01_AC_METAREVIEW.md`. Keep reviewer and AC sources at the top level.
- `theory_results/`: rebuttal principles, theory/planning notes, result reports,
  manifests, and curated result JSONs.
- `experiments/`: all rebuttal experiment packages, helper code, and launchers.
- Repository-level `tests/` remains the test location.

## 0. 核心索引

| 文件 | 作用 | 权威边界 |
| --- | --- | --- |
| `00_REVIEWER_27BE_OFFICIAL_REVIEW.md` | Reviewer 27bE 原文与 `R27bE.1`–`R27bE.5` | 带 OpenReview revision URL 与 source payload hash |
| `01_AC_METAREVIEW.md` | AC 原文与 `AC.1`–`AC.4` | 作者提供文本；当前无独立 URL/hash |
| `theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` | AC concern、方法边界、理论层级、实验原则与停止条件 | 实验设计权威规则 |
| `theory_results/EXPERIMENT_REPORT_20260724.md` | 0723–0724 数值结果总入口 | 所有正负结果必须连同各节 claim boundary 使用 |
| `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md` | 99-run raw 的共同 metric 重算 | 支持 fallible prior；不支持 near-optimal scaling law |
| `theory_results/ROPE_RANGE_SHAPE_MAPPING_THEORY_AND_5090_PLAN_20260724.md` | range/shape 可识别性与 matched-range 三臂计划 | 新实验尚未运行 |
| `theory_results/MLA_YARN_OPERATOR_PARITY_5090_PLAN.md` | MLA operator-parity 后续设计 | 尚无 fresh anchors/READY，不是 GPU 启动授权 |
| `theory_results/FREQUENCY_DEFINITION_MANIFEST.json` / `experiments/geo_rope_contract.py` | Paper-Geo、Std-Geo 与 EVQ 频率身份及 guard | 实现身份，不是实验结果 |

## 1. 当前证据权限

1. `00_REVIEWER_27BE_OFFICIAL_REVIEW.md` 是仓库内目前唯一逐字、带来源哈希的
   正式 review，稳定 ID 为 `R27bE.1`–`R27bE.5`。
2. `01_AC_METAREVIEW.md` 是当前 AC 权威入口，但其 source URL/hash 尚未
   独立核验，不能与 Reviewer 27bE 的 provenance 等同。
3. `theory_results/EXPERIMENT_REPORT_20260724.md` 是当前数值结论入口；FMRoPE、500M
   undertraining、held-out base/head、native-Geo 与 real-shape 结果必须连同
   各节 claim boundary 使用。native/real-shape 聚合原始结果另存为
   `native_attention_shape_l128_results_20260724.json`。
4. `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md` 已从本机 99-run raw 重算共同
   PPL metric：formula tau 对 midpoint-Geo 为 7/9 配置均值获胜，但对
   pilot-selected neighbor 的 held-out 比较仅 3/9 获胜。不得恢复
   `near-optimal across the grid`。
5. `experiments/mla_scarcity_5090/` 的 seed-42 六臂、selection/test 与 YaRN 诊断已完成。
   注册 shape gate 虽为 PASS，但 K=8/8K 的正 interaction 来自 range control
   崩坏；raw EVQ 仍比 native 差 0.6522 NLL。因此终止多 seed 扩展，不能写成
   practical scarce-channel advantage。
6. `theory_results/MLA_YARN_OPERATOR_PARITY_5090_PLAN.md` 是针对上述结果的新 2×2 设计：
   native/EVQ 共用同一 YaRN index mask 与 `mscale`。离线 runner、门控与测试
   已完成，但目标服务器尚无 fresh anchors/READY；CPU preflight 通过前不得
   启动 GPU。
7. `theory_results/EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md` 是宽方案库，不是运行
   授权。它假设 96GB RTX Pro 6000，并把 1.5B/4B-token/RULER 设为 P0；这些
   假设与当前 5090 路线和 response-only 原则不一致。
8. `../pre_rebuttal/` 只提供事实底稿、推导和历史负结果。若与正式 review、AC 或
   更新后的实验 artifact 冲突，不得覆盖后者。
9. `experiments/frequency_band_usage_5090.py` 的 500M checkpoint-only 诊断已完成：广义
   phase predictor 在三臂上将 Q/K norm band 定位到 0/0/1 pair 误差，但该
   band 的等宽 p-RoPE 删除只在 2/12 格中最重要。它支持 band-location
   prediction，不支持把 high-norm band 当成普适因果 useful band。
10. `experiments/frequency_causal_spectrum_5090.py` 与 `experiments/heldout_causal_pruning_5090.py`
    进一步否定静态 band proxy：Q/K norm 与 phase utility 对逐频因果贡献的
    中位 Spearman 分别为 -0.167/-0.045；但独立 selection 选出的干扰频对在
    disjoint test 上联合删除后，Geo/EVQ 的 1K--8K NLL 均显著下降。它支持
    “训练后通道语义与频率绑定发生 OOD 干扰”，不构成新方法或论文 claim。

0723 宽计划中提到的 `zWsa`、`Dz6s` 原始 review 当前不在仓库。补齐逐字来源
前，只能称为计划中的问题概括，不能称为已核验 reviewer request。

## 2. Reviewer 27bE 映射

| ID | 实际问题 | 现有材料能回答什么 | 仍缺什么 / 当前动作 |
| --- | --- | --- | --- |
| `R27bE.1` | `C_app → pure-tether → small-\(\tau\) → deployed \(\tau\)` 未单独归因 | pre-rebuttal 理论审计可严格拆开 exact surrogate、conditional proxy 和 empirical rule | 回答中承认链条不等价；若跑实验，只用 shape/tau 包做直接归因 |
| `R27bE.2` | 小模型、单一 base、单一架构谱系 | held-out base=1M / `d_head=128` 三 seed 已完成且各外推长度均改善 | 仍不补“大模型泛化”；生产规模请求保持 open |
| `R27bE.3` | DAPE 比较混合 shape、capacity 和 tuning | full audit 已确认该行实际是 shared learnable `inv_freq` | 必须纠正方法身份；固定 schedule 对照只能回答 shape，不能修复 DAPE head-to-head |
| `R27bE.4` | tuned \(\tau\) 和 matched non-cosh schedule | 独立 tau sweep 与三 seed matched-shape 已完成；Cosh 有效但不唯一最优 | 用完整正负结果收窄 claim，不能再声称 Cosh 唯一对应 attention optimum |
| `R27bE.5` | held-out base 与更大规模预注册训练 | held-out base=1M、`d_head=128` 前半问已有三 seed 结果 | reviewer-grade 大模型新结果仍缺失 |

## 3. 当前实验登记

| 包 | 直接问题 | 仓库状态 | 允许的结论 |
| --- | --- | --- | --- |
| `experiments/fmrope_125m_l256/` | FMRoPE 的 train/inference base retarget 是否解释 EVQ 效果 | 结果已汇总；FMRoPE/YaRN range scaling 明显强于 raw schedules | 方法级小模型诊断；不能称 EVQ 替代 range scaling |
| `experiments/fmrope_125m_l256_500m/` | 增加训练 token 是否修复 100M undertraining | 500M 路线与报告已登记 | 只能解释该小模型训练预算 |
| `experiments/fmrope_evq_combo_l256/` | EVQ 与 target-aware FMR range 的组合 | 已完成并写入总报告 | 现有组合不支持协同或性能优越性 |
| `experiments/reviewer27be_shape_base/` | Paper-Geo/EVQ、tau、matched shape、held-out base/head 与 native Std-RoPE | 三 seed 结果已汇总；native/real-shape 聚合 JSON 已跟踪 | 回答 shape/base 归因，同时证明 Cosh 不唯一最优 |
| `experiments/mla_scarcity_5090/` + `theory_results/mla_scarcity_seed42_result_20260724.json` | 稀缺 active frequency budget 下 shape gain 是否超过 range gain | seed-42 六臂与独立 test 已完成；不扩 seed | raw practical claim 未通过；YaRN-derived K=8 长外推趋势仅作 single-seed/operator-qualified supporting diagnostic |
| `experiments/mla_yarn_operator_parity_5090/` | native/EVQ 共用 YaRN mask/mscale 的 operator parity | runner 已准备；没有 fresh anchors/READY | 仅是备选执行包，不是结果 |
| `experiments/frequency_band_usage_5090.py` | FMR phase predictor 是否推广到 EVQ，预测 band 是否具有因果贡献 | seed-42 500M 三 checkpoint 已完成；predictor 命中但 causal gate 失败 | 仅支持 channel-location 机制；不得据此提出 band-centered 新方法 |
| `experiments/frequency_causal_spectrum_5090.py` + `experiments/heldout_causal_pruning_5090.py` | 逐频因果贡献能否由静态 proxy 预测、干扰是否在 held-out anchors 泛化 | seed-42 500M 三 checkpoint 已完成；proxy 失败，held-out pruning 为正 | 支持 channel-frequency binding/OOD interference 诊断；不是新推理方法或 paper claim |
| `../../experiments/rebuttal_2026/sft_distillation/` | 为 Paper-Geo/EVQ 生成完全相同的通用短上下文能力 SFT 数据 | code + offline dry-run present；API audit not yet run | 先 100 条人工门禁，再 3000/400/400 pilot；不使用或模仿 RULER |

这些从零训练包都使用全参数训练，使频率表从 step 1 参与 attention；它们没有
把 8B 预训练模型能否被短 LoRA 重写混入频率分配比较。反过来，它们也不能回答
生产规模模型是否迁移。

## 3A. AC 映射

| ID | 核心问题 | 当前证据 | 仍缺什么 |
| --- | --- | --- | --- |
| `AC.1` | 相对 FMRoPE/dead-frequency 的新颖性 | FMRoPE 小模型对照已完成，并暴露 raw EVQ 的 range/shape 混合 | matched-range shape 结果与准确 related-work positioning |
| `AC.2` | 规模、benchmark 与下游不足 | 当前仍以小模型机制诊断为主；8B LoRA 未形成能力提升 | 更强 benchmark 或更大模型的受控证据 |
| `AC.3` | surrogate/cosh/operating rule 未闭环 | pre-rebuttal 理论审计和 Phase16 重算已收窄 claim | Geo/Cosh/Exp matched-range 归因 |
| `AC.4` | 只有 score-changing 证据才可能改变推荐 | 当前负结果和边界已完整保留 | 在预算内先跑 seed-42 gate，再决定是否扩展 |

## 4. 回答与披露边界

- `R27bE.1/.4` 已直接触发 \(\tau\) 与 surrogate-to-deployment 的边界。若回答
  使用 KL 论证，必须明确 ordinary KL 从 \(O(\tau^4)\) 起，不能沿用旧推导。
- `R27bE.3` 已直接触发 submitted “DAPE” 行。回复中必须称其为
  `learnable shared inverse frequencies`，不能继续用 DAPE 身份辩护。
- `Std-Geo` 固定为 \(u_k=k/K\)；`Paper-Geo` 固定为
  \(u_k=(k+\tfrac12)/K\)；`EVQ-Cosh` 使用与 Paper-Geo 相同的 midpoint
  quantizer。主比较只能写 Paper-Geo vs EVQ，Std-Geo 是小规模附加消融。
- 在 `d=64,b=500K`，Paper-Geo/Std-Geo 频率比恒为
  `0.8146172338565447`，波长比为 `1.2275703955658044`；它不是一个单一
  standard-base 替换。若逐通道强行换算，只有 \(k>0\) 时存在
  \(b_{\mathrm{eff},k}=b^{1+1/(2k)}\)，而 \(k=0\) 无解。
- 历史 Primary-II 结果记录的 hash 前缀与保存的同构 `inv_freq.npy`
  完全匹配 `88654f1fe2a414d3...`；但对应 125M 主 checkpoint bytes 不在
  仓库，不能声称完成了历史 checkpoint byte-level audit。
- fixed-ramp/official-YaRN 身份、`c_coll` 和 Phase16 reporting 并未被 27bE
  明确点名。若回复不依赖这些证据，不要在每条回答中发散；是否做一条合并
  integrity disclosure 由作者决定。
- 任何新实验都只能回答它实际隔离的变量。负结果、无效 run、seed 和评测
  protocol 不得选择性省略。

## 5. 下一步顺序

1. profiled-residual 与 frequency-band bridge 均已失败；逐频 follow-up 只支持
   channel-frequency binding/OOD interference。未经新的明确决策，不启动
   matched-range 新训练或 band-centered EVQ-v2，也不再追加 raw-tau sweep。
2. 补齐其余正式 reviews 的逐字文件和来源哈希；未补齐前不做“全 reviewer”
   优先级判断。
3. 先写 `R27bE.1/.3/.4` 的短回答骨架，明确 correction、existing evidence、
   remaining limitation。
4. MLA scarcity 已终止：不得因原注册 gate 的形式 PASS 重启 seed 43/88；
   实际 native-advantage 门禁失败，详见实验报告第 8 节。
5. 新结果先做 raw artifact/provenance 审核，再决定是否进入 response；现有
   shape/base 结果也必须保持报告中的负面边界。
6. 不自动启动 1.5B/4B-token/RULER campaign；只有原始 reviewer/AC 文本与预算
   共同证明其 score-changing 价值时重新立项。
