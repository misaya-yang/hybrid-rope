# Rebuttal 0723: current review cycle

最后更新：2026-07-24

状态：`official_review_received / triage_only / no_response_yet`

本目录是当前 rebuttal 的唯一操作入口。先固定 reviewer 实际问了什么，再决定
是否使用历史分析或新实验；不得反过来从已有材料拼一个泛化答辩。

## 1. 当前证据权限

1. `00_REVIEWER_27BE_OFFICIAL_REVIEW.md` 是仓库内目前唯一逐字、带来源哈希的
   正式 review，稳定 ID 为 `R27bE.1`–`R27bE.5`。
2. `EXPERIMENT_REPORT_20260724.md` 是当前数值结论入口；FMRoPE、500M
   undertraining、held-out base/head、native-Geo 与 real-shape 结果必须连同
   各节 claim boundary 使用。native/real-shape 聚合原始结果另存为
   `native_attention_shape_l128_results_20260724.json`。
3. `mla_scarcity_5090/` 是下一项已冻结且通过本地测试的实验代码，尚无训练
   结果。它固定模型参数量，只改变 active frequency-pair budget；不得把
   code-ready 写成 evidence-ready。
4. `EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md` 是宽方案库，不是运行
   授权。它假设 96GB RTX Pro 6000，并把 1.5B/4B-token/RULER 设为 P0；这些
   假设与当前 5090 路线和 response-only 原则不一致。
5. `../pre_rebuttal/` 只提供事实底稿、推导和历史负结果。若与正式 review 或
   更新后的实验 artifact 冲突，不得覆盖后者。

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
| `fmrope_125m_l256/` | FMRoPE 的 train/inference base retarget 是否解释 EVQ 效果 | 结果已汇总；FMRoPE/YaRN range scaling 明显强于 raw schedules | 方法级小模型诊断；不能称 EVQ 替代 range scaling |
| `reviewer27be_shape_base/` | Paper-Geo/EVQ、tau、matched shape、held-out base/head 与 native Std-RoPE | 三 seed 结果已汇总；native/real-shape 聚合 JSON 已跟踪 | 回答 shape/base 归因，同时证明 Cosh 不唯一最优 |
| `mla_scarcity_5090/` | 稀缺 active frequency budget 下 shape gain 是否超过 range gain | spec/code/tests present；no training result | seed-42 gate 与三 seed test 完成后才可支持 scarce-budget interaction |
| `../../experiments/rebuttal_2026/sft_distillation/` | 为 Paper-Geo/EVQ 生成完全相同的通用短上下文能力 SFT 数据 | code + offline dry-run present；API audit not yet run | 先 100 条人工门禁，再 3000/400/400 pilot；不使用或模仿 RULER |

这些从零训练包都使用全参数训练，使频率表从 step 1 参与 attention；它们没有
把 8B 预训练模型能否被短 LoRA 重写混入频率分配比较。反过来，它们也不能回答
生产规模模型是否迁移。

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

1. 补齐其余正式 reviews 的逐字文件和来源哈希；未补齐前不做“全 reviewer”
   优先级判断。
2. 先写 `R27bE.1/.3/.4` 的短回答骨架，明确 correction、existing evidence、
   remaining limitation。
3. 若启动 MLA scarcity，只允许在数据、环境、磁盘与 READY receipt 离线验证后
   运行 seed-42 gate；gate 未通过不得扩到 seed 43/88。
4. 新结果先做 raw artifact/provenance 审核，再决定是否进入 response；现有
   shape/base 结果也必须保持报告中的负面边界。
5. 不自动启动 1.5B/4B-token/RULER campaign；只有原始 reviewer 文本与预算
   共同证明其 score-changing 价值时重新立项。
