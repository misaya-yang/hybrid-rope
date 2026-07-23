# Rebuttal 0723: current review cycle

最后更新：2026-07-24

状态：`official_review_received / triage_only / no_response_yet`

本目录是当前 rebuttal 的唯一操作入口。先固定 reviewer 实际问了什么，再决定
是否使用历史分析或新实验；不得反过来从已有材料拼一个泛化答辩。

## 1. 当前证据权限

1. `00_REVIEWER_27BE_OFFICIAL_REVIEW.md` 是仓库内目前唯一逐字、带来源哈希的
   正式 review，稳定 ID 为 `R27bE.1`–`R27bE.5`。
2. `fmrope_125m_l256/` 和 `reviewer27be_shape_base/` 是协议与代码，不是结果。
   仓库内目前没有对应 `results.json`、完成 receipt 或可写进 rebuttal 的数字。
   两个包共享 `geo_rope_contract.py` 与
   `FREQUENCY_DEFINITION_MANIFEST.json`，不再依赖含混的 `Geo` 名称。
3. `EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md` 是宽方案库，不是运行
   授权。它假设 96GB RTX Pro 6000，并把 1.5B/4B-token/RULER 设为 P0；这些
   假设与当前 5090 路线和 response-only 原则不一致。
4. `../pre_rebuttal/` 只提供事实底稿、推导和历史负结果。若与正式 review 或
   更新后的实验 artifact 冲突，不得覆盖后者。

0723 宽计划中提到的 `zWsa`、`Dz6s` 原始 review 当前不在仓库。补齐逐字来源
前，只能称为计划中的问题概括，不能称为已核验 reviewer request。

## 2. Reviewer 27bE 映射

| ID | 实际问题 | 现有材料能回答什么 | 仍缺什么 / 当前动作 |
| --- | --- | --- | --- |
| `R27bE.1` | `C_app → pure-tether → small-\(\tau\) → deployed \(\tau\)` 未单独归因 | pre-rebuttal 理论审计可严格拆开 exact surrogate、conditional proxy 和 empirical rule | 回答中承认链条不等价；若跑实验，只用 shape/tau 包做直接归因 |
| `R27bE.2` | 小模型、单一 base、单一架构谱系 | 已有小模型与 supporting scale 事实可如实列出 | `heldout_b1m_d128` 只补 held-out base / `d_head`，不补“大模型泛化” |
| `R27bE.3` | DAPE 比较混合 shape、capacity 和 tuning | full audit 已确认该行实际是 shared learnable `inv_freq` | 必须纠正方法身份；固定 schedule 对照只能回答 shape，不能修复 DAPE head-to-head |
| `R27bE.4` | tuned \(\tau\) 和 matched non-cosh schedule | `reviewer27be_shape_base/shape_l128` 直接预注册 tau scan、span-matched power/exp | 尚无结果；正负都必须完整报告 |
| `R27bE.5` | held-out base 与更大规模预注册训练 | `heldout_b1m_d128` 直接覆盖 base=1M、`d_head=128` 的前半问 | 当前没有 reviewer-grade 大模型新结果；不得用宽 1.5B 计划冒充完成 |

## 3. 当前实验登记

| 包 | 直接问题 | 仓库状态 | 允许的结论 |
| --- | --- | --- | --- |
| `fmrope_125m_l256/` | 在 Paper-Geo/EVQ 主配对旁，FMRoPE 的 train/inference base retarget 是否解释 EVQ 效果 | spec/code present；no repository result | 方法级单 seed 小模型比较；只有原始 reviewer source 补齐后才可称 review-triggered |
| `reviewer27be_shape_base/` | Paper-Geo vs EVQ、tuned \(\tau\)、non-cosh matched shape、held-out base/`d_head`；seed-42 Std-Geo 小消融 | spec/code present；no repository result | 对 `R27bE.1/.2/.4/.5` 的定向证据；不能支持普适最优或大模型 SOTA |
| `../../experiments/rebuttal_2026/sft_distillation/` | 为 Paper-Geo/EVQ 生成完全相同的通用短上下文能力 SFT 数据 | code + offline dry-run present；API audit not yet run | 先 100 条人工门禁，再 3000/400/400 pilot；不使用或模仿 RULER |

两个包都从随机初始化全参数训练，使频率表从 step 1 参与 attention；它们没有
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
3. 只有实验数据、环境和 READY receipt 全部离线验证后，才由作者明确选择是否
   启动两个窄实验包。
   旧 schema-v1 `data_manifest.json` / `preflight.json` 不含 frequency
   contract，必须重新生成；已校验的 token NPY/parquet cache 不必重下。
4. 结果出来后先做 raw artifact/provenance 审核，再决定是否进入 response。
5. 不自动启动 1.5B/4B-token/RULER campaign；只有原始 reviewer 文本与预算
   共同证明其 score-changing 价值时重新立项。
