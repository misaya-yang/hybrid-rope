# Round 12 预注册增补 V2（2026-09-06）

**状态**：本增补是 `PREGLUCTION_ROUND12.md`（v1）的增补，不废除 v1；
v1 中未被本文件修改的条款（冻结资产清单、评分口径、禁止事项）全部继续有效。

**授权来源**：
1. Pro 第一性原理综合文档 `HYBRID_ROPE_FIRST_PRINCIPLES_SYNTHESIS.md`（2026-09-06）
   §11（仅授权两个只读诊断问题）、§12（伪代码纪律）、§14（不新增训练矩阵）。
2. 用户 2026-09-06 裁决："你不需要收窄，你只需要重新安排优先级；对于 7B 和
   更多数据，我们可以尝试一下 4080 是否支持（虽然大概率爆炸）；三者结合。"

## 1. 优先级变更记录（相对 v1）
| 项目 | v1 | v2 |
|---|---|---|
| P0 诊断（§11.1 写分解 + §11.2 分叉 E/B） | 无 | **新增，置顶** |
| 4080/7B 能力探测 | 无（7B 直接排入 Track A） | 新增；7B 由探测门控 |
| Track A 1B 静态矩阵 | P0 | P1（内容不变） |
| Track B CPT+SFT 训练 | P1 | **P3，降级、条件执行**（需当日用户明确确认；Pro §0/§14 明示反对，用户裁决保留但降级——两方立场如实记录于 RUNBOOK） |
| 冻结表/任务/视图/CPT/配方 | 冻结 | **不变** |

## 2. P0a 预注册：内容分叉 E/B（§11.2，脚本 `code/diag_fork_eb.py`）
- **问题**：各系统是否"两个世界都对"，即 E > |B|（§6.1）；失败时损失来自
  条件判别项还是质量分配项（§6.2 精确分解）。
- **系统**：T0（arm N 无 adapter）、Z0（arm Z）、ZC（Z+`out/train/step_128`）、
  ZF（Z+`out_zf/train/step_128`）、ON（N+`out_on/train/step_128`）。
  adapter 全部为 round-11 已注册检查点，不重训、不重选。
- **数据**：`transport_views.jsonl`（冻结），family=single_evidence，
  split=**validation**，layouts={compact, near, far}。
  注：round-11 receipt 登记在 validation split（128 行=64 组×2 世界），
  与 round12_tasks（train split）不同源，不可互换。
- **指标**：d0、d1、E、B、pair_order_correct ⇔ E>|B|；
  margins（gold vs best other / vs best non-candidate）；
  §6.2 分解：ce_conditional_discrimination、ce_mass_allocation、
  ce_pair_actual、ce_identity_residual（代数恒等式，预期 ≤1e-6，超出即实现错误）。
- **登记分数**：已注册的逐行 strict receipt 原样 join（full_exact_eos），
  不重打分、不重排候选、不改 strict 定义。T0 无逐行 far/near receipt，
  相应字段记 null，聚合值引用 review.json 既有披露（0/0）。
- **性质**：teacher-forced 诊断，不是生成主张。

## 3. P0b 预注册：attention-write / logit 分解（§11.1，脚本 `code/diag_write_decomp.py`）
- **问题**：ZF/ON 相对 T0 的输出变化中，多少来自 value-write 内容变化
  （§5.2 第一项）、多少来自路由变化（第二项）、多少来自新增 token 竞争
  （第三项 β(ū_D−ū_S)）；以及 §2.3 中直接项与继承项的相对大小。
- **实例选择规则（确定性，预注册）**：validation split、family=single_evidence、
  far 16384、按 `source_proofs.jsonl` 文件顺序，取在 Z0、ZC、ZF、ON 四个系统
  receipt 中**均存在**的前两个 semantic_id。执行时把锁定的两个 id 记入
  `diag/wd/pregluction_locked_instances.json`（含选择规则全文与时间戳）。
- **已锁定（2026-09-06，无卡模式下按上述规则预选，共 32 个合格组）**：
  1. `0ee492a7351f230cc7aac34a6970df2e00f7aa838fb5569e3fbd8cc9208a9a2d`
  2. `50b88ec1b6454878c549aa1e5cd39cb8ebdde7979aeddb98d9bbd97350b7afa0`
  执行日必须复核：两个 id 在 validation far-16384 视图中双世界齐全、在四个
  系统 receipt 中均存在；不符则按同一规则现场重选并在增补记录原因，
  不得主观挑选。
- **层/头/查询位置**：层 {0,4,8,12,15}；保留全部 16 头；query = prompt 最后一个
  位置。所有系统相同，不因结果调整。
- **实现保真度门槛（先于解读）**：
  - 外部重建核对（计算的 attn_out vs hook 捕获）相对误差 ≤5e-3；
  - §5.2 三项恒等式残差 ≤1e-4；
  - §2.3 直接/继承重建相对残差报告 max/mean/p99，p99 ≤1e-3 视为通过。
  任一不达标 = 实现或对齐错误，**停止解读**，保留现场上报；不得放宽容差。
- **参考系**：T0 先跑，dump q_pre/k_pre/cos/sin（fp16）作为参考；
  Z0/ZF/ON 以 `--ref-dump` 对齐。保存带符号项（不只范数），遵循 §12。
- **报告**：`--mode report` 产出 `theory_observation_map.json`，
  携带 §12 旗标：preserve_all_registered_negative_results=true、
  no_method_or_checkpoint_selection=true、no_new_training=true。

## 4. P2 预注册：4080/7B 能力探测（脚本 `code/probe_capability.py`）
- 分级：stage0 环境 → stage1 1B bf16 前向基线（2K/8K/16K，必须过）→
  stage2 7B bf16 CPU 载入 + `.to(cuda)`（预期 OOM 边界）→ stage3 7B 前向
  1K→16K（仅当 stage2 通过）。
- **失败即结果**：每阶段后立即写 `probe_result.json`；OOM 记录为边界，
  不用量化/卸载绕过（属方法变更，需另行决策）。
- 门控规则：仅当 stage2+stage3(16K) 全过，才进入 7B Track A（先 64 条资格，
  后全量 {N,Z,Y,M}）；否则 7B 矩阵当日不执行，升级路径记入当日记录。

## 5. P0 期间禁止事项（承 §11/§12，叠加于 v1 禁止清单）
- 不新增表、增益、头选择器或任何候选；不改冻结资产。
- P0 内不发生任何训练；诊断输出不得用于重选检查点或方法。
- 所有负结果原样保留、原样披露（含 T0 far/near 0/0 的再确认）。
- 不重启已暂停的 351 案例盲标与 teacher-prefix 面板（§11.2 明示）。
- P0 结束后只做解读小结（对照 §11.3 决策表），不自动启动任何后续训练。

## 6. 预算与顺序
见 `RUNBOOK_ROUND12.md` v2：Phase 0 ≤2.5 GPU-h → Phase 1 ≤3 → Phase 2 条件
≤6.5 → Phase 3 条件 ≤12；当日 ≤20 GPU-h；单 GPU 进程；失败保留不续跑。
