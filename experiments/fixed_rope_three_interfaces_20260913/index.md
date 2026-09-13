# 固定 RoPE 表区间确认与三接口实验流水线

更新：2026-09-13。这里实现
`fixed_rope_three_interfaces_execution_plan_20260913.md` 中可直接执行的部分，但按当前
论文目标重排了优先级：先确认已经出现强开发信号的 Llama Solver 固定表，再做
band、profile depth 和 transition 的机制干预。当前状态是 **代码完成、目标 GPU
环境已验证并完成多轮 OLMo/Llama S=4 实验**。当前结果owner见
[Llama S=4与三接口报告](../../docs/research/next_stage_20260912/LLAMA_S4_RANGE_CONFIRM_AND_INTERFACE_RESULT_20260913.md)；
不得再按本页早期“未运行”状态重复启动旧队列。

研究层总判断见
[区间最优固定表研究总自查](../../docs/research/next_stage_20260912/RANGE_OPTIMAL_FIXED_ROPE_SELF_AUDIT_20260913.md)。

## 审计后的执行选择

采纳：

- 一张表用于所有层和全部长度；设计倍率 `S` 与运行长度严格分开。
- 真实主指标为 RULER 完整输出 official contains、task-equal 曲线、log-length AUC、
  worst-length、逐任务及 EOS/cap。
- low 只诊断协议、明显崩塌和交互方向，不充当自动淘汰门槛。
- 历史结果仅按 prompt hash 补缺口；每份复用结果必须附覆盖或运行 receipt，能读取的
  model/table/decoder/precision 身份逐项核对。
- Llama/Qwen 长窗使用 chunked prefill；同一模型只加载一次并顺序安装多张固定表。

拒绝或降级：

- 不把 checkpoint replay 当任务优化目标，也不允许 replay 分数过滤真实任务候选。
- `m=c b` 改变整个 profile，不称“只改低频 tail”。
- C1→C2→C3 只能解释为顺序条件效应；OLMo 使用同 band 的
  `shape × depth` 2×2 检查交互。
- static YaRN 只作零训练诊断，不冒充官方需要训练的 YaRN 方法对照。
- 不从有限长度网格宣称连续 `[L,SL]` 无深坑。

## 文件

- `tables.py`：包装现成精确数组；构造 Native/BM/MrPro/static YaRN；产生完整 exponent、
  increment support、band envelope、depth、sum、gain 和 FP32 table hash receipt。
- `prepare_server.py`：从当前服务器资产冻结 low/native 子面板、候选表、实验合同和
  missing-row queue；只准备，不加载模型。
- `pipeline.py`：身份检查、永久基线复用、断点 queue、official 重评分、task-equal
  AUC、worst/regret 和 paired bootstrap。
- `resident_eval.py`：单模型驻留、多表顺序评测；每次换表校验 FP32 hash；已有相同
  table/prompt 输出可在同一流水线中无 GPU 复制复用。
- `night_run.py`：薄调度外壳，只负责固定阶段顺序、失败停止和阶段后评分，不根据分数
  取消后续实验。
- `matched_generation_report.py`：从多份已完成 JSONL 合并严格同prompt臂，生成
  task-equal曲线、log-AUC、worst、任务族和配对重采样报告。
- `margin_direction.py`：在冻结checkpoint上，以指定合法答案token margin计算
  transition局部方向并做正反有限响应；CAL结果不替代自由生成。
- `prepare_margin_targets.py`：把已被official scorer接受的教师输出冻结为margin目标，
  并显式拆分fit/select。
- `target_support_z.py`、`full_z_fwe_repair.py`：固定目标频谱两端、直接参数化63个正
  log-frequency gap，并以FWE首token margin求最小位移；首轮真实有限响应未通过，
  没有产生可评候选。
- `exponent_box_z.py`、`constrained_full_z_fwe_repair.py`：直接参数化内部exponent，
  显式强制`0≤m≤1`、单调和固定端点；两级信赖域精确响应均未通过CAL，未生成候选。
- `../../tests/test_fixed_rope_three_interfaces.py`：表几何、全 profile depth、断点恢复、
  task-equal 短窗汇总和 receipt 合同单测。
- `scale_transfer_audit.py`：验证冻结 exponent allocation 在不同S之间的逐槽恒等式，
  分开共同绝对长度的频率变化与各自目标端的文本长度变化。
- `matched_point_report.py`：为单一目标长度生成同prompt、task-equal、配对bootstrap报告。
- `../llama3_60dir_20260911/prepare_planb_panel.py`：旧Llama严格身份合同保持默认不变；
  新增显式generic checkpoint模式和任务子集，用于生成OLMo S8缺失的32K冻结输入。

checkpoint Q/K capture、finite circular replay 和 active-set/QP 基础实现继续复用
[`checkpoint_attention_replay_20260913`](../checkpoint_attention_replay_20260913/index.md)。
只有它先通过 R0 实现等价和 R1 已有候选回放排序审计，才允许冻结一张新表进入真实
任务；它不是当前 Llama 论文确认的前置条件。

## 已完成的关键判决

- Llama S=4 `mix075 [16,34]` 的log-gain中点在Core-6×8/16/32K×12行/格上得到
  83.61/81.62/76.94，AUC 80.95、worst 76.94；BM/MrPro/C42的AUC分别为
  74.68/73.15/75.06，worst分别为61.62/58.33/68.96。
  相对BM/MrPro/C42的AUC配对区间均为正。它仍未修复Native约9.26pp缺口。
- 未参与gain选择的第二6行块上，候选相对标准gain、BM、MrPro、C42的AUC区间也
  均为正；但它不是全新benchmark，且部分rows曾用于其他table开发。
- 同prompt Native@8K为92.87，候选83.61，主要是FWE立即EOS；当前候选是强远端
  Pareto点，不是满足Native保持的最终解。
- 把整条profile的slow端从`/4`拉回`/3.6`后，32K六任务全部归零；这否定该联合
  depth修复，但不是tail-only因果结论。
- 显式约束`0≤m≤1`且单调的full-z一阶QP在两级信赖域都未通过3条FWE真实有限
  margin，未产生生成候选；只否定该父表附近的当前一阶射线。
- gain1下完全front-loaded `[16,34]`把32K MK2/QA/VT宏平均从33.33提高到45.56，
  仍低于中点gain父表54.44；进一步前移到`[14,32]`降至43.33。中点gain配完全
  front-loaded则使8K FWE六条全部立即EOS。本补偿路线已收束，不进入完整mini。
- OLMo band局部四格、tail更快侧、transition mix和answer-margin方向均已运行；
  正负结果和解释边界统一保留在结果owner中。
- Llama FWE局部transition与full-z一阶修复均未把首token margin推过0；full-z QP还
  越出`m∈[0,1]`预算，因此没有追加自由生成或扩大步长。

## 原始冻结队列（历史计划）

1. `paper_confirm`：Llama Solver `[14,32]`、BM、MrPro、static YaRN 在同一
   Core-6 × 8/32/64K × 18 行面板重跑；另用 8K × Core-6 × 18 行检查 Solver 与
   Native。旧 64K parity 失败结果不复用。
2. `olmo_depth_shape`：C42/BM 在 `[14,31]` 上的 full-depth 与 Dlog-depth 2×2；
   永久 BM/MrPro 只从已完成 mini 缓存取对应 low 子集。
3. `qwen_confirm_depth`：补齐既有 C42 `[22,39]`、BM、MrPro mini 缺口，同时做
   C42 full-depth 对 Dlog-depth 的 low 干预。Qwen 结果是开发/回顾证据，不伪装为
   新理论留出。
4. `llama_depth`：在不改变 band、shape 和 gain 时比较 C42 full-depth/Dlog-depth；
   Solver/BM/MrPro 的 low rows 从第一阶段相同 table/prompt 输出复用。
5. `checkpoint_replay`：仅在 R0/R1 通过后执行，当前合同保留阶段但不自动产生 GPU job。

## 服务器完整开机后的命令

先做极轻量检查：

```bash
cd /root/autodl-tmp/hybrid-rope
python -m py_compile experiments/fixed_rope_three_interfaces_20260913/*.py
pytest -q tests/test_fixed_rope_three_interfaces.py tests/test_checkpoint_attention_replay.py
python -m experiments.fixed_rope_three_interfaces_20260913.prepare_server
python -m experiments.fixed_rope_three_interfaces_20260913.night_run \
  --queue /root/autodl-tmp/fixed_rope_three_interfaces_20260913/queue/queue.json
```

确认 plan-only 输出和显卡状态后，唯一启动命令为：

```bash
python -m experiments.fixed_rope_three_interfaces_20260913.night_run \
  --queue /root/autodl-tmp/fixed_rope_three_interfaces_20260913/queue/queue.json \
  --execute
```

不要在当前无卡容器或本地弱机上运行这些测试来冒充服务器验收。实际启动后由 15 分钟
heartbeat 检查 `night_status.json`、各 job 的 `live.json/status.json`、GPU 利用率、
显存和磁盘；异常时先区分工程失败、身份失败和真实任务结果，不自动筛掉后续候选。

## 结果边界

第一阶段成功意味着：在明确的 Core-6/三长度 mini 合同上，当前 Solver 表的
task-equal AUC、worst-length 与 Native 保持相对 BM/MrPro 有论文推进价值；它不自动
意味着逐任务、逐长度支配。失败则说明现有小屏强信号没有扩展到宽任务同口径确认，
但仍不否定 band/profile 研究空间。三接口干预的正负结果分别回答条件化变量是否有
任务信号，不能被写成三个独立主效应或通用闭式最优性。
