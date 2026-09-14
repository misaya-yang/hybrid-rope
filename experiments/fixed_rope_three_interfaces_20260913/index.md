# 固定 RoPE 表区间确认与三接口实验流水线

更新：2026-09-13。这里实现
`fixed_rope_three_interfaces_execution_plan_20260913.md` 中可直接执行的部分，但按当前
论文目标重排了优先级：先确认已经出现强开发信号的 Llama Solver 固定表，再做
band、profile depth 和 transition 的机制干预。当前状态是 **代码完成，目标 GPU
环境已完成OLMo/Llama/Qwen多轮实验，本轮队列已自然结束**。阶段总判决见
[理论与实验阶段报告](../../docs/research/next_stage_20260912/THEORY_AND_EXPERIMENT_PAUSE_REPORT_20260914.md)，
Llama S4细节见[结果owner](../../docs/research/next_stage_20260912/LLAMA_S4_RANGE_CONFIRM_AND_INTERFACE_RESULT_20260913.md)；
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
- `../olmo_recovery_20260912/recovery_v2_eval.py`：支持`--batch-size`对相邻等长、等输出
  上限样本做Flash-SDPA批量greedy；本轮Qwen队列未传该参数而使用默认1，后续小模型
  应先用2/4短canary冻结最大稳定批量，Llama 64K仍保持单条。

## 2026-09-14 fixed-u决定性方法补充

fixed-u计划及其停止条件见[决定性闭环](../../docs/research/next_stage_20260912/DECISIVE_FIXED_U_METHOD_PLAN_20260914.md)。
OLMo机制块已经完成，[结果owner](../../docs/research/next_stage_20260912/OLMO_S8_FIXED_U_TRANSPORT_RESULT_20260914.md)
显示fixed-u相对fixed-m的AUC差为负且区间不跨零，因此终止该倍率迁移分支；作者随后要求
整体零训练研究继续，不能把分支停止扩大成总任务暂停。

- `tables.py scale-transport`：从实际S4父表直接生成现有resident runner可安装的
  `fixed_m`或`fixed_u`冻结receipt；实际Native FP32数组决定band log坐标。
- `math_and_transport.py`：31项CPU恒等式核验和实际数组capsule；不加载checkpoint。
- `tailspline_verification.py`：独立核验有限网格TailSpline闭式增量、KKT唯一解、
  one-sided roughness最优值及其与有限网格BM/front精确混合的等价性；明确不把CPU代数
  当作checkpoint或benchmark优势证据。
- `factorial_report.py`：完成方法胜出后才使用的同prompt table×gain四格分解。
- `causal_intervention.py`：实现中层当前query的phase-only/donor-output诊断，但已从
  主GPU队列移除；它不能决定最终方法是否胜过强基线。
- `run_qwen25_s2_full324.sh`：在Qwen2.5-3B上串行运行mix075、MrPro、static YaRN和BM，
  使用同一Core-6×32/48/64K×18行/格完整面板、batch 2和冻结静态表；单臂完成不停止队列。
  先前batch 4正式运行在114/324遇到48K峰值OOM，partial raw保留但不拼入正式结果。
- `analyze_qwen25_s2_full324.sh`：四臂结束后独立核验324行、prompt集合、generated ids、
  official score、table/gain receipt及raw hash，并生成mix075对三个强基线的配对区间报告；
  它在后台CPU运行，不阻塞后继Llama占用GPU。
- `run_llama_s4_factorial_gap48.sh`：复用三个已完成且48个prompt完全重合的析因格，
  只补C42 table×mix075 midpoint gain的48条缺口，随后生成VT/FWE、8K/32K的配对2×2报告。
- `chain_qwen_to_llama_factorial.sh`：只在Qwen四臂监督器写出`QUEUE_COMPLETE`后衔接上述
  Llama缺格；若Qwen异常退出则拒绝越过失败继续运行，交给监控按原合同恢复。
- `chain_qwen_to_analysis.sh`：为本次已经启动的S2监督器补挂CPU分析；等待同一
  `QUEUE_COMPLETE`后核验和出报告，不与Llama的GPU衔接互相阻塞。
- `run_qwen25_s4_full.sh`：在未跑过同表的Qwen2.5-3B已有完整资产上预注册S4比较；
  32K/64K复用18行/格面板，128K复用32行/格heldout面板，依次运行mix075、canonical
  MrPro、static YaRN和同band BM。近端batch 2、128K batch 1，生成三长度配对报告。
- `run_tailspline_qwen25_s2_full324.sh`：按作者2026-09-14新方案，以精确有限网格
  TailSpline作为唯一候选，在统一canonical band/gain下与MrPro、static YaRN、BM使用
  同一324条Qwen2.5-3B面板比较；旧mix075只作为近似开发prior，不冒充精确表结果。
- `chain_current_mix_to_tailspline.sh`：等待已接近完成的mix075首臂写出324行和COMPLETE，
  随即终止已冻结的旧监督器并启动TailSpline；不会继续旧S2三基线或Llama局部修复队列。
- `../../tests/test_rope_today_plan.py`：fixed-u数值/复合性、capsule、四格分解、案例冻结
  与完整key竞争有限干预的CPU检查。

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
- Llama S8在64K相对MrPro为`+4.95pp`且区间为正，与BM持平；Native 8K为
  `-15.83pp`且区间为负，因此只形成端点Pareto点，没有匹配区间解。
- OLMo S8父gain候选在4/16/32K采样网格的AUC为41.87，BM/MrPro为20.28/6.02；
  绝对32K仅15且Native未测。
- Qwen S2累计18行/格AUC为79.72，BM/MrPro/C42为75.84/77.58/77.43；仅相对BM的
  95%区间为正。独立追加12行块三项差值均跨0；Native 32K累计差`+6.51pp`且区间
  为正。详见[Qwen owner](../../docs/research/next_stage_20260912/QWEN_S2_MIX075_RANGE_RESULT_20260913.md)。

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

## 历史启动命令（当前不得自动重放）

以下命令只保留可复现性。本轮队列已完成；没有新的作者指令时不得据此重启旧queue。

当未来明确恢复同一队列时，先做极轻量检查：

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

不要在当前无卡容器或本地弱机上运行这些测试来冒充服务器验收。旧heartbeat已不是
当前执行合同；异常时仍需区分工程失败、身份失败和真实任务结果，不自动筛掉候选。

## 结果边界

第一阶段成功意味着：在明确的 Core-6/三长度 mini 合同上，当前 Solver 表的
task-equal AUC、worst-length 与 Native 保持相对 BM/MrPro 有论文推进价值；它不自动
意味着逐任务、逐长度支配。失败则说明现有小屏强信号没有扩展到宽任务同口径确认，
但仍不否定 band/profile 研究空间。三接口干预的正负结果分别回答条件化变量是否有
任务信号，不能被写成三个独立主效应或通用闭式最优性。
