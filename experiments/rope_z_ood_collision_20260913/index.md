# RoPE phase-OOD / collision 审计与 E0–E5 流水线

更新：2026-09-13。本目录是对两份 Web 方案的实现级二次审查，不是新的 GPU
结果 owner。当前主问题仍是：在给定 Native 长度与 horizon 内，全部层和全部运行
长度只安装一张固定 Native-relative RoPE 表；不按输入长度动态换表。

## 当前状态

- `build_mechanism_arms.py` 已在 CPU 上对 Web 方案指定的 `B=500000, K=32,
  L=2048, factors=2/4/8` 做完公式构造；phase arc按真实2048-token窗口的最大
  relative distance 2047计算，与collision的`0..L-1`域一致。结果是
  **Pareto / 无严格共同 δ**，详见
  [完整 CPU receipt](cpu_mechanism_b500k_k32_l2048.json)。它由标准 `k/K` Geo
  公式生成；正式训练若要使用任何 arm，须以实际运行 Geo tensor 重新传入
  `--geo-json`，不能把本 receipt 的 hash 冒充服务器 tensor 身份。
- 原 purity 合同要求两组 signed target change 均至少 `0.10`、nuisance ratio
  不高于 `0.20` 且保持 25% Geo gap。九个 δ 中没有一个同时通过。`0.02`
  仍保持排序，OOD target change 为 `0.10548`，但 collision target change 只有
  `0.06430`，且 collision 方向的 OOD leakage ratio 为 `0.68888`；`0.03`
  起排序 margin 已失败。因而 receipt 正确标为 `CPU_MECHANISM_FAILURE`。
- 这只说明“当前局部投影＋固定剂量网格不能宣称干净正交隔离”，**不说明 phase
  OOD 或 collision 对模型无效**。完整九档指标、五张表、FP32/FP64 排序、端点、
  displacement 和 tensor hash 都保留；有序且方向正确的 `0.0025–0.02` 均保留为
  Pareto arms，没有被 purity 门删除。
- 远端只读快照（`2026-09-13T14:06:03+08:00`）：C42V24 初始化的正式 solver
  已完成 1 个 accepted feasibility step；full-fit hard constraints 未全部满足，故
  它是 **Pareto fit candidate**，不是严格可行解也不是任务结果。BM 正式 solver
  0 accepted step，不能算新候选。C42 的 select 自由生成随后已由主调度启动；本目录
  没有启动、终止或修改该 GPU 链。后续 select 已完成：`SolverC42_r1` 的七点最弱
  task-macro official 为 `58.33%`（C42V24 `50.00%`），16K 为 `58.33%`
  （C42V24 `50.00%`），但 log-length AUC 为 `62.95%`（C42V24 `68.14%`），
  且 source pair-follow 各扩展表仍为 0。它是小样本上的 Pareto 移动，不是赢家；
  适合保留后进入 internal-confirm，而非按单一 AUC 或最弱点自动筛掉。

## 两份 Web 方案的审查修正

### Phase-OOD × collision 方案

保留的内容：连续 phase-arc 公式、完整 sine/cosine pair 子空间、causal distance
measure、对称正负方向和 CPU 先冻结剂量。这些都是可复算的 operator-level 构造。

必须修正的内容：

1. phase burden 与 canonical overlap 都是无模型权重的 proxy。已有 12-profile 审计
   存在 22 个 OOD-max/SEP 几何支配而任务排序反转；历史上自然 NLL 还曾沿整表走线
   改善而检索准确率显著变差。proxy 只能提出候选，不能替代 held-out greedy 生成。
2. `g_nuisance^T d=0` 只是 Geo 点的一阶条件。有限 δ 下，plus/minus 的 signed
   nuisance contrast 小，并不保证每个单臂相对 Geo 的所有其它谱属性不变。因此
   允许检验“该局部方向的预测”，不能据此识别两个机制是唯一原因。
3. Web 方案把 151.9M、2K、15 条五臂训练写成现成 S1 合同并不准确。仓库现有 S1
   是 Geo/Cosh/full-z、两个 support 的学习期比较；15 条新训练既未授权，也会绕开
   当前优先的冻结 OLMo 区间 solver。CPU purity 已失败时更不能把它自动排入 GPU。
4. 无严格 δ 不是工程报错。流水线用 `strict-feasible / pareto / unresolved` 三态；
   数值或身份损坏才是 `ENGINEERING_ERROR`。

### 固定表 E0–E5 方案

保留的内容：单固定表、worst measured regret 主/AUC 次、两套 Llama panel 不拼接、
CI 方向修正、E2/E3 结果优先复用、E4 不用检索小面板代替 VT/QA/自然任务，以及
E5 必须从共同干预前父状态分支。

必须修正的内容：

1. `gamma6/8` 是已被当前 9/13 计划降为历史比较的方向，gamma3 与表空间 midpoint
   已有负结果；不能按 Web 文档重新排队扫表。
2. 当前 E1 已由模型条件化 solver 承担：固定 OLMo、S=4、band `[14,32]`，以
   answer+EOS NLL、source-counterfactual margin 与 Native KL 提议整表。fit 结果仍是
   proxy；是否有用由未参与梯度的 free generation 决定。
3. hard constraints 不通过的 accepted solver step 不应被悄悄删掉。它作为 Pareto
   candidate 进入 E2，完整报告代价；0-step 则没有产生新表，只记 unresolved。
4. Web 文档建议的 2pp Native/endpoint 门尚未成为当前生成合同中的锁定常数。
   本流水线不自动按它过滤候选；E2 后用不可变 selection receipt 显式记录
   `strict-feasible / pareto / unresolved` 选择与理由。
5. E5 的共同父 checkpoint、两臂 optimizer lineage 与“任意 solver tensor＋
   r32/alpha32 全层 QKVO+FFN”启动器当前未同时证实。代码只做 lineage 审计并输出
   blocked receipt，不伪造 matched 训练命令。

## 修正后的 E0–E5

| 阶段 | 任务 | 当前退出条件 |
|---|---|---|
| E0 | 核对 8/4/4 主任务与 source-CF rows、answer+EOS、四个 fit 基线的一次性身份 | 数据/行身份损坏才是工程阻塞；不做重复 hash 审批 |
| E1 | 复用现有 BM/C42 solver receipts；缺失 run 只在显式 `--execute` 时启动 | 0-step=unresolved；accepted+全约束=strict；accepted+未全过=Pareto；都不是任务胜利 |
| E2 | 对所有 strict/Pareto 候选与四个基线跑 `select` free generation | official、完整答案+EOS、source pair-follow 分列；保留所有候选 |
| E3 | 依据 E2 写 immutable selection receipt，再跑 `internal_confirm` | 开发选择不冒称独立确认；不自动从 Pareto 集删臂 |
| E4 | 仅在明确给出归档自然/端点/VT-QA panel 与匹配 baseline refs 后生成候选 | 缺格留作未测，不跨 panel 拼 AUC/regret |
| E5 | 审计共同父状态、两臂 optimizer lineage 与 r32/alpha32 全层模块合同 | 当前 launcher 身份不足，保持 plan-only/blocked |

## 代码入口

- `mechanisms.py`：phase arc、完整 pair collision、finite-difference 稳定性、
  vectorized/direct 离散交叉核验，以及全 δ 三态审计。
- `build_mechanism_arms.py`：默认 plan-only；`--execute` 仅执行 CPU 构造并写完整
  receipt，不授权 GPU。
- `pipeline.py`：审计现有 E0/E1 receipts，并复用
  `experiments/olmo_recovery_20260912/` 的 solver、evaluation 与 summarizer。默认只打印
  E0–E5 状态；任何模型/GPU命令只有显式 `--execute` 才运行。
- `lock_selection.py`：保留全部 E2 候选的同时，冻结一个进入 E3 的开发选择。

典型 plan-only 调用如下；路径均由当前机器显式传入，不写入仓库导航：

```bash
python -m experiments.rope_z_ood_collision_20260913.pipeline \
  --stage e2 \
  --model <MODEL> \
  --data-dir <RUN_ROOT>/data_r0 \
  --source-cf-dir <RUN_ROOT>/source_cf_r0 \
  --baselines-dir <RUN_ROOT>/baselines_r0 \
  --native-docs <NATIVE_DOCS> \
  --run-root <RUN_ROOT> \
  --solver-run bm=BM_g4=<RUN_ROOT>/solver_bm_formal_r0 \
  --solver-run SolverC42_r1=C42V24_g4=<RUN_ROOT>/solver_c42_formal_r0
```

输出中的命令带 `--execute`，但上面的顶层调用未带 `--execute` 时不会执行它们。
正式 E2 前再次读取当前 `select_r0/`；已完成的 arm 会跳过，避免重复基线。

## GPU 队列建议

1. 让当前 C42 select 链完成，不抢占、不重启。
2. 汇总 select 的四个复用基线与 `SolverC42_r1`；fit Pareto 身份随结果保留。
3. 若 select 有值得确认的 trade-off，先写 selection lock，再跑唯一的
   `internal_confirm`；没有价值也保留原始输出并结束该候选。
4. E4 只接已有端点/自然/VT-QA 资产，缺少匹配 baseline identity 时停在 blocked。
5. E5 继续 blocked，直到共同父状态与 optimizer lineage 可读取且任意静态 tensor 的
   匹配训练路径完成代码审计。
6. 不把 CPU 机制 Pareto arms 或 15 个 scratch runs 自动加入当前 GPU 队列。若后续
   作者单独授权机制方向实验，应先用实际 Geo tensor 重建 receipt，并把它定位为方向
   压力测试而非两机制唯一因果识别。

## 验证

对应测试：

- `tests/test_rope_z_ood_collision_mechanisms.py`
- `tests/test_rope_z_ood_collision_pipeline.py`

它们覆盖 phase 分段/峰值、完整 pair overlap、effective-rank identity、direct/vectorized
collision 一致性、全 δ Pareto 保留、E0 数据身份、0-step 与 Pareto solver 分类、
静态表 generation 命令和 E5 blocked 行为。CPU 单测不加载模型、不初始化 CUDA。
