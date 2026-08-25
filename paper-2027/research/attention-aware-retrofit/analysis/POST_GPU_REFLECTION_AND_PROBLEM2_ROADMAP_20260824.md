# 2026-08-24 GPU 实验复盘：理论表如何在大模型上兼得

- **状态：** 内部决策报告；不是新的实验结果或论文 claim
- **作用：** 解释 2026-08-24/25 GPU 窗口为何偏离主线，并恢复问题 2 的正确研究路径
- **本次修订：** 取代本文件在 commit `bef08c2` 中错误的“小模型 joint-search / collision-frontier-first”方案
- **数字边界：** 所有实验数字仍由链接的 result/evidence owner 持有

**2026-08-25 supersession.** The mature co-adaptive allocation oracle, matched
dense-natural controls, and full-200 2Wiki comparison are now owned by
[`../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`](../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md).
They empirically separate continued adaptation from the table's full/tail
redistribution. This memo remains historical interpretation; current action
priority is owned by repository [`../../../../INDEX.md`](../../../../INDEX.md)
§6.

必读 owner：

1. [`../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) — `x=a+Rz`、阶段和 zero-training 语义；
2. [`../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](../../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) — full sin/cos geometry、collision 反例、weights×table co-adaptation；
3. [`../../three_completions/optimization_notes.md`](../../three_completions/optimization_notes.md) — 位置核、统一泛函、EVQ-Cosh-R 与 exact-kernel/Nyström 优化；
4. [`../results/EXPERIMENT_REPORT_20260821.md`](../results/EXPERIMENT_REPORT_20260821.md) — phase-chord 两种子结果；
5. [`../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](../results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md)、[`../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`](../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md)、[`../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) — 本次 GPU 窗口。

## 0. 正确结论

我们要研究的不是：

- 1× 必须 bitwise/exact Native；
- 再从小模型搜索一个 `z`；
- 用 collision scalar 代替 LM 结果；
- 把 phase-chord 当经验 warm-start；
- 为 fixed table 发明新的“零运行时参数”术语。

真正问题是：

> 已有位置核理论给出一张面向 OOD/碰撞的第三轴表；直接研究它在成熟大模型上能否以可接受的窗口内代价换取稳定外推。该理论表本身就是 zero-training hard-swap 方法，不以 LoRA/continued adaptation 作为失败后的修补路线。

“兼得”不等于 1× 完全无损。YaRN、LongRoPE 和本文已有训练结果都允许一个可接受、必须报告的窗口内代价。成功标准是同一张表形成有竞争力的 in-window/OOD Pareto，并保持能力，不是满足人为 exact-retention 条件。

## 1. 已有理论解，不应再退回搜索

### 1.1 phase-chord 来自 RoPE 位置核

RoPE pair 在距离 `Δ` 上的相位差由

\[
1-\cos(\omega\Delta)
\]

给出；它是旋转矩阵 chord energy 的常数倍。Phase-chord 不是从 NLL 表格拟合出的方向，而是先让距离通过 RoPE phase response，再形成频率需求和 channel allocation。

已有两种子训练结果相对 FMRoPE 为

\[
+0.00070/-0.16051/-0.15577/-0.20522
\]

（`1x/2x/4x/8x` tail NLL）。它说明理论方向可以做到窗口内近似平价并改善所有测试外推长度。小模型在这里已经完成了 feasibility 角色，不再承担下一方法选择。

### 1.2 最近理论已进一步优化

`optimization_notes.md` 已经指出旧理论的真正缺口：collision 只选形状、resolution utility 只选尺度，两者没有统一。O4 将它们合并为

\[
\mathcal J[\rho]
=\frac\alpha2\int\rho^2
+\frac\beta2\iint\rho(\phi)\rho(\psi)\min(\phi,\psi)
-\lambda\int\rho q_\mu,
\qquad \int\rho=1.
\]

受迫 ODE 给出分辨率感知的闭式 **EVQ-Cosh-R**：

- `κ=0` 精确退化为原 EVQ-Cosh；
- 在理论分辨率阈值 `φ*` 处出现可解释跃变；
- 形状与尺度由一个泛函共同决定；
- 逆 CDF 保持闭式；
- 数值验证覆盖归一化、跳跃、退化极限和反演误差。

O5 进一步说明：若不用 Green-kernel 近似，exact positive kernel 对应第二类 Fredholm 方程；在真实 `K≤64` channel grid 上可用 Nyström/约束二次问题一次求解。这里的数值求解仍然是理论表构造，不是通过 LM validation 搜索一张表。

因此下一步不存在“再从 2–6 维 basis 搜索 `z`”的必要。需要做的是：选择最新、可复现的理论 construction，物化到目标模型的 `(K,L_native,Ω_native)`，冻结表身份，然后在大模型上检验。

### 1.3 理论解的正确 claim ceiling

理论给出 OOD/collision-oriented allocation，不等于已经证明 LM 最优。必须区分：

- 位置核/phase resolution：理论对象；
- fixed-support `z`：可识别设计轴；
- in-window/OOD LM 与能力：大模型实验端点；
- co-adaptation：冻结换表与共同训练之间的差异。

这不是让 collision 降格为无用 diagnostic，也不是让 collision scalar 单独预测 LM；而是先由理论给出表，再用 LM/capability 检验理论表是否兑现它的设计目标。

## 2. 为什么上一轮方案仍然错

| 错误 | 为什么错 | 正确替代 |
| --- | --- | --- |
| 将 1× retention 设为硬约束 | 把 engineering parity 当科学必要条件，排除正常 Pareto | 预注册“可接受损失”，同时报告 1× 和 OOD 全向量 |
| 回到 50M/151.9M 搜索 | 小模型已证明第三轴和 phase-chord feasibility，不能回答大模型 | 直接进入 1.485B zero-training hard swap |
| phase-chord 作为 warm-start | 忽略其 RoPE position-kernel 推导和后续 O4/O5 理论优化 | 物化理论表并冻结，不用 OOD labels 搜索 |
| worst-OOD constrained search 作为主方法 | 又把论文变成 generic hyperparameter optimization | 理论 construction 是方法，worst-OOD 是评测汇总 |
| collision-frontier-first | 新造一个未拥有的 objective，绕开已有理论解 | 使用现有 phase-kernel/EVQ-Cosh-R owner |
| “零运行时训练参数”术语 | 论文没有这个问题，反而模糊 zero learned parameters / zero training | 沿用锁定术语，不新增防御性标签 |

## 3. 2026-08-24/25 实验的真实位置

### 3.1 direct-`z` pilot

它只说明两个 design documents 无法稳定识别 62-effective-degree frozen table；没有使用 phase-kernel 理论表，也不是对理论方法的检验。不得继续 sweep，也不得用它否定第三轴。

### 3.2 两个 analytic static candidates

Anchored `tau=2` 与 protected-band candidates 在冻结 OLMo 上改善 2×、损伤 1×。它们不是最新 phase-chord/EVQ-Cosh-R construction；结果只否定两个候选。允许的窗口内代价本来也不应被误写成 exact zero。

### 3.3 Fresh FineWeb session-s4

它确认当前 engineering fallback 在新 shard rows 上持续有效，并证明几个具体 `z` 在 8K/16K 排序不同。它没有测试最新理论表，也不把 routing 升级为理论答案。

### 3.4 真正的失误

GPU 窗口本应优先检查“理论 phase-kernel table 在 1.485B 上的 zero-parameter Pareto”。实际却花在欠定 `z` 拟合、两个旧式候选和已有 fallback confirmation 上。实验可用，但没有击中最高价值问题。

## 4. 正确的大模型实验路线

### M0：无 GPU 的理论表物化

这一阶段不搜索 LM loss，只冻结理论对象：

1. 从 phase-kernel/最新优化 owner 选择唯一 construction；
2. 输入目标 checkpoint 的 `K`、Native spectrum/window 及理论要求的可测量量；
3. 生成 realised float32 inverse-frequency tensor；
4. 记录 `(a,R,z)`、排序、最小 spacing、position-kernel/collision receipts；
5. 同时生成 same-support geometric control；
6. 冻结 table/code/input hash，之后不得按评测结果改表。

如果最新理论选择 EVQ-Cosh-R，则 `tau/φ*/κ` 必须来自该理论 owner 的公式或测量协议；不能从大模型验证集 sweep。若 exact-kernel Nyström 版本是最终 construction，同样只解一次理论方程，不读 LM/OOD labels。

### M1：1.485B zero-training hard swap

模型：released OLMo-2-0425-1B-Instruct。第一门直接测试理论表，而不是训练小模型。

最低因果矩阵：

| Arm | Table | Weights/adaptation | 作用 |
| --- | --- | --- | --- |
| Native | Native | frozen | checkpoint baseline |
| same-support Geo | geometric control | frozen | support/control |
| theory table | phase-kernel / latest optimized table | frozen | zero-training third-axis intervention |
| current session-s4 | existing frozen policy | frozen | engineering reference；不拥有理论机制 |

Gain 必须固定或形成单独 table×gain 2×2，不能将 gain 收益写给 `z`。

端点：

- 1× natural tail NLL 与代表性 capability；
- 最少 2× natural NLL/OOD；
- 4× 作为 far-OOD/ranking-reversal 诊断；
- 严格生成或 source-use endpoint，避免 NLL=capability；
- 与 YaRN 等成熟 extension 的同协议参考，但论文故事不是“击败 YaRN”。

**1× 判定不是 exact parity。** 在运行前根据 Native 波动、现有 YaRN/extension trade-off 和论文可接受 claim 冻结损失上限。结果必须以完整 `1x/2x/4x` 向量判断 Pareto，不能因轻微 1× 代价直接杀掉远端有效表，也不能用远端收益掩盖 catastrophic retention。

### M2：失败审计，不是 adaptation fallback

理论 construction 的目标就是 zero-training hard swap。若 M1 没有形成预期 Pareto，按冻结身份依次审计：

1. theory owner 到 realised tensor 的物化是否正确；
2. `K/L_native/Ω_native` 与理论测量量是否绑定正确；
3. same-support、gain、route 和 evaluator 是否发生身份混淆；
4. 1× 代价是否仍在预注册可接受范围；
5. OOD endpoint 是否真正测到位置/source-use，而非 capability floor。

审计后仍失败，只能报告该理论 construction 在该 checkpoint/protocol 上失败，并回到理论 owner；不能接 LoRA/continued training 把 zero-training failure 改写成 adaptation success。

新 1B FineWeb-Edu corpus 是独立数据资产，不自动进入本方法训练。它目前仍在已关机实例系统盘，tracked receipt 不能替代 raw corpus。

### M3：第二模型/8B

只有 M1 在 1.485B 上形成清楚 zero-training Pareto 后，才进入第二 checkpoint 或 8B。优先验证不同 Native window/base 下同一理论 construction，而不是增加 OLMo 表格数量。

## 5. 第三轴在正确路线中的作用

第三轴不是搜索空间口号，而是理论 construction 的因果坐标：

1. 理论 position kernel 输出 channel density/quantiles，最终落到 `z`；
2. same-support Geo versus theory table 固定 `(a,R)`，只改变 `z`；
3. M1 直接测试该 `z` 在冻结大模型上的 zero-training 价值；
4. same-support/gain controls 将该价值与其他变量分离；
5. 跨模型重复同一 construction 才回答 scale/profile generalization。

这正对应论文主线：拥有的是 finite spectral allocation 轴和从 RoPE 位置核推导的构造，不是一组可随 benchmark 调节的指数。

## 6. 立即停止与下一动作

停止：

- 1× exact-retention hard constraint；
- 50M/151.9M 新搜索或 62D/低维 `z` optimization；
- 把 phase-chord 当 warm-start；
- 新 collision frontier 取代已有位置核理论；
- 将 LoRA/continued adaptation 作为 theory hard swap 的 fallback；
- `tau`、protected band、gain、routing 或 fresh shard002 重复 sweep；
- “零运行时训练参数”等新术语；
- 未通过 1.485B 就直接做 8B。

唯一下一动作：

> 完成 M0：从最新理论 owner 物化唯一大模型 theory table 与 same-support Geo control，冻结 identities，并写 M1 zero-training hard-swap preflight。没有新表搜索、LoRA fallback，也不启动 GPU。

## 7. 一句话回答“如何兼得”

> 不是在小模型上搜索一个满足 1× 硬约束的 `z`，而是把 RoPE 位置核与 collision/resolution 统一泛函给出的理论表直接装到成熟大模型上，允许并量化合理窗口内代价，检验其 zero-training OOD Pareto。若结果不符，审计或修正理论 construction；不以适配掩盖 hard-swap failure。第三轴拥有理论表与几何对照的因果差异。
