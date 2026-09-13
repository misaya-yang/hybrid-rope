# 单固定表：跨模型判决与无固定边界求解流水线

更新：2026-09-13。本页承接
[4080S固定表结果](4080_FIXED_TABLE_RANGE_RESULT_20260913.md)，是当前实验目标、
顺序与判决 owner。Web计划、子代理建议与历史队列只提供输入；本页按实际代码、
服务器资产和已完成结果二次审查后排序。

## 当前目标

给定预训练RoPE模型、Native长度L和部署horizon H=sL，只安装一张固定表：

\[
\omega'_k=\omega^N_k S^{-m_k}.
\]

寻找在Native保留、内部最坏regret、目标端点和log-length AUC之间最优的可部署
配置。最终搜索空间不预设YaRN的transition band：

\[
0\le m_1\le m_2\le\cdots\le m_K\le1.
\]

`l/h`、零位移平台和完整位移平台应由解事后读出；固定YaRN/MrPro边界的求解只
能作为低成本近似或机制控制，不能承担最终答案。gain独立联合优化并单独归因。

## 当前证据状态

| 证据 | 当前结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| OLMo C42邻域求解 | fit的range/endpoint/Native/source指标共同改善；select/confirm为Pareto | 模型条件化方向可找到有用表 | 代理改善不保证宽任务或source pair-follow |
| OLMo 350行shape×gain | C42 54.40；只换gain 53.96；只换shape 51.86；全换50.39 | 宽任务损失主要来自allocation，且约有负交互 | 不证明allocation普遍有害；它改善了其他长度/任务 |
| OLMo 391行自然QA | Solver 29.57、BM 29.41、MrPro 25.92 | 候选没有自然QA整体崩溃，并优于MrPro | 相对BM仍统计未决；面板有高floor且非独立确认 |
| Llama g8 full-profile迁移 | 8/16/32/48/64K为95.83/97.92/100/95.83/85.42；AUC96.91 | 一张固定迁移表在当前小面板显著强于BM/MrPro AUC | 只有3任务×4行；同时改变band placement和内部shape |
| Llama YaRN-band remap | 8/16/32/48/64K为97.92/97.92/97.92/87.50/45.83 | 机械平移band保短中程但损伤端点 | 它不是“正确边界”；作为机制反例和8–32K Pareto点保留 |
| Llama最小band屏幕 | S=4的`[16,34]`在8/16/32K同时支配BM/MrPro和相邻C42表 | Llama当前小面板存在尖锐、非单调band最佳区 | S=2最佳不同；不能称模型级SOTA |
| Llama赢家直迁OLMo | `[16,34]`在OLMo 4/8/16K NIAH为100/62.5/50，BM为100/75/75 | 相同槽数/base仍不足以保证band迁移 | 不否定Llama候选，也不否定模型条件化band |
| OLMo局部band屏幕 | 当前最佳basin为low=14、high=31--32 | low/high需独立作为变量；high=31/32当前未决 | 小面板不能给出普适边界 |

full-profile使用OLMo的完整64槽布局，transition约为`[14,32]`。band-remap将同一
内部累计shape插值到Llama的YaRN-native控制区间`[18,35]`。前者是合法候选；后者
只是用来区分绝对槽位、band placement与内部shape的控制。

## 判决流水线

作者于2026-09-13提出的无人工band闭式候选及所有权、定理边界和单臂判决见
[Winding-Matched RoPE作者方案](WINDING_MATCHED_ROPE_AUTHOR_PROPOSAL_20260913.md)。
其Llama g8实际结果为8K 89.58%、16--64K均0，已否定“单一端点phase匹配足以
保证任务端点或全区间”的实用假说；闭式代数本身仍成立。当前最小band结果见
[Band位置最小筛选](BAND_POSITION_MINIMAL_SCREEN_20260913.md)。

进度百分比表示距离“当前声明范围内的可部署解法”还剩多少证据环，不是成功概率。

| 顺序 | 实验与目的 | 最小执行、复用与成本 | 成功有什么用 | 失败有什么用 | 里程碑 |
|---|---|---|---|---|---|
| P0 | 完成Llama `YaRN-band remap`，区分full-profile与按Native平移band | 同一60行、标准g8 gain；BM/MrPro永久复用；约11–12分钟 | 若也强，内部shape在相对band坐标可迁移 | 若弱，支持更早band placement或绝对槽位是full-profile优势来源 | 候选+初步机制，约35% |
| P1 | 64K八任务端点复核，检查简单NIAH之外的VT/FWE/QA/多证据能力 | 先用新runner对BM_g8做8行canary；一致后复用旧runner的32行BM/MR，只跑1–2个候选，各约12–15分钟 | 强候选跨任务，进入全区间 | 定位任务特异损伤；保留专用或Pareto价值，不全删 | 任务广度，约50% |
| P2 | 宽长度八任务开发面板，直接找内部深坑 | 8/16/24/32/40/48/64K；复用已有192行输入并补64K。每个新benchmark的BM/MrPro只跑一次；候选约35–40分钟/臂 | 得到真实worst-length、AUC和任务交叉 | 若交叉，证明单小面板winner不可部署，并给robust solver活跃约束 | 区间广度，约70% |
| P3 | Llama自然任务复核，检查合成检索收益能否迁移 | 优先找回13格与64K自然原始输入/基线；身份一致则只跑候选，否则把缺口显式登记 | 支持实际部署有用性 | 负结果把方法定位为任务条件化，不否定检索区间收益 | 下游用途，约75% |
| P4 | 全谱任务×长度梯度冲突矩阵，解释“为什么总是部分改善” | 在OLMo冻结模型上对全64槽shape与gain分别记录3任务×7长度、宽任务代表行和source-CF梯度；约10–20分钟GPU | 稳定负夹角证明局部Pareto冲突，并给minimax活跃组 | 若梯度同向而生成冲突，证明主要是NLL/解码代理错配 | 机制解释，约60% |
| P5 | 无固定`l/h`的full-spectrum robust solver | 63个非负增量累计为单调m；Native/endpoint/worst-regret字典序，AUC次级；从强表做小trust-region，不用单平均损失 | 得到由checkpoint和任务分布决定的表及可解释边界 | 若无共同方向，输出可复现Pareto前沿而非继续猜曲线 | 方法算法，约80% |
| P6 | allocation dose与边界方向确认 | 依据P4/P5方向选少量预注册剂量；复用端点，不扫密网格 | 中间剂量最好则校正过冲；反向更好则修正代理 | 非单调/任务峰值不同证明需多目标组合，不是一个标量步长 | 局部可靠性，约85% |
| P7 | Llama直接模型条件化小步 | 在可承受的8/16K上求全谱shape+gain小步，32/48/64K只做固定表生成回验 | 回答“给定任意checkpoint怎样求表”的跨模型算法问题 | 若梯度不迁移到长端，成果收敛为可迁移经验profile而非通用solver | 方法泛化，约90% |
| Final | 新seed全区间、自然任务和短窗/NLL最终确认 | 只保留1–2个非支配候选；新seed 8任务×全长度，BM/MrPro各一次 | 锁定声明范围内的Llama可部署固定表 | 收窄到明确模型/任务范围，保留反例和成本 | 90–100% |

P1先于大范围新面板，因为它用最少GPU暴露当前强结果是否只属于三类检索。P4先于
P6，因为没有冲突矩阵的alpha扫描仍是盲调参。P5不再固定`[14,32]`；现有固定band
求解器保留为历史首轮和对照。

## 判定合同

- 只有实现错误、表身份错误、动态换表、runner不等价等协议错误可以作废一次运行。
- 小样本未达阈值、bootstrap跨零、单个任务或长度下降，只标记`Pareto/unresolved`。
- 代理NLL、KL、source hinge和几何量只负责提出或解释候选，不替代真实生成。
- compact面板可揭示任务特异性，不能认证winner。
- full-profile与band-remap各有胜负时两者都进入下一层，最多保留两个以控制成本。
- 只有在足量、row-matched的任务×长度面板上被另一张固定表明确支配，且没有独特
  任务、长度、Native或自然任务收益时，才结束候选；结论只限该模型和benchmark。
- Llama成功不能抹去OLMo反例；Llama失败也不能反推OLMo候选无效。

## 执行与资产

- 服务器：`westc:53405`；32GB RTX 4080 SUPER。
- 当前模型：`/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct`。
- Llama g8结果根：`/root/autodl-tmp/fixed_table_interval_20260913/`。
- OLMo求解根：`/root/autodl-tmp/model_conditioned_range_20260913/`。
- 任意固定表入口：`recovery_v2_eval.py --static-table-json ... --table-label ...`；
  64K使用8K分块prefill，候选串行，当前峰值约31.4GiB。
- 旧64K八任务BM/MR来自另一runner。先做每任务1行BM canary，并从原始输出用同一
  scorer重算；一致后永久复用，若不一致则给旧runner增加静态JSON入口，只跑候选。
- 目前数据盘约19GiB可用。表、contract、raw generations、summary、基线和Pareto
  候选不删；报告回写后才清理失败`.incomplete`、临时缓存或未采用导数缓存。

## 当前下一跳

1. 在Qwen2.5-1.5B、S=2、32/64K上比较`[14,32]`、`[16,34]`、
   等winding `[22,39]` 与本地几何 `[23,40]`；先每格4行与两篇PPL；
2. 64K生成主判，32K作Native保护；只给最佳两臂补剩余行，BM/MrPro永久复用；
3. Qwen若支持等winding，只把它作为跨模型初始化规则，再由checkpoint小面板校正；
   若失败，则正式放弃纯配置几何统一，转向模型条件化band/full-spectrum求解；
4. 下一代求解器使用全64槽单调allocation，不固定YaRN的`l/h`，并把placement
   与总压缩剂量分开控制。
