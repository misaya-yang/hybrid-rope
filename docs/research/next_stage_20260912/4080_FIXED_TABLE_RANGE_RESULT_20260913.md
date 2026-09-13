# 4080S：单固定表区间诊断与下一轮求解目标

更新：2026-09-13。本页是 RTX 4080 SUPER 新实例上首轮零训练实验与 CPU
几何判决的结果 owner。远端原始生成保留在数据盘
`fixed_table_interval_20260913/`；该目录是 ignored 运行资产，不属于 Git 交付。

## 结论

1. **问题已锁定为单固定表。** 对每个 arm 只加载一张 Native-relative RoPE 表，
   同一表用于全部层和整个8K--64K会话；运行长度改变时不切表。设计倍率
   (S=8) 与实际运行倍率 (r\in\{1,2,4,6,8\}) 分开记录。
2. **当前小面板的 minimax 候选是 BM_g8 的区间均值 gain。** 它保持 BM 频率
   tensor 不变，只将固定 gain 从 (1+0.1\ln8) 改为
   (1+0.05\ln8)。相对逐点较强的 BM/MrPro 静态基线，最大 regret 从 BM 的
   10.42pp 降为8.33pp；已测最弱点由72.92%升至75.00%。代价是四个倍增点
   log-length AUC 比 BM 低0.35pp，加入48K后的五点梯形AUC低0.82pp。
3. **这不是最终方法胜利。** 面板只有三类RULER检索、每格4行；RangeGain的
   32K/48K均低于BM。它只支持“把gain纳入区间配置有价值”，尚未覆盖VT、QA、
   自然任务或独立确认。
4. **对称gamma=3与BM/MrPro等距插值均失败。** 两者在48K仍很高，却在64K
   分别跌至60.42%和56.25%；区间行为不随表空间或长度平滑变化，不能继续用
   gamma或凸插值扫表。
5. **严格的 OOD-max + SEP-min 几何求表路线被现有12-profile面板否定。** 其
   严格不等式仍成立，但两个量不能排序冻结模型任务表现；因此不从该Pareto面
   继续生成GPU候选，也不事后补第三个几何指标。
6. **模型条件化求解得到一个值得保留、但尚未胜出的C42邻域候选。** 它在fit
   teacher-forced目标上同时改善区间regret、端点、来源margin和Native项；独立
   select/confirm与350行宽任务生成显示收益分布不一致。因此将其分类为Pareto，
   不因任一开发阈值或单任务损失淘汰，也不称为当前最优表。

## 磁盘整理与保留资产

新实例入口为 `westc:53405`。启动时系统盘约19/30GB、数据盘65/68GB；GPU空闲。
只删除两项明确与本轮无关、且不承担当前证据身份的重复资产：

- 系统盘旧 MiniCPM-4.1 模型及其独立运行时，约17GB；
- 数据盘 `Meta-Llama-3-8B` base副本，约15GB；保留独立的
  `Meta-Llama-3-8B-Instruct`。

清理后系统盘2.5/30GB、数据盘50/68GB，分别约28GB和19GB可用。保留：
Llama-3-8B-Instruct、OLMo-2-1B、Qwen模型、S1 checkpoint、RULER源数据、既有
表与历史结果。没有删除当前checkpoint、训练数据或已登记证据。

## Llama固定表协议

- checkpoint：Meta-Llama-3-8B-Instruct，Native (L=8192)，RoPE base 500000；
- horizon：(H=65536=8L)，所有扩展臂固定 (S=8)；
- 长度：8/16/32/64K自然倍增网格，另加48K内部压力点；
- 任务：`niah_single_1`、`niah_multikey_1`、`niah_multivalue`，每格4行；
- 主分数：NVIDIA RULER contains式 official task score，三任务等权；
- 推理：BF16、greedy；超过8K用8K分块prefill并保留完整KV cache；
- 资源：64K时板载显存最高约31.4GB，GPU利用率100%、约310W，无OOM；
- Native只在8K测一次；BM_g8、MrPro_g8在这批benchmark各测一次并永久复用。

| 固定配置 | 8K | 16K | 32K | 48K | 64K | 4点AUC | 5点AUC | 最弱点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native（只测8K） | 97.92 | -- | -- | -- | -- | -- | -- | -- |
| BM_g8 | 95.83 | 85.42 | **97.92** | 95.83 | 72.92 | **89.24** | 91.33 | 72.92 |
| MrPro_g8 | 91.67 | **95.83** | 87.50 | **100.00** | 68.75 | 87.85 | **91.76** | 68.75 |
| BetaSym-gamma3_g8 | 95.83 | 87.50 | 95.83 | 95.83 | 60.42 | 87.15 | 90.61 | 60.42 |
| BM/MrPro exponent midpoint | 95.83 | 89.58 | 87.50 | 91.67 | 56.25 | 84.38 | 88.12 | 56.25 |
| **BM_g8 + range-mean gain** | 95.83 | 89.58 | 91.67 | 91.67 | **75.00** | 88.89 | 90.51 | **75.00** |

数字均为百分比。5点AUC把48K作为额外插值结点，不能和只在倍增网格计算的4点
AUC混称同一统计量。逐点强基线包络只用于 regret 诊断，不是一张可部署的动态表。
endpoint floor应来自同一张固定强基线；本面板64K最强旧表是BM_g8的72.92%。

## CPU OOD / collision 判决

输入为 `paper-2027/figs/profile_diagnostic_inputs.json` 的12张OLMo冻结表、共同
checkpoint/gain/350行16K得分。按完整sine--cosine pair计算：

- 训练离散码：Native距离0--4096；部署距离0--16384；
- OOD：每个部署码到离散Native训练码的精确最近邻，float64分块矩阵计算；
- SEP：对所有整数 separation (h=1,\ldots,16384) 精确枚举；
- 每张表均验证了 `SEP <= 2 * OOD_max`。

结果：

| 关系 | Spearman rho |
|---|---:|
| task score vs `-OOD_max` | 0.378 |
| task score vs `-OOD_p95` | 0.853 |
| task score vs `SEP_min` | -0.713 |

所有12张表的 `SEP_min` 都发生在 `h=1`，数值只在0.20499--0.21018间变化；它在
该面板上退化为局部一步分辨率，不是远距near-collision读数。按原定义的
`OOD_max + SEP_min` 有22个几何支配但任务分数反转；最强反例是MrPro同时有比
MrProBM更低的OOD-max和更高的SEP，却在同一任务面板低34.59pp。即使事后换成
`OOD_p95 + SEP` 仍有9个反转。p95的相关性是可记录现象，不足以恢复原Pareto
求表主张。

CPU实现为 `scripts/analysis/audit_rope_ood_collision_pareto.py`；单元测试验证
完整码距离、离散Native零OOD和Pareto反转记账。

## 模型条件化区间分配：已完成首轮

没有在48行Llama开发面板上拟合18维表，也没有做64K反向。首轮求解对象固定为：

- 模型：冻结 OLMo-2-0425-1B-Instruct，无LoRA；
- (S=4)，transition band固定 `[14,32]`；全层/全长度一张表；
- 18个正增量以softmax参数化并强制logits零均值，17个有效形状自由度；另有一个
  有界 `log_gain`，共18个有效变量；
- fit使用4/6/8/10/12/14/16K的single、multikey与单答案QA，完整canonical
  answer+EOS NLL，任务与长度等权；source-counterfactual margin作为独立组；
- Native输出KL只作局部trust region，同时保留4K自然NLL和最终短窗自由生成；
- 优化用一阶顺序线性trust-region：每轮真实backward，限制
  `max|delta m| <= 0.03`、`|delta log gain| <= 0.02`，对1/0.5/0.25/0.125步长
  真实回验；本轮分别从BM和C42V24初始化，不跑random simplex；
- select/confirm使用未参与梯度的真实greedy生成。fit、select、内部confirm按
  每格8/4/4拆分；最终再生成新seed独立确认。

附件建议的完整Hessian/Fisher QCQP未采用：真实答案CE Hessian不保证半正定，
仓库旧 `joint_kkt` 使用的是各向同性拟曲率；旧direct-z也已经证明小数据Adam加
平均NLL会在held-out行上失败。优化小批只估计梯度，接受或拒绝使用全部168条fit
任务行与128对来源反事实的真实前向值；4份Native KL材料逐步回验。

### 求解回执

BM初始化在首轮没有找到同时改善全fit目标且不扩大其余冲突的局部步。初始最坏/
平均range regret为0.231864/0.054264，Native最坏任务差0.193143，端点最坏差
0.090462，source-CF最坏hinge 1.093844。它只否定当前trust region内这一次局部
提议，不否定BM，也不否定更大表空间。

C42V24初始化接受一个alpha=0.5的可行性改善步，随后一轮未找到进一步步。固定gain
从1.138629降至1.127300；全fit真实值变化如下：

| 指标 | C42V24初始化 | SolverC42候选 | 方向 |
|---|---:|---:|---|
| 最坏range regret | 0.227306 | **0.210802** | 改善 |
| 平均range regret | 0.068548 | **0.028782** | 改善 |
| 最坏endpoint差 | 0.153359 | **0.066740** | 改善 |
| 最坏source-CF hinge | 0.879791 | **0.721508** | 改善 |
| 最坏Native任务差 | 0.215313 | **0.210802** | 小幅改善 |

四份Native KL为0.11818/0.25642/0.10979/0.08026，均低于本轮复用的BM限制。
候选仍未满足所有预设绝对阈值；这些阈值只描述约束冲突，不作为删除候选的门禁。

### 独立生成与宽任务

select和内部confirm每个长度/任务格仅4行，必须与宽面板一起解释：

| 固定表 | select AUC / 最弱点 / 16K | confirm AUC / 最弱点 / 16K |
|---|---:|---:|
| BM | 65.25 / 41.67 / 41.67 | 82.02 / 58.33 / 58.33 |
| C42V24 | **68.14** / 50.00 / 50.00 | 87.16 / **66.67** / **66.67** |
| SolverC42 | 62.95 / **58.33** / **58.33** | **87.98** / 58.33 / 58.33 |

select偏向SolverC42的最弱点/端点，confirm偏向其AUC、但偏向C42V24的端点/最弱点；
没有单一胜者。350行、七任务、全16K旧开发面板复用同row的C42/C42V24历史输出，
只新生成SolverC42：

| 固定表 | 七任务task-equal official |
|---|---:|
| C42 | 43.67 |
| C42V24 | **54.40** |
| SolverC42 | 50.39 |

SolverC42相对C42为+6.71pp，相对C42V24为-4.01pp；逐任务相对C42V24在
single1、multikey1持平，在single3、multivalue、multikey3和QA2下降，CWE近似
持平。相对C42则除multikey3外多数任务改善。该结果支持“候选有跨面板价值但仍在
Pareto面上”，不支持淘汰或宣布胜出。

391行、五任务、全16K自然QA并集同样只生成SolverC42，并复用BM/MrPro历史原始
行；主口径为完整响应F1、任务等权：

| 固定表 | 五任务task-equal whole-response F1 |
|---|---:|
| MrPro | 25.92 |
| BM | 29.41 |
| SolverC42 | **29.57** |

候选相对BM为+0.17pp，逐行68胜/68负/255平；相对MrPro为+3.65pp。相对BM，
HotpotQA为+5.16pp，NarrativeQA近似持平，2Wiki/MultiField/Qasper分别为
-1.28/-0.93/-2.33pp。该面板有已知高floor且不是独立确认，故只说明候选没有在
自然QA上整体崩溃、其收益继续表现为任务间重分配；不能据此宣称全面超过BM。

为防止把点估计误作硬门禁，另做20,000次任务内配对row bootstrap、再任务等权。
宽面板相对C42的95%敏感性区间为[+3.23,+10.36]pp，相对C42V24为
[-6.81,-1.34]pp；自然QA相对BM为[-1.98,+2.39]pp，相对MrPro为
[+0.36,+7.04]pp。这只是当前冻结行的重采样敏感性区间，不是总体置信保证；尤其
相对BM应判为统计未决，而不是把+0.17pp解释为胜利或把区间含零解释为失败。

判定采用三层：协议/实现错误否决运行；任务层明确受支配才淘汰表；小样本不达线、
代理冲突和单任务损失只标为Pareto/未决并继续独立复核。MrPro/BM/Native在每个
benchmark只生成一次并永久复用。

## 实现与原始结果

- 固定区间数据：`experiments/olmo_recovery_20260912/prepare_fixed_table_interval.py`；
- 固定表评价与汇总：`recovery_v2_eval.py`、`score_fixed_table_interval.py`；
- 表构造与range gain：`recovery_v2_runtime.py`；
- 模型条件化求解：`native_relative_allocation.py`、`solve_range_table.py`；
- 独立生成汇总：`summarize_range_generation.py`、
  `compare_broad_fixed_candidate.py`、`compare_natural_fixed_candidate.py`；
- 配对敏感性分析：`bootstrap_task_equal_contrast.py`；
- 本轮单测：`test_fixed_table_interval.py`、`tests/test_rope_ood_collision_pareto.py`、
  `tests/test_range_table_solver.py`、`tests/test_range_table_eval.py`、
  `tests/test_broad_fixed_candidate.py`、`tests/test_natural_fixed_candidate.py`；
- 远端摘要：`fixed_table_interval_20260913/summary.json`；
- CPU摘要：`fixed_table_interval_20260913/cpu_ood_sep_audit.json`；
- 求解器与生成：`model_conditioned_range_20260913/`。

本轮没有训练模型权重，没有下载新模型，没有提交或推送Git，也没有关机。实例
继续保留，下一步用同一宽面板拆分SolverC42的allocation与gain贡献。
