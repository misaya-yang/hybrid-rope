# Band 跨模型、跨倍率机制研究：早期证据与下一阶段

更新：2026-09-13。本页执行
`Codex_Band_下一阶段执行计划_20260913.md` 的早期开放格，并维护后续统一机制实验。
所有结果都是零训练、全层共享、全会话一张固定表。小开发面板与扩充确认严格分开；
本页不把点估计称为SOTA，也不把三个模型各自的小屏赢家误称为最终band。

## 当前阶段裁决：现象已出现，统一规律尚未建立

Native winding

\[
t_k=\frac{L_{native}\omega_k}{2\pi}
\]

对跨模型候选生成有实用预测力：由Llama映射到Qwen的`[22,39]`取得当前最高平均
分。但这不是唯一解释。Llama同一C42 shape在S=4的小屏偏好`[16,34]`，到了S=8
则`[14,32]`的区间AUC更高；说明band至少与目标倍率及中间长度发生交互。当前证据
不支持一个跨模型、跨倍率都固定的槽位band，也还不足以把问题收窄为逐模型调参。
本阶段真正检验的工作假说是：

> **存在一个相对于模型Native频率结构与目标倍率的band坐标，能够预测不同模型、
> 不同S下的有用basin；checkpoint学得的Q/K频率使用只提供可测的条件化修正。**

候选统一坐标不预先限定为YaRN低/中/高边界。现阶段并列检验：固定槽位、Native
winding、目标窗口winding、Native窗相位扰动

\[
D_N(k)=L\omega_k\left(1-S^{-m_k}\right)
\]

和目标端相位扰动`D_H=S D_N`。Llama S2到S4的`+2`槽位移动与`D_N`的预测接近，
但这只有一次同模型跨倍率线索；必须由S8和跨模型leave-one-model-out预测证伪。

PPL只作健康检查或近似平局诊断，不选择band。WindingMatched、连续solver和新shape
不再作为无目的大搜索；但为区分band、shape、总压缩剂量和gain，预注册的最小
factorial控制必须保留。任何统一规律都必须经过三个模型当前候选的mini、同模型
跨S和leave-one-model-out后才能进入论文结论。

## B1：Qwen S=2 的 winding 与本地几何

checkpoint为Qwen2.5-1.5B-Instruct，Native 32K、base 1M、64槽。选择屏幕使用
`niah_single_2`、`niah_multikey_2`、`niah_multiquery`；32K每任务2行，64K每任务
4行。两篇固定PG19只报告tail-512 NLL。

| 固定表 | 32K official | 64K official | 32K NLL | 64K NLL |
|---|---:|---:|---:|---:|
| Native | 79.17 | 52.08 | **3.1995** | 2.8275 |
| BM | 79.17 | 83.33 | 3.2613 | 2.8340 |
| MrPro `[23,40]` | 95.83 | 79.17 | 3.2546 | 2.8370 |
| C42 `[14,32]` | 95.83 | 81.25 | 3.2473 | 2.8296 |
| C42 `[16,34]` | **100.00** | 87.50 | **3.2449** | 2.8329 |
| C42 `[22,39]` | **100.00** | **91.67** | 3.2535 | **2.8247** |
| C42 `[23,40]` | 95.83 | 83.33 | 3.2510 | 2.8267 |

这里的粗体只标同类扩展表中的点估计，不是显著性声明。`[22,39]`来自Llama
`[16,34]`的等Native-winding映射；`[23,40]`是Qwen本地约32圈/1圈几何边界。

为避免4行噪声，只对预定两候选和现有基线把64K补到每任务16行；没有增加band，
也没有再次运行PPL：

| 固定表 | single2 | multikey2 | multiquery | 64K task-equal macro |
|---|---:|---:|---:|---:|
| Native | 62.50 | 25.00 | 53.12 | 46.88 |
| BM | 100.00 | 43.75 | **98.44** | 80.73 |
| MrPro | 100.00 | 56.25 | 92.19 | 82.81 |
| `[16,34]` | 100.00 | 56.25 | 93.75 | 83.33 |
| **`[22,39]`** | 100.00 | **68.75** | 93.75 | **87.50** |
| `[23,40]` | 100.00 | 50.00 | 95.31 | 81.77 |

平均上`[22,39]`比`[23,40]`高5.73pp、比MrPro高4.69pp、比BM高6.77pp。
20,000次任务内配对row bootstrap的95%重采样区间分别为：

- 相对`[23,40]`：`[-0.52,+13.02]pp`；
- 相对MrPro：`[-0.52,+11.46]pp`；
- 相对BM：`[0.00,+14.58]pp`。

因此B1只支持“winding候选进入最佳basin，点估计偏`[22,39]`”；尚不能区分为
全局唯一槽位。冻结Qwen basin为`[22--23,39--40]`，不再增加`[23,39]`或
`[22,40]`等开发候选。

## B2：Llama S=4 band 能否跨到 S=8

这一问题分两步。第一步使用Solver shape的三张表，暴露了强烈位置响应，但混入了
shape；第二步固定为Llama S=4筛选时相同的C42 shape、S=8振幅、标准gain、FP32
构表路径、BF16 checkpoint、60行输入、greedy decoder与official scorer，才是
匹配shape的band对照。

| 固定表 | 8K | 16K | 32K | 48K | 64K | 5点log-AUC | 最弱点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM | 95.83 | 85.42 | 97.92 | 95.83 | 72.92 | 91.33 | 72.92 |
| MrPro | 91.67 | 95.83 | 87.50 | **100.00** | 68.75 | 91.76 | 68.75 |
| Solver shape `[14,32]` | 95.83 | **97.92** | **100.00** | 95.83 | **85.42** | **96.91** | **85.42** |
| Solver shape `[16,34]` | **97.92** | **97.92** | 97.92 | 95.83 | 60.42 | 94.98 | 60.42 |
| Solver shape `[18,35]` | **97.92** | **97.92** | 97.92 | 87.50 | 45.83 | 92.58 | 45.83 |
| C42 shape `[14,32]` | 95.83 | **97.92** | **100.00** | 95.83 | 81.25 | 96.62 | 81.25 |
| C42 shape `[16,34]` | **97.92** | **97.92** | 87.50 | 95.83 | 81.25 | 93.67 | 81.25 |

Solver shape下，`[16,34]`相对`[18,35]`在48K/64K为+8.33/+14.58pp，说明向更快
频段恢复有用；但相对`[14,32]`在64K为-25pp。匹配C42 shape后，`[16,34]`与
`[14,32]`的64K同为81.25%，主要差异转到32K：100%降到87.5%，5点AUC低2.96pp。
因此此前的25pp端点差不能归因于band本身；匹配对照仍提示band×S×内部长度交互，
但效应位置与大小强烈依赖shape。

小屏不支持把S=4选点直接冻结到S=8，但也不能据此淘汰`[16,34]`：每格只有4行、
三个检索任务。两张C42表都进入跨S机制账本；完整mini后再估计平均、最坏长度和
任务族交互。

## B3：OLMo S=4 首个Core-6 mini

冻结面板为6任务×4/8/16K×18行，共324行。所有旧、新输出按prompt hash合并并用
同一当前official scorer从raw text重算；名字相似但hash不匹配的历史输出不复用。
MrPro与BM的实际缺口均补齐后，三臂覆盖都是324/324。

| 固定表 | 4K | 8K | 16K | 3点log-AUC | 最弱点 |
|---|---:|---:|---:|---:|---:|
| BM | **81.67** | 71.31 | 48.50 | 68.20 | 48.50 |
| MrPro | 42.28 | 25.63 | 8.18 | 25.43 | 8.18 |
| C42 `[14,31]` | 79.65 | **71.84** | **53.15** | **69.12** | **53.15** |

C42相对BM的AUC点估计为+0.92pp，16K与最弱点均为+4.65pp，4K为-2.02pp。
20,000次任务内配对bootstrap的AUC差区间为`[-2.48,+4.36]pp`，
`P(delta>0)=0.698`。任务AUC差异不是同向：multikey、multiquery、FWE和single
改善，VT为-8.61pp，QA为-1.39pp。因此这是一张区间/任务Pareto表，不是已认证
赢家；它同时证明4行NIAH开发屏会遗漏关键任务交互，不能用来关闭shape或band机制
哨兵。

原始冻结面板与三臂汇总位于服务器
`band_mini_20260913/olmo_s4/frozen_324/`和
`band_mini_20260913/olmo_s4/summary.json`。

## 理论边界

单层局部展开

\[
a(d)=\sum_k\operatorname{Re}(C_ke^{id\nu_k}),\qquad
\frac{\partial a}{\partial m_k}
=\log S\,d\nu_k\operatorname{Im}(C_ke^{id\nu_k})
\]

解释了非单调、任务交叉与checkpoint依赖：导数符号随距离、层、头、内容和权重
变化。Native winding统一了几何尺度，却没有包含`C_k`，因此只能给初始化，不能
给唯一最优边界。band整体移动还会改变`sum(m)`，当前选择的是完整部署profile，
不是已经分离的纯placement因果量。

## 从mini确认到统一规律的连通实验

当前小屏存在强选择偏差，第一步必须把三个模型的各自最佳表至少推进到mini；这只
完成整个机制研究的第一个确认层，不是收束：

- OLMo S4使用C42 `[14,31]`，Llama S8使用Solver `[14,32]`，Qwen S2使用
  C42 `[22,39]`，按[RULER四级评价合同](RULER_TIERED_PANEL_CONTRACT_20260913.md)
  各完成324行mini；当前候选已经过更小开发屏，因此不以108行low代替mini；
- BM/MrPro/Native按benchmark登记键永久复用，只补不存在的task×length×row。

第二步建立连通的模型×S矩阵，而不是三个孤立赢家：

| 模型 | S=2 | S=4 | S=8 |
|---|---|---|---|
| OLMo，Native 4K | Core机制矩阵 | 完整band×shape矩阵 | 延后 |
| Llama，Native 8K | 复用既有 | 完整band×shape矩阵 | Core机制矩阵 |
| Qwen1.5B，Native 32K | Core机制矩阵 | 完整band×shape矩阵 | 物理长度延后 |

每个Core矩阵至少保留MrPro、BM、C42和Solver两个机制哨兵；S4用相同候选band的
BM/C42/Solver factorial分离band主效应、shape主效应及交互。Llama S8还需以
`[14,32]`为锚检验固定槽位、`D_N`和`D_H`预测；预测值必须先冻结，邻点只量化
prediction regret，不能反向改规则。随后做三折leave-one-model-out：两模型确定
坐标规则，直接预测第三模型，不在留出模型上重新拟合。

只有以下证据同时成立，论文才可写“跨模型、跨倍率的预测性band坐标”：同模型跨S
命中非支配basin；三折留一预测稳定；matched-shape/gain/dose控制不反转；mini或更高
面板没有Native、端点或内部长度灾难。否则分别收窄为checkpoint条件化尺度律、完整
profile联合优化，或用反例否定纯几何band规则。

## 远端结果

- Qwen选择与PPL：`qwen15_minimal_band_s2_20260913/summary.json`；
- Qwen 64K扩充：`qwen15_minimal_band_s2_20260913/confirm_remaining/`；
- Llama新表：`fixed_table_interval_20260913/SolverProfileBand16_34_g8_table.json`；
- Llama生成：`fixed_table_interval_20260913/SolverProfileBand16_34_g8/`；
- Llama比较：`fixed_table_interval_20260913/SolverProfileBand16_34_g8_comparison.json`。
- 匹配C42 shape的Llama S8表：
  `fixed_table_interval_20260913/C42Band16_34_g8_table.json`；
- 匹配C42 shape生成：`fixed_table_interval_20260913/C42Band16_34_g8/`。

以上均位于服务器`/root/autodl-tmp/`，属于ignored运行资产。没有训练权重、没有
重新下载模型、没有执行关机、提交或推送。
