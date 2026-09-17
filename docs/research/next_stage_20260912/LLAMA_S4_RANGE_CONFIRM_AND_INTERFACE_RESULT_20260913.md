# Llama S=4 固定表区间确认与三接口结果

更新：2026-09-13。状态：**同口径 Core-6×三长度×12 行/格的 BM、MrPro、C42
比较已完成；Native 8K 对照已完成。log-gain中点候选在8/16/32K点估计、AUC和
worst上同时超过三个扩展基线，AUC与worst的配对区间均为正；但它仍有严重的Native
FWE损伤，因此是当前最强固定扩展表，不是已经闭合的Native保持解。**

## 1. 问题与固定表身份

模型是 `Meta-Llama-3-8B-Instruct`，Native 窗口 `L=8192`，设计倍率 `S=4`，运行
长度为 8K/16K/32K。每个方法在全部层、全部长度使用同一张固定表，没有按长度换表，
没有训练或权重更新。

本轮候选 `mix075 [16,34]` 为

\[
m=0.25m_{\mathrm{BM}}+0.75m_{\mathrm{frontloaded}},\qquad
m_{\mathrm{frontloaded}}(q)=1-\frac{(n-q)(n-q+1)}{n(n+1)},
\]

band 为 `[16,34]`，slow plateau 完整 `/4`，gain 为
`1+0.1 ln 4 = 1.1386294361`。比较对象为 canonical BM、canonical MrPro，以及相同
`[16,34]` band 的 C42。BM/MrPro 没有改 band，不把机制 control 冒充 canonical
baseline。

在相同频率表上，唯一新增强候选把gain冻结为log-space中点
`sqrt(1.1386294361)=1.0670658068`。它不是重新搜索频率，也没有按长度或任务换gain。

## 2. Benchmark 与可比性

- 任务：`niah_single_2`、`niah_multikey_2`、`niah_multiquery`、`vt`、`fwe`、`qa_1`。
- 长度：8K、16K、32K。
- 每个 task×length 有两个互不重复的 6-row 数据块，累计 12 行；每臂 216 prompts。
- 指标：RULER official contains，先 cell mean，再 task-equal length macro，最后按
  log-length 梯形 AUC；同时报告 worst-length、任务族、EOS 和 cap。
- 四臂 prompt hash 完全配对，解码、精度、runner 和 scorer 相同。
- 候选与 C42 的数据块在本轮 BM/MrPro 补齐前已经分批读过，不称完全未开封 holdout；
  新增价值是补齐此前缺失的 canonical BM/MrPro 同 prompt 对照。

## 3. 主结果

| 固定表 | 8K | 16K | 32K | log-AUC | worst |
|---|---:|---:|---:|---:|---:|
| **mix075 `[16,34]` + log-gain中点** | **83.61** | **81.62** | **76.94** | **80.95** | **76.94** |
| mix075 `[16,34]` + 标准gain | 80.58 | 76.62 | 73.66 | 76.87 | 73.66 |
| BM | 82.45 | 77.31 | 61.62 | 74.68 | 61.62 |
| MrPro | 77.08 | 78.59 | 58.33 | 73.15 | 58.33 |
| C42 `[16,34]` | 81.11 | 75.09 | 68.96 | 75.06 | 68.96 |

候选差值：

| 对照 | 8K | 16K | 32K | AUC | worst |
|---|---:|---:|---:|---:|---:|
| log-gain中点−BM | +1.16 | +4.31 | **+15.32** | **+6.27** | **+15.32** |
| log-gain中点−MrPro | +6.53 | +3.03 | **+18.61** | **+7.80** | **+18.61** |
| log-gain中点−C42 | +2.50 | +6.53 | **+7.99** | **+5.89** | **+7.99** |

配对、task 内重采样 20,000 次的 95% 区间：

- 相对 BM：AUC `[+3.10,+9.45]pp`；worst `[+7.99,+21.51]pp`。
- 相对 MrPro：AUC `[+3.64,+12.07]pp`；worst `[+10.56,+25.46]pp`。
- 相对 C42：AUC `[+2.77,+9.05]pp`；worst `[+0.39,+15.14]pp`。

这些区间支持当前注册面板上的三长度AUC和worst均超过BM/MrPro/C42；它不证明每项
任务逐点支配、跨benchmark SOTA或跨checkpoint泛化。重采样正差比例不是后验胜率。

log-gain中点由第一块选择，因此累计12行/格区间不能冒充选型后独立确认。未参与gain
选择的第二块单独得到：中点8K/16K/32K `83.70/81.71/82.22`，AUC/worst
`82.34/81.71`；标准gain为AUC/worst `77.05/74.44`，BM
`76.46/71.11`，MrPro `75.20/67.78`，C42 `76.53/74.03`。中点的AUC差95%区间
相对标准gain、BM、MrPro、C42分别为 `[+1.24,+9.69]`、`[+1.46,+10.26]`、
`[+1.56,+12.62]`、`[+1.79,+10.06]pp`。该块已有其他表的历史暴露，但没有参与
本次gain选择；准确身份是gain-selection holdout block，不是全新benchmark。

log-gain中点的任务族AUC为 retrieval/tracking/aggregation/QA
`95.14/93.33/46.53/60.42`。它显著修复了标准gain候选的aggregation `23.61`，同时
QA低于BM/MrPro；这是正式任务族分解。32K 的 EOS/cap 为 `98.61%/1.39%`，好于 BM 的
`90.28%/9.72%`，因此远端收益不只是 contains scorer 在截断输出上的偶然命中。

## 4. Native 保留反例

在完全相同的 72 个 8K prompts 上，Native 为 `92.87`，标准gain候选为 `80.58`，
log-gain中点为 `83.61`。中点把差缩小到 `−9.26pp`，但主要损伤仍来自 FWE：中点
`33.33`，Native `88.89`；其余NIAH/VT/QA基本保持。这不是小噪声，也不能用相对扩展
基线的正差掩盖。中点−Native的配对task-equal 95%区间为
`[-13.24,-5.28]pp`。

因此中点候选满足“相对现有固定扩展表提升全区间AUC和 worst”的实用比较，但**不满足原问题
要求的 Native 保持约束**。论文若现在使用，只能写 Pareto/endpoint-robust candidate，
不能写成完整 range-optimal 解。

## 5. gain 单变量结果

保持候选 64 个 FP32 频率值逐位不变，只把 gain 从 `1.138629` 改为 `1.0`。在第一批
6 行/格上：

| gain | 8K | 32K | 8K FWE | 32K VT |
|---|---:|---:|---:|---:|
| 1.138629 | 83.38 | 65.79 | 11.11 | 93.33 |
| 1.0 | **86.90** | 56.71 | **72.22** | 50.00 |

gain 明确有强条件效应和长度/任务异质性：降低 gain 大幅修复 Native FWE，却同时
破坏远端 VT/QA。这排除了“Native 损伤完全由 band 或 transition 频率造成”的解释，
也说明不能把 gain 当无关部署常数。这里只有同一张表的 gain 主效应；没有第二张
频率表的匹配两格，不能称 shape×gain 交互。

在 `log gain` 上取两端中点，即 `g=sqrt(1.138629)=1.067066`，第一批结果为8K
`83.52`、16K `81.53`、32K `71.67`：相对标准 gain 是
`+0.14/+2.73/+5.88pp`，8K FWE 提高到 `27.78`。随后冻结同一中点补第二个独立
6-row block，得到8K/16K/32K `83.70/81.71/82.22`；累计12行/格为
`83.61/81.62/76.94`。中点因此从响应检查晋升为当前强候选，但相对同prompt Native
8K仍低约`9.26pp`。log-space中点本身不保证任务Pareto最优，也不据此继续宽gain
sweep。

FWE raw tokens 进一步定位到终止行为，而不是 scorer 漂移：标准 gain 在8K有5/6条
只生成1个EOS token，中点有4/6条，gain 1.0只有1/6条；三者EOS均合法记录且没有
触 cap。32K则是中点只有1/6条立即EOS，标准/gain1分别4/6和3/6。gain 对不同长度
的EOS阈值明显非单调，这解释了宏平均的长度交换，但不证明频率分配与内容读出无关。

### Llama 全-profile depth

以log-gain中点为父表，保持band、normalized transition shape和gain固定，把全部
exponent乘 `c=log_4(3.6)=0.923998`，即slow端从完整`/4`变成`/3.6`。这是整个
transition+tail共同缩浅的**全-profile剂量干预**，不是只改低频tail。第一批6行/格：

| depth | 8K | 32K |
|---|---:|---:|
| full `/4` | 83.52 | 71.67 |
| full-profile `/3.6` | 83.52 | **0.00** |

32K六个任务全部为0；36条中22条触 cap，其余多为短的无关文本，不是立即EOS或
scorer异常。它支持当前Llama S=4表需要full depth维持目标端，且解释了为何只把整张
表“稍微拉回Native”不是可行修复；它不能证明每个独立slow槽的`/S`都任务最优，问题②
的tail-only慢/快双侧仍与本结果分开。

## 6. OLMo 三接口的本轮约束

### Band 局部四格

同一 OLMo S=4 `mix075` transition、full `/4` tail 和 gain 下，Core-6 low108：

| band | 4K | 8K | 16K | AUC | worst |
|---|---:|---:|---:|---:|---:|
| `[14,31]` | 79.77 | 73.70 | **54.77** | 70.49 | **54.77** |
| `[15,31]` | 79.49 | **75.00** | 53.15 | **70.66** | 53.15 |
| `[14,32]` | **82.13** | 72.13 | 53.19 | 69.90 | 53.19 |
| `[15,32]` | 79.63 | 73.94 | 52.92 | 70.10 | 52.92 |

没有一张联合表优于所有父表。AUC 的交互差分点估计约 `+0.03pp`，但分长度为
`−2.22/+0.51/+1.34pp`，所以不能写“没有交互”；只能写这个局部邻域没有找到联合
改进，slow 边界 31 对 endpoint/worst 更稳。

### Tail depth

在 OLMo `[14,31]` BM 父表上，把 slow tail 从 `/4` 改快到 `/3.9`，累计 20 行/格
得到 8K `+1.29pp`、16K `−5.35pp`；`/3.6` 损伤更大。它证明更快侧存在清晰剂量
响应但没有 range win；没有测试更慢侧，不能声称 `/S` 为任务全局或局部最优。

### Transition 与完整答案 margin

OLMo `mix075` 在 frozen324 上相对 BM/C42 有正 AUC 点估计，但 fresh60 出现长度交换，
不能宣称稳定支配。随后从完整答案+EOS逐 token bottleneck margin 出发，在9条 CAL
上得到 gap `10→7` 的 checkpoint-conditioned 方向：一阶预测在4K/8K/16K全改善，
有限正步也全改善且反方向全变差。它在未参与选择的36条自由生成上却得到：

| 表 | 4K | 8K | 16K | AUC | worst |
|---|---:|---:|---:|---:|---:|
| parent mix075 | 66.67 | 58.33 | 66.67 | 62.50 | 58.33 |
| margin gap proposal | 66.67 | 58.33 | 50.00 | 58.33 | 50.00 |

负差全部来自16K multikey。结论是：该完整模型 margin 在 CAL 内的导数和有限响应成立，
但 9 条 CAL 的方向没有跨样本迁移；不继续按确认分数重选位置。它淘汰当前低样本
selector，不否定命题中的完整输出判据，也不否定 z/transition 空间。

事后目标对齐检查发现，EOS-inclusive CAL 的9条 bottleneck 中有6条落在EOS，而
official contains不要求指定EOS时刻。去掉EOS、只对答案token重新求梯度后，gap族仍
选择完全相同的 `10→7`、步长0.0075表，三长度有限响应也仍匹配；因此上表负结果已经
覆盖修正后的gap提议，无需重复生成。同矩方向换成另一位置，但8K有限响应与一阶方向
相反，未进入自由生成。准确结论是当前低样本selector缺乏迁移支持，不是margin恒等式
错误。

Llama 上进一步把3条8K FWE的Native已接受输出冻结为答案路径，从log-gain中点表
出发求方向。全答案soft-bottleneck的gap `7→3` 在CAL有限响应改善，但72条自由生成
的8K/32K全部official宏分数与父表相同；19条raw输出虽改变，fit/select各3条FWE的
分数和立即EOS数量均未变。原因是该目标仍主要受后续token支配，而真实错误发生在
首token。改为只优化首答案token并用最大信赖步0.03后，3条CAL margin仅从
`[-0.375,0,-0.375]` 到 `[-0.25,0,-0.25]`，仍未越过greedy阈值，所以没有再做一轮
72条生成。这说明当前局部transition信赖域不足以修复FWE终止深坑；下一方法需要
非局部full-z改变或不同约束求解，不能靠重复小步方向包装成功。

### Llama fast-boundary 条件干预

保持slow边界34、transition规则、full `/4` tail和标准gain，只把fast边界
`16→18`。第一批结果从 `83.38/65.79` 变为8K/32K `82.96/65.19`；8K FWE仍为
`11.11`。因此保护两个额外fast槽没有修复Native终止深坑，也没有提高区间宏平均；
不据此继续向20盲扫。由OLMo方向提出的slow边界 `34→35` 结果为8K/32K
`82.22/65.23`，FWE `0/38.89`，同样没有联合改善；Llama 的局部 band 平移到此
收束。

## 7. 当前科学判定

已经得到的实际贡献：

1. 一张固定、零训练 Llama S=4 表在较宽 Core-6 面板上显著提高 32K 和 worst，且
   AUC 点估计超过 BM、MrPro、同 band C42；这是此前窄三任务结果的真实扩展。
2. Native 对照发现一个不能隐藏的 FWE 深坑，说明只和扩展基线比较会错误宣告问题
   已解决。
3. gain-only、band 2×2 和 margin-direction 负迁移分别排除了三个简单解释：损伤不只
   来自 frequency placement；band 联合格不能由两个单边差值直接相加预测；CAL 内
   局部 margin 改善不自动泛化为自由生成收益。

尚未达到的目标：没有一张表同时满足 Native 保留、三长度 AUC、worst 与 endpoint。
下一步只允许能直接修复 Native FWE、同时保留32K VT/retrieval的条件干预；不能继续
围绕已经失败的 margin selector、tail 更快侧或无预测的曲线枚举。

最后执行了这一非局部最小判决：从log-gain中点表出发，直接参数化63个正
log-frequency gap，固定最快端、最慢`/4`端和gain，不预设连续band。以3条8K FWE
首token margin达到`+0.25`为线性约束，最小范数QP给出的最大gap-logit位移只有
`0.01246`。但0.5/1/1.5/2倍的真实有限响应均未让两条负margin转正，且每个候选都使
至少一个Native-relative exponent低于0（最轻为`−0.0050`），违反声明的`m∈[0,1]`
预算。实现因此没有输出候选表，也没有进入自由生成。

这项结果只表明当前一阶QP的参数化可行域遗漏了`m∈[0,1]`约束，且其局部线性响应
没有预测这3条CAL上的真实有限变化；不能据此归因为BF16，也不否定full-z存在可行解。
若继续，必须把`m∈[0,1]`和真实有限margin直接放进约束，或转向训练适配，不能继续
沿同一线性梯度扩大步长追正例。

### 约束 full-z 与 transition×gain 补偿判决

随后实现了与声明空间一致的约束版 full-z：直接参数化62个内部 exponent，固定
`m[0]=0`、`m[63]=1`和gain，QP显式强制`0≤m≤1`及`m`单调。两次最小判决均未产生
可进入生成的候选：

- 绝对目标`+0.125`、单槽信赖域`0.03`时，线性预测把三条margin从
  `[-0.25,+0.125,-0.25]`推到`[+0.125,+0.270,+0.125]`，但合法射线上的真实结果
  最好只有`[-0.125,+0.125,-0.25]`；
- 改为一次只提高一个BF16 margin档、目标`-0.125`、信赖域`0.01`后，真实结果在
  0.25/0.5/0.75倍步长完全不动，1倍反而把第三条降到`-0.375`。

因此现在可以排除的是“这个父表附近、固定端点/固定gain/单调exponent族中的当前
一阶QP射线”，不是full-z方法族。未生成候选，也未消耗32K评测。

为检验降低gain后能否由更强transition压缩补回远端，做了一个最小交叉干预。所有
分数都来自同一批6行/任务的官方contains：

| 频率表与gain | 8K FWE | 32K MK2 | 32K QA | 32K VT | 32K三任务宏平均 |
|---|---:|---:|---:|---:|---:|
| mix075 `[16,34]`, gain 1.0 | 72.22 | 33.33 | 16.67 | 50.00 | 33.33 |
| front-loaded `[16,34]`, gain 1.0 | 未补 | 50.00 | 50.00 | 36.67 | 45.56 |
| front-loaded `[14,32]`, gain 1.0 | 未补 | **66.67** | 33.33 | 30.00 | 43.33 |
| mix075 `[16,34]`, gain 1.067066 | 27.78 | 50.00 | 33.33 | **80.00** | **54.44** |
| front-loaded `[16,34]`, gain 1.067066 | **0.00** | 未补 | 未补 | 未补 | 未补 |

在gain 1.0下，从mix075改为完全front-loaded确实补回`+12.22pp`远端宏平均，证明
transition强度可以补偿一部分gain损失；但VT发生反转，且仍比当前最佳中点gain父表
低`8.89pp`。再把band整体前移两槽也没有继续提高宏平均。另一方面，在中点gain下
完全front-loaded使6条8K FWE全部立即EOS、得分归零。故shape与gain存在强条件交互，
“降低gain保Native，再单调加大中频压缩补远端”不足以闭合问题；本路线到此停止，
不补完整面板，也不沿相邻band继续扫格子。

## 8. 代码与原始结果

可复核实现：

- `experiments/fixed_rope_three_interfaces_20260913/tables.py`
- `experiments/fixed_rope_three_interfaces_20260913/matched_generation_report.py`
- `experiments/fixed_rope_three_interfaces_20260913/margin_direction.py`
- `experiments/fixed_rope_three_interfaces_20260913/target_support_z.py`
- `experiments/fixed_rope_three_interfaces_20260913/full_z_fwe_repair.py`
- `experiments/fixed_rope_three_interfaces_20260913/exponent_box_z.py`
- `experiments/fixed_rope_three_interfaces_20260913/constrained_full_z_fwe_repair.py`
- `experiments/olmo_recovery_20260912/recovery_v2_eval.py`
- `tests/test_fixed_rope_three_interfaces.py`

当前服务器 owner：

- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_vs_bm_mrpro_c42_core6_balanced12.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_vs_native_8k_balanced12.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_loggain_mid_vs_bm_mrpro_c42_core6_balanced12.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_mix075_loggain_mid_vs_native_8k_balanced12.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_loggain_mid_gain_selection_holdout_block2.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/runs/llama_s4_mix075_loggain_mid_depth3p6_core6_screen6_8k32k/`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/margin_direction_mix075_r0/proposal.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/margin_direction_llama_fwe_firsttoken_r1/proposal.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/full_z_fwe_repair_midgain_r0/proposal.json`
- `/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/olmo_margin_gap_select36.json`

远端 raw generations 与 table receipts 保留在同一根目录的 `runs/`、`tables/`；这些是
本次可读 owner，不因本报告存在而声称已经随 Git 分发。
