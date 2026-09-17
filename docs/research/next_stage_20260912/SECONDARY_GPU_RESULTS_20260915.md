# TailSpline–MrPro辅助GPU结果

更新：2026-09-15。本文只汇总三个已完成的辅助实验，保留各自合同和解释边界；它们不替代
Llama clean Full-13主结果，也不合并成一个总分。

## 1. Qwen2.5-3B：32K/64K跨模型小面板

### 合同

- checkpoint：`Qwen2.5-3B-Instruct`，Native 32K；冻结权重；
- 方法：exact TailSpline vs exact MrPro，共同`S=2`、band `[23,40]`、gain
  `1.0693147180559945`；
- 面板：Core-6任务`fwe`、`niah_multikey_2`、`niah_multiquery`、
  `niah_single_2`、`qa_1`、`vt`，32K/64K各18行/任务，共216个严格配对prompt/臂；
- 指标：RULER official contains分数，先任务等权汇总每个长度，再计算log-length AUC；
- 推理：batch 2、8K prefill chunk。TailSpline行复用同一冻结三长度运行中的32K/64K子集，
  MrPro单独生成；验证报告确认两臂prompt集合相同。

### 结果

| 指标 | TailSpline | MrPro | TailSpline−MrPro | 95%配对区间 |
|---|---:|---:|---:|---:|
| 32K task macro | 0.905247 | 0.885340 | **+1.9907pp** | [−0.7407,+5.3086]pp |
| 64K task macro | 0.789969 | 0.811265 | **−2.1296pp** | [−7.3611,+2.7932]pp |
| log-length AUC | 0.847608 | 0.848302 | **−0.0694pp** | [−3.0633,+2.8858]pp |

这张固定Qwen小面板的正式结果是：TailSpline在32K领先`1.99pp`，MrPro在64K领先
`2.13pp`，两长度AUC由MrPro领先`0.07pp`，总体几乎持平。它准确记录了S2下的长度反转，
与后续Qwen S4/128K和S8/256K正结果并列使用；不把不同倍率、任务集合和样本重新揉成一个总分。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_32k64k/reports/tailspline_vs_mrpro_32k64k.json`
（SHA256 `9fb351b32ecd148cfc16f2286e75049c8a6b2fe7d8e522949da9ada780f72535`）。

## 2. Llama S4：NIAH长度×深度诊断与Full20确认

### 合同

- checkpoint：`Meta-Llama-3-8B-Instruct`；exact TailSpline vs exact MrPro，沿用经典S4
  两臂的同band、gain和静态表；
- pilot网格：8K/16K/24K/32K × depth 10%至90%（步长10%）× 3 repeats，108行/臂；
- Full20确认：相同36格、全新seed、每格20 repeats，720行/臂；两次运行的样本与结果分开；
- 任务：Paul Graham filler中的单个数字needle；指标为ROUGE-1 recall，且报告确认每一行都与
  official substring recall一致；
- 推断：在每个冻结length-depth格内成对重采样repeat，再对36个格等权汇总。

### 三重复pilot

TailSpline/MrPro全格macro为`88.89%/92.59%`，差`−3.70pp`，95%区间
`[−7.41,0.00]pp`。按长度：

| 长度 | TailSpline | MrPro | 差值 |
|---|---:|---:|---:|
| 8K | 74.07% | 85.19% | −11.11pp |
| 16K | 88.89% | 92.59% | −3.70pp |
| 24K | 92.59% | 92.59% | 0.00pp |
| 32K | 100.00% | 100.00% | 0.00pp |

该固定pilot上MrPro领先`3.70pp`，负格集中在8K/16K；32K两臂均为100%。每格3次意味着
单次变化对应`33.33pp`，因此它的任务职责是定位长度×深度差异，随后由Full20给出更稳定的同协议成绩。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap/reports/tailspline_vs_mrpro_niah_heatmap.json`
（SHA256 `f6d1a30e0254fbc681aba104c414cceb72093f2dee017c40e73e4bf8b1f88ae4`）。

### Full20确认

两臂各720条均完成，正式全格macro为TailSpline/MrPro `87.50%/88.47%`，差
`−0.97pp`，即该固定单针网格由MrPro领先`0.97pp`；配对格内重采样95%区间为
`[−3.06,+1.11]pp`。720个配对中TailSpline独对26条、MrPro独对33条、两者同结果661条；
两臂均无空输出或触顶输出。相比pilot，差值从`−3.70pp`收缩到`−0.97pp`。

| 长度 | TailSpline | MrPro | 差值 | 95%配对区间 |
|---|---:|---:|---:|---:|
| 8K | 74.44% | 78.33% | −3.89pp | [−9.44,+1.67]pp |
| 16K | 81.11% | 81.11% | 0.00pp | [−5.00,+5.00]pp |
| 24K | 94.44% | 95.56% | −1.11pp | [−4.44,+1.67]pp |
| 32K | 100.00% | 98.89% | +1.11pp | [0.00,+2.78]pp |

长度分解的正式点分为8K `−3.89pp`、16K持平、24K `−1.11pp`、32K `+1.11pp`；
九个depth分解中40% depth为`−5.00pp`。这些格定位单针任务的局部轮廓，Full20总体成绩仍为
MrPro领先`0.97pp`。

便携报告：
`experiments/iclr2027_three_track_sprint_20260915/reports/niah_full20_tailspline_vs_mrpro.json`
（SHA256 `d29f78ab35921912bce717198bbfe26809dda1a6a436cada5cbdc938b90d6b18`）。服务器raw SHA256：
TailSpline `90fa3fcbfdc5ef386254b96293edca39270c3f2e05ed7693a83342c9666ce19d`，
MrPro `faf1a2867a8f572e3b5fcfeee1e64a2580f8e850ed6b7fbcea25eb056964373b`。

本实验属于RULER retrieval family内部诊断。正确论文表述是：
**TailSpline在完整RULER-13上大幅领先；在该单针长度×深度网格上MrPro领先`0.97pp`。**

## 3. Llama S4：ProofPile-only PPL曲线

### 合同

- checkpoint：`Meta-Llama-3-8B-Instruct`；exact TailSpline vs exact MrPro；
- 数据：冻结的32篇ProofPile test文档，同一文档跨8K/16K/32K共享；各臂分别评估
  262,144/524,288/1,048,576个目标token；
- 精度：bfloat16 checkpoint forward，float32 logits/loss accumulation；
- 指标：先按各长度全部token汇总whole-prefix NLL并转PPL，再计算log-length PPL AUC；
  文档配对bootstrap保持同一抽中文档跨三个长度共同出现。

### 结果

| 长度 | TailSpline PPL | MrPro PPL | TailSpline−MrPro |
|---|---:|---:|---:|
| 8K | 3.784889 | 3.776884 | +0.008005 |
| 16K | 3.062598 | 3.053680 | +0.008917 |
| 32K | 2.669506 | 2.695104 | −0.025598 |
| log-length AUC | 3.144897 | 3.144837 | **+0.000060** |

PPL越低越好：MrPro在8K/16K分别低`0.0080/0.0089`，TailSpline在32K低`0.0256`；
AUC由MrPro低`0.000060`，属于报告精度下的总体近似持平，同时保留清晰的长度方向变化。
文档配对区间`[−0.001814,+0.001917]`作为换文档敏感性列入附录。这里使用32篇冻结文档，
多于MrRoPE论文披露的10条随机ProofPile序列。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap/reports/proofpile32_ppl_curve.json`
（SHA256 `243f95827abbd7341ac1f7a7266f218779b7529a624797e9ab71a8bc3907bbff`）。

## 4. Qwen极限面板与解释：2026-09-16

最新Qwen-3B S2/64K NIAH-8为T/P86.25/80.625%，S4/128K为72.50/73.125%；
Qwen-1.5B S4/128K为81.25/76.875%。每任务5条、每臂40条；正式点分分别由
TailSpline、MrPro、TailSpline领先。
这与本页旧Core-6面板属于不同协议。

[逐任务与同倍率几何分析](../reviews/QWEN_ALLOCATION_RESPONSE_ANALYSIS_20260916.md)
核对最新报告、实际表、配对行和输出：3B128K当前小面板的宏平均接近来自样本内满分/零分与局部抵消；
这些小面板用于定位Qwen的倍率和任务响应，完整clean结果另列；
同S4下与Llama的几何改动幅度相同量级。固定模型对中频配置的使用差异是待辨别机制，
不能仅由base、head数或两个不同seed的小面板推出因果解释。

## 5. Qwen256K检索与S4自然QA完成更新（2026-09-16）

新完成报告已同步至实验目录：

- [S8/256K单针三任务](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen3b_s8_256k_health.json)：每任务5条，T/P93.33/40.00%，差+53.33pp，配对区间[+33.33,+66.67]pp。
- [S4/128K预算InfiniteBench En.QA](../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_qwen3b_s4_128k_en_qa.json)：35题/7个来源上下文簇；F1为19.14/15.90%，差+3.24pp，overall簇区间[+0.20,+7.41]pp；逐题13/6/16。

前述旧Qwen小面板只描述对应倍率和样本；当前Qwen总体结论以最新Full-13与极限面板为准。
同S4下单针饱和与自然QA收益并存，说明应按任务和运行条件解释配置价值。
[base与尺度综合分析](../reviews/QWEN_BASE_SCALE_SYNTHESIS_20260916.md)已更新；未据此修改论文或新增GPU任务。
