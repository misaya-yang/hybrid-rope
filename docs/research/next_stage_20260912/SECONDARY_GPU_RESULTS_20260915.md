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

32K点估计为正，64K点估计反转；AUC差几乎为零且区间跨零。该结果只能写成
“Qwen小面板未确认跨模型优势”，不能写成TailSpline在Qwen稳定胜出或稳定失败。它只有6个
任务和每任务每长度18行，也不能替代Llama Full-13主实验；64K单点的负值不足以单独推导
极限外推失效机制。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_qwen25_s2_32k64k/reports/tailspline_vs_mrpro_32k64k.json`
（SHA256 `9fb351b32ecd148cfc16f2286e75049c8a6b2fe7d8e522949da9ada780f72535`）。

## 2. Llama S4：NIAH长度×深度诊断

### 合同

- checkpoint：`Meta-Llama-3-8B-Instruct`；exact TailSpline vs exact MrPro，沿用经典S4
  两臂的同band、gain和静态表；
- 网格：8K/16K/24K/32K × depth 10%至90%（步长10%）× 3 repeats，108行/臂；
- 任务：Paul Graham filler中的单个数字needle；指标为ROUGE-1 recall，且报告确认每一行都与
  official substring recall一致；
- 推断：在每个冻结length-depth格内成对重采样repeat，再对36个格等权汇总。

### 结果

TailSpline/MrPro全格macro为`88.89%/92.59%`，差`−3.70pp`，95%区间
`[−7.41,0.00]pp`。按长度：

| 长度 | TailSpline | MrPro | 差值 |
|---|---:|---:|---:|
| 8K | 74.07% | 85.19% | −11.11pp |
| 16K | 88.89% | 92.59% | −3.70pp |
| 24K | 92.59% | 92.59% | 0.00pp |
| 32K | 100.00% | 100.00% | 0.00pp |

每格只有3次重复，单次失败会让格分数跳变`33.33pp`，因此该诊断对局部差异明显欠功效；
32K两臂各9个深度格、每格3次均成功，又形成明显ceiling。它说明负格集中在8K/16K，可用于定位
需要确认的区域，但既不能确认MrPro总体优于TailSpline，也不构成独立于RULER retrieval
family的新benchmark。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap/reports/tailspline_vs_mrpro_niah_heatmap.json`
（SHA256 `f6d1a30e0254fbc681aba104c414cceb72093f2dee017c40e73e4bf8b1f88ae4`）。

新的Full20确认使用全新seed运行中（20 repeats/cell，720行/臂）；本页不记录中途分数，
待完整报告生成后再独立判读。

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

PPL越低越好：TailSpline在8K/16K略差，在32K略好；AUC差仅`+0.000060`，95%文档配对
区间`[−0.001814,+0.001917]`跨零。因此ProofPile-only证据支持“1×–4×范围内总体PPL
近似持平、长度间存在方向变化”，不支持逐长度全面占优，也不支持用该近零差解释RULER
主结果。这里使用32篇冻结文档，多于MrRoPE论文披露的10条随机ProofPile序列，但并不因此
变成对论文整套评测协议的完整复现。

远端报告：
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap/reports/proofpile32_ppl_curve.json`
（SHA256 `243f95827abbd7341ac1f7a7266f218779b7529a624797e9ab71a8bc3907bbff`）。
