# Qwen2.5-1.5B S2 mix075 固定表区间结果

2026-09-13。**状态：低6开发块及全部强对照已完成；额外12行/任务/长度的冻结确认块
正在运行。以下低6结果真实有效，但在确认完成前不称最终mini结论。**

## 1. 冻结构造

模型是`Qwen2.5-1.5B-Instruct`，Native长度`L=32768`，部署倍率`S=2`。候选由Llama/
OLMo阶段冻结的规则迁移：

- band按Native winding坐标映射为`[22,39]`；该band在历史Qwen开发实验中已开封，
  不能称新prospective预测；
- exponent allocation固定为`0.25 BM + 0.75 front-loaded`；
- full `/2` slow endpoint；
- gain固定为`sqrt(1+0.1 ln2)=1.0340767467`；
- 全部层和32K/48K/64K始终使用同一张FP32表，无运行时换表，无权重更新。

面板为Core-6 × 32K/48K/64K × 6行，完整生成后按RULER official contains计分，
长度内task-equal，再计算采样网格log-AUC和worst。对照为canonical MrPro、BM、同band
C42、static-YaRN频率图control及Native@32K。

## 2. 低6开发结果

| 方法 | 32K | 48K | 64K | log-AUC | worst |
|---|---:|---:|---:|---:|---:|
| **mix075 `[22,39]`** | **86.30** | **85.65** | 84.54 | **85.61** | **84.54** |
| same-band C42 | 83.98 | 76.39 | 82.04 | 79.78 | 76.39 |
| canonical MrPro | 83.01 | 79.31 | 80.28 | 80.59 | 79.31 |
| BM | 78.56 | 78.75 | 77.78 | 78.49 | 77.78 |
| static YaRN control | 76.90 | 75.00 | **85.42** | 77.72 | 75.00 |

候选从32K到64K只下降`1.76pp`。static YaRN的64K端点高`0.88pp`，但32K/48K低
`9.40/10.65pp`，提供了“端点强不等于区间最优”的同checkpoint实例。static YaRN
没有配套继续训练，不能冒充完整YaRN方法。

配对bootstrap结果：

| 对照 | 候选AUC差 | AUC 95%区间 | 候选worst差 | worst 95%区间 |
|---|---:|---:|---:|---:|
| BM | +7.11 | **[+3.03,+11.62]** | +6.76 | **[+1.02,+14.35]** |
| C42 | +5.83 | **[+1.03,+11.00]** | +8.15 | [-0.83,+14.72] |
| MrPro | +5.02 | [-0.93,+10.98] | +5.23 | [-1.99,+13.15] |
| static YaRN | +7.89 | **[+2.88,+13.26]** | +9.54 | [-0.46,+16.85] |

正重采样比例不是后验概率或自动门禁。当前低6已确认候选完整配置的AUC超过BM、
C42与static-YaRN；对MrPro虽三个长度点估计都正，区间仍跨0，需由冻结确认块决定。
候选与C42虽然band相同，但gain不同，所以该差值是组合配置新增价值，不是transition
形状的纯因果效应。

## 3. Native保持

同一36个32K prompts上，候选`86.30`、Native`76.02`，差`+10.28pp`，配对95%区间
`[+2.13,+19.40]pp`。逐任务：FWE与single持平；MK2 `+33.33pp`、multiquery
`+8.33pp`、QA `+16.67pp`、VT `+3.33pp`。因此在这组开发面板上，候选满足Native
不退化且明显提高；这不等于所有Native任务或完整benchmark已证明无损。

## 4. 当前科学含义

1. 在第三checkpoint上，冻结mix075组合配置不仅超过BM/MrPro，也在同band C42上
   取得正AUC差，说明“只知道band位置”不足以复现候选收益；由于gain也不同，不能把
   差值全部归因于transition形状。
2. 候选的平坦区间曲线直接贴合“固定单表最大化`[L,SL]`效用”的问题，而不是只在
   2L端点取胜。
3. Llama存在严重Native FWE损伤、OLMo绝对32K仍低；Qwen正结果不能倒推成跨模型
   普遍无损。更可信的结论是allocation显示迁移潜力，但gain和任务代价依checkpoint
   条件化。
4. `[22,39]`已参与理论形成，故本结果是冻结transition规则的回顾/迁移验证，不是
   band公式的独立留出证明。

## 5. 确认合同与owner

确认块从既有冻结324行源面板中取每格索引6–17，共216条，方法与对照均不再调参。
只扩候选、MrPro、BM、C42和Native@32K；不扩明显较弱且不是完整方法的static-YaRN。

- 低6综合报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_mrpro_yarn_bm_c42_core6_32k48k64k6.json`
- Native报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_native_core6_32k6.json`
- 确认面板：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/qwen_s2_core6_block12_rows6_17/`
- raw runs：同一根目录`runs/qwen15_s2_*`。

确认完成后必须同时报告第二块本身和累计18结果；若方向反转，保留反转，不能按累计
均值掩盖。
