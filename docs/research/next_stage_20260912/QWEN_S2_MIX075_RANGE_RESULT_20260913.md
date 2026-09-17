# Qwen2.5-1.5B S2 mix075 固定表区间结果

2026-09-13。**状态：低6开发块、额外12行/任务/长度的冻结确认块及全部预定强对照
均已完成。没有在确认块后调参或增加候选。**

## 1. 冻结构造

模型是`Qwen2.5-1.5B-Instruct`，Native长度`L=32768`，部署倍率`S=2`。候选由Llama/
OLMo阶段冻结的规则迁移：

- band按Native winding坐标映射为`[22,39]`；该band在历史Qwen开发实验中已开封，
  不能称新prospective预测；
- exponent allocation固定为`0.25 BM + 0.75 front-loaded`；其中`0.75`是前序开发中
  冻结的经验系数，不是由当前理论唯一推出；
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

正重采样比例不是后验概率或自动门禁。低6只构成开发块证据；候选完整配置在该块的
AUC超过BM、C42与static-YaRN，对MrPro虽三个长度点估计都正，区间仍跨0。
候选与C42虽然band相同，但gain不同，所以该差值是组合配置新增价值，不是transition
形状的纯因果效应。

## 3. 冻结确认块与累计18结果

确认块使用每个任务/长度索引6--17的12行，共216条。该块本身的结果为：

| 方法 | 32K | 48K | 64K | log-AUC | worst |
|---|---:|---:|---:|---:|---:|
| **mix075 `[22,39]`** | **81.81** | 76.64 | 70.02 | **76.78** | 70.02 |
| same-band C42 | 77.01 | 76.46 | **74.72** | 76.26 | **74.72** |
| canonical MrPro | 77.99 | **77.34** | 70.35 | 76.08 | 70.35 |
| BM | 75.56 | 77.06 | 66.90 | 74.51 | 66.90 |

独立追加块的正式AUC由候选相对BM、C42、MrPro分别领先
`+2.27/+0.52/+0.70pp`；候选worst比C42低`4.70pp`。对应95%重采样区间为
`[-1.65,+6.02]`、`[-2.26,+3.25]`、`[-3.30,+4.75]pp`，作为该12行块的
采样稳定性。这个块保留“区间平均领先、C42远端更强”的完整轮廓。

将预先报告的低6与冻结追加12行合并，每任务/长度18行、总计324条：

| 方法 | 32K | 48K | 64K | log-AUC | worst |
|---|---:|---:|---:|---:|---:|
| **mix075 `[22,39]`** | **83.30** | **79.65** | 74.86 | **79.72** | 74.86 |
| same-band C42 | 79.34 | 76.44 | **77.16** | 77.43 | **76.44** |
| canonical MrPro | 79.66 | 77.99 | 73.66 | 77.58 | 73.66 |
| BM | 76.56 | 77.62 | 70.52 | 75.84 | 70.52 |

累计配对结果：

| 对照 | 候选AUC差 | AUC 95%区间 | 候选worst差 | worst 95%区间 |
|---|---:|---:|---:|---:|
| BM | **+3.88** | **[+0.81,+6.93]** | +4.34 | [-0.35,+9.03] |
| C42 | +2.29 | [-0.30,+4.97] | -1.57 | [-5.43,+4.98] |
| MrPro | +2.14 | [-1.32,+5.59] | +1.20 | [-3.19,+5.74] |

所以本轮累计固定面板的正式AUC由候选相对BM、C42、MrPro分别领先
`+3.88/+2.29/+2.14pp`。重采样区间对BM全为正，对C42/MrPro跨零，作为稳定性分解。
累计结果包含已经开封的低6，不冒充独立holdout；追加块仍单独列出，便于读者看到
冻结追加数据上的实际点分。

## 4. Native保持

追加块72个32K prompts上，候选`81.81`、Native`77.18`，正式差值`+4.63pp`，95%区间
`[-1.44,+11.04]pp`作为该块稳定性。累计108个匹配prompts上，候选
`83.30`、Native`76.79`，差`+6.51pp`，95%区间`[+1.45,+11.88]pp`。累计逐任务没有
负点估计：FWE `+1.85pp`、MK2 `+16.67pp`、multiquery `+2.78pp`、single持平、QA
`+16.67pp`、VT `+1.11pp`。因此在该Core-6采样面板上满足Native保持；这不等于所有
Native任务、完整RULER或连续区间已证明无损。

## 5. 当前科学含义

1. 在第三checkpoint的累计固定面板上，冻结mix075组合配置的AUC超过BM、MrPro和
   same-band C42，差值分别为`+3.88/+2.14/+2.29pp`。由于C42的gain也不同，
   这是完整配置比较；transition形状的单因素归因交给匹配控制。
2. 候选在32/48/64K采样网格上获得较高AUC并保持Native，但C42的64K与worst更高；
   这是区间平均与端点/最坏长度不等价的直接实例，不是全指标支配。
3. Qwen结果建立了第三checkpoint的正面迁移读数；Llama Native FWE与OLMo绝对32K
   结果各自在自己的合同中保留，用于刻画checkpoint条件下的gain与任务响应。
4. `[22,39]`已参与理论形成，故本结果是冻结transition规则的回顾/迁移验证，不是
   band公式的独立留出证明。

## 6. 确认合同与owner

确认块从既有冻结324行源面板中取每格索引6–17，共216条，方法与对照均未再调参。
只扩候选、MrPro、BM、C42和Native@32K；没有扩明显较弱且不是完整方法的
static-YaRN。

- 低6综合报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_mrpro_yarn_bm_c42_core6_32k48k64k6.json`
- Native报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_native_core6_32k6.json`
- 追加块综合报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_mrpro_bm_c42_core6_block12_rows6_17.json`
- 累计18综合报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_mrpro_bm_c42_core6_32k48k64k18.json`
- 累计Native报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/qwen15_s2_mix075_vs_native_core6_32k18.json`
- 确认面板：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/qwen_s2_core6_block12_rows6_17/`
- raw runs：同一根目录`runs/qwen15_s2_*`。

本轮GPU实验到此停止。后续若更换benchmark或新增独立面板，必须复用这四张表和完整
配置，不得把当前追加块重新并入调参数据。
