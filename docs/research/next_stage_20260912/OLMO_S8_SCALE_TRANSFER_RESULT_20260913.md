# OLMo S8 冻结 allocation 迁移结果

2026-09-13。**状态：候选、canonical MrPro、BM与static-YaRN的Core-6
4K/16K/32K×6行已完成；Native 4K同prompt尚未补，因此不称完整Native保持解。**

## 1. 冻结构造与面板

模型为`OLMo-2-0425-1B-Instruct`，Native长度`L=4096`。冻结OLMo S4阶段的band
`[14,31]`和`mix075=0.25 BM+0.75 front-loaded`，只把部署倍率改为S8，slow端完整
`/8`。`0.75`是前序开发冻结的经验系数，不是由当前理论唯一推出。比较两个gain：

- `g=1.099065`：从Llama得到的统一log中点规则；
- `g=1.138629`：保持OLMo S4父表的实际gain，用于隔离频率倍率迁移。

4K/16K来自已有冻结OLMo Core-6低档面板；32K由同一RULER上游、OLMo tokenizer和
独立seed生成，每任务6行、32640输入token+128输出预留。没有跨模型复用token IDs。
主指标为完整生成RULER official contains、task-equal宏平均和该4K/16K/32K采样网格的
log-AUC。

## 2. 主结果

| 方法 | 4K | 16K | 32K | log-AUC | worst |
|---|---:|---:|---:|---:|---:|
| **mix075 S8, 保持OLMo父gain** | **61.16** | **37.96** | **15.00** | **41.87** | **15.00** |
| mix075 S8, 统一log中点gain | 55.79 | 31.39 | 11.85 | 36.27 | 11.85 |
| BM S8 | 25.23 | 19.72 | 12.04 | 20.28 | 12.04 |
| canonical MrPro S8 | 8.33 | 5.56 | 2.78 | 6.02 | 2.78 |
| static YaRN S8 control | 8.80 | 5.56 | 5.56 | 6.64 | 5.56 |

父gain候选相对BM的正式AUC与worst分别领先`+21.59pp`和`+2.96pp`；配对95%区间
分别为`[+13.70,+29.59]pp`和`[-7.78,+14.81]pp`。相对MrPro的AUC差`+35.85pp`，区间
`[+29.15,+42.84]pp`，worst差`+12.22pp`，区间`[+4.44,+22.22]pp`。相对
static-YaRN的AUC与worst区间也均为正。

这构成第二checkpoint上的真实相对正结果：冻结mix075 allocation显著提高采样区间
平均，并在32K相对BM领先`2.96pp`、相对MrPro领先`12.22pp`。绝对32K分数为15，
它同时记录该1B checkpoint在S8压力端点的能力水平。

## 3. gain迁移判决

保持频率表逐位不变，把gain从统一规则`1.099065`恢复为OLMo父gain`1.138629`后，
4K/16K/32K分别提高`+5.37/+6.57/+3.15pp`，AUC提高`+5.60pp`。逐任务有反转，
但总曲线一致改善。因此：

- 同一候选频率表内，两个gain的差异得到直接识别；候选相对BM/MrPro仍是完整
  `frequency × gain`配置比较，不能把全部正差归为纯频率因果效应；
- Llama选择出的log中点gain规则在该OLMo面板上不是两个已测gain中的较优配置；
- 现有两点不足以证明该规则完全不迁移或“必须”逐checkpoint校准，只证明gain不能在
  未经对照时与transition形状捆成已验证的通用公式。

## 4. 结论与归因边界

- 正式结论：同一mix系数和模型内冻结band从S4迁到S8，在OLMo上超过BM/MrPro的
  采样网格AUC，并在已测32K端点保持正差。
- 归因：两个gain的直接对照表明OLMo父gain更适合该checkpoint；连续长度行为与Native
  参考需由对应面板回答，不由三个采样长度外推。
- static YaRN只代表官方频率映射在冻结checkpoint上的零训练control，不代表配套
  训练后的完整YaRN方法。
- 任务族分解与宏平均并列报告，保留各项实际正负分数。

## 5. owner

- 32K数据：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/olmo_s8_core6_32k6_seed20260921/`
- 父gain配对报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/olmo_s8_mix075_g4standard_vs_mrpro_yarn_bm_core6_4k16k32k6.json`
- 统一gain配对报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/olmo_s8_mix075_vs_mrpro_yarn_bm_core6_4k16k32k6.json`
- raw runs：同一根目录`runs/olmo_*_s8_*`。
- 数据准备：`experiments/llama3_60dir_20260911/prepare_planb_panel.py`的显式generic模式。

本轮GPU实验到此停止。Native 4K仍是未测边界；若未来因新benchmark重新开启，只需
补匹配Native，不应触发新的OLMo band或gain sweep。
