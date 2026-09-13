# OLMo S8 冻结 allocation 迁移结果

2026-09-13。**状态：候选、canonical MrPro、BM与static-YaRN的Core-6
4K/16K/32K×6行已完成；Native 4K同prompt尚未补，因此不称完整Native保持解。**

## 1. 冻结构造与面板

模型为`OLMo-2-0425-1B-Instruct`，Native长度`L=4096`。冻结OLMo S4阶段的band
`[14,31]`和`mix075=0.25 BM+0.75 front-loaded`，只把部署倍率改为S8，slow端完整
`/8`。比较两个gain：

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

父gain候选相对BM的AUC差为`+21.59pp`，配对95%区间
`[+13.70,+29.59]pp`；worst点估计`+2.96pp`，区间`[-7.78,+14.81]pp`，因此端点
只能称不可区分/未确认优于BM。相对MrPro的AUC差`+35.85pp`，区间
`[+29.15,+42.84]pp`，worst差`+12.22pp`，区间`[+4.44,+22.22]pp`。相对
static-YaRN的AUC与worst区间也均为正。

这构成第二checkpoint上的真实相对正结果：冻结mix075 allocation显著提高采样区间
平均，同时32K端点没有低于BM点估计。绝对32K分数仍只有15，不能写成“32K能力已
解决”或完整RULER强模型。

## 3. gain迁移判决

保持频率表逐位不变，把gain从统一规则`1.099065`恢复为OLMo父gain`1.138629`后，
4K/16K/32K分别提高`+5.37/+6.57/+3.15pp`，AUC提高`+5.60pp`。逐任务有反转，
但总曲线一致改善。因此：

- 可迁移的正证据主要属于频率allocation；
- Llama选择出的log中点gain规则不跨checkpoint；
- gain必须作为checkpoint条件配置或单独校准，不能与transition形状捆成通用公式。

## 4. 解释边界

- 支持：同一mix系数和模型内冻结band从S4迁到S8，在OLMo上显著超过BM/MrPro的
  采样网格AUC。
- 不支持：Llama与OLMo使用同一最佳gain、候选任务级支配BM、连续`[L,8L]`无深坑、
  Native保持或跨所有checkpoint通用。
- static YaRN只代表官方频率映射在冻结checkpoint上的零训练control，不代表配套
  训练后的完整YaRN方法。
- 与Llama相同，任务族交换很强；相对宏平均正差不应覆盖单任务失败。

## 5. owner

- 32K数据：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/olmo_s8_core6_32k6_seed20260921/`
- 父gain配对报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/olmo_s8_mix075_g4standard_vs_mrpro_yarn_bm_core6_4k16k32k6.json`
- 统一gain配对报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/olmo_s8_mix075_vs_mrpro_yarn_bm_core6_4k16k32k6.json`
- raw runs：同一根目录`runs/olmo_*_s8_*`。
- 数据准备：`experiments/llama3_60dir_20260911/prepare_planb_panel.py`的显式generic模式。

下一步优先完成Qwen第三checkpoint冻结迁移；Native 4K只需复用/补同prompt一次，不
应触发新的OLMo band或gain sweep。
