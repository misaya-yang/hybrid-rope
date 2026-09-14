# Llama S8 冻结规则迁移结果

2026-09-13。**状态：本轮已收束。S4→S8共同32K桥接、S8候选8K/32K/64K、同口径
MrPro/static-YaRN/BM 64K和Native 8K已完成；16K曲线及matched-gain机制臂未运行，
不由旧runner结果替代，也不列为自动队列。**

## 1. 问题与冻结构造

模型为`Meta-Llama-3-8B-Instruct`，Native长度`L=8192`。从已确认的S4候选冻结：

- band `[16,34]`；
- exponent allocation `m=0.25 m_BM+0.75 m_frontloaded`；`0.75`来自开发冻结，不是
  理论唯一解；
- full `/S` slow endpoint；
- 一张表用于全部层和全部输入长度；
- S8规则gain `sqrt(1+0.1 ln 8)=1.0990651274`；共同长度桥接另固定S4中点gain
  `1.0670658068`。

没有依据S8结果重新选band或混合权重。逐槽审计得到S4/S8 exponent最大差
`5.80e-8`，`nu_8/nu_4=2^{-m}`观测相对误差最大`1.21e-7`；各自目标端相位倍率
范围为`[1,2]`。因此这是冻结 allocation 的倍率迁移，不是重新拟合一张S8表。

面板为Core-6：`niah_single_2`、`niah_multikey_2`、`niah_multiquery`、`vt`、`fwe`、
`qa_1`，每任务/长度6行。主指标为完整生成后的RULER official contains、task-equal
宏平均；EOS/cap同时记录。

## 2. 共同32K桥接

同一批36个prompts：

| 表 | gain | 32K official macro |
|---|---:|---:|
| S4 mix075 `[16,34]` | 1.067066 | 71.67 |
| S8，同m、固定S4 gain | 1.067066 | 73.94 |
| S8，同m、S8规则gain | 1.099065 | **74.86** |

S8-fixed-g减S4为`+2.27pp`，说明更强频率缩放在共同绝对长度上没有先行崩溃；
S8-rule再加`+0.93pp`。逐任务仍高度异质：从S4到S8-rule，QA为`+33.33pp`、
MK2为`+16.67pp`、VT为`+6.67pp`，FWE为`-33.33pp`、multiquery为`-4.17pp`、
single持平。不能把宏平均增益解释成所有功能共同改善。

## 3. 64K端点：候选与三基线

| 方法 | official macro | EOS | cap |
|---|---:|---:|---:|
| **S8 mix075 `[16,34]` + rule gain** | **57.45** | 91.67 | 8.33 |
| canonical MrPro S8 | 52.50 | 94.44 | 5.56 |
| BM S8 | **57.50** | 100.00 | 0.00 |
| static YaRN S8（零训练频率图control） | 0.00 | 69.44 | 30.56 |

候选减MrPro为`+4.95pp`；36个同prompt、任务固定、任务内配对bootstrap的95%区间为
`[+2.22,+8.06]pp`。候选减BM为`-0.05pp`，区间`[-9.21,+8.70]pp`，准确结论是
宏平均不可区分而不是候选胜出；候选减static YaRN为`+57.45pp`，区间
`[+50.42,+64.49]pp`。static YaRN没有配套训练，只是官方频率方程在冻结checkpoint
上的control，不能冒充完整YaRN方法。正重采样比例不是后验概率或自动门禁。

| task | 候选 | MrPro | 差值 |
|---|---:|---:|---:|
| niah_single_2 | 100.00 | 100.00 | 0.00 |
| niah_multikey_2 | 0.00 | 0.00 | 0.00 |
| niah_multiquery | 95.83 | 91.67 | +4.17 |
| vt | 93.33 | 73.33 | **+20.00** |
| fwe | 5.56 | 0.00 | +5.56 |
| qa_1 | 50.00 | 50.00 | 0.00 |

这是一项当前有效的8×端点正结果：优势主要来自tracking，并有少量multiquery/FWE
增益；不是passkey或QA单项驱动。BM则由FWE/MK2支撑、VT只有20，候选与BM是明显
任务族Pareto交换。它还不是完整区间比较，因为三基线的短端曲线没有运行；现有
8/32/64K候选点可以定义一个稀疏采样曲线，但不能与只有64K的基线拼成AUC，更不能
声称连续区间无深坑。

## 4. Native 8K保留

候选与Native在完全相同的36个prompts上为`76.94`对`92.78`，差`-15.83pp`，配对
95%区间`[-19.12,-12.59]pp`。候选FWE为0，Native对应子集为83.33；其余主要损伤
包括multiquery与VT，QA点估计反而更高。因此S8候选没有满足Native保留，且不能用
64K对MrPro的正结果抵消这一事实。

## 5. 当前判定边界

- 支持：冻结S4 allocation迁移到S8后，在共同32K不退化，并在64K同口径小面板上
  超过canonical MrPro。
- 不支持：跨模型通用、连续`[L,8L]`无深坑、Native保持、完整RULER SOTA，或把全部
  组合收益归因于transition形状。
- FWE在32K随S8配置下降、64K也很弱，说明此前Native/aggregation缺口并未消失。
- 64K候选与BM宏平均不可区分且任务族相反；本轮停止前没有形成匹配区间AUC，不能
  由端点外推。
- 旧64K runner在当前36个prompts中只重合24个，且已有parity差异；只作历史旁证，
  不进入正式差值。

## 6. 可复核实现与owner

- 构表：`experiments/fixed_rope_three_interfaces_20260913/tables.py`
- 倍率恒等式审计：`experiments/fixed_rope_three_interfaces_20260913/scale_transfer_audit.py`
- 单长度配对报告：`experiments/fixed_rope_three_interfaces_20260913/matched_point_report.py`
- 生成入口：`experiments/olmo_recovery_20260912/recovery_v2_eval.py`
- 面板：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/llama_low108/`
- S4→S8审计：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s4_to_s8_mix075_scale_transfer_audit.json`
- MrPro配对报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s8_mix075_vs_mrpro_core6_64k6.json`
- 三基线端点报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s8_mix075_vs_mrpro_yarn_bm_core6_64k6.json`
- Native 8K报告：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/reports/llama_s8_mix075_vs_native_core6_8k6.json`
- raw runs位于同一根目录`runs/llama_s8_*`与`runs/llama_mrpro_s8_core6_64k6/`。

若未来因新benchmark重新开启，应继续使用本owner并复用现有基线；当前未完成项是
证据边界，不是已经授权的待跑队列。
