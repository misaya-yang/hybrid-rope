# Hybrid-RoPE关键实验罗盘

更新：2026-09-14。用途：只保留能够决定论文主张、方法去留和下一步投入的实验；不是
历史运行目录，也不把计划、CPU定理或开发proxy写成模型结果。

## 中心目标

论文已经成立的核心不是“某条曲线永远最好”，而是：**在端点与支持固定时，RoPE内部
指数分配是独立因果变量；模型会与频率槽形成共适应；这一自由度能够被解析构造用于
学习与冻结部署。** 当前新增方法目标进一步收窄为：**零训练的exact TailSpline能否稳定
胜过exact MrRoPE-Pro。** YaRN比较与机制归因后置。

## 罗盘

### 北：已经成立的论文骨架，不重跑

| 证据 | 最关键读数/发现 | 支持什么 | 不支持什么 | 动作 |
|---|---|---|---|---|
| A01 固定支持151.9M三seed | 每个seed只改30个内部频率；固定范围下1024/2048 tail NLL差为`−0.176/−0.146`，三seed同向；256窗内有约`+2.65%` PPL代价 | allocation在端点固定后仍是可识别因果变量，并产生窗内—窗外交换 | Cosh普适最优、成熟模型能力 | 永久复用，不重训 |
| A03–A06 几何与crossing | 完整sin/cos位置基、有限窗方向、权重×表交叉及同谱补偿边界均已证明/实测 | “几何供给”与“权重学会使用”必须分开，checkpoint compatibility真实存在 | 静态几何量可直接选出赢家 | 用于理论和解释，不再做proxy筛表 |
| A09 MLA三seed | 432M稀缺旋转通道下，EVQ在16K及更长PPL曲线明显低于GEO；三seed保留 | 分配在训练期与受限rotary budget中有实用后果 | 旧cache是严格独立holdout、冻结部署方法胜负 | 作为学习期旗舰证据，不与当前冻结结果拼分数 |
| A12 完整答案+EOS | OLMo匹配读出中完整答案加立即EOS：8K约`18%→98%`，16K `0%→60%` | 远距信息利用可以落实为完整可终止输出 | 零长位置暴露外推；多seed结论 | 保留raw lineage，不重训 |
| A14 BM自然QA | 631条自然QA中BM约`25.44%`，MrPro约`21.62%` | 冻结成熟checkpoint上allocation会改变自然输出；MrPro并非处处占优 | BM跨模型普适胜出 | 作为重要反例与任务证据复用 |

### 东：能够形成方法贡献的现有结果

| 证据 | 状态与结果 | 当前判决 | 下一动作 |
|---|---|---|---|
| A15 C2紧凑profile | Qwen 64K/128K保留了64维transport的主要行为，但Native 32K由`0.82`降至`0.7125`，注册双门失败 | 证明低维描述可能，但不是合格统一部署法 | 不再调C2；只作“可压缩但有代价”证据 |
| A16 C42/C42V24 | 同总位移、同质心受控对在350行开发RULER为`+10.73pp`，16文档NLL同向`−0.1109` | 总剂量与质心不充分；高阶shape是真变量 | 尚无独立任务确认，不拿开发赢家作新主方法 |
| A31 旧mix075 Llama S4 | Core-6开发AUC `80.95`，BM `74.68`、MrPro `73.15`；Native 8K有代价 | 为TailSpline提供强开发先验 | 0.75、band和gain有开发暴露；不能替代exact TailSpline |
| A39 exact TailSpline Llama S4 | Full-13 `0.7880 vs 0.7560`，差`+3.20pp`、95%区间`[+0.65,+5.79]pp`；NIAH `+3.75pp`；PPL AUC `−0.00449`，3/3方向通过 | **当前第一次以完整经典合同确认TailSpline整体胜MrPro** | MrPro raw永久复用；正在做OLMo两臂跨模型确认 |
| A40 OLMo exact TailSpline S4 | Full-13 `0.6660 vs 0.1736`，差`+49.23pp`、95%区间`[+45.34,+53.04]pp`；NIAH `+64.45pp`；PPL AUC `−3.853`；13任务差全部为正 | **第二checkpoint前瞻确认完成，方法方向与Llama一致** | 冻结TailSpline，不追加YaRN/BM/新曲线 |

### 南：已经否决或必须停止扩张的路线

| 路线 | 否决证据 | 以后怎么处理 |
|---|---|---|
| fixed-u倍率迁移 | A36：OLMo S4→S8同prompt对照显著差于fixed-m | 关闭，不换倍率或模型继续救 |
| OOD/SEP、Fisher、局部margin、coherence等proxy选表 | 多次与完整任务反转；局部指标不能预测Full-13 | 只可作事后诊断，不再生成候选 |
| 由漂亮roughness泛函宣称全局最优 | A37：one-sided唯一解是TailSpline，symmetric唯一解是BM；CPU不能选择边界先验 | 定理只写条件唯一性，任务偏好交给冻结模型实验 |
| 开发面板继续调系数/band/gain | mix075及多表结果存在选择暴露与任务集中 | 不再调`0.75`、band、gain、lambda或按失败任务修表 |
| YaRN/MrPro/Fast/Slow“四格等剂量”原案 | 只有YaRN与MrPro等剂量；Llama n=17时Fast/Slow总指数为`6.02651/6.64016`，相对共同`6.33333`为`±0.30682` | 不能声称独立识别前段、后段和剂量；YaRN后置，当前不跑 |

### 西：尚未关闭的四个问题

1. **跨checkpoint方法性。** Llama与OLMo已经同向胜MrPro；这支持两个模型族上的方法性，
   但OLMo历史方向先验和不同tokenizer面板意味着它还不是完全独立盲测。
2. **机制解混。** TailSpline与MrPro同时改变`sum(m)`与细形状。即使两模型都胜，也只能先说
   方法结果；one-sided tail landing是否为原因仍需同剂量控制。
3. **独立性。** 当前Full-13是完整任务覆盖、每格10条，但方法来自历史开发。跨模型确认和
   后续独立prompt/checkpoint的职责不同，不能把更多同面板行自动称为独立泛化。
4. **YaRN实用对照。** 作者已决定先赢MrPro，YaRN后做。它不阻塞TailSpline方法结论，也不能
   用历史非匹配配置拼入当前主表。

## 当前唯一决策树

- **当前已经发生：Llama与OLMo都是3/3胜MrPro。** 冻结TailSpline，不再搜索曲线；下一步
  只能在“真正独立确认”与“最小同剂量机制控制”中择一，先整理方法论文主表。
- **若以后独立确认反转：** 将方法主张缩为两项已测合同，不调系数、band或gain挽救。

无论哪种结果，都不自动启动YaRN四格、OLMo调参、Qwen复测、训练或新的理论曲线。

## 直接入口

- [当前研究入口](index.md)
- [Llama exact TailSpline主结果](TAILSPLINE_LLAMA_CLASSIC_RESULT_20260914.md)
- [OLMo exact TailSpline跨模型结果](TAILSPLINE_OLMO_CLASSIC_RESULT_20260914.md)
- [TailSpline方法与统一评测合同](TAILSPLINE_ROPE_METHOD_AND_UNIFIED_EVAL_20260914.md)
- [论文证据索引](../../../paper-2027/research/evidence/index.md)
- [YaRN–MrPro等剂量CPU审计](MRROPE_YARN_EQUAL_DOSE_PRINCIPLE_AUDIT_20260914.md)
