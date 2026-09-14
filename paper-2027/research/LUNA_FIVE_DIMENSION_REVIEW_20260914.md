# 六个Luna代理：五维对照审查与本轮改稿

2026-09-14。用户要求以提供的七篇论文Markdown为对照，独立核查当前稿件，DeepSeek意见只作待核实输入。六个代理均使用gpt-5.6-luna、只读审查；五个分管维度，第六个交叉检查。比较集为MrRoPE、Decoupling、GRAPE、Deconstructing、Selective RoPE、RePo、PPE。目录中的旧Spectral Budget稿是本项目旧稿，未当外部对照。不从这组样本推断会议统一证据门槛或录用概率。

## 五维综合判断

| 维度 | 有依据的缺陷 | 本轮处置 |
|---|---|---|
| 贡献与定位 | 范围/分配的独立控制与行为交互容易混淆；BM相对MrRoPE的新增内容不够明确 | 使用independently controllable；明确BM在mixed-radix坐标内替换增量形状；相关工作区分operator、head行为与table/coordinate兼容性。 |
| 理论 | overlap几何到Cosh密度目标仍有设计先验，不能写成已经从任务目标导出 | 明确geometry-inspired、weight-independent prior；保留精确定理和实证用途。加入principal-angle和frame-potential谱系，并说明换频率保留RoPE相对组合与范数性质。 |
| 实验 | 训练期固定支持证据主要来自151M；更大冻结模型已有固定支持干预，两者不能混称。C42细形状结果仍是开发证据 | 保留各协议身份、种子和关键Native/模型代价；不新增训练或capture。只有提出更强迁移或中介claim才需要相应新实验。 |
| 叙事与图表 | 第一页术语密集，构造介绍顺序与正文不一致，图1标题混淆固定范围和range retargeting | 简化开篇，先讲同端点干预与发现；按正文顺序介绍BM和Cosh；图1明确三类控制及tail NLL方向，三项贡献放在一起。 |
| 复现与证据表达 | SUPPLEMENT_README仍指旧标题/ZIP和当前包不存在的训练命令 | 改为当前源码包说明，列出真实可运行CPU命令；将README纳入canonical源码包并在空目录解包执行核验。 |

## 没有照单采纳的意见

- 当前已有几何热图和有效秩曲线，不为“缺几何图”的旧批评重复制作。
- 活跃TeX引用图有49个引用键，main.bbl也只有49项；83条bib记录中的34条未使用记录不出现在PDF。交叉代理按更宽文件范围得到51/32，不适用于当前稿的实际引用图。保留可复用bib库，不把它当审稿可见缺陷。
- “没有大模型固定支持干预”不成立；OLMo/Qwen冻结模型控制已在正文。151M训练识别的规模范围另行准确表达。
- 不将额外attention诊断、larger scratch pair或同总量新确认自动升级为必要工作。新构造不读取权重或激活；当前论文可以围绕已有识别、理论和验证成立其声明范围。
- 范围重定向反转是兼容性的关键发现，BM跨模型差异限制其实际收益范围，均保留。C2拟合细节和8B LoRA来源使用/读出分歧仍完整在附录；主文以相关问题指向，不按正负筛选结果。
- 两个代理建议更多限定句；综合后采用具体对象名称和协议位置，避免重新堆叠防御性叙事。没有加入新的rank中介或checkpoint功能拟合claim。

## 本轮实际修改

重写摘要、引言和结论；精简主文M4细项、C2拟合探索和LoRA读出探索；区间条件最优细节回到已有附录。加强经典几何引用与NTK-aware定位，补充可核实的核心训练token预算。章节仍为识别/几何、学得兼容、冻结部署、学习期构造；未按某个模型评语机械换序。

全部原始实验数字、证明主体和结果附录保留；未赋予TailSpline未完成的任务结果。改前TeX/PDF/源码包已在本机临时备份，不作为跨机器依赖。

## 验证与交付

- pdflatex/bibtex构建：9页科学正文、52页总量，引用从11页开始；未定义引用0，overfull hbox 0，字体嵌入与匿名检查通过。
- 逐页检查9页主文，保留现有三幅主图和学习期表；修复了贡献列表跨页分散的问题。
- 当前`exponent-allocation-source.zip`含81个文件；PDF与工作区输出一致，ZIP及SHA256SUMS核验通过。
- 在空临时目录解包后，`verify_explicit_geometry.py`、`verify_profile_diagnostics.py`、`verify_recovered_assets.py`、`verify_routing_schedule.py`全部通过。
- 本轮文档导航检查保持原有两个历史证据哈希差异；仅更新本轮修改的04_construction.tex和03_compatibility.tex登记哈希，不改旧结果hash以消除告警。

[当前PDF](../main.pdf)、[当前源码包](../exponent-allocation-source.zip)、[修订目标](../REVISION_BRIEF.md)。

## 主代理独立复核与引用补正

独立通读当前正文并核对相关附录后修正三处：摘要末句的并列语法；结论不再把已有可复用构造整体写成未来目标，而将未解问题定位到边界/终点/过渡的联合选择；附录H.4撤去依赖Q/K capture和replay选表的未来方案，保留符号化attention频率交互推导。其内容更新同步到05_threeband.tex登记哈希，实验数字不变。

按用户提醒追溯YaRN的原始参考链接，NTK-aware和Dynamic NTK分别直接引用bloc97(2023)与emozilla(2023)公开原帖，均使用misc/Reddit来源身份。此后活跃引用由49增至51；正文9页、总量52页，编译零未定义引用、零overfull。源码包已同步。
