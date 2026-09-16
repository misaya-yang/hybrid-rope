# 十篇位置编码论文的审稿经验：哪些真正帮助当前论文

日期：2026-09-16。来源为用户提供的[完整审稿合集](reassessment_20260916/ALL_REVIEWS.md)。
阅读范围包含十篇的官方评审、归档中的AC总结及与关切解决有关的讨论/回复。
[来源身份](reassessment_20260916/sources.json)保留原文件哈希。
这是审稿意见的适用性分析，不把评审意见自动当成正确理论。

## 结论

最值得借鉴的共同标准是：**清楚的问题、能支撑设计的解释、与主张匹配的对照和任务结果、
以及正文可独立理解的论证。** 更长附录、更多模型或更多形式化本身都不能替代这些。
我们已有固定支持、等位移、跨模型任务和自然QA证据；下一轮应把这组证据组织得更好，
优先补原生独立确认和直接基线，而不是按十篇论文的所有问题扩大实验矩阵。

归档包含ICLR与ICML，不直接换算评分；ICML2025的1–5量表不同于ICLR。
这些已接收论文是用户选择的案例，不是录取概率样本。
本合集不包含MrRoPE；其审稿经验来自另行提供的材料，不能混称这十篇的一员。

## 十篇逐项映射

| 论文 / 可定位评审 | 核心追问 | 后续处理或评价 | 对我们的具体用途 |
|---|---|---|---|
| **Round and Round We Go!**：[forum](https://openreview.net/forum?id=GtvuNrk58a)，R38mxyPD8S、xIZr2627Lb；AC iSH3ZBnZXn | 单pair结论能否推广、partial RoPE区分、PPL是否转化为任务质量、频段角色是否真被干预识别 | AC认可结构认识；明确未把decoder-only与其他PE覆盖不足视为必须修复；跨模型和消融增强说服力 | 保留完整pair/内容坐标区分；GLM partial RoPE安装要准确；用已有RULER与自然QA支撑实用性，别给频段固定语义标签 |
| **Wavelet-based Positional Representation**：[forum](https://openreview.net/forum?id=OhauMUNW8T)，yM57GeN1C3、E6duFTc7ts；AC 2LXw0S6n40 | 数学名称是否满足定义、分析为什么导出该构造、为何选这些尺度、仅PPL是否充分 | AC记录理论澄清和追加评测；剩余更大模型/更多基线建议不都视为关键 | Cosh密度目标准确命名；说明结构偏好如何引出构造，避免把代理写成完整任务损失 |
| **Eliminating Position Bias / PINE**：[forum](https://openreview.net/forum?id=fvkElsJOsN)；CnClJSl4Y2、l4cOSOy8Jy；AC gHkEXCVYzb | “不变性”对象、图示与算法不清、对照是否同预算、简单排序是否已足够 | 评审澄清后加分；AC仍建议清楚算法描述，认可限定输入场景内的价值 | 固定权重/固定状态、z/π、幅度作用必须易懂；简单安装步骤比额外形式化有用；不增加无关文档排序实验 |
| **A Formal Framework for Understanding Length Generalization**：[forum](https://openreview.net/forum?id=U49N5V51rU)，8x83XrtdWm、1ehrSNl582；AC FYK2JNvCmK | 72页、62页附录导致正文像摘要；理论假设与真实训练有什么联系；动机埋在FAQ | AC认可条件性理论贡献，记录引言澄清和针对性检验 | 正文要独立说清；保留29页现稿方向，必要细节加回也不追求理论数量；参考假设放在相关结论附近 |
| **Fourier Position Embedding / FoPE**：[forum](https://openreview.net/forum?id=ZfDNDkg7Dh)，qjqfqNTfEc、jILJLHuBRr、YhCxrdG5sx | 复数/实旋转表达、undertrained定义、阈值事前如何选、计算成本与基线 | qjqfqNTfEc在理解复数与二维旋转对应后撤回相关误解；另一评审因追加比较更新判断 | 审稿问题先判断对错，不能把误读写成论文缺陷；公开规则与安装公式直接展示；不用动态拟合代替解释 |
| **LongRoPE2**：[forum](https://openreview.net/forum?id=jwMjzGpzi4)，YBGS9h8WSG、GscOWQBEYB | RULER优势能否代表广泛长文能力、选哪些任务、共享训练技巧是否真新颖、机制图如何获得 | 归档保留一位评审rebuttal后仍认为图示解释不足；肯定方法价值与保留关切并存 | Qwen/OLMo自然QA具有独立价值；所有任务按预设协议呈现；复用技术归因清楚；不把检索/PPL/QA合成总冠军分数 |
| **LieRE**：[forum](https://openreview.net/forum?id=yMJAYbGcCc)，Rxw4EVd8Pl、ynusNde6cA | 广义设计空间带来什么实际收益、超参数是否可预测、图注/空白/未定义引用影响阅读 | 评审对写作有分歧，部分承认强分辨率迁移；不可把某位意见当作AC总判断 | z的贡献靠干预和结果，不靠坐标重写；现有CPU规则与闭式解要便于使用；下一版保留排版验收 |
| **TAPE / Contextualized Equivariant PE**：[forum](https://openreview.net/forum?id=wgGC1N4rKy)，u9A5LTtbuN、u4v3Yd8gXc | 理论性质为何支撑方法、多组件归因、首次等变性主张、推理成本 | 也有评审认为实验充分；有的追加要求是可选增强而非共同缺陷 | 三构造分别说明职责；等位移C回答特定形状问题，不强称唯一机制；固定表不增算子与墙钟零开销分开 |
| **Group Representational Position Encoding / GRAPE**：[forum](https://openreview.net/forum?id=itoNJ3gJl2)；WJod8GcscH、F0BCGcsCSA；AC BQQVON5PDk | 复杂框架解决什么问题、理论主分支是否有对应实验、只有loss曲线不足 | AC认为追加下游任务加强接收理由；有评审提醒重大新增结果可能不纳入原稿评价 | 每个主文概念必须回答一个问题并有证据；重要自然任务尽量在提交稿呈现，不把rebuttal当补全原稿的保障 |
| **Selective RoPE**：[forum](https://openreview.net/forum?id=AQo1SEElNb)；6H5GN7zbKm、QLlAawBCMK；AC WxvAn5YB8H | RFF近似/归一化假设与真实softmax的关系、缺原生RoPE基线、理论与应用架构不同、训练稳定性 | AC认可补齐NoPE/RoPE与softmax实验；动态训练稳定性仍是该方法的具体问题 | 将参考对象和实测模型分开；直接基线有价值；其可学习动态旋转的不稳定性不自动成为静态TailSpline的负担 |

ICML四篇在所给归档中没有与ICLR格式相同的AC总结；上述“处理”只引用实际出现的评审更新，
不虚构最终AC理由。归档中的评审别名可能在AC摘要中不同，定位优先用note/review ID。

## 共性问题如何分流

### 已有证据可以直接解决

- **只有PPL？**已有完整RULER任务族、OLMo和Qwen自然QA、Llama自然任务比较。
- **只是改范围？**151.9M固定支持和成熟模型固定支持控制。
- **只是总位移？**BM–Uni与Llama clean T–C；后者不必重跑。
- **几何量就是任务能力？**完整pair分析及T/P、T/C rank与任务结果分离。
- **只有一种模型结构？**Llama/OLMo/Qwen已完成；GLM partial RoPE在执行，不预写胜负。
- **Cosh学习与零训练混用？**按每个协议的权重更新、网格和端点分别陈述。

### 真正值得准备的新工作

1. 取得正在进行的GLM与Qwen官方YaRN的正式配对报告，准确给出作用范围。
2. NCP固定规则的新来源原生Full-13确认，预先保留VT及QA分解。
3. Llama clean32K官方静态YaRN单臂：同输入复用已有T/P，不用跨论文胜负传递。
4. 若需要把“内容利用”升为机制主张，复用288题反事实面板；否则先保留条件解释。

### 不继承为必做清单

70B、所有PE、视觉/音频、重训LeRoPE/LongRoPE、完整head搜索、动态旋转稳定性消融、
大范围band/强度搜索。它们只有在我们实际提出相应主张时才构成具体需求。

## 下一版正文阅读检查

- 第一页能否说清具体科学问题和贡献，而不依赖读者知道全部历史方法？
- 理论对象、构造目标和真实任务之间是否逐步衔接？每个证明的假设是否就地可见？
- TailSpline与NCP分别解决哪个部署场景？Cosh支持的是哪种实验问题？
- 主表能否直接看清模型、倍率、长度、方法和指标？
- 一张图是否回答一个关键问题？完整任务表和复现细节是否有唯一清楚的附录位置？
- 摘要无数字、第一页无图，措辞积极准确，正文九页独立成立。

该清单用于编辑和证据安排，不给下一轮PDF审稿预设评分，不把每条建议自动写成不足。
