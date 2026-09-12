# Pro领域分析：吸收、纠正与当前稿完善

日期：2026-09-12。输入原文及SHA保存在[外部分析](external-reviews/pro-field-map-20260912/index.md)。此文记录主代理对建议的判断和已落实的修改；外部文字中的“必做/不做/半天/1–2天”不是用户当前执行约束。

## 最有价值的认识

这份分析有价值之处，是把当前论文放回不同研究对象之间：模型获得什么频率基、怎样学习使用它、怎样学习频率、怎样在部署时变换，以及怎样改变位置映射。当前稿的独特价值来自固定支持识别、完整位置基、共适应与可用构造的组合。它不需要以一条Cosh曲线包办全部结论，也不应被压缩成只有“allocation matters”的坐标识别。

本轮保留中心命题和章节骨架，增强这些联系；完整答案/EOS、selective-QK、C2、C42等已经恢复的高价值资产继续留有明确位置。

## 建议逐项处理

| Pro建议 | 判断 | 实际处理 |
|---|---|---|
| A1：第一页给研究对象地图 | 采纳，替换已有泛述以控制篇幅 | 引言区分usage、learned frequencies、target-length transformation、positional-map extension；已有文献直接引用，不重复新列一整页名字 |
| 中心句重复到摘要/引言/图注/结论 | 部分采纳 | 引言用“range locates / allocation distributes”作记忆点；摘要原首句已清楚，避免四处逐字重复口号 |
| Figure1概念化标签 | 采纳 | Interior allocation / Paired learning effect / Range interaction / Learned compatibility；保留原始数据和四联结构 |
| A2：Cosh改为one/minimal prior | 采纳对象清晰化，保留构造价值 | 标题改为“An analytic construction from a tail-energy prior”；定理名为“Unique minimizer of the tail-energy prior”。不无依据宣称minimal，也不只剩一个“先验”标签 |
| Green核和H−1分类 | 核验后采纳 | 主文简述Dirichlet–Neumann Green kernel；附录补g(0)=0、g'(1)=0下的能量等式，明确这是给定目标的数学身份 |
| A3：缩减大模型/生成资产 | 不按模型大小撤下 | 将“远处来源使用”和“完整答案生成”各自显式命名，指出OLMo是独立匹配研究；完整EOS和自然QA保留。其他完整曲线和展开仍在附录 |
| A4：b256立即连接大base factorial | 采纳 | §3.1直接指向§5.2的500K/1M、12配置；不是以factorial替代长预算三seed识别 |
| A5：C2 gate同步披露 | 采纳，同时解释它的正面用途 | 主文给OLMo 0.870971 < 0.875，连同Qwen32K代价；保留无Qwen重拟合的长端紧凑描述。C2与C42共同说明低维描述可有用，但单一位移总量不足 |
| A6：宣称不存在universal table | 改为证据支持的正面结论 | Discussion归纳range、learned model与preferred operating point的关系；有限组排序反转不是“任何普适最优表不存在”的全称证明 |
| B：fixed-support learnable-z | 强化既有最高优先级，不恢复短预算默认 | 下一阶段计划保留实际2K、两support、三seed、充分训练，以及同批模型机制干预；现有full-z实现和成熟oracle已登记，不能说从未学z |
| C：Pareto图 | 采纳联合工作点展示，拒绝过度解释 | 全12配置×4非均匀主臂，共48点，每点平均3配对seeds；入附录、主文指向。不拟合跨配置Pareto前沿或bootstrap ellipse，不用离散点证明连续最优边界 |
| Related Work对象矩阵 | 吸收分类，纠正二元标签 | 内部矩阵见下；主文直接点名GRAPE/Selective，保留Möbius在frequency/boundary讨论中。避免把整个本文误标为所有实验均固定support |
| Agent-range不塞主稿 | 作为本稿范围安排采纳 | 当前不新增该线方法/实验；这不是否定其他任务中的独立研究，亦不撤销作者另行授权 |
| 评分6→7预测、理论强于同类、唯一有价值的新实验 | 不作为事实或计划依据 | 新增认识、匹配证据和可读性决定投入，不能由文献类比推出录用分数。MLA独立复评、C42/C2确认和机制干预仍有各自知识价值 |

## 已核对的文献对象

本轮定向核对以下官方论文集/作者论文页面；没有声称完成Pro所列所有文献的全文复审。现有bibliography已有对应条目，本轮主要修改正文连接。

| 文献 | 准确对象与本稿关系 | 本轮依据 |
|---|---|---|
| FMRoPE | base/训练长度与学得frequency band；本稿另识别固定端点的内部位置 | [ICLR2026官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/993dbaec0418a6449090f1debbcb8844-Abstract-Conference.html) |
| Frequency Entropy | 对已学习模型的逐频usage度量与推理干预；不是无权重的位置基几何 | [ICLR2026官方论文集](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract-Conference.html) |
| LeRoPE / AdaRoPE | 学习shared或head-specific频率；本稿的固定端点学习比较有清楚价值 | [LeRoPE](https://arxiv.org/abs/2607.10134)、[AdaRoPE](https://arxiv.org/abs/2607.19363) |
| Data Shapes | 数据依赖距离与频率使用、尺度匹配；启发usage测量，不直接证明我方模型的机制 | [作者论文](https://arxiv.org/abs/2607.07678) |
| GRAPE | 群作用框架，RoPE可作特殊情形；不能仅用“改频率”概括 | [作者论文](https://arxiv.org/abs/2512.07805) |
| Selective RoPE | input-dependent rotation机制；不同于固定安装表 | [作者论文](https://arxiv.org/abs/2511.17388) |
| Möbius RoPE | 通过anti-periodic frequency ladder及head子集设置边界条件，仍可用标准rotary计算；不能简单标成改变旋转算子 | [作者全文§3–4](https://arxiv.org/html/2607.21405v1)明确different constant table；本稿区别是采样端点固定的识别对象，不是“只有我们保留RoPE算子” |

尤其不能把“where frequency range lies / learned usage / positional operator”写成穷尽性互斥分类：同一工作可涉及多种对象，本稿的应用实验也不全是纯fixed-support。

## 新图与数字核验

源：`rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json`。输入含180个预指定主臂run；不混入后续extreme arms。

- 横轴：与匹配Geo的1× NLL差，由原始四位小数PPL取log；这点在caption明示。
- 纵轴：原始full-precision weighted OOD NLL差，权重正比log2(r+1)，r=2/4/8；与12配置原始aggregate逐项误差<1e-12。
- 每点：同配置的3个seed差值平均；48点、四分面、统一坐标，无删点。用原PPL重建weighted NLL的误差<1e-6。
- OOD改善配置数：0.75×/reference/1.25×/Exp分别8/7/10/9；两轴同时改善分别7/6/9/8。
- 配置是设计网格，不是随机抽取的模型总体；图不证明连续前沿、跨配置等价或同一主方向。

复算：`python3 paper-2027/figs/make_m4_tradeoff.py`；可移植输入、CSV和脚本随匿名源码包交付。论文新图用已有结果，不运行模型。

## 交付与验收

本轮论文仍为科学正文9页，总计49页；增加的附录页用于完整工作点图。编译无未定义引用、0pt overfull，匿名/Letter/嵌入字体检查通过。新增数学内容是原目标的Green能量分类，没有新增下游最优性或性能定理。

最终机器可读核验见[pro_field_map_validation.json](pro_field_map_validation.json)。源包独立解压复建、图表重算、最终PDF渲染、当前索引/source hash更新在该记录中分开标记。原两轮PDF-only审稿历史不改，不将本轮完善计作第三轮独立审稿。

恢复点在`internal/local_snapshots/pro_refinement_20260912_013502/`（本机ignored）；核心文档和输入路径采用Git仓库相对路径。不启动模型，不改远端，不提交或推送；保留已有暂存的两个大小写重命名。

## 回查发现并修正的额外事实错误

当前table_pe_dominant原caption仍是125M；历史`rebuttal/rebuttal_playbook.md` PrimaryII与`main:rebuttal/pre_rebuttal/FULL_PAPER_INTEGRITY_AUDIT_20260713.md` E-02均确认约151.9M、15M tokens、midpoint-Geo、headline seed42（learnable-tau为三seed）。本轮改正表头与Geo身份，注明learned inverse frequencies不固定sampled endpoints。已读取`8616af4:experiments/run_128tok_pe_quality.py`的历史runner指针；不把其中旧125M文案覆盖后续实现审计。原PPL数值不变。
