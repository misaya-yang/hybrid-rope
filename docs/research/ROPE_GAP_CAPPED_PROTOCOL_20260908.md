# Gap-capped MrRoPE：由既有最大radix扩张约束推导最小原生改动

状态：GPU已完成，结果为负；[完成结果](ROPE_GAP_CAPPED_RESULT_20260908.md)。
下文保留GPU输出前协议。目标是README的零训练真实长任务收益。
起点为[MrRoPE原文§3.2](https://arxiv.org/html/2601.22181v1#S3.SS2)：
渐增radix扩张用于保护较高频率。以下是本次从该设计取舍出发的独立构造，
不声称原文提出了本方法，也不以几何最优冒充能力最优。

## 缺口与有条件的解

记中段宽N、累计指数m_q=sum(i<=q)epsilon_i。MrPro采用
epsilon_i=2i/[N(N+1)]，其最大单步额外log-gap为c log S，c=2/(N+1)。
它的算术序列并不是在这一原有最大扩张约束下改动较高频率最少的序列。

保留其高/低频、总scale、gain与原最大单步扩张，约束为
0<=epsilon_i<=c、sum epsilon_i=1。对任意q，剩余N-q个槽最多承担(N-q)c，故

    m_q >= max(0, 1-(N-q)c).

取m*_q=max(0,1-2(N-q)/(N+1))，它的增量满足上述全部约束，并同时取到每个
累计量的下界。因此该累计向量是唯一的逐分量最小值；对累计改动严格递增的
目标均最优。它也保持增量非递减，不是放弃MrPro的progressive方向。
没有学习系数、选择cap或扫描切分点；c来自未调参MrPro的最大增量。

部署频率为omega'_j=omega_j S^(-m*_{j-l})，两端直接保持Native/PI；S4，gain
1+.1ln4。Qwen N17、l23/h40：前8个中段累计量为0，余9步各增加1/9。
OLMo N18、l14/h32使用同一公式，非零边界随几何推导，不转移Qwen槽号。

**独立复核补充的精确身份：** m*=clip((q-(N-1)/2)/((N+1)/2),0,1)，因此是
缩窄过渡区的MrUni：Qwen的有效区间31→40；OLMo22.5→32。
潜在贡献是从既有cap推导分区选择及实测收益，不是新函数族或新RoPE算子。
详见[独立构造分析](ROPE_MRPRO_BM_CONSTRUCTION_ANALYSIS_20260908.md)。

## 为什么值得测，以及不能推出什么

逐槽m*<=m_Mr，所以Native>=omega*>=omega_Mr。相对Native的频率位移不增，
最大相邻log-gap不增，慢端目标相位范围与MrPro相同。对于一旋转平面和连续
距离区间[0,D]，最坏算子差恰为

    sup ||R(omega' delta)-R(omega delta)||_2
      = 2 sin(min(D |omega'-omega|, pi)/2).

因此本候选的这项最坏扰动不大于MrPro。这个结论与实际某一delta下的误差大小
不同：sin的周期性允许逐点误差反转；多槽相干、softmax、V、跨层状态和生成
决策也不由该界排序。约束选择是保守设计判断，不是语言模型唯一正确约束。

正向预测：MrPro保护较高频率的设计仍有可用余量，进一步减小这些频率的原生
扰动，可能保护128K背景中绑定/追踪的状态形成。BM已在3B显示全中段更压缩
并非统一受益，因此值得一次反方向的、明确约束下的构造比较。这不是从两例
cross-cache必然推出改频方向，而是结合MrPro方法原则提出可证伪预测。

最强竞争解释：更晚、更集中的压缩使更多相邻槽同时达到原最大gap；粗糙度
反而更大，原有中段联合远距关系可能被破坏。若长端不改善，该构造不晋级，
不据此宣判所有原生保护原则无效，也不随后盲扫cap。

与旧FullLagP2、C1、Carrier、Native-sector及BM的构造/数组已区分；关键词及
现有相关owner未发现此相同cap下逐分量极小构造的已测结果。尚未完成全面
文献新颖性判定，不能称为已证明新颖。

## 实现与唯一比较

[代码](../../scripts/lib/rope/gap_capped.py)、[两模型完整数组](ROPE_GAP_CAPPED_CANDIDATE_20260908.json)。
独立线性规划在N2/3/7/17/18的全部内部q核对闭式；两模型FP32频率、外频、gain、
最大gap与方向检查通过。测试2项通过。Qwen频率SHA：
`d6708ca473e2d1695f3d9a0299c2a93701da66f6bd76a623cf8eabc0ef8c25fd`。

只运行Qwen2.5-3B-Instruct本候选，复用`prepared_qwen3_01`的36条六任务输入、
官方逐任务预算、greedy与EOS；完整MrPro/BM输出已归档，**不重跑基线**。
运行器复用`layer_screen.py`的全表相同分支，实际走stock attention，无层hook、
无缓存移植、无oracle标签参与构造。旧uniform资格已通过，不重复资格回放。

主终点：128K六任务等权均分，32K另报，逐行完整生成/EOS保留。超过MrPro
则冻结同一公式转到其他模型/任务；确实缺可比MrPro基线才运行一次并存档。
当前36条是开发子集，不将与论文完整13项53.2的差值作为本次收益。
预计本候选约15分钟，按实际完成记录成本，不为估计时间截断任务。
