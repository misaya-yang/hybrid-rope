# 固定位置、同一可见集合的两个干预时点

- 日期：2026-09-08；状态：**PREPARED_CODE / RUNTIME_UNQUALIFIED / NOT_RUN**。
- 角色：一次oracle因果判别，不是新频率方法、能力优化器或SOTA验证。
- 当前作者要求整理推送后转公司PC继续；不自动恢复GPU或旧队列。

## 要回答的唯一问题

在同一完整冻结表下，移除明确无关背景后，原长prefill的KV是否足以支持完整
答案，还是必须改变证据表示形成过程？这比重复E/B、attention mass或write norm
多识别一个实际干预时点。7月已有dense-prefill后的稀疏/forced-gold，Round12已有
分解和完整答案前缀margin；不把本次包装成第一次研究这些量。

## 固定对象与资产

优先复用Qwen2.5-1.5B-Instruct和FullLagP2；Native频率SHA为138c99b1…，
实际部署表SHA为ecd0c280…，gain=1.102585782722872，均由
[已存候选](ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json)给出完整值。
基线输出见[配对结果](ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json)。不换成3B，
不增加频率候选，也不重跑Mr/13项矩阵。

已有首个诊断行是`niah_multikey_2_65536_137_0`，65370输入token，prompt文本SHA
`8e27fa4b2b1364f6f8be0173363f630b4b35e09655d155c01f833530c7a082c0`。
本方原输出为错误号码且有EOS，该行不是未见确认集。完整原始token输入和来源
边界仍需从已有EVQ资产读取；本地紧凑回执含输出与已观察record spans，不能把
这些局部spans当成已完成的模板/来源proof。不得按干预结果再挑行或改mask。

保留集合S必须由生成器/来源proof冻结：问题、模板、必要结构及全部真正相关
证据；D才是可删除背景。不把VT的ICL示例归为D，也不删多跳依赖。当前尚未冻结
完整S及反事实配对输入；公司PC应先完成这项CPU准备。若扩成两世界，只使用
预先定义的来源内容变更并保存新身份，原始行保持不变，仍按开发诊断报告。

## 四个计算条件

| 条件 | Prefill输入 | RoPE位置 | 答案读取的历史KV |
| --- | --- | --- | --- |
| O | 原完整prompt | 原位置 | 全部 |
| L | 原完整prompt，先算到倒数第二token | 原位置 | 只保留S的原KV；重新算最后prompt query |
| P | S子序列 | S在原prompt的位置 | 只含S，从头重算 |
| C | 与P完全相同的token | 连续紧凑位置 | 只含S，从头重算 |

固定频率/gain、eval模式、causal顺序及逐token norm/MLP时，P在数学上等价于
原长序列每一层对S query屏蔽D keys后的S状态。按层归纳即可证明；它不等价于
dense prefill后删KV。无sliding、压缩cache、跨token归一化或动态RoPE。

**首token边界：** 首个答案由最后prompt token的输出产生；L必须先处理这行，
不能等第一个生成token入cache后才动手。物理cache长度和原始position IDs分开。

**Decoder边界：** 原greedy继承repetition_penalty=1.1。O/L/P/C均把原完整prompt
加各自已经生成的token交给processor；P/C不能因物理裁切而改变惩罚历史。
不teacher-force正确内容。返回的scores属于各自自由生成轨迹，分歧后不能按
共同gold序列直接索引成teacher-forced margin。保留全词表scores供条件化复查，
终点仍是实际完整输出、正确断言和EOS，不是attention质量。

## 预测与结果到行动

原路径失败、且相同S在C有资格后，再判L/P：

| L | P | 得到的有限结论 |
| --- | --- | --- |
| 成功 | 成功 | 两种干预都足够，不识别唯一机制 |
| 成功 | 失败 | 早期删除损失有用计算，不能假定早删更优 |
| 失败 | 成功 | 该晚期控制不足，改变prefill计算在这组处理之间不可替代 |
| 失败 | 失败 | 两种具体控制不足，不关闭全部attention/静态频率/LoRA路线 |

这是可区分预测：若“原KV可直接读出，只需该晚期去背景”的假设成立，L必须
恢复完整答案；L失败/P成功反驳它。恢复只证明该oracle控制在这些输入上的充分性，
不证明背景数量是唯一原因，不产生无标签的部署mask，也不保证CPT一定修复。
下一步方法设计必须使用实际分支结果，不能提前排新的曲线或训练网格。

## 最低成本执行准备

实现：[position_visibility.py](../../scripts/experiments/scale_transport/position_visibility.py)，
由现有[carrier_ruler_run.py](../../scripts/experiments/scale_transport/carrier_ruler_run.py)
的可选`visibility_replay`字段调用，继续复用模型加载、输入哈希、Flash限制、
原评分、逐行输出和超时监督器；没有新自动调度器。

计划字段：`mode`为O/L/P/C；`layouts_path`保存按row_id索引的`ids_sha256`
（原token的little-endian int32 SHA）与`keep_positions`；`generation_parameters_path`
保存既有解析后的decoder参数。这两个文件必须在原plan的`input_files`中冻结。
诊断不载LoRA、不复用旧QKV-trace的计数假设；单独存实际processed scores。

公司PC继续时先跑已有NumPy回归和纯布局测试，再在工作机完成最短runtime资格：
all-keep时O/L/P/C首token及完整输出一致；O与原generate路径/已有原始行一致。
这项检查未通过时只修实现，不解释科学结果。tiny输入的数值smoke不要求长矩阵。

随后先用C确认保留内容可解，再跑冻结的L/P，参考O能复用就复用。已有该1.5B
64K行单次完整生成约17.05秒；这是费用估算依据，不是新实现实测速率。单行
O/L加短P/C预期分钟量级；应给整个小阶段一个共同硬timeout，不能只依赖两次
forward之间的时钟检查。当前未设新的付费阶段预算或恢复deadline。

## 当前验证边界

已完成纯Python布局/非法mask检查、AST解析；两代理独立复核数学、首query、
decoder历史与代码边界。真实PyTorch/Flash/KV及canonical pytest尚未执行。
这份协议与代码是可审查准备，不能标成已完成实验、已确定根因或已找到最优表。
