# 固定学习bias位置项：构造、风险与第一轮

状态：实现和CPU检查完成，GPU未运行。零训练、无新系数、从首个token起全层应用。

## 机制与公式

Qwen2的归一化前仿射Q/K为q=u+bq、k=v+bk，投影后没有Q/K norm。
取相对位移d=key_position−query_position，M/B为原MrPro/BM旋转，gain同为a。

s_H = a²/√D [ qᵀ R_B(d) k + bqᵀ (R_M(d)−R_B(d)) bk ]。

它只把纯bias–bias项从BM换为Mr版本；内容–内容、两个内容–bias交叉项仍为BM。
这些bias在Native相位下训练，所以不能称Mr版本就是原始学习位置先验。
这是明确的相对logit bias修正，不是完整Q/K上的纯正交旋转，也不声称zero-key保持。
OLMo2没有这些投影bias，实现直接走原BM attention，公式和执行均退化为原BM；
因此复用其已有全部BM结果，不重复跑OLMo来制造新收益。

动机是：Qwen上的BM全中频压缩也改变了内容无关的这一位置项。真实3B六层权重
给出中频修正的Cauchy上界，head中位约0.97–2.92、最大约5.15 logits（尚未乘a²）。
它说明可能不可忽略，不等于实际修正幅度或失败原因。原生8篇校准文本中，中频
均值与bias方向的head中位cos约Q .83–.91、K .86–.98，未发现均值被近乎抵消的
普遍情况；这是有限输入的描述，不保证每个token不存在抵消。

最强反例：u=−(1−ε)bq、v=−(1−ε)bk时，完整q/k均非零但原score仅ε² f_B，
新修正f_M−f_B仍为常量级。没有相对完整Q/K模长的统一小扰动保证。若它不改善
固定案例，就不能继续把bias范数上界当原因证据。

## Flash实现

Qwen3B两表仅16个中频槽不同，取其32个实坐标：
Q_aug = [Q_B, a R_M(qpos)bq_mid, a R_B(qpos)bq_mid]；
K_aug = [K_B, a R_M(kpos)bk_mid, −a R_B(kpos)bk_mid]；V_aug=[V,0,0]。
总宽192，scale仍1/√128。一个共同softmax；输出逐head取前128维再经过W_O。
原KV缓存仍128维，位置项在attention时构造。只支持连续从零开始的未padding序列。
CPU独立复数相对旋转对照包含GQA、128K位置和非单位gain；检查实际tiny Qwen的
cache形状/逐步解码/注册接口恢复，以及无bias OLMo完全相同的执行路径。
这些不是GPU能力证据。GPU保持Flash-only，MLP采用已验证4096token分块。

## 冻结首轮输入

两条负例：niah_multikey_2_131072_1、vt_131072_1。
两条长端既有BM正例：niah_multiquery_131072_3、vt_131072_0。
两条短端既有BM正例：vt_32768_1、fwe_32768_1。

首轮另外只在两负例运行192维零补齐BM，量化内核形状变化；保留原128维BM及
MrPro存档，不能把数值差异计成位置项收益。候选六例不用于最终均值宣称。
两负例均不改善时，停止此候选，不扫描层、修正幅度或改锚点。若有实质恢复，
结合正例保留情况决定完整36条筛查；看整体得失，不设每个子项必胜的硬门槛。
候选通过后换Qwen7B并扩任务。OLMo分支按恒等退化复用原BM证据，不冒充新测量。
