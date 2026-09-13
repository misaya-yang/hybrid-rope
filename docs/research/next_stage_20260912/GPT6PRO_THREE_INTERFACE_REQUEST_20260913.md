# 固定RoPE表三接口研究请求

请只结合唯一附件《Hybrid-RoPE 固定表三接口：全证据、理论边界与夜间实验简报》，
独立判断下面的问题。该附件已经包含本任务所需的实验事实、数学推导、理论边界和
执行条件；不需要第二个附件。

## 背景

我们研究一个已经预训练好的RoPE模型。Native窗口为`L`，部署horizon为`S L`；
整个会话和全部层只能安装一张固定Native-relative频率表，不能随输入长度动态换表。
候选表可写为`omega'_k=omega^N_k S^{-m_k}`。当前证据显示band、慢端压缩深度、
transition内部增量分配及gain存在强交互；Band已有多模型、多倍率小面板和一个
OLMo Core-6 mini，transition已有BM/C42等证据，而tail depth尚未做真实任务干预。

当前还有三组应被独立审计、不能预设为真的理论候选：slow边界约在Native窗口一圈
附近；fast边界可能沿反事实Native相位预算
`Dbar_N(k;S)=L omega_k^N(1-1/S)`迁移；isolated slow-block的log-uniform目标满足
`J'(a)=[F(aS-1)-F(a-1)]/(a log S)`，只有`F`偶对称且随误差幅值严格增大时才给
`a=2/(S+1)`。若使用明确的分段二次不对称代价，才有
`a(kappa)=(1+sqrt(kappa))/(1+S sqrt(kappa))`；现有实验没有识别真实`kappa`。
Winding-Matched失败只否定端点同余的充分性，不能据此推出`F`单调。

该附件已经列出正结果、反例、不同benchmark身份、未完成实验、代码状态和32GB GPU
约束。现有checkpoint replay与三接口框架可以被保留、修改或完全拒绝；不要为了
顺从当前理论而确认它。

## 目标

确定：对于给定checkpoint和目标倍率S，怎样构造一张固定RoPE表，使模型在Native
到S倍窗口的整个区间尽可能强，并避免Native、内部长度或端点深坑。方案应解释该附件
中的正负交叉，而不是只拟合一个赢家。

## 真实约束

- 当前主线为零训练固定表；LoRA只作历史能力证据，不作为今晚默认解；
- 32GB单GPU，Llama 64K几乎占满显存；应使用单模型驻留顺序多表，避免四个长窗
  进程并发；
- BM/MrPro/Native在相同benchmark身份下只运行一次并永久复用；
- 真实主指标是RULER official完整输出contains分数、task-equal区间曲线、
  log-length AUC、worst-length、任务族、EOS/cap；PPL/KL/几何只作代理；
- 不把小样本跨零或单任务损伤自动当失败，也不把有限反例扩大成方法族不可能；
- Qwen1.5结果已经参与新理论形成，不能冒充该理论的prospective留出。

## 需要的最终结果

请给出一份可以直接交给实验代理执行的研究方案，至少包括：

1. 对三接口理论和checkpoint replay目标的严格审计：哪些恒等式正确，哪些假设不足；
   特别判断一圈slow锚、`Dbar_N` fast预算和非对称`F_kappa`能否在目标benchmark
   开封前由checkpoint识别；
2. 你推荐的最有希望的统一方法，给出明确数学对象、约束、求解方式和可安装表的
   构造；若不存在可信统一方法，也请明确说明并给最佳条件化方案；
3. 解释现有BM/MrPro/C42/Solver、band跨模型/跨S、PPL解离和任务族反转；
4. 对OLMo、Llama、Qwen及S=2/4/8给出可证伪预测，尤其是band、tail depth和
   transition分别应怎样变化；不要从区间最优公式无依据地推出endpoint也会提高，
   也不要给无法由模型推出的具体任务分数；
5. 把一整晚GPU/CPU实验排成有优先级的最小判决队列：每项写清干预、强对照、
   主终点、成功说明什么、失败说明什么、是否进入下一层和预估成本；
6. 明确哪些实验绝不能重复，哪些现有结论仍需mini/medium/high确认；
7. 说明不同结果在论文中可以支持的贡献、不能支持的表述，以及最关键的图表设计。

请严格区分：该附件中已经观察到的事实、由事实支持的推断、你新提出的假说。允许
重新定义变量、目标和实验顺序，但最终方案必须满足单固定表部署条件并能在现有
32GB硬件上执行。
