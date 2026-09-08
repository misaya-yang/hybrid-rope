# OLMo 1.485B短评测准备

- 日期 / 状态：2026-09-08；代码及实际CPU分词准备完成，不是新能力结果。
  冻结24行，共228099输入tokens，单行4038–16335。GPU吞吐、Native资格和baseline分数尚未测得。
- 问题：在一个固定小型、多任务开发面板上，候选是否优于同模型、同输入、同gain
  的MrRoPE-Pro，并保留4K分项？目标仍为最多10个经审查的候选，目前只有
  [MrPro-BM](ROPE_MRPRO_BM_PROTOCOL_20260908.md)准备完成。
- 原[ψ投影C1](ROPE_MRPRO_TRANSITION_REVIEW_20260908.md)保留未调整数组及审查，
  不推荐GPU。BM作为第一项；其余尚未完成，不用未经定义的曲线凑数。

## 固定模型与部署

复用allenai/OLMo-2-0425-1B-Instruct，实际1,484,916,736参数，revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`。Native4096、base500000、16层、
16Q/16KV heads、head dimension128。基座权重冻结。

MrPro采用已有真实64槽数组、静态s=4、gain1.138629436111989。4K、8K和16K
输入全部使用同一部署表；标准RoPE和Flash SDPA保持一致，不随长度更换频率。
共享模型只加载一次，完整MrPro结果缓存供后续同身份候选复用。

## 自组开发benchmark

四类独立构造的任务，不直接取一个RULER子任务，也不跑完整公开benchmark：

| 任务 | 必须使用的信息 | 对应的失败可观察量 |
| --- | --- | --- |
| 精确记录检索 | 指定ID的标签 | 是否找错相似/无关记录 |
| 两步关联检索 | item→holder→label | 是否丢失两条证据的绑定 |
| 最后一次更新 | 相同ID的有序多次记录 | 是否读到旧值 |
| 双属性绑定 | 同一记录同时满足两个条件 | 是否将不同记录的属性拼接 |

每类含4K/8K/16K各两个反事实世界，共24行。每对世界的query和干扰项一致，
正确答案随证据改变；阻止只靠query猜答案。目标均为一个普通标签词。生成seed、
文本、实际token IDs、顺序和SHA在模型输出出现前冻结。

另有8行≤768token的Native资格检查，使用不同seed。通过条件预先固定为至少
6/8正确且每类至少1/2正确；未通过先停止，不用地板任务给候选排名。旧OLMo开发
结果中确有复杂绑定和16K任务地板，因此不能在没有本次资格结果时宣称能力已足够。
这也不证明OLMo不能胜任这里更简单的任务。

解码为greedy、beam1、repetition_penalty1、最多16个新token，实际merged
GenerationConfig落盘。主分数为整个答案归一化后的单词exact match；允许大小写、
外围标点和固定Answer前缀，不在长输出里搜索正确词。EOS只单独报告，不额外扣分。
四类等权；保留逐题输出及所有分项，不以单项优势代替总分。

## 五分钟成本目标与结果分支

- 每个主评测约5分钟是成本估计，模型加载是一次公共成本，另记。先从baseline
  得到真实耗时；超过估计继续完成固定样本并记录成本，不强杀、不删题报分、
  不更改某个候选的decoder。进程异常或作者停止造成的半次结果不能当完整分数。
- 完整MrPro只运行一次。只有模型、输入、scorer、decoder或执行实现发生相关
  变化才使旧baseline失效；新增固定候选本身不使baseline失效。
- 总macro严格上升且4K正确数不下降，标为开发胜出，停止筛选并分析该候选，
  再定义更深入的比较。总分上升但4K下降单列tradeoff；无收益则进入下一项已冻结
  且经审查的候选。开发胜出不等于泛化或SOTA。
- 仅当作者实际给出总时间预算时，启动命令才设置deadline；恢复不刷新这一预算。
  未给出时不从“预计5分钟”制造总时限。监督器报告进度和超出估计，保留独占锁
  与STOP路径，断开SSH不丢监督；半次输出不自动当完整结果重用。

当前只有BM单项待运行，GPU阶段未启动。可执行队列为空时，普通启动会在
加载GPU模型前退出；只有明确的baseline-only运行或已完成审查的候选才可执行。

## 入口与验证

- [任务与scorer](../../scripts/experiments/olmo_fast_screen/bench.py)
- [CPU分词与冻结](../../scripts/experiments/olmo_fast_screen/prepare.py)
- [共享模型、baseline缓存和候选评测](../../scripts/experiments/olmo_fast_screen/run.py)
- [持久预算监督器](../../scripts/experiments/olmo_fast_screen/supervise.py)
- [相关测试](../../tests/test_olmo_fast_screen.py)

原服务器既有Python环境的14项相关测试通过，覆盖独立BM最优解、非平凡投影最小值、实际未裁剪
数组、反事实输入、scorer、完整配对、理论审查状态和恢复不重置deadline。
这些不是模型能力或GPU验证；首次真实模型调用必须保持Flash后端，禁止退回
长序列math attention。真实运行准备和资源位置由HANDOFF统一维护。
