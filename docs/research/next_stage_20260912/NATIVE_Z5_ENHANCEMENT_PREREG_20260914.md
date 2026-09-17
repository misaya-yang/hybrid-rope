# Native-Z5成熟checkpoint原生窗口增强预注册（2026-09-14）

## 科学问题与价值

冻结`OLMo-2-0425-1B-Instruct`全部模型权重，只改变RoPE两个实际端点之间的
内部归一化坐标z，能否在原生4K窗口内优于模型发布时的geometric Native表？

这与TailSpline外推确认回答不同问题：

- TailSpline队列检验解析静态表在2--4倍原生长度是否优于MrPro及其他静态基线；
- 本实验检验geometric RoPE是否已经是成熟checkpoint自身的Native optimum。

若独立4K自然文本NLL和至少一个任务族都可靠改善，可支持：对这个冻结checkpoint，
z不仅是外推分配变量，原生几何表也不是事后部署的最优点。该结论仍是
checkpoint-calibrated，不产生不读模型的通用构表规则。

## 固定干预

- 模型：`allenai/OLMo-2-0425-1B-Instruct`；Native长度4096；
- 参数化：现有`Z5KnotRotaryEmbedding`；七个固定pair-index knots，两个端点
  固定，五个有效内部z自由度；
- `support_factor=1`，最快/最慢实际频率、全support、base、gain=1均保持；
- 六个正gap由softmax参数化，累计后严格有序；
- 模型权重更新数为0，只优化z gap logits；
- 单一Native初始化、AdamW、40步、lr `0.003`、seed `20260914`；不扫学习率、
  步数、初始化或参数量。

## 目标函数

每个4097-token连续文本提供4096个next-token targets。记录四个1024-target
位置段：`1--1024`、`1025--2048`、`2049--3072`、`3073--4096`。

目标为四段mean NLL的等权平均。因为四段target数完全相同，该目标在代数上
等于完整4K dense NLL；分段用于检查改善是否只集中在尾部，不能当成另一种
独立目标。

## 数据隔离与选择

优化数据来自PG19 validation parquet的50本书，每本按文本SHA确定一个连续
4097-token窗口，按SHA顺序冻结：

- design 16本：计算梯度；
- selection 16本：在step 0--40中选最低dense NLL状态；
- internal-confirm 18本：选表后首次读取。

最终自然文本holdout复用既有PPL46：ProofPile test 32文档、PG19 test 14文档，
与PG19 validation优化书籍分离。比较Native与冻结Z5表在1K/2K/4K前缀的
paired document NLL；主要NLL终点为4K。

任务分数不参与选z：

- OLMo classic Native-4K RULER Full-13×10，共130条；
- 原冻结、未截断、prompt+generation不超过4096的HotpotQA/2Wiki/Qasper，
  共99条，任务数为5/24/70。

自然QA的Hotpot只有5条，因此任务等权macro与区间必须同时报告该限制；不能
将该块写成完整或平衡LongBench。

## 判决

同时报告点方向与较强判据：

- heldout 4K NLL：candidate-Native的paired-document 95%区间上界小于0；
- RULER或自然QA至少一个task-equal差值的95%区间下界大于0。

两者同时满足才称`strong_native_enhancement`。只有点方向同时改善时称
`directional_native_enhancement`，不能写成稳定胜出。NLL改善但任务未定时，
只能主张原生语言建模可被post-hoc z校准改善；任务反向必须完整保留。

若selection最优仍是step0，输出Native相同表并登记负结果；不据此调优化器。

## 与既有证据的边界

旧62自由度direct-z用两篇design文档优化2x tail NLL并以1x作no-harm guard，
不能回答本问题；其一条2x heldout回归超门只否定该旧小样本协议。

LeRoPE的Fixed LeRoPE结果说明，在另一训练run学习频率后，以这些固定频率从头
训练能保留部分收益。它不证明成熟checkpoint事后换表会改善；本实验正是对此
差异的直接检验。

## 实现与队列

- 资产：`experiments/native_z_enhancement_20260914/prepare.py`；
- z优化与1/2/4K holdout：`optimize.py`；
- Native/Z5任务比较：`run.sh`；
- 配对报告：`report.py`；
- 后继队列：`followup_chain.sh`，等待当前TailSpline主链完成后自动启动。

服务器根：

`/root/autodl-tmp/today_rope_plan_20260914/olmo_native_z5_enhancement`

冻结资产身份：

- PG19 validation `50×4097` tokens SHA256：
  `26226aac62a30e3b29d521dfa177ee6c22c8bc6affa086b735eea628d864d12b`；
- RULER130 SHA256：
  `cdf69c94ecc8fd3d95fac7547137ad5f5cdcc2c2ae139ae63f49275e8c06b5b2`；
- Natural-QA99 SHA256：
  `a94428d3513f7578c9d013aecd015f427c02566ee20bb72caee95e5ab087f877`；
- PPL46 array SHA256：
  `0864553fbed3d4967dcb517c867284d0c0856d6fd714d0ecc9888bfce7724f1a`。
