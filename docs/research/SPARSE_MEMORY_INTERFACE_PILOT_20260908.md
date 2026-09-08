# 压缩记忆来源绑定：两小时自主开发实验

- 日期：2026-09-08。状态：RUNNING；首轮 A/B 已冻结，正在启动。
- 问题：保留 learned channel gate、块内 APE、因果上游状态和共享 KV 读取时，
  gate 后、聚合前加入来源相对旋转，能否改善窗口内关系答案生成？
- 本阶段为合成小模型开发筛选，不是 V4 复现、自然文本能力或论文新颖性证明。
  资源授权及实时进程只由 [HANDOFF](../../paper-2027/HANDOFF.md) 维护。

## 来源与适用范围

作者提供的 `sparse_position_research_20260907.md` 是设计输入，SHA256
`124fa965a76837afe0af5d7e1ca04302ebbef6b9283667a76a4affa3b2dcbfb9`。
此前对话全文读取；本次重读实际压缩接口、任务和§13实验条款。
2026-09-08重新取得并审核 [V4官方实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py)，
本地取得原字节SHA256 `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`。
核对 gate+APE → 按来源softmax → 逐通道乘法 → 聚合 → learned RMSNorm →
块锚点旋转，以及共享 KV 的共同 softmax 和输出逆旋转。

保留的核心算子不包含 V4 的 MoE、CSA重叠/索引器、QAT或真实预训练权重。
本次用3层、width192、3个query头、共享KV128维（末16维为8个固定旋转对）、
压缩率16、局部窗16；频率为 `40000^(-2i/16), i=0..7`，gain=1。
各层局部与压缩读取使用同一表；可学习逐通道 RMSNorm 保持原顺序，固定块起点锚定。
可读记忆预算为每层8个压缩KV加局部窗；训练通过掩码约束可读集合，尚未实现持久的逐token推理cache，不声称实测压缩部署加速。A/B/C参数量相同。窗口内总输入132 token。
CUDA只允许 memory-efficient SDPA，不静默回退math；短序列用稠密掩码表达
稀疏可见集合，不声称已实现长序列稀疏kernel吞吐。

## 冻结任务与判别

32个四token事件后接4个query token。远端同一16-token块含同实体三次赋值和
一个marker，marker位置随机为块内第2或第3事件；正确答案是marker前最近一次值。
交换紧邻marker两侧的赋值值，保留token多重集合、位置、query及其他内容，答案改变。
三次赋值避免“总取第一次”绕过marker；target/unique实体和值均随机化。
来源到query距离超过3层纯局部传播上限45 token，基线仍可通过真实压缩路径学习。
每个配对另有一个唯一实体内容检索query，训练比例关系:内容=2:1。

- train：40000对/120000行，seed2026090801。
- development：512对/1536行，seed2026090802。
- test：2048对/6144行，seed2026090803；首次A/B完成前不读取预测。
- 数据固定一次、记录文件SHA、完整生成器及逐对布局。早期两赋值草稿未训练，保留为
  工程草稿；正式身份是三赋值`data_v1`。
- 模型从同一随机初始化训练；batch索引使用独立固定RNG，使相同seed各臂逐步配对。
  标签不在模型输入中，生成是完整64词表argmax所得一个值token，不使用候选答案mask。
- 主指标：两世界同时正确的pair EM；同时报告单行relation EM、content EM、
  正确值对最强竞争token的logit/log-probability margin、NLL和两种marker布局分桶。
  独立事件解释器验标签；因果检查覆盖未完成块和未来token扰动。

## 首轮预先固定的科学预测与动作

A baseline：`R(anchor) Norm(sum_j gate_j * u_j)`。
B TP：`R(anchor) Norm(sum_j R(offset_j) (gate_j * u_j))`。
预测：B改善pair EM，且普通内容读取不受实质损伤；无此差异不能用相位/摘要距离
或NLL代替实际收益。本轮不宣称从代数推出学习收益的保证。

首轮seed137，每臂3000次AdamW更新、batch64、lr0.0005，100步warmup后cosine至0.1倍，
weight_decay0.01，betas=(0.9,0.95)，梯度范数裁剪1。各臂192000个监督答案token、
25344000个输入token；固定第3000步为最终checkpoint，500步development只作学习诊断。
先测生产形状10次更新吞吐；若首轮预计超过剩余预算60%，只在任何正式结果前
共同下调更新数并记前瞻修订；不凭验证分数选训练长度。

- 若B相对A的dev pair EM≥5个百分点且content下降≤1个百分点：继续最小必要
  seed256配对复核；若方向一致且有预算，才训练C位置边际控制以区分extent解释。
- 若B改善但低于门槛，或内容代价超门槛：保留该候选开发未过线，不升格成功。
  若独立seed可直接决定保留/淘汰，在预算内做一次配对复核。
- 若双方pair EM<20%或都>95%：该终点地板/天花板，不能判机制不存在。先检查
  学习曲线和已经通过的代码证据；必要时只加入能直接确认任务可学性的最小控制，
  在新协议条款中先写明配置、预测和成本，不开宽扫。
- 有判别力的负结果结束当前接口在该assay上的候选主张，不能关闭全部压缩方法。
- test在A/B固定checkpoint后一次评估；若后续根据development选择控制，test一旦
  曝光视为开发证据，不能再称独立确认。seed间差异与样本配对置信区间分别报告。

## 工程审核、成本和停止

已人工沿输入→gate/旋转→共同attention→逆旋转→完整词表argmax→pair评分审查。
无卡工作机CPU检查：`python -m pytest tests/test_sparse_memory.py -q`，4 passed、1 GPU skipped（5.56秒）。验证独立复数旋转、单来源退化、非零梯度、未来不泄漏、标签和配对
多重集合。GPU开启后执行一次后端前后向及CPU数值参照，随后立即生产形状吞吐测量。
已有通过且未改动检查不为凑次数重跑。

代码：[model](../../scripts/experiments/sparse_memory/model.py)、
[data](../../scripts/experiments/sparse_memory/data.py)、
[run](../../scripts/experiments/sparse_memory/run.py)、
[supervisor](../../scripts/experiments/sparse_memory/supervise.py)、
[tests](../../tests/test_sparse_memory.py)。
双层排他锁、不可覆盖的run目录/attempt回执、脱离SSH的监督器、绝对deadline和
进程组timeout防止重复/越界。SIGTERM只在完整更新边界保存nonfinal recovery，
不把它当最终checkpoint。故障保留回执，修复后新attempt继续，共享同一阶段期限。
预算涵盖准备、失败、修复、训练和分析；不会重置为每个job两小时。

支持：上述实现和有限合成assay的实际结果。未支持：真实模型瓶颈、稳健自然文本
收益、SOTA、首次旋转聚合、oral或新主线已经成功。历史静态频率收益与负结果均保留，
不重启旧表搜索或LoRA队列；活动论文和`main_0726`归档不动。

## 首轮运行前回执

- GPU：RTX 4080 SUPER，32760 MiB；GPU专项1 passed（3.19秒）。
- 生产配置1581120参数；12次吞吐更新后10次均值0.024509秒，peak626221056字节。
- 保留3000步共同终点，不改数据/频率/阈值。首轮预计训练纯更新约147秒，另含验证及启动。
- 03:56 UTC原EVQ有卡启动，原宿主机短暂缺卡后恢复，未创建克隆或启用其他实例。

## V1判决与唯一一次任务修订（04:02 UTC，V2运行前）

首轮seed137两臂均完成3000步；development pair EM均100%。内容A18.359%、
B20.117%；原指标因关系天花板不具判别力，不解释为TP有效或机制不存在。
代码复查确认：三次赋值虽避免“总取第一次”，但marker前邻事件始终是查询实体，
因此直接复制marker前一个值即可答关系题，无须实体条件检索。V1作为无效诊断
保留，不用于候选选择。该问题出在任务结构，不是旋转/评分bug。

V2只修正这个已识别的捷径，并固定更拥挤的同缓存读写场景：

- 每个64-token块含5个实体各3次赋值，三轮顺序独立随机，marker插在第二轮内部。
  query在生成完整上下文后随机选5实体之一，通常与marker前邻实体不同；每个
  query实体在marker两侧均有赋值，交换其最近前驱和第一后继的值。
- 8块共512 context token+4 query token；压缩率64，仍为8个128维摘要+窗16，
  3层/width192/固定8对频率/gate/APE/RMSNorm/共同读取均保留。改动是任务密度与
  压缩率，绝不与V1分数做方法效应的合并比较；V2 A/B互相严格匹配。
- 所有块密度相同；唯一实体内容控制放在其他远端块。答案跨度仍在训练窗口内。
  `data_selective.py`生成独立seed 2026090811/12/13；train40000对、dev512对、
  test2048对。元数据额外冻结同上下文、不同实体且答案不同的alternate query。
- 独立解释器1000对核验通过；仅约1/5 query可由marker邻居捷径回答，>95%输入
  有可验证的异答案替代query。工作机专项1 passed（1.26秒）。
- 预先固定seed137 A/B各6000步，batch64、优化器和学习率同V1；每1000步dev诊断，
  最终6000步checkpoint。仅以实测吞吐确认预算，不根据dev选择终点。
- 预测与≥5pp/content≤1pp继续门槛不变。若有收益先seed256复核，再C边际控制；
  若两臂继续地板，只运行一个不压缩的可学性控制来区分“任务尚不可学”与
  “压缩路径限制”，不继续反复改题造收益。若天花板再次出现，终止此合成assay，
  如实报告现有强基线足以解决当前负荷，未识别实际接口收益。
- V1 test一旦读取不再封存；V2为前瞻性开发修订，有独立生成数据但不是论文确认集。

### 若V2两臂地板：已准备的可学性控制（04:10 UTC，未启动）

只在两臂固定6000步均pair EM<20%时执行。保持V2模型宽度/深度、token数据、
seed137、6000更新、batch64、lr和优化器；把attention可见集合改为全部因果原KV，
并屏蔽压缩摘要。这是取消压缩约束的诊断控制，不是同缓存候选方法，也不参与
B≥5pp的方法门槛。先预测：若原token控制pair EM≥50%，而A/B<20%，说明这个
同规模训练设置能学到明显关系能力，但本次压缩配置/候选未保留足够可用信息。
若原token控制也<20%，本阶段训练/架构可学性仍未建立，不能归咎于压缩接口。
20–50%作为部分可学，报告实际差值，不改阈值称成功；content和alternate-query
检查辅助定位，不能替代pair生成。最多一个这样的控制；不再改第三套数据。

人工代码复审补上：eval/benchmark共享训练锁，监督器收到SIGTERM后转发到子进程组
并停止后续队列。原V1/V2训练代码快照保持不动，新检查/诊断使用`code_reviewed`；
模型算子在dense_control=false下无变化，A/B原初始化和数据SHA完全一致。

复审验证：dense可见性CPU检查通过；监督器真实子进程SIGTERM测试通过（3.85秒），
确认子进程回收且不启动第二job。第一次测试fixture把总剩余时间设为45秒，被已有
“少于60秒不开新job”正确拒绝；只修fixture为120秒，保留失败日志，未修改运行门槛。

V2首轮6000步development：A/B pair EM 8.59375%/8.3984375%，差-0.1953125pp；
5000次样本对bootstrap区间[-2.734375,2.34375]pp（仅条件于seed137）。
A-only/B-only为22/21对。内容9.5703125%/13.8671875%。触发地板诊断分支，
不触发seed复核或C方法矩阵。04:13 UTC启动已冻结的dense控制；三臂初始参数SHA
`e8ef2ec2a7fdde92c3527cc3d91cba5bdd5456c4fab31c51ae4a41a7a4db594b`相同，训练数据SHA相同。
