# Hybrid-RoPE 当前交接

- **更新：** 2026-09-07 23:01 UTC；研究goal持续，作者最新要求第一性原理和1.5B最小验证。
- **当前状态：** 3B完整RULER队列已停止于9项，未完成macro、未证明超过MrPro。
  1.5B数据已冻结，三臂首条64K probe已完成，固定24条三臂已全部完成，同表1.5B 128K长度检查已运行。不得恢复旧13项或88行Mr建议。
- **入口：** [AGENTS](../AGENTS.md)、[INDEX](../INDEX.md)、
  [当前小模型协议](../docs/research/ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md)、
  [统一研究计划](../docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)。
  HANDOFF独占现场状态；历史状态用Git和结果owner，不另建交接档案。

## 最新授权与预算

作者要求先用原理和已有有效材料确定频率改动，以尽量少的GPU生成验证。允许最多
两位代理，当前分别完成数学/旧方法核对，正在补两个有界实现；不得再增加代理。
GPU优先用已有Qwen2.5-1.5B验证，不能因为利用率而继续长矩阵或扫描新曲线。
零训练有效后再研究真实长上下文LoRA及能力保留，稀疏注意力位置编码最后；三合一
目标保留，论文接收/SOTA均是目标而非已获结果。

EVQ整夜研究已授权；3次是效率期待，不是硬上限。后续直接指令覆盖最初不启动/
两小时/硬3次限制。目标工具不能编辑活动目标卡片，旧文字以此处最新授权为准。
**沿用同一资源截止2026-09-08 02:17 UTC（1788833820）**，所有失败、准备前向、
smoke和正式生成累计，不重置。没有可继续的高价值工作或到期时按授权释放GPU；
小结果/状态回答不是任务终点。保持监督器硬超时、唯一job ID和同一GPU锁。
作者允许必要时清理确认不用且可重下载的权重；本轮尚未删除任何权重。

## 当前1.5B固定比较

- [唯一候选与两条参照](../docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json)：
  完整旧p2规则只修复lag子采样混叠，保留p2、rcond和历史gain；不称为理论最优。
  FullLagP2 tensor SHA `ecd0c280a11788e0c4a869a3d162880964ba3f371f589f7e2f5ae471d461096b`。
  对照官方MrPro及同gain MrPro，分开完整方法收益和gain混杂。
- 已有Qwen2.5-1.5B-Instruct官方revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，
  28层/12Q/2KV；权重SHA `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
  与官方LFS匹配，无需下载。独立工作目录`qwen15_mechanism_phase_01`、代码根
  `code_qwen15_01`，复用现有权重符号链接，不改旧运行代码。
- 固定64K MK2/VT/FWE各首8条、seed137。先三臂各首条MK2，两条同gain臂保存
  真正生成每一步的Q/K/V/query和LM分数；每臂180秒硬上限，随后有界补齐固定
  24条并复用探针，扩展总硬上限900秒。详见协议的结果到行动映射，非完整macro。
- 实际生成trace的微型CPU集成测试已通过：`test_long_lora_native_teacher.py`
  **5 passed，5.12秒**，包含新加的hook不改greedy输出/逐决策query测试；此前稳定
  Native KL独立测试已有通过记录。首条三臂均0分并EOS；本方4792062，两Mr为7315917，参考8948515。普通生成约6秒，带trace约17秒，峰值9.323GB；三项均超过官方及同gain Mr：MK2 37.5/12.5/25、VT87.5/82.5/77.5、FWE70.83/45.83/45.83（本方/官方Mr/同gain Mr）。三臂各24条均EOS，局部小样本结果，不是完整macro。
- CPU分析已证实旧p2高频尖峰主要来自16步lag抽样混叠；完整lag中段与旧有效表
  频率差至多0.01243%。旧有效表结果不能改名给新表，也不能由1.5B推定3B有效。
- 载波只旋转低频复数包络；背景能量下降不等于正确/干扰log odds提升。旧3B缓存
  只记录提示词末端query，多数局部变化小于BF16重放误差，因此本次保存真实决策。

## 已完成作业与保留的证据

- 实验1 P2Middle44生成：长UUID0对Mr12.5、VT65对62.5，短能力退化。
  [owner](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)。状态及Jacobian检查
  是诊断，不是能力优化器；不重启18样本64维梯度路线。
- 实验2原Carrier：20条开发长UUID/VT均0，另50条数字检索32，EOS14/50；停止
  余下矩阵。[owner](../docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md)。Native8C
  均值资产SHA `b6d5103b3b30d1a56472fd82fb635cb491c9f3a3e701a87bf725e56835bb0d30`保留。
- 实验3相位约束Carrier：[owner](../docs/research/ROPE_NATIVE_SECTOR_CARRIER_20260907.md)。
  20条开发比较后9项各50行：single1=100、multikey2=30、multikey3=6、VT=60.8、
  CWE=7.2、FWE=70、single2=96、single3=98、multikey1=92。450行部分集合均值62.22，
  **不是13项macro，不能与Mr论文53.2直接判胜**；余4项没有运行。
  队列7351已核对命令后终止，`native_sector_phase_01/phase_reorientation_02.json`
  记录queue_absent=true。续接先查真实进程，不据历史PID操作。
- 实验3完整128K矩阵相位数值检查已通过，HF sin/cos与独立FP32参考误差0；
  它排除该实现差异，不证明候选正确。已完成任务的原始输出/计划/失败均保留。
- 监督器累计已知完成 **21109.157秒**：此前17363.493秒加第8项1880.339秒及
  第9项1865.325秒。包含已记录失败/数值/状态作业，不冒充云账单。新阶段继续累加。

## 后续可复用资产

- EVQ为RTX4080 SUPER、32760MiB；Torch2.8.0+cu128，Flash SDPA only，不能静默
  回退math attention。个人PC只做轻CPU/code/docs。新作业前已确认旧队列无模型进程；QWEN15_128_FULLLAGP2_01正在运行，随后同输入Mr；控制器17730、子进程17735仅为本次快照。plan SHA `0bcc77e1a695e2e3e11ec8cf2d180a52cb174d8af544ec24af1f90559cfcb19f`，100%利用率，15889MiB。
  任务工作区仍为`rope_qwen_baseline_20260907`；GPU锁为`job_state/gpu.lock`。
  最新数据盘约11GB可用，双臂决策缓存预计约4GB，逐层CPU分析，不复制大缓存到个人PC。
- `long64k_training_assets_01/prepared/train64k.npy`已准备：PG19 train128本不同书，
  shape[128,65537]、8388608预测token，SHA
  `27fb63e425b7ff8eee85b63a646843ad1c5015c743333ced49aa2e4a99c9b918`。
  1.5B/3B tokenizer字节相同，可复用；尚无64K训练/训练smoke。
- Native replay128 train/128 validation、四类各32、source IDs不交叉，位于
  `long64k_training_assets_01/native_rows`；train SHA
  `103ba80663f21f0cb2a14a10e877ae184a69ad04a61dae91d4c728c94af5668e`，val SHA
  `f26a560ba03a706351d3e03885a3c4aaf2395d0fc65920ff78701f5e1b0f4426`。
  属历史development资产，不能称盲测；新模型需重新采其真实原生teacher。
- `long_lora.py`复用已有stable_teacher_kl，all-linear r16/alpha16、真实64K CE+
  Native KL；128步方案未执行。旧13项first10评测输入仅CPU准备，不是当前必跑队列。
  未获零训练有效结果前不提前训练。YaRN训练/动态推理、MrPro无LoRA结果、V4共享KV
  数学检查及限制见统一计划，不在此恢复旧建议。
- 本任务心跳`hybrid-rope-2`为ACTIVE每15分钟，已改为上述小模型/机制优先顺序，
  明确禁止恢复旧矩阵；其他旧心跳PAUSED。无变化保持安静，到期暂停本夜心跳。

## Git与论文

分支`main_0726_09_06`已含40d1ad6，提交0a363f6、ba39875、2f8183f、9d04ddd、
9acede6。相关验证后继续scoped commit，不自动push、保留无关工作。新trace/小模型协议已提交194fe71、8f17843。作者Pro原文逐字节保留其两处尾空格；main_0726归档未操作。
活动TeX/PDF未修改或编译，PDF SHA
`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`。
没有已证实的SOTA、新方法最终胜利或论文接收结论。

继续记录：首臂23条补齐监督器155.028秒。原交互SSH控制器在该臂完成后连接被
远端关闭，两参照尚无attempt；已在相同plan SHA
`57c56fe3c46375b4227ebc66b83605976d27b4777902d369cb4a1549141380f1`下恢复两参照，
未重跑本方。控制器`qwen15_mechanism_phase_01/refs_controller_01.py`已脱离SSH，
PID17348仅为当时记录；续接读实际job状态和controller_complete文件。累计预算不变。

1.5B当前64K结果在ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json，完整生成hash已
核对；128K两臂各24，首8条来自原固定50条输入，不称盲确认。新LoRA代码根
`code_qwen15_lora_01`只复制已有实现、接受table_key=FullLagP2和真实1.5B身份；
超参数/训练输入未改，尚无LoRA GPU启动。不能将此准备里程碑当训练结果。
