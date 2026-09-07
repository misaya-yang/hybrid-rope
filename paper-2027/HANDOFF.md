# Hybrid-RoPE 当前交接

- **更新：** 2026-09-07 20:35 UTC；家用PC已同步并核对`40d1ad6`，研究goal持续执行。
- **当前状态：** 核心实验3已完成20条开发比较，正在运行同一固定候选的完整
  13项RULER单臂；尚未证明超过MrPro，不提前进入权重训练。
- **入口：** 规则在[AGENTS](../AGENTS.md)，路由在[INDEX](../INDEX.md)，研究问题和
  方法沿革在[统一研究计划](../docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)。
  本文件只保留现场状态；旧交接记录在Git历史及各结果owner，不是新启动指令。

## 作者的最新目标与授权

1. 先零训练，以作者给的低频去载波思路推进，优先只跑本方方法、对照MrPro
   已公布的同模型同基准数据；复用已有Mr同输入输出，避免重跑对手矩阵。
2. 有有效零训练结果后，研究LoRA能否继续提高、如何训练及避免遗忘。作者指出
   旧Qwen仅16K物理训练、低于Native32K且数据太少；后续优先真实长序列和
   足够数据，不能拿虚拟位置跨度冒充物理64K，也不能把小数据失败泛化到LoRA。
3. 最后研究适配稀疏、压缩和混合attention的位置编码，允许非RoPE。先核实架构
   和明确算子；不默认在这张GPU上下载/训练旗舰模型。三合一长期目标仍保留。

作者允许EVQ整夜工作；**3次是效率期待，不是硬上限**。数值和smoke成本单列，
效果实验如实累计，不隐藏候选扫描、不重启18样本/64维行为梯度路线。后续直接
指令已覆盖最初“不启动实验、两小时、最多3次”的限制；旧计划身份不追溯改变。
目标工具仅能改完成/阻塞，不能编辑活动目标卡片，旧文字以本节授权为准。

沿用已告知作者的12小时安排，资源截止为 **2026-09-08 02:17 UTC
（1788833820）**。所有尝试、失败和重启计入同一预算。作者允许确实无法解决时
关机；小实验或报告完成不是停工点。没有可继续的有价值授权工作时释放GPU。

作者允许必要时清理确认不用且可重新下载的权重，保留当前模型、有效adapter、
原始证据和回执。最近实查系统盘约16GB、数据盘约11GB空闲，**尚未删除权重**。

## 当前实验3：逐槽原生相位约束去载波

- [协议与结果owner](../docs/research/ROPE_NATIVE_SECTOR_CARRIER_20260907.md)，
  [完整数组](../docs/research/ROPE_NATIVE_SECTOR_CANDIDATE_20260907.json)。
- 同一冻结Native背景二次目标，加入`0<=c<=min_B omega`，得到唯一投影
  `c=1.2409377632138785e-6`；j0..46逐位保留MrPro，j47..63为`(omega-c)/4`。
  最末槽为0，所有慢槽在目标窗口内保持自己的Native因果相位区间。没有扫描c。
  这是新候选，不是原Carrier的原地改表，也不保证能力无损。
- tensor SHA `5e51ee46d1bce4adf8b81e6e1594df233f97022ea2fb5ed13fcc658dbee6ec7f`，
  gain=1.138629436111989。当前Qwen2.5-3B-Instruct官方revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`，未训练权重。
- `NATIVE_SECTOR_DIAGNOSTIC_01`已完整完成，监督器590.086秒。128K UUID12.5
  （Mr12.5）、VT70（Mr62.5），VT三条各+20、其余不变；前30-token VT为65
  （Mr60）。短UUID仍50（Mr100）、短VT90（Mr90），不能宣布Native无损。
- 为核对真实128K整段HF相位矩阵，已安排一次无权重/无答案的数值检查插在分片间。
  检查期间队列主进程在分片间暂时暂停，GPU分片正常完成；driver取得同一
  GPU锁完成数值核验后已自动恢复。检查已通过并自动恢复队列；Native及当前表在32K/128K整段sin/cos与逐元素FP32参考
  全部误差0。`native_sector_phase_01/full_phase_precision_result.json`保存结果；原运行参数未改。
- 已启动并实查RUNNING的完整单臂队列：`native_sector_phase_01/run_full_ruler.py`，
  来源SHA `1217d3fe88411ea52c27bbb43db2fc9bed54615f16eac68d912bae10a3b18e4b`。
  job前缀`NATIVE_SECTOR_FULL128_`；首项`NATIVE_SECTOR_FULL128_NIAH_SINGLE_1`。
  输出前缀`runs/native_sector_full128_`，使用独立`code_native_sector_01`代码根。
- 已准备并核验13项×50条128K数据，seed137；`carrier_phase_01/full_data_receipt.json`
  记录650行身份。队列按原任务预算生成（VT30、NIAH128、CWE120、FWE50、QA32）。
  完整13项之后才能计算macro，对照Mr论文53.2；这是跨报告比较，样本数、版本/
  模板/精度差异要披露，不称为与论文配对复现。部分开发输入此前暴露，非盲确认。
- 已完成7项共350行：single1=100、multikey2=30、multikey3=6、VT=60.8、CWE=7.2、
  FWE=70、single2=96（EOS仅9/50）。最近实查`NATIVE_SECTOR_FULL128_NIAH_SINGLE_3`
  为RUNNING，PID12933。GPU此前持续100%、约22GiB；继续时读取监督器，不能据旧PID操作。
  前7项均无另一个合法EOS151643，因此停止token集合不解释这些未结束行为。
- 监督器共用`job_state/gpu.lock`和唯一job ID，检查代码/输入SHA并执行硬超时。
  不重复启动旧ID。当前phase下的plan在执行前写定，每项最多2400秒，仍服从全局截止。

## 已完成且不得重开的分支

- 实验1：P2Middle与MrPro，共44生成。128K UUID0对12.5，VT65对62.5，短能力
  受损；[原owner](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)。
  同实验状态采集/数值分解完成，不能把局部Jacobian当能力梯度。
- 实验2：[作者原始Carrier](../docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md)，
  YaRN中高频、Native估计c=1.2365128835371404e-5，53..63槽反向。20条开发长
  UUID/VT均0；另50条数字检索32分、EOS14/50。后续12项停止排队，不能算完整macro。
  原code根/plan/输出保留。背景统计改善73.85%不等于能力改善；逐槽相位违规也
  尚未独立证明全部错误的因果归因。
- 原8C全文Native均值的`_01`尝试因文本SHA/NPY SHA混淆在前向前失败，`_02`
  完成；`runs/carrier_native_means_02/means.npz` SHA
  `b6d5103b3b30d1a56472fd82fb635cb491c9f3a3e701a87bf725e56835bb0d30`。
  两张Carrier均复用此资产，不重新拟合答案。CWE缺LFS、HTTP下载截断也已修复并保留失败。
- 旧Qwen1.5B有效p2与当前3B Native几何相同、效果不能直接移植；
  [恢复表](../docs/research/ROPE_RECOVERED_QWEN_P2_20260907.json)三哈希匹配。
  不恢复旧Z-only1528步或另开对手训练。

## 资源、CPU并行工作和下一步

- EVQ实际为RTX4080 SUPER、32760MiB；Torch2.8.0+cu128，Flash SDPA only。
  没有flash_attn包，不静默回退二次math attention，不在个人PC重建Torch环境。
- 服务器工作区沿用本任务已有`rope_qwen_baseline_20260907`；`model/`和
  `model_ready.json`有效。完整缓存和原始生成保留服务器，本地保留紧凑回读。
- 已知已完成监督器累计约 **17363.493秒**：此前5191.304秒，加实验3完整基准
  前7项12162.187秒及整段数值检查10.002秒。这包括初始875.55秒、此前实验/状态、
  均值失败与恢复；是作业时长，不冒充云账单。第8项及后续继续累加，不重置截止。
- CPU已准备来自官方PG19 **train** split的128篇不同长书，各取真实连续64K
  窗口，作为可能的后续训练资产；本地下载目录`artifacts/external/pg19_train64k_sources`。
  选择规则在下载前固定为目录首128篇至少500000字节的train书，共110304242字节，
  按云端目录size/MD5校验。128篇已全部校验、传到服务器并完成token化，工作目录
  `long64k_training_assets_01`；`prepared/manifest.json`记录shape[128,65537]、8388608预测token，
  `train64k.npy` SHA `27fb63e425b7ff8eee85b63a646843ad1c5015c743333ced49aa2e4a99c9b918`。
  尚未训练，不能使用PG19 test。
- Native replay的128 train/128 validation行已准备，位于
  `long64k_training_assets_01/native_rows`，四类各32且source IDs不交叉。训练行SHA
  `103ba80663f21f0cb2a14a10e877ae184a69ad04a61dae91d4c728c94af5668e`；验证行SHA
  `f26a560ba03a706351d3e03885a3c4aaf2395d0fc65920ff78701f5e1b0f4426`。
  `long_lora.py`已通过独立原始小模型的teacher隔离/greedy前缀/完整EOS集合CPU测试，
  加上未合并BF16 adapter保存/重载与既有稳定KL检查，共6 passed（5.43秒）；真实64K
  smoke尚未运行。复用旧`stable_teacher_kl`，教师保存FP32 logits，避免已知恒等KL伪梯度。
  `native_lora_eval.py`完成CPU导入，尚未真实生成。独立`code_lora_01`代码根用于
  这些实现，不改运行中的实验3代码。最初CPU测试因漏拷贝既有training.py未能收集，
  已补齐依赖；后续日志在`long64k_training_assets_01/cpu_teacher_test_02.log`至`_05.log`。
- 已核查YaRN正式训练的真实64K长度与约25.17亿token、MrPro不含微调实验、
  LongLoRA的可训练范围区别，以及Qwen3.8/DeepSeekV4的RoPE与压缩接口。见统一计划。
- V4官方model/kernel代码已确认共享KV与输出逆query旋转；固定选集的输出导数
  必须包含value运输项。独立NumPy绝对/相对旋转、有限差分及sink检查通过，
  不属于模型能力实验，也不把现有Qwen固定V响应器直接搬过去。推导与代码见统一计划。
- 本任务心跳`hybrid-rope-2`为ACTIVE，每15分钟接续，只通知实质变化；其他旧心跳
  仍PAUSED。阶段完成/预算结束后暂停本夜心跳，不让过期授权自动启动新GPU作业。
- 下一步：持续核验实验3各分片，检查原始答案与EOS，完成完整macro后判断是否
  达到零训练目标；期间准备真实64K训练的输入和成本估计。未证明获益前不提前训练。

## Git与论文

- 分支`main_0726_09_06`，已含`40d1ad6`；此前提交`0a363f6`、`ba39875`、`2f8183f`、
  `9d04ddd`。训练准备和前6项结果已提交；后续分析继续scoped commit，没有push，保留无关工作。
- 作者提供Pro原文逐字节保留，含两处原有行尾空格；SHA与附件一致，其余diff检查通过。
- `paper-2027` TeX/PDF未改、未编译；活动PDF SHA
  `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`。
  `main_0726`及其`paper/`归档未操作。没有已证实的SOTA或论文接收结论。
