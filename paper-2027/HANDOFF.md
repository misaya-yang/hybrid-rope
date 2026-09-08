# Hybrid-RoPE 当前交接

## 当前授权阶段：压缩记忆来源绑定（2026-09-08）

- 作者已授权约两小时全自主代码准备、审查、无卡关机/有卡启动、训练、分析与必要修复；Chrome可作内置浏览器备用。不逐项询问。
- 阶段开始约03:40 UTC，硬截止05:40 UTC（2026-09-08）；最迟05:38停止新训练并收集，05:40释放GPU。所有attempt共享此期限，不继承旧过夜预算。
- 当前：04:13 UTC V2 A/B各6000步完成，dev pair EM为8.594%/8.398%，均地板。按前瞻条款已启动单个不压缩的同配置可学性控制，监督器PID2918，计划`dense_control_plan.json`，`code_reviewed`，输出`runs/selective_dense_seed137`；日志`dense_control_supervisor.log`和`supervision/selective_dense_seed137.log`。原A/B监督器PID2245已完成，计划和原快照保留。无卡CPU4 passed、GPU专项1 passed，数据冻结。新主线按[接口pilot协议](../docs/research/SPARSE_MEMORY_INTERFACE_PILOT_20260908.md)；旧L/P、静态表和LoRA均非本轮队列。
- 仅操作原EVQ实例 `c904489327-8b72fcf9`，SSH端口27741；不触碰GRPO实例。服务器阶段根`/root/autodl-tmp/sparse_memory_20260908`，解释器`/root/miniconda3/bin/python`，正式数据`data_v1`。
- 首轮已完成监督器PID1523，计划`initial_plan.json`，日志`initial_supervisor.log`及`supervision/seed137_*.log`；run目录`runs/seed137_{baseline,tp}`。V1各3000步结果保留；当前入口`code_selective`/`data_selective`，config_selective.json压缩率64，8份摘要与原缓存预算一致。V2 A/B预定各6000步，只修一次任务捷径，按协议V2条款继续。Chrome已保存定时关机：页面服务器时间UTC-4，2026-09-08 01:40（即05:40 UTC），显示设置成功。训练截止05:38 UTC；提前结束时主动关机。停止单个run：SIGTERM其`supervision/<job>.json`记录的进程组；nonfinal快照不能当完成。
- 下面为上一阶段历史状态；其中“仅文档/不启动/缺预算”等限制已被本次明确授权取代。

- **更新：2026-09-08。按作者最新要求完善失败复盘、实验代码审核和判据。**
  本轮仅文档工作；不生成新表，不运行GPU，不恢复旧队列，原科研目标仍未完成。
- **先读：** [本地失败谱系与综合分析](../docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)，
  [逐源证据/原始数据复算](../docs/research/ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json)，
  [有条件的固定位置判别](../docs/research/ROPE_FIXED_POSITION_VISIBILITY_PROTOCOL_20260908.md)。
  按INDEX定位原owner即可，不需要重新生成一批综述。

## 最近科研交付与验证限制

- 新审查：指定的Gemini整改报告已全文阅读；事实沿用各自原始owner，广义
  Non-Identifiability Theorem和未经识别的机制不采纳。结论按“事实→排除的具体
  解释→未知→下一项判别”整理于失败谱系§5–7。局部Taylor/Fisher在相应假设及
  trust region内仍可用；路径积分、Gram重排不成为新能力selector。
- 作者重点纠正：实验前先审核实际代码；稳定路径复用验证，不反复hash或smoke；
  工程检查、科学预测、实用门槛分开。L/P补齐分支取舍、无效诊断和停止条件，
  仍只是单行oracle诊断，不是方法验收或最优频率的答案。
- 两位代理全文复核10份RoPE来源，共7310行；9份新来源已按原字节加入
  `research/external-reviews/pro-materials-20260908/`，1份scale来源复用。
  另2份仅做主题筛查后排除；不声称12份全部全文阅读。
- 本地范围包含tracked文档、ignored results/artifacts/internal/outputs等及
  main_0726只读归档。全文/节选/清单/原始数据复算分别标记，不把报告数当实验数。
  19个原始manifest+raw身份、9424行重核一致，另一代理6144行复核与之重叠。
- 已修旧ALS首轮`inf<=inf`虚假收敛，补非平凡回归例；独立显式旋转核验通过。
  已纠正任意LoRA包含关系、rank按头均分、残差界方向等错误理论说明。
- 已给仍可检索的Phase16、YaRN/Y2、Native compact统计判读、频带定位及Solver Z
  混用锚点等原文加可见纠正。AGENTS只写前瞻规则，细节在研究文档。
- 通用单表安装器不再用频率降序拒绝原始有效p2，保留64槽有限值与冻结SHA；
  原始表未排序、未重跑。FullLag修复表不能冒充原始p2。
- 固定位置可见性代码已接入现有单表评测器，状态为
  **PREPARED_CODE / RUNTIME_UNQUALIFIED / NOT_RUN**。已做轻CPU布局/语法和数学
  复核，未执行真实Torch/Flash/KV或canonical pytest，未得到新方法能力结果。

## 公司PC从这里继续

1. 在`main_0726_09_06`普通拉取origin最新提交；先保留该PC自己的dirty work。
   已有分支包含40d1ad6，本轮不改历史、不force push。
2. 先按L/P协议确认有哪个具体设计决定依赖它；若没有就搁置诊断，继续CPU研究。
   若继续准备，先审核实际输入→干预/KV→decoder/scorer→监督器执行链，修实质
   错误；在原协议简记审核代码状态，不能用hash匹配或测试次数替代代码审核。
3. 只准备既定1.5B/FullLagP2单行、同一S集合，不新增表或两世界。完整原token
   与来源/模板边界从既有资产读取并冻结一次，模型/数据/表manifest复用。
   已通过且无相关变化的检查不重跑；未完成的相关工作机测试按协议一次补齐。
   新KV路径仅做一次tiny all-keep联合检查，再用C及冻结L/P取得实际判别。
   不同模式不各自重复准备；保持原位置、最后prompt query时点和完整decoder历史。
4. 不重启64维行为梯度、频率候选扫描、13任务全矩阵或旧LoRA队列。新阶段的
   资源/预算以作者当时指令为准，不继承已关闭过夜deadline。四种L/P结果分开，
   原分数、EOS、资格失败分开；部分收益不放宽原判据，不自动扩矩阵或转LoRA。
   oracle恢复不当作可部署方法或SOTA，也不提前保证某种CPT配方有效。

当前仍缺：完整S输入、针对修订协议的执行链审核与新KV runtime验证，以及新
付费阶段预算/deadline。文档已准备不等于这些工作已完成；本轮不为补齐它们开机。
开始下一轮已有授权工作后，由执行者完成必要审核和一次检查，无须逐项再问作者。

## 已收尾的GPU现场

- 仅本任务EVQ：`ssh -p 27741 root@connect.westc.seetacloud.com`。
  工作根`/root/autodl-tmp/rope_qwen_baseline_20260907`；Python为
  `/root/miniconda3/bin/python`。原环境Torch2.8+cu128、Transformers5.15.1，
  4080 SUPER 32760MiB；恢复后必须复查，不沿用旧在线状态。
- 正式LoRA在59/128完整更新后按作者收尾请求停止，3866624预测token。
  无最终adapter/manifest，无训练后能力或遗忘结论；FAILED/exit=-15是操作员SIGTERM。
- 待评测控制器已停止，心跳`hybrid-rope-2`已PAUSED，其他旧心跳未恢复。
  停止后实查GPU为0%/0MiB，模型、缓存、teacher资产保留，没有清理模型权重。
- 上一批报告/代码推送后，于2026-09-08 00:34 UTC执行服务商关机流程，SSH随即
  断开，复查连接超时；云控制台电源/计费状态未另行读取。本轮未重启、未再SSH。
- 本夜34个监督器作业累计24924.960秒（6.92小时），含失败/中断，非云账单。
  [整夜报告](../docs/research/ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md)与
  [作业账本](../docs/research/ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json)保留详情。
- 本地`artifacts/closeout_20260908`原账本SHA为
  `b78b4efa436549bc7bf010c79da5e8a0f275540cfc02ae3fcfd82d916e64cf9c`，紧凑回执tar SHA为
  `521af8982a8df2cd4cd9cc9776145a2c50257076454b63f0b3ca73f56886e179`。
  大Q/K/V与模型仍在服务器持久存储；Git只包含必要分析/身份，不包含这些大资产。

## 不能丢失的已有结果

- 1.5B64K每项8条：FullLagP2/Mr/同gain Mr的MK2为37.5/12.5/25，
  VT为87.5/82.5/77.5，FWE为70.83/45.83/45.83；局部三项胜，非完整RULER/SOTA。
- 1.5B128K每项8条：本方/Mr为MK2 0/0、VT85/72.5、FWE50/50。
- 3B64K每项4条：本方/Mr为MK2 75/50、VT90/95、FWE66.67/75，有迁移取舍。
- 完整表、gain、模型与decoder身份由
  [小模型结果](../docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json)管理。
  相同Native几何不构成效果直接移植证明。

## Git与论文

整夜收尾提交为`0b9b9b6179fcedf641a3590f5ba476c76b57b649`；CPU失败谱系复核、ALS
修复、原始Pro材料及可见性代码准备已由`ccf33c288289de94edfda7531235e9e374889fdf`
提交并普通推送。其后`a8b3be3`统一文档路由。本轮只改Markdown规则、失败解释、
协议及接续状态；未修改代码、结果、数组或原始Pro材料，未新增实验或重跑模型
检查。指定ignored整改报告原正文保留并加纠正提示；三个已有untracked诊断脚本
保持原状，不纳入本次提交或称作canonical验证。最新文档
提交查看`git log -1`和origin tracking。原始来源尾空格继续按字节身份保留。
`main_0726`及其中paper归档未修改、编译或生成；活动TeX/PDF未变，PDF SHA仍为
`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`。
