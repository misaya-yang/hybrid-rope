# LESSONS — Round 10-12 踩坑记录（2026-09-06 汇总）

> **2026-09-08 纠正：** 本页原先把 Y2 错称忠实 YaRN，把统计不拒绝和
> 未学成结果作了过强解释。以下已纠正；历史数字仍以各轮报告及实际回执为准。
> 这里记录发生过的问题，前瞻执行规则只由根 AGENTS.md 管理。

## 评测与对照
1. **评测模式失配曾产生零分**：该轮 Qwen2.5-Instruct 的 Native32K raw
   completion 控制复述 filler、不答题、无 EOS（0/8）。它否定该任务中的模式
   资格，不证明所有 Instruct 模型的 raw completion 都无效。后续使用模型对应
   chat template，且记录实际 token；OLMo 的模式对照不能代替 Qwen 的判断。
2. **对照纠错本身也曾出错**：原条目把 cos/sin 幅度 `a=1+0.1*ln(s)`
   误判为错误，并称 `sqrt(a)` 的 Y2 忠实。固定上游 YaRN 的确让 cos/sin 乘
   `a`、QK logits 乘 `a²`，采用频率维索引的线性 ramp；Y2 的 smoothstep+
   `sqrt(a)` 是另一算子。后续实际数组核对也未支持对原 Y/M 的整体否定。
   依据：[固定实现](../../scripts/lib/rope/official_yarn.py)、
   [实际身份核对](../research/CROSS_AUDIT_EXPERIMENT_PROTOCOL_20260907.md)。
3. **成功标准必须锚定模型自己的原生上下文**：在各模型自己的 2×/4× 上比
   （OLMo 4K→8K/16K；Qwen 32K→64K/128K）。Qwen8K/16K 可用于窗内能力和
   适配分析，但不构成超过预训练长度的证据；原文“无意义”过度概括。
4. **论文百分比不是本任务上界**：旧记录摘录的 Qwen1.5B+ 四倍长度
   81–97%、0.5B 约38%来自特定公开任务/配方，不能当成本机合同下的上界、
   保证收益，或容量/训练量的因果证明。比较须保留模型、任务、评分和训练条件。

## 训练
5. **未学成没有唯一归因**：128 步配方的 DID_NOT_LEARN 只报告该预算下
   未达到终点；负 margin 不识别“增加到500M就能解决”。原生长度、物理训练
   长度、监督内容、可训练模块和预算必须分开。Native compact 对照实际未跑，
   Z compact 的20/32与完整输入23/32也不证明等效或“LoRA只修接口”。
   [Round10纠正](reports/ROUND10_LORA_RESULTS_20260905.md)保留原判读以防重用。
6. **小卡装大模型**：7B+LoRA+16K 于 32G 卡，KL 教师必须离线预缓存
   （512 行 prediction_positions 全固定 → 可预计算，训练不载教师）；
   先探针门控（显存 + 每 micro 墙钟）再开跑。
7. **检查点与磁盘预算要一起算**：里程碑检查点个数受磁盘空闲约束
   （125M 里程碑与 11G 空闲冲突，开机后先核对再启动）。

## 工程
8. **ssh 纪律**：挂 setsid 子进程会让前台 ssh 挂起 → 用后台启动，另开连接
   验证存活；heredoc 引号会被 ssh 层吃掉 → 本地写文件 + rsync + md5 核验；
   setsid 重定向前先 mkdir 目标目录；网关 ~3.5min 断连；`pkill -f` 会匹配到
   自己所在 shell 的命令行（用字符类 `[t]` 规避）；前台长 sleep 会被拦。
9. **契约常数必须可溯源**：e1_fit_readout.py 曾硬编码"2/128、1/32"，
   不对应任何产物（与 Track A zero-shot 数字混淆）；任何门控数字
   必须指向具体 artifact，否则改为信息性参考。
10. **提交纪律（本次教训）**：存档目录提交前先与仓库现有内容逐文件比对——
    round10 package 484 文件与仓库 HEAD 逐字节相同，纯冗余；
    标注"不得提交"的机器私有配置一律不进仓库。

11. **监督器停止状态不能代替科学判读**：09-07夜间LoRA在59/128更新后按作者
    收尾要求停止，FAILED/exit=-15是SIGTERM，无最终adapter或训练后结果。
    [整夜报告](../../docs/research/ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md)
    单独记录已消耗算力、工程错误、实际负结果和未完成项。
