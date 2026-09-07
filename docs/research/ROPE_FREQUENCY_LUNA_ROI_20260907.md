# 三阶段频率研究：Luna 执行结果与 ROI 调整建议

> **2026-09-07 status / use:** 当前用途：本文件拥有早期 OLMo E0/E1 结果与当时的 ROI 建议。后续已转入 Qwen 自有方法研究；旧 Z-only 训练没有启动，也不是自动下一步。最新工作见 [Qwen owner](ROPE_SCALE_TRANSPORT_PILOT_20260907.md) 和 HANDOFF。

> **当前执行范围纠正（2026-09-07）：** 作者撤销 YaRN/MrPro/Z 三臂 full adaptation；MrRoPE/YaRN 保持冻结参考，只保留本方微调路线。Luna 已确认旧九个 E2 job 全为 NOT_STARTED、无 GPU/训练进程。以下 E1 数据不变，旧“继续三臂”的建议和启动说明均由本段替代。缩减的本方 Z 配方仍为 1528 steps、final-only；不自动更改为 LoRA 或新候选，不启动 GPU。

- **状态 / 日期：** 2026-09-07；只读结果核对及调度建议。没有发送执行消息、修改 heartbeat、启动/停止 job 或更改远端文件。
- **问题：** 在作者最新“三合一、寻找 SOTA”目标下，已有 E0/E1 回答了什么，下一笔计算应优先区分什么？
- **快照：** 远端 job receipts 于 07:51 UTC 回查，原始输出哈希与分组汇总于 07:53 UTC 核对；随后读取“实验管理-luna”任务至约 08:04 UTC 的总结和作者输入。后者是任务记录，不是新一次服务器探测。这是定时截面，不是实时监控页面。
- **协议：** OLMo 1.485B、固定 s4 表、各方法声明的 gain、同一模板和样本；本文的能力数字仅属 development。scratch 使用两个训练种子和 32 个历史 anchors。
- **支持 / 不支持：** 支持执行状态、已存输出的分项计数、相同样本身份及成本范围；不支持 SOTA、泛化、显著性或任何方法类的关闭。
- **关联：** [统一研究计划](ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)、[Pro 提示词](ROPE_FREQUENCY_PRO_PROMPT_20260907.md)、[原实验协议](../../paper-2027/research/CROSS_AUDIT_EXPERIMENT_PROTOCOL_20260907.md)。本备忘录更新研究建议，具体执行状态和许可仍在 [HANDOFF](../../paper-2027/HANDOFF.md)。

## 1. 已经完成的工作，不再重复购买

| 项目 | 快照状态 | 可复用价值 |
| --- | --- | --- |
| E0 八个任务 | 8/8 COMPLETE、exit 0 | Native 控制、教师缓存、三候选控制与 full cost probes |
| E1 scratch | COMPLETE，2560 CSV 行 | Geo/Cosh 历史权重下的强变换比较 |
| E1 五方法 long + Native | 10/10 COMPLETE、exit 0 | 同样本的 frozen 表差异和 Native 分项 |
| E2 / E3 | 无已启动 state / 结果 manifest，旧计划仍有预算/对手占位 | 尚未购买的正式适配矩阵，不能当作自动续跑 |

本次实际使用的是已获作者授权的 32GB 级 RTX 4080 SUPER，BF16 / Flash SDPA。E0 job elapsed 求和约 9.35 分钟；E1 约 46.26 分钟。它们是监督器记录的任务时间，不是 CUDA kernel 时间，也不是以后新任务的保证。

Luna 的 07:52 UTC 最终记录确认 E1 报告 `E1_report_1788767307` 已生成，E0/E1 heartbeat 已由该任务删除。作者随后于 07:58 UTC 明确要求“启动”；Luna 提出 E2 共 1528 steps、每臂 5 小时上限，但未启动，先遇到旧 checkpoint guard，之后报告 SSH connection refused。本轮没有独立复核该后续连接错误。**本备忘录不撤销已有启动要求；下面是提交作者的研究重排建议，不是给 Luna 的停止命令。**

三个四步 full probe 的内部 wall_seconds 为 YaRN 79.34、MrPro 79.24、Z 80.87 秒，包含已声明的训练和保存过程。每个只做四步，不能用于“已完成正式适配”或长训练效果结论。E0 的 teacher 和成本回执保留，不因计划重排而清除。

## 2. 当前 frozen 终点不能排出 SOTA

下表为 single-evidence 的完整组成功：每组两个世界都完整答对且正常 EOS。五臂样本身份一致，原始 examples 文件哈希与 manifest 一致；组计数已从原始行重算。

| Arm | 2K compact，/32 | 16K near，/32 | 16K far，/32 |
| --- | ---: | ---: | ---: |
| Native | 27 | 0 | 0 |
| YaRN | 27 | 0 | 0 |
| MrUni | 26 | 3 | 1 |
| MrPro | 27 | 0 | 0 |
| Z | 26 | 6 | 1 |

double-evidence 与 binding 的 16K near/far 全部是 0/16。Native compact 分别只有 2/16 与 8/16；deleted 组成功均为 0。E0 的“resolved=true”表示通过最低程序控制门，不代表每个任务具有足够的效应区分能力。

single-evidence far 的实际输入为 16301–16313 tokens。完整答案+EOS 的行计数 Native/YaRN/MrUni/MrPro/Z 为 0/0/3/0/4（各 64 行），EOS 计数为 0/23/21/26/38。正确答案行计数与正确答案+EOS 行计数在这个格子相同，因此不能把全部零分归因于“只差 EOS”。但生成预算、模板适配和输出样式仍需通过已存文本诊断，不能从聚合数认定具体机制。

**判断：** Z 对单证据 near 有开发信号，但 far 只有一组成功，不足以宣称打穿频率问题。MrPro 在此 16K 协议为零也不能否定其原论文的其他模型和评测。应保留这个困难格子，同时建立更有区分力的开发条件，不通过放松 scorer 把失败改为成功。

## 3. Native 分项也存在地板效应

每个生成类别 64 行；text 另有 64 行，只计算文本 NLL。

| Arm | instruction 完整答案+EOS | reasoning 完整答案+EOS | position-format 完整答案+EOS | text token-weighted NLL |
| --- | ---: | ---: | ---: | ---: |
| Native | 20 | 2 | 13 | 2.7108 |
| YaRN | 13 | 0 | 13 | 2.7958 |
| MrUni | 15 | 1 | 13 | 2.7890 |
| MrPro | 13 | 1 | 13 | 2.8021 |
| Z | 17 | 0 | 13 | 2.7708 |

text NLL 使用 sum(nll_sum)/sum(prediction_tokens)，每臂 35470 个位置，不与答案 NLL 混合。样本 ID / row hash 跨臂一致，原始 rows 文件哈希已核验。

text 代价较小不能覆盖 instruction 的变化。reasoning 原模型仅 2/64 成功，不能把该类当成充分的通用 reasoning 保持测试；应检查已有 outputs，并用原模型能完成、在看新方法结果前确定的任务建立下一份能力面板。不能按候选成功的题目反向挑测试集。

## 4. 已有 scratch 结果是部署边界，不是新训练方法

下表是 2K、32 anchors 配对均值，Cosh−Geo tail128 NLL；负值有利 Cosh。

| 推理策略 | seed137 | seed256 |
| --- | ---: | ---: |
| identity | −0.1809 | −0.1431 |
| fixed-s4 YaRN-derived | −0.0465 | −0.0554 |
| fixed-s4 shared-reference MrPro | −0.2175 | −0.1810 |
| target-s8 YaRN-derived | +0.0736 | +0.1123 |
| target-s8 shared-reference MrPro | +0.2217 | +0.2252 |

同一个训练配置在不同目标策略下排序不同；本轮 1K 的 fixed-s4 MrPro 也反转为 +0.1219/+0.1118。不要只取 fixed-s4 的 2K 格子宣称通用协同。这里的非几何扩展是本项目明确命名的推广，不是 MrRoPE 官方实现原样支持任意表的证据。

结果已存在，不补 seed42、不重跑这些 overlays，也不为旧 Cosh 再买 sweep。它们用于新统一规则的机制对照；从零训练的新频率表仍须真正配对训练，不能由旧权重替代。

## 5. 建议调整的优先级

三阶段均保留，顺序按可获得的信息选择，不做所有表 × 所有阶段 × 所有模型。

| 顺序 | 建议动作 | 为什么现在值得做 | 新 GPU 成本 |
| --- | --- | --- | --- |
| P0 | 完成 E1 报告与输出失败分类，整理逐长度/Native 分项 | 已有数据可直接避免错误选方向；本备忘录已完成身份及计数核对 | 0，读取和统计即可 |
| P1 | Pro 给一个明确共同构造，CPU 检查数组、端点、s=1 退化与槽位约定 | 新方法尚未定义；先避免再训练旧候选来替代方法研究 | 无 LM 计算；未知统计成本单列 |
| P2 | 短开发区间的机制消融与完整 frozen 主对比 | 区分尾段/过渡机制；恢复 4K/8K 等能力可分辨条件，保留 16K 难格子 | 先冻结少量格子，按现有逐行耗时估算并重新确认预算 |
| P3 | 同一构造的新 scratch 配对、本方轻量适配及前/后评测 | 三合一的两个独立证据面；不默认微调 MrRoPE/YaRN | 按 trainer / 数据 / 存储实测冻结，不沿用 100h 总数 |
| P4 | 主对比重复、独立文档与外部模型确认 | 防止开发集和单 checkpoint 偶然优势；优先于增加无关臂 | 为最有价值的三阶段结论保留预算 |

P2 是便宜的冻结兼容性检验，不是 P3 scratch 的统一科学门槛。允许阶段条件不同；如果冻结分支不行但新 scratch 实例有明确预测，仍可做小型配对。反过来，只有 scratch 成功不能宣称零训练也有优势。

**已修正 E2/E3：** 旧三臂矩阵已撤销；当前缩减为本方 Z 适配及最终 long/Native 评测，复用冻结参考。旧 ready-plan 原件先保留，再将原入口标成 blocked，使旧 SHA 不能继续启动。没有旧训练在运行，也没有删除实验资产。轻量适配、新方法和重复实验另列，不自动接续。

## 6. 应怎样让下一次结果可用于决策

- 先读失败文本和 stop reason；若模板/生成预算不适合任务，定义一个修正后的新协议，并对全部对比臂重新配对。旧结果原样保留；不把新旧分数直接拼接。
- 分开看频率和 gain。保留完整默认方法比较；主要因果对比补共同 gain，避免“尾部更好”其实只是温度变化。
- 新增 4K/8K 的任务必须明确改变的是物理长度、证据距离还是干扰数量。不得截断远端证据，或悄悄把难任务替换掉后仍沿用原协议名字。
- 开发阶段可以从现有任务诊断方法，但独立确认使用新文档/实例。训练见过某个 generator family，测试该 family 新实例只能写家族内迁移。
- 当前准备版有 8K/16K CPT；若沿用，16K 属适配已见长度。未见长度声明需另有更长测试。保存中间 checkpoint 不等于已测生成曲线；成本主张需要其实际生成评测。
- Native 采用各阶段适当基准：成熟模型对原始 checkpoint；scratch 对配对 geometric 模型的 ID 能力。阈值、误差范围及最终主终点在观察新候选结果前确定。

评测 batching 可作为一次有界工程优化，在冻结新批次前核对 padding、position IDs、缓存、EOS 与输出差异。不能只为填显存而让多个模型进程争用一张卡，也不为优化引入比评测本身更长的工作。当前已完成 E1 无须为 batching 重跑。等待 Pro 本身不需要占用 GPU；资源退出遵循已有授权，本文不执行关机。

### 保存策略按需要的证据决定

Luna 后续指出旧实现按三个中间/最终权重快照及恢复状态预留空间。这个 guard 是实现选择，不是科学硬需求；已有 base 无需复制成新的输入备份。也不能据旧 guard 把三臂约 90GiB 宣称为不可避免的实验需求。

若只比较固定最终 checkpoint，保留 final 权重、部署/训练 manifest、原始评价和训练日志即可；无需为了不存在的续训需求保留 Adam/RNG resume。若要生成能力的预算曲线，可以在预定里程碑当场评测并保存原始输出，再按明确保存策略处理临时权重。只留 steps.jsonl 无法事后恢复中间模型，也不能代替里程碑生成结果。

此建议只涉及将来的输出策略。更改时同步成本估算和 plan identity，不删除已有证据或更改比较臂预算；本轮没有修改 trainer、保存策略或任何 checkpoint。

## 7. 回执定位与验证限度

实际执行 plan SHA-256：`df062bc5c13ff96abb5c6699c13615f1b7e533bbe40904abcb13c35851ebe6fb`。与原 5090 准备计划分开，不能混用状态。

本轮私有只读回执按文件名及 SHA 定位，原始地址由执行任务和本机维护目录保存，不写入可分享提示词：

| 回执 | SHA-256 |
| --- | --- |
| execution_snapshot.json | `f84155cf1b4efb76240d6aac916b9bce9f1bc37ddd88ea05e5043b2d21da5960` |
| result_readback.json | `44b38cce6fe264cf974d4c2f4fc7822388f1c2cc9c5031647282d896a638a9a3` |
| existing_checkpoint_overlay.csv | `a8153d60da0a1da993880e81d8d0809cbd5df9629be282aa4874444c90143a0b` |
| scratch_paired_readback.json | `dc2ce056b827945fed32827860ef67aaad54a9522cf6579c27c397dc3dce572d` |

主要 long manifest SHA：

| Arm | manifest SHA-256 |
| --- | --- |
| Native | `6093e70301be400d22aad374701e88ff26c1ce5984f3c7bd97ee8cd2202c76e9` |
| YaRN | `5a4e84122d0a80722126d422396dbe2dc5b69c1670a87393af2ac2cffc24bc21` |
| MrUni | `240593f3a0778806368ab00ac3374ebf5efe0c26cf98b8789462ad7002b39db6` |
| MrPro | `ba515c9a7bb3d72f58f02abce85ad32ca2cfcb396b88854f8752faf465ef9343` |
| Z | `27536be6414f699158aa3df62cfea9318307812cf22b79179faf48269c14d6b2` |

已核对十份 E1 long/native 原始行文件的内容哈希、跨臂样本身份、long 组汇总与 scratch CSV hash / 配对 target hash。没有重新推理、全面复审任务语义、验证独立确认或进行显著性检验；也未把程序完成当成科学通过。
