# Hybrid-RoPE 当前交接

- **更新：2026-09-08。作者要求尽快整理、推送，转到公司PC继续研究。**
  本任务在完成本轮提交推送后收尾；原科研目标未完成，不自动恢复GPU或旧队列。
- **先读：** [本地失败谱系与综合分析](../docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md)，
  [逐源证据/原始数据复算](../docs/research/ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json)，
  [下一项固定位置判别](../docs/research/ROPE_FIXED_POSITION_VISIBILITY_PROTOCOL_20260908.md)。
  按INDEX定位原owner即可，不需要重新生成一批综述。

## 本轮完成与限制

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
2. 先跑相关工作机测试：
   `conda run --no-capture-output -n aidemo python -m pytest scripts/analysis/rope_transport/tests/test_transport.py tests/test_position_visibility.py -q`。
   这是代码验证，不是能力实验；失败先修实现并保留本次回执。
3. 下一项只准备一个固定表、同一S集合的L/P因果判别。完整原token与来源/模板
   边界需要从EVQ既有资产读取并冻结；不能把局部record span当成完整mask proof。
   先资格核对all-keep四路一致和O与原generate一致，再判断是否开始科学干预。
   保持原position IDs、最后prompt query干预时点、原完整decoder history。
4. 不重启64维行为梯度、频率候选扫描、13任务全矩阵或旧LoRA队列。新阶段的
   资源/预算以作者当时指令为准，不继承已关闭过夜deadline。四种L/P结果分开，
   oracle恢复不当作可部署方法或SOTA，也不提前保证某种CPT配方有效。

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

上一批已推送收尾提交为`0b9b9b6179fcedf641a3590f5ba476c76b57b649`；本次CPU复核、
修复、原始Pro来源和本HANDOFF由其后一个 scoped commit 一并交付，具体HEAD查看
`git log -1`及origin tracking。原始来源尾空格为字节身份保留，新增分析按通常检查。
`main_0726`及其中paper归档未修改、编译或生成；活动TeX/PDF未变，PDF SHA仍为
`37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`。
