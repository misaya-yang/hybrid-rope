# 固定BM的Qwen2.5-7B零训练迁移

筛查COMPLETE / NO_LONG_GAIN。不是新候选，也未按7B结果选择参数。

模型Qwen/Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`，
7,615,616,512参数。官方元数据与四分片SHA冻结；镜像下载逐分片大小/SHA全部匹配。
静态S4、Native32K、theta1e6、head_dim128，MrPro/BM过渡区23→40；继承同gain。

tokenizer、模板及生成配置与既有3B输入准备完全匹配，因此原36条输入直接复用。
第一次筛查选每任务源顺序首1条32K、首2条128K，共18条/臂。该选法在7B输出前
冻结；初定每长端仅1条过薄，运行前统一扩为2条，没有查看7B得分。
该模型没有匹配MrPro基线，补跑一次并归档，然后跑原BM。不是跨模型借用分数。

为适配32GiB，Qwen2逐token独立MLP按4096token分块。注意力仍完整Flash SDPA，
不缩短上下文、不分块注意力、不改KV或权重。双方使用同一路径；先核查实际
第一MLP在非整除块上的BF16误差，记录回执。CPU门控MLP测试包含双batch与末尾
不满块。数学逐token公式相同，不宣称任意BF16整网生成逐位等同未分块执行。

用六任务长端等权分数、短端、逐题完整输出与EOS判断整体价值，允许任务内胜负。
有价值则扩展其余冻结输入和新任务；若没有，不扫描频率系数来修当前样本。
本轮只是开发筛查，不是全RULER或问题已解决。

远端根`/root/autodl-tmp/bm_transfer_qwen7b_20260908`，`prepared_01`、`code_01`、
`screen_spec.json`、`screen_selection.json`、`run_screen_01`；本任务独占GPU执行。

## 执行恢复

run_screen_01完成6条短MrPro后，首128K在RMSNorm分配1.75GiB时OOM；当时
PyTorch实际分配24.38GiB、保留未用6.08GiB。保留失败状态和六条完整输出。
run_screen_02使用expandable_segments:True、同4096-token MLP实现，逐行核对
科学契约/输入/分数/EOS后复用六条，不重跑。实际128K已成功，约51–60秒/条。
实际第一MLP分块资格：最大绝对误差3.05e-5，相对L2 1.51e-4，不是bitwise。
两臂使用相同分块路径；allocator变化未更换注意力或模型数学。

## 完整18条结果

32K BM80.00%对Mr83.33%，128K BM71.11%对Mr84.44%；0胜、3负、15平。
128K三个检索任务均满分，FWE同66.67%；VT从90%降至60%，QA从50%降至0%。
短端VT从100%降至80%。这是有限筛查，不足以宣称所有任务失败，但本轮没有新增
赢过Mr的样本，不能把OLMo收益推广至这个Qwen模型。停止未具收益依据的全量扩展。
run_screen_02新增30次生成、复用6条，1380.62秒，峰值分配27.38GiB；首轮OOM
仍保留。完整回执在[结果JSON](ROPE_QWEN7_BM_RESULT_20260908.json)。
